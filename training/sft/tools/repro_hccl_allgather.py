#!/usr/bin/env python3
"""Four-rank BF16 collective stress reproducer using only torch + torch_npu.

An async host return is not device completion. Buffers and Work handles remain
alive until every Work has been waited AND the NPU has been synchronized.
Failure alone does not identify the Jetty issue: inspect the CANN first error.
"""
import argparse
from datetime import timedelta
from importlib.metadata import version
import json
import os
from pathlib import Path
import time
import traceback


def positive_int(text):
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--iterations", type=positive_int, default=60000,
                        help="total AllGather calls per rank, not per communicator")
    parser.add_argument("--workload", choices=("allgather", "mixed"), default="allgather",
                        help="mixed: AllGather batches on group 0, then one ReduceScatter on group 1 (or 0 with --groups 1)")
    parser.add_argument("--rs-elements", type=positive_int, default=77791232,
                        help="mixed mode: ReduceScatter OUTPUT elements per rank; default matches the training trace")
    parser.add_argument("--groups", type=int, choices=(1, 2), default=2,
                        help="two groups: alternate AllGather batches, or assign AllGather/ReduceScatter separately in mixed mode")
    parser.add_argument("--layout", choices=("inplace", "separate"), default="inplace")
    parser.add_argument("--input-mib", type=positive_int, default=24)
    parser.add_argument("--window", type=positive_int, default=8,
                        help="independent buffers per batch; 1 is a serialized control")
    parser.add_argument("--check-every", type=positive_int, default=1000,
                        help="sample-check outputs and report progress after this many calls")
    parser.add_argument("--timeout-seconds", type=positive_int, default=600)
    return parser.parse_args(argv)


def payload(iteration, rank):
    # Exactly representable BF16 integers; distinguish rank and successive calls.
    return (iteration % 16) * 8 + rank + 1


def reduction_payload(batch, source_rank, destination_rank):
    # Every possible partial sum is an exactly representable BF16 integer < 256.
    # Distinguish destination partitions as well as source ranks and batches.
    return (batch % 4) * 8 + source_rank + 1 + destination_rank * 2


def run_loop(torch, dist, groups, args, rank, emit):
    world = 4
    elements = args.input_mib * 1024 * 1024 // 2
    slots, pending = [], []
    completed, submitted, attempted = 0, 0, 0
    rs_completed = 0
    stage = "allocate"
    started = time.monotonic()
    try:
        for _ in range(min(args.window, args.iterations)):
            output = torch.empty(world * elements, dtype=torch.bfloat16, device=f"npu:{rank}")
            source = (output.narrow(0, rank * elements, elements) if args.layout == "inplace"
                      else torch.empty(elements, dtype=torch.bfloat16, device=f"npu:{rank}"))
            slots.append((source, output))
        emit("buffers_ready", input_elements=elements, input_bytes=elements * 2,
             output_bytes=world * elements * 2, slots=len(slots), layout=args.layout,
             allocated_tensor_bytes=len(slots) * elements * 2 * (world + (args.layout == "separate")))
        if args.workload == "mixed":
            # The training trace uses a full concatenated input, with the result
            # aliasing this rank's quarter. Keep a separate control layout too.
            rs_source = torch.empty(world * args.rs_elements, dtype=torch.bfloat16, device=f"npu:{rank}")
            rs_output = (rs_source.narrow(0, rank * args.rs_elements, args.rs_elements)
                         if args.layout == "inplace" else
                         torch.empty(args.rs_elements, dtype=torch.bfloat16, device=f"npu:{rank}"))
            emit("reduce_scatter_buffers_ready", input_elements=world * args.rs_elements,
                 output_elements=args.rs_elements, input_bytes=world * args.rs_elements * 2,
                 output_bytes=args.rs_elements * 2, layout=args.layout,
                 allocated_tensor_bytes=args.rs_elements * 2 * (world + (args.layout == "separate")))
        last_check = 0
        for start in range(0, args.iterations, len(slots)):
            pending = []
            end = min(start + len(slots), args.iterations)
            # Switch groups only after draining the previous batch. This avoids
            # introducing unrelated cross-process-group stream-ordering hazards.
            batch = start // len(slots)
            group_index = 0 if args.workload == "mixed" else batch % len(groups)
            for iteration in range(start, end):
                slot = iteration - start
                source, output = slots[slot]
                attempted = iteration + 1
                stage = "fill_input"
                source.fill_(payload(iteration, rank))
                stage = "all_gather_submit"
                emit("submit_enter", iteration=attempted, group=group_index, slot=slot,
                     operation="all_gather_into_tensor",
                     input_ptr=hex(source.data_ptr()), output_ptr=hex(output.data_ptr()),
                     input_elements=elements, dtype="bfloat16", async_op=True)
                work = dist.all_gather_into_tensor(output, source, group=groups[group_index], async_op=True)
                pending.append((iteration, slot, work))
                submitted = attempted
            emit("drain_begin", first=start + 1, last=end)
            stage = "work_wait"
            for _, _, work in pending:
                work.wait()
            stage = "device_synchronize"
            torch.npu.synchronize()
            completed = end
            emit("device_complete", completed=completed)
            check_batch = end - last_check >= args.check_every or end == args.iterations
            if check_batch:
                stage = "sample_validation"
                for iteration, slot, _ in pending:
                    # Small host copy; no extra collective. This is sampled, not full-tensor validation.
                    samples = slots[slot][1].view(world, elements)[:, [0, elements // 2, elements - 1]].cpu().tolist()
                    expected = [[payload(iteration, r)] * 3 for r in range(world)]
                    if samples != expected:
                        raise RuntimeError(f"Data mismatch at iteration {iteration + 1}: {samples} != {expected}")
                emit("sample_check_passed", completed=completed, samples_per_rank_partition=3)
            # Draining and synchronizing before switching groups is deliberate:
            # this is a safe combination test, not an exact replay of ZeRO scheduling.
            pending.clear()
            if args.workload == "mixed":
                stage = "reduce_scatter_fill"
                for destination in range(world):
                    rs_source.narrow(0, destination * args.rs_elements, args.rs_elements).fill_(
                        reduction_payload(batch, rank, destination))
                stage = "reduce_scatter_submit"
                rs_group = len(groups) - 1
                emit("reduce_scatter_enter", batch=batch + 1, after_allgathers=completed,
                     group=rs_group, operation="reduce_scatter_tensor", async_op=False,
                     input_elements=world * args.rs_elements, output_elements=args.rs_elements,
                     input_ptr=hex(rs_source.data_ptr()), output_ptr=hex(rs_output.data_ptr()),
                     dtype="bfloat16", reduction="SUM")
                # async_op=False matches the observed training call. Synchronize
                # explicitly before reading/reusing buffers or switching groups.
                dist.reduce_scatter_tensor(rs_output, rs_source, op=dist.ReduceOp.SUM,
                                           group=groups[rs_group], async_op=False)
                stage = "reduce_scatter_synchronize"
                torch.npu.synchronize()
                rs_completed += 1
                emit("reduce_scatter_device_complete", completed=rs_completed, after_allgathers=completed)
                if check_batch:
                    stage = "reduce_scatter_validation"
                    samples = rs_output.view(1, args.rs_elements)[:, [0, args.rs_elements // 2, args.rs_elements - 1]].cpu().tolist()
                    expected = [[sum(reduction_payload(batch, r, rank) for r in range(world))] * 3]
                    if samples != expected:
                        raise RuntimeError(f"ReduceScatter data mismatch at batch {batch + 1}: {samples} != {expected}")
                    emit("reduce_scatter_sample_check_passed", completed=rs_completed, samples=3)
            if check_batch:
                if rank == 0:
                    print(f"[HCCL repro] completed={completed}/{args.iterations} "
                          f"reduce_scatter_completed={rs_completed} "
                          f"elapsed={time.monotonic() - started:.1f}s", flush=True)
                last_check = end
            # Only now may slots be reused; no input/output for an in-flight Work is overwritten.
        emit("iterations_completed", completed=completed, reduce_scatter_completed=rs_completed,
             elapsed_s=time.monotonic() - started)
        return completed
    except Exception:
        emit("error", stage=stage, attempted=attempted, submitted=submitted, completed=completed,
             reduce_scatter_completed=rs_completed,
             pending_iterations=[i + 1 for i, _, _ in pending], traceback=traceback.format_exc())
        # No cleanup collective or further device call after a device failure.
        raise


def main():
    args = parse_args()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if int(os.environ["WORLD_SIZE"]) != 4 or local_rank != rank:
        raise RuntimeError("Use a single node with exactly four torchrun workers")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / f"rank_{rank}.jsonl"
    with path.open("x", encoding="utf-8", buffering=1) as log:
        def emit(event, **fields):
            log.write(json.dumps(dict(time_ns=time.time_ns(), rank=rank, pid=os.getpid(),
                                      event=event, **fields), default=str) + "\n")
        stage = "import"
        try:
            import torch
            import torch_npu  # noqa: F401; registers the NPU device and HCCL backend
            import torch.distributed as dist
            emit("configuration", arguments=vars(args), versions={p: version(p) for p in ("torch", "torch-npu")},
                 environment={k: os.environ.get(k) for k in (
                     "ASCEND_RT_VISIBLE_DEVICES", "ASCEND_HOME_PATH", "LD_LIBRARY_PATH",
                     "HCCL_OP_EXPANSION_MODE", "HCCL_ENTRY_LOG_ENABLE", "HCCL_BUFFSIZE",
                     "ASCEND_GLOBAL_LOG_LEVEL", "ASCEND_PROCESS_LOG_PATH")})
            stage = "init_process_group"
            torch.npu.set_device(local_rank)
            timeout = timedelta(seconds=args.timeout_seconds)
            dist.init_process_group(backend="hccl", timeout=timeout)
            groups = [dist.group.WORLD]
            if args.groups == 2:
                groups.append(dist.new_group(ranks=list(range(4)), backend="hccl", timeout=timeout))
            emit("groups_ready", count=len(groups))
            stage = "run_loop"
            run_loop(torch, dist, groups, args, rank, emit)
            stage = "normal_shutdown"
            dist.barrier()
            for group in reversed(groups[1:]):
                dist.destroy_process_group(group)
            dist.destroy_process_group()
            emit("passed", completed=args.iterations,
                 workload=args.workload,
                 reduce_scatter_completed=((args.iterations + args.window - 1) // args.window
                                           if args.workload == "mixed" else 0),
                 meaning="No failure in this test; does not rule out the training-specific trigger")
        except Exception:
            if stage != "run_loop":
                emit("error", stage=stage, traceback=traceback.format_exc())
            print(f"[HCCL repro] rank={rank} FAILED; Python log: {path}; inspect CANN first error", flush=True)
            raise


if __name__ == "__main__":
    main()
