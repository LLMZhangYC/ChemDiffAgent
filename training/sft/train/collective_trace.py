"""Opt-in, host-only DeepSpeed collective tracing; never waits on Work handles.

The sequence below counts observed backend calls, NOT HCCL opIndex. Calls made
directly through torch.distributed are outside this trace; use CANN entry logs
alongside it. A host_return is not device completion.
"""
from collections import defaultdict
from contextvars import ContextVar
from functools import wraps
import inspect
import json
import os
import threading
import time
import traceback


def tensor_metadata(value):
    if isinstance(value, (list, tuple)):
        return [tensor_metadata(item) for item in value]
    if not hasattr(value, "numel"):
        return str(value)
    return {"shape": list(value.shape), "dtype": str(value.dtype),
            "numel": value.numel(), "bytes": value.numel() * value.element_size(),
            "ptr": hex(value.data_ptr()), "device": str(value.device),
            "stride": list(value.stride()), "storage_offset": value.storage_offset()}


class CollectiveTrace:
    METHODS = ("all_gather_into_tensor", "all_gather", "all_gather_base", "all_gather_coalesced",
               "reduce_scatter_tensor", "reduce_scatter", "all_reduce", "all_reduce_coalesced",
               "broadcast", "reduce", "barrier")
    TENSORS = ("tensor", "tensor_list", "tensors", "input", "output", "input_list",
               "input_tensor", "output_tensor", "input_tensors", "output_tensors")

    def __init__(self, phase_trace, start=1):
        if isinstance(start, bool) or not isinstance(start, int) or start < 1:
            raise ValueError("diagnostics.collectives_start_micro_step must be a positive integer")
        self.phase = phase_trace
        self.start = start
        self.path = phase_trace.path.with_name(f"collectives_rank_{phase_trace.rank}.jsonl")
        self.sequences = defaultdict(int)
        self.lock = threading.Lock()
        self.context = ContextVar("zero_fetch_context", default=None)
        self.restores = []

    def active(self):
        return self.phase.active() and self.phase.context["micro_step"] >= self.start

    def write(self, event, **fields):
        record = dict(time_ns=time.time_ns(), pid=os.getpid(), rank=self.phase.rank,
                      **self.phase.context, event=event, **fields)
        with self.lock, self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    def patch(self, obj, name, replacement):
        # Remember whether the original was an instance override or a class method.
        self.restores.append((obj, name, name in vars(obj), getattr(obj, name)))
        setattr(obj, name, replacement)

    def install_backend(self, backend, describe_group):
        installed = []
        for name in self.METHODS:
            original = getattr(backend, name, None)
            if original is None:
                continue
            signature = inspect.signature(original)

            def make_wrapper(original, name, signature):
                @wraps(original)
                def call(*args, **kwargs):
                    bound = signature.bind(*args, **kwargs)
                    bound.apply_defaults()
                    values = bound.arguments
                    group = describe_group(values.get("group"))
                    key = json.dumps(group, sort_keys=True)
                    # Count from installation, even before the recording window.
                    with self.lock:
                        self.sequences[key] += 1
                        sequence = self.sequences[key]
                    if not self.active():
                        return original(*args, **kwargs)
                    fields = dict(group=group, sequence=sequence, op=name,
                                  async_op=values.get("async_op", False),
                                  tensors={k: tensor_metadata(v) for k, v in values.items() if k in self.TENSORS},
                                  options={k: str(values[k]) for k in ("op", "src", "dst") if k in values},
                                  fetch_context=self.context.get())
                    self.write("enter", **fields)
                    try:
                        result = original(*args, **kwargs)
                    except Exception:
                        self.write("error", group=group, sequence=sequence, op=name,
                                   traceback=traceback.format_exc())
                        raise
                    self.write("host_return", group=group, sequence=sequence, op=name)
                    return result  # Preserve the exact Work object; do not wait or wrap it.
                return call

            self.patch(backend, name, make_wrapper(original, name, signature))
            installed.append(name)
        if "all_gather_into_tensor" not in installed:
            self.close()
            raise RuntimeError("Collective tracing requires DeepSpeed all_gather_into_tensor")
        self.write("installed", methods=installed, start_micro_step=self.start,
                   scope="DeepSpeed backend calls after prepare; excludes direct torch.distributed calls",
                   sequence_is_hccl_opindex=False)

    def install_fetch_context(self, coordinator, module):
        names = {id(child): name for name, child in module.named_modules()}
        original = coordinator.fetch_sub_module

        @wraps(original)
        def fetch(current_submodule, forward):
            if not self.active():
                return original(current_submodule, forward)
            context = {"module": names.get(id(current_submodule), type(current_submodule).__name__),
                       "forward": forward,
                       "note": "fetch may also prefetch parameters of later modules"}
            token = self.context.set(context)
            try:
                return original(current_submodule, forward)
            except Exception:
                self.write("fetch_error", fetch_context=context, traceback=traceback.format_exc())
                raise
            finally:
                self.context.reset(token)
        self.patch(coordinator, "fetch_sub_module", fetch)

    def close(self):
        for obj, name, was_override, original in reversed(self.restores):
            if was_override:
                setattr(obj, name, original)
            else:
                delattr(obj, name)
        self.restores.clear()


def install_collective_trace(engine, phase_trace, settings):
    if not settings.get("trace_collectives", False):
        return None
    if not phase_trace.enabled:
        raise ValueError("diagnostics.trace_collectives requires diagnostics.enabled=true")
    import deepspeed.comm.comm as comm
    import torch.distributed as dist

    cache = {}
    def describe_group(group):
        group = dist.group.WORLD if group is None else group
        if group not in cache:
            cache[group] = {"name": str(group.group_name), "ranks": dist.get_process_group_ranks(group)}
        return cache[group]

    tracer = CollectiveTrace(phase_trace, settings.get("collectives_start_micro_step", 1))
    try:
        tracer.install_backend(comm.cdb, describe_group)
        tracer.install_fetch_context(engine.optimizer.parameter_offload.param_coordinator, engine.module)
    except Exception:
        tracer.close()
        raise
    print(f"[SFT collectives] rank={phase_trace.rank} file={tracer.path}; "
          f"recording from micro_step={tracer.start}; host entries only", flush=True)
    return tracer
