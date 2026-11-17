"""Control-flow tests only; no real device or communication is exercised."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train.collective_trace import CollectiveTrace, install_collective_trace
from train.diagnostics import PhaseTrace
from tools.compare_collectives import compare


class Tensor:
    shape = (12,)
    dtype = "torch.bfloat16"
    device = "npu:0"
    def numel(self): return 12
    def element_size(self): return 2
    def data_ptr(self): return 4096
    def stride(self): return (1,)
    def storage_offset(self): return 0


class Backend:
    def __init__(self):
        self.calls = []
        self.work = object()
        self.failure = None
    def all_gather_into_tensor(self, output_tensor, input_tensor, group=None, async_op=False):
        self.calls.append((output_tensor, input_tensor, group, async_op))
        if self.failure:
            raise self.failure
        return self.work


class CollectiveTraceTests(unittest.TestCase):
    def test_installer_uses_backend_and_zero3_coordinator_without_extra_collectives(self):
        self.assertIsNone(install_collective_trace(None, None, {}))
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            phase = PhaseTrace({"enabled": True}, directory, 0, lambda: self.fail("unexpected sync"))
            phase.context["micro_step"] = 363
            modules = {name: ModuleType(name) for name in
                       ("deepspeed", "deepspeed.comm", "deepspeed.comm.comm", "torch", "torch.distributed")}
            modules["deepspeed"].comm = modules["deepspeed.comm"]
            modules["deepspeed.comm"].comm = modules["deepspeed.comm.comm"]
            modules["torch"].distributed = dist = modules["torch.distributed"]
            class Group:
                group_name = "0"
            dist.group = NS(WORLD=Group())
            dist.get_process_group_ranks = lambda group: [0, 1, 2, 3]
            modules["deepspeed.comm.comm"].cdb = backend = Backend()
            coordinator = NS(fetch_sub_module=lambda module, forward: None)
            engine = NS(module=NS(named_modules=lambda: []),
                        optimizer=NS(parameter_offload=NS(param_coordinator=coordinator)))
            with patch.dict(sys.modules, modules):
                tracer = install_collective_trace(engine, phase, {"trace_collectives": True})
                self.assertEqual(backend.calls, [])
                self.assertIs(backend.all_gather_into_tensor(Tensor(), Tensor(), async_op=True), backend.work)
                tracer.close()
            records = [json.loads(line) for line in tracer.path.read_text().splitlines()]
            self.assertEqual(records[1]["group"], {"name": "0", "ranks": [0, 1, 2, 3]})

    def test_window_counts_all_calls_and_preserves_exact_handle_and_arguments(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            phase = PhaseTrace({"enabled": True}, directory, 0, lambda: self.fail("unexpected sync"))
            tracer = CollectiveTrace(phase, start=320)
            backend = Backend()
            tracer.install_backend(backend, lambda g: {"name": "0", "ranks": [0, 1, 2, 3]})
            tensor, group = Tensor(), object()
            phase.context["micro_step"] = 319
            self.assertIs(backend.all_gather_into_tensor(tensor, tensor, group, True), backend.work)
            phase.context["micro_step"] = 320
            self.assertIs(backend.all_gather_into_tensor(tensor, tensor, group=group, async_op=True), backend.work)
            self.assertEqual(backend.calls, [(tensor, tensor, group, True)] * 2)
            records = [json.loads(line) for line in tracer.path.read_text().splitlines()]
            self.assertEqual([r["event"] for r in records], ["installed", "enter", "host_return"])
            self.assertEqual(records[1]["sequence"], 2)
            self.assertEqual(records[1]["tensors"]["input_tensor"]["bytes"], 24)
            tracer.close()
            self.assertNotIn("all_gather_into_tensor", vars(backend))
            self.assertIs(backend.all_gather_into_tensor(tensor, tensor), backend.work)

    def test_failure_is_reraised_and_fetch_context_is_reset(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            phase = PhaseTrace({"enabled": True}, directory, 1, lambda: self.fail("unexpected sync"))
            phase.context["micro_step"] = 363
            tracer, backend = CollectiveTrace(phase), Backend()
            backend.failure = RuntimeError("injected HCCL failure")
            tracer.install_backend(backend, lambda g: {"name": "0", "ranks": [0, 1, 2, 3]})
            child, tensor = object(), Tensor()
            original = lambda module, forward: backend.all_gather_into_tensor(tensor, tensor, async_op=True)
            coordinator = NS(fetch_sub_module=original)
            tracer.install_fetch_context(coordinator, NS(named_modules=lambda: [("model.layers.2.mlp.gate_proj", child)]))
            with self.assertRaises(RuntimeError) as error:
                coordinator.fetch_sub_module(child, forward=True)
            self.assertIs(error.exception, backend.failure)
            self.assertIsNone(tracer.context.get())
            records = [json.loads(line) for line in tracer.path.read_text().splitlines()]
            self.assertEqual([r["event"] for r in records], ["installed", "enter", "error", "fetch_error"])
            self.assertEqual(records[1]["fetch_context"]["module"], "model.layers.2.mlp.gate_proj")
            tracer.close()
            self.assertIs(coordinator.fetch_sub_module, original)

    def test_comparison_ignores_rank_local_addresses_but_detects_count_mismatch(self):
        rows = []
        for rank in range(4):
            rows.append(dict(event="enter", rank=rank, group={"name": "0", "ranks": list(range(4))},
                             sequence=100, op="all_gather_into_tensor", options={},
                             tensors={"input_tensor": dict(numel=12, dtype="torch.bfloat16",
                                      ptr=hex(4096 + rank * 24), storage_offset=rank * 12, device=f"npu:{rank}")}))
        result = compare(rows)[0]
        self.assertEqual(result["common_calls"], 1)
        self.assertIsNone(result["first_signature_mismatch"])
        rows[2]["tensors"]["input_tensor"]["numel"] = 24
        self.assertEqual(compare(rows)[0]["first_signature_mismatch"]["sequence"], 100)
        result = compare(rows[:3])[0]
        self.assertEqual(result["common_calls"], 0)
        self.assertEqual(result["missing_sequence_examples"][0]["missing_ranks"], [3])
        with self.assertRaisesRegex(ValueError, "Duplicate sequence"):
            compare(rows + rows[:1])


if __name__ == "__main__":
    unittest.main()
