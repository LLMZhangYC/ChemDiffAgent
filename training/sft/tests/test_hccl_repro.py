"""CPU control-flow tests; do not validate HCCL or NPU kernels."""
from contextlib import redirect_stdout
import io
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.repro_hccl_allgather import parse_args, run_loop


class Tensor:
    next_ptr = 4096
    def __init__(self, count, storage=None, offset=0):
        self.count, self.offset = count, offset
        if storage is None:
            storage = NS(ptr=Tensor.next_ptr, busy=False, samples=None)
            Tensor.next_ptr += count * 2 + 4096
        self.storage = storage
    def narrow(self, dim, offset, count): return Tensor(count, self.storage, offset)
    def data_ptr(self): return self.storage.ptr + self.offset * 2
    def fill_(self, value):
        if self.storage.busy:
            raise AssertionError("Reused buffer before NPU synchronize")
        self.value = value
    def view(self, *shape): return self
    def __getitem__(self, indices): return self
    def cpu(self):
        if self.storage.busy:
            raise AssertionError("Read output before NPU synchronize")
        return self
    def tolist(self): return self.storage.samples


class Backend:
    def __init__(self, rank=2, fail_at=None, corrupt=False):
        self.rank, self.fail_at, self.corrupt = rank, fail_at, corrupt
        self.inflight, self.calls, self.syncs, self.max_inflight = [], [], 0, 0
        self.failure = RuntimeError("injected communication failure")
        self.torch = NS(bfloat16="bfloat16", empty=lambda count, **kwargs: Tensor(count),
                        npu=NS(synchronize=self.synchronize))
    def all_gather_into_tensor(self, output, source, group, async_op):
        assert async_op is True
        assert not source.storage.busy and not output.storage.busy
        if self.calls and group != self.calls[-1][0]:
            assert not self.inflight, "Switched process groups before draining previous operations"
        source.storage.busy = output.storage.busy = True
        work = NS(waited=False, output=output, source=source)
        index = len(self.calls) + 1
        def wait():
            if index == self.fail_at:
                raise self.failure
            work.waited = True  # wait alone intentionally does not complete the fake device operation.
        work.wait = wait
        self.inflight.append(work)
        self.calls.append((group, source.data_ptr(), output.data_ptr()))
        self.max_inflight = max(self.max_inflight, len(self.inflight))
        return work
    def synchronize(self):
        self.syncs += 1
        for work in self.inflight:
            assert work.waited
            base = work.source.value - self.rank
            work.output.storage.samples = [[base + rank] * 3 for rank in range(4)]
            if self.corrupt:
                work.output.storage.samples[0][0] += 1
            work.source.storage.busy = work.output.storage.busy = False
        self.inflight.clear()


class HcclReproTests(unittest.TestCase):
    def test_inplace_layout_alternation_partial_batch_and_buffer_lifetime(self):
        args = parse_args(["--output-dir", "unused", "--iterations", "10", "--window", "3", "--check-every", "4"])
        backend, records = Backend(), []
        with redirect_stdout(io.StringIO()):
            completed = run_loop(backend.torch, backend, ["group0", "group1"], args, 2,
                                 lambda event, **kw: records.append(dict(event=event, **kw)))
        self.assertEqual(completed, 10)
        self.assertEqual([call[0] for call in backend.calls], ["group0"] * 3 + ["group1"] * 3 + ["group0"] * 3 + ["group1"])
        self.assertEqual({source - output for _, source, output in backend.calls}, {2 * 24 * 1024 * 1024})
        self.assertEqual(backend.max_inflight, 3)
        self.assertEqual(backend.syncs, 4)
        self.assertEqual(records[-1]["event"], "iterations_completed")

    def test_separate_layout_single_group_serial_control(self):
        args = parse_args(["--output-dir", "unused", "--iterations", "3", "--window", "1", "--groups", "1", "--layout", "separate"])
        backend = Backend()
        with redirect_stdout(io.StringIO()):
            run_loop(backend.torch, backend, ["group0"], args, 2, lambda *a, **kw: None)
        self.assertEqual(backend.max_inflight, 1)
        self.assertEqual(backend.syncs, 3)
        self.assertTrue(all(source > output + 96 * 1024 * 1024 for _, source, output in backend.calls))

    def test_wait_failure_preserves_error_and_last_device_completed_batch(self):
        args = parse_args(["--output-dir", "unused", "--iterations", "10", "--window", "3"])
        backend, records = Backend(fail_at=5), []
        with self.assertRaises(RuntimeError) as error:
            run_loop(backend.torch, backend, [0, 1], args, 2,
                     lambda event, **kw: records.append(dict(event=event, **kw)))
        self.assertIs(error.exception, backend.failure)
        self.assertEqual(records[-1]["stage"], "work_wait")
        self.assertEqual(records[-1]["completed"], 3)
        self.assertEqual(records[-1]["pending_iterations"], [4, 5, 6])
        self.assertEqual(backend.syncs, 1)

    def test_validation_failure_is_not_reported_as_a_jetty_diagnosis(self):
        args = parse_args(["--output-dir", "unused", "--iterations", "2"])
        backend, records = Backend(corrupt=True), []
        with self.assertRaisesRegex(RuntimeError, "Data mismatch"):
            run_loop(backend.torch, backend, [0, 1], args, 2,
                     lambda event, **kw: records.append(dict(event=event, **kw)))
        self.assertEqual(records[-1]["stage"], "sample_validation")
        self.assertEqual(records[-1]["completed"], 2)


if __name__ == "__main__":
    unittest.main()
