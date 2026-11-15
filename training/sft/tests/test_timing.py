"""Microstep wall time includes data fetch but excludes work between steps."""
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train.timing import MicrostepTimer


class TimingTests(unittest.TestCase):
    def test_fetch_and_training_included_reporting_and_validation_excluded(self):
        clock = [0.0]
        def batches():
            for _ in range(2):
                clock[0] += 2  # Data fetch.
                yield "batch"
        timer = MicrostepTimer(NS(device=torch.device("cpu"), num_processes=1))
        with patch("train.timing.time.perf_counter", side_effect=lambda: clock[0]):
            iterator = timer.batches(batches())
            for _ in range(2):
                self.assertEqual(next(iterator), "batch")
                clock[0] += 3  # Training.
                self.assertEqual(timer.finish(), {"microstep_time_s": 5})
                clock[0] += 100  # Logging, profiler export, validation or checkpoint.
            with self.assertRaises(StopIteration):
                next(iterator)

    def test_device_completion_and_slowest_rank(self):
        synchronize = Mock()
        accelerator = NS(device=torch.device("cpu"), num_processes=2,
                         gather=Mock(return_value=torch.tensor([3., 7.])))
        timer = MicrostepTimer(accelerator)
        with patch.object(timer, "synchronize", synchronize), patch(
            "train.timing.time.perf_counter", side_effect=[10., 13.]
        ):
            next(timer.batches(["batch"]))
            self.assertEqual(timer.finish(), {"microstep_time_s": 7})
        self.assertEqual(synchronize.call_count, 2)
        torch.testing.assert_close(accelerator.gather.call_args.args[0], torch.tensor([3.]))
        with patch("train.timing.torch.get_device_module") as module:
            timer.accelerator.device = NS(type="npu")
            timer.synchronize()
            module.return_value.synchronize.assert_called_once_with(timer.accelerator.device)

    def test_disabled_has_no_clock_sync_or_collective(self):
        timer = MicrostepTimer(NS(), enabled=False)
        with patch("train.timing.time.perf_counter") as clock:
            self.assertEqual(list(timer.batches([1, 2])), [1, 2])
            self.assertEqual(timer.finish(), {})
            clock.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
