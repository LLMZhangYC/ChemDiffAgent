"""Synchronized wall-clock microstep timing, excluding reporting/evaluation."""
import time

import torch


class MicrostepTimer:
    def __init__(self, accelerator, enabled=True):
        self.accelerator = accelerator
        self.enabled = enabled

    def synchronize(self):
        device = self.accelerator.device
        if device.type != "cpu":
            torch.get_device_module(device).synchronize(device)

    def batches(self, iterable):
        iterator = iter(iterable)
        while True:
            if self.enabled:
                self.synchronize()
                self.started = time.perf_counter()
            try:
                batch = next(iterator)
            except StopIteration:
                return
            yield batch

    def finish(self):
        if not self.enabled:
            return {}
        self.synchronize()
        elapsed = time.perf_counter() - self.started
        # Report the slowest rank; timing aggregation itself is outside the window.
        if self.accelerator.num_processes > 1:
            elapsed = self.accelerator.gather(torch.tensor(
                [elapsed], device=self.accelerator.device, dtype=torch.float32
            )).max().item()
        return {"microstep_time_s": elapsed}
