"""Completed-forward wall time, including accelerator completion."""
from contextlib import contextmanager
import time

import torch


class StepTimer:
    def __init__(self, device):
        self.device = device
        self.records = []

    def reset(self):
        self.records = []

    def _synchronize(self):
        if self.device.type in {'npu', 'cuda'}:
            getattr(torch, self.device.type).synchronize(self.device)

    @contextmanager
    def measure(self, phase, block_index, step_index, fallback):
        self._synchronize()
        started = time.perf_counter()
        yield
        self._synchronize()
        self.records.append({
            'phase': phase, 'block_index': block_index, 'step_index': step_index,
            'fallback': fallback, 'seconds': time.perf_counter() - started,
        })
