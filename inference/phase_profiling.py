"""Isolate selected forward calls so unrelated phases cannot enter the trace."""
from contextlib import contextmanager
from pathlib import Path

from training.sft.train.profiling import create_profiler


class PhaseProfiler:
    def __init__(self, settings, output_dir, rank, device_type, world_size, phase):
        self.settings = dict(settings)
        self.output_dir = output_dir
        self.rank = rank
        self.device_type = device_type
        self.world_size = world_size
        self.phases = {'prefill', 'denoise'} if phase == 'both' else {phase}
        self.counts = {name: 0 for name in self.phases}
        for key, minimum in [('skip_first', 0), ('warmup', 0), ('active', 1)]:
            value = self.settings[key]
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f'profiling.{key} must be an integer >= {minimum}')
        if self.settings['enabled'] and device_type != 'npu':
            raise ValueError('NPU profiling requires an NPU device')

    @contextmanager
    def forward(self, phase):
        if not self.settings['enabled'] or phase not in self.phases:
            yield
            return
        index = self.counts[phase]
        self.counts[phase] += 1
        start = self.settings['skip_first'] + self.settings['warmup']
        if not start <= index < start + self.settings['active']:
            yield
            return
        import torch
        # Finish earlier work before starting a trace; stop before any other phase.
        torch.npu.synchronize()
        settings = {**self.settings, 'skip_first': 0, 'warmup': 0, 'active': 1,
                    'output_dir': str(Path(self.settings['output_dir']) / phase / f'forward_{index}')}
        with create_profiler(settings, self.output_dir, self.rank, self.device_type,
                             world_size=self.world_size, schedule_unit=f'{phase}_forward') as profiler:
            try:
                with torch.profiler.record_function(f'inference::{phase}'):
                    yield
            finally:
                torch.npu.synchronize()
            profiler.step()
