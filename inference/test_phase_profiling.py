import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from phase_profiling import PhaseProfiler


class PhaseTests(unittest.TestCase):
    def test_both_uses_independent_windows_and_directories(self):
        settings = dict(enabled=True, output_dir='prof', skip_first=0, warmup=1, active=2)
        owner = PhaseProfiler(settings, '.', 0, 'npu', 1, 'both')
        from unittest.mock import MagicMock
        with patch.object(torch, 'npu', SimpleNamespace(synchronize=Mock()), create=True), \
             patch('phase_profiling.create_profiler', side_effect=lambda *a, **k: MagicMock()) as factory:
            for phase in ['prefill', 'denoise', 'denoise', 'cache_commit', 'denoise',
                          'denoise', 'prefill', 'prefill', 'prefill']:
                with owner.forward(phase):
                    pass
        self.assertEqual(factory.call_count, 4)
        paths = [Path(call.args[0]['output_dir']) for call in factory.call_args_list]
        self.assertEqual(paths, [Path('prof/denoise/forward_1'), Path('prof/denoise/forward_2'),
                                 Path('prof/prefill/forward_1'), Path('prof/prefill/forward_2')])
        self.assertEqual(owner.counts, {'prefill': 4, 'denoise': 4})

    def test_only_selected_active_calls_are_captured(self):
        settings = dict(enabled=True, output_dir='prof', skip_first=0, warmup=1, active=2)
        owner = PhaseProfiler(settings, '.', 0, 'npu', 1, 'denoise')
        events = []
        class Native:
            def __enter__(self):
                events.append('start')
                return self
            def __exit__(self, *args):
                events.append('stop')
            def step(self):
                events.append('step')
        with patch.object(torch, 'npu', SimpleNamespace(synchronize=Mock()), create=True), \
             patch('phase_profiling.create_profiler', side_effect=lambda *a, **k: Native()) as factory:
            for phase in ['prefill', 'denoise', 'cache_commit', 'denoise', 'prefill', 'denoise', 'denoise']:
                with owner.forward(phase):
                    events.append(phase)
        self.assertEqual(factory.call_count, 2)
        self.assertEqual(events, ['prefill', 'denoise', 'cache_commit',
                                 'start', 'denoise', 'step', 'stop', 'prefill',
                                 'start', 'denoise', 'step', 'stop', 'denoise'])


if __name__ == '__main__':
    unittest.main()
