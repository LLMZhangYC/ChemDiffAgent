from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

from step_timing import StepTimer
from rollout import run_one


class StepTimingTests(unittest.TestCase):
    def test_accelerator_completion_and_reset(self):
        timer = StepTimer(NS(type='npu'))
        sync = Mock()
        with patch('step_timing.torch.npu', NS(synchronize=sync), create=True), \
             patch('step_timing.time.perf_counter', side_effect=[10.0, 10.25]):
            with timer.measure('denoise', 2, 3, False):
                self.assertEqual(sync.call_count, 1)
            self.assertEqual(sync.call_count, 2)
        self.assertEqual(timer.records, [dict(phase='denoise', block_index=2,
                                            step_index=3, fallback=False, seconds=0.25)])
        timer.reset()
        self.assertEqual(timer.records, [])

    def test_failed_forward_is_not_recorded(self):
        timer = StepTimer(NS(type='cpu'))
        with self.assertRaises(RuntimeError):
            with timer.measure('prefill', None, None, False):
                raise RuntimeError('failed')
        self.assertEqual(timer.records, [])

    def test_rollout_saves_records_even_when_output_format_is_invalid(self):
        class Generator:
            tokenizer = NS(encode=lambda *args, **kwargs: [1])
            step_timer = StepTimer(NS(type='cpu'))
            def generate(self, *args, **kwargs):
                self.step_timer.reset()
                with self.step_timer.measure('denoise', 0, 0, False):
                    pass
                return 'plain text without agent tags'
        with patch('rollout.unified_system_prompt', return_value='system'), \
             patch('rollout.ToolExecutor'):
            row = run_one({'question': 'test', 'sample_id': 'test'}, Generator(), [],
                          max_turns=2, max_new_tokens=64, max_context_tokens=1024,
                          tool_timeout_seconds=1)
        self.assertEqual(row['termination_reason'], 'invalid_generation')
        record = row['timing']['forward_steps'][0]
        self.assertEqual(record['turn'], 1)
        self.assertEqual(record['step_index'], 0)
        self.assertGreaterEqual(record['seconds'], 0)


if __name__ == '__main__':
    unittest.main()
