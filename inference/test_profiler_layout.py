"""Shared profiler keeps epoch layout for training only."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import MagicMock, Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.sft.train.profiling import create_profiler


class ProfilerLayoutTests(unittest.TestCase):
    def test_inference_and_training_layouts(self):
        for epoch in (None, 0, 2):
            with self.subTest(epoch=epoch), tempfile.TemporaryDirectory() as root:
                api = NS(profile=Mock(return_value=MagicMock()), schedule=Mock(),
                         tensorboard_trace_handler=Mock(), _ExperimentalConfig=Mock(),
                         ProfilerLevel=NS(Level1=1), AiCMetrics=NS(PipeUtilization=2),
                         ProfilerActivity=NS(CPU='cpu', NPU='npu'))
                with patch.dict(sys.modules, {'torch_npu': NS(profiler=api)}):
                    kwargs = {} if epoch is None else {'epoch': epoch}
                    with create_profiler({'enabled': True}, root, 0, 'npu',
                                         schedule_unit='model_forward', **kwargs):
                        pass
                directory = Path(root) / 'prof/rank_0'
                if epoch is not None:
                    directory /= f'epoch_{epoch}'
                metadata = json.loads((directory / 'metadata.json').read_text())
                self.assertEqual('epoch' in metadata, epoch is not None)
                self.assertEqual(metadata['schedule_unit'], 'model_forward')
                api.tensorboard_trace_handler.assert_called_once_with(str(directory))
                if epoch is None:
                    self.assertEqual(list(Path(root).rglob('epoch_*')), [])


if __name__ == '__main__':
    unittest.main()
