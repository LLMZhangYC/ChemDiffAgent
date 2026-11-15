"""Training and inference resolve timestamped runs against the invocation cwd."""
from datetime import datetime
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.run_paths import run_directory


class RunPathsTests(unittest.TestCase):
    def test_training_and_inference_share_default_layout(self):
        source = Path(__file__).resolve().parents[3] / 'inference/launch_npu.py'
        spec = importlib.util.spec_from_file_location('inference_launch_paths', source)
        inference = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(inference)
        now = datetime(2026, 11, 15, 12, 30, 45)
        with tempfile.TemporaryDirectory() as directory, patch(
            'pathlib.Path.cwd', return_value=Path(directory)
        ):
            training_path = run_directory('configs/full.yaml', now=now)
            self.assertEqual(training_path, inference.run_directory('configs/full.yaml', now))
            self.assertEqual(training_path, Path(directory) / 'res/20261115_123045_000000-full')
            self.assertFalse(training_path.exists())

    def test_explicit_override_keeps_fixed_path(self):
        self.assertEqual(run_directory('full.yaml', './my-run'), Path('./my-run').resolve())
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(run_directory('full.yaml', directory), Path(directory).resolve())


if __name__ == '__main__':
    unittest.main(verbosity=2)
