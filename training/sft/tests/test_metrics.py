"""Local metric persistence tests; no W&B or accelerator dependencies."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train.metrics import MetricsLogger


class MetricsTests(unittest.TestCase):
    def test_train_validation_share_steps_and_are_immediately_readable(self):
        with tempfile.TemporaryDirectory() as directory:
            logger = MetricsLogger(directory)
            for step in range(1, 4):
                logger.log("train", 0, step, step, loss=1/step, learning_rate=1e-5)
            logger.log("validation", 0, 3, 3, loss=.8, masked_accuracy=.5)
            records = [json.loads(s) for s in logger.path.read_text().splitlines()]
            self.assertEqual([r["phase"] for r in records], ["train"]*3+["validation"])
            self.assertEqual([r["micro_step"] for r in records], [1, 2, 3, 3])
            self.assertEqual({r["run_id"] for r in records}, {logger.run_id})
            other = MetricsLogger(directory)
            other.log("train", 0, 1, 1, loss=2)
            records = [json.loads(s) for s in logger.path.read_text().splitlines()]
            self.assertEqual(len(records), 5)  # Existing records are never truncated.
            self.assertNotEqual(records[-1]["run_id"], records[0]["run_id"])

    def test_non_main_rank_does_not_write_and_nonfinite_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            logger = MetricsLogger(Path(directory)/"rank1", enabled=False)
            logger.log("train", 0, 1, 1, loss=1)
            self.assertFalse(logger.path.exists())
            logger = MetricsLogger(Path(directory)/"rank0")
            with self.assertRaises(ValueError):
                logger.log("train", 0, 1, 1, loss=float("nan"))
            self.assertFalse(logger.path.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
