"""Checkpoint retention and real save-function tests without model dependencies."""
import ast
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train.checkpointing import prune_checkpoints, validate_checkpoint_limit


class RetentionTests(unittest.TestCase):
    def test_five_epochs_keep_three_and_sort_numerically(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for epoch in range(5):
                (root/f"ckpt_{epoch}").mkdir()
                prune_checkpoints(root, 3, f"ckpt_{epoch}")
            self.assertEqual(sorted(p.name for p in root.iterdir()), ["ckpt_2", "ckpt_3", "ckpt_4"])
            (root/"ckpt_10").mkdir()
            self.assertEqual(prune_checkpoints(root, 3, "ckpt_10"), ["ckpt_2"])
            self.assertTrue((root/"ckpt_10").exists())

    def test_repeated_save_protects_current_and_ignores_unrelated_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ["ckpt_0", "ckpt_1", "ckpt_2", "ckpt", "checkpoint-0", "ckpt_backup", "prof"]:
                (root/name).mkdir()
            (root/"ckpt_99").write_text("a file, not a checkpoint directory")
            self.assertEqual(prune_checkpoints(root, 3, "ckpt_0"), [])
            self.assertEqual(prune_checkpoints(root, 1, "ckpt_0"), ["ckpt_1", "ckpt_2"])
            for name in ["ckpt_0", "ckpt", "checkpoint-0", "ckpt_backup", "prof", "ckpt_99"]:
                self.assertTrue((root/name).exists())

    def test_disabled_retention_and_invalid_limits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/"ckpt_0").mkdir()
            self.assertEqual(prune_checkpoints(root, None, "ckpt_0"), [])
            for limit in [0, -1, True, "3", 1.5]:
                with self.subTest(limit=limit), self.assertRaises(ValueError):
                    validate_checkpoint_limit(limit)
            self.assertTrue((root/"ckpt_0").exists())

    def test_external_symlink_is_not_deleted(self):
        with tempfile.TemporaryDirectory() as directory:
            root, outside = Path(directory)/"outputs", Path(directory)/"outside"
            root.mkdir(); outside.mkdir()
            (outside/"keep").write_text("safe")
            try:
                (root/"ckpt_0").symlink_to(outside, target_is_directory=True)
            except OSError:
                self.skipTest("Creating directory symlinks is not permitted on this host")
            (root/"ckpt_1").mkdir()
            prune_checkpoints(root, 1, "ckpt_1")
            self.assertTrue((outside/"keep").exists())
            self.assertTrue((root/"ckpt_0").is_symlink())


class SaveIntegrationTests(unittest.TestCase):
    def test_cleanup_occurs_only_after_successful_save_and_on_main_rank(self):
        source = Path(__file__).resolve().parents[1]/"train/sft_sdar.py"
        function = next(n for n in ast.parse(source.read_text(encoding="utf-8")).body
                        if isinstance(n, ast.FunctionDef) and n.name == "save_checkpoint")
        scope = {"prune_checkpoints": prune_checkpoints, "validate_checkpoint_limit": validate_checkpoint_limit,
                 "logger": NS(info=lambda x: None, warning=lambda x: None)}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), scope)
        save = scope["save_checkpoint"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/"ckpt_0").mkdir()
            def fail(*a, **kw): raise RuntimeError("disk write failed")
            model = NS(save_pretrained=fail)
            class Settings(dict):
                __getattr__ = dict.__getitem__
            config = NS(experiment=Settings(output_dir=directory, checkpoints_total_limit=1, project="test"))
            calls = []
            accelerator = NS(is_main_process=True, unwrap_model=lambda x: x, get_state_dict=lambda x: {},
                             save=lambda *a: None, wait_for_everyone=lambda: calls.append("barrier"))
            tokenizer = NS(save_pretrained=lambda path: (Path(path)/"tokenizer.json").write_text("{}"))
            with self.assertRaisesRegex(RuntimeError, "disk write failed"):
                save(model, tokenizer, config, accelerator, "optimized", epoch=1)
            self.assertTrue((root/"ckpt_0").exists())
            def succeed(path, **kw):
                Path(path).mkdir(exist_ok=True, parents=True)
                (Path(path)/"model.safetensors").write_bytes(b"fixture")
            model.save_pretrained = succeed
            accelerator.is_main_process = False
            save(model, tokenizer, config, accelerator, "optimized", epoch=1)
            self.assertTrue((root/"ckpt_0").exists())
            accelerator.is_main_process = True
            save(model, tokenizer, config, accelerator, "optimized", epoch=1)
            self.assertFalse((root/"ckpt_0").exists())
            self.assertTrue((root/"ckpt_1/optimized/tokenizer.json").exists())
            self.assertTrue((root/"ckpt_1/metadata.json").exists())
            self.assertEqual(len(calls), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
