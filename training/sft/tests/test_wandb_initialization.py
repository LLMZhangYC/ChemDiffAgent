"""Exercise the real SFT WandB setup block without loading a model or an NPU."""
import ast
from contextlib import redirect_stdout
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from omegaconf import OmegaConf

SFT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SFT_ROOT))
from train.utils import flatten_omega_conf

SOURCE = SFT_ROOT / "train/sft_sdar.py"
main = next(node for node in ast.parse(SOURCE.read_text()).body
            if isinstance(node, ast.FunctionDef) and node.name == "main")
setup = next(
    node for node in main.body
    if isinstance(node, ast.If) and any(
        isinstance(child, ast.Call)
        and isinstance(child.func, ast.Attribute)
        and isinstance(child.func.value, ast.Name)
        and child.func.value.id == "wandb"
        and child.func.attr == "init"
        for child in ast.walk(node)
    )
)
SETUP_CODE = compile(ast.fix_missing_locations(ast.Module(
    body=[setup], type_ignores=[])), str(SOURCE), "exec")


def execute_setup(config, backend, is_main_process=True):
    scope = {
        "config": config,
        "wandb": backend,
        "accelerator": SimpleNamespace(is_main_process=is_main_process),
        "pretrained_model": config.model.pretrained_model,
        "flatten_omega_conf": flatten_omega_conf,
    }
    with redirect_stdout(io.StringIO()):
        exec(SETUP_CODE, scope)


class WandbInitializationTests(unittest.TestCase):
    def make_config(self):
        return OmegaConf.create({
            "wandb": {"entity": None, "resume": False, "mode": "offline"},
            "experiment": {"project": "sft-test", "run_name": "test"},
            "model": {"pretrained_model": "/fixture/model"},
        })

    def backend(self, run_id="generated-id"):
        # Deliberately no util.generate_id: newer SDKs do not expose it.
        return SimpleNamespace(init=Mock(return_value=SimpleNamespace(id=run_id)))

    def test_missing_id_uses_sdk_and_persists_actual_id(self):
        config, backend = self.make_config(), self.backend()
        execute_setup(config, backend)
        backend.init.assert_called_once()
        kwargs = backend.init.call_args.kwargs
        self.assertNotIn("id", kwargs)
        self.assertNotIn("resume", kwargs)
        self.assertEqual(kwargs["mode"], "offline")
        self.assertEqual(config.wandb.run_id, "generated-id")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            OmegaConf.save(config, path)
            self.assertEqual(OmegaConf.load(path).wandb.run_id, "generated-id")

    def test_explicit_id_is_passed_to_sdk(self):
        config, backend = self.make_config(), self.backend("existing-id")
        config.wandb.run_id = "existing-id"
        execute_setup(config, backend)
        self.assertEqual(backend.init.call_args.kwargs["id"], "existing-id")
        self.assertNotIn("resume", backend.init.call_args.kwargs)
        self.assertEqual(config.wandb.run_id, "existing-id")

    def test_resume_with_explicit_id(self):
        config, backend = self.make_config(), self.backend("existing-id")
        config.wandb.run_id = "existing-id"
        config.wandb.resume = True
        execute_setup(config, backend)
        self.assertEqual(backend.init.call_args.kwargs["id"], "existing-id")
        self.assertEqual(backend.init.call_args.kwargs["resume"], "allow")

    def test_resume_without_id_starts_new_run(self):
        config, backend = self.make_config(), self.backend()
        config.wandb.resume = True
        execute_setup(config, backend)
        self.assertNotIn("id", backend.init.call_args.kwargs)
        self.assertNotIn("resume", backend.init.call_args.kwargs)
        self.assertEqual(config.wandb.run_id, "generated-id")

    def test_disabled_or_non_main_process_does_not_initialize(self):
        for enabled, is_main in ((False, True), (True, False)):
            with self.subTest(enabled=enabled, is_main=is_main):
                config, backend = self.make_config(), self.backend()
                config.wandb.enabled = enabled
                execute_setup(config, backend, is_main)
                backend.init.assert_not_called()
                self.assertNotIn("run_id", config.wandb)


if __name__ == "__main__":
    unittest.main(verbosity=2)
