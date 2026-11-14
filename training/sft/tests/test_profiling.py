"""Profiler configuration/lifecycle tests; no NPU or training packages needed."""
import ast
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train.profiling import create_profiler, DisabledProfiler, WindowProfiler


class ProfilingTests(unittest.TestCase):
    def test_disabled_needs_no_npu(self):
        with patch.dict(sys.modules, {"torch_npu": None}):
            with create_profiler({}, ".", 0, "cpu", 0) as profiler:
                profiler.step()

    def test_rejects_wrong_device_and_bad_window(self):
        with self.assertRaises(ValueError):
            create_profiler({"enabled": True}, ".", 0, "cpu", 0)
        for key, value in [("active", 0), ("warmup", -1), ("repeat", True)]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                create_profiler({"enabled": True, key: value}, ".", 0, "npu", 0)

    def test_configuration_rank_paths_and_exception_cleanup(self):
        class Context:
            exited = False
            def __enter__(self): return self
            def __exit__(self, *exc): self.exited = True
            def step(self): pass
        context = Context()
        api = NS(profile=Mock(return_value=context), schedule=Mock(return_value="schedule"),
                 tensorboard_trace_handler=Mock(return_value="handler"),
                 _ExperimentalConfig=Mock(return_value="experimental"),
                 ProfilerLevel=NS(Level1=1), AiCMetrics=NS(PipeUtilization=2),
                 ProfilerActivity=NS(CPU="cpu", NPU="npu"))
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(sys.modules, {"torch_npu": NS(profiler=api)}):
                profiler = create_profiler({"enabled": True}, directory, 2, "npu", 1)
            expected = Path(directory) / "prof/rank_2/epoch_1"
            self.assertTrue(expected.is_dir())
            self.assertEqual(json.loads((expected / "metadata.json").read_text())["schedule_unit"], "optimizer_step")
            api.tensorboard_trace_handler.assert_called_once_with(str(expected))
            api.schedule.assert_called_once_with(skip_first=0, wait=0, warmup=1, active=2, repeat=1)
            self.assertEqual(api.profile.call_args.kwargs["activities"], ["cpu", "npu"])
            with self.assertRaisesRegex(RuntimeError, "training failed"):
                with profiler:
                    raise RuntimeError("training failed")
            self.assertTrue(context.exited)

    def test_training_steps_before_early_stop_and_saves_outside_capture(self):
        # Execute the real epoch loop with stand-ins: validate early-stop ordering.
        path = Path(__file__).resolve().parents[1] / "train/sft_sdar.py"
        main = next(n for n in ast.parse(path.read_text(encoding="utf-8")).body
                    if isinstance(n, ast.FunctionDef) and n.name == "main")
        loop = next(n for n in main.body if isinstance(n, ast.For) and n.target.id == "epoch")
        for accumulation, cap in ((1, 3), (2, 3), (1, None), (2, None)):
            events = []
            class Tensor:
                def to(self, *a): return self
                def detach(self): return self
                def item(self): return 1.0
            class Progress(list):
                def set_postfix(self, **kw): pass
            class Capture:
                def __enter__(self): events.append("start"); return self
                def step(self): events.append("profile_step")
                def __exit__(self, *exc): events.append("stop")
            native_creations = []
            def profiler_factory(settings, *args, **kwargs):
                if not settings.get("enabled", False):
                    return DisabledProfiler()
                native_creations.append(1)
                return WindowProfiler(Capture(), 3)
            model = NS(global_steps=0, train=lambda: None, parameters=lambda: [])
            micro_calls = [0]
            def backward(loss):
                micro_calls[0] += 1
                if micro_calls[0] % accumulation == 0:
                    model.global_steps += 1
            batch = {k: Tensor() for k in ["extended_input_ids", "p_mask", "tok_idx_ext", "labels"]}
            env = dict(first_epoch=0, num_train_epochs=5, model=model, global_steps=0,
                       completed_optimizer_steps=0, max_optimizer_steps=cap, output_dir=".", tokenizer=None,
                       validation_enabled=False,
                       tqdm=lambda data, **kw: Progress(data), train_dataloader_lm=[batch]*10,
                       create_profiler=profiler_factory, time=NS(time=lambda: 0), end=0,
                       data_time_m=NS(update=lambda x: None), forward_process=lambda **kw: Tensor(),
                       torch=NS(isfinite=lambda x: NS(all=lambda: True)), logger=NS(info=lambda x: None),
                       optimizer=NS(step=lambda: None, zero_grad=lambda **kw: None),
                       lr_scheduler=NS(step=lambda: None),
                       save_checkpoint=lambda *a, **kw: events.append("save"),
                       config=NS(get=lambda *a: {"enabled": True}, training=NS(max_grad_norm=1),
                                 wandb=NS(get=lambda *a: False), model=NS(optimized_name="optimized")),
                       accelerator=NS(device=NS(type="npu"), process_index=0, num_processes=1, is_local_main_process=True,
                                      is_main_process=True, backward=backward,
                                      clip_grad_norm_=lambda *a: None, wait_for_everyone=lambda: None))
            exec(compile(ast.Module(body=[loop], type_ignores=[]), str(path), "exec"), env)
            saves = 1 if cap else 5
            self.assertEqual(events, ["start", "profile_step", "profile_step", "profile_step", "stop"] + ["save"] * saves)
            self.assertEqual(len(native_creations), 1)
            self.assertEqual(env["completed_optimizer_steps"], cap or 50 // accumulation)
            self.assertEqual(env["global_steps"], cap * accumulation if cap else 50)



if __name__ == "__main__":
    unittest.main(verbosity=2)
