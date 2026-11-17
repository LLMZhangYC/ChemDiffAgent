"""CPU-only control-flow regression tests. These do not validate NPU/HCCL."""
import ast
from contextlib import contextmanager, redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train.diagnostics import PhaseTrace
from train.profiling import DisabledProfiler


class TraceTests(unittest.TestCase):
    def test_disabled_is_noop_and_has_no_output(self):
        with tempfile.TemporaryDirectory() as directory:
            trace = PhaseTrace({}, directory, 0, lambda: self.fail("unexpected sync"))
            with trace.phase("forward"):
                pass
            self.assertEqual(list(trace.batches([1, 2], 0, 1)), [1, 2])
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_rank_window_sync_and_exception_location(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            calls = []
            trace = PhaseTrace({"enabled": True, "synchronize": True, "start_micro_step": 90},
                               directory, 3, lambda: calls.append("sync"))
            trace.context["micro_step"] = 89
            with trace.phase("ignored"):
                pass
            self.assertEqual(calls, [])
            trace.context["micro_step"] = 95
            with self.assertRaisesRegex(ValueError, "original failure"):
                with trace.phase("engine.backward"):
                    raise ValueError("original failure")
            records = [json.loads(line) for line in trace.path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(records[-1]["event"], "error")
            self.assertEqual(records[-1]["where"], "body")
            self.assertEqual(records[-1]["rank"], 3)
            self.assertIn("original failure", records[-1]["traceback"])
            self.assertEqual(calls, ["sync"])

    def test_engine_wrapper_preserves_result_and_nested_phases(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            trace = PhaseTrace({"enabled": True}, directory, 0, lambda: self.fail("unexpected sync"))
            engine = NS(global_steps=0, _config=NS(zero_config={"stage": 3}),
                        gradient_accumulation_steps=lambda: 32, is_gradient_accumulation_boundary=lambda: False,
                        backward=lambda loss: loss + 1, step=lambda: "updated")
            trace.wrap_deepspeed(engine)
            trace.context["micro_step"] = 95
            with trace.phase("accelerator_backward"):
                self.assertEqual(engine.backward(5), 6)
                self.assertEqual(engine.step(), "updated")
            records = [json.loads(line) for line in trace.path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([x["phase"] for x in records if x["event"] == "enter"],
                             ["accelerator_backward", "engine.backward", "engine.step"])
            self.assertNotIn("device_complete", [x["event"] for x in records])

    def test_sync_failure_has_no_device_complete_marker(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            count = [0]
            def sync():
                count[0] += 1
                if count[0] == 2:
                    raise RuntimeError("device error")
            trace = PhaseTrace({"enabled": True, "synchronize": True}, directory, 0, sync)
            with self.assertRaisesRegex(RuntimeError, "device error"):
                with trace.phase("forward"):
                    pass
            records = [json.loads(line) for line in trace.path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(records[-1]["where"], "post_sync")
            self.assertNotIn("device_complete", [x["event"] for x in records])


class AccumulationTests(unittest.TestCase):
    def run_real_loop(self, count, accumulation, stop=None, cap=None, fail_at=None):
        # Execute the repository's actual epoch loop with a stateful engine double.
        # Like Accelerate 1.15, backward updates whenever sync_gradients is true;
        # it does NOT independently infer the boundary from the YAML value.
        path = Path(__file__).resolve().parents[1] / "train/sft_sdar.py"
        main = next(n for n in ast.parse(path.read_text(encoding="utf-8")).body
                    if isinstance(n, ast.FunctionDef) and n.name == "main")
        loop = next(n for n in main.body if isinstance(n, ast.For) and isinstance(n.target, ast.Name)
                    and n.target.id == "epoch")
        function = ast.FunctionDef(name="run", args=ast.arguments(
            posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]), decorator_list=[],
            body=[ast.Global(names=["global_steps", "completed_optimizer_steps"]), loop])
        class Tensor:
            def to(self, *args): return self
            def detach(self): return self
            def float(self): return self
            def item(self): return 1.0
        class Progress(list):
            def set_postfix(self, **kw): pass
        records, events = [], []
        model = NS(global_steps=0, train=lambda: None, parameters=lambda: [])
        scheduler = NS(last_epoch=0)
        scheduler.step = lambda: setattr(scheduler, "last_epoch", scheduler.last_epoch + 1)
        current = [0]
        accelerator = NS(device=NS(type="cpu"), process_index=0, num_processes=4, is_local_main_process=True,
                         sync_gradients=True, reduce=lambda value, **kw: value,
                         clip_grad_norm_=lambda *args: events.append("clip"),
                         wait_for_everyone=lambda: events.append("barrier"),
                         end_training=lambda: events.append("end"))
        @contextmanager
        def accumulate(_):
            current[0] += 1
            accelerator.sync_gradients = current[0] % accumulation == 0 or current[0] == count
            yield
        accelerator.accumulate = accumulate
        def backward(loss):
            if fail_at == current[0]:
                raise RuntimeError("injected backward failure")
            if accelerator.sync_gradients:
                model.global_steps += 1
        accelerator.backward = backward
        batch = {key: Tensor() for key in ("extended_input_ids", "p_mask", "tok_idx_ext", "labels")}
        class Timer:
            def __init__(self, *args): pass
            def batches(self, data): return iter(data)
            def finish(self): return {}
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            trace = PhaseTrace({"enabled": True, "stop_after_micro_steps": stop}, directory, 0, lambda: None)
            env = dict(first_epoch=0, num_train_epochs=1, global_steps=0, completed_optimizer_steps=0,
                       model=model, optimizer=NS(step=lambda: None, zero_grad=lambda **kw: None,
                                                param_groups=[{"lr": 1e-5}]), lr_scheduler=scheduler,
                       accelerator=accelerator, trace=trace, output_dir=directory, tokenizer=None,
                       train_dataloader_lm=[batch] * count, tqdm=lambda rows, **kw: Progress(rows),
                       create_profiler=lambda *a, **kw: DisabledProfiler(), MicrostepTimer=Timer,
                       forward_process=lambda **kw: Tensor(), torch=NS(isfinite=lambda x: NS(all=lambda: True)),
                       logger=NS(info=lambda *a: None), max_optimizer_steps=cap, validation_enabled=False,
                       config=NS(get=lambda *a: {}, model=NS(optimized_name="optimized"),
                                 training=NS(max_grad_norm=1, get=lambda *a: False)),
                       metrics_logger=NS(log=lambda phase, epoch, micro, update, **kw:
                                         records.append(dict(micro=micro, update=update, **kw))),
                       save_checkpoint=lambda *a, **kw: events.append("save"))
            exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), str(path), "exec"), env)
            if fail_at:
                with self.assertRaisesRegex(RuntimeError, "injected backward failure"):
                    env["run"]()
            else:
                env["run"]()
            traces = [json.loads(line) for line in trace.path.read_text(encoding="utf-8").splitlines()]
        return records, events, traces

    def test_32_microsteps_per_update_and_diagnostic_stop(self):
        records, events, _ = self.run_real_loop(160, 32, stop=128)
        self.assertEqual([r["micro"] for r in records if r["optimizer_updated"]], [32, 64, 96, 128])
        self.assertEqual(records[93]["update"], 2)
        self.assertEqual(records[93]["scheduler_step"], 2)
        self.assertEqual(records[-1]["update"], 4)
        self.assertEqual(events, ["clip"] * 4 + ["barrier", "end"])

    def test_epoch_remainder_flush_and_update_cap(self):
        records, events, _ = self.run_real_loop(70, 32)
        self.assertEqual([r["micro"] for r in records if r["optimizer_updated"]], [32, 64, 70])
        self.assertIn("save", events)
        records, _, _ = self.run_real_loop(160, 32, cap=3)
        self.assertEqual(records[-1]["micro"], 96)

    def test_rank_error_is_localized_to_microstep_95(self):
        records, _, traces = self.run_real_loop(160, 32, fail_at=95)
        self.assertEqual(records[-1]["micro"], 94)
        self.assertEqual(traces[-1]["phase"], "accelerator_backward")
        self.assertEqual(traces[-1]["micro_step"], 95)
        self.assertEqual(traces[-1]["event"], "error")


if __name__ == "__main__":
    unittest.main(verbosity=2)
