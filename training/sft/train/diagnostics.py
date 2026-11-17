"""Opt-in per-rank phase tracing; never adds a distributed collective."""
from contextlib import contextmanager
from functools import wraps
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import time
import traceback


class PhaseTrace:
    def __init__(self, settings, output_dir, rank, synchronize):
        self.enabled = settings.get("enabled", False)
        self.sync = settings.get("synchronize", False)
        self.start = settings.get("start_micro_step", 1)
        self.stop = settings.get("stop_after_micro_steps", None)
        for key, value in (("start_micro_step", self.start), ("stop_after_micro_steps", self.stop)):
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
                raise ValueError(f"diagnostics.{key} must be a positive integer or null")
        if self.start is None:
            raise ValueError("diagnostics.start_micro_step must be a positive integer")
        if self.stop is not None and not self.enabled:
            raise ValueError("diagnostics.stop_after_micro_steps requires diagnostics.enabled=true")
        if self.stop is not None and self.stop < self.start:
            raise ValueError("diagnostics.stop_after_micro_steps must be >= start_micro_step")
        self.rank = rank
        self.synchronize = synchronize
        self.context = {"epoch": None, "micro_step": 0}
        self.engine = None
        self.path = Path(output_dir) / "diagnostics" / f"rank_{rank}.jsonl"
        if self.enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            packages = {}
            for name in ("accelerate", "deepspeed", "torch", "torch-npu", "transformers"):
                try:
                    packages[name] = version(name)
                except PackageNotFoundError:
                    packages[name] = None
            self.emit("diagnostics", "enabled", versions=packages, synchronize=self.sync,
                      visible_devices=os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
                      expansion_mode=os.environ.get("HCCL_OP_EXPANSION_MODE", "unset"))
            print(f"[SFT diagnostics] rank={rank} file={self.path}", flush=True)

    def active(self):
        return self.enabled and (self.context["micro_step"] == 0 or self.context["micro_step"] >= self.start)

    def emit(self, phase, event, **fields):
        if not self.active():
            return
        record = dict(time_ns=time.time_ns(), monotonic_ns=time.monotonic_ns(), pid=os.getpid(),
                      rank=self.rank, **self.context, phase=phase, event=event, **fields)
        if self.engine is not None:
            record["engine_optimizer_step"] = self.engine.global_steps
        # Close each record so a later process/device failure does not lose it.
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    @contextmanager
    def phase(self, name, **fields):
        if not self.active():
            yield
            return
        self.emit(name, "enter", **fields)
        where = "body"
        try:
            if self.sync:
                where = "pre_sync"
                self.emit(name, "pre_sync")
                self.synchronize()
            where = "body"
            self.emit(name, "body_begin")
            yield
            self.emit(name, "host_return")
            if self.sync:
                where = "post_sync"
                self.emit(name, "post_sync")
                self.synchronize()
                self.emit(name, "device_complete")
        except Exception:
            self.emit(name, "error", where=where, traceback=traceback.format_exc())
            print(f"[SFT diagnostics] rank={self.rank} micro_step={self.context['micro_step']} "
                  f"phase={name} failed in {where}; see {self.path}", flush=True)
            raise

    def batches(self, iterable, epoch, first_micro_step):
        iterator = iter(iterable)
        sentinel = object()
        micro_step = first_micro_step
        while True:
            self.context.update(epoch=epoch, micro_step=micro_step)
            with self.phase("data_fetch_and_timer_start"):
                batch = next(iterator, sentinel)
            if batch is sentinel:
                self.context["micro_step"] = micro_step - 1
                return
            yield batch
            micro_step += 1

    def wrap_deepspeed(self, engine):
        if not self.enabled:
            return
        self.engine = engine
        self.emit("deepspeed", "configuration", gradient_accumulation_steps=engine.gradient_accumulation_steps(),
                  zero_config=str(engine._config.zero_config))
        # Accelerate calls both methods inside accelerator.backward(). Observe
        # their original bound methods without replacing DeepSpeed's behavior.
        def wrapped(original, name):
            @wraps(original)
            def call(*args, **kwargs):
                with self.phase(name, accumulation_boundary=engine.is_gradient_accumulation_boundary()):
                    return original(*args, **kwargs)
            return call
        engine.backward = wrapped(engine.backward, "engine.backward")
        engine.step = wrapped(engine.step, "engine.step")

    def should_stop(self, micro_step):
        return self.enabled and self.stop is not None and micro_step >= self.stop
