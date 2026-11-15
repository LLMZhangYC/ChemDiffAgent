"""Optional NPU profiling, isolated from checkpoint saving and CPU imports."""
from pathlib import Path
import json
from importlib.metadata import PackageNotFoundError, version


class DisabledProfiler:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def step(self):
        pass


class WindowProfiler:
    """Stop the native profiler immediately after the configured finite window."""
    def __init__(self, profiler, steps):
        self.profiler = profiler
        self.remaining = steps
        self.running = False

    def __enter__(self):
        self.profiler.__enter__()
        self.running = True
        return self

    def __exit__(self, *exc):
        if self.running:
            self.running = False
            return self.profiler.__exit__(*exc)
        return False

    def step(self):
        if self.running:
            self.profiler.step()
            self.remaining -= 1
            if self.remaining == 0:
                self.__exit__(None, None, None)


def create_profiler(settings, output_dir, rank, device_type, epoch, world_size=1):
    """Create the first epoch's finite window; one step is one optimizer update.

    Matches Kimi's Level1/PipeUtilization setup. Relative output paths are
    anchored to the training output directory, not the subprocess working dir.
    """
    if not settings.get("enabled", False):
        return DisabledProfiler()
    if device_type != "npu":
        raise ValueError("NPU profiling requires an NPU training device")

    window = {}
    for name, default, minimum in (("skip_first", 0, 0), ("wait", 0, 0),
                                    ("warmup", 1, 0), ("active", 2, 1), ("repeat", 1, 1)):
        value = settings.get(name, default)
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"profiling.{name} must be an integer >= {minimum}")
        window[name] = value

    import torch_npu

    api = torch_npu.profiler
    directory = Path(settings.get("output_dir", "prof"))
    if not directory.is_absolute():
        directory = Path(output_dir) / directory
    directory = directory / f"rank_{rank}" / f"epoch_{epoch}"
    directory.mkdir(parents=True, exist_ok=True)
    try:
        torch_version = version("torch")
    except PackageNotFoundError:
        torch_version = None
    metadata = {"rank": rank, "world_size": world_size, "epoch": epoch,
                "schedule_unit": "optimizer_step", "schedule": window,
                "torch_version": torch_version,
                "torch_npu_version": getattr(torch_npu, "__version__", None)}
    (directory / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    experimental = api._ExperimentalConfig(
        profiler_level=api.ProfilerLevel.Level1,
        aic_metrics=api.AiCMetrics.PipeUtilization,
    )
    native = api.profile(
        activities=[api.ProfilerActivity.CPU, api.ProfilerActivity.NPU],
        schedule=api.schedule(**window),
        on_trace_ready=api.tensorboard_trace_handler(str(directory)),
        experimental_config=experimental,
        record_shapes=settings.get("record_shapes", False),
        profile_memory=settings.get("profile_memory", False),
        with_stack=settings.get("with_stack", False),
    )
    steps = window["skip_first"] + window["repeat"] * (window["wait"] + window["warmup"] + window["active"])
    return WindowProfiler(native, steps)
