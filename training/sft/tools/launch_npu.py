#!/usr/bin/env python3
"""Launch the existing SFT entry or CPU checkpoint exporter from one YAML."""
import os
from pathlib import Path
import subprocess
import sys
import time

from omegaconf import OmegaConf


def main():
    root = Path(__file__).resolve().parents[1]
    cli = OmegaConf.from_cli()
    path = Path(cli.get("config", root / "configs/chemdiffagent_sft_npu_1l.yaml")).resolve()
    config = OmegaConf.merge(OmegaConf.load(path), cli)
    # Preserve the caller's output directory before the subprocess changes cwd.
    config.experiment.output_dir = str(Path(config.experiment.output_dir).expanduser().resolve())
    action = config.runtime.get("action", "train")
    if action not in ("train", "export"):
        raise ValueError("runtime.action must be train or export")
    work = Path(config.runtime.work_dir).resolve()
    work.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    for key, value in config.runtime.get("env", {}).items():
        if not isinstance(key, str) or not key or "=" in key or "\0" in key:
            raise ValueError("runtime.env keys must be valid environment variable names")
        if not isinstance(value, (str, bool, int, float)):
            raise ValueError(f"runtime.env.{key} must be a scalar")
        env[key] = str(value).lower() if isinstance(value, bool) else str(value)
    env.update(PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1",
               TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    paths = {"TMPDIR": "tmp", "HF_HOME": "cache/hf", "XDG_CACHE_HOME": "cache",
             "ASCEND_CACHE_PATH": "cache/ascend", "ASCEND_PROCESS_LOG_PATH": "logs/ascend",
             "TORCH_EXTENSIONS_DIR": "cache/torch_extensions", "TRITON_CACHE_DIR": "cache/triton",
             "WANDB_DIR": "wandb"}
    for key, relative in paths.items():
        value = work / relative
        value.mkdir(parents=True, exist_ok=True)
        env[key] = str(value)
    python = str(Path(config.runtime.python).absolute())
    env["PATH"] = str(Path(python).parent) + os.pathsep + env.get("PATH", "")
    run_id = f"{time.time_ns()}-{os.getpid()}"
    resolved = work / f"config-{run_id}.yaml"
    OmegaConf.save(config, resolved)
    if action == "export":
        command = [python, str(root / "tools/export_reduced_sdar.py"), f"config={resolved}"]
    else:
        print(f"[SFT launch] config={path} epochs={config.training.num_train_epochs} "
              f"max_steps={config.training.get('max_steps')} "
              f"gradient_accumulation_steps={config.training.gradient_accumulation_steps}", flush=True)
        checkpoint = Path(config.model.pretrained_model)
        if not (checkpoint / "config.json").is_file():
            raise FileNotFoundError("Reduced checkpoint missing; set source_model and run runtime.action=export first")
        launch = OmegaConf.load(root / "accelerate_configs/1_npu_zero3.yaml")
        launch.num_processes = int(config.runtime.get("num_processes", 1))
        launch.mixed_precision = config.training.mixed_precision
        launch.deepspeed_config.gradient_accumulation_steps = config.training.gradient_accumulation_steps
        launch.deepspeed_config.gradient_clipping = config.training.max_grad_norm
        launch_path = work / f"accelerate-{run_id}.yaml"
        OmegaConf.save(launch, launch_path)
        env["DS_ACCELERATOR"] = "npu"
        env.setdefault("HCCL_NPU_SOCKET_PORT_RANGE", "auto")
        devices = config.runtime.get("visible_devices")
        if devices is not None:
            env["ASCEND_RT_VISIBLE_DEVICES"] = str(devices)
        command = [python, "-m", "accelerate.commands.launch", "--config_file", str(launch_path),
                   "--main_process_port", str(config.runtime.get("main_process_port", 29527)),
                   str(root / "train/sft_sdar.py"), f"config={resolved}"]
    # Source only the configured CANN environment. Keep all caches/output under work.
    setup = str(Path(config.runtime.cann_root) / "set_env.sh")
    raise SystemExit(subprocess.call(
        ["bash", "-c", 'source "$1" || exit; shift; exec "$@"', "chemdiff-npu", setup, *command],
        env=env, cwd=root))


if __name__ == "__main__":
    main()
