"""NPU training run directories follow the inference launcher convention."""
from datetime import datetime
from pathlib import Path


def run_directory(config_path, override=None, now=None):
    if override is not None:
        return Path(override).expanduser().resolve()
    stamp = (now or datetime.now()).strftime('%Y%m%d_%H%M%S_%f')
    return Path.cwd() / 'res' / f'{stamp}-{Path(config_path).stem}'
