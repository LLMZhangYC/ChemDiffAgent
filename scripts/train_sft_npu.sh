#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
# The existing environment is reused; project files, caches and outputs live in /data/hd.
PYTHON_BIN=${CHEMDIFF_PYTHON:-/data/workspace/zyc/ChemDiffAgent/envs/sft/bin/python}
export PYTHONDONTWRITEBYTECODE=1
exec "${PYTHON_BIN}" "${ROOT}/training/sft/tools/launch_npu.py" "$@"
