#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
# The existing environment is reused; project files, caches and outputs live in /data/hd.
PYTHON_BIN=${CHEMDIFF_PYTHON:-/data/workspace/zyc/ChemDiffAgent/envs/sft/bin/python}
export PYTHONDONTWRITEBYTECODE=1
# Disable OpenTelemetry export by default, independently of NPU profiling.
export OTEL_SDK_DISABLED=${OTEL_SDK_DISABLED:-true}
export OTEL_METRICS_EXPORTER=${OTEL_METRICS_EXPORTER:-none}
exec "${PYTHON_BIN}" "${ROOT}/training/sft/tools/launch_npu.py" "$@"
