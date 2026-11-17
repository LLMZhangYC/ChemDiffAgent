#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
# Reuse the activated environment, or an explicitly selected interpreter.
# Export the resolved path for configs using ${oc.env:CHEMDIFF_PYTHON}.
PYTHON_BIN=${CHEMDIFF_PYTHON:-python}
export PYTHONDONTWRITEBYTECODE=1
export CHEMDIFF_PYTHON
CHEMDIFF_PYTHON=$("${PYTHON_BIN}" -c 'import sys; print(sys.executable)')
exec "${CHEMDIFF_PYTHON}" "${ROOT}/training/sft/tools/launch_npu.py" "$@"
