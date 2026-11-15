#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
if [[ "${1:-}" == *.yaml || "${1:-}" == *.yml ]]; then
    exec "${PYTHON:-python3}" "${ROOT}/inference/launch_npu.py" "$@"
fi
# Source the server's CANN environment before invoking this script.
export DEVICE=${DEVICE:-npu:0}
export PROFILE=${PROFILE:-1}
export RXN_T5CHEM_DEVICE=${RXN_T5CHEM_DEVICE:-cpu}
bash "${ROOT}/scripts/run_inference.sh" "$@"
