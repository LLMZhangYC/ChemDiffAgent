#!/usr/bin/env bash
# No model, DeepSpeed, Accelerate, YAML, downloads, or package installation.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=${CHEMDIFF_PYTHON:-python}
CANN_ROOT=${CHEMDIFF_CANN_ROOT:-/usr/local/Ascend/ascend-toolkit/latest}
RUN_DIR=${HCCL_REPRO_RUN_DIR:-${ROOT}/res/$(date +%Y%m%d_%H%M%S_%N)-hccl_allgather}
mkdir -p -- "$(dirname -- "${RUN_DIR}")"
mkdir -- "${RUN_DIR}" # Refuse to mix evidence from different runs.
RUN_DIR=$(cd -- "${RUN_DIR}" && pwd)
mkdir -p -- "${RUN_DIR}/output" "${RUN_DIR}/work/tmp" "${RUN_DIR}/work/cache/ascend" \
  "${RUN_DIR}/work/cache/torch_extensions" "${RUN_DIR}/work/logs/ascend"
export TMPDIR="${RUN_DIR}/work/tmp" TMP="${RUN_DIR}/work/tmp" TEMP="${RUN_DIR}/work/tmp"
export XDG_CACHE_HOME="${RUN_DIR}/work/cache" ASCEND_CACHE_PATH="${RUN_DIR}/work/cache/ascend"
export TORCH_EXTENSIONS_DIR="${RUN_DIR}/work/cache/torch_extensions"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
if [[ ! -r "${CANN_ROOT}/set_env.sh" ]]; then
  echo "CANN set_env.sh missing: ${CANN_ROOT}; set CHEMDIFF_CANN_ROOT to the training environment." >&2
  exit 2
fi
set +u # Vendor environment scripts may reference unset shell variables.
source "${CANN_ROOT}/set_env.sh"
set -u
export ASCEND_PROCESS_LOG_PATH="${RUN_DIR}/work/logs/ascend"
export ASCEND_RT_VISIBLE_DEVICES=${HCCL_REPRO_DEVICES:-4,5,6,7}
export HCCL_ENTRY_LOG_ENABLE=1
export ASCEND_GLOBAL_LOG_LEVEL=${HCCL_REPRO_LOG_LEVEL:-2}
# Preserve the training environment's HCCL algorithm/expansion selection.
printf '[HCCL repro] output=%s\n[HCCL repro] CANN=%s devices=%s expansion=%s\n' \
  "${RUN_DIR}" "${CANN_ROOT}" "${ASCEND_RT_VISIBLE_DEVICES}" "${HCCL_OP_EXPANSION_MODE:-unset}"
"${PYTHON_BIN}" -u -B -m torch.distributed.run --nnodes=1 --nproc-per-node=4 \
  --master-addr=127.0.0.1 --master-port="${HCCL_REPRO_PORT:-29547}" --max-restarts=0 \
  "${ROOT}/training/sft/tools/repro_hccl_allgather.py" --output-dir "${RUN_DIR}/output" "$@" \
  2>&1 | tee "${RUN_DIR}/console.log"
