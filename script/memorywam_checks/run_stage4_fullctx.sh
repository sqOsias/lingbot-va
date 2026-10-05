#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_ROOT="${RMBENCH_DATA_ROOT:-/root/wyc/data/rmbench/lingbot_va2}"
DATASET_ROOT="${1:-${DATA_ROOT}/rearrange_blocks_50demo_fullctx_apf16_cf8}"
OUTPUT_ROOT="${2:-${REPO_ROOT}/artifacts/memorywam/stage4_alignment/rearrange_train}"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "${REPO_ROOT}"
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

"${PYTHON_BIN}" -B -m script.memorywam_checks.fullctx_sidecar \
  --dataset-root "${DATASET_ROOT}" \
  --output "${OUTPUT_ROOT}"

"${PYTHON_BIN}" -B -m script.memorywam_checks.trace_fullctx_dataset \
  --dataset-root "${DATASET_ROOT}" \
  --sidecar "${OUTPUT_ROOT}/episodes.jsonl" \
  --all \
  --quiet \
  --output "${OUTPUT_ROOT}/dataset_trace_all.json"

"${PYTHON_BIN}" -B -m script.memorywam_checks.run_checks
