#!/usr/bin/env bash
set -euo pipefail

REPO=${LINGBOT_VA_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
PYTHON=${PYTHON_BIN:-python}
DATA_ROOT=${RMBENCH_DATA_ROOT:-/root/wyc/data/rmbench/lingbot_va2}
DATASET=${MEMORYWAM_TRAIN_DATASET:-${DATA_ROOT}/rearrange_blocks_50demo_fullctx_apf16_cf8}
ALIGNMENT=${MEMORYWAM_TRAIN_ALIGNMENT:-${REPO}/artifacts/memorywam/stage4_alignment/rearrange_train/episodes.jsonl}
OUTPUT=${1:-${REPO}/artifacts/memorywam/stage7_cpu_2step_pass}

cd "${REPO}"
OMP_NUM_THREADS=${OMP_NUM_THREADS:-1} \
MKL_NUM_THREADS=${MKL_NUM_THREADS:-1} \
CUDA_VISIBLE_DEVICES='' \
"${PYTHON}" -B -m script.memorywam_checks.cpu_two_step_smoke \
  --dataset-root "${DATASET}" --alignment "${ALIGNMENT}" \
  --output "${OUTPUT}"

test -f "${OUTPUT}/cpu_2step_pass.json"
echo "CPU 2-STEP PASS: ${OUTPUT}/cpu_2step_pass.json"
