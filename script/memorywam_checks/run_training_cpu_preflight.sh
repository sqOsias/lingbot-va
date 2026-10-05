#!/usr/bin/env bash
set -euo pipefail

REPO=${LINGBOT_VA_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
PYTHON=${PYTHON_BIN:-python}
DATA_ROOT=${RMBENCH_DATA_ROOT:-/root/wyc/data/rmbench/lingbot_va2}
INITIALIZER=${MEMORYWAM_INITIALIZER:-${REPO}/train_out/rearrange_blocks_50demo_1000step_20260921_180241/checkpoints/checkpoint_step_750}
TRAIN=${MEMORYWAM_TRAIN_DATASET:-${DATA_ROOT}/rearrange_blocks_50demo_fullctx_apf16_cf8}
VALIDATION=${MEMORYWAM_VALIDATION_DATASET:-${DATA_ROOT}/rearrange_blocks_5demo_validation_fullctx_apf16_cf8}
TRAIN_ALIGNMENT=${MEMORYWAM_TRAIN_ALIGNMENT:-${REPO}/artifacts/memorywam/stage4_alignment/rearrange_train/episodes.jsonl}
VALIDATION_ALIGNMENT=${MEMORYWAM_VALIDATION_ALIGNMENT:-${REPO}/artifacts/memorywam/stage4_alignment/rearrange_validation/episodes.jsonl}
OUTPUT=${1:-${REPO}/artifacts/memorywam/stage6_training_ready/cpu_preflight.json}

cd "${REPO}"
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
"${PYTHON}" -B -m script.memorywam_checks.run_checks

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
"${PYTHON}" -B -m script.memorywam_checks.preflight_training \
  --initializer "${INITIALIZER}" \
  --train-dataset "${TRAIN}" \
  --validation-dataset "${VALIDATION}" \
  --train-alignment "${TRAIN_ALIGNMENT}" \
  --validation-alignment "${VALIDATION_ALIGNMENT}" \
  --output "${OUTPUT}"

"${PYTHON}" - "${OUTPUT}" <<'PY'
import json, sys
report = json.load(open(sys.argv[1]))
assert report["status"] == "PASS", report
assert report["checkpoint"]["legacy_header_exact_match"], report
assert report["checkpoint"]["memorywam_new_parameters"] == ["gist_tokens"], report
print(f"CPU PREFLIGHT PASS: {sys.argv[1]}")
PY
