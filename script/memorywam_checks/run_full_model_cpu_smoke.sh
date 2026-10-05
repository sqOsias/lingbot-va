#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: run_full_model_cpu_smoke.sh STAGE [OUTPUT] [extra Python arguments]

STAGE: plan | load | forward | train-gist | train-all

Examples:
  bash script/memorywam_checks/run_full_model_cpu_smoke.sh plan
  bash script/memorywam_checks/run_full_model_cpu_smoke.sh load
  bash script/memorywam_checks/run_full_model_cpu_smoke.sh forward
  bash script/memorywam_checks/run_full_model_cpu_smoke.sh train-gist -- --steps 2
  bash script/memorywam_checks/run_full_model_cpu_smoke.sh train-all -- \
    --steps 2 --allow-full-backbone-update

Use --save-full-checkpoint only when at least 24 GiB (gist) or 80 GiB (all)
of output-disk space is available. Each run creates a new timestamped output.
EOF
}

STAGE=${1:-}
case "${STAGE}" in
  plan|load|forward|train-gist|train-all) ;;
  *) usage >&2; exit 2 ;;
esac
shift

REPO=${LINGBOT_VA_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
PYTHON=${PYTHON_BIN:-python}
DATA_ROOT=${RMBENCH_DATA_ROOT:-/root/wyc/data/rmbench/lingbot_va2}
INITIALIZER=${MEMORYWAM_INITIALIZER:-${REPO}/train_out/rearrange_blocks_50demo_1000step_20260921_180241/checkpoints/checkpoint_step_750}
DATASET=${MEMORYWAM_TRAIN_DATASET:-${DATA_ROOT}/rearrange_blocks_50demo_fullctx_apf16_cf8}
ALIGNMENT=${MEMORYWAM_TRAIN_ALIGNMENT:-${REPO}/artifacts/memorywam/stage4_alignment/rearrange_train/episodes.jsonl}
STAMP=$(date +%Y%m%d_%H%M%S)
if [[ $# -gt 0 && "$1" != "--" ]]; then
  OUTPUT=$1
  shift
else
  OUTPUT=${REPO}/artifacts/memorywam/stage8_full_cpu_${STAGE}_${STAMP}
fi
if [[ ${1:-} == "--" ]]; then
  shift
fi

cd "${REPO}"
OMP_NUM_THREADS=${OMP_NUM_THREADS:-32} \
MKL_NUM_THREADS=${MKL_NUM_THREADS:-32} \
CUDA_VISIBLE_DEVICES='' \
"${PYTHON}" -B -m script.memorywam_checks.full_model_cpu_smoke \
  --stage "${STAGE}" --initializer "${INITIALIZER}" \
  --dataset-root "${DATASET}" --alignment "${ALIGNMENT}" \
  --output "${OUTPUT}" "$@"

test -f "${OUTPUT}/full_cpu_pass.json"
echo "FULL MODEL CPU ${STAGE} PASS: ${OUTPUT}/full_cpu_pass.json"
