#!/usr/bin/env bash
set -euo pipefail

# Two-stage RMBench training launcher.
#
#   bash script/rmbench_train/train_rmbench_readable.sh smoke
#   bash script/rmbench_train/train_rmbench_readable.sh formal
#
# Optional overrides:
#   bash script/rmbench_train/train_rmbench_readable.sh smoke 10 /absolute/output/path

MODE="${1:-smoke}"
STEPS="${2:-}"
OUTPUT_ROOT="${3:-}"

REPO_ROOT="/root/wyc/code/lingbot-va2"
PYTHON="/root/miniconda3/envs/lingbot/bin/python"
SEED=42
GPU_COUNT=3
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

case "${MODE}" in
  smoke)
    CONFIG_NAME="rmbench_overfit_train"
    STEPS="${STEPS:-10}"
    RUN_KIND="episode0_smoke"
    ;;
  formal)
    CONFIG_NAME="rmbench_train"
    STEPS="${STEPS:-1000}"
    RUN_KIND="50episode_formal"
    ;;
  *)
    echo "Usage: $0 {smoke|formal} [steps] [absolute_output_path]" >&2
    exit 2
    ;;
esac

if ! [[ "${STEPS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "steps must be a positive integer, got: ${STEPS}" >&2
  exit 2
fi

if [[ -z "${OUTPUT_ROOT}" ]]; then
  OUTPUT_ROOT="${REPO_ROOT}/checkpoints/rmbench_${RUN_KIND}_${STEPS}step_${TIMESTAMP}"
fi

mkdir -p "${OUTPUT_ROOT}"
cd "${REPO_ROOT}"

# Reproducibility and runtime settings shared by smoke and formal runs.
export PYTHONHASHSEED="${SEED}"
export CUDA_VISIBLE_DEVICES="0,1,2"
export PYTORCH_ALLOC_CONF="expandable_segments:True"
export WANDB_PROJECT="${WANDB_PROJECT:-lingbot-va-rmbench}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-${RUN_KIND}_${STEPS}step_${TIMESTAMP}}"
export WANDB_MODE="${WANDB_MODE:-online}"

echo "============================================================"
echo "RMBench training"
echo "  mode:        ${MODE}"
echo "  dataset:     ${CONFIG_NAME}"
echo "  max steps:   ${STEPS}"
echo "  seed:        ${SEED}"
echo "  GPUs:        ${CUDA_VISIBLE_DEVICES}"
echo "  output:      ${OUTPUT_ROOT}"
echo "  W&B project: ${WANDB_PROJECT}"
echo "  W&B run:     ${WANDB_RUN_NAME}"
echo "============================================================"

"${PYTHON}" -m torch.distributed.run \
  --standalone \
  --nproc_per_node="${GPU_COUNT}" \
  -m wan_va.train \
  --config-name "${CONFIG_NAME}" \
  --save-root "${OUTPUT_ROOT}" \
  --num-steps "${STEPS}" \
  2>&1 | tee "${OUTPUT_ROOT}/training.log"

SUMMARY="${OUTPUT_ROOT}/training_summary.json"
CHECKPOINT="${OUTPUT_ROOT}/checkpoints/checkpoint_step_${STEPS}"

if [[ ! -s "${SUMMARY}" ]]; then
  echo "ERROR: training finished without ${SUMMARY}" >&2
  exit 1
fi

# A converged formal run may stop before STEPS; in that case early stopping
# writes its own final checkpoint, so require at least one checkpoint directory.
if ! compgen -G "${OUTPUT_ROOT}/checkpoints/checkpoint_step_*" >/dev/null; then
  echo "ERROR: training finished without a checkpoint" >&2
  exit 1
fi

echo "Training completed successfully."
echo "Summary: ${SUMMARY}"
if [[ -d "${CHECKPOINT}" ]]; then
  echo "Final checkpoint: ${CHECKPOINT}"
fi
