#!/usr/bin/env bash
set -euo pipefail

STEPS="${1:-1000}"
OUTPUT="${2:-/root/wyc/code/lingbot-va2/checkpoints/rearrange_blocks_episode0_fullctx_${STEPS}step_$(date +%Y%m%d_%H%M%S)}"
REPO=/root/wyc/code/lingbot-va2
DATASET=/root/wyc/data/rmbench/lingbot_va2/rearrange_blocks_episode0_fullctx_apf16
BASE="${LINGBOT_PRETRAINED_MODEL:-${REPO}/checkpoints/base}"
[[ "${STEPS}" =~ ^[1-9][0-9]*$ ]] || { echo "steps must be positive" >&2; exit 2; }
test -s "${DATASET}/full_context_audit.json"
/root/miniconda3/envs/lingbot/bin/python -c "import json; assert json.load(open('${DATASET}/full_context_audit.json'))['status']=='PASS'"
mkdir -p "${OUTPUT}"
export RMBENCH_DATASET_PATH="${DATASET}"
export LINGBOT_PRETRAINED_MODEL="${BASE}"
export PYTHONHASHSEED=42 CUDA_VISIBLE_DEVICES=0,1,2 PYTORCH_ALLOC_CONF=expandable_segments:True
export WANDB_PROJECT="${WANDB_PROJECT:-lingbot-va-rmbench}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-rearrange_episode0_fullctx_${STEPS}step}"
export WANDB_MODE="${WANDB_MODE:-online}"
export LD_LIBRARY_PATH="/root/miniconda3/envs/lingbot/targets/x86_64-linux/lib:/root/miniconda3/envs/lingbot/lib:${LD_LIBRARY_PATH:-}"
cd "${REPO}"
echo "Full-context episode0 training: dataset=${DATASET} base=${BASE} steps=${STEPS} output=${OUTPUT}"
/root/miniconda3/envs/lingbot/bin/python -m torch.distributed.run --standalone --nproc_per_node=3 \
  -m wan_va.train --config-name rmbench_overfit_train --save-root "${OUTPUT}" --num-steps "${STEPS}" \
  2>&1 | tee "${OUTPUT}/training.log"
test -s "${OUTPUT}/training_summary.json"
echo "Completed: ${OUTPUT}/training_summary.json"
