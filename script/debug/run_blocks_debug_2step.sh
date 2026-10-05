#!/usr/bin/env bash
set -euo pipefail

REPO=/root/wyc/code/lingbot-va2
WINDOW_ROOT=/root/wyc/data/rmbench/lingbot_va2/blocks_ranking_try_bounded_v1_h32_f64
STAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT="${REPO}/train_out/debug_blocks_2step_${STAMP}"

cd "${REPO}"

export BLOCKS_SMOKE_FORCE_EPISODE=20
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_DEBUG=WARN
export CUDA_LAUNCH_BLOCKING=1
export WANDB_MODE=offline
export PYTHONFAULTHANDLER=1

bash script/rmbench_train/train_blocks_bounded.sh \
  2 \
  "${WINDOW_ROOT}" \
  "${OUTPUT}"

echo "Debug output: ${OUTPUT}"
echo "Log: ${OUTPUT}/training.log"
echo "Config: ${OUTPUT}/run_config.json"
echo "Summary: ${OUTPUT}/training_summary.json"