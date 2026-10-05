#!/usr/bin/env bash
set -euo pipefail

REPO=/root/wyc/code/lingbot-va2
PYTHON=/root/miniconda3/envs/lingbot/bin/python
CUDA_LIB=/root/miniconda3/envs/lingbot/targets/x86_64-linux/lib:/root/miniconda3/envs/lingbot/lib
OVERFIT=$REPO/checkpoints/rmbench_episode0_official_overfit_1k
FORMAL=$REPO/checkpoints/rmbench_blocks_ranking_50demo_official_3k

run_training() {
  local config_name=$1
  local output_root=$2
  local port=$3
  local title=$4
  mkdir -p "$output_root"
  cat > "$output_root/run_config.json" <<EOF
{"config":"$config_name","base":"$REPO/checkpoints/base","world_size":3,"batch_per_gpu":1,"gradient_accumulation":1,"learning_rate":1e-5,"fsdp_cpu_offload":true,"seed":42,"early_stop":true}
EOF
  cd "$REPO"
  CUDA_VISIBLE_DEVICES=0,1,2 \
  LD_LIBRARY_PATH="$CUDA_LIB" \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$PYTHON" -m torch.distributed.run \
    --nproc_per_node=3 --local-ranks-filter=0 --master_port="$port" --tee=3 \
    -m wan_va.train --config-name "$config_name" --save-root "$output_root" \
    2>&1 | tee "$output_root/training.log"
  "$PYTHON" "$REPO/script/rmbench_train/plot_loss_svg.py" \
    --metrics "$output_root/training_metrics.jsonl" \
    --output "$output_root/loss_curve.svg" --title "$title"
}

run_training rmbench_overfit_train "$OVERFIT" 29620 "RMBench episode0 official LingBot overfit"
test -s "$OVERFIT/training_summary.json"
find "$OVERFIT/checkpoints" -name diffusion_pytorch_model.safetensors -type f -size +0c -print -quit | grep -q .
run_training rmbench_train "$FORMAL" 29621 "RMBench 50-demo official LingBot training"
test -s "$FORMAL/training_summary.json"
find "$FORMAL/checkpoints" -name diffusion_pytorch_model.safetensors -type f -size +0c -print -quit | grep -q .
date --iso-8601=seconds > "$REPO/checkpoints/rmbench_two_trainings_complete.txt"
