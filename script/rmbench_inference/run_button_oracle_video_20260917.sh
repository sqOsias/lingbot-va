#!/usr/bin/env bash
set -euo pipefail
while [ ! -f /root/wyc/code/lingbot-va2/artifacts/button_cross_history_20260917_a/DONE ]; do
  kill -0 2706408 2>/dev/null || { echo 'Preceding experiment exited without DONE'; exit 1; }
  sleep 15
done
sleep 10
cd /root/wyc/code/lingbot-va2
export RMBENCH_DATASET_PATH=/root/wyc/data/rmbench/lingbot_va2/rearrange_blocks_50demo_w193_apf16_official
export PYTHONPATH=/root/wyc/code/lingbot-va2:/root/wyc/code/lingbot-va2/script
export CUDA_VISIBLE_DEVICES=0,1,2
export WANDB_MODE=disabled
export PYTORCH_ALLOC_CONF=expandable_segments:True
export LD_LIBRARY_PATH=/root/miniconda3/envs/lingbot/targets/x86_64-linux/lib:/root/miniconda3/envs/lingbot/lib
exec /root/miniconda3/envs/lingbot/bin/python -m torch.distributed.run --standalone --nproc_per_node=3 script/rmbench_inference/probe_button_oracle_video_20260917.py /root/wyc/code/lingbot-va2/artifacts/button_oracle_video_20260917_a
