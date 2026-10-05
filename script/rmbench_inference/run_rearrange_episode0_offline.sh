#!/usr/bin/env bash
set -euo pipefail

HISTORY_SOURCE="${1:-expert}"
if [[ "${HISTORY_SOURCE}" != "expert" && "${HISTORY_SOURCE}" != "predicted" ]]; then
  echo "Usage: $0 {expert|predicted}" >&2
  exit 2
fi

REPO="/root/wyc/code/lingbot-va2"
DATASET="/root/wyc/data/rmbench/lingbot_va2/rearrange_blocks_50demo_w193_apf16_official"
CHECKPOINT="${REPO}/checkpoints/rearrange_blocks_episode0_overfit_1000step_20260915/checkpoints/checkpoint_step_1000/transformer"
BASE="${REPO}/checkpoints/base"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT="${REPO}/artifacts/rearrange_blocks_episode0_offline_${HISTORY_SOURCE}_${TIMESTAMP}"
MODEL="${OUTPUT}/model"

mkdir -p "${MODEL}"
ln -s "${CHECKPOINT}" "${MODEL}/transformer"
ln -s "${BASE}/vae" "${MODEL}/vae"
ln -s "${BASE}/tokenizer" "${MODEL}/tokenizer"
ln -s "${BASE}/text_encoder" "${MODEL}/text_encoder"
ln -s "${BASE}/configuration.json" "${MODEL}/configuration.json"

export PYTHONHASHSEED=42
export RMBENCH_DATASET_PATH="${DATASET}"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0,1,2
export LD_LIBRARY_PATH="/root/miniconda3/envs/lingbot/targets/x86_64-linux/lib:/root/miniconda3/envs/lingbot/lib:${LD_LIBRARY_PATH:-}"

echo "Exact-data offline inference: Rearrange Blocks episode 0"
echo "Dataset: ${DATASET}"
echo "Checkpoint: ${CHECKPOINT}"
echo "Output: ${OUTPUT}"
echo "History source: ${HISTORY_SOURCE}"

cd "${REPO}"
/root/miniconda3/envs/lingbot/bin/python -m torch.distributed.run \
  --standalone --nproc_per_node=3 \
  script/rmbench_inference/infer_rearrange_episode0_offline.py \
  --dataset "${DATASET}" \
  --model "${MODEL}" \
  --output "${OUTPUT}" \
  --seed 42 \
  --video-steps 25 \
  --action-steps 50 \
  --history-source "${HISTORY_SOURCE}" \
  2>&1 | tee "${OUTPUT}/inference.log"

for name in predicted_episode0 target_episode0 target_left_predicted_right; do
  ffmpeg -y -v error -i "${OUTPUT}/${name}.mp4" \
    -c:v libx264 -preset medium -crf 18 -pix_fmt yuv420p -movflags +faststart \
    "${OUTPUT}/${name}_h264.mp4"
done

echo "Generated video: ${OUTPUT}/predicted_episode0.mp4"
echo "Comparison video: ${OUTPUT}/target_left_predicted_right.mp4"
echo "Metrics: ${OUTPUT}/metrics.json"
