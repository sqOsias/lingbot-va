#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT="${1:?Usage: $0 /absolute/checkpoint_step_N [faithful|fast] [output]}"
MODE="${2:-faithful}"
REPO=/root/wyc/code/lingbot-va2
RMBENCH=/root/wyc/code/RMBench
DATASET=/root/wyc/data/rmbench/lingbot_va2/rearrange_blocks_episode0_fullctx_apf16
BASE=${REPO}/checkpoints/base
SEED_MANIFEST=${RMBENCH}/data/data/rearrange_blocks/demo_clean/episode0_seed_manifest.json
PORT="${RMBENCH_INFERENCE_PORT:-29557}"
OUTPUT="${3:-${REPO}/artifacts/rearrange_blocks_episode0_fullctx_inference_${MODE}_$(date +%Y%m%d_%H%M%S)}"
case "${MODE}" in faithful) VIDEO_STEPS=25; ACTION_STEPS=50;; fast) VIDEO_STEPS=8; ACTION_STEPS=16;; *) exit 2;; esac
test -s "${CHECKPOINT}/transformer/diffusion_pytorch_model.safetensors"
test -s "${DATASET}/latents/chunk-000/observation.images.cam_high/episode_000000_0_401.pth"
mkdir -p "${OUTPUT}/model" "${OUTPUT}/eval" "${OUTPUT}/server"
ln -s "${CHECKPOINT}/transformer" "${OUTPUT}/model/transformer"
for name in vae tokenizer text_encoder configuration.json; do ln -s "${BASE}/${name}" "${OUTPUT}/model/${name}"; done
export RMBENCH_LINGBOT_MODEL="${OUTPUT}/model"
export RMBENCH_LINGBOT_STATS="${DATASET}/rmbench_action_stats.json"
export RMBENCH_LINGBOT_PROMPT_EMB="${DATASET}/latents/chunk-000/observation.images.cam_high/episode_000000_0_401.pth"
export RMBENCH_LINGBOT_EMPTY_EMB="${DATASET}/empty_emb.pt"
export RMBENCH_LINGBOT_VIDEO_INFERENCE_STEPS="${VIDEO_STEPS}" RMBENCH_LINGBOT_ACTION_INFERENCE_STEPS="${ACTION_STEPS}"
export RMBENCH_LINGBOT_WINDOW_RESET_INTERVAL=0 RMBENCH_LINGBOT_IMAGE_COLOR_ORDER=legacy_bgr
export RMBENCH_LINGBOT_STEP_TRACE_PATH="${OUTPUT}/step_trace.jsonl" RMBENCH_LINGBOT_TASK_STATE_TRACE=1
export RMBENCH_EVAL_STEP_LIMIT=404 RMBENCH_EVAL_STALL_ENABLE=0
export WANDB_PROJECT="${WANDB_PROJECT:-lingbot-va-rmbench}" WANDB_INFERENCE_RUN_NAME="rearrange_episode0_fullctx_${MODE}" WANDB_MODE="${WANDB_MODE:-online}"
export LD_LIBRARY_PATH="/root/miniconda3/envs/lingbot/targets/x86_64-linux/lib:/root/miniconda3/envs/lingbot/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_ALLOC_CONF=expandable_segments:True
cd "${REPO}"
CUDA_VISIBLE_DEVICES=1,2 /root/miniconda3/envs/lingbot/bin/python -m torch.distributed.run --standalone --nproc_per_node=2 \
  -m wan_va.wan_va_server --config-name rmbench --port "${PORT}" --save_root "${OUTPUT}/server" >"${OUTPUT}/server.log" 2>&1 &
SERVER_PID=$!
cleanup(){ kill -TERM "${SERVER_PID}" 2>/dev/null || true; wait "${SERVER_PID}" 2>/dev/null || true; }
trap cleanup EXIT INT TERM
for attempt in $(seq 1 300); do
  awk -v suffix=":$(printf '%04X' "${PORT}")" '$2 ~ suffix "$" && $4 == "0A" {ok=1} END {exit(ok?0:1)}' /proc/net/tcp /proc/net/tcp6 2>/dev/null && break
  kill -0 "${SERVER_PID}" 2>/dev/null || { tail -80 "${OUTPUT}/server.log"; exit 1; }
  [[ "${attempt}" -lt 300 ]] || exit 1; sleep 1
done
cd "${RMBENCH}"
CUDA_VISIBLE_DEVICES=0 /root/miniconda3/envs/RMBench/bin/python script/eval_policy.py \
  --config policy/LingBot_VA/deploy_policy.yml --overrides --task_name rearrange_blocks --task_config demo_clean \
  --ckpt_setting fullctx --seed 0 --policy_name LingBot_VA --port "${PORT}" --validated_seed_file "${SEED_MANIFEST}" \
  --instruction_type seen --eval_num_episodes 1 --eval_output_dir "${OUTPUT}/eval" >"${OUTPUT}/client.log" 2>&1
echo "Result: ${OUTPUT}/eval/_result.txt"
