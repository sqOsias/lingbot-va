#!/usr/bin/env bash
set -euo pipefail

# Closed-loop evaluation of the exact Rearrange Blocks episode0 used for overfit.
# Usage:
#   bash script/rmbench_inference/run_rearrange_episode0_seed0_inference.sh faithful
#   bash script/rmbench_inference/run_rearrange_episode0_seed0_inference.sh fast

MODE="${1:-faithful}"
REPO="/root/wyc/code/lingbot-va2"
RMBENCH="/root/wyc/code/RMBench"
DATASET="/root/wyc/data/rmbench/lingbot_va2/rearrange_blocks_50demo_w193_apf16_official"
TRAIN_ROOT="${REPO}/checkpoints/rearrange_blocks_episode0_overfit_1000step_20260915"
CHECKPOINT="${TRAIN_ROOT}/checkpoints/checkpoint_step_1000/transformer"
BASE_MODEL="${REPO}/checkpoints/base"
SEED_MANIFEST="${RMBENCH}/data/data/rearrange_blocks/demo_clean/episode0_seed_manifest.json"
PORT="${RMBENCH_INFERENCE_PORT:-29557}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT="${REPO}/artifacts/rearrange_blocks_episode0_seed0_inference_${MODE}_${TIMESTAMP}"

case "${MODE}" in
  faithful)
    VIDEO_STEPS=25
    ACTION_STEPS=50
    EXPECTED_MINUTES="45-55"
    ;;
  fast)
    VIDEO_STEPS=8
    ACTION_STEPS=16
    EXPECTED_MINUTES="15-25"
    ;;
  *)
    echo "Usage: $0 {faithful|fast}" >&2
    exit 2
    ;;
esac

if [[ ! -s "${TRAIN_ROOT}/training_summary.json" ]]; then
  echo "ERROR: training is not complete: ${TRAIN_ROOT}/training_summary.json is missing" >&2
  exit 1
fi
if [[ ! -s "${CHECKPOINT}/diffusion_pytorch_model.safetensors" ]]; then
  echo "ERROR: final checkpoint is missing: ${CHECKPOINT}" >&2
  exit 1
fi

mkdir -p "${OUTPUT}/model" "${OUTPUT}/eval" "${OUTPUT}/server"
ln -s "${CHECKPOINT}" "${OUTPUT}/model/transformer"
ln -s "${BASE_MODEL}/vae" "${OUTPUT}/model/vae"
ln -s "${BASE_MODEL}/tokenizer" "${OUTPUT}/model/tokenizer"
ln -s "${BASE_MODEL}/text_encoder" "${OUTPUT}/model/text_encoder"
ln -s "${BASE_MODEL}/configuration.json" "${OUTPUT}/model/configuration.json"

PROMPT_EMB="${DATASET}/latents/chunk-000/observation.images.cam_high/episode_000000_0_193.pth"
test -s "${PROMPT_EMB}"
test -s "${DATASET}/rmbench_action_stats.json"
test -s "${DATASET}/empty_emb.pt"

export RMBENCH_LINGBOT_MODEL="${OUTPUT}/model"
export RMBENCH_LINGBOT_STATS="${DATASET}/rmbench_action_stats.json"
export RMBENCH_LINGBOT_PROMPT_EMB="${PROMPT_EMB}"
export RMBENCH_LINGBOT_EMPTY_EMB="${DATASET}/empty_emb.pt"
export RMBENCH_LINGBOT_VIDEO_INFERENCE_STEPS="${VIDEO_STEPS}"
export RMBENCH_LINGBOT_ACTION_INFERENCE_STEPS="${ACTION_STEPS}"
export RMBENCH_LINGBOT_WINDOW_RESET_INTERVAL=0
export RMBENCH_LINGBOT_IMAGE_COLOR_ORDER=legacy_bgr
export RMBENCH_LINGBOT_STEP_TRACE_PATH="${OUTPUT}/step_trace.jsonl"
export RMBENCH_LINGBOT_TASK_STATE_TRACE=1
export RMBENCH_LINGBOT_CAPTURE_DIR="${OUTPUT}/capture"
export RMBENCH_EVAL_STEP_LIMIT=404
export RMBENCH_EVAL_STALL_ENABLE=1
export RMBENCH_EVAL_STALL_WARMUP_STEPS=192
export RMBENCH_EVAL_STALL_WINDOW_STEPS=96
export RMBENCH_EVAL_STALL_EEF_PATH_MM=10
export RMBENCH_EVAL_STALL_EEF_RANGE_MM=3
export RMBENCH_EVAL_STALL_COMMAND_PATH_MM=10
export RMBENCH_EVAL_STALL_GRIPPER_RANGE=0.002
export RMBENCH_EVAL_STALL_OBJECT_RANGE_MM=3
export WANDB_PROJECT="${WANDB_PROJECT:-lingbot-va-rmbench}"
export WANDB_INFERENCE_RUN_NAME="rearrange_episode0_${MODE}_${TIMESTAMP}"
export WANDB_MODE="${WANDB_MODE:-online}"
export LD_LIBRARY_PATH="/root/miniconda3/envs/lingbot/targets/x86_64-linux/lib:/root/miniconda3/envs/lingbot/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_ALLOC_CONF=expandable_segments:True

cat <<EOF
============================================================
Rearrange Blocks episode0 closed-loop inference
  mode:                 ${MODE}
  checkpoint:           ${CHECKPOINT}
  video/action steps:   ${VIDEO_STEPS}/${ACTION_STEPS}
  GPU allocation:       RMBench=0, LingBot server=1,2
  environment limit:    404 (training episode0 length)
  expected wall time:   ${EXPECTED_MINUTES} minutes
  output:               ${OUTPUT}
  validated seed:       0 (${SEED_MANIFEST})
============================================================
EOF

cd "${REPO}"
STARTED_AT="$(date +%s)"
CUDA_VISIBLE_DEVICES=1,2 /root/miniconda3/envs/lingbot/bin/python -m torch.distributed.run \
  --standalone --nproc_per_node=2 \
  -m wan_va.wan_va_server --config-name rmbench --port "${PORT}" \
  --save_root "${OUTPUT}/server" > "${OUTPUT}/server.log" 2>&1 &
SERVER_PID=$!

cleanup() {
  kill -TERM "${SERVER_PID}" 2>/dev/null || true
  wait "${SERVER_PID}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

for attempt in $(seq 1 300); do
  awk -v suffix=":$(printf '%04X' "${PORT}")" \
    '$2 ~ suffix "$" && $4 == "0A" {ready=1} END {exit(ready?0:1)}' \
    /proc/net/tcp /proc/net/tcp6 2>/dev/null && break
  kill -0 "${SERVER_PID}" 2>/dev/null || {
    echo "ERROR: inference server exited; inspect ${OUTPUT}/server.log" >&2
    tail -n 80 "${OUTPUT}/server.log" >&2 || true
    exit 1
  }
  [[ "${attempt}" -lt 300 ]] || {
    echo "ERROR: inference server startup timed out" >&2
    exit 1
  }
  sleep 1
done

cd "${RMBENCH}"
if ! CUDA_VISIBLE_DEVICES=0 /root/miniconda3/envs/RMBench/bin/python script/eval_policy.py \
    --config policy/LingBot_VA/deploy_policy.yml --overrides \
    --task_name rearrange_blocks --task_config demo_clean \
    --ckpt_setting step1000 --seed 0 --policy_name LingBot_VA --port "${PORT}" \
    --validated_seed_file "${SEED_MANIFEST}" \
    --instruction_type seen --eval_num_episodes 1 \
    --eval_output_dir "${OUTPUT}/eval" > "${OUTPUT}/client.log" 2>&1; then
  echo "ERROR: closed-loop inference failed." >&2
  echo "Client log: ${OUTPUT}/client.log" >&2
  echo "Server log: ${OUTPUT}/server.log" >&2
  echo "----- client log tail -----" >&2
  tail -n 80 "${OUTPUT}/client.log" >&2 || true
  echo "----- server log tail -----" >&2
  tail -n 80 "${OUTPUT}/server.log" >&2 || true
  exit 1
fi

FINISHED_AT="$(date +%s)"
ELAPSED_SECONDS="$((FINISHED_AT - STARTED_AT))"
printf '%s\n' "${ELAPSED_SECONDS}" > "${OUTPUT}/elapsed_seconds.txt"

echo "Inference finished in $((ELAPSED_SECONDS / 60)) minutes."
echo "Result: ${OUTPUT}/eval/_result.txt"
echo "Video directory: ${OUTPUT}/eval"
echo "Timing trace: ${OUTPUT}/step_trace.jsonl"
