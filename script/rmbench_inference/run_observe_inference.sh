#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: run_observe_inference.sh CHECKPOINT [train|validation|test] [COUNT] [faithful|fast] [OUTPUT]

CHECKPOINT: absolute path to checkpoint_step_N
train evaluates training scene seed 0 and requires COUNT=1.
validation/test use the independent expert-validated 5-seed manifests; COUNT=1..5.
EOF
}

[[ $# -ge 1 ]] || { usage >&2; exit 2; }
CHECKPOINT=$1
SPLIT=${2:-test}
COUNT=${3:-5}
MODE=${4:-faithful}
REPO=/root/wyc/code/lingbot-va2
RMBENCH=/root/wyc/code/RMBench
PYTHON=/root/miniconda3/envs/lingbot/bin/python
EVAL_PYTHON=/root/miniconda3/envs/RMBench/bin/python
BASE=${REPO}/checkpoints/base
DATASET=/root/wyc/data/rmbench/lingbot_va2/observe_and_pickup_50demo_fullctx_apf16_cf8
SEED_ROOT=/root/wyc/data/rmbench/lingbot_va2/holdout_work/observe_and_pickup
PORT=${RMBENCH_INFERENCE_PORT:-29571}
SERVER_START_TIMEOUT=${RMBENCH_SERVER_START_TIMEOUT:-1200}
EVAL_STEP_LIMIT=${RMBENCH_EVAL_STEP_LIMIT:-250}
STAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT=${5:-${REPO}/artifacts/observe_and_pickup_50demo_${SPLIT}_${COUNT}ep_${MODE}_${STAMP}}
CUDA_LIB=/root/miniconda3/envs/lingbot/targets/x86_64-linux/lib:/root/miniconda3/envs/lingbot/lib
SERVER_PID=""

[[ "${COUNT}" =~ ^[1-9][0-9]*$ ]] || { echo "COUNT must be positive" >&2; exit 2; }
case "${SPLIT}" in
  train) (( COUNT == 1 )) || { echo "train seed-0 replay requires COUNT=1" >&2; exit 2; } ;;
  validation|test) (( COUNT <= 5 )) || { echo "${SPLIT} contains five independent seeds" >&2; exit 2; } ;;
  *) usage >&2; exit 2 ;;
esac
case "${MODE}" in
  faithful) VIDEO_STEPS=25; ACTION_STEPS=50 ;;
  fast) VIDEO_STEPS=8; ACTION_STEPS=16 ;;
  *) usage >&2; exit 2 ;;
esac
[[ ! -e "${OUTPUT}" ]] || { echo "Output already exists: ${OUTPUT}" >&2; exit 2; }
test -s "${CHECKPOINT}/transformer/diffusion_pytorch_model.safetensors"
test -s "${DATASET}/rmbench_action_stats.json"
test -s "${DATASET}/empty_emb.pt"
PROMPT_EMB=$(find "${DATASET}/latents/chunk-000/observation.images.cam_high" -maxdepth 1 -name 'episode_000000_0_*.pth' -print -quit)
test -n "$PROMPT_EMB"
test -s "$PROMPT_EMB"

mkdir -p "${OUTPUT}/model" "${OUTPUT}/eval" "${OUTPUT}/server"
if [[ "${SPLIT}" == train ]]; then
  SEED_MANIFEST=${OUTPUT}/train_seed0_manifest.json
  printf '%s\n' '{"schema_version":1,"protocol":"training scene seed-0 replay","task":"observe_and_pickup","task_config":"demo_clean","complete":true,"seeds":[{"seed":0}]}' > "${SEED_MANIFEST}"
else
  SEED_MANIFEST=${SEED_ROOT}/${SPLIT}/expert_seeds.json
fi
test -s "${SEED_MANIFEST}"
"${PYTHON}" - "${SEED_MANIFEST}" "${COUNT}" <<'PY'
import json, sys
manifest = json.load(open(sys.argv[1]))
count = int(sys.argv[2])
assert manifest['task'] == 'observe_and_pickup', manifest
assert manifest['complete'] is True, manifest
assert len(manifest['seeds']) >= count, manifest
PY

ln -s "${CHECKPOINT}/transformer" "${OUTPUT}/model/transformer"
for name in vae tokenizer text_encoder configuration.json; do
  ln -s "${BASE}/${name}" "${OUTPUT}/model/${name}"
done

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
export RMBENCH_EVAL_STEP_LIMIT="${EVAL_STEP_LIMIT}"
export RMBENCH_EVAL_STALL_ENABLE=0
export WANDB_PROJECT="${WANDB_PROJECT:-lingbot-va-rmbench}"
export WANDB_INFERENCE_RUN_NAME="observe_and_pickup_50demo_${SPLIT}_${COUNT}ep_${MODE}_${STAMP}"
export WANDB_MODE="${WANDB_MODE:-online}"
export LD_LIBRARY_PATH="${CUDA_LIB}:${LD_LIBRARY_PATH:-}"
export PYTORCH_ALLOC_CONF=expandable_segments:True

cleanup() {
  if [[ -n "${SERVER_PID}" ]]; then
    kill -TERM "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

cd "${REPO}"
CUDA_VISIBLE_DEVICES=1,2 "${PYTHON}" -m torch.distributed.run \
  --nproc_per_node=2 --master_port="$((PORT + 100))" \
  -m wan_va.wan_va_server --config-name rmbench --port "${PORT}" \
  --save_root "${OUTPUT}/server" >"${OUTPUT}/server.log" 2>&1 &
SERVER_PID=$!

ready=0
for attempt in $(seq 1 "${SERVER_START_TIMEOUT}"); do
  if awk -v suffix=":$(printf '%04X' "${PORT}")" '$2 ~ suffix "$" && $4 == "0A" {ok=1} END {exit(ok?0:1)}' /proc/net/tcp /proc/net/tcp6 2>/dev/null; then
    ready=1
    break
  fi
  kill -0 "${SERVER_PID}" 2>/dev/null || { tail -100 "${OUTPUT}/server.log" >&2; exit 1; }
  sleep 1
done
(( ready == 1 )) || { echo "Inference server did not become ready" >&2; exit 1; }

cd "${RMBENCH}"
CUDA_VISIBLE_DEVICES=0 "${EVAL_PYTHON}" script/eval_policy.py \
  --config policy/LingBot_VA/deploy_policy.yml --overrides \
  --task_name observe_and_pickup --task_config demo_clean \
  --ckpt_setting "formal50_${SPLIT}" --seed 0 --policy_name LingBot_VA \
  --port "${PORT}" --validated_seed_file "${SEED_MANIFEST}" \
  --instruction_type seen --eval_num_episodes "${COUNT}" \
  --eval_output_dir "${OUTPUT}/eval" >"${OUTPUT}/client.log" 2>&1

test -s "${OUTPUT}/eval/_result.txt"
echo "Result: ${OUTPUT}/eval/_result.txt"
