#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  run_observe_blocks_inference.sh OBS_CHECKPOINT BLOCKS_CHECKPOINT [COUNT] [PROTOCOL] [MODE] [OUTPUT_ROOT]

  COUNT:     number of episodes for each task (default: 1)
  PROTOCOL:  overfit | generalize (default: overfit)
             overfit replays training seed 0 and requires COUNT=1
             generalize uses official validated seeds and supports COUNT=1..100
  MODE:      faithful | fast (default: faithful)
EOF
}

[[ $# -ge 2 ]] || { usage >&2; exit 2; }
OBS_CHECKPOINT=$1
BLOCKS_CHECKPOINT=$2
COUNT=${3:-1}
PROTOCOL=${4:-overfit}
MODE=${5:-faithful}

REPO=/root/wyc/code/lingbot-va2
RMBENCH=/root/wyc/code/RMBench
PYTHON=/root/miniconda3/envs/lingbot/bin/python
EVAL_PYTHON=/root/miniconda3/envs/RMBench/bin/python
BASE=${REPO}/checkpoints/base
STAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_ROOT=${6:-${REPO}/artifacts/observe_blocks_inference_${PROTOCOL}_${COUNT}ep_${STAMP}}
CUDA_LIB=/root/miniconda3/envs/lingbot/targets/x86_64-linux/lib:/root/miniconda3/envs/lingbot/lib
SERVER_START_TIMEOUT=${RMBENCH_SERVER_START_TIMEOUT:-1200}
SERVER_PID=""

[[ "${COUNT}" =~ ^[1-9][0-9]*$ ]] || { echo "COUNT must be a positive integer" >&2; exit 2; }
[[ "${SERVER_START_TIMEOUT}" =~ ^[1-9][0-9]*$ ]] || { echo "RMBENCH_SERVER_START_TIMEOUT must be a positive integer" >&2; exit 2; }
(( COUNT <= 100 )) || { echo "COUNT cannot exceed the 100 validated seeds" >&2; exit 2; }
case "${PROTOCOL}" in
  overfit) (( COUNT == 1 )) || { echo "overfit protocol requires COUNT=1 because validated seeds must be unique" >&2; exit 2; } ;;
  generalize) ;;
  *) echo "PROTOCOL must be overfit or generalize" >&2; exit 2 ;;
esac
case "${MODE}" in
  faithful) VIDEO_STEPS=25; ACTION_STEPS=50 ;;
  fast) VIDEO_STEPS=8; ACTION_STEPS=16 ;;
  *) echo "MODE must be faithful or fast" >&2; exit 2 ;;
esac
[[ ! -e "${OUTPUT_ROOT}" ]] || { echo "Output already exists: ${OUTPUT_ROOT}" >&2; exit 2; }
test -s "${OBS_CHECKPOINT}/transformer/diffusion_pytorch_model.safetensors"
test -s "${BLOCKS_CHECKPOINT}/transformer/diffusion_pytorch_model.safetensors"
mkdir -p "${OUTPUT_ROOT}"

write_overfit_manifest() {
  local task=$1
  local path=$2
  cat > "${path}" <<EOF
{"schema_version":1,"protocol":"episode 0 exact seed replay","task":"${task}","task_config":"demo_clean","complete":true,"seeds":[{"seed":0,"episode_info":{"source_episode":0}}]}
EOF
}

run_inference() {
  local task=$1
  local checkpoint=$2
  local dataset=$3
  local prompt_name=$4
  local step_limit=$5
  local port=$6
  local output=${OUTPUT_ROOT}/${task}
  local seed_manifest

  mkdir -p "${output}/model" "${output}/eval" "${output}/server"
  if [[ "${PROTOCOL}" == overfit ]]; then
    seed_manifest=${output}/seed0_manifest.json
    write_overfit_manifest "${task}" "${seed_manifest}"
  else
    seed_manifest=${RMBENCH}/expert_seed_manifests/full_100/${task}.json
  fi
  test -s "${seed_manifest}"
  test -s "${dataset}/latents/chunk-000/observation.images.cam_high/${prompt_name}"

  ln -s "${checkpoint}/transformer" "${output}/model/transformer"
  for name in vae tokenizer text_encoder configuration.json; do
    ln -s "${BASE}/${name}" "${output}/model/${name}"
  done

  export RMBENCH_LINGBOT_MODEL="${output}/model"
  export RMBENCH_LINGBOT_STATS="${dataset}/rmbench_action_stats.json"
  export RMBENCH_LINGBOT_PROMPT_EMB="${dataset}/latents/chunk-000/observation.images.cam_high/${prompt_name}"
  export RMBENCH_LINGBOT_EMPTY_EMB="${dataset}/empty_emb.pt"
  export RMBENCH_LINGBOT_VIDEO_INFERENCE_STEPS="${VIDEO_STEPS}"
  export RMBENCH_LINGBOT_ACTION_INFERENCE_STEPS="${ACTION_STEPS}"
  export RMBENCH_LINGBOT_WINDOW_RESET_INTERVAL=0
  export RMBENCH_LINGBOT_IMAGE_COLOR_ORDER=legacy_bgr
  export RMBENCH_LINGBOT_STEP_TRACE_PATH="${output}/step_trace.jsonl"
  export RMBENCH_LINGBOT_TASK_STATE_TRACE=1
  export RMBENCH_EVAL_STEP_LIMIT="${step_limit}"
  export RMBENCH_EVAL_STALL_ENABLE=0
  export WANDB_PROJECT="${WANDB_PROJECT:-lingbot-va-rmbench}"
  export WANDB_INFERENCE_RUN_NAME="${task}_${PROTOCOL}_${COUNT}ep_${MODE}_${STAMP}"
  export WANDB_MODE="${WANDB_MODE:-online}"
  export LD_LIBRARY_PATH="${CUDA_LIB}:${LD_LIBRARY_PATH:-}"
  export PYTORCH_ALLOC_CONF=expandable_segments:True

  cd "${REPO}"
  CUDA_VISIBLE_DEVICES=1,2 "${PYTHON}" -m torch.distributed.run \
    --nproc_per_node=2 --master_port="$((port + 100))" \
    -m wan_va.wan_va_server --config-name rmbench --port "${port}" \
    --save_root "${output}/server" >"${output}/server.log" 2>&1 &
  SERVER_PID=$!
  cleanup_server() {
    if [[ -n "${SERVER_PID:-}" ]]; then
      kill -TERM "${SERVER_PID}" 2>/dev/null || true
      wait "${SERVER_PID}" 2>/dev/null || true
      SERVER_PID=""
    fi
  }
  trap cleanup_server EXIT INT TERM

  local ready=0
  for attempt in $(seq 1 "${SERVER_START_TIMEOUT}"); do
    if awk -v suffix=":$(printf '%04X' "${port}")" '$2 ~ suffix "$" && $4 == "0A" {ok=1} END {exit(ok?0:1)}' /proc/net/tcp /proc/net/tcp6 2>/dev/null; then
      ready=1
      break
    fi
    kill -0 "${SERVER_PID}" 2>/dev/null || { tail -100 "${output}/server.log" >&2; return 1; }
    sleep 1
  done
  (( ready == 1 )) || { echo "Server did not become ready for ${task} within ${SERVER_START_TIMEOUT}s" >&2; return 1; }

  cd "${RMBENCH}"
  CUDA_VISIBLE_DEVICES=0 "${EVAL_PYTHON}" script/eval_policy.py \
    --config policy/LingBot_VA/deploy_policy.yml --overrides \
    --task_name "${task}" --task_config demo_clean --ckpt_setting "${PROTOCOL}" \
    --seed 0 --policy_name LingBot_VA --port "${port}" \
    --validated_seed_file "${seed_manifest}" --instruction_type seen \
    --eval_num_episodes "${COUNT}" --eval_output_dir "${output}/eval" \
    >"${output}/client.log" 2>&1

  cleanup_server
  trap - EXIT INT TERM
  test -s "${output}/eval/_result.txt"
  echo "${task}: ${output}/eval/_result.txt"
}

# Inference is sequential because each task uses all three GPUs.
if [[ "${PROTOCOL}" == generalize ]]; then
  BLOCKS_STEP_LIMIT=3500
else
  BLOCKS_STEP_LIMIT=1033
fi
run_inference observe_and_pickup "${OBS_CHECKPOINT}" \
  /root/wyc/data/rmbench/lingbot_va2/observe_and_pickup_50demo_fullctx_apf16_cf8 \
  episode_000000_0_145.pth 250 29571
run_inference blocks_ranking_try "${BLOCKS_CHECKPOINT}" \
  /root/wyc/data/rmbench/lingbot_va2/blocks_ranking_try_episode0_fullctx_apf16_cf8 \
  episode_000000_0_1033.pth "${BLOCKS_STEP_LIMIT}" 29572

echo "Both inference runs completed: ${OUTPUT_ROOT}"
