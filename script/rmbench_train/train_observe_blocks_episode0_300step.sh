#!/usr/bin/env bash
set -euo pipefail

REPO=/root/wyc/code/lingbot-va2
PYTHON=/root/miniconda3/envs/lingbot/bin/python
BASE=${LINGBOT_PRETRAINED_MODEL:-${REPO}/checkpoints/base}
STEPS=300
STAMP=$(date +%Y%m%d_%H%M%S)
RUN_ROOT=${1:-${REPO}/checkpoints/observe_blocks_episode0_300step_${STAMP}}
CUDA_LIB=/root/miniconda3/envs/lingbot/targets/x86_64-linux/lib:/root/miniconda3/envs/lingbot/lib

OBS_DATASET=/root/wyc/data/rmbench/lingbot_va2/observe_and_pickup_50demo_fullctx_apf16_cf8
OBS_PREFLIGHT=${REPO}/artifacts/observe_and_pickup_episode0_preflight.json
BLOCKS_DATASET=/root/wyc/data/rmbench/lingbot_va2/blocks_ranking_try_episode0_fullctx_apf16_cf8
BLOCKS_PREFLIGHT=${REPO}/artifacts/blocks_ranking_try_episode0_fullctx_preflight.json

[[ ! -e "${RUN_ROOT}" ]] || { echo "Output already exists: ${RUN_ROOT}" >&2; exit 2; }
test -s "${BASE}/transformer/diffusion_pytorch_model.safetensors.index.json"

check_gate() {
  local report=$1
  local dataset=$2
  local prompt=$3
  test -s "${report}"
  test -s "${dataset}/rmbench_action_stats.json"
  test -s "${dataset}/empty_emb.pt"
  test -s "${dataset}/latents/chunk-000/observation.images.cam_high/${prompt}"
  "${PYTHON}" -c "import json; d=json.load(open('${report}')); assert d['status']=='PASS', d; assert d['dataset_episode_indices']==[0], d; assert d['dataset_windows']==1, d"
}

check_gate "${OBS_PREFLIGHT}" "${OBS_DATASET}" episode_000000_0_145.pth
check_gate "${BLOCKS_PREFLIGHT}" "${BLOCKS_DATASET}" episode_000000_0_1033.pth
mkdir -p "${RUN_ROOT}"

run_training() {
  local task=$1
  local dataset=$2
  local port=$3
  local output=${RUN_ROOT}/${task}
  mkdir -p "${output}"
  printf '%s\n' \
    "task=${task}" \
    "dataset=${dataset}" \
    "base=${BASE}" \
    "steps=${STEPS}" \
    "config=rmbench_overfit_train" \
    "episode_indices=0" > "${output}/run_config.txt"

  echo "[$(date --iso-8601=seconds)] Starting ${task}: ${output}"
  cd "${REPO}"
  RMBENCH_DATASET_PATH="${dataset}" \
  LINGBOT_PRETRAINED_MODEL="${BASE}" \
  WANDB_PROJECT="${WANDB_PROJECT:-lingbot-va-rmbench}" \
  WANDB_RUN_NAME="${task}_episode0_fullctx_300step_${STAMP}" \
  WANDB_MODE="${WANDB_MODE:-online}" \
  PYTHONHASHSEED=42 \
  CUDA_VISIBLE_DEVICES=0,1,2 \
  LD_LIBRARY_PATH="${CUDA_LIB}:${LD_LIBRARY_PATH:-}" \
  PYTORCH_ALLOC_CONF=expandable_segments:True \
  "${PYTHON}" -m torch.distributed.run \
    --nproc_per_node=3 --master_port="${port}" \
    -m wan_va.train --config-name rmbench_overfit_train \
    --save-root "${output}" --num-steps "${STEPS}" \
    2>&1 | tee "${output}/training.log"

  test -s "${output}/training_summary.json"
  test -s "${output}/checkpoints/checkpoint_step_${STEPS}/transformer/diffusion_pytorch_model.safetensors"
  echo "[$(date --iso-8601=seconds)] Completed ${task}"
}

# Strictly sequential: Blocks starts only after Observe exits successfully.
run_training observe_and_pickup "${OBS_DATASET}" 29661
run_training blocks_ranking_try "${BLOCKS_DATASET}" 29662

cat > "${RUN_ROOT}/completed.json" <<EOF
{
  "status": "PASS",
  "steps": ${STEPS},
  "observe_checkpoint": "${RUN_ROOT}/observe_and_pickup/checkpoints/checkpoint_step_${STEPS}",
  "blocks_checkpoint": "${RUN_ROOT}/blocks_ranking_try/checkpoints/checkpoint_step_${STEPS}"
}
EOF
echo "Both trainings completed: ${RUN_ROOT}/completed.json"
