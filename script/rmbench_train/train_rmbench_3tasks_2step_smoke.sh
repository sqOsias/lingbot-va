#!/usr/bin/env bash
set -euo pipefail

# Sequential smoke test launcher: runs 2-step train on all 3 RMBench tasks,
# one after another. Each task occupies the same 3 GPUs with distinct ports.

REPO=/root/wyc/code/lingbot-va2
PYTHON=/root/miniconda3/envs/lingbot/bin/python
BASE=${LINGBOT_PRETRAINED_MODEL:-${REPO}/checkpoints/base}
STEPS=2
STAMP=$(date +%Y%m%d_%H%M%S)
RUN_ROOT=${1:-${REPO}/train_out/rmbench_3tasks_2step_smoke_${STAMP}}
CUDA_LIB=/root/miniconda3/envs/lingbot/targets/x86_64-linux/lib:/root/miniconda3/envs/lingbot/lib

TASKS=(observe_and_pickup rearrange_blocks blocks_ranking_try)
PORTS=(29671 29672 29673)

[[ ! -e "${RUN_ROOT}" ]] || { echo "Output already exists: ${RUN_ROOT}" >&2; exit 2; }

# ---- Global preflight: base model + all 9 dataset directories ----
test -s "${BASE}/transformer/diffusion_pytorch_model.safetensors.index.json"

check_dataset() {
  local path=$1
  local expect_ep=$2
  test -s "${path}/rmbench_action_stats.json"
  test -s "${path}/empty_emb.pt"
  test -s "${path}/full_context_latent_audit.json"
  "${PYTHON}" -c "import json; d=json.load(open('${path}/full_context_latent_audit.json')); \
    assert d['status']=='PASS', ('${path}', d); \
    assert d['episodes']==${expect_ep}, ('${path}', d)"
}

for TASK in "${TASKS[@]}"; do
  check_dataset "/root/wyc/data/rmbench/lingbot_va2/${TASK}_50demo_fullctx_apf16_cf8" 50
  check_dataset "/root/wyc/data/rmbench/lingbot_va2/${TASK}_5demo_validation_fullctx_apf16_cf8" 5
  check_dataset "/root/wyc/data/rmbench/lingbot_va2/${TASK}_5demo_test_fullctx_apf16_cf8" 5
done

mkdir -p "${RUN_ROOT}"

# ---- Per-task training helper ----
run_training() {
  local task=$1
  local port=$2
  local output=${RUN_ROOT}/${task}
  local dataset="/root/wyc/data/rmbench/lingbot_va2/${task}_50demo_fullctx_apf16_cf8"
  local validation="/root/wyc/data/rmbench/lingbot_va2/${task}_5demo_validation_fullctx_apf16_cf8"
  local testset="/root/wyc/data/rmbench/lingbot_va2/${task}_5demo_test_fullctx_apf16_cf8"

  mkdir -p "${output}"
  "${PYTHON}" - "${output}/run_config.json" <<PY
import json, sys
json.dump({"task":"${task}", "dataset":"${dataset}", "base":"${BASE}",
           "requested_steps":${STEPS}, "training_episodes":50,
           "validation_dataset":"${validation}", "test_dataset":"${testset}",
           "early_stop":True, "config":"rmbench_train"},
          open(sys.argv[1], "w"), indent=2)
PY

  echo "[$(date --iso-8601=seconds)] === START ${task} @ port=${port} ==="
  cd "${REPO}"
  RMBENCH_DATASET_PATH="${dataset}" \
  RMBENCH_VALIDATION_DATASET_PATH="${validation}" \
  RMBENCH_EARLY_STOP=1 \
  LINGBOT_PRETRAINED_MODEL="${BASE}" \
  WANDB_PROJECT="${WANDB_PROJECT:-lingbot-va-rmbench}" \
  WANDB_RUN_NAME="${task}_50demo_${STEPS}step_smoke_${STAMP}" \
  WANDB_MODE="${WANDB_MODE:-online}" \
  PYTHONHASHSEED=42 \
  CUDA_VISIBLE_DEVICES=0,1,2 \
  LD_LIBRARY_PATH="${CUDA_LIB}:${LD_LIBRARY_PATH:-}" \
  PYTORCH_ALLOC_CONF=expandable_segments:True \
  "${PYTHON}" -m torch.distributed.run --nproc_per_node=3 --master_port="${port}" \
    -m wan_va.train --config-name rmbench_train --save-root "${output}" \
    --num-steps "${STEPS}" 2>&1 | tee "${output}/training.log"

  test -s "${output}/training_summary.json"
  local stop_step
  stop_step=$("${PYTHON}" -c "import json; print(json.load(open('${output}/training_summary.json'))['stop_step'])")
  test -s "${output}/checkpoints/checkpoint_step_${stop_step}/transformer/diffusion_pytorch_model.safetensors"
  echo "[$(date --iso-8601=seconds)] === PASS ${task} stop_step=${stop_step} ==="
}

# ---- Strictly sequential: next task starts only after previous exits successfully ----
for i in "${!TASKS[@]}"; do
  run_training "${TASKS[$i]}" "${PORTS[$i]}"
done

# ---- Final summary ----
"${PYTHON}" - "${RUN_ROOT}/completed.json" <<PY
import json, os
result = {"status": "PASS", "smoke_steps": ${STEPS}, "tasks": {}}
for t in ["observe_and_pickup", "rearrange_blocks", "blocks_ranking_try"]:
    summary = json.load(open(os.path.join("${RUN_ROOT}", t, "training_summary.json")))
    result["tasks"][t] = {"stop_step": summary["stop_step"],
                          "checkpoint": os.path.join("${RUN_ROOT}", t, "checkpoints",
                                                     f"checkpoint_step_{summary['stop_step']}")}
json.dump(result, open(sys.argv[1], "w"), indent=2)
print("ALL 3 TASKS PASSED")
PY
echo "All 3 smoke trainings complete: ${RUN_ROOT}/completed.json"