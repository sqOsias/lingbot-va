#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: train_rmbench_50demo.sh TASK STEPS [OUTPUT]
TASK: observe_and_pickup | rearrange_blocks | blocks_ranking_try

All 50 training episodes participate in gradient updates. Early stopping uses a
separate five-episode validation dataset. A separate five-episode test dataset
is reserved for final evaluation and is never read by this launcher.
EOF
}

TASK=${1:-}
STEPS=${2:-}
case "${TASK}" in
  observe_and_pickup|rearrange_blocks|blocks_ranking_try) ;;
  *) usage >&2; exit 2 ;;
esac
[[ "${STEPS}" =~ ^[1-9][0-9]*$ ]] || { usage >&2; exit 2; }

REPO=/root/wyc/code/lingbot-va2
PYTHON=/root/miniconda3/envs/lingbot/bin/python
BASE=${LINGBOT_PRETRAINED_MODEL:-${REPO}/checkpoints/base}
DATASET=/root/wyc/data/rmbench/lingbot_va2/${TASK}_50demo_fullctx_apf16_cf8
VALIDATION=/root/wyc/data/rmbench/lingbot_va2/${TASK}_5demo_validation_fullctx_apf16_cf8
TEST=/root/wyc/data/rmbench/lingbot_va2/${TASK}_5demo_test_fullctx_apf16_cf8
STAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT=${3:-${REPO}/train_out/${TASK}_50demo_${STEPS}step_${STAMP}}
MASTER_PORT=${MASTER_PORT:-29671}
CUDA_LIB=/root/miniconda3/envs/lingbot/targets/x86_64-linux/lib:/root/miniconda3/envs/lingbot/lib

[[ ! -e "${OUTPUT}" ]] || { echo "Output already exists: ${OUTPUT}" >&2; exit 2; }
test -s "${BASE}/transformer/diffusion_pytorch_model.safetensors.index.json"
test -s "${DATASET}/rmbench_action_stats.json"
test -s "${DATASET}/empty_emb.pt"
test -s "${DATASET}/full_context_latent_audit.json"
"${PYTHON}" -c "import json; d=json.load(open('${DATASET}/full_context_latent_audit.json')); assert d['status']=='PASS', d"
for holdout in "${VALIDATION}" "${TEST}"; do
  test -s "${holdout}/full_context_latent_audit.json"
  "${PYTHON}" -c "import json; d=json.load(open('${holdout}/full_context_latent_audit.json')); assert d['status']=='PASS', d; assert d['episodes']==5, d"
done

mkdir -p "${OUTPUT}"
"${PYTHON}" - "${OUTPUT}/run_config.json" <<PY
import json, sys
json.dump({"task":"${TASK}", "dataset":"${DATASET}", "base":"${BASE}",
           "requested_steps":${STEPS}, "training_episodes":50,
           "validation_dataset":"${VALIDATION}", "test_dataset":"${TEST}",
           "early_stop":True,
           "config":"rmbench_train"}, open(sys.argv[1], "w"), indent=2)
PY

cd "${REPO}"
# Three FSDP ranks share 52 physical CPU cores.
# Override AutoDL's per-process value of 48 to avoid 3 x 48 oversubscription.
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
export NUMEXPR_NUM_THREADS=16
RMBENCH_DATASET_PATH="${DATASET}" \
RMBENCH_VALIDATION_DATASET_PATH="${VALIDATION}" \
RMBENCH_EARLY_STOP=1 \
LINGBOT_PRETRAINED_MODEL="${BASE}" \
WANDB_PROJECT="${WANDB_PROJECT:-lingbot-va-rmbench}" \
WANDB_RUN_NAME="${TASK}_50demo_${STEPS}step_${STAMP}" \
WANDB_MODE="${WANDB_MODE:-online}" \
PYTHONHASHSEED=42 CUDA_VISIBLE_DEVICES=0,1,2 \
LD_LIBRARY_PATH="${CUDA_LIB}:${LD_LIBRARY_PATH:-}" \
PYTORCH_ALLOC_CONF=expandable_segments:True \
"${PYTHON}" -m torch.distributed.run --nproc_per_node=3 --master_port="${MASTER_PORT}" \
  -m wan_va.train --config-name rmbench_train --save-root "${OUTPUT}" \
  --num-steps "${STEPS}" 2>&1 | tee "${OUTPUT}/training.log"

test -s "${OUTPUT}/training_summary.json"
STOP_STEP=$("${PYTHON}" -c "import json; print(json.load(open('${OUTPUT}/training_summary.json'))['stop_step'])")
test -s "${OUTPUT}/checkpoints/checkpoint_step_${STOP_STEP}/transformer/diffusion_pytorch_model.safetensors"
echo "PASS task=${TASK} requested_steps=${STEPS} stop_step=${STOP_STEP} output=${OUTPUT}"
