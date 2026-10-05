#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 2 || ! ${1:-} =~ ^[1-9][0-9]*$ ]]; then
  echo 'Usage: train_blocks_bounded.sh STEPS WINDOW_ROOT [OUTPUT]' >&2; exit 2
fi
STEPS=$1
WINDOW_ROOT=$(realpath "$2")
REPO=/root/wyc/code/lingbot-va2
PYTHON=/root/miniconda3/envs/lingbot/bin/python
BASE=${LINGBOT_PRETRAINED_MODEL:-${REPO}/checkpoints/base}
STAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT=${3:-${REPO}/train_out/blocks_bounded_${STEPS}step_${STAMP}}
MASTER_PORT=${MASTER_PORT:-29671}
CUDA_LIB=/root/miniconda3/envs/lingbot/targets/x86_64-linux/lib:/root/miniconda3/envs/lingbot/lib
test ! -e "$OUTPUT" || { echo "Output exists: $OUTPUT" >&2; exit 2; }
test -s "$BASE/transformer/diffusion_pytorch_model.safetensors.index.json"
test -s "$WINDOW_ROOT/audit.json"
test -s "$WINDOW_ROOT/train/index.jsonl"
test -s "$WINDOW_ROOT/validation/index.jsonl"
"$PYTHON" - "$WINDOW_ROOT" "$OUTPUT" "$STEPS" <<'PY'
import hashlib, json, os, pathlib, sys
root, output, steps = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), int(sys.argv[3])
manifest = json.loads((root / 'manifest.json').read_text())
audit = json.loads((root / 'audit.json').read_text())
assert manifest['task'] == 'blocks_ranking_try'
assert manifest['split_sources']['train']['episodes'] == 50
assert manifest['split_sources']['validation']['episodes'] == 5
assert audit['status'] == 'PASS', audit
assert audit['manifest_sha256'] == hashlib.sha256((root / 'manifest.json').read_bytes()).hexdigest()
if os.environ.get('BLOCKS_SMOKE_FORCE_EPISODE') and steps > 2:
    raise ValueError('BLOCKS_SMOKE_FORCE_EPISODE is only permitted for <=2-step smoke tests')
output.mkdir(parents=True)
record = dict(mode='bounded_v1', task='blocks_ranking_try', requested_steps=steps,
              window_root=str(root), manifest=manifest,
              validation_source=manifest['split_sources']['validation']['source'],
              train_source=manifest['split_sources']['train']['source'],
              max_latent_frames=manifest['max_latent_frames'],
              history_latent_frames=manifest['history_latent_frames'],
              smoke_force_episode=os.environ.get('BLOCKS_SMOKE_FORCE_EPISODE'))
(output / 'run_config.json').write_text(json.dumps(record, indent=2) + '\n')
PY
DATASET=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["split_sources"]["train"]["source"])' "$WINDOW_ROOT/manifest.json")
VALIDATION=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["split_sources"]["validation"]["source"])' "$WINDOW_ROOT/manifest.json")
cd "$REPO"
RMBENCH_DATASET_PATH="$DATASET" RMBENCH_VALIDATION_DATASET_PATH="$VALIDATION" \
RMBENCH_BOUNDED_TRAIN_INDEX="$WINDOW_ROOT/train/index.jsonl" \
RMBENCH_BOUNDED_VALIDATION_INDEX="$WINDOW_ROOT/validation/index.jsonl" \
RMBENCH_EARLY_STOP=1 LINGBOT_PRETRAINED_MODEL="$BASE" \
WANDB_PROJECT="${WANDB_PROJECT:-lingbot-va-rmbench}" \
WANDB_RUN_NAME="blocks_bounded_${STEPS}_${STAMP}" WANDB_MODE="${WANDB_MODE:-offline}" \
PYTHONHASHSEED=42 CUDA_VISIBLE_DEVICES=0,1,2 \
LD_LIBRARY_PATH="$CUDA_LIB:${LD_LIBRARY_PATH:-}" PYTORCH_ALLOC_CONF=expandable_segments:True \
"$PYTHON" -m torch.distributed.run --nproc_per_node=3 --master_port="$MASTER_PORT" \
  -m wan_va.train --config-name rmbench_train --save-root "$OUTPUT" \
  --num-steps "$STEPS" 2>&1 | tee "$OUTPUT/training.log"
test -s "$OUTPUT/training_summary.json"
STOP_STEP=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["stop_step"])' "$OUTPUT/training_summary.json")
test -s "$OUTPUT/checkpoints/checkpoint_step_${STOP_STEP}/transformer/diffusion_pytorch_model.safetensors"
echo "PASS: $OUTPUT/checkpoints/checkpoint_step_${STOP_STEP}"
