#!/usr/bin/env bash
set -euo pipefail

STEPS=${1:-1000}
OUTPUT=${2:-}
REPO=/root/wyc/code/lingbot-va2

[[ "$STEPS" =~ ^[1-9][0-9]*$ ]] || {
  echo 'Usage: run_observe_training.sh [STEPS] [OUTPUT]' >&2
  exit 2
}

cd "$REPO"
if [[ -n "$OUTPUT" ]]; then
  WANDB_MODE="${WANDB_MODE:-online}" \
    bash script/rmbench_train/train_rmbench_50demo.sh \
      observe_and_pickup "$STEPS" "$OUTPUT"
else
  WANDB_MODE="${WANDB_MODE:-online}" \
    bash script/rmbench_train/train_rmbench_50demo.sh \
      observe_and_pickup "$STEPS"
fi
