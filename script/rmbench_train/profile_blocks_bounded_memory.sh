#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 1 || $# -gt 3 ]]; then
  echo 'Usage: profile_blocks_bounded_memory.sh WINDOW_ROOT [MAX_FRAMES] [HISTORY_FRAMES]' >&2; exit 2
fi
WINDOW_ROOT=$(realpath "$1")
MAX_FRAMES=${2:-64}
HISTORY_FRAMES=${3:-32}
PYTHON=/root/miniconda3/envs/lingbot/bin/python
"$PYTHON" - "$WINDOW_ROOT/manifest.json" "$MAX_FRAMES" "$HISTORY_FRAMES" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
assert m['max_latent_frames'] == int(sys.argv[2]), m
assert m['history_latent_frames'] == int(sys.argv[3]), m
PY
REPO=/root/wyc/code/lingbot-va2
OUTPUT="$REPO/train_out/blocks_bounded_profile_$(date +%Y%m%d_%H%M%S)"
BLOCKS_SMOKE_FORCE_EPISODE=20 RMBENCH_EARLY_STOP_INTERVAL=1 \
RMBENCH_EARLY_STOP_MIN_STEPS=3 WANDB_MODE=offline \
  bash "$REPO/script/rmbench_train/train_blocks_bounded.sh" 2 "$WINDOW_ROOT" "$OUTPUT"
"$PYTHON" - "$OUTPUT" "$MAX_FRAMES" <<'PY'
import json, pathlib, sys
root, maximum = pathlib.Path(sys.argv[1]), int(sys.argv[2])
rank_files = [root / f'bounded_rank_{r}.jsonl' for r in range(3)]
assert all(p.is_file() for p in rank_files), rank_files
records = [json.loads(line) for p in rank_files for line in p.read_text().splitlines()]
assert len(records) >= 6, records
forced = [r for r in records if r['episode_index'] == 20 and r['latent_frames'] == maximum]
assert forced, 'episode 20 maximum-length window did not complete a training step'
assert (root / 'validation_metrics.jsonl').is_file(), 'held-out validation did not run'
print('PROFILE PASS:', root)
print('forced long windows:', forced)
for rank in range(3):
    items = [r for r in records if r['rank'] == rank]
    print('rank', rank, 'max peak allocated GiB', max(x['peak_allocated_bytes'] for x in items) / 2**30,
          'max peak reserved GiB', max(x['peak_reserved_bytes'] for x in items) / 2**30)
PY
