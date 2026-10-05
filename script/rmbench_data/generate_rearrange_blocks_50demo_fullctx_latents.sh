#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  generate_rearrange_blocks_50demo_fullctx_latents.sh [all|rearrange_blocks|blocks_ranking_try] [GPU_ID]

Defaults: all, GPU_ID=0
The script is resumable: existing latent .pth files are skipped.
EOF
}

TARGET=${1:-all}
GPU_ID=${2:-0}
case "${TARGET}" in all|rearrange_blocks|blocks_ranking_try) ;; *) usage >&2; exit 2;; esac
[[ "${GPU_ID}" =~ ^[0-9]+$ ]] || { echo "GPU_ID must be a non-negative integer" >&2; exit 2; }

REPO=/root/wyc/code/lingbot-va2
PYTHON=/root/miniconda3/envs/lingbot/bin/python
MODEL=${REPO}/checkpoints/base
PREP=${REPO}/script/rmbench_data/prepare_rmbench_apf16_dataset.py
EXTRACT=${REPO}/script/rmbench_data/extract_rmbench_latents.py
WINDOW_SIZE=4097

test -s "${MODEL}/transformer/diffusion_pytorch_model.safetensors.index.json"
test -s "${PREP}"
test -s "${EXTRACT}"

validate_dataset() {
  local dataset=$1
  "${PYTHON}" - "${dataset}" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
episodes = [json.loads(line) for line in (root / "meta/episodes.jsonl").read_text().splitlines() if line]
assert len(episodes) == 50, len(episodes)
for episode in episodes:
    windows = episode.get("action_config", [])
    assert len(windows) == 1, (episode["episode_index"], windows)
    window = windows[0]
    assert int(window["start_frame"]) == 0, (episode["episode_index"], window)
    assert (int(window["end_frame"]) - 1) % 4 == 0, (episode["episode_index"], window)
print("DATASET_CONTRACT_PASS", root)
PY
}

validate_latents() {
  local dataset=$1
  "${PYTHON}" - "${dataset}" <<'PY'
import json, sys, torch
from pathlib import Path
root = Path(sys.argv[1])
episodes = [json.loads(line) for line in (root / "meta/episodes.jsonl").read_text().splitlines() if line]
expected = {}
for episode in episodes:
    window = episode["action_config"][0]
    expected[int(episode["episode_index"])] = (int(window["start_frame"]), int(window["end_frame"]))
cameras = ("observation.images.cam_high", "observation.images.cam_left_wrist", "observation.images.cam_right_wrist")
checked = 0
for episode, (start, end) in expected.items():
    for camera in cameras:
        path = root / "latents/chunk-000" / camera / f"episode_{episode:06d}_{start}_{end}.pth"
        assert path.is_file() and path.stat().st_size > 0, path
        payload = torch.load(path, map_location="cpu", weights_only=False)
        frame_ids = payload["frame_ids"]
        expected_frame_ids = list(range(start, end, 4))
        expected_frame_ids = expected_frame_ids[:1 + 4 * ((len(expected_frame_ids) - 1) // 4)]
        assert frame_ids == expected_frame_ids, (path, frame_ids[:3], frame_ids[-3:])
        assert int(payload["latent_num_frames"]) == (len(frame_ids) - 1) // 4 + 1, path
        checked += 1
manifest = json.loads((root / "latent_extraction_manifest.json").read_text())
assert manifest["frame_step"] == 4, manifest
assert manifest["chunk_frames"] == 8, manifest
assert manifest["segments"] == 50, manifest
assert len(manifest["files"]) == 150, len(manifest["files"])
report = {"status": "PASS", "episodes": 50, "segments": 50, "latent_files": checked, "frame_step": 4, "chunk_frames": 8}
(root / "full_context_latent_audit.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report))
PY
}

run_task() {
  local task=$1
  local source raw output
  case "${task}" in
    rearrange_blocks)
      source=/root/wyc/data/rmbench/lingbot_va2/rearrange_blocks_50demo_w193_apf16_official
      raw=/root/wyc/code/RMBench/data/data/rearrange_blocks/demo_clean/data
      output=/root/wyc/data/rmbench/lingbot_va2/rearrange_blocks_50demo_fullctx_apf16_cf8
      ;;
    blocks_ranking_try)
      source=/root/wyc/data/rmbench/lingbot_va2/blocks_ranking_try_50demo_w193_apf16_official
      raw=/root/wyc/code/RMBench/data/blocks_ranking_try/demo_clean/data
      output=/root/wyc/data/rmbench/lingbot_va2/blocks_ranking_try_50demo_fullctx_apf16_cf8
      ;;
  esac

  if [[ ! -e "${output}" ]]; then
    "${PYTHON}" "${PREP}" \
      --source "${source}" --output "${output}" --raw-root "${raw}" \
      --regenerate-window-size "${WINDOW_SIZE}"
  fi
  validate_dataset "${output}"
  mkdir -p "${output}/logs"
  echo "[$(date --iso-8601=seconds)] Extracting ${task} on physical GPU ${GPU_ID}" | tee -a "${output}/logs/full_context_latents.log"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON}" "${EXTRACT}" \
    --dataset-root "${output}" --model-root "${MODEL}" --device cuda:0 \
    --frame-step 4 --chunk-frames 8 --episode-start 0 --episode-end 50 \
    2>&1 | tee -a "${output}/logs/full_context_latents.log"
  validate_latents "${output}"
  echo "[$(date --iso-8601=seconds)] Completed ${task}: ${output}"
}

case "${TARGET}" in
  all)
    run_task rearrange_blocks
    run_task blocks_ranking_try
    ;;
  *) run_task "${TARGET}" ;;
esac
