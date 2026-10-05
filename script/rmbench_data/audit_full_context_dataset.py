#!/usr/bin/env python3
"""Task-independent structural audit for a full-context RMBench latent dataset."""
import argparse
import json
from pathlib import Path

import torch

CAMERAS = (
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--expected-episodes", type=int, default=50)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.dataset.resolve()
    errors = []
    episodes_path = root / "meta/episodes.jsonl"
    episodes = [json.loads(line) for line in episodes_path.read_text().splitlines() if line.strip()]
    if len(episodes) != args.expected_episodes:
        errors.append(f"episode count {len(episodes)} != {args.expected_episodes}")
    expected_indices = list(range(args.expected_episodes))
    indices = [int(record["episode_index"]) for record in episodes]
    if indices != expected_indices:
        errors.append(f"episode indices are not contiguous 0..{args.expected_episodes - 1}")
    checked = 0
    for record in episodes:
        episode = int(record["episode_index"])
        windows = record.get("action_config", [])
        if len(windows) != 1:
            errors.append(f"episode {episode}: expected one window, found {len(windows)}")
            continue
        start, end = int(windows[0]["start_frame"]), int(windows[0]["end_frame"])
        if start != 0:
            errors.append(f"episode {episode}: start_frame={start}, expected 0")
        for camera in CAMERAS:
            path = root / "latents/chunk-000" / camera / f"episode_{episode:06d}_{start}_{end}.pth"
            if not path.is_file() or path.stat().st_size == 0:
                errors.append(f"missing latent: {path}")
                continue
            try:
                payload = torch.load(path, map_location="cpu", weights_only=False)
                frame_ids = [int(value) for value in payload["frame_ids"]]
                expected = list(range(start, end, 4))
                expected = expected[:1 + 4 * ((len(expected) - 1) // 4)]
                if frame_ids != expected:
                    errors.append(f"{path}: frame_ids do not follow APF16 trimming")
                latent_frames = int(payload["latent_num_frames"])
                if latent_frames != (len(frame_ids) - 1) // 4 + 1:
                    errors.append(f"{path}: latent_num_frames={latent_frames} inconsistent")
                checked += 1
            except Exception as exc:
                errors.append(f"{path}: cannot audit ({exc})")
    for required in ("rmbench_action_stats.json", "empty_emb.pt", "observation_anchors.json"):
        if not (root / required).is_file():
            errors.append(f"missing required file: {root / required}")
    report = {
        "status": "PASS" if not errors else "FAIL",
        "dataset": str(root),
        "expected_episodes": args.expected_episodes,
        "episodes": len(episodes),
        "latent_files_checked": checked,
        "expected_latent_files": args.expected_episodes * len(CAMERAS),
        "errors": errors,
    }
    output = args.output or root / "full_context_latent_audit.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if not errors else 1)


if __name__ == "__main__":
    main()
