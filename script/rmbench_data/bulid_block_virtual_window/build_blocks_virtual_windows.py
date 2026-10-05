#!/usr/bin/env python3
"""Index bounded slices of already-encoded, full-context RMBench episodes.

This never re-encodes video or rewrites source datasets. The target intervals
partition latent frames [1, F); frame zero is the initial condition.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch


CAMS = (
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def windows_for_episode(episode, frames, maximum, history):
    """Return exact, non-overlapping target coverage with overlapping context."""
    if frames < 2 or not 0 < history < maximum:
        raise ValueError((episode, frames, maximum, history))
    records = []
    cursor = 1
    while cursor < frames:
        start = max(0, cursor - history)
        end = min(frames, cursor + maximum - history)
        if end - start > maximum:
            raise AssertionError((start, end, maximum))
        records.append({
            "episode_index": episode,
            "global_latent_start": start,
            "global_latent_end": end,
            "target_latent_start": cursor,
            "target_latent_end": end,
            "history_latent_frames": cursor - start,
            "episode_latent_frames": frames,
            "max_latent_frames": maximum,
            "origin_start_frame": 0,
        })
        cursor = end
    return records


def build_split(root, count, maximum, history):
    episodes_path = root / "meta/episodes.jsonl"
    episodes = [json.loads(line) for line in episodes_path.read_text().splitlines() if line.strip()]
    if len(episodes) != count or [e["episode_index"] for e in episodes] != list(range(count)):
        raise ValueError(f"Unexpected episode indices in {root}: {len(episodes)}")
    audit = json.loads((root / "full_context_latent_audit.json").read_text())
    if audit.get("status") != "PASS" or audit.get("episodes") != count:
        raise ValueError(f"Source audit not PASS: {root}")
    anchors = json.loads((root / "observation_anchors.json").read_text())
    records = []
    files = {}
    for episode in episodes:
        ep = int(episode["episode_index"])
        configs = episode.get("action_config", [])
        if len(configs) != 1 or int(configs[0]["start_frame"]) != 0:
            raise ValueError(f"Expected one full-context sample: {root}, episode {ep}")
        end = int(configs[0]["end_frame"])
        anchor = anchors[f"{ep}:0"]
        if int(anchor["origin_start_frame"]) != 0:
            raise ValueError(f"Non-global action origin: {root}, episode {ep}")
        frame_ids = None
        frames = None
        for cam in CAMS:
            path = root / "latents/chunk-000" / cam / f"episode_{ep:06d}_0_{end}.pth"
            payload = torch.load(path, map_location="cpu", weights_only=False)
            these_ids = list(map(int, payload["frame_ids"]))
            these_frames = int(payload["latent_num_frames"])
            if frames is None:
                frames, frame_ids = these_frames, these_ids
            elif these_frames != frames or these_ids != frame_ids:
                raise ValueError(f"Camera latent alignment mismatch: {path}")
            files[str(path.relative_to(root))] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
        if not frame_ids or frame_ids[0] != 0 or any(b <= a for a, b in zip(frame_ids, frame_ids[1:])):
            raise ValueError(f"Non-monotonic source frame IDs: {root}, episode {ep}")
        for item in windows_for_episode(ep, frames, maximum, history):
            item["full_context_end_frame"] = end
            records.append(item)
    fingerprints = {}
    for name in ("meta/episodes.jsonl", "rmbench_action_stats.json", "observation_anchors.json", "continuous_contract.json", "full_context_latent_audit.json"):
        fingerprints[name] = sha256(root / name)
    return records, {"source": str(root.resolve()), "episodes": count, "source_sha256": fingerprints, "latent_files": files}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-source", type=Path, required=True)
    parser.add_argument("--validation-source", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-latent-frames", type=int, required=True)
    parser.add_argument("--history-latent-frames", type=int, required=True)
    args = parser.parse_args()
    if args.output_root.exists():
        parser.error(f"Output exists: {args.output_root}")
    if not 0 < args.history_latent_frames < args.max_latent_frames:
        parser.error("Require 0 < history < max latent frames")
    splits = {}
    manifests = {}
    for split, root, count in (("train", args.train_source, 50), ("validation", args.validation_source, 5)):
        splits[split], manifests[split] = build_split(root, count, args.max_latent_frames, args.history_latent_frames)
    args.output_root.mkdir(parents=True)
    for split, records in splits.items():
        directory = args.output_root / split
        directory.mkdir()
        (directory / "index.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    manifest = {
        "schema_version": 1,
        "task": "blocks_ranking_try",
        "max_latent_frames": args.max_latent_frames,
        "history_latent_frames": args.history_latent_frames,
        "split_sources": manifests,
        "window_count": {key: len(value) for key, value in splits.items()},
        "builder_sha256": sha256(Path(__file__)),
    }
    (args.output_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"status": "BUILT", "output": str(args.output_root), "window_count": manifest["window_count"]}))


if __name__ == "__main__":
    main()
