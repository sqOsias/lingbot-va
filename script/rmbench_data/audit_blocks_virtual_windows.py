#!/usr/bin/env python3
"""Audit bounded-window indices against the unmodified full-context tensors."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

# wan_va 和 script 包都在项目根目录下，直接运行脚本时不会自动加入 sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from easydict import EasyDict

from wan_va.dataset.bounded_full_context_dataset import BoundedFullContextDataset
def sha256(path):
    """Hash a source file without depending on the window builder location."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_records(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def audit_split(source, index_path, expected, source_manifest, train_source):
    errors = []
    if str(source.resolve()) != str(Path(source_manifest['source']).resolve()):
        errors.append(f"Source mismatch: {source} != {source_manifest['source']}")
    for name, digest in source_manifest['source_sha256'].items():
        if sha256(source / name) != digest:
            errors.append(f"Source fingerprint changed: {source / name}")
    for name, details in source_manifest['latent_files'].items():
        path = source / name
        if path.stat().st_size != details['bytes'] or sha256(path) != details['sha256']:
            errors.append(f"Latent source changed: {path}")
    config = EasyDict(
        dataset_path=str(source),
        empty_emb_path=str(source / "empty_emb.pt"),
        cfg_prob=0.0,
        obs_cam_keys=[
            "observation.images.cam_high",
            "observation.images.cam_left_wrist",
            "observation.images.cam_right_wrist",
        ],
        env_type="robotwin_tshape",
        action_per_frame=16,
        inverse_used_action_channel_ids=[0] * 30,
        dataset_adapter="rmbench",
        episode_indices=None,
    )
    # Use the same official 30-channel mapping and source train stats.
    from wan_va.configs.va_rmbench_train_cfg import va_rmbench_train_cfg
    config.inverse_used_action_channel_ids = va_rmbench_train_cfg.inverse_used_action_channel_ids
    # Validation must use the training normalizer, matching the Trainer.
    stats = json.loads((train_source / "rmbench_action_stats.json").read_text())
    config.norm_stat = {key: stats[key] for key in ("q01", "q99")}
    bounded = BoundedFullContextDataset(config, index_path)
    if len(bounded.by_episode) != expected:
        errors.append(f"episode count {len(bounded.by_episode)} != {expected}")
    missing = duplicate = checked = 0
    for ep in range(expected):
        full = bounded.full[ep]
        frames = full["latents"].shape[1]
        coverage = torch.zeros(frames, dtype=torch.int32)
        for idx in bounded.by_episode[ep]:
            record = bounded.records[idx]
            start = int(record["global_latent_start"])
            end = int(record["global_latent_end"])
            target = int(record["target_latent_start"])
            if not 0 <= start < target < end <= frames or end - start > int(record["max_latent_frames"]):
                errors.append(f"episode {ep}, window {idx}: invalid bounds")
                continue
            actual = bounded[idx]
            for key in ("latents", "actions", "actions_mask"):
                if not torch.equal(actual[key], full[key][:, start:end]):
                    errors.append(f"episode {ep}, window {idx}: {key} differs from full source")
            if int(actual["global_latent_start"]) != start or int(actual["history_frame_count"]) != target - start:
                errors.append(f"episode {ep}, window {idx}: offset/history mismatch")
            if not actual["actions_mask"][:, target - start:].any():
                errors.append(f"episode {ep}, window {idx}: no valid target actions")
            coverage[target:end] += 1
            checked += 1
        valid_frames = full['actions_mask'].bool().any(dim=(0, 2, 3))
        missing += int(((coverage[1:] == 0) & valid_frames[1:]).sum())
        duplicate += int(((coverage[1:] > 1) & valid_frames[1:]).sum())
        if coverage[0] != 0:
            errors.append(f"episode {ep}: initial condition frame supervised")
    return {
        "episodes": expected,
        "windows_checked": checked,
        "new_missing_target_slots": missing,
        "new_duplicate_target_slots": duplicate,
        "errors": errors,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--window-root", type=Path, required=True)
    parser.add_argument("--train-source", type=Path, required=True)
    parser.add_argument("--validation-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.window_root / "manifest.json").read_text())
    reports = {}
    for split, source, count in (("train", args.train_source, 50), ("validation", args.validation_source, 5)):
        reports[split] = audit_split(
            source, args.window_root / split / "index.jsonl", count,
            manifest["split_sources"][split], args.train_source,
        )
    report = {
        "manifest_sha256": sha256(args.window_root / "manifest.json"),
        "status": "PASS" if all(not r["errors"] and r["new_missing_target_slots"] == 0 and r["new_duplicate_target_slots"] == 0 for r in reports.values()) else "FAIL",
        "train_episodes": reports["train"]["episodes"],
        "validation_episodes": reports["validation"]["episodes"],
        "new_missing_target_slots": sum(r["new_missing_target_slots"] for r in reports.values()),
        "new_duplicate_target_slots": sum(r["new_duplicate_target_slots"] for r in reports.values()),
        "splits": reports,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
