"""Verify that the opt-in production sidecar is a lossless extension.

This check instantiates the real ``MultiLatentLeRobotDataset`` twice: once
with the legacy configuration and once with the MemoryWAM alignment sidecar.
It compares every original model input tensor bit-for-bit and validates the
new temporal identity fields for every episode.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch

from wan_va.configs import VA_CONFIGS
from wan_va.dataset import MultiLatentLeRobotDataset


LEGACY_KEYS = ("latents", "actions", "actions_mask", "text_emb")


def _config(dataset_root: Path, alignment: Path | None):
    config = copy.deepcopy(VA_CONFIGS["rmbench_train"])
    config.dataset_path = str(dataset_root.resolve())
    config.empty_emb_path = str(dataset_root.resolve() / "empty_emb.pt")
    config.episode_indices = None
    config.cfg_prob = 0.0
    stats = json.loads(
        (dataset_root / "rmbench_action_stats.json").read_text())
    config.norm_stat = {key: stats[key] for key in ("q01", "q99")}
    config.memory_alignment_path = (str(alignment.resolve())
                                    if alignment is not None else None)
    return config


def verify(dataset_root: Path, alignment: Path) -> dict:
    legacy = MultiLatentLeRobotDataset(
        config=_config(dataset_root, None), num_init_worker=1)
    aligned = MultiLatentLeRobotDataset(
        config=_config(dataset_root, alignment), num_init_worker=1)
    if len(legacy) != len(aligned):
        raise ValueError(("dataset length changed", len(legacy), len(aligned)))

    episodes = []
    total_frames = total_valid_actions = 0
    for index in range(len(legacy)):
        old = legacy[index]
        new = aligned[index]
        if set(old) != set(LEGACY_KEYS):
            raise ValueError((index, "legacy keys changed", sorted(old)))
        for key in LEGACY_KEYS:
            torch.testing.assert_close(old[key], new[key], rtol=0, atol=0,
                                       msg=lambda message, key=key:
                                       f"sample {index} {key}: {message}")

        frames = old["latents"].shape[1]
        frame_ids = new["global_frame_ids"]
        expected = torch.arange(frames, dtype=frame_ids.dtype)
        if not torch.equal(frame_ids, expected):
            raise ValueError((index, "non-contiguous frame IDs",
                              frame_ids.tolist()))
        if new["video_target_mask"].shape != (frames,):
            raise ValueError((index, "video target mask shape"))
        expected_video_mask = torch.ones(frames, dtype=torch.bool)
        expected_video_mask[0] = False
        if not torch.equal(new["video_target_mask"], expected_video_mask):
            raise ValueError((index, "video target mask contents"))
        if len(new["decision_frame_ids"]) != frames - 1:
            raise ValueError((index, "decision count", frames))
        if not torch.equal(new["target_frame_ids"], frame_ids[1:]):
            raise ValueError((index, "target frame relation"))

        actual_action_slots = new["actions_mask"].any(dim=(0, 3)).sum(dim=1)
        if not torch.equal(actual_action_slots,
                           new["action_valid_slots_by_latent"]):
            raise ValueError((index, "action validity differs from sidecar"))
        if bool(actual_action_slots[0]):
            raise ValueError((index, "front dummy group is not empty"))
        if not bool(new["reset_mask"]):
            raise ValueError((index, "full episode must reset memory"))

        valid_actions = int(actual_action_slots.sum())
        total_frames += frames
        total_valid_actions += valid_actions
        episodes.append({
            "dataset_index": index,
            "dataset_numeric_id": int(new["dataset_numeric_id"]),
            "episode_index": int(new["episode_index"]),
            "latent_frames": frames,
            "valid_action_slots": valid_actions,
            "legacy_tensors_bitwise_equal": True,
        })

    return {
        "schema_version": 1,
        "kind": "production_memorywam_dataset_regression",
        "dataset_root": str(dataset_root.resolve()),
        "alignment": str(alignment.resolve()),
        "episodes_checked": len(episodes),
        "latent_frames_checked": total_frames,
        "valid_action_slots_checked": total_valid_actions,
        "legacy_tensors_bitwise_equal": True,
        "episodes": episodes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--alignment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = verify(args.dataset_root, args.alignment)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2,
                                     sort_keys=True) + "\n")
    print(json.dumps({key: report[key] for key in (
        "episodes_checked", "latent_frames_checked",
        "valid_action_slots_checked", "legacy_tensors_bitwise_equal")},
                     ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
