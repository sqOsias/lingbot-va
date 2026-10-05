"""CPU trace of the formal RMBench Dataset/DataLoader against a sidecar."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from wan_va.configs import VA_CONFIGS
from wan_va.dataset import MultiLatentLeRobotDataset

from .aligned_fullctx_dataset import AlignedFullctxDataset


def _read_sidecar(path: Path) -> dict[tuple[int, int, int], dict[str, Any]]:
    records = [json.loads(line) for line in path.read_text().splitlines()
               if line.strip()]
    return {(int(item["episode_index"]),
             int(item["segment_start_frame"]),
             int(item["segment_end_frame"])): item for item in records}


def _identity(dataset: MultiLatentLeRobotDataset,
              index: int) -> tuple[dict[str, Any], int, int]:
    dataset_id = dataset.item_id_to_dataset_id[index]
    local_index = index - dataset.acc_dset_num[dataset_id]
    inner = dataset._datasets[dataset_id]
    return inner.new_metas[local_index], dataset_id, local_index


def _expected_action_slots(record: dict[str, Any]) -> torch.Tensor:
    counts = [int(item) for item in record["action_valid_slots_by_latent"]]
    action_per_latent = int(record["action_per_latent"])
    expected = torch.zeros(len(counts), action_per_latent, dtype=torch.bool)
    for frame, count in enumerate(counts):
        expected[frame, :count] = True
    return expected


def trace(dataset_root: Path, sidecar_path: Path,
          indices: list[int]) -> dict[str, Any]:
    dataset_root = dataset_root.resolve()
    config = copy.deepcopy(VA_CONFIGS["rmbench_train"])
    config.dataset_path = str(dataset_root)
    config.empty_emb_path = str(dataset_root / "empty_emb.pt")
    config.episode_indices = None
    config.cfg_prob = 0.0
    stats = json.loads((dataset_root / "rmbench_action_stats.json").read_text())
    config.norm_stat = {"q01": stats["q01"], "q99": stats["q99"]}
    dataset = MultiLatentLeRobotDataset(config=config, num_init_worker=1)
    sidecar = _read_sidecar(sidecar_path)
    if not indices:
        indices = list(range(len(dataset)))
    resolved = []
    for value in indices:
        index = value if value >= 0 else len(dataset) + value
        if not 0 <= index < len(dataset):
            raise IndexError((value, index, len(dataset)))
        resolved.append(index)

    samples = []
    for index in resolved:
        meta, dataset_id, local_index = _identity(dataset, index)
        key = (int(meta["episode_index"]), int(meta["start_frame"]),
               int(meta["end_frame"]))
        if key not in sidecar:
            raise KeyError(f"formal dataset item has no sidecar identity: {key}")
        record = sidecar[key]
        sample = dataset[index]
        required = {"latents", "actions", "actions_mask", "text_emb"}
        if set(sample) != required:
            raise ValueError(("formal dataset keys changed", sorted(sample)))
        latents = sample["latents"]
        actions = sample["actions"]
        mask = sample["actions_mask"]
        frames = int(record["latent_frames"])
        action_per_latent = int(record["action_per_latent"])
        if latents.ndim != 4 or latents.shape[1] != frames:
            raise ValueError((index, "latent shape", tuple(latents.shape), frames))
        if actions.shape != mask.shape or actions.ndim != 4:
            raise ValueError((index, "action/mask shape", tuple(actions.shape),
                              tuple(mask.shape)))
        if actions.shape[1:3] != (frames, action_per_latent):
            raise ValueError((index, "action temporal shape", tuple(actions.shape)))
        actual_slots = mask.any(dim=(0, 3)).cpu()
        expected_slots = _expected_action_slots(record)
        if not torch.equal(actual_slots, expected_slots):
            mismatch = torch.nonzero(actual_slots != expected_slots).tolist()
            raise ValueError((index, "action mask differs from sidecar", mismatch[:20]))
        if not torch.isfinite(latents).all() or not torch.isfinite(actions).all():
            raise ValueError((index, "non-finite model inputs"))
        invalid = ~mask
        if torch.count_nonzero(actions.masked_select(invalid)):
            raise ValueError((index, "masked action values are not zero"))
        samples.append({
            "dataset_index": index,
            "inner_dataset_id": dataset_id,
            "inner_local_index": local_index,
            "episode_id": record["episode_id"],
            "episode_index": key[0],
            "segment": [key[1], key[2]],
            "global_frame_range": [record["global_frame_ids"][0],
                                   record["global_frame_ids"][-1]],
            "latent_shape": list(latents.shape),
            "action_shape": list(actions.shape),
            "action_valid_slots": int(actual_slots.sum()),
            "action_dummy_front_slots": int((~actual_slots[0]).sum()),
            "text_emb_shape": list(sample["text_emb"].shape),
            "finite": True,
            "masked_actions_zero": True,
        })

    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    aligned_dataset = AlignedFullctxDataset(dataset, sidecar_path)
    aligned_loader = DataLoader(aligned_dataset, batch_size=1, shuffle=False,
                                num_workers=0)
    aligned_batch = next(iter(aligned_loader))
    return {
        "schema_version": 1,
        "kind": "formal_fullctx_dataset_trace",
        "dataset_root": str(dataset_root),
        "dataset_length": len(dataset),
        "config": {
            "dataset_adapter": config.dataset_adapter,
            "camera_order": list(config.obs_cam_keys),
            "action_per_frame": int(config.action_per_frame),
            "cfg_prob_for_trace": float(config.cfg_prob),
        },
        "formal_dataset_returned_keys": sorted(batch),
        "dataloader_batch_shapes": {
            key: list(value.shape) for key, value in batch.items()
            if torch.is_tensor(value)
        },
        "identity_fields_present_in_formal_batch": {
            key: key in batch for key in (
                "episode_id", "episode_index", "global_frame_ids",
                "decision_frame_ids", "target_frame_ids",
                "target_action_group_ids", "reset_mask")
        },
        "aligned_batch_keys": sorted(aligned_batch),
        "aligned_batch_shapes": {
            key: list(value.shape) for key, value in aligned_batch.items()
            if torch.is_tensor(value)
        },
        "identity_fields_present_in_aligned_batch": {
            key: key in aligned_batch for key in (
                "episode_id", "episode_index", "global_frame_ids",
                "decision_frame_ids", "target_frame_ids",
                "target_action_group_ids", "reset_mask")
        },
        "trainer_currently_consumes": [
            "latents", "actions", "actions_mask", "text_emb"
        ],
        "samples": samples,
        "source_modified": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--sidecar", type=Path, required=True)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--index", type=int, action="append")
    selection.add_argument("--all", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    indices = [] if args.all else (args.index if args.index else [0, -1])
    report = trace(args.dataset_root, args.sidecar, indices)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2,
                                     sort_keys=True) + "\n")
    if args.quiet:
        print(json.dumps({
            "output": str(args.output),
            "dataset_length": report["dataset_length"],
            "samples_checked": len(report["samples"]),
            "all_finite": all(item["finite"] for item in report["samples"]),
            "all_masked_actions_zero": all(
                item["masked_actions_zero"] for item in report["samples"]),
            "valid_action_slots": sum(
                item["action_valid_slots"] for item in report["samples"]),
            "aligned_identity_fields": report[
                "identity_fields_present_in_aligned_batch"],
        }, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
