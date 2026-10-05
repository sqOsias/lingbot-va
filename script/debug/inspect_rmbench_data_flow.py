#!/usr/bin/env python3
import argparse
import copy
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from wan_va.configs import VA_CONFIGS
from wan_va.dataset.bounded_full_context_dataset import (
    BoundedFullContextDataset,
)


def describe(value, name="root", depth=0, max_depth=8):
    prefix = "  " * depth

    if isinstance(value, dict):
        print(f"{prefix}{name}: dict")
        for key, child in value.items():
            if depth < max_depth:
                describe(child, str(key), depth + 1, max_depth)
        return

    if isinstance(value, (list, tuple)):
        print(f"{prefix}{name}: {type(value).__name__}, len={len(value)}")
        if value and depth < max_depth:
            describe(value[0], "[0]", depth + 1, max_depth)
        return

    if torch.is_tensor(value):
        tensor = value.detach().cpu()
        finite = torch.isfinite(tensor) if tensor.is_floating_point() else None

        line = (
            f"{prefix}{name}: Tensor "
            f"shape={tuple(tensor.shape)} "
            f"dtype={tensor.dtype}"
        )

        if tensor.numel() and (
            tensor.is_floating_point()
            or tensor.dtype in (torch.int8, torch.int16, torch.int32, torch.int64)
        ):
            line += (
                f" min={tensor.min().item():.6g}"
                f" max={tensor.max().item():.6g}"
            )

        if finite is not None:
            line += f" finite={finite.all().item()}"

        print(line)
        return

    if isinstance(value, np.ndarray):
        print(
            f"{prefix}{name}: ndarray "
            f"shape={value.shape} dtype={value.dtype} "
            f"min={value.min():.6g} max={value.max():.6g}"
        )
        return

    print(f"{prefix}{name}: {type(value).__name__} = {value!r}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--window-root", required=True)
    parser.add_argument("--split", choices=["train", "validation"], default="train")
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()

    window_root = Path(args.window_root).resolve()
    manifest_path = window_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())

    split_info = manifest["split_sources"][args.split]
    dataset_path = Path(split_info["source"])
    index_path = window_root / args.split / "index.jsonl"

    config = copy.deepcopy(VA_CONFIGS["rmbench_train"])
    config.dataset_path = str(dataset_path)
    config.empty_emb_path = str(dataset_path / "empty_emb.pt")
    config.episode_indices = None
    config.cfg_prob = 0.0

    stats = json.loads(
        (dataset_path / "rmbench_action_stats.json").read_text()
    )
    config.norm_stat = {
        "q01": stats["q01"],
        "q99": stats["q99"],
    }

    print("=== CONFIGURATION ===")
    print("dataset_path:", config.dataset_path)
    print("index_path:", index_path)
    print("camera keys:", config.obs_cam_keys)
    print("action_per_frame:", config.action_per_frame)
    print("dataset_adapter:", config.dataset_adapter)
    print("max_latent_frames:", manifest["max_latent_frames"])
    print("history_latent_frames:", manifest["history_latent_frames"])

    dataset = BoundedFullContextDataset(config, index_path)

    print("\n=== DATASET ===")
    print("window count:", len(dataset))
    print("episode count:", len(dataset.by_episode))
    print("window counts:", dict(sorted(dataset.window_counts.items())))

    idx = args.index % len(dataset)
    record = dataset.records[idx]

    print("\n=== WINDOW RECORD ===")
    print(json.dumps(record, indent=2))

    sample = dataset[idx]

    print("\n=== DATASET ITEM ===")
    describe(sample, "sample")

    mask = sample["actions_mask"]
    valid_per_frame = mask.any(dim=(0, 2, 3))

    print("\n=== ACTION MASK ===")
    print("latent frames:", mask.shape[1])
    print("valid latent frames:", valid_per_frame.sum().item())
    print("masked latent frames:", (~valid_per_frame).sum().item())
    print("valid elements:", mask.sum().item())
    print("history_frame_count:", sample["history_frame_count"].item())
    print("global_latent_start:", sample["global_latent_start"].item())
    print("episode_index:", sample["episode_index"].item())

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )
    batch = next(iter(loader))

    print("\n=== DATALOADER BATCH ===")
    describe(batch, "batch")

    print("\n=== EXPECTED TRAINER INPUT ===")
    print("""
batch
├── latents:       [B, C_video, F, H, W]
├── actions:       [B, C_action, F, 16, 1]
├── actions_mask:  [B, C_action, F, 16, 1]
├── text_emb
├── global_latent_start
├── history_frame_count
├── episode_index
└── validation_weight

Trainer._prepare_input_dict(batch)
├── latent_dict
│   ├── timesteps
│   ├── noisy_latents
│   ├── targets
│   ├── latent
│   ├── cond_timesteps
│   ├── grid_id
│   └── text_emb
├── action_dict
│   ├── timesteps
│   ├── noisy_latents
│   ├── targets
│   ├── latent
│   ├── cond_timesteps
│   ├── grid_id
│   ├── text_emb
│   └── actions_mask
├── chunk_size
└── window_size
""")


if __name__ == "__main__":
    main()