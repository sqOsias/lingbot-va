"""Sidecar-backed wrapper that adds explicit temporal identity to fullctx data.

This module is an isolated reference implementation.  Importing it does not
change the production dataset or trainer.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


def read_records(path: Path) -> dict[tuple[int, int, int], dict[str, Any]]:
    records = [json.loads(line) for line in path.read_text().splitlines()
               if line.strip()]
    result = {}
    for record in records:
        key = (int(record["episode_index"]),
               int(record["segment_start_frame"]),
               int(record["segment_end_frame"]))
        if key in result:
            raise ValueError(f"duplicate sidecar identity: {key}")
        result[key] = record
    return result


def attach_record_fields(sample: dict[str, Any],
                         record: dict[str, Any]) -> dict[str, Any]:
    result = dict(sample)
    decisions = record["decisions"]
    frames = int(record["latent_frames"])
    if sample["latents"].shape[1] != frames:
        raise ValueError(("sidecar/sample latent length", frames,
                          sample["latents"].shape[1]))
    result.update({
        "episode_id": record["episode_id"],
        "episode_index": torch.tensor(record["episode_index"],
                                      dtype=torch.long),
        "segment_index": torch.tensor(record["segment_index"],
                                      dtype=torch.long),
        "global_frame_ids": torch.tensor(record["global_frame_ids"],
                                         dtype=torch.long),
        "source_rgb_indices": torch.tensor(
            record["latent_source_rgb_indices"], dtype=torch.long),
        "decision_frame_ids": torch.tensor(
            [item["decision_global_frame_id"] for item in decisions],
            dtype=torch.long),
        "target_frame_ids": torch.tensor(
            [item["target_global_frame_id"] for item in decisions],
            dtype=torch.long),
        "target_action_group_ids": torch.tensor(
            [item["target_action_group_id"] for item in decisions],
            dtype=torch.long),
        "parquet_row_start": torch.tensor(
            [item["parquet_row_start"] if item["parquet_row_start"] is not None
             else -1 for item in decisions], dtype=torch.long),
        "parquet_row_end": torch.tensor(
            [item["parquet_row_end"] if item["parquet_row_end"] is not None
             else -1 for item in decisions], dtype=torch.long),
        "physical_target_start": torch.tensor(
            [item["physical_target_start"]
             if item["physical_target_start"] is not None else -1
             for item in decisions], dtype=torch.long),
        "physical_target_end": torch.tensor(
            [item["physical_target_end"]
             if item["physical_target_end"] is not None else -1
             for item in decisions], dtype=torch.long),
        "video_target_mask": torch.tensor(
            record["video_target_valid_by_latent"], dtype=torch.bool),
        "action_valid_slots_by_latent": torch.tensor(
            record["action_valid_slots_by_latent"], dtype=torch.long),
        # The existing fullctx item is one complete independent episode.
        "reset_mask": torch.tensor(True, dtype=torch.bool),
    })
    return result


class AlignedFullctxDataset(Dataset):
    """Wrap ``MultiLatentLeRobotDataset`` without modifying its tensors."""

    def __init__(self, base_dataset: Dataset, sidecar_path: Path) -> None:
        self.base = base_dataset
        self.records = read_records(sidecar_path)

    def __len__(self) -> int:
        return len(self.base)

    def _base_identity(self, index: int) -> tuple[int, int, int]:
        # MultiLatentLeRobotDataset exposes the mapping but not the identity in
        # each sample.  This coupling is why production should return identity
        # directly from LatentLeRobotDataset rather than keep this wrapper.
        dataset_id = self.base.item_id_to_dataset_id[index]
        local_index = index - self.base.acc_dset_num[dataset_id]
        inner = self.base._datasets[dataset_id]
        meta = inner.new_metas[local_index]
        return (int(meta["episode_index"]), int(meta["start_frame"]),
                int(meta["end_frame"]))

    def __getitem__(self, index: int) -> dict[str, Any]:
        key = self._base_identity(index)
        if key not in self.records:
            raise KeyError(f"dataset sample has no sidecar record: {key}")
        return attach_record_fields(self.base[index], self.records[key])
