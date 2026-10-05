#!/usr/bin/env python3
"""Compute model-layout q01/q99 from episode-origin relative RMBench actions."""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial.transform import Rotation


USED_MODEL_CHANNELS = [0, 1, 2, 3, 4, 5, 6, 28, 7, 8, 9, 10, 11, 12, 13, 29]


def physical16(handle):
    parts = []
    for arm in ("left", "right"):
        pose = np.asarray(handle[f"endpose/{arm}_endpose"], dtype=np.float64)
        parts.append(np.concatenate([pose[:, :3], pose[:, 4:7], pose[:, 3:4]], axis=1))
        parts.append(np.asarray(handle[f"endpose/{arm}_gripper"], dtype=np.float64)[:, None])
    return np.concatenate(parts, axis=1)


def relative16(values, origin):
    out = np.empty_like(values)
    for offset in (0, 8):
        out[:, offset:offset + 3] = values[:, offset:offset + 3] - origin[offset:offset + 3]
        out[:, offset + 3:offset + 7] = (
            Rotation.from_quat(origin[offset + 3:offset + 7]).inv()
            * Rotation.from_quat(values[:, offset + 3:offset + 7])
        ).as_quat()
        out[:, offset + 7] = values[:, offset + 7]
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lower-quantile", type=float, default=0.01)
    parser.add_argument("--upper-quantile", type=float, default=0.99)
    args = parser.parse_args()

    records = [
        json.loads(line)
        for line in (args.dataset / "meta/episodes.jsonl").read_text().splitlines()
        if line.strip()
    ]
    samples = []
    for record in records:
        episode = int(record["episode_index"])
        with h5py.File(args.raw_root / f"episode{episode}.hdf5") as handle:
            absolute = physical16(handle)
        # Each physical action is included once; overlapping observation frames
        # do not duplicate labels when statistics are computed at episode scope.
        samples.append(relative16(absolute[1:int(record["length"]) + 1], absolute[0]))

    values = np.concatenate(samples)
    physical_q01 = np.quantile(values, args.lower_quantile, axis=0)
    physical_q99 = np.quantile(values, args.upper_quantile, axis=0)

        # Preserve the quantile interval center, but expand each channel enough
    # to keep every training action inside [-1.49, 1.49].
    normalization_limit = 1.49

    physical_min = values.min(axis=0)
    physical_max = values.max(axis=0)

    center = (physical_q01 + physical_q99) / 2.0
    half_range = (physical_q99 - physical_q01) / 2.0

    required_half_range = np.maximum(
        center - physical_min,
        physical_max - center,
    ) / normalization_limit

    half_range = np.maximum(half_range, required_half_range)

    # Avoid zero-width intervals on constant channels.
    half_range = np.maximum(half_range, 1e-6)

    physical_q01 = center - half_range
    physical_q99 = center + half_range



    q01 = np.zeros(30, dtype=np.float64)
    q99 = np.zeros(30, dtype=np.float64)
    q01[USED_MODEL_CHANNELS] = physical_q01
    q99[USED_MODEL_CHANNELS] = physical_q99
    result = {
        "method": (
            "episode-origin relative EEF XYZW; raw episode actions; "
            f"quantiles {args.lower_quantile:g}/{args.upper_quantile:g}; "
            "q01/q99-centered bounds expanded per channel so all training "
            "actions are within +/-1.49"
        ),
        "dataset_root": str(args.dataset),
        "samples": int(len(values)),
        "episodes": len(records),
        "q01": q01.tolist(),
        "q99": q99.tolist(),
        "used_action_channel_ids": USED_MODEL_CHANNELS,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": "PASS", "episodes": len(records), "samples": len(values)}))


if __name__ == "__main__":
    main()
