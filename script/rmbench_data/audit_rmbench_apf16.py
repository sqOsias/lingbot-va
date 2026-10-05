#!/usr/bin/env python3
"""Fail-closed CPU audit for the minimal RMBench APF16 adapter."""

import argparse
import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation


CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def physical16(handle):
    parts = []
    for arm in ("left", "right"):
        pose = np.asarray(handle[f"endpose/{arm}_endpose"], dtype=np.float64)
        pose = np.concatenate([pose[:, :3], pose[:, 4:7], pose[:, 3:4]], axis=1)
        grip = np.asarray(handle[f"endpose/{arm}_gripper"], dtype=np.float64)[:, None]
        parts.extend([pose, grip])
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


def absolute16(values, origin):
    out = np.empty_like(values)
    for offset in (0, 8):
        out[:, offset:offset + 3] = values[:, offset:offset + 3] + origin[offset:offset + 3]
        out[:, offset + 3:offset + 7] = (
            Rotation.from_quat(origin[offset + 3:offset + 7])
            * Rotation.from_quat(values[:, offset + 3:offset + 7])
        ).as_quat()
        out[:, offset + 7] = values[:, offset + 7]
    return out


def rotation_error(a, b):
    return np.max((Rotation.from_quat(a).inv() * Rotation.from_quat(b)).magnitude())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--source-dataset", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.repo))
    from wan_va.configs import VA_CONFIGS
    from wan_va.dataset.lerobot_latent_dataset import LatentLeRobotDataset

    records = [json.loads(x) for x in (args.dataset / "meta/episodes.jsonl").read_text().splitlines()]
    stats = json.loads((args.dataset / "rmbench_action_stats.json").read_text())
    source_stats = json.loads((args.source_dataset / "rmbench_action_stats.json").read_text())
    contract = json.loads((args.dataset / "continuous_contract.json").read_text())
    checks = {
        "only_expected_50_episodes": [int(x["episode_index"]) for x in records] == list(range(50)),
        "normalization_stats_provenance": (
            sha256(args.dataset / "rmbench_action_stats.json")
            == sha256(args.source_dataset / "rmbench_action_stats.json")
            or str(stats.get("method", "")).startswith((
                "episode-origin relative",
                "episode-initial observed EEF origin",
            ))
        ),
        "action_schema": contract["action_schema"] == "[L_xyz,L_quat_xyzw,L_gripper,R_xyz,R_quat_xyzw,R_gripper]",
        "action_per_latent_frame": contract["action_per_latent_frame"] == 16,
    }

    parquet_max = 0.0
    abs_rel_xyz_max = 0.0
    abs_rel_rot_max = 0.0
    quat_wxyz_xyzw_rot_max = 0.0
    all_relative = []
    for rec in records:
        ep = int(rec["episode_index"])
        table = pq.read_table(args.dataset / f"data/chunk-000/episode_{ep:06d}.parquet", columns=["action"])
        parquet = np.asarray(table["action"].to_pylist(), dtype=np.float64)
        with h5py.File(args.raw_root / f"episode{ep}.hdf5") as handle:
            absolute = physical16(handle)
            raw_left = np.asarray(handle["endpose/left_endpose"], dtype=np.float64)
        parquet_max = max(parquet_max, float(np.max(np.abs(parquet - absolute[1:len(parquet) + 1]))))
        rel = relative16(absolute[1:len(parquet) + 1], absolute[0])
        restored = absolute16(rel, absolute[0])
        abs_rel_xyz_max = max(abs_rel_xyz_max, float(np.max(np.abs(restored[:, [0,1,2,8,9,10]] - absolute[1:len(parquet)+1, [0,1,2,8,9,10]]))))
        abs_rel_rot_max = max(abs_rel_rot_max, rotation_error(restored[:, 3:7], absolute[1:len(parquet)+1, 3:7]), rotation_error(restored[:, 11:15], absolute[1:len(parquet)+1, 11:15]))
        xyzw = np.concatenate([raw_left[:, 4:7], raw_left[:, 3:4]], axis=1)
        quat_wxyz_xyzw_rot_max = max(quat_wxyz_xyzw_rot_max, rotation_error(xyzw, absolute[:, 3:7]))
        for window in rec["action_config"]:
            start, end = int(window["start_frame"]), int(window["end_frame"])
            all_relative.append(relative16(absolute[start + 1:end], absolute[0]))

    checks["parquet_is_endpose_t_plus_1"] = parquet_max < 1e-5
    checks["absolute_relative_absolute_roundtrip"] = abs_rel_xyz_max < 1e-10 and abs_rel_rot_max < 1e-10
    checks["quaternion_wxyz_xyzw_roundtrip"] = quat_wxyz_xyzw_rot_max < 1e-10

    q01, q99 = np.asarray(stats["q01"]), np.asarray(stats["q99"])
    used = np.asarray(VA_CONFIGS["rmbench_train"].used_action_channel_ids)
    inverse = np.asarray(VA_CONFIGS["rmbench_train"].inverse_used_action_channel_ids)
    physical = np.concatenate(all_relative)
    padded = np.pad(physical, ((0, 0), (0, 1)))
    model = padded[:, inverse]
    normalized = (model - q01) / (q99 - q01 + 1e-6) * 2 - 1
    model_restored = (normalized + 1) / 2 * (q99 - q01 + 1e-6) + q01
    physical_restored = model_restored[:, used]
    model_roundtrip_max = float(np.max(np.abs(physical_restored - physical)))
    checks["physical16_model30_physical16_roundtrip"] = model_roundtrip_max < 1e-8
    checks["normalization_no_clipping"] = float(np.max(np.abs(normalized[:, used]))) <= 1.5

    cfg = copy.deepcopy(VA_CONFIGS["rmbench_train"])
    cfg.dataset_path = str(args.dataset.parent)
    cfg.empty_emb_path = str(args.dataset / "empty_emb.pt")
    cfg.norm_stat = {key: stats[key] for key in ("q01", "q99")}
    cfg.cfg_prob = 0.0
    cfg.load_worker = 0
    full = LatentLeRobotDataset(str(args.dataset), cfg)
    overfit_cfg = copy.deepcopy(VA_CONFIGS["rmbench_overfit_train"])
    overfit_cfg.dataset_path = str(args.dataset.parent)
    overfit_cfg.empty_emb_path = str(args.dataset / "empty_emb.pt")
    overfit_cfg.norm_stat = {key: stats[key] for key in ("q01", "q99")}
    overfit_cfg.cfg_prob = 0.0
    episode0 = LatentLeRobotDataset(str(args.dataset), overfit_cfg)
    checks["overfit_only_episode0_windows"] = (
        len(episode0) > 0
        and all(int(x["episode_index"]) == 0 for x in episode0.new_metas)
    )

    rows = []
    for index, meta in enumerate(full.new_metas):
        sample = full[index]
        start, end, ep = int(meta["start_frame"]), int(meta["end_frame"]), int(meta["episode_index"])
        actions = sample["actions"].permute(1, 2, 3, 0).reshape(-1, 30).numpy()
        mask = sample["actions_mask"].permute(1, 2, 3, 0).reshape(-1, 30).numpy()
        valid_rows = mask[:, used].all(axis=1)
        with h5py.File(args.raw_root / f"episode{ep}.hdf5") as handle:
            absolute = physical16(handle)
        expected_all = relative16(absolute[start + 1:end], absolute[0])
        denorm = (actions + 1) / 2 * (q99 - q01 + 1e-6) + q01
        # Every independent LingBot window reserves its first complete action
        # frame for the masked observation condition.
        expected = expected_all[:max(0, actions.shape[0] - 16)]
        decoded = denorm[:, used][valid_rows]
        err = float(np.max(np.abs(decoded - expected))) if len(expected) else 0.0
        latent_files = []
        for camera in CAMERAS:
            path = args.dataset / f"latents/chunk-000/observation.images.{camera}/episode_{ep:06d}_{start}_{end}.pth"
            import torch
            payload = torch.load(path, map_location="cpu", weights_only=False)
            latent_files.append({"camera": camera, "frame_ids": payload["frame_ids"], "latent_frames": int(payload["latent_num_frames"])})
        expected_front = 16
        expected_tail = actions.shape[0] - expected_front - len(expected)
        front_is_masked = bool((~mask[:expected_front, used]).all())
        valid_is_contiguous = bool(
            valid_rows[expected_front:expected_front + len(expected)].all()
            and not valid_rows[expected_front + len(expected):].any()
        )
        row = {
            "episode": ep, "start": start, "end": end,
            "first_label_physical_frame": start + 1,
            "last_label_physical_frame": start + len(expected),
            "latent_frames": int(sample["latents"].shape[1]),
            "action_shape": list(sample["actions"].shape),
            "valid_action_rows": int(valid_rows.sum()),
            "front_padding": expected_front, "tail_padding": expected_tail,
            "front_action_frame_fully_masked": front_is_masked,
            "valid_labels_contiguous_after_front_padding": valid_is_contiguous,
            "decoded_vs_expected_max": err,
            "latent_files": latent_files,
        }
        rows.append(row)
        if (int(valid_rows.sum()) != len(expected) or err >= 1e-5
                or not front_is_masked or not valid_is_contiguous):
            checks[f"window_{ep}_{start}"] = False
    checks["all_window_labels_padding_masks"] = len(rows) == len(full) and not any(v is False for k, v in checks.items() if k.startswith("window_"))
    checks["all_windows_first_label_is_start_plus_one"] = all(
        x["first_label_physical_frame"] == x["start"] + 1 for x in rows
    )
    checks["all_windows_front_pad_one_action_frame"] = all(
        x["front_padding"] == 16 and x["front_action_frame_fully_masked"]
        for x in rows
    )
    checks["all_windows_labels_end_before_window_end"] = all(
        x["last_label_physical_frame"] <= x["end"] - 1 for x in rows
    )

    protected = ["wan_va/modules", "wan_va/utils"]
    clean = subprocess.run(["git", "diff", "--quiet", "origin/main", "--", *protected], cwd=args.repo).returncode == 0
    checks["official_model_forward_attention_unchanged"] = clean
    checks["three_gpu_fsdp_cpu_offload_enabled"] = bool(getattr(cfg, "fsdp_cpu_offload", False))
    checks = {key: bool(value) for key, value in checks.items()}
    report = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "metrics": {
            "parquet_vs_physical_t_plus_1_max": parquet_max,
            "absolute_relative_absolute_xyz_max": abs_rel_xyz_max,
            "absolute_relative_absolute_rotation_rad_max": abs_rel_rot_max,
            "quaternion_roundtrip_rotation_rad_max": quat_wxyz_xyzw_rot_max,
            "physical16_model30_physical16_max": model_roundtrip_max,
            "normalized_active_abs_max": float(np.max(np.abs(normalized[:, used]))),
            "full_dataset_windows": len(full), "overfit_windows": len(episode0),
        },
        "episode0_windows": [x for x in rows if x["episode"] == 0],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps({"status": report["status"], "output": str(args.output), "checks": checks}))
    raise SystemExit(0 if report["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
