"""Build a read-only MemoryWAM alignment sidecar for an RMBench fullctx set.

The source dataset is never modified.  ``continuous_contract.json`` is the
authority for valid physical actions; latent file headers are checked against
that contract for every configured camera.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch


DEFAULT_CAMERAS = (
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines()
            if line.strip()]


def _load_header(path: Path) -> dict[str, Any]:
    value = torch.load(path, map_location="cpu", weights_only=True)
    required = {
        "latent", "latent_num_frames", "latent_height", "latent_width",
        "video_num_frames", "text_emb", "text", "frame_ids",
        "start_frame", "end_frame",
    }
    missing = required - set(value)
    if missing:
        raise ValueError(f"{path}: missing latent keys {sorted(missing)}")
    latent = value["latent"]
    frames = int(value["latent_num_frames"])
    height = int(value["latent_height"])
    width = int(value["latent_width"])
    if latent.ndim != 2 or latent.shape[0] != frames * height * width:
        raise ValueError(
            f"{path}: latent shape {tuple(latent.shape)} is inconsistent "
            f"with {(frames, height, width)}")
    return {
        "latent_num_frames": frames,
        "latent_height": height,
        "latent_width": width,
        "latent_channels": int(latent.shape[1]),
        "video_num_frames": int(value["video_num_frames"]),
        "text_emb_shape": list(value["text_emb"].shape),
        "text": str(value["text"]),
        "frame_ids": [int(item) for item in value["frame_ids"]],
        "start_frame": int(value["start_frame"]),
        "end_frame": int(value["end_frame"]),
    }


def _decisions(window: dict[str, Any], *, action_per_latent: int,
               vae_temporal_compression: int) -> list[dict[str, Any]]:
    sampled = [int(item) for item in window["sampled_video_frame_ids"]]
    latent_frames = int(window["latent_frames"])
    latent_rgb = sampled[::vae_temporal_compression][:latent_frames]
    if len(latent_rgb) != latent_frames:
        raise ValueError(("latent/source RGB length", len(latent_rgb),
                          latent_frames))
    if any(value % action_per_latent for value in latent_rgb):
        raise ValueError(("non-integral global latent frame IDs", latent_rgb))
    valid_first, valid_last = map(int,
                                  window["valid_action_physical_frames"])
    result = []
    for local_id, (source_rgb, target_rgb) in enumerate(
            zip(latent_rgb, latent_rgb[1:])):
        physical_start = max(source_rgb + 1, valid_first)
        physical_end = min(target_rgb, valid_last)
        valid_count = max(0, physical_end - physical_start + 1)
        result.append({
            "decision_local_latent_id": local_id,
            "decision_global_frame_id": source_rgb // action_per_latent,
            "target_local_latent_id": local_id + 1,
            "target_global_frame_id": target_rgb // action_per_latent,
            "target_action_group_id": target_rgb // action_per_latent,
            "source_rgb_index": source_rgb,
            "target_rgb_index": target_rgb,
            "parquet_row_start": physical_start - 1 if valid_count else None,
            "parquet_row_end": physical_end if valid_count else None,
            "physical_target_start": physical_start if valid_count else None,
            "physical_target_end": physical_end if valid_count else None,
            "valid_action_count": valid_count,
            "video_target_valid": True,
        })
    if sum(item["valid_action_count"] for item in result) != int(
            window["valid_actions"]):
        raise ValueError(("valid action total", result,
                          window["valid_actions"]))
    return result


def build_sidecar(dataset_root: Path,
                  cameras: tuple[str, ...] = DEFAULT_CAMERAS
                  ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    dataset_root = dataset_root.resolve()
    contract_path = dataset_root / "continuous_contract.json"
    episodes_path = dataset_root / "meta/episodes.jsonl"
    contract = json.loads(contract_path.read_text())
    episodes = _read_jsonl(episodes_path)
    episode_meta = {int(item["episode_index"]): item for item in episodes}
    contracts = {int(item["episode_index"]): item
                 for item in contract["contracts"]}
    if set(episode_meta) != set(contracts):
        raise ValueError("episodes.jsonl and continuous_contract disagree")

    action_per_latent = int(contract["action_per_latent_frame"])
    vae_temporal_compression = int(contract["vae_temporal_compression"])
    records = []
    latent_shapes: set[tuple[int, int, int]] = set()
    for episode_index in sorted(episode_meta):
        episode = episode_meta[episode_index]
        episode_contract = contracts[episode_index]
        action_configs = episode["action_config"]
        windows = episode_contract["windows"]
        if len(action_configs) != len(windows):
            raise ValueError((episode_index, "segment count",
                              len(action_configs), len(windows)))
        for segment_index, (action_config, window) in enumerate(
                zip(action_configs, windows)):
            start = int(window["start"])
            end = int(window["end"])
            if (start, end) != (int(action_config["start_frame"]),
                                int(action_config["end_frame"])):
                raise ValueError((episode_index, segment_index,
                                  "action_config/window mismatch"))
            expected_frame_ids = [int(item) for item in
                                  window["sampled_video_frame_ids"]]
            camera_headers = {}
            for camera in cameras:
                latent_file = (dataset_root / "latents" / "chunk-000" /
                               camera /
                               f"episode_{episode_index:06d}_{start}_{end}.pth")
                if not latent_file.is_file():
                    raise FileNotFoundError(latent_file)
                header = _load_header(latent_file)
                if header["frame_ids"] != expected_frame_ids:
                    raise ValueError((latent_file, "frame IDs differ"))
                if header["latent_num_frames"] != int(window["latent_frames"]):
                    raise ValueError((latent_file, "latent frame count differs"))
                if (header["start_frame"], header["end_frame"]) != (start, end):
                    raise ValueError((latent_file, "segment bounds differ"))
                if header["text"] != action_config["action_text"]:
                    raise ValueError((latent_file, "text differs"))
                latent_shapes.add((header["latent_channels"],
                                   header["latent_height"],
                                   header["latent_width"]))
                camera_headers[camera] = header
            first_header = camera_headers[cameras[0]]
            for camera in cameras[1:]:
                for key in ("latent_num_frames", "video_num_frames",
                            "frame_ids", "text_emb_shape", "text"):
                    if camera_headers[camera][key] != first_header[key]:
                        raise ValueError((episode_index, camera, key,
                                          "camera headers disagree"))
            decisions = _decisions(
                window, action_per_latent=action_per_latent,
                vae_temporal_compression=vae_temporal_compression)
            valid_by_group = [0] + [item["valid_action_count"]
                                    for item in decisions]
            if len(valid_by_group) != int(window["latent_frames"]):
                raise ValueError((episode_index, "action/latent group length"))
            records.append({
                "schema_version": 1,
                "dataset_id": dataset_root.name,
                "episode_id": f"{dataset_root.name}:{episode_index}",
                "episode_index": episode_index,
                "segment_index": segment_index,
                "segment_start_frame": start,
                "segment_end_frame": end,
                "task": action_config["action_text"],
                "source_physical_frames": int(
                    episode_contract["source_physical_frames"]),
                "lerobot_observation_frames": int(
                    episode_contract["lerobot_observation_frames"]),
                "latent_frames": int(window["latent_frames"]),
                "global_frame_ids": [
                    item // action_per_latent
                    for item in expected_frame_ids[::vae_temporal_compression]
                    [:int(window["latent_frames"])]
                ],
                "latent_source_rgb_indices": expected_frame_ids[
                    ::vae_temporal_compression][:int(window["latent_frames"])],
                "action_per_latent": action_per_latent,
                "action_valid_slots_by_latent": valid_by_group,
                "video_target_valid_by_latent": [False] + [
                    True for _ in decisions],
                "front_dummy_action_slots": int(window["front_padding"]),
                "tail_padding_action_slots": int(window["tail_padding"]),
                "excluded_tail_physical_actions": int(
                    episode_contract["excluded_tail"]),
                "decisions": decisions,
                "camera_order": list(cameras),
                "camera_headers": camera_headers,
            })

    manifest = {
        "schema_version": 1,
        "kind": "rmbench_fullctx_memory_alignment",
        "dataset_root": str(dataset_root),
        "dataset_id": dataset_root.name,
        "task": contract["task"],
        "source_contract_sha256": _sha256(contract_path),
        "source_episodes_sha256": _sha256(episodes_path),
        "episodes": len(episode_meta),
        "segments": len(records),
        "decisions": sum(len(item["decisions"]) for item in records),
        "valid_actions": sum(sum(item["action_valid_slots_by_latent"])
                             for item in records),
        "camera_order": list(cameras),
        "latent_chw_shapes": [list(item) for item in sorted(latent_shapes)],
        "action_per_latent": action_per_latent,
        "rgb_stride": int(
            contract["sampled_video_frame_stride_environment_actions"]),
        "vae_temporal_compression": vae_temporal_compression,
        "relation": "observed z[t] -> action group (t,t+1] and target z[t+1]",
        "source_modified": False,
    }
    return manifest, records


def write_sidecar(dataset_root: Path, output: Path,
                  cameras: tuple[str, ...] = DEFAULT_CAMERAS) -> None:
    manifest, records = build_sidecar(dataset_root, cameras)
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2,
                   sort_keys=True) + "\n")
    with (output / "episodes.jsonl").open("w") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False,
                                    sort_keys=True) + "\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--camera", action="append")
    args = parser.parse_args()
    cameras = tuple(args.camera) if args.camera else DEFAULT_CAMERAS
    write_sidecar(args.dataset_root, args.output, cameras)


if __name__ == "__main__":
    main()
