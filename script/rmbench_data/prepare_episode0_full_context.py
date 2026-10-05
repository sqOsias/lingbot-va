#!/usr/bin/env python3
"""Create an isolated episode-0, one-window LeRobot view for full-context training."""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--end-frame", type=int, default=401)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    a.output.mkdir(parents=True)
    for name in ("data", "videos"):
        os.symlink((a.source / name).resolve(), a.output / name, target_is_directory=True)
    shutil.copytree(a.source / "meta", a.output / "meta")
    for name in ("rmbench_action_stats.json",):
        shutil.copy2(a.source / name, a.output / name)

    ep_path = a.output / "meta/episodes.jsonl"
    episodes = [json.loads(line) for line in ep_path.read_text().splitlines()]
    ep0 = next(x for x in episodes if int(x["episode_index"]) == 0)
    if int(ep0["length"]) != 404 or a.end_frame != 401:
        raise ValueError((ep0["length"], a.end_frame))
    ep0["action_config"] = [{
        "start_frame": 0,
        "end_frame": a.end_frame,
        "action_text": ep0["tasks"][0],
    }]
    ep_path.write_text("".join(json.dumps(x) + "\n" for x in episodes))

    anchors = json.loads((a.source / "observation_anchors.json").read_text())
    anchor = anchors["0:0"]
    anchor.update({
        "end_frame": a.end_frame,
        "valid_action_physical_frames": [1, 400],
        "valid_actions": 400,
        "latent_frames": 26,
        "action_slots": 416,
        "front_padding": 16,
        "tail_padding": 0,
        "sampled_video_frame_ids": list(range(0, 401, 4)),
    })
    (a.output / "observation_anchors.json").write_text(json.dumps({"0:0": anchor}, indent=2) + "\n")
    contract = {
        "status": "prepared_pending_latents",
        "episode_index": 0,
        "window": [0, a.end_frame],
        "source_physical_frames": 405,
        "lerobot_observation_frames": 404,
        "sampled_video_frame_stride_environment_actions": 4,
        "sampled_video_frames": 101,
        "latent_frames": 26,
        "action_per_latent_frame": 16,
        "action_slots": 416,
        "condition_actions_masked": 16,
        "valid_actions": 400,
        "relative_origin": "episode physical frame 0",
        "vae_streaming_chunk_frames": 8,
    }
    (a.output / "full_context_contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    print(json.dumps({"output": str(a.output), **contract}))


if __name__ == "__main__":
    main()
