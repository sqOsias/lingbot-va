#!/usr/bin/env python3
"""Build independent 5-episode validation and test datasets for RMBench tasks."""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path


TASKS = ("observe_and_pickup", "rearrange_blocks", "blocks_ranking_try")
SPLITS = {"validation": 100000, "test": 200000}


def run(command, cwd=None, env=None):
    print("+", " ".join(map(str, command)), flush=True)
    subprocess.run([str(value) for value in command], cwd=cwd, env=env, check=True)


def instruction(train: Path) -> str:
    record = json.loads(next(line for line in (train / "meta/episodes.jsonl").read_text().splitlines() if line))
    return str(record["tasks"][0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("all", *TASKS), default="all")
    parser.add_argument("--split", choices=("all", *SPLITS), default="all")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--phase", choices=("all", "seeds", "collect", "convert", "latents", "audit"), default="all")
    parser.add_argument("--count", type=int, default=5, help="Number of episodes in each new holdout batch")
    parser.add_argument("--start-seed", type=int, help="Expert seed search start; default is split-specific")
    parser.add_argument("--batch-name", help="Unique suffix for a supplemental batch, e.g. extra01")
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count must be positive")
    if args.batch_name and not re.fullmatch(r"[a-zA-Z0-9_-]+", args.batch_name):
        parser.error("--batch-name may contain only letters, digits, underscores, and hyphens")
    if args.batch_name and args.start_seed is None:
        parser.error("supplemental batches require an explicit --start-seed")
    if args.start_seed is not None and args.split == "all":
        parser.error("an explicit --start-seed requires --split validation or --split test")
    if args.start_seed is not None and args.start_seed < 50:
        parser.error("holdout --start-seed must not overlap training seeds 0..49")
    repo = Path("/root/wyc/code/lingbot-va2")
    rmbench = Path("/root/wyc/code/RMBench")
    data_root = Path("/root/wyc/data/rmbench/lingbot_va2")
    lingbot_python = Path("/root/miniconda3/envs/lingbot/bin/python")
    rmbench_python = Path("/root/miniconda3/envs/RMBench/bin/python")
    tasks = TASKS if args.task == "all" else (args.task,)
    splits = SPLITS if args.split == "all" else {args.split: SPLITS[args.split]}
    phases = ("seeds", "collect", "convert", "latents", "audit") if args.phase == "all" else (args.phase,)

    for task in tasks:
        train = data_root / f"{task}_50demo_fullctx_apf16_cf8"
        if not train.is_dir():
            raise FileNotFoundError(train)
        task_instruction = instruction(train)
        for split, start_seed in splits.items():
            if args.start_seed is not None:
                start_seed = args.start_seed
            batch_label = split if not args.batch_name else f"{split}_{args.batch_name}"
            work = data_root / "holdout_work" / task / batch_label
            seed_manifest = work / "expert_seeds.json"
            raw = work / "raw"
            collection_manifest = work / "collection_manifest.json"
            lerobot = work / "lerobot"
            output = data_root / f"{task}_{args.count}demo_{batch_label}_fullctx_apf16_cf8"
            work.mkdir(parents=True, exist_ok=True)

            if "seeds" in phases:
                run([rmbench_python, rmbench / "tools/collect_expert_seeds.py",
                     "--task", task, "--count", str(args.count), "--start-seed", str(start_seed),
                     "--max-attempts", str(max(100, args.count * 50)), "--output", seed_manifest], cwd=rmbench)

            if "collect" in phases:
                seeds_payload = json.loads(seed_manifest.read_text())
                if not seeds_payload.get("complete"):
                    raise RuntimeError(f"expert seed search incomplete: {seed_manifest}")
                seeds = [int(item["seed"]) for item in seeds_payload["seeds"][:args.count]]
                if len(seeds) != args.count:
                    raise RuntimeError(f"expected {args.count} accepted seeds, found {len(seeds)}")
                collection = {
                    "schema_version": 1, "task": task, "split": split,
                    "batch_name": args.batch_name,
                    "seed_policy": "expert-validated; disjoint high seed range",
                    "seeds": seeds, "training_seed_range": [0, 49],
                    "episodes": [{"episode_id": i, "seed": seed, "expert_success": None}
                                 for i, seed in enumerate(seeds)],
                }
                collection_manifest.write_text(json.dumps(collection, indent=2) + "\n")
                run([rmbench_python, repo / "script/rmbench_data/collect_rmbench_holdout_experts.py",
                     "--rmbench-root", rmbench, "--output", raw, "--manifest", collection_manifest,
                     "--task", task, "--instruction", task_instruction,
                     "--seeds", *map(str, seeds)], cwd=rmbench)

            if "convert" in phases:
                run([lingbot_python, repo / "script/rmbench_data/convert_rmbench_to_lingbot_lerobot.py",
                     "--task-root", raw, "--output-root", lerobot, "--repo-id", str(lerobot),
                     "--task-name", task, "--instruction-split", "seen", "--fps", "15",
                     "--window-frames", "4097"], cwd=repo)
                shutil.copy2(train / "rmbench_action_stats.json", lerobot / "rmbench_action_stats.json")
                run([lingbot_python, repo / "script/rmbench_data/prepare_rmbench_apf16_dataset.py",
                     "--source", lerobot, "--output", output, "--raw-root", raw / "data",
                     "--regenerate-window-size", "4097"], cwd=repo)
                shutil.copy2(collection_manifest, output / "holdout_manifest.json")

            if "latents" in phases:
                run([lingbot_python, repo / "script/rmbench_data/extract_rmbench_latents.py",
                     "--dataset-root", output, "--model-root", repo / "checkpoints/base",
                     "--device", "cuda:0", "--frame-step", "4", "--chunk-frames", "8",
                     "--episode-start", "0", "--episode-end", str(args.count)], cwd=repo,
                    env={**__import__("os").environ, "CUDA_VISIBLE_DEVICES": str(args.gpu)})

            if "audit" in phases:
                run([lingbot_python, repo / "script/rmbench_data/audit_full_context_dataset.py",
                     "--dataset", output, "--expected-episodes", str(args.count)], cwd=repo)
                report = json.loads((output / "full_context_latent_audit.json").read_text())
                if report["status"] != "PASS":
                    raise RuntimeError(report)
            print(f"PASS task={task} split={split} output={output}")


if __name__ == "__main__":
    main()
