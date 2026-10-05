#!/usr/bin/env python3
"""Upload JSONL training metrics to W&B without touching the trainer process."""

import json
import os
import time
from pathlib import Path

import wandb


REPO = Path("/root/wyc/code/lingbot-va2")
PIPELINE_PID = REPO / "checkpoints/rmbench_two_trainings.pid"
PROJECT = os.environ.get("WANDB_PROJECT", "va_robotwin")
ENTITY = os.environ.get(
    "WANDB_ENTITY", "sqroasis-south-china-university-of-technology"
)
SYNC_ROOT = Path("/tmp/lingbot_wandb_sidecar")

STAGES = [
    {
        "name": "RMBench episode0 official LingBot overfit 1K",
        "id": "lingbot-rmbench-episode0-overfit-1k-20260914",
        "root": REPO / "checkpoints/rmbench_episode0_official_overfit_1k",
        "steps": 1000,
        "episodes": 1,
    },
    {
        "name": "RMBench 50-demo official LingBot training 3K",
        "id": "lingbot-rmbench-50demo-official-3k-20260914",
        "root": REPO / "checkpoints/rmbench_blocks_ranking_50demo_official_3k",
        "steps": 3000,
        "episodes": 50,
    },
]


def pipeline_alive():
    if not PIPELINE_PID.is_file():
        return False
    try:
        os.kill(int(PIPELINE_PID.read_text().strip()), 0)
        return True
    except (OSError, ValueError):
        return False


def atomic_json(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2))
    temporary.replace(path)


def upload_stage(stage):
    metrics = stage["root"] / "training_metrics.jsonl"
    state_path = stage["root"] / "wandb_sync_state.json"
    while not metrics.is_file():
        if not pipeline_alive():
            raise RuntimeError(f"pipeline stopped before {stage['name']} started")
        time.sleep(15)
    state = json.loads(state_path.read_text()) if state_path.is_file() else {"last_step": 0}
    SYNC_ROOT.mkdir(parents=True, exist_ok=True)
    run = wandb.init(
        entity=ENTITY,
        project=PROJECT,
        id=stage["id"],
        name=stage["name"],
        resume="allow",
        dir=str(SYNC_ROOT),
        config={
            "model": "LingBot-VA official base",
            "dataset": "RMBench blocks_ranking_try",
            "episodes": stage["episodes"],
            "optimizer_steps": stage["steps"],
            "world_size": 3,
            "batch_per_gpu": 1,
            "gradient_accumulation": 1,
            "learning_rate": 1e-5,
            "action_contract": "APF4 physical16 -> LingBot model30",
            "action_per_latent_frame": 16,
            "fsdp_cpu_offload": True,
        },
    )
    run.define_metric("optimizer_step")
    run.define_metric("loss/*", step_metric="optimizer_step")
    run.define_metric("train/*", step_metric="optimizer_step")
    try:
        while True:
            rows = [
                json.loads(line) for line in metrics.read_text().splitlines()
                if line.strip()
            ]
            for row in rows:
                step = int(row["optimizer_step"])
                if step <= int(state["last_step"]):
                    continue
                run.log({
                    "optimizer_step": step,
                    "loss/video": row["latent_loss"],
                    "loss/action": row["action_loss"],
                    "loss/max_video": row["max_latent_loss"],
                    "loss/max_action": row["max_action_loss"],
                    "train/grad_norm": row["grad_norm"],
                    "train/learning_rate": row["lr"],
                    "train/elapsed_seconds": row["elapsed_seconds"],
                })
                state = {"last_step": step, "wandb_run_id": stage["id"]}
                atomic_json(state_path, state)
            complete = (
                int(state["last_step"]) >= stage["steps"]
                and (stage["root"] / "loss_curve.svg").is_file()
            )
            if complete:
                run.summary["training_status"] = "complete"
                run.summary["final_optimizer_step"] = int(state["last_step"])
                return
            if not pipeline_alive():
                run.summary["training_status"] = "failed_or_interrupted"
                raise RuntimeError(f"pipeline stopped during {stage['name']}")
            time.sleep(15)
    finally:
        run.finish()


def main():
    for stage in STAGES:
        upload_stage(stage)


if __name__ == "__main__":
    main()
