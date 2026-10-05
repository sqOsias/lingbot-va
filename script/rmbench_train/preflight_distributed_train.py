#!/usr/bin/env python3
"""Run two disposable official Trainer steps on three GPUs."""

import json
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from wan_va.configs import VA_CONFIGS
from wan_va.distributed.util import init_distributed
from wan_va.train import Trainer
from wan_va.utils import init_logger


def local_copy(parameter):
    value = parameter.detach()
    if hasattr(value, "to_local"):
        value = value.to_local()
    return value.float().clone()


def main():
    init_logger()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 3:
        raise AssertionError(world_size)
    torch.manual_seed(42 + rank)
    init_distributed(world_size, local_rank, rank)
    config = VA_CONFIGS["rmbench_overfit_train"]
    config.rank, config.local_rank, config.world_size = rank, local_rank, world_size
    config.load_worker = 0
    trainer = Trainer(config)
    parameter = None
    for candidate in trainer.transformer.parameters():
        if local_copy(candidate).numel():
            parameter = candidate
            break
    if parameter is None:
        raise RuntimeError(f"rank {rank} has no non-empty local parameter shard")
    rows = []
    for optimizer_step in range(2):
        before = local_copy(parameter)
        batch = trainer._get_next_batch()
        losses = trainer._train_step(batch, 0)
        after = local_copy(parameter)
        delta = torch.tensor(float((after - before).abs().max()), device=trainer.device)
        dist.all_reduce(delta, op=dist.ReduceOp.MAX)
        row = {
            "optimizer_step": optimizer_step + 1,
            "lr_after_scheduler": trainer.lr_scheduler.get_last_lr()[0],
            "latent_loss": float(losses["latent_loss"]),
            "action_loss": float(losses["action_loss"]),
            "grad_norm": float(losses["total_norm"]),
            "parameter_delta_max": float(delta),
        }
        rows.append(row)
    finite = all(
        all(torch.isfinite(torch.tensor(row[key])) for key in ("latent_loss", "action_loss", "grad_norm"))
        for row in rows
    )
    passed = finite and rows[0]["lr_after_scheduler"] > 0 and rows[1]["parameter_delta_max"] > 0
    report = {
        "status": "PASS" if passed else "FAIL",
        "world_size": world_size,
        "base": config.wan22_pretrained_model_name_or_path,
        "dataset_episode_indices": list(config.episode_indices),
        "dataset_windows": len(trainer.train_loader.dataset),
        "official_warmup_note": "first optimizer call uses lr=0; second call verifies parameter update",
        "steps": rows,
        "checkpoint_written": False,
    }
    if rank == 0:
        output = Path(os.environ["PREFLIGHT_OUTPUT"])
        output.write_text(json.dumps(report, indent=2))
        print(json.dumps(report))
    dist.barrier()
    dist.destroy_process_group()
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
