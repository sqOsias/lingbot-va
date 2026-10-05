"""小模型 delayed-recall 对照；不是 RMBench 结果。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from .memory_core import MemoryPolicy
from .toy_model import ToyMemoryModel


def _episode(model: ToyMemoryModel, label: float,
             frames: int) -> list[torch.Tensor]:
    values = [torch.zeros(model.video_tokens_per_frame, model.hidden_dim)
              for _ in range(frames)]
    values[0][0, 0] = label
    return values


def _train(policy: MemoryPolicy, *, seed: int, frames: int,
           steps: int) -> dict:
    torch.manual_seed(seed)
    model = ToyMemoryModel(policy=policy)
    readout = nn.Linear(model.hidden_dim, 1)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(readout.parameters()), lr=3e-3)
    losses = []
    for step in range(steps):
        loss = 0.0
        for offset, label in enumerate((-1.0, 1.0)):
            state, _ = model.prefill_streaming(
                _episode(model, label, frames),
                episode_id=f"train-{step}-{offset}", detach=False)
            action = model.denoise_action(
                torch.zeros(2, model.hidden_dim), state,
                request_id=f"train-{step}-{offset}", denoise_step=500)
            prediction = readout(action[-1]).squeeze()
            loss = loss + (prediction - label).square() / 2
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    with torch.no_grad():
        predictions = []
        for offset, label in enumerate((-1.0, 1.0)):
            state, _ = model.prefill_streaming(
                _episode(model, label, frames),
                episode_id=f"eval-{offset}", detach=True)
            action = model.denoise_action(
                torch.zeros(2, model.hidden_dim), state,
                request_id=f"eval-{offset}", denoise_step=500)
            predictions.append(float(readout(action[-1])))
    return {"initial_loss": losses[0], "final_loss": losses[-1],
            "predictions_for_minus_plus": predictions}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {
        "scope": "toy shared-backbone delayed recall; not RMBench",
        "gist": _train(MemoryPolicy.gist_only(2), seed=args.seed,
                       frames=args.frames, steps=args.steps),
        "recent_only": _train(MemoryPolicy.recent_only(1), seed=args.seed,
                              frames=args.frames, steps=args.steps),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
