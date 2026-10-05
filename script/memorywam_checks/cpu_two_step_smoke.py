"""CPU-only two-step smoke through the production MemoryWAM training path.

This deliberately uses a tiny Transformer but real Rearrange fullctx tensors,
alignment metadata, EEF30/APF16 actions, masks, Trainer noise construction and
loss functions.  It does not validate 5B FSDP, CUDA attention, VRAM or NCCL.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import resource
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
)
from torch.utils.data._utils.collate import default_collate

from wan_va.configs import VA_CONFIGS
from wan_va.dataset import MultiLatentLeRobotDataset
from wan_va.distributed.fsdp import apply_ac
from wan_va.modules.model import WanTransformer3DModel
from wan_va.train import Trainer
from wan_va.utils import FlowMatchScheduler


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _seed(value: int) -> None:
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)


def _real_batch(dataset_root: Path, alignment: Path) -> tuple[dict, dict]:
    config = copy.deepcopy(VA_CONFIGS["rmbench_train"])
    config.dataset_path = str(dataset_root)
    config.empty_emb_path = str(dataset_root / "empty_emb.pt")
    config.memory_alignment_path = str(alignment)
    config.episode_indices = [0]
    dataset = MultiLatentLeRobotDataset(config=config)
    if len(dataset) != 1:
        raise ValueError(f"Expected exactly episode 0, got {len(dataset)}")
    source = dataset[0]
    source_shapes = {
        key: list(value.shape) for key, value in source.items()
        if torch.is_tensor(value)
    }

    frames = 3
    batch = {
        "latents": source["latents"][:, :frames, :4, :4].clone(),
        "text_emb": source["text_emb"][:16].clone(),
        "actions": source["actions"][:, :frames].clone(),
        "actions_mask": source["actions_mask"][:, :frames].clone(),
        "episode_index": source["episode_index"].clone(),
        "segment_index": source["segment_index"].clone(),
        "global_frame_ids": source["global_frame_ids"][:frames].clone(),
        "source_rgb_indices": source["source_rgb_indices"][:frames].clone(),
        "decision_frame_ids": source["decision_frame_ids"][:frames - 1].clone(),
        "target_frame_ids": source["target_frame_ids"][:frames - 1].clone(),
        "target_action_group_ids": source[
            "target_action_group_ids"][:frames - 1].clone(),
        "video_target_mask": source["video_target_mask"][:frames].clone(),
        "action_valid_slots_by_latent": source[
            "action_valid_slots_by_latent"][:frames].clone(),
        "reset_mask": source["reset_mask"].clone(),
        "dataset_numeric_id": source["dataset_numeric_id"].clone(),
    }
    batch = default_collate([batch])
    if batch["latents"].shape != (1, 48, 3, 4, 4):
        raise ValueError(("cropped latent shape", batch["latents"].shape))
    if batch["actions"].shape != (1, 30, 3, 16, 1):
        raise ValueError(("cropped action shape", batch["actions"].shape))
    if bool(batch["actions_mask"][:, :, 0].any()):
        raise ValueError("The initial dummy action frame must be fully masked")
    if not bool(batch["actions_mask"][:, :, 1:].any()):
        raise ValueError("The cropped sample has no supervised action target")
    return batch, source_shapes


def _harness() -> Trainer:
    trainer = Trainer.__new__(Trainer)
    trainer.device = torch.device("cpu")
    trainer.patch_size = [1, 2, 2]
    trainer.gradient_accumulation_steps = 1
    trainer.config = SimpleNamespace(
        memorywam_enabled=True,
        memorywam_detach_interval=0,
    )
    trainer.train_scheduler_latent = FlowMatchScheduler(
        shift=5.0, sigma_min=0.0, extra_one_step=True)
    trainer.train_scheduler_latent.set_timesteps(1000, training=True)
    trainer.train_scheduler_action = FlowMatchScheduler(
        shift=5.0, sigma_min=0.0, extra_one_step=True)
    trainer.train_scheduler_action.set_timesteps(1000, training=True)
    return trainer


def _model() -> WanTransformer3DModel:
    model = WanTransformer3DModel(
        patch_size=[1, 2, 2],
        num_attention_heads=2,
        attention_head_dim=8,
        in_channels=48,
        out_channels=48,
        action_dim=30,
        text_dim=4096,
        freq_dim=8,
        ffn_dim=32,
        num_layers=2,
        cross_attn_norm=True,
        eps=1e-6,
        rope_max_seq_len=64,
        pos_embed_seq_len=None,
        attn_mode="torch",
        memorywam_enabled=True,
        memorywam_gist_tokens=2,
        memorywam_anchor_frames=1,
        memorywam_recent_frames=1,
        memorywam_mode="hybrid",
    ).to(torch.bfloat16)
    apply_ac(model)
    return model


def _clone_tree(value):
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: _clone_tree(item) for key, item in value.items()}
    return copy.deepcopy(value)


def _save_checkpoint(root: Path, model, optimizer, step: int) -> None:
    transformer = root / "transformer"
    transformer.mkdir(parents=True)
    state = get_model_state_dict(
        model,
        options=StateDictOptions(full_state_dict=True, cpu_offload=True),
    )
    state = {key: value.to(torch.bfloat16).contiguous()
             for key, value in state.items()}
    save_file(state, transformer / "diffusion_pytorch_model.safetensors")
    config = dict(model.config)
    config.pop("_name_or_path", None)
    (transformer / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n")
    torch.save({
        "step": step,
        "optimizer_state_dict": optimizer.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
    }, root / "training_state.pt")


def _maximum_difference(expected, actual) -> dict:
    differences = []
    for expected_tensor, actual_tensor in zip(expected, actual):
        differences.append(float(
            (expected_tensor.float() - actual_tensor.float()).abs().max()))
    return {"per_output": differences, "maximum": max(differences)}


def run(dataset_root: Path, alignment: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.mkdir(parents=True)
    started = time.perf_counter()
    _seed(20261004)
    batch, source_shapes = _real_batch(dataset_root, alignment)
    trainer = _harness()
    model = _model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    metrics = []
    last_inputs = None

    model.train()
    for step in range(2):
        _seed(20261004 + step)
        optimizer.zero_grad(set_to_none=True)
        inputs = trainer._prepare_input_dict(_clone_tree(batch))
        video, action = model(inputs, train_mode=True)
        video_loss, action_loss = trainer.compute_loss(
            inputs, (video, action))
        loss = video_loss + action_loss
        loss.backward()
        gradient = model.gist_tokens.grad
        if gradient is None:
            raise RuntimeError("gist_tokens did not receive a gradient")
        gist_grad_norm = float(gradient.float().norm())
        before = model.gist_tokens.detach().clone()
        optimizer.step()
        gist_update_max = float(
            (model.gist_tokens.detach() - before).abs().max())
        values = {
            "optimizer_step": step + 1,
            "video_loss": float(video_loss.detach()),
            "action_loss": float(action_loss.detach()),
            "total_loss": float(loss.detach()),
            "gist_grad_norm": gist_grad_norm,
            "gist_update_max": gist_update_max,
        }
        if not all(np.isfinite(value) for key, value in values.items()
                   if key != "optimizer_step"):
            raise ValueError(("non-finite metric", values))
        if gist_grad_norm <= 0 or gist_update_max <= 0:
            raise ValueError(("gist was not trained", values))
        metrics.append(values)
        last_inputs = _clone_tree(inputs)

    checkpoint = output / "checkpoint_step_2"
    _save_checkpoint(checkpoint, model, optimizer, step=2)

    model.eval()
    with torch.no_grad():
        expected = model(_clone_tree(last_inputs), train_mode=True)
        expected_repeat = model(_clone_tree(last_inputs), train_mode=True)

    restored = WanTransformer3DModel.from_pretrained(
        checkpoint / "transformer", torch_dtype=torch.bfloat16)
    apply_ac(restored)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)
    training_state = torch.load(
        checkpoint / "training_state.pt", map_location="cpu",
        weights_only=False)
    restored_optimizer.load_state_dict(training_state["optimizer_state_dict"])
    if training_state["step"] != 2 or not restored_optimizer.state:
        raise ValueError("optimizer/step state did not reload")
    restored.eval()
    saved_state = load_file(
        checkpoint / "transformer" / "diffusion_pytorch_model.safetensors")
    restored_state = get_model_state_dict(
        restored,
        options=StateDictOptions(full_state_dict=True, cpu_offload=True),
    )
    if saved_state.keys() != restored_state.keys():
        raise ValueError(("reloaded state keys differ", saved_state.keys(),
                          restored_state.keys()))
    restored_dtype_changes = []
    for key in saved_state:
        if restored_state[key].dtype != saved_state[key].dtype:
            restored_dtype_changes.append({
                "key": key,
                "serialized": str(saved_state[key].dtype),
                "restored": str(restored_state[key].dtype),
            })
        torch.testing.assert_close(
            restored_state[key].cpu().to(saved_state[key].dtype),
            saved_state[key], rtol=0, atol=0,
            msg=lambda message, key=key: f"state mismatch for {key}: {message}")
    with torch.no_grad():
        actual = restored(_clone_tree(last_inputs), train_mode=True)
        actual_repeat = restored(_clone_tree(last_inputs), train_mode=True)
    same_model_difference = _maximum_difference(expected, expected_repeat)
    restored_repeat_difference = _maximum_difference(actual, actual_repeat)
    reload_difference = _maximum_difference(expected, actual)
    reproducibility_tolerance = 2e-2
    if same_model_difference["maximum"] > reproducibility_tolerance:
        raise ValueError(("original model forward is unstable",
                          same_model_difference))
    if restored_repeat_difference["maximum"] > reproducibility_tolerance:
        raise ValueError(("restored model forward is unstable",
                          restored_repeat_difference))
    if reload_difference["maximum"] > reproducibility_tolerance:
        raise ValueError(("reload output difference exceeds BF16 tolerance",
                          reload_difference))

    report = {
        "schema_version": 1,
        "kind": "memorywam_cpu_two_step_smoke",
        "status": "PASS",
        "scope": {
            "validated": [
                "real fullctx Dataset and alignment load",
                "production MemoryWAM full-episode forward",
                "Trainer diffusion noise and video/action loss",
                "two backward and AdamW optimizer steps",
                "non-zero gist gradient and update",
                "model/optimizer/RNG checkpoint write",
                "model and optimizer reload",
                "exact checkpoint tensor reload",
                "post-reload forward within explicit BF16 absolute tolerance",
            ],
            "not_validated": [
                "5B weights",
                "FSDP/HSDP or distributed optimizer state",
                "CUDA attention kernels, NCCL, VRAM or GPU throughput",
                "online VAE or RMBench environment execution",
            ],
        },
        "dataset": {
            "root": str(dataset_root.resolve()),
            "alignment": str(alignment.resolve()),
            "alignment_sha256": _sha256(alignment),
            "episode_index": 0,
            "source_shapes": source_shapes,
            "smoke_shapes": {
                "latents": list(batch["latents"].shape),
                "text_emb": list(batch["text_emb"].shape),
                "actions": list(batch["actions"].shape),
                "actions_mask": list(batch["actions_mask"].shape),
            },
            "crop": "first 3 latent frames, 4x4 latent spatial crop, first 16 text tokens",
        },
        "model": {
            "layers": 2,
            "hidden_dim": 16,
            "parameters": sum(parameter.numel()
                              for parameter in model.parameters()),
            "dtype": "bfloat16",
            "memory_mode": "hybrid",
            "gist_tokens": 2,
            "anchor_frames": 1,
            "recent_frames": 1,
            "activation_checkpointing": True,
        },
        "metrics": metrics,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_serialized_values_reload_exact": True,
        "restored_dtype_changes": restored_dtype_changes,
        "forward_reproducibility": {
            "absolute_tolerance": reproducibility_tolerance,
            "same_model": same_model_difference,
            "restored_model": restored_repeat_difference,
            "before_vs_after_reload": reload_difference,
        },
        "optimizer_state_reload": True,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "smoke_script_sha256": _sha256(Path(__file__)),
    }
    (output / "cpu_2step_pass.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root", type=Path,
        default=Path("/root/wyc/data/rmbench/lingbot_va2/"
                     "rearrange_blocks_50demo_fullctx_apf16_cf8"))
    parser.add_argument(
        "--alignment", type=Path,
        default=Path("artifacts/memorywam/stage4_alignment/"
                     "rearrange_train/episodes.jsonl"))
    parser.add_argument(
        "--output", type=Path,
        default=Path("artifacts/memorywam/stage7_cpu_2step_pass"))
    args = parser.parse_args()
    report = run(args.dataset_root, args.alignment, args.output)
    print(json.dumps({
        "status": report["status"],
        "metrics": report["metrics"],
        "checkpoint": report["checkpoint"],
        "elapsed_seconds": report["elapsed_seconds"],
        "peak_rss_kib": report["peak_rss_kib"],
    }, indent=2))


if __name__ == "__main__":
    main()
