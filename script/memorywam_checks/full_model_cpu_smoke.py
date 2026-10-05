"""Staged CPU validation for the complete LingBot-VA 5B Transformer.

The default ``plan`` stage performs no weight loading.  Expensive stages are
explicit and write progress events so an interrupted run remains diagnosable.
No stage initializes CUDA or a distributed process group.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import os
import random
import resource
import shutil
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
)

from script.memorywam_checks.cpu_two_step_smoke import (
    _clone_tree,
    _harness,
    _real_batch,
)
from wan_va.distributed.fsdp import apply_ac
from wan_va.modules.memorywam import commit_memory
from wan_va.modules.model import WanTransformer3DModel
from wan_va.utils import get_mesh_id


STAGE_MEMORY_GIB = {
    "plan": 0,
    "load": 24,
    "forward": 48,
    "train-gist": 96,
    "train-all": 256,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _available_memory_gib() -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1024**2
    raise RuntimeError("/proc/meminfo has no MemAvailable")


def _rss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2


def _event(path: Path, name: str, **values) -> None:
    record = {"event": name, "unix_time": time.time(),
              "elapsed_seconds": time.perf_counter() - _event.started,
              "peak_rss_gib": _rss_gib(), **values}
    with path.open("a") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(record, sort_keys=True), flush=True)


_event.started = time.perf_counter()


def _checkpoint_header(root: Path) -> dict:
    config_path = root / "config.json"
    weight_path = root / "diffusion_pytorch_model.safetensors"
    if not config_path.is_file() or not weight_path.is_file():
        raise FileNotFoundError(f"Incomplete transformer checkpoint: {root}")
    with safe_open(weight_path, framework="pt", device="cpu") as stream:
        shapes = {key: list(stream.get_slice(key).get_shape())
                  for key in stream.keys()}
        dtypes = {key: str(stream.get_slice(key).get_dtype())
                  for key in stream.keys()}
    return {
        "root": str(root.resolve()),
        "config_sha256": _sha256(config_path),
        # Do not hash the complete 10.18 GB file in the plan stage. The
        # consolidated CPU preflight already binds its config hash, byte size
        # and all key shapes; the full loader below is the content check.
        "weights_sha256": None,
        "weight_mtime_ns": weight_path.stat().st_mtime_ns,
        "weight_bytes": weight_path.stat().st_size,
        "keys": len(shapes),
        "shapes": shapes,
        "dtype_counts": {
            dtype: sum(value == dtype for value in dtypes.values())
            for dtype in sorted(set(dtypes.values()))
        },
    }


def _load_model(root: Path, header: dict, events: Path):
    _event(events, "full_model_load_started", checkpoint=str(root))
    started = time.perf_counter()
    model = WanTransformer3DModel.from_pretrained(
        root,
        torch_dtype=torch.bfloat16,
        attn_mode="torch",
        low_cpu_mem_usage=True,
    )
    state = model.state_dict()
    expected = header["shapes"]
    missing = sorted(set(expected) - set(state))
    unexpected = sorted(set(state) - set(expected))
    mismatch = {
        key: {"checkpoint": expected[key], "model": list(state[key].shape)}
        for key in set(expected) & set(state)
        if expected[key] != list(state[key].shape)
    }
    if missing or unexpected or mismatch:
        raise ValueError({"missing": missing, "unexpected": unexpected,
                          "shape_mismatch": mismatch})
    legacy_parameters = sum(parameter.numel()
                            for parameter in model.parameters())
    model.enable_memorywam(
        gist_tokens=8, anchor_frames=2, recent_frames=4, mode="hybrid")
    if model.gist_tokens.shape != (8, 3072):
        raise ValueError(("gist shape", model.gist_tokens.shape))
    result = {
        "load_seconds": time.perf_counter() - started,
        "legacy_state_keys": len(expected),
        "missing": missing,
        "unexpected": unexpected,
        "shape_mismatch": mismatch,
        "legacy_parameters": legacy_parameters,
        "memorywam_parameters": sum(parameter.numel()
                                    for parameter in model.parameters()),
        "gist_shape": list(model.gist_tokens.shape),
        "model_dtype_counts": {
            str(dtype): sum(parameter.dtype == dtype
                            for parameter in model.parameters())
            for dtype in sorted({parameter.dtype
                                 for parameter in model.parameters()},
                                key=str)
        },
        "attention_mode": "torch (CPU override; checkpoint config is flex)",
    }
    _event(events, "full_model_load_passed", **result)
    return model, result


def _load_real_batch(dataset_root: Path, alignment: Path, events: Path):
    _event(events, "dataset_load_started", dataset=str(dataset_root))
    batch, source_shapes = _real_batch(dataset_root, alignment)
    _event(events, "dataset_load_passed", source_shapes=source_shapes)
    return batch, source_shapes


def _forward_smoke(model, batch: dict, events: Path) -> dict:
    model.eval()
    latent = batch["latents"][:, :, :1, :2, :2].to(torch.bfloat16)
    text = batch["text_emb"][:, :1].to(torch.bfloat16)
    video_grid = get_mesh_id(1, 1, 1, t=0, f_w=1)[None]
    state = model.create_memorywam_state(
        "full-5b-cpu-forward", 1, device=torch.device("cpu"),
        dtype=torch.bfloat16)
    before = (state.version, state.next_frame_id,
              tuple(layer.key.shape[1] for layer in state.layers))
    _event(events, "full_model_prefill_started",
           latent_shape=list(latent.shape), text_shape=list(text.shape))
    started = time.perf_counter()
    with torch.no_grad():
        hidden, delta = model.memorywam_prefill_observation(
            {
                "noisy_latents": latent,
                "timesteps": torch.zeros(1, 1),
                "grid_id": video_grid,
                "text_emb": text,
            }, state, frame_id=0, observation_id="obs-0")
        state = commit_memory(
            state, delta, model.memorywam_policy, detach=True)
    prefill_seconds = time.perf_counter() - started
    if state.version != 1 or state.next_frame_id != 1:
        raise ValueError(("prefill did not commit once", before, state))
    if not bool(torch.isfinite(hidden).all()):
        raise ValueError("prefill produced non-finite hidden states")
    _event(events, "full_model_prefill_passed",
           seconds=prefill_seconds,
           hidden_shape=list(hidden.shape),
           memory_tokens_per_layer=state.layers[0].key.shape[1])

    action = batch["actions"][:, :, 1:2, :1].to(torch.bfloat16)
    action = action + torch.randn_like(action) * 0.1
    action_grid = get_mesh_id(1, 1, 1, t=1, f_w=1,
                              action=True)[None]
    query = {
        "noisy_latents": action,
        "timesteps": torch.full((1, 1), 500),
        "grid_id": action_grid,
        "text_emb": text,
    }
    identity = (state.version, state.next_frame_id,
                tuple(layer.key.shape for layer in state.layers))
    _event(events, "full_model_denoise_started",
           action_shape=list(action.shape))
    started = time.perf_counter()
    with torch.no_grad():
        first = model.memorywam_denoise(query, state, action_mode=True)
        second = model.memorywam_denoise(query, state, action_mode=True)
    denoise_seconds = time.perf_counter() - started
    after = (state.version, state.next_frame_id,
             tuple(layer.key.shape for layer in state.layers))
    if identity != after:
        raise ValueError("denoise mutated persistent state")
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    if not bool(torch.isfinite(first).all()):
        raise ValueError("denoise produced non-finite actions")
    result = {
        "prefill_seconds": prefill_seconds,
        "denoise_two_calls_seconds": denoise_seconds,
        "prefill_hidden_shape": list(hidden.shape),
        "action_output_shape": list(first.shape),
        "memory_tokens_per_layer": state.layers[0].key.shape[1],
        "state_after_commit": {"version": state.version,
                               "next_frame_id": state.next_frame_id},
        "denoise_state_unchanged": True,
        "repeated_denoise_exact": True,
    }
    _event(events, "full_model_forward_passed", **result)
    return result


def _training_batch(batch: dict) -> dict:
    result = _clone_tree(batch)
    result["latents"] = result["latents"][:, :, :2, :2, :2]
    result["text_emb"] = result["text_emb"][:, :1]
    result["actions"] = result["actions"][:, :, :2, :1]
    result["actions_mask"] = result["actions_mask"][:, :, :2, :1]
    for key in ("global_frame_ids", "source_rgb_indices",
                "video_target_mask", "action_valid_slots_by_latent"):
        result[key] = result[key][:, :2]
    for key in ("decision_frame_ids", "target_frame_ids",
                "target_action_group_ids"):
        result[key] = result[key][:, :1]
    if not bool(result["actions_mask"][:, :, 1].any()):
        raise ValueError("The one-slot training crop has no valid action")
    return result


def _train_smoke(model, batch: dict, events: Path, *, scope: str,
                 steps: int, learning_rate: float) -> tuple[dict, object]:
    if scope == "gist":
        model.requires_grad_(False)
        model.gist_tokens.requires_grad_(True)
    elif scope == "all":
        model.requires_grad_(True)
    else:
        raise ValueError(scope)
    apply_ac(model)
    trainable = [parameter for parameter in model.parameters()
                 if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=learning_rate, betas=(0.9, 0.95),
        weight_decay=0.1, foreach=False, fused=False)
    trainer = _harness()
    cropped = _training_batch(batch)
    metrics = []
    model.train()
    _event(events, "full_model_training_started", scope=scope, steps=steps,
           learning_rate=learning_rate,
           trainable_parameters=sum(value.numel() for value in trainable),
           latent_shape=list(cropped["latents"].shape),
           action_shape=list(cropped["actions"].shape),
           note="CPU crop uses one action slot; production data remains APF16")
    for step in range(steps):
        random.seed(20261004 + step)
        np.random.seed(20261004 + step)
        torch.manual_seed(20261004 + step)
        optimizer.zero_grad(set_to_none=True)
        inputs = trainer._prepare_input_dict(_clone_tree(cropped))
        started = time.perf_counter()
        prediction = model(inputs, train_mode=True)
        video_loss, action_loss = trainer.compute_loss(inputs, prediction)
        loss = video_loss + action_loss
        loss.backward()
        if model.gist_tokens.grad is None:
            raise RuntimeError("full model gist gradient is missing")
        gist_grad_norm = float(model.gist_tokens.grad.float().norm())
        before = model.gist_tokens.detach().clone()
        optimizer.step()
        gist_update_max = float(
            (model.gist_tokens.detach() - before).abs().max())
        metric = {
            "optimizer_step": step + 1,
            "seconds": time.perf_counter() - started,
            "video_loss": float(video_loss.detach()),
            "action_loss": float(action_loss.detach()),
            "total_loss": float(loss.detach()),
            "gist_grad_norm": gist_grad_norm,
            "gist_update_max": gist_update_max,
            "peak_rss_gib": _rss_gib(),
        }
        if not all(np.isfinite(value) for key, value in metric.items()
                   if key != "optimizer_step"):
            raise ValueError(("non-finite training metric", metric))
        if gist_grad_norm <= 0:
            raise ValueError(("gist gradient is zero", metric))
        metrics.append(metric)
        _event(events, "full_model_optimizer_step_passed", **metric)
    result = {
        "scope": scope,
        "steps": steps,
        "learning_rate": learning_rate,
        "metrics": metrics,
        "trainable_parameters": sum(value.numel() for value in trainable),
        "activation_checkpointing": True,
        "training_crop": {
            "latents": list(cropped["latents"].shape),
            "text_emb": list(cropped["text_emb"].shape),
            "actions": list(cropped["actions"].shape),
        },
    }
    _event(events, "full_model_training_passed", **result)
    return result, optimizer


def _save_full_checkpoint(output: Path, model, optimizer, *, step: int,
                          scope: str, events: Path) -> dict:
    checkpoint = output / f"checkpoint_step_{step}"
    transformer = checkpoint / "transformer"
    transformer.mkdir(parents=True)
    _event(events, "full_checkpoint_save_started", path=str(checkpoint))
    started = time.perf_counter()
    state = get_model_state_dict(
        model,
        options=StateDictOptions(full_state_dict=True, cpu_offload=True))
    state = {key: value.to(torch.bfloat16).contiguous()
             for key, value in state.items()}
    save_file(state, transformer / "diffusion_pytorch_model.safetensors")
    config = dict(model.config)
    config.pop("_name_or_path", None)
    (transformer / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n")
    torch.save({
        "step": step,
        "scope": scope,
        "optimizer_state_dict": optimizer.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
    }, checkpoint / "training_state.pt")
    result = {
        "path": str(checkpoint.resolve()),
        "seconds": time.perf_counter() - started,
        "weight_bytes": (transformer /
                         "diffusion_pytorch_model.safetensors").stat().st_size,
        "training_state_bytes": (checkpoint / "training_state.pt").stat().st_size,
    }
    _event(events, "full_checkpoint_save_passed", **result)
    return result


def run(args) -> dict:
    output = args.output
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.mkdir(parents=True)
    events = output / "events.jsonl"
    _event.started = time.perf_counter()
    available = _available_memory_gib()
    required = STAGE_MEMORY_GIB[args.stage]
    checkpoint = args.initializer / "transformer"
    header = _checkpoint_header(checkpoint)
    disk = shutil.disk_usage(output)
    plan = {
        "schema_version": 1,
        "kind": "memorywam_full_model_cpu_smoke",
        "stage": args.stage,
        "status": "PLANNED" if args.stage == "plan" else "RUNNING",
        "initializer": str(args.initializer.resolve()),
        "checkpoint": {key: value for key, value in header.items()
                       if key != "shapes"},
        "dataset_root": str(args.dataset_root.resolve()),
        "alignment": str(args.alignment.resolve()),
        "available_memory_gib_at_start": available,
        "minimum_memory_guard_gib": required,
        "disk_free_gib_at_start": disk.free / 1024**3,
        "threads": args.threads,
        "steps": args.steps,
        "learning_rate": args.learning_rate,
        "save_full_checkpoint": args.save_full_checkpoint,
        "script_sha256": _sha256(Path(__file__)),
        "limits": [
            "CPU attention mode overrides checkpoint flex attention",
            "forward uses one observed video patch and one action token",
            "training uses 2 frames, one video patch/frame, one action slot/frame",
            "no FSDP/HSDP, CUDA, NCCL, VRAM, online VAE or environment",
        ],
    }
    (output / "plan.json").write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n")
    _event(events, "plan_written", stage=args.stage,
           available_memory_gib=available, required_memory_gib=required)
    if args.stage == "plan":
        plan["status"] = "PASS"
        (output / "full_cpu_pass.json").write_text(
            json.dumps(plan, indent=2, sort_keys=True) + "\n")
        return plan
    if available < required and not args.ignore_memory_guard:
        raise MemoryError(
            f"stage={args.stage} requires at least {required} GiB available; "
            f"found {available:.1f} GiB")
    if args.stage == "train-all" and not args.allow_full_backbone_update:
        raise PermissionError(
            "train-all requires --allow-full-backbone-update")
    if args.save_full_checkpoint:
        minimum_disk = (80 if args.stage == "train-all" else 24)
        if disk.free / 1024**3 < minimum_disk:
            raise OSError(
                f"full checkpoint save requires {minimum_disk} GiB free")

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(max(1, min(4, args.threads)))
    torch.manual_seed(20261004)
    model, load_result = _load_model(checkpoint, header, events)
    report = {**plan, "load": load_result}
    batch = None
    if args.stage in ("forward", "train-gist", "train-all"):
        batch, source_shapes = _load_real_batch(
            args.dataset_root, args.alignment, events)
        report["dataset_source_shapes"] = source_shapes
    if args.stage == "forward":
        report["forward"] = _forward_smoke(model, batch, events)
    elif args.stage in ("train-gist", "train-all"):
        scope = "gist" if args.stage == "train-gist" else "all"
        training, optimizer = _train_smoke(
            model, batch, events, scope=scope, steps=args.steps,
            learning_rate=args.learning_rate)
        report["training"] = training
        if args.save_full_checkpoint:
            report["saved_checkpoint"] = _save_full_checkpoint(
                output, model, optimizer, step=args.steps, scope=scope,
                events=events)
    report["status"] = "PASS"
    report["elapsed_seconds"] = time.perf_counter() - _event.started
    report["peak_rss_gib"] = _rss_gib()
    (output / "full_cpu_pass.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n")
    _event(events, "stage_passed", stage=args.stage,
           elapsed_seconds=report["elapsed_seconds"])
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage", choices=tuple(STAGE_MEMORY_GIB), default="plan")
    parser.add_argument(
        "--initializer", type=Path,
        default=Path("train_out/rearrange_blocks_50demo_1000step_20260921_180241/"
                     "checkpoints/checkpoint_step_750"))
    parser.add_argument(
        "--dataset-root", type=Path,
        default=Path("/root/wyc/data/rmbench/lingbot_va2/"
                     "rearrange_blocks_50demo_fullctx_apf16_cf8"))
    parser.add_argument(
        "--alignment", type=Path,
        default=Path("artifacts/memorywam/stage4_alignment/"
                     "rearrange_train/episodes.jsonl"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument("--steps", type=int, choices=(1, 2), default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--ignore-memory-guard", action="store_true")
    parser.add_argument("--allow-full-backbone-update", action="store_true")
    parser.add_argument("--save-full-checkpoint", action="store_true")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    try:
        report = run(args)
    except BaseException as error:
        if args.output.exists():
            failure = {
                "status": "FAIL",
                "stage": args.stage,
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "elapsed_seconds": time.perf_counter() - _event.started,
                "peak_rss_gib": _rss_gib(),
            }
            (args.output / "full_cpu_failure.json").write_text(
                json.dumps(failure, indent=2, sort_keys=True) + "\n")
        raise
    print(json.dumps({
        "status": report["status"],
        "stage": report["stage"],
        "output": str(args.output.resolve()),
        "elapsed_seconds": report.get("elapsed_seconds", 0),
        "peak_rss_gib": report.get("peak_rss_gib", _rss_gib()),
    }, indent=2))


if __name__ == "__main__":
    main()
