#!/usr/bin/env python3
"""One exact-data free-sampling inference pass for Rearrange Blocks episode 0."""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from diffusers.video_processor import VideoProcessor
from diffusers.utils import export_to_video
from einops import rearrange
from tqdm import tqdm

from wan_va.configs import VA_CONFIGS
from wan_va.dataset.lerobot_latent_dataset import LatentLeRobotDataset
from wan_va.distributed.util import init_distributed
from wan_va.utils import data_seq_to_patch
from wan_va.wan_va_server import VA_Server


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--video-steps", type=int, default=25)
    parser.add_argument("--action-steps", type=int, default=50)
    parser.add_argument(
        "--history-source", choices=("expert", "predicted"), default="expert",
        help="History committed to KV cache after each generated chunk.",
    )
    return parser.parse_args()


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def sample_chunk(model: VA_Server, target_latents: torch.Tensor,
                 target_actions: torch.Tensor, chunk_id: int):
    device, dtype = model.device, model.dtype
    target_latents = target_latents.to(device=device, dtype=dtype)
    target_actions = target_actions.to(device=device, dtype=dtype)
    frames = target_latents.shape[2]
    frame_st_id = chunk_id * model.job_config.frame_chunk_size

    latents = torch.randn_like(target_latents)
    latent_cond = target_latents[:, :, :1] if chunk_id == 0 else None
    actions = torch.randn_like(target_actions)
    action_cond = torch.zeros_like(actions[:, :, :1]) if chunk_id == 0 else None

    model.scheduler.set_timesteps(model.job_config.num_inference_steps)
    model.action_scheduler.set_timesteps(model.job_config.action_num_inference_steps)
    video_timesteps = F.pad(model.scheduler.timesteps, (0, 1), value=0)
    action_timesteps = F.pad(model.action_scheduler.timesteps, (0, 1), value=0)

    for index, timestep in enumerate(tqdm(video_timesteps, desc=f"chunk {chunk_id} video", disable=model.job_config.rank != 0)):
        last = index == len(video_timesteps) - 1
        inputs = model._prepare_latent_input(
            latents, None, timestep, timestep, latent_cond, None, frame_st_id=frame_st_id
        )["latent_res_lst"]
        prediction = model.transformer(
            model._repeat_input_for_cfg(inputs),
            update_cache=1 if last else 0,
            cache_name=model.cache_name,
            action_mode=False,
        )
        if not last:
            prediction = data_seq_to_patch(
                model.job_config.patch_size,
                prediction,
                frames,
                target_latents.shape[-2],
                target_latents.shape[-1],
                batch_size=2 if model.use_cfg else 1,
            )
            if model.job_config.guidance_scale > 1:
                prediction = prediction[1:] + model.job_config.guidance_scale * (prediction[:1] - prediction[1:])
            else:
                prediction = prediction[:1]
            latents = model.scheduler.step(prediction, timestep, latents, return_dict=False)
        if latent_cond is not None:
            latents[:, :, :1] = latent_cond

    for index, timestep in enumerate(tqdm(action_timesteps, desc=f"chunk {chunk_id} action", disable=model.job_config.rank != 0)):
        last = index == len(action_timesteps) - 1
        inputs = model._prepare_latent_input(
            None, actions, timestep, timestep, None, action_cond, frame_st_id=frame_st_id
        )["action_res_lst"]
        prediction = model.transformer(
            model._repeat_input_for_cfg(inputs),
            update_cache=1 if last else 0,
            cache_name=model.cache_name,
            action_mode=True,
        )
        if not last:
            prediction = rearrange(prediction, "b (f n) c -> b c f n 1", f=frames)
            if model.job_config.action_guidance_scale > 1:
                prediction = prediction[1:] + model.job_config.action_guidance_scale * (prediction[:1] - prediction[1:])
            else:
                prediction = prediction[:1]
            actions = model.action_scheduler.step(prediction, timestep, actions, return_dict=False)
        if action_cond is not None:
            actions[:, :, :1] = action_cond

    actions[:, ~model.action_mask] = 0
    return latents.cpu(), actions.cpu()


@torch.no_grad()
def commit_history_cache(model: VA_Server, history_latents: torch.Tensor,
                         history_actions: torch.Tensor, frame_st_id: int):
    """Commit a two-frame history chunk using the server cache-update path."""
    model.transformer.clear_pred_cache(model.cache_name)
    latents = history_latents.to(device=model.device, dtype=model.dtype)
    actions = history_actions.to(device=model.device, dtype=model.dtype)
    inputs = model._prepare_latent_input(
        latents, actions, frame_st_id=frame_st_id
    )
    model.transformer(
        model._repeat_input_for_cfg(inputs["latent_res_lst"]),
        update_cache=2, cache_name=model.cache_name, action_mode=False,
    )
    model.transformer(
        model._repeat_input_for_cfg(inputs["action_res_lst"]),
        update_cache=2, cache_name=model.cache_name, action_mode=True,
    )


def physical_actions(normalized: torch.Tensor, q01: np.ndarray, q99: np.ndarray, used: list[int]):
    values = normalized[0, :, :, :, 0].float().numpy()
    values = (values + 1.0) / 2.0 * (q99[:, None, None] - q01[:, None, None] + 1e-6) + q01[:, None, None]
    return values[np.asarray(used)]


def action_metrics(pred, target, mask):
    valid = mask.astype(bool)
    diff = pred - target
    flat_valid = np.moveaxis(valid, 0, -1).reshape(-1, valid.shape[0]).all(axis=1)
    pred_rows = np.moveaxis(pred, 0, -1).reshape(-1, pred.shape[0])[flat_valid]
    target_rows = np.moveaxis(target, 0, -1).reshape(-1, target.shape[0])[flat_valid]
    row_diff = pred_rows - target_rows
    metrics = {
        "valid_actions": int(len(pred_rows)),
        "physical_component_mae": float(np.abs(row_diff).mean()),
        "left_translation_mae_mm": float(np.linalg.norm(row_diff[:, 0:3], axis=1).mean() * 1000),
        "right_translation_mae_mm": float(np.linalg.norm(row_diff[:, 8:11], axis=1).mean() * 1000),
        "left_gripper_mae": float(np.abs(row_diff[:, 7]).mean()),
        "right_gripper_mae": float(np.abs(row_diff[:, 15]).mean()),
    }
    for name, slc in (("left", slice(3, 7)), ("right", slice(11, 15))):
        p = pred_rows[:, slc]
        t = target_rows[:, slc]
        p /= np.maximum(np.linalg.norm(p, axis=1, keepdims=True), 1e-8)
        t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-8)
        dots = np.clip(np.abs((p * t).sum(axis=1)), 0, 1)
        metrics[f"{name}_rotation_mae_deg"] = float(np.degrees(2 * np.arccos(dots)).mean())
    return metrics, pred_rows, target_rows


def decode_video(model: VA_Server, latents: torch.Tensor):
    model.streaming_vae.clear_cache()
    if model.streaming_vae_half is not None:
        model.streaming_vae_half.clear_cache()
    vae_device = next(model.vae.parameters()).device
    # This is evaluation-only decoding.  Without inference_mode the VAE output
    # keeps an autograd graph and diffusers cannot convert it to NumPy.
    with torch.inference_mode():
        return np.asarray(model.decode_one_video(latents.detach().to(vae_device), "np")[0])


def main():
    args = parse_args()
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    init_distributed(world_size, local_rank, rank)
    seed_everything(args.seed + rank)

    cfg = copy.deepcopy(VA_CONFIGS["rmbench_overfit_train"])
    cfg.rank, cfg.local_rank, cfg.world_size = rank, local_rank, world_size
    cfg.dataset_path = str(args.dataset)
    cfg.empty_emb_path = str(args.dataset / "empty_emb.pt")
    with (args.dataset / "rmbench_action_stats.json").open() as stream:
        stats = json.load(stream)
    cfg.norm_stat = {key: stats[key] for key in ("q01", "q99")}
    cfg.episode_indices = [0]
    cfg.cfg_prob = 0.0
    cfg.load_worker = 0
    cfg.enable_wandb = False
    cfg.enable_offload = True
    cfg.fsdp_cpu_offload = True
    cfg.wan22_pretrained_model_name_or_path = str(args.model)
    cfg.save_root = str(args.output)
    cfg.num_inference_steps = args.video_steps
    cfg.action_num_inference_steps = args.action_steps
    # Keep the official online inference contract: two latent frames per chunk
    # and one continuous cache across the complete episode.
    cfg.frame_chunk_size = 2
    cfg.attn_window = 72

    dataset = LatentLeRobotDataset(str(args.dataset), cfg)
    if len(dataset) != 3:
        raise RuntimeError(f"Expected exactly 3 episode0 windows, got {len(dataset)}")

    model = VA_Server(cfg)
    model.use_cfg = cfg.guidance_scale > 1 or cfg.action_guidance_scale > 1
    model.empty_emb = torch.load(cfg.empty_emb_path, map_location="cpu", weights_only=False)
    model.video_processor = VideoProcessor(vae_scale_factor=1)

    args.output.mkdir(parents=True, exist_ok=True)
    predicted_chunks, target_chunks = [], []
    q01 = np.asarray(cfg.norm_stat["q01"], dtype=np.float32)
    q99 = np.asarray(cfg.norm_stat["q99"], dtype=np.float32)
    used = list(cfg.used_action_channel_ids)

    # Reconstruct the 26-frame latent/action episode. Adjacent 193-frame
    # training windows overlap by their boundary latent frame.
    samples = [dataset[i] for i in range(len(dataset))]
    episode_latents = torch.cat(
        [samples[0]["latents"]] + [s["latents"][:, 1:] for s in samples[1:]], dim=1
    ).unsqueeze(0)
    episode_actions = torch.cat(
        [samples[0]["actions"]] + [s["actions"][:, 1:] for s in samples[1:]], dim=1
    ).unsqueeze(0)
    episode_mask = torch.cat(
        [samples[0]["actions_mask"]] + [s["actions_mask"][:, 1:] for s in samples[1:]], dim=1
    )
    if episode_latents.shape[2] != 26:
        raise RuntimeError(f"Expected 26 episode latent frames, got {episode_latents.shape[2]}")

    model._reset(prompt=None)
    model.prompt_embeds = samples[0]["text_emb"].unsqueeze(0).to(model.device, model.dtype)
    model.negative_prompt_embeds = model.empty_emb.unsqueeze(0).to(model.device, model.dtype)

    predicted_action_chunks = []
    for chunk_id, start in enumerate(range(0, episode_latents.shape[2], cfg.frame_chunk_size)):
        stop = start + cfg.frame_chunk_size
        target_latent_chunk = episode_latents[:, :, start:stop]
        target_action_chunk = episode_actions[:, :, start:stop]
        pred_latents, pred_actions = sample_chunk(
            model, target_latent_chunk, target_action_chunk, chunk_id
        )
        predicted_chunks.append(pred_latents)
        predicted_action_chunks.append(pred_actions)
        target_chunks.append(target_latent_chunk)
        if rank == 0:
            torch.save(
                {"latents": pred_latents, "actions": pred_actions},
                args.output / f"prediction_chunk_{chunk_id:02d}.pt",
            )
        if stop < episode_latents.shape[2]:
            if args.history_source == "expert":
                history_latents, history_actions = target_latent_chunk, target_action_chunk
            else:
                history_latents, history_actions = pred_latents, pred_actions
            commit_history_cache(model, history_latents, history_actions, start)
        dist.barrier()

    if rank == 0:
        predicted_latents = torch.cat(predicted_chunks, dim=2)
        predicted_actions = torch.cat(predicted_action_chunks, dim=2)
        selected_mask = episode_mask[used, :, :, 0].numpy()
        predicted_physical = physical_actions(predicted_actions, q01, q99, used)
        target_physical = physical_actions(episode_actions, q01, q99, used)
        aggregate, predicted, target = action_metrics(
            predicted_physical, target_physical, selected_mask
        )
        np.save(args.output / "predicted_actions_physical.npy", predicted)
        np.save(args.output / "target_actions_physical.npy", target)
        report = {
            "episode_index": 0,
            "seed": args.seed,
            "checkpoint": str(args.model / "transformer"),
            "dataset": str(args.dataset),
            "video_steps": args.video_steps,
            "action_steps": args.action_steps,
            "protocol": (
                "official_2frame_continuous_kv_teacher_forced_episode_history"
                if args.history_source == "expert"
                else "official_2frame_continuous_kv_autoregressive_predicted_history"
            ),
            "history_source": args.history_source,
            "latent_frames": int(episode_latents.shape[2]),
            "chunks": int(len(predicted_chunks)),
            "aggregate": aggregate,
        }
        (args.output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")

        # Match the official generate() path and decode each continuous latent
        # stream once, preserving temporal VAE state across chunk boundaries.
        if model.enable_offload:
            model.vae = model.vae.to(model.device).to(model.dtype)
        pred_video = decode_video(model, predicted_latents)
        target_video = decode_video(model, episode_latents)
        count = min(len(pred_video), len(target_video))
        comparison = np.concatenate([target_video[:count], pred_video[:count]], axis=2)
        export_to_video(pred_video, str(args.output / "predicted_episode0.mp4"), fps=4)
        export_to_video(target_video, str(args.output / "target_episode0.mp4"), fps=4)
        export_to_video(comparison, str(args.output / "target_left_predicted_right.mp4"), fps=4)
        print(json.dumps(report, indent=2), flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
