# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
import argparse
import copy
import os
import sys
from pathlib import Path
import wandb

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    get_optimizer_state_dict,
    set_optimizer_state_dict,
    StateDictOptions,
)
from safetensors.torch import save_file, load_file
import json

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from configs import VA_CONFIGS
from distributed.fsdp import shard_model, apply_ac, resolve_fsdp_topology
from distributed.util import (
    _configure_model, 
    init_distributed, 
    dist_mean, 
    dist_max
)
from einops import rearrange
from modules.utils import (
    load_transformer,
)
from utils import (
    init_logger, 
    logger, 
    get_mesh_id, 
    sample_timestep_id,
    data_seq_to_patch,
    warmup_constant_lambda,
    FlowMatchScheduler
)

from dataset import MultiLatentLeRobotDataset
from dataset.bounded_full_context_dataset import (
    BoundedFullContextDataset, EpisodeBalancedDistributedSampler,
)
import gc
from functools import partial
import time
import random
import numpy as np


class Trainer:
    def __init__(self, config):
        if config.enable_wandb and config.rank == 0:
            if os.getenv('WANDB_API_KEY'):
                wandb.login(host=os.getenv('WANDB_BASE_URL'), key=os.environ['WANDB_API_KEY'])
            self.wandb = wandb
            self.wandb.init(
                entity=os.getenv("WANDB_TEAM_NAME"),
                project=os.getenv("WANDB_PROJECT", "lingbot-va-rmbench"),
                # dir=log_dir,
                config=config,
                mode=os.getenv("WANDB_MODE", "online"),
                name=os.getenv("WANDB_RUN_NAME", getattr(config, "run_name", "lingbot-va-rmbench"))
                # name=os.path.basename(os.path.normpath(job_config.job.dump_folder))
            )
            logger.info("WandB logging enabled")
        self.step = 0
        self.config = config
        self.device = torch.device(f"cuda:{config.local_rank}")
        self.dtype = config.param_dtype
        self.patch_size = config.patch_size
        self.fsdp_cpu_offload = bool(getattr(config, 'fsdp_cpu_offload', False))
        self.fsdp_shard_size = int(getattr(config, 'fsdp_shard_size', 0))
        self.fsdp_reshard_after_forward = bool(getattr(
            config, 'fsdp_reshard_after_forward', True))
        self.fsdp_root_reshard_after_forward = bool(getattr(
            config, 'fsdp_root_reshard_after_forward', True))
        self.fsdp_topology = resolve_fsdp_topology(
            config.world_size, self.fsdp_shard_size)
        if config.rank == 0:
            logger.info(
                "FSDP topology: world=%d shard=%d replicate=%d HSDP=%s "
                "cpu_offload=%s reshard=%s root_reshard=%s",
                self.fsdp_topology.world_size,
                self.fsdp_topology.shard_size,
                self.fsdp_topology.replicate_size,
                self.fsdp_topology.uses_hsdp,
                self.fsdp_cpu_offload,
                self.fsdp_reshard_after_forward,
                self.fsdp_root_reshard_after_forward)
        self.cpu_process_group = (
            dist.new_group(backend='gloo')
            if self.fsdp_cpu_offload and dist.is_initialized() else None
        )

        # Load models
        logger.info("Loading models...")

        # Load and shard transformer with FSDP
        logger.info("Loading transformer...")

        if hasattr(config, 'resume_from') and config.resume_from:
            transformer_path = os.path.join(config.resume_from, 'transformer')
            if config.rank == 0:
                logger.info(f"Resuming from checkpoint: {transformer_path}")
        else:
            transformer_path = os.path.join(config.wan22_pretrained_model_name_or_path, 'transformer')

        self.transformer = load_transformer(
            transformer_path,
            torch_dtype=torch.float32,
            torch_device='cpu',
            attn_mode="flex"
        )
        if getattr(config, "memorywam_enabled", False):
            self.transformer.enable_memorywam(
                config.memorywam_gist_tokens,
                config.memorywam_anchor_frames,
                config.memorywam_recent_frames,
                config.memorywam_mode)

        logger.info("Setting up activation checkpointing ...")
        apply_ac(self.transformer)

        logger.info("Setting up FSDP...")
        shard_fn = partial(
            shard_model,
            cpu_offload=self.fsdp_cpu_offload,
            shard_size=self.fsdp_shard_size,
            reshard_after_forward=self.fsdp_reshard_after_forward,
            root_reshard_after_forward=self.fsdp_root_reshard_after_forward)
        self.transformer = _configure_model(
            model=self.transformer,
            shard_fn=shard_fn,
            param_dtype=self.dtype,
            device=self.device,
            eval_mode=False,
        )
        self.transformer.train()
        self.transformer.requires_grad_(True)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            [p for p in self.transformer.parameters() if p.requires_grad],
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            eps=1e-8,
            weight_decay=config.weight_decay,
            fused=not self.fsdp_cpu_offload,
            foreach=False,
        )

        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, 
            lr_lambda=lambda step: warmup_constant_lambda(step, warmup_steps=config.warmup_steps))

        # Setup dataloaders
        logger.info("Setting up datasets...")
        bounded = bool(getattr(config,'bounded_window_index',None))
        if getattr(config, "memorywam_enabled", False):
            if bounded:
                raise ValueError(
                    "MemoryWAM episode sequencing is not implemented for the "
                    "legacy bounded-window wrapper")
            if not getattr(config, "memory_alignment_path", None):
                raise ValueError(
                    "MemoryWAM requires RMBENCH_MEMORY_ALIGNMENT_PATH")
        if bounded:
            train_dataset = BoundedFullContextDataset(config, config.bounded_window_index)
            train_sampler = EpisodeBalancedDistributedSampler(
                train_dataset, config.world_size, config.rank, seed=int(config.seed)
            )
        else:
            train_dataset = MultiLatentLeRobotDataset(config=config)
            train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=config.world_size,
                rank=config.rank,
                shuffle=True,
                seed=int(config.seed)
            ) if config.world_size > 1 else None
        self.loader_generator = torch.Generator().manual_seed(int(config.seed))
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=(train_sampler is None), 
            num_workers=config.load_worker,
            sampler=train_sampler,
            generator=self.loader_generator,
        )
        self.validation_loader = None
        validation_dataset_path = getattr(config, 'validation_dataset_path', None)
        if validation_dataset_path:
            validation_config = copy.deepcopy(config)
            validation_config.dataset_path = validation_dataset_path
            validation_config.empty_emb_path = os.path.join(validation_dataset_path, "empty_emb.pt")
            validation_config.episode_indices = None
            validation_config.memory_alignment_path = getattr(
                config, "validation_memory_alignment_path", None)
            validation_dataset = MultiLatentLeRobotDataset(config=validation_config)
            if len(validation_dataset) == 0:
                raise ValueError("Validation split contains no loadable latent windows")
            # Every FSDP rank evaluates the same five episodes so all ranks make
            # the same number of forward calls. The rank-specific fixed noise
            # seeds additionally average validation loss across three noises.
            self.validation_loader = DataLoader(
                validation_dataset, batch_size=1, shuffle=False,
                num_workers=config.load_worker,
            )
            logger.info(
                "Independent validation dataset: %s (%d windows)",
                validation_dataset_path, len(validation_dataset),
            )
        elif bool(getattr(config, 'enable_early_stop', False)):
            raise ValueError(
                "Early stopping requires RMBENCH_VALIDATION_DATASET_PATH"
            )
        # TODO set latent scheduler and action scheduler
        self.train_scheduler_latent = FlowMatchScheduler(shift=self.config.snr_shift, sigma_min=0.0, extra_one_step=True)
        self.train_scheduler_latent.set_timesteps(1000, training=True)
        self.train_scheduler_action = FlowMatchScheduler(shift=self.config.action_snr_shift, sigma_min=0.0, extra_one_step=True)
        self.train_scheduler_action.set_timesteps(1000, training=True)

        self.save_dir = Path(config.save_root) / "checkpoints"
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = Path(config.save_root) / "training_metrics.jsonl"
        self.training_started_at = time.time()
        if config.rank == 0:
            self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
            self.metrics_path.write_text("")

        self.gradient_accumulation_steps = getattr(config, 'gradient_accumulation_steps', 1)
        self.train_loader_iter = None
        self.validation_history = []
        # if hasattr(config, 'resume_from') and config.resume_from:
        #     self._load_training_state(config.resume_from)
    
    def _get_next_batch(self):
        """Get next batch from iterator, reset if epoch is finished."""
        if self.train_loader_iter is None:
            self.train_loader_iter = iter(self.train_loader)
        
        try:
            batch = next(self.train_loader_iter)
        except StopIteration:
            # Reset sampler and iterator when epoch finishes
            if hasattr(self.train_loader.sampler, 'set_epoch'):
                self.train_loader.sampler.set_epoch(self.train_loader.sampler.epoch + 1)
            self.train_loader_iter = iter(self.train_loader)
            batch = next(self.train_loader_iter)
        
        return batch

    @torch.no_grad()
    def _add_noise(self, latent, train_scheduler, action_mask=False, action_mode=False, noisy_cond_prob=0.):
        # 输入形状：[batch, channel, frame, height, width]
        B, C, F, H, W = latent.shape
        # 为每个时间帧随机采样扩散时刻timestep_ids
        timestep_ids = sample_timestep_id(batch_size=F, num_train_timesteps=train_scheduler.num_train_timesteps)
        # 生成噪声，构造模型输入和监督目标
        noise = torch.zeros_like(latent).normal_()
        timesteps = train_scheduler.timesteps[timestep_ids].to(device=self.device)
        noisy_latents =train_scheduler.add_noise(latent, noise, timesteps, t_dim=2)
        targets =train_scheduler.training_target(latent, noise, timesteps)
        # 视频按照 patch 划分；动作不划分 patch
        patch_f, patch_h, patch_w = self.patch_size
        if action_mode:
            patch_f = patch_h = patch_w = 1
        # TODO generate grid_id
        latent_grid_id = get_mesh_id(
            latent.shape[-3] // patch_f,  # F
            latent.shape[-2] // patch_h,  # H
            latent.shape[-1] // patch_w,  # W
            t=1 if action_mode else 0,  # 1 for action mode (0 for latent), not used
            f_w=1,
            f_shift=0,
            action=action_mode
        ).to(self.device)  # shape: [4, seq_len]
        latent_grid_id = latent_grid_id[None].repeat(B, 1, 1)
        # 随机给条件 latent 加噪，模拟推理时不完美的历史条件
        if torch.rand(1).item() < noisy_cond_prob:
            cond_timestep_ids = sample_timestep_id(
                    batch_size=F,
                    min_timestep_bd=0.5, 
                    max_timestep_bd=1.0, 
                    num_train_timesteps=train_scheduler.num_train_timesteps,
                )
            noise = torch.zeros_like(latent).normal_()
            cond_timesteps = train_scheduler.timesteps[cond_timestep_ids].to(device=self.device)
            latent = train_scheduler.add_noise(latent, noise, cond_timesteps, t_dim=2)
        else:
            cond_timesteps = torch.zeros_like(timesteps)

        if action_mask is not None:
            noisy_latents *= action_mask.float()
            targets *= action_mask.float()
            latent *= action_mask.float()
        # 所有 batch 样本共用同一组逐帧 timestep
        timesteps_per_batch = timesteps[None].repeat(B, 1)
        if action_mask is not None:
             # 判断每个动作帧是否至少包含一个有效动作
            # A fully masked action frame is a deterministic condition/padding
            # frame. Inference presents such a frame at timestep zero, so use
            # the same time embedding during training. Partially masked frames
            # retain their sampled diffusion timestep.
            valid_action_frames = action_mask.bool().any(dim=(1, 3, 4))
            if valid_action_frames.shape != (B, F):
                raise ValueError(
                    "Action mask frame shape does not match action timesteps: "
                    f"{tuple(valid_action_frames.shape)} != {(B, F)}"
                )
            timesteps_per_batch = timesteps_per_batch.masked_fill(
                ~valid_action_frames, 0
            )

        return dict(
            timesteps=timesteps_per_batch,                    # 主输入噪声时刻 shape: [B, seq_len]
            noisy_latents=noisy_latents,                      # 主输入噪声 shape: [B, C, seq_len, H, W]
            targets=targets,                                  # 主输入噪声监督目标 shape: [B, C, seq_len, H, W]
            latent=latent,                                    # 条件噪声 shape: [B, C, seq_len, H, W]
            cond_timesteps=cond_timesteps[None].repeat(B, 1), # 条件噪声时刻 shape: [B, seq_len]
            grid_id=latent_grid_id,                           # 输入网格索引 shape: [B, seq_len, 4]
        )

    @torch.no_grad()
    def _prepare_input_dict(self, batch_dict):
        """Prepare input dict following infer code pattern from wan_va_server.py."""
        # Generate grid_id following infer code (no batch dimension yet)
        # For action mode: get_mesh_id(shape[-3], shape[-2], shape[-1], t=1, f_w=1, f_shift, action=True)
        latent_dict = self._add_noise(
            latent=batch_dict['latents'], 
            train_scheduler=self.train_scheduler_latent, 
            action_mask=None, 
            action_mode=False,
            noisy_cond_prob=(0.0 if getattr(self.config, "memorywam_enabled",
                                            False) else 0.5))
        
        action_dict = self._add_noise(
            latent=batch_dict['actions'], 
            train_scheduler=self.train_scheduler_action, 
            action_mask=batch_dict['actions_mask'], 
            action_mode=True,
            noisy_cond_prob=0.0)

        latent_dict['text_emb'] = batch_dict['text_emb']
        action_dict['text_emb'] = batch_dict['text_emb']
        action_dict['actions_mask'] = batch_dict['actions_mask']

        input_dict = {
            'latent_dict': latent_dict,
            'action_dict': action_dict,
            'chunk_size': torch.randint(1, 5, (1,)).item(),
            'window_size': torch.randint(4, 65, (1,)).item(),
        }
        memory_keys = (
            "dataset_numeric_id", "episode_index", "segment_index",
            "global_frame_ids", "source_rgb_indices", "decision_frame_ids",
            "target_frame_ids", "target_action_group_ids",
            "video_target_mask", "action_valid_slots_by_latent", "reset_mask",
        )
        if any(key in batch_dict for key in memory_keys):
            missing = [key for key in memory_keys if key not in batch_dict]
            if missing:
                raise ValueError(f"Incomplete MemoryWAM batch metadata: {missing}")
            input_dict["memory_meta"] = {
                key: batch_dict[key] for key in memory_keys
            }
            input_dict["memory_meta"]["detach_interval"] = int(getattr(
                self.config, "memorywam_detach_interval", 0))
        return input_dict

    def convert_input_format(self, input_dict):
        """Convert input dict to match transformer input format if needed."""
        for key, value in input_dict.items():
            if torch.is_tensor(value):
                input_dict[key] = value.to(self.device)#.to(self.dtype)
        return input_dict

    @torch.no_grad()
    def _validation_loss(self):
        """Evaluate the complete held-out split with deterministic diffusion noise."""
        if self.validation_loader is None:
            raise RuntimeError("Validation was requested without a held-out loader")
        python_state, numpy_state = random.getstate(), np.random.get_state()
        cpu_state, cuda_state = torch.get_rng_state(), torch.cuda.get_rng_state_all()
        was_training = self.transformer.training
        try:
            validation_seed = int(self.config.seed) + 100000 + int(self.config.rank)
            random.seed(validation_seed)
            np.random.seed(validation_seed)
            torch.manual_seed(validation_seed)
            torch.cuda.manual_seed_all(validation_seed)
            self.transformer.eval()
            values = torch.zeros(3, device=self.device, dtype=torch.float64)
            for batch in self.validation_loader:
                batch = {
                    key: value.to(self.device) if torch.is_tensor(value) else value
                    for key, value in batch.items()
                }
                inputs = self._prepare_input_dict(batch)
                losses = self.compute_loss(inputs, self.transformer(inputs, train_mode=True))
                values[0] += losses[0].double()
                values[1] += losses[1].double()
                values[2] += 1
            if dist.is_initialized():
                dist.all_reduce(values, op=dist.ReduceOp.SUM)
            if values[2].item() == 0:
                raise RuntimeError("Validation loader produced zero batches")
            return float((values[0] / values[2]).cpu()), float((values[1] / values[2]).cpu())
        finally:
            if was_training:
                self.transformer.train()
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            torch.set_rng_state(cpu_state)
            torch.cuda.set_rng_state_all(cuda_state)

    def _validation_converged(self):
        patience = int(getattr(self.config, "early_stop_patience", 5))
        delta = float(getattr(self.config, "early_stop_min_relative_improvement", 0.01))
        if len(self.validation_history) < patience + 1:
            return False
        recent = self.validation_history[-(patience + 1):]
        def stalled(key):
            baseline = recent[0][key]
            best_later = min(record[key] for record in recent[1:])
            improvement = (baseline - best_later) / max(abs(baseline), 1e-12)
            return improvement < delta
        return stalled("latent") and stalled("action")

    def compute_loss(self,
        input_dict,
        pred
    ):
        latent_pred, action_pred = pred
        action_pred = rearrange(action_pred, 'b (f n) c -> b c f n 1', f=input_dict['action_dict']['targets'].shape[-3])
        latent_pred = data_seq_to_patch(
                        self.patch_size, latent_pred,
                        input_dict['latent_dict']['targets'].shape[-3], input_dict['latent_dict']['targets'].shape[-2],
                        input_dict['latent_dict']['targets'].shape[-1], batch_size=latent_pred.shape[0])
        Bn, Fn = input_dict['latent_dict']['timesteps'].shape
        latent_loss_weight = self.train_scheduler_latent.training_weight(input_dict['latent_dict']['timesteps'].flatten()).reshape(Bn, Fn)
        action_loss_weight = self.train_scheduler_action.training_weight(input_dict['action_dict']['timesteps'].flatten()).reshape(Bn, Fn)

        # Frame-wise video loss calculation
        latent_loss = F.mse_loss(latent_pred.float(), input_dict['latent_dict']['targets'].float().detach(), reduction='none')
        latent_loss = latent_loss * latent_loss_weight[:, None, :, None, None]
        # Permute to (B, F, H, W, C) and flatten to (B*F, H*W*C)
        latent_loss = latent_loss.permute(0, 2, 3, 4, 1)  # (B, C, F, H, W) -> (B, F, H, W, C)
        latent_loss = latent_loss.flatten(0, 1).flatten(1)  # (B, F, H, W, C) -> (B*F, H*W*C)
        # Sum per frame and compute mask per frame
        latent_loss_per_frame = latent_loss.sum(dim=1)  # (B*F,)
        latent_mask_per_frame = torch.ones_like(latent_loss).sum(dim=1)  # (B*F,)
        if "memory_meta" in input_dict:
            video_target_mask = input_dict["memory_meta"][
                "video_target_mask"].reshape(-1).bool()
            if video_target_mask.shape != latent_loss_per_frame.shape:
                raise ValueError(("video target mask shape",
                                  tuple(video_target_mask.shape),
                                  tuple(latent_loss_per_frame.shape)))
            if not torch.any(video_target_mask):
                raise ValueError("MemoryWAM batch has no video targets")
            latent_loss_per_frame = latent_loss_per_frame[video_target_mask]
            latent_mask_per_frame = latent_mask_per_frame[video_target_mask]
        latent_loss = (latent_loss_per_frame /
                       (latent_mask_per_frame + 1e-6)).mean()

        # Frame-wise action loss calculation
        action_loss = F.mse_loss(action_pred.float(), input_dict['action_dict']['targets'].float().detach(), reduction='none')
        action_loss = action_loss * action_loss_weight[:, None, :, None, None]
        action_loss = action_loss * input_dict['action_dict']['actions_mask'].float()
        # Permute to (B, F, H, W, C) and flatten to (B*F, H*W*C)
        action_loss = action_loss.permute(0, 2, 3, 4, 1)  # (B, C, F, H, W) -> (B, F, H, W, C)
        action_mask = input_dict['action_dict']['actions_mask'].float().permute(0, 2, 3, 4, 1)  # (B, C, F, H, W) -> (B, F, H, W, C)
        action_loss = action_loss.flatten(0, 1).flatten(1)  # (B, F, H, W, C) -> (B*F, H*W*C)
        action_mask = action_mask.flatten(0, 1).flatten(1)  # (B, F, H, W, C) -> (B*F, H*W*C)
        # Sum per frame and normalize by mask per frame
        action_loss_per_frame = action_loss.sum(dim=1)  # (B*F,)
        action_mask_per_frame = action_mask.sum(dim=1)  # (B*F,)
        # RMBench reserves fully masked action frames for the initial
        # observation condition and any tail padding. Exclude those frames
        # from the outer mean while preserving LingBot's per-frame weighting.
        valid_action_frames = action_mask_per_frame > 0
        if not torch.any(valid_action_frames):
            raise ValueError(
                "Action loss received a batch with no valid action frames"
            )
        action_loss = (
            action_loss_per_frame[valid_action_frames]
            / action_mask_per_frame[valid_action_frames]
        ).mean()

        return latent_loss / self.gradient_accumulation_steps, action_loss / self.gradient_accumulation_steps

    def _train_step(self, batch, batch_idx):
        """Train a single batch, returns losses for logging."""
        batch = self.convert_input_format(batch)
        input_dict = self._prepare_input_dict(batch)
        
        should_sync = (batch_idx + 1) % self.gradient_accumulation_steps == 0
        
        if not should_sync:
            self.transformer.set_requires_gradient_sync(False)
        else:
            self.transformer.set_requires_gradient_sync(True)

        output = self.transformer(input_dict, train_mode=True)
        latent_loss, action_loss = self.compute_loss(input_dict, output)
        loss = latent_loss + action_loss

        loss.backward()

        losses = {'latent_loss': latent_loss.detach(), 'action_loss': action_loss.detach()}
        
        # Only update weights after accumulating gradients
        if should_sync:
            total_norm = self._clip_grad_norm(2.0)
            gist_grad_norm = None
            gist_before = None
            if (getattr(self.config, "memorywam_enabled", False) and
                    self.transformer.memorywam_policy.gist_per_frame):
                gist_grad_norm = self._parameter_grad_norm(
                    self.transformer.gist_tokens)
                gist_before = self._local_parameter_copy(
                    self.transformer.gist_tokens)
            self.optimizer.step()
            self.lr_scheduler.step()
            if gist_before is not None:
                gist_update_max = self._parameter_update_max(
                    self.transformer.gist_tokens, gist_before)
            self.optimizer.zero_grad()
            
            losses['total_norm'] = total_norm
            if gist_grad_norm is not None:
                losses['gist_grad_norm'] = gist_grad_norm
                losses['gist_update_max'] = gist_update_max
            losses['should_log'] = True
        else:
            losses['should_log'] = False

        return losses

    @torch.no_grad()
    def _parameter_grad_norm(self, parameter):
        """Return a global L2 norm for a possibly sharded parameter grad."""
        if parameter.grad is None:
            return torch.zeros((), device=self.device)
        gradient = parameter.grad
        if hasattr(gradient, 'to_local'):
            gradient = gradient.to_local()
        square = gradient.detach().double().square().sum()
        if dist.is_initialized():
            dist.all_reduce(square, op=dist.ReduceOp.SUM)
        return square.sqrt().float()

    @staticmethod
    @torch.no_grad()
    def _local_parameter_copy(parameter):
        value = parameter.detach()
        if hasattr(value, 'to_local'):
            value = value.to_local()
        return value.float().clone()

    @torch.no_grad()
    def _parameter_update_max(self, parameter, before):
        after = parameter.detach()
        if hasattr(after, 'to_local'):
            after = after.to_local()
        if before.numel():
            update = (after.float() - before).abs().max()
        else:
            update = torch.zeros((), device=self.device)
        if dist.is_initialized():
            dist.all_reduce(update, op=dist.ReduceOp.MAX)
        return update

    @torch.no_grad()
    def _clip_grad_norm(self, max_norm):
        if not self.fsdp_cpu_offload:
            return torch.nn.utils.clip_grad_norm_(
                self.transformer.parameters(), max_norm
            )

        local_gradients = []
        local_sq_sum = torch.zeros((), dtype=torch.float64, device='cpu')
        for parameter in self.transformer.parameters():
            if parameter.grad is None:
                continue
            gradient = parameter.grad
            local_gradient = (
                gradient.to_local() if hasattr(gradient, 'to_local') else gradient
            )
            local_gradients.append(local_gradient)
            local_sq_sum += local_gradient.detach().double().square().sum()
        if dist.is_initialized():
            dist.all_reduce(local_sq_sum, op=dist.ReduceOp.SUM,
                            group=self.cpu_process_group)
        total_norm = local_sq_sum.sqrt()
        coefficient = min(float(max_norm) / (total_norm.item() + 1e-6), 1.0)
        if coefficient < 1.0:
            for local_gradient in local_gradients:
                local_gradient.mul_(coefficient)
        return total_norm.float()

    def save_checkpoint(self,):
        """Save model checkpoint in the same format as pretrained model."""
        try:
            state_dict = get_model_state_dict(
                self.transformer,
                options=StateDictOptions(full_state_dict=True, cpu_offload=True),
            )
            state_dict_bf16 = {k: v.to(torch.bfloat16) for k, v in state_dict.items()}
            # optim_state = get_optimizer_state_dict(
            #         self.transformer, self.optimizer,
            #         options=StateDictOptions(full_state_dict=True, cpu_offload=True),
            #     )

            # Only rank 0 saves the checkpoint
            if self.config.rank == 0:
                checkpoint_dir = self.save_dir / f"checkpoint_step_{self.step}"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)

                # Save transformer in the same format as pretrained model
                transformer_dir = checkpoint_dir / "transformer"
                transformer_dir.mkdir(parents=True, exist_ok=True)

                logger.info(f"Saving transformer to {transformer_dir}")

                # Manually save in diffusers format (outside FSDP context to avoid deadlock)
                # Save model weights
                model_file = transformer_dir / "diffusion_pytorch_model.safetensors"
                save_file(state_dict_bf16, model_file)

                # Save config (copy from original transformer config and update _name_or_path)
                config_file = transformer_dir / "config.json"
                config_dict = dict(self.transformer.config)
                config_dict.pop('_name_or_path', None)
                with open(config_file, 'w') as f:
                    json.dump(config_dict, f, indent=2)

                # # Save optimizer state and training metadata in PyTorch format
                # training_state_path = checkpoint_dir / "training_state.pt"
                # logger.info(f"Saving training state to {training_state_path}")
                # torch.save({
                #     'step': self.step,
                #     'optimizer_state_dict': optim_state,
                #     'config': vars(self.config),
                # }, training_state_path)

                logger.info(f"Checkpoint saved successfully at step {self.step}")

            # Synchronize all processes after saving
            if dist.is_initialized():
                dist.barrier()

        except Exception as e:
            if self.config.rank == 0:
                logger.error(f"Failed to save checkpoint: {e}")
                import traceback
                logger.error(traceback.format_exc())
            # A run without a complete checkpoint must fail.  Swallowing this
            # exception can leave a training_summary.json that falsely looks
            # successful while the only checkpoint is partial or absent.
            if dist.is_initialized():
                dist.barrier()
            raise

    def _load_training_state(self, checkpoint_path):
        """Load training state (optimizer + step) after FSDP and optimizer creation."""
        checkpoint_dir = Path(checkpoint_path)
        training_state_path = checkpoint_dir / "training_state.pt"

        if not training_state_path.exists():
            if self.config.rank == 0:
                logger.warning(f"Training state not found: {training_state_path}, starting from step 0")
            return

        if self.config.rank == 0:
            logger.info(f"Loading training state from {training_state_path}")

        # All ranks load the training state directly
        training_state = torch.load(training_state_path, map_location='cpu', weights_only=False)

        # All ranks load optimizer state (required for FSDP)
        set_optimizer_state_dict(
            self.transformer, self.optimizer,
            optim_state_dict=training_state['optimizer_state_dict'],
            options=StateDictOptions(full_state_dict=True, strict=False)
        )
        self.step = training_state.get('step', 0)

        if self.config.rank == 0:
            logger.info(f"Training state loaded, resuming from step {self.step}")

        # Synchronize all ranks
        if dist.is_initialized():
            dist.barrier()

    def train(self):
        """Main training loop - train by steps instead of epochs."""
        logger.info(f"Starting training for {self.config.num_steps} steps...")
        self.transformer.train()

        progress_bar = tqdm(
            total=self.config.num_steps,
            desc="Training",
            disable=(self.config.rank != 0),
            leave=True,
            dynamic_ncols=True,
            initial=self.step
        )

        self.optimizer.zero_grad()
        accumulated_latent_losses = []
        accumulated_action_losses = []
        accumulated_local_latent_frames = 0
        step_in_accumulation = 0

        stop_reason = "max_steps"
        while self.step < self.config.num_steps:
            if step_in_accumulation == 0:
                optimizer_cycle_started = time.perf_counter()
            # Get next batch (handles epoch reset automatically)
            batch = self._get_next_batch()
            accumulated_local_latent_frames += (
                int(batch['latents'].shape[0]) *
                int(batch['latents'].shape[2]))
            
            losses = self._train_step(batch, step_in_accumulation)
            
            # Accumulate losses for logging
            accumulated_latent_losses.append(losses['latent_loss'])
            accumulated_action_losses.append(losses['action_loss'])
            step_in_accumulation += 1

            # Log and checkpoint when optimizer steps
            if losses['should_log']:
                lr = self.lr_scheduler.get_last_lr()[0]

                # Average accumulated losses
                latent_loss_show = dist_mean(torch.stack(accumulated_latent_losses).sum()).detach().cpu().item()
                action_loss_show = dist_mean(torch.stack(accumulated_action_losses).sum()).detach().cpu().item()
                max_latent_loss_show = dist_max(torch.stack(accumulated_latent_losses).sum()).detach().cpu().item()
                max_action_loss_show = dist_max(torch.stack(accumulated_action_losses).sum()).detach().cpu().item()

                # Clear accumulated losses
                accumulated_latent_losses = []
                accumulated_action_losses = []
                step_in_accumulation = 0

                torch.cuda.synchronize()
                optimizer_step_seconds = torch.tensor(
                    time.perf_counter() - optimizer_cycle_started,
                    dtype=torch.float64,
                    device=self.device)
                if dist.is_initialized():
                    dist.all_reduce(
                        optimizer_step_seconds, op=dist.ReduceOp.MAX)
                global_latent_frames = torch.tensor(
                    accumulated_local_latent_frames,
                    dtype=torch.int64,
                    device=self.device)
                if dist.is_initialized():
                    dist.all_reduce(
                        global_latent_frames, op=dist.ReduceOp.SUM)
                accumulated_local_latent_frames = 0
                if self.step % self.config.gc_interval == 0:
                    torch.cuda.empty_cache()
                    gc.collect()
                peak_memory = torch.tensor([
                    torch.cuda.max_memory_allocated(self.device),
                    torch.cuda.max_memory_reserved(self.device),
                ], dtype=torch.int64, device=self.device)
                if dist.is_initialized():
                    dist.all_reduce(peak_memory, op=dist.ReduceOp.MAX)

                #写 training_metrics.jsonl；更新进度条；上传 WandB
                if self.config.rank == 0:
                    total_norm = losses['total_norm']
                    metric = {
                        'optimizer_step': self.step + 1,
                        'latent_loss': latent_loss_show,
                        'action_loss': action_loss_show,
                        'max_latent_loss': max_latent_loss_show,
                        'max_action_loss': max_action_loss_show,
                        'grad_norm': total_norm.item(),
                        'lr': lr,
                        'elapsed_seconds': time.time() - self.training_started_at,
                        'optimizer_step_seconds': float(
                            optimizer_step_seconds.item()),
                        'global_latent_frames': int(
                            global_latent_frames.item()),
                        'global_latent_frames_per_second': float(
                            global_latent_frames.item() /
                            optimizer_step_seconds.item()),
                        'peak_memory_allocated_bytes': int(peak_memory[0]),
                        'peak_memory_reserved_bytes': int(peak_memory[1]),
                    }
                    if 'gist_grad_norm' in losses:
                        metric['gist_grad_norm'] = float(
                            losses['gist_grad_norm'].item())
                        metric['gist_update_max'] = float(
                            losses['gist_update_max'].item())
                    with self.metrics_path.open('a') as stream:
                        stream.write(json.dumps(metric) + '\n')
                    progress_bar.n += 1
                    progress_bar.set_postfix({
                        'latent_loss': f'{latent_loss_show:.4f}',
                        'action_loss': f'{action_loss_show:.4f}',
                        'step': self.step,
                        'grad_norm': f'{total_norm.item():.2f}',
                        'lr': f'{lr:.2e}'
                    })
                    if self.config.enable_wandb:
                        wandb_metric = {
                            'loss_metrics/global_avg_video_loss': latent_loss_show,
                            'loss_metrics/global_avg_action_loss': action_loss_show,
                            'loss_metrics/global_max_video_loss': max_latent_loss_show,
                            'loss_metrics/global_max_action_loss': max_action_loss_show,
                            'grad_norm': total_norm.item(),
                            'lr': lr,
                            'peak_memory_allocated_bytes': metric[
                                'peak_memory_allocated_bytes'],
                            'peak_memory_reserved_bytes': metric[
                                'peak_memory_reserved_bytes'],
                            'optimizer_step_seconds': metric[
                                'optimizer_step_seconds'],
                            'global_latent_frames_per_second': metric[
                                'global_latent_frames_per_second'],
                        }
                        if 'gist_grad_norm' in metric:
                            wandb_metric['gist_grad_norm'] = metric[
                                'gist_grad_norm']
                            wandb_metric['gist_update_max'] = metric[
                                'gist_update_max']
                        self.wandb.log(wandb_metric, step=self.step)
                
                self.step += 1
                # 是否到验证间隔50步；是否到保存间隔3000步；是否满足early stop
                validation_interval = int(getattr(self.config, "early_stop_check_interval", 0))
                should_stop = False
                if (bool(getattr(self.config, "enable_early_stop", False))
                        and validation_interval and self.step % validation_interval == 0):
                    validation_latent, validation_action = self._validation_loss()
                    if self.config.rank == 0:
                        record = {"step": self.step, "latent": validation_latent,
                                  "action": validation_action}
                        self.validation_history.append(record)
                        with (Path(self.config.save_root) / "validation_metrics.jsonl").open("a") as stream:
                            stream.write(json.dumps(record) + "\n")
                        if self.config.enable_wandb:
                            self.wandb.log({"validation/latent_loss": validation_latent,
                                            "validation/action_loss": validation_action},
                                           step=self.step)
                        should_stop = (
                            self.step >= int(getattr(self.config, "early_stop_min_steps", 0))
                            and self._validation_converged()
                        )
                    decision = [should_stop]
                    if dist.is_initialized():
                        dist.broadcast_object_list(decision, src=0)
                    should_stop = decision[0]
                
                if self.step % self.config.save_interval == 0:
                    if self.config.rank == 0:
                        logger.info(f"Starting save model at step {self.step}")
                    self.save_checkpoint()

                if should_stop:
                    stop_reason = "held_out_validation_losses_converged"
                    if self.config.rank == 0:
                        logger.info(f"Early stopping at step {self.step}: {stop_reason}")
                    self.save_checkpoint()
                    break

            if dist.is_initialized():
                dist.barrier()

        progress_bar.close()
        # A short smoke run may end before the periodic save interval. Always
        # persist its final optimizer step so the complete training path is tested.
        if stop_reason == "max_steps" and self.step % self.config.save_interval != 0:
            if self.config.rank == 0:
                logger.info(f"Saving final model at step {self.step}")
            self.save_checkpoint()
        if self.config.rank == 0:
            summary = {"stop_step": self.step, "stop_reason": stop_reason,
                       "validation_history": self.validation_history}
            (Path(self.config.save_root) / "training_summary.json").write_text(
                json.dumps(summary, indent=2) + "\n")
            if self.config.enable_wandb:
                self.wandb.summary.update({"stop_step": self.step, "stop_reason": stop_reason})
                self.wandb.finish()
        logger.info("Training completed!")


def run(args):
    """Main entry point."""
    config = copy.deepcopy(VA_CONFIGS[args.config_name])
    if getattr(config, "dataset_adapter", None) == "rmbench":
        stats_path = Path(config.dataset_path) / "rmbench_action_stats.json"
        try:
            with stats_path.open() as stream:
                stats = json.load(stream)
            config.norm_stat = {key: stats[key] for key in ("q01", "q99")}
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"RMBench action statistics not found: {stats_path}. "
                "Set RMBENCH_DATASET_PATH to a prepared dataset."
            ) from exc
        except KeyError as exc:
            raise ValueError(f"Invalid RMBench stats file {stats_path}: missing {exc}") from exc

    # TODO init distributed environment
    rank = int(os.getenv("RANK", 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    init_distributed(world_size, local_rank, rank)

    if os.environ.get("RMBENCH_DEBUGPY") == "1" and rank == 0:
        import debugpy

        port = int(os.environ.get("RMBENCH_DEBUGPY_PORT", "5678"))
        debugpy.listen(("0.0.0.0", port))
        print(f"[rank 0] waiting for debugger on port {port}", flush=True)
        debugpy.wait_for_client()
        print("[rank 0] debugger attached", flush=True)

    if dist.is_initialized():
        dist.barrier()

    config.rank = rank
    config.local_rank = local_rank
    config.world_size = world_size

    base_seed = int(getattr(config, "seed", 42))
    rank_seed = base_seed + rank
    random.seed(rank_seed)
    np.random.seed(rank_seed)
    torch.manual_seed(rank_seed)
    torch.cuda.manual_seed_all(rank_seed)

    if args.save_root is not None:
        config.save_root = args.save_root
    if args.num_steps is not None:
        config.num_steps = args.num_steps

    if rank == 0:
        logger.info(f"Using config: {args.config_name}")
        logger.info(f"World size: {world_size}, Local rank: {local_rank}")

    trainer = Trainer(config)
    trainer.train()
    if dist.is_initialized():
        dist.destroy_process_group()


def main():
    """Parse arguments and run training."""
    parser = argparse.ArgumentParser(description="Train WAN model for robotics")
    parser.add_argument(
        "--config-name",
        type=str,
        default='robotwin_train',
        help="Config name",
    )
    parser.add_argument(
        "--save-root",
        type=str,
        default=None,
        help="Root directory for saving checkpoints",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=None,
        help="Override the config's maximum optimizer steps (for example 10 for smoke testing)",
    )

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    init_logger()
    main()
