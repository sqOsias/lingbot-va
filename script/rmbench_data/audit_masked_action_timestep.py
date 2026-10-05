#!/usr/bin/env python3
import json
from pathlib import Path
import sys
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from wan_va.train import Trainer

class Scheduler:
    num_train_timesteps = 1000
    timesteps = torch.arange(1000, dtype=torch.float32)
    def add_noise(self, latent, noise, timesteps, t_dim=2):
        sigma = (timesteps / 999.0).view(1, 1, -1, 1, 1)
        return (1 - sigma) * latent + sigma * noise
    def training_target(self, latent, noise, timesteps):
        return noise - latent

trainer = Trainer.__new__(Trainer)
trainer.patch_size = (1, 2, 2)
trainer.device = torch.device('cpu')
latent = torch.randn(2, 30, 3, 16, 1)
mask = torch.ones_like(latent, dtype=torch.bool)
mask[:, :, 0] = False
mask[1, :, 2] = False
mask[0, 0, 1] = False  # partial channel mask: frame remains a diffusion frame

torch.manual_seed(20260915)
baseline = trainer._add_noise(latent.clone(), Scheduler(), action_mask=None,
                              action_mode=True, noisy_cond_prob=0.0)
torch.manual_seed(20260915)
masked = trainer._add_noise(latent.clone(), Scheduler(), action_mask=mask,
                            action_mode=True, noisy_cond_prob=0.0)
valid_frames = mask.any(dim=(1, 3, 4))
unmasked_elements = mask
checks = {
    'timestep_shape_BF': list(masked['timesteps'].shape) == [2, 3],
    'fully_masked_timesteps_zero': bool((masked['timesteps'][~valid_frames] == 0).all()),
    'valid_timesteps_unchanged': bool(torch.equal(masked['timesteps'][valid_frames], baseline['timesteps'][valid_frames])),
    'partial_mask_frame_keeps_sampled_timestep': bool(masked['timesteps'][0, 1] == baseline['timesteps'][0, 1]),
    'fully_masked_noisy_values_zero': bool((masked['noisy_latents'].permute(0,2,1,3,4)[~valid_frames] == 0).all()),
    'fully_masked_targets_zero': bool((masked['targets'].permute(0,2,1,3,4)[~valid_frames] == 0).all()),
    'unmasked_noise_values_unchanged': bool(torch.equal(masked['noisy_latents'][unmasked_elements], baseline['noisy_latents'][unmasked_elements])),
    'unmasked_targets_unchanged': bool(torch.equal(masked['targets'][unmasked_elements], baseline['targets'][unmasked_elements])),
    'condition_timesteps_zero': bool((masked['cond_timesteps'] == 0).all()),
}
report = {
    'status': 'PASS' if all(checks.values()) else 'FAIL',
    'checks': checks,
    'sampled_timesteps_without_mask': baseline['timesteps'].tolist(),
    'timesteps_with_mask': masked['timesteps'].tolist(),
    'valid_action_frames': valid_frames.tolist(),
}
out = Path('artifacts/temporal_alignment_fix/masked_action_timestep_audit.json')
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report))
raise SystemExit(0 if report['status'] == 'PASS' else 1)
