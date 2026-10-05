from __future__ import annotations

import torch
import unittest

from .memory_core import MemoryPolicy
from .toy_model import ToyMemoryModel


def _model_and_frames(seed: int):
    torch.manual_seed(seed)
    model = ToyMemoryModel(policy=MemoryPolicy.gist_only(2))
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name == "gist_tokens")
    frames = [torch.randn(model.video_tokens_per_frame, model.hidden_dim)
              for _ in range(6)]
    return model, frames


class GistGradientTest(unittest.TestCase):
    def test_gist_gets_gradient_from_future_video_loss(self) -> None:
        model, frames = _model_and_frames(21)
        _, outputs = model.full_prefix(frames, episode_id="episode-0")
        target = torch.zeros_like(outputs[-1].video_output)
        loss = torch.nn.functional.mse_loss(outputs[-1].video_output, target)
        gradient, = torch.autograd.grad(loss, model.gist_tokens)
        self.assertTrue(bool(torch.isfinite(gradient).all()))
        self.assertGreater(float(gradient.abs().sum()), 0)

    def test_gist_gets_gradient_from_action_loss(self) -> None:
        model, frames = _model_and_frames(22)
        state, _ = model.prefill_streaming(frames, episode_id="episode-0",
                                           detach=False)
        action = torch.zeros(3, model.hidden_dim)
        prediction = model.denoise_action(action, state, request_id="train",
                                          denoise_step=500)
        loss = prediction.square().mean()
        gradient, = torch.autograd.grad(loss, model.gist_tokens)
        self.assertTrue(bool(torch.isfinite(gradient).all()))
        self.assertGreater(float(gradient.abs().sum()), 0)

    def test_detached_state_truncates_cross_frame_gradient(self) -> None:
        model, frames = _model_and_frames(23)
        state, _ = model.prefill_streaming(frames, episode_id="episode-0",
                                           detach=True)
        prediction = model.denoise_action(torch.zeros(3, model.hidden_dim), state,
                                          request_id="inference", denoise_step=1)
        self.assertFalse(prediction.requires_grad)
