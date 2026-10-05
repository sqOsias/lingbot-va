from __future__ import annotations

import unittest
from dataclasses import replace
from types import SimpleNamespace
from tempfile import TemporaryDirectory

import torch

from wan_va.distributed.fsdp import apply_ac
from wan_va.modules.memorywam import commit_memory
from wan_va.modules.model import WanTransformer3DModel
from wan_va.wan_va_server import VA_Server


def _video_grid(frame_id: int) -> torch.Tensor:
    values = []
    for height in range(2):
        for width in range(2):
            values.append((frame_id, height, width, 0))
    return torch.tensor(values, dtype=torch.float32).T.unsqueeze(0)


def _action_grid(frame_id: int) -> torch.Tensor:
    values = [(frame_id + (index + 1) / 4, -1, -1, 1)
              for index in range(3)]
    return torch.tensor(values, dtype=torch.float32).T.unsqueeze(0)


class ProductionMemoryWAMTest(unittest.TestCase):
    def _model(self, enabled: bool = True) -> WanTransformer3DModel:
        torch.manual_seed(123)
        return WanTransformer3DModel(
            patch_size=[1, 1, 1], num_attention_heads=2,
            attention_head_dim=8, in_channels=4, out_channels=4,
            action_dim=6, text_dim=8, freq_dim=8, ffn_dim=32,
            num_layers=2, cross_attn_norm=True, eps=1e-6,
            rope_max_seq_len=32, pos_embed_seq_len=None, attn_mode="torch",
            memorywam_enabled=enabled, memorywam_gist_tokens=2,
            memorywam_anchor_frames=1, memorywam_recent_frames=1)

    def test_disabled_model_has_no_new_checkpoint_parameter(self) -> None:
        model = self._model(enabled=False)
        self.assertIsNone(model.gist_tokens)
        self.assertFalse(any(name == "gist_tokens"
                             for name in model.state_dict()))

    def test_memory_checkpoint_round_trip_preserves_policy_and_gist(self) -> None:
        model = self._model()
        with TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            restored = WanTransformer3DModel.from_pretrained(directory)
        self.assertEqual(restored.memorywam_policy, model.memorywam_policy)
        torch.testing.assert_close(restored.gist_tokens, model.gist_tokens,
                                   rtol=0, atol=0)

    def test_all_production_ablation_modes_are_checkpointed(self) -> None:
        expected = {
            "hybrid": (2, True, True, True, False),
            "recent-only": (0, False, False, True, False),
            "gist-only": (2, True, False, False, False),
            "full-history": (0, False, False, False, True),
        }
        for mode, values in expected.items():
            with self.subTest(mode=mode):
                model = self._model(enabled=False)
                model.enable_memorywam(2, 1, 1, mode)
                policy = model.memorywam_policy
                self.assertEqual((policy.gist_per_frame, policy.keep_gist,
                                  policy.keep_anchor, policy.keep_recent,
                                  policy.full_history), values)
                if values[0]:
                    self.assertEqual(model.gist_tokens.shape[0], values[0])
                else:
                    self.assertIsNone(model.gist_tokens)
                self.assertEqual(model.config.memorywam_mode, mode)
                self.assertEqual("gist_tokens" in model.state_dict(),
                                 values[0] > 0)
                with TemporaryDirectory() as directory:
                    model.save_pretrained(directory)
                    restored = WanTransformer3DModel.from_pretrained(
                        directory)
                self.assertEqual(restored.memorywam_policy, policy)
                self.assertEqual(restored.config.memorywam_mode, mode)
                self.assertEqual("gist_tokens" in restored.state_dict(),
                                 values[0] > 0)
                state = restored.create_memorywam_state(
                    f"episode-{mode}", 1)
                text = torch.randn(1, 5, 8)
                _, delta = restored.memorywam_prefill_observation(
                    {
                        "noisy_latents": torch.randn(1, 4, 1, 2, 2),
                        "timesteps": torch.zeros(1, 1),
                        "grid_id": _video_grid(0),
                        "text_emb": text,
                    }, state, frame_id=0, observation_id=f"obs-{mode}")
                state = commit_memory(
                    state, delta, restored.memorywam_policy, detach=True)
                output = restored.memorywam_denoise(
                    {
                        "noisy_latents": torch.randn(1, 6, 1, 3, 1),
                        "timesteps": torch.full((1, 1), 500),
                        "grid_id": _action_grid(0),
                        "text_emb": text,
                    }, state, action_mode=True)
                self.assertTrue(bool(torch.isfinite(output).all()))

    def test_prefill_commit_and_read_only_denoise_train_gist(self) -> None:
        model = self._model()
        text = torch.randn(1, 5, 8)
        state = model.create_memorywam_state("episode-0", 1)
        prefill = {
            "noisy_latents": torch.randn(1, 4, 1, 2, 2),
            "timesteps": torch.zeros(1, 1),
            "grid_id": _video_grid(0),
            "text_emb": text,
        }
        _, delta = model.memorywam_prefill_observation(
            prefill, state, frame_id=0, observation_id="obs-0")
        state = commit_memory(state, delta, model.memorywam_policy,
                              detach=False)
        self.assertIs(commit_memory(state, delta, model.memorywam_policy,
                                    detach=False), state)
        with self.assertRaisesRegex(ValueError, "different content"):
            commit_memory(
                state, replace(delta, content_digest="conflict"),
                model.memorywam_policy, detach=False)
        before = (state.version, state.next_frame_id,
                  tuple(layer.key.shape for layer in state.layers))
        action = {
            "noisy_latents": torch.randn(1, 6, 1, 3, 1),
            "timesteps": torch.full((1, 1), 500),
            "grid_id": _action_grid(0),
            "text_emb": text,
        }
        first = model.memorywam_denoise(action, state, action_mode=True)
        second = model.memorywam_denoise(action, state, action_mode=True)
        torch.testing.assert_close(first, second)
        after = (state.version, state.next_frame_id,
                 tuple(layer.key.shape for layer in state.layers))
        self.assertEqual(before, after)
        gradient, = torch.autograd.grad(first.square().mean(),
                                        model.gist_tokens)
        self.assertTrue(bool(torch.isfinite(gradient).all()))
        self.assertGreater(float(gradient.abs().sum()), 0.0)

    def test_service_prefill_commits_one_real_observation(self) -> None:
        model = WanTransformer3DModel(
            patch_size=[1, 2, 2], num_attention_heads=2,
            attention_head_dim=8, in_channels=4, out_channels=4,
            action_dim=6, text_dim=8, freq_dim=8, ffn_dim=32,
            num_layers=2, cross_attn_norm=True, eps=1e-6,
            rope_max_seq_len=32, pos_embed_seq_len=None, attn_mode="torch",
            memorywam_enabled=True, memorywam_gist_tokens=2,
            memorywam_anchor_frames=1, memorywam_recent_frames=1)
        server = VA_Server.__new__(VA_Server)
        server.transformer = model
        server.device = torch.device("cpu")
        server.dtype = torch.float32
        server.use_cfg = False
        server.prompt_embeds = torch.randn(1, 5, 8)
        server.memorywam_episode_id = "episode-service"
        server.memorywam_state = model.create_memorywam_state(
            server.memorywam_episode_id, 1, device=server.device,
            dtype=server.dtype)
        server.frame_st_id = 0
        server.job_config = SimpleNamespace(obs_cam_keys=("camera",))
        latent = torch.randn(1, 4, 1, 2, 2)
        server._memorywam_commit_latents(
            latent, "observation-0", global_frame_id=0,
            observation_digest="raw-image-digest")
        self.assertEqual(server.memorywam_state.next_frame_id, 1)
        self.assertEqual(server.memorywam_state.version, 1)
        self.assertEqual(server.frame_st_id, 1)
        receipt = server._memorywam_observation_receipt("observation-0")
        self.assertIsNotNone(receipt)
        self.assertEqual(receipt.digest, "raw-image-digest")

    def test_causal_full_episode_training_path(self) -> None:
        model = self._model().to(torch.bfloat16)
        apply_ac(model)
        frames = 3
        text = torch.randn(1, 5, 8, dtype=torch.bfloat16)
        video_grid = torch.cat([_video_grid(frame) for frame in range(frames)],
                               dim=2)
        action_grid = torch.cat([_action_grid(frame) for frame in range(frames)],
                                dim=2)
        actions_mask = torch.ones(1, 6, frames, 3, 1, dtype=torch.bool)
        actions_mask[:, :, 0] = False
        input_dict = {
            "latent_dict": {
                "noisy_latents": torch.randn(
                    1, 4, frames, 2, 2, dtype=torch.bfloat16),
                "latent": torch.randn(
                    1, 4, frames, 2, 2, dtype=torch.bfloat16),
                "timesteps": torch.full((1, frames), 500),
                "grid_id": video_grid,
                "text_emb": text,
            },
            "action_dict": {
                "noisy_latents": torch.randn(
                    1, 6, frames, 3, 1, dtype=torch.bfloat16),
                "latent": torch.zeros(
                    1, 6, frames, 3, 1, dtype=torch.bfloat16),
                "timesteps": torch.full((1, frames), 500),
                "grid_id": action_grid,
                "text_emb": text,
                "actions_mask": actions_mask,
            },
            "memory_meta": {
                "dataset_numeric_id": torch.tensor([0]),
                "episode_index": torch.tensor([7]),
                "global_frame_ids": torch.arange(frames).unsqueeze(0),
                "video_target_mask": torch.tensor([[False, True, True]]),
                "detach_interval": 0,
            },
        }
        video, action = model(input_dict, train_mode=True)
        self.assertEqual(video.shape, (1, frames * 4, 4))
        self.assertEqual(action.shape, (1, frames * 3, 6))
        loss = video[:, 4:].float().square().mean()
        loss = loss + action[:, 3:].float().square().mean()
        gradient, = torch.autograd.grad(loss, model.gist_tokens)
        self.assertTrue(bool(torch.isfinite(gradient).all()))
        self.assertGreater(float(gradient.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
