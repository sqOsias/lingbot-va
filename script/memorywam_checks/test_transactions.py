from __future__ import annotations

from dataclasses import replace

import torch
import unittest

from .memory_core import MemoryPolicy, commit
from .toy_model import ToyMemoryModel


def _one_delta():
    torch.manual_seed(7)
    model = ToyMemoryModel(policy=MemoryPolicy.hybrid(2, 1, 2))
    state = model.new_state("episode-0")
    frame = torch.randn(model.video_tokens_per_frame, model.hidden_dim)
    result = model.prefill_observation(frame, state, frame_id=0,
                                       observation_id="obs-0")
    return model, state, result


class TransactionTest(unittest.TestCase):
    def test_commit_is_functional_and_identical_retry_is_idempotent(self) -> None:
        model, state, result = _one_delta()
        before = state.digest()
        committed = commit(state, result.delta, model.policy, detach=True)
        self.assertEqual(state.digest(), before)
        retried = commit(committed, result.delta, model.policy, detach=True)
        self.assertIs(retried, committed)

    def test_conflicting_retry_stale_and_cross_episode_are_rejected(self) -> None:
        model, state, result = _one_delta()
        committed = commit(state, result.delta, model.policy, detach=True)
        changed_layer = replace(result.delta.layers[0],
                                key=result.delta.layers[0].key + 1)
        conflicting = replace(result.delta,
                              layers=(changed_layer,) + result.delta.layers[1:])
        with self.assertRaisesRegex(ValueError, "reused"):
            commit(committed, conflicting, model.policy, detach=True)
        stale = replace(result.delta, observation_id="obs-stale")
        with self.assertRaisesRegex(ValueError, "stale"):
            commit(committed, stale, model.policy, detach=True)
        with self.assertRaisesRegex(ValueError, "episode"):
            commit(state, replace(result.delta, episode_id="episode-x"),
                   model.policy, detach=True)

    def test_action_denoising_is_read_only_and_repeatable(self) -> None:
        torch.manual_seed(11)
        model = ToyMemoryModel(policy=MemoryPolicy.hybrid(2, 1, 2))
        frames = [torch.randn(model.video_tokens_per_frame, model.hidden_dim)
                  for _ in range(8)]
        state, _ = model.prefill_streaming(frames, episode_id="episode-0",
                                           detach=True)
        action = torch.randn(3, model.hidden_dim)
        before = state.digest()
        first = model.denoise_action(action, state, request_id="request-1",
                                     denoise_step=17)
        second = model.denoise_action(action, state, request_id="request-1",
                                      denoise_step=17)
        torch.testing.assert_close(first, second)
        for step in range(50):
            _ = model.denoise_action(action, state, request_id="request-2",
                                     denoise_step=step)
        self.assertEqual(state.digest(), before)
