from __future__ import annotations

import torch
import unittest

from .memory_core import MemoryPolicy
from .toy_model import ToyMemoryModel


def _fixture(seed: int = 42, frames: int = 8):
    torch.manual_seed(seed)
    model = ToyMemoryModel(policy=MemoryPolicy.hybrid(2, 1, 2))
    sequence = [torch.randn(model.video_tokens_per_frame, model.hidden_dim)
                for _ in range(frames)]
    return model, sequence


def _assert_states_equal(left, right) -> None:
    assert left.episode_id == right.episode_id
    assert left.next_frame_id == right.next_frame_id
    assert left.version == right.version
    assert [(item.observation_id, item.frame_id) for item in left.receipts] == [
        (item.observation_id, item.frame_id) for item in right.receipts
    ]
    for a, b in zip(left.layers, right.layers):
        assert a.meta == b.meta
        torch.testing.assert_close(a.key, b.key, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(a.value, b.value, atol=1e-6, rtol=1e-5)


class StreamingTest(unittest.TestCase):
    def test_streaming_matches_vectorized_full_prefix(self) -> None:
        model, sequence = _fixture()
        stream_state, stream = model.prefill_streaming(
            sequence, episode_id="episode-0", detach=False)
        full_state, full = model.full_prefix(sequence, episode_id="episode-0")
        _assert_states_equal(stream_state, full_state)
        for a, b in zip(stream, full):
            torch.testing.assert_close(a.video_output, b.video_output,
                                       atol=1e-6, rtol=1e-5)
            torch.testing.assert_close(a.gist_output, b.gist_output,
                                       atol=1e-6, rtol=1e-5)

    def test_future_values_and_gradients_do_not_change_past(self) -> None:
        model, sequence = _fixture()
        sequence = [value.clone().requires_grad_(True) for value in sequence]
        _, original = model.full_prefix(sequence, episode_id="episode-0")
        past_value = original[3].video_output.square().mean()
        gradients = torch.autograd.grad(past_value, sequence, allow_unused=True)
        for gradient in gradients[4:]:
            self.assertTrue(gradient is None or torch.count_nonzero(gradient) == 0)

        changed = [value.detach().clone() for value in sequence]
        for value in changed[4:]:
            value.add_(100.0)
        _, modified = model.full_prefix(changed, episode_id="episode-0")
        for index in range(4):
            torch.testing.assert_close(original[index].video_output,
                                       modified[index].video_output)

    def test_global_frame_ids_not_storage_slots_drive_rope(self) -> None:
        model, sequence = _fixture(frames=4)
        _, normal = model.full_prefix(sequence, episode_id="episode-0")
        # 同样内容重复在另一episode仍应一致；frame ID改变则RoPE输出改变。
        _, other_episode = model.full_prefix(sequence, episode_id="episode-x")
        torch.testing.assert_close(normal[-1].video_output,
                                   other_episode[-1].video_output)
        reversed_sequence = list(reversed(sequence))
        _, reversed_result = model.full_prefix(reversed_sequence,
                                               episode_id="episode-0")
        self.assertFalse(torch.allclose(normal[-1].video_output,
                                       reversed_result[0].video_output))
