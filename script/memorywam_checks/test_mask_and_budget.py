from __future__ import annotations

import torch
import unittest

from .mask_oracle import build_visibility
from .memory_core import (LayerKV, MemoryPolicy, TokenKind, TokenMeta,
                          retain_layer)


def _three_frame_meta() -> tuple[TokenMeta, ...]:
    result = []
    for frame in range(3):
        result.extend((
            TokenMeta("episode-0", frame, TokenKind.CLEAN_VIDEO, 0),
            TokenMeta("episode-0", frame, TokenKind.GIST, 0),
            TokenMeta("episode-0", frame, TokenKind.ACTION, 0, f"a{frame}"),
        ))
    return tuple(result)


class MaskAndBudgetTest(unittest.TestCase):
    def test_paper_figure_three_style_mask(self) -> None:
        policy = MemoryPolicy.hybrid(gist_per_frame=1, anchor_frames=1,
                                     recent_frames=1)
        meta = _three_frame_meta()
        actual = build_visibility(meta, meta, policy)
        expected_rows = (
            {0, 1}, {0, 1}, {0, 1, 2},
            {0, 1, 3, 4}, {0, 1, 3, 4}, {0, 1, 3, 4, 5},
            {0, 1, 4, 6, 7}, {0, 1, 4, 6, 7}, {0, 1, 4, 6, 7, 8},
        )
        expected = torch.zeros_like(actual)
        for row, columns in enumerate(expected_rows):
            expected[row, list(columns)] = True
        torch.testing.assert_close(actual, expected)

    def test_cross_episode_future_and_invalid_are_hidden(self) -> None:
        policy = MemoryPolicy.hybrid(gist_per_frame=1, anchor_frames=1,
                                     recent_frames=1)
        query = TokenMeta("e0", 2, TokenKind.ACTION, 0, "a2")
        keys = (
            TokenMeta("e1", 0, TokenKind.GIST, 0),
            TokenMeta("e0", 3, TokenKind.GIST, 0),
            TokenMeta("e0", 0, TokenKind.GIST, 0, valid=False),
        )
        self.assertFalse(bool(build_visibility((query,), keys, policy).any()))

    def test_hybrid_budget_and_no_duplicate_overlap(self) -> None:
        policy = MemoryPolicy.hybrid(gist_per_frame=2, anchor_frames=2,
                                     recent_frames=4)
        video_tokens = 3
        for frame_count in (1, 2, 4, 6, 7, 10, 100, 1000):
            meta = []
            for frame in range(frame_count):
                meta.extend(TokenMeta("e", frame, TokenKind.CLEAN_VIDEO, i)
                            for i in range(video_tokens))
                meta.extend(TokenMeta("e", frame, TokenKind.GIST, i)
                            for i in range(policy.gist_per_frame))
            tensor = torch.arange(len(meta), dtype=torch.float32).view(-1, 1, 1)
            retained = retain_layer(LayerKV(tensor, tensor.clone(), tuple(meta)),
                                    frame_count - 1, policy)
            full_frames = set(range(min(policy.anchor_frames, frame_count)))
            full_frames.update(range(max(0, frame_count - policy.recent_frames),
                                     frame_count))
            clean = [m for m in retained.meta if m.kind is TokenKind.CLEAN_VIDEO]
            gist = [m for m in retained.meta if m.kind is TokenKind.GIST]
            self.assertEqual(len(clean), len(full_frames) * video_tokens)
            self.assertEqual(len(gist), frame_count * policy.gist_per_frame)
            self.assertEqual(len({m.identity for m in retained.meta}),
                             len(retained.meta))

    def test_ablation_policies_have_distinct_storage(self) -> None:
        meta = tuple(
            TokenMeta("e", frame, kind, slot)
            for frame in range(6)
            for kind, count in ((TokenKind.CLEAN_VIDEO, 2),
                                (TokenKind.GIST, 1))
            for slot in range(count)
        )
        value = torch.zeros(len(meta), 1, 2)
        layer = LayerKV(value, value.clone(), meta)
        counts = {}
        for name, policy in {
            "hybrid": MemoryPolicy.hybrid(1, 1, 2),
            "recent": MemoryPolicy.recent_only(2),
            "gist": MemoryPolicy.gist_only(1),
            "full": MemoryPolicy.full(),
        }.items():
            counts[name] = retain_layer(layer, 5, policy).key.shape[0]
        self.assertEqual(counts, {
            "hybrid": 12,
            "recent": 4,
            "gist": 6,
            "full": 12,
        })
