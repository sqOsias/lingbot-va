"""带 temporal RoPE 的小型共享骨干，用于机制验证。

``prefill_streaming`` 每次只计算一个新帧；``full_prefix`` 在每一层一次性计算
整个因果前缀。两条路径共享参数和数学定义，但不共享执行拓扑。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from .mask_oracle import build_visibility
from .memory_core import (CommitReceipt, LayerKV, MemoryDelta, MemoryPolicy,
                          MemoryState, TokenKind, TokenMeta, commit,
                          empty_state, retain_layer, visible_history)


def _temporal_rope(x: torch.Tensor, meta: tuple[TokenMeta, ...]) -> torch.Tensor:
    """按 global frame ID 对 ``[S,H,D]`` 的 Q/K 应用简化 temporal RoPE。"""
    if x.shape[-1] % 2:
        raise ValueError("RoPE head_dim must be even")
    half = x.shape[-1] // 2
    position = torch.tensor([m.frame_id for m in meta], device=x.device,
                            dtype=torch.float32)[:, None, None]
    frequency = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=x.device,
                                          dtype=torch.float32) / max(half, 1)
    )[None, None, :]
    angle = position * frequency
    cos, sin = angle.cos().to(x.dtype), angle.sin().to(x.dtype)
    left, right = x[..., :half], x[..., half:]
    return torch.cat((left * cos - right * sin,
                      right * cos + left * sin), dim=-1)


class ToyBlock(nn.Module):
    def __init__(self, hidden_dim: int, heads: int) -> None:
        super().__init__()
        if hidden_dim % heads:
            raise ValueError("hidden_dim must be divisible by heads")
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.head_dim = hidden_dim // heads
        self.norm = nn.LayerNorm(hidden_dim)
        self.q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(nn.Linear(hidden_dim, hidden_dim * 2),
                                 nn.GELU(),
                                 nn.Linear(hidden_dim * 2, hidden_dim))

    def project(self, hidden: torch.Tensor,
                meta: tuple[TokenMeta, ...]) -> tuple[torch.Tensor, ...]:
        normalized = self.norm(hidden)
        shape = (hidden.shape[0], self.heads, self.head_dim)
        query = _temporal_rope(self.q(normalized).view(shape), meta)
        key = _temporal_rope(self.k(normalized).view(shape), meta)
        value = self.v(normalized).view(shape)
        return query, key, value

    def finish(self, hidden: torch.Tensor, attended: torch.Tensor) -> torch.Tensor:
        hidden = hidden + self.out(attended.reshape(hidden.shape))
        return hidden + self.ffn(self.ffn_norm(hidden))


@dataclass(frozen=True)
class PrefillResult:
    video_output: torch.Tensor
    gist_output: torch.Tensor
    delta: MemoryDelta


class ToyMemoryModel(nn.Module):
    def __init__(self, *, hidden_dim: int = 32, heads: int = 4,
                 layers: int = 2, video_tokens_per_frame: int = 6,
                 policy: MemoryPolicy | None = None) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.head_dim = hidden_dim // heads
        self.video_tokens_per_frame = video_tokens_per_frame
        self.policy = policy or MemoryPolicy.hybrid(gist_per_frame=2,
                                                     anchor_frames=1,
                                                     recent_frames=2)
        self.blocks = nn.ModuleList([ToyBlock(hidden_dim, heads)
                                     for _ in range(layers)])
        self.gist_tokens = nn.Parameter(
            torch.randn(self.policy.gist_per_frame, hidden_dim) * 0.02
        )

    @property
    def tokens_per_frame(self) -> int:
        return self.video_tokens_per_frame + self.policy.gist_per_frame

    def new_state(self, episode_id: str) -> MemoryState:
        return empty_state(episode_id, len(self.blocks), self.heads,
                           self.head_dim, device=self.gist_tokens.device,
                           dtype=self.gist_tokens.dtype)

    def _frame_meta(self, episode_id: str, frame_id: int) -> tuple[TokenMeta, ...]:
        video = tuple(TokenMeta(episode_id, frame_id, TokenKind.CLEAN_VIDEO, i)
                      for i in range(self.video_tokens_per_frame))
        gist = tuple(TokenMeta(episode_id, frame_id, TokenKind.GIST, i)
                     for i in range(self.policy.gist_per_frame))
        return video + gist

    def _frame_hidden(self, frame: torch.Tensor) -> torch.Tensor:
        expected = (self.video_tokens_per_frame, self.hidden_dim)
        if frame.shape != expected:
            raise ValueError(f"expected frame {expected}, got {tuple(frame.shape)}")
        if not self.policy.gist_per_frame:
            return frame
        return torch.cat((frame, self.gist_tokens), dim=0)

    def _attention(self, query: torch.Tensor, key: torch.Tensor,
                   value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        logits = torch.einsum("qhd,khd->hqk", query, key)
        logits = logits / math.sqrt(self.head_dim)
        logits = logits.masked_fill(~mask.unsqueeze(0), float("-inf"))
        if (~mask.any(dim=1)).any():
            raise ValueError("a query has no visible key")
        weights = torch.softmax(logits, dim=-1)
        return torch.einsum("hqk,khd->qhd", weights, value)

    def _stream_layer(self, block: ToyBlock, hidden: torch.Tensor,
                      current_meta: tuple[TokenMeta, ...],
                      history: LayerKV) -> tuple[torch.Tensor, LayerKV]:
        query, key, value = block.project(hidden, current_meta)
        current = LayerKV(key, value, current_meta)
        all_kv = history.append(current)
        mask = build_visibility(current_meta, all_kv.meta, self.policy,
                                device=hidden.device)
        attended = self._attention(query, all_kv.key, all_kv.value, mask)
        return block.finish(hidden, attended), current

    def prefill_observation(self, frame: torch.Tensor, state: MemoryState, *,
                            frame_id: int, observation_id: str) -> PrefillResult:
        if frame_id != state.next_frame_id:
            raise ValueError("prefill frame must equal state.next_frame_id")
        meta = self._frame_meta(state.episode_id, frame_id)
        hidden = self._frame_hidden(frame)
        layers: list[LayerKV] = []
        for index, block in enumerate(self.blocks):
            history = visible_history(state.layers[index], meta, self.policy)
            hidden, current = self._stream_layer(block, hidden, meta, history)
            layers.append(current)
        split = self.video_tokens_per_frame
        return PrefillResult(hidden[:split], hidden[split:],
                             MemoryDelta(state.episode_id, frame_id,
                                         observation_id, tuple(layers),
                                         state.version))

    def prefill_streaming(self, frames: list[torch.Tensor], *, episode_id: str,
                          detach: bool = False
                          ) -> tuple[MemoryState, list[PrefillResult]]:
        state = self.new_state(episode_id)
        results: list[PrefillResult] = []
        for frame_id, frame in enumerate(frames):
            result = self.prefill_observation(
                frame, state, frame_id=frame_id,
                observation_id=f"{episode_id}:obs:{frame_id}")
            state = commit(state, result.delta, self.policy, detach=detach)
            results.append(result)
        return state, results

    def full_prefix(self, frames: list[torch.Tensor], *, episode_id: str
                    ) -> tuple[MemoryState, list[PrefillResult]]:
        """每层一次性计算所有帧，作为 streaming 的独立拓扑 reference。"""
        if not frames:
            return self.new_state(episode_id), []
        meta = tuple(item for frame_id in range(len(frames))
                     for item in self._frame_meta(episode_id, frame_id))
        hidden = torch.cat([self._frame_hidden(frame) for frame in frames], dim=0)
        all_layers: list[LayerKV] = []
        for block in self.blocks:
            query, key, value = block.project(hidden, meta)
            mask = build_visibility(meta, meta, self.policy, device=hidden.device)
            attended = self._attention(query, key, value, mask)
            all_layers.append(LayerKV(key, value, meta))
            hidden = block.finish(hidden, attended)

        per_frame_layers: list[tuple[LayerKV, ...]] = []
        for frame_id in range(len(frames)):
            start, end = frame_id * self.tokens_per_frame, (frame_id + 1) * self.tokens_per_frame
            per_frame_layers.append(tuple(layer.select(range(start, end))
                                          for layer in all_layers))
        receipts = []
        results = []
        for frame_id, layers in enumerate(per_frame_layers):
            start, end = frame_id * self.tokens_per_frame, (frame_id + 1) * self.tokens_per_frame
            frame_hidden = hidden[start:end]
            observation_id = f"{episode_id}:obs:{frame_id}"
            delta = MemoryDelta(episode_id, frame_id, observation_id,
                                layers, frame_id)
            receipts.append(CommitReceipt(observation_id, frame_id,
                                           delta.digest()))
            split = self.video_tokens_per_frame
            results.append(PrefillResult(frame_hidden[:split],
                                         frame_hidden[split:], delta))
        final_frame = len(frames) - 1
        state = MemoryState(
            episode_id,
            tuple(retain_layer(layer, final_frame, self.policy)
                  for layer in all_layers),
            len(frames), len(frames), tuple(receipts),
        )
        return state, results

    def denoise_action(self, noisy_action: torch.Tensor, state: MemoryState, *,
                       request_id: str, denoise_step: int) -> torch.Tensor:
        if state.next_frame_id == 0:
            raise ValueError("action denoise requires an observed frame")
        if noisy_action.ndim != 2 or noisy_action.shape[1] != self.hidden_dim:
            raise ValueError("action tokens must have shape [A, hidden_dim]")
        frame_id = state.next_frame_id - 1
        group = f"{request_id}:tau:{denoise_step}"
        meta = tuple(TokenMeta(state.episode_id, frame_id, TokenKind.ACTION,
                               index, group)
                     for index in range(noisy_action.shape[0]))
        # 一个确定性的 timestep signal；它不是环境 frame ID，也不会写入 cache。
        hidden = noisy_action + float(denoise_step) / 1000.0
        for index, block in enumerate(self.blocks):
            history = visible_history(state.layers[index], meta, self.policy)
            hidden, _ = self._stream_layer(block, hidden, meta, history)
        return hidden
