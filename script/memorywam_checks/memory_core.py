"""MemoryWAM toy/reference 的 typed token 与函数式记忆状态。

本模块固定机制合同，不承担 5B 模型接线。所有持久状态更新均返回新对象；
同一 observation ID 的相同提交是幂等操作，不同内容复用同一 ID 会报错。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
from typing import Iterable, Sequence

import torch


class TokenKind(str, Enum):
    CLEAN_VIDEO = "clean_video"
    GIST = "gist"
    ACTION = "action"
    NOISY_VIDEO = "noisy_video"


@dataclass(frozen=True)
class MemoryPolicy:
    """一组可消融的视觉记忆保留规则。"""

    gist_per_frame: int = 8
    anchor_frames: int = 2
    recent_frames: int = 4
    keep_gist: bool = True
    keep_anchor: bool = True
    keep_recent: bool = True
    full_history: bool = False

    def __post_init__(self) -> None:
        if min(self.gist_per_frame, self.anchor_frames, self.recent_frames) < 0:
            raise ValueError("memory budgets must be non-negative")
        if not self.keep_gist and self.gist_per_frame:
            object.__setattr__(self, "gist_per_frame", 0)

    @classmethod
    def hybrid(cls, gist_per_frame: int = 8, anchor_frames: int = 2,
               recent_frames: int = 4) -> "MemoryPolicy":
        return cls(gist_per_frame, anchor_frames, recent_frames)

    @classmethod
    def recent_only(cls, recent_frames: int = 4) -> "MemoryPolicy":
        return cls(0, 0, recent_frames, False, False, True, False)

    @classmethod
    def gist_only(cls, gist_per_frame: int = 8) -> "MemoryPolicy":
        return cls(gist_per_frame, 0, 0, True, False, False, False)

    @classmethod
    def full(cls) -> "MemoryPolicy":
        return cls(0, 0, 0, False, False, False, True)


@dataclass(frozen=True)
class TokenMeta:
    episode_id: str
    frame_id: int
    kind: TokenKind
    slot_in_frame: int
    target_group: str | None = None
    valid: bool = True

    @property
    def identity(self) -> tuple[str, int, str, int, str | None]:
        return (self.episode_id, self.frame_id, self.kind.value,
                self.slot_in_frame, self.target_group)


@dataclass(frozen=True)
class LayerKV:
    """单层已经过 RoPE 的 K 与 V，形状均为 ``[S,H,D]``。"""

    key: torch.Tensor
    value: torch.Tensor
    meta: tuple[TokenMeta, ...]

    def __post_init__(self) -> None:
        if self.key.ndim != 3 or self.key.shape != self.value.shape:
            raise ValueError("key/value must have identical [tokens, heads, dim] shape")
        if self.key.shape[0] != len(self.meta):
            raise ValueError("metadata and KV token counts differ")

    def select(self, indices: Sequence[int]) -> "LayerKV":
        index = torch.as_tensor(indices, dtype=torch.long, device=self.key.device)
        return LayerKV(self.key.index_select(0, index),
                       self.value.index_select(0, index),
                       tuple(self.meta[i] for i in indices))

    def append(self, other: "LayerKV") -> "LayerKV":
        if not self.meta:
            return other
        if not other.meta:
            return self
        return LayerKV(torch.cat((self.key, other.key)),
                       torch.cat((self.value, other.value)),
                       self.meta + other.meta)

    def detached(self) -> "LayerKV":
        return LayerKV(self.key.detach(), self.value.detach(), self.meta)


@dataclass(frozen=True)
class CommitReceipt:
    observation_id: str
    frame_id: int
    delta_digest: str


@dataclass(frozen=True)
class MemoryState:
    episode_id: str
    layers: tuple[LayerKV, ...]
    next_frame_id: int = 0
    version: int = 0
    receipts: tuple[CommitReceipt, ...] = ()

    def digest(self) -> str:
        digest = sha256()
        digest.update(repr((self.episode_id, self.next_frame_id, self.version,
                            self.receipts)).encode())
        for layer in self.layers:
            digest.update(repr(tuple(item.identity for item in layer.meta)).encode())
            for tensor in (layer.key, layer.value):
                value = tensor.detach().cpu().contiguous()
                digest.update(str((value.dtype, tuple(value.shape))).encode())
                digest.update(value.numpy().tobytes())
        return digest.hexdigest()

    def detached(self) -> "MemoryState":
        return MemoryState(self.episode_id,
                           tuple(layer.detached() for layer in self.layers),
                           self.next_frame_id, self.version, self.receipts)


@dataclass(frozen=True)
class MemoryDelta:
    episode_id: str
    frame_id: int
    observation_id: str
    layers: tuple[LayerKV, ...]
    base_version: int

    def digest(self) -> str:
        digest = sha256(repr((self.episode_id, self.frame_id,
                              self.observation_id)).encode())
        for layer in self.layers:
            digest.update(repr(tuple(item.identity for item in layer.meta)).encode())
            for tensor in (layer.key, layer.value):
                value = tensor.detach().cpu().contiguous()
                digest.update(str((value.dtype, tuple(value.shape))).encode())
                digest.update(value.numpy().tobytes())
        return digest.hexdigest()


def empty_state(episode_id: str, num_layers: int, heads: int, head_dim: int,
                *, device: torch.device | str = "cpu",
                dtype: torch.dtype = torch.float32) -> MemoryState:
    layers = tuple(
        LayerKV(torch.empty(0, heads, head_dim, device=device, dtype=dtype),
                torch.empty(0, heads, head_dim, device=device, dtype=dtype), ())
        for _ in range(num_layers)
    )
    return MemoryState(episode_id, layers)


def _keep_clean_video(frame_id: int, query_frame: int,
                      policy: MemoryPolicy) -> bool:
    if policy.full_history:
        return True
    anchor = policy.keep_anchor and frame_id < policy.anchor_frames
    recent_start = max(0, query_frame - policy.recent_frames + 1)
    recent = policy.keep_recent and policy.recent_frames > 0 and frame_id >= recent_start
    return anchor or recent


def token_visible(query: TokenMeta, key: TokenMeta,
                  policy: MemoryPolicy) -> bool:
    """Dense oracle 的单个边；任何未来帧或跨 episode 边均禁止。"""
    if not query.valid or not key.valid or query.episode_id != key.episode_id:
        return False
    if key.frame_id > query.frame_id:
        return False
    if key.kind is TokenKind.ACTION:
        return (query.kind is TokenKind.ACTION and
                key.frame_id == query.frame_id and
                key.target_group is not None and
                key.target_group == query.target_group)
    if key.kind is TokenKind.NOISY_VIDEO:
        return (query.kind is TokenKind.NOISY_VIDEO and
                key.frame_id == query.frame_id and
                key.target_group is not None and
                key.target_group == query.target_group)
    if key.kind is TokenKind.GIST:
        return policy.keep_gist
    if key.kind is TokenKind.CLEAN_VIDEO:
        # prefill 当前帧时，当前 clean tokens 始终可见；是否持久保留另由 policy 决定。
        return key.frame_id == query.frame_id or _keep_clean_video(
            key.frame_id, query.frame_id, policy)
    return False


def retain_layer(layer: LayerKV, current_frame: int,
                 policy: MemoryPolicy) -> LayerKV:
    indices: list[int] = []
    seen: set[tuple] = set()
    for index, meta in enumerate(layer.meta):
        keep = ((meta.kind is TokenKind.GIST and policy.keep_gist) or
                (meta.kind is TokenKind.CLEAN_VIDEO and
                 _keep_clean_video(meta.frame_id, current_frame, policy)))
        if keep and meta.valid and meta.frame_id <= current_frame:
            if meta.identity in seen:
                raise ValueError(f"duplicate token identity: {meta.identity}")
            seen.add(meta.identity)
            indices.append(index)
    return layer.select(indices)


def visible_history(layer: LayerKV, queries: Iterable[TokenMeta],
                    policy: MemoryPolicy) -> LayerKV:
    queries = tuple(queries)
    return layer.select([
        index for index, key in enumerate(layer.meta)
        if any(token_visible(query, key, policy) for query in queries)
    ])


def commit(state: MemoryState, delta: MemoryDelta, policy: MemoryPolicy,
           *, detach: bool) -> MemoryState:
    """原子提交；重复同一请求返回原对象，冲突重放拒绝。"""
    if delta.episode_id != state.episode_id:
        raise ValueError("episode mismatch")
    delta_digest = delta.digest()
    for receipt in state.receipts:
        if receipt.observation_id == delta.observation_id:
            if receipt.frame_id == delta.frame_id and receipt.delta_digest == delta_digest:
                return state
            raise ValueError("observation ID was reused with different content")
    if delta.base_version != state.version:
        raise ValueError("stale delta")
    if delta.frame_id != state.next_frame_id:
        raise ValueError("non-contiguous frame ID")
    if len(delta.layers) != len(state.layers):
        raise ValueError("layer count mismatch")
    layers = tuple(retain_layer(old.append(new), delta.frame_id, policy)
                   for old, new in zip(state.layers, delta.layers))
    if detach:
        layers = tuple(layer.detached() for layer in layers)
    receipt = CommitReceipt(delta.observation_id, delta.frame_id, delta_digest)
    return MemoryState(state.episode_id, layers, state.next_frame_id + 1,
                       state.version + 1, state.receipts + (receipt,))
