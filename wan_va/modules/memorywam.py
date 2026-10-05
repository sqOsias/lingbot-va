"""Typed, transactional persistent memory for the production Transformer.

The state is owned by the caller.  Denoising reads it without mutation;
observed clean frames produce a delta that is committed exactly once.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from hashlib import sha256

import torch


class MemoryTokenKind(IntEnum):
    CLEAN_VIDEO = 0
    GIST = 1


@dataclass(frozen=True)
class MemoryWAMPolicy:
    gist_per_frame: int = 8
    anchor_frames: int = 2
    recent_frames: int = 4
    keep_gist: bool = True
    keep_anchor: bool = True
    keep_recent: bool = True
    full_history: bool = False

    def __post_init__(self) -> None:
        if min(self.gist_per_frame, self.anchor_frames,
               self.recent_frames) < 0:
            raise ValueError("memory budgets must be non-negative")

    @classmethod
    def hybrid(cls, gist_per_frame: int = 8, anchor_frames: int = 2,
               recent_frames: int = 4) -> "MemoryWAMPolicy":
        return cls(gist_per_frame, anchor_frames, recent_frames)

    @classmethod
    def recent_only(cls, recent_frames: int = 4) -> "MemoryWAMPolicy":
        return cls(0, 0, recent_frames, False, False, True, False)

    @classmethod
    def gist_only(cls, gist_per_frame: int = 8) -> "MemoryWAMPolicy":
        return cls(gist_per_frame, 0, 0, True, False, False, False)

    @classmethod
    def full(cls) -> "MemoryWAMPolicy":
        return cls(0, 0, 0, False, False, False, True)

    @classmethod
    def from_mode(cls, mode: str, *, gist_per_frame: int = 8,
                  anchor_frames: int = 2,
                  recent_frames: int = 4) -> "MemoryWAMPolicy":
        normalized = str(mode).strip().lower().replace("_", "-")
        if normalized == "hybrid":
            return cls.hybrid(gist_per_frame, anchor_frames, recent_frames)
        if normalized == "recent-only":
            return cls.recent_only(recent_frames)
        if normalized == "gist-only":
            return cls.gist_only(gist_per_frame)
        if normalized in ("full", "full-history"):
            return cls.full()
        raise ValueError(
            "memory mode must be one of hybrid, recent-only, gist-only, "
            f"full-history; got {mode!r}")


@dataclass(frozen=True)
class MemoryLayerKV:
    key: torch.Tensor
    value: torch.Tensor
    frame_ids: torch.Tensor
    kinds: torch.Tensor
    slots: torch.Tensor

    def __post_init__(self) -> None:
        if self.key.ndim != 4 or self.key.shape != self.value.shape:
            raise ValueError("K/V must have identical [B,S,H,D] shape")
        size = self.key.shape[1]
        for name, item in (("frame_ids", self.frame_ids),
                           ("kinds", self.kinds), ("slots", self.slots)):
            if item.ndim != 1 or item.numel() != size:
                raise ValueError(f"{name} must have shape [S]")

    def select(self, indices: torch.Tensor) -> "MemoryLayerKV":
        indices = indices.to(device=self.key.device, dtype=torch.long)
        meta_indices = indices.to(self.frame_ids.device)
        return MemoryLayerKV(
            self.key.index_select(1, indices),
            self.value.index_select(1, indices),
            self.frame_ids.index_select(0, meta_indices),
            self.kinds.index_select(0, meta_indices),
            self.slots.index_select(0, meta_indices),
        )

    def append(self, other: "MemoryLayerKV") -> "MemoryLayerKV":
        if self.key.shape[0] != other.key.shape[0]:
            raise ValueError("memory batch size changed")
        return MemoryLayerKV(
            torch.cat((self.key, other.key), dim=1),
            torch.cat((self.value, other.value), dim=1),
            torch.cat((self.frame_ids, other.frame_ids)),
            torch.cat((self.kinds, other.kinds)),
            torch.cat((self.slots, other.slots)),
        )

    def detached(self) -> "MemoryLayerKV":
        return MemoryLayerKV(self.key.detach(), self.value.detach(),
                             self.frame_ids, self.kinds, self.slots)


@dataclass(frozen=True)
class MemoryCommitReceipt:
    observation_id: str
    frame_id: int
    digest: str


@dataclass(frozen=True)
class MemoryWAMState:
    episode_id: str
    layers: tuple[MemoryLayerKV, ...]
    next_frame_id: int = 0
    version: int = 0
    receipts: tuple[MemoryCommitReceipt, ...] = ()

    def detached(self) -> "MemoryWAMState":
        return MemoryWAMState(self.episode_id,
                              tuple(layer.detached() for layer in self.layers),
                              self.next_frame_id, self.version, self.receipts)


@dataclass(frozen=True)
class MemoryWAMDelta:
    episode_id: str
    frame_id: int
    observation_id: str
    layers: tuple[MemoryLayerKV, ...]
    base_version: int
    content_digest: str | None = None

    def digest(self) -> str:
        if self.content_digest is not None:
            return self.content_digest
        digest = sha256(
            repr((self.episode_id, self.frame_id, self.observation_id)).encode())
        for layer in self.layers:
            for meta in (layer.frame_ids, layer.kinds, layer.slots):
                value = meta.detach().cpu().contiguous()
                digest.update(value.view(torch.uint8).numpy().tobytes())
            for tensor in (layer.key, layer.value):
                value = tensor.detach().cpu().contiguous()
                digest.update(str((value.dtype, tuple(value.shape))).encode())
                digest.update(value.view(torch.uint8).numpy().tobytes())
        return digest.hexdigest()


def empty_memory_state(episode_id: str, *, layers: int, batch_size: int,
                       heads: int, head_dim: int, device: torch.device,
                       dtype: torch.dtype) -> MemoryWAMState:
    empty_meta = torch.empty(0, dtype=torch.long, device=device)
    values = tuple(
        MemoryLayerKV(
            torch.empty(batch_size, 0, heads, head_dim, device=device,
                        dtype=dtype),
            torch.empty(batch_size, 0, heads, head_dim, device=device,
                        dtype=dtype),
            empty_meta.clone(), empty_meta.clone(), empty_meta.clone())
        for _ in range(layers))
    return MemoryWAMState(episode_id, values)


def frame_metadata(frame_id: int, *, video_tokens: int, gist_tokens: int,
                   device: torch.device) -> tuple[torch.Tensor, ...]:
    frame_ids = torch.full((video_tokens + gist_tokens,), frame_id,
                           dtype=torch.long, device=device)
    kinds = torch.cat((
        torch.full((video_tokens,), int(MemoryTokenKind.CLEAN_VIDEO),
                   dtype=torch.long, device=device),
        torch.full((gist_tokens,), int(MemoryTokenKind.GIST),
                   dtype=torch.long, device=device),
    ))
    slots = torch.cat((torch.arange(video_tokens, device=device),
                       torch.arange(gist_tokens, device=device)))
    return frame_ids, kinds, slots


def _keep_clean(frame_ids: torch.Tensor, current_frame: int,
                policy: MemoryWAMPolicy) -> torch.Tensor:
    if policy.full_history:
        return torch.ones_like(frame_ids, dtype=torch.bool)
    anchor = ((frame_ids < policy.anchor_frames) if policy.keep_anchor
              else torch.zeros_like(frame_ids, dtype=torch.bool))
    recent_start = max(0, current_frame - policy.recent_frames + 1)
    recent = ((frame_ids >= recent_start) if policy.keep_recent and
              policy.recent_frames else torch.zeros_like(frame_ids,
                                                          dtype=torch.bool))
    return anchor | recent


def retain_layer(layer: MemoryLayerKV, current_frame: int,
                 policy: MemoryWAMPolicy) -> MemoryLayerKV:
    clean = layer.kinds == int(MemoryTokenKind.CLEAN_VIDEO)
    gist = layer.kinds == int(MemoryTokenKind.GIST)
    keep = (clean & _keep_clean(layer.frame_ids, current_frame, policy))
    if policy.keep_gist:
        keep |= gist
    indices = torch.nonzero(keep, as_tuple=False).squeeze(-1)
    selected = layer.select(indices)
    identities = torch.stack((selected.frame_ids, selected.kinds,
                              selected.slots), dim=1)
    if identities.numel() and torch.unique(identities, dim=0).shape[0] != len(
            identities):
        raise ValueError("duplicate typed memory token")
    return selected


def commit_memory(state: MemoryWAMState, delta: MemoryWAMDelta,
                  policy: MemoryWAMPolicy, *, detach: bool,
                  record_receipt: bool = True) -> MemoryWAMState:
    if delta.episode_id != state.episode_id:
        raise ValueError("episode mismatch")
    delta_digest = None
    if record_receipt:
        delta_digest = delta.digest()
        for receipt in state.receipts:
            if receipt.observation_id == delta.observation_id:
                if (receipt.frame_id == delta.frame_id and
                        receipt.digest == delta_digest):
                    return state
                raise ValueError("observation ID reused with different content")
    if delta.base_version != state.version:
        raise ValueError("stale memory delta")
    if delta.frame_id != state.next_frame_id:
        raise ValueError("non-contiguous global frame ID")
    if len(delta.layers) != len(state.layers):
        raise ValueError("memory layer count changed")
    layers = tuple(retain_layer(old.append(new), delta.frame_id, policy)
                   for old, new in zip(state.layers, delta.layers))
    if detach:
        layers = tuple(layer.detached() for layer in layers)
    receipts = state.receipts
    if record_receipt:
        receipt = MemoryCommitReceipt(delta.observation_id, delta.frame_id,
                                      delta_digest)
        receipts = receipts + (receipt,)
    return MemoryWAMState(state.episode_id, layers, state.next_frame_id + 1,
                          state.version + 1, receipts)
