"""不依赖 FlexAttention 的 MemoryWAM dense mask oracle。"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from .memory_core import MemoryPolicy, TokenMeta, token_visible


def build_visibility(queries: Sequence[TokenMeta], keys: Sequence[TokenMeta],
                     policy: MemoryPolicy, *, device=None) -> torch.Tensor:
    return torch.tensor(
        [[token_visible(query, key, policy) for key in keys]
         for query in queries],
        dtype=torch.bool,
        device=device,
    )
