# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import gc
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy, CPUOffloadPolicy

from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper as ptd_checkpoint_wrapper,
)

def apply_ac(model):
    """Apply activation checkpointing to the model."""
    for layer_id, transformer_block in enumerate(model.blocks):
        transformer_block = ptd_checkpoint_wrapper(transformer_block, preserve_rng_state=False)
        model.blocks[layer_id] = transformer_block


@dataclass(frozen=True)
class FSDPTopology:
    """Resolved FSDP/HSDP layout, independent of CUDA initialization."""

    world_size: int
    shard_size: int
    replicate_size: int

    @property
    def uses_hsdp(self):
        return self.replicate_size > 1

    @property
    def mesh_shape(self):
        if self.uses_hsdp:
            return (self.replicate_size, self.shard_size)
        return (self.shard_size,)


def resolve_fsdp_topology(world_size, shard_size=0):
    """Validate a requested shard group without touching CUDA or collectives.

    ``shard_size=0`` preserves the legacy behavior: shard over the complete
    world. A proper divisor creates replicated shard groups (HSDP). Uneven
    groups are rejected instead of silently leaving ranks idle or unmatched.
    """
    world_size = int(world_size)
    shard_size = int(shard_size or world_size)
    if world_size < 1:
        raise ValueError(f"world_size must be positive, got {world_size}")
    if shard_size < 1 or shard_size > world_size:
        raise ValueError(
            f"shard_size must be in [1, {world_size}], got {shard_size}")
    if world_size % shard_size:
        raise ValueError(
            f"world_size={world_size} is not divisible by shard_size={shard_size}; "
            "uneven FSDP/HSDP groups are unsupported")
    return FSDPTopology(
        world_size=world_size,
        shard_size=shard_size,
        replicate_size=world_size // shard_size,
    )


def _create_device_mesh(topology):
    if not topology.uses_hsdp:
        # None asks FSDP2 to use the default one-dimensional world mesh and is
        # exactly the layout used by the legacy implementation.
        return None
    return init_device_mesh(
        "cuda",
        topology.mesh_shape,
        mesh_dim_names=("replicate", "shard"),
    )


def shard_model(model,
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.float32,
                cpu_offload=False,
                shard_size=0,
                reshard_after_forward=True,
                root_reshard_after_forward=True):
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    topology = resolve_fsdp_topology(world_size, shard_size)
    mesh = _create_device_mesh(topology)
    mp_policy = MixedPrecisionPolicy(
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
        cast_forward_inputs=False,
    )
    fsdp_config = {
        "mp_policy": mp_policy,
        "reshard_after_forward": bool(reshard_after_forward),
    }
    if mesh is not None:
        fsdp_config["mesh"] = mesh
    if cpu_offload:
        fsdp_config["offload_policy"] = CPUOffloadPolicy(pin_memory=True)

    for block in model.blocks:
        fully_shard(block.attn1, **fsdp_config)
        fully_shard(block.attn2, **fsdp_config)
        fully_shard(block.ffn, **fsdp_config)
        fully_shard(block, **fsdp_config)

    root_config = dict(fsdp_config)
    root_config["reshard_after_forward"] = bool(root_reshard_after_forward)
    fully_shard(model, **root_config)
    model._fsdp_topology = topology
    return model


def free_model(model):
    del model
    gc.collect()
    torch.cuda.empty_cache()
