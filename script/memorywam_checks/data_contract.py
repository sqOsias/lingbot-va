"""RMBench fullctx 的显式 episode/frame/action 时间合同。

这里不把 ``z[t] -> action[t+1]`` 简写成单个数组偏移。每条记录同时保存
决策 latent、目标 latent、parquet label 行和对应 physical target 序号，避免
把“动作组编号”和“物理动作时刻”混为一谈。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch


@dataclass(frozen=True)
class DecisionAlignment:
    episode_id: str
    decision_latent_id: int
    target_latent_id: int
    target_action_group_id: int
    source_rgb_index: int
    target_rgb_index: int
    parquet_row_start: int
    parquet_row_end: int
    physical_target_start: int
    physical_target_end: int


@dataclass(frozen=True)
class EpisodeAlignment:
    episode_id: str
    latent_frames: int
    action_per_latent: int
    source_rgb_stride: int
    vae_temporal_stride: int
    available_action_rows: int
    supervised_action_rows: int
    truncated_tail_rows: int
    decisions: tuple[DecisionAlignment, ...]

    def to_dict(self) -> dict:
        result = asdict(self)
        result["schema_version"] = 1
        result["relation"] = "z[t] -> action_group[t+1] and target_z[t+1]"
        return result


def build_episode_alignment(episode_id: str, *, latent_frames: int,
                            available_action_rows: int,
                            action_per_latent: int = 16,
                            source_rgb_stride: int = 4,
                            vae_temporal_stride: int = 4
                            ) -> EpisodeAlignment:
    if latent_frames < 1 or available_action_rows < 0:
        raise ValueError("invalid episode lengths")
    if action_per_latent != source_rgb_stride * vae_temporal_stride:
        raise ValueError("APF must equal RGB stride times VAE temporal stride")
    maximum = (latent_frames - 1) * action_per_latent
    supervised = min(available_action_rows, maximum)
    decisions = []
    for decision in range(latent_frames - 1):
        row_start = decision * action_per_latent
        row_end = min((decision + 1) * action_per_latent,
                      supervised)
        if row_end <= row_start:
            continue
        decisions.append(DecisionAlignment(
            episode_id=episode_id,
            decision_latent_id=decision,
            target_latent_id=decision + 1,
            target_action_group_id=decision + 1,
            source_rgb_index=decision * action_per_latent,
            target_rgb_index=(decision + 1) * action_per_latent,
            parquet_row_start=row_start,
            parquet_row_end=row_end,
            physical_target_start=row_start + 1,
            physical_target_end=row_end,
        ))
    return EpisodeAlignment(episode_id, latent_frames, action_per_latent,
                            source_rgb_stride, vae_temporal_stride,
                            available_action_rows, supervised,
                            available_action_rows - supervised,
                            tuple(decisions))


def attach_alignment_fields(sample: dict, alignment: EpisodeAlignment) -> dict:
    """为训练样本增加稳定身份和显式目标关系，不修改输入字典。"""
    result = dict(sample)
    result.update({
        "episode_id": alignment.episode_id,
        "global_frame_ids": torch.arange(alignment.latent_frames,
                                          dtype=torch.long),
        "decision_frame_ids": torch.tensor(
            [item.decision_latent_id for item in alignment.decisions],
            dtype=torch.long),
        "target_frame_ids": torch.tensor(
            [item.target_latent_id for item in alignment.decisions],
            dtype=torch.long),
        "target_action_group_ids": torch.tensor(
            [item.target_action_group_id for item in alignment.decisions],
            dtype=torch.long),
        "alignment": alignment,
    })
    return result
