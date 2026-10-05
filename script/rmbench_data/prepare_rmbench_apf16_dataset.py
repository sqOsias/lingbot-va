#!/usr/bin/env python3
"""准备 APF16 全上下文 RMBench 数据集视图。

=================================== 脚本作用 ===================================

  本脚本是 holdout 流水线【convert 阶段】第二步，承接上一步
  `convert_rmbench_to_lingbot_lerobot.py` 产出的 LeRobot 标准数据集，
  生成 APF16 训练/推理所需的两个核心契约文件：

    1) observation_anchors.json —— 按 (episode, window_start) 索引的
       「原点物理位姿 + 窗口首个 label」记录，供 `LatentLeRobotDataset`
       加载器推理时还原绝对位姿。
    2) continuous_contract.json —— 全局 action slots 布局契约，声明
       每窗口的 VAE 下采样帧数、前后 padding 数、有效 action 范围等，
       训练时据此构造 Flex Attention 掩码和 Flow Matching 时域位置。

================================================================================
"""

import argparse
import json
import os
import shutil
from pathlib import Path

import h5py
import numpy as np


def physical16(handle):
    """从原始 HDF5 全量 T 帧物理位姿组装 [T, 16] 的完整 action 序列（含初始帧 t=0）。

    与 convert 阶段的 source_action() 逻辑等价，但这里读取的是所有 T 帧
    （包括 t=0 初始位姿，convert 阶段丢弃它用来做 image→action 对齐）。
    输出用于计算 origin（t=0 绝对位姿）和每个窗口的 first_label（t=start+1 绝对位姿）。
    """
    parts = []
    for arm in ("left", "right"):
        # 7D WXYZ → 7D XYZW（x y z qx qy qz qw）
        pose = np.asarray(handle[f"endpose/{arm}_endpose"], dtype=np.float32)
        pose_xyzw = np.concatenate([pose[:, :3], pose[:, 4:7], pose[:, 3:4]], axis=1)
        grip = np.asarray(handle[f"endpose/{arm}_gripper"], dtype=np.float32)[:, None]
        parts.extend([pose_xyzw, grip])
    return np.concatenate(parts, axis=1)  # [T, 16]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument(
        "--regenerate-window-size", type=int, default=None,
        help="Replace legacy action_config segments with fixed overlapping windows",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 阶段1：构造 APF16 视图目录（data/videos 软链接复用，meta 深拷贝）
    # ------------------------------------------------------------------
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    # data(Parquet) 和 videos(mp4) 用软链接 → 零拷贝复用 LeRobot 原始产物
    for name in ("data", "videos"):
        os.symlink((args.source / name).resolve(), args.output / name, target_is_directory=True)
    # meta 目录深拷贝，因为本脚本可能会覆写 episodes.jsonl（窗口再生时）
    shutil.copytree(args.source / "meta", args.output / "meta")
    shutil.copy2(args.source / "rmbench_action_stats.json", args.output / "rmbench_action_stats.json")
    # latents 目录预先占位，后续 extract_rmbench_latents.py 会在这里写 safetensors
    (args.output / "latents").mkdir()

    episodes = [
        json.loads(line) for line in (args.output / "meta/episodes.jsonl").read_text().splitlines()
        if line.strip()
    ]

    # ------------------------------------------------------------------
    # 阶段2 [可选]：若传了 --regenerate-window-size，按固定窗口重切 action_config
    #   窗口规则：start += stride，每窗口 end = start + available + 1，
    #   available 向下取整到 4 的倍数（VAE 下采样要求）。
    # ------------------------------------------------------------------
    if args.regenerate_window_size is not None:
        window_size = args.regenerate_window_size
        stride = window_size - 1
        if window_size < 2 or stride % 4:
            raise ValueError("window size minus one must be a positive multiple of 4")
        for episode in episodes:
            length = int(episode["length"])
            action_text = episode["tasks"][0]
            windows = []
            for start in range(0, length - 1, stride):
                available = min(stride, length - start - 1)
                available -= available % 4  # 向下取整到 4 的倍数
                if available == 0:
                    continue
                windows.append({
                    "start_frame": start,
                    "end_frame": start + available + 1,  # 左闭右开，含 start 图像 + 后面 available 个动作
                    "action_text": action_text,
                })
            episode["action_config"] = windows
        # 写回 episodes.jsonl（覆盖之前的子任务分段/旧窗口）
        (args.output / "meta/episodes.jsonl").write_text(
            "".join(json.dumps(episode) + "\n" for episode in episodes)
        )

    # ------------------------------------------------------------------
    # 阶段3：逐 episode → 逐窗口 生成 observation_anchors 和窗口级 contract 条目
    # ------------------------------------------------------------------
    anchors = {}     # key = "episode_index:start_frame"，value = 原点位姿+首标签
    contracts = []   # 每个 episode 一条，内部 windows 列表逐窗口展开

    for episode in episodes:
        ep = int(episode["episode_index"])
        length = int(episode["length"])  # LeRobot 观测帧数 = 源物理帧 T - 1
        # 从原始 HDF5 重取全量 T 帧物理位姿（含 t=0 初始位姿，length+1 帧）
        with h5py.File(args.raw_root / f"episode{ep}.hdf5") as handle:
            states = physical16(handle)
        if len(states) != length + 1:
            raise AssertionError((ep, len(states), length))

        origin = states[0]       # 本集原点 = 物理 t=0 位姿（绝对，整集固定不随窗口重置）
        used = 0                  # 本集所有窗口累计有效 action 数（用于 excluded_tail 审计）
        windows = []

        for window in episode["action_config"]:
            start, end = int(window["start_frame"]), int(window["end_frame"])
            available = end - start - 1  # 本窗口原始可容纳的 action 数
            if available < 0 or available % 4:
                raise AssertionError((ep, start, end, available))

            # 3.1 计算 VAE 下采样后的视频帧 id 列表（每 4 帧采 1 个）
            sampled_ids = list(range(start, end, 4))
            # 截取到满足 `1 + 4*k` 个采样点（Wan2.2 VAE CF 约束：时间维度 1+4k）
            sampled_ids = sampled_ids[:1 + 4 * ((len(sampled_ids) - 1) // 4)]
            latent_frames = (len(sampled_ids) - 1) // 4 + 1  # 下采样后的 latent 帧数
            slots = latent_frames * 16  # 本窗口总 action slots（1 slot = 1 步 16D 动作）

            # 3.2 计算 front_padding（首 chunk 预留 16 slot 推理空窗）+ 有效/尾部 padding
            front_padding = 16                                        # APF16 首 chunk 契约：前 16 slot 是占位
            valid = min(available, slots - front_padding)             # 实际能塞进的有效 action 数
            tail_padding = slots - front_padding - valid              # 剩余尾部 padding slots
            if tail_padding < 0:
                raise AssertionError((ep, start, end, slots, valid))

            # 3.3 写 anchor：下游加载器根据 (ep, start) 查到 origin 和 first_label 还原绝对位姿
            anchors[f"{ep}:{start}"] = {
                "episode_index": ep,
                "start_frame": start,
                "origin_start_frame": 0,     # 原点固定在物理 t=0，跨窗口不重置
                "origin": origin.tolist(),                     # 物理 t=0 的 16D 绝对位姿（XYZW）
                "first_label_physical_frame": start + 1,       # 窗口第一个有效 label 的物理帧号（= start+1）
                "first_label": states[start + 1].tolist(),     # 该帧的 16D 绝对位姿
            }

            # 3.4 写窗口级 slots 布局：训练/推理构造掩码和对齐用
            windows.append({
                "start": start,
                "end": end,
                "sampled_video_frame_ids": sampled_ids,   # VAE 实际编码的视频帧号（步长 4）
                "latent_frames": latent_frames,           # 下采样后的 latent 帧数
                "action_slots": slots,                    # 总 slots = latent_frames * 16
                "valid_action_physical_frames": [start + 1, start + valid],  # 有效 label 的物理帧范围 [闭,闭]
                "valid_actions": valid,                   # 有效 action 步数
                "front_padding": front_padding,           # 前面 padding 槽数（恒 16）
                "tail_padding": tail_padding,             # 后面 padding 槽数
            })
            used += valid

        # 3.5 写 episode 级 contract 条目（含审计字段 excluded_tail）
        contracts.append({
            "episode_index": ep,
            "source_physical_frames": len(states),     # 源 HDF5 物理帧数（含初始帧：T）
            "lerobot_observation_frames": length,      # LeRobot 观测帧数（T-1）
            "episode_origin_physical_frame": 0,        # 原点锚定物理 t=0
            "used_actions": used,                      # 所有窗口累计有效 action 数
            "excluded_tail": length - 1 - used,        # 因 VAE 1+4k 约束丢弃的尾部 action 数
            "windows": windows,                        # 逐窗口 slots 布局明细
        })

    # ------------------------------------------------------------------
    # 阶段4：生成全局 continuous_contract.json（顶层训练元信息 + 所有 episode 明细）
    # ------------------------------------------------------------------
    contract = {
        "task": episodes[0]["tasks"][0],
        "episodes": len(episodes),
        "action_schema": "[L_xyz,L_quat_xyzw,L_gripper,R_xyz,R_quat_xyzw,R_gripper]",
        "parquet_alignment": "observation/image row t stores physical endpose target t+1",
        "sampled_video_frame_stride_environment_actions": 4,  # 每两个相邻采样视频帧之间有 4 步动作
        "vae_temporal_compression": 4,                         # VAE 时间下采样率
        "action_per_sampled_video_frame": 4,                   # 每个视频帧对应 4 个动作
        "action_per_latent_frame": 16,                         # 每个 latent 帧 → 16 个 action slots
        "frame_chunk_size_latents": 2,                         # 推理时（非首 chunk）每步摄入 2 个 latent 帧
        "nominal_actions_per_inference_chunk": 32,             # 每 chunk 名义 32 有效 action
        "first_chunk_valid_actions": 16,                       # 首 chunk 仅 16 有效（扣 front_padding）
        "relative_origin": "episode physical frame 0; never reset at continuation windows",
        "normalization": "byte-for-byte copy of original 50-episode statistics",
        "contracts": contracts,
    }
    (args.output / "observation_anchors.json").write_text(json.dumps(anchors, indent=2))
    (args.output / "continuous_contract.json").write_text(json.dumps(contract, indent=2))
    print(json.dumps({
        "status": "PASS", "episodes": len(episodes), "segments": len(anchors),
        "action_per_latent_frame": 16,
    }))


if __name__ == "__main__":
    main()
