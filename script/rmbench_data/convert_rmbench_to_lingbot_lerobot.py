#!/usr/bin/env python3
"""Convert RMBench demonstrations to LingBot-VA's LeRobot v2.1 input.

=================================== 脚本作用 ===================================

  本脚本是 holdout 流水线的【convert 阶段】第一步（共两步），将上一阶段
  `collect_rmbench_holdout_experts.py` 产出的 RMBench 原生 HDF5 数据转换为
  LeRobot v2.1 标准数据集格式，供下游 `prepare_rmbench_apf16_dataset.py`
  进一步加工为 APF16 全上下文数据集。

  核心数据转换约定（极其重要，上下游全部依赖）：

  1) 16D Action 表示：绝对末端执行器（EEF）控制，左右臂各占 8D：
       left:  [x, y, z, qx, qy, qz, qw, gripper]   → 8D
       right: [x, y, z, qx, qy, qz, qw, gripper]   → 8D
       RMBench 存储四元数为 WXYZ，本脚本显式转换为 LingBot 的 XYZW 顺序。

  2) 帧对齐（Image-Action Pairing）：
       观测图像 observation.image.cam_*[t]  ←→  目标 action[t+1]
       即「看到第 t 帧图像后，执行到下一帧 t+1 的末端位姿」。
       因此输出帧数 = 源帧数 - 1（最后一帧只有观测没有下一个动作）。

  3) action_config 两种模式：
       A. 子任务分段模式（默认，--window-frames 未指定）：
          从 language_annotation.json 读取分步标注，每段一个 action_text。
       B. 全上下文窗口模式（--window-frames=4097）：
          整条轨迹按固定窗口切分，所有窗口的 action_text 都是完整任务指令。
          本项目主要使用此模式配合 fullctx 训练。

  输入目录结构（来自 collect 阶段）：
    {task_root}/
    ├── language_annotation.json     # 子任务级分步语言 + 帧长度
    ├── instructions/
    │   └── episode{I}.json          # seen/unseen 任务指令 + seed
    └── data/
        └── episode{I}.hdf5          # RGB 视频 + 末端位姿 + 夹爪开度

  输出目录结构（LeRobot v2.1）：
    {output_root}/
    ├── meta/
    │   ├── episodes.jsonl           # 每集元数据（含 action_config 分段）
    │   ├── stats.safetensors        # 各特征统计量（后续填充）
    │   └── info.json                # 数据集 schema / fps / robot_type
    ├── videos/
    │   └── episode_{I}_*-{cam}.mp4  # 三路相机独立视频（pyav 编码）
    ├── conversion_manifest.json     # 本脚本的转换溯源清单
    └── ...
================================================================================
"""

from __future__ import annotations

import argparse
from io import BytesIO
import json
from pathlib import Path
import re
import shutil

import h5py
import numpy as np
from PIL import Image
from lerobot.datasets.lerobot_dataset import LeRobotDataset


# 相机名映射：LeRobot 观测特征 key → RMBench HDF5 内部路径
# 三个相机：俯视主视角（cam_high） + 左右腕部视角（wrist）
CAMERA_MAP = {
    "cam_high": "/observation/head_camera/rgb",
    "cam_left_wrist": "/observation/left_camera/rgb",
    "cam_right_wrist": "/observation/right_camera/rgb",
}

# HDF5 文件名正则，提取 episode 序号（要求从 0 连续）
EPISODE_RE = re.compile(r"episode(\d+)\.hdf5$")


def episode_index(path: Path) -> int:
    """从 episode 文件路径中解析序号，用于排序和连续性校验。"""
    match = EPISODE_RE.fullmatch(path.name)
    if match is None:
        raise ValueError(f"Unexpected episode filename: {path.name}")
    return int(match.group(1))


def decode_rgb(encoded: object) -> Image.Image:
    """从 HDF5 中存储的字节（可能是 bytes 也可能是 h5py 数组）解码为 PIL RGB 图像。

    RMBench 在 HDF5 中存储的是 JPEG 编码后的二进制，需要 PIL 重新解码并
    强制转换为 RGB（三通道），避免出现 RGBA / L / P 等格式导致下游 VAE 报错。
    """
    if hasattr(encoded, "tobytes"):
        # h5py Dataset 对象 → 转成原始 bytes
        encoded = encoded.tobytes()
    return Image.open(BytesIO(encoded)).convert("RGB")


def wxyz_pose_to_xyzw(pose: np.ndarray) -> np.ndarray:
    """四元数分量顺序转换：RMBench(WXYZ) → LingBot(XYZW)。

    输入形状 [..., 7] = [x, y, z, qw, qx, qy, qz]
    输出形状 [..., 7] = [x, y, z, qx, qy, qz, qw]

    注意：这是本项目最关键的格式转换之一，错一位会导致模型输出的姿态完全反了，
    且训练时 loss 不降。如果在实际机器人上观察到机械臂乱转，第一怀疑点就是这里。
    """
    pose = np.asarray(pose)
    if pose.shape[-1] != 7:
        raise ValueError(f"Expected [...,7] pose, got {pose.shape}")
    # xyz 保持不变；把原来的 qw(索引3) 从中间搬到最后
    return np.concatenate([pose[..., :3], pose[..., 4:7], pose[..., 3:4]], axis=-1)


def load_instruction(path: Path, split: str) -> str:
    """从 episode 指令文件中读取指定 split（seen/unseen）的任务指令。

    holdout 评估时 seen/unseen 通常填相同指令（评估新场景而非新语言），
    但保留 split 机制以便未来做跨指令泛化实验。
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    choices = payload.get(split, [])
    if not isinstance(choices, list) or not choices:
        raise ValueError(f"No `{split}` instruction in {path}")
    text = str(choices[0]).strip()
    if not text:
        raise ValueError(f"Empty `{split}` instruction in {path}")
    return text


def load_action_config(
    language_annotations: dict,
    source_index: int,
    source_frames: int,
    output_frames: int,
) -> list[dict]:
    """【子任务分段模式】从 language_annotation.json 解析分步动作分段。

    language_annotations 格式（RMBench 自动生成的阶段标注）：
      {"episode_0": [["移动左臂到...", 120], ["抓取红色方块", 45], ...], ...}
      每一项 = [动作描述文本, 该阶段占用的**源帧数**]

    关键帧长转换：源帧长总和 = source_frames（含初始帧），
    但输出帧长 = output_frames = source_frames - 1，
    所以最后一段会截断 `min(start+length, output_frames)` 对齐。

    最终 action_config 覆盖 [0, output_frames) 的每一帧，供下游数据集加载器
    做时域掩码 / 分段 loss / 文本对齐等操作。
    """
    entries = language_annotations.get(f"episode_{source_index}")
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"Missing phase annotations for episode_{source_index}")
    lengths = [int(entry[1]) for entry in entries]
    if sum(lengths) != source_frames:
        raise ValueError(
            f"Phase lengths for episode_{source_index} sum to {sum(lengths)}, "
            f"but HDF5 has {source_frames} frames"
        )
    result = []
    start = 0
    for entry, length in zip(entries, lengths):
        # 关键：输出帧比输入帧少 1，所以每段用 min 截断对齐
        end = min(start + length, output_frames)
        if end > start:
            result.append({
                "start_frame": start,
                "end_frame": end,        # 左闭右开：[start, end)
                "action_text": str(entry[0]).strip(),
            })
        start += length
    if not result or result[-1]["end_frame"] != output_frames:
        raise ValueError(
            f"Action segments do not cover output episode {source_index}: "
            f"last={result[-1] if result else None}, frames={output_frames}"
        )
    return result


def make_task_windows(output_frames: int, instruction: str, window_frames: int) -> list[dict]:
    """【全上下文窗口模式】按固定窗口大小切分轨迹，每个窗口绑定完整任务指令。

    当 `--window-frames=4097`（本项目 fullctx 默认值）时：
      - 只要 output_frames <= 4096，整条轨迹就是一个窗口，action_text 不变。
      - 这对应「全上下文」训练模式：模型一次看到整条轨迹的所有历史帧。

    尾部窗口保护：若最后一段窗口 < 5 帧，则合并到前一个窗口，避免出现过短
    片段导致后续 VAE 编码 / 注意力掩码出现异常（cf8 需要至少 8 帧）。
    """
    if window_frames < 2:
        raise ValueError(f"--window-frames must be >= 2, got {window_frames}")
    # 从 0 开始，步长 = window_frames，等步长切分
    windows = [
        {
            "start_frame": start,
            "end_frame": min(start + window_frames, output_frames),
            "action_text": instruction,  # 所有窗口共享同一条完整任务指令
        }
        for start in range(0, output_frames, window_frames)
    ]
    # 尾部保护：最后一个窗口 < 5 帧就吸收进前一个窗口
    if len(windows) > 1 and windows[-1]["end_frame"] - windows[-1]["start_frame"] < 5:
        windows[-2]["end_frame"] = output_frames
        windows.pop()
    # 最终断言：必须完整无重叠覆盖 [0, output_frames)，且每段至少 5 帧
    if (
        not windows
        or windows[0]["start_frame"] != 0
        or windows[-1]["end_frame"] != output_frames
        or any(window["end_frame"] - window["start_frame"] < 5 for window in windows)
    ):
        raise AssertionError(f"Task windows do not cover [0, {output_frames}): {windows}")
    return windows


def make_features(height: int, width: int) -> dict:
    """构造 LeRobot Dataset 的 schema，声明每个特征的 dtype / shape / 命名。

    注意 action.names 是 **双重嵌套列表** `[[...]]`：LeRobot 用它表示「16 维
    组成一个逻辑块」，下游 APF16 会根据这些 names 做 slot 切分和掩码生成。
    """
    features = {
        "action": {
            "dtype": "float32",
            "shape": (16,),
            # 双层嵌套 = 16D 是一个逻辑块（左臂 8D + 右臂 8D）
            "names": [
                [
                    "left_x", "left_y", "left_z", "left_qx", "left_qy", "left_qz", "left_qw", "left_gripper",
                    "right_x", "right_y", "right_z", "right_qx", "right_qy", "right_qz", "right_qw", "right_gripper",
                ]
            ],
        }
    }
    # 三路相机：均为视频类型，CHW 顺序（channels-first）
    for target_key in CAMERA_MAP:
        features[f"observation.images.{target_key}"] = {
            "dtype": "video",
            "shape": (3, height, width),
            "names": ["channels", "height", "width"],
        }
    return features


def source_action(handle: h5py.File) -> np.ndarray:
    """从 HDF5 中读取并拼接出 [T, 16] 的 action 序列（每帧是下一帧的目标位姿）。

    组装顺序：left_ee(7D-XYZW) + left_gripper + right_ee(7D-XYZW) + right_gripper
    返回 shape = [T, 16]，其中 T = 源 HDF5 中的总帧数（含初始帧）。
    后续在 main 中会取 actions[t+1] 作为第 t 帧观测对应的 action。
    """
    left  = wxyz_pose_to_xyzw(np.asarray(handle["/endpose/left_endpose"],  dtype=np.float32))
    right = wxyz_pose_to_xyzw(np.asarray(handle["/endpose/right_endpose"], dtype=np.float32))
    left_grip  = np.asarray(handle["/endpose/left_gripper"],  dtype=np.float32)[:, None]
    right_grip = np.asarray(handle["/endpose/right_gripper"], dtype=np.float32)[:, None]
    return np.concatenate([left, left_grip, right, right_grip], axis=-1)


def probe_source(path: Path) -> tuple[int, int, int]:
    """【Fail-Closed 预检】用第一个 episode 探针校验 HDF5 的完整性与一致性。

    检查项：
      ① action 维度必须为 [T>=2, 16]（至少 2 帧才能形成一对 image→action）
      ② 三路相机在 HDF5 中都存在且时序长度与 action 完全相等
      ③ 三路相机每帧的分辨率完全一致（不允许混合 240p/480p 等）

    返回：(源总帧数 T, 图像高度 height, 图像宽度 width)
    如果任何一项不满足，直接抛异常，拒绝后续转换（防止生成脏数据集）。
    """
    with h5py.File(path, "r") as handle:
        actions = source_action(handle)
        if actions.ndim != 2 or actions.shape[1] != 16 or actions.shape[0] < 2:
            raise ValueError(f"Expected actions [T>=2,16], got {actions.shape} in {path}")
        lengths = {actions.shape[0]}
        image_size = None
        for key in CAMERA_MAP.values():
            if key not in handle:
                raise KeyError(f"Missing {key} in {path}")
            lengths.add(handle[key].shape[0])
            size = decode_rgb(handle[key][0]).size
            if image_size is None:
                image_size = size
            elif image_size != size:
                raise ValueError(f"Camera sizes differ in {path}: {image_size} vs {size}")
        if len(lengths) != 1:
            raise ValueError(f"Action/image lengths differ in {path}: {sorted(lengths)}")
        assert image_size is not None
        width, height = image_size  # PIL.size 返回 (width, height)，要与 CHW 对齐
        return actions.shape[0], height, width


def rewrite_episode_metadata(output_root: Path, configs: dict[int, list[dict]]) -> None:
    """把 action_config（分段 / 窗口信息）写回 LeRobot 的 meta/episodes.jsonl。

    LeRobotDataset 在 save_episode() 时只会写基础的 episode_index / frame_index
    等字段，action_config 需要作为额外元数据**二次注入**。
    这份文件是下游 `prepare_rmbench_apf16_dataset.py` 生成训练窗口和
    文本 token 对齐的核心依据。
    """
    episodes_path = output_root / "meta" / "episodes.jsonl"
    records = [json.loads(line) for line in episodes_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for record in records:
        index = int(record["episode_index"])
        if index not in configs:
            raise KeyError(f"No action_config for output episode {index}")
        record["action_config"] = configs[index]
    episodes_path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def main() -> int:
    # 阶段1：解析命令行参数（输入/输出目录、数据集标识、action_config 模式等）
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--instruction-split", choices=("seen", "unseen"), default="seen")
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--window-frames",
        type=int,
        help="Use continuous windows with the full task instruction instead of subtask annotations.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    # 阶段2：清理输出目录（带 overwrite 保护，防止误删已有数据）
    if args.output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output exists; pass --overwrite: {args.output_root}")
        shutil.rmtree(args.output_root)

    # 阶段3：收集源 HDF5 文件，校验序号必须从 0 连续（防止缺漏 episode）
    sources = sorted((args.task_root / "data").glob("episode*.hdf5"), key=episode_index)
    if args.limit is not None:
        sources = sources[: args.limit]
    indices = [episode_index(path) for path in sources]
    if not sources or indices != list(range(len(sources))):
        raise ValueError(f"Source episodes must be contiguous from zero, got {indices}")

    # 阶段4：Fail-Closed 预检 + 创建 LeRobot 数据集骨架（schema、视频编码后端等）
    first_frames, height, width = probe_source(sources[0])
    print(json.dumps({
        "event": "source_probe", "episodes": len(sources),
        "first_source_frames": first_frames, "height": height, "width": width,
    }))
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        root=args.output_root,
        fps=args.fps,
        robot_type="aloha-agilex-eef",
        features=make_features(height, width),
        use_videos=True,
        video_backend="pyav",
    )
    # 预加载语言分步标注 + 初始化 action_config（分段/窗口信息）与溯源清单
    annotations = json.loads((args.task_root / "language_annotation.json").read_text(encoding="utf-8"))
    action_configs: dict[int, list[dict]] = {}
    manifest = []

    # 阶段5：逐 episode 转换核心逻辑
    for output_index, source in enumerate(sources):
        source_index = episode_index(source)
        # 5.1 读取该集的任务指令（seen/unseen split）
        instruction = load_instruction(
            args.task_root / "instructions" / f"episode{source_index}.json",
            args.instruction_split,
        )
        with h5py.File(source, "r") as handle:
            # 5.2 读取 16D action 序列（含 WXYZ→XYZW 四元数转换）
            actions = source_action(handle)
            output_frames = actions.shape[0] - 1  # 最后一帧无下一动作，丢弃
            # 5.3 根据模式生成 action_config：窗口切分 or 子任务分段
            action_configs[output_index] = (
                make_task_windows(output_frames, instruction, args.window_frames)
                if args.window_frames is not None
                else load_action_config(annotations, source_index, actions.shape[0], output_frames)
            )
            # 5.4 逐帧写入：image[t] ←→ action[t+1] 对齐
            for frame_index in range(output_frames):
                frame = {"action": actions[frame_index + 1]}
                for target_key, source_key in CAMERA_MAP.items():
                    frame[f"observation.images.{target_key}"] = decode_rgb(handle[source_key][frame_index])
                dataset.add_frame(frame, task=instruction)
        # 5.5 保存当前 episode（pyav 编码 mp4 + episodes.jsonl 基础条目）
        dataset.save_episode()
        # 5.6 记录溯源信息，写入 manifest
        record = {
            "source": str(source),
            "source_frames": int(actions.shape[0]),
            "output_frames": int(output_frames),
            "instruction": instruction,
            "action_config": action_configs[output_index],
            "quaternion_conversion": "RMBench WXYZ -> LingBot XYZW",
        }
        manifest.append(record)
        print(json.dumps({"event": "episode_saved", "episode": output_index, **record}))

    # 阶段6：后处理
    # 6.1 将 action_config 二次注入 episodes.jsonl（LeRobotDataset 原生不支持此字段）
    rewrite_episode_metadata(args.output_root, action_configs)
    # 6.2 写最终转换溯源清单 conversion_manifest.json，记录全链路信息
    payload = {
        "source_task_root": str(args.task_root.resolve()),
        "output_root": str(args.output_root.resolve()),
        "repo_id": args.repo_id,
        "task_name": args.task_name,
        "fps": args.fps,
        "window_frames": args.window_frames,
        "alignment": "image[t], action=absolute_eef_xyzw[t+1]",
        "episodes": manifest,
    }
    (args.output_root / "conversion_manifest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({"event": "complete", "episodes": len(manifest), "output": str(args.output_root)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
