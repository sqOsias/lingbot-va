#!/usr/bin/env python3
"""Collect exactly the requested RMBench expert seeds; never resample.

================================== 脚本作用 ==================================


  功能定位：
    对上游 `seeds` 阶段已专家验证过的「固定种子列表」，逐个运行 RMBench
    仿真环境采集专家演示轨迹。核心约束是「绝不重采样」——即使种子失败也
    不换种子重试，保证相同输入参数每次运行产出 100% 相同的数据集。


  核心执行模型：两阶段采集（消除规划器非确定性）
    ┌─ 阶段 1：规划求解 ────────────────────────────────────────────┐
    │  need_plan=True, save_data=False                               │
    │  用固定种子初始化场景 → 运动规划器求解路径 → 双重成功校验      │
    │  → 仅保存 left/right_joint_path 关节路径数组（不录视频）       │
    └───────────────────────────────────────────────────────────────┘
                              ↓ 成功才继续
    ┌─ 阶段 2：重放录制 ────────────────────────────────────────────┐
    │  need_plan=False, save_data=True                               │
    │  同种子重建相同场景 → 注入阶段1保存的关节路径绕过规划器        │
    │  → 纯运动学回放 + 录制传感器/视频 → 打包 episodeX.hdf5         │
    └───────────────────────────────────────────────────────────────┘

  输出产物：
    {output}/
    ├── seed.txt                       # 明文凭据：本批次所有种子（便于 grep）
    ├── language_annotation.json       # 分步语言描述（子步骤级自然语言）
    ├── instructions/
    │   ├── episode0.json              # 指令入口（seen/unseen 同指令 + seed）
    │   ├── episode1.json
    │   └── ...
    └── data/
        ├── episode0.hdf5              # 主数据：多视角视频 + 关节状态 + 动作
        ├── episode1.hdf5
        └── ...
    同时，输入的 --manifest 文件会被原地原子更新，追加：
      - expert_success / failure  专家成功标记和失败原因
      - initial_state_sha256      初始场景状态指纹
      - task_config_sha256        配置文件指纹
      - raw_trajectory_sha256     最终轨迹文件指纹
      - frames / actions          帧数 / 动作数
      - initial_states_unique     所有 episode 初始状态是否两两唯一
      - seed_overlap_with_training  是否与训练集种子 0..49 重叠（空=安全）
      - status                    COLLECTED / COLLECTED_WITH_FIXED_SEED_FAILURES

"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import shutil
import sys
import traceback
from pathlib import Path

import h5py
import numpy as np
import yaml


def stable_hash(value) -> str:
    """对嵌套字典（可能含 numpy 数组）计算内容稳定的 SHA256。

    使用 sort_keys + 紧凑分隔符保证相同内容无论键序如何都生成相同 hash，
    用于检测不同种子是否意外生成了相同初始场景状态。
    """
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=lambda x: np.asarray(x).tolist()).encode()
    return hashlib.sha256(encoded).hexdigest()


def file_hash(path: Path) -> str:
    """分块（1MB）计算文件内容的 SHA256，避免大文件内存爆炸。

    用于对任务配置文件 demo_clean.yml 和最终轨迹 episodeX.hdf5 生成指纹，
    支持事后审计和复现实验。
    """
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    """原子写入 JSON：先写 .tmp 临时文件，再通过 os.replace 原子替换。

    os.replace 在 POSIX 上是原子操作，保证即使脚本中途被 kill，
    也只会留下 .tmp 垃圾文件，绝不会损坏已有的合法 JSON 数据。
    这是长时运行流水线的关键可靠性保障。
    """
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def resolve_args(root: Path, output: Path, task_name: str) -> dict:
    """从 RMBench 三层 YAML 配置链中装配环境运行参数。

    配置解析链路：
      1. task_config/demo_clean.yml          → 基础任务参数 + embodiment 类型
      2. task_config/_embodiment_config.yml  → embodiment 名 → 机器人文件路径映射
      3. {robot_file}/config.yml             → 具体机器人的关节 / 连杆参数

    注意：此处返回 need_plan=True, save_data=False 用于【第一阶段：规划求解】，
          【第二阶段：重放录制】时会在调用侧覆盖为 need_plan=False, save_data=True。
    """
    config_path = root / "task_config/demo_clean.yml"
    args = yaml.safe_load(config_path.read_text())
    embodiment_types = yaml.safe_load((root / "task_config/_embodiment_config.yml").read_text())
    embodiment = args["embodiment"]
    if len(embodiment) != 1:
        raise ValueError(embodiment)
    robot_file = embodiment_types[embodiment[0]]["file_path"]
    robot_config = yaml.safe_load((Path(robot_file) / "config.yml").read_text())
    args.update(
        task_name=task_name, task_config="demo_clean",
        left_robot_file=robot_file, right_robot_file=robot_file,
        left_embodiment_config=robot_config, right_embodiment_config=robot_config,
        dual_arm_embodied=True, embodiment_name=str(embodiment[0]),
        save_path=str(output),
        need_plan=True,     # 第一阶段：启用运动规划器求解路径
        collect_data=True,
        save_data=False,    # 第一阶段：先不录制传感器数据，只求解关节路径
    )
    return args


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rmbench-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--instruction", required=True)
    parsed = parser.parse_args()

    # ------- 前置防御性校验 -------
    # 种子列表非空且无重复；重复种子会导致同一 episode 被覆盖两次
    if not parsed.seeds or len(set(parsed.seeds)) != len(parsed.seeds):
        raise ValueError("holdout collection requires one or more unique expert-validated seeds")
    # 输出目录必须不存在，防止与旧批次数据混合（fail-fast 原则）
    if parsed.output.exists():
        raise FileExistsError(parsed.output)
    parsed.output.mkdir(parents=True)

    # 落盘一份明文凭据，便于事后通过 grep 快速检索批次包含哪些种子
    (parsed.output / "seed.txt").write_text(" ".join(map(str, parsed.seeds)) + "\n")

    # ------- 动态加载 RMBench 运行时 -------
    # RMBench 不是 pip 包，必须手工把源码目录插入 sys.path 并切换工作目录
    # （RMBench 内部有大量基于 __file__ / cwd 的相对路径读取）
    sys.path[:0] = [str(parsed.rmbench_root), str(parsed.rmbench_root / "script")]
    os.chdir(parsed.rmbench_root)
    from test_render import Sapien_TEST
    from envs.utils.create_actor import UnStableError

    # SAPIEN 渲染后端初始化：必须在创建任务环境之前调用，链接显示驱动 / EGL
    Sapien_TEST()

    # 按任务名动态导入对应的环境类（RMBench 每个任务是独立的模块 + 同名类）
    task_cls = getattr(importlib.import_module(f"envs.{parsed.task}"), parsed.task)
    task = task_cls()

    # 装配第一阶段（规划求解）的参数模板
    args = resolve_args(parsed.rmbench_root, parsed.output, parsed.task)
    manifest = json.loads(parsed.manifest.read_text())

    # 计算任务配置文件指纹，写入每个 episode record 作为溯源凭证
    config_hash = file_hash(parsed.rmbench_root / "task_config/demo_clean.yml")

    # ====================================================================
    #  核心循环：对每个种子执行【两阶段执行模型】
    #
    #  为什么要分两阶段？——RMBench 运动规划器是非确定性的，即使种子相同，
    #  不同次调用求解出的路径在时间戳上也可能有微小抖动。
    #  两阶段策略彻底消除此问题：
    #    阶段 1：need_plan=True   → 规划器求解 + 仅保存关节路径数组
    #    阶段 2：need_plan=False  → 直接注入阶段1保存的路径数组，绕过规划器
    #                             → 开启 save_data 录制传感器 / 视频
    # ====================================================================
    for index, seed in enumerate(parsed.seeds):
        record = manifest["episodes"][index]
        initial_state = None
        error = None
        success = False

        # ---------------- 阶段 1：规划求解（不录制，只算路径） ----------------
        try:
            # 用固定 seed 初始化场景，保证环境随机性完全可控
            task.setup_demo(now_ep_num=index, seed=seed, **args)
            # 抓取初始场景快照（物体位姿等 ground truth），用于唯一性校验和溯源
            if hasattr(task, "get_ground_truth"):
                initial_state = task.get_ground_truth()
            # 调用运动规划器求解 + 在模拟中执行一次
            task.play_once()
            # 双重成功判定：① 规划器找到了路径 ② 任务目标确实被满足（避免假阳性）
            success = bool(task.plan_success and task.check_success())
            if success:
                # 仅保存左右臂关节路径数组（left_joint_path / right_joint_path）
                # 这是阶段 2 重放时的「金标准」输入
                task.save_traj_data(index)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()
        finally:
            # 异常安全：无论成功 / 失败 / 崩溃，必须释放 SAPIEN 环境资源，
            # 防止显存 / 句柄泄漏导致长时间批量运行后卡死
            try:
                task.close_env(clear_cache=True)
            except Exception:
                traceback.print_exc()

        # ---- 原子写入阶段 1 元数据：即使后续崩溃也不丢失进度 ----
        record.update(
            expert_success=success,
            # failure 字段三态：None=成功 / 异常字符串=抛异常 / 固定字符串=专家没成功但没报错
            failure=error if error else (None if success else "expert_failed_fixed_seed"),
            initial_state=initial_state,
            initial_state_sha256=stable_hash(initial_state) if initial_state is not None else None,
            task_config_sha256=config_hash,
        )
        atomic_json(parsed.manifest, manifest)

        # Never-resample 原则：种子失败就跳过，绝不换种子重试
        # （保证每次相同输入参数 → 相同输出数据集）
        if not success:
            continue

        # ---------------- 阶段 2：重放录制（绕过规划器，直接录数据） ----------------
        # 覆盖参数：关闭规划 + 开启传感器 / 视频录制
        replay = dict(args, need_plan=False, save_data=True)

        # 用相同种子重建完全相同的场景（状态必须与阶段 1 逐字节一致）
        task.setup_demo(now_ep_num=index, seed=seed, **replay)

        # 加载阶段 1 保存的关节路径数组
        trajectory = task.load_tran_data(index)
        # 直接注入路径到环境参数中，后续 set_path_lst() 会覆盖规划器输出
        replay["left_joint_path"] = trajectory["left_joint_path"]
        replay["right_joint_path"] = trajectory["right_joint_path"]
        task.set_path_lst(replay)

        # 纯运动学回放 + 录制：此时机械臂走的是阶段 1 求解的完全相同的路径
        task.play_once()

        # 收集环境自动生成的分步语言描述（如「Move left arm to above block」）
        language_path = parsed.output / "language_annotation.json"
        language = json.loads(language_path.read_text()) if language_path.exists() else {}
        language[f"episode_{index}"] = task.language_annotation
        atomic_json(language_path, language)

        # 关闭本 episode 环境，然后做数据打包
        task.close_env(clear_cache=True)
        # 将散落的 pickle 帧 + 视频流合并为单一的 episodeX.hdf5
        task.merge_pkl_to_hdf5_video()
        # 删除中间 pickle 缓存文件，节省磁盘空间（hdf5 已包含全部信息）
        task.remove_data_cache()

        # ---- 读取最终 HDF5 做完整性校验，并写入指纹 ----
        source = parsed.output / "data" / f"episode{index}.hdf5"
        with h5py.File(source) as handle:
            # 左右臂末端位姿序列长度应该相等；取其作为物理帧总数
            lengths = [len(handle[f"endpose/{arm}_endpose"]) for arm in ("left", "right")]
        record.update(
            raw_trajectory_sha256=file_hash(source),  # 轨迹文件指纹（溯源凭证）
            frames=lengths[0],                         # 物理帧数 = 末端位姿数（含初始帧）
            actions=lengths[0] - 1,                    # 动作数 = 帧数 - 1（下游流水线依赖此约定）
        )

        # ---- 生成指令文件：这是下游 convert 脚本读取语言指令的入口 ----
        instruction_dir = parsed.output / "instructions"
        instruction_dir.mkdir(exist_ok=True)
        instruction = parsed.instruction
        # holdout 评估「新场景」而非「新语言」，因此 seen 和 unseen 使用相同指令
        atomic_json(instruction_dir / f"episode{index}.json", {
            "seen": [instruction],
            "unseen": [instruction],
            "seed": seed,
        })

        # 再次原子落盘：更新 trajectory_sha256 / frames / actions 等字段
        atomic_json(parsed.manifest, manifest)

    # ====================================================================
    #  收尾：全局一致性审计（所有 episode 处理完毕后执行）
    # ====================================================================

    # 1) 初始状态唯一性检测：不同种子理论上应生成不同场景
    #    如果 hash 去重后数量减少，说明存在 bug（不同种子生成了同一初始状态）
    hashes = [r.get("initial_state_sha256") for r in manifest["episodes"] if r.get("initial_state_sha256")]
    manifest["initial_states_unique"] = len(hashes) == len(set(hashes))

    # 2) 训练集种子重叠硬校验：安全网的安全网
    #    上游 prepare_rmbench_holdouts.py 已检查过 --start-seed >= 50，
    #    这里再从【实际执行的种子列表】检查一遍，防止中间环节篡改导致泄露
    manifest["seed_overlap_with_training"] = sorted(set(parsed.seeds) & set(range(50)))
    if manifest["seed_overlap_with_training"]:
        raise RuntimeError("holdout seed overlaps the official training seed range")

    # 3) 批次整体状态标记：
    #    COLLECTED = 全部成功；COLLECTED_WITH_FIXED_SEED_FAILURES = 有失败但未重采样
    manifest["status"] = "COLLECTED" if all(r["expert_success"] for r in manifest["episodes"]) else "COLLECTED_WITH_FIXED_SEED_FAILURES"
    atomic_json(parsed.manifest, manifest)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
