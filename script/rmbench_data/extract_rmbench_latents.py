#!/usr/bin/env python3
"""从 LeRobot v2.1 数据集中提取 LingBot-VA Wan2.2 视频 latent。

本脚本将 LeRobot v2.1 数据集中的原始 mp4 视频观测转换为预计算的 Wan2.2 VAE latents，
并与对应任务指令的 CLIP 文本嵌入配对保存。
采用两段式加载策略以适配消费级显存：先计算并持久化所有文本嵌入，然后卸载文本模型，
再执行因果 VAE 流式编码。
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
from pathlib import Path
import sys

import av
import numpy as np
import torch
import torch.nn.functional as F
from diffusers.pipelines.wan.pipeline_wan import prompt_clean
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from tqdm import tqdm


# 项目根目录（脚本位于 script/rmbench_data/，向上两级为项目根）
LINGBOT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LINGBOT_ROOT))
from wan_va.modules.utils import (  # noqa: E402
    WanVAEStreamingWrapper,
    load_text_encoder,
    load_tokenizer,
    load_vae,
)


# 各路摄像头视频输入到 VAE 前的目标分辨率 (H, W)：
#   - cam_high：高位俯视全局视角，使用较大分辨率 256x320 以保留工作台细节。
#   - 左/右腕部相机：视角紧贴操作区域，若使用高分辨率会导致 Wan 压缩率
#     下承载过多无信息量的特写像素，因此降为 128x160。
CAMERA_SIZES = {
    "observation.images.cam_high": (256, 320),
    "observation.images.cam_left_wrist": (128, 160),
    "observation.images.cam_right_wrist": (128, 160),
}


def get_text_embedding(tokenizer, encoder, text: str, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """将单条任务指令编码为 CLIP hidden state 张量。

    处理流程：
      1. 使用 diffusers 的 prompt_clean 去除风格类 token；
      2. tokenizer 补零/截断到固定 512 token 长度并生成 attention mask；
      3. torch.no_grad 下跑文本编码器，取 last_hidden_state；
      4. 结果转至 device+dtype 后搬回 CPU，便于第一阶段批量结束后持久化存储，
         并让后续 VAE 阶段独占 GPU 显存。
    """
    cleaned = prompt_clean(text)
    inputs = tokenizer(
        [cleaned],
        padding="max_length",
        max_length=512,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    encoder_device = next(encoder.parameters()).device
    with torch.no_grad():
        hidden = encoder(
            inputs.input_ids.to(encoder_device),
            inputs.attention_mask.to(encoder_device),
        ).last_hidden_state
    return hidden[0].to(device=device, dtype=dtype).cpu()


def read_selected_frames(
    video_path: Path,
    start_frame: int,
    end_frame: int,
    frame_step: int,
):
    """从视频中解码 ``[start_frame, end_frame)`` 范围内按 ``frame_step`` 等间隔采样的帧。

    Wan 时序编码器遵循 ``1 + 4k`` 源帧数约束：首帧用于初始化因果缓存，
    之后每 4 个源帧生成 1 个 latent 帧。
    为保证每个后续流式块都合法，这里最多丢弃末尾 3 个采样帧，使
    ``(有效采样数 - 1)`` 能被 4 整除。

    返回值通过 yield 产生 (原视频帧索引, RGB24 ndarray)，
    索引用于后续写入 payload 的 frame_ids 字段，作为与环境动作 slot 时序对齐的凭证。
    """
    frame_ids = list(range(start_frame, end_frame, frame_step))
    # Wan 时序编码器要求 1 + 4k 源帧数。最多丢弃末尾 3 帧，
    # 以保证每个后续流式 chunk（每 4 帧生成 1 latent 帧）都合法。
    valid_count = 1 + 4 * ((len(frame_ids) - 1) // 4)
    frame_ids = frame_ids[:valid_count]
    selected = set(frame_ids)
    container = av.open(str(video_path))
    try:
        for frame_index, frame in enumerate(container.decode(video=0)):
            if frame_index >= end_frame:
                break
            if frame_index in selected:
                yield frame_index, frame.to_ndarray(format="rgb24")
    finally:
        container.close()


def normalize_latents(vae, mu: torch.Tensor) -> torch.Tensor:
    """使用 VAE 配置中保存的 mean / std 对 latent 做标准化。

    Wan 预训练 VAE 在 config 中随权重附带 latents_mean 与 latents_std，
    用于将后验均值平移缩放为下游扩散模型期望的单位方差信号。
    计算时先以 float32 做归一化以保数值精度，完成后再 cast 回原始 dtype。
    """
    mean = torch.tensor(vae.config.latents_mean, device=mu.device).view(1, -1, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std, device=mu.device).view(1, -1, 1, 1, 1)
    return ((mu.float() - mean) / std).to(mu.dtype)


def encode_camera_segment(
    wrapper: WanVAEStreamingWrapper,
    video_path: Path,
    start_frame: int,
    end_frame: int,
    frame_step: int,
    target_size: tuple[int, int],
    chunk_frames: int,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, list[int]]:
    """对单路摄像头的一个演示片段做因果 Wan VAE 流式编码。

    编码调度严格对齐 LingBot 在线推理流水线：
      1. 用 1 帧 初始块播种因果缓存（满足 1 + 0*4）；
      2. 后续每块送入 ``chunk_frames`` 帧，要求 ``chunk_frames % 4 == 0``；
      3. 若末尾剩余不足 ``chunk_frames`` 但仍满足 4 的倍数，则对输入
         进行因果重复 padding 到 chunk_frames 长度，编码后再裁掉
         padding 对应的输出 latent，保证 VAE 缓存的滑动窗口始终合法。

    参数:
        wrapper: 携带跨 chunk KV 缓存的 WanVAEStreamingWrapper 实例。
                 函数开头会调用 clear_cache() 以确保不同片段/相机相互独立。
        video_path: 该摄像头对应的 LeRobot mp4 文件绝对路径。
        start_frame / end_frame: 演示片段在源视频中的半开区间帧索引，
                                 对应 action_config 中指定的子段范围。
        frame_step: 源视频子采样步长。默认 4，将 240Hz 源视频降至
                    LingBot APF16 契约使用的 60Hz action-slot 频率。
        target_size: VAE 输入目标分辨率 (H, W)，使用双线性插值。
        chunk_frames: 初始块之后每个流式块的源帧数，必须被 4 整除。
        dtype: VAE 前向计算精度（通常为 bfloat16）。

    返回:
        (latent, frame_ids) 元组：
          - latent: shape ``[T_latent, H_latent, W_latent, C]``，
            后续写 payload 时会扁平化为 ``[T*H*W, C]``。
          - frame_ids: 实际参与编码的源帧索引列表，用于与环境动作
            序列精确对应。
    """
    if chunk_frames % 4 != 0:
        raise ValueError(f"Wan 流式块在首帧之后的帧数必须被 4 整除，当前为 {chunk_frames}")
    wrapper.clear_cache()
    device = next(wrapper.vae.parameters()).device
    chunks: list[torch.Tensor] = []
    frame_ids: list[int] = []
    pending: list[np.ndarray] = []

    def encode_pending() -> None:
        """对当前 pending 缓冲中的批次做 resize、归一化并送入 VAE。

        张量布局变化：
          ``list[ndarray(H, W, 3)]``
          -> ``[N, 3, H_resized, W_resized]`` float32, 范围 [0, 255]
          -> ``[1, 3, N, H_resized, W_resized]``, 归一化到 [-1, +1]
        编码结果沿 dim=1 切分以提取后验均值 mu（对数方差部分在提取阶段
        被丢弃，因为我们确定性地存储分布众数用于重建），做 normalize 后
        立即搬回 CPU 以避免跨片段占用 VRAM。
        """
        if not pending:
            return
        video = torch.from_numpy(np.stack(pending)).to(torch.float32).permute(0, 3, 1, 2)
        video = F.interpolate(video, size=target_size, mode="bilinear", align_corners=False)
        # 变换为 VAE 的 (B, C, T, H, W) 输入格式，并做 [0,255] -> [-1,1] 归一化
        video = video.permute(1, 0, 2, 3).unsqueeze(0) / 255.0 * 2.0 - 1.0
        real_frames = video.shape[2]
        if chunks and real_frames % 4 != 0:
            raise ValueError(
                f"VAE 后续块帧数不合法: {real_frames}, 必须为 4 的倍数"
            )
        # AutoencoderKLWan 的流式缓存即使在长度满足 4 整除条件下也可能
        # 拒绝过短的最后一块。因此对不足 chunk_frames 的尾部做因果重复
        # padding 到完整 chunk 长度，使真实前缀内容不受影响；
        # 输出 latent 时再裁掉 padding 对应部分。
        if chunks and real_frames < chunk_frames:
            video = torch.cat(
                [video, video[:, :, -1:].repeat(1, 1, chunk_frames - real_frames, 1, 1)],
                dim=2,
            )
        with torch.no_grad():
            encoded = wrapper.encode_chunk(video.to(device=device, dtype=dtype))
            mu, _ = torch.chunk(encoded, 2, dim=1)
            normalized = normalize_latents(wrapper.vae, mu)
            if chunks and real_frames < chunk_frames:
                normalized = normalized[:, :, : real_frames // 4]
            chunks.append(normalized.cpu())
        pending.clear()

    for frame_index, frame in read_selected_frames(video_path, start_frame, end_frame, frame_step):
        frame_ids.append(frame_index)
        pending.append(frame)
        # Wan 因果 VAE 首块使用 1 帧初始化缓存；之后每块必须含 4 的倍数帧，
        # 与 LingBot 在线推理保持一致。
        target_chunk_size = 1 if not chunks else chunk_frames
        if len(pending) == target_chunk_size:
            encode_pending()
    encode_pending()
    if not chunks:
        raise ValueError(f"未从视频中解码出任何帧: {video_path} [{start_frame}, {end_frame})")

    # 沿时间维拼接 chunk，压缩 batch 维，重排为 [T, H, W, C]。
    # 此布局使得后续 reshape(-1, C) 扁平化时能按"逐帧、行优先"的顺序
    # 排列空间 token，与下游 LatentLeRobotDataset 加载器约定一致。
    latent = torch.cat(chunks, dim=2)[0].permute(1, 2, 3, 0).contiguous()
    expected_latent_frames = (len(frame_ids) - 1) // 4 + 1
    if latent.shape[0] != expected_latent_frames:
        raise ValueError(
            f"视频 {video_path} 时序 latent 维度不匹配: 实际 {latent.shape[0]}, "
            f"按 {len(frame_ids)} 个采样帧应得到 {expected_latent_frames}。"
            "请调整 --chunk-frames 以保持 Wan 因果流式对齐。"
        )
    return latent, frame_ids


def atomic_torch_save(payload: dict, path: Path) -> None:
    """通过 临时文件 + os.replace 原子地写入 torch 序列化 payload。

    防止进程在序列化中途被 kill 时留下半截损坏的 .pth 文件。
    临时文件名嵌入 ``os.getpid()``，避免多进程并发处理同一数据集时
    互相覆盖对方的中间文件。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    torch.save(payload, temp)
    os.replace(temp, path)


def episode_video_path(root: Path, meta: LeRobotDatasetMetadata, episode: int, key: str) -> Path:
    """获取某个 episode 指定摄像头键对应的 mp4 绝对路径。"""
    return root / meta.get_video_file_path(episode, key)


def main() -> int:
    # ---- 命令行参数 ----
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True,
                        help="LeRobot v2.1 数据集根目录（包含 episodes.jsonl / videos/）。")
    parser.add_argument("--model-root", type=Path, required=True,
                        help="Wan2.2 模型根目录，应包含 tokenizer/、text_encoder/、vae/ 子目录。")
    parser.add_argument("--device", default="cuda:0",
                        help="文本编码器与 VAE 使用的 Torch 设备。")
    # frame_step=4：将 240Hz 源视频子采样到 60Hz action-slot 频率（APF16 契约），
    # 每 1 latent 帧对应 16 步环境动作。
    parser.add_argument("--frame-step", type=int, default=4)
    # chunk_frames 必须为 4 的倍数；默认 16 在 GPU 吞吐与因果缓存占用间取得平衡。
    parser.add_argument("--chunk-frames", type=int, default=16)
    parser.add_argument("--max-episodes", type=int, help="可选：仅处理前 N 个 episode。")
    parser.add_argument("--max-segments", type=int, help="可选：仅处理前 N 个 action_config 片段。")
    parser.add_argument("--episode-start", type=int, default=0, help="起始 episode 索引（含）。")
    parser.add_argument("--episode-end", type=int, help="终止 episode 索引（不含）。")
    parser.add_argument("--manifest-name", default="latent_extraction_manifest.json",
                        help="生成的提取清单 JSON 文件名，存于 --dataset-root 下。")
    parser.add_argument("--overwrite", action="store_true",
                        help="即使目标 .pth 已存在也强制重新编码。")
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = torch.bfloat16
    meta = LeRobotDatasetMetadata(repo_id=str(args.dataset_root), root=args.dataset_root)
    source_fps = int(meta.fps)
    frame_step = int(args.frame_step)
    if frame_step <= 0:
        raise ValueError(f"--frame-step 必须为正整数，当前为 {frame_step}")

    # ---- 筛选 episode 与 action 片段 ----
    # meta.episodes 对应 episodes.jsonl 反序列化结果，键为 episode 索引字符串。
    # 先按索引范围过滤，再把每个 episode 的 action_config 列表展平为独立片段：
    # 一条 LeRobot episode 可能包含多个带标注的指令子段，每段都有独立的
    # start_frame / end_frame / action_text 三元组。
    episodes = [
        item for item in sorted(meta.episodes.items())
        if int(item[0]) >= args.episode_start
        and (args.episode_end is None or int(item[0]) < args.episode_end)
    ]
    if args.max_episodes is not None:
        episodes = episodes[: args.max_episodes]
    segments = []
    for episode_index, episode_record in episodes:
        for config in episode_record.get("action_config", []):
            segments.append((int(episode_index), config))
    if args.max_segments is not None:
        segments = segments[: args.max_segments]
    if not segments:
        raise ValueError("未选择任何 action_config 片段")

    # ---- 阶段 1：文本嵌入（两段式显存策略） ----
    # 先对指令文本去重：多个演示常共用同一条任务提示，去重可显著减少编码量。
    # 生成的嵌入常驻 CPU，为后续 VAE 阶段腾出完整 GPU 显存预算。
    # empty_emb.pt 单独持久化，供训练时 Classifier-Free Guidance 读取
    # 无条件嵌入而无需再次加载 CLIP 编码器。
    unique_texts = sorted({str(config["action_text"]) for _, config in segments})
    tokenizer = load_tokenizer(str(args.model_root / "tokenizer"))
    text_encoder = load_text_encoder(
        str(args.model_root / "text_encoder"), torch_dtype=dtype, torch_device=device
    )
    text_embeddings = {
        text: get_text_embedding(tokenizer, text_encoder, text, dtype, torch.device("cpu"))
        for text in tqdm(unique_texts, desc="文本嵌入")
    }
    empty_embedding = get_text_embedding(tokenizer, text_encoder, "", dtype, torch.device("cpu"))
    atomic_torch_save(empty_embedding, args.dataset_root / "empty_emb.pt")
    # 显式释放 CLIP 后再加载 VAE。以 24GB 消费级卡为例，可释放 ~8GB 权重+KV
    # 缓存供 VAE 阶段激活使用。
    del text_encoder, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    # ---- 阶段 2：VAE latent 提取 ----
    vae = load_vae(str(args.model_root / "vae"), torch_dtype=dtype, torch_device=device)
    wrapper = WanVAEStreamingWrapper(vae)
    written = []
    for episode_index, config in tqdm(segments, desc="片段"):
        start = int(config["start_frame"])
        end = int(config["end_frame"])
        text = str(config["action_text"])
        # LeRobot 将 episode 按最多每 1000 条一个目录分片
        episode_chunk = meta.get_episode_chunk(episode_index)
        for camera_key, target_size in CAMERA_SIZES.items():
            # 输出目录结构：dataset_root/latents/chunk-XXX/摄像头键/episode_序号_起始_结束.pth
            output_path = (
                args.dataset_root
                / "latents"
                / f"chunk-{episode_chunk:03d}"
                / camera_key
                / f"episode_{episode_index:06d}_{start}_{end}.pth"
            )
            # 默认跳过已存在文件以支持断点续跑；传 --overwrite 则强制重算
            if output_path.exists() and not args.overwrite:
                written.append(str(output_path))
                continue
            video_path = episode_video_path(args.dataset_root, meta, episode_index, camera_key)
            latent, frame_ids = encode_camera_segment(
                wrapper=wrapper,
                video_path=video_path,
                start_frame=start,
                end_frame=end,
                frame_step=frame_step,
                target_size=target_size,
                chunk_frames=args.chunk_frames,
                dtype=dtype,
            )
            latent_frames, latent_height, latent_width, channels = latent.shape
            # 按帧扁平化空间 token: [T, H, W, C] -> [T*H*W, C]
            # 此布局与 LatentLeRobotDataset 的加载契约严格对齐，
            # 世界模型 Transformer 可直接读取为时序 token 序列。
            payload = {
                "latent": latent.reshape(-1, channels).to(torch.bfloat16),
                "latent_num_frames": latent_frames,
                "latent_height": latent_height,
                "latent_width": latent_width,
                "video_num_frames": len(frame_ids),
                "video_height": target_size[0],
                "video_width": target_size[1],
                "text_emb": text_embeddings[text].to(torch.bfloat16),
                "text": text,
                "frame_ids": frame_ids,
                "start_frame": start,
                "end_frame": end,
                "fps": source_fps / frame_step,
                "ori_fps": source_fps,
            }
            atomic_torch_save(payload, output_path)
            written.append(str(output_path))
            # 以 JSON-lines 形式打印进度，便于上层编排者通过 tail -f 聚合
            # 多个并行 worker 的结果。
            print(json.dumps({
                "event": "latent_saved",
                "episode": episode_index,
                "camera": camera_key,
                "segment": [start, end],
                "video_frames": len(frame_ids),
                "latent_shape": list(latent.shape),
                "path": str(output_path),
            }))

    # ---- 最终清单 manifest ----
    # 汇总本次运行的全部参数与输出文件路径，下游 prepare / 校验脚本
    # 可直接据此校验 latent 目录完整度，而无需遍历枚举动辄 10 万+ 的 pth。
    manifest = {
        "dataset_root": str(args.dataset_root.resolve()),
        "model_root": str(args.model_root.resolve()),
        "source_fps": source_fps,
        "sampled_fps": source_fps / frame_step,
        "frame_step": frame_step,
        "chunk_frames": args.chunk_frames,
        "segments": len(segments),
        "files": written,
    }
    (args.dataset_root / args.manifest_name).write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps({"event": "complete", "files": len(written), "segments": len(segments)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
