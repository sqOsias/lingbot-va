# MemoryWAM 生产接入操作文档：GPU 门禁延后与 CPU 可验证部分

日期：2026-10-04
适用仓库：`/root/wyc/code/lingbot-va2`
前置文档：[execution1.md](./execution1.md)、[execution2.md](./execution2.md)

## 1. 本轮决定

当前 GPU 不可用时，可以继续生产接入，但不能用历史训练日志替代当前 5B 数值回归。实际采用以下门禁调整：

1. **阶段 0A（静态 legacy 冻结）已完成。** 固定代码状态、step-750 checkpoint、fullctx 数据、配置和历史结果；checkpoint 与数据文件均记录 SHA256。
2. **阶段 0B（当前工作树加载 5B 的数值回归）延后。** 历史日志证明旧运行确实完成到 optimizer step 750，并有对应验证 loss；它不证明修改后的 `memory off` 前向与旧工作树逐值一致。
3. **小模型机制验证不需要跳过，也不需要 GPU。** typed token、四种 policy、因果 mask、streaming/prefix 等价、事务提交、去噪只读及 gist 双 loss 梯度均已有 CPU 测试，本轮再次运行通过。
4. **生产接入允许继续，但全部默认关闭。** 未设置 MemoryWAM 训练开关、或 checkpoint 配置未声明 MemoryWAM 时，Dataset 返回原四个字段，Transformer 走原 `forward_train`，服务走原 KV cache。
5. **不启动正式训练或闭环评测。** 当前生产训练实现是可审计的完整 episode reference，每个目标时刻分别做 prefill、视频 query 和动作 query；它用于正确性 smoke，不是吞吐优化后的正式训练实现。

因此，本轮“跳过”的是需要 GPU 的 5B 现场数值门禁，不是把它判定为通过。恢复 GPU 后必须补做该门禁，才能把 legacy 兼容性标记为完成。

## 2. 已冻结的 legacy 证据

静态清单位于：

```text
artifacts/memorywam/stage0_legacy_static_20261004/manifest.json
artifacts/memorywam/stage0_legacy_static_20261004/working_tree.patch
```

固定 checkpoint：

```text
train_out/rearrange_blocks_50demo_1000step_20260921_180241/
└── checkpoints/checkpoint_step_750/transformer/
```

固定训练数据：

```text
/root/wyc/data/rmbench/lingbot_va2/
rearrange_blocks_50demo_fullctx_apf16_cf8
```

`training_summary.json` 证明该运行实际在 step 750 因 held-out validation loss 收敛而停止；最后记录的 latent/action validation loss 分别为 `0.1266236628` 和 `0.0022535763`。这属于历史日志证据，不是当前代码现场复算结果。旧 checkpoint 可作为共享骨干初始化和 legacy 诊断对象；它没有 `gist_tokens`，不能直接作为 MemoryWAM 推理 checkpoint，也不能称为完整断点续训。

尚未生成：

```text
artifacts/memorywam/legacy_current/reference.pt
artifacts/memorywam/legacy_current/compare.json
```

所以 `memory off` 的 5B 输出、loss 和完整去噪结果仍是延后门禁。

## 3. 已接入的生产代码

| 文件 | 已实现行为 | 默认关闭时行为 |
| --- | --- | --- |
| `wan_va/modules/memorywam.py` | 不可变逐层 K/V state、typed clean/gist token、anchor/recent/gist/full policy、delta/commit、版本与 receipt 幂等检查 | 不创建 state，不进入模型调用 |
| `wan_va/modules/model.py` | attention 显式读取外置 memory；逐帧 clean prefill；只读 video/action denoise；完整 episode 因果 reference；共享 `[M,D]` gist 参数 | `gist_tokens=None`，state dict 无新增 key，沿用原 forward/cache |
| `wan_va/dataset/lerobot_latent_dataset.py` | 可选 sidecar，返回 episode、全局 latent frame、源 RGB、target frame/action 与有效 mask | sidecar 未设置时仍只返回 `latents/actions/actions_mask/text_emb` |
| `wan_va/train.py` | MemoryWAM 模式检查 sidecar，模型加载后且 FSDP/optimizer 前注册 gist，应用视频 target mask | 原 Dataset、噪声增强、loss 和 forward 路径不变 |
| `wan_va/configs/va_rmbench_train_cfg.py` | 环境变量控制开关和 M/A/R 预算 | `RMBENCH_MEMORYWAM_ENABLED` 默认为 `0` |
| `wan_va/wan_va_server.py` | MemoryWAM checkpoint 自动进入 action-only 分支；真实 observation 单次 prefill/commit；所有去噪只读；reset 清状态；request/observation ID 支持幂等；冲突图像拒绝 | legacy checkpoint 仍创建并使用原缓存 |

生产模型采用上一轮已固定的因果合同：观察到 `z[t]` 后提交 `f[t]+g[t]`，视频 target `z[t+1]` 与动作 group `t+1` 分别作为 noisy query 读取同一 `C(t)`。动作训练不读取 target `z[t+1]` 的 clean token，避免未来真值泄漏。预测视频不写持久状态。

当前服务保持旧 RMBench 客户端兼容：第一次 action response 前置一个 dummy action frame，并返回 `action_start_index=1`；后续每次只返回一个 APF16 action frame并返回 `action_start_index=0`。服务不再把客户端 `state` 中的整段预测计划写入 MemoryWAM；下一状态只由执行后回传的真实 RGB 生成。

## 4. 数据验证结果

生产 Dataset 的全量 CPU 回归产物：

```text
artifacts/memorywam/stage5_production_cpu/
├── rearrange_train_dataset_regression.json
└── rearrange_validation_dataset_regression.json
```

| split | episode | latent frame | 有效 action slot | 原四类 tensor |
| --- | ---: | ---: | ---: | --- |
| train50 | 50 | 1281 | 19696 | 每个样本 `rtol=0, atol=0` 相等 |
| validation5 | 5 | 128 | 1968 | 每个样本 `rtol=0, atol=0` 相等 |

检查还验证了：全局 latent frame 为每 episode 的 `0..F-1`；首 action group 完全 dummy；`target_frame_ids == global_frame_ids[1:]`；sidecar 的逐帧有效动作数与正式 `actions_mask` 一致；每个 fullctx 样本显式 reset。

这些结果证明现有 fullctx latent、文本 embedding、动作和统计不需要重提取。MemoryWAM 机制版只需要 sidecar 和新的采样/训练语义。严格论文版的 RGB mosaic、joint14/proprio 和独立动作专家仍是另一个数据/模型版本。

## 5. CPU 验证命令

从仓库根目录运行完整检查：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
/root/miniconda3/envs/lingbot/bin/python -B \
  -m script.memorywam_checks.run_checks
```

本轮结果：`Ran 24 tests ... OK`。覆盖隔离机制、生产 Dataset、真实 `WanTransformer3DModel`、activation checkpoint、gist 视频/动作梯度、服务单 observation commit、幂等与冲突提交。

重新运行 train50 全量生产 Dataset 回归：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
/root/miniconda3/envs/lingbot/bin/python -B \
  -m script.memorywam_checks.verify_production_dataset \
  --dataset-root /root/wyc/data/rmbench/lingbot_va2/rearrange_blocks_50demo_fullctx_apf16_cf8 \
  --alignment artifacts/memorywam/stage4_alignment/rearrange_train/episodes.jsonl \
  --output artifacts/memorywam/stage5_production_cpu/rearrange_train_dataset_regression.json
```

validation5 使用：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
/root/miniconda3/envs/lingbot/bin/python -B \
  -m script.memorywam_checks.verify_production_dataset \
  --dataset-root /root/wyc/data/rmbench/lingbot_va2/rearrange_blocks_5demo_validation_fullctx_apf16_cf8 \
  --alignment artifacts/memorywam/stage4_alignment/rearrange_validation/episodes.jsonl \
  --output artifacts/memorywam/stage5_production_cpu/rearrange_validation_dataset_regression.json
```

## 6. GPU 恢复后的第一个动作

先补阶段 0B，不直接训练。使用冻结的 step-750、validation episode 0、固定初始 video/action noise、相同 torch attention backend 和相同去噪 schedule，生成 `memory off` reference；随后在当前生产代码仍关闭 memory 的条件下复算并比较。

通过条件：

```text
checkpoint/config/输入哈希一致
模型 key 无未声明 missing/unexpected
固定前向与完整动作解码在 atol=1e-5, rtol=1e-4 内一致
服务 reset/infer/cache-update 响应形状和 frame_st_id 一致
```

若 0B 通过，再做最多 1～2 optimizer step 的 MemoryWAM 5B smoke。建议环境如下；该命令目前仅记录，不应在无可用 GPU 时运行：

```bash
export RMBENCH_DATASET_PATH=/root/wyc/data/rmbench/lingbot_va2/rearrange_blocks_50demo_fullctx_apf16_cf8
export RMBENCH_VALIDATION_DATASET_PATH=/root/wyc/data/rmbench/lingbot_va2/rearrange_blocks_5demo_validation_fullctx_apf16_cf8
export LINGBOT_PRETRAINED_MODEL=/root/wyc/code/lingbot-va2/train_out/rearrange_blocks_50demo_1000step_20260921_180241/checkpoints/checkpoint_step_750
export RMBENCH_MEMORYWAM_ENABLED=1
export RMBENCH_MEMORY_ALIGNMENT_PATH=/root/wyc/code/lingbot-va2/artifacts/memorywam/stage4_alignment/rearrange_train/episodes.jsonl
export RMBENCH_VALIDATION_MEMORY_ALIGNMENT_PATH=/root/wyc/code/lingbot-va2/artifacts/memorywam/stage4_alignment/rearrange_validation/episodes.jsonl
export RMBENCH_MEMORYWAM_GIST_TOKENS=8
export RMBENCH_MEMORYWAM_ANCHOR_FRAMES=2
export RMBENCH_MEMORYWAM_RECENT_FRAMES=4
```

不要直接使用配置默认的 `num_steps=3000` 做 smoke；先增加受审查的短运行覆盖或命令行步数覆盖，并记录 GPU、峰值显存、每个 forward 时间、gist gradient、保存后 reload key。当前完整 episode reference 是 `O(F)` 次逐帧模型调用，5B 吞吐和显存尚未实测。

## 7. MemoryWAM 服务协议

MemoryWAM checkpoint 的 `transformer/config.json` 会保存 `memorywam_enabled=true` 和 M/A/R，服务加载后自动进入新分支。旧 checkpoint 没有这些字段，因此不会误开随机 gist。

推荐请求顺序：

```python
# reset
{"reset": True, "episode_id": "eval-0001", "prompt": prompt}

# first decision; RGB is current observation z[0]
{
    "request_id": "eval-0001-request-0",
    "observation_id": "eval-0001-observation-0",
    "global_frame_id": 0,
    "obs": current_observation,
}

# after executing the returned APF16 group, commit only real RGB samples
{
    "compute_kv_cache": True,
    "observation_id": "eval-0001-observation-1",
    "global_frame_id": 1,
    "obs": four_post_action_rgb_samples,
}

# next action request reads memory version 2 without writing it
{"request_id": "eval-0001-request-1", "obs": None}
```

相同 `request_id` 返回已缓存的相同 action response；相同 `observation_id` 与相同图像返回当前版本且不重复提交；相同 ID 配不同图像报错；非连续 `global_frame_id` 报错。没有显式 ID 的旧客户端仍可工作，但网络重试的端到端幂等性没有证据，不能宣称通过该门禁。

## 8. 当前完成边界与下一门禁

已经完成：静态 legacy 资产冻结、fullctx 正式 Dataset 无损接入、生产外置记忆 state、逐层 KV 读取、gist 参数、完整 episode 因果 reference、loss mask、FSDP 前参数注册、服务 action-only 事务分支，以及 CPU 单元/全量数据验证。

尚未完成：

- 当前工作树的 5B `memory off` 数值回归。
- 5B MemoryWAM forward/backward、FSDP 保存/重载和显存/吞吐 smoke。
- 在线 VAE 四张 RGB 到一个 latent 的 GPU 数值检查。
- 使用显式 ID 的真实 websocket/RMBench 往返与异常重试测试。
- 训练吞吐优化后的 episode-ordered chunk/TBPTT 实现。
- hybrid/recent-only/gist-only/full-history 的生产配置选择与公平训练消融。

因此，代码已达到“生产路径默认关闭接入、可做 GPU smoke”的状态，还没有达到“MemoryWAM 已训练并可正式闭环评测”的状态。下一步唯一顺序是：补 0B legacy 数值门禁 → 1～2 step 5B MemoryWAM smoke → 显式 ID 服务单 episode 闭环 → 再决定训练吞吐实现和正式预算。
