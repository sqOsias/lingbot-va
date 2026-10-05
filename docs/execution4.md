# MemoryWAM 无 GPU 训练准备与恢复 GPU 后启动流程

日期：2026-10-04
前置状态：[execution3.md](./execution3.md)

当前完整代码变动、验证证据和剩余阻塞项汇总见 [MemoryWAM_current_progress_20261004.md](./MemoryWAM_current_progress_20261004.md)。

## 1. 结论

无 GPU 阶段可以完成且本轮已经完成的内容包括：

- 5B checkpoint **只读 header** 与当前模型结构逐 key/shape 比较，不加载权重内容。
- MemoryWAM 新参数集合检查和小模型 checkpoint 保存/加载 round-trip。
- train50/validation5 正式 Dataset 的全量无损回归。
- sidecar 唯一性、哈希、episode/decision 数和时间合同检查。
- production Transformer、训练 loss、activation checkpoint、服务事务的 CPU 测试。
- checkpoint 保存失败的强制失败语义。
- GPU smoke 的固定数据、初始化、policy、输出目录和验收脚本。

GPU 恢复后可以直接启动**两步 GPU smoke**。不能把它省略后直接提交 3000-step 正式训练：当前完整 episode 实现是正确性 reference，尚无 5B 峰值显存和 step latency；两步 smoke 的目的就是决定该实现能否直接扩大预算，还是必须先实现顺序 chunk/TBPTT 或并行 mask 优化。

## 2. CPU 预检已验证事实

预检产物：

```text
artifacts/memorywam/stage6_training_ready/cpu_preflight.json
```

本轮结果为 `PASS`：

| 项目 | 结果 |
| --- | --- |
| 初始化 checkpoint | `train_out/rearrange_blocks_50demo_1000step_20260921_180241/checkpoints/checkpoint_step_750` |
| 权重文件 | consolidated safetensors，10,177,831,668 bytes |
| legacy key | 839 个；missing=0，unexpected=0，shape mismatch=0 |
| MemoryWAM 新参数 | 仅 `gist_tokens` |
| gist shape | `[8,3072]` |
| train alignment | 50 records、1231 decisions，SHA256 已写入报告 |
| validation alignment | 5 records、123 decisions，SHA256 已写入报告 |
| train tensor 回归 | 50 episodes、1281 latent frames、19696 action slots，逐值相等 |
| validation tensor 回归 | 5 episodes、128 latent frames、1968 action slots，逐值相等 |
| CPU tests | 30 项通过（含生产消融模式、FSDP/HSDP 拓扑与历史日志审计） |
| checkpoint round-trip | tiny production MemoryWAM config、policy、gist tensor 逐值一致 |

上述检查没有读取 test split，也没有加载 5B tensor 内容。

## 3. 一键重跑 CPU 门禁

```bash
cd /root/wyc/code/lingbot-va2
script/memorywam_checks/run_training_cpu_preflight.sh
```

该脚本先运行全部 30 项 CPU 测试，再运行 5B header/meta model 比较及 train/validation 全量 Dataset 回归。成功时最后输出：

```text
CPU PREFLIGHT PASS: .../cpu_preflight.json
```

若修改以下任一内容，必须重跑：模型参数名/shape、attention、memory policy、Dataset、sidecar、动作 mask、checkpoint 初始化路径或训练配置。

## 4. 已固定的首次 GPU 训练条件

| 项目 | 固定值 |
| --- | --- |
| task | `rearrange_blocks` |
| initializer | legacy step-750，只作为权重初始化，不是完整 resume |
| train | `rearrange_blocks_50demo_fullctx_apf16_cf8` |
| validation | `rearrange_blocks_5demo_validation_fullctx_apf16_cf8` |
| action | 现有 EEF16→model30，APF16 |
| policy | gist=8、anchor=2、recent=4 |
| detach | 0；完整 episode reference 保留跨帧梯度 |
| batch | 每 rank 1；world size 由 `NGPU` 显式记录 |
| init/load | 839 个 legacy key 严格匹配；新增 gist 随机初始化并训练 |
| smoke | 2 optimizer steps，W&B 默认 disabled，early stop disabled |
| checkpoint | 第2步保存，必须含 `gist_tokens` 和 memory config |

首次 smoke 不使用 test split，不做闭环成功率，不改变 joint/action/RGB 表示。

## 5. GPU 恢复后的唯一启动命令

默认按历史训练的 3-rank 形式启动；若可用 GPU 数不同，显式设置 `NGPU`，并把 world size 差异记录为新的资源配置：

```bash
cd /root/wyc/code/lingbot-va2
NGPU=3 MASTER_PORT=29681 \
  script/rmbench_train/train_memorywam_rearrange.sh 2
```

脚本在加载模型前检查：

- CPU preflight 必须为 `PASS`。
- preflight 内记录的生产代码 fingerprint 必须与当前文件逐字节一致；代码变化后旧 PASS 自动失效。
- 输出目录不得已存在。
- `nvidia-smi` 报告的设备数不少于 `NGPU`。
- initializer、train/validation sidecar 和固定配置路径必须可读。

训练结束后自动检查：

- 实际完成恰好2个 optimizer step。
- video loss、action loss、总 grad norm、gist grad norm 均有限。
- 每步 gist grad norm 大于0。
- 第2步 `gist_update_max>0`，确认 warmup 后新增参数实际更新。
- checkpoint 保存成功且包含 `gist_tokens`。
- checkpoint config 含 `memorywam_enabled=true`、M=8。
- 记录所有 rank 的最大 `memory_allocated` 和 `memory_reserved`。

通过后生成：

```text
<output>/gpu_smoke_pass.json
```

## 6. 两步 smoke 的切换条件

只有同时满足以下条件，才允许增加训练步数：

1. 先补做 [execution3.md](./execution3.md) 中的 5B memory-off 数值门禁，或明确把当前 run 标为“新实现 smoke”，不用于 legacy 公平比较。
2. 两步 loss、grad、gist grad 全部有限且非零。
3. 第2步发生参数更新；当前 warmup 的第一步学习率可能为0，因此不能只看第一步。
4. checkpoint header/config 完整，并至少完成一次实际 reload forward。
5. 峰值显存留有足够余量，单步耗时可接受。
6. FSDP/activation-checkpoint 重算没有重复 commit 或 state version 改变。

如果完整 episode reference 的单步时间或显存不可接受，停止扩大步数，优先实现 episode-ordered chunk/TBPTT。不要通过 `detach_interval=1` 直接降显存：它会切断 target loss 到当前 committed gist K/V 的梯度，改变要验证的机制。

## 7. 通过 smoke 后的受控训练命令

当前脚本允许复用同一固定条件增加步数，但要求显式提供两步 gate，并确认仍在使用 reference 实现。例如先做 50-step 工程验证：

```bash
export MEMORYWAM_GPU_GATE=/absolute/path/to/gpu_smoke_pass.json
export MEMORYWAM_ALLOW_REFERENCE_TRAINING=1
export RMBENCH_SAVE_INTERVAL=50

NGPU=3 MASTER_PORT=29682 \
  script/rmbench_train/train_memorywam_rearrange.sh 50
```

50-step 仍属于工程验证。是否进入正式预算，应根据实测吞吐、验证 loss、checkpoint reload 和单 episode 服务推理决定。脚本对超过2步的运行默认启用独立 validation early stopping，并每50步保存一次权重。

## 8. 仍需 GPU 才能回答的问题

- 当前工作树 memory-off 5B 是否与冻结 reference 数值一致。
- 5B FSDP2 是否正确分片新增 gist 参数及其 optimizer state。
- 完整 episode 反向的峰值显存、通信开销和 step latency。
- consolidated checkpoint 是否能在实际 5B 模型上完整 reload。
- 在线 streaming VAE 的四 RGB→一 latent 是否与离线 fullctx 一致。
- websocket 重试、真实 RMBench 执行和 observation commit 是否端到端幂等。

## 9. 当前限制

`Trainer` 目前保存完整模型权重和模型 config，但历史代码没有启用 optimizer/scheduler/RNG/data-cursor 的完整恢复。因此：

- step-750 是权重初始化，不是完整断点续训。
- 本轮 MemoryWAM checkpoint 在完成 optimizer-state 实现和 GPU 验证前，也只能保证权重恢复。
- 正式长训练若要求故障续跑，应先实现并验证完整 training state；不要用重新加载权重并把 step 改回去冒充无缝 resume。

保存过程中任何异常现在会直接终止训练，不再吞掉异常后生成成功 summary。
