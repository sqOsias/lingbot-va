# MemoryWAM CPU 两步完整语义链路验证

日期：2026-10-04
仓库：`/root/wyc/code/lingbot-va2`

## 1. 验证目标与边界

GPU不可用时不能在CPU上现实地运行5B模型，也不能验证FSDP/HSDP、CUDA
attention、NCCL、显存或GPU吞吐。本检查验证可独立成立的训练语义链路：

```text
正式fullctx Dataset + alignment sidecar
 -> 真实latent/text/action/action mask
 -> Trainer diffusion noise与grid构造
 -> 生产MemoryWAM完整episode forward
 -> video/action loss
 -> backward
 -> AdamW optimizer更新
 -> 重复2个optimizer step
 -> 模型、optimizer、RNG状态保存
 -> 模型和optimizer重载
 -> 重载后forward比较
```

它是GPU gate之前的CPU预检，不替代`train_memorywam_rearrange.sh`的5B两步smoke。

## 2. 输入与缩放

实际读取：

```text
/root/wyc/data/rmbench/lingbot_va2/
  rearrange_blocks_50demo_fullctx_apf16_cf8
artifacts/memorywam/stage4_alignment/rearrange_train/episodes.jsonl
```

选择episode 0。原始tensor：

```text
latent  [48,26,24,20] bfloat16
text    [512,4096]     bfloat16
action  [30,26,16,1]  float32
mask    [30,26,16,1]  bool
```

CPU smoke保留真实48 latent通道、4096文本维度、model30动作和APF16，仅取前三个
latent帧、4×4 latent空间和前16个文本token。模型使用2层、2 heads、hidden=16、
BF16、activation checkpointing和hybrid M=2/A=1/R=1，共150,046参数。

## 3. 执行

```bash
cd /root/wyc/code/lingbot-va2
bash script/memorywam_checks/run_cpu_two_step_smoke.sh
```

脚本拒绝覆盖已有输出。需要保留多次结果时传入新目录：

```bash
bash script/memorywam_checks/run_cpu_two_step_smoke.sh \
  artifacts/memorywam/stage7_cpu_2step_run2
```

实现：`script/memorywam_checks/cpu_two_step_smoke.py`
标准结果：`artifacts/memorywam/stage7_cpu_2step_pass/cpu_2step_pass.json`

## 4. 验收项

- 正式Dataset只选择episode 0，首帧dummy action必须全mask，后续必须存在有效动作。
- video loss、action loss、total loss均为有限值。
- 两步gist gradient和gist参数更新均非零。
- safetensors、config和包含optimizer/RNG的`training_state.pt`均写出。
- optimizer state和step=2可重载。
- safetensors序列化值逐tensor精确恢复；模块策略允许部分BF16权重重载为FP32。
- 原模型及重载模型各自重复forward逐bit一致。
- 保存前与重载后输出最大绝对差不超过0.02；该差异来自生产保存路径把所有
  state tensor量化为BF16后，部分Norm/time-embedding参数重载为FP32。

## 5. 本轮结果

状态：`PASS`。

| step | video loss | action loss | gist grad norm | gist update max |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 3.4513404 | 0.8879023 | 0.0215649 | 0.001953125 |
| 2 | 2.6816492 | 0.9620216 | 0.0182413 | 0.001953125 |

标准运行的训练/保存/重载段耗时2.10秒，进程峰值RSS约902.85MiB。
checkpoint序列化值精确恢复；同模型重复forward最大差为0；保存前对重载后最大
绝对差为0.015625，低于预设0.02。

前两次诊断目录保留了失败现场：第一次直接要求保存前/后逐bit一致；第二次发现
保存BF16与重载FP32的dtype策略差异。最终脚本先验证序列化值精确恢复，再独立
报告量化输出误差，没有用放宽容差掩盖缺失权重。

## 6. 仍必须在GPU上验证

- step-750初始化后的5B memory-off数值回归。
- 5B MemoryWAM两步前后向和gist optimizer更新。
- 全world FSDP与2×2 HSDP DeviceMesh。
- 分布式optimizer state、checkpoint保存与完整重载。
- 真实峰值显存、latent frames/s和多卡通信。
- 在线VAE、websocket和RMBench环境执行。
