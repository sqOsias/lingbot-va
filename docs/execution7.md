# 完整 LingBot-VA 5B 模型的 CPU 分级测试

日期：2026-10-04
仓库：`/root/wyc/code/lingbot-va2`

## 1. 是否应该加载完整模型

是。tiny生产模型的两步语义链路已经通过后，下一项独立证据是完整step-750
checkpoint能否以BF16加载、启用MemoryWAM，并穿过全部30层。但CPU测试不能
替代FSDP/HSDP、CUDA attention、NCCL和显存验证，所以脚本按成本拆成五个阶段，
每个阶段使用新输出目录并保留进度事件。

初始化固定为：

```text
train_out/rearrange_blocks_50demo_1000step_20260921_180241/
checkpoints/checkpoint_step_750
```

权重为10,177,831,668 bytes、839个BF16 key。当前plan实测约484.99GiB可用RAM、
1.94TiB可用磁盘，满足脚本的容量门槛；实际执行前会重新检查。

## 2. 脚本

实现：

```text
script/memorywam_checks/full_model_cpu_smoke.py
script/memorywam_checks/run_full_model_cpu_smoke.sh
script/memorywam_checks/run_full_model_cpu_sequence.sh
```

每次运行生成：

```text
plan.json                 固定输入、资源、脚本hash与限制
events.jsonl              每个已开始/已完成阶段，立即flush并fsync
full_cpu_pass.json        完成时生成
full_cpu_failure.json     Python异常或Ctrl+C时生成
```

如果进程被操作系统直接OOM kill，可能没有failure JSON，但`events.jsonl`仍显示最后
进入的阶段。脚本拒绝覆盖已有目录。

## 3. 推荐执行顺序

### 3.0 一条命令按顺序执行

已经提供串行脚本，固定执行`plan -> load -> forward -> train-gist`，其中
`train-gist`默认为2个optimizer step：

```bash
cd /root/wyc/code/lingbot-va2
bash script/memorywam_checks/run_full_model_cpu_sequence.sh
```

脚本为每个阶段启动独立进程，因此上一阶段的完整模型和中间量会先释放，再进入
下一阶段。任一阶段失败都会立即停止，不会继续消耗资源，也绝不会自动运行
`train-all`。统一输出包含：

```text
artifacts/memorywam/stage8_full_cpu_sequence_<UTC时间>/
  sequence.log
  sequence_status.txt
  plan/
  load/
  forward/
  train-gist/
```

需要指定输出目录时：

```bash
bash script/memorywam_checks/run_full_model_cpu_sequence.sh \
  artifacts/memorywam/my_full_cpu_sequence
```

需要改变gist训练步数时通过环境变量设置；正式的当前门禁仍使用2步：

```bash
MEMORYWAM_CPU_GIST_STEPS=2 \
  bash script/memorywam_checks/run_full_model_cpu_sequence.sh
```

### 3.1 Plan：已执行并通过

```bash
cd /root/wyc/code/lingbot-va2
bash script/memorywam_checks/run_full_model_cpu_smoke.sh plan
```

该阶段只读config和safetensors header，不加载10GB tensor。标准证据：

```text
artifacts/memorywam/stage8_full_cpu_plan_v2_20261004/full_cpu_pass.json
```

### 3.2 Load：建议首先由用户执行

```bash
bash script/memorywam_checks/run_full_model_cpu_smoke.sh load
```

门槛：至少24GiB可用RAM。行为：

- 以BF16、`low_cpu_mem_usage=True`加载完整30层模型。
- CPU强制使用`attn_mode=torch`；原checkpoint config为`flex`。
- 比较全部839个legacy key及shape，要求missing/unexpected/mismatch均为空。
- 加入hybrid M=8/A=2/R=4，并要求`gist_tokens[8,3072]`。
- 记录真实参数量、dtype分布、加载耗时和峰值RSS。

该阶段不执行forward，最适合作为第一项完整模型CPU证据。

### 3.3 Forward：Load通过后执行

```bash
bash script/memorywam_checks/run_full_model_cpu_smoke.sh forward
```

门槛：至少48GiB可用RAM。使用真实episode 0，但把计算压到一个2×2 latent patch、
一个文本token和一个动作token：

```text
完整5B加载
 -> clean observation prefill穿过30层
 -> atomic commit一次
 -> action denoise穿过30层，重复两次
 -> 验证输出有限、重复结果逐bit相同、state完全不变
```

这是完整权重/完整层数的MemoryWAM核心forward，不是APF16性能测试，也不包含
视频/动作loss。

### 3.4 Train-gist：Forward通过后可选执行

```bash
bash script/memorywam_checks/run_full_model_cpu_smoke.sh train-gist -- \
  --steps 2
```

门槛：至少96GiB可用RAM。冻结5B backbone，只训练`gist_tokens`，但前向、反向和
gist梯度仍穿过全部30层。训练裁剪为2帧、每帧一个视频patch、每帧一个动作slot、
一个文本token，并启用activation checkpointing。默认LR为`1e-3`，只用于确认
CPU BF16下gist发生可测更新，不代表正式训练超参数。

需要同时保存完整10GB模型和小optimizer状态时：

```bash
bash script/memorywam_checks/run_full_model_cpu_smoke.sh train-gist -- \
  --steps 2 --save-full-checkpoint
```

### 3.5 Train-all：默认不建议

```bash
bash script/memorywam_checks/run_full_model_cpu_smoke.sh train-all -- \
  --steps 2 --allow-full-backbone-update
```

门槛：至少256GiB可用RAM，并必须显式提供
`--allow-full-backbone-update`。这会对全部5B参数构建梯度和AdamW状态，CPU耗时
可能很长。它仍是单进程、一个动作slot的缩小训练，不能验证FSDP。

如还要保存完整模型和大optimizer状态：

```bash
bash script/memorywam_checks/run_full_model_cpu_smoke.sh train-all -- \
  --steps 2 --allow-full-backbone-update --save-full-checkpoint
```

脚本要求至少80GiB可用输出磁盘。该阶段的收益有限，通常应等待GPU恢复。

## 4. 自定义输出与线程

包装脚本默认创建带时间戳的输出。也可指定目录：

```bash
bash script/memorywam_checks/run_full_model_cpu_smoke.sh load \
  artifacts/memorywam/my_full_load
```

传递Python参数时使用`--`：

```bash
bash script/memorywam_checks/run_full_model_cpu_smoke.sh forward -- \
  --threads 52
```

默认32个CPU线程。更多线程不保证更快，双NUMA机器可能受内存带宽影响。

## 5. 验收与停止条件

| 阶段 | PASS条件 | 失败后动作 |
| --- | --- | --- |
| load | 839 key/shape完整；gist shape正确；有限RSS | 先查missing/shape/dtype，不进入forward |
| forward | prefill提交一次；两次denoise逐bit相同；state不变；输出有限 | 定位最早失败层，不进入backward |
| train-gist | 两步loss有限；gist梯度非零；optimizer执行 | 检查checkpoint wrapper和BF16更新，不运行train-all |
| train-all | 两步loss有限；gist梯度非零；全参数optimizer执行 | 不通过增加步数规避 |
| save | safetensors/config/training_state完整写出 | 输出目录视为失败产物，不作为初始化 |

任何阶段出现swap快速增长、系统内存压力、不可接受的单层耗时或管理员要求释放
资源时应中止。保留events和failure报告即可，不反复重启同一高成本阶段。

## 6. 证据边界

即便`train-all`在CPU通过，仍必须执行GPU两步gate，因为以下路径没有被测试：

- FSDP2参数分片、HSDP DeviceMesh和分布式optimizer。
- flex/CUDA attention与CPU torch attention的数值关系。
- CPU offload、NCCL collective和多卡吞吐。
- 真实APF16完整训练长度、线上VAE、websocket和环境执行。

因此推荐实际顺序为：`load -> forward -> train-gist`。`train-all`是可用但低收益的
诊断项，不是GPU恢复前的必需门禁。
