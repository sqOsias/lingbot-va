# MemoryWAM 无 GPU 收尾与多卡训练拓扑

日期：2026-10-04
适用仓库：`/root/wyc/code/lingbot-va2`

本文记录 GPU 不可用期间已经完成的工作、仍可完成但不应替代 GPU
smoke 的检查，以及 GPU 恢复后的唯一测速顺序。本文不会把 CPU 静态检查写成
5B FSDP/HSDP 已运行通过。

## 1. 当前可以提前完成且本轮已执行的工作

### 1.1 多卡历史证据盘点

新增：`script/memorywam_checks/audit_multigpu_history.py`

执行：

```bash
/root/miniconda3/envs/lingbot/bin/python \
  -m script.memorywam_checks.audit_multigpu_history \
  --train-out train_out \
  --output artifacts/memorywam/stage6_training_ready/multigpu_history.json
```

本轮结果：

- 扫到 16 个具有 world-size 或 collective timeout 证据的运行。
- 16 个都记录 `World size: 3`。
- 4 个 debug 运行包含 collective/all-gather timeout 文本。
- 没有保留下来的 4-rank `training.log`，因此目前不能从历史日志定量证明
  四卡比三卡慢；四卡变慢是用户运行观察，源码能解释其可能原因。
- 老指标只有累计 `elapsed_seconds`。由相邻记录计算出的 step delta 可能包含
  validation/save，只可作同一运行内的粗略证据。

### 1.2 FSDP/HSDP 拓扑配置与 CPU 校验

修改：`wan_va/distributed/fsdp.py`

- `resolve_fsdp_topology(world_size, shard_size)` 是不初始化 CUDA 的纯函数。
- `shard_size=0` 保持原行为，即在全部 rank 上做一维 FSDP。
- `world_size=4, shard_size=2` 构造 `(replicate=2, shard=2)` 的二维
  DeviceMesh，由 FSDP2 执行 HSDP。
- world size 不能整除 shard size 时立即报错。例如 4 张卡不能组成“3 卡分片
  + 1 卡副本”的同步训练拓扑。
- 原默认仍为 CPU offload、所有子模块 forward 后 reshard、root forward 后
  reshard，避免在没有 GPU 回归前改变 legacy 数值路径。

修改：`wan_va/configs/va_rmbench_train_cfg.py`

新增环境变量：

| 变量 | 默认值 | 说明 |
| --- | ---: | --- |
| `RMBENCH_FSDP_SHARD_SIZE` | `0` | `0` 表示全 world 分片；四卡设 `2` 表示 2×2 HSDP |
| `RMBENCH_FSDP_CPU_OFFLOAD` | `1` | 保持历史默认；显存允许时测速 `0` |
| `RMBENCH_FSDP_RESHARD_AFTER_FORWARD` | `1` | 控制已包装子模块；设 `0` 会显著增加显存 |
| `RMBENCH_FSDP_ROOT_RESHARD_AFTER_FORWARD` | `1` | 单独控制 root；默认保持旧行为 |

修改：`wan_va/train.py`

- 训练初始化时解析并打印 world/shard/replicate/HSDP/offload/reshard。
- 相同 mesh 传给所有 nested `fully_shard`，防止每层拓扑不一致。
- 每个 optimizer step 记录各 rank 最大的
  `optimizer_step_seconds`。
- 同时记录所有 rank 实际处理的 `global_latent_frames` 和
  `global_latent_frames_per_second`。fullctx episode 长度不同，后者比单看
  秒/step 更适合比较三卡和四卡。

修改：`script/rmbench_train/train_memorywam_rearrange.sh`

- 加载 5B 前先验证 shard 拓扑。
- `run_config.json` 固定四个 FSDP 开关。
- 2-step gate 要求时间和 latent-frame throughput 为正且有限。

新增 CPU 测试：

- `test_fsdp_topology.py`：legacy 三卡、四卡 2×2 HSDP、非法不均匀分组。
- `test_multigpu_history.py`：world-size、timeout 和 legacy timing 解析。

本轮完整 CPU suite：30/30 通过。GPU 不可用警告符合当前环境，不影响这些
CPU 检查。

### 1.3 生产消融模式

生产模型和训练入口现已显式支持：

```text
RMBENCH_MEMORYWAM_MODE=hybrid
RMBENCH_MEMORYWAM_MODE=recent-only
RMBENCH_MEMORYWAM_MODE=gist-only
RMBENCH_MEMORYWAM_MODE=full-history
```

模式写入 transformer checkpoint config 和 `run_config.json`。hybrid/gist-only
创建可训练 gist 参数；recent-only/full-history 不创建无意义的零长度参数，
prefill 直接构造零长度临时张量。四种模式的小模型 checkpoint round-trip 以及
prefill/commit/denoise 已通过。该测试实际发现并修复了 non-persistent meta buffer
在 `from_pretrained` 后不可物化的问题。GPU gate 会核对 policy mode；需要 gist
的模式还会要求 gist gradient 和 update 非零。

### 1.4 训练前门禁

`preflight_training.py` 的代码 fingerprint 已加入
`wan_va/distributed/fsdp.py`。因此以后改变分片实现会使旧 preflight 失效，
必须重新运行：

```bash
bash script/memorywam_checks/run_training_cpu_preflight.sh
```

该命令检查 checkpoint header、MemoryWAM 新参数集合、train50/validation5
sidecar 和 Dataset 无损兼容；不加载 5B 权重内容到 RAM/GPU。

## 2. 为什么四卡可能比三卡慢

当前实现不是“把一个模型切开后每张卡只算自己的层”。FSDP2 的每个 rank
都处理自己的 batch，并在执行每个被包装模块前 all-gather 参数、forward 后
重新分片；backward 还需要参数收集和梯度通信。world size 从 3 变 4 后：

1. 每卡常驻参数状态下降，但 collective 参与者和同步开销上升。
2. 当前 `fsdp_cpu_offload=True`，参数/梯度还要经过 CPU 与 PCIe；第四个 rank
   会增加主存带宽、pinned memory 和 PCIe 竞争。
3. `attn1`、`attn2`、`ffn`、block 和 root 都被包装，且
   `reshard_after_forward=True`，会产生很多 all-gather/reshard。
4. MemoryWAM 完整 episode reference 会逐帧多次穿过 Transformer，使上述开销
   随 episode 长度重复。
5. 当前 batch/GPU=1、gradient accumulation=1。三卡每 step 处理 3 个样本，
   四卡处理 4 个样本；四卡 step 更慢并不自动表示吞吐更低。必须同时比较
   latent frames/s。

本地已有的四次 3-rank debug timeout 能证明 all-gather 路径曾经失败，不能
证明正常运行时 collective 是唯一瓶颈。需要 GPU profile 才能区分通信、CPU
offload、数据加载和逐帧 reference 计算。

## 3. 推荐解决顺序

### 3.1 先保持三卡 legacy 作为计时基准

```bash
NGPU=3 \
RMBENCH_FSDP_SHARD_SIZE=0 \
RMBENCH_FSDP_CPU_OFFLOAD=1 \
bash script/rmbench_train/train_memorywam_rearrange.sh 2 \
  train_out/memorywam_smoke_3gpu_legacy
```

这是历史拓扑，不是最终性能配置。它给出同一份新代码上的可比较基准。

### 3.2 复现四卡全分片和 CPU offload

先保持历史offload设置，只改变world size，以复现并量化用户观察：

```bash
NGPU=4 \
RMBENCH_FSDP_SHARD_SIZE=0 \
RMBENCH_FSDP_CPU_OFFLOAD=1 \
bash script/rmbench_train/train_memorywam_rearrange.sh 2 \
  train_out/memorywam_smoke_4gpu_legacy
```

### 3.3 四卡测试全分片且关闭 CPU offload

仅在四卡显存允许时运行：

```bash
NGPU=4 \
RMBENCH_FSDP_SHARD_SIZE=0 \
RMBENCH_FSDP_CPU_OFFLOAD=0 \
bash script/rmbench_train/train_memorywam_rearrange.sh 2 \
  train_out/memorywam_smoke_4gpu_no_offload
```

原因：若瓶颈来自 CPU/PCIe，去掉 offload 比改变 shard group 更直接。若 OOM，
该配置记为“不满足容量”，不降低模型、序列或 memory budget 来掩盖问题。

### 3.4 测试四卡 2×2 HSDP

```bash
NGPU=4 \
RMBENCH_FSDP_SHARD_SIZE=2 \
RMBENCH_FSDP_CPU_OFFLOAD=1 \
bash script/rmbench_train/train_memorywam_rearrange.sh 2 \
  train_out/memorywam_smoke_4gpu_hsdp_2x2
```

该配置把参数 all-gather 限制在两卡组内，并在两个组之间复制训练。代价是每卡
参数、梯度和 optimizer state 大于四卡全分片，而且副本维度仍需同步梯度。
它是需要实测的候选，不保证比四卡全分片快。

如果显存允许，可把 HSDP 的 CPU offload 也设为0，作为后续可选诊断点。第一轮
不要改reshard；只有基础矩阵表明通信仍是瓶颈且显存有明显余量时，再测试：

```bash
RMBENCH_FSDP_ROOT_RESHARD_AFTER_FORWARD=0
```

不要首先把所有子模块的 `RMBENCH_FSDP_RESHARD_AFTER_FORWARD` 设为 0；这可能
在一个 forward 内保留大量完整参数，显存影响更大。

### 3.5 选择规则

每个候选先运行2-step容量/正确性 gate。通过后，使用该候选自己的
`gpu_smoke_pass.json` 放行5-step短程计时；启动脚本会拒绝 policy、world size
或 FSDP 拓扑不一致的 gate。性能比较去掉第一个 warm-up step，至少使用后续
4个 step 的中位数，并同时比较：

- `global_latent_frames_per_second`，越高越好；
- `optimizer_step_seconds`；
- `peak_memory_allocated_bytes`、`peak_memory_reserved_bytes`；
- loss、grad norm、gist grad/update 是否有限；
- checkpoint save/reload 是否通过。

选择吞吐最高且保留至少 15% 峰值显存余量的配置。四卡所有候选吞吐都不高于
三卡时，正式训练继续用三卡，第四卡用于独立评测或数据服务。不要强行构造不均匀
3+1 FSDP。若未来有 6 卡，可以测试 `(replicate=2, shard=3)` HSDP。

Pipeline parallel、tensor parallel 或把不同 MemoryWAM 帧手工分到不同 GPU
都会改动模型执行语义和服务路径，不属于当前最小改造；只有以上 FSDP/HSDP
矩阵仍无法满足容量/吞吐时再评估。

## 4. GPU 恢复前仍可做的工作

以下工作不需要 GPU，但优先级低于当前已经完成的门禁：

1. 为训练状态补齐 optimizer、scheduler、RNG、sampler/data cursor 的真正恢复
   合同和 CPU 序列化测试。FSDP optimizer state 的跨 rank 恢复仍必须用 GPU 验证。
2. 生成 Observe/Blocks Ranking 的 alignment sidecar 与全量 Dataset 无损报告。
   首个 Rearrange 实验已具备所需数据，因此这不会阻塞第一个 GPU smoke。
3. 固定 GPU smoke 的设备 UUID、`nvidia-smi topo -m`、驱动、PyTorch/NCCL 和
   CUDA 环境快照，以便三卡/四卡结果可追溯。

目前不应做完整训练、全量闭环评测、重提取 latent，或在未测 5B 显存前关闭
所有层的 reshard。

## 5. 当前完成边界

可以确认：拓扑参数解析、非法分组拒绝、默认 legacy 行为、指标字段、历史日志
审计、四种生产消融 checkpoint round-trip 和启动脚本静态语法均通过 CPU 验证。

仍待 GPU 确认：二维 DeviceMesh 的 5B FSDP2 构造、CPU-offload/HSDP 组合、
两步 forward/backward、真实显存、真实性能、checkpoint 保存和重载。因此当前
新增的 HSDP 路径应称为“实现完成、GPU gate 未通过”。
