# MemoryWAM 当前工作进度与代码变动说明

日期：2026-10-04
仓库：`/root/wyc/code/lingbot-va2`
当前 `HEAD`：`d986be8b9c3147dbeada91097b32cc3ea8db2c73`

**文档定位：本文是当前任务的主进度与交接入口。** 后续每次实现、验证、失败
证据或路线切换都应先更新本文的任务总表、验证结果和唯一下一步；`execution1.md`
至`execution5.md`保留分阶段命令和设计细节，不能单独代表最新状态。

相关文档：

- [机制与源码映射](./MemoryWAM_LingBotVA_implementation_plan.md)
- [legacy、toy 机制和阶段门禁](./execution1.md)
- [fullctx sidecar 与数据检查](./execution2.md)
- [生产接入和服务协议](./execution3.md)
- [无 GPU 训练准备和启动流程](./execution4.md)
- [无 GPU 收尾与多卡拓扑](./execution5.md)
- [CPU两步完整语义链路](./execution6.md)
- [完整5B模型CPU分级测试](./execution7.md)

## 1. 当前结论

当前代码已达到以下状态：

1. **MemoryWAM 生产路径已经接入且默认关闭。** 旧 checkpoint/config 不会创建随机 gist，也不会进入新的 Dataset、训练或服务分支。
2. **fullctx 数据无需重提取。** Rearrange train50 和 validation5 的原 latent、action、action mask、text embedding 在加入 sidecar 后逐 tensor、`rtol=0/atol=0` 相等。
3. **step-750 可以作为共享骨干权重初始化。** safetensors header 的839个 key与当前 legacy 模型逐项、逐 shape 完全匹配；开启 MemoryWAM 后唯一新增模型参数是 `gist_tokens[8,3072]`。
4. **CPU 可验证的机制和训练前门禁已经通过。** 当前共30项测试通过；新增生产消融模式、FSDP/HSDP 拓扑和历史多卡审计测试。2026-10-05 在加入本地可移植路径后重新运行 CPU training preflight 并为 `PASS`，代码 fingerprint 为 `bb5d2785fa3055fcba611f818a263f27b4bcfed524d636e9876827b41567c743`。
5. **尚未完成 5B/GPU 运行验证。** 当前不能宣称 memory-off 数值回归、5B FSDP 训练、完整 checkpoint reload、在线 VAE 或 RMBench 闭环已通过。
6. **GPU 恢复后先运行2-step smoke。** 当前完整 episode 路径是正确性 reference；是否能扩大训练预算取决于实测显存、step latency、gist 更新和 checkpoint reload。
7. **多卡速度已有可配置路径，但尚未通过 GPU gate。** 默认仍为历史的全 world FSDP + CPU offload；新增四卡 2×2 HSDP、关闭 CPU offload 和 reshard 开关。历史审计只找到16个三卡运行，没有四卡日志可用于定量回归。
8. **真实fullctx上的CPU两步语义链路已通过。** tiny生产模型完成两次视频/动作loss、反向、AdamW更新、gist更新、checkpoint与optimizer重载；该证据不包含5B、FSDP/HSDP、CUDA或RMBench环境执行。
9. **完整5B CPU测试脚本已就绪，plan阶段通过。** 当前资源满足load/forward/train-gist脚本门槛，但尚未实际加载10GB权重；推荐用户依次执行load、forward、train-gist，train-all默认不建议。

当前没有启动完整训练、全量评测或数据重建，也没有修改 `/root/wyc/data` 下的数据资产。

### 1.1 完整任务总表

状态定义：`完成（CPU）` 表示实现及可在无 GPU 环境验证的证据均已落盘；
`实现完成/GPU待验` 表示代码已接入但不能声称5B运行通过；`未开始` 表示仍没有
实现；`条件后置` 表示只有前置机制验证通过后才进入。

| 阶段 | 完整事项 | 当前状态 | 已有证据/产物 | 进入下一阶段的条件 |
| --- | --- | --- | --- | --- |
| 0 | 审计 AGENTS、入口、历史训练/推理、fullctx、checkpoint 和论文 | 完成（静态/历史证据） | `RMBench_history_fullctx_audit_20261003.md`、`rmbench_run_tables_20261003.md`、`MemoryWAM_LingBotVA_implementation_plan.md` | 已满足 |
| 1 | 冻结 legacy 代码、配置、数据、checkpoint、历史输出 | 完成（CPU） | `artifacts/memorywam/stage0_legacy_static_20261004/` | GPU 后补5B固定输入数值 reference |
| 2 | typed token、gist/anchor/recent/full-history、因果 mask、函数式 commit、幂等、denoise只读 | 完成（CPU） | toy/production tests；30项 suite的一部分 | 已满足 |
| 3 | streaming 逐帧 prefill 与完整前缀等价、无未来信息、RoPE使用全局 frame ID | 完成（CPU） | `test_streaming.py`、`test_mask_and_budget.py` | 5B backend仍需GPU数值确认 |
| 4 | gist 从视频/动作损失获得梯度，detach/TBPTT语义 | 完成（CPU小模型） | `test_gist_gradients.py`、`train_delayed_recall.py` | 5B需检查 `gist_grad_norm/update` |
| 5 | 显式 episode ID、global frame ID、`z[t] -> action[t+1]` sidecar；保持原tensor无损 | 完成（CPU） | train50/val5 sidecar、Dataset逐tensor零误差报告 | Rearrange首实验已满足 |
| 6 | Transformer/attention/Trainer生产接入，默认关闭和旧checkpoint兼容 | 实现完成/GPU待验 | tiny production forward、checkpoint round-trip、CPU preflight | 2-step 5B smoke |
| 7 | 服务接入、episode reset、observation/request幂等、去噪不写缓存 | 实现完成/GPU待验 | production service CPU test | 在线VAE与RMBench单episode闭环 |
| 8 | hybrid/recent-only/gist-only/full-history生产配置与checkpoint固化 | 完成（CPU） | 四模式 save/load + prefill/commit/denoise测试 | 各模式GPU smoke后才可训练比较 |
| 9 | FSDP全分片、2×2 HSDP、CPU offload/reshard开关和公平吞吐指标 | 实现完成/GPU待验 | topology纯函数测试、历史多卡审计、启动前校验 | 三卡/四卡2-step容量门禁和5-step计时 |
| 10 | 两步训练链路：loss、backward、optimizer、保存/重载 | CPU tiny链路完成；5B GPU链路未运行 | 真实fullctx CPU smoke为PASS；CPU header/meta结构通过 | GPU恢复后运行5B gate |
| 10a | 完整5B单进程CPU加载/forward/gist-backward | 分级脚本完成；仅plan已运行 | stage8 plan PASS，约485GiB RAM/1.94TiB磁盘可用 | 用户依次执行load、forward、train-gist |
| 11 | 完整断点续训：optimizer、scheduler、RNG、sampler/data cursor | 未实现 | 当前只能权重初始化/权重恢复 | 正式长训练前完成并验证 |
| 12 | Rearrange短程工程训练与线上单episode诊断 | 未运行（GPU阻塞） | 启动脚本和门禁已就绪 | 阶段10通过 |
| 13 | 固定数据/初始化/预算的五路比较：legacy、hybrid、recent-only、gist-only、full-history | 未运行（GPU阻塞） | policy入口已就绪 | 基线和机制训练语义稳定 |
| 14 | RMBench正式开发集/测试集闭环，固定seed与checkpoint选择规则 | 未运行 | 协议草案和服务实现已有 | 单episode闭环通过；不得用测试集调参 |
| 15 | MemoryWAM论文级独立动作专家、joint14、可能的数据重编码 | 条件后置 | 尚未实现，也未决定必须采用 | 共享骨干机制收益确认后单独立项 |

当前唯一主线是：**完成GPU容量/正确性门禁和基线吞吐选择，再做Rearrange短程
训练与闭环；此时之前不进入论文级动作专家或joint14改造。**

## 2. 生产代码变动

### 2.1 新增纯函数记忆状态

文件：`wan_va/modules/memorywam.py`

| 类/函数 | 变动 | 语义 |
| --- | --- | --- |
| `MemoryWAMPolicy` | 新增 hybrid、recent-only、gist-only、full-history policy | 主实验默认 M=8、anchor=2、recent=4；四种 policy 已在 toy/CPU 测试中区分 |
| `MemoryLayerKV` | 新增逐层不可变 K/V 容器 | K/V 为 `[B,S,H,D]`；同时保存 frame ID、token kind 和帧内 slot |
| `MemoryWAMState` | 新增 episode 级状态 | 保存每层 memory、`next_frame_id`、version 和 commit receipts |
| `MemoryWAMDelta` | 新增未提交的 staged delta | prefill 只产生 delta，调用方成功完成全部层后才 commit |
| `retain_layer` | 新增预算选择 | 所有历史 gist 保留；完整视觉只保留 anchor/recent 并集；重叠不重复 |
| `commit_memory` | 新增事务提交 | 检查 episode、版本、连续 frame；同 observation 的相同提交幂等，冲突提交报错；支持 detach/TBPTT |

运行时状态由调用方持有，不写入模型内部 legacy KV 池。动作/video 去噪只读取 state，不调用 commit。

### 2.2 Transformer 与 attention

文件：`wan_va/modules/model.py`

主要变动：

- `WanAttention.forward` 增加显式 `memory_kv` 和 `return_current_kv`：
  - persistent K/V 与当前 K/V 在 attention 前拼接。
  - MemoryWAM state 与 legacy cache 禁止混用。
  - 当前去噪 token 不写持久状态。
- `WanTransformerBlock.forward` 可返回本层 pre-attention current K/V，供 observation prefill 形成 staged delta。
- `WanTransformer3DModel.__init__` 增加默认关闭的 memory 参数。关闭时 `gist_tokens=None`，state dict 不增加 key。
- `enable_memorywam` 在 FSDP/optimizer 包装前解析并固化 policy：
  - hybrid/gist-only 创建共享 `[M,D]` 可训练 gist 参数。
  - recent-only/full-history 不创建 gist 参数；prefill使用临时零长度张量，避免
    无意义参数以及 `from_pretrained` 的 meta buffer问题。
  - mode、M/A/R 写入模型 config，旧 MemoryWAM config缺少 mode时默认为hybrid。
- `create_memorywam_state` 创建与模型层数、head 数和 dtype 匹配的 episode state。
- `memorywam_prefill_observation`：
  - 每次只接收一个 clean latent frame。
  - 追加共享 gist token。
  - clean/gist 在所有层读取既有历史并产生新的逐层 K/V delta。
  - gist RoPE 使用 `(global_frame_id,-2,-2)`；这是已记录的工程选择。
- `memorywam_denoise`：
  - video/action noisy query 读取同一个已提交视觉 memory。
  - 不改变 state version、frame ID 或 token 数。
- `forward_train_memorywam`：
  - 作为完整 episode 因果 reference，按 `z[t] -> {video[t+1], action_group[t+1]}` 运行。
  - 不把 target `z[t+1]` 的 clean token暴露给动作 query。
  - 第一帧视频输出和第一组 dummy action不计监督。
  - 当前 reference 限制 `batch_size=1`、episode global frame ID 为 `0..F-1`。

兼容性：`memorywam_policy is None` 时仍调用原 `forward_train`；普通推理仍使用原 `forward/update_cache` 路径。

### 2.3 正式 Dataset 与时间 sidecar

文件：`wan_va/dataset/lerobot_latent_dataset.py`

当 `memory_alignment_path` 未设置时，Dataset 仍只返回：

```text
latents / actions / actions_mask / text_emb
```

设置 sidecar 后额外返回：

```text
dataset_numeric_id
episode_index
segment_index
global_frame_ids
source_rgb_indices
decision_frame_ids
target_frame_ids
target_action_group_ids
video_target_mask
action_valid_slots_by_latent
reset_mask
```

sidecar 使用 `(episode_index, segment_start_frame, segment_end_frame)` 匹配正式样本；缺失或重复记录立即报错。多数据集 Dataset 使用 `dataset_numeric_id + episode_index` 形成跨数据集身份，避免 episode 串扰。

数据本体未改写；sidecar 和检查产物保存在仓库 `artifacts/memorywam/`。

### 2.4 训练配置

文件：`wan_va/configs/va_rmbench_train_cfg.py`

新增环境变量：

```text
RMBENCH_MEMORYWAM_ENABLED                 默认0
RMBENCH_MEMORY_ALIGNMENT_PATH
RMBENCH_VALIDATION_MEMORY_ALIGNMENT_PATH
RMBENCH_MEMORYWAM_GIST_TOKENS            默认8
RMBENCH_MEMORYWAM_ANCHOR_FRAMES          默认2
RMBENCH_MEMORYWAM_RECENT_FRAMES          默认4
RMBENCH_MEMORYWAM_MODE                   默认hybrid；另支持recent-only/gist-only/full-history
RMBENCH_MEMORYWAM_DETACH_INTERVAL        默认0
RMBENCH_FSDP_SHARD_SIZE                  默认0（全world）；四卡设2为2×2 HSDP
RMBENCH_FSDP_CPU_OFFLOAD                 默认1
RMBENCH_FSDP_RESHARD_AFTER_FORWARD       默认1
RMBENCH_FSDP_ROOT_RESHARD_AFTER_FORWARD  默认1
RMBENCH_SAVE_INTERVAL                    默认3000
RMBENCH_LOAD_WORKER                      沿用原默认值
```

没有修改原 RMBench inference config 的默认行为。服务是否启用 MemoryWAM 由加载的 transformer checkpoint config 决定。

### 2.5 Trainer

文件：`wan_va/train.py`

主要变动：

- 模型加载完成后、activation checkpoint/FSDP/optimizer 创建前注册 gist 参数。
- MemoryWAM 模式强制要求 alignment sidecar，并拒绝尚未对齐的 legacy bounded wrapper。
- validation 使用独立 validation alignment。
- `convert_input_format` 只对 tensor 调用 `.to(device)`，允许身份元数据进入 batch。
- `_prepare_input_dict` 把完整时间身份放入 `memory_meta`；MemoryWAM clean prefill 不再对条件 latent 随机加噪。
- `compute_loss` 用 `video_target_mask` 排除初始 observed frame；action loss 继续使用正式 `actions_mask` 排除 dummy、padding 和无效通道。
- 训练指标新增：
  - `gist_grad_norm`：全 rank、可能分片的 gist 梯度 L2 norm。
  - `gist_update_max`：optimizer step 前后 gist 的全 rank最大绝对变化。
  - `peak_memory_allocated_bytes` 和 `peak_memory_reserved_bytes`：所有 rank 的最大值。
  - `optimizer_step_seconds`：包含数据读取和完整累积周期，取所有 rank 的最大值。
  - `global_latent_frames`、`global_latent_frames_per_second`：按所有 rank 实际
    episode长度汇总，用于比较三卡/四卡而不是只看秒/step。
- checkpoint 保存异常现在重新抛出，使训练失败；不再吞掉异常并继续生成成功 summary。

仍保留的限制：当前 checkpoint 保存模型权重和模型 config，但 optimizer、scheduler、RNG 和 data cursor 的完整恢复尚未启用。所有旧 checkpoint 和当前新 checkpoint在该功能验证前都只能称为权重初始化/权重恢复，不能称完整断点续训。

### 2.6 生产服务

文件：`wan_va/wan_va_server.py`

MemoryWAM checkpoint 加载后自动进入 action-only 分支：

```text
reset episode
 -> encode current real observation
 -> clean prefill and atomic commit once
 -> all action denoise steps read the same immutable state
 -> execute action
 -> commit only the returned real observation
```

具体变动：

- reset 创建新 episode state，并清空 request response cache、Transformer legacy cache和 streaming VAE state。
- `_memorywam_commit_latents` 强制每条 observation message 对应一个 latent frame，并检查连续 global frame ID。
- observation receipt 保存原始图像 digest；相同 ID/内容幂等，相同 ID/不同内容拒绝。
- `_infer_memorywam` 每次生成一个 APF16 action group，所有 diffusion step均只读 state，并检查 denoise 前后 version 不变。
- 相同 `request_id` 返回缓存的相同动作结果。
- 兼容现有 RMBench client：第一次 response 前置 dummy action frame并返回 `action_start_index=1`；后续为0。
- 客户端旧 `state` 字段中的整段预测计划不再写入 MemoryWAM；只有真实回传 RGB 推进历史。

尚未验证：5B 服务加载、在线 VAE 数值、websocket 重试、异常中断和真实 RMBench episode 闭环。

### 2.7 FSDP/HSDP 拓扑与性能度量

文件：`wan_va/distributed/fsdp.py`、`wan_va/configs/va_rmbench_train_cfg.py`、
`wan_va/train.py`

- 保留默认全 world FSDP 行为，新增纯函数
  `resolve_fsdp_topology(world_size, shard_size)`。
- 四卡 `shard_size=2` 构造 `(replicate=2, shard=2)` HSDP；不整除的拓扑
  在加载5B前失败。
- CPU offload、子模块 reshard 和 root reshard 均可通过 RMBench 环境变量显式
  设置；默认值与历史训练相同。
- 新训练指标记录各 rank 最大的 `optimizer_step_seconds`、全 rank
  `global_latent_frames` 和 `global_latent_frames_per_second`，用于公平比较不同
  world size 和不同 episode 长度。
- 本地历史审计见
  `artifacts/memorywam/stage6_training_ready/multigpu_history.json`：16个可追溯
  运行均为 world size 3，4个 debug 运行含 collective timeout；未找到四卡历史
  timing，因此 HSDP 和关闭 offload 的效果必须等 GPU 恢复后实测。

## 3. 新增审计、测试和训练工具

目录：`script/memorywam_checks/`

| 工具 | 用途 |
| --- | --- |
| `memory_core.py`、`toy_model.py` | 与5B隔离的 typed memory、mask、streaming 和梯度 oracle |
| `fullctx_sidecar.py` | 从正式合同/latent header生成 episode 时间 sidecar |
| `aligned_fullctx_dataset.py` | 早期隔离 wrapper，保留作 reference |
| `trace_fullctx_dataset.py` | 正式 Dataset/DataLoader 全量 trace |
| `verify_production_dataset.py` | 正式 Dataset legacy/aligned 双实例逐 tensor 零误差比较 |
| `legacy_baseline.py`、`numeric_reference.py` | 静态资产冻结和未来5B数值 reference 比较 |
| `preflight_training.py` | 只读 safetensors header + meta model结构比较、代码 fingerprint、sidecar和全量 Dataset训练预检 |
| `run_checks.py` | 汇总运行所有 `test_*.py` |
| `run_training_cpu_preflight.sh` | 一键运行30项测试和完整 CPU training preflight |
| `train_delayed_recall.py` | 验证 gist-only 可学习长时信息，而 recent-only 在预算不足时不可解 |
| `audit_multigpu_history.py` | 只读历史 training log/metrics，汇总 world size、collective timeout和可用计时证据 |
| `cpu_two_step_smoke.py` | 用真实fullctx与tiny生产模型运行两步loss/backward/AdamW/保存重载 |
| `run_cpu_two_step_smoke.sh` | CPU两步链路的一键入口，拒绝覆盖已有证据目录 |
| `full_model_cpu_smoke.py` | 完整5B的plan/load/forward/train-gist/train-all分级测试和资源门禁 |
| `run_full_model_cpu_smoke.sh` | 完整5B CPU测试包装器，默认使用时间戳隔离输出 |
| `run_full_model_cpu_sequence.sh` | 按plan→load→forward→train-gist串行执行，失败即停并汇总日志 |

测试覆盖：

- typed token、anchor/recent去重和预算。
- Figure 3风格的因果可见性、跨 episode隔离和未来屏蔽。
- streaming 与完整前缀等价。
- observation commit 的幂等、冲突、stale version和跨 episode拒绝。
- denoise read-only和重复调用一致。
- 视频 loss、动作 loss分别向 gist 提供有限非零梯度。
- detach 截断跨帧梯度。
- production Dataset 无损扩展。
- production tiny Transformer完整 episode训练路径和activation checkpoint。
- production service单 observation commit。
- 四种生产policy的 checkpoint config/state round-trip及实际
  prefill/commit/denoise；其中 no-gist 模式不产生 gist checkpoint key。

## 4. GPU 启动工具

文件：`script/rmbench_train/train_memorywam_rearrange.sh`

该脚本固定首个任务、数据、初始化和 M/A/R；policy mode及FSDP拓扑由显式环境
变量决定。在真正加载 GPU 前检查：

- CPU preflight 为 `PASS`。
- preflight 记录的生产代码 fingerprint 与当前文件内容一致；代码变化会使旧 PASS 失效。
- MemoryWAM mode只能是hybrid、recent-only、gist-only、full-history之一。
- world size可被shard size整除；四卡3-way shard等不均匀拓扑直接拒绝。
- 输出目录不存在。
- `nvidia-smi` 可用且设备数不少于 `NGPU`。

2-step smoke 结束后自动检查：

- 恰好完成2个 optimizer step。
- video/action loss、总 grad norm、step时间和latent frames/s均有限且为正。
- hybrid/gist-only要求gist gradient每步非零，第二步 `gist_update_max>0`；
  recent-only/full-history要求指标和checkpoint均不包含gist参数。
- checkpoint config启用MemoryWAM，且mode和有效M/A/R与运行一致。
- gate记录world size和完整FSDP拓扑。
- 生成 `<output>/gpu_smoke_pass.json`。

超过2步的运行必须显式提供已通过的GPU gate，并设置
`MEMORYWAM_ALLOW_REFERENCE_TRAINING=1`。启动脚本会要求gate与后续运行的mode、
world size、shard size、offload和reshard逐项一致，防止用另一拓扑的两步结果
错误放行长运行。

## 5. 已生成的数据与证据产物

### 5.1 legacy 静态冻结

```text
artifacts/memorywam/stage0_legacy_static_20261004/
├── manifest.json
└── working_tree.patch
```

固定 initializer：

```text
train_out/rearrange_blocks_50demo_1000step_20260921_180241/
checkpoints/checkpoint_step_750
```

### 5.2 时间 sidecar

```text
artifacts/memorywam/stage4_alignment/rearrange_train/
artifacts/memorywam/stage4_alignment/rearrange_validation/
```

train：50 records、1231 decisions。
validation：5 records、123 decisions。

### 5.3 production Dataset 回归

```text
artifacts/memorywam/stage5_production_cpu/
├── rearrange_train_dataset_regression.json
└── rearrange_validation_dataset_regression.json
```

| split | episode | latent frame | valid action slot | 结果 |
| --- | ---: | ---: | ---: | --- |
| train | 50 | 1281 | 19696 | 原四类 tensor逐值相等 |
| validation | 5 | 128 | 1968 | 原四类 tensor逐值相等 |

### 5.4 训练 CPU preflight

```text
artifacts/memorywam/stage6_training_ready/cpu_preflight.json
```

关键结果：

```text
status = PASS
legacy keys = 839
missing/unexpected/shape mismatch = 0/0/0
new parameter = gist_tokens
gist shape = [8,3072]
code fingerprint = bb5d2785fa3055fcba611f818a263f27b4bcfed524d636e9876827b41567c743
```

### 5.5 历史多卡审计

```text
artifacts/memorywam/stage6_training_ready/multigpu_history.json
```

结果：16个具有world-size或timeout证据的保留运行全部是world size 3；其中4个
debug运行出现collective/all-gather timeout文本。没有保留的四卡训练日志，因而
不能从历史产物定量证明四卡慢于三卡。历史 `elapsed_seconds` 差分可能包含验证和
保存，只作粗略证据；新运行使用显式step时间和latent frames/s。

### 5.6 CPU两步完整语义链路

```text
artifacts/memorywam/stage7_cpu_2step_pass/
├── checkpoint_step_2/
│   ├── transformer/config.json
│   ├── transformer/diffusion_pytorch_model.safetensors
│   └── training_state.pt
└── cpu_2step_pass.json
```

输入来自正式Rearrange train50的episode 0与正式alignment。原始
`latent[48,26,24,20]`、`text[512,4096]`、`action/mask[30,26,16,1]`；
smoke保留真实通道、动作和APF16，只取前三帧、4×4 latent空间、前16个文本token。

两步结果：

| step | video loss | action loss | gist grad norm | gist update max |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 3.4513404 | 0.8879023 | 0.0215649 | 0.001953125 |
| 2 | 2.6816492 | 0.9620216 | 0.0182413 | 0.001953125 |

safetensors序列化值逐tensor精确恢复，optimizer state成功恢复。保存前与重载后
输出最大绝对差0.015625；原模型和重载模型各自重复forward差均为0。差异来自
保存路径将state统一量化到BF16，而15个Norm/time-embedding参数重载为FP32，
不是missing key。报告预设绝对容差0.02。脚本SHA256为
`fdf060b5420d9d9234f6d23db66303949c04040f56eaa68440a4143c7dab685d`。

### 5.7 完整5B CPU测试计划

```text
artifacts/memorywam/stage8_full_cpu_plan_v2_20261004/
├── events.jsonl
├── plan.json
└── full_cpu_pass.json
```

plan阶段已通过：checkpoint为10,177,831,668 bytes、839个BF16 key；开始时约
484.99GiB可用RAM、1.94TiB可用磁盘。该阶段只读取safetensors header，没有加载
权重tensor。第一次plan曾因不必要地计算整个10GB文件SHA256而被主动中止；脚本
已改为复用config hash、文件大小、mtime及逐key header，避免规划阶段消耗。

## 6. 本轮实际验证

### 6.1 CPU 单元与集成检查

命令：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
/root/miniconda3/envs/lingbot/bin/python -B \
  -m script.memorywam_checks.run_checks
```

结果：

```text
Ran 30 tests
OK
```

### 6.2 完整 CPU training preflight

命令：

```bash
script/memorywam_checks/run_training_cpu_preflight.sh
```

分步命令和结果见 [execution4.md](./execution4.md)。本轮已单独实际运行 `preflight_training.py` 并得到 `PASS`；没有调用 GPU、没有加载5B tensor内容。

### 6.3 静态检查

已通过：

```text
python -m py_compile ...
bash -n run_training_cpu_preflight.sh train_memorywam_rearrange.sh
git diff --check
preflight code fingerprint match
```

### 6.4 CPU两步训练smoke

命令：

```bash
bash script/memorywam_checks/run_cpu_two_step_smoke.sh
```

结果：`PASS`。标准结果见
`artifacts/memorywam/stage7_cpu_2step_pass/cpu_2step_pass.json`；详细范围、裁剪和
两次诊断过程见[execution6.md](./execution6.md)。该命令没有加载5B权重、没有
初始化GPU或分布式进程组。

### 6.5 完整5B CPU plan

命令：

```bash
bash script/memorywam_checks/run_full_model_cpu_smoke.sh plan \
  artifacts/memorywam/stage8_full_cpu_plan_v2_20261004
```

结果：`PASS`。只验证CLI、checkpoint header、内存/磁盘门禁和报告写出；完整模型
load、forward和backward尚未执行。用户运行说明见[execution7.md](./execution7.md)。

## 7. GPU 恢复后的执行顺序

### 7.1 固定环境证据

首先记录设备UUID、驱动、PyTorch/CUDA/NCCL和拓扑：

```bash
nvidia-smi -L
nvidia-smi topo -m
/root/miniconda3/envs/lingbot/bin/python - <<'PY'
import json, torch
print(json.dumps({
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "nccl": torch.cuda.nccl.version(),
}, default=str, indent=2))
PY
```

### 7.2 各拓扑两步容量/正确性 gate

先跑历史三卡同代码基准：

```bash
cd /root/wyc/code/lingbot-va2
NGPU=3 RMBENCH_FSDP_SHARD_SIZE=0 RMBENCH_FSDP_CPU_OFFLOAD=1 \
MASTER_PORT=29681 \
  script/rmbench_train/train_memorywam_rearrange.sh 2 \
  train_out/memorywam_smoke_3gpu_legacy
```

先复现四卡全分片、CPU offload的用户观察，以便单独识别world size影响：

```bash
NGPU=4 RMBENCH_FSDP_SHARD_SIZE=0 RMBENCH_FSDP_CPU_OFFLOAD=1 \
MASTER_PORT=29682 \
  script/rmbench_train/train_memorywam_rearrange.sh 2 \
  train_out/memorywam_smoke_4gpu_legacy
```

再测试四卡全分片、关闭CPU offload；显存不足时记录OOM而不缩短历史：

```bash
NGPU=4 RMBENCH_FSDP_SHARD_SIZE=0 RMBENCH_FSDP_CPU_OFFLOAD=0 \
MASTER_PORT=29683 \
  script/rmbench_train/train_memorywam_rearrange.sh 2 \
  train_out/memorywam_smoke_4gpu_no_offload
```

再测试四卡2×2 HSDP：

```bash
NGPU=4 RMBENCH_FSDP_SHARD_SIZE=2 RMBENCH_FSDP_CPU_OFFLOAD=1 \
MASTER_PORT=29684 \
  script/rmbench_train/train_memorywam_rearrange.sh 2 \
  train_out/memorywam_smoke_4gpu_hsdp_2x2
```

每个gate回答：5B是否可加载、FSDP/HSDP是否接受新参数、完整反向是否运行、
gist是否有梯度并更新、显存是否满足、checkpoint是否写出。二维DeviceMesh当前只
通过CPU拓扑检查，第一次5B运行必须视为GPU实现验证。

### 7.3 通过候选的5-step短程计时

每个候选只能使用自己生成的 `gpu_smoke_pass.json`：

```bash
MEMORYWAM_GPU_GATE=/absolute/path/to/candidate/gpu_smoke_pass.json \
MEMORYWAM_ALLOW_REFERENCE_TRAINING=1 \
NGPU=<same-world-size> \
RMBENCH_FSDP_SHARD_SIZE=<same-shard-size> \
RMBENCH_FSDP_CPU_OFFLOAD=<same-offload> \
RMBENCH_FSDP_RESHARD_AFTER_FORWARD=<same-reshard> \
RMBENCH_FSDP_ROOT_RESHARD_AFTER_FORWARD=<same-root-reshard> \
MASTER_PORT=<new-port> \
  script/rmbench_train/train_memorywam_rearrange.sh 5 \
  train_out/<candidate>_5step_timing
```

去掉第一个warm-up step，比较后续4步的
`global_latent_frames_per_second`、`optimizer_step_seconds`及峰值显存。选择
吞吐最高且至少保留15%峰值显存余量的配置。四卡都不优于三卡时继续三卡训练，
第四卡用于评测或数据服务；不构造不均匀3+1 FSDP。

### 7.4 50-step工程验证

仅对胜出的同一拓扑执行：

```bash
export MEMORYWAM_GPU_GATE=/absolute/path/to/gpu_smoke_pass.json
export MEMORYWAM_ALLOW_REFERENCE_TRAINING=1
export RMBENCH_SAVE_INTERVAL=50

NGPU=<selected> \
RMBENCH_FSDP_SHARD_SIZE=<selected-shard-size> \
RMBENCH_FSDP_CPU_OFFLOAD=<selected-offload> \
RMBENCH_FSDP_RESHARD_AFTER_FORWARD=<selected-reshard> \
RMBENCH_FSDP_ROOT_RESHARD_AFTER_FORWARD=<selected-root-reshard> \
MASTER_PORT=<new-port> \
  script/rmbench_train/train_memorywam_rearrange.sh 50
```

50-step仍是工程验证，不是正式论文结果。若2-step显示完整 episode reference吞吐不可接受，先实现 episode-ordered chunk/TBPTT或并行mask，不通过缩短历史、关闭gist梯度或使用未来真值来绕开问题。

### 7.5 服务与闭环顺序

50-step工程语义通过后，依次进行：保存checkpoint完整重载forward；同一真实RGB
流的离线/在线VAE对比；单episode websocket往返；reset、重复请求、异常中断和
部分动作执行；最后才运行小规模开发集rollout。专家动作回放只验证数据/控制，
不得计作模型成功率。

## 8. 尚未完成及阻塞项

| 项目 | 当前状态 | 下一证据 |
| --- | --- | --- |
| 5B memory-off数值回归 | 延后，历史日志不能替代 | 固定输入/噪声/backend的reference和candidate比较 |
| CPU tiny MemoryWAM两步完整语义链路 | 已通过真实fullctx输入 | 保留PASS报告，不能替代5B证据 |
| 完整5B CPU load/forward/train-gist | 脚本及plan已通过，重阶段未执行 | 用户按load→forward→train-gist逐级运行 |
| 5B MemoryWAM forward/backward | 未运行 | 2-step GPU smoke |
| FSDP gist分片与optimizer更新 | 仅代码/小模型验证 | `gist_grad_norm`、`gist_update_max`和参数检查 |
| 5B checkpoint实际reload | header和tiny round-trip已过 | 保存后完整reload forward |
| 三卡/四卡正式训练吞吐 | 拓扑和指标已实现，未测 | 各候选2-step gate + 5-step frames/s和峰值显存 |
| 2×2 HSDP DeviceMesh | CPU拓扑构造逻辑已测 | 四卡5B首次构造、前后向、保存/重载 |
| 完整断点续训 | 未实现 | optimizer/scheduler/RNG/data cursor保存与恢复测试 |
| 在线VAE一致性 | 未测 | 同一真实RGB流离线/在线latent数值比较 |
| RMBench闭环 | 未测 | 显式episode/request/observation ID单episode往返 |
| policy消融GPU验证 | 四模式生产配置、checkpoint和CPU forward已完成 | 同初始化/预算的四模式GPU smoke、训练和闭环 |
| legacy/full-history比较 | 入口已具备，尚无同预算结果 | 固定数据、初始化、动作接口、目标暴露量与checkpoint规则 |
| Observe/Blocks Ranking sidecar | Rearrange已完成，其他任务未生成 | 仅在扩展任务前生成并做Dataset零误差回归 |
| 论文级MoT/joint14/mosaic | 未开始 | 仅在共享骨干机制验证后另立数据/模型版本 |

## 9. 工作树说明

当前 `git status` 中 MemoryWAM 相关变动包括：

```text
wan_va/configs/va_rmbench_train_cfg.py
wan_va/dataset/lerobot_latent_dataset.py
wan_va/distributed/fsdp.py
wan_va/modules/model.py
wan_va/modules/memorywam.py                  新增
wan_va/train.py
wan_va/wan_va_server.py
script/memorywam_checks/                     新增/扩展
script/rmbench_train/train_memorywam_rearrange.sh  新增
docs/execution1.md ... execution7.md
docs/MemoryWAM_current_progress_20261004.md   新增
```

`.gitignore` 的既有本地修改和未跟踪 `AGENTS.md` 不属于本轮 MemoryWAM 生产实现，本轮没有覆盖或删除它们。`docs/` 和 `artifacts/` 受当前 `.gitignore` 影响，文件存在于工作区但通常不会出现在普通 `git status` 输出中。

## 10. 当前下一步与禁止跳过项

GPU继续不可用时，下一条可执行路径是完整5B的分级CPU检查：先运行`load`，通过后
运行`forward`，再按需运行`train-gist --steps 2`。每一级都使用新输出目录，并在
首个失败处停止。具体命令、资源门槛和证据边界见
[execution7.md](./execution7.md)；也可用`run_full_model_cpu_sequence.sh`一条命令按
该顺序串行执行。`train-all`会为完整骨干构建梯度和optimizer状态，
CPU代价高且不能验证分布式路径，不作为GPU恢复前的必需门禁。

GPU恢复后的唯一下一步是第7节的三卡/四卡2-step矩阵。必须先得到至少一个
`gpu_smoke_pass.json`，然后才允许5-step计时；拓扑胜出后才允许50-step工程
验证。以下证据不可互相替代：

- 历史训练日志不能替代当前5B MemoryWAM前后向。
- tiny模型通过不能替代FSDP/HSDP实际分片和checkpoint reload。
- teacher-forced loss不能替代完整动作去噪。
- 完整动作解码不能替代环境正确执行。
- 单episode成功不能替代固定seed、独立场景的正式闭环成功率。
- 机制收益未确认前，不开始独立动作专家、joint14或全量latent重提取。

完整5B的`load/forward/train-gist`完成后，仍可在无GPU条件下补齐完整断点续训合同，
以及未来扩展到Observe/Blocks Ranking时所需的sidecar；这些工作都不应改动当前
已冻结的Rearrange首实验数据和初始化。
