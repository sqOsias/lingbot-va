# MemoryWAM 第一阶段执行文档：隔离验证、legacy 门禁与生产接入顺序

日期：2026-10-04
状态：小模型机制验证已实现并通过；生产模型尚未接入；5B 数值 legacy reference 尚未生成。

本文取代此前的 `execution1.md`。它是可执行的阶段门禁，不是另一份架构概念设计。论文机制依据来自本地 [MemoryWAM PDF](<./Yang 等 - 2026 - MemoryWAM Efficient World Action Modeling with Persistent Memory.pdf>)；LingBot-VA 与 fullctx 的现状依据见 [机制移植审计](./MemoryWAM_LingBotVA_implementation_plan.md)和 [RMBench/fullctx 审计](./RMBench_history_fullctx_audit_20261003.md)。历史报告只作为导航；本文标为“已通过”的项目均在本轮重新运行。

## 1. 对原步骤的判断与调整

原路线的总体方向正确：先固定 legacy，再用小模型验证机制，最后才接 5B、服务和 RMBench；独立动作专家与 14D joint 应继续放在机制验证之后。以下顺序需要修正。

| 原步骤中的问题 | 为什么会影响结论 | 调整 |
| --- | --- | --- |
| “固定推理输出”与“不加载 5B”被放在同一个立即完成的阶段 | 文件、配置和历史输出可以静态冻结，但当前工作树的数值输出必须实际加载目标 checkpoint 才能生成。历史 rollout 不能证明当前代码能产生同样输出 | 把 legacy 拆为 **0A 静态冻结**和 **0B 当前数值 reference**。先完成 0A 和全部小模型验证；任何生产 `model.py` 修改前必须完成 0B |
| typed token/mask 之后才考虑服务幂等性 | 如果缓存 API 一开始就允许 denoise 写入，后面很难无风险补事务语义 | 在小模型第一阶段就定义 `prefill -> MemoryDelta -> commit`，同一 `observation_id` 的相同重试幂等，不同内容冲突拒绝；denoise 只能读 |
| streaming 等价测试可能由两条相同逐帧调用链完成 | 这只能证明函数重复调用稳定，不能验证一次性完整前缀 attention 与缓存推理等价 | full-prefix reference 必须在每层用一次 batched attention 计算整个序列；streaming 每次只处理一个新帧，两者只共享参数和 mask 规则 |
| “gist 有梯度”判据太弱 | 任意旁路或错误连接也可能产生非零梯度；单一总 loss 无法证明视频和动作监督都能训练 gist | 分别用未来视频 loss 和动作 loss 验证有限且非零的 gist 梯度，再用必须依赖早期信息的 delayed-recall 小任务验证 gist-only 可学习、recent-only 在预算不足时不能解题 |
| 数据字段改造放在 gist 参数之后但未设生产门禁 | 小模型不依赖 RMBench 数据，可以先验证；生产训练若没有 episode/global frame/target 三元关系，则 mask、RoPE 和监督都无法审计 | 小模型阶段之后立即冻结显式数据合同；完成真实 fullctx adapter 检查后才允许接生产训练 |
| 接服务和训练的先后关系没有拆开 | 服务缓存生命周期与训练梯度生命周期不同，同时改会把错误来源混在一起 | 先完成 production default-off 接口和离线前向，再接 episode 顺序训练，随后接服务事务和闭环 |

调整后的唯一执行顺序是：

```text
0A 静态 legacy 冻结（已完成）
 -> 1 typed token / mask / 事务（已完成）
 -> 2 streaming 与 batched full-prefix 等价（已完成）
 -> 3 gist 梯度与可学习性（已完成）
 -> 4 RMBench 显式时间合同与 dataset adapter
 -> 0B 当前 5B 数值 legacy reference
 -> 5 生产模型 default-off 接入
 -> 6 episode 顺序训练与 TBPTT
 -> 7 服务/RMBench 事务接入
 -> 8 固定条件消融与 full-history 对照
 -> 9 决定是否做独立动作专家、14D joint 与输入重编码
```

0B 排在小模型之后是为了先不占用 5B 资源；它仍然是**修改生产模型前的硬门禁**。如果准备提前编辑 `wan_va/modules/model.py`，必须先把 0B 提前完成。

## 2. 论文机制边界与本阶段工程合同

### 2.1 论文明确规定

- §3.2 式(2)(3)：world/action 建模使用历史视觉上下文；不能把普通自回归 LLM cache 语义直接套入。
- §3.3 式(4)至式(7)、图3：历史由 gist、anchor、recent 的并集构成；同一帧的 gist 与视觉 token 是不同表示。
- §4.1、附录 A.1：默认 `M=8` gist/frame、`Ninit=2` anchor frames、`Nrecent=4` recent frames；每帧 gist KV 保留，anchor/recent 保存完整视觉 KV；gist 通过视频和动作 flow loss 学习。
- 图3：当前帧视觉与其 gist 可互见；未来帧不可见；历史完整视觉帧是否可见由 anchor/recent 规则决定。

### 2.2 本阶段工程假设

以下不是作者代码事实，后续实验必须保留为显式配置：

- 一个共享的 `[M,D]` 可学习 gist 参数逐帧复制，而不是每个绝对时刻一套参数。
- token 元数据使用 `(episode_id, global_frame_id, kind, slot_in_frame, target_group)`；动作/noisy-video token 只在本次去噪组内互见且永不提交持久状态。
- anchor 与 recent 重叠时完整视觉 token 只保存一份；gist 与完整视觉 token 允许同时存在。
- 已旋转 K 保留其原始 global frame 坐标；滑窗淘汰不重编号。小模型只实现 temporal RoPE，生产模型仍须沿用现有 3D RoPE。
- episode 间状态必须重建；训练跨 chunk 默认在边界 `detach`，是否保留若干 chunk 的梯度属于后续 TBPTT 配置。

### 2.3 五种比较必须使用不同且明确的存储策略

| 名称 | 持久视觉状态 | 用途 |
| --- | --- | --- |
| legacy | 当前 LingBot-VA 原行为 | 回归门禁与历史结果延续 |
| hybrid | 所有帧 gist + 前 `Ninit` 帧完整 token + 最近 `Nrecent` 帧完整 token | MemoryWAM 机制主实验 |
| recent-only | 仅最近 `Nrecent` 帧完整 token | 验证短期窗口贡献 |
| gist-only | 仅所有帧 gist | 验证压缩长时信息贡献 |
| full-history | 所有历史完整视觉 token，不含 gist | 质量上界/计算增长对照 |

这五种策略不能通过“同一个缓存先写满再在输出端忽略”来模拟；写入、保留、淘汰和 attention key 集合都必须符合各自定义。

## 3. 本轮代码处置与当前证据

### 3.1 已撤回的生产修改

此前未经小模型门禁就改入 5B 路径的内容已删除或恢复：

- `wan_va/modules/model.py` 已恢复到本轮开始前的 tracked 内容。
- `wan_va/configs/__init__.py` 已恢复，不再注册尚未验证的 MemoryWAM 配置。
- 已删除提前加入生产目录的 `wan_va/modules/memory.py`、`wan_va/modules/memory_mask.py`、`wan_va/dataset/episode_memory_dataset.py`、`wan_va/configs/va_rmbench_memory_cfg.py` 和 `wan_va/configs/va_rmbench_memory_train_cfg.py`。
- 保留用户已有的 `.gitignore` 修改和未跟踪 `AGENTS.md`；本轮没有覆盖它们。

因此当前 MemoryWAM 代码只存在于 `script/memorywam_checks/`，不会被正式训练或服务入口导入。

### 3.2 新增的隔离验证代码

| 文件 | 责任 | 不承担的责任 |
| --- | --- | --- |
| `memory_core.py` | typed token、不可变 `MemoryState`、每层 KV、预算、可见性、纯函数 commit、幂等 receipt | 不接 5B，不假定现有 `WanAttention` 已满足事务语义 |
| `mask_oracle.py` | 小规模 dense 布尔 mask reference | 不是生产 FlexAttention 优化实现 |
| `toy_model.py` | 小型共享骨干、global-frame temporal RoPE、逐帧 prefill、batched full-prefix、只读动作 denoise | 不声称复现论文 6B 双专家或 3D 视频 patch 几何 |
| `data_contract.py` | 显式 episode/frame/action 对齐 reference | 尚未替换正式 Dataset/Sampler |
| `legacy_baseline.py` | 不加载模型地冻结 git/config/checkpoint/data/输出资产 | 历史输出不能自动升级为当前数值 reference |
| `numeric_reference.py` | 对一次固定请求导出的 tensor container 记录摘要并逐 tensor 比较 | 不自行启动 5B 服务 |
| `train_delayed_recall.py` | CPU delayed-recall 可学习性检查 | 不是 RMBench 训练或成功率 |
| `test_*.py`、`run_checks.py` | 标准库 unittest 入口 | 不要求安装 pytest，不运行分布式或 GPU 作业 |

### 3.3 本轮运行结果

执行命令：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
  /root/miniconda3/envs/lingbot/bin/python \
  -m script.memorywam_checks.run_checks
```

本轮最终结果为 16 项测试全部通过，耗时约 0.41 秒。覆盖范围：

- 图3风格三帧可见性、未来信息阻断、跨 episode 阻断、无效 token 阻断。
- 1000 帧下 hybrid 预算和 anchor/recent 去重。
- hybrid、recent-only、gist-only、full-history 的实际保留集合不同。
- 每层一次 batched attention 的 full-prefix 与逐帧 streaming 在 FP32 下 `atol=1e-6, rtol=1e-5` 一致。
- 修改未来输入不改变过去输出；未来 loss 不向过去不可见输入传梯度。
- RoPE 使用 global frame ID，不使用缓存槽位编号。
- prefill 不就地改状态；相同 observation 重试幂等；相同 ID 不同内容、过期版本和跨 episode 提交被拒绝。
- 50 次动作 denoise 不改变 state digest；同一 request/step/input 得到相同输出。
- gist 分别从 future-video MSE 和 action MSE 获得有限、非零梯度；detach 能截断跨 chunk 梯度。
- APF16 编号序列和三个本地 fullctx episode-0 元数据合同通过。

可学习性实验：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
  /root/miniconda3/envs/lingbot/bin/python \
  -m script.memorywam_checks.train_delayed_recall \
  --steps 200 \
  --output docs/memorywam_toy_recall_20261004.json
```

固定 200 步结果：gist-only loss 从 `1.41324` 降到 `3.79e-09`，两类预测约为 `[-0.99996, 0.99993]`；recent-only 在看不到首帧时终值为 `1.0` 且两类输出相同。这个结果只证明小模型中的长时信息通路可训练，不代表 RMBench 收益。

本轮没有启动正式训练、5B forward、服务、仿真或环境安装。PyTorch 导入可能打印 CUDA 初始化警告，但命令通过 `CUDA_VISIBLE_DEVICES=''` 在 CPU 执行。

## 4. 分阶段操作、产物和门禁

### 阶段 0A：静态 legacy 冻结（已完成）

选择 `rearrange_blocks` 是因为该运行具有可追踪的 50-demo fullctx 数据、step-750 checkpoint 和历史 5-episode 输出；它适合第一条服务回归链。冻结命令：

```bash
/root/miniconda3/envs/lingbot/bin/python \
  -m script.memorywam_checks.legacy_baseline \
  --checkpoint train_out/rearrange_blocks_50demo_1000step_20260921_180241/checkpoints/checkpoint_step_750 \
  --dataset /root/wyc/data/rmbench/lingbot_va2/rearrange_blocks_50demo_fullctx_apf16_cf8 \
  --config rmbench_train \
  --config rmbench \
  --inference-output artifacts/rearrange_blocks_50demo_test_5ep_faithful_20260922_100217 \
  --inference-provenance historical \
  --hash-large-files \
  --output artifacts/memorywam/stage0_legacy_static_20261004
```

产物是 `artifacts/memorywam/stage0_legacy_static_20261004/manifest.json` 和 `working_tree.patch`。checkpoint 和 dataset 的每个文件均有完整 SHA256。该 manifest 的 `numeric_regression_ready=false` 是预期行为：历史 rollout 只能证明资产来源，不能作为当前工作树逐值 reference。

门禁：静态 manifest 必须能重建相同文件集合、配置值和 git diff；不能把不同时间生成的历史输出标成 `current_snapshot`。

### 阶段 1：typed token、预算、mask 和事务（已完成）

实现对象：

```python
MemoryState(
    episode_id: str,
    layers: tuple[LayerKV, ...],
    next_frame_id: int,
    version: int,
    receipts: tuple[CommitReceipt, ...],
)

result = model.prefill_observation(frame, state,
                                   frame_id=global_frame_id,
                                   observation_id=observation_id)
next_state = commit(state, result.delta, policy, detach=True)
action = model.denoise_action(noisy_action, next_state,
                              request_id=request_id,
                              denoise_step=tau)
```

规则：

1. `prefill_observation` 只返回 `MemoryDelta`，不修改原状态。
2. 只有环境确认的新真实观测可以 `commit`；预测视频、noisy action 和任一中间去噪步都不能提交。
3. 相同 `observation_id + frame_id + delta digest` 的重试返回原 state；同 ID 不同内容直接报错。
4. `episode_id`、连续 `frame_id`、`base_version` 任一不匹配均拒绝提交。
5. anchor/recent 的完整视觉 token 重叠时只保留一个；action token 不进入持久 state。

生产门禁：dense oracle 的每一行必须能由带编号的小序列人工解释；优化 mask 必须与 dense oracle 逐元素一致后才能替换。

### 阶段 2：逐帧 streaming 与完整前缀等价（已完成）

reference 定义：对 `[f0,g0,f1,g1,...]` 在每层只调用一次 attention，用逐 query 的 hybrid 因果 mask 得到全部层 KV 和输出。streaming 定义：每个环境观测时刻只计算新 `[ft,gt]`，读取以前提交的逐层 KV。两者共享相同权重、RoPE 和可见性定义，但不共享执行拓扑。

门禁：

- FP32、dropout 关闭时所有帧输出和最终 retained KV 在声明容差内一致。
- future input 扰动不改变历史输出；未来 loss 不向不可见的历史/未来边传播错误梯度。
- cache 删除只改变 key 集合，不改变保留 K 的 global-frame RoPE 相位。
- 比较的是每层 K/V、输出和元数据，不能只比较最后一层均值。

### 阶段 3：gist 梯度与可学习性（已完成）

门禁分三层：

1. future-video loss 对共享 gist 参数产生有限、非零梯度。
2. action loss 独立产生有限、非零梯度。
3. 在 recent window 无法访问首帧的 delayed-recall 中，gist-only 学会任务，而 recent-only 保持信息论上的不可解状态。

不新增 gist 重建、对比或蒸馏 loss。论文只给出视频和动作 flow loss；辅助目标只能作为后续探索变量，不能混入第一条复现基线。

### 阶段 4：显式 RMBench 时间合同与 Dataset adapter（下一步）

先保留现有 RGB、latent、文本 embedding、动作 parquet 和统计文件，只增加版本化 index/manifest；没有证据要求全量重提取。每条训练决策必须带：

```text
episode_id
decision_global_frame_id       # 已观测 z[t]
target_global_frame_id         # 监督 target z[t+1]
target_action_group_id         # action group[t+1]
source_rgb_index / target_rgb_index
parquet_row_start / parquet_row_end
physical_target_start / physical_target_end
valid_video_mask / valid_action_mask
```

当前 reference 将 APF16 明确展开为：原始 RGB 步长 4 × VAE 时间压缩 4；`z[t]` 的决策监督下一组最多 16 个物理动作以及 `z[t+1]`。`action group ID`、parquet 行号和实际控制时刻分别保存，禁止再用一个 `t+1` 同时指代三者。

操作顺序：

1. 用 `continuous_contract.json`、`meta/episodes.jsonl` 和 latent 文件头生成 episode-level alignment sidecar。
2. 全量检查 ID 唯一性、长度、尾部截断、split、统计来源；数值内容只做定向抽查。
3. Dataset 返回显式字段；sampler 以 episode 为单位组织连续 chunk，batch slot 变化时携带 reset mask。
4. 用 numbered synthetic episode 和真实 episode-0 对照 Trainer 收到的张量及 loss mask。

门禁：同一轨迹不能跨 train/validation；尾部不足动作必须通过 mask 排除；任何 batch slot 不得继承另一个 episode 的 state。现有底层 fullctx latent 可继续使用，除非 adapter 检查发现 VAE/相机/采样 manifest 与目标配置不一致。

### 阶段 0B：当前 5B 数值 legacy reference（生产修改前必须完成）

固定单条已知输入、seed、dtype、attention backend、denoise schedule、prompt、checkpoint 和设备。现有 inference 路径增加一个只读诊断出口，将至少以下内容保存为只含 tensor/list/dict 的 `.pt`：

```text
preprocessed observation latent
video denoise final latent or decoded frame fingerprint input
action denoise final model-space action
denormalized EEF action
每次环境提交后的 cache valid count / frame_st_id
```

保存后记录摘要：

```bash
/root/miniconda3/envs/lingbot/bin/python \
  -m script.memorywam_checks.numeric_reference capture \
  --input artifacts/memorywam/legacy_current/reference.pt \
  --output artifacts/memorywam/legacy_current/reference.json
```

新代码关闭 memory 后比较：

```bash
/root/miniconda3/envs/lingbot/bin/python \
  -m script.memorywam_checks.numeric_reference compare \
  --reference artifacts/memorywam/legacy_current/reference.pt \
  --candidate artifacts/memorywam/legacy_current/candidate.pt \
  --atol 1e-5 --rtol 1e-4 \
  --output artifacts/memorywam/legacy_current/compare.json
```

随机去噪必须保存并复用同一初始噪声，不能只设置 seed 后假设分布式调用顺序相同。FlashAttention 与 torch attention 可能需要分别保存 reference；若 backend 改变，应使用声明的数值容差而不是要求 bitwise 相等。

门禁：`numeric_regression_ready=true` 的 manifest 必须由当前冻结代码现场生成；历史 5/5 rollout 不替代此门禁。

### 阶段 5：生产模型 default-off 接入

只在阶段 0B 完成后修改生产路径。建议最小文件范围：

| 位置 | 最小改动 |
| --- | --- |
| `wan_va/modules/model.py` | `forward` 接受显式只读 memory state；attention 返回候选 current-frame KV；memory disabled 时完全走原分支 |
| 建议新增 `wan_va/modules/memory_state.py` | 从已验证的纯函数 state/delta/commit 迁入生产形状；不持有模型和服务对象 |
| 建议新增 `wan_va/modules/memory_mask.py` | dense oracle 对应的 FlexAttention mask 与小序列一致性检查 |
| `wan_va/configs/va_rmbench_train_cfg.py`、`va_rmbench_cfg.py` | 增加默认关闭的 memory block；不改现有默认值 |

生产接口必须分开：

```python
delta, observation_output = model.prefill_observation(
    clean_latent, text_condition, state=state,
    episode_id=episode_id, global_frame_id=frame_id)

prediction = model.denoise_action(
    noisy_action, text_condition, state=state,
    request_id=request_id, denoise_step=tau)

state = memory.commit(state, delta, detach=not retain_graph)
```

严禁继续依赖现有 `update_cache=0` 的“临时写后回滚”来表达只读，因为旧实现缓存满时可能为临时 token 淘汰历史，而 restore 不恢复被淘汰项。生产 memory 应当在容量检查和写入前区分 read 与 commit。

门禁：memory off 的模型 keys、输出、loss 和服务响应通过 0B 回归；旧 checkpoint strict load 不出现未声明 missing/unexpected keys。开启 memory 时只有 gist 参数是新权重；anchor/recent KV 本身不新增参数。

### 阶段 6：episode 顺序训练与 TBPTT

先做一个 episode 的完整前缀 reference，再实现顺序 chunk。每个 batch slot 保存自己的 state 和 episode ID：

```python
for batch in episode_ordered_loader:
    states = reset_slots(states, batch.episode_id, batch.reset_mask)
    loss = 0
    for chunk in batch.chunks:
        outputs, deltas = model.forward_chunk(chunk, states)
        loss += masked_video_loss(outputs, chunk) + masked_action_loss(outputs, chunk)
        states = commit_observed_frames(states, deltas)
    (loss / valid_target_count).backward()
    optimizer.step()
    states = detach_all(states)  # 默认一个 optimizer step 为 TBPTT 边界
```

完整 episode、顺序 chunk 和记忆关闭的独立窗口是三种不同训练语义，日志中必须分别命名。history 是否计算 loss 是配置项：作为 burn-in 时 mask 掉；作为完整前缀训练样本时可以有 loss。不能仅因“history 有 loss”自动判为错误。

门禁：训练可见信息不超过推理已获得的真实观测；梯度检查点重算不重复 commit；FSDP 保存包含 gist 参数，恢复训练时 optimizer/scheduler/RNG 和数据游标完整，旧 checkpoint 只作为权重初始化时明确标注。

### 阶段 7：服务与 RMBench 事务接入

一次环境时刻的顺序必须固定为：

```text
接收 observation + episode_id + global_frame_id + observation_id
 -> 若 receipt 已存在，返回缓存的同一请求结果
 -> clean observation prefill，产生 delta（尚未提交）
 -> 原子 commit 一次
 -> 多次 video/action denoise 只读 committed state
 -> 返回 action plan 和 request_id
 -> 环境实际执行部分动作
 -> 下一真实 observation 到达后再进入下一次 commit
```

请求 ID 与 observation ID 分开：网络重试同一请求应返回相同计划；新的请求可在同一已提交观测上重新采样，但不能再次推进 frame。episode reset、成功、失败、截断、异常中断均清理 Transformer memory 和 VAE streaming state。未执行的预测动作不能作为当前 proprio 或已执行历史提交。

门禁：缓存接近预算上限时连续 50 次去噪 state digest 不变；重复请求不增加 frame；冲突请求拒绝；部分执行后只以回传真实观测推进；两路并发 episode 无串扰。

### 阶段 8：固定条件消融

第一轮机制比较固定：

- 数据：同一版 `rearrange_blocks_50demo_fullctx_apf16_cf8` 及同一 split/index。
- 初始化：同一基础 checkpoint；legacy 历史 step-750 可用于服务回归，但不能与从 base 初始化的新 memory run 冒充同训练预算比较。
- 表示：现有 EEF16 → model30 接口、同 APF16、同三相机 latent、同 prompt。
- 优化：相同有效 target exposure、batch、梯度累积、optimizer、scheduler 和 checkpoint 选择规则。
- 推理：同 seed 集、最大步数、执行 horizon、denoise schedule、attention backend 和硬件。

比较 `legacy / hybrid / recent-only / gist-only / full-history`。先在开发 seed 做少量诊断；最终 seed 和 checkpoint 规则在看最终结果前固定。记录成功率之外的 peak memory、每个环境 decision latency、prefill latency、denoise latency、逐层 KV token 数。

门禁：memory off 回归先过；每种 policy 的实际缓存集合通过 trace 复核；full-history 的计算/显存随历史增长，hybrid 的完整视觉部分有界、gist 部分按 `M × frames` 线性增长。

### 阶段 9：论文级结构决策

只有阶段 8 能回答共享骨干下 memory 是否带来可测收益后，才决定以下独立变量：

1. 论文级独立动作专家。
2. 14D joint action 与 proprioception。
3. 先拼 RGB 再 VAE 等输入重编码。

这些变化属于严格协议对齐，不应和 memory 首次接入同一次实验。它们会改变可复用参数、数据表示和服务动作接口，混在一起将无法归因 gist/anchor/recent 的贡献。

## 5. 下一步唯一推荐动作

下一步实施**阶段 4 的只读 sidecar 与 Dataset adapter 检查**，仍不改模型。原因是小模型三项机制门禁已经通过，而生产接入最早的未解决依赖是 episode/global frame/target 的可机读合同。具体范围：

1. 为 `rearrange_blocks_50demo_fullctx_apf16_cf8` 全量生成 alignment sidecar，不改原 latent/parquet。
2. 用正式 Dataset/Sampler 跑 CPU 单 batch trace，打印观测范围、目标视频、动作行、valid mask 和 episode reset。
3. 对三个代表 episode 与 numbered synthetic sequence 逐项比对。
4. 若全量元数据一致，保留现有 latent；只有发现相机、VAE、采样 manifest 不一致才触发重提取。

阶段 4 通过后，在有一张不占用任务的目标 GPU 时完成阶段 0B 的单请求 5B 数值 reference。随后才创建生产 memory 文件和配置。

## 6. 停止与切换条件

- 若 fullctx sidecar 发现系统性时间错位：停止生产接入，修 index/mask；底层 RGB/latent 先保留。
- 若 latent manifest 与当前 VAE、相机拼接或采样规则不一致：只重提取受影响特征；有证据表明转换源本身错误时才回到原始示范。
- 若 0B 无法复现固定请求：先定位 legacy 的 seed/backend/cache 生命周期，不把差异归因给 MemoryWAM。
- 若 production memory-off 不通过数值回归：禁止训练 memory-on。
- 若 streaming 与 full-prefix 在生产小尺寸 forward 不一致：先修 mask/RoPE/KV 层语义，不用增加训练步数掩盖。
- 若 gist 只有 action 或只有 video 一侧梯度：检查可见性和 loss mask；不先加辅助 loss。
- 若共享骨干 hybrid 在公平预算下无收益：该结果只否定当前共享骨干实现，不自动证明独立动作专家或 joint14 会改善；阶段 9 需作为新实验立项。

## 7. 本阶段不声称的结论

- 小模型通过不等于 5B attention/cache 已正确接入。
- delayed-recall 收敛不等于 RMBench 闭环成功率提高。
- 历史 5/5 rollout 不等于当前代码的数值回归通过。
- 同一 RMBench 名称不等于与 MemoryWAM 论文使用同一数据、动作空间、初始化或评测协议。
- 当前共享骨干、EEF16/model30、分别编码相机 latent 的路线可以用于机制消融，但不是论文 6B 双专家、joint14 协议的严格复现。
