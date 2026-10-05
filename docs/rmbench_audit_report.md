# RMBench 历史训练/推理审计与 MemoryWAM 下一步路线

审计日期：2026-09-30。本文是对上一轮 `MemoryWAM_LingBotVA_implementation_plan.md`（机制移植方案）的**落地核验与路线收敛**，不是新的架构设计。所有结论都标注证据类型：

- 【本轮运行】本轮在 `lingbot` 环境下重新执行的 CPU 脚本/命令输出。
- 【源码】本轮读取的实际源码（行号对应当前工作区快照 `HEAD=d986be8`）。
- 【历史日志】`train_out/*/training.log`、`artifacts/*/eval/_result.txt`、`client.log`、`step_trace.jsonl`。
- 【历史审计】上一轮/之前留下的 `docs/rmbench_cpu_checks.json`、`docs/rmbench_history_audit_inventory.json`、`script/rmbench_data/*.py` 的既有审计记录。
- 【论文】本地 MemoryWAM PDF 与作者公开资料。
- 【待验证】无法从现有材料恢复、需要对照实验才能定性的项。

---

## 0. 直接回答四个开头问题

### 0.1 以前在 RMBench 上的 LingBot-VA 训练/推理，哪些正确、哪些有问题、哪些缺证据

**正确（有证据）：**

1. **数据转换的动作时间契约正确。**【本轮运行 + 源码】`convert_rmbench_to_lingbot_lerobot.py:381-386` 写 `image[t] ← action[t+1]`（`source_action:243-254` 读 endpose，`wxyz_pose_to_xyzw:98-111` 做 WXYZ→XYZW），`lerobot_latent_dataset.py:_rmbench_action_post_process:278-348` 做相对化、首帧 dummy、APF16、16→30 通道映射、q01/q99 归一化。`script/audit_rmbench_cpu_checks.py` 本轮重跑通过：180 条 fullctx episode 的 parquet 数值、`raw_episode0_alignment_max_error < 1e-6`（parquet 行 t 与原始 HDF5 物理帧 t+1 对齐）、合成带编号序列、相对/绝对位姿往返误差 < 1e-12 全部通过。
2. **归一化统计来自训练数据且无泄漏。**【历史审计】9 个 fullctx 数据集的 `rmbench_action_stats.json` 是“原始 50-episode 统计量的逐字节复制”（`continuous_contract.json`），validation/test 复用同一份 stats；`split_action_duplicates` 为 `[0,0,0]`（三个 split 的动作轨迹不重叠）。
3. **训练主循环（optimizer/scheduler/梯度累积/验证早停）正常。**【历史日志】所有正式运行的 `training_metrics.jsonl` 步号连续（`steps_contiguous=True`）、loss 有限（`metrics_finite=True`）、`grad_norm` 正常下降、早停按 `validation_metrics.jsonl` 触发。
4. **50-demo 训练用整条 episode（fullctx）作为单个样本，动作首帧 dummy 正确 mask。**【源码】fullctx 每个 episode 只有 1 个 `action_config`（start=0, end=episode_length），`parse_meta:175-201` 展开成 1 个样本；`_rmbench_action_post_process` 对 frame0 置 dummy 且 `mask[:,0]=False`。
5. **动作→环境执行链路（delta→absolute、四元数、夹爪）正确。**【本轮运行 + 源码】`lingbot_client.py:_delta_to_absolute:288-319` 与 CPU 检查中的 `delta_to_absolute_eef` 反向验证误差 < 1e-12。

**已确认的问题（有证据，见 §4 排序）：**

1. **P0 推理 KV cache 的“只读”在缓存满时并非只读**（`model.py:368-414` 的 `allocate_slots` 会永久淘汰最旧 slot，`restore_cache:413` 只清本次 slots，不恢复被淘汰的旧历史）。【本轮运行】CPU 复现：容量 4、写满 4 后再以 `update_cache=0` 写 2 个 token，restore 后 live=2 且 `old_payload_overwritten=true`。影响 blocks_ranking_try 这类长 episode（见 §4.1）。
2. **P0 推理图像颜色序与离线 latent 不一致**：离线 `extract_rmbench_latents.py:108` 用 `format="rgb24"`（RGB）提 latent，而推理脚本 `run_*_inference.sh` 硬编码 `RMBENCH_LINGBOT_IMAGE_COLOR_ORDER=legacy_bgr`，客户端 `lingbot_client.py:241-253` 会把 RGB 翻成 BGR 再送给 VAE。客户端 docstring 明确“模拟器与录制视频是原生 RGB，legacy_bgr 仅用于旧 checkpoint”。fullctx（RGB）模型却用 legacy_bgr 评测。【源码 + 历史日志】
3. **P1 bounded 实验的四个索引字段全部未生效**（`train.py:_prepare_input_dict:292-321` 只取 `latents/actions/actions_mask/text_emb`）：`BoundedFullContextDataset.__getitem__:36-61` 返回的 `global_latent_start`（位置）、`history_frame_count`（burn-in 长度）、`validation_weight`（episode 均衡）在 Trainer 中都没被读；`bounded_validation_index` 未用于构造验证集（`train.py:167` 仍用完整 `MultiLatentLeRobotDataset`）。因此“bounded 有 32 帧 history”实际上是把 history 也当成监督目标（history 占窗口 43.6%），且每个窗口 RoPE 从 0 重编号、动作原点仍是 episode 帧 0（模型看不到）。【源码 + 本轮运行量化】
4. **P1 所有 checkpoint 都是“仅权重”，不是断点续训**（`train.py:save_checkpoint:495-540` 只存 bf16 safetensors + config；optimizer/scheduler/RNG 保存被注释，`_load_training_state` 从未被调用）。`resume_from` 在所有运行里都是 None，全部从 `checkpoints/base` 初始化。【历史日志】
5. **P2 训练/推理注意力窗口与 chunk 不一致**：训练 `window_size` 随机 4..64 latent 帧、`chunk_size` 1..4（`train.py:318-319`）；推理 `attn_window=72` 的滚动缓存且无 window mask（`model.py:custom_sdpa` 与 flash 不遮），chunk 固定 2。推理实际注意力范围（~72 latent 帧）超出训练窗口上限（64）。【源码】

**仍缺证据（明确记录缺口）：**

1. **`extract_rmbench_latents.py` 从未进 git**（`script/rmbench_data/` 被 `.gitignore` 忽略，`git log` 无记录），无法证明 Sep 15-20 提取 fullctx latent 时用的就是当前这份 `rgb24` 版本。只能以当前源码 + 客户端 docstring 推断 RGB。
2. **旧训练运行时的源码不可恢复**：训练发生在 9/20-9/27，`63b1ba9`(9/22) 与 `d986be8`(9/30) 之间可能对 `train.py`/`model.py` 有改动；无法严格证明每次运行时 `update_cache`/`_prepare_input_dict` 与当前快照逐行一致。缓存 bug 属于机制性代码路径，其结构在多版本中稳定，但“某次运行是否真的触发了淘汰”只能靠 episode 长度推断（blocks 27/50 条 > 72 latent 帧）。
3. **observe_and_pickup 失败根因未定位**：它 episode 只有 10 latent 帧，不受缓存 bug 影响，但 5/5 失败。候选原因是颜色序、任务更难、或 550 步早停导致 action_loss（0.0127）比 rearrange（0.0017）高 7 倍，需对照实验定性。

### 0.2 fulltx/fullctx 数据哪些能沿用、哪些要改/重提取/重建

实际准确名称是 **`<task>_<split>_fullctx_apf16_cf8`**（fullctx = full-context，apf16 = action-per-frame 16，cf8 = 提取时 chunk_frames=8）。没有名为 `fulltx` 的目录；`fulltx` 是会话中的口语指代。9 个正式数据集（3 任务 × {50demo, 5demo_validation, 5demo_test}）经审计为 clean。

结论见 §3 资产复用表。一句话：**fullctx 的 latent/动作/文本/统计当前可直接复用做 LingBot 基线；做 MemoryWAM 机制版可保留底层数据但必须改索引/采样/mask；做论文严格对齐（joint14 + RGB mosaic + proprio）需要回到原始 RMBench 数据重新转换，不能复用现有 latent。**

### 0.3 从哪个 checkpoint、哪套数据与哪个配置开始下一步

**checkpoint**：`train_out/rearrange_blocks_50demo_1000step_20260921_180241/checkpoints/checkpoint_step_750/transformer`（rearrange 唯一在独立场景达到 5/5 的 50-demo 模型），或 `observe_50demo_1000step_20260924_06/checkpoints/checkpoint_step_550`、`blocks_bounded_3000step_20260924_224653/checkpoints/checkpoint_step_2100`。这些都只能当**初始化权重**，不能当断点续训（无 optimizer/RNG）。

**数据**：`rearrange_blocks_50demo_fullctx_apf16_cf8` + `rearrange_blocks_5demo_test_fullctx_apf16_cf8`（5 个 held-out 场景 seed 200000-200004）。

**配置**：`rmbench_train`（`va_rmbench_train_cfg.py`，继承 `va_robotwin_train_cfg.py`，实际 batch=1/gradacc=1/world=3/lr=1e-5/apf16/fcs2/attn_window72/seed42/cfg_prob0.1）与 `rmbench`（`va_rmbench_cfg.py` 推理，25 视频步/50 动作步）。**但下一步不是直接续训，而是先用修复后的推理复测已有 checkpoint（见 §6）。**

### 0.4 与 MemoryWAM 报告结果的实际差距，以及最小返工的基线/收益验证路径

**本地结果不能与 MemoryWAM 论文结果直接比较**，因为：动作空间（本地 EEF16D→30D vs 论文 joint14D）、观测（本地三相机分别 VAE 后拼 latent vs 论文先拼 RGB mosaic 再 VAE）、模型（本地共享 30 层×3072 骨干 vs 论文视频骨干 + 独立 1024 宽动作专家）、示范量与协议、以及本地存在的缓存/颜色/bounded 三个已确认问题。**“同一 benchmark”不等于“同一数据与协议”。**

最小返工路径：**先修基线（缓存淘汰 bug + 颜色序），复测现有 fullctx checkpoint，得到可信的 LingBot-VA 本地基线；再在同一数据/初始化/动作表示/训练预算下加 gist/anchor/recent 做机制对照**。不要先重做数据，也不要先堆独立动作专家/joint14（详见 §7）。

---

## 1. 历史运行与 checkpoint 对应表

正式运行均：初始化自 `/root/wyc/code/lingbot-va2/checkpoints/base/transformer`（3 分片 ~10.2GB，30 层/24 头/head128/ffn14336/action_dim30/text4096/patch(1,2,2)）；3×FSDP rank；batch=1、gradaccum=1（有效 batch=3）、lr=1e-5、warmup10、wd0.1、seed42、apf16、fcs2、attn_window72、cfg_prob0.1。加载时日志对每个 rank 报 `patch_embedding.weight/bias` 未使用——这是 Wan2.2 视频 DiT 遗留键，LingBot 用 `patch_embedding_mlp` 替换，属良性（【历史日志】）。

| 运行目录 | 任务/数据 | 完成步数(早停) | 权重 checkpoint | 评测结果 |
| --- | --- | --- | --- | --- |
| `rearrange_blocks_50demo_300step_20260921_104117` | rearrange / fullctx50demo | 300 | step_300 | test 1ep=1.0, 2ep=0.5 |
| `rearrange_blocks_50demo_1000step_20260921_180241` | rearrange / fullctx50demo | 750(早停) | step_750 | test 5ep=**1.0 (5/5)** |
| `observe_50demo_1000step_20260924_06` | observe_and_pickup / fullctx50demo | 550(早停) | step_550 | test 5ep=**0.0 (0/5)** |
| `observe_50demo_1000step_20260924_01` | observe_and_pickup / fullctx50demo | 8(中断) | 无 | — |
| `blocks_bounded_formal_20260922_214515` | blocks_ranking / **bounded** h32_f64 | 1000(早停) | step_1000 | test 5ep=**0.2 (1/5)** |
| `blocks_bounded_3000step_20260924_224653` | blocks_ranking / **bounded** h32_f64 | 2100(早停) | step_2100 | test 5ep=**0.6 (3/5)** |
| `blocks_bounded_pilot_20260922_161832` | blocks_ranking / bounded | 100 | step_100 | train 1ep=0.0 |
| `blocks_bounded_pilot_20260922_161648` | blocks_ranking / bounded | 0(未启动) | 无 | — |
| `blocks_bounded_profile_20260922_160912` | blocks_ranking / bounded | 2(profile) | step_2 | — |
| `blocks_ranking_try` | blocks_ranking / fullctx | 1(OOM) | 无 | — |
| `observe_and_pickup` | observe / fullctx | 2(smoke) | step_2 | — |
| `rearrange_blocks` | rearrange / fullctx | 2(smoke) | step_2 | — |
| `blocks_debug_2step`、`debug_blocks_2step_*` | blocks / debug | 失败(`invalid device ordinal`) | 无 | — |
| `rearrange_debug_2step_20260930_*`(×5) | rearrange / debug | 失败(`ChildFailedError`) | 无 | — |

早期 episode0 过拟合（不在 `train_out`，在 `checkpoints/`）：
- `rearrange_blocks_episode0_overfit_1000step_20260915` → seed0 推理 0.0（w193 时代，BGR）。
- `rearrange_blocks_episode0_fullctx_300step_20260917` → fullctx 推理 1.0（RGB fullctx）。
- `observe_blocks_episode0_300step_20260919_151950` → observe/blocks episode0 各 1.0。

评测协议可追溯性（【历史日志 + 源码】）：所有闭环评测走外部 `/root/wyc/code/RMBench/script/eval_policy.py`，`--instruction_type seen` 表示**任务指令文本是 canonical 提示**（不是 held-out 指令），泛化维度是**场景 seed**：validation=100000-100004、test=200000-200004（`holdout_work/<task>/{split}/expert_seeds.json`，与训练 seed 区间不相交）；成功率分母 = `--eval_num_episodes`（test 为 5）；`train` split 是训练场景 seed0 回放（COUNT=1，不能算泛化）；最大步数由 `RMBENCH_EVAL_STEP_LIMIT` 控制（rearrange=500、observe=250、blocks=3000），stall/window reset 均已禁用（`RMBENCH_EVAL_STALL_ENABLE=0`、`RMBENCH_LINGBOT_WINDOW_RESET_INTERVAL=0`）。因此 rearrange 的 5/5、observe 的 0/5、blocks 的 0.2/0.6 都是**独立 held-out 场景**的泛化结果，不是训练集回放。

关键事实（【历史日志】）：
- 每个 checkpoint 都是 **839 张量、约 10.18GB、无 `training_state.pt`** → 全部“仅权重”，不可断点续训。
- **blocks_ranking_try 从未做过整段 fullctx 训练**，只有 bounded（窗口 max64/history32）与一次 2-step OOM smoke。它的 50 条 episode latent 帧数 35..152（均值 92），其中 **27 条超过 72 latent 帧**（缓存容量上限，见 §4.1）。
- observe_and_pickup 50demo 每条仅 10 latent 帧；rearrange 25-26 latent 帧。

---

## 2. fulltx/fullctx 数据盘点（20 个数据集目录）

`/root/wyc/data/rmbench/lingbot_va2/` 下 20 个数据集，按代际分三类：

- **旧 w193（BGR，已弃用）**：`blocks_episode0_w193`、`blocks_episode0_w193s64`、`blocks_ranking_try_50demo_w193_{apf16_official,v1,v2,v3,v4_lossless,v5_fresh_latents}`。契约字段 `image_color_order: legacy_bgr`，窗口 stride192，APF4（v1/v2）。这些是早期调试产物，**不要再用于新训练**。
- **fullctx（RGB，正式，9 个）**：`{rearrange_blocks, observe_and_pickup, blocks_ranking_try}_{50demo, 5demo_validation, 5demo_test}_fullctx_apf16_cf8`。每条 episode 一个 `action_config`（start=0..episode_length），`frame_step=4`、`chunk_frames=8`、`sampled_fps=3.75`、`action_per_latent_frame=16`、`frame_chunk_size_latents=2`、`first_chunk_valid_actions=16`。`continuous_contract.json` 记录完整时序契约（`sampled_video_frame_ids`、`latent_frames`、`valid_action_physical_frames`、`front_padding=16`、`tail_padding`）。latent/动作/文本/统计经 `audit_rmbench_cpu_checks.py` 验证 clean（0 issue）。
- **派生/中间产物**：`blocks_ranking_try_episode0_fullctx_apf16_cf8`（267 个 missing latent，已废弃的半成品）、`blocks_ranking_try_bounded_v1_h32_f64`（bounded 索引，159 train + 14 validation 窗口）、`holdout_work/`（validation/test 的 expert seed 清单与源）。

fullctx 的 `data/`、`videos/` 是指向 `/root/wyc/data/rmbench/lingbot_lerobot/<task>/` 的软链接；真正的重资产是 `latents/`（预提取的 VAE latent + text emb）。

时序合同（已核对源码 + 契约 + 本轮运行）：设源帧步长 4，所选 RGB 源索引 `0,4,8,12,...`，latent 终点 `z0@0, z1@16, z2@32,...`；存储动作组 `A0=dummy(不参与 loss)`、`A1=物理动作1..16`、`A2=17..32`。**parquet 行 t 存物理目标 t+1**，因此 `A[t]` 数组下标与 `z[t]` 数组下标相同但语义不同（`A[t]` 是“见 z[t-1] 后执行的下一段”）。推理侧动作 chunk 用的是 `z[0..t]` 预测 `A[t+1]`（详见 §4.5 关于未来视频 teacher forcing 的讨论）。

---

## 3. fulltx/fullctx 资产复用表

“原 LingBot 基线”= 修复后的本地 EEF16D 共享骨干；“MemoryWAM 机制版”= 在共享骨干上加入逐帧 gist + anchor/recent 视觉记忆、action-only 闭环；“严格论文对齐”= joint14D + proprio + RGB mosaic 先拼后 VAE + 独立动作专家。

| 资产/版本 | 已验证事实 | 原 LingBot 基线 | MemoryWAM 机制版 | 严格论文对齐 | 必要处理 |
| --- | --- | --- | --- | --- | --- |
| fullctx parquet 动作 `[T,16]` EEF+夹爪 | 行 t=物理 t+1，四元数 XYZW，q01/q99 来自训练集 | **可直接复用** | **可保留底层数据，但必须修改索引、采样或 mask**（需把 A[t+1] 决策时刻显式化，动作历史仅在 Swap T 类任务提供） | **需要回到原始数据重新转换**（要 joint14D + 当前 proprio，转换器目前不保存 proprio） | 机制版新增 episode UID/全局 frame_id 元数据 |
| fullctx VAE latent（三相机分别编码后拼 24×20×48） | 150/任务 latent 文件 PASS，finite，RGB 编码，因果流式 | **可直接复用** | **可保留底层数据，但必须修改索引、采样或 mask**（要保留连续历史 + 全局 latent id，不做随机窗口） | **需要重新提取特征**（论文是先拼 RGB mosaic 再 VAE，数值不同） | 不要覆盖旧 latent；mosaic 单列新目录 |
| fullctx 文本 emb（T5 512×4096） | finite，与 action_text 对齐，cfg_prob=.1 换 empty | **可直接复用** | **可直接复用** | **可直接复用**（若指令一致） | 保持任务级指令，禁用专家阶段特权文本 |
| `rmbench_action_stats.json`（q01/q99） | 逐字节复制原 50-episode 统计 | **可直接复用** | **可直接复用**（EEF16D） | **需要重新提取特征**（joint14D 需重算，仅 train split） | — |
| `observation_anchors.json` | 是**动作坐标原点**，不是 MemoryWAM 视觉 anchor | **可直接复用** | 部分复用（仅作 EEF 相对原点） | **需要回到原始数据重建** | 勿与视觉 anchor 混用 |
| bounded 索引 `blocks_ranking_try_bounded_v1_h32_f64` | 159+14 窗口，字段完整但 Trainer 未消费 | 不推荐沿用（语义未生效） | 不沿用（采样不满足逐帧记忆） | 不沿用 | 若保留须先修 Trainer 消费 + burn-in mask |
| 旧 w193 系列 | BGR、APF4、stride192，早期弃用 | 不沿用 | 不沿用 | 不沿用 | 仅作历史，不参与新实验 |
| `empty_emb.pt` | 每数据集一份 | **可直接复用** | **可直接复用** | **可直接复用** | — |

---

## 4. 按严重程度排序的问题

### P0-1：推理 KV cache `update_cache=0` 满时非只读，静默淘汰最旧历史

- **证据**：【源码】`model.py:368-414`（`allocate_slots` 空间不足按写入 id 淘汰最旧 slot）、`model.py:447-463`（`forward` 无条件先 `update_cache` 再在 `update_cache==0` 时 `restore_cache`，而 `restore_cache:413` 只清本次 slots 的 mask，不恢复被淘汰的旧 slot）。【本轮运行】CPU 复现：容量 4 写满 4，`update_cache=0` 再写 2 → restore 后 live=2，`old_payload_overwritten=true`。
- **影响范围**：`_infer:524/564` 视频去噪 25 步、动作去噪 50 步都以 `update_cache=0`（非最后一步）反复前向，每个前向写 `latent_token_per_chunk=240` 或 `action_token_per_chunk=32` 个 token 再“还原”。缓存容量 = `(72//2)*240 + (72//2)*32 = 9792` token ≈ **72 latent 帧**。当 episode > 72 latent 帧，每个 chunk 的第一次只读视频前向会**永久淘汰最旧 240 token（最旧 2 个 latent 帧）**。
- **受影响数据/运行**：blocks_ranking_try（50 条中 **27 条 > 72 latent 帧**，最长 152）。rearrange（≤26）、observe（10）不受影响。
- **最小修复**：`restore_cache` 之外，在 `allocate_slots` 淘汰前先保存被淘汰 slot 的 `(mask,id,k,v)`，`update_cache=0` 路径在 restore 时一并回滚；或给只读前向用独立 staging 不写持久池。~20 行改动。
- **是否要重训**：否，只改推理；但 blocks_ranking_try 模型本身还有 bounded 训练问题（P1-1），修缓存只能止损，不能单独恢复其成功率。

### P0-2：推理图像颜色序 `legacy_bgr` 与 fullctx 离线 RGB latent 不一致

- **证据**：【源码】`extract_rmbench_latents.py:108` `format="rgb24"`；`lingbot_client.py:241-253` 的 `_model_image` 在 `RMBENCH_LINGBOT_IMAGE_COLOR_ORDER=legacy_bgr` 时做 `image[...,::-1]`（RGB→BGR），且 docstring 写明“模拟器与录制视频是原生 RGB，legacy_bgr 仅用于旧 checkpoint”。【历史日志】`run_rearrange_50demo_inference.sh:82`、`run_blocks_bounded_inference.sh:82` 都硬编码 `legacy_bgr`。
- **影响范围**：当前推理脚本硬编码 `legacy_bgr`，因此用它们跑出的 fullctx（RGB）模型评测（rearrange 5/5、observe 0/5、blocks 0.2/0.6）都会以 BGR 输入、与 RGB 训练 latent 不一致。注意：`script/rmbench_inference/` 同样被 `.gitignore` 忽略、无版本记录，故“某次评测当时用的脚本就是当前这份”仍是【待验证】，但当前脚本的 `legacy_bgr` 与离线 rgb24 的矛盾是确定存在的。
- **受影响数据/运行**：全部 fullctx 推理。
- **最小修复**：把两个推理脚本的 `RMBENCH_LINGBOT_IMAGE_COLOR_ORDER` 改为 `rgb`（或删掉该 env，默认即 rgb）。
- **是否要重训**：否。**待验证**：rearrange 在 BGR 下仍 5/5，说明它对此不敏感；observe 的 0/5 是否因此尚未定位，需 rgb 复测定性。

### P1-1：bounded 训练的 global/history/validation_weight/bounded_validation_index 全部未生效

- **证据**：【源码】`BoundedFullContextDataset.__getitem__:49-58` 返回 4 个元数据；`train.py:_prepare_input_dict:292-321` 只读 `latents/actions/actions_mask/text_emb`（位置 grid 由 `_add_noise` 从 0 重建，`f_shift=0`）；`compute_loss:381-431` 只有 `actions_mask`，无 history burn-in；`train.py:167` 验证集固定 `MultiLatentLeRobotDataset`，忽略 `bounded_validation_index`。【本轮运行】量化：train 159 窗口 history 占 43.6%（3538/8110 latent 帧），validation 40.2%。
- **影响范围**：blocks_ranking_try 唯一训练路径（bounded）实际上变成“随机中段窗口 + 全窗口监督 + RoPE 归零 + 相对 episode 帧 0 原点（不可见）”，训练分布与推理（从 episode 起点累积 KV、全局位置）不一致。
- **最小修复**：二选一 —— (a) 给 Trainer 消费 `global_latent_start`/`history_frame_count` 并做 burn-in mask、用 `bounded_validation_index` 建验证集；或 (b) 放弃 bounded，直接对 blocks_ranking_try 做整段 fullctx 训练（配更长 `attn_window` 或先修缓存淘汰）。
- **是否要重训**：是（blocks_ranking_try 需要重训；现有两个 bounded checkpoint 只能作诊断，不能作正式基线）。

### P1-2：checkpoint 仅权重、无完整训练态

- **证据**：【源码】`train.py:495-540` optimizer/RNG 保存被注释；`:202-203` 与 `_load_training_state` 均未接线。【历史日志】所有 checkpoint 无 `training_state.pt`。
- **影响**：任何“续训”都只能从 base 重载权重、重置 optimizer/scheduler/RNG，等价于微调新任务，不能精确复现中间 step 的采样与优化器动量。
- **最小修复**：接回 `save_checkpoint` 的 optimizer/scheduler/RNG/sampler 状态（保存 `training_state.pt`），并在 `Trainer.__init__` 调用 `_load_training_state`。
- **是否要重训**：不影响已训权重复用，但未来正式训练必须补。

### P2-1：训练/推理注意力窗口与 chunk 不一致

- **证据**：【源码】训练 `window_size∈[4,64]`、`chunk_size∈[1,4]`（`train.py:318-319`）；推理 `attn_window=72` 滚动缓存、`custom_sdpa`/flash 无 window mask、chunk 固定 2（`model.py:37-40,304`；`va_robotwin_cfg.py:11-12`）。
- **影响**：推理实际注意力范围（~72 latent 帧）超出训练窗口上限 64；chunk=2 在训练范围内但非专精。对长 episode 影响更大。
- **最小修复**：把训练 `window_size` 上界对齐到 ≥72，或推理显式施加与训练一致的 window mask。**待验证**：对当前短任务（rearrange/observe）影响有限，主要影响 blocks 与 MemoryWAM 长历史。

### P2-2（信息）：`patch_embedding` 未使用告警

- 良性。base 是 Wan2.2 视频 DiT，含 `patch_embedding.weight/bias`，LingBot 用 `patch_embedding_mlp`（`model.py:633`），`_keys_to_ignore_on_load_unexpected` 未含它导致告警，但无功能影响。

### 附：observe_and_pickup 0/5 的定位（非独立 bug，是待验证根因）

- 训练 action_loss=0.0127（rearrange 为 0.0017，高约 7 倍）、episode 仅 10 latent 帧、早停在 550 步。候选：颜色序、抓取任务更难、训练不足。不新增修复，先做 P0-2 的 rgb 复测。

---

## 5. 正确性判定（区分三个目标）

- **A. 原 LingBot-VA 训练/推理语义**：动作后处理、未来视频→动作的两段生成、teacher-forced 未来视频条件、dummy 首帧、APF16 都**正确且自洽**。缓存只读 bug 是通用推理缺陷，非 LingBot 语义本身错误。
- **B. 本地 RMBench 适配所声称的语义**：数据转换（image[t]→action[t+1]、WXYZ→XYZW、EEF16D、三相机）**正确**；但“bounded 有 32 帧 history burn-in”的声称**未实现**，颜色序**未与离线对齐**，缓存**满时非只读**。
- **C. MemoryWAM 机制需要的语义**：当前**不具备**（无 gist/anchor/recent、无逐帧混合可见性、无 action-only 闭环、无全局 latent 时间坐标贯穿训练）。注意：**“没有跨 batch KV”不等于“没有记忆训练”**——完整序列因果 mask 训练本身可以在单前向内提供长历史；当前真正的缺口是**训练与推理可见性/坐标不一致**，而非缺跨 batch 持久缓存。

“不同于 MemoryWAM”≠“以前训练错误”；上面 A/B/C 的判定已分开。

### 5.1 训练/推理关键语义逐项核对（回答 §四/§五 清单）

**attention mask 的条件结构（视频生成→动作生成）**【源码 `model.py:95-201,740-791`】：

- 训练 token 物理顺序是 `[noisy_video, condition_video, noisy_action, condition_action, padding]`（`forward_train:740-743`），不是逐帧 interleave；真正的时序/模态顺序由 mask 表达。
- `frame_ids`：视频块号 `= frame//chunk_size*2`（偶数），动作块号 `= frame//chunk_size*2+1`（奇数），即视频块 0、动作块 0、视频块 1、动作块 1… 交错。
- 三条核心规则：`clean→clean` 用 `kv_block<=q_block` 因果；`noise→clean` 用 `kv_block<q_block`（严格更早）；`noise→noise` 只允许同块（`block_self`）；再交 `seq_mask`（同样本）与 `window_mask`（`|frame差|<=window_size`）。
- **由此推出**：动作去噪 query 在动作块 k（id 2k+1）能读到 id<2k+1 的 clean token = **clean 视频块 0..k（含 z_k，即动作块 k 执行后的未来视频）+ clean 动作块 0..k-1（历史动作的真值）**。动作条件 `noisy_cond_prob=0.0`（`train.py:309`），所以动作块 k 去噪时拿的是**真值未来视频 + 真值历史动作**。这是 LingBot-VA“先生成未来视频再预测动作”架构的 **teacher forcing**，属自洽设计；但对 MemoryWAM 的 action-only 闭环，保留这个映射就会引入推理不可见条件（`z_k` 只能由预测得到，不能给真值）。

**视频/动作 loss 的有效维度与权重**【源码 `train.py:compute_loss:381-431`】：

- 视频：逐帧 MSE × 训练权重，分母 = 每帧 H·W·C 元素数，再对帧平均；**视频不 mask**（fullctx 无视频 padding，`tail_padding` 已在数据侧截断）。
- 动作：逐帧 MSE × 训练权重 × `actions_mask`，每帧 sum/该帧有效通道数，**只对含有效动作的帧取外层平均**（首帧 dummy 与尾部全 mask 帧被排除，`valid_action_frames`）。16→30 通道中 14 个无效维度 mask 恒为 False：`_add_noise` 会把它乘进 noisy/target/condition（`train.py:262-264`），loss 也乘 `actions_mask`（`train.py:409`），因此无效维度不参与训练；推理端再显式置零（`wan_va_server.py:583` `actions[:, ~self.action_mask] *= 0`）。
- 最终 `loss = latent_loss + action_loss`（1:1，各自除以 `gradient_accumulation_steps`）。没有独立的 gist/压缩/对比 loss。

**state 字段的真实含义**【源码 `lingbot_client.py:217-237` + `wan_va_server.py:602-603`】：`state = self._last_raw_action`，是模型上一段**动作计划**（delta EEF `[16,2,16]`，含首帧 dummy 与未执行尾部），经 `preprocess_action` 归一化后以 `update_cache=2` 提交为“条件动作 KV”。**它不是当前本体状态（proprio）**，也不该被当成“已执行控制指令”；MemoryWAM 需要的是 14D 当前 proprio，必须回到原始数据重建，不能用这个字段替代。

**首轮/后续 horizon、dummy、图像采样、APF 一致性**【历史日志 `step_trace.jsonl` + 源码 `wan_va_server.py:_infer/_compute_kv_cache`】：

- 首轮 `frame_st_id=0`：视频首帧用 `init_latent` 作条件、动作帧 0 用零作 dummy，输出 2 latent 帧 = 32 动作槽，客户端 `start_frame=1` 跳过 dummy → **实际执行 16 步**，回传 4 张关键帧（每 4 步采 1）。
- 后续轮：输出 2 latent 帧 = 32 动作槽（无 dummy）→ 执行 32 步，回传 8 张关键帧。
- APF16 与 VAE 时域压缩 4 一致：4 关键帧 → 1 latent 帧；8 关键帧 → 2 latent 帧；离线提取 `frame_step=4`、`chunk_frames=8` 与在线流式编码规则一致（首块 1 帧、后续 4 的倍数）。
- **唯一不一致**：训练是整段 episode 一个样本（fullctx），不区分“首轮 16/后续 32”；推理才有这个隐式首轮差别。训练 `chunk_size∈[1,4]` 覆盖了推理的 chunk=2，但并非专精。

**reset / 部分执行 / 异常中断的历史处理**【源码 `wan_va_server.py:396-431,592-624`】：`_reset` 清 Transformer 缓存 + 两个 VAE 流式缓存 + `frame_st_id=0`；`_compute_kv_cache` 先 `clear_pred_cache` 再提交真实观测。`update_cache=0` 的只读语义失效已列 P0-1；此外，若中途只执行了 k<16 步而 episode 继续，客户端仍把 16 步整段计划当 `_last_raw_action` 提交（未按实际执行步数裁剪），这是与 MemoryWAM“按真实观测提交”的生命周期不一致的**待验证风险**（当前评测未见异常，但长任务/提前终止下会影响条件动作 KV）。

---

## 6. 下一步唯一推荐路线与第一个可执行实验

### 唯一推荐路线：先修基线，不重做数据，不接入 gist/anchor/recent

顺序（明确切换条件）：

1. **修 P0-2（颜色序）**：把 `run_rearrange_50demo_inference.sh`、`run_blocks_bounded_inference.sh` 的颜色序改为 `rgb`（默认）。成本极低（改 2 个 env）。
2. **修 P0-1（缓存淘汰）**：`model.py` 的 `restore_cache` 回滚被淘汰 slot（或只读前向用 staging）。成本 ~20 行。
3. **复测现有 checkpoint**（不改训练）：rearrange step_750、observe step_550、blocks step_2100，各在 5-seed test 上跑。得到**可信的本地 LingBot 基线**。
4. **切换条件**：只有当 (a) rearrange 在 rgb 下仍 ≥0.8、且 (b) observe/blocks 的失败可归因于“训练/表示”而非“推理 bug”之后，才进入下一步。若修推理后基线仍低，先修训练（observe 加步数、blocks 改整段 fullctx），**不要**此时接入 gist/anchor/recent。
5. 基线可信后，再在同一数据/初始化/动作表示/训练预算下加 gist/anchor/recent 做机制对照（见 §7）。

### 第一个可执行实验

- **任务**：rearrange_blocks，5-seed test（seed 200000-200004，`holdout_work/rearrange_blocks/test/expert_seeds.json`）。
- **数据路径**：`/root/wyc/data/rmbench/lingbot_va2/rearrange_blocks_50demo_fullctx_apf16_cf8`（stats/empty_emb）+ `rearrange_blocks_5demo_test_fullctx_apf16_cf8` 的 seed manifest。
- **初始化**：`train_out/rearrange_blocks_50demo_1000step_20260921_180241/checkpoints/checkpoint_step_750/transformer`（仅权重）。
- **配置**：`rmbench` 推理，faithful（视频 25 步/动作 50 步），`RMBENCH_LINGBOT_IMAGE_COLOR_ORDER=rgb`（修 P0-2），其余沿用 `run_rearrange_50demo_inference.sh`。
- **硬件需求**：推理服务 2×GPU（server）+ 1×GPU（env）合计 3×3090；当前本机仅见 1×3090（23.5GB 空闲，另有 560MiB 小进程），故**需先确认 3 卡可用，且不抢占现有任务**。若仅 1 卡，退化为只跑离线 forward/解码诊断，不做闭环。
- **训练/评测预算**：零训练；闭环评测 5 episode，rearrange step_limit=500，预计每 episode 推理 ~100s/chunk × ~13 chunk。
- **预期回答的问题**：颜色序修复是否改变 rearrange 结果（若仍 5/5，说明 rearrange 对颜色不敏感，可把它当作“推理 bug 无关”的干净基线；若下降，说明 rgb 才是正确序，之前的 1.0 有 BGR 训练残差成分）。
- **通过条件**：`eval/_result.txt` 成功率为 5/5 或 ≥4/5，且 `step_trace.jsonl` 无 NaN/爆量动作；否则回退排查。

---

## 7. 与 MemoryWAM 对齐表与后续改进假设

### 7.1 对齐表（本地 vs 论文，仅列可核实项）

| 维度 | MemoryWAM（论文/作者资料） | 本地 LingBot-VA（当前） | 差距 | 对齐所需 |
| --- | --- | --- | --- | --- |
| 动作 | joint14D（14 维关节） | EEF16D→模型 30D | 空间不同 | 严格复现需重新提取 joint+proprio |
| 观测编码 | 先拼 RGB mosaic 再 VAE（384×320） | 三相机分别 VAE 后拼 latent（24×20） | latent 数值不同 | 严格复现需重提取 mosaic latent |
| 模型 | 视频骨干 + 独立 1024 宽动作专家 | 共享 30 层×3072 | 参数结构不同 | 严格复现需新增动作专家 |
| 记忆 | 每帧 M=8 gist + anchor2 + recent4 | 无（只有无类型 KV 池，还带淘汰 bug） | 缺机制 | 机制版先加 gist/anchor/recent |
| 推理闭环 | action-only，动作读记忆，不生成未来视频 | 先视频去噪再动作去噪，动作读预测视频 | 推理路径不同 | 机制版改 action-only + clean prefill |
| 训练噪声 | §4.1 logit-normal + 条件增强 p=1 | 均匀 timestep index + 条件加噪 p=0.5 | 分布不同 | 严格复现需换 scheduler |
| 示范/协议 | 作者报告其数量/seed/次数 | 本地 50 demo、5+5 holdout seed | 不可直接比较 | 对齐评测协议后重测 |

**可比较性结论**：本地已有结果**不能**与论文结果直接比较；本地 LingBot 基线是否“偏低”也**不能**下结论，因为本地带三个已确认推理 bug 且动作/观测/模型都不同。**最便宜的证伪实验**是修掉 P0-1/P0-2 后复测现有 checkpoint，先拿到“推理 bug 无关”的本地基线，再谈与论文的差距。

### 7.2 现有共享骨干能验证哪些记忆收益

能验证：**“逐帧 gist + anchor/recent 视觉记忆是否降低长 episode 的动作误差/成功率”**——在完全相同的共享骨干、EEF16D、fullctx 数据、训练预算下，对照“无记忆（修好缓存的因果 fullctx）”vs“加 gist/anchor/recent”。这验证**机制贡献**，不声称复现论文的 6B 架构/延迟/成功率。

### 7.3 哪些是严格复现必需 vs 有证据改善当前任务

- **严格复现必需**：joint14D 动作 + 当前 proprio、RGB mosaic 先拼后 VAE、独立 1024 宽动作专家、论文噪声方案、action-only 闭环。这些当前**都没有证据**表明会改善本地任务成功率；它们是“复现论文”的必需品，不是“修复本地基线”的必需品。
- **有证据会改善当前任务**：只有修 P0-1 缓存淘汰（blocks 长 episode 直接受益）与修 P0-2 颜色序（待验证）。其余（gist/anchor/recent）是**探索性改进**，无现有证据证明能超过修好后的因果 fullctx 基线。
- **成功率接近上限的任务（如 rearrange 已 1.0）**：可以且应该把目标改为“保持成功率并降低延迟/显存”——这正是 MemoryWAM 声称的效率收益（recent4 而非 72 帧缓存），是当前共享骨干下最有把握验证的收益点。

### 7.4 最多三个后续改进假设（各列瓶颈→改动→机制→最小实验→成本→失败解释）

**H1（修复现有问题）**：缓存淘汰 + 颜色序 → 复测现有 fullctx checkpoint → 得到可信基线。
- 瓶颈：长 episode 静默丢历史 + 输入颜色错配。
- 最小实验：§6 的 rearrange/observe/blocks 复测。
- 成本：~20 行模型改动 + 2 个 env + 3×5 episode 闭环。
- 失败解释：若 rearrange 仍 1.0 而 observe/blocks 仍低，则问题在训练/表示，不在推理。

**H2（机制对照，探索性）**：在共享骨干上加逐帧 gist(M=8)+anchor(2)+recent(4)，训练/推理一致的混合可见性，action-only 闭环，固定数据/初始化/动作表示/预算，对照 H1 基线。
- 机制：gist 提供不随缓存淘汰的压缩历史，anchor 保任务起点，recent 保近期细节。
- 最小实验：blocks_ranking_try 或一个合成长程记忆任务，先离线逐帧 reference 验证可见性 mask 正确，再小步闭环。
- 成本：模型/训练/服务三处改动（见上一轮 plan §4），中等。
- 失败解释：若机制版不超过因果 fullctx 基线，说明共享骨干 + EEF 表示下记忆收益有限，需转向动作专家/joint 表示（H3）。

**H3（严格复现，仅在 H2 证伪后）**：独立动作专家 + joint14D + RGB mosaic + 论文噪声，对齐论文协议后报告差距。
- 机制：动作专家解耦动作条件、mosaic 统一观测、joint 动作对齐 RMBench 真实控制。
- 最小实验：先重提取小规模（5 demo）mosaic latent + joint 动作做数值/形状验收，再决定是否全量。
- 成本：高（数据重提取 + 模型新模块）。
- 失败解释：若仍不达论文，说明未对齐的初始化/预训练/评测次数等仍有关键差距，需回作者资料核实。

---

## 8. 明确结论（“特别明确”清单）

- **不用重做的数据**：9 个 fullctx 数据集的 parquet 动作、三相机 latent、文本 emb、q01/q99 统计、empty_emb，全部可复用（机制版只需改索引/采样/mask 与新增元数据，不需要重提取）。
- **需要回到原始数据重建的**：只有“严格论文对齐”的 joint14D+proprio、RGB mosaic latent、以及 MemoryWAM 视觉 anchor（现有 `observation_anchors.json` 是动作原点，不是视觉 anchor）。
- **可作为初始化但不能称断点续训的旧 checkpoint**：`checkpoint_step_750`（rearrange）、`checkpoint_step_550`（observe）、`checkpoint_step_2100`（blocks bounded）、`checkpoint_step_300`/`_1000` 等全部旧权重——都可作初始化，但都无 optimizer/scheduler/RNG，任何续训等价于从 base 微调。
- **可保留的旧结果**：rearrange 50demo 的 5/5（作短任务上界参考，但需 rgb 复测确认）、episode0 过拟合 1.0（证明管线闭环可行）、CPU 数据/动作/mask/缓存审计（数据与转换正确性的直接证据）。
- **必须重测的旧结果**：observe 0/5、blocks bounded 0.2/0.6（受 P0-1/P0-2/P1-1 污染，不可作为基线）；rearrange 的 1.0 也建议在 rgb 下复测一次以排除 BGR 残差。
- **下一步应该：先修复基线**（修缓存淘汰 + 颜色序 + 复测），**不补齐数据**（fullctx 已够），**暂不接入 gist/anchor/recent**（等基线可信后再做机制对照）。训练主流程只做补丁级修改（§4 的最小修复），不同时重构模型/数据/服务协议。

---

## 9. 本轮新增/修改文件清单

- **新增** `docs/rmbench_audit_report.md`（本文）：审计报告与路线。
- **未修改任何训练/推理源文件**（所有修复都只给出补丁建议，见 §4/§6）。
- **复用并重跑**了既有审计脚本 `script/audit_rmbench_cpu_checks.py`（输出刷新 `docs/rmbench_cpu_checks.json`，本轮重跑确认 180 episode / 0 issue / 缓存复现）。
- 历史资产 `docs/rmbench_history_audit_inventory.json`、`docs/MemoryWAM_LingBotVA_implementation_plan.md` 保持不动，作为输入材料。

---

## 10. 留给你决策的问题（无法从现有材料解决、且会改变路线）

1. **颜色序真相**：fullctx 离线 latent 是否确定是 RGB（当前只能证明“当前提取脚本用 rgb24 + 客户端 docstring 说原生 RGB”，但提取脚本未进 git，无法证明提取时版本）。这会改变“rearrange 1.0 是否可信”。建议允许我在有 GPU 时跑一个 ~5 分钟的对照（同一场景分别用 `rgb`/`legacy_bgr` 编码首帧 latent 比数值），或你直接确认 RMBench `head_camera/rgb` 的通道序。
2. **observe_and_pickup 目标**：它 0/5 且 action_loss 高 7 倍。若 rgb 复测后仍 0/5，你希望“先给它加训练步数补基线”，还是“先把它与 blocks 一起并入机制对照、不单独追求其基线达标”？这决定下一步是否要占用一块 GPU 做 observe 加训。
3. **GPU 可用性**：本机当前仅见 1×3090（另有 560MiB 小进程），而训练/闭环需要 3 卡。请确认下一步闭环评测是否有 3 卡可用，否则我先只做离线 forward/解码诊断。
