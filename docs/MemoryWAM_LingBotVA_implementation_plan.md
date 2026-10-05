# MemoryWAM 在 LingBot-VA 上的机制移植与 RMBench 实施方案

审计日期：2026-09-30。本文是源码审计和实施设计，**不是已完成的实现或实验报告**。本轮只新增本文；没有启动训练、模型推理、仿真、安装依赖或修改源文件。下文验收项均为后续实施要求。

证据标签：**【论文】**表示本地 MemoryWAM PDF 的明确文字、公式或图；**【源码】**表示已读取的实际实现；**【工程假设】**表示为移植选定、但论文没有充分公开的细节。推导结果另标为“推导”。行号对应本次工作区快照，后续修改后应更新。

## 1. 可行性结论与最关键的架构差距

### 1.1 结论与复现边界

**可行，但不是给现有 KV 池加三个标签就能完成。** 当前 LingBot-VA 已具备逐层 self-attention KV、视频/动作嵌入、时间调制、3D RoPE、训练期 FlexAttention 和流式 VAE，可以承载 MemoryWAM 的混合视觉记忆。缺少的是可训练 gist token、与逐帧缓存等价的混合可见性规则、按真实观测提交的缓存生命周期，以及训练数据的连续历史契约。

必须区分两个交付目标：

1. **机制移植/最小验证**：保留 LingBot-VA 的共享 Transformer、现有 30 维输出和本地 16 维 EEF 适配，加入逐帧 gist、初始 anchor、recent 的视频记忆，并用训练/推理一致的可见性学习。先通过离线逐帧 reference，再支持 clean-video-prefill + action-only 去噪。这可以验证 MemoryWAM 的记忆机制，不能宣称复现论文的 6B 架构、延迟或成功率。
2. **完整论文复现目标**：在同一训练/服务入口下，增加独立的 1024 宽动作专家、14 维关节动作与 proprioception、先拼 RGB 再 VAE 的输入、论文训练噪声与任务协议。论文未公开的实现细节必须固定并报告；在作者代码缺失时，只能称“依据论文的复现”，不能保证逐位等价。

| 关键差距 | 论文依据 | 当前事实 | 必要处理 |
| --- | --- | --- | --- |
| gist 是训练出的压缩表示 | §3.3、§4.1、图3、附录 A.1 | 无 gist 参数、token 类型或输出 KV 选择接口 | 每帧插入 M 个可学习 token，用混合 mask 训练；保留每层 gist KV |
| 双专家与共享骨干不同 | §3.2、§4.1、图2 | `WanTransformer3DModel.blocks` 只有一套 3072 宽骨干，视频/动作有不同输入输出与 time embedder | 机制验证复用共享骨干；完整复现独立增加动作专家，不在第一阶段拆整个项目 |
| 推理不生成未来视频 | 式(2)(3)、A.1 | `_infer` 先视频去噪，再动作去噪；动作读到预测视频 KV | 训练时先解除动作对未来 clean/predicted video 的依赖，再添加一次真实视频 prefill 的推理路径 |
| 三类记忆是有类型的视觉 KV | 式(4)(7)、A.1 | `attn1` 缓存把视频和动作混存，仅记录写入编号、有效位和预测位 | 单独管理视频/gist；动作历史做明确可选分支，不能默认整池保留 |
| 逐帧因果而非随机 chunk 因果 | 图3、§4.1 | 训练随机 `chunk_size=1..4`，同 chunk 视频内部互见 | 用全局观测时刻和 token 类型建 mask，默认一帧一个观测单元 |
| 长历史训练 | §4.1 每 episode 自回归序列 | 普通 dataset 以 `action_config` 为独立样本，随机采样；bounded sampler 也不连续 | 保留 episode 标识、时间映射；前缀训练作为 reference，再考虑顺序 chunk/TBPTT |
| RMBench 表示不同 | §4.1 | 论文关节14D；本地 EEF16D→模型30D；相机分别 VAE 后拼 latent | 独立版本化数据/动作契约，避免用历史实验配置冒充论文配置 |

**三项不能误读的结论：**

- gist 不是全局固定数量的 RNN 槽位，也不是平均池化后的 latent；每帧都有 M 个 gist 的逐层 KV，附录 A.1 明确“每个历史帧的 context tokens 永不淘汰”。总缓存仍为 O(NM)，不是 O(1)。
- anchor 实现是任务最初的 `Ninit=2` 个 clean latent 帧，没有公开在线事件检测器、关键帧打分器或周期性重选算法。
- 原 checkpoint 没有学过 gist mask/压缩瓶颈；仅推理时加随机 gist、剪 KV 或关闭视频生成，没有论文依据可视为有效复现。

### 1.2 本轮材料与版本边界

本地主要材料：

- [MemoryWAM PDF](<./Yang 等 - 2026 - MemoryWAM Efficient World Action Modeling with Persistent Memory.pdf>)，15页，arXiv:2606.20562v1，2026-06-18；附录 A.1 在 PDF 第14页，已阅读。SHA256：`a2beffa47141bcc8d5593d0eb50be681ee286a75f543848cef86e7f56d310c8a`。图3已渲染核对，不能只靠 PDF 文本抽取推断色块。
- [现有 attention 阅读说明](./attention_code_reading.md)，只作导航；已逐项对照源码。例如文中曾写 `WanVAServer`，真实类名是 `VA_Server`。
- [AGENTS.md](../AGENTS.md)、[README.md](../README.md)、[INSTALL.md](../INSTALL.md)、[RMBench 本地脚本说明](../script/README_RMBENCH.md)，以及配置、两条正式模块启动脚本。子目录中未发现附加 AGENTS.md。没有采用 INSTALL 中未核实的 `tests/test.sh` 作为测试入口。
- [作者项目页](https://yangsizhe.github.io/MemoryWAM/)与本地论文的单次视频前向、混合 KV 说明一致；本次读取未发现可供实现核对的官方代码入口。没有用其他 memory 工作代替 MemoryWAM。
- [RMBench 官方仓库](https://github.com/RoboTwin-Platform/RMBench)：用于确认是 RoboTwin 系列的记忆依赖操作 benchmark；官方当前说明包含 `demo_clean` 和图像解码格式变化。具体本机行为以下面本地 adapter 的代码为准，不假定与在线主分支一致。

【源码】工作区 `HEAD=d986be8`，远端配置为 `https://github.com/Robbyant/lingbot-va`；**本地保存的** `origin/main=7c6ffa9`，没有 fetch，不能把它说成截至今天的最新上游。`63b1ba9`、`d986be8` 含本地 RMBench 适配、FSDP offload、训练诊断等变化。相对 `7c6ffa9`，`model.py` 主要是注释和局部变量改名，attention/cache 主机制没有变成 MemoryWAM。

开始审计时未提交内容：`.gitignore` 已修改，`AGENTS.md` 和 `script/debug/` 未跟踪；`docs/`、`script/rmbench_data/`、`script/rmbench_train/`、`script/rmbench_inference/` 被现有规则忽略。**本文虽已写到 docs，但不会自然出现在普通 git status 中；本轮不修改忽略规则，也不提交。**

本地启动脚本实际调用的外部 RMBench 位于 `/root/wyc/code/RMBench`，`HEAD=6d36e28`，其环境、评测脚本、LingBot adapter 有未提交改动；这里只读审计被调用的 adapter，没有修改外部仓库，也未将其当成官方默认。

## 2. 论文机制与当前源码的对应关系

### 2.1 三类记忆的完整定义

以下的“帧”以模型的 **clean latent 帧**为记忆单位；不是一个 RGB 原始帧，也不是一次 diffusion step。`L` 是一帧 patch 后的视频 token 数，论文为120；`M` 是每帧 gist 数，论文为8。

| 项目 | gist | anchor | recent |
| --- | --- | --- | --- |
| 保存的信息与来源 | 【论文 §3.3】每帧附加 learnable tokens，读取本帧视觉 token 及历史上下文，形成压缩历史 | 【论文 §3.3】任务起始状态的完整视觉 token 表示 | 【论文 §3.3】近期交互细节的完整视觉 token 表示 |
| 表示层次 | 输入是可学习 token embedding；层间是隐藏状态；跨步保存其**每层 self-attention K/V**，不是只存最终层向量（A.1） | 每层 clean video K/V（A.1） | 每层 clean video K/V（A.1） |
| 初始化 | 【论文】可学习参数；【工程假设】一个共享 `[M,Dv]` 表逐帧复制，标准小随机初始化；论文未给初始化分布及共享粒度代码 | episode 从空缓存开始，前2个真实 clean 帧到达时自然填充，不提前伪造第二帧 | 从空开始，随真实观测到达填充，少于4帧时使用已有帧 |
| 生成/更新 | 本帧视频和 gist 一起逐层前向，每层生成 KV；不是在帧被淘汰时才临时压缩（§3.3、A.1） | 观测第一次 prefill 产生；后续读取已有 KV | 观测第一次 prefill 产生；近期集合随帧推进 |
| 保留与淘汰 | **每帧8个，永不淘汰**；并非只保留“已经变老”的帧（A.1） | 前2帧完整 KV 持续保留至 reset（§4.1、A.1） | 最近4个 clean 帧完整 KV；离开窗口且不是 anchor 才删完整视频 KV（A.1） |
| 参与层/分支 | 【论文 图2、A.1】每个 transformer block 的视觉记忆；视频 self-attention 形成，动作专家的 attention 读取 | 同左 | 同左 |
| 参数与训练 | 有新增参数；通过视频+动作 flow loss 学习。论文没有专门 gist 重建/对比/蒸馏 loss，没有独立压缩网络的规定 | 没有单独可学习选择器；KV 的生成权重随主模型训练 | 没有单独可学习选择器 |
| 位置 | 【论文 §4.1】与所属帧共用3D RoPE时间坐标，h/w固定 marker；具体数值未给 | 所属原始 latent 时间和空间网格 | 同左，不随滑窗重编号 |
| 尚不明确 | embedding 的精确初始化、是否另有类型 embedding、gist 的 time modulation、层内实现与训练增强复用细节 | 多次任务指令/子任务切换是否重新定义 anchor 未给 | 边界实现、不同长度 batch 存储方式未给；图3支持“最近窗口包含当前帧” |

论文没有说明仅挑选部分层使用记忆，因此默认 **30层都使用**。图2中 text/proprio 走条件 cross-attention；混合视觉记忆参与视频/动作间的联合 attention，不应另造一个文本 memory 分支。最小移植中把视觉 KV 接到现有 `attn1`，现有 `attn2` 继续读文本；独立动作专家阶段使动作 Q 与视觉 K/V 在相同 `24×128` attention 空间相遇。

**层语义的重要限制（由 Transformer 计算顺序推导）：**在现有 `WanAttention.forward` 中，KV 来自本层 attention **之前**的隐藏状态。因此第一层 gist KV 来源仍主要是可学习输入及 time modulation，尚未读取图像；第二层及更深层的 gist KV 才携带先前层聚合的视觉信息。这不应被“修正”为取层后 gist 再伪装成本层 KV。若额外再跑一次压缩器，便改变了论文图2/图3可支持的普通 Transformer 路径，必须单列为变体。

### 2.2 组合、预算、重叠与因果 mask

记零起始观测帧为 `i`，当前已观测到 `t`：

```text
A(t) = {i | 0 <= i <= t and i < Ninit}
R(t) = {i | max(0, t-Nrecent+1) <= i <= t}
G(t) = {g_i | 0 <= i <= t}
C(t) = video_KV[A(t) ∪ R(t)] ∪ gist_KV[G(t)]
```

【论文】式(4)(7)是混合缓存的并集；A.1确定所有帧的 gist 都保留。**完整视频 token 在 anchor/recent 中重合时只保存一份**，不能重复拼入 softmax；同一帧的 gist 和完整视觉 token 则是不同 token，允许并存。gist 内含本帧和历史信息，会与 anchor/recent 内容重叠，论文没有去冗余损失、互斥分配或覆盖写回机制。

以图3明确的 `Ninit=1, Nrecent=1` 为例，可见性为：

| Query | 可见 clean/gist keys | 可见 action keys |
| --- | --- | --- |
| `f0, g0` | `f0, g0` | 无 |
| `a0` | `f0, g0` | 本组 `a0` |
| `f1, g1` | `f0, g0, f1, g1` | 无 |
| `a1` | `f0, g0, f1, g1` | 本组 `a1` |
| `f2, g2` | `f0, g0, g1, f2, g2`；**没有 f1** | 无 |
| `a2` | 同上一行 | 本组 `a2` |

这证明本帧 `f_i↔g_i` 可以互见，gist 不只是单向读取；同一帧空间 token、gist 内部双向不会泄漏未来，只要它们完全由**当前已获得的观测**构成。历史 clean/gist query 不能看未来；同帧 f/g 不能看正在去噪的动作；普通任务也不能读取历史动作（Swap T 例外见 A.1）。噪声动作之间只允许同一目标 chunk 内的联合去噪可见性。

【工程假设，边界选择】将 recent=4解释为包含当前帧，即 action query `a_t` 读 `t-3..t`；图3提供依据。prefill `f_t/g_t` 时也用相同的、相对 query 时间定义的集合，不能在整段训练中统一只保留“episode末尾4帧”。否则早期 query 的历史被错误裁掉，甚至泄漏未来。

【论文】§4.1规定训练可见性复现推理缓存，图3省略 noisy video 并称其与动作读相同历史；但未公开完整的 clean/noisy video 的物理帧移位与 mask 代码。**不能把训练目标 `z_{t+1}` 的 clean token/gist 暴露给预测 `a_t` 的 query。** 推荐以决策时刻为显式索引：在 `C(t)` 上预测下一段动作和下一观测视频，二者的 noisy query 各自 self-attend，只读 `C(t)`，不读目标的 clean `f_{t+1}/g_{t+1}`。此映射是符合式(2)(3)和闭环因果性的工程选择，不声称论文公开了完全相同的张量排列。

预算可直接用论文 `Ninit=2, Nrecent=4, M=8`，不需要用户凭经验指定历史实验的 `w193` 或 `h32_f64`。APF=16成立时，recent 覆盖4个 latent 时刻，最早到最晚的时间差是48个动作步；这些 latent 的因果感受野还包含更早输入，不能简单称“只含最近64张RGB”。秒数需要实际控制周期，不能从 MP4 fps、仿真物理步长或旧注释推定。anchor 第二帧通常对应执行首段16步后取得的观测，不是两张连续原始图像。

### 2.3 现有模型、attention 与位置编码

【源码】已读 [实际 transformer 配置](../checkpoints/base/transformer/config.json)：30 blocks，24 heads，head_dim=128，`D=3072`，FFN=14336，latent channels=48，patch=(1,2,2)，action_dim=30，text_dim=4096。不是仅引用 Python 默认值。[VAE 配置](../checkpoints/base/vae/config.json)给出时域压缩4、空间压缩16、latent48。

| 实际位置 | 已实现的机制与限制 |
| --- | --- |
| `wan_va/modules/model.py:607` `WanTransformer3DModel.__init__` | 一套 `blocks`；视频 patch MLP、动作 linear、两个 time embedder；没有独立 action DiT、gist、proprio projector |
| `model.py:680` `_input_embed`、`:721` `forward_train` | 视频 `[B,48,F,Hl,Wl]→[B,FL,D]`，动作 `[B,30,F,P,1]→[B,FP,D]`，文本 `[B,512,4096]→[B,512,D]` |
| `model.py:740` `forward_train` | 物理拼接顺序是 `[noisy_video, condition_video, noisy_action, condition_action, padding]`，不是内存里逐帧 interleave；batch 被打平成外层B=1的长序列。mask才实现模态/时间顺序 |
| `model.py:95` `FlexAttnFunc.init_mask` | 每 token 的 seq_id隔离样本；视频块号 `2*floor(f/chunk_size)`，动作块号再+1；noise_id 0=噪声、1=条件；padding=-1。内部变量名称不能代替实际顺序判断 |
| `model.py:154` `_get_mask_mod` | 条件→条件 `k_block<=q_block`；噪声→条件严格更早；噪声→噪声同块；再相交绝对块差窗口。当前不是 MemoryWAM 的帧级 hybrid mask |
| `model.py:37` `custom_sdpa`、`:304` `WanAttention.__init__` | 推理 torch/flash 无显式 mask，使用池中所有有效 key；支持机制所需数学运算，但API还不支持带类型、逐query可见性的缓存 |
| `model.py:243` `WanRotaryPosEmbed`、`:435` `WanAttention.forward` | 三轴RoPE；head128分为时间44、高42、宽42；Q/K旋转，V不旋转；缓存保存的是已经旋转的K |
| `wan_va/utils/utils.py:33` `get_mesh_id` | 返回 `[4,S]`；前3行为f/h/w，第4行模态标记不被RoPE使用；动作时间 `f+(j+1)/(P+1)`，h=w=-1。训练从0起，推理用frame_st_id。不能把第四行当可用的类型embedding |
| `model.py:279` `WanRotaryPosEmbed.forward` | 直接用grid乘频率；`rope_max_seq_len=1024`不是实际索引表上限。超长时应测精度与泛化，不能擅自取模或滑窗归零 |
| `model.py:471` `WanTransformerBlock.forward` | time embedding调制 self-attention和FFN；文本在每层attn2进入，之后层的视觉KV自然依赖文本条件 |
| `model.py:46` `FlexAttnFunc` | mask为类变量；activation checkpoint重算期间若被另一个forward覆盖会用错mask，顺序chunk训练需改为调用级显式参数 |

RoPE推荐：保留全局 latent frame_id，gist置 `(frame_id,-2,-2)`，动作暂沿用当前小数时间和`(-1,-1)`；`-2`只是**工程假设**，论文只规定固定marker而没给数值。完整复现动作专家复用视频RoPE频率。缓存压缩仅删除项，不能改变已存K的时间相位；若需整体重定位，必须同时明确变换所有K与Q，不能只改元数据。

在新的决策索引下，推荐动作query坐标为`t+(j+1)/(P+1)`，其中t是已观测到的决策时刻，虽然它的监督标签存于`A_store[t+1]`。下一视频query的目标坐标为t+1，但可见性cutoff仍为t。**位置坐标、label数组下标、可见性cutoff是三个不同字段**，不能相互替代。

### 2.4 已有缓存并非可直接复用的 MemoryWAM 缓存

【源码】`WanAttention.attn_caches`（`model.py:333`）由**每层 attn1 自己持有**；attn2没有KV缓存。`VA_Server._reset`（`:396`）触发模型级 `create_empty_cache`（`model.py:670`）。每层有 `[B,S,24,128]` 的K/V以及一维 `[S]` 的`id/mask/is_pred`；`id`是每次写入的编号，不是环境时刻，也没有episode/modality/gist标识。

`WanAttention.forward:446` **不论 update_cache 的值都先写池**：0=写入后撤销本次slots；1=留下且标prediction；2=留下且标非prediction。`allocate_slots:368`空间不足按写入id淘汰最早token，可能切开一帧；`restore_cache:413`只清当前slots的有效位，**不会恢复为本次临时写入让位的旧历史**。因此 `update_cache=0`并不保证整个持久缓存只读；这是推理去噪次数可能间接影响历史的风险，需用满池测试证实边界，不能将它作为事务回滚接口。

`_compute_kv_cache:592`先清prediction，再写真实视频和 `obs['state']` 的动作KV，推进frame_st_id；`_infer:463`多次去噪本身不推进环境frame_st_id。`_reset`同时清理Transformer和两个VAE流式卷积缓存。VAE的 `feat_cache`（`modules/utils.py:67`）是另一种历史状态，不是attention KV。

缓存可复用性：

| 状态 | 何时可复用 | 何时不能直接复用 |
| --- | --- | --- |
| 历史clean video/gist逐层KV | 同episode、同模型版本、同条件分支、同输入预处理与坐标；它生成时没有读未来/噪声action；历史可见性被冻结为因果前缀 | 换prompt/CFG分支、换权重/训练step、改相机或归一化、改RoPE规则；多帧双向prefill使旧帧曾读取较新帧 |
| 当前denoising action/video KV | 同一个前向内部可用于self-attention | 下一个噪声时刻、下一个输入或条件不一致时；“最后一步”也不能当真实观测提交 |
| anchor/recent转为gist-only | 保留当时已经产生的gist KV，淘汰完整video KV | 不能从最终video hidden随便投影一个gist，也不能修改旧gist后不更新受其影响的后续层/帧 |
| 文本encoder embedding | prompt、encoder、预处理不变时 | prompt变化；每层attn2投影权重训练后缓存的text K/V也过期 |
| VAE feat_cache / 预提取latent | 因果编码、同一episode连续帧、相同采样/拼图和权重 | 任意随机窗口直接继承上个样本缓存；换RGB拼图方式；编码时读了未来；重复prefill相同RGB推进了卷积状态 |
| 历史动作（Swap T变体） | 只能是已经执行且可观测的动作历史，明确固定编码时刻 | 本地client回传的整段计划包含未执行部分；不能当当前proprio，也不能默认所有任务使用 |

同一帧 f/g 内部双向是安全的；将两帧同时无mask prefill则不保证逐帧等价。最小实现默认每次prefill一帧；多帧合并优化必须使用相同图3规则，不能以“它们现在都已经观测到”为理由破坏训练时历史KV的定义。

## 3. 训练和推理的数据流、记忆生命周期

### 3.1 实际入口与上游、本地行为的区分

| 范围 | 实际入口/配置 | 审计结论 |
| --- | --- | --- |
| LingBot训练入口 | `script/run_va_posttrain.sh:34` → `python -m torch.distributed.run ... -m wan_va.train` → `run:729` → `Trainer.train:586` | 默认robotwin_train；没有运行该脚本 |
| LingBot推理入口 | `script/run_launch_va_server_sync.sh:29` → `-m wan_va.wan_va_server` → `run:711` → `VA_Server` | 实际服务启动取决于所选配置的infer_mode；RMBench配置明确为server |
| Attention模式 | `train.py:98` 显式传flex；`wan_va_server.py:99`显式传torch；`modules/utils.py:41`透传到from_pretrained | README要求手改checkpoint config，但**当前实际调用已有覆盖**，应记录最终effective config，不为新增记忆修改原checkpoint目录 |
| 普通训练配置 | `va_robotwin_train_cfg.py:18`及继承的`va_robotwin_cfg.py:11` | lr=1e-5、wd=.1、batch1；推理chunk2、APF16、attn_window72、video25步/action50步是该配置值，不是全部任务统一标准 |
| 本地RMBench训练 | `va_rmbench_train_cfg.py`、`script/rmbench_train/train_rmbench_50demo.sh:23` | 配置默认指向w193数据；脚本通过环境变量改成fullctx_apf16_cf8，3个FSDP rank、独立5条validation早停。这是本地实验 |
| 本地bounded训练 | `script/rmbench_train/train_blocks_bounded.sh:45` → 同一Trainer | 设置bounded索引；读取的manifest为max64/history32。不是MemoryWAM的recent4/anchor2；不能沿用其数字作为论文预算 |
| 本地RMBench闭环 | `script/rmbench_inference/run_rearrange_50demo_inference.sh:100`启动服务；`:118`调用外部RMBench `script/eval_policy.py` | `faithful`只代表该脚本的LingBot视频25/action50配置；并不代表MemoryWAM。`fast`8/16步也是本地选项 |

当前只检查了实际代码、配置和部分数据元数据，不宣称检查了所有训练产物的有效配置或所有episode的数值正确性。已读的full-context Blocks audit文件记录50 episodes/150 latent files PASS，但这是**已有审计记录**，不是本轮重跑结果。

### 3.2 数据组织、模态形状与时间对应

【源码】`LatentLeRobotDataset.parse_meta:175`将每个episode的`action_config`逐段展开；latent路径含episode、start、end。`__getitem__:387`内部知道episode与源帧，但最终只返回`latents/text_emb/actions/actions_mask`，没有把episode身份/时间索引交给Trainer。窗口内部按时间排列，**相邻样本的顺序不保留**（`Trainer.__init__:134`随机sampler/shuffle）；没有训练态跨窗口KV。

`BoundedFullContextDataset.__getitem__:36`从整段latent切片，并返回episode_index、global_latent_start、history_frame_count、validation_weight；`EpisodeBalancedDistributedSampler.__iter__:81`是随机选episode再随机窗口，不是顺序流。其episode整数还被当作`MultiLatentLeRobotDataset`的样本索引，仅在“每episode恰好一个action_config、索引一致”的本地契约下成立，不适合直接泛化多数据集。

**本地实现缺口：**`Trainer._prepare_input_dict:293`没有读取上述global/history字段，grid从0重建；`compute_loss:381`没有屏蔽history区loss；配置中的`bounded_validation_index`没有用于构造验证集，`Trainer.__init__:167`仍用完整`MultiLatentLeRobotDataset`。因此“bounded有32历史帧”不能解释成模型已具有带全球位置和无loss warmup的跨窗口记忆。此处是静态调用链结论，没有运行故障复现。

当前RMBench观测与动作契约：

| 层次 | 实际数据/形状 | 来源与含义 |
| --- | --- | --- |
| 原始相机 | head、left/right wrist | 转换脚本`CAMERA_MAP`；实际采样频率须读数据，不能从历史脚本240Hz/60Hz注释推断 |
| 视频预处理 | head256×320，wrist各128×160 | `VA_Server._encode_obs:344`；三相机**分别编码**后拼latent，训练`_cat_video_latents:247`一致 |
| VAE latent | head16×20、wrist各8×10；拼成24×20，48通道 | 根据实际VAE空间stride16与配置推导；head置底，wrist拼上方 |
| 视频token | `[B,F*120,3072]` | patch2×2：`24*20/4=120`；与论文L相同，但**先VAE后拼**不等于论文先拼RGB再VAE |
| 文本 | `[B,512,4096]→[B,512,3072]` | T5预提取；训练按概率cfg_prob=.1换empty embedding；不是CLIP（提取脚本中的CLIP注释与实际load_text_encoder不符） |
| 原始动作 | `[T,16]`：左xyz+xyzw+gripper，右同 | `convert_rmbench_to_lingbot_lerobot.py:243 source_action`读endpose；`:381` image[t]配action[t+1] |
| 模型动作 | `[B,30,F,16,1]→[B,F*16,3072]` | `lerobot_latent_dataset.py:278`；有效16通道映射到`[0..6,28,7..13,29]`；其余填0并mask |
| 条件state | cache请求中的 `[16,F,16]` | 外部 `LingBotClient.update_kv_cache:217`回传`_last_raw_action`，是模型上一段计划，不是论文14D proprio |
| 单层Q/K/V | `[B,S,24,128]` | `WanAttention.forward:428`；训练外层B=1且原样本打包 |

**时间合同必须显式重建。** 本地提取 `extract_rmbench_latents.py:211`按frame_step选源图，VAE首块1帧，此后4的倍数；`frame_ids`记录的是**被选中的RGB源索引**，不是latent索引。若从源帧0开始、frame_step=4：

```text
所选RGB源索引:   0, 4, 8, 12, 16, 20, ...
latent终点索引:  z0@0, z1@16, z2@32, ...  （各z有因果感受野）
存储动作组:      A0=dummy且不计loss
                 A1=物理动作1..16；A2=17..32；...
决策时刻t:       只已见 z0..zt，应该预测 A[t+1]，监督视频 z[t+1]
```

这由 `_rmbench_action_post_process:299–348`的首部16步padding和 `frame_stride*4`推导。不要把存储`A[t]`与`z[t]`相同数组下标误当作“看到z[t]再执行A[t]”：后者包含动作执行结束后的图像。现有LingBot先生成未来视频再预测动作，训练给动作未来clean视频是一种该架构的teacher forcing；**一旦改成论文的action-only闭环，继续保留这个映射就会引入推理不可见条件。**

本地fullctx Blocks metadata第0条范围0..1033，对应使用1024个可对齐动作并有初始dummy；不能据此把所有episode长度设为1033。末尾不足完整latent/动作chunk的标签需mask，是否丢弃尾部及成功判定影响要列入数据版本。`observation_anchors.json`的anchor是**动作坐标原点**，不是MemoryWAM视觉anchor。

### 3.3 原有训练与去噪/环境时间轴

原有训练调用链：

```text
随机独立action_config/窗口
  → 预提取视频latent + text + 归一化动作/mask
  → Trainer._add_noise（逐latent帧采样noise timestep）
  → _prepare_input_dict（随机chunk/window）
  → forward_train（四段token，Flex mask，30个共享block）
  → video/action flow velocity输出 → 加权MSE相加 → backward/optimizer
```

【源码】`utils/utils.py:77 sample_timestep_id`是均匀采样timestep **index**；`scheduler.py:33`构造shifted sigma表，`:66`构造特定指数权重，`:111`目标为`noise-sample`；不等于已实现论文的shifted logit-normal采样与权重。`_prepare_input_dict`的条件视频加噪概率是.5，且用后半段index，论文§4.1是视频条件以p=1混合Uniform[0,1]比例Gaussian。数值配置相同的shift5/1不意味着噪声训练方案相同。

当前训练全Transformer可训练（`train.py:118 requires_grad_(True)`）；VAE/T5不在训练图中，因为latent/text已离线编码。`apply_ac`给每个block包装activation checkpoint（`distributed/fsdp.py:11`），preserve_rng_state=False；FSDP2/fully_shard包attn1/attn2/FFN、block和全模型，bf16参数计算、float32 reduce，本地可CPU offload。当前梯度clip=2（`train.py:455`），论文A.1为1。

原有推理调用链：

```text
reset(prompt) → 清Transformer/VAE缓存、frame_st_id=0
infer(初始obs) → 视频若干flow steps + 额外最终前向 → 预测video KV
              → 动作若干flow steps + 最终前向 → 预测action KV
client执行动作，采集关键RGB
compute_kv_cache(obs,state) → 删除prediction KV → 真实观测视频KV + action/state KV
                           → frame_st_id += 新latent帧数
infer → 重复；下一个episode显式reset
```

视频默认25个scheduler step再附加t=0的cache前向，动作默认50步再附加t=0前向；截断`video_exec_step`时最后cache甚至可能来自非零噪声，不能称clean observation。`frame_chunk_size=2`首轮含初始条件和一个未来latent组，client跳过dummy动作组，因此首次执行16步；后续通常执行32步后回传8张关键图。论文A.1每轮固定16动作和4张关键图，**不能只改缓存策略而保留这个隐式首轮/后续差别**。

### 3.4 建议的训练方式及其等价性

| 训练方式 | 可见性是否满足论文 | 梯度与代价 | 推荐用途 |
| --- | --- | --- | --- |
| 完整episode或完整因果前缀，混合mask并行训练 | 可以；每个query按其决策时刻只读对应历史，clean/gist路径与在线prefill一致 | 对全部相关历史可反传；完整激活可能很大，省推理KV不等于训练显存常数 | **首个正确性reference**；最接近§4.1所述per-episode序列 |
| 顺序chunk携带KV，固定权重，不detach直到序列loss | 若显式mask、时间/文本/增强固定，可与完整序列前向和梯度等价（算子误差除外） | 图仍跨chunk，不能靠拆chunk消除长图；权重step前要完成所需backward | 小规模等价性测试 |
| 顺序chunk + TBPTT，边界detach | 前向可见性可等价；梯度不等价，后续loss不能教更早帧如何生成gist | 减少保留图长度；持久KV仍随gist数增长 | 后续规模化工程近似，必须披露并测TBPTT跨度消融 |
| 独立随机窗口，无前缀重放 | **不满足跨窗口长期记忆**；中途开始会缺真实anchor/旧gist | 开销低，但网络没有学习更远检索；每窗口最初2帧不应伪装episode anchor | 仅机制单元/短程对照 |
| 随机目标窗口 + 从episode起点重放真实前缀 | 前向可满足；若前缀no_grad则相当于截断梯度 | 每次重放有开销；不会把前一个随机样本的KV错给当前样本 | 无法承受完整反传时优先于直接无历史窗口 |

**新增记忆与optimizer的版本问题：**训练step更新后，旧KV是旧权重的输出，即使detach也不解决陈旧权重问题。推荐先不跨optimizer step携带状态：一个episode/选定前缀处理完、loss backward后step，再丢弃训练memory；TBPTT可在episode内分段backward并detach、但整条处理完才step。若为了频繁更新选择跨step保留旧KV，应明确是额外近似；另一选择是每个step从原始已见前缀用当前权重重建。论文没有公开TBPTT或stale-cache训练策略。

每个序列的条件文本、CFG drop分支和clean augmentation在一次训练展开内保持稳定，历史KV生成后不随后续chunk重新随机改条件。随机增强在checkpointed block之外完成；block内部不写持久池、不采新随机噪声。前缀一旦被detach，仍可作为Q/K/V计算的常量供当前层查询，但不再声称梯度穿过了其压缩器。

### 3.5 建议的环境生命周期与RMBench边界

建议的主推理路径遵循式(2)(3)、A.1：

```text
episode reset（空视频KV、空VAE状态、文本/条件签名）
  → 编码初始真实观测z0
  → clean f0 + g0 一次prefill，所有层成功后原子提交
  → 固定C(0)，对下一段16动作执行50步flow去噪；始终只读memory
  → 执行16动作，收集子步3/7/11/15后的4张RGB与当前本体状态
  → 因果VAE得到z1 → clean f1+g1一次prefill → 提交C(1)
  → 淘汰离开recent且非anchor的完整video KV，保留所有gist
  → 继续；done/truncated/reset清理所有状态
```

【工程假设】session owner持有memory；每条状态键是`(session_id, episode_uid, cfg_branch, condition_version, model_version)`，token唯一键再含`frame_id/type/slot_in_frame/layer`。prefill提交键用真实`observation_id`或frame范围，**先检查幂等再编码VAE**。重复相同消息返回已提交版本，不写第二份；冲突payload或非连续时序报错，不默默当新帧。

去噪不调用commit，不分配持久slots；当前动作KV只存在于本次layer计算。既不把50次前向计为50帧，也不因为重试infer把同一观测写50遍。视频prefill在动作生成前提交的是已经见到的观测，不需要等动作成功；未来观测必须等待真实执行结果到达。预测视频（如另开诊断）始终隔离为临时分支。

【源码】外部真实客户端 `policy/LingBot_VA/lingbot_client.py:165`返回EEF动作；`:217`的state是整段`_last_raw_action`。`deploy_policy.py:200 eval`会在`:301`用`action_type='ee'`执行，`:329`每4步采一帧，`:350`完成且未终止才更新缓存；还含本地window reset和stall选项。MemoryWAM需显式新协议，不能用“state”一个名称混淆当前proprio、已执行动作和预测计划。

最小机制阶段保留EEF与现有归一化，但更改为明确的16步chunk response并去掉首轮dummy暴露；client按协商的horizon执行，不再硬编码首次跳过一组。完整复现阶段读取14维关节state/action，设置真实关节控制模式并验证上下限、单位、左右臂顺序和gripper；**禁止将14维关节结果送进现有 `_delta_to_absolute` EEF转换**。当前转换器未保存proprio，必须回到RMBench原始数据重新构建，不能用未来目标动作代替当前state。

RGB层面，论文§4.1为384×320拼图**共同VAE编码**；当前分别VAE的latent不能直接拼接后声称完全一致。推荐机制阶段保持旧输入建立对照，完整复现用独立数据版本重提取，比较离线与流式编码的一致性。VAE运行仍冻结；预提取按episode从头建立因果状态，而非每个随机窗口都重置后当作连续latent。

不得输入RMBench特权任务状态：本地adapter的`capture_button_state`/trace可以用于调试验收，不能作为gist输入、anchor选择、动作条件或终止判定的额外捷径。任务阶段文本若来源于专家执行轨迹而部署时不可见，也不能用于动态选择训练窗口指令；采用rollout起点可得的整体任务指令。

结束/截断策略：真正环境done、时间上限终止、session断开或重新reset时清所有KV/VAE/动作队列/坐标原点；数据加载的chunk边界不是episode结束。若中途只执行了k<16步而环境继续，不能把16步整计划写成历史；未凑满4张采样图时保留待编码RGB缓冲并等待/显式走短chunk协议，禁止重复最后帧后按完整16步推进时间。若此时episode结束，直接丢弃待提交状态，不伪造完成观测。

## 4. 具体文件及函数修改清单

下表是**实施建议，尚未修改**。标“现有”的路径/类/函数已经核对；新函数、新文件均明确标注。相对路径均相对本仓库，外部适配器用绝对路径。`S`=memory token数，`Q`=当前query数，`P`=每latent动作步数。

| 实际文件路径 | 类/函数 | 当前行为 | 建议修改 | 输入/输出及形状变化 | 修改原因 | 依赖与风险 |
| --- | --- | --- | --- | --- | --- | --- |
| **现有** `wan_va/configs/shared_config.py`；`va_rmbench_train_cfg.py:9`；`va_rmbench_cfg.py:7`；`configs/__init__.py` | EasyDict配置、VA_CONFIGS | RMBench继承Robotwin；含环境覆盖，未版本化memory | 默认memory关闭；注册专用配置，不改现有任务默认；启动打印effective config | 增加enabled、M/A/R、mode、action_history、train_history_mode、data_contract、RoPE marker、版本与max_episode_frames | 隔离原行为与论文实验 | backend由入口覆盖；配置文件名和实际解析值可能不同 |
| **建议新增** `wan_va/configs/va_rmbench_memory_cfg.py`、`va_rmbench_memory_train_cfg.py` | 建议配置对象 | 不存在 | 明确区分shared-mechanism与paper-MoT两种profile；记录数据预处理、动作表示、噪声方案 | 主配置新增字段，不改变旧张量 | 预算来自§4.1/A.1 | 不隐式继承旧w193、chunk2、历史动作开关 |
| **建议新增** `wan_va/modules/memory.py` | 建议 `MemoryState/MemoryDelta/TokenMeta/MemoryOwner` | 不存在；旧KV分散在attn1 | 状态外置，按episode/branch分片；read-only view、分层delta、整次prefill原子commit/reset、幂等校验；anchor/recent去重 | 每层K/V `[B,S,24,128]`或逐stream列表；meta `[B,S]`/ragged；新增frame/type/uid/version | A.1三类逐层KV，避免旧池误淘汰 | train不得用in-place池保存带图tensor；CFG不共享conditioned KV |
| **现有** `wan_va/modules/model.py:607` | `WanTransformer3DModel.__init__` | 无gist | enabled时增加注册参数`video_gist_tokens`；建议按帧共享 | 新参数 `[M,3072]`；输入追加 `[B,F*M,3072]` | §3.3的learnable gist | 必须在FSDP、optimizer构建之前注册；原checkpoint缺该key |
| **现有** `model.py:680,704,721,819` | `_input_embed/_time_embed/forward_train/forward`；**建议新增** `forward_hybrid` | 固定四段布局、两种mode；输出视频/动作 | 保留legacy分支；memory分支显式接TokenMeta、memory view、mode；gist从embedding直入不走patch MLP；输出仅有监督query及可选delta | 序列增加F*M；返回`prediction, optional MemoryDelta`；legacy返回完全不变 | 插入gist并保留token→帧映射 | gist不得送proj_out/VAE decoder或算普通视频MSE；packing offsets必须重算 |
| **现有** `model.py:471,518` | `WanTransformerBlock.forward` | 内部attn1可改缓存，attn2读text | 新分支显式传layer memory、mask及meta，返回当前层的观测/gist KV；禁止持久写入 | 当前hidden `[B,Q,D]`；历史K/V作为旁路输入，不回填hidden序列 | 纯函数有利于checkpoint重算 | 原层KV来自pre-attention hidden，不取层后hidden冒充 |
| **现有** `model.py:291,416` | `WanAttention.forward`；旧allocate/update/restore | 每次前向临时写持久池，可能提前evict | memory分支用`cat(selected_history, current_KV)`或分段kernel，不写owner；返回staged current_KV；旧接口保留仅legacy使用 | Q `[B,Q,H,d]`；K/V `[B,S_selected+Q,H,d]`；mask可矩形 | 同一去噪多次读取不能改变历史 | 第一步先支持torch显式mask reference；优化flash需确认支持相同mask，不静默丢mask |
| **现有** `model.py:42,95,154`；**建议新增** `wan_va/modules/memory_mask.py` | `FlexAttnFunc`；建议 `build_memory_mask/visible` | 类变量mask、原chunk因果和window | memory路径按episode_uid/frame/type/decision_cutoff构造；Flex mask作为forward参数；padding全拒绝 | 元数据由四段扩成typed layout；Q/K长度可不同 | 图3与推理保留集合一致 | activation checkpoint时不能读被后续forward覆盖的全局mask |
| **现有** `model.py:243`、`wan_va/utils/utils.py:33` | `WanRotaryPosEmbed.forward/get_mesh_id` | 视频整数/动作小数时间，h/w=-1用于动作 | 新接口接global latent_id；gist固定h/w；保留原freq；动作专家共享同一basis | grid仍 `[B,4,S]`，新增gist坐标；TokenMeta单独携带类型 | §4.1；删除token不改变其余位置 | 不把slot编号当时间；新marker需训练；长度>1024需数值验收 |
| **现有** `wan_va/dataset/lerobot_latent_dataset.py:175,278,387` | `parse_meta/_rmbench_action_post_process/__getitem__` | episode内片段、EEF相对原点、最终丢metadata | enabled路径返回稳定dataset+episode UID、源RGB索引、latent终点、action range、valid_length、is_first/is_last；明确A[t+1]映射 | 增加metadata；paper profile增加当前proprio `[B,F,14]`，动作改14D | 防跨episode和目标泄漏 | 保持旧模式字典兼容；不能对每chunk重置EEF原点 |
| **现有** `wan_va/dataset/bounded_full_context_dataset.py:14,64`；**建议新增** `wan_va/dataset/episode_memory_dataset.py` | `BoundedFullContextDataset/EpisodeBalancedDistributedSampler`；建议 `EpisodeMemoryDataset/SequentialEpisodeSampler` | 随机独立窗口 | 不把旧sampler改成隐式有状态；新增完整前缀/按episode顺序模式，明确burn-in和loss mask | sample新增`context_end,target_start,episode_done`；变长batch需collator/mask | §4.1需要历史上下文 | FSDP各rank前向/反向次数对齐；rank间episode可不同、不能共享memory |
| **现有** `wan_va/train.py:58,222,293,381,433` | `Trainer.__init__/_add_noise/_prepare_input_dict/compute_loss/_train_step` | 随机窗口、无memory、两MSE、全参数训练 | memory模式构造shifted目标、混合mask；前缀无目标loss；训练memory显式生命周期；新增gist进optimizer；单列论文噪声模式 | 模型loss输出仍video/action；metadata不再无差别`.to()`；新增proprio输入可选 | 训练不能仅在推理端补memory | raw tensor准备可no_grad，模型prefill不可整体no_grad；loss归一化避免长episode无意加权 |
| **现有** `wan_va/utils/scheduler.py:33,99`、`utils/utils.py:77` | `FlowMatchScheduler/sample_timestep_id` | uniform index + shifted sigma +现有weight | 保留legacy；专用paper noise sampler与condition augmentation | `[B,F]`噪声系数/权重；新方案独立配置 | §4.1噪声分布不同 | 论文未给logit-normal全部参数和weight精式，不能虚称已精确匹配 |
| **现有** `wan_va/distributed/fsdp.py:11,18`；`distributed/util.py:6` | `apply_ac/shard_model/_configure_model` | 逐block AC、FSDP2；推理requires_grad=False | 机制阶段保持分片架构，新增参数在包装前加入；完整MoT时扩展包装覆盖动作blocks；AC输入含稳定mask/view | 参数新增gist/可选expert；memory是每rank运行态，不是parameter | A.1与训练梯度正确性 | FSDP不会自动分片普通KV；CPU参数offload也不意味着KV会offload |
| **现有** `wan_va/wan_va_server.py:344,396,463,592,627` | `VA_Server._encode_obs/_reset/_infer/_compute_kv_cache/infer` | legacy预测→执行→替换cache，单全局session | enabled路径初始化owner；真实观测prefill一次；action去噪只读；request带uid/id/cutoff/done；原子提交与reset | 新protocol显式horizon及memory_version；动作输出契约可保持EEF16D或新joint14D | 式(2)(3)、A.1 | 不可仅设置video_exec_step=0后照旧调用；训练依赖与首轮dummy都要同步 |
| **现有** `wan_va/modules/utils.py:67` | `WanVAEStreamingWrapper.encode_chunk/clear_cache` | 流式卷积状态，无episode身份 | owner管理其生命周期；重试在encode前去重；失败回滚/重建；paper profile使用RGB mosaic单流 | 独立相机→单mosaic仅paper模式；输出尺寸相同但数值不同 | 训练/推理latent一致且无重复提交 | A.1没有给VAE缓存实现；需检测4帧块与离线编码等价，不能仅检查shape |
| **现有** `wan_va/utils/sever_utils.py:22,42`；`utils/Simple_Remote_Infer/deploy/websocket_policy_server.py:50` | `distributed_infer/worker_loop/WebsocketPolicyServer._handler` | 请求广播各rank；所有连接共用一个policy；断线不清状态 | 最小版限制单活跃session且绑定episode；广播相同reset/commit版本；异常/断连清理；后续再做多session池 | request新增session/episode/request id；所有rank相同状态迁移 | 防串扰及分布式死锁 | 不扩展成通用高并发服务；一个rank拒绝不能让其余rank独自进collective |
| **现有** `wan_va/train.py:495,555`；`wan_va/modules/utils.py:41` | `save_checkpoint/_load_training_state/load_transformer` | 保存bf16权重+model config；optimizer保存和resume调用被注释；KV不保存 | 保存memory schema/参数/数据契约和训练状态；旧checkpoint迁移只容许预定新key缺失；记录new init列表；运行态KV默认不入checkpoint | 新key `video_gist_tokens`；paper模式专家/prop新key；legacy shape不变 | 可复用旧权重且避免静默错误加载 | 正式续训需optimizer/scheduler/RNG/sampler；优先episode边界恢复，不把旧模型KV写入新版本 |
| **现有** `script/rmbench_data/convert_rmbench_to_lingbot_lerobot.py:243,311`、`extract_rmbench_latents.py:125,255` | `source_action/main/encode_camera_segment` | EEF16D、单相机VAE、task/phase窗口 | paper数据新增joint14D+proprio、mosaic编码、完整episode来源索引；新输出目录与schema | latent保持48/F/24/20但数值变；action/proprio14D | §4.1与RMBench可部署性 | 不覆盖旧数据；重新统计仅train split；图像颜色版本不可凭文件名猜 |
| **现有，外部仓库** `/root/wyc/code/RMBench/policy/LingBot_VA/lingbot_client.py:153,165,217`、`deploy_policy.py:180,200` | `LingBotClient.reset/infer/update_kv_cache`、`reset_model/eval` | EEF、首组跳过、整段计划state、4步图像抽样 | 新协议协商16步、一次prefill、执行回执和episode uid；paper profile新增joint控制adapter而非复用EEF转换 | 可执行动作`[16,14]`或机制版`[16,16]`；状态与动作分字段 | 避免错动作空间/未执行状态写入 | 未来修改外部仓库另行形成独立补丁；本轮只审计 |
| **建议新增，完整复现阶段** `wan_va/modules/memorywam_action.py` | 建议 `MemoryWAMActionExpert` | 不存在 | Da=1024、FFN4096、30层、attention内部宽3072；文本+proprio条件；输出14D | action `[B,16,14]→[B,16,1024]`，Q仍24×128，读video KV | §4.1与图2 | 现有block以dim//heads决定head_dim，直接设dim=1024无法得到24×128，需解耦model_dim/attention_dim |

### 4.1 checkpoint复用与训练范围

- **共享骨干机制版**：原视频patch MLP、30层Transformer、视频头、动作30维输入输出、两个time embedder、文本projection、RoPE、VAE/T5参数都可按相同shape加载；新增gist参数必须训练。anchor/recent规则、cache元数据不增加模型参数。建议沿用现有全Transformer微调，VAE/T5冻结，不把“只训gist”当论文默认。
- **完整MoT版**：视频骨干/T5/VAE可复用；论文§4.1的动作专家由视频权重沿hidden维插值初始化，但精确插值轴/方式/归一化未公开，应写迁移脚本并报告。新增动作专家和proprio projection必须训练，旧30D EEF头不能直接等同14D joint头；不要把两者参数shape相近当作动作语义相同。
- 论文A.1还规定任务例外：Swap T使用action history；Observe and Pick Up在RoboTwin预训练，其他任务不使用该额外预训练/动作历史。这里“不预训练”不能解读为不使用Wan视频预训练。用已进行机器人预训练的LingBot base做所有任务属于移植设置，不能冒充论文完全一致的初始化。
- 旧权重加载时`memory.enabled=false`走原类配置/原运算；enabled时采用显式迁移，仅允许gist及已声明的新模块缺失，其他missing/unexpected/shape mismatch立即报告。避免全局`strict=False`吞掉既有层损坏。新权重再关闭记忆只代表新参数下的无记忆消融；**不可能恢复微调前权重本身**，数值回归应比较同一套legacy权重。
- `Trainer.save_checkpoint`当前不提供完整训练续跑；新增记忆后更需要保存optimizer、LR scheduler、step、RNG、数据版本与sampler进度。推荐仅在episode/截断段安全边界保存；恢复时重放前缀重建KV，不序列化带autograd图的memory。推理运行态cache、VAE feat_cache默认不放safetensors。

## 5. 核心接口与伪代码

下文所有新名字均为**建议接口草案**，不是声称仓库已有这些类或函数。核心约束是模型forward纯计算、owner负责提交；不引入额外压缩网络，也不改成标准逐token语言模型解码。

### 5.1 配置和状态草案

```python
@dataclass(frozen=True)
class MemoryConfig:                       # 建议新增
    enabled: bool = False
    mechanism: str = "memorywam_v1"
    gist_per_frame: int = 8               # 论文
    anchor_frames: int = 2                # 论文
    recent_frames: int = 4                # 论文；包含当前帧
    gist_eviction: str = "never"          # 论文
    gist_hw_marker: tuple = (-2, -2)      # 工程假设
    action_history: bool = False         # 论文普通任务；Swap T另配
    expert_layout: str = "shared"        # 移植阶段；paper_mot另配
    training_history: str = "full_prefix"
    tbptt_frames: int | None = None       # 可选工程近似
    max_episode_frames: int | None = None # 容量计划/越界拒绝，不默默丢gist

@dataclass
class TokenMeta:                          # 建议新增
    stream_uid: Tensor                   # 含dataset/episode、CFG branch
    frame_id: Tensor                     # 全局latent时间，非slot/call编号
    kind: Tensor                         # CLEAN_VIDEO/GIST/NOISY_VIDEO/ACTION
    slot_in_frame: Tensor                # patch序号或gist序号
    decision_cutoff: Tensor              # query能够见到的最后观测时间
    valid: Tensor                        # padding/无效帧屏蔽

@dataclass
class LayerKV:                            # 建议新增
    k_rotated: Tensor                    # [B,S,24,128]，ragged可用列表
    v: Tensor                            # 同shape；训练时可以带图
    meta: TokenMeta

@dataclass
class MemoryState:                        # 建议新增；不属于model parameter
    layers: list[LayerKV]                 # 30层
    episode_uid: str
    condition_version: str                # prompt/CFG/预处理/动作契约
    model_version: int
    next_frame_id: int
    version: int                         # 并发/幂等提交用
    committed_observation_ids: set

@dataclass
class MemoryDelta:                        # 建议新增
    layers: list[LayerKV]                 # 仅新增真实f/g；不含噪声预测
    observation_id: str
    base_version: int
    new_frame_id: int

# 建议在WanTransformer3DModel增加；旧forward在enabled=false时不改变
def forward_hybrid(inputs, *, memory_view, token_meta, mode,
                   return_memory_delta=False):
    # mode: prefill_observation / denoise_action / train_packed
    # 不修改memory_view或server owner
    return predictions, optional_delta
```

`MemoryState`以上展示单stream版本；batch时以显式stream-key列表构成batch，不能把batch index当episode identity。一维共享mask/id只适用于同步同长度分支，不能直接套给异步多episode。第一阶段B=1、单session；第二阶段测试ragged batch2，其中一条reset/结束而另一条继续。CFG扩增的B=2是同episode的两个条件分支，不是两个环境。

当前proprio只进入动作分支，不影响视频prefill的条件签名，因此每步正常proprio变化不要求清掉视觉KV。打包训练的cross-attention也需隔离：视频/gist只读本episode文本；动作query读本episode文本及**该决策时刻**的proprio token，不能把整个episode的proprio拼成对所有query可见的公共text context。现有固定512文本seq_id构造需同步扩展。

### 5.2 可见性与逐层prefill

```python
# 工程化表达图3；所有布尔判断还需同stream、同分支且valid
def visible_observation_key(key, cutoff, cfg):
    if key.frame_id > cutoff:
        return False
    if key.kind == GIST:
        return True
    if key.kind == CLEAN_VIDEO:
        anchor = key.frame_id < cfg.anchor_frames
        recent = key.frame_id >= cutoff - cfg.recent_frames + 1
        return anchor or recent
    return False                         # 普通任务不读历史动作/噪声视频

def visible(query, key, cfg):
    if not same_stream_and_valid(query, key):
        return False
    if query.kind in (CLEAN_VIDEO, GIST):
        # 生成本帧gist时始终读本帧f/g，即使做R=0消融也一样。
        # 这是当前计算的局部token；不意味着commit后保留完整video。
        local_fg = (key.frame_id == query.frame_id
                    and key.kind in (CLEAN_VIDEO, GIST))
        return local_fg or visible_observation_key(key, query.frame_id, cfg)
    if query.kind in (NOISY_VIDEO, ACTION):
        return (visible_observation_key(key, query.decision_cutoff, cfg)
                or same_noisy_target_group(query, key))
    return False

def prefill_observation(z_t, text, state, frame_id):
    assert frame_id == state.next_frame_id
    x_video = model.patch_embedding_mlp(patchify(z_t))  # [B,L,D]
    x_gist = model.video_gist_tokens.expand(B, M, D)    # 帧间共享参数
    x = cat([x_video, x_gist], dim=1)
    meta = current_clean_and_gist_meta(frame_id)
    delta_layers = []
    for layer_id, block in enumerate(model.blocks):
        # 包含当前帧后的recent应只有R帧；旧池中的第R个旧帧可能需屏蔽
        history = state.layers[layer_id].select(
            lambda k: visible_observation_key(k, frame_id, cfg))
        # 在block内：time调制 -> QKV -> Q/K RoPE -> attention
        # 本帧所有f/g互见；历史不重新作为query，不重新生成旧KV
        x, current_kv = block.functional_memory_forward(
            x, text, history, meta, timestep=0)
        delta_layers.append(current_kv)               # pre-attn投影KV
    return MemoryDelta(delta_layers, ..., new_frame_id=frame_id)

def commit(state, delta):
    assert delta.base_version == state.version
    assert delta.observation_id not in state.committed_observation_ids
    assert delta.new_frame_id == state.next_frame_id
    assert all_layers_present_and_same_token_ids(delta)
    candidate = append_without_mutating_old_view(state, delta)
    # 所有历史gist保留；完整video仅anchor∪recent；按frame/type去重
    candidate = retain_hybrid(candidate, current_frame=delta.new_frame_id)
    candidate.next_frame_id += 1
    candidate.version += 1
    candidate.committed_observation_ids.add(delta.observation_id)
    return candidate
```

prefill时`functional_memory_forward`返回的是**本层输入**投影后的KV，下一层才使用本层更新后的hidden。上面拼接不是再把整个历史hidden送入Transformer；计算成本只涉及当前query与历史KV。训练期间commit用返回新state的函数式形式；推理可优化为预分配ring存recent、固定anchor区和按页增长gist区，但先通过等价性测试。

`same_noisy_target_group`必须同时检查目标chunk和模态：noisy action只与本动作chunk互见，noisy video只与本视频目标组互见，不能因为target时刻相同就让action读取noisy future video。上述`commit`展示一帧一个事件；若一个请求编码出多帧，则事务内部用`(observation_id, frame_id)`作子事件键，逐帧stage，最终对整个request原子发布并登记幂等键；中途异常不能登记完成。

### 5.3 一次训练迭代：完整前缀优先

完整前缀并行路径最接近论文训练描述，也与现有`forward_train`的Flex设计最接近。它无需跨batch维护真实KV池，但在每层mask中实现等价记忆；不能因为没有物理缓存就认为没有记忆训练。

```python
def train_iteration(batch):
    if not cfg.memory.enabled:
        return legacy_train_step(batch)              # 原运算/RNG路径不变

    # batch内每个sample是单独episode的完整前缀；结束处padding
    raw = collate_episode_prefixes(batch)
    # raw含z[0:T], A_store[0:T], valid，metadata；A_store[0]是dummy
    z_clean = augment_condition_once(raw.z, rng)      # 论文p=1；工程模式可旧方案
    z_noisy, video_targets, tv = flow_noise(raw.z[:, 1:])
    a_noisy, action_targets, ta = flow_noise(raw.A_store[:, 1:])

    # 决策t：target video z[t+1]、action A_store[t+1]，cutoff=t
    # pack可沿用现有分段方式，meta必须与真实token顺序严格一致
    tokens, meta, targets = pack_hybrid_training(
        z_clean=z_clean, gist=model.video_gist_tokens,
        noisy_video=z_noisy, noisy_action=a_noisy,
        global_frame_ids=raw.latent_ids, episode_uids=raw.episode_uids,
        target_decision_cutoffs=raw.latent_ids[:, :-1],
        text=raw.episode_text, proprio=raw.proprio_at_decision,
        valid=raw.valid)
    mask = build_memory_mask(meta, cfg.memory)
    # 模型部分必须保留autograd；没有持久owner，重算是纯函数
    pred = model.forward_hybrid(tokens, token_meta=meta,
                                memory_view=None, mode="train_packed",
                                mask=mask)
    loss = weighted_video_mse(pred, targets, target_valid_mask)
    loss += weighted_action_mse(pred, targets, target_valid_mask)
    # gist、clean条件无独立目标；通过后续query的video/action loss反传
    loss.backward()
    clip_grad_norm_(trainable_parameters, configured_clip)
    optimizer.step()
    optimizer.zero_grad()
    # 本iteration的临时图/状态释放；下一随机episode不继承任何KV
```

raw VAE/T5输出及监督targets是常量；gist参数、当前clean路径/噪声路径各层的计算保留梯度。每个query的meta须写明自己的cutoff，不能只对最终层输出裁未来信息；未来信息一旦进入更早层gist会间接泄漏给后续query。若支持Swap T，新增**已执行**历史动作token及严格时间规则，但不更改上述普通任务默认。

顺序chunk版本的最小reference与TBPTT变体：

```python
def train_episode_sequential(episode, tbptt_frames=None):
    optimizer.zero_grad()
    state = empty_train_state(episode.uid, model_version)
    segment_loss = 0
    for t in range(episode.num_observations - 1):
        # 本episode的augmentation/text drop已一次生成，不在此重随机历史
        delta = model.prefill_with_grad(episode.condition_z[t], state, frame_id=t)
        state = functional_commit(state, delta)       # 保留图，不in-place
        predictions = model.train_targets_with_grad(
            noisy_video=episode.noisy_z[t+1], noisy_action=episode.noisy_A[t+1],
            memory_view=state.read_only(), decision_cutoff=t)
        segment_loss += normalized_loss(predictions, episode.targets[t+1])
        if tbptt_frames and reached_segment_boundary(t, tbptt_frames):
            segment_loss.backward()
            state = tree_detach(state)               # 显式截断跨段梯度
            segment_loss = 0
    if has_loss(segment_loss):
        segment_loss.backward()
    optimizer.step()                                 # episode内权重始终不变
    state = None                                    # 不携带到新权重/新episode
```

这段伪代码表达语义；正式FSDP执行不能让某rank在不同次数的forward/backward中单独退出。episode长度需要bucket、padding或协调的chunk调度，所有rank同次数调用，loss屏蔽padding。长序列不detach的版本只用于小reference，避免保留巨大图；梯度累积的归一化按实际有效目标数固定，而不是每段再独立平均后相加。

### 5.4 一次环境推理：读取、计算、提交三分离

```python
@torch.no_grad()
def on_observation(req):
    owner = sessions.require(req.session_id, req.episode_uid)
    if owner.has_observation(req.observation_id):
        owner.verify_same_payload(req)
        return owner.version                          # 幂等；不再encode VAE
    owner.require_condition_and_model_version(req)
    owner.require_contiguous_time(req)

    # VAE也是有状态的；事务覆盖VAE + 所有Transformer层
    with owner.observation_transaction() as txn:
        z_new = txn.vae.encode_new_observations(req.rgb)
        for z_t, frame_id in ordered_new_latents(z_new):
            delta = model.prefill_observation(
                z_t, owner.text, txn.memory.read_only(), frame_id)
            txn.stage(delta)                          # 已分层计算，尚未对外发布
        txn.commit_all_once(req.observation_id)        # 所有层成功才生效
    return owner.version

@torch.no_grad()
def infer_action(req):
    if not cfg.memory.enabled:
        return legacy_infer(req)
    owner = sessions.require(req.session_id, req.episode_uid)
    if req.has_uncommitted_real_observation:
        on_observation(req.observation)
    if owner.has_action_response(req.request_id):
        return owner.action_response(req.request_id)   # 防网络重试重复计划
    view = owner.memory.snapshot_read_only()
    version_before = view.version
    x = sample_action_noise(horizon=16, action_dim=action_contract.dim)
    for tau in action_scheduler.timesteps:             # 仿真论文50步
        pred, delta = model.forward_hybrid(
            action_input(x, tau, req.current_proprio), memory_view=view,
            token_meta=action_meta(cutoff=owner.last_frame_id),
            mode="denoise_action", return_memory_delta=False)
        assert delta is None
        x = action_scheduler.step(pred, tau, x)
    assert owner.memory.version == version_before
    # 不提交x的KV；只有可选Swap T在执行确认后独立编码动作历史
    response = decode_actions(x, horizon=16, contract=action_contract)
    return owner.record_response(req.request_id, response)

def reset_or_end(session_id, episode_uid, reason):
    # done / 真正truncated / 显式reset / disconnect均走这里
    # 数据chunk边界不调用；真实任务子阶段不自动重置anchor
    sessions.drop_memory_and_vae(session_id, episode_uid)
    sessions.drop_pending_rgb_actions_and_dedup(session_id, episode_uid)
    # 新episode分配新uid，旧的迟到请求必须被拒绝
```

推理全程no_grad；训练的prefill不能使用这个no_grad入口。snapshot只读意味着action去噪中不存在临时slot抢占历史，底层可拼当前KV而无需额外持久容量。若显存优化让owner与view共享tensor存储，action过程必须禁止并发prefill覆盖这些tensor，直到本轮snapshot释放。

batch处理时reset只清对应stream；交换batch排列也不会交换memory归属。每rank的请求版本必须一致；异常时统一广播abort，避免某rankcommit而另一个rank仍停在旧version。第一阶段单session序列服务足以验证机制，不必同时重构成异步多环境调度器。

## 6. 分阶段实施与验收计划

### 6.1 按依赖推进，禁止混合归因

| 阶段 | 修改范围 | 预期结果 | 验收条件与停止门槛 |
| --- | --- | --- | --- |
| 0. 固定证据和基线 | 后续新增审计/回归脚本，记录commit、checkpoint/config/data hash；读取实际batch/frame ids | 可重现的legacy训练loss与逐步推理输出、cache时间线 | 不安装/训练也能完成静态契约；数值基线随后用同权重/RNG/输入保存。区分本地和7c6ffa9基线，不能先改旧路径再测一致 |
| 1. 纯元数据和mask参考 | 建议memory.py、memory_mask.py及同目录附近的专项checks | 图3的typed mask、A/R/G集合、逐帧唯一性和预算oracle | Ninit=1/R=1的三帧图3逐格一致；2/4/8跨边界集合、重叠去重、batch换位/reset、空前缀、padding通过；不需要加载5B模型 |
| 2. 接入gist和纯函数attention | model.py新增enabled分支；旧forward默认不变；明确layer delta与RoPE | 小型多层Transformer可完整前缀和streaming运行，gist有梯度 | 相同输入/条件下dense与streaming的逐层KV、video/action输出一致；改变未来帧不能改变过去KV；gist≥第二层有依赖本帧的证据；满池去噪不改历史 |
| 3. 长历史训练数据与循环 | episode_memory_dataset、Trainer的mask/目标移位/有效loss；保留FSDP/AC主流程 | 在共享LingBot骨干上训练混合记忆，保留EEF接口；先full-prefix，后可选TBPTT | A[t+1]时间金标准通过；无未来image/action/proprio泄漏；gist/相关投影梯度非零；AC开关与单/多rank一致；短序列可降低训练loss且通过延迟信息合成任务 |
| 4. 真实观测闭环提交 | VA_Server、session生命周期、RMBench客户端协议；action-only推理 | 真实视频一次prefill + 16步动作生成，重复去噪只读 | 50与其他步数的memory版本/数量完全一样；重试不增帧；中途终止/reset/异常不遗留；offline teacher-forced与在线相同观测前缀输出匹配；再开始小规模RMBench rollout |
| 5. 机制效果及效率 | 不改架构；同数据/动作/初始化/训练预算开展消融和长rollout profile | 可判断gist/anchor/recent的贡献与计算趋势 | 无memory、recent-only、anchor-only、gist-only、两两组合、全组合、full-video history均有明确设置；统计成功率、误差和KV增长，不能只报告loss下降 |
| 6. 完整论文架构与数据 | 独立action expert、joint14/proprio、新mosaic数据、paper噪声训练、任务特例 | 依据§3–4、A.1的完整实现profile，与机制版分开 | 逐项配置对齐；14D joint闭环round-trip/动作单位验收；初始权重与预训练来源披露；论文缺省处标为工程选择，不借用机制版成功率冒充论文复现 |
| 7. 论文级RMBench评估 | 按任务准备50 demonstrations，9任务、每任务100 rollouts；匹配仿真版本/seed/终止协议 | 报告成功率与效率及未对齐项 | 对齐§4.3任务范围；Swap T action history、Observe预训练按A.1处理；评测不可偷偷使用task state或oracle动作；不以本地3任务/5条holdout宣称完成九任务复现 |

阶段1–4属于**最小机制实现与一致性验证**；完成阶段5才能讨论移植效果；阶段6–7才是完整论文复现目标。保持机制版与paper profile独立结果表，以分离记忆、动作专家、数据表示和额外预训练的影响。公开材料没有给定总训练步数、完整seed列表等，阶段7报告这些偏差而非伪造“官方一致”。

建议专项检查脚本放入**建议新增** `script/memorywam_checks/`，例如`test_mask.py`、`test_lifecycle.py`、`test_streaming_equivalence.py`、`test_gradients.py`与`profile_memory.py`。这些目前不存在，后续按AGENTS要求建立近任务的pytest检查，不假设已有顶层tests。完整5B GPU测试之前先用小模型验证语义；本轮没有创建或运行这些脚本。

### 6.2 验收矩阵：具体检查什么

| 验证项 | 方法与反例 | 通过条件 |
| --- | --- | --- |
| 关闭memory等价 | 固定同一legacy checkpoint、输入、scheduler/RNG/backend；比较原版与新增代码enabled=false的训练输出/loss、各去噪步输出与cache | 同backend/无随机差异优先逐位一致；kernel导致差异时先测legacy自重复误差，再设阈值。建议FP32小reference atol1e-5/rtol1e-4作为起点，BF16以实测误差标定，不把宽松阈值当掩盖bug的理由；旧分支不初始化gist以免改变RNG |
| 形状/预算/淘汰 | 依次prefill N=0..10，特测N=2、4、6、7；记录每层(frame,type,slot)；长到100/1000 | video唯一token数 `L*min(N,A+R)`（A、R非负连续首尾集合），gist=N*M；30层一致；N=7起淘汰中间完整帧但gist不减；不同配置含A/R=0通过 |
| 图3/因果性 | 构造1anchor/1recent三帧，逐格比较；随机改future RGB、future target、future proprio、后续指令而保持当前前缀 | 当前及过去clean/gist KV和当前动作预测不变；当前proprio只属于该决策时刻，不广播未来proprio到全episode text |
| 防间接泄漏 | 给未来clean frame设置可辨别水印；用梯度/扰动追踪早期gist；测两帧合并prefill与一帧一帧prefill | 未来到过去的梯度为0、数值无依赖；无mask多帧prefill必须被测试捕获 |
| episode/batch隔离 | 两episode相同frame_id不同uid、swap batch slots、一条reset另一条继续；CFG正负条件；相同episode编号不同dataset | 与分别单独执行结果一致；session重连迟到消息被拒；不能从一条episode读另一条gist |
| 位置编码 | 索引递增、删完整帧、ring复用slot、>1024长度；与无压缩reference的保留token相位比较 | 缓存K相位不因storage变化而改变；g_t时间=t；动作位置始终与同一决策时间合同对应，h/w marker稳定 |
| 去噪不提交 | 1/10/50步action循环前后比较所有持久KV、meta、VAE状态和version；缓存容量恰满；重复infer/obs request | 去噪循环持久状态逐位不变；每个真实observation id至多提交一次；绝不读上一步噪声action/video KV |
| 事务/异常 | 第k层故意抛错、VAE成功但Transformer失败、某rankreject | 所有层和VAE共同回滚或该episode明确失效后重建；没有半套版本cache；各rank行为一致 |
| 梯度流 | full-prefix vs相同weights的顺序reference；AC开/关；读取历史gist的延迟监督；TBPTT开/关 | gist参数与预期主干有梯度；不detach版本与并行reference接近；TBPTT边界之前梯度按设计截断；不在no_grad中误生成全部训练memory |
| teacher-forcing/在线一致 | 固定相同真实obs前缀，关闭训练增强，分别batch mask与streaming prefill；比较逐层KV和动作条件 | 同前缀/同noise/time输出一致；不借clean future video改善训练loss；启用增强后的差异能由augmentation解释 |
| VAE/标签对齐 | 使用带编号的RGB序列和递增动作，验证首帧+四帧组；联合mosaic数据重新编码；检查尾部mask | 每个动作chunk仅使用执行前观测，物理源frame、latent frame、decision cutoff可反查；VAE不会读future，离线/流式达到预定误差 |
| RMBench控制合同 | joint/EEF分别round-trip，核对左右臂、gripper、单位、quat顺序、frame0 dummy | 实际take_action类型匹配输出schema；当前state来源为传感观测，不是label/整段预测计划；中途截断不提交未执行历史 |
| checkpoint | legacy关闭、新checkpoint保存重载、缺失必要旧key、错误action schema、resume重建 | 旧权重不被改写；新增key列表明确；错误schema拒绝；重载输出一致；恢复训练有明确step/RNG/重放策略 |
| 计算与显存 | N增大测live KV字节、allocated/reserved/peak、prefill时间、每个action step和整chunk时间；固定硬件/精度/CFG | KV斜率匹配N*M；attention随N/d增长；没有随去噪步数积累持久KV；同时报告瞬时scratch和FSDP通信，不只报Python tensor长度 |
| 消融贡献 | 保持训练/评测预算一致，消融分别训练；另报同checkpoint推理消融作为分布外诊断 | A/R/G关闭方式明确定义、当前观测总存在；报告成功率置信区间/seed方差；不能用仅推理剪token推断“该模块无需训练” |

“无记忆”对照应保留当前真实观测条件并去掉跨时历史；“关闭memory模块”的回归则走原LingBot路径，可能仍有原始LingBot KV。二者是不同对照，报告时分别命名，不能把legacy有KV称作无历史。移除recent的论文消融用gist代替近期完整video，依据§4.5；这种设置也仍保留当前观测产生的gist，不能让policy完全看不到当前状态。

### 6.3 参数与缓存显存估算

**实际配置及计数来源：**本轮只读三份safetensors的JSON头部（没有加载权重或模型）并累计shape乘积，checkpoint包含 **5,089,465,566 个张量元素**，索引没有gist/memory/anchor参数名。这个计数是checkpoint的tensor元素计数，不是VAE+T5总系统大小，也不等于训练峰值内存。

符号：`J=30`层、`Dkv=24*128=3072`、bf16每元素`b=2`字节；`B_eff=环境batch数×CFG分支数`。以下忽略metadata、allocator、attention scratch和梯度，数值均为**解析估算**，不是实测。

新增参数：

| 方案 | 公式与估算 | 限定 |
| --- | --- | --- |
| 共享gist输入表 | `M*Dv=8*3072=24,576`，BF16约48KiB | 工程默认跨帧共享、仅输入层一套；论文未公开精确参数共享实现 |
| anchor/recent管理 | 0个可训练参数 | 只有索引与KV内存 |
| 可选proprio→T5 context维度 | 若14→4096且含bias：`(14+1)*4096=61,440` | “text-token dimension”取原T5维4096的工程解释；若在动作hidden1024后投影则为15,360，需固定实现并报告 |
| 独立action expert | 忽略bias/norm/time/input heads，若self/cross各4个Da↔Dkv投影，`J*(8*Da*Dkv + 2*Da*FFNa)`；Da1024、FFNa4096时约1,006,632,960 | 与论文约1B一致的量级估算；文本先投到Da、各attention投影设置是工程实现前提。实际参数必须等实现后count，不能把本公式当作者精确架构 |

新增gist训练状态还包括grad、FP32主权重/Adam moments：若每参数按FP32 param+grad+2 moments则约`16*24,576=393,216 bytes`，另有BF16计算副本与FSDP通信；真正主要代价是长历史激活，不是这几万个参数。全模型训练显存不能用“只新增48KiB权重”估算。

假定全部N帧gist保留、完整video去重，容量为：

```text
V(N) = L * |A(N) ∪ R(N)| = L * min(N, Ninit + Nrecent)
G(N) = M * N
S(N) = V(N) + G(N)
每层KV字节 = 2 * B_eff * S(N) * Dkv * b
全部层KV字节 = 2 * J * B_eff * S(N) * Dkv * b
```

`Ninit=2,Nrecent=4,L=120,M=8`且N≥6时，`S=720+8N`。每token全30层占368,640 bytes=0.3515625MiB；每新增latent帧长期净增8个gist，约**2.8125MiB/分支**。每层则每token12,288 bytes；所有状态都按layer独立保存，没有把一个final hidden误当30层KV。

| 已提交latent帧N | Hybrid token S | Hybrid KV，30层/单分支 | Full-video KV（NL），同口径 |
| --- | ---: | ---: | ---: |
| 6 | 768 | 270 MiB | 253.125 MiB |
| 100 | 1,520 | 534.375 MiB | 4,218.75 MiB |
| 1,000 | 8,720 | 3,065.625 MiB ≈2.99GiB | 42,187.5 MiB |

短历史增加gist会略增加缓存；“15×压缩”是长期每帧120→8的渐近比率，不是所有长度都15×。CFG双分支应将表中值乘2。独立action expert读取同一份video KV时不需要复制第二份视觉缓存；若实现不必要地复制应计入实际显存。Swap T若保留历史动作KV，需要另加`2*Ja*B_eff*S_action*Dkv*b`；A.1未给动作历史预算，不能藏进视觉公式。

现有默认cache容量（`model.py:673`、`VA_Server._reset:419`）为：

```text
S_legacy = (attn_window//2) * (frame_chunk_size*L + frame_chunk_size*P)
         = 36 * (2*120 + 2*16) = 9792
KV_legacy = 3442.5 MiB / 分支；CFG=2时为6885 MiB
```

这是**混合视频+动作的固定预分配池**，历史超预算会FIFO淘汰，不是无限full-video缓存。MemoryWAM是保留全历史gist的增长池，两者保存的信息不同。按此默认，新混合视觉缓存在N≈1134后甚至超过旧固定池容量；不能向用户承诺任意rollout长度始终比现有池更省。若需要绝对显存上限，可预设最大episode长度/拒绝越界，或研究额外gist淘汰/合并，但后者就是偏离论文的变体，不能默认启用。

FSDP分片的是模型参数，现有KV为普通tensor；每个推理rank接收同一请求并持有完整本rankKV。不能把上表数值除以GPU数，也不能将CPU参数offload计成KV已经offload。

### 6.4 Attention与总计算估算

按一次乘加=2 FLOPs，单层QKᵀ与AV主项约：

```text
F_attn ≈ 4 * B_eff * Q * K * Dkv
prefill单帧: Q=L+M，K≈保留的历史+当前(L+M)
action单次去噪: Q=P，K=S(N)+P
一段动作: Ka * J * 4 * B_eff * P * (S(N)+P) * Dkv
```

其中Ka=50为论文仿真动作步数；公式只算attention乘法，不含softmax、QKV投影、FFN、text cross-attention、VAE、通信与内存搬运。固定历史S时，gist让当前prefill token从120增到128，token-linear投影/FFN工作约增加6.7%；但历史K数量大幅下降。N=100时action memory相关attention主项相对full-video约`(1520+16)/(12000+16)=12.8%`；N=1000约`(8720+16)/(120000+16)=7.28%`。这**不是端到端延迟降到同样百分比**。

完整prefix训练若保留所有帧clean/gist query，需要计算每帧与之前所有gist的关系，总边数仍含`O(N²*(L+M)*M)`，加上局部/anchor边；推理每步O(NM)、完整rollout累计O(N²M)。论文§3.3/Fig1讨论的是缓存与单步长历史代价，不能误写成完整episode训练或整段rolloutO(N)。稀疏Flex kernel的实际块利用率和packing会影响效率。

机制版可以共享权重执行video prefill/action-only，但动作去噪仍跑3072宽FFN；论文1024宽动作专家可能是端到端提速的重要来源。需要独立测量：VAE编码/文本缓存时间、视频一次prefill、每动作step、50-step总时间、cache gather/分页、FSDP all-gather、CPU offload、环境执行等待。论文Fig4是**单层单次前向**的比较，不可用作本项目5B完整服务延迟的承诺。

训练显存估算还需测实际batch的F和稀疏布局。仅一个未checkpoint的hidden激活下限量级为`B*T_tokens*D*b`，训练packed tokens约`B*F*(2L+M+P)`（普通任务去掉condition-action；具体target对齐少一个frame），层数、中间MLP、autograd保存、FSDP参数驻留和attention实现再加开销。AC减少中间激活但不消除输入/跨段图；TBPTT缩短梯度图但gist运行态仍增长。以profile峰值为准，不编造可训练最大episode长度或GPU卡数。

## 7. 尚待确认的问题及推荐默认选择

### 7.1 无需等待用户、可以自行确定并记录的工程选择

| 事项 | 已知依据/不确定性 | 推荐默认与影响 |
| --- | --- | --- |
| M、anchor/recent预算 | §4.1/A.1已有8/2/4 | 用论文值；按latent帧而非diffusion step；不重新询问用户历史实验参数 |
| gist共享粒度与初始化 | 论文只有learnable parameters，没有代码级共享/初始化 | `[8,Dv]`跨帧共享、随机小方差；8个向量不能全相同初始化导致对称。记录seed/std，并测试训练；不添加独立compressor |
| gist RoPE marker | §4.1固定h/w，未给值 | 采用(-2,-2)区别已有action(-1,-1)，作为工程假设；不添加RoPE新维度 |
| gist时间调制 | 没给是否沿用condition噪声t | 推理用clean t=0；训练先给gist和同帧condition一致的t，具体增强的t编码需与所选实现固定。可比较“增强输入但t保持0”与“显式编码增强sigma”两种，不能说论文明确选了任一 |
| recent是否含当前 | 图3 recent=1只保留本帧完整video，A.1称最近clean帧 | 默认含当前，prefill和action一致；用图3测试锁定 |
| anchor和recent重复 | 式(4)是并集 | 同帧完整video只出现一次；所有帧gist额外保留 |
| 是否刷新旧gist | A.1是prefill新帧并保留过去gist；没有递归重写规则 | 永不重算旧gist，减少时刻/条件混杂；新gist可读取旧gist |
| 新mask后端 | 当前训练Flex、推理torch无mask | 保留两套backend，用共同visibility oracle；小torch dense reference先通过，再优化稀疏/分页；不强行依赖flash任意mask支持 |
| 训练历史方案 | §4.1描述per-episode，没有TBPTT细节 | full-prefix作为语义reference；显存不够时选前缀重放/顺序TBPTT并披露；不直接随机窗口携带上一batch KV |
| 冻结范围 | 论文没有完整冻结清单；当前全Transformer训练 | 机制版沿用全Transformer可训练，VAE/T5冻结；gist-only可做快速机制实验但不作复现默认 |
| 推理CFG | 本地video guidance5会使use_cfg=True，paper未给CFG细节 | 机制版保留分支兼容；paper profile先action CFG=1并明确无CFG，作为待作者确认的工程选择；不要误留无用途的双倍视觉cache |
| 相同帧重试/异常 | 论文未给服务协议 | observation/request id+原子提交，decode前去重，VAE和Transformer一起回滚或重建 |
| gist硬上限 | A.1明确never evicted | 不设置静默淘汰；配置max_episode用于capacity planning/拒绝越界；O(1)压缩另立实验名 |
| checkpoint runtime memory | 论文未给保存协议；旧项目只存weights/config | 不保存临时运行KV；训练边界续跑或重放已见前缀；保存必要RNG/optimizer/data契约 |
| 本地资料冲突 | 文件名含official/fullctx、脚本注释和运行入口不完全一致 | 最终以effective config、真实source index和调用链为准；“official”不作来源证明 |

### 7.2 真正影响完整复现架构、公开材料未能完全解决的项

这些不阻塞阶段0–3的独立工作；到完整论文profile实现前必须选择并记入设计记录。能依据因果性决定的已有推荐，不把可以自行处理的琐碎项留给用户。

1. **动作/噪声视频的精确索引与MoT层交互。** 图3展示clean/gist/action，省略noisy video；§4.1没有完整packing和mask代码。推荐显式决策cutoff：动作和下一视频都只看已见前缀，彼此噪声query不互相依赖。需要作者代码才能确认其精确排列；不能拿同下标clean target直接给动作。此选择影响训练目标和是否能无video-generation部署。
2. **独立action expert的权重插值细节与cross-attention尺寸。** 论文给Da1024但head空间3072，未给每个矩阵沿哪一维如何插值。推荐保留24×128的Q/K/V空间、FFN输入输出用1024、视觉KV直接读取；建立独立迁移manifest并量化初始化误差。不是给现有`dim`改成1024就结束。
3. **Swap T的action history表示和预算。** A.1明确只该任务使用自回归action history，但未给其缓存窗口、clean action位置和与video交互mask。默认普通9任务中的其余任务禁用；Swap T单列配置，推荐先保存已经执行的完整动作chunk并做因果参考，预算必须显式报告。若使用gist压缩动作，属于新机制，没有本论文依据。
4. **logit-normal采样/weight和训练总量。** shift5/1、lr2e-4、wd.01、betas(.9,.95)、batch1/GPU×8、clip1、双loss权重1来自§4.1/A.1；分布μ/σ、精确loss weight、训练步数、warmup、各任务seed未给。推荐保留独立paper-schedule入口，先声明选用的标准logit-normal参数再做敏感性分析；不把当前uniform-index scheduler改名伪装对齐。
5. **任务数据/预训练的严格可比性。** 本机已有EEF/fullctx数据和LingBot机器人预训练权重，论文为joint14/mosaic且有任务特定预训练。推荐先完成机制移植，完整复现另建数据与初始化profile；若只使用现有EEF checkpoint，应将最终结论限定为“MemoryWAM机制在LingBot-VA上的效果”，不对论文成功率作等价声明。
6. **RMBench精确评测版本和控制时钟。** 本地adapter和环境有改动，公开官方仓库也在演进；本文已核对APF16的本地contract，但控制步的真实秒数和完整九任务seed协议仍需从目标评测版本冻结。默认禁用本地stall/window-reset捷径，遵守对应环境的成功/超时协议。论文每任务100rollouts与本地5-seed holdout不同，不能替代。

### 7.3 需要保持的判断

- 现有attention具备所需QKV/位置/稀疏计算基础，但**不具备可直接使用的MemoryWAM缓存语义**；最小改造是typed token metadata、gist输入、混合mask、函数式KV delta和owner原子提交。
- gist必须是训练出的token表示，现有checkpoint可提供视觉/动作能力的初始化，不能提供未训练的gist压缩能力。没有必须额外增加压缩MLP或gist专用loss的论文依据。
- anchor/recent选择已有论文默认，不需用启发式事件检测替换；gist保留全历史，因此总显存应按线性斜率验收。
- 安全复用的是条件、权重和因果前缀稳定的真实观测KV；噪声迭代KV、未来条件污染的hidden和换prompt后的旧层KV不能混用。
- 一次真实观测prefill一次提交；动作去噪不提交。环境动作执行和diffusion timestep是两个独立时间轴。
- 完整前缀训练可满足论文的可见性；顺序chunk需要等价mask、位置和固定权重，TBPTT是梯度近似；独立窗口无历史重放不等价。仅推理端加memory只能作为未训练消融/诊断，不能算论文复现。

本轮完成的验证限于：本地PDF正文/附录/图3核对、实际代码及调用链阅读、Git差异和忽略规则检查、checkpoint配置与张量头部计数、少量数据契约/元数据抽查、作者项目页与RMBench官方来源核对，以及本文路径/公式检查。GPU数值一致性、梯度、训练收敛、闭环成功率和性能profile均未运行。
