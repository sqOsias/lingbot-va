# RMBench 历史训练、fullctx 资产与下一步基线审计

审计日期：2026-10-03。项目根 `R=/root/wyc/code/lingbot-va2`；数据根 `D=/root/wyc/data/rmbench/lingbot_va2`。下文 `R/`、`D/` 均为这些绝对路径的缩写。外部评测仓库按实际启动脚本确认为 `/root/wyc/code/RMBench`。

本轮优先完成运行/配置/checkpoint 盘点、全量元数据检查、CPU 动作与索引检查、源码静态审计。未运行模型 forward、训练、专家回放、环境 rollout、特征重提取或大规模评测；未修改正式模型、数据、训练/推理脚本。新增审计脚本和报告不改变旧资产。

证据标签：**[运行]** 本轮实际执行的 CPU/文件检查；**[静态]** 当前源码推导；**[历史]** 旧日志/清单/权重；**[论文]** 本地 PDF 或作者资料；**[假设]** 尚未验证。读到历史 PASS 不计作本轮测试通过。

## 1. 旧训练/推理目前是否可信

**结论：核心 fullctx 数据可以继续使用；旧 checkpoint 可用于诊断和权重初始化，不能称完整断点续训。先修正并验证基线的缓存与评测契约，再接入 gist/anchor/recent；不建议现在重做全部数据或增加训练步数。**

|结论|证据与边界|
|---|---|
|正式数据并非只有一条 episode，也不是旧 w193 的别名|[运行] 3 任务各有 50 train + 5 validation + 5 test，共 9 套、180 episode、540 个 camera latent；每 episode 一个 action_config。逐条元数据、latent 头部和全部 parquet 数值检查通过。具体目录见第3节。|
|当前 APF16 的首帧 dummy、动作标签偏移、16→30 通道映射有实证支持|[运行] 调用从实际源码 AST 提取的动作函数，编号序列与每任务 raw episode0 对照通过；31组姿态往返误差 <1e-12。不能据此声称专家控制回放通过。|
|确实有训练更新，不是只打印 loss|[历史+运行] 主要运行有连续 optimizer-step 指标；15 份存量权重的 `action_proj_out.bias` 均有限且不同于 base（最大差约1.91e-6～3.97e-4）。这只证明该参数更新，不等于已经学会闭环任务。|
|当前 attention 条件结构符合视频生成后再动作生成的顺序|[运行+静态] 对真实 mask predicate 的625对 token 检查无跨样本、padding 或跨未来 block 违规；动作允许读同一 chunk 的条件视频。线上确实先生成视频，所以不能直接称它“未来真值泄漏”。训练真值/扰动视频与线上生成视频仍有 exposure bias。|
|`update_cache=0` 在满缓存时不满足只读恢复契约|[运行] capacity=4，已有4个有效 token，临时写2个再 restore 后仅剩2个旧有效 token，旧 payload 已覆盖。`WanAttention.allocate_slots/update_cache/restore_cache/forward`，`R/wan_va/modules/model.py:368–462`。这是已复现的状态突变；对闭环失败的因果贡献尚未量化。|
|bounded 验证配置没有落到实际 loader|[静态+历史] `R/wan_va/train.py:160–179` 始终建立 `MultiLatentLeRobotDataset`；2100-step 运行日志逻辑行52写明 **5 windows**，而 bounded manifest 是 **14 validation windows**。不能称其使用了 bounded 加权验证。|
|bounded 不是目标区专属 burn-in 训练|[静态] `BoundedFullContextDataset.__getitem__` 返回 global/history/weight，但 `_prepare_input_dict:293` 不读取；history 的原 mask 保留并参与 loss。现有结果是带重叠历史的局部窗口训练，可自洽，但不等价于 target-only 前缀重放。是否违背旧设计必须以目标定义为准。|
|不能将 legacy_bgr 确认为旧评测错误|[运行+静态] JPEG 采集链本身存在 RGB 数组交给 BGR 编码器的交换；9 个真实首帧的 MP4 与 raw JPEG PIL 解码 MAE=.899～1.265，交换后 MAE=5.151～8.048。旧报告仅依据 `rgb24` 推断线上错色，证据不成立。当前推荐保留 legacy_bgr，待同源线上观测完整对照。|
|旧成功率是历史小样本结果|[历史] Rearrange750为5/5，Observe550为0/5，Blocks1000为1/5、2100为3/5；checkpoint symlink和client日志可对应。可以保存原始结果，不可以当作100-rollout论文复现结果或本轮重测结果。|

本轮初始 Git HEAD 为 `d986be8`。已有 `.gitignore` 修改、未跟踪 AGENTS/审计脚本/debug 目录保留。W&B 恢复的旧 HEAD 包含 `7c6ffa9` 和 `63b1ba9`；前者是本地保存的上游提交，后者是本地 RMBench 适配。**旧运行的未提交 diff 未完整归档，HEAD 不是运行源码快照**。例如 bounded 运行记为63b1ba9，但该提交尚无当前 bounded 类，说明确有未提交扩展。不能把当前 train.py 的每项修复自动追溯到所有9月运行。

三个目标应分别评价：

- **A 原 LingBot 语义**：共享骨干、clean/noisy block causal 训练、先视频后动作、执行后重填真实视频/动作历史是本仓库实际路线。独立 batch 间无 KV 不否定完整序列中的历史训练；公开 README 也明确本仓库发布的是共享骨干版本。
- **B 本地 RMBench 语义**：正式 fullctx 的 episode-origin EEF/APF16 契约得到支持；bounded 的 target-only/global-position/validation-weight 语义尚未实现；实时 proprio 未作为独立动作条件，`state` 是计划动作历史，不能叫“当前本体状态”。
- **C MemoryWAM**：目前没有经过训练的 gist token及其mask，没有论文的clean-video混合记忆路径，不能把现有有限 KV 称为 MemoryWAM。局部 bounded 窗口不携带此前历史；完整 episode causal 训练可以学习跨帧依赖，但当前mask/读写对象仍与论文不同。

本轮纠正旧 `docs/rmbench_audit_report.md` 中几项过强结论：legacy_bgr不是已证实错误；相同stats哈希不能单独证明无统计泄漏；没有跨batch KV不等于没有历史训练；从零位置与history参与loss不是自动成立的bug；Blocks1000停止原因为max_steps，不能写成提前收敛。

## 2. 历史运行与 checkpoint 对应表

完整逐运行表、时间、停止原因、有效配置出处及早期episode0运行见 [运行明细](rmbench_run_tables_20261003.md)，机器可读见 [运行CSV](rmbench_run_inventory_20261003.csv)。这里给出可影响下一步的主表。每行证据为该目录的 `run_config.json`、`training_metrics.jsonl`、`training_summary.json`、`training.log` 和清单中的 W&B record，不以目录名代替实跑步数。

|R 下运行目录|实际训练数据/窗口|计划→记录step；样本呈现量|现存权重/结束原因|最后验证 video / action|
|---|---|---|---|---|
|`train_out/rearrange_blocks_50demo_300step_20260921_104117`|Rearrange fullctx50，完整25～26latent/episode|300→300；900|checkpoint_step_300；max_steps|.13297946 / .00342880|
|`train_out/rearrange_blocks_50demo_1000step_20260921_180241`|同上，独立初始化运行|1000→750；2250|checkpoint_step_750；验证收敛规则触发|.12662366 / .00225358|
|`train_out/observe_50demo_1000step_20260924_01`|Observe fullctx50，每episode10latent|1000→8；24|无权重，结束原因unknown|无|
|`train_out/observe_50demo_1000step_20260924_06`|同上，新的独立运行|1000→550；1650|checkpoint_step_550；验证收敛规则触发|.15491598 / .00417039|
|`train_out/blocks_bounded_profile_20260922_160912`|Blocks fullctx50的bounded视图；159训练窗口；强制episode20特例|2→2；6|checkpoint_step_2；max_steps|.68153 / .50221|
|`train_out/blocks_bounded_pilot_20260922_161648`|仅启动配置意图可读|100→0；unknown|无权重；未证明开始更新|无|
|`train_out/blocks_bounded_pilot_20260922_161832`|bounded h32/f64|100→100；300|checkpoint_step_100；max_steps|.15632491 / .00458214|
|`train_out/blocks_bounded_formal_20260922_214515`|bounded h32/f64|1000→1000；3000|checkpoint_step_1000；max_steps|.11236389 / .00163396|
|`train_out/blocks_bounded_3000step_20260924_224653`|bounded h32/f64；不是接1000步续训|3000→2100；6300|checkpoint_step_2100；验证收敛规则触发|.10702931 / .00122421|
|`train_out/rmbench_3tasks_2step_smoke_20260920_175325/{observe_and_pickup,rearrange_blocks,blocks_ranking_try}`|各自fullctx50|分别2→2/2/1；6/6/3|前两者checkpoint_step_2；Blocks第1步后OOM|诊断用，非正式性能|
|`train_out/blocks_debug_2step`、`train_out/debug_blocks_2step_20260929_112209`|bounded调试意图|2→0|invalid device ordinal；无checkpoint|无|
|`train_out/rearrange_debug_2step_20260930_{093142,153846,155544,165113,170743}`|Rearrange fullctx调试|2→0，各自独立失败|ChildFailedError/NCCL等；无checkpoint|无|

表内主运行的已恢复有效配置均逐一确认：base=`R/checkpoints/base`，batch每rank1、累积1、world3、有效batch3、lr=1e-5、warmup10、seed42；APF16、模型30维、物理16维、配置chunk2、attn_window72、主相机256×320。三套fullctx50均配独立5-episode validation；test路径保留用于评测，当前launcher只检查其审计文件，不把test样本送进trainer。**无法恢复有效配置的失败运行不套这些值。**

`attn_window=72` 是记录的配置值，实际训练 `Trainer._prepare_input_dict` 随机chunk∈[1,4]、window∈[4,64]，mask的window单位是视频/动作交错block（`model.py:115,189`），并非简单64个latent。完整episode输入不等于每个query总能看到全部前缀。bounded最多64latent，history名义32（首窗等边界不同），实际仅看到被裁出的局部段。未找到这些正式运行使用“随机目标窗口+从episode开头重放前缀”的证据。

样本呈现量=step×3，包括随机重复/DistributedSampler补齐，不是独立episode数。fullctx每样本覆盖一个episode；bounded每样本覆盖一个窗口。没有逐step窗口ID记录，不能精确恢复有效动作监督累计次数。日志时间起止见明细，时间原样保留，未假定时区；例如2100-step横跨9月24日至27日，名字里的3000不能作为已完成预算。

早期episode0运行位于 `R/checkpoints/` 而不在train_out，单独列在明细：Observe/Blocks各300步；Rearrange fullctx300步；名为fullctx1000的目录只有74条指标且无权重；Rearrange w193 episode0 1000步；早期官方风格episode0 1000步；若干10-step和未更新尝试。episode0 filter `[0]` 解释了旧部分latent目录为何当时可运行，不能推断它们是完整50-demo训练。

**权重完整性及恢复：** [运行] train_out有9份、早期目录6份，共15份单文件safetensors；逐份header、dtype、连续offset与文件总长度通过。每份839个tensor，与base三分片合并比较仅少`patch_embedding.weight/bias`，共有参数shape无差异、无额外key。当前模型实际使用 `patch_embedding_mlp`（`model.py:680`），日志也报告旧patch_embedding未使用，不能把这2个key判成动作头丢失。只检查header和一个小bias的数值，**没有全量tensor校验和/有限性扫描或完整load-forward测试**。

`Trainer.save_checkpoint:495–538`保存transformer config和bf16权重；optimizer保存注释，恢复状态调用在203行注释。15份目录未见完整optimizer/scheduler/RNG/sampler状态。它们可作**权重初始化/推理候选**，不能保证完整断点续训。下次训练应标记new optimizer，或者另加真实完整resume保存，不得沿用“resume”日志字样冒充状态恢复。

历史闭环记录（分母由client逐episode日志核对）：

|artifact目录（R/artifacts下）|对应权重|结果/最大控制调用数|证据边界|
|---|---|---|---|
|`rearrange_blocks_50demo_test_5ep_faithful_20260922_100217`|Rearrange750|5/5；500|`eval/_result.txt`、`client.log` PROTOCOL及最终5/5；模型symlink可解析|
|`observe_and_pickup_50demo_test_5ep_faithful_20260924_193143`|Observe550|0/5；250|同上|
|`blocks_bounded_test_5ep_faithful_20260923_132458`|Blocks1000|1/5；3000|同上|
|`blocks_bounded_test_5ep_faithful_20260928_125809`|Blocks2100|3/5；3000|同上，不能与上行合成4/10独立样本|
|Rearrange300的`...test_1ep...160134`及`...test_2ep...163234`|Rearrange300|1/1、1/2|前缀seed重复，不是2/3独立样本|

四个5-episode日志都指向对应task的`D/holdout_work/<task>/test/expert_seeds.json`，seeds=200000～200004；instruction=seen。当前faithful脚本video25/action50、window_reset0、stall0、legacy_bgr；这是当前脚本证据，旧运行的所有环境变量及外部仓库diff没有完整归档，不能保证逐项历史一致。仍应保留这些真实结果及失败trace；要宣称修复后基线或与论文比较，必须按冻结协议重测。

## 3. fulltx/fullctx 资产复用表与实际数据链

实际命名是 **fullctx**；本轮未发现名为fulltx的正式数据集。数据根下20个目录有`meta/info.json`，其中11个名字含fullctx；不能把这11个全部选作训练数据。全量结构清单见 [inventory JSON](rmbench_inventory_20261003.json)，逐episode/segment的帧数与缺失位置见 [episode CSV](rmbench_episode_inventory_20261003.csv)。额外bounded和holdout_work是索引/原始资产，不是同一schema的数据集。

### 3.1 正式九套数据

下表路径统一以 `D/` 开头，均是本轮实际读取，而非根据名称猜测。

|目录|episode/segment|parquet总行数|每episode latent帧数|本轮结论|
|---|---:|---:|---:|---|
|`blocks_ranking_try_50demo_fullctx_apf16_cf8`|50/50|73575|35～152|结构/动作检查通过|
|`blocks_ranking_try_5demo_validation_fullctx_apf16_cf8`|5/5|6988|65～122|同上|
|`blocks_ranking_try_5demo_test_fullctx_apf16_cf8`|5/5|6524|35～152|同上；不可用来调参|
|`observe_and_pickup_50demo_fullctx_apf16_cf8`|50/50|7473|10|同上|
|`observe_and_pickup_5demo_validation_fullctx_apf16_cf8`|5/5|753|10|同上|
|`observe_and_pickup_5demo_test_fullctx_apf16_cf8`|5/5|743|10|同上；不可用来调参|
|`rearrange_blocks_50demo_fullctx_apf16_cf8`|50/50|20085|25～26|同上|
|`rearrange_blocks_5demo_validation_fullctx_apf16_cf8`|5/5|2010|25～26|同上|
|`rearrange_blocks_5demo_test_fullctx_apf16_cf8`|5/5|2007|25～26|同上；不可用来调参|

每episode有 `data/chunk-000/episode_NNNNNN.parquet`，三路 `videos/chunk-000/observation.images.<camera>/episode_NNNNNN.mp4`，三路 `latents/chunk-000/observation.images.<camera>/episode_NNNNNN_start_end.pth`。latent文件存 `latent`、`frame_ids`、F/H/W、text/text_emb及区间。单相机latent序列形状`[F×H×W,48]`；主视角16×20、腕部8×10。按“腕部左右拼宽后放主视角上方”拼latent，最终 `[48,F,24,20]`，patch后每帧120视频token；actions为`[30,F,16,1]`，text_emb=`[512,4096]`。这是**分别VAE编码再拼latent**，不是论文先拼RGB再编码。`lerobot_latent_dataset.py:247–275`和server `_encode_obs:344–394`在这点一致。

全180episode动作均有限，quaternion范数合格、episode/frame index连续，3组split之间完整动作数组哈希无重合。validation/test的holdout manifest分别为100000～100004/200000～200004，都是独立采集5条示范，而非从train切窗。训练raw instructions没有可直接核对的seed字段；高seed排除和manifest是来源证据，动作哈希不同不能单独证明场景绝对独立。未做全部视频逐帧解码或全部latent数值扫描；数值latent/text只抽查每任务train episode0三相机，共9个文件。不存在“540份全量数值验证通过”的结论。

统计来源：三任务train/val/test stats均各自逐字节相同；当前holdout准备脚本105行明确复制train stats。[运行] 另外独立读取全部150条train raw末端位姿重算：Rearrange按其声明min/max得到最大误差1.11e-16，Observe按1%/99%+1.49范围扩张得到0；支持这两者由train数据产生。Blocks来自旧v5的404窗口/73448样本，重算其存储base分位数+range_expansion等式得到0，但未独立重建旧选择集的分位数；**其源统计严格无泄漏仍留来源证据缺口**。180条当前动作归一化后超过±1.5的元素计数为0；不可因此声称所有新场景也不会clipping。

### 3.2 时间、动作和条件：已核对的具体映射

转换入口 `script/rmbench_data/convert_rmbench_to_lingbot_lerobot.py`：`decode_rgb:86`、`wxyz_pose_to_xyzw:98`，源HDF5来自 ` /root/wyc/data/rmbench/raw/<task>/demo_clean/data/episodeN.hdf5`（holdout在holdout_work的raw）。parquet row t配observation t，label是physical endpose t+1。[运行] 此偏移由每任务raw episode0与完整parquet动作对照验证，不是预设A[t+1]。执行label究竟在真实控制器中产生同一轨迹，需专家回放；本轮只验证记录/转换关系。

对正式fullctx：选RGB `0,4,8,...,16(F−1)`；VAE首帧单独编码，之后每4个采样RGB成为1latent，流式提取chunk_frames=8（是RGB采样后帧数，**不是8latent**）。latent j>0累积至source RGB `16j`；A0的16个槽是dummy全部mask=false；Aj(j≥1)对应parquet `[16(j−1),16j)`、physical目标 `[16(j−1)+1,16j]`。

|代表样本/决策|当时可见观测（线上）|动作监督/预测范围|视频目标/条件|有效mask|
|---|---|---|---|---|
|所有任务首个c=2 chunk|source0，latent0|A0 dummy跳过；A1→physical1..16|生成latent1，对应后续RGB4/8/12/16|A0全0，A1的16物理通道×16步有效|
|第二个c=2 chunk|执行后观测至source16；历史latent0/1|A2/A3→physical17..48|先生成latent2/3，对应至source32/48|未到尾部时两个动作帧均有效|
|Rearrange train ep0|404parquet行；101采样RGB；26latent|训练有效physical1..400|最后采样RGB400|末4条parquet标签未监督|
|Observe train ep0|148行；action_config截止145；37RGB；10latent|有效physical1..144|最后采样RGB144|末4条未监督|
|Blocks train ep0|1033行；257RGB；65latent|有效physical1..1024|最后采样RGB1024|末9条未监督|
|bounded后续窗例 `[32,96)`，target64|局部历史latent32..63；并非完整0..63|当前代码A32..95均参与loss；A32不能再误作dummy|输入来自连续fullctx切片|沿用全episode mask，无target-only区分|

最后一行是**索引说明例**，不是声称某个episode必有该窗口；实际159/14条index另有清单。尾部未监督遵循当前4×4对齐/裁切规则，不是已证实错位。若失败发生在末端闭合/抬升，可针对原始尾帧做补齐与valid-mask实验；无需先推翻所有特征。原始RGB、parquet保留尾部，必要时能小范围延长提取。metadata fps=15、采样manifest=3.75；提取脚本注释中的240Hz不足以证明环境真实控制频率。本文使用源帧/控制调用编号，不换算秒。

EEF原始16维为左右各`xyz+quat+gripper`，quaternion从WXYZ转XYZW。相对平移是在原世界坐标系做差（不是初始末端旋转坐标系中的位移），相对旋转`R_origin⁻¹R_t`，origin固定episode0。通道映射物理16→模型`[0..6,28,7..13,29]`，另14维全mask。normalize `(x-q01)/(q99-q01+1e-6)*2-1`，server inverse使用同一epsilon；clip范围外不可逆，但本轮180episode无该类clipping。`_rmbench_action_post_process:278`及外部client `delta_to_absolute_eef:71/_delta_to_absolute:288`为实际依据。四元数归一化及XYZW→WXYZ发生在发往环境之前；单位继承RMBench原endpose，未进行独立物理单位标定。

正式九套text是任务级静态指令，存储text与action_config相同；未发现按专家阶段切换text作为正式fullctx条件。旧分阶段/窗口数据应另查，不能跨版本推定。task-state trace仅用于日志/可选stall判定，不能接进policy作为输入。客户端`observation.state`取joint_action vector，但server编码只取图像；`compute_kv_cache`的`state`实际是`_last_raw_action`，是**预测且打算执行的命令**。正常完整chunk后可作为已执行命令历史，仍不是测量proprio或已达到EEF真值。

### 3.3 资产复用决策

|资产/版本|已验证事实|原LingBot基线|MemoryWAM机制版（共享骨干）|更严格论文对齐|必要处理|
|---|---|---|---|---|---|
|上述9套正式fullctx的RGB/parquet/任务text|全结构、动作数值通过；RGB首帧9路抽查|**可直接复用**当前EEF/APF16契约|**可保留底层数据，但必须修改索引、采样或mask**以显式前缀/目标/记忆生命周期|RGB/text可保留；joint/proprio不是EEF16的可逆替代|保留原文件；冻结版本hash；开发使用validation|
|上述9套连续camera latent、T5 embedding|全540头部通过；9份数值抽查；base路径、cf8记录可读|**可直接复用**，受离线/在线VAE数值一致性尚未实测的限制|**可保留底层数据，但必须修改索引、采样或mask**；不必为gist重提特征|论文先RGB mosaic编码：**需要重新提取特征**；T5任务文本可保留|若对齐论文只重提视频分支，勿连带重建动作/文本|
|`blocks_ranking_try_bounded_v1_h32_f64`|159train/14val索引；源50/5episode|局部窗口实验可沿用；target-only称谓不可用|**可保留底层数据，但必须修改索引、采样或mask**|同左，且需论文架构数据条件|源latent不动；加入可靠episode-ID查表、target mask、明确prefix策略|
|`blocks_ranking_try_episode0_fullctx_apf16_cf8`|50episode/90segment元数据；267个latent缺失；ep0三路完整|明确filter=0时可复查旧过拟合；不能当50demo|**证据不足**，已有正式fullctx替代|无须修补该旧视图|保留归档，禁止按目录名选它|
|`rearrange_blocks_episode0_fullctx_apf16`|50episode/148segment；441个latent缺失|同上|同上|同上|选正式50demo目录，不花成本补旧视图|
|`rearrange_blocks_50demo_w193_apf16_official`|50episode/150segment；441个latent缺失|只对已抽特征的episode0成立|**证据不足**|无须优先补齐|底层RGB/parquet可留存|
|`blocks_ranking_try_50demo_w193_apf16_official`|50/404；文件存在检查通过，未做全量数值|APF16独立窗口历史可复查；**证据不足**直接替代正式fullctx|连续历史优先正式fullctx；若硬用该版需重提连续特征|论文mosaic需重提|不要把每窗重启VAE的latent串成原生连续episode|
|Blocks `w193_v1/v2/v3/v4_lossless/v5_fresh_latents`、`blocks_episode0_w193/w193s64`|APF4旧契约，origin/归一化版本各异；存在性可查|旧实验归档，**证据不足**直接配APF16运行|底层数据可保留；若无连续特征需重提|按所需模态选择重提|不混用v1窗口origin、v3episodeorigin与v4/v5扩张stats；名称fresh不是一致性证明|
|原始HDF5及holdout_work raw|训练150条endpose读取，episode0图像抽查；holdout来源清单可查|**可直接复用**为原始来源|同左|joint14/proprio路线：**需要回到原始数据重新转换**这些字段|这是新增动作/状态派生资产；不等于重做所有RGB/text|
|15份旧checkpoint|header/shape一致；有动作bias更新；无完整训练状态|可推理诊断/权重初始化|共享结构参数可复用，gist需新训练|MoT动作专家维度不同，不能直接宣称完整兼容|加载strict差异报告；旧optimizer/RNG不可恢复|

## 4. 按严重程度排序的问题及最小补丁建议

以下为建议，**本轮未修改正式源码**。

|优先级/分类|证据|影响范围/受影响运行|最小修复或证伪|重提取/重训？|
|---|---|---|---|---|
|P0 已复现实现错误：临时缓存写入不可回滚|本轮cache CPU反例；model.py:368–462|当前长rollout缓存满后；Blocks最值得核查。旧同代码运行是否到边界需结合trace，不归咎全部失败|`update_cache=0`构建临时K/V视图，不改持久mask/id/payload；若定义历史+query总budget则仅在临时视图裁历史，持久提交时再淘汰。容量边界、重复去噪、异常路径快照一致测试|无需重提；先旧权重重测，不先重训|
|P1 已确认配置未生效：bounded_validation_index|trainer167直接MultiLatent；2100日志5 windows vs manifest14|bounded pilot/formal/2100的“bounded验证”说法不成立；现有full-episode验证数字本身可保留|明确选择保留full-episode验证并改配置说明，或实例化Bounded validation并使用episode权重。为baseline最少返工推荐先保留full-episode验证，不改变旧指标定义|无需重提；改验证目标后须重测验证，未必重训|
|P1 已确认与target-only/burn-in目标不等价|bounded36–57返回字段；trainer293只读4类tensor，history mask原样|所有当前bounded训练；不能称完整前缀记忆训练|若下一步采用target-only，新增target mask贯穿video/action loss；单独构建prefix replay/完整episode方案。global位置是否用绝对值需按跨chunk缓存决定|不重提；新训练目标须另起训练版本|
|P1 历史代码/评测配置追溯不足|W&B HEAD有但旧diff不全；外部RMBench启动参数可查、源码版本未锁定|所有公开比较声明；早期episode0尤其弱|保存2仓库commit+dirty diff+有效config+数据hash+命令；当前旧结果仅历史diagnostic等级|无需重提；正式基线须重测|
|P1 缺完整训练状态|15份权重目录及save/load源码|所有“继续训练”|区分warm start；未来保存optimizer/scheduler/step/RNG/sampler，新增格式不改旧权重|不重提；不可恢复过去缺失状态|
|P2 待验证：生成视频/专家动作条件带来的线上差异|mask动作可读同chunk条件视频，线上generated；历史动作训练标签 vs 预测执行指令|三任务，Observe失败需优先定位，不是既证泄漏|固定validation ep、相同噪声做真实条件诊断vs生成条件完整动作decode，检查误差最先出现在哪一支|先离线decode；有系统性差异再改训练，别先加步数|
|P2 待验证：离线/在线VAE状态数值一致性|两侧同base路径/归一化/相机拼法；离线cf8，首个执行后线上4RGB，后续8RGB|连续fullctx及长rollout|同episode固定17采样RGB，比较离线1+8+8与线上1+4+8+4的latent；先CPU索引再空闲GPU数值|只有证实编码状态差异才重提受影响特征|
|P2 版本误选：旧episode0目录不全|全元数据扫描的267/441缺失|只影响将旧视图当50demo；不影响已filter0的过拟合解释|显式数据manifest、episode filter检查，选完整9套|不用重建正式数据|
|P2 待验证：裁尾是否截掉决定性动作|CSV保存逐episode未监督尾部；Observe ep0丢4label|任务尾端表现；不是已证错位|只对失败阶段核对raw事件；必要时补4的倍数末帧和valid mask|最多少量延长VAE提取/重采样；不需全量重新转换|
|P2 待验证：Blocks旧统计来源选择|存储base分位数与扩张等式成立，原404窗集合尚未重导|严格来源声明；当前180episode无clipping|重导旧选择索引后只重算stats比较；不要直接换stats破坏旧checkpoint标度|不用先重提或重训|

位置从零重启对孤立窗口的共同RoPE平移可能自洽，不能仅此判bug；与跨窗口已有K/V拼接时才必须统一位置。当前trainer窗口mask在交错block上的定义，也不能直接拿64与推理72对比得出“不可能看到72帧”的结论。

推理生命周期静态追踪：外部 `deploy_policy.eval:200` → `LingBotClient.infer:165` → server `_infer:463`先video再action，最后零噪声前向写pred cache →反归一化/EEF绝对转换→环境`take_action(..., action_type='ee')` →每4动作回传一张观测→`_compute_kv_cache:594`清pred、VAE编码真实图像，以t=0重填视频与动作，推进`frame_st_id`。首轮跳过16dummy，后续通常32动作；所以同一环境时刻的25/50次去噪不是25/50次时间推进。

reset在server396清transformer与两路VAE，client153/外部reset_model180重置初始origin/首轮标志。当前deploy_policy350–359在成功、step_limit、window reset边界和stall提前终止时不写部分chunk，正常完整chunk才写；未发现可据此确定的“必然把未执行尾计划提交”的常规路径。异常中断后复用session仍应强制reset，不能在新Memory代码中默默继续。原实现单会话`cache_name='pos'`，未来多episode batching必须显式隔离；本轮未运行多客户端串扰测试。

## 5. 下一步唯一推荐路线与切换条件

**选择“保留正式fullctx + 现有共享骨干EEF/APF16，先用Rearrange750建立可复查的推理基线，再用Blocks长历史验证记忆收益”。** 不先换MoT、joint动作或RGB mosaic，不以Observe0/5直接决定重训。

1. **当前已完成**：盘点、动作/索引/结构检查、mask predicate、cache反例和来源核查。底层正式数据无需返工。
2. **先做基线最小修复/诊断**：仅改临时cache写入为无副作用；显式固定旧full-episode验证含义，增加运行快照。以旧Rearrange750权重做第6节的固定样本完整动作decode和开发seed诊断。位置、归一化、动作horizon、legacy_bgr保持对应契约。
3. **基线闸门通过后**：在Blocks2100上做不训练的长历史cache边界诊断，确认是否触及容量、作用时刻、延迟/显存。Blocks50demo fullctx35～152latent已有连续历史，先保留，不重建数据。此时再实现论文的gist/anchor/recent训练mask与提交规则。
4. **机制比较另开成对训练**：从同一 `R/checkpoints/base` 初始化共享骨干baseline和混合记忆，固定三任务中选定任务的数据、EEF16/30映射、样本调度、有效batch、噪声seed和目标暴露量。先1episode/2chunk梯度验证，再小预算；不要拿训练2100步的旧checkpoint对比新训练不同预算的memory结果。也可对两个分支同用某个旧权重warm-start，但必须两边一致并明确继承的训练偏差。

切换条件：若同源VAE分块编码出现超过重复同路径数值基线的差异，先修编码契约、仅重提受影响latent；若专家动作转换/回放失败，先修控制适配；若旧权重完整decode在真实条件好而生成视频条件差，先定位视频条件偏差；若这些都成立而短程基线仍失败，才启动有预算的重新训练。只有目标升级为**严格论文架构/协议复现**，才切MoT+joint14+RGB mosaic派生数据。不得因闭环失败循环追加训练步数。

旧结果保留为历史证据，不删除；修复后比较及论文声明都需新测。最终test的200000～200004已经在历史多次查看，不适合作为全新“未看过最终测试集”；后续开发用validation100000～100004，正式100-rollout集合应另行预注册且两分支一致，不能通过最终test择checkpoint。推荐checkpoint规则：开发协议固定后，按同一验证指标/固定step规则选择，保存所有候选及选择日志；不采用“哪个test成功率高用哪个”。

## 6. 第一个可执行实验（下一轮，不自动启动）

### 6.1 本轮已执行的诊断与可复跑入口

已有 `script/audit_rmbench_history.py` 只读审计器本轮重跑输出新文件；新脚本独立于正式训练流程。用现有环境，不安装包：

```bash
/root/miniconda3/envs/lingbot/bin/python script/audit_rmbench_history.py --latent-metadata --output docs/rmbench_inventory_20261003.json
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /root/miniconda3/envs/lingbot/bin/python script/audit_rmbench_cpu_20261003.py
/root/miniconda3/envs/lingbot/bin/python script/audit_rmbench_provenance_20261003.py
OMP_NUM_THREADS=1 /root/miniconda3/envs/lingbot/bin/python script/audit_rmbench_contracts_20261003.py
OMP_NUM_THREADS=1 /root/miniconda3/envs/lingbot/bin/python script/audit_rmbench_mask_20261003.py
```

输出：[CPU检查](rmbench_cpu_checks_20261003.json)、[来源与权重header](rmbench_provenance_20261003.json)、[颜色/统计/小tensor检查](rmbench_contracts_20261003.json)、[mask检查](rmbench_mask_20261003.json)。mask脚本初次运行因Python int而非tensor索引触发helper类型错误，修正审计脚本后完成625对检查；未修改模型。权重对比先合并base的三个分片，不能只拿一个分片判断missing keys。

|已跑检查|结果|不覆盖|
|---|---|---|
|19个train_out配置目录，额外10个早期日志目录|29份日志中24份恢复W&B配置；15份权重header通过|无日志不代表从未启动；未恢复全部旧diff|
|20套数据全元数据；正式9套全部540latent头部|正式180episode零结构问题；旧部分视图缺失已定位|全视频解码、全latent数值|
|正式180episode动作/索引/有效mask/归一化范围|零异常、split整轨迹动作哈希不重复|真实环境动作执行、独立场景泛化|
|每任务raw ep0→parquet；编号序列；31姿态往返|偏移、dummy和inverse通过|所有raw逐行与parquet交叉比对|
|150条train raw的统计来源核查|Rearrange/Observe重算匹配；Blocks扩张等式匹配|Blocks原404窗分位数采样重建|
|3任务×3相机首帧颜色；9个latent/text数值|结果见JSON；支持保留既有色彩契约|VAE重新编码数值等价|
|真实mask predicate；满cache临时写入反例|625对mask约束通过；cache恢复失败已复现|GPUattention kernel、模型loss/forward等价|

### 6.2 下一步首个模型实验：Rearrange750固定验证片段完整动作解码

|项|固定选择|
|---|---|
|任务/数据|rearrange_blocks；`D/rearrange_blocks_5demo_validation_fullctx_apf16_cf8`，episode0，前2个决策chunk；原始RGB可取同目录video，起始origin取observation_anchors。train50目录仅提供训练stats和固定task prompt；不读test。|
|checkpoint|`R/train_out/rearrange_blocks_50demo_1000step_20260921_180241/checkpoints/checkpoint_step_750/transformer`，VAE/T5/tokenizer来自`R/checkpoints/base`。这是权重加载，optimizer step预算0。选择它是已完成且可追溯的同任务候选；5/5只作历史线索，不作为后续最终test择优规则。|
|配置|现有`rmbench`推理绑定的base维度，EEF16→30、APF16、c=2、主256×320、三相机分开编码；stats=`D/rearrange_blocks_50demo_fullctx_apf16_cf8/rmbench_action_stats.json`；seed42；faithful video25/action50；显式固定guidance及所有继承配置并导出有效JSON；legacy_bgr用于nativeRGB在线输入，已经JPEG解码的离线video不得再次盲目swap。|
|对照|相同噪声下，①真实观测+真实未来视频/动作历史仅作oracle诊断；②实际先生成video再完整action decode；两者均解码到物理EEF。oracle不能报策略成功。另做同一RGB的离线/在线VAE分块对照。|
|预算/退出|最多2决策chunk×2条件版本=4次完整action decode（每次50步及末尾clean写入），生成版最多2×25video步骤；无梯度。任一shape、非有限、映射或reset断言失败立即停止；单进程最多30分钟超时。|
|硬件|沿用现有FSDP+offload服务路径时申请2张空闲24GB级GPU及CPU内存；这是已有启动拓扑参考，不是本轮实测最低需求。若无空闲资源不启动。记录加载/单chunk耗时、逐rank峰值allocated/reserved和CPU RAM。环境诊断另需RMBench仿真资源，避免与现有任务竞争。|
|通过条件|strict加载差异仅已知兼容项；输入/输出/反归一化有限；dummy首轮跳过16，第二轮32；full decode的误差与任务阶段可追踪；reset清理完整；短程有无cache修复在未满容量时同随机种子输出在预先测定bf16重复运行误差内一致。不是任意设一个loss阈值即宣称学会任务。|
|回答的问题|问题最早出在VAE预处理、动作decode、生成视频条件还是控制接口？旧teacher-forced低loss能否转化为可执行动作？|

离线通过后，只对**预先固定validation seed100000**运行一次最多500控制调用的闭环诊断；若需定位控制适配，可额外一次专家动作回放，独立记账，绝不计作模型成功率。该1-rollout仅诊断，不证明泛化或论文水平。之后才扩大开发集与训练预算。

## 7. 与 MemoryWAM 对齐及最多三个后续改进假设

依据本地 `docs/Yang 等 - 2026 - MemoryWAM Efficient World Action Modeling with Persistent Memory.pdf` §4.1/4.3/4.5、Appendix A.1；本轮另核对[作者官方README](https://github.com/yangsizhe/MemoryWAM/blob/main/README.md)，其中结果与本地Table1一致，训练/推理代码仍列在TODO。本轮不使用同名工作的机制替代。

|项目|论文明确内容|本地当前/实际差距|
|---|---|---|
|任务结果（LingBot→MemoryWAM）|Table1：Observe13→27；Rearrange100→100；PutBack100→100；SwapBlocks99→100；SwapT88→94；Battery41→41；BlocksRanking100→100；Cover79→98；Press84→87；平均78.2→83.0（%）|本地只核到三任务的小样本，0/5、5/5、3/5不是同协议估计；未覆盖其余任务|
|数据/评测|每任务50expert demo，100rollouts|本地同样名义50demo，但示范文件、场景seed、任务版本、step_limit及专家筛选协议不保证相同；只5rollouts|
|历史baseline|§4.3/官方README明确LingBot保留full historical KV|本地attn_window72的有限cache，长Blocks超过72latent；两者不是同一baseline|
|动作/状态|§4.1为joint14；独立proprio投影加入action context|本地EEF16映射30，独立当前proprio未输入；不能从EEF唯一恢复joint标签|
|骨干|video Wan2.2 5B、30层D3072；action独立约1B、D1024，attention24×128|本地共享3072骨干、已有action投影/条件embedding。该路线能验证混合记忆机制贡献，不能称完整MoT复现|
|视觉|三图先拼384×320 RGB mosaic再VAE，120token/latent|本地分别编码后拼latent，也120token；token数相同不等价，严格对齐需要视频特征重提|
|闭环协议|A.1动作50去噪步，video generation关闭；每16动作取子步3/7/11/15观测|本地faithful先video25，再action50；首轮16/后续32动作；不能直接关video而保留训练mask声称等价|
|记忆|§4.1/A.1：每帧8个可学习gist，首2clean帧sink/anchor，最近4clean帧；每个历史帧gist不淘汰|本地未训练gist。不要另加动态anchor/辅助loss；每帧gist累计仍O(T)，不是严格常量显存|
|额外预训练/动作历史|A.1：Observe额外RoboTwin预训练；SwapT纳入动作历史，其他任务不使用该项额外预训练/动作历史；无预训练Observe另报Memory5/LingBot3|本地LingBot base已经是预训练权重，不能与“无预训练3%”混作同设置；本地始终有动作cache历史|
|效率测量|§4.2为受控记忆机制比较，A.2单RTX4090描述真实机器人部署|本地无对应硬件/测量范围匹配的实测；不能把4090套到所有simulation图表，必须分别记录KV预填/去噪/VAE/传输/环境时间|

本地当前结果只能作**同benchmark下有条件的参考**。Observe0/5对13%的论文baseline不足以证明显著退化；Blocks3/5对100%的差距值得排查，但最便宜证伪优先是cache边界和有效历史对照，而不是加训练步数。Rearrange短任务可先检查基线契约；未来成功率接近上限时，可预注册“在成功率非劣条件下减小延迟和显存”的目标，但5/5不代表已达到成功率上限。

三项后续方向按顺序推进：

|类别/瓶颈|改动与预期机制|最小实验/成本|失败时如何解释|
|---|---|---|---|
|1 修复现有问题：cache读写与基线可追溯性|只读denoise临时视图、真实观测commit、reset断言；保留其他架构|CPU边界测试已给；Rearrange短片段+Blocks超过容量的固定历史离线对照，0训练步|若修复不改善动作或闭环，说明边界突变不是主要性能瓶颈；仍应保留正确生命周期|
|2 机制移植必需：历史太贵/有限窗口丢早期线索|在同共享骨干上加入论文gist、首2anchor、recent4的mask与训练；固定数据动作和初始化作成对比较|先toy2chunk梯度/无记忆等价，再同预算小训练；完整成本待真实profile。消融无记忆/anchor/recent/gist及组合|若仅效率改善且成功率维持，也有价值；若效果差，先查gist梯度、训练可见历史和提交语义，不马上堆压缩器|
|3 严格论文复现必需：MoT/joint/proprio/mosaic及协议未对齐|在第2项机制成立后单独迁移独立动作专家、raw joint14/proprio和RGB mosaic，再匹配预训练与rollout协议|先仅1episode重建相关模态与加载shape试验，再决定全量重提；成本最高|论文配置不同不证明它必然改善本地任务；若无收益，报告架构/控制/数据差异，不承诺超过论文|

需区分两类最终比较：①固定本地数据、初始化、EEF动作、训练预算，仅改记忆识别机制贡献；②对齐论文任务/动作/预训练/评测协议，仍列出无法对齐的来源和硬件差异。两者不能合并成一个“达到MemoryWAM”的数字。

量级核对（来自base实际config）：L=30、heads24、head_dim128、bf16 K/V。当前c=2时每chunk video240+action32token，attn_window72分配36×272=9792token/层。KV总字节=`2×B_cfg×30×9792×3072×2`，约3.36GiB×B_cfg（CFG batch=2时约6.72GiB），不含临时gather、激活、权重、VAE/FSDP。论文式video记忆每层token约`120×|anchor∪recent|+8T`（gist保留/去重细节另按方案固定），长时仍随T线性增长但斜率较小；attention score工作量约`2×B×heads×Q×K×head_dim`乘QK/AV两项相应常数。不能据此承诺实测延迟。若只新增共享gist embedding且Mv=8,D=3072，参数量24576；是否每层独立、如何与现有共享骨干结合是工程选择，非论文公开代码事实。

尚未解决但不阻塞当前路线的核心未知：旧运行完整dirty源码、作者实际RMBench示范/任务版本/测试seed、Blocks旧统计的完整选择重建、在线/离线VAE数值等价、真实专家回放及完整模型decode质量。本轮不把这些unknown补成默认值，也无需让用户先替我们决定常规实现细节。下一步路线已经唯一：**先完成可复查的旧权重基线诊断和最小cache修复，保留正式fullctx，再接记忆。**
