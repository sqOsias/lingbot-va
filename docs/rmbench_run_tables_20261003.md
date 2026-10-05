# 本轮历史清单明细（2026-10-03）

路径根：`/root/wyc/code/lingbot-va2/`；数据根：`/root/wyc/data/rmbench/lingbot_va2/`。时间为日志原样，未假定时区；逻辑行使用 Python splitlines，包含进度条 CR。W&B HEAD 不代表无未提交改动。

## train_out：19 个 run_config 运行

|运行目录（相对项目根）|日志时间起止|模式/实际源|计划→已记录 optimizer step；样本呈现量|现存 checkpoint|停止/失败证据|
|---|---|---|---|---|---|
|`train_out/blocks_bounded_3000step_20260924_224653`|2026-09-24 22:47:52 → 2026-09-27 02:52:27|bounded_v1；blocks_ranking_try_50demo_fullctx_apf16_cf8|3000 → **2100**；6300|checkpoint_step_2100|held_out_validation_losses_converged|
|`train_out/blocks_bounded_formal_20260922_214515`|2026-09-22 21:46:15 → 2026-09-23 13:18:29|bounded_v1；blocks_ranking_try_50demo_fullctx_apf16_cf8|1000 → **1000**；3000|checkpoint_step_1000|max_steps|
|`train_out/blocks_bounded_pilot_20260922_161648`|unknown → unknown|bounded_v1；unknown|100 → **0**；unknown|无|无更新证据；空日志/未证明启动|
|`train_out/blocks_bounded_pilot_20260922_161832`|2026-09-22 16:19:31 → 2026-09-22 18:04:19|bounded_v1；blocks_ranking_try_50demo_fullctx_apf16_cf8|100 → **100**；300|checkpoint_step_100|max_steps|
|`train_out/blocks_bounded_profile_20260922_160912`|2026-09-22 16:10:10 → 2026-09-22 16:16:13|bounded_v1；blocks_ranking_try_50demo_fullctx_apf16_cf8|2 → **2**；6|checkpoint_step_2|max_steps|
|`train_out/blocks_debug_2step`|unknown → unknown|bounded_v1；unknown|2 → **0**；unknown|无|无更新证据；invalid device ordinal|
|`train_out/debug_blocks_2step_20260929_112209`|unknown → unknown|bounded_v1；unknown|2 → **0**；unknown|无|无更新证据；invalid device ordinal|
|`train_out/observe_50demo_1000step_20260924_01`|2026-09-24 10:55:35 → 2026-09-24 10:58:12|fullctx；observe_and_pickup_50demo_fullctx_apf16_cf8|1000 → **8**；24|无|step 8 后无 checkpoint；终止原因 unknown|
|`train_out/observe_50demo_1000step_20260924_06`|2026-09-24 13:39:55 → 2026-09-24 18:05:17|fullctx；observe_and_pickup_50demo_fullctx_apf16_cf8|1000 → **550**；1650|checkpoint_step_550|held_out_validation_losses_converged|
|`train_out/rearrange_blocks_50demo_1000step_20260921_180241`|2026-09-21 18:03:40 → 2026-09-22 06:49:37|fullctx；rearrange_blocks_50demo_fullctx_apf16_cf8|1000 → **750**；2250|checkpoint_step_750|held_out_validation_losses_converged|
|`train_out/rearrange_blocks_50demo_300step_20260921_104117`|2026-09-21 10:42:15 → 2026-09-21 15:03:39|fullctx；rearrange_blocks_50demo_fullctx_apf16_cf8|300 → **300**；900|checkpoint_step_300|max_steps|
|`train_out/rearrange_debug_2step_20260930_093142`|2026-09-30 09:33:50 → 2026-09-30 09:39:04|fullctx；rearrange_blocks_50demo_fullctx_apf16_cf8|2 → **0**；0|无|无更新证据；ChildFailedError（debug）|
|`train_out/rearrange_debug_2step_20260930_153846`|2026-09-30 15:39:49 → 2026-09-30 15:40:37|fullctx；rearrange_blocks_50demo_fullctx_apf16_cf8|2 → **0**；0|无|无更新证据；ChildFailedError（debug）|
|`train_out/rearrange_debug_2step_20260930_155544`|2026-09-30 15:57:15 → 2026-09-30 15:59:01|fullctx；rearrange_blocks_50demo_fullctx_apf16_cf8|2 → **0**；0|无|无更新证据；ChildFailedError（debug）|
|`train_out/rearrange_debug_2step_20260930_165113`|2026-09-30 16:52:24 → 2026-09-30 16:53:02|fullctx；rearrange_blocks_50demo_fullctx_apf16_cf8|2 → **0**；0|无|无更新证据；ChildFailedError（debug）|
|`train_out/rearrange_debug_2step_20260930_170743`|2026-09-30 17:09:03 → 2026-09-30 17:09:48|fullctx；rearrange_blocks_50demo_fullctx_apf16_cf8|2 → **0**；0|无|无更新证据；ChildFailedError（debug）|
|`train_out/rmbench_3tasks_2step_smoke_20260920_175325/blocks_ranking_try`|2026-09-20 18:07:33 → 2026-09-20 18:08:23|fullctx；blocks_ranking_try_50demo_fullctx_apf16_cf8|2 → **1**；3|无|step 1 后 OOM|
|`train_out/rmbench_3tasks_2step_smoke_20260920_175325/observe_and_pickup`|2026-09-20 17:54:24 → 2026-09-20 17:58:34|fullctx；observe_and_pickup_50demo_fullctx_apf16_cf8|2 → **2**；6|checkpoint_step_2|max_steps|
|`train_out/rmbench_3tasks_2step_smoke_20260920_175325/rearrange_blocks`|2026-09-20 18:00:29 → 2026-09-20 18:06:25|fullctx；rearrange_blocks_50demo_fullctx_apf16_cf8|2 → **2**；6|checkpoint_step_2|max_steps|

“呈现量”=配置 batch×累积×world×已记录 step；包括 sampler 重复/补齐，不等于独立示范数或有效动作目标数。bounded 抽样窗口记录未逐 step 保存，精确 target 暴露量 unknown。0 表示未找到 update 记录，不保证进程从未启动。

## 有效配置逐运行

详见同目录 CSV `rmbench_run_inventory_20261003.csv`；二进制配置与代码 hash 详见 `rmbench_provenance_20261003.json`。以下所有找回配置的 train_out 运行均单独核对得到 batch=1、累积=1、world=3、lr=1e-5、warmup=10、seed=42、APF=16、模型 action_dim=30、配置 chunk=2、attn_window=72、主视角 256×320、cfg_prob=.1、base 初始化。无配置的三个 blocks 失败/空运行不能套用这些数值。

## checkpoints 下早期运行（不与 train_out 合并）

|运行目录|日志首尾|已记录 step|配置数据/episode filter|现存权重|
|---|---|---|---|---|
|`checkpoints/observe_blocks_episode0_300step_20260919_151950/blocks_ranking_try`|2026-09-19 21:34:55 → 2026-09-20 06:49:57|300|blocks_ranking_try_episode0_fullctx_apf16_cf8；[0]|checkpoint_step_300|
|`checkpoints/observe_blocks_episode0_300step_20260919_151950/observe_and_pickup`|2026-09-19 15:20:48 → 2026-09-19 21:32:56|300|observe_and_pickup_50demo_fullctx_apf16_cf8；[0]|checkpoint_step_300|
|`checkpoints/rearrange_blocks_episode0_fullctx_1000step_20260917`|2026-09-17 21:07:23 → 2026-09-17 21:08:14|74|rearrange_blocks_episode0_fullctx_apf16；[0]|无|
|`checkpoints/rearrange_blocks_episode0_fullctx_300step_20260917`|2026-09-17 22:28:05 → 2026-09-18 06:49:45|300|rearrange_blocks_episode0_fullctx_apf16；[0]|checkpoint_step_300|
|`checkpoints/rearrange_blocks_episode0_overfit_1000step_20260915`|2026-09-15 23:56:59 → 2026-09-16 20:40:58|1000|rearrange_blocks_50demo_w193_apf16_official；[0]|checkpoint_step_1000|
|`checkpoints/rmbench_50episode_formal_1000step_20260915`|2026-09-15 23:37:31 → 2026-09-15 23:40:17|0|unknown；unknown|无|
|`checkpoints/rmbench_episode0_official_overfit_1k`|2026-09-14 12:12:52 → 2026-09-15 02:15:25|1000|unknown；unknown|checkpoint_step_1000|
|`checkpoints/rmbench_episode0_smoke_10step_20260915`|2026-09-15 22:58:26 → 2026-09-15 22:59:30|10|blocks_ranking_try_50demo_w193_apf16_official；[0]|无|
|`checkpoints/rmbench_episode0_smoke_10step_fixed_20260915`|2026-09-15 23:23:11 → 2026-09-15 23:35:16|10|blocks_ranking_try_50demo_w193_apf16_official；[0]|checkpoint_step_10|
|`checkpoints/rmbench_episode0_smoke_10step_screen_20260915`|2026-09-15 23:11:16 → 2026-09-15 23:12:48|0|blocks_ranking_try_50demo_w193_apf16_official；[0]|无|

早期指标仅按文件呈现，不推断结束状态。`rearrange_blocks_episode0_fullctx_1000step_20260917` 虽名 1000，仅有 74 行指标，日志时间范围与指标不完整匹配，不能称完成1000步；`rmbench_episode0_official_overfit_1k/run_config.json`仅保留 base、world=3、batch=1、累积=1、lr=1e-5 等，数据版本/完整历史源码 unknown。`train_out/training_metrics.jsonl` 是空文件，`train_out/checkpoints/` 未见权重，不新增为训练运行。

## 最终日志损失（不是新 forward）

|运行|训练 video/action|验证 video/action|
|---|---|---|
|blocks_bounded_3000step_20260924_224653|0.09386855 / 0.00096678|0.10702931 / 0.00122421|
|blocks_bounded_formal_20260922_214515|0.09157181 / 0.00172671|0.11236389 / 0.00163396|
|blocks_bounded_pilot_20260922_161832|0.12862141 / 0.00391676|0.15632491 / 0.00458214|
|observe_50demo_1000step_20260924_06|0.08661348 / 0.01273708|0.15491598 / 0.00417039|
|rearrange_blocks_50demo_1000step_20260921_180241|0.09207477 / 0.00170414|0.12662366 / 0.00225358|
|rearrange_blocks_50demo_300step_20260921_104117|0.11329517 / 0.00177800|0.13297946 / 0.00342880|

各指标原件：对应运行 `training_metrics.jsonl` 最后行、`validation_metrics.jsonl` 最后行及 `training_summary.json`；JSON 清单保留第一/最后指标、日志片段及逻辑行号。损失随帧/掩码平均，不可跨任务简单比较。
