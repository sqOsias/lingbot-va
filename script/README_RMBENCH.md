# RMBench 脚本目录

`script/` 根目录保留 LingBot-VA 原有脚本。本项目新增的 RMBench 脚本按用途放在以下目录。

## `script/rmbench_data/`

数据准备、latent 提取、动作统计重算与数据/标签审计脚本。

- `prepare_rmbench_apf16_dataset.py`：建立 APF16 数据视图与窗口契约。
- `prepare_episode0_full_context.py`：建立 rearrange episode 0 full-context 数据视图。
- `extract_rmbench_latents.py`：提取三相机连续 VAE latent。
- `recompute_rmbench_episode_origin_stats.py`：重算 episode-origin 动作归一化统计。
- `audit_full_context_dataset.py`：与任务名、episode 长度无关的50条 full-context latent 完整性审计。
- `prepare_rmbench_holdouts.py`：用与训练种子不重叠的专家可解场景，为每个任务构造5条 validation 和5条 test，再转换、提取 full-context latent 并审计。
- `collect_rmbench_holdout_experts.py`、`convert_rmbench_to_lingbot_lerobot.py`：上述流程调用的专家轨迹采集与 LeRobot 转换工具。
- 其余 `audit_*.py`：早期 episode 0 数据对齐、mask、动作标签专项诊断，保留用于问题复现，不作为正式50条训练入口。

## `script/rmbench_train/`

训练预检、正式训练及训练指标辅助脚本。

- `preflight_distributed_train.py`：三卡真实2-step前反向预检。
- `train_observe_blocks_episode0_300step.sh`：串行训练 Observe 与 Blocks episode 0，各300 steps。
- `train_rearrange_episode0_fullctx.sh`：训练 Rearrange episode 0 full-context。
- `train_rmbench_50demo.sh`：三个任务统一正式训练入口，可指定任务和最大 optimizer steps；默认使用独立验证集和早停。
- `train_rmbench_readable.sh`、`run_two_trainings.sh`：早期训练入口，保留用于复现。
- `plot_loss_svg.py`、`sync_training_metrics_wandb.py`：训练指标工具。

## `script/rmbench_inference/`

在线/离线推理及按钮阶段诊断脚本。

- `run_observe_blocks_inference.sh`：串行评测 Observe 与 Blocks，支持 overfit/generalize 和可配置次数。
- `run_rearrange_episode0_fullctx_inference.sh`：Rearrange full-context faithful/fast 推理。
- `run_rearrange_episode0_*.sh`：早期 Rearrange 推理与复现实验。
- `infer_rearrange_episode0_offline.py`：离线推理。
- `probe_button_*.py`、`run_button_*.sh`：按钮阶段因果诊断。

新命令示例：

```bash
cd /root/wyc/code/lingbot-va2
./script/rmbench_train/train_observe_blocks_episode0_300step.sh
./script/rmbench_inference/run_observe_blocks_inference.sh OBS_CKPT BLOCKS_CKPT 1 overfit faithful

# 构造三个任务各5条 validation + 5条 test（资源密集，支持按任务/阶段运行）
./script/rmbench_data/prepare_rmbench_holdouts.py --task all --split all --gpu 0

# 2-step smoke test（训练集仍为完整50条；不足50步，不触发验证）
WANDB_MODE=disabled ./script/rmbench_train/train_rmbench_50demo.sh observe_and_pickup 2

# 正式训练；STEPS是上限，验证损失收敛可提前停止
./script/rmbench_train/train_rmbench_50demo.sh rearrange_blocks 3000
```

正式模式的50条 episode 全部参与梯度更新；独立的5条 validation 只用于早停，另5条 test 只留给最终评估。holdout 使用训练集的 normalization stats，避免验证/测试信息泄漏到归一化统计。`script/rmbench_inference/0920.sh` 是本机临时启动文件，已排除版本控制；正式入口均在上述三个子目录中。
