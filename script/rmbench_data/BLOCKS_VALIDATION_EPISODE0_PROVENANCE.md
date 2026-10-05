# Blocks Ranking Try validation：episode 0 的来源与处理过程

本文以 `/root/wyc/data/rmbench/lingbot_va2/blocks_ranking_try_5demo_validation_fullctx_apf16_cf8` 中的 `episode_000000` 为例，记录从仿真场景到模型实际读取张量的完整链路。数据和脚本路径均指服务器。以下数值已经逐项读取文件核对。

## 1. 这条数据的身份

| 项目 | 实际值 |
|---|---|
| 任务 | `blocks_ranking_try`，配置 `demo_clean` |
| 数据集内编号 | `episode_index=0` |
| 仿真 seed | `100000` |
| 所属集合 | validation，共5条，seed `100000`–`100004` |
| 训练集合 | 另一份50条训练数据，原始 seed `0`–`49` |
| raw HDF5 | `/root/wyc/data/rmbench/lingbot_va2/holdout_work/blocks_ranking_try/validation/raw/data/episode0.hdf5` |
| 最终 APF16 数据集 | `/root/wyc/data/rmbench/lingbot_va2/blocks_ranking_try_5demo_validation_fullctx_apf16_cf8` |

`episode_index=0` 是 validation 数据集内部编号，不是训练集的 episode 0；真正用于区分场景的是 seed `100000`。来源记录见 `holdout_manifest.json`、`holdout_work/.../expert_seeds.json` 和 `collection_manifest.json`。

该 raw HDF5 的 SHA-256 是 `d636bb34453e89013e2b07c4a7fe3ae0f1967ac1fd2bb58ed0777e6deab08d31`，与 `holdout_manifest.json` 记录一致。采集时 expert planner 报告成功，随后按保存的关节路径重新回放并录制 HDF5。`initial_state_sha256` 为 `null`：当前记录能够证明 seed 不重叠、轨迹不同，但**没有独立的初始状态哈希对比证据**；不要把清单中的 `initial_states_unique=true` 解读为已逐条验证初始状态唯一。

## 2. 原始专家轨迹如何产生

总入口是：

```bash
cd /root/wyc/code/lingbot-va2
/root/miniconda3/envs/lingbot/bin/python \
  script/rmbench_data/prepare_rmbench_holdouts.py \
  --task blocks_ranking_try --split validation --gpu 0
```

这个入口依次调用：

1. RMBench 的 `tools/collect_expert_seeds.py`：从 seed `100000` 开始，用官方任务环境和 expert planner 找到5个可完成任务的场景；实际本批 `100000`–`100004` 都成功。
2. LingBot 仓库的 `script/rmbench_data/collect_rmbench_holdout_experts.py`：以选定 seed 运行规划，保存专家关节路径，再以 `need_plan=False, save_data=True` 回放，生成 HDF5、语言标注和 instruction 文件。

相关输入和输出：

```text
输入：/root/wyc/code/RMBench/task_config/demo_clean.yml
      /root/wyc/code/RMBench/envs/blocks_ranking_try.py
      /root/wyc/data/rmbench/lingbot_va2/blocks_ranking_try_50demo_fullctx_apf16_cf8/meta/episodes.jsonl
      （从训练集读与训练完全相同的任务文本）

输出：holdout_work/blocks_ranking_try/validation/expert_seeds.json
      holdout_work/blocks_ranking_try/validation/collection_manifest.json
      holdout_work/blocks_ranking_try/validation/raw/data/episode0.hdf5
      holdout_work/blocks_ranking_try/validation/raw/instructions/episode0.json
      holdout_work/blocks_ranking_try/validation/raw/language_annotation.json
```

episode 0 的 raw HDF5 有1030个物理帧。双臂末端位姿 `endpose/{left,right}_endpose` 为 `[1030,7]`，夹爪各为 `[1030]`；头部、左右腕部相机的 JPEG 字节帧各为1030个。raw 还存有前方相机、关节动作和相机标定，但这几个字段不进入本次 LingBot 三相机训练视图。

原始数据的生成命令由总入口自动调用；不要直接在已存在的 `raw` 目录上重跑，因为采集器会报 `FileExistsError`，以免覆盖已有数据。

## 3. raw HDF5 转为 LeRobot v2.1

脚本为 `script/rmbench_data/convert_rmbench_to_lingbot_lerobot.py`，本批采用 `--fps 15 --window-frames 4097`。转换器从 HDF5 取三路图像：

| LeRobot 键 | raw HDF5 键 |
|---|---|
| `observation.images.cam_high` | `/observation/head_camera/rgb` |
| `observation.images.cam_left_wrist` | `/observation/left_camera/rgb` |
| `observation.images.cam_right_wrist` | `/observation/right_camera/rgb` |

动作从双臂末端位姿和夹爪拼成16维：`[L_xyz,L_quat_xyzw,L_gripper,R_xyz,R_quat_xyzw,R_gripper]`。raw quaternion 是 WXYZ，转换为 XYZW。时间对应是 `image[t] -> action=physical EEF[t+1]`，故1030个 raw 帧变为1029行 LeRobot 数据。对这条数据，Parquet 动作与 raw 末端位姿下一帧的最大绝对差为 `2.98e-08`。

中间 LeRobot 根目录：

```text
/root/wyc/data/rmbench/lingbot_va2/holdout_work/blocks_ranking_try/validation/lerobot/
```

episode 0 对应：

```text
data/chunk-000/episode_000000.parquet
videos/chunk-000/observation.images.cam_high/episode_000000.mp4
videos/chunk-000/observation.images.cam_left_wrist/episode_000000.mp4
videos/chunk-000/observation.images.cam_right_wrist/episode_000000.mp4
meta/episodes.jsonl
conversion_manifest.json
```

视频是15 fps、320×240 的 AV1。`meta/episodes.jsonl` 中 episode 0 的 `length=1029`，唯一窗口为 `[start_frame=0,end_frame=1029)`。`--window-frames 4097` 是容纳完整轨迹的窗口上限；这条轨迹没有被切成多窗口。`meta/info.json` 的 `splits.train="0:5"` 是 LeRobot 的内部元数据命名，不能据此认定这些数据参与模型训练；实际用途由单独的 validation 目录和训练配置决定。

## 4. APF16 full-context 视图

入口调用 `script/rmbench_data/prepare_rmbench_apf16_dataset.py`，参数：

```bash
/root/miniconda3/envs/lingbot/bin/python \
  script/rmbench_data/prepare_rmbench_apf16_dataset.py \
  --source /root/wyc/data/rmbench/lingbot_va2/holdout_work/blocks_ranking_try/validation/lerobot \
  --output /root/wyc/data/rmbench/lingbot_va2/blocks_ranking_try_5demo_validation_fullctx_apf16_cf8 \
  --raw-root /root/wyc/data/rmbench/lingbot_va2/holdout_work/blocks_ranking_try/validation/raw/data \
  --regenerate-window-size 4097
```

它把 LeRobot 的 `data/`、`videos/` 链接进最终目录，复制 `meta/`，生成 `observation_anchors.json` 与 `continuous_contract.json`。因此**不要删除 `holdout_work/.../lerobot`**：最终数据集的视频和 Parquet 依赖这两个符号链接。

episode 0 的 `observation_anchors.json` 将双臂相对位姿原点固定为 raw 物理帧0。模型取样时（`wan_va/dataset/lerobot_latent_dataset.py`）将动作位移、旋转变为相对于该原点的动作，随后按训练集统计值归一化，扩展到模型30通道；非动作通道和无效时间位置的 mask 为0。

此条 episode 的 APF16 对齐数字：

| 项目 | 数值 |
|---|---:|
| LeRobot 观察/动作行 | 1029 |
| 单窗口 | `[0,1029)` |
| 原始可用动作候选 | 1028（物理帧1–1028） |
| 每4个源视频帧取1帧 | 257帧，ID `0,4,...,1024` |
| VAE 压缩后 latent 帧 | 65 |
| 每 latent 帧 action slots | 16 |
| 总 action slots | 1040 |
| 首 latent 帧的 condition slots | 16，mask=0 |
| 进入 loss 的有效动作 | 1024（物理帧1–1024） |
| 窗口末尾排除动作 | 4 |

这里的1024不是 raw 轨迹只有1024个动作，而是 `65×16−16=1024` 个有效槽位上限。末尾4个动作因窗口对齐被裁掉，不是漏处理数据或生成失败。

`rmbench_action_stats.json` 从50条训练集复制，没有使用 validation 数据重估。当前 validation 文件和训练集文件 SHA-256 均为 `1528f6168e79ad86846a86f465927519fb3b7a759d5046c5198d8276c1142d50`。

## 5. 三路相机 latent 与文本 embedding

脚本为 `script/rmbench_data/extract_rmbench_latents.py`，实际参数：

```bash
cd /root/wyc/code/lingbot-va2
CUDA_VISIBLE_DEVICES=0 /root/miniconda3/envs/lingbot/bin/python \
  script/rmbench_data/extract_rmbench_latents.py \
  --dataset-root /root/wyc/data/rmbench/lingbot_va2/blocks_ranking_try_5demo_validation_fullctx_apf16_cf8 \
  --model-root /root/wyc/code/lingbot-va2/checkpoints/base \
  --device cuda:0 --frame-step 4 --chunk-frames 8 \
  --episode-start 0 --episode-end 5
```

它使用 base checkpoint 的 tokenizer/text encoder 编码任务文本与空文本，写入 `empty_emb.pt`；使用 VAE 将三路视频分别编码。`chunk-frames 8` 是流式 VAE 分块大小，不是把训练 episode 切成8帧窗口。对 episode 0，三个文件均为 `episode_000000_0_1029.pth`：

| 相机 | 路径相对数据集根目录 | latent 形状 |
|---|---|---|
| 主视角 | `latents/chunk-000/observation.images.cam_high/episode_000000_0_1029.pth` | `[20800,48] = 65×16×20×48` |
| 左腕 | `latents/chunk-000/observation.images.cam_left_wrist/episode_000000_0_1029.pth` | `[5200,48] = 65×8×10×48` |
| 右腕 | `latents/chunk-000/observation.images.cam_right_wrist/episode_000000_0_1029.pth` | `[5200,48] = 65×8×10×48` |

每个 `.pth` 还包含 `text_emb`（`[512,4096]`）、文本、`frame_ids`、源/抽帧 fps、latent 空间尺寸和窗口起止帧。原始15 fps 抽帧后为3.75 fps。数据集总计5个窗口、15个相机 latent 文件；`latent_extraction_manifest.json` 记录模型路径与提取参数。

## 6. 审计结果与局限

运行：

```bash
/root/miniconda3/envs/lingbot/bin/python \
  script/rmbench_data/audit_full_context_dataset.py \
  --dataset /root/wyc/data/rmbench/lingbot_va2/blocks_ranking_try_5demo_validation_fullctx_apf16_cf8 \
  --expected-episodes 5
```

当前 `full_context_latent_audit.json` 为 `PASS`：5 episodes、15个 latent 文件、每条一个从0开始的窗口，`frame_ids` 与 latent 帧数一致。该通用脚本做结构检查，**不负责**验证 expert 场景是否成功、raw→Parquet 数值一致、训练/验证 seed 不重叠或归一化统计相同；上述项目需要结合 manifest、哈希和专项数值核对。本文对 episode 0 已做 raw→Parquet 数值检查。

## 7. 后续补充数据：生成独立新批次

当前5条数据是固定目录。原入口的 `collect`、`convert` 阶段遇到已存在的 raw/LeRobot/APF16 目录会拒绝执行。补充时使用新批次名和未用过的 seed 区间，生成单独的数据集；保留原 validation 不动。例如再生成5条 Blocks validation：

```bash
cd /root/wyc/code/lingbot-va2
/root/miniconda3/envs/lingbot/bin/python \
  script/rmbench_data/prepare_rmbench_holdouts.py \
  --task blocks_ranking_try \
  --split validation \
  --count 5 \
  --start-seed 300000 \
  --batch-name extra01 \
  --gpu 0
```

新输出会是：

```text
/root/wyc/data/rmbench/lingbot_va2/holdout_work/blocks_ranking_try/validation_extra01/
/root/wyc/data/rmbench/lingbot_va2/blocks_ranking_try_5demo_validation_extra01_fullctx_apf16_cf8/
```

可以用 `--count 1` 构造单条新 batch；必须给它独立 `--batch-name` 和 seed。追加到**同一个**5条数据集目录需要额外做 LeRobot 合并、重新编号、重建 `meta/info.json`、action anchor、latent manifest 和审计，不能简单复制 `episode5.hdf5` 或 `.pth`。当前训练脚本只读取一个 validation 根目录；新 batch 建好后也不会自动进入早停，需要另行合并或扩展训练配置。这一点在启动正式训练前应再次核查。

分阶段执行时可追加 `--phase seeds|collect|convert|latents|audit`；中断后先查看对应目录状态，再从尚未完成的阶段继续。`latents` 阶段会跳过已有 `.pth`，所以该阶段可恢复；`collect` 和 `convert` 阶段不支持在同名已有目录内无损续跑。不要给转换器传 `--overwrite` 指向现有正式 validation 根目录。
