# MemoryWAM 阶段 4 操作文档：fullctx 时间 sidecar 与正式 Dataset 验证

日期：2026-10-04
适用仓库：`/root/wyc/code/lingbot-va2`
前置文档：[execution1.md](./execution1.md)
当前状态：本文记录阶段 4 当时的 sidecar 与隔离验证结果。生产 Dataset、Trainer、模型和服务随后已按默认关闭方式接入；最新状态、命令和限制见 [execution3.md](./execution3.md)。

## 1. 本阶段目的与资源要求

本阶段解决生产 MemoryWAM 接入前的数据身份问题：保留现有 fullctx RGB、latent、文本 embedding、动作 parquet 和统计文件，通过只读 sidecar 显式记录：

```text
episode ID
global latent frame ID
source RGB/environment frame ID
observed z[t] -> target z[t+1]
target action group
parquet row range
physical action target range
视频与动作有效 mask
episode reset
```

本阶段全部使用 CPU，不加载 5B，不启动训练或仿真，不改 `/root/wyc/data` 下的源数据。脚本逐文件处理，不把整个数据集同时载入内存。

GPU 只在本阶段通过后用于下一门禁“0B 当前 5B 数值 legacy reference”。该门禁是短时单卡推理，不是正式训练；显存需求以现有 `rmbench` 服务能够加载目标 checkpoint 的实际配置为准，本文不编造显存数值。

## 2. 输入、代码和产物

### 2.1 当前固定输入

```text
训练数据:
/root/wyc/data/rmbench/lingbot_va2/rearrange_blocks_50demo_fullctx_apf16_cf8

验证数据（下一项检查使用，不参与调参）:
/root/wyc/data/rmbench/lingbot_va2/rearrange_blocks_5demo_validation_fullctx_apf16_cf8

测试数据:
/root/wyc/data/rmbench/lingbot_va2/rearrange_blocks_5demo_test_fullctx_apf16_cf8
```

首轮只处理 train 和 validation。test 在最终评测协议、seed 和 checkpoint 选择规则固定后再检查，不用 test 调整实现。

### 2.2 本阶段代码

| 文件 | 功能 |
| --- | --- |
| `script/memorywam_checks/fullctx_sidecar.py` | 从真实合同和全部相机 latent 生成只读 episode/decision sidecar |
| `script/memorywam_checks/aligned_fullctx_dataset.py` | 隔离 wrapper；在不改变原模型输入张量的情况下增加身份、时间和 target 字段 |
| `script/memorywam_checks/trace_fullctx_dataset.py` | 实例化正式 `MultiLatentLeRobotDataset` 和 `DataLoader`，逐样本比较 sidecar、shape、mask 和数值 |
| `script/memorywam_checks/run_stage4_fullctx.sh` | 串行执行 sidecar、全量 Dataset trace 和 CPU 单元测试 |
| `script/memorywam_checks/test_fullctx_sidecar.py` | 带编号序列和部分尾部动作的映射测试 |
| `script/memorywam_checks/test_aligned_fullctx_dataset.py` | 验证新增元数据不会替换或修改原四类模型输入 |

### 2.3 输出目录

默认输出：

```text
artifacts/memorywam/stage4_alignment/rearrange_train/
├── manifest.json
├── episodes.jsonl
├── dataset_trace.json
└── dataset_trace_all.json
```

- `manifest.json`：输入合同哈希、episode/decision/action 总数、相机顺序和 latent 形状。
- `episodes.jsonl`：每个 episode/segment 一行，内部包含逐决策时间映射。
- `dataset_trace.json`：代表样本 trace。
- `dataset_trace_all.json`：全部正式 Dataset 样本 trace。

`artifacts/` 和 `docs/` 当前被仓库 `.gitignore` 忽略，文件会保留在工作区，但不会自然显示在 `git status` 中。

## 3. sidecar 如何确定时间关系

sidecar 不以目录名或 `end-start-1` 猜测有效动作。依据优先级是：

1. `continuous_contract.json`：有效 physical target 范围、RGB stride、VAE temporal compression、latent 数和动作 padding。
2. `meta/episodes.jsonl`：正式 Dataset 的 episode、segment 和 task text。
3. 每个相机 `.pth` 文件头：实际 `frame_ids`、latent 数、空间尺寸、文本和 segment bounds。

当前 APF16 的显式映射为：

```text
sampled RGB stride = 4 个环境动作
VAE temporal compression = 4 个 sampled RGB
1 latent step = 16 个环境动作

z[t] at physical frame 16t
  -> parquet rows [16t, 16(t+1))
  -> physical targets [16t+1, 16(t+1)]
  -> target z[t+1] at physical frame 16(t+1)
```

第 0 个 action latent group 是完整 dummy condition，因此其 16 个 slot 均无 loss。后续每个 action group 与 `target_global_frame_id` 和 `target_action_group_id` 明确绑定。若最后一组不足 16 个动作，sidecar 保存实际 `valid_action_count`。

代表 decision 记录：

```json
{
  "decision_global_frame_id": 0,
  "target_global_frame_id": 1,
  "target_action_group_id": 1,
  "source_rgb_index": 0,
  "target_rgb_index": 16,
  "parquet_row_start": 0,
  "parquet_row_end": 16,
  "physical_target_start": 1,
  "physical_target_end": 16,
  "valid_action_count": 16,
  "video_target_valid": true
}
```

## 4. 一键执行

从仓库根目录执行：

```bash
cd /root/wyc/code/lingbot-va2
script/memorywam_checks/run_stage4_fullctx.sh
```

显式指定输入和输出：

```bash
PYTHON_BIN=/root/miniconda3/envs/lingbot/bin/python \
script/memorywam_checks/run_stage4_fullctx.sh \
  /root/wyc/data/rmbench/lingbot_va2/rearrange_blocks_50demo_fullctx_apf16_cf8 \
  /root/wyc/code/lingbot-va2/artifacts/memorywam/stage4_alignment/rearrange_train
```

脚本固定 `CUDA_VISIBLE_DEVICES=''`、`OMP_NUM_THREADS=1` 和 `MKL_NUM_THREADS=1`。导入现有配置时 PyTorch 可能打印 CUDA unavailable warning；只要报告正常生成且命令退出码为 0，这不是 GPU 执行。

## 5. 分步执行与诊断

### 5.1 生成 sidecar

```bash
CUDA_VISIBLE_DEVICES='' \
/root/miniconda3/envs/lingbot/bin/python -B \
  -m script.memorywam_checks.fullctx_sidecar \
  --dataset-root /root/wyc/data/rmbench/lingbot_va2/rearrange_blocks_50demo_fullctx_apf16_cf8 \
  --output artifacts/memorywam/stage4_alignment/rearrange_train
```

脚本对所有 episode、所有 segment 和三路相机执行：

- `action_config` 与 continuous window bounds 一致。
- 三相机文件均存在。
- 实际 `frame_ids` 与合同完全一致。
- latent 数、展平 token 数与 `(F,H,W,C)` 一致。
- 三相机的时间、文本和 text embedding shape 一致。
- 逐决策有效动作和等于合同 `valid_actions`。
- 所有 global frame ID 能由 physical frame/APF16 精确得到。

任一检查失败，脚本立即非零退出，不生成“通过但带 warning”的 sidecar。

### 5.2 检查 sidecar 摘要

```bash
/root/miniconda3/envs/lingbot/bin/python -B - <<'PY'
import json
from pathlib import Path

root = Path("artifacts/memorywam/stage4_alignment/rearrange_train")
manifest = json.loads((root / "manifest.json").read_text())
print(json.dumps(manifest, indent=2, ensure_ascii=False))
print("episode records:", sum(1 for _ in (root / "episodes.jsonl").open()))
PY
```

当前实际结果：

```text
episodes       = 50
segments       = 50
decisions      = 1231
valid actions  = 19696
action/latent  = 16
camera count   = 3
source data modified = false
```

相机 latent 原始空间 shape 有两种：主相机 `[48,16,20]`，腕部相机 `[48,8,10]`。正式 loader 按现有 `robotwin_tshape` 路径组合后得到 `[48,F,24,20]`；这不是缺失或 shape 冲突。

### 5.3 检查代表性正式 Dataset 样本

```bash
CUDA_VISIBLE_DEVICES='' \
/root/miniconda3/envs/lingbot/bin/python -B \
  -m script.memorywam_checks.trace_fullctx_dataset \
  --dataset-root /root/wyc/data/rmbench/lingbot_va2/rearrange_blocks_50demo_fullctx_apf16_cf8 \
  --sidecar artifacts/memorywam/stage4_alignment/rearrange_train/episodes.jsonl \
  --index 0 --index 24 --index -1 \
  --output artifacts/memorywam/stage4_alignment/rearrange_train/dataset_trace.json
```

每个样本核验正式输入 keys、latent/action 的时间长度、逐 slot action mask、dummy/padding 零值、有限数值和 wrapper 字段的 DataLoader collate。

### 5.4 检查全部 50 个正式 Dataset 样本

```bash
CUDA_VISIBLE_DEVICES='' \
/root/miniconda3/envs/lingbot/bin/python -B \
  -m script.memorywam_checks.trace_fullctx_dataset \
  --dataset-root /root/wyc/data/rmbench/lingbot_va2/rearrange_blocks_50demo_fullctx_apf16_cf8 \
  --sidecar artifacts/memorywam/stage4_alignment/rearrange_train/episodes.jsonl \
  --all --quiet \
  --output artifacts/memorywam/stage4_alignment/rearrange_train/dataset_trace_all.json
```

本轮已实际执行，结果：

```text
samples checked          = 50/50
latent frame lengths     = 25 or 26
all finite               = true
all masked actions zero  = true
valid action slots sum   = 19696
sidecar/mask mismatch    = 0
```

### 5.5 运行隔离测试

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
/root/miniconda3/envs/lingbot/bin/python -B \
  -m script.memorywam_checks.run_checks
```

本轮结果：19 项全部通过，约 0.39 秒。包括此前的小模型 memory/mask/streaming/gist 检查以及新增 sidecar/wrapper 检查。

## 6. 当前发现及其含义

### 6.1 已通过

- Rearrange train50 的三相机 latent、合同、episode metadata 和正式 action mask 一致。
- 现有底层 RGB、latent、文本 embedding、动作和统计文件无需重建。
- 第 0 个动作组确实是 16-slot dummy；25/26 latent episode 分别有 384/400 个有效动作。
- sidecar wrapper 不复制、不重新归一化也不修改四类现有模型输入，只增加身份字段。

### 6.2 已确认的生产缺口

正式 Dataset batch 当前没有：

```text
episode_id
episode_index
global_frame_ids
decision_frame_ids
target_frame_ids
target_action_group_ids
reset_mask
```

隔离 wrapper 已证明这些字段可加入，形状示例为：

```text
global_frame_ids          [B,F]
decision_frame_ids        [B,F-1]
target_frame_ids          [B,F-1]
target_action_group_ids   [B,F-1]
video_target_mask         [B,F]
action_valid_slots        [B,F]
reset_mask                [B]
```

当前 `Trainer.convert_input_format` 对 batch 的每个 value 无条件调用 `.to(device)`。因此不能把字符串 `episode_id` 直接送入现有 Trainer；生产补丁必须改为仅移动 tensor，并把字符串 dataset/episode key 留在 CPU metadata 中，或者使用 `(dataset_numeric_id, episode_index)` 两个整数张量。

当前 `_prepare_input_dict` 只消费四类原字段。仅把 sidecar 字段加进 Dataset 还不会启用 memory，Trainer 需要显式把 `memory_meta` 传给模型。该修改属于后续 production default-off 阶段。

### 6.3 隔离 wrapper 不能直接成为最终生产实现

`AlignedFullctxDataset` 为了恢复样本身份读取了 `MultiLatentLeRobotDataset` 的 `_datasets/new_metas` 私有字段。它适合验证 sidecar，但生产应在 `LatentLeRobotDataset.__getitem__` 内直接返回当前 `cur_meta` 对应的身份，避免依赖私有映射。

## 7. validation 数据检查

训练数据通过后，用同一代码检查 validation；不读取 test 指标：

```bash
script/memorywam_checks/run_stage4_fullctx.sh \
  /root/wyc/data/rmbench/lingbot_va2/rearrange_blocks_5demo_validation_fullctx_apf16_cf8 \
  /root/wyc/code/lingbot-va2/artifacts/memorywam/stage4_alignment/rearrange_validation
```

本轮已实际执行并通过：5 个 episode、123 条 decision、1968 个有效动作；5/5 正式 Dataset 样本均有限，masked action 均为零，正式 mask 与 sidecar 无差异，源数据未修改。

train/validation 的来源独立性不能只靠它们各自从 0 开始的 episode index 判断。当前本地合同未保存稳定的原始 demonstration UUID，因此本阶段只能排除文件级结构错误；严格 split provenance 仍需回溯 holdout 生成 manifest。这是证据缺口。

## 8. 生产 Dataset 最小补丁顺序

sidecar 和 validation 检查都通过后，按以下顺序修改正式代码；一次只做数据身份，不同时改 attention 或服务：

1. `LatentLeRobotDataset.__init__` 可选读取 `memory_alignment_path`。
2. `LatentLeRobotDataset.__getitem__` 根据当前 `cur_meta` 直接附加数值身份字段。
3. `MultiLatentLeRobotDataset` 为多数据集场景添加 `dataset_numeric_id`，与 `episode_index` 共同构成唯一身份。
4. `Trainer.convert_input_format` 只对 tensor 调用 `.to(device)`。
5. `_prepare_input_dict` 将身份字段放入独立 `memory_meta`，memory disabled 时模型忽略它。
6. 运行 memory-off 数据回归，确认原四类 tensor 逐值不变。

建议接口：

```python
def convert_input_format(self, batch):
    return {
        key: value.to(self.device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }

input_dict["memory_meta"] = {
    "dataset_numeric_id": batch["dataset_numeric_id"],
    "episode_index": batch["episode_index"],
    "global_frame_ids": batch["global_frame_ids"],
    "decision_frame_ids": batch["decision_frame_ids"],
    "target_frame_ids": batch["target_frame_ids"],
    "target_action_group_ids": batch["target_action_group_ids"],
    "reset_mask": batch["reset_mask"],
}
```

`batch_size=1` 的完整 episode 可以直接 collate。后续若同一 batch 放入不同长度 episode/chunk，必须新增 padding collate 和对应 valid mask；不能依靠默认 collate。

## 9. 下一门禁：需要 GPU 的 5B legacy 数值 reference

完成 validation sidecar 和生产 Dataset 的 memory-off tensor 回归后，才运行 5B。目标 checkpoint：

```text
train_out/rearrange_blocks_50demo_1000step_20260921_180241/
checkpoints/checkpoint_step_750
```

需要固定当前 commit/diff、完整有效配置、单个 validation 输入、seed、初始 video/action noise、dtype、attention backend、denoise schedule、guidance、prompt、动作统计和 cache 状态。

服务诊断出口应保存只含 tensor/list/dict 的文件：

```python
torch.save({
    "observation_latent": observation_latent.detach().cpu(),
    "initial_video_noise": initial_video_noise.detach().cpu(),
    "final_video_latent": final_video_latent.detach().cpu(),
    "initial_action_noise": initial_action_noise.detach().cpu(),
    "model_action": model_action.detach().cpu(),
    "denormalized_action": denormalized_action.detach().cpu(),
    "cache_valid_count": torch.tensor(cache_valid_count),
    "frame_st_id": torch.tensor(frame_st_id),
}, "artifacts/memorywam/legacy_current/reference.pt")
```

生成摘要：

```bash
/root/miniconda3/envs/lingbot/bin/python -B \
  -m script.memorywam_checks.numeric_reference capture \
  --input artifacts/memorywam/legacy_current/reference.pt \
  --output artifacts/memorywam/legacy_current/reference.json
```

生产 memory 接入后关闭 memory，使用完全相同的初始噪声生成 candidate，再比较：

```bash
/root/miniconda3/envs/lingbot/bin/python -B \
  -m script.memorywam_checks.numeric_reference compare \
  --reference artifacts/memorywam/legacy_current/reference.pt \
  --candidate artifacts/memorywam/legacy_current/candidate.pt \
  --atol 1e-5 --rtol 1e-4 \
  --output artifacts/memorywam/legacy_current/compare.json
```

生产模型改造前必须得到 `passed=true` 的可复查 reference 流程。历史 rollout 不能替代这一门禁。

## 10. 停止条件

- sidecar 生成失败：先定位 metadata/latent/contract 最早不一致处，不修改训练 mask 掩盖问题。
- 正式 action mask 与 sidecar 不同：停止生产接入，检查 `_rmbench_action_post_process` 和合同来源。
- 原四类 tensor 在加入身份字段后改变：撤回 Dataset 补丁；身份改造必须是无损扩展。
- validation 来源无法证明独立：可以继续机制开发，但不能把其指标称为严格 held-out 泛化结果。
- 5B memory-off 数值回归失败：禁止开始 memory-on 训练，先定位 seed、backend、initial noise 或 legacy cache 生命周期。
