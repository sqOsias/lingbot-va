# LingBot-VA / MemoryWAM 迁移到本地电脑操作文档

日期：2026-10-05
代码仓库：`https://github.com/Robbyant/lingbot-va`
代码分支：`codex/memorywam-mechanism-v1`

## 1. 迁移边界

GitHub 只保存源码、测试、RMBench 适配脚本、操作文档和一个可审计的 RMBench
补丁。以下内容不会提交到 GitHub：

- `checkpoints/`、`train_out/` 中的模型权重；
- `/root/wyc/data/rmbench/lingbot_va2/` 中的数据集；
- `artifacts/`、W&B 日志、运行日志和 Python cache；
- 本地 MemoryWAM 论文 PDF；
- Git credential、环境文件或 API token。

因此 GitHub clone 本身不能开始 5B 测试。至少还要传输 step-750 初始化权重和
Rearrange 的训练 fullctx；正式训练还需要独立验证集，闭环推理还需要 base 模型、
RMBench 和 holdout seeds。

## 2. 已核实的资产清单

以下数字来自服务器 2026-10-05 的只读盘点。`du` 显示值可能因文件系统略有差异，
表中的 bytes 和文件数用于迁移后结构核验。

| 用途 | 服务器路径 | bytes / 文件数 | 是否必需 | 公开来源与结论 |
| --- | --- | ---: | --- | --- |
| 当前初始化 Transformer | `train_out/rearrange_blocks_50demo_1000step_20260921_180241/checkpoints/checkpoint_step_750` | 10,177,840,309 / 2 | 所有完整模型测试和训练必需 | 本地训练产物；无已确认公开地址，必须从服务器传输 |
| LingBot-VA base | `checkpoints/base` | 24,382,340,104 / 20 | latent 重提取和推理必需；只训练已有 latent 时不需要 | `robbyant/lingbot-va-base` 可公开下载，但未记录服务器 snapshot revision；严格复现优先传服务器副本 |
| train fullctx | `/root/wyc/data/rmbench/lingbot_va2/rearrange_blocks_50demo_fullctx_apf16_cf8` | 693,228,290 / 161 | 训练必需 | 50 episodes、150 latent 文件；本地派生数据，必须从服务器传输 |
| validation fullctx | `/root/wyc/data/rmbench/lingbot_va2/rearrange_blocks_5demo_validation_fullctx_apf16_cf8` | 73,123,489 / 26 | 正式训练门禁必需 | 5 episodes、15 latent 文件；必须从服务器传输 |
| test fullctx | `/root/wyc/data/rmbench/lingbot_va2/rearrange_blocks_5demo_test_fullctx_apf16_cf8` | 73,169,514 / 26 | teacher-forced 独立测试或闭环阶段建议 | 5 episodes、15 latent 文件；必须从服务器传输 |
| Rearrange holdout | `/root/wyc/data/rmbench/lingbot_va2/holdout_work/rearrange_blocks` | 345,347,153 / 100 | 闭环 validation/test 必需 | 每个 split 含 5 个 expert-validated seeds、原始 HDF5 和转换元数据 |
| RMBench 工作区 | `/root/wyc/code/RMBench` | 30,738,487,144 / 6,105 | 闭环环境必需 | 上游 commit `6d36e28ccce86710627edac26b5899bd5b6994c2`；服务器另有 8 个 tracked 文件改动 |

step-750 中真正的权重文件为：

```text
transformer/config.json                                      449 bytes
transformer/diffusion_pytorch_model.safetensors       10,177,831,668 bytes
```

base 模型包含三片 text encoder、三片 Transformer、VAE、tokenizer 和配置。首个
MemoryWAM 训练从 step-750 的 Transformer 初始化，并直接读取离线 latent/text
embedding，因此不需要把 base Transformer 再复制一份到训练 initializer。闭环服务
会把新 checkpoint 的 `transformer/` 与 base 的 `vae/`、`tokenizer/`、
`text_encoder/`、`configuration.json` 组合成完整模型目录。

## 3. 本地目录布局

以下命令使用可自行修改的目录，不要求在本地创建 `/root/wyc`：

```bash
export LOCAL_WORK="$HOME/work"
export LINGBOT_VA_REPO="$LOCAL_WORK/lingbot-va2"
export RMBENCH_ROOT="$LOCAL_WORK/RMBench"
export RMBENCH_DATA_ROOT="$LOCAL_WORK/data/rmbench/lingbot_va2"
```

最终目录应为：

```text
$LOCAL_WORK/
├── lingbot-va2/
│   ├── checkpoints/base/                         # 推理/重提取时需要
│   └── train_out/rearrange_blocks_50demo_1000step_20260921_180241/
│       └── checkpoints/checkpoint_step_750/transformer/
├── RMBench/                                      # 闭环时需要
└── data/rmbench/lingbot_va2/
    ├── rearrange_blocks_50demo_fullctx_apf16_cf8/
    ├── rearrange_blocks_5demo_validation_fullctx_apf16_cf8/
    ├── rearrange_blocks_5demo_test_fullctx_apf16_cf8/   # 闭环阶段
    └── holdout_work/rearrange_blocks/                    # 闭环阶段
```

## 4. 下载 GitHub 代码

在本地执行：

```bash
mkdir -p "$LOCAL_WORK"
git clone --branch codex/memorywam-mechanism-v1 --single-branch \
  https://github.com/Robbyant/lingbot-va.git "$LINGBOT_VA_REPO"
cd "$LINGBOT_VA_REPO"
git status --short --branch
```

若已经 clone：

```bash
cd "$LINGBOT_VA_REPO"
git fetch origin codex/memorywam-mechanism-v1
git switch codex/memorywam-mechanism-v1
git pull --ff-only
```

## 5. 从服务器传输权重和数据

在本地终端设置服务器 SSH 地址。不要把用户名、IP 或私钥写入 Git：

```bash
export SERVER='your_user@your_server'
mkdir -p "$LINGBOT_VA_REPO/train_out" "$LINGBOT_VA_REPO/checkpoints"
mkdir -p "$RMBENCH_DATA_ROOT"
```

### 5.1 最小训练资产

```bash
rsync -aHvhP --partial --info=progress2 \
  "$SERVER:/root/wyc/code/lingbot-va2/train_out/rearrange_blocks_50demo_1000step_20260921_180241" \
  "$LINGBOT_VA_REPO/train_out/"

rsync -aHvhP --partial --info=progress2 \
  "$SERVER:/root/wyc/data/rmbench/lingbot_va2/rearrange_blocks_50demo_fullctx_apf16_cf8" \
  "$RMBENCH_DATA_ROOT/"

rsync -aHvhP --partial --info=progress2 \
  "$SERVER:/root/wyc/data/rmbench/lingbot_va2/rearrange_blocks_5demo_validation_fullctx_apf16_cf8" \
  "$RMBENCH_DATA_ROOT/"
```

这里复制了原运行目录内可能存在的少量日志；若只想传 10 GB initializer，可以把
第一个源路径改为其 `checkpoints/checkpoint_step_750` 子目录，并在本地创建相同层级。

### 5.2 推理所需 base 模型

严格保持服务器版本：

```bash
rsync -aHvhP --partial --info=progress2 \
  "$SERVER:/root/wyc/code/lingbot-va2/checkpoints/base" \
  "$LINGBOT_VA_REPO/checkpoints/"
```

也可从公开 Hugging Face 仓库下载：

```bash
huggingface-cli download robbyant/lingbot-va-base \
  --local-dir "$LINGBOT_VA_REPO/checkpoints/base"
```

公开下载方式没有绑定服务器当时使用的 revision，不能直接宣称与历史运行逐字节
一致。需要严格回归时使用 `rsync` 的服务器副本。

### 5.3 闭环额外资产

```bash
rsync -aHvhP --partial --info=progress2 \
  "$SERVER:/root/wyc/data/rmbench/lingbot_va2/rearrange_blocks_5demo_test_fullctx_apf16_cf8" \
  "$RMBENCH_DATA_ROOT/"

rsync -aHvhP --partial --info=progress2 \
  "$SERVER:/root/wyc/data/rmbench/lingbot_va2/holdout_work/rearrange_blocks" \
  "$RMBENCH_DATA_ROOT/holdout_work/"
```

## 6. RMBench 代码选择

闭环有两种迁移方式。

### 6.1 推荐：clone 固定上游 commit，再应用本仓库补丁

```bash
git clone https://github.com/RoboTwin-Platform/RMBench.git "$RMBENCH_ROOT"
git -C "$RMBENCH_ROOT" checkout 6d36e28ccce86710627edac26b5899bd5b6994c2
gzip -dc "$LINGBOT_VA_REPO/docs/patches/rmbench_local_changes_6d36e28.patch.gz" \
  | git -C "$RMBENCH_ROOT" apply --check
gzip -dc "$LINGBOT_VA_REPO/docs/patches/rmbench_local_changes_6d36e28.patch.gz" \
  | git -C "$RMBENCH_ROOT" apply
```

补丁 SHA-256（先校验 gzip 文件，再校验解压内容）：

```text
gzip: 05cb74cc85d3a3f15a68d2eabe3547afcb623fe1fb141c5ea1cce30d75945985
raw:  bb1cdcc7c769ca8cfdfee334a5ed1c955f88e2f71691c513fbb2cc7d3dadf90b
```

补丁包含当前闭环实际依赖的 env、`policy/LingBot_VA` client/deploy 和
`script/eval_policy.py` 改动。服务器 RMBench 中其他未跟踪的实验 policy、日志和
debug 文件没有纳入补丁。

### 6.2 严格复制整个服务器 RMBench 工作区

服务器目录约 30.7 GB，并含数据或本地产物。只有需要逐文件保留现有环境时使用：

```bash
rsync -aHvhP --partial --info=progress2 \
  "$SERVER:/root/wyc/code/RMBench/" "$RMBENCH_ROOT/"
```

clone+补丁仍需按 RMBench 官方说明下载仿真 assets、创建环境；Git 代码本身不等于
完整仿真安装。

## 7. 传输完整性验证

首先运行仓库内的结构检查。该脚本不加载 tensor、不初始化 CUDA：

```bash
cd "$LINGBOT_VA_REPO"
python -m script.memorywam_checks.verify_migrated_assets \
  --scope training \
  --repo "$LINGBOT_VA_REPO" \
  --data-root "$RMBENCH_DATA_ROOT" \
  --output artifacts/memorywam/migrated_assets_training.json
```

加入 base 模型后：

```bash
python -m script.memorywam_checks.verify_migrated_assets \
  --scope inference \
  --repo "$LINGBOT_VA_REPO" \
  --data-root "$RMBENCH_DATA_ROOT" \
  --output artifacts/memorywam/migrated_assets_inference.json
```

闭环资产全部就位后：

```bash
python -m script.memorywam_checks.verify_migrated_assets \
  --scope closed-loop \
  --repo "$LINGBOT_VA_REPO" \
  --data-root "$RMBENCH_DATA_ROOT" \
  --rmbench-root "$RMBENCH_ROOT" \
  --output artifacts/memorywam/migrated_assets_closed_loop.json
```

结构检查不能发现同大小文件的内容损坏。每次 `rsync` 后可执行第二次 checksum dry
run；无文件列表输出才表示源端和本地内容一致：

```bash
rsync -aHnvc --delete \
  "$SERVER:/root/wyc/data/rmbench/lingbot_va2/rearrange_blocks_50demo_fullctx_apf16_cf8/" \
  "$RMBENCH_DATA_ROOT/rearrange_blocks_50demo_fullctx_apf16_cf8/"
```

对 step-750、validation、base 和 holdout 分别执行同样命令。`-c` 会读取所有文件，
大模型校验需要时间，但不会修改源端或本地文件。

## 8. Python 环境

官方仓库当前声明 Python 3.10.16、PyTorch 2.9.0、CUDA 12.6。GPU 电脑建议按
`INSTALL.md` 和 `README.md` 安装。训练额外需要 `lerobot==0.3.3`、SciPy 和 W&B。

```bash
conda create -n lingbot python=3.10.16 -y
conda activate lingbot
cd "$LINGBOT_VA_REPO"
pip install -e '.[dev]'
pip install lerobot==0.3.3 scipy wandb --no-deps
export PYTHON_BIN="$CONDA_PREFIX/bin/python"
```

`flash-attn` 应在匹配 CUDA/PyTorch 后单独安装。CPU 资产结构检查不需要
`flash-attn`；完整 CPU forward 会强制使用 `torch` attention。

## 9. 重建 alignment 和 CPU 门禁

设置统一路径：

```bash
export LINGBOT_VA_REPO="$LOCAL_WORK/lingbot-va2"
export RMBENCH_DATA_ROOT="$LOCAL_WORK/data/rmbench/lingbot_va2"
export PYTHON_BIN="$CONDA_PREFIX/bin/python"
export MEMORYWAM_INITIALIZER="$LINGBOT_VA_REPO/train_out/rearrange_blocks_50demo_1000step_20260921_180241/checkpoints/checkpoint_step_750"
cd "$LINGBOT_VA_REPO"
```

alignment 是 fullctx 的可重建 sidecar，不需要从服务器复制：

```bash
bash script/memorywam_checks/run_stage4_fullctx.sh \
  "$RMBENCH_DATA_ROOT/rearrange_blocks_50demo_fullctx_apf16_cf8" \
  artifacts/memorywam/stage4_alignment/rearrange_train

bash script/memorywam_checks/run_stage4_fullctx.sh \
  "$RMBENCH_DATA_ROOT/rearrange_blocks_5demo_validation_fullctx_apf16_cf8" \
  artifacts/memorywam/stage4_alignment/rearrange_validation
```

运行不加载 5B tensor 的训练 preflight：

```bash
bash script/memorywam_checks/run_training_cpu_preflight.sh
```

随后按顺序执行完整 5B CPU 检查：

```bash
bash script/memorywam_checks/run_full_model_cpu_sequence.sh
```

该序列固定执行 `plan -> load -> forward -> train-gist(2 steps)`，失败即停，不会运行
`train-all`。详情见 `docs/execution7.md`。

## 10. GPU 恢复后的首个训练

CPU 门禁全部通过后才运行 2-step GPU smoke：

```bash
cd "$LINGBOT_VA_REPO"
export LINGBOT_VA_REPO RMBENCH_DATA_ROOT PYTHON_BIN MEMORYWAM_INITIALIZER
export WANDB_MODE=disabled

NGPU=3 MASTER_PORT=29681 \
  bash script/rmbench_train/train_memorywam_rearrange.sh 2 \
  train_out/memorywam_smoke_3gpu
```

脚本要求有效的当前 CPU preflight，并在输出目录生成 `gpu_smoke_pass.json`。不要在
2-step 通过前启动长训练。三卡/四卡拓扑比较、5-step 计时和 50-step 工程验证见
`docs/MemoryWAM_current_progress_20261004.md`。

## 11. 哪些内容无需下载

- 不需要复制全部 `train_out/`；首实验只用指定的 step-750。
- 不需要复制所有旧 episode0、Observe、Blocks Ranking checkpoint。
- 只使用已有离线 latent 训练时，不需要原始 RGB/HDF5，也不需要 base VAE。
- alignment、CPU preflight 和其他 `artifacts/memorywam/` 输出可以在本地重建。
- W&B、历史日志和旧推理视频不影响初始化或数据语义。
- MemoryWAM gist 是新增参数，旧 checkpoint 中本来不存在；由当前代码初始化并训练。

## 12. 最小磁盘预算

- 训练最小资产约 10.2 GiB，另需 Python/CUDA 环境和新 checkpoint 输出空间。
- 加入 base 模型后额外约 22.7 GiB。
- test+holdout 额外约 0.39 GiB。
- 完整服务器 RMBench 工作区约 28.6 GiB；clone+补丁的源码更小，但仿真 assets 仍需
  按 RMBench 安装说明准备。

建议本地至少预留 80 GiB；若运行 full-backbone AdamW 或保留多个完整 checkpoint，
应预留更大的空间。
