#!/bin/bash
set -euo pipefail

REPO=/root/wyc/code/lingbot-va2
RUN_ROOT=/root/wyc/code/lingbot-va2/checkpoints/observe_blocks_episode0_300step_20260919_151950
OBS_CKPT="$RUN_ROOT/observe_and_pickup/checkpoints/checkpoint_step_300"
BLOCKS_CKPT="$RUN_ROOT/blocks_ranking_try/checkpoints/checkpoint_step_300"
INFER_SCRIPT="${REPO}/script/rmbench_inference/run_observe_blocks_inference.sh"
GENERATE_SCRIPT="${REPO}/script/rmbench_data/generate_rearrange_blocks_50demo_fullctx_latents.sh"

# 循环次数，按需修改
LOOP_TIMES=3

# 前置检查：失败立即退出，不浪费循环时间
test -d "${RUN_ROOT}"
test -s "${OBS_CKPT}/transformer/diffusion_pytorch_model.safetensors"
test -s "${BLOCKS_CKPT}/transformer/diffusion_pytorch_model.safetensors"
test -x "${INFER_SCRIPT}"
test -x "${GENERATE_SCRIPT}"

# 提前检查 run_observe_blocks_inference.sh 内部写死的数据集是否存在
# （blocks 数据集名与 generate 脚本产出的 50demo 版不一致，需手动确认）
OBS_DATASET=/root/wyc/data/rmbench/lingbot_va2/observe_and_pickup_50demo_fullctx_apf16_cf8
BLOCKS_DATASET=/root/wyc/data/rmbench/lingbot_va2/blocks_ranking_try_episode0_fullctx_apf16_cf8
test -s "${OBS_DATASET}/rmbench_action_stats.json"
test -s "${OBS_DATASET}/empty_emb.pt"
test -s "${OBS_DATASET}/latents/chunk-000/observation.images.cam_high/episode_000000_0_145.pth"
test -s "${BLOCKS_DATASET}/rmbench_action_stats.json"
test -s "${BLOCKS_DATASET}/empty_emb.pt"
test -s "${BLOCKS_DATASET}/latents/chunk-000/observation.images.cam_high/episode_000000_0_1033.pth"

# WandB 默认 offline，避免无网/无 key 时直接失败；外部可 export WANDB_MODE=online 覆盖
export WANDB_MODE="${WANDB_MODE:-online}"

for ((i=1; i<=LOOP_TIMES; i++)); do
    STAMP=$(date +%Y%m%d_%H%M%S)
    OUTPUT_ROOT="${REPO}/artifacts/observe_blocks_inference_overfit_1ep_faithful_round${i}_${STAMP}"
    echo "==================== 第 $i 轮推理开始 $(date) 输出=${OUTPUT_ROOT} ===================="
    "${INFER_SCRIPT}" \
      "$OBS_CKPT" \
      "$BLOCKS_CKPT" \
      1 \
      overfit \
      faithful \
      "${OUTPUT_ROOT}"
    echo "==================== 第 $i 轮推理结束 $(date) ===================="
done

echo -e "\n>>>>>>>>>> 所有推理循环执行完毕，开始执行generate脚本 <<<<<<<<<<<<"
bash "${GENERATE_SCRIPT}" all 0
echo "全部任务完成 $(date)"
