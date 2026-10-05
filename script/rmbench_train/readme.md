## blocks ranking try
bash script/rmbench_train/train_blocks_bounded.sh \
  3000 \
  /root/wyc/data/rmbench/lingbot_va2/blocks_ranking_try_bounded_v1_h32_f64

BLOCKS_CKPT=/root/wyc/code/lingbot-va2/train_out/blocks_bounded_formal_20260922_214515/checkpoints/checkpoint_step_1000
bash script/rmbench_inference/run_blocks_bounded_inference.sh \
  "$BLOCKS_CKPT" \
  train \
  1 \
  faithful