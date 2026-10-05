### launch

```
cd /root/wyc/code/lingbot-va2

unset BLOCKS_SMOKE_FORCE_EPISODE

export RMBENCH_DEBUGPY=1
export RMBENCH_DEBUGPY_PORT=5678
export WANDB_MODE=offline
export PYTHONFAULTHANDLER=1

bash script/rmbench_train/train_rmbench_50demo.sh \
  rearrange_blocks \
  2 \
  "/root/wyc/code/lingbot-va2/train_out/rearrange_debug_2step_$(date +%Y%m%d_%H%M%S)"
 ```