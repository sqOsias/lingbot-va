"""RMBench inference binding; files are loaded only after config selection."""
import os
from easydict import EasyDict
from .va_robotwin_cfg import va_robotwin_cfg

va_rmbench_cfg=EasyDict(__name__="Config: VA RMBench inference")
va_rmbench_cfg.update(va_robotwin_cfg)
va_rmbench_cfg.infer_mode="server"
va_rmbench_cfg.fsdp_cpu_offload=True
va_rmbench_cfg.enable_offload=True
va_rmbench_cfg.wan22_pretrained_model_name_or_path=os.environ.get("RMBENCH_LINGBOT_MODEL","/root/wyc/code/lingbot-va2/checkpoints/base")
va_rmbench_cfg.rmbench_stats_path=os.environ.get("RMBENCH_LINGBOT_STATS","/root/wyc/data/rmbench/lingbot_va2/blocks_ranking_try_50demo_w193_apf16_official/rmbench_action_stats.json")
va_rmbench_cfg.seed=42
va_rmbench_cfg.enable_wandb=True
va_rmbench_cfg.num_inference_steps=int(os.environ.get(
    "RMBENCH_LINGBOT_VIDEO_INFERENCE_STEPS", va_rmbench_cfg.num_inference_steps))
va_rmbench_cfg.action_num_inference_steps=int(os.environ.get(
    "RMBENCH_LINGBOT_ACTION_INFERENCE_STEPS", va_rmbench_cfg.action_num_inference_steps))
