"""RMBench training defaults; dataset files are loaded only after config selection."""
import os

from easydict import EasyDict

from .va_robotwin_train_cfg import va_robotwin_train_cfg


va_rmbench_train_cfg = EasyDict(__name__="Config: VA RMBench train")
va_rmbench_train_cfg.update(va_robotwin_train_cfg)

va_rmbench_train_cfg.dataset_path = os.environ.get(
    "RMBENCH_DATASET_PATH",
    "/root/wyc/data/rmbench/lingbot_va2/blocks_ranking_try_50demo_w193_apf16_official",
)
va_rmbench_train_cfg.empty_emb_path = os.path.join(
    va_rmbench_train_cfg.dataset_path, "empty_emb.pt"
)
va_rmbench_train_cfg.wan22_pretrained_model_name_or_path = os.environ.get(
    "LINGBOT_PRETRAINED_MODEL", "/root/wyc/code/lingbot-va2/checkpoints/base"
)
va_rmbench_train_cfg.dataset_adapter = "rmbench"
va_rmbench_train_cfg.enable_wandb = True
va_rmbench_train_cfg.fsdp_cpu_offload = True
va_rmbench_train_cfg.num_steps = 3000
va_rmbench_train_cfg.save_interval = 3000
va_rmbench_train_cfg.seed = 42
va_rmbench_train_cfg.run_name = "rmbench-50demo-formal"
va_rmbench_train_cfg.enable_early_stop = os.environ.get("RMBENCH_EARLY_STOP", "1") == "1"
va_rmbench_train_cfg.validation_dataset_path = os.environ.get("RMBENCH_VALIDATION_DATASET_PATH")
va_rmbench_train_cfg.early_stop_min_steps = int(os.environ.get("RMBENCH_EARLY_STOP_MIN_STEPS", 300))
va_rmbench_train_cfg.early_stop_check_interval = int(os.environ.get("RMBENCH_EARLY_STOP_INTERVAL", 50))
va_rmbench_train_cfg.early_stop_patience = int(os.environ.get("RMBENCH_EARLY_STOP_PATIENCE", 3))
va_rmbench_train_cfg.early_stop_min_relative_improvement = float(
    os.environ.get("RMBENCH_EARLY_STOP_MIN_RELATIVE_IMPROVEMENT", 0.01)
)
