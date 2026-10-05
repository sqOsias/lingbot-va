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
va_rmbench_train_cfg.fsdp_cpu_offload = os.environ.get(
    "RMBENCH_FSDP_CPU_OFFLOAD", "1") == "1"
# 0 preserves legacy whole-world FSDP. On four ranks, shard_size=2 selects
# two replicated two-rank shard groups (2D HSDP).
va_rmbench_train_cfg.fsdp_shard_size = int(os.environ.get(
    "RMBENCH_FSDP_SHARD_SIZE", 0))
va_rmbench_train_cfg.fsdp_reshard_after_forward = os.environ.get(
    "RMBENCH_FSDP_RESHARD_AFTER_FORWARD", "1") == "1"
va_rmbench_train_cfg.fsdp_root_reshard_after_forward = os.environ.get(
    "RMBENCH_FSDP_ROOT_RESHARD_AFTER_FORWARD", "1") == "1"
# va_rmbench_train_cfg.load_worker = 4
va_rmbench_train_cfg.num_steps = 3000
va_rmbench_train_cfg.save_interval = int(os.environ.get(
    "RMBENCH_SAVE_INTERVAL", 3000))
va_rmbench_train_cfg.load_worker = int(os.environ.get(
    "RMBENCH_LOAD_WORKER", va_rmbench_train_cfg.load_worker))
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
# Opt-in only. Existing Rearrange/Observe/full-context runs are unchanged.
va_rmbench_train_cfg.bounded_window_index = os.environ.get("RMBENCH_BOUNDED_TRAIN_INDEX")
va_rmbench_train_cfg.bounded_validation_index = os.environ.get("RMBENCH_BOUNDED_VALIDATION_INDEX")

# MemoryWAM is opt-in.  Legacy runs return the original four dataset fields and
# use the unchanged Transformer forward when these variables are unset.
va_rmbench_train_cfg.memorywam_enabled = os.environ.get(
    "RMBENCH_MEMORYWAM_ENABLED", "0") == "1"
va_rmbench_train_cfg.memory_alignment_path = os.environ.get(
    "RMBENCH_MEMORY_ALIGNMENT_PATH")
va_rmbench_train_cfg.validation_memory_alignment_path = os.environ.get(
    "RMBENCH_VALIDATION_MEMORY_ALIGNMENT_PATH")
va_rmbench_train_cfg.memorywam_gist_tokens = int(os.environ.get(
    "RMBENCH_MEMORYWAM_GIST_TOKENS", 8))
va_rmbench_train_cfg.memorywam_anchor_frames = int(os.environ.get(
    "RMBENCH_MEMORYWAM_ANCHOR_FRAMES", 2))
va_rmbench_train_cfg.memorywam_recent_frames = int(os.environ.get(
    "RMBENCH_MEMORYWAM_RECENT_FRAMES", 4))
va_rmbench_train_cfg.memorywam_mode = os.environ.get(
    "RMBENCH_MEMORYWAM_MODE", "hybrid")
va_rmbench_train_cfg.memorywam_detach_interval = int(os.environ.get(
    "RMBENCH_MEMORYWAM_DETACH_INTERVAL", 0))
