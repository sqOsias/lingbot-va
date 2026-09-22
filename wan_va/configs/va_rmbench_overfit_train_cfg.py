"""Independent episode0 overfit run using the official LingBot trainer."""
from easydict import EasyDict

from .va_rmbench_train_cfg import va_rmbench_train_cfg


va_rmbench_overfit_train_cfg = EasyDict(__name__="Config: VA RMBench episode0 overfit")
va_rmbench_overfit_train_cfg.update(va_rmbench_train_cfg)
va_rmbench_overfit_train_cfg.episode_indices = [0]
va_rmbench_overfit_train_cfg.num_steps = 300
va_rmbench_overfit_train_cfg.save_interval = 300
va_rmbench_overfit_train_cfg.run_name = "rmbench-episode0-overfit"
