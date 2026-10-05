from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

import torch

from wan_va.configs import VA_CONFIGS
from wan_va.dataset import MultiLatentLeRobotDataset


class ProductionDatasetTest(unittest.TestCase):
    def test_alignment_is_a_lossless_opt_in_extension(self) -> None:
        root = Path(
            "/root/wyc/data/rmbench/lingbot_va2/"
            "rearrange_blocks_50demo_fullctx_apf16_cf8")
        sidecar = Path(
            "artifacts/memorywam/stage4_alignment/"
            "rearrange_train/episodes.jsonl")
        if not root.exists() or not sidecar.exists():
            self.skipTest("local fullctx dataset or generated sidecar unavailable")
        stats = json.loads((root / "rmbench_action_stats.json").read_text())

        def configuration(alignment=None):
            config = copy.deepcopy(VA_CONFIGS["rmbench_train"])
            config.dataset_path = str(root)
            config.empty_emb_path = str(root / "empty_emb.pt")
            config.episode_indices = [0]
            config.cfg_prob = 0.0
            config.norm_stat = {"q01": stats["q01"], "q99": stats["q99"]}
            config.memory_alignment_path = alignment
            return config

        legacy = MultiLatentLeRobotDataset(configuration(),
                                           num_init_worker=1)[0]
        aligned = MultiLatentLeRobotDataset(
            configuration(str(sidecar)), num_init_worker=1)[0]
        self.assertEqual(set(legacy), {
            "latents", "actions", "actions_mask", "text_emb"
        })
        for key in legacy:
            torch.testing.assert_close(legacy[key], aligned[key],
                                       atol=0.0, rtol=0.0)
        for key in (
                "dataset_numeric_id", "episode_index", "global_frame_ids",
                "decision_frame_ids", "target_frame_ids",
                "target_action_group_ids", "video_target_mask", "reset_mask"):
            self.assertIn(key, aligned)


if __name__ == "__main__":
    unittest.main()
