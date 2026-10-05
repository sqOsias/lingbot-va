import unittest

import torch

from .aligned_fullctx_dataset import attach_record_fields


class AlignedFullctxDatasetTest(unittest.TestCase):
    def test_explicit_identity_fields_do_not_change_model_inputs(self) -> None:
        sample = {
            "latents": torch.randn(48, 3, 2, 2),
            "actions": torch.randn(30, 3, 16, 1),
            "actions_mask": torch.ones(30, 3, 16, 1, dtype=torch.bool),
            "text_emb": torch.randn(4, 8),
        }
        record = {
            "episode_id": "dataset:7",
            "episode_index": 7,
            "segment_index": 0,
            "latent_frames": 3,
            "global_frame_ids": [0, 1, 2],
            "latent_source_rgb_indices": [0, 16, 32],
            "video_target_valid_by_latent": [False, True, True],
            "action_valid_slots_by_latent": [0, 16, 16],
            "decisions": [
                {
                    "decision_global_frame_id": 0,
                    "target_global_frame_id": 1,
                    "target_action_group_id": 1,
                    "parquet_row_start": 0,
                    "parquet_row_end": 16,
                    "physical_target_start": 1,
                    "physical_target_end": 16,
                },
                {
                    "decision_global_frame_id": 1,
                    "target_global_frame_id": 2,
                    "target_action_group_id": 2,
                    "parquet_row_start": 16,
                    "parquet_row_end": 32,
                    "physical_target_start": 17,
                    "physical_target_end": 32,
                },
            ],
        }
        result = attach_record_fields(sample, record)
        for key in ("latents", "actions", "actions_mask", "text_emb"):
            self.assertIs(result[key], sample[key])
        self.assertEqual(result["episode_id"], "dataset:7")
        torch.testing.assert_close(result["global_frame_ids"],
                                   torch.tensor([0, 1, 2]))
        torch.testing.assert_close(result["target_action_group_ids"],
                                   torch.tensor([1, 2]))
        self.assertTrue(bool(result["reset_mask"]))


if __name__ == "__main__":
    unittest.main()
