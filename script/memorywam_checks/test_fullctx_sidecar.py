import unittest

from .fullctx_sidecar import _decisions


class FullctxSidecarTest(unittest.TestCase):
    def test_numbered_contract_maps_physical_targets_and_parquet_rows(self) -> None:
        window = {
            "sampled_video_frame_ids": list(range(0, 33, 4)),
            "latent_frames": 3,
            "valid_action_physical_frames": [1, 32],
            "valid_actions": 32,
        }
        decisions = _decisions(window, action_per_latent=16,
                               vae_temporal_compression=4)
        self.assertEqual(len(decisions), 2)
        self.assertEqual(decisions[0]["source_rgb_index"], 0)
        self.assertEqual(decisions[0]["target_rgb_index"], 16)
        self.assertEqual(decisions[0]["parquet_row_start"], 0)
        self.assertEqual(decisions[0]["parquet_row_end"], 16)
        self.assertEqual(decisions[0]["physical_target_start"], 1)
        self.assertEqual(decisions[0]["physical_target_end"], 16)
        self.assertEqual(decisions[1]["parquet_row_start"], 16)
        self.assertEqual(decisions[1]["parquet_row_end"], 32)
        self.assertEqual(decisions[1]["physical_target_start"], 17)
        self.assertEqual(decisions[1]["physical_target_end"], 32)

    def test_partial_valid_range_is_explicit(self) -> None:
        window = {
            "sampled_video_frame_ids": list(range(0, 33, 4)),
            "latent_frames": 3,
            "valid_action_physical_frames": [1, 20],
            "valid_actions": 20,
        }
        decisions = _decisions(window, action_per_latent=16,
                               vae_temporal_compression=4)
        self.assertEqual([item["valid_action_count"] for item in decisions],
                         [16, 4])
        self.assertEqual(decisions[1]["parquet_row_start"], 16)
        self.assertEqual(decisions[1]["parquet_row_end"], 20)


if __name__ == "__main__":
    unittest.main()
