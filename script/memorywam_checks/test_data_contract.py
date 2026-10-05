from __future__ import annotations

import json
from pathlib import Path

import unittest

from .data_contract import build_episode_alignment


class DataContractTest(unittest.TestCase):
    def test_numbered_apf16_alignment(self) -> None:
        alignment = build_episode_alignment("e0", latent_frames=3,
                                            available_action_rows=32)
        self.assertEqual(len(alignment.decisions), 2)
        first, second = alignment.decisions
        self.assertEqual((first.decision_latent_id, first.target_latent_id,
                          first.target_action_group_id), (0, 1, 1))
        self.assertEqual((first.parquet_row_start, first.parquet_row_end), (0, 16))
        self.assertEqual((first.physical_target_start,
                          first.physical_target_end), (1, 16))
        self.assertEqual((second.parquet_row_start,
                          second.parquet_row_end), (16, 32))
        self.assertEqual((second.physical_target_start,
                          second.physical_target_end), (17, 32))

    def test_actual_episode0_metadata(self) -> None:
        cases = (
            ("rearrange_blocks_50demo_fullctx_apf16_cf8", 26, 400),
            ("observe_and_pickup_50demo_fullctx_apf16_cf8", 10, 144),
            ("blocks_ranking_try_50demo_fullctx_apf16_cf8", 65, 1024),
        )
        for dataset, expected_f, expected_valid in cases:
            with self.subTest(dataset=dataset):
                root = Path("/root/wyc/data/rmbench/lingbot_va2") / dataset
                if not root.exists():
                    self.skipTest("local RMBench data is unavailable")
                record = json.loads(
                    (root / "meta/episodes.jsonl").read_text().splitlines()[0])
                segment = record["action_config"][0]
                usable = segment["end_frame"] - segment["start_frame"] - 1
                alignment = build_episode_alignment(
                    f"{dataset}:0", latent_frames=expected_f,
                    available_action_rows=usable)
                self.assertEqual(alignment.supervised_action_rows,
                                 expected_valid)
                self.assertEqual(sum(
                    item.parquet_row_end - item.parquet_row_start
                    for item in alignment.decisions), expected_valid)
                self.assertEqual(alignment.decisions[-1].physical_target_end,
                                 expected_valid)
