import json
import tempfile
import unittest
from pathlib import Path

from script.memorywam_checks.audit_multigpu_history import audit


class MultigpuHistoryAuditTest(unittest.TestCase):
    def test_world_size_timeout_and_legacy_timing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "example"
            run.mkdir()
            (run / "training.log").write_text(
                "World size: 3, Local rank: 0\n_ALLGATHER_BASE timeout\n")
            rows = [
                {"optimizer_step": 1, "elapsed_seconds": 10.0},
                {"optimizer_step": 2, "elapsed_seconds": 16.0},
            ]
            (run / "training_metrics.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows))
            report = audit(root)
            self.assertEqual(report["world_size_run_counts"], {"3": 1})
            self.assertFalse(report["has_four_rank_timing_evidence"])
            self.assertEqual(
                report["runs_with_collective_timeout_mentions"], 1)
            self.assertEqual(
                report["runs"][0]["median_elapsed_delta_seconds"], 6.0)


if __name__ == "__main__":
    unittest.main()
