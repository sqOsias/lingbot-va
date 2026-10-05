import tempfile
import unittest
from pathlib import Path

import torch

from .numeric_reference import compare, summarize


class NumericReferenceTest(unittest.TestCase):
    def test_capture_and_compare(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "reference.pt"
            equal = root / "equal.pt"
            changed = root / "changed.pt"
            payload = {
                "action": torch.arange(6, dtype=torch.float32).reshape(2, 3),
                "cache": [torch.tensor([1, 2], dtype=torch.int64)],
            }
            torch.save(payload, reference)
            torch.save(payload, equal)
            torch.save({**payload, "action": payload["action"] + 0.1}, changed)

            summary = summarize(reference)
            self.assertEqual(set(summary["tensors"]), {"action", "cache[0]"})
            self.assertTrue(summary["tensors"]["action"]["all_finite"])
            self.assertTrue(compare(reference, equal, atol=0.0, rtol=0.0)["passed"])
            self.assertFalse(compare(reference, changed,
                                     atol=1e-6, rtol=1e-6)["passed"])


if __name__ == "__main__":
    unittest.main()
