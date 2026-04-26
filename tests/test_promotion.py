from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from engine.batch import run_batch
from engine.promotion import promote_good_candidates


class PromotionTest(unittest.TestCase):
    def test_promote_good_candidates_writes_queue(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir) / "data"
            out_dir = Path(temp_dir) / "promotions"
            run_batch(
                budget=1,
                targets=["omega_L6_squared"],
                depths=[2],
                widths=[20],
                data_dir=data_dir,
            )

            result = promote_good_candidates(data_dir, out_dir=out_dir)

            self.assertGreaterEqual(result["candidate_count"], 1)
            queue_path = Path(result["queue_path"])
            jsonl_path = Path(result["jsonl_path"])
            self.assertTrue(queue_path.exists())
            self.assertTrue(jsonl_path.exists())
            queue = json.loads(queue_path.read_text(encoding="utf-8"))
            self.assertEqual(queue["schema_version"], "aisis.verification_queue.v1")
            self.assertEqual(queue["candidates"][0]["queue_status"], "pending_symbolic")


if __name__ == "__main__":
    unittest.main()
