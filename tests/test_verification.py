from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from engine.batch import run_batch
from engine.promotion import promote_good_candidates
from core.expr import Derivative, Norm, Power
from ns.variables import omega
from verifier.numeric import stress_candidate
from verifier.pipeline import run_verification_pipeline


class VerificationPipelineTest(unittest.TestCase):
    def test_pipeline_updates_promoted_candidate_status(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir) / "data"
            run_batch(
                budget=1,
                targets=["omega_L6_squared"],
                depths=[2],
                widths=[20],
                data_dir=data_dir,
            )
            promote_good_candidates(data_dir)

            result = run_verification_pipeline(data_dir)

            self.assertEqual(result["candidate_count"], 1)
            queue = json.loads(Path(result["queue_path"]).read_text(encoding="utf-8"))
            candidate = queue["candidates"][0]
            self.assertEqual(candidate["queue_status"], "failed_relevance")
            checks = candidate["verification"]["checks"]
            self.assertEqual([check["gate"] for check in checks], ["symbolic_replay", "scaling_audit", "target_relevance_check"])
            self.assertEqual(checks[0]["status"], "passed")
            self.assertEqual(checks[1]["status"], "passed")
            self.assertEqual(checks[2]["status"], "failed")
            self.assertTrue(Path(candidate["review_packet_path"]).exists())

            promoted_again = promote_good_candidates(data_dir)
            self.assertEqual(promoted_again["candidates"][0]["queue_status"], "failed_relevance")

    def test_numeric_stress_detects_frequency_growth(self) -> None:
        lhs = Power(Norm(Derivative(omega, 1), 2), 2)
        rhs = Power(Norm(omega, 2), 2)

        result = stress_candidate(lhs, rhs, grid_size=8, growth_limit=2)

        self.assertFalse(result["passed"])
        self.assertIn("grows", result["reason"])

    def test_numeric_stress_accepts_sobolev_shape(self) -> None:
        lhs = Power(Norm(omega, 6), 2)
        rhs = Power(Norm(Derivative(omega, 1), 2), 2)

        result = stress_candidate(lhs, rhs, grid_size=8)

        self.assertTrue(result["passed"])


if __name__ == "__main__":
    unittest.main()
