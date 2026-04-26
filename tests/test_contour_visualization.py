from __future__ import annotations

import unittest

from core.expr import Derivative, Norm, Power
from ns.variables import omega
from verifier import numeric
from verifier.contour import compute_contour_fields, select_frontier_candidates, slice_array, verification_progress


@unittest.skipUnless(numeric.np is not None, "numpy is required for contour diagnostics")
class ContourVisualizationTests(unittest.TestCase):
    def test_compute_contour_fields_for_norm_inequality(self) -> None:
        fields = {
            "omega": numeric.make_field("two_scale", 1.0, 2, 0, 12, phase=0.0),
            "u": numeric.make_field("two_scale", 0.7, 1, 17, 12, phase=0.37),
        }
        sample = numeric.FieldSample("two_scale", 1.0, 2, 0, fields)
        lhs = Power(Norm(omega, 6), 2)
        rhs = Power(Norm(Derivative(omega, 1), 2), 2)

        contour_fields = compute_contour_fields(lhs, rhs, sample)

        self.assertEqual(contour_fields.lhs_density.shape, (12, 12, 12))
        self.assertEqual(contour_fields.rhs_density.shape, (12, 12, 12))
        self.assertGreater(contour_fields.aggregates["rhs_mean"], 0.0)
        self.assertTrue(numeric.np.isfinite(contour_fields.log_ratio).all())

    def test_slice_array_defaults_to_middle_slice(self) -> None:
        field = numeric.np.zeros((6, 8, 10))

        sliced = slice_array(field, axis="z")

        self.assertEqual(sliced.shape, (6, 8))


class FrontierCandidateTests(unittest.TestCase):
    def test_select_frontier_candidates_prefers_furthest_gate_and_dedupes(self) -> None:
        candidates = [
            {
                "candidate_id": "a",
                "queue_status": "failed_symbolic",
                "score": 200,
                "lhs_text": "A",
                "rhs_text": "B",
                "proof_rules": ["R1"],
            },
            {
                "candidate_id": "b",
                "queue_status": "failed_relevance",
                "score": 10,
                "lhs_text": "A",
                "rhs_text": "B",
                "proof_rules": ["R1"],
            },
            {
                "candidate_id": "c",
                "queue_status": "failed_numeric",
                "score": 5,
                "lhs_text": "C",
                "rhs_text": "D",
                "proof_rules": ["R2"],
            },
        ]

        selected = select_frontier_candidates(candidates, limit=3)

        self.assertEqual([candidate["candidate_id"] for candidate in selected], ["c", "b"])
        self.assertGreater(verification_progress(selected[0]), verification_progress(selected[1]))

    def test_select_frontier_candidates_dedupes_same_inequality_across_proofs(self) -> None:
        candidates = [
            {
                "candidate_id": "a",
                "queue_status": "failed_relevance",
                "score": 10,
                "lhs_text": "A",
                "rhs_text": "B",
                "proof_rules": ["R1"],
            },
            {
                "candidate_id": "b",
                "queue_status": "failed_relevance",
                "score": 9,
                "lhs_text": "A",
                "rhs_text": "B",
                "proof_rules": ["R2"],
            },
        ]

        selected = select_frontier_candidates(candidates, limit=3)

        self.assertEqual([candidate["candidate_id"] for candidate in selected], ["a"])


if __name__ == "__main__":
    unittest.main()
