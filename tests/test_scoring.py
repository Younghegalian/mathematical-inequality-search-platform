from __future__ import annotations

import unittest

from core.closure import ClosureResult
from core.scoring import nonlinear_relevance_score, score
from ns.targets import OMEGA_L6_SQUARED, VORTEX_STRETCHING, omega_h1


class ScoringTest(unittest.TestCase):
    def test_nonlinear_relevance_beats_pure_good_closure(self) -> None:
        pure_good = score(
            OMEGA_L6_SQUARED.dissipation,
            ClosureResult("GOOD", 0, "endpoint embedding closes"),
            target=OMEGA_L6_SQUARED,
            lhs=OMEGA_L6_SQUARED.expr,
            steps=["Sobolev"],
        )
        nonlinear_bad = score(
            omega_h1,
            ClosureResult("BAD", 3, "superlinear but nonlinear-term relevant"),
            target=VORTEX_STRETCHING,
            lhs=VORTEX_STRETCHING.expr,
            steps=["Holder", "Biot-Savart", "Gagliardo-Nirenberg", "Young"],
        )

        self.assertGreater(nonlinear_bad, pure_good)

    def test_relevance_detects_nonlinear_target_and_rules(self) -> None:
        relevance = nonlinear_relevance_score(
            lhs=VORTEX_STRETCHING.expr,
            rhs=omega_h1,
            target_name=VORTEX_STRETCHING.name,
            proof_rules=["Holder", "Biot-Savart"],
        )

        self.assertGreaterEqual(relevance, 120)


if __name__ == "__main__":
    unittest.main()
