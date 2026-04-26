from __future__ import annotations

import unittest
from fractions import Fraction

from core.scaling import scaling
from engine.runner import SearchRunner
from ns.generated import STRAIN_STRAIN_VORTICITY, STRAIN_VORTICITY, VELOCITY_STRAIN_GRADIENT, critical_omega_norm_target
from ns.targets import OMEGA_L3, OMEGA_L6_SQUARED, VORTEX_STRETCHING


class BaselineSearchTest(unittest.TestCase):
    def test_scaling_matches_for_baseline_quantities(self) -> None:
        self.assertEqual(scaling(OMEGA_L3.expr), Fraction(3))
        self.assertEqual(scaling(OMEGA_L3.dissipation), Fraction(3))

    def test_gn_young_baseline_is_bad_closure(self) -> None:
        result = SearchRunner(depth=2, width=20).run(OMEGA_L3)
        state = result.most_informative()

        self.assertEqual([step.rule_name for step in state.steps], ["Gagliardo-Nirenberg", "Young"])
        self.assertIsNotNone(state.closure)
        self.assertEqual(state.closure.status, "BAD")
        self.assertEqual(state.closure.ode_power, Fraction(3))

    def test_sobolev_endpoint_is_good_closure(self) -> None:
        result = SearchRunner(depth=2, width=20).run(OMEGA_L6_SQUARED)
        state = result.most_informative()

        self.assertEqual([step.rule_name for step in state.steps], ["Sobolev"])
        self.assertIsNotNone(state.closure)
        self.assertEqual(state.closure.status, "GOOD")

    def test_vortex_stretching_reaches_standard_bad_closure(self) -> None:
        result = SearchRunner(depth=5, width=50).run(VORTEX_STRETCHING)
        state = result.most_informative()
        rule_names = [step.rule_name for step in state.steps]

        self.assertIn("Holder", rule_names)
        self.assertIn("Biot-Savart", rule_names)
        self.assertIn("Gagliardo-Nirenberg", rule_names)
        self.assertIn("Young", rule_names)
        self.assertIsNotNone(state.closure)
        self.assertEqual(state.closure.status, "BAD")
        self.assertEqual(state.closure.ode_power, Fraction(3))

    def test_generated_critical_l4_target_reaches_bad_closure(self) -> None:
        target = critical_omega_norm_target(Fraction(4))
        result = SearchRunner(depth=3, width=50).run(target)
        state = result.most_informative()

        self.assertEqual(scaling(target.expr), Fraction(3))
        self.assertIn("Gagliardo-Nirenberg", [step.rule_name for step in state.steps])
        self.assertIsNotNone(state.closure)
        self.assertEqual(state.closure.status, "BAD")

    def test_strain_vorticity_target_reaches_bad_closure(self) -> None:
        result = SearchRunner(depth=5, width=50).run(STRAIN_VORTICITY)
        state = result.most_informative()
        rule_names = [step.rule_name for step in state.steps]

        self.assertIn("Holder", rule_names)
        self.assertIn("Biot-Savart", rule_names)
        self.assertIn("Gagliardo-Nirenberg", rule_names)
        self.assertIn("Young", rule_names)
        self.assertIsNotNone(state.closure)
        self.assertEqual(state.closure.status, "BAD")

    def test_new_nonlinear_targets_reach_standard_bad_closure(self) -> None:
        for target in [VELOCITY_STRAIN_GRADIENT, STRAIN_STRAIN_VORTICITY]:
            with self.subTest(target=target.name):
                result = SearchRunner(depth=6, width=80).run(target)
                state = result.most_informative()
                rule_names = [step.rule_name for step in state.steps]

                self.assertIn("Holder", rule_names)
                self.assertIn("Biot-Savart", rule_names)
                self.assertIn("Gagliardo-Nirenberg", rule_names)
                self.assertIn("Young", rule_names)
                self.assertIsNotNone(state.closure)
                self.assertEqual(state.closure.status, "BAD")


if __name__ == "__main__":
    unittest.main()
