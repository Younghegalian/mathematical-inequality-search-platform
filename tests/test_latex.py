from __future__ import annotations

import unittest

from core.expr import Derivative, Norm, Power
from core.latex import inequality_to_latex, inequality_to_unicode
from ns.variables import omega


class LatexFormattingTest(unittest.TestCase):
    def test_formats_norm_derivative_inequality(self) -> None:
        lhs = Power(Norm(omega, 6), 2)
        rhs = Power(Norm(Derivative(omega, 1), 2), 2)

        rendered = inequality_to_latex(lhs, rhs)

        self.assertIn(r"\omega", rendered)
        self.assertIn(r"\nabla", rendered)
        self.assertIn(r"\leq", rendered)
        self.assertIn(r"\left\|", rendered)

    def test_formats_unicode_math_symbols(self) -> None:
        lhs = Power(Norm(omega, 6), 2)
        rhs = Power(Norm(Derivative(omega, 1), 2), 2)

        rendered = inequality_to_unicode(lhs, rhs)

        self.assertEqual(rendered, "‖ω‖₆² ≤ ‖∇ω‖₂²")


if __name__ == "__main__":
    unittest.main()
