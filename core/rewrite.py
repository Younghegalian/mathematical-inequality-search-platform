"""Apply local inequality rules throughout positive expressions."""

from __future__ import annotations

from core.expr import Expr, Product, Sum
from core.rules import Rule, RuleResult


def apply_rule_deep(expr: Expr, rule: Rule) -> list[RuleResult]:
    """Apply a rule at the root and to safe algebraic children.

    AISIS treats sums and products of tracked quantities as positive, so replacing
    a term/factor by an upper bound preserves the inequality direction. We do
    not rewrite inside Norm, Derivative, or Integral here; rules for those forms
    should match them at the root.
    """

    results = list(rule.apply(expr))

    if isinstance(expr, Product):
        for index, term in enumerate(expr.terms):
            for child in apply_rule_deep(term, rule):
                terms = list(expr.terms)
                terms[index] = child.expression
                results.append(
                    RuleResult(
                        child.rule_name,
                        Product(terms),
                        f"{child.note}; rewritten in product factor {index}",
                    )
                )

    if isinstance(expr, Sum):
        for index, term in enumerate(expr.terms):
            for child in apply_rule_deep(term, rule):
                terms = list(expr.terms)
                terms[index] = child.expression
                results.append(
                    RuleResult(
                        child.rule_name,
                        Sum(terms),
                        f"{child.note}; rewritten in sum term {index}",
                    )
                )

    return results
