"""Heuristic scores for search states."""

from __future__ import annotations

from core.closure import ClosureResult
from core.expr import Expr, Sum
from core.filters import complexity


def score(rhs: Expr, closure: ClosureResult) -> int:
    value = 0
    if closure.status == "GOOD":
        value += 100
    elif closure.status == "BAD":
        value += 25
    elif closure.status == "UNKNOWN":
        value += 0

    if isinstance(rhs, Sum):
        value += 10

    value -= complexity(rhs)
    return value
