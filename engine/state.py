"""Search state and proof-step records."""

from __future__ import annotations

from dataclasses import dataclass, field

from core.canonical import fingerprint
from core.closure import ClosureResult
from core.expr import Expr


@dataclass(frozen=True)
class ProofStep:
    rule_name: str
    before: Expr
    after: Expr
    note: str


@dataclass(frozen=True)
class SearchState:
    lhs: Expr
    rhs: Expr
    steps: tuple[ProofStep, ...] = field(default_factory=tuple)
    closure: ClosureResult | None = None
    score: int = 0

    def state_id(self) -> str:
        return fingerprint(self.rhs)

    def extend(self, rule_name: str, after: Expr, note: str) -> "SearchState":
        step = ProofStep(rule_name, self.rhs, after, note)
        return SearchState(self.lhs, after, self.steps + (step,))
