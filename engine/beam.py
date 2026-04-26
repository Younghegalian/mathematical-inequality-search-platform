"""Beam selection helpers."""

from __future__ import annotations

from core.filters import complexity
from engine.state import SearchState


def select_top_k(states: list[SearchState], width: int) -> list[SearchState]:
    return sorted(states, key=lambda item: (-item.score, complexity(item.rhs), len(item.steps)))[:width]
