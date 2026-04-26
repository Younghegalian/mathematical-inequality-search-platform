"""Beam-search runner for inequality rules."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from core.canonical import fingerprint
from core.closure import classify_closure
from core.filters import basic_filter
from core.rewrite import apply_rule_deep
from core.rules import Rule, default_rules
from core.scoring import score as score_expr
from engine.beam import select_top_k
from engine.recording import RunRecorder
from engine.state import SearchState
from ns.targets import TargetSpec


@dataclass(frozen=True)
class SearchResult:
    target: TargetSpec
    states: list[SearchState]
    run_path: Path | None = None
    summary_path: Path | None = None

    def most_informative(self) -> SearchState:
        classified = [
            state
            for state in self.states
            if state.closure is not None and state.closure.status in {"GOOD", "BAD"}
        ]
        if classified:
            return sorted(classified, key=lambda item: (-len(item.steps), -item.score))[0]
        return sorted(self.states, key=lambda item: (-item.score, -len(item.steps)))[0]


class SearchRunner:
    def __init__(
        self,
        rules: list[Rule] | None = None,
        width: int = 20,
        depth: int = 2,
        recorder: RunRecorder | None = None,
    ) -> None:
        self.rules = rules if rules is not None else default_rules()
        self.width = width
        self.depth = depth
        self.recorder = recorder

    def _annotate(self, state: SearchState, target: TargetSpec) -> SearchState:
        closure = classify_closure(
            state.rhs,
            dissipation=target.dissipation,
            controlled_norm=target.controlled_norm,
        )
        return SearchState(
            lhs=state.lhs,
            rhs=state.rhs,
            steps=state.steps,
            closure=closure,
            score=score_expr(state.rhs, closure),
        )

    def run(self, target: TargetSpec) -> SearchResult:
        initial = self._annotate(SearchState(target.expr, target.expr), target)
        beam = [initial]
        all_states = [initial]
        seen = {fingerprint(initial.rhs)}

        if self.recorder is not None:
            self.recorder.record_start(
                target,
                width=self.width,
                depth=self.depth,
                rules=[rule.name for rule in self.rules],
            )
            self.recorder.record_initial_state(initial)

        for step_depth in range(1, self.depth + 1):
            candidates: list[SearchState] = []

            for state in beam:
                for rule in self.rules:
                    for result in apply_rule_deep(state.rhs, rule):
                        decision = basic_filter(target.expr, result.expression)
                        if not decision.keep:
                            if self.recorder is not None:
                                self.recorder.record_transition(
                                    depth=step_depth,
                                    parent=state,
                                    rule_name=result.rule_name,
                                    note=result.note,
                                    rhs_after=result.expression,
                                    filter_keep=False,
                                    filter_reason=decision.reason,
                                    child=None,
                                    status="rejected",
                                )
                            continue

                        next_state = state.extend(result.rule_name, result.expression, result.note)
                        next_state = self._annotate(next_state, target)
                        key = fingerprint(next_state.rhs)
                        if key in seen:
                            if self.recorder is not None:
                                self.recorder.record_transition(
                                    depth=step_depth,
                                    parent=state,
                                    rule_name=result.rule_name,
                                    note=result.note,
                                    rhs_after=result.expression,
                                    filter_keep=True,
                                    filter_reason=decision.reason,
                                    child=next_state,
                                    status="duplicate",
                                )
                            continue

                        seen.add(key)
                        candidates.append(next_state)
                        all_states.append(next_state)
                        if self.recorder is not None:
                            self.recorder.record_transition(
                                depth=step_depth,
                                parent=state,
                                rule_name=result.rule_name,
                                note=result.note,
                                rhs_after=result.expression,
                                filter_keep=True,
                                filter_reason=decision.reason,
                                child=next_state,
                                status="kept",
                            )

            if not candidates:
                break
            beam = select_top_k(candidates, self.width)
            if self.recorder is not None:
                self.recorder.record_beam(depth=step_depth, beam=beam)

        run_path = self.recorder.run_path if self.recorder is not None else None
        summary_path = self.recorder.summary_path if self.recorder is not None else None
        result = SearchResult(target=target, states=all_states, run_path=run_path, summary_path=summary_path)
        if self.recorder is not None:
            self.recorder.write_summary(target, result.most_informative(), all_states)
        return result
