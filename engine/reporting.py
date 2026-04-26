"""Static HTML reports for AISIS search traces."""

from __future__ import annotations

import json
import hashlib
import math
import re
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from html import escape
from pathlib import Path
from typing import Any

from engine.indexing import fetch_index_summaries
from engine.promotion import promote_good_candidates


STATUS_CLASS = {
    "GOOD": "good",
    "BAD": "bad",
    "NEAR": "near",
    "UNKNOWN": "unknown",
    "NONE": "none",
    "kept": "kept",
    "rejected": "rejected",
    "duplicate": "duplicate",
}


@dataclass(frozen=True)
class RunReport:
    run_id: str
    path: Path
    rows: list[dict[str, Any]]
    summary: dict[str, Any] | None

    @property
    def start(self) -> dict[str, Any] | None:
        return next((row for row in self.rows if row.get("event") == "run_start"), None)

    @property
    def transitions(self) -> list[dict[str, Any]]:
        return [row for row in self.rows if row.get("event") == "transition"]

    @property
    def beam_events(self) -> list[dict[str, Any]]:
        return [row for row in self.rows if row.get("event") == "beam_selected"]

    @property
    def target_name(self) -> str:
        if self.summary:
            return str(self.summary.get("target_name", "unknown"))
        if self.start:
            return str(self.start.get("target_name", "unknown"))
        return "unknown"

    @property
    def search(self) -> dict[str, Any]:
        if self.start:
            return self.start.get("search", {})
        return {}

    @property
    def depth(self) -> int:
        return int(self.search.get("depth", 0))

    @property
    def width(self) -> int:
        return int(self.search.get("width", 0))

    @property
    def case_key(self) -> tuple[str, int, int]:
        return (self.target_name, self.depth, self.width)

    @property
    def best_state(self) -> dict[str, Any] | None:
        if self.summary:
            return self.summary.get("best_state")
        kept_children = [
            row.get("child_state")
            for row in self.transitions
            if row.get("status") == "kept" and row.get("child_state") is not None
        ]
        if not kept_children:
            return None
        return sorted(kept_children, key=lambda state: state.get("score", 0), reverse=True)[0]

    @property
    def best_closure(self) -> str:
        best = self.best_state
        if not best or not best.get("closure"):
            return "NONE"
        return str(best["closure"].get("status", "NONE"))

    @property
    def best_score(self) -> int:
        best = self.best_state
        if not best:
            return 0
        return int(best.get("score", 0))


@dataclass(frozen=True)
class IndexedRun:
    run_id: str
    path: Path
    summary: dict[str, Any]
    start: dict[str, Any] | None = None

    @property
    def target_name(self) -> str:
        return str(self.summary.get("target_name") or (self.start or {}).get("target_name", "unknown"))

    @property
    def search(self) -> dict[str, Any]:
        search = self.summary.get("search")
        if isinstance(search, dict) and search:
            return search
        if self.start:
            return self.start.get("search", {})
        return {}

    @property
    def depth(self) -> int:
        return int(self.search.get("depth", 0))

    @property
    def width(self) -> int:
        return int(self.search.get("width", 0))

    @property
    def case_key(self) -> tuple[str, int, int]:
        return (self.target_name, self.depth, self.width)

    @property
    def best_state(self) -> dict[str, Any] | None:
        best = self.summary.get("best_state")
        return best if isinstance(best, dict) else None

    @property
    def best_closure(self) -> str:
        best = self.best_state
        if not best or not best.get("closure"):
            return "NONE"
        return str(best["closure"].get("status", "NONE"))

    @property
    def best_score(self) -> int:
        best = self.best_state
        if not best:
            return 0
        return int(best.get("score", 0))

    @property
    def transition_count(self) -> int:
        return int(self.summary.get("transition_count", 0))

    @property
    def closure_histogram(self) -> Counter[str]:
        return Counter(self.summary.get("closure_histogram", {}))


def generate_reports(
    data_dir: Path,
    out_dir: Path,
    *,
    index_path: Path | None = None,
    max_run_pages: int = 250,
    scan_limit: int = 5000,
    promotion_limit: int = 1000,
) -> dict[str, Any]:
    source_index_path = index_path
    indexed_runs = load_indexed_runs(data_dir, index_path=index_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_dir = out_dir / "runs"
    run_dir.mkdir(parents=True, exist_ok=True)

    selected_ids = select_run_page_ids(indexed_runs, limit=max_run_pages)
    scan_ids = {run.run_id for run in indexed_runs} if len(indexed_runs) <= scan_limit else selected_ids
    scanned_runs = load_runs_by_ids(data_dir, scan_ids, indexed_runs)
    selected_runs = [run for run in scanned_runs if run.run_id in selected_ids]

    for run in selected_runs:
        (run_dir / f"{run.run_id}.html").write_text(render_run_report(run), encoding="utf-8")

    promotion = promote_good_candidates(data_dir, limit=promotion_limit)

    index_path = out_dir / "index.html"
    index_path.write_text(
        render_indexed_index(
            indexed_runs,
            scanned_runs,
            data_dir,
            generated_run_ids=selected_ids,
            promotion=promotion,
            exact_transition_scan=len(indexed_runs) <= scan_limit,
        ),
        encoding="utf-8",
    )

    return {
        "index_path": str(index_path),
        "run_report_count": len(indexed_runs),
        "generated_run_pages": len(selected_runs),
        "run_report_dir": str(run_dir),
        "promotion_queue": promotion["queue_path"],
        "promoted_candidates": promotion["candidate_count"],
        "index_source": str(source_index_path) if source_index_path and source_index_path.exists() else "summaries",
    }


def select_run_page_ids(runs: list[IndexedRun], *, limit: int) -> set[str]:
    if limit <= 0:
        return set()

    selected: list[str] = []

    def add(run: IndexedRun) -> None:
        if len(selected) >= limit:
            return
        if run.run_id not in selected:
            selected.append(run.run_id)

    for run in sorted(runs, key=lambda item: (item.best_closure != "GOOD", -item.best_score, item.run_id)):
        add(run)

    representatives = representative_runs(runs)
    for run, _ in representatives:
        add(run)

    for run in sorted(runs, key=lambda item: item.run_id, reverse=True):
        add(run)

    return set(selected)


def representative_runs(runs: list[IndexedRun | RunReport]) -> list[tuple[IndexedRun | RunReport, int]]:
    grouped: dict[tuple[str, int, int], list[IndexedRun | RunReport]] = {}
    for run in runs:
        grouped.setdefault(run.case_key, []).append(run)

    representatives: list[tuple[IndexedRun | RunReport, int]] = []
    for group in grouped.values():
        representative = sorted(group, key=lambda run: (-run.best_score, run.run_id))[0]
        representatives.append((representative, len(group)))
    return sorted(
        representatives,
        key=lambda item: (-item[1], item[0].target_name, item[0].depth, item[0].width, item[0].run_id),
    )


def load_runs(data_dir: Path) -> list[RunReport]:
    run_dir = data_dir / "runs"
    summary_dir = data_dir / "results"
    runs: list[RunReport] = []
    for path in sorted(run_dir.glob("*.jsonl")):
        rows = read_jsonl(path)
        if not rows:
            continue
        run_id = str(rows[0].get("run_id", path.stem))
        summary = load_summary(summary_dir, run_id)
        runs.append(RunReport(run_id=run_id, path=path, rows=rows, summary=summary))
    return runs


def load_indexed_runs(data_dir: Path, *, index_path: Path | None = None) -> list[IndexedRun]:
    if index_path is not None and index_path.exists():
        return load_indexed_runs_from_summaries(data_dir, fetch_index_summaries(index_path), read_missing_starts=False)

    run_dir = data_dir / "runs"
    summary_dir = data_dir / "results"
    summaries = []
    for summary_path in sorted(summary_dir.glob("summary_*.json")):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary.setdefault("_summary_path", str(summary_path))
        summaries.append(summary)
    return load_indexed_runs_from_summaries(data_dir, summaries, read_missing_starts=True)


def load_indexed_runs_from_summaries(
    data_dir: Path,
    summaries: list[dict[str, Any]],
    *,
    read_missing_starts: bool,
) -> list[IndexedRun]:
    run_dir = data_dir / "runs"
    runs: list[IndexedRun] = []
    for summary in summaries:
        summary_path = Path(str(summary.get("_summary_path", "")))
        fallback_id = summary_path.stem.removeprefix("summary_") if summary_path.name else "unknown"
        run_id = str(summary.get("run_id", fallback_id))
        run_path = Path(str(summary.get("run_path") or (run_dir / f"{run_id}.jsonl")))
        start = None
        if read_missing_starts and not summary.get("search"):
            start = read_run_start(run_path)
        runs.append(IndexedRun(run_id=run_id, path=run_path, summary=summary, start=start))
    return runs


def load_runs_by_ids(data_dir: Path, run_ids: set[str], indexed_runs: list[IndexedRun]) -> list[RunReport]:
    summary_dir = data_dir / "results"
    by_id = {run.run_id: run for run in indexed_runs}
    runs: list[RunReport] = []
    for run_id in sorted(run_ids):
        indexed = by_id.get(run_id)
        if indexed is None or not indexed.path.exists():
            continue
        rows = read_jsonl(indexed.path)
        if not rows:
            continue
        summary = indexed.summary or load_summary(summary_dir, run_id)
        runs.append(RunReport(run_id=run_id, path=indexed.path, rows=rows, summary=summary))
    return runs


def read_run_start(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("event") == "run_start":
                return row
            if row.get("event") != "state_initial":
                continue
    return None


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_summary(summary_dir: Path, run_id: str) -> dict[str, Any] | None:
    path = summary_dir / f"summary_{run_id}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def render_index(runs: list[RunReport], data_dir: Path) -> str:
    transitions = [transition for run in runs for transition in run.transitions]
    status_counts = Counter(transition.get("status", "unknown") for transition in transitions)
    action_counts = Counter(transition.get("action", "unknown") for transition in transitions)
    closure_counts = Counter(run.best_closure for run in runs)
    target_counts = Counter(run.target_name for run in runs)
    proof_paths = Counter(proof_path_text(run) for run in runs if proof_path_text(run))
    candidate_states = all_child_states(runs)
    case_counts = Counter(run.case_key for run in runs)
    repeated_run_count = sum(count - 1 for count in case_counts.values() if count > 1)

    cards = [
        ("Runs", len(runs)),
        ("Unique Cases", len(case_counts)),
        ("Repeated Runs", repeated_run_count),
        ("Transitions", len(transitions)),
        ("Kept", status_counts.get("kept", 0)),
        ("Rejected", status_counts.get("rejected", 0)),
        ("Duplicate Edges", status_counts.get("duplicate", 0)),
        ("Best GOOD", closure_counts.get("GOOD", 0)),
        ("Best BAD", closure_counts.get("BAD", 0)),
        ("Best UNKNOWN", closure_counts.get("UNKNOWN", 0)),
    ]

    run_rows, run_note = run_table_rows(runs, case_counts)

    body = f"""
    <section class="hero">
      <div>
        <p class="eyebrow">AISIS Search Report</p>
        <h1>Proof-Search Dataset Overview</h1>
        <p>Static dashboard generated from <code>{escape(str(data_dir))}</code>.</p>
      </div>
    </section>
    {card_grid(cards)}
    <section class="visual-banner">
      <div>
        <p class="eyebrow">Geometric Diagnostics</p>
        <h2>Absorption Geometry And Search Flow</h2>
        <p>The phase map colors Young absorbability by ODE power; points are actual Young transitions discovered in the trace.</p>
      </div>
    </section>
    {panel("Coverage Balance", case_coverage_panel(case_counts))}
    <section class="grid two visual-grid">
      {panel("Young Absorption Phase Map", young_phase_map(transitions))}
      {panel("Candidate Landscape", candidate_landscape(candidate_states))}
    </section>
    <section class="grid two visual-grid">
      {panel("Rule Transition Heatmap", rule_transition_heatmap(runs))}
      {panel("Closure Spectrum", closure_spectrum(candidate_states))}
    </section>
    <section class="grid two">
      {panel("Transition Statuses", bar_chart(status_counts))}
      {panel("Rule Actions", bar_chart(action_counts))}
    </section>
    <section class="grid two">
      {panel("Best Closure Per Run", bar_chart(closure_counts))}
      {panel("Targets", bar_chart(target_counts))}
    </section>
    {panel("Case Summary", case_summary_table(runs))}
    {panel("Common Proof Paths", ordered_counter(proof_paths))}
    {panel(
        "Representative Runs",
        run_note + table(
            ["Run", "Target", "Depth", "Width", "Repeats", "Best Closure", "Score", "Transitions", "Best RHS"],
            run_rows,
            raw_rows=True,
        ),
    )}
    """
    return page("AISIS Report", body)


def render_indexed_index(
    indexed_runs: list[IndexedRun],
    scanned_runs: list[RunReport],
    data_dir: Path,
    *,
    generated_run_ids: set[str],
    promotion: dict[str, Any],
    exact_transition_scan: bool,
) -> str:
    scanned_transitions = [transition for run in scanned_runs for transition in run.transitions]
    status_counts = aggregate_summary_counter(indexed_runs, "transition_status_histogram")
    action_counts = aggregate_summary_counter(indexed_runs, "action_histogram")
    if not status_counts:
        status_counts = Counter(transition.get("status", "unknown") for transition in scanned_transitions)
    if not action_counts:
        action_counts = Counter(transition.get("action", "unknown") for transition in scanned_transitions)

    closure_counts = Counter(run.best_closure for run in indexed_runs)
    proof_paths = Counter(proof_path_text(run) for run in indexed_runs if proof_path_text(run))
    case_counts = Counter(run.case_key for run in indexed_runs)
    repeated_run_count = sum(count - 1 for count in case_counts.values() if count > 1)
    transition_total = sum(run.transition_count for run in indexed_runs) or len(scanned_transitions)
    promising_count = len(promising_runs(indexed_runs))

    cards = [
        ("Runs", len(indexed_runs)),
        ("Promising", promising_count),
        ("Promoted GOOD", promotion.get("candidate_count", 0)),
        ("Unique Cases", len(case_counts)),
        ("Run Pages", len(generated_run_ids)),
        ("Transitions", transition_total),
        ("Best GOOD", closure_counts.get("GOOD", 0)),
        ("Best UNKNOWN", closure_counts.get("UNKNOWN", 0)),
    ]

    scanned_transition_counts = {run.run_id: len(run.transitions) for run in scanned_runs}
    run_rows, run_note = run_table_rows(
        indexed_runs,
        case_counts,
        generated_run_ids=generated_run_ids,
        transition_counts=scanned_transition_counts,
    )
    scan_note = ""
    if not exact_transition_scan:
        scan_note = (
            '<p class="viz-note">Large-scale mode: run detail pages and transition-heavy visuals are sampled. '
            'Summary-level counts remain exact for runs written with the current summary schema.</p>'
        )

    body = f"""
    <section class="hero">
      <div>
        <p class="eyebrow">AISIS Search Report</p>
        <h1>Candidate-Focused Search Report</h1>
        <p>Static dashboard generated from <code>{escape(str(data_dir))}</code>.</p>
      </div>
    </section>
    {scan_note}
    {card_grid(cards)}
    {panel("Project And Pipeline Guide", project_overview_panel(len(indexed_runs), len(case_counts), transition_total, promotion))}
    {panel("Massive Search Funnel", massive_search_funnel_panel(indexed_runs, status_counts, closure_counts, case_counts, promotion, transition_total))}
    {panel("Monte Carlo Frontier Map", monte_carlo_frontier_panel(indexed_runs, promotion))}
    {reading_guide_panel()}
    {panel("Survivor Queue Details", promotion_panel(promotion, generated_run_ids))}
    {details_panel(
        "Raw Diagnostics",
        '<section class="grid two">'
        + panel("Transition Filter Accounting", bar_chart(status_counts))
        + panel("Rule Actions", bar_chart(action_counts, limit=8))
        + '</section>'
        + panel("Rule Transition Heatmap", rule_transition_heatmap(indexed_runs))
        + panel("Common Proof Paths", ordered_counter(proof_paths, limit=6))
        + panel(
            "Drill-Down Runs",
            run_note + table(
                ["Run", "Target", "Depth", "Width", "Repeats", "Best Closure", "Score", "Transitions", "Best RHS"],
                run_rows[:12],
                raw_rows=True,
            ),
        ),
    )}
    """
    return page("AISIS Report", body)


def project_overview_panel(
    run_count: int,
    unique_cases: int,
    transition_total: int,
    promotion: dict[str, Any],
) -> str:
    candidates = list(promotion.get("candidates", []))
    promoted_count = int(promotion.get("candidate_count", 0))
    failed_relevance = sum(1 for item in candidates if str(item.get("queue_status", "")) == "failed_relevance")
    passed_numeric = sum(1 for item in candidates if str(item.get("queue_status", "")) in {"passed_numeric", "verified", "accepted"})

    if promoted_count == 0:
        verdict = "No candidates have reached the promotion queue yet. The useful next step is still broad exploration."
    elif passed_numeric:
        verdict = f"{format_count(passed_numeric)} candidate(s) have reached the post-numeric stage. Human mathematical review is now the bottleneck."
    elif failed_relevance:
        verdict = (
            f"{format_count(promoted_count)} candidate(s) reached promotion, and {format_count(failed_relevance)} currently stop at target relevance. "
            "They look closure-friendly, but they are still pure norm embeddings rather than nonlinear-term control statements. "
            "A cluster near the right edge is therefore a bias warning, not a reason to search only there."
        )
    else:
        verdict = f"{format_count(promoted_count)} candidate(s) are in the promotion queue and moving through verification."

    cards = [
        (
            "What AISIS searches for",
            "AISIS is not directly solving the PDE. It searches for inequality chains that could absorb nonlinear growth into dissipation plus a controlled remainder.",
        ),
        (
            "What each run does",
            "A run chooses a target, depth, width, and rule set, then applies transformations such as Holder, Sobolev, Gagliardo-Nirenberg, Young, and Biot-Savart.",
        ),
        (
            "Why the data matters",
            "The system stores both survivors and rejection points, so a future ML policy can learn which targets and rule orders are worth pushing.",
        ),
        (
            "What GOOD means",
            "GOOD only means the current closure classifier likes the form. It is not a theorem candidate until replay, scaling, relevance, numeric stress, and human review pass.",
        ),
    ]
    card_html = []
    for title, text in cards:
        card_html.append(
            f"""
            <div class="overview-card">
              <h3>{escape(title)}</h3>
              <p>{escape(text)}</p>
            </div>
            """
        )

    steps = [
        ("1", "Massive random samples", f"{format_count(run_count)} runs spread across target, depth, width, and search-seed settings."),
        ("2", "Case coverage", f"After repeated settings are folded together, the run set covers {format_count(unique_cases)} case groups."),
        ("3", "Rule expansion", f"The search produced {format_count(transition_total)} rule transitions. This stage can be larger than the run count because each run branches."),
        ("4", "Closure classifier", "Candidate right-hand sides are quickly labeled GOOD, BAD, or UNKNOWN under the current closure model."),
        ("5", "Dedupe + promotion", "Equivalent formulas and proof paths collapse to one candidate; only strong survivors move to the verification queue."),
        ("6", "Verification pipeline", "Candidates are checked by symbolic replay, scaling audit, target relevance, numeric stress, then human review."),
    ]
    step_html = []
    for number, title, text in steps:
        step_html.append(
            f"""
            <div class="overview-step">
              <span>{escape(number)}</span>
              <div>
                <strong>{escape(title)}</strong>
                <p>{escape(text)}</p>
              </div>
            </div>
            """
        )

    return f"""
    <section class="overview-stack">
      <div class="overview-lead">
        <p><strong>Short version:</strong> AISIS focuses on finding the critical inequalities that could make a Navier-Stokes regularity argument close.</p>
        <p>The target template is <code>T(u) &lt;= epsilon * D(u) + C * L(u)</code>. If the nonlinear growth term can be absorbed into dissipation and the remainder is controlled by a lower-order energy, the PDE question can reduce to an ODE-style boundedness argument.</p>
      </div>
      <div class="overview-grid">{''.join(card_html)}</div>
      <div class="overview-steps">{''.join(step_html)}</div>
      <div class="overview-verdict">
        <strong>Current verdict:</strong> {escape(verdict)}
      </div>
    </section>
    """


def process_overview_panel() -> str:
    stages = [
        (
            "1",
            "Target",
            "Build a nonlinear quantity such as ||omega||_p^q or an integral product.",
        ),
        (
            "2",
            "Rewrite",
            "Apply Holder, Sobolev, Gagliardo-Nirenberg, Biot-Savart, and Young rules.",
        ),
        (
            "3",
            "Score",
            "Classify whether the right side closes as dissipation plus controlled lower order terms.",
        ),
        (
            "4",
            "Promote",
            "Only GOOD or near-GOOD candidates stay visible; the rest becomes a density cloud.",
        ),
        (
            "5",
            "Verify",
            "Promoted candidates enter symbolic replay, scaling audit, numeric stress tests, and human review.",
        ),
    ]
    items = []
    for number, title, text in stages:
        items.append(
            f"""
            <div class="process-step">
              <div class="process-index">{escape(number)}</div>
              <div>
                <h3>{escape(title)}</h3>
                <p>{escape(text)}</p>
              </div>
            </div>
            """
        )
    return '<section class="process-flow">' + "\n".join(items) + "</section>"


def reading_guide_panel() -> str:
    cards = [
        (
            "What AISIS is doing",
            "AISIS is not solving Navier-Stokes directly. It searches for inequality chains that could turn a nonlinear term into absorbable dissipation plus a controlled remainder.",
        ),
        (
            "What GOOD means",
            "GOOD means the discovered right-hand side is closure-friendly under the current symbolic model. It is still only a candidate until replay, scaling, relevance, stress, and human checks pass.",
        ),
        (
            "What the dots mean",
            "Dots are a deterministic uniform sample of real runs. Highlighted promoted candidates are overlays, not extra evidence that the surrounding region is mathematically sufficient.",
        ),
        (
            "Where to look first",
            "Read Massive Search Funnel first. It tells you where the frontier is and which verification gate is blocking the surviving candidates.",
        ),
    ]
    items = []
    for title, text in cards:
        items.append(
            f"""
            <div class="guide-card">
              <h3>{escape(title)}</h3>
              <p>{escape(text)}</p>
            </div>
            """
        )
    return '<section class="guide-grid">' + "\n".join(items) + "</section>"


def massive_search_funnel_panel(
    runs: list[IndexedRun],
    status_counts: Counter[str],
    closure_counts: Counter[str],
    case_counts: Counter[tuple[str, int, int]],
    promotion: dict[str, Any],
    transition_total: int,
) -> str:
    run_count = len(runs)
    unique_cases = len(case_counts)
    repeated_runs = sum(count - 1 for count in case_counts.values() if count > 1)
    kept = int(status_counts.get("kept", 0))
    rejected = int(status_counts.get("rejected", 0))
    duplicate = int(status_counts.get("duplicate", 0))
    best_good = int(closure_counts.get("GOOD", 0))
    best_unknown = int(closure_counts.get("UNKNOWN", 0))
    best_bad = int(closure_counts.get("BAD", 0))
    promising_count = len(promising_runs(runs))
    promoted_count = int(promotion.get("candidate_count", 0))
    candidates = list(promotion.get("candidates", []))
    queue_status_counts = Counter(str(candidate.get("queue_status", "pending_symbolic")) for candidate in candidates)
    gate_counts = verification_gate_counts(queue_status_counts)
    progress_counts = verification_progress_counts(candidates, queue_status_counts)
    frontier = pipeline_frontier(promoted_count, gate_counts, promising_count, best_good, run_count)

    summary = (
        f'<p class="progress-summary"><strong>Frontier:</strong> {escape(frontier.rstrip(".") + ".")} '
        f'The search has produced {format_count(run_count)} sampled runs across {format_count(unique_cases)} case groups. '
        f'{format_count(best_good)} runs reached GOOD, but dedupe collapses them to {format_count(promoted_count)} promoted candidate(s). '
        'The gate cards below show exactly where the survivors passed, failed, or stopped.</p>'
    )

    gate_rows = [
        ("Random sample field", run_count, "broad target/depth/width draw", "done" if run_count else "empty"),
        ("Case coverage", unique_cases, f"{format_count(repeated_runs)} repeated runs", "done" if unique_cases else "empty"),
        (
            "Rule expansion",
            transition_total,
            f"{format_count(kept)} kept · {format_count(rejected)} rejected · {format_count(duplicate)} duplicate",
            "done" if transition_total else "empty",
        ),
        ("Closure classifier", best_good, f"GOOD · {format_count(best_unknown)} unknown · {format_count(best_bad)} bad", "done" if best_good else "watch"),
        ("Dedupe shortlist", promising_count, "unique promising forms", "done" if promising_count else "empty"),
        ("Promotion queue", promoted_count, "exported GOOD candidates", "done" if promoted_count else "empty"),
        gate_row("Symbolic replay", "symbolic_replay", "proof path replay", progress_counts),
        gate_row("Scaling audit", "scaling_audit", "critical-dimension check", progress_counts),
        gate_row("Target relevance", "target_relevance_check", "nonlinear-term usefulness", progress_counts),
        gate_row("Numeric stress", "numeric_counterexample_search", "counterexample search", progress_counts),
        gate_row("Human review", "human_math_review", "math review", progress_counts),
    ]

    return (
        summary
        + exploration_funnel_svg(runs, closure_counts, promotion)
        + gate_board(gate_rows)
        + candidate_gate_progress(candidates)
    )


def exploration_funnel_svg(
    runs: list[IndexedRun],
    closure_counts: Counter[str],
    promotion: dict[str, Any],
) -> str:
    width = 1080
    height = 500
    field_x = 36
    field_y = 58
    field_w = 540
    field_h = 310
    candidates = list(promotion.get("candidates", []))
    sampled_runs = sample_particle_runs(runs, limit=520)
    target_points = [target_coordinate(run.target_name)[0] for run in sampled_runs]
    budget_points = [search_budget_coordinate(run) for run in sampled_runs]
    x_min, x_max = padded_range(target_points, fallback=(0.8, 6.2))
    y_min, y_max = padded_range(budget_points, fallback=(0.0, 10.0))

    grid_svg = []
    for tick in linear_ticks(x_min, x_max, count=5):
        x = field_x + scale_between(tick, x_min, x_max) * field_w
        grid_svg.append(f'<line x1="{x:.1f}" y1="{field_y}" x2="{x:.1f}" y2="{field_y + field_h}" stroke="#edf1f6" />')
        grid_svg.append(
            f'<text class="tick-label" x="{x:.1f}" y="{field_y + field_h + 18}" text-anchor="middle">{escape(format_axis_value(tick))}</text>'
        )
    for tick in linear_ticks(y_min, y_max, count=4):
        y = field_y + field_h - scale_between(tick, y_min, y_max) * field_h
        grid_svg.append(f'<line x1="{field_x}" y1="{y:.1f}" x2="{field_x + field_w}" y2="{y:.1f}" stroke="#edf1f6" />')
        grid_svg.append(
            f'<text class="tick-label" x="{field_x - 8}" y="{y + 3:.1f}" text-anchor="end">{escape(format_axis_value(tick))}</text>'
        )

    particles = []
    for run in sampled_runs:
        target_value, target_label = target_coordinate(run.target_name)
        budget_value = search_budget_coordinate(run)
        x = field_x + scale_between(target_value, x_min, x_max) * field_w
        y = field_y + field_h - scale_between(budget_value, y_min, y_max) * field_h
        x += (unit_hash(run.run_id, "target-jitter") - 0.5) * 5.0
        y += (unit_hash(run.run_id, "budget-jitter") - 0.5) * 5.0
        if run.best_closure == "GOOD":
            fill = status_color("GOOD")
            opacity = 0.75
            radius = 2.4
        elif run.best_closure == "UNKNOWN":
            fill = status_color("UNKNOWN")
            opacity = 0.28
            radius = 1.8
        else:
            fill = status_color("BAD")
            opacity = 0.13
            radius = 1.5
        particles.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius:.1f}" fill="{fill}" opacity="{opacity}">'
            f'<title>{escape(run.best_closure)} | {escape(run.target_name)} | x={escape(target_label)} | '
            f'budget={budget_value:.2f} (depth {run.depth}, width {run.width}) | score {run.best_score}</title></circle>'
        )

    promoted_markers = []
    for candidate in candidates[:12]:
        target_name = str(candidate.get("target_name", "unknown"))
        search = candidate.get("search", {})
        depth = int(search.get("depth", 0)) if isinstance(search, dict) else 0
        width_value = int(search.get("width", 0)) if isinstance(search, dict) else 0
        target_value, target_label = target_coordinate(target_name)
        budget_value = float(depth) + math.log10(max(1, width_value))
        x = field_x + scale_between(target_value, x_min, x_max) * field_w
        y = field_y + field_h - scale_between(budget_value, y_min, y_max) * field_h
        queue_status = str(candidate.get("queue_status", "pending"))
        stroke = status_color("BAD") if queue_status.startswith("failed") else status_color("GOOD")
        promoted_markers.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5.8" fill="none" stroke="{stroke}" stroke-width="2.2">'
            f'<title>promoted candidate | {escape(target_name)} | {escape(queue_status)} | x={escape(target_label)}</title></circle>'
        )

    gate_profile = [
        ("Runs", len(runs), "#6a7280"),
        ("Cases", len(set(run.case_key for run in runs)), "#496d93"),
        ("Transitions", sum(run.transition_count for run in runs), "#496d93"),
        ("GOOD", int(closure_counts.get("GOOD", 0)), status_color("GOOD")),
        ("Dedupe", len(promising_runs(runs)), "#2e7d82"),
        ("Promoted", int(promotion.get("candidate_count", 0)), "#8d6416"),
    ]
    max_gate_log = max(math.log10(count + 1) for _, count, _ in gate_profile) if gate_profile else 1.0
    gate_svg = [
        '<text class="viz-title" x="630" y="60">Gate profile (exact counts)</text>',
        '<text class="axis-label" x="630" y="78">Bars are log-scaled so rare survivors remain visible.</text>',
    ]
    for index, (label, count, color) in enumerate(gate_profile):
        y = 105 + index * 46
        bar_width = 34 + (math.log10(count + 1) / max_gate_log) * 300 if max_gate_log else 34
        gate_svg.append(
            f"""
            <g>
              <text class="gate-mini-label" x="630" y="{y + 13}">{escape(label)}</text>
              <rect x="720" y="{y}" width="334" height="18" rx="5" fill="#f0f3f7" />
              <rect x="720" y="{y}" width="{bar_width:.1f}" height="18" rx="5" fill="{color}" opacity="0.76" />
              <text class="gate-mini-count" x="1064" y="{y + 13}" text-anchor="end">{escape(format_count(count))}</text>
            </g>
            """
        )

    return f"""
    <div class="funnel-shell">
      <svg class="funnel-viz" viewBox="0 0 {width} {height}" role="img" aria-label="Search projection and gate profile">
        <text class="viz-title" x="36" y="26">Search projection, diagnostic only</text>
        <rect x="{field_x}" y="{field_y}" width="{field_w}" height="{field_h}" rx="8" fill="#fbfcfe" stroke="#dfe4ec" />
        <text class="axis-label" x="{field_x}" y="{field_y - 24}">uniform capped sample of real runs</text>
        <text class="axis-label" x="{field_x}" y="{field_y - 10}">x: target-coordinate projection · y: depth + log10(width)</text>
        <g>{''.join(grid_svg)}</g>
        <g>{''.join(particles)}</g>
        <g>{''.join(promoted_markers)}</g>
        <text class="axis-label" x="{field_x + field_w / 2}" y="{field_y + field_h + 38}" text-anchor="middle">target coordinate: proxy family to Lp exponent</text>
        <text class="axis-label" transform="translate({field_x - 30} {field_y + field_h / 2}) rotate(-90)" text-anchor="middle">search budget</text>

        {''.join(gate_svg)}

        <g transform="translate(38 452)">
          <circle cx="0" cy="0" r="4" fill="{status_color("GOOD")}" opacity="0.75" />
          <text class="axis-label" x="12" y="4">GOOD sample</text>
          <circle cx="118" cy="0" r="4" fill="{status_color("UNKNOWN")}" opacity="0.36" />
          <text class="axis-label" x="130" y="4">UNKNOWN</text>
          <circle cx="220" cy="0" r="4" fill="{status_color("BAD")}" opacity="0.20" />
          <text class="axis-label" x="232" y="4">BAD</text>
          <circle cx="294" cy="0" r="5" fill="none" stroke="{status_color("BAD")}" stroke-width="2" />
          <text class="axis-label" x="306" y="4">promoted but failed gate</text>
        </g>
      </svg>
      <div class="interpretation-guard">
        <strong>Interpretation guard:</strong>
        The right-edge survivor cluster is not a recommendation to search only there. The promoted examples currently fail target relevance because they are pure Sobolev-style norm embeddings, not nonlinear control inequalities. Treat that cluster as evidence of scoring/search bias until a candidate passes relevance and numeric stress.
      </div>
      <p class="viz-note">Each particle is a real run from the data index. The plot is a low-dimensional projection, so distance in this panel is not mathematical distance between proof states. Small deterministic jitter only separates overlapping dots.</p>
    </div>
    """


def sample_particle_runs(runs: list[IndexedRun], *, limit: int) -> list[IndexedRun]:
    if len(runs) <= limit:
        return runs
    return sorted(runs, key=lambda run: unit_hash(run.run_id, "particle-sample"))[:limit]


def monte_carlo_frontier_panel(runs: list[IndexedRun], promotion: dict[str, Any]) -> str:
    if not runs:
        return '<p class="viz-note">No runs available for frontier estimation.</p>'

    basis_runs = sample_particle_runs(runs, limit=1400)
    candidates = list(promotion.get("candidates", []))
    x_values = [target_coordinate(run.target_name)[0] for run in basis_runs]
    y_values = [search_budget_coordinate(run) for run in basis_runs]
    for candidate in candidates:
        target_name = str(candidate.get("target_name", "unknown"))
        search = candidate.get("search", {})
        if isinstance(search, dict):
            x_values.append(target_coordinate(target_name)[0])
            y_values.append(float(search.get("depth", 0)) + math.log10(max(1, int(search.get("width", 1)))))

    x_min, x_max = padded_range(x_values, fallback=(0.8, 6.2))
    y_min, y_max = padded_range(y_values, fallback=(0.0, 10.0))

    weighted_points: list[tuple[float, float, float, float]] = []
    for run in basis_runs:
        x = scale_between(target_coordinate(run.target_name)[0], x_min, x_max)
        y = scale_between(search_budget_coordinate(run), y_min, y_max)
        weighted_points.append((x, y, run_progress_level(run), 1.0))
    for candidate in candidates:
        target_name = str(candidate.get("target_name", "unknown"))
        search = candidate.get("search", {})
        if not isinstance(search, dict):
            continue
        depth = int(search.get("depth", 0))
        width_value = int(search.get("width", 1))
        x = scale_between(target_coordinate(target_name)[0], x_min, x_max)
        y = scale_between(float(depth) + math.log10(max(1, width_value)), y_min, y_max)
        weighted_points.append((x, y, candidate_progress_level(str(candidate.get("queue_status", "pending_symbolic"))), 4.0))

    cols = 42
    rows = 22
    probes_per_cell = 3
    plot_x = 52
    plot_y = 36
    plot_w = 840
    plot_h = 286
    cell_w = plot_w / cols
    cell_h = plot_h / rows
    sigma2 = 0.018
    cells = []
    for row in range(rows):
        for col in range(cols):
            progress_sum = 0.0
            evidence_sum = 0.0
            for probe in range(probes_per_cell):
                px = (col + deterministic_probe("mc-x", row, col, probe)) / cols
                py = (row + deterministic_probe("mc-y", row, col, probe)) / rows
                progress, evidence = estimate_frontier_at(px, py, weighted_points, sigma2=sigma2)
                progress_sum += progress
                evidence_sum += evidence
            progress_value = progress_sum / probes_per_cell
            evidence_value = min(1.0, evidence_sum / probes_per_cell / 3.2)
            color = progress_color(progress_value)
            opacity = 0.18 + 0.72 * evidence_value
            x = plot_x + col * cell_w
            y = plot_y + plot_h - (row + 1) * cell_h
            cells.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{cell_w + 0.25:.2f}" height="{cell_h + 0.25:.2f}" '
                f'fill="{color}" opacity="{opacity:.2f}"><title>estimated frontier {progress_value:.2f}; evidence {evidence_value:.2f}</title></rect>'
            )

    promoted = []
    for candidate in candidates[:16]:
        target_name = str(candidate.get("target_name", "unknown"))
        search = candidate.get("search", {})
        if not isinstance(search, dict):
            continue
        depth = int(search.get("depth", 0))
        width_value = int(search.get("width", 1))
        x_value = scale_between(target_coordinate(target_name)[0], x_min, x_max)
        y_value = scale_between(float(depth) + math.log10(max(1, width_value)), y_min, y_max)
        x = plot_x + x_value * plot_w
        y = plot_y + plot_h - y_value * plot_h
        status = str(candidate.get("queue_status", "pending_symbolic"))
        stroke = "#b63232" if status.startswith("failed") else "#20242c"
        promoted.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6.4" fill="none" stroke="{stroke}" stroke-width="2.3">'
            f'<title>{escape(target_name)} | {escape(status)}</title></circle>'
        )

    ticks = []
    for tick in linear_ticks(x_min, x_max, count=6):
        x = plot_x + scale_between(tick, x_min, x_max) * plot_w
        ticks.append(f'<line x1="{x:.1f}" y1="{plot_y}" x2="{x:.1f}" y2="{plot_y + plot_h}" stroke="#ffffff" opacity="0.55" />')
        ticks.append(f'<text class="tick-label" x="{x:.1f}" y="{plot_y + plot_h + 18}" text-anchor="middle">{escape(format_axis_value(tick))}</text>')
    for tick in linear_ticks(y_min, y_max, count=4):
        y = plot_y + plot_h - scale_between(tick, y_min, y_max) * plot_h
        ticks.append(f'<line x1="{plot_x}" y1="{y:.1f}" x2="{plot_x + plot_w}" y2="{y:.1f}" stroke="#ffffff" opacity="0.55" />')
        ticks.append(f'<text class="tick-label" x="{plot_x - 8}" y="{y + 3:.1f}" text-anchor="end">{escape(format_axis_value(tick))}</text>')

    legend_items = [
        (1.0, "sampled"),
        (2.5, "unknown/near"),
        (4.0, "GOOD"),
        (6.0, "verified gates"),
        (8.0, "deep frontier"),
    ]
    legend_bits = []
    for index, (value, label) in enumerate(legend_items):
        x = 930
        y = 62 + index * 34
        legend_bits.append(
            f'<rect x="{x}" y="{y}" width="18" height="18" rx="4" fill="{progress_color(value)}" />'
            f'<text class="axis-label" x="{x + 28}" y="{y + 13}">{escape(label)}</text>'
        )

    return f"""
    <div class="mc-map-shell">
      <svg class="mc-map" viewBox="0 0 1080 410" role="img" aria-label="Monte Carlo frontier map">
        <text class="viz-title" x="{plot_x}" y="20">Monte Carlo estimate of farthest local progress</text>
        <rect x="{plot_x}" y="{plot_y}" width="{plot_w}" height="{plot_h}" rx="8" fill="#f7f9fc" stroke="#dfe4ec" />
        <g>{''.join(cells)}</g>
        <g>{''.join(ticks)}</g>
        <g>{''.join(promoted)}</g>
        <text class="axis-label" x="{plot_x + plot_w / 2}" y="{plot_y + plot_h + 42}" text-anchor="middle">target coordinate projection</text>
        <text class="axis-label" transform="translate(17 {plot_y + plot_h / 2}) rotate(-90)" text-anchor="middle">search budget</text>
        <text class="viz-title" x="930" y="36">Color meaning</text>
        {''.join(legend_bits)}
        <circle cx="938" cy="257" r="6.4" fill="none" stroke="#b63232" stroke-width="2.3" />
        <text class="axis-label" x="958" y="261">promoted, failed gate</text>
      </svg>
      <div class="interpretation-guard">
        <strong>Interpretation guard:</strong>
        This is a Monte Carlo smoothing of the existing dataset, not a proof landscape. It estimates where the current search process gets far, so it can reveal bias and under-sampled zones. A bright area means "the current engine advances there", not "the inequality is true there".
      </div>
      <p class="viz-note">Method: each heat cell is sampled by deterministic random probe points. Each probe estimates local progress from nearby runs and promoted candidates using distance weights in the projected target/search-budget plane.</p>
    </div>
    """


def deterministic_probe(salt: str, row: int, col: int, probe: int) -> float:
    return unit_hash(f"{row}:{col}:{probe}", salt)


def estimate_frontier_at(
    x: float,
    y: float,
    points: list[tuple[float, float, float, float]],
    *,
    sigma2: float,
) -> tuple[float, float]:
    numerator = 0.0
    denominator = 0.0
    evidence = 0.0
    frontier_signal = 0.0
    for px, py, progress, importance in points:
        distance2 = (x - px) ** 2 + (y - py) ** 2
        weight = math.exp(-distance2 / (2 * sigma2)) * importance
        if weight < 0.003:
            continue
        numerator += weight * progress
        denominator += weight
        evidence += min(weight, 1.0)
        frontier_signal = max(frontier_signal, (progress - 1.0) * min(1.0, weight))
    if denominator <= 0:
        return 1.0, 0.0
    local_average = numerator / denominator
    local_frontier = 1.0 + frontier_signal
    return max(local_average, local_frontier), evidence


def run_progress_level(run: IndexedRun) -> float:
    if run.best_closure == "GOOD":
        return 4.0
    if run.best_score >= 70:
        return 3.1
    if run.best_closure == "UNKNOWN":
        return 2.4
    if run.best_score > 0:
        return 1.7
    return 1.0


def candidate_progress_level(status: str) -> float:
    levels = {
        "pending_symbolic": 4.6,
        "symbolic_replay": 5.0,
        "failed_symbolic": 5.0,
        "passed_symbolic": 5.7,
        "pending_scaling": 5.7,
        "scaling_audit": 6.2,
        "failed_scaling": 6.2,
        "passed_scaling": 6.8,
        "pending_relevance": 6.8,
        "target_relevance_check": 7.2,
        "failed_relevance": 7.2,
        "passed_relevance": 7.8,
        "pending_numeric": 7.8,
        "numeric_counterexample_search": 8.3,
        "failed_numeric": 8.3,
        "passed_numeric": 8.8,
        "pending_human_review": 8.8,
        "human_math_review": 9.3,
        "verified": 10.0,
        "accepted": 10.0,
    }
    return levels.get(status, 4.6)


def progress_color(value: float) -> str:
    stops = [
        (1.0, (232, 237, 243)),
        (2.5, (154, 182, 207)),
        (4.0, (85, 190, 146)),
        (6.0, (242, 193, 78)),
        (7.4, (231, 111, 81)),
        (10.0, (122, 78, 163)),
    ]
    if value <= stops[0][0]:
        return rgb_hex(stops[0][1])
    for (left_value, left_color), (right_value, right_color) in zip(stops, stops[1:]):
        if value <= right_value:
            ratio = (value - left_value) / (right_value - left_value)
            color = tuple(
                round(left_color[index] + (right_color[index] - left_color[index]) * ratio)
                for index in range(3)
            )
            return rgb_hex(color)
    return rgb_hex(stops[-1][1])


def rgb_hex(color: tuple[int, int, int]) -> str:
    return "#" + "".join(f"{component:02x}" for component in color)


def target_coordinate(target_name: str) -> tuple[float, str]:
    lowered = target_name.lower()
    proxy_coordinates = {
        "vortex_stretching": (1.05, "vortex stretching proxy"),
        "strain_vorticity": (1.35, "strain-vorticity proxy"),
        "omega_cubic_integral": (3.0, "omega cubic integral, p=3"),
    }
    for token, coordinate in proxy_coordinates.items():
        if token in lowered:
            return coordinate

    match = re.search(r"omega_L(?P<num>\d+)(?:_(?P<den>\d+))?(?=_crit|_squared|$)", target_name)
    if match:
        numerator = int(match.group("num"))
        denominator = int(match.group("den") or "1")
        value = numerator / denominator
        return value, f"omega L^{format_axis_value(value)}"

    fallback = 0.8 + unit_hash(target_name, "target-coordinate") * 5.4
    return fallback, "stable fallback coordinate"


def search_budget_coordinate(run: IndexedRun) -> float:
    return float(run.depth) + math.log10(max(1, run.width))


def padded_range(values: list[float], *, fallback: tuple[float, float]) -> tuple[float, float]:
    if not values:
        return fallback
    lower = min(values)
    upper = max(values)
    if math.isclose(lower, upper):
        lower -= 0.5
        upper += 0.5
    padding = max(0.08, (upper - lower) * 0.06)
    return lower - padding, upper + padding


def scale_between(value: float, lower: float, upper: float) -> float:
    if math.isclose(lower, upper):
        return 0.5
    return max(0.0, min(1.0, (value - lower) / (upper - lower)))


def linear_ticks(lower: float, upper: float, *, count: int) -> list[float]:
    if count <= 1 or math.isclose(lower, upper):
        return [(lower + upper) / 2]
    return [lower + (upper - lower) * index / (count - 1) for index in range(count)]


def format_axis_value(value: float) -> str:
    if abs(value - round(value)) < 0.02:
        return str(int(round(value)))
    return f"{value:.2f}".rstrip("0").rstrip(".")


def unit_hash(value: str, salt: str) -> float:
    digest = hashlib.sha256(f"{salt}:{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64 - 1)


def short_count(value: int) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 10_000:
        return f"{value / 1_000:.0f}k"
    if value >= 1_000:
        return f"{value / 1_000:.1f}k"
    return str(value)


def gate_board(rows: list[tuple[str, int, str, str]]) -> str:
    cards = []
    for index, (label, count, note, klass) in enumerate(rows, start=1):
        cards.append(
            f"""
            <div class="gate-card {escape(klass)}">
              <div class="gate-index">Gate {index}</div>
              <div class="gate-count">{format_count(count)}</div>
              <div class="gate-label">{escape(label)}</div>
              <div class="gate-note">{escape(note)}</div>
            </div>
            """
        )
    return '<div class="gate-board">' + "\n".join(cards) + "</div>"


def gate_row(
    label: str,
    key: str,
    fallback_note: str,
    progress_counts: dict[str, Counter[str]],
) -> tuple[str, int, str, str]:
    counts = progress_counts.get(key, Counter())
    reached = sum(counts.values())
    if not reached:
        return (label, 0, fallback_note, "empty")

    passed = counts.get("passed", 0)
    failed = counts.get("failed", 0)
    pending = counts.get("pending", 0)
    note_parts = []
    if passed:
        note_parts.append(f"{format_count(passed)} passed")
    if failed:
        note_parts.append(f"{format_count(failed)} failed")
    if pending:
        note_parts.append(f"{format_count(pending)} pending")
    klass = "failed" if failed else "frontier" if pending else "done"
    return (label, reached, " · ".join(note_parts) or fallback_note, klass)


def search_progress_panel(
    runs: list[IndexedRun],
    status_counts: Counter[str],
    closure_counts: Counter[str],
    case_counts: Counter[tuple[str, int, int]],
    promotion: dict[str, Any],
    transition_total: int,
) -> str:
    run_count = len(runs)
    unique_cases = len(case_counts)
    repeated_runs = sum(count - 1 for count in case_counts.values() if count > 1)
    kept = int(status_counts.get("kept", 0))
    rejected = int(status_counts.get("rejected", 0))
    duplicate = int(status_counts.get("duplicate", 0))
    best_good = int(closure_counts.get("GOOD", 0))
    best_unknown = int(closure_counts.get("UNKNOWN", 0))
    best_bad = int(closure_counts.get("BAD", 0))
    promising_count = len(promising_runs(runs))
    promoted_count = int(promotion.get("candidate_count", 0))
    candidates = list(promotion.get("candidates", []))
    queue_status_counts = Counter(str(candidate.get("queue_status", "pending_symbolic")) for candidate in candidates)

    gate_counts = verification_gate_counts(queue_status_counts)
    frontier = pipeline_frontier(promoted_count, gate_counts, promising_count, best_good, run_count)
    downstream_clear = "No candidate has passed symbolic replay yet."
    if gate_counts["verified"]:
        downstream_clear = f'{gate_counts["verified"]} candidate(s) reached verified status.'
    elif gate_counts["human"]:
        downstream_clear = f'{gate_counts["human"]} candidate(s) reached human math review.'
    elif gate_counts["numeric"]:
        downstream_clear = f'{gate_counts["numeric"]} candidate(s) reached numeric stress testing.'
    elif gate_counts["relevance"]:
        downstream_clear = f'{gate_counts["relevance"]} candidate(s) reached target relevance checking.'
    elif gate_counts["scaling"]:
        downstream_clear = f'{gate_counts["scaling"]} candidate(s) reached scaling audit.'

    stages = [
        {
            "label": "Random Samples",
            "count": run_count,
            "note": "search executions",
            "class": "done" if run_count else "empty",
        },
        {
            "label": "Unique Cases",
            "count": unique_cases,
            "note": f"{repeated_runs} repeat runs",
            "class": "done" if unique_cases else "empty",
        },
        {
            "label": "Rule Edges",
            "count": transition_total,
            "note": f"{kept} kept · {rejected} rejected · {duplicate} dup",
            "class": "done" if transition_total else "empty",
        },
        {
            "label": "Best Closure",
            "count": best_good,
            "note": f"GOOD · {best_unknown} unknown · {best_bad} bad",
            "class": "done" if best_good else "watch" if best_unknown else "empty",
        },
        {
            "label": "Shortlist",
            "count": promising_count,
            "note": "deduped promising forms",
            "class": "done" if promising_count else "empty",
        },
        {
            "label": "Promotion Queue",
            "count": promoted_count,
            "note": "deduped GOOD candidates",
            "class": "done" if promoted_count else "empty",
        },
        {
            "label": "Verification Gates",
            "count": promoted_count,
            "note": frontier,
            "class": "frontier" if promoted_count else "empty",
        },
    ]

    stage_cards = []
    for index, stage in enumerate(stages, start=1):
        stage_cards.append(
            f"""
            <div class="pipeline-step-card {escape(stage["class"])}">
              <div class="pipeline-step-kicker">Phase {index}</div>
              <div class="pipeline-step-count">{format_count(stage["count"])}</div>
              <div class="pipeline-step-label">{escape(stage["label"])}</div>
              <div class="pipeline-step-note">{escape(stage["note"])}</div>
            </div>
            """
        )

    closure_bits = " ".join(
        [
            f'{badge("GOOD")} {best_good}',
            f'{badge("UNKNOWN")} {best_unknown}',
            f'{badge("BAD")} {best_bad}',
        ]
    )
    gate_bits = verification_gate_badges(queue_status_counts)
    frontier_sentence = frontier.rstrip(".") + "."
    explanation = (
        f'<p class="progress-summary"><strong>Current frontier:</strong> {escape(frontier_sentence)} '
        f'{escape(downstream_clear)} Counts mix run-level samples and edge-level transitions, so this is a progress map, not a strict funnel.</p>'
        f'<p class="viz-note">Best-closure split: {closure_bits}</p>'
        f'<p class="viz-note">Queue statuses: {gate_bits}</p>'
    )

    candidate_progress = candidate_gate_progress(candidates)
    return explanation + '<div class="pipeline-flow">' + "\n".join(stage_cards) + "</div>" + candidate_progress


def verification_gate_counts(queue_status_counts: Counter[str]) -> dict[str, int]:
    return {
        "symbolic": sum(queue_status_counts.get(key, 0) for key in ["pending_symbolic", "symbolic_replay", "failed_symbolic"]),
        "scaling": sum(queue_status_counts.get(key, 0) for key in ["pending_scaling", "scaling_audit", "passed_symbolic", "failed_scaling"]),
        "relevance": sum(queue_status_counts.get(key, 0) for key in ["pending_relevance", "target_relevance_check", "passed_scaling", "failed_relevance"]),
        "numeric": sum(queue_status_counts.get(key, 0) for key in ["pending_numeric", "numeric_counterexample_search", "passed_relevance", "failed_numeric"]),
        "human": sum(queue_status_counts.get(key, 0) for key in ["pending_human_review", "human_math_review", "passed_numeric"]),
        "verified": sum(queue_status_counts.get(key, 0) for key in ["verified", "accepted"]),
    }


def verification_progress_counts(
    candidates: list[dict[str, Any]],
    queue_status_counts: Counter[str],
) -> dict[str, Counter[str]]:
    counts: dict[str, Counter[str]] = {
        "symbolic_replay": Counter(),
        "scaling_audit": Counter(),
        "target_relevance_check": Counter(),
        "numeric_counterexample_search": Counter(),
        "human_math_review": Counter(),
    }

    saw_checks = False
    for candidate in candidates:
        verification = candidate.get("verification", {})
        if not isinstance(verification, dict):
            continue
        checks = verification.get("checks", [])
        if not isinstance(checks, list) or not checks:
            continue
        saw_checks = True
        for check in checks:
            if not isinstance(check, dict):
                continue
            gate = str(check.get("gate", ""))
            status = str(check.get("status", "unknown"))
            if gate in counts:
                counts[gate][status] += 1

    if saw_checks:
        return counts

    fallback_gate_by_status = {
        "pending_symbolic": "symbolic_replay",
        "symbolic_replay": "symbolic_replay",
        "failed_symbolic": "symbolic_replay",
        "pending_scaling": "scaling_audit",
        "scaling_audit": "scaling_audit",
        "passed_symbolic": "scaling_audit",
        "failed_scaling": "scaling_audit",
        "pending_relevance": "target_relevance_check",
        "target_relevance_check": "target_relevance_check",
        "passed_scaling": "target_relevance_check",
        "failed_relevance": "target_relevance_check",
        "pending_numeric": "numeric_counterexample_search",
        "numeric_counterexample_search": "numeric_counterexample_search",
        "passed_relevance": "numeric_counterexample_search",
        "failed_numeric": "numeric_counterexample_search",
        "pending_human_review": "human_math_review",
        "human_math_review": "human_math_review",
        "passed_numeric": "human_math_review",
    }
    status_kind = {
        "failed_symbolic": "failed",
        "failed_scaling": "failed",
        "failed_relevance": "failed",
        "failed_numeric": "failed",
        "pending_symbolic": "pending",
        "pending_scaling": "pending",
        "pending_relevance": "pending",
        "pending_numeric": "pending",
        "pending_human_review": "pending",
    }
    for status, count in queue_status_counts.items():
        gate = fallback_gate_by_status.get(status)
        if gate is None:
            continue
        counts[gate][status_kind.get(status, "pending")] += count
    return counts


def pipeline_frontier(
    promoted_count: int,
    gate_counts: dict[str, int],
    promising_count: int,
    best_good: int,
    run_count: int,
) -> str:
    if gate_counts["verified"]:
        return f'{gate_counts["verified"]} candidate(s) verified'
    if gate_counts["human"]:
        return f'{gate_counts["human"]} candidate(s) at human review'
    if gate_counts["numeric"]:
        return f'{gate_counts["numeric"]} candidate(s) at numeric stress test'
    if gate_counts["relevance"]:
        return f'{gate_counts["relevance"]} candidate(s) at target relevance check'
    if gate_counts["scaling"]:
        return f'{gate_counts["scaling"]} candidate(s) at scaling audit'
    if gate_counts["symbolic"]:
        return f'{gate_counts["symbolic"]} candidate(s) waiting for symbolic replay'
    if promoted_count:
        return f"{promoted_count} candidate(s) exported"
    if promising_count:
        return f"{promising_count} promising form(s), not promoted yet"
    if best_good:
        return f"{best_good} GOOD run(s), awaiting dedupe promotion"
    if run_count:
        return "search mass exists, no GOOD candidate promoted yet"
    return "no search data yet"


def verification_gate_badges(queue_status_counts: Counter[str]) -> str:
    if not queue_status_counts:
        return '<span class="badge none">empty</span>'
    return " ".join(
        f'<span class="badge none">{escape(status)} {count}</span>'
        for status, count in queue_status_counts.most_common()
    )


def candidate_gate_progress(candidates: list[dict[str, Any]], *, limit: int = 4) -> str:
    if not candidates:
        return '<p class="viz-note">No promoted candidates yet, so downstream verification gates are empty.</p>'

    cards = []
    for candidate in candidates[:limit]:
        status = str(candidate.get("queue_status", "pending_symbolic"))
        stage_index = queue_status_stage(status)
        gates = [
            "Sample",
            "GOOD",
            "Promote",
            "Symbolic",
            "Scaling",
            "Relevance",
            "Numeric",
            "Human",
        ]
        pills = []
        for index, gate in enumerate(gates):
            if status in {"verified", "accepted"}:
                klass = "done"
            elif status.startswith("failed") and index == stage_index:
                klass = "failed"
            elif index < stage_index:
                klass = "done"
            elif index == stage_index:
                klass = "frontier"
            else:
                klass = "future"
            pills.append(f'<span class="gate-pill {klass}">{escape(gate)}</span>')
        run_id = str(candidate.get("run_id", ""))
        run_cell = f'<a href="runs/{escape(run_id)}.html">{escape(run_id)}</a>'
        cards.append(
            f"""
            <div class="candidate-progress-card">
              <div class="candidate-title"><strong>{escape(str(candidate.get("target_name", "")))}</strong>{badge(str(candidate.get("closure_status", "NONE")))}</div>
              <div class="candidate-meta">score {escape(str(candidate.get("score", "")))} · {escape(status)} · {run_cell}</div>
              <div class="gate-strip">{"".join(pills)}</div>
              {verification_reason_note(candidate)}
              <div class="candidate-rhs">{escape(str(candidate.get("rhs_text", "")))}</div>
            </div>
            """
        )
    hidden = max(0, len(candidates) - limit)
    note = f'<p class="viz-note">Showing {min(len(candidates), limit)} promoted candidate path(s). {hidden} additional path(s) hidden.</p>' if hidden else ""
    return note + '<div class="candidate-progress-list">' + "".join(cards) + "</div>"


def verification_reason_note(candidate: dict[str, Any]) -> str:
    verification = candidate.get("verification", {})
    if not isinstance(verification, dict):
        return ""
    checks = verification.get("checks", [])
    if not isinstance(checks, list):
        return ""
    failed = next((check for check in checks if isinstance(check, dict) and check.get("status") == "failed"), None)
    if failed is None:
        pending = next((check for check in checks if isinstance(check, dict) and check.get("status") == "pending"), None)
        if pending is None:
            return ""
        return f'<p class="candidate-note">Waiting: {escape(str(pending.get("reason", "")))}</p>'
    return f'<p class="candidate-note failed">Failed {escape(str(failed.get("gate", "")))}: {escape(str(failed.get("reason", "")))}</p>'


def queue_status_stage(status: str) -> int:
    mapping = {
        "pending_symbolic": 3,
        "symbolic_replay": 3,
        "failed_symbolic": 3,
        "pending_scaling": 4,
        "scaling_audit": 4,
        "passed_symbolic": 4,
        "failed_scaling": 4,
        "pending_relevance": 5,
        "target_relevance_check": 5,
        "passed_scaling": 5,
        "failed_relevance": 5,
        "pending_numeric": 6,
        "numeric_counterexample_search": 6,
        "passed_relevance": 6,
        "failed_numeric": 6,
        "pending_human_review": 7,
        "human_math_review": 7,
        "passed_numeric": 7,
        "verified": 8,
        "accepted": 8,
    }
    return mapping.get(status, 3)


def format_count(value: int) -> str:
    return f"{value:,}"


def aggregate_summary_counter(runs: list[IndexedRun], field: str) -> Counter[str]:
    counter: Counter[str] = Counter()
    for run in runs:
        value = run.summary.get(field, {})
        if isinstance(value, dict):
            counter.update(value)
    return counter


def proof_young_transitions(runs: list[IndexedRun]) -> list[dict[str, Any]]:
    transitions: list[dict[str, Any]] = []
    for run in runs:
        best = run.best_state or {}
        for step in best.get("proof", []):
            if step.get("rule_name") != "Young":
                continue
            transitions.append({"action": "Young", "rhs_before": step.get("before")})
    return transitions


def promotion_panel(promotion: dict[str, Any], generated_run_ids: set[str], limit: int = 6) -> str:
    candidates = promotion.get("candidates", [])[:limit]
    queue_path = promotion.get("queue_path", "")
    jsonl_path = promotion.get("jsonl_path", "")
    if not candidates:
        return (
            '<p class="viz-note">No GOOD candidates are waiting yet. When one appears, this panel will list it and '
            'the queue files will be refreshed automatically.</p>'
            f'<p class="viz-note">Queue target: <code>{escape(str(queue_path))}</code></p>'
        )

    cards = []
    for candidate in candidates:
        run_id = str(candidate.get("run_id", ""))
        if run_id in generated_run_ids:
            run_cell = f'<a href="runs/{escape(run_id)}.html">{escape(run_id)}</a>'
        else:
            run_cell = escape(run_id)
        cards.append(
            candidate_card(
                title=escape(str(candidate.get("target_name", ""))),
                status=str(candidate.get("closure_status", "NONE")),
                meta=(
                    f"candidate {escape(str(candidate.get('candidate_id', '')))} · "
                    f"score {escape(str(candidate.get('score', '')))} · "
                    f"{escape(str(candidate.get('queue_status', '')))}<br>{run_cell}"
                ),
                rhs=escape(str(candidate.get("rhs_text", ""))),
            )
        )

    note = (
        f'<p class="viz-note">Promoted {promotion.get("candidate_count", 0)} deduplicated candidates. '
        f'Queue: <code>{escape(str(queue_path))}</code>. JSONL: <code>{escape(str(jsonl_path))}</code>.</p>'
    )
    return note + '<div class="candidate-list">' + "".join(cards) + "</div>"


def is_promising_run(run: IndexedRun | RunReport) -> bool:
    if run.best_closure == "GOOD":
        return True
    if run.best_score >= 70:
        return True
    if run.best_closure == "UNKNOWN" and run.best_score >= 10:
        return True
    return False


def promising_runs(runs: list[IndexedRun | RunReport]) -> list[IndexedRun | RunReport]:
    by_key: dict[tuple[str, str, str], IndexedRun | RunReport] = {}
    for run in runs:
        if not is_promising_run(run):
            continue
        best = run.best_state or {}
        key = (
            run.target_name,
            text_of(best.get("rhs")),
            proof_path_text(run),
        )
        previous = by_key.get(key)
        if previous is None or run.best_score > previous.best_score:
            by_key[key] = run
    return sorted(
        by_key.values(),
        key=lambda run: (run.best_closure != "GOOD", -run.best_score, run.target_name, run.run_id),
    )


def promising_candidates_panel(
    runs: list[IndexedRun],
    generated_run_ids: set[str],
    *,
    limit: int = 8,
) -> str:
    selected = promising_runs(runs)[:limit]
    if not selected:
        return (
            '<p class="viz-note">No promising candidates yet. The report will stay compact until GOOD or near-GOOD states appear.</p>'
        )

    cards = []
    for run in selected:
        best = run.best_state or {}
        rhs = text_of(best.get("rhs"))
        proof = proof_path_text(run)
        if run.run_id in generated_run_ids:
            run_cell = f'<a href="runs/{escape(run.run_id)}.html">{escape(run.run_id)}</a>'
        else:
            run_cell = escape(run.run_id)
        cards.append(
            candidate_card(
                title=escape(run.target_name),
                status=run.best_closure,
                meta=(
                    f"score {run.best_score} · depth {run.depth} · width {run.width}"
                    f"<br>{run_cell}<br>{escape(proof)}"
                ),
                rhs=escape(rhs),
            )
        )

    hidden = max(0, len(promising_runs(runs)) - len(selected))
    note = ""
    if hidden:
        note = f'<p class="viz-note">Showing top {len(selected)} promising candidates. {hidden} additional deduplicated candidates hidden.</p>'
    return note + '<div class="candidate-list">' + "".join(cards) + "</div>"


def candidate_card(*, title: str, status: str, meta: str, rhs: str) -> str:
    return f"""
    <div class="candidate-item">
      <div class="candidate-title"><strong>{title}</strong>{badge(status)}</div>
      <div class="candidate-meta">{meta}</div>
      <div class="candidate-rhs">{rhs}</div>
    </div>
    """


def search_outcome_cloud(runs: list[IndexedRun]) -> str:
    if not runs:
        return '<p class="viz-note">No runs available.</p>'

    width = 620
    height = 270
    left = 42
    top = 26
    plot_w = 530
    plot_h = 188
    x_bins = 42
    y_bins = 16
    scores = [run.best_score for run in runs]
    min_score = min(scores)
    max_score = max(scores)
    if min_score == max_score:
        max_score += 1

    bins: Counter[tuple[int, int, str]] = Counter()
    for run in runs:
        x_index = int((run.best_score - min_score) / (max_score - min_score) * (x_bins - 1))
        y_index = stable_bucket(f"{run.target_name}:{run.depth}:{run.width}", y_bins)
        status = run.best_closure if run.best_closure in {"GOOD", "UNKNOWN"} else "BAD"
        bins[(x_index, y_index, status)] += 1

    max_count = max(bins.values())
    dots = []
    for (x_index, y_index, status), count in bins.items():
        x = left + (x_index + 0.5) * plot_w / x_bins
        y = top + plot_h - (y_index + 0.5) * plot_h / y_bins
        radius = 2.2 + 8.0 * (count / max_count) ** 0.5
        opacity = "0.88" if status == "GOOD" else "0.46" if status == "UNKNOWN" else "0.20"
        stroke = "#111827" if status == "GOOD" else "none"
        stroke_width = "1.2" if status == "GOOD" else "0"
        dots.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius:.1f}" fill="{status_color(status)}" '
            f'fill-opacity="{opacity}" stroke="{stroke}" stroke-width="{stroke_width}">'
            f'<title>{escape(status)} bin: {count} runs</title></circle>'
        )

    ticks = []
    for value in [min_score, int((min_score + max_score) / 2), max_score]:
        x = left + (value - min_score) / (max_score - min_score) * plot_w
        ticks.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_h}" stroke="#dfe4ec" />')
        ticks.append(f'<text class="tick-label" x="{x:.1f}" y="{top + plot_h + 20}" text-anchor="middle">{value}</text>')

    return f"""
    <svg class="viz" viewBox="0 0 {width} {height}" role="img" aria-label="Search outcome density cloud">
      <text class="viz-title" x="{left}" y="16">Outcome density cloud</text>
      <rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" rx="6" fill="#fbfcfe" stroke="#dfe4ec" />
      {''.join(ticks)}
      {''.join(dots)}
      <text class="axis-label" x="{left + plot_w / 2}" y="{height - 24}" text-anchor="middle">best score</text>
      {legend(left + plot_w - 220, top + 14)}
    </svg>
    <p class="viz-note">Unpromising runs are compressed into faint density dots. Bright outlined dots are GOOD bins.</p>
    """


def stable_bucket(value: str, bucket_count: int) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % bucket_count


def run_table_rows(
    runs: list[RunReport] | list[IndexedRun],
    case_counts: Counter[tuple[str, int, int]],
    limit: int = 60,
    *,
    generated_run_ids: set[str] | None = None,
    transition_counts: dict[str, int] | None = None,
) -> tuple[list[str], str]:
    if generated_run_ids is None:
        generated_run_ids = {run.run_id for run in runs}
    if transition_counts is None:
        transition_counts = {}

    representatives = representative_runs(runs)
    run_rows = []
    for run, repeats in representatives[:limit]:
        if run.run_id in generated_run_ids:
            run_cell = f'<a href="runs/{escape(run.run_id)}.html">{escape(run.run_id)}</a>'
        else:
            run_cell = escape(run.run_id)
        best = run.best_state or {}
        rhs = text_of(best.get("rhs"))
        if isinstance(run, RunReport):
            transition_count = len(run.transitions)
        else:
            transition_count = run.transition_count or transition_counts.get(run.run_id, 0)
        run_rows.append(
            table_row(
                [
                    run_cell,
                    escape(run.target_name),
                    str(run.depth),
                    str(run.width),
                    str(repeats),
                    badge(run.best_closure),
                    str(run.best_score),
                    str(transition_count),
                    escape(rhs),
                ],
                raw=True,
            )
        )
    hidden = max(0, len(representatives) - limit)
    note = ""
    if hidden:
        note = f'<p class="viz-note">Showing {limit} representative case groups. {hidden} additional case groups are hidden here; all run pages still exist under <code>data/reports/runs</code>.</p>'
    return run_rows, note


def render_run_report(run: RunReport) -> str:
    transitions = run.transitions
    status_counts = Counter(transition.get("status", "unknown") for transition in transitions)
    action_counts = Counter(transition.get("action", "unknown") for transition in transitions)
    child_closures = Counter(
        (((transition.get("child_state") or {}).get("closure") or {}).get("status", "NONE"))
        for transition in transitions
    )

    best = run.best_state or {}
    candidate_states = [
        transition.get("child_state")
        for transition in transitions
        if transition.get("child_state") is not None
    ]
    proof = best.get("proof", [])
    target_text = text_of(best.get("lhs")) or text_of((run.start or {}).get("target"))
    rhs_text = text_of(best.get("rhs"))
    closure = best.get("closure") or {}

    cards = [
        ("Target", run.target_name),
        ("Transitions", len(transitions)),
        ("Best Closure", run.best_closure),
        ("Best Score", run.best_score),
        ("Best Depth", best.get("depth", 0)),
        ("Rows", len(run.rows)),
    ]

    transition_rows = []
    for transition in transitions:
        child = transition.get("child_state") or {}
        child_closure = (child.get("closure") or {}).get("status", "NONE")
        score = child.get("score", "")
        transition_rows.append(
            table_row(
                [
                    str(transition.get("depth", "")),
                    badge(str(transition.get("status", ""))),
                    escape(str(transition.get("action", ""))),
                    badge(str(child_closure)),
                    str(score),
                    escape(str((transition.get("filter") or {}).get("reason", ""))),
                    escape(text_of(transition.get("rhs_before"))),
                    escape(text_of(transition.get("rhs_after"))),
                ],
                raw=True,
            )
        )

    beam_rows = []
    for event in run.beam_events:
        states = event.get("states", [])
        labels = []
        for state in states[:5]:
            closure_status = ((state.get("closure") or {}).get("status", "NONE"))
            labels.append(f"{badge(closure_status)} {escape(text_of(state.get('rhs')))}")
        if len(states) > 5:
            labels.append(f"+ {len(states) - 5} more")
        beam_rows.append(table_row([str(event.get("depth", "")), str(len(states)), "<br>".join(labels)], raw=True))

    body = f"""
    <nav class="topnav"><a href="../index.html">Back to index</a></nav>
    <section class="hero">
      <div>
        <p class="eyebrow">Run Detail</p>
        <h1>{escape(run.run_id)}</h1>
        <p>{escape(target_text)} &le; {escape(rhs_text)}</p>
      </div>
    </section>
    {card_grid(cards)}
    <section class="verdict {escape(run.best_closure.lower())}">
      <strong>{escape(run.best_closure)}</strong>
      <span>{escape(str(closure.get("reason", "No closure reason recorded.")))}</span>
    </section>
    <section class="grid two visual-grid">
      {panel("Young Absorption Phase Map", young_phase_map(transitions))}
      {panel("Candidate Landscape", candidate_landscape(candidate_states))}
    </section>
    {panel("Rule Transition Heatmap", rule_transition_heatmap([run]))}
    {panel("Best Proof Path", proof_path(proof))}
    <section class="grid two">
      {panel("Transition Statuses", bar_chart(status_counts))}
      {panel("Rule Actions", bar_chart(action_counts))}
    </section>
    <section class="grid two">
      {panel("Child Closure Statuses", bar_chart(child_closures))}
      {panel("Beam Timeline", table(["Depth", "Beam Size", "Top States"], beam_rows, raw_rows=True))}
    </section>
    {panel(
        "Transitions",
        table(
            ["Depth", "Status", "Action", "Child Closure", "Score", "Filter", "Before", "After"],
            transition_rows,
            raw_rows=True,
        ),
    )}
    """
    return page(f"AISIS Run {run.run_id}", body)


def page(title: str, body: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(title)}</title>
  <style>{css()}</style>
</head>
<body>
  <main>
    {body}
  </main>
</body>
</html>
"""


def css() -> str:
    return """
:root {
  color-scheme: light;
  --bg: #f4f7fb;
  --panel: #ffffff;
  --ink: #20242c;
  --muted: #626b7a;
  --line: #dfe4ec;
  --good: #188453;
  --bad: #b63232;
  --unknown: #6a7280;
  --kept: #1d6fb8;
  --duplicate: #8d6a1f;
  --rejected: #a43f54;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background:
    radial-gradient(circle at 10% -10%, rgba(62, 142, 168, 0.16), transparent 28%),
    linear-gradient(180deg, #f7f9fd 0%, var(--bg) 42%, #eef3f8 100%);
  color: var(--ink);
  font: 14px/1.45 ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
main { max-width: 1280px; margin: 0 auto; padding: 28px; }
h1, h2, h3, p { margin-top: 0; }
h1 { font-size: 30px; margin-bottom: 8px; letter-spacing: 0; }
h2 { font-size: 18px; margin-bottom: 14px; letter-spacing: 0; }
code { background: #eef1f5; border: 1px solid var(--line); border-radius: 4px; padding: 1px 4px; overflow-wrap: anywhere; word-break: break-word; }
.hero {
  border-bottom: 1px solid var(--line);
  margin-bottom: 20px;
  padding-bottom: 18px;
}
.eyebrow {
  color: var(--muted);
  font-size: 12px;
  font-weight: 700;
  letter-spacing: .08em;
  text-transform: uppercase;
  margin-bottom: 8px;
}
.topnav { margin-bottom: 12px; }
.topnav a { color: var(--kept); text-decoration: none; font-weight: 700; }
.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 12px; margin-bottom: 18px; }
.card, .panel {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
  box-shadow: 0 1px 2px rgba(20, 28, 38, 0.04);
}
.card { padding: 14px; }
.card .label { color: var(--muted); font-size: 12px; }
.card .value { font-size: 22px; font-weight: 750; margin-top: 4px; overflow-wrap: anywhere; }
.panel { padding: 16px; margin-bottom: 18px; overflow: auto; }
.details-panel summary { cursor: pointer; font-size: 18px; font-weight: 750; margin-bottom: 10px; }
.details-panel[open] summary { margin-bottom: 14px; }
.grid.two { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 18px; }
.overview-stack {
  display: grid;
  gap: 14px;
}
.overview-lead {
  border: 1px solid #cbdde7;
  border-radius: 8px;
  background: linear-gradient(180deg, #fbfdff 0%, #f1f7fa 100%);
  padding: 14px;
}
.overview-lead p,
.overview-card p,
.overview-step p,
.overview-verdict {
  color: var(--muted);
  margin-bottom: 0;
}
.overview-lead p + p { margin-top: 8px; }
.overview-lead strong,
.overview-verdict strong { color: var(--ink); }
.overview-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 12px;
}
.overview-card {
  border: 1px solid var(--line);
  border-radius: 8px;
  background: #ffffff;
  padding: 13px;
}
.overview-card h3 {
  font-size: 14px;
  margin-bottom: 7px;
}
.overview-card p { font-size: 12px; }
.overview-steps {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 10px;
}
.overview-step {
  display: grid;
  grid-template-columns: 30px 1fr;
  gap: 9px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: #fbfcfe;
  padding: 11px;
}
.overview-step span {
  display: grid;
  place-items: center;
  width: 26px;
  height: 26px;
  border-radius: 50%;
  background: #1e5269;
  color: #ffffff;
  font-size: 12px;
  font-weight: 800;
}
.overview-step strong {
  display: block;
  font-size: 12px;
  margin-bottom: 4px;
}
.overview-step p {
  font-size: 11px;
}
.overview-verdict {
  border-left: 4px solid #d9b55c;
  background: #fff9ea;
  border-radius: 8px;
  padding: 12px 14px;
}
.progress-summary {
  color: var(--muted);
  margin-bottom: 10px;
}
.progress-summary strong { color: var(--ink); }
.funnel-shell {
  border: 1px solid var(--line);
  border-radius: 8px;
  background: linear-gradient(180deg, #ffffff 0%, #fbfdff 100%);
  padding: 12px;
  margin: 14px 0;
  overflow: auto;
}
.funnel-viz {
  width: 100%;
  min-width: 760px;
  height: auto;
  display: block;
}
.funnel-count {
  fill: var(--ink);
  font-size: 13px;
  font-weight: 800;
}
.funnel-label, .survivor-label {
  fill: var(--muted);
  font-size: 10px;
  font-weight: 700;
}
.gate-mini-label {
  fill: var(--ink);
  font-size: 11px;
  font-weight: 750;
}
.gate-mini-count {
  fill: var(--muted);
  font-size: 11px;
  font-weight: 750;
}
.interpretation-guard {
  border-left: 4px solid #d9b55c;
  border-radius: 8px;
  background: #fff9ea;
  color: var(--muted);
  font-size: 12px;
  margin-top: 10px;
  padding: 11px 13px;
}
.interpretation-guard strong { color: var(--ink); }
.gate-board {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 10px;
  margin: 14px 0 4px;
}
.gate-card {
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 12px;
  background: #fbfcfe;
}
.gate-card.done { border-color: #bddfcf; background: linear-gradient(180deg, #ffffff 0%, #f0faf5 100%); }
.gate-card.frontier { border-color: #d9b55c; background: linear-gradient(180deg, #ffffff 0%, #fff8e8 100%); }
.gate-card.failed { border-color: #e4b8b8; background: linear-gradient(180deg, #ffffff 0%, #fff3f3 100%); }
.gate-card.watch { border-color: #cfd5df; background: linear-gradient(180deg, #ffffff 0%, #f5f7fa 100%); }
.gate-card.empty { color: var(--muted); background: #f6f8fb; }
.gate-index {
  color: var(--muted);
  font-size: 10px;
  font-weight: 800;
  letter-spacing: .08em;
  text-transform: uppercase;
  margin-bottom: 5px;
}
.gate-count {
  font-size: 22px;
  font-weight: 800;
  line-height: 1.1;
}
.gate-label {
  font-weight: 750;
  margin-top: 6px;
}
.gate-note {
  color: var(--muted);
  font-size: 11px;
  margin-top: 4px;
}
.pipeline-flow {
  display: grid;
  grid-template-columns: repeat(7, minmax(0, 1fr));
  gap: 10px;
  margin: 14px 0;
}
.pipeline-step-card {
  position: relative;
  min-height: 126px;
  padding: 12px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: #fbfcfe;
}
.pipeline-step-card::after {
  content: "";
  position: absolute;
  right: -8px;
  top: 50%;
  width: 12px;
  height: 2px;
  background: var(--line);
}
.pipeline-step-card:last-child::after { display: none; }
.pipeline-step-card.done { border-color: #bddfcf; background: linear-gradient(180deg, #ffffff 0%, #f0faf5 100%); }
.pipeline-step-card.frontier { border-color: #d9b55c; background: linear-gradient(180deg, #ffffff 0%, #fff8e8 100%); box-shadow: inset 0 0 0 1px rgba(217, 181, 92, 0.28); }
.pipeline-step-card.watch { border-color: #cfd5df; background: linear-gradient(180deg, #ffffff 0%, #f5f7fa 100%); }
.pipeline-step-card.empty { color: var(--muted); background: #f6f8fb; }
.pipeline-step-kicker {
  color: var(--muted);
  font-size: 10px;
  font-weight: 800;
  letter-spacing: .08em;
  text-transform: uppercase;
  margin-bottom: 6px;
}
.pipeline-step-count {
  font-size: 24px;
  font-weight: 800;
  line-height: 1.1;
  overflow-wrap: anywhere;
}
.pipeline-step-label {
  font-weight: 750;
  margin-top: 6px;
}
.pipeline-step-note {
  color: var(--muted);
  font-size: 11px;
  margin-top: 5px;
}
.candidate-progress-list {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  gap: 12px;
  margin-top: 12px;
}
.candidate-progress-card {
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 12px;
  background: #fbfcfe;
}
.gate-strip {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin: 10px 0;
}
.gate-pill {
  border: 1px solid var(--line);
  border-radius: 999px;
  padding: 4px 8px;
  font-size: 11px;
  font-weight: 750;
  color: var(--muted);
  background: #f2f5f8;
}
.gate-pill.done { color: var(--good); background: #eaf7f0; border-color: #c7ead8; }
.gate-pill.frontier { color: #8d6416; background: #fff5db; border-color: #efd38c; }
.gate-pill.future { color: #87909d; background: #f6f8fb; }
.gate-pill.failed { color: var(--bad); background: #faeeee; border-color: #f0c7c7; }
.process-flow {
  display: grid;
  grid-template-columns: repeat(5, minmax(0, 1fr));
  gap: 10px;
  margin-bottom: 18px;
}
.process-step {
  display: grid;
  grid-template-columns: 36px 1fr;
  gap: 10px;
  align-items: start;
  background: linear-gradient(180deg, #ffffff 0%, #f7fbfc 100%);
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 12px;
  min-height: 132px;
}
.process-index {
  display: grid;
  place-items: center;
  width: 32px;
  height: 32px;
  border-radius: 999px;
  background: #1e5269;
  color: white;
  font-weight: 800;
}
.process-step h3, .guide-card h3 { font-size: 14px; margin-bottom: 6px; }
.process-step p, .guide-card p { color: var(--muted); font-size: 12px; margin-bottom: 0; }
.guide-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 12px;
  margin-bottom: 18px;
}
.guide-card {
  border: 1px solid var(--line);
  border-left: 4px solid #2e7d82;
  background: #ffffff;
  border-radius: 8px;
  padding: 13px 14px;
}
.visual-grid .panel { background: linear-gradient(180deg, #ffffff 0%, #fbfdff 100%); }
.visual-banner {
  background: linear-gradient(135deg, #1e5269 0%, #295f74 48%, #743e63 100%);
  color: white;
  border-radius: 8px;
  padding: 20px;
  margin-bottom: 18px;
  box-shadow: 0 12px 28px rgba(31, 60, 82, 0.16);
}
.visual-banner h2 { margin-bottom: 6px; }
.visual-banner p { color: rgba(255, 255, 255, 0.82); margin-bottom: 0; }
.visual-banner .eyebrow { color: rgba(255, 255, 255, 0.7); }
.viz { width: 100%; max-width: 100%; height: auto; display: block; }
.axis-label { fill: #626b7a; font-size: 11px; }
.tick-label { fill: #626b7a; font-size: 10px; }
.viz-title { fill: #20242c; font-size: 12px; font-weight: 700; }
.viz-note { color: var(--muted); font-size: 12px; margin-top: 8px; }
.heat-cell { stroke: rgba(255,255,255,0.65); stroke-width: 1; }
.point-label { fill: #20242c; font-size: 10px; font-weight: 700; }
.point-label.light { fill: #ffffff; font-size: 9px; text-shadow: 0 1px 1px rgba(0,0,0,0.35); }
.mini-table { max-height: 170px; overflow: auto; border: 1px solid var(--line); border-radius: 8px; margin-top: 10px; }
.mini-table table { font-size: 12px; }
.bar-row { display: grid; grid-template-columns: minmax(120px, 190px) 1fr 48px; gap: 10px; align-items: center; margin: 8px 0; }
.bar-label { overflow-wrap: anywhere; }
.bar-track { height: 12px; border-radius: 999px; background: #edf1f6; overflow: hidden; border: 1px solid var(--line); }
.bar-fill { height: 100%; background: #496d93; }
.bar-count { color: var(--muted); text-align: right; }
.badge {
  display: inline-block;
  border: 1px solid var(--line);
  border-radius: 999px;
  padding: 2px 8px;
  font-size: 12px;
  font-weight: 700;
  white-space: nowrap;
}
.badge.good { color: var(--good); background: #eaf7f0; border-color: #c7ead8; }
.badge.near { color: #8d6416; background: #fff5db; border-color: #efd38c; }
.badge.bad { color: var(--bad); background: #faeeee; border-color: #f0c7c7; }
.badge.unknown, .badge.none { color: var(--unknown); background: #f0f2f5; border-color: #d9dee7; }
.badge.kept { color: var(--kept); background: #ecf5fd; border-color: #c7dff3; }
.badge.duplicate { color: var(--duplicate); background: #faf4e5; border-color: #ecdba8; }
.badge.rejected { color: var(--rejected); background: #fbedf1; border-color: #efc7d1; }
.verdict {
  display: flex;
  gap: 12px;
  align-items: center;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: var(--panel);
  padding: 14px 16px;
  margin-bottom: 18px;
}
.verdict strong { font-size: 18px; }
.verdict.good strong { color: var(--good); }
.verdict.bad strong { color: var(--bad); }
.verdict.unknown strong { color: var(--unknown); }
.proof { display: grid; gap: 10px; }
.proof-step { border: 1px solid var(--line); border-radius: 8px; padding: 12px; background: #fbfcfe; }
.proof-rule { font-weight: 750; margin-bottom: 8px; }
.expr { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; overflow-wrap: anywhere; }
.note { color: var(--muted); margin-top: 6px; }
.candidate-list { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 12px; }
.candidate-item { border: 1px solid var(--line); border-radius: 8px; padding: 12px; background: #fbfcfe; }
.candidate-title { display: flex; align-items: center; justify-content: space-between; gap: 10px; margin-bottom: 8px; }
.candidate-title strong { min-width: 0; overflow-wrap: anywhere; }
.candidate-title .badge { flex: 0 0 auto; }
.candidate-title a { color: var(--kept); font-weight: 750; text-decoration: none; overflow-wrap: anywhere; }
.candidate-meta { color: var(--muted); font-size: 12px; margin-bottom: 8px; overflow-wrap: anywhere; }
.candidate-note {
  color: var(--muted);
  font-size: 12px;
  margin: 8px 0;
  overflow-wrap: anywhere;
}
.candidate-note.failed { color: var(--bad); }
.candidate-rhs { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 12px; overflow-wrap: anywhere; }
table { width: 100%; border-collapse: collapse; }
th, td { border-bottom: 1px solid var(--line); padding: 8px 10px; text-align: left; vertical-align: top; }
th { color: var(--muted); font-size: 12px; background: #fbfcfe; position: sticky; top: 0; }
td { overflow-wrap: anywhere; }
ol.counter-list { padding-left: 22px; margin: 0; }
ol.counter-list li { margin: 6px 0; }
@media (max-width: 760px) {
  main { padding: 18px; }
  .grid.two { grid-template-columns: 1fr; }
  .process-flow, .guide-grid, .pipeline-flow, .candidate-progress-list, .candidate-list, .overview-grid, .overview-steps { grid-template-columns: 1fr; }
  .pipeline-step-card::after { display: none; }
  .process-step { min-height: auto; }
  .bar-row { grid-template-columns: 1fr; }
  .bar-count { text-align: left; }
}
"""


def card_grid(cards: list[tuple[str, Any]]) -> str:
    items = []
    for label, value in cards:
        items.append(
            f"""
            <div class="card">
              <div class="label">{escape(str(label))}</div>
              <div class="value">{escape(str(value))}</div>
            </div>
            """
        )
    return '<section class="cards">' + "\n".join(items) + "</section>"


def panel(title: str, content: str) -> str:
    return f'<section class="panel"><h2>{escape(title)}</h2>{content}</section>'


def details_panel(title: str, content: str, *, open_by_default: bool = False) -> str:
    open_attr = " open" if open_by_default else ""
    return f'<details class="panel details-panel"{open_attr}><summary>{escape(title)}</summary>{content}</details>'


def bar_chart(counter: Counter[str], *, limit: int | None = None) -> str:
    if not counter:
        return '<p class="muted">No data.</p>'
    max_value = max(counter.values())
    rows = []
    items = counter.most_common(limit)
    hidden = len(counter) - len(items)
    for key, count in items:
        width = 100 * count / max_value if max_value else 0
        rows.append(
            f"""
            <div class="bar-row">
              <div class="bar-label">{escape(str(key))}</div>
              <div class="bar-track"><div class="bar-fill" style="width:{width:.1f}%"></div></div>
              <div class="bar-count">{count}</div>
            </div>
            """
        )
    if hidden > 0:
        rows.append(f'<p class="viz-note">{hidden} lower-frequency entries hidden.</p>')
    return "\n".join(rows)


def case_coverage_panel(case_counts: Counter[tuple[str, int, int]]) -> str:
    if not case_counts:
        return '<p class="viz-note">No run cases recorded yet.</p>'

    repeat_histogram = Counter(case_counts.values())
    max_value = max(repeat_histogram.values())
    rows = []
    for repeat_count in sorted(repeat_histogram):
        bucket_count = repeat_histogram[repeat_count]
        width = 100 * bucket_count / max_value if max_value else 0
        label = f"{repeat_count} run" if repeat_count == 1 else f"{repeat_count} runs"
        rows.append(
            f"""
            <div class="bar-row">
              <div class="bar-label">{escape(label)}</div>
              <div class="bar-track"><div class="bar-fill" style="width:{width:.1f}%"></div></div>
              <div class="bar-count">{bucket_count}</div>
            </div>
            """
        )

    total_cases = len(case_counts)
    total_runs = sum(case_counts.values())
    singleton_cases = repeat_histogram.get(1, 0)
    repeated_cases = total_cases - singleton_cases
    top_case, top_count = case_counts.most_common(1)[0]
    top_target, top_depth, top_width = top_case
    summary = (
        f'<p class="viz-note">{total_cases} unique case groups across {total_runs} runs. '
        f'{singleton_cases} singletons, {repeated_cases} repeated groups. '
        f'Most repeated: <code>{escape(top_target)}</code> depth={top_depth} width={top_width} ({top_count} runs).</p>'
    )
    return summary + "\n".join(rows)


def ordered_counter(counter: Counter[str], *, limit: int = 12) -> str:
    if not counter:
        return "<p>No proof paths recorded.</p>"
    rows = []
    for key, count in counter.most_common(limit):
        rows.append(f"<li><span class=\"expr\">{escape(key)}</span> <strong>{count}</strong></li>")
    hidden = len(counter) - len(rows)
    note = f'<p class="viz-note">{hidden} lower-frequency proof paths hidden.</p>' if hidden > 0 else ""
    return '<ol class="counter-list">' + "\n".join(rows) + "</ol>" + note


def table(headers: list[str], rows: list[str], *, raw_rows: bool = False) -> str:
    head = "".join(f"<th>{escape(header)}</th>" for header in headers)
    body = "\n".join(rows if raw_rows else [escape(row) for row in rows])
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def table_row(cells: list[str], *, raw: bool = False) -> str:
    if raw:
        return "<tr>" + "".join(f"<td>{cell}</td>" for cell in cells) + "</tr>"
    return "<tr>" + "".join(f"<td>{escape(cell)}</td>" for cell in cells) + "</tr>"


def badge(value: str) -> str:
    klass = STATUS_CLASS.get(value, "none")
    return f'<span class="badge {klass}">{escape(value)}</span>'


def proof_path(proof: list[dict[str, Any]]) -> str:
    if not proof:
        return "<p>No proof path recorded.</p>"
    steps = []
    for step in proof:
        steps.append(
            f"""
            <div class="proof-step">
              <div class="proof-rule">{escape(str(step.get("index", "")))}. {escape(str(step.get("rule_name", "")))}</div>
              <div class="expr">{escape(text_of(step.get("before")))}</div>
              <div class="expr">&darr;</div>
              <div class="expr">{escape(text_of(step.get("after")))}</div>
              <div class="note">{escape(str(step.get("note", "")))}</div>
            </div>
            """
        )
    return '<div class="proof">' + "\n".join(steps) + "</div>"


def proof_path_text(run: RunReport) -> str:
    best = run.best_state
    if not best:
        return ""
    proof = best.get("proof", [])
    return " -> ".join(str(step.get("rule_name", "")) for step in proof)


def all_child_states(runs: list[RunReport]) -> list[dict[str, Any]]:
    states: list[dict[str, Any]] = []
    for run in runs:
        for transition in run.transitions:
            child = transition.get("child_state")
            if child is not None:
                states.append(child)
        if run.best_state is not None:
            states.append(run.best_state)
    return states


def case_summary_table(
    runs: list[RunReport] | list[IndexedRun],
    limit: int = 12,
    *,
    generated_run_ids: set[str] | None = None,
) -> str:
    if generated_run_ids is None:
        generated_run_ids = {run.run_id for run in runs}

    groups: dict[tuple[str, int, int], list[RunReport | IndexedRun]] = {}
    for run in runs:
        groups.setdefault(run.case_key, []).append(run)

    rows = []
    sorted_groups = sorted(
        groups.items(),
        key=lambda item: (-len(item[1]), item[0][0], item[0][1], item[0][2]),
    )
    for key, group in sorted_groups[:limit]:
        target, depth, width = key
        closures = Counter(run.best_closure for run in group)
        representative = sorted(group, key=lambda run: (-run.best_score, run.run_id))[0]
        if representative.run_id in generated_run_ids:
            run_cell = f'<a href="runs/{escape(representative.run_id)}.html">{escape(representative.run_id)}</a>'
        else:
            run_cell = escape(representative.run_id)
        closure_bits = " ".join(f"{badge(status)} {count}" for status, count in closures.most_common())
        rows.append(
            table_row(
                [
                    escape(target),
                    str(depth),
                    str(width),
                    str(len(group)),
                    closure_bits,
                    run_cell,
                ],
                raw=True,
            )
        )
    hidden = max(0, len(sorted_groups) - limit)
    note = ""
    if hidden:
        note = f'<p class="viz-note">Showing top {limit} case groups by run count. {hidden} additional groups are hidden from this overview.</p>'
    return note + table(["Target", "Depth", "Width", "Runs", "Best Closures", "Representative"], rows, raw_rows=True)


def young_phase_map(transitions: list[dict[str, Any]]) -> str:
    raw_points = [point for point in (young_point(transition) for transition in transitions) if point is not None]
    points = aggregate_young_points(raw_points)
    width = 700
    height = 390
    left = 58
    top = 26
    plot_w = 500
    plot_h = 294
    alpha_max = 1.95
    beta_max = 6.0
    cols = 26
    rows = 18

    def px(alpha: float) -> float:
        return left + (alpha / alpha_max) * plot_w

    def py(beta: float) -> float:
        return top + plot_h - (beta / beta_max) * plot_h

    cells = []
    for ix in range(cols):
        for iy in range(rows):
            alpha = (ix + 0.5) * alpha_max / cols
            beta = (iy + 0.5) * beta_max / rows
            ode_power = beta / max(2 - alpha, 0.05)
            x = left + ix * plot_w / cols
            y = top + (rows - iy - 1) * plot_h / rows
            cells.append(
                f'<rect class="heat-cell" x="{x:.2f}" y="{y:.2f}" '
                f'width="{plot_w / cols + 0.2:.2f}" height="{plot_h / rows + 0.2:.2f}" '
                f'fill="{severity_color(ode_power)}" />'
            )

    grid = []
    for alpha in [0, 0.5, 1.0, 1.5, 1.95]:
        x = px(alpha)
        grid.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_h}" stroke="#ffffff" stroke-opacity="0.42" />')
        grid.append(f'<text class="tick-label" x="{x:.1f}" y="{top + plot_h + 20}" text-anchor="middle">{alpha:g}</text>')
    for beta in [0, 1, 2, 3, 4, 5, 6]:
        y = py(beta)
        grid.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#ffffff" stroke-opacity="0.42" />')
        grid.append(f'<text class="tick-label" x="{left - 12}" y="{y + 3:.1f}" text-anchor="end">{beta:g}</text>')

    point_svg = []
    for index, point in enumerate(points, start=1):
        x = px(float(point["alpha"]))
        y = py(float(point["beta"]))
        ode_power = float(point["ode_power"])
        count = int(point["count"])
        if ode_power <= 1:
            closure = "GOOD"
            radius = 7 + min(count, 12) * 0.45
            fill = status_color("GOOD")
            opacity = 0.92
            stroke = "#111827"
            stroke_width = 1.4
            show_label = True
        elif ode_power <= 1.2:
            closure = "NEAR"
            radius = 4 + min(count, 10) * 0.25
            fill = severity_color(1.15)
            opacity = 0.78
            stroke = "#111827"
            stroke_width = 0.8
            show_label = index <= 8
        else:
            closure = "BAD"
            radius = 2 + min(2.4, math.log1p(count) * 0.32)
            fill = status_color("BAD")
            opacity = 0.18 if ode_power > 2 else 0.26
            stroke = "none"
            stroke_width = 0
            show_label = False
        point_svg.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius:.1f}" fill="{fill}" '
            f'fill-opacity="{opacity:.2f}" stroke="{stroke}" stroke-width="{stroke_width}">'
            f'<title>{escape(closure)} | {escape(point["label"])}</title></circle>'
        )
        if show_label:
            point_svg.append(
                f'<text class="point-label light" x="{x:.1f}" y="{y + 3.5:.1f}" text-anchor="middle">{index}</text>'
            )

    if not point_svg:
        point_svg.append(
            f'<text class="axis-label" x="{left + plot_w / 2}" y="{top + plot_h / 2}" text-anchor="middle">'
            "No Young transitions in this selection</text>"
        )

    critical = []
    for alpha in [i * alpha_max / 80 for i in range(81)]:
        beta = 2 - alpha
        if beta < 0:
            continue
        critical.append(f"{px(alpha):.1f},{py(beta):.1f}")
    critical_polyline = " ".join(critical)

    return f"""
    <svg class="viz" viewBox="0 0 {width} {height}" role="img" aria-label="Young absorption phase map">
      <text class="viz-title" x="{left}" y="16">Young absorption phase (color = ODE power)</text>
      <rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="#eef2f7" rx="6" />
      {''.join(cells)}
      {''.join(grid)}
      <polyline points="{critical_polyline}" fill="none" stroke="#111827" stroke-width="2.5" stroke-dasharray="6 5">
        <title>critical line: beta = 2 - alpha, ODE power = 1</title>
      </polyline>
      {''.join(point_svg)}
      <rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="none" stroke="#20242c" stroke-width="1.2" rx="6" />
      <text class="axis-label" x="{left + plot_w / 2}" y="{height - 20}" text-anchor="middle">dissipation exponent &amp;alpha;</text>
      <text class="axis-label" transform="translate(16 {top + plot_h / 2}) rotate(-90)" text-anchor="middle">controlled exponent &amp;beta;</text>
      <g transform="translate({left + plot_w + 34} {top + 20})">
        <text class="tick-label" x="0" y="0">GOOD</text>
        <rect x="0" y="8" width="18" height="18" fill="{severity_color(0.8)}" />
        <text class="tick-label" x="26" y="22">&lt;= 1</text>
        <rect x="0" y="36" width="18" height="18" fill="{severity_color(1.4)}" />
        <text class="tick-label" x="26" y="50">near</text>
        <rect x="0" y="64" width="18" height="18" fill="{severity_color(2.6)}" />
        <text class="tick-label" x="26" y="78">bad</text>
        <rect x="0" y="92" width="18" height="18" fill="{severity_color(5.0)}" />
        <text class="tick-label" x="26" y="106">severe</text>
      </g>
    </svg>
    <p class="viz-note">Formula: ||Domega||2^alpha ||omega||2^beta maps to X^(beta/(2-alpha)). Dashed curve is the closure threshold. BAD points are deliberately small and faint; only GOOD or near-threshold points get visual emphasis.</p>
    {young_point_table(points)}
    """


def candidate_landscape(states: list[dict[str, Any]]) -> str:
    if not states:
        return '<p class="viz-note">No candidate states recorded.</p>'

    grouped_states = aggregate_states(states)
    width = 620
    height = 390
    left = 56
    top = 28
    plot_w = 500
    plot_h = 292
    points = []
    for state in grouped_states:
        rhs = state.get("rhs") or {}
        complexity = ast_complexity((rhs.get("ast") or {}))
        score = int(state.get("score", 0))
        closure = ((state.get("closure") or {}).get("status", "NONE"))
        depth = int(state.get("depth", 0))
        count = int(state.get("count", 1))
        points.append((complexity, score, closure, depth, text_of(rhs), count))

    min_c = min(point[0] for point in points)
    max_c = max(point[0] for point in points)
    min_s = min(point[1] for point in points)
    max_s = max(point[1] for point in points)
    if min_c == max_c:
        max_c += 1
    if min_s == max_s:
        max_s += 1

    def px(complexity: int) -> float:
        return left + (complexity - min_c) / (max_c - min_c) * plot_w

    def py(score: int) -> float:
        return top + plot_h - (score - min_s) / (max_s - min_s) * plot_h

    circles = []
    for complexity, score, closure, depth, label, count in points:
        x = px(complexity)
        y = py(score)
        radius = 4.5 + min(depth, 6) * 1.2 + min(count - 1, 12) * 0.35
        circles.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius:.1f}" fill="{status_color(closure)}" '
            f'fill-opacity="0.78" stroke="#20242c" stroke-width="0.8">'
            f'<title>{escape(closure)} | score={score} | complexity={complexity} | count={count} | {escape(label)}</title></circle>'
        )

    ticks = []
    for value in [min_c, (min_c + max_c) // 2, max_c]:
        x = px(value)
        ticks.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_h}" stroke="#dfe4ec" />')
        ticks.append(f'<text class="tick-label" x="{x:.1f}" y="{top + plot_h + 20}" text-anchor="middle">{value}</text>')
    for value in [min_s, int((min_s + max_s) / 2), max_s]:
        y = py(value)
        ticks.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#dfe4ec" />')
        ticks.append(f'<text class="tick-label" x="{left - 12}" y="{y + 3:.1f}" text-anchor="end">{value}</text>')

    return f"""
    <svg class="viz" viewBox="0 0 {width} {height}" role="img" aria-label="Candidate complexity score landscape">
      <text class="viz-title" x="{left}" y="16">Candidate landscape: complexity x score</text>
      <rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" rx="6" fill="#fbfcfe" stroke="#dfe4ec" />
      {''.join(ticks)}
      {''.join(circles)}
      <text class="axis-label" x="{left + plot_w / 2}" y="{height - 20}" text-anchor="middle">AST complexity</text>
      <text class="axis-label" transform="translate(16 {top + plot_h / 2}) rotate(-90)" text-anchor="middle">score</text>
      {legend(left + plot_w - 230, top + 16)}
    </svg>
    <p class="viz-note">Identical states are aggregated. Bigger circles mean deeper candidates and/or repeated visits. Hover a point to inspect the expression.</p>
    """


def rule_transition_heatmap(runs: list[RunReport]) -> str:
    pairs: Counter[tuple[str, str]] = Counter()
    rules: set[str] = set()
    for run in runs:
        best = run.best_state
        if not best:
            continue
        path = [str(step.get("rule_name", "")) for step in best.get("proof", []) if step.get("rule_name")]
        for rule in path:
            rules.add(rule)
        for left, right in zip(path, path[1:]):
            pairs[(left, right)] += 1

    if not rules:
        return '<p class="viz-note">No proof path transitions recorded.</p>'

    ordered = sorted(rules)
    short_labels = {rule: short_rule_label(rule) for rule in ordered}
    cell = 46
    label_w = 72
    top = 54
    left = label_w
    width = left + cell * len(ordered) + 260
    height = top + cell * len(ordered) + 68
    max_count = max(pairs.values()) if pairs else 1

    row_labels = []
    col_labels = []
    cells = []
    for i, rule in enumerate(ordered):
        y = top + i * cell + cell / 2 + 4
        row_labels.append(
            f'<text class="tick-label" x="{label_w - 10}" y="{y:.1f}" text-anchor="end">'
            f'<title>{escape(rule)}</title>{escape(short_labels[rule])}</text>'
        )
        x = left + i * cell + cell / 2
        col_labels.append(
            f'<text class="tick-label" x="{x:.1f}" y="{top - 14}" text-anchor="middle">'
            f'<title>{escape(rule)}</title>{escape(short_labels[rule])}</text>'
        )

    for row, source in enumerate(ordered):
        for col, target in enumerate(ordered):
            count = pairs.get((source, target), 0)
            intensity = count / max_count if max_count else 0
            fill = mix_color("#f3f6fb", "#425f8f", intensity)
            x = left + col * cell
            y = top + row * cell
            cells.append(
                f'<rect x="{x}" y="{y}" width="{cell - 2}" height="{cell - 2}" rx="5" fill="{fill}" stroke="#ffffff">'
                f'<title>{escape(source)} -&gt; {escape(target)}: {count}</title></rect>'
            )
            if count:
                cells.append(f'<text class="point-label" x="{x + cell / 2 - 1:.1f}" y="{y + cell / 2 + 4:.1f}" text-anchor="middle">{count}</text>')

    return f"""
    <svg class="viz" viewBox="0 0 {width} {height}" role="img" aria-label="Rule transition heatmap">
      <text class="viz-title" x="0" y="18">Best-proof rule transition matrix</text>
      <text class="axis-label" x="{left + cell * len(ordered) / 2}" y="{height - 16}" text-anchor="middle">next rule</text>
      <text class="axis-label" transform="translate(16 {top + cell * len(ordered) / 2}) rotate(-90)" text-anchor="middle">current rule</text>
      {''.join(col_labels)}
      {''.join(row_labels)}
      {''.join(cells)}
      {heatmap_legend(left + cell * len(ordered) + 18, top, ordered, short_labels)}
    </svg>
    <p class="viz-note">Darker cells mark more frequent adjacent rule transitions in best derivations. Axis labels are abbreviated to avoid overlap.</p>
    """


def closure_spectrum(states: list[dict[str, Any]]) -> str:
    states = aggregate_states(states)
    counts = Counter(((state.get("closure") or {}).get("status", "NONE")) for state in states)
    total = sum(counts.values())
    if not total:
        return '<p class="viz-note">No closure states recorded.</p>'

    width = 620
    height = 180
    x = 36
    y = 66
    bar_w = 540
    bar_h = 30
    order = ["GOOD", "UNKNOWN", "BAD", "NONE"]
    segments = []
    cursor = x
    for status in order:
        count = counts.get(status, 0)
        if not count:
            continue
        seg_w = bar_w * count / total
        segments.append(
            f'<rect x="{cursor:.1f}" y="{y}" width="{seg_w:.1f}" height="{bar_h}" fill="{status_color(status)}">'
            f'<title>{escape(status)}: {count}</title></rect>'
        )
        if seg_w > 42:
            segments.append(
                f'<text class="point-label" x="{cursor + seg_w / 2:.1f}" y="{y + 20}" text-anchor="middle" fill="#fff">{escape(status)}</text>'
            )
        cursor += seg_w

    labels = []
    label_x = x
    for status in order:
        count = counts.get(status, 0)
        if not count:
            continue
        labels.append(
            f'<span class="badge {STATUS_CLASS.get(status, "none")}">{escape(status)} {count}</span>'
        )
        label_x += 1

    return f"""
    <svg class="viz" viewBox="0 0 {width} {height}" role="img" aria-label="Closure spectrum">
      <text class="viz-title" x="{x}" y="24">Closure spectrum across candidate states</text>
      <rect x="{x}" y="{y}" width="{bar_w}" height="{bar_h}" fill="#edf1f6" rx="8" />
      <g>{''.join(segments)}</g>
      <text class="axis-label" x="{x}" y="{y + 58}">unique candidate count: {total}</text>
    </svg>
    <p class="viz-note">{' '.join(labels)}</p>
    """


def young_point(transition: dict[str, Any]) -> dict[str, Any] | None:
    if transition.get("action") != "Young":
        return None
    ast = ((transition.get("rhs_before") or {}).get("ast") or {})
    alpha, beta = exponents_for_young(ast)
    if alpha is None or beta is None or alpha >= 2:
        return None
    ode_power = beta / (Fraction(2) - alpha)
    return {
        "alpha": alpha,
        "beta": beta,
        "ode_power": ode_power,
        "count": 1,
        "label": f"alpha={format_float(alpha)}, beta={format_float(beta)}, ODE power={format_float(ode_power)}",
    }


def aggregate_young_points(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], dict[str, Any]] = {}
    for point in points:
        key = (str(point["alpha"]), str(point["beta"]), str(point["ode_power"]))
        if key not in grouped:
            grouped[key] = dict(point)
            grouped[key]["count"] = 0
        grouped[key]["count"] += 1

    aggregated = []
    for point in grouped.values():
        point["label"] = (
            f"alpha={format_float(point['alpha'])}, beta={format_float(point['beta'])}, "
            f"ODE power={format_float(point['ode_power'])}, count={point['count']}"
        )
        aggregated.append(point)
    return sorted(aggregated, key=lambda point: (float(point["alpha"]), float(point["beta"])))


def young_point_table(points: list[dict[str, Any]], limit: int = 8) -> str:
    if not points:
        return ""
    rows = []
    candidates = [point for point in points if float(point["ode_power"]) <= 1.2]
    if not candidates:
        bad_count = sum(int(point["count"]) for point in points)
        return f'<p class="viz-note">No near-closure Young phase points in this view. {bad_count} BAD Young transitions are suppressed to faint dots.</p>'

    ranked = sorted(candidates, key=lambda point: (float(point["ode_power"]), -int(point["count"]), float(point["alpha"])))
    for index, point in enumerate(ranked[:limit], start=1):
        closure = "GOOD" if point["ode_power"] <= 1 else "NEAR"
        rows.append(
            table_row(
                [
                    str(index),
                    format_float(point["alpha"]),
                    format_float(point["beta"]),
                    format_float(point["ode_power"]),
                    str(point["count"]),
                    badge(closure),
                ],
                raw=True,
            )
        )
    hidden = max(0, len(candidates) - len(rows))
    suppressed = sum(int(point["count"]) for point in points if float(point["ode_power"]) > 1.2)
    notes = []
    if hidden:
        notes.append(f"{hidden} lower-priority near-threshold points hidden.")
    if suppressed:
        notes.append(f"{suppressed} BAD Young transitions suppressed to faint dots.")
    note = f'<p class="viz-note">{" ".join(notes)}</p>' if notes else ""
    return '<div class="mini-table">' + table(["#", "alpha", "beta", "ODE power", "Count", "Closure"], rows, raw_rows=True) + "</div>" + note


def aggregate_states(states: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for state in states:
        if not isinstance(state, dict):
            continue
        key = str(state.get("state_id") or ((state.get("rhs") or {}).get("fingerprint")) or id(state))
        if key not in grouped:
            grouped[key] = dict(state)
            grouped[key]["count"] = 0
        grouped[key]["count"] += 1
        grouped[key]["depth"] = max(int(grouped[key].get("depth", 0)), int(state.get("depth", 0)))
    return list(grouped.values())


def exponents_for_young(ast: dict[str, Any]) -> tuple[Fraction | None, Fraction | None]:
    alpha = Fraction(0)
    beta = Fraction(0)
    found_alpha = False
    found_beta = False
    for term in product_terms(ast):
        exponent = Fraction(1)
        base = term
        if term.get("type") == "Power":
            exponent = parse_fraction(str(term.get("exp", "1")))
            base = term.get("base") or {}
        if base.get("type") != "Norm":
            continue
        if is_domega_l2_norm(base):
            alpha += exponent
            found_alpha = True
        elif is_omega_l2_norm(base):
            beta += exponent
            found_beta = True
    return (alpha if found_alpha else None, beta if found_beta else None)


def product_terms(ast: dict[str, Any]) -> list[dict[str, Any]]:
    if ast.get("type") == "Product":
        return [term for term in ast.get("terms", []) if isinstance(term, dict)]
    return [ast]


def is_omega_l2_norm(ast: dict[str, Any]) -> bool:
    expr = ast.get("expr") or {}
    return ast.get("p") == "2" and expr.get("type") == "Variable" and expr.get("name") == "omega"


def is_domega_l2_norm(ast: dict[str, Any]) -> bool:
    expr = ast.get("expr") or {}
    inner = expr.get("expr") or {}
    return (
        ast.get("p") == "2"
        and expr.get("type") == "Derivative"
        and expr.get("order") == 1
        and inner.get("type") == "Variable"
        and inner.get("name") == "omega"
    )


def ast_complexity(ast: dict[str, Any]) -> int:
    if not isinstance(ast, dict):
        return 0
    typ = ast.get("type")
    if typ in {"Constant", "Variable"}:
        return 1
    if typ in {"Derivative", "Norm", "Integral"}:
        return 1 + ast_complexity(ast.get("expr") or {})
    if typ == "Power":
        return 1 + ast_complexity(ast.get("base") or {})
    if typ in {"Product", "Sum"}:
        return 1 + sum(ast_complexity(term) for term in ast.get("terms", []) if isinstance(term, dict))
    return 1


def parse_fraction(value: str) -> Fraction:
    return Fraction(value)


def format_float(value: Fraction | float) -> str:
    return f"{float(value):.2f}".rstrip("0").rstrip(".")


def severity_color(ode_power: Fraction | float) -> str:
    value = float(ode_power)
    if value <= 1:
        return mix_color("#1fbf8f", "#f2d45c", max(value - 0.4, 0) / 0.6)
    if value <= 2:
        return mix_color("#f2d45c", "#f08a4b", value - 1)
    if value <= 4:
        return mix_color("#f08a4b", "#d64f7a", (value - 2) / 2)
    return "#743e63"


def status_color(status: str) -> str:
    return {
        "GOOD": "#1fbf8f",
        "BAD": "#d64f7a",
        "UNKNOWN": "#7b8798",
        "NONE": "#c9d0db",
    }.get(status, "#7b8798")


def mix_color(left: str, right: str, amount: float) -> str:
    amount = max(0.0, min(1.0, amount))
    lrgb = hex_to_rgb(left)
    rrgb = hex_to_rgb(right)
    mixed = tuple(round(l + (r - l) * amount) for l, r in zip(lrgb, rrgb))
    return "#" + "".join(f"{channel:02x}" for channel in mixed)


def hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return (int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16))


def legend(x: float, y: float) -> str:
    items = [("GOOD", status_color("GOOD")), ("UNKNOWN", status_color("UNKNOWN")), ("BAD", status_color("BAD"))]
    rows = []
    for index, (label, color) in enumerate(items):
        yy = y + index * 22
        rows.append(f'<circle cx="{x}" cy="{yy}" r="6" fill="{color}" />')
        rows.append(f'<text class="tick-label" x="{x + 12}" y="{yy + 4}">{label}</text>')
    return "<g>" + "".join(rows) + "</g>"


def short_rule_label(rule: str) -> str:
    labels = {
        "Biot-Savart": "BS",
        "Gagliardo-Nirenberg": "GN",
        "Holder": "Ho",
        "Integral-to-Norm": "I2N",
        "Sobolev": "Sob",
        "Young": "Y",
    }
    return labels.get(rule, rule[:5])


def heatmap_legend(x: float, y: float, rules: list[str], short_labels: dict[str, str]) -> str:
    rows = ['<text class="tick-label" x="0" y="0">Legend</text>']
    for index, rule in enumerate(rules):
        yy = 20 + index * 18
        rows.append(
            f'<text class="tick-label" x="0" y="{yy}"><tspan font-weight="700">'
            f'{escape(short_labels[rule])}</tspan> = {escape(rule)}</text>'
        )
    return f'<g transform="translate({x:.1f} {y:.1f})">' + "".join(rows) + "</g>"


def text_of(record: Any) -> str:
    if not isinstance(record, dict):
        return ""
    return str(record.get("text", ""))
