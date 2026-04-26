#!/usr/bin/env python3
"""Run randomized AISIS exploration over generated target families."""

from __future__ import annotations

import argparse
import json
import random
import sqlite3
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.rules import Rule, default_rules
from engine.recording import RunRecorder
from engine.runner import SearchRunner
from ns.generated import BASE_AND_GENERATED_TARGETS, NONLINEAR_TARGETS
from ns.targets import TargetSpec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run randomized AISIS exploration")
    parser.add_argument("--budget", type=int, default=100)
    parser.add_argument("--seed", type=int, default=260426)
    parser.add_argument("--depths", nargs="+", type=int, default=[2, 3, 4, 5, 6, 7])
    parser.add_argument("--widths", nargs="+", type=int, default=[10, 20, 50, 100])
    parser.add_argument("--targets", nargs="+", default=None)
    parser.add_argument("--mode", choices=["adaptive", "uniform"], default="adaptive")
    parser.add_argument(
        "--target-policy",
        choices=["coverage", "family-balanced", "frontier"],
        default="frontier",
        help="How adaptive mode chooses target families before choosing a concrete case.",
    )
    parser.add_argument(
        "--rule-policy",
        choices=["default", "shuffle", "sample"],
        default="sample",
        help="How each run chooses inequality rules.",
    )
    parser.add_argument("--min-rules", type=int, default=4, help="Minimum rules kept when --rule-policy=sample.")
    parser.add_argument(
        "--failure-penalty",
        type=float,
        default=0.25,
        help="Weight multiplier for targets that already failed verification relevance.",
    )
    parser.add_argument("--dedupe-cases", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--case-count-source", choices=["auto", "index", "files"], default="auto")
    parser.add_argument("--data-dir", default=str(ROOT / "data"))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def case_key(target_name: str, depth: int, width: int) -> tuple[str, int, int]:
    return (target_name, depth, width)


def load_existing_case_counts(data_dir: Path, *, source: str = "auto") -> tuple[Counter[tuple[str, int, int]], str]:
    index_path = data_dir / "index" / "aisis.sqlite"
    if source in {"auto", "index"} and index_path.exists():
        try:
            return load_existing_case_counts_from_index(index_path), f"index:{index_path}"
        except sqlite3.Error:
            if source == "index":
                raise

    if source == "index":
        raise SystemExit(f"case count index not found: {index_path}")
    return load_existing_case_counts_from_files(data_dir), "files"


def load_existing_case_counts_from_index(index_path: Path) -> Counter[tuple[str, int, int]]:
    counts: Counter[tuple[str, int, int]] = Counter()
    with sqlite3.connect(index_path) as connection:
        rows = connection.execute(
            "select target_name, depth, width, count(*) from runs group by target_name, depth, width"
        ).fetchall()
    for target_name, depth, width, count in rows:
        counts[case_key(str(target_name), int(depth), int(width))] = int(count)
    return counts


def load_existing_case_counts_from_files(data_dir: Path) -> Counter[tuple[str, int, int]]:
    counts: Counter[tuple[str, int, int]] = Counter()
    for path in sorted((data_dir / "runs").glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("event") != "run_start":
                continue
            search = row.get("search") or {}
            counts[
                case_key(
                    str(row.get("target_name", "unknown")),
                    int(search.get("depth", 0)),
                    int(search.get("width", 0)),
                )
            ] += 1
            break
    return counts


def select_targets(target_names: list[str] | None) -> list[TargetSpec]:
    if target_names is None:
        return [BASE_AND_GENERATED_TARGETS[name] for name in sorted(BASE_AND_GENERATED_TARGETS)]

    unknown = [name for name in target_names if name not in BASE_AND_GENERATED_TARGETS]
    if unknown:
        choices = ", ".join(sorted(BASE_AND_GENERATED_TARGETS))
        raise SystemExit(f"unknown targets {unknown}; choices: {choices}")
    return [BASE_AND_GENERATED_TARGETS[name] for name in target_names]


def target_family(target_name: str) -> str:
    if target_name in NONLINEAR_TARGETS:
        return "nonlinear"
    if target_name in {"omega_L6_squared", "omega_L6_crit_q2"}:
        return "endpoint_embedding"
    if target_name.startswith("omega_L") and "_crit_q" in target_name:
        return "critical_norm"
    if target_name.startswith("omega_") or target_name.startswith("omega_L"):
        return "base_norm"
    return "other"


def family_weight(family: str, *, policy: str) -> float:
    if policy == "family-balanced":
        return 1.0
    weights = {
        "nonlinear": 8.0,
        "base_norm": 1.8,
        "critical_norm": 1.0,
        "endpoint_embedding": 0.15,
        "other": 1.0,
    }
    return weights.get(family, 1.0)


def load_verification_penalties(data_dir: Path, *, failure_penalty: float) -> dict[str, float]:
    queue_path = data_dir / "promotions" / "verification_queue.json"
    if failure_penalty >= 1.0 or not queue_path.exists():
        return {}

    try:
        queue = json.loads(queue_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}

    penalties: dict[str, float] = {}
    for candidate in queue.get("candidates", []):
        if not isinstance(candidate, dict):
            continue
        target_name = str(candidate.get("target_name", ""))
        status = str(candidate.get("queue_status", ""))
        checks = (candidate.get("verification") or {}).get("checks", [])
        reason = " ".join(str(check.get("reason", "")) for check in checks if isinstance(check, dict))
        if status == "failed_relevance" or "pure norm embedding" in reason:
            penalties[target_name] = min(penalties.get(target_name, 1.0), max(0.01, failure_penalty))
    return penalties


def build_target_weights(targets: list[TargetSpec], penalties: dict[str, float]) -> dict[str, float]:
    weights: dict[str, float] = {}
    for target in targets:
        weight = penalties.get(target.name, 1.0)
        if target_family(target.name) == "endpoint_embedding":
            weight *= 0.2
        weights[target.name] = max(0.01, weight)
    return weights


def weighted_choice(items: list[object], weights: list[float], rng: random.Random) -> object:
    total = sum(max(0.0, weight) for weight in weights)
    if total <= 0:
        return rng.choice(items)
    threshold = rng.random() * total
    running = 0.0
    for item, weight in zip(items, weights):
        running += max(0.0, weight)
        if running >= threshold:
            return item
    return items[-1]


def build_rule_plan(
    *,
    target: TargetSpec,
    rng: random.Random,
    policy: str,
    min_rules: int,
) -> list[Rule]:
    rules = default_rules()
    if policy == "default":
        return rules

    if policy == "shuffle":
        shuffled = list(rules)
        rng.shuffle(shuffled)
        return shuffled

    by_name = {rule.name: rule for rule in rules}
    family = target_family(target.name)
    required_names = ["Young"]
    if family == "nonlinear":
        required_names.extend(["Holder", "Biot-Savart"])
    elif target.name.startswith("omega_cubic") or target.name.startswith("omega_L"):
        required_names.append("Gagliardo-Nirenberg")
    if "int(" in str(target.expr):
        required_names.append("Integral-to-Norm")

    required = [by_name[name] for name in dict.fromkeys(required_names) if name in by_name]
    required_names_set = {rule.name for rule in required}
    optional = [rule for rule in rules if rule.name not in required_names_set]
    lower = max(0, min_rules - len(required))
    upper = len(optional)
    optional_count = rng.randint(min(lower, upper), upper) if optional else 0
    selected_optional = rng.sample(optional, optional_count) if optional_count else []
    selected = required + selected_optional
    rng.shuffle(selected)
    return selected


def build_jobs(
    *,
    budget: int,
    targets: list[TargetSpec],
    depths: list[int],
    widths: list[int],
    rng: random.Random,
    mode: str,
    dedupe_cases: bool,
    existing_counts: Counter[tuple[str, int, int]],
    target_policy: str = "coverage",
    target_weights: dict[str, float] | None = None,
) -> list[dict[str, object]]:
    jobs: list[dict[str, object]] = []
    running_counts = Counter(existing_counts)
    target_weights = target_weights or {}
    cases = [(target, depth, width) for target in targets for depth in depths for width in widths]
    if not cases:
        return jobs
    cases_by_family: dict[str, list[tuple[TargetSpec, int, int]]] = {}
    for case in cases:
        cases_by_family.setdefault(target_family(case[0].name), []).append(case)

    for index in range(1, budget + 1):
        if mode == "uniform" and not dedupe_cases:
            target = rng.choice(targets)
            depth = rng.choice(depths)
            width = rng.choice(widths)
        elif target_policy in {"family-balanced", "frontier"}:
            family_scores: list[tuple[str, float, float]] = []
            for family, family_cases in cases_by_family.items():
                average_seen = sum(running_counts[case_key(target.name, depth, width)] for target, depth, width in family_cases)
                average_seen /= len(family_cases)
                priority = family_weight(family, policy=target_policy) / (1.0 + average_seen)
                family_scores.append((family, average_seen, priority))

            if target_policy == "family-balanced":
                min_seen = min(score[1] for score in family_scores)
                eligible_families = [score for score in family_scores if score[1] == min_seen]
            else:
                eligible_families = family_scores
            family = weighted_choice(
                [score[0] for score in eligible_families],
                [score[2] for score in eligible_families],
                rng,
            )
            assert isinstance(family, str)
            family_cases = cases_by_family[family]
            min_count = min(running_counts[case_key(target.name, depth, width)] for target, depth, width in family_cases)
            eligible = [
                (target, depth, width)
                for target, depth, width in family_cases
                if running_counts[case_key(target.name, depth, width)] == min_count
            ]
            chosen = weighted_choice(
                eligible,
                [target_weights.get(target.name, 1.0) for target, _, _ in eligible],
                rng,
            )
            target, depth, width = chosen  # type: ignore[misc]
        else:
            min_count = min(running_counts[case_key(target.name, depth, width)] for target, depth, width in cases)
            eligible = [
                (target, depth, width)
                for target, depth, width in cases
                if running_counts[case_key(target.name, depth, width)] == min_count
            ]
            chosen = weighted_choice(
                eligible,
                [target_weights.get(target.name, 1.0) for target, _, _ in eligible],
                rng,
            )
            target, depth, width = chosen  # type: ignore[misc]

        key = case_key(target.name, depth, width)
        before = running_counts[key]
        running_counts[key] += 1
        jobs.append(
            {
                "index": index,
                "target": target,
                "depth": depth,
                "width": width,
                "case_count_before": before,
                "target_family": target_family(target.name),
            }
        )
    return jobs


def main() -> int:
    args = parse_args()
    if args.budget <= 0:
        raise SystemExit("--budget must be positive")
    if any(depth <= 0 for depth in args.depths):
        raise SystemExit("--depths must be positive")
    if any(width <= 0 for width in args.widths):
        raise SystemExit("--widths must be positive")

    rng = random.Random(args.seed)
    data_dir = Path(args.data_dir)
    explore_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ_random")
    targets = select_targets(args.targets)
    penalties = load_verification_penalties(data_dir, failure_penalty=args.failure_penalty)
    target_weights = build_target_weights(targets, penalties)
    existing_counts, case_count_source = load_existing_case_counts(data_dir, source=args.case_count_source)
    jobs = build_jobs(
        budget=args.budget,
        targets=targets,
        depths=args.depths,
        widths=args.widths,
        rng=rng,
        mode=args.mode,
        dedupe_cases=args.dedupe_cases,
        existing_counts=existing_counts,
        target_policy=args.target_policy,
        target_weights=target_weights,
    )

    print(f"Random exploration: {explore_id}")
    print(f"Seed: {args.seed}")
    print(f"Budget: {args.budget}")
    print(f"Mode: {args.mode}")
    print(f"Target policy: {args.target_policy}")
    print(f"Rule policy: {args.rule_policy}")
    print(f"Dedupe cases: {args.dedupe_cases}")
    print(f"Target family size: {len(targets)}")
    print(f"Penalized targets: {len(penalties)}")
    print(f"Existing unique cases: {len(existing_counts)}")
    print(f"Case count source: {case_count_source}")
    print(f"Dry run: {args.dry_run}")

    if args.dry_run:
        for job in jobs[: min(20, len(jobs))]:
            target = job["target"]
            assert isinstance(target, TargetSpec)
            print(
                f"  {job['index']:>4}. {target.name} depth={job['depth']} "
                f"width={job['width']} family={job['target_family']} seen={job['case_count_before']}"
            )
        if len(jobs) > 20:
            print(f"  ... {len(jobs) - 20} more")
        return 0

    batch_dir = data_dir / "batches"
    batch_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "explore_id": explore_id,
        "seed": args.seed,
        "budget": args.budget,
        "mode": args.mode,
        "target_policy": args.target_policy,
        "rule_policy": args.rule_policy,
        "min_rules": args.min_rules,
        "dedupe_cases": args.dedupe_cases,
        "target_family_size": len(targets),
        "penalized_targets": sorted(penalties),
        "existing_unique_cases": len(existing_counts),
        "case_count_source": case_count_source,
        "runs": [],
    }

    closure_counts: Counter[str] = Counter()
    target_counts: Counter[str] = Counter()
    for job in jobs:
        target = job["target"]
        assert isinstance(target, TargetSpec)
        depth = int(job["depth"])
        width = int(job["width"])
        metadata = {
            "random_explore_id": explore_id,
            "random_seed": args.seed,
            "random_index": job["index"],
            "random_budget": args.budget,
            "random_mode": args.mode,
            "target_policy": args.target_policy,
            "target_family": job["target_family"],
            "case_count_before": job["case_count_before"],
        }
        recorder = RunRecorder.create(target.name, data_dir, metadata=metadata)
        rule_plan = build_rule_plan(
            target=target,
            rng=rng,
            policy=args.rule_policy,
            min_rules=args.min_rules,
        )
        result = SearchRunner(depth=depth, width=width, rules=rule_plan, recorder=recorder).run(target)
        best = result.most_informative()
        closure = best.closure.status if best.closure is not None else "NONE"
        closure_counts[closure] += 1
        target_counts[target.name] += 1
        manifest["runs"].append(
            {
                "index": job["index"],
                "target_name": target.name,
                "depth": depth,
                "width": width,
                "case_count_before": job["case_count_before"],
                "target_family": job["target_family"],
                "rules": [rule.name for rule in rule_plan],
                "run_id": recorder.run_id,
                "run_path": str(result.run_path),
                "summary_path": str(result.summary_path),
                "best_closure": closure,
                "best_score": best.score,
            }
        )

    manifest_path = batch_dir / f"{explore_id}.json"
    manifest["manifest_path"] = str(manifest_path)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True) + "\n")

    print()
    print("Completed.")
    print("Closures:", dict(closure_counts))
    print("Targets:", dict(target_counts.most_common()))
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
