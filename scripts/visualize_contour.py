#!/usr/bin/env python3
"""Render spatial field diagnostics for a promoted inequality candidate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.serde import expr_from_json
from verifier.contour import (
    compute_contour_fields,
    render_contour_figure,
    select_frontier_candidates,
    title_for_candidate,
    verification_progress,
)
from verifier.numeric import FieldSample, make_field, np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render spatial field diagnostics for an inequality candidate")
    parser.add_argument("--queue-path", default=str(ROOT / "data" / "promotions" / "verification_queue.json"))
    parser.add_argument("--summary-path", default=None)
    parser.add_argument("--candidate-id", default=None)
    parser.add_argument("--candidate-index", type=int, default=0)
    parser.add_argument("--frontier", action="store_true", help="Render only the candidates that advanced furthest")
    parser.add_argument("--top", type=int, default=6, help="Number of frontier candidates to render")
    parser.add_argument("--dedupe", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--out-dir", default=str(ROOT / "data" / "visualizations"))
    parser.add_argument("--grid-size", type=int, default=64)
    parser.add_argument("--profile", choices=["single_mode", "multi_mode", "localized_bump", "two_scale"], default="two_scale")
    parser.add_argument("--amplitude", type=float, default=1.0)
    parser.add_argument("--frequency", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--slice-axis", choices=["x", "y", "z"], default="z")
    parser.add_argument("--slice-index", type=int, default=None)
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--constant-c", type=float, default=1.0)
    parser.add_argument("--constant-c-eps", type=float, default=1.0)
    parser.add_argument("--dpi", type=int, default=160)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if np is None:
        raise RuntimeError("numpy is required for contour visualization")

    if args.frontier:
        return render_frontier(args)

    candidate = load_candidate(args)
    image_path, metadata_path, fields = render_candidate(candidate, args)
    print(f"Image: {image_path}")
    print(f"Metadata: {metadata_path}")
    print("Aggregates:")
    for key, value in fields.aggregates.items():
        print(f"  {key}: {value:.6g}")
    return 0


def render_frontier(args: argparse.Namespace) -> int:
    candidates = load_queue_candidates(Path(args.queue_path))
    frontier = select_frontier_candidates(candidates, limit=args.top, dedupe=args.dedupe)
    if not frontier:
        raise ValueError("no frontier candidates found")

    records = []
    for rank, candidate in enumerate(frontier, start=1):
        image_path, metadata_path, fields = render_candidate(candidate, args, rank=rank)
        records.append(
            {
                "rank": rank,
                "candidate": candidate,
                "image_path": image_path,
                "metadata_path": metadata_path,
                "aggregates": fields.aggregates,
            }
        )

    index_path = write_frontier_index(Path(args.out_dir), records, args)
    print(f"Frontier candidates: {len(records)}")
    print(f"Index: {index_path}")
    for record in records:
        candidate = record["candidate"]
        print(
            f"  #{record['rank']} progress={verification_progress(candidate)} "
            f"status={candidate.get('queue_status')} target={candidate.get('target_name')} "
            f"image={record['image_path']}"
        )
    return 0


def render_candidate(
    candidate: dict[str, Any],
    args: argparse.Namespace,
    *,
    rank: int | None = None,
) -> tuple[Path, Path, Any]:
    lhs = expr_from_json(candidate["lhs"]["ast"])
    rhs = expr_from_json(candidate["rhs"]["ast"])
    sample = build_sample(args)
    constants = {"eps": args.epsilon, "C": args.constant_c, "C_eps": args.constant_c_eps}
    fields = compute_contour_fields(lhs, rhs, sample, constants=constants)
    out_dir = Path(args.out_dir)
    stem = output_stem(candidate, args, rank=rank)
    image_path = out_dir / f"{stem}.png"
    metadata_path = out_dir / f"{stem}.json"
    subtitle = (
        f"profile={args.profile}, amplitude={args.amplitude:g}, frequency={args.frequency}, seed={args.seed}, "
        f"grid={args.grid_size}^3, slice={args.slice_axis}:{slice_label(args.slice_index, args.grid_size)}"
    )
    render_contour_figure(
        fields,
        output_path=image_path,
        title=title_for_candidate(candidate),
        subtitle=subtitle,
        axis=args.slice_axis,
        index=args.slice_index,
        dpi=args.dpi,
    )
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(
            {
                "candidate_id": candidate.get("candidate_id"),
                "target_name": candidate.get("target_name"),
                "queue_status": candidate.get("queue_status"),
                "verification_progress": verification_progress(candidate),
                "lhs": candidate.get("lhs_text"),
                "rhs": candidate.get("rhs_text"),
                "sample": sample.metadata(),
                "slice_axis": args.slice_axis,
                "slice_index": args.slice_index,
                "constants": constants,
                "aggregates": fields.aggregates,
                "image_path": str(image_path),
            },
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return image_path, metadata_path, fields


def load_candidate(args: argparse.Namespace) -> dict[str, Any]:
    if args.summary_path:
        summary_path = Path(args.summary_path)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        best = summary.get("best_state")
        if not isinstance(best, dict):
            raise ValueError(f"summary has no best_state: {summary_path}")
        return {
            "candidate_id": summary_path.stem.removeprefix("summary_"),
            "target_name": summary.get("target_name", "unknown"),
            "lhs": best.get("lhs") or summary.get("target"),
            "rhs": best.get("rhs"),
            "lhs_text": text_of(best.get("lhs") or summary.get("target")),
            "rhs_text": text_of(best.get("rhs")),
        }

    candidates = load_queue_candidates(Path(args.queue_path))
    if args.candidate_id:
        for candidate in candidates:
            if str(candidate.get("candidate_id")) == args.candidate_id:
                return candidate
        raise ValueError(f"candidate id not found in queue: {args.candidate_id}")
    if args.candidate_index < 0 or args.candidate_index >= len(candidates):
        raise ValueError(f"candidate index {args.candidate_index} outside queue with {len(candidates)} candidates")
    return candidates[args.candidate_index]


def load_queue_candidates(queue_path: Path) -> list[dict[str, Any]]:
    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    candidates = queue.get("candidates", [])
    if not isinstance(candidates, list) or not candidates:
        raise ValueError(f"queue has no candidates: {queue_path}")
    return [candidate for candidate in candidates if isinstance(candidate, dict)]


def build_sample(args: argparse.Namespace) -> FieldSample:
    fields = {
        "omega": make_field(args.profile, args.amplitude, args.frequency, args.seed, args.grid_size, phase=0.0),
        "u": make_field(
            args.profile,
            0.7 * args.amplitude,
            max(1, args.frequency // 2),
            args.seed + 17,
            args.grid_size,
            phase=0.37,
        ),
    }
    return FieldSample(args.profile, args.amplitude, args.frequency, args.seed, fields)


def output_stem(candidate: dict[str, Any], args: argparse.Namespace, *, rank: int | None = None) -> str:
    candidate_id = str(candidate.get("candidate_id") or "candidate")[:16]
    target = safe_name(str(candidate.get("target_name", "unknown")))
    prefix = f"frontier_{rank:02d}_" if rank is not None else ""
    return (
        f"{prefix}{target}_{candidate_id}_{args.profile}_"
        f"A{args.amplitude:g}_k{args.frequency}_seed{args.seed}_{args.slice_axis}"
    )


def safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value)


def text_of(record: Any) -> str:
    if isinstance(record, dict):
        return str(record.get("text", ""))
    return ""


def slice_label(index: int | None, grid_size: int) -> int:
    return grid_size // 2 if index is None else index


def write_frontier_index(out_dir: Path, records: list[dict[str, Any]], args: argparse.Namespace) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    index_path = out_dir / "frontier_contours.html"
    cards = []
    for record in records:
        candidate = record["candidate"]
        image_path = Path(record["image_path"])
        aggregates = record["aggregates"]
        cards.append(
            f"""
            <article class="card">
              <header>
                <div class="rank">#{record['rank']}</div>
                <div>
                  <h2>{html_escape(str(candidate.get('target_name', 'unknown')))}</h2>
                  <p class="status">progress {verification_progress(candidate)} · {html_escape(str(candidate.get('queue_status', 'unknown')))}</p>
                </div>
              </header>
              <p class="ineq"><code>{html_escape(str(candidate.get('lhs_text', '')))} &lt;= {html_escape(str(candidate.get('rhs_text', '')))}</code></p>
              <img src="{html_escape(image_path.name)}" alt="frontier contour {record['rank']}">
              <dl>
                <div><dt>LHS mean</dt><dd>{aggregates['lhs_mean']:.4g}</dd></div>
                <div><dt>RHS mean</dt><dd>{aggregates['rhs_mean']:.4g}</dd></div>
                <div><dt>max residual</dt><dd>{aggregates['residual_max']:.4g}</dd></div>
                <div><dt>max log ratio</dt><dd>{aggregates['log_ratio_max']:.4g}</dd></div>
              </dl>
            </article>
            """
        )
    index_path.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AISIS Frontier Contours</title>
  <style>
    body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #172033; background: #f5f7fb; }}
    main {{ max-width: 1240px; margin: 0 auto; padding: 34px 22px 48px; }}
    h1 {{ margin: 0 0 6px; font-size: 34px; letter-spacing: 0; }}
    .sub {{ margin: 0 0 24px; color: #667085; font-size: 15px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(520px, 1fr)); gap: 20px; }}
    .card {{ background: white; border: 1px solid #d9e1ec; border-radius: 8px; padding: 16px; box-shadow: 0 10px 30px rgba(20, 33, 61, 0.08); }}
    header {{ display: flex; align-items: center; gap: 12px; margin-bottom: 8px; }}
    .rank {{ width: 44px; height: 44px; border-radius: 50%; background: #172033; color: white; display: grid; place-items: center; font-weight: 800; }}
    h2 {{ margin: 0; font-size: 18px; }}
    .status {{ margin: 2px 0 0; color: #667085; font-size: 13px; }}
    .ineq {{ margin: 10px 0 14px; color: #344054; overflow-wrap: anywhere; }}
    code {{ font-family: "SFMono-Regular", Consolas, monospace; font-size: 13px; }}
    img {{ width: 100%; display: block; border: 1px solid #e4eaf2; border-radius: 6px; background: #fff; }}
    dl {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; margin: 14px 0 0; }}
    dl div {{ background: #f4f7fb; border-radius: 6px; padding: 8px; }}
    dt {{ color: #667085; font-size: 11px; text-transform: uppercase; }}
    dd {{ margin: 2px 0 0; font-weight: 800; }}
    @media (max-width: 640px) {{ .grid {{ grid-template-columns: 1fr; }} dl {{ grid-template-columns: repeat(2, 1fr); }} }}
  </style>
</head>
<body>
  <main>
    <h1>Frontier Contour Diagnostics</h1>
    <p class="sub">Only candidates that advanced furthest through the verification pipeline are shown. Sample: profile={html_escape(args.profile)}, grid={args.grid_size}^3, frequency={args.frequency}, seed={args.seed}.</p>
    <section class="grid">
      {''.join(cards)}
    </section>
  </main>
</body>
</html>
""",
        encoding="utf-8",
    )
    return index_path


def html_escape(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


if __name__ == "__main__":
    raise SystemExit(main())
