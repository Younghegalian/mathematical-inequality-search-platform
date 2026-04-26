"""SQLite summary index for large AISIS datasets."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "aisis.index.v1"
INDEXER_VERSION = "2"


@dataclass(frozen=True)
class IndexBuildResult:
    index_path: Path
    scanned: int
    inserted_or_updated: int
    deleted: int


def build_sqlite_index(data_dir: Path, index_path: Path) -> IndexBuildResult:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(index_path)
    try:
        previous_indexer_version = read_meta(connection, "indexer_version")
        ensure_schema(connection)
        result = upsert_summaries(
            connection,
            data_dir,
            index_path,
            force_refresh=previous_indexer_version != INDEXER_VERSION,
        )
        connection.execute(
            "insert or replace into meta(key, value) values('indexer_version', ?)",
            (INDEXER_VERSION,),
        )
        connection.commit()
        return result
    finally:
        connection.close()


def ensure_schema(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        create table if not exists meta (
            key text primary key,
            value text not null
        )
        """
    )
    connection.execute(
        """
        create table if not exists runs (
            run_id text primary key,
            target_name text not null,
            depth integer not null,
            width integer not null,
            best_closure text not null,
            best_score integer not null,
            transition_count integer not null,
            num_states integer not null,
            proof_path text not null,
            rhs_text text not null,
            summary_path text not null,
            run_path text not null,
            summary_mtime_ns integer not null,
            summary_json text not null
        )
        """
    )
    connection.execute("create index if not exists idx_runs_case on runs(target_name, depth, width)")
    connection.execute("create index if not exists idx_runs_closure_score on runs(best_closure, best_score)")
    connection.execute(
        "insert or replace into meta(key, value) values('schema_version', ?)",
        (SCHEMA_VERSION,),
    )


def read_meta(connection: sqlite3.Connection, key: str) -> str | None:
    try:
        row = connection.execute("select value from meta where key = ?", (key,)).fetchone()
    except sqlite3.OperationalError:
        return None
    return str(row[0]) if row else None


def upsert_summaries(
    connection: sqlite3.Connection,
    data_dir: Path,
    index_path: Path,
    *,
    force_refresh: bool = False,
) -> IndexBuildResult:
    summary_dir = data_dir / "results"
    paths = sorted(summary_dir.glob("summary_*.json"))
    live_summary_paths = {str(path) for path in paths}
    inserted_or_updated = 0

    known = {
        row[0]: row[1]
        for row in connection.execute("select summary_path, summary_mtime_ns from runs").fetchall()
    }

    for summary_path in paths:
        mtime_ns = summary_path.stat().st_mtime_ns
        if not force_refresh and known.get(str(summary_path)) == mtime_ns:
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        row = row_from_summary(summary, summary_path, data_dir, mtime_ns)
        connection.execute(
            """
            insert or replace into runs(
                run_id, target_name, depth, width, best_closure, best_score,
                transition_count, num_states, proof_path, rhs_text,
                summary_path, run_path, summary_mtime_ns, summary_json
            )
            values(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            row,
        )
        inserted_or_updated += 1

    deleted = 0
    for summary_path in set(known) - live_summary_paths:
        connection.execute("delete from runs where summary_path = ?", (summary_path,))
        deleted += 1

    return IndexBuildResult(
        index_path=index_path,
        scanned=len(paths),
        inserted_or_updated=inserted_or_updated,
        deleted=deleted,
    )


def row_from_summary(
    summary: dict[str, Any],
    summary_path: Path,
    data_dir: Path,
    mtime_ns: int,
) -> tuple[Any, ...]:
    run_id = str(summary.get("run_id", summary_path.stem.removeprefix("summary_")))
    run_path = str(summary.get("run_path") or (data_dir / "runs" / f"{run_id}.jsonl"))
    search = summary.get("search") if isinstance(summary.get("search"), dict) else {}
    if not search:
        search = read_search_from_run_start(Path(run_path)) or {}
    best = summary.get("best_state") or {}
    closure = best.get("closure") or {}
    proof = best.get("proof") or []
    proof_path = " -> ".join(str(step.get("rule_name", "")) for step in proof if step.get("rule_name"))
    rhs_text = text_of(best.get("rhs"))
    summary_for_index = dict(summary)
    summary_for_index["search"] = search
    summary_for_index["run_path"] = run_path
    summary_for_index["transition_count"] = int(summary.get("transition_count", 0))
    return (
        run_id,
        str(summary.get("target_name", "unknown")),
        int(search.get("depth", 0)),
        int(search.get("width", 0)),
        str(closure.get("status", "NONE")),
        int(best.get("score", 0)),
        int(summary.get("transition_count", 0)),
        int(summary.get("num_states", 0)),
        proof_path,
        rhs_text,
        str(summary_path),
        run_path,
        mtime_ns,
        json.dumps(summary_for_index, ensure_ascii=True, sort_keys=True),
    )


def read_search_from_run_start(run_path: Path) -> dict[str, Any] | None:
    if not run_path.exists():
        return None
    with run_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("event") == "run_start":
                search = row.get("search")
                return search if isinstance(search, dict) else None
    return None


def fetch_index_summaries(index_path: Path) -> list[dict[str, Any]]:
    connection = sqlite3.connect(index_path)
    try:
        rows = connection.execute(
            "select summary_json, run_path from runs order by run_id"
        ).fetchall()
    finally:
        connection.close()

    summaries = []
    for summary_json, run_path in rows:
        summary = json.loads(summary_json)
        if not summary.get("run_path"):
            summary["run_path"] = run_path
        summaries.append(summary)
    return summaries


def text_of(record: Any) -> str:
    if not isinstance(record, dict):
        return ""
    return str(record.get("text", ""))
