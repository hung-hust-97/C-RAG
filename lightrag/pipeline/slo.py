"""Baseline measurements for pipeline latency/throughput (read-only analytics).

Pure functions over stage-run rows so the numbers can be unit-tested; the CLI in
``.kiro/scripts/measure_pipeline_slo.py`` only fetches rows and prints the report.
This module reports observed baselines; it deliberately proposes no SLO targets.
"""

from __future__ import annotations

import math
from collections import defaultdict
from datetime import datetime
from typing import Any, Iterable

DEFAULT_MIN_SAMPLES = 30
FINAL_STAGE = "finalize"
TERMINAL_FAILURE_STATUSES = {"failed"}


def percentile(sorted_values: list[float], pct: float) -> float | None:
    """Linear-interpolated percentile of an already sorted list (pct in 0..100)."""
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (len(sorted_values) - 1) * pct / 100
    lower = math.floor(rank)
    upper = math.ceil(rank)
    weight = rank - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def summarize(values: Iterable[float], min_samples: int = DEFAULT_MIN_SAMPLES) -> dict:
    data = sorted(v for v in values if v is not None and v >= 0)
    return {
        "count": len(data),
        "p50": percentile(data, 50),
        "p95": percentile(data, 95),
        "p99": percentile(data, 99),
        "max": data[-1] if data else None,
        # p95/p99 from few samples are not meaningful; flag instead of hiding.
        "low_confidence": len(data) < min_samples,
    }


def _seconds(start: datetime | None, end: datetime | None) -> float | None:
    if start is None or end is None:
        return None
    return (end - start).total_seconds()


def build_report(
    rows: list[dict[str, Any]],
    *,
    window_seconds: float | None = None,
    min_samples: int = DEFAULT_MIN_SAMPLES,
) -> dict[str, Any]:
    """Aggregate stage-run rows into per-doc_type/stage baselines.

    Row keys: run_id, stage, status, attempt, doc_type, started_at, completed_at,
    queued_at (any may be None). Queue wait = started_at - queued_at, stage
    duration = completed_at - started_at (completed stages only), end-to-end =
    final-stage completion minus the earliest queue/start time of the run.
    """
    waits: dict[tuple, list[float]] = defaultdict(list)
    durations: dict[tuple, list[float]] = defaultdict(list)
    counts: dict[tuple, dict[str, int]] = defaultdict(
        lambda: {"total": 0, "completed": 0, "failed": 0, "retried": 0}
    )
    runs: dict[str, dict[str, Any]] = {}

    for row in rows:
        doc_type = row.get("doc_type") or "general"
        key = (doc_type, row["stage"])
        status = row.get("status")
        bucket = counts[key]
        bucket["total"] += 1
        if status == "completed":
            bucket["completed"] += 1
        if status in TERMINAL_FAILURE_STATUSES:
            bucket["failed"] += 1
        if (row.get("attempt") or 1) > 1:
            bucket["retried"] += 1

        wait = _seconds(row.get("queued_at"), row.get("started_at"))
        if wait is not None:
            waits[key].append(wait)
        if status == "completed":
            duration = _seconds(row.get("started_at"), row.get("completed_at"))
            if duration is not None:
                durations[key].append(duration)

        run = runs.setdefault(
            row["run_id"],
            {"doc_type": doc_type, "begin": None, "end": None, "failed": False},
        )
        begin = row.get("queued_at") or row.get("started_at")
        if begin is not None and (run["begin"] is None or begin < run["begin"]):
            run["begin"] = begin
        if status in TERMINAL_FAILURE_STATUSES:
            run["failed"] = True
        if (
            row["stage"] == FINAL_STAGE
            and status == "completed"
            and row.get("completed_at") is not None
        ):
            run["end"] = row["completed_at"]

    end_to_end: dict[str, list[float]] = defaultdict(list)
    throughput: dict[str, int] = defaultdict(int)
    run_totals: dict[str, dict[str, int]] = defaultdict(
        lambda: {"runs": 0, "finished": 0, "failed": 0}
    )
    for run in runs.values():
        totals = run_totals[run["doc_type"]]
        totals["runs"] += 1
        if run["failed"]:
            totals["failed"] += 1
        if run["end"] is not None and not run["failed"]:
            totals["finished"] += 1
            throughput[run["doc_type"]] += 1
            seconds = _seconds(run["begin"], run["end"])
            if seconds is not None:
                end_to_end[run["doc_type"]].append(seconds)

    stages: dict[str, dict[str, Any]] = defaultdict(dict)
    for key in sorted(counts):
        doc_type, stage = key
        stages[doc_type][stage] = {
            "queue_wait_seconds": summarize(waits[key], min_samples),
            "duration_seconds": summarize(durations[key], min_samples),
            **counts[key],
        }

    per_doc_type: dict[str, Any] = {}
    for doc_type in sorted(set(stages) | set(run_totals)):
        entry = {
            "stages": stages.get(doc_type, {}),
            "end_to_end_seconds": summarize(end_to_end[doc_type], min_samples),
            **run_totals[doc_type],
        }
        if window_seconds:
            entry["docs_per_hour"] = throughput[doc_type] / (window_seconds / 3600)
        per_doc_type[doc_type] = entry

    return {
        "window_seconds": window_seconds,
        "min_samples": min_samples,
        "doc_types": per_doc_type,
    }

