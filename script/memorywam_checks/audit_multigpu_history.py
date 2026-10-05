"""Summarize locally recorded world sizes and collective failures.

This is intentionally metadata-only: it reads text/JSON logs and never loads
checkpoint tensors or initializes CUDA.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import Counter
from pathlib import Path


WORLD_RE = re.compile(r"World size:\s*(\d+)")


def _metrics_summary(run: Path) -> dict:
    path = run / "training_metrics.jsonl"
    if not path.is_file():
        return {"metrics_rows": 0}
    rows = []
    for line in path.read_text(errors="replace").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    result = {"metrics_rows": len(rows)}
    direct = [float(row["optimizer_step_seconds"]) for row in rows
              if "optimizer_step_seconds" in row]
    if direct:
        result["median_optimizer_step_seconds"] = statistics.median(direct)
        result["timing_source"] = "explicit_max_across_ranks"
        return result
    elapsed = [float(row["elapsed_seconds"]) for row in rows
               if "elapsed_seconds" in row]
    if len(elapsed) >= 2:
        deltas = [b - a for a, b in zip(elapsed, elapsed[1:]) if b >= a]
        if deltas:
            result["median_elapsed_delta_seconds"] = statistics.median(deltas)
            result["timing_source"] = (
                "legacy_elapsed_delta_may_include_validation_or_save")
    return result


def audit(root: Path) -> dict:
    runs = []
    for log in sorted(root.rglob("training.log")):
        text = log.read_text(errors="replace")
        sizes = sorted({int(value) for value in WORLD_RE.findall(text)})
        timeouts = text.count("_ALLGATHER_BASE") + text.count(
            "collective operation timeout")
        if not sizes and not timeouts:
            continue
        run = log.parent
        entry = {
            "run": str(run.relative_to(root)),
            "world_sizes": sizes,
            "allgather_or_collective_timeout_mentions": timeouts,
        }
        entry.update(_metrics_summary(run))
        runs.append(entry)
    counts = Counter(size for run in runs for size in run["world_sizes"])
    return {
        "schema_version": 1,
        "kind": "local_multigpu_history_audit",
        "train_out": str(root.resolve()),
        "runs_with_world_size_or_timeout_evidence": len(runs),
        "world_size_run_counts": {str(key): counts[key]
                                  for key in sorted(counts)},
        "has_four_rank_timing_evidence": counts[4] > 0,
        "runs_with_collective_timeout_mentions": sum(
            bool(run["allgather_or_collective_timeout_mentions"])
            for run in runs),
        "evidence_limit": (
            "Absence from local logs does not prove a topology was never run; "
            "it means no retained training.log matched the recorded pattern."),
        "runs": runs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-out", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.train_out)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: report[key] for key in (
        "runs_with_world_size_or_timeout_evidence", "world_size_run_counts",
        "has_four_rank_timing_evidence",
        "runs_with_collective_timeout_mentions")}, indent=2))


if __name__ == "__main__":
    main()
