#!/usr/bin/env python3
"""Collect GenAI-Perf JSON exports from results/raw/ into one summary CSV.

Directory naming convention (set by run_benchmark.sh):
    results/raw/<config>_<workload>_c<concurrency>/.../*genai_perf.json

Usage:
    python scripts/parse_results.py results/raw -o results/summary.csv
"""
import argparse
import json
import re
from pathlib import Path

import pandas as pd

DIR_RE = re.compile(r"(?P<config>[^_]+)_(?P<workload>.+)_c(?P<concurrency>\d+)$")

# metric name in genai-perf export -> column prefix
METRICS = {
    "time_to_first_token": "ttft_ms",
    "inter_token_latency": "tpot_ms",
    "request_latency": "e2e_ms",
    "output_token_throughput": "out_tok_per_s",
    "request_throughput": "req_per_s",
}
STATS = ["avg", "p50", "p95", "p99"]


def parse_run(run_dir: Path) -> dict | None:
    m = DIR_RE.match(run_dir.name)
    if not m:
        return None
    json_files = list(run_dir.rglob("*genai_perf.json"))
    if not json_files:
        print(f"  skip (no genai_perf.json yet): {run_dir.name}")
        return None
    data = json.loads(json_files[0].read_text())
    row: dict = {
        "config": m["config"],
        "workload": m["workload"],
        "concurrency": int(m["concurrency"]),
    }
    for metric, prefix in METRICS.items():
        entry = data.get(metric, {})
        for stat in STATS:
            if stat in entry:
                row[f"{prefix}_{stat}"] = entry[stat]
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("raw_dir", type=Path)
    ap.add_argument("-o", "--output", type=Path, default=Path("results/summary.csv"))
    args = ap.parse_args()

    rows = [r for d in sorted(args.raw_dir.iterdir()) if d.is_dir()
            for r in [parse_run(d)] if r]
    if not rows:
        raise SystemExit(f"No parseable runs under {args.raw_dir}")

    df = pd.DataFrame(rows).sort_values(["workload", "config", "concurrency"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"{len(df)} runs -> {args.output}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
