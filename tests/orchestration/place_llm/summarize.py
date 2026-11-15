#!/usr/bin/env python3
"""Markdown table of results/trials.csv per noise and variant (ROS free).

python3 tests/orchestration/place_llm/summarize.py --run "2026-10-11" "2026-10-11 rerun"
"""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent


def rate(rows, key) -> str:
    values = [r[key] == "True" for r in rows if r.get(key) in ("True", "False")]
    return f"{sum(values)}/{len(values)}" if values else "–"


def values(rows, key) -> list[float]:
    return [float(r[key]) for r in rows if r.get(key) not in ("", None)]


def mean(rows, key, digits=1) -> str:
    v = values(rows, key)
    return f"{statistics.mean(v):.{digits}f}" if v else "–"


def median_max(rows, key) -> str:
    v = values(rows, key)
    return f"{statistics.median(v):.1f} / {max(v):.1f}" if v else "–"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--csv", type=Path, default=HERE / "results" / "trials.csv")
    parser.add_argument("--run", nargs="+", help="only rows of these runs")
    parser.add_argument("--cases", nargs="+", help="only these cases")
    args = parser.parse_args()
    with args.csv.open(newline="") as stream:
        selected = [
            r
            for r in csv.DictReader(stream)
            if (not args.run or r["run"] in args.run)
            and (not args.cases or r["case"] in args.cases)
        ]
    # A trial repeated in a later run (e.g. after a server outage) replaces the earlier.
    latest = {(r["noise"], r["case"], r["variant"]): r for r in selected}
    rows = list(latest.values())
    print(
        "| noise | variant | trials | level | decision | place OK | placed correctly "
        "| observe s | LLM s (median / max) | place s (OK) | prompt tok | completion tok |"
    )
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for noise in sorted({r["noise"] for r in rows}):
        for variant in sorted({r["variant"] for r in rows}):
            group = [r for r in rows if r["noise"] == noise and r["variant"] == variant]
            if not group:
                continue
            placed = [r for r in group if r.get("place_success") == "True"]
            print(
                f"| {noise} | {variant} | {len(group)} | {rate(group, 'level_ok')} "
                f"| {rate(group, 'decision_ok')} | {rate(group, 'place_success')} "
                f"| {rate(group, 'placed_ok')} | {mean(group, 'observe_s')} "
                f"| {median_max(group, 'llm_s')} | {mean(placed, 'place_s')} "
                f"| {mean(group, 'prompt_tokens', 0)} | {mean(group, 'completion_tokens', 0)} |"
            )


if __name__ == "__main__":
    main()
