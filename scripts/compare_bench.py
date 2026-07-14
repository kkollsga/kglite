#!/usr/bin/env python3
"""Perf-regression gate for `pytest-benchmark` JSON output.

Compares two pytest-benchmark JSON files (a baseline and a current run)
and exits non-zero when any common benchmark regresses by more than
`--threshold` percent on the chosen metric.

Usage:
    python scripts/compare_bench.py BASELINE CURRENT \\
        [--metric min|mean|median] \\
        [--threshold PERCENT] \\
        [--require-exact-set] \\
        [--quiet]

By default, `min` is the gating metric (per CLAUDE.md performance protocol:
"Trust `min` over `median` for sub-millisecond benches"). Threshold defaults
to 20% — anything tighter than that flakes too readily against the macOS /
GitHub-runner variance the tracked benchmarks see in practice.

The summary table is always printed so a passing gate still gives an
"at-a-glance, am I trending in the right direction" view. `--quiet` drops
it.

A benchmark newly present in the current file is informational until the next
baseline refresh. A benchmark present in the baseline but missing from the
current run fails the gate: benchmark coverage must not disappear silently.
Use `--require-exact-set` for CI baselines, where newly collected benchmarks
must also have a committed baseline row before the gate can pass.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def _load(path: Path) -> dict[str, float]:
    """Load a pytest-benchmark JSON and return `{name: stats}` per the
    `--metric` chosen later. We return the full stats dict for each
    benchmark so the caller picks the metric without re-reading."""
    data = json.loads(path.read_text())
    return {b["name"]: b["stats"] for b in data["benchmarks"]}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("baseline", type=Path, help="Baseline JSON (e.g. tests/benchmarks/baselines/0_9_52.json).")
    p.add_argument("current", type=Path, help="Current run JSON.")
    p.add_argument("--metric", default="min", choices=["min", "mean", "median"], help="Gating metric (default: min).")
    p.add_argument("--threshold", type=float, default=20.0, help="Regression threshold in percent (default: 20.0).")
    p.add_argument(
        "--require-exact-set",
        action="store_true",
        help="Fail when the current run contains benchmarks absent from the baseline.",
    )
    p.add_argument("--quiet", action="store_true", help="Suppress the summary table on pass.")
    args = p.parse_args()

    if not args.baseline.exists():
        print(f"baseline missing: {args.baseline}", file=sys.stderr)
        return 2
    if not args.current.exists():
        print(f"current missing: {args.current}", file=sys.stderr)
        return 2

    baseline = _load(args.baseline)
    current = _load(args.current)

    common = sorted(set(baseline) & set(current))
    only_baseline = sorted(set(baseline) - set(current))
    only_current = sorted(set(current) - set(baseline))

    if only_baseline:
        print(f"error: {len(only_baseline)} benchmark(s) in baseline but not in current:")
        for name in only_baseline:
            print(f"  - {name}")
    if only_current:
        level = "error" if args.require_exact_set else "info"
        print(f"{level}: {len(only_current)} new benchmark(s) in current run (no baseline row):")
        for name in only_current:
            print(f"  + {name}")

    # Compute deltas. A positive delta means "current is slower" (regression).
    rows = []
    for name in common:
        b = baseline[name][args.metric]
        c = current[name][args.metric]
        delta_pct = (c / b - 1) * 100 if b > 0 else 0.0
        rows.append((name, b, c, delta_pct))

    # Sort worst regressions first.
    rows.sort(key=lambda r: -r[3])

    regressions = [r for r in rows if r[3] > args.threshold]

    if not args.quiet or regressions:
        print(f"\nperf comparison ({args.metric}, threshold {args.threshold:+.1f}%)")
        print(f"baseline: {args.baseline}")
        print(f"current:  {args.current}\n")
        name_w = max((len(r[0]) for r in rows), default=20)
        print(f"  {'benchmark':<{name_w}}  {'baseline':>14}  {'current':>14}  {'delta':>8}")
        print(f"  {'-' * name_w}  {'-' * 14}  {'-' * 14}  {'-' * 8}")
        for name, b, c, delta in rows:
            flag = " ←" if delta > args.threshold else ""
            print(f"  {name:<{name_w}}  {b:>14.3e}  {c:>14.3e}  {delta:>+7.1f}%{flag}")

    unbaselined = only_current if args.require_exact_set else []
    if regressions or only_baseline or unbaselined:
        if only_baseline:
            print(
                f"\nFAIL: {len(only_baseline)} tracked benchmark(s) were not collected. "
                "Restore them or intentionally refresh the baseline."
            )
            for name in only_baseline:
                print(f"  - {name}")
        if unbaselined:
            print(
                f"\nFAIL: {len(unbaselined)} collected benchmark(s) have no baseline row. "
                "Capture the complete benchmark set before enabling this gate."
            )
            for name in unbaselined:
                print(f"  + {name}")
    if regressions:
        print(f"\nFAIL: {len(regressions)} benchmark(s) regressed > {args.threshold:+.1f}% on {args.metric}:")
        for name, _, _, delta in regressions:
            print(f"  - {name}: {delta:+.1f}%")
        print(
            "\nIf the regression is intentional (e.g. behaviour change worth the cost), "
            "refresh the baseline via `make refresh-release-constants` and explain in "
            "the CHANGELOG entry. Otherwise investigate before merging."
        )
    if regressions or only_baseline or unbaselined:
        return 1

    print(f"\nOK: no regressions > {args.threshold:+.1f}% on {args.metric} across {len(common)} benchmark(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
