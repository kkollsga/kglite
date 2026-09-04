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
        [--quiet] \\
        [--record-history PATH]

By default, `min` is the gating metric (per CLAUDE.md performance protocol:
"Trust `min` over `median` for sub-millisecond benches"). Threshold defaults
to 20% — anything tighter than that flakes too readily against the macOS /
GitHub-runner variance the tracked benchmarks see in practice.

The summary table is always printed so a passing gate still gives an
"at-a-glance, am I trending in the right direction" view. `--quiet` drops
it.

Cells that pass but sit just under the threshold are reported separately as
an APPROACHING watch band (see `WATCH_BAND_WIDTH_PCT`). The band never
changes the exit code; it exists so a cell that creeps up release after
release is visible the release *before* it crosses.

`--record-history PATH` appends one summary row per run to a CSV recurrence
record (verdict, worst cell, watch band). It is a local developer aid wired
into `make bench-check`; CI does not pass it.

A benchmark newly present in the current file is informational until the next
baseline refresh. A benchmark present in the baseline but missing from the
current run fails the gate: benchmark coverage must not disappear silently.
Use `--require-exact-set` for CI baselines, where newly collected benchmarks
must also have a committed baseline row before the gate can pass.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
from pathlib import Path
import subprocess
import sys

try:
    from scripts.benchmark_qualification import QualificationError, Registry, compatible, validate_measurements
except ModuleNotFoundError:  # Direct script execution.
    from benchmark_qualification import QualificationError, Registry, compatible, validate_measurements

# Width of the "APPROACHING" watch band, in percentage points below the gate
# threshold: a +20% gate reports every passing cell in +12%..+20%.
#
# Why a band at all: the 0.16.6 leg-1 "flake" sat at +17-19% for several
# releases and only became visible when it finally crossed 20%. Nothing in a
# green run said the cell had been sitting one bad round from red — the gate's
# only two states were silence and failure. The band adds the missing third
# state without touching the exit code.
#
# Why 8pp: wide enough that a cell has to have moved most of the way to the
# gate before it is named (a fresh cell at +5% stays silent), narrow enough
# that a healthy run prints nothing at all — silence has to keep meaning
# something, or the block becomes noise everyone scrolls past.
WATCH_BAND_WIDTH_PCT = 8.0

# Column header for the recurrence record written by --record-history.
GATE_HISTORY_HEADER = (
    "date",
    "sha",
    "verdict",
    "metric",
    "threshold_pct",
    "worst_cell",
    "worst_delta_pct",
    "approaching",
)


def _load(path: Path) -> dict[str, float]:
    """Load a pytest-benchmark JSON and return `{name: stats}` per the
    `--metric` chosen later. We return the full stats dict for each
    benchmark so the caller picks the metric without re-reading."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return {b["name"]: b["stats"] for b in data["benchmarks"]}


def _watch_band(rows: list[tuple[str, float, float, float]], threshold: float) -> list[tuple[str, float, float, float]]:
    """Passing cells within `WATCH_BAND_WIDTH_PCT` of the gate threshold.

    Regressions (delta > threshold) are excluded — they are the FAIL list, not
    the watch band. `rows` is already sorted worst-first, so the result is too.
    """
    floor = threshold - WATCH_BAND_WIDTH_PCT
    return [r for r in rows if floor <= r[3] <= threshold]


def _print_watch_band(band: list[tuple[str, float, float, float]], threshold: float, name_w: int) -> None:
    """Print the APPROACHING block. Callers skip an empty band: a run with
    nothing near the gate must stay byte-identical to the pre-band output."""
    floor = threshold - WATCH_BAND_WIDTH_PCT
    print(f"\nAPPROACHING ({floor:+.1f}%..{threshold:+.1f}%)")
    print(f"  {'benchmark':<{name_w}}  {'baseline':>14}  {'current':>14}  {'delta':>8}")
    print(f"  {'-' * name_w}  {'-' * 14}  {'-' * 14}  {'-' * 8}")
    for name, b, c, delta in band:
        print(f"  {name:<{name_w}}  {b:>14.3e}  {c:>14.3e}  {delta:>+7.1f}%")
    print(
        f"\n{len(band)} cell(s) approaching the {threshold:+.1f}% threshold — "
        "a crossing next release is likely drift, not noise."
    )


def _git_short_sha() -> str:
    """Short HEAD sha, or "unknown" outside a work tree / without git."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return "unknown"
    return out.stdout.strip() or "unknown"


def _append_gate_history(
    path: Path,
    *,
    verdict: str,
    metric: str,
    threshold: float,
    rows: list[tuple[str, float, float, float]],
    band: list[tuple[str, float, float, float]],
) -> None:
    """Append one summary row to the recurrence record at `path`.

    Best-effort by design: the record is a local working file and a gate must
    never fail (or pass) because writing it did or did not work. A missing
    parent directory means the local working folder is not present (fresh
    clone, CI checkout) — say so once and move on.
    """
    if not path.parent.is_dir():
        print(f"\ninfo: recurrence record skipped, {path.parent} does not exist.")
        return
    worst = rows[0] if rows else None
    row = [
        datetime.date.today().isoformat(),
        _git_short_sha(),
        verdict,
        metric,
        f"{threshold:.1f}",
        worst[0] if worst else "",
        f"{worst[3]:+.1f}" if worst else "",
        ";".join(f"{name}:{delta:+.1f}%" for name, _, _, delta in band),
    ]
    try:
        exists = path.exists()
        with path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            if not exists:
                writer.writerow(GATE_HISTORY_HEADER)
            writer.writerow(row)
    except OSError as exc:  # pragma: no cover - local filesystem failure
        print(f"\nwarning: could not append to the recurrence record {path}: {exc}")
        return
    print(f"\nrecorded: {path} <- {','.join(row)}")


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
    p.add_argument(
        "--record-history",
        type=Path,
        default=None,
        metavar="PATH",
        help="Append one summary row per run (verdict, worst cell, watch band) to this CSV.",
    )
    args = p.parse_args()

    if not args.baseline.exists():
        print(f"baseline missing: {args.baseline}", file=sys.stderr)
        return 2
    if not args.current.exists():
        print(f"current missing: {args.current}", file=sys.stderr)
        return 2

    try:
        registry = Registry(args.baseline.parent)
        registry.candidate(args.current)
        Registry(args.current.parent).candidate(args.current)
        args.baseline = registry.reference(args.baseline)
        compatible(args.baseline, args.current)
        baseline = _load(args.baseline)
        current = _load(args.current)
        validate_measurements(baseline, args.metric)
        validate_measurements(current, args.metric)
    except (QualificationError, OSError, ValueError) as error:
        print(f"no valid perf verdict: {error}", file=sys.stderr)
        return 2

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
    band = _watch_band(rows, args.threshold)
    name_w = max((len(r[0]) for r in rows), default=20)

    if not args.quiet or regressions:
        print(f"\nperf comparison ({args.metric}, threshold {args.threshold:+.1f}%)")
        print(f"baseline: {args.baseline}")
        print(f"current:  {args.current}\n")
        print(f"  {'benchmark':<{name_w}}  {'baseline':>14}  {'current':>14}  {'delta':>8}")
        print(f"  {'-' * name_w}  {'-' * 14}  {'-' * 14}  {'-' * 8}")
        for name, b, c, delta in rows:
            flag = " <<" if delta > args.threshold else ""
            print(f"  {name:<{name_w}}  {b:>14.3e}  {c:>14.3e}  {delta:>+7.1f}%{flag}")

    # The watch band prints in every run, quiet or not: it is the whole point
    # of the band that a *passing* run carries the warning.
    if band:
        _print_watch_band(band, args.threshold, name_w)

    unbaselined = only_current if args.require_exact_set else []
    failed = bool(regressions or only_baseline or unbaselined)
    if failed:
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
            "capture via `make refresh-release-constants`, qualify the capture with "
            "scripts/benchmark_qualification.py after repeat/control checks, and explain "
            "the decision in CHANGELOG. Otherwise investigate before merging."
        )
    if args.record_history is not None:
        _append_gate_history(
            args.record_history,
            verdict="FAIL" if failed else "OK",
            metric=args.metric,
            threshold=args.threshold,
            rows=rows,
            band=band,
        )

    if failed:
        return 1

    print(f"\nOK: no regressions > {args.threshold:+.1f}% on {args.metric} across {len(common)} benchmark(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
