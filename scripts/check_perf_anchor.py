#!/usr/bin/env python3
"""Cumulative perf-drift gate: compare the newest release baseline against an
anchor several releases back.

The per-release gates (CI's Linux comparison, local `bench-check`) compare
against the *immediately previous* state with a 20% threshold, and the
baseline is recaptured from current code every release. That ratchet means a
persistent ~10%-per-release regression never trips any single gate. This
check closes that hole: it compares the newest per-release baseline in
`tests/benchmarks/baselines/` against the baseline from `--releases-back`
releases earlier (default 3) at a wider threshold (default 30%), over the
*intersection* of benchmark names — benchmark sets legitimately evolve across
releases, so a bench present only on one side is reported, not failed
(`compare_bench.py` is the wrong tool here for exactly that reason).

Platform note: per-release baselines are captured on the release machine
(macOS bare names; `.linux` files exist only where CI captured them), so this
is a same-machine longitudinal comparison. A hardware change resets the
comparison window — pass `--releases-back 1` for the first release on new
hardware.

Usage:
    python scripts/check_perf_anchor.py [--releases-back N] [--threshold PCT]
        [--metric min|mean|median] [--current PATH] [--min-overlap N]

Exits non-zero when any common benchmark regressed more than the threshold
against the anchor. Run at release time (`make bench-anchor`, wired into the
release skill) after `make refresh-release-constants` has captured the new
baseline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

REPO_ROOT = Path(__file__).resolve().parent.parent
BASELINES_DIR = REPO_ROOT / "tests" / "benchmarks" / "baselines"
VERSION_RE = re.compile(r"^(\d+)_(\d+)_(\d+)\.json$")


def release_baselines() -> list[tuple[tuple[int, int, int], Path]]:
    """All per-release (bare/macOS) baselines, oldest first."""
    found = []
    for path in BASELINES_DIR.iterdir():
        match = VERSION_RE.match(path.name)
        if match:
            found.append((tuple(int(g) for g in match.groups()), path))
    return sorted(found)


def load_metric(path: Path, metric: str) -> dict[str, float]:
    data = json.loads(path.read_text())
    return {b["name"]: b["stats"][metric] for b in data["benchmarks"]}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--releases-back",
        type=int,
        default=3,
        help="How many releases before the current one to anchor on (default: 3; clamped to the oldest available).",
    )
    p.add_argument(
        "--threshold", type=float, default=30.0, help="Cumulative regression threshold in percent (default: 30)."
    )
    p.add_argument("--metric", default="min", choices=["min", "mean", "median"], help="Gating metric (default: min).")
    p.add_argument(
        "--current",
        type=Path,
        default=None,
        help="Current baseline JSON (default: the highest-version per-release file).",
    )
    p.add_argument(
        "--min-overlap",
        type=int,
        default=10,
        help="Minimum common benchmarks for a verdict (default: 10); below this, report and pass.",
    )
    args = p.parse_args()

    releases = release_baselines()
    if args.current is not None:
        current_path = args.current
        releases = [(v, path) for v, path in releases if path.resolve() != current_path.resolve()]
    else:
        if not releases:
            print("perf anchor: no per-release baselines found — nothing to compare")
            return 0
        _, current_path = releases[-1]
        releases = releases[:-1]

    if not releases:
        print(f"perf anchor: {current_path.name} is the only per-release baseline — nothing to compare")
        return 0

    anchor_index = max(0, len(releases) - args.releases_back)
    anchor_version, anchor_path = releases[anchor_index]

    current = load_metric(current_path, args.metric)
    anchor = load_metric(anchor_path, args.metric)
    common = sorted(set(current) & set(anchor))
    only_anchor = sorted(set(anchor) - set(current))
    only_current = sorted(set(current) - set(anchor))

    print(f"perf anchor: {current_path.name} vs {anchor_path.name} ({args.metric}, threshold +{args.threshold:.1f}%)")
    if only_anchor:
        preview = ", ".join(only_anchor[:5]) + ("…" if len(only_anchor) > 5 else "")
        print(f"  {len(only_anchor)} benchmark(s) only in the anchor (retired since): {preview}")
    if only_current:
        print(f"  {len(only_current)} benchmark(s) newer than the anchor (no longitudinal view yet)")

    if len(common) < args.min_overlap:
        print(f"perf anchor: only {len(common)} common benchmark(s) (< {args.min_overlap}) — too few for a verdict")
        return 0

    rows = sorted(((current[name] / anchor[name] - 1.0) * 100.0, name) for name in common)
    for pct, name in reversed(rows):
        marker = " <-- REGRESSED" if pct > args.threshold else ""
        print(f"  {name:110s} {anchor[name]:.3e}  {current[name]:.3e}  {pct:+6.1f}%{marker}")

    regressed = [(pct, name) for pct, name in rows if pct > args.threshold]
    if regressed:
        print(f"\nFAIL: {len(regressed)} benchmark(s) drifted > +{args.threshold:.1f}% since {anchor_path.name}:")
        for pct, name in regressed:
            print(f"  - {name}: {pct:+.1f}%")
        print(
            "Cumulative drift accumulated across releases without any single release "
            "tripping the 20% gate. Investigate before shipping; refreshing the "
            "baseline does NOT clear this check — only recovering the performance does."
        )
        return 1

    worst = max(rows)[0] if rows else 0.0
    print(f"OK: no cumulative drift > +{args.threshold:.1f}% across {len(common)} benchmark(s) (worst {worst:+.1f}%).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
