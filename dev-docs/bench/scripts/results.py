"""Query and summarise `bench/benchmark_full.csv` results.

The benchmark accumulates rows across many runs (one row per
(run, mode, dataset)). This script renders that CSV in a readable
shape — `runs`, `latest`, `trends`, `deltas`, `compare`.

Usage:
    # List every recorded run + how many cells each covered.
    python bench/results.py runs

    # Most recent measurement per (mode, dataset).
    python bench/results.py latest
    python bench/results.py latest --mode disk
    python bench/results.py latest --mode disk --dataset wiki1000m

    # Per-run time-series for the filtered cells.
    python bench/results.py trends --mode disk --dataset wiki1000m

    # Consecutive-run deltas — surfaces regressions immediately.
    python bench/results.py deltas --mode disk
    python bench/results.py deltas --dataset wiki100m

    # Side-by-side compare of two named runs (timestamp prefix or
    # the keywords `latest` / `previous`).
    python bench/results.py compare latest previous
    python bench/results.py compare 2026-04-25T03 2026-04-25T06

    # Custom columns.
    python bench/results.py latest --cols build_ms,save_ms,total_ms
    python bench/results.py latest --cols all

Default columns: build_ms, save_ms, load_ms, mutate_ms, resave_ms,
total_ms (the lifecycle pipeline). Pass `--cols all` to see every
column.

Errors column is always shown when populated, so you can see what
broke without rooting around in the sidecar log.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

try:
    import pandas as pd
except ImportError:
    print("error: pandas required (pip install pandas)", file=sys.stderr)
    sys.exit(2)


DEFAULT_CSV = Path(__file__).parent / "benchmark_full.csv"

LIFECYCLE_COLS = ["build_ms", "save_ms", "load_ms", "mutate_ms", "resave_ms", "total_ms"]


def _load(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        print(f"error: no benchmark CSV at {path}", file=sys.stderr)
        sys.exit(1)
    df = pd.read_csv(path)
    df["run_started_at"] = pd.to_datetime(df["run_started_at"], utc=True)
    return df


def _filter(df: pd.DataFrame, mode: str | None, dataset: str | None) -> pd.DataFrame:
    if mode:
        df = df[df["mode"] == mode]
    if dataset:
        df = df[df["dataset"] == dataset]
    return df


def _resolve_cols(spec: str, df: pd.DataFrame) -> list[str]:
    if spec == "all":
        # Skip identity columns; show everything else.
        ident = {"run_started_at", "version", "mode", "dataset"}
        return [c for c in df.columns if c not in ident]
    if spec:
        wanted = [c.strip() for c in spec.split(",") if c.strip()]
        for c in wanted:
            if c not in df.columns:
                print(f"warning: column '{c}' not in CSV", file=sys.stderr)
        return [c for c in wanted if c in df.columns]
    return [c for c in LIFECYCLE_COLS if c in df.columns]


def _fmt_ms(v) -> str:
    """Render a single ms value. Returns the input unchanged for
    non-numeric inputs so columns like `errors` can be passed through
    `--cols` without a TypeError."""
    if v is None:
        return "—"
    try:
        if pd.isna(v):
            return "—"
    except (TypeError, ValueError):
        return str(v)
    if not isinstance(v, (int, float)):
        return str(v)
    if v >= 1000:
        return f"{v / 1000:.2f}s"
    return f"{v:.0f}ms"


def cmd_latest(df: pd.DataFrame, cols: list[str]) -> None:
    """Most recent measurement per (mode, dataset)."""
    if df.empty:
        print("(no rows match)")
        return
    grouped = df.sort_values("run_started_at").groupby(["mode", "dataset"]).tail(1).reset_index(drop=True)
    out_cols: list[str] = []
    for c in ["mode", "dataset", "version", "run_started_at", *cols, "errors"]:
        if c in grouped.columns and c not in out_cols:
            out_cols.append(c)
    pretty = grouped[out_cols].copy()
    for c in cols:
        if c in pretty.columns and c != "errors":
            pretty[c] = pretty[c].apply(_fmt_ms)
    pretty["run_started_at"] = pretty["run_started_at"].dt.strftime("%Y-%m-%d %H:%M")
    # Drop empty error column entirely if no row has one.
    if "errors" in pretty.columns and pretty["errors"].fillna("").eq("").all():
        pretty = pretty.drop(columns=["errors"])
    print(pretty.to_string(index=False))


def cmd_trends(df: pd.DataFrame, cols: list[str]) -> None:
    """Time-series — every run for the (filtered) cells, oldest first."""
    if df.empty:
        print("(no rows match)")
        return
    keep = ["run_started_at", "version", "mode", "dataset", *cols]
    keep = [c for c in keep if c in df.columns]
    pretty = df.sort_values("run_started_at")[keep].copy()
    for c in cols:
        if c in pretty.columns:
            pretty[c] = pretty[c].apply(_fmt_ms)
    pretty["run_started_at"] = pretty["run_started_at"].dt.strftime("%Y-%m-%d %H:%M")
    print(pretty.to_string(index=False))


def cmd_runs(df: pd.DataFrame) -> None:
    """List every recorded run with version, cell coverage, totals."""
    if df.empty:
        print("(no rows)")
        return
    # Group by run_started_at so cells from one invocation collapse.
    grouped = (
        df.groupby(["run_started_at", "version"])
        .agg(
            cells=("mode", "size"),
            modes=("mode", lambda s: ",".join(sorted(set(s)))),
            datasets=("dataset", lambda s: ",".join(sorted(set(s), key=_ds_sort_key))),
            total_ms=("total_ms", "sum"),
        )
        .reset_index()
        .sort_values("run_started_at")
    )
    grouped["run_started_at"] = grouped["run_started_at"].dt.strftime("%Y-%m-%d %H:%M")
    grouped["total_ms"] = grouped["total_ms"].apply(_fmt_ms)
    print(grouped.to_string(index=False))


def _ds_sort_key(label: str) -> int:
    """Sort wiki500k → wiki1000m by triple count, not lexicographically."""
    suffix = label.replace("wiki", "")
    if suffix.endswith("k"):
        return int(suffix[:-1]) * 1_000
    if suffix.endswith("m"):
        return int(suffix[:-1]) * 1_000_000
    return 0


def _resolve_run(df: pd.DataFrame, name: str):
    """Resolve a run identifier to a Timestamp.

    `latest` / `previous` return the newest / second-newest run.
    Anything else is treated as a timestamp prefix and matched
    against `run_started_at` formatted as ISO-8601.
    """
    runs_sorted = df["run_started_at"].drop_duplicates().sort_values().tolist()
    if not runs_sorted:
        raise SystemExit("no runs in CSV")
    if name == "latest":
        return runs_sorted[-1]
    if name == "previous":
        if len(runs_sorted) < 2:
            raise SystemExit("only one run in CSV; nothing to compare to 'previous'")
        return runs_sorted[-2]
    matches = [r for r in runs_sorted if r.isoformat().startswith(name)]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        formatted = ", ".join(m.isoformat() for m in matches)
        raise SystemExit(f"prefix '{name}' matches multiple runs: {formatted}")
    available = ", ".join(r.isoformat() for r in runs_sorted)
    raise SystemExit(f"no run matches '{name}'. Available: {available}")


def cmd_compare(df: pd.DataFrame, run_a: str, run_b: str, cols: list[str]) -> None:
    """Side-by-side diff between two runs at the (mode, dataset) cell
    level. Cells present in only one run are noted explicitly; cells
    present in both render `before → after (+/- delta)` for each
    requested column."""
    ts_a = _resolve_run(df, run_a)
    ts_b = _resolve_run(df, run_b)
    print(f"  A: {ts_a.isoformat()}")
    print(f"  B: {ts_b.isoformat()}")
    print()

    a = df[df["run_started_at"] == ts_a].set_index(["mode", "dataset"])
    b = df[df["run_started_at"] == ts_b].set_index(["mode", "dataset"])
    keys = sorted(set(a.index) | set(b.index), key=lambda k: (k[0], _ds_sort_key(k[1])))

    # Column header
    header = f"  {'mode':<8s} {'dataset':<10s}"
    for c in cols:
        header += f"  {c:>22s}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for mode, dataset in keys:
        row = f"  {mode:<8s} {dataset:<10s}"
        if (mode, dataset) not in a.index:
            row += "  (only in B)"
            print(row)
            continue
        if (mode, dataset) not in b.index:
            row += "  (only in A)"
            print(row)
            continue
        for c in cols:
            try:
                va = float(a.loc[(mode, dataset), c])
                vb = float(b.loc[(mode, dataset), c])
            except (KeyError, TypeError, ValueError):
                row += f"  {'—':>22s}"
                continue
            if pd.isna(va) and pd.isna(vb):
                row += f"  {'—':>22s}"
                continue
            if pd.isna(va):
                row += f"  {'→ ' + _fmt_ms(vb):>22s}"
                continue
            if pd.isna(vb):
                row += f"  {_fmt_ms(va) + ' →':>22s}"
                continue
            delta = vb - va
            arrow = "→" if abs(delta) < max(va, 1) * 0.005 else ("↑" if delta > 0 else "↓")
            cell = f"{_fmt_ms(va)} {arrow} {_fmt_ms(vb)}"
            if abs(delta) >= 0.5:
                pct = (delta / va * 100) if va else 0
                cell += f" ({pct:+.0f}%)"
            row += f"  {cell:>22s}"
        print(row)


def cmd_deltas(df: pd.DataFrame, cols: list[str]) -> None:
    """Per-cell consecutive-run deltas. One row per (mode, dataset, run)
    showing absolute change vs. previous run for each tracked column."""
    if df.empty:
        print("(no rows match)")
        return
    df = df.sort_values(["mode", "dataset", "run_started_at"])
    pieces = []
    for (mode, dataset), grp in df.groupby(["mode", "dataset"]):
        if len(grp) < 2:
            continue
        diff = grp[cols].astype(float).diff().iloc[1:]
        run_at = grp["run_started_at"].iloc[1:].dt.strftime("%Y-%m-%d %H:%M").values
        ver = grp["version"].iloc[1:].values
        for col in cols:
            diff[col] = diff[col].apply(
                lambda v: (
                    ("+" if v > 0 else "") + _fmt_ms(abs(v))
                    if pd.notna(v) and v != 0
                    else ("0" if pd.notna(v) else "—")
                )
            )
        diff.insert(0, "run_started_at", run_at)
        diff.insert(1, "version", ver)
        diff.insert(0, "dataset", dataset)
        diff.insert(0, "mode", mode)
        pieces.append(diff)
    if not pieces:
        print("(need at least 2 runs per cell to compute deltas)")
        return
    print(pd.concat(pieces).to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["runs", "latest", "trends", "deltas", "compare"])
    ap.add_argument("--csv", default=str(DEFAULT_CSV), help=f"Default: {DEFAULT_CSV}")
    ap.add_argument("--mode", help="Filter: default | mapped | disk")
    ap.add_argument("--dataset", help="Filter: wiki500k, wiki5m, …, wiki1000m")
    ap.add_argument(
        "--cols",
        default="",
        help="Comma-separated columns to render. Default: lifecycle (build/save/load/mutate/resave/total). "
        "Use 'all' to render every numeric column.",
    )
    ap.add_argument(
        "compare_args",
        nargs="*",
        help="For `compare`: two run identifiers (timestamp prefix, or 'latest' / 'previous').",
    )
    args = ap.parse_args()

    df = _load(Path(args.csv))
    if args.cmd != "compare":
        df = _filter(df, args.mode, args.dataset)
    cols = _resolve_cols(args.cols, df)

    if args.cmd == "runs":
        cmd_runs(df)
    elif args.cmd == "latest":
        cmd_latest(df, cols)
    elif args.cmd == "trends":
        cmd_trends(df, cols)
    elif args.cmd == "deltas":
        cmd_deltas(df, cols)
    elif args.cmd == "compare":
        if len(args.compare_args) != 2:
            ap.error("compare needs two run identifiers (e.g. 'compare latest previous')")
        # Apply --mode/--dataset filters to BOTH runs symmetrically.
        if args.mode:
            df = df[df["mode"] == args.mode]
        if args.dataset:
            df = df[df["dataset"] == args.dataset]
        cmd_compare(df, args.compare_args[0], args.compare_args[1], cols)


if __name__ == "__main__":
    main()
