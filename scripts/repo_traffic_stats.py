#!/usr/bin/env python3
"""Hand-rolled GitHub repository traffic collector — no third-party Actions.

GitHub keeps traffic (views / clones / referrers / popular content) for only
the last 14 days. This fetches them from GitHub's own Traffic API (via `gh`),
upserts the daily series into CSVs so repeated runs accumulate full history,
and renders a dark-mode PDF report. Only first-party pieces: `gh` + this
script + matplotlib.

Usage:
    GH_TOKEN=<token> python repo_traffic_stats.py --repo owner/name --data-dir DIR

Note on referrers / popular content: GitHub returns these as a single 14-day
aggregate (top 10), NOT a per-day series. We snapshot that aggregate daily
(stamped with the capture date) and chart it as standing bars; once ≥2 capture
dates exist it becomes a separate curve per entry over the capture dates.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import datetime as dt
import json
from pathlib import Path
import subprocess
import sys


def gh_api(path: str) -> dict | list:
    """Call `gh api <path>` and return parsed JSON (exits on failure)."""
    proc = subprocess.run(["gh", "api", path], capture_output=True, text=True)
    if proc.returncode != 0:
        sys.exit(f"gh api {path} failed:\n{proc.stderr.strip()}")
    return json.loads(proc.stdout)


def upsert_timeseries(path: Path, rows: list[dict], fields: list[str]) -> dict[str, dict]:
    """Merge `rows` (keyed by 'date') into the CSV — upsert by date, new wins."""
    existing: dict[str, dict] = {}
    if path.exists():
        with path.open(newline="") as f:
            for r in csv.DictReader(f):
                existing[r["date"]] = r
    for r in rows:
        existing[r["date"]] = {k: str(r[k]) for k in fields}
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for key in sorted(existing):
            w.writerow(existing[key])
    return existing


def append_snapshot(path: Path, rows: list[dict], fields: list[str]) -> None:
    """Append a dated snapshot (referrers / paths), replacing any rows already
    captured today so a same-day re-run doesn't duplicate them."""
    today = rows[0]["captured"] if rows else None
    kept: list[dict] = []
    if path.exists():
        with path.open(newline="") as f:
            kept = [r for r in csv.DictReader(f) if r.get("captured") != today]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in kept:
            w.writerow({k: r.get(k, "") for k in fields})
        for r in rows:
            w.writerow({k: str(r[k]) for k in fields})


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


# ── PDF rendering (dark mode) ────────────────────────────────────────────────


def _plot_timeseries(ax, data: dict, total_key: str, label: str) -> None:
    """Total on the left y-axis; unique on a secondary (right) y-axis."""
    ax.set_title(label, fontsize=10)
    if not data:
        ax.text(0.5, 0.5, f"no {label.lower()} data yet", ha="center", va="center", color="#9E9E9E")
        return
    items = sorted(data.values(), key=lambda r: r["date"])
    xs = [dt.date.fromisoformat(r["date"]) for r in items]
    total = [int(r[total_key]) for r in items]
    uniq = [int(r["unique"]) for r in items]
    (l_total,) = ax.plot(xs, total, marker="o", ms=3, lw=1.4, color="#4FC3F7", label=f"{label.lower()} (total)")
    ax.set_ylabel(f"{label.lower()} (total)", color="#4FC3F7", fontsize=8)
    ax.tick_params(axis="y", labelcolor="#4FC3F7")
    ax.grid(True, alpha=0.25)
    ax2 = ax.twinx()  # secondary axis for unique
    (l_uniq,) = ax2.plot(xs, uniq, marker="o", ms=3, lw=1.4, color="#FFB74D", label=f"unique {label.lower()}")
    ax2.set_ylabel(f"unique {label.lower()}", color="#FFB74D", fontsize=8)
    ax2.tick_params(axis="y", labelcolor="#FFB74D")
    ax.set_title(
        f"{label}: {sum(total)} total / {sum(uniq)} unique  ({xs[0]:%Y-%m-%d} → {xs[-1]:%Y-%m-%d})",
        fontsize=10,
    )
    ax.legend(handles=[l_total, l_uniq], fontsize=8, loc="upper left")


def _plot_aggregate_by_day(
    ax, rows: list[dict], key: str, title: str, label_field: str | None = None, top: int = 8
) -> None:
    """Standing (vertical) bars of the current 14-day snapshot; once ≥2 capture
    dates exist, a separate curve per entry over the capture dates. Always
    rendered, even with no data yet."""
    label_field = label_field or key
    ax.set_title(title, fontsize=10)
    if not rows:
        ax.text(0.5, 0.5, "no data yet (last 14 days)", ha="center", va="center", color="#9E9E9E", fontsize=8)
        ax.set_yticks([])
        ax.set_xticks([])
        return

    by_count: dict[str, dict[str, int]] = defaultdict(dict)
    labels: dict[str, str] = {}
    dates = sorted({r["captured"] for r in rows})
    for r in rows:
        by_count[r[key]][r["captured"]] = int(r["count"])
        labels[r[key]] = r.get(label_field) or r[key]
    latest = dates[-1]
    ranked = sorted(by_count, key=lambda e: by_count[e].get(latest, 0), reverse=True)[:top]

    if len(dates) < 2:
        vals = [by_count[e].get(latest, 0) for e in ranked]
        ax.bar(range(len(ranked)), vals, color="#4FC3F7")
        ax.set_xticks(range(len(ranked)))
        ax.set_xticklabels([labels[e][:24] for e in ranked], rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("count (last 14 days)", fontsize=8)
        ax.set_title(f"{title} (14-day snapshot {latest})", fontsize=10)
    else:
        xs = [dt.date.fromisoformat(d) for d in dates]
        for e in ranked:
            ys = [by_count[e].get(d, 0) for d in dates]
            ax.plot(xs, ys, marker="o", ms=2.5, lw=1.2, label=labels[e][:24])
        ax.legend(fontsize=6, ncol=2, loc="upper left")
        ax.grid(True, alpha=0.25)
        ax.set_ylabel("rolling 14-day count", fontsize=8)
        ax.set_title(f"{title} (by capture date)", fontsize=10)


def render_pdf(out_dir: Path, repo: str, views: dict, clones: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.dates import DateFormatter
    import matplotlib.pyplot as plt

    plt.style.use("dark_background")
    referrers = read_csv(out_dir / "referrers.csv")
    paths = read_csv(out_dir / "paths.csv")
    now = dt.datetime.now(dt.timezone.utc)

    with PdfPages(out_dir / "report.pdf") as pdf:
        # Page 1 (portrait) — views + clones; total left axis, unique right axis.
        fig, axes = plt.subplots(2, 1, figsize=(8.5, 11))
        fig.suptitle(f"GitHub traffic — {repo}\ngenerated {now:%Y-%m-%d %H:%M UTC}", fontsize=13)
        _plot_timeseries(axes[0], views, "views", "Views")
        _plot_timeseries(axes[1], clones, "clones", "Clones")
        for ax in axes:
            if ax.has_data():
                ax.xaxis.set_major_formatter(DateFormatter("%m-%d"))
        fig.autofmt_xdate()
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        pdf.savefig(fig)
        plt.close(fig)

        # Page 2 (landscape) — referrers + popular content, SIDE BY SIDE.
        fig, axes = plt.subplots(1, 2, figsize=(11, 6))
        fig.suptitle(f"Referrers & popular content — {repo}", fontsize=13)
        _plot_aggregate_by_day(axes[0], referrers, "referrer", "Referring sites")
        _plot_aggregate_by_day(axes[1], paths, "path", "Popular content", label_field="title")
        for ax in axes:
            if ax.lines:  # only the (multi-date) curve variant has a dated x-axis
                ax.xaxis.set_major_formatter(DateFormatter("%m-%d"))
        fig.text(
            0.5,
            0.02,
            "GitHub exposes referrers/content only as a 14-day aggregate — bars now, "
            "per-entry curves as daily snapshots accumulate.",
            ha="center",
            fontsize=7,
            color="#9E9E9E",
        )
        fig.tight_layout(rect=(0, 0.04, 1, 0.93))
        pdf.savefig(fig)
        plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", required=True, help="owner/name")
    ap.add_argument("--data-dir", required=True, help="directory for the CSVs + PDF")
    args = ap.parse_args()

    out = Path(args.data_dir)
    out.mkdir(parents=True, exist_ok=True)
    captured = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")

    views_raw = gh_api(f"repos/{args.repo}/traffic/views")
    clones_raw = gh_api(f"repos/{args.repo}/traffic/clones")
    refs_raw = gh_api(f"repos/{args.repo}/traffic/popular/referrers")
    paths_raw = gh_api(f"repos/{args.repo}/traffic/popular/paths")

    v_rows = [
        {"date": x["timestamp"][:10], "views": x["count"], "unique": x["uniques"]} for x in views_raw.get("views", [])
    ]
    c_rows = [
        {"date": x["timestamp"][:10], "clones": x["count"], "unique": x["uniques"]}
        for x in clones_raw.get("clones", [])
    ]
    views = upsert_timeseries(out / "views.csv", v_rows, ["date", "views", "unique"]) if v_rows else {}
    clones = upsert_timeseries(out / "clones.csv", c_rows, ["date", "clones", "unique"]) if c_rows else {}

    ref_rows = [
        {"captured": captured, "referrer": r["referrer"], "count": r["count"], "unique": r["uniques"]} for r in refs_raw
    ]
    if ref_rows:
        append_snapshot(out / "referrers.csv", ref_rows, ["captured", "referrer", "count", "unique"])
    path_rows = [
        {"captured": captured, "path": p["path"], "title": p["title"], "count": p["count"], "unique": p["uniques"]}
        for p in paths_raw
    ]
    if path_rows:
        append_snapshot(out / "paths.csv", path_rows, ["captured", "path", "title", "count", "unique"])

    render_pdf(out, args.repo, views, clones)
    print(f"OK: views={len(views)}d clones={len(clones)}d referrers={len(ref_rows)} paths={len(path_rows)} → {out}/")


if __name__ == "__main__":
    main()
