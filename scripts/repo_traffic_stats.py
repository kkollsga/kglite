#!/usr/bin/env python3
"""Hand-rolled GitHub repository traffic collector — no third-party Actions.

GitHub keeps traffic (views / clones / referrers) for only the last 14 days.
This fetches them from GitHub's own Traffic API (via the `gh` CLI), upserts the
daily time series into CSVs so repeated runs accumulate full history, and
renders a PDF report. Only first-party pieces: the `gh` CLI + this script +
matplotlib.

Usage:
    GH_TOKEN=<token> python repo_traffic_stats.py --repo owner/name --data-dir DIR

Requires `gh` on PATH (uses GH_TOKEN if set) and matplotlib. The token needs
traffic access — a PAT with `repo` (classic) / `Administration: Read`
(fine-grained), or the Actions GITHUB_TOKEN with `administration: read`.
"""

from __future__ import annotations

import argparse
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
    """Merge `rows` (keyed by their 'date') into the CSV at `path`.

    GitHub returns the trailing 14-day window each call; a given past day's
    counts are final once the day closes, so upsert-by-date (new wins) both
    fills gaps and never double-counts.
    """
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


def render_pdf(path: Path, repo: str, views: dict, clones: dict, referrers: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.dates import DateFormatter
    import matplotlib.pyplot as plt

    def series(data: dict, total_key: str):
        items = sorted(data.values(), key=lambda r: r["date"])
        xs = [dt.date.fromisoformat(r["date"]) for r in items]
        return xs, [int(r[total_key]) for r in items], [int(r["unique"]) for r in items]

    now = dt.datetime.now(dt.timezone.utc)
    with PdfPages(path) as pdf:
        fig, axes = plt.subplots(2, 1, figsize=(8.5, 11))
        fig.suptitle(f"GitHub traffic — {repo}\ngenerated {now:%Y-%m-%d %H:%M UTC}", fontsize=13)
        for ax, data, total_key, label in (
            (axes[0], views, "views", "Views"),
            (axes[1], clones, "clones", "Clones"),
        ):
            if data:
                xs, total, uniq = series(data, total_key)
                ax.plot(xs, total, marker="o", ms=3, lw=1.3, label=f"{label.lower()} (total)")
                ax.plot(xs, uniq, marker="o", ms=3, lw=1.3, label=f"unique {label.lower()}")
                ax.set_title(
                    f"{label}: {sum(total)} total over {len(xs)} day(s) ({xs[0]:%Y-%m-%d} → {xs[-1]:%Y-%m-%d})",
                    fontsize=10,
                )
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)
                ax.xaxis.set_major_formatter(DateFormatter("%m-%d"))
                fig.autofmt_xdate()
            else:
                ax.text(0.5, 0.5, f"no {label.lower()} data yet", ha="center", va="center")
                ax.set_axis_off()
        if referrers:
            top = ", ".join(f"{r['referrer']} ({r['count']})" for r in referrers[:5])
            fig.text(0.5, 0.02, f"Top referrers (last 14d): {top}", ha="center", fontsize=7, wrap=True)
        fig.tight_layout(rect=(0, 0.04, 1, 0.95))
        pdf.savefig(fig)
        plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", required=True, help="owner/name")
    ap.add_argument("--data-dir", required=True, help="directory for the CSVs + PDF")
    args = ap.parse_args()

    out = Path(args.data_dir)
    out.mkdir(parents=True, exist_ok=True)

    views_raw = gh_api(f"repos/{args.repo}/traffic/views")
    clones_raw = gh_api(f"repos/{args.repo}/traffic/clones")
    refs_raw = gh_api(f"repos/{args.repo}/traffic/popular/referrers")

    v_rows = [
        {"date": x["timestamp"][:10], "views": x["count"], "unique": x["uniques"]} for x in views_raw.get("views", [])
    ]
    c_rows = [
        {"date": x["timestamp"][:10], "clones": x["count"], "unique": x["uniques"]}
        for x in clones_raw.get("clones", [])
    ]
    views = upsert_timeseries(out / "views.csv", v_rows, ["date", "views", "unique"]) if v_rows else {}
    clones = upsert_timeseries(out / "clones.csv", c_rows, ["date", "clones", "unique"]) if c_rows else {}

    # Referrers are a 14-day snapshot (not a daily series): append, stamped with
    # the capture date, so you keep a history of where traffic came from.
    captured = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")
    referrers = [
        {"captured": captured, "referrer": r["referrer"], "count": r["count"], "unique": r["uniques"]} for r in refs_raw
    ]
    if referrers:
        ref_path = out / "referrers.csv"
        new = not ref_path.exists()
        with ref_path.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["captured", "referrer", "count", "unique"])
            if new:
                w.writeheader()
            w.writerows(referrers)

    render_pdf(out / "report.pdf", args.repo, views, clones, referrers)
    print(
        f"OK: views={len(views)} days, clones={len(clones)} days, "
        f"referrers={len(referrers)} → {out}/ (views.csv, clones.csv, referrers.csv, report.pdf)"
    )


if __name__ == "__main__":
    main()
