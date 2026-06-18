"""Render the results datafile as a per-group comparison matrix.

Reports the *combined time for all tests in each group* (the group's
min-over-repeats wall time) for the most recent run of each library, plus
a total. Skipped groups show as `skip`, errors as `err`.
"""

from __future__ import annotations

from typing import Any

from .base import GROUPS
from .results import RESULTS_PATH, load


def _fmt(seconds: float | None) -> str:
    if seconds is None:
        return ""
    if seconds < 1e-3:
        return f"{seconds * 1e6:.0f}us"
    if seconds < 1.0:
        return f"{seconds * 1e3:.1f}ms"
    return f"{seconds:.3f}s"


def latest_per_library(data: dict[str, Any], signature: str | None = None) -> dict[str, dict[str, Any]]:
    """Most recent run per library, optionally filtered to one dataset
    signature."""
    out: dict[str, dict[str, Any]] = {}
    for run in data["runs"]:
        if signature is not None and run["dataset"]["signature"] != signature:
            continue
        lib = run["library"]
        if lib not in out or run["run_date"] > out[lib]["run_date"]:
            out[lib] = run
    return out


def render(signature: str | None = None) -> str:
    data = load()
    runs = latest_per_library(data, signature)
    if not runs:
        return "(no runs recorded)"

    libs = sorted(runs)
    lines: list[str] = []

    # dataset banner from any run
    sample = next(iter(runs.values()))
    dsd = sample["dataset"]
    lines.append(
        f"Dataset: scale={dsd['scale']} signature={dsd['signature']} nodes={dsd['n_nodes']:,} edges={dsd['n_edges']:,}"
    )
    lines.append("Combined wall-time per group (min over repeats). Lower is better.")
    lines.append("")

    namew = max(len(g[0]) for g in GROUPS) + 1
    colw = max(12, max(len(lib) for lib in libs) + 1)

    header = "group".ljust(namew) + "".join(lib.rjust(colw) for lib in libs)
    lines.append(header)
    lines.append("-" * len(header))

    totals: dict[str, float] = {lib: 0.0 for lib in libs}
    counted: dict[str, int] = {lib: 0 for lib in libs}

    for gid, _desc, _m in GROUPS:
        row = gid.ljust(namew)
        for lib in libs:
            g = runs[lib]["groups"].get(gid, {})
            status = g.get("status")
            if status == "ok":
                cell = _fmt(g.get("min_s"))
                totals[lib] += g.get("min_s") or 0.0
                counted[lib] += 1
            elif status == "skip":
                cell = "skip"
            elif status == "err":
                cell = "err"
            else:
                cell = "-"
            row += cell.rjust(colw)
        lines.append(row)

    lines.append("-" * len(header))
    # Average per group (mean over the groups that ran). Fairer than a sum:
    # it doesn't penalise a library for covering more groups than another.
    avg_row = "AVG/group".ljust(namew)
    for lib in libs:
        avg = (totals[lib] / counted[lib]) if counted[lib] else None
        avg_row += _fmt(avg).rjust(colw)
    lines.append(avg_row)
    nrow = "(groups ok)".ljust(namew)
    for lib in libs:
        nrow += f"{counted[lib]}/{len(GROUPS)}".rjust(colw)
    lines.append(nrow)
    lines.append("")
    lines.append("Versions: " + ", ".join(f"{lib}={runs[lib]['version']}" for lib in libs))
    lines.append(f"Datafile: {RESULTS_PATH}")
    return "\n".join(lines)


def render_parity(signature: str | None = None) -> str:
    """Cross-backend result-parity report.

    For each group, cluster the libraries by their result digest. If every
    library agrees there is one cluster (PASS). Divergence among the kglite
    **Cypher-driven storage modes** (memory / mapped / disk / bolt — which
    run the *identical* query) is an ERROR: those must be byte-for-byte
    equal. Divergence involving a different surface (fluent) or a different
    library is INFO (expected for a couple of groups — see the README
    walk-vs-trail note).
    """
    data = load()
    runs = latest_per_library(data, signature)
    if not runs:
        return "(no runs recorded)"
    libs = sorted(runs)
    # the modes that run identical Cypher and therefore MUST agree exactly
    strict = {"kglite-cypher", "kglite-mapped", "kglite-disk", "kglite-bolt"}

    lines = ["Result parity across backends (per group). digest = hash of the actual result set."]
    sample = next(iter(runs.values()))["dataset"]
    lines.append(f"Dataset: {sample['signature']}  ({sample['n_nodes']:,} nodes, {sample['n_edges']:,} edges)")
    lines.append("")

    kglite_errors = 0
    for gid, _desc, _m in GROUPS:
        if gid == "mutations":
            continue  # per-backend write workloads differ; not a parity target
        clusters: dict[str, list[str]] = {}
        for lib in libs:
            g = runs[lib]["groups"].get(gid, {})
            if g.get("status") != "ok" or "digest" not in g:
                continue
            clusters.setdefault(g["digest"], []).append(lib)
        if not clusters:
            continue
        if len(clusters) == 1:
            verdict = "PASS"
        else:
            # do the strict (identical-query) kglite modes disagree?
            strict_digests = {d for d, ls in clusters.items() if any(li in strict for li in ls)}
            if len(strict_digests) > 1:
                verdict = "ERROR(kglite storage modes disagree)"
                kglite_errors += 1
            else:
                verdict = "INFO(library/surface variance)"
        lines.append(f"  {gid:<22} {verdict}")
        if len(clusters) > 1:
            for d, ls in sorted(clusters.items(), key=lambda kv: -len(kv[1])):
                n = runs[ls[0]]["groups"][gid].get("sanity")
                lines.append(f"       digest {d} (n={n}): {', '.join(ls)}")
    lines.append("")
    lines.append(
        "SUMMARY: "
        + (
            "kglite storage-mode parity OK (memory/mapped/disk/bolt identical)"
            if kglite_errors == 0
            else f"{kglite_errors} kglite storage-mode PARITY FAILURE(s)"
        )
    )
    return "\n".join(lines)


if __name__ == "__main__":
    print(render())
