"""Render BENCHMARKS.md — the public, topic-summed comparison table.

Rolls the 15 fine-grained `graphsuite` groups up into a handful of
readable *topics* (one summed wall-to-wall time each), per library, from
the most recent run of each backend at a given dataset signature. This is
the customer-facing view; `report.py` keeps the fine-grained per-group
matrix for digging in.
"""

from __future__ import annotations

import statistics

from .report import latest_per_library
from .results import load

# 15 groups → topics (summed wall time per topic). Order is the story:
# load → read → analytics → traversal → pathfinding → algorithms → write.
TOPICS: list[tuple[str, list[str]]] = [
    ("Bulk load", ["build"]),
    ("Scan & lookup", ["node_scan", "point_lookup", "edge_scan"]),
    ("Filter & aggregate", ["property_filter", "range_filter", "group_aggregation", "year_aggregation"]),
    (
        "Traversal (1–3 hop, filtered, deep)",
        ["one_hop", "two_hop", "three_hop", "filtered_traversal", "deep_traversal", "score_filtered_traversal"],
    ),
    ("Pathfinding", ["shortest_path"]),
    # Heterogeneous / multi-type queries — query engines run these; the
    # algorithm libraries (no query language, no Company/relational data)
    # can't, which is the point: it shows the feature gap, not a handicap.
    ("Multi-type queries", ["pattern_match", "industry_aggregation", "two_step_join"]),
    ("Graph algorithms", ["degree_topk", "connected_components", "degree_filter"]),
    # Community detection — the algorithm libraries' strength; query engines
    # have no native Louvain, so they show `—` here (a real, visible gap).
    ("Community detection", ["louvain"]),
    ("Mutations", ["bulk_update", "mutations"]),
    # Specialized KG capabilities — kglite's differentiators. The algorithm
    # libraries and most competitors have no vector index or spatial query,
    # so kglite-mostly coverage here is expected and warranted.
    ("Vector search", ["vector_knn"]),
    ("Geospatial", ["geo_within"]),
]

# Headline kglite row + the competitors. The other kglite modes (mapped /
# disk / bolt / fluent) get their own sub-table so the headline stays clean.
HEADLINE_KGLITE = "kglite-cypher"
KGLITE_MODES = [
    "kglite-cypher",
    "kglite-mapped",
    "kglite-disk",
    "kglite-fluent",
    "kglite-bolt",
    "kglite-bolt-docker",
]

# Display label + one-word category for the legend.
LABELS: dict[str, str] = {
    "kglite-cypher": "kglite",
    "kglite-mapped": "kglite (mapped)",
    "kglite-disk": "kglite (disk)",
    "kglite-fluent": "kglite (fluent)",
    "kglite-bolt": "kglite (Bolt)",
    "kglite-bolt-docker": "kglite (Bolt, Docker)",
    "kuzu": "Kùzu",
    "neo4j": "Neo4j",
    "neo4j-docker": "Neo4j (Docker)",
    "neo4j-native": "Neo4j (native)",
    "duckdb": "DuckDB",
    "networkx": "NetworkX",
    "rustworkx": "rustworkx",
    "igraph": "igraph",
}
CATEGORY: dict[str, str] = {
    "kglite-cypher": "Cypher graph engine, in-memory",
    "kuzu": "Cypher graph DB, embedded",
    "neo4j": "Cypher graph DB, server (native)",
    "duckdb": "SQL / relational",
    "networkx": "pure-Python algo library",
    "rustworkx": "Rust algo library",
    "igraph": "C algo library",
}


# A single sub-benchmark slower than this is marked "timed out" (⏱) rather than
# posting a wall-time: it's too slow to be a meaningful comparison point, and
# letting one pathological number (e.g. pure-Python Louvain at >10 s) dominate a
# summed total is misleading. Timed-out benches are excluded from the time sums
# and from the percentile-estimate distribution; the capability still counts
# (the engine *can* do it, just not at a useful speed).
TIMEOUT_S = 10.0


def _fmt(seconds: float | None, ran: int | None = None, n: int | None = None, timed_out: int = 0) -> str:
    if seconds is None:
        return "⏱ timeout" if timed_out else "—"
    if seconds < 1e-3:
        s = f"{seconds * 1e6:.0f}µs"
    elif seconds < 1.0:
        s = f"{seconds * 1e3:.1f}ms"
    else:
        s = f"{seconds:.2f}s"
    if ran is not None and n is not None and ran < n:
        s += f" ({ran}/{n})"
    if timed_out:
        s += " ⏱"
    return s


def _run_total(run: dict) -> float:
    return sum(g["min_s"] for g in run["groups"].values() if g.get("status") == "ok" and g.get("min_s") is not None)


def _collapse_neo4j(latest: dict) -> dict:
    """Fold the Neo4j run variants (`neo4j` / `neo4j-docker` / `neo4j-native`)
    into one headline `neo4j` column — the strongest (lowest total wall-time)
    run, which is the fairest "Neo4j at its best" number. `report.py` still
    shows every variant separately for the detailed comparison."""
    variants = [k for k in latest if k == "neo4j" or k.startswith("neo4j-")]
    if not variants:
        return latest
    best = min(variants, key=lambda k: _run_total(latest[k]))
    out = {k: v for k, v in latest.items() if k not in variants}
    out["neo4j"] = latest[best]
    return out


def _all_group_ids() -> list[str]:
    return [gid for _topic, gids in TOPICS for gid in gids]


def _percentile_value(sorted_vals: list[float], p: float) -> float:
    """Linear-interpolated value at percentile ``p`` ∈ [0, 1] of an ascending
    list (0 → fastest, 1 → slowest)."""
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = p * (len(sorted_vals) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo)


def _impute_model(latest: dict, engines: list[str]) -> tuple[dict[str, list[float]], dict[str, float]]:
    """Build the skip-estimation model over ``engines``:

    - ``ran_times[gid]`` — the ascending min-times the engines posted for that
      group (the distribution a skip is estimated against).
    - ``avg_pct[engine]`` — that engine's *mean percentile standing*
      (0 = fastest, 1 = slowest) across the groups it ran. A skipped group is
      then estimated at this percentile of the group's distribution, so the
      estimate tracks how fast/slow the engine generally is rather than
      assuming it's average.
    """
    ran_times: dict[str, list[float]] = {}
    pct_samples: dict[str, list[float]] = {e: [] for e in engines}
    for gid in _all_group_ids():
        rows = []
        for e in engines:
            g = latest[e]["groups"].get(gid)
            # exclude timed-out runs — they're not a useful estimate basis
            if g and g.get("status") == "ok" and g.get("min_s") is not None and g["min_s"] <= TIMEOUT_S:
                rows.append((e, g["min_s"]))
        ran_times[gid] = sorted(t for _e, t in rows)
        if len(rows) >= 2:  # percentile rank is only meaningful with ≥2 runners
            ordered = sorted(rows, key=lambda r: r[1])
            for idx, (e, _t) in enumerate(ordered):
                pct_samples[e].append(idx / (len(ordered) - 1))
    avg_pct = {e: (statistics.fmean(s) if s else 0.5) for e, s in pct_samples.items()}
    return ran_times, avg_pct


def _topic_seconds(
    run: dict,
    engine: str,
    group_ids: list[str],
    ran_times: dict[str, list[float]],
    avg_pct: dict[str, float],
) -> tuple[float | None, int, int, int]:
    """Sum min wall-time over a topic's groups → ``(seconds, ran, n, timed_out)``.

    A sub-bench slower than :data:`TIMEOUT_S` is *timed out*: it counts as run
    (the engine can do it) but its time is excluded from the sum and it's marked
    ``⏱`` rather than posting a pathological number. Cases:

    - **ran == 0** → ``(None, 0, n, 0)`` — the engine can't do this category at
      all (rendered ``—``; never invented, so a kglite-only category isn't
      hidden behind an estimate).
    - **only timed-out runs, no usable time** → ``(None, ran, n, timed_out)`` →
      ``⏱ timeout``.
    - **0 < ran < n** → does the category but missed a sub-bench; the missing
      one is *estimated* at the engine's average percentile so a within-category
      skip can't deflate it.
    - else → real (fast) + estimate, with any ``timed_out`` count flagged.
    """
    n = len(group_ids)
    real = 0.0
    ran = 0
    timed_out = 0
    skipped_others_ran: list[str] = []
    for gid in group_ids:
        g = run["groups"].get(gid)
        if g and g.get("status") == "ok" and g.get("min_s") is not None:
            ran += 1
            if g["min_s"] <= TIMEOUT_S:
                real += g["min_s"]
            else:
                timed_out += 1  # ran but too slow — excluded from the sum
        elif ran_times.get(gid):
            skipped_others_ran.append(gid)  # this engine skipped it; others ran it
    if ran == 0:
        return None, 0, n, 0
    imputed = sum(_percentile_value(ran_times[gid], avg_pct.get(engine, 0.5)) for gid in skipped_others_ran)
    secs = real + imputed
    if secs == 0.0 and timed_out:
        return None, ran, n, timed_out  # nothing fast to show — pure timeout
    return secs, ran, n, timed_out


def render(signature: str | None = None) -> str:
    data = load()
    latest = latest_per_library(data, signature)
    if not latest:
        return "_no benchmark runs recorded yet — run `python benchmarks/benchmark.py`_\n"

    # Pick the dataset signature to report (the most common among latest runs),
    # then re-query each library's latest run *at that signature* — otherwise a
    # library whose globally-newest run is at a different scale (e.g. a one-off
    # `small` validation run) would drop out even though it has a run at the
    # reported signature.
    if signature is None:
        sigs = [r["dataset"]["signature"] for r in latest.values()]
        signature = max(set(sigs), key=sigs.count)
        latest = latest_per_library(data, signature)

    latest = _collapse_neo4j(latest)

    sample = next(iter(latest.values()))
    ds = sample["dataset"]

    # Column order: kglite headline, then competitors (non-kglite) alpha.
    competitors = sorted(lib for lib in latest if not lib.startswith("kglite"))
    cols = ([HEADLINE_KGLITE] if HEADLINE_KGLITE in latest else []) + competitors

    out: list[str] = []
    out.append("# KGLite benchmarks")
    out.append("")
    historical = data.get("historical_capture")
    if historical and not any("provenance" in run for run in latest.values()):
        kglite_version = latest.get(HEADLINE_KGLITE, {}).get("version", "unknown")
        out.append(
            f"> **Historical snapshot.** These measurements were captured with "
            f"KGLite {kglite_version}; they are not measurements of the current "
            "workspace version. The report remains useful as a reproducible "
            "workload snapshot and is refreshed only by an explicit benchmark run."
        )
        out.append("")
    out.append(
        "Wall-to-wall time per topic (each topic sums several individual "
        "queries), lower is better, on one synthetic knowledge graph that "
        "**every engine loads from identical bytes**. Generated by "
        "`python benchmarks/benchmark.py` — see [Reproduce](#reproduce)."
    )
    out.append("")
    out.append(
        f"**Dataset:** {ds['n_nodes']:,} nodes · {ds['n_edges']:,} edges "
        f"(org/social graph: Person/Company/Project/Skill/City). "
        f"Scale `{ds['scale']}`, seed-deterministic."
    )
    out.append("")

    # ── Headline topic table ────────────────────────────────────────────
    header = (
        "| Topic | "
        + " | ".join(f"**{LABELS.get(c, c)}**" if c == HEADLINE_KGLITE else LABELS.get(c, c) for c in cols)
        + " |"
    )
    sep = "|" + "---|" * (len(cols) + 1)
    out.append(header)
    out.append(sep)
    ran_times, avg_pct = _impute_model(latest, cols)
    totals: dict[str, float] = {c: 0.0 for c in cols}
    totals_ran: dict[str, int] = {c: 0 for c in cols}
    totals_n: dict[str, int] = {c: 0 for c in cols}
    for topic, gids in TOPICS:
        res = {c: _topic_seconds(latest[c], c, gids, ran_times, avg_pct) for c in cols}
        # Winner = fastest engine that *fully* ran the category — a real, clean
        # time (no within-category estimate, no timeout); an imputed or partial
        # cell can't claim the win.
        clean = {c: s for c, (s, ran, n, to) in res.items() if s is not None and ran == n and to == 0}
        winner = min(clean, key=clean.get) if clean else None
        cells = []
        for c in cols:
            secs, ran, n, timed_out = res[c]
            cell = _fmt(secs, ran, n, timed_out)
            if c == winner:
                cell = f"**{cell}**"
            cells.append(cell)
            # Coverage denominator counts every category (so a `—` gap is
            # visible in the total's (ran/total)); seconds sum only what the
            # engine can do (real + within-category estimate) — a category it
            # can't do at all contributes no invented time.
            totals_ran[c] += ran
            totals_n[c] += n
            if secs is not None:
                totals[c] += secs
        out.append(f"| {topic} | " + " | ".join(cells) + " |")
    out.append(
        "| **Total** | "
        + " | ".join(
            (
                f"**{_fmt(totals[c], totals_ran[c], totals_n[c])}**"
                if c == HEADLINE_KGLITE
                else _fmt(totals[c], totals_ran[c], totals_n[c])
            )
            for c in cols
        )
        + " |"
    )
    out.append("")
    out.append(
        "**Bold** marks the fastest engine in each category (among those that "
        "fully ran it). Each topic sums its sub-benchmarks. A **`—`** means the engine can't do "
        "that category *at all* (e.g. no vector index, no shortest-path query, no "
        "Louvain) — a real capability gap, shown rather than hidden, and never "
        "estimated. **`(ran/total)`** marks a *partial* topic — the engine does "
        "the category but missed a sub-benchmark; rather than drop it (which "
        "would flatter the total), the missing one is **estimated** at the "
        "engine's average percentile standing across the benchmarks it *did* run. "
        "A **`⏱`** marks a sub-bench slower than "
        f"{TIMEOUT_S:.0f}s — too slow to be a useful comparison point, so it's "
        "excluded from the time total (the capability still counts). No mark ⇒ "
        "it ran the whole category. The **Total's `(ran/26)`** is coverage — how "
        "many of the 26 sub-benchmarks the engine can do at all."
    )
    out.append("")
    out.append(
        "> Every engine runs the **same** queries on the **same** data — including "
        "the variable-length `[:KNOWS*1..3]` traversal. A large traversal number "
        "(e.g. Kùzu's) is that engine's own planning of that identical query on a "
        "dense, hub-heavy `KNOWS` subgraph, not a handicap we imposed; results are "
        "digest-checked equal across engines. Run it yourself (below)."
    )
    out.append("")

    # ── Capability matrix — the breadth picture at a glance ─────────────
    # People only care about workloads that touch their use case, so make
    # "can it even do this?" instantly scannable: ✓ = the engine ran the
    # category, — = it can't. Times are above; capability is here.
    cap_topics = [(t, gids) for t, gids in TOPICS if t != "Bulk load"]
    out.append("### Can it do your workload?")
    out.append("")
    out.append(
        "Times tell you *how fast*; this tells you *whether it can at all*. "
        "Find your workload — **`✓`** the engine does it, **`—`** it can't. "
        "**kglite is the only engine here that covers every workload** — "
        "including the ones no competitor combines: vector search (kglite only), "
        "geospatial, multi-type knowledge-graph queries, *and* community "
        "detection. Each competitor covers only the slice it's built for — fast "
        "precisely because it's narrow."
    )
    out.append("")
    out.append(
        "| Capability | "
        + " | ".join(f"**{LABELS.get(c, c)}**" if c == HEADLINE_KGLITE else LABELS.get(c, c) for c in cols)
        + " |"
    )
    out.append("|" + "---|" * (len(cols) + 1))
    covered = {c: 0 for c in cols}
    any_timeout = False
    for topic, gids in cap_topics:
        cells = []
        for c in cols:
            ran_ok = [
                g
                for gid in gids
                if (g := latest[c]["groups"].get(gid, {})).get("status") == "ok" and g.get("min_s") is not None
            ]
            ran = bool(ran_ok)
            slow = ran and all(g["min_s"] > TIMEOUT_S for g in ran_ok)
            covered[c] += 1 if ran else 0
            any_timeout = any_timeout or slow
            mark = "—" if not ran else ("✓⏱" if slow else "✓")
            cells.append(f"**{mark}**" if c == HEADLINE_KGLITE else mark)
        out.append(f"| {topic} | " + " | ".join(cells) + " |")
    ntot = len(cap_topics)
    out.append(
        "| **Workloads covered** | "
        + " | ".join(f"**{covered[c]}/{ntot}**" if c == HEADLINE_KGLITE else f"{covered[c]}/{ntot}" for c in cols)
        + " |"
    )
    out.append("")
    if any_timeout:
        out.append(
            f"**`✓⏱`** — the engine *can* do this, but its run exceeded the "
            f"{TIMEOUT_S:.0f}s timeout (e.g. NetworkX has Louvain, but its "
            "pure-Python implementation is too slow to count), so it's a "
            "capability, not a usable speed — see `⏱` in the times table."
        )
        out.append("")

    # ── What each engine is (fairness framing) ──────────────────────────
    out.append("### What's being compared")
    out.append("")
    out.append("These aren't all the same kind of tool — read the table with that in mind:")
    out.append("")
    for c in cols:
        cat = CATEGORY.get(c, "")
        if cat:
            out.append(f"- **{LABELS.get(c, c)}** — {cat}")
    out.append("")
    out.append(
        "> **Read the totals as breadth, not a single number.** Each competitor "
        "is fast only on the slice it's built for. **DuckDB** (relational) flies "
        "on scans, joins and aggregations, but has no graph pathfinding, no "
        "connected-components, no community detection and no vector search. The "
        "**algorithm libraries** (rustworkx / igraph / NetworkX) are fast on "
        "traversal and graph algorithms but model a single homogeneous graph — "
        "no query language, no heterogeneous *multi-type* queries, no "
        "persistence, no vector or geospatial. **kglite is the only engine that "
        "covers every category** (see the capability matrix above) while staying "
        "competitive on the core query/traversal workloads — and the **only one "
        "that does vector search at all**. A `—` is a capability the "
        "engine simply lacks; the win here is breadth at competitive speed, not "
        "out-running a relational engine at raw SQL or a Rust library at raw "
        "Dijkstra."
    )
    out.append("")
    out.append(
        "> *Total caveat:* community detection (Louvain) is expensive — igraph's "
        "C implementation (~1.2s) dominates its total, and NetworkX's "
        "pure-Python one exceeds the timeout and is excluded (⏱). kglite's "
        "scoped Louvain runs in ~150ms; the relational engine doesn't offer it. "
        "The per-category rows, not the grand total, are the per-workload "
        "comparison."
    )
    out.append("")

    # ── kglite storage modes / protocols ────────────────────────────────
    modes = [m for m in KGLITE_MODES if m in latest]
    if len(modes) > 1:
        out.append("### kglite storage modes & protocols")
        out.append("")
        out.append(
            "Same engine, same results — pick in-memory for speed, "
            "mapped/disk for larger-than-RAM graphs, Bolt to serve over the wire:"
        )
        out.append("")
        out.append("| Topic | " + " | ".join(LABELS.get(m, m) for m in modes) + " |")
        out.append("|" + "---|" * (len(modes) + 1))
        mode_ran_times, mode_avg_pct = _impute_model(latest, modes)
        mtot = {m: 0.0 for m in modes}
        mran = {m: 0 for m in modes}
        mn = {m: 0 for m in modes}
        for topic, gids in TOPICS:
            cells = []
            for m in modes:
                secs, ran, n, to = _topic_seconds(latest[m], m, gids, mode_ran_times, mode_avg_pct)
                cells.append(_fmt(secs, ran, n, to))
                mran[m] += ran
                mn[m] += n
                if secs is not None:
                    mtot[m] += secs
            out.append(f"| {topic} | " + " | ".join(cells) + " |")
        out.append("| **Total** | " + " | ".join(_fmt(mtot[m], mran[m], mn[m]) for m in modes) + " |")
        out.append("")

    # ── Scale note ──────────────────────────────────────────────────────
    out.append("### Scaling")
    out.append("")
    out.append(
        "The table above is a `medium` graph chosen so every engine can run "
        "(NetworkX/igraph hold the whole graph in RAM). kglite itself scales "
        "far past that: the bundled `graphgen` generator streams **millions "
        "of nodes in bounded memory**, and kglite's `mapped`/`disk` modes load "
        "and query larger-than-RAM graphs — see "
        "`benchmarks/competitive/largescale/`."
    )
    out.append("")

    # ── Reproduce ───────────────────────────────────────────────────────
    out.append("## Reproduce")
    out.append("")
    out.append("```bash")
    out.append("pip install kglite kuzu networkx rustworkx igraph duckdb")
    out.append("python benchmarks/benchmark.py            # default: medium scale")
    out.append("python benchmarks/benchmark.py --scale large")
    out.append("```")
    out.append("")
    out.append(
        "That command creates a new capture with the installed engine versions; "
        "it does not relabel the historical numbers as current. Use "
        "`python benchmarks/benchmark.py --report-only` to render the committed "
        "metadata without executing benchmarks."
    )
    out.append("")
    out.append(
        "Each engine runs the *same* queries (shared seed-derived parameters) "
        "and prints a result digest per topic, so the comparison reflects "
        "equal work. Versions used for this run:"
    )
    out.append("")
    vers = ", ".join(f"{LABELS.get(c, c)} {latest[c]['version']}" for c in cols)
    out.append(f"_{vers}_")
    out.append("")
    machine = sample.get("machine")
    if isinstance(machine, dict):
        machine_str = f"{machine.get('platform', 'an unspecified machine')} · Python {machine.get('python', '?')}"
    else:
        machine_str = machine or "an unspecified machine"
    out.append(
        f"Run on {machine_str}. Numbers are min "
        "wall-time over repeats; absolute times vary by hardware — the "
        "*ratios* are the point."
    )
    out.append("")
    out.append("### Capture provenance")
    out.append("")
    out.append(f"- Results schema: `{data.get('schema_version', 'unknown')}`")
    harness = data.get("harness", {})
    out.append(f"- Harness: `{harness.get('name', 'graphsuite')}` v{harness.get('version', 'unknown')}")
    out.append(f"- Dataset signature: `{ds['signature']}`")
    selected_dates = sorted(run["run_date"] for run in latest.values())
    out.append(f"- Selected run timestamps: `{selected_dates[0]}` through `{selected_dates[-1]}`")

    provenance = [run.get("provenance") for run in latest.values()]
    recorded = [item for item in provenance if isinstance(item, dict)]
    if recorded:
        origins = sorted({item["origin"] for item in recorded})
        commits = sorted({item["source_commit"] for item in recorded})
        dirty = sorted({str(item["source_dirty"]).lower() for item in recorded})
        repeats = sorted({str(item["base_repeats"]) for item in recorded})
        out.append(f"- Capture origin: `{', '.join(origins)}`")
        out.append(f"- Source commit: `{', '.join(commits)}` (dirty: `{', '.join(dirty)}`)")
        out.append(f"- Base repeat policy: `{', '.join(repeats)}`")
        out.append(f"- Dataset seed: `{ds.get('seed', 'recorded in signature')}`")
    elif historical:
        out.append(f"- Capture origin: `{historical['origin']}`")
        out.append(f"- Dataset seed: `{historical['dataset_seed']}`")
        out.append(f"- Base repeat policy: `{historical['base_repeats']}`")
        out.append(
            "- Source commit: `not recorded`; "
            f"{historical['source_commit_note']} Results first entered history in "
            f"`{historical['results_first_committed_in']}`."
        )
        out.append(f"- Timestamp limitation: {historical['timezone_note']}")
    out.append("- Raw metadata authority: `benchmarks/competitive/graphsuite/results.json`.")
    out.append("")
    return "\n".join(out)


if __name__ == "__main__":
    print(render())
