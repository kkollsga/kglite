"""Render the public, topic-summed adapter-workload comparison."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from .report import latest_per_library
from .results import load, publication_issues

# Build is intentionally absent: these different kinds of tools materialise
# different data and indices, so their construction times are not one workload.
TOPICS: list[tuple[str, list[str]]] = [
    ("Node/edge scans and ID lookup", ["node_scan", "point_lookup", "edge_scan"]),
    ("Property filters and aggregation", ["property_filter", "range_filter", "group_aggregation", "year_aggregation"]),
    (
        "Reachability and traversal",
        ["one_hop", "two_hop", "three_hop", "filtered_traversal", "deep_traversal", "score_filtered_traversal"],
    ),
    ("Shortest-path queries", ["shortest_path"]),
    ("Typed graph joins and aggregation", ["pattern_match", "industry_aggregation", "two_step_join"]),
    ("Degree and connected-components operations", ["degree_topk", "connected_components", "degree_filter"]),
    ("Louvain community detection", ["louvain"]),
    ("Updates and create/delete batch", ["bulk_update", "mutations"]),
    ("Exact vector scoring", ["vector_knn"]),
    ("Latitude/longitude bounding-box filter", ["geo_within"]),
]
MODE_TOPICS = [("Graph construction", ["build"]), *TOPICS]

HEADLINE_KGLITE = "kglite-cypher"
KGLITE_MODES = [
    "kglite-cypher",
    "kglite-mapped",
    "kglite-disk",
    "kglite-fluent",
    "kglite-bolt",
    "kglite-bolt-docker",
]
LABELS = {
    "kglite-cypher": "kglite",
    "kglite-mapped": "kglite (mapped)",
    "kglite-disk": "kglite (disk)",
    "kglite-fluent": "kglite (fluent)",
    "kglite-bolt": "kglite (Bolt)",
    "kglite-bolt-docker": "kglite (Bolt, Docker)",
    "ladybug": "LadybugDB",
    "kuzu": "Kùzu (historical)",
    "neo4j": "Neo4j (external)",
    "neo4j-docker": "Neo4j (Docker)",
    "neo4j-native": "Neo4j (native)",
    "duckdb": "DuckDB",
    "networkx": "NetworkX",
    "rustworkx": "rustworkx",
    "igraph": "igraph",
}
CATEGORY = {
    "kglite-cypher": "Cypher graph engine, in-memory",
    "ladybug": "Cypher graph database, embedded",
    "kuzu": "archived Cypher graph database, embedded",
    "neo4j": "Cypher graph database, separately managed server",
    "neo4j-docker": "Cypher graph database, Docker server",
    "neo4j-native": "Cypher graph database, native server",
    "duckdb": "SQL/relational database",
    "networkx": "pure-Python graph library",
    "rustworkx": "Rust-backed graph library",
    "igraph": "C-backed graph library",
}

# This is a display threshold, not an execution timeout. The measured time is
# retained in its topic sum and rendered rather than discarded.
SLOW_S = 10.0


def _fmt(seconds: float | None, ran: int = 0, total: int = 0, slow: bool = False) -> str:
    if seconds is None:
        return "not exercised"
    if seconds < 1e-3:
        text = f"{seconds * 1e6:.0f}µs"
    elif seconds < 1.0:
        text = f"{seconds * 1e3:.1f}ms"
    else:
        text = f"{seconds:.2f}s"
    if ran < total:
        text += f" ({ran}/{total} exercised)"
    if slow:
        text += " (measured >10s)"
    return text


def _topic_seconds(run: dict[str, Any], group_ids: list[str]) -> tuple[float | None, int, int, bool]:
    values = [
        group["min_s"]
        for gid in group_ids
        if (group := run.get("groups", {}).get(gid, {})).get("status") == "ok" and group.get("min_s") is not None
    ]
    if not values:
        return None, 0, len(group_ids), False
    return sum(values), len(values), len(group_ids), any(value > SLOW_S for value in values)


def _qualified_capture(data: dict[str, Any], signature: str | None) -> dict[str, dict[str, Any]]:
    captures: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in data.get("runs", []):
        provenance = run.get("provenance", {})
        capture_id = provenance.get("capture_id")
        if capture_id and provenance.get("publication_qualified") is True:
            captures[capture_id].append(run)

    eligible: list[list[dict[str, Any]]] = []
    for runs in captures.values():
        requested = runs[0].get("provenance", {}).get("requested_libraries")
        if not isinstance(requested, list) or publication_issues(runs, requested):
            continue
        signatures = {run.get("dataset", {}).get("signature") for run in runs}
        if signature is None or signatures == {signature}:
            eligible.append(runs)
    if not eligible:
        return {}
    selected = max(eligible, key=lambda runs: max(run["run_date"] for run in runs))
    return {run["library"]: run for run in selected}


def publication_runs(data: dict[str, Any], signature: str | None = None) -> tuple[dict[str, dict[str, Any]], bool]:
    """Select one qualified capture, or explicitly fall back to legacy rows."""
    qualified = _qualified_capture(data, signature)
    if qualified:
        return qualified, True

    legacy = latest_per_library(data, signature)
    if signature is None and legacy:
        signatures = [run["dataset"]["signature"] for run in legacy.values()]
        signature = max(set(signatures), key=lambda item: (signatures.count(item), item))
        legacy = latest_per_library(data, signature)
    return legacy, False


def _table(out: list[str], latest: dict[str, dict[str, Any]], cols: list[str], topics) -> None:
    out.append("| Workload topic | " + " | ".join(LABELS.get(col, col) for col in cols) + " |")
    out.append("|" + "---|" * (len(cols) + 1))
    for topic, group_ids in topics:
        result = {col: _topic_seconds(latest[col], group_ids) for col in cols}
        complete = {
            col: seconds
            for col, (seconds, ran, total, slow) in result.items()
            if seconds is not None and ran == total and not slow
        }
        winner = min(complete, key=complete.get) if complete else None
        cells = []
        for col in cols:
            cell = _fmt(*result[col])
            if col == winner:
                cell = f"**{cell}**"
            cells.append(cell)
        out.append(f"| {topic} | " + " | ".join(cells) + " |")


def render(signature: str | None = None) -> str:
    data = load()
    latest, qualified = publication_runs(data, signature)
    if not latest:
        return "_no benchmark runs recorded yet — run `python benchmarks/benchmark.py`_\n"

    sample = next(iter(latest.values()))
    dataset = sample["dataset"]
    competitors = sorted(lib for lib in latest if not lib.startswith("kglite"))
    cols = ([HEADLINE_KGLITE] if HEADLINE_KGLITE in latest else []) + competitors

    out = ["# KGLite benchmarks", ""]
    if not qualified:
        out.extend(
            [
                "> **Unqualified historical snapshot.** These rows predate publication qualification. "
                "They may combine different invocations or dirty source revisions and must not be described "
                "as a current capture.",
                "",
            ]
        )
    out.extend(
        [
            "Minimum observed wall time for the workloads each adapter exercised on one seed-deterministic "
            "synthetic graph. "
            "Lower is better within a row; `not exercised` is not a claim about the underlying product's capabilities.",
            "",
            f"**Dataset:** {dataset['n_nodes']:,} nodes · {dataset['n_edges']:,} edges "
            f"(Person/Company/Project/Skill/City), scale `{dataset['scale']}`.",
            "",
        ]
    )
    _table(out, latest, cols, TOPICS)
    out.extend(
        [
            "",
            "Bold marks the fastest complete, non-slow measurement in that row. Partial cells sum only the "
            "named executed groups; missing work is not estimated. Values above 10 seconds remain in the "
            "measurement and are labelled, not treated as timeouts. "
            "There is deliberately no grand total across unequal workload coverage.",
            "",
            "Adapters receive the same generated input records and shared query parameters, but use different "
            "data models and idiomatic APIs. Their operations are corresponding workloads, not necessarily "
            "identical query text or identical execution semantics. The detailed parity report records result "
            "differences instead of treating every cross-library difference as a failure.",
            "",
            "### Workloads exercised by these adapters",
            "",
            "This is coverage of the recorded adapter implementations, not a product capability matrix.",
            "",
            "| Workload topic | " + " | ".join(LABELS.get(col, col) for col in cols) + " |",
            "|" + "---|" * (len(cols) + 1),
        ]
    )
    for topic, group_ids in TOPICS:
        cells = []
        for col in cols:
            _seconds, ran, total, _slow = _topic_seconds(latest[col], group_ids)
            cells.append("not exercised" if ran == 0 else ("✓" if ran == total else f"{ran}/{total} exercised"))
        out.append(f"| {topic} | " + " | ".join(cells) + " |")

    out.extend(["", "### What's being compared", ""])
    for col in cols:
        if category := CATEGORY.get(col):
            out.append(f"- **{LABELS.get(col, col)}** — {category}")
    out.extend(
        [
            "",
            "Graph-construction time is omitted from the cross-kind headline because these adapters materialise "
            "different internal representations and indices. Compare workload rows individually and consult "
            "the raw per-group report for detail.",
            "",
        ]
    )

    modes = [mode for mode in KGLITE_MODES if mode in latest]
    if len(modes) > 1:
        out.extend(["### kglite storage modes and protocols", ""])
        _table(out, latest, modes, MODE_TOPICS)
        out.append("")

    out.extend(
        [
            "### Scaling",
            "",
            f"The headline uses the `{dataset['scale']}` graph selected above. "
            "For the separate historical load-and-first-query study of disk-backed modes, see "
            "[`benchmarks/competitive/largescale/`](benchmarks/competitive/largescale/README.md).",
            "",
            "## Reproduce",
            "",
            "```bash",
            "uv pip install --python .venv/bin/python ladybug networkx rustworkx python-igraph duckdb neo4j",
            "uv run --no-sync maturin develop --release",
            "make build-bolt-server",
            ".venv/bin/python benchmarks/benchmark.py",
            "```",
            "",
            "The public command publishes only a clean, complete, error-free invocation containing every "
            "requested adapter. See [`graphsuite/README.md`]"
            "(benchmarks/competitive/graphsuite/README.md) for opt-in native and Docker server profiles.",
            "",
            "### Measurement protocol",
            "",
            "Each query group records the minimum observed wall time from an adaptive 1–5 repetitions; mutation "
            "groups use at most two. Graph construction uses one measurement, or two for inexpensive non-Bolt "
            "builds. There is no warm-up phase. "
            "These labels describe this harness exactly; they are not confidence intervals or enforced timeouts.",
            "",
            "Versions used for the selected rows:",
            "",
            "_" + ", ".join(f"{LABELS.get(col, col)} {latest[col]['version']}" for col in cols) + "_",
            "",
        ]
    )
    machine = sample.get("machine")
    if isinstance(machine, dict):
        out.append(f"Run on {machine.get('platform', 'an unspecified machine')} · Python {machine.get('python', '?')}.")
    else:
        out.append(f"Run on {machine or 'an unspecified machine'}.")
    out.extend(["", "### Capture provenance", ""])
    out.append(f"- Publication qualification: `{'qualified' if qualified else 'legacy / unqualified'}`")
    out.append(f"- Results schema: `{data.get('schema_version', 'unknown')}`")
    harness = data.get("harness", {})
    out.append(f"- Results file writer: `{harness.get('name', 'graphsuite')}` v{harness.get('version', 'unknown')}")
    out.append(f"- Dataset signature: `{dataset['signature']}`")
    dates = sorted(run["run_date"] for run in latest.values())
    out.append(f"- Selected run timestamps: `{dates[0]}` through `{dates[-1]}`")
    provenances = [run.get("provenance") for run in latest.values() if isinstance(run.get("provenance"), dict)]
    if provenances:
        capture_harnesses = sorted({str(p.get("harness_version", "not recorded")) for p in provenances})
        capture_ids = sorted({p.get("capture_id", "not recorded") for p in provenances})
        commits = sorted({p.get("source_commit", "not recorded") for p in provenances})
        dirty = sorted({str(p.get("source_dirty", "not recorded")).lower() for p in provenances})
        repeats = sorted({str(p.get("base_repeats", "not recorded")) for p in provenances})
        out.append(f"- Selected capture harness: `v{', v'.join(capture_harnesses)}`")
        out.append(f"- Capture id: `{', '.join(capture_ids)}`")
        out.append(f"- Source commit: `{', '.join(commits)}` (dirty: `{', '.join(dirty)}`)")
        out.append(f"- Base repeat policy: `{', '.join(repeats)}`")
    else:
        out.append("- Selected capture harness, id, and source revision: `not recorded`")
    out.append(f"- Dataset seed: `{dataset.get('seed', 'recorded in signature')}`")
    out.append("- Raw metadata authority: `benchmarks/competitive/graphsuite/results.json`.")
    out.append("")
    return "\n".join(out)


if __name__ == "__main__":
    print(render())
