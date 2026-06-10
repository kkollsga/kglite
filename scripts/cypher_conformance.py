#!/usr/bin/env python3
"""On-demand openCypher conformance check vs Neo4j.

Runs every query in ``tests/test_cypher_differential.py::DIFFERENTIAL_QUERIES``
against KGLite *and* a Neo4j instance and reports row-level divergences.
The Neo4j side is populated from the same fixture builder KGLite uses,
via the existing ``kglite.to_neo4j()`` export, so both engines see
identical input data.

This script is deliberately **not** wired into the pytest suite or CI:

  - No external service dependency for the regular test run.
  - No per-PR latency / flakiness from Docker / network.
  - The user opts in when they want a correctness oracle (e.g. when
    investigating a NULL-semantics or aggregate question).

Workflow:

  1. ``make neo4j-up`` — starts a fresh Neo4j 5 container on bolt://localhost:7687.
  2. ``make neo4j-conformance`` — runs this script. Output is a summary
     plus per-failure detail (query text, KGLite rows, Neo4j rows, diff).
  3. ``make neo4j-down`` — tears the container down.

Deliberate divergences (places where KGLite ships a different
behaviour than Neo4j *on purpose*) go in ``INTENTIONAL_DIVERGENCES``
below — each entry carries a one-line rationale.

Exit code: 0 if all unregistered divergences are absent; 1 otherwise.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tests"))

# Reuse the test fixtures and corpus — single source of truth for both
# the conformance check and the per-PR differential gate.
from conftest import build_small_graph, build_social_graph  # type: ignore  # noqa: E402
from test_cypher_differential import DIFFERENTIAL_QUERIES  # type: ignore  # noqa: E402

import kglite  # noqa: E402

# Map fixture-name → builder. The corpus references fixtures by name;
# this dict resolves them.
FIXTURE_BUILDERS = {
    "small_graph": build_small_graph,
    "social_graph": build_social_graph,
}


# ──────────────────────────────────────────────────────────────────────
# Intentional divergences
#
# Each entry: query name → one-line rationale. These are queries where
# KGLite ships a behaviour that deliberately differs from Neo4j —
# either because the spec ambiguous and KGLite's choice is documented
# (e.g. `int / int → int` per 0.9.0 §5), or because KGLite has a
# narrower domain. Entries should be the exception, not the rule —
# anything not in this list is expected to row-match Neo4j.
#
# When adding a divergence:
#   1. Investigate whether it's a real KGLite bug. Most "Neo4j differs"
#      findings are bugs to fix, not divergences to register.
#   2. If the divergence is intentional, link to the
#      CHANGELOG entry or doc that explains it.
# ──────────────────────────────────────────────────────────────────────

INTENTIONAL_DIVERGENCES: dict[str, str] = {
    # No entries yet. Populate as the first conformance run surfaces
    # spec-defensible KGLite choices (e.g. division-by-zero → NULL
    # instead of Neo4j's Infinity/NaN — see CHANGELOG 0.9.52).
}


# ──────────────────────────────────────────────────────────────────────
# Query filter — some corpus queries call KGLite-only procs / functions
# that have no Neo4j equivalent. Skip them rather than failing.
# ──────────────────────────────────────────────────────────────────────

KGLITE_ONLY_MARKERS = (
    "kglite.",  # `CALL kglite.affected_tests(...)` etc.
    "refresh_stats",  # KGLite-only procedure
    "text_score",  # KGLite-only semantic search function
    "vector_score",  # KGLite-only vector similarity function
    "FORMAT CSV",  # KGLite-only result-set materialiser
)


def _is_kglite_only(query: str) -> bool:
    return any(marker in query for marker in KGLITE_ONLY_MARKERS)


# ──────────────────────────────────────────────────────────────────────
# Result normalisation. Both engines need to produce the same row set
# modulo ordering (unless the query has ORDER BY, in which case order
# matters). We normalise by sorting rows by their repr.
# ──────────────────────────────────────────────────────────────────────


def _canonical(value) -> str:
    """Recursively canonicalise a value for comparison. Map key order is
    not part of Cypher map semantics (the direct path emits BTreeMap-
    sorted keys while Bolt PackStream preserves a different order), so
    nested dicts render with sorted keys. List order IS semantic and is
    preserved. Scalars stringify via repr to dodge type-mismatch noise
    (e.g. Neo4j's `int` vs KGLite's `int64`)."""
    if isinstance(value, dict):
        return "{" + ", ".join(f"{k!r}: {_canonical(value[k])}" for k in sorted(value)) + "}"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_canonical(v) for v in value) + "]"
    return repr(value)


def _normalise_row(row: dict) -> tuple:
    """Stable, hashable representation of a single row. Keys sorted;
    values canonicalised recursively (see `_canonical`)."""
    return tuple((k, _canonical(row[k])) for k in sorted(row.keys()))


def _normalise(rows: list[dict], order_sensitive: bool) -> list[tuple]:
    out = [_normalise_row(r) for r in rows]
    return out if order_sensitive else sorted(out)


def _is_order_sensitive(query: str) -> bool:
    """Treat ORDER BY as the signal that row order is part of the
    contract. LIMIT without ORDER BY is non-deterministic by spec, so
    we still sort and compare as sets."""
    return "ORDER BY" in query.upper()


# ──────────────────────────────────────────────────────────────────────
# Neo4j-side query runner.
# ──────────────────────────────────────────────────────────────────────


def _run_neo4j(driver, query: str, params: dict | None) -> list[dict]:
    with driver.session() as session:
        result = session.run(query, **(params or {}))
        # Convert Neo4j Records to plain dicts.
        return [dict(record) for record in result]


# ──────────────────────────────────────────────────────────────────────
# Main loop.
# ──────────────────────────────────────────────────────────────────────


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--uri", default="bolt://localhost:7687", help="Neo4j bolt URI.")
    p.add_argument("--user", default="neo4j", help="Neo4j username.")
    p.add_argument("--password", default="conformance", help="Neo4j password.")
    p.add_argument("--database", default="neo4j", help="Neo4j database name.")
    p.add_argument("--filter", help="Only run queries whose name contains this substring.")
    p.add_argument("--verbose", action="store_true", help="Print every query result, not just failures.")
    args = p.parse_args()

    try:
        from neo4j import GraphDatabase
    except ImportError:
        print("This script requires the neo4j Python driver. Install with:", file=sys.stderr)
        print("    pip install -e '.[neo4j]'", file=sys.stderr)
        return 2

    print(f"connecting to {args.uri} as {args.user}…")
    driver = GraphDatabase.driver(args.uri, auth=(args.user, args.password))

    # Per-fixture KGLite cache. Each fixture is built once and pushed
    # to Neo4j once (with clear=True) — subsequent queries reuse both.
    kglite_graphs: dict[str, "kglite.KnowledgeGraph"] = {}
    pushed_to_neo4j: set[str] = set()

    stats = {"pass": 0, "fail": 0, "skip_kglite_only": 0, "skip_divergence": 0, "error": 0}
    failures: list[tuple[str, str, list[dict], list[dict]]] = []

    try:
        for entry in DIFFERENTIAL_QUERIES:
            name, fixture, query, params = entry

            if args.filter and args.filter not in name:
                continue

            if name in INTENTIONAL_DIVERGENCES:
                print(f"  SKIP   {name:<40}  (intentional: {INTENTIONAL_DIVERGENCES[name]})")
                stats["skip_divergence"] += 1
                continue

            if _is_kglite_only(query):
                print(f"  SKIP   {name:<40}  (KGLite-only feature)")
                stats["skip_kglite_only"] += 1
                continue

            builder = FIXTURE_BUILDERS.get(fixture)
            if builder is None:
                print(f"  SKIP   {name:<40}  (no builder for fixture '{fixture}')")
                stats["skip_kglite_only"] += 1
                continue

            # Build KGLite graph + push to Neo4j (once per fixture).
            if fixture not in kglite_graphs:
                kglite_graphs[fixture] = builder()
            kg = kglite_graphs[fixture]

            if fixture not in pushed_to_neo4j:
                # `clear=True` wipes Neo4j first so each fixture is
                # exported into a clean target.
                kglite.to_neo4j(
                    kg, args.uri, auth=(args.user, args.password), database=args.database, clear=True, verbose=False
                )
                pushed_to_neo4j.add(fixture)

            try:
                kg_rows = kg.cypher(query, params=params).to_list()
                neo4j_rows = _run_neo4j(driver, query, params)
            except Exception as e:
                print(f"  ERROR  {name:<40}  {type(e).__name__}: {str(e)[:80]}")
                stats["error"] += 1
                continue

            order_sensitive = _is_order_sensitive(query)
            kg_norm = _normalise(kg_rows, order_sensitive)
            neo4j_norm = _normalise(neo4j_rows, order_sensitive)

            if kg_norm == neo4j_norm:
                stats["pass"] += 1
                if args.verbose:
                    print(f"  PASS   {name:<40}  ({len(kg_rows)} rows)")
            else:
                stats["fail"] += 1
                print(f"  FAIL   {name:<40}  kglite={len(kg_rows)} rows, neo4j={len(neo4j_rows)} rows")
                failures.append((name, query, kg_rows, neo4j_rows))

    finally:
        driver.close()

    # Summary.
    total = sum(stats.values())
    print()
    print(
        f"summary: {total} queries — "
        f"{stats['pass']} pass, "
        f"{stats['fail']} fail, "
        f"{stats['skip_kglite_only']} skipped (kglite-only), "
        f"{stats['skip_divergence']} skipped (intentional divergence), "
        f"{stats['error']} errors."
    )

    if failures:
        print()
        print("=" * 72)
        print("FAILURES")
        print("=" * 72)
        for name, query, kg_rows, neo4j_rows in failures:
            print(f"\n--- {name} ---")
            print(f"  query: {query}")
            print(f"  kglite rows ({len(kg_rows)}):  {kg_rows[:5]}{' …' if len(kg_rows) > 5 else ''}")
            print(f"  neo4j rows  ({len(neo4j_rows)}): {neo4j_rows[:5]}{' …' if len(neo4j_rows) > 5 else ''}")
            kg_set = {repr(_normalise_row(r)) for r in kg_rows}
            neo_set = {repr(_normalise_row(r)) for r in neo4j_rows}
            only_kg = kg_set - neo_set
            only_neo = neo_set - kg_set
            if only_kg:
                print(f"  only in kglite ({len(only_kg)}): {sorted(only_kg)[:3]}")
            if only_neo:
                print(f"  only in neo4j  ({len(only_neo)}): {sorted(only_neo)[:3]}")

    return 0 if (stats["fail"] == 0 and stats["error"] == 0) else 1


if __name__ == "__main__":
    sys.exit(main())
