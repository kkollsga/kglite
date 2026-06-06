#!/usr/bin/env python3
"""On-demand Bolt wire-protocol conformance check.

Runs every query in ``tests/test_cypher_differential.py::DIFFERENTIAL_QUERIES``
through ``kglite-bolt-server`` (over the neo4j Python driver) and compares
each result set against direct-Rust ``KnowledgeGraph.cypher()`` on the same
fixture. Direct-Rust is the oracle: any divergence is a PackStream / wire
round-trip bug in the Bolt server, since both sides run the *same* engine.

This complements ``scripts/cypher_conformance.py`` (which diffs KGLite vs
Neo4j as the spec oracle). Here Neo4j is not involved at all — no Docker,
no external service. We spawn our own ``kglite-bolt-server`` on an
ephemeral port, reusing the exact launch path the test suite trusts
(``tests/conftest.py``), and the only oracle is KGLite-in-process.

This script is deliberately **not** wired into CI (same discipline as
``scripts/cypher_conformance.py``): it's an on-demand correctness oracle.

Workflow:

  1. Build the release binary: ``cargo build -p kglite-bolt-server --release``
     (or ``make build-bolt-server``).
  2. ``make bolt-conformance`` — runs this script and prints a summary plus
     per-failure detail (query, direct rows, bolt rows, diff).

Exit code: 0 if every query round-trips identically; 1 on any divergence
or error; 2 if the binary or the neo4j driver is missing.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import tempfile

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tests"))

# Reuse the corpus, the fixture builders, the bolt-server launch helpers,
# and the diff machinery — single source of truth shared with the test
# suite and the Neo4j conformance runner.
from conftest import (  # type: ignore  # noqa: E402
    _BOLT_BINARY,
    _spawn_bolt_server,
    _teardown_bolt_server,
)
from cypher_conformance import (  # type: ignore  # noqa: E402
    FIXTURE_BUILDERS,
    _is_kglite_only,
    _is_order_sensitive,
    _normalise,
    _normalise_row,
)
from test_cypher_differential import DIFFERENTIAL_QUERIES  # type: ignore  # noqa: E402


def _run_bolt(driver, query: str, params: dict | None) -> list[dict]:
    """Run a query over the Bolt wire and return rows as plain dicts."""
    with driver.session() as session:
        result = session.run(query, **(params or {}))
        return [dict(record) for record in result]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--filter", help="Only run queries whose name contains this substring.")
    p.add_argument("--verbose", action="store_true", help="Print every query result, not just failures.")
    args = p.parse_args()

    if not _BOLT_BINARY.exists():
        print(f"kglite-bolt-server binary not found at {_BOLT_BINARY}", file=sys.stderr)
        print("Build it with: cargo build -p kglite-bolt-server --release", file=sys.stderr)
        return 2

    try:
        from neo4j import GraphDatabase
    except ImportError:
        print("This script requires the neo4j Python driver. Install with:", file=sys.stderr)
        print("    pip install -e '.[neo4j]'", file=sys.stderr)
        return 2

    # One bolt server + one in-process oracle graph per fixture. Built
    # lazily and torn down at the end. Corpus queries are read-only, so a
    # single graph/server pair is reused across all queries of a fixture.
    servers: dict[str, tuple] = {}  # fixture -> (proc, url)
    oracle_graphs: dict[str, object] = {}  # fixture -> KnowledgeGraph
    tmpdir = tempfile.TemporaryDirectory(prefix="kglite-bolt-conformance-")

    stats = {"pass": 0, "fail": 0, "skip_kglite_only": 0, "skip_no_builder": 0, "error": 0}
    failures: list[tuple[str, str, list[dict], list[dict]]] = []

    print(f"spawning kglite-bolt-server from {_BOLT_BINARY}…")
    try:
        for entry in DIFFERENTIAL_QUERIES:
            name, fixture, query, params = entry

            if args.filter and args.filter not in name:
                continue

            if _is_kglite_only(query):
                print(f"  SKIP   {name:<40}  (KGLite-only feature)")
                stats["skip_kglite_only"] += 1
                continue

            builder = FIXTURE_BUILDERS.get(fixture)
            if builder is None:
                print(f"  SKIP   {name:<40}  (no builder for fixture '{fixture}')")
                stats["skip_no_builder"] += 1
                continue

            # Build the oracle graph + spawn a bolt server for this
            # fixture on first use.
            if fixture not in oracle_graphs:
                kg = builder()
                oracle_graphs[fixture] = kg
                kgl_path = Path(tmpdir.name) / f"{fixture}.kgl"
                kg.save(str(kgl_path))
                servers[fixture] = _spawn_bolt_server(kgl_path)
            kg = oracle_graphs[fixture]
            _proc, url = servers[fixture]

            try:
                direct_rows = kg.cypher(query, params=params).to_list()
                with GraphDatabase.driver(url, auth=("neo4j", "password")) as driver:
                    bolt_rows = _run_bolt(driver, query, params)
            except Exception as e:
                print(f"  ERROR  {name:<40}  {type(e).__name__}: {str(e)[:80]}")
                stats["error"] += 1
                continue

            order_sensitive = _is_order_sensitive(query)
            direct_norm = _normalise(direct_rows, order_sensitive)
            bolt_norm = _normalise(bolt_rows, order_sensitive)

            if direct_norm == bolt_norm:
                stats["pass"] += 1
                if args.verbose:
                    print(f"  PASS   {name:<40}  ({len(direct_rows)} rows)")
            else:
                stats["fail"] += 1
                print(f"  FAIL   {name:<40}  direct={len(direct_rows)} rows, bolt={len(bolt_rows)} rows")
                failures.append((name, query, direct_rows, bolt_rows))

    finally:
        for proc, _url in servers.values():
            _teardown_bolt_server(proc)
        tmpdir.cleanup()

    # Summary.
    total = sum(stats.values())
    print()
    print(
        f"summary: {total} queries — "
        f"{stats['pass']} pass, "
        f"{stats['fail']} fail, "
        f"{stats['skip_kglite_only']} skipped (kglite-only), "
        f"{stats['skip_no_builder']} skipped (no builder), "
        f"{stats['error']} errors."
    )

    if failures:
        print()
        print("=" * 72)
        print("FAILURES (bolt wire round-trip diverged from direct .cypher())")
        print("=" * 72)
        for name, query, direct_rows, bolt_rows in failures:
            print(f"\n--- {name} ---")
            print(f"  query: {query}")
            print(f"  direct rows ({len(direct_rows)}): {direct_rows[:5]}{' …' if len(direct_rows) > 5 else ''}")
            print(f"  bolt rows   ({len(bolt_rows)}): {bolt_rows[:5]}{' …' if len(bolt_rows) > 5 else ''}")
            direct_set = {repr(_normalise_row(r)) for r in direct_rows}
            bolt_set = {repr(_normalise_row(r)) for r in bolt_rows}
            only_direct = direct_set - bolt_set
            only_bolt = bolt_set - direct_set
            if only_direct:
                print(f"  only in direct ({len(only_direct)}): {sorted(only_direct)[:3]}")
            if only_bolt:
                print(f"  only in bolt   ({len(only_bolt)}): {sorted(only_bolt)[:3]}")

    return 0 if (stats["fail"] == 0 and stats["error"] == 0) else 1


if __name__ == "__main__":
    sys.exit(main())
