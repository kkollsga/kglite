"""Differential test: every query in `DIFFERENTIAL_QUERIES` runs both
via direct `KnowledgeGraph.cypher()` AND via the Bolt wire; results
must match.

This is the **strongest correctness gate** for the bolt-server: the
27-query corpus in `tests/test_cypher_differential.py` already exercises
every optimizer pass + historical correctness-bug shapes. Running each
over Bolt confirms that the Bolt projection path (Phase C.4 to_bolt,
Phase A.1 Value::Node) doesn't silently drop, duplicate, or reorder
rows.

When the row sets diverge, the failure shows which query exposed the
divergence — investigate whether (a) the Cypher engine produced
different results on the two paths, (b) `to_bolt` mangled a value, or
(c) the test's normalization is wrong.

Marker: `bolt` (default-excluded; opt-in via `pytest -m bolt`).
"""

from __future__ import annotations

import pytest

# Reuse the corpus + the fixture builders the pyapi differential test uses.
from tests.conftest import (
    _BOLT_BINARY,
    _spawn_bolt_server,
    _teardown_bolt_server,
    build_small_graph,
    build_social_graph,
)
from tests.test_cypher_differential import DIFFERENTIAL_QUERIES

neo4j = pytest.importorskip("neo4j")

pytestmark = [pytest.mark.bolt]


# ─── Helpers ───────────────────────────────────────────────────────────────


def _normalize_rows(rows: list) -> list:
    """Normalize a row list for comparison. The corpus's queries don't
    all pin ordering, so we sort by repr(row) for a deterministic
    comparison. Cell values are converted to plain Python types where
    needed (driver returns neo4j.time.* for temporal; we don't have
    those in the corpus today)."""
    return sorted([tuple(row.values()) if hasattr(row, "values") else tuple(row) for row in rows])


def _direct_run(query: str, params: dict | None, fixture_name: str) -> list:
    """Run via direct kglite Python; return a normalized row list."""
    import kglite  # noqa: F401  -- ensures the extension is loaded

    if fixture_name == "small_graph":
        graph = build_small_graph()
    elif fixture_name == "social_graph":
        graph = build_social_graph()
    else:
        pytest.skip(f"differential corpus uses unknown fixture: {fixture_name}")
    result = graph.cypher(query, params=params)
    # ResultView → list of tuples via iteration.
    rows = []
    for row in result:
        # row is a dict-like or list-like; normalize to a tuple of values
        if hasattr(row, "values"):
            rows.append(tuple(row.values()))
        else:
            rows.append(tuple(row))
    return sorted(rows)


def _bolt_run(url: str, query: str, params: dict | None) -> list:
    """Run via the Bolt wire; return a normalized row list."""
    with neo4j.GraphDatabase.driver(url, auth=("neo4j", "password")) as driver:
        with driver.session() as session:
            if params is None:
                result = session.run(query)
            else:
                result = session.run(query, **params)
            rows = [tuple(record.values()) for record in result]
    return sorted(rows)


# ─── Fixture pools (one bolt-server per fixture name; session-scoped) ──────
#
# Spawning a server per test is too expensive — the corpus has 27 queries
# split between small_graph and social_graph. We spin two servers at module
# scope and reuse them.


@pytest.fixture(scope="module")
def _small_graph_bolt_url(tmp_path_factory):
    if not _BOLT_BINARY.exists():
        pytest.skip(f"bolt-server binary not built at {_BOLT_BINARY}")
    tmp = tmp_path_factory.mktemp("differential_small")
    fixture = tmp / "small.kgl"
    g = build_small_graph()
    g.save(str(fixture))
    proc, url = _spawn_bolt_server(fixture)
    yield url
    _teardown_bolt_server(proc)


@pytest.fixture(scope="module")
def _social_graph_bolt_url(tmp_path_factory):
    if not _BOLT_BINARY.exists():
        pytest.skip(f"bolt-server binary not built at {_BOLT_BINARY}")
    tmp = tmp_path_factory.mktemp("differential_social")
    fixture = tmp / "social.kgl"
    g = build_social_graph()
    g.save(str(fixture))
    proc, url = _spawn_bolt_server(fixture)
    yield url
    _teardown_bolt_server(proc)


# ─── Parametrized test ────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "name,fixture,query,params",
    DIFFERENTIAL_QUERIES,
    ids=[entry[0] for entry in DIFFERENTIAL_QUERIES],
)
def test_differential_bolt_matches_direct(name, fixture, query, params, _small_graph_bolt_url, _social_graph_bolt_url):
    """Each corpus query: direct cypher() vs Bolt session.run() must
    return the same normalized row set."""
    # Skip queries whose fixture isn't one we built bolt-servers for.
    if fixture == "small_graph":
        url = _small_graph_bolt_url
    elif fixture == "social_graph":
        url = _social_graph_bolt_url
    else:
        pytest.skip(f"corpus query {name!r} uses unsupported fixture {fixture!r}")

    direct = _direct_run(query, params, fixture)
    bolt = _bolt_run(url, query, params)

    assert direct == bolt, (
        f"differential mismatch on {name!r} ({fixture}):\n"
        f"  query:  {query}\n"
        f"  params: {params}\n"
        f"  direct: {direct[:10]}{'...' if len(direct) > 10 else ''} "
        f"({len(direct)} rows)\n"
        f"  bolt:   {bolt[:10]}{'...' if len(bolt) > 10 else ''} "
        f"({len(bolt)} rows)"
    )
