"""Bolt overhead + throughput characterization benchmarks.

These quantify the Bolt-server wire tax on top of the direct kglite
`cypher()` call. Numbers go into `docs/operators/bolt-server.md` and
become a regression gate once a release captures baselines.

Run with: `pytest tests/benchmarks/test_bench_bolt.py -m benchmark -v`

No baselines captured here (the bolt-server hasn't shipped a release
yet). When it does, `make refresh-release-constants` picks these up.

Fixtures: spawn a single bolt-server per module to amortize the
~50 ms boot cost across all benchmarks.
"""

from __future__ import annotations

import pytest

from tests.conftest import (
    _BOLT_BINARY,
    _spawn_bolt_server,
    _teardown_bolt_server,
)

neo4j = pytest.importorskip("neo4j")

pytestmark = [pytest.mark.benchmark, pytest.mark.bolt]


# ───────────────────────────────────────────────────────────────────────────
# Module-scoped fixtures: one server, one driver, one large pre-built graph
# ───────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def _bench_server(tmp_path_factory):
    """Spin a bolt-server on a 10k-node fixture; yield (url, kg)."""
    if not _BOLT_BINARY.exists():
        pytest.skip(f"bolt-server binary not built at {_BOLT_BINARY}")
    import pandas as pd

    import kglite

    tmp = tmp_path_factory.mktemp("bolt_bench")
    fixture_path = tmp / "bench.kgl"

    # 10k Person + 30k KNOWS — matches the B.3 baseline shape so
    # bolt overhead can be compared against the existing
    # test_bench_return_node_10k / test_bench_return_node_rel_node_100.
    n = 10_000
    g = kglite.KnowledgeGraph()
    nodes = pd.DataFrame(
        {
            "pid": list(range(n)),
            "name": [f"P{i}" for i in range(n)],
            "age": [20 + (i % 60) for i in range(n)],
            "city": [f"city_{i % 100}" for i in range(n)],
        }
    )
    g.add_nodes(nodes, "Person", "pid", "name")
    edges = pd.DataFrame(
        {
            "s": [i % n for i in range(3 * n)],
            "d": [(i * 13 + 7) % n for i in range(3 * n)],
        }
    )
    g.add_connections(edges, "KNOWS", "Person", "s", "Person", "d")
    g.save(str(fixture_path))

    proc, url = _spawn_bolt_server(fixture_path)
    yield url, g
    _teardown_bolt_server(proc)


@pytest.fixture(scope="module")
def _bench_driver(_bench_server):
    url, _ = _bench_server
    driver = neo4j.GraphDatabase.driver(url, auth=("neo4j", "password"))
    yield driver
    driver.close()


# ───────────────────────────────────────────────────────────────────────────
# Benchmarks
# ───────────────────────────────────────────────────────────────────────────


def test_bench_bolt_connect_and_run(benchmark, _bench_server):
    """Cost to open a fresh driver session, run one trivial query,
    close. This includes the full handshake + LOGON + first-query
    overhead. Worst-case latency for a 'one shot' bolt client."""
    url, _ = _bench_server

    def round_trip():
        with neo4j.GraphDatabase.driver(url, auth=("neo4j", "password")) as driver:
            with driver.session() as session:
                session.run("RETURN 1 AS x").consume()

    benchmark(round_trip)


def test_bench_bolt_run_overhead_vs_direct(benchmark, _bench_driver):
    """Cost of running a small read-only query over an already-open
    Bolt session. Reuses the driver — the measured cost is just one
    RUN/PULL round trip + result materialization on the driver side.

    Compare against `test_bench_return_node_rel_node_100` (the direct
    pyapi version) to derive the Bolt wire tax in microseconds."""
    session = _bench_driver.session()
    try:

        def run():
            session.run("MATCH (n:Person) RETURN n.name AS name LIMIT 100").consume()

        benchmark(run)
    finally:
        session.close()


def test_bench_bolt_pull_10k_scalars(benchmark, _bench_driver):
    """Cost of streaming 10k scalar rows (RUN+PULL one big result).
    Tests boltr's PULL pagination + driver-side materialization at
    a realistic workload."""
    session = _bench_driver.session()
    try:

        def run():
            list(session.run("MATCH (n:Person) RETURN n.name AS name ORDER BY n.pid"))

        benchmark(run)
    finally:
        session.close()


def test_bench_bolt_pull_10k_nodes(benchmark, _bench_driver):
    """Cost of streaming 10k Node structs (Phase A.1 Value::Node →
    Phase C.4 BoltNode encoding). Compare against
    `test_bench_return_node_10k` for the wire tax on the full Node
    projection path."""
    session = _bench_driver.session()
    try:

        def run():
            list(session.run("MATCH (n:Person) RETURN n ORDER BY n.pid"))

        benchmark(run)
    finally:
        session.close()


def test_bench_bolt_tx_commit_no_writes(benchmark, _bench_driver):
    """Cost of BEGIN + commit() on a tx that performed no writes —
    measures the per-tx Arc snapshot + handle minting + cleanup. The
    'no-write' branch in commit() should be essentially free."""
    session = _bench_driver.session()
    try:

        def run():
            tx = session.begin_transaction()
            tx.commit()

        benchmark(run)
    finally:
        session.close()


def test_bench_bolt_tx_commit_with_100_writes(benchmark, _bench_driver):
    """Cost of BEGIN + 100 CREATEs + COMMIT. Includes:
    - Arc::try_unwrap to materialize the working DirGraph
    - 100 execute_mutable calls
    - Arc swap into the shared backend.graph on commit
    """
    session = _bench_driver.session()
    counter = [0]
    try:

        def run():
            tx = session.begin_transaction()
            for i in range(100):
                # Use a per-bench-iteration offset so we don't trip
                # any uniqueness constraints.
                nid = 100_000_000 + counter[0] * 1000 + i
                tx.run(f"CREATE (:Person {{id: {nid}, title: 't{nid}'}})").consume()
            tx.commit()
            counter[0] += 1

        benchmark.pedantic(run, rounds=10, warmup_rounds=2, iterations=1)
    finally:
        session.close()
