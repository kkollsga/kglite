"""Program benchmarks for the opt-in parallel Cypher runtime.

Deliberately **not** in ``test_bench_core.py``: the Linux perf gate runs that
file with ``--require-exact-set``, so a win-meter cell added there fails CI
until a release promotes the baseline. These cells are win meters, not gates —
nothing here is wired into ``tests/benchmarks/baselines/``.

Run with::

    uv run --no-sync maturin develop --release
    .venv/bin/python -m pytest tests/benchmarks/test_bench_parallel_runtime.py \
        -m benchmark --benchmark-min-rounds=... -s

The fixture is a ~1M-node / ~11M-edge graphgen graph and costs roughly 30-60 s
to build once per session; every cell shares it.
"""

import threading

import pytest

import kglite

PERSONS = 800_000

# Scan-bound aggregate over the whole Person set. The predicate and the
# aggregate both compile to column routes, so this is the `Compiled` side of
# the runtime cost-class gate.
SCAN_AGG_COUNT = "MATCH (p:Person) WHERE p.score > 0.5 RETURN count(*) AS n"

# Same scan, grouped on a low-cardinality column: adds per-partition group maps
# and a merge, which is what the partitioned design has to pay for.
SCAN_AGG_GROUPED = "MATCH (p:Person) WHERE p.score > 0.5 RETURN p.joined_year AS y, count(*) AS n, avg(p.age) AS a"


# Scan + filter. The first filters in the *candidate scan* (an inline property
# map on the pattern is what the planner rewrites a scannable equality to); the
# second keeps the predicate in the fused operator; the third adds a per-row
# projection on top, which is the allocation-bound half.
SCAN_FILTER_PROPERTY = "MATCH (p:Person {{joined_year: {year}}}) RETURN count(*) AS n"
SCAN_FILTER_WHERE = "MATCH (p:Person) WHERE p.name CONTAINS 'a1' RETURN count(*) AS n"
SCAN_FILTER_PROJECT = "MATCH (p:Person) WHERE p.score > 0.99 RETURN p.name AS nm, p.age AS a"


@pytest.fixture(scope="session")
def joined_year(parallel_graph):
    """A `joined_year` the fixture actually carries, so the property scan is
    not measuring an empty result."""
    rows = parallel_graph.cypher(
        "MATCH (p:Person) RETURN p.joined_year AS y, count(*) AS n ORDER BY n DESC LIMIT 1"
    ).to_list()
    return rows[0]["y"]


@pytest.fixture(scope="session")
def parallel_graph():
    """~1M nodes / ~11M edges. Session-scoped: the build dominates otherwise."""
    return kglite.graphgen(persons=PERSONS, seed=1234)


@pytest.mark.benchmark
@pytest.mark.parametrize("parallel", [False, True], ids=["serial", "parallel"])
def test_bench_parallel_scan_agg_count(benchmark, parallel_graph, parallel):
    """Ungrouped scan + filter + count — the shape the stop rule is measured on."""
    benchmark(parallel_graph.cypher, SCAN_AGG_COUNT, parallel=parallel)


@pytest.mark.benchmark
@pytest.mark.parametrize("parallel", [False, True], ids=["serial", "parallel"])
def test_bench_parallel_scan_agg_grouped(benchmark, parallel_graph, parallel):
    """Grouped aggregate — pays the per-partition map + deterministic merge."""
    benchmark(parallel_graph.cypher, SCAN_AGG_GROUPED, parallel=parallel)


@pytest.mark.benchmark
def test_bench_across_query_throughput_control(benchmark, parallel_graph):
    """Eight concurrent read clients, every one of them ``parallel=False``.

    The composition control for the whole program: the claim is "one heavy
    analytical query can use the machine; concurrent-client throughput is
    unchanged". That is two claims, and they only compose if this cell stays
    flat while the parallel cells move. It exists from Q2 so later phases have
    a baseline to compare against rather than a number invented at the end.
    """
    query = "MATCH (p:Person) WHERE p.score > 0.9 RETURN count(*) AS n"

    def eight_concurrent_readers():
        threads = [threading.Thread(target=parallel_graph.cypher, args=(query,)) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

    benchmark(eight_concurrent_readers)


@pytest.mark.benchmark
@pytest.mark.parametrize("parallel", [False, True], ids=["serial", "parallel"])
def test_bench_parallel_scan_filter_property(benchmark, parallel_graph, joined_year, parallel):
    """Filtering happens inside the partitioned candidate scan."""
    query = SCAN_FILTER_PROPERTY.format(year=joined_year)
    benchmark(parallel_graph.cypher, query, parallel=parallel)


@pytest.mark.benchmark
@pytest.mark.parametrize("parallel", [False, True], ids=["serial", "parallel"])
def test_bench_parallel_scan_filter_where(benchmark, parallel_graph, parallel):
    """Interpreted text predicate — the expensive-per-row cost class."""
    benchmark(parallel_graph.cypher, SCAN_FILTER_WHERE, parallel=parallel)


@pytest.mark.benchmark
@pytest.mark.parametrize("parallel", [False, True], ids=["serial", "parallel"])
def test_bench_parallel_scan_filter_project(benchmark, parallel_graph, parallel):
    """Scan + filter + per-row projection. The projection half allocates a
    bindings map per surviving row, so this cell is where the scan's win and
    the row loop's allocation cost are visible together."""
    benchmark(parallel_graph.cypher, SCAN_FILTER_PROJECT, parallel=parallel)
