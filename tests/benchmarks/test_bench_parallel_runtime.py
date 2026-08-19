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


# ── Grouped aggregation (materialized path) ─────────────────────────────────
#
# `collect` / `median` / `percentile_cont` are declined by the streaming
# aggregate, which is what routes these to the materialized grouping path —
# the one Q4 partitions. Cardinality is the axis that matters: the grouping
# pass merges one partial map per partition, so many groups means a bigger
# merge, and few groups means the across-group evaluation has few tasks.

AGG_LOW_CARD = "MATCH (p:Person) RETURN p.joined_year AS y, collect(p.age) AS ages"
AGG_MID_CARD = "MATCH (p:Person) RETURN p.city AS c, collect(p.age) AS ages"
AGG_HIGH_CARD = "MATCH (p:Person) RETURN p.name AS nm, collect(p.age) AS ages"
AGG_PERCENTILE = "MATCH (p:Person) RETURN p.joined_year AS y, median(p.age) AS m, percentile_cont(p.age, 0.9) AS p90"


@pytest.mark.benchmark
@pytest.mark.parametrize("parallel", [False, True], ids=["serial", "parallel"])
@pytest.mark.parametrize(
    ("label", "query"),
    [
        ("low_card", AGG_LOW_CARD),
        ("mid_card", AGG_MID_CARD),
        ("high_card", AGG_HIGH_CARD),
        ("percentile", AGG_PERCENTILE),
    ],
)
def test_bench_parallel_grouped_aggregation(benchmark, parallel_graph, label, query, parallel):
    benchmark(parallel_graph.cypher, query, parallel=parallel)


# ── ORDER BY sort-key precompute, and the regex-cache contention probe ──────

ORDER_BY_SORT = "MATCH (p:Person) RETURN p.name AS nm ORDER BY p.age DESC, p.name ASC"

# R7 probe: `=~` resolves its compiled pattern through a process-global
# RwLock-guarded cache on *every row*. `CONTAINS` does the same per-row work
# without the lock, so the gap between these two speedups is the contention.
REGEX_PREDICATE = "MATCH (p:Person) WHERE p.name =~ '.*a1.*' RETURN count(*) AS n"
CONTAINS_PREDICATE = "MATCH (p:Person) WHERE p.name CONTAINS 'a1' RETURN count(*) AS n"


@pytest.mark.benchmark
@pytest.mark.parametrize("parallel", [False, True], ids=["serial", "parallel"])
def test_bench_parallel_order_by_sort(benchmark, parallel_graph, parallel):
    """800k rows sorted. Only the sort-key precompute can fan out; the sort
    itself is stable and stays sequential."""
    benchmark(parallel_graph.cypher, ORDER_BY_SORT, parallel=parallel)


@pytest.mark.benchmark
@pytest.mark.parametrize("parallel", [False, True], ids=["serial", "parallel"])
def test_bench_parallel_regex_predicate(benchmark, parallel_graph, parallel):
    benchmark(parallel_graph.cypher, REGEX_PREDICATE, parallel=parallel)


@pytest.mark.benchmark
@pytest.mark.parametrize("parallel", [False, True], ids=["serial", "parallel"])
def test_bench_parallel_contains_predicate(benchmark, parallel_graph, parallel):
    """Control for the regex cell: same shape, no global cache lookup."""
    benchmark(parallel_graph.cypher, CONTAINS_PREDICATE, parallel=parallel)


@pytest.mark.benchmark
def test_bench_throughput_control_with_a_parallel_query_admitted(benchmark, parallel_graph):
    """The composition cell: the same eight concurrent readers as the control
    above, but with one `parallel=True` aggregate running alongside them.

    This is the honesty meter for the program's claim — "one heavy analytical
    query can use the machine; concurrent-client throughput is unchanged". The
    two halves of that only compose if admitting a parallel query does not
    wreck the readers, and a parallel query by definition takes cores the
    readers were using. There is no gate here: the number is recorded so the
    claim can be stated with its interference cost attached rather than
    without.
    """
    query = "MATCH (p:Person) WHERE p.score > 0.9 RETURN count(*) AS n"
    heavy = "MATCH (p:Person) WHERE p.score > 0.5 RETURN p.joined_year AS y, count(*) AS n"

    def readers_alongside_one_parallel_query():
        hog = threading.Thread(target=parallel_graph.cypher, args=(heavy,), kwargs={"parallel": True})
        hog.start()
        threads = [threading.Thread(target=parallel_graph.cypher, args=(query,)) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        hog.join()

    benchmark(readers_alongside_one_parallel_query)
