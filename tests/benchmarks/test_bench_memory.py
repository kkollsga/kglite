"""Benchmarks for memory management: spill, unspill, vacuum, and save.

Compares fully heap-resident columns against columns spilled to disk under
`set_memory_limit`. There is no shape to switch: properties live in per-type
columns from the first node, and the only axis here is where those columns'
bytes sit.
Run with: pytest tests/benchmarks/test_bench_memory.py -m benchmark -v -s
"""

import subprocess
import sys
import textwrap

import pandas as pd
import pytest

from kglite import KnowledgeGraph

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _build_graph(n=5000):
    """Build a graph with n nodes and multiple property types."""
    graph = KnowledgeGraph()
    nodes = pd.DataFrame(
        {
            "nid": list(range(n)),
            "name": [f"Node_{i}" for i in range(n)],
            "value": [float(i) for i in range(n)],
            "category": [f"cat_{i % 50}" for i in range(n)],
            "score": [float(i * 0.1) for i in range(n)],
            "flag": [i % 2 == 0 for i in range(n)],
        }
    )
    graph.add_nodes(nodes, "Item", "nid", "name")

    edges = pd.DataFrame(
        {
            "from_id": [i % n for i in range(n * 2)],
            "to_id": [(i * 7 + 13) % n for i in range(n * 2)],
            "weight": [float(i % 100) for i in range(n * 2)],
        }
    )
    graph.add_connections(edges, "LINKS", "Item", "from_id", "Item", "to_id", columns=["weight"])
    return graph


@pytest.fixture
def graph_5k():
    """5000-node graph, columns heap-resident.

    Was two fixtures — `graph_5k` ("compact storage") and `graph_5k`
    ("columnar, heap-backed") — with identical bodies, because the shapes they
    named were never distinguishable by anything a fixture could do. One now.
    """
    return _build_graph(5000)


@pytest.fixture
def bench_graph_1k():
    """1000-node graph — the shape the retired tracked cell measured."""
    return _build_graph(1000)


@pytest.fixture
def graph_5k_spilled(tmp_path):
    """5000-node graph (columnar, spilled to disk).

    A `save()` is the enforcement point that applies the limit; there is no
    regime switch to call.
    """
    g = _build_graph(5000)
    g.set_memory_limit(1024)  # force full spill
    g.save(str(tmp_path / "spill-trigger.kgl"))
    return g


# ---------------------------------------------------------------------------
# Unspill
# ---------------------------------------------------------------------------


@pytest.mark.benchmark
def test_bench_unspill_5k(benchmark, graph_5k_spilled, tmp_path):
    """Time to move spilled data back to heap (5000 nodes).

    The re-spill runs in `setup`, not in the measured body: it is a `save()`
    now, so folding it into the timed call would report file I/O as unspill
    cost.
    """

    def setup():
        graph_5k_spilled.set_memory_limit(1024)
        graph_5k_spilled.save(str(tmp_path / "respill.kgl"))

    benchmark.pedantic(graph_5k_spilled.unspill, setup=setup, rounds=20, iterations=1)


# ---------------------------------------------------------------------------
# Query performance: heap vs spilled
# ---------------------------------------------------------------------------


@pytest.mark.benchmark
def test_bench_query_where_heap_5k(benchmark, graph_5k):
    """Filtered query on heap-resident columns (5000 nodes)."""
    benchmark(
        graph_5k.cypher,
        "MATCH (n:Item) WHERE n.value > 4000 RETURN n.title, n.value",
    )


@pytest.mark.benchmark
def test_bench_query_where_spilled_5k(benchmark, graph_5k_spilled):
    """Filtered query on spilled columns (5000 nodes)."""
    benchmark(
        graph_5k_spilled.cypher,
        "MATCH (n:Item) WHERE n.value > 4000 RETURN n.title, n.value",
    )


@pytest.mark.benchmark
def test_bench_query_match_heap_5k(benchmark, graph_5k):
    """Simple MATCH on heap-resident columns (5000 nodes)."""
    benchmark(
        graph_5k.cypher,
        "MATCH (n:Item) RETURN n.title, n.value LIMIT 100",
    )


@pytest.mark.benchmark
def test_bench_query_match_spilled_5k(benchmark, graph_5k_spilled):
    """Simple MATCH on spilled columns (5000 nodes)."""
    benchmark(
        graph_5k_spilled.cypher,
        "MATCH (n:Item) RETURN n.title, n.value LIMIT 100",
    )


@pytest.mark.benchmark
def test_bench_query_aggregation_heap_5k(benchmark, graph_5k):
    """Aggregation on heap-resident columns."""
    benchmark(
        graph_5k.cypher,
        "MATCH (n:Item) RETURN count(n) AS cnt, avg(n.value) AS avg_val",
    )


@pytest.mark.benchmark
def test_bench_query_aggregation_spilled_5k(benchmark, graph_5k_spilled):
    """Aggregation on spilled columns."""
    benchmark(
        graph_5k_spilled.cypher,
        "MATCH (n:Item) RETURN count(n) AS cnt, avg(n.value) AS avg_val",
    )


# ---------------------------------------------------------------------------
# Vacuum and consolidation
# ---------------------------------------------------------------------------


@pytest.mark.benchmark
def test_bench_vacuum_5k(benchmark):
    """Vacuum after deleting 60% of nodes.

    Was a `columnar` / `no_columnar` pair whose bodies were character-identical
    — the "baseline comparison" arm never built a different graph, so the two
    cells reported the same operation under two names.
    """

    def run():
        g = _build_graph(5000)
        g.set_auto_vacuum(None)
        g.cypher("MATCH (n:Item) WHERE n.value < 3000 DETACH DELETE n")
        g.vacuum()

    benchmark(run)


@pytest.mark.benchmark
def test_bench_unspill_rebuild_1k(benchmark, bench_graph_1k):
    """One full consolidation pass over a heap-resident graph's columns.

    `unspill()` is the public route to the rebuild `save()` and `vacuum()` also
    run. The cell lived in `test_bench_core.py` as `test_bench_columnar_enable`
    and timed a `disable_columnar()` / `enable_columnar()` round trip; both are
    gone and the operation is not the same one, so the tracked cell was retired
    rather than renamed over an anchor value that measured something else. It
    is untracked here until a release capture can baseline it on both
    platforms.
    """
    benchmark(bench_graph_1k.unspill)


# ---------------------------------------------------------------------------
# Save: heap vs spilled
# ---------------------------------------------------------------------------


@pytest.mark.benchmark
def test_bench_save_kgl_heap_5k(benchmark, graph_5k, tmp_path):
    """Save a `.kgl` from heap-resident columns."""
    counter = [0]

    def save():
        graph_5k.save(str(tmp_path / f"save_{counter[0]}.kgl"))
        counter[0] += 1

    benchmark(save)


@pytest.mark.benchmark
def test_bench_save_kgl_spilled_5k(benchmark, graph_5k_spilled, tmp_path):
    """Save a `.kgl` from spilled columns."""
    counter = [0]

    def save():
        graph_5k_spilled.save(str(tmp_path / f"save_{counter[0]}.kgl"))
        counter[0] += 1

    benchmark(save)


# ---------------------------------------------------------------------------
# Variable-length k-hop peak memory (part-6 program, phase V1)
# ---------------------------------------------------------------------------
#
# Not a benchmark: this is a LAW. A multi-seed k-hop query's intermediate rows
# are `seeds x |reachable|`, so a row-shaped expansion's peak grows with the
# seed count even though the *answer* -- the union of the reachable sets --
# barely moves. That is the +3 GB the part-6 eval reported, fully attributed
# (~300 B/row, no leak).
#
# V4 made the expansion deduplicate targets globally for a distinct-only
# consumer, so those pair-shaped rows are never built. V4b threaded the same
# seen-set across the separate expansions `UNWIND` drives, so both spellings of
# one reachability answer cost one peak.
#
# # The meter (rewritten 2026-08-21)
#
# The original probe read `resource.getrusage().ru_maxrss` around the query.
# That is a process **high-water mark**, and the fixture build's own transient
# (Python edge lists -> DataFrame -> engine) sets it far above where the
# process settles afterwards. A query allocating less than the build's
# overshoot therefore moves it by *nothing*: on the CI Linux runners the probe
# read exactly 0.0 MB and all three tests failed on their own floor guard,
# correctly refusing to divide by it. macOS reported a few MB for the same
# reason -- the residue of the same contaminated instrument, not a measurement.
#
# The probe now samples the process's **current footprint**
# (`ri_phys_footprint` on macOS, `/proc/self/statm` resident on Linux -- the
# pair `test_trim_memory` uses) from a watcher thread while the query runs, and
# reports `max(sample) - baseline`. `cypher()` releases the GIL, so the thread
# samples freely: ~150 samples over a 0.2 s query, and the probe fails if it
# collected too few to have seen a peak. The baseline is taken after
# `gc.collect()` + `kglite.trim_memory()` + a settle sleep, because until the
# allocator has finished purging the build's pages the window carries +28/-20 MB
# excursions that belong to the fixture, not the query.
#
# # The scale
#
# The fixture is 200 000 nodes, up from 30 000. At 30 000 the post-V4 peak is
# ~4 MB, a signal of the same order as a single allocator step; at 200 000 it is
# ~28 MB, far above page- or arena-granularity on either platform, and the
# fixture still builds in ~1.6 s. The point of the bigger fixture is that no
# accounting coarse enough to exist can round a 28 MB delta to zero.
#
# # What the numbers are (debug extension, macOS arm64, 2026-08-21)
#
#                          |  25 seeds |  50 seeds | ratio
#   in_list, today         |   28.3 MB |   28.7 MB | 1.02x
#   unwind,  today         |   28.2 MB |   28.2 MB | 1.00x
#   in_list, row-shaped    |  173.5 MB |  243.9 MB | 1.41x
#   unwind,  row-shaped    |  125.2 MB |  165.0 MB | 1.32x
#
# The row-shaped rows are *measured*, not remembered: the same query with a
# non-distinct consumer bolted on (`RETURN count(DISTINCT f), count(f)`)
# returns the same `reached` answer but disqualifies the distinct-only dedup,
# so it materializes exactly the pair rows V4 removed. Three consecutive runs
# of every "today" cell agreed to within 0.1%.
#
# That table is also why `MAX_PEAK_DELTA_MB`, not `MAX_SEED_GROWTH_FACTOR`, is
# the detector for a lost dedup: read on a meter that is not dominated by the
# build, the row-shaped analogue grows only 1.32-1.41x with the seeds and would
# pass an 1.8x ceiling. (The 1.96x that originally justified 1.8 came off the
# high-water meter, which was reporting the build's overshoot as much as the
# query's peak.) The ratio stays, because it is the law's statement and a
# regression that allocates *per seed* still trips it -- but it is the absolute
# ceiling that separates 28 MB from 125 MB.

#: Nodes in the probe fixture. Sized so the post-V4 peak is ~28 MB: see "The
#: scale" above.
KHOP_NODE_COUNT = 200_000

#: 2x the seeds may cost at most this much more peak. The union of reachable
#: sets grows ~1.27x between the two sizes and today's peak grows 1.00-1.02x,
#: so 1.8x is clear of any legitimate verdict; it catches a regression whose
#: memory is proportional to the seed count.
MAX_SEED_GROWTH_FACTOR = 1.8

#: Absolute ceiling on a single peak measurement, in MB. Today's is 28-29 MB on
#: either spelling and either seed count; the row-shaped analogue measures
#: 125-244 MB. 90 sits ~3.1x above the former and ~1.4x below the lowest of the
#: latter, so neither verdict is close to it.
MAX_PEAK_DELTA_MB = 90.0

#: A measurement below this is not a measurement, and no ratio may be formed
#: from it. Was 1.0 MB against a ~4 MB signal; the fixture is now sized for a
#: ~28 MB one, so the floor rises with it. This is the guard that caught the
#: high-water meter reading 0.0 on Linux -- it stays.
MIN_MEASURABLE_PEAK_MB = 10.0

_KHOP_PEAK_PROBE = textwrap.dedent(
    """
    import ctypes, ctypes.util, gc, os, struct, sys, threading, time
    import pandas as pd
    import kglite

    if sys.platform == "darwin":
        _libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
        _buf = ctypes.create_string_buffer(512)

        def footprint_mb():
            # proc_pid_rusage(pid, RUSAGE_INFO_V0, &buf); rusage_info_v0 is a
            # 16-byte uuid followed by uint64s and ri_phys_footprint is the 8th.
            if _libc.proc_pid_rusage(ctypes.c_int(os.getpid()), ctypes.c_int(0), ctypes.byref(_buf)) != 0:
                raise OSError(ctypes.get_errno(), "proc_pid_rusage failed")
            return struct.unpack_from("=8Q", _buf.raw, 16)[7] / (1024 * 1024)
    else:
        _page = os.sysconf("SC_PAGE_SIZE")

        def footprint_mb():
            with open("/proc/self/statm", encoding="ascii") as handle:
                return int(handle.read().split()[1]) * _page / (1024 * 1024)

    node_count, seed_count, attachments = %(node_count)d, int(sys.argv[1]), 2
    spelling = sys.argv[2]

    repeated = list(range(attachments))
    src, dst = [], []
    state = 20_260_821
    for new in range(attachments, node_count):
        chosen = []
        while len(chosen) < attachments:
            state = (state * 1_103_515_245 + 12_345) & 0x7FFF_FFFF
            candidate = repeated[state %% len(repeated)]
            if candidate not in chosen:
                chosen.append(candidate)
        for target in chosen:
            src.append(new)
            dst.append(target)
            repeated.append(target)
        repeated.extend([new] * attachments)
    src, dst = src + dst, dst + src

    graph = kglite.KnowledgeGraph()
    graph.add_nodes(
        pd.DataFrame({"pid": list(range(node_count)), "name": [f"P{i}" for i in range(node_count)]}),
        "Person", "pid", "name",
    )
    graph.add_connections(pd.DataFrame({"s": src, "d": dst}), "KNOWS", "Person", "s", "Person", "d")

    ids = [(i * 197 + 13) %% node_count for i in range(seed_count)]
    graph.cypher("MATCH (p:Person) RETURN count(p) AS n").to_list()

    query = {
        # The WHERE-IN spelling drives one expansion whose source rows share a
        # seen-set; the UNWIND spelling drives one expansion per seed, so the
        # set has to be shared across them. Same answer, same fixture.
        "in_list": "MATCH (p:Person)-[:KNOWS*1..3]->(f:Person) WHERE p.id IN $ids "
        "RETURN count(DISTINCT f) AS reached",
        "unwind": "UNWIND $ids AS i MATCH (p:Person {id: i})-[:KNOWS*1..3]->(f:Person) "
        "RETURN count(DISTINCT f) AS reached",
    }[spelling]

    # Hand the fixture's transient back before measuring: the allocator is
    # still purging it, and those excursions dwarf the query.
    del src, dst, repeated
    gc.collect()
    kglite.trim_memory()
    time.sleep(0.4)

    samples = []
    stop = threading.Event()

    def watch():
        while not stop.is_set():
            samples.append(footprint_mb())
            time.sleep(0.001)

    baseline = footprint_mb()
    watcher = threading.Thread(target=watch, daemon=True)
    watcher.start()
    rows = graph.cypher(query, params={"ids": ids}).to_list()
    stop.set()
    watcher.join()

    print(max(samples) - baseline, rows[0]["reached"], len(samples))
    """
    % {"node_count": KHOP_NODE_COUNT}
)

#: Fewer samples than this and the watcher never saw the query: the reported
#: maximum would understate the peak, in the reassuring direction.
_MIN_PROBE_SAMPLES = 20

requires_footprint = pytest.mark.skipif(
    sys.platform != "darwin" and not sys.platform.startswith("linux"),
    reason="the peak probe reads ri_phys_footprint (macOS) or /proc/self/statm (Linux)",
)


def _khop_peak_delta_mb(seed_count: int, spelling: str) -> tuple[float, int]:
    """Peak-footprint delta (MB) and reached-node count for one seed count.

    A fresh subprocess per size, for the reason `test_unwind_scope_narrowing`
    documents: run both in one process and the allocator's retained arenas from
    the first size absorb the second, understating the growth -- a failure in
    the reassuring direction.
    """
    proc = subprocess.run(
        [sys.executable, "-c", _KHOP_PEAK_PROBE, str(seed_count), spelling],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, (
        f"{spelling} probe at {seed_count} seeds failed (rc={proc.returncode})\nstderr:\n{proc.stderr}"
    )
    delta, reached, samples = proc.stdout.strip().splitlines()[-1].split()
    assert int(samples) >= _MIN_PROBE_SAMPLES, (
        f"{spelling} probe at {seed_count} seeds collected {samples} footprint "
        f"samples (floor {_MIN_PROBE_SAMPLES}); the watcher thread did not run "
        "for the query, so its maximum is not the peak."
    )
    return float(delta), int(reached)


def _assert_seed_growth_law(
    spelling: str,
    small_peak: float,
    small_reached: int,
    large_peak: float,
    large_reached: int,
) -> None:
    """The seed-growth law, over already-measured numbers.

    Split from the measurement so the thresholds can be exercised on synthetic
    inputs -- `test_the_peak_memory_law_rejects_a_regressed_measurement` feeds
    it a zeroed probe and the measured row-shaped numbers, which is the only
    red-first available for a defect whose engine no longer exists.
    """
    # The law is only meaningful while the reachable sets stay comparable: if
    # 2x the seeds really did reach 2x the nodes, 2x the peak would be correct.
    target_growth = large_reached / small_reached
    assert target_growth < 1.5, (
        f"fixture drift: 50 seeds reach {target_growth:.2f}x the nodes 25 do "
        f"({small_reached} -> {large_reached}). The law compares seed growth "
        "against target growth and needs them to differ."
    )
    # Guard against a small side so close to zero that any large side passes.
    assert small_peak >= MIN_MEASURABLE_PEAK_MB, (
        f"[{spelling}] peak delta at 25 seeds was {small_peak:.1f} MB, under the "
        f"{MIN_MEASURABLE_PEAK_MB} MB floor -- too small to form a ratio from. "
        "Today's fixture measures ~28 MB there; a near-zero reading means the "
        "probe stopped measuring, not that the query stopped allocating."
    )

    for seeds, peak in ((25, small_peak), (50, large_peak)):
        assert peak <= MAX_PEAK_DELTA_MB, (
            f"[{spelling}] {seeds} seeds peaked {peak:.1f} MB above baseline "
            f"(ceiling {MAX_PEAK_DELTA_MB} MB). Today this shape costs ~28 MB; "
            "materializing one row per (seed, target) pair costs 125-244 MB."
        )

    seed_growth = large_peak / small_peak
    assert seed_growth <= MAX_SEED_GROWTH_FACTOR, (
        f"[{spelling}] 2x the seeds cost {seed_growth:.2f}x the peak "
        f"({small_peak:.1f} MB -> {large_peak:.1f} MB) while reaching only "
        f"{target_growth:.2f}x the nodes. Peak is following the seed count, "
        f"not the reachable set (ceiling {MAX_SEED_GROWTH_FACTOR}x)."
    )


def _assert_khop3_peak_follows_targets(spelling: str) -> None:
    """Doubling the seeds must not double the peak, for one query spelling.

    The seeds only decide how many BFS roots there are; the answer is the union
    of what they reach, which grows far more slowly. An implementation whose
    memory follows the *targets* satisfies this; one whose memory follows the
    *rows* does not.
    """
    small_peak, small_reached = _khop_peak_delta_mb(25, spelling)
    large_peak, large_reached = _khop_peak_delta_mb(50, spelling)
    _assert_seed_growth_law(spelling, small_peak, small_reached, large_peak, large_reached)


@requires_footprint
def test_khop3_peak_memory_scales_with_targets_not_seeds():
    """The WHERE-IN spelling: one expansion, its source rows share a seen-set."""
    _assert_khop3_peak_follows_targets("in_list")


@requires_footprint
def test_khop3_unwind_peak_memory_scales_with_targets_not_seeds():
    """The UNWIND spelling of the same law -- one expansion per driving row.

    Its twin above passes on a build where this fails: the seen-set the
    matcher shares across the source rows of one expansion does not, by
    itself, span the separate expansions UNWIND drives. Both spellings answer
    `count(DISTINCT f)` over the same union, so both peaks must follow the
    union.
    """
    _assert_khop3_peak_follows_targets("unwind")


#: How much more peak the UNWIND spelling may cost than the WHERE-IN one.
#: Measured on this fixture: 1.00x at 25 seeds and 0.98x at 50. The row-shaped
#: analogue of a lost driving-row dedup costs 125-165 MB against the WHERE-IN
#: spelling's 28 MB, i.e. 4.4-5.7x, so 1.35x is far from either verdict.
MAX_SPELLING_PEAK_FACTOR = 1.35


def _assert_spelling_parity(
    seeds: int,
    unwind_peak: float,
    unwind_reached: int,
    in_list_peak: float,
    in_list_reached: int,
) -> None:
    """The two-spelling parity check, over already-measured numbers."""
    # Not a tautology: a pair that stopped computing the same answer is a
    # ratio between two unrelated numbers.
    assert unwind_reached == in_list_reached, (
        f"the spellings disagree at {seeds} seeds: UNWIND reached {unwind_reached}, WHERE-IN reached {in_list_reached}"
    )
    assert in_list_peak >= MIN_MEASURABLE_PEAK_MB, (
        f"WHERE-IN peak at {seeds} seeds was {in_list_peak:.1f} MB, under the "
        f"{MIN_MEASURABLE_PEAK_MB} MB floor -- too small to divide by."
    )
    assert unwind_peak <= MAX_PEAK_DELTA_MB, (
        f"the UNWIND spelling peaked {unwind_peak:.1f} MB above baseline at "
        f"{seeds} seeds (ceiling {MAX_PEAK_DELTA_MB} MB); today it costs ~28 MB."
    )

    factor = unwind_peak / in_list_peak
    assert factor <= MAX_SPELLING_PEAK_FACTOR, (
        f"at {seeds} seeds the UNWIND spelling cost {factor:.2f}x the "
        f"WHERE-IN spelling's peak ({unwind_peak:.1f} MB vs "
        f"{in_list_peak:.1f} MB) for the same {unwind_reached}-node answer "
        f"(ceiling {MAX_SPELLING_PEAK_FACTOR}x). The driving-row branch is "
        "building one row per (seed, target) pair."
    )


@requires_footprint
def test_khop3_unwind_peak_memory_matches_the_where_in_spelling():
    """The two spellings of one reachability answer must cost one peak.

    A self-controlling pair, the memory counterpart of the `khop3_*` benchmark
    cells: same graph, same seeds, same `count(DISTINCT f)` -- so any gap is
    the branch difference and nothing else. This is the sharp detector for the
    driving-row dedup; the seed-growth law above reads a lost dedup as only
    1.32x on this fixture, which its own ceiling would let through.
    """
    for seeds in (25, 50):
        unwind_peak, unwind_reached = _khop_peak_delta_mb(seeds, "unwind")
        in_list_peak, in_list_reached = _khop_peak_delta_mb(seeds, "in_list")
        _assert_spelling_parity(seeds, unwind_peak, unwind_reached, in_list_peak, in_list_reached)


def test_the_peak_memory_law_rejects_a_regressed_measurement():
    """Red-first for the two laws above, on numbers instead of on an engine.

    The defect they guard was fixed in V4/V4b and its engine cannot be rebuilt
    cheaply, so the thresholds are exercised directly: a probe that measured
    nothing, and the row-shaped costs measured today by re-running the same
    query with a non-distinct consumer (see the table above). Without this,
    "the memory law passes" would only mean "the numbers it was handed passed".
    """
    # A probe that measures nothing must not produce a verdict. This is the
    # exact reading CI got from the old high-water meter.
    with pytest.raises(AssertionError, match="too small to form a ratio"):
        _assert_seed_growth_law("in_list", 0.0, 65297, 0.0, 83006)
    with pytest.raises(AssertionError, match="too small to divide by"):
        _assert_spelling_parity(25, 0.0, 65297, 0.0, 65297)

    # The measured row-shaped costs: caught by the absolute ceiling on both
    # spellings, at both seed counts.
    with pytest.raises(AssertionError, match="ceiling 90.0 MB"):
        _assert_seed_growth_law("in_list", 173.5, 65297, 243.9, 83006)
    with pytest.raises(AssertionError, match="ceiling 90.0 MB"):
        _assert_seed_growth_law("unwind", 125.2, 65297, 165.0, 83006)
    with pytest.raises(AssertionError, match="ceiling 90.0 MB"):
        _assert_spelling_parity(25, 125.2, 65297, 28.3, 65297)
    # A partial loss of the driving-row dedup, under the absolute ceiling: the
    # spelling gap is what catches it.
    with pytest.raises(AssertionError, match="ceiling 1.35x"):
        _assert_spelling_parity(25, 60.0, 65297, 28.3, 65297)

    # A regression that allocates per seed rather than per target is what the
    # ratio is for; it trips even while both sides sit under the ceiling.
    with pytest.raises(AssertionError, match="Peak is following the seed count"):
        _assert_seed_growth_law("in_list", 28.3, 65297, 56.6, 83006)

    # A pair that stopped answering the same question is not a ratio.
    with pytest.raises(AssertionError, match="the spellings disagree"):
        _assert_spelling_parity(25, 28.2, 64000, 28.3, 65297)
    # Neither is a fixture whose two seed counts stopped reaching a comparable
    # union.
    with pytest.raises(AssertionError, match="fixture drift"):
        _assert_seed_growth_law("in_list", 28.3, 65297, 28.7, 130000)

    # Today's measurements pass both.
    _assert_seed_growth_law("in_list", 28.3, 65297, 28.7, 83006)
    _assert_seed_growth_law("unwind", 28.2, 65297, 28.2, 83006)
    _assert_spelling_parity(50, 28.2, 83006, 28.7, 83006)
