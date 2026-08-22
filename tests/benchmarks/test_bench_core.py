"""Core benchmarks using pytest-benchmark for historical tracking.

These benchmarks measure the key operations and are tracked over time.
Run with: make bench-save (to save a baseline) or make bench-compare (to compare).
"""

import inspect
import sys
from typing import NamedTuple

import pandas as pd
import pytest

import kglite
from kglite import KnowledgeGraph

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def bench_graph():
    """Graph with 1000 nodes and 2000 edges for benchmarking."""
    graph = KnowledgeGraph()

    nodes = pd.DataFrame(
        {
            "nid": list(range(1000)),
            "name": [f"Node_{i}" for i in range(1000)],
            "value": [float(i) for i in range(1000)],
            "category": [f"cat_{i % 10}" for i in range(1000)],
        }
    )
    graph.add_nodes(nodes, "Item", "nid", "name")

    edges = pd.DataFrame(
        {
            "from_id": [i % 1000 for i in range(2000)],
            "to_id": [(i * 7 + 13) % 1000 for i in range(2000)],
            "weight": [float(i % 100) for i in range(2000)],
        }
    )
    graph.add_connections(edges, "LINKS", "Item", "from_id", "Item", "to_id", columns=["weight"])

    return graph


@pytest.fixture(scope="module")
def grouped_count_graph():
    """10k+10k nodes and 30k edges for grouped-count top-k regressions.

    Both endpoints intentionally repeat their grouping property across many
    nodes. This keeps the benchmark honest: the fast path must aggregate by
    the resolved property value, not by node identity.
    """
    graph = KnowledgeGraph()
    n = 10_000
    graph.add_nodes(
        pd.DataFrame(
            {
                "sid": list(range(n)),
                "name": [f"Source_{i}" for i in range(n)],
                "bucket": [f"source_bucket_{i % 100}" for i in range(n)],
            }
        ),
        "Source",
        "sid",
        "name",
    )
    graph.add_nodes(
        pd.DataFrame(
            {
                "gid": list(range(n)),
                "name": [f"Group_{i}" for i in range(n)],
                "bucket": [f"target_bucket_{i % 100}" for i in range(n)],
            }
        ),
        "Group",
        "gid",
        "name",
    )
    graph.add_connections(
        pd.DataFrame(
            {
                "source": [i % n for i in range(3 * n)],
                "target": [(i * 13 + (i // n) * 997 + 7) % n for i in range(3 * n)],
                "tag": [f"Edge_{i}" for i in range(3 * n)],
            }
        ),
        "RELATES_TO",
        "Source",
        "source",
        "Group",
        "target",
        columns=["tag"],
    )
    return graph


@pytest.fixture(scope="module")
def indexed_node_scan_graph():
    """100k nodes with a unique equality index for fused-scan routing."""
    graph = KnowledgeGraph()
    n = 100_000
    graph.add_nodes(
        pd.DataFrame(
            {
                "nid": list(range(n)),
                "name": [f"Item_{i}" for i in range(n)],
                "code": [f"code_{i}" for i in range(n)],
                "bucket": [f"bucket_{i % 100}" for i in range(n)],
                "score": list(range(n)),
            }
        ),
        "Item",
        "nid",
        "name",
        columns=["code", "bucket", "score"],
    )
    graph.create_index("Item", "code")
    return graph


@pytest.fixture(scope="module")
def indexed_graph_with_unrelated_secondary_label():
    """100k indexed nodes plus a secondary label on another type."""
    graph = KnowledgeGraph()
    n = 100_000
    graph.add_nodes(
        pd.DataFrame(
            {
                "nid": list(range(n)),
                "name": [f"Item_{i}" for i in range(n)],
                "code": [f"code_{i}" for i in range(n)],
            }
        ),
        "Item",
        "nid",
        "name",
        columns=["code"],
    )
    graph.add_nodes(
        pd.DataFrame({"oid": [0], "name": ["Other"]}),
        "Other",
        "oid",
        "name",
        labels=["Unrelated"],
    )
    graph.create_index("Item", "code")
    return graph


@pytest.fixture(scope="module")
def in_selectivity_graph():
    """Dense pattern with a non-indexed IN side and an ID anchor."""
    graph = KnowledgeGraph()
    n = 10_000
    graph.add_nodes(
        pd.DataFrame(
            {
                "bid": list(range(n)),
                "name": [f"Broad_{i}" for i in range(n)],
                "code": [f"code_{i}" for i in range(n)],
            }
        ),
        "Broad",
        "bid",
        "name",
        columns=["code"],
    )
    graph.add_nodes(
        pd.DataFrame(
            {
                "aid": list(range(n)),
                "name": [f"Anchor_{i}" for i in range(n)],
            }
        ),
        "Anchor",
        "aid",
        "name",
    )
    graph.add_connections(
        pd.DataFrame(
            {
                "source": [i % n for i in range(30 * n)],
                "target": [(i % n + i // n) % n for i in range(30 * n)],
            }
        ),
        "LINK",
        "Broad",
        "source",
        "Anchor",
        "target",
    )
    return graph


@pytest.fixture(scope="module")
def consecutive_match_anchor_graph():
    """Broad first MATCH followed by a shared-variable ID anchor."""
    graph = KnowledgeGraph()
    n = 10_000
    for label in ("Hub", "Leaf", "Anchor"):
        graph.add_nodes(
            pd.DataFrame(
                {
                    "id": list(range(n)),
                    "name": [f"{label}_{i}" for i in range(n)],
                }
            ),
            label,
            "id",
            "name",
        )
    graph.add_connections(
        pd.DataFrame(
            {
                "source": [i % n for i in range(30 * n)],
                "target": [(i % n + i // n) % n for i in range(30 * n)],
            }
        ),
        "WIDE",
        "Hub",
        "source",
        "Leaf",
        "target",
    )
    graph.add_connections(
        pd.DataFrame({"source": list(range(n)), "target": list(range(n))}),
        "ANCHORED",
        "Hub",
        "source",
        "Anchor",
        "target",
    )
    return graph


@pytest.fixture(scope="module")
def wide_edge_count_graph():
    """One million homogeneous edges, matching the reported legal graph scale."""
    graph = KnowledgeGraph()
    node_count = 20_000
    edge_count = 1_000_000
    graph.add_nodes(
        pd.DataFrame(
            {
                "nid": list(range(node_count)),
                "name": [f"Node_{i}" for i in range(node_count)],
            }
        ),
        "Item",
        "nid",
        "name",
    )
    graph.add_connections(
        pd.DataFrame(
            {
                "source": [i % node_count for i in range(edge_count)],
                "target": [(i * 13 + 7) % node_count for i in range(edge_count)],
            }
        ),
        "LINKS",
        "Item",
        "source",
        "Item",
        "target",
    )
    return graph


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------


@pytest.mark.benchmark
def test_bench_add_nodes(benchmark):
    """Bulk node insertion (1000 nodes)."""
    graph = KnowledgeGraph()
    nodes = pd.DataFrame(
        {
            "nid": list(range(1000)),
            "name": [f"Node_{i}" for i in range(1000)],
            "value": [float(i) for i in range(1000)],
        }
    )

    benchmark(graph.add_nodes, nodes, "Item", "nid", "name")


@pytest.mark.benchmark
def test_bench_add_connections(benchmark):
    """Bulk edge insertion (2000 edges)."""
    graph = KnowledgeGraph()
    nodes = pd.DataFrame(
        {
            "nid": list(range(1000)),
            "name": [f"Node_{i}" for i in range(1000)],
        }
    )
    graph.add_nodes(nodes, "Item", "nid", "name")

    edges = pd.DataFrame(
        {
            "from_id": [i % 1000 for i in range(2000)],
            "to_id": [(i * 7 + 13) % 1000 for i in range(2000)],
            "weight": [float(i % 100) for i in range(2000)],
        }
    )

    benchmark(graph.add_connections, edges, "LINKS", "Item", "from_id", "Item", "to_id", columns=["weight"])


@pytest.mark.benchmark
def test_bench_cypher_match(benchmark, bench_graph):
    """Simple MATCH...RETURN query."""
    benchmark(bench_graph.cypher, "MATCH (n:Item) RETURN n.title, n.value LIMIT 100")


@pytest.mark.benchmark
def test_bench_cypher_match_materialized(benchmark, bench_graph):
    """Simple MATCH consumed into Python rows (includes lazy materialization)."""

    def query_and_consume():
        return bench_graph.cypher("MATCH (n:Item) RETURN n.title, n.value LIMIT 100").to_list()

    benchmark(query_and_consume)


@pytest.mark.benchmark
def test_bench_cypher_where(benchmark, bench_graph):
    """Filtered MATCH...WHERE...RETURN query."""
    benchmark(bench_graph.cypher, "MATCH (n:Item) WHERE n.value > 500 RETURN n.title, n.value")


@pytest.mark.benchmark
def test_bench_grouped_count_top_k_target_property(benchmark, grouped_count_graph):
    """User shape: count incoming rows, group on target property, order + limit."""

    def query_and_consume():
        return grouped_count_graph.cypher(
            "MATCH (s:Source)-[:RELATES_TO]->(g:Group) "
            "RETURN g.bucket AS bucket, count(s) AS uses "
            "ORDER BY uses DESC LIMIT 10"
        ).to_list()

    result = benchmark(query_and_consume)
    assert len(result) == 10
    assert all(row["uses"] == 300 for row in result)


@pytest.mark.benchmark
def test_bench_grouped_count_top_k_source_property(benchmark, grouped_count_graph):
    """User shape: count outgoing rows, group on source property, order + limit."""

    def query_and_consume():
        return grouped_count_graph.cypher(
            "MATCH (s:Source)-[:RELATES_TO]->(g:Group) "
            "RETURN s.bucket AS bucket, count(g) AS uses "
            "ORDER BY uses DESC LIMIT 10"
        ).to_list()

    result = benchmark(query_and_consume)
    assert len(result) == 10
    assert all(row["uses"] == 300 for row in result)


@pytest.mark.benchmark
def test_bench_untyped_edge_count_1m(benchmark, wide_edge_count_graph):
    """Wide `MATCH ()-[r]->()` count used by graph inventory interfaces."""

    def query_and_consume():
        return wide_edge_count_graph.cypher("MATCH ()-[r]->() RETURN count(r) AS edges").to_list()

    result = benchmark(query_and_consume)
    assert result == [{"edges": 1_000_000}]


@pytest.mark.benchmark
@pytest.mark.parametrize(
    ("operator", "needle", "expected_rows"),
    [("CONTAINS", "Group_1", 20), ("ENDS WITH", "_1", 4)],
)
def test_bench_two_edge_distinct_filtered_path(benchmark, grouped_count_graph, operator, needle, expected_rows):
    """Consumed two-edge text-filter path, covering substring and suffix routing."""

    def query_and_consume():
        return grouped_count_graph.cypher(
            f"MATCH (g:Group)<-[:RELATES_TO]-(s:Source)-[:RELATES_TO]->(peer:Group) "
            f"WHERE g.name {operator} $needle "
            "RETURN DISTINCT peer.bucket AS bucket LIMIT 20",
            params={"needle": needle},
        ).to_list()

    result = benchmark(query_and_consume)
    assert len(result) == expected_rows


@pytest.mark.benchmark
@pytest.mark.parametrize(
    ("operator", "needle", "expected_rows"),
    [("CONTAINS", "Edge_12345", 2), ("ENDS WITH", "_1", 2)],
)
def test_bench_two_edge_relationship_text_filter(benchmark, grouped_count_graph, operator, needle, expected_rows):
    """Consumed two-hop relationship-text filter, including parameter routing."""

    def query_and_consume():
        return grouped_count_graph.cypher(
            "MATCH (g:Group)<-[r:RELATES_TO]-(s:Source)-[:RELATES_TO]->(peer:Group) "
            f"WHERE r.tag {operator} $needle "
            "RETURN DISTINCT peer.bucket AS bucket LIMIT 20",
            params={"needle": needle},
        ).to_list()

    result = benchmark(query_and_consume)
    assert len(result) == expected_rows


@pytest.mark.benchmark
@pytest.mark.parametrize(
    ("query", "expected"),
    [
        (
            "MATCH (n:Item {code: $code}) RETURN count(*) AS n",
            [{"n": 1}],
        ),
        (
            "MATCH (n:Item) WHERE n.code = $code RETURN n.bucket AS bucket, count(*) AS n",
            [{"bucket": "bucket_21", "n": 1}],
        ),
        (
            "MATCH (n:Item {code: $code}) RETURN n.code AS code, n.score AS score ORDER BY n.score DESC LIMIT 5",
            [{"code": "code_54321", "score": 54321}],
        ),
    ],
)
def test_bench_fused_indexed_node_scan(benchmark, indexed_node_scan_graph, query, expected):
    """Fused aggregate/top-K operators must reuse the unique property index."""

    def query_and_consume():
        return indexed_node_scan_graph.cypher(query, params={"code": "code_54321"}).to_list()

    result = benchmark(query_and_consume)
    assert result == expected


@pytest.mark.benchmark
def test_bench_nonindexed_in_vs_id_anchor(benchmark, in_selectivity_graph):
    """A linear-scan IN predicate must not tie an O(1) endpoint ID anchor."""
    query = "MATCH (a:Broad)-[:LINK]->(b:Anchor {id: $anchor}) WHERE a.code IN $codes RETURN count(*) AS n"

    def query_and_consume():
        return in_selectivity_graph.cypher(
            query,
            params={"anchor": 7_321, "codes": ["code_7321"]},
        ).to_list()

    result = benchmark(query_and_consume)
    assert result == [{"n": 1}]


@pytest.mark.benchmark
def test_bench_index_with_unrelated_secondary_label(benchmark, indexed_graph_with_unrelated_secondary_label):
    """A secondary label on another type must not force an indexed type scan."""
    query = "MATCH (n:Item {code: $code}) RETURN n.id AS id"

    def query_and_consume():
        return indexed_graph_with_unrelated_secondary_label.cypher(query, params={"code": "code_54321"}).to_list()

    result = benchmark(query_and_consume)
    assert result == [{"id": 54_321}]


@pytest.mark.benchmark
def test_bench_consecutive_match_id_anchor(benchmark, consecutive_match_anchor_graph):
    """A later shared-variable ID anchor should drive a broad MATCH span."""
    query = """
        MATCH (h:Hub)-[:WIDE]->(leaf:Leaf)
        MATCH (h)-[:ANCHORED]->(anchor:Anchor {id: $anchor})
        RETURN count(*) AS n
    """

    def query_and_consume():
        return consecutive_match_anchor_graph.cypher(query, params={"anchor": 7_321}).to_list()

    result = benchmark(query_and_consume)
    assert result == [{"n": 30}]


@pytest.mark.benchmark
def test_bench_traversal(benchmark, bench_graph):
    """Multi-hop traversal via fluent API."""
    benchmark(bench_graph.select("Item").where({"id": 0}).traverse, "LINKS")


@pytest.mark.benchmark
def test_bench_shortest_path(benchmark, bench_graph):
    """Shortest path computation."""
    benchmark(bench_graph.cypher, "MATCH p = shortestPath((a:Item {id: 0})-[*]-(b:Item {id: 500})) RETURN length(p)")


# ---------------------------------------------------------------------------
# Save throughput
# ---------------------------------------------------------------------------
#
# `bench_graph_columnar` used to stand beside `bench_graph` as the "columnar"
# fixture. The two built byte-identical graphs, because nothing in either body
# ever changed a storage shape, and construction is columnar from the first
# node now — so the pair, and the two query cells that only differed by which
# of them they took, were measuring one thing twice. Merged into `bench_graph`
# and `test_bench_cypher_{where,match}`.


@pytest.mark.benchmark
def test_bench_save_kgl(benchmark, bench_graph, tmp_path):
    """Save to one `.kgl` path, overwriting it every round.

    fsync=False: this tracks *serialization + write* throughput, the thing
    kglite controls. The fsync durability barrier (default in save()) is a
    fixed OS-level cost orthogonal to serialization — including it would make a
    µs-scale bench dominated by ms-scale disk-flush latency.
    """
    path = str(tmp_path / "bench.kgl")
    benchmark(lambda: bench_graph.save(path, fsync=False))


@pytest.mark.benchmark
def test_bench_save_kgl_new_file(benchmark, bench_graph, tmp_path):
    """Save to a fresh `.kgl` path every round (fsync=False — see above).

    The pair with `test_bench_save_kgl` separates writing over an existing file
    from creating one, which are different syscall paths on every platform this
    ships to. Formerly `test_bench_save_v3`, named for a container version two
    bumps out of date; the operation and the fixture are unchanged.
    """
    counter = [0]

    def save():
        bench_graph.save(str(tmp_path / f"fresh_{counter[0]}.kgl"), fsync=False)
        counter[0] += 1

    benchmark(save)


# ---------------------------------------------------------------------------
# Load throughput
# ---------------------------------------------------------------------------
#
# The file under test is written by the build under test, and that is the
# whole point of the cell. 0.16.6 added two integrity layers that only a file
# *carrying* them pays for, and every load-shaped benchmark this repo had read
# either a stored fixture or a directory, so the class was invisible: the
# release measured "+11-15% on save, loads unmoved" and shipped a +85% load
# regression, reported from downstream two days later. A cell whose fixture
# predates the change under test cannot see the change under test.
#
# Version-independent by construction, which the frozen CI harness needs: it
# times each version's own write-then-read path, with no argument or format
# assumption. A 0.13.2 reference writes a digest-free file and reads it back;
# this tree writes digests and verifies them. That difference is exactly the
# quantity being gated.

#: Alphabet for `_entropic_strings` — 64 symbols, so six LCG bits index it.
_ENTROPY_ALPHABET = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+/"


def _entropic_strings(count: int, width: int, seed: int) -> list[str]:
    """`count` deterministic, poorly-compressible strings of `width` chars.

    High entropy on purpose. Sections are zstd-compressed and the integrity
    digest runs over the *compressed* bytes, so a fixture built from
    templated text (`f"Node_{i}"`) collapses to a fraction of a megabyte and
    measures fixed overhead rather than throughput — the 157k-node graph this
    cell was sized against compressed 195 MB of real text to 13 MB of
    `f`-strings. An inlined LCG rather than `random.Random` for the same
    reason as `_scale_free_edges`: the fixture must be byte-identical on every
    interpreter this harness runs under.
    """
    state = seed
    out = []
    for _ in range(count):
        chars = []
        for _ in range(width):
            state = (state * 1_103_515_245 + 12_345) & 0x7FFF_FFFF
            chars.append(_ENTROPY_ALPHABET[(state >> 16) & 63])
        out.append("".join(chars))
    return out


@pytest.fixture(scope="module")
def written_kgl_path(tmp_path_factory):
    """A ~4 MB `.kgl` written by the build under test; ~10 ms to load.

    Sized so the container payload dominates: at this scale the digest layers
    0.16.6 introduced cost +72% against a digest-free write of the same graph,
    which is comfortably outside the 20% gate, while 100 rounds still finish
    in about a second.
    """
    nodes, edges = 20_000, 40_000
    graph = KnowledgeGraph()
    graph.add_nodes(
        pd.DataFrame(
            {
                "nid": list(range(nodes)),
                "name": [f"N{i}" for i in range(nodes)],
                "body": _entropic_strings(nodes, 200, 20_260_822),
            }
        ),
        "Doc",
        "nid",
        "name",
    )
    graph.add_connections(
        pd.DataFrame(
            {
                "s": [i % nodes for i in range(edges)],
                "d": [(i * 7919 + 13) % nodes for i in range(edges)],
                "note": _entropic_strings(edges, 24, 20_260_823),
            }
        ),
        "CITES",
        "Doc",
        "s",
        "Doc",
        "d",
        columns=["note"],
    )
    path = str(tmp_path_factory.mktemp("load_bench") / "written.kgl")
    graph.save(path)
    return path


@pytest.mark.benchmark
def test_bench_load_kgl(benchmark, written_kgl_path):
    """Deserialize a `.kgl` this build wrote — the process-start latency.

    The timed region is the load alone; writing happens once in the fixture.
    """
    graph = benchmark(kglite.load, written_kgl_path)
    assert graph.cypher("MATCH (n:Doc) RETURN count(n) AS c").to_list() == [{"c": 20_000}]


@pytest.fixture(scope="module")
def written_disk_dir(tmp_path_factory):
    """A disk-mode graph directory published by the build under test.

    Same 20k/40k shape and the same high-entropy payload as
    :func:`written_kgl_path`, for the same reason — the reopen cost below
    tracks the published bytes, not a fixed header — but a different seed, so
    neither fixture can be mistaken for the other's artifact.

    The graph handle is dropped when this fixture returns, which releases the
    single-writer lease the disk directory holds. Without that, the first
    ``open()`` in the cell below would be refused by this same process.
    """
    nodes, edges = 20_000, 40_000
    root = tmp_path_factory.mktemp("disk_reopen_bench") / "graph"
    graph = KnowledgeGraph(storage="disk", path=str(root))
    graph.add_nodes(
        pd.DataFrame(
            {
                "nid": list(range(nodes)),
                "name": [f"N{i}" for i in range(nodes)],
                "body": _entropic_strings(nodes, 200, 20_260_824),
            }
        ),
        "Doc",
        "nid",
        "name",
    )
    graph.add_connections(
        pd.DataFrame(
            {
                "s": [i % nodes for i in range(edges)],
                "d": [(i * 7919 + 13) % nodes for i in range(edges)],
                "note": _entropic_strings(edges, 24, 20_260_825),
            }
        ),
        "CITES",
        "Doc",
        "s",
        "Doc",
        "d",
        columns=["note"],
    )
    graph.save(str(root), fsync=False)
    return str(root)


@pytest.mark.benchmark
def test_bench_disk_dir_reopen_fresh(benchmark, written_disk_dir):
    """Reopen a disk directory this build published — the reload path.

    The other half of the class `test_bench_load_kgl` covers: a disk graph is
    reopened by resolving the newest generation and mapping its segments, and
    that path — the one the "reload instead of rebuild" story is about — had no
    gating cell at all, on an artifact written by any build, fresh or stored.

    `pedantic` with a setup step rather than a plain `benchmark(...)` call:
    `open()` takes the directory's single-writer lease, and pytest-benchmark
    holds the previous round's return value across the next round's call, so a
    plain call would ask this same process to reopen a directory it has not let
    go of yet and be refused. The setup drops the previous handle *outside* the
    timed region, which leaves the timed region as the open alone. Rounds and
    warmup match what `make bench-check` passes the plain cells.
    """
    handle: dict[str, KnowledgeGraph] = {}

    def release_previous_handle():
        handle.pop("graph", None)

    def reopen():
        handle["graph"] = kglite.open(written_disk_dir)

    benchmark.pedantic(reopen, setup=release_previous_handle, rounds=100, warmup_rounds=20, iterations=1)
    assert handle["graph"].cypher("MATCH (n:Doc) RETURN count(n) AS c").to_list() == [{"c": 20_000}]


# ---------------------------------------------------------------------------
# Persistence-surface coverage registry
# ---------------------------------------------------------------------------
#
# 0.16.6 measured its saves honestly and still shipped a +85% load regression,
# because no cell anywhere read back an artifact the build under test had just
# written: the read half of every persistence surface was structurally
# invisible. Adding a load cell fixes one instance. This registry is what
# stops the *class* from recurring — it pairs each persisted artifact kind with
# the cell that reads a freshly written one, and the meta-test below fails when
# a writer or reader cell in this module is missing from it. A new
# `test_bench_save_foo` therefore cannot land read-blind: the gate names the
# surface.


class PersistenceSurface(NamedTuple):
    """One persisted artifact kind: its writer cells and its fresh reader."""

    #: Benchmark cells in this module that *write* the artifact.
    writers: tuple[str, ...]
    #: The cell that reads one back. Non-optional by construction: a surface
    #: entry exists to name this cell.
    reader: str
    #: The fixture the reader takes, which must be the one that writes the
    #: artifact with the build under test. That provenance is the whole
    #: property — a reader repointed at a stored fixture keeps its name, keeps
    #: passing, and stops being able to see a write-format change.
    fresh_fixture: str


PERSISTENCE_SURFACES: dict[str, PersistenceSurface] = {
    "kgl_file": PersistenceSurface(
        writers=("test_bench_save_kgl", "test_bench_save_kgl_new_file"),
        reader="test_bench_load_kgl",
        fresh_fixture="written_kgl_path",
    ),
    "disk_dir": PersistenceSurface(
        # Empty on purpose, and not a gap this table is hiding: no cell in this
        # module publishes a disk directory (the fixture does, untimed). The
        # rule enforced below is one-directional — a writer cell *here* needs a
        # reader *here* — so an empty tuple claims nothing that is not true. A
        # disk publish cell added to this module has to land in this tuple.
        writers=(),
        reader="test_bench_disk_dir_reopen_fresh",
        fresh_fixture="written_disk_dir",
    ),
}

#: Underscore-separated name components that mark a cell as writing, or as
#: reading, a persistence artifact. Whole components, never substrings: a
#: substring test drags in every unrelated cell whose name happens to contain
#: the letters, and the resulting noise is what gets a gate switched off.
WRITER_NAME_COMPONENTS = frozenset({"save", "write", "persist", "checkpoint"})
READER_NAME_COMPONENTS = frozenset({"load", "reopen", "restore"})


def _benchmark_cells() -> dict[str, object]:
    """Every benchmark cell defined in this module, by name."""
    return {
        name: obj
        for name, obj in vars(sys.modules[__name__]).items()
        if name.startswith("test_bench_") and callable(obj)
    }


# Unmarked on purpose, like the var-length companion tests at the end of this
# file: `-m benchmark` deselects it, so it never runs inside `make bench-check`
# or CI's 0.13.2 reference leg, while `make test` runs it on every change.


def test_every_persistence_cell_is_registered_with_its_counterpart():
    """The registry names a fresh reader for every persistence cell here.

    Red in both directions, which is what makes it worth having: an
    unregistered writer cell (someone adds `test_bench_save_foo` and no reader)
    fails naming the cell, and a deleted registry entry fails naming its
    now-unregistered reader.
    """
    cells = _benchmark_cells()

    for surface, entry in PERSISTENCE_SURFACES.items():
        for name in (*entry.writers, entry.reader):
            assert name in cells, f"{surface}: {name} is not a benchmark cell in this module"
            marks = getattr(cells[name], "pytestmark", [])
            assert any(mark.name == "benchmark" for mark in marks), f"{surface}: {name} is not benchmark-marked"
        parameters = inspect.signature(cells[entry.reader]).parameters
        assert entry.fresh_fixture in parameters, (
            f"{surface}: {entry.reader} does not take {entry.fresh_fixture}, the fixture that writes the "
            "artifact with the build under test — a reader reading a stored artifact cannot see a "
            "write-path change"
        )

    registered_writers = {name for entry in PERSISTENCE_SURFACES.values() for name in entry.writers}
    registered_readers = {entry.reader for entry in PERSISTENCE_SURFACES.values()}
    for components, registered, role in (
        (WRITER_NAME_COMPONENTS, registered_writers, "writer"),
        (READER_NAME_COMPONENTS, registered_readers, "reader"),
    ):
        unregistered = sorted(name for name in cells if components & set(name.split("_")) and name not in registered)
        assert not unregistered, (
            f"persistence {role} cells missing from PERSISTENCE_SURFACES: {unregistered}. "
            f"Every {role} cell belongs to a surface entry pairing it with its counterpart — an unpaired "
            "persistence cell is exactly how the 0.16.6 load regression shipped."
        )


# ---------------------------------------------------------------------------
# Value::Node projection benchmarks (shared with the Bolt consumer)
# ---------------------------------------------------------------------------
#
# The Value enum carries Node / Relationship / Path / List / Map variants, so
# `RETURN n` does not collapse to a title string — it materializes a full
# {id, labels, properties} structure. The Bolt server routes this over
# PackStream as a Node struct, so any regression in projection cost shows up
# in both Python `cypher()` and Bolt PULL.
#
# These benchmarks are the baseline for that path. Captured to
# `tests/benchmarks/baselines/<version>.json` on the next release commit
# via `make refresh-release-constants`.


@pytest.fixture
def node_projection_graph():
    """10k Person nodes + ~30k KNOWS edges — sized so projection cost
    dominates over query planning."""
    graph = KnowledgeGraph()
    n = 10_000
    nodes = pd.DataFrame(
        {
            "pid": list(range(n)),
            "name": [f"P{i}" for i in range(n)],
            "age": [20 + (i % 60) for i in range(n)],
            "city": [f"city_{i % 100}" for i in range(n)],
        }
    )
    graph.add_nodes(nodes, "Person", "pid", "name")

    edges = pd.DataFrame(
        {
            "s": [i % n for i in range(3 * n)],
            "d": [(i * 13 + 7) % n for i in range(3 * n)],
        }
    )
    graph.add_connections(edges, "KNOWS", "Person", "s", "Person", "d")
    return graph


@pytest.mark.benchmark
def test_bench_return_node_10k(benchmark, node_projection_graph):
    """RETURN n over 10k nodes — eager Value::Node projection.

    Drives the projection path shared between Python `cypher()` and the
    Bolt server's RECORD emission. Regressions here are visible
    everywhere downstream of Value::Node projection.
    """
    benchmark(node_projection_graph.cypher, "MATCH (n:Person) RETURN n")


@pytest.mark.benchmark
def test_bench_return_id_10k(benchmark, node_projection_graph):
    """RETURN id(n) over 10k nodes — node-identity projection only.

    Companion to `return_node_10k`: isolates the id-resolution path
    (`NodeView::id()` → ColumnStore) from full-node materialization, so a
    regression in identity reads is visible without the property/label
    materialization cost on top. `id(n)` must not pay full-node materialization.
    """
    benchmark(node_projection_graph.cypher, "MATCH (n:Person) RETURN id(n)")


@pytest.mark.benchmark
def test_bench_return_node_rel_node_100(benchmark, node_projection_graph):
    """Multi-binding projection: `a`, `r`, `b` LIMIT 100.

    Exercises Node + Relationship + Node materialization in the same
    record — the typical shape of a Bolt PULL response for graph
    visualization clients (Neo4j Browser, BloodHound).
    """
    benchmark(
        node_projection_graph.cypher,
        "MATCH (a:Person)-[r:KNOWS]->(b:Person) RETURN a, r, b LIMIT 100",
    )


# ---------------------------------------------------------------------------
# Variable-length traversal (part-6 program, phase V1)
# ---------------------------------------------------------------------------
#
# These cells are the measurement instrument for the var-length work: `*k..k`
# lowering (V2), the EXISTS witness early-exit (V3), and the DISTINCT pushdown
# in the UNWIND branch (V4). Each expensive cell ships with a cheap control
# that the same phase must NOT move, so a capture can tell a real win from the
# machine drifting under it.
#
# Captured *after* the V0 soundness fix on purpose. Before V0 the fast
# var-length path answered distance reachability where Cypher asks for trail
# reachability, so a pre-V0 number is a measurement of a different — and wrong
# — computation and cannot baseline anything.
#
# NOTE ON ASSERTIONS. `make bench-check` is not the only consumer of this file:
# CI copies it outside the checkout and runs it with `-m benchmark` against the
# published kglite 0.13.2 wheel to get a same-runner reference leg. 0.13.2
# predates V0, so its var-length answers differ from this tree's (6856 vs 6867
# reachable on `khop_social_graph`). Absolute expectations therefore live in the
# unmarked companion tests below, which that `-m benchmark` run deselects; the
# benchmark bodies assert only version-independent invariants.


def _scale_free_edges(node_count: int, attachments: int = 2) -> tuple[list[int], list[int]]:
    """Deterministic preferential attachment — a social-graph degree shape.

    Each new node attaches to `attachments` distinct existing nodes drawn from
    a repeated-node list, so the draw probability is proportional to degree and
    the result carries hubs (the property that makes k-hop reachability large
    and traversal cost interesting). The draw uses an inlined LCG rather than
    `random.Random`: the fixture must produce byte-identical graphs on every
    interpreter this harness is run under, including the frozen CI copy and the
    3.12 reference environment, and only arithmetic guarantees that.

    Returned as a mutual (both-directions) edge list. A one-directional
    attachment graph bounds 3-hop reach at `attachments ** 3` nodes, which
    measures call overhead rather than traversal.
    """
    repeated = list(range(attachments))
    src: list[int] = []
    dst: list[int] = []
    state = 20_260_821
    for new in range(attachments, node_count):
        chosen: list[int] = []
        while len(chosen) < attachments:
            state = (state * 1_103_515_245 + 12_345) & 0x7FFF_FFFF
            candidate = repeated[state % len(repeated)]
            if candidate not in chosen:
                chosen.append(candidate)
        for target in chosen:
            src.append(new)
            dst.append(target)
            repeated.append(target)
        repeated.extend([new] * attachments)
    return src + dst, dst + src


def _social_graph(node_count: int) -> KnowledgeGraph:
    """`node_count` Person nodes joined by mutual KNOWS edges."""
    src, dst = _scale_free_edges(node_count)
    graph = KnowledgeGraph()
    graph.add_nodes(
        pd.DataFrame(
            {
                "pid": list(range(node_count)),
                "name": [f"P{i}" for i in range(node_count)],
            }
        ),
        "Person",
        "pid",
        "name",
    )
    graph.add_connections(
        pd.DataFrame({"s": src, "d": dst}),
        "KNOWS",
        "Person",
        "s",
        "Person",
        "d",
    )
    return graph


#: 50 seed ids spread across `khop_social_graph`. Arithmetic rather than
#: sampled, for the same reproducibility reason as `_scale_free_edges`; the
#: stride is coprime with the node count over 50 terms, so they are distinct.
KHOP_SEED_IDS = [(i * 197 + 13) % 10_000 for i in range(50)]


@pytest.fixture(scope="module")
def khop_social_graph():
    """10k Person nodes, ~40k mutual KNOWS — the multi-seed k-hop shape."""
    return _social_graph(10_000)


@pytest.fixture(scope="module")
def two_hop_social_graph():
    """600 Person nodes — sized so the *unanchored* two-hop count stays a
    low-millisecond cell.

    The `*2..2`-vs-fixed gap only exists on the unanchored spelling: with an
    id-anchored source set both spellings collapse to the same per-source
    expansion and measure nothing (measured 1.02x at 50 seeds, against 11x
    unanchored). Keeping the source set whole is what makes this cell a
    lowering target, so the node count carries the runtime budget instead.
    """
    return _social_graph(600)


@pytest.mark.benchmark
def test_bench_khop3_in_list_count_distinct(benchmark, khop_social_graph):
    """50-seed 3-hop reach, counted distinct — the eval's reported shape.

    The single-pattern spelling with the IN-list on the same MATCH: this is the
    first-MATCH branch, the one that already has the DISTINCT pushdown. Paired
    with `khop3_unwind_distinct`, which reaches the same answer through the
    UNWIND branch (V4).
    """

    def query_and_consume():
        return khop_social_graph.cypher(
            "MATCH (p:Person)-[:KNOWS*1..3]->(f:Person) WHERE p.id IN $ids RETURN count(DISTINCT f) AS reached",
            params={"ids": KHOP_SEED_IDS},
        ).to_list()

    result = benchmark(query_and_consume)
    assert result[0]["reached"] > 1_000


@pytest.mark.benchmark
def test_bench_khop3_unwind_distinct(benchmark, khop_social_graph):
    """The UNWIND spelling of `khop3_in_list_count_distinct`'s answer.

    Self-controlling pair: the two cells compute the same number over the same
    graph, so the ratio between them is the branch difference and nothing else.
    Closing that ratio is V4's stop rule.
    """

    def query_and_consume():
        return khop_social_graph.cypher(
            "UNWIND $ids AS i MATCH (p:Person {id: i})-[:KNOWS*1..3]->(f:Person) RETURN count(DISTINCT f) AS reached",
            params={"ids": KHOP_SEED_IDS},
        ).to_list()

    result = benchmark(query_and_consume)
    assert result[0]["reached"] > 1_000


@pytest.mark.benchmark
def test_bench_var_length_2_2_count_star(benchmark, two_hop_social_graph):
    """`*2..2` counted per path — the `*k..k` lowering target (V2).

    `count(*)` is not dedup-safe, so this is the per-path expansion under trail
    semantics, exactly what the fixed spelling below computes by a different
    route. V2 ships when this cell is within 1.2x of that control.
    """

    def query_and_consume():
        return two_hop_social_graph.cypher(
            "MATCH (p:Person)-[:KNOWS*2..2]->(f:Person) RETURN count(*) AS paths"
        ).to_list()

    result = benchmark(query_and_consume)
    assert result[0]["paths"] > 1_000


@pytest.mark.benchmark
def test_bench_fixed_two_hop_count_star(benchmark, two_hop_social_graph):
    """Explicit two-hop control for `var_length_2_2_count_star`.

    Relationship uniqueness inside one MATCH makes this the same set of paths
    the `*2..2` spelling must produce; the companion test pins that equality.
    A V2 that moves this cell has changed the fixed-hop path, not lowered the
    variable-length one.
    """

    def query_and_consume():
        return two_hop_social_graph.cypher(
            "MATCH (p:Person)-[:KNOWS]->(x:Person)-[:KNOWS]->(f:Person) RETURN count(*) AS paths"
        ).to_list()

    result = benchmark(query_and_consume)
    assert result[0]["paths"] > 1_000


@pytest.mark.benchmark
def test_bench_exists_var_length_witness(benchmark, khop_social_graph):
    """`EXISTS { (p)-[:KNOWS*1..3]->(:Person) }` — one witness is enough.

    The pattern predicate needs a single match to answer, but the var-length
    expansion behind it runs to completion for every candidate row. V3's stop
    rule is this cell within 2x of the fixed-hop control below.
    """

    def query_and_consume():
        return khop_social_graph.cypher(
            "MATCH (p:Person) WHERE p.id IN $ids AND EXISTS { (p)-[:KNOWS*1..3]->(:Person) } "
            "RETURN count(p) AS witnessed",
            params={"ids": KHOP_SEED_IDS},
        ).to_list()

    result = benchmark(query_and_consume)
    assert result[0]["witnessed"] == len(KHOP_SEED_IDS)


@pytest.mark.benchmark
def test_bench_var_length_1_8_count_distinct(benchmark, khop_social_graph):
    """50-seed **eight**-hop reach, counted distinct — the depth cell (V8).

    The `khop3_*` pair measures traversal at a depth where the frontier is
    still growing (7 690 of 10 000 nodes at three hops). This cell measures it
    past saturation: at eight hops every seed's own BFS has covered the whole
    component, so the cost is the flat plateau of the distance BFS rather than
    a point on its climb. V8's scaling study measured the plateau starting at
    k=7 and holding to k=12 within 2.5%, which is the semantic floor for a
    per-source BFS — so a future change that reintroduces per-level work would
    show up here as a climb, and nowhere else in this file.

    Deliberately the `count(DISTINCT)` consumer: it is the dedup-safe shape
    that V0's proof licenses onto the fast path, so this cell also guards that
    the licence still applies at depth.
    """

    def query_and_consume():
        return khop_social_graph.cypher(
            "MATCH (p:Person)-[:KNOWS*1..8]->(f:Person) WHERE p.id IN $ids RETURN count(DISTINCT f) AS reached",
            params={"ids": KHOP_SEED_IDS},
        ).to_list()

    result = benchmark(query_and_consume)
    assert result[0]["reached"] > 1_000


@pytest.mark.benchmark
def test_bench_exists_fixed_hop(benchmark, khop_social_graph):
    """Single-hop EXISTS control for `exists_var_length_witness`.

    Same candidate rows, same predicate shape, no variable-length expansion —
    so it is the cost of everything *except* the thing V3 changes.
    """

    def query_and_consume():
        return khop_social_graph.cypher(
            "MATCH (p:Person) WHERE p.id IN $ids AND EXISTS { (p)-[:KNOWS]->(:Person) } RETURN count(p) AS witnessed",
            params={"ids": KHOP_SEED_IDS},
        ).to_list()

    result = benchmark(query_and_consume)
    assert result[0]["witnessed"] == len(KHOP_SEED_IDS)


# ---------------------------------------------------------------------------
# Companion correctness tests for the var-length cells
# ---------------------------------------------------------------------------
#
# Unmarked on purpose: `-m benchmark` deselects them, which keeps them out of
# both `make bench-check` and CI's 0.13.2 reference leg, while `make test`
# still runs them. They are what makes the benchmark pairs trustworthy — a
# self-controlling pair whose two halves stopped computing the same answer is
# a ratio between two unrelated numbers.


def test_khop3_spellings_agree(khop_social_graph):
    """All three spellings of "3-hop reach from 50 seeds" return one answer.

    The two-MATCH spelling is included because it is the shape V0's Bug B was
    about: a second MATCH whose consumer is a plain projection was getting the
    first clause's dedup proof applied to it.
    """
    params = {"ids": KHOP_SEED_IDS}
    single = khop_social_graph.cypher(
        "MATCH (p:Person)-[:KNOWS*1..3]->(f:Person) WHERE p.id IN $ids RETURN count(DISTINCT f) AS reached",
        params=params,
    ).to_list()
    two_match = khop_social_graph.cypher(
        "MATCH (p:Person) WHERE p.id IN $ids MATCH (p)-[:KNOWS*1..3]->(f:Person) RETURN count(DISTINCT f) AS reached",
        params=params,
    ).to_list()
    unwind = khop_social_graph.cypher(
        "UNWIND $ids AS i MATCH (p:Person {id: i})-[:KNOWS*1..3]->(f:Person) RETURN count(DISTINCT f) AS reached",
        params=params,
    ).to_list()

    assert single == two_match == unwind
    # Not a tautology: a shared bug returning 0 everywhere would satisfy the
    # equality above. The seeds reach most of a 10k graph at three hops.
    assert single[0]["reached"] > 5_000


def test_var_length_1_8_reach_contains_the_1_3_reach(khop_social_graph):
    """Eight-hop reach is a superset of three-hop reach, and strictly bigger.

    The companion for `var_length_1_8_count_distinct`. `*1..k` reachability is
    monotone in `k` by construction — every `*1..3` trail is a `*1..8` trail —
    so an eight-hop answer below the three-hop one is a bug, not a slower
    query. The second assertion is what makes this non-vacuous: on this fixture
    the two numbers genuinely differ (7 690 against the full 10 000), so a
    change that collapsed the depth cell into a three-hop query would be caught
    here rather than showing up only as a suspiciously fast benchmark.
    """
    params = {"ids": KHOP_SEED_IDS}
    deep = khop_social_graph.cypher(
        "MATCH (p:Person)-[:KNOWS*1..8]->(f:Person) WHERE p.id IN $ids RETURN count(DISTINCT f) AS reached",
        params=params,
    ).to_list()[0]["reached"]
    shallow = khop_social_graph.cypher(
        "MATCH (p:Person)-[:KNOWS*1..3]->(f:Person) WHERE p.id IN $ids RETURN count(DISTINCT f) AS reached",
        params=params,
    ).to_list()[0]["reached"]

    assert deep >= shallow
    assert deep > shallow, (
        f"the depth cell must not be measuring the same reachable set as the three-hop cells (both {deep})"
    )


def test_var_length_2_2_matches_fixed_two_hop(two_hop_social_graph):
    """`*2..2` and the explicit two-hop spelling count the same paths.

    Both are trails: relationship uniqueness inside a single MATCH forbids
    reusing an edge, and `*2..2` enforces the same across its hops. The pair is
    only a lowering target while this holds — V2 rewrites one into the other.
    """
    var_length = two_hop_social_graph.cypher(
        "MATCH (p:Person)-[:KNOWS*2..2]->(f:Person) RETURN count(*) AS paths"
    ).to_list()
    fixed = two_hop_social_graph.cypher(
        "MATCH (p:Person)-[:KNOWS]->(x:Person)-[:KNOWS]->(f:Person) RETURN count(*) AS paths"
    ).to_list()

    assert var_length == fixed
    assert var_length[0]["paths"] > 1_000


def test_exists_var_length_matches_fixed_hop(khop_social_graph):
    """The two EXISTS cells answer the same question on this fixture.

    Semantics differ in general — "reachable within 3 hops" is weaker than "has
    an outgoing edge". They coincide here because the mutual attachment graph
    leaves no isolated node, which the first assertion pins: if a future fixture
    change introduced one, this test fails rather than the control quietly
    drifting into measuring a different row set.
    """
    isolated = khop_social_graph.cypher(
        "MATCH (p:Person) WHERE NOT EXISTS { (p)-[:KNOWS]->(:Person) } RETURN count(p) AS n"
    ).to_list()
    assert isolated == [{"n": 0}]

    params = {"ids": KHOP_SEED_IDS}
    var_length = khop_social_graph.cypher(
        "MATCH (p:Person) WHERE p.id IN $ids AND EXISTS { (p)-[:KNOWS*1..3]->(:Person) } RETURN count(p) AS witnessed",
        params=params,
    ).to_list()
    fixed = khop_social_graph.cypher(
        "MATCH (p:Person) WHERE p.id IN $ids AND EXISTS { (p)-[:KNOWS]->(:Person) } RETURN count(p) AS witnessed",
        params=params,
    ).to_list()

    assert var_length == fixed == [{"witnessed": len(KHOP_SEED_IDS)}]
