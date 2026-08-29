"""Integration tests for the per-type property column store.

Properties live in columns from the first node in every mode, so what these
pin is that the one shape carries property semantics correctly across Cypher
queries, mutations, save/load, and bulk operations — and that a consolidation
pass over the columns (``unspill()``, ``save()``) changes nothing an observer
can see. The file was ``test_columnar_storage.py``, named for a storage mode
that no longer exists as a distinct thing to be in.
"""

import os
import tempfile

import pandas as pd
import pytest

import kglite

# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def person_graph():
    """Graph with 5 Person nodes having mixed property types."""
    kg = kglite.KnowledgeGraph()
    df = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "full_name": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
            "age": [30, 25, 35, 28, 42],
            "score": [1.5, 2.7, 3.9, 4.1, 0.8],
            "active": [True, False, True, True, False],
        }
    )
    kg.add_nodes(df, "Person", "id", "id")
    return kg


@pytest.fixture
def multi_type_graph():
    """Graph with Person + Company node types."""
    kg = kglite.KnowledgeGraph()
    persons = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "full_name": ["Alice", "Bob", "Charlie"],
            "age": [30, 25, 35],
        }
    )
    companies = pd.DataFrame(
        {
            "id": [10, 20],
            "company_name": ["Acme", "Globex"],
            "employees": [100, 200],
        }
    )
    kg.add_nodes(persons, "Person", "id", "id")
    kg.add_nodes(companies, "Company", "id", "id")
    edges = pd.DataFrame({"from": [1, 2], "to": [10, 20]})
    kg.add_connections(edges, "WORKS_AT", "Person", "from", "Company", "to")
    return kg


# ── The one shape ────────────────────────────────────────────────────────────


class TestConstruction:
    def test_columnar_from_construction(self, person_graph):
        """A freshly built graph already holds its properties in columns.

        There is no flag to read and nothing to switch: ``graph_info()`` is
        where the columns are observable, and it reports a row per live node
        before the graph has ever been saved.
        """
        info = person_graph.graph_info()
        assert info["columnar_live_rows"] == 5
        assert info["columnar_total_rows"] == 5


# ── Property preservation ────────────────────────────────────────────────────


class TestPropertyPreservation:
    def test_all_properties_survive_consolidation(self, person_graph):
        before = person_graph.cypher(
            "MATCH (n:Person) RETURN n.full_name, n.age, n.score, n.active ORDER BY n.age"
        ).to_list()

        # `unspill()` is the public route to a full column rebuild — the same
        # pass `save()` and `vacuum()` run.
        person_graph.unspill()

        after = person_graph.cypher(
            "MATCH (n:Person) RETURN n.full_name, n.age, n.score, n.active ORDER BY n.age"
        ).to_list()
        assert before == after

    def test_multi_type_properties(self, multi_type_graph):
        kg = multi_type_graph
        persons_before = kg.cypher("MATCH (n:Person) RETURN n.full_name, n.age ORDER BY n.age").to_list()
        companies_before = kg.cypher(
            "MATCH (n:Company) RETURN n.company_name, n.employees ORDER BY n.employees"
        ).to_list()

        kg.unspill()

        persons_after = kg.cypher("MATCH (n:Person) RETURN n.full_name, n.age ORDER BY n.age").to_list()
        companies_after = kg.cypher(
            "MATCH (n:Company) RETURN n.company_name, n.employees ORDER BY n.employees"
        ).to_list()
        assert persons_before == persons_after
        assert companies_before == companies_after


# ── Cypher queries on columnar storage ───────────────────────────────────────


class TestCypherOverColumns:
    def test_where_int_filter(self, person_graph):
        result = person_graph.cypher(
            "MATCH (n:Person) WHERE n.age > 30 RETURN n.full_name ORDER BY n.full_name"
        ).to_list()
        names = [r["n.full_name"] for r in result]
        assert names == ["Charlie", "Eve"]

    def test_where_float_filter(self, person_graph):
        result = person_graph.cypher(
            "MATCH (n:Person) WHERE n.score >= 3.0 RETURN n.full_name ORDER BY n.full_name"
        ).to_list()
        names = [r["n.full_name"] for r in result]
        assert names == ["Charlie", "Diana"]

    def test_where_bool_filter(self, person_graph):
        result = person_graph.cypher(
            "MATCH (n:Person) WHERE n.active = true RETURN n.full_name ORDER BY n.full_name"
        ).to_list()
        names = [r["n.full_name"] for r in result]
        assert names == ["Alice", "Charlie", "Diana"]

    def test_where_string_equals(self, person_graph):
        result = person_graph.cypher("MATCH (n:Person) WHERE n.full_name = 'Bob' RETURN n.age").to_list()
        assert result == [{"n.age": 25}]

    def test_order_by_columnar(self, person_graph):
        result = person_graph.cypher("MATCH (n:Person) RETURN n.full_name ORDER BY n.score DESC").to_list()
        names = [r["n.full_name"] for r in result]
        assert names == ["Diana", "Charlie", "Bob", "Alice", "Eve"]

    def test_aggregation_on_columnar(self, person_graph):
        result = person_graph.cypher("MATCH (n:Person) RETURN count(n) AS cnt, avg(n.age) AS avg_age").to_list()
        assert result[0]["cnt"] == 5
        assert abs(result[0]["avg_age"] - 32.0) < 0.01

    def test_relationship_traversal_with_columnar(self, multi_type_graph):
        kg = multi_type_graph
        result = kg.cypher(
            "MATCH (p:Person)-[:WORKS_AT]->(c:Company) RETURN p.full_name, c.company_name ORDER BY p.full_name"
        ).to_list()
        assert len(result) == 2
        assert result[0]["p.full_name"] == "Alice"
        assert result[0]["c.company_name"] == "Acme"


# ── Save/Load ────────────────────────────────────────────────────────────────


class TestSaveLoad:
    def test_save_load_roundtrip(self, person_graph):
        before = person_graph.cypher("MATCH (n:Person) RETURN n.full_name, n.age, n.score ORDER BY n.age").to_list()

        with tempfile.TemporaryDirectory() as td:
            fp = os.path.join(td, "test.kgl")
            person_graph.save(fp)
            kg2 = kglite.load(fp)

        assert kg2.graph_info()["columnar_live_rows"] == 5

        after = kg2.cypher("MATCH (n:Person) RETURN n.full_name, n.age, n.score ORDER BY n.age").to_list()
        assert before == after

    def test_save_load_multi_type(self, multi_type_graph):
        kg = multi_type_graph

        with tempfile.TemporaryDirectory() as td:
            fp = os.path.join(td, "test.kgl")
            kg.save(fp)
            kg2 = kglite.load(fp)

        # Check both types survived
        persons = kg2.cypher("MATCH (n:Person) RETURN n.full_name ORDER BY n.full_name").to_list()
        companies = kg2.cypher("MATCH (n:Company) RETURN n.company_name ORDER BY n.company_name").to_list()
        assert [r["n.full_name"] for r in persons] == ["Alice", "Bob", "Charlie"]
        assert [r["n.company_name"] for r in companies] == ["Acme", "Globex"]

    def test_save_load_preserves_edges(self, multi_type_graph):
        kg = multi_type_graph

        with tempfile.TemporaryDirectory() as td:
            fp = os.path.join(td, "test.kgl")
            kg.save(fp)
            kg2 = kglite.load(fp)

        result = kg2.cypher(
            "MATCH (p:Person)-[:WORKS_AT]->(c:Company) RETURN p.full_name, c.company_name ORDER BY p.full_name"
        ).to_list()
        assert len(result) == 2


# ── Mutations on columnar storage ────────────────────────────────────────────


class TestMutations:
    def test_set_property_cypher(self, person_graph):
        person_graph.cypher("MATCH (n:Person) WHERE n.full_name = 'Alice' SET n.age = 99")
        result = person_graph.cypher("MATCH (n:Person) WHERE n.full_name = 'Alice' RETURN n.age").to_list()
        assert result == [{"n.age": 99}]

    def test_set_new_property_cypher(self, person_graph):
        person_graph.cypher("MATCH (n:Person) WHERE n.full_name = 'Bob' SET n.email = 'bob@test.com'")
        result = person_graph.cypher("MATCH (n:Person) WHERE n.full_name = 'Bob' RETURN n.email").to_list()
        assert result == [{"n.email": "bob@test.com"}]

    def test_remove_property_cypher(self, person_graph):
        person_graph.cypher("MATCH (n:Person) WHERE n.full_name = 'Charlie' REMOVE n.score")
        result = person_graph.cypher("MATCH (n:Person) WHERE n.full_name = 'Charlie' RETURN n.score").to_list()
        assert result == [{"n.score": None}]


# ── Node count and graph stats ───────────────────────────────────────────────


class TestStats:
    def test_node_count_unchanged(self, person_graph):
        count_before = person_graph.cypher("MATCH (n:Person) RETURN count(n) AS c").to_list()[0]["c"]
        person_graph.unspill()
        count_after = person_graph.cypher("MATCH (n:Person) RETURN count(n) AS c").to_list()[0]["c"]
        assert count_before == count_after == 5

    def test_graph_info_with_columnar(self, person_graph):
        info = person_graph.graph_info()
        assert info["node_count"] == 5


# ── V3 save/load roundtrip ──────────────────────────────────────────────────


class TestKglRoundtrip:
    def test_save_load_v3_basic(self, person_graph, tmp_path):
        """v3 save/load roundtrip preserves all data and loads as columnar."""
        before = person_graph.cypher("MATCH (n:Person) RETURN n.full_name, n.age, n.score ORDER BY n.age").to_list()

        fp = str(tmp_path / "test.kgl")
        person_graph.save(fp)

        kg2 = kglite.load(fp)
        assert kg2.graph_info()["columnar_live_rows"] == 5
        after = kg2.cypher("MATCH (n:Person) RETURN n.full_name, n.age, n.score ORDER BY n.age").to_list()
        assert before == after

    def test_save_load_v3_multi_type(self, multi_type_graph, tmp_path):
        """v3 roundtrip with multiple node types."""
        fp = str(tmp_path / "multi.kgl")
        multi_type_graph.save(fp)

        kg2 = kglite.load(fp)
        assert kg2.graph_info()["columnar_live_rows"] == 5
        persons = kg2.cypher("MATCH (n:Person) RETURN n.full_name ORDER BY n.full_name").to_list()
        companies = kg2.cypher("MATCH (n:Company) RETURN n.company_name ORDER BY n.company_name").to_list()
        assert [r["n.full_name"] for r in persons] == ["Alice", "Bob", "Charlie"]
        assert [r["n.company_name"] for r in companies] == ["Acme", "Globex"]

    def test_save_load_v3_preserves_edges(self, multi_type_graph, tmp_path):
        """v3 roundtrip preserves edges between node types."""
        fp = str(tmp_path / "edges.kgl")
        multi_type_graph.save(fp)

        kg2 = kglite.load(fp)
        result = kg2.cypher(
            "MATCH (p:Person)-[:WORKS_AT]->(c:Company) RETURN p.full_name, c.company_name ORDER BY p.full_name"
        ).to_list()
        assert len(result) == 2
        assert result[0]["p.full_name"] == "Alice"

    def test_v3_query_after_load(self, person_graph, tmp_path):
        """Queries work on v3-loaded graphs."""
        fp = str(tmp_path / "query.kgl")
        person_graph.save(fp)

        kg2 = kglite.load(fp)
        result = kg2.cypher("MATCH (n:Person) WHERE n.age > 30 RETURN n.full_name ORDER BY n.full_name").to_list()
        names = [r["n.full_name"] for r in result]
        assert names == ["Charlie", "Eve"]

    def test_current_magic_and_codec_bytes(self, person_graph, tmp_path):
        """v6 starts with RGF\\x06 and explicitly selects Postcard (tag 2)."""
        fp = str(tmp_path / "magic.kgl")
        person_graph.save(fp)

        with open(fp, "rb") as f:
            header = f.read(5)
        assert header == b"RGF\x06\x02"

    def test_save_does_not_change_the_storage_shape(self, person_graph, tmp_path):
        """A graph is columnar before its first save, after it, and on reload.

        This used to be ``test_save_auto_columnar``: it asserted that save()
        *converted* a non-columnar graph, which is precisely the shape change
        the convergence programme removed. What survives is the round-trip —
        the shape is the same on all three sides of a save/load, read through
        the only place it is still observable.
        """
        assert person_graph.graph_info()["columnar_live_rows"] == 5
        fp = str(tmp_path / "auto.kgl")
        person_graph.save(fp)
        assert person_graph.graph_info()["columnar_live_rows"] == 5
        kg2 = kglite.load(fp)
        assert kg2.graph_info()["columnar_live_rows"] == 5


# ── Temp directory cleanup ────────────────────────────────────────────────────


class TestTempDirCleanup:
    """The spill directory a portable load mmaps its big columns out of.

    Both tests used ``person_graph`` — five rows, whose every column packs
    into a few hundred bytes. Nothing in that load can clear the 256 KB spill
    threshold, so what they pinned was an *empty* directory being minted and
    removed, and a loader that stopped minting it would have failed them for
    doing the right thing. They spill for real now.
    """

    @staticmethod
    def _spilling_graph():
        """A graph whose ``Doc.body`` column packs well past 256 KB."""
        kg = kglite.KnowledgeGraph()
        df = pd.DataFrame(
            {
                "id": list(range(4000)),
                "body": ["x" * 128] * 4000,
            }
        )
        kg.add_nodes(df, "Doc", "id", "id")
        return kg

    def test_load_cleans_temp_dir_on_drop(self, tmp_path):
        """Temp dirs created during a spilling load are cleaned up on drop."""
        import gc
        import glob

        fp = str(tmp_path / "cleanup.kgl")
        self._spilling_graph().save(fp)

        # The pid-scoped pattern can also match graphs other tests left alive in
        # this process, so measure the DELTA this test creates, not an absolute
        # count.
        pid = os.getpid()
        pattern = os.path.join(tempfile.gettempdir(), f"kglite_portable_{pid}_*")
        baseline = set(glob.glob(pattern))

        kg2 = kglite.load(fp)
        assert kg2.graph_info()["node_count"] == 4000
        created = set(glob.glob(pattern)) - baseline
        assert created, f"Expected a new temp dir matching {pattern}"
        assert any(os.listdir(d) for d in created), (
            f"the load minted a directory but spilled nothing into it: {created}"
        )

        # Drop the graph — the dir(s) it created must be gone (gc.collect forces
        # the Rust Drop to run promptly).
        del kg2
        gc.collect()
        leaked = created & set(glob.glob(pattern))
        assert not leaked, f"Temp dirs leaked: {leaked}"

    def test_a_load_with_nothing_to_spill_creates_no_temp_dir(self, person_graph, tmp_path):
        """Five rows spill nothing, so they earn no directory in $TMPDIR.

        A load that mints a directory it never writes to still races every
        other load for that name, and still leaves the tree behind when the
        process is killed — which is how a downstream accumulated thousands of
        empty ones.
        """
        import glob

        fp = str(tmp_path / "small.kgl")
        person_graph.save(fp)
        assert os.path.getsize(fp) < 256 * 1024

        pid = os.getpid()
        pattern = os.path.join(tempfile.gettempdir(), f"kglite_portable_{pid}_*")
        baseline = set(glob.glob(pattern))

        kg2 = kglite.load(fp)
        assert kg2.graph_info()["node_count"] == 5
        created = set(glob.glob(pattern)) - baseline
        assert not created, f"a load with nothing to spill wrote to $TMPDIR: {created}"

    def test_multiple_loads_no_leak(self, tmp_path):
        """Multiple load/drop cycles don't accumulate temp dirs."""
        import gc
        import glob

        fp = str(tmp_path / "multi.kgl")
        self._spilling_graph().save(fp)

        pid = os.getpid()
        pattern = os.path.join(tempfile.gettempdir(), f"kglite_portable_{pid}_*")
        baseline = set(glob.glob(pattern))

        for _ in range(5):
            kg = kglite.load(fp)
            assert kg.graph_info()["node_count"] == 4000
            del kg
            gc.collect()

        # No dirs beyond what was already alive before this test remain.
        leaked = set(glob.glob(pattern)) - baseline
        assert not leaked, f"Temp dirs leaked after 5 cycles: {leaked}"


class TestIdTitleSentinel:
    """Consolidation nulls a node's inline id/title once they live in the
    column store, and every later rebuild must read them back out of it.

    Regression guard for the topology-bloat bug: a default-mode build that
    saves directly used to serialize id/title twice (inline in the topology
    *and* in the column section), inflating the saved file by ~27 B/node.
    A load round-trip nulled the inline copies, which is why load->save
    compacted. The fix nulls them at columnarize time so a fresh build
    matches the loaded form byte-for-byte.
    """

    @staticmethod
    def _topology_bytes(path: str) -> bytes:
        """Extract the compressed topology section from a v4 .kgl file.

        Layout: [0..4] magic, [4] codec tag, [9..13] metadata_len (u32 LE),
        [13..13+len] JSON metadata (carries ``topology_compressed_size``), then the topology
        section. The topology holds the node/edge structure incl. each node's
        inline id/title; it is the section the id/title-dedup fix shrinks, and
        it is deterministic (node order = insertion order, no zstd-ordering
        ambiguity), unlike the column sections.
        """
        import json
        import struct

        with open(path, "rb") as f:
            b = f.read()
        assert b[:5] == b"RGF\x06\x02"
        mlen = struct.unpack_from("<I", b, 9)[0]
        meta = json.loads(b[13 : 13 + mlen])
        start = 13 + mlen
        return b[start : start + meta["topology_compressed_size"]]

    def _create_graph(self):
        kg = kglite.KnowledgeGraph()
        kg.cypher("UNWIND range(1,2000) AS i CREATE (:N {id:i, name:'entity_'+toString(i), score:i%100})")
        return kg

    def test_topology_has_no_idtitle_duplication(self, tmp_path):
        """The core invariant: a fresh build's topology carries no inline
        id/title (they live only in the column section), so it is byte-identical
        to the topology of a load->save round-trip. Asserting on the topology
        section — not the whole file — keeps this independent of the (zstd-
        order-sensitive) column sections.
        """
        built = str(tmp_path / "built.kgl")
        resaved = str(tmp_path / "resaved.kgl")
        self._create_graph().save(built)
        kglite.load(built).save(resaved)
        tb, tr = self._topology_bytes(built), self._topology_bytes(resaved)
        assert tb == tr, (
            f"as-built topology ({len(tb)}B) != load->resave topology ({len(tr)}B): "
            "id/title duplication leaked into the topology section"
        )

    def test_save_is_deterministic(self, tmp_path):
        """Saving the same in-memory graph twice yields byte-identical files.

        Guards the deterministic schema/column ordering in the CREATE path:
        `properties` is a std HashMap whose iteration order is randomized per
        process, so without an explicit sort the saved column order — and thus
        the compressed bytes — would vary run to run (the original flaky-test
        cause). This is the anti-flake guarantee.
        """
        kg = self._create_graph()
        a = str(tmp_path / "a.kgl")
        b = str(tmp_path / "b.kgl")
        kg.save(a)
        kg.save(b)
        with open(a, "rb") as fa, open(b, "rb") as fb:
            assert fa.read() == fb.read(), (
                "saving the same graph twice produced different bytes — "
                "non-deterministic column/schema ordering regressed"
            )

    def test_rebuild_preserves_ids_on_loaded_graph(self, tmp_path):
        """A column rebuild must not lose id/title for null-sentinel nodes.

        A loaded node holds ``Null`` inline and gets its identity from the
        store, so a rebuild that reads nodes rather than the store drops every
        id silently. ``unspill()`` is the public rebuild; this used to run the
        same assertion through ``disable_columnar()``.
        """
        kg = kglite.KnowledgeGraph()
        kg.cypher("UNWIND range(1,500) AS i CREATE (:N {id:i, name:'n'+toString(i)})")
        fp = str(tmp_path / "g.kgl")
        kg.save(fp)
        loaded = kglite.load(fp)  # null-sentinel columnar nodes
        loaded.unspill()
        res = loaded.cypher("MATCH (n:N) RETURN count(n) AS total, count(n.id) AS with_id")
        row = res.to_dicts()[0] if hasattr(res, "to_dicts") else res
        assert row["total"] == 500
        assert row["with_id"] == 500, "the column rebuild dropped node ids"
        sample = loaded.cypher("MATCH (n:N) WHERE n.id = 7 RETURN n.id AS id, n.name AS nm").to_dicts()[0]
        assert sample["id"] == 7 and sample["nm"] == "n7"
