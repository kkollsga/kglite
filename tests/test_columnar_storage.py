"""Integration tests for columnar property storage (Phase B).

Tests verify that enable_columnar() / disable_columnar() preserve
property semantics across Cypher queries, mutations, save/load, and
bulk operations.
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


# ── Basic enable/disable ─────────────────────────────────────────────────────


class TestColumnarBasic:
    def test_enable_disable_flag(self, person_graph):
        assert not person_graph.is_columnar
        person_graph.enable_columnar()
        assert person_graph.is_columnar
        person_graph.disable_columnar()
        assert not person_graph.is_columnar

    def test_enable_idempotent(self, person_graph):
        person_graph.enable_columnar()
        person_graph.enable_columnar()  # second call should be safe
        assert person_graph.is_columnar

    def test_disable_on_non_columnar(self, person_graph):
        person_graph.disable_columnar()  # no-op, should not crash
        assert not person_graph.is_columnar


# ── Property preservation ────────────────────────────────────────────────────


class TestColumnarPropertyPreservation:
    def test_all_properties_survive_enable(self, person_graph):
        before = person_graph.cypher(
            "MATCH (n:Person) RETURN n.full_name, n.age, n.score, n.active ORDER BY n.age"
        ).to_list()

        person_graph.enable_columnar()

        after = person_graph.cypher(
            "MATCH (n:Person) RETURN n.full_name, n.age, n.score, n.active ORDER BY n.age"
        ).to_list()
        assert before == after

    def test_roundtrip_enable_disable(self, person_graph):
        before = person_graph.cypher("MATCH (n:Person) RETURN n.full_name, n.age, n.score ORDER BY n.age").to_list()

        person_graph.enable_columnar()
        person_graph.disable_columnar()

        after = person_graph.cypher("MATCH (n:Person) RETURN n.full_name, n.age, n.score ORDER BY n.age").to_list()
        assert before == after

    def test_multi_type_properties(self, multi_type_graph):
        kg = multi_type_graph
        persons_before = kg.cypher("MATCH (n:Person) RETURN n.full_name, n.age ORDER BY n.age").to_list()
        companies_before = kg.cypher(
            "MATCH (n:Company) RETURN n.company_name, n.employees ORDER BY n.employees"
        ).to_list()

        kg.enable_columnar()

        persons_after = kg.cypher("MATCH (n:Person) RETURN n.full_name, n.age ORDER BY n.age").to_list()
        companies_after = kg.cypher(
            "MATCH (n:Company) RETURN n.company_name, n.employees ORDER BY n.employees"
        ).to_list()
        assert persons_before == persons_after
        assert companies_before == companies_after


# ── Cypher queries on columnar storage ───────────────────────────────────────


class TestColumnarCypher:
    def test_where_int_filter(self, person_graph):
        person_graph.enable_columnar()
        result = person_graph.cypher(
            "MATCH (n:Person) WHERE n.age > 30 RETURN n.full_name ORDER BY n.full_name"
        ).to_list()
        names = [r["n.full_name"] for r in result]
        assert names == ["Charlie", "Eve"]

    def test_where_float_filter(self, person_graph):
        person_graph.enable_columnar()
        result = person_graph.cypher(
            "MATCH (n:Person) WHERE n.score >= 3.0 RETURN n.full_name ORDER BY n.full_name"
        ).to_list()
        names = [r["n.full_name"] for r in result]
        assert names == ["Charlie", "Diana"]

    def test_where_bool_filter(self, person_graph):
        person_graph.enable_columnar()
        result = person_graph.cypher(
            "MATCH (n:Person) WHERE n.active = true RETURN n.full_name ORDER BY n.full_name"
        ).to_list()
        names = [r["n.full_name"] for r in result]
        assert names == ["Alice", "Charlie", "Diana"]

    def test_where_string_equals(self, person_graph):
        person_graph.enable_columnar()
        result = person_graph.cypher("MATCH (n:Person) WHERE n.full_name = 'Bob' RETURN n.age").to_list()
        assert result == [{"n.age": 25}]

    def test_order_by_columnar(self, person_graph):
        person_graph.enable_columnar()
        result = person_graph.cypher("MATCH (n:Person) RETURN n.full_name ORDER BY n.score DESC").to_list()
        names = [r["n.full_name"] for r in result]
        assert names == ["Diana", "Charlie", "Bob", "Alice", "Eve"]

    def test_aggregation_on_columnar(self, person_graph):
        person_graph.enable_columnar()
        result = person_graph.cypher("MATCH (n:Person) RETURN count(n) AS cnt, avg(n.age) AS avg_age").to_list()
        assert result[0]["cnt"] == 5
        assert abs(result[0]["avg_age"] - 32.0) < 0.01

    def test_relationship_traversal_with_columnar(self, multi_type_graph):
        kg = multi_type_graph
        kg.enable_columnar()
        result = kg.cypher(
            "MATCH (p:Person)-[:WORKS_AT]->(c:Company) RETURN p.full_name, c.company_name ORDER BY p.full_name"
        ).to_list()
        assert len(result) == 2
        assert result[0]["p.full_name"] == "Alice"
        assert result[0]["c.company_name"] == "Acme"


# ── Save/Load ────────────────────────────────────────────────────────────────


class TestColumnarSaveLoad:
    def test_save_load_roundtrip(self, person_graph):
        person_graph.enable_columnar()
        before = person_graph.cypher("MATCH (n:Person) RETURN n.full_name, n.age, n.score ORDER BY n.age").to_list()

        with tempfile.TemporaryDirectory() as td:
            fp = os.path.join(td, "test.kgl")
            person_graph.save(fp)
            kg2 = kglite.load(fp)

        # v3 files always load as columnar
        assert kg2.is_columnar

        after = kg2.cypher("MATCH (n:Person) RETURN n.full_name, n.age, n.score ORDER BY n.age").to_list()
        assert before == after

    def test_save_load_multi_type(self, multi_type_graph):
        kg = multi_type_graph
        kg.enable_columnar()

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
        kg.enable_columnar()

        with tempfile.TemporaryDirectory() as td:
            fp = os.path.join(td, "test.kgl")
            kg.save(fp)
            kg2 = kglite.load(fp)

        result = kg2.cypher(
            "MATCH (p:Person)-[:WORKS_AT]->(c:Company) RETURN p.full_name, c.company_name ORDER BY p.full_name"
        ).to_list()
        assert len(result) == 2


# ── Mutations on columnar storage ────────────────────────────────────────────


class TestColumnarMutations:
    def test_set_property_cypher(self, person_graph):
        person_graph.enable_columnar()
        person_graph.cypher("MATCH (n:Person) WHERE n.full_name = 'Alice' SET n.age = 99")
        result = person_graph.cypher("MATCH (n:Person) WHERE n.full_name = 'Alice' RETURN n.age").to_list()
        assert result == [{"n.age": 99}]

    def test_set_new_property_cypher(self, person_graph):
        person_graph.enable_columnar()
        person_graph.cypher("MATCH (n:Person) WHERE n.full_name = 'Bob' SET n.email = 'bob@test.com'")
        result = person_graph.cypher("MATCH (n:Person) WHERE n.full_name = 'Bob' RETURN n.email").to_list()
        assert result == [{"n.email": "bob@test.com"}]

    def test_remove_property_cypher(self, person_graph):
        person_graph.enable_columnar()
        person_graph.cypher("MATCH (n:Person) WHERE n.full_name = 'Charlie' REMOVE n.score")
        result = person_graph.cypher("MATCH (n:Person) WHERE n.full_name = 'Charlie' RETURN n.score").to_list()
        assert result == [{"n.score": None}]


# ── Node count and graph stats ───────────────────────────────────────────────


class TestColumnarStats:
    def test_node_count_unchanged(self, person_graph):
        count_before = person_graph.cypher("MATCH (n:Person) RETURN count(n) AS c").to_list()[0]["c"]
        person_graph.enable_columnar()
        count_after = person_graph.cypher("MATCH (n:Person) RETURN count(n) AS c").to_list()[0]["c"]
        assert count_before == count_after == 5

    def test_graph_info_with_columnar(self, person_graph):
        person_graph.enable_columnar()
        info = person_graph.graph_info()
        assert info["node_count"] == 5


# ── V3 save/load roundtrip ──────────────────────────────────────────────────


class TestV3Roundtrip:
    def test_save_load_v3_basic(self, person_graph, tmp_path):
        """v3 save/load roundtrip preserves all data and loads as columnar."""
        before = person_graph.cypher("MATCH (n:Person) RETURN n.full_name, n.age, n.score ORDER BY n.age").to_list()

        fp = str(tmp_path / "test.kgl")
        person_graph.save(fp)

        kg2 = kglite.load(fp)
        assert kg2.is_columnar
        after = kg2.cypher("MATCH (n:Person) RETURN n.full_name, n.age, n.score ORDER BY n.age").to_list()
        assert before == after

    def test_save_load_v3_multi_type(self, multi_type_graph, tmp_path):
        """v3 roundtrip with multiple node types."""
        fp = str(tmp_path / "multi.kgl")
        multi_type_graph.save(fp)

        kg2 = kglite.load(fp)
        assert kg2.is_columnar
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
        """v5 starts with RGF\\x05 and explicitly selects Postcard (tag 2)."""
        fp = str(tmp_path / "magic.kgl")
        person_graph.save(fp)

        with open(fp, "rb") as f:
            header = f.read(5)
        assert header == b"RGF\x05\x02"

    def test_save_auto_columnar(self, person_graph, tmp_path):
        """save() auto-enables columnar for non-columnar graphs (stays columnar)."""
        assert not person_graph.is_columnar
        fp = str(tmp_path / "auto.kgl")
        person_graph.save(fp)
        # Graph is now columnar after save (no disable step)
        assert person_graph.is_columnar
        # Loaded graph is also columnar
        kg2 = kglite.load(fp)
        assert kg2.is_columnar


# ── Temp directory cleanup ────────────────────────────────────────────────────


class TestTempDirCleanup:
    def test_load_cleans_temp_dir_on_drop(self, person_graph, tmp_path):
        """Temp dirs created during portable load are cleaned up on drop."""
        import gc
        import glob

        fp = str(tmp_path / "cleanup.kgl")
        person_graph.save(fp)

        # The pid-scoped pattern can also match graphs other tests left alive in
        # this process, so measure the DELTA this test creates, not an absolute
        # count.
        pid = os.getpid()
        pattern = os.path.join(tempfile.gettempdir(), f"kglite_portable_{pid}_*")
        baseline = set(glob.glob(pattern))

        kg2 = kglite.load(fp)
        assert kg2.is_columnar
        created = set(glob.glob(pattern)) - baseline
        assert created, f"Expected a new temp dir matching {pattern}"

        # Drop the graph — the dir(s) it created must be gone (gc.collect forces
        # the Rust Drop to run promptly).
        del kg2
        gc.collect()
        leaked = created & set(glob.glob(pattern))
        assert not leaked, f"Temp dirs leaked: {leaked}"

    def test_multiple_loads_no_leak(self, person_graph, tmp_path):
        """Multiple load/drop cycles don't accumulate temp dirs."""
        import gc
        import glob

        fp = str(tmp_path / "multi.kgl")
        person_graph.save(fp)

        pid = os.getpid()
        pattern = os.path.join(tempfile.gettempdir(), f"kglite_portable_{pid}_*")
        baseline = set(glob.glob(pattern))

        for _ in range(5):
            kg = kglite.load(fp)
            assert kg.is_columnar
            del kg
            gc.collect()

        # No dirs beyond what was already alive before this test remain.
        leaked = set(glob.glob(pattern)) - baseline
        assert not leaked, f"Temp dirs leaked after 5 cycles: {leaked}"


class TestIdTitleSentinel:
    """enable_columnar() must null inline id/title once they live in the
    column store, and disable_columnar() must restore them from the store.

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
        assert b[:5] == b"RGF\x05\x02"
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

    def test_disable_columnar_preserves_ids_on_loaded_graph(self, tmp_path):
        """disable_columnar() must not lose id/title for null-sentinel nodes."""
        kg = kglite.KnowledgeGraph()
        kg.cypher("UNWIND range(1,500) AS i CREATE (:N {id:i, name:'n'+toString(i)})")
        fp = str(tmp_path / "g.kgl")
        kg.save(fp)
        loaded = kglite.load(fp)  # null-sentinel columnar nodes
        loaded.disable_columnar()
        res = loaded.cypher("MATCH (n:N) RETURN count(n) AS total, count(n.id) AS with_id")
        row = res.to_dicts()[0] if hasattr(res, "to_dicts") else res
        assert row["total"] == 500
        assert row["with_id"] == 500, "disable_columnar dropped node ids"
        sample = loaded.cypher("MATCH (n:N) WHERE n.id = 7 RETURN n.id AS id, n.name AS nm").to_dicts()[0]
        assert sample["id"] == 7 and sample["nm"] == "n7"
