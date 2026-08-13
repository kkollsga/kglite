"""Tests for incremental columnar insertion in mapped mode (Phase 2A-2B)."""

import pandas as pd

from kglite import KnowledgeGraph


class TestIncrementalColumnar:
    """Nodes added in mapped mode should use columnar storage from the start."""

    def test_properties_accessible_after_add_nodes(self):
        graph = KnowledgeGraph(storage="mapped")
        df = pd.DataFrame(
            {
                "nid": list(range(50)),
                "name": [f"Node_{i}" for i in range(50)],
                "score": [float(i * 10) for i in range(50)],
            }
        )
        graph.add_nodes(df, "Item", "nid", "name")

        # Properties should be readable via Cypher
        r = graph.cypher("MATCH (n:Item {id: 25}) RETURN n.score").to_df()
        assert r["n.score"][0] == 250.0

    def test_multiple_node_types(self):
        graph = KnowledgeGraph(storage="mapped")
        df1 = pd.DataFrame({"nid": [1, 2], "name": ["A", "B"], "age": [30, 40]})
        df2 = pd.DataFrame({"cid": [10, 20], "title": ["X", "Y"], "size": [100, 200]})
        graph.add_nodes(df1, "Person", "nid", "name")
        graph.add_nodes(df2, "Company", "cid", "title")

        r1 = graph.cypher("MATCH (n:Person {id: 1}) RETURN n.age").to_df()
        assert r1["n.age"][0] == 30

        r2 = graph.cypher("MATCH (n:Company {id: 10}) RETURN n.size").to_df()
        assert r2["n.size"][0] == 100

    def test_incremental_add_nodes_same_type(self):
        graph = KnowledgeGraph(storage="mapped")
        df1 = pd.DataFrame({"nid": [1, 2], "name": ["A", "B"], "value": [10, 20]})
        df2 = pd.DataFrame({"nid": [3, 4], "name": ["C", "D"], "value": [30, 40]})
        graph.add_nodes(df1, "Item", "nid", "name")
        graph.add_nodes(df2, "Item", "nid", "name")

        assert graph.select("Item").len() == 4
        r = graph.cypher("MATCH (n:Item {id: 3}) RETURN n.value").to_df()
        assert r["n.value"][0] == 30

    def test_schema_extension(self):
        """Second batch adds a new property column — schema should extend."""
        graph = KnowledgeGraph(storage="mapped")
        df1 = pd.DataFrame({"nid": [1], "name": ["A"], "x": [10]})
        graph.add_nodes(df1, "Item", "nid", "name")

        df2 = pd.DataFrame({"nid": [2], "name": ["B"], "x": [20], "y": [30]})
        graph.add_nodes(df2, "Item", "nid", "name")

        # Both x and y should be accessible
        r = graph.cypher("MATCH (n:Item {id: 2}) RETURN n.x, n.y").to_df()
        assert r["n.x"][0] == 20
        assert r["n.y"][0] == 30

    def test_cypher_where_filter(self):
        graph = KnowledgeGraph(storage="mapped")
        df = pd.DataFrame(
            {
                "nid": list(range(100)),
                "name": [f"N{i}" for i in range(100)],
                "val": [float(i) for i in range(100)],
            }
        )
        graph.add_nodes(df, "Item", "nid", "name")
        r = graph.cypher("MATCH (n:Item) WHERE n.val > 90 RETURN count(n) AS c").to_df()
        assert r["c"][0] == 9

    def test_is_columnar_in_mapped_mode(self):
        graph = KnowledgeGraph(storage="mapped")
        df = pd.DataFrame({"nid": [1], "name": ["A"]})
        graph.add_nodes(df, "Item", "nid", "name")
        assert graph.is_columnar

    def test_mapped_construction_actually_maps(self):
        """An in-process mapped graph is file-backed once its ingest lands.

        ``storage="mapped"`` is ``memory_limit = 0``, and until the limit was
        enforced on the ingest path a mapped graph built in-process stayed
        wholly on the heap — only a *load* ever produced mapped columns. The
        heap assertion is what makes this more than a flag check: the flag reads
        ``any(column is mapped)``, so it can be True while most of the data is
        still resident.
        """
        graph = KnowledgeGraph(storage="mapped")
        rows = 20_000
        df = pd.DataFrame(
            {
                "nid": range(rows),
                "name": [f"N{i}" for i in range(rows)],
                "v": [float(i) for i in range(rows)],
            }
        )
        graph.add_nodes(df, "Item", "nid", "name")

        info = graph.graph_info()
        assert info["memory_limit"] == 0
        assert info["columnar_is_mapped"] is True
        # One byte per row of tombstone bitmap is the heap floor; the id, name
        # and value columns are all read through their mapping.
        assert info["columnar_heap_bytes"] <= 2 * rows, info["columnar_heap_bytes"]
        assert graph.cypher("MATCH (n:Item) WHERE n.v > 19998 RETURN count(n) AS c").to_list()[0]["c"] == 1

    def test_default_mode_is_columnar_from_construction(self):
        """Default mode is columnar from its first node — no save required.

        This test used to assert the opposite (``test_default_mode_not_columnar``).
        The behaviour it pinned was the shape split the convergence programme
        removed: a default-mode graph built row-shaped properties and only
        became columnar when ``save()`` rebuilt them, so every graph changed
        write regime the first time it was saved. Both storage modes now build
        the same shape, which is what makes the mapped assertion above a parity
        check rather than a mode difference.
        """
        graph = KnowledgeGraph()
        df = pd.DataFrame({"nid": [1], "name": ["A"]})
        graph.add_nodes(df, "Item", "nid", "name")
        assert graph.is_columnar

    def test_save_load_roundtrip_preserves_data(self, tmp_path):
        graph = KnowledgeGraph(storage="mapped")
        df = pd.DataFrame(
            {
                "nid": list(range(20)),
                "name": [f"N{i}" for i in range(20)],
                "score": [float(i * 5) for i in range(20)],
            }
        )
        graph.add_nodes(df, "Item", "nid", "name")

        path = str(tmp_path / "test_col.kgl")
        graph.save(path)

        from kglite import load

        loaded = load(path)
        assert loaded.select("Item").len() == 20
        r = loaded.cypher("MATCH (n:Item {id: 10}) RETURN n.score").to_df()
        assert r["n.score"][0] == 50.0


class TestNTriplesColumnar:
    """N-Triples loader in mapped mode should produce columnar nodes."""

    def test_ntriples_properties_in_mapped_mode(self):
        graph = KnowledgeGraph(storage="mapped")
        graph.load_ntriples(
            "tests/data/sample_wikidata.nt",
            languages=["en"],
        )
        # Cross-mode parity (0.11.0): id is the integer; query the string Q-code via nid
        r = graph.cypher('MATCH (n {nid: "Q42"}) RETURN n.description, n.P1082').to_df()
        assert r["n.description"][0] == "English author and humourist"
        assert r["n.P1082"][0] == 42

    def test_ntriples_edges_in_mapped_mode(self):
        graph = KnowledgeGraph(storage="mapped")
        graph.load_ntriples(
            "tests/data/sample_wikidata.nt",
            languages=["en"],
        )
        # Cross-mode parity (0.11.0): id is the integer; query the string Q-code via nid
        r = graph.cypher('MATCH (n {nid: "Q42"})-[:P27]->(m) RETURN m.title').to_df()
        assert len(r) == 1
        assert r["m.title"][0] == "United Kingdom"


class TestAppendingToALoadedDiskGraph:
    """A `CREATE` into a graph loaded from disk appends to a store that still
    reads through its mmap base, and that store must be made fully owned first.

    `push_id` / `push_title` build their overlay columns starting at row zero.
    Alongside a live mmap base those overlays are offset by the whole existing
    row count *and* shadow the mapped originals on every read, so a single
    `CREATE` re-points the first node's id and title at the new node's and
    blanks every other node's. Measured on a three-node fixture before the fix:
    `[(1, 'z'), (2, None), (3, None), (None, None)]` for what should be
    `[(1, 'a'), (2, 'b'), (3, 'c'), (9, 'z')]`.

    The bulk ingest funnel has always called `materialize_for_append`; the
    Cypher create funnel did not, and what kept the damage off this path was
    that the id column used to be untyped — which made the unified disk-column
    writer skip the type entirely rather than serialize the misalignment.
    Typing the id column (so it can be spilled) removed that accident, so the
    invariant is now enforced where it belongs.
    """

    def _seeded(self, tmp_path):
        path = str(tmp_path / "g")
        graph = KnowledgeGraph(storage="disk", path=path)
        graph.add_nodes(
            pd.DataFrame({"id": [1, 2, 3], "name": ["a", "b", "c"], "v": [10, 20, 30]}),
            "Item",
            "id",
            "name",
        )
        graph.save(path)
        del graph
        return path

    @staticmethod
    def _rows(graph):
        return sorted(
            (r["i"], r["t"], r["v"])
            for r in graph.cypher("MATCH (n:Item) RETURN n.id AS i, n.title AS t, n.v AS v").to_list()
        )

    def test_create_does_not_disturb_the_loaded_rows(self, tmp_path):
        import kglite

        path = self._seeded(tmp_path)
        graph = kglite.open(path)
        graph.cypher("CREATE (:Item {id: 9, name: 'z', v: 90})")

        assert self._rows(graph) == [(1, "a", 10), (2, "b", 20), (3, "c", 30), (9, "z", 90)]

        graph.save()
        del graph
        assert self._rows(kglite.load(path)) == [
            (1, "a", 10),
            (2, "b", 20),
            (3, "c", 30),
            (9, "z", 90),
        ]

    def test_create_with_a_new_property_grows_the_schema_in_place(self, tmp_path):
        """The same funnel, with the appended-column path also exercised.

        `fresh` has to be *declared* for the `CREATE` to name it: the planner's
        unknown-property guard is a deliberate typo-guard. What changed is that
        declaring it through `define_schema` now lifts the guard — until this
        phase the declaration was ignored, because the guard read only the
        property metadata accumulated from values already written, so a Cypher
        statement stream could not grow a type's schema at all.
        """
        import kglite

        path = self._seeded(tmp_path)
        graph = kglite.open(path)
        graph.define_schema({"nodes": {"Item": {"optional": ["fresh"]}}})
        graph.cypher("CREATE (:Item {id: 9, name: 'z', v: 90, fresh: 5})")

        assert self._rows(graph) == [(1, "a", 10), (2, "b", 20), (3, "c", 30), (9, "z", 90)]
        assert graph.cypher("MATCH (n:Item {id: 9}) RETURN n.fresh AS f").scalar() == 5
        # A property the older rows never carried reads absent, not backfilled.
        assert graph.cypher("MATCH (n:Item {id: 1}) RETURN n.fresh AS f").scalar() is None

        graph.save()
        del graph
        reloaded = kglite.load(path)
        assert self._rows(reloaded) == [(1, "a", 10), (2, "b", 20), (3, "c", 30), (9, "z", 90)]
        assert reloaded.cypher("MATCH (n:Item {id: 9}) RETURN n.fresh AS f").scalar() == 5
