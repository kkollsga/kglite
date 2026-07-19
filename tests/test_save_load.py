"""Tests for save/load persistence."""

import os
import tempfile

import pandas as pd
import pytest

import kglite
from kglite import KnowledgeGraph


class TestBasicSaveLoad:
    def test_save_and_load(self, small_graph):
        with tempfile.NamedTemporaryFile(suffix=".kgl", delete=False) as f:
            path = f.name
        try:
            small_graph.save(path)
            loaded = kglite.load(path)
            assert loaded.select("Person").len() == 3
        finally:
            os.unlink(path)

    def test_save_load_preserves_properties(self, small_graph):
        with tempfile.NamedTemporaryFile(suffix=".kgl", delete=False) as f:
            path = f.name
        try:
            small_graph.save(path)
            loaded = kglite.load(path)
            alice = loaded.node("Person", 1)
            assert alice["title"] == "Alice"
            assert alice["age"] == 28
        finally:
            os.unlink(path)

    def test_save_load_preserves_connections(self, small_graph):
        with tempfile.NamedTemporaryFile(suffix=".kgl", delete=False) as f:
            path = f.name
        try:
            small_graph.save(path)
            loaded = kglite.load(path)
            alice = loaded.select("Person").where({"title": "Alice"})
            friends = alice.traverse(connection_type="KNOWS", direction="outgoing")
            assert friends.len() == 2
        finally:
            os.unlink(path)


class TestSaveLoadWithFeatures:
    def test_save_load_with_indexes(self):
        graph = KnowledgeGraph()
        df = pd.DataFrame(
            {
                "id": list(range(10)),
                "name": [f"N_{i}" for i in range(10)],
                "cat": [f"C_{i % 3}" for i in range(10)],
            }
        )
        graph.add_nodes(df, "Node", "id", "name")
        graph.create_index("Node", "cat")

        with tempfile.NamedTemporaryFile(suffix=".kgl", delete=False) as f:
            path = f.name
        try:
            graph.save(path)
            loaded = kglite.load(path)
            assert loaded.has_index("Node", "cat")
        finally:
            os.unlink(path)

    def test_save_load_with_schema(self):
        graph = KnowledgeGraph()
        df = pd.DataFrame({"id": [1], "name": ["A"]})
        graph.add_nodes(df, "Node", "id", "name")
        graph.define_schema({"nodes": {"Node": {"required": ["id", "title"]}}})

        with tempfile.NamedTemporaryFile(suffix=".kgl", delete=False) as f:
            path = f.name
        try:
            graph.save(path)
            loaded = kglite.load(path)
            assert loaded.has_schema()
        finally:
            os.unlink(path)

    def test_save_load_large(self):
        graph = KnowledgeGraph()
        df = pd.DataFrame(
            {
                "id": list(range(1000)),
                "name": [f"Node_{i}" for i in range(1000)],
                "value": list(range(1000)),
            }
        )
        graph.add_nodes(df, "LargeType", "id", "name")

        with tempfile.NamedTemporaryFile(suffix=".kgl", delete=False) as f:
            path = f.name
        try:
            graph.save(path)
            loaded = kglite.load(path)
            assert loaded.select("LargeType").len() == 1000
        finally:
            os.unlink(path)


class TestV3Format:
    """Tests for the columnar binary format.

    Class name kept as `TestV3Format` for git-blame continuity; Phase A.1 / C5
    introduced v4; the current v5 header adds an explicit Postcard codec tag.
    """

    def test_v3_magic_bytes(self):
        """Saved files should start with v5 magic and Postcard codec tag 2."""
        graph = KnowledgeGraph()
        df = pd.DataFrame({"id": [1], "name": ["A"]})
        graph.add_nodes(df, "Node", "id", "name")

        with tempfile.NamedTemporaryFile(suffix=".kgl", delete=False) as f:
            path = f.name
        try:
            graph.save(path)
            with open(path, "rb") as f:
                header = f.read(5)
            assert header == b"RGF\x05\x02", f"Expected v5/Postcard header, got {header!r}"
        finally:
            os.unlink(path)

    def test_v3_header_structure(self):
        """Verify magic + codec + core_version + metadata_length."""
        import struct

        graph = KnowledgeGraph()
        df = pd.DataFrame({"id": [1], "name": ["A"]})
        graph.add_nodes(df, "Node", "id", "name")

        with tempfile.NamedTemporaryFile(suffix=".kgl", delete=False) as f:
            path = f.name
        try:
            graph.save(path)
            with open(path, "rb") as f:
                magic = f.read(4)
                codec = f.read(1)
                core_version = struct.unpack("<I", f.read(4))[0]
                metadata_len = struct.unpack("<I", f.read(4))[0]

            assert magic == b"RGF\x05"
            assert codec == b"\x02"
            # 2 — Phase A.1 (Value enum); 3 — 0.10.29 (embedding model_id + text_hashes)
            assert core_version == 3
            assert metadata_len > 0  # metadata should not be empty
        finally:
            os.unlink(path)

    def test_v3_metadata_is_json(self):
        """The metadata section should be valid JSON."""
        import json
        import struct

        graph = KnowledgeGraph()
        df = pd.DataFrame({"id": [1, 2], "name": ["A", "B"], "val": [10, 20]})
        graph.add_nodes(df, "Node", "id", "name")

        with tempfile.NamedTemporaryFile(suffix=".kgl", delete=False) as f:
            path = f.name
        try:
            graph.save(path)
            with open(path, "rb") as f:
                f.read(4)  # magic
                f.read(1)  # codec
                f.read(4)  # core_version
                metadata_len = struct.unpack("<I", f.read(4))[0]
                metadata_bytes = f.read(metadata_len)

            metadata = json.loads(metadata_bytes)
            assert "core_data_version" in metadata
            assert "library_version" in metadata
            assert "node_type_metadata" in metadata
            assert "Node" in metadata["node_type_metadata"]
        finally:
            os.unlink(path)

    def test_v3_preserves_node_type_metadata(self):
        """Node type metadata should survive save/load cycle."""
        graph = KnowledgeGraph()
        df = pd.DataFrame({"id": [1], "name": ["A"], "city": ["Oslo"]})
        graph.add_nodes(df, "Person", "id", "name")

        with tempfile.NamedTemporaryFile(suffix=".kgl", delete=False) as f:
            path = f.name
        try:
            graph.save(path)
            loaded = kglite.load(path)
            info = loaded.graph_info()
            assert info["type_count"] >= 1
        finally:
            os.unlink(path)

    def test_v3_preserves_connection_type_metadata(self):
        """Connection type metadata should survive save/load."""
        graph = KnowledgeGraph()
        persons = pd.DataFrame({"id": [1, 2], "name": ["A", "B"]})
        graph.add_nodes(persons, "Person", "id", "name")
        conns = pd.DataFrame({"source_id": [1], "target_id": [2], "weight": [1.5]})
        graph.add_connections(conns, "KNOWS", "Person", "source_id", "Person", "target_id")

        with tempfile.NamedTemporaryFile(suffix=".kgl", delete=False) as f:
            path = f.name
        try:
            graph.save(path)
            loaded = kglite.load(path)
            # Connection type should be recognized
            info = loaded.graph_info()
            assert info["edge_count"] == 1
        finally:
            os.unlink(path)

    def test_v3_preserves_indexes(self):
        """Property and composite indexes should survive v3 save/load."""
        graph = KnowledgeGraph()
        df = pd.DataFrame(
            {
                "id": list(range(20)),
                "name": [f"N_{i}" for i in range(20)],
                "cat": [f"C_{i % 3}" for i in range(20)],
                "sub": [f"S_{i % 5}" for i in range(20)],
            }
        )
        graph.add_nodes(df, "Item", "id", "name")
        graph.create_index("Item", "cat")
        graph.create_composite_index("Item", ["cat", "sub"])

        with tempfile.NamedTemporaryFile(suffix=".kgl", delete=False) as f:
            path = f.name
        try:
            graph.save(path)
            loaded = kglite.load(path)
            assert loaded.has_index("Item", "cat")
            assert loaded.has_composite_index("Item", ["cat", "sub"])
        finally:
            os.unlink(path)

    def test_v3_preserves_schema(self):
        """Schema definitions should survive v3 save/load."""
        graph = KnowledgeGraph()
        df = pd.DataFrame({"id": [1], "name": ["A"]})
        graph.add_nodes(df, "Node", "id", "name")
        graph.define_schema({"nodes": {"Node": {"required": ["id", "title"]}}})

        with tempfile.NamedTemporaryFile(suffix=".kgl", delete=False) as f:
            path = f.name
        try:
            graph.save(path)
            loaded = kglite.load(path)
            assert loaded.has_schema()
        finally:
            os.unlink(path)

    def test_v3_roundtrip_full(self):
        """Full roundtrip: nodes, edges, indexes, schema, Cypher queries."""
        graph = KnowledgeGraph()
        persons = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "name": ["Alice", "Bob", "Charlie"],
                "age": [30, 25, 35],
            }
        )
        graph.add_nodes(persons, "Person", "id", "name")
        conns = pd.DataFrame(
            {
                "source_id": [1, 2],
                "target_id": [2, 3],
            }
        )
        graph.add_connections(conns, "KNOWS", "Person", "source_id", "Person", "target_id")
        graph.create_index("Person", "age")

        with tempfile.NamedTemporaryFile(suffix=".kgl", delete=False) as f:
            path = f.name
        try:
            graph.save(path)
            loaded = kglite.load(path)

            # Check nodes
            assert loaded.len() == 3
            alice = loaded.node("Person", 1)
            assert alice["title"] == "Alice"
            assert alice["age"] == 30

            # Check edges
            assert loaded.graph_info()["edge_count"] == 2

            # Check index survived
            assert loaded.has_index("Person", "age")

            # Check Cypher works
            result = loaded.cypher("MATCH (p:Person) RETURN p.name ORDER BY p.name")
            names = [r["p.name"] for r in result]
            assert names == ["Alice", "Bob", "Charlie"]
        finally:
            os.unlink(path)

    def test_corrupt_file_error(self):
        """Loading a corrupt/random file should give a helpful error."""
        with tempfile.NamedTemporaryFile(suffix=".kgl", delete=False) as f:
            f.write(b"this is not a valid rusty-graph file")
            path = f.name
        try:
            with pytest.raises(Exception, match="(?i)(unrecognized|format|rebuild)"):
                kglite.load(path)
        finally:
            os.unlink(path)

    def test_truncated_current_file_error(self):
        """A truncated v5 file should give a clear error."""
        with tempfile.NamedTemporaryFile(suffix=".kgl", delete=False) as f:
            # v5 magic + Postcard tag + core version, but no metadata length.
            f.write(b"RGF\x05\x02\x03\x00\x00\x00")
            path = f.name
        try:
            with pytest.raises(Exception, match="(?i)(truncated|incomplete|failed)"):
                kglite.load(path)
        finally:
            os.unlink(path)

    def test_future_core_version_error(self):
        """A file with a future core_data_version should give a helpful upgrade message."""
        import struct

        # v5 + Postcard. Future core_data_version=99 still trips the
        # in-loader "supports up to version N" error.
        metadata = b"{}"
        header = b"RGF\x05\x02"
        header += struct.pack("<I", 99)  # future core version
        header += struct.pack("<I", len(metadata))

        with tempfile.NamedTemporaryFile(suffix=".kgl", delete=False) as f:
            f.write(header)
            f.write(metadata)
            f.write(b"\x00" * 20)  # dummy graph data
            path = f.name
        try:
            with pytest.raises(Exception, match="(?i)(upgrade|version 99|supports up to)"):
                kglite.load(path)
        finally:
            os.unlink(path)

    def test_empty_file_error(self):
        """An empty file should give a clear error."""
        with tempfile.NamedTemporaryFile(suffix=".kgl", delete=False) as f:
            path = f.name
        try:
            with pytest.raises(Exception, match="(?i)(too small|failed|empty)"):
                kglite.load(path)
        finally:
            os.unlink(path)

    def test_header_only_corrupt(self):
        """A file with valid header bytes but corrupted body must fail cleanly."""
        graph = KnowledgeGraph()
        graph.add_nodes(pd.DataFrame({"id": [1, 2], "name": ["A", "B"]}), "Person", "id", "name")
        with tempfile.NamedTemporaryFile(suffix=".kgl", delete=False) as f:
            path = f.name
        try:
            graph.save(path)
            with open(path, "rb") as f:
                data = f.read()
            # Flip a byte deep in the payload — header + metadata survive.
            bad = bytearray(data)
            bad[len(data) // 2] ^= 0xFF
            with open(path, "wb") as f:
                f.write(bad)
            # Load must either succeed with valid data or raise cleanly —
            # never segfault, never silent corruption.
            try:
                loaded = kglite.load(path)
                # If load succeeded, observable queries must not panic.
                list(loaded.cypher("MATCH (n) RETURN count(n) AS c"))
            except Exception:
                pass  # expected path
        finally:
            if os.path.exists(path):
                os.unlink(path)

    @pytest.mark.skipif(
        not hasattr(__import__("resource"), "RLIMIT_FSIZE"),
        reason="RLIMIT_FSIZE not available on this platform",
    )
    def test_disk_full_during_save(self, tmp_path):
        """Simulate disk-full via RLIMIT_FSIZE; save must fail cleanly and not
        corrupt any pre-existing valid save at a different path."""
        import resource

        safe = str(tmp_path / "safe.kgl")
        victim = str(tmp_path / "victim.kgl")

        graph = KnowledgeGraph()
        graph.add_nodes(
            pd.DataFrame({"id": list(range(50)), "name": [f"N{i}" for i in range(50)]}),
            "Person",
            "id",
            "name",
        )
        # Baseline: a valid save at `safe` before the limit kicks in.
        graph.save(safe)
        assert os.path.getsize(safe) > 0

        # Clamp file size below what we expect `victim` to need; 256 bytes
        # is well under the ~800 byte minimum observed for small graphs.
        original = resource.getrlimit(resource.RLIMIT_FSIZE)
        resource.setrlimit(resource.RLIMIT_FSIZE, (256, original[1]))
        try:
            # Save may succeed if graph is tiny, or raise an OS-level error.
            # What must NOT happen: corruption of `safe` or a Python-side
            # state mutation (graph still queryable after).
            try:
                graph.save(victim)
            except Exception:
                pass
            # Graph is still usable after the failed save.
            assert graph.cypher("MATCH (n:Person) RETURN count(n) AS c").to_list()[0]["c"] == 50
        finally:
            resource.setrlimit(resource.RLIMIT_FSIZE, original)

        # The earlier `safe` file is unchanged and still loads.
        reloaded = kglite.load(safe)
        assert reloaded.cypher("MATCH (n:Person) RETURN count(n) AS c").to_list()[0]["c"] == 50

    def test_node_count_after_load(self):
        """node_count() on a freshly loaded graph should return total count."""
        graph = KnowledgeGraph()
        df = pd.DataFrame({"id": list(range(50)), "name": [f"N_{i}" for i in range(50)]})
        graph.add_nodes(df, "Thing", "id", "name")

        with tempfile.NamedTemporaryFile(suffix=".kgl", delete=False) as f:
            path = f.name
        try:
            graph.save(path)
            loaded = kglite.load(path)
            assert loaded.len() == 50
        finally:
            os.unlink(path)


class TestIncrementalSaveLoad:
    """Regression tests for incremental build workflows: load → modify → save → load."""

    def test_load_save_load_no_changes(self):
        """Save → load → save → load without changes must not corrupt the file."""
        graph = KnowledgeGraph()
        for i in range(25):
            df = pd.DataFrame([{"id": f"a_{i}", "name": f"Artist {i}", "plays": i * 100}])
            graph.add_nodes(df, "Artist", "id", "name", conflict_handling="update")

        with tempfile.NamedTemporaryFile(suffix=".kgl", delete=False) as f:
            path1 = f.name
        with tempfile.NamedTemporaryFile(suffix=".kgl", delete=False) as f:
            path2 = f.name
        try:
            graph.save(path1)
            loaded = kglite.load(path1)
            loaded.save(path2)
            reloaded = kglite.load(path2)

            assert reloaded.select("Artist").len() == 25
            node = reloaded.node("Artist", "a_12")
            assert node["plays"] == 1200
        finally:
            os.unlink(path1)
            os.unlink(path2)

    def test_load_add_save_load(self):
        """Load → add new nodes → save → load must preserve all properties."""
        graph = KnowledgeGraph()
        df = pd.DataFrame([{"id": f"a_{i}", "name": f"Artist {i}", "plays": i * 100} for i in range(25)])
        graph.add_nodes(df, "Artist", "id", "name")

        with tempfile.NamedTemporaryFile(suffix=".kgl", delete=False) as f:
            path1 = f.name
        with tempfile.NamedTemporaryFile(suffix=".kgl", delete=False) as f:
            path2 = f.name
        try:
            graph.save(path1)
            loaded = kglite.load(path1)

            # Add 75 more artists to the loaded graph
            df2 = pd.DataFrame([{"id": f"a_{i}", "name": f"Artist {i}", "plays": i * 100} for i in range(25, 100)])
            loaded.add_nodes(df2, "Artist", "id", "name")
            loaded.save(path2)

            reloaded = kglite.load(path2)
            assert reloaded.select("Artist").len() == 100

            # Check original nodes
            node0 = reloaded.node("Artist", "a_0")
            assert node0["plays"] == 0
            node24 = reloaded.node("Artist", "a_24")
            assert node24["plays"] == 2400

            # Check new nodes
            node25 = reloaded.node("Artist", "a_25")
            assert node25["plays"] == 2500
            node99 = reloaded.node("Artist", "a_99")
            assert node99["plays"] == 9900
        finally:
            os.unlink(path1)
            os.unlink(path2)

    def test_load_update_save_load(self):
        """Load → update existing nodes with new property → save → load."""
        graph = KnowledgeGraph()
        df = pd.DataFrame([{"id": f"a_{i}", "name": f"Artist {i}", "plays": i * 100} for i in range(25)])
        graph.add_nodes(df, "Artist", "id", "name")

        with tempfile.NamedTemporaryFile(suffix=".kgl", delete=False) as f:
            path1 = f.name
        with tempfile.NamedTemporaryFile(suffix=".kgl", delete=False) as f:
            path2 = f.name
        try:
            graph.save(path1)
            loaded = kglite.load(path1)

            # Update existing artists with a new 'genre' property
            df2 = pd.DataFrame([{"id": f"a_{i}", "name": f"Artist {i}", "genre": "rock"} for i in range(25)])
            loaded.add_nodes(df2, "Artist", "id", "name", conflict_handling="update")
            loaded.save(path2)

            reloaded = kglite.load(path2)
            assert reloaded.select("Artist").len() == 25

            node = reloaded.node("Artist", "a_5")
            assert node["genre"] == "rock"
            assert node["plays"] == 500
        finally:
            os.unlink(path1)
            os.unlink(path2)

    def test_incremental_one_at_a_time_save_load(self):
        """Add nodes one at a time, save, load — all properties preserved."""
        graph = KnowledgeGraph()
        for i in range(100):
            df = pd.DataFrame([{"id": f"n_{i}", "title": f"Node {i}", "score": float(i)}])
            graph.add_nodes(df, "Item", "id", "title", conflict_handling="update")

        with tempfile.NamedTemporaryFile(suffix=".kgl", delete=False) as f:
            path = f.name
        try:
            graph.save(path)
            loaded = kglite.load(path)
            assert loaded.select("Item").len() == 100

            node = loaded.node("Item", "n_99")
            assert node["score"] == 99.0
        finally:
            os.unlink(path)

    def test_small_graph_save_load(self):
        """Very small graphs (5 + 24 nodes) must not corrupt on save/load."""
        graph = KnowledgeGraph()
        artists = pd.DataFrame([{"id": f"artist_{i}", "name": f"Artist {i}"} for i in range(5)])
        tags = pd.DataFrame([{"id": f"tag_{i}", "name": f"Tag {i}"} for i in range(24)])
        graph.add_nodes(artists, "Artist", "id", "name")
        graph.add_nodes(tags, "Tag", "id", "name")

        with tempfile.NamedTemporaryFile(suffix=".kgl", delete=False) as f:
            path = f.name
        try:
            graph.save(path)
            loaded = kglite.load(path)
            assert loaded.select("Artist").len() == 5
            assert loaded.select("Tag").len() == 24
        finally:
            os.unlink(path)

    def test_multiple_save_load_cycles(self):
        """Three consecutive save/load cycles with additions each time."""
        with tempfile.NamedTemporaryFile(suffix=".kgl", delete=False) as f:
            path = f.name
        try:
            # Cycle 1: create + save
            g = KnowledgeGraph()
            df1 = pd.DataFrame([{"id": f"n_{i}", "name": f"N{i}", "v": i} for i in range(10)])
            g.add_nodes(df1, "X", "id", "name")
            g.save(path)

            # Cycle 2: load + add + save
            g2 = kglite.load(path)
            df2 = pd.DataFrame([{"id": f"n_{i}", "name": f"N{i}", "v": i} for i in range(10, 20)])
            g2.add_nodes(df2, "X", "id", "name")
            g2.save(path)

            # Cycle 3: load + add + save
            g3 = kglite.load(path)
            df3 = pd.DataFrame([{"id": f"n_{i}", "name": f"N{i}", "v": i} for i in range(20, 30)])
            g3.add_nodes(df3, "X", "id", "name")
            g3.save(path)

            # Final load: all 30 nodes with properties
            final = kglite.load(path)
            assert final.select("X").len() == 30

            for i in [0, 9, 10, 19, 20, 29]:
                node = final.node("X", f"n_{i}")
                assert node["v"] == i, f"n_{i} expected v={i}, got {node['v']}"
        finally:
            os.unlink(path)
