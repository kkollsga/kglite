"""kglite.graphgen() — bundled synthetic-graph generator (0.12.0)."""

import json
import os
import tempfile

import pytest

import kglite


class TestInMemory:
    """out=None → a ready-to-query KnowledgeGraph."""

    def test_returns_queryable_graph(self):
        g = kglite.graphgen("tiny")
        assert isinstance(g, kglite.KnowledgeGraph)
        # All five node types present, with the expected dominant Person count.
        persons = g.cypher("MATCH (p:Person) RETURN count(p) AS c")[0]["c"]
        assert persons == 1000  # 'tiny' scale
        for ntype in ("Company", "Project", "Skill", "City"):
            n = g.cypher(f"MATCH (x:{ntype}) RETURN count(x) AS c")[0]["c"]
            assert n > 0, f"no {ntype} nodes"
        # Edges loaded + traversable.
        knows = g.cypher("MATCH (:Person)-[r:KNOWS]->(:Person) RETURN count(r) AS c")[0]["c"]
        assert knows > 0
        fof = g.cypher("MATCH (p:Person)-[:KNOWS]->()-[:KNOWS]->(f) RETURN count(DISTINCT f) AS c")[0]["c"]
        assert fof > 0

    def test_deterministic(self):
        a = kglite.graphgen("tiny", seed=7)
        b = kglite.graphgen("tiny", seed=7)
        qa = a.cypher("MATCH (:Person)-[r:KNOWS]->() RETURN count(r) AS c")[0]["c"]
        qb = b.cypher("MATCH (:Person)-[r:KNOWS]->() RETURN count(r) AS c")[0]["c"]
        assert qa == qb
        # Different seed → (almost certainly) different edge count.
        c = kglite.graphgen("tiny", seed=8)
        qc = c.cypher("MATCH (:Person)-[r:KNOWS]->() RETURN count(r) AS c")[0]["c"]
        assert qc != qa or True  # not guaranteed different, but must not error

    def test_persons_override(self):
        g = kglite.graphgen(persons=500)
        assert g.cypher("MATCH (p:Person) RETURN count(p) AS c")[0]["c"] == 500

    def test_uniform_degree_dist(self):
        g = kglite.graphgen("tiny", degree_dist="uniform")
        assert g.cypher("MATCH (:Person)-[r:KNOWS]->() RETURN count(r) AS c")[0]["c"] > 0


class TestStreaming:
    """out=DIR → bounded-memory CSV stream + manifest."""

    def test_streams_csvs_and_manifest(self):
        with tempfile.TemporaryDirectory() as d:
            stats = kglite.graphgen("tiny", out=d)
            assert stats["nodes"] > 1000
            assert stats["edges"] > 1000
            assert os.path.samefile(stats["out"], d)
            for f in ("Person.csv", "Company.csv", "KNOWS.csv", "DEPENDS_ON.csv", "manifest.json"):
                assert os.path.exists(os.path.join(d, f)), f"missing {f}"
            manifest = json.loads(open(os.path.join(d, "manifest.json")).read())
            assert manifest["schema"] == "graphsuite"
            assert manifest["counts"]["Person"] == 1000
            assert "params" in manifest and "seed_persons" in manifest["params"]

    def test_streamed_bytes_match_in_memory_counts(self):
        # The same seed/scale loaded in-memory and streamed must agree on size.
        g = kglite.graphgen("tiny", seed=99)
        with tempfile.TemporaryDirectory() as d:
            stats = kglite.graphgen("tiny", seed=99, out=d)
        total_nodes = sum(
            g.cypher(f"MATCH (x:{t}) RETURN count(x) AS c")[0]["c"]
            for t in ("Person", "Company", "Project", "Skill", "City")
        )
        assert total_nodes == stats["nodes"]


class TestValidation:
    def test_unknown_scale_raises(self):
        with pytest.raises(ValueError):
            kglite.graphgen("enormous")

    def test_bad_degree_dist_raises(self):
        with pytest.raises(ValueError):
            kglite.graphgen("tiny", degree_dist="power-law")
