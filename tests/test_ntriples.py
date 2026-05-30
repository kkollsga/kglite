"""Tests for the N-Triples loader."""

import bz2
import gzip
from pathlib import Path

import pytest

from kglite import KnowledgeGraph

# Minimal N-Triples fixture lives in a file to avoid line-length issues
_SAMPLE_NT_PATH = Path(__file__).parent / "data" / "sample_wikidata.nt"


@pytest.fixture
def nt_file():
    """Path to the sample N-Triples file."""
    return str(_SAMPLE_NT_PATH)


@pytest.fixture
def nt_bz2_file(tmp_path):
    """Write sample N-Triples to a .nt.bz2 file."""
    data = _SAMPLE_NT_PATH.read_text(encoding="utf-8")
    path = tmp_path / "sample.nt.bz2"
    with bz2.open(str(path), "wt", encoding="utf-8") as f:
        f.write(data)
    return str(path)


@pytest.fixture
def nt_gz_file(tmp_path):
    """Write sample N-Triples to a .nt.gz file."""
    data = _SAMPLE_NT_PATH.read_text(encoding="utf-8")
    path = tmp_path / "sample.nt.gz"
    with gzip.open(str(path), "wt", encoding="utf-8") as f:
        f.write(data)
    return str(path)


class TestLoadPlainNT:
    """Load from plain .nt file."""

    def test_basic_load(self, nt_file):
        graph = KnowledgeGraph()
        stats = graph.load_ntriples(nt_file, languages=["en"])
        assert stats["entities"] == 4
        assert stats["triples_scanned"] == 14

    def test_node_titles(self, nt_file):
        graph = KnowledgeGraph()
        graph.load_ntriples(nt_file, languages=["en"])
        r = graph.cypher('MATCH (n {title: "Douglas Adams"}) RETURN n.id').to_df()
        assert len(r) == 1

    def test_description_property(self, nt_file):
        graph = KnowledgeGraph()
        graph.load_ntriples(nt_file, languages=["en"])
        r = graph.cypher('MATCH (n {title: "Douglas Adams"}) RETURN n.description').to_df()
        assert r["n.description"][0] == "English author and humourist"


class TestLoadCompressed:
    """Load from compressed files."""

    def test_bz2(self, nt_bz2_file):
        graph = KnowledgeGraph()
        stats = graph.load_ntriples(nt_bz2_file, languages=["en"])
        assert stats["entities"] == 4

    def test_gz(self, nt_gz_file):
        graph = KnowledgeGraph()
        stats = graph.load_ntriples(nt_gz_file, languages=["en"])
        assert stats["entities"] == 4


class TestFiltering:
    """Predicate and language filtering."""

    def test_predicate_filter(self, nt_file):
        graph = KnowledgeGraph()
        graph.load_ntriples(nt_file, predicates=["P31"], languages=["en"])
        # Only P31 edges should exist
        r = graph.cypher("MATCH ()-[r]->() RETURN type(r) AS t, count(*) AS c").to_df()
        assert len(r) <= 1  # only P31 or nothing if targets missing
        if len(r) > 0:
            assert r["t"][0] == "P31"

    def test_language_filter_en(self, nt_file):
        graph = KnowledgeGraph()
        graph.load_ntriples(nt_file, languages=["en"])
        r = graph.cypher('MATCH (n {title: "Douglas Adams"}) RETURN n.title').to_df()
        assert len(r) == 1  # English label used

    def test_language_filter_de(self, nt_file):
        graph = KnowledgeGraph()
        graph.load_ntriples(nt_file, languages=["de"])
        # German label "Douglas Adams" (same in both langs for this entity)
        r = graph.cypher('MATCH (n {title: "Douglas Adams"}) RETURN n.title').to_df()
        assert len(r) == 1

    def test_max_entities(self, nt_file):
        graph = KnowledgeGraph()
        stats = graph.load_ntriples(nt_file, max_entities=2, languages=["en"])
        assert stats["entities"] == 2

    def test_predicate_labels(self, nt_file):
        graph = KnowledgeGraph()
        graph.load_ntriples(
            nt_file,
            languages=["en"],
            predicate_labels={"P31": "instance_of"},
        )
        r = graph.cypher("MATCH ()-[r:instance_of]->() RETURN count(r) AS c").to_df()
        assert r["c"][0] > 0

    def test_node_types_mapping(self, nt_file):
        graph = KnowledgeGraph()
        graph.load_ntriples(
            nt_file,
            languages=["en"],
            node_types={"Q5": "Person", "Q6256": "Country"},
        )
        persons = graph.select("Person").len()
        countries = graph.select("Country").len()
        assert persons >= 1  # Douglas Adams
        assert countries >= 1  # United Kingdom


class TestWikidataIdNidSemantics:
    """Cross-mode parity (0.10.10): a Q-code is the integer `id` in EVERY mode;
    the string form `"Q42"` is the `nid` property. Query by `{nid: 'Q42'}`."""

    def test_lookup_by_nid(self, nt_file):
        graph = KnowledgeGraph(storage="mapped")
        graph.load_ntriples(nt_file, languages=["en"])
        r = graph.cypher('MATCH (n {nid: "Q42"}) RETURN n.title').to_df()
        assert r["n.title"][0] == "Douglas Adams"

    def test_id_is_integer_nid_is_string(self, nt_file):
        graph = KnowledgeGraph(storage="mapped")
        graph.load_ntriples(nt_file, languages=["en"])
        r = graph.cypher('MATCH (n {nid: "Q42"}) RETURN n.id, n.nid').to_df()
        assert r["n.id"][0] == 42  # integer identity (same in every mode)
        assert r["n.nid"][0] == "Q42"  # string form
        # And the integer id is directly matchable.
        r2 = graph.cypher("MATCH (n {id: 42}) RETURN n.title").to_df()
        assert r2["n.title"][0] == "Douglas Adams"

    def test_edges_work_by_nid(self, nt_file):
        graph = KnowledgeGraph(storage="mapped")
        graph.load_ntriples(nt_file, languages=["en"])
        # Q42 -> Q145 via P27 (citizenship)
        r = graph.cypher('MATCH (n {nid: "Q42"})-[:P27]->(m) RETURN m.title').to_df()
        assert len(r) == 1
        assert r["m.title"][0] == "United Kingdom"

    def test_default_mode_same_as_mapped(self, nt_file):
        graph = KnowledgeGraph()
        graph.load_ntriples(nt_file, languages=["en"])
        r = graph.cypher('MATCH (n {nid: "Q42"}) RETURN n.title, n.id').to_df()
        assert r["n.title"][0] == "Douglas Adams"
        assert r["n.id"][0] == 42  # integer in default mode too (was 'Q42' pre-0.10.10)

    def test_typed_literal_property(self, nt_file):
        graph = KnowledgeGraph(storage="mapped")
        graph.load_ntriples(nt_file, languages=["en"])
        r = graph.cypher('MATCH (n {nid: "Q42"}) RETURN n.P1082').to_df()
        assert r["n.P1082"][0] == 42  # decimal literal parsed as int


class TestEdgeCases:
    """Edge cases and error handling."""

    def test_empty_file(self, tmp_path):
        path = tmp_path / "empty.nt"
        path.write_text("", encoding="utf-8")
        graph = KnowledgeGraph()
        stats = graph.load_ntriples(str(path))
        assert stats["entities"] == 0
        assert stats["edges"] == 0

    def test_comments_and_blank_lines(self, tmp_path):
        content = "# This is a comment\n\n# Another comment\n"
        path = tmp_path / "comments.nt"
        path.write_text(content, encoding="utf-8")
        graph = KnowledgeGraph()
        stats = graph.load_ntriples(str(path))
        assert stats["entities"] == 0

    def test_nonexistent_file(self):
        graph = KnowledgeGraph()
        with pytest.raises(RuntimeError, match="Cannot open"):
            graph.load_ntriples("/nonexistent/path.nt")
