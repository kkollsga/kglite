"""Tests for the streaming disk-to-disk subgraph filter.

The public API is ``KnowledgeGraph.save_subset(path)`` on the fluent
chain, which produces an independent on-disk graph file that reloads via
``kglite.load(path)``. The Pass A scan underneath it has no Python route;
its end-to-end tests live in
``crates/kglite/src/graph/mutation/subgraph_streaming.rs``.
"""

import os
import shutil
import tempfile

import pandas as pd
import pytest

from kglite import KnowledgeGraph, load


@pytest.fixture
def disk_dir():
    d = tempfile.mkdtemp(prefix="kglite_subgraph_streaming_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


def build_disk_graph_with_articles_and_authors(path: str) -> KnowledgeGraph:
    """A miniature 'Wikidata-shape' disk graph: Articles, Authors, and
    extra noise types/edges so the filter actually has to discriminate.
    """
    g = KnowledgeGraph(storage="disk", path=path)

    articles = pd.DataFrame(
        {
            "aid": ["a1", "a2", "a3", "a4"],
            "title": ["Paper One", "Paper Two", "Paper Three", "Paper Four"],
        }
    )
    g.add_nodes(articles, "Article", "aid", "title")

    authors = pd.DataFrame(
        {
            "uid": ["p1", "p2", "p3"],
            "name": ["Alice", "Bob", "Carol"],
        }
    )
    g.add_nodes(authors, "Author", "uid", "name")

    venues = pd.DataFrame(
        {
            "vid": ["v1", "v2"],
            "name": ["Venue One", "Venue Two"],
        }
    )
    g.add_nodes(venues, "Venue", "vid", "name")

    # AUTHORED_BY: a1↔p1, a1↔p2, a2↔p2, a3↔p3, a4↔p3.
    # Touches 4 articles + 3 authors = 7 nodes.
    authored = pd.DataFrame({"from_id": ["a1", "a1", "a2", "a3", "a4"], "to_id": ["p1", "p2", "p2", "p3", "p3"]})
    g.add_connections(authored, "AUTHORED_BY", "Article", "from_id", "Author", "to_id")

    # PUBLISHED_IN: a1→v1, a2→v1, a3→v2, a4→v2 (4 edges, mixes Venue in).
    published = pd.DataFrame({"from_id": ["a1", "a2", "a3", "a4"], "to_id": ["v1", "v1", "v2", "v2"]})
    g.add_connections(published, "PUBLISHED_IN", "Article", "from_id", "Venue", "to_id")

    return g


class TestSaveSubsetRoundTrip:
    """End-to-end: a fluent selection produces an independent subgraph
    file that reloads with the right shape and properties intact.
    """

    def _count_by_type(self, kg, node_type: str) -> int:
        rows = kg.cypher(f"MATCH (n:{node_type}) RETURN count(n) AS c").to_df()
        return int(rows["c"][0])

    def _count_edges(self, kg, edge_type: str | None = None) -> int:
        if edge_type:
            q = f"MATCH ()-[r:{edge_type}]->() RETURN count(r) AS c"
        else:
            q = "MATCH ()-[r]->() RETURN count(r) AS c"
        rows = kg.cypher(q).to_df()
        return int(rows["c"][0])

    def test_disk_source_articles_only(self, disk_dir):
        src = build_disk_graph_with_articles_and_authors(disk_dir)
        out_path = os.path.join(disk_dir, "articles_only.kgl")

        # Just Articles — no expansion. Subgraph contains only the 4
        # Article nodes and zero edges (no edges between articles).
        src.select("Article").save_subset(out_path)

        sub = load(out_path)
        assert self._count_by_type(sub, "Article") == 4
        assert self._count_by_type(sub, "Author") == 0
        assert self._count_by_type(sub, "Venue") == 0

    def test_disk_source_articles_with_neighbours(self, disk_dir):
        src = build_disk_graph_with_articles_and_authors(disk_dir)
        out_path = os.path.join(disk_dir, "articles_with_neighbours.kgl")

        # Articles + 1-hop neighbours of any edge type. expand() does
        # not currently take an edge-type filter, so we get authors and
        # venues both. All 9 nodes survive.
        src.select("Article").expand(hops=1).save_subset(out_path)

        sub = load(out_path)
        assert self._count_by_type(sub, "Article") == 4
        assert self._count_by_type(sub, "Author") == 3
        assert self._count_by_type(sub, "Venue") == 2

        assert self._count_edges(sub, "AUTHORED_BY") == 5
        assert self._count_edges(sub, "PUBLISHED_IN") == 4

    def test_node_properties_round_trip(self, disk_dir):
        src = build_disk_graph_with_articles_and_authors(disk_dir)
        out_path = os.path.join(disk_dir, "with_props.kgl")

        src.select("Article").expand(hops=1).save_subset(out_path)
        sub = load(out_path)

        rows = sub.cypher("MATCH (a:Article) RETURN a.title AS t ORDER BY t").to_df()
        titles = sorted(rows["t"].tolist())
        assert titles == ["Paper Four", "Paper One", "Paper Three", "Paper Two"]

        rows_a = sub.cypher("MATCH (p:Author) RETURN p.name AS n ORDER BY n").to_df()
        names = sorted(rows_a["n"].tolist())
        assert names == ["Alice", "Bob", "Carol"]

    def test_save_subset_in_memory_source(self):
        # save_subset works on in-memory sources too — same machinery,
        # any storage mode.
        src = KnowledgeGraph()
        src.add_nodes(
            pd.DataFrame({"nid": ["a", "b", "c"], "name": ["A", "B", "C"]}),
            "Person",
            "nid",
            "name",
        )
        src.add_connections(
            pd.DataFrame({"from_id": ["a", "b"], "to_id": ["b", "c"]}),
            "KNOWS",
            "Person",
            "from_id",
            "Person",
            "to_id",
        )

        with tempfile.TemporaryDirectory() as d:
            out_path = os.path.join(d, "people.kgl")
            src.select("Person").save_subset(out_path)
            sub = load(out_path)
            assert self._count_by_type(sub, "Person") == 3

    def test_matches_to_subgraph_then_save(self, disk_dir):
        # Differential: save_subset must produce the same logical graph
        # as the explicit two-step `to_subgraph().save()` chain.
        src = build_disk_graph_with_articles_and_authors(disk_dir)

        path_subset = os.path.join(disk_dir, "via_subset.kgl")
        path_explicit = os.path.join(disk_dir, "via_explicit.kgl")

        src.select("Article").expand(hops=1).save_subset(path_subset)
        src.select("Article").expand(hops=1).to_subgraph().save(path_explicit)

        a = load(path_subset)
        b = load(path_explicit)

        for t in ["Article", "Author", "Venue"]:
            assert self._count_by_type(a, t) == self._count_by_type(b, t)
        assert self._count_edges(a) == self._count_edges(b)
