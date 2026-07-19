"""Tests for embedding storage and vector search functionality."""

import math
import os
import tempfile
import warnings

import pandas as pd
import pytest

import kglite


@pytest.fixture
def graph_with_embeddings():
    """Create a graph with articles and embeddings."""
    graph = kglite.KnowledgeGraph()

    # Create articles
    df = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "title": ["Alpha", "Beta", "Gamma", "Delta", "Epsilon"],
            "category": ["politics", "sports", "politics", "tech", "sports"],
            "summary": ["alpha text", "beta text", "gamma text", "delta text", "epsilon text"],
        }
    )
    graph.add_nodes(df, "Article", "id", "title")

    # Set embeddings (3-dimensional for simplicity)
    embeddings = {
        1: [1.0, 0.0, 0.0],  # Alpha: points along x-axis
        2: [0.0, 1.0, 0.0],  # Beta: points along y-axis
        3: [0.9, 0.1, 0.0],  # Gamma: mostly x-axis (similar to Alpha)
        4: [0.0, 0.0, 1.0],  # Delta: points along z-axis
        5: [0.1, 0.9, 0.0],  # Epsilon: mostly y-axis (similar to Beta)
    }
    result = graph.set_embeddings("Article", "summary", embeddings)

    assert result["embeddings_stored"] == 5
    assert result["dimension"] == 3
    assert result["skipped"] == 0

    return graph


class TestHybridRagRetrieval:
    """#20 — filtered hybrid retrieval: a graph predicate AND vector ranking in
    one Cypher query (the RAG-over-a-knowledge-graph shape). The point is that
    the graph filter and the semantic ranking compose — you don't need a
    separate vector store + join."""

    def test_graph_filter_overrides_vector_similarity(self, graph_with_embeddings):
        g = graph_with_embeddings
        # Query points along y — the closest vectors are Beta (id 2) and Epsilon
        # (id 5), but both are 'sports'. A 'politics' filter must exclude them
        # and return only Alpha/Gamma, proving the filter composes with ranking.
        q = [0.0, 1.0, 0.0]
        rows = g.cypher(
            "MATCH (a:Article) WHERE a.category = 'politics' "
            "RETURN a.title AS t, vector_score(a, 'summary_emb', $q) AS s "
            "ORDER BY s DESC LIMIT 5",
            params={"q": q},
        ).to_list()
        assert {r["t"] for r in rows} == {"Alpha", "Gamma"}
        scores = [r["s"] for r in rows]
        assert scores == sorted(scores, reverse=True)  # ranked by similarity

    def test_top_k_semantic_retrieval(self, graph_with_embeddings):
        g = graph_with_embeddings
        q = [1.0, 0.0, 0.0]  # nearest: Alpha (1.0), Gamma (~0.99)
        rows = g.cypher(
            "MATCH (a:Article) RETURN a.title AS t, vector_score(a, 'summary_emb', $q) AS s ORDER BY s DESC LIMIT 2",
            params={"q": q},
        ).to_list()
        assert [r["t"] for r in rows] == ["Alpha", "Gamma"]


# ── set_embeddings / embeddings ────────────────────────────────────────


class TestSetGetEmbeddings:
    def test_set_and_embeddings(self, graph_with_embeddings):
        graph = graph_with_embeddings
        embs = graph.select("Article").embeddings("summary")

        assert len(embs) == 5
        assert embs[1] == [1.0, 0.0, 0.0]
        assert embs[2] == [0.0, 1.0, 0.0]

    def test_embeddings_filtered_selection(self, graph_with_embeddings):
        graph = graph_with_embeddings
        embs = graph.select("Article").where({"category": "politics"}).embeddings("summary")

        assert len(embs) == 2
        assert 1 in embs  # Alpha
        assert 3 in embs  # Gamma

    def test_set_embeddings_with_missing_ids(self):
        graph = kglite.KnowledgeGraph()
        df = pd.DataFrame({"id": [1, 2], "title": ["A", "B"]})
        graph.add_nodes(df, "Node", "id", "title")

        result = graph.set_embeddings("Node", "title", {1: [1.0, 2.0], 2: [3.0, 4.0], 999: [5.0, 6.0]})

        assert result["embeddings_stored"] == 2
        assert result["skipped"] == 1

    def test_set_embeddings_inconsistent_dimensions(self):
        graph = kglite.KnowledgeGraph()
        df = pd.DataFrame({"id": [1, 2], "title": ["A", "B"]})
        graph.add_nodes(df, "Node", "id", "title")

        with pytest.raises(ValueError, match="Inconsistent embedding dimensions"):
            graph.set_embeddings("Node", "title", {1: [1.0, 2.0], 2: [3.0, 4.0, 5.0]})

    def test_set_embeddings_empty_dict(self):
        graph = kglite.KnowledgeGraph()
        df = pd.DataFrame({"id": [1], "title": ["A"]})
        graph.add_nodes(df, "Node", "id", "title")

        result = graph.set_embeddings("Node", "title", {})
        assert result["embeddings_stored"] == 0

    def test_embeddings_two_arg_form(self, graph_with_embeddings):
        """embeddings(node_type, text_column) returns all embeddings."""
        graph = graph_with_embeddings
        embs = graph.embeddings("Article", "summary")

        assert len(embs) == 5
        assert embs[1] == [1.0, 0.0, 0.0]
        assert embs[2] == [0.0, 1.0, 0.0]

    def test_embeddings_two_arg_nonexistent(self, graph_with_embeddings):
        """Two-arg form returns empty dict for nonexistent store."""
        graph = graph_with_embeddings
        embs = graph.embeddings("Article", "nonexistent")
        assert embs == {}

    def test_embedding_single_node(self, graph_with_embeddings):
        """embedding(node_type, text_column, node_id) returns one vector."""
        graph = graph_with_embeddings
        vec = graph.embedding("Article", "summary", 1)
        assert vec == [1.0, 0.0, 0.0]

        vec2 = graph.embedding("Article", "summary", 4)
        assert vec2 == [0.0, 0.0, 1.0]

    def test_embedding_nonexistent_node(self, graph_with_embeddings):
        """embedding returns None for a node ID that doesn't exist."""
        graph = graph_with_embeddings
        assert graph.embedding("Article", "summary", 999) is None

    def test_embedding_nonexistent_store(self, graph_with_embeddings):
        """embedding returns None for a property name that doesn't exist."""
        graph = graph_with_embeddings
        assert graph.embedding("Article", "no_such", 1) is None


class TestSetEmbeddingsValidation:
    """Tests for set_embeddings source column validation."""

    def test_set_embeddings_missing_node_type(self):
        """set_embeddings raises when node type doesn't exist."""
        graph = kglite.KnowledgeGraph()
        df = pd.DataFrame({"id": [1], "title": ["A"]})
        graph.add_nodes(df, "Node", "id", "title")

        with pytest.raises(ValueError, match="does not exist"):
            graph.set_embeddings("NonExistent", "title", {1: [1.0, 2.0]})

    def test_set_embeddings_missing_source_column(self):
        """set_embeddings raises when source column doesn't exist on nodes."""
        graph = kglite.KnowledgeGraph()
        df = pd.DataFrame({"id": [1], "title": ["A"]})
        graph.add_nodes(df, "Node", "id", "title")

        with pytest.raises(ValueError, match="Source column"):
            graph.set_embeddings("Node", "nonexistent", {1: [1.0, 2.0]})

    def test_set_embeddings_builtin_columns_accepted(self):
        """Builtin columns (id, title, type) are always accepted."""
        graph = kglite.KnowledgeGraph()
        df = pd.DataFrame({"id": [1], "title": ["A"]})
        graph.add_nodes(df, "Node", "id", "title")

        for col in ["id", "title", "type"]:
            result = graph.set_embeddings("Node", col, {1: [1.0, 2.0]})
            assert result["embeddings_stored"] == 1


# ── list_embeddings / remove_embeddings ────────────────────────────────────


class TestListRemoveEmbeddings:
    def test_list_embeddings(self, graph_with_embeddings):
        graph = graph_with_embeddings
        listing = graph.list_embeddings()

        assert len(listing) == 1
        info = listing[0]
        assert info["node_type"] == "Article"
        assert info["text_column"] == "summary"
        assert info["dimension"] == 3
        assert info["count"] == 5

    def test_remove_embeddings(self, graph_with_embeddings):
        graph = graph_with_embeddings
        graph.remove_embeddings("Article", "summary")

        assert len(graph.list_embeddings()) == 0

    def test_list_embeddings_empty(self):
        graph = kglite.KnowledgeGraph()
        assert graph.list_embeddings() == []


# ── vector_search ──────────────────────────────────────────────────────────


class TestVectorSearch:
    def test_basic_cosine_search(self, graph_with_embeddings):
        graph = graph_with_embeddings

        # Search for vectors similar to [1, 0, 0] (Alpha direction)
        results = graph.select("Article").vector_search("summary", [1.0, 0.0, 0.0], top_k=3)

        assert len(results) == 3
        # Alpha (exact match) should be first
        assert results[0]["id"] == 1
        assert results[0]["score"] == pytest.approx(1.0, abs=1e-5)
        # Gamma (0.9, 0.1, 0) should be second
        assert results[1]["id"] == 3
        # Scores should be descending
        assert results[0]["score"] >= results[1]["score"] >= results[2]["score"]

    def test_search_with_filter(self, graph_with_embeddings):
        graph = graph_with_embeddings

        # Search only within "sports" category
        results = (
            graph.select("Article").where({"category": "sports"}).vector_search("summary", [0.0, 1.0, 0.0], top_k=10)
        )

        # Only sports articles: Beta(2) and Epsilon(5)
        assert len(results) == 2
        ids = [r["id"] for r in results]
        assert 2 in ids
        assert 5 in ids

    def test_search_result_contains_properties(self, graph_with_embeddings):
        graph = graph_with_embeddings
        results = graph.select("Article").vector_search("summary", [1.0, 0.0, 0.0], top_k=1)

        assert len(results) == 1
        r = results[0]
        assert "id" in r
        assert "title" in r
        assert "type" in r
        assert "score" in r
        assert "category" in r
        assert r["title"] == "Alpha"
        assert r["type"] == "Article"

    def test_search_top_k_limits_results(self, graph_with_embeddings):
        graph = graph_with_embeddings
        results = graph.select("Article").vector_search("summary", [1.0, 0.0, 0.0], top_k=2)

        assert len(results) == 2

    def test_search_dot_product_metric(self, graph_with_embeddings):
        graph = graph_with_embeddings
        results = graph.select("Article").vector_search(
            "summary",
            [1.0, 0.0, 0.0],
            top_k=3,
            metric="dot_product",
        )

        assert len(results) == 3
        # Alpha has highest dot product with [1,0,0]
        assert results[0]["id"] == 1
        assert results[0]["score"] == pytest.approx(1.0, abs=1e-5)

    def test_search_euclidean_metric(self, graph_with_embeddings):
        graph = graph_with_embeddings
        results = graph.select("Article").vector_search(
            "summary",
            [1.0, 0.0, 0.0],
            top_k=3,
            metric="euclidean",
        )

        assert len(results) == 3
        # Alpha has distance 0 (neg_euclidean = 0.0)
        assert results[0]["id"] == 1
        assert results[0]["score"] == pytest.approx(0.0, abs=1e-5)

    def test_search_dimension_mismatch(self, graph_with_embeddings):
        graph = graph_with_embeddings

        with pytest.raises(ValueError, match="dimension"):
            graph.select("Article").vector_search(
                "summary",
                [1.0, 0.0],
                top_k=3,  # 2D instead of 3D
            )

    def test_search_invalid_metric(self, graph_with_embeddings):
        graph = graph_with_embeddings

        with pytest.raises(ValueError, match="Unknown metric"):
            graph.select("Article").vector_search("summary", [1.0, 0.0, 0.0], metric="manhattan")

    def test_search_to_df(self, graph_with_embeddings):
        graph = graph_with_embeddings
        df = graph.select("Article").vector_search("summary", [1.0, 0.0, 0.0], top_k=3, to_df=True)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert "score" in df.columns
        assert "id" in df.columns

    def test_search_list_and_dataframe_conversion_match_after_reload(self, tmp_path):
        graph = kglite.KnowledgeGraph()
        graph.cypher(
            "CREATE (:Article {id:1, title:'A', summary:'a', scalar:7, "
            "tags:['x','y'], meta:{rank:1, nested:[true,false]}}), "
            "(:Article {id:2, title:'B', summary:'b', scalar:8, "
            "tags:['z'], meta:{rank:2, nested:[false]}})"
        )
        graph.set_embeddings("Article", "summary", {1: [1.0, 0.0], 2: [0.0, 1.0]})

        def assert_parity(candidate):
            rows = candidate.select("Article").vector_search("summary", [1.0, 0.0], top_k=2, to_df=False)
            frame_rows = (
                candidate.select("Article").vector_search("summary", [1.0, 0.0], top_k=2, to_df=True).to_dict("records")
            )
            assert frame_rows == rows

        assert_parity(graph)
        path = str(tmp_path / "vector_conversion.kgl")
        graph.save(path)
        assert_parity(kglite.load(path))

    def test_search_dataframe_propagates_value_conversion_error(self):
        graph = kglite.KnowledgeGraph()
        graph.cypher("CREATE (:Article {id:1, title:'A', summary:'a', bad:datetime('10000-01-01T00:00:00')})")
        graph.set_embeddings("Article", "summary", {1: [1.0, 0.0]})

        for to_df in (False, True):
            with pytest.raises(ValueError, match="year"):
                graph.select("Article").vector_search("summary", [1.0, 0.0], top_k=1, to_df=to_df)

    def test_search_empty_selection(self):
        graph = kglite.KnowledgeGraph()
        results = graph.vector_search("text", [1.0, 0.0, 0.0], top_k=3)
        assert results == []


# ── Embeddings invisible to node API ──────────────────────────────────────


class TestEmbeddingsInvisible:
    def test_embeddings_not_in_get_nodes(self, graph_with_embeddings):
        graph = graph_with_embeddings
        nodes = graph.select("Article").collect()

        for node in nodes:
            assert "summary_emb" not in node

    def test_embeddings_not_in_to_df(self, graph_with_embeddings):
        graph = graph_with_embeddings
        df = graph.select("Article").to_df()

        assert "summary_emb" not in df.columns


# ── Persistence ───────────────────────────────────────────────────────────


class TestEmbeddingPersistence:
    def test_save_and_load_with_embeddings(self, graph_with_embeddings):
        graph = graph_with_embeddings

        with tempfile.NamedTemporaryFile(suffix=".kgl", delete=False) as f:
            path = f.name

        try:
            graph.save(path)

            loaded = kglite.load(path)

            # Check embeddings survived save/load
            listing = loaded.list_embeddings()
            assert len(listing) == 1
            assert listing[0]["node_type"] == "Article"
            assert listing[0]["text_column"] == "summary"
            assert listing[0]["dimension"] == 3
            assert listing[0]["count"] == 5

            # Verify vector search still works
            results = loaded.select("Article").vector_search("summary", [1.0, 0.0, 0.0], top_k=3)
            assert len(results) == 3
            assert results[0]["id"] == 1
            assert results[0]["score"] == pytest.approx(1.0, abs=1e-5)

            # Verify actual embedding values
            embs = loaded.select("Article").embeddings("summary")
            assert embs[1] == pytest.approx([1.0, 0.0, 0.0])
            assert embs[2] == pytest.approx([0.0, 1.0, 0.0])
        finally:
            os.unlink(path)

    def test_save_and_load_without_embeddings(self):
        """Ensure save/load still works for graphs without embeddings."""
        graph = kglite.KnowledgeGraph()
        df = pd.DataFrame({"id": [1, 2], "title": ["A", "B"]})
        graph.add_nodes(df, "Node", "id", "title")

        with tempfile.NamedTemporaryFile(suffix=".kgl", delete=False) as f:
            path = f.name

        try:
            graph.save(path)
            loaded = kglite.load(path)

            assert loaded.list_embeddings() == []
            assert loaded.select("Node").len() == 2
        finally:
            os.unlink(path)


# ── add_nodes with embedding columns ─────────────────────────────────────


class TestAddNodesEmbeddings:
    def test_add_nodes_with_embedding_column_type(self):
        graph = kglite.KnowledgeGraph()

        df = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "title": ["A", "B", "C"],
                "text_emb": [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]],
            }
        )
        graph.add_nodes(
            df,
            "Doc",
            "id",
            "title",
            column_types={"text_emb": "embedding"},
        )

        # Embeddings should be stored
        listing = graph.list_embeddings()
        assert len(listing) == 1
        assert listing[0]["text_column"] == "text"
        assert listing[0]["dimension"] == 2
        assert listing[0]["count"] == 3

        # Embeddings should NOT be in node properties
        nodes = graph.select("Doc").collect()
        for node in nodes:
            assert "text_emb" not in node

        # Vector search should work
        results = graph.select("Doc").vector_search("text", [1.0, 0.0], top_k=2)
        assert len(results) == 2
        assert results[0]["id"] == 1  # Most similar to [1, 0]


# ── Cypher vector_score() function ────────────────────────────────────────


class TestCypherVectorScore:
    def test_vector_score_in_return(self, graph_with_embeddings):
        graph = graph_with_embeddings
        result = graph.cypher(
            "MATCH (n:Article) "
            "RETURN n.title AS title, vector_score(n, 'summary_emb', [1.0, 0.0, 0.0]) AS score "
            "ORDER BY score DESC LIMIT 3",
            to_df=True,
        )
        assert len(result) == 3
        # Alpha (exact match) should be first
        assert result.iloc[0]["title"] == "Alpha"
        assert result.iloc[0]["score"] == pytest.approx(1.0, abs=1e-5)

    def test_vector_score_in_where(self, graph_with_embeddings):
        graph = graph_with_embeddings
        result = graph.cypher(
            "MATCH (n:Article) WHERE vector_score(n, 'summary_emb', [1.0, 0.0, 0.0]) > 0.9 RETURN n.title AS title",
            to_df=True,
        )
        titles = result["title"].tolist()
        # Alpha (1.0) and Gamma (0.99+) should pass threshold
        assert "Alpha" in titles
        assert "Gamma" in titles
        assert "Beta" not in titles

    def test_vector_score_with_property_filter(self, graph_with_embeddings):
        graph = graph_with_embeddings
        result = graph.cypher(
            "MATCH (n:Article) "
            "WHERE n.category = 'sports' "
            "RETURN n.title AS title, vector_score(n, 'summary_emb', [0.0, 1.0, 0.0]) AS score "
            "ORDER BY score DESC",
            to_df=True,
        )
        assert len(result) == 2
        # Beta and Epsilon are sports, Beta is closer to [0,1,0]
        assert result.iloc[0]["title"] == "Beta"

    def test_vector_score_with_params(self, graph_with_embeddings):
        graph = graph_with_embeddings
        result = graph.cypher(
            "MATCH (n:Article) "
            "RETURN n.title AS title, vector_score(n, 'summary_emb', $qvec) AS score "
            "ORDER BY score DESC LIMIT 2",
            params={"qvec": [1.0, 0.0, 0.0]},
            to_df=True,
        )
        assert len(result) == 2
        assert result.iloc[0]["title"] == "Alpha"

    def test_vector_score_dot_product_metric(self, graph_with_embeddings):
        graph = graph_with_embeddings
        result = graph.cypher(
            "MATCH (n:Article) "
            "RETURN n.title AS title, "
            "vector_score(n, 'summary_emb', [1.0, 0.0, 0.0], 'dot_product') AS score "
            "ORDER BY score DESC LIMIT 1",
            to_df=True,
        )
        assert result.iloc[0]["title"] == "Alpha"
        assert result.iloc[0]["score"] == pytest.approx(1.0, abs=1e-5)

    def test_vector_score_missing_embedding(self, graph_with_embeddings):
        graph = graph_with_embeddings
        with pytest.raises(kglite.KgError, match="no embedding"):
            graph.cypher("MATCH (n:Article) RETURN vector_score(n, 'nonexistent_emb', [1.0, 0.0, 0.0]) AS score")

    def test_vector_score_dimension_mismatch(self, graph_with_embeddings):
        graph = graph_with_embeddings
        with pytest.raises(kglite.KgError, match="dimension"):
            graph.cypher("MATCH (n:Article) RETURN vector_score(n, 'summary_emb', [1.0, 0.0]) AS score")


# ── embed_texts / search_text (text-level API) ────────────────────────────


class MockEmbedder:
    """Mock embedding model for testing. Maps text to simple vectors."""

    def __init__(self, dimension=3):
        self.dimension = dimension

    def embed(self, texts):
        """Deterministic embeddings: hash text to a unit vector."""
        import hashlib

        vectors = []
        for text in texts:
            h = hashlib.md5(text.encode()).digest()
            raw = [float(b) for b in h[: self.dimension]]
            norm = math.sqrt(sum(x * x for x in raw))
            if norm > 0:
                vectors.append([x / norm for x in raw])
            else:
                vectors.append([0.0] * self.dimension)
        return vectors


class TestSetEmbedder:
    def test_set_embedder_basic(self):
        """set_embedder registers a valid model."""
        graph = kglite.KnowledgeGraph()
        model = MockEmbedder(dimension=3)
        graph.set_embedder(model)  # Should not raise

    def test_set_embedder_missing_dimension(self):
        """Model without dimension attribute raises."""
        graph = kglite.KnowledgeGraph()

        class BadModel:
            def embed(self, texts):
                return []

        with pytest.raises(AttributeError, match="dimension"):
            graph.set_embedder(BadModel())

    def test_set_embedder_missing_embed(self):
        """Model without embed method raises."""
        graph = kglite.KnowledgeGraph()

        class BadModel:
            dimension = 3

        with pytest.raises(AttributeError, match="embed"):
            graph.set_embedder(BadModel())


class TestEmbedTexts:
    def test_embed_texts_basic(self):
        """Embed a text column and verify embeddings are stored."""
        graph = kglite.KnowledgeGraph()
        df = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "title": ["Alpha", "Beta", "Gamma"],
                "summary": ["AI research paper", "Sports article", "Tech news"],
            }
        )
        graph.add_nodes(df, "Article", "id", "title")

        graph.set_embedder(MockEmbedder(dimension=3))
        result = graph.embed_texts("Article", "summary", show_progress=False)

        assert result["embedded"] == 3
        assert result["skipped"] == 0
        assert result["dimension"] == 3

        # Verify embeddings are stored as {text_column}_emb
        listing = graph.list_embeddings()
        assert len(listing) == 1
        assert listing[0]["text_column"] == "summary"
        assert listing[0]["count"] == 3

    def test_embed_texts_missing_text_skipped(self):
        """Nodes with null/missing text column are skipped."""
        graph = kglite.KnowledgeGraph()
        df = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "title": ["A", "B", "C"],
                "summary": ["Has text", None, "Also has text"],
            }
        )
        graph.add_nodes(df, "Doc", "id", "title")

        graph.set_embedder(MockEmbedder(dimension=3))
        result = graph.embed_texts("Doc", "summary", show_progress=False)

        assert result["embedded"] == 2
        assert result["skipped"] == 1

    def test_embed_texts_batch_sizes(self):
        """Test with various batch sizes."""
        graph = kglite.KnowledgeGraph()
        df = pd.DataFrame(
            {
                "id": list(range(10)),
                "title": [f"Node{i}" for i in range(10)],
                "text": [f"Text content {i}" for i in range(10)],
            }
        )
        graph.add_nodes(df, "Item", "id", "title")

        graph.set_embedder(MockEmbedder(dimension=4))

        # batch_size=1 (many batches) — uses "text" column → "text_emb"
        result1 = graph.embed_texts("Item", "text", batch_size=1, show_progress=False)
        assert result1["embedded"] == 10

        # Verify embeddings stored
        embs = graph.embeddings("Item", "text")
        assert len(embs) == 10

    def test_embed_texts_dimension_mismatch(self):
        """Model returns wrong dimension → error."""
        graph = kglite.KnowledgeGraph()
        df = pd.DataFrame({"id": [1], "title": ["A"], "text": ["some text"]})
        graph.add_nodes(df, "Node", "id", "title")

        class BadModel:
            dimension = 5  # Claims 5

            def embed(self, texts):
                return [[1.0, 2.0, 3.0]] * len(texts)  # Returns 3

        graph.set_embedder(BadModel())
        with pytest.raises(ValueError, match="dimension"):
            graph.embed_texts("Node", "text", show_progress=False)

    def test_embed_texts_no_matching_nodes(self):
        """No nodes of that type → embedded=0."""
        graph = kglite.KnowledgeGraph()
        df = pd.DataFrame({"id": [1], "title": ["A"], "text": ["hello"]})
        graph.add_nodes(df, "Node", "id", "title")

        graph.set_embedder(MockEmbedder(dimension=3))
        result = graph.embed_texts("Other", "text", show_progress=False)

        assert result["embedded"] == 0
        assert result["skipped"] == 0

    def test_embed_texts_no_model_set(self):
        """Calling embed_texts without set_embedder raises with skeleton."""
        graph = kglite.KnowledgeGraph()
        df = pd.DataFrame({"id": [1], "title": ["A"], "text": ["hello"]})
        graph.add_nodes(df, "Node", "id", "title")

        with pytest.raises(RuntimeError, match="set_embedder"):
            graph.embed_texts("Node", "text")

    def test_embed_texts_show_progress(self):
        """show_progress=True works (uses tqdm if available)."""
        graph = kglite.KnowledgeGraph()
        df = pd.DataFrame(
            {
                "id": list(range(5)),
                "title": [f"N{i}" for i in range(5)],
                "text": [f"Text {i}" for i in range(5)],
            }
        )
        graph.add_nodes(df, "Item", "id", "title")
        graph.set_embedder(MockEmbedder(dimension=3))

        # show_progress=True (default) — should work even if tqdm is missing
        result = graph.embed_texts("Item", "text", batch_size=2, show_progress=True)
        assert result["embedded"] == 5

    def test_embed_texts_show_progress_false(self):
        """show_progress=False skips tqdm."""
        graph = kglite.KnowledgeGraph()
        df = pd.DataFrame({"id": [1, 2], "title": ["A", "B"], "text": ["hello", "world"]})
        graph.add_nodes(df, "Node", "id", "title")
        graph.set_embedder(MockEmbedder(dimension=3))

        result = graph.embed_texts("Node", "text", show_progress=False)
        assert result["embedded"] == 2

    def test_embed_texts_skips_existing(self):
        """Re-running embed_texts skips already-embedded nodes."""
        graph = kglite.KnowledgeGraph()
        df = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "title": ["A", "B", "C"],
                "text": ["hello", "world", "test"],
            }
        )
        graph.add_nodes(df, "Item", "id", "title")
        graph.set_embedder(MockEmbedder(dimension=3))

        # First run: embeds all 3
        r1 = graph.embed_texts("Item", "text", show_progress=False)
        assert r1["embedded"] == 3
        assert r1["skipped_existing"] == 0

        # Second run: all 3 already exist → skip all
        r2 = graph.embed_texts("Item", "text", show_progress=False)
        assert r2["embedded"] == 0
        assert r2["skipped_existing"] == 3

    def test_embed_texts_skips_existing_partial(self):
        """Adding new nodes then re-running embeds only new ones."""
        graph = kglite.KnowledgeGraph()
        df1 = pd.DataFrame({"id": [1, 2], "title": ["A", "B"], "text": ["hello", "world"]})
        graph.add_nodes(df1, "Item", "id", "title")
        graph.set_embedder(MockEmbedder(dimension=3))

        r1 = graph.embed_texts("Item", "text", show_progress=False)
        assert r1["embedded"] == 2

        # Add more nodes
        df2 = pd.DataFrame({"id": [3, 4], "title": ["C", "D"], "text": ["foo", "bar"]})
        graph.add_nodes(df2, "Item", "id", "title")

        # Re-run: only new nodes get embedded
        r2 = graph.embed_texts("Item", "text", show_progress=False)
        assert r2["embedded"] == 2
        assert r2["skipped_existing"] == 2

        # All 4 now have embeddings
        embs = graph.embeddings("Item", "text")
        assert len(embs) == 4

    def test_embed_texts_replace(self):
        """mode='all' re-embeds everything."""
        graph = kglite.KnowledgeGraph()
        df = pd.DataFrame({"id": [1, 2], "title": ["A", "B"], "text": ["hello", "world"]})
        graph.add_nodes(df, "Item", "id", "title")
        graph.set_embedder(MockEmbedder(dimension=3))

        r1 = graph.embed_texts("Item", "text", show_progress=False)
        assert r1["embedded"] == 2

        # mode='all' forces re-embed
        r2 = graph.embed_texts("Item", "text", show_progress=False, mode="all")
        assert r2["embedded"] == 2
        assert r2["skipped_existing"] == 0


class TestSearchText:
    def test_search_text_basic(self):
        """Search with text query and verify results."""
        graph = kglite.KnowledgeGraph()
        df = pd.DataFrame(
            {
                "id": [1, 2, 3, 4, 5],
                "title": ["Alpha", "Beta", "Gamma", "Delta", "Epsilon"],
                "summary": [
                    "artificial intelligence",
                    "football game",
                    "machine learning",
                    "quantum physics",
                    "basketball match",
                ],
            }
        )
        graph.add_nodes(df, "Article", "id", "title")

        model = MockEmbedder(dimension=8)
        graph.set_embedder(model)
        graph.embed_texts("Article", "summary", show_progress=False)

        # Search with text — uses "summary" which resolves to "summary_emb"
        results = graph.select("Article").search_text("summary", "artificial intelligence", top_k=3)

        assert len(results) == 3
        # First result should be the exact text match
        assert results[0]["id"] == 1
        assert results[0]["score"] == pytest.approx(1.0, abs=1e-5)
        assert "title" in results[0]

    def test_search_text_with_filter(self):
        """search_text within a filtered selection."""
        graph = kglite.KnowledgeGraph()
        df = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "title": ["A", "B", "C"],
                "category": ["tech", "sports", "tech"],
                "text": ["AI paper", "Football", "ML study"],
            }
        )
        graph.add_nodes(df, "Doc", "id", "title")

        graph.set_embedder(MockEmbedder(dimension=4))
        graph.embed_texts("Doc", "text", show_progress=False)

        results = graph.select("Doc").where({"category": "tech"}).search_text("text", "AI paper", top_k=5)

        # Only tech docs
        ids = [r["id"] for r in results]
        assert 2 not in ids  # sports doc excluded
        assert 1 in ids
        assert 3 in ids

    def test_search_text_to_df(self):
        """search_text with to_df=True returns DataFrame."""
        graph = kglite.KnowledgeGraph()
        df = pd.DataFrame({"id": [1, 2], "title": ["A", "B"], "text": ["hello", "world"]})
        graph.add_nodes(df, "Node", "id", "title")

        graph.set_embedder(MockEmbedder(dimension=3))
        graph.embed_texts("Node", "text", show_progress=False)

        result = graph.select("Node").search_text("text", "hello", top_k=2, to_df=True)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert "score" in result.columns

    def test_search_text_matches_vector_search(self):
        """search_text and vector_search produce identical results."""
        graph = kglite.KnowledgeGraph()
        df = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "title": ["A", "B", "C"],
                "text": ["alpha beta", "gamma delta", "epsilon zeta"],
            }
        )
        graph.add_nodes(df, "Node", "id", "title")

        model = MockEmbedder(dimension=5)
        graph.set_embedder(model)
        graph.embed_texts("Node", "text", show_progress=False)

        # Get query vector manually
        query_vec = model.embed(["alpha beta"])[0]

        # Both paths — search_text uses "text" → "text_emb"
        text_results = graph.select("Node").search_text("text", "alpha beta", top_k=3)
        vec_results = graph.select("Node").vector_search("text", query_vec, top_k=3)

        assert len(text_results) == len(vec_results)
        for tr, vr in zip(text_results, vec_results):
            assert tr["id"] == vr["id"]
            assert tr["score"] == pytest.approx(vr["score"], abs=1e-5)

    def test_search_text_no_model_set(self):
        """Calling search_text without set_embedder raises with skeleton."""
        graph = kglite.KnowledgeGraph()
        df = pd.DataFrame({"id": [1], "title": ["A"], "text": ["hello"]})
        graph.add_nodes(df, "Node", "id", "title")

        with pytest.raises(RuntimeError, match="set_embedder"):
            graph.select("Node").search_text("text", "hello")


# ── load/unload lifecycle ─────────────────────────────────────────────────


class MockEmbedderWithLifecycle:
    """Mock embedder that tracks load/unload calls."""

    def __init__(self, dimension=3):
        self.dimension = dimension
        self.load_count = 0
        self.unload_count = 0

    def load(self):
        self.load_count += 1

    def unload(self):
        self.unload_count += 1

    def embed(self, texts):
        import hashlib

        vectors = []
        for text in texts:
            h = hashlib.md5(text.encode()).digest()
            raw = [float(b) for b in h[: self.dimension]]
            norm = math.sqrt(sum(x * x for x in raw))
            if norm > 0:
                vectors.append([x / norm for x in raw])
            else:
                vectors.append([0.0] * self.dimension)
        return vectors


class TestEmbedderLifecycle:
    def _make_graph(self):
        graph = kglite.KnowledgeGraph()
        df = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "title": ["A", "B", "C"],
                "text": ["alpha", "beta", "gamma"],
            }
        )
        graph.add_nodes(df, "Node", "id", "title", ["text"])
        return graph

    def test_embed_texts_calls_load_and_unload(self):
        """embed_texts calls load() before and unload() after embedding."""
        graph = self._make_graph()
        model = MockEmbedderWithLifecycle(dimension=3)
        graph.set_embedder(model)

        result = graph.embed_texts("Node", "text", show_progress=False)

        assert result["embedded"] == 3
        assert model.load_count == 1
        assert model.unload_count == 1

    def test_embed_texts_calls_unload_on_empty(self):
        """unload() is called even when there are no texts to embed."""
        graph = kglite.KnowledgeGraph()
        df = pd.DataFrame({"id": [1], "title": ["A"]})
        graph.add_nodes(df, "Node", "id", "title")

        model = MockEmbedderWithLifecycle(dimension=3)
        graph.set_embedder(model)

        result = graph.embed_texts("Node", "text", show_progress=False)

        assert result["embedded"] == 0
        assert model.load_count == 1
        assert model.unload_count == 1

    def test_search_text_calls_load_and_unload(self):
        """search_text calls load() before and unload() after embedding the query."""
        graph = self._make_graph()
        model = MockEmbedderWithLifecycle(dimension=3)
        graph.set_embedder(model)
        graph.embed_texts("Node", "text", show_progress=False)
        model.load_count = 0
        model.unload_count = 0

        results = graph.select("Node").search_text("text", "alpha", top_k=2)

        assert len(results) == 2
        assert model.load_count == 1
        assert model.unload_count == 1

    def test_multiple_embed_calls_load_unload_each_time(self):
        """Each embed_texts call triggers its own load/unload cycle."""
        graph = self._make_graph()
        model = MockEmbedderWithLifecycle(dimension=3)
        graph.set_embedder(model)

        graph.embed_texts("Node", "text", show_progress=False)
        graph.embed_texts("Node", "text", show_progress=False, mode="all")

        assert model.load_count == 2
        assert model.unload_count == 2

    def test_model_without_load_unload_works(self):
        """Models without load/unload still work fine (methods are optional)."""
        graph = self._make_graph()
        model = MockEmbedder(dimension=3)  # no load/unload
        graph.set_embedder(model)

        result = graph.embed_texts("Node", "text", show_progress=False)
        assert result["embedded"] == 3

        results = graph.select("Node").search_text("text", "alpha", top_k=2)
        assert len(results) == 2

    def test_unload_called_on_embed_error(self):
        """unload() is called even if embed() raises an exception."""
        graph = self._make_graph()

        class FailingEmbedder:
            dimension = 3
            load_count = 0
            unload_count = 0

            def load(self):
                self.load_count += 1

            def unload(self):
                self.unload_count += 1

            def embed(self, texts):
                raise RuntimeError("boom")

        model = FailingEmbedder()
        graph.set_embedder(model)

        with pytest.raises(RuntimeError, match="boom"):
            graph.embed_texts("Node", "text", show_progress=False)

        assert model.load_count == 1
        assert model.unload_count == 1


# ── text_score() in Cypher (AST rewrite to vector_score) ──────────────────


class TestCypherTextScore:
    """Tests for text_score() Cypher function."""

    def _make_graph(self):
        graph = kglite.KnowledgeGraph()
        df = pd.DataFrame(
            {
                "id": [1, 2, 3, 4, 5],
                "title": ["Alpha", "Beta", "Gamma", "Delta", "Epsilon"],
                "summary": [
                    "artificial intelligence",
                    "football game",
                    "machine learning",
                    "quantum physics",
                    "basketball match",
                ],
            }
        )
        graph.add_nodes(df, "Article", "id", "title", ["summary"])
        model = MockEmbedder(dimension=8)
        graph.set_embedder(model)
        graph.embed_texts("Article", "summary", show_progress=False)
        return graph, model

    def test_text_score_in_return(self):
        """text_score in RETURN with ORDER BY + LIMIT."""
        graph, _ = self._make_graph()
        result = graph.cypher(
            "MATCH (n:Article) "
            "RETURN n.title AS title, "
            "text_score(n, 'summary', 'artificial intelligence') AS score "
            "ORDER BY score DESC LIMIT 3",
            to_df=True,
        )
        assert len(result) == 3
        # The same text should score highest (cosine = 1.0)
        assert result.iloc[0]["title"] == "Alpha"
        assert result.iloc[0]["score"] == pytest.approx(1.0, abs=1e-5)

    def test_text_score_in_where(self):
        """text_score in WHERE clause threshold filter."""
        graph, _ = self._make_graph()
        result = graph.cypher(
            "MATCH (n:Article) "
            "WHERE text_score(n, 'summary', 'artificial intelligence') > 0.99 "
            "RETURN n.title AS title",
            to_df=True,
        )
        titles = result["title"].tolist()
        assert "Alpha" in titles

    def test_text_score_with_param(self):
        """text_score with $param for query text."""
        graph, _ = self._make_graph()
        result = graph.cypher(
            "MATCH (n:Article) "
            "RETURN n.title AS title, "
            "text_score(n, 'summary', $query) AS score "
            "ORDER BY score DESC LIMIT 1",
            params={"query": "artificial intelligence"},
            to_df=True,
        )
        assert result.iloc[0]["title"] == "Alpha"

    def test_text_score_matches_vector_score(self):
        """text_score produces same results as manual embed + vector_score."""
        graph, model = self._make_graph()
        query = "artificial intelligence"
        query_vec = model.embed([query])[0]

        ts_result = graph.cypher(
            "MATCH (n:Article) RETURN n.title, text_score(n, 'summary', $q) AS score ORDER BY score DESC",
            params={"q": query},
            to_df=True,
        )
        vs_result = graph.cypher(
            "MATCH (n:Article) RETURN n.title, vector_score(n, 'summary_emb', $v) AS score ORDER BY score DESC",
            params={"v": query_vec},
            to_df=True,
        )
        assert len(ts_result) == len(vs_result)
        for i in range(len(ts_result)):
            assert ts_result.iloc[i]["n.title"] == vs_result.iloc[i]["n.title"]
            assert ts_result.iloc[i]["score"] == pytest.approx(vs_result.iloc[i]["score"], abs=1e-5)

    def test_text_score_no_embedder_error(self):
        """text_score without set_embedder raises helpful error."""
        graph = kglite.KnowledgeGraph()
        df = pd.DataFrame({"id": [1], "title": ["A"], "summary": ["hello"]})
        graph.add_nodes(df, "Article", "id", "title", ["summary"])
        graph.set_embeddings("Article", "summary", {1: [1.0, 0.0, 0.0]})

        with pytest.raises(kglite.KgError, match="set_embedder"):
            graph.cypher("MATCH (n:Article) RETURN text_score(n, 'summary', 'hello') AS score")

    def test_text_score_wrong_arg_count(self):
        """text_score with wrong number of args raises clear error."""
        graph, _ = self._make_graph()
        with pytest.raises(kglite.KgError, match="3 arguments"):
            graph.cypher("MATCH (n:Article) RETURN text_score(n, 'summary') AS score")

    def test_text_score_non_string_column(self):
        """text_score with non-string column name raises."""
        graph, _ = self._make_graph()
        with pytest.raises(kglite.KgError, match="string literal"):
            graph.cypher("MATCH (n:Article) RETURN text_score(n, n.summary, 'hello') AS score")

    def test_text_score_fused_topk(self):
        """text_score benefits from fused top-k optimization."""
        graph, _ = self._make_graph()
        plan = graph.cypher(
            "EXPLAIN MATCH (n:Article) "
            "RETURN n.title AS title, "
            "text_score(n, 'summary', 'artificial intelligence') AS score "
            "ORDER BY score DESC LIMIT 2"
        )
        ops = [r["operation"] for r in plan.to_list()]
        assert any("FusedVectorScoreTopK" in op for op in ops)

    def test_text_score_load_unload_lifecycle(self):
        """text_score triggers embedder load/unload lifecycle."""
        graph = kglite.KnowledgeGraph()
        df = pd.DataFrame(
            {
                "id": [1, 2],
                "title": ["A", "B"],
                "text": ["hello", "world"],
            }
        )
        graph.add_nodes(df, "Node", "id", "title", ["text"])
        model = MockEmbedderWithLifecycle(dimension=3)
        graph.set_embedder(model)
        graph.embed_texts("Node", "text", show_progress=False)
        model.load_count = 0
        model.unload_count = 0

        graph.cypher("MATCH (n:Node) RETURN text_score(n, 'text', 'hello') AS score")
        assert model.load_count == 1
        assert model.unload_count == 1

    def test_text_score_with_property_filter(self):
        """text_score combined with WHERE property filter."""
        graph, _ = self._make_graph()
        graph.cypher("MATCH (n:Article) WHERE n.id = 1 SET n.category = 'tech'")
        graph.cypher("MATCH (n:Article) WHERE n.id = 3 SET n.category = 'tech'")

        result = graph.cypher(
            "MATCH (n:Article) "
            "WHERE n.category = 'tech' "
            "RETURN n.title, "
            "text_score(n, 'summary', 'machine learning') AS score "
            "ORDER BY score DESC",
            to_df=True,
        )
        assert len(result) == 2
        titles = result["n.title"].tolist()
        assert "Alpha" in titles or "Gamma" in titles

    def test_text_score_regular_queries_unaffected(self):
        """Regular queries without text_score work normally."""
        graph, _ = self._make_graph()
        result = graph.cypher(
            "MATCH (n:Article) RETURN n.title LIMIT 3",
            to_df=True,
        )
        assert len(result) == 3


# ── Poincaré metric tests ────────────────────────────────────────────────


@pytest.fixture
def graph_with_poincare_embeddings():
    """Create a graph with embeddings inside the Poincaré ball (norms < 1)."""
    graph = kglite.KnowledgeGraph()

    df = pd.DataFrame(
        {
            "id": ["root", "mid_a", "mid_b", "leaf_a", "leaf_b"],
            "title": ["Root", "MidA", "MidB", "LeafA", "LeafB"],
        }
    )
    graph.add_nodes(df, "Concept", "id", "title")

    # Embeddings inside the unit ball — norm encodes hierarchy depth
    embeddings = {
        "root": [0.0, 0.0, 0.0],  # origin = root (norm 0)
        "mid_a": [0.3, 0.2, 0.1],  # moderate depth (norm ~0.374)
        "mid_b": [0.2, 0.3, 0.1],  # moderate depth, similar to mid_a
        "leaf_a": [0.8, 0.3, 0.1],  # near boundary (norm ~0.860)
        "leaf_b": [0.7, 0.5, 0.2],  # near boundary (norm ~0.894)
    }
    graph.set_embeddings("Concept", "title", embeddings)
    return graph


class TestPoincareMetric:
    def test_vector_search_poincare(self, graph_with_poincare_embeddings):
        """Poincaré vector search returns results ordered by hyperbolic distance."""
        graph = graph_with_poincare_embeddings
        results = graph.select("Concept").vector_search(
            "title",
            [0.3, 0.2, 0.1],  # query = mid_a's embedding
            top_k=5,
            metric="poincare",
        )
        assert len(results) == 5
        # mid_a should be closest to itself (distance 0, score 0.0)
        assert results[0]["id"] == "mid_a"
        assert results[0]["score"] == pytest.approx(0.0, abs=1e-4)

    def test_poincare_identical_vectors_zero_distance(self, graph_with_poincare_embeddings):
        """Identical vectors have Poincaré distance 0 (score 0.0)."""
        graph = graph_with_poincare_embeddings
        results = graph.select("Concept").vector_search(
            "title",
            [0.0, 0.0, 0.0],  # root's embedding
            top_k=1,
            metric="poincare",
        )
        assert results[0]["id"] == "root"
        assert results[0]["score"] == pytest.approx(0.0, abs=1e-5)

    def test_poincare_boundary_larger_distance(self, graph_with_poincare_embeddings):
        """Vectors near the boundary have larger Poincaré distances from origin."""
        graph = graph_with_poincare_embeddings
        results = graph.select("Concept").vector_search(
            "title",
            [0.0, 0.0, 0.0],
            top_k=5,
            metric="poincare",
        )
        scores = {r["id"]: r["score"] for r in results}
        # Root is closest (score 0), leaves are farthest (most negative scores)
        assert scores["root"] > scores["mid_a"]
        assert scores["mid_a"] > scores["leaf_a"]

    def test_cypher_vector_score_poincare(self, graph_with_poincare_embeddings):
        """vector_score() with 'poincare' metric works in Cypher."""
        graph = graph_with_poincare_embeddings
        result = graph.cypher(
            "MATCH (n:Concept) "
            "RETURN n.id AS id, "
            "vector_score(n, 'title_emb', [0.3, 0.2, 0.1], 'poincare') AS score "
            "ORDER BY score DESC LIMIT 1",
            to_df=True,
        )
        assert result.iloc[0]["id"] == "mid_a"
        assert result.iloc[0]["score"] == pytest.approx(0.0, abs=1e-4)


class TestEmbeddingNorm:
    def test_embedding_norm_basic(self, graph_with_poincare_embeddings):
        """embedding_norm() returns L2 norm of embedding vector."""
        graph = graph_with_poincare_embeddings
        result = graph.cypher(
            "MATCH (n:Concept) RETURN n.id AS id, embedding_norm(n, 'title_emb') AS norm ORDER BY norm ASC",
            to_df=True,
        )
        # Root has norm 0 (at origin), leaves have highest norms
        assert result.iloc[0]["id"] == "root"
        assert result.iloc[0]["norm"] == pytest.approx(0.0, abs=1e-5)

    def test_embedding_norm_hierarchy_order(self, graph_with_poincare_embeddings):
        """Norms reflect hierarchy depth: root < mid < leaf."""
        graph = graph_with_poincare_embeddings
        result = graph.cypher(
            "MATCH (n:Concept) RETURN n.id AS id, embedding_norm(n, 'title_emb') AS norm",
            to_df=True,
        )
        norms = dict(zip(result["id"], result["norm"]))
        assert norms["root"] < norms["mid_a"]
        assert norms["mid_a"] < norms["leaf_a"]

    def test_embedding_norm_values(self, graph_with_poincare_embeddings):
        """embedding_norm() computes correct L2 norms."""
        graph = graph_with_poincare_embeddings
        result = graph.cypher(
            "MATCH (n:Concept {id: 'mid_a'}) RETURN embedding_norm(n, 'title_emb') AS norm",
            to_df=True,
        )
        expected = math.sqrt(0.3**2 + 0.2**2 + 0.1**2)
        assert result.iloc[0]["norm"] == pytest.approx(expected, abs=1e-5)

    def test_embedding_norm_missing_property_error(self, graph_with_poincare_embeddings):
        """embedding_norm() with wrong property raises error."""
        graph = graph_with_poincare_embeddings
        with pytest.raises(kglite.KgError, match="no embedding"):
            graph.cypher("MATCH (n:Concept) RETURN embedding_norm(n, 'nonexistent') AS norm")

    def test_embedding_norm_wrong_arg_count(self, graph_with_poincare_embeddings):
        """embedding_norm() with wrong number of args raises error."""
        graph = graph_with_poincare_embeddings
        with pytest.raises(kglite.KgError, match="requires 2 arguments"):
            graph.cypher("MATCH (n:Concept) RETURN embedding_norm(n) AS norm")

    def test_embedding_norm_in_where(self, graph_with_poincare_embeddings):
        """embedding_norm() can be used in WHERE for filtering."""
        graph = graph_with_poincare_embeddings
        result = graph.cypher(
            "MATCH (n:Concept) WHERE embedding_norm(n, 'title_emb') < 0.5 RETURN n.id AS id",
            to_df=True,
        )
        ids = result["id"].tolist()
        assert "root" in ids
        assert "mid_a" in ids  # norm ~0.374
        assert "leaf_a" not in ids  # norm ~0.860


class TestStoredMetric:
    def test_set_embeddings_with_metric(self):
        """set_embeddings(metric=...) stores the metric and list_embeddings shows it."""
        graph = kglite.KnowledgeGraph()
        df = pd.DataFrame({"id": [1, 2], "title": ["A", "B"]})
        graph.add_nodes(df, "Node", "id", "title")
        graph.set_embeddings("Node", "title", {1: [0.1, 0.2], 2: [0.3, 0.4]}, metric="poincare")

        stores = graph.list_embeddings()
        assert len(stores) == 1
        assert stores[0]["metric"] == "poincare"

    def test_stored_metric_used_as_default(self):
        """vector_search() uses the stored metric when none is explicitly provided."""
        graph = kglite.KnowledgeGraph()
        df = pd.DataFrame({"id": ["a", "b", "c"], "title": ["A", "B", "C"]})
        graph.add_nodes(df, "Node", "id", "title")

        # Embeddings inside unit ball
        graph.set_embeddings(
            "Node",
            "title",
            {"a": [0.0, 0.0], "b": [0.3, 0.3], "c": [0.8, 0.4]},
            metric="poincare",
        )

        # No explicit metric — should use stored "poincare"
        results_default = graph.select("Node").vector_search("title", [0.0, 0.0], top_k=3)
        # Explicit poincare — should give same results
        results_explicit = graph.select("Node").vector_search("title", [0.0, 0.0], top_k=3, metric="poincare")

        assert [r["id"] for r in results_default] == [r["id"] for r in results_explicit]
        assert [r["score"] for r in results_default] == pytest.approx([r["score"] for r in results_explicit], abs=1e-6)

    def test_explicit_metric_overrides_stored(self):
        """Explicit metric= arg overrides the stored metric."""
        graph = kglite.KnowledgeGraph()
        df = pd.DataFrame({"id": ["a", "b"], "title": ["A", "B"]})
        graph.add_nodes(df, "Node", "id", "title")

        graph.set_embeddings("Node", "title", {"a": [0.1, 0.2], "b": [0.3, 0.4]}, metric="poincare")

        # Override with cosine
        results = graph.select("Node").vector_search("title", [0.1, 0.2], top_k=2, metric="cosine")
        # Should work and return cosine-based scores
        assert len(results) == 2
        # Cosine of identical vectors is 1.0
        assert results[0]["score"] == pytest.approx(1.0, abs=1e-4)

    def test_no_stored_metric_defaults_to_cosine(self):
        """Without stored metric, vector_search() defaults to cosine."""
        graph = kglite.KnowledgeGraph()
        df = pd.DataFrame({"id": [1, 2], "title": ["A", "B"]})
        graph.add_nodes(df, "Node", "id", "title")

        # No metric= arg
        graph.set_embeddings("Node", "title", {1: [1.0, 0.0], 2: [0.0, 1.0]})

        stores = graph.list_embeddings()
        assert stores[0]["metric"] == "cosine"

        results = graph.select("Node").vector_search("title", [1.0, 0.0], top_k=2)
        # Cosine: identical to [1,0] → score 1.0
        assert results[0]["id"] == 1
        assert results[0]["score"] == pytest.approx(1.0, abs=1e-5)

    def test_stored_metric_in_cypher_vector_score(self):
        """vector_score() in Cypher uses stored metric when no explicit metric given."""
        graph = kglite.KnowledgeGraph()
        df = pd.DataFrame({"id": ["a", "b", "c"], "title": ["A", "B", "C"]})
        graph.add_nodes(df, "Node", "id", "title")

        graph.set_embeddings(
            "Node",
            "title",
            {"a": [0.0, 0.0], "b": [0.3, 0.3], "c": [0.8, 0.4]},
            metric="poincare",
        )

        # No metric arg in vector_score — should use stored poincare
        result_default = graph.cypher(
            "MATCH (n:Node) "
            "RETURN n.id AS id, vector_score(n, 'title_emb', [0.0, 0.0]) AS score "
            "ORDER BY score DESC LIMIT 1",
            to_df=True,
        )
        result_explicit = graph.cypher(
            "MATCH (n:Node) "
            "RETURN n.id AS id, vector_score(n, 'title_emb', [0.0, 0.0], 'poincare') AS score "
            "ORDER BY score DESC LIMIT 1",
            to_df=True,
        )

        assert result_default.iloc[0]["id"] == result_explicit.iloc[0]["id"]
        assert result_default.iloc[0]["score"] == pytest.approx(result_explicit.iloc[0]["score"], abs=1e-6)


class TestImportEmbeddingsSilentDrop:
    """Regression: import_embeddings should warn when imported=0 and the file
    contained data, so users notice the silent-drop case (e.g. .kgle exported
    from a graph with different IDs / types — the symptom the MCP-servers
    project's wishlist flagged with 1923 skipped, 0 imported)."""

    def _build_source_graph(self, tmp_path):
        """Source: 3 articles with embeddings, exported to a .kgle path."""
        src = kglite.KnowledgeGraph()
        df = pd.DataFrame(
            {
                "id": ["a1", "a2", "a3"],
                "title": ["A1", "A2", "A3"],
                "summary": ["s1", "s2", "s3"],
            }
        )
        src.add_nodes(df, "Article", "id", "title")
        src.set_embeddings(
            "Article",
            "summary",
            {"a1": [1.0, 0.0], "a2": [0.0, 1.0], "a3": [0.5, 0.5]},
        )
        kgle = os.path.join(tmp_path, "src.kgle")
        src.export_embeddings(kgle)
        return kgle

    def test_round_trip_succeeds_silently(self, tmp_path):
        """Same-graph re-import: no warning, all embeddings imported."""
        kgle = self._build_source_graph(tmp_path)

        target = kglite.KnowledgeGraph()
        df = pd.DataFrame(
            {
                "id": ["a1", "a2", "a3"],
                "title": ["A1", "A2", "A3"],
                "summary": ["s1", "s2", "s3"],
            }
        )
        target.add_nodes(df, "Article", "id", "title")

        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            stats = target.import_embeddings(kgle)
        user_warnings = [w for w in captured if issubclass(w.category, UserWarning)]
        assert user_warnings == [], (
            f"Did not expect a UserWarning on a clean round-trip, got: {[str(w.message) for w in user_warnings]}"
        )
        assert stats["imported"] == 3
        assert stats["skipped"] == 0
        assert stats["dropped_stores"] == 0
        assert stats["stores"] == 1

    def test_warns_when_all_ids_mismatch(self, tmp_path):
        """imported=0, skipped>0 → loud UserWarning, dropped_stores reported."""
        kgle = self._build_source_graph(tmp_path)

        target = kglite.KnowledgeGraph()
        df = pd.DataFrame({"id": ["x1", "x2"], "title": ["X1", "X2"], "summary": ["s", "s"]})
        target.add_nodes(df, "Article", "id", "title")

        with pytest.warns(UserWarning, match="imported 0 embeddings"):
            stats = target.import_embeddings(kgle)
        assert stats["imported"] == 0
        assert stats["skipped"] == 3
        assert stats["dropped_stores"] == 1
        assert stats["stores"] == 0

    def test_warns_when_node_type_missing(self, tmp_path):
        """File has Article but target only has Author → store dropped, warn."""
        kgle = self._build_source_graph(tmp_path)

        target = kglite.KnowledgeGraph()
        df = pd.DataFrame({"id": ["p1"], "title": ["P1"], "bio": ["b"]})
        target.add_nodes(df, "Author", "id", "title")

        with pytest.warns(UserWarning, match="imported 0 embeddings"):
            stats = target.import_embeddings(kgle)
        assert stats["imported"] == 0
        assert stats["dropped_stores"] == 1

    def test_partial_drop_warns(self, tmp_path):
        """Two stores; one type matches, the other is missing → per-store warning."""
        src = kglite.KnowledgeGraph()
        df_articles = pd.DataFrame({"id": ["a1", "a2"], "title": ["A1", "A2"], "summary": ["s1", "s2"]})
        df_authors = pd.DataFrame({"id": ["p1"], "title": ["P1"], "bio": ["b1"]})
        src.add_nodes(df_articles, "Article", "id", "title")
        src.add_nodes(df_authors, "Author", "id", "title")
        src.set_embeddings("Article", "summary", {"a1": [1.0, 0.0], "a2": [0.0, 1.0]})
        src.set_embeddings("Author", "bio", {"p1": [0.5, 0.5]})
        kgle = os.path.join(tmp_path, "two_stores.kgle")
        src.export_embeddings(kgle)

        target = kglite.KnowledgeGraph()
        target.add_nodes(df_articles, "Article", "id", "title")

        with pytest.warns(UserWarning, match="dropped"):
            stats = target.import_embeddings(kgle)
        assert stats["imported"] == 2
        assert stats["skipped"] == 1
        assert stats["dropped_stores"] == 1
        assert stats["stores"] == 1

    def test_empty_file_no_warning(self, tmp_path):
        """Edge case: file with zero embeddings shouldn't warn (nothing to drop)."""
        src = kglite.KnowledgeGraph()
        df = pd.DataFrame({"id": ["a1"], "title": ["A1"]})
        src.add_nodes(df, "Article", "id", "title")
        # No set_embeddings call → file will contain no stores.
        kgle = os.path.join(tmp_path, "empty.kgle")
        src.export_embeddings(kgle)

        target = kglite.KnowledgeGraph()
        target.add_nodes(df, "Article", "id", "title")
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            stats = target.import_embeddings(kgle)
        user_warnings = [w for w in captured if issubclass(w.category, UserWarning)]
        assert user_warnings == []
        assert stats["imported"] == 0
        assert stats["skipped"] == 0
        assert stats["dropped_stores"] == 0


class TestEmbeddingDiagnostics:
    """Coverage diagnostics for embedding stores. Companion to
    list_embeddings() that surfaces the silent-drop case (store exists
    but underlying property is absent in the current graph)."""

    def test_status_embedded(self):
        """Nodes have property AND store exists → status='embedded'."""
        graph = kglite.KnowledgeGraph()
        df = pd.DataFrame({"id": [1, 2, 3], "title": ["a", "b", "c"], "summary": ["x", "y", "z"]})
        graph.add_nodes(df, "Article", "id", "title")
        graph.set_embeddings("Article", "summary", {1: [1.0, 0.0], 2: [0.0, 1.0], 3: [0.5, 0.5]})

        diag = graph.embedding_diagnostics()
        rows = [d for d in diag if d["text_column"] == "summary"]
        assert len(rows) == 1
        r = rows[0]
        assert r["node_type"] == "Article"
        assert r["embedding_key"] == "summary_emb"
        assert r["nodes_with_property"] == 3
        assert r["nodes_embedded"] == 3
        assert r["dimension"] == 2
        assert r["metric"] == "cosine"
        assert r["status"] == "embedded"

    def test_status_embeddable(self):
        """Nodes have property but no store → status='embeddable'."""
        graph = kglite.KnowledgeGraph()
        df = pd.DataFrame({"id": [1, 2], "title": ["a", "b"], "summary": ["text1", "text2"]})
        graph.add_nodes(df, "Article", "id", "title")

        diag = graph.embedding_diagnostics()
        rows = [d for d in diag if d["text_column"] == "summary"]
        assert len(rows) == 1
        r = rows[0]
        assert r["status"] == "embeddable"
        assert r["nodes_with_property"] == 2
        assert r["nodes_embedded"] == 0
        assert r["dimension"] is None
        assert r["metric"] is None

    def test_status_store_orphan(self, tmp_path):
        """Store exists but the property is absent on every node → 'store_orphan'.
        Reproduces the silent-drop case: import a .kgle into a graph whose IDs
        don't match — the store would only be created if at least one matched,
        so we simulate via direct API construction instead."""
        # Build source with embeddings, export.
        src = kglite.KnowledgeGraph()
        df_src = pd.DataFrame({"id": [1, 2], "title": ["a", "b"], "summary": ["x", "y"]})
        src.add_nodes(df_src, "Article", "id", "title")
        src.set_embeddings("Article", "summary", {1: [1.0, 0.0], 2: [0.0, 1.0]})
        kgle = os.path.join(tmp_path, "src.kgle")
        src.export_embeddings(kgle)

        # Target: same IDs (so the import succeeds and creates a store)
        # but no `summary` property on any node — simulates a target
        # that lost the source column between releases.
        target = kglite.KnowledgeGraph()
        df_target = pd.DataFrame({"id": [1, 2], "title": ["a", "b"]})
        target.add_nodes(df_target, "Article", "id", "title")
        target.import_embeddings(kgle)

        diag = target.embedding_diagnostics()
        rows = [d for d in diag if d["text_column"] == "summary"]
        assert len(rows) == 1
        r = rows[0]
        assert r["status"] == "store_orphan"
        assert r["nodes_with_property"] == 0
        assert r["nodes_embedded"] == 2

    def test_filter_by_node_type(self):
        """Passing node_type scopes the scan; other types are not in output."""
        graph = kglite.KnowledgeGraph()
        articles = pd.DataFrame({"id": [1, 2], "title": ["a", "b"], "summary": ["x", "y"]})
        authors = pd.DataFrame({"id": [10], "title": ["alice"], "bio": ["story"]})
        graph.add_nodes(articles, "Article", "id", "title")
        graph.add_nodes(authors, "Author", "id", "title")

        diag = graph.embedding_diagnostics(node_type="Article")
        types_in_output = {d["node_type"] for d in diag}
        assert types_in_output == {"Article"}

    def test_unknown_type_raises(self):
        graph = kglite.KnowledgeGraph()
        df = pd.DataFrame({"id": [1], "title": ["a"]})
        graph.add_nodes(df, "Article", "id", "title")
        with pytest.raises(ValueError, match="does not exist"):
            graph.embedding_diagnostics(node_type="Nonexistent")

    def test_empty_graph_returns_empty(self):
        graph = kglite.KnowledgeGraph()
        assert graph.embedding_diagnostics() == []

    def test_builtin_title_column_counted(self):
        """A `title_emb` store keys against the builtin title — diagnostics
        should report nodes_with_property = total node count, not 0."""
        graph = kglite.KnowledgeGraph()
        df = pd.DataFrame({"id": [1, 2, 3], "title": ["a", "b", "c"]})
        graph.add_nodes(df, "Article", "id", "title")
        graph.set_embeddings("Article", "title", {1: [1.0, 0.0], 2: [0.0, 1.0], 3: [0.5, 0.5]})
        diag = graph.embedding_diagnostics()
        rows = [d for d in diag if d["text_column"] == "title"]
        assert len(rows) == 1
        r = rows[0]
        assert r["status"] == "embedded"
        assert r["nodes_with_property"] == 3
        assert r["nodes_embedded"] == 3

    def test_status_embedded_after_save_reload(self, tmp_path):
        """Regression: after save+reload, nodes use PropertyStorage::Columnar
        whose `property_iter` yields nothing — the pre-fix
        embedding_diagnostics would report nodes_with_property=0 for a
        healthy graph and flip status to 'store_orphan'. Verify it now
        sees the columnar properties correctly."""
        # Build, embed, save.
        src = kglite.KnowledgeGraph()
        df = pd.DataFrame(
            {
                "id": [f"f{i}" for i in range(5)],
                "title": [f"fn_{i}" for i in range(5)],
                "docstring": [f"doc string number {i}" for i in range(5)],
            }
        )
        src.add_nodes(df, "Function", "id", "title")
        src.set_embeddings(
            "Function",
            "docstring",
            {f"f{i}": [float(i), float(5 - i)] for i in range(5)},
        )
        kgl = os.path.join(tmp_path, "code.kgl")
        src.save(kgl)

        # Reload — this is the path that triggers columnar storage on the
        # docstring property in the MCP operator's bug report.
        reloaded = kglite.load(kgl)
        diag = reloaded.embedding_diagnostics()
        rows = [d for d in diag if d["text_column"] == "docstring"]
        assert len(rows) == 1, f"expected one diagnostic row, got: {diag}"
        r = rows[0]
        assert r["nodes_with_property"] == 5, (
            f"expected 5 nodes_with_property post-reload, got {r['nodes_with_property']} "
            f"(if 0, the column-store scan regressed again)"
        )
        assert r["nodes_embedded"] == 5
        assert r["status"] == "embedded"

    def test_filter_by_type_surfaces_embeddable(self):
        """Regression B2: embedding_diagnostics(node_type='X') used to return
        [] for types with a string property but no embedding store, hiding
        the 'embeddable' status entirely. Verify the embeddable row is now
        emitted under a type filter."""
        graph = kglite.KnowledgeGraph()
        df = pd.DataFrame(
            {
                "id": [1, 2],
                "title": ["A", "B"],
                "summary": ["body one", "body two"],
            }
        )
        graph.add_nodes(df, "Article", "id", "title")
        # No set_embeddings — Article has a `summary` string property but
        # no `summary_emb` store yet.
        diag = graph.embedding_diagnostics(node_type="Article")
        rows = [d for d in diag if d["text_column"] == "summary"]
        assert len(rows) == 1, f"embeddable row should appear; got {diag}"
        assert rows[0]["status"] == "embeddable"
        assert rows[0]["nodes_with_property"] == 2
        assert rows[0]["nodes_embedded"] == 0


class TestVectorSearchPropertiesAfterReload:
    """Regression (kglite-docs 2026-05-28): vector_search dropped all
    non-core properties after save+load. property_iter() yields nothing
    for PropertyStorage::Columnar (the post-reload variant)."""

    def test_full_properties_returned_after_reload(self, graph_with_embeddings, tmp_path):
        graph = graph_with_embeddings
        save_path = str(tmp_path / "round_trip.kgl")
        graph.save(save_path)

        loaded = kglite.load(save_path)
        results = loaded.select("Article").vector_search("summary", [1.0, 0.0, 0.0], top_k=3)
        assert len(results) == 3

        top = results[0]
        # Core fields are present in both pre- and post-reload outputs.
        assert {"id", "title", "type", "score"}.issubset(top.keys())
        # The kglite-docs bug: these non-core fields used to disappear.
        assert "category" in top, f"missing 'category' in {top.keys()}"
        assert "summary" in top, f"missing 'summary' in {top.keys()}"
        # Sanity: values round-trip.
        assert top["id"] == 1
        assert top["category"] == "politics"
        assert top["summary"] == "alpha text"

    def test_to_df_full_columns_after_reload(self, graph_with_embeddings, tmp_path):
        graph = graph_with_embeddings
        save_path = str(tmp_path / "round_trip_df.kgl")
        graph.save(save_path)

        loaded = kglite.load(save_path)
        df = loaded.select("Article").vector_search("summary", [1.0, 0.0, 0.0], top_k=3, to_df=True)
        assert {"id", "title", "type", "score", "category", "summary"}.issubset(df.columns)


class TestAddEmbeddings:
    """add_embeddings appends to an existing store without read-merge-write."""

    def test_add_embeddings_creates_store_on_first_call(self):
        graph = kglite.KnowledgeGraph()
        df = pd.DataFrame({"id": [1, 2, 3], "title": ["A", "B", "C"], "summary": ["a", "b", "c"]})
        graph.add_nodes(df, "Article", "id", "title")
        result = graph.add_embeddings(
            "Article",
            "summary",
            {1: [0.1, 0.2], 2: [0.3, 0.4]},
        )
        assert result["embeddings_stored"] == 2
        assert result["dimension"] == 2
        assert result["skipped"] == 0
        assert result["store_created"] is True

    def test_add_embeddings_extends_existing_store(self):
        graph = kglite.KnowledgeGraph()
        df = pd.DataFrame({"id": [1, 2, 3], "title": ["A", "B", "C"], "summary": ["a", "b", "c"]})
        graph.add_nodes(df, "Article", "id", "title")
        graph.set_embeddings("Article", "summary", {1: [0.1, 0.2]})
        result = graph.add_embeddings("Article", "summary", {2: [0.3, 0.4], 3: [0.5, 0.6]})
        assert result["embeddings_stored"] == 3
        assert result["store_created"] is False
        # Original entry still present.
        all_embs = graph.embeddings("Article", "summary")
        assert all_embs[1] == pytest.approx([0.1, 0.2])
        assert all_embs[2] == pytest.approx([0.3, 0.4])

    def test_add_embeddings_dim_mismatch_rejected(self):
        graph = kglite.KnowledgeGraph()
        df = pd.DataFrame({"id": [1, 2], "title": ["A", "B"], "summary": ["a", "b"]})
        graph.add_nodes(df, "Article", "id", "title")
        graph.set_embeddings("Article", "summary", {1: [0.1, 0.2]})
        with pytest.raises(ValueError, match="Inconsistent embedding dimension"):
            graph.add_embeddings("Article", "summary", {2: [0.3, 0.4, 0.5]})

    def test_add_embeddings_skips_unknown_ids(self):
        graph = kglite.KnowledgeGraph()
        df = pd.DataFrame({"id": [1, 2], "title": ["A", "B"], "summary": ["a", "b"]})
        graph.add_nodes(df, "Article", "id", "title")
        result = graph.add_embeddings(
            "Article",
            "summary",
            {1: [0.1, 0.2], 99: [0.3, 0.4]},
        )
        assert result["embeddings_stored"] == 1
        assert result["skipped"] == 1


class TestSetEmbedderUnbind:
    """set_embedder(None) unbinds without raising."""

    def test_set_embedder_accepts_none_when_unbound(self):
        g = kglite.KnowledgeGraph()
        # No-op: unbind when nothing was bound. Must not raise.
        g.set_embedder(None)

    def test_set_embedder_none_after_registering(self):
        class FakeEmbedder:
            dimension = 3

            def embed(self, texts):
                return [[0.0, 0.0, 0.0] for _ in texts]

        g = kglite.KnowledgeGraph()
        g.set_embedder(FakeEmbedder())
        g.set_embedder(None)  # must not raise


class TestEmbeddingDiagnosticsLengthStats:
    """embedding_diagnostics now exposes length_stats so callers can
    filter candidate columns by length distribution + cardinality."""

    def test_length_stats_fields_present(self):
        g = kglite.KnowledgeGraph()
        df = pd.DataFrame(
            {
                "id": [1, 2, 3, 4],
                "title": ["A", "B", "C", "D"],
                "body": [
                    "long-enough paragraph one",
                    "another long-enough paragraph",
                    "third entry with detail",
                    "fourth long-enough entry",
                ],
                "status": ["new", "new", "done", "new"],
            }
        )
        g.add_nodes(df, "Note", "id", "title")
        diag = g.embedding_diagnostics(node_type="Note")
        body_row = next(d for d in diag if d["text_column"] == "body")
        status_row = next(d for d in diag if d["text_column"] == "status")

        for row in (body_row, status_row):
            stats = row["length_stats"]
            assert set(stats.keys()) == {"mean_length", "max_length", "distinct_count", "distinct_ratio"}

        # body: long strings, mostly distinct.
        assert body_row["length_stats"]["mean_length"] >= 20
        assert body_row["length_stats"]["distinct_count"] == 4
        assert body_row["length_stats"]["distinct_ratio"] == pytest.approx(1.0)

        # status: short strings, low cardinality.
        assert status_row["length_stats"]["mean_length"] < 5
        assert status_row["length_stats"]["distinct_count"] == 2
        assert status_row["length_stats"]["distinct_ratio"] == pytest.approx(0.5)


class TestNormCacheParity:
    """The cosine fast path reads a per-vector L2 norm cached in the
    EmbeddingStore instead of recomputing it on every query. The cache is
    derived from the vectors (``#[serde(skip)]``), so it must be rebuilt on
    load — these tests pin that the cached-norm path matches a from-scratch
    reference and survives a ``.kgl`` round-trip identically."""

    @staticmethod
    def _ref_cosine(q, v):
        dot = sum(a * b for a, b in zip(q, v))
        na = math.sqrt(sum(a * a for a in q))
        nb = math.sqrt(sum(b * b for b in v))
        return dot / (na * nb) if na * nb > 0 else 0.0

    def _build(self, n=300, d=48, seed=11):
        import random

        rng = random.Random(seed)
        rows = {
            "id": list(range(n)),
            "title": [f"n{i}" for i in range(n)],
            "summary": [f"text {i}" for i in range(n)],
        }
        g = kglite.KnowledgeGraph()
        g.add_nodes(pd.DataFrame(rows), "Doc", "id", "title")
        emb = {i: [rng.gauss(0, 1) for _ in range(d)] for i in range(n)}
        g.set_embeddings("Doc", "summary", emb, metric="cosine")
        q = [rng.gauss(0, 1) for _ in range(d)]
        return g, emb, q

    def _topk(self, g, q, k=10):
        rows = g.cypher(
            "MATCH (d:Doc) RETURN d.id AS id, vector_score(d, 'summary_emb', $q) AS s ORDER BY s DESC LIMIT $k",
            params={"q": q, "k": k},
        )
        return [(r["id"], r["s"]) for r in rows]

    def test_cosine_matches_reference(self):
        g, emb, q = self._build()
        got = self._topk(g, q)
        ref = sorted(((i, self._ref_cosine(q, emb[i])) for i in emb), key=lambda t: -t[1])[:10]
        assert [i for i, _ in got] == [i for i, _ in ref]
        for (_, gs), (_, rs) in zip(got, ref):
            assert abs(gs - rs) < 1e-4

    def test_roundtrip_identical(self):
        g, _emb, q = self._build()
        before = self._topk(g, q)
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "g.kgl")
            g.save(p)
            g2 = kglite.load(p)
        after = self._topk(g2, q)
        assert [i for i, _ in before] == [i for i, _ in after]
        for (_, b), (_, a) in zip(before, after):
            assert abs(b - a) < 1e-6

    def test_compact_preserves_results(self):
        """compact() remaps embedding slots wholesale; the norm cache must be
        rebuilt so cosine results are unchanged after a delete + compact."""
        g, emb, q = self._build(n=120)
        before = self._topk(g, q, k=5)
        # Delete a node that is NOT in the top-5, then compact.
        keep_ids = {i for i, _ in before}
        victim = next(i for i in emb if i not in keep_ids)
        g.cypher("MATCH (d:Doc) WHERE d.id = $v DETACH DELETE d", params={"v": victim})
        g.compact()
        after = self._topk(g, q, k=5)
        assert [i for i, _ in before] == [i for i, _ in after]
        for (_, b), (_, a) in zip(before, after):
            assert abs(b - a) < 1e-6
