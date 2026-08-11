package io.github.kkollsga.kglite;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/** Ingesting vectors from Java and querying the store by a caller-supplied vector. */
class EmbeddingTest {

    /** Three Note nodes with a {@code body} column and 2-D vectors, id 1 nearest [1,0]. */
    private static KnowledgeGraph notes() {
        KnowledgeGraph graph = KnowledgeGraph.createInMemory();
        graph.cypher("CREATE (:Note {id: 1, title: 'a', body: 'rust'})");
        graph.cypher("CREATE (:Note {id: 2, title: 'b', body: 'java'})");
        graph.cypher("CREATE (:Note {id: 3, title: 'c', body: 'ruby'})");
        return graph;
    }

    private static Map<Object, float[]> vectors() {
        Map<Object, float[]> byId = new LinkedHashMap<>();
        byId.put(1L, new float[] {1.0f, 0.0f});
        byId.put(2L, new float[] {0.0f, 1.0f});
        byId.put(3L, new float[] {0.9f, 0.1f}); // closest to id 1 after itself
        return byId;
    }

    @Test
    @DisplayName("setEmbeddings then vector_score ranks by a caller-supplied float[] query vector")
    void ingestThenQueryByVector() {
        try (KnowledgeGraph graph = notes()) {
            Map<String, Object> report = graph.setEmbeddings("Note", "body", vectors());
            assertEquals(3L, report.get("embeddings_stored"));
            assertEquals(2L, report.get("dimension"));
            assertEquals(0L, report.get("skipped"));
            assertEquals(Boolean.TRUE, report.get("store_created"));

            // The query vector is a float[] parameter — the marshalling this
            // feature depends on. cosine([1,0]) ranks id 1 (1.0), id 3 (~0.99),
            // id 2 (0.0).
            List<Map<String, Object>> hits = graph.query(
                    "MATCH (n:Note) RETURN n.id AS id, vector_score(n, 'body_emb', $q) AS s "
                            + "ORDER BY s DESC",
                    Map.of("q", new float[] {1.0f, 0.0f}));
            assertEquals(List.of(1L, 3L, 2L), hits.stream().map(r -> r.get("id")).toList());
            assertTrue((Double) hits.get(0).get("s") > (Double) hits.get(1).get("s"));
        }
    }

    @Test
    @DisplayName("the column-named text_score spelling scores the same store with a raw vector")
    void textScoreColumnSpelling() {
        try (KnowledgeGraph graph = notes()) {
            graph.setEmbeddings("Note", "body", vectors());
            List<Map<String, Object>> hits = graph.query(
                    "MATCH (n:Note) RETURN n.id AS id, text_score(n, 'body', $q) AS s "
                            + "ORDER BY s DESC",
                    Map.of("q", new float[] {1.0f, 0.0f}));
            assertEquals(List.of(1L, 3L, 2L), hits.stream().map(r -> r.get("id")).toList());
        }
    }

    @Test
    @DisplayName("a List<Float> query vector marshals the same as a float[]")
    void listFloatQueryVector() {
        try (KnowledgeGraph graph = notes()) {
            graph.setEmbeddings("Note", "body", vectors());
            List<Map<String, Object>> hits = graph.query(
                    "MATCH (n:Note) RETURN n.id AS id, vector_score(n, 'body_emb', $q) AS s "
                            + "ORDER BY s DESC",
                    Map.of("q", List.of(1.0f, 0.0f)));
            assertEquals(List.of(1L, 3L, 2L), hits.stream().map(r -> r.get("id")).toList());
        }
    }

    @Test
    @DisplayName("buildVectorIndex reports the corpus size and the resolved metric")
    void buildIndex() {
        try (KnowledgeGraph graph = notes()) {
            graph.setEmbeddings("Note", "body", vectors(), "cosine");
            Map<String, Object> report = graph.buildVectorIndex("Note", "body");
            assertEquals(3L, report.get("indexed"));
            assertEquals("cosine", report.get("metric"));
            assertNotNull(report.get("m"));

            // The index still answers a whole-corpus top-k the same way.
            List<Map<String, Object>> hits = graph.query(
                    "MATCH (n:Note) RETURN n.id AS id, vector_score(n, 'body_emb', $q) AS s "
                            + "ORDER BY s DESC LIMIT 2",
                    Map.of("q", new float[] {1.0f, 0.0f}));
            assertEquals(List.of(1L, 3L), hits.stream().map(r -> r.get("id")).toList());
        }
    }

    @Test
    @DisplayName("listEmbeddings projects the source column, dimension, count and metric")
    void listStores() {
        try (KnowledgeGraph graph = notes()) {
            assertEquals(List.of(), graph.listEmbeddings(), "no stores before ingest");
            graph.setEmbeddings("Note", "body", vectors(), "dot_product");

            List<Map<String, Object>> stores = graph.listEmbeddings();
            assertEquals(1, stores.size());
            Map<String, Object> store = stores.get(0);
            assertEquals("Note", store.get("node_type"));
            assertEquals("body", store.get("text_column")); // source column, not "body_emb"
            assertEquals(2L, store.get("dimension"));
            assertEquals(3L, store.get("count"));
            assertEquals("dot_product", store.get("metric"));
        }
    }

    @Test
    @DisplayName("addEmbeddings creates then extends one store, reporting store_created")
    void addUpserts() {
        try (KnowledgeGraph graph = notes()) {
            Map<Object, float[]> first = new LinkedHashMap<>();
            first.put(1L, new float[] {1.0f, 0.0f});
            Map<String, Object> created = graph.addEmbeddings("Note", "body", first);
            assertEquals(Boolean.TRUE, created.get("store_created"));
            assertEquals(1L, created.get("embeddings_stored"));

            Map<Object, float[]> more = new LinkedHashMap<>();
            more.put(2L, new float[] {0.0f, 1.0f});
            more.put(3L, new float[] {0.9f, 0.1f});
            Map<String, Object> extended = graph.addEmbeddings("Note", "body", more);
            assertEquals(Boolean.FALSE, extended.get("store_created"));
            assertEquals(3L, extended.get("embeddings_stored"));
        }
    }

    @Test
    @DisplayName("an empty batch is a no-op that writes no store")
    void emptyBatchNoOp() {
        try (KnowledgeGraph graph = notes()) {
            Map<String, Object> report = graph.setEmbeddings("Note", "body", new LinkedHashMap<>());
            assertEquals(0L, report.get("embeddings_stored"));
            assertEquals(List.of(), graph.listEmbeddings());
        }
    }

    @Test
    @DisplayName("ragged vectors are rejected before the native call")
    void raggedVectorsRejected() {
        try (KnowledgeGraph graph = notes()) {
            Map<Object, float[]> ragged = new LinkedHashMap<>();
            ragged.put(1L, new float[] {1.0f, 0.0f});
            ragged.put(2L, new float[] {0.0f, 1.0f, 0.5f});
            KgliteException e = assertThrows(
                    KgliteException.class, () -> graph.setEmbeddings("Note", "body", ragged));
            assertTrue(e.getMessage().contains("dimension"), e.getMessage());
        }
    }

    @Test
    @DisplayName("a store saved from Java reloads and scores the same after reopen")
    void savedStoreRoundTrips(@TempDir Path dir) {
        Path path = dir.resolve("notes.kgl");
        try (WriterLease lease = WriterLease.acquire(path);
                KnowledgeGraph graph = KnowledgeGraph.open(path, StorageMode.MEMORY)) {
            assertEquals(path.toAbsolutePath(), lease.path(), "the lease covers the saved path");
            graph.cypher("CREATE (:Note {id: 1, title: 'a', body: 'rust'})");
            graph.cypher("CREATE (:Note {id: 2, title: 'b', body: 'java'})");
            graph.cypher("CREATE (:Note {id: 3, title: 'c', body: 'ruby'})");
            graph.setEmbeddings("Note", "body", vectors());
            graph.buildVectorIndex("Note", "body");
            graph.save(path); // embeddings ride the checkpoint
        }

        try (KnowledgeGraph graph = KnowledgeGraph.open(path)) {
            assertEquals(1, graph.listEmbeddings().size(), "the store survived save + reopen");
            List<Map<String, Object>> hits = graph.query(
                    "MATCH (n:Note) RETURN n.id AS id, vector_score(n, 'body_emb', $q) AS s "
                            + "ORDER BY s DESC",
                    Map.of("q", new float[] {1.0f, 0.0f}));
            assertEquals(List.of(1L, 3L, 2L), hits.stream().map(r -> r.get("id")).toList());
        }
    }
}
