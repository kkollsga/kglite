package io.github.kkollsga.kglite;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/** The open / mutate / save / reopen cycle across the C ABI. */
class KnowledgeGraphTest {

    @Test
    @DisplayName("create, write, save under lease, reopen, read back the same rows")
    void roundTrip(@TempDir Path dir) {
        Path path = dir.resolve("people.kgl");

        try (WriterLease lease = WriterLease.acquire(path);
                KnowledgeGraph graph = KnowledgeGraph.open(path, StorageMode.MAPPED)) {
            assertEquals(path.toAbsolutePath(), lease.path(), "the lease covers the saved path");
            assertTrue(graph.convertedFrom().isEmpty(), "a fresh creation converts nothing");
            graph.cypher(
                    "CREATE (:Person {id: $id, title: $name, score: $score, tags: $tags})",
                    Map.of("id", 1, "name", "Ada", "score", 2.5, "tags", List.of("a", "b")));
            graph.cypher("CREATE (:Person {id: 2, title: 'Grace'})");
            graph.save(path);
        }

        assertTrue(Files.isRegularFile(path), "save produced no file at " + path);

        // Unspecified mode: comes back in the mode the checkpoint recorded.
        try (KnowledgeGraph graph = KnowledgeGraph.open(path)) {
            assertTrue(graph.convertedFrom().isEmpty(), "an unspecified open converts nothing");
            List<Map<String, Object>> rows = graph.query(
                    "MATCH (p:Person) RETURN p.id AS id, p.title AS name ORDER BY p.id");
            assertEquals(2, rows.size());
            assertEquals(List.of("id", "name"), List.copyOf(rows.get(0).keySet()),
                    "rows are keyed in column order");
            assertEquals(1L, rows.get(0).get("id"));
            assertEquals("Ada", rows.get(0).get("name"));
            assertEquals(2L, rows.get(1).get("id"));
            assertEquals("Grace", rows.get(1).get("name"));

            // Non-scalar cells survive the JSON boundary as natural Java values.
            Map<String, Object> ada = graph.query(
                    "MATCH (p:Person {id: 1}) RETURN p.score AS score, p.tags AS tags").get(0);
            assertEquals(2.5, ada.get("score"));
            assertEquals(List.of("a", "b"), ada.get("tags"));
        }
    }

    @Test
    @DisplayName("an explicit mode converts an existing graph and reports what it was")
    void modeConversionIsReported(@TempDir Path dir) {
        Path path = dir.resolve("converted.kgl");
        try (WriterLease lease = WriterLease.acquire(path);
                KnowledgeGraph graph = KnowledgeGraph.open(path, StorageMode.MAPPED)) {
            assertEquals(path.toAbsolutePath(), lease.path(), "the lease covers the saved path");
            graph.cypher("CREATE (:Thing {id: 1, title: 'kept'})");
            graph.save(path);
        }

        // Reopening in the mode it already is converts nothing...
        try (KnowledgeGraph graph = KnowledgeGraph.open(path, StorageMode.MAPPED)) {
            assertTrue(graph.convertedFrom().isEmpty(),
                    "reopening in the recorded mode should not convert");
        }

        // ...and reopening in another one reports the mode it came back in,
        // which is how we know the checkpoint really recorded MAPPED.
        try (KnowledgeGraph graph = KnowledgeGraph.open(path, StorageMode.MEMORY)) {
            assertEquals(java.util.Optional.of(StorageMode.MAPPED), graph.convertedFrom());
            assertEquals("kept", graph.query("MATCH (t:Thing) RETURN t.title AS title")
                    .get(0).get("title"));
        }
    }

    @Test
    @DisplayName("an unspecified-mode open of a missing path is an error, not a silent create")
    void unspecifiedOpenOfMissingPathFails(@TempDir Path dir) {
        Path missing = dir.resolve("nope.kgl");
        KgliteException error =
                assertThrows(KgliteException.class, () -> KnowledgeGraph.open(missing));
        assertEquals("FileNotFound", error.statusName());
        assertFalse(Files.exists(missing), "the failed open must not have created anything");
    }

    @Test
    @DisplayName("a bad query surfaces the engine's own message and status name")
    void errorMapping() {
        try (KnowledgeGraph graph = KnowledgeGraph.createInMemory()) {
            KgliteException error = assertThrows(KgliteException.class,
                    () -> graph.cypher("MATCH (n RETURN n"));
            assertEquals("CypherSyntax", error.statusName());
            assertEquals(1, error.statusCode());
            assertTrue(error.getMessage().length() > "CypherSyntax".length(),
                    "the engine's own detail should be carried: " + error.getMessage());

            // A wrapper-side failure is distinguishable from an engine one.
            KgliteException marshalling = assertThrows(KgliteException.class,
                    () -> graph.cypher("RETURN $x AS x", Map.of("x", new Object())));
            assertEquals(-1, marshalling.statusCode());
            assertEquals("WrapperError", marshalling.statusName());
        }
    }

    @Test
    @DisplayName("a fresh graph can be created in a non-default mode, and disk needs a path")
    void createInExplicitMode(@TempDir Path dir) {
        try (KnowledgeGraph graph = KnowledgeGraph.create(StorageMode.MAPPED, null)) {
            graph.cypher("CREATE (:Thing {id: 1, title: 'built'})");
            assertEquals("built",
                    graph.query("MATCH (t:Thing) RETURN t.title AS title").get(0).get("title"));
            graph.save(dir.resolve("mapped.kgl"));
        }
        KgliteException error = assertThrows(KgliteException.class,
                () -> KnowledgeGraph.create(StorageMode.DISK, null));
        assertEquals("InvalidArgument", error.statusName());
    }

    @Test
    @DisplayName("close is idempotent and use-after-close is a clear error")
    void closeIsIdempotent() {
        KnowledgeGraph graph = KnowledgeGraph.createInMemory();
        graph.cypher("CREATE (:X {id: 1, title: 'x'})");
        graph.close();
        graph.close();
        graph.close();
        assertThrows(IllegalStateException.class, () -> graph.query("MATCH (n) RETURN n.id AS id"));
    }
}
