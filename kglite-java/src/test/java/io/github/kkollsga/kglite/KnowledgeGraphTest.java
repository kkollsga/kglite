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
            assertEquals(StorageMode.MAPPED, graph.storageMode(), "created in the mode asked for");
            graph.cypher("CREATE (:Thing {id: 1, title: 'kept'})");
            graph.save(path);
        }

        // The checkpoint recorded MAPPED — asserted directly off the reopened
        // graph rather than inferred from a conversion report, which is silent
        // in exactly the case that matters here (nothing was converted).
        try (KnowledgeGraph graph = KnowledgeGraph.open(path)) {
            assertEquals(StorageMode.MAPPED, graph.storageMode(),
                    "an unspecified open must land on the recorded mode");
            assertTrue(graph.convertedFrom().isEmpty(), "an unspecified open converts nothing");
        }

        // Reopening in the mode it already is converts nothing...
        try (KnowledgeGraph graph = KnowledgeGraph.open(path, StorageMode.MAPPED)) {
            assertEquals(StorageMode.MAPPED, graph.storageMode());
            assertTrue(graph.convertedFrom().isEmpty(),
                    "reopening in the recorded mode should not convert");
        }

        // ...and reopening in another one really lands on the new backend, not
        // merely reports that it meant to.
        try (KnowledgeGraph graph = KnowledgeGraph.open(path, StorageMode.MEMORY)) {
            assertEquals(StorageMode.MEMORY, graph.storageMode(),
                    "the conversion must have actually happened");
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

    /**
     * The value-mapping table in {@code KnowledgeGraph}'s class documentation,
     * asserted cell by cell — including the row that surprises people.
     *
     * <p>Documented behaviour with no test is documentation that drifts, and
     * this table is the first thing a consumer of the binding relies on. The
     * whole-node case is pinned rather than merely described: it is the
     * consequence of the C ABI having no JSON shape for a node, so if the
     * engine ever grows one, this test is the thing that says the docs and the
     * README now lie.
     */
    @Test
    @DisplayName("the documented value mapping holds in both directions")
    void valueMapping() {
        try (KnowledgeGraph graph = KnowledgeGraph.createInMemory()) {
            // Java parameter -> engine -> Java cell, for every legal input type.
            Map<String, Object> back = graph.query(
                    "RETURN $i AS i, $l AS l, $d AS d, $b AS b, $s AS s,"
                            + " $n AS n, $list AS list, $map AS map",
                    mapOfNullable(
                            "i", 7,              // Integer widens...
                            "l", 8L,
                            "d", 2.5,
                            "b", true,
                            "s", "hi",
                            "n", null,
                            "list", List.of(1, "two"),
                            "map", Map.of("k", 3)))
                    .get(0);
            assertEquals(7L, back.get("i"), "an Integer parameter comes back as a Long");
            assertEquals(8L, back.get("l"));
            assertEquals(2.5, back.get("d"));
            assertEquals(true, back.get("b"));
            assertEquals("hi", back.get("s"));
            assertEquals(List.of(1L, "two"), back.get("list"), "list elements map recursively");
            assertEquals(Map.of("k", 3L), back.get("map"));

            // A null cell is a present key with a null value, not an absent key.
            assertTrue(back.containsKey("n"), "a null cell keeps its key");
            assertEquals(null, back.get("n"));

            // Rows are unmodifiable and an empty result is an empty list.
            assertThrows(UnsupportedOperationException.class, () -> back.put("x", 1));
            assertEquals(List.of(), graph.query("MATCH (n:NoSuchLabel) RETURN n.id AS id"));

            graph.cypher("CREATE (a:Person {id: 1, title: 'Ada'})-[:KNOWS {since: 2020}]->"
                    + "(b:Person {id: 2, title: 'Grace'})");

            // The documented trap: a whole node/relationship/path has no JSON
            // shape, so it arrives as the engine's Debug rendering in a String.
            Object node = graph.query("MATCH (p:Person {id: 1}) RETURN p AS p").get(0).get("p");
            assertTrue(node instanceof String,
                    "RETURN of a whole node is documented as a String, got " + node.getClass());
            assertTrue(((String) node).startsWith("Node("),
                    "the debug rendering the docs quote, got " + node);
            Object rel = graph.query("MATCH ()-[r:KNOWS]->() RETURN r AS r").get(0).get("r");
            assertTrue(rel instanceof String && ((String) rel).startsWith("Relationship("),
                    "RETURN of a whole relationship is documented as a String, got " + rel);

            // ...and the routes the docs point at instead.
            Map<String, Object> parts = graph.query(
                    "MATCH (p:Person {id: 1}) RETURN properties(p) AS props, labels(p) AS labels,"
                            + " id(p) AS id").get(0);
            assertTrue(parts.get("props") instanceof Map, "properties() is a Map");
            assertEquals("Ada", ((Map<?, ?>) parts.get("props")).get("title"));
            assertEquals(List.of("Person"), parts.get("labels"), "labels() is a List of String");
            assertTrue(parts.get("id") instanceof Long, "id() is a Long");
        }
    }

    /** {@code Map.of} rejects a null value; the null cell case needs one. */
    private static Map<String, Object> mapOfNullable(Object... pairs) {
        Map<String, Object> map = new java.util.LinkedHashMap<>();
        for (int i = 0; i < pairs.length; i += 2) {
            map.put((String) pairs[i], pairs[i + 1]);
        }
        return map;
    }

    @Test
    @DisplayName("query refuses a mutation, names it, and leaves the graph usable")
    void queryRefusesMutations() {
        try (KnowledgeGraph graph = KnowledgeGraph.createInMemory()) {
            KgliteException error = assertThrows(KgliteException.class,
                    () -> graph.query("CREATE (:Person {id: 1, title: 'Ada'})"));
            assertEquals("InvalidArgument", error.statusName(),
                    "the status name the cypher-vs-query docs quote");
            assertTrue(error.getMessage().contains("mutation query"),
                    "the message the docs quote: " + error.getMessage());

            // Nothing was written, and the same statement works on the write path.
            assertEquals(List.of(),
                    graph.query("MATCH (p:Person) RETURN p.id AS id"));
            graph.cypher("CREATE (:Person {id: 1, title: 'Ada'})");
            assertEquals(1L,
                    graph.query("MATCH (p:Person) RETURN p.id AS id").get(0).get("id"));
        }
    }

    @Test
    @DisplayName("save is the only thing that persists, and the lease does not gate it")
    void saveIsRequiredAndLeaseIsAdvisory(@TempDir Path dir) {
        Path path = dir.resolve("durability.kgl");

        // Mutating and closing without saving loses the work, silently.
        try (KnowledgeGraph graph = KnowledgeGraph.open(path, StorageMode.MEMORY)) {
            graph.cypher("CREATE (:Person {id: 1, title: 'Ada'})");
        }
        assertFalse(Files.exists(path), "close() must not have written anything");

        // No lease is held here at all: the documented contract is cooperative,
        // and save neither takes nor checks one. A test that asserted the
        // opposite would be asserting a guarantee the engine does not make.
        try (KnowledgeGraph graph = KnowledgeGraph.open(path, StorageMode.MEMORY)) {
            graph.cypher("CREATE (:Person {id: 2, title: 'Grace'})");
            graph.save(path);
        }
        try (KnowledgeGraph graph = KnowledgeGraph.open(path)) {
            assertEquals(List.of(Map.of("id", 2L)),
                    graph.query("MATCH (p:Person) RETURN p.id AS id"),
                    "only the saved mutation survived");
        }
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
            assertEquals(StorageMode.MAPPED, graph.storageMode());
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
        assertEquals(StorageMode.MEMORY, graph.storageMode());
        graph.cypher("CREATE (:X {id: 1, title: 'x'})");
        graph.close();
        graph.close();
        graph.close();
        assertThrows(IllegalStateException.class, () -> graph.query("MATCH (n) RETURN n.id AS id"));
    }
}
