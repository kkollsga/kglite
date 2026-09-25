package io.github.kkollsga.kglite;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/** The open / mutate / save / reopen cycle across the C ABI. */
class KnowledgeGraphTest {

    @Test
    void positiveTimeoutsRoundUpWithoutOverflow() {
        assertEquals(0L, KnowledgeGraph.timeoutMillis(null));
        assertEquals(0L, KnowledgeGraph.timeoutMillis(Duration.ZERO));
        assertEquals(0L, KnowledgeGraph.timeoutMillis(Duration.ofNanos(-1)));
        assertEquals(1L, KnowledgeGraph.timeoutMillis(Duration.ofNanos(1)));
        assertEquals(1L, KnowledgeGraph.timeoutMillis(Duration.ofNanos(999_999)));
        assertEquals(1L, KnowledgeGraph.timeoutMillis(Duration.ofMillis(1)));
        assertEquals(2L, KnowledgeGraph.timeoutMillis(Duration.ofNanos(1_000_001)));
        assertEquals(
                Long.MAX_VALUE,
                KnowledgeGraph.timeoutMillis(Duration.ofSeconds(Long.MAX_VALUE, 999_999_999)));
    }

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
     * this table is the first thing a consumer of the binding relies on, so
     * the structured shapes (node, relationship, date, duration) are pinned
     * rather than merely described.
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

            // Whole nodes and relationships arrive as structured maps (the
            // javadoc table pins these shapes).
            Object node = graph.query("MATCH (p:Person {id: 1}) RETURN p AS p").get(0).get("p");
            assertTrue(node instanceof Map,
                    "RETURN of a whole node is a structured Map, got " + node.getClass());
            Map<?, ?> nodeMap = (Map<?, ?>) node;
            assertEquals(List.of("Person"), nodeMap.get("labels"), "node.labels");
            assertEquals("Ada", ((Map<?, ?>) nodeMap.get("properties")).get("title"),
                    "node.properties carries the stored values");
            Object rel = graph.query("MATCH ()-[r:KNOWS]->() RETURN r AS r").get(0).get("r");
            assertTrue(rel instanceof Map, "RETURN of a whole relationship is a Map, got " + rel);
            Map<?, ?> relMap = (Map<?, ?>) rel;
            assertEquals("KNOWS", relMap.get("type"), "relationship.type");
            assertEquals(2020L, ((Map<?, ?>) relMap.get("properties")).get("since"),
                    "relationship.properties");

            // ...and the routes the docs point at instead.
            Map<String, Object> parts = graph.query(
                    "MATCH (p:Person {id: 1}) RETURN properties(p) AS props, labels(p) AS labels,"
                            + " id(p) AS id").get(0);
            assertTrue(parts.get("props") instanceof Map, "properties() is a Map");
            assertEquals("Ada", ((Map<?, ?>) parts.get("props")).get("title"));
            assertEquals(List.of("Person"), parts.get("labels"), "labels() is a List of String");
            assertTrue(parts.get("id") instanceof Long, "id() is a Long");

            Object path = graph.query("MATCH p = (:Person)-[:KNOWS]->(:Person) RETURN p AS p")
                    .get(0).get("p");
            assertTrue(path instanceof Map<?, ?> pathMap
                    && pathMap.get("nodes") instanceof List<?> nodes && nodes.size() == 2
                    && pathMap.get("relationships") instanceof List<?> rels && rels.size() == 1,
                    "a path is a Map of node and relationship lists, got " + path);
            assertEquals(List.of(nodeMap),
                    graph.query("MATCH (p:Person {id: 1}) RETURN collect(p) AS ps").get(0).get("ps"),
                    "collect(n) is a List of node maps");

            // Temporal and spatial cells.
            Map<String, Object> scalars = graph.query(
                    "RETURN date('2020-01-01') AS d, datetime('2020-01-01T10:00:00+02:00') AS t,"
                            + " duration({days: 2}) AS dur,"
                            + " point(1.5, 2.5) AS pt").get(0);
            assertEquals("2020-01-01", scalars.get("d"));
            assertEquals("2020-01-01T08:00:00", scalars.get("t"), "normalised to UTC, no suffix");
            assertEquals(Map.of("months", 0L, "days", 2L, "seconds", 0L), scalars.get("dur"));
            assertEquals(Map.of("latitude", 1.5, "longitude", 2.5), scalars.get("pt"));
        }
    }

    @Test
    @DisplayName("java.time parameters bind as dates and datetimes, matching stored values")
    void dateAndDatetimeParameters() {
        try (KnowledgeGraph graph = KnowledgeGraph.createInMemory()) {
            graph.cypher("CREATE (:A {id: 1})-[:R {vf: date('2020-01-01'),"
                    + " at: datetime('2020-01-01T10:00:00+02:00')}]->(:B {id: 2})");
            String count = "MATCH ()-[r:R]->() WHERE r.vf = $v AND r.at = $t RETURN count(*) AS c";
            List<Object> datetimes = List.of(
                    java.time.LocalDateTime.of(2020, 1, 1, 8, 0),
                    java.time.OffsetDateTime.parse("2020-01-01T10:00+02:00"),
                    java.time.ZonedDateTime.parse("2020-01-01T09:00+01:00[Europe/Oslo]"),
                    java.time.Instant.parse("2020-01-01T08:00:00Z"));
            for (Object t : datetimes) {
                Map<String, Object> params = Map.of("v", java.time.LocalDate.of(2020, 1, 1), "t", t);
                assertEquals(1L, graph.query(count, params).get(0).get("c"), t.toString());
            }

            // The string a date cell comes back as, rebound as a LocalDate, matches.
            Object vf = graph.query("MATCH ()-[r:R]->() RETURN r.vf AS vf").get(0).get("vf");
            assertEquals("2020-01-01", vf);
            assertEquals(1L, graph.query("MATCH ()-[r:R]->() WHERE r.vf = $v RETURN count(*) AS c",
                    Map.of("v", java.time.LocalDate.parse((String) vf))).get(0).get("c"));
        }
    }

    @Test
    @DisplayName("change capture publishes cypher() and transaction writes")
    void changeCapturePublishesWrites() {
        try (KnowledgeGraph graph = KnowledgeGraph.createInMemory()) {
            graph.cypher("CALL db.cdc.enable()");
            graph.cypher("CREATE (:Person {id: 1})");
            graph.cypher("MATCH (p:Person) SET p.age = $age", Map.of("age", 30));
            Transaction tx = graph.beginTransaction();
            tx.add("CREATE (:Person {id: 2})");
            tx.commit();

            List<Map<String, Object>> changes = graph.query(
                    "CALL db.cdc.query({}) YIELD operation, nodeId RETURN operation, nodeId");
            assertEquals(List.of(
                    Map.of("operation", "create", "nodeId", 1L),
                    Map.of("operation", "update", "nodeId", 1L),
                    Map.of("operation", "create", "nodeId", 2L)), changes);
        }
    }

    @Test
    @DisplayName("queryResult and cypherResult carry the engine's warnings and diagnostics")
    void resultsCarryWarningsAndDiagnostics() {
        try (KnowledgeGraph graph = KnowledgeGraph.createInMemory()) {
            QueryResult created = graph.cypherResult("CREATE (:City {id: 1}) RETURN 1 AS one", Map.of());
            assertEquals(List.of(Map.of("one", 1L)), created.rows());
            assertEquals(List.of(), created.warnings(), "a clean statement warns about nothing");
            assertTrue(created.diagnostics().containsKey("elapsed_ms"), created.diagnostics().toString());

            QueryResult typo = graph.queryResult("MATCH (c:Cty) RETURN c.id AS id", Map.of());
            assertEquals(List.of(), typo.rows());
            assertEquals(1, typo.warnings().size(), typo.warnings().toString());
            assertTrue(typo.warnings().get(0).contains("'City'"), typo.warnings().get(0));
            assertEquals(typo.warnings(), typo.diagnostics().get("warnings"));
            assertThrows(UnsupportedOperationException.class, () -> typo.warnings().add("x"));

            assertThrows(KgliteException.class,
                    () -> graph.queryResult("CREATE (:City {id: 2})", Map.of()),
                    "the read path still refuses a mutation");
        }
    }

    @Test
    @DisplayName("a PROFILE statement's clause statistics arrive through QueryResult.profile()")
    void profileStatisticsArriveInTheResult() {
        try (KnowledgeGraph graph = KnowledgeGraph.createInMemory()) {
            graph.cypher("CREATE (:City {id: 1}), (:City {id: 2})");
            QueryResult profiled = graph.queryResult("PROFILE MATCH (c:City) RETURN c.id AS id", Map.of());
            assertEquals(2, profiled.rows().size());
            List<Map<String, Object>> clauses = profiled.profile();
            assertEquals(List.of("Match :City", "Return"),
                    clauses.stream().map(c -> c.get("clause")).toList(), clauses.toString());
            assertEquals(2L, clauses.get(0).get("rows_out"));
            assertEquals(2L, clauses.get(1).get("rows_in"));
            assertTrue(clauses.get(0).get("elapsed_us") instanceof Long, clauses.toString());
            assertEquals(clauses, profiled.diagnostics().get("profile"));

            assertEquals(List.of(), graph.queryResult("MATCH (c:City) RETURN c.id", Map.of()).profile(),
                    "an unprofiled statement has no clause statistics");

            try (Transaction tx = graph.beginTransaction()) {
                tx.add("PROFILE CREATE (:City {id: 3})");
                List<QueryResult> results = tx.commitResults();
                assertEquals(List.of("Create"),
                        results.get(0).profile().stream().map(c -> c.get("clause")).toList());
            }
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
