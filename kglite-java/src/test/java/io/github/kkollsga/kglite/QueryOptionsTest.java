package io.github.kkollsga.kglite;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.time.Duration;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/**
 * The per-query timeout and work-budget overloads, over
 * {@code kglite_session_execute_read_opts} / {@code _mut_opts}.
 *
 * <p>Covers the two things the header defines that a caller can get wrong: that
 * {@code 0} (and a {@code null}/zero {@code Duration}) is "unlimited", not
 * "expire immediately", and that a {@code maxWorkUnits} overflow is a thrown
 * error rather than a silent truncation.
 */
class QueryOptionsTest {

    private static KnowledgeGraph fivePeople() {
        KnowledgeGraph graph = KnowledgeGraph.createInMemory();
        graph.cypher("UNWIND range(1, 5) AS i CREATE (:Person {id: i, title: 'p' + i})");
        return graph;
    }

    @Test
    @DisplayName("a generous timeout returns the rows")
    void generousTimeoutReturns() {
        try (KnowledgeGraph graph = fivePeople()) {
            List<Map<String, Object>> rows = graph.query(
                    "MATCH (p:Person) RETURN p.id AS id ORDER BY id", Duration.ofSeconds(30));
            assertEquals(5, rows.size());
            assertEquals(1L, rows.get(0).get("id"));
        }
    }

    @Test
    @DisplayName("a zero, negative, or null Duration means no deadline, not immediate expiry")
    void zeroTimeoutIsUnlimited() {
        try (KnowledgeGraph graph = fivePeople()) {
            // If 0 meant "expire now" every one of these would throw CypherTimeout.
            assertEquals(5, graph.query("MATCH (p:Person) RETURN p.id", Duration.ZERO).size());
            assertEquals(5, graph.query(
                    "MATCH (p:Person) RETURN p.id", Duration.ofSeconds(-1)).size());
            assertEquals(5, graph.query(
                    "MATCH (p:Person) RETURN p.id", (Duration) null).size());
        }
    }

    @Test
    @DisplayName("maxWorkUnits budgets a read: an overflow is an error, not a truncation")
    void maxWorkUnitsBudgetsRead() {
        try (KnowledgeGraph graph = fivePeople()) {
            // 5 rows through a budget of 2 is rejected outright.
            KgliteException e = assertThrows(KgliteException.class, () -> graph.query(
                    "MATCH (p:Person) RETURN p.id", Map.of(), null, 2));
            assertTrue(e.getMessage().length() > 0, "the overflow carries a message");

            // 0 is unlimited; a budget above the work the query does returns everything.
            assertEquals(5, graph.query("MATCH (p:Person) RETURN p.id", Map.of(), null, 0).size());
            assertEquals(5, graph.query("MATCH (p:Person) RETURN p.id", Map.of(), null, 10).size());
        }
    }

    @Test
    @DisplayName("maxWorkUnits on the write path rejects and rolls back the whole statement")
    void maxWorkUnitsBudgetsWriteAndRollsBack() {
        try (KnowledgeGraph graph = KnowledgeGraph.createInMemory()) {
            assertThrows(KgliteException.class, () -> graph.cypher(
                    "UNWIND range(1, 5) AS i CREATE (:X {id: i}) RETURN i", Map.of(), null, 2));
            // The budget failure rolled the statement back: no X nodes reached the graph.
            assertEquals(0L, graph.query("MATCH (x:X) RETURN count(x) AS n").get(0).get("n"));
        }
    }

    @Test
    @DisplayName("the write-path timeout overload runs a parameterised mutation")
    void writeTimeoutOverloadRuns() {
        try (KnowledgeGraph graph = KnowledgeGraph.createInMemory()) {
            graph.cypher("CREATE (:Person {id: $id, title: $t})",
                    Map.of("id", 1, "t", "Ada"), Duration.ofSeconds(30));
            assertEquals("Ada",
                    graph.query("MATCH (p:Person {id: 1}) RETURN p.title AS t").get(0).get("t"));
        }
    }
}
