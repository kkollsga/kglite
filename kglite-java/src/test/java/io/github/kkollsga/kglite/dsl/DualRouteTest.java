package io.github.kkollsga.kglite.dsl;

import static io.github.kkollsga.kglite.dsl.Cypher.match;
import static io.github.kkollsga.kglite.dsl.Cypher.node;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.kkollsga.kglite.KnowledgeGraph;
import java.util.List;
import java.util.Map;
import java.util.stream.Stream;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

/**
 * D2 gate (b): every expressible corpus entry runs both ways against the real engine — through the
 * DSL and through the hand-written string — and the rows must be equal.
 *
 * <p>This is the gate that catches an emitter producing <em>different but valid</em> Cypher, which
 * exact-string equality cannot see: a renderer that dropped a {@code WHERE} would fail gate (a)
 * only if gate (a)'s expectation were right, whereas here the engine itself is the oracle.
 *
 * <p>Both routes run against separately-built fixture graphs, so neither can see the other's
 * effects, and both run through the same jar and the same native library the consumer would use.
 */
class DualRouteTest {

    static Stream<Corpus.Entry> expressibleEntries() {
        return Corpus.entries().stream().filter(entry -> entry.dsl() != null);
    }

    @ParameterizedTest(name = "{0}")
    @MethodSource("expressibleEntries")
    @DisplayName("the DSL route and the raw route return the same rows")
    void routesAgree(Corpus.Entry entry) {
        List<Map<String, Object>> viaDsl;
        List<Map<String, Object>> viaRaw;
        try (KnowledgeGraph graph = Corpus.build(entry.fixture())) {
            viaDsl = entry.dsl().on(graph);
        }
        try (KnowledgeGraph graph = Corpus.build(entry.fixture())) {
            viaRaw = graph.query(entry.cypher(), entry.params());
        }
        Rows.assertRows(viaRaw, viaDsl, entry.orderSensitive(), entry.name() + ": route equality");

        // And both must still be the rows D0 froze, so an emitter and a corpus that drifted
        // together cannot agree their way to green.
        Rows.assertRows(entry.expectedRows(), viaDsl, entry.orderSensitive(),
                entry.name() + ": DSL route against the frozen expectation");
    }

    @Test
    @DisplayName("the gate covers the whole read half")
    void coverageIsComplete() {
        long covered = expressibleEntries().count();
        long readHalf = Corpus.entries().stream()
                .filter(entry -> entry.expressibility() == Corpus.Expressibility.READ_HALF)
                .count();
        assertEquals(readHalf, covered,
                "every read-half entry must run through both routes");
        assertTrue(covered >= 30, "the dual-route gate must cover at least 30 statements");
    }

    /**
     * The R1 positive control. Two statements that differ in a way exact-string equality would
     * catch anyway are no test of this gate; the interesting failure is an emitter that produces
     * <em>valid, different</em> Cypher. So: run a deliberately-divergent statement through the same
     * comparison and show it caught.
     */
    @Test
    @DisplayName("positive control: a divergent-but-valid emission is caught by the row comparison")
    void divergentEmissionIsCaught() {
        Node person = node("Person").named("p");
        String raw = "MATCH (p:Person) WHERE p.age > $p0 RETURN p.id AS id ORDER BY p.id ASC";
        Map<String, Object> params = Map.of("p0", 30);

        // Divergence one: the predicate silently dropped. Valid Cypher, wrong rows.
        Statement withoutFilter = match(person)
                .returning(person.prop("id").as("id"))
                .orderBy(person.prop("id").asc());
        // Divergence two: the ordering silently reversed. Valid Cypher, right rows, wrong order.
        Statement reversed = match(person)
                .where(person.prop("age").gt(30))
                .returning(person.prop("id").as("id"))
                .orderBy(person.prop("id").desc());
        // The correct statement, to prove the harness clears a good implementation too.
        Statement correct = match(person)
                .where(person.prop("age").gt(30))
                .returning(person.prop("id").as("id"))
                .orderBy(person.prop("id").asc());

        try (KnowledgeGraph graph = Corpus.build(Corpus.Fixture.PEOPLE)) {
            List<Map<String, Object>> expected = graph.query(raw, params);

            assertEquals(expected, correct.on(graph),
                    "the correct statement must clear the gate");
            assertNotEquals(expected, withoutFilter.on(graph),
                    "the gate cannot go red: a dropped WHERE returned the same rows");
            assertNotEquals(expected, reversed.on(graph),
                    "the gate cannot go red: a reversed ORDER BY returned the same rows");

            // ...and the order-insensitive path must still catch the dropped predicate, because
            // an unordered entry's comparison is the weaker of the two.
            assertThrowsAssertion(() -> Rows.assertRows(expected, withoutFilter.on(graph), false,
                    "positive control"));
        }
    }

    private static void assertThrowsAssertion(Runnable body) {
        try {
            body.run();
        } catch (AssertionError expected) {
            return;
        }
        throw new AssertionError(
                "the multiset row comparison did not catch a statement returning extra rows");
    }
}
