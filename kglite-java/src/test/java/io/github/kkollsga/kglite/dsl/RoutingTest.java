package io.github.kkollsga.kglite.dsl;

import static io.github.kkollsga.kglite.dsl.Cypher.match;
import static io.github.kkollsga.kglite.dsl.Cypher.node;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.kkollsga.kglite.KgliteException;
import io.github.kkollsga.kglite.KnowledgeGraph;
import io.github.kkollsga.kglite.Transaction;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/**
 * Where a statement goes when you run it: the two targets, and the entry point each kind takes.
 *
 * <p>The binding's documented footgun is choosing {@code cypher} versus {@code query} — the read
 * path <em>refuses</em> a mutation, so the wrong choice is an exception at the caller's call site.
 * A DSL caller cannot make that choice at all: {@code on(graph)} is one method and the statement's
 * type decides. The write direction is directly observable and is asserted below — the same text
 * that {@code on(graph)} runs happily throws when handed to {@code query}. The read direction has
 * no observable counterpart (the write path accepts reads too), which is exactly why the footgun
 * only ever bit in one direction.
 */
class RoutingTest {

    @Test
    @DisplayName("a write statement takes the write path, which is why it works at all")
    void writesRouteToTheWritePath() {
        WriteStatement statement = match(node("Person").named("p").withProperty("id", 1))
                .set(node("Person").named("p").prop("city").to("Oslo"));

        try (KnowledgeGraph graph = Corpus.build(Corpus.Fixture.PEOPLE)) {
            // The read path refuses this text outright...
            KgliteException refused = assertThrows(KgliteException.class,
                    () -> graph.query(statement.cypher(), statement.params()));
            assertEquals("InvalidArgument", refused.statusName());

            // ...and on(graph) runs it, so it cannot have taken that path.
            statement.on(graph);
            assertEquals("Oslo", graph.query("MATCH (p:Person {id: 1}) RETURN p.city AS city")
                    .get(0).get("city"));
        }
    }

    @Test
    @DisplayName("a read statement returns its rows through on(graph)")
    void readsRunThroughTheSameMethod() {
        Statement statement = match(node("Person").named("p"))
                .returning(node("Person").named("p").prop("id").as("id"));
        try (KnowledgeGraph graph = Corpus.build(Corpus.Fixture.PEOPLE)) {
            assertEquals(4, statement.on(graph).size());
        }
    }

    @Test
    @DisplayName("on(tx) stages rather than executes, and the batch applies at commit")
    void statementsStageIntoATransaction() {
        Node person = node("Person").named("p");
        WriteStatement create = Cypher.create(node("Person")
                .withProperty("id", 90).withProperty("title", "Eve"));
        WriteStatement rename = match(node("Person").named("p").withProperty("id", 90))
                .set(person.prop("title").to("Evelyn"));
        Statement read = match(node("Person").named("p").withProperty("id", 90))
                .returning(person.prop("title").as("title"));

        try (KnowledgeGraph graph = Corpus.build(Corpus.Fixture.PEOPLE)) {
            try (Transaction tx = graph.beginTransaction()) {
                // Staging chains, and returns the transaction rather than rows.
                assertSame(tx, create.on(tx));
                rename.on(tx);
                read.on(tx);

                assertEquals(0, graph.query(
                        "MATCH (p:Person {id: 90}) RETURN p.id AS id").size(),
                        "on(tx) executed something instead of staging it");

                List<List<Map<String, Object>>> results = tx.commit();
                assertEquals(3, results.size());
                assertEquals(List.of(Map.of("title", "Evelyn")), results.get(2),
                        "the staged read must see the staged writes, in the engine");
            }
            assertEquals(1, graph.query("MATCH (p:Person {id: 90}) RETURN p.id AS id").size());
        }
    }

    @Test
    @DisplayName("a transaction of DSL statements is atomic like any other")
    void stagedStatementsAreAtomic() {
        try (KnowledgeGraph graph = Corpus.build(Corpus.Fixture.PEOPLE)) {
            long before = people(graph);
            try (Transaction tx = graph.beginTransaction()) {
                Cypher.create(node("Person").withProperty("id", 90)).on(tx);
                // A raw statement alongside the built ones, and the one that fails.
                tx.add("MATCH (p:Person) SET p.title = $missing");
                Cypher.create(node("Person").withProperty("id", 91)).on(tx);
                assertThrows(KgliteException.class, tx::commit);
            }
            assertEquals(before, people(graph), "a failed batch of DSL statements wrote something");
        }
    }

    @Test
    @DisplayName("a transaction that is never committed applies no staged statement")
    void stagedStatementsRollBack() {
        try (KnowledgeGraph graph = Corpus.build(Corpus.Fixture.PEOPLE)) {
            long before = people(graph);
            try (Transaction tx = graph.beginTransaction()) {
                Cypher.create(node("Person").withProperty("id", 90)).on(tx);
            }
            assertEquals(before, people(graph));
        }
    }

    @Test
    @DisplayName("on() refuses a null target rather than failing later")
    void nullTargetsAreRefused() {
        Statement statement = match(node("Person").named("p"))
                .returning(node("Person").named("p").prop("id").as("id"));
        assertThrows(IllegalArgumentException.class,
                () -> statement.on((KnowledgeGraph) null));
        assertThrows(IllegalArgumentException.class,
                () -> statement.on((Transaction) null));

        WriteStatement write = Cypher.create(node("Person").withProperty("id", 1));
        assertThrows(IllegalArgumentException.class, () -> write.on((KnowledgeGraph) null));
        assertThrows(IllegalArgumentException.class, () -> write.on((Transaction) null));
    }

    @Test
    @DisplayName("the DSL's text and parameters are the same on both targets")
    void bothTargetsGetTheSameStatement() {
        WriteStatement statement = Cypher.create(node("Person")
                .withProperty("id", 90).withProperty("title", "Eve"));
        assertEquals("CREATE (:Person {id: $p0, title: $p1})", statement.cypher());
        assertEquals(Map.of("p0", 90, "p1", "Eve"), statement.params());

        try (KnowledgeGraph graph = Corpus.build(Corpus.Fixture.PEOPLE)) {
            List<String> direct;
            try (KnowledgeGraph other = Corpus.build(Corpus.Fixture.PEOPLE)) {
                statement.on(other);
                direct = Corpus.state(other);
            }
            try (Transaction tx = graph.beginTransaction()) {
                statement.on(tx);
                tx.commit();
            }
            assertEquals(direct, Corpus.state(graph),
                    "the same statement left different graphs behind on the two targets");
            assertTrue(direct.stream().anyMatch(line -> line.contains("Eve")),
                    "the comparison would pass on two graphs where nothing happened");
        }
    }

    private static long people(KnowledgeGraph graph) {
        return ((Number) graph.query("MATCH (p:Person) RETURN count(p) AS n").get(0).get("n"))
                .longValue();
    }
}
