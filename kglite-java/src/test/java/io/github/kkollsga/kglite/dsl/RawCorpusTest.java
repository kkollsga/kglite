package io.github.kkollsga.kglite.dsl;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
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
 * D0: the corpus runs green through the raw route, before any DSL is involved.
 *
 * <p>This is the test that makes the other two gates mean something. Emitted-Cypher equality only
 * proves the renderer reproduces a string; dual-route equality only proves two routes agree. Both
 * are satisfiable by a corpus of queries that return the wrong rows. Running every entry raw
 * against real fixture data with hand-written expectations is what pins the corpus to reality.
 */
class RawCorpusTest {

    static Stream<Corpus.Entry> entries() {
        return Corpus.entries().stream();
    }

    @ParameterizedTest(name = "{0}")
    @MethodSource("entries")
    @DisplayName("every corpus entry returns its expected rows through the raw route")
    void rawRouteReturnsExpectedRows(Corpus.Entry entry) {
        try (KnowledgeGraph graph = Corpus.build(entry.fixture())) {
            List<Map<String, Object>> rows = run(graph, entry);
            if (entry.expectedRows() != null) {
                Rows.assertRows(entry.expectedRows(), rows, entry.orderSensitive(), entry.name());
            } else {
                assertNotNull(entry.expectedRowCount(),
                        entry.name() + ": an entry needs either expected rows or a row count");
                assertEquals(entry.expectedRowCount().intValue(), rows.size(),
                        entry.name() + ": row count (rows were " + rows + ")");
            }
        }
    }

    private static List<Map<String, Object>> run(KnowledgeGraph graph, Corpus.Entry entry) {
        if (entry.route() == Corpus.Route.WRITE) {
            graph.cypher(entry.cypher(), entry.params());
            return graph.query(entry.verify());
        }
        return graph.query(entry.cypher(), entry.params());
    }

    @Test
    @DisplayName("R13 stop rule: the v1 clause set expresses enough of the corpus to be worth "
            + "building")
    void expressibilityCensus() {
        List<Corpus.Entry> entries = Corpus.entries();
        long readHalf = count(entries, Corpus.Expressibility.READ_HALF);
        long writeHalf = count(entries, Corpus.Expressibility.WRITE_HALF);
        long later = count(entries, Corpus.Expressibility.V1_LATER);
        long outOfV1 = count(entries, Corpus.Expressibility.OUT_OF_V1);
        long expressible = readHalf + writeHalf + later;

        // The rule as written before the corpus was collected: fewer than 20 in 30 expressible
        // means the clause set is wrong and must be re-scoped before an emitter is written.
        long floor = Math.max(20, Math.round(entries.size() * 20.0 / 30.0));
        assertTrue(entries.size() >= 30,
                "the corpus must carry at least 30 entries, has " + entries.size());
        assertTrue(expressible >= floor,
                "R13 stop rule: only " + expressible + " of " + entries.size()
                        + " corpus entries are expressible in the v1 clause set (floor " + floor
                        + "). Re-scope the clause set before building the emitter.");
        assertTrue(outOfV1 > 0,
                "a census with nothing outside v1 is not a census — the corpus must include the "
                        + "shapes the clause set deliberately refuses");
        assertTrue(readHalf >= 30,
                "the read half must carry the bulk of the corpus, has " + readHalf);
        assertTrue(writeHalf >= 10,
                "the write half must carry a representative statement per clause, has "
                        + writeHalf);

        System.out.println("corpus census: " + entries.size() + " entries — "
                + readHalf + " read-half, " + writeHalf + " write-half, "
                + later + " later v1 clauses, " + outOfV1 + " deliberately outside v1");
    }

    private static long count(List<Corpus.Entry> entries, Corpus.Expressibility bucket) {
        return entries.stream().filter(entry -> entry.expressibility() == bucket).count();
    }

    @Test
    @DisplayName("corpus entry names are unique")
    void entryNamesAreUnique() {
        List<String> names = Corpus.entries().stream().map(Corpus.Entry::name).toList();
        assertEquals(names.size(), names.stream().distinct().count(),
                "duplicate corpus entry names make failures ambiguous: " + names);
    }
}
