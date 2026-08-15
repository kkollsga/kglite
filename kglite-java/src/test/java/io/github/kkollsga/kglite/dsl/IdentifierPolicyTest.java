package io.github.kkollsga.kglite.dsl;

import static io.github.kkollsga.kglite.dsl.Cypher.match;
import static io.github.kkollsga.kglite.dsl.Cypher.node;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.kkollsga.kglite.KnowledgeGraph;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/**
 * The escaping policy, executed against the engine rather than copied from prose.
 *
 * <p>An emitter that carries a hard-coded reserved-word list is coupled to a dialect that can
 * change under it, and the failure mode is silent: a word that stops being reserved gets quoted
 * forever (harmless but wrong), and a word that starts being reserved gets emitted bare and
 * produces a syntax error at the user's call site, not ours. So every entry in
 * {@link Ident}'s sets is asserted here to <em>actually</em> break a bare use and <em>actually</em>
 * work quoted, and a sample outside them is asserted to work bare.
 *
 * <p>When the dialect moves, this test goes red naming the word, and the fix is a one-line set edit
 * with the failure as its evidence.
 */
class IdentifierPolicyTest {

    private static KnowledgeGraph graph;

    /** Words probed to be legal bare in every position; the negative half of the contract. */
    private static final List<String> NOT_RESERVED_ANYWHERE =
            List.of("DISTINCT", "COUNT", "Person", "value", "_private", "x1");

    /**
     * Words legal bare as a label or property key but not as a variable.
     *
     * <p>{@code TRUE}, {@code FALSE} and {@code NULL} joined this list at engine 0.16.0. They are
     * schema names in every name position — CYPHER.md, "Reserved keywords as names", citing
     * openCypher's {@code SchemaName = SymbolicName | ReservedWord} — while
     * {@code Variable = SymbolicName} keeps excluding them, which is exactly this asymmetry.
     */
    private static final List<String> RESERVED_ONLY_FOR_VARIABLES =
            List.of("ORDER", "IN", "IS", "NOT", "CREATE", "SET", "AS", "ON", "TRUE", "FALSE",
                    "NULL");

    @BeforeAll
    static void openGraph() {
        graph = KnowledgeGraph.createInMemory();
        graph.cypher("CREATE (:Person {id: 1, title: 'Ada'})");
    }

    @AfterAll
    static void closeGraph() {
        graph.close();
    }

    @Test
    @DisplayName("the reserved sets are non-empty and the variable set strictly contains the "
            + "pattern set")
    void setsAreShaped() {
        assertFalse(Ident.RESERVED_IN_PATTERNS.isEmpty(), "an empty set would gate nothing");
        assertTrue(Ident.RESERVED_IN_VARIABLES.containsAll(Ident.RESERVED_IN_PATTERNS),
                "a word that breaks a bare label must also break a bare variable");
        assertTrue(Ident.RESERVED_IN_VARIABLES.size() > Ident.RESERVED_IN_PATTERNS.size(),
                "the variable position was probed to reject strictly more words");
        assertTrue(Ident.RESERVED_IN_ALIASES.isEmpty(),
                "the alias position was probed to accept every keyword bare");
    }

    @Test
    @DisplayName("every word reserved for labels breaks a bare label and works quoted")
    void labelReservationsHold() {
        List<String> wrong = new ArrayList<>();
        for (String word : Ident.RESERVED_IN_PATTERNS) {
            if (parses("MATCH (n:" + word + ") RETURN count(n) AS c")) {
                wrong.add(word + ": bare use parsed, so it is not reserved after all");
            }
            if (!parses("MATCH (n:`" + word + "`) RETURN count(n) AS c")) {
                wrong.add(word + ": quoted use failed, so quoting is not the escape");
            }
            assertEquals("`" + word + "`", Ident.label(word).toString(),
                    word + ": must be emitted quoted");
        }
        assertEquals(List.of(), wrong);
    }

    @Test
    @DisplayName("every word reserved for variables breaks a bare variable and works quoted")
    void variableReservationsHold() {
        List<String> wrong = new ArrayList<>();
        for (String word : Ident.RESERVED_IN_VARIABLES) {
            if (parses("MATCH (" + word + ":Person) RETURN count(" + word + ") AS c")) {
                wrong.add(word + ": bare use parsed, so it is not reserved after all");
            }
            if (!parses("MATCH (`" + word + "`:Person) RETURN count(`" + word + "`) AS c")) {
                wrong.add(word + ": quoted use failed, so quoting is not the escape");
            }
            assertEquals("`" + word + "`", Ident.variable(word).toString(),
                    word + ": must be emitted quoted");
        }
        assertEquals(List.of(), wrong);
    }

    @Test
    @DisplayName("every word reserved for property keys breaks a bare key and works quoted")
    void propertyKeyReservationsHold() {
        List<String> wrong = new ArrayList<>();
        for (String word : Ident.RESERVED_IN_PATTERNS) {
            if (parses("MATCH (n:Person) RETURN n." + word + " AS c")) {
                wrong.add(word + ": bare use parsed, so it is not reserved after all");
            }
            if (!parses("MATCH (n:Person) RETURN n.`" + word + "` AS c")) {
                wrong.add(word + ": quoted use failed, so quoting is not the escape");
            }
            assertEquals("`" + word + "`", Ident.propertyKey(word).toString(),
                    word + ": must be emitted quoted");
        }
        assertEquals(List.of(), wrong);
    }

    @Test
    @DisplayName("the alias position really does accept keywords bare, which is why its set is "
            + "empty")
    void aliasesNeedNoReservations() {
        for (String word : List.of("MATCH", "RETURN", "NULL", "TRUE", "LIMIT", "ORDER", "END")) {
            assertTrue(parses("MATCH (n:Person) RETURN n.id AS " + word),
                    word + ": no longer legal bare as an alias — RESERVED_IN_ALIASES must grow");
            assertEquals(word, Ident.alias(word).toString(),
                    word + ": must be emitted bare");
        }
    }

    @Test
    @DisplayName("words outside the sets are emitted bare and the engine accepts them bare")
    void unreservedWordsStayBare() {
        for (String word : NOT_RESERVED_ANYWHERE) {
            assertTrue(parses("MATCH (n:" + word + ") RETURN count(n) AS c"),
                    word + ": bare label rejected, so it belongs in RESERVED_IN_PATTERNS");
            assertTrue(parses("MATCH (" + word + ":Person) RETURN count(" + word + ") AS c"),
                    word + ": bare variable rejected, so it belongs in RESERVED_IN_VARIABLES");
            assertEquals(word, Ident.label(word).toString(), word + ": must be emitted bare");
            assertEquals(word, Ident.variable(word).toString(), word + ": must be emitted bare");
        }
    }

    @Test
    @DisplayName("the variable position's extra reservations are real: bare label yes, bare "
            + "variable no")
    void variableOnlyReservationsAreAsymmetric() {
        for (String word : RESERVED_ONLY_FOR_VARIABLES) {
            assertTrue(Ident.RESERVED_IN_VARIABLES.contains(word), word + ": missing from the set");
            assertFalse(Ident.RESERVED_IN_PATTERNS.contains(word),
                    word + ": this test is about words reserved only for variables");
            assertTrue(parses("MATCH (n:" + word + ") RETURN count(n) AS c"),
                    word + ": bare label rejected — the asymmetry has gone, merge the sets");
            assertFalse(parses("MATCH (" + word + ":Person) RETURN count(" + word + ") AS c"),
                    word + ": bare variable accepted — remove it from RESERVED_IN_VARIABLES");
            assertEquals(word, Ident.label(word).toString(), word + ": bare as a label");
            assertEquals("`" + word + "`", Ident.variable(word).toString(),
                    word + ": quoted as a variable");
        }
    }

    @Test
    @DisplayName("the value literals are schema names in the key and relationship-type positions "
            + "too")
    void valueLiteralsAreSchemaNamesNotJustLabels() {
        // CYPHER.md, "Reserved keywords as names": a schema name is
        // `SymbolicName | ReservedWord`, so TRUE/FALSE/NULL are labels, relationship types *and*
        // property keys written bare, in either case. The label and variable halves are covered
        // by variableOnlyReservationsAreAsymmetric and variableReservationsHold above; these are
        // the two positions neither of them reaches. The value position stays the literal, which
        // the corpus pins independently (`with_filter_then_set` filters on `p.seen = true`).
        for (String word : List.of("TRUE", "FALSE", "NULL", "true", "null")) {
            assertTrue(parses("MATCH (n:Person) RETURN n." + word + " AS c"),
                    word + ": bare property key rejected — it belongs in RESERVED_IN_PATTERNS");
            assertEquals(word, Ident.propertyKey(word).toString(),
                    word + ": must be emitted bare as a property key");
        }
        assertTrue(parses("MATCH (a:Person)-[:TRUE]->(b) RETURN count(b) AS c"),
                "TRUE: bare relationship type rejected");
        assertEquals("TRUE", Ident.relationshipType("TRUE").toString(),
                "TRUE: must be emitted bare as a relationship type");
    }

    @Test
    @DisplayName("a quoted identifier still has a character set, and Ident refuses what the "
            + "dialect cannot parse")
    void quotedPatternIdentifiersHaveACharacterSet() {
        // Probed to work inside backticks in a label position.
        for (String name : List.of("My Label", "with-dash", "with.dot", "with/slash")) {
            assertTrue(parses("MATCH (n:`" + name + "`) RETURN count(n) AS c"),
                    name + ": no longer parses quoted");
            assertEquals("`" + name + "`", Ident.label(name).toString());
        }
        // A leading underscore is a plain word character, so it needs no quoting at all.
        assertTrue(parses("MATCH (n:_leading) RETURN count(n) AS c"));
        assertEquals("_leading", Ident.label("_leading").toString());
        // Probed to be a syntax error even inside backticks; Ident refuses them at construction
        // rather than emitting a query that fails at the caller's call site.
        for (String name : List.of("with$dollar", "1leading", "with%pct", "with'quote",
                "with\"dquote", "with,comma", "emoji😀")) {
            assertFalse(parses("MATCH (n:`" + name + "`) RETURN count(n) AS c"),
                    name + ": now parses quoted — Ident could accept it");
            assertThrows(IllegalArgumentException.class, () -> Ident.label(name),
                    name + ": accepted by Ident but rejected by the engine");
        }
        // ...while property keys and aliases genuinely accept anything backtick-free.
        for (String name : List.of("with$dollar", "1leading", "with%pct", "emoji😀")) {
            assertTrue(parses("MATCH (n:Person) RETURN n.`" + name + "` AS c"),
                    name + ": no longer legal as a quoted property key");
            assertEquals("`" + name + "`", Ident.propertyKey(name).toString());
        }
    }

    /**
     * The same reserved sets, in the clauses that write.
     *
     * <p>The tests above probe reserved words where they are read — a {@code MATCH} label, a
     * {@code RETURN} key. A word could in principle tokenize differently after {@code SET},
     * {@code REMOVE} or {@code MERGE}, and the failure would be a syntax error at the caller's
     * call site rather than ours, so the write positions get probed rather than assumed.
     *
     * <p>Runs against a scratch graph: unlike the read probes, these statements mutate.
     */
    @Test
    @DisplayName("the reserved sets hold in the SET, REMOVE and MERGE positions too")
    void reservationsHoldInWritePositions() {
        List<String> wrong = new ArrayList<>();
        for (String word : Ident.RESERVED_IN_PATTERNS) {
            try (KnowledgeGraph scratch = KnowledgeGraph.createInMemory()) {
                scratch.cypher("CREATE (:Person {id: 1, title: 'Ada'})");
                if (mutates(scratch, "MATCH (n:Person) SET n." + word + " = 1")) {
                    wrong.add(word + ": bare SET key parsed, so it is not reserved after all");
                }
                if (!mutates(scratch, "MATCH (n:Person) SET n.`" + word + "` = 1")) {
                    wrong.add(word + ": quoted SET key failed, so quoting is not the escape");
                }
                if (!mutates(scratch, "MATCH (n:Person) REMOVE n.`" + word + "`")) {
                    wrong.add(word + ": quoted REMOVE key failed");
                }
                if (!mutates(scratch, "MERGE (:`" + word + "` {id: 1})")) {
                    wrong.add(word + ": quoted MERGE label failed");
                }
            }
        }
        assertEquals(List.of(), wrong);

        // ...and the emitted form is the same one the read half gets, from the same Ident.
        Node person = node("Person").named("n");
        assertEquals("MATCH (n:Person) SET n.`MATCH` = $p0",
                match(person).set(person.prop("MATCH").to(1)).cypher());
        assertEquals("MATCH (n:Person) REMOVE n.`MATCH`",
                match(person).remove(person.prop("MATCH")).cypher());
        assertEquals("MERGE (:`MATCH` {id: $p0})",
                Cypher.merge(node("MATCH").withProperty("id", 1)).cypher());
    }

    @Test
    @DisplayName("empty identifiers are rejected in every position")
    void emptyIdentifiersAreRejected() {
        assertThrows(IllegalArgumentException.class, () -> Ident.label(""));
        assertThrows(IllegalArgumentException.class, () -> Ident.relationshipType(""));
        assertThrows(IllegalArgumentException.class, () -> Ident.variable(""));
        assertThrows(IllegalArgumentException.class, () -> Ident.propertyKey(""));
        assertThrows(IllegalArgumentException.class, () -> Ident.alias(""));
        assertThrows(IllegalArgumentException.class, () -> Ident.label(null));
    }

    /** Whether the engine accepts a mutating statement at all; the effect is irrelevant here. */
    private static boolean mutates(KnowledgeGraph scratch, String statement) {
        try {
            scratch.cypher(statement, Map.of());
            return true;
        } catch (RuntimeException e) {
            return false;
        }
    }

    /** Whether the engine accepts a query at all; the rows are irrelevant here. */
    private static boolean parses(String query) {
        try {
            graph.query(query, Map.of());
            return true;
        } catch (RuntimeException e) {
            return false;
        }
    }
}
