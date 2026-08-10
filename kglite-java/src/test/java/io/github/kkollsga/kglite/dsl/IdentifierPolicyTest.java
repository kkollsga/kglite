package io.github.kkollsga.kglite.dsl;

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

    /** Words legal bare as a label or property key but not as a variable. */
    private static final List<String> RESERVED_ONLY_FOR_VARIABLES =
            List.of("ORDER", "IN", "IS", "NOT", "CREATE", "SET", "AS", "ON");

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
            if (isBooleanLiteral(word)) {
                // Probed: the tokenizer resolves these to the literal even inside backticks, so
                // they are unrepresentable as labels and the DSL refuses them rather than emitting
                // something the engine will reject at the caller's call site.
                assertFalse(parses("MATCH (n:`" + word + "`) RETURN count(n) AS c"),
                        word + ": now representable as a quoted label — the Ident rejection and "
                                + "this expectation both need revisiting");
                assertThrows(IllegalArgumentException.class, () -> Ident.label(word));
                continue;
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
            if (isBooleanLiteral(word)) {
                assertThrows(IllegalArgumentException.class, () -> Ident.variable(word));
                continue;
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

    private static boolean isBooleanLiteral(String word) {
        return word.equals("TRUE") || word.equals("FALSE");
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
