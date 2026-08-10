package io.github.kkollsga.kglite.dsl;

import static io.github.kkollsga.kglite.dsl.Cypher.count;
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
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/**
 * The injection gate, with its naive-renderer positive control.
 *
 * <p>The threat here is not hypothetical. Backtick-quoting a caller-supplied identifier — the
 * obvious implementation, and the one a hurried afternoon produces — is a working injection against
 * this engine: {@code "Person`) DETACH DELETE n //"} closes the quote, appends a clause and
 * comments out the rest, and the query that "reads a count" deletes every node instead. The same
 * closer works in all four identifier positions.
 *
 * <p>The DSL's answer is structural rather than careful:
 *
 * <ol>
 *   <li>A caller <em>value</em> can never become syntax, because no method signature offers that
 *       path — every value-position parameter is {@code Object} and every one of them is emitted
 *       as {@code $p<n>}.
 *   <li>A caller <em>identifier</em> is validated in {@link Ident}'s constructor, so an
 *       unrepresentable one cannot exist in the AST at all.
 * </ol>
 *
 * <p>Claim two needs a harness that can fail. This test runs the same hostile strings through the
 * real renderer and through a deliberately-naive one, using one shared checker, and asserts the
 * checker clears the first and catches the second. A harness that cannot catch the known-vulnerable
 * implementation would be no evidence about the safe one.
 */
class InjectionTest {

    /** The strings the investigation reproduced an engine-level exploit with, plus the family. */
    private static final List<String> EXPLOITS = List.of(
            "Person`) DETACH DELETE n //",
            "Person`) RETURN n.title AS leaked //",
            "n` :Secret) RETURN n.title AS leaked //",
            "Person`}) DETACH DELETE n //",
            "Person) DETACH DELETE n //",
            "Person'; DROP GRAPH; --");

    /** Values that must survive a round trip untouched, because they are only ever data. */
    private static final List<String> HOSTILE_VALUES = List.of(
            "back`tick",
            "back\\slash",
            "new\nline",
            "single'quote",
            "double\"quote",
            "semi;colon",
            "dash--dash",
            "emoji😀",
            "') DETACH DELETE n //",
            "' OR ''='");

    /** What a renderer produced for one caller-supplied label, or why it refused. */
    private record Attempt(String cypher, Map<String, Object> params, String rejection) {
        static Attempt rejected(String reason) {
            return new Attempt(null, null, reason);
        }
    }

    /** Turns a caller-supplied label into a "count the nodes with this label" statement. */
    private interface LabelRenderer {
        Attempt render(String label);
    }

    /** The real thing: an identifier goes through {@link Ident}, a value goes to a parameter. */
    private static final LabelRenderer SAFE = label -> {
        try {
            Node target = node(label).named("n");
            Statement statement = match(target).returning(count(target.ref()).as("c"));
            return new Attempt(statement.cypher(), statement.params(), null);
        } catch (IllegalArgumentException e) {
            return Attempt.rejected(e.getMessage());
        }
    };

    /**
     * The positive control: quote the identifier in backticks and hope. This is not a straw man —
     * it is what "escaping" looks like to anyone who has not probed the tokenizer, and it is the
     * implementation the investigation demonstrated an exploit against.
     */
    private static final LabelRenderer NAIVE = label ->
            new Attempt("MATCH (n:`" + label + "`) RETURN count(n) AS c", Map.of(), null);

    @Test
    @DisplayName("hostile identifiers are inert through the DSL, and the same check catches a "
            + "naive renderer")
    void identifiersAreInertAndTheHarnessCanGoRed() {
        List<String> safeDamage = damageFrom(SAFE);
        assertEquals(List.of(), safeDamage,
                "the DSL let a hostile identifier change or leak the graph");

        List<String> naiveDamage = damageFrom(NAIVE);
        assertFalse(naiveDamage.isEmpty(),
                "the harness cannot go red: the naive backtick-quoting renderer passed the same "
                        + "check that cleared the DSL, so clearing the DSL proves nothing");
        assertTrue(naiveDamage.stream().anyMatch(damage -> damage.contains("Person nodes")),
                "the naive renderer was expected to destroy data; damage was " + naiveDamage);
        System.out.println("injection positive control — naive renderer: " + naiveDamage);
    }

    /**
     * Runs every exploit string through a renderer and reports what it did to the graph.
     *
     * <p>Deliberately executed through {@code cypher()}, the mutating entry point, so an injected
     * {@code DETACH DELETE} really runs. Routing through {@code query()} instead would have the
     * read/write guard block the attack and the test would prove the guard, not the renderer.
     */
    private static List<String> damageFrom(LabelRenderer renderer) {
        List<String> damage = new ArrayList<>();
        try (KnowledgeGraph graph = graphWithASecret()) {
            for (String exploit : EXPLOITS) {
                Attempt attempt = renderer.render(exploit);
                if (attempt.rejection() != null) {
                    continue;
                }
                List<Map<String, Object>> rows;
                try {
                    rows = graph.cypher(attempt.cypher(), attempt.params());
                } catch (RuntimeException e) {
                    // A syntax error is a fine outcome: the attack did not run.
                    continue;
                }
                if (rows.toString().contains("classified")) {
                    damage.add("leaked the secret through " + quote(exploit));
                }
            }
            long people = countOf(graph, "MATCH (n:Person) RETURN count(n) AS c");
            long secrets = countOf(graph, "MATCH (n:Secret) RETURN count(n) AS c");
            if (people != 2) {
                damage.add("Person nodes went from 2 to " + people);
            }
            if (secrets != 1) {
                damage.add("Secret nodes went from 1 to " + secrets);
            }
        }
        return damage;
    }

    @Test
    @DisplayName("a backtick-bearing identifier is rejected at construction, in every position")
    void backtickBearingIdentifiersAreRejected() {
        for (String exploit : EXPLOITS) {
            if (exploit.indexOf('`') < 0) {
                continue;
            }
            assertRejected(() -> Ident.label(exploit), "label", exploit);
            assertRejected(() -> Ident.relationshipType(exploit), "relationship type", exploit);
            assertRejected(() -> Ident.variable(exploit), "pattern variable", exploit);
            assertRejected(() -> Ident.propertyKey(exploit), "property key", exploit);
            assertRejected(() -> Ident.alias(exploit), "RETURN alias", exploit);
        }
    }

    @Test
    @DisplayName("a backtick-free but syntactically hostile name is refused in pattern positions "
            + "and quoted inertly elsewhere")
    void hostileButQuotableNames() {
        String hostile = "Person) DETACH DELETE n //";

        // Label, relationship type and variable: the dialect cannot represent the character set
        // even inside backticks, so these are refused outright rather than emitted and hoped for.
        assertThrows(IllegalArgumentException.class, () -> Ident.label(hostile));
        assertThrows(IllegalArgumentException.class, () -> Ident.relationshipType(hostile));
        assertThrows(IllegalArgumentException.class, () -> Ident.variable(hostile));

        // Property keys and aliases accept it, quote it, and the result is one identifier.
        Node person = node("Person").named("p");
        Statement statement = match(person).returning(person.prop(hostile).as(hostile));
        assertEquals("MATCH (p:Person) RETURN p.`" + hostile + "` AS `" + hostile + "`",
                statement.cypher());
        try (KnowledgeGraph graph = graphWithASecret()) {
            List<Map<String, Object>> rows = graph.cypher(statement.cypher(), statement.params());
            assertEquals(2, rows.size(), "the quoted key is a key, not a clause");
            assertEquals(2L, countOf(graph, "MATCH (n:Person) RETURN count(n) AS c"),
                    "nothing was deleted");
        }
    }

    @Test
    @DisplayName("hostile values round-trip as data, in both write and read positions")
    void hostileValuesAreOnlyEverParameters() {
        try (KnowledgeGraph graph = graphWithASecret()) {
            for (int i = 0; i < HOSTILE_VALUES.size(); i++) {
                String value = HOSTILE_VALUES.get(i);
                graph.cypher("CREATE (:Payload {id: $id, body: $body})",
                        Map.of("id", i, "body", value));

                Node payload = node("Payload").named("x");
                Statement statement = match(payload)
                        .where(payload.prop("body").eq(value))
                        .returning(payload.prop("body").as("body"));

                // The value is nowhere in the text; it is entirely in the parameter map.
                assertEquals("MATCH (x:Payload) WHERE x.body = $p0 RETURN x.body AS body",
                        statement.cypher());
                assertEquals(Map.of("p0", value), statement.params());
                assertFalse(statement.cypher().contains(value.substring(0, 3)),
                        "the value leaked into the query text: " + statement.cypher());

                assertEquals(List.of(Map.of("body", value)), statement.on(graph),
                        "value did not round-trip: " + quote(value));
            }
            assertEquals(2L, countOf(graph, "MATCH (n:Person) RETURN count(n) AS c"));
            assertEquals(1L, countOf(graph, "MATCH (n:Secret) RETURN count(n) AS c"));
        }
    }

    private static void assertRejected(Runnable construction, String position, String name) {
        IllegalArgumentException thrown =
                assertThrows(IllegalArgumentException.class, construction::run,
                        position + " accepted " + quote(name));
        assertTrue(thrown.getMessage().contains(position),
                "the rejection must name the position, said: " + thrown.getMessage());
        assertTrue(thrown.getMessage().contains("backtick"),
                "the rejection must name the reason, said: " + thrown.getMessage());
    }

    private static KnowledgeGraph graphWithASecret() {
        KnowledgeGraph graph = KnowledgeGraph.createInMemory();
        graph.cypher("CREATE (:Person {id: 1, title: 'Ada'})");
        graph.cypher("CREATE (:Person {id: 2, title: 'Bob'})");
        graph.cypher("CREATE (:Secret {id: 3, title: 'classified'})");
        return graph;
    }

    private static long countOf(KnowledgeGraph graph, String query) {
        Object value = graph.query(query).get(0).get("c");
        return ((Number) value).longValue();
    }

    private static String quote(String s) {
        return "\"" + s.replace("\n", "\\n") + "\"";
    }
}
