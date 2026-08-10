package io.github.kkollsga.kglite.dsl;

import static io.github.kkollsga.kglite.dsl.Cypher.count;
import static io.github.kkollsga.kglite.dsl.Cypher.match;
import static io.github.kkollsga.kglite.dsl.Cypher.node;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.kkollsga.kglite.KgliteException;
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

    /**
     * The same exploit strings, through the builders that emit a <em>write</em>.
     *
     * <p>The read half's identifier positions are a leak at worst; a {@code MERGE} label, a
     * {@code SET} key or a {@code REMOVE} key that broke out of its quoting would let an attack
     * string reach a clause that changes the graph. The renderers differ, so the gate has to run
     * against each of them rather than assume the label case covers everything.
     *
     * <p>Executed through {@code cypher()}, the mutating entry point, so an injected
     * {@code DETACH DELETE} really would run.
     */
    @Test
    @DisplayName("hostile identifiers are inert in write positions too")
    void writePositionsAreInert() {
        List<String> damage = new ArrayList<>();
        try (KnowledgeGraph graph = graphWithASecret()) {
            int built = 0;
            for (String exploit : EXPLOITS) {
                List<Statement> statements = writeStatements(exploit);
                built += statements.size();
                for (Statement statement : statements) {
                    try {
                        graph.cypher(statement.cypher(), statement.params());
                    } catch (RuntimeException e) {
                        // A syntax or execution error is a fine outcome: nothing ran.
                        continue;
                    }
                }
            }
            assertTrue(built >= 8,
                    "the write-position gate built only " + built + " statements — a gate that "
                            + "runs nothing cannot catch anything");
            long people = countOf(graph, "MATCH (n:Person) RETURN count(n) AS c");
            long secrets = countOf(graph, "MATCH (n:Secret) RETURN count(n) AS c");
            if (people != 2) {
                damage.add("Person nodes went from 2 to " + people);
            }
            if (secrets != 1) {
                damage.add("Secret nodes went from 1 to " + secrets);
            }
        }
        assertEquals(List.of(), damage, "a hostile identifier reached a write clause");
    }

    /**
     * Every write statement the DSL can build from one caller-supplied identifier.
     *
     * <p>The created and merged patterns carry a {@code Payload} label rather than {@code Person}
     * so that the damage check above measures the exploit and not the harness: a statement that
     * legitimately creates a {@code Person} would move the count on its own.
     *
     * <p>The label and relationship-type positions are asserted to refuse the whole exploit family
     * at construction, which is why nothing is added for them.
     */
    private static List<Statement> writeStatements(String hostile) {
        List<Statement> statements = new ArrayList<>();
        Node person = node("Person").named("p");
        // Property keys accept any backtick-free text, so these are the write positions where a
        // hostile string genuinely reaches the emitted query and has to be quoted inertly.
        if (hostile.indexOf('`') < 0) {
            statements.add(match(person).set(person.prop(hostile).to("owned")));
            statements.add(match(person).remove(person.prop(hostile)));
            statements.add(Cypher.create(node("Payload").withProperty(hostile, "owned")));
            statements.add(Cypher.merge(node("Payload").named("q").withProperty(hostile, "owned")));
        }
        assertThrows(IllegalArgumentException.class, () -> Cypher.merge(node(hostile)),
                "a MERGE accepted the hostile label " + quote(hostile));
        assertThrows(IllegalArgumentException.class,
                () -> match(person).create(person.to(Cypher.rel(hostile), person)),
                "a CREATE accepted the hostile relationship type " + quote(hostile));
        return statements;
    }

    /**
     * The positive control for the write positions: the exploit that works there, shown working.
     *
     * <p>The read half's exploit closes a backticked <em>label</em>; the write half's closes a
     * backticked <em>property key</em> in a {@code SET}, which turns the rest of the caller's
     * string into clauses of its own. Naive quoting runs it; {@link Ident} refuses to build it,
     * because a backtick has no escape in this dialect.
     */
    @Test
    @DisplayName("positive control: naive quoting of a SET key is a working delete, and Ident "
            + "refuses to build it")
    void naiveWriteQuotingIsAWorkingExploit() {
        String exploit = "x` = 1 DETACH DELETE p //";
        String naive = "MATCH (p:Person) SET p.`" + exploit + "` = 'owned'";

        try (KnowledgeGraph graph = graphWithASecret()) {
            graph.cypher(naive);
            assertEquals(0L, countOf(graph, "MATCH (n:Person) RETURN count(n) AS c"),
                    "the naive spelling no longer deletes, so this control has stopped being "
                            + "about a real exploit");
        }

        Node person = node("Person").named("p");
        assertRejected(() -> person.prop(exploit), "property key", exploit);
        assertRejected(() -> Cypher.create(node("Payload").withProperty(exploit, "owned")),
                "property key", exploit);
        assertRejected(() -> Ident.label(exploit), "node label", exploit);
    }

    @Test
    @DisplayName("the write builders emit a hostile-but-quotable key as exactly one identifier")
    void hostileKeysAreQuotedInWritePositions() {
        String hostile = "Person) DETACH DELETE n //";
        List<Statement> statements = writeStatements(hostile);
        assertEquals(4, statements.size(),
                "the write-position gate built no statements, so it proved nothing");
        assertEquals("MATCH (p:Person) SET p.`" + hostile + "` = $p0", statements.get(0).cypher());
        assertEquals("MATCH (p:Person) REMOVE p.`" + hostile + "`", statements.get(1).cypher());
        assertEquals("CREATE (:Payload {`" + hostile + "`: $p0})", statements.get(2).cypher());
        assertEquals("MERGE (q:Payload {`" + hostile + "`: $p0})", statements.get(3).cypher());

        try (KnowledgeGraph graph = graphWithASecret()) {
            for (Statement statement : statements) {
                graph.cypher(statement.cypher(), statement.params());
            }
            assertEquals(2L, countOf(graph, "MATCH (n:Person) RETURN count(n) AS c"),
                    "nothing was deleted");
            assertEquals(1L, countOf(graph, "MATCH (n:Payload) RETURN count(n) AS c"),
                    "the CREATE wrote one node and the MERGE matched it — which it can only do "
                            + "if both emitted the hostile string as the same single key");
        }
        // ...and the SET on its own — the REMOVE above takes the property back off again —
        // stored the whole hostile string as one key.
        try (KnowledgeGraph graph = graphWithASecret()) {
            Statement set = statements.get(0);
            graph.cypher(set.cypher(), set.params());
            assertEquals("owned", graph.query(
                            "MATCH (p:Person) WHERE p.id = 1 RETURN p.`" + hostile + "` AS v")
                    .get(0).get("v"),
                    "the quoted key must have been written as a property, not run as a clause");
        }
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

    /**
     * The control that scopes the guarantee: the boundary sits <em>exactly</em> at
     * {@link Cypher#raw}, and nowhere earlier.
     *
     * <p>Everything above shows hostile text is inert through the modelled surface. That is only
     * half the claim worth publishing, because the DSL also ships an escape hatch that emits
     * caller text verbatim — and a reader who took "injection is structurally impossible" as
     * unconditional would be wrong in exactly one place. So this test takes one string, walks it
     * through every position the DSL models (refused or quoted inertly, never syntax), and then
     * puts the same string through the hatch and watches it delete the graph.
     *
     * <p>It is a positive control in both directions. If the modelled surface ever lets the string
     * through, the first half goes red; if {@code raw} ever started escaping — quietly turning the
     * documented "this is your responsibility" into a promise the DSL cannot keep for every
     * dialect — the second half goes red and the docs get revisited deliberately.
     */
    @Test
    @DisplayName("positive control: raw is the only path a hostile string survives as syntax")
    void rawIsTheOnlyPathThatCarriesCallerText() {
        String exploit = "DETACH DELETE p //";
        // The same attack with a closing parenthesis, which pattern positions cannot represent at
        // all — the two spellings cover both halves of the identifier policy.
        String closer = "Person) DETACH DELETE n //";
        Node person = node("Person").named("p");

        // 1. The modelled surface. A pattern identifier is either refused outright...
        assertThrows(IllegalArgumentException.class, () -> node(closer));
        assertThrows(IllegalArgumentException.class, () -> Cypher.rel(closer));
        assertThrows(IllegalArgumentException.class, () -> node("Person").named(closer));

        // ...or quoted into one identifier, which is a label nothing has, not a clause.
        assertEquals("MATCH (n:`" + exploit + "`) RETURN count(n) AS c",
                match(node(exploit).named("n"))
                        .returning(count(node(exploit).named("n").ref()).as("c"))
                        .cypher());

        // Property keys and aliases take it too, and quote it the same way.
        assertEquals("MATCH (p:Person) RETURN p.`" + exploit + "` AS `" + exploit + "`",
                match(person).returning(person.prop(exploit).as(exploit)).cypher());

        // ...and a value position never puts it in the text at all.
        Statement asValue = match(person)
                .where(person.prop("title").eq(exploit))
                .returning(count(person.ref()).as("c"));
        assertEquals("MATCH (p:Person) WHERE p.title = $p0 RETURN count(p) AS c",
                asValue.cypher());
        assertEquals(Map.of("p0", exploit), asValue.params());

        // The hatch does not leak backwards either: a Raw is a query element, so it is refused
        // where data belongs rather than serialised as a parameter value.
        assertThrows(IllegalArgumentException.class,
                () -> person.prop("title").eq(Cypher.raw("p.title")));

        // 2. The hatch. The same string is now the query, character for character.
        Statement viaClause = match(person)
                .rawClause(exploit)
                .returning(count(person.ref()).as("c"));
        assertEquals("MATCH (p:Person) " + exploit + " RETURN count(p) AS c", viaClause.cypher());
        Statement viaExpression = match(person)
                .where(Cypher.raw("p.id > 0 " + exploit))
                .returning(count(person.ref()).as("c"));
        assertEquals("MATCH (p:Person) WHERE p.id > 0 " + exploit + " RETURN count(p) AS c",
                viaExpression.cypher());

        // The routing guard is still in the way of a read that mutates — a second line of defence
        // that is worth knowing about and is not the guarantee being scoped here.
        try (KnowledgeGraph graph = graphWithASecret()) {
            KgliteException refused = assertThrows(KgliteException.class, () -> viaClause.on(graph));
            assertEquals("InvalidArgument", refused.statusName());
            assertEquals(2L, countOf(graph, "MATCH (n:Person) RETURN count(n) AS c"));
        }

        // ...and on the write route, which is where a caller who built the fragment from user
        // input would end up, it does exactly what it says.
        for (Statement statement : List.of(viaClause, viaExpression)) {
            try (KnowledgeGraph graph = graphWithASecret()) {
                graph.cypher(statement.cypher(), statement.params());
                assertEquals(0L, countOf(graph, "MATCH (n:Person) RETURN count(n) AS c"),
                        "raw no longer carries caller text to the engine unchanged — the "
                                + "documented scope of this DSL's injection property has moved, "
                                + "so revisit Raw's javadoc and the README section with it");
            }
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
