package io.github.kkollsga.kglite.dsl;

import static io.github.kkollsga.kglite.dsl.Cypher.alias;
import static io.github.kkollsga.kglite.dsl.Cypher.count;
import static io.github.kkollsga.kglite.dsl.Cypher.match;
import static io.github.kkollsga.kglite.dsl.Cypher.node;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.stream.Stream;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

/**
 * D2 gate (a): the DSL emits exactly the corpus's Cypher — character for character, parameter for
 * parameter, with no normalisation on either side.
 *
 * <p>Read the header of {@link Corpus} before changing an expectation here. The golden strings live
 * in the corpus because they are also the queries the raw route runs, so an expectation cannot be
 * quietly relaxed to match a renderer that has drifted: relaxing it changes what D0 executes and
 * what D2's dual-route gate compares.
 */
class EmittedCypherTest {

    static Stream<Corpus.Entry> expressibleEntries() {
        return Corpus.entries().stream().filter(entry -> entry.dsl() != null);
    }

    @ParameterizedTest(name = "{0}")
    @MethodSource("expressibleEntries")
    @DisplayName("the DSL emits the corpus entry's Cypher exactly")
    void emitsExactly(Corpus.Entry entry) {
        assertEquals(entry.cypher(), entry.dsl().cypher(), entry.name() + ": emitted Cypher");
        assertEquals(entry.params(), entry.dsl().params(), entry.name() + ": emitted parameters");
    }

    @Test
    @DisplayName("every expressible entry carries a statement, so the gate cannot pass vacuously")
    void everyExpressibleEntryIsCovered() {
        List<String> uncovered = Corpus.entries().stream()
                .filter(entry -> entry.expressibility() == Corpus.Expressibility.READ_HALF
                        || entry.expressibility() == Corpus.Expressibility.WRITE_HALF)
                .filter(entry -> entry.dsl() == null)
                .map(Corpus.Entry::name)
                .toList();
        assertTrue(uncovered.isEmpty(),
                "these entries claim to be expressible by the v1 clause set but carry no "
                        + "statement: " + uncovered);
        assertTrue(expressibleEntries().count() >= 60,
                "the emitted-Cypher gate must cover at least 60 statements");
    }

    /**
     * The R1 positive control. A renderer that dropped {@code DESC} would still produce valid
     * Cypher and still return rows, so the only thing standing between that bug and a green build
     * is this comparison — which means the comparison itself has to be shown catching it.
     */
    @Test
    @DisplayName("positive control: a one-token mutation of the emitted text is caught")
    void mutatedEmissionIsCaught() {
        Node person = node("Person").named("p");
        String golden = "MATCH (p:Person) RETURN p.title AS name ORDER BY name DESC";

        Statement correct = match(person)
                .returning(person.prop("title").as("name"))
                .orderBy(alias("name").desc());
        assertEquals(golden, correct.cypher(), "the un-mutated statement must clear the gate");

        // The mutation: ascending instead of descending. Valid Cypher, wrong answer order.
        Statement mutated = match(person)
                .returning(person.prop("title").as("name"))
                .orderBy(alias("name").asc());
        assertNotEquals(golden, mutated.cypher(),
                "the gate cannot go red: a flipped sort direction produced an identical string");
        assertEquals("MATCH (p:Person) RETURN p.title AS name ORDER BY name ASC", mutated.cypher());

        // And a mutation on the parameter side: a second value must take a second slot, never
        // reuse the first, or two callers' values would collide in one map entry.
        Statement twoEqualValues = match(person)
                .where(person.prop("age").gt(30).and(person.prop("id").gt(30)))
                .returning(person.prop("id").as("id"));
        assertEquals("MATCH (p:Person) WHERE p.age > $p0 AND p.id > $p1 RETURN p.id AS id",
                twoEqualValues.cypher());
        assertEquals(Map.of("p0", 30, "p1", 30), twoEqualValues.params(),
                "equal values must not be deduplicated into one parameter");
    }

    /**
     * Cheap insurance against the corpus lagging the API, which is how "we have tests" becomes
     * untrue quietly: an entry point nobody exercises is an entry point nobody has checked emits
     * anything sensible.
     */
    @Test
    @DisplayName("every Cypher entry point is exercised by the corpus")
    void everyEntryPointIsExercised() {
        Path source = Path.of("src/test/java/io/github/kkollsga/kglite/dsl/Corpus.java");
        String corpus;
        try {
            corpus = Files.readString(source);
        } catch (IOException e) {
            throw new UncheckedIOException("cannot read " + source.toAbsolutePath(), e);
        }
        List<String> unexercised = new ArrayList<>();
        for (Method method : Cypher.class.getDeclaredMethods()) {
            if (!Modifier.isPublic(method.getModifiers())) {
                continue;
            }
            // Static-imported, so the call site is the bare name; the lookbehind keeps
            // `Cypher.and` from being satisfied by a `.and(` chained onto a Condition.
            java.util.regex.Pattern call =
                    java.util.regex.Pattern.compile("(?<![\\w.])" + method.getName() + "\\(");
            if (!call.matcher(corpus).find()) {
                unexercised.add(method.getName());
            }
        }
        unexercised.sort(String::compareTo);
        assertEquals(List.of(), unexercised,
                "these Cypher entry points are not used by any corpus entry, so nothing checks "
                        + "what they emit");
    }

    @Test
    @DisplayName("parameters are numbered in emission order, left to right")
    void parametersFollowEmissionOrder() {
        Node person = node("Person").named("p");
        Statement statement = match(person.withProperty("city", "London"))
                .where(person.prop("age").gt(30))
                .returning(person.prop("id").as("id"))
                .orderBy(person.prop("id").asc())
                .skip(1)
                .limit(2);
        assertEquals("MATCH (p:Person {city: $p0}) WHERE p.age > $p1 RETURN p.id AS id "
                + "ORDER BY p.id ASC SKIP $p2 LIMIT $p3", statement.cypher());
        assertEquals(List.of("p0", "p1", "p2", "p3"), List.copyOf(statement.params().keySet()));
        assertEquals("London", statement.params().get("p0"));
        assertEquals(30, statement.params().get("p1"));
        assertEquals(1L, statement.params().get("p2"));
        assertEquals(2L, statement.params().get("p3"));
    }

    @Test
    @DisplayName("duplicate RETURN aliases are rejected while the statement is being built")
    void duplicateAliasesAreRejectedAtBuildTime() {
        Node person = node("Person").named("p");
        IllegalArgumentException thrown = assertThrows(IllegalArgumentException.class,
                () -> match(person).returning(
                        person.prop("id").as("v"),
                        person.prop("title").as("v")));
        assertTrue(thrown.getMessage().contains("duplicate RETURN alias \"v\""), thrown.getMessage());
        assertTrue(thrown.getMessage().contains("position 1"), thrown.getMessage());

        // The raw route's behaviour, for contrast: it accepts the same query and silently keeps
        // one column. The builder rejects it regardless of what the engine does with it.
        Statement distinctAliases = match(person)
                .returning(person.prop("id").as("v"), person.prop("title").as("w"));
        assertEquals("MATCH (p:Person) RETURN p.id AS v, p.title AS w", distinctAliases.cypher());
    }

    @Test
    @DisplayName("a WITH stage projects, deduplicates and filters, and its aliases must be distinct")
    void withStagesProjectAndFilter() {
        Node person = node("Person").named("p");

        // DISTINCT is a token of the clause, so the two spellings must not render alike — the
        // mutation this catches is a renderer that dropped it, which stays valid Cypher and
        // quietly returns duplicate rows to every following stage.
        Statement plain = match(person)
                .with(person.prop("city").as("city"))
                .returning(alias("city").as("city"));
        Statement deduplicated = match(person)
                .withDistinct(person.prop("city").as("city"))
                .returning(alias("city").as("city"));
        assertEquals("MATCH (p:Person) WITH p.city AS city RETURN city AS city", plain.cypher());
        assertEquals("MATCH (p:Person) WITH DISTINCT p.city AS city RETURN city AS city",
                deduplicated.cypher());
        assertNotEquals(plain.cypher(), deduplicated.cypher());

        // A WHERE attaches to the stage it follows, whether that stage matched or projected.
        assertEquals("MATCH (p:Person) WHERE p.age > $p0 WITH p.city AS city, count(p) AS n "
                        + "WHERE n > $p1 RETURN city AS city",
                match(person)
                        .where(person.prop("age").gt(30))
                        .with(person.prop("city").as("city"), count(person.ref()).as("n"))
                        .where(alias("n").gt(1))
                        .returning(alias("city").as("city"))
                        .cypher());

        // Duplicate aliases are rejected in a WITH for a sharper reason than in a RETURN: the
        // lost column is missing from the scope every following stage reads, not only from the
        // output. The message names the clause so the two cases are distinguishable.
        IllegalArgumentException thrown = assertThrows(IllegalArgumentException.class,
                () -> match(person).with(
                        person.prop("id").as("v"),
                        person.prop("title").as("v")));
        assertTrue(thrown.getMessage().contains("duplicate WITH alias \"v\""), thrown.getMessage());
    }

    /**
     * The escape hatch is the one place caller text is emitted unchanged, so the checks around it
     * are about the machinery, not the text: a fragment must not reach into the emitter's own
     * parameter namespace, and a declared parameter must be one the fragment actually uses.
     */
    @Test
    @DisplayName("a raw fragment keeps its own parameters and cannot touch the emitter's namespace")
    void rawFragmentsCarryTheirOwnParameters() {
        Node person = node("Person").named("p");

        // The generated numbering is unaffected by a raw fragment's own names: the LIMIT below is
        // still $p0, because the two namespaces are counted separately.
        Statement mixed = match(person)
                .where(Cypher.raw("size(p.title) > $min", Map.of("min", 2)))
                .returning(person.prop("title").as("name"))
                .limit(2);
        assertEquals("MATCH (p:Person) WHERE size(p.title) > $min RETURN p.title AS name "
                + "LIMIT $p0", mixed.cypher());
        assertEquals(List.of("min", "p0"), List.copyOf(mixed.params().keySet()));

        // Empty, and a fragment reaching into the emitter's namespace.
        assertThrows(IllegalArgumentException.class, () -> Cypher.raw(" "));
        assertTrue(assertThrows(IllegalArgumentException.class,
                        () -> Cypher.raw("p.age > $p0")).getMessage().contains("namespace"));
        assertTrue(assertThrows(IllegalArgumentException.class,
                        () -> Cypher.raw("p.age > $q", Map.of("p0", 1)))
                .getMessage().contains("namespace"));

        // A declared parameter the fragment never refers to is a typo, and $min must not be
        // satisfied by $min2 — the reference check matches whole names.
        assertTrue(assertThrows(IllegalArgumentException.class,
                        () -> Cypher.raw("size(p.title) > $min", Map.of("mim", 2)))
                .getMessage().contains("never refers to $mim"));
        assertThrows(IllegalArgumentException.class,
                () -> Cypher.raw("size(p.title) > $min2", Map.of("min", 2)));

        // One name, one value, per statement: agreeing repeats are fine, disagreeing ones are the
        // case where the emitted text and the parameter map would describe different queries.
        Statement repeated = match(person)
                .where(Cypher.raw("size(p.title) > $min", Map.of("min", 2)))
                .returning(Cypher.raw("size(p.title) - $min", Map.of("min", 2)).as("over"));
        assertEquals("MATCH (p:Person) WHERE size(p.title) > $min "
                + "RETURN size(p.title) - $min AS over", repeated.cypher());
        assertEquals(Map.of("min", 2), repeated.params());
        assertTrue(assertThrows(IllegalArgumentException.class,
                        () -> match(person)
                                .where(Cypher.raw("size(p.title) > $min", Map.of("min", 2)))
                                .returning(Cypher.raw("size(p.title) - $min", Map.of("min", 3))
                                        .as("over")))
                .getMessage().contains("bound twice"));
    }

    @Test
    @DisplayName("there is no returning(node): the structural functions are the offered route")
    void wholeNodeProjectionIsNotOffered() {
        Node person = node("Person").named("p");

        // What the DSL does offer, all four of them, and all of them returning real values.
        assertEquals("MATCH (p:Person) RETURN properties(p) AS a, labels(p) AS b, id(p) AS c",
                match(person).returning(
                        person.properties().as("a"),
                        person.labels().as("b"),
                        person.id().as("c")).cypher());

        // And the bare variable exists only as an aggregate argument, never as a projection —
        // Projection is only constructible through Expr.as, and count(p) is what p.ref() is for.
        assertEquals("MATCH (p:Person) RETURN count(p) AS n",
                match(person).returning(count(person.ref()).as("n")).cypher());
    }

    @Test
    @DisplayName("a value position rejects a query element instead of serialising it")
    void valuePositionRejectsQueryElements() {
        Node person = node("Person").named("p");
        Node other = node("Person").named("q");
        IllegalArgumentException thrown = assertThrows(IllegalArgumentException.class,
                () -> person.prop("city").eq(other.prop("city")));
        assertTrue(thrown.getMessage().contains("value position"), thrown.getMessage());
    }

    @Test
    @DisplayName("an unnamed pattern element cannot be referenced")
    void unnamedElementsCannotBeReferenced() {
        assertThrows(IllegalStateException.class, () -> node("Person").prop("title"));
        assertThrows(IllegalStateException.class, () -> Cypher.rel("KNOWS").type());
    }
}
