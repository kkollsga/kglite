package io.github.kkollsga.kglite.dsl;

import static io.github.kkollsga.kglite.dsl.Cypher.alias;
import static io.github.kkollsga.kglite.dsl.Cypher.and;
import static io.github.kkollsga.kglite.dsl.Cypher.anyNode;
import static io.github.kkollsga.kglite.dsl.Cypher.anyRel;
import static io.github.kkollsga.kglite.dsl.Cypher.avg;
import static io.github.kkollsga.kglite.dsl.Cypher.collect;
import static io.github.kkollsga.kglite.dsl.Cypher.collectDistinct;
import static io.github.kkollsga.kglite.dsl.Cypher.count;
import static io.github.kkollsga.kglite.dsl.Cypher.countAll;
import static io.github.kkollsga.kglite.dsl.Cypher.countDistinct;
import static io.github.kkollsga.kglite.dsl.Cypher.match;
import static io.github.kkollsga.kglite.dsl.Cypher.max;
import static io.github.kkollsga.kglite.dsl.Cypher.min;
import static io.github.kkollsga.kglite.dsl.Cypher.node;
import static io.github.kkollsga.kglite.dsl.Cypher.not;
import static io.github.kkollsga.kglite.dsl.Cypher.or;
import static io.github.kkollsga.kglite.dsl.Cypher.rel;
import static io.github.kkollsga.kglite.dsl.Cypher.sum;

import io.github.kkollsga.kglite.KnowledgeGraph;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * The frozen corpus: real Cypher, real fixture data, real expected rows.
 *
 * <p>It was written raw-first — every statement here ran green against the engine before any
 * emitter existed — and it is the single input to all three gates. An entry's {@code cypher} field
 * therefore does double duty: it is the query the raw route runs, <em>and</em> it is the golden
 * string the DSL must emit character for character.
 *
 * <p><b>If a change to the dialect makes one of these strings wrong, edit it in the same commit
 * that names the dialect change, and say which production moved.</b> An expectation that gets
 * regenerated to go green is not an expectation. The one legitimate reason to touch a golden
 * string is that the Cypher it names has changed; "the renderer now emits something else" is a
 * renderer bug, not a corpus update.
 *
 * <p>Provenance of the entries: the docs-blind consumer sandbox's own queries (its {@code
 * Tests.java} and three probes, written by an agent with no repo access), the README examples, and
 * one representative statement per clause of the v1 clause set.
 */
final class Corpus {

    private Corpus() {}

    /** Which fixture graph an entry needs. */
    enum Fixture {
        /** People, companies, {@code KNOWS} and {@code WORKS_AT}. The workhorse. */
        PEOPLE,
        /** Identifiers that must be backtick-quoted, including a reserved-word label. */
        ODD_NAMES,
    }

    /** Whether the statement is a read or a mutation, i.e. which entry point it needs. */
    enum Route {
        READ,
        WRITE,
    }

    /** How the entry relates to the v1 clause set — the census the R13 stop rule counts. */
    enum Expressibility {
        /** Expressible by the read half, and carrying the {@link Statement} that proves it. */
        READ_HALF,
        /** Inside the v1 clause set, but a later phase's clauses (writes, {@code WITH}). */
        V1_LATER,
        /** Deliberately outside v1: the clause earns no builder, or the shape is a known wart. */
        OUT_OF_V1,
    }

    /**
     * One corpus entry.
     *
     * @param name a stable identifier, used in assertion messages
     * @param fixture the graph to build before running
     * @param route whether {@code cypher} is a read or a mutation
     * @param cypher the raw Cypher — and the golden string the DSL must emit
     * @param params the parameters {@code cypher} refers to
     * @param verify for mutations, the read whose rows are the expectation; {@code null} for reads
     * @param expectedRows the expected rows, or {@code null} when only the count is asserted
     * @param expectedRowCount used when {@code expectedRows} is {@code null}
     * @param dsl the DSL statement that must emit {@code cypher}, or {@code null}
     * @param expressibility the census bucket
     * @param note why this entry is here, and for a non-expressible one, why it is not
     */
    record Entry(
            String name,
            Fixture fixture,
            Route route,
            String cypher,
            Map<String, Object> params,
            String verify,
            List<Map<String, Object>> expectedRows,
            Integer expectedRowCount,
            Statement dsl,
            Expressibility expressibility,
            String note) {

        /** Rows are compared in order exactly when the statement asked for an order. */
        boolean orderSensitive() {
            return (verify != null ? verify : cypher).contains("ORDER BY");
        }

        /** The name alone: the generated record toString would swamp every test report line. */
        @Override
        public String toString() {
            return name;
        }
    }

    // ---------------------------------------------------------------------------------------
    // Fixture graphs.
    // ---------------------------------------------------------------------------------------

    /**
     * Builds a fixture graph. Each entry gets its own, so a mutating entry cannot disturb a
     * later one and the corpus has no ordering dependency.
     *
     * @param fixture which graph to build
     * @return an open in-memory graph the caller must close
     */
    static KnowledgeGraph build(Fixture fixture) {
        KnowledgeGraph graph = KnowledgeGraph.createInMemory();
        try {
            for (String statement : setup(fixture)) {
                graph.cypher(statement);
            }
        } catch (RuntimeException e) {
            graph.close();
            throw e;
        }
        return graph;
    }

    private static List<String> setup(Fixture fixture) {
        return switch (fixture) {
            case PEOPLE -> List.of(
                    "CREATE (:Person {id: 1, title: 'Ada', age: 36, city: 'London', "
                            + "email: 'ada@example.com'})",
                    "CREATE (:Person {id: 2, title: 'Bob', age: 41, city: 'Paris'})",
                    "CREATE (:Person {id: 3, title: 'Cy', age: 29, city: 'London', "
                            + "email: 'cy@example.com'})",
                    "CREATE (:Person {id: 4, title: 'Dee', age: 29, city: 'Berlin', "
                            + "email: 'dee@example.com'})",
                    "CREATE (:Company {id: 10, title: 'Acme', city: 'London'})",
                    "CREATE (:Company {id: 11, title: 'Globex', city: 'Berlin'})",
                    "MATCH (a:Person {id: 1}), (b:Person {id: 2}) "
                            + "CREATE (a)-[:KNOWS {since: 2020}]->(b)",
                    "MATCH (a:Person {id: 2}), (b:Person {id: 3}) "
                            + "CREATE (a)-[:KNOWS {since: 2021}]->(b)",
                    "MATCH (a:Person {id: 1}), (c:Company {id: 10}) "
                            + "CREATE (a)-[:WORKS_AT {role: 'eng'}]->(c)",
                    "MATCH (a:Person {id: 3}), (c:Company {id: 10}) "
                            + "CREATE (a)-[:WORKS_AT {role: 'ops'}]->(c)",
                    "MATCH (a:Person {id: 4}), (c:Company {id: 11}) "
                            + "CREATE (a)-[:WORKS_AT {role: 'sales'}]->(c)");
            case ODD_NAMES -> List.of(
                    "CREATE (:`My Label` {id: 1, `my key`: 'v1', order: 5})",
                    "CREATE (:`MATCH` {id: 2, title: 'reserved'})");
        };
    }

    // ---------------------------------------------------------------------------------------
    // Reusable pattern pieces. Statements are immutable, so these are shared freely.
    // ---------------------------------------------------------------------------------------

    private static final Node PERSON = node("Person").named("p");
    private static final Node A = node("Person").named("a");
    private static final Node B = node("Person").named("b");
    private static final Node C = node("Person").named("c");
    private static final Node COMPANY = node("Company").named("c");

    // ---------------------------------------------------------------------------------------
    // The entries.
    // ---------------------------------------------------------------------------------------

    static List<Entry> entries() {
        List<Entry> entries = new ArrayList<>();

        entries.add(read(
                "all_titles",
                "MATCH (p:Person) RETURN p.title AS name ORDER BY p.id ASC",
                Map.of(),
                match(PERSON)
                        .returning(PERSON.prop("title").as("name"))
                        .orderBy(PERSON.prop("id").asc()),
                rows(row("name", "Ada"), row("name", "Bob"), row("name", "Cy"), row("name", "Dee")),
                "the shape that dominates the docs-blind consumer's corpus"));

        entries.add(read(
                "filter_greater_than",
                "MATCH (p:Person) WHERE p.age > $p0 RETURN p.title AS name ORDER BY p.age DESC",
                Map.of("p0", 30),
                match(PERSON)
                        .where(PERSON.prop("age").gt(30))
                        .returning(PERSON.prop("title").as("name"))
                        .orderBy(PERSON.prop("age").desc()),
                rows(row("name", "Bob"), row("name", "Ada")),
                "comparison operator, value as a parameter"));

        entries.add(read(
                "filter_less_than",
                "MATCH (p:Person) WHERE p.age < $p0 RETURN p.id AS id ORDER BY p.id ASC",
                Map.of("p0", 30),
                match(PERSON)
                        .where(PERSON.prop("age").lt(30))
                        .returning(PERSON.prop("id").as("id"))
                        .orderBy(PERSON.prop("id").asc()),
                rows(row("id", 3L), row("id", 4L)),
                "comparison operator"));

        entries.add(read(
                "filter_not_equal",
                "MATCH (p:Person) WHERE p.age <> $p0 RETURN p.id AS id ORDER BY p.id ASC",
                Map.of("p0", 36),
                match(PERSON)
                        .where(PERSON.prop("age").ne(36))
                        .returning(PERSON.prop("id").as("id"))
                        .orderBy(PERSON.prop("id").asc()),
                rows(row("id", 2L), row("id", 3L), row("id", 4L)),
                "comparison operator"));

        entries.add(read(
                "filter_at_most",
                "MATCH (p:Person) WHERE p.age <= $p0 RETURN p.id AS id ORDER BY p.id ASC",
                Map.of("p0", 29),
                match(PERSON)
                        .where(PERSON.prop("age").le(29))
                        .returning(PERSON.prop("id").as("id"))
                        .orderBy(PERSON.prop("id").asc()),
                rows(row("id", 3L), row("id", 4L)),
                "comparison operator"));

        entries.add(read(
                "filter_at_least",
                "MATCH (p:Person) WHERE p.age >= $p0 RETURN p.id AS id ORDER BY p.id ASC",
                Map.of("p0", 41),
                match(PERSON)
                        .where(PERSON.prop("age").ge(41))
                        .returning(PERSON.prop("id").as("id"))
                        .orderBy(PERSON.prop("id").asc()),
                rows(row("id", 2L)),
                "comparison operator"));

        entries.add(read(
                "filter_conjunction",
                "MATCH (p:Person) WHERE p.city = $p0 AND p.age < $p1 RETURN p.id AS id "
                        + "ORDER BY p.id ASC",
                Map.of("p0", "London", "p1", 30),
                match(PERSON)
                        .where(PERSON.prop("city").eq("London").and(PERSON.prop("age").lt(30)))
                        .returning(PERSON.prop("id").as("id"))
                        .orderBy(PERSON.prop("id").asc()),
                rows(row("id", 3L)),
                "AND, unparenthesised because both operands are simple"));

        entries.add(read(
                "filter_grouped_disjunction",
                "MATCH (p:Person) WHERE (p.city = $p0 OR p.city = $p1) AND p.age > $p2 "
                        + "RETURN p.id AS id ORDER BY p.id ASC",
                Map.of("p0", "London", "p1", "Paris", "p2", 30),
                match(PERSON)
                        .where(and(
                                or(PERSON.prop("city").eq("London"),
                                        PERSON.prop("city").eq("Paris")),
                                PERSON.prop("age").gt(30)))
                        .returning(PERSON.prop("id").as("id"))
                        .orderBy(PERSON.prop("id").asc()),
                rows(row("id", 1L), row("id", 2L)),
                "a composite operand is parenthesised, so precedence never has to be reasoned "
                        + "about"));

        entries.add(read(
                "filter_negation",
                "MATCH (p:Person) WHERE NOT (p.city = $p0) RETURN p.id AS id ORDER BY p.id ASC",
                Map.of("p0", "London"),
                match(PERSON)
                        .where(not(PERSON.prop("city").eq("London")))
                        .returning(PERSON.prop("id").as("id"))
                        .orderBy(PERSON.prop("id").asc()),
                rows(row("id", 2L), row("id", 4L)),
                "NOT always wraps its operand"));

        entries.add(read(
                "filter_in_list",
                "MATCH (p:Person) WHERE p.id IN $p0 RETURN p.id AS id ORDER BY p.id ASC",
                Map.of("p0", List.of(1, 3)),
                match(PERSON)
                        .where(PERSON.prop("id").in(List.of(1, 3)))
                        .returning(PERSON.prop("id").as("id"))
                        .orderBy(PERSON.prop("id").asc()),
                rows(row("id", 1L), row("id", 3L)),
                "one list-valued parameter, not a generated OR chain that would hit the "
                        + "expression-nesting cap"));

        entries.add(read(
                "filter_starts_with",
                "MATCH (p:Person) WHERE p.title STARTS WITH $p0 RETURN p.title AS name "
                        + "ORDER BY name ASC",
                Map.of("p0", "A"),
                match(PERSON)
                        .where(PERSON.prop("title").startsWith("A"))
                        .returning(PERSON.prop("title").as("name"))
                        .orderBy(alias("name").asc()),
                rows(row("name", "Ada")),
                "string predicate; ORDER BY references the RETURN alias"));

        entries.add(read(
                "filter_ends_with",
                "MATCH (p:Person) WHERE p.title ENDS WITH $p0 RETURN p.title AS name "
                        + "ORDER BY name ASC",
                Map.of("p0", "b"),
                match(PERSON)
                        .where(PERSON.prop("title").endsWith("b"))
                        .returning(PERSON.prop("title").as("name"))
                        .orderBy(alias("name").asc()),
                rows(row("name", "Bob")),
                "string predicate"));

        entries.add(read(
                "filter_contains",
                "MATCH (p:Person) WHERE p.city CONTAINS $p0 RETURN p.id AS id ORDER BY p.id ASC",
                Map.of("p0", "ond"),
                match(PERSON)
                        .where(PERSON.prop("city").contains("ond"))
                        .returning(PERSON.prop("id").as("id"))
                        .orderBy(PERSON.prop("id").asc()),
                rows(row("id", 1L), row("id", 3L)),
                "string predicate"));

        entries.add(read(
                "filter_regex",
                "MATCH (p:Person) WHERE p.title =~ $p0 RETURN p.title AS name ORDER BY name ASC",
                Map.of("p0", "^[AB].*"),
                match(PERSON)
                        .where(PERSON.prop("title").matches("^[AB].*"))
                        .returning(PERSON.prop("title").as("name"))
                        .orderBy(alias("name").asc()),
                rows(row("name", "Ada"), row("name", "Bob")),
                "the regex is a value, so it travels as a parameter like any other"));

        entries.add(read(
                "filter_is_null",
                "MATCH (p:Person) WHERE p.email IS NULL RETURN p.id AS id ORDER BY p.id ASC",
                Map.of(),
                match(PERSON)
                        .where(PERSON.prop("email").isNull())
                        .returning(PERSON.prop("id").as("id"))
                        .orderBy(PERSON.prop("id").asc()),
                rows(row("id", 2L)),
                "from the consumer corpus: three of its queries test null handling"));

        entries.add(read(
                "filter_is_not_null",
                "MATCH (p:Person) WHERE p.email IS NOT NULL RETURN p.id AS id ORDER BY p.id ASC",
                Map.of(),
                match(PERSON)
                        .where(PERSON.prop("email").isNotNull())
                        .returning(PERSON.prop("id").as("id"))
                        .orderBy(PERSON.prop("id").asc()),
                rows(row("id", 1L), row("id", 3L), row("id", 4L)),
                "from the consumer corpus"));

        entries.add(read(
                "inline_pattern_property",
                "MATCH (p:Person {city: $p0}) RETURN p.id AS id ORDER BY p.id ASC",
                Map.of("p0", "London"),
                match(node("Person").named("p").withProperty("city", "London"))
                        .returning(PERSON.prop("id").as("id"))
                        .orderBy(PERSON.prop("id").asc()),
                rows(row("id", 1L), row("id", 3L)),
                "inline equality with the value parameterised — the shape hand-writers "
                        + "concatenate"));

        entries.add(read(
                "distinct_projection",
                "MATCH (p:Person) RETURN DISTINCT p.city AS city ORDER BY city ASC",
                Map.of(),
                match(PERSON)
                        .returningDistinct(PERSON.prop("city").as("city"))
                        .orderBy(alias("city").asc()),
                rows(row("city", "Berlin"), row("city", "London"), row("city", "Paris")),
                "RETURN DISTINCT"));

        entries.add(read(
                "aggregate_count",
                "MATCH (p:Person) RETURN count(p) AS n",
                Map.of(),
                match(PERSON).returning(count(PERSON.ref()).as("n")),
                rows(row("n", 4L)),
                "from the consumer corpus, which counts in four places"));

        entries.add(read(
                "aggregate_count_all",
                "MATCH (p:Person) RETURN count(*) AS n",
                Map.of(),
                match(PERSON).returning(countAll().as("n")),
                rows(row("n", 4L)),
                "count(*) counts rows rather than non-null values"));

        entries.add(read(
                "aggregate_count_distinct",
                "MATCH (p:Person) RETURN count(DISTINCT p.city) AS n",
                Map.of(),
                match(PERSON).returning(countDistinct(PERSON.prop("city")).as("n")),
                rows(row("n", 3L)),
                "DISTINCT inside an aggregate"));

        entries.add(read(
                "aggregate_collect",
                "MATCH (p:Person) RETURN collect(p.title) AS names",
                Map.of(),
                match(PERSON).returning(collect(PERSON.prop("title")).as("names")),
                rows(row("names", List.of("Ada", "Bob", "Cy", "Dee"))),
                "from the consumer corpus"));

        entries.add(read(
                "aggregate_collect_distinct",
                "MATCH (p:Person) RETURN collect(DISTINCT p.city) AS cities",
                Map.of(),
                match(PERSON).returning(collectDistinct(PERSON.prop("city")).as("cities")),
                rows(row("cities", List.of("London", "Paris", "Berlin"))),
                "DISTINCT inside an aggregate"));

        entries.add(read(
                "aggregate_numeric_family",
                "MATCH (p:Person) RETURN sum(p.age) AS total, avg(p.age) AS mean, "
                        + "min(p.age) AS youngest, max(p.age) AS oldest",
                Map.of(),
                match(PERSON).returning(
                        sum(PERSON.prop("age")).as("total"),
                        avg(PERSON.prop("age")).as("mean"),
                        min(PERSON.prop("age")).as("youngest"),
                        max(PERSON.prop("age")).as("oldest")),
                rows(row("total", 135L, "mean", 33.75d, "youngest", 29L, "oldest", 41L)),
                "the whole numeric aggregate family in one row"));

        entries.add(read(
                "structural_properties",
                "MATCH (p:Person) WHERE p.id = $p0 RETURN properties(p) AS props",
                Map.of("p0", 1),
                match(PERSON)
                        .where(PERSON.prop("id").eq(1))
                        .returning(PERSON.properties().as("props")),
                rows(row("props", map(
                        "age", 36L,
                        "city", "London",
                        "email", "ada@example.com",
                        "id", 1L,
                        "title", "Ada",
                        "type", "Person"))),
                "one of the four structural functions the DSL steers users to instead of "
                        + "RETURN n"));

        entries.add(read(
                "structural_labels_and_id",
                "MATCH (p:Person) WHERE p.id = $p0 RETURN labels(p) AS labels, id(p) AS internal",
                Map.of("p0", 1),
                match(PERSON)
                        .where(PERSON.prop("id").eq(1))
                        .returning(PERSON.labels().as("labels"), PERSON.id().as("internal")),
                rows(row("labels", List.of("Person"), "internal", 1L)),
                "two more of the structural four"));

        entries.add(read(
                "structural_type",
                "MATCH ()-[r:KNOWS]->() RETURN type(r) AS kind ORDER BY kind ASC",
                Map.of(),
                match(anyNode().to(rel("KNOWS").named("r"), anyNode()))
                        .returning(rel("KNOWS").named("r").type().as("kind"))
                        .orderBy(alias("kind").asc()),
                rows(row("kind", "KNOWS"), row("kind", "KNOWS")),
                "the fourth structural function, over anonymous endpoints"));

        entries.add(read(
                "relationship_outgoing",
                "MATCH (a:Person)-[r:KNOWS]->(b:Person) RETURN a.title AS source, "
                        + "b.title AS target ORDER BY source ASC",
                Map.of(),
                match(A.to(rel("KNOWS").named("r"), B))
                        .returning(A.prop("title").as("source"), B.prop("title").as("target"))
                        .orderBy(alias("source").asc()),
                rows(row("source", "Ada", "target", "Bob"),
                        row("source", "Bob", "target", "Cy")),
                "directed relationship pattern"));

        entries.add(read(
                "relationship_incoming",
                "MATCH (a:Person)<-[r:KNOWS]-(b:Person) RETURN a.title AS target, "
                        + "b.title AS source ORDER BY source ASC",
                Map.of(),
                match(A.from(rel("KNOWS").named("r"), B))
                        .returning(A.prop("title").as("target"), B.prop("title").as("source"))
                        .orderBy(alias("source").asc()),
                rows(row("target", "Bob", "source", "Ada"),
                        row("target", "Cy", "source", "Bob")),
                "the arrow the other way round; direction belongs to the hop, not the "
                        + "relationship"));

        entries.add(read(
                "relationship_undirected",
                "MATCH (a:Person)-[r:KNOWS]-(b:Person) RETURN a.title AS one, b.title AS two "
                        + "ORDER BY one ASC, two ASC",
                Map.of(),
                match(A.related(rel("KNOWS").named("r"), B))
                        .returning(A.prop("title").as("one"), B.prop("title").as("two"))
                        .orderBy(alias("one").asc(), alias("two").asc()),
                rows(row("one", "Ada", "two", "Bob"),
                        row("one", "Bob", "two", "Ada"),
                        row("one", "Bob", "two", "Cy"),
                        row("one", "Cy", "two", "Bob")),
                "undirected traversal, and a two-key ORDER BY"));

        entries.add(read(
                "relationship_bare_arrow",
                "MATCH (a:Person)-->(x) RETURN count(x) AS n",
                Map.of(),
                match(A.to(anyRel(), anyNode().named("x")))
                        .returning(count(anyNode().named("x").ref()).as("n")),
                rows(row("n", 5L)),
                "a relationship with no type, variable or property emits the arrow-only form"));

        entries.add(read(
                "relationship_property_filter",
                "MATCH (a:Person)-[r:KNOWS {since: $p0}]->(b:Person) RETURN b.title AS target",
                Map.of("p0", 2020),
                match(A.to(rel("KNOWS").named("r").withProperty("since", 2020), B))
                        .returning(B.prop("title").as("target")),
                rows(row("target", "Bob")),
                "inline property on a relationship, parameterised"));

        entries.add(read(
                "relationship_property_projection",
                "MATCH (a:Person)-[r:KNOWS]->(b:Person) RETURN r.since AS since "
                        + "ORDER BY since ASC",
                Map.of(),
                match(A.to(rel("KNOWS").named("r"), B))
                        .returning(rel("KNOWS").named("r").prop("since").as("since"))
                        .orderBy(alias("since").asc()),
                rows(row("since", 2020L), row("since", 2021L)),
                "projecting a relationship property"));

        entries.add(read(
                "two_hop_path",
                "MATCH (a:Person)-[:KNOWS]->(b:Person)-[:KNOWS]->(c:Person) "
                        + "RETURN a.title AS first, c.title AS third ORDER BY first ASC",
                Map.of(),
                match(A.to(rel("KNOWS"), B).to(rel("KNOWS"), C))
                        .returning(A.prop("title").as("first"), C.prop("title").as("third"))
                        .orderBy(alias("first").asc()),
                rows(row("first", "Ada", "third", "Cy")),
                "a path of two hops in one pattern"));

        entries.add(read(
                "anonymous_source_node",
                "MATCH (:Person)-[:KNOWS]->(b:Person) RETURN b.title AS name ORDER BY name ASC",
                Map.of(),
                match(node("Person").to(rel("KNOWS"), B))
                        .returning(B.prop("title").as("name"))
                        .orderBy(alias("name").asc()),
                rows(row("name", "Bob"), row("name", "Cy")),
                "an unbound endpoint needs no variable"));

        entries.add(read(
                "comma_joined_patterns",
                "MATCH (p:Person), (c:Company) RETURN count(p) AS people, count(c) AS companies",
                Map.of(),
                match(PERSON, COMPANY)
                        .returning(count(PERSON.ref()).as("people"),
                                count(COMPANY.ref()).as("companies")),
                rows(row("people", 8L, "companies", 8L)),
                "two patterns in one stage, i.e. a cross product"));

        entries.add(read(
                "optional_match",
                "MATCH (p:Person) OPTIONAL MATCH (p)-[:KNOWS]->(f:Person) RETURN p.id AS id, "
                        + "f.title AS friend ORDER BY p.id ASC",
                Map.of(),
                match(PERSON)
                        .optionalMatch(anyNode().named("p")
                                .to(rel("KNOWS"), node("Person").named("f")))
                        .returning(PERSON.prop("id").as("id"),
                                node("Person").named("f").prop("title").as("friend"))
                        .orderBy(PERSON.prop("id").asc()),
                rows(row("id", 1L, "friend", "Bob"),
                        row("id", 2L, "friend", "Cy"),
                        row("id", 3L, "friend", null),
                        row("id", 4L, "friend", null)),
                "null extension is exactly where a hand-writer guesses wrong"));

        entries.add(read(
                "optional_match_with_filter",
                "MATCH (p:Person) OPTIONAL MATCH (p)-[r:WORKS_AT]->(c:Company) "
                        + "WHERE c.city = $p0 RETURN p.id AS id, c.title AS company "
                        + "ORDER BY p.id ASC",
                Map.of("p0", "London"),
                match(PERSON)
                        .optionalMatch(anyNode().named("p")
                                .to(rel("WORKS_AT").named("r"), COMPANY))
                        .where(COMPANY.prop("city").eq("London"))
                        .returning(PERSON.prop("id").as("id"),
                                COMPANY.prop("title").as("company"))
                        .orderBy(PERSON.prop("id").asc()),
                rows(row("id", 1L, "company", "Acme"),
                        row("id", 3L, "company", "Acme")),
                "a WHERE attaches to the stage it follows, optional or not. Note the recorded "
                        + "engine behaviour: the predicate filters the rows the optional stage "
                        + "produced, dropping the null-extended ones, rather than restricting the "
                        + "optional match itself the way Neo4j does — Bob and Dee disappear "
                        + "instead of appearing with a null company. Freezing it here means the "
                        + "day it changes is a red test rather than a silent change of results"));

        entries.add(read(
                "group_by_projection",
                "MATCH (p:Person)-[r:WORKS_AT]->(c:Company) RETURN c.title AS company, "
                        + "count(p) AS headcount ORDER BY company ASC",
                Map.of(),
                match(PERSON.to(rel("WORKS_AT").named("r"), COMPANY))
                        .returning(COMPANY.prop("title").as("company"),
                                count(PERSON.ref()).as("headcount"))
                        .orderBy(alias("company").asc()),
                rows(row("company", "Acme", "headcount", 2L),
                        row("company", "Globex", "headcount", 1L)),
                "implicit grouping by the non-aggregated projection"));

        entries.add(read(
                "order_by_two_keys",
                "MATCH (p:Person) RETURN p.title AS name, p.age AS age ORDER BY p.age DESC, "
                        + "p.title ASC",
                Map.of(),
                match(PERSON)
                        .returning(PERSON.prop("title").as("name"), PERSON.prop("age").as("age"))
                        .orderBy(PERSON.prop("age").desc(), PERSON.prop("title").asc()),
                rows(row("name", "Bob", "age", 41L),
                        row("name", "Ada", "age", 36L),
                        row("name", "Cy", "age", 29L),
                        row("name", "Dee", "age", 29L)),
                "multi-key ordering, mixed directions"));

        entries.add(read(
                "skip_and_limit",
                "MATCH (p:Person) RETURN p.id AS id ORDER BY p.id ASC SKIP $p0 LIMIT $p1",
                params("p0", 1L, "p1", 2L),
                match(PERSON)
                        .returning(PERSON.prop("id").as("id"))
                        .orderBy(PERSON.prop("id").asc())
                        .skip(1)
                        .limit(2),
                rows(row("id", 2L), row("id", 3L)),
                "paging is parameterised end to end, never string-formatted"));

        entries.add(read(
                "limit_only",
                "MATCH (p:Person) RETURN p.id AS id ORDER BY p.id DESC LIMIT $p0",
                params("p0", 2L),
                match(PERSON)
                        .returning(PERSON.prop("id").as("id"))
                        .orderBy(PERSON.prop("id").desc())
                        .limit(2),
                rows(row("id", 4L), row("id", 3L)),
                "LIMIT without SKIP"));

        entries.add(read(
                "empty_result",
                "MATCH (p:Nothing) RETURN p.id AS id ORDER BY p.id ASC",
                Map.of(),
                match(node("Nothing").named("p"))
                        .returning(node("Nothing").named("p").prop("id").as("id"))
                        .orderBy(node("Nothing").named("p").prop("id").asc()),
                rows(),
                "a query that legitimately matches nothing"));

        entries.add(readIn(
                "quoted_identifiers",
                Fixture.ODD_NAMES,
                "MATCH (n:`My Label`) RETURN n.`my key` AS `the key`, n.order AS ordinal "
                        + "ORDER BY ordinal ASC",
                Map.of(),
                match(node("My Label").named("n"))
                        .returning(node("My Label").named("n").prop("my key").as("the key"),
                                node("My Label").named("n").prop("order").as("ordinal"))
                        .orderBy(alias("ordinal").asc()),
                rows(row("the key", "v1", "ordinal", 5L)),
                "a label and a key that need quoting, and a property key (order) that the "
                        + "probe showed does not"));

        entries.add(readIn(
                "reserved_word_label",
                Fixture.ODD_NAMES,
                "MATCH (n:`MATCH`) RETURN n.title AS title",
                Map.of(),
                match(node("MATCH").named("n"))
                        .returning(node("MATCH").named("n").prop("title").as("title")),
                rows(row("title", "reserved")),
                "a reserved word is quoted in label position, where bare use is a syntax "
                        + "error"));

        // -----------------------------------------------------------------------------------
        // Inside the v1 clause set, but the clauses of a later phase. Raw-route only for now.
        // -----------------------------------------------------------------------------------

        entries.add(write(
                "create_node",
                "CREATE (:Person {id: $p0, title: $p1, age: $p2, city: $p3})",
                params("p0", 5, "p1", "Eve", "p2", 50, "p3", "Rome"),
                "MATCH (p:Person) RETURN count(p) AS n",
                rows(row("n", 5L)),
                "half the consumer corpus is writes"));

        entries.add(write(
                "create_relationship",
                "MATCH (a:Person {id: $p0}), (b:Person {id: $p1}) "
                        + "CREATE (a)-[:KNOWS {since: $p2}]->(b)",
                params("p0", 3, "p1", 4, "p2", 2022),
                "MATCH ()-[r:KNOWS]->() RETURN count(r) AS n",
                rows(row("n", 3L)),
                "the consumer corpus writes edges through named endpoints"));

        entries.add(write(
                "set_property",
                "MATCH (p:Person {id: $p0}) SET p.city = $p1",
                params("p0", 2, "p1", "Lyon"),
                "MATCH (p:Person {id: 2}) RETURN p.city AS city",
                rows(row("city", "Lyon")),
                "SET with a parameterised value"));

        entries.add(write(
                "set_property_map",
                "MATCH (p:Person {id: $p0}) SET p += $p1",
                params("p0", 2, "p1", map("city", "Lyon", "age", 42)),
                "MATCH (p:Person {id: 2}) RETURN p.city AS city, p.age AS age",
                rows(row("city", "Lyon", "age", 42L)),
                "+= with a map parameter is the only parameterised way to write a "
                        + "runtime-shaped property set"));

        entries.add(write(
                "remove_property",
                "MATCH (p:Person {id: $p0}) REMOVE p.email",
                params("p0", 1),
                "MATCH (p:Person) WHERE p.email IS NULL RETURN count(p) AS n",
                rows(row("n", 2L)),
                "REMOVE"));

        entries.add(write(
                "detach_delete",
                "MATCH (p:Person {id: $p0}) DETACH DELETE p",
                params("p0", 1),
                "MATCH (p:Person) RETURN count(p) AS n",
                rows(row("n", 3L)),
                "DETACH DELETE"));

        entries.add(write(
                "merge_with_on_match",
                "MERGE (p:Person {id: $p0}) ON CREATE SET p.title = $p1 ON MATCH SET p.title = $p2",
                params("p0", 1, "p1", "created", "p2", "matched"),
                "MATCH (p:Person {id: 1}) RETURN p.title AS title",
                rows(row("title", "matched")),
                "upsert is the idiom users get wrong by hand"));

        entries.add(write(
                "unwind_batch_create",
                "UNWIND $p0 AS row CREATE (:Person {id: row.id, title: row.title})",
                params("p0", List.of(map("id", 6, "title", "Fay"), map("id", 7, "title", "Gil"))),
                "MATCH (p:Person) RETURN count(p) AS n",
                rows(row("n", 6L)),
                "the only way to write N nodes in one statement"));

        entries.add(writeLater(
                "with_aggregate_filter",
                "MATCH (p:Person) WITH p.city AS city, count(p) AS n WHERE n > $p0 "
                        + "RETURN city AS city, n AS n ORDER BY city ASC",
                params("p0", 1),
                rows(row("city", "London", "n", 2L)),
                "WITH is in the v1 clause set in its narrow project-aggregate-filter form"));

        // -----------------------------------------------------------------------------------
        // Deliberately outside v1. These are why the census is a census and not a formality.
        // -----------------------------------------------------------------------------------

        entries.add(outOfV1(
                "return_scalar_literals",
                "RETURN 1 AS i, 2.5 AS d, 'x' AS s, true AS b, null AS n",
                Map.of(),
                rows(row("i", 1L, "d", 2.5d, "s", "x", "b", true, "n", null)),
                null,
                "the consumer corpus does this repeatedly. v1 has no literal expressions and no "
                        + "statement without a MATCH; a literal in a value position would be the "
                        + "one thing the parameter rule forbids"));

        entries.add(outOfV1(
                "return_whole_node",
                "MATCH (p:Person) WHERE p.id = $p0 RETURN p AS node",
                Map.of("p0", 1),
                null,
                1,
                "deliberate omission, not a gap: RETURN n crosses the C ABI as a Rust debug "
                        + "string, so the DSL offers properties()/labels()/id()/type() instead and "
                        + "gains returning(node) when the ABI gains a structured node shape"));

        entries.add(outOfV1(
                "return_map_projection",
                "MATCH (p:Person) WHERE p.id = $p0 RETURN {id: p.id, title: p.title} AS summary",
                Map.of("p0", 1),
                rows(row("summary", map("id", 1L, "title", "Ada"))),
                null,
                "map literals are out of v1: no assembly problem to solve and no value position "
                        + "to protect"));

        entries.add(outOfV1(
                "return_path",
                "MATCH path = (:Person)-[:KNOWS]->(:Person) RETURN path AS p",
                Map.of(),
                null,
                2,
                "named paths are out of v1 along with variable-length patterns"));

        entries.add(outOfV1(
                "union_of_two_reads",
                "MATCH (p:Person) RETURN p.title AS name UNION MATCH (c:Company) "
                        + "RETURN c.title AS name",
                Map.of(),
                null,
                6,
                "UNION is out of v1: the raw string is already the best Java for it"));

        return List.copyOf(entries);
    }

    // ---------------------------------------------------------------------------------------
    // Entry constructors.
    // ---------------------------------------------------------------------------------------

    private static Entry read(
            String name,
            String cypher,
            Map<String, Object> params,
            Statement dsl,
            List<Map<String, Object>> expected,
            String note) {
        return readIn(name, Fixture.PEOPLE, cypher, params, dsl, expected, note);
    }

    private static Entry readIn(
            String name,
            Fixture fixture,
            String cypher,
            Map<String, Object> params,
            Statement dsl,
            List<Map<String, Object>> expected,
            String note) {
        return new Entry(name, fixture, Route.READ, cypher, params, null, expected, null, dsl,
                Expressibility.READ_HALF, note);
    }

    private static Entry write(
            String name,
            String cypher,
            Map<String, Object> params,
            String verify,
            List<Map<String, Object>> expected,
            String note) {
        return new Entry(name, Fixture.PEOPLE, Route.WRITE, cypher, params, verify, expected, null,
                null, Expressibility.V1_LATER, note);
    }

    private static Entry writeLater(
            String name,
            String cypher,
            Map<String, Object> params,
            List<Map<String, Object>> expected,
            String note) {
        return new Entry(name, Fixture.PEOPLE, Route.READ, cypher, params, null, expected, null,
                null, Expressibility.V1_LATER, note);
    }

    private static Entry outOfV1(
            String name,
            String cypher,
            Map<String, Object> params,
            List<Map<String, Object>> expected,
            Integer expectedRowCount,
            String note) {
        return new Entry(name, Fixture.PEOPLE, Route.READ, cypher, params, null, expected,
                expectedRowCount, null, Expressibility.OUT_OF_V1, note);
    }

    // ---------------------------------------------------------------------------------------
    // Small helpers, so the entries above stay readable.
    // ---------------------------------------------------------------------------------------

    @SafeVarargs
    @SuppressWarnings("varargs")
    static List<Map<String, Object>> rows(Map<String, Object>... rows) {
        return List.of(rows);
    }

    static Map<String, Object> row(Object... keysAndValues) {
        return map(keysAndValues);
    }

    static Map<String, Object> params(Object... keysAndValues) {
        return map(keysAndValues);
    }

    /** Order-preserving, null-tolerant map literal — {@code Map.of} rejects null values. */
    static Map<String, Object> map(Object... keysAndValues) {
        if (keysAndValues.length % 2 != 0) {
            throw new IllegalArgumentException(
                    "expected key/value pairs, got " + Arrays.toString(keysAndValues));
        }
        Map<String, Object> map = new LinkedHashMap<>();
        for (int i = 0; i < keysAndValues.length; i += 2) {
            map.put((String) keysAndValues[i], keysAndValues[i + 1]);
        }
        return map;
    }
}
