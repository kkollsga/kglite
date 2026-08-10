package io.github.kkollsga.kglite.dsl;

/**
 * A batch of rows being written in one statement: {@code UNWIND $p0 AS row …}.
 *
 * <p>This is the only way to write N elements with one statement rather than N, and it is the
 * shape a hand-writer most often builds by string-concatenating a loop's worth of {@code CREATE}s
 * — so it is also the shape where doing it wrong is most expensive. Here the whole list is one
 * parameter and the per-row values are read out of it by the engine:
 *
 * <pre>{@code
 * UnwindStep rows = unwind(List.of(Map.of("id", 6, "title", "Fay"),
 *                                  Map.of("id", 7, "title", "Gil")));
 * WriteStatement stmt = rows.create(node("Person")
 *         .withPropertyFrom("id", rows.field("id"))
 *         .withPropertyFrom("title", rows.field("title")));
 * // UNWIND $p0 AS row CREATE (:Person {id: row.id, title: row.title})
 * }</pre>
 *
 * <p>The loop variable is always named {@code row}; {@link #field(String)} is how a pattern refers
 * to one of its keys, and nothing else in the DSL can produce a reference to it, so a
 * {@code row.x} cannot appear in a statement that has no {@code UNWIND}.
 *
 * <p>v1 offers {@code CREATE} and {@code MERGE} over the rows and nothing else — no {@code MATCH}
 * after the {@code UNWIND}, so a batch that attaches relationships to pre-existing nodes by id is
 * still a raw-route statement.
 */
public final class UnwindStep {

    private final Ast.Unwind unwind;

    UnwindStep(Ast.Unwind unwind) {
        this.unwind = unwind;
    }

    /**
     * A reference to one key of the current row.
     *
     * <p>Emits: {@code row.<key>}
     *
     * @param key the key to read from each row
     * @return the property reference, for a pattern's
     *     {@link Node#withPropertyFrom(String, Expr)}
     * @throws IllegalArgumentException if the key is not representable
     */
    public Property field(String key) {
        return new Ast.PropertyRef(unwind.variable(), Ident.propertyKey(key));
    }

    /**
     * Creates the given patterns once per row.
     *
     * <p>Emits: {@code CREATE <pattern>[, <pattern>…]}
     *
     * @param patterns the comma-joined patterns; at least one
     * @return the finished statement
     */
    public WriteStatement create(Pattern... patterns) {
        return WriteQuery.opening(unwind, new Ast.Create(Ast.patterns(patterns)));
    }

    /**
     * Matches or creates the pattern once per row — the batch upsert.
     *
     * <p>Emits: {@code MERGE <pattern>}
     *
     * @param pattern the pattern to match or create
     * @return the next chain step, which is already a complete statement
     */
    public MergeStep merge(Pattern pattern) {
        return WriteQuery.opening(unwind, Ast.merge(pattern));
    }
}
