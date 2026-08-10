package io.github.kkollsga.kglite.dsl;

/**
 * The updating clauses, at every chain position that can carry one.
 *
 * <p>{@link MatchStep} and {@link WhereStep} both extend this, so a write reads the same after a
 * bare {@code MATCH} and after a {@code WHERE} without either interface restating it.
 *
 * <p>One updating clause per statement, deliberately. Cypher allows a chain of them
 * ({@code MATCH … SET … REMOVE …}) and v1 does not: every clause here returns a finished
 * statement, so the surface stays the shapes the corpus actually contains, and a caller who needs
 * a longer chain has {@link Statement#cypher()} and the raw route. {@code SET} takes varargs, which
 * covers the common reason to want two.
 */
public interface UpdatingStep {

    /**
     * Creates the given patterns. A pattern element that names a variable bound by an earlier
     * stage refers to it; one that does not is created.
     *
     * <p>Emits: {@code CREATE <pattern>[, <pattern>…]}
     *
     * @param patterns the comma-joined patterns; at least one
     * @return the finished statement
     */
    WriteStatement create(Pattern... patterns);

    /**
     * Matches the pattern or creates it — the upsert.
     *
     * <p>Emits: {@code MERGE <pattern>}
     *
     * @param pattern the pattern to match or create
     * @return the next chain step, which is already a complete statement
     */
    MergeStep merge(Pattern pattern);

    /**
     * Assigns properties.
     *
     * <p>Emits: {@code SET <assignment>[, <assignment>…]}
     *
     * @param assignments the assignments, from {@link Property#to(Object)} or
     *     {@link Node#plusProperties(java.util.Map)}; at least one
     * @return the finished statement
     */
    WriteStatement set(Assignment... assignments);

    /**
     * Removes properties.
     *
     * <p>Emits: {@code REMOVE <variable>.<key>[, …]}
     *
     * @param properties the properties to remove; at least one
     * @return the finished statement
     */
    WriteStatement remove(Property... properties);

    /**
     * Deletes matched elements. Deleting a node that still has relationships is an engine error;
     * {@link #detachDelete(Variable...)} is the form that removes them with it.
     *
     * <p>Emits: {@code DELETE <variable>[, …]}
     *
     * @param elements the bound nodes or relationships to delete; at least one
     * @return the finished statement
     */
    WriteStatement delete(Variable... elements);

    /**
     * Deletes matched nodes together with every relationship attached to them.
     *
     * <p>Emits: {@code DETACH DELETE <variable>[, …]}
     *
     * @param elements the bound nodes to delete; at least one
     * @return the finished statement
     */
    WriteStatement detachDelete(Variable... elements);
}
