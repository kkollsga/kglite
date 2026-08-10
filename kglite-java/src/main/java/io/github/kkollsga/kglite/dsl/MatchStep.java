package io.github.kkollsga.kglite.dsl;

/**
 * The chain position after a {@code MATCH}: another matching stage, a predicate for this one, or
 * the projection that ends the statement.
 *
 * <p>The step interfaces are the grammar made static. An out-of-order call — {@code ORDER BY}
 * before {@code RETURN}, a second {@code WHERE} on one stage — is not offered here, so it is a
 * compile error rather than a runtime one, and the completion popup only lists legal
 * continuations.
 */
public interface MatchStep extends UpdatingStep {

    /**
     * Opens another matching stage.
     *
     * <p>Emits: {@code MATCH <pattern>[, <pattern>…]}
     *
     * @param patterns the comma-joined patterns; at least one
     * @return the next chain step
     */
    MatchStep match(Pattern... patterns);

    /**
     * Opens an optional matching stage, which extends rows with nulls instead of dropping them.
     *
     * <p>Emits: {@code OPTIONAL MATCH <pattern>[, <pattern>…]}
     *
     * @param patterns the comma-joined patterns; at least one
     * @return the next chain step
     */
    MatchStep optionalMatch(Pattern... patterns);

    /**
     * Filters the stage just opened.
     *
     * <p>Emits: {@code WHERE <predicate>}
     *
     * @param predicate the filter
     * @return the next chain step
     */
    WhereStep where(Condition predicate);

    /**
     * Projects the result and ends the reading half of the statement.
     *
     * <p>Emits: {@code RETURN <expression> AS <alias>[, …]}
     *
     * @param projections the aliased columns; at least one, with distinct aliases
     * @return the next chain step
     */
    ReturnStep returning(Projection... projections);

    /**
     * Projects the result, deduplicating rows.
     *
     * <p>Emits: {@code RETURN DISTINCT <expression> AS <alias>[, …]}
     *
     * @param projections the aliased columns; at least one, with distinct aliases
     * @return the next chain step
     */
    ReturnStep returningDistinct(Projection... projections);
}
