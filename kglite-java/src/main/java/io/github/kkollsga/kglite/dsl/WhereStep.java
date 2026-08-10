package io.github.kkollsga.kglite.dsl;

/**
 * The chain position after a {@code WHERE}: a further matching stage, or the projection.
 *
 * <p>There is no second {@code where} here on purpose — one stage carries one predicate, and
 * conjunction is spelled {@link Condition#and(Condition)}, which keeps the emitted clause a single
 * unambiguous expression.
 */
public interface WhereStep extends UpdatingStep {

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
     * Opens an optional matching stage.
     *
     * <p>Emits: {@code OPTIONAL MATCH <pattern>[, <pattern>…]}
     *
     * @param patterns the comma-joined patterns; at least one
     * @return the next chain step
     */
    MatchStep optionalMatch(Pattern... patterns);

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
