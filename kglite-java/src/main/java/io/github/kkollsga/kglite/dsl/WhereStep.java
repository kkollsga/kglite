package io.github.kkollsga.kglite.dsl;

import java.util.Map;

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
     * Projects the filtered rows mid-pipeline. Read {@link WithStep} first: a {@code WITH}
     * <em>replaces</em> the scope, and grouping is implicit in what the projection does not
     * aggregate.
     *
     * <p>Emits: {@code WITH <expression> AS <alias>[, …]}
     *
     * @param projections the aliased columns; at least one, with distinct aliases
     * @return the next chain step
     */
    WithStep with(Projection... projections);

    /**
     * Appends a whole clause this DSL does not model, verbatim — the clause-level escape hatch.
     *
     * <p>For a clause v1 has no builder for and that belongs in the middle of a pipeline: a
     * mid-pipeline {@code ORDER BY}/{@code LIMIT} on a {@code WITH}, an {@code UNWIND} of a list
     * that is not a batch write, a procedure call. The step this hands back offers no
     * {@code where()}: this DSL cannot know what the fragment bound, so the filtering belongs
     * inside the fragment.
     *
     * <p><b>Read {@link Raw} before using it.</b> The text is emitted as given, so this is one of
     * the two places where the injection property is the caller's responsibility.
     *
     * <p>Emits: the fragment, verbatim
     *
     * @param fragment the Cypher clause text
     * @return the next chain step
     * @throws IllegalArgumentException if the fragment is empty or refers to the emitter's own
     *     {@code $p<digits>} parameter namespace
     */
    WhereStep rawClause(String fragment);

    /**
     * Appends a whole clause this DSL does not model, with the values it refers to travelling as
     * parameters.
     *
     * <p>Emits: the fragment, verbatim
     *
     * @param fragment the Cypher clause text
     * @param params the parameters the fragment refers to, by the names it uses
     * @return the next chain step
     * @throws IllegalArgumentException if the fragment is empty, refers to {@code $p<digits>}, or
     *     declares a parameter that is not a Cypher identifier, claims the emitter's namespace, or
     *     is never referred to by the fragment
     */
    WhereStep rawClause(String fragment, Map<String, Object> params);

    /**
     * Projects the filtered rows mid-pipeline, deduplicating them.
     *
     * <p>Emits: {@code WITH DISTINCT <expression> AS <alias>[, …]}
     *
     * @param projections the aliased columns; at least one, with distinct aliases
     * @return the next chain step
     */
    WithStep withDistinct(Projection... projections);

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
