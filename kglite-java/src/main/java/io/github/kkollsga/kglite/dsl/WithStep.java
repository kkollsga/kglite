package io.github.kkollsga.kglite.dsl;

import java.util.Map;

/**
 * The chain position after a {@code WITH}: filter the projection, match against it, project again,
 * or end the statement.
 *
 * <p><b>A {@code WITH} replaces the scope, it does not add to it.</b> Only the aliases named in the
 * projection exist afterwards — a pattern variable that was not carried through is gone, and
 * referring to it in a later stage is an engine error rather than a Java one. That is the single
 * thing to know about this clause, and it is the reason the DSL asks for an alias on every
 * projection: {@code with(p.ref().as("p"), c.prop("title").as("company"))} says exactly what
 * survives.
 *
 * <p><b>Grouping is implicit, and this is where Java developers are most often surprised.</b>
 * Cypher has no {@code GROUP BY}. When a projection mixes aggregates with non-aggregates, the
 * non-aggregated columns <em>are</em> the grouping key: {@code with(c.prop("title").as("company"),
 * count(p.ref()).as("headcount"))} emits {@code WITH c.title AS company, count(p) AS headcount} and
 * produces one row per distinct {@code company}. Add a second non-aggregated column and the groups
 * get finer; remove them all and the whole result collapses to one row. Nothing in this DSL
 * re-orders or infers that — the grouping is whatever the projection you wrote implies.
 *
 * <p>{@link #where(Condition)} after a projection is the one thing a {@code RETURN} cannot do:
 * filtering on an aggregate. {@code count(p) AS headcount} followed by
 * {@code where(alias("headcount").gt(1))} is Cypher's answer to SQL's {@code HAVING}, and it is
 * why the narrow {@code WITH} earns its place in the clause set at all.
 *
 * <p>The narrowness is deliberate: this DSL models {@code WITH} as project, optionally aggregate,
 * optionally filter. There is no {@code WITH *}, and no {@code ORDER BY}/{@code SKIP}/{@code LIMIT}
 * inside a {@code WITH} — those exist on the final projection, and mid-pipeline paging is a raw
 * route statement.
 */
public interface WithStep extends UpdatingStep {

    /**
     * Filters the projection just made — including on an aggregate, which is what a {@code RETURN}
     * cannot do.
     *
     * <p>Emits: {@code WHERE <predicate>}
     *
     * @param predicate the filter, over the aliases this stage projected
     * @return the next chain step
     */
    WhereStep where(Condition predicate);

    /**
     * Opens a matching stage against the projected rows.
     *
     * <p>Emits: {@code MATCH <pattern>[, <pattern>…]}
     *
     * @param patterns the comma-joined patterns; at least one
     * @return the next chain step
     */
    MatchStep match(Pattern... patterns);

    /**
     * Opens an optional matching stage against the projected rows.
     *
     * <p>Emits: {@code OPTIONAL MATCH <pattern>[, <pattern>…]}
     *
     * @param patterns the comma-joined patterns; at least one
     * @return the next chain step
     */
    MatchStep optionalMatch(Pattern... patterns);

    /**
     * Projects again, narrowing or aggregating what this stage produced.
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
     * Projects again, deduplicating the projected rows.
     *
     * <p>Emits: {@code WITH DISTINCT <expression> AS <alias>[, …]}
     *
     * @param projections the aliased columns; at least one, with distinct aliases
     * @return the next chain step
     */
    WithStep withDistinct(Projection... projections);

    /**
     * Projects the result and ends the statement.
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
