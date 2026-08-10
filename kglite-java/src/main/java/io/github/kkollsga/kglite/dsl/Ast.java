package io.github.kkollsga.kglite.dsl;

import java.util.List;
import java.util.Map;

/**
 * The immutable AST nodes behind the public sealed interfaces.
 *
 * <p>Package-private on purpose. Users build these through {@link Cypher} and the fluent chain and
 * never name them, so they stay out of the published surface; the renderer, which lives in this
 * package, switches over them exhaustively.
 */
final class Ast {

    private Ast() {}

    /** {@code <variable>.<key>} */
    record PropertyRef(Ident variable, Ident key) implements Property {}

    /** A bare pattern variable, as passed to {@code count()}, {@code properties()} and friends. */
    record VarRef(Ident variable) implements Variable {}

    /** A reference to a {@code RETURN} alias, legal in {@code ORDER BY}. */
    record AliasRef(Ident alias) implements Expr {}

    /**
     * Caller-written Cypher, emitted verbatim in an expression or predicate position.
     *
     * @param fragment the text, checked by {@link RawFragment}
     * @param params the parameters it refers to, by their own names
     */
    record RawExpr(String fragment, Map<String, Object> params) implements Raw {}

    /**
     * Caller-written Cypher, emitted verbatim as a whole clause of the reading pipeline.
     *
     * @param fragment the text, checked by {@link RawFragment}
     * @param params the parameters it refers to, by their own names
     */
    record RawStage(String fragment, Map<String, Object> params) implements ReadStage {}

    /**
     * A function call this DSL knows how to emit.
     *
     * @param name the Cypher function name, lower-case as emitted
     * @param distinct whether {@code DISTINCT} precedes the argument
     * @param star whether the argument is {@code *} (only {@code count(*)})
     * @param argument the single argument, or {@code null} when {@code star} is set
     */
    record FunctionExpr(String name, boolean distinct, boolean star, Expr argument)
            implements Expr {}

    /**
     * A binary predicate against a parameterised value. Covers the comparison operators and the
     * string/list operators, which differ only in the token between the operands.
     *
     * @param left the left-hand expression
     * @param operator the emitted operator token
     * @param value the right-hand value, which becomes a parameter
     */
    record Comparison(Expr left, String operator, Object value) implements Condition {}

    /** {@code <expression> IS [NOT] NULL} */
    record NullCheck(Expr operand, boolean negated) implements Condition {}

    /** {@code NOT (<predicate>)} */
    record Not(Condition operand) implements Condition {}

    /** {@code <predicate> AND <predicate> [AND ...]} */
    record And(List<Condition> operands) implements Condition {}

    /** {@code <predicate> OR <predicate> [OR ...]} */
    record Or(List<Condition> operands) implements Condition {}

    /**
     * One stage of the reading pipeline, before the terminal {@code RETURN} or updating clause.
     *
     * <p>Sealed so the renderer's switch is exhaustive: a stage kind the renderer was never taught
     * about is a compile error.
     */
    sealed interface ReadStage permits MatchStage, WithStage, RawStage {}

    /**
     * One reading stage: a {@code MATCH} or {@code OPTIONAL MATCH} with its optional
     * {@code WHERE}.
     *
     * @param optional whether the stage emits {@code OPTIONAL MATCH}
     * @param patterns the comma-joined patterns
     * @param where the stage's predicate, or {@code null}
     */
    record MatchStage(boolean optional, List<Pattern> patterns, Condition where)
            implements ReadStage {}

    /**
     * One {@code WITH} stage: project (aggregating or not), then optionally filter.
     *
     * @param distinct whether {@code DISTINCT} follows the keyword
     * @param projections the aliased columns the following stages see
     * @param where the stage's predicate, or {@code null}
     */
    record WithStage(boolean distinct, List<Projection> projections, Condition where)
            implements ReadStage {}

    /**
     * A copy of a stage carrying a predicate — how {@code where(...)} attaches to whichever stage
     * was opened last, matching or projecting.
     *
     * @param stage the stage the predicate belongs to
     * @param predicate the predicate
     * @return the stage with its {@code WHERE}
     */
    static ReadStage filtered(ReadStage stage, Condition predicate) {
        return switch (stage) {
            case MatchStage match ->
                    new MatchStage(match.optional(), match.patterns(), predicate);
            case WithStage with -> new WithStage(with.distinct(), with.projections(), predicate);
            // Unreachable through the public step types: a raw clause hands back a step that
            // offers no where(), because this DSL cannot know what the fragment bound or whether
            // the dialect even accepts a WHERE after it. Kept so a future step-interface change
            // fails loudly rather than emitting a predicate attached to arbitrary text.
            case RawStage raw -> throw new IllegalStateException(
                    "a WHERE cannot be attached to the raw clause \"" + raw.fragment()
                            + "\": put the filtering inside the fragment");
        };
    }

    // ---- the writing half ---------------------------------------------------------------

    /** {@code <variable>.<key> = $p<n>} */
    record PropertyAssignment(Property target, Object value) implements Assignment {}

    /** {@code <variable> += $p<n>} — merge a map of properties into an element. */
    record MapAssignment(Ident variable, Object values) implements Assignment {}

    /**
     * {@code UNWIND $p<n> AS <variable>} — the batch opener.
     *
     * @param rows the list, which travels as one parameter
     * @param variable the loop variable a pattern reads fields from
     */
    record Unwind(Object rows, Ident variable) {}

    /** One updating clause. Sealed so the renderer's switch is exhaustive. */
    sealed interface WriteClause permits Create, MergeClause, SetClause, RemoveClause, DeleteClause {}

    /** {@code CREATE <pattern>[, …]} */
    record Create(List<Pattern> patterns) implements WriteClause {}

    /**
     * {@code MERGE <pattern> [ON CREATE SET …] [ON MATCH SET …]}.
     *
     * @param pattern the pattern to match or create
     * @param onCreate assignments applied only on the creating branch; may be empty
     * @param onMatch assignments applied only on the matching branch; may be empty
     */
    record MergeClause(Pattern pattern, List<Assignment> onCreate, List<Assignment> onMatch)
            implements WriteClause {}

    /** {@code SET <assignment>[, …]} */
    record SetClause(List<Assignment> assignments) implements WriteClause {}

    /** {@code REMOVE <property>[, …]} */
    record RemoveClause(List<Property> properties) implements WriteClause {}

    /** {@code [DETACH] DELETE <variable>[, …]} */
    record DeleteClause(List<Variable> elements, boolean detach) implements WriteClause {}

    /** A {@code MERGE} with no conditional assignments yet. */
    static MergeClause merge(Pattern pattern) {
        if (pattern == null) {
            throw new IllegalArgumentException("merge() needs a pattern");
        }
        return new MergeClause(pattern, List.of(), List.of());
    }

    /** Validates a comma-joined pattern list, shared by every clause that takes one. */
    static List<Pattern> patterns(Pattern... patterns) {
        return checked(patterns, "pattern", "a clause needs at least one pattern");
    }

    /** Validates a varargs clause argument list: non-empty, no nulls. */
    static <T> List<T> checked(T[] items, String what, String emptyMessage) {
        if (items == null || items.length == 0) {
            throw new IllegalArgumentException(emptyMessage);
        }
        for (T item : items) {
            if (item == null) {
                throw new IllegalArgumentException("a clause may not contain a null " + what);
            }
        }
        return List.of(items);
    }
}
