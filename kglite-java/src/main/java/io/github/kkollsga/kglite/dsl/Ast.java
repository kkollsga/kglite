package io.github.kkollsga.kglite.dsl;

import java.util.List;

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
    record PropertyRef(Ident variable, Ident key) implements Expr {}

    /** A bare pattern variable, as passed to {@code count()}, {@code properties()} and friends. */
    record VarRef(Ident variable) implements Expr {}

    /** A reference to a {@code RETURN} alias, legal in {@code ORDER BY}. */
    record AliasRef(Ident alias) implements Expr {}

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
     * One reading stage: a {@code MATCH} or {@code OPTIONAL MATCH} with its optional
     * {@code WHERE}.
     *
     * @param optional whether the stage emits {@code OPTIONAL MATCH}
     * @param patterns the comma-joined patterns
     * @param where the stage's predicate, or {@code null}
     */
    record MatchStage(boolean optional, List<Pattern> patterns, Condition where) {}
}
