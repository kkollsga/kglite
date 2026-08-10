package io.github.kkollsga.kglite.dsl;

import java.util.List;

/**
 * A boolean predicate for a {@code WHERE} clause.
 *
 * <p>Sealed with package-private implementations, for the same reason as {@link Expr}: the
 * renderer switches exhaustively over the permitted set, so a new predicate shape cannot be
 * forgotten.
 *
 * <p>Emission is fully parenthesised where it needs to be and nowhere else: a composite operand of
 * {@code AND}/{@code OR} is wrapped, a simple one is not, and {@code NOT} always wraps its
 * operand. The result is unambiguous without depending on the dialect's precedence table.
 */
public sealed interface Condition
        permits Raw, Ast.Comparison, Ast.NullCheck, Ast.Not, Ast.And, Ast.Or {

    /**
     * Conjunction with another predicate.
     *
     * <p>Emits: {@code <this> AND <other>}
     *
     * @param other the right-hand predicate
     * @return the conjunction
     */
    default Condition and(Condition other) {
        return new Ast.And(List.of(this, requireCondition(other)));
    }

    /**
     * Disjunction with another predicate.
     *
     * <p>Emits: {@code <this> OR <other>}
     *
     * @param other the right-hand predicate
     * @return the disjunction
     */
    default Condition or(Condition other) {
        return new Ast.Or(List.of(this, requireCondition(other)));
    }

    private static Condition requireCondition(Condition other) {
        if (other == null) {
            throw new IllegalArgumentException("predicate must not be null");
        }
        return other;
    }
}
