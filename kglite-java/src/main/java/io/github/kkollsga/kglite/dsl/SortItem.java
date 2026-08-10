package io.github.kkollsga.kglite.dsl;

/**
 * One key of an {@code ORDER BY} clause.
 *
 * <p>Built with {@link Expr#asc()} or {@link Expr#desc()}. The direction is always emitted, even
 * for the ascending default, so the emitted text is one form rather than two.
 */
public final class SortItem {

    private final Expr expression;
    private final boolean descending;

    SortItem(Expr expression, boolean descending) {
        this.expression = expression;
        this.descending = descending;
    }

    Expr expression() {
        return expression;
    }

    boolean descending() {
        return descending;
    }
}
