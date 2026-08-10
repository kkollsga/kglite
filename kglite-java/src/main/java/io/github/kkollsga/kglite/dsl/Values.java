package io.github.kkollsga.kglite.dsl;

/**
 * The one guard on the value side of the API.
 *
 * <p>Every value-position parameter in this DSL is declared {@code Object}, which is what makes
 * "a value can never become syntax" a property of the signatures rather than of the renderer.
 * The cost of {@code Object} is that a caller can pass a DSL node by mistake — {@code
 * p.prop("a").eq(q.prop("b"))} reads plausibly — and it would silently be serialised as a
 * parameter value instead of comparing two properties. This rejects that at build time and says
 * what v1 does not do, rather than emitting a query that returns wrong rows.
 */
final class Values {

    private Values() {}

    /**
     * Checks that an object is a value rather than a piece of the query.
     *
     * @param value the caller-supplied value
     * @return the same value
     */
    static Object check(Object value) {
        if (value instanceof Expr
                || value instanceof Condition
                || value instanceof Pattern
                || value instanceof Rel
                || value instanceof Projection
                || value instanceof SortItem
                || value instanceof Statement) {
            throw new IllegalArgumentException(
                    "a value position received a query element ("
                            + value.getClass().getSimpleName()
                            + "). Values in this DSL are always parameters, so only data belongs "
                            + "here; comparing two expressions to each other is not part of v1.");
        }
        return value;
    }
}
