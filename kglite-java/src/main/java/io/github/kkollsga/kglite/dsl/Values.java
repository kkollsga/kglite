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
 *
 * <p>The test is "does this object come from this package", not a list of the types that do. A list
 * has to be revisited every time the DSL grows a type, and the day it is not is the day the new
 * type serialises silently as JSON — {@code Assignment} and {@code UnwindStep} arrived exactly
 * that way. Nothing in this package is data, so the package is the right question to ask.
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
        if (value != null && Values.class.getPackage().equals(value.getClass().getPackage())) {
            throw new IllegalArgumentException(
                    "a value position received a query element ("
                            + value.getClass().getSimpleName()
                            + "). Values in this DSL are always parameters, so only data belongs "
                            + "here; comparing two expressions to each other is not part of v1.");
        }
        return value;
    }
}
