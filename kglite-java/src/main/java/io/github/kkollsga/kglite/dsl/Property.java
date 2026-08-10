package io.github.kkollsga.kglite.dsl;

/**
 * A reference to one property of a bound pattern element — {@code n.title} — as opposed to any
 * other expression.
 *
 * <p>It has its own type because three clauses accept a property and nothing else: {@code SET}
 * assigns to one, {@code REMOVE} removes one, and neither has a meaning for {@code count(n)} or a
 * {@code RETURN} alias. Typing the parameter is what makes {@code remove(count(p.ref()))} a
 * compile error instead of a runtime check nobody wrote.
 *
 * <p>Produced by {@link Node#prop(String)} and {@link Rel#prop(String)}; it is an {@link Expr}, so
 * everything an expression can do — comparisons, projection, ordering — is unchanged.
 */
public sealed interface Property extends Expr permits Ast.PropertyRef {

    /**
     * Assigns a value to this property.
     *
     * <p>Emits: {@code <variable>.<key> = $p<n>}
     *
     * @param value the value to store; emitted as a parameter, never as text
     * @return the assignment, for {@code set}, {@code onCreateSet} or {@code onMatchSet}
     * @throws IllegalArgumentException if a query element is passed where a value belongs
     */
    default Assignment to(Object value) {
        return new Ast.PropertyAssignment(this, Values.check(value));
    }
}
