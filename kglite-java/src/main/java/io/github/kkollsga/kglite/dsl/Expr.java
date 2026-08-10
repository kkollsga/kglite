package io.github.kkollsga.kglite.dsl;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

/**
 * An expression: a property reference, a pattern variable, a {@code RETURN} alias, or a call to
 * one of the aggregate and structural functions this DSL emits.
 *
 * <p>The interface is sealed and its implementations are package-private, so the renderer's
 * {@code switch} is exhaustive without a default branch: an expression node nobody taught the
 * renderer about is a compile error rather than a runtime surprise.
 *
 * <p>The comparison builders below all take {@code Object value}. That is the injection-relevant
 * signature in the whole DSL: there is no overload anywhere that accepts a Cypher fragment in a
 * value position, so a value cannot become syntax. Every value passed here is emitted as a
 * {@code $p<n>} parameter and travels in {@link Statement#params()}.
 */
public sealed interface Expr
        permits Property, Variable, Ast.AliasRef, Ast.FunctionExpr {

    /**
     * Equality against a value.
     *
     * <p>Emits: {@code <expression> = $p<n>}
     *
     * @param value the value to compare against; emitted as a parameter, never as text
     * @return the predicate
     * @throws IllegalArgumentException if a DSL node is passed where a value belongs
     */
    default Condition eq(Object value) {
        return new Ast.Comparison(this, "=", Values.check(value));
    }

    /**
     * Inequality against a value.
     *
     * <p>Emits: {@code <expression> <> $p<n>}
     *
     * @param value the value to compare against; emitted as a parameter
     * @return the predicate
     * @throws IllegalArgumentException if a DSL node is passed where a value belongs
     */
    default Condition ne(Object value) {
        return new Ast.Comparison(this, "<>", Values.check(value));
    }

    /**
     * Strictly-less-than against a value.
     *
     * <p>Emits: {@code <expression> < $p<n>}
     *
     * @param value the value to compare against; emitted as a parameter
     * @return the predicate
     * @throws IllegalArgumentException if a DSL node is passed where a value belongs
     */
    default Condition lt(Object value) {
        return new Ast.Comparison(this, "<", Values.check(value));
    }

    /**
     * Less-than-or-equal against a value.
     *
     * <p>Emits: {@code <expression> <= $p<n>}
     *
     * @param value the value to compare against; emitted as a parameter
     * @return the predicate
     * @throws IllegalArgumentException if a DSL node is passed where a value belongs
     */
    default Condition le(Object value) {
        return new Ast.Comparison(this, "<=", Values.check(value));
    }

    /**
     * Strictly-greater-than against a value.
     *
     * <p>Emits: {@code <expression> > $p<n>}
     *
     * @param value the value to compare against; emitted as a parameter
     * @return the predicate
     * @throws IllegalArgumentException if a DSL node is passed where a value belongs
     */
    default Condition gt(Object value) {
        return new Ast.Comparison(this, ">", Values.check(value));
    }

    /**
     * Greater-than-or-equal against a value.
     *
     * <p>Emits: {@code <expression> >= $p<n>}
     *
     * @param value the value to compare against; emitted as a parameter
     * @return the predicate
     * @throws IllegalArgumentException if a DSL node is passed where a value belongs
     */
    default Condition ge(Object value) {
        return new Ast.Comparison(this, ">=", Values.check(value));
    }

    /**
     * Membership in a list, emitted as a single list-valued parameter.
     *
     * <p>Cypher's {@code IN} is the reason this method exists rather than a generated {@code OR}
     * chain: a chain of equalities hits the dialect's expression-nesting cap, and one parameter
     * carrying the whole list does not.
     *
     * <p>Emits: {@code <expression> IN $p<n>}
     *
     * @param values the candidate values; the collection itself becomes one parameter
     * @return the predicate
     * @throws IllegalArgumentException if a DSL node is passed where a value belongs
     */
    default Condition in(Collection<?> values) {
        if (values == null) {
            throw new IllegalArgumentException("IN requires a collection of values, not null");
        }
        List<Object> copy = new ArrayList<>(values.size());
        for (Object value : values) {
            copy.add(Values.check(value));
        }
        return new Ast.Comparison(this, "IN", List.copyOf(copy));
    }

    /**
     * Prefix match.
     *
     * <p>Emits: {@code <expression> STARTS WITH $p<n>}
     *
     * @param prefix the prefix; emitted as a parameter
     * @return the predicate
     */
    default Condition startsWith(String prefix) {
        return new Ast.Comparison(this, "STARTS WITH", Values.check(prefix));
    }

    /**
     * Suffix match.
     *
     * <p>Emits: {@code <expression> ENDS WITH $p<n>}
     *
     * @param suffix the suffix; emitted as a parameter
     * @return the predicate
     */
    default Condition endsWith(String suffix) {
        return new Ast.Comparison(this, "ENDS WITH", Values.check(suffix));
    }

    /**
     * Substring match.
     *
     * <p>Emits: {@code <expression> CONTAINS $p<n>}
     *
     * @param substring the substring; emitted as a parameter
     * @return the predicate
     */
    default Condition contains(String substring) {
        return new Ast.Comparison(this, "CONTAINS", Values.check(substring));
    }

    /**
     * Regular-expression match. The pattern is a value, so it travels as a parameter and cannot
     * reach the query text.
     *
     * <p>Emits: {@code <expression> =~ $p<n>}
     *
     * @param regex the regular expression; emitted as a parameter
     * @return the predicate
     */
    default Condition matches(String regex) {
        return new Ast.Comparison(this, "=~", Values.check(regex));
    }

    /**
     * Null test.
     *
     * <p>Emits: {@code <expression> IS NULL}
     *
     * @return the predicate
     */
    default Condition isNull() {
        return new Ast.NullCheck(this, false);
    }

    /**
     * Non-null test.
     *
     * <p>Emits: {@code <expression> IS NOT NULL}
     *
     * @return the predicate
     */
    default Condition isNotNull() {
        return new Ast.NullCheck(this, true);
    }

    /**
     * Names this expression in the result row. Every projection must be aliased: the alias is the
     * result-row key, and requiring it is what lets duplicate keys be rejected while the statement
     * is still being built.
     *
     * <p>Emits: {@code <expression> AS <alias>}
     *
     * @param alias the result-row key
     * @return the projection
     * @throws IllegalArgumentException if the alias is empty or contains a backtick
     */
    default Projection as(String alias) {
        return new Projection(this, Ident.alias(alias));
    }

    /**
     * Ascending sort on this expression.
     *
     * <p>Emits: {@code <expression> ASC}
     *
     * @return the sort item
     */
    default SortItem asc() {
        return new SortItem(this, false);
    }

    /**
     * Descending sort on this expression.
     *
     * <p>Emits: {@code <expression> DESC}
     *
     * @return the sort item
     */
    default SortItem desc() {
        return new SortItem(this, true);
    }
}
