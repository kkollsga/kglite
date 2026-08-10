package io.github.kkollsga.kglite.dsl;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * A relationship pattern: the {@code [variable:TYPE {key: $p0}]} between two nodes. Direction is
 * not part of the relationship — it is chosen by the {@link Node#to(Rel, Node)},
 * {@link Node#from(Rel, Node)} and {@link Node#related(Rel, Node)} that consume it, so one
 * relationship value can be reused in either direction.
 *
 * <p>Immutable; every builder method returns a new instance.
 *
 * <p>Type alternation ({@code -[:A|B]->}) is not offered: it parses in this dialect but was probed
 * to return incomplete results, so emitting it would make the DSL a route to wrong answers.
 */
public final class Rel {

    private final Ident type;
    private final Ident variable;
    private final Map<Ident, Object> properties;

    Rel(Ident type, Ident variable, Map<Ident, Object> properties) {
        this.type = type;
        this.variable = variable;
        this.properties = Collections.unmodifiableMap(new LinkedHashMap<>(properties));
    }

    /**
     * Binds this relationship pattern to a variable.
     *
     * <p>Emits: {@code [<variable>:<TYPE>]}
     *
     * @param name the variable name, validated as a pattern variable
     * @return a new relationship pattern carrying the variable
     * @throws IllegalArgumentException if the name is not representable as a variable
     */
    public Rel named(String name) {
        return new Rel(type, Ident.variable(name), properties);
    }

    /**
     * Adds an inline property equality to the relationship pattern; the value becomes a parameter.
     *
     * <p>Emits: {@code [<variable>:<TYPE> {<key>: $p<n>}]}
     *
     * @param key the property key
     * @param value the value to match; emitted as a parameter, never as text
     * @return a new relationship pattern carrying the property
     * @throws IllegalArgumentException if the key is not representable, is already present, or the
     *     value is a query element
     */
    public Rel withProperty(String key, Object value) {
        Ident ident = Ident.propertyKey(key);
        if (properties.containsKey(ident)) {
            throw new IllegalArgumentException(
                    "inline pattern property \"" + key + "\" is already set on this relationship");
        }
        Map<Ident, Object> next = new LinkedHashMap<>(properties);
        next.put(ident, Values.check(value));
        return new Rel(type, variable, next);
    }

    /**
     * A reference to one of this relationship's properties.
     *
     * <p>Emits: {@code <variable>.<key>}
     *
     * @param key the property key
     * @return the property reference
     * @throws IllegalStateException if this relationship has no variable
     * @throws IllegalArgumentException if the key is not representable
     */
    public Expr prop(String key) {
        return new Ast.PropertyRef(requireVariable("prop"), Ident.propertyKey(key));
    }

    /**
     * This relationship's type, as a string value.
     *
     * <p>Emits: {@code type(<variable>)}
     *
     * @return the function call
     * @throws IllegalStateException if this relationship has no variable
     */
    public Expr type() {
        return new Ast.FunctionExpr("type", false, false, ref());
    }

    /**
     * The bare variable, for the aggregate functions that take a relationship.
     *
     * <p>Emits: {@code <variable>}
     *
     * @return the variable reference
     * @throws IllegalStateException if this relationship has no variable
     */
    public Expr ref() {
        return new Ast.VarRef(requireVariable("ref"));
    }

    private Ident requireVariable(String method) {
        if (variable == null) {
            throw new IllegalStateException(
                    method + "() needs a variable: call named(\"…\") on the relationship first");
        }
        return variable;
    }

    Ident typeIdent() {
        return type;
    }

    Ident variable() {
        return variable;
    }

    Map<Ident, Object> inlineProperties() {
        return properties;
    }

    boolean isBare() {
        return type == null && variable == null && properties.isEmpty();
    }
}
