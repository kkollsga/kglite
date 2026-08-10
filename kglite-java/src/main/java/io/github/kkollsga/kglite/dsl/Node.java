package io.github.kkollsga.kglite.dsl;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * A node pattern: {@code (variable:Label {key: $p0})}, with every part optional except the
 * parentheses.
 *
 * <p>Immutable. Every builder method returns a new instance, so a {@code Node} held in a field can
 * be shared between statements and across threads.
 *
 * <p>There is deliberately no way to project a whole node. {@code RETURN n} crosses the C ABI as a
 * Rust debug string rather than as structured data, so instead of documenting a footgun this DSL
 * omits it: use {@link #properties()}, {@link #labels()}, {@link #id()} or {@link #prop(String)},
 * which return real values. When the ABI grows a structured node shape the projection can be added
 * as a pure addition.
 */
public final class Node implements Pattern {

    private final Ident label;
    private final Ident variable;
    private final Map<Ident, Object> properties;

    Node(Ident label, Ident variable, Map<Ident, Object> properties) {
        this.label = label;
        this.variable = variable;
        // Insertion order is part of the emitted text and therefore part of the contract, so this
        // is a LinkedHashMap wrapper rather than Map.copyOf, whose iteration order is unspecified.
        this.properties = Collections.unmodifiableMap(new LinkedHashMap<>(properties));
    }

    /**
     * Binds this node pattern to a variable so it can be referenced by predicates and projections.
     *
     * <p>Emits: {@code (<variable>:<Label>)}
     *
     * @param name the variable name, validated as a pattern variable
     * @return a new node pattern carrying the variable
     * @throws IllegalArgumentException if the name is not representable as a variable
     */
    public Node named(String name) {
        return new Node(label, Ident.variable(name), properties);
    }

    /**
     * Adds an inline property equality to the pattern. The value becomes a parameter, so this is
     * an injection-free spelling of the shape hand-writers most often build by concatenation.
     *
     * <p>Emits: {@code (<variable>:<Label> {<key>: $p<n>})}
     *
     * @param key the property key
     * @param value the value to match; emitted as a parameter, never as text
     * @return a new node pattern carrying the property
     * @throws IllegalArgumentException if the key is not representable, is already present, or the
     *     value is a query element
     */
    public Node withProperty(String key, Object value) {
        Ident ident = Ident.propertyKey(key);
        if (properties.containsKey(ident)) {
            throw new IllegalArgumentException(
                    "inline pattern property \"" + key + "\" is already set on this node");
        }
        Map<Ident, Object> next = new LinkedHashMap<>(properties);
        next.put(ident, Values.check(value));
        return new Node(label, variable, next);
    }

    /**
     * A reference to one of this node's properties.
     *
     * <p>Emits: {@code <variable>.<key>}
     *
     * @param key the property key
     * @return the property reference
     * @throws IllegalStateException if this node has no variable
     * @throws IllegalArgumentException if the key is not representable
     */
    public Expr prop(String key) {
        return new Ast.PropertyRef(requireVariable("prop"), Ident.propertyKey(key));
    }

    /**
     * This node's property map, as a value.
     *
     * <p>Emits: {@code properties(<variable>)}
     *
     * @return the function call
     * @throws IllegalStateException if this node has no variable
     */
    public Expr properties() {
        return new Ast.FunctionExpr("properties", false, false, ref());
    }

    /**
     * This node's labels, as a list value.
     *
     * <p>Emits: {@code labels(<variable>)}
     *
     * @return the function call
     * @throws IllegalStateException if this node has no variable
     */
    public Expr labels() {
        return new Ast.FunctionExpr("labels", false, false, ref());
    }

    /**
     * This node's identity.
     *
     * <p>Emits: {@code id(<variable>)}
     *
     * @return the function call
     * @throws IllegalStateException if this node has no variable
     */
    public Expr id() {
        return new Ast.FunctionExpr("id", false, false, ref());
    }

    /**
     * The bare variable, for the aggregate functions that take a node rather than a property —
     * {@code count(n)}, {@code collect(n)} is deliberately not offered, see the class javadoc.
     *
     * <p>Emits: {@code <variable>}
     *
     * @return the variable reference
     * @throws IllegalStateException if this node has no variable
     */
    public Expr ref() {
        return new Ast.VarRef(requireVariable("ref"));
    }

    /**
     * Extends this node into an outgoing relationship pattern.
     *
     * <p>Emits: {@code (<this>)-[<relationship>]-&gt;(<target>)}
     *
     * @param relationship the relationship pattern
     * @param target the node at the arrow's head
     * @return the path pattern
     */
    public Path to(Rel relationship, Node target) {
        return Path.start(this).to(relationship, target);
    }

    /**
     * Extends this node into an incoming relationship pattern.
     *
     * <p>Emits: {@code (<this>)&lt;-[<relationship>]-(<source>)}
     *
     * @param relationship the relationship pattern
     * @param source the node at the arrow's tail
     * @return the path pattern
     */
    public Path from(Rel relationship, Node source) {
        return Path.start(this).from(relationship, source);
    }

    /**
     * Extends this node into an undirected relationship pattern.
     *
     * <p>Emits: {@code (<this>)-[<relationship>]-(<other>)}
     *
     * @param relationship the relationship pattern
     * @param other the node at the far end
     * @return the path pattern
     */
    public Path related(Rel relationship, Node other) {
        return Path.start(this).related(relationship, other);
    }

    private Ident requireVariable(String method) {
        if (variable == null) {
            throw new IllegalStateException(
                    method + "() needs a variable: call named(\"…\") on the node pattern first");
        }
        return variable;
    }

    Ident label() {
        return label;
    }

    Ident variable() {
        return variable;
    }

    Map<Ident, Object> inlineProperties() {
        return properties;
    }
}
