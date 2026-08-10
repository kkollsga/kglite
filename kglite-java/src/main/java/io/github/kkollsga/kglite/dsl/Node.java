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
        return withEntry(Ident.propertyKey(key), Values.check(value));
    }

    private Node withEntry(Ident key, Object value) {
        if (properties.containsKey(key)) {
            throw new IllegalArgumentException(
                    "inline pattern property \"" + key.name() + "\" is already set on this node");
        }
        Map<Ident, Object> next = new LinkedHashMap<>(properties);
        next.put(key, value);
        return new Node(label, variable, next);
    }

    /**
     * Adds an inline property whose value is an <em>expression</em> rather than data — in v1, a
     * field of the current {@code UNWIND} row or an alias carried by a {@code WITH}.
     *
     * <p>Separate from {@link #withProperty(String, Object)} rather than an overload of it: an
     * overload would make {@code withProperty("k", null)} ambiguous, and keeping the two apart is
     * also what keeps the value-position rule readable — {@code withProperty} takes data and only
     * data, this one takes a piece of the query and nothing else.
     *
     * <p>Emits: {@code (<variable>:<Label> {<key>: <expression>})}
     *
     * @param key the property key
     * @param expression the value expression, from {@link UnwindStep#field(String)} or
     *     {@link Cypher#alias(String)}
     * @return a new node pattern carrying the property
     * @throws IllegalArgumentException if the key is not representable, is already present, or the
     *     expression is {@code null}
     */
    public Node withPropertyFrom(String key, Expr expression) {
        if (expression == null) {
            throw new IllegalArgumentException(
                    "withPropertyFrom(\"" + key + "\", …) needs an expression");
        }
        return withEntry(Ident.propertyKey(key), expression);
    }

    /**
     * Merges a map of properties into this node — the {@code SET n += $map} form, and the only
     * parameterised way to write a property set whose shape is decided at runtime.
     *
     * <p>Keys already on the node are overwritten; keys not mentioned are left alone. (Assigning
     * the whole property map, {@code SET n = $map}, is not offered: it silently drops every
     * property the map omits.)
     *
     * <p>Emits: {@code <variable> += $p<n>}
     *
     * @param values the properties to merge in
     * @return the assignment, for {@code set}, {@code onCreateSet} or {@code onMatchSet}
     * @throws IllegalStateException if this node has no variable
     * @throws IllegalArgumentException if {@code values} is {@code null} or one of its values is a
     *     query element rather than data
     */
    public Assignment plusProperties(Map<String, Object> values) {
        if (values == null) {
            throw new IllegalArgumentException("plusProperties() needs a map of properties");
        }
        Map<String, Object> copy = new LinkedHashMap<>();
        for (Map.Entry<String, Object> entry : values.entrySet()) {
            copy.put(entry.getKey(), Values.check(entry.getValue()));
        }
        return new Ast.MapAssignment(
                requireVariable("plusProperties"), Collections.unmodifiableMap(copy));
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
    public Property prop(String key) {
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
    public Variable ref() {
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
