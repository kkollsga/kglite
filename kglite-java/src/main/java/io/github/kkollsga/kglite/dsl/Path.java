package io.github.kkollsga.kglite.dsl;

import java.util.ArrayList;
import java.util.List;

/**
 * A path pattern: a node, then one or more relationship hops.
 *
 * <p>Created by {@link Node#to(Rel, Node)} and its siblings rather than directly, and extended by
 * calling them again — {@code a.to(knows, b).to(worksAt, c)} emits one three-node pattern.
 *
 * <p>Immutable; every hop returns a new instance.
 */
public final class Path implements Pattern {

    /** One relationship hop and the node it lands on. */
    record Hop(Rel relationship, Direction direction, Node node) {}

    /** Which way the arrowhead points, or that there is none. */
    enum Direction {
        OUTGOING,
        INCOMING,
        UNDIRECTED,
    }

    private final Node start;
    private final List<Hop> hops;

    private Path(Node start, List<Hop> hops) {
        this.start = start;
        this.hops = List.copyOf(hops);
    }

    static Path start(Node start) {
        return new Path(start, List.of());
    }

    /**
     * Appends an outgoing hop.
     *
     * <p>Emits: {@code -[<relationship>]-&gt;(<target>)}
     *
     * @param relationship the relationship pattern
     * @param target the node at the arrow's head
     * @return a new path with the hop appended
     */
    public Path to(Rel relationship, Node target) {
        return append(relationship, Direction.OUTGOING, target);
    }

    /**
     * Appends an incoming hop.
     *
     * <p>Emits: {@code &lt;-[<relationship>]-(<source>)}
     *
     * @param relationship the relationship pattern
     * @param source the node at the arrow's tail
     * @return a new path with the hop appended
     */
    public Path from(Rel relationship, Node source) {
        return append(relationship, Direction.INCOMING, source);
    }

    /**
     * Appends an undirected hop.
     *
     * <p>Emits: {@code -[<relationship>]-(<other>)}
     *
     * @param relationship the relationship pattern
     * @param other the node at the far end
     * @return a new path with the hop appended
     */
    public Path related(Rel relationship, Node other) {
        return append(relationship, Direction.UNDIRECTED, other);
    }

    private Path append(Rel relationship, Direction direction, Node node) {
        if (relationship == null || node == null) {
            throw new IllegalArgumentException("a path hop needs both a relationship and a node");
        }
        List<Hop> next = new ArrayList<>(hops);
        next.add(new Hop(relationship, direction, node));
        return new Path(start, next);
    }

    Node startNode() {
        return start;
    }

    List<Hop> hops() {
        return hops;
    }
}
