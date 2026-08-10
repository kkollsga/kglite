package io.github.kkollsga.kglite.dsl;

/**
 * Something a {@code MATCH} can match: a single node, or a path of nodes joined by relationships.
 *
 * <p>Sealed over exactly {@link Node} and {@link Path}; v1 models no other pattern shape (no
 * variable-length relationships, no named paths, no {@code shortestPath}).
 */
public sealed interface Pattern permits Node, Path {}
