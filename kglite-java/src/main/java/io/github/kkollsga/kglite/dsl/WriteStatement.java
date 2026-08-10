package io.github.kkollsga.kglite.dsl;

import io.github.kkollsga.kglite.KnowledgeGraph;
import io.github.kkollsga.kglite.Transaction;

/**
 * A finished statement that changes the graph.
 *
 * <p>The only thing it adds to {@link Statement} is which entry point
 * {@link Statement#on(KnowledgeGraph)} takes: a write routes to
 * {@code KnowledgeGraph.cypher}, a read to {@code KnowledgeGraph.query}, and the statement's own
 * type decides — so the binding's documented {@code cypher}-versus-{@code query} footgun is not
 * something a DSL caller can get wrong. There is no method here to call and no cast to write.
 *
 * <p>{@link Statement#on(Transaction)} is the same method for both kinds: a write staged into a
 * transaction is one of the batch's statements, and the batch is applied atomically at
 * {@code commit()}.
 */
public interface WriteStatement extends Statement {}
