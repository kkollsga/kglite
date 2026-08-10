package io.github.kkollsga.kglite.dsl;

import io.github.kkollsga.kglite.KnowledgeGraph;
import io.github.kkollsga.kglite.Transaction;
import java.util.List;
import java.util.Map;

/**
 * A finished read statement: its Cypher text, its parameters, and the ability to run itself.
 *
 * <p>The text and the parameter map are always available, which is both the escape hatch and the
 * reason the emitted-Cypher gate can exist: a caller who needs something the DSL cannot say takes
 * {@link #cypher()}, edits it, and calls {@code KnowledgeGraph.query} directly.
 *
 * <p>Statements are immutable and safe to hold in a static field and reuse across threads.
 */
public interface Statement {

    /**
     * The emitted Cypher, in the one rendering style this DSL has: keywords upper-case, single
     * spaces, no newlines, no comments, and every value replaced by a {@code $p<n>} parameter.
     *
     * <p>Emits: nothing further — this is the accessor for the text already built.
     *
     * @return the query text
     */
    String cypher();

    /**
     * The parameters the text refers to, in the order they appear in it: {@code p0}…{@code pN} for
     * every value a builder parameterised, plus whatever names a {@link Raw} fragment declared for
     * itself. The map is complete either way — nothing has to be merged in before execution, and
     * the two namespaces cannot collide because a raw fragment may not claim {@code p<digits>}.
     *
     * <p>Emits: nothing further — this is the accessor for the values already collected.
     *
     * @return an unmodifiable map in emission order
     */
    Map<String, Object> params();

    /**
     * Runs this statement against a graph, now.
     *
     * <p>A read routes to {@code KnowledgeGraph.query} and a {@link WriteStatement} to
     * {@code KnowledgeGraph.cypher}; the choice is made by the statement's type rather than by the
     * caller, so the binding's documented {@code cypher}-versus-{@code query} footgun is
     * unreachable through the DSL.
     *
     * <p>Emits: nothing further — this executes {@link #cypher()} with {@link #params()}.
     *
     * @param graph the graph to run against
     * @return the result rows, exactly as the raw route returns them
     */
    List<Map<String, Object>> on(KnowledgeGraph graph);

    /**
     * Stages this statement into a transaction, to run when it commits.
     *
     * <p>The same method for every kind of statement, because a transaction takes text and
     * parameters and applies them atomically whatever they say — including a read, whose rows come
     * back from {@code commit()} in position. Nothing executes here: see {@link Transaction} for
     * what staging does and does not promise.
     *
     * <p>Emits: nothing further — this stages {@link #cypher()} with {@link #params()}.
     *
     * @param transaction the transaction to stage into
     * @return that transaction, so staging chains
     * @throws IllegalArgumentException if {@code transaction} is {@code null}
     */
    default Transaction on(Transaction transaction) {
        if (transaction == null) {
            throw new IllegalArgumentException("on() needs a transaction");
        }
        return transaction.add(cypher(), params());
    }
}
