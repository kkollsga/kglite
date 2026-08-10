package io.github.kkollsga.kglite.dsl;

import io.github.kkollsga.kglite.KnowledgeGraph;
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
     * The parameters the text refers to, keyed {@code p0}…{@code pN} in the order they appear in
     * the text. The map is complete: this DSL has no caller-named parameters, so nothing needs to
     * be merged in before execution.
     *
     * <p>Emits: nothing further — this is the accessor for the values already collected.
     *
     * @return an unmodifiable map in emission order
     */
    Map<String, Object> params();

    /**
     * Runs this statement against a graph.
     *
     * <p>Read statements route to {@code KnowledgeGraph.query}, which is the read entry point; the
     * choice is made by the type rather than by the caller, so the binding's documented
     * {@code cypher}-versus-{@code query} footgun is unreachable through the DSL.
     *
     * <p>Emits: nothing further — this executes {@link #cypher()} with {@link #params()}.
     *
     * @param graph the graph to query
     * @return the result rows, exactly as the raw route returns them
     */
    List<Map<String, Object>> on(KnowledgeGraph graph);
}
