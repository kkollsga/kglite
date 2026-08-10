package io.github.kkollsga.kglite.dsl;

import io.github.kkollsga.kglite.KnowledgeGraph;
import java.util.List;
import java.util.Map;

/**
 * A projected statement: stages, projections, and the optional ordering and paging.
 *
 * <p>The text is rendered in the constructor rather than lazily, so anything the renderer can
 * reject is rejected while the statement is being built rather than when it is first run.
 */
final class ProjectedQuery implements ReturnStep, OrderStep, SkipStep {

    private final List<Ast.ReadStage> stages;
    private final boolean distinct;
    private final List<Projection> projections;
    private final List<SortItem> sorts;
    private final Long skip;
    private final Long limit;
    private final Renderer.Rendered rendered;

    ProjectedQuery(
            List<Ast.ReadStage> stages,
            boolean distinct,
            List<Projection> projections,
            List<SortItem> sorts,
            Long skip,
            Long limit) {
        this.stages = List.copyOf(stages);
        this.distinct = distinct;
        this.projections = List.copyOf(projections);
        this.sorts = List.copyOf(sorts);
        this.skip = skip;
        this.limit = limit;
        this.rendered = Renderer.render(this.stages, distinct, this.projections, this.sorts,
                this.skip, this.limit);
    }

    @Override
    public OrderStep orderBy(SortItem... items) {
        if (items == null || items.length == 0) {
            throw new IllegalArgumentException("ORDER BY needs at least one sort key");
        }
        for (SortItem item : items) {
            if (item == null) {
                throw new IllegalArgumentException("ORDER BY may not contain a null sort key");
            }
        }
        return new ProjectedQuery(stages, distinct, projections, List.of(items), skip, limit);
    }

    @Override
    public SkipStep skip(long rows) {
        return new ProjectedQuery(stages, distinct, projections, sorts, nonNegative(rows, "SKIP"),
                limit);
    }

    @Override
    public Statement limit(long rows) {
        return new ProjectedQuery(stages, distinct, projections, sorts, skip,
                nonNegative(rows, "LIMIT"));
    }

    @Override
    public String cypher() {
        return rendered.cypher();
    }

    @Override
    public Map<String, Object> params() {
        return rendered.params();
    }

    @Override
    public List<Map<String, Object>> on(KnowledgeGraph graph) {
        if (graph == null) {
            throw new IllegalArgumentException("on() needs a graph");
        }
        return graph.query(cypher(), params());
    }

    @Override
    public String toString() {
        return cypher();
    }

    private static long nonNegative(long rows, String clause) {
        if (rows < 0) {
            throw new IllegalArgumentException(clause + " must not be negative, got " + rows);
        }
        return rows;
    }
}
