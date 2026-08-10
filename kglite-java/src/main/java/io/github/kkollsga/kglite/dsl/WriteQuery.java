package io.github.kkollsga.kglite.dsl;

import io.github.kkollsga.kglite.KnowledgeGraph;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * A finished updating statement: an optional {@code UNWIND}, the matching stages that preceded it,
 * and one updating clause.
 *
 * <p>Implements every step interface that is legal at this point, the same way {@link ReadingQuery}
 * does for the reading half. The step a caller holds is what limits them: {@code create} hands back
 * a {@link WriteStatement}, so {@code onCreateSet} is not offered on it, and {@code merge} hands
 * back a {@link MergeStep}, where it is.
 *
 * <p>Rendered in the constructor, so anything the renderer rejects is rejected while the statement
 * is being built rather than when it is first run.
 */
final class WriteQuery implements MergeStep, MergeMatchStep {

    private final Ast.Unwind unwind;
    private final List<Ast.ReadStage> stages;
    private final List<Ast.WriteClause> clauses;
    private final Renderer.Rendered rendered;

    private WriteQuery(
            Ast.Unwind unwind, List<Ast.ReadStage> stages, List<Ast.WriteClause> clauses) {
        this.unwind = unwind;
        this.stages = List.copyOf(stages);
        this.clauses = List.copyOf(clauses);
        this.rendered = Renderer.render(unwind, this.stages, this.clauses);
    }

    /** A statement that opens with an {@code UNWIND}. */
    static WriteQuery opening(Ast.Unwind unwind, Ast.WriteClause clause) {
        return new WriteQuery(unwind, List.of(), List.of(clause));
    }

    /** A statement whose updating clause follows matching stages, or none at all. */
    static WriteQuery after(List<Ast.ReadStage> stages, Ast.WriteClause clause) {
        return new WriteQuery(null, stages, List.of(clause));
    }

    @Override
    public MergeMatchStep onCreateSet(Assignment... assignments) {
        return withMerge(merge -> new Ast.MergeClause(merge.pattern(),
                assignments(assignments, "onCreateSet"), merge.onMatch()));
    }

    @Override
    public WriteStatement onMatchSet(Assignment... assignments) {
        return withMerge(merge -> new Ast.MergeClause(merge.pattern(), merge.onCreate(),
                assignments(assignments, "onMatchSet")));
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
        // The write path, chosen by this statement's type rather than by the caller.
        return graph.cypher(cypher(), params());
    }

    @Override
    public String toString() {
        return cypher();
    }

    /** Replaces the trailing {@code MERGE} with a version carrying one more conditional clause. */
    private WriteQuery withMerge(java.util.function.UnaryOperator<Ast.MergeClause> edit) {
        Ast.WriteClause last = clauses.get(clauses.size() - 1);
        if (!(last instanceof Ast.MergeClause merge)) {
            // Unreachable through the public step types, which only offer these two methods on a
            // MergeStep. Kept so a future step-interface change fails loudly rather than emitting
            // an ON CREATE SET attached to a CREATE.
            throw new IllegalStateException(
                    "ON CREATE SET / ON MATCH SET belong to a MERGE, and this statement ends with "
                            + last.getClass().getSimpleName());
        }
        List<Ast.WriteClause> next = new ArrayList<>(clauses.subList(0, clauses.size() - 1));
        next.add(edit.apply(merge));
        return new WriteQuery(unwind, stages, next);
    }

    static List<Assignment> assignments(Assignment[] assignments, String clause) {
        return Ast.checked(assignments, "assignment", clause + " needs at least one assignment");
    }
}
