package io.github.kkollsga.kglite.dsl;

import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

/**
 * The reading half of a statement under construction: the {@code MATCH}/{@code OPTIONAL MATCH}
 * stages and their predicates, before anything has been projected.
 *
 * <p>Implements both step interfaces that are legal at this point. There is no state in which an
 * illegal call is reachable, because {@code returning} hands over to a different class.
 */
final class ReadingQuery implements MatchStep, WhereStep {

    private final List<Ast.MatchStage> stages;

    private ReadingQuery(List<Ast.MatchStage> stages) {
        this.stages = List.copyOf(stages);
    }

    static ReadingQuery matching(boolean optional, Pattern... patterns) {
        return new ReadingQuery(List.of(new Ast.MatchStage(optional, patterns(patterns), null)));
    }

    @Override
    public MatchStep match(Pattern... patterns) {
        return added(new Ast.MatchStage(false, patterns(patterns), null));
    }

    @Override
    public MatchStep optionalMatch(Pattern... patterns) {
        return added(new Ast.MatchStage(true, patterns(patterns), null));
    }

    @Override
    public WhereStep where(Condition predicate) {
        if (predicate == null) {
            throw new IllegalArgumentException("where() needs a predicate");
        }
        Ast.MatchStage last = stages.get(stages.size() - 1);
        List<Ast.MatchStage> next = new ArrayList<>(stages.subList(0, stages.size() - 1));
        next.add(new Ast.MatchStage(last.optional(), last.patterns(), predicate));
        return new ReadingQuery(next);
    }

    @Override
    public ReturnStep returning(Projection... projections) {
        return project(false, projections);
    }

    @Override
    public ReturnStep returningDistinct(Projection... projections) {
        return project(true, projections);
    }

    // ---- the updating clauses, all of which end the statement ----------------------------

    @Override
    public WriteStatement create(Pattern... patterns) {
        return WriteQuery.after(stages, new Ast.Create(Ast.patterns(patterns)));
    }

    @Override
    public MergeStep merge(Pattern pattern) {
        return WriteQuery.after(stages, Ast.merge(pattern));
    }

    @Override
    public WriteStatement set(Assignment... assignments) {
        return WriteQuery.after(stages,
                new Ast.SetClause(WriteQuery.assignments(assignments, "set")));
    }

    @Override
    public WriteStatement remove(Property... properties) {
        return WriteQuery.after(stages, new Ast.RemoveClause(
                Ast.checked(properties, "property", "remove() needs at least one property")));
    }

    @Override
    public WriteStatement delete(Variable... elements) {
        return deleting(elements, false, "delete");
    }

    @Override
    public WriteStatement detachDelete(Variable... elements) {
        return deleting(elements, true, "detachDelete");
    }

    private WriteStatement deleting(Variable[] elements, boolean detach, String method) {
        return WriteQuery.after(stages, new Ast.DeleteClause(
                Ast.checked(elements, "element", method + "() needs at least one element"),
                detach));
    }

    private ReturnStep project(boolean distinct, Projection... projections) {
        return new ProjectedQuery(stages, distinct, checkedProjections(projections),
                List.of(), null, null);
    }

    private ReadingQuery added(Ast.MatchStage stage) {
        List<Ast.MatchStage> next = new ArrayList<>(stages);
        next.add(stage);
        return new ReadingQuery(next);
    }

    private static List<Pattern> patterns(Pattern... patterns) {
        if (patterns == null || patterns.length == 0) {
            throw new IllegalArgumentException("a matching stage needs at least one pattern");
        }
        for (Pattern pattern : patterns) {
            if (pattern == null) {
                throw new IllegalArgumentException("a matching stage may not contain a null pattern");
            }
        }
        return List.of(patterns);
    }

    /**
     * Rejects duplicate aliases while the statement is still being built.
     *
     * <p>The raw route lets the second projection quietly overwrite the first in the result map, so
     * one of the two columns disappears with no error anywhere. A builder can see both aliases
     * before anything executes, so it says so — and it keeps saying so regardless of what the
     * engine does with duplicates later.
     */
    private static List<Projection> checkedProjections(Projection... projections) {
        if (projections == null || projections.length == 0) {
            throw new IllegalArgumentException("RETURN needs at least one projection");
        }
        Set<String> seen = new LinkedHashSet<>();
        for (int i = 0; i < projections.length; i++) {
            Projection projection = projections[i];
            if (projection == null) {
                throw new IllegalArgumentException("RETURN may not contain a null projection");
            }
            if (!seen.add(projection.alias())) {
                throw new IllegalArgumentException(
                        "duplicate RETURN alias \"" + projection.alias() + "\" at position " + i
                                + ": result rows are keyed by alias, so the columns would collide "
                                + "and one of them would silently vanish from every row");
            }
        }
        return List.of(projections);
    }
}
