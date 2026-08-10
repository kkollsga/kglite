package io.github.kkollsga.kglite.dsl;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * The one renderer. There is no pretty-printing mode and no plug-point: a second rendering style
 * would double every emitted-equality expectation for no user-visible gain.
 *
 * <p>The contract it keeps:
 *
 * <ul>
 *   <li>Clause order is the grammar's; stage order is the caller's. The step interfaces already
 *       made an out-of-order call a compile error, so there is no normalisation pass.
 *   <li>Keywords upper-case, exactly one space between tokens, {@code ", "} between list items, no
 *       newlines, no comments, no trailing whitespace.
 *   <li>Every value is a parameter named {@code p<n>}, numbered from zero in the order the
 *       parameters appear in the finished text. <b>Values are never deduplicated</b>: two equal
 *       values get two parameters, because dedup would make the numbering — and therefore the
 *       emitted text — depend on value equality between arguments the caller happened to repeat.
 *   <li>Nothing is rewritten. Inline pattern properties stay inline, a predicate stays in its
 *       {@code WHERE}, and no {@code OR} chain becomes an {@code IN}. A DSL that optimises is a
 *       DSL whose emitted-equality tests encode planner behaviour.
 *   <li>A raw fragment is copied out character for character, and the parameters it names keep
 *       their names — the fragment refers to them, so renumbering them would break the text being
 *       copied. Nothing about a fragment is inspected, escaped or repaired; see {@link Raw} for
 *       what that means for the injection property.
 * </ul>
 */
final class Renderer {

    /**
     * A finished rendering: the text and the parameters it refers to, which only make sense
     * together.
     *
     * @param cypher the query text
     * @param params the parameters, in emission order
     */
    record Rendered(String cypher, Map<String, Object> params) {}

    private final StringBuilder out = new StringBuilder();
    private final Map<String, Object> params = new LinkedHashMap<>();

    /**
     * The next {@code p<n>} to hand out. Counted separately from {@code params.size()} because a
     * raw fragment's own named parameters share that map: numbering off the map's size would leave
     * gaps in the generated sequence as soon as one appeared.
     */
    private int generated;

    private Renderer() {}

    static Rendered render(
            List<Ast.ReadStage> stages,
            boolean distinct,
            List<Projection> projections,
            List<SortItem> sorts,
            Long skip,
            Long limit) {
        Renderer renderer = new Renderer();
        renderer.statement(stages, distinct, projections, sorts, skip, limit);
        return new Rendered(
                renderer.out.toString(), Collections.unmodifiableMap(renderer.params));
    }

    /**
     * Renders an updating statement: an optional {@code UNWIND}, the matching stages, and the
     * updating clauses, in that order.
     *
     * @param unwind the batch opener, or {@code null}
     * @param stages the matching stages; may be empty
     * @param clauses the updating clauses; at least one
     * @return the finished rendering
     */
    static Rendered render(
            Ast.Unwind unwind, List<Ast.ReadStage> stages, List<Ast.WriteClause> clauses) {
        Renderer renderer = new Renderer();
        if (unwind != null) {
            renderer.out.append("UNWIND ");
            renderer.parameter(unwind.rows());
            renderer.out.append(" AS ").append(unwind.variable().rendered());
        }
        renderer.stages(stages);
        for (Ast.WriteClause clause : clauses) {
            renderer.space();
            renderer.writeClause(clause);
        }
        return new Rendered(
                renderer.out.toString(), Collections.unmodifiableMap(renderer.params));
    }

    private void statement(
            List<Ast.ReadStage> stages,
            boolean distinct,
            List<Projection> projections,
            List<SortItem> sorts,
            Long skip,
            Long limit) {
        stages(stages);
        out.append(" RETURN ");
        if (distinct) {
            out.append("DISTINCT ");
        }
        projections(projections);
        if (!sorts.isEmpty()) {
            out.append(" ORDER BY ");
            for (int i = 0; i < sorts.size(); i++) {
                separate(i);
                SortItem item = sorts.get(i);
                expression(item.expression());
                out.append(item.descending() ? " DESC" : " ASC");
            }
        }
        if (skip != null) {
            out.append(" SKIP ");
            parameter(skip);
        }
        if (limit != null) {
            out.append(" LIMIT ");
            parameter(limit);
        }
    }

    private void stages(List<Ast.ReadStage> stages) {
        for (Ast.ReadStage stage : stages) {
            space();
            switch (stage) {
                case Ast.MatchStage match -> {
                    out.append(match.optional() ? "OPTIONAL MATCH " : "MATCH ");
                    patterns(match.patterns());
                    filter(match.where());
                }
                case Ast.WithStage with -> {
                    out.append("WITH ");
                    if (with.distinct()) {
                        out.append("DISTINCT ");
                    }
                    projections(with.projections());
                    filter(with.where());
                }
                case Ast.RawStage rawStage -> raw(rawStage.fragment(), rawStage.params());
            }
        }
    }

    /** The stage's {@code WHERE}, if it has one. */
    private void filter(Condition where) {
        if (where != null) {
            out.append(" WHERE ");
            condition(where);
        }
    }

    /** The {@code <expression> AS <alias>} list shared by {@code WITH} and {@code RETURN}. */
    private void projections(List<Projection> projections) {
        for (int i = 0; i < projections.size(); i++) {
            separate(i);
            Projection projection = projections.get(i);
            expression(projection.expression());
            out.append(" AS ").append(projection.aliasIdent().rendered());
        }
    }

    private void writeClause(Ast.WriteClause clause) {
        switch (clause) {
            case Ast.Create create -> {
                out.append("CREATE ");
                patterns(create.patterns());
            }
            case Ast.MergeClause merge -> {
                out.append("MERGE ");
                pattern(merge.pattern());
                if (!merge.onCreate().isEmpty()) {
                    out.append(" ON CREATE SET ");
                    assignments(merge.onCreate());
                }
                if (!merge.onMatch().isEmpty()) {
                    out.append(" ON MATCH SET ");
                    assignments(merge.onMatch());
                }
            }
            case Ast.SetClause set -> {
                out.append("SET ");
                assignments(set.assignments());
            }
            case Ast.RemoveClause remove -> {
                out.append("REMOVE ");
                for (int i = 0; i < remove.properties().size(); i++) {
                    separate(i);
                    expression(remove.properties().get(i));
                }
            }
            case Ast.DeleteClause delete -> {
                out.append(delete.detach() ? "DETACH DELETE " : "DELETE ");
                for (int i = 0; i < delete.elements().size(); i++) {
                    separate(i);
                    expression(delete.elements().get(i));
                }
            }
        }
    }

    private void assignments(List<Assignment> assignments) {
        for (int i = 0; i < assignments.size(); i++) {
            separate(i);
            switch (assignments.get(i)) {
                case Ast.PropertyAssignment property -> {
                    expression(property.target());
                    out.append(" = ");
                    parameter(property.value());
                }
                case Ast.MapAssignment map -> {
                    out.append(map.variable().rendered()).append(" += ");
                    parameter(map.values());
                }
            }
        }
    }

    private void patterns(List<Pattern> patterns) {
        for (int i = 0; i < patterns.size(); i++) {
            separate(i);
            pattern(patterns.get(i));
        }
    }

    /** {@code ", "} before every item but the first. */
    private void separate(int index) {
        if (index > 0) {
            out.append(", ");
        }
    }

    /** One space between clauses, and none before the first. */
    private void space() {
        if (out.length() > 0) {
            out.append(' ');
        }
    }

    private void pattern(Pattern pattern) {
        switch (pattern) {
            case Node node -> node(node);
            case Path path -> path(path);
        }
    }

    private void path(Path path) {
        node(path.startNode());
        for (Path.Hop hop : path.hops()) {
            relationship(hop.relationship(), hop.direction());
            node(hop.node());
        }
    }

    private void node(Node node) {
        out.append('(');
        if (node.variable() != null) {
            out.append(node.variable().rendered());
        }
        if (node.label() != null) {
            out.append(':').append(node.label().rendered());
        }
        inlineProperties(node.inlineProperties());
        out.append(')');
    }

    private void relationship(Rel rel, Path.Direction direction) {
        boolean incoming = direction == Path.Direction.INCOMING;
        boolean outgoing = direction == Path.Direction.OUTGOING;
        if (rel.isBare()) {
            // (a)-->(b) rather than (a)-[]->(b): the arrow-only form is the one the dialect was
            // probed with, and it is what a hand-writer writes.
            out.append(incoming ? "<--" : outgoing ? "-->" : "--");
            return;
        }
        out.append(incoming ? "<-[" : "-[");
        if (rel.variable() != null) {
            out.append(rel.variable().rendered());
        }
        if (rel.typeIdent() != null) {
            out.append(':').append(rel.typeIdent().rendered());
        }
        inlineProperties(rel.inlineProperties());
        out.append(outgoing ? "]->" : "]-");
    }

    private void inlineProperties(Map<Ident, Object> properties) {
        if (properties.isEmpty()) {
            return;
        }
        out.append(" {");
        int index = 0;
        for (Map.Entry<Ident, Object> entry : properties.entrySet()) {
            separate(index++);
            out.append(entry.getKey().rendered()).append(": ");
            // An expression here came from withPropertyFrom, the one inline-property
            // spelling that is not a value — a row field under an UNWIND. Every other
            // value in the map is caller data and becomes a parameter.
            if (entry.getValue() instanceof Expr expression) {
                expression(expression);
            } else {
                parameter(entry.getValue());
            }
        }
        out.append('}');
    }

    private void condition(Condition condition) {
        switch (condition) {
            case Ast.Comparison comparison -> {
                expression(comparison.left());
                out.append(' ').append(comparison.operator()).append(' ');
                parameter(comparison.value());
            }
            case Ast.NullCheck nullCheck -> {
                expression(nullCheck.operand());
                out.append(nullCheck.negated() ? " IS NOT NULL" : " IS NULL");
            }
            case Ast.Not not -> {
                out.append("NOT (");
                condition(not.operand());
                out.append(')');
            }
            case Ast.And and -> junction(and.operands(), " AND ");
            case Ast.Or or -> junction(or.operands(), " OR ");
            case Ast.RawExpr rawExpr -> raw(rawExpr.fragment(), rawExpr.params());
        }
    }

    private void junction(List<Condition> operands, String keyword) {
        for (int i = 0; i < operands.size(); i++) {
            if (i > 0) {
                out.append(keyword);
            }
            operand(operands.get(i));
        }
    }

    /**
     * Wraps a composite operand and leaves a simple one bare, so precedence never matters.
     *
     * <p>A raw fragment counts as composite. This DSL cannot see inside it, so a fragment that is
     * itself a disjunction would otherwise bind to the surrounding {@code AND} by the dialect's
     * precedence table rather than as the single operand the caller passed.
     */
    private void operand(Condition condition) {
        boolean composite = condition instanceof Ast.And
                || condition instanceof Ast.Or
                || condition instanceof Raw;
        if (composite) {
            out.append('(');
        }
        condition(condition);
        if (composite) {
            out.append(')');
        }
    }

    private void expression(Expr expression) {
        switch (expression) {
            case Ast.PropertyRef property ->
                    out.append(property.variable().rendered())
                            .append('.')
                            .append(property.key().rendered());
            case Ast.VarRef variable -> out.append(variable.variable().rendered());
            case Ast.AliasRef alias -> out.append(alias.alias().rendered());
            case Ast.FunctionExpr function -> {
                out.append(function.name()).append('(');
                if (function.star()) {
                    out.append('*');
                } else {
                    if (function.distinct()) {
                        out.append("DISTINCT ");
                    }
                    expression(function.argument());
                }
                out.append(')');
            }
            case Ast.RawExpr rawExpr -> raw(rawExpr.fragment(), rawExpr.params());
        }
    }

    /** Allocates the next {@code $p<n>} and records its value. The only route a value has. */
    private void parameter(Object value) {
        String name = "p" + generated++;
        params.put(name, value);
        out.append('$').append(name);
    }

    /**
     * Emits caller-written text verbatim and merges the parameters it names.
     *
     * <p>The names are the caller's and are never renumbered — the fragment refers to them by name,
     * so renaming them would break the very text this method is copying. They cannot collide with
     * the emitter's own namespace because {@link RawFragment} refuses a name matching
     * {@code p<digits>}. Two fragments naming the same parameter agree or the statement is refused:
     * one name means one value in one statement, and silently keeping the last write would make the
     * emitted text and the parameter map disagree about what the query asked for.
     */
    private void raw(String fragment, Map<String, Object> fragmentParams) {
        out.append(fragment);
        for (Map.Entry<String, Object> entry : fragmentParams.entrySet()) {
            String name = entry.getKey();
            Object value = entry.getValue();
            if (params.containsKey(name) && !Objects.equals(params.get(name), value)) {
                throw new IllegalArgumentException(
                        "raw parameter \"" + name + "\" is bound twice in one statement, to "
                                + params.get(name) + " and " + value
                                + ". A parameter name means one value; use two names.");
            }
            params.put(name, value);
        }
    }
}
