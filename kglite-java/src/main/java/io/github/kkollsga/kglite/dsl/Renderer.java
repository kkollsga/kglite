package io.github.kkollsga.kglite.dsl;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

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

    private Renderer() {}

    static Rendered render(
            List<Ast.MatchStage> stages,
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
            Ast.Unwind unwind, List<Ast.MatchStage> stages, List<Ast.WriteClause> clauses) {
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
            List<Ast.MatchStage> stages,
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
        for (int i = 0; i < projections.size(); i++) {
            separate(i);
            Projection projection = projections.get(i);
            expression(projection.expression());
            out.append(" AS ").append(projection.aliasIdent().rendered());
        }
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

    private void stages(List<Ast.MatchStage> stages) {
        for (Ast.MatchStage stage : stages) {
            space();
            out.append(stage.optional() ? "OPTIONAL MATCH " : "MATCH ");
            patterns(stage.patterns());
            if (stage.where() != null) {
                out.append(" WHERE ");
                condition(stage.where());
            }
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

    /** Wraps a composite operand and leaves a simple one bare, so precedence never matters. */
    private void operand(Condition condition) {
        boolean composite = condition instanceof Ast.And || condition instanceof Ast.Or;
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
        }
    }

    /** Allocates the next {@code $p<n>} and records its value. The only route a value has. */
    private void parameter(Object value) {
        String name = "p" + params.size();
        params.put(name, value);
        out.append('$').append(name);
    }
}
