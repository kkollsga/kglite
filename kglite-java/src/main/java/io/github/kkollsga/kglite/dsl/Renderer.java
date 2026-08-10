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

    private void statement(
            List<Ast.MatchStage> stages,
            boolean distinct,
            List<Projection> projections,
            List<SortItem> sorts,
            Long skip,
            Long limit) {
        for (Ast.MatchStage stage : stages) {
            if (out.length() > 0) {
                out.append(' ');
            }
            out.append(stage.optional() ? "OPTIONAL MATCH " : "MATCH ");
            for (int i = 0; i < stage.patterns().size(); i++) {
                if (i > 0) {
                    out.append(", ");
                }
                pattern(stage.patterns().get(i));
            }
            if (stage.where() != null) {
                out.append(" WHERE ");
                condition(stage.where());
            }
        }
        out.append(" RETURN ");
        if (distinct) {
            out.append("DISTINCT ");
        }
        for (int i = 0; i < projections.size(); i++) {
            if (i > 0) {
                out.append(", ");
            }
            Projection projection = projections.get(i);
            expression(projection.expression());
            out.append(" AS ").append(projection.aliasIdent().rendered());
        }
        if (!sorts.isEmpty()) {
            out.append(" ORDER BY ");
            for (int i = 0; i < sorts.size(); i++) {
                if (i > 0) {
                    out.append(", ");
                }
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
        boolean first = true;
        for (Map.Entry<Ident, Object> entry : properties.entrySet()) {
            if (!first) {
                out.append(", ");
            }
            first = false;
            out.append(entry.getKey().rendered()).append(": ");
            parameter(entry.getValue());
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
