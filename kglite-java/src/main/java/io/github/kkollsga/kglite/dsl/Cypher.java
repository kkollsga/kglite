package io.github.kkollsga.kglite.dsl;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Map;

/**
 * The one class to import.
 *
 * <p>{@code import static io.github.kkollsga.kglite.dsl.Cypher.*;} and everything the DSL offers
 * is one completion popup away; from there the step interfaces show only the continuations the
 * grammar allows.
 *
 * <pre>{@code
 * Node p = node("Person").named("p");
 * Statement stmt = match(p)
 *         .where(p.prop("age").gt(30))
 *         .returning(p.prop("title").as("name"))
 *         .orderBy(alias("name").desc())
 *         .limit(10);
 * List<Map<String, Object>> rows = stmt.on(graph);
 * }</pre>
 *
 * <p>{@code stmt.cypher()} for that statement is
 * {@code MATCH (p:Person) WHERE p.age > $p0 RETURN p.title AS name ORDER BY name DESC LIMIT $p1},
 * with {@code {p0=30, p1=10}} as its parameters — the values never appear in the text, which is
 * the property that makes injection through this route structurally impossible rather than
 * carefully avoided.
 *
 * <p>What the DSL covers is the read half of the clause set that is both common and easy to
 * assemble wrongly by hand. Everything else — graph algorithms, vector search, procedure calls,
 * subqueries, DDL — stays on the raw {@code cypher}/{@code query} route, where the string
 * <em>is</em> the best Java for the job. See {@code CYPHER.md} for what the dialect supports; this
 * DSL never restates a clause's semantics, it only names the production each method emits.
 */
public final class Cypher {

    private Cypher() {}

    /**
     * A labelled node pattern.
     *
     * <p>Emits: {@code (:<Label>)}
     *
     * @param label the node label
     * @return the node pattern
     * @throws IllegalArgumentException if the label is not representable
     */
    public static Node node(String label) {
        return new Node(Ident.label(label), null, Map.of());
    }

    /**
     * An unlabelled node pattern, for matching any node or for referring to a variable bound by an
     * earlier stage.
     *
     * <p>Emits: {@code ()}
     *
     * @return the node pattern
     */
    public static Node anyNode() {
        return new Node(null, null, Map.of());
    }

    /**
     * A typed relationship pattern.
     *
     * <p>Emits: {@code [:<TYPE>]}
     *
     * @param type the relationship type
     * @return the relationship pattern
     * @throws IllegalArgumentException if the type is not representable
     */
    public static Rel rel(String type) {
        return new Rel(Ident.relationshipType(type), null, Map.of());
    }

    /**
     * An untyped relationship pattern, matching a relationship of any type.
     *
     * <p>Emits: {@code -->} (or {@code <--} / {@code --}, per the direction it is used in)
     *
     * @return the relationship pattern
     */
    public static Rel anyRel() {
        return new Rel(null, null, Map.of());
    }

    /**
     * Opens a statement with a matching stage.
     *
     * <p>Emits: {@code MATCH <pattern>[, <pattern>…]}
     *
     * @param patterns the comma-joined patterns; at least one
     * @return the first chain step
     */
    public static MatchStep match(Pattern... patterns) {
        return ReadingQuery.matching(false, patterns);
    }

    /**
     * Opens a statement that creates the given patterns.
     *
     * <p>Emits: {@code CREATE <pattern>[, <pattern>…]}
     *
     * @param patterns the comma-joined patterns; at least one
     * @return the finished statement
     */
    public static WriteStatement create(Pattern... patterns) {
        return WriteQuery.after(List.of(), new Ast.Create(Ast.patterns(patterns)));
    }

    /**
     * Opens a statement that matches the pattern or creates it — the upsert.
     *
     * <p>Emits: {@code MERGE <pattern>}
     *
     * @param pattern the pattern to match or create
     * @return the next chain step, which is already a complete statement
     */
    public static MergeStep merge(Pattern pattern) {
        return WriteQuery.after(List.of(), Ast.merge(pattern));
    }

    /**
     * Opens a batch write: one statement that applies its clause once per row.
     *
     * <p>The whole list travels as a single parameter and the loop variable is named {@code row};
     * {@link UnwindStep#field(String)} is how a pattern reads one of its keys. See
     * {@link UnwindStep} for the shape.
     *
     * <p>Emits: {@code UNWIND $p<n> AS row}
     *
     * @param rows the rows, typically maps; the collection becomes one parameter
     * @return the batch step
     * @throws IllegalArgumentException if {@code rows} is {@code null} or an element is a query
     *     element rather than data
     */
    public static UnwindStep unwind(Collection<?> rows) {
        if (rows == null) {
            throw new IllegalArgumentException("unwind() requires a collection of rows, not null");
        }
        List<Object> copy = new ArrayList<>(rows.size());
        for (Object row : rows) {
            copy.add(Values.check(row));
        }
        return new UnwindStep(new Ast.Unwind(List.copyOf(copy), Ident.variable("row")));
    }

    /**
     * A reference to a {@code RETURN} alias, for ordering by a projected column.
     *
     * <p>Emits: {@code <alias>}
     *
     * @param alias the alias named by an earlier {@link Expr#as(String)}
     * @return the expression
     * @throws IllegalArgumentException if the alias is not representable
     */
    public static Expr alias(String alias) {
        return new Ast.AliasRef(Ident.alias(alias));
    }

    /**
     * Counts non-null occurrences of an expression.
     *
     * <p>Emits: {@code count(<expression>)}
     *
     * @param expression what to count
     * @return the aggregate
     */
    public static Expr count(Expr expression) {
        return new Ast.FunctionExpr("count", false, false, required(expression, "count"));
    }

    /**
     * Counts distinct occurrences of an expression.
     *
     * <p>Emits: {@code count(DISTINCT <expression>)}
     *
     * @param expression what to count
     * @return the aggregate
     */
    public static Expr countDistinct(Expr expression) {
        return new Ast.FunctionExpr("count", true, false, required(expression, "countDistinct"));
    }

    /**
     * Counts rows, including those where every projected value is null.
     *
     * <p>Emits: {@code count(*)}
     *
     * @return the aggregate
     */
    public static Expr countAll() {
        return new Ast.FunctionExpr("count", false, true, null);
    }

    /**
     * Gathers values into a list.
     *
     * <p>Emits: {@code collect(<expression>)}
     *
     * @param expression what to gather
     * @return the aggregate
     */
    public static Expr collect(Expr expression) {
        return new Ast.FunctionExpr("collect", false, false, required(expression, "collect"));
    }

    /**
     * Gathers distinct values into a list.
     *
     * <p>Emits: {@code collect(DISTINCT <expression>)}
     *
     * @param expression what to gather
     * @return the aggregate
     */
    public static Expr collectDistinct(Expr expression) {
        return new Ast.FunctionExpr("collect", true, false, required(expression, "collectDistinct"));
    }

    /**
     * Sums numeric values.
     *
     * <p>Emits: {@code sum(<expression>)}
     *
     * @param expression what to sum
     * @return the aggregate
     */
    public static Expr sum(Expr expression) {
        return new Ast.FunctionExpr("sum", false, false, required(expression, "sum"));
    }

    /**
     * Averages numeric values.
     *
     * <p>Emits: {@code avg(<expression>)}
     *
     * @param expression what to average
     * @return the aggregate
     */
    public static Expr avg(Expr expression) {
        return new Ast.FunctionExpr("avg", false, false, required(expression, "avg"));
    }

    /**
     * The smallest value.
     *
     * <p>Emits: {@code min(<expression>)}
     *
     * @param expression what to reduce
     * @return the aggregate
     */
    public static Expr min(Expr expression) {
        return new Ast.FunctionExpr("min", false, false, required(expression, "min"));
    }

    /**
     * The largest value.
     *
     * <p>Emits: {@code max(<expression>)}
     *
     * @param expression what to reduce
     * @return the aggregate
     */
    public static Expr max(Expr expression) {
        return new Ast.FunctionExpr("max", false, false, required(expression, "max"));
    }

    /**
     * Conjunction of two or more predicates.
     *
     * <p>Emits: {@code <predicate> AND <predicate>[ AND …]}
     *
     * @param predicates the operands; at least two
     * @return the conjunction
     */
    public static Condition and(Condition... predicates) {
        return new Ast.And(operands(predicates, "and"));
    }

    /**
     * Disjunction of two or more predicates.
     *
     * <p>Emits: {@code <predicate> OR <predicate>[ OR …]}
     *
     * @param predicates the operands; at least two
     * @return the disjunction
     */
    public static Condition or(Condition... predicates) {
        return new Ast.Or(operands(predicates, "or"));
    }

    /**
     * Negation. The operand is always parenthesised, so the negation binds exactly what it was
     * given.
     *
     * <p>Emits: {@code NOT (<predicate>)}
     *
     * @param predicate the operand
     * @return the negation
     */
    public static Condition not(Condition predicate) {
        if (predicate == null) {
            throw new IllegalArgumentException("not() needs a predicate");
        }
        return new Ast.Not(predicate);
    }

    private static Expr required(Expr expression, String function) {
        if (expression == null) {
            throw new IllegalArgumentException(function + "() needs an expression");
        }
        return expression;
    }

    private static List<Condition> operands(Condition[] predicates, String function) {
        if (predicates == null || predicates.length < 2) {
            throw new IllegalArgumentException(function + "() needs at least two predicates");
        }
        List<Condition> operands = new ArrayList<>(predicates.length);
        for (Condition predicate : predicates) {
            if (predicate == null) {
                throw new IllegalArgumentException(function + "() may not take a null predicate");
            }
            operands.add(predicate);
        }
        return List.copyOf(operands);
    }
}
