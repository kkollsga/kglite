//! Planner annotations applied after structural rewrites.

use super::{is_aggregate_expression, Clause, CypherQuery, Expression, PassCtx, ReturnItem};
use crate::graph::core::pattern_matching::Pattern;
use crate::graph::core::pattern_matching::PatternElement;
use crate::graph::languages::cypher::ast::{CaseCondition, MapProjectionItem, Predicate};
use crate::graph::schema::DirGraph;
use std::collections::HashSet;

/// **Pass:** `mark_fast_var_length_paths` — When a variable-length
/// edge `[:T*min..N]` has `min <= 1`, no path assignment and no edge
/// variable, AND *its own* clause's next RETURN/WITH is `DISTINCT` or
/// composed of dedup-safe aggregates
/// (`min/max/count(DISTINCT)/collect(DISTINCT)`), mark
/// `needs_path_info=false` so the executor uses a fast BFS with
/// global target-node dedup. Both gates are correctness, not
/// heuristics: the fast BFS answers distance reachability rather than
/// Cypher's trail reachability (equivalent only for `min <= 1`), and
/// row count is implicit path count, so dedup-by-target silently drops
/// rows when the consumer is a plain per-path projection like
/// `RETURN q.name`. WHY-BAIL: anything else stays on the slow per-
/// path BFS — correct, just not as fast.
pub(super) fn pass_mark_fast_var_length_paths(query: &mut CypherQuery, _ctx: &PassCtx) {
    mark_fast_var_length_paths(query)
}

/// **Pass:** `mark_disjoint_fixed_trails` — When a MATCH has one
/// unassigned fixed-length pattern whose relationship type sets are
/// pairwise disjoint, mark its edges `needs_path_info=false`. A relationship
/// cannot occur twice when every hop accepts a different type, so retaining
/// and cloning the exact trail cannot affect Cypher's relationship-uniqueness
/// rule. WHY-BAIL: path assignments, comma patterns, variable-length,
/// untyped, or overlapping-type edges keep full trail tracking.
pub(super) fn pass_mark_disjoint_fixed_trails(query: &mut CypherQuery, _ctx: &PassCtx) {
    mark_disjoint_fixed_trails(query)
}

/// **Pass:** `mark_skip_target_type_check` — When connection-type
/// metadata guarantees an edge's target node type, mark the edge as
/// `skip_target_type_check=true` so the executor doesn't redundantly
/// re-verify the type during BFS. Saves one slab dereference per
/// visited node.
pub(super) fn pass_mark_skip_target_type_check(query: &mut CypherQuery, ctx: &PassCtx) {
    mark_skip_target_type_check(query, ctx.graph)
}

/// Mark variable-length edges that don't need path tracking.
///
/// The executor's fast expansion answers *distance* reachability with one
/// global visited set. Cypher asks for *trail* reachability. Marking an edge
/// swaps one for the other, so the mark is legal only where the two relations
/// provably coincide:
///
/// 1. **`min_hops <= 1`.** For `min_hops >= 2` walk, trail and distance
///    reachability all differ — on a directed triangle
///    `(a)-[:R*2..2]-(b)` is trail-reachable to both peers and
///    distance-reachable to neither — and no set-based computation closes
///    the gap. (Source inclusion inside the `min_hops <= 1` window is the
///    executor's job; see `expand_var_length_fast`.)
/// 2. **This clause's own consumer collapses row multiplicity.** Row count is
///    an implicit path count in Cypher, so dedup-by-target silently drops rows
///    unless the consumer is `DISTINCT` or made of multiplicity-invariant
///    aggregates. The proof must be redone per MATCH: a query can open with
///    `WITH DISTINCT b` and end in a plain projection, and proving only the
///    first consumer applied that `DISTINCT` to every later clause too
///    (`MATCH …*1..2 … WITH DISTINCT b MATCH (b)-[…*1..2]->(c) RETURN c.id`
///    returned 2 rows where the graph has 3).
/// 3. The clause has no path assignment and the edge has no variable — either
///    one needs the exact relationship sequence.
///
/// Caught by `tests/test_cypher_differential.py::var_length_no_var`,
/// which previously xfail'd because the un-gated fast path returned
/// 2 rows where Neo4j semantics demand 3, and by the cyclic fixtures added
/// alongside this gate.
fn mark_fast_var_length_paths(query: &mut CypherQuery) {
    let clause_count = query.clauses.len();
    let consumer_safe: Vec<bool> = (0..clause_count)
        .map(|idx| {
            matches!(
                query.clauses[idx],
                Clause::Match(_) | Clause::OptionalMatch(_)
            ) && consumer_is_dedup_safe(&query.clauses, idx)
        })
        .collect();

    for (idx, clause) in query.clauses.iter_mut().enumerate() {
        // An `EXISTS { … }` subquery is dedup-safe wherever it appears: its
        // consumer is one boolean, so no dropped duplicate is observable. It
        // is reached from any clause, not only a dedup-safe MATCH.
        for_each_exists_subquery(clause, &mut |patterns| {
            // A subquery has no path assignments to preserve — the grammar
            // gives it patterns and an optional WHERE, nothing else.
            mark_private_var_length_edges(patterns)
        });

        if !consumer_safe[idx] {
            continue;
        }
        let mc = match clause {
            Clause::Match(mc) | Clause::OptionalMatch(mc) => mc,
            _ => continue,
        };

        // If there are path assignments, path info is needed for all patterns
        if !mc.path_assignments.is_empty() {
            continue;
        }

        mark_private_var_length_edges(&mut mc.patterns);
    }
}

/// Mark every variable-length edge in one relationship-uniqueness scope whose
/// exact trail nothing can observe.
///
/// The per-edge gates are `min_hops <= 1` and "no relationship variable" (both
/// documented on [`mark_fast_var_length_paths`]). The third gate is about the
/// *scope*: openCypher forbids one relationship occurring twice across the
/// whole scope, and the executor enforces that by comparing the trails each
/// segment recorded. A segment that drops its trail is invisible to that
/// check, so a sibling edge is free to re-bind a relationship the segment
/// already walked — `(a)-[:R*1..2]->(x)-[:R]->(c)` on a 2-cycle answered
/// `[1, 2]` where trail semantics give `[1]`. So a segment may drop its trail
/// only when no sibling edge in the scope can bind a relationship it binds:
/// either it is the scope's only edge, or every edge in the scope is typed and
/// this one's types are disjoint from all the others'.
fn mark_private_var_length_edges(patterns: &mut [Pattern]) {
    // Phase 1 — the type set of every edge in the scope, in element order.
    let type_sets: Vec<Option<Vec<String>>> = patterns
        .iter()
        .flat_map(|pattern| pattern.elements.iter())
        .filter_map(|element| match element {
            PatternElement::Edge(ep) => Some(segment_types(ep)),
            PatternElement::Node(_) => None,
        })
        .collect();

    // Phase 2 — mark, skipping each edge's own entry.
    let mut position = 0usize;
    for pattern in patterns.iter_mut() {
        for element in &mut pattern.elements {
            let PatternElement::Edge(ep) = element else {
                continue;
            };
            let here = position;
            position += 1;
            if ep.variable.is_some() || !ep.var_length.is_some_and(|(min, _)| min <= 1) {
                continue;
            }
            if relationships_are_private(&type_sets, here) {
                ep.needs_path_info = false;
            }
        }
    }
}

/// The relationship types a segment accepts, or `None` when it accepts any
/// (an untyped segment, or the degenerate empty alternation).
fn segment_types(edge: &crate::graph::core::pattern_matching::EdgePattern) -> Option<Vec<String>> {
    match &edge.connection_types {
        Some(types) if !types.is_empty() => Some(types.clone()),
        Some(_) => None,
        None => edge.connection_type.as_ref().map(|ty| vec![ty.clone()]),
    }
}

/// Whether the edge at `here` can share a relationship with any sibling.
fn relationships_are_private(type_sets: &[Option<Vec<String>>], here: usize) -> bool {
    if type_sets.len() == 1 {
        return true;
    }
    let Some(mine) = type_sets[here].as_ref() else {
        // Untyped: it accepts whatever a sibling accepts.
        return false;
    };
    type_sets.iter().enumerate().all(|(other, types)| {
        other == here
            || types
                .as_ref()
                .is_some_and(|theirs| theirs.iter().all(|ty| !mine.contains(ty)))
    })
}

/// Apply `visit` to the pattern list of every `EXISTS { … }` subquery
/// reachable from this clause.
///
/// `EXISTS` parses to [`Predicate::Exists`] wherever it is written — a bare
/// pattern predicate in `WHERE`, `NOT EXISTS`, a projected `RETURN EXISTS {…}`
/// or a `CASE WHEN EXISTS {…}` all land on that one node — so one walk over
/// predicates and expressions covers every spelling.
///
/// `COUNT { … }` is deliberately NOT visited: its consumer is a row count, so
/// nothing about it is dedup-safe. Write clauses are not visited either; the
/// walk stays on the read surface where the annotation is worth having.
fn for_each_exists_subquery(clause: &mut Clause, visit: &mut impl FnMut(&mut [Pattern])) {
    match clause {
        Clause::Match(mc) | Clause::OptionalMatch(mc) => {
            if let Some(wc) = &mut mc.where_clause {
                walk_predicate(&mut wc.predicate, visit);
            }
        }
        Clause::Where(wc) => walk_predicate(&mut wc.predicate, visit),
        Clause::With(wc) => {
            for item in &mut wc.items {
                walk_expression(&mut item.expression, visit);
            }
            if let Some(inner) = &mut wc.where_clause {
                walk_predicate(&mut inner.predicate, visit);
            }
        }
        Clause::Return(rc) => {
            for item in &mut rc.items {
                walk_expression(&mut item.expression, visit);
            }
            if let Some(having) = &mut rc.having {
                walk_predicate(having, visit);
            }
        }
        Clause::OrderBy(ob) => {
            for item in &mut ob.items {
                walk_expression(&mut item.expression, visit);
            }
        }
        Clause::Unwind(uc) => walk_expression(&mut uc.expression, visit),
        _ => {}
    }
}

/// Recurse into every `EXISTS { … }` under `pred`.
fn walk_predicate(pred: &mut Predicate, visit: &mut impl FnMut(&mut [Pattern])) {
    match pred {
        Predicate::Exists {
            patterns,
            where_clause,
            ..
        } => {
            visit(patterns);
            if let Some(inner) = where_clause {
                walk_predicate(inner, visit);
            }
        }
        Predicate::And(a, b) | Predicate::Or(a, b) | Predicate::Xor(a, b) => {
            walk_predicate(a, visit);
            walk_predicate(b, visit);
        }
        Predicate::Not(inner) => walk_predicate(inner, visit),
        Predicate::Comparison { left, right, .. }
        | Predicate::StartsWith {
            expr: left,
            pattern: right,
        }
        | Predicate::EndsWith {
            expr: left,
            pattern: right,
        }
        | Predicate::Contains {
            expr: left,
            pattern: right,
        }
        | Predicate::InExpression {
            expr: left,
            list_expr: right,
        } => {
            walk_expression(left, visit);
            walk_expression(right, visit);
        }
        Predicate::IsNull(expr)
        | Predicate::IsNotNull(expr)
        | Predicate::InLiteralSet { expr, .. } => walk_expression(expr, visit),
        Predicate::In { expr, list } => {
            walk_expression(expr, visit);
            for item in list {
                walk_expression(item, visit);
            }
        }
        Predicate::LabelCheck { .. } => {}
    }
}

/// Recurse into every `EXISTS { … }` under `expr`.
///
/// Exhaustive on purpose — no wildcard arm — so a new `Expression` variant
/// that can carry a predicate has to be considered here rather than silently
/// dropping out of the annotation.
fn walk_expression(expr: &mut Expression, visit: &mut impl FnMut(&mut [Pattern])) {
    match expr {
        Expression::PredicateExpr(pred) => walk_predicate(pred, visit),
        Expression::Add(a, b)
        | Expression::Subtract(a, b)
        | Expression::Multiply(a, b)
        | Expression::Divide(a, b)
        | Expression::Modulo(a, b)
        | Expression::Concat(a, b)
        | Expression::IndexAccess { expr: a, index: b } => {
            walk_expression(a, visit);
            walk_expression(b, visit);
        }
        Expression::Negate(inner)
        | Expression::IsNull(inner)
        | Expression::IsNotNull(inner)
        | Expression::ExprPropertyAccess { expr: inner, .. } => walk_expression(inner, visit),
        Expression::FunctionCall { args, .. } => {
            for arg in args {
                walk_expression(arg, visit);
            }
        }
        Expression::ListLiteral(items) => {
            for item in items {
                walk_expression(item, visit);
            }
        }
        Expression::MapLiteral(entries) => {
            for (_, value) in entries {
                walk_expression(value, visit);
            }
        }
        Expression::MapProjection { items, .. } => {
            for item in items {
                if let MapProjectionItem::Alias { expr, .. } = item {
                    walk_expression(expr, visit);
                }
            }
        }
        Expression::Case {
            operand,
            when_clauses,
            else_expr,
        } => {
            if let Some(operand) = operand {
                walk_expression(operand, visit);
            }
            for (condition, result) in when_clauses {
                match condition {
                    CaseCondition::Predicate(pred) => walk_predicate(pred, visit),
                    CaseCondition::Expression(expr) => walk_expression(expr, visit),
                }
                walk_expression(result, visit);
            }
            if let Some(else_expr) = else_expr {
                walk_expression(else_expr, visit);
            }
        }
        Expression::ListComprehension {
            list_expr,
            filter,
            map_expr,
            ..
        } => {
            walk_expression(list_expr, visit);
            if let Some(filter) = filter {
                walk_predicate(filter, visit);
            }
            if let Some(map_expr) = map_expr {
                walk_expression(map_expr, visit);
            }
        }
        Expression::ListSlice { expr, start, end } => {
            walk_expression(expr, visit);
            if let Some(start) = start {
                walk_expression(start, visit);
            }
            if let Some(end) = end {
                walk_expression(end, visit);
            }
        }
        Expression::QuantifiedList {
            list_expr, filter, ..
        } => {
            walk_expression(list_expr, visit);
            walk_predicate(filter, visit);
        }
        Expression::Reduce {
            init,
            list_expr,
            body,
            ..
        } => {
            walk_expression(init, visit);
            walk_expression(list_expr, visit);
            walk_expression(body, visit);
        }
        Expression::WindowFunction {
            partition_by,
            order_by,
            ..
        } => {
            for expr in partition_by {
                walk_expression(expr, visit);
            }
            for item in order_by {
                walk_expression(&mut item.expression, visit);
            }
        }
        // `count { … }` counts rows, so its var-length segments must keep
        // their trails — the row multiplicity IS the answer.
        Expression::CountSubquery { where_clause, .. } => {
            if let Some(inner) = where_clause {
                walk_predicate(inner, visit);
            }
        }
        Expression::PropertyAccess { .. }
        | Expression::Variable(_)
        | Expression::Literal(_)
        | Expression::Star
        | Expression::Parameter(_) => {}
    }
}

/// Whether the clause at `idx` feeds a projection that collapses row
/// multiplicity.
///
/// Walks forward to the first `WITH`/`RETURN` — that clause is this MATCH's
/// consumer — and proves dedup-safety on it. Anything on the way that reads
/// row multiplicity itself (a write clause creating one node per row, a
/// `CALL`, a `UNION` arm, a pre-fused aggregate) is a barrier: the rows the
/// fast path would drop are observable before the projection ever runs.
fn consumer_is_dedup_safe(clauses: &[Clause], idx: usize) -> bool {
    for clause in &clauses[idx + 1..] {
        match clause {
            Clause::Return(r) => return projection_is_dedup_safe(r.distinct, &r.items),
            Clause::With(w) => return projection_is_dedup_safe(w.distinct, &w.items),
            // Row-count-blind pass-throughs: they reshape or filter rows but
            // never turn a dropped duplicate into a different answer.
            Clause::Match(_)
            | Clause::OptionalMatch(_)
            | Clause::Where(_)
            | Clause::Unwind(_)
            | Clause::OrderBy(_) => continue,
            _ => return false,
        }
    }
    false
}

/// Returns true iff this projection collapses row multiplicity, so the fast
/// var-length BFS's dedup-by-target cannot change the answer.
///
/// Two safe cases:
/// - `DISTINCT` — row tuples are deduped at projection anyway, so a fast-path
///   target-dedup is consistent. Not when an item *is* multiplicity-sensitive
///   in its own right: `WITH DISTINCT a, count(b)` groups before it dedups, so
///   `count(b)` is a per-group row count and moves with the dropped rows.
/// - Every item is a multiplicity-invariant aggregate: `min`/`max`,
///   `count(DISTINCT _)`, `collect(DISTINCT _)`. Plain `count(*)` over
///   var-length matches counts paths, not targets, so it is rejected.
///
/// Conservative anywhere else: we'd rather skip the optimization than
/// silently drop rows.
fn projection_is_dedup_safe(distinct: bool, items: &[ReturnItem]) -> bool {
    if items.is_empty() {
        return false;
    }
    if distinct {
        return items.iter().all(|item| {
            !is_aggregate_expression(&item.expression)
                || is_distinct_safe_aggregate(&item.expression)
        });
    }
    items
        .iter()
        .all(|item| is_distinct_safe_aggregate(&item.expression))
}

/// Remove fixed-trail bookkeeping when relationship reuse is impossible by type.
fn mark_disjoint_fixed_trails(query: &mut CypherQuery) {
    for clause in &mut query.clauses {
        let mc = match clause {
            Clause::Match(mc) | Clause::OptionalMatch(mc) => mc,
            _ => continue,
        };
        if !mc.path_assignments.is_empty() || mc.patterns.len() != 1 {
            continue;
        }

        let pattern = &mut mc.patterns[0];
        if !fixed_edge_types_are_pairwise_disjoint(pattern) {
            continue;
        }
        for element in &mut pattern.elements {
            if let PatternElement::Edge(edge) = element {
                edge.needs_path_info = false;
            }
        }
    }
}

/// True when every edge is fixed-length, typed, and accepts no type accepted
/// by any other edge in the same pattern.
pub(super) fn fixed_edge_types_are_pairwise_disjoint(
    pattern: &crate::graph::core::pattern_matching::Pattern,
) -> bool {
    let mut seen = HashSet::new();
    let mut edge_count = 0usize;

    for element in &pattern.elements {
        let PatternElement::Edge(edge) = element else {
            continue;
        };
        edge_count += 1;
        if edge.var_length.is_some() {
            return false;
        }

        if let Some(types) = &edge.connection_types {
            if types.is_empty() || types.iter().any(|ty| !seen.insert(ty.as_str())) {
                return false;
            }
        } else if let Some(ty) = &edge.connection_type {
            if !seen.insert(ty.as_str()) {
                return false;
            }
        } else {
            return false;
        }
    }

    edge_count > 0
}

/// True when an expression is an aggregate that's invariant to row
/// multiplicity: `count(DISTINCT _)`, `min/max(_)`, `collect(DISTINCT _)`.
/// Plain `count(_)` and `sum(_)` would shift with row count, so they
/// don't qualify.
fn is_distinct_safe_aggregate(expr: &Expression) -> bool {
    if let Expression::FunctionCall {
        name,
        args: _,
        distinct,
    } = expr
    {
        let nm = name.to_lowercase();
        if matches!(nm.as_str(), "min" | "max") {
            return true;
        }
        if *distinct && matches!(nm.as_str(), "count" | "collect") {
            return true;
        }
    }
    false
}

/// Skip node type checks when the connection type metadata guarantees the target type.
///
/// For a pattern like `(a:Person)-[:AUTHORED]->(b:Paper)`, if `AUTHORED` edges
/// only ever connect Person→Paper, then checking `node_weight(target).node_type`
/// in the BFS inner loop is redundant. This saves one `StableDiGraph` slab
/// dereference per visited node.
fn mark_skip_target_type_check(query: &mut CypherQuery, graph: &DirGraph) {
    use crate::graph::core::pattern_matching::EdgeDirection;

    for clause in &mut query.clauses {
        let mc = match clause {
            Clause::Match(mc) | Clause::OptionalMatch(mc) => mc,
            _ => continue,
        };

        for pattern in &mut mc.patterns {
            let elements = &mut pattern.elements;
            // Walk elements in triples: Node, Edge, Node
            let len = elements.len();
            for i in 0..len {
                if i + 2 >= len {
                    break;
                }
                // Extract edge and target node info without overlapping borrows
                let (conn_types, direction, target_node_type) = {
                    let edge = match &elements[i + 1] {
                        PatternElement::Edge(ep) => ep,
                        _ => continue,
                    };
                    let target = match &elements[i + 2] {
                        PatternElement::Node(np) => np,
                        _ => continue,
                    };
                    // The connection-type guarantee covers only the target's
                    // PRIMARY type. If the pattern also carries secondary
                    // labels (`(b:Type:Extra)`), skipping the check would drop
                    // the `:Extra` filter — never skip in that case.
                    if !target.extra_labels.is_empty() {
                        continue;
                    }
                    // An alternation's guarantee is the conjunction over its
                    // branches. Reading only `connection_type` took the
                    // guarantee of the FIRST branch and applied it to all of
                    // them, so `[:KNOWS|WORKS_AT]->(x:Person)` skipped the
                    // label check on WORKS_AT's Company targets and returned
                    // them as Persons.
                    let types: Vec<String> = match &edge.connection_types {
                        Some(types) if !types.is_empty() => types.clone(),
                        _ => match &edge.connection_type {
                            Some(ct) => vec![ct.clone()],
                            None => continue,
                        },
                    };
                    match &target.node_type {
                        Some(nt) => (types, edge.direction, nt.clone()),
                        None => continue,
                    }
                };

                // Every accepted connection type must guarantee the target's
                // primary label; one unknown or non-guaranteeing branch and
                // the check has to stay.
                let guaranteed = conn_types.iter().all(|conn_type| {
                    graph.connection_type_metadata.get(conn_type).is_some_and(
                        |info| match direction {
                            EdgeDirection::Outgoing => {
                                info.target_types.len() == 1
                                    && info.target_types.contains(&target_node_type)
                            }
                            EdgeDirection::Incoming => {
                                info.source_types.len() == 1
                                    && info.source_types.contains(&target_node_type)
                            }
                            EdgeDirection::Both => false, // can't guarantee for bidirectional
                        },
                    )
                });
                if guaranteed {
                    if let PatternElement::Edge(ep) = &mut elements[i + 1] {
                        ep.skip_target_type_check = true;
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{fixed_edge_types_are_pairwise_disjoint, mark_fast_var_length_paths};
    use crate::graph::core::pattern_matching::{parse_pattern, PatternElement};
    use crate::graph::languages::cypher::parser::parse_cypher;

    /// `needs_path_info` of every variable-length edge, in clause order.
    fn var_length_marks(query: &str) -> Vec<bool> {
        let mut parsed = parse_cypher(query).unwrap_or_else(|e| panic!("{query}: {e}"));
        mark_fast_var_length_paths(&mut parsed);
        let mut marks = Vec::new();
        for clause in &parsed.clauses {
            let mc = match clause {
                super::Clause::Match(mc) | super::Clause::OptionalMatch(mc) => mc,
                _ => continue,
            };
            for pattern in &mc.patterns {
                for element in &pattern.elements {
                    if let PatternElement::Edge(ep) = element {
                        if ep.var_length.is_some() {
                            marks.push(ep.needs_path_info);
                        }
                    }
                }
            }
        }
        marks
    }

    #[test]
    fn dedup_safe_consumer_marks_a_min_one_segment() {
        // `false` == marked fast.
        assert_eq!(
            var_length_marks("MATCH (a:N)-[:R*1..3]->(b:N) RETURN DISTINCT b.id"),
            vec![false]
        );
        assert_eq!(
            var_length_marks("MATCH (a:N)-[:R*0..3]->(b:N) RETURN count(DISTINCT b) AS n"),
            vec![false]
        );
    }

    #[test]
    fn min_hops_of_two_or_more_is_never_marked() {
        // The fast BFS answers distance reachability; for `min >= 2` that is
        // a different relation from Cypher's trail reachability, and no
        // downstream DISTINCT makes it the same one.
        for query in [
            "MATCH (a:N)-[:R*2..2]->(b:N) RETURN DISTINCT b.id",
            "MATCH (a:N)-[:R*2..3]-(b:N) RETURN count(DISTINCT b) AS n",
            "MATCH (a:N)-[:R*3..3]->(b:N) RETURN DISTINCT b.id",
            "MATCH (a:N)-[:R*5]->(b:N) RETURN DISTINCT b.id",
        ] {
            assert_eq!(var_length_marks(query), vec![true], "{query}");
        }
    }

    #[test]
    fn each_clause_is_proved_against_its_own_consumer() {
        // The first consumer's DISTINCT does not license the second clause,
        // whose own consumer is a plain per-path projection. Marking both
        // returned 2 rows where the graph has 3.
        assert_eq!(
            var_length_marks(
                "MATCH (a:N)-[:R*1..2]->(b:N) WITH DISTINCT b                  MATCH (b)-[:R*1..2]->(c:N) RETURN c.id"
            ),
            vec![false, true]
        );
        // ... and the converse: a later clause with a dedup-safe consumer is
        // marked even though the query's first projection is not one.
        assert_eq!(
            var_length_marks(
                "MATCH (a:N)-[:R*1..2]->(b:N) WITH b                  MATCH (b)-[:R*1..2]->(c:N) RETURN DISTINCT c.id"
            ),
            vec![true, false]
        );
    }

    #[test]
    fn distinct_over_a_row_counting_aggregate_is_not_dedup_safe() {
        // `WITH/RETURN DISTINCT a, count(b)` groups before it dedups, so
        // `count(b)` is a per-group row count: dropping duplicate targets
        // changes it (3 -> 2 on a triangle).
        assert_eq!(
            var_length_marks("MATCH (a:N)-[:R*1..3]->(b:N) RETURN DISTINCT a.id, count(b) AS n"),
            vec![true]
        );
        assert_eq!(
            var_length_marks(
                "MATCH (a:N)-[:R*1..3]->(b:N) RETURN DISTINCT a.id, count(DISTINCT b) AS n"
            ),
            vec![false]
        );
    }

    #[test]
    fn a_write_between_the_match_and_its_projection_is_a_barrier() {
        // One CREATE per row: the rows the fast path would drop are
        // observable as missing nodes long before the RETURN dedups.
        assert_eq!(
            var_length_marks(
                "MATCH (a:N)-[:R*1..2]->(b:N) CREATE (:Log {t: b.id}) RETURN DISTINCT b.id"
            ),
            vec![true]
        );
    }

    #[test]
    fn a_path_assignment_or_edge_variable_keeps_the_exact_trail() {
        assert_eq!(
            var_length_marks("MATCH p = (a:N)-[:R*1..2]->(b:N) RETURN DISTINCT b.id"),
            vec![true]
        );
        assert_eq!(
            var_length_marks("MATCH (a:N)-[r:R*1..2]->(b:N) RETURN DISTINCT b.id"),
            vec![true]
        );
    }

    /// `needs_path_info` of every variable-length edge inside an
    /// `EXISTS { … }` subquery, in the order the walk reaches them.
    fn exists_var_length_marks(query: &str) -> Vec<bool> {
        let mut parsed = parse_cypher(query).unwrap_or_else(|e| panic!("{query}: {e}"));
        mark_fast_var_length_paths(&mut parsed);
        let mut marks = Vec::new();
        for clause in &mut parsed.clauses {
            super::for_each_exists_subquery(clause, &mut |patterns| {
                for pattern in patterns.iter() {
                    for element in &pattern.elements {
                        if let PatternElement::Edge(ep) = element {
                            if ep.var_length.is_some() {
                                marks.push(ep.needs_path_info);
                            }
                        }
                    }
                }
            });
        }
        marks
    }

    #[test]
    fn an_exists_subquery_is_dedup_safe_wherever_it_is_written() {
        // The consumer is one boolean, so no dropped duplicate is observable
        // — and that holds for every spelling EXISTS has. `false` == marked.
        for query in [
            "MATCH (a:N) WHERE EXISTS { (a)-[:R*1..3]->(:N) } RETURN a.id",
            "MATCH (a:N) WHERE NOT EXISTS { (a)-[:R*1..3]->(:N) } RETURN a.id",
            "MATCH (a:N) RETURN EXISTS { (a)-[:R*1..3]->(:N) } AS reachable",
            "MATCH (a:N) RETURN CASE WHEN EXISTS { (a)-[:R*1..3]->(:N) } THEN 1 ELSE 0 END AS r",
            "MATCH (a:N) WITH a WHERE EXISTS { (a)-[:R*0..3]->(:N) } RETURN a.id",
            "MATCH (a:N) WHERE EXISTS { (a)-[:R*1..3]->(:N) } AND a.id > 1 RETURN a.id",
            "MATCH (a:N) OPTIONAL MATCH (a)-[:S]->(b) WHERE EXISTS { (b)-[:R*1..2]->(:N) } RETURN a.id",
        ] {
            assert_eq!(exists_var_length_marks(query), vec![false], "{query}");
        }
    }

    #[test]
    fn an_exists_subquery_keeps_the_gates_that_are_about_correctness() {
        // `min >= 2` is not distance-equivalent, an edge variable needs the
        // exact trail, and `count { … }` counts rows so nothing about it is
        // dedup-safe.
        for query in [
            "MATCH (a:N) WHERE EXISTS { (a)-[:R*2..3]->(:N) } RETURN a.id",
            "MATCH (a:N) WHERE EXISTS { (a)-[r:R*1..3]->(:N) } RETURN a.id",
        ] {
            assert_eq!(exists_var_length_marks(query), vec![true], "{query}");
        }
        // COUNT { … } is not walked at all: its segments keep their trails.
        let mut parsed =
            parse_cypher("MATCH (a:N) RETURN COUNT { (a)-[:R*1..3]->(:N) } AS n").unwrap();
        mark_fast_var_length_paths(&mut parsed);
        assert!(count_subquery_var_length_marks(&parsed)
            .iter()
            .all(|marked| *marked));
    }

    fn count_subquery_var_length_marks(query: &super::CypherQuery) -> Vec<bool> {
        use super::Expression;
        let mut marks = Vec::new();
        for clause in &query.clauses {
            let super::Clause::Return(rc) = clause else {
                continue;
            };
            for item in &rc.items {
                if let Expression::CountSubquery { patterns, .. } = &item.expression {
                    for pattern in patterns {
                        for element in &pattern.elements {
                            if let PatternElement::Edge(ep) = element {
                                if ep.var_length.is_some() {
                                    marks.push(ep.needs_path_info);
                                }
                            }
                        }
                    }
                }
            }
        }
        assert!(!marks.is_empty(), "no COUNT subquery segment found");
        marks
    }

    #[test]
    fn a_segment_sharing_relationships_with_a_sibling_keeps_its_trail() {
        // Dropping the trail hides the segment's relationships from the
        // clause's uniqueness check, so a sibling edge may re-bind one:
        // `(a)-[:R*1..2]->(x)-[:R]->(c)` on a 2-cycle answered [1, 2] where
        // trail semantics give [1].
        assert_eq!(
            var_length_marks("MATCH (a:N)-[:R*1..2]->(x)-[:R]->(c) RETURN DISTINCT c.id"),
            vec![true]
        );
        // An untyped sibling can bind anything, so the same applies.
        assert_eq!(
            var_length_marks("MATCH (a:N)-[:R*1..2]->(x)-->(c) RETURN DISTINCT c.id"),
            vec![true]
        );
        // ... and an untyped segment can be bound by anything.
        assert_eq!(
            var_length_marks("MATCH (a:N)-[*1..2]->(x)-[:R]->(c) RETURN DISTINCT c.id"),
            vec![true]
        );
        // Comma patterns share one uniqueness scope too.
        assert_eq!(
            var_length_marks("MATCH (a:N)-[:R*1..2]->(x), (y)-[:R]->(c) RETURN DISTINCT c.id"),
            vec![true]
        );
        // Disjoint types cannot collide: the segment is still marked.
        assert_eq!(
            var_length_marks("MATCH (a:N)-[:R*1..2]->(x)-[:S]->(c) RETURN DISTINCT c.id"),
            vec![false]
        );
        // A lone untyped segment has no sibling to collide with.
        assert_eq!(
            var_length_marks("MATCH (a:N)-[*1..2]->(c) RETURN DISTINCT c.id"),
            vec![false]
        );
    }

    #[test]
    fn disjoint_fixed_edge_types_need_no_trail() {
        let pattern = parse_pattern("(a)-[:JUDGED_BY]-(b)-[:CITES]->(c)").unwrap();
        assert!(fixed_edge_types_are_pairwise_disjoint(&pattern));

        let single = parse_pattern("(a)-[:CITES]->(b)").unwrap();
        assert!(fixed_edge_types_are_pairwise_disjoint(&single));
    }

    #[test]
    fn overlapping_or_unbounded_edge_types_keep_trail() {
        for text in [
            "(a)-[:CITES]->(b)-[:CITES]->(c)",
            "(a)-[:CITES|REFERS_TO]->(b)-[:REFERS_TO]->(c)",
            "(a)-->(b)-[:CITES]->(c)",
            "(a)-[:CITES*1..2]->(b)-[:REFERS_TO]->(c)",
        ] {
            let pattern = parse_pattern(text).unwrap();
            assert!(!fixed_edge_types_are_pairwise_disjoint(&pattern), "{text}");
        }
    }
}
