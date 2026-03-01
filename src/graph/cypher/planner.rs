// src/graph/cypher/planner.rs
// Query optimizer: predicate pushdown, index hints, limit pushdown

use super::ast::*;
use crate::datatypes::values::Value;
use crate::graph::pattern_matching::{PatternElement, PropertyMatcher};
use crate::graph::schema::DirGraph;
use std::collections::HashMap;

/// Optimize a parsed Cypher query before execution.
/// Accepts query parameters so that `WHERE n.prop = $param` can be pushed
/// into MATCH patterns the same way literal equalities are.
pub fn optimize(query: &mut CypherQuery, graph: &DirGraph, params: &HashMap<String, Value>) {
    push_where_into_match(query, params);
    push_limit_into_match(query, graph);
    fuse_count_short_circuits(query);
    fuse_optional_match_aggregate(query);
    fuse_match_return_aggregate(query);
    fuse_vector_score_order_limit(query);
    fuse_order_by_top_k(query);
    reorder_predicates_by_cost(query);
    mark_fast_var_length_paths(query);
}

/// Mark variable-length edges that don't need path tracking.
///
/// When a MATCH clause has no path assignments (`p = ...`) and the edge
/// has no named variable (`[r:T*1..N]`), the full path vector is never
/// read downstream.  Setting `needs_path_info = false` lets the pattern
/// executor use a fast BFS with global dedup instead of tracking every path.
fn mark_fast_var_length_paths(query: &mut CypherQuery) {
    for clause in &mut query.clauses {
        let mc = match clause {
            Clause::Match(mc) | Clause::OptionalMatch(mc) => mc,
            _ => continue,
        };

        // If there are path assignments, path info is needed for all patterns
        if !mc.path_assignments.is_empty() {
            continue;
        }

        for pattern in &mut mc.patterns {
            for element in &mut pattern.elements {
                if let PatternElement::Edge(ep) = element {
                    if ep.var_length.is_some() && ep.variable.is_none() {
                        ep.needs_path_info = false;
                    }
                }
            }
        }
    }
}

/// Fuse MATCH + RETURN into O(1)/O(types) count short-circuits.
///
/// Detects three patterns and replaces them with specialized fused clauses:
/// - `MATCH (n) RETURN count(n)` → `FusedCountAll` (O(1))
/// - `MATCH (n) RETURN n.type, count(n)` → `FusedCountByType` (O(types))
/// - `MATCH ()-[r]->() RETURN type(r), count(*)` → `FusedCountEdgesByType` (O(E) single pass)
///
/// Any trailing ORDER BY / LIMIT clauses are left in place since they
/// operate on the tiny fused result set.
fn fuse_count_short_circuits(query: &mut CypherQuery) {
    if query.clauses.len() < 2 {
        return;
    }

    // First two clauses must be Match + Return
    let is_match_return = matches!(
        (&query.clauses[0], &query.clauses[1]),
        (Clause::Match(_), Clause::Return(_))
    );
    if !is_match_return {
        return;
    }

    let match_clause = if let Clause::Match(m) = &query.clauses[0] {
        m
    } else {
        return;
    };
    let return_clause = if let Clause::Return(r) = &query.clauses[1] {
        r
    } else {
        return;
    };

    // No DISTINCT on RETURN
    if return_clause.distinct {
        return;
    }

    // Must have exactly 1 pattern
    if match_clause.patterns.len() != 1 {
        return;
    }
    let pat = &match_clause.patterns[0];

    // ---- Pattern A: MATCH (n) RETURN count(n) / count(*) ----
    //   Also handles: MATCH (n:Type) RETURN count(n)  → FusedCountTypedNode
    if pat.elements.len() == 1 {
        let node = match &pat.elements[0] {
            PatternElement::Node(np) => np,
            _ => return,
        };
        // Cannot short-circuit with property filters
        if node.properties.is_some() {
            return;
        }

        let node_var = node.variable.as_deref();

        // Typed node count: MATCH (n:Type) RETURN count(n)
        if let Some(ref node_type) = node.node_type {
            if return_clause.items.len() == 1
                && is_count_of_var_or_star(&return_clause.items[0].expression, node_var)
            {
                let alias = return_item_column_name(&return_clause.items[0]);
                let nt = node_type.clone();
                query.clauses.drain(0..2);
                query.clauses.insert(
                    0,
                    Clause::FusedCountTypedNode {
                        node_type: nt,
                        alias,
                    },
                );
            }
            return;
        }

        if return_clause.items.len() == 1 {
            // Single item: must be count(var) or count(*)
            let item = &return_clause.items[0];
            if !is_count_of_var_or_star(&item.expression, node_var) {
                return;
            }
            let alias = return_item_column_name(item);
            // Replace Match + Return with FusedCountAll; keep trailing clauses
            query.clauses.drain(0..2);
            query.clauses.insert(0, Clause::FusedCountAll { alias });
            return;
        }

        if return_clause.items.len() == 2 {
            // Two items: one must be n.type / labels(n), the other count(var) / count(*)
            let (type_idx, count_idx) = identify_type_count_pair(&return_clause.items, node_var);
            if let Some((ti, ci)) = type_idx.zip(count_idx) {
                let type_alias = return_item_column_name(&return_clause.items[ti]);
                let count_alias = return_item_column_name(&return_clause.items[ci]);
                query.clauses.drain(0..2);
                query.clauses.insert(
                    0,
                    Clause::FusedCountByType {
                        type_alias,
                        count_alias,
                    },
                );
                return;
            }
        }
        return;
    }

    // ---- Pattern C: MATCH ()-[r]->() RETURN type(r), count(*) ----
    //   Also handles: MATCH ()-[r:Type]->() RETURN count(*)  → FusedCountTypedEdge
    if pat.elements.len() == 3 {
        let src_node = match &pat.elements[0] {
            PatternElement::Node(np) => np,
            _ => return,
        };
        let edge = match &pat.elements[1] {
            PatternElement::Edge(ep) => ep,
            _ => return,
        };
        let tgt_node = match &pat.elements[2] {
            PatternElement::Node(np) => np,
            _ => return,
        };

        // Both nodes must be anonymous/unfiltered
        if src_node.node_type.is_some()
            || src_node.properties.is_some()
            || tgt_node.node_type.is_some()
            || tgt_node.properties.is_some()
        {
            return;
        }

        // Edge must have no property filters or var_length
        if edge.properties.is_some() || edge.var_length.is_some() {
            return;
        }

        let edge_var = edge.variable.as_deref();

        // Sub-pattern C1: Typed edge count — MATCH ()-[r:Type]->() RETURN count(*)
        if let Some(ref edge_type) = edge.connection_type {
            if return_clause.items.len() == 1
                && is_count_of_var_or_star(&return_clause.items[0].expression, edge_var)
            {
                let alias = return_item_column_name(&return_clause.items[0]);
                let et = edge_type.clone();
                query.clauses.drain(0..2);
                query.clauses.insert(
                    0,
                    Clause::FusedCountTypedEdge {
                        edge_type: et,
                        alias,
                    },
                );
            }
            return;
        }

        // Sub-pattern C2: Untyped edge count by type — MATCH ()-[r]->() RETURN type(r), count(*)
        if return_clause.items.len() != 2 {
            return;
        }

        // Identify type(r) and count(*) / count(r)
        let (type_idx, count_idx) = identify_edge_type_count_pair(&return_clause.items, edge_var);
        if let Some((ti, ci)) = type_idx.zip(count_idx) {
            let type_alias = return_item_column_name(&return_clause.items[ti]);
            let count_alias = return_item_column_name(&return_clause.items[ci]);
            query.clauses.drain(0..2);
            query.clauses.insert(
                0,
                Clause::FusedCountEdgesByType {
                    type_alias,
                    count_alias,
                },
            );
        }
    }
}

/// Check if an expression is `count(var)`, `count(*)`, or `count()` matching the given variable.
fn is_count_of_var_or_star(expr: &Expression, node_var: Option<&str>) -> bool {
    if let Expression::FunctionCall {
        name,
        args,
        distinct,
    } = expr
    {
        if name.to_lowercase() != "count" || *distinct {
            return false;
        }
        if args.len() == 1 {
            return match &args[0] {
                Expression::Star => true,
                Expression::Variable(v) => node_var.is_some_and(|nv| v == nv),
                _ => false,
            };
        }
    }
    false
}

/// For `RETURN n.type, count(n)` — identify which item is the type accessor and which is the count.
/// Returns (type_item_index, count_item_index) or (None, None) if pattern doesn't match.
fn identify_type_count_pair(
    items: &[ReturnItem],
    node_var: Option<&str>,
) -> (Option<usize>, Option<usize>) {
    let mut type_idx = None;
    let mut count_idx = None;

    for (i, item) in items.iter().enumerate() {
        if is_count_of_var_or_star(&item.expression, node_var) {
            count_idx = Some(i);
        } else if is_node_type_accessor(&item.expression, node_var) {
            type_idx = Some(i);
        }
    }
    (type_idx, count_idx)
}

/// Check if expression is `n.type`, `n.node_type`, `n.label`, or `labels(n)`.
fn is_node_type_accessor(expr: &Expression, node_var: Option<&str>) -> bool {
    match expr {
        Expression::PropertyAccess { variable, property } => {
            let is_type_prop = matches!(property.as_str(), "type" | "node_type" | "label");
            is_type_prop && node_var.is_some_and(|nv| variable == nv)
        }
        Expression::FunctionCall { name, args, .. } => {
            if name.to_lowercase() == "labels" && args.len() == 1 {
                if let Expression::Variable(v) = &args[0] {
                    return node_var.is_some_and(|nv| v == nv);
                }
            }
            false
        }
        _ => false,
    }
}

/// For `RETURN type(r), count(*)` — identify edge type function and count.
fn identify_edge_type_count_pair(
    items: &[ReturnItem],
    edge_var: Option<&str>,
) -> (Option<usize>, Option<usize>) {
    let mut type_idx = None;
    let mut count_idx = None;

    for (i, item) in items.iter().enumerate() {
        if is_count_of_var_or_star(&item.expression, edge_var) {
            count_idx = Some(i);
        } else if is_edge_type_function(&item.expression, edge_var) {
            type_idx = Some(i);
        }
    }
    (type_idx, count_idx)
}

/// Check if expression is `type(r)`.
fn is_edge_type_function(expr: &Expression, edge_var: Option<&str>) -> bool {
    if let Expression::FunctionCall { name, args, .. } = expr {
        if name.to_lowercase() == "type" && args.len() == 1 {
            if let Expression::Variable(v) = &args[0] {
                return edge_var.is_some_and(|ev| v == ev);
            }
        }
    }
    false
}

/// Push simple equality predicates from WHERE into MATCH pattern properties.
/// This enables the pattern executor to filter during matching rather than after.
///
/// Example: MATCH (n:Person) WHERE n.age = 30
/// Becomes: MATCH (n:Person {age: 30}) (WHERE removed if fully consumed)
///
/// Also handles parameterized equalities:
/// MATCH (n:Person) WHERE n.age = $min_age  (with params = {min_age: 30})
/// Becomes: MATCH (n:Person {age: 30})
fn push_where_into_match(query: &mut CypherQuery, params: &HashMap<String, Value>) {
    let mut i = 0;
    while i + 1 < query.clauses.len() {
        let can_push = matches!(
            (&query.clauses[i], &query.clauses[i + 1]),
            (Clause::Match(_), Clause::Where(_))
        );

        if !can_push {
            i += 1;
            continue;
        }

        // Extract the WHERE predicate
        let where_pred = if let Clause::Where(w) = &query.clauses[i + 1] {
            w.predicate.clone()
        } else {
            i += 1;
            continue;
        };

        // Collect variables defined in the MATCH patterns
        let match_vars: Vec<(String, Option<String>)> = if let Clause::Match(m) = &query.clauses[i]
        {
            collect_pattern_variables(&m.patterns)
        } else {
            i += 1;
            continue;
        };

        // Split predicate into pushable conditions and remainder
        let (pushable, pushable_in, remaining) =
            extract_pushable_equalities(&where_pred, &match_vars, params);

        // Apply pushable conditions to MATCH patterns
        if !pushable.is_empty() || !pushable_in.is_empty() {
            if let Clause::Match(ref mut m) = query.clauses[i] {
                for (var_name, property, value) in &pushable {
                    apply_property_to_patterns(&mut m.patterns, var_name, property, value.clone());
                }
                for (var_name, property, values) in pushable_in {
                    apply_in_property_to_patterns(&mut m.patterns, &var_name, &property, values);
                }
            }

            // Update or remove WHERE clause
            match remaining {
                Some(pred) => {
                    query.clauses[i + 1] = Clause::Where(WhereClause { predicate: pred });
                }
                None => {
                    query.clauses.remove(i + 1);
                    continue; // Don't increment, check the new i+1
                }
            }
        }

        i += 1;
    }
}

/// Push LIMIT into MATCH when there's no ORDER BY/aggregation between them.
/// This allows the pattern executor to stop early via max_matches.
///
/// Safe when: MATCH → RETURN → LIMIT, with RETURN having no aggregation or DISTINCT.
/// The LIMIT clause is removed from the pipeline and its value is stored in
/// `MatchClause.limit_hint`, which `execute_match` passes to PatternExecutor.
fn push_limit_into_match(query: &mut CypherQuery, _graph: &DirGraph) {
    if query.clauses.len() < 3 {
        return;
    }
    let mut i = 0;
    while i + 2 < query.clauses.len() {
        // Look for MATCH → RETURN → LIMIT
        let is_pattern = matches!(
            (
                &query.clauses[i],
                &query.clauses[i + 1],
                &query.clauses[i + 2]
            ),
            (Clause::Match(_), Clause::Return(_), Clause::Limit(_))
        );
        if !is_pattern {
            i += 1;
            continue;
        }

        // Safety check: RETURN must have no aggregation and no DISTINCT
        let safe = if let Clause::Return(r) = &query.clauses[i + 1] {
            !r.distinct
                && !r
                    .items
                    .iter()
                    .any(|item| super::executor::is_aggregate_expression(&item.expression))
        } else {
            false
        };
        if !safe {
            i += 1;
            continue;
        }

        // Extract LIMIT value — must be a literal positive integer
        let limit_val = if let Clause::Limit(l) = &query.clauses[i + 2] {
            match &l.count {
                Expression::Literal(Value::Int64(n)) if *n > 0 => Some(*n as usize),
                _ => None,
            }
        } else {
            None
        };
        let Some(limit) = limit_val else {
            i += 1;
            continue;
        };

        // All checks passed: push limit into MATCH and remove LIMIT clause
        if let Clause::Match(ref mut m) = query.clauses[i] {
            m.limit_hint = Some(limit);
        }
        query.clauses.remove(i + 2); // Remove LIMIT
                                     // Don't increment — check the new i+2
    }
}

/// Fuse consecutive OPTIONAL MATCH + WITH (containing count aggregation) into
/// a single `FusedOptionalMatchAggregate` clause. This avoids materializing
/// N×M intermediate rows when only the count is needed.
///
/// Criteria for fusion:
/// 1. `clauses[i]` is `OptionalMatch` and `clauses[i+1]` is `With`
/// 2. The WITH has at least one `count(variable)` aggregate
/// 3. All non-aggregate items in the WITH are simple variable pass-throughs
/// 4. ALL count aggregate variables come from THIS OPTIONAL MATCH pattern
///    (not from earlier OPTIONAL MATCHes — otherwise the fused execution would
///    assign a single match_count to all count columns, producing wrong results)
/// 5. The count aggregates do NOT use DISTINCT (the fused fast-path counts raw
///    matches and cannot perform deduplication)
fn fuse_optional_match_aggregate(query: &mut CypherQuery) {
    let mut i = 0;
    while i + 1 < query.clauses.len() {
        let can_fuse = matches!(
            (&query.clauses[i], &query.clauses[i + 1]),
            (Clause::OptionalMatch(_), Clause::With(_))
        );

        if !can_fuse {
            i += 1;
            continue;
        }

        // Check that the WITH contains count() aggregation and simple pass-through group keys
        let fusable = if let Clause::With(w) = &query.clauses[i + 1] {
            is_fusable_with_clause(w)
        } else {
            false
        };

        if !fusable {
            i += 1;
            continue;
        }

        // Collect variables defined in the OPTIONAL MATCH pattern
        let opt_match_vars: std::collections::HashSet<String> =
            if let Clause::OptionalMatch(m) = &query.clauses[i] {
                collect_pattern_variables(&m.patterns)
                    .into_iter()
                    .map(|(name, _)| name)
                    .collect()
            } else {
                i += 1;
                continue;
            };

        // Verify ALL count aggregate variables come from THIS OPTIONAL MATCH,
        // and none use DISTINCT (which the fused path cannot handle)
        let all_counts_local = if let Clause::With(w) = &query.clauses[i + 1] {
            w.items.iter().all(|item| {
                if let Expression::FunctionCall {
                    name,
                    args,
                    distinct,
                } = &item.expression
                {
                    if name.eq_ignore_ascii_case("count") {
                        // Reject DISTINCT — fused path can't deduplicate
                        if *distinct {
                            return false;
                        }
                        // count(*) is always fine
                        if args.len() == 1 && matches!(args[0], Expression::Star) {
                            return true;
                        }
                        // count(var) — var must come from this OPTIONAL MATCH
                        if let Some(Expression::Variable(var)) = args.first() {
                            return opt_match_vars.contains(var);
                        }
                        // count(expr) — not a simple variable, bail
                        return false;
                    }
                }
                true // non-aggregate items (group keys) are fine
            })
        } else {
            false
        };

        if !all_counts_local {
            i += 1;
            continue;
        }

        // Extract both clauses and replace with fused variant
        let with_clause = if let Clause::With(w) = query.clauses.remove(i + 1) {
            w
        } else {
            unreachable!()
        };
        let match_clause = if let Clause::OptionalMatch(m) = query.clauses.remove(i) {
            m
        } else {
            unreachable!()
        };

        query.clauses.insert(
            i,
            Clause::FusedOptionalMatchAggregate {
                match_clause,
                with_clause,
            },
        );

        i += 1;
    }
}

/// Check if a WITH clause is eligible for fusion with an OPTIONAL MATCH.
/// Must have: simple variable group keys + count() aggregates only.
fn is_fusable_with_clause(with: &WithClause) -> bool {
    use super::executor::is_aggregate_expression;

    let mut has_count = false;

    for item in &with.items {
        if is_aggregate_expression(&item.expression) {
            // Only fuse for count() — not sum/collect/avg etc.
            match &item.expression {
                Expression::FunctionCall { name, .. } if name.eq_ignore_ascii_case("count") => {
                    has_count = true;
                }
                _ => return false, // Non-count aggregate → bail
            }
        } else {
            // Group key must be a simple variable pass-through
            if !matches!(&item.expression, Expression::Variable(_)) {
                return false;
            }
        }
    }

    has_count
}

/// Fuse MATCH (node-edge-node) + RETURN (group-by + count) into a single
/// pass that counts edges directly per node instead of materializing all rows.
///
/// Criteria for fusion:
/// 1. `clauses[i]` is `Match` with exactly 1 pattern of 3 elements (node-edge-node)
/// 2. `clauses[i+1]` is `Return` with at least one `count()` aggregate
/// 3. All non-aggregate RETURN items are PropertyAccess on the first node variable
/// 4. All `count()` args reference the second node variable (or `*`)
/// 5. No DISTINCT on count, no property filters on edge or second node
///    (required by `try_count_simple_pattern`)
fn fuse_match_return_aggregate(query: &mut CypherQuery) {
    use super::executor::is_aggregate_expression;

    let mut i = 0;
    while i + 1 < query.clauses.len() {
        let can_fuse = matches!(
            (&query.clauses[i], &query.clauses[i + 1]),
            (Clause::Match(_), Clause::Return(_))
        );
        if !can_fuse {
            i += 1;
            continue;
        }

        // Check MATCH: exactly 1 pattern with 3 or 5 elements
        let (first_var, second_var, edge_has_props, second_has_props) = if let Clause::Match(m) =
            &query.clauses[i]
        {
            let n_elems = m.patterns[0].elements.len();
            if m.patterns.len() != 1 || (n_elems != 3 && n_elems != 5) {
                i += 1;
                continue;
            }
            let pat = &m.patterns[0];
            let first_var = match &pat.elements[0] {
                PatternElement::Node(np) => np.variable.clone(),
                _ => {
                    i += 1;
                    continue;
                }
            };
            let edge_has_props = match &pat.elements[1] {
                PatternElement::Edge(ep) => ep.properties.is_some() || ep.var_length.is_some(),
                _ => {
                    i += 1;
                    continue;
                }
            };

            if n_elems == 5 {
                // 5-element: (a)-[e1]->(b)<-[e2]-(c)
                // Middle node (elements[2]) must have no properties
                let mid_has_props = match &pat.elements[2] {
                    PatternElement::Node(np) => np.properties.is_some(),
                    _ => {
                        i += 1;
                        continue;
                    }
                };
                let edge2_has_props = match &pat.elements[3] {
                    PatternElement::Edge(ep) => ep.properties.is_some() || ep.var_length.is_some(),
                    _ => {
                        i += 1;
                        continue;
                    }
                };
                let (last_var, last_has_props) = match &pat.elements[4] {
                    PatternElement::Node(np) => (np.variable.clone(), np.properties.is_some()),
                    _ => {
                        i += 1;
                        continue;
                    }
                };
                if mid_has_props || edge2_has_props || last_has_props {
                    i += 1;
                    continue;
                }
                (first_var, last_var, edge_has_props, false)
            } else {
                // 3-element: (a)-[e]->(b)
                let (second_var, second_has_props) = match &pat.elements[2] {
                    PatternElement::Node(np) => (np.variable.clone(), np.properties.is_some()),
                    _ => {
                        i += 1;
                        continue;
                    }
                };
                (first_var, second_var, edge_has_props, second_has_props)
            }
        } else {
            i += 1;
            continue;
        };

        // try_count_simple_pattern requires no property filters on edge or second node
        if edge_has_props || second_has_props {
            i += 1;
            continue;
        }

        let first_var = match first_var {
            Some(v) => v,
            None => {
                i += 1;
                continue;
            }
        };

        // Check RETURN: must have count() aggregate + group-by on first node only
        let fusable = if let Clause::Return(r) = &query.clauses[i + 1] {
            if r.distinct {
                false
            } else {
                let mut has_count = false;
                let mut all_valid = true;
                for item in &r.items {
                    if is_aggregate_expression(&item.expression) {
                        match &item.expression {
                            Expression::FunctionCall {
                                name,
                                args,
                                distinct,
                            } if name.eq_ignore_ascii_case("count") => {
                                if *distinct {
                                    all_valid = false;
                                    break;
                                }
                                // count(*) is fine
                                if args.len() == 1 && matches!(args[0], Expression::Star) {
                                    has_count = true;
                                    continue;
                                }
                                // count(var) — var must be second node
                                if let Some(Expression::Variable(var)) = args.first() {
                                    if second_var.as_deref() == Some(var.as_str()) {
                                        has_count = true;
                                        continue;
                                    }
                                }
                                all_valid = false;
                                break;
                            }
                            _ => {
                                all_valid = false;
                                break;
                            }
                        }
                    } else {
                        // Group-by item must reference first node variable
                        match &item.expression {
                            Expression::PropertyAccess { variable, .. }
                                if variable == &first_var =>
                            {
                                // OK
                            }
                            Expression::Variable(v) if v == &first_var => {
                                // OK
                            }
                            _ => {
                                all_valid = false;
                                break;
                            }
                        }
                    }
                }
                // Don't fuse when ALL items are aggregates (no group keys).
                // The standard execute_return_with_aggregation handles this
                // correctly as a single-group aggregate over all rows.
                let has_group_keys = r
                    .items
                    .iter()
                    .any(|item| !is_aggregate_expression(&item.expression));
                has_count && all_valid && has_group_keys
            }
        } else {
            false
        };

        if !fusable {
            i += 1;
            continue;
        }

        // All checks passed — fuse MATCH + RETURN
        let return_clause = if let Clause::Return(r) = query.clauses.remove(i + 1) {
            r
        } else {
            unreachable!()
        };
        let match_clause = if let Clause::Match(m) = query.clauses.remove(i) {
            m
        } else {
            unreachable!()
        };

        query.clauses.insert(
            i,
            Clause::FusedMatchReturnAggregate {
                match_clause,
                return_clause,
            },
        );

        i += 1;
    }
}

/// Collect variable names and their node types from patterns
fn collect_pattern_variables(
    patterns: &[crate::graph::pattern_matching::Pattern],
) -> Vec<(String, Option<String>)> {
    let mut vars = Vec::new();
    for pattern in patterns {
        for element in &pattern.elements {
            if let PatternElement::Node(np) = element {
                if let Some(ref var) = np.variable {
                    vars.push((var.clone(), np.node_type.clone()));
                }
            }
        }
    }
    vars
}

/// (equality_conditions, in_conditions, remaining_predicate)
type PushableResult = (
    Vec<(String, String, Value)>,
    Vec<(String, String, Vec<Value>)>,
    Option<Predicate>,
);

/// Extract pushable predicates from a WHERE clause into MATCH patterns.
/// Returns (equality_conditions, in_conditions, remaining_predicate).
///
/// Pushes conditions of the form:
/// - `variable.property = literal_value` (equality)
/// - `variable.property = $param` (equality with param)
/// - `variable.property IN [literal, ...]` (IN list)
///
/// The variable must be defined in MATCH.
fn extract_pushable_equalities(
    pred: &Predicate,
    match_vars: &[(String, Option<String>)],
    params: &HashMap<String, Value>,
) -> PushableResult {
    let mut pushable = Vec::new();
    let mut pushable_in = Vec::new();
    let remaining =
        extract_from_predicate(pred, match_vars, params, &mut pushable, &mut pushable_in);
    (pushable, pushable_in, remaining)
}

/// Recursively extract pushable predicates from a predicate tree.
/// Returns the remaining predicate (None if fully consumed).
fn extract_from_predicate(
    pred: &Predicate,
    match_vars: &[(String, Option<String>)],
    params: &HashMap<String, Value>,
    pushable: &mut Vec<(String, String, Value)>,
    pushable_in: &mut Vec<(String, String, Vec<Value>)>,
) -> Option<Predicate> {
    match pred {
        Predicate::Comparison {
            left,
            operator: ComparisonOp::Equals,
            right,
        } => {
            // Check if this is variable.property = literal or variable.property = $param
            if let Some((var, prop, val)) = try_extract_equality(left, right, match_vars, params) {
                pushable.push((var, prop, val));
                None // Fully consumed
            } else {
                Some(pred.clone()) // Keep as-is
            }
        }
        Predicate::In { expr, list } => {
            // Push variable.property IN [literal, ...] into MATCH pattern
            if let Expression::PropertyAccess { variable, property } = expr {
                if match_vars.iter().any(|(v, _)| v == variable) {
                    let all_literals: Option<Vec<Value>> = list
                        .iter()
                        .map(|item| {
                            if let Expression::Literal(val) = item {
                                Some(val.clone())
                            } else {
                                None
                            }
                        })
                        .collect();
                    if let Some(values) = all_literals {
                        pushable_in.push((variable.clone(), property.clone(), values));
                        return None; // Fully consumed
                    }
                }
            }
            Some(pred.clone())
        }
        Predicate::And(left, right) => {
            let left_remaining =
                extract_from_predicate(left, match_vars, params, pushable, pushable_in);
            let right_remaining =
                extract_from_predicate(right, match_vars, params, pushable, pushable_in);

            match (left_remaining, right_remaining) {
                (None, None) => None,
                (Some(l), None) => Some(l),
                (None, Some(r)) => Some(r),
                (Some(l), Some(r)) => Some(Predicate::And(Box::new(l), Box::new(r))),
            }
        }
        // Other predicate types can't be pushed
        _ => Some(pred.clone()),
    }
}

/// Try to extract a simple equality: variable.property = literal_or_param
fn try_extract_equality(
    left: &Expression,
    right: &Expression,
    match_vars: &[(String, Option<String>)],
    params: &HashMap<String, Value>,
) -> Option<(String, String, Value)> {
    // Left is property access, right is literal
    if let (Expression::PropertyAccess { variable, property }, Expression::Literal(val)) =
        (left, right)
    {
        if match_vars.iter().any(|(v, _)| v == variable) {
            return Some((variable.clone(), property.clone(), val.clone()));
        }
    }

    // Right is property access, left is literal (commutative)
    if let (Expression::Literal(val), Expression::PropertyAccess { variable, property }) =
        (left, right)
    {
        if match_vars.iter().any(|(v, _)| v == variable) {
            return Some((variable.clone(), property.clone(), val.clone()));
        }
    }

    // Left is property access, right is parameter (resolve from params)
    if let (Expression::PropertyAccess { variable, property }, Expression::Parameter(name)) =
        (left, right)
    {
        if let Some(val) = params.get(name.as_str()) {
            if match_vars.iter().any(|(v, _)| v == variable) {
                return Some((variable.clone(), property.clone(), val.clone()));
            }
        }
    }

    // Right is property access, left is parameter (commutative)
    if let (Expression::Parameter(name), Expression::PropertyAccess { variable, property }) =
        (left, right)
    {
        if let Some(val) = params.get(name.as_str()) {
            if match_vars.iter().any(|(v, _)| v == variable) {
                return Some((variable.clone(), property.clone(), val.clone()));
            }
        }
    }

    None
}

/// Apply a property equality condition to the matching node pattern in MATCH
fn apply_property_to_patterns(
    patterns: &mut [crate::graph::pattern_matching::Pattern],
    var_name: &str,
    property: &str,
    value: Value,
) {
    for pattern in patterns.iter_mut() {
        for element in &mut pattern.elements {
            if let PatternElement::Node(ref mut np) = element {
                if np.variable.as_deref() == Some(var_name) {
                    let props = np.properties.get_or_insert_with(Default::default);
                    props.insert(property.to_string(), PropertyMatcher::Equals(value));
                    return;
                }
            }
        }
    }
}

/// Apply an IN-list property condition to the matching node pattern in MATCH
fn apply_in_property_to_patterns(
    patterns: &mut [crate::graph::pattern_matching::Pattern],
    var_name: &str,
    property: &str,
    values: Vec<Value>,
) {
    for pattern in patterns.iter_mut() {
        for element in &mut pattern.elements {
            if let PatternElement::Node(ref mut np) = element {
                if np.variable.as_deref() == Some(var_name) {
                    let props = np.properties.get_or_insert_with(Default::default);
                    props.insert(property.to_string(), PropertyMatcher::In(values));
                    return;
                }
            }
        }
    }
}

// ============================================================================
// Fused RETURN + ORDER BY + LIMIT for vector_score
// ============================================================================

/// Detect `RETURN ... vector_score(...) AS s ... ORDER BY s DESC LIMIT k`
/// and replace with a fused clause that uses a min-heap (O(n log k) vs O(n log n))
/// and projects RETURN expressions only for the k surviving rows.
fn fuse_vector_score_order_limit(query: &mut CypherQuery) {
    if query.clauses.len() < 3 {
        return;
    }

    let mut i = 0;
    while i + 2 < query.clauses.len() {
        // Check for RETURN + ORDER BY + LIMIT pattern
        let is_pattern = matches!(
            (
                &query.clauses[i],
                &query.clauses[i + 1],
                &query.clauses[i + 2]
            ),
            (Clause::Return(_), Clause::OrderBy(_), Clause::Limit(_))
        );
        if !is_pattern {
            i += 1;
            continue;
        }

        // Extract references for analysis (before removing)
        let (score_idx, alias) = if let Clause::Return(r) = &query.clauses[i] {
            // Don't fuse if RETURN has aggregation or DISTINCT
            if r.distinct {
                i += 1;
                continue;
            }
            // Find the vector_score item
            let found = r.items.iter().enumerate().find(|(_, item)| {
                matches!(
                    &item.expression,
                    Expression::FunctionCall { name, .. }
                        if name.eq_ignore_ascii_case("vector_score")
                )
            });
            match found {
                Some((idx, item)) => {
                    let col = return_item_column_name(item);
                    (idx, col)
                }
                None => {
                    i += 1;
                    continue;
                }
            }
        } else {
            i += 1;
            continue;
        };

        // Check ORDER BY references the score alias and has exactly one item
        let descending = if let Clause::OrderBy(o) = &query.clauses[i + 1] {
            if o.items.len() != 1 {
                i += 1;
                continue;
            }
            let sort_name = match &o.items[0].expression {
                Expression::Variable(v) => v.clone(),
                other => expression_to_column_name(other),
            };
            if sort_name != alias {
                i += 1;
                continue;
            }
            !o.items[0].ascending
        } else {
            i += 1;
            continue;
        };

        // Extract LIMIT value (must be a literal non-negative integer)
        let limit = if let Clause::Limit(l) = &query.clauses[i + 2] {
            match &l.count {
                Expression::Literal(Value::Int64(n)) if *n > 0 => *n as usize,
                _ => {
                    i += 1;
                    continue;
                }
            }
        } else {
            i += 1;
            continue;
        };

        // All checks passed — fuse the three clauses
        query.clauses.remove(i + 2); // LIMIT
        query.clauses.remove(i + 1); // ORDER BY
        let return_clause = if let Clause::Return(r) = query.clauses.remove(i) {
            r
        } else {
            unreachable!()
        };

        query.clauses.insert(
            i,
            Clause::FusedVectorScoreTopK {
                return_clause,
                score_item_index: score_idx,
                descending,
                limit,
            },
        );

        i += 1;
    }
}

/// Column name for a return item (mirrors executor's return_item_column_name).
fn return_item_column_name(item: &ReturnItem) -> String {
    if let Some(ref alias) = item.alias {
        alias.clone()
    } else {
        expression_to_column_name(&item.expression)
    }
}

/// Simple expression-to-string for column name matching in the planner.
fn expression_to_column_name(expr: &Expression) -> String {
    match expr {
        Expression::Variable(name) => name.clone(),
        Expression::PropertyAccess { variable, property } => format!("{}.{}", variable, property),
        Expression::FunctionCall { name, args, .. } => {
            let args_str: Vec<String> = args.iter().map(expression_to_column_name).collect();
            format!("{}({})", name, args_str.join(", "))
        }
        _ => format!("{:?}", expr),
    }
}

// ============================================================================
// General Top-K ORDER BY LIMIT Fusion
// ============================================================================

/// Fuse RETURN + ORDER BY + LIMIT into a single top-k heap pass.
/// Generalizes `fuse_vector_score_order_limit` to any numeric sort expression.
/// Runs after the vector_score-specific pass so it only handles non-vector_score cases.
fn fuse_order_by_top_k(query: &mut CypherQuery) {
    if query.clauses.len() < 3 {
        return;
    }

    let mut i = 0;
    while i + 2 < query.clauses.len() {
        // Check for RETURN + ORDER BY + LIMIT pattern
        let is_pattern = matches!(
            (
                &query.clauses[i],
                &query.clauses[i + 1],
                &query.clauses[i + 2]
            ),
            (Clause::Return(_), Clause::OrderBy(_), Clause::Limit(_))
        );
        if !is_pattern {
            i += 1;
            continue;
        }

        // Reject if a SKIP clause precedes LIMIT (SKIP+LIMIT needs full sort)
        if i + 3 < query.clauses.len() && matches!(&query.clauses[i + 2], Clause::Skip(_)) {
            i += 1;
            continue;
        }

        let (score_idx, alias) = if let Clause::Return(r) = &query.clauses[i] {
            // Don't fuse if RETURN has DISTINCT
            if r.distinct {
                i += 1;
                continue;
            }
            // Don't fuse if any RETURN item has aggregation
            if r.items
                .iter()
                .any(|item| super::executor::is_aggregate_expression(&item.expression))
            {
                i += 1;
                continue;
            }
            // We need ORDER BY to reference a RETURN alias — find which one
            let order_alias = if let Clause::OrderBy(o) = &query.clauses[i + 1] {
                if o.items.len() != 1 {
                    i += 1;
                    continue;
                }
                match &o.items[0].expression {
                    Expression::Variable(v) => v.clone(),
                    other => expression_to_column_name(other),
                }
            } else {
                i += 1;
                continue;
            };
            // Find matching RETURN item
            let found = r
                .items
                .iter()
                .enumerate()
                .find(|(_, item)| return_item_column_name(item) == order_alias);
            match found {
                Some((idx, _)) => (idx, order_alias),
                None => {
                    i += 1;
                    continue;
                }
            }
        } else {
            i += 1;
            continue;
        };
        // Verify alias matches ORDER BY (already extracted above)
        let _ = &alias;

        // Extract ORDER BY direction
        let descending = if let Clause::OrderBy(o) = &query.clauses[i + 1] {
            !o.items[0].ascending
        } else {
            i += 1;
            continue;
        };

        // Extract LIMIT (must be positive integer literal)
        let limit = if let Clause::Limit(l) = &query.clauses[i + 2] {
            match &l.count {
                Expression::Literal(Value::Int64(n)) if *n > 0 => *n as usize,
                _ => {
                    i += 1;
                    continue;
                }
            }
        } else {
            i += 1;
            continue;
        };

        // All checks passed — fuse the three clauses
        query.clauses.remove(i + 2); // LIMIT
        query.clauses.remove(i + 1); // ORDER BY
        let return_clause = if let Clause::Return(r) = query.clauses.remove(i) {
            r
        } else {
            unreachable!()
        };

        query.clauses.insert(
            i,
            Clause::FusedOrderByTopK {
                return_clause,
                score_item_index: score_idx,
                descending,
                limit,
            },
        );

        i += 1;
    }
}

// ============================================================================
// Predicate Cost-Based Reordering
// ============================================================================

/// Reorder AND/OR children in WHERE predicates so cheaper predicates
/// are evaluated first (enabling short-circuit evaluation).
fn reorder_predicates_by_cost(query: &mut CypherQuery) {
    for clause in &mut query.clauses {
        if let Clause::Where(ref mut w) = clause {
            w.predicate = reorder_predicate(std::mem::replace(
                &mut w.predicate,
                Predicate::IsNull(Expression::Literal(Value::Null)),
            ));
        }
    }
}

/// Recursively reorder a predicate tree by estimated cost.
fn reorder_predicate(pred: Predicate) -> Predicate {
    match pred {
        Predicate::And(left, right) => {
            let left = reorder_predicate(*left);
            let right = reorder_predicate(*right);
            // Put cheaper predicate first for short-circuit
            if estimate_predicate_cost(&right) < estimate_predicate_cost(&left) {
                Predicate::And(Box::new(right), Box::new(left))
            } else {
                Predicate::And(Box::new(left), Box::new(right))
            }
        }
        Predicate::Or(left, right) => {
            let left = reorder_predicate(*left);
            let right = reorder_predicate(*right);
            // Put cheaper predicate first for short-circuit
            if estimate_predicate_cost(&right) < estimate_predicate_cost(&left) {
                Predicate::Or(Box::new(right), Box::new(left))
            } else {
                Predicate::Or(Box::new(left), Box::new(right))
            }
        }
        Predicate::Not(inner) => Predicate::Not(Box::new(reorder_predicate(*inner))),
        other => other,
    }
}

/// Estimate the relative cost of evaluating a predicate.
fn estimate_predicate_cost(pred: &Predicate) -> u32 {
    match pred {
        Predicate::Comparison { left, right, .. } => {
            estimate_expression_cost(left) + estimate_expression_cost(right) + 1
        }
        Predicate::And(l, r) | Predicate::Or(l, r) => {
            estimate_predicate_cost(l) + estimate_predicate_cost(r)
        }
        Predicate::Not(inner) => estimate_predicate_cost(inner) + 1,
        Predicate::IsNull(_) | Predicate::IsNotNull(_) => 2,
        Predicate::In { list, .. } => 3 + list.len() as u32,
        Predicate::StartsWith { .. } | Predicate::EndsWith { .. } | Predicate::Contains { .. } => 5,
        Predicate::Exists { .. } => 100, // Pattern existence checks are expensive
    }
}

/// Estimate the relative cost of evaluating an expression.
fn estimate_expression_cost(expr: &Expression) -> u32 {
    match expr {
        Expression::Literal(_) => 1,
        Expression::Parameter(_) => 1,
        Expression::PropertyAccess { .. } => 2,
        Expression::Variable(_) => 1,
        Expression::Star => 1,
        Expression::FunctionCall { name, args, .. } => {
            let base = match name.to_lowercase().as_str() {
                "point" => 3,
                "distance" => 10,
                "contains" => 50,
                "intersects" => 60,
                "centroid" => 30,
                "area" => 40,
                "perimeter" => 40,
                "latitude" | "longitude" => 2,
                "tostring" | "tointeger" | "tofloat" | "toboolean" => 2,
                "size" | "length" | "type" | "id" => 2,
                "tolower" | "toupper" | "trim" | "ltrim" | "rtrim" => 3,
                "substring" | "replace" | "split" => 5,
                "abs" | "ceil" | "ceiling" | "floor" | "round" | "sqrt" | "sign" => 2,
                "vector_score" => 200, // Embedding lookup + similarity computation
                "valid_at" | "valid_during" => 5, // 2 property lookups + 2 comparisons
                _ => 5,                // Unknown functions get moderate cost
            };
            let arg_cost: u32 = args.iter().map(estimate_expression_cost).sum();
            base + arg_cost
        }
        Expression::Add(l, r)
        | Expression::Subtract(l, r)
        | Expression::Multiply(l, r)
        | Expression::Divide(l, r) => estimate_expression_cost(l) + estimate_expression_cost(r) + 1,
        Expression::Negate(inner) => estimate_expression_cost(inner) + 1,
        Expression::ListLiteral(items) => {
            items.iter().map(estimate_expression_cost).sum::<u32>() + 1
        }
        Expression::Case {
            when_clauses,
            else_expr,
            ..
        } => {
            let clause_cost: u32 = when_clauses
                .iter()
                .map(|(_, e)| estimate_expression_cost(e) + 2)
                .sum();
            clause_cost
                + else_expr
                    .as_ref()
                    .map_or(0, |e| estimate_expression_cost(e))
        }
        Expression::IndexAccess { expr, index } => {
            estimate_expression_cost(expr) + estimate_expression_cost(index) + 1
        }
        Expression::ListSlice { expr, start, end } => {
            estimate_expression_cost(expr)
                + start.as_ref().map_or(0, |s| estimate_expression_cost(s))
                + end.as_ref().map_or(0, |e| estimate_expression_cost(e))
                + 1
        }
        _ => 5, // ListComprehension, MapProjection
    }
}

// ============================================================================
// text_score → vector_score AST Rewrite
// ============================================================================

/// Collected texts that the caller must embed before execution.
/// Each entry is `(param_name, query_text)` — the caller embeds the text and
/// inserts the resulting vector into the params map under `param_name`.
pub struct TextScoreRewrite {
    pub texts_to_embed: Vec<(String, String)>,
}

/// Walk the AST and rewrite all `text_score(node, col, query_text)` calls
/// to `vector_score(node, col_emb, $__ts_N)`.
///
/// The text argument can be a string literal or a `$parameter` (resolved from
/// `params`).  Returns the collected texts so the caller can embed them and
/// inject the resulting vectors into the params map before optimization.
pub fn rewrite_text_score(
    query: &mut CypherQuery,
    params: &HashMap<String, Value>,
) -> Result<TextScoreRewrite, String> {
    let mut collector = TextScoreCollector {
        counter: 0,
        texts_to_embed: Vec::new(),
    };

    for clause in &mut query.clauses {
        match clause {
            Clause::Return(r) => {
                for item in &mut r.items {
                    collector.rewrite_expr(&mut item.expression, params)?;
                }
            }
            Clause::Where(w) => {
                collector.rewrite_pred(&mut w.predicate, params)?;
            }
            Clause::With(w) => {
                for item in &mut w.items {
                    collector.rewrite_expr(&mut item.expression, params)?;
                }
                if let Some(ref mut wh) = w.where_clause {
                    collector.rewrite_pred(&mut wh.predicate, params)?;
                }
            }
            Clause::OrderBy(o) => {
                for item in &mut o.items {
                    collector.rewrite_expr(&mut item.expression, params)?;
                }
            }
            Clause::Unwind(u) => {
                collector.rewrite_expr(&mut u.expression, params)?;
            }
            Clause::Delete(d) => {
                for expr in &mut d.expressions {
                    collector.rewrite_expr(expr, params)?;
                }
            }
            Clause::Set(s) => {
                for item in &mut s.items {
                    if let SetItem::Property {
                        ref mut expression, ..
                    } = item
                    {
                        collector.rewrite_expr(expression, params)?;
                    }
                }
            }
            Clause::Create(c) => {
                for pattern in &mut c.patterns {
                    for element in &mut pattern.elements {
                        match element {
                            CreateElement::Node(n) => {
                                for (_, expr) in &mut n.properties {
                                    collector.rewrite_expr(expr, params)?;
                                }
                            }
                            CreateElement::Edge(e) => {
                                for (_, expr) in &mut e.properties {
                                    collector.rewrite_expr(expr, params)?;
                                }
                            }
                        }
                    }
                }
            }
            Clause::Merge(m) => {
                for element in &mut m.pattern.elements {
                    match element {
                        CreateElement::Node(n) => {
                            for (_, expr) in &mut n.properties {
                                collector.rewrite_expr(expr, params)?;
                            }
                        }
                        CreateElement::Edge(e) => {
                            for (_, expr) in &mut e.properties {
                                collector.rewrite_expr(expr, params)?;
                            }
                        }
                    }
                }
                if let Some(ref mut items) = m.on_create {
                    for item in items {
                        if let SetItem::Property {
                            ref mut expression, ..
                        } = item
                        {
                            collector.rewrite_expr(expression, params)?;
                        }
                    }
                }
                if let Some(ref mut items) = m.on_match {
                    for item in items {
                        if let SetItem::Property {
                            ref mut expression, ..
                        } = item
                        {
                            collector.rewrite_expr(expression, params)?;
                        }
                    }
                }
            }
            Clause::Skip(s) => {
                collector.rewrite_expr(&mut s.count, params)?;
            }
            Clause::Limit(l) => {
                collector.rewrite_expr(&mut l.count, params)?;
            }
            // Match/OptionalMatch: patterns only, no function call expressions
            // Remove: no expressions
            // Fused clauses: don't exist yet (created by optimize, which runs after rewrite)
            _ => {}
        }
    }

    Ok(TextScoreRewrite {
        texts_to_embed: collector.texts_to_embed,
    })
}

struct TextScoreCollector {
    counter: usize,
    texts_to_embed: Vec<(String, String)>,
}

impl TextScoreCollector {
    /// Rewrite an expression in-place.  Turns `text_score(...)` into `vector_score(...)`.
    fn rewrite_expr(
        &mut self,
        expr: &mut Expression,
        params: &HashMap<String, Value>,
    ) -> Result<(), String> {
        match expr {
            Expression::FunctionCall { name, args, .. }
                if name.eq_ignore_ascii_case("text_score") =>
            {
                if args.len() != 3 && args.len() != 4 {
                    return Err(
                        "text_score() requires 3 arguments: (node, text_column, query_text) \
                         with optional 4th metric argument"
                            .into(),
                    );
                }

                // arg[1]: text column — must be a string literal
                let col_name =
                    match &args[1] {
                        Expression::Literal(Value::String(s)) => s.clone(),
                        _ => return Err(
                            "text_score(): second argument must be a string literal column name"
                                .into(),
                        ),
                    };

                // arg[2]: query text — string literal or $param
                let query_text = match &args[2] {
                    Expression::Literal(Value::String(s)) => s.clone(),
                    Expression::Parameter(param_name) => match params.get(param_name.as_str()) {
                        Some(Value::String(s)) => s.clone(),
                        Some(_) => {
                            return Err(format!(
                                "text_score(): parameter ${} must be a string",
                                param_name
                            ))
                        }
                        None => {
                            return Err(format!(
                                "text_score(): parameter ${} not found",
                                param_name
                            ))
                        }
                    },
                    _ => {
                        return Err(
                            "text_score(): third argument must be a string literal or $parameter"
                                .into(),
                        )
                    }
                };

                // Deduplicate: reuse param if same query text already collected
                let param_name = if let Some((existing, _)) =
                    self.texts_to_embed.iter().find(|(_, t)| t == &query_text)
                {
                    existing.clone()
                } else {
                    let pname = format!("__ts_{}", self.counter);
                    self.counter += 1;
                    self.texts_to_embed.push((pname.clone(), query_text));
                    pname
                };

                // Rewrite: text_score(n, 'summary', ...) → vector_score(n, 'summary_emb', $__ts_N)
                *name = "vector_score".to_string();
                args[1] = Expression::Literal(Value::String(format!("{}_emb", col_name)));
                args[2] = Expression::Parameter(param_name);

                Ok(())
            }
            Expression::FunctionCall { args, .. } => {
                for arg in args.iter_mut() {
                    self.rewrite_expr(arg, params)?;
                }
                Ok(())
            }
            Expression::Add(l, r)
            | Expression::Subtract(l, r)
            | Expression::Multiply(l, r)
            | Expression::Divide(l, r)
            | Expression::Concat(l, r) => {
                self.rewrite_expr(l, params)?;
                self.rewrite_expr(r, params)?;
                Ok(())
            }
            Expression::Negate(inner) => self.rewrite_expr(inner, params),
            Expression::ListLiteral(items) => {
                for item in items.iter_mut() {
                    self.rewrite_expr(item, params)?;
                }
                Ok(())
            }
            Expression::Case {
                operand,
                when_clauses,
                else_expr,
            } => {
                if let Some(op) = operand {
                    self.rewrite_expr(op, params)?;
                }
                for (cond, result) in when_clauses.iter_mut() {
                    match cond {
                        CaseCondition::Expression(e) => self.rewrite_expr(e, params)?,
                        CaseCondition::Predicate(p) => self.rewrite_pred(p, params)?,
                    }
                    self.rewrite_expr(result, params)?;
                }
                if let Some(el) = else_expr {
                    self.rewrite_expr(el, params)?;
                }
                Ok(())
            }
            Expression::IndexAccess { expr, index } => {
                self.rewrite_expr(expr, params)?;
                self.rewrite_expr(index, params)?;
                Ok(())
            }
            Expression::ListSlice { expr, start, end } => {
                self.rewrite_expr(expr, params)?;
                if let Some(s) = start {
                    self.rewrite_expr(s, params)?;
                }
                if let Some(e) = end {
                    self.rewrite_expr(e, params)?;
                }
                Ok(())
            }
            Expression::ListComprehension {
                list_expr,
                filter,
                map_expr,
                ..
            } => {
                self.rewrite_expr(list_expr, params)?;
                if let Some(f) = filter {
                    self.rewrite_pred(f, params)?;
                }
                if let Some(m) = map_expr {
                    self.rewrite_expr(m, params)?;
                }
                Ok(())
            }
            Expression::MapProjection { items, .. } => {
                for item in items.iter_mut() {
                    if let MapProjectionItem::Alias { expr, .. } = item {
                        self.rewrite_expr(expr, params)?;
                    }
                }
                Ok(())
            }
            Expression::MapLiteral(entries) => {
                for (_, expr) in entries.iter_mut() {
                    self.rewrite_expr(expr, params)?;
                }
                Ok(())
            }
            // Leaf nodes
            Expression::PropertyAccess { .. }
            | Expression::Variable(_)
            | Expression::Literal(_)
            | Expression::Parameter(_)
            | Expression::Star => Ok(()),
            Expression::IsNull(inner) | Expression::IsNotNull(inner) => {
                self.rewrite_expr(inner, params)
            }
            Expression::QuantifiedList {
                list_expr, filter, ..
            } => {
                self.rewrite_expr(list_expr, params)?;
                self.rewrite_pred(filter, params)?;
                Ok(())
            }
        }
    }

    /// Rewrite predicates in-place (for WHERE clauses).
    fn rewrite_pred(
        &mut self,
        pred: &mut Predicate,
        params: &HashMap<String, Value>,
    ) -> Result<(), String> {
        match pred {
            Predicate::Comparison { left, right, .. } => {
                self.rewrite_expr(left, params)?;
                self.rewrite_expr(right, params)?;
                Ok(())
            }
            Predicate::And(l, r) | Predicate::Or(l, r) => {
                self.rewrite_pred(l, params)?;
                self.rewrite_pred(r, params)?;
                Ok(())
            }
            Predicate::Not(inner) => self.rewrite_pred(inner, params),
            Predicate::IsNull(e) | Predicate::IsNotNull(e) => self.rewrite_expr(e, params),
            Predicate::In { expr, list } => {
                self.rewrite_expr(expr, params)?;
                for item in list.iter_mut() {
                    self.rewrite_expr(item, params)?;
                }
                Ok(())
            }
            Predicate::StartsWith { expr, pattern }
            | Predicate::EndsWith { expr, pattern }
            | Predicate::Contains { expr, pattern } => {
                self.rewrite_expr(expr, params)?;
                self.rewrite_expr(pattern, params)?;
                Ok(())
            }
            Predicate::Exists { .. } => Ok(()),
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::cypher::parser::parse_cypher;

    #[test]
    fn test_predicate_pushdown_simple() {
        let mut query = parse_cypher("MATCH (n:Person) WHERE n.age = 30 RETURN n").unwrap();

        let graph = DirGraph::new();
        let params = HashMap::new();
        optimize(&mut query, &graph, &params);

        // WHERE should be removed (fully consumed)
        assert_eq!(query.clauses.len(), 2); // MATCH + RETURN
        assert!(matches!(&query.clauses[0], Clause::Match(_)));
        assert!(matches!(&query.clauses[1], Clause::Return(_)));

        // The MATCH pattern should now have {age: 30} as a property
        if let Clause::Match(m) = &query.clauses[0] {
            if let PatternElement::Node(np) = &m.patterns[0].elements[0] {
                assert!(np.properties.is_some());
                let props = np.properties.as_ref().unwrap();
                assert!(props.contains_key("age"));
            } else {
                panic!("Expected node pattern");
            }
        }
    }

    #[test]
    fn test_predicate_pushdown_partial() {
        let mut query =
            parse_cypher("MATCH (n:Person) WHERE n.age = 30 AND n.score > 100 RETURN n").unwrap();

        let graph = DirGraph::new();
        let params = HashMap::new();
        optimize(&mut query, &graph, &params);

        // n.age = 30 should be pushed, n.score > 100 should remain
        assert_eq!(query.clauses.len(), 3); // MATCH + WHERE + RETURN
        assert!(matches!(&query.clauses[1], Clause::Where(_)));

        if let Clause::Where(w) = &query.clauses[1] {
            // Remaining should be n.score > 100
            assert!(matches!(
                &w.predicate,
                Predicate::Comparison {
                    operator: ComparisonOp::GreaterThan,
                    ..
                }
            ));
        }
    }

    #[test]
    fn test_no_pushdown_for_non_equality() {
        let mut query = parse_cypher("MATCH (n:Person) WHERE n.age > 30 RETURN n").unwrap();

        let graph = DirGraph::new();
        let params = HashMap::new();
        optimize(&mut query, &graph, &params);

        // No pushdown - WHERE should remain
        assert_eq!(query.clauses.len(), 3); // MATCH + WHERE + RETURN
    }

    #[test]
    fn test_predicate_pushdown_parameter() {
        let mut query = parse_cypher("MATCH (n:Person) WHERE n.name = $name RETURN n").unwrap();

        let graph = DirGraph::new();
        let mut params = HashMap::new();
        params.insert("name".to_string(), Value::String("Alice".to_string()));
        optimize(&mut query, &graph, &params);

        // WHERE should be removed (parameter resolved and pushed)
        assert_eq!(query.clauses.len(), 2); // MATCH + RETURN

        // The MATCH pattern should now have {name: 'Alice'} as a property
        if let Clause::Match(m) = &query.clauses[0] {
            if let PatternElement::Node(np) = &m.patterns[0].elements[0] {
                assert!(np.properties.is_some());
                let props = np.properties.as_ref().unwrap();
                assert!(props.contains_key("name"));
                assert!(matches!(
                    props.get("name"),
                    Some(PropertyMatcher::Equals(Value::String(s))) if s == "Alice"
                ));
            } else {
                panic!("Expected node pattern");
            }
        }
    }

    #[test]
    fn test_predicate_pushdown_parameter_partial() {
        let mut query =
            parse_cypher("MATCH (n:Person) WHERE n.name = $name AND n.age > $min_age RETURN n")
                .unwrap();

        let graph = DirGraph::new();
        let mut params = HashMap::new();
        params.insert("name".to_string(), Value::String("Alice".to_string()));
        params.insert("min_age".to_string(), Value::Int64(25));
        optimize(&mut query, &graph, &params);

        // n.name = $name should be pushed, n.age > $min_age should remain (not equality)
        assert_eq!(query.clauses.len(), 3); // MATCH + WHERE + RETURN
    }
}
