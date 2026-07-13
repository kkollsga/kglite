//! Planner annotations applied after structural rewrites.

use super::{Clause, CypherQuery, Expression, PassCtx};
use crate::graph::core::pattern_matching::PatternElement;
use crate::graph::schema::DirGraph;
use std::collections::HashSet;

/// **Pass:** `mark_fast_var_length_paths` — When a variable-length
/// edge `[:T*1..N]` has no path assignment and no edge variable AND
/// the downstream RETURN/WITH is `DISTINCT` or composed of dedup-safe
/// aggregates (`min/max/count(DISTINCT)/collect(DISTINCT)`), mark
/// `needs_path_info=false` so the executor uses a fast BFS with
/// global target-node dedup. The downstream-safety check is critical:
/// row count is implicit path count, so dedup-by-target silently
/// drops rows when the user wrote a plain per-path projection like
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
/// When a MATCH clause has no path assignments (`p = ...`) and the edge
/// has no named variable (`[r:T*1..N]`), AND the query's downstream
/// projection is provably indifferent to row multiplicity, the executor
/// can use a fast BFS with global target-node dedup instead of tracking
/// every distinct path.
///
/// The "indifferent to row multiplicity" check is critical: row count
/// is itself an implicit count of paths in Cypher's semantics, so
/// dedup-by-target silently drops rows when the user wrote a plain
/// per-path projection like `RETURN q.name`. The fast path is only
/// safe when the downstream is `DISTINCT`, or every projection is an
/// aggregate (multiplicity collapses inside the aggregate).
///
/// Caught by `tests/test_cypher_differential.py::var_length_no_var`,
/// which previously xfail'd because the un-gated fast path returned
/// 2 rows where Neo4j semantics demand 3.
fn mark_fast_var_length_paths(query: &mut CypherQuery) {
    if !downstream_is_dedup_safe(query) {
        return;
    }
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
fn fixed_edge_types_are_pairwise_disjoint(
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

/// Returns true iff the query's first downstream projection collapses
/// row multiplicity. The fast var-length BFS dedups by target node;
/// that's only correct when the surrounding query doesn't depend on
/// per-path row counts.
///
/// Two safe cases:
/// - `RETURN/WITH DISTINCT` — row tuples are deduped at projection
///   anyway, so a fast-path target-dedup is consistent.
/// - `RETURN/WITH` whose every item is an aggregate — multiplicity is
///   collapsed by the aggregate. (`count(*)` over the matches: paths
///   would count differently than targets, so we don't allow `count(*)`
///   here unless it's `count(DISTINCT target)` — but the simpler check
///   "every item is an aggregate" handles `count(DISTINCT target)`,
///   `sum(target.x)`, etc. uniformly. Plain `count(*)` over var-length
///   matches is a real semantic question; we conservatively reject
///   non-DISTINCT `count(*)` by requiring DISTINCT-aware aggregates.)
///
/// Conservative anywhere else: we'd rather skip the optimization than
/// silently drop rows.
fn downstream_is_dedup_safe(query: &CypherQuery) -> bool {
    for clause in &query.clauses {
        match clause {
            Clause::Return(r) => {
                if r.distinct {
                    return true;
                }
                let all_agg_distinct = !r.items.is_empty()
                    && r.items
                        .iter()
                        .all(|item| is_distinct_safe_aggregate(&item.expression));
                return all_agg_distinct;
            }
            Clause::With(w) => {
                if w.distinct {
                    return true;
                }
                let all_agg_distinct = !w.items.is_empty()
                    && w.items
                        .iter()
                        .all(|item| is_distinct_safe_aggregate(&item.expression));
                return all_agg_distinct;
            }
            _ => continue,
        }
    }
    false
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
                let (conn_type, direction, target_node_type) = {
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
                    match (&edge.connection_type, edge.direction, &target.node_type) {
                        (Some(ct), dir, Some(nt)) => (ct.clone(), dir, nt.clone()),
                        _ => continue,
                    }
                };

                // Look up connection type metadata
                if let Some(info) = graph.connection_type_metadata.get(&conn_type) {
                    let guaranteed = match direction {
                        EdgeDirection::Outgoing => {
                            info.target_types.len() == 1
                                && info.target_types.contains(&target_node_type)
                        }
                        EdgeDirection::Incoming => {
                            info.source_types.len() == 1
                                && info.source_types.contains(&target_node_type)
                        }
                        EdgeDirection::Both => false, // can't guarantee for bidirectional
                    };
                    if guaranteed {
                        if let PatternElement::Edge(ep) = &mut elements[i + 1] {
                            ep.skip_target_type_check = true;
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::fixed_edge_types_are_pairwise_disjoint;
    use crate::graph::core::pattern_matching::parse_pattern;

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
