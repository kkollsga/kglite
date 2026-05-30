//! Count-fusion passes — `MATCH (n) RETURN count(*)` / `RETURN n.type, count(*)`
//! and edge-count short-circuits, plus their predicate helpers.
//!
//! Split out of the former monolithic `fusion.rs` (0.10.10).

use super::*;
use crate::datatypes::values::Value;
use crate::graph::core::pattern_matching::PatternElement;
use crate::graph::languages::cypher::ast::*;
use crate::graph::schema::DirGraph;

pub(crate) fn fuse_anchored_edge_count(query: &mut CypherQuery, graph: &DirGraph) {
    use crate::graph::core::pattern_matching::{EdgeDirection, PropertyMatcher};

    if query.clauses.len() < 2 {
        return;
    }
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
    if return_clause.distinct || return_clause.having.is_some() {
        return;
    }
    if match_clause.patterns.len() != 1 || !match_clause.path_assignments.is_empty() {
        return;
    }
    let pat = &match_clause.patterns[0];
    if pat.elements.len() != 3 {
        return;
    }

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

    if edge.properties.is_some() || edge.var_length.is_some() {
        return;
    }
    if edge.direction == EdgeDirection::Both {
        return;
    }

    // Helper: does the node look like a pure `{id: VAL}` literal anchor —
    // no type, no variable, exactly one property keyed `id` with a literal
    // Equals matcher? Returns the id value on match.
    let as_anchor_id = |np: &crate::graph::core::pattern_matching::NodePattern| -> Option<Value> {
        if np.node_type.is_some() || np.variable.is_some() {
            return None;
        }
        let props = np.properties.as_ref()?;
        if props.len() != 1 {
            return None;
        }
        if let Some(PropertyMatcher::Equals(val)) = props.get("id") {
            Some(val.clone())
        } else {
            None
        }
    };
    // Helper: the other side is a named variable with no type/property filter.
    fn as_pure_var(np: &crate::graph::core::pattern_matching::NodePattern) -> Option<&String> {
        if np.node_type.is_some() || np.properties.is_some() {
            return None;
        }
        np.variable.as_ref()
    }

    let (var_name, anchor_val, anchor_dir) = match (as_pure_var(src_node), as_anchor_id(tgt_node)) {
        (Some(v), Some(id)) => {
            // var -[edge]-> {id: V}
            // anchor is the TARGET; traverse from anchor in the opposite dir.
            let dir = match edge.direction {
                EdgeDirection::Outgoing => petgraph::Direction::Incoming,
                EdgeDirection::Incoming => petgraph::Direction::Outgoing,
                EdgeDirection::Both => return,
            };
            (v, id, dir)
        }
        _ => match (as_anchor_id(src_node), as_pure_var(tgt_node)) {
            (Some(id), Some(v)) => {
                // {id: V} -[edge]-> var
                let dir = match edge.direction {
                    EdgeDirection::Outgoing => petgraph::Direction::Outgoing,
                    EdgeDirection::Incoming => petgraph::Direction::Incoming,
                    EdgeDirection::Both => return,
                };
                (v, id, dir)
            }
            _ => return,
        },
    };

    // RETURN must be exactly one item, which is count(var) or count(*).
    if return_clause.items.len() != 1 {
        return;
    }
    if !is_count_of_var_or_star(&return_clause.items[0].expression, Some(var_name)) {
        return;
    }

    // Resolve the anchor across node types. O(types) HashMap lookups; at
    // typical schema sizes this is negligible, and on Wikidata-scale (~88 k
    // types) we still only do one `HashMap::get` per type.
    let mut resolved: Option<petgraph::graph::NodeIndex> = None;
    for node_type in graph.type_indices.keys() {
        if let Some(idx) = graph.lookup_by_id_readonly(node_type, &anchor_val) {
            resolved = Some(idx);
            break;
        }
    }
    let anchor_idx = match resolved {
        Some(idx) => idx.index() as u32,
        None => return, // anchor not found — leave unfused, normal path returns 0
    };

    let alias = return_item_column_name(&return_clause.items[0]);
    let edge_type = edge.connection_type.clone();

    query.clauses.drain(0..2);
    query.clauses.insert(
        0,
        Clause::FusedCountAnchoredEdges {
            anchor_idx,
            anchor_direction: anchor_dir,
            edge_type,
            alias,
        },
    );
}

pub(crate) fn fuse_count_short_circuits(
    query: &mut CypherQuery,
    has_secondary_labels: bool,
    type_shadowed: bool,
) {
    use crate::graph::core::pattern_matching::EdgeDirection;

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

        // Multi-label patterns (`MATCH (n:A:B) RETURN count(n)`) require an
        // intersection across the labels, which the O(1) type-bucket count
        // can't express. Bail to the full matcher, which AND-intersects via
        // `node_labels`. (Single-label secondary counts ARE handled — the
        // FusedCountTypedNode executor unions the primary + secondary
        // buckets for `node_type`.)
        if !node.extra_labels.is_empty() {
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
            let (type_idx, count_idx) = identify_type_count_pair(
                &return_clause.items,
                node_var,
                has_secondary_labels,
                type_shadowed,
            );
            if let Some((ti, ci)) = type_idx.zip(count_idx) {
                let type_alias = return_item_column_name(&return_clause.items[ti]);
                let count_alias = return_item_column_name(&return_clause.items[ci]);
                // `labels(n)` projects a list; `n.type`/`n.node_type`/`n.label`
                // project a scalar. Preserve each accessor's natural shape.
                let type_as_list = is_labels_call(&return_clause.items[ti].expression, node_var);
                query.clauses.drain(0..2);
                query.clauses.insert(
                    0,
                    Clause::FusedCountByType {
                        type_alias,
                        count_alias,
                        type_as_list,
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

        // Edge must have no property filters or var_length, and must be directed
        if edge.properties.is_some()
            || edge.var_length.is_some()
            || edge.direction == EdgeDirection::Both
        {
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
pub(crate) fn is_count_of_var_or_star(expr: &Expression, node_var: Option<&str>) -> bool {
    if let Expression::FunctionCall {
        name,
        args,
        distinct,
    } = expr
    {
        if name != "count" || *distinct {
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
pub(crate) fn identify_type_count_pair(
    items: &[ReturnItem],
    node_var: Option<&str>,
    has_secondary_labels: bool,
    type_shadowed: bool,
) -> (Option<usize>, Option<usize>) {
    let mut type_idx = None;
    let mut count_idx = None;

    for (i, item) in items.iter().enumerate() {
        if is_count_of_var_or_star(&item.expression, node_var) {
            count_idx = Some(i);
        } else if (!type_shadowed && is_primary_type_accessor(&item.expression, node_var))
            || (!has_secondary_labels && is_labels_call(&item.expression, node_var))
        {
            // `n.type` is a valid fuse key only when unshadowed (KG-1); under a
            // shadow it is property-first and would group by the wrong key.
            // `labels(n)` can't be shadowed (gated on no secondary labels).
            type_idx = Some(i);
        }
    }
    (type_idx, count_idx)
}

/// `n.type` / `n.node_type` / `n.label` — a scalar primary-type accessor.
/// Valid as a `FusedCountByType` key only when no type stores a property of
/// that name (KG-1); callers gate on `!graph.has_type_shadowing_property()`.
pub(crate) fn is_primary_type_accessor(expr: &Expression, node_var: Option<&str>) -> bool {
    match expr {
        Expression::PropertyAccess { variable, property } => {
            matches!(property.as_str(), "type" | "node_type" | "label")
                && node_var.is_some_and(|nv| variable == nv)
        }
        _ => false,
    }
}

/// Check if expression is `labels(n)`. Grouping by `labels(n)` is only
/// equivalent to grouping by primary type when no node carries a secondary
/// label — otherwise a multi-labelled node forms its own label-set group
/// that the per-primary-type `FusedCountByType` count can't express. Callers
/// must gate this on `!has_secondary_labels`.
pub(crate) fn is_labels_call(expr: &Expression, node_var: Option<&str>) -> bool {
    if let Expression::FunctionCall { name, args, .. } = expr {
        if name == "labels" && args.len() == 1 {
            if let Expression::Variable(v) = &args[0] {
                return node_var.is_some_and(|nv| v == nv);
            }
        }
    }
    false
}

/// For `RETURN type(r), count(*)` — identify edge type function and count.
pub(crate) fn identify_edge_type_count_pair(
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
pub(crate) fn is_edge_type_function(expr: &Expression, edge_var: Option<&str>) -> bool {
    if let Expression::FunctionCall { name, args, .. } = expr {
        if name == "type" && args.len() == 1 {
            if let Expression::Variable(v) = &args[0] {
                return edge_var.is_some_and(|ev| v == ev);
            }
        }
    }
    false
}
