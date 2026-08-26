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
    // `[:A|B]` — the singular `connection_type` holds only the first branch,
    // so fusing on it counted that branch alone. Carry every branch; the
    // executor sums one CSR offset read per type. Duplicated branches
    // (`[:A|A]`) would double-count, so they are dropped here.
    let edge_types = match &edge.connection_types {
        Some(types) if !types.is_empty() => {
            let mut deduped: Vec<String> = Vec::with_capacity(types.len());
            for ty in types {
                if !deduped.contains(ty) {
                    deduped.push(ty.clone());
                }
            }
            Some(deduped)
        }
        _ => edge.connection_type.clone().map(|ty| vec![ty]),
    };

    query.clauses.drain(0..2);
    query.clauses.insert(
        0,
        Clause::FusedCountAnchoredEdges {
            anchor_idx,
            anchor_direction: anchor_dir,
            edge_types,
            alias,
        },
    );
}

/// Takes `graph` rather than a pre-computed `type_shadowed` flag because
/// [`DirGraph::has_type_shadowing_property`] is an O(#types) scan and the flag
/// is consulted only on the `RETURN <type-accessor>, count(…)` branch far
/// below. Evaluated as a call argument it ran on **every** statement the
/// planner touched — measured at ~23 ns per declared node type, i.e. 4.6 µs of
/// pure waste per statement on a 200-type schema, which is exactly what its own
/// doc-comment's "only consulted for count-by-type-shaped queries" promised it
/// did not do.
pub(crate) fn fuse_count_short_circuits(
    query: &mut CypherQuery,
    has_secondary_labels: bool,
    graph: &DirGraph,
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

    // Aggregation modifiers and path assignments require the general executor.
    if return_clause.distinct
        || return_clause.having.is_some()
        || !match_clause.path_assignments.is_empty()
    {
        return;
    }

    // Must have exactly 1 pattern
    if match_clause.patterns.len() != 1 {
        return;
    }
    let pat = &match_clause.patterns[0];

    // ---- Pattern A: a lone node — every `count`-shaped RETURN over it ----
    if pat.elements.len() == 1 {
        let PatternElement::Node(node) = &pat.elements[0] else {
            return;
        };
        if let Some(fused) =
            fuse_single_node_count(node, return_clause, has_secondary_labels, graph)
        {
            query.clauses.drain(0..2);
            query.clauses.insert(0, fused);
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
            || src_node.multi_label_constrained()
            || src_node.properties.is_some()
            || tgt_node.node_type.is_some()
            || tgt_node.multi_label_constrained()
            || tgt_node.properties.is_some()
        {
            return;
        }

        // Edge must have no property filters or var_length, and must be directed
        if edge.properties.is_some()
            || edge.var_length.is_some()
            || edge.connection_types.is_some()
            || edge.edge_filter.is_some()
            || edge.direction == EdgeDirection::Both
        {
            return;
        }

        // Re-used names impose identity constraints (`(a)-[r]->(a)` matches
        // self-loops only). A global edge count is valid only when the three
        // pattern elements are independent.
        let mut variables = std::collections::HashSet::new();
        for variable in [
            src_node.variable.as_deref(),
            edge.variable.as_deref(),
            tgt_node.variable.as_deref(),
        ]
        .into_iter()
        .flatten()
        {
            if !variables.insert(variable) {
                return;
            }
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

        // Sub-pattern C2: untyped global edge count. Every live directed
        // edge contributes exactly one row, including parallel edges and
        // self-loops. The strict guards above exclude all filters and
        // identity constraints, so GraphRead::edge_count() is exact.
        if return_clause.items.len() == 1
            && is_count_of_var_or_star(&return_clause.items[0].expression, edge_var)
        {
            let alias = return_item_column_name(&return_clause.items[0]);
            query.clauses.drain(0..2);
            query
                .clauses
                .insert(0, Clause::FusedCountAllEdges { alias });
            return;
        }

        // Sub-pattern C3: Untyped edge count by type — MATCH ()-[r]->() RETURN type(r), count(*)
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

/// Pattern A of [`fuse_count_short_circuits`]: `MATCH (n…) RETURN <count>`
/// over a single-element pattern → the clause it fuses to, or `None` to leave
/// the query for the matcher. Split out so the caller stays a dispatcher over
/// pattern shapes and each shape's own guards live in one place.
///
/// Returning the clause rather than rewriting in place is what lets the whole
/// decision run off `&`-borrows of the very clauses the caller then drains.
fn fuse_single_node_count(
    node: &crate::graph::core::pattern_matching::NodePattern,
    return_clause: &ReturnClause,
    has_secondary_labels: bool,
    graph: &DirGraph,
) -> Option<Clause> {
    // Property filters need the matcher; no bucket length can express them.
    if node.properties.is_some() {
        return None;
    }
    // `MATCH (n:A:B) RETURN count(n)` requires an intersection across the
    // labels, which the O(1) type-bucket count can't express. Bail to the full
    // matcher, which AND-intersects via `node_labels`. (Single-label secondary
    // counts ARE handled — the FusedCountTypedNode executor unions the primary
    // + secondary buckets for `node_type`.)
    if !node.extra_labels.is_empty() {
        return None;
    }

    let node_var = node.variable.as_deref();

    // A label constraint, alternation or not, decides between the two typed
    // counts and admits no other RETURN shape.
    if let Some(node_type) = node.node_type.as_ref() {
        let item = return_clause.items.first()?;
        if return_clause.items.len() != 1 || !is_count_of_var_or_star(&item.expression, node_var) {
            return None;
        }
        let alias = return_item_column_name(item);
        return match node.alt_labels.as_ref() {
            // `MATCH (n:A|B) RETURN count(n)`: a per-branch cardinality sum,
            // but only where the branches provably cannot overlap.
            Some(alts) => disjoint_alternation_branches(graph, alts)
                .map(|labels| Clause::FusedCountLabelUnion { labels, alias }),
            None => Some(Clause::FusedCountTypedNode {
                node_type: node_type.clone(),
                alias,
            }),
        };
    }

    // Untyped: `RETURN count(n)` is the whole graph; `RETURN n.type, count(n)`
    // is the per-type histogram.
    if return_clause.items.len() == 1 {
        let item = &return_clause.items[0];
        return is_count_of_var_or_star(&item.expression, node_var).then(|| {
            Clause::FusedCountAll {
                alias: return_item_column_name(item),
            }
        });
    }
    if return_clause.items.len() != 2 {
        return None;
    }
    let (type_idx, count_idx) = identify_type_count_pair(
        &return_clause.items,
        node_var,
        has_secondary_labels,
        graph.has_type_shadowing_property(),
    );
    let (ti, ci) = type_idx.zip(count_idx)?;
    Some(Clause::FusedCountByType {
        type_alias: return_item_column_name(&return_clause.items[ti]),
        count_alias: return_item_column_name(&return_clause.items[ci]),
        // `labels(n)` projects a list; `n.type`/`n.node_type`/`n.label`
        // project a scalar. Preserve each accessor's natural shape.
        type_as_list: is_labels_call(&return_clause.items[ti].expression, node_var),
    })
}

/// The branch labels of `(n:A|B|C)`, deduplicated, when a per-branch
/// cardinality sum counts every node exactly once — `None` when it could not,
/// which leaves the count to the matcher's union-and-dedup path.
///
/// **The disjointness proof is per-label, not global.** A node reaches a
/// branch either as its primary type — of which it has exactly one — or as a
/// secondary label. So two branches can share a node only if at least one of
/// *them* has secondary carriers; `has_secondary_labels` being true somewhere
/// else in the graph is irrelevant, and bailing on it would refuse the fusion
/// for every alternation the moment one unrelated label gained a carrier (the
/// 71×/33× mistake `multi_label_fuse_unsafe` was written to undo).
/// `secondary_label_index` is consulted by key presence rather than bucket
/// contents, so an emptied bucket keeps the answer conservative.
///
/// Dedup is not cosmetic: `MATCH (n:$a|$b)` whose parameters bind to the same
/// name arrives here as two identical branches (`dynamic_labels::resolve_pattern`
/// substitutes in place and cannot dedup — the slot indices are what the
/// markers address), and summing them would double the count.
///
/// Reading the graph at plan time is sound because the plan cache is keyed on
/// `(graph_id, version, …)` and every mutation bumps `version`, so a cached
/// plan is only ever replayed against the exact state that minted it.
fn disjoint_alternation_branches(graph: &DirGraph, alts: &[String]) -> Option<Vec<String>> {
    use crate::graph::schema::InternedKey;

    let mut labels: Vec<String> = Vec::with_capacity(alts.len());
    for label in alts {
        if graph.has_secondary_labels
            && graph
                .secondary_label_index
                .contains_key(&InternedKey::from_str(label))
        {
            return None;
        }
        if !labels.contains(label) {
            labels.push(label.clone());
        }
    }
    (!labels.is_empty()).then_some(labels)
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
