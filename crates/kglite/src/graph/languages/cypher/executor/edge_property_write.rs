//! Relationship property writes — the `SET r.p` / `REMOVE r.p` half of the
//! Cypher write path.
//!
//! Its own module because it shares nothing with the node path it used to sit
//! inside: an edge has no id/type guard to run, no columnar column to route a
//! value to, and no secondary index to maintain. What it does have is the
//! declared relationship constraints, gated here — ahead of `edge_weight_mut`,
//! which publishes an edge-update capture whether or not a property lands, so
//! a gate placed inside it would log a change the graph refused to make.

use super::super::ast::Expression;
use super::super::result::ResultRow;
use super::CypherExecutor;
use crate::datatypes::values::Value;
use crate::graph::languages::cypher::result::MutationStats;
use crate::graph::schema::{DirGraph, EdgeData};
use crate::graph::storage::{GraphRead, GraphWrite};
use std::collections::{HashMap, HashSet};

/// Apply one `SET r.p = …` to a relationship, reporting whether the variable
/// was in fact bound to one.
///
/// Split out of `execute_set` because it shares nothing with the node path it
/// sat inside: an edge has no id/type guard to run, no columnar column to
/// route to and no secondary index to maintain, and the constraint it does
/// have is judged against the incoming value alone.
///
/// `Ok(false)` means "not a relationship variable" and hands the row back to
/// the node path — the node branch's own not-bound diagnostics are what should
/// report an unbound variable, not this one.
pub(super) fn set_edge_property(
    graph: &mut DirGraph,
    row: &ResultRow,
    item: (&String, &String, &Expression),
    params: &HashMap<String, Value>,
    stats: &mut MutationStats,
    edges_to_stamp: &mut HashSet<petgraph::graph::EdgeIndex>,
) -> Result<bool, String> {
    let (variable, property, expression) = item;
    if row.node_bindings.contains_key(variable) {
        return Ok(false);
    }
    let Some(edge_binding) = row.edge_bindings.get(variable) else {
        return Ok(false);
    };
    let edge_index = edge_binding.edge_index;
    let value = {
        let executor = CypherExecutor::with_params(graph, params, None);
        executor.evaluate_expression(expression, row)?
    };

    // Declared relationship constraints, gated before `edge_weight_mut` rather
    // than around the write inside it: that call publishes an edge-update
    // capture whether or not a property lands, so a gate placed any later
    // would log a change the graph refused to make.
    if let Some(rel_type) = constrained_edge_type(graph, edge_index) {
        graph.check_rel_property_write(&rel_type, property, Some(&value))?;
    }

    let key = graph.interner.get_or_intern(property);
    if let Some(EdgeData {
        properties: edge_props,
        ..
    }) = GraphWrite::edge_weight_mut(&mut graph.graph, edge_index)
    {
        // `SET r.p = null` leaves the property absent, not present-and-null —
        // the node rule (a null cell is skipped by the columnar store, so
        // `keys(n)` never reports it), applied to the key/value vector an edge
        // stores instead. `SET r += {p: null}` desugars to this same item, so
        // the map form follows. Counted as a property *set*, matching what the
        // node path reports for the identical statement; `REMOVE r.p` is the
        // spelling that counts a removal.
        if matches!(value, Value::Null) {
            edge_props.retain(|(ek, _)| *ek != key);
        } else if let Some((_, existing)) = edge_props.iter_mut().find(|(ek, _)| *ek == key) {
            *existing = value;
        } else {
            edge_props.push((key, value));
        }
        stats.properties_set += 1;
    }

    // Record for a post-loop updated_at bump if the edge type opted in (skip
    // writes to the reserved key).
    if property != "updated_at" {
        // Arena guard: edge_weight materializes on the disk backend (protocol
        // in disk/graph.rs); scoped so the borrow ends before the next item's
        // &mut uses.
        let ct_key = {
            let _arena_guard = graph.graph.begin_query();
            graph
                .graph
                .edge_weight(edge_index)
                .map(|e| e.connection_type)
        };
        if let Some(ct_key) = ct_key {
            let ct = graph.interner.resolve(ct_key).to_string();
            if graph.auto_timestamp_for_connection(&ct) {
                edges_to_stamp.insert(edge_index);
            }
        }
    }
    Ok(true)
}

/// Remove one property from a relationship, reporting whether the variable was
/// in fact bound to one. The `REMOVE` counterpart of [`set_edge_property`],
/// split out for the same reason.
pub(super) fn remove_edge_property(
    graph: &mut DirGraph,
    row: &ResultRow,
    variable: &str,
    property: &str,
    stats: &mut MutationStats,
) -> Result<bool, String> {
    if row.node_bindings.contains_key(variable) {
        return Ok(false);
    }
    let Some(edge_binding) = row.edge_bindings.get(variable) else {
        return Ok(false);
    };
    let edge_index = edge_binding.edge_index;

    // Removing a required property leaves it absent, which is what NOT NULL
    // forbids. A declared *type* is unaffected: absence satisfies every type.
    // Gated before `edge_weight_mut` for the same no-phantom reason as the SET
    // branch.
    if let Some(rel_type) = constrained_edge_type(graph, edge_index) {
        graph.check_rel_property_write(&rel_type, property, None)?;
    }

    let key = graph.interner.get_or_intern(property);
    if let Some(EdgeData {
        properties: edge_props,
        ..
    }) = GraphWrite::edge_weight_mut(&mut graph.graph, edge_index)
    {
        let before = edge_props.len();
        edge_props.retain(|(ek, _)| *ek != key);
        if edge_props.len() != before {
            stats.properties_removed += 1;
        }
    }
    Ok(true)
}

/// The connection type of `edge_index`, but only when the graph declares a
/// relationship constraint at all.
///
/// The `None` it returns for an unconstrained graph *is* the fast-out: the
/// per-edge type resolution below costs an edge read and an interner lookup,
/// and no write should pay either to discover there was nothing to check. One
/// `is_empty` pair is the whole cost for a graph that declares nothing.
pub(super) fn constrained_edge_type(
    graph: &DirGraph,
    edge_index: petgraph::graph::EdgeIndex,
) -> Option<String> {
    if !graph.has_rel_constraints() {
        return None;
    }
    // Arena guard: `edge_weight` materializes on the disk backend (protocol in
    // disk/graph.rs); scoped so the borrow ends before the caller's `&mut`.
    let connection_type = {
        let _arena_guard = graph.graph.begin_query();
        graph
            .graph
            .edge_weight(edge_index)
            .map(|e| e.connection_type)
    }?;
    let rel_type = graph.interner.resolve(connection_type).to_string();
    graph
        .type_has_rel_constraints(&rel_type)
        .then_some(rel_type)
}
