//! Write targets that reach a clause as projected VALUES rather than live
//! bindings — `UNWIND collect(n) AS x`, a `FOREACH` loop variable, a `WITH`
//! that carried a node or relationship forward.
//!
//! openCypher reuses an already-bound variable, and a value is that binding
//! in another shape. Before these resolvers every write clause treated such a
//! variable as unbound: `CREATE` fabricated a second node, `MERGE` matched
//! nothing and created, `SET` / `REMOVE` refused the name, `DELETE` was a
//! silent no-op. A value can also outlive the slot it names (a row deleted
//! earlier in the same statement, a value round-tripped through a parameter
//! from another graph), so each resolver verifies the slot before handing it
//! out, and a failure is an error rather than a fall-through to "create".

use super::relationship_identity::StatementRelationshipIdentities;
use crate::datatypes::values::{RelValue, Value};
use crate::graph::languages::cypher::ast::{RemoveClause, RemoveItem, SetClause, SetItem};
use crate::graph::languages::cypher::result::{EdgeBinding, ResultRow, ResultSet};
use crate::graph::schema::DirGraph;
use crate::graph::storage::GraphRead;
use petgraph::graph::{EdgeIndex, NodeIndex};

/// The node a variable names when it reaches this clause only as a projected
/// VALUE. `Ok(None)` means "not a node value": the caller falls back to
/// whatever it does for an unbound name.
///
/// The `Value::Node` shape records its labels, so its slot is checked for
/// liveness *and* for still holding a node of the same primary type; the
/// transient `NodeRef` shape (`collect(a)[0]`, `head(…)`) carries only the
/// index, so only liveness is checkable. Write scope is deliberately *not*
/// checked here: the callers apply the same gate they apply to a
/// `node_bindings` target, so a projected target is authorized exactly as a
/// bound one is.
pub(super) fn projected_node_target(
    graph: &DirGraph,
    row: &ResultRow,
    variable: &str,
) -> Result<Option<NodeIndex>, String> {
    match row.projected.get(variable) {
        Some(Value::Node(node)) => {
            let node_idx = NodeIndex::new(node.id as usize);
            // `labels[0]` is the primary type (`DirGraph::node_labels` pushes
            // it before the sorted secondaries).
            verify_projected_node_slot(graph, variable, node_idx, node.labels.first())?;
            Ok(Some(node_idx))
        }
        Some(Value::NodeRef(index)) => {
            let node_idx = NodeIndex::new(*index as usize);
            verify_projected_node_slot(graph, variable, node_idx, None)?;
            Ok(Some(node_idx))
        }
        // A relationship value in a node position is a type error, not a
        // request for a fresh node: `CREATE (r)-[:S]->(m)` over collected
        // relationships used to create an anonymous node per row and wire the
        // relationship nowhere near it.
        Some(Value::Relationship(_)) => Err(format!(
            "Variable '{variable}' holds a relationship, not a node"
        )),
        _ => Ok(None),
    }
}

fn verify_projected_node_slot(
    graph: &DirGraph,
    variable: &str,
    node_idx: NodeIndex,
    primary_label: Option<&String>,
) -> Result<(), String> {
    let Some(node) = graph.graph.node_view(node_idx) else {
        return Err(format!(
            "Node value '{variable}' no longer exists in this graph"
        ));
    };
    if let Some(label) = primary_label {
        let live = node.node_type_str(&graph.interner);
        if live != label.as_str() {
            return Err(format!(
                "Node value '{variable}' no longer occupies the storage slot it names \
                 (expected :{label}, found :{live})"
            ));
        }
    }
    Ok(())
}

/// The node a SET / REMOVE item writes to. `Ok(None)` is openCypher's
/// null-target no-op for this row (an unmatched OPTIONAL MATCH variable or an
/// explicit NULL projection); an `Err` names a target that is neither a node
/// nor null. `clause` is the clause name the diagnostic quotes.
pub(super) fn resolve_node_write_target(
    graph: &DirGraph,
    row: &ResultRow,
    variable: &str,
    clause: &str,
) -> Result<Option<NodeIndex>, String> {
    if let Some(&node_idx) = row.node_bindings.get(variable) {
        return Ok(Some(node_idx));
    }
    if let Some(node_idx) = projected_node_target(graph, row, variable)? {
        return Ok(Some(node_idx));
    }
    if is_null_write_target(row, variable) {
        return Ok(None);
    }
    Err(format!(
        "Variable '{variable}' not bound to a node in {clause}"
    ))
}

/// True when `variable` is a bound-but-null write target on this row —
/// e.g. an unmatched OPTIONAL MATCH variable (no binding at all) or an
/// explicit NULL projection. openCypher: SET / REMOVE on a NULL target is
/// a no-op for that row, mirroring how DELETE already skips NULLs. A
/// *truly undefined* name never reaches here — the planner's scope
/// validation (`validate_scope`) rejects it before execution — so any
/// remaining non-entity target that isn't NULL is a genuine type error
/// and the caller keeps returning its descriptive error for it.
pub(super) fn is_null_write_target(row: &ResultRow, variable: &str) -> bool {
    !row.node_bindings.contains_key(variable)
        && !row.edge_bindings.contains_key(variable)
        && !row.path_bindings.contains_key(variable)
        && matches!(row.projected.get(variable), None | Some(Value::Null))
}

/// The binding a projected relationship value stands for, once verified: it
/// carries a token this statement issued, the token is still current for the
/// slot (a DELETE earlier in the statement retires it), and the slot's
/// endpoints are the ones the value records — a value from another graph or
/// a rebuilt slot can still name a live edge that the caller never selected.
/// Write scope is the caller's gate, as for [`projected_node_target`].
pub(super) fn projected_edge_binding(
    graph: &DirGraph,
    variable: &str,
    rel: &RelValue,
    identities: &StatementRelationshipIdentities,
) -> Result<EdgeBinding, String> {
    let edge_index = EdgeIndex::new(rel.id as usize);
    let token = rel.incarnation.ok_or_else(|| {
        format!("Relationship value '{variable}' was not bound by this statement")
    })?;
    if !identities.accepts(edge_index, token) {
        return Err(format!(
            "Relationship '{variable}' is stale after its storage slot was reused"
        ));
    }
    let binding = EdgeBinding {
        incarnation: Some(token),
        source: NodeIndex::new(rel.start_id as usize),
        target: NodeIndex::new(rel.end_id as usize),
        edge_index,
    };
    if graph.graph.edge_endpoints(edge_index) != Some((binding.source, binding.target)) {
        return Err(format!(
            "Relationship '{variable}' no longer occupies the storage slot it names"
        ));
    }
    Ok(binding)
}

/// The rows a SET or REMOVE clause runs over when any of them names a
/// projected relationship value: a copy with every such value promoted to a
/// verified `edge_bindings` entry. The relationship write paths key on
/// `edge_bindings` alone, so this is the one place a value becomes a target.
/// `None` when no row names one — the caller keeps its own rows, uncopied.
pub(super) fn promote_projected_relationships(
    graph: &DirGraph,
    result_set: &ResultSet,
    variables: &[&str],
    identities: &StatementRelationshipIdentities,
) -> Result<Option<ResultSet>, String> {
    let names_projected_relationship = |row: &ResultRow| {
        variables.iter().any(|&variable| {
            !row.edge_bindings.contains_key(variable)
                && matches!(row.projected.get(variable), Some(Value::Relationship(_)))
        })
    };
    if !result_set.rows.iter().any(names_projected_relationship) {
        return Ok(None);
    }
    let mut rows = result_set.rows.clone();
    for row in &mut rows {
        for &variable in variables {
            if row.edge_bindings.contains_key(variable) {
                continue;
            }
            if let Some(Value::Relationship(rel)) = row.projected.get(variable) {
                let binding = projected_edge_binding(graph, variable, rel, identities)?;
                row.edge_bindings.insert(variable.to_string(), binding);
            }
        }
    }
    Ok(Some(ResultSet {
        rows,
        columns: result_set.columns.clone(),
        lazy_return_items: None,
    }))
}

pub(super) fn set_clause_variables(set: &SetClause) -> Vec<&str> {
    set.items
        .iter()
        .map(|item| match item {
            SetItem::Property { variable, .. }
            | SetItem::Label { variable, .. }
            | SetItem::Map { variable, .. } => variable.as_str(),
        })
        .collect()
}

/// Re-materialise the node and relationship VALUES a SET / REMOVE clause just
/// wrote through, so the rows it hands downstream read the written state. A
/// projected `Value::Node` / `Value::Relationship` is a snapshot taken when
/// it was projected (`UNWIND collect(n) AS x`); the write reaches the stored
/// element, and without this refresh `x.name` and `properties(x)` after the
/// clause answer the pre-write value. Values held inside lists or maps are
/// left alone — they are snapshots by construction. A slot that no longer
/// holds the element keeps its old value.
pub(super) fn refresh_written_projections(
    graph: &DirGraph,
    result_set: &mut ResultSet,
    variables: &[&str],
) {
    for row in &mut result_set.rows {
        for &variable in variables {
            let refreshed = match row.projected.get(variable) {
                Some(Value::Node(node)) => {
                    super::helpers::materialize_node_value(NodeIndex::new(node.id as usize), graph)
                        .map(|fresh| Value::Node(Box::new(fresh)))
                }
                Some(Value::Relationship(rel)) => {
                    super::helpers::materialize_rel_value_with_incarnation(
                        EdgeIndex::new(rel.id as usize),
                        graph,
                        rel.incarnation,
                    )
                    .map(|fresh| Value::Relationship(Box::new(fresh)))
                }
                _ => None,
            };
            if let Some(fresh) = refreshed {
                row.projected.insert(variable.to_string(), fresh);
            }
        }
    }
}

pub(super) fn remove_clause_variables(remove: &RemoveClause) -> Vec<&str> {
    remove
        .items
        .iter()
        .map(|item| match item {
            RemoveItem::Property { variable, .. } | RemoveItem::Label { variable, .. } => {
                variable.as_str()
            }
        })
        .collect()
}
