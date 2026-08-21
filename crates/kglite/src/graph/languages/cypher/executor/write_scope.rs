//! The `write_scope` perimeter — role-scoped write authorization.
//!
//! One module because it is one rule with several spellings, and every Cypher
//! write verb has to reach the same answer. Splitting it out of `write.rs`
//! also keeps the refusal *message* in one place: an agent reads the error, so
//! a second wording would be a second contract.
//!
//! The rule, in full:
//!
//! - A **node** write is judged by the node's **stored type**. That is the
//!   label-smuggling defence: `SET n:Task` cannot talk an `Algorithm` into
//!   scope, because nothing here reads a pattern label.
//! - A **relationship** write is allowed iff **at least one endpoint's stored
//!   type is in scope** (rationale on [`enforce_edge_write_scope`]).
//! - `DETACH DELETE`'s incident-edge collateral is authorized by the node
//!   delete and not re-checked per far endpoint (note at the
//!   `detach_delete_nodes` call in `write::execute_delete`).
//!
//! `None` = unrestricted, and every entry point fast-outs on it before doing
//! any work; an empty set denies everything.

use super::super::result::EdgeBinding;
use crate::graph::schema::DirGraph;
use crate::graph::storage::GraphRead;
use petgraph::graph::NodeIndex;
use std::collections::HashSet;

/// Enforce the whitelist against a node type the statement is about to write.
///
/// The direct entry point for the verbs that *carry* their own type —
/// `CREATE`, `SET n.p`, `REMOVE n.p`, node-type schema DDL. Verbs that only
/// have a node index reach the same check through
/// [`enforce_node_write_scope`]. `None` = unrestricted (the common case; a
/// single `Option` check with no allocation). See
/// [`crate::graph::DirGraph::active_write_scope`].
pub(super) fn enforce_write_scope(graph: &DirGraph, node_type: &str) -> Result<(), String> {
    if let Some(scope) = &graph.active_write_scope {
        if !scope.contains(node_type) {
            return Err(format!(
                "write scope violation: node type '{}' is not in the allowed write set ({})",
                node_type,
                allowed_write_set(scope)
            ));
        }
    }
    Ok(())
}

/// The whitelist as a sorted, comma-joined list — the tail of every refusal
/// message, and stable across the `HashSet`'s iteration order.
fn allowed_write_set(scope: &HashSet<String>) -> String {
    let mut types: Vec<&str> = scope.iter().map(|s| s.as_str()).collect();
    types.sort_unstable();
    types.join(", ")
}

/// A node's **stored** type — the one the scope check judges. Reading the
/// stored type (rather than a pattern label) is what defeats label smuggling:
/// `MATCH (n:Algorithm) SET n:Task REMOVE n.x` cannot talk its way into scope.
fn stored_node_type(graph: &DirGraph, node_idx: NodeIndex) -> String {
    // Arena guard: node_view materializes on the disk backend (protocol in
    // disk/graph.rs); scoped so the borrow ends before the caller's `&mut`.
    let _arena_guard = graph.graph.begin_query();
    graph
        .node_view(node_idx)
        .map(|n| n.get_node_type_ref(&graph.interner).to_string())
        .unwrap_or_default()
}

/// Enforce the write whitelist against an **existing** node's stored type.
///
/// This is the verb-agnostic half of the perimeter: `DELETE`, `DETACH DELETE`,
/// `REMOVE n.p`, `REMOVE n:L` and `SET n:L` all mutate a node the statement
/// did not create, so none of them carries a type of its own to check.
/// (`CREATE` and `SET n.p` call [`enforce_write_scope`] directly with the type
/// they are about to write.)
pub(super) fn enforce_node_write_scope(
    graph: &DirGraph,
    node_idx: NodeIndex,
) -> Result<(), String> {
    if graph.active_write_scope.is_none() {
        return Ok(());
    }
    enforce_write_scope(graph, &stored_node_type(graph, node_idx))
}

/// Enforce the write whitelist against a relationship write (`CREATE` of an
/// edge, `DELETE r`, `SET r.p`, `REMOVE r.p`).
///
/// **The rule: at least one endpoint's stored type must be in scope.** An edge
/// is not owned by either endpoint alone, and linking a node the role owns to a
/// *matched* out-of-scope node does not mutate that node — that is the
/// load-bearing agent-contract pattern (link a runtime `Task` to a managed
/// `AlgorithmSpec`), and it stays allowed. What the one-endpoint rule adds is
/// the other half: a relationship between two nodes the role owns *nothing* in
/// is a write it has no standing to make, so edge forgery between two
/// out-of-scope nodes is refused. A newly created endpoint is in scope by
/// construction — its node `CREATE` went through [`enforce_write_scope`] — so
/// this never contradicts the node rule.
///
/// `DETACH DELETE`'s incident-edge collateral is deliberately *not* routed
/// here; see the note at the `detach_delete_nodes` call in `execute_delete`.
pub(super) fn enforce_edge_write_scope(
    graph: &DirGraph,
    rel_type: &str,
    source: NodeIndex,
    target: NodeIndex,
) -> Result<(), String> {
    let Some(scope) = &graph.active_write_scope else {
        return Ok(());
    };
    let source_type = stored_node_type(graph, source);
    let target_type = stored_node_type(graph, target);
    if scope.contains(&source_type) || scope.contains(&target_type) {
        return Ok(());
    }
    Err(format!(
        "write scope violation: relationship '{}' connects '{}' to '{}' and neither endpoint \
         type is in the allowed write set ({})",
        rel_type,
        source_type,
        target_type,
        allowed_write_set(scope)
    ))
}

/// [`enforce_edge_write_scope`] for an edge already in the graph, which has to
/// resolve its own connection type for the refusal message. The type read is
/// paid for only when a scope is active.
pub(super) fn enforce_bound_edge_write_scope(
    graph: &DirGraph,
    binding: &EdgeBinding,
) -> Result<(), String> {
    if graph.active_write_scope.is_none() {
        return Ok(());
    }
    let rel_type = {
        // Arena guard: edge_weight materializes on the disk backend.
        let _arena_guard = graph.graph.begin_query();
        graph
            .graph
            .edge_weight(binding.edge_index)
            .map(|e| e.connection_type)
    }
    .map(|key| graph.interner.resolve(key).to_string())
    .unwrap_or_default();
    enforce_edge_write_scope(graph, &rel_type, binding.source, binding.target)
}
