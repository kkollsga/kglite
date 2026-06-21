//! `CALL dead_code({include_public: false, include_tests: false}) YIELD node`
//!
//! Reports `Function` nodes with no inbound *use* edge — the graph-native
//! answer to "what code is never reached" (the capability CodeGraphContext
//! ships as a Python `find_dead_code` tool; here it is one Cypher CALL over
//! the graph we already build).
//!
//! A function is a candidate when nothing CALLS it, nothing references it
//! as a value (REFERENCES_FN — a callback passed by reference), no Route
//! HANDLES it, no Procedure is IMPLEMENTED_BY it, and it isn't tied into a
//! framework via DECORATES (either as a decorator or a decoratee). Those
//! edge types are exactly the false positives a naive "no inbound CALLS"
//! query reports — bundling them is why this earns a procedure over raw
//! Cypher (it's `orphan_node` generalised across every "is used" edge type
//! plus the entry-point heuristics below).
//!
//! Always excluded as implicit entry points: test functions (pass
//! `include_tests: true` to keep them), dunder methods (`__x__`, called
//! implicitly by the language), and `main`. Public / exported functions are
//! *included by default* — in Python every non-underscore name is nominally
//! "public", so filtering on visibility by default would hide almost
//! everything; Rust-style codebases where `pub` is a real API marker can
//! pass `exclude_public: true` to drop them. Yield the node and refine in
//! Cypher as needed — e.g. `CALL dead_code() YIELD node RETURN
//! node.qualified_name, node.file_path ORDER BY node.file_path`.
//!
//! @procedure: dead_code

use std::collections::HashMap;

use petgraph::graph::NodeIndex;
use petgraph::Direction;

use super::super::ast::YieldItem;
use super::super::result::ResultRow;
use super::rule_procedures::{make_node_row, require_node_yield, type_indices};
use crate::datatypes::values::Value;
use crate::graph::dir_graph::DirGraph;
use crate::graph::schema::InternedKey;
use crate::graph::storage::GraphRead;

const PROC: &str = "dead_code";

/// True if any edge proves the function is reachable / used: an inbound
/// CALLS / REFERENCES_FN / HANDLES / IMPLEMENTED_BY / DECORATES, or an
/// outbound DECORATES (the function is itself a decorator applied elsewhere).
fn is_used(graph: &DirGraph, nidx: NodeIndex) -> bool {
    let calls = InternedKey::from_str("CALLS");
    let refs_fn = InternedKey::from_str("REFERENCES_FN");
    let handles = InternedKey::from_str("HANDLES");
    let implemented_by = InternedKey::from_str("IMPLEMENTED_BY");
    let decorates = InternedKey::from_str("DECORATES");

    for er in graph.graph.edges_directed(nidx, Direction::Incoming) {
        let k = er.weight().connection_type;
        if k == calls || k == refs_fn || k == handles || k == implemented_by || k == decorates {
            return true;
        }
    }
    for er in graph.graph.edges_directed(nidx, Direction::Outgoing) {
        if er.weight().connection_type == decorates {
            return true;
        }
    }
    false
}

fn bool_param(params: &HashMap<String, Value>, key: &str) -> bool {
    matches!(params.get(key), Some(Value::Boolean(true)))
}

/// Short name = terminal segment of a `.`/`::`-separated qualified name.
fn short_name(qname: &str) -> &str {
    qname.rsplit(['.', ':']).next().unwrap_or(qname)
}

pub(super) fn execute_dead_code(
    graph: &DirGraph,
    params: &HashMap<String, Value>,
    yield_items: &[YieldItem],
) -> Result<Vec<ResultRow>, String> {
    let yield_var = require_node_yield(yield_items, PROC, "node")?;
    let exclude_public = bool_param(params, "exclude_public");
    let include_tests = bool_param(params, "include_tests");

    let funcs = match type_indices(graph, "Function") {
        Ok(f) => f,
        Err(_) => return Ok(Vec::new()), // not a code graph
    };

    const PUBLIC_VIS: &[&str] = &["pub", "public", "export", "exported"];
    let mut rows = Vec::new();
    for nidx in funcs.iter() {
        let node = match graph.graph.node_weight(nidx) {
            Some(n) => n,
            None => continue,
        };

        // Flag-based exclusions first (cheaper than the edge scan).
        if !include_tests
            && matches!(
                node.get_property("is_test").as_deref(),
                Some(Value::Boolean(true))
            )
        {
            continue;
        }
        if exclude_public {
            if let Some(Value::String(v)) = node.get_property("visibility").as_deref() {
                if PUBLIC_VIS.contains(&v.as_str()) {
                    continue;
                }
            }
        }
        // Implicit / entry-point names: `main` and dunder methods (`__x__`)
        // are invoked by the language / runtime, not via a CALLS edge.
        if let Value::String(qn) = node.id().as_ref() {
            let short = short_name(qn);
            if short == "main" || (short.starts_with("__") && short.ends_with("__")) {
                continue;
            }
        }

        if is_used(graph, nidx) {
            continue;
        }
        rows.push(make_node_row(&yield_var, nidx));
    }
    Ok(rows)
}
