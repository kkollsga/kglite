//! `CALL rev_diff({from, to [, node_type]}) YIELD bucket, type, qualified_name, name, file, line`
//!
//! The Cypher-side reader for the multi-rev code graphs built by
//! [`crate::code_tree::rev::build_code_tree_revs`]. Each merged node carries two
//! native list props — `revs: [str]` (the revisions the entity appears in,
//! oldest → newest) and `rev_fp: [int]` (a per-rev shape fingerprint, positionally
//! aligned with `revs`). `rev_diff` reads those two lists straight off each node
//! and classifies the entity's fate between the `from` and `to` revs:
//!
//! - `from` present, `to` absent  → **removed** (existed at `from`, gone by `to`),
//! - `to` present, `from` absent  → **added**,
//! - both present, `rev_fp` differs → **changed** (signature/value/body edit),
//! - both present, `rev_fp` equal   → unchanged (no row).
//!
//! Pure set-and-fingerprint membership, no source re-parse — it reports *that* an
//! entity changed (and its current/newest value via the ordinary property
//! columns), matching the two-graph `diff`'s honest contract. Nodes-only in v1
//! (the `revs`/`rev_fp` columns are node-shaped); edge add/remove is a documented
//! deferral.
//!
//! @procedure: rev_diff

use std::collections::HashMap;

use petgraph::graph::NodeIndex;

use super::super::ast::YieldItem;
use super::super::result::ResultRow;
use super::helpers::yield_alias;
use crate::datatypes::values::Value;
use crate::graph::dir_graph::DirGraph;
use crate::graph::storage::GraphRead;

const PROC: &str = "rev_diff";

pub(super) fn execute_rev_diff(
    graph: &DirGraph,
    params: &HashMap<String, Value>,
    yield_items: &[YieldItem],
) -> Result<Vec<ResultRow>, String> {
    let from_rev = require_string_param(params, "from")?;
    let to_rev = require_string_param(params, "to")?;

    // Optional {node_type} scoping — a string or a list of strings. Absent →
    // every node. Near-free: reuse the per-type index the graph already keeps.
    let node_types = string_list_param(params, "node_type");

    // Column aliases (each individually optional; unknown YIELD names are
    // rejected upstream in execute_call).
    let bucket_var = yield_alias(yield_items, "bucket");
    let type_var = yield_alias(yield_items, "type");
    let qn_var = yield_alias(yield_items, "qualified_name");
    let name_var = yield_alias(yield_items, "name");
    let file_var = yield_alias(yield_items, "file");
    let line_var = yield_alias(yield_items, "line");

    // Candidate node set: the scoped types' indices, or the whole graph.
    let candidates: Vec<NodeIndex> = match &node_types {
        Some(types) => {
            let mut v = Vec::new();
            for t in types {
                if let Some(idxs) = graph.type_indices.get(t.as_str()) {
                    v.extend(idxs.iter());
                }
            }
            v
        }
        None => graph.graph.node_indices().collect(),
    };

    // One pass validates the graph is multi-rev and the two revs exist, and
    // classifies each candidate. `available` collects the union of rev labels
    // across candidates for the unknown-rev error (and the not-multi-rev guard).
    let mut available: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    let mut saw_revs_prop = false;
    let mut hits: Vec<(NodeIndex, &'static str)> = Vec::new();

    for &nidx in &candidates {
        let Some(node) = graph.graph.node_weight(nidx) else {
            continue;
        };
        let Some(Value::List(revs)) = node.get_property_value("revs") else {
            continue;
        };
        saw_revs_prop = true;
        for r in &revs {
            if let Some(s) = r.as_string() {
                available.insert(s);
            }
        }
        let pos_from = list_position(&revs, &from_rev);
        let pos_to = list_position(&revs, &to_rev);
        let bucket = match (pos_from, pos_to) {
            (Some(_), None) => "removed",
            (None, Some(_)) => "added",
            (Some(i), Some(j)) => {
                // Changed ⟺ the per-rev fingerprint diverged. Missing/short
                // rev_fp (types with no defined fingerprint hash to 0) ⇒ equal.
                let fp = match node.get_property_value("rev_fp") {
                    Some(Value::List(items)) => items,
                    _ => Vec::new(),
                };
                match (fp.get(i), fp.get(j)) {
                    (Some(a), Some(b)) if a != b => "changed",
                    _ => continue,
                }
            }
            (None, None) => continue,
        };
        hits.push((nidx, bucket));
    }

    if !saw_revs_prop {
        return Err(format!(
            "CALL {PROC}: this graph has no `revs` property — it is not a multi-rev graph. \
             Build one with code_tree.build(path, revs=['v1', 'v2', ...])."
        ));
    }
    for rev in [&from_rev, &to_rev] {
        if !available.contains(rev) {
            let avail: Vec<&str> = available.iter().map(|s| s.as_str()).collect();
            return Err(format!(
                "CALL {PROC}: revision {rev:?} is not present in this graph. Available revs: [{}].",
                avail.join(", ")
            ));
        }
    }

    // Emit one row per hit, sorted (bucket, qualified_name) for determinism.
    let mut rows: Vec<(String, String, ResultRow)> = Vec::with_capacity(hits.len());
    for (nidx, bucket) in hits {
        let Some(node) = graph.graph.node_weight(nidx) else {
            continue;
        };
        let node_type = node.node_type_str(&graph.interner).to_string();
        let qualified_name = value_to_string(&node.id());
        let name = value_to_string(&node.title());

        let mut row = ResultRow::new();
        if let Some(a) = &bucket_var {
            row.projected
                .insert(a.clone(), Value::String(bucket.to_string()));
        }
        if let Some(a) = &type_var {
            row.projected
                .insert(a.clone(), Value::String(node_type.clone()));
        }
        if let Some(a) = &qn_var {
            row.projected
                .insert(a.clone(), Value::String(qualified_name.clone()));
        }
        if let Some(a) = &name_var {
            row.projected.insert(a.clone(), Value::String(name));
        }
        if let Some(a) = &file_var {
            let file = node.get_property_value("file_path").unwrap_or(Value::Null);
            row.projected.insert(a.clone(), file);
        }
        if let Some(a) = &line_var {
            let line = node
                .get_property_value("line_number")
                .unwrap_or(Value::Null);
            row.projected.insert(a.clone(), line);
        }
        rows.push((bucket.to_string(), qualified_name, row));
    }
    rows.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
    Ok(rows.into_iter().map(|(_, _, row)| row).collect())
}

/// The index of the first list element whose string form equals `needle`.
fn list_position(list: &[Value], needle: &str) -> Option<usize> {
    list.iter().position(|v| match v.as_string() {
        Some(s) => s == needle,
        None => false,
    })
}

/// Render a node id/title `Value` to its plain string form (unquoted).
fn value_to_string(v: &Value) -> String {
    v.as_string().unwrap_or_else(|| v.to_string())
}

/// A required string procedure parameter (map syntax), with a clear error.
fn require_string_param(params: &HashMap<String, Value>, key: &str) -> Result<String, String> {
    match params.get(key) {
        Some(Value::String(s)) => Ok(s.clone()),
        Some(other) => Err(format!(
            "CALL {PROC}: parameter '{key}' must be a string, got {other:?}"
        )),
        None => Err(format!(
            "CALL {PROC}: missing required parameter '{key}'. \
             Use map syntax — e.g. CALL {PROC}({{from: 'v1', to: 'v2'}})."
        )),
    }
}

/// A procedure parameter that may be a single string or a list of strings.
/// Returns `None` when absent or holding no usable strings.
fn string_list_param(params: &HashMap<String, Value>, key: &str) -> Option<Vec<String>> {
    match params.get(key) {
        Some(Value::String(s)) => Some(vec![s.clone()]),
        Some(Value::List(items)) => {
            let v: Vec<String> = items.iter().filter_map(|x| x.as_string()).collect();
            if v.is_empty() {
                None
            } else {
                Some(v)
            }
        }
        _ => None,
    }
}
