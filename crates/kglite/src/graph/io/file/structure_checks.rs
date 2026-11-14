//! The structural half of the eager verify tier. A lazy open checks the
//! checksums of what it reads and the shapes that are cheap; these checks read
//! whole arrays, so they run only for `Verify::Eager` and `verify_graph`.
//! Each returns the section it found wrong and the first offending position.
//! A renamed but still ascending `conn_types` list cannot be detected
//! structurally: nothing in the file records the names a second time.

use std::io;

use petgraph::Direction;

use super::index_sections::{count_peers_dense, count_peers_hashed, PeerCounts};
use super::*;
use crate::graph::schema::InternedKey;
use crate::graph::storage::GraphRead;

/// One failed check: the section (or index) named, and what is wrong.
pub(super) struct StructureFailure {
    pub(super) section: String,
    pub(super) error: io::Error,
}

fn failure(section: &str, message: String) -> StructureFailure {
    StructureFailure {
        section: section.to_string(),
        error: invalid_data(message),
    }
}

/// Every structural failure of `graph`'s mapped sections. A topology whose
/// endpoint array or adjacency fails stops the checks that read through them.
pub(super) fn check_structure(graph: &DirGraph) -> Vec<StructureFailure> {
    let mut failures = Vec::new();
    if let Some(error) = check_column_flags(graph) {
        failures.push(error);
    }
    let Some(frozen) = graph.graph.frozen_base() else {
        return failures;
    };
    if let Err(message) = frozen.check_structure() {
        let section = message
            .split_whitespace()
            .next()
            .filter(|word| word.starts_with("topo:"))
            .unwrap_or("topo:edges")
            .to_string();
        failures.push(failure(&section, message));
        return failures;
    }
    failures.extend(check_peer_counts(graph));
    failures.extend(check_id_indexes(graph));
    failures
}

/// Null and bool flag bytes of every column must be 0 or 1.
fn check_column_flags(graph: &DirGraph) -> Option<StructureFailure> {
    let mut stores = graph.column_stores_by_name();
    stores.sort_by(|a, b| a.0.cmp(b.0));
    stores.into_iter().find_map(|(type_name, store)| {
        store.first_bad_flag().map(|(column, what, row, byte)| {
            failure(
                &format!("col:{type_name}"),
                format!("column {column} of type {type_name}: {what} byte {byte} at row {row}, expected 0 or 1"),
            )
        })
    })
}

/// The recorded `(node, count)` pairs of each connection type against the
/// counts of the edges themselves.
fn check_peer_counts(graph: &DirGraph) -> Option<StructureFailure> {
    let base = graph.graph.frozen_base()?.peer_base()?;
    let actual: Vec<(InternedKey, PeerCounts)> = count_peers_dense(graph)
        .unwrap_or_else(|| count_peers_hashed(graph))
        .into_iter()
        .collect();
    let name_of = |key: u64| {
        graph
            .interner
            .try_resolve(InternedKey::from_u64(key))
            .unwrap_or("?")
            .to_string()
    };
    for key in base.keys() {
        if !actual.iter().any(|(k, _)| k.as_u64() == key) {
            let section = format!("idx:peer:{}", name_of(key));
            return Some(failure(&section, "counts a type that has no edges".into()));
        }
    }
    for (key, (by_target, by_source)) in &actual {
        let Some(peers) = base.get(key.as_u64()) else {
            continue;
        };
        for (suffix, dir, pairs) in [
            ("out", Direction::Outgoing, by_target),
            ("in", Direction::Incoming, by_source),
        ] {
            let recorded = peers.counts(dir);
            let wrong = pairs
                .iter()
                .find(|(node, count)| recorded.get(node) != Some(&i64::from(*count)))
                .map(|(node, _)| *node)
                .or_else(|| (recorded.len() != pairs.len()).then_some(0));
            if let Some(node) = wrong {
                let section = format!("idx:peer:{}:{suffix}", name_of(key.as_u64()));
                return Some(failure(
                    &section,
                    format!("the recorded count disagrees with the edges (first at node {node})"),
                ));
            }
        }
    }
    None
}

/// Every node must be found by its own id in the id index served for its
/// type, and the node found must carry that id. One lookup per node; no new
/// structure is built.
fn check_id_indexes(graph: &DirGraph) -> Option<StructureFailure> {
    let mut types: Vec<_> = graph.type_indices.iter().collect();
    types.sort_by(|a, b| a.0.cmp(b.0));
    for (type_name, members) in types {
        for node in members.iter() {
            let Some(id) = graph.graph.get_node_id(node) else {
                continue;
            };
            let found = graph
                .lookup_by_id_readonly(type_name, &id)
                .and_then(|hit| graph.graph.get_node_id(hit));
            if found.as_ref() != Some(&id) {
                return Some(failure(
                    &format!("idx:id:{type_name}"),
                    format!(
                        "node {} of type {type_name} is not found by its own id {id:?}",
                        node.index()
                    ),
                ));
            }
        }
    }
    None
}
