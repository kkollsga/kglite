//! Load-time advisories for files written by builds with known data-shape bugs.
//!
//! A `.kgl` outlives the build that wrote it, and a few past releases wrote a
//! shape later fixed (folded history edges, per-row parent-edge copies, a
//! mis-spelled implicit parent edge). Nothing in the loaded graph says so, and
//! the stamped writer version alone is a poor signal both ways: a clean graph
//! can carry an affected version and an affected graph can be re-saved under
//! a fixed one. Every advisory here therefore needs *both* a writer-version
//! precondition (against the oldest writer, `SaveMetadata::oldest_writer`) and
//! a cheap data predicate over metadata the load already holds.

use crate::graph::dir_graph::DirGraph;
use crate::graph::storage::GraphRead;
use petgraph::Direction;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

/// Most source nodes the duplicate-edge confirmation walks per type.
const CONFIRM_SCAN_NODES: usize = 200_000;

/// Code of the advisory a durable open raises when it set a damaged
/// write-ahead log aside; `affected` holds the quarantine file's path.
pub(crate) const WAL_QUARANTINED: &str = "wal_quarantined";

/// Code of the advisory a durable open raises when it cut a non-zero log tail
/// and saved it; `affected` holds the saved file's path.
pub(crate) const WAL_TAIL_SAVED: &str = "wal_tail_saved";

/// One reason to distrust a loaded file's data shape.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DataAdvisory {
    /// Stable code: `folded_history_edges`, `timeseries_parent_copies`,
    /// `implicit_parent_edge_duplicates`, `wal_quarantined` or `wal_tail_saved`.
    pub code: String,
    /// The oldest library version that wrote the file's data.
    pub writer: String,
    pub message: String,
    /// The node or relationship types the data predicate matched.
    pub affected: Vec<String>,
}

/// The advisories computed when `graph` was loaded; empty for a graph built in
/// this process, a file written by a fixed build, or data the predicates find
/// clean.
pub fn data_advisories(graph: &DirGraph) -> Vec<DataAdvisory> {
    graph.advisories.clone()
}

/// `major.minor.patch` of a version string, ignoring a pre-release or build
/// suffix; `None` for anything else (empty, unknown).
pub(crate) fn parse_version(version: &str) -> Option<(u32, u32, u32)> {
    let core = version.split(['-', '+']).next()?;
    let mut parts = core.split('.');
    let major = parts.next()?.parse().ok()?;
    let minor = parts.next()?.parse().ok()?;
    let patch = parts.next()?.parse().ok()?;
    parts.next().is_none().then_some((major, minor, patch))
}

/// Run every advisory check against a freshly loaded graph.
pub(crate) fn compute_advisories(graph: &DirGraph) -> Vec<DataAdvisory> {
    let writer = &graph.save_metadata.oldest_writer;
    let Some(version) = parse_version(writer) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    if version < (0, 19, 0) {
        out.extend(folded_history_edges(graph, writer));
    }
    if ((0, 19, 0)..(0, 19, 2)).contains(&version) {
        out.extend(timeseries_parent_copies(graph, writer));
    }
    if version == (0, 19, 2) {
        out.extend(implicit_parent_edge_duplicates(graph, writer));
    }
    out
}

/// A1: before 0.19.0 a relationship type with several source types and
/// recorded properties may have had its history versions folded into one edge.
fn folded_history_edges(graph: &DirGraph, writer: &str) -> Option<DataAdvisory> {
    let mut affected: Vec<String> = graph
        .connection_type_metadata
        .iter()
        .filter(|(_, info)| info.source_types.len() >= 2 && !info.property_types.is_empty())
        .map(|(name, _)| name.clone())
        .collect();
    if affected.is_empty() {
        return None;
    }
    affected.sort();
    Some(DataAdvisory {
        code: "folded_history_edges".into(),
        writer: writer.into(),
        message: format!(
            "written by kglite {writer}: relationship types with several source types and \
             properties ({}) may have folded history edges; rebuild the graph from its source \
             with the current version to restore them",
            preview(&affected)
        ),
        affected,
    })
}

/// A2: 0.19.0 and 0.19.1 wrote one parent edge per time-series row, so a
/// time-series type's edges to a parent type outnumber its source nodes and
/// some (source, target, type) pair repeats.
fn timeseries_parent_copies(graph: &DirGraph, writer: &str) -> Option<DataAdvisory> {
    let mut affected: Vec<String> = graph
        .timeseries_configs
        .keys()
        .filter(|node_type| has_duplicate_edge_pair(graph, node_type))
        .cloned()
        .collect();
    if affected.is_empty() {
        return None;
    }
    affected.sort();
    Some(DataAdvisory {
        code: "timeseries_parent_copies".into(),
        writer: writer.into(),
        message: format!(
            "written by kglite {writer}: time-series types ({}) carry repeated edges to their \
             parent nodes, so counts across those edges are inflated; rebuild the graph from \
             its source with the current version",
            preview(&affected)
        ),
        affected,
    })
}

/// Whether any node of `node_type` has two outgoing edges of one type to the
/// same target.
fn has_duplicate_edge_pair(graph: &DirGraph, node_type: &str) -> bool {
    let Some(nodes) = graph.type_indices.get(node_type) else {
        return false;
    };
    for node in nodes.iter().take(CONFIRM_SCAN_NODES) {
        let mut seen = HashSet::new();
        for edge in graph.graph.edges_directed(node, Direction::Outgoing) {
            if !seen.insert((edge.connection_type(), edge.target())) {
                return true;
            }
        }
    }
    false
}

/// `OF_` + the type name upper-cased with no word breaks, the spelling 0.19.2's
/// implicit parent edge used (`OF_PROJECTPHASE`).
fn unsplit_parent_edge_name(parent_type: &str) -> String {
    format!("OF_{}", parent_type.to_uppercase())
}

/// A3: 0.19.2's implicit parent edge spelled a multi-word parent without word
/// breaks, so a type that also links to the parent through its own edge got a
/// second edge of similar count.
fn implicit_parent_edge_duplicates(graph: &DirGraph, writer: &str) -> Option<DataAdvisory> {
    let candidates = unsplit_edge_candidates(graph);
    if candidates.is_empty() {
        return None;
    }
    // Counts come from the connectivity cache (computed here only when a
    // candidate shape exists, i.e. for a file that already looks affected).
    let triples = graph.get_or_compute_type_connectivity();
    let count = |src: &str, conn: &str, tgt: &str| {
        triples
            .iter()
            .find(|t| t.src == src && t.conn == conn && t.tgt == tgt)
            .map_or(0, |t| t.count)
    };
    let mut affected: Vec<String> = candidates
        .into_iter()
        .filter(|(src, conn, tgt, twins)| {
            let implicit = count(src, conn, tgt);
            twins
                .iter()
                .any(|twin| similar_count(count(src, twin, tgt), implicit))
        })
        .map(|(src, conn, tgt, _)| format!("{src} -[{conn}]-> {tgt}"))
        .collect();
    if affected.is_empty() {
        return None;
    }
    affected.sort();
    Some(DataAdvisory {
        code: "implicit_parent_edge_duplicates".into(),
        writer: writer.into(),
        message: format!(
            "written by kglite 0.19.2: its implicit parent edge duplicated an explicit one \
             ({}) and may have created placeholder parent nodes; rebuild the graph from its \
             source with the current version (purge_provisional() removes leftover \
             placeholders)",
            preview(&affected)
        ),
        affected,
    })
}

/// `(source type, unsplit edge name, parent type, other edge types the same
/// pair is linked by)` for every relationship type spelled like 0.19.2's
/// implicit edge to a multi-word parent, read from the type metadata alone.
fn unsplit_edge_candidates(graph: &DirGraph) -> Vec<(String, String, String, Vec<String>)> {
    let mut out = Vec::new();
    for (conn, info) in graph.connection_type_metadata.iter() {
        for parent in &info.target_types {
            if *conn != unsplit_parent_edge_name(parent) || !is_multi_word(parent) {
                continue;
            }
            for source in &info.source_types {
                let twins: Vec<String> = graph
                    .connection_type_metadata
                    .iter()
                    .filter(|(other, o)| {
                        *other != conn
                            && o.source_types.contains(source)
                            && o.target_types.contains(parent)
                    })
                    .map(|(other, _)| other.clone())
                    .collect();
                if !twins.is_empty() {
                    out.push((source.clone(), conn.clone(), parent.clone(), twins));
                }
            }
        }
    }
    out
}

/// A CamelCase name with a word break (`ProjectPhase`), not `Team`/`HQ`.
fn is_multi_word(name: &str) -> bool {
    let chars: Vec<char> = name.chars().collect();
    chars
        .windows(2)
        .any(|w| w[0].is_lowercase() && w[1].is_uppercase())
}

/// Counts within a factor of two of each other.
fn similar_count(a: usize, b: usize) -> bool {
    a > 0 && b > 0 && a <= b.saturating_mul(2) && b <= a.saturating_mul(2)
}

fn preview(names: &[String]) -> String {
    const SHOWN: usize = 5;
    let mut text = names
        .iter()
        .take(SHOWN)
        .cloned()
        .collect::<Vec<_>>()
        .join(", ");
    if names.len() > SHOWN {
        text.push_str(&format!(", and {} more", names.len() - SHOWN));
    }
    text
}
