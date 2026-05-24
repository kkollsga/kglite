//! Type capabilities + endpoint-type discovery helpers.
//!
//! Used by describe() to show what each node type supports (how many
//! properties, which connection types, neighbor types, etc.).

use crate::graph::schema::{DirGraph, InternedKey};
use crate::graph::storage::GraphRead;
use petgraph::Direction;
use std::collections::{HashMap, HashSet};

use super::describe::xml_escape;
use super::schema_overview::compute_neighbors_schema;
use super::{NeighborConnection, NeighborsSchema};

// ── Describe helpers ────────────────────────────────────────────────────────

/// Capability flags for a node type (used by `describe()`).
pub(super) struct TypeCapabilities {
    pub(super) has_timeseries: bool,
    pub(super) has_location: bool,
    pub(super) has_geometry: bool,
    pub(super) has_embeddings: bool,
}

impl TypeCapabilities {
    /// Format inline capability flags: "ts", "geo", "loc", "vec".
    pub(super) fn flags_csv(&self) -> String {
        let mut flags = Vec::new();
        if self.has_timeseries {
            flags.push("ts");
        }
        if self.has_geometry {
            flags.push("geo");
        }
        if self.has_location && !self.has_geometry {
            flags.push("loc");
        }
        if self.has_embeddings {
            flags.push("vec");
        }
        flags.join(",")
    }

    /// Merge another type's capabilities into this one (for bubbling up).
    fn merge(&mut self, other: &TypeCapabilities) {
        self.has_timeseries |= other.has_timeseries;
        self.has_location |= other.has_location;
        self.has_geometry |= other.has_geometry;
        self.has_embeddings |= other.has_embeddings;
    }
}

/// Property complexity marker based on property count.
pub(super) fn property_complexity(count: usize) -> &'static str {
    match count {
        0..=3 => "vl",
        4..=8 => "l",
        9..=15 => "m",
        16..=30 => "h",
        _ => "vh",
    }
}

/// Size tier for node count: vs (<10), s (10-99), m (100-999), l (1K-9999), vl (10K+).
pub(super) fn size_tier(count: usize) -> &'static str {
    match count {
        0..=9 => "vs",
        10..=99 => "s",
        100..=999 => "m",
        1000..=9999 => "l",
        _ => "vl",
    }
}

/// Format a compact type descriptor: `Name[size,complexity,flags]` or `Name[size,complexity]`.
pub(super) fn format_type_descriptor(
    name: &str,
    count: usize,
    prop_count: usize,
    caps: &TypeCapabilities,
) -> String {
    let size = size_tier(count);
    let complexity = property_complexity(prop_count);
    let flags = caps.flags_csv();
    if flags.is_empty() {
        format!("{}[{},{}]", xml_escape(name), size, complexity)
    } else {
        format!("{}[{},{},{}]", xml_escape(name), size, complexity, flags)
    }
}

/// Bubble capabilities from supporting types up to their parent core types.
pub(super) fn bubble_capabilities(
    caps: &mut HashMap<String, TypeCapabilities>,
    parent_types: &HashMap<String, String>,
) {
    // Collect child caps first to avoid borrow issues
    let child_caps: Vec<(String, TypeCapabilities)> = parent_types
        .iter()
        .filter_map(|(child, parent)| {
            caps.get(child).map(|c| {
                (
                    parent.clone(),
                    TypeCapabilities {
                        has_timeseries: c.has_timeseries,
                        has_location: c.has_location,
                        has_geometry: c.has_geometry,
                        has_embeddings: c.has_embeddings,
                    },
                )
            })
        })
        .collect();
    for (parent, child_cap) in &child_caps {
        if let Some(parent_cap) = caps.get_mut(parent) {
            parent_cap.merge(child_cap);
        }
    }
}

/// Count supporting children per parent type.
pub(super) fn children_counts(parent_types: &HashMap<String, String>) -> HashMap<String, usize> {
    let mut counts: HashMap<String, usize> = HashMap::new();
    for parent in parent_types.values() {
        *counts.entry(parent.clone()).or_insert(0) += 1;
    }
    counts
}

/// Detect capabilities for all node types in the graph.
pub(super) fn compute_type_capabilities(graph: &DirGraph) -> HashMap<String, TypeCapabilities> {
    let mut caps: HashMap<String, TypeCapabilities> = HashMap::new();

    for node_type in graph.type_indices.keys() {
        let mut tc = TypeCapabilities {
            has_timeseries: false,
            has_location: false,
            has_geometry: false,
            has_embeddings: false,
        };

        // Timeseries
        tc.has_timeseries = graph.timeseries_configs.contains_key(node_type);

        // Spatial
        if let Some(sc) = graph.spatial_configs.get(node_type) {
            tc.has_location = sc.location.is_some() || !sc.points.is_empty();
            tc.has_geometry = sc.geometry.is_some() || !sc.shapes.is_empty();
        }

        // Also check metadata for point-type fields (no SpatialConfig set)
        if !tc.has_location {
            if let Some(meta) = graph.node_type_metadata.get(node_type) {
                tc.has_location = meta.values().any(|t| t.eq_ignore_ascii_case("point"));
            }
        }

        // Embeddings
        tc.has_embeddings = graph.embeddings.keys().any(|(nt, _)| nt == node_type);

        caps.insert(node_type.to_string(), tc);
    }
    caps
}

/// Detect capabilities for specific node types only (avoids scanning all types).
pub(super) fn compute_type_capabilities_for(
    graph: &DirGraph,
    type_names: &[&str],
) -> HashMap<String, TypeCapabilities> {
    let mut caps: HashMap<String, TypeCapabilities> = HashMap::new();

    for &node_type in type_names {
        if !graph.type_indices.contains_key(node_type) {
            continue;
        }
        let mut tc = TypeCapabilities {
            has_timeseries: false,
            has_location: false,
            has_geometry: false,
            has_embeddings: false,
        };

        tc.has_timeseries = graph.timeseries_configs.contains_key(node_type);

        if let Some(sc) = graph.spatial_configs.get(node_type) {
            tc.has_location = sc.location.is_some() || !sc.points.is_empty();
            tc.has_geometry = sc.geometry.is_some() || !sc.shapes.is_empty();
        }
        if !tc.has_location {
            if let Some(meta) = graph.node_type_metadata.get(node_type) {
                tc.has_location = meta.values().any(|t| t.eq_ignore_ascii_case("point"));
            }
        }

        tc.has_embeddings = graph.embeddings.keys().any(|(nt, _)| nt == node_type);

        caps.insert(node_type.to_string(), tc);
    }
    caps
}

/// Compute neighbor schema by sampling first `max_nodes` nodes of a type.
/// Extrapolates counts to full population. Used for types too large to scan fully.
pub(super) fn compute_neighbors_schema_sampled(
    graph: &DirGraph,
    node_type: &str,
    max_nodes: usize,
) -> Result<NeighborsSchema, String> {
    let node_indices = graph
        .type_indices
        .get(node_type)
        .ok_or_else(|| format!("Node type '{}' not found", node_type))?;

    let total_nodes = node_indices.len();
    let sample_count = max_nodes.min(total_nodes);
    if sample_count == 0 {
        return Ok(NeighborsSchema {
            outgoing: Vec::new(),
            incoming: Vec::new(),
        });
    }

    let mut outgoing: HashMap<(String, String), usize> = HashMap::new();
    let mut incoming: HashMap<(String, String), usize> = HashMap::new();

    let g = &graph.graph;
    for node_idx in node_indices.iter().take(sample_count) {
        for edge_ref in g.edges_directed(node_idx, Direction::Outgoing) {
            if let Some(target_node) = graph.get_node(edge_ref.target()) {
                let key = (
                    edge_ref
                        .weight()
                        .connection_type_str(&graph.interner)
                        .to_string(),
                    target_node.node_type_str(&graph.interner).to_string(),
                );
                *outgoing.entry(key).or_insert(0) += 1;
            }
        }
        for edge_ref in g.edges_directed(node_idx, Direction::Incoming) {
            if let Some(source_node) = graph.get_node(edge_ref.source()) {
                let key = (
                    edge_ref
                        .weight()
                        .connection_type_str(&graph.interner)
                        .to_string(),
                    source_node.node_type_str(&graph.interner).to_string(),
                );
                *incoming.entry(key).or_insert(0) += 1;
            }
        }
    }

    // Extrapolate counts to full population
    let scale = if sample_count < total_nodes {
        total_nodes as f64 / sample_count as f64
    } else {
        1.0
    };

    let mut outgoing_list: Vec<NeighborConnection> = outgoing
        .into_iter()
        .map(|((ct, ot), count)| NeighborConnection {
            connection_type: ct,
            other_type: ot,
            count: (count as f64 * scale).round() as usize,
        })
        .collect();
    outgoing_list.sort_by(|a, b| {
        b.count
            .cmp(&a.count)
            .then_with(|| a.connection_type.cmp(&b.connection_type))
    });

    let mut incoming_list: Vec<NeighborConnection> = incoming
        .into_iter()
        .map(|((ct, ot), count)| NeighborConnection {
            connection_type: ct,
            other_type: ot,
            count: (count as f64 * scale).round() as usize,
        })
        .collect();
    incoming_list.sort_by(|a, b| {
        b.count
            .cmp(&a.count)
            .then_with(|| a.connection_type.cmp(&b.connection_type))
    });

    Ok(NeighborsSchema {
        outgoing: outgoing_list,
        incoming: incoming_list,
    })
}

/// Bounded neighbor schema: samples if type has more than `threshold` nodes.
pub(super) fn compute_neighbors_schema_bounded(
    graph: &DirGraph,
    node_type: &str,
    sample_threshold: usize,
) -> Result<NeighborsSchema, String> {
    let count = graph
        .type_indices
        .get(node_type)
        .map(|v| v.len())
        .unwrap_or(0);
    if count > sample_threshold {
        compute_neighbors_schema_sampled(graph, node_type, 10_000)
    } else {
        compute_neighbors_schema(graph, node_type)
    }
}

/// Bounded single-pass endpoint type discovery for connection types with empty metadata.
/// Scans up to `max_total_scan` edges total, collecting source/target node types per connection type.
pub(super) fn discover_endpoint_types_batch(
    graph: &DirGraph,
    max_total_scan: usize,
) -> HashMap<String, (HashSet<String>, HashSet<String>)> {
    // Use edge_endpoint_keys for zero-allocation iteration on disk graphs
    let mut result: HashMap<InternedKey, (HashSet<InternedKey>, HashSet<InternedKey>)> =
        HashMap::new();

    for (scanned, (src_idx, tgt_idx, conn_key)) in graph.graph.edge_endpoint_keys().enumerate() {
        if scanned >= max_total_scan {
            break;
        }
        let entry = result
            .entry(conn_key)
            .or_insert_with(|| (HashSet::new(), HashSet::new()));
        if let Some(sk) = graph.graph.node_type_of(src_idx) {
            entry.0.insert(sk);
        }
        if let Some(tk) = graph.graph.node_type_of(tgt_idx) {
            entry.1.insert(tk);
        }
    }

    // Resolve interned keys to strings
    result
        .into_iter()
        .map(|(ck, (srcs, tgts))| {
            let conn = graph.interner.resolve(ck).to_string();
            let src_set = srcs
                .into_iter()
                .map(|k| graph.interner.resolve(k).to_string())
                .collect();
            let tgt_set = tgts
                .into_iter()
                .map(|k| graph.interner.resolve(k).to_string())
                .collect();
            (conn, (src_set, tgt_set))
        })
        .collect()
}
