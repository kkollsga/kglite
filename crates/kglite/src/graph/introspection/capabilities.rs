//! Type capabilities + endpoint-type discovery helpers.
//!
//! Used by describe() to show what each node type supports.

use crate::graph::schema::{DirGraph, InternedKey};
use crate::graph::storage::GraphRead;
use petgraph::Direction;
use std::collections::{HashMap, HashSet};

use super::describe::xml_escape;
use super::schema_overview::compute_neighbors_schema;
use super::{NeighborConnection, NeighborsSchema};

/// What one node type supports, as the four independent facts `describe()`
/// renders into the `Name[size,complexity,flags]` badge:
///
/// - **`ts`** — the type has a timeseries configuration
///   ([`has_timeseries`](Self::has_timeseries)).
/// - **`loc`** — cheap coordinates: a spatial config naming a location/point
///   field, or a metadata field typed `point`
///   ([`has_location`](Self::has_location)).
/// - **`geo`** — WKT geometry: a spatial config naming a geometry/shape field
///   ([`has_geometry`](Self::has_geometry)).
/// - **`vec`** — at least one embedding store is registered for the type
///   ([`has_embeddings`](Self::has_embeddings)).
///
/// `loc` and `geo` are **independent**: a type declaring lat/lon columns *and*
/// a WKT field carries both, and a consumer that treats `geo` as implying no
/// cheap coordinates will parse polygons to recover floats sitting next door.
///
/// The type is opaque: read it through the accessors or
/// [`flags_csv`](Self::flags_csv), never by field, so a fifth capability is an
/// additive change rather than a breaking one. Build the map with
/// [`compute_type_capabilities`] (all types) or
/// [`compute_type_capabilities_for`] (a named subset).
pub struct TypeCapabilities {
    pub(super) has_timeseries: bool,
    pub(super) has_location: bool,
    pub(super) has_geometry: bool,
    pub(super) has_embeddings: bool,
}

impl TypeCapabilities {
    /// The type has a timeseries configuration (badge flag `ts`).
    pub fn has_timeseries(&self) -> bool {
        self.has_timeseries
    }

    /// The type carries plain coordinates — a location/point field in its
    /// spatial config, or a metadata field typed `point` (badge flag `loc`).
    pub fn has_location(&self) -> bool {
        self.has_location
    }

    /// The type carries WKT geometry — a geometry/shape field in its spatial
    /// config (badge flag `geo`). Independent of
    /// [`has_location`](Self::has_location).
    pub fn has_geometry(&self) -> bool {
        self.has_geometry
    }

    /// At least one embedding store is registered for the type (badge flag
    /// `vec`).
    pub fn has_embeddings(&self) -> bool {
        self.has_embeddings
    }

    /// The set flags as the comma-separated badge text `describe()` renders,
    /// in the fixed order `ts,geo,loc,vec`; empty when the type supports none.
    pub fn flags_csv(&self) -> String {
        let mut flags = Vec::new();
        if self.has_timeseries {
            flags.push("ts");
        }
        if self.has_geometry {
            flags.push("geo");
        }
        // `loc` and `geo` are independent facts — a type declaring lat/lon
        // columns *and* a WKT field carries both, and suppressing `loc` here
        // hid the cheap-coordinate half from every reader of the badge.
        if self.has_location {
            flags.push("loc");
        }
        if self.has_embeddings {
            flags.push("vec");
        }
        flags.join(",")
    }

    fn merge(&mut self, other: &TypeCapabilities) {
        self.has_timeseries |= other.has_timeseries;
        self.has_location |= other.has_location;
        self.has_geometry |= other.has_geometry;
        self.has_embeddings |= other.has_embeddings;
    }
}

pub(super) fn property_complexity(count: usize) -> &'static str {
    match count {
        0..=3 => "vl",
        4..=8 => "l",
        9..=15 => "m",
        16..=30 => "h",
        _ => "vh",
    }
}

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

/// [`TypeCapabilities`] for every node type in the graph, keyed by type name.
/// Reads the registered configs (timeseries, spatial, embeddings) plus the
/// declared property types — it never scans node data, so the cost is in the
/// number of *types*, not nodes.
pub fn compute_type_capabilities(graph: &DirGraph) -> HashMap<String, TypeCapabilities> {
    let mut caps: HashMap<String, TypeCapabilities> = HashMap::new();

    for node_type in graph.type_indices.keys() {
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

        // Also check metadata for point-type fields (no SpatialConfig set)
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

/// [`compute_type_capabilities`] restricted to `type_names` — same answer for
/// those types, without walking every type in the graph. Names that are not
/// node types of `graph` are skipped, so the result may be smaller than the
/// request.
pub fn compute_type_capabilities_for(
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

/// Neighbor schema from the first `max_nodes` nodes of a type, with counts
/// extrapolated to the full population.
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
            if let Some(target_node) = graph.node_view(edge_ref.target()) {
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
            if let Some(source_node) = graph.node_view(edge_ref.source()) {
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

/// Endpoint-type discovery for connection types with empty metadata: one pass,
/// bounded at `max_total_scan` edges across all connection types.
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

#[cfg(test)]
mod tests {
    use super::TypeCapabilities;

    fn caps(ts: bool, loc: bool, geo: bool, vec: bool) -> TypeCapabilities {
        TypeCapabilities {
            has_timeseries: ts,
            has_location: loc,
            has_geometry: geo,
            has_embeddings: vec,
        }
    }

    /// `loc` and `geo` are independent facts about a type, and a type that
    /// declares both must advertise both. `geo` used to suppress `loc`, so a
    /// type carrying lat/lon columns *and* a WKT field looked geometry-only —
    /// a downstream reading the badge went to parse polygons to recover
    /// coordinates that were sitting in plain float columns next door
    /// (measured on 37 of 38 sodir types, all of which declare both).
    #[test]
    fn location_and_geometry_flags_are_independent() {
        assert_eq!(caps(false, true, true, false).flags_csv(), "geo,loc");
        assert_eq!(caps(false, true, false, false).flags_csv(), "loc");
        assert_eq!(caps(false, false, true, false).flags_csv(), "geo");
        assert_eq!(caps(false, false, false, false).flags_csv(), "");
        assert_eq!(caps(true, true, true, true).flags_csv(), "ts,geo,loc,vec");
    }

    /// The accessors are the only way an out-of-crate consumer reads a
    /// capability, so each one must answer for the same flag `flags_csv()`
    /// emits — a swapped pair would send a downstream to the wrong column.
    /// Swept over all 16 flag combinations.
    #[test]
    fn accessors_agree_with_flags_csv() {
        for bits in 0u8..16 {
            let c = caps(bits & 1 != 0, bits & 2 != 0, bits & 4 != 0, bits & 8 != 0);
            let csv = c.flags_csv();
            let flags: Vec<&str> = if csv.is_empty() {
                Vec::new()
            } else {
                csv.split(',').collect()
            };
            assert_eq!(
                c.has_timeseries(),
                flags.contains(&"ts"),
                "bits={bits}: has_timeseries vs `ts`"
            );
            assert_eq!(
                c.has_geometry(),
                flags.contains(&"geo"),
                "bits={bits}: has_geometry vs `geo`"
            );
            assert_eq!(
                c.has_location(),
                flags.contains(&"loc"),
                "bits={bits}: has_location vs `loc`"
            );
            assert_eq!(
                c.has_embeddings(),
                flags.contains(&"vec"),
                "bits={bits}: has_embeddings vs `vec`"
            );
        }
    }
}
