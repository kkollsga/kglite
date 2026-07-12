// src/graph/traversal.rs
use crate::datatypes::values::FilterCondition;
use crate::datatypes::values::Value;
use crate::graph::schema::{
    CurrentSelection, DirGraph, InternedKey, NodeData, SelectionOperation, SpatialConfig,
    TemporalConfig,
};
use crate::graph::storage::GraphRead;
use chrono::NaiveDate;
use geo::geometry::Geometry;
use petgraph::graph::NodeIndex;
use petgraph::Direction;
use std::collections::{HashMap, HashSet};

/// Temporal filter for edge traversal.
/// Carries multiple TemporalConfig entries to support shared connection type names
/// across source types (e.g., HAS_LICENSEE used by Field, Licence, BusinessArrangement).
pub enum TemporalEdgeFilter {
    /// Point-in-time: valid_from <= date AND (valid_to IS NULL OR valid_to >= date)
    At(Vec<TemporalConfig>, NaiveDate),
    /// Range overlap: valid_from <= end AND (valid_to IS NULL OR valid_to >= start)
    During(Vec<TemporalConfig>, NaiveDate, NaiveDate),
}

// ── Comparison-based traversal types ─────────────────────────────────────────

/// How polygon nodes should be spatially resolved.
/// When set, overrides the default "location → centroid fallback" behavior.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SpatialResolve {
    /// Use geometry centroid (ignoring location fields)
    Centroid,
    /// Use closest point on geometry boundary (for distance calculations)
    Closest,
    /// Use full polygon geometry (for containment checks)
    Geometry,
}

/// Parsed configuration from the Python `method=` parameter (str or dict).
pub struct MethodConfig {
    pub method_type: String,
    pub resolve: Option<SpatialResolve>,
    pub max_distance_m: Option<f64>,
    pub geometry_field: Option<String>,
    pub property: Option<String>,
    pub threshold: Option<f64>,
    pub metric: Option<String>,
    pub algorithm: Option<String>,
    pub features: Option<Vec<String>>,
    pub k: Option<usize>,
    pub eps: Option<f64>,
    pub min_samples: Option<usize>,
}

impl MethodConfig {
    /// Build from a string shorthand (no extra settings).
    pub fn from_string(method_type: String) -> Self {
        Self {
            method_type,
            resolve: None,
            max_distance_m: None,
            geometry_field: None,
            property: None,
            threshold: None,
            metric: None,
            algorithm: None,
            features: None,
            k: None,
            eps: None,
            min_samples: None,
        }
    }

    /// Parse `resolve` string to enum.
    pub fn parse_resolve(s: &str) -> Result<SpatialResolve, String> {
        match s {
            "centroid" => Ok(SpatialResolve::Centroid),
            "closest" => Ok(SpatialResolve::Closest),
            "geometry" => Ok(SpatialResolve::Geometry),
            _ => Err(format!(
                "Unknown resolve mode: '{}'. Valid: 'centroid', 'closest', 'geometry'",
                s
            )),
        }
    }

    /// Build from already-extracted dict-style fields. The PyO3 binding
    /// in kglite-py unpacks a `Bound<PyAny>` into these arguments and
    /// calls this constructor; pure-Rust callers (CLI tools, JSON/YAML
    /// config loaders, future bindings) can call it directly with the
    /// same data shape. Lifted from kglite-py in 0.10.1.
    ///
    /// `method_type` is required — the dict's `"type"` key.
    /// `resolve_str` is the dict's `"resolve"` key, validated via
    /// [`Self::parse_resolve`]. All other fields are optional and
    /// passed through unchanged.
    #[allow(clippy::too_many_arguments)]
    pub fn from_components(
        method_type: String,
        resolve_str: Option<String>,
        max_distance_m: Option<f64>,
        geometry_field: Option<String>,
        property: Option<String>,
        threshold: Option<f64>,
        metric: Option<String>,
        algorithm: Option<String>,
        features: Option<Vec<String>>,
        k: Option<usize>,
        eps: Option<f64>,
        min_samples: Option<usize>,
    ) -> Result<Self, String> {
        let resolve = match resolve_str {
            Some(s) => Some(Self::parse_resolve(&s)?),
            None => None,
        };
        Ok(Self {
            method_type,
            resolve,
            max_distance_m,
            geometry_field,
            property,
            threshold,
            metric,
            algorithm,
            features,
            k,
            eps,
            min_samples,
        })
    }
}

/// Check if edge properties match all given filter conditions
fn edge_matches_conditions(
    properties: &[(InternedKey, Value)],
    conditions: &HashMap<String, FilterCondition>,
) -> bool {
    conditions.iter().all(|(field, condition)| {
        let ik = InternedKey::from_str(field);
        match properties.iter().find(|(k, _)| *k == ik).map(|(_, v)| v) {
            Some(value) => crate::graph::core::filtering::matches_condition(value, condition),
            None => {
                // Missing field is treated as null
                matches!(condition, FilterCondition::IsNull)
            }
        }
    })
}

/// Check if edge properties pass a temporal filter.
/// Tries multiple configs to find one matching the edge's field names.
fn edge_passes_temporal(properties: &[(InternedKey, Value)], filter: &TemporalEdgeFilter) -> bool {
    match filter {
        TemporalEdgeFilter::At(configs, date) => {
            crate::graph::features::temporal::is_temporally_valid_multi(properties, configs, date)
        }
        TemporalEdgeFilter::During(configs, start, end) => {
            crate::graph::features::temporal::overlaps_range_multi(properties, configs, start, end)
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn make_traversal(
    graph: &DirGraph,
    selection: &mut CurrentSelection,
    connection_type: String,
    level_index: Option<usize>,
    direction: Option<String>,
    filter_target: Option<&HashMap<String, FilterCondition>>,
    filter_connection: Option<&HashMap<String, FilterCondition>>,
    sort_target: Option<&Vec<(String, bool)>>,
    max_nodes: Option<usize>,
    new_level: Option<bool>,
    temporal_filter: Option<&TemporalEdgeFilter>,
    target_type: Option<&[String]>,
) -> Result<(), String> {
    // Arena guard: disk-backed node/edge reads materialize into the query
    // arena (protocol in disk/graph.rs); no-op on memory/mapped.
    let _arena_guard = graph.graph.begin_query();
    // Validate connection type exists
    if !graph.has_connection_type(&connection_type) {
        return Err(format!(
            "Connection type '{}' does not exist in graph",
            connection_type
        ));
    }

    // First get the source level index
    let source_level_index =
        level_index.unwrap_or_else(|| selection.get_level_count().saturating_sub(1));

    let create_new_level = new_level.unwrap_or(true);

    // Get source level
    let source_level = selection
        .get_level(source_level_index)
        .ok_or_else(|| "No valid source level found for traversal".to_string())?;

    // Early empty check
    if source_level.is_empty() {
        return Err("No source nodes available for traversal".to_string());
    }

    // Set up traversal directions
    let dir = match direction.as_deref() {
        Some("incoming") => Some(Direction::Incoming),
        Some("outgoing") => Some(Direction::Outgoing),
        Some(d) => {
            return Err(format!(
                "Invalid direction: {}. Must be 'incoming' or 'outgoing'",
                d
            ))
        }
        None => None, // Both directions
    };

    // FAST PATH: No filtering, sorting, or limits - optimized for common case
    // target_type is kept in fast path since it's a cheap string comparison
    let use_fast_path = filter_target.is_none()
        && filter_connection.is_none()
        && sort_target.is_none()
        && max_nodes.is_none()
        && create_new_level
        && temporal_filter.is_none();

    if use_fast_path {
        return make_traversal_fast(
            graph,
            selection,
            &connection_type,
            source_level_index,
            dir,
            target_type,
        );
    }

    // SLOW PATH: Full processing with filtering/sorting/limits
    make_traversal_full(
        graph,
        selection,
        connection_type,
        source_level_index,
        dir,
        filter_target,
        filter_connection,
        sort_target,
        max_nodes,
        create_new_level,
        temporal_filter,
        target_type,
    )
}

/// Fast traversal path for the common case: no filtering, no sorting, no limits.
/// Avoids HashMap overhead by collecting all targets directly.
fn make_traversal_fast(
    graph: &DirGraph,
    selection: &mut CurrentSelection,
    connection_type: &str,
    source_level_index: usize,
    direction: Option<Direction>,
    target_type: Option<&[String]>,
) -> Result<(), String> {
    // Get source nodes using iterator to avoid allocation
    let source_level = selection
        .get_level(source_level_index)
        .ok_or_else(|| "No valid source level found for traversal".to_string())?;

    // Collect source nodes (we need this twice - once for iteration, once for the parent map)
    let source_nodes: Vec<NodeIndex> = source_level.iter_node_indices().collect();

    // Create new level
    selection.add_level();
    let target_level_index = selection.get_level_count() - 1;

    // Pre-intern connection type for fast u64 == u64 comparison in inner loop
    let conn_key = InternedKey::from_str(connection_type);

    // Pre-allocate targets HashSet with estimated capacity
    let mut all_targets_per_parent: HashMap<NodeIndex, Vec<NodeIndex>> =
        HashMap::with_capacity(source_nodes.len());

    let g = &graph.graph;
    // Process each source node
    for &source_node in &source_nodes {
        let mut targets: HashSet<NodeIndex> = HashSet::new();

        // Helper: check if a target node passes the type filter
        let type_ok = |idx: petgraph::graph::NodeIndex| -> bool {
            match target_type {
                None => true,
                Some(types) => {
                    let nt = g[idx].node_type;
                    types.iter().any(|t| InternedKey::from_str(t) == nt)
                }
            }
        };

        // Process edges based on direction. Use edges_directed_filtered so
        // disk backends can binary-search the CSR range by connection type
        // (O(log D) on csr_sorted_by_type graphs vs O(D) unfiltered scan).
        // Heap backends ignore the hint; post-filter still covers them.
        match direction {
            Some(Direction::Outgoing) => {
                for edge in
                    g.edges_directed_filtered(source_node, Direction::Outgoing, Some(conn_key))
                {
                    if edge.weight().connection_type == conn_key {
                        let t = edge.target();
                        if type_ok(t) {
                            targets.insert(t);
                        }
                    }
                }
            }
            Some(Direction::Incoming) => {
                for edge in
                    g.edges_directed_filtered(source_node, Direction::Incoming, Some(conn_key))
                {
                    if edge.weight().connection_type == conn_key {
                        let t = edge.source();
                        if type_ok(t) {
                            targets.insert(t);
                        }
                    }
                }
            }
            None => {
                // Both directions
                for edge in
                    g.edges_directed_filtered(source_node, Direction::Outgoing, Some(conn_key))
                {
                    if edge.weight().connection_type == conn_key {
                        let t = edge.target();
                        if type_ok(t) {
                            targets.insert(t);
                        }
                    }
                }
                for edge in
                    g.edges_directed_filtered(source_node, Direction::Incoming, Some(conn_key))
                {
                    if edge.weight().connection_type == conn_key {
                        let t = edge.source();
                        if type_ok(t) {
                            targets.insert(t);
                        }
                    }
                }
            }
        }

        // Store targets for this parent
        if !targets.is_empty() {
            all_targets_per_parent.insert(source_node, targets.into_iter().collect());
        }
    }

    // Get target level and populate it
    let level = selection
        .get_level_mut(target_level_index)
        .ok_or_else(|| "Failed to access target selection level".to_string())?;

    // Set up operation
    level.operations = vec![SelectionOperation::Traverse {
        connection_type: connection_type.to_string(),
        direction: direction.map(|d| {
            if d == Direction::Incoming {
                "incoming"
            } else {
                "outgoing"
            }
            .to_string()
        }),
        max_nodes: None,
    }];

    // Add all parent->children mappings
    for (parent, children) in all_targets_per_parent {
        level.add_selection(Some(parent), children);
    }

    Ok(())
}

/// Full traversal path with filtering, sorting, and limits support.
#[allow(clippy::too_many_arguments)]
fn make_traversal_full(
    graph: &DirGraph,
    selection: &mut CurrentSelection,
    connection_type: String,
    source_level_index: usize,
    direction: Option<Direction>,
    filter_target: Option<&HashMap<String, FilterCondition>>,
    filter_connection: Option<&HashMap<String, FilterCondition>>,
    sort_target: Option<&Vec<(String, bool)>>,
    max_nodes: Option<usize>,
    create_new_level: bool,
    temporal_filter: Option<&TemporalEdgeFilter>,
    target_type: Option<&[String]>,
) -> Result<(), String> {
    // Get source level
    let source_level = selection
        .get_level(source_level_index)
        .ok_or_else(|| "No valid source level found for traversal".to_string())?;

    // Collect all necessary data from source level
    let parents: Vec<NodeIndex> = if create_new_level {
        source_level.iter_node_indices().collect()
    } else {
        source_level.selections.keys().filter_map(|k| *k).collect()
    };

    // Create a mapping of parent nodes to their source nodes
    let source_nodes_map: HashMap<NodeIndex, Vec<NodeIndex>> = if create_new_level {
        parents
            .iter()
            .map(|&parent| (parent, vec![parent]))
            .collect()
    } else {
        source_level
            .selections
            .iter()
            .filter_map(|(parent, children)| parent.map(|p| (p, children.clone())))
            .collect()
    };

    // Now we can safely modify the selection
    if create_new_level {
        selection.add_level();
    }

    let target_level_index = if create_new_level {
        selection.get_level_count() - 1
    } else {
        source_level_index
    };

    // Get and initialize target level
    let level = selection
        .get_level_mut(target_level_index)
        .ok_or_else(|| "Failed to access target selection level".to_string())?;

    // Set up operation
    let operation = SelectionOperation::Traverse {
        connection_type: connection_type.clone(),
        direction: direction.map(|d| {
            if d == Direction::Incoming {
                "incoming"
            } else {
                "outgoing"
            }
            .to_string()
        }),
        max_nodes,
    };
    level.operations = vec![operation];

    // Define an empty vector to use when no source nodes exist
    let empty_vec: Vec<NodeIndex> = Vec::new();

    // Process each parent node once
    for &parent in &parents {
        // Use a reference to an existing empty vector to avoid temporary lifetime issues
        let source_nodes = source_nodes_map.get(&parent).unwrap_or(&empty_vec);

        if !create_new_level {
            // Clear existing selection for this parent
            level.selections.entry(Some(parent)).or_default().clear();
        }

        // Collect all targets for this parent in one pass
        let mut targets = HashSet::new();

        // Pre-intern connection type for fast u64 == u64 comparison
        let conn_key = InternedKey::from_str(&connection_type);

        let g = &graph.graph;
        // Helper: check if a target node passes the type filter
        let type_ok = |idx: NodeIndex| -> bool {
            match target_type {
                None => true,
                Some(types) => {
                    let nt = g[idx].node_type;
                    types.iter().any(|t| InternedKey::from_str(t) == nt)
                }
            }
        };

        // Process edges based on direction. See make_traversal_fast for the
        // rationale behind edges_directed_filtered.
        for &source_node in source_nodes {
            match direction {
                Some(Direction::Outgoing) => {
                    for edge in
                        g.edges_directed_filtered(source_node, Direction::Outgoing, Some(conn_key))
                    {
                        if edge.weight().connection_type == conn_key {
                            if let Some(conn_filter) = filter_connection {
                                if !edge_matches_conditions(&edge.weight().properties, conn_filter)
                                {
                                    continue;
                                }
                            }
                            if let Some(tf) = &temporal_filter {
                                if !edge_passes_temporal(&edge.weight().properties, tf) {
                                    continue;
                                }
                            }
                            let t = edge.target();
                            if type_ok(t) {
                                targets.insert(t);
                            }
                        }
                    }
                }
                Some(Direction::Incoming) => {
                    for edge in
                        g.edges_directed_filtered(source_node, Direction::Incoming, Some(conn_key))
                    {
                        if edge.weight().connection_type == conn_key {
                            if let Some(conn_filter) = filter_connection {
                                if !edge_matches_conditions(&edge.weight().properties, conn_filter)
                                {
                                    continue;
                                }
                            }
                            if let Some(tf) = &temporal_filter {
                                if !edge_passes_temporal(&edge.weight().properties, tf) {
                                    continue;
                                }
                            }
                            let t = edge.source();
                            if type_ok(t) {
                                targets.insert(t);
                            }
                        }
                    }
                }
                None => {
                    // Both directions
                    for edge in
                        g.edges_directed_filtered(source_node, Direction::Outgoing, Some(conn_key))
                    {
                        if edge.weight().connection_type == conn_key {
                            if let Some(conn_filter) = filter_connection {
                                if !edge_matches_conditions(&edge.weight().properties, conn_filter)
                                {
                                    continue;
                                }
                            }
                            if let Some(tf) = &temporal_filter {
                                if !edge_passes_temporal(&edge.weight().properties, tf) {
                                    continue;
                                }
                            }
                            let t = edge.target();
                            if type_ok(t) {
                                targets.insert(t);
                            }
                        }
                    }
                    for edge in
                        g.edges_directed_filtered(source_node, Direction::Incoming, Some(conn_key))
                    {
                        if edge.weight().connection_type == conn_key {
                            if let Some(conn_filter) = filter_connection {
                                if !edge_matches_conditions(&edge.weight().properties, conn_filter)
                                {
                                    continue;
                                }
                            }
                            if let Some(tf) = &temporal_filter {
                                if !edge_passes_temporal(&edge.weight().properties, tf) {
                                    continue;
                                }
                            }
                            let t = edge.source();
                            if type_ok(t) {
                                targets.insert(t);
                            }
                        }
                    }
                }
            }
        }

        // Convert to Vec for processing
        let target_vec: Vec<NodeIndex> = targets.into_iter().collect();

        // Apply filtering and sorting in one pass
        let processed_nodes = crate::graph::core::filtering::process_nodes(
            graph,
            target_vec,
            filter_target,
            sort_target,
            max_nodes,
        );

        // Add the processed nodes to the selection
        level.add_selection(Some(parent), processed_nodes);
    }

    Ok(())
}

// ── Comparison-based traversal ───────────────────────────────────────────────

/// Dispatcher for comparison-based traversal methods.
/// When `method` is specified, traverse() switches from edge-based to comparison-based mode:
/// the first arg becomes the target node type, and matches are discovered via spatial,
/// semantic, or clustering comparisons rather than pre-existing edges.
pub fn make_comparison_traversal(
    graph: &DirGraph,
    selection: &mut CurrentSelection,
    target_type: Option<&str>,
    config: &MethodConfig,
    filter_target: Option<&HashMap<String, FilterCondition>>,
    sort_target: Option<&Vec<(String, bool)>>,
    max_nodes: Option<usize>,
) -> Result<(), String> {
    // Arena guard: disk-backed node/edge reads materialize into the query
    // arena (protocol in disk/graph.rs); no-op on memory/mapped.
    let _arena_guard = graph.graph.begin_query();
    match config.method_type.as_str() {
        "contains" => {
            let tt = target_type
                .ok_or("method 'contains' requires a target_type (first arg to traverse)")?;
            spatial_contains_traversal(
                graph,
                selection,
                tt,
                config.resolve,
                config.geometry_field.as_deref(),
                filter_target,
                sort_target,
                max_nodes,
            )
        }
        "intersects" => {
            let tt = target_type
                .ok_or("method 'intersects' requires a target_type (first arg to traverse)")?;
            spatial_intersects_traversal(
                graph,
                selection,
                tt,
                config.geometry_field.as_deref(),
                filter_target,
                sort_target,
                max_nodes,
            )
        }
        "distance" => {
            let tt = target_type
                .ok_or("method 'distance' requires a target_type (first arg to traverse)")?;
            let max_dist = config.max_distance_m.ok_or(
                "method 'distance' requires 'max_m' (dict) or max_distance_m parameter",
            )?;
            spatial_distance_traversal(
                graph,
                selection,
                tt,
                max_dist,
                config.resolve,
                config.geometry_field.as_deref(),
                filter_target,
                sort_target,
                max_nodes,
            )
        }
        "text_score" => {
            let tt = target_type
                .ok_or("method 'text_score' requires a target_type (first arg to traverse)")?;
            let prop = config
                .property
                .as_deref()
                .ok_or("method 'text_score' requires 'property'")?;
            let thresh = config.threshold.unwrap_or(0.0);
            let dist_metric = match config.metric.as_deref() {
                Some("dot_product") => crate::graph::algorithms::vector::DistanceMetric::DotProduct,
                Some("euclidean") => crate::graph::algorithms::vector::DistanceMetric::Euclidean,
                Some("poincare") => crate::graph::algorithms::vector::DistanceMetric::Poincare,
                _ => crate::graph::algorithms::vector::DistanceMetric::Cosine,
            };
            semantic_score_traversal(
                graph,
                selection,
                tt,
                prop,
                thresh,
                dist_metric,
                filter_target,
                sort_target,
                max_nodes,
            )
        }
        "cluster" => {
            let algo = config
                .algorithm
                .as_deref()
                .ok_or("method 'cluster' requires 'algorithm' (e.g. 'kmeans')")?;
            let feats = config
                .features
                .as_deref()
                .ok_or("method 'cluster' requires 'features'")?;
            cluster_traversal(
                graph, selection, target_type, algo, feats, config.k, config.eps,
                config.min_samples,
            )
        }
        _ => Err(format!(
            "Unknown traversal method: '{}'. Valid: 'contains', 'intersects', 'distance', 'text_score', 'cluster'",
            config.method_type
        )),
    }
}

// ── Spatial helpers ─────────────────────────────────────────────────────────

/// Resolve the geometry field name for a node type, checking override then SpatialConfig.
fn resolve_geometry_field<'a>(
    spatial_config: Option<&'a SpatialConfig>,
    geometry_field_override: Option<&'a str>,
) -> Option<&'a str> {
    geometry_field_override.or_else(|| spatial_config.and_then(|sc| sc.geometry.as_deref()))
}

/// Extract a parsed WKT geometry from a node's properties.
fn node_geometry(node: &NodeData, geom_field: &str) -> Option<Geometry<f64>> {
    match node.get_property(geom_field).as_deref() {
        Some(Value::String(wkt)) => crate::graph::features::spatial::parse_wkt(wkt).ok(),
        _ => None,
    }
}

/// Extract (lat, lon) from a node using SpatialConfig (location fields + geometry centroid fallback).
fn node_lat_lon(node: &NodeData, spatial_config: Option<&SpatialConfig>) -> Option<(f64, f64)> {
    let sc = spatial_config?;
    if let Some((ref lat_f, ref lon_f)) = sc.location {
        if let Some((lat, lon)) = extract_lat_lon(node, lat_f, lon_f) {
            return Some((lat, lon));
        }
    }
    // Fallback to geometry centroid
    if let Some(ref geom_f) = sc.geometry {
        if let Some(geom) = node_geometry(node, geom_f) {
            return crate::graph::features::spatial::geometry_centroid(&geom).ok();
        }
    }
    None
}

/// Resolve a node to a (lat, lon) point respecting the `resolve` mode.
/// - None: default (location → geometry centroid fallback)
/// - Centroid: force geometry centroid (skip location fields)
/// - Closest/Geometry: also resolve to geometry centroid as a point
///   (actual geometry usage is handled by the caller)
fn resolve_node_point(
    node: &NodeData,
    spatial_config: Option<&SpatialConfig>,
    resolve: Option<SpatialResolve>,
    geometry_field_override: Option<&str>,
) -> Option<(f64, f64)> {
    match resolve {
        Some(SpatialResolve::Centroid)
        | Some(SpatialResolve::Closest)
        | Some(SpatialResolve::Geometry) => {
            // Force geometry centroid — skip location fields
            let geom_field = geometry_field_override
                .or_else(|| spatial_config.and_then(|sc| sc.geometry.as_deref()))?;
            let geom = node_geometry(node, geom_field)?;
            crate::graph::features::spatial::geometry_centroid(&geom).ok()
        }
        None => {
            // Default: location → geometry centroid fallback
            node_lat_lon(node, spatial_config)
        }
    }
}

/// Get the parsed geometry for a node (for resolve='geometry' or 'closest' mode).
fn resolve_node_geom(
    node: &NodeData,
    spatial_config: Option<&SpatialConfig>,
    geometry_field_override: Option<&str>,
) -> Option<Geometry<f64>> {
    let geom_field =
        geometry_field_override.or_else(|| spatial_config.and_then(|sc| sc.geometry.as_deref()))?;
    node_geometry(node, geom_field)
}

fn extract_lat_lon(node: &NodeData, lat_field: &str, lon_field: &str) -> Option<(f64, f64)> {
    let lat = node
        .get_property(lat_field)
        .as_deref()
        .and_then(value_to_f64)?;
    let lon = node
        .get_property(lon_field)
        .as_deref()
        .and_then(value_to_f64)?;
    Some((lat, lon))
}

fn value_to_f64(v: &Value) -> Option<f64> {
    match v {
        Value::Float64(f) => Some(*f),
        Value::Int64(i) => Some(*i as f64),
        Value::String(s) => s.parse().ok(),
        _ => None,
    }
}

/// Collect source nodes and determine source type from the current selection.
fn get_source_info(
    graph: &DirGraph,
    selection: &CurrentSelection,
) -> Result<(Vec<NodeIndex>, String), String> {
    let level_idx = selection.get_level_count().saturating_sub(1);
    let level = selection
        .get_level(level_idx)
        .ok_or("No source level for comparison traversal")?;
    let source_nodes: Vec<NodeIndex> = level.iter_node_indices().collect();
    if source_nodes.is_empty() {
        return Err("No source nodes for comparison traversal".into());
    }
    let source_type = graph
        .get_node(source_nodes[0])
        .map(|n| n.node_type_str(&graph.interner).to_string())
        .ok_or("Cannot determine source node type")?;
    Ok((source_nodes, source_type))
}

/// Get all candidate target nodes from type_indices.
fn get_target_candidates(graph: &DirGraph, target_type: &str) -> Result<Vec<NodeIndex>, String> {
    graph
        .type_indices
        .get(target_type)
        .map(|v| v.to_vec())
        .ok_or_else(|| {
            let available: Vec<&str> = graph.type_indices.keys().collect();
            format!(
                "Target type '{}' not found in graph. Available: {:?}",
                target_type, available
            )
        })
}

/// Insert matched pairs into a new selection level, applying optional filter/sort/limit.
fn insert_matches_into_selection(
    graph: &DirGraph,
    selection: &mut CurrentSelection,
    matches: HashMap<NodeIndex, Vec<NodeIndex>>,
    method: &str,
    filter_target: Option<&HashMap<String, FilterCondition>>,
    sort_target: Option<&Vec<(String, bool)>>,
    max_nodes: Option<usize>,
) {
    selection.add_level();
    let target_level_idx = selection.get_level_count() - 1;
    let level = selection.get_level_mut(target_level_idx).unwrap();

    level.operations = vec![SelectionOperation::Custom(format!(
        "compare(method='{}')",
        method
    ))];

    for (parent, children) in matches {
        let processed = crate::graph::core::filtering::process_nodes(
            graph,
            children,
            filter_target,
            sort_target,
            max_nodes,
        );
        if !processed.is_empty() {
            level.add_selection(Some(parent), processed);
        }
    }
}

// ── Spatial: contains ───────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn spatial_contains_traversal(
    graph: &DirGraph,
    selection: &mut CurrentSelection,
    target_type: &str,
    resolve: Option<SpatialResolve>,
    geometry_field: Option<&str>,
    filter_target: Option<&HashMap<String, FilterCondition>>,
    sort_target: Option<&Vec<(String, bool)>>,
    max_nodes: Option<usize>,
) -> Result<(), String> {
    let (source_nodes, source_type) = get_source_info(graph, selection)?;
    let target_candidates = get_target_candidates(graph, target_type)?;

    let source_spatial = graph.get_spatial_config(&source_type);
    let target_spatial = graph.get_spatial_config(target_type);

    // Source needs a geometry field (polygon) for containment
    let src_geom_field =
        resolve_geometry_field(source_spatial, geometry_field).ok_or_else(|| {
            format!(
                "method 'contains' requires source type '{}' to have a geometry. \
             Set via set_spatial() or pass geometry='field' in method dict",
                source_type
            )
        })?;

    let use_full_geometry = resolve == Some(SpatialResolve::Geometry);

    // Build matches: for each source geometry, find targets contained within it
    let mut matches: HashMap<NodeIndex, Vec<NodeIndex>> =
        HashMap::with_capacity(source_nodes.len());

    for &src_idx in &source_nodes {
        let src_node = match graph.get_node(src_idx) {
            Some(n) => n,
            None => continue,
        };
        let src_geom = match node_geometry(src_node, src_geom_field) {
            Some(g) => g,
            None => continue,
        };

        // Compute bounding box for pre-filter
        let src_bbox = geo::BoundingRect::bounding_rect(&src_geom);

        let mut matched = Vec::new();
        for &tgt_idx in &target_candidates {
            let tgt_node = match graph.get_node(tgt_idx) {
                Some(n) => n,
                None => continue,
            };

            if use_full_geometry {
                // resolve='geometry': polygon-in-polygon containment
                if let Some(tgt_geom) = resolve_node_geom(tgt_node, target_spatial, geometry_field)
                {
                    if crate::graph::features::spatial::geometry_contains_geometry(
                        &src_geom, &tgt_geom,
                    ) {
                        matched.push(tgt_idx);
                    }
                }
            } else {
                // Default / resolve='centroid': target as point → point-in-polygon
                if let Some((lat, lon)) =
                    resolve_node_point(tgt_node, target_spatial, resolve, geometry_field)
                {
                    // Bounding box pre-filter
                    if let Some(ref bbox) = src_bbox {
                        if lat < bbox.min().y
                            || lat > bbox.max().y
                            || lon < bbox.min().x
                            || lon > bbox.max().x
                        {
                            continue;
                        }
                    }
                    let pt = geo::geometry::Point::new(lon, lat);
                    if crate::graph::features::spatial::geometry_contains_point(&src_geom, &pt) {
                        matched.push(tgt_idx);
                    }
                }
            }
        }

        if !matched.is_empty() {
            matches.insert(src_idx, matched);
        }
    }

    insert_matches_into_selection(
        graph,
        selection,
        matches,
        "contains",
        filter_target,
        sort_target,
        max_nodes,
    );
    Ok(())
}

// ── Spatial: intersects ─────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn spatial_intersects_traversal(
    graph: &DirGraph,
    selection: &mut CurrentSelection,
    target_type: &str,
    geometry_field: Option<&str>,
    filter_target: Option<&HashMap<String, FilterCondition>>,
    sort_target: Option<&Vec<(String, bool)>>,
    max_nodes: Option<usize>,
) -> Result<(), String> {
    let (source_nodes, source_type) = get_source_info(graph, selection)?;
    let target_candidates = get_target_candidates(graph, target_type)?;

    let source_spatial = graph.get_spatial_config(&source_type);
    let target_spatial = graph.get_spatial_config(target_type);

    let src_geom_field =
        resolve_geometry_field(source_spatial, geometry_field).ok_or_else(|| {
            format!(
                "method 'intersects' requires source type '{}' to have a geometry. \
             Set via set_spatial() or pass geometry='field' in method dict",
                source_type
            )
        })?;
    let tgt_geom_field =
        resolve_geometry_field(target_spatial, geometry_field).ok_or_else(|| {
            format!(
                "method 'intersects' requires target type '{}' to have a geometry. \
             Set via set_spatial() or pass geometry='field' in method dict",
                target_type
            )
        })?;

    let mut matches: HashMap<NodeIndex, Vec<NodeIndex>> =
        HashMap::with_capacity(source_nodes.len());

    for &src_idx in &source_nodes {
        let src_node = match graph.get_node(src_idx) {
            Some(n) => n,
            None => continue,
        };
        let src_geom = match node_geometry(src_node, src_geom_field) {
            Some(g) => g,
            None => continue,
        };

        let mut matched = Vec::new();
        for &tgt_idx in &target_candidates {
            let tgt_node = match graph.get_node(tgt_idx) {
                Some(n) => n,
                None => continue,
            };
            if let Some(tgt_geom) = node_geometry(tgt_node, tgt_geom_field) {
                if crate::graph::features::spatial::geometries_intersect(&src_geom, &tgt_geom) {
                    matched.push(tgt_idx);
                }
            }
        }

        if !matched.is_empty() {
            matches.insert(src_idx, matched);
        }
    }

    insert_matches_into_selection(
        graph,
        selection,
        matches,
        "intersects",
        filter_target,
        sort_target,
        max_nodes,
    );
    Ok(())
}

// ── Spatial: distance ───────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn spatial_distance_traversal(
    graph: &DirGraph,
    selection: &mut CurrentSelection,
    target_type: &str,
    max_distance_m: f64,
    resolve: Option<SpatialResolve>,
    geometry_field: Option<&str>,
    filter_target: Option<&HashMap<String, FilterCondition>>,
    sort_target: Option<&Vec<(String, bool)>>,
    max_nodes: Option<usize>,
) -> Result<(), String> {
    let (source_nodes, source_type) = get_source_info(graph, selection)?;
    let target_candidates = get_target_candidates(graph, target_type)?;

    let source_spatial = graph.get_spatial_config(&source_type);
    let target_spatial = graph.get_spatial_config(target_type);

    let use_closest = resolve == Some(SpatialResolve::Closest);

    if use_closest {
        // ── resolve='closest': use geometry boundaries for minimum distance ──
        distance_closest_mode(
            graph,
            selection,
            &source_nodes,
            &target_candidates,
            max_distance_m,
            source_spatial,
            target_spatial,
            geometry_field,
            filter_target,
            sort_target,
            max_nodes,
        )
    } else {
        // ── Default / resolve='centroid': point-to-point geodesic distance ──
        distance_point_mode(
            graph,
            selection,
            &source_nodes,
            &target_candidates,
            max_distance_m,
            resolve,
            source_spatial,
            target_spatial,
            geometry_field,
            filter_target,
            sort_target,
            max_nodes,
        )
    }
}

/// Distance using point-to-point (default or centroid resolve).
#[allow(clippy::too_many_arguments)]
fn distance_point_mode(
    graph: &DirGraph,
    selection: &mut CurrentSelection,
    source_nodes: &[NodeIndex],
    target_candidates: &[NodeIndex],
    max_distance_m: f64,
    resolve: Option<SpatialResolve>,
    source_spatial: Option<&SpatialConfig>,
    target_spatial: Option<&SpatialConfig>,
    geometry_field: Option<&str>,
    filter_target: Option<&HashMap<String, FilterCondition>>,
    sort_target: Option<&Vec<(String, bool)>>,
    max_nodes: Option<usize>,
) -> Result<(), String> {
    struct TargetLoc {
        idx: NodeIndex,
        lat: f64,
        lon: f64,
    }

    // Pre-compute target points
    let mut target_locs: Vec<TargetLoc> = Vec::with_capacity(target_candidates.len());
    for &tgt_idx in target_candidates {
        if let Some(tgt_node) = graph.get_node(tgt_idx) {
            if let Some((lat, lon)) =
                resolve_node_point(tgt_node, target_spatial, resolve, geometry_field)
            {
                target_locs.push(TargetLoc {
                    idx: tgt_idx,
                    lat,
                    lon,
                });
            }
        }
    }

    let mut matches: HashMap<NodeIndex, Vec<NodeIndex>> =
        HashMap::with_capacity(source_nodes.len());

    for &src_idx in source_nodes {
        let src_node = match graph.get_node(src_idx) {
            Some(n) => n,
            None => continue,
        };

        let (src_lat, src_lon) =
            match resolve_node_point(src_node, source_spatial, resolve, geometry_field) {
                Some(loc) => loc,
                None => continue,
            };

        let mut matched = Vec::new();
        for tgt in &target_locs {
            let dist = crate::graph::features::spatial::geodesic_distance(
                src_lat, src_lon, tgt.lat, tgt.lon,
            );
            if dist <= max_distance_m {
                matched.push(tgt.idx);
            }
        }

        if !matched.is_empty() {
            matches.insert(src_idx, matched);
        }
    }

    insert_matches_into_selection(
        graph,
        selection,
        matches,
        "distance",
        filter_target,
        sort_target,
        max_nodes,
    );
    Ok(())
}

/// Distance using closest boundary points (resolve='closest').
#[allow(clippy::too_many_arguments)]
fn distance_closest_mode(
    graph: &DirGraph,
    selection: &mut CurrentSelection,
    source_nodes: &[NodeIndex],
    target_candidates: &[NodeIndex],
    max_distance_m: f64,
    source_spatial: Option<&SpatialConfig>,
    target_spatial: Option<&SpatialConfig>,
    geometry_field: Option<&str>,
    filter_target: Option<&HashMap<String, FilterCondition>>,
    sort_target: Option<&Vec<(String, bool)>>,
    max_nodes: Option<usize>,
) -> Result<(), String> {
    let mut matches: HashMap<NodeIndex, Vec<NodeIndex>> =
        HashMap::with_capacity(source_nodes.len());

    for &src_idx in source_nodes {
        let src_node = match graph.get_node(src_idx) {
            Some(n) => n,
            None => continue,
        };

        let src_geom = resolve_node_geom(src_node, source_spatial, geometry_field);
        // Fallback to centroid point if no geometry
        let src_point = resolve_node_point(
            src_node,
            source_spatial,
            Some(SpatialResolve::Centroid),
            geometry_field,
        );

        if src_geom.is_none() && src_point.is_none() {
            continue;
        }

        let mut matched = Vec::new();
        for &tgt_idx in target_candidates {
            let tgt_node = match graph.get_node(tgt_idx) {
                Some(n) => n,
                None => continue,
            };

            let tgt_geom = resolve_node_geom(tgt_node, target_spatial, geometry_field);
            let tgt_point = resolve_node_point(
                tgt_node,
                target_spatial,
                Some(SpatialResolve::Centroid),
                geometry_field,
            );

            // Compute minimum boundary distance using best available info
            let dist = match (&src_geom, &tgt_geom) {
                (Some(sg), Some(tg)) => {
                    // Both have geometry: use point_to_geometry for better approximation
                    // (try both directions, take minimum)
                    let d1 = src_point.and_then(|(lat, lon)| {
                        crate::graph::features::spatial::point_to_geometry_distance_m(lat, lon, tg)
                            .ok()
                    });
                    let d2 = tgt_point.and_then(|(lat, lon)| {
                        crate::graph::features::spatial::point_to_geometry_distance_m(lat, lon, sg)
                            .ok()
                    });
                    match (d1, d2) {
                        (Some(a), Some(b)) => Some(a.min(b)),
                        (Some(a), None) => Some(a),
                        (None, Some(b)) => Some(b),
                        (None, None) => {
                            // Last resort: centroid-to-centroid
                            crate::graph::features::spatial::geometry_to_geometry_distance_m(sg, tg)
                                .ok()
                        }
                    }
                }
                (Some(sg), None) => {
                    // Source has geometry, target is a point
                    tgt_point.and_then(|(lat, lon)| {
                        crate::graph::features::spatial::point_to_geometry_distance_m(lat, lon, sg)
                            .ok()
                    })
                }
                (None, Some(tg)) => {
                    // Source is a point, target has geometry
                    src_point.and_then(|(lat, lon)| {
                        crate::graph::features::spatial::point_to_geometry_distance_m(lat, lon, tg)
                            .ok()
                    })
                }
                (None, None) => {
                    // Both are points — fallback to geodesic
                    match (src_point, tgt_point) {
                        (Some((lat1, lon1)), Some((lat2, lon2))) => {
                            Some(crate::graph::features::spatial::geodesic_distance(
                                lat1, lon1, lat2, lon2,
                            ))
                        }
                        _ => None,
                    }
                }
            };

            if let Some(d) = dist {
                if d <= max_distance_m {
                    matched.push(tgt_idx);
                }
            }
        }

        if !matched.is_empty() {
            matches.insert(src_idx, matched);
        }
    }

    insert_matches_into_selection(
        graph,
        selection,
        matches,
        "distance",
        filter_target,
        sort_target,
        max_nodes,
    );
    Ok(())
}

// ── Semantic: text_score ────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn semantic_score_traversal(
    graph: &DirGraph,
    selection: &mut CurrentSelection,
    target_type: &str,
    embedding_property: &str,
    threshold: f64,
    metric: crate::graph::algorithms::vector::DistanceMetric,
    filter_target: Option<&HashMap<String, FilterCondition>>,
    sort_target: Option<&Vec<(String, bool)>>,
    max_nodes: Option<usize>,
) -> Result<(), String> {
    let (source_nodes, source_type) = get_source_info(graph, selection)?;
    let target_candidates = get_target_candidates(graph, target_type)?;

    // Get embedding stores for source and target types
    let src_store = graph
        .embeddings
        .get(&(source_type.clone(), embedding_property.to_string()))
        .ok_or_else(|| {
            format!(
                "No embeddings found for type '{}', property '{}'. Use set_embedder() first.",
                source_type, embedding_property
            )
        })?;

    let tgt_store = graph
        .embeddings
        .get(&(target_type.to_string(), embedding_property.to_string()))
        .ok_or_else(|| {
            format!(
                "No embeddings found for type '{}', property '{}'. Use set_embedder() first.",
                target_type, embedding_property
            )
        })?;

    let threshold_f32 = threshold as f32;

    let mut matches: HashMap<NodeIndex, Vec<NodeIndex>> =
        HashMap::with_capacity(source_nodes.len());

    for &src_idx in &source_nodes {
        let src_embedding = match src_store.get_embedding(src_idx.index()) {
            Some(e) => e,
            None => continue,
        };

        // The source vector is constant across the inner target loop, so bind a
        // Scorer to it once (precomputing its norm for cosine); each target then
        // contributes only a dot product + its cached norm.
        let scorer = crate::graph::algorithms::vector::Scorer::new(metric, src_embedding);

        let mut matched = Vec::new();
        for &tgt_idx in &target_candidates {
            // Skip self-matches
            if tgt_idx == src_idx {
                continue;
            }
            if let Some((tgt_embedding, tgt_norm)) =
                tgt_store.get_embedding_with_norm(tgt_idx.index())
            {
                let score = scorer.score(src_embedding, tgt_embedding, tgt_norm);
                if score >= threshold_f32 {
                    matched.push(tgt_idx);
                }
            }
        }

        if !matched.is_empty() {
            matches.insert(src_idx, matched);
        }
    }

    insert_matches_into_selection(
        graph,
        selection,
        matches,
        "text_score",
        filter_target,
        sort_target,
        max_nodes,
    );
    Ok(())
}

// ── Clustering ──────────────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn cluster_traversal(
    graph: &DirGraph,
    selection: &mut CurrentSelection,
    target_type: Option<&str>,
    algorithm: &str,
    features: &[String],
    k: Option<usize>,
    eps: Option<f64>,
    min_samples: Option<usize>,
) -> Result<(), String> {
    let level_idx = selection.get_level_count().saturating_sub(1);
    let level = selection
        .get_level(level_idx)
        .ok_or("No source level for cluster traversal")?;
    let source_nodes: Vec<NodeIndex> = level.iter_node_indices().collect();
    if source_nodes.is_empty() {
        return Err("No source nodes for cluster traversal".into());
    }

    // Optionally filter by target_type
    let nodes: Vec<NodeIndex> = if let Some(tt) = target_type {
        source_nodes
            .into_iter()
            .filter(|&idx| {
                graph
                    .get_node(idx)
                    .map(|n| n.node_type == InternedKey::from_str(tt))
                    .unwrap_or(false)
            })
            .collect()
    } else {
        source_nodes
    };

    if nodes.is_empty() {
        return Err("No nodes remain after type filter for clustering".into());
    }

    // Check if features are spatial (latitude, longitude) — use haversine distance matrix
    let source_type = graph
        .get_node(nodes[0])
        .map(|n| n.node_type_str(&graph.interner).to_string())
        .unwrap_or_default();
    let spatial_cfg = graph.get_spatial_config(&source_type);
    let is_spatial = features.len() >= 2 && {
        if let Some(sc) = spatial_cfg {
            if let Some((ref lat_f, ref lon_f)) = sc.location {
                features.contains(&lat_f.to_string()) && features.contains(&lon_f.to_string())
            } else {
                false
            }
        } else {
            false
        }
    };

    // Extract feature matrix
    let mut feature_matrix: Vec<Vec<f64>> = Vec::with_capacity(nodes.len());
    for &idx in &nodes {
        let node = graph.get_node(idx).unwrap();
        let mut row = Vec::with_capacity(features.len());
        for feat in features {
            let val = node
                .get_property(feat)
                .as_deref()
                .and_then(value_to_f64)
                .unwrap_or(0.0);
            row.push(val);
        }
        feature_matrix.push(row);
    }

    let assignments = match algorithm {
        "kmeans" => {
            let k_val = k.ok_or("method='cluster' with algorithm='kmeans' requires k parameter")?;
            crate::graph::algorithms::clustering::kmeans(
                &feature_matrix,
                k_val,
                100,
                crate::graph::algorithms::Interrupt::default(),
            )
        }
        "dbscan" => {
            let eps_val =
                eps.ok_or("method='cluster' with algorithm='dbscan' requires eps parameter")?;
            let min_pts = min_samples.unwrap_or(5);
            let distances = if is_spatial {
                // Extract lat/lon columns for haversine
                let lat_idx = features
                    .iter()
                    .position(|f| {
                        spatial_cfg
                            .and_then(|sc| sc.location.as_ref())
                            .map(|(lat_f, _)| f == lat_f)
                            .unwrap_or(false)
                    })
                    .unwrap_or(0);
                let lon_idx = features
                    .iter()
                    .position(|f| {
                        spatial_cfg
                            .and_then(|sc| sc.location.as_ref())
                            .map(|(_, lon_f)| f == lon_f)
                            .unwrap_or(false)
                    })
                    .unwrap_or(1);
                let coords: Vec<(f64, f64)> = feature_matrix
                    .iter()
                    .map(|row| (row[lat_idx], row[lon_idx]))
                    .collect();
                crate::graph::algorithms::clustering::haversine_distance_matrix(
                    &coords,
                    crate::graph::algorithms::Interrupt::default(),
                )
            } else {
                let mut feat_clone = feature_matrix.clone();
                crate::graph::algorithms::clustering::normalize_features(&mut feat_clone);
                crate::graph::algorithms::clustering::euclidean_distance_matrix(
                    &feat_clone,
                    crate::graph::algorithms::Interrupt::default(),
                )
            };
            crate::graph::algorithms::clustering::dbscan(
                &distances,
                eps_val,
                min_pts,
                crate::graph::algorithms::Interrupt::default(),
            )
        }
        _ => {
            return Err(format!(
                "Unknown clustering algorithm: '{}'. Valid: 'kmeans', 'dbscan'",
                algorithm
            ))
        }
    };

    // Build selection hierarchy: cluster_id -> member nodes
    // Use a synthetic parent = None for each cluster group
    selection.add_level();
    let target_level_idx = selection.get_level_count() - 1;
    let level = selection.get_level_mut(target_level_idx).unwrap();
    level.operations = vec![SelectionOperation::Custom(format!(
        "compare(method='cluster', algorithm='{}')",
        algorithm
    ))];

    // Group nodes by cluster
    let mut clusters: HashMap<i64, Vec<NodeIndex>> = HashMap::new();
    for assign in &assignments {
        clusters
            .entry(assign.cluster)
            .or_default()
            .push(nodes[assign.index]);
    }

    // For clustering, we don't have natural parent nodes. We insert each cluster
    // group with parent=None. The first node of each cluster serves as a representative parent.
    // This allows downstream methods like statistics() to group by cluster.
    for members in clusters.values() {
        if members.is_empty() {
            continue;
        }
        // Use first member as the "parent" representative for this cluster group
        let representative = members[0];
        let children: Vec<NodeIndex> = members[1..].to_vec();
        if children.is_empty() {
            // Single-member cluster: insert with None parent
            level.add_selection(None, vec![representative]);
        } else {
            level.add_selection(Some(representative), children);
        }
    }

    Ok(())
}

pub struct ChildPropertyGroup {
    pub parent_idx: NodeIndex,
    pub parent_title: String,
    pub values: Vec<String>,
}

pub fn get_children_properties(
    graph: &DirGraph,
    selection: &CurrentSelection,
    property: &str,
) -> Vec<ChildPropertyGroup> {
    // Arena guard: disk-backed node/edge reads materialize into the query
    // arena (protocol in disk/graph.rs); no-op on memory/mapped.
    let _arena_guard = graph.graph.begin_query();
    let mut result = Vec::new();

    // Get the current level index
    let level_index = selection.get_level_count().saturating_sub(1);

    // Get all parents with their children
    if let Some(level) = selection.get_level(level_index) {
        for (&parent_opt, children) in &level.selections {
            if let Some(parent) = parent_opt {
                // Get parent title
                let parent_title = if let Some(node) = graph.get_node(parent) {
                    match node.get_field_ref("title").as_deref() {
                        Some(Value::String(s)) => s.clone(),
                        _ => format!("node_{}", parent.index()),
                    }
                } else {
                    format!("node_{}", parent.index())
                };

                // For each parent, collect property values from children
                let mut values_list = Vec::new();

                for &child_idx in children {
                    if let Some(node) = graph.get_node(child_idx) {
                        let value = match node.get_field_ref(property).as_deref() {
                            Some(Value::String(s)) => s.clone(),
                            Some(Value::Int64(i)) => i.to_string(),
                            Some(Value::Float64(f)) => f.to_string(),
                            Some(Value::Boolean(b)) => b.to_string(),
                            Some(Value::UniqueId(u)) => u.to_string(),
                            Some(Value::DateTime(d)) => d.format("%Y-%m-%d").to_string(),
                            Some(Value::Point { lat, lon }) => {
                                format!("point({}, {})", lat, lon)
                            }
                            Some(Value::Duration {
                                months,
                                days,
                                seconds,
                            }) => format!(
                                "duration(months={}, days={}, seconds={})",
                                months, days, seconds
                            ),
                            Some(Value::Null) => "null".to_string(),
                            Some(Value::NodeRef(idx)) => format!("node#{}", idx),
                            // Phase A.1 — collection / graph-entity property
                            // values are an edge case for hierarchical
                            // child-grouping; delegate to format_value.
                            Some(other) => crate::datatypes::values::format_value(other),
                            None => continue,
                        };

                        values_list.push(value);
                    }
                }

                result.push(ChildPropertyGroup {
                    parent_idx: parent,
                    parent_title,
                    values: values_list,
                });
            }
        }
    }

    result
}

/// Helper to format a list of values with optional truncation
fn format_property_list(values: &[String], max_length: Option<usize>) -> String {
    let joined = values.join(", ");
    match max_length {
        Some(max) if joined.len() > max => {
            format!("{}...", &joined[..max.saturating_sub(3)])
        }
        _ => joined,
    }
}

pub fn format_for_storage(
    property_groups: &[ChildPropertyGroup],
    max_length: Option<usize>,
) -> Vec<(Option<NodeIndex>, Value)> {
    property_groups
        .iter()
        .map(|group| {
            let formatted = format_property_list(&group.values, max_length);
            (Some(group.parent_idx), Value::String(formatted))
        })
        .collect()
}

pub fn format_for_dictionary(
    property_groups: &[ChildPropertyGroup],
    max_length: Option<usize>,
) -> Vec<(String, String)> {
    property_groups
        .iter()
        .map(|group| {
            let formatted = format_property_list(&group.values, max_length);
            (group.parent_title.clone(), formatted)
        })
        .collect()
}
