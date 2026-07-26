//! `add_properties` — enrich the leaf level of a traversal selection with
//! properties copied, renamed, aggregated, or spatially computed from ancestor
//! levels.
//!
//! Split out of `maintain.rs`, which had grown past the file-size ceiling. This
//! is a self-contained unit: one public entry point (`add_properties`), the
//! aggregate-mode branch it delegates to, and the expression/geometry helpers
//! only those two use. Nothing here writes topology.
//!
//! # Index maintenance is part of the contract here
//!
//! Both write loops apply property changes straight to `node.properties` rather
//! than going through the Cypher SET path, so they bypass the per-write index
//! maintenance `DirGraph::plan_property_write` performs. `try_index_lookup`
//! consults `property_indices` with no freshness check, so a stale index does not
//! degrade to a scan — it returns a node under a value it no longer holds. Every
//! write therefore funnels through [`apply_property_updates`], which refreshes
//! the indexes of each written type and publishes the version bump.

use crate::datatypes::Value;
// Defined in `maintain` rather than here on purpose: it is the return type of the
// public `kglite::api::mutation::add_properties`, and the pinned Rust API
// baseline records that type at its canonical path. Moving the definition would
// churn a public signature for an internal file split.
use crate::graph::mutation::maintain::AddPropertiesReport;
use crate::graph::schema::{CurrentSelection, DirGraph, InternedKey};
use crate::graph::storage::{GraphRead, GraphWrite};
use petgraph::graph::NodeIndex;
use std::collections::{HashMap, HashSet};

/// Specifies how properties should be copied from a source type.
#[derive(Debug)]
pub enum PropertySpec {
    /// Copy listed properties as-is: `['name', 'status']`
    CopyList(Vec<String>),
    /// Copy all properties: `[]`
    CopyAll,
    /// Rename/aggregate/spatial: `{'new_name': 'source_expr'}`
    RenameMap(HashMap<String, String>),
}

/// Enriches the leaf (most recent) level nodes by copying, renaming, aggregating,
/// or computing properties from ancestor nodes in the traversal hierarchy.
pub fn add_properties(
    graph: &mut DirGraph,
    selection: &CurrentSelection,
    property_spec: HashMap<String, PropertySpec>,
) -> Result<AddPropertiesReport, String> {
    let _arena_guard = graph.graph.begin_query(); // disk arena guard (owned; no-op on memory/mapped)
    graph
        .prepare_disk_mutation()
        .map_err(|e| format!("disk mutation lease failed: {e}"))?;
    let level_count = selection.get_level_count();
    if level_count == 0 {
        return Ok(AddPropertiesReport {
            nodes_updated: 0,
            properties_set: 0,
        });
    }

    let target_level = level_count - 1;

    // Build type → level index map
    let mut type_to_level: HashMap<String, usize> = HashMap::new();
    for lvl_idx in 0..level_count {
        if let Some(level) = selection.get_level(lvl_idx) {
            for node_idx in level.iter_node_indices() {
                if let Some(node) = graph.get_node(node_idx) {
                    type_to_level
                        .entry(node.node_type_str(&graph.interner).to_string())
                        .or_insert(lvl_idx);
                }
            }
        }
    }

    // Validate requested types exist in the traversal chain
    for source_type in property_spec.keys() {
        if !type_to_level.contains_key(source_type) {
            return Err(format!(
                "Source type '{}' not found in traversal chain. Available: {:?}",
                source_type,
                type_to_level.keys().collect::<Vec<_>>()
            ));
        }
    }

    // Build reverse parent maps: child → parent for each level
    let mut parent_maps: Vec<HashMap<NodeIndex, NodeIndex>> = vec![HashMap::new(); level_count];
    for (lvl_idx, pmap) in parent_maps.iter_mut().enumerate().skip(1) {
        if let Some(level) = selection.get_level(lvl_idx) {
            for (parent_opt, children) in level.iter_groups() {
                if let Some(parent) = parent_opt {
                    for &child in children {
                        pmap.insert(child, *parent);
                    }
                }
            }
        }
    }

    // Check if any spec requires aggregation
    let has_aggregation = property_spec.values().any(|spec| {
        if let PropertySpec::RenameMap(map) = spec {
            map.values().any(|expr| is_aggregate_expr(expr))
        } else {
            false
        }
    });

    if has_aggregation {
        return add_properties_aggregate(
            graph,
            selection,
            &property_spec,
            &type_to_level,
            &parent_maps,
            target_level,
        );
    }

    // Standard mode: copy/rename from ancestor onto each leaf node
    let target_level_data = match selection.get_level(target_level) {
        Some(level) if !level.is_empty() => level,
        _ => {
            return Ok(AddPropertiesReport {
                nodes_updated: 0,
                properties_set: 0,
            });
        }
    };

    // Collect updates first (to avoid borrow issues with graph)
    let mut updates: Vec<(NodeIndex, HashMap<String, Value>)> = Vec::new();

    // Arena guard: node_weight materializes on the disk backend (protocol
    // in disk/graph.rs); dropped before the &mut apply loop below.
    let collect_guard = graph.graph.begin_query();
    for (_parent_opt, targets) in target_level_data.iter_groups() {
        for &target_idx in targets {
            let mut props_to_set: HashMap<String, Value> = HashMap::new();

            for (source_type, spec) in &property_spec {
                let source_level = match type_to_level.get(source_type) {
                    Some(&lvl) => lvl,
                    None => continue,
                };

                let ancestor_idx =
                    walk_to_ancestor(target_idx, target_level, source_level, &parent_maps);
                let ancestor_idx = match ancestor_idx {
                    Some(idx) => idx,
                    None => continue,
                };

                let ancestor_node = match graph.graph.node_weight(ancestor_idx) {
                    Some(n) => n,
                    None => continue,
                };

                match spec {
                    PropertySpec::CopyAll => {
                        for (k, v) in ancestor_node.property_iter(&graph.interner) {
                            props_to_set.insert(k.to_string(), v.clone());
                        }
                    }
                    PropertySpec::CopyList(prop_names) => {
                        for prop_name in prop_names {
                            if let Some(val) = ancestor_node.get_property(prop_name) {
                                props_to_set.insert(prop_name.clone(), val.into_owned());
                            }
                        }
                    }
                    PropertySpec::RenameMap(map) => {
                        for (target_name, source_expr) in map {
                            if is_spatial_compute(source_expr) {
                                if let Some(val) = compute_spatial_property(
                                    graph,
                                    target_idx,
                                    ancestor_idx,
                                    source_expr,
                                ) {
                                    props_to_set.insert(target_name.clone(), val);
                                }
                            } else if let Some(val) = ancestor_node.get_property(source_expr) {
                                props_to_set.insert(target_name.clone(), val.into_owned());
                            }
                        }
                    }
                }
            }

            if !props_to_set.is_empty() {
                updates.push((target_idx, props_to_set));
            }
        }
    }

    drop(collect_guard);

    Ok(apply_property_updates(graph, updates))
}

/// Apply collected per-node property writes, then refresh the secondary indexes
/// of every written node type and publish the mutation.
///
/// Shared by [`add_properties`] and [`add_properties_aggregate`], which both
/// write `node.properties` directly rather than going through the Cypher SET
/// path. That bypasses the per-write index maintenance
/// `DirGraph::plan_property_write` performs, and `try_index_lookup`
/// (`core/pattern_matching/matcher.rs`) consults `property_indices`
/// unconditionally, with no freshness check. So a stale index does not degrade
/// to a scan: `MATCH (n:T {prop: <overwritten value>})` keeps returning the node
/// that used to hold that value — a wrong answer, not a slow one.
///
/// One rebuild per written type, which is a no-op (three `is_empty` checks) when
/// the type carries no index. Extracted so the two callers cannot drift: the
/// duplicated tail is exactly how one of them lost this maintenance.
fn apply_property_updates<I>(graph: &mut DirGraph, updates: I) -> AddPropertiesReport
where
    I: IntoIterator<Item = (NodeIndex, HashMap<String, Value>)>,
{
    let mut nodes_updated = 0;
    let mut properties_set = 0;
    let mut touched_types: HashSet<String> = HashSet::new();
    for (node_idx, props) in updates {
        // Pre-intern keys before getting mutable node reference (split borrow)
        let interned_props: Vec<(InternedKey, Value)> = props
            .into_iter()
            .map(|(k, v)| (graph.interner.get_or_intern(&k), v))
            .collect();
        if let Some(node) = GraphWrite::node_weight_mut(&mut graph.graph, node_idx) {
            let count = interned_props.len();
            for (ik, v) in interned_props {
                node.properties.insert(ik, v);
            }
            nodes_updated += 1;
            properties_set += count;
            touched_types.insert(node.node_type_str(&graph.interner).to_string());
        }
    }

    for node_type in &touched_types {
        graph.refresh_indexes_for_type(node_type);
    }
    if nodes_updated > 0 {
        // Publish the write so version-keyed caches and freshness checks see it.
        graph.bump_version();
    }

    AddPropertiesReport {
        nodes_updated,
        properties_set,
    }
}

fn walk_to_ancestor(
    start: NodeIndex,
    start_level: usize,
    target_level: usize,
    parent_maps: &[HashMap<NodeIndex, NodeIndex>],
) -> Option<NodeIndex> {
    if start_level == target_level {
        return Some(start);
    }
    if target_level >= start_level {
        return None;
    }
    let mut current = start;
    for lvl in (target_level + 1..=start_level).rev() {
        current = *parent_maps[lvl].get(&current)?;
    }
    Some(current)
}

fn is_aggregate_expr(expr: &str) -> bool {
    let trimmed = expr.trim();
    trimmed == "count(*)"
        || trimmed.starts_with("sum(")
        || trimmed.starts_with("mean(")
        || trimmed.starts_with("avg(")
        || trimmed.starts_with("min(")
        || trimmed.starts_with("max(")
        || trimmed.starts_with("std(")
        || trimmed.starts_with("collect(")
}

fn is_spatial_compute(expr: &str) -> bool {
    matches!(
        expr.trim(),
        "distance" | "area" | "perimeter" | "centroid_lat" | "centroid_lon"
    )
}

fn extract_agg_property(expr: &str) -> Option<&str> {
    let trimmed = expr.trim();
    if trimmed == "count(*)" {
        return None;
    }
    let start = trimmed.find('(')?;
    let end = trimmed.rfind(')')?;
    if start + 1 < end {
        Some(trimmed[start + 1..end].trim())
    } else {
        None
    }
}

fn compute_spatial_property(
    graph: &DirGraph,
    leaf_idx: NodeIndex,
    ancestor_idx: NodeIndex,
    spatial_fn: &str,
) -> Option<Value> {
    let leaf_node = graph.get_node(leaf_idx)?;
    let ancestor_node = graph.get_node(ancestor_idx)?;
    let leaf_spatial = graph.get_spatial_config(leaf_node.node_type_str(&graph.interner));
    let ancestor_spatial = graph.get_spatial_config(ancestor_node.node_type_str(&graph.interner));

    match spatial_fn.trim() {
        "distance" => {
            let (lat1, lon1) = resolve_location(leaf_node, leaf_spatial)?;
            let (lat2, lon2) = resolve_location(ancestor_node, ancestor_spatial)?;
            Some(Value::Float64(
                crate::graph::features::spatial::geodesic_distance(lat1, lon1, lat2, lon2),
            ))
        }
        "area" => {
            let geom = resolve_geometry(ancestor_node, ancestor_spatial)?;
            crate::graph::features::spatial::geometry_area_m2(&geom)
                .ok()
                .map(Value::Float64)
        }
        "perimeter" => {
            let geom = resolve_geometry(ancestor_node, ancestor_spatial)?;
            crate::graph::features::spatial::geometry_perimeter_m(&geom)
                .ok()
                .map(Value::Float64)
        }
        "centroid_lat" => {
            let geom = resolve_geometry(ancestor_node, ancestor_spatial)?;
            crate::graph::features::spatial::geometry_centroid(&geom)
                .ok()
                .map(|(lat, _)| Value::Float64(lat))
        }
        "centroid_lon" => {
            let geom = resolve_geometry(ancestor_node, ancestor_spatial)?;
            crate::graph::features::spatial::geometry_centroid(&geom)
                .ok()
                .map(|(_, lon)| Value::Float64(lon))
        }
        _ => None,
    }
}

fn resolve_location(
    node: &crate::graph::schema::NodeData,
    spatial_config: Option<&crate::graph::schema::SpatialConfig>,
) -> Option<(f64, f64)> {
    let sc = spatial_config?;
    if let Some((ref lat_f, ref lon_f)) = sc.location {
        let lat = node
            .get_property(lat_f)
            .as_deref()
            .and_then(mg_value_to_f64)?;
        let lon = node
            .get_property(lon_f)
            .as_deref()
            .and_then(mg_value_to_f64)?;
        return Some((lat, lon));
    }
    if let Some(ref geom_f) = sc.geometry {
        if let Some(Value::String(wkt)) = node.get_property(geom_f).as_deref() {
            if let Ok(geom) = crate::graph::features::spatial::parse_wkt(wkt) {
                return crate::graph::features::spatial::geometry_centroid(&geom).ok();
            }
        }
    }
    None
}

fn resolve_geometry(
    node: &crate::graph::schema::NodeData,
    spatial_config: Option<&crate::graph::schema::SpatialConfig>,
) -> Option<geo::geometry::Geometry<f64>> {
    let sc = spatial_config?;
    let geom_field = sc.geometry.as_deref()?;
    match node.get_property(geom_field).as_deref() {
        Some(Value::String(wkt)) => crate::graph::features::spatial::parse_wkt(wkt).ok(),
        _ => None,
    }
}

fn mg_value_to_f64(v: &Value) -> Option<f64> {
    match v {
        Value::Float64(f) => Some(*f),
        Value::Int64(i) => Some(*i as f64),
        Value::String(s) => s.parse().ok(),
        _ => None,
    }
}

/// Aggregation mode: groups leaf nodes by ancestor and computes aggregate values.
#[allow(clippy::too_many_arguments)]
fn add_properties_aggregate(
    graph: &mut DirGraph,
    selection: &CurrentSelection,
    property_spec: &HashMap<String, PropertySpec>,
    type_to_level: &HashMap<String, usize>,
    parent_maps: &[HashMap<NodeIndex, NodeIndex>],
    target_level: usize,
) -> Result<AddPropertiesReport, String> {
    let target_level_data = match selection.get_level(target_level) {
        Some(level) if !level.is_empty() => level,
        _ => {
            return Ok(AddPropertiesReport {
                nodes_updated: 0,
                properties_set: 0,
            });
        }
    };

    let mut updates: HashMap<NodeIndex, HashMap<String, Value>> = HashMap::new();

    // Arena guard: node_weight materializes on the disk backend (protocol
    // in disk/graph.rs); dropped before the &mut apply loop below.
    let collect_guard = graph.graph.begin_query();
    for (source_type, spec) in property_spec {
        let source_level = match type_to_level.get(source_type) {
            Some(&lvl) => lvl,
            None => continue,
        };

        match spec {
            PropertySpec::CopyList(props) => {
                for (_parent_opt, targets) in target_level_data.iter_groups() {
                    for &target_idx in targets {
                        if let Some(ancestor_idx) =
                            walk_to_ancestor(target_idx, target_level, source_level, parent_maps)
                        {
                            if let Some(ancestor_node) = graph.get_node(ancestor_idx) {
                                for prop_name in props {
                                    if let Some(val) = ancestor_node.get_property(prop_name) {
                                        updates
                                            .entry(target_idx)
                                            .or_default()
                                            .insert(prop_name.clone(), val.into_owned());
                                    }
                                }
                            }
                        }
                    }
                }
            }
            PropertySpec::CopyAll => {
                for (_parent_opt, targets) in target_level_data.iter_groups() {
                    for &target_idx in targets {
                        if let Some(ancestor_idx) =
                            walk_to_ancestor(target_idx, target_level, source_level, parent_maps)
                        {
                            if let Some(ancestor_node) = graph.graph.node_weight(ancestor_idx) {
                                for (k, v) in ancestor_node.property_iter(&graph.interner) {
                                    updates
                                        .entry(target_idx)
                                        .or_default()
                                        .insert(k.to_string(), v.clone());
                                }
                            }
                        }
                    }
                }
            }
            PropertySpec::RenameMap(rename_map) => {
                for (target_name, source_expr) in rename_map {
                    if is_aggregate_expr(source_expr) {
                        let agg_prop = extract_agg_property(source_expr);

                        // Group leaf nodes by ancestor at source_level
                        let mut groups: HashMap<NodeIndex, Vec<NodeIndex>> = HashMap::new();
                        for (_parent_opt, targets) in target_level_data.iter_groups() {
                            for &target_idx in targets {
                                if let Some(ancestor) = walk_to_ancestor(
                                    target_idx,
                                    target_level,
                                    source_level,
                                    parent_maps,
                                ) {
                                    groups.entry(ancestor).or_default().push(target_idx);
                                }
                            }
                        }

                        for (ancestor_idx, leaf_indices) in &groups {
                            let values: Vec<f64> = if let Some(prop) = agg_prop {
                                leaf_indices
                                    .iter()
                                    .filter_map(|&idx| {
                                        graph.get_node(idx).and_then(|n| {
                                            n.get_property(prop)
                                                .as_deref()
                                                .and_then(mg_value_to_f64)
                                        })
                                    })
                                    .collect()
                            } else {
                                vec![]
                            };

                            let agg_value =
                                compute_aggregate(source_expr, &values, leaf_indices.len());
                            updates
                                .entry(*ancestor_idx)
                                .or_default()
                                .insert(target_name.clone(), agg_value);
                        }
                    } else if is_spatial_compute(source_expr) {
                        for (_parent_opt, targets) in target_level_data.iter_groups() {
                            for &target_idx in targets {
                                if let Some(ancestor_idx) = walk_to_ancestor(
                                    target_idx,
                                    target_level,
                                    source_level,
                                    parent_maps,
                                ) {
                                    if let Some(val) = compute_spatial_property(
                                        graph,
                                        target_idx,
                                        ancestor_idx,
                                        source_expr,
                                    ) {
                                        updates
                                            .entry(target_idx)
                                            .or_default()
                                            .insert(target_name.clone(), val);
                                    }
                                }
                            }
                        }
                    } else {
                        // Simple rename
                        for (_parent_opt, targets) in target_level_data.iter_groups() {
                            for &target_idx in targets {
                                if let Some(ancestor_idx) = walk_to_ancestor(
                                    target_idx,
                                    target_level,
                                    source_level,
                                    parent_maps,
                                ) {
                                    if let Some(ancestor_node) = graph.get_node(ancestor_idx) {
                                        if let Some(val) = ancestor_node.get_property(source_expr) {
                                            updates
                                                .entry(target_idx)
                                                .or_default()
                                                .insert(target_name.clone(), val.into_owned());
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    drop(collect_guard);

    Ok(apply_property_updates(graph, updates))
}

fn compute_aggregate(expr: &str, values: &[f64], count: usize) -> Value {
    let trimmed = expr.trim();
    if trimmed == "count(*)" {
        return Value::Int64(count as i64);
    }
    if trimmed.starts_with("collect(") {
        let s = values
            .iter()
            .map(|v| format!("{}", v))
            .collect::<Vec<_>>()
            .join(", ");
        return Value::String(s);
    }
    if values.is_empty() {
        return Value::Null;
    }
    if trimmed.starts_with("sum(") {
        Value::Float64(values.iter().sum())
    } else if trimmed.starts_with("mean(") || trimmed.starts_with("avg(") {
        Value::Float64(values.iter().sum::<f64>() / values.len() as f64)
    } else if trimmed.starts_with("min(") {
        Value::Float64(values.iter().copied().fold(f64::INFINITY, f64::min))
    } else if trimmed.starts_with("max(") {
        Value::Float64(values.iter().copied().fold(f64::NEG_INFINITY, f64::max))
    } else if trimmed.starts_with("std(") {
        if values.len() < 2 {
            Value::Float64(0.0)
        } else {
            let mean = values.iter().sum::<f64>() / values.len() as f64;
            let variance =
                values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (values.len() - 1) as f64;
            Value::Float64(variance.sqrt())
        }
    } else {
        Value::Null
    }
}
