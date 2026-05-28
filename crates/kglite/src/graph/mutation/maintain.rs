// src/graph/maintain.rs
use crate::datatypes::{DataFrame, Value};
use crate::graph::introspection::reporting::{ConnectionOperationReport, NodeOperationReport};
use crate::graph::mutation::batch::{
    BatchProcessor, ConflictHandling, ConnectionBatchProcessor, NodeAction,
};
use crate::graph::schema::{CurrentSelection, DirGraph, InternedKey, TypeSchema, PROVISIONAL_KEY};
use crate::graph::storage::lookups::{CombinedTypeLookup, TypeLookup};
use crate::graph::storage::{GraphRead, GraphWrite};
use petgraph::graph::{EdgeIndex, NodeIndex};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

fn check_data_validity(df_data: &DataFrame, unique_id_field: &str) -> Result<(), String> {
    // Remove strict UniqueId type verification to allow nulls
    if !df_data.verify_column(unique_id_field) {
        let available_cols: Vec<_> = df_data.get_column_names();
        return Err(format!(
            "Column '{}' not found in DataFrame. Available columns: [{}]",
            unique_id_field,
            available_cols.join(", ")
        ));
    }
    Ok(())
}

fn get_column_types(df_data: &DataFrame) -> HashMap<String, String> {
    let mut types = HashMap::new();
    for col_name in df_data.get_column_names() {
        let col_type = df_data.get_column_type(&col_name);
        types.insert(col_name.clone(), col_type.to_string());
    }
    types
}

pub fn add_nodes(
    graph: &mut DirGraph,
    df_data: DataFrame,
    node_type: String,
    unique_id_field: String,
    node_title_field: Option<String>,
    conflict_handling: Option<String>,
) -> Result<NodeOperationReport, String> {
    // Parse conflict handling option
    let conflict_mode = match conflict_handling.as_deref() {
        Some("replace") => ConflictHandling::Replace,
        Some("skip") => ConflictHandling::Skip,
        Some("preserve") => ConflictHandling::Preserve,
        Some("sum") => ConflictHandling::Sum,
        Some("update") | None => ConflictHandling::Update, // Default
        Some(other) => return Err(format!(
            "Unknown conflict handling mode: '{}'. Valid options: 'update' (default), 'replace', 'skip', 'preserve', 'sum'",
            other
        )),
    };

    let should_update_title = node_title_field.is_some();
    let title_field = node_title_field.unwrap_or_else(|| unique_id_field.clone());
    check_data_validity(&df_data, &unique_id_field)?;

    // Track errors
    let mut errors = Vec::new();

    let df_column_types = get_column_types(&df_data);

    // Check for type mismatches if metadata already exists
    if let Some(existing_meta) = graph.get_node_type_metadata(&node_type) {
        for (col_name, col_type) in &df_column_types {
            if let Some(existing_type) = existing_meta.get(col_name) {
                if existing_type != col_type {
                    errors.push(format!(
                        "Type mismatch for property '{}': existing schema has '{}', but data has '{}'",
                        col_name, existing_type, col_type
                    ));
                }
            }
        }
    }

    // Upsert node type metadata (merges new column types into existing)
    graph.upsert_node_type_metadata(&node_type, df_column_types);

    // Record original field name aliases so users can query by original column name
    if unique_id_field != "id" {
        graph
            .id_field_aliases
            .insert(node_type.clone(), unique_id_field.clone());
    }
    // Only register the title alias when the caller explicitly named one.
    // Otherwise a follow-up add_nodes(..., node_title_field=None) would
    // silently rebind the alias to unique_id_field, making `s.id` resolve
    // to the stored title.
    if should_update_title && title_field != "title" {
        graph
            .title_field_aliases
            .insert(node_type.clone(), title_field.clone());
    }

    let type_lookup =
        TypeLookup::from_id_indices(&graph.id_indices, &graph.graph, node_type.clone())?;
    let id_idx = df_data
        .get_column_index(&unique_id_field)
        .ok_or_else(|| format!("Column '{}' not found", unique_id_field))?;
    let title_idx = df_data
        .get_column_index(&title_field)
        .ok_or_else(|| format!("Column '{}' not found", title_field))?;

    // OPTIMIZATION: Pre-compute property column info (name + index) to avoid repeated lookups
    // This avoids: 1) string comparisons in the loop, 2) HashMap lookups per property
    let property_columns: Vec<(String, usize)> = df_data
        .get_column_names()
        .into_iter()
        .filter_map(|col_name| {
            if col_name != unique_id_field && col_name != title_field {
                df_data
                    .get_column_index(&col_name)
                    .map(|idx| (col_name, idx))
            } else {
                None
            }
        })
        .collect();

    // Build TypeSchema from DataFrame columns for compact storage
    let schema_keys: Vec<InternedKey> = property_columns
        .iter()
        .map(|(col_name, _)| graph.interner.get_or_intern(col_name))
        .collect();
    let type_schema = Arc::new(TypeSchema::from_keys(schema_keys));

    // Store or extend the schema for this node type
    let existing = graph.type_schemas.get(&node_type).cloned();
    if let Some(existing_schema) = existing {
        // Extend the existing schema with any new keys
        let mut merged = (*existing_schema).clone();
        for (_, key) in type_schema.iter() {
            merged.add_key(key);
        }
        let merged_arc = Arc::new(merged);
        graph.type_schemas.insert(node_type.clone(), merged_arc);
    } else {
        graph.type_schemas.insert(node_type.clone(), type_schema);
    }

    // Pre-intern property column keys once (avoids re-interning per row)
    let interned_columns: Vec<(InternedKey, usize)> = property_columns
        .iter()
        .map(|(col_name, col_idx)| (graph.interner.get_or_intern(col_name), *col_idx))
        .collect();
    let property_count = property_columns.len();
    let mut batch = BatchProcessor::new(df_data.row_count());
    let mut skipped_count = 0;
    let mut skipped_null_id = 0;
    let mut skipped_parse_fail = 0;

    for row_idx in 0..df_data.row_count() {
        let id = match df_data.get_value_by_index(row_idx, id_idx) {
            Some(Value::Null) => {
                skipped_count += 1;
                skipped_null_id += 1;
                continue;
            }
            Some(id) => id,
            None => {
                skipped_count += 1;
                skipped_parse_fail += 1;
                continue;
            }
        };

        let title = df_data
            .get_value_by_index(row_idx, title_idx)
            .unwrap_or(Value::Null);

        // Use pre-interned keys — avoids HashMap allocation and string cloning per row
        let mut properties_interned = Vec::with_capacity(property_count);
        for (interned_key, col_idx) in &interned_columns {
            let value = df_data
                .get_value_by_index(row_idx, *col_idx)
                .unwrap_or(Value::Null);
            if !matches!(value, Value::Null) {
                properties_interned.push((*interned_key, value));
            }
        }

        let action = match type_lookup.check_uid(&id) {
            Some(node_idx) => {
                // Determine if we should update the title
                let title_update = if should_update_title {
                    Some(title)
                } else {
                    None
                };

                // Update path still uses HashMap (less frequent, interning handled in batch)
                let mut properties = HashMap::with_capacity(properties_interned.len());
                for (ik, v) in properties_interned {
                    let name = graph.interner.resolve(ik);
                    properties.insert(name.to_string(), v);
                }

                NodeAction::Update {
                    node_idx,
                    title: title_update,
                    properties,
                    conflict_mode,
                }
            }
            None => NodeAction::CreateInterned {
                node_type: node_type.clone(),
                id,
                title,
                properties: properties_interned,
            },
        };
        batch.add_action(action, graph)?;
    }

    // Report skip reasons
    if skipped_null_id > 0 {
        errors.push(format!(
            "Skipped {} rows: null values in ID field '{}'",
            skipped_null_id, unique_id_field
        ));
    }
    if skipped_parse_fail > 0 {
        errors.push(format!(
            "Skipped {} rows: could not parse ID field '{}'. If IDs are strings, pass column_types={{'{}'
: 'string'}}",
            skipped_parse_fail, unique_id_field, unique_id_field
        ));
    }

    // Execute the batch and get the statistics
    let (stats, metrics) = batch.execute(graph)?;

    // Invalidate the type's id_index. The batch added rows to type_indices
    // but no longer touches id_indices, so any pre-existing entry is now
    // missing the freshly-added ids. Dropping the entry forces the next
    // `build_id_index` (or `lookup_by_id`) to rebuild from type_indices.
    graph.id_indices.remove(&node_type);

    // Calculate elapsed time
    let elapsed_ms = metrics.processing_time * 1000.0; // Convert to milliseconds

    // Create and return the operation report with timestamp and errors
    let mut report = NodeOperationReport::new(
        "add_nodes".to_string(),
        stats.creates,
        stats.updates,
        skipped_count,
        elapsed_ms,
    );

    // Add errors if we found any
    if !errors.is_empty() {
        report = report.with_errors(errors);
    }

    Ok(report)
}

#[allow(clippy::too_many_arguments)]
pub fn add_connections(
    graph: &mut DirGraph,
    df_data: DataFrame,
    connection_type: String,
    source_type: String,
    source_id_field: String,
    target_type: String,
    target_id_field: String,
    source_title_field: Option<String>,
    target_title_field: Option<String>,
    conflict_handling: Option<String>,
) -> Result<ConnectionOperationReport, String> {
    // Parse conflict handling option
    let conflict_mode = match conflict_handling.as_deref() {
        Some("replace") => ConflictHandling::Replace,
        Some("skip") => ConflictHandling::Skip,
        Some("preserve") => ConflictHandling::Preserve,
        Some("sum") => ConflictHandling::Sum,
        Some("update") | None => ConflictHandling::Update, // Default
        Some(other) => return Err(format!(
            "Unknown conflict handling mode: '{}'. Valid options: 'update' (default), 'replace', 'skip', 'preserve', 'sum'",
            other
        )),
    };

    // Track errors
    let mut errors = Vec::new();

    let available_cols: Vec<_> = df_data.get_column_names();
    if !df_data.verify_column(&source_id_field) {
        return Err(format!(
            "Source ID column '{}' not found in DataFrame. Available columns: [{}]",
            source_id_field,
            available_cols.join(", ")
        ));
    }
    if !df_data.verify_column(&target_id_field) {
        return Err(format!(
            "Target ID column '{}' not found in DataFrame. Available columns: [{}]",
            target_id_field,
            available_cols.join(", ")
        ));
    }

    // A source/target type that doesn't exist yet is no longer an
    // error: an edge to a missing endpoint vivifies a stub node (which
    // registers the type). See Pass B below.

    let source_id_idx = df_data
        .get_column_index(&source_id_field)
        .ok_or_else(|| format!("Source ID column '{}' not found", source_id_field))?;
    let target_id_idx = df_data
        .get_column_index(&target_id_field)
        .ok_or_else(|| format!("Target ID column '{}' not found", target_id_field))?;

    // Use as_ref() to borrow rather than move
    let source_title_idx = source_title_field
        .as_ref()
        .and_then(|field| df_data.get_column_index(field));
    let target_title_idx = target_title_field
        .as_ref()
        .and_then(|field| df_data.get_column_index(field));

    let lookup = CombinedTypeLookup::from_id_indices(
        &graph.id_indices,
        &graph.graph,
        source_type.clone(),
        target_type.clone(),
    )?;
    let mut batch = ConnectionBatchProcessor::new(df_data.row_count());
    // Set the conflict handling mode
    batch.set_conflict_mode(conflict_mode);
    // Skip edge existence checks on initial load (no existing edges of this type)
    let is_initial_load = !graph
        .connection_type_metadata
        .contains_key(&connection_type);
    batch.set_skip_existence_check(is_initial_load);

    let mut skipped_count = 0;
    let mut skipped_null_source = 0;
    let mut skipped_null_target = 0;
    // Edges whose endpoint has no node are deferred — not dropped: the
    // missing endpoints are vivified as provisional stub nodes (Pass B)
    // and the rows replayed (Pass C). `missing_*` are deduped ordered
    // id lists; `deferred` holds (row, source_id, target_id).
    let mut deferred: Vec<(usize, Value, Value)> = Vec::new();
    let mut missing_sources: Vec<Value> = Vec::new();
    let mut missing_targets: Vec<Value> = Vec::new();
    let mut seen_missing_source: HashSet<Value> = HashSet::new();
    let mut seen_missing_target: HashSet<Value> = HashSet::new();

    // Cache column names and pre-compute which columns are property columns (not ID or title fields)
    // This avoids repeated allocations and string comparisons in the loop
    let property_columns: Vec<String> = df_data
        .get_column_names()
        .into_iter()
        .filter(|col_name| {
            let is_id_field = *col_name == source_id_field || *col_name == target_id_field;
            let is_source_title = source_title_field
                .as_ref()
                .is_some_and(|field| *col_name == *field);
            let is_target_title = target_title_field
                .as_ref()
                .is_some_and(|field| *col_name == *field);
            !is_id_field && !is_source_title && !is_target_title
        })
        .collect();

    // Extract a row's edge properties — shared by the happy path and
    // the deferred-row replay (Pass C). Skip nulls: property access
    // returns Null for missing keys anyway.
    let extract_props = |row_idx: usize| -> HashMap<String, Value> {
        let mut properties = HashMap::with_capacity(property_columns.len());
        for col_name in &property_columns {
            if let Some(value) = df_data.get_value(row_idx, col_name) {
                if !matches!(value, Value::Null) {
                    properties.insert(col_name.clone(), value);
                }
            }
        }
        properties
    };

    // Pass A — connect rows whose endpoints both exist; defer the rest.
    for row_idx in 0..df_data.row_count() {
        let source_id = match df_data.get_value_by_index(row_idx, source_id_idx) {
            Some(Value::Null) | None => {
                skipped_count += 1;
                skipped_null_source += 1;
                continue;
            }
            Some(id) => id,
        };
        let target_id = match df_data.get_value_by_index(row_idx, target_id_idx) {
            Some(Value::Null) | None => {
                skipped_count += 1;
                skipped_null_target += 1;
                continue;
            }
            Some(id) => id,
        };

        let (source_idx, target_idx) = match (
            lookup.check_source(&source_id),
            lookup.check_target(&target_id),
        ) {
            (Some(src_idx), Some(tgt_idx)) => (src_idx, tgt_idx),
            (s_opt, t_opt) => {
                // One or both endpoints missing — defer the row rather
                // than drop the edge. The missing ids are vivified as
                // provisional stubs in Pass B, then replayed in Pass C.
                if s_opt.is_none() && seen_missing_source.insert(source_id.clone()) {
                    missing_sources.push(source_id.clone());
                }
                if t_opt.is_none() && seen_missing_target.insert(target_id.clone()) {
                    missing_targets.push(target_id.clone());
                }
                deferred.push((row_idx, source_id, target_id));
                continue;
            }
        };

        update_node_titles(
            graph,
            source_idx,
            target_idx,
            row_idx,
            source_title_idx,
            target_title_idx,
            &df_data,
        )?;
        if let Err(e) = batch.add_connection(
            source_idx,
            target_idx,
            extract_props(row_idx),
            graph,
            &connection_type,
        ) {
            skipped_count += 1;
            errors.push(format!("Failed to add connection: {}", e));
        }
    }

    // Pass B — vivify the missing endpoints as provisional stub nodes.
    let mut stubs_vivified = 0usize;
    if !missing_sources.is_empty() {
        stubs_vivified += vivify_stubs(graph, &source_type, &missing_sources)?;
    }
    if !missing_targets.is_empty() {
        stubs_vivified += vivify_stubs(graph, &target_type, &missing_targets)?;
    }

    // Pass C — replay the deferred rows now that every endpoint exists.
    if !deferred.is_empty() {
        let lookup2 = CombinedTypeLookup::from_id_indices(
            &graph.id_indices,
            &graph.graph,
            source_type.clone(),
            target_type.clone(),
        )?;
        for (row_idx, source_id, target_id) in deferred {
            let (source_idx, target_idx) = match (
                lookup2.check_source(&source_id),
                lookup2.check_target(&target_id),
            ) {
                (Some(s), Some(t)) => (s, t),
                _ => {
                    // Vivification did not produce the node — count as
                    // a genuine skip (should not happen in practice).
                    skipped_count += 1;
                    continue;
                }
            };
            update_node_titles(
                graph,
                source_idx,
                target_idx,
                row_idx,
                source_title_idx,
                target_title_idx,
                &df_data,
            )?;
            if let Err(e) = batch.add_connection(
                source_idx,
                target_idx,
                extract_props(row_idx),
                graph,
                &connection_type,
            ) {
                skipped_count += 1;
                errors.push(format!("Failed to add connection: {}", e));
            }
        }
    }

    // Report skip reasons — genuine skips only (null ids). Missing
    // endpoints are vivified, not skipped.
    if skipped_null_source > 0 {
        errors.push(format!(
            "Skipped {} rows: null values in source ID field '{}'",
            skipped_null_source, source_id_field
        ));
    }
    if skipped_null_target > 0 {
        errors.push(format!(
            "Skipped {} rows: null values in target ID field '{}'",
            skipped_null_target, target_id_field
        ));
    }

    update_schema_node(
        graph,
        &connection_type,
        lookup.get_source_type(),
        lookup.get_target_type(),
        batch.get_schema_properties(),
    )?;

    // Execute the batch and get the statistics
    let (stats, metrics) = batch.execute(graph, connection_type)?;

    // Invalidate edge-cardinality caches whenever the batch produced
    // edges. Pre-0.9.35 this path didn't invalidate, so a sequence of
    // Cypher CREATE → Python add_connections → planner-cost query
    // could read a stale edge-type-count map; the existing Cypher
    // executor at write.rs:346 invalidated correctly but the bulk
    // Python API did not. Fixing here covers both the new
    // type_connectivity cache (selectivity-aware planning) and the
    // pre-existing edge_type_counts_cache used by reorder_match_clauses.
    if stats.connections_created > 0 {
        graph.invalidate_edge_type_counts_cache();
    }

    // Create and return the operation report
    let mut report = ConnectionOperationReport::new(
        "add_connections".to_string(),
        stats.connections_created,
        skipped_count,
        stats.properties_tracked,
        metrics.processing_time * 1000.0, // Convert to milliseconds
    );
    report.stubs_vivified = stubs_vivified;

    // Add errors if we found any
    if !errors.is_empty() {
        report = report.with_errors(errors);
    }

    Ok(report)
}

/// Auto-vivify missing edge endpoints as provisional stub nodes.
///
/// Each id in `ids` becomes a node of `node_type` carrying only its id
/// (also used as the title) and a `_provisional = true` marker. Routed
/// through `add_nodes` so a stub lands in the same storage (columnar,
/// on the disk/mapped backends) as every other node; `preserve` mode
/// makes a re-vivified id (same id missing as both a source and a
/// target on a same-type edge) a no-op. Returns the count actually
/// created.
fn vivify_stubs(graph: &mut DirGraph, node_type: &str, ids: &[Value]) -> Result<usize, String> {
    let rows: Vec<Vec<Value>> = ids
        .iter()
        .map(|id| vec![id.clone(), Value::Boolean(true)])
        .collect();
    let df =
        DataFrame::from_cypher_rows(vec!["id".to_string(), PROVISIONAL_KEY.to_string()], rows)?;
    let report = add_nodes(
        graph,
        df,
        node_type.to_string(),
        "id".to_string(),
        None,
        Some("preserve".to_string()),
    )?;
    Ok(report.nodes_created)
}

/// DETACH-delete a set of nodes: remove every incident edge, then the
/// nodes, then clean the type / id / property / composite indexes.
/// Shared by the Cypher DETACH DELETE executor and `purge_provisional`.
/// Returns `(nodes_deleted, edges_removed)`.
///
/// Clearing `connection_types` matters on disk graphs: the lazy
/// `has_connection_type` cache would otherwise report a still-live
/// type as gone after a delete.
pub(crate) fn detach_delete_nodes(
    graph: &mut DirGraph,
    nodes_to_delete: &HashSet<NodeIndex>,
) -> (usize, usize) {
    if nodes_to_delete.is_empty() {
        return (0, 0);
    }

    // Remove every incident edge — a self-loop is listed twice, so dedup.
    let mut deleted_edges: HashSet<EdgeIndex> = HashSet::new();
    for &node_idx in nodes_to_delete {
        let incident: Vec<EdgeIndex> = graph
            .graph
            .edges_directed(node_idx, petgraph::Direction::Outgoing)
            .chain(
                graph
                    .graph
                    .edges_directed(node_idx, petgraph::Direction::Incoming),
            )
            .map(|e| e.id())
            .collect();
        for edge_idx in incident {
            if deleted_edges.insert(edge_idx) {
                GraphWrite::remove_edge(&mut graph.graph, edge_idx);
            }
        }
    }
    let edges_removed = deleted_edges.len();
    if edges_removed > 0 {
        graph.invalidate_edge_type_counts_cache();
        graph.connection_types.clear();
    }

    // Affected node types — collected before deletion for index cleanup.
    let mut affected_types: HashSet<String> = HashSet::new();
    for &node_idx in nodes_to_delete {
        if let Some(node) = graph.graph.node_weight(node_idx) {
            affected_types.insert(node.get_node_type_ref(&graph.interner).to_string());
        }
    }

    for &node_idx in nodes_to_delete {
        GraphWrite::remove_node(&mut graph.graph, node_idx);
        graph.timeseries_store.remove(&node_idx.index());
    }

    // Index cleanup — StableDiGraph keeps surviving indices stable.
    for node_type in &affected_types {
        graph
            .type_indices
            .retain_in_type(node_type, |idx| !nodes_to_delete.contains(idx));
        graph.id_indices.remove(node_type);
        let prop_keys: Vec<_> = graph
            .property_indices
            .keys()
            .filter(|(nt, _)| nt == node_type)
            .cloned()
            .collect();
        for key in prop_keys {
            if let Some(value_map) = graph.property_indices.get_mut(&key) {
                for indices in value_map.values_mut() {
                    indices.retain(|idx| !nodes_to_delete.contains(idx));
                }
            }
        }
        let comp_keys: Vec<_> = graph
            .composite_indices
            .keys()
            .filter(|(nt, _)| nt == node_type)
            .cloned()
            .collect();
        for key in comp_keys {
            if let Some(value_map) = graph.composite_indices.get_mut(&key) {
                for indices in value_map.values_mut() {
                    indices.retain(|idx| !nodes_to_delete.contains(idx));
                }
            }
        }
    }

    (nodes_to_delete.len(), edges_removed)
}

/// Delete every node still marked `_provisional` — a stub vivified for
/// an edge but never promoted by a real node row — along with all its
/// incident edges. Returns `(nodes_purged, edges_removed)`.
pub fn purge_provisional_nodes(graph: &mut DirGraph) -> (usize, usize) {
    let provisional_key = graph.interner.get_or_intern(PROVISIONAL_KEY);
    let mut to_delete: HashSet<NodeIndex> = HashSet::new();
    for node_idx in graph.graph.node_indices() {
        if matches!(
            GraphRead::get_node_property(&graph.graph, node_idx, provisional_key),
            Some(Value::Boolean(true))
        ) {
            to_delete.insert(node_idx);
        }
    }
    detach_delete_nodes(graph, &to_delete)
}

fn update_node_titles(
    graph: &mut DirGraph,
    source_idx: NodeIndex,
    target_idx: NodeIndex,
    row_idx: usize,
    source_title_idx: Option<usize>,
    target_title_idx: Option<usize>,
    df_data: &DataFrame,
) -> Result<(), String> {
    if let Some(title_idx) = source_title_idx {
        if let Some(title) = df_data.get_value_by_index(row_idx, title_idx) {
            if let Some(node) = graph.get_node_mut(source_idx) {
                node.title = title;
            }
        }
    }
    if let Some(title_idx) = target_title_idx {
        if let Some(title) = df_data.get_value_by_index(row_idx, title_idx) {
            if let Some(node) = graph.get_node_mut(target_idx) {
                node.title = title;
            }
        }
    }
    Ok(())
}

fn update_schema_node(
    graph: &mut DirGraph,
    connection_type: &str,
    source_type: &str,
    target_type: &str,
    properties: &HashSet<String>,
) -> Result<(), String> {
    if !graph.has_node_type(source_type) {
        return Err(format!(
            "Source type '{}' does not exist in graph",
            source_type
        ));
    }
    if !graph.has_node_type(target_type) {
        return Err(format!(
            "Target type '{}' does not exist in graph",
            target_type
        ));
    }

    // Build property type map — all connection properties default to "Unknown"
    let prop_types: HashMap<String, String> = properties
        .iter()
        .map(|prop| (prop.clone(), "Unknown".to_string()))
        .collect();

    graph.upsert_connection_type_metadata(connection_type, source_type, target_type, prop_types);
    Ok(())
}

pub fn create_connections(
    graph: &mut DirGraph,
    selection: &CurrentSelection,
    connection_type: String,
    conflict_handling: Option<String>,
    copy_properties: Option<HashMap<String, Vec<String>>>, // node_type → prop names to copy onto edge
    source_type_filter: Option<String>,                    // override source level by node type
    target_type_filter: Option<String>,                    // override target level by node type
) -> Result<ConnectionOperationReport, String> {
    let conflict_mode = match conflict_handling.as_deref() {
        Some("replace") => ConflictHandling::Replace,
        Some("skip") => ConflictHandling::Skip,
        Some("preserve") => ConflictHandling::Preserve,
        Some("sum") => ConflictHandling::Sum,
        Some("update") | None => ConflictHandling::Update,
        Some(other) => {
            return Err(format!(
                "Unknown conflict handling mode: '{}'. Valid: 'update' (default), 'replace', 'skip', 'preserve', 'sum'",
                other
            ))
        }
    };

    let level_count = selection.get_level_count();
    if level_count == 0 {
        return Ok(ConnectionOperationReport::new(
            "create_connections".to_string(),
            0,
            0,
            0,
            0.0,
        ));
    }

    // --- Determine which level each node type lives at ---
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

    // --- Resolve source and target levels ---
    let source_level = if let Some(ref st) = source_type_filter {
        *type_to_level.get(st).ok_or_else(|| {
            format!(
                "source_type '{}' not found in traversal chain. Available: {:?}",
                st,
                type_to_level.keys().collect::<Vec<_>>()
            )
        })?
    } else {
        0
    };

    let target_level = if let Some(ref tt) = target_type_filter {
        *type_to_level.get(tt).ok_or_else(|| {
            format!(
                "target_type '{}' not found in traversal chain. Available: {:?}",
                tt,
                type_to_level.keys().collect::<Vec<_>>()
            )
        })?
    } else {
        level_count - 1
    };

    if source_level >= target_level {
        return Err(format!(
            "source level ({}) must be before target level ({})",
            source_level, target_level
        ));
    }

    // --- Iterate target level groups to create edges ---
    // Each group at the target level has (parent, children). For each target node,
    // walk up through group parents to find the source node at source_level.
    // A child can appear in multiple groups (different parents), producing one edge
    // per distinct (source, target) pair.
    let target_level_data = match selection.get_level(target_level) {
        Some(level) if !level.is_empty() => level,
        _ => {
            return Ok(ConnectionOperationReport::new(
                "create_connections".to_string(),
                0,
                0,
                0,
                0.0,
            ));
        }
    };

    let mut batch = ConnectionBatchProcessor::new(target_level_data.node_count());
    batch.set_conflict_mode(conflict_mode);

    let mut skipped = 0;
    let mut errors = Vec::new();
    let mut detected_source_type = None;
    let mut detected_target_type = None;

    // For the common 2-level case (source_level=0, target_level=1), each group's
    // parent IS the source node, so we don't need parent maps at all.
    // For multi-level cases, build reverse parent maps: child → parents (plural).
    let parent_maps: Vec<HashMap<NodeIndex, Vec<NodeIndex>>> = if target_level - source_level > 1 {
        let mut maps: Vec<HashMap<NodeIndex, Vec<NodeIndex>>> = vec![HashMap::new(); level_count];
        for (lvl_idx, pmap) in maps.iter_mut().enumerate().skip(1) {
            if let Some(level) = selection.get_level(lvl_idx) {
                for (parent_opt, children) in level.iter_groups() {
                    if let Some(parent) = parent_opt {
                        for &child in children {
                            pmap.entry(child).or_default().push(*parent);
                        }
                    }
                }
            }
        }
        maps
    } else {
        Vec::new()
    };

    // Helper: walk from a node at `start_level` up to `source_level`, returning
    // all possible source nodes. For a 1-step walk, this is just the immediate parent.
    let walk_to_sources = |start_node: NodeIndex, start_level: usize| -> Vec<NodeIndex> {
        if start_level == source_level {
            return vec![start_node];
        }
        // BFS walk up through parent maps
        let mut current_nodes = vec![start_node];
        for lvl in (source_level + 1..=start_level).rev() {
            let mut next_nodes = Vec::new();
            for node in &current_nodes {
                if let Some(parents) = parent_maps[lvl].get(node) {
                    next_nodes.extend(parents);
                }
            }
            if next_nodes.is_empty() {
                return Vec::new(); // Orphan — no path to source
            }
            current_nodes = next_nodes;
        }
        current_nodes
    };

    for (parent_opt, targets) in target_level_data.iter_groups() {
        let Some(parent_idx) = parent_opt else {
            // Root-level targets have no parent — skip
            skipped += targets.len();
            continue;
        };

        // Resolve the source node(s) for this group's parent
        let source_nodes = if target_level - source_level == 1 {
            // Direct parent IS the source
            vec![*parent_idx]
        } else {
            walk_to_sources(*parent_idx, target_level - 1)
        };

        if source_nodes.is_empty() {
            skipped += targets.len();
            continue;
        }

        for &target_idx in targets {
            if detected_target_type.is_none() {
                if let Some(node) = graph.get_node(target_idx) {
                    detected_target_type = Some(node.node_type_str(&graph.interner).to_string());
                }
            }

            for &source_idx in &source_nodes {
                if detected_source_type.is_none() {
                    if let Some(node) = graph.get_node(source_idx) {
                        detected_source_type =
                            Some(node.node_type_str(&graph.interner).to_string());
                    }
                }

                // Collect properties from nodes in the chain (source → ... → target)
                let edge_props = if let Some(ref prop_spec) = copy_properties {
                    let mut props = HashMap::new();
                    // Add source and target node properties
                    for &node_idx in &[source_idx, target_idx] {
                        if let Some(node) = graph.graph.node_weight(node_idx) {
                            let nt = node.node_type_str(&graph.interner);
                            if let Some(requested_props) = prop_spec.get(nt) {
                                if requested_props.is_empty() {
                                    for (k, v) in node.property_iter(&graph.interner) {
                                        props.insert(k.to_string(), v.clone());
                                    }
                                } else {
                                    for prop_name in requested_props {
                                        if let Some(val) = node.get_property(prop_name) {
                                            props.insert(prop_name.clone(), val.into_owned());
                                        }
                                    }
                                }
                            }
                        }
                    }
                    props
                } else {
                    HashMap::new()
                };

                if let Err(e) = batch.add_connection(
                    source_idx,
                    target_idx,
                    edge_props,
                    graph,
                    &connection_type,
                ) {
                    skipped += 1;
                    errors.push(format!("Failed to add connection: {}", e));
                    continue;
                }
            }
        }
    }

    if let (Some(source), Some(target)) = (detected_source_type, detected_target_type) {
        update_schema_node(
            graph,
            &connection_type,
            &source,
            &target,
            batch.get_schema_properties(),
        )?;
    }

    let (stats, metrics) = batch.execute(graph, connection_type)?;

    let mut report = ConnectionOperationReport::new(
        "create_connections".to_string(),
        stats.connections_created,
        skipped,
        stats.properties_tracked,
        metrics.processing_time * 1000.0,
    );

    if !errors.is_empty() {
        report = report.with_errors(errors);
    }

    Ok(report)
}

pub fn update_node_properties(
    graph: &mut DirGraph,
    nodes: &[(Option<NodeIndex>, Value)],
    property: &str,
) -> Result<NodeOperationReport, String> {
    if nodes.is_empty() {
        return Err("No nodes to update".to_string());
    }

    // Track start time for the report
    let start_time = std::time::Instant::now();

    // Create property string once
    let property_string = property.to_string();

    // Track errors
    let mut errors = Vec::new();

    // Step 1: Collect information about node types and check if schema update is needed
    let mut node_types = HashMap::new();
    let mut first_value_type = None;
    let mut skipped_count = 0;

    for (node_idx_opt, value) in nodes {
        if let Some(node_idx) = node_idx_opt {
            if let Some(node) = graph.get_node(*node_idx) {
                // Track node type and count for each node
                *node_types
                    .entry(node.node_type_str(&graph.interner).to_string())
                    .or_insert(0) += 1;

                // Capture type of first value for schema
                if first_value_type.is_none() {
                    first_value_type = Some(match value {
                        Value::Int64(_) => "Int64",
                        Value::Float64(_) => "Float64",
                        Value::String(_) => "String",
                        Value::UniqueId(_) => "UniqueId",
                        _ => "Unknown",
                    });
                }
            } else {
                skipped_count += 1;
                errors.push(format!("Node index {:?} not found in graph", node_idx));
            }
        } else {
            skipped_count += 1;
        }
    }

    // Step 2: Update node type metadata for each affected node type
    let type_string = first_value_type
        .map(|t| t.to_string())
        .unwrap_or_else(|| "Calculated".to_string());

    for (node_type, _count) in node_types.iter() {
        // Check for type mismatch with existing metadata
        if let Some(existing_meta) = graph.get_node_type_metadata(node_type) {
            if let Some(existing_type) = existing_meta.get(&property_string) {
                if existing_type != &type_string {
                    errors.push(format!(
                        "Type mismatch for property '{}': existing schema has '{}', but data has '{}'",
                        property_string, existing_type, type_string
                    ));
                }
            }
        }

        let mut new_prop_types = HashMap::new();
        new_prop_types.insert(property_string.clone(), type_string.clone());
        graph.upsert_node_type_metadata(node_type, new_prop_types);
    }

    // Step 3: Prepare batch updates for nodes
    let batch_size = nodes.len();
    let mut batch = BatchProcessor::new(batch_size);

    for (node_idx_opt, value) in nodes {
        if let Some(node_idx) = node_idx_opt {
            // Only add valid nodes to batch
            if graph.graph.node_weight(*node_idx).is_some() {
                let mut properties = HashMap::new();
                properties.insert(property_string.clone(), value.clone());

                // Create update action
                let action = NodeAction::Update {
                    node_idx: *node_idx,
                    title: None, // Don't update title
                    properties,
                    conflict_mode: ConflictHandling::Update,
                };

                if let Err(e) = batch.add_action(action, graph) {
                    errors.push(format!("Failed to update node property: {}", e));
                    skipped_count += 1;
                }
            } else {
                skipped_count += 1;
                errors.push(format!("Node index {:?} is out of bounds", node_idx));
            }
        } else {
            skipped_count += 1;
        }
    }

    // Step 4: Execute batch update
    let (stats, _metrics) = match batch.execute(graph) {
        Ok(result) => result,
        Err(e) => {
            errors.push(format!("Failed to execute batch update: {}", e));
            return Err(format!("Failed to execute batch update: {}", e));
        }
    };

    if stats.updates == 0 && errors.is_empty() {
        errors.push("No nodes were updated".to_string());
    }

    // Calculate elapsed time
    let elapsed_ms = start_time.elapsed().as_secs_f64() * 1000.0;

    // Create and return the operation report
    let mut report = NodeOperationReport::new(
        "update_node_properties".to_string(),
        0, // We don't create nodes in this function
        stats.updates,
        skipped_count,
        elapsed_ms,
    );

    // Add errors if we found any
    if !errors.is_empty() {
        report = report.with_errors(errors);
    }

    Ok(report)
}

// ── add_properties ──────────────────────────────────────────────────────────

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

/// Report returned by add_properties().
pub struct AddPropertiesReport {
    pub nodes_updated: usize,
    pub properties_set: usize,
}

/// Enriches the leaf (most recent) level nodes by copying, renaming, aggregating,
/// or computing properties from ancestor nodes in the traversal hierarchy.
pub fn add_properties(
    graph: &mut DirGraph,
    selection: &CurrentSelection,
    property_spec: HashMap<String, PropertySpec>,
) -> Result<AddPropertiesReport, String> {
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

    // Apply updates
    let mut nodes_updated = 0;
    let mut properties_set = 0;
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
        }
    }

    Ok(AddPropertiesReport {
        nodes_updated,
        properties_set,
    })
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

    let mut nodes_updated = 0;
    let mut properties_set = 0;

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
        }
    }

    Ok(AddPropertiesReport {
        nodes_updated,
        properties_set,
    })
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
