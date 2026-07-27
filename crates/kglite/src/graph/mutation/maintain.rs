// src/graph/maintain.rs
use crate::datatypes::{DataFrame, Value};
use crate::graph::constraints::{ConstraintViolation, UniqueConstraintKey};
use crate::graph::introspection::reporting::{ConnectionOperationReport, NodeOperationReport};
use crate::graph::mutation::batch::{
    BatchProcessor, ConflictHandling, ConnectionBatchProcessor, NodeAction,
};
use crate::graph::schema::{
    CompositeValue, CurrentSelection, DirGraph, InternedKey, TypeSchema, PROVISIONAL_KEY,
    RESERVED_PROVENANCE_KEYS,
};
use crate::graph::storage::lookups::{CombinedTypeLookup, TypeLookup};
use crate::graph::storage::undo::BucketId;
use crate::graph::storage::{GraphRead, GraphWrite};
use petgraph::graph::{EdgeIndex, NodeIndex};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

/// Report returned by `add_properties()`.
///
/// Lives here rather than beside `add_properties` itself: it is the return type
/// of the public `kglite::api::mutation::add_properties`, and the pinned Rust API
/// baseline records it at this canonical path.
pub struct AddPropertiesReport {
    pub nodes_updated: usize,
    pub properties_set: usize,
}

/// Column lookup for the bulk path's constraint gates: user-facing property
/// name → column index, resolved once per call so the per-row read is a HashMap
/// hit instead of repeated `get_column_index` string scans.
///
/// The identity and title columns are handled by the caller's closure (they may
/// be named `npdid` / `prospect_name` rather than `id` / `title`), so this only
/// covers ordinary property columns.
struct ConstraintColumns {
    by_name: HashMap<String, usize>,
}

/// One input row, as the constraint gate needs to see it: where to read
/// ordinary columns from, plus the identity values and the column names they
/// arrived under.
///
/// A constraint names the property the *user* queries by, which for the identity
/// and title columns may be the original column name (`npdid`) rather than the
/// canonical `id` / `title`.
struct GateRow<'a> {
    df_data: &'a DataFrame,
    row_idx: usize,
    id: &'a Value,
    title: &'a Value,
    id_field: &'a str,
    title_field: &'a str,
}

impl ConstraintColumns {
    /// The column lookup a batch of `node_type` rows needs, or `None` when the
    /// type declares no constraint at all.
    ///
    /// UNIQUE / NOT NULL enforcement has to happen on the bulk path too: a
    /// constraint the batch engine bypassed would be theatre, since `add_nodes`
    /// is how blueprints, `from_records`, OKF, WAL replay and `extend_graph` all
    /// reach storage. Returning `None` for an unconstrained type is what keeps
    /// the common path free — one `is_empty` and one `Option` check per call
    /// rather than per row.
    fn for_batch(graph: &mut DirGraph, node_type: &str, df_data: &DataFrame) -> Option<Self> {
        // Batch entry point: no violation parked by an earlier load may be
        // attributed to this one.
        graph.clear_pending_constraint_violation();
        (graph.has_unique_constraints() || graph.has_required_fields(node_type))
            .then(|| Self::new(df_data))
    }

    /// [`Self::gate_row`] for callers on the `Result<_, String>` channel: parks
    /// the structured violation so the binding raises
    /// `ConstraintViolationError` instead of a generic argument error, and
    /// returns the prose the caller would have produced itself.
    fn gate_row_parked(
        &self,
        graph: &mut DirGraph,
        node_type: &str,
        row: GateRow<'_>,
        existing_idx: Option<NodeIndex>,
        batch_claims: &mut HashSet<(UniqueConstraintKey, CompositeValue)>,
    ) -> Result<(), String> {
        // Bound before reporting: recording needs a second mutable borrow.
        let gated = self.gate_row(graph, node_type, row, existing_idx, batch_claims);
        match gated {
            Ok(()) => Ok(()),
            Err(violation) => Err(graph.record_constraint_violation(violation)),
        }
    }

    fn new(df_data: &DataFrame) -> Self {
        let by_name = df_data
            .get_column_names()
            .into_iter()
            .filter_map(|name| df_data.get_column_index(&name).map(|idx| (name, idx)))
            .collect();
        Self { by_name }
    }

    /// The row's value for `property`, or `None` when the column is absent or
    /// the cell is null — both of which a constraint treats identically.
    fn read(&self, df_data: &DataFrame, row_idx: usize, property: &str) -> Option<Value> {
        let column = *self.by_name.get(property)?;
        match df_data.get_value_by_index(row_idx, column) {
            Some(Value::Null) | None => None,
            Some(value) => Some(value),
        }
    }

    /// Gate one input row against the declared NOT NULL and UNIQUE constraints.
    ///
    /// Called before the row is queued onto the batch, so a violation aborts the
    /// whole `add_nodes` call with the graph untouched — no rollback needed and
    /// no half-applied load. `batch_claims` carries the tuples earlier rows of
    /// this same input already claimed, so a repeat *within* one batch is
    /// rejected even when neither row conflicts with stored data.
    fn gate_row(
        &self,
        graph: &DirGraph,
        node_type: &str,
        row: GateRow<'_>,
        existing_idx: Option<NodeIndex>,
        batch_claims: &mut HashSet<(UniqueConstraintKey, CompositeValue)>,
    ) -> Result<(), ConstraintViolation> {
        // Both identity spellings are accepted, and identity wins over a
        // same-named ordinary column — matching `DirGraph::resolve_alias` on the
        // read side.
        let read = |property: &str| -> Option<Value> {
            if property == "id" || property == row.id_field {
                return (!matches!(row.id, Value::Null)).then(|| row.id.clone());
            }
            if property == "title" || property == row.title_field {
                return (!matches!(row.title, Value::Null)).then(|| row.title.clone());
            }
            self.read(row.df_data, row.row_idx, property)
        };
        // Every failure this gate can report is a declared-constraint
        // violation, so it returns the structured value and lets `add_nodes`
        // decide how to render it.
        graph.check_required_fields(node_type, read)?;
        let claims = graph.unique_claims(node_type, read);
        graph.check_unique_claims(&claims, existing_idx)?;
        for claim in &claims {
            if !batch_claims.insert((claim.key.clone(), claim.value.clone())) {
                return Err(graph.unique_batch_conflict(claim));
            }
        }
        Ok(())
    }
}
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
        // Names come from get_column_names(), so the lookup always succeeds.
        if let Some(col_type) = df_data.get_column_type(&col_name) {
            types.insert(col_name.clone(), col_type.to_string());
        }
    }
    types
}

fn preflight_interner_names<'a>(
    graph: &DirGraph,
    names: impl IntoIterator<Item = &'a str>,
) -> Result<(), String> {
    graph
        .interner
        .validate_names(names)
        .map(|_| ())
        .map_err(|e| e.to_string())
}

/// Append a human-readable reason for every row `add_nodes` dropped. Both cases
/// are input problems the caller can act on, so they are reported rather than
/// silently tolerated — and the parse failure names the `column_types` override
/// that fixes it.
fn describe_skipped_rows(
    errors: &mut Vec<String>,
    skipped_null_id: usize,
    skipped_parse_fail: usize,
    unique_id_field: &str,
) {
    if skipped_null_id > 0 {
        errors.push(format!(
            "Skipped {skipped_null_id} rows: null values in ID field '{unique_id_field}'"
        ));
    }
    if skipped_parse_fail > 0 {
        errors.push(format!(
            "Skipped {skipped_parse_fail} rows: could not parse ID field \
             '{unique_id_field}'. If IDs are strings, pass \
             column_types={{'{unique_id_field}': 'string'}}"
        ));
    }
}

/// Everything about an `add_nodes` call that does not vary per row, so the row
/// loop reads as "gate the row, build its action, queue it" rather than
/// re-deriving the same context each iteration.
struct RowBuilder<'a> {
    node_type: &'a str,
    interned_columns: &'a [(InternedKey, usize)],
    provenance_stamps: &'a [(InternedKey, Value)],
    property_count: usize,
    should_update_title: bool,
    conflict_mode: ConflictHandling,
}

impl RowBuilder<'_> {
    /// One row's properties with keys already interned — no HashMap allocation
    /// and no per-row string cloning. Null cells are dropped rather than
    /// stored; provenance stamps are applied last so they win over a column of
    /// the same name.
    fn properties(&self, df_data: &DataFrame, row_idx: usize) -> Vec<(InternedKey, Value)> {
        let mut properties = Vec::with_capacity(self.property_count);
        for (interned_key, col_idx) in self.interned_columns {
            let value = df_data
                .get_value_by_index(row_idx, *col_idx)
                .unwrap_or(Value::Null);
            if !matches!(value, Value::Null) {
                properties.push((*interned_key, value));
            }
        }
        for (key, value) in self.provenance_stamps {
            properties.retain(|(interned, _)| interned != key);
            properties.push((*key, value.clone()));
        }
        properties
    }

    /// The batch action for one row: an update when the id already resolves to a
    /// node, a create otherwise. The update path resolves keys back to strings
    /// because it goes through the HashMap-based `NodeAction::Update` — less
    /// frequent than create, which keeps its interned keys all the way down.
    fn action(
        &self,
        graph: &DirGraph,
        df_data: &DataFrame,
        row_idx: usize,
        id: Value,
        title: Value,
        existing_idx: Option<NodeIndex>,
    ) -> NodeAction {
        let properties_interned = self.properties(df_data, row_idx);
        match existing_idx {
            Some(node_idx) => {
                let mut properties = HashMap::with_capacity(properties_interned.len());
                for (ik, v) in properties_interned {
                    properties.insert(graph.interner.resolve(ik).to_string(), v);
                }
                NodeAction::Update {
                    node_idx,
                    title: self.should_update_title.then_some(title),
                    properties,
                    conflict_mode: self.conflict_mode,
                }
            }
            None => NodeAction::CreateInterned {
                node_type: self.node_type.to_string(),
                id,
                title,
                properties: properties_interned,
            },
        }
    }
}

pub fn add_nodes(
    graph: &mut DirGraph,
    df_data: DataFrame,
    node_type: String,
    unique_id_field: String,
    node_title_field: Option<String>,
    conflict_handling: Option<String>,
) -> Result<NodeOperationReport, String> {
    let _arena_guard = graph.graph.begin_query(); // disk arena guard (owned; no-op on memory/mapped)
    let mut interned_names = vec![node_type.as_str(), PROVISIONAL_KEY];
    interned_names.extend(RESERVED_PROVENANCE_KEYS.iter().copied());
    let column_names = df_data.get_column_names();
    interned_names.extend(column_names.iter().map(String::as_str));
    preflight_interner_names(graph, interned_names)?;
    graph
        .prepare_disk_mutation()
        .map_err(|e| format!("disk mutation lease failed: {e}"))?;
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
    let mut schema_keys: Vec<InternedKey> = property_columns
        .iter()
        .map(|(col_name, _)| graph.interner.get_or_intern(col_name))
        .collect();
    // Register every active reserved key so compact and columnar stores can
    // persist the complete engine-owned provenance stamp.
    let provenance_stamps: Vec<(InternedKey, Value)> = if graph.auto_timestamp_for(&node_type) {
        graph
            .provenance_props()
            .into_iter()
            .map(|(name, value)| (graph.interner.get_or_intern(name), value))
            .collect()
    } else {
        Vec::new()
    };
    schema_keys.extend(provenance_stamps.iter().map(|(key, _)| *key));
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
    // One clock read per call; every row receives the same complete stamp.
    let mut batch = BatchProcessor::new(df_data.row_count());
    let mut skipped_count = 0;
    let mut skipped_null_id = 0;
    let mut skipped_parse_fail = 0;

    // For a declared-PRIMARY-KEY type, a within-batch duplicate id is a data
    // error (the id-index would silently collapse it into a hidden duplicate),
    // so reject it — consistent with Cypher CREATE's reject-on-dup. The
    // conflict-handling modes still apply to duplicates vs. the *existing*
    // graph (add_nodes is the upsert path, like MERGE); this only guards
    // repeats *within the same input batch*. Gated on a declared PK, so the
    // common bulk path allocates nothing.
    let pk_enforced = graph.primary_key_for(&node_type).is_some();
    let mut seen_pk_ids: std::collections::HashSet<Value> = if pk_enforced {
        std::collections::HashSet::with_capacity(df_data.row_count())
    } else {
        std::collections::HashSet::new()
    };

    let constraint_columns = ConstraintColumns::for_batch(graph, &node_type, &df_data);
    // Tuples claimed by earlier rows of this same batch. A repeat inside one
    // input is rejected even when neither row conflicts with the stored graph —
    // otherwise the post-load index rebuild would silently keep one row and drop
    // the other.
    let row_builder = RowBuilder {
        node_type: &node_type,
        interned_columns: &interned_columns,
        provenance_stamps: &provenance_stamps,
        property_count,
        should_update_title,
        conflict_mode,
    };

    let mut batch_claims: std::collections::HashSet<(UniqueConstraintKey, CompositeValue)> =
        std::collections::HashSet::new();

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

        if pk_enforced && !seen_pk_ids.insert(id.clone()) {
            return Err(format!(
                "duplicate primary key: node type '{node_type}' declares a primary key but \
                 the input has more than one row with id {id}. Deduplicate the input before \
                 add_nodes, or drop the primary-key declaration."
            ));
        }

        let title = df_data
            .get_value_by_index(row_idx, title_idx)
            .unwrap_or(Value::Null);

        // Constraint gates. Run here, before the row is queued onto the batch,
        // so a violation aborts the whole call with the graph untouched — no
        // rollback needed and no half-applied load.
        let existing_idx = type_lookup.check_uid(&id);
        if let Some(columns) = &constraint_columns {
            columns.gate_row_parked(
                graph,
                &node_type,
                GateRow {
                    df_data: &df_data,
                    row_idx,
                    id: &id,
                    title: &title,
                    id_field: &unique_id_field,
                    title_field: &title_field,
                },
                existing_idx,
                &mut batch_claims,
            )?;
        }

        let action = row_builder.action(graph, &df_data, row_idx, id, title, existing_idx);
        batch.add_action(action, graph)?;
    }

    describe_skipped_rows(
        &mut errors,
        skipped_null_id,
        skipped_parse_fail,
        &unique_id_field,
    );

    // Execute the batch and get the statistics
    let (stats, metrics) = batch.execute(graph)?;

    // Rebuild the type's id_index. The batch added rows to type_indices but
    // doesn't touch id_indices, so any pre-existing entry is now stale.
    // Rebuild it eagerly (rather than only invalidating) so the read path is
    // O(1): `lookup_by_id_readonly` — used by `MATCH (n {id:X})` and the
    // `MERGE` match — does NOT build the index, it falls back to an O(n)
    // linear scan when the index is absent. Pre-fix the index stayed absent
    // after add_nodes, so every id-equality read scanned (issue #20: lookups
    // were O(node position), e.g. 26µs for a high-id node on 30k rows). The
    // build is O(nodes-of-type), matching the cost the load path already pays
    // via build_id_index.
    graph.id_indices.remove(&node_type);
    graph.build_id_index(&node_type);

    // Same staleness hazard for the *secondary* indexes: the batch path skips
    // the per-write incremental maintenance the Cypher executor runs, and
    // `try_index_lookup` trusts `property_indices` unconditionally — so an
    // index built before this call would silently hide every row we just
    // loaded. Rebuild the covering indexes once (no-op when the type has none).
    graph.refresh_indexes_for_type(&node_type);

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

    graph.bump_version();
    Ok(report)
}

/// A single edge to bulk-create, addressed by stable node id + type —
/// the binding-friendly, DataFrame-free counterpart of an
/// [`add_connections`] row.
#[derive(Debug, Clone)]
pub struct EdgeSpec {
    pub source_type: String,
    pub source_id: Value,
    pub target_type: String,
    pub target_id: Value,
    pub edge_type: String,
    pub properties: HashMap<String, Value>,
}

/// Outcome of [`add_edges_from_specs`].
#[derive(Debug, Default, Clone)]
pub struct EdgeSpecReport {
    /// Edges the batch engine actually created.
    pub connections_created: usize,
    /// Edges skipped because a source or target id had no node of its
    /// declared type. Unlike [`add_connections`], this primitive does NOT
    /// vivify stub endpoints — endpoints must already exist.
    pub skipped_missing_endpoint: usize,
}

/// Bulk-create edges from explicit specs, addressed by stable node id +
/// type. The DataFrame-free sibling of [`add_connections`]: it drives the
/// *same* engine (`CombinedTypeLookup` + `ConnectionBatchProcessor`) but
/// takes a spec list instead of a [`DataFrame`] — the path the C ABI (and
/// future Go / JS / JVM bindings, which can't cheaply build a DataFrame)
/// use, plus any caller that already has edges as records. (That
/// `DataFrame` is kglite's own columnar container in
/// `crate::datatypes::values`; kglite does not depend on polars.)
///
/// Specs are grouped by `(source_type, target_type, edge_type)`; each
/// group gets one type lookup and one batch, mirroring `add_connections`.
/// Endpoints must already exist — an edge whose source or target id isn't
/// found for its declared type is counted in `skipped_missing_endpoint`
/// and skipped (no stub vivification).
pub fn add_edges_from_specs(
    graph: &mut DirGraph,
    specs: Vec<EdgeSpec>,
) -> Result<EdgeSpecReport, String> {
    let _arena_guard = graph.graph.begin_query(); // disk arena guard (owned; no-op on memory/mapped)
    use std::collections::BTreeMap;
    let mut interned_names = Vec::from(RESERVED_PROVENANCE_KEYS);
    for spec in &specs {
        interned_names.extend([
            spec.source_type.as_str(),
            spec.target_type.as_str(),
            spec.edge_type.as_str(),
        ]);
        interned_names.extend(spec.properties.keys().map(String::as_str));
    }
    preflight_interner_names(graph, interned_names)?;
    graph
        .prepare_disk_mutation()
        .map_err(|e| format!("disk mutation lease failed: {e}"))?;

    // Group by (source_type, target_type, edge_type) for deterministic,
    // one-lookup-one-batch-per-group processing.
    type EdgeRows = Vec<(Value, Value, HashMap<String, Value>)>;
    let mut groups: BTreeMap<(String, String, String), EdgeRows> = BTreeMap::new();
    for spec in specs {
        groups
            .entry((spec.source_type, spec.target_type, spec.edge_type))
            .or_default()
            .push((spec.source_id, spec.target_id, spec.properties));
    }

    let mut report = EdgeSpecReport::default();
    // The id→node lookup depends only on (source_type, target_type), not the
    // edge type, and creating edges never invalidates it (no nodes added). So
    // cache it per node-type pair instead of rebuilding the full type scan for
    // every edge type over the same pair (e.g. Person KNOWS/FOLLOWS/BLOCKS
    // Person was K identical materializations; now one).
    let mut lookup_cache: HashMap<(String, String), CombinedTypeLookup> = HashMap::new();
    for ((source_type, target_type, edge_type), edges) in groups {
        let pair = (source_type.clone(), target_type.clone());
        if !lookup_cache.contains_key(&pair) {
            let lookup = CombinedTypeLookup::from_id_indices(
                &graph.id_indices,
                &graph.graph,
                source_type.clone(),
                target_type.clone(),
            )?;
            lookup_cache.insert(pair.clone(), lookup);
        }
        let lookup = &lookup_cache[&pair];
        let mut batch = ConnectionBatchProcessor::new(edges.len());
        // Same initial-load fast path as add_connections: skip per-edge
        // existence checks when this connection type has no edges yet.
        let is_initial_load = !graph.connection_type_metadata.contains_key(&edge_type);
        batch.set_skip_existence_check(is_initial_load);

        for (source_id, target_id, props) in edges {
            match (
                lookup.check_source(&source_id),
                lookup.check_target(&target_id),
            ) {
                (Some(src_idx), Some(tgt_idx)) => {
                    batch.add_connection(src_idx, tgt_idx, props, graph, &edge_type)?;
                }
                _ => report.skipped_missing_endpoint += 1,
            }
        }

        // Register the connection in the schema before consuming the batch.
        update_schema_node(
            graph,
            &edge_type,
            &source_type,
            &target_type,
            batch.get_schema_properties(),
        )?;

        let (stats, _metrics) = batch.execute(graph, edge_type)?;
        report.connections_created += stats.connections_created;
    }
    graph.bump_version();
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
    let _arena_guard = graph.graph.begin_query(); // disk arena guard (owned; no-op on memory/mapped)
    let column_names = df_data.get_column_names();
    let mut interned_names = vec![
        connection_type.as_str(),
        source_type.as_str(),
        target_type.as_str(),
        PROVISIONAL_KEY,
    ];
    interned_names.extend(RESERVED_PROVENANCE_KEYS.iter().copied());
    interned_names.extend(column_names.iter().map(String::as_str));
    preflight_interner_names(graph, interned_names)?;
    graph
        .prepare_disk_mutation()
        .map_err(|e| format!("disk mutation lease failed: {e}"))?;
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

    graph.bump_version();
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

/// Journal every inverted-index eviction this delete is about to perform,
/// with the *position* each doomed member occupies.
///
/// Position and not merely membership, because bucket order is the scan order
/// an un-`ORDER BY`'d `MATCH` returns: `type_indices` drives a label scan and
/// the user-index buckets are handed straight to the matcher. Must be called
/// while the doomed indices are still in their buckets. One `Option` check
/// when no checkpoint is open, which is every non-mutating call.
fn journal_bucket_evictions(
    graph: &mut DirGraph,
    affected_types: &HashSet<String>,
    nodes_to_delete: &HashSet<NodeIndex>,
) {
    if graph.graph.undo_journal_mut().is_none() {
        return;
    }
    let mut evictions: Vec<(BucketId, Vec<NodeIndex>)> = Vec::new();
    for node_type in affected_types {
        if let Some(members) = graph.type_indices.get(node_type) {
            evictions.push((BucketId::NodeType(node_type.clone()), members.to_vec()));
        }
        // User-created indexes, same treatment. Only buckets that actually
        // hold a doomed node are captured, so the cost tracks the deletion
        // rather than the size of the index.
        for (key, value_map) in &graph.property_indices {
            if &key.0 != node_type {
                continue;
            }
            for (value, members) in value_map {
                if members.iter().any(|idx| nodes_to_delete.contains(idx)) {
                    evictions.push((
                        BucketId::PropertyValue {
                            key: key.clone(),
                            value: value.clone(),
                        },
                        members.clone(),
                    ));
                }
            }
        }
        for (key, btree) in &graph.range_indices {
            if &key.0 != node_type {
                continue;
            }
            for (value, members) in btree {
                if members.iter().any(|idx| nodes_to_delete.contains(idx)) {
                    evictions.push((
                        BucketId::RangeValue {
                            key: key.clone(),
                            value: value.clone(),
                        },
                        members.clone(),
                    ));
                }
            }
        }
        for (key, comp_map) in &graph.composite_indices {
            if &key.0 != node_type {
                continue;
            }
            for (value, members) in comp_map {
                if members.iter().any(|idx| nodes_to_delete.contains(idx)) {
                    evictions.push((
                        BucketId::CompositeTuple {
                            key: key.clone(),
                            value: value.clone(),
                        },
                        members.clone(),
                    ));
                }
            }
        }
    }
    if graph.has_secondary_labels {
        for (label, members) in &graph.secondary_label_index {
            evictions.push((BucketId::SecondaryLabel(*label), members.clone()));
        }
    }
    if let Some(journal) = graph.graph.undo_journal_mut() {
        for (bucket, members) in &evictions {
            journal.note_bucket_retain(bucket, members.iter().copied(), nodes_to_delete);
        }
    }
}

/// DETACH-delete a set of nodes: remove every incident edge, then the
/// nodes, then clean the type / id / property / composite / secondary-label
/// indexes. Shared by the Cypher DETACH DELETE executor and
/// `purge_provisional`. Returns `(nodes_deleted, edges_removed)`.
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
        // Scope the arena guard to the read: edge iteration materializes
        // into the disk backend's query arena, which must run under a
        // DiskQueryGuard (arena protocol in disk/graph.rs, enforced by a
        // debug assert). The guard is dropped before the `&mut`
        // remove_edge calls below.
        let incident: Vec<EdgeIndex> = {
            let _guard = graph.graph.begin_query();
            graph
                .graph
                .edges_directed(node_idx, petgraph::Direction::Outgoing)
                .chain(
                    graph
                        .graph
                        .edges_directed(node_idx, petgraph::Direction::Incoming),
                )
                .map(|e| e.id())
                .collect()
        };
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
    // Same guard scoping as the edge collection above (node_weight
    // materializes on the disk backend).
    let mut affected_types: HashSet<String> = HashSet::new();
    // The doomed nodes' ids, per type, so the id index can be edited in place
    // instead of being dropped and rebuilt by the next lookup. Read here
    // because after `remove_node` below the weights are gone.
    let mut doomed_ids: HashMap<String, Vec<(Value, NodeIndex)>> = HashMap::new();
    {
        let _guard = graph.graph.begin_query();
        for &node_idx in nodes_to_delete {
            if let Some(node) = graph.graph.node_weight(node_idx) {
                let node_type = node.get_node_type_ref(&graph.interner).to_string();
                let node_id = node.id().into_owned();
                doomed_ids
                    .entry(node_type.clone())
                    .or_default()
                    .push((node_id, node_idx));
                affected_types.insert(node_type);
            }
        }
    }

    // Whether each type's id index may be edited in place rather than dropped.
    //
    // In-place eviction removes exactly the ids handed to it; a rebuild
    // re-derives the map from the surviving nodes. The two agree only while
    // ids are unique within the type. If two live nodes share an id the index
    // holds one of them, and deleting that one must leave the *other*
    // reachable by id — which only a rebuild achieves. Duplicates are
    // detectable in O(1): the index was built with one entry per node of the
    // type, so a shorter index is exactly the signature of collapsed
    // duplicates. Unequal (or unbuilt, or base-resident) falls back to the
    // whole-type invalidation this path has always done.
    //
    // Note this differs from the create path, which may maintain the index
    // incrementally unconditionally: inserting a duplicate collapses it the
    // same way a rebuild would, so the two stay in agreement there.
    let evictable: HashSet<String> = affected_types
        .iter()
        .filter(|node_type| {
            let indexed = graph.id_indices.overlay_len(node_type);
            let live = graph.type_indices.get(node_type).map(|m| m.len());
            matches!((indexed, live), (Some(i), Some(l)) if i == l)
        })
        .cloned()
        .collect();

    for &node_idx in nodes_to_delete {
        GraphWrite::remove_node(&mut graph.graph, node_idx);
        if let Some(prior) = graph.timeseries_store.remove(&node_idx.index()) {
            // Statement-rollback capture: `timeseries_store` is O(V) and so is
            // deliberately not part of the checkpoint's schema clone; the
            // journal carries the dropped value instead.
            if let Some(journal) = graph.graph.undo_journal_mut() {
                journal.note_timeseries_removed(node_idx.index(), prior);
            }
        }
    }

    // Statement-rollback capture, before the `retain` sweeps below strip the
    // doomed members out of the buckets.
    journal_bucket_evictions(graph, &affected_types, nodes_to_delete);

    // Index cleanup — StableDiGraph keeps surviving indices stable.
    for node_type in &affected_types {
        graph
            .type_indices
            .retain_in_type(node_type, |idx| !nodes_to_delete.contains(idx));
        match doomed_ids.get(node_type) {
            Some(entries) if evictable.contains(node_type) => {
                graph.id_indices.evict_entries(node_type, entries);
            }
            _ => {
                graph.id_indices.remove(node_type);
            }
        }
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
        // The B-tree range index was omitted from this cleanup, so a deleted
        // node stayed in its value bucket and `lookup_range` — the candidate
        // source for `WHERE n.prop > x` on an indexed property, in both the
        // matcher and the fluent filter path — kept handing out tombstoned
        // NodeIndexes. Same shape as the property/composite eviction above.
        let range_keys: Vec<_> = graph
            .range_indices
            .keys()
            .filter(|(nt, _)| nt == node_type)
            .cloned()
            .collect();
        for key in range_keys {
            if let Some(value_map) = graph.range_indices.get_mut(&key) {
                for indices in value_map.values_mut() {
                    indices.retain(|idx| !nodes_to_delete.contains(idx));
                }
                value_map.retain(|_, indices| !indices.is_empty());
            }
        }
        // A deleted node must give up its UNIQUE tuples, or the value stays
        // reserved forever and re-inserting it is rejected.
        graph.evict_unique_claims_for_nodes(node_type, nodes_to_delete);
    }

    // Secondary-label index is keyed by label (not primary type), so a
    // deleted node may sit in any bucket — evict outside the per-type loop.
    // Without this, the StableDiGraph keeps the deleted NodeIndex live in
    // the index, so `MATCH (n:SecLabel) RETURN count(n)` (and the load path)
    // would over-count tombstoned nodes.
    if graph.has_secondary_labels {
        graph.secondary_label_index.retain(|_, bucket| {
            bucket.retain(|idx| !nodes_to_delete.contains(idx));
            !bucket.is_empty()
        });
        if graph.secondary_label_index.is_empty() {
            graph.has_secondary_labels = false;
        }
    }

    (nodes_to_delete.len(), edges_removed)
}

/// Replace the `connection_type` edges of the source nodes named in
/// `df_data`, then add the edges the DataFrame describes.
///
/// **Per-source semantics.** Only edges that are (a) outgoing from a
/// source node *present in `df_data`* and (b) of *this* connection type
/// are removed. Edges from untouched sources, and edges of other types
/// from the same sources, survive. This makes a re-sync idempotent —
/// "set the current MENTIONS of exactly these documents to this list" —
/// without a full-graph wipe.
///
/// **Validate-then-mutate.** The id columns are verified to exist
/// *before* any edge is removed, so a malformed DataFrame can't leave
/// the graph half-cleared. The add is delegated to [`add_connections`],
/// so conflict handling, stub vivification of missing endpoints, and the
/// report shape are identical to a plain add.
#[allow(clippy::too_many_arguments)]
pub fn replace_connections(
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
    let _arena_guard = graph.graph.begin_query(); // disk arena guard (owned; no-op on memory/mapped)
    let column_names = df_data.get_column_names();
    let mut interned_names = vec![
        connection_type.as_str(),
        source_type.as_str(),
        target_type.as_str(),
        PROVISIONAL_KEY,
    ];
    interned_names.extend(RESERVED_PROVENANCE_KEYS.iter().copied());
    interned_names.extend(column_names.iter().map(String::as_str));
    preflight_interner_names(graph, interned_names)?;
    graph
        .prepare_disk_mutation()
        .map_err(|e| format!("disk mutation lease failed: {e}"))?;
    // --- Validate column presence BEFORE deleting (atomicity-by-validation) ---
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

    // --- Collect the distinct, non-null source ids present in the DataFrame ---
    let source_id_idx = df_data
        .get_column_index(&source_id_field)
        .ok_or_else(|| format!("Source ID column '{}' not found", source_id_field))?;
    let mut seen: HashSet<Value> = HashSet::new();
    let mut distinct_sources: Vec<Value> = Vec::new();
    for row in 0..df_data.row_count() {
        if let Some(id) = df_data.get_value_by_index(row, source_id_idx) {
            if matches!(id, Value::Null) {
                continue;
            }
            if seen.insert(id.clone()) {
                distinct_sources.push(id);
            }
        }
    }

    // --- Remove the existing edges of this type from those sources ---
    // Nothing to clear if the source type was never created — the add
    // below vivifies it. `lookup_by_id_readonly` self-heals the id index.
    if graph.has_node_type(&source_type) {
        let conn_key = InternedKey::from_str(&connection_type);
        let mut to_remove: Vec<EdgeIndex> = Vec::new();
        for id in &distinct_sources {
            if let Some(node_idx) = graph.lookup_by_id_readonly(&source_type, id) {
                for edge in graph.graph.edges_directed_filtered(
                    node_idx,
                    petgraph::Direction::Outgoing,
                    Some(conn_key),
                ) {
                    // Disk pre-filters; memory/mapped still post-filter.
                    if edge.connection_type() == conn_key {
                        to_remove.push(edge.id());
                    }
                }
            }
        }
        if !to_remove.is_empty() {
            for edge_idx in to_remove {
                GraphWrite::remove_edge(&mut graph.graph, edge_idx);
            }
            graph.invalidate_edge_type_counts_cache();
            graph.connection_types.clear();
        }
    }

    // --- Add the edges the DataFrame describes ---
    add_connections(
        graph,
        df_data,
        connection_type,
        source_type,
        source_id_field,
        target_type,
        target_id_field,
        source_title_field,
        target_title_field,
        conflict_handling,
    )
}

/// Delete every node still marked `_provisional` — a stub vivified for
/// an edge but never promoted by a real node row — along with all its
/// incident edges. Returns `(nodes_purged, edges_removed)`.
pub fn purge_provisional_nodes(graph: &mut DirGraph) -> (usize, usize) {
    let _arena_guard = graph.graph.begin_query(); // disk arena guard (owned; no-op on memory/mapped)
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
    let _arena_guard = graph.graph.begin_query(); // disk arena guard (owned; no-op on memory/mapped)
    graph
        .prepare_disk_mutation()
        .map_err(|e| format!("disk mutation lease failed: {e}"))?;
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
                // Arena guard: get_node -> node_weight materializes on the
                // disk backend (protocol in disk/graph.rs); scoped so the
                // borrow ends before the batch's &mut graph calls.
                let _arena_guard = graph.graph.begin_query();
                if let Some(node) = graph.get_node(target_idx) {
                    detected_target_type = Some(node.node_type_str(&graph.interner).to_string());
                }
            }

            for &source_idx in &source_nodes {
                if detected_source_type.is_none() {
                    // Arena guard: scoped read (see above).
                    let _arena_guard = graph.graph.begin_query();
                    if let Some(node) = graph.get_node(source_idx) {
                        detected_source_type =
                            Some(node.node_type_str(&graph.interner).to_string());
                    }
                }

                // Collect properties from nodes in the chain (source → ... → target)
                let edge_props = if let Some(ref prop_spec) = copy_properties {
                    // Arena guard: node_weight materializes on the disk
                    // backend; scoped so the borrow ends before
                    // batch.add_connection's &mut graph below.
                    let _arena_guard = graph.graph.begin_query();
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

    graph.bump_version();
    Ok(report)
}

pub fn update_node_properties(
    graph: &mut DirGraph,
    nodes: &[(Option<NodeIndex>, Value)],
    property: &str,
) -> Result<NodeOperationReport, String> {
    let _arena_guard = graph.graph.begin_query(); // disk arena guard (owned; no-op on memory/mapped)
    if nodes.is_empty() {
        return Err("No nodes to update".to_string());
    }
    graph
        .prepare_disk_mutation()
        .map_err(|e| format!("disk mutation lease failed: {e}"))?;

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

    for node_type in node_types.keys() {
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
            // Only add valid nodes to batch. Arena guard: node_weight
            // materializes on the disk backend; scoped read.
            let is_live = {
                let _arena_guard = graph.graph.begin_query();
                graph.graph.node_weight(*node_idx).is_some()
            };
            if is_live {
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

    // The batch path writes the property map without the per-write index
    // maintenance the Cypher SET path runs (`DirGraph::plan_property_write`),
    // and `try_index_lookup` trusts `property_indices` unconditionally — so an
    // index built before this call keeps answering with the *old* value and a
    // `MATCH (n:T {prop: <old>})` returns a node that no longer holds it.
    // Same hazard, same remedy as the bulk loader (see `add_nodes` above).
    // A no-op when the touched types carry no index.
    for node_type in node_types.keys() {
        graph.refresh_indexes_for_type(node_type);
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

    graph.bump_version();
    Ok(report)
}

#[cfg(test)]
#[path = "maintain_edge_spec_tests.rs"]
mod edge_spec_tests;

#[cfg(test)]
mod id_index_tests {
    use super::*;

    /// Regression (issue #20): after `add_nodes`, the type's `id_indices`
    /// entry must be present so the read path (`lookup_by_id_readonly`, used
    /// by `MATCH (n {id:X})` and the `MERGE` match) is O(1). Pre-fix the
    /// index was removed and never rebuilt for reads, so id-equality lookups
    /// fell back to an O(node-position) linear scan (e.g. ~26µs for a high-id
    /// node on 30k rows vs ~0.9µs after the fix).
    #[test]
    fn add_nodes_builds_id_index() {
        let mut g = DirGraph::new();
        let rows: Vec<Vec<Value>> = (0..1000).map(|i| vec![Value::Int64(i)]).collect();
        let df = DataFrame::from_cypher_rows(vec!["id".to_string()], rows).unwrap();
        add_nodes(
            &mut g,
            df,
            "Person".to_string(),
            "id".to_string(),
            Some("id".to_string()),
            None,
        )
        .unwrap();

        assert!(
            g.id_indices.contains_key("Person"),
            "id_index must be built after add_nodes so reads are O(1), not a linear scan"
        );
        // The index resolves a high-position id without a scan.
        assert!(g
            .lookup_by_id_readonly("Person", &Value::Int64(999))
            .is_some());
    }

    #[test]
    fn add_nodes_collision_preflight_leaves_graph_unchanged() {
        let mut g = DirGraph::new();
        let incoming = "CollisionType";
        g.interner
            .try_register(
                crate::graph::schema::InternedKey::from_str(incoming),
                "conflicting-existing",
            )
            .unwrap();
        let before_interner: Vec<_> = g
            .interner
            .iter()
            .map(|(key, value)| (key, value.to_string()))
            .collect();
        let df = DataFrame::from_cypher_rows(vec!["id".to_string()], vec![vec![Value::Int64(1)]])
            .unwrap();
        let err = add_nodes(
            &mut g,
            df,
            incoming.to_string(),
            "id".to_string(),
            None,
            None,
        )
        .unwrap_err();
        assert!(err.contains("hash collision"));
        assert_eq!(g.graph.node_count(), 0);
        assert!(g.node_type_metadata.is_empty());
        assert!(g.type_indices.is_empty());
        assert_eq!(
            g.interner
                .iter()
                .map(|(key, value)| (key, value.to_string()))
                .collect::<Vec<_>>(),
            before_interner
        );
    }

    /// A declared-PRIMARY-KEY type rejects a within-batch duplicate id; a
    /// clean batch loads. Undeclared types keep the permissive default.
    #[test]
    fn add_nodes_rejects_within_batch_pk_duplicate() {
        use crate::graph::schema::{NodeSchemaDefinition, SchemaDefinition, SchemaInstall};

        let mut g = DirGraph::new();
        let mut schema = SchemaDefinition::new();
        schema.add_node_schema(
            "Person".to_string(),
            NodeSchemaDefinition {
                primary_key: Some("id".to_string()),
                ..Default::default()
            },
        );
        g.set_schema(schema, SchemaInstall::Merge)
            .expect("schema install");

        let dup = DataFrame::from_cypher_rows(
            vec!["id".to_string()],
            vec![
                vec![Value::Int64(1)],
                vec![Value::Int64(2)],
                vec![Value::Int64(2)],
            ],
        )
        .unwrap();
        let err = add_nodes(
            &mut g,
            dup,
            "Person".to_string(),
            "id".to_string(),
            Some("id".to_string()),
            None,
        )
        .unwrap_err();
        assert!(err.contains("duplicate primary key"), "got: {err}");

        // A clean batch on the same declared-PK type succeeds.
        let clean = DataFrame::from_cypher_rows(
            vec!["id".to_string()],
            vec![vec![Value::Int64(10)], vec![Value::Int64(11)]],
        )
        .unwrap();
        let report = add_nodes(
            &mut g,
            clean,
            "Person".to_string(),
            "id".to_string(),
            Some("id".to_string()),
            None,
        );
        assert!(report.is_ok(), "clean batch should load: {report:?}");
    }

    /// Partial-update guarantee (load-bearing contract): `conflict_handling =
    /// Update` writes only the columns present in the batch, leaving other
    /// properties of the existing node untouched. A reload can re-assert a
    /// subset of fields without clobbering fields another writer owns.
    #[test]
    fn add_nodes_update_is_partial() {
        let mut g = DirGraph::new();
        // Seed: id + status + notes.
        let seed = DataFrame::from_cypher_rows(
            vec!["id".to_string(), "status".to_string(), "notes".to_string()],
            vec![vec![
                Value::Int64(1),
                Value::String("in_progress".into()),
                Value::String("agent work".into()),
            ]],
        )
        .unwrap();
        add_nodes(
            &mut g,
            seed,
            "Task".to_string(),
            "id".to_string(),
            Some("id".to_string()),
            None,
        )
        .unwrap();

        // Reload: only id + spec_link (the "research re-assert").
        let reload = DataFrame::from_cypher_rows(
            vec!["id".to_string(), "spec_link".to_string()],
            vec![vec![Value::Int64(1), Value::String("AlgoSpec-7".into())]],
        )
        .unwrap();
        add_nodes(
            &mut g,
            reload,
            "Task".to_string(),
            "id".to_string(),
            Some("id".to_string()),
            Some("update".to_string()),
        )
        .unwrap();

        let idx = g.lookup_by_id("Task", &Value::Int64(1)).unwrap();
        let node = g.graph.node_weight(idx).unwrap();
        // Agent-owned fields preserved; new field added.
        assert_eq!(
            node.get_field_ref("status").as_deref(),
            Some(&Value::String("in_progress".into())),
            "status must survive a partial update"
        );
        assert_eq!(
            node.get_field_ref("notes").as_deref(),
            Some(&Value::String("agent work".into())),
            "notes must survive a partial update"
        );
        assert_eq!(
            node.get_field_ref("spec_link").as_deref(),
            Some(&Value::String("AlgoSpec-7".into())),
            "the new field must be written"
        );
    }

    /// Regression (issue #20): the read path self-heals. When the index is
    /// *absent* for a type (the state CREATE / DELETE leave it in), the very
    /// first `lookup_by_id_readonly` — a `&self` call — must build and cache
    /// the index, so every subsequent id-equality lookup is O(1) instead of
    /// the old O(node-position) linear scan that re-ran on each read.
    #[test]
    fn readonly_lookup_self_heals_when_index_absent() {
        let mut g = DirGraph::new();
        let rows: Vec<Vec<Value>> = (0..1000).map(|i| vec![Value::Int64(i)]).collect();
        let df = DataFrame::from_cypher_rows(vec!["id".to_string()], rows).unwrap();
        add_nodes(
            &mut g,
            df,
            "Person".to_string(),
            "id".to_string(),
            Some("id".to_string()),
            None,
        )
        .unwrap();

        // Simulate the post-CREATE / post-DELETE state: index invalidated.
        g.id_indices.remove("Person");
        assert!(!g.id_indices.contains_key("Person"));

        // A read-only lookup must still find the node...
        assert!(g
            .lookup_by_id_readonly("Person", &Value::Int64(999))
            .is_some());
        // ...and must have cached the index so the next read is O(1).
        assert!(
            g.id_indices.contains_key("Person"),
            "read path must build + cache the id_index on a miss (issue #20)"
        );
        // A genuinely absent id still resolves to None (no false positives).
        assert!(g
            .lookup_by_id_readonly("Person", &Value::Int64(424242))
            .is_none());
    }
}

#[cfg(test)]
mod replace_connections_tests {
    use super::*;

    fn doc_entity_graph() -> DirGraph {
        let mut g = DirGraph::new();
        let docs = DataFrame::from_cypher_rows(
            vec!["id".to_string()],
            vec![vec![Value::Int64(1)], vec![Value::Int64(2)]],
        )
        .unwrap();
        add_nodes(
            &mut g,
            docs,
            "Doc".to_string(),
            "id".to_string(),
            Some("id".to_string()),
            None,
        )
        .unwrap();
        let ents = DataFrame::from_cypher_rows(
            vec!["id".to_string()],
            vec![
                vec![Value::String("A".into())],
                vec![Value::String("B".into())],
                vec![Value::String("C".into())],
            ],
        )
        .unwrap();
        add_nodes(
            &mut g,
            ents,
            "Entity".to_string(),
            "id".to_string(),
            Some("id".to_string()),
            None,
        )
        .unwrap();
        g
    }

    fn edges_df(pairs: &[(i64, &str)]) -> DataFrame {
        let rows: Vec<Vec<Value>> = pairs
            .iter()
            .map(|(s, t)| vec![Value::Int64(*s), Value::String((*t).into())])
            .collect();
        DataFrame::from_cypher_rows(vec!["s".to_string(), "t".to_string()], rows).unwrap()
    }

    fn count_edges_of_type(g: &DirGraph, node_type: &str, id: i64, conn: &str) -> usize {
        let idx = g
            .lookup_by_id_readonly(node_type, &Value::Int64(id))
            .unwrap();
        let key = InternedKey::from_str(conn);
        g.graph
            .edges_directed_filtered(idx, petgraph::Direction::Outgoing, Some(key))
            .filter(|e| e.connection_type() == key)
            .count()
    }

    fn add_mentions(g: &mut DirGraph, pairs: &[(i64, &str)]) {
        add_connections(
            g,
            edges_df(pairs),
            "MENTIONS".to_string(),
            "Doc".to_string(),
            "s".to_string(),
            "Entity".to_string(),
            "t".to_string(),
            None,
            None,
            None,
        )
        .unwrap();
    }

    fn replace_mentions(g: &mut DirGraph, pairs: &[(i64, &str)]) {
        replace_connections(
            g,
            edges_df(pairs),
            "MENTIONS".to_string(),
            "Doc".to_string(),
            "s".to_string(),
            "Entity".to_string(),
            "t".to_string(),
            None,
            None,
            None,
        )
        .unwrap();
    }

    /// The defining behaviour: a source's edges of the named type become
    /// exactly the supplied set — stale edges are pruned, new ones added.
    #[test]
    fn replace_sets_exact_edge_set() {
        let mut g = doc_entity_graph();
        add_mentions(&mut g, &[(1, "A"), (1, "B")]);
        assert_eq!(count_edges_of_type(&g, "Doc", 1, "MENTIONS"), 2);

        replace_mentions(&mut g, &[(1, "B"), (1, "C")]);
        assert_eq!(count_edges_of_type(&g, "Doc", 1, "MENTIONS"), 2);
    }

    /// Only sources present in the input are pruned; other sources keep
    /// their edges, and edges of other types from the same source survive.
    #[test]
    fn replace_is_scoped_to_input_sources_and_type() {
        let mut g = doc_entity_graph();
        add_mentions(&mut g, &[(1, "A"), (2, "A")]);
        add_connections(
            &mut g,
            edges_df(&[(1, "B")]),
            "CITES".to_string(),
            "Doc".to_string(),
            "s".to_string(),
            "Entity".to_string(),
            "t".to_string(),
            None,
            None,
            None,
        )
        .unwrap();

        replace_mentions(&mut g, &[(1, "C")]);

        assert_eq!(count_edges_of_type(&g, "Doc", 1, "MENTIONS"), 1);
        // doc 2 (absent from the input) keeps its MENTIONS edge.
        assert_eq!(count_edges_of_type(&g, "Doc", 2, "MENTIONS"), 1);
        // The CITES edge from doc 1 is a different type — untouched.
        assert_eq!(count_edges_of_type(&g, "Doc", 1, "CITES"), 1);
    }

    /// Validation runs before any prune — a bad column errors with the
    /// graph's existing edges intact.
    #[test]
    fn replace_validates_before_pruning() {
        let mut g = doc_entity_graph();
        add_mentions(&mut g, &[(1, "A")]);
        let bad = DataFrame::from_cypher_rows(
            vec!["s".to_string(), "wrong".to_string()],
            vec![vec![Value::Int64(1), Value::String("B".into())]],
        )
        .unwrap();
        let err = replace_connections(
            &mut g,
            bad,
            "MENTIONS".to_string(),
            "Doc".to_string(),
            "s".to_string(),
            "Entity".to_string(),
            "t".to_string(),
            None,
            None,
            None,
        );
        assert!(err.is_err());
        // The pre-existing edge must not have been pruned.
        assert_eq!(count_edges_of_type(&g, "Doc", 1, "MENTIONS"), 1);
    }
}

#[cfg(test)]
#[path = "maintain_delete_id_index_tests.rs"]
mod delete_id_index_tests;
