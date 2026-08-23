use crate::datatypes::values::{classify_value_set, ValueSetType};
use crate::datatypes::{DataFrame, Value};
use crate::graph::constraints::{ConstraintResult, UniqueConstraintKey};
use crate::graph::introspection::reporting::{ConnectionOperationReport, NodeOperationReport};
use crate::graph::mutation::batch::{
    BatchProcessor, BatchStats, ConflictHandling, ConnectionBatchProcessor, NodeAction,
};
use crate::graph::mutation::delete_state::remove_doomed_nodes;
use crate::graph::mutation::edge_props::{
    intern_edge_props, register_used_edge_property_names, resolve_edge_property_columns,
};
use crate::graph::mutation::rel_constraint_gate::{ConnectionBatchGate, RowFolding};
use crate::graph::schema::{
    CompositeValue, CurrentSelection, DirGraph, InternedKey, TypeSchema, PROVISIONAL_KEY,
    RESERVED_PROVENANCE_KEYS,
};
use crate::graph::storage::lookups::CombinedTypeLookup;
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
    /// UNIQUE / NOT NULL / property-type enforcement has to happen on the bulk
    /// path too: a
    /// constraint the batch engine bypassed would be theatre, since `add_nodes`
    /// is how blueprints, `from_records`, OKF, WAL replay and `extend_graph` all
    /// reach storage. Returning `None` for an unconstrained type is what keeps
    /// the common path free — one `is_empty` and one `Option` check per call
    /// rather than per row.
    fn for_batch(graph: &mut DirGraph, node_type: &str, df_data: &DataFrame) -> Option<Self> {
        // Batch entry point: no violation parked by an earlier load may be
        // attributed to this one.
        graph.clear_pending_constraint_violation();
        (graph.has_unique_constraints()
            || graph.has_required_fields(node_type)
            || graph.type_has_property_type_constraints(node_type))
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
            Err(violation) => Err(graph.record_constraint_violation(*violation)),
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

    /// Gate one input row against the declared NOT NULL, UNIQUE and
    /// property-type constraints.
    ///
    /// Called from [`gate_batch`], which runs over the whole frame *before*
    /// `add_nodes` writes anything at all, so a violation aborts the call with
    /// the graph untouched — no rollback needed and no half-applied load.
    /// `batch_claims` carries the tuples earlier rows of this same input
    /// already claimed, so a repeat *within* one batch is rejected even when
    /// neither row conflicts with stored data.
    fn gate_row(
        &self,
        graph: &DirGraph,
        node_type: &str,
        row: GateRow<'_>,
        existing_idx: Option<NodeIndex>,
        batch_claims: &mut HashSet<(UniqueConstraintKey, CompositeValue)>,
    ) -> ConstraintResult<()> {
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
        graph.check_property_types(node_type, read)?;
        let claims = graph.unique_claims(node_type, read);
        graph.check_unique_claims(&claims, existing_idx)?;
        for claim in &claims {
            if !batch_claims.insert((claim.key.clone(), claim.value.clone())) {
                return Err(Box::new(graph.unique_batch_conflict(claim)));
            }
        }
        Ok(())
    }
}
/// Refuse the whole batch before anything is written, by checking every row's
/// declared constraints (and, for a primary-key type, within-batch id repeats)
/// in one pass ahead of the build loop.
///
/// Every refusal `add_nodes` can raise has to happen before the first byte of
/// observed state lands, and the build loop is not a safe place for one: it
/// commits the node type's metadata and columnar schema before it starts, and
/// `BatchProcessor::add_action` *flushes a chunk into the graph* once a large
/// frame passes the chunk threshold. A gate inside that loop therefore aborts
/// with the type's schema already widened and, past the threshold, with rows
/// already created — the half-applied load the gate exists to prevent.
///
/// Ordering the two phases instead of the two writes is what keeps the
/// promise: the loop below may then assume every row is admissible.
///
/// Costs nothing for the common bulk path — an unconstrained type with no
/// primary key returns before touching a row.
#[allow(clippy::too_many_arguments)]
fn gate_batch(
    graph: &mut DirGraph,
    node_type: &str,
    df_data: &DataFrame,
    columns: Option<&ConstraintColumns>,
    pk_enforced: bool,
    id_idx: usize,
    title_idx: usize,
    unique_id_field: &str,
    title_field: &str,
) -> Result<(), String> {
    if columns.is_none() && !pk_enforced {
        return Ok(());
    }
    let mut batch_claims: HashSet<(UniqueConstraintKey, CompositeValue)> = HashSet::new();
    let mut seen_pk_ids: HashSet<Value> = if pk_enforced {
        HashSet::with_capacity(df_data.row_count())
    } else {
        HashSet::new()
    };
    for row_idx in 0..df_data.row_count() {
        // A row the build loop skips is not loaded, so it is not gated either —
        // the two passes agree on which rows exist by applying the same rule.
        let Some(id) = df_data.get_value_by_index(row_idx, id_idx) else {
            continue;
        };
        if matches!(id, Value::Null) {
            continue;
        }
        if pk_enforced && !seen_pk_ids.insert(id.clone()) {
            return Err(format!(
                "duplicate primary key: node type '{node_type}' declares a primary key but \
                 the input has more than one row with id {id}. Deduplicate the input before \
                 add_nodes, or drop the primary-key declaration."
            ));
        }
        let Some(columns) = columns else {
            continue;
        };
        let title = df_data
            .get_value_by_index(row_idx, title_idx)
            .unwrap_or(Value::Null);
        let existing_idx = graph.id_indices.lookup(node_type, &id);
        columns.gate_row_parked(
            graph,
            node_type,
            GateRow {
                df_data,
                row_idx,
                id: &id,
                title: &title,
                id_field: unique_id_field,
                title_field,
            },
            existing_idx,
            &mut batch_claims,
        )?;
    }
    Ok(())
}

fn check_data_validity(df_data: &DataFrame, unique_id_field: &str) -> Result<(), String> {
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
            "Skipped {skipped_parse_fail} rows: no usable value in ID field \
             '{unique_id_field}' — the cell was empty, or held something the column's \
             stored key type cannot represent. If the ids are integers, pass \
             column_types={{'{unique_id_field}': 'int64'}}; if they are text, pass \
             column_types={{'{unique_id_field}': 'string'}}. Both change the stored key \
             type, so name the one the ids actually are"
        ));
    }
}

/// The index bookkeeping a batch's *updated* rows owe, collected while the
/// stored values are still the ones the batch is about to overwrite.
///
/// The fold that consumes this ([`DirGraph::fold_batch_into_user_indexes`])
/// runs after the batch, when the old value of every indexed property — and
/// the unique tuple each updated node occupied — is gone. Reading it there is
/// impossible; reading it per row here is what replaced an O(nodes-of-type)
/// rebuild per covering index.
///
/// Costs nothing on the common bulk-load shape: an unindexed, unconstrained
/// type resolves `properties` to empty once per call and every row then takes
/// one `bool` test.
///
/// **A capture is not free, and it is taken before anything is known.** The
/// pre-image has to be read in the row loop, so a call cannot first find out
/// how many rows update and then decide; by the time the fold could decline,
/// the reads are spent. A batch whose row count is a large fraction of the
/// type is therefore refused a capture up front — that is the re-load shape
/// `refresh_indexes_for_type` exists for, and it keeps exactly the cost it had.
struct UpdateFold {
    /// [`DirGraph::maintained_index_properties`], resolved once per call.
    properties: Vec<String>,
    /// The type carries an index or a constraint, *and* this batch is small
    /// enough against the type for a per-row capture to be worth taking.
    capturing: bool,
    /// Nodes already captured — a repeated id in one input updates one node,
    /// whose pre-image is what stood before the *first* of those rows.
    seen: HashSet<NodeIndex>,
    /// Capture order, deliberately a `Vec`: a moved node joins the end of its
    /// new bucket, so a `HashMap`'s iteration order would make an indexed
    /// `MATCH`'s row order vary between processes.
    pre_images: Vec<crate::graph::dir_graph::indexes::UpdatedRowPreImage>,
}

impl UpdateFold {
    /// Rows-per-member below which a per-row capture is worth taking. A
    /// capture reads and clones each maintained property of the row's node;
    /// the rebuild it avoids reads every member of the type once. The ratio is
    /// deliberately conservative — being wrong costs time on one call and
    /// never an answer, since both paths leave the same index.
    const CAPTURE_RATIO: usize = 4;

    fn for_batch(graph: &DirGraph, node_type: &str, rows: usize) -> Self {
        let properties = graph.maintained_index_properties(node_type);
        let maintains = !properties.is_empty() || graph.type_has_unique_constraints(node_type);
        let members = graph.type_indices.get(node_type).map_or(0, |m| m.len());
        Self {
            properties,
            capturing: maintains && rows.saturating_mul(Self::CAPTURE_RATIO) < members,
            seen: HashSet::new(),
            pre_images: Vec::new(),
        }
    }

    /// Capture one row's target, if the row updates a node and this call is
    /// taking captures at all.
    fn observe(&mut self, graph: &mut DirGraph, node_type: &str, existing_idx: Option<NodeIndex>) {
        let Some(node_idx) = existing_idx.filter(|_| self.capturing) else {
            return;
        };
        if self.seen.insert(node_idx) {
            self.pre_images.push(graph.capture_update_pre_image(
                node_type,
                node_idx,
                &self.properties,
            ));
        }
    }

    /// The pre-images the fold needs, or `None` when this call declined to take
    /// them and the batch updated something anyway — in which case the fold
    /// cannot know what to vacate and only the rebuild is correct.
    fn pre_images(
        &self,
        updated: usize,
    ) -> Option<&[crate::graph::dir_graph::indexes::UpdatedRowPreImage]> {
        (self.capturing || updated == 0).then_some(&self.pre_images[..])
    }

    /// Fold this call's deltas into the type's covering indexes, falling back
    /// to the whole-type rebuild wherever the fold declines.
    fn fold_or_rebuild(&self, graph: &mut DirGraph, node_type: &str, stats: BatchStats) {
        let folded = self.pre_images(stats.updates).is_some_and(|pre_images| {
            graph.fold_batch_into_user_indexes(node_type, stats.creates, pre_images)
        });
        if !folded {
            graph.refresh_indexes_for_type(node_type);
        }
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
    /// node, a create otherwise. Both arms keep the interned keys all the way
    /// down.
    ///
    /// The create arm therefore *cannot* produce a second node under a live
    /// `(type, id)`: `existing_idx` is a lookup in the type's freshly built id
    /// index, so an id that exists takes the update arm. That is why the durable
    /// duplicate-id refusal (`write.rs::create_node`) has no counterpart here —
    /// a gate on this path could never fire. It is also why WAL replay, which
    /// folds its net state through `add_nodes`, cannot refuse its own history.
    fn action(
        &self,
        df_data: &DataFrame,
        row_idx: usize,
        id: Value,
        title: Value,
        existing_idx: Option<NodeIndex>,
    ) -> NodeAction {
        let properties_interned = self.properties(df_data, row_idx);
        match existing_idx {
            Some(node_idx) => NodeAction::Update {
                node_idx,
                title: self.should_update_title.then_some(title),
                properties: properties_interned,
                conflict_mode: self.conflict_mode,
            },
            None => NodeAction::CreateInterned {
                node_type: self.node_type.to_string(),
                id,
                title,
                properties: properties_interned,
            },
        }
    }
}

/// Parse the user-facing `conflict_handling` option shared by `add_nodes`
/// and `add_connections`; `None` and `"update"` are the default mode.
fn parse_conflict_mode(option: Option<&str>) -> Result<ConflictHandling, String> {
    match option {
        Some("replace") => Ok(ConflictHandling::Replace),
        Some("skip") => Ok(ConflictHandling::Skip),
        Some("preserve") => Ok(ConflictHandling::Preserve),
        Some("sum") => Ok(ConflictHandling::Sum),
        Some("update") | None => Ok(ConflictHandling::Update),
        Some(other) => Err(format!(
            "Unknown conflict handling mode: '{}'. Valid options: 'update' (default), 'replace', 'skip', 'preserve', 'sum'",
            other
        )),
    }
}

/// Track the numeric ids a bulk load supplies, so the engine's auto-id
/// high-water mark can be raised past them once after the row loop (see
/// `DirGraph::next_auto_node_id` — an unraised mark would let a later
/// bare `CREATE` mint a live id).
fn note_loaded_id(max_loaded_id: &mut u32, id: &Value) {
    match id {
        Value::UniqueId(u) => *max_loaded_id = (*max_loaded_id).max(*u),
        Value::Int64(i) if *i >= 0 && *i <= u32::MAX as i64 => {
            *max_loaded_id = (*max_loaded_id).max(*i as u32)
        }
        _ => {}
    }
}

/// Merge this call's column types into the node type's metadata and register
/// the id/title field aliases, appending one message to `errors` for every
/// column whose type disagrees with the stored schema. Cold once-per-call
/// prologue of `add_nodes`, split out to keep that function under the
/// complexity ceiling.
fn install_node_type_metadata(
    graph: &mut DirGraph,
    node_type: &str,
    df_data: &DataFrame,
    unique_id_field: &str,
    title_field: &str,
    should_update_title: bool,
    errors: &mut Vec<String>,
) {
    let df_column_types = get_column_types(df_data);

    if let Some(existing_meta) = graph.get_node_type_metadata(node_type) {
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

    graph.upsert_node_type_metadata(node_type, df_column_types);

    // Record original field name aliases so users can query by original column name
    if unique_id_field != "id" {
        graph
            .id_field_aliases_mut()
            .insert(node_type.to_string(), unique_id_field.to_string());
    }
    // Only register the title alias when the caller explicitly named one.
    // Otherwise a follow-up add_nodes(..., node_title_field=None) would
    // silently rebind the alias to unique_id_field, making `s.id` resolve
    // to the stored title.
    if should_update_title && title_field != "title" {
        graph
            .title_field_aliases_mut()
            .insert(node_type.to_string(), title_field.to_string());
    }
}

/// Build the `TypeSchema` for this call's property columns plus any active
/// provenance keys, and store or extend the node type's schema with it.
/// Returns the interned provenance stamps every row of the batch receives.
/// Cold once-per-call prologue of `add_nodes`.
fn install_type_schema(
    graph: &mut DirGraph,
    node_type: &str,
    property_columns: &[(String, usize)],
) -> Vec<(InternedKey, Value)> {
    let mut schema_keys: Vec<InternedKey> = property_columns
        .iter()
        .map(|(col_name, _)| graph.interner.get_or_intern(col_name))
        .collect();
    // Register every active reserved key so compact and columnar stores can
    // persist the complete engine-owned provenance stamp.
    let provenance_stamps: Vec<(InternedKey, Value)> = if graph.auto_timestamp_for(node_type) {
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

    let existing = graph.type_schemas.get(node_type).cloned();
    if let Some(existing_schema) = existing {
        let mut merged = (*existing_schema).clone();
        for (_, key) in type_schema.iter() {
            merged.add_key(key);
        }
        let merged_arc = Arc::new(merged);
        graph
            .type_schemas_mut()
            .insert(node_type.to_string(), merged_arc);
    } else {
        graph
            .type_schemas_mut()
            .insert(node_type.to_string(), type_schema);
    }
    provenance_stamps
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
    let conflict_mode = parse_conflict_mode(conflict_handling.as_deref())?;

    let should_update_title = node_title_field.is_some();
    let title_field = node_title_field.unwrap_or_else(|| unique_id_field.clone());
    check_data_validity(&df_data, &unique_id_field)?;

    let mut errors = Vec::new();

    // The per-row conflict check reads the type's live id index. Building it
    // here (a no-op once it exists) replaces the `TypeLookup` snapshot, which
    // materialized every id of the type into an owned map on *every* call —
    // half of the O(N_type) cost of appending ten rows to a large type (perf
    // scan 2026-08-14 #2). The index is not written again until after the
    // batch, so the rows still see exactly the pre-call state a snapshot gave
    // them: a duplicate *within* this input is caught by the primary-key and
    // claim checks below, never by reading back a row this call created.
    graph.build_id_index(&node_type);
    let id_idx = df_data
        .get_column_index(&unique_id_field)
        .ok_or_else(|| format!("Column '{}' not found", unique_id_field))?;
    let title_idx = df_data
        .get_column_index(&title_field)
        .ok_or_else(|| format!("Column '{}' not found", title_field))?;

    // Every refusal happens here, ahead of the first write. See `gate_batch`:
    // the metadata install below and the build loop's chunk flushes are both
    // observable, so a gate placed among them cannot leave the graph untouched.
    let constraint_columns = ConstraintColumns::for_batch(graph, &node_type, &df_data);
    let pk_enforced = graph.primary_key_for(&node_type).is_some();
    gate_batch(
        graph,
        &node_type,
        &df_data,
        constraint_columns.as_ref(),
        pk_enforced,
        id_idx,
        title_idx,
        &unique_id_field,
        &title_field,
    )?;

    install_node_type_metadata(
        graph,
        &node_type,
        &df_data,
        &unique_id_field,
        &title_field,
        should_update_title,
        &mut errors,
    );

    // Property column (name + index) resolved once: no per-row string compares
    // and no per-property HashMap lookup in the loop below.
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

    // One clock read per call; every row receives the same complete stamp.
    let provenance_stamps = install_type_schema(graph, &node_type, &property_columns);

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

    let row_builder = RowBuilder {
        node_type: &node_type,
        interned_columns: &interned_columns,
        provenance_stamps: &provenance_stamps,
        property_count,
        should_update_title,
        conflict_mode,
    };

    let mut update_fold = UpdateFold::for_batch(graph, &node_type, df_data.row_count());
    // Loaded ids raise the engine's auto-id high-water mark (applied once
    // after the loop, so the borrow stays out of the hot row path). Without
    // it, a load of sparse ids — one row with id 5 into an empty graph —
    // leaves the mark at 0 and a later `CREATE` with no `id` walks up onto a
    // live id. See `DirGraph::next_auto_node_id`.
    let mut max_loaded_id: u32 = 0;

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

        note_loaded_id(&mut max_loaded_id, &id);

        let title = df_data
            .get_value_by_index(row_idx, title_idx)
            .unwrap_or(Value::Null);

        // Every row here already passed `gate_batch` above.
        let existing_idx = graph.id_indices.lookup(&node_type, &id);

        update_fold.observe(graph, &node_type, existing_idx);
        let action = row_builder.action(&df_data, row_idx, id, title, existing_idx);
        batch.add_action(action, graph)?;
    }

    graph.observe_explicit_id(&Value::UniqueId(max_loaded_id));

    describe_skipped_rows(
        &mut errors,
        skipped_null_id,
        skipped_parse_fail,
        &unique_id_field,
    );

    let (stats, metrics) = batch.execute(graph)?;

    // Fold this call's creations into the type's id_index. The batch adds rows
    // to type_indices but deliberately does not touch id_indices, so the entry
    // is stale until one of these two runs.
    //
    // The index must end up *present*, not merely valid: `lookup_by_id_readonly`
    // — `MATCH (n {id:X})` and the `MERGE` match — does not build it, and an
    // absent entry sent every id-equality read down an O(node-position) scan
    // (issue #20). Folding keeps it present at O(created); the rebuild is the
    // fallback for the cases the fold declines (see
    // `fold_appended_ids_into_index`), and is what the whole call used to pay
    // unconditionally.
    if !graph.fold_appended_ids_into_index(&node_type, stats.creates) {
        graph.id_indices.remove(&node_type);
        graph.build_id_index(&node_type);
    }

    // Same staleness hazard for the *secondary* indexes: the batch path skips
    // the per-write incremental maintenance the Cypher executor runs, and
    // `try_index_lookup` trusts `property_indices` unconditionally, so a stale
    // index silently hides every row this call loaded. Creates get the
    // per-node maintenance a `CREATE` gives them; updates move buckets and
    // re-claim tuples from `UpdateFold`'s pre-images; the rebuild stays the
    // fallback (see `fold_batch_into_user_indexes` for which cases take it).
    update_fold.fold_or_rebuild(graph, &node_type, stats);

    let elapsed_ms = metrics.processing_time * 1000.0;

    let mut report = NodeOperationReport::new(
        "add_nodes".to_string(),
        stats.creates,
        stats.updates,
        skipped_count,
        elapsed_ms,
    );

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
/// Endpoints must already exist (see `skipped_missing_endpoint`).
pub fn add_edges_from_specs(
    graph: &mut DirGraph,
    specs: Vec<EdgeSpec>,
) -> Result<EdgeSpecReport, String> {
    // An empty batch changes nothing, so it must not touch the graph: the
    // disk-mutation lease, the interner preflight and — the one a caller can
    // observe — the version bump at the end were all paid for zero edges, and
    // an unnecessary bump costs a concurrent OCC committer its race. Reachable
    // from `kglite_create_edges_batch` with `[]`.
    if specs.is_empty() {
        return Ok(EdgeSpecReport::default());
    }
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
                    // The batch carries interned keys; spec properties arrive
                    // named, so intern here rather than a layer deeper.
                    let props: Vec<(InternedKey, Value)> = props
                        .into_iter()
                        .map(|(k, v)| (graph.interner.get_or_intern(&k), v))
                        .collect();
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
            batch.schema_property_types(graph),
        )?;

        let (stats, _metrics) = batch.execute(graph, edge_type)?;
        report.connections_created += stats.connections_created;
    }
    graph.bump_version();
    Ok(report)
}

/// What one `add_connections` frame's rows resolved to, produced before any
/// mutation runs (see the call site for why the split exists).
struct ResolvedEndpoints {
    /// `(row, source, target)` for the rows whose endpoints both exist.
    matched: Vec<(usize, NodeIndex, NodeIndex)>,
    /// Rows held back because an endpoint is missing, with their raw ids.
    /// The missing ids are vivified as provisional stubs (Pass B) and these
    /// rows replayed (Pass C) — an edge to a missing endpoint is never
    /// dropped.
    deferred: Vec<(usize, Value, Value)>,
    /// Deduped, order-preserving ids to vivify.
    missing_sources: Vec<Value>,
    missing_targets: Vec<Value>,
    null_source_rows: usize,
    null_target_rows: usize,
}

/// Resolve every row's endpoints against the endpoint types' id indices.
///
/// Probes `graph.id_indices` in place when both types are overlay-resident —
/// the case for every heap-resident graph — and otherwise falls back to
/// exactly the lookup this replaced (`CombinedTypeLookup::from_id_indices`,
/// which materializes a base entry or, failing that, scans the graph).
fn resolve_endpoints(
    graph: &DirGraph,
    df_data: &DataFrame,
    source_type: &str,
    target_type: &str,
    source_id_idx: usize,
    target_id_idx: usize,
) -> Result<ResolvedEndpoints, String> {
    if let Some(resolved) =
        graph
            .id_indices
            .with_overlay_type_pair(source_type, target_type, |source, target| {
                scan_endpoint_rows(
                    df_data,
                    source_id_idx,
                    target_id_idx,
                    |id| source.get(id),
                    |id| target.get(id),
                )
            })
    {
        return Ok(resolved);
    }
    let lookup = CombinedTypeLookup::from_id_indices(
        &graph.id_indices,
        &graph.graph,
        source_type.to_string(),
        target_type.to_string(),
    )?;
    Ok(scan_endpoint_rows(
        df_data,
        source_id_idx,
        target_id_idx,
        |id| lookup.check_source(id),
        |id| lookup.check_target(id),
    ))
}

/// The row walk both resolution paths share; generic over the two probes so
/// each one monomorphizes into a direct call.
fn scan_endpoint_rows(
    df_data: &DataFrame,
    source_id_idx: usize,
    target_id_idx: usize,
    check_source: impl Fn(&Value) -> Option<NodeIndex>,
    check_target: impl Fn(&Value) -> Option<NodeIndex>,
) -> ResolvedEndpoints {
    let mut out = ResolvedEndpoints {
        matched: Vec::with_capacity(df_data.row_count()),
        deferred: Vec::new(),
        missing_sources: Vec::new(),
        missing_targets: Vec::new(),
        null_source_rows: 0,
        null_target_rows: 0,
    };
    let mut seen_missing_source: HashSet<Value> = HashSet::new();
    let mut seen_missing_target: HashSet<Value> = HashSet::new();

    for row_idx in 0..df_data.row_count() {
        let source_id = match df_data.get_value_by_index(row_idx, source_id_idx) {
            Some(Value::Null) | None => {
                out.null_source_rows += 1;
                continue;
            }
            Some(id) => id,
        };
        let target_id = match df_data.get_value_by_index(row_idx, target_id_idx) {
            Some(Value::Null) | None => {
                out.null_target_rows += 1;
                continue;
            }
            Some(id) => id,
        };
        match (check_source(&source_id), check_target(&target_id)) {
            (Some(source_idx), Some(target_idx)) => {
                out.matched.push((row_idx, source_idx, target_idx))
            }
            (s_opt, t_opt) => {
                if s_opt.is_none() && seen_missing_source.insert(source_id.clone()) {
                    out.missing_sources.push(source_id.clone());
                }
                if t_opt.is_none() && seen_missing_target.insert(target_id.clone()) {
                    out.missing_targets.push(target_id.clone());
                }
                out.deferred.push((row_idx, source_id, target_id));
            }
        }
    }
    out
}

/// Resolve an already-extracted `(row, source_id, target_id)` list — the
/// deferred-row replay. Same index precedence as [`resolve_endpoints`];
/// returns one entry per input row, in order.
fn resolve_pairs(
    graph: &DirGraph,
    source_type: &str,
    target_type: &str,
    rows: &[(usize, Value, Value)],
) -> Result<Vec<Option<(NodeIndex, NodeIndex)>>, String> {
    fn walk(
        rows: &[(usize, Value, Value)],
        check_source: impl Fn(&Value) -> Option<NodeIndex>,
        check_target: impl Fn(&Value) -> Option<NodeIndex>,
    ) -> Vec<Option<(NodeIndex, NodeIndex)>> {
        rows.iter()
            .map(|(_, source_id, target_id)| {
                match (check_source(source_id), check_target(target_id)) {
                    (Some(s), Some(t)) => Some((s, t)),
                    _ => None,
                }
            })
            .collect()
    }

    if let Some(resolved) =
        graph
            .id_indices
            .with_overlay_type_pair(source_type, target_type, |source, target| {
                walk(rows, |id| source.get(id), |id| target.get(id))
            })
    {
        return Ok(resolved);
    }
    let lookup = CombinedTypeLookup::from_id_indices(
        &graph.id_indices,
        &graph.graph,
        source_type.to_string(),
        target_type.to_string(),
    )?;
    Ok(walk(
        rows,
        |id| lookup.check_source(id),
        |id| lookup.check_target(id),
    ))
}

/// Column indices for the optional endpoint title fields, `None` for a field
/// that was not named or is not in the frame.
fn title_column_indices(
    df_data: &DataFrame,
    source_title_field: Option<&str>,
    target_title_field: Option<&str>,
) -> (Option<usize>, Option<usize>) {
    (
        source_title_field.and_then(|field| df_data.get_column_index(field)),
        target_title_field.and_then(|field| df_data.get_column_index(field)),
    )
}

/// Append a line per endpoint field whose null ids cost rows.
///
/// Genuine skips only: a row whose endpoint is *missing* is vivified as a stub
/// and replayed, not skipped, so it is never reported here.
fn report_null_id_skips(errors: &mut Vec<String>, source: (usize, &str), target: (usize, &str)) {
    for (skipped, field, side) in [
        (source.0, source.1, "source"),
        (target.0, target.1, "target"),
    ] {
        if skipped > 0 {
            errors.push(format!(
                "Skipped {skipped} rows: null values in {side} ID field '{field}'"
            ));
        }
    }
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
    let conflict_mode = parse_conflict_mode(conflict_handling.as_deref())?;

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

    // A source/target type that doesn't exist yet is not an error: an edge to a
    // missing endpoint vivifies a stub node, which registers the type (Pass B).

    let source_id_idx = df_data
        .get_column_index(&source_id_field)
        .ok_or_else(|| format!("Source ID column '{}' not found", source_id_field))?;
    let target_id_idx = df_data
        .get_column_index(&target_id_field)
        .ok_or_else(|| format!("Target ID column '{}' not found", target_id_field))?;

    let (source_title_idx, target_title_idx) = title_column_indices(
        &df_data,
        source_title_field.as_deref(),
        target_title_field.as_deref(),
    );

    // Endpoint resolution is its own pass over the frame, ahead of every
    // mutation, because the id-index probe borrows `graph.id_indices` while
    // the batch needs `&mut graph`. Splitting it that way is what lets the
    // probe read the index *in place*: the per-call materialized
    // `id -> NodeIndex` map it replaces cost one insert per node of the whole
    // endpoint type and was 53% of a property-free `add_connections` at 100k
    // nodes / 24k edges (samply, 2026-08-15), growing with the graph while
    // the row count stayed fixed. It is snapshot-equivalent to the old code:
    // that map was also built before the first mutation, and nothing in the
    // mutating passes below moves an existing node's id.
    let ResolvedEndpoints {
        matched,
        deferred,
        missing_sources,
        missing_targets,
        null_source_rows: skipped_null_source,
        null_target_rows: skipped_null_target,
    } = resolve_endpoints(
        graph,
        &df_data,
        &source_type,
        &target_type,
        source_id_idx,
        target_id_idx,
    )?;
    let mut batch = ConnectionBatchProcessor::new(df_data.row_count());
    batch.set_conflict_mode(conflict_mode);
    // Skip edge existence checks on initial load (no existing edges of this type)
    let is_initial_load = !graph
        .connection_type_metadata
        .contains_key(&connection_type);
    batch.set_skip_existence_check(is_initial_load);

    let mut skipped_count = skipped_null_source + skipped_null_target;

    let property_columns = resolve_edge_property_columns(
        &df_data,
        &source_id_field,
        &target_id_field,
        source_title_field.as_deref(),
        target_title_field.as_deref(),
    );

    // Extract a row's edge properties — shared by the happy path and
    // the deferred-row replay (Pass C). Skip nulls: property access
    // returns Null for missing keys anyway, and an all-null column must
    // register nothing at all (not on the edge, not in the connection type's
    // property list, not in the interner).
    let extract_props = |row_idx: usize| -> Vec<(InternedKey, Value)> {
        let mut properties = Vec::with_capacity(property_columns.len());
        for (_, interned_key, col_idx) in &property_columns {
            if let Some(value) = df_data.get_value_by_index(row_idx, *col_idx) {
                if !matches!(value, Value::Null) {
                    properties.push((*interned_key, value));
                }
            }
        }
        properties
    };

    ConnectionBatchGate {
        connection_type: &connection_type,
        df_data: &df_data,
        property_columns: &property_columns,
        matched: &matched,
        deferred: &deferred,
        conflict_mode,
        folding: RowFolding::for_load(is_initial_load),
    }
    .run(graph)?;

    // Pass A — connect the rows whose endpoints both exist (resolved above).
    for (row_idx, source_idx, target_idx) in matched {
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
        // Same resolve-then-mutate split as Pass A, re-read after Pass B so
        // the freshly vivified stubs are visible.
        let replayed = resolve_pairs(graph, &source_type, &target_type, &deferred)?;
        for ((row_idx, _, _), endpoints) in deferred.iter().zip(replayed) {
            let row_idx = *row_idx;
            let (source_idx, target_idx) = match endpoints {
                Some(pair) => pair,
                None => {
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

    report_null_id_skips(
        &mut errors,
        (skipped_null_source, &source_id_field),
        (skipped_null_target, &target_id_field),
    );

    register_used_edge_property_names(
        &mut graph.interner,
        &property_columns,
        batch.get_schema_properties(),
    );

    update_schema_node(
        graph,
        &connection_type,
        &source_type,
        &target_type,
        batch.schema_property_types(graph),
    )?;

    let (stats, metrics) = batch.execute(graph, connection_type)?;

    // A batch that produced edges must invalidate the edge-cardinality caches,
    // or a Cypher CREATE → add_connections → planner-cost query reads a stale
    // edge-type-count map. Covers both the type_connectivity cache
    // (selectivity-aware planning) and the edge_type_counts_cache used by
    // reorder_match_clauses.
    if stats.connections_created > 0 {
        graph.invalidate_edge_type_counts_cache();
    }

    let mut report = ConnectionOperationReport::new(
        "add_connections".to_string(),
        stats.connections_created,
        skipped_count,
        stats.properties_tracked,
        metrics.processing_time * 1000.0,
    );
    report.stubs_vivified = stubs_vivified;

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

/// Above this share of a type's members, locating the doomed rows costs more
/// than the walk it replaces, so the delete keeps the walk.
///
/// Locating is `k log N` probes into a `Vec` far larger than cache, plus a
/// `k`-entry position list; the retain is one linear `N` pass. The crossover
/// is `k = N / log2(N)` — 1/20th of the bucket at a million rows — and this is
/// that, rounded to a power of two with margin. It only applies to buckets big
/// enough for the difference to exist: below [`POSITIONAL_MIN_BUCKET`] both
/// paths are microseconds and the positional one keeps its (tested) coverage.
const POSITIONAL_MAX_SHARE: usize = 32;

/// Buckets at or below this size always take the positional path.
const POSITIONAL_MIN_BUCKET: usize = 1024;

/// Where each doomed member sits in its type's bucket, for the types whose
/// bucket can answer that in O(k log N).
///
/// Resolving this once serves both halves of the delete: the journal records
/// `BucketRemoved` positions, and the bucket edit closes exactly those gaps.
/// A type missing from the returned map takes the full-bucket retain — see
/// [`TypeIndexStore::positions_of`](crate::graph::storage::disk::type_index::TypeIndexStore::positions_of).
///
/// Declines wholesale when some doomed node's type could not be read, because
/// the retain removes every doomed index from every affected bucket while this
/// removes only the ones it located.
fn doomed_bucket_positions(
    graph: &mut DirGraph,
    doomed_ids: &HashMap<String, Vec<(Value, NodeIndex)>>,
    nodes_to_delete: &HashSet<NodeIndex>,
) -> HashMap<String, Vec<(usize, NodeIndex)>> {
    let typed: usize = doomed_ids.values().map(Vec::len).sum();
    if typed != nodes_to_delete.len() {
        return HashMap::new();
    }
    let mut resolved = HashMap::with_capacity(doomed_ids.len());
    for (node_type, entries) in doomed_ids {
        let bucket_len = graph
            .type_indices
            .get(node_type)
            .map(|members| members.len())
            .unwrap_or(0);
        if bucket_len > POSITIONAL_MIN_BUCKET
            && entries.len().saturating_mul(POSITIONAL_MAX_SHARE) > bucket_len
        {
            continue;
        }
        let members: Vec<NodeIndex> = entries.iter().map(|(_, idx)| *idx).collect();
        if let Some(hits) = graph.type_indices.positions_of(node_type, &members) {
            resolved.insert(node_type.clone(), hits);
        }
    }
    resolved
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
    bucket_positions: &HashMap<String, Vec<(usize, NodeIndex)>>,
) {
    if graph.graph.undo_journal_mut().is_none() {
        return;
    }
    // Types whose positions are already known: recorded directly, descending,
    // which is the order `note_bucket_retain` produces by scanning the bucket.
    // Copying the whole bucket to re-derive them would reintroduce the O(N_type)
    // term on the journalled path that the positional edit removes on the
    // unjournalled one.
    let mut positional: Vec<(BucketId, usize, NodeIndex)> = Vec::new();
    let mut evictions: Vec<(BucketId, Vec<NodeIndex>)> = Vec::new();
    for node_type in affected_types {
        if let Some(hits) = bucket_positions.get(node_type) {
            positional.extend(
                hits.iter()
                    .rev()
                    .map(|(pos, idx)| (BucketId::NodeType(node_type.clone()), *pos, *idx)),
            );
        } else if let Some(members) = graph.type_indices.get(node_type) {
            evictions.push((BucketId::NodeType(node_type.clone()), members.to_vec()));
        }
        // User-created indexes, same treatment. Only buckets that actually
        // hold a doomed node are captured, so the cost tracks the deletion
        // rather than the size of the index.
        for (key, value_map) in &graph.property_indices {
            if &key.0 != node_type {
                continue;
            }
            for (value, members) in value_map.iter() {
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
            for (value, members) in btree.iter() {
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
            for (value, members) in comp_map.iter() {
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
        for (bucket, pos, idx) in positional {
            journal.note_bucket_removed(bucket, idx, pos);
        }
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
            if let Some(node) = graph.graph.node_view(node_idx) {
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

    remove_doomed_nodes(graph, nodes_to_delete);

    // Where each doomed member sits in its type bucket, resolved once for both
    // the journal and the edit. `remove_node` above does not touch
    // `type_indices`, so these positions are still the pre-delete ones.
    let bucket_positions = doomed_bucket_positions(graph, &doomed_ids, nodes_to_delete);

    // Statement-rollback capture, before the sweeps below strip the doomed
    // members out of the buckets.
    journal_bucket_evictions(graph, &affected_types, nodes_to_delete, &bucket_positions);

    // Index cleanup — StableDiGraph keeps surviving indices stable.
    for node_type in &affected_types {
        match bucket_positions.get(node_type) {
            // O(k log N) + one memmove per surviving run, instead of a walk of
            // the whole type with a hashed probe per member.
            Some(hits) => graph.type_indices.remove_positions(node_type, hits),
            None => graph
                .type_indices
                .retain_in_type(node_type, |idx| !nodes_to_delete.contains(idx)),
        }
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
                value_map.retain_members(|idx| !nodes_to_delete.contains(idx));
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
                value_map.retain_members(|idx| !nodes_to_delete.contains(idx));
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
                value_map.retain_members_pruning_empty(|idx| !nodes_to_delete.contains(idx));
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

    // --- Everything else that can refuse the call, also before the delete ---
    //
    // `replace_connections` destroys data and then rebuilds it, so any refusal
    // raised past the delete leaves the graph with neither the old edges nor
    // the new ones. Column presence and the interner were already checked
    // above for exactly that reason; these are the two remaining refusals
    // `add_connections` could still raise afterwards.
    //
    // 1. The conflict mode. An unknown one is a caller mistake, and it was
    //    being discovered only after the edges were gone.
    let conflict_mode = parse_conflict_mode(conflict_handling.as_deref())?;

    let source_id_idx = df_data
        .get_column_index(&source_id_field)
        .ok_or_else(|| format!("Source ID column '{}' not found", source_id_field))?;
    let target_id_idx = df_data
        .get_column_index(&target_id_field)
        .ok_or_else(|| format!("Target ID column '{}' not found", target_id_field))?;

    // 2. Stub vivification. An edge to a missing endpoint creates a stub node,
    //    and that stub goes through the node-side constraint gate — a refusal
    //    there aborts the add. Vivifying here means the gate fires while the
    //    old edges are still intact; `add_connections` then resolves every
    //    endpoint and vivifies nothing, so the graph ends up identical. The
    //    stubs this creates are counted back into the report below, and an
    //    endpoint resolution is a pass of index probes over the frame — paid
    //    only on the replace path, never on plain `add_connections`.
    let resolved = resolve_endpoints(
        graph,
        &df_data,
        &source_type,
        &target_type,
        source_id_idx,
        target_id_idx,
    )?;
    // 3. Declared relationship constraints. `add_connections` gates them too,
    //    but that gate runs *after* the delete below — so the frame is judged
    //    here as well, against the state the replace will leave: the delete
    //    removes whatever these pairs hold, which makes every row a create.
    ConnectionBatchGate {
        connection_type: &connection_type,
        df_data: &df_data,
        property_columns: &resolve_edge_property_columns(
            &df_data,
            &source_id_field,
            &target_id_field,
            source_title_field.as_deref(),
            target_title_field.as_deref(),
        ),
        matched: &resolved.matched,
        deferred: &resolved.deferred,
        conflict_mode,
        // Why this regime and not the loader's: `RowFolding::for_replace`.
        folding: RowFolding::for_replace(graph, &connection_type),
    }
    .run(graph)?;

    let mut stubs_vivified = 0usize;
    if !resolved.missing_sources.is_empty() {
        stubs_vivified += vivify_stubs(graph, &source_type, &resolved.missing_sources)?;
    }
    if !resolved.missing_targets.is_empty() {
        stubs_vivified += vivify_stubs(graph, &target_type, &resolved.missing_targets)?;
    }
    drop(resolved);

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

    let mut report = add_connections(
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
    )?;
    // The stubs vivified above are this call's, so they belong in its count —
    // `add_connections` found those endpoints already present and reported none.
    report.stubs_vivified += stubs_vivified;
    Ok(report)
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
            GraphWrite::set_node_title(&mut graph.graph, source_idx, title);
        }
    }
    if let Some(title_idx) = target_title_idx {
        if let Some(title) = df_data.get_value_by_index(row_idx, title_idx) {
            GraphWrite::set_node_title(&mut graph.graph, target_idx, title);
        }
    }
    Ok(())
}

fn update_schema_node(
    graph: &mut DirGraph,
    connection_type: &str,
    source_type: &str,
    target_type: &str,
    prop_types: HashMap<String, String>,
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

    // The caller supplies observed types (batch.schema_property_types), so
    // "Unknown" survives only for properties never seen with a non-null value.
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

    let mut type_to_level: HashMap<String, usize> = HashMap::new();
    for lvl_idx in 0..level_count {
        if let Some(level) = selection.get_level(lvl_idx) {
            for node_idx in level.iter_node_indices() {
                if let Some(node) = graph.node_view(node_idx) {
                    type_to_level
                        .entry(node.node_type_str(&graph.interner).to_string())
                        .or_insert(lvl_idx);
                }
            }
        }
    }

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

    let walk_to_sources = |start_node: NodeIndex, start_level: usize| -> Vec<NodeIndex> {
        if start_level == source_level {
            return vec![start_node];
        }
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
                if let Some(node) = graph.node_view(target_idx) {
                    detected_target_type = Some(node.node_type_str(&graph.interner).to_string());
                }
            }

            for &source_idx in &source_nodes {
                if detected_source_type.is_none() {
                    // Arena guard: scoped read (see above).
                    let _arena_guard = graph.graph.begin_query();
                    if let Some(node) = graph.node_view(source_idx) {
                        detected_source_type =
                            Some(node.node_type_str(&graph.interner).to_string());
                    }
                }

                let edge_props = if let Some(ref prop_spec) = copy_properties {
                    // Arena guard: node_weight materializes on the disk
                    // backend; scoped so the borrow ends before
                    // batch.add_connection's &mut graph below.
                    let _arena_guard = graph.graph.begin_query();
                    let mut props = HashMap::new();
                    for &node_idx in &[source_idx, target_idx] {
                        if let Some(node) = graph.graph.node_view(node_idx) {
                            let nt = node.node_type_str(&graph.interner);
                            if let Some(requested_props) = prop_spec.get(nt) {
                                if requested_props.is_empty() {
                                    for (k, v) in node.property_pairs_named(&graph.interner) {
                                        props.insert(k, v);
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
                let edge_props = intern_edge_props(edge_props, &mut graph.interner);

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
            batch.schema_property_types(graph),
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

/// The observed type string `update_node_properties` records for a batch,
/// classified across **every** value it writes rather than the first one — a
/// heterogeneous batch has no single type, so naming one made the metadata
/// state the opposite of what was stored (`[1, "two", 3]` recorded `Int64`).
/// Those record `"mixed"`, the string the columnar store and WAL replay
/// (`wal_replay::declared_type_name`) already use for a column whose values
/// disagree and which no type-knowledge source reads as a claim. Everything
/// else records what a bulk load of the same values records
/// ([`get_column_types`], via [`classify_value_set`]).
fn observed_type_string(nodes: &[(Option<NodeIndex>, Value)], validated: &[bool]) -> String {
    let written = || {
        nodes
            .iter()
            .zip(validated)
            .filter(|(_, ok)| **ok)
            .map(|((_, value), _)| value)
    };
    match classify_value_set(written()) {
        ValueSetType::Uniform(col_type) => col_type.to_string(),
        ValueSetType::Mixed => "mixed".to_string(),
        // `Point`/`Duration` (no column names them, and this path does not
        // render them as text the way a frame would) and all-null batches
        // both observe nothing, which this path has always spelled
        // `"Unknown"`. With *no* writable row the string is unused: the loop
        // below is keyed off the validated rows.
        ValueSetType::Shapeless | ValueSetType::Empty => "Unknown".to_string(),
    }
}

pub fn update_node_properties(
    graph: &mut DirGraph,
    nodes: &[(Option<NodeIndex>, Value)],
    property: &str,
) -> Result<NodeOperationReport, String> {
    if nodes.is_empty() {
        return Err("No nodes to update".to_string());
    }
    graph
        .prepare_disk_mutation()
        .map_err(|e| format!("disk mutation lease failed: {e}"))?;

    let start_time = std::time::Instant::now();

    let property_string = property.to_string();

    let mut errors = Vec::new();

    let mut node_types = HashMap::new();
    // Cache the validation result for the batch loop below. `node_type_of` is a
    // granular, allocation-free liveness/type lookup on every backend; unlike
    // `get_node`, it does not materialize one full `NodeData` per row into the
    // disk query arena. Keeping the result aligned with `nodes` also avoids a
    // second backend lookup when the batch actions are assembled.
    let mut validated_nodes = Vec::with_capacity(nodes.len());
    let mut skipped_count = 0;

    for (node_idx_opt, _) in nodes {
        if let Some(node_idx) = node_idx_opt {
            if let Some(node_type) = GraphRead::node_type_of(&graph.graph, *node_idx) {
                *node_types
                    .entry(graph.interner.resolve(node_type).to_string())
                    .or_insert(0) += 1;
                validated_nodes.push(true);
            } else {
                validated_nodes.push(false);
                skipped_count += 1;
                errors.push(format!("Node index {:?} not found in graph", node_idx));
            }
        } else {
            validated_nodes.push(false);
            skipped_count += 1;
        }
    }

    let type_string = observed_type_string(nodes, &validated_nodes);

    for node_type in node_types.keys() {
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

    let batch_size = nodes.len();
    let property_key = graph.interner.get_or_intern(&property_string);
    let mut batch = BatchProcessor::new(batch_size);

    for ((node_idx_opt, value), is_validated) in nodes.iter().zip(validated_nodes) {
        if let Some(node_idx) = node_idx_opt {
            if is_validated {
                let action = NodeAction::Update {
                    node_idx: *node_idx,
                    title: None,
                    properties: vec![(property_key, value.clone())],
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

    let elapsed_ms = start_time.elapsed().as_secs_f64() * 1000.0;

    let mut report = NodeOperationReport::new(
        "update_node_properties".to_string(),
        0, // We don't create nodes in this function
        stats.updates,
        skipped_count,
        elapsed_ms,
    );

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
#[path = "maintain_connection_property_tests.rs"]
mod connection_property_tests;

#[cfg(test)]
#[path = "maintain_id_index_tests.rs"]
mod id_index_tests;

#[cfg(test)]
#[path = "maintain_replace_connections_tests.rs"]
mod replace_connections_tests;

#[cfg(test)]
#[path = "maintain_delete_id_index_tests.rs"]
mod delete_id_index_tests;

#[cfg(test)]
#[path = "maintain_incremental_index_tests.rs"]
mod incremental_index_tests;

#[cfg(test)]
#[path = "maintain_positional_delete_tests.rs"]
mod positional_delete_tests;

#[cfg(test)]
#[path = "maintain_property_type_tests.rs"]
mod property_type_tests;

#[cfg(test)]
#[path = "maintain_add_property_type_tests.rs"]
mod add_property_type_tests;
