//! Cypher mutation execution — execute_mutable + per-clause helpers
//! (execute_create, execute_set, execute_delete, execute_remove, execute_merge).

use super::super::ast::*;
use super::super::result::*;
use super::{clause_display_name, CypherExecutor};
use crate::datatypes::values::Value;
use crate::graph::schema::{DirGraph, EdgeData, InternedKey, NodeData, TypeSchema};
use crate::graph::storage::{GraphRead, GraphWrite};
use petgraph::graph::NodeIndex;
use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

// ============================================================================
// Mutation Execution
// ============================================================================

/// Check if a query contains any mutation clauses
pub fn is_mutation_query(query: &CypherQuery) -> bool {
    query.clauses.iter().any(|c| {
        matches!(
            c,
            Clause::Create(_)
                | Clause::Set(_)
                | Clause::Delete(_)
                | Clause::Remove(_)
                | Clause::Merge(_)
        )
    })
}

/// Execute a mutation query against a mutable graph.
/// Called instead of CypherExecutor::execute() when the query contains CREATE/SET/DELETE.
pub fn execute_mutable(
    graph: &mut DirGraph,
    query: &CypherQuery,
    params: HashMap<String, Value>,
    deadline: Option<Instant>,
) -> Result<CypherResult, String> {
    GraphRead::reset_arenas(&graph.graph);

    let mut result_set = ResultSet::new();
    let mut stats = MutationStats::default();
    let profiling = query.profile;
    let mut profile_stats: Vec<ClauseStats> = Vec::new();

    for (i, clause) in query.clauses.iter().enumerate() {
        if let Some(dl) = deadline {
            if Instant::now() > dl {
                return Err("Query timed out".to_string());
            }
        }
        // Seed first-clause WITH/UNWIND (same as read-only path)
        if i == 0
            && result_set.rows.is_empty()
            && matches!(clause, Clause::With(_) | Clause::Unwind(_))
        {
            result_set.rows.push(ResultRow::new());
        }

        let rows_in = if profiling { result_set.rows.len() } else { 0 };
        let start = if profiling {
            Some(Instant::now())
        } else {
            None
        };

        // If a prior clause produced 0 rows, MATCH/OPTIONAL MATCH cannot
        // extend an empty pipeline — short-circuit to 0 rows.
        if i > 0
            && result_set.rows.is_empty()
            && matches!(clause, Clause::Match(_) | Clause::OptionalMatch(_))
        {
            if let Some(s) = start {
                profile_stats.push(ClauseStats {
                    clause_name: clause_display_name(clause),
                    rows_in,
                    rows_out: 0,
                    elapsed_us: s.elapsed().as_micros() as u64,
                });
            }
            continue;
        }

        match clause {
            // Write clauses: mutate graph directly
            Clause::Create(create) => {
                result_set = execute_create(graph, create, result_set, &params, &mut stats)?;
            }
            Clause::Set(set) => {
                execute_set(graph, set, &result_set, &params, &mut stats)?;
                // Flush staged writes so any subsequent clause's reads
                // (including a trailing RETURN's property projection)
                // observe the SET. SET routes through node_weight_mut →
                // node_mut_cache on disk; without this flush, the next
                // `node_weight` reads through `column_stores` and
                // returns the pre-SET values.
                GraphWrite::flush_pending_writes(&mut graph.graph);
                // Disk: mirror disk's freshly-flushed column_stores back
                // into DirGraph.column_stores so a subsequent add_nodes
                // (which calls sync_disk_column_stores DirGraph→Disk)
                // doesn't clobber the post-SET state with the stale
                // pre-SET DirGraph snapshot. Without this, a multi-stage
                // SET → add_nodes → read pipeline silently loses the
                // SET's effects on disk-mode graphs.
                graph.sync_column_stores_from_disk();
            }
            Clause::Delete(del) => {
                execute_delete(graph, del, &result_set, &mut stats)?;
            }
            Clause::Remove(rem) => {
                execute_remove(graph, rem, &result_set, &mut stats)?;
                // Same rationale as SET — REMOVE goes through
                // node_weight_mut on disk.
                GraphWrite::flush_pending_writes(&mut graph.graph);
                graph.sync_column_stores_from_disk();
            }
            Clause::Merge(merge) => {
                result_set = execute_merge(graph, merge, result_set, &params, &mut stats)?;
                // MERGE may invoke ON MATCH SET / ON CREATE SET via
                // `execute_set`; flush so any following clause sees the
                // mutations.
                GraphWrite::flush_pending_writes(&mut graph.graph);
                graph.sync_column_stores_from_disk();
            }
            // Read clauses: create temporary immutable executor
            _ => {
                let executor = CypherExecutor::with_params(graph, &params, deadline);
                result_set = executor.execute_single_clause(clause, result_set)?;
            }
        }

        if let Some(s) = start {
            profile_stats.push(ClauseStats {
                clause_name: clause_display_name(clause),
                rows_in,
                rows_out: result_set.rows.len(),
                elapsed_us: s.elapsed().as_micros() as u64,
            });
        }
    }

    // Flush any pending mutation state into the steady-state stores so
    // (a) the trailing RETURN's reads observe the writes from this same
    // query, and (b) any subsequent read-only query started by the user
    // sees them too. No-op on memory/mapped (writes land in
    // `StableDiGraph` directly); on disk, drains
    // `node_mut_cache`/`edge_mut_cache` into `column_stores` /
    // `edge_properties` via the same clone-apply-replace path
    // `clear_arenas` runs lazily before the next `&mut self` op.
    // Without this, Cypher SET on a disk-backed graph appeared to no-op
    // until the next mutation/save flushed the cache — see CHANGELOG.
    GraphWrite::flush_pending_writes(&mut graph.graph);
    graph.sync_column_stores_from_disk();

    // Finalize: if RETURN was in the query, finalize with column projection
    let has_return = query.clauses.iter().any(|c| matches!(c, Clause::Return(_)));
    let profile = if profiling { Some(profile_stats) } else { None };

    if has_return || !result_set.columns.is_empty() {
        let executor = CypherExecutor::with_params(graph, &params, deadline);
        let mut result = executor.finalize_result(result_set)?;
        result.stats = Some(stats);
        result.profile = profile;
        Ok(result)
    } else {
        // No RETURN: return empty result with stats
        Ok(CypherResult {
            columns: Vec::new(),
            rows: Vec::new(),
            stats: Some(stats),
            profile,
            diagnostics: None,
            lazy: None,
        })
    }
}

/// Execute a CREATE clause, creating nodes and edges in the graph.
fn execute_create(
    graph: &mut DirGraph,
    create: &CreateClause,
    existing: ResultSet,
    params: &HashMap<String, Value>,
    stats: &mut MutationStats,
) -> Result<ResultSet, String> {
    // Refuse Cypher CREATE on disk-backed graphs. The disk-storage
    // `add_node` path only writes a slot (node_type + row_id) and
    // drops the per-node `NodeData.properties` / `title` / `id`
    // fields — so a naive CREATE would silently lose every property
    // and the auto-generated title. The supported paths for adding
    // nodes to a disk graph are:
    //   1. `graph.add_nodes(df, type, id_field, title_field=...)` —
    //      goes through `batch.rs::flush_chunk`'s deferred-columnar
    //      pass which writes properties to the ColumnStore before
    //      calling `add_node` with `PropertyStorage::Columnar`.
    //   2. Build the graph in default (heap) storage with `CREATE`,
    //      then call `graph.to_disk(path)` to materialise the
    //      column stores from the in-memory state.
    // SET and DELETE on disk graphs work (they route through
    // `node_weight_mut` → `node_mut_cache` → `clear_arenas` flush);
    // only the new-node path is affected by this guard. Cypher MERGE
    // is also refused (separate guard in execute_merge) when it would
    // create new nodes. REMOVE on disk graphs is separately broken
    // (silent no-op) and documented as a known limitation; not a
    // regression introduced by this guard.
    //
    // Loud-fail rather than silent corruption was the explicit
    // operator-facing call in 0.9.26 (see CHANGELOG known-issue
    // entry). A proper write path is tracked for a future release.
    if graph.graph.is_disk() {
        return Err(
            "Cypher CREATE is not supported on storage=\"disk\" graphs in 0.9.26. \
             Use graph.add_nodes(...) for bulk loading, or build the graph in default \
             (heap) storage with CREATE and call graph.to_disk(path) afterward. \
             SET and DELETE on disk-backed graphs work as expected; REMOVE is \
             currently a silent no-op (separate known limitation)."
                .to_string(),
        );
    }

    let source_rows = if existing.rows.is_empty() {
        // No prior MATCH: execute once with an empty row
        vec![ResultRow::new()]
    } else {
        existing.rows
    };

    let mut new_rows = Vec::with_capacity(source_rows.len());

    for row in &source_rows {
        let mut new_row = row.clone();

        for pattern in &create.patterns {
            // Collect variable -> NodeIndex mappings for this pattern
            let mut pattern_vars: HashMap<String, petgraph::graph::NodeIndex> = HashMap::new();

            // Seed with existing bindings from MATCH
            for (var, idx) in row.node_bindings.iter() {
                pattern_vars.insert(var.clone(), *idx);
            }

            // First pass: create all new nodes
            for element in &pattern.elements {
                if let CreateElement::Node(node_pat) = element {
                    // If variable already bound (from MATCH), skip creation
                    if let Some(ref var) = node_pat.variable {
                        if pattern_vars.contains_key(var) {
                            continue;
                        }
                    }

                    let node_idx = create_node(graph, node_pat, &new_row, params, stats)?;

                    if let Some(ref var) = node_pat.variable {
                        pattern_vars.insert(var.clone(), node_idx);
                        new_row.node_bindings.insert(var.clone(), node_idx);
                    }
                }
            }

            // Second pass: create edges
            // Elements are [Node, Edge, Node, Edge, Node, ...]
            let mut i = 1;
            while i < pattern.elements.len() {
                if let CreateElement::Edge(edge_pat) = &pattern.elements[i] {
                    let source_var = get_create_node_variable(&pattern.elements[i - 1]);
                    let target_var = get_create_node_variable(&pattern.elements[i + 1]);

                    let source_idx = resolve_create_node_idx(source_var, &pattern_vars)?;
                    let target_idx = resolve_create_node_idx(target_var, &pattern_vars)?;

                    // Determine actual source/target based on direction
                    let (actual_source, actual_target) = match edge_pat.direction {
                        CreateEdgeDirection::Outgoing => (source_idx, target_idx),
                        CreateEdgeDirection::Incoming => (target_idx, source_idx),
                    };

                    // Schema lock validation for edge
                    if graph.schema_locked {
                        let src_type = graph
                            .get_node(actual_source)
                            .map(|n| n.get_node_type_ref(&graph.interner).to_string())
                            .unwrap_or_default();
                        let tgt_type = graph
                            .get_node(actual_target)
                            .map(|n| n.get_node_type_ref(&graph.interner).to_string())
                            .unwrap_or_default();
                        crate::graph::mutation::validation::validate_edge_creation(
                            &edge_pat.connection_type,
                            &src_type,
                            &tgt_type,
                            &graph.connection_type_metadata,
                            &graph.node_type_metadata,
                        )?;
                    }

                    // Evaluate edge properties
                    let mut edge_props = HashMap::new();
                    {
                        let executor = CypherExecutor::with_params(graph, params, None);
                        for (key, expr) in &edge_pat.properties {
                            let val = executor.evaluate_expression(expr, &new_row)?;
                            edge_props.insert(key.clone(), val);
                        }
                    }

                    graph.register_connection_type(edge_pat.connection_type.clone());
                    stats.relationships_created += 1;

                    let edge_data = EdgeData::new(
                        edge_pat.connection_type.clone(),
                        edge_props,
                        &mut graph.interner,
                    );
                    let edge_index = GraphWrite::add_edge(
                        &mut graph.graph,
                        actual_source,
                        actual_target,
                        edge_data,
                    );

                    // Bind edge variable if named
                    if let Some(ref var) = edge_pat.variable {
                        new_row.edge_bindings.insert(
                            var.clone(),
                            EdgeBinding {
                                source: actual_source,
                                target: actual_target,
                                edge_index,
                            },
                        );
                    }
                }
                i += 2; // Skip to next edge position
            }
        }

        new_rows.push(new_row);
    }

    // Invalidate edge type count cache if any edges were created
    if stats.relationships_created > 0 {
        graph.invalidate_edge_type_counts_cache();
    }

    Ok(ResultSet {
        rows: new_rows,
        columns: existing.columns,
        lazy_return_items: None,
    })
}

/// Create a single node from a CreateNodePattern
fn create_node(
    graph: &mut DirGraph,
    node_pat: &CreateNodePattern,
    row: &ResultRow,
    params: &HashMap<String, Value>,
    stats: &mut MutationStats,
) -> Result<petgraph::graph::NodeIndex, String> {
    // Evaluate property expressions (borrow graph immutably, then drop)
    let mut properties = HashMap::new();
    {
        let executor = CypherExecutor::with_params(graph, params, None);
        for (key, expr) in &node_pat.properties {
            let val = executor.evaluate_expression(expr, row)?;
            properties.insert(key.clone(), val);
        }
    }

    // Generate ID
    let id = Value::UniqueId(graph.graph.node_bound() as u32);

    // Determine title: use 'name' or 'title' property if present
    let title = properties
        .get("name")
        .or_else(|| properties.get("title"))
        .cloned()
        .unwrap_or_else(|| {
            let label = node_pat.label.as_deref().unwrap_or("Node");
            Value::String(format!("{}_{}", label, graph.graph.node_bound()))
        });

    let label = node_pat.label.clone().unwrap_or_else(|| "Node".to_string());

    // Schema lock validation
    if graph.schema_locked {
        crate::graph::mutation::validation::validate_node_creation(
            &label,
            &properties,
            &graph.node_type_metadata,
            graph.schema_definition.as_ref(),
        )?;
    }

    // Pre-intern all property keys (borrows only graph.interner)
    let interned_keys: Vec<InternedKey> = properties
        .keys()
        .map(|k| graph.interner.get_or_intern(k))
        .collect();

    // Build or extend the TypeSchema for this label (borrows only graph.type_schemas)
    let schema_entry = graph
        .type_schemas
        .entry(label.clone())
        .or_insert_with(|| Arc::new(TypeSchema::new()));
    let schema_mut = Arc::make_mut(schema_entry);
    for &ik in &interned_keys {
        schema_mut.add_key(ik);
    }
    let schema = Arc::clone(graph.type_schemas.get(&label).unwrap());

    // Create compact node using the shared TypeSchema
    let node_data = NodeData::new_compact(
        id,
        title,
        label.clone(),
        properties,
        &mut graph.interner,
        &schema,
    );

    let node_idx = GraphWrite::add_node(&mut graph.graph, node_data);

    // Update type_indices
    graph
        .type_indices
        .entry_or_default(label.clone())
        .push(node_idx);

    // Invalidate id_indices for this type (lazy rebuild on next lookup)
    graph.id_indices.remove(&label);

    // Update property and composite indices for the new node
    graph.update_property_indices_for_add(&label, node_idx);

    // Ensure type metadata exists for this type (consistent with Python add_nodes API)
    ensure_type_metadata(graph, &label, node_idx);

    stats.nodes_created += 1;

    Ok(node_idx)
}

/// Ensure type metadata exists for the given node type.
/// Reads property types from the sample node and upserts them into graph metadata.
/// This mirrors the behavior of the Python add_nodes() API in maintain.rs.
fn ensure_type_metadata(
    graph: &mut DirGraph,
    node_type: &str,
    sample_node_idx: petgraph::graph::NodeIndex,
) {
    // Read sample node properties for type inference
    let sample_props: HashMap<String, String> = match graph.graph.node_weight(sample_node_idx) {
        Some(node) => node
            .property_iter(&graph.interner)
            .map(|(k, v)| (k.to_string(), value_type_name(v)))
            .collect(),
        None => return,
    };

    graph.upsert_node_type_metadata(node_type, sample_props);
}

/// Map a Value variant to its type name string (for SchemaNode property types).
fn value_type_name(v: &Value) -> String {
    match v {
        Value::String(_) => "String",
        Value::Int64(_) => "Int64",
        Value::Float64(_) => "Float64",
        Value::Boolean(_) => "Boolean",
        Value::UniqueId(_) => "UniqueId",
        Value::DateTime(_) => "DateTime",
        Value::Point { .. } => "Point",
        Value::Duration { .. } => "Duration",
        Value::Null => "Null",
        Value::NodeRef(_) => "NodeRef",
    }
    .to_string()
}

/// Extract the variable name from a CreateElement::Node
fn get_create_node_variable(element: &CreateElement) -> Option<&str> {
    match element {
        CreateElement::Node(np) => np.variable.as_deref(),
        _ => None,
    }
}

/// Resolve a variable name to a NodeIndex from the pattern vars map
fn resolve_create_node_idx(
    var: Option<&str>,
    pattern_vars: &HashMap<String, petgraph::graph::NodeIndex>,
) -> Result<petgraph::graph::NodeIndex, String> {
    match var {
        Some(name) => pattern_vars
            .get(name)
            .copied()
            .ok_or_else(|| format!("Unbound variable '{}' in CREATE edge", name)),
        None => Err("CREATE edge requires named source and target nodes".to_string()),
    }
}

/// Execute a SET clause, modifying node properties in the graph.
fn execute_set(
    graph: &mut DirGraph,
    set: &SetClause,
    result_set: &ResultSet,
    params: &HashMap<String, Value>,
    stats: &mut MutationStats,
) -> Result<(), String> {
    // Track which Columnar node types we wrote into so we can refresh
    // per-node Arc<ColumnStore> handles in one O(N-per-type) sweep at
    // the end. Without this batching, every row's `set_property` calls
    // `Arc::make_mut(store)` which clones the entire shared columnar
    // store (one clone per row → O(N²) work, OOM on 1k rows of a
    // type with 6.8k+ nodes — see CHANGELOG note for SET-on-Prospect
    // regression on the loaded Sodir graph).
    let mut touched_columnar_types: std::collections::HashSet<String> =
        std::collections::HashSet::new();

    for row in &result_set.rows {
        for item in &set.items {
            match item {
                SetItem::Property {
                    variable,
                    property,
                    expression,
                } => {
                    // Validate: cannot change id or type
                    if property == "id" {
                        return Err("Cannot SET node id — it is immutable".to_string());
                    }
                    if property == "type" || property == "node_type" || property == "label" {
                        return Err("Cannot SET node type via property assignment".to_string());
                    }

                    // Resolve the node
                    let node_idx = row.node_bindings.get(variable).ok_or_else(|| {
                        format!("Variable '{}' not bound to a node in SET", variable)
                    })?;

                    // Evaluate the expression (borrows graph immutably)
                    let value = {
                        let executor = CypherExecutor::with_params(graph, params, None);
                        executor.evaluate_expression(expression, row)?
                    };

                    // Capture old value + node_type before mutable borrow (for index update)
                    let (old_value, node_type_str) = match graph.get_node(*node_idx) {
                        Some(node) => {
                            let nt = node.get_node_type_ref(&graph.interner).to_string();
                            let old = match property.as_str() {
                                "name" => node.get_field_ref("name").map(Cow::into_owned),
                                _ => node.get_field_ref(property).map(Cow::into_owned),
                            };
                            (old, nt)
                        }
                        None => continue,
                    };

                    // Schema lock validation for SET
                    if graph.schema_locked {
                        crate::graph::mutation::validation::validate_property_set(
                            &node_type_str,
                            property,
                            &value,
                            &graph.node_type_metadata,
                        )?;
                    }

                    // Clone value before it may be consumed by the mutation
                    let value_for_index = value.clone();

                    // Fast path for Columnar storage when the graph's master
                    // `Arc<ColumnStore>` for this node-type is available:
                    // route the write through the master once per batch
                    // instead of through each node's Arc handle. The per-
                    // node Arcs all point at the same allocation, so
                    // `Arc::make_mut` on a node Arc clones the entire store
                    // on every write — O(N²) total for batch SETs. The
                    // master Arc has refcount=1 inside this batch (after
                    // the initial clone, if any), so subsequent writes
                    // mutate in place. We refresh the per-node Arcs in a
                    // single sweep at end of batch (see below).
                    let columnar_row_id =
                        match graph.graph.node_weight(*node_idx).map(|n| &n.properties) {
                            Some(crate::graph::schema::PropertyStorage::Columnar {
                                row_id,
                                ..
                            }) => Some(*row_id),
                            _ => None,
                        };
                    let mut wrote_via_master = false;
                    // Disk-backed graphs use a separate write path; the
                    // master `column_stores` Arc is for the in-memory
                    // Columnar mode only.
                    let is_in_memory = !graph.graph.is_disk();
                    if is_in_memory && property != "title" && property != "name" {
                        if let Some(row_id) = columnar_row_id {
                            // Register the property name in the graph's
                            // StringInterner BEFORE borrowing column_stores.
                            // The non-master path does this via
                            // `node.set_property(..., &mut graph.interner)`;
                            // the master path used `InternedKey::from_str()`
                            // which only hashes — leaving `save()` unable
                            // to resolve the key back to a string at
                            // serialize time. Symptom: every Cypher-SET
                            // property on a 0.8.39 in-memory Sodir-scale
                            // graph survived in-memory but vanished after
                            // save+load, accompanied by
                            // `BUG: InternedKey N not found in StringInterner`.
                            let key = graph.interner.get_or_intern(property);
                            if let Some(master) = graph.column_stores.get_mut(&node_type_str) {
                                Arc::make_mut(master).set(row_id, key, &value, None);
                                touched_columnar_types.insert(node_type_str.clone());
                                stats.properties_set += 1;
                                wrote_via_master = true;
                            }
                        }
                    }
                    if !wrote_via_master {
                        // Compact / Map storage, or title/name, or a Columnar
                        // node whose type isn't registered in
                        // `graph.column_stores` (e.g. disk-mode graphs that
                        // wrap a different store): fall through to the
                        // existing per-node setter.
                        if let Some(node) = GraphWrite::node_weight_mut(&mut graph.graph, *node_idx)
                        {
                            match property.as_str() {
                                "title" => {
                                    node.title = value;
                                }
                                "name" => {
                                    // "name" maps to title in Cypher reads;
                                    // update both title and properties for consistency
                                    node.title = value.clone();
                                    node.set_property("name", value, &mut graph.interner);
                                }
                                _ => {
                                    node.set_property(property, value, &mut graph.interner);
                                }
                            }
                            stats.properties_set += 1;
                        }
                    }

                    // Ensure the DirGraph-level TypeSchema includes this property key
                    if property != "title" {
                        let ik = InternedKey::from_str(property);
                        if let Some(schema_arc) = graph.type_schemas.get_mut(&node_type_str) {
                            if schema_arc.slot(ik).is_none() {
                                Arc::make_mut(schema_arc).add_key(ik);
                            }
                        }
                    }

                    // Update property/composite indices (no active borrows)
                    // "title" only changes the title field, not a HashMap property
                    if property != "title" {
                        graph.update_property_indices_for_set(
                            &node_type_str,
                            *node_idx,
                            property,
                            old_value.as_ref(),
                            &value_for_index,
                        );
                    }

                    // Keep node_type_metadata in sync so schema() is accurate
                    {
                        let mut prop_type = HashMap::new();
                        prop_type.insert(property.clone(), value_type_name(&value_for_index));
                        graph.upsert_node_type_metadata(&node_type_str, prop_type);
                    }
                }
                SetItem::Label { variable, label } => {
                    return Err(format!(
                        "SET label (SET {}:{}) is not yet supported",
                        variable, label
                    ));
                }
            }
        }
    }

    // Refresh per-node `Arc<ColumnStore>` handles for every type we wrote
    // into during this batch. Each node holds its own Arc clone for
    // efficient property reads; after the batch wrote through the
    // graph master, those per-node handles are stale and would surface
    // pre-batch values. This sweep is O(N) per touched type and runs
    // once per SET clause regardless of row count.
    for node_type in touched_columnar_types {
        let new_master = match graph.column_stores.get(&node_type) {
            Some(m) => Arc::clone(m),
            None => continue,
        };
        let indices: Vec<NodeIndex> = graph
            .type_indices
            .get(&node_type)
            .map(|s| s.iter().collect())
            .unwrap_or_default();
        for idx in indices {
            if let Some(node) = GraphWrite::node_weight_mut(&mut graph.graph, idx) {
                if let crate::graph::schema::PropertyStorage::Columnar { store, .. } =
                    &mut node.properties
                {
                    *store = Arc::clone(&new_master);
                }
            }
        }
    }
    Ok(())
}

/// Execute a DELETE clause, removing nodes and/or edges from the graph.
fn execute_delete(
    graph: &mut DirGraph,
    delete: &DeleteClause,
    result_set: &ResultSet,
    stats: &mut MutationStats,
) -> Result<(), String> {
    use std::collections::HashSet;

    let mut nodes_to_delete: HashSet<petgraph::graph::NodeIndex> = HashSet::new();
    // For edge deletion we store edge indices directly — O(1) lookup
    let mut edge_vars_to_delete: Vec<(String, petgraph::graph::EdgeIndex)> = Vec::new();

    // Phase 1: collect all nodes and edges to delete across all rows
    for row in &result_set.rows {
        for expr in &delete.expressions {
            let var_name = match expr {
                Expression::Variable(name) => name,
                other => return Err(format!("DELETE expects variable names, got {:?}", other)),
            };

            if let Some(&node_idx) = row.node_bindings.get(var_name) {
                nodes_to_delete.insert(node_idx);
            } else if let Some(edge_binding) = row.edge_bindings.get(var_name) {
                edge_vars_to_delete.push((var_name.clone(), edge_binding.edge_index));
            } else {
                return Err(format!(
                    "Variable '{}' not bound to a node or relationship in DELETE",
                    var_name
                ));
            }
        }
    }

    // Phase 2: for plain DELETE (not DETACH), verify no node has edges
    if !delete.detach {
        for &node_idx in &nodes_to_delete {
            let has_edges = graph
                .graph
                .edges_directed(node_idx, petgraph::Direction::Outgoing)
                .next()
                .is_some()
                || graph
                    .graph
                    .edges_directed(node_idx, petgraph::Direction::Incoming)
                    .next()
                    .is_some();
            if has_edges {
                let name = graph
                    .graph
                    .node_weight(node_idx)
                    .map(|n| {
                        n.get_field_ref("name")
                            .or_else(|| n.get_field_ref("title"))
                            .map(|v| format!("{:?}", v))
                            .unwrap_or_else(|| format!("index {}", node_idx.index()))
                    })
                    .unwrap_or_else(|| "unknown".to_string());
                return Err(format!(
                    "Cannot delete node '{}' because it still has relationships. Use DETACH DELETE to delete the node and all its relationships.",
                    name
                ));
            }
        }
    }

    // Phase 3: delete explicitly-requested edges (from edge variable bindings)
    let mut deleted_edges: HashSet<petgraph::graph::EdgeIndex> = HashSet::new();
    for (_var, edge_index) in &edge_vars_to_delete {
        if deleted_edges.insert(*edge_index) {
            GraphWrite::remove_edge(&mut graph.graph, *edge_index);
            stats.relationships_deleted += 1;
        }
    }

    // Phase 3's explicit edge-variable deletes still need cache
    // invalidation (`detach_delete_nodes` only covers its own edges).
    if stats.relationships_deleted > 0 {
        graph.invalidate_edge_type_counts_cache();
        graph.connection_types.clear();
    }

    // Phase 4-7: DETACH-delete the nodes — incident edges, the nodes,
    // and index cleanup. For a plain DELETE, Phase 2 has verified the
    // nodes carry no edges, so none are removed here. Shared with
    // `purge_provisional` via `maintain::detach_delete_nodes`.
    let (nodes_deleted, edges_removed) =
        crate::graph::mutation::maintain::detach_delete_nodes(graph, &nodes_to_delete);
    stats.nodes_deleted += nodes_deleted;
    stats.relationships_deleted += edges_removed;

    Ok(())
}

/// Execute a REMOVE clause, removing properties from nodes.
fn execute_remove(
    graph: &mut DirGraph,
    remove: &RemoveClause,
    result_set: &ResultSet,
    stats: &mut MutationStats,
) -> Result<(), String> {
    for row in &result_set.rows {
        for item in &remove.items {
            match item {
                RemoveItem::Property { variable, property } => {
                    // Protect immutable fields
                    if property == "id" {
                        return Err("Cannot REMOVE node id — it is immutable".to_string());
                    }
                    if property == "type" || property == "node_type" || property == "label" {
                        return Err("Cannot REMOVE node type".to_string());
                    }

                    let node_idx = row.node_bindings.get(variable).ok_or_else(|| {
                        format!("Variable '{}' not bound to a node in REMOVE", variable)
                    })?;

                    // Read node_type before mutable borrow (for index update)
                    let node_type_str = graph
                        .get_node(*node_idx)
                        .map(|n| n.get_node_type_ref(&graph.interner).to_string())
                        .unwrap_or_default();

                    // Remove property (mutable borrow, returns old value).
                    //
                    // On disk-backed graphs, the staged-write flush only
                    // persists keys *present* in the staged property Map
                    // — a bare `remove_property` leaves the column store
                    // unchanged and the next read returns the old value.
                    // `clear_property` inserts Null instead so the flush
                    // writes through, matching SET-to-null semantics
                    // (verified working on disk).
                    let is_disk = graph.graph.is_disk();
                    let removed_value = if let Some(node) = graph.get_node_mut(*node_idx) {
                        if is_disk {
                            node.clear_property(property)
                        } else {
                            node.remove_property(property)
                        }
                    } else {
                        None
                    };

                    // Update stats + indices (no active borrows)
                    if let Some(old_val) = removed_value {
                        stats.properties_removed += 1;
                        graph.update_property_indices_for_remove(
                            &node_type_str,
                            *node_idx,
                            property,
                            &old_val,
                        );
                    }
                }
                RemoveItem::Label { variable, label } => {
                    return Err(format!(
                        "REMOVE label (REMOVE {}:{}) is not supported — kglite uses single node_type",
                        variable, label
                    ));
                }
            }
        }
    }
    Ok(())
}

/// Execute a MERGE clause: match-or-create a pattern.
fn execute_merge(
    graph: &mut DirGraph,
    merge: &MergeClause,
    existing: ResultSet,
    params: &HashMap<String, Value>,
    stats: &mut MutationStats,
) -> Result<ResultSet, String> {
    // Refuse MERGE on disk-backed graphs — when the match misses,
    // MERGE creates new nodes through the same disk `add_node` path
    // that CREATE uses, and silently loses properties. Refuse at the
    // entry rather than partial-pattern-match-then-fail. See the
    // `execute_create` guard for the full rationale + workaround.
    if graph.graph.is_disk() {
        return Err(
            "Cypher MERGE is not supported on storage=\"disk\" graphs in 0.9.26 \
             (MERGE can create new nodes through the same write path as CREATE). \
             Use graph.add_nodes(...) / graph.add_connections(...) for bulk loading, \
             or build the graph in default (heap) storage and call \
             graph.to_disk(path) afterward."
                .to_string(),
        );
    }

    let source_rows = if existing.rows.is_empty() {
        vec![ResultRow::new()]
    } else {
        existing.rows
    };

    let mut new_rows = Vec::with_capacity(source_rows.len());

    // Use into_iter to own rows — avoids cloning each row upfront
    for mut new_row in source_rows {
        // Try to match the MERGE pattern
        let matched = try_match_merge_pattern(graph, &merge.pattern, &new_row, params)?;

        if let Some(bound_row) = matched {
            // Pattern matched — merge bindings into row
            for (var, idx) in &bound_row.node_bindings {
                new_row.node_bindings.insert(var.clone(), *idx);
            }
            for (var, binding) in &bound_row.edge_bindings {
                new_row.edge_bindings.insert(var.clone(), *binding);
            }

            // Execute ON MATCH SET
            if let Some(ref set_items) = merge.on_match {
                let set_clause = SetClause {
                    items: set_items.clone(),
                };
                let temp_rs = ResultSet {
                    rows: vec![new_row.clone()],
                    columns: Vec::new(),
                    lazy_return_items: None,
                };
                execute_set(graph, &set_clause, &temp_rs, params, stats)?;
            }
        } else {
            // No match — CREATE the pattern
            let create_clause = CreateClause {
                patterns: vec![merge.pattern.clone()],
            };
            let temp_rs = ResultSet {
                rows: vec![new_row.clone()],
                columns: existing.columns.clone(),
                lazy_return_items: None,
            };
            let created = execute_create(graph, &create_clause, temp_rs, params, stats)?;

            // Merge newly created bindings into our row
            if let Some(created_row) = created.rows.into_iter().next() {
                for (var, idx) in created_row.node_bindings {
                    new_row.node_bindings.insert(var, idx);
                }
                for (var, binding) in created_row.edge_bindings {
                    new_row.edge_bindings.insert(var, binding);
                }
            }

            // Execute ON CREATE SET
            if let Some(ref set_items) = merge.on_create {
                let set_clause = SetClause {
                    items: set_items.clone(),
                };
                let temp_rs = ResultSet {
                    rows: vec![new_row.clone()],
                    columns: Vec::new(),
                    lazy_return_items: None,
                };
                execute_set(graph, &set_clause, &temp_rs, params, stats)?;
            }
        }

        new_rows.push(new_row);
    }

    Ok(ResultSet {
        rows: new_rows,
        columns: existing.columns,
        lazy_return_items: None,
    })
}

/// Try to match a MERGE pattern against the graph.
/// Returns Some(ResultRow) with variable bindings if a match is found, None otherwise.
fn try_match_merge_pattern(
    graph: &DirGraph,
    pattern: &CreatePattern,
    row: &ResultRow,
    params: &HashMap<String, Value>,
) -> Result<Option<ResultRow>, String> {
    let executor = CypherExecutor::with_params(graph, params, None);

    match pattern.elements.len() {
        1 => {
            // Node-only MERGE: (var:Label {key: val, ...})
            if let CreateElement::Node(node_pat) = &pattern.elements[0] {
                // If variable is already bound from prior MATCH, it's already matched
                if let Some(ref var) = node_pat.variable {
                    if let Some(&existing_idx) = row.node_bindings.get(var) {
                        if graph.graph.node_weight(existing_idx).is_some() {
                            let mut result_row = ResultRow::new();
                            result_row.node_bindings.insert(var.clone(), existing_idx);
                            return Ok(Some(result_row));
                        }
                    }
                }

                let label = node_pat.label.as_deref().unwrap_or("Node");

                // Evaluate expected properties
                let expected_props: Vec<(&str, Value)> = node_pat
                    .properties
                    .iter()
                    .map(|(key, expr)| {
                        executor
                            .evaluate_expression(expr, row)
                            .map(|val| (key.as_str(), val))
                    })
                    .collect::<Result<Vec<_>, _>>()?;

                // Helper: verify a candidate node matches all expected properties
                let node_matches_all = |idx: NodeIndex, props: &[(&str, Value)]| -> bool {
                    if let Some(node) = graph.graph.node_weight(idx) {
                        props.iter().all(|(key, expected)| {
                            let value = if *key == "name" || *key == "title" {
                                node.get_field_ref("title")
                            } else {
                                node.get_field_ref(key)
                            };
                            value.as_deref() == Some(expected)
                        })
                    } else {
                        false
                    }
                };

                let build_result = |idx: NodeIndex| -> ResultRow {
                    let mut result_row = ResultRow::new();
                    if let Some(ref var) = node_pat.variable {
                        result_row.node_bindings.insert(var.clone(), idx);
                    }
                    result_row
                };

                // --- Index-accelerated matching ---

                // 1. If pattern contains "id" property, use O(1) id_index lookup
                if let Some((_, id_value)) = expected_props.iter().find(|(k, _)| *k == "id") {
                    if let Some(idx) = graph.lookup_by_id_readonly(label, id_value) {
                        // ID matched — verify remaining properties (if any)
                        if expected_props.len() == 1 || node_matches_all(idx, &expected_props) {
                            return Ok(Some(build_result(idx)));
                        }
                    }
                    return Ok(None);
                }

                // 2. Single non-id property: try property index
                if expected_props.len() == 1 {
                    let (key, ref value) = expected_props[0];
                    // Map name/title aliases to the stored field name
                    let index_key = if key == "name" || key == "title" {
                        "title"
                    } else {
                        key
                    };
                    if let Some(candidates) = graph.lookup_by_index(label, index_key, value) {
                        for &idx in &candidates {
                            if node_matches_all(idx, &expected_props) {
                                return Ok(Some(build_result(idx)));
                            }
                        }
                        return Ok(None);
                    }
                    // No index — fall through to linear scan
                }

                // 3. Multi-property: try composite index
                if expected_props.len() >= 2 {
                    // Build sorted key/value arrays for composite lookup
                    // (exclude id/name/title which use special storage)
                    let mut indexable: Vec<(&str, &Value)> = expected_props
                        .iter()
                        .filter(|(k, _)| *k != "id" && *k != "name" && *k != "title")
                        .map(|(k, v)| (*k, v))
                        .collect();
                    if indexable.len() >= 2 {
                        indexable.sort_by(|a, b| a.0.cmp(b.0));
                        let names: Vec<String> =
                            indexable.iter().map(|(k, _)| k.to_string()).collect();
                        let values: Vec<Value> =
                            indexable.iter().map(|(_, v)| (*v).clone()).collect();
                        if let Some(candidates) =
                            graph.lookup_by_composite_index(label, &names, &values)
                        {
                            for &idx in &candidates {
                                if node_matches_all(idx, &expected_props) {
                                    return Ok(Some(build_result(idx)));
                                }
                            }
                            return Ok(None);
                        }
                    }
                }

                // 4. Fall back to linear scan (no index covers the pattern)
                if let Some(type_nodes) = graph.type_indices.get(label) {
                    for idx in type_nodes.iter() {
                        if node_matches_all(idx, &expected_props) {
                            return Ok(Some(build_result(idx)));
                        }
                    }
                }
                Ok(None)
            } else {
                Err("MERGE pattern must start with a node".to_string())
            }
        }
        3 => {
            // Relationship MERGE: (a)-[r:TYPE]->(b)
            let source_var = get_create_node_variable(&pattern.elements[0]);
            let target_var = get_create_node_variable(&pattern.elements[2]);

            let source_idx = source_var
                .and_then(|v| row.node_bindings.get(v).copied())
                .ok_or("MERGE path: source node must be bound by prior MATCH")?;
            let target_idx = target_var
                .and_then(|v| row.node_bindings.get(v).copied())
                .ok_or("MERGE path: target node must be bound by prior MATCH")?;

            if let CreateElement::Edge(edge_pat) = &pattern.elements[1] {
                let (actual_src, actual_tgt) = match edge_pat.direction {
                    CreateEdgeDirection::Outgoing => (source_idx, target_idx),
                    CreateEdgeDirection::Incoming => (target_idx, source_idx),
                };

                // Search for existing edge matching type
                let interned_ct = InternedKey::from_str(&edge_pat.connection_type);
                let matching_edge = graph
                    .graph
                    .edges_directed(actual_src, petgraph::Direction::Outgoing)
                    .find(|e| {
                        e.target() == actual_tgt && e.weight().connection_type == interned_ct
                    });

                if let Some(edge_ref) = matching_edge {
                    let mut result_row = ResultRow::new();
                    if let Some(ref var) = edge_pat.variable {
                        result_row.edge_bindings.insert(
                            var.clone(),
                            EdgeBinding {
                                source: actual_src,
                                target: actual_tgt,
                                edge_index: edge_ref.id(),
                            },
                        );
                    }
                    Ok(Some(result_row))
                } else {
                    Ok(None)
                }
            } else {
                Err("Expected edge in MERGE path pattern".to_string())
            }
        }
        _ => Err("MERGE supports single-node or single-edge patterns only".to_string()),
    }
}
