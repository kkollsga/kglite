//! Row-wise property writes onto selected nodes — the fluent `update()` and
//! every `store_as=` writer (`calculate`, `count`, `unique_values`,
//! `collect_children`).
//!
//! Both entry points judge the written rows against the node types'
//! valid-time declarations before anything is written, by the rule a Cypher
//! `SET` follows: a bound that is not a date, or an inverted interval, refuses
//! the whole call; an empty interval is written and counted into one warning.
//! Without the check these writers left rows the declaration rejects, and every
//! `AS OF` read of the type then raised. The fluent `add_properties` writes
//! through its own path (`add_properties::apply_property_updates`) and applies
//! the same check.

use std::collections::HashMap;

use petgraph::graph::NodeIndex;

use crate::datatypes::values::{classify_value_set, ValueSetType};
use crate::datatypes::Value;
use crate::graph::features::temporal::{check_node_update, EmptyIntervals};
use crate::graph::introspection::reporting::NodeOperationReport;
use crate::graph::mutation::batch::{BatchProcessor, ConflictHandling, NodeAction};
use crate::graph::mutation::maintain::type_mismatch_message;
use crate::graph::mutation::ontology_frame_gate::{as_diagnostics, gate_node_updates};
use crate::graph::schema::DirGraph;
use crate::graph::storage::GraphRead;

/// Write `property` onto each `(node, value)`; a `None` node is skipped.
/// Refused, with nothing written, when a value would leave a node breaking a
/// valid-time declaration on its type.
pub fn update_node_properties(
    graph: &mut DirGraph,
    nodes: &[(Option<NodeIndex>, Value)],
    property: &str,
) -> Result<NodeOperationReport, String> {
    if nodes.is_empty() {
        return Err("No nodes to update".to_string());
    }
    let mut empty = EmptyIntervals::default();
    for (idx, value) in nodes {
        if let Some(idx) = idx {
            check_node_update(graph, *idx, &[(property, value)], &mut empty)?;
        }
    }
    let ontology_warnings = gate_node_updates(
        graph,
        nodes
            .iter()
            .filter_map(|(idx, value)| idx.map(|idx| (idx, vec![(property, value)]))),
    )?;
    let mut report = write_node_property(graph, nodes, property)?;
    report.warn_all(empty.diagnostic());
    report.warn_all(as_diagnostics(ontology_warnings));
    Ok(report)
}

/// Write every `(property, value)` of `properties` onto each of `nodes`.
///
/// The end state is judged once, with all of `properties` applied, before
/// anything is written: moving a validity interval by writing both bounds is
/// legal even when one bound alone would invert it. Refused, with nothing
/// written, when a node's end state breaks a valid-time declaration.
/// `nodes_updated` counts one per node per property written.
pub fn update_node_property_set(
    graph: &mut DirGraph,
    nodes: &[NodeIndex],
    properties: &[(String, Value)],
) -> Result<NodeOperationReport, String> {
    if nodes.is_empty() {
        return Err("No nodes to update".to_string());
    }
    let values: Vec<(&str, &Value)> = properties
        .iter()
        .map(|(property, value)| (property.as_str(), value))
        .collect();
    let mut empty = EmptyIntervals::default();
    for &idx in nodes {
        check_node_update(graph, idx, &values, &mut empty)?;
    }
    let ontology_warnings =
        gate_node_updates(graph, nodes.iter().map(|&idx| (idx, values.clone())))?;
    let mut report = NodeOperationReport::new("update_node_property_set".to_string(), 0, 0, 0, 0.0);
    for (property, value) in properties {
        let rows: Vec<(Option<NodeIndex>, Value)> = nodes
            .iter()
            .map(|&idx| (Some(idx), value.clone()))
            .collect();
        let written = write_node_property(graph, &rows, property)?;
        // One count per property written, as the per-property loop it
        // replaced reported.
        report.nodes_updated += written.nodes_updated;
        report.nodes_skipped = report.nodes_skipped.max(written.nodes_skipped);
        report.processing_time_ms += written.processing_time_ms;
        report.errors.extend(written.errors);
    }
    report.warn_all(empty.diagnostic());
    report.warn_all(as_diagnostics(ontology_warnings));
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
/// (`maintain::get_column_types`, via [`classify_value_set`]).
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

/// The batch write of one property onto `nodes`, already judged against the
/// valid-time declarations by its caller.
fn write_node_property(
    graph: &mut DirGraph,
    nodes: &[(Option<NodeIndex>, Value)],
    property: &str,
) -> Result<NodeOperationReport, String> {
    let mut nodes = nodes.to_vec();
    crate::graph::session::snapshot_property_values(
        &graph.graph,
        nodes.iter_mut().map(|(_, value)| value),
    );
    graph
        .prepare_mutation()
        .map_err(|e| format!("disk mutation lease failed: {e}"))?;

    let start_time = std::time::Instant::now();

    let property_string = property.to_string();

    let mut errors = Vec::new();

    let mut node_types = HashMap::new();
    // Cache the validation result for the batch loop below. `node_type_of` is a
    // granular, allocation-free liveness/type lookup on every backend; unlike
    // `get_node`, it does not materialize one full `NodeData` per row on the
    // disk backend. Keeping the result aligned with `nodes` also avoids a
    // second backend lookup when the batch actions are assembled.
    let mut validated_nodes = Vec::with_capacity(nodes.len());
    let mut skipped_count = 0;

    for (node_idx_opt, _) in &nodes {
        if let Some(node_idx) = *node_idx_opt {
            if let Some(node_type) = GraphRead::node_type_of(&graph.graph, node_idx) {
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

    let type_string = observed_type_string(&nodes, &validated_nodes);

    for node_type in node_types.keys() {
        let recorded = graph
            .get_node_type_metadata(node_type)
            .and_then(|meta| meta.get(&property_string))
            .cloned();
        let mut new_prop_types = HashMap::new();
        new_prop_types.insert(property_string.clone(), type_string.clone());
        graph.upsert_node_type_metadata(node_type, new_prop_types);
        if let Some(recorded) = recorded {
            let now = graph
                .get_node_type_metadata(node_type)
                .and_then(|meta| meta.get(&property_string))
                .map_or(type_string.as_str(), String::as_str);
            if let Some(message) =
                type_mismatch_message(&property_string, &recorded, &type_string, now)
            {
                errors.push(message);
            }
        }
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
    // Same hazard, same remedy as the bulk loader (`add_nodes` in
    // `maintain.rs`).
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
