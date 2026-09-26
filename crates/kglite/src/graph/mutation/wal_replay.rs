//! Typed WAL recovery on an unpublished working graph.
//!
//! The fold retains node deletion barriers as well as final values: a recreated
//! logical identity cannot inherit the prior incarnation's labels or incident
//! edges. All eligible frames fold once, avoiding per-frame index rebuilds.
//!
//! Replay does not pass values through import columns. The private installer
//! uses the canonical typed batch and delete seams, then checks the complete
//! final constraint state. A constraint violation already present in a legacy
//! checkpoint may remain only for the same surviving occupants/invalid values.
//! Neither failure nor an error opening the resumed WAL publishes the workspace.

use crate::datatypes::Value;
use crate::graph::schema::{DirGraph, EdgeData};
use crate::graph::storage::{GraphRead, GraphWrite};
use crate::graph::wal::{MutationOp, WalFrame};
use petgraph::graph::{EdgeIndex, NodeIndex};
use std::collections::BTreeSet;

#[path = "wal_replay/declarations.rs"]
mod declarations;
#[path = "wal_replay/edge_embedding_delta.rs"]
pub(crate) mod edge_embedding_delta;
#[path = "wal_replay/edge_embeddings.rs"]
pub(crate) mod edge_embeddings;
#[path = "wal_replay/install.rs"]
mod install;
#[path = "wal_replay/plan.rs"]
mod plan;
#[path = "wal_replay/validate.rs"]
mod validate;

/// Constraint state for a complete loaded snapshot, captured independently of
/// derived indexes that have not been rebuilt yet.
pub(crate) struct CompleteConstraintState(validate::ConstraintState);

pub(crate) fn capture_complete_constraints(graph: &DirGraph) -> CompleteConstraintState {
    CompleteConstraintState(validate::ConstraintState::capture_all(graph))
}

pub(crate) fn validate_complete_constraint_successor(
    before: &CompleteConstraintState,
    graph: &DirGraph,
) -> Result<(), String> {
    before.0.validate_successor_for(
        &validate::ConstraintState::capture_all(graph),
        "legacy endpoint-reference normalization",
    )
}

/// Fold every frame with `lsn > after_lsn`, publishing only a fully validated
/// recovered state. Returns the highest eligible LSN, or `after_lsn`.
pub fn apply_frames(
    graph: &mut DirGraph,
    frames: &[WalFrame],
    after_lsn: u64,
) -> Result<u64, String> {
    let (prepared, lsn) = prepare_replay(graph, frames, after_lsn)?;
    if let Some(prepared) = prepared {
        *graph = prepared;
    }
    Ok(lsn)
}

/// Durable open retains the workspace until its writer is opened successfully.
/// No-op histories avoid cloning a checkpoint merely to open an empty log.
pub(crate) fn prepare_replay(
    graph: &DirGraph,
    frames: &[WalFrame],
    after_lsn: u64,
) -> Result<(Option<DirGraph>, u64), String> {
    refuse_ambiguous_legacy_references(frames, after_lsn)?;
    let plan = plan::ReplayPlan::fold(frames, after_lsn);
    if plan.is_empty() {
        return Ok((None, plan.max_lsn));
    }
    let mut working = graph.clone();
    working.graph.adopt_shared_writer_lineage(&graph.graph);
    working
        .prepare_mutation()
        .map_err(|e| format!("disk mutation lease failed: {e}"))?;
    working.materialize_indexes();
    let initial_edge_embeddings =
        capture_edge_embedding_state(&working, &plan.embedding_conn_types())?;
    let before = validate::ConstraintState::capture(&working, &plan, &Default::default());
    let created = install::apply(&mut working, &plan)?;
    let embedded_groups: BTreeSet<_> = plan
        .edge_embedding_events
        .iter()
        .filter_map(|event| match event {
            edge_embeddings::OrderedEdgeEmbeddingEvent::ReplaceEmbeddings { key, .. }
            | edge_embeddings::OrderedEdgeEmbeddingEvent::PatchEmbeddings { key, .. } => {
                Some(key.clone())
            }
            _ => None,
        })
        .collect();
    // Every topology record is kept, paired or not: an unpaired one still moves
    // its group's members, and the next record's base is digested over them.
    // Which ones park a base is decided by position, inside the interpreter.
    let embedding_events = plan.edge_embedding_events.clone();
    if let Some(key) = plan
        .edge_embedding_events
        .iter()
        .find_map(|event| match event {
            edge_embeddings::OrderedEdgeEmbeddingEvent::ReplaceTopology { key, .. }
                if initial_edge_embeddings
                    .stores
                    .keys()
                    .any(|store| store.conn_type == key.conn_type)
                    && !embedded_groups.contains(key) =>
            {
                Some(key)
            }
            _ => None,
        })
    {
        return Err(format!(
            "WAL relationship group '{}' changes topology without complete relationship embedding state; recovery cannot infer vector identity",
            key.conn_type
        ));
    }
    let has_edge_embedding_payload = embedding_events.iter().any(|event| {
        !matches!(
            event,
            edge_embeddings::OrderedEdgeEmbeddingEvent::ReplaceTopology { .. }
        )
    });
    if has_edge_embedding_payload {
        let final_edge_embeddings =
            edge_embedding_delta::DeltaInterpreter::from_state(initial_edge_embeddings)
                .interpret(embedding_events.clone())?;
        install_edge_embedding_state(&mut working, &final_edge_embeddings, &embedding_events)?;
    }
    let after = validate::ConstraintState::capture(&working, &plan, &created);
    before.validate_successor(&after)?;
    working.reindex();
    for node_type in plan.node_types() {
        working.build_id_index(&node_type);
    }
    // After the comparison above, and after the rows are indexed: a replayed
    // `CREATE CONSTRAINT` is a rule the log declared, not a violation the
    // replay introduced, and its own declarer is what checks it against the
    // recovered rows. A replayed `CREATE INDEX` is rebuilt from those rows.
    plan.declarations.install_schema(&mut working)?;
    // Last, because the vector index is rebuilt from the vectors this installs
    // and the timeseries are keyed by ids the id index above just built.
    plan.declarations.install_payloads(&mut working)?;
    working.bump_version();
    Ok((Some(working), plan.max_lsn))
}

/// The checkpoint's relationship embedding state, as the base every patch in
/// `frames` is resolved against.
///
/// `touched_conn_types` names the relationship types the plan's embedding
/// events mention. A type with **no store at checkpoint** still needs its base
/// groups recorded, because a commit that creates the type's first store
/// digests its base over the group's members as they were before the commit —
/// members that are already there. Skipping them leaves replay's base with
/// empty properties where capture's had rows, and the patch is refused. The set
/// bounds the extra scan to the types a frame actually touched; a reopen whose
/// log mentions no embeddings pays nothing.
fn capture_edge_embedding_state(
    graph: &DirGraph,
    touched_conn_types: &BTreeSet<String>,
) -> Result<edge_embeddings::LogicalEdgeEmbeddingState, String> {
    let mut state = edge_embeddings::LogicalEdgeEmbeddingState::default();
    for ((conn_type, property), store) in &graph.edge_embeddings {
        let text_column = crate::graph::embeddings::text_column_of(property)
            .ok_or_else(|| format!("invalid relationship embedding store key '{property}'"))?;
        state.stores.insert(
            edge_embeddings::LogicalStoreKey {
                conn_type: conn_type.clone(),
                text_column: text_column.to_string(),
            },
            edge_embeddings::LogicalStoreMetadata {
                dimension: store.dimension(),
                metric: store.metric().map(str::to_string),
                model_id: store.model_id().map(str::to_string),
            },
        );
    }
    if state.stores.is_empty() && touched_conn_types.is_empty() {
        return Ok(state);
    }

    let guard = graph.begin_read_pass();
    let mut edges: Vec<_> = graph.graph.edge_indices().collect();
    edges.sort_unstable_by_key(|edge| edge.index());
    for edge in edges {
        let Some(weight) = graph.graph.edge_weight(edge) else {
            continue;
        };
        let conn_type = weight.connection_type_str(&graph.interner);
        let matching: Vec<_> = state
            .stores
            .keys()
            .filter(|key| key.conn_type == conn_type)
            .cloned()
            .collect();
        if matching.is_empty() && !touched_conn_types.contains(conn_type) {
            continue;
        }
        let (source, target) = graph
            .graph
            .edge_endpoints(edge)
            .ok_or_else(|| format!("relationship slot {} has no endpoints", edge.index()))?;
        let source_node = graph
            .graph
            .node_view(source)
            .ok_or_else(|| format!("relationship slot {} has a dead source", edge.index()))?;
        let target_node = graph
            .graph
            .node_view(target)
            .ok_or_else(|| format!("relationship slot {} has a dead target", edge.index()))?;
        let key = edge_embeddings::LogicalGroupKey {
            conn_type: conn_type.to_string(),
            src_type: source_node.node_type_str(&graph.interner).to_string(),
            src_id: source_node.id().into_owned(),
            tgt_type: target_node.node_type_str(&graph.interner).to_string(),
            tgt_id: target_node.id().into_owned(),
        };
        let group = state.groups.entry(key).or_insert_with(|| {
            let stores = matching
                .iter()
                .map(|key| (key.text_column.clone(), Vec::new()))
                .collect();
            edge_embeddings::LogicalGroupState {
                properties: Vec::new(),
                stores,
            }
        });
        group.properties.push(
            weight
                .properties_cloned(&graph.interner)
                .into_iter()
                .collect(),
        );
        for store_key in matching {
            let store = &graph.edge_embeddings[&crate::graph::edge_embeddings::edge_store_key(
                &store_key.conn_type,
                &store_key.text_column,
            )];
            group
                .stores
                .get_mut(&store_key.text_column)
                .expect("all matching stores were initialized")
                .push(
                    store
                        .get(edge)
                        .map(|vector| crate::graph::wal::EdgeVectorWalState {
                            vector: vector.to_vec(),
                            text_hash: store.text_hash(edge),
                        }),
                );
        }
    }
    drop(guard);
    state.validate_complete()?;
    Ok(state)
}

fn install_edge_embedding_state(
    graph: &mut DirGraph,
    state: &edge_embeddings::LogicalEdgeEmbeddingState,
    events: &[edge_embeddings::OrderedEdgeEmbeddingEvent],
) -> Result<(), String> {
    let final_keys: BTreeSet<_> = state
        .stores
        .keys()
        .map(|key| crate::graph::edge_embeddings::edge_store_key(&key.conn_type, &key.text_column))
        .collect();
    graph
        .edge_embeddings
        .retain(|key, _| final_keys.contains(key));
    for (key, metadata) in &state.stores {
        let physical_key =
            crate::graph::edge_embeddings::edge_store_key(&key.conn_type, &key.text_column);
        if graph
            .edge_embeddings
            .get(&physical_key)
            .is_some_and(|store| store.dimension() != metadata.dimension)
        {
            graph.edge_embeddings.insert(
                physical_key.clone(),
                crate::graph::edge_embeddings::EdgeEmbeddingStore::new(
                    metadata.dimension,
                    metadata.metric.as_deref(),
                ),
            );
        }
        let store = graph
            .edge_embeddings
            .entry(physical_key)
            .or_insert_with(|| {
                crate::graph::edge_embeddings::EdgeEmbeddingStore::new(
                    metadata.dimension,
                    metadata.metric.as_deref(),
                )
            });
        store.set_wal_metadata(
            metadata.dimension,
            metadata.metric.clone(),
            metadata.model_id.clone(),
        )?;
    }

    let touched: BTreeSet<_> = events
        .iter()
        .filter_map(|event| match event {
            edge_embeddings::OrderedEdgeEmbeddingEvent::ReplaceTopology { key, .. }
            | edge_embeddings::OrderedEdgeEmbeddingEvent::ReplaceEmbeddings { key, .. }
            | edge_embeddings::OrderedEdgeEmbeddingEvent::PatchEmbeddings { key, .. } => {
                Some(key.clone())
            }
            edge_embeddings::OrderedEdgeEmbeddingEvent::SetStore { .. } => None,
        })
        .collect();
    for key in touched {
        let group = state
            .groups
            .get(&key)
            .ok_or_else(|| format!("missing final relationship group '{}'", key.conn_type))?;
        reinstall_edge_embedding_group(graph, &key, group)?;
    }
    Ok(())
}

fn reinstall_edge_embedding_group(
    graph: &mut DirGraph,
    key: &edge_embeddings::LogicalGroupKey,
    group: &edge_embeddings::LogicalGroupState,
) -> Result<(), String> {
    let endpoints = find_logical_node(graph, &key.src_type, &key.src_id).zip(find_logical_node(
        graph,
        &key.tgt_type,
        &key.tgt_id,
    ));
    let Some((source, target)) = endpoints else {
        return if group.properties.is_empty() {
            Ok(())
        } else {
            Err(format!(
                "relationship embedding group '{}' has missing endpoints",
                key.conn_type
            ))
        };
    };
    let kind = graph.interner.get_or_intern(&key.conn_type);
    let existing: Vec<EdgeIndex> = {
        let guard = graph.begin_read_pass();
        let edges = graph
            .graph
            .edges_connecting(source, target)
            .filter(|edge| edge.weight().connection_type == kind)
            .map(|edge| edge.id())
            .collect();
        drop(guard);
        edges
    };
    for edge in existing {
        crate::graph::edge_embeddings::remove_edge_with_embeddings(graph, edge);
    }
    for (member, properties) in group.properties.iter().enumerate() {
        let properties = properties
            .iter()
            .map(|(name, value)| (graph.interner.get_or_intern(name), value.clone()))
            .collect();
        let edge = graph
            .graph
            .add_edge(source, target, EdgeData::new_interned(kind, properties));
        crate::graph::index_freshness::write_hooks::note_edge_created(graph, edge);
        for (text_column, cells) in &group.stores {
            let Some(vector) = &cells[member] else {
                continue;
            };
            graph
                .edge_embeddings
                .get_mut(&crate::graph::edge_embeddings::edge_store_key(
                    &key.conn_type,
                    text_column,
                ))
                .expect("validated logical store exists")
                .install_wal_vector(edge, &vector.vector, vector.text_hash);
        }
    }
    graph.graph.flush_pending_writes();
    Ok(())
}

fn find_logical_node(graph: &DirGraph, node_type: &str, id: &Value) -> Option<NodeIndex> {
    let guard = graph.begin_read_pass();
    let found = graph.graph.node_indices().find(|&node| {
        graph.graph.node_type_of(node).is_some_and(|kind| {
            graph.interner.resolve(kind) == node_type
                && graph.graph.get_node_id(node).as_ref() == Some(id)
        })
    });
    drop(guard);
    found
}

/// A property NodeRef in an old WAL records only a physical u32 slot, without
/// the checkpoint's slot-to-identity map. Refuse it before folding, cloning,
/// replay mutation, or opening/truncating the sidecar. Identity fields and
/// relationship endpoints remain logical WAL keys and are not stored payloads.
fn refuse_ambiguous_legacy_references(frames: &[WalFrame], after_lsn: u64) -> Result<(), String> {
    if let Some(frame) = frames
        .iter()
        .filter(|frame| frame.lsn > after_lsn)
        .find(|frame| frame.ops.iter().any(mutation_op_has_legacy_reference))
    {
        return Err(format!(
            "WAL frame {} contains a legacy endpoint reference in stored node or relationship state. Its physical node slot has no originating identity map, so replay is refused before graph mutation or WAL repair",
            frame.lsn
        ));
    }
    Ok(())
}

fn mutation_op_has_legacy_reference(op: &MutationOp) -> bool {
    let values_contain_reference = |values: &[(String, Value)]| {
        values
            .iter()
            .any(|(_, value)| crate::graph::session::noderefs::property_value_needs_snapshot(value))
    };
    match op {
        MutationOp::UpsertNode {
            title, properties, ..
        }
        | MutationOp::ReplaceNodeState {
            title, properties, ..
        } => {
            crate::graph::session::noderefs::property_value_needs_snapshot(title)
                || values_contain_reference(properties)
        }
        MutationOp::UpsertEdge { properties, .. } => values_contain_reference(properties),
        MutationOp::ReplaceEdgeGroup { edges, .. } => edges
            .iter()
            .any(|properties| values_contain_reference(properties)),
        // Identity and declaration ops carry no user values at all.
        MutationOp::RemoveNode { .. }
        | MutationOp::RemoveEdge { .. }
        | MutationOp::SetNodeLabels { .. }
        | MutationOp::SetTypeFieldAliases { .. }
        | MutationOp::SetTypeParent { .. }
        | MutationOp::SetOntology { .. }
        | MutationOp::SetSchemaVersion { .. }
        | MutationOp::SetSpatialConfig { .. }
        | MutationOp::SetPropertyIndex { .. }
        | MutationOp::SetConstraint { .. }
        // The payload ops carry channel floats and embedding vectors, neither
        // of which can hold a `Value` at all, so no node reference can hide
        // in one.
        | MutationOp::SetNodeTimeseries { .. }
        | MutationOp::SetTimeseriesConfig { .. }
        | MutationOp::SetEmbeddings { .. }
        | MutationOp::SetVectorIndex { .. }
        | MutationOp::SetEdgeVectorIndex { .. }
        | MutationOp::SetTemporalDeclaration { .. }
        | MutationOp::SetEdgeEmbeddingStore { .. }
        | MutationOp::ReplaceEdgeGroupEmbeddings { .. }
        | MutationOp::PatchEdgeGroupEmbeddings { .. } => false,
    }
}

fn declared_type_name<'a>(values: impl Iterator<Item = &'a Value>) -> String {
    let mut seen = None;
    for value in values.filter(|value| !matches!(value, Value::Null)) {
        let name = value.type_name();
        if seen.is_some_and(|prior| prior != name) {
            return "mixed".into();
        }
        seen = Some(name);
    }
    seen.unwrap_or("mixed").into()
}

#[cfg(test)]
#[path = "wal_replay/regression_tests.rs"]
mod regression_tests;
#[cfg(test)]
#[path = "wal_replay/temporal_tests.rs"]
mod temporal_tests;
#[cfg(test)]
#[path = "wal_replay/tests.rs"]
mod tests;
