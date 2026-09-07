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
use crate::graph::schema::DirGraph;
use crate::graph::wal::{MutationOp, WalFrame};

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
    let before = validate::ConstraintState::capture(&working, &plan, &Default::default());
    let created = install::apply(&mut working, &plan)?;
    let after = validate::ConstraintState::capture(&working, &plan, &created);
    before.validate_successor(&after)?;
    working.reindex();
    for node_type in plan.node_types() {
        working.build_id_index(&node_type);
    }
    working.bump_version();
    Ok((Some(working), plan.max_lsn))
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
        | MutationOp::SetTypeFieldAliases { .. } => false,
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
#[path = "wal_replay/tests.rs"]
mod tests;
