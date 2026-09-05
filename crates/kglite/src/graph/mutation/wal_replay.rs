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
use crate::graph::wal::WalFrame;

#[path = "wal_replay/install.rs"]
mod install;
#[path = "wal_replay/plan.rs"]
mod plan;
#[path = "wal_replay/validate.rs"]
mod validate;

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
    let plan = plan::ReplayPlan::fold(frames, after_lsn);
    if plan.is_empty() {
        return Ok((None, plan.max_lsn));
    }
    let mut working = graph.clone();
    working.graph.adopt_shared_writer_lineage(&graph.graph);
    working
        .prepare_disk_mutation()
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
