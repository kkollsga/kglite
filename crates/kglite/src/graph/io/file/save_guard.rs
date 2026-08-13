//! The save side of the recovery rule: a checkpoint write must not strand
//! write-ahead frames in front of itself.
//!
//! [`crate::graph::durability::ensure_recovered`] refuses an *open* that
//! attaches no log over a live sidecar. That closes the route through
//! [`open_or_create_graph`](crate::graph::io::open::open_or_create_graph), but
//! not the one through [`load_file`](crate::graph::io::file::load_file), which
//! is deliberately unguarded — it is the primitive durable recovery is built
//! on, and the documented way to read a graph another process writes durably.
//!
//! So the same hazard still reaches the disk through the *other* end: `load`,
//! mutate, `save` back over the path (`kglite.load()` + `save()`, or a
//! non-durable `Session::save`). That save neither stamps `checkpoint_lsn` nor
//! truncates the sidecar, so the frames outlive it and the next durable open
//! replays them over the newer state — the saved value comes back overwritten
//! by an older commit. Refusing at the save is what closes it without taking
//! the read away.
//!
//! ## The rule
//!
//! A save to `path` is refused while `<path>-wal` holds frames whose `lsn` is
//! greater than the `checkpoint_lsn` of the graph being written — which is
//! exactly the set a later durable open would replay *over* the file this save
//! is about to produce, because replay is gated on the `checkpoint_lsn` the
//! `.kgl` carries.
//!
//! ## Why a durable owner's checkpoint is never caught by it
//!
//! Not by an exemption, but by the checkpoint's own ordering: step 2 stamps
//! `checkpoint_lsn = next_lsn - 1` into the graph *before* the save
//! ([`checkpoint_prologue`](crate::graph::durability::checkpoint_prologue)), so
//! every frame in that owner's log sits at or below the stamp and the predicate
//! is false by construction. The rule therefore needs no "is this the recording
//! owner?" branch — and gets a property such a branch would have thrown away:
//! a *recording* graph saved through some route that skipped the prologue
//! strands its own frames exactly like a non-durable owner does, and is
//! refused for it. Frames at or below the stamp are crash residue between the
//! `.kgl` write and the truncation and are not grounds to refuse, exactly as in
//! [`ensure_recovered`](crate::graph::durability::ensure_recovered).

use std::sync::Arc;

use crate::graph::dir_graph::DirGraph;
use crate::graph::durability::{ensure_save_target_recovered, DurableOpenError};

/// Why a save did not happen.
///
/// Two variants rather than one string because bindings map them to different
/// error classes — the Python wheel raises `IOError` for [`Io`](Self::Io) and
/// `ValueError` for [`Refused`](Self::Refused), which is the class every other
/// durability refusal already raises — and flattening them would change what a
/// caller's `except` clause catches.
#[derive(Debug)]
pub enum SaveError {
    /// The write itself failed: serialize, temp-file, rename, fsync, or a disk
    /// generation publish. The path may or may not have been touched.
    Io(String),
    /// The save was refused *before* the path was touched, because writing
    /// here would strand unreplayed write-ahead frames in front of the new
    /// checkpoint (module docs). Nothing was written.
    Refused(String),
}

impl std::fmt::Display for SaveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(m) | Self::Refused(m) => f.write_str(m),
        }
    }
}

impl std::error::Error for SaveError {}

impl From<DurableOpenError> for SaveError {
    fn from(error: DurableOpenError) -> Self {
        match error {
            // An unreadable sidecar is not a refusal — it says nothing about
            // whether the data is unrecovered, only that we could not look.
            DurableOpenError::Io(message) | DurableOpenError::Replay(message) => Self::Io(message),
            DurableOpenError::Refused(message) => Self::Refused(message),
        }
    }
}

/// The guard every save runs before writing `path` (module docs for the rule).
///
/// **Cost, measured release-profile 2026-08-13** (min-of-N, machine under
/// normal load): 3.2 µs when no sidecar exists — a failed `open`, and the
/// overwhelmingly common case — against 5.5 ms for the whole save of even a
/// one-node graph, i.e. 0.06%. When a sidecar *does* exist the scan is
/// proportional to it: 27 µs for one frame, 1.7 ms for 10 000 frames
/// (350 KB, ~0.17 µs/frame). That is the durable owner's own log at
/// checkpoint time, and it is bounded by the same thing the checkpoint is:
/// the truncation at the end of each checkpoint means the log only ever holds
/// the frames written since the last one.
pub(crate) fn ensure_target_recovered(graph: &Arc<DirGraph>, path: &str) -> Result<(), SaveError> {
    ensure_save_target_recovered(std::path::Path::new(path), graph.checkpoint_lsn)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::datatypes::Value;
    use crate::graph::durability::{self, checkpoint_prologue};
    use crate::graph::io::file::{load_file, save_graph};
    use crate::graph::storage::GraphRead;
    use crate::graph::wal::{wal_path, DurabilityLevel, MutationOp, SyncMode, Wal, WalFrame};
    use std::path::Path;

    fn person_frame(lsn: u64, age: i64) -> WalFrame {
        WalFrame {
            lsn,
            ops: vec![MutationOp::UpsertNode {
                node_type: "Person".into(),
                id: Value::Int64(1),
                title: Value::String("Alice".into()),
                properties: vec![("age".to_string(), Value::Int64(age))],
            }],
        }
    }

    fn graph_with_person(age: i64) -> Arc<DirGraph> {
        let mut graph = Arc::new(DirGraph::new());
        crate::graph::mutation::wal_replay::apply_frames(
            crate::graph::handle::make_dir_graph_mut(&mut graph),
            &[person_frame(1, age)],
            0,
        )
        .unwrap();
        graph
    }

    fn age_of(graph: &mut Arc<DirGraph>) -> Option<Value> {
        let dir = crate::graph::handle::make_dir_graph_mut(graph);
        let idx = dir.lookup_by_id("Person", &Value::Int64(1))?;
        dir.graph
            .node_view(idx)
            .and_then(|n| n.get_field_ref("age").map(|c| c.into_owned()))
    }

    /// The third route into the same corruption, and the one `load_file`'s
    /// deliberate lack of a guard leaves open: load the checkpoint (missing a
    /// committed frame), write, save back. Measured before this guard, through
    /// the Python wheel: the saved `age=3` came back as `age=2`.
    ///
    /// The refusal must name both exits, and — the half that makes refusing
    /// better than discarding — the commit must still be there for a durable
    /// open to recover.
    #[test]
    fn a_save_that_would_strand_frames_is_refused() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("graph.kgl");
        let path_str = path.to_string_lossy().into_owned();

        // A durable owner checkpointed `age=1` at lsn 1, then committed
        // `age=2` as frame lsn 2 and died before its next checkpoint.
        let mut seeded = graph_with_person(1);
        crate::graph::handle::make_dir_graph_mut(&mut seeded).checkpoint_lsn = 1;
        save_graph(&mut seeded, &path_str).unwrap();
        Wal::open(wal_path(&path), SyncMode::Barrier)
            .unwrap()
            .append(&person_frame(2, 2))
            .unwrap();

        // A non-durable owner loads the checkpoint (`age=1`, the commit is
        // invisible to it), writes `age=3`, and saves back over the path.
        let mut loaded = load_file(&path_str).unwrap();
        crate::graph::mutation::wal_replay::apply_frames(
            crate::graph::handle::make_dir_graph_mut(&mut loaded),
            &[person_frame(3, 3)],
            2,
        )
        .unwrap();
        let refusal = match save_graph(&mut loaded, &path_str) {
            Err(SaveError::Refused(message)) => message,
            other => panic!("a save that would strand committed frames must be refused: {other:?}"),
        };
        assert!(refusal.contains("graph.kgl-wal"), "{refusal}");
        assert!(refusal.contains("'full' or 'normal'"), "{refusal}");
        assert!(refusal.contains("move the sidecar aside"), "{refusal}");

        // Nothing was written, and nothing was consumed: the checkpoint still
        // holds the value it did, and the commit is still recoverable durably.
        let mut untouched = load_file(&path_str).unwrap();
        assert_eq!(age_of(&mut untouched), Some(Value::Int64(1)));
        let mut recovered = load_file(&path_str).unwrap();
        durability::open_log(&mut recovered, &path, DurabilityLevel::Full).unwrap();
        assert_eq!(age_of(&mut recovered), Some(Value::Int64(2)));
    }

    /// The boundary, from both sides, so the comparison cannot drift: a frame
    /// the checkpoint being written already contains is crash residue and
    /// saves fine (`>=` would refuse it); one frame past it is unrecovered
    /// data (`>` reversed, or `<`, would let it through).
    #[test]
    fn frames_at_or_below_the_checkpoint_still_save() {
        let tmp = tempfile::tempdir().unwrap();

        let residue = tmp.path().join("residue.kgl");
        let mut folded = graph_with_person(1);
        crate::graph::handle::make_dir_graph_mut(&mut folded).checkpoint_lsn = 7;
        Wal::open(wal_path(&residue), SyncMode::Barrier)
            .unwrap()
            .append(&person_frame(7, 9))
            .unwrap();
        save_graph(&mut folded, &residue.to_string_lossy())
            .expect("a frame the checkpoint already folded in is harmless residue");

        let ahead = tmp.path().join("ahead.kgl");
        let mut same = graph_with_person(1);
        crate::graph::handle::make_dir_graph_mut(&mut same).checkpoint_lsn = 7;
        Wal::open(wal_path(&ahead), SyncMode::Barrier)
            .unwrap()
            .append(&person_frame(8, 9))
            .unwrap();
        assert!(
            matches!(
                save_graph(&mut same, &ahead.to_string_lossy()),
                Err(SaveError::Refused(_))
            ),
            "one frame past the checkpoint would be replayed over this save"
        );
    }

    /// A virgin path — no sidecar at all — is the overwhelmingly common save,
    /// and must be untouched by any of this.
    #[test]
    fn a_path_with_no_sidecar_saves() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("fresh.kgl");
        save_graph(&mut graph_with_person(1), &path.to_string_lossy()).unwrap();
        assert!(path.exists());
    }

    /// The durable owner's checkpoint, in the four-step order the format
    /// requires, over its own live sidecar: the prologue stamps
    /// `checkpoint_lsn` *before* the save, so its own frames are at or below
    /// the stamp and the guard has nothing to refuse.
    ///
    /// This is what makes the guard safe without a "recording owner" branch —
    /// and it is asserted here rather than left to the Python durability suite
    /// because it is the ordering, not the graph's backend, that exempts it.
    #[test]
    fn a_durable_checkpoint_over_its_own_log_is_unaffected() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("durable.kgl");
        let path_str = path.to_string_lossy().into_owned();

        // A durable owner: log opened over the checkpoint, one commit logged.
        let mut graph = graph_with_person(1);
        let (mut wal, next_lsn) = durability::open_log(&mut graph, &path, DurabilityLevel::Full)
            .unwrap()
            .expect("a logging level attaches a log");
        wal.append(&person_frame(next_lsn, 2)).unwrap();
        let next_lsn = next_lsn + 1;

        // Steps 1–2, then the save that would otherwise look exactly like the
        // stranding case above.
        checkpoint_prologue(
            &mut wal,
            next_lsn,
            crate::graph::handle::make_dir_graph_mut(&mut graph),
        )
        .unwrap();
        save_graph(&mut graph, &path_str).expect("a durable checkpoint must not refuse itself");

        // And without the stamp — the mutation that proves the exemption is
        // the ordering — the same save is refused.
        let unstamped = tmp.path().join("unstamped.kgl");
        let mut skipped = graph_with_person(1);
        Wal::open(wal_path(&unstamped), SyncMode::Barrier)
            .unwrap()
            .append(&person_frame(1, 2))
            .unwrap();
        assert!(
            matches!(
                save_graph(&mut skipped, &unstamped.to_string_lossy()),
                Err(SaveError::Refused(_))
            ),
            "a save that skips the checkpoint stamp strands the frames it left behind"
        );
    }

    /// A disk graph is a directory, and disk mode keeps no logical log, so the
    /// guard must not invent one — but a sidecar sitting beside the directory
    /// path is the same hazard and is refused there too.
    #[test]
    fn disk_directories_save_without_a_sidecar_and_refuse_with_one() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().join("disk");
        let mut graph = graph_with_person(1);
        crate::graph::handle::make_dir_graph_mut(&mut graph)
            .enable_disk_mode()
            .unwrap();
        save_graph(&mut graph, &root.to_string_lossy()).expect("a disk publish needs no sidecar");

        Wal::open(wal_path(Path::new(&root)), SyncMode::Barrier)
            .unwrap()
            .append(&person_frame(1, 2))
            .unwrap();
        assert!(matches!(
            save_graph(&mut graph, &root.to_string_lossy()),
            Err(SaveError::Refused(_))
        ));
    }
}
