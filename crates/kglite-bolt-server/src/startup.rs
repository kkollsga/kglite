//! Graph startup: take cross-process write ownership of the graph path, open
//! it, and wrap it in the session the backend serves.
//!
//! Split out of `main` so the acquire-then-open *order* is testable. Ordering
//! is the whole point of the lease: acquiring it after the read is a guard
//! that lets the very race it exists to stop happen first.
//!
//! The session is built here rather than in the backend because at
//! `--durability full`/`normal` its construction *is* part of opening the path:
//! [`Session::open_durable`] recovers the write-ahead sidecar into the graph
//! before the first client can connect, and it must happen inside the writer
//! lease this module already takes.

use std::path::Path;
use std::time::Duration;

use anyhow::{Context, Result};

use kglite::api::durable::DurabilityLevel;
use kglite::api::io::{open_or_create_graph_in_mode, GraphWriterLease, OpenDisposition};
use kglite::api::session::Session;
use kglite::api::storage::{live_storage_mode, StorageMode};

/// How long startup waits for another writer to release the graph.
///
/// Zero — fail fast, unlike the MCP server's 30s. That server acquires inside
/// an interactive tool call, where waiting for a colleague's writer to finish
/// is usually what the operator wants. This is process startup under a
/// supervisor: a server that sits silently for 30s before it binds a port is
/// indistinguishable from a hung one, and `systemd`'s `Restart=`/`RestartSec=`
/// already expresses "try again shortly" better than a blocking sleep can.
const WRITER_LEASE_TIMEOUT: Duration = Duration::ZERO;

/// Ordered startup steps, recorded through an injected sink so a test can
/// assert the lease is taken *before* the graph is read instead of inferring
/// it from a timing race between two processes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum StartupStep {
    LeaseAcquired,
    GraphOpened,
    SessionOpened,
}

/// What the operator asked of the write-ahead log, and whether they asked.
///
/// The pair travels together because the two answers mean different things to
/// a graph that cannot carry a log: an explicitly requested level is a startup
/// error (the engine's, naming the reason), while the server's *default* level
/// degrades to `off` so that serving a disk-mode graph does not stop working
/// the day the default changes.
#[derive(Debug, Clone, Copy)]
pub(crate) struct DurabilityRequest {
    pub(crate) level: DurabilityLevel,
    /// False when `level` came from the server's default rather than from
    /// `--durability` or its environment mirror.
    pub(crate) explicit: bool,
}

/// A session ready to serve, plus the write ownership it depends on.
pub(crate) struct StartedGraph {
    pub(crate) session: Session,
    /// The level the session is actually logging at. Equal to the requested
    /// level except where a default was degraded (see [`DurabilityRequest`]) —
    /// the caller must serve and report *this* one, not what it asked for.
    pub(crate) level: DurabilityLevel,
    pub(crate) disposition: OpenDisposition,
    /// The mode the served graph is actually in, read while startup still owns
    /// the graph alone. Read here rather than from a snapshot later because a
    /// snapshot Arc held across a checkpoint turns its `Arc::make_mut` into a
    /// deep clone of the whole graph.
    pub(crate) live_mode: StorageMode,
    /// The mode the graph was in before `--storage` converted it, so startup
    /// can say so. `None` when nothing was converted.
    pub(crate) converted_from: Option<StorageMode>,
    /// `None` for `--readonly`, which publishes nothing and so must not
    /// exclude a writer. Otherwise held for as long as the server serves —
    /// its `Drop` is what releases the lease.
    pub(crate) writer_lease: Option<GraphWriterLease>,
}

/// Acquire write ownership (unless `readonly`), open or create the graph in
/// `requested_mode`, and wrap it in a session logging at `durability`.
///
/// `requested_mode` is the operator's `--storage`, and it means the same thing
/// on both branches: create a missing graph in it, and convert an existing one
/// to it. `None` — no flag — means the checkpoint decides, matching
/// `kglite.open(path)` with no `storage=`. A request with no conversion (either
/// disk direction) fails startup rather than serving a different mode silently.
///
/// `durability` is the operator's `--durability`, and it travels into *both*
/// steps for one reason: recovery. The open is told a log is coming so it does
/// not refuse a sidecar holding commits the checkpoint lacks, and
/// [`Session::open_durable`] then replays exactly those frames before the
/// server binds its port. At `off` neither happens and the open keeps the
/// refusal, which is what makes an operator who turns durability off over a
/// crashed server's graph hear about it instead of serving a graph that is
/// missing committed writes.
///
/// A `--storage` conversion is safe under a log because it is performed in
/// memory, before the session exists: the `.kgl` on disk — and therefore the
/// `checkpoint_lsn` the replay is gated on — is untouched until the first
/// checkpoint, which then writes the converted graph and truncates the log
/// together.
pub(crate) fn start_graph(
    path: &Path,
    requested_mode: Option<StorageMode>,
    readonly: bool,
    durability: DurabilityRequest,
    record: &mut dyn FnMut(StartupStep),
) -> Result<StartedGraph> {
    let writer_lease = if readonly {
        None
    } else {
        let lease = GraphWriterLease::acquire(path, WRITER_LEASE_TIMEOUT).with_context(|| {
            format!(
                "acquiring the writer lease for {}; pass --readonly to serve this graph \
                 alongside its writer",
                path.display()
            )
        })?;
        record(StartupStep::LeaseAcquired);
        Some(lease)
    };
    let opened = open_or_create_graph_in_mode(path, requested_mode, durability.level)
        .with_context(|| format!("opening or creating {}", path.display()))?;
    record(StartupStep::GraphOpened);
    let live_mode = live_storage_mode(&opened.graph);
    // The one refusal that becomes a degrade: a disk graph has no logical WAL
    // at any level, so an operator who asked for one hears the engine's error
    // below, while a graph merely inheriting the server's default is served
    // unlogged rather than refused. Decided here because it is the first point
    // where the *live* mode is known — `--storage` may have converted, and a
    // graph opened without the flag reports whatever it was saved in.
    let level = if durability.level.logs() && !durability.explicit && live_mode == StorageMode::Disk
    {
        tracing::info!(
            default_level = durability.level.name(),
            "disk-mode graph: serving at durability off (a disk graph commits by \
             publishing a generation, so it carries no write-ahead log)"
        );
        DurabilityLevel::Off
    } else {
        durability.level
    };
    // `opened.graph` is the only reference to this graph, which is what
    // `open_durable` requires: a shared Arc would be deep-cloned here and the
    // other holder would keep mutating an unlogged copy.
    let session =
        Session::open_durable(opened.graph, &path.to_string_lossy(), level).map_err(|e| {
            anyhow::anyhow!(
                "opening {} at --durability {}: {e}",
                path.display(),
                level.name()
            )
        })?;
    record(StartupStep::SessionOpened);
    Ok(StartedGraph {
        session,
        level,
        disposition: opened.disposition,
        live_mode,
        converted_from: opened.converted_from,
        writer_lease,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::sync::Arc;
    use std::time::{SystemTime, UNIX_EPOCH};

    /// A level the operator named — what every test here means unless it is
    /// specifically about the default's degrade path.
    fn explicit(level: DurabilityLevel) -> DurabilityRequest {
        DurabilityRequest {
            level,
            explicit: true,
        }
    }

    /// The server's default level, i.e. one nobody typed.
    fn defaulted(level: DurabilityLevel) -> DurabilityRequest {
        DurabilityRequest {
            level,
            explicit: false,
        }
    }

    /// Unique scratch directory, removed on drop. Mirrors `backend.rs`'s
    /// temp-path idiom rather than adding a dev-dependency for one test file.
    struct ScratchDir(PathBuf);

    impl ScratchDir {
        fn new(tag: &str) -> Self {
            let nonce = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("clock after epoch")
                .as_nanos();
            let dir = std::env::temp_dir().join(format!(
                "kglite-bolt-startup-{tag}-{}-{nonce}",
                std::process::id()
            ));
            std::fs::create_dir_all(&dir).expect("create scratch dir");
            Self(dir)
        }

        fn graph_path(&self) -> PathBuf {
            self.0.join("graph.kgl")
        }
    }

    impl Drop for ScratchDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    /// The ordering contract. Mutation evidence: moving the acquisition below
    /// the open records `[GraphOpened, LeaseAcquired]` and fails here.
    #[test]
    fn writable_startup_takes_the_lease_before_opening_the_graph() {
        let scratch = ScratchDir::new("order");
        let mut steps = Vec::new();
        let started = start_graph(
            &scratch.graph_path(),
            Some(StorageMode::Memory),
            false,
            explicit(DurabilityLevel::Off),
            &mut |step| steps.push(step),
        )
        .expect("writable startup on a free path");

        assert_eq!(
            steps,
            vec![
                StartupStep::LeaseAcquired,
                StartupStep::GraphOpened,
                StartupStep::SessionOpened
            ]
        );
        assert!(
            started.writer_lease.is_some(),
            "a writable server must retain the lease it acquired"
        );
    }

    /// A second writable server on the same path is refused, and the refusal
    /// names the holder rather than saying "busy".
    #[test]
    fn second_writable_startup_fails_naming_the_holder() {
        let scratch = ScratchDir::new("contended");
        let path = scratch.graph_path();
        let held = GraphWriterLease::acquire(&path, Duration::ZERO).expect("first writer");

        let mut steps = Vec::new();
        // `expect_err` is unavailable: `StartedGraph` holds a `DirGraph` and a
        // lease, neither of which is `Debug`.
        let error = match start_graph(
            &path,
            Some(StorageMode::Memory),
            false,
            explicit(DurabilityLevel::Off),
            &mut |step| steps.push(step),
        ) {
            Ok(_) => panic!("a second writable server must not open a leased graph"),
            Err(error) => error,
        };

        let message = format!("{error:#}");
        assert!(
            message.contains(&format!("pid {}", std::process::id())),
            "refusal must name the holding process: {message}"
        );
        assert!(
            message.contains("only one process may write a graph at a time"),
            "refusal must explain the constraint: {message}"
        );
        assert!(
            steps.is_empty(),
            "a refused writer must not have touched the graph: {steps:?}"
        );
        drop(held);
    }

    // ── `--storage` on an existing graph ────────────────────────────────────
    //
    // The flag used to be a create-only default: an existing graph was loaded
    // and the requested mode dropped on the floor, so an operator who wrote
    // `--storage mapped` in a unit file got a memory server and no message.
    // Now the file decides when nothing is asked, and an explicit request is
    // either applied or refused.

    /// Save a two-node graph at `path` in `mode`, then drop it.
    fn seed_graph(path: &Path, mode: StorageMode) {
        let mut graph = Arc::new(
            kglite::api::storage::new_dir_graph_in_mode(mode, None).expect("portable mode"),
        );
        kglite::api::io::save_graph(&mut graph, &path.to_string_lossy()).expect("seed save");
    }

    fn start(path: &Path, requested: Option<StorageMode>) -> Result<StartedGraph> {
        start_graph(
            path,
            requested,
            false,
            explicit(DurabilityLevel::Off),
            &mut |_| {},
        )
    }

    #[test]
    fn no_storage_flag_serves_the_mode_the_checkpoint_recorded() {
        let scratch = ScratchDir::new("recorded");
        let path = scratch.graph_path();
        seed_graph(&path, StorageMode::Mapped);

        let started = start(&path, None).expect("startup on a mapped checkpoint");
        assert_eq!(
            started.live_mode,
            StorageMode::Mapped,
            "with no --storage the file decides, and this one recorded mapped"
        );
    }

    #[test]
    fn explicit_matching_storage_proceeds_without_converting() {
        let scratch = ScratchDir::new("agree");
        let path = scratch.graph_path();
        seed_graph(&path, StorageMode::Mapped);

        let started = start(&path, Some(StorageMode::Mapped)).expect("agreeing request");
        assert_eq!(started.live_mode, StorageMode::Mapped);
        assert_eq!(
            started.converted_from, None,
            "nothing was converted, so nothing may be reported as converted"
        );
    }

    #[test]
    fn explicit_mismatching_portable_storage_converts_and_reports_it() {
        for (recorded, requested) in [
            (StorageMode::Memory, StorageMode::Mapped),
            (StorageMode::Mapped, StorageMode::Memory),
        ] {
            let scratch = ScratchDir::new("convert");
            let path = scratch.graph_path();
            seed_graph(&path, recorded);

            let mut steps = Vec::new();
            let started = start_graph(
                &path,
                Some(requested),
                false,
                explicit(DurabilityLevel::Off),
                &mut |step| steps.push(step),
            )
            .expect("portable conversion at startup");

            assert_eq!(
                started.live_mode, requested,
                "the server must serve the mode the operator asked for"
            );
            assert_eq!(
                started.converted_from,
                Some(recorded),
                "a conversion the operator did not see is as bad as an ignored flag"
            );
            assert_eq!(
                steps,
                vec![
                    StartupStep::LeaseAcquired,
                    StartupStep::GraphOpened,
                    StartupStep::SessionOpened
                ],
                "converting must not disturb the acquire-then-open order"
            );
        }
    }

    #[test]
    fn disk_request_on_a_portable_graph_is_refused_with_the_core_reason() {
        let scratch = ScratchDir::new("disk-request");
        let path = scratch.graph_path();
        seed_graph(&path, StorageMode::Memory);

        let error = match start(&path, Some(StorageMode::Disk)) {
            Ok(_) => panic!("a disk request on a `.kgl` must not be silently ignored"),
            Err(error) => format!("{error:#}"),
        };
        assert!(
            error.contains("enable_disk_mode()") && error.contains("directory"),
            "the refusal must carry the core reason and remedy: {error}"
        );

        // The refused startup must not strand the lease it took first.
        start(&path, None).expect("the path stays openable after a refusal");
    }

    /// `--readonly` publishes nothing, so it neither takes nor waits on the
    /// lease: it starts fine beside a live writer.
    #[test]
    fn readonly_startup_succeeds_beside_a_held_lease() {
        let scratch = ScratchDir::new("readonly");
        let path = scratch.graph_path();
        let held = GraphWriterLease::acquire(&path, Duration::ZERO).expect("writer lease");

        let mut steps = Vec::new();
        let started = start_graph(
            &path,
            Some(StorageMode::Memory),
            true,
            explicit(DurabilityLevel::Off),
            &mut |step| steps.push(step),
        )
        .expect("readonly startup beside a live writer");

        assert_eq!(
            steps,
            vec![StartupStep::GraphOpened, StartupStep::SessionOpened]
        );
        assert!(
            started.writer_lease.is_none(),
            "a readonly server must not take write ownership"
        );
        drop(held);
    }

    // ── Durability ──────────────────────────────────────────────────────────

    /// Commit one mutation the way the Bolt backend does — begin, mutate the
    /// working copy, commit — so the write travels the path a durable session
    /// logs.
    fn commit_write(session: &Session, query: &str) {
        use kglite::api::session::{execute_mut, CommitOutcome, ExecuteOptions};
        let params = std::collections::HashMap::new();
        let mut tx = session.begin();
        execute_mut(
            tx.working_mut().expect("a fresh transaction is writable"),
            query,
            &ExecuteOptions::eager(&params),
        )
        .expect("the mutation runs");
        match session.commit(tx, true) {
            CommitOutcome::Committed { .. } => {}
            other => panic!("the commit must publish: {other:?}"),
        }
    }

    fn person_count(session: &Session) -> i64 {
        use kglite::api::session::{execute_read, ExecuteOptions};
        let params = std::collections::HashMap::new();
        let snapshot = session.snapshot();
        let outcome = execute_read(
            &snapshot,
            "MATCH (p:Person) RETURN count(p) AS c",
            &ExecuteOptions::eager(&params),
        )
        .expect("the count runs");
        match outcome.result.rows.first().and_then(|row| row.first()) {
            Some(kglite::api::Value::Int64(c)) => *c,
            other => panic!("expected a count, got {other:?}"),
        }
    }

    /// The headline of the rung, at the layer the server starts from: a
    /// durable session whose process dies without ever checkpointing comes
    /// back with its committed writes, because startup recovers the sidecar
    /// before anything else happens.
    ///
    /// Mutation evidence: passing `DurabilityLevel::Off` for the *first*
    /// session (nothing logged) or for the second (nothing replayed) both
    /// drop the count back to 0.
    #[test]
    fn a_crashed_durable_session_replays_on_the_next_startup() {
        for level in [DurabilityLevel::Full, DurabilityLevel::Normal] {
            let scratch = ScratchDir::new("replay");
            let path = scratch.graph_path();
            seed_graph(&path, StorageMode::Memory);

            let first = start_graph(&path, None, false, explicit(level), &mut |_| {})
                .expect("durable startup on a saved graph");
            commit_write(&first.session, "CREATE (:Person {id: 1, title: 'Zed'})");
            assert_eq!(person_count(&first.session), 1);
            // No checkpoint, no clean shutdown: drop everything exactly as a
            // killed process would, leaving the commit only in the sidecar.
            drop(first);

            let reopened = start_graph(&path, None, false, explicit(level), &mut |_| {})
                .unwrap_or_else(|e| panic!("durable restart at {}: {e:#}", level.name()));
            assert_eq!(
                person_count(&reopened.session),
                1,
                "a committed write must survive a crash-shaped restart at {} with no \
                 checkpoint in between",
                level.name()
            );
        }
    }

    /// The inherited half of the same rule: turning durability off over a
    /// sidecar that still holds commits is refused at startup, not served as a
    /// graph missing them. The refusal is the engine's; what is asserted here
    /// is that the bolt server takes it rather than routing around it.
    #[test]
    fn startup_at_off_refuses_an_unrecovered_sidecar() {
        let scratch = ScratchDir::new("off-refusal");
        let path = scratch.graph_path();
        seed_graph(&path, StorageMode::Memory);

        let durable = start_graph(
            &path,
            None,
            false,
            explicit(DurabilityLevel::Full),
            &mut |_| {},
        )
        .expect("durable startup on a saved graph");
        commit_write(&durable.session, "CREATE (:Person {id: 1, title: 'Zed'})");
        drop(durable);

        let error = match start_graph(
            &path,
            None,
            false,
            explicit(DurabilityLevel::Off),
            &mut |_| {},
        ) {
            Ok(_) => panic!("serving at off would silently drop the logged commit"),
            Err(error) => format!("{error:#}"),
        };
        assert!(
            error.contains("graph.kgl-wal") && error.contains("'full' or 'normal'"),
            "the refusal must name the sidecar and the way out: {error}"
        );

        // Non-vacuity: the same path opens at a logging level, with the write.
        let recovered = start_graph(
            &path,
            None,
            false,
            explicit(DurabilityLevel::Full),
            &mut |_| {},
        )
        .expect("the refusal is about the level, not the path");
        assert_eq!(person_count(&recovered.session), 1);
    }
    /// A disk graph carries no logical log at any level, and after the
    /// default flipped to `normal` that would make *every* disk-mode server
    /// fail to start. The default degrades instead: the graph is served,
    /// unlogged.
    ///
    /// Mutation evidence: passing `explicit(...)` here fails with the engine's
    /// disk refusal, which is the next test.
    #[test]
    fn a_default_level_degrades_to_off_for_a_disk_graph() {
        let scratch = ScratchDir::new("disk-default");
        let path = scratch.0.join("disk-graph");

        let started = start_graph(
            &path,
            Some(StorageMode::Disk),
            false,
            defaulted(DurabilityLevel::Normal),
            &mut |_| {},
        )
        .expect("a disk graph must still be servable once the default logs");

        assert_eq!(started.live_mode, StorageMode::Disk);
        assert_eq!(
            started.level,
            DurabilityLevel::Off,
            "the degraded level is what the caller must serve and report"
        );
        assert_eq!(
            started.session.durability(),
            None,
            "a degraded session must carry no log at all"
        );
    }

    /// The other half of the same rule: a level the operator *typed* is
    /// refused for a disk graph rather than quietly weakened.
    #[test]
    fn an_explicit_level_is_still_refused_for_a_disk_graph() {
        let scratch = ScratchDir::new("disk-explicit");
        let path = scratch.0.join("disk-graph");

        let error = match start_graph(
            &path,
            Some(StorageMode::Disk),
            false,
            explicit(DurabilityLevel::Normal),
            &mut |_| {},
        ) {
            Ok(_) => panic!("an operator who asked for a log must not be given none"),
            Err(error) => format!("{error:#}"),
        };
        assert!(
            error.contains("storage='disk'"),
            "the refusal must name the reason: {error}"
        );
    }
}
