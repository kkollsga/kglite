//! Graph startup: take cross-process write ownership of the graph path, then
//! open it.
//!
//! Split out of `main` so the acquire-then-open *order* is testable. Ordering
//! is the whole point of the lease: acquiring it after the read is a guard
//! that lets the very race it exists to stop happen first.

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};

use kglite::api::io::{open_or_create_graph, GraphWriterLease, OpenDisposition};
use kglite::api::storage::StorageMode;
use kglite::api::DirGraph;

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
}

/// A graph ready to serve, plus the write ownership it depends on.
pub(crate) struct StartedGraph {
    pub(crate) graph: Arc<DirGraph>,
    pub(crate) disposition: OpenDisposition,
    /// `None` for `--readonly`, which publishes nothing and so must not
    /// exclude a writer. Otherwise held for as long as the server serves —
    /// its `Drop` is what releases the lease.
    pub(crate) writer_lease: Option<GraphWriterLease>,
}

/// Acquire write ownership (unless `readonly`), then open or create the graph.
pub(crate) fn start_graph(
    path: &Path,
    create_mode: Option<StorageMode>,
    readonly: bool,
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
    let opened = open_or_create_graph(path, create_mode)
        .with_context(|| format!("opening or creating {}", path.display()))?;
    record(StartupStep::GraphOpened);
    Ok(StartedGraph {
        graph: opened.graph,
        disposition: opened.disposition,
        writer_lease,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

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
            &mut |step| steps.push(step),
        )
        .expect("writable startup on a free path");

        assert_eq!(
            steps,
            vec![StartupStep::LeaseAcquired, StartupStep::GraphOpened]
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
        let error = match start_graph(&path, Some(StorageMode::Memory), false, &mut |step| {
            steps.push(step)
        }) {
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

    /// `--readonly` publishes nothing, so it neither takes nor waits on the
    /// lease: it starts fine beside a live writer.
    #[test]
    fn readonly_startup_succeeds_beside_a_held_lease() {
        let scratch = ScratchDir::new("readonly");
        let path = scratch.graph_path();
        let held = GraphWriterLease::acquire(&path, Duration::ZERO).expect("writer lease");

        let mut steps = Vec::new();
        let started = start_graph(&path, Some(StorageMode::Memory), true, &mut |step| {
            steps.push(step)
        })
        .expect("readonly startup beside a live writer");

        assert_eq!(steps, vec![StartupStep::GraphOpened]);
        assert!(
            started.writer_lease.is_none(),
            "a readonly server must not take write ownership"
        );
        drop(held);
    }
}
