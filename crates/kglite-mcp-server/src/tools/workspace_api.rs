//! Public workspace-graph extension surface: the request/result/relevance
//! types an embedding binary sees, and the hooks it injects.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::Result;

/// Refusal surfaced when a workspace mode has no graph producer configured.
pub(crate) const NO_BUILDER_MSG: &str =
    "workspace-graph building is not configured in this binary. \
Embed kglite-mcp-server and inject WorkspaceGraphHooks through \
ServerExtensions::with_workspace_graph. For source-code graphs, use codingest-mcp. \
Reading existing .kgl graphs with --graph remains available.";

/// Server mode that requested a workspace graph.
///
/// Producers own all domain policy derived from this value, including which
/// files to ingest. KGLite does not assume source languages or documentation
/// behavior.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum WorkspaceGraphMode {
    /// Clone-backed `--workspace` mode.
    Workspace,
    /// Manifest-declared local workspace activated through `set_root_dir`.
    LocalWorkspace,
    /// Fixed-directory `--watch` mode.
    Watch,
}

/// Change scope attached to one workspace graph producer request.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum WorkspaceGraphChanges {
    /// Build from the complete source view. Used for boot, activation,
    /// revision, and other explicitly requested builds.
    Full,
    /// Relevant watcher paths accepted since the last successful rebuild.
    ///
    /// Paths are absolute, non-empty, sorted, deduplicated, and already
    /// filtered for the active root and producer relevance. A path can name a
    /// deleted file. This is a parsing hint only: the producer must still
    /// return a complete replacement graph.
    Changed(Vec<PathBuf>),
}

/// One producer request for a workspace graph.
pub struct WorkspaceGraphRequest {
    root: PathBuf,
    revisions: Option<Vec<String>>,
    mode: WorkspaceGraphMode,
    changes: WorkspaceGraphChanges,
}

impl WorkspaceGraphRequest {
    pub(crate) fn new(
        root: PathBuf,
        revisions: Option<Vec<String>>,
        mode: WorkspaceGraphMode,
        changes: WorkspaceGraphChanges,
    ) -> Self {
        Self {
            root,
            revisions,
            mode,
            changes,
        }
    }

    /// Canonical source root to build.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Resolved revisions requested by activation, oldest to newest.
    /// `None` means the producer should build its ordinary working-tree view.
    pub fn revisions(&self) -> Option<&[String]> {
        self.revisions.as_deref()
    }

    /// Workspace mode that originated the request.
    pub fn mode(&self) -> WorkspaceGraphMode {
        self.mode
    }

    /// Whether this is a complete build or a watcher-triggered rebuild with a
    /// filtered changed-path hint.
    pub fn changes(&self) -> &WorkspaceGraphChanges {
        &self.changes
    }
}

/// Completed graph plus the canonical revision labels represented by it.
pub struct WorkspaceGraphResult {
    graph: Arc<kglite::api::DirGraph>,
    revisions: Option<Vec<String>>,
}

impl WorkspaceGraphResult {
    /// Return a normal working-tree graph.
    pub fn new(graph: Arc<kglite::api::DirGraph>) -> Self {
        Self {
            graph,
            revisions: None,
        }
    }

    /// Return a graph spanning canonicalized revision labels.
    pub fn with_revisions(graph: Arc<kglite::api::DirGraph>, revisions: Vec<String>) -> Self {
        Self {
            graph,
            revisions: Some(revisions),
        }
    }

    pub(crate) fn into_parts(self) -> (Arc<kglite::api::DirGraph>, Option<Vec<String>>) {
        (self.graph, self.revisions)
    }
}

/// Borrowed watch-change context passed to the producer's relevance policy.
pub struct WorkspaceGraphRelevance<'a> {
    path: &'a Path,
    mode: WorkspaceGraphMode,
}

impl<'a> WorkspaceGraphRelevance<'a> {
    pub(crate) fn new(path: &'a Path, mode: WorkspaceGraphMode) -> Self {
        Self { path, mode }
    }

    /// Changed path reported by the watcher.
    pub fn path(&self) -> &'a Path {
        self.path
    }

    /// Workspace mode whose active graph would be rebuilt.
    pub fn mode(&self) -> WorkspaceGraphMode {
        self.mode
    }
}

/// Unified plain/revision-set workspace graph build closure.
pub type WorkspaceGraphBuildFn =
    dyn Fn(WorkspaceGraphRequest) -> Result<WorkspaceGraphResult, String> + Send + Sync;

/// Producer-owned watch relevance policy.
pub type WorkspaceGraphRelevanceFn =
    dyn for<'a> Fn(WorkspaceGraphRelevance<'a>) -> bool + Send + Sync;

/// Generic workspace-graph lifecycle extension for embedding binaries.
pub struct WorkspaceGraphHooks {
    /// Build the graph requested by KGLite. The producer owns revision
    /// canonicalization and all domain-specific ingestion policy.
    pub build: Box<WorkspaceGraphBuildFn>,
    /// Return whether a changed path can affect the active graph.
    pub is_relevant: Box<WorkspaceGraphRelevanceFn>,
}
