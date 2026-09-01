//! [`GraphState`]'s workspace-graph lifecycle: change binding, dirty
//! tagging, the lazy freshness rebuild, and prepare/commit publication.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use anyhow::Result;
use kglite::api::KnowledgeGraph;

use crate::tools::*;

impl GraphState {
    /// Tag the installed workspace graph as needing rebuild. Called from the
    /// watch callback; non-blocking (two short lock-protected reads/writes).
    /// The observed root/revision identity permits binding to the latest
    /// compatible generation while preventing a deferred event from silently
    /// crossing into another source snapshot.
    /// The actual rebuild happens lazily on the next tool call via
    /// [`ensure_workspace_graph_fresh`].
    pub(crate) fn bind_workspace_graph_changes(
        &self,
        observed: &WorkspaceGraphTarget,
        changed_paths: &[PathBuf],
    ) -> WorkspaceGraphEnqueue {
        if changed_paths.is_empty() {
            return WorkspaceGraphEnqueue::Empty;
        }
        let changed_paths = changed_paths
            .iter()
            .filter_map(|path| absolute_lexical_path(path))
            .filter(|path| path.starts_with(&observed.root))
            .collect::<BTreeSet<_>>();
        if changed_paths.is_empty() {
            return WorkspaceGraphEnqueue::Empty;
        }

        // Bind against the latest compatible generation under active ->
        // pending. There is no finite snapshot/retry window for activation to
        // race through.
        let active = read_lock(&self.inner);
        let Some(current) = active.as_ref().and_then(ActiveGraph::workspace_target) else {
            return WorkspaceGraphEnqueue::IncompatibleSource;
        };
        if !observed.same_source(&current) {
            return WorkspaceGraphEnqueue::IncompatibleSource;
        }
        tracing::debug!(
            target = %current.root.display(),
            revisions = ?current.revisions,
            generation = current.generation,
            changed_paths = changed_paths.len(),
            "workspace graph tagged for rebuild"
        );
        let mut pending = write_lock(&self.pending_rebuild);
        merge_ready_pending(&mut pending, &current, changed_paths);
        WorkspaceGraphEnqueue::Enqueued
    }

    #[cfg(test)]
    pub(crate) fn tag_workspace_graph_dirty(&self, changed_paths: &[PathBuf]) {
        let Some(target) = self.workspace_target_receipt() else {
            return;
        };
        let _ = self.bind_workspace_graph_changes(&target, changed_paths);
    }

    /// Rebuild the workspace graph if the watcher tagged it dirty since the
    /// last call. Called by each MCP tool entry that reads the graph
    /// (cypher_query / graph_overview / save_graph / read_code_source
    /// / explore). No-op when nothing's pending.
    ///
    /// **Failure policy.** A failed rebuild must not silently serve a
    /// stale graph forever: the dirty marker is restored so the next
    /// tool call retries, and the error is recorded on `rebuild_status`
    /// (surfaced next to the built-at identity in graph_overview /
    /// cypher_query output). To avoid a hot retry loop when the source
    /// dir is permanently broken, after
    /// [`MAX_CONSECUTIVE_REBUILD_FAILURES`] consecutive failures for the
    /// same target the marker becomes dormant — the stale graph keeps being
    /// served (error still surfaced), its paths remain retained, and the next
    /// retry happens only when a fresh FS event re-arms the target.
    pub fn ensure_workspace_graph_fresh(&self) {
        // Ordinary loaded/in-memory graphs never participate in workspace
        // rebuilds and must not contend on workspace lifecycle state.
        if self.workspace_mode.is_none() {
            return;
        }

        // `pending_rebuild` is empty while an owner prepares off-lock. Keep
        // later freshness callers behind that owner until it has installed
        // the new generation or published a typed failure snapshot.
        let _rebuild_owner = self.rebuild_gate.enter();
        let pending_rebuild = {
            let mut pending = write_lock(&self.pending_rebuild);
            if pending.as_ref().is_some_and(|pending| pending.ready) {
                pending.take()
            } else {
                None
            }
        };
        let Some(pending_rebuild) = pending_rebuild else {
            return;
        };
        let target = pending_rebuild.target.clone();
        tracing::info!(
            target = %target.root.display(),
            revisions = ?target.revisions,
            generation = target.generation,
            changed_paths = pending_rebuild.changed_paths.len(),
            "rebuilding workspace graph (lazy, FS changed)"
        );
        let changes =
            WorkspaceGraphChanges::Changed(pending_rebuild.changed_paths.iter().cloned().collect());
        match self.prepare_workspace_graph(&target.root, target.revisions.as_deref(), changes) {
            Ok(prepared) => match self.commit_workspace_rebuild(prepared, &pending_rebuild) {
                WorkspaceRebuildCommit::Installed => {
                    *write_lock(&self.rebuild_status) = RebuildStatus::default();
                }
                WorkspaceRebuildCommit::RequeuedCompatible => {
                    tracing::debug!(
                        target = %target.root.display(),
                        generation = target.generation,
                        "requeued consumed paths after compatible activation superseded rebuild"
                    );
                }
                WorkspaceRebuildCommit::DiscardedIncompatible => {
                    tracing::debug!(
                        target = %target.root.display(),
                        generation = target.generation,
                        "discarding workspace rebuild prepared for an incompatible graph"
                    );
                }
            },
            Err(e) => match self.record_and_restore_current_rebuild(pending_rebuild, &e) {
                WorkspaceRebuildFailureDisposition::Current(failures) => {
                    tracing::warn!(error = %e, "lazy workspace graph rebuild failed");
                    if failures >= MAX_CONSECUTIVE_REBUILD_FAILURES {
                        tracing::warn!(
                            target = %target.root.display(),
                            failures,
                            "workspace graph rebuild keeps failing — serving the stale \
                             graph; retrying only on the next FS event"
                        );
                    }
                }
                WorkspaceRebuildFailureDisposition::RequeuedCompatible => {
                    tracing::debug!(
                        target = %target.root.display(),
                        generation = target.generation,
                        "requeued consumed paths after compatible activation superseded failed rebuild"
                    );
                }
                WorkspaceRebuildFailureDisposition::DiscardedIncompatible => {
                    tracing::debug!(
                        target = %target.root.display(),
                        generation = target.generation,
                        "discarding workspace rebuild failure for an incompatible graph"
                    );
                }
            },
        }
    }

    /// Snapshot the failed rebuild for the currently installed workspace
    /// generation. Call after [`Self::ensure_workspace_graph_fresh`] when a
    /// route must reject stale evidence instead of rendering the legacy
    /// warning and continuing.
    pub(crate) fn workspace_rebuild_failure(&self) -> Option<WorkspaceRebuildFailureSnapshot> {
        let active = read_lock(&self.inner);
        let active_target = active.as_ref().and_then(ActiveGraph::workspace_target)?;
        let status = read_lock(&self.rebuild_status);
        if status.failed_target.as_ref() != Some(&active_target) {
            return None;
        }
        Some(WorkspaceRebuildFailureSnapshot {
            reason: if status.consecutive_failures >= MAX_CONSECUTIVE_REBUILD_FAILURES {
                WorkspaceRebuildFailureReason::HotFail
            } else {
                WorkspaceRebuildFailureReason::RebuildFailed
            },
            message: status.last_error.clone()?,
            failed_at: status.failed_at?,
            consecutive_failures: status.consecutive_failures,
            retry_limit: MAX_CONSECUTIVE_REBUILD_FAILURES,
        })
    }

    /// Ask the configured producer for a workspace graph without publishing
    /// it. Expensive parsing and summary generation happen here, outside the
    /// mcp-methods activation commit lock.
    pub(crate) fn prepare_workspace_graph(
        &self,
        root: &Path,
        revisions: Option<&[String]>,
        changes: WorkspaceGraphChanges,
    ) -> Result<PreparedWorkspaceGraph> {
        let Some(hooks) = &self.workspace_graph_hooks else {
            anyhow::bail!(NO_BUILDER_MSG);
        };
        let Some(mode) = self.workspace_mode else {
            anyhow::bail!("workspace-graph build requested outside a workspace/watch mode");
        };
        let request = WorkspaceGraphRequest::new(
            root.to_path_buf(),
            revisions.map(|revs| revs.to_vec()),
            mode,
            changes,
        );
        let result = (hooks.build)(request)
            .map_err(|e| anyhow::anyhow!("workspace-graph build hook failed: {e}"))?;
        let (graph, revisions) = result.into_parts();
        let mut kg = KnowledgeGraph::from_arc(graph);
        // A producer rebuild is a graph swap like any other: re-apply the
        // state's bound embedder so `text_score()` survives it.
        self.apply_bound_embedder(&mut kg);
        let active = ActiveGraph {
            kg,
            source_path: None,
            ownership: None,
            root: Some(root.to_path_buf()),
            revs: revisions,
            built_at: SystemTime::now(),
            generation: 0,
        };
        let summary = activation_summary_for_active(&active);
        Ok(PreparedWorkspaceGraph { active, summary })
    }

    /// Publish one already-prepared graph and return the summary computed from
    /// that exact artifact. Keep this to a single slot swap: it may run inside
    /// mcp-methods' generation commit boundary.
    pub(crate) fn commit_workspace_graph(
        &self,
        mut prepared: PreparedWorkspaceGraph,
    ) -> Option<String> {
        let mut slot = write_lock(&self.inner);
        let mut pending = write_lock(&self.pending_rebuild);
        prepared.active.generation = slot
            .as_ref()
            .map_or(1, |active| active.generation.saturating_add(1));
        let installed_target = prepared
            .active
            .workspace_target()
            .expect("workspace activation always has a root");
        *slot = Some(prepared.active);
        match pending.as_mut() {
            Some(existing) if existing.target.same_source(&installed_target) => {
                existing.target = installed_target;
                existing.ready = true;
            }
            Some(_) => *pending = None,
            None => {}
        }
        prepared.summary
    }

    /// Publish a lazy rebuild only if its exact source generation remains
    /// installed. If compatible activation superseded it, atomically requeue
    /// the consumed paths against the latest generation instead of losing
    /// them; incompatible source changes discard the obsolete work.
    pub(crate) fn commit_workspace_rebuild(
        &self,
        mut prepared: PreparedWorkspaceGraph,
        consumed: &PendingWorkspaceRebuild,
    ) -> WorkspaceRebuildCommit {
        let mut active_slot = write_lock(&self.inner);
        let Some(current) = active_slot.as_ref().and_then(ActiveGraph::workspace_target) else {
            return WorkspaceRebuildCommit::DiscardedIncompatible;
        };
        if current != consumed.target {
            if !current.same_source(&consumed.target) {
                return WorkspaceRebuildCommit::DiscardedIncompatible;
            }
            let mut pending = write_lock(&self.pending_rebuild);
            merge_ready_pending(
                &mut pending,
                &current,
                consumed.changed_paths.iter().cloned(),
            );
            return WorkspaceRebuildCommit::RequeuedCompatible;
        }
        prepared.active.generation = consumed.target.generation.saturating_add(1);
        let installed_target = prepared
            .active
            .workspace_target()
            .expect("workspace rebuild always has a root");
        *active_slot = Some(prepared.active);

        let mut pending = write_lock(&self.pending_rebuild);
        if let Some(pending) = pending
            .as_mut()
            .filter(|pending| pending.target == consumed.target)
        {
            pending.target = installed_target;
        }
        WorkspaceRebuildCommit::Installed
    }

    pub(crate) fn workspace_target_receipt(&self) -> Option<WorkspaceGraphTarget> {
        read_lock(&self.inner)
            .as_ref()
            .and_then(ActiveGraph::workspace_target)
    }

    #[cfg(test)]
    pub(crate) fn active_workspace_target(&self) -> Option<WorkspaceGraphTarget> {
        self.workspace_target_receipt()
    }

    /// Record a rebuild failure and restore its consumed paths only while its
    /// source graph remains current. Holding active → pending → status closes
    /// activation races and preserves the global lifecycle lock order.
    pub(crate) fn record_and_restore_current_rebuild(
        &self,
        mut failed: PendingWorkspaceRebuild,
        error: &anyhow::Error,
    ) -> WorkspaceRebuildFailureDisposition {
        let active = read_lock(&self.inner);
        let Some(current) = active.as_ref().and_then(ActiveGraph::workspace_target) else {
            return WorkspaceRebuildFailureDisposition::DiscardedIncompatible;
        };
        if current != failed.target {
            if !current.same_source(&failed.target) {
                return WorkspaceRebuildFailureDisposition::DiscardedIncompatible;
            }
            let mut pending = write_lock(&self.pending_rebuild);
            merge_ready_pending(&mut pending, &current, failed.changed_paths.iter().cloned());
            return WorkspaceRebuildFailureDisposition::RequeuedCompatible;
        }
        let mut pending = write_lock(&self.pending_rebuild);
        let mut status = write_lock(&self.rebuild_status);
        if status.failed_target.as_ref() == Some(&failed.target) {
            status.consecutive_failures += 1;
        } else {
            status.consecutive_failures = 1;
            status.failed_target = Some(failed.target.clone());
        }
        status.last_error = Some(error.to_string());
        status.failed_at = Some(SystemTime::now());
        let failures = status.consecutive_failures;
        match pending.as_mut() {
            Some(newer) if newer.target == failed.target => {
                newer.changed_paths.extend(failed.changed_paths);
            }
            None => {
                failed.ready = failures < MAX_CONSECUTIVE_REBUILD_FAILURES;
                *pending = Some(failed);
            }
            _ => {}
        }
        WorkspaceRebuildFailureDisposition::Current(failures)
    }

    /// Build and publish outside activation transactions (boot and lazy-watch
    /// paths). Activation uses the prepare/commit pair directly.
    pub fn build_workspace_graph(&self, root: &Path, revisions: Option<&[String]>) -> Result<()> {
        let prepared =
            self.prepare_workspace_graph(root, revisions, WorkspaceGraphChanges::Full)?;
        self.commit_workspace_graph(prepared);
        Ok(())
    }

    /// Root of the exact workspace graph currently installed.
    #[cfg(test)]
    pub(crate) fn active_workspace_root(&self) -> Option<std::path::PathBuf> {
        read_lock(&self.inner)
            .as_ref()
            .and_then(|active| active.root.clone())
    }

    #[cfg(test)]
    pub(crate) fn active_workspace_revisions(&self) -> Option<Vec<String>> {
        read_lock(&self.inner)
            .as_ref()
            .and_then(|active| active.revs.clone())
    }
}
