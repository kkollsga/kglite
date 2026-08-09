//! Workspace activation transaction: the prepare/commit adapter that lets
//! a superseded graph build be discarded before it becomes visible.

use std::sync::Arc;

use mcp_methods::server::{
    ActivationBuild, ActivationRequest, ActivationTransactionHook, PreparedActivation,
};

use crate::tools::GraphState;
use crate::*;

/// Request-scoped prepare/commit adapter shared by clone-backed and local
/// workspaces. Expensive producer work runs before publication; mcp-methods
/// invokes the returned commit closure only while this request is current.
pub(crate) fn workspace_activation_transaction(
    graph_state: &GraphState,
) -> ActivationTransactionHook {
    let state = graph_state.clone();
    Arc::new(move |request: &ActivationRequest| {
        tracing::info!(
            activation_id = %request.id(),
            root = %request.path().display(),
            build = ?request.build(),
            "preparing workspace graph activation"
        );
        if matches!(request.build(), ActivationBuild::Reuse) {
            let summary = state
                .reusable_activation_summary(request.path())
                .or_else(|| state.no_builder_summary());
            return Ok(PreparedActivation::summary(summary));
        }
        if !state.has_workspace_graph_builder() {
            tracing::warn!(
                activation_id = %request.id(),
                "no workspace-graph producer injected; source tools only"
            );
            return Ok(PreparedActivation::summary(state.no_builder_summary()));
        }
        let revisions = match request.build() {
            ActivationBuild::Revisions(revisions) => Some(revisions.as_slice()),
            ActivationBuild::Plain => None,
            ActivationBuild::Reuse => unreachable!("reuse returned above"),
        };
        let prepared = state.prepare_workspace_graph(
            request.path(),
            revisions,
            WorkspaceGraphChanges::Full,
        )?;
        let commit_state = state.clone();
        Ok(PreparedActivation::new(move || {
            Ok(commit_state.commit_workspace_graph(prepared))
        }))
    })
}

#[cfg(test)]
mod activation_transaction_tests {
    use super::*;
    use std::path::Path;
    use std::sync::Arc;

    use anyhow::Result;

    use crate::tools::GraphState;

    use kglite::api::storage::{new_dir_graph_in_mode, StorageMode};
    use mcp_methods::server::RevsRequest;
    use std::process::Command;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Barrier;

    fn graph_result(request: WorkspaceGraphRequest) -> Result<WorkspaceGraphResult, String> {
        let mut graph = new_dir_graph_in_mode(StorageMode::Memory, None)?;
        let params = std::collections::HashMap::new();
        let options = kglite::api::session::ExecuteOptions::eager(&params);
        kglite::api::session::execute_mut(&mut graph, "CREATE (:File {id:'fixture.rs'})", &options)
            .map_err(|error| error.to_string())?;
        let graph = Arc::new(graph);
        Ok(match request.revisions() {
            Some(revisions) => WorkspaceGraphResult::with_revisions(graph, revisions.to_vec()),
            None => WorkspaceGraphResult::new(graph),
        })
    }

    fn local_state(hooks: WorkspaceGraphHooks) -> GraphState {
        GraphState::new(Some(WorkspaceGraphMode::LocalWorkspace))
            .with_workspace_graph(Some(Arc::new(hooks)))
    }

    fn run_git(repo: &Path, args: &[&str]) {
        let status = Command::new("git")
            .arg("-C")
            .arg(repo)
            .args(args)
            .status()
            .expect("run git");
        assert!(status.success(), "git {args:?} failed");
    }

    #[test]
    fn newer_root_commits_while_older_preparation_is_discarded() {
        let temp = tempfile::tempdir().expect("tempdir");
        let slow_root = temp.path().join("slow");
        let fast_root = temp.path().join("fast");
        std::fs::create_dir_all(&slow_root).expect("slow root");
        std::fs::create_dir_all(&fast_root).expect("fast root");
        let slow_root = slow_root.canonicalize().expect("canonical slow");
        let fast_root = fast_root.canonicalize().expect("canonical fast");
        let slow_entered = Arc::new(Barrier::new(2));
        let release_slow = Arc::new(Barrier::new(2));
        let hooks = WorkspaceGraphHooks {
            build: Box::new({
                let slow_root = slow_root.clone();
                let slow_entered = slow_entered.clone();
                let release_slow = release_slow.clone();
                move |request| {
                    if request.root() == slow_root {
                        slow_entered.wait();
                        release_slow.wait();
                    }
                    graph_result(request)
                }
            }),
            is_relevant: Box::new(|_| true),
        };
        let state = local_state(hooks);
        let workspace =
            local_workspace(temp.path().to_path_buf(), &state, None).expect("workspace");

        let slow_workspace = workspace.clone();
        let slow_for_thread = slow_root.clone();
        let slow = std::thread::spawn(move || slow_workspace.set_root_dir(&slow_for_thread, None));
        slow_entered.wait();
        let fast_output = workspace.set_root_dir(&fast_root, None);
        release_slow.wait();
        let slow_output = slow.join().expect("slow activation");

        assert!(fast_output.contains(&fast_root.display().to_string()));
        assert!(fast_output.contains("Graph ready: 1 nodes"));
        assert!(slow_output.contains("superseded by request 2"));
        assert_eq!(state.active_workspace_root(), Some(fast_root));
    }

    #[test]
    fn same_root_revision_activation_supersedes_plain_preparation() {
        let temp = tempfile::tempdir().expect("tempdir");
        let repo = temp.path().join("repo");
        std::fs::create_dir_all(&repo).expect("repo");
        run_git(&repo, &["init", "-q"]);
        run_git(&repo, &["config", "user.email", "test@example.com"]);
        run_git(&repo, &["config", "user.name", "KGLite Test"]);
        std::fs::write(repo.join("fixture.rs"), "fn fixture() {}\n").expect("fixture");
        run_git(&repo, &["add", "fixture.rs"]);
        run_git(&repo, &["commit", "-q", "-m", "fixture"]);
        let repo = repo.canonicalize().expect("canonical repo");
        let plain_entered = Arc::new(Barrier::new(2));
        let release_plain = Arc::new(Barrier::new(2));
        let hooks = WorkspaceGraphHooks {
            build: Box::new({
                let plain_entered = plain_entered.clone();
                let release_plain = release_plain.clone();
                move |request| {
                    if request.revisions().is_none() {
                        plain_entered.wait();
                        release_plain.wait();
                    }
                    graph_result(request)
                }
            }),
            is_relevant: Box::new(|_| true),
        };
        let state = local_state(hooks);
        let workspace = local_workspace(repo.clone(), &state, None).expect("workspace");

        let plain_workspace = workspace.clone();
        let plain_root = repo.clone();
        let plain = std::thread::spawn(move || plain_workspace.set_root_dir(&plain_root, None));
        plain_entered.wait();
        let revisions = RevsRequest::List(vec!["HEAD".to_string()]);
        let revision_output = workspace.set_root_dir(&repo, Some(&revisions));
        release_plain.wait();
        let plain_output = plain.join().expect("plain activation");

        assert!(revision_output.contains("revs: HEAD"));
        assert!(revision_output.contains("revision 'HEAD'"));
        assert!(plain_output.contains("superseded by request 2"));
        assert_eq!(state.active_workspace_root(), Some(repo));
        assert_eq!(
            state.active_workspace_revisions(),
            Some(vec!["HEAD".into()])
        );
    }

    #[test]
    fn failed_current_preparation_preserves_committed_graph() {
        let temp = tempfile::tempdir().expect("tempdir");
        let good_root = temp.path().join("good");
        let broken_root = temp.path().join("broken");
        std::fs::create_dir_all(&good_root).expect("good root");
        std::fs::create_dir_all(&broken_root).expect("broken root");
        let good_root = good_root.canonicalize().expect("canonical good");
        let broken_root = broken_root.canonicalize().expect("canonical broken");
        let hooks = WorkspaceGraphHooks {
            build: Box::new({
                let broken_root = broken_root.clone();
                move |request| {
                    if request.root() == broken_root {
                        return Err("builder rejected broken root".to_string());
                    }
                    graph_result(request)
                }
            }),
            is_relevant: Box::new(|_| true),
        };
        let state = local_state(hooks);
        let workspace =
            local_workspace(temp.path().to_path_buf(), &state, None).expect("workspace");

        let good_output = workspace.set_root_dir(&good_root, None);
        let broken_output = workspace.set_root_dir(&broken_root, None);

        assert!(good_output.contains("Graph ready: 1 nodes"));
        assert!(broken_output.contains("failed during preparation"));
        assert!(broken_output.contains("builder rejected broken root"));
        assert_eq!(state.active_workspace_root(), Some(good_root));
    }

    #[test]
    fn same_plain_target_reuses_the_committed_graph_and_summary() {
        let temp = tempfile::tempdir().expect("tempdir");
        let root = temp.path().join("root");
        std::fs::create_dir_all(&root).expect("root");
        let root = root.canonicalize().expect("canonical root");
        let builds = Arc::new(AtomicUsize::new(0));
        let hooks = WorkspaceGraphHooks {
            build: Box::new({
                let builds = builds.clone();
                move |request| {
                    builds.fetch_add(1, Ordering::SeqCst);
                    graph_result(request)
                }
            }),
            is_relevant: Box::new(|_| true),
        };
        let state = local_state(hooks);
        let workspace = local_workspace(root.clone(), &state, None).expect("workspace");

        let first = workspace.set_root_dir(&root, None);
        let reused = workspace.set_root_dir(&root, None);

        assert!(first.contains("Graph ready: 1 nodes"));
        assert!(reused.contains("build skipped"));
        assert!(reused.contains("Graph ready: 1 nodes"));
        assert_eq!(builds.load(Ordering::SeqCst), 1);
        assert_eq!(state.active_workspace_root(), Some(root));
    }
}
