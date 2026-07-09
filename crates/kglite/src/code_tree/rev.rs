//! Build a code graph from a git revision without disturbing the working tree.
//!
//! `git archive <rev>` streams a tar of exactly the **tracked** files at that
//! revision — it never touches `HEAD` or the working tree, and it only emits
//! *committed* files, so untracked junk and `.gitignore`d paths are absent for
//! free (a superset of `.gitignore` semantics). We extract that tar into a
//! `tempfile::tempdir()` and run the ordinary build pipeline
//! ([`crate::code_tree::builder::run_with_options`]) over it unchanged — zero
//! parser changes, zero walk changes. The existing `manifest::walk_filter`
//! still runs on the extracted tree, so any *committed* `node_modules` /
//! vendored build output is dropped on top.
//!
//! Reuses the established git-shelling convention from `repo.rs`
//! (`std::process::Command`, list args, no shell).

use crate::graph::dir_graph::DirGraph;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::Arc;

/// Build a code graph from `src_dir` as it existed at git revision `rev`.
///
/// `repo_root` overrides repo-root resolution (defaults to `git -C <src_dir>
/// rev-parse --show-toplevel`). When `src_dir` is a subdirectory of the repo
/// root, the build is scoped to the matching subdirectory of the extracted
/// snapshot, mirroring how a working-tree build of that subdir would behave.
#[allow(clippy::too_many_arguments)]
pub fn archive_and_build(
    src_dir: &Path,
    rev: &str,
    repo_root: Option<&Path>,
    verbose: bool,
    include_tests: bool,
    save_to: Option<&Path>,
    max_loc_per_file: Option<usize>,
    include_docs: bool,
) -> Result<Arc<DirGraph>, String> {
    // Resolve the repo root: an explicit `repo_root` wins, else ask git.
    let repo_root = match repo_root {
        Some(r) => r.to_path_buf(),
        None => resolve_repo_root(src_dir)?,
    };

    // Validate the rev resolves to a commit before doing any work — a clean
    // Python error on a bad rev / non-git dir, never a panic or empty graph.
    let sha = verify_rev(&repo_root, rev)?;

    // Materialize the tracked tree at `rev` into a throwaway tempdir. The
    // `TempDir` guard cleans up on drop, including on any `?` bail below.
    // A *visible* prefix is load-bearing: `tempfile::tempdir()` names dirs
    // `.tmpXXXX` (leading dot), and the builder's `walk_filter` skips hidden
    // directories at any depth — including the walk root — which would yield
    // an empty graph. A non-dot prefix keeps the snapshot root walkable.
    let tmp = tempfile::Builder::new()
        .prefix("kglite-rev-")
        .tempdir()
        .map_err(|e| format!("could not create tempdir: {}", e))?;
    archive_into(&repo_root, rev, tmp.path())?;

    // Scope the build to the same relative subpath the caller pointed at, so
    // `build("/repo/src", rev=…)` builds `src` of the snapshot, not the whole
    // repo. Falls back to the snapshot root when `src_dir` is the repo root or
    // does not resolve under it.
    let build_input = rebase_input(src_dir, &repo_root, tmp.path());

    let graph = crate::code_tree::builder::run_with_options(
        &build_input,
        verbose,
        include_tests,
        save_to,
        max_loc_per_file,
        include_docs,
    )?;

    stamp_rev_provenance(graph, rev, &sha, &repo_root)
}

/// `git -C <src_dir> rev-parse --show-toplevel` → the repo's work-tree root.
/// A non-git directory (or missing path) surfaces as a clean error.
fn resolve_repo_root(src_dir: &Path) -> Result<PathBuf, String> {
    let out = Command::new("git")
        .arg("-C")
        .arg(src_dir)
        .args(["rev-parse", "--show-toplevel"])
        .output()
        .map_err(|e| format!("git command failed: {}", e))?;
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr).trim().to_string();
        return Err(format!(
            "not a git repository: {} (pass repo_root= if the git root is elsewhere){}",
            src_dir.display(),
            if stderr.is_empty() {
                String::new()
            } else {
                format!(" — git said: {}", stderr)
            }
        ));
    }
    let root = String::from_utf8_lossy(&out.stdout).trim().to_string();
    Ok(PathBuf::from(root))
}

/// `git -C <repo_root> rev-parse --verify <rev>^{commit}` → the full SHA.
/// Rejects tags/branches/shas that don't resolve to a commit with a clear
/// message (used for both the bad-rev and non-git-dir cases).
fn verify_rev(repo_root: &Path, rev: &str) -> Result<String, String> {
    let out = Command::new("git")
        .arg("-C")
        .arg(repo_root)
        .args(["rev-parse", "--verify"])
        .arg(format!("{}^{{commit}}", rev))
        .output()
        .map_err(|e| format!("git command failed: {}", e))?;
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr).trim().to_string();
        return Err(format!(
            "could not resolve git revision {:?} in {}: {}",
            rev,
            repo_root.display(),
            if stderr.is_empty() {
                "unknown revision".to_string()
            } else {
                stderr
            }
        ));
    }
    Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

/// Stream `git archive --format=tar <rev>` into `tar -x -C <dest>`. Both run
/// concurrently over a pipe (bounded memory — the tar is never buffered
/// whole); list args, no shell.
fn archive_into(repo_root: &Path, rev: &str, dest: &Path) -> Result<(), String> {
    let mut archive = Command::new("git")
        .arg("-C")
        .arg(repo_root)
        .args(["archive", "--format=tar"])
        .arg(rev)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("git archive failed to start: {}", e))?;

    let archive_stdout = archive
        .stdout
        .take()
        .ok_or_else(|| "git archive produced no stdout".to_string())?;

    let tar = Command::new("tar")
        .arg("-x")
        .arg("-C")
        .arg(dest)
        .stdin(Stdio::from(archive_stdout))
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("tar failed to start: {}", e))?;

    // tar reads the pipe as git writes; wait for the reader first, then the
    // writer, so neither blocks on a full pipe.
    let tar_out = tar
        .wait_with_output()
        .map_err(|e| format!("tar wait failed: {}", e))?;
    let archive_out = archive
        .wait_with_output()
        .map_err(|e| format!("git archive wait failed: {}", e))?;

    if !archive_out.status.success() {
        return Err(format!(
            "git archive failed: {}",
            String::from_utf8_lossy(&archive_out.stderr).trim()
        ));
    }
    if !tar_out.status.success() {
        return Err(format!(
            "tar extract failed: {}",
            String::from_utf8_lossy(&tar_out.stderr).trim()
        ));
    }
    Ok(())
}

/// Map the caller's `src_dir` onto the extracted snapshot: the subpath of
/// `src_dir` relative to `repo_root`, joined onto `snapshot`. When `src_dir`
/// is the repo root (or does not resolve under it), builds the snapshot root.
fn rebase_input(src_dir: &Path, repo_root: &Path, snapshot: &Path) -> PathBuf {
    let src = src_dir
        .canonicalize()
        .unwrap_or_else(|_| src_dir.to_path_buf());
    let root = repo_root
        .canonicalize()
        .unwrap_or_else(|_| repo_root.to_path_buf());
    match src.strip_prefix(&root) {
        Ok(rel) if !rel.as_os_str().is_empty() => snapshot.join(rel),
        _ => snapshot.to_path_buf(),
    }
}

/// Record what this graph represents, so an agent inspecting it via
/// `describe()` sees it is a point-in-time snapshot, not the working tree.
/// Written to the default instructions channel (a freshly-built code_tree
/// graph has none), rendered verbatim at the top of `describe()`.
fn stamp_rev_provenance(
    mut graph: Arc<DirGraph>,
    rev: &str,
    sha: &str,
    repo_root: &Path,
) -> Result<Arc<DirGraph>, String> {
    let short = &sha[..sha.len().min(12)];
    let text = format!(
        "Built from git revision '{rev}' ({short}) of {}. Reflects committed \
         content at that revision, not the current working tree.",
        repo_root.display()
    );
    // The graph is uniquely owned immediately after build (the builder mutates
    // it via `Arc::get_mut`), so this stamping cannot fail in practice.
    let g = Arc::get_mut(&mut graph)
        .ok_or_else(|| "graph not uniquely owned when stamping rev provenance".to_string())?;
    g.set_instructions(&text, None);
    Ok(graph)
}
