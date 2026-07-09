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

use crate::datatypes::Value;
use crate::graph::dir_graph::DirGraph;
use crate::graph::mutation::extend::extend_graph;
use crate::graph::storage::{GraphRead, GraphWrite};
use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
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

    let graph = archive_and_build_into(
        &repo_root,
        rev,
        src_dir,
        tmp.path(),
        verbose,
        include_tests,
        save_to,
        max_loc_per_file,
        include_docs,
    )?;

    stamp_rev_provenance(graph, rev, &sha, &repo_root)
}

/// Extract the tracked tree at `rev` into `snapshot_dir`, then build a code
/// graph from the subpath of `snapshot_dir` matching `src_dir`'s position under
/// `repo_root`. The shared archive→build core behind both the single-rev
/// [`archive_and_build`] and the multi-rev [`build_code_tree_revs`] merge — the
/// caller owns `snapshot_dir`'s lifecycle (a throwaway `TempDir` for one rev, or
/// a reused fixed-basename subdir cleared between revs for a merge). Does no
/// rev-provenance stamping — the caller stamps once it knows the full rev set.
#[allow(clippy::too_many_arguments)]
fn archive_and_build_into(
    repo_root: &Path,
    rev: &str,
    src_dir: &Path,
    snapshot_dir: &Path,
    verbose: bool,
    include_tests: bool,
    save_to: Option<&Path>,
    max_loc_per_file: Option<usize>,
    include_docs: bool,
) -> Result<Arc<DirGraph>, String> {
    archive_into(repo_root, rev, snapshot_dir)?;

    // Scope the build to the same relative subpath the caller pointed at, so
    // `build("/repo/src", rev=…)` builds `src` of the snapshot, not the whole
    // repo. Falls back to the snapshot root when `src_dir` is the repo root or
    // does not resolve under it.
    let build_input = rebase_input(src_dir, repo_root, snapshot_dir);

    crate::code_tree::builder::run_with_options(
        &build_input,
        verbose,
        include_tests,
        save_to,
        max_loc_per_file,
        include_docs,
    )
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

// ─── Multi-rev merge (B.2b) ─────────────────────────────────────────────────
//
// One graph holding N revisions via shared identity + rev-sets: one node per
// entity, native list props `revs: [str]` / `rev_fp: [int]` on nodes and
// `revs: [str]` on edges. Unchanged entities are stored once, so the merged
// graph is ≈ base + deltas. See `dev-docs/plans/rev-aware-code-graphs.md`
// "B.2 design" for the eight settled decisions.

/// Collapse a requested rev-label list to its order-preserving unique form
/// (first occurrence wins). A duplicate label resolves to the same tree, so
/// re-folding it only inflates the per-entity `revs`/`rev_fp` lists and the
/// provenance banner. Shared by the core builder and the MCP activation
/// wrapper so both agree on the canonical label set.
pub fn dedup_revs(revs: &[String]) -> Vec<String> {
    let mut seen: std::collections::HashSet<&str> = std::collections::HashSet::new();
    revs.iter()
        .filter(|r| seen.insert(r.as_str()))
        .cloned()
        .collect()
}

/// Node provenance manifest: `(node_type, id)` → (revs the entity appears in,
/// per-rev fingerprint hashes aligned positionally with those revs).
type NodeRevManifest = HashMap<(String, String), (Vec<Value>, Vec<Value>)>;

/// Edge provenance manifest: `(connection_type, src_id, dst_id)` → the revs the
/// edge appears in (edges carry no fingerprint — existence-per-rev is the signal).
type EdgeRevManifest = HashMap<(String, String, String), Vec<Value>>;

/// Fingerprint field sets per code-entity type — mirrors `_FINGERPRINT` in
/// `kglite/code_tree/_diff.py` so multi-rev change-detection agrees with the
/// two-graph `diff`. `loc_span` is synthetic (`end_line - line_number`), a
/// position-independent body-size proxy: it flags a body edit that grew/shrank
/// an entity without a false positive on every symbol below an unrelated edit.
/// Types absent here (File / Module / Project / markup) have no defined
/// fingerprint and hash to 0 — a `rev_diff` reports them present across revs but
/// never "changed", matching `diff`'s compared-type set.
fn fingerprint_fields(node_type: &str) -> &'static [&'static str] {
    match node_type {
        "Function" => &["signature", "visibility", "loc_span"],
        "Class" | "Mixin" | "Trait" | "Protocol" | "Interface" => &["visibility", "loc_span"],
        "Struct" => &["visibility", "fields", "loc_span"],
        "Enum" => &["visibility", "variants", "loc_span"],
        "Constant" => &["visibility", "value_preview"],
        _ => &[],
    }
}

/// A single `i64` capturing an entity's *shape* at one rev — the hash of its
/// per-type fingerprint fields. `rev_fp[i] != rev_fp[j]` ⇒ the entity changed
/// between `revs[i]` and `revs[j]`. Hashed inputs are the Display form of each
/// [`fingerprint_fields`] value (with `loc_span` derived from `end_line -
/// line_number`); the field *name* is folded in too so a value moving between
/// fields still perturbs the hash. Stability is only required *within one
/// merged graph* (rev_fp is compared to sibling rev_fp entries on the same
/// node, and persisted verbatim on save), so `DefaultHasher` is sufficient.
fn node_fingerprint(node_type: &str, props: &HashMap<String, Value>) -> i64 {
    let fields = fingerprint_fields(node_type);
    if fields.is_empty() {
        return 0;
    }
    let mut hasher = DefaultHasher::new();
    for field in fields {
        field.hash(&mut hasher);
        if *field == "loc_span" {
            let span = match (props.get("end_line"), props.get("line_number")) {
                (Some(Value::Int64(end)), Some(Value::Int64(line))) => Some(end - line),
                _ => None,
            };
            span.hash(&mut hasher);
        } else {
            match props.get(*field) {
                Some(v) => v.to_string().hash(&mut hasher),
                None => 0u8.hash(&mut hasher),
            }
        }
    }
    hasher.finish() as i64
}

/// Build one merged code graph spanning `revs` (git revspecs — tags, SHAs,
/// `HEAD`), oldest → newest in list order.
///
/// Each rev is archived-and-built independently (reusing [`archive_and_build`]'s
/// machinery via a shared **fixed snapshot basename**, so qualified_names / ids
/// align natively across revs), then folded into an accumulator by
/// `(node_type, id)` node identity + `(connection_type, src, tgt)` edge identity
/// — the exact identity [`extend_graph`] already uses. The newest rev an entity
/// appears in wins its ordinary property columns (so plain Cypher reports HEAD's
/// values), and every node/edge carries the rev-set it belongs to:
///
/// - nodes: `revs: [str]` (revs present in, oldest → newest) + `rev_fp: [int]`
///   (aligned per-rev [`node_fingerprint`]s),
/// - edges: `revs: [str]` (edges have no body — existence-per-rev is the whole
///   signal).
///
/// Scope queries with `WHERE '<rev>' IN n.revs`; unscoped queries span *all*
/// revs (an over-count trap the provenance instructions warn about). Peak memory
/// is ≈ two graphs (each rev graph is dropped after folding). In-memory only —
/// `archive_and_build` produces the `Default` backend [`extend_graph`] requires.
///
/// A single-element `revs` yields the same graph shape as [`archive_and_build`]
/// plus the (single-element) `revs` / `rev_fp` tags.
#[allow(clippy::too_many_arguments)]
pub fn build_code_tree_revs(
    src_dir: &Path,
    revs: &[String],
    repo_root: Option<&Path>,
    verbose: bool,
    include_tests: bool,
    save_to: Option<&Path>,
    max_loc_per_file: Option<usize>,
    include_docs: bool,
) -> Result<Arc<DirGraph>, String> {
    if revs.is_empty() {
        return Err("build_code_tree_revs requires at least one revision".to_string());
    }
    // Dedup the requested labels (order-preserving, first occurrence wins). A
    // repeated label re-folds the identical tree — inflating every entity's
    // `revs`/`rev_fp` list with a duplicate and adding a spurious column to the
    // provenance banner — so collapse it to a single fold before any work.
    let revs = dedup_revs(revs);
    let revs = revs.as_slice();
    let repo_root = match repo_root {
        Some(r) => r.to_path_buf(),
        None => resolve_repo_root(src_dir)?,
    };

    // Fixed snapshot basename (Decision 4): every rev extracts into
    // `<tmp>/snapshot`, so all revs share one build-root prefix and their
    // qualified_names / ids align natively across revs — the empirical
    // `_root_alias` heuristic that `_diff.py` needs is unnecessary here.
    let tmp = tempfile::Builder::new()
        .prefix("kglite-revs-")
        .tempdir()
        .map_err(|e| format!("could not create tempdir: {}", e))?;
    let snapshot = tmp.path().join("snapshot");

    // Per-rev provenance manifests, keyed by cross-rev identity. Small (a few
    // strings + ints per entity), independent of per-rev graph size — this is
    // what lets us keep peak memory at ≈ two graphs while still knowing every
    // entity's full rev-set after all folds.
    let mut node_revs: NodeRevManifest = HashMap::new();
    let mut edge_revs: EdgeRevManifest = HashMap::new();

    let mut sha_of_rev: Vec<(String, String)> = Vec::with_capacity(revs.len());
    let mut accumulator: Option<Arc<DirGraph>> = None;

    for rev in revs {
        let sha = verify_rev(&repo_root, rev)?;
        sha_of_rev.push((rev.clone(), sha));

        // Fresh extraction into the fixed snapshot dir.
        if snapshot.exists() {
            std::fs::remove_dir_all(&snapshot)
                .map_err(|e| format!("could not clear snapshot dir: {}", e))?;
        }
        std::fs::create_dir_all(&snapshot)
            .map_err(|e| format!("could not create snapshot dir: {}", e))?;

        let rev_graph = archive_and_build_into(
            &repo_root,
            rev,
            src_dir,
            &snapshot,
            verbose,
            include_tests,
            None, // never save a pre-merge rev graph — we save the merged one
            max_loc_per_file,
            include_docs,
        )?;

        // Record this rev's manifest before folding (both read `rev_graph`).
        record_rev_manifest(&rev_graph, rev, &mut node_revs, &mut edge_revs);

        match accumulator.as_mut() {
            None => accumulator = Some(rev_graph), // oldest = base, full structure kept
            Some(acc) => {
                let target = Arc::get_mut(acc)
                    .ok_or_else(|| "accumulator not uniquely owned during merge".to_string())?;
                extend_graph(target, &rev_graph, Some("update".to_string()))?;
                // `rev_graph` dropped here → peak memory ≈ two graphs.
            }
        }
    }

    let mut graph = accumulator.expect("revs is non-empty");
    {
        let g = Arc::get_mut(&mut graph)
            .ok_or_else(|| "merged graph not uniquely owned when stamping revs".to_string())?;
        stamp_node_revs(g, &node_revs)?;
        stamp_edge_revs(g, &edge_revs);
    }

    if let Some(dest) = save_to {
        // Mirror the builder's save prep so property column stores materialise
        // (including the new `revs` / `rev_fp` list columns) before write.
        crate::graph::io::file::prepare_save(&mut graph);
        Arc::make_mut(&mut graph).enable_columnar();
        let dest_str = dest.to_string_lossy();
        crate::graph::io::file::write_kgl(&graph, &dest_str).map_err(|e| e.to_string())?;
    }

    stamp_rev_provenance_multi(graph, &sha_of_rev, &repo_root)
}

/// Read a freshly-built rev graph and fold its nodes/edges into the running
/// provenance manifests, keyed by cross-rev identity. Read-only over `rev_graph`.
fn record_rev_manifest(
    rev_graph: &DirGraph,
    rev: &str,
    node_revs: &mut NodeRevManifest,
    edge_revs: &mut EdgeRevManifest,
) {
    let rev_val = Value::String(rev.to_string());

    for idx in rev_graph.graph.node_indices() {
        let Some(node) = rev_graph.graph.node_weight(idx) else {
            continue;
        };
        let node_type = node.node_type_str(&rev_graph.interner).to_string();
        let id = node.id().to_string();
        let props = node.properties_cloned(&rev_graph.interner);
        let fp = node_fingerprint(&node_type, &props);
        let entry = node_revs.entry((node_type, id)).or_default();
        entry.0.push(rev_val.clone());
        entry.1.push(Value::Int64(fp));
    }

    for eidx in rev_graph.graph.edge_indices() {
        let Some(edge) = rev_graph.graph.edge_weight(eidx) else {
            continue;
        };
        let Some((s, t)) = rev_graph.graph.edge_endpoints(eidx) else {
            continue;
        };
        let (Some(sn), Some(tn)) = (
            rev_graph.graph.node_weight(s),
            rev_graph.graph.node_weight(t),
        ) else {
            continue;
        };
        let conn = edge.connection_type_str(&rev_graph.interner).to_string();
        let key = (conn, sn.id().to_string(), tn.id().to_string());
        let revs = edge_revs.entry(key).or_default();
        // A code_tree graph dedups edges by (conn, src, tgt), so an edge appears
        // once per rev — guard the append anyway (idempotent within a rev).
        if revs.last() != Some(&rev_val) {
            revs.push(rev_val.clone());
        }
    }
}

/// Stamp `revs` / `rev_fp` list props onto every merged node, in place. Each
/// node's other properties are preserved (`PropertyStorage::insert` extends the
/// node's own compact schema for the new key), and the two keys are registered
/// in the graph's `type_schemas` + `node_type_metadata` so `enable_columnar()`
/// materialises them as columns on `.kgl` save.
///
/// Direct insertion — not `add_nodes` — because `add_nodes` with a 3-column
/// (`id`/`revs`/`rev_fp`) update DataFrame rebuilds each matched node from just
/// those columns, dropping `name`/`qualified_name`/`signature`/…; the whole
/// point of the merge is to *keep* the newest rev's full property set.
fn stamp_node_revs(graph: &mut DirGraph, node_revs: &NodeRevManifest) -> Result<(), String> {
    let revs_key = graph.interner.get_or_intern("revs");
    let fp_key = graph.interner.get_or_intern("rev_fp");

    // Phase 1: resolve each node's identity (read-only), gather its list values
    // + the set of node types touched (for schema registration).
    let mut updates: Vec<(_, Value, Value)> = Vec::new();
    let mut types_touched: std::collections::HashSet<String> = std::collections::HashSet::new();
    for idx in graph.graph.node_indices() {
        let Some(node) = graph.graph.node_weight(idx) else {
            continue;
        };
        let node_type = node.node_type_str(&graph.interner).to_string();
        let id = node.id().to_string();
        if let Some((revs, fps)) = node_revs.get(&(node_type.clone(), id)) {
            updates.push((idx, Value::List(revs.clone()), Value::List(fps.clone())));
            types_touched.insert(node_type);
        }
    }

    // Register the two list columns in every touched type's schema + metadata so
    // the columnar-save path emits them (mirrors what `add_nodes` does for lists).
    for node_type in &types_touched {
        if let Some(existing) = graph.type_schemas.get(node_type).cloned() {
            let mut merged = (*existing).clone();
            merged.add_key(revs_key);
            merged.add_key(fp_key);
            graph
                .type_schemas
                .insert(node_type.clone(), Arc::new(merged));
        }
        let mut meta = HashMap::new();
        meta.insert("revs".to_string(), "List".to_string());
        meta.insert("rev_fp".to_string(), "List".to_string());
        graph.upsert_node_type_metadata(node_type, meta);
    }

    // Phase 2: apply (mutable). Pre-interned keys → no interner borrow needed.
    for (idx, revs_val, fp_val) in updates {
        if let Some(node) = graph.graph.node_weight_mut(idx) {
            node.properties.insert(revs_key, revs_val);
            node.properties.insert(fp_key, fp_val);
        }
    }
    Ok(())
}

/// Stamp `revs` directly onto each merged edge's property vector. The DataFrame
/// edge path (`add_connections`) drops list props to null, but `EdgeData`
/// serializes its `Vec<(InternedKey, Value)>` verbatim, so a directly-set list
/// round-trips through `.kgl` (guarded by `test_edge_list_properties.py`).
fn stamp_edge_revs(graph: &mut DirGraph, edge_revs: &EdgeRevManifest) {
    let revs_key = graph.interner.get_or_intern("revs");

    // Phase 1: resolve each edge's identity (read-only), collect the updates.
    let mut updates = Vec::new();
    for eidx in graph.graph.edge_indices() {
        let Some(edge) = graph.graph.edge_weight(eidx) else {
            continue;
        };
        let Some((s, t)) = graph.graph.edge_endpoints(eidx) else {
            continue;
        };
        let (Some(sn), Some(tn)) = (graph.graph.node_weight(s), graph.graph.node_weight(t)) else {
            continue;
        };
        let conn = edge.connection_type_str(&graph.interner).to_string();
        let key = (conn, sn.id().to_string(), tn.id().to_string());
        if let Some(revs) = edge_revs.get(&key) {
            updates.push((eidx, Value::List(revs.clone())));
        }
    }

    // Phase 2: apply (mutable) — upsert the `revs` slot on each edge.
    for (eidx, revs_val) in updates {
        if let Some(edge) = graph.graph.edge_weight_mut(eidx) {
            match edge.properties.iter_mut().find(|(k, _)| *k == revs_key) {
                Some(slot) => slot.1 = revs_val,
                None => edge.properties.push((revs_key, revs_val)),
            }
        }
    }
}

/// Record which revs a multi-rev graph spans + teach the rev-scoping idiom, in
/// the instructions channel `describe()` renders verbatim. Mirrors
/// [`stamp_rev_provenance`] for the single-rev case; the newest rev is surfaced
/// so an agent can scope "head only" (Decision 7).
fn stamp_rev_provenance_multi(
    mut graph: Arc<DirGraph>,
    revs: &[(String, String)],
    repo_root: &Path,
) -> Result<Arc<DirGraph>, String> {
    let labels: Vec<String> = revs
        .iter()
        .map(|(rev, sha)| format!("{} ({})", rev, &sha[..sha.len().min(12)]))
        .collect();
    // A single-rev graph is a point-in-time snapshot: nothing over-counts and
    // there is no second rev to diff, so it reads plainly — no scoping/over-count
    // warning, no `CALL rev_diff` (which needs two revs). Only ≥2 revs get the
    // full multi-rev steering. It still carries `revs`/`rev_fp` list props.
    let text = if revs.len() == 1 {
        let (rev, sha) = &revs[0];
        format!(
            "Code graph of {} at revision '{}' ({}). Reflects committed content at \
             that revision, not the current working tree. Every entity carries \
             `revs: [str]` (this single rev) + `rev_fp: [int]` (per-rev shape hash).",
            repo_root.display(),
            rev,
            &sha[..sha.len().min(12)],
        )
    } else {
        let newest = revs.last().map(|(rev, _)| rev.as_str()).unwrap_or("");
        format!(
            "Multi-rev code graph of {}, spanning {} revisions (oldest → newest): \
             {}. One node per entity across revs; every node carries `revs: [str]` \
             (revs it appears in) + `rev_fp: [int]` (per-rev shape hash), every edge \
             carries `revs: [str]`. Ordinary properties (signature, value_preview, …) \
             report the NEWEST rev ('{newest}') an entity appears in. UNSCOPED queries \
             span ALL revs (e.g. `MATCH (n:Function) RETURN count(n)` over-counts) — \
             scope with `WHERE '<rev>' IN n.revs` (head only: `WHERE '{newest}' IN \
             n.revs`). For deltas between two revs use \
             `CALL rev_diff({{from: '<rev>', to: '<rev>'}}) YIELD bucket, type, \
             qualified_name, name, file, line` (added / removed / changed).",
            repo_root.display(),
            revs.len(),
            labels.join(", "),
        )
    };
    let g = Arc::get_mut(&mut graph)
        .ok_or_else(|| "graph not uniquely owned when stamping multi-rev provenance".to_string())?;
    g.set_instructions(&text, None);
    Ok(graph)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::storage::interner::InternedKey;
    use std::process::Command;

    /// Run a git subcommand in `dir`, panicking on failure.
    fn git(dir: &Path, args: &[&str]) -> String {
        let out = Command::new("git")
            .arg("-C")
            .arg(dir)
            .args(args)
            .output()
            .expect("git spawn");
        assert!(
            out.status.success(),
            "git {:?} failed: {}",
            args,
            String::from_utf8_lossy(&out.stderr)
        );
        String::from_utf8_lossy(&out.stdout).trim().to_string()
    }

    /// Write `m.py`, stage, commit (with deterministic identity), return the SHA.
    fn commit(dir: &Path, body: &str) -> String {
        std::fs::write(dir.join("m.py"), body).unwrap();
        git(dir, &["add", "-A"]);
        git(
            dir,
            &[
                "-c",
                "user.email=t@t",
                "-c",
                "user.name=t",
                "commit",
                "-q",
                "-m",
                "c",
            ],
        );
        git(dir, &["rev-parse", "HEAD"])
    }

    /// A 3-commit fixture repo. Returns (tempdir guard, [sha1, sha2, sha3]).
    /// r1: foo(a) + gone(); r2: gone removed, bar added, foo now CALLS bar;
    /// r3: foo signature widened to (a, b).
    fn fixture() -> (tempfile::TempDir, [String; 3]) {
        let tmp = tempfile::Builder::new()
            .prefix("kglite-revtest-")
            .tempdir()
            .unwrap();
        let dir = tmp.path();
        git(dir, &["init", "-q"]);
        let s1 = commit(
            dir,
            "def foo(a):\n    return a + 1\n\n\ndef gone():\n    return 0\n",
        );
        let s2 = commit(
            dir,
            "def foo(a):\n    return bar(a)\n\n\ndef bar(x):\n    return x + 1\n",
        );
        let s3 = commit(
            dir,
            "def foo(a, b):\n    return bar(a) + b\n\n\ndef bar(x):\n    return x + 1\n",
        );
        (tmp, [s1, s2, s3])
    }

    fn build(dir: &Path, revs: &[String]) -> Arc<DirGraph> {
        build_code_tree_revs(dir, revs, Some(dir), false, false, None, None, false)
            .expect("build_code_tree_revs")
    }

    /// The `revs` list of the single Function named `name`, as strings.
    fn fn_revs(graph: &DirGraph, name: &str) -> Vec<String> {
        list_prop(graph, "Function", name, "revs")
    }

    /// True when the node's title (where code_tree stores the simple `name`;
    /// `qualified_name` is the node `id`, so neither is a plain property) equals
    /// `name`.
    fn title_is(node: &crate::graph::schema::NodeData, name: &str) -> bool {
        node.title().as_ref() == &Value::String(name.to_string())
    }

    /// A node's list property, resolved to `Vec<String>` (Display per element).
    /// Asserts exactly one node of that type+name exists (no cross-rev dup).
    fn list_prop(graph: &DirGraph, node_type: &str, name: &str, prop: &str) -> Vec<String> {
        let mut found: Vec<Vec<String>> = Vec::new();
        for idx in graph.graph.node_indices() {
            let Some(node) = graph.graph.node_weight(idx) else {
                continue;
            };
            if node.node_type_str(&graph.interner) != node_type {
                continue;
            }
            if !title_is(node, name) {
                continue;
            }
            let list = match node.get_property_value(prop) {
                Some(Value::List(items)) => items
                    .iter()
                    .map(|v| v.as_string().unwrap_or_else(|| v.to_string()))
                    .collect(),
                _ => Vec::new(),
            };
            found.push(list);
        }
        assert_eq!(found.len(), 1, "expected exactly one {node_type} {name}");
        found.into_iter().next().unwrap()
    }

    /// The `revs` of the CALLS edge `caller` → `callee`, as strings.
    fn calls_edge_revs(graph: &DirGraph, caller: &str, callee: &str) -> Vec<String> {
        let revs_key = InternedKey::from_str("revs");
        for eidx in graph.graph.edge_indices() {
            let Some(edge) = graph.graph.edge_weight(eidx) else {
                continue;
            };
            if edge.connection_type_str(&graph.interner) != "CALLS" {
                continue;
            }
            let Some((s, t)) = graph.graph.edge_endpoints(eidx) else {
                continue;
            };
            let (Some(sn), Some(tn)) = (graph.graph.node_weight(s), graph.graph.node_weight(t))
            else {
                continue;
            };
            if title_is(sn, caller) && title_is(tn, callee) {
                return match edge.properties.iter().find(|(k, _)| *k == revs_key) {
                    Some((_, Value::List(items))) => items
                        .iter()
                        .map(|v| v.as_string().unwrap_or_else(|| v.to_string()))
                        .collect(),
                    _ => Vec::new(),
                };
            }
        }
        panic!("no CALLS edge {caller} -> {callee}");
    }

    #[test]
    fn multi_rev_merge_tracks_presence_change_and_edges() {
        let (tmp, [s1, s2, s3]) = fixture();
        let revs = vec![s1.clone(), s2.clone(), s3.clone()];
        let graph = build(tmp.path(), &revs);

        // Presence across revs, one node per entity.
        assert_eq!(
            fn_revs(&graph, "foo"),
            vec![s1.clone(), s2.clone(), s3.clone()]
        );
        assert_eq!(fn_revs(&graph, "bar"), vec![s2.clone(), s3.clone()]); // added in rev2
        assert_eq!(fn_revs(&graph, "gone"), vec![s1.clone()]); // removed after rev1

        // Fingerprint: foo's signature widened only in rev3, so rev_fp[2] diverges
        // from the (equal) earlier revs. A whole-hash equality is all we assert —
        // body-only edits (r1→r2) are intentionally invisible (matches `diff`).
        let fp = match graph
            .graph
            .node_indices()
            .filter_map(|i| graph.graph.node_weight(i))
            .find(|n| n.node_type_str(&graph.interner) == "Function" && title_is(n, "foo"))
            .and_then(|n| n.get_property_value("rev_fp"))
        {
            Some(Value::List(items)) => items,
            other => panic!("foo rev_fp missing/not a list: {other:?}"),
        };
        assert_eq!(fp.len(), 3, "one fingerprint per rev");
        assert_eq!(fp[0], fp[1], "signature unchanged rev1→rev2");
        assert_ne!(fp[1], fp[2], "signature widened in rev3");

        // Edge appearing in rev2+: foo CALLS bar exists only once foo calls it.
        assert_eq!(calls_edge_revs(&graph, "foo", "bar"), vec![s2, s3]);

        // Newest-wins property columns: foo's signature is rev3's widened one.
        let sig = graph
            .graph
            .node_indices()
            .filter_map(|i| graph.graph.node_weight(i))
            .find(|n| n.node_type_str(&graph.interner) == "Function" && title_is(n, "foo"))
            .and_then(|n| n.get_property_value("signature"))
            .and_then(|v| v.as_string())
            .unwrap();
        assert!(
            sig.contains('b'),
            "newest-wins signature (a, b), got {sig:?}"
        );
    }

    #[test]
    fn single_rev_matches_a1_shape_plus_rev_tags() {
        let (tmp, [_s1, _s2, s3]) = fixture();
        let revs = vec![s3.clone()];
        let merged = build(tmp.path(), &revs);

        // Same entity shape as a plain rev build: foo + bar present, gone absent.
        assert_eq!(fn_revs(&merged, "foo"), vec![s3.clone()]);
        assert_eq!(fn_revs(&merged, "bar"), vec![s3.clone()]);
        let names: Vec<String> = merged
            .graph
            .node_indices()
            .filter_map(|i| merged.graph.node_weight(i))
            .filter(|n| n.node_type_str(&merged.interner) == "Function")
            .map(|n| n.title().into_owned())
            .filter_map(|v| v.as_string())
            .collect();
        assert!(names.contains(&"foo".to_string()) && names.contains(&"bar".to_string()));
        assert!(!names.contains(&"gone".to_string()), "gone not in rev3");

        // Single-rev edge is tagged with just that rev.
        assert_eq!(calls_edge_revs(&merged, "foo", "bar"), vec![s3]);
    }

    /// Commit `files` (relative path → contents) to a fresh git repo and
    /// return (tempdir guard, HEAD sha). Used to exercise the parallel-parse
    /// determinism of the multi-rev merge with many HTML inline scripts.
    fn commit_files(files: &[(&str, &str)]) -> (tempfile::TempDir, String) {
        let tmp = tempfile::Builder::new()
            .prefix("kglite-revhtml-")
            .tempdir()
            .unwrap();
        let dir = tmp.path();
        git(dir, &["init", "-q"]);
        for (rel, body) in files {
            let path = dir.join(rel);
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent).unwrap();
            }
            std::fs::write(&path, body).unwrap();
        }
        git(dir, &["add", "-A"]);
        git(
            dir,
            &[
                "-c",
                "user.email=t@t",
                "-c",
                "user.name=t",
                "commit",
                "-q",
                "-m",
                "c",
            ],
        );
        let sha = git(dir, &["rev-parse", "HEAD"]);
        (tmp, sha)
    }

    /// A repo with many HTML files, each carrying a first inline `<script>`
    /// block. The scripts define a shared-named `gtag` plus a per-file helper —
    /// the exact shape (`layout.html:script_1.gtag`) that surfaced the
    /// non-idempotency: every file's first script mapped to the same
    /// `{pid}-{counter}` temp path, so parallel parses raced and corrupted each
    /// other's extraction.
    fn html_script_fixture() -> (tempfile::TempDir, String) {
        let mut files: Vec<(String, String)> = Vec::new();
        for i in 0..24 {
            let rel = format!("page_{i}.html");
            let body = format!(
                "<!doctype html>\n<html><head>\n<script>\n\
                 function gtag(){{ window.dataLayer.push(arguments); }}\n\
                 function helper_{i}(x){{ return x + {i}; }}\n\
                 </script>\n</head><body><h1>Page {i}</h1></body></html>\n"
            );
            files.push((rel, body));
        }
        let refs: Vec<(&str, &str)> = files
            .iter()
            .map(|(a, b)| (a.as_str(), b.as_str()))
            .collect();
        commit_files(&refs)
    }

    /// Sorted node (type, id) + edge (conn, src_id, dst_id) identity sets.
    type EntitySets = (Vec<(String, String)>, Vec<(String, String, String)>);

    /// Sorted (node_type, id) + (conn, src_id, dst_id) identity sets — the exact
    /// cross-rev merge keys. Two builds of the same tree must produce equal sets.
    fn entity_sets(graph: &DirGraph) -> EntitySets {
        let mut nodes: Vec<(String, String)> = graph
            .graph
            .node_indices()
            .filter_map(|i| graph.graph.node_weight(i))
            .map(|n| {
                (
                    n.node_type_str(&graph.interner).to_string(),
                    n.id().to_string(),
                )
            })
            .collect();
        nodes.sort();
        let mut edges: Vec<(String, String, String)> = graph
            .graph
            .edge_indices()
            .filter_map(|e| {
                let edge = graph.graph.edge_weight(e)?;
                let (s, t) = graph.graph.edge_endpoints(e)?;
                let sn = graph.graph.node_weight(s)?;
                let tn = graph.graph.node_weight(t)?;
                Some((
                    edge.connection_type_str(&graph.interner).to_string(),
                    sn.id().to_string(),
                    tn.id().to_string(),
                ))
            })
            .collect();
        edges.sort();
        (nodes, edges)
    }

    /// Every node's `revs` list (as strings) must equal `expect` — the merge is
    /// idempotent only if re-folding the identical tree adds a label to each
    /// existing node without minting a new one.
    fn assert_all_nodes_have_revs(graph: &DirGraph, expect: &[String]) {
        for idx in graph.graph.node_indices() {
            let Some(node) = graph.graph.node_weight(idx) else {
                continue;
            };
            let revs: Vec<String> = match node.get_property_value("revs") {
                Some(Value::List(items)) => items
                    .iter()
                    .map(|v| v.as_string().unwrap_or_else(|| v.to_string()))
                    .collect(),
                _ => Vec::new(),
            };
            assert_eq!(
                revs,
                expect,
                "node {:?} carries revs {:?}, expected {:?}",
                node.id(),
                revs,
                expect
            );
        }
    }

    /// Defect A regression: merging the IDENTICAL tree under two distinct labels
    /// (a SHA + `HEAD` that resolve to the same commit) must yield exactly the
    /// single-rev entity set — no phantom nodes/edges — with every entity
    /// carrying both labels. Exercises the HTML inline-script parallel-parse
    /// path that was nondeterministic (`{pid}-{counter}` temp-dir collisions).
    #[test]
    fn multi_rev_identical_tree_is_idempotent() {
        let (tmp, sha) = html_script_fixture();

        // Baseline: the single-rev build.
        let single = build(tmp.path(), std::slice::from_ref(&sha));
        let (single_nodes, single_edges) = entity_sets(&single);
        // Sanity: the inline scripts really produced Function nodes.
        assert!(
            single_nodes
                .iter()
                .any(|(t, id)| t == "Function" && id.contains("gtag")),
            "fixture should extract gtag functions from inline scripts"
        );

        // Two labels for the same commit → identical tree folded twice.
        let dual = build(tmp.path(), &[sha.clone(), "HEAD".to_string()]);
        let (dual_nodes, dual_edges) = entity_sets(&dual);

        assert_eq!(
            single_nodes, dual_nodes,
            "folding the identical tree twice must not change the node set"
        );
        assert_eq!(
            single_edges, dual_edges,
            "folding the identical tree twice must not change the edge set"
        );
        // Every node carries BOTH labels (order-preserving: sha then HEAD).
        assert_all_nodes_have_revs(&dual, &[sha, "HEAD".to_string()]);
    }

    /// Defect A regression: two independent builds of the SAME single rev must
    /// produce byte-identical entity sets — the fingerprint / extraction path
    /// must not depend on parse ordering or parallelism.
    #[test]
    fn single_rev_build_is_deterministic() {
        let (tmp, sha) = html_script_fixture();
        let a = build(tmp.path(), std::slice::from_ref(&sha));
        let b = build(tmp.path(), std::slice::from_ref(&sha));
        assert_eq!(
            entity_sets(&a),
            entity_sets(&b),
            "two builds of the same rev diverged — extraction is nondeterministic"
        );
    }

    /// Defect B regression: duplicate rev labels are collapsed (order-preserving,
    /// first occurrence wins) before folding, so nodes carry the label once —
    /// not `["HEAD", "HEAD"]`.
    #[test]
    fn duplicate_rev_labels_are_deduped() {
        let (tmp, [_s1, _s2, s3]) = fixture();
        let graph = build(tmp.path(), &[s3.clone(), s3.clone(), "HEAD".to_string()]);
        // s3 == HEAD, so all three labels resolve to one commit; dedup keeps the
        // first two DISTINCT labels (s3, HEAD) and drops the repeated s3.
        assert_eq!(fn_revs(&graph, "foo"), vec![s3, "HEAD".to_string()]);
    }

    #[test]
    fn merged_graph_revs_survive_save_reload() {
        let (tmp, [s1, s2, s3]) = fixture();
        let revs = vec![s1.clone(), s2.clone(), s3.clone()];
        let out = tmp.path().join("merged.kgl");
        let built = build_code_tree_revs(
            tmp.path(),
            &revs,
            Some(tmp.path()),
            false,
            false,
            Some(&out),
            None,
            false,
        )
        .expect("build+save");
        // Sanity on the in-memory graph before reload.
        assert_eq!(fn_revs(&built, "bar"), vec![s2.clone(), s3.clone()]);

        let reloaded = crate::graph::io::file::load_file(out.to_str().unwrap()).expect("reload");
        assert_eq!(
            fn_revs(&reloaded, "foo"),
            vec![s1.clone(), s2.clone(), s3.clone()]
        );
        assert_eq!(fn_revs(&reloaded, "bar"), vec![s2.clone(), s3.clone()]);
        // rev_fp list persists too.
        assert_eq!(list_prop(&reloaded, "Function", "gone", "revs"), vec![s1]);
        // Edge revs survive the round-trip (EdgeData property vector).
        assert_eq!(calls_edge_revs(&reloaded, "foo", "bar"), vec![s2, s3]);
    }
}
