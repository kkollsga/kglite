//! The provenance pair: what a fingerprint notices, what it does not, and what
//! `rebuild_if_changed` does with it.

use super::*;
use crate::graph::schema::EmbeddingStore;
use crate::graph::storage::GraphRead;
use crate::okf::cache::is_cache_artifact;
use crate::okf::model::{Dialect, RebuildOptions};
use std::fs;
use std::path::PathBuf;
use tempfile::TempDir;

fn vault_opts() -> BuildOptions {
    BuildOptions::for_dialect(Dialect::Obsidian)
}

fn write(root: &Path, rel: &str, text: &str) {
    let path = root.join(rel);
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(path, text).unwrap();
}

/// A vault with two notes, one picture and a `.kglite/vault.yaml`.
fn vault() -> TempDir {
    let dir = tempfile::tempdir().unwrap();
    write(
        dir.path(),
        "notes/alpha.md",
        "---\nid: alpha\n---\nAlpha.\n",
    );
    write(
        dir.path(),
        "notes/beta.md",
        "---\nid: beta\n---\nSee [[alpha]] and ![map](../img/x.png).\n",
    );
    write(dir.path(), "img/x.png", "not really a png");
    write(
        dir.path(),
        ".kglite/vault.yaml",
        "kglite_vault: 1\ndefault_label: Note\n",
    );
    dir
}

/// Move a file's mtime forward without touching its size, which is the change
/// a "same length, rewritten" edit makes.
fn touch(path: &Path) {
    let later = std::time::SystemTime::now() + std::time::Duration::from_secs(120);
    let file = fs::File::options().write(true).open(path).unwrap();
    file.set_modified(later).unwrap();
}

/// Every `Note`'s `(id, node slot)`, so a test can address the same note
/// before and after a rebuild has renumbered the slots.
fn note_slots(graph: &DirGraph) -> Vec<(String, usize)> {
    graph
        .type_indices
        .get("Note")
        .map(|indices| indices.to_vec())
        .unwrap_or_default()
        .into_iter()
        .map(|idx| {
            let id = match graph.graph.node_view(idx).unwrap().id().into_owned() {
                crate::datatypes::values::Value::String(s) => s.to_string(),
                other => other.to_string(),
            };
            (id, idx.index())
        })
        .collect()
}

/// The `body` property of the note in that slot.
fn body_at(graph: &DirGraph, slot: usize) -> String {
    let idx = petgraph::graph::NodeIndex::new(slot);
    match graph
        .graph
        .node_view(idx)
        .unwrap()
        .get(crate::graph::storage::interner::InternedKey::from_str(
            "body",
        ))
        .map(|v| v.into_owned())
    {
        Some(crate::datatypes::values::Value::String(s)) => s.to_string(),
        other => panic!("no body in slot {slot}: {other:?}"),
    }
}

fn fp(dir: &TempDir) -> u64 {
    fingerprint(dir.path(), &vault_opts()).unwrap()
}

#[test]
fn the_same_tree_fingerprints_the_same_twice() {
    let dir = vault();
    assert_eq!(fp(&dir), fp(&dir));
    // …and the value does not depend on the order the filesystem hands the
    // entries back: a second walk of a copy made in a different order agrees.
    let copy = tempfile::tempdir().unwrap();
    for rel in [
        "img/x.png",
        ".kglite/vault.yaml",
        "notes/beta.md",
        "notes/alpha.md",
    ] {
        let from = dir.path().join(rel);
        let to = copy.path().join(rel);
        fs::create_dir_all(to.parent().unwrap()).unwrap();
        fs::copy(&from, &to).unwrap();
        let modified = fs::metadata(&from).unwrap().modified().unwrap();
        fs::File::options()
            .write(true)
            .open(&to)
            .unwrap()
            .set_modified(modified)
            .unwrap();
    }
    assert_eq!(
        fp(&dir),
        fp(&copy),
        "same paths, sizes and whole-second mtimes — the same vault"
    );
}

#[test]
fn every_kind_of_edit_moves_the_fingerprint() {
    type Edit = Box<dyn Fn(&Path)>;
    let cases: Vec<(&str, Edit)> = vec![
        (
            "a note's text changed",
            Box::new(|root: &Path| write(root, "notes/alpha.md", "---\nid: alpha\n---\nMore.\n")),
        ),
        (
            "a note was touched but not resized",
            Box::new(|root: &Path| touch(&root.join("notes/alpha.md"))),
        ),
        (
            "a note was renamed",
            Box::new(|root: &Path| {
                fs::rename(root.join("notes/alpha.md"), root.join("notes/renamed.md")).unwrap()
            }),
        ),
        (
            "a note was added",
            Box::new(|root: &Path| write(root, "notes/gamma.md", "new")),
        ),
        (
            "a note was deleted",
            Box::new(|root: &Path| fs::remove_file(root.join("notes/beta.md")).unwrap()),
        ),
        (
            "an attachment changed",
            Box::new(|root: &Path| write(root, "img/x.png", "a different picture")),
        ),
        (
            "an attachment was added",
            Box::new(|root: &Path| write(root, "img/y.png", "another")),
        ),
        (
            "the vault config changed",
            Box::new(|root: &Path| {
                write(
                    root,
                    ".kglite/vault.yaml",
                    "kglite_vault: 1\ndefault_label: Article\n",
                )
            }),
        ),
        (
            "a carried skill was added",
            Box::new(|root: &Path| {
                write(
                    root,
                    ".kglite/skills/one.md",
                    "---\nname: one\ndescription: d\n---\nbody\n",
                )
            }),
        ),
    ];
    for (name, edit) in cases {
        let dir = vault();
        let before = fp(&dir);
        edit(dir.path());
        assert_ne!(before, fp(&dir), "{name}");
    }
}

/// The length prefix on each path is not decoration: without it the entry
/// stream is `path ++ size ++ mtime` repeated, and two *different* file lists
/// can fold to the same bytes when one path's characters line up with the
/// next entry's fixed-width size and mtime fields. These two lists are exactly
/// that pair — byte-identical without the prefix, different with it.
#[test]
fn two_different_file_lists_cannot_fold_to_one_value() {
    let left = [
        ("ab", 0u64, Some(0x6200_0000_0000_0000i64)),
        ("c", 5, Some(7)),
    ];
    let right = [("a", 98u64, Some(0i64)), ("bc", 5, Some(7))];
    let fold_all = |entries: &[(&str, u64, Option<i64>)]| {
        entries
            .iter()
            .fold(FNV_OFFSET, |hash, (path, size, mtime)| {
                fold_entry(hash, path, *size, *mtime)
            })
    };
    assert_ne!(fold_all(&left), fold_all(&right));
}

/// The `.kglite/` scan sorts its own entries, because `WalkDir` hands them
/// back in whatever order the filesystem lists them and the fingerprint is
/// compared across machines. A directory written in reverse order still reads
/// in path order.
#[test]
fn the_kglite_scan_is_sorted_whatever_order_the_files_were_written_in() {
    let dir = tempfile::tempdir().unwrap();
    for rel in [
        ".kglite/skills/z.md",
        ".kglite/vault.yaml",
        ".kglite/recipes/a.md",
    ] {
        write(dir.path(), rel, "x");
    }
    let paths: Vec<String> = kglite_dir_entries(dir.path())
        .into_iter()
        .map(|(path, _, _)| path)
        .collect();
    assert_eq!(
        paths,
        vec![
            ".kglite/recipes/a.md".to_string(),
            ".kglite/skills/z.md".to_string(),
            ".kglite/vault.yaml".to_string(),
        ]
    );
}

/// An empty vault is still a vault: it carries its `.kglite/` declarations, and
/// `build` returns early for it — a path that has to stamp provenance too, or
/// the first note written to it could never be noticed.
#[test]
fn a_vault_with_no_notes_is_stamped_too() {
    let dir = tempfile::tempdir().unwrap();
    write(
        dir.path(),
        ".kglite/vault.yaml",
        "kglite_vault: 1\ndefault_label: Note\n",
    );
    let out = build(dir.path(), &vault_opts()).unwrap();
    assert_eq!(out.report.concepts, 0);
    assert!(out.graph.source_root.is_some());
    assert_eq!(out.graph.source_fingerprint, Some(fp(&dir)));

    write(dir.path(), "first.md", "The first note.\n");
    let filled = rebuild(&out.graph).expect("the first note is a change");
    assert_eq!(filled.report.concepts, 1);
}

/// The fingerprint must read the vault's *own* `skip_dirs`, not just the
/// caller's: `build` applies the config before it walks, so a fingerprint that
/// did not would describe a different file set and report a change on every
/// call — a rebuild loop that never settles.
#[test]
fn the_vaults_own_skip_dirs_reach_the_fingerprint() {
    let dir = vault();
    write(
        dir.path(),
        ".kglite/vault.yaml",
        "kglite_vault: 1\ndefault_label: Note\nskip_dirs:\n  - drafts\n",
    );
    write(dir.path(), "drafts/wip.md", "half a thought");
    let built = build(dir.path(), &vault_opts()).unwrap().graph;
    assert_eq!(built.source_fingerprint, Some(fp(&dir)));

    write(dir.path(), "drafts/wip.md", "a different half a thought");
    assert!(
        rebuild(&built).is_none(),
        "a file the vault declared out of the build is not a change to it"
    );
}

/// A diverted `index.md` is still a file the build read — under `okf` it
/// becomes the folder's description — so editing one has to move the value.
#[test]
fn a_reserved_filename_is_still_fingerprinted() {
    let dir = tempfile::tempdir().unwrap();
    write(dir.path(), "notes/a.md", "body");
    write(dir.path(), "notes/index.md", "the notes folder");
    let okf = BuildOptions::default();
    let before = fingerprint(dir.path(), &okf).unwrap();
    write(dir.path(), "notes/index.md", "a different description");
    assert_ne!(before, fingerprint(dir.path(), &okf).unwrap());
}

/// `.kglite/` is a vault input and nothing else, so it counts for the dialect
/// that reads it and not for the two that do not — a fingerprint that reported
/// a change no rebuild could act on would rebuild forever.
#[test]
fn the_kglite_directory_counts_only_where_it_is_read() {
    let dir = vault();
    let okf = BuildOptions::default();
    let before = fingerprint(dir.path(), &okf).unwrap();
    write(
        dir.path(),
        ".kglite/vault.yaml",
        "kglite_vault: 1\ndefault_label: Article\n",
    );
    assert_eq!(
        before,
        fingerprint(dir.path(), &okf).unwrap(),
        "an OKF build never reads it"
    );
}

/// The cache the vault carries is not a build input, so writing it — and the
/// lease, owner-record and in-flight temp siblings `save_graph` leaves beside
/// it — must leave the fingerprint exactly where it was. Without the
/// exclusion the cache invalidates itself on the save that writes it and can
/// never hit.
#[test]
fn the_cache_and_its_siblings_do_not_move_the_fingerprint() {
    let dir = vault();
    let before = fp(&dir);
    for rel in [
        ".kglite/graph.kgl",
        ".kglite/graph.kgl.lock",
        ".kglite/graph.kgl.lock-owner",
        ".kglite/graph.kgl.tmp.4242.17",
        ".kglite/export-manifest.json",
    ] {
        write(dir.path(), rel, "cache bytes");
        assert_eq!(before, fp(&dir), "`{rel}` is not a build input");
    }
    // …and the exclusion is exactly those names: a note the author happened
    // to keep in `.kglite/` still counts.
    write(dir.path(), ".kglite/skills/one.md", "a carried skill");
    assert_ne!(before, fp(&dir), "a carried skill is a build input");
}

/// The predicate is about the cache set alone, keyed on the whole relative
/// path: a `graph.kgl` the author put somewhere else in the vault is an
/// ordinary file, and so is a directory that merely shares a name.
#[test]
fn only_the_kglite_cache_set_is_a_cache_artifact() {
    for rel in [
        ".kglite/graph.kgl",
        ".kglite/graph.kgl.lock",
        ".kglite/graph.kgl.lock-owner",
        ".kglite/graph.kgl.tmp.1.2",
        ".kglite/export-manifest.json",
    ] {
        assert!(is_cache_artifact(Path::new(rel)), "{rel}");
    }
    for rel in [
        "graph.kgl",
        "notes/graph.kgl",
        ".kglite/vault.yaml",
        ".kglite/skills/one.md",
        ".kglite/graph.kgl/inner",
        ".kglite/graph.kgl.notes.md",
        "other/.kglite/graph.kgl",
    ] {
        assert!(!is_cache_artifact(Path::new(rel)), "{rel}");
    }
}

#[test]
fn a_missing_root_is_the_walks_own_refusal() {
    let dir = tempfile::tempdir().unwrap();
    let gone = dir.path().join("nowhere");
    assert!(fingerprint(&gone, &vault_opts())
        .unwrap_err()
        .contains("does not exist"));
}

// ── provenance ──────────────────────────────────────────────────────────────

#[test]
fn a_build_stamps_the_root_and_the_fingerprint_it_read() {
    let dir = vault();
    let out = build(dir.path(), &vault_opts()).unwrap();
    let expected = dir.path().canonicalize().unwrap();
    assert_eq!(
        out.graph.source_root.as_deref(),
        Some(expected.to_string_lossy().as_ref()),
        "absolute, because the graph outlives the working directory"
    );
    assert_eq!(out.graph.source_fingerprint, Some(fp(&dir)));
}

#[test]
fn a_graph_that_was_not_built_from_a_directory_carries_no_provenance() {
    let graph = DirGraph::new();
    assert_eq!(graph.source_root, None);
    assert_eq!(graph.source_fingerprint, None);
}

#[test]
fn provenance_survives_a_save_and_load() {
    let dir = vault();
    let mut built = build(dir.path(), &vault_opts()).unwrap().graph;
    // Saved beside the vault, never inside it (the fingerprint would move) and
    // never at a path two tests share: `../vault.kgl` is one /tmp file.
    let out = tempfile::tempdir().unwrap();
    let file = out.path().join("vault.kgl");
    crate::graph::io::file::save_graph(&mut built, &file.to_string_lossy()).unwrap();
    let loaded = crate::graph::io::file::load_file(&file.to_string_lossy()).unwrap();
    assert_eq!(loaded.source_root, built.source_root);
    assert_eq!(loaded.source_fingerprint, built.source_fingerprint);
}

// ── rebuild_if_changed ──────────────────────────────────────────────────────

fn vault_rebuild_opts() -> RebuildOptions {
    RebuildOptions::for_dialect(Dialect::Obsidian)
}

fn rebuild(graph: &DirGraph) -> Option<BuildOutput> {
    rebuild_if_changed(graph, &vault_rebuild_opts(), None).unwrap()
}

#[test]
fn an_unchanged_vault_rebuilds_nothing_and_a_changed_one_rebuilds() {
    let dir = vault();
    let built = build(dir.path(), &vault_opts()).unwrap().graph;
    assert!(rebuild(&built).is_none(), "nothing moved");

    write(dir.path(), "notes/gamma.md", "---\nid: gamma\n---\nNew.\n");
    let again = rebuild(&built).expect("a new note is a change");
    assert_eq!(again.report.concepts, 3);
    assert_eq!(
        again.graph.source_fingerprint,
        Some(fp(&dir)),
        "the rebuilt graph carries the tree it just read"
    );
    assert!(rebuild(&again.graph).is_none(), "and is itself current now");
}

#[test]
fn a_graph_with_no_source_root_cannot_be_rebuilt() {
    let error = rebuild_if_changed(&DirGraph::new(), &vault_rebuild_opts(), None)
        .map(|_| "rebuilt")
        .unwrap_err();
    assert!(error.contains("source_root"), "{error}");
}

/// The point of the carry: a rebuild after editing one note keeps every other
/// note's vector *and its text hash*, so the next changed-mode pass re-embeds
/// exactly the note that was edited. Without the hashes a 7 000-note vault
/// re-embeds itself on every save.
#[test]
fn a_rebuild_carries_the_vectors_and_hashes_of_the_notes_that_did_not_change() {
    let dir = vault();
    let mut built = build(dir.path(), &vault_opts()).unwrap().graph;
    let graph = crate::graph::handle::make_dir_graph_mut(&mut built);
    let mut store = EmbeddingStore::new(2);
    for (_, slot) in note_slots(graph) {
        store.set_embedding(slot, &[1.0, 0.0]);
        store.set_text_hash(slot, EmbeddingStore::text_hash(&body_at(graph, slot)));
    }
    graph.set_embedding_store("Note", "body", store);

    write(
        dir.path(),
        "notes/beta.md",
        "---\nid: beta\n---\nBeta, rewritten.\n",
    );
    let out = rebuild(&built).expect("the edit is a change");
    let store = out
        .graph
        .embeddings
        .get(&("Note".to_string(), "body_emb".to_string()))
        .expect("the carried store");
    assert_eq!(store.len(), 2, "both notes kept their vectors");
    for (id, slot) in note_slots(&out.graph) {
        let stale = store.is_stale(slot, EmbeddingStore::text_hash(&body_at(&out.graph, slot)));
        assert_eq!(
            stale,
            id == "beta",
            "only the rewritten note should look stale (`{id}`)"
        );
    }
}

/// Same-label carry only (plan §2.4): a note that moves between folders under
/// a label-from-folder profile is a *different* label, so its vector does not
/// travel and the next embed pass computes it again.
#[test]
fn a_relabelled_note_loses_its_vector() {
    let dir = tempfile::tempdir().unwrap();
    write(dir.path(), "notes/alpha.md", "---\nid: alpha\n---\nA.\n");
    let mut opts = vault_opts();
    opts.profile.default_label = None;
    let mut built = build(dir.path(), &opts).unwrap().graph;
    assert_eq!(built.type_indices.get("notes").map(|n| n.len()), Some(1));

    let graph = crate::graph::handle::make_dir_graph_mut(&mut built);
    let idx = graph.type_indices.get("notes").unwrap().to_vec()[0];
    let mut store = EmbeddingStore::new(2);
    store.set_embedding(idx.index(), &[1.0, 0.0]);
    graph.set_embedding_store("notes", "body", store);

    fs::create_dir_all(dir.path().join("archive")).unwrap();
    fs::rename(
        dir.path().join("notes/alpha.md"),
        dir.path().join("archive/alpha.md"),
    )
    .unwrap();
    let out = rebuild_if_changed(
        &built,
        &RebuildOptions {
            profile: Some(opts.profile.clone()),
            ..vault_rebuild_opts()
        },
        None,
    )
    .unwrap()
    .expect("the move is a change");
    assert_eq!(
        out.graph.type_indices.get("archive").map(|n| n.len()),
        Some(1)
    );
    assert!(
        !out.graph
            .embeddings
            .contains_key(&("archive".to_string(), "body_emb".to_string())),
        "the vector was keyed by the label the note used to have"
    );
}

#[test]
fn declared_embed_targets_warn_when_no_embedder_is_given() {
    let dir = vault();
    write(
        dir.path(),
        ".kglite/vault.yaml",
        "kglite_vault: 1\ndefault_label: Note\nembed:\n  Note: body\n",
    );
    let built = build(dir.path(), &vault_opts()).unwrap().graph;
    write(dir.path(), "notes/gamma.md", "---\nid: gamma\n---\nNew.\n");
    let out = rebuild(&built).expect("a change");
    assert!(
        out.report
            .warnings
            .iter()
            .any(|w| w.contains("no embedder was given")),
        "{:?}",
        out.report.warnings
    );
    assert!(out.graph.embeddings.is_empty());
}

/// A path that no longer exists is the walk's refusal, not a silent `None`:
/// "the vault is gone" must never read as "the vault is unchanged".
#[test]
fn a_vanished_source_root_is_an_error_not_an_unchanged_verdict() {
    let dir = vault();
    let built = build(dir.path(), &vault_opts()).unwrap().graph;
    let moved: PathBuf = dir.path().to_path_buf();
    drop(dir);
    assert!(!moved.exists());
    let error = rebuild_if_changed(&built, &vault_rebuild_opts(), None)
        .map(|_| "rebuilt")
        .unwrap_err();
    assert!(error.contains("does not exist"), "{error}");
}

// ── the build dialect (VAULT.md §12) ────────────────────────────────────────

/// The trap the stamp closes: a vault-built graph rebuilt by a caller who
/// named no dialect was rebuilt as an OKF bundle — a different file set, a
/// different fingerprint, so "changed" every time, and the result was an
/// almost empty graph that loads and looks valid.
#[test]
fn a_rebuild_that_was_not_told_a_dialect_uses_the_one_the_build_stamped() {
    let dir = vault();
    let built = build(dir.path(), &vault_opts()).unwrap().graph;
    let out = rebuild_if_changed(&built, &RebuildOptions::default(), None).unwrap();
    assert!(
        out.is_none(),
        "nothing in the vault moved, so an unasked dialect must read the stamp"
    );
}

#[test]
fn a_build_stamps_the_dialect_it_read_with_and_it_survives_a_save_and_load() {
    let dir = vault();
    let mut built = build(dir.path(), &vault_opts()).unwrap().graph;
    assert_eq!(built.source_dialect.as_deref(), Some("obsidian"));
    assert_eq!(
        build(dir.path(), &BuildOptions::default())
            .unwrap()
            .graph
            .source_dialect
            .as_deref(),
        Some("okf"),
        "the stamp is what was read, not what a vault would have been"
    );

    // Saved beside the vault, never inside it (the fingerprint would move) and
    // never at a path two tests share: `../vault.kgl` is one /tmp file.
    let out = tempfile::tempdir().unwrap();
    let file = out.path().join("vault.kgl");
    crate::graph::io::file::save_graph(&mut built, &file.to_string_lossy()).unwrap();
    let loaded = crate::graph::io::file::load_file(&file.to_string_lossy()).unwrap();
    assert_eq!(loaded.source_dialect, built.source_dialect);
    assert_eq!(stamped_dialect(&loaded).unwrap(), Some(Dialect::Obsidian));
}

/// The two readings of one directory are different builds, and the numbers
/// they compare are not comparable. Obeying the caller would rebuild the
/// vault into a bundle; obeying the stamp would ignore what they asked for.
#[test]
fn a_dialect_that_contradicts_the_stamp_is_refused_naming_both() {
    let dir = vault();
    let built = build(dir.path(), &vault_opts()).unwrap().graph;
    let error = rebuild_if_changed(&built, &RebuildOptions::for_dialect(Dialect::Okf), None)
        .map(|_| "rebuilt")
        .unwrap_err();
    assert!(error.contains("`obsidian`"), "the stamp is named: {error}");
    assert!(error.contains("`okf`"), "and so is the request: {error}");
}

/// A dialect a newer build knows and this one does not must not degrade to
/// `okf` the way [`Dialect::parse`] does — that is exactly the silent wrong
/// reading the stamp exists to prevent.
#[test]
fn an_unknown_stamped_dialect_is_reported_rather_than_guessed() {
    let dir = vault();
    let mut built = build(dir.path(), &vault_opts()).unwrap().graph;
    crate::graph::handle::make_dir_graph_mut(&mut built).source_dialect =
        Some("hypergraph".to_string());
    let error = stamped_dialect(&built).map(|_| "parsed").unwrap_err();
    assert!(error.contains("hypergraph"), "{error}");
    assert!(rebuild_if_changed(&built, &RebuildOptions::default(), None)
        .map(|_| "rebuilt")
        .unwrap_err()
        .contains("hypergraph"));
}

/// A `.kgl` from 0.17.8–0.17.10 carries the root and the fingerprint and no
/// dialect. Nothing changes for it — the caller's dialect still decides — but
/// the report says so, because that is the graph on which an untouched vault
/// can still read as changed.
#[test]
fn a_graph_with_no_dialect_stamp_keeps_the_callers_dialect_and_says_so() {
    let dir = vault();
    let mut built = build(dir.path(), &vault_opts()).unwrap().graph;
    crate::graph::handle::make_dir_graph_mut(&mut built).source_dialect = None;
    assert_eq!(stamped_dialect(&built).unwrap(), None);

    let out = rebuild(&built);
    assert!(
        out.is_none(),
        "the caller named `obsidian`, which is what it was built as"
    );

    write(dir.path(), "notes/gamma.md", "---\nid: gamma\n---\nNew.\n");
    let out = rebuild(&built).expect("a new note is a change");
    assert_eq!(out.graph.source_dialect.as_deref(), Some("obsidian"));
    assert!(
        out.report
            .warnings
            .iter()
            .any(|w| w.contains("no dialect stamp")),
        "{:?}",
        out.report.warnings
    );
}
