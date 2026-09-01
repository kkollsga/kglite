//! Lost-update, lease-lifetime and rollback contracts of [`WriteOwnership`].

use super::*;

use crate::graph::handle::make_dir_graph_mut;
use crate::graph::io::open::GraphWriterLease;

/// A saved, path-backed graph plus the ownership that tracks it — the state
/// every caller reaches immediately after `open_or_create_graph`.
fn owned(path: &Path, keep_pristine: bool) -> (Arc<DirGraph>, WriteOwnership) {
    let mut graph = Arc::new(DirGraph::new());
    save_graph(&mut graph, &path.to_string_lossy()).unwrap();
    let identity = GraphFileIdentity::capture(path).unwrap();
    let ownership = WriteOwnership::new(
        path.to_path_buf(),
        identity,
        &graph,
        Some("test".to_string()),
        keep_pristine,
    );
    (graph, ownership)
}

/// A mutation, reduced to what this module cares about: the version moves.
fn mutate(graph: &mut Arc<DirGraph>) {
    make_dir_graph_mut(graph);
}

/// Whether anybody could take the writer lease right now. The whole point of
/// releasing on a refusal is that this answers `true` afterwards.
fn lockable(path: &Path) -> bool {
    GraphWriterLease::acquire(path, Duration::ZERO).is_ok()
}

/// `flock` is per open file *description*, not per process, so two owners in
/// one process exclude each other exactly as two processes would — the same
/// property `writer_lease_is_exclusive_within_one_process` relies on in
/// `open.rs`.
#[test]
fn a_second_owner_of_the_same_path_is_refused() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("contended.kgl");
    let (mut first_graph, mut first) = owned(&path, true);
    let identity = GraphFileIdentity::capture(&path).unwrap();
    let mut second_graph = Arc::clone(&first_graph);
    let mut second = WriteOwnership::new(
        path.clone(),
        identity,
        &second_graph,
        Some("peer".to_string()),
        true,
    );

    assert_eq!(
        first.begin_write(&mut first_graph).unwrap(),
        BeginWrite::Acquired
    );
    assert_eq!(
        first.begin_write(&mut first_graph).unwrap(),
        BeginWrite::Held
    );
    match second.begin_write(&mut second_graph) {
        Err(WriteRefusal::Contended(refusal)) => {
            assert!(refusal.holder.unwrap().is_self());
        }
        other => panic!("expected contention, got {other:?}"),
    }
    assert!(!second.holds_lease());

    first.publish(&mut first_graph).unwrap();
    assert_eq!(
        second
            .begin_write(&mut second_graph)
            .unwrap_err()
            .to_string(),
        WriteRefusal::Stale { path }.to_string(),
        "the peer's own snapshot is stale once the first owner published"
    );
}

#[test]
fn a_stale_publish_keeps_both_the_mutations_and_the_lease() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("clobber.kgl");
    let (mut graph, mut ownership) = owned(&path, true);
    ownership.begin_write(&mut graph).unwrap();
    mutate(&mut graph);
    let dirty_version = graph.version();

    std::fs::write(&path, b"a competing writer got here first").unwrap();
    let refusal = ownership.publish(&mut graph).unwrap_err();

    assert!(matches!(refusal, WriteRefusal::Stale { .. }));
    assert_eq!(
        std::fs::read(&path).unwrap(),
        b"a competing writer got here first",
        "the refusal must be decided before the path is touched"
    );
    assert_eq!(graph.version(), dirty_version, "mutations must survive");
    assert!(ownership.is_dirty(&graph));
    assert!(
        ownership.holds_lease(),
        "the caller still has a choice to make (save_as / discard) and needs the lease for it"
    );
}

#[test]
fn a_stale_begin_write_releases_the_lease() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("moved-on.kgl");
    let (mut graph, mut ownership) = owned(&path, true);

    std::fs::write(&path, b"replaced before the first write").unwrap();
    let refusal = ownership.begin_write(&mut graph).unwrap_err();

    assert!(matches!(refusal, WriteRefusal::Stale { .. }));
    assert!(!ownership.holds_lease());
    assert!(
        lockable(&path),
        "nothing was mutated, so no peer may be blocked by the refusal"
    );
}

#[test]
fn discard_restores_and_releases_even_when_the_file_is_unreadable() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("garbage.kgl");
    let (mut graph, mut ownership) = owned(&path, true);
    let clean_version = graph.version();
    ownership.begin_write(&mut graph).unwrap();
    mutate(&mut graph);

    // Rollback must not depend on the file: this is the state a caller is in
    // when its peer is mid-republish, which is exactly when it most needs out.
    std::fs::write(&path, b"not a kgl file at all").unwrap();
    let discarded = ownership.discard(&mut graph);

    assert!(discarded.restored);
    assert!(!ownership.holds_lease());
    assert!(lockable(&path));
    assert!(!ownership.is_dirty(&graph));
    assert!(
        graph.version() > clean_version,
        "the restored graph must not reuse a version its lineage has already cached plans under"
    );
}

#[test]
fn discard_clears_every_version_the_dirty_lineage_reached() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("high-water.kgl");
    let (mut graph, mut ownership) = owned(&path, true);
    ownership.begin_write(&mut graph).unwrap();
    let mut reached = vec![graph.version()];
    for _ in 0..5 {
        mutate(&mut graph);
        reached.push(graph.version());
    }

    ownership.discard(&mut graph);

    let highest = reached.iter().copied().max().unwrap();
    assert!(
        graph.version() > highest,
        "restored at {} but the discarded lineage reached {highest}",
        graph.version()
    );
}

#[test]
fn discard_without_a_pristine_snapshot_only_releases() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("no-snapshot.kgl");
    let (mut graph, mut ownership) = owned(&path, false);
    ownership.begin_write(&mut graph).unwrap();
    mutate(&mut graph);

    let discarded = ownership.discard(&mut graph);

    assert!(!discarded.restored);
    assert!(lockable(&path));
    assert!(
        ownership.is_dirty(&graph),
        "with nothing to roll back to, the graph must keep reporting the changes it still holds"
    );
}

/// The invariant the whole dirty/clean signal rests on: if a save bumped
/// `version`, every published graph would immediately report itself dirty
/// again and the lease would never be released.
#[test]
fn publishing_does_not_bump_the_version() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("stable-version.kgl");
    let (mut graph, mut ownership) = owned(&path, true);
    ownership.begin_write(&mut graph).unwrap();
    mutate(&mut graph);
    let before = graph.version();

    ownership.publish(&mut graph).unwrap();

    assert_eq!(graph.version(), before);
    assert!(!ownership.is_dirty(&graph));
    assert!(!ownership.holds_lease());
    assert!(lockable(&path));
}

/// A clean graph still publishes. The MCP server materializes an ontology at
/// boot through paths that never bump `version`, so refusing to save a clean
/// graph would make that materialization unreachable on disk.
#[test]
fn a_clean_graph_still_publishes() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("clean.kgl");
    let (mut graph, mut ownership) = owned(&path, true);
    assert!(!ownership.is_dirty(&graph));
    let before = GraphFileIdentity::capture(&path).unwrap();

    ownership.publish(&mut graph).unwrap();

    assert_ne!(
        ownership.synced(),
        &before,
        "the file was rewritten, so the recaptured identity must reflect it"
    );
    assert!(lockable(&path));
}

/// `make_dir_graph_mut` bumps `version` *before* the statement runs, so a
/// statement that fails leaves a dirty graph with nothing applied. Discarding
/// there is the only thing that keeps a lease from being parked over nothing.
#[test]
fn a_failed_first_write_leaves_the_file_lockable() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("failed-first.kgl");
    let (mut graph, mut ownership) = owned(&path, true);

    assert_eq!(
        ownership.begin_write(&mut graph).unwrap(),
        BeginWrite::Acquired
    );
    mutate(&mut graph);
    ownership.discard(&mut graph);

    assert!(!ownership.is_dirty(&graph));
    assert!(lockable(&path));
}

#[test]
fn retargeting_releases_the_old_files_lease() {
    let tmp = tempfile::tempdir().unwrap();
    let from = tmp.path().join("from.kgl");
    let to = tmp.path().join("to.kgl");
    let (mut graph, mut ownership) = owned(&from, true);
    ownership.begin_write(&mut graph).unwrap();
    mutate(&mut graph);
    assert!(!lockable(&from));

    ownership.retarget(to.clone(), GraphFileIdentity::capture(&to).unwrap());

    assert!(!ownership.holds_lease());
    assert!(lockable(&from));
    assert_eq!(ownership.path(), to);
    ownership.publish(&mut graph).unwrap();
    assert!(to.is_file());
}

/// An adopted lease belongs to the caller, so a publish must not hand it back
/// — the CLI's `--save-on-exit` session holds one from open to exit and would
/// otherwise lose it at the first mid-session save.
#[test]
fn an_adopted_lease_survives_a_publish() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("adopted.kgl");
    let (mut graph, mut ownership) = owned(&path, false);
    let lease = GraphWriterLease::acquire(&path, Duration::ZERO).unwrap();
    ownership.adopt_lease(lease, &graph, true);
    mutate(&mut graph);

    ownership.publish(&mut graph).unwrap();

    assert!(ownership.holds_lease());
    assert!(!lockable(&path));
}

#[test]
fn resynced_adopts_a_caller_reloaded_graph() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("resync.kgl");
    let (mut graph, mut ownership) = owned(&path, true);
    ownership.begin_write(&mut graph).unwrap();
    mutate(&mut graph);
    std::fs::write(&path, b"replaced by a peer").unwrap();
    assert!(matches!(
        ownership.publish(&mut graph).unwrap_err(),
        WriteRefusal::Stale { .. }
    ));

    // What the caller's own open path produces, handed back.
    let mut reloaded = Arc::new(DirGraph::new());
    save_graph(&mut reloaded, &path.to_string_lossy()).unwrap();
    ownership.resynced(GraphFileIdentity::capture(&path).unwrap(), &reloaded);

    assert!(!ownership.is_dirty(&reloaded));
    ownership.publish(&mut reloaded).unwrap();
}

/// The clean counterpart of the dirty stale-publish contract: with nothing to
/// lose, the refusal must not leave a lease parked over a clean graph.
#[test]
fn a_stale_publish_of_a_clean_graph_releases_the_lease() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("clean_stale.kgl");
    let (mut graph, mut ownership) = owned(&path, true);
    std::fs::write(&path, b"replaced by a peer").unwrap();

    let refusal = ownership.publish(&mut graph).unwrap_err();
    assert!(matches!(refusal, WriteRefusal::Stale { .. }), "{refusal}");
    assert!(!ownership.holds_lease());
    assert!(lockable(&path));
}
