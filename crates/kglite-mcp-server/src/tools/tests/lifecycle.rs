//! Graph lifecycle tests: create, mutate, save, load, write-path routes.

use std::sync::Arc;
use std::time::Duration;

use kglite::api::storage::StorageMode;

use super::*;

#[test]
fn lifecycle_create_mutate_save_load() {
    let p = tmp_kgl("lifecycle");
    let s = GraphState::default();
    // create empty → mutate via the write path → save_as
    s.create_in_mode(&p, StorageMode::Memory).unwrap();
    assert!(
        !s.is_dirty(),
        "a freshly created graph matches the file it was just written to"
    );
    let r = s
        .with_active_mut(|a| write(a, "CREATE (:Task {id:'t1', status:'todo'})", None))
        .unwrap();
    assert!(r.is_ok(), "{r:?}");
    assert!(s.is_dirty(), "an applied mutation is an unsaved change");
    s.save_as(&p).unwrap();
    assert!(!s.is_dirty(), "a save clears the unsaved-change state");
    drop(s);
    // load into a *fresh* state → the node survived (the 0.12.2 fix path too)
    let s2 = GraphState::default();
    s2.load_kgl(&p).unwrap();
    assert_eq!(s2.schema().unwrap().0, 1, "expected 1 node after reload");
    drop(s2);
    let _ = std::fs::remove_file(&p);
}

// ── `--storage` on an existing graph ────────────────────────────────────────
//
// Same contract as the wheel's `kglite.open(path, storage=...)` and the Bolt
// server's `--storage`: no request means the checkpoint decides, an explicit
// portable request converts, and a disk request on a `.kgl` is refused with
// the core reason rather than dropped. The dropped-flag shape is the one this
// crate has already had to fix twice elsewhere (`storage=` on `open()`,
// `from_blueprint(save=True)`), so it gets a test here rather than a comment.

/// The mode the active graph is actually running on.
fn active_mode(state: &GraphState) -> String {
    state
        .with_active(|active| {
            kglite::api::storage::live_storage_mode(active.kg.dir())
                .as_str()
                .to_string()
        })
        .expect("active graph")
}

/// Seed a saved graph at `path` in `mode`, then release it.
fn seed(path: &std::path::Path, mode: StorageMode) {
    let state = GraphState::default();
    state.create_in_mode(path, mode).unwrap();
    state.save_as(path).unwrap();
}

#[test]
fn open_without_a_requested_mode_serves_what_the_checkpoint_recorded() {
    let tmp = tempfile::tempdir().unwrap();
    let p = tmp.path().join("recorded.kgl");
    seed(&p, StorageMode::Mapped);

    let state = GraphState::default();
    state.open_or_create(&p, None).unwrap();
    assert_eq!(active_mode(&state), "mapped");
}

#[test]
fn explicit_portable_mode_converts_an_existing_graph() {
    for (recorded, requested, want) in [
        (StorageMode::Memory, StorageMode::Mapped, "mapped"),
        (StorageMode::Mapped, StorageMode::Memory, "memory"),
    ] {
        let tmp = tempfile::tempdir().unwrap();
        let p = tmp.path().join("convert.kgl");
        seed(&p, recorded);

        let state = GraphState::default();
        state.open_or_create(&p, Some(requested)).unwrap();
        assert_eq!(
            active_mode(&state),
            want,
            "an explicit --storage must be applied, not dropped"
        );
    }
}

#[test]
fn explicit_matching_mode_is_accepted() {
    let tmp = tempfile::tempdir().unwrap();
    let p = tmp.path().join("agree.kgl");
    seed(&p, StorageMode::Mapped);

    let state = GraphState::default();
    state.open_or_create(&p, Some(StorageMode::Mapped)).unwrap();
    assert_eq!(active_mode(&state), "mapped");
}

#[test]
fn disk_request_on_a_portable_graph_is_refused_with_the_core_reason() {
    let tmp = tempfile::tempdir().unwrap();
    let p = tmp.path().join("disk-request.kgl");
    seed(&p, StorageMode::Memory);

    let state = GraphState::default();
    let error = state
        .open_or_create(&p, Some(StorageMode::Disk))
        .expect_err("a disk request on a .kgl must not be silently ignored")
        .to_string();
    assert!(
        error.contains("enable_disk_mode()") && error.contains("directory"),
        "the refusal must carry the core reason and remedy: {error}"
    );

    // The refused open must leave the path openable — its lease is released.
    let after = GraphState::default();
    after.open_or_create(&p, None).unwrap();
    assert_eq!(active_mode(&after), "memory");
}

/// This server opens graphs with no write-ahead log attached, so a path whose
/// sidecar still holds frames the checkpoint has not folded in must be refused
/// rather than served. Opening it would hand back a graph missing committed
/// writes, and the next `save_graph` would strand those frames in front of a
/// newer checkpoint for a later durable open to replay back over it.
///
/// The refusal is the engine's (`api::io::open_or_create_graph`); asserted here
/// because inheriting it is the whole point of routing through that entry.
#[test]
fn an_unrecovered_wal_sidecar_is_refused_rather_than_served() {
    use kglite::api::durable::{wal_path, DurabilityLevel};
    use kglite::api::session::{execute_mut, CommitOutcome, ExecuteOptions, Session};

    let tmp = tempfile::tempdir().unwrap();
    let p = tmp.path().join("unrecovered.kgl");
    let path = p.to_string_lossy().into_owned();
    seed(&p, StorageMode::Memory);

    // A real durable writer commits one frame and dies before checkpointing:
    // the checkpoint just seeded carries `checkpoint_lsn` 0, so that frame is
    // unfolded, and dropping the session is the crash (no save, no truncate).
    {
        let graph = kglite::api::io::load_file(&path).unwrap();
        let session = Session::open_durable(graph, &path, DurabilityLevel::Full).unwrap();
        let mut tx = session.begin();
        let params = std::collections::HashMap::new();
        let opts = ExecuteOptions::eager(&params);
        execute_mut(tx.working_mut().unwrap(), "CREATE (:Task {id:'t1'})", &opts).unwrap();
        assert!(matches!(
            session.commit(tx, true),
            CommitOutcome::Committed { .. }
        ));
    }

    let state = GraphState::default();
    let error = state
        .open_or_create(&p, None)
        .expect_err("a log-less open over unfolded frames must be refused")
        .to_string();
    assert!(
        error.contains("unrecovered.kgl-wal")
            && error.contains("holds commits this checkpoint does not contain"),
        "the refusal must carry the core reason: {error}"
    );

    // The named remedy works: with the sidecar moved aside, the same open
    // succeeds — and the refusal released the writer lease it took.
    std::fs::rename(wal_path(&p), tmp.path().join("aside")).unwrap();
    GraphState::default().open_or_create(&p, None).unwrap();
}

/// Creating a graph *is* a write, so the lease is taken at the open and held
/// until the first save publishes what was created — the one window where a
/// half-built graph must not be raced.
#[test]
fn a_created_graph_holds_the_writer_lease_until_its_first_save() {
    let tmp = tempfile::tempdir().unwrap();
    let p = tmp.path().join("retained_lease.kgl");
    let state = GraphState::default();
    state.create_in_mode(&p, StorageMode::Memory).unwrap();
    assert_eq!(Arc::strong_count(&state.inner), 1);
    assert!(kglite::api::io::GraphWriterLease::acquire(&p, Duration::ZERO).is_err());
    state.save_as(&p).unwrap();
    kglite::api::io::GraphWriterLease::acquire(&p, Duration::ZERO)
        .expect("the created graph's lease is released once it is published");
    drop(state);
}

/// The inversion this program exists for: *opening* an existing file is not a
/// write, so it takes no lease. Four MCP clients can serve one `.kgl`; the
/// lease appears only around a real unsaved change.
#[test]
fn an_opened_path_backed_graph_takes_no_lease_until_it_writes() {
    let tmp = tempfile::tempdir().unwrap();
    let p = tmp.path().join("lazy_lease.kgl");
    seed_with_nodes(&p, 1);

    let state = GraphState::default();
    state.open_or_create(&p, None).unwrap();
    assert!(
        external_lease_is_available(&p),
        "an opened graph must not lock the file before it has anything to save"
    );

    state
        .with_active_mut(|a| write(a, "CREATE (:Task {id:'t1'})", None))
        .unwrap()
        .unwrap();
    assert!(
        !external_lease_is_available(&p),
        "the first unsaved change takes the lease"
    );

    state.with_active_mut(run_save).unwrap().unwrap();
    assert!(
        external_lease_is_available(&p),
        "publishing gives the lease back — the dirty window is over"
    );
}

#[test]
fn load_graph_swaps_active() {
    let pa = tmp_kgl("swapA");
    let pb = tmp_kgl("swapB");
    // build two distinct graphs on disk
    for (p, n) in [(&pa, 1u64), (&pb, 3u64)] {
        let s = GraphState::default();
        s.create_in_mode(p, StorageMode::Memory).unwrap();
        for i in 0..n {
            s.with_active_mut(|a| write(a, &format!("CREATE (:N {{id:'{i}'}})"), None))
                .unwrap()
                .unwrap();
        }
        s.save_as(p).unwrap();
    }
    // one state loads A then B → active reflects B
    let s = GraphState::default();
    s.load_kgl(&pa).unwrap();
    assert_eq!(s.schema().unwrap().0, 1);
    s.load_kgl(&pb).unwrap();
    assert_eq!(s.schema().unwrap().0, 3, "load_graph should swap to B");
    drop(s);
    let _ = std::fs::remove_file(&pa);
    let _ = std::fs::remove_file(&pb);
}

// ── reload: re-opening the active graph's own source path ───────────────────
//
// What the `reload_graph` tool does, at the state level: `open_or_create` on
// the path already active, with `requested_mode: None`. These pin the three
// properties the tool's contract rests on — the new bytes win, the writer
// lease and the bound embedder survive, and a failed re-read leaves the
// previous graph serving.

/// Build a graph of `nodes` nodes at `path`, then release it.
fn seed_with_nodes(path: &std::path::Path, nodes: u64) {
    let s = GraphState::default();
    s.create_in_mode(path, StorageMode::Memory).unwrap();
    for i in 0..nodes {
        s.with_active_mut(|a| write(a, &format!("CREATE (:N {{id:'{i}'}})"), None))
            .unwrap()
            .unwrap();
    }
    s.save_as(path).unwrap();
}

/// The active graph's generation counter — bumped by every swap. Reads the
/// same accessor the `reload_graph` response reports from.
fn generation(state: &GraphState) -> u64 {
    state.generation().expect("a graph must be active")
}

#[test]
fn reload_serves_an_externally_rewritten_file() {
    let tmp = tempfile::tempdir().unwrap();
    let p = tmp.path().join("reloaded.kgl");
    let outside = tmp.path().join("rebuilt.kgl");
    seed_with_nodes(&p, 1);
    seed_with_nodes(&outside, 3);

    let s = GraphState::default();
    s.bind_embedder(Arc::new(TestEmbedder)).unwrap();
    s.open_or_create(&p, None).unwrap();
    assert_eq!(s.schema().unwrap().0, 1);
    let generation_before = generation(&s);

    // An external producer republishes the served path. kglite's own save is a
    // rename-over, and the writer lease lives on a `<path>.lock` sidecar, so
    // this is the same on-disk event a real rebuild produces.
    std::fs::rename(&outside, &p).unwrap();
    assert_eq!(
        s.schema().unwrap().0,
        1,
        "a rewritten file must not change the served graph until it is re-read"
    );

    s.open_or_create(&p, None).unwrap();

    assert_eq!(s.schema().unwrap().0, 3, "reload must serve the new bytes");
    assert_eq!(
        generation(&s),
        generation_before + 1,
        "a reload installs a new graph and must bump the generation"
    );
    // Nothing was ever mutated here, so no lease was ever taken — and the
    // reload carries that "no lease" across the swap rather than acquiring
    // one. A clean reader must leave the file lockable on both sides of a
    // re-read; that is what lets the external producer republish it again.
    assert!(
        read_lock(&s.inner)
            .as_ref()
            .is_some_and(|active| active.ownership.is_some()),
        "a write-enabled state keeps its write ownership across a reload"
    );
    assert!(
        external_lease_is_available(&p),
        "a clean reload must not have quietly acquired the path's writer lease"
    );
    // A reload is a graph swap, so it inherits the B2 fix: the boot-bound
    // embedder is re-applied to the fresh handle.
    assert!(
        s.with_kg(|kg| kg.embedder().is_some()).unwrap(),
        "the bound embedder must survive a reload"
    );
    let after = text_score_result(&s);
    assert!(
        !missing_embedder(&after),
        "text_score must still resolve after a reload, got {after:?}"
    );
}

#[test]
fn a_failed_reload_keeps_the_previous_graph_serving() {
    let tmp = tempfile::tempdir().unwrap();
    let p = tmp.path().join("torn.kgl");
    seed_with_nodes(&p, 2);

    let s = GraphState::default();
    s.open_or_create(&p, None).unwrap();
    assert_eq!(s.schema().unwrap().0, 2);
    let generation_before = generation(&s);

    // A producer writing the file non-atomically (or a truncated copy) leaves
    // bytes that cannot be opened.
    std::fs::write(&p, b"not a kgl file").unwrap();
    let error = s
        .open_or_create(&p, None)
        .expect_err("an unreadable file must not be installed")
        .to_string();
    assert!(
        error.contains("kglite graph open/create failed"),
        "the failure must carry the core reason: {error}"
    );

    // Every load failure returns *before* the write lock is taken, so the
    // previous graph is untouched — still serving, still the same generation.
    assert_eq!(
        s.schema().unwrap().0,
        2,
        "a failed reload must leave the old graph active"
    );
    assert_eq!(
        generation(&s),
        generation_before,
        "a failed reload installs nothing and must not bump the generation"
    );
}

// ── writer-lease scoping: who owns the served path ──────────────────────────
//
// A `--writable` / `save_graph` server owns the file it serves and holds the
// cross-process lease for its lifetime. A read-only `--graph` server never
// writes that file, and holding the lease there only refuses the external
// rebuilder that wants to republish it (`kglite.open(path)` fails fast, and
// its error names nothing about this server).
//
// Gap, deliberately not covered here: the disk-graph *directory* case, which
// keeps the lease under either policy. Building a disk graph needs the
// multi-file arena/`CURRENT` scaffolding, which is exercised by the engine's
// own storage tests and by `tests/test_storage_parity.py`; the branch it takes
// here is one `path.is_file()` check shared with the missing-path case below.

/// Whether a process that is *not* this state can take the path's writer lease
/// right now. Fail-fast (`Duration::ZERO`) — a contended lease answers
/// immediately, exactly as `kglite.open(path)` asks it to.
fn external_lease_is_available(path: &std::path::Path) -> bool {
    kglite::api::io::GraphWriterLease::acquire(path, Duration::ZERO).is_ok()
}

#[test]
fn a_read_only_state_leaves_the_served_file_lockable() {
    let tmp = tempfile::tempdir().unwrap();
    let p = tmp.path().join("read_only.kgl");
    seed_with_nodes(&p, 1);

    let s = GraphState::default().with_writer_lease_policy(WriterLeasePolicy::ReadOnly);
    s.open_or_create(&p, None).unwrap();
    assert_eq!(s.schema().unwrap().0, 1, "the graph must be served");

    assert!(
        read_lock(&s.inner)
            .as_ref()
            .is_some_and(|active| active.ownership.is_none()),
        "a read-only state must carry no write ownership of a regular-file graph"
    );
    assert!(
        external_lease_is_available(&p),
        "an external rebuilder must be able to lock the file a read-only server serves"
    );
}

#[test]
fn a_write_enabled_state_locks_the_served_file_only_while_it_is_dirty() {
    let tmp = tempfile::tempdir().unwrap();
    let p = tmp.path().join("write_enabled.kgl");
    seed_with_nodes(&p, 1);

    // The default policy — what every constructor that declares nothing gets.
    let s = GraphState::default();
    assert_eq!(s.writer_lease_policy, WriterLeasePolicy::Exclusive);
    s.open_or_create(&p, None).unwrap();

    assert!(
        external_lease_is_available(&p),
        "a write-enabled server that has written nothing must not refuse a peer"
    );
    s.with_active_mut(|a| write(a, "CREATE (:Task {id:'t1'})", None))
        .unwrap()
        .unwrap();
    assert!(
        !external_lease_is_available(&p),
        "an unsaved change must refuse a second writer — that is the lost-update window"
    );
    s.with_active_mut(run_save).unwrap().unwrap();
    assert!(
        external_lease_is_available(&p),
        "and the window closes at the save, not at process exit"
    );
    drop(s);
}

/// A write refused by the engine must not leave the lease parked over a graph
/// nothing landed on. `make_dir_graph_mut` bumps the version *before* the
/// statement runs, so without the rollback the server would look permanently
/// dirty — and hold the file — over a mutation that never happened.
#[test]
fn a_failed_first_write_gives_the_lease_back() {
    let tmp = tempfile::tempdir().unwrap();
    let p = tmp.path().join("failed_first_write.kgl");
    seed_with_nodes(&p, 1);

    let s = GraphState::default();
    s.open_or_create(&p, None).unwrap();
    let scope = vec!["Task".to_string()];
    let refusal = s
        .with_active_mut(|a| write(a, "CREATE (:Algorithm {id:'a1'})", Some(&scope)))
        .unwrap()
        .expect_err("an out-of-scope write must be refused");
    assert!(refusal.contains("write scope"), "{refusal}");

    assert!(
        external_lease_is_available(&p),
        "a refused write must release the lease it took to attempt the write"
    );
    assert!(
        !s.is_dirty(),
        "a refused write leaves nothing unsaved, so the server must not report changes"
    );
}

/// Two servers on one file: the second one's write is refused by name, and the
/// refusal tells it what to do — which becomes possible as soon as the first
/// one saves.
#[test]
fn a_second_server_is_refused_by_name_until_the_first_one_saves() {
    let tmp = tempfile::tempdir().unwrap();
    let p = tmp.path().join("two_servers.kgl");
    seed_with_nodes(&p, 1);

    let first = GraphState::default().with_lease_label(Some("Claude Desktop".to_string()));
    let second = GraphState::default().with_lease_label(Some("Codex".to_string()));
    first.open_or_create(&p, None).unwrap();
    second.open_or_create(&p, None).unwrap();

    first
        .with_active_mut(|a| write(a, "CREATE (:Task {id:'t1'})", None))
        .unwrap()
        .unwrap();

    // The label the refusal renders cross-process. Both states live in *this*
    // process, where `LeaseHolder::describe` deliberately says "this same
    // process" instead of naming a label — so the label is asserted where it
    // is produced (the owner record), and the rendered form is pinned by the
    // cross-process test in `tests/test_mcp_server_smoke.py`.
    let owner = std::fs::read_to_string(p.with_extension("kgl.lock-owner"))
        .expect("the holder publishes an owner record");
    assert!(
        owner.contains("label=Claude Desktop"),
        "the holding server must name itself for a peer to read: {owner}"
    );

    let refusal = second
        .with_active_mut(|a| write(a, "CREATE (:Task {id:'t2'})", None))
        .unwrap()
        .expect_err("a peer holding the lease must refuse the second write");
    assert!(
        refusal.contains("only one process may write"),
        "the refusal identifies the holder: {refusal}"
    );
    assert!(
        refusal.contains("Nothing was changed here"),
        "the refusal states that no data was lost: {refusal}"
    );
    assert!(
        refusal.contains("reload_graph"),
        "the refusal names the way out: {refusal}"
    );
    // The refused server is still a working reader.
    assert_eq!(second.schema().unwrap().0, 1);

    first.with_active_mut(run_save).unwrap().unwrap();
    // The second server was loaded from the pre-save bytes, so its write is
    // now refused for *staleness* rather than contention — and that refusal
    // names the reload, not the holder.
    let stale = second
        .with_active_mut(|a| write(a, "CREATE (:Task {id:'t2'})", None))
        .unwrap()
        .expect_err("a file rewritten since load must not be written over");
    assert!(stale.contains("changed on disk"), "{stale}");
    assert!(stale.contains("reload_graph"), "{stale}");

    second.open_or_create(&p, None).unwrap();
    second
        .with_active_mut(|a| write(a, "CREATE (:Task {id:'t2'})", None))
        .unwrap()
        .expect("after refreshing, the second server writes normally");
}

/// `save_graph_as` is the escape hatch from a jammed file, so it must not keep
/// the jam: retargeting releases the source path's lease.
#[test]
fn save_as_releases_the_source_files_lease() {
    let tmp = tempfile::tempdir().unwrap();
    let source = tmp.path().join("source.kgl");
    let elsewhere = tmp.path().join("elsewhere.kgl");
    seed_with_nodes(&source, 1);

    let s = GraphState::default();
    s.open_or_create(&source, None).unwrap();
    s.with_active_mut(|a| write(a, "CREATE (:Task {id:'t1'})", None))
        .unwrap()
        .unwrap();
    assert!(!external_lease_is_available(&source));

    s.save_as(&elsewhere).unwrap();

    assert!(
        external_lease_is_available(&source),
        "the graph is not going back to the source path — its lease must be released"
    );
    assert!(
        external_lease_is_available(&elsewhere),
        "and the published target's lease is released like any other save"
    );
    assert!(!s.is_dirty(), "the work is saved, just somewhere else");
    // The bytes went to the *new* path, and the source was left exactly as the
    // peer this manoeuvre exists to avoid would find it.
    let written = GraphState::default();
    written.open_or_create(&elsewhere, None).unwrap();
    assert_eq!(
        written.schema().unwrap().0,
        2,
        "the new path holds the work"
    );
    drop(written);
    let untouched = GraphState::default();
    untouched.open_or_create(&source, None).unwrap();
    assert_eq!(
        untouched.schema().unwrap().0,
        1,
        "save_graph_as must not have written the source path as well"
    );
}

// ── unsaved changes vs. routes that replace the active graph ────────────────

/// A reload discards whatever is unsaved, so it refuses rather than doing it
/// silently — and names the flag that says "yes, discard".
#[test]
fn a_dirty_state_refuses_a_reload_until_the_discard_is_explicit() {
    let tmp = tempfile::tempdir().unwrap();
    let p = tmp.path().join("dirty_reload.kgl");
    seed_with_nodes(&p, 1);

    let s = GraphState::default();
    s.open_or_create(&p, None).unwrap();
    s.with_active_mut(|a| write(a, "CREATE (:Task {id:'t1'})", None))
        .unwrap()
        .unwrap();
    assert!(s.is_dirty());

    let refusal = refused_while_dirty("reload_graph");
    assert!(refusal.contains("discard_unsaved=true"), "{refusal}");
    assert!(refusal.contains("save_graph"), "{refusal}");

    // The explicit discard restores the file's version of the graph and hands
    // the lease back — without reading the file, so it works even when the
    // file cannot be read.
    assert!(
        s.discard_unsaved_changes(),
        "a snapshot was there to restore"
    );
    assert!(!s.is_dirty());
    assert_eq!(
        s.schema().unwrap().0,
        1,
        "the discarded CREATE must be gone from the served graph"
    );
    assert!(
        external_lease_is_available(&p),
        "discarding releases the lease the discarded change was holding"
    );
}

/// The plan-cache trap: `graph_id` survives the rollback, and the cache is
/// keyed `(graph_id, version)`, so a restored graph must land *above* every
/// version the discarded lineage reached rather than back at its baseline.
#[test]
fn a_discard_lands_above_the_versions_the_discarded_writes_reached() {
    let tmp = tempfile::tempdir().unwrap();
    let p = tmp.path().join("discard_version.kgl");
    seed_with_nodes(&p, 1);

    let s = GraphState::default();
    s.open_or_create(&p, None).unwrap();
    let baseline = s.with_kg(|kg| kg.dir().version()).unwrap();
    for i in 0..3 {
        s.with_active_mut(|a| write(a, &format!("CREATE (:Task {{id:'t{i}'}})"), None))
            .unwrap()
            .unwrap();
    }
    let dirty_version = s.with_kg(|kg| kg.dir().version()).unwrap();
    assert!(dirty_version > baseline);

    s.discard_unsaved_changes();

    assert!(
        s.with_kg(|kg| kg.dir().version()).unwrap() > dirty_version,
        "a rollback that re-used a version the discarded lineage cached against \
         would serve that lineage's plans"
    );
}

#[test]
fn a_read_only_state_creating_a_missing_graph_still_takes_the_lease() {
    let tmp = tempfile::tempdir().unwrap();
    let p = tmp.path().join("created.kgl");

    // Creating the graph *is* a write, whatever the tool surface allows later,
    // so the lease is taken even under the read-only policy.
    let s = GraphState::default().with_writer_lease_policy(WriterLeasePolicy::ReadOnly);
    s.create_in_mode(&p, StorageMode::Memory).unwrap();
    assert!(
        !external_lease_is_available(&p),
        "a created graph is owned by the process that created it"
    );
}

#[test]
fn a_leaseless_reload_still_serves_the_new_bytes() {
    let tmp = tempfile::tempdir().unwrap();
    let p = tmp.path().join("leaseless_reload.kgl");
    let outside = tmp.path().join("rebuilt.kgl");
    seed_with_nodes(&p, 1);
    seed_with_nodes(&outside, 3);

    let s = GraphState::default().with_writer_lease_policy(WriterLeasePolicy::ReadOnly);
    s.open_or_create(&p, None).unwrap();
    let generation_before = generation(&s);

    // The rebuild an unleased path makes possible: an external process locks
    // the file, republishes it, and releases.
    {
        let _external = kglite::api::io::GraphWriterLease::acquire(&p, Duration::ZERO)
            .expect("the external rebuilder must get the lease");
        std::fs::rename(&outside, &p).unwrap();
    }

    // Same-path reload with no lease to carry across: the `reuse_existing`
    // branch takes `None` from the old slot and the swap proceeds.
    s.open_or_create(&p, None).unwrap();
    assert_eq!(s.schema().unwrap().0, 3, "reload must serve the new bytes");
    assert_eq!(generation(&s), generation_before + 1);
    assert!(
        external_lease_is_available(&p),
        "the reload must not have quietly acquired a lease"
    );
}

#[test]
fn write_path_creates_and_reads_back() {
    let mut a = fresh_active();
    write(&mut a, "CREATE (:Task {id:'t1', status:'todo'})", None).unwrap();
    // A subsequent read on the writable path observes the mutation.
    let out = write(&mut a, "MATCH (t:Task) RETURN count(t) AS c", None).unwrap();
    assert!(out.contains('1'), "expected 1 task, got: {out}");
}

#[test]
fn write_with_no_return_acknowledges_stats() {
    // A CREATE/SET/MERGE with no RETURN must NOT read back "No results"
    // (indistinguishable from a no-op match) — it acknowledges the write.
    let mut a = fresh_active();
    let out = write(&mut a, "CREATE (:Task {id:'t1'})", None).unwrap();
    assert_eq!(
        out,
        format!(
            "OK: 1 node(s) created. [engine {}]{}",
            env!("CARGO_PKG_VERSION"),
            // The ack self-identifies its graph like every read does: an agent
            // that mutated the wrong graph has no later call that tells it so.
            a.identity_footer()
        ),
        "write ACK is a legacy text contract"
    );
    // SET acks too.
    let out = write(&mut a, "MATCH (t:Task{id:'t1'}) SET t.status='done'", None).unwrap();
    assert_eq!(
        out,
        format!(
            "OK: 1 property(ies) set. [engine {}]{}",
            env!("CARGO_PKG_VERSION"),
            a.identity_footer()
        )
    );
    // A read that matches nothing still says "No results" (distinct signal) —
    // and, since `Nope` is a type this graph has never had, says why.
    let out = write(&mut a, "MATCH (x:Nope) RETURN x", None).unwrap();
    assert!(out.starts_with("No results.\n"), "{out}");
    assert!(out.contains("unknown node label 'Nope'"), "{out}");
    // A read that matches nothing against a type that DOES exist gets the
    // bare signal — the warning block is earned, not decoration.
    let out = write(&mut a, "MATCH (t:Task {id:'absent'}) RETURN t", None).unwrap();
    assert_eq!(out, "No results.");
}

#[test]
fn write_scope_blocks_out_of_scope_create() {
    let mut a = fresh_active();
    let scope = vec!["Plan".to_string(), "Task".to_string()];
    // In-scope is allowed.
    write(&mut a, "CREATE (:Task {id:'t1'})", Some(&scope)).unwrap();
    // Out-of-scope is rejected.
    let err = write(&mut a, "CREATE (:Algorithm {id:'a1'})", Some(&scope)).unwrap_err();
    assert!(
        err.contains("write scope"),
        "expected scope error, got: {err}"
    );
    // The rejected CREATE did not land.
    let out = write(&mut a, "MATCH (n:Algorithm) RETURN count(n) AS c", None).unwrap();
    assert!(
        out.contains('0') || out.contains("No results"),
        "got: {out}"
    );
}

// ── operator-pinned write scope ─────────────────────────────────────────────
//
// The agent's `write_scope` argument is role hygiene (it picked it); the
// operator's `--write-scope` / `extensions.write_scope` pin is access control
// (the agent cannot reach it). These pin the four combinations at the same
// seam the `cypher_query` route calls.

fn scope(names: &[&str]) -> Vec<String> {
    names.iter().map(|s| (*s).to_string()).collect()
}

#[test]
fn operator_scope_refuses_an_out_of_scope_agent_write() {
    let mut a = fresh_active();
    let pin = scope(&["Plan", "Task"]);
    // In-scope for both parties: allowed.
    write_pinned(&mut a, "CREATE (:Task {id:'t1'})", Some(&pin), Some(&pin)).unwrap();
    // The agent asks for a type the operator never pinned.
    let agent = scope(&["Algorithm"]);
    let err = write_pinned(
        &mut a,
        "CREATE (:Algorithm {id:'a1'})",
        Some(&pin),
        Some(&agent),
    )
    .unwrap_err();
    assert!(
        err.starts_with("no writes permitted under this server's write scope"),
        "the refusal must name the server's scope: {err}"
    );
    assert!(err.contains("[Plan, Task]"), "{err}");
    assert!(
        write(&mut a, "MATCH (n:Algorithm) RETURN count(n) AS c", None)
            .unwrap()
            .contains('0'),
        "the refused CREATE must not have landed"
    );
}

#[test]
fn operator_scope_applies_when_the_agent_omits_its_own() {
    // The fail-open this pin exists to close: before it, an agent that simply
    // left `write_scope` out wrote anything the server could write.
    let mut a = fresh_active();
    let pin = scope(&["Plan", "Task"]);
    write_pinned(&mut a, "CREATE (:Task {id:'t1'})", Some(&pin), None).unwrap();
    let err = write_pinned(&mut a, "CREATE (:Algorithm {id:'a1'})", Some(&pin), None).unwrap_err();
    assert!(
        err.contains("write scope"),
        "an omitted agent scope must not fall back to unrestricted: {err}"
    );
    assert!(
        write(&mut a, "MATCH (n:Algorithm) RETURN count(n) AS c", None)
            .unwrap()
            .contains('0'),
        "the refused CREATE must not have landed"
    );
}

#[test]
fn operator_and_agent_scopes_intersect() {
    let mut a = fresh_active();
    let pin = scope(&["Plan", "Task"]);
    let agent = scope(&["Task", "Algorithm"]);
    // In both lists: allowed.
    write_pinned(&mut a, "CREATE (:Task {id:'t1'})", Some(&pin), Some(&agent)).unwrap();
    // In the pin but not the agent's list: the agent narrowed itself out.
    let err =
        write_pinned(&mut a, "CREATE (:Plan {id:'p1'})", Some(&pin), Some(&agent)).unwrap_err();
    assert!(err.contains("write scope"), "{err}");
    // In the agent's list but not the pin: the agent cannot widen.
    let err = write_pinned(
        &mut a,
        "CREATE (:Algorithm {id:'a1'})",
        Some(&pin),
        Some(&agent),
    )
    .unwrap_err();
    assert!(err.contains("write scope"), "{err}");
}

#[test]
fn an_empty_intersection_refuses_before_the_mutation_runs() {
    let mut a = fresh_active();
    write(&mut a, "CREATE (:Task {id:'t1'})", None).unwrap();
    let pin = scope(&["Plan"]);
    let agent = scope(&["Algorithm"]);
    let err = write_pinned(
        &mut a,
        "MATCH (t:Task {id:'t1'}) DETACH DELETE t",
        Some(&pin),
        Some(&agent),
    )
    .unwrap_err();
    assert_eq!(
        err,
        "no writes permitted under this server's write scope [Plan]: the requested write_scope \
         [Algorithm] shares no node type with it",
        "the refusal text is the agent's only view of the server's pin"
    );
    assert!(
        write(&mut a, "MATCH (t:Task) RETURN count(t) AS c", None)
            .unwrap()
            .contains('1'),
        "refusal must precede the mutation"
    );
}

#[test]
fn an_operator_pin_leaves_reads_alone() {
    let mut a = fresh_active();
    write(&mut a, "CREATE (:Task {id:'t1'})", None).unwrap();
    let pin = scope(&["Plan"]);
    let out = write_pinned(
        &mut a,
        "MATCH (t:Task) RETURN count(t) AS c",
        Some(&pin),
        None,
    )
    .expect("a pin restricts writes, not reads");
    assert!(out.contains('1'), "{out}");
}

#[test]
fn new_edge_type_via_write_path_registers() {
    // The 0.12.2 edge-persistence fix in action through the MCP write path:
    // a brand-new relationship type is registered (queryable, would persist).
    let mut a = fresh_active();
    write(&mut a, "CREATE (:Task {id:'t'})", None).unwrap();
    write(&mut a, "CREATE (:Spec {id:'s'})", None).unwrap();
    write(
        &mut a,
        "MATCH (t:Task{id:'t'}),(s:Spec{id:'s'}) CREATE (t)-[:IMPLEMENTS_SPEC]->(s)",
        None,
    )
    .unwrap();
    let out = write(
        &mut a,
        "MATCH (:Task)-[:IMPLEMENTS_SPEC]->() RETURN count(*) AS c",
        None,
    )
    .unwrap();
    assert!(out.contains('1'), "expected 1 edge, got: {out}");
}

// ── manifest ontology: configuration, not an unsaved change ─────────────────

/// `Person` nodes for the ontology to hang a supertype on.
fn seed_people(path: &std::path::Path, mode: StorageMode) {
    let s = GraphState::default();
    s.create_in_mode(path, mode).unwrap();
    for i in 0..3 {
        s.with_active_mut(|a| write(a, &format!("CREATE (:Person {{id:'p{i}'}})"), None))
            .unwrap()
            .unwrap();
    }
    s.save_as(path).unwrap();
}

/// The manifest ontology the two tests below bind: `Person` is an `Agent`, so
/// materializing it stamps an `Agent` label on every person.
fn agent_ontology(materialize: bool) -> BoundOntology {
    let store = kglite::api::ontology_from_json(
        r#"{"classes": {"Agent": {"abstract": true}, "Person": {"is_a": "Agent"}}}"#,
    )
    .expect("ontology parses");
    BoundOntology {
        store: Arc::new(store),
        materialize,
    }
}

/// How many nodes carry the materialized supertype label.
fn agent_count(state: &GraphState) -> String {
    state
        .with_kg(|kg| {
            run_cypher_inner(
                kg,
                "MATCH (a:Agent) RETURN count(a) AS c",
                std::collections::HashMap::new(),
                ExecPolicy::default(),
                CSV_OFF,
            )
        })
        .expect("a graph must be active")
        .expect("the label query must run")
}

/// Boot-time materialization is *server configuration*, not an agent's change:
/// it must not make the server report unsaved work it did not do. And because
/// a clean graph still publishes, an explicit save is still how that
/// materialization reaches disk.
#[test]
fn a_materializing_server_boots_clean_and_its_clean_save_persists_the_labels() {
    let tmp = tempfile::tempdir().unwrap();
    let p = tmp.path().join("materialized.kgl");
    seed_people(&p, StorageMode::Memory);

    let s = GraphState::default();
    s.bind_ontology(agent_ontology(true));
    s.open_or_create(&p, None).unwrap();

    assert!(
        agent_count(&s).contains('3'),
        "the boot materialization must have stamped the supertype"
    );
    assert!(
        !s.is_dirty(),
        "a manifest ontology is configuration — a materializing server must boot clean"
    );
    assert!(
        external_lease_is_available(&p),
        "and a clean server must not be holding the file"
    );

    s.with_active_mut(run_save)
        .unwrap()
        .expect("a clean graph still publishes");
    drop(s);

    // A plain reader with no ontology bound sees the stamped labels, so the
    // save really did carry the materialization to disk.
    let plain = GraphState::default();
    plain.open_or_create(&p, None).unwrap();
    assert!(
        agent_count(&plain).contains('3'),
        "save_graph on a clean materializing server must still write the labels"
    );
}

/// A mapped-mode graph forces the `Arc::make_mut` inside `apply_bound_embedder`
/// down its deep-copy path; doing that without re-adopting the writer lineage
/// left the ontology applied to a graph the state no longer served.
#[test]
fn a_mapped_graph_boots_with_a_materialized_ontology() {
    let tmp = tempfile::tempdir().unwrap();
    let p = tmp.path().join("mapped_materialized.kgl");
    seed_people(&p, StorageMode::Mapped);

    let s = GraphState::default();
    s.bind_ontology(agent_ontology(true));
    s.open_or_create(&p, None).unwrap();

    assert_eq!(
        active_mode(&s),
        "mapped",
        "the fixture must exercise mapped"
    );
    assert!(
        agent_count(&s).contains('3'),
        "a mapped graph must serve the boot-materialized labels"
    );
    assert!(
        !s.is_dirty(),
        "boot materialization is not an unsaved change"
    );
}

// ── embedder survival across graph swaps ────────────────────────────────────

/// Deterministic two-dimension embedder: enough for `text_score()` to embed
/// its query text, with no model download and no I/O.
struct TestEmbedder;

impl kglite::api::Embedder for TestEmbedder {
    fn dimension(&self) -> usize {
        2
    }

    fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, String> {
        Ok(texts.iter().map(|t| vec![t.len() as f32, 1.0]).collect())
    }

    fn model_id(&self) -> Option<String> {
        Some("test/deterministic".into())
    }
}

/// Run a `text_score()` read through the same seam the `cypher_query` tool
/// uses (`opts.embedder = kg.embedder().cloned()`). The engine resolves the
/// query text into a vector *before* matching, so an empty graph still
/// exercises the embedder lookup.
fn text_score_result(state: &GraphState) -> Result<String, String> {
    state
        .with_kg(|kg| {
            run_cypher_inner(
                kg,
                "MATCH (d:Doc) RETURN text_score(d, 'body', 'hello') AS s",
                std::collections::HashMap::new(),
                ExecPolicy::default(),
                CSV_OFF,
            )
        })
        .expect("a graph must be active")
}

/// Whether the engine refused the query for want of a bound embedder — the
/// exact user-visible symptom of a dropped binding.
fn missing_embedder(outcome: &Result<String, String>) -> bool {
    outcome
        .as_ref()
        .err()
        .is_some_and(|e| e.contains("requires a registered embedding model"))
}

#[test]
fn bound_embedder_survives_a_graph_swap() {
    // `bind_manifest_embedder` binds once at boot; the `load_graph` tool then
    // swaps a *fresh* `KnowledgeGraph` (embedder: None) into the slot. Before
    // the fix, `text_score()` silently died at the first swap.
    let pa = tmp_kgl("embedder_swap_a");
    let pb = tmp_kgl("embedder_swap_b");
    // The swap target has to exist on disk: `load_graph` passes no storage
    // mode, so `open_or_create` opens rather than creates.
    let seed = GraphState::default();
    seed.create_in_mode(&pb, StorageMode::Memory).unwrap();
    seed.save_as(&pb).unwrap();
    drop(seed);

    let s = GraphState::default();
    s.create_in_mode(&pa, StorageMode::Memory).unwrap();
    s.bind_embedder(Arc::new(TestEmbedder)).unwrap();
    assert!(
        s.with_kg(|kg| kg.embedder().is_some()).unwrap(),
        "bind_embedder must reach the active graph"
    );
    let before = text_score_result(&s);
    assert!(
        !missing_embedder(&before),
        "sanity: text_score resolves before the swap, got {before:?}"
    );

    s.load_kgl(&pb).unwrap();

    assert!(
        s.with_kg(|kg| kg.embedder().is_some()).unwrap(),
        "the boot-bound embedder must survive a load_graph swap"
    );
    let after = text_score_result(&s);
    assert!(
        !missing_embedder(&after),
        "text_score must still resolve after load_graph, got {after:?}"
    );
    drop(s);
    let _ = std::fs::remove_file(&pa);
    let _ = std::fs::remove_file(&pb);
}

#[test]
fn embedder_bound_before_any_graph_reaches_the_first_one() {
    // Boot order: `bind_manifest_embedder` runs after tool registration, and
    // modes that install no graph at boot bind against an empty slot. The
    // binding is deferred, as the code has always claimed — not discarded.
    let p = tmp_kgl("embedder_deferred");
    let s = GraphState::default();
    s.bind_embedder(Arc::new(TestEmbedder)).unwrap();
    s.create_in_mode(&p, StorageMode::Memory).unwrap();
    assert!(
        s.with_kg(|kg| kg.embedder().is_some()).unwrap(),
        "an embedder bound before the first graph must reach it"
    );
    drop(s);
    let _ = std::fs::remove_file(&p);
}

#[test]
fn bound_embedder_survives_a_workspace_rebuild() {
    // The workspace producer publishes its own fresh `KnowledgeGraph`; a lazy
    // rebuild is a graph swap like any other.
    let s = GraphState::new(Some(WorkspaceGraphMode::LocalWorkspace))
        .with_workspace_graph(Some(test_hooks()));
    let workspace = tempfile::tempdir().expect("workspace tempdir");
    let root = workspace.path().to_path_buf();
    s.bind_embedder(Arc::new(TestEmbedder)).unwrap();
    s.build_workspace_graph(&root, None)
        .expect("install workspace graph");
    assert!(
        s.with_kg(|kg| kg.embedder().is_some()).unwrap(),
        "the boot-bound embedder must survive a workspace graph build"
    );
}
