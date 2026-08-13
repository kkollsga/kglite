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
    let r = s
        .with_active_mut(|a| write(a, "CREATE (:Task {id:'t1', status:'todo'})", None))
        .unwrap();
    assert!(r.is_ok(), "{r:?}");
    s.save_as(&p).unwrap();
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
    state.with_active(|active| {
        kglite::api::storage::live_storage_mode(active.kg.dir())
            .as_str()
            .to_string()
    })
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

#[test]
fn path_backed_active_graph_retains_writer_lease() {
    let tmp = tempfile::tempdir().unwrap();
    let p = tmp.path().join("retained_lease.kgl");
    let state = GraphState::default();
    state.create_in_mode(&p, StorageMode::Memory).unwrap();
    assert_eq!(Arc::strong_count(&state.inner), 1);
    assert!(kglite::api::io::GraphWriterLease::acquire(&p, Duration::ZERO).is_err());
    drop(state);
    kglite::api::io::GraphWriterLease::acquire(&p, Duration::ZERO).unwrap();
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
            "OK: 1 node(s) created. [engine {}]",
            env!("CARGO_PKG_VERSION")
        ),
        "write ACK is a legacy text contract"
    );
    // SET acks too.
    let out = write(&mut a, "MATCH (t:Task{id:'t1'}) SET t.status='done'", None).unwrap();
    assert_eq!(
        out,
        format!(
            "OK: 1 property(ies) set. [engine {}]",
            env!("CARGO_PKG_VERSION")
        )
    );
    // A read that matches nothing still says "No results" (distinct signal).
    let out = write(&mut a, "MATCH (x:Nope) RETURN x", None).unwrap();
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
