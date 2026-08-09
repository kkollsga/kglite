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
