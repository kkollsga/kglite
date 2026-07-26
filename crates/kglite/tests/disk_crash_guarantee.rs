//! What a `SIGKILL` costs a `storage="disk"` graph.
//!
//! Disk mode has no write-ahead log: `kglite.open(path, storage="disk")`
//! opens non-durable and `durable=True` is refused. That is often read as
//! "disk mode has no crash safety", which is wrong in a way worth pinning
//! down with a test rather than prose.
//!
//! A disk graph commits by publishing an **immutable generation**: the
//! staged snapshot is fsync'd file-by-file, renamed into `generations/`,
//! and only then does an atomically-persisted `CURRENT` pointer select it
//! (`storage::disk::generation::GenerationTxn::publish`). So the guarantee
//! disk mode *does* offer is precise:
//!
//! > A crash loses exactly the mutations made since the last `save()`, and
//! > nothing else. The graph reopens at the last published generation,
//! > complete and uncorrupted — never at a partially-written one.
//!
//! Nothing is ever acknowledged as durable in between, so no acknowledged
//! commit is lost; the boundary is the `save()` call itself. These tests
//! are the evidence for that sentence, which the disk-mode documentation
//! now states as the contract.
//!
//! The parent asserts the child died on signal 9 (`SIGKILL`) — uncatchable,
//! so not even Rust `Drop` or the process's own teardown runs. A test that
//! degraded into a graceful-shutdown test would prove nothing about a
//! crash, which is why this file is Unix-only rather than substituting a
//! catchable signal on Windows (the same doctrine as
//! `tests/test_durability.py::requires_sigkill`).

#![cfg(unix)]

use kglite::api::cypher::{execute_mutable, parse_cypher};
use kglite::api::io::{load_file, save_graph};
use kglite::api::storage::{new_dir_graph_in_mode, StorageMode};
use kglite::api::{DirGraph, GraphRead, Value};
use std::collections::HashMap;
use std::os::unix::process::ExitStatusExt;
use std::path::Path;
use std::process::Command;
use std::sync::Arc;
use std::time::{Duration, Instant};

const CHILD_GRAPH: &str = "KGLITE_DISK_CRASH_GRAPH";
const CHILD_READY: &str = "KGLITE_DISK_CRASH_READY";

fn run(graph: &mut Arc<DirGraph>, query: &str) {
    let parsed = parse_cypher(query).expect("parse");
    let dir = kglite::api::make_dir_graph_mut(graph);
    execute_mutable(dir, &parsed, HashMap::new(), Default::default()).expect("execute");
}

/// Every node title in the graph, sorted. Disk-backed reads borrow out of
/// the arena, so the read guard must outlive the borrows.
fn titles(graph: &DirGraph) -> Vec<String> {
    let _guard = graph.begin_read_pass();
    let mut found: Vec<String> = graph
        .graph
        .node_indices()
        .filter_map(|idx| graph.get_node(idx))
        .map(|node| match &node.title {
            Value::String(title) => title.to_string(),
            other => other.to_string(),
        })
        .collect();
    found.sort();
    found
}

/// Child half of the crash tests: publish one generation, then mutate
/// *without* saving and park until the parent kills it. Not a test on its
/// own — it returns immediately unless the parent's env vars are set.
#[test]
fn disk_crash_child() {
    let Some(root) = std::env::var_os(CHILD_GRAPH) else {
        return;
    };
    let ready = std::env::var_os(CHILD_READY).expect("ready path");
    let root = Path::new(&root);

    let mut graph =
        Arc::new(new_dir_graph_in_mode(StorageMode::Disk, Some(root)).expect("create disk graph"));
    run(&mut graph, "CREATE (:Person {id: 1, title: 'saved'})");
    save_graph(&mut graph, &root.to_string_lossy()).expect("publish generation");

    // Committed to the process's heap overlay only — no generation is
    // published for these, so they are exactly what the crash must lose.
    run(&mut graph, "CREATE (:Person {id: 2, title: 'unsaved'})");
    run(&mut graph, "CREATE (:Person {id: 3, title: 'unsaved-too'})");

    std::fs::write(ready, b"ready").expect("signal readiness");
    // The parent kills us here. Long enough that a hang is a test failure,
    // not a race.
    std::thread::sleep(Duration::from_secs(60));
}

/// Spawn the child, wait for it to park, and `SIGKILL` it.
fn crash_child(root: &Path, ready: &Path) {
    let mut child = Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "disk_crash_child", "--nocapture"])
        .env(CHILD_GRAPH, root)
        .env(CHILD_READY, ready)
        .spawn()
        .expect("spawn child");

    let started = Instant::now();
    while !ready.exists() && started.elapsed() < Duration::from_secs(30) {
        std::thread::sleep(Duration::from_millis(20));
    }
    assert!(ready.exists(), "child never reached its unsaved mutations");

    child.kill().expect("SIGKILL child");
    let status = child.wait().expect("reap child");
    assert_eq!(
        status.signal(),
        Some(9),
        "child must die on SIGKILL, or this is not a crash test"
    );
}

#[test]
fn sigkill_loses_exactly_the_mutations_since_the_last_save() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path().join("graph");
    crash_child(&root, &tmp.path().join("ready"));

    let recovered = load_file(&root.to_string_lossy()).expect("reopen after crash");
    assert_eq!(
        titles(&recovered),
        vec!["saved".to_string()],
        "the published generation survives intact; the unsaved mutations are gone"
    );
}

#[test]
fn sigkill_leaves_current_selecting_a_complete_generation() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path().join("graph");
    crash_child(&root, &tmp.path().join("ready"));

    // `CURRENT` must name a real, complete generation — never a `.stage-*`
    // directory and never a half-written one. `resolve_snapshot` rejects an
    // incomplete target, so a successful load *is* that assertion; check the
    // pointer's shape directly too, so a regression names the pointer rather
    // than surfacing as a confusing load error.
    let current = std::fs::read_to_string(root.join("CURRENT")).expect("CURRENT survives");
    let name = current
        .strip_suffix('\n')
        .expect("CURRENT ends in a newline");
    assert!(
        name.starts_with("gen_"),
        "CURRENT must select a published generation, got {name:?}"
    );
    let selected = root.join("generations").join(name);
    assert!(
        selected.join("metadata.json").is_file(),
        "the selected generation must carry its completion marker"
    );
    assert!(
        !name.contains(".stage-"),
        "CURRENT must never select a staging directory"
    );
}

#[test]
fn a_crashed_writer_releases_its_lease_to_the_next_process() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path().join("graph");
    crash_child(&root, &tmp.path().join("ready"));

    // The killed writer's `.kglite.lock` file is still on disk, but liveness
    // is OS advisory-lock based, so a fresh writer takes over and can publish
    // a new generation on top of the recovered one.
    let mut recovered = load_file(&root.to_string_lossy()).expect("reopen after crash");
    run(&mut recovered, "CREATE (:Person {id: 4, title: 'after'})");
    save_graph(&mut recovered, &root.to_string_lossy()).expect("publish after recovery");

    let reloaded = load_file(&root.to_string_lossy()).expect("reopen after republish");
    assert_eq!(
        titles(&reloaded),
        vec!["after".to_string(), "saved".to_string()],
        "recovery is writable: the next save publishes on top of the survivor"
    );
}
