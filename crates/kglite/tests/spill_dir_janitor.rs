//! The portable-load spill directory: where it lands, and who reclaims the
//! ones a killed process left behind.
//!
//! Its own integration binary on purpose. The janitor sweeps once per process
//! and `KGLITE_TMPDIR` is process-global, so both are only observable in a
//! process whose first `.kgl` load is this test's.

#![cfg(unix)]

use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

use kglite::api::io::{load_file, save_graph};
use kglite::api::mutation::{add_nodes, ColumnData, ColumnType, DataFrame};
use kglite::api::storage::{new_dir_graph_in_mode, StorageMode};
use kglite::api::DirGraph;
use std::sync::Arc;

/// A graph whose single `Doc.body` column packs well past the 256 KB
/// `MMAP_THRESHOLD`, so loading it back is guaranteed to spill.
fn spilling_graph() -> Arc<DirGraph> {
    let mut graph = new_dir_graph_in_mode(StorageMode::Memory, None).expect("create graph");
    let rows = 4_000;
    let body = "x".repeat(128);
    let mut df = DataFrame::new(Vec::new());
    df.add_column(
        "id".to_string(),
        ColumnType::String,
        ColumnData::String((0..rows).map(|i| Some(format!("n{i}"))).collect()),
    )
    .expect("id column");
    df.add_column(
        "body".to_string(),
        ColumnType::String,
        ColumnData::String((0..rows).map(|_| Some(body.clone())).collect()),
    )
    .expect("body column");
    add_nodes(
        &mut graph,
        df,
        "Doc".to_string(),
        "id".to_string(),
        None,
        None,
    )
    .expect("add nodes");
    Arc::new(graph)
}

/// The pid of a child that has already exited and been reaped — dead for as
/// long as the kernel has not wrapped its pid counter back onto it.
fn reaped_child_pid() -> u32 {
    let mut child = std::process::Command::new("/bin/sh")
        .args(["-c", "exit 0"])
        .spawn()
        .expect("spawn child");
    let pid = child.id();
    child.wait().expect("reap child");
    pid
}

fn mint_dir(root: &Path, name: &str, age: Duration) -> PathBuf {
    let dir = root.join(name);
    fs::create_dir_all(dir.join("type_0")).expect("mint dir");
    let stamp = SystemTime::now() - age;
    fs::File::open(&dir)
        .expect("open dir")
        .set_modified(stamp)
        .expect("set mtime");
    dir
}

fn spill_dirs_of(root: &Path) -> Vec<PathBuf> {
    let mine = format!("kglite_portable_{}_", std::process::id());
    fs::read_dir(root)
        .expect("read root")
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with(&mine))
        })
        .collect()
}

#[test]
fn load_sweeps_dead_pid_spill_dirs_and_honors_kglite_tmpdir() {
    let root = tempfile::tempdir().expect("tmpdir");
    // Set before the first load in this process: the sweep is `Once`-guarded
    // and reads the root at that moment.
    std::env::set_var("KGLITE_TMPDIR", root.path());

    // A spawned-and-reaped child's pid is genuinely dead, which no constant
    // can guarantee.
    let dead_pid = reaped_child_pid();
    let dead_old = mint_dir(
        root.path(),
        &format!("kglite_portable_{dead_pid}_abc"),
        Duration::from_secs(7200),
    );
    let dead_young = mint_dir(
        root.path(),
        &format!("kglite_portable_{dead_pid}_def"),
        Duration::from_secs(60),
    );
    let live = mint_dir(
        root.path(),
        &format!("kglite_portable_{}_beef", std::process::id()),
        Duration::from_secs(7200),
    );
    let junk = mint_dir(
        root.path(),
        "kglite_portable_notapid_x",
        Duration::from_secs(7200),
    );
    let unrelated = mint_dir(root.path(), "someone-elses-data", Duration::from_secs(7200));

    let path = root.path().join("g.kgl");
    let mut graph = spilling_graph();
    save_graph(&mut graph, path.to_str().unwrap()).expect("save");
    drop(graph);

    let loaded = load_file(path.to_str().unwrap()).expect("load");

    let spills = spill_dirs_of(root.path());
    assert!(
        spills.iter().any(|p| p != &live),
        "the load spilled nowhere under KGLITE_TMPDIR: {spills:?}"
    );

    assert!(!dead_old.exists(), "old dead-pid orphan survived the sweep");
    assert!(dead_young.exists(), "young dead-pid dir was swept");
    assert!(live.exists(), "live-pid dir was swept");
    assert!(junk.exists(), "unparseable name was swept");
    assert!(unrelated.exists(), "non-matching name was swept");

    drop(loaded);
    std::env::remove_var("KGLITE_TMPDIR");
}
