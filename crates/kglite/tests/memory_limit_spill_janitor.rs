//! The `set_memory_limit` spill directory: where it lands, and who reclaims the
//! ones a killed process left behind.
//!
//! The sibling of `spill_dir_janitor`, which covers the `.kgl` load's
//! directories, and its own integration binary for the same two reasons: the
//! janitor sweeps once per process, and `KGLITE_TMPDIR` is process-global — so
//! both are only observable in a process whose first spill is this test's.
//!
//! Until this landed, `maybe_spill_columns` minted
//! `$TMPDIR/kglite_spill_<pid>_<nanos>` with `std::env::temp_dir()` directly and
//! the janitor swept only `kglite_portable_`, so these trees accumulated
//! unreclaimed after every kill — the same leak the load path had, in a second
//! site, with `KGLITE_TMPDIR` ignored on top.

#![cfg(unix)]

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, SystemTime};

use kglite::api::mutation::{add_nodes, ColumnData, ColumnType, DataFrame};
use kglite::api::storage::{new_dir_graph_in_mode, StorageMode};
use kglite::api::DirGraph;

/// The pid of a child that has already exited and been reaped — dead for as
/// long as the kernel has not wrapped its pid counter back onto it, which no
/// constant can promise.
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
    fs::File::open(&dir)
        .expect("open dir")
        .set_modified(SystemTime::now() - age)
        .expect("set mtime");
    dir
}

fn dirs_named(root: &Path, prefix: &str) -> Vec<PathBuf> {
    fs::read_dir(root)
        .expect("read root")
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with(prefix))
        })
        .collect()
}

/// A mapped-mode graph whose column data is well past the spill threshold.
///
/// `StorageMode::Mapped` *is* `memory_limit = Some(0)`, so the batch's closing
/// `maybe_spill_columns` has to materialise the store — this is the production
/// route into the code under test, not a poked field.
fn spilling_mapped_graph() -> Arc<DirGraph> {
    let mut graph = new_dir_graph_in_mode(StorageMode::Mapped, None).expect("create graph");
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

#[test]
fn a_memory_limit_spill_honors_kglite_tmpdir_and_sweeps_both_prefixes() {
    let root = tempfile::tempdir().expect("tmpdir");
    // Set before the first spill in this process: the sweep is `Once`-guarded
    // and reads the root at that moment.
    std::env::set_var("KGLITE_TMPDIR", root.path());

    let dead_pid = reaped_child_pid();
    let dead_old = mint_dir(
        root.path(),
        &format!("kglite_spill_{dead_pid}_abc"),
        Duration::from_secs(7200),
    );
    let dead_young = mint_dir(
        root.path(),
        &format!("kglite_spill_{dead_pid}_def"),
        Duration::from_secs(60),
    );
    let live = mint_dir(
        root.path(),
        &format!("kglite_spill_{}_beef", std::process::id()),
        Duration::from_secs(7200),
    );
    let junk = mint_dir(
        root.path(),
        "kglite_spill_notapid_x",
        Duration::from_secs(7200),
    );
    let unrelated = mint_dir(root.path(), "someone-elses-data", Duration::from_secs(7200));
    // One leftover from the *other* producer, to prove a single sweep reclaims
    // both rather than each prefix needing its own trigger.
    let dead_load_orphan = mint_dir(
        root.path(),
        &format!("kglite_portable_{dead_pid}_aa"),
        Duration::from_secs(7200),
    );

    let graph = spilling_mapped_graph();

    // The spill landed under `KGLITE_TMPDIR`, which is the half that was
    // broken: this path used to resolve `std::env::temp_dir()` itself.
    let ours = format!("kglite_spill_{}_", std::process::id());
    let spills: Vec<PathBuf> = dirs_named(root.path(), &ours)
        .into_iter()
        .filter(|p| p != &live)
        .collect();
    assert!(
        !spills.is_empty(),
        "the memory-limit spill went somewhere other than KGLITE_TMPDIR"
    );

    assert!(!dead_old.exists(), "old dead-pid orphan survived the sweep");
    assert!(
        !dead_load_orphan.exists(),
        "one sweep must reclaim both producers' orphans"
    );
    assert!(dead_young.exists(), "young dead-pid dir was swept");
    assert!(live.exists(), "live-pid dir was swept");
    assert!(junk.exists(), "unparseable name was swept");
    assert!(unrelated.exists(), "non-matching name was swept");

    // Drop-based cleanup still owns the tree this graph made: the janitor is
    // the backstop for a *killed* process, not a replacement for it.
    drop(graph);
    assert!(
        spills.iter().all(|p| !p.exists()),
        "the graph's own spill tree outlived it"
    );

    std::env::remove_var("KGLITE_TMPDIR");
}
