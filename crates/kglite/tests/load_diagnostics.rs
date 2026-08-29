//! What loading a `.kgl` touches outside the file itself, and what it does
//! when several loads run at once.
//!
//! Its own integration binary, like `spill_dir_janitor`: the spill root is
//! process-global and these tests measure what appears under it, so they must
//! not share a process with tests that load `.kgl` files for other reasons.
//!
//! Provoked by a 0.16.14 downstream report — an 11,728-byte `.kgl` that failed
//! to load twice in one day under a heavily concurrent `make gate`, with two
//! different bare OS errors (EEXIST, EINVAL) and no way to tell what syscall
//! produced either. A file that small carries no column section anywhere near
//! the 256 KB spill threshold, so the loader has no business writing anything.

#![cfg(unix)]

use std::path::PathBuf;
use std::sync::Arc;

use kglite::api::io::{load_file, save_graph};
use kglite::api::mutation::{add_nodes, ColumnData, ColumnType, DataFrame};
use kglite::api::storage::{new_dir_graph_in_mode, StorageMode};
use kglite::api::{DirGraph, GraphRead};

/// `rows` nodes with a `body` string of `body_len` bytes each — the knob is
/// whether the packed `Doc.body` column clears the 256 KB `MMAP_THRESHOLD`
/// that decides whether loading it spills.
fn doc_graph(rows: usize, body_len: usize) -> Arc<DirGraph> {
    let mut graph = new_dir_graph_in_mode(StorageMode::Memory, None).expect("create graph");
    let body = "x".repeat(body_len);
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

fn save_to(graph: Arc<DirGraph>, path: &std::path::Path) {
    let mut graph = graph;
    save_graph(&mut graph, path.to_str().expect("utf-8 path")).expect("save");
}

/// Every spill directory this process owns, right now.
fn our_spill_dirs() -> Vec<PathBuf> {
    let mine = format!("kglite_portable_{}_", std::process::id());
    let Ok(entries) = std::fs::read_dir(std::env::temp_dir()) else {
        return Vec::new();
    };
    entries
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with(&mine))
        })
        .collect()
}

/// Spilling is what earns a temp directory — not loading.
///
/// Both halves in one test on purpose: the measurement is a delta over a
/// process-global root, so a second test spilling in parallel would read as
/// this one's directory.
#[test]
fn only_a_load_that_spills_touches_the_spill_root() {
    let root = tempfile::tempdir().expect("tmpdir");

    // A graph whose every column packs well under the threshold. This is the
    // downstream's file: small, columnar, and with nothing to map.
    let small = root.path().join("small.kgl");
    save_to(doc_graph(20, 16), &small);
    assert!(
        std::fs::metadata(&small).expect("small.kgl").len() < 256 * 1024,
        "the small fixture must be smaller than one spill threshold"
    );

    let before = our_spill_dirs();
    let loaded = load_file(small.to_str().unwrap()).expect("load small");
    assert_eq!(loaded.graph.node_count(), 20);
    let created: Vec<PathBuf> = our_spill_dirs()
        .into_iter()
        .filter(|p| !before.contains(p))
        .collect();
    assert!(
        created.is_empty(),
        "a load with nothing to spill still wrote to the spill root: {created:?}"
    );
    drop(loaded);

    // The other side of the same contract: a load that does spill still gets
    // its directory, and still cleans it up on drop.
    let big = root.path().join("big.kgl");
    save_to(doc_graph(4_000, 128), &big);

    let before = our_spill_dirs();
    let loaded = load_file(big.to_str().unwrap()).expect("load big");
    assert_eq!(loaded.graph.node_count(), 4_000);
    let created: Vec<PathBuf> = our_spill_dirs()
        .into_iter()
        .filter(|p| !before.contains(p))
        .collect();
    assert_eq!(
        created.len(),
        1,
        "a spilling load must mint exactly one directory: {created:?}"
    );
    assert!(
        std::fs::read_dir(&created[0])
            .expect("read spill dir")
            .next()
            .is_some(),
        "the spill directory it minted is empty"
    );

    drop(loaded);
    assert!(
        !created[0].exists(),
        "the spill directory outlived the graph that owned it"
    );
}

/// Many threads loading one file at once, which is how the downstream's suite
/// reads its fixture. Each load owns its own spill naming and its own graph;
/// nothing here may collide with, or reclaim, another load's state.
#[test]
fn concurrent_loads_of_one_file_all_succeed() {
    let root = tempfile::tempdir().expect("tmpdir");
    let path = root.path().join("fixture.kgl");
    save_to(doc_graph(50, 24), &path);
    let path = path.to_str().expect("utf-8 path").to_string();

    std::thread::scope(|scope| {
        for _ in 0..16 {
            scope.spawn(|| {
                for _ in 0..25 {
                    let graph = load_file(&path).expect("concurrent load");
                    assert_eq!(graph.graph.node_count(), 50);
                }
            });
        }
    });
}
