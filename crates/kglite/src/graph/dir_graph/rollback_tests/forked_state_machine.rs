//! Randomised write sequences through a `Session`, checked against a graph
//! that never forks (issue #195).
//!
//! Each step is a create, a targeted delete, a delete-all, a MERGE, an edge
//! create, delete or weight write, or taking/dropping a snapshot, and each write goes through either
//! `Session::begin`/`commit` or the `Session::write` guard. Deletes leave slots
//! on the free lists in arbitrary orders, so forks, fold-backs and the
//! deep-copy fallback all occur. After every step:
//!
//! - the session's graph equals the reference, slot for slot;
//! - every type-index entry resolves to a live node of that type, and the
//!   index, `MATCH (n:T)` and the backend agree on the count;
//! - a held snapshot still reads exactly as when it was taken.
//!
//! With reloads on, a step may also save both graphs and serve the rest of
//! the run from the reloaded files, whose free lists serde rebuilt.
//!
//! The generator is a fixed-seed xorshift, so a failure names its seed and
//! step and replays exactly.

use super::*;
use crate::graph::session::{CommitOutcome, Session};
use crate::graph::storage::mode::{new_dir_graph_in_mode, StorageMode};

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

fn content(graph: &DirGraph) -> Fingerprint {
    let mut copy = graph.clone();
    let mut print = fingerprint(&mut copy);
    print.version = 0;
    print
}

fn count(graph: &DirGraph, query: &str) -> usize {
    let params = HashMap::new();
    let out =
        crate::graph::session::execute::execute_read(graph, query, &ExecuteOptions::eager(&params))
            .unwrap_or_else(|e| panic!("{query}: {e}"));
    match out.result.rows[0][0] {
        Value::Int64(n) => n as usize,
        ref other => panic!("{query}: expected a count, got {other:?}"),
    }
}

fn assert_indexes_resolve(graph: &DirGraph, context: &str) {
    let mut indexed = 0;
    for (name, members) in graph.type_indices.iter() {
        for idx in members.to_vec() {
            let view = graph
                .graph
                .node_view(idx)
                .unwrap_or_else(|| panic!("{context}: {name} index lists dead slot {idx:?}"));
            assert_eq!(
                view.node_type_str(&graph.interner),
                name.to_string(),
                "{context}: {name} index lists a node of another type"
            );
            indexed += 1;
        }
    }
    assert_eq!(
        indexed,
        graph.graph.node_count(),
        "{context}: index vs backend"
    );
    assert_eq!(
        count(graph, "MATCH (n:T) RETURN count(n)"),
        graph.graph.node_count(),
        "{context}: MATCH (n:T) vs backend"
    );
    assert_eq!(
        count(graph, "MATCH (n) WHERE n.k IS NULL RETURN count(n)"),
        0,
        "{context}: a node lost its property"
    );
}

fn new_graph(mode: StorageMode, dir: Option<&std::path::Path>) -> DirGraph {
    new_dir_graph_in_mode(mode, dir).expect("graph in mode")
}

/// `graph` saved to a `.kgl` and loaded back.
fn reloaded(graph: &DirGraph) -> DirGraph {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("g.kgl").to_string_lossy().into_owned();
    crate::graph::io::file::save_graph(&mut Arc::new(graph.clone()), &path).expect("save");
    let loaded = crate::graph::io::file::load_file(&path).expect("load");
    Arc::try_unwrap(loaded)
        .ok()
        .expect("a fresh load is unshared")
}

fn run_machine(mode: StorageMode, seed: u64, steps: usize) {
    run_machine_with(mode, seed, steps, false);
}

/// Runs the machine; returns how many reloads restored a graph with a hole.
fn run_machine_with(mode: StorageMode, seed: u64, steps: usize, reloads: bool) -> usize {
    let mut holed_reloads = 0;
    let dirs = (
        tempfile::tempdir().expect("tempdir"),
        tempfile::tempdir().expect("tempdir"),
    );
    let disk = matches!(mode, StorageMode::Disk);
    let mut session = Session::new(new_graph(mode, disk.then(|| dirs.0.path())));
    let mut reference = new_graph(mode, disk.then(|| dirs.1.path()));
    let mut rng = Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
    let mut live: Vec<String> = Vec::new();
    let mut next_key = 0usize;
    let mut held: Option<(Arc<DirGraph>, Fingerprint)> = None;

    for step in 0..steps {
        let context = format!("{mode:?} seed {seed} step {step}");
        let query = match rng.below(12) {
            0..=2 => {
                let keys: Vec<String> = (0..1 + rng.below(3))
                    .map(|_| {
                        next_key += 1;
                        format!("'k{next_key}'")
                    })
                    .collect();
                format!("UNWIND [{}] AS k CREATE (:T {{k: k}})", keys.join(", "))
            }
            3 | 4 if !live.is_empty() => {
                let k = live[rng.below(live.len())].clone();
                format!("MATCH (n:T {{k: '{k}'}}) DETACH DELETE n")
            }
            5 if rng.below(3) == 0 => "MATCH (n) DETACH DELETE n".to_string(),
            6 => {
                next_key += 1;
                let fresh = format!("k{next_key}");
                let k = if live.is_empty() || rng.below(2) == 0 {
                    fresh
                } else {
                    live[rng.below(live.len())].clone()
                };
                format!("MERGE (n:T {{k: '{k}'}}) SET n.v = {step}")
            }
            7 if live.len() >= 2 => {
                let a = live[rng.below(live.len())].clone();
                let b = live[rng.below(live.len())].clone();
                format!("MATCH (a:T {{k: '{a}'}}), (b:T {{k: '{b}'}}) CREATE (a)-[:R]->(b)")
            }
            10 if live.len() >= 2 => "MATCH (:T)-[r:R]->(:T) WITH r LIMIT 1 DELETE r".to_string(),
            11 if live.len() >= 2 => {
                format!("MATCH (:T)-[r:R]->(:T) SET r.w = {step}")
            }
            8 => {
                held = match held.take() {
                    Some(_) => None,
                    None => {
                        let snapshot = session.snapshot();
                        let print = content(&snapshot);
                        Some((snapshot, print))
                    }
                };
                continue;
            }
            9 if reloads => {
                held = None;
                session = Session::new(reloaded(&session.snapshot()));
                reference = reloaded(&reference);
                if reference.graph.node_count() < reference.graph.node_bound()
                    || reference.graph.edge_count() < reference.graph.edge_bound()
                {
                    holed_reloads += 1;
                    let mut writer = Arc::new(reference.clone());
                    let _reader = Arc::clone(&writer);
                    assert!(
                        crate::graph::handle::make_dir_graph_mut(&mut writer)
                            .graph
                            .is_forked(),
                        "{context}: a reloaded graph with holes must fork"
                    );
                }
                assert_eq!(
                    content(&session.snapshot()),
                    content(&reference),
                    "{context}: reload"
                );
                continue;
            }
            _ => continue,
        };

        let via_tx = rng.below(2) == 0;
        if std::env::var("SM_TRACE").is_ok() {
            eprintln!("{context}: tx={via_tx} held={} {query}", held.is_some());
        }
        if via_tx {
            let mut tx = session.begin();
            run(tx.working_mut().expect("read-write tx"), &query);
            let outcome = session.commit(tx, true);
            assert!(
                matches!(outcome, CommitOutcome::Committed { .. }),
                "{context}: {query}: {outcome:?}"
            );
        } else {
            run(&mut session.write(), &query);
        }
        run(&mut reference, &query);

        let snapshot = session.snapshot();
        assert_eq!(
            content(&snapshot),
            content(&reference),
            "{context}: {query}"
        );
        assert_indexes_resolve(&snapshot, &context);
        if let Some((reader, print)) = &held {
            assert_eq!(&content(reader), print, "{context}: held snapshot moved");
        }
        drop(snapshot);

        live = {
            let params = HashMap::new();
            crate::graph::session::execute::execute_read(
                &reference,
                "MATCH (n:T) RETURN n.k",
                &ExecuteOptions::eager(&params),
            )
            .expect("live keys")
            .result
            .rows
            .into_iter()
            .map(|row| match &row[0] {
                Value::String(k) => k.clone(),
                other => panic!("{context}: key {other:?}"),
            })
            .collect()
        };
    }
    holed_reloads
}

#[test]
fn random_writes_in_memory_match_a_graph_that_never_forks() {
    for seed in 0..120 {
        run_machine(StorageMode::Memory, seed, 80);
    }
}

/// The same machine, with saves and reloads between writes: a reloaded graph
/// with holes forks off its restored free lists.
#[test]
fn random_writes_across_reloads_match_a_graph_that_never_forks() {
    let holed: usize = (0..24)
        .map(|seed| run_machine_with(StorageMode::Memory, seed, 80, true))
        .sum();
    assert!(holed > 0, "no reload restored a hole: the test is vacuous");
}

#[test]
fn random_writes_in_mapped_mode_match_a_graph_that_never_forks() {
    for seed in 0..6 {
        run_machine(StorageMode::Mapped, seed, 60);
    }
}

#[test]
fn random_writes_in_disk_mode_match_a_graph_that_never_forks() {
    for seed in 0..3 {
        run_machine(StorageMode::Disk, seed, 40);
    }
}

/// Issue #195's shape with relationships: delete everything, create a batch
/// and connect it, round after round, with a reader held on every third round.
/// Every round's nodes and edges land on slots the previous round vacated, in
/// the order its deletes freed them, so a fork-back that allocated any other
/// slot would lose or duplicate rows. Counts must be exact at every step, on
/// the forked graph and on a graph that never forks.
#[test]
fn two_hundred_delete_all_and_create_rounds_with_edges_keep_exact_counts() {
    let session = Session::new(new_graph(StorageMode::Memory, None));
    let mut reference = new_graph(StorageMode::Memory, None);
    let mut held: Option<(Arc<DirGraph>, Fingerprint)> = None;
    for round in 0..200 {
        let queries = [
            format!(
                "UNWIND range(1, {n}) AS i CREATE (:T {{k: 'r{round}-' + toString(i)}})",
                n = 3 + round % 4
            ),
            "MATCH (a:T), (b:T) WHERE a.k < b.k CREATE (a)-[:R]->(b)".to_string(),
        ];
        for query in &queries {
            if round % 2 == 0 {
                let mut tx = session.begin();
                run(tx.working_mut().expect("read-write tx"), query);
                assert!(matches!(
                    session.commit(tx, true),
                    CommitOutcome::Committed { .. }
                ));
            } else {
                run(&mut session.write(), query);
            }
            run(&mut reference, query);
        }
        if round % 3 == 0 {
            let snapshot = session.snapshot();
            let print = content(&snapshot);
            held = Some((snapshot, print));
        }
        let snapshot = session.snapshot();
        let n = 3 + round % 4;
        assert_eq!(
            count(&snapshot, "MATCH (n:T) RETURN count(n)"),
            n,
            "round {round}"
        );
        assert_eq!(
            count(&snapshot, "MATCH (:T)-[r:R]->(:T) RETURN count(r)"),
            n * (n - 1) / 2,
            "round {round}: edges"
        );
        // Slot for slot, but not the column stores: only the session compacts
        // the rows its deletes leave behind.
        let (mine, theirs) = (content(&snapshot), content(&reference));
        assert_eq!(mine.nodes, theirs.nodes, "round {round}: nodes");
        assert_eq!(mine.edges, theirs.edges, "round {round}: edges");
        assert_eq!(mine.type_indices, theirs.type_indices, "round {round}");
        drop(snapshot);

        let query = "MATCH (n) DETACH DELETE n";
        run(&mut session.write(), query);
        run(&mut reference, query);
        let snapshot = session.snapshot();
        assert_eq!(
            count(&snapshot, "MATCH (n) RETURN count(n)"),
            0,
            "round {round}"
        );
        assert_eq!(
            snapshot.graph.edge_count(),
            0,
            "round {round}: edges after delete"
        );
        if let Some((reader, print)) = &held {
            assert_eq!(
                &content(reader),
                print,
                "round {round}: held snapshot moved"
            );
        }
    }
}
