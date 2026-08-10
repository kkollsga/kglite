//! Where does the held-reader fork actually spend its time?
//!
//! `Arc::make_mut` on a shared `Arc<DirGraph>` deep-clones the whole graph
//! (`handle.rs::make_dir_graph_mut`), and that clone is the ~27.6 ms cliff at
//! 1M nodes. The existing oracle for it — `BACKEND_CLONE_NODES`
//! (`storage/backend.rs`) — **counts nodes only**, which
//! `rollback_tests.rs::assert_statements_copy_zero_nodes` says out loud: *"a
//! `HashMap` clone bumps nothing"*. So nothing in the tree has ever measured
//! the non-backend half of a `DirGraph` clone, and every statement about which
//! field dominates has been inference from type shapes.
//!
//! This module measures it. It times **each of the ten fields
//! `rollback::swap_data_scale` parks** — the codebase's own closed list of
//! "data-scale" fields — cloned in isolation, plus the whole `DirGraph::clone`
//! for reconciliation. The residue (total minus the sum of the ten) is the
//! "O(schema) shell" that the rollback checkpoint already pays per statement.
//!
//! ## Why it is `#[ignore]`d
//!
//! It builds million-node fixtures and reports wall time. It is a
//! **measurement instrument**, not a gate: it asserts only the invariants that
//! would make its own numbers meaningless — that each fixture is the shape it
//! claims (nodes present, id index warm, the index family under test actually
//! populated). It does **not** check that its ten rows still match
//! `swap_data_scale`'s ten fields; that list is hand-maintained and diffed by
//! eye, per the comment on `apportion`. Nothing gates the drift, which is
//! tolerable only because forgetting to park a field costs speed rather than
//! correctness (`rollback.rs`'s module doc states the asymmetry). Run it
//! deliberately:
//!
//! ```text
//! cargo test -p kglite --release fork_apportionment -- --ignored --nocapture
//! ```
//!
//! **Release profile is mandatory** per `CLAUDE.md`'s performance protocol —
//! a debug-profile clone time is not a number, and this file will refuse to
//! print one.
//!
//! ## What the fixtures deliberately do not have
//!
//! - **No edges.** The fixture mirrors `tests/benchmarks/
//!   test_bench_fast_write_path.py::_base_graph`, which is the shape the
//!   27.6 ms figure was measured on, so the `graph` row is the *node* half of
//!   the backend term. Edges add to that same row, never to another.
//! - **No timeseries.** `timeseries_store` has no Cypher-reachable writer, so
//!   it is structurally zero on every shape an application benchmark can
//!   build. It stays in the table as a zero rather than being dropped, because
//!   an absent row reads as "not measured".
//! - **`secondary_label_index` is only non-empty on the `labelled` fixture**,
//!   which exists for exactly that reason.

use std::collections::HashMap;
use std::hint::black_box;
use std::time::Instant;

use super::DirGraph;
use crate::graph::schema::EmbeddingStore;
use crate::graph::session::execute::{execute_mut, ExecuteOptions};
use crate::graph::storage::GraphRead;

/// Nodes per fixture. One decade below the 1M the cliff was measured at makes
/// every row of the table smaller than the noise of the build; one decade
/// above is minutes of fixture time for a ratio that does not move.
const NODES: usize = 1_000_000;

/// Embedding width for the `embeddings` fixture. Narrower than a real model
/// (384–1536) on purpose: the row measures *shape* (a flat `Vec<f32>` plus two
/// per-node `HashMap`s, all cloned deeply, no `Arc` anywhere), and the vector
/// half scales exactly linearly in this constant, so a reader can rescale it.
const EMBED_DIM: usize = 64;

/// Clone repetitions per field. Enough to reject a one-off page-fault storm;
/// `min` is the statistic, per `CLAUDE.md`.
const REPS: usize = 3;

fn run(graph: &mut DirGraph, query: &str) {
    let params = HashMap::new();
    execute_mut(graph, query, &ExecuteOptions::eager(&params))
        .unwrap_or_else(|e| panic!("fixture query failed: {query}: {e}"));
}

/// Minimum wall time of `reps` clones of `value`, in microseconds.
///
/// `black_box` around the clone is load-bearing: without it a release build is
/// free to delete a clone whose result is only dropped, and every row would
/// read as zero — a table that certifies "nothing costs anything".
fn min_clone_us<T: Clone>(value: &T, reps: usize) -> f64 {
    let mut best = f64::MAX;
    for _ in 0..reps {
        let started = Instant::now();
        let copy = value.clone();
        let elapsed = started.elapsed().as_secs_f64() * 1e6;
        drop(black_box(copy));
        best = best.min(elapsed);
    }
    best
}

/// One row of the apportionment table.
struct FieldCost {
    name: &'static str,
    micros: f64,
    /// A size the reader can sanity-check the cost against (entries, nodes).
    extent: usize,
}

/// Time each of the ten `swap_data_scale` fields, plus the whole graph.
///
/// The order matches `swap_data_scale`'s own order so the two lists can be
/// diffed by eye when a field is added.
fn apportion(graph: &DirGraph) -> (Vec<FieldCost>, f64) {
    let fields = vec![
        FieldCost {
            name: "graph (backend)",
            micros: min_clone_us(&graph.graph, REPS),
            extent: graph.graph.node_count(),
        },
        FieldCost {
            name: "type_indices",
            micros: min_clone_us(&graph.type_indices, REPS),
            extent: graph.graph.node_count(),
        },
        FieldCost {
            name: "id_indices",
            micros: min_clone_us(&graph.id_indices, REPS),
            extent: graph.id_indices.overlay_len("Item").unwrap_or(0),
        },
        FieldCost {
            name: "property_indices",
            micros: min_clone_us(&graph.property_indices, REPS),
            extent: graph.property_indices.values().map(|m| m.len()).sum(),
        },
        FieldCost {
            name: "composite_indices",
            micros: min_clone_us(&graph.composite_indices, REPS),
            extent: graph.composite_indices.values().map(|m| m.len()).sum(),
        },
        FieldCost {
            name: "range_indices",
            micros: min_clone_us(&graph.range_indices, REPS),
            extent: graph.range_indices.values().map(|m| m.len()).sum(),
        },
        FieldCost {
            name: "secondary_label_index",
            micros: min_clone_us(&graph.secondary_label_index, REPS),
            extent: graph.secondary_label_index.values().map(|v| v.len()).sum(),
        },
        FieldCost {
            name: "embeddings",
            micros: min_clone_us(&graph.embeddings, REPS),
            extent: graph
                .embeddings
                .values()
                .map(|s| s.slot_to_node.len())
                .sum(),
        },
        FieldCost {
            name: "timeseries_store",
            micros: min_clone_us(&graph.timeseries_store, REPS),
            extent: graph.timeseries_store.len(),
        },
        FieldCost {
            name: "unique_indices",
            micros: min_clone_us(&graph.unique_indices, REPS),
            extent: graph.unique_indices.values().map(|m| m.len()).sum(),
        },
    ];
    let total = min_clone_us(graph, REPS);
    (fields, total)
}

fn report(fixture: &str, graph: &DirGraph) {
    let (fields, total) = apportion(graph);
    let sum: f64 = fields.iter().map(|f| f.micros).sum();
    println!(
        "\n### fixture `{fixture}` — nodes {}, columnar {}",
        graph.graph.node_count(),
        graph.is_columnar()
    );
    println!("| field | µs | % of total | extent |");
    println!("|---|---:|---:|---:|");
    for field in &fields {
        println!(
            "| `{}` | {:.0} | {:.1}% | {} |",
            field.name,
            field.micros,
            100.0 * field.micros / total,
            field.extent
        );
    }
    println!(
        "| **sum of the ten** | **{sum:.0}** | **{:.1}%** | |",
        100.0 * sum / total
    );
    println!("| **whole `DirGraph::clone`** | **{total:.0}** | 100% | |");
    println!(
        "| *residue (O(schema) shell)* | {:.0} | {:.1}% | |",
        total - sum,
        100.0 * (total - sum) / total
    );
}

/// The plain fixture: what `_base_graph` builds in the Python harness.
fn build_plain(nodes: usize) -> DirGraph {
    let mut graph = DirGraph::new();
    run(
        &mut graph,
        &format!(
            "UNWIND range(0, {}) AS i CREATE (:Item {{id: i, name: 'item-' + toString(i), \
             code: 'code-' + toString(i), qty: i % 977}})",
            nodes - 1
        ),
    );
    // Warm the id index, exactly as the Python harness does — an unwarmed
    // `id_indices` is empty and its row would read as free.
    run(&mut graph, "MATCH (n:Item {id: 0}) RETURN n.id");
    assert_eq!(
        graph.graph.node_count(),
        nodes,
        "fixture must have the nodes it claims"
    );
    assert!(
        graph.id_indices.overlay_len("Item").unwrap_or(0) > 0,
        "the id index must be warm, or its row is vacuously zero"
    );
    graph
}

#[test]
#[ignore = "measurement instrument: builds 1M-node fixtures. \
            cargo test -p kglite --release fork_apportionment -- --ignored --nocapture"]
fn apportion_the_fork_across_the_data_scale_fields() {
    if cfg!(debug_assertions) {
        panic!("run this in release; a debug-profile clone time is not a measurement (CLAUDE.md)");
    }

    let base = build_plain(NODES);
    report("plain", &base);

    // ⚠ Every fixture below is built **from scratch**, never as `base.clone()`.
    //
    // A clone of `base` is a *fork*: its backend is a copy-on-write overlay
    // over `base`'s, so what `report` then measures is a clone-of-a-fork, which
    // is not the clone a user's graph performs. It reads wrong in the
    // reassuring direction for some fixtures and the alarming direction for
    // others — the `saved` row reported **10.9 ms** of backend copy on
    // 2026-08-10 because `enable_columnar` had to flatten the overlay first and
    // the rebuilt backend could no longer fork, while a real saved 1M graph
    // measures `g.copy()` at **0.6 µs**. Independent builds cost one more
    // pass each and describe the product.

    {
        let mut saved = build_plain(NODES);
        saved.enable_columnar();
        assert!(
            saved.is_columnar(),
            "the saved fixture must own master column stores"
        );
        report("saved (enable_columnar, as save() does)", &saved);
    }

    {
        let mut indexed = build_plain(NODES);
        indexed.create_index("Item", "code");
        indexed.create_index("Item", "qty");
        indexed.create_range_index("Item", "qty");
        indexed.create_composite_index("Item", &["code", "qty"]);
        assert!(
            !indexed.property_indices.is_empty()
                && !indexed.range_indices.is_empty()
                && !indexed.composite_indices.is_empty(),
            "all three user-index families must be live or the indexed rows are vacuous"
        );
        report("indexed (2 property + 1 range + 1 composite)", &indexed);
    }

    {
        let mut labelled = build_plain(NODES);
        run(&mut labelled, "MATCH (n:Item) SET n:Featured");
        assert!(
            !labelled.secondary_label_index.is_empty(),
            "the labelled fixture must populate secondary_label_index"
        );
        report("labelled (one secondary label on every node)", &labelled);
    }

    {
        let mut embedded = build_plain(NODES);
        let mut store = EmbeddingStore::new(EMBED_DIM);
        let vector: Vec<f32> = (0..EMBED_DIM).map(|i| i as f32 * 0.01).collect();
        for idx in 0..NODES {
            store.set_embedding(idx, &vector);
        }
        assert_eq!(
            store.slot_to_node.len(),
            NODES,
            "every node must carry an embedding"
        );
        embedded
            .embeddings
            .insert(("Item".to_string(), "name".to_string()), store);
        report(
            &format!("embeddings (dim {EMBED_DIM}, no HNSW index)"),
            &embedded,
        );
    }
}
