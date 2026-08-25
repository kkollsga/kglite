//! Incremental catch-up for text indexes: what the write path notices, what a
//! refresh does about it, and the one invariant everything else rests on —
//! **refresh must be indistinguishable from a rebuild**.
//!
//! Every equivalence assertion below compares against a graph-driven rebuild
//! rather than against a hand-written expected score, because BM25 scores shift
//! as the corpus does: only "the same corpus scores the same" is a stable
//! claim.

use super::*;
use crate::datatypes::Value;
use crate::graph::index_freshness::write_hooks;
use crate::graph::schema::NodeData;
use crate::graph::session::execute::{execute_mut, ExecuteOptions};
use crate::graph::storage::GraphWrite;
use std::collections::{HashMap, HashSet};

// ── fixtures ─────────────────────────────────────────────────────────

/// Add one `Doc` node the way both production creation funnels do: storage
/// insert, type bucket, freshness notification.
fn push_doc(graph: &mut DirGraph, id: i64, body: &str) -> NodeIndex {
    let mut props = HashMap::new();
    props.insert("body".to_string(), Value::String(body.to_string()));
    props.insert("tag".to_string(), Value::String("untouched".to_string()));
    let data = NodeData::new(
        Value::Int64(id),
        Value::String(format!("doc-{id}")),
        "Doc".to_string(),
        props,
        &mut graph.interner,
    );
    let idx = GraphWrite::add_node(&mut graph.graph, data);
    graph
        .type_indices
        .entry_or_default("Doc".to_string())
        .push(idx);
    write_hooks::note_node_created(graph, idx, "Doc");
    idx
}

fn indexed_corpus() -> (DirGraph, Vec<NodeIndex>) {
    let mut graph = DirGraph::new();
    let nodes = vec![
        push_doc(&mut graph, 1, "the quick brown fox"),
        push_doc(&mut graph, 2, "a quick brown marmoset appears"),
        push_doc(&mut graph, 3, "slow green turtles"),
    ];
    graph.build_id_index("Doc");
    build_text_index(&mut graph, "Doc", "body", None).expect("build");
    (graph, nodes)
}

fn store(graph: &DirGraph) -> &TextIndexStore {
    graph
        .text_indexes
        .get(&index_key("Doc", "body"))
        .expect("index built")
}

fn run(graph: &mut DirGraph, query: &str) {
    let params = HashMap::new();
    let opts = ExecuteOptions::eager(&params);
    execute_mut(graph, query, &opts).unwrap_or_else(|e| panic!("query failed: {query}: {e}"));
}

/// Every live `Doc`'s score for `query`, keyed by the node's own id so the
/// answer survives a slot renumbering. `None` = the row is unindexed.
fn score_map(graph: &DirGraph, query: &str) -> Vec<(i64, Option<f64>)> {
    let store = store(graph);
    let view = store.read();
    let prepared = view.prepare_query(query);
    let mut out: Vec<(i64, Option<f64>)> = graph
        .type_indices
        .get("Doc")
        .map(|members| members.to_vec())
        .unwrap_or_default()
        .into_iter()
        .filter_map(|node| {
            let id = match graph.graph.get_node_id(node)? {
                Value::Int64(id) => id,
                Value::UniqueId(id) => i64::from(id),
                other => panic!("unexpected id shape {other:?}"),
            };
            Some((id, view.score(node, &prepared)))
        })
        .collect();
    out.sort_by_key(|(id, _)| *id);
    out
}

/// The scores a wholesale rebuild of the same graph would produce. The oracle
/// for every catch-up assertion: `build` and `add_doc` are deliberately
/// separate code paths in `TextIndex`, so this is a real comparison.
fn rebuild_scores(graph: &DirGraph, query: &str) -> Vec<(i64, Option<f64>)> {
    let mut rebuilt = graph.clone();
    build_text_index(&mut rebuilt, "Doc", "body", None).expect("rebuild");
    score_map(&rebuilt, query)
}

fn assert_matches_rebuild(graph: &DirGraph, query: &str) {
    assert_eq!(
        score_map(graph, query),
        rebuild_scores(graph, query),
        "a refreshed index must be indistinguishable from a rebuilt one ({query})"
    );
    assert!(store(graph).validate().is_ok(), "views must still agree");
}

// ── creations after the build ────────────────────────────────────────

#[test]
fn a_post_build_creation_refreshes_into_scores_equal_to_a_rebuild() {
    let (mut graph, _) = indexed_corpus();
    assert!(!store(&graph).is_stale(&graph), "a fresh build is current");

    push_doc(&mut graph, 4, "another quick marmoset");

    assert!(store(&graph).is_stale(&graph));
    assert_eq!(store(&graph).delta_size(&graph), 1);
    assert!(store(&graph).can_auto_refresh(&graph));

    assert_eq!(store(&graph).refresh(&graph, "Doc"), 1, "one slot re-read");

    assert!(!store(&graph).is_stale(&graph));
    assert_eq!(store(&graph).documents(), 4);
    assert_matches_rebuild(&graph, "quick marmoset");
    assert_matches_rebuild(&graph, "turtles");
}

#[test]
fn a_refresh_with_nothing_outstanding_does_no_work() {
    let (graph, _) = indexed_corpus();
    assert_eq!(store(&graph).refresh(&graph, "Doc"), 0);
}

/// Creations of a type this index does not cover must not make it look stale —
/// otherwise loading an unrelated table would push every index over its limit.
#[test]
fn creating_another_node_type_leaves_the_index_current() {
    let (mut graph, _) = indexed_corpus();

    run(&mut graph, "CREATE (:Company {id: 1, name: 'Acme'})");
    run(&mut graph, "CREATE (:Company {id: 2, name: 'Globex'})");

    assert!(
        !store(&graph).is_stale(&graph),
        "no Doc changed, so the Doc index is not behind"
    );
    assert_eq!(store(&graph).delta_size(&graph), 0);
}

// ── the recycled-slot hole ───────────────────────────────────────────

/// The watermark alone is blind here: the new node's slot is *below* it, so the
/// gap arithmetic reads it as already indexed. Without the creation-site slot
/// comparison the new document is never indexed and the old one never removed.
#[test]
fn a_creation_into_a_recycled_slot_is_caught_and_refreshed() {
    let (mut graph, nodes) = indexed_corpus();
    let doomed = nodes[1];
    let watermark_before = store(&graph).delta_size(&graph);
    assert_eq!(watermark_before, 0);

    crate::graph::mutation::maintain::detach_delete_nodes(&mut graph, &HashSet::from([doomed]));
    let reused = push_doc(&mut graph, 99, "entirely different content");
    assert_eq!(
        reused, doomed,
        "the fixture only proves anything if petgraph recycled the index"
    );

    assert!(
        store(&graph).is_stale(&graph),
        "a below-watermark creation must reach the dirty set"
    );
    assert_eq!(store(&graph).delta_size(&graph), 1);

    store(&graph).refresh(&graph, "Doc");

    let view = store(&graph).read();
    let marmoset = view.prepare_query("marmoset");
    assert_eq!(
        view.score(reused, &marmoset),
        Some(0.0),
        "the recycled slot must not still score the deleted document's terms"
    );
    let different = view.prepare_query("entirely different");
    assert!(
        view.score(reused, &different).expect("indexed") > 0.0,
        "and it must carry its own"
    );
    drop(view);
    assert_matches_rebuild(&graph, "marmoset");
    assert_matches_rebuild(&graph, "entirely different content");
}

// ── property writes ──────────────────────────────────────────────────

/// The discrimination that keeps an ordinary `SET` off the dirty set. Without
/// it every write to an indexed type would queue a re-read, which is the
/// per-write maintenance tax this design exists to avoid.
#[test]
fn a_set_of_the_indexed_property_dirties_and_another_property_does_not() {
    let (mut graph, _) = indexed_corpus();

    run(
        &mut graph,
        "MATCH (d:Doc) WHERE d.id = 1 SET d.tag = 'touched'",
    );
    assert!(
        !store(&graph).is_stale(&graph),
        "'tag' is not the indexed property"
    );

    run(
        &mut graph,
        "MATCH (d:Doc) WHERE d.id = 1 SET d.body = 'rewritten marmoset prose'",
    );
    assert!(store(&graph).is_stale(&graph));
    assert_eq!(store(&graph).delta_size(&graph), 1);

    store(&graph).refresh(&graph, "Doc");
    assert_matches_rebuild(&graph, "rewritten marmoset");
    assert_matches_rebuild(&graph, "quick brown");
}

#[test]
fn removing_the_indexed_property_drops_the_document_on_refresh() {
    let (mut graph, nodes) = indexed_corpus();

    run(&mut graph, "MATCH (d:Doc) WHERE d.id = 1 REMOVE d.body");
    assert!(store(&graph).is_stale(&graph));

    store(&graph).refresh(&graph, "Doc");

    assert!(
        !store(&graph).contains_node(nodes[0]),
        "a node with no string left is not a document"
    );
    assert_eq!(store(&graph).documents(), 2);
    assert_matches_rebuild(&graph, "quick brown fox");
}

/// The bulk paths (`add_nodes` upserts, `add_properties`) write a whole
/// property map and do not decompose it, so they mark field-blind. That is the
/// over-approximation contract, and this is the half of it that could be wrong:
/// an unnamed field must dirty the node, not be dismissed as "not the indexed
/// one".
#[test]
fn a_field_blind_write_marks_the_node_whatever_the_index_holds() {
    let (mut graph, nodes) = indexed_corpus();

    write_hooks::note_property_written(&graph, nodes[0], "Doc", None);
    assert!(store(&graph).is_stale(&graph));
    assert_eq!(store(&graph).delta_size(&graph), 1);

    // …and it is still scoped to the type. A field-blind write to some other
    // type is not this index's business.
    build_text_index(&mut graph, "Doc", "body", None).expect("rebuild clears the mark");
    write_hooks::note_property_written(&graph, nodes[0], "Company", None);
    assert!(!store(&graph).is_stale(&graph));
}

// ── threshold policy ─────────────────────────────────────────────────

/// The threshold decides whether a *caller* folds the delta in; `refresh`
/// itself has no opinion and always refreshes when asked.
#[test]
fn an_over_threshold_delta_is_reported_but_never_silently_refreshed() {
    let mut graph = DirGraph::new();
    push_doc(&mut graph, 1, "seed document");
    graph.build_id_index("Doc");
    build_text_index(&mut graph, "Doc", "body", Some(2)).expect("build");
    assert_eq!(store(&graph).auto_refresh_limit(), 2);

    for id in 2..=4 {
        push_doc(&mut graph, id, "later document");
    }

    assert_eq!(store(&graph).delta_size(&graph), 3);
    assert!(store(&graph).is_stale(&graph));
    assert!(
        !store(&graph).can_auto_refresh(&graph),
        "3 > the limit of 2, so a query must serve stale rather than pause"
    );
    assert_eq!(
        store(&graph).documents(),
        1,
        "nothing folded the delta in behind the caller's back"
    );

    assert_eq!(store(&graph).refresh(&graph, "Doc"), 3);
    assert_eq!(store(&graph).documents(), 4);
    assert!(!store(&graph).is_stale(&graph));
}

#[test]
fn a_rebuild_keeps_the_limit_its_author_set_unless_a_new_one_is_given() {
    let (mut graph, _) = indexed_corpus();
    assert_eq!(
        store(&graph).auto_refresh_limit(),
        crate::graph::index_freshness::DEFAULT_AUTO_REFRESH_LIMIT
    );

    build_text_index(&mut graph, "Doc", "body", Some(7)).expect("rebuild");
    assert_eq!(store(&graph).auto_refresh_limit(), 7);

    build_text_index(&mut graph, "Doc", "body", None).expect("rebuild");
    assert_eq!(
        store(&graph).auto_refresh_limit(),
        7,
        "omitting the limit must not quietly restore the default"
    );
}

// ── read-only graphs ─────────────────────────────────────────────────

#[test]
fn a_read_only_graph_reports_staleness_and_refuses_to_catch_up() {
    let (mut graph, _) = indexed_corpus();
    push_doc(&mut graph, 4, "a later document");
    graph.read_only = true;

    assert!(store(&graph).is_stale(&graph));
    assert_eq!(
        store(&graph).refresh(&graph, "Doc"),
        0,
        "catching up would be the one write a read-only handle performed"
    );
    assert_eq!(store(&graph).documents(), 3);
}

// ── rollback ─────────────────────────────────────────────────────────

/// A delete prunes the document immediately, so a *reversed* delete restores
/// the node with no document. The journal entry marks the slot instead of
/// carrying a copy of the text, and the refresh puts it back.
#[test]
fn a_rolled_back_delete_leaves_the_node_refreshable() {
    let (mut graph, nodes) = indexed_corpus();
    let params = HashMap::new();
    let opts = ExecuteOptions::eager(&params);

    // Deletes `id = 2`, then fails on the immutable-id `SET` of the other
    // binding — a statement that has already written when it is rejected.
    let failed = execute_mut(
        &mut graph,
        "MATCH (a:Doc), (b:Doc) WHERE a.id = 1 AND b.id = 2 DELETE b SET a.id = 99",
        &opts,
    );
    assert!(failed.is_err(), "the statement must fail after its delete");

    assert_eq!(
        graph.type_indices.get("Doc").map(|m| m.len()),
        Some(3),
        "the rollback restored the node"
    );
    assert!(
        !store(&graph).contains_node(nodes[1]),
        "its document was pruned at delete time and is not journal-restorable"
    );
    assert!(
        store(&graph).is_stale(&graph),
        "so the rollback must have marked the slot for re-reading"
    );

    store(&graph).refresh(&graph, "Doc");

    assert!(store(&graph).contains_node(nodes[1]));
    assert_matches_rebuild(&graph, "marmoset");
}

// ── the global gate ──────────────────────────────────────────────────

/// The zero-cost promise for graphs without an index, asserted structurally:
/// the write path may evaluate the gate, and must do nothing behind it.
#[test]
fn an_unindexed_graph_does_no_work_behind_the_write_path_gate() {
    let mut graph = DirGraph::new();
    push_doc(&mut graph, 1, "the quick brown fox");
    graph.build_id_index("Doc");
    let before = write_hooks::work_past_gate();

    run(&mut graph, "CREATE (:Doc {id: 2, body: 'created'})");
    run(
        &mut graph,
        "MATCH (d:Doc) WHERE d.id = 1 SET d.body = 'set'",
    );
    run(&mut graph, "MATCH (d:Doc) WHERE d.id = 1 REMOVE d.tag");
    run(&mut graph, "MATCH (d:Doc) WHERE d.id = 2 DELETE d");

    assert_eq!(
        write_hooks::work_past_gate(),
        before,
        "no text index exists, so every hook must return at its first branch"
    );

    // …and the same statements on an indexed graph do get past it, so the
    // counter is measuring something.
    build_text_index(&mut graph, "Doc", "body", None).expect("build");
    run(&mut graph, "CREATE (:Doc {id: 3, body: 'created'})");
    assert!(write_hooks::work_past_gate() > before);
}

// ── the equivalence oracle, randomized ───────────────────────────────

/// Random create/set/delete interleavings, each followed by a catch-up, must
/// leave an index no read can tell from a rebuilt one — and no deleted node may
/// ever score.
#[test]
fn randomized_crud_with_refresh_stays_identical_to_a_rebuild() {
    const WORDS: [&str; 8] = [
        "quick", "brown", "fox", "marmoset", "slow", "green", "turtle", "prose",
    ];
    let mut seed: u64 = 0x9E3779B97F4A7C15;
    let mut next = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };

    let mut graph = DirGraph::new();
    for id in 0..12 {
        push_doc(&mut graph, id, WORDS[(id as usize) % WORDS.len()]);
    }
    graph.build_id_index("Doc");
    build_text_index(&mut graph, "Doc", "body", None).expect("build");

    let mut next_id: i64 = 12;
    for step in 0..120 {
        let members: Vec<NodeIndex> = graph
            .type_indices
            .get("Doc")
            .map(|m| m.to_vec())
            .unwrap_or_default();
        match next() % 3 {
            0 => {
                let text = format!(
                    "{} {}",
                    WORDS[(next() % 8) as usize],
                    WORDS[(next() % 8) as usize]
                );
                push_doc(&mut graph, next_id, &text);
                next_id += 1;
            }
            1 if !members.is_empty() => {
                let victim = members[(next() as usize) % members.len()];
                let text = format!("rewritten {}", WORDS[(next() % 8) as usize]);
                assert!(graph.set_node_property(victim, "body", Value::String(text)));
                write_hooks::note_property_written(&graph, victim, "Doc", Some("body"));
            }
            _ if !members.is_empty() => {
                let victim = members[(next() as usize) % members.len()];
                crate::graph::mutation::maintain::detach_delete_nodes(
                    &mut graph,
                    &HashSet::from([victim]),
                );
                let view = store(&graph).read();
                assert!(
                    !view.contains_node(victim),
                    "step {step}: a deleted node kept its document"
                );
            }
            _ => {}
        }
        store(&graph).refresh(&graph, "Doc");
        assert_matches_rebuild(&graph, "quick marmoset prose");
    }
}
