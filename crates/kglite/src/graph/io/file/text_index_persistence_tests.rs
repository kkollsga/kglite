//! What a `.kgl` round-trip owes a text index.
//!
//! The oracle throughout is **the score**, not the structure: term ids are
//! internal and the round-trip deliberately re-assigns them, so "the same
//! index came back" can only mean "every document answers every query with the
//! same number". Where staleness is involved the oracle is a *rebuild*, for
//! the reason the freshness tests give — BM25 scores move as the corpus does,
//! so only "the same corpus scores the same" is a stable claim.

use super::super::{
    load_kgl_bytes, prepare_kgl_write, section_digest, write_kgl_to, zstd_compress,
};
use super::*;
use crate::datatypes::{DataFrame, Value};
use crate::graph::session::execute::{execute_mut, ExecuteOptions};
use crate::graph::storage::GraphRead;
use crate::graph::text_indexes::{build_text_index, index_key, TextIndexStore};
use std::collections::HashMap;
use std::sync::Arc;

/// Run a mutating statement through the production funnel, so every write hook
/// this lane depends on fires exactly as it does for a user.
fn run(graph: &mut DirGraph, statement: &str) {
    let params = HashMap::new();
    let opts = ExecuteOptions::eager(&params);
    execute_mut(graph, statement, &opts)
        .unwrap_or_else(|e| panic!("statement failed: {statement}: {e}"));
}

const QUERIES: [&str; 4] = ["quick brown", "turtles", "marmoset appears", "nothing here"];

/// Bodies chosen so the corpus has: shared terms (IDF that moves), terms
/// unique to one document (which the free list reclaims when it is deleted),
/// and an empty document (indexed, in no posting list, counted in `N`).
fn bodies() -> Vec<(i64, &'static str)> {
    vec![
        (1, "the quick brown fox"),
        (2, "a quick brown marmoset appears"),
        (3, "slow green turtles"),
        (4, "quick turtles are a contradiction"),
        (5, ""),
        (6, "solitary hapax legomenon"),
    ]
}

fn corpus_graph() -> DirGraph {
    let rows: Vec<Vec<Value>> = bodies()
        .into_iter()
        .map(|(id, body)| {
            vec![
                Value::Int64(id),
                Value::String(format!("doc-{id}")),
                Value::String(body.to_string()),
            ]
        })
        .collect();
    let df = DataFrame::from_cypher_rows(
        vec!["id".to_string(), "title".to_string(), "body".to_string()],
        rows,
    )
    .unwrap();
    let mut graph = DirGraph::new();
    crate::graph::mutation::maintain::add_nodes(
        &mut graph,
        df,
        "Doc".to_string(),
        "id".to_string(),
        Some("title".to_string()),
        None,
    )
    .unwrap();
    graph.build_id_index("Doc");
    graph
}

fn store(graph: &DirGraph) -> &TextIndexStore {
    graph
        .text_indexes
        .get(&index_key("Doc", "body"))
        .expect("index present")
}

/// Every live `Doc`'s score for `query`, keyed by the node's own id so the
/// answer survives any slot renumbering. `None` = the row carries no document.
fn score_map(graph: &DirGraph, query: &str) -> Vec<(i64, Option<f64>)> {
    let view = store(graph).read();
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

fn all_scores(graph: &DirGraph) -> Vec<Vec<(i64, Option<f64>)>> {
    QUERIES.iter().map(|q| score_map(graph, q)).collect()
}

/// Scores a wholesale rebuild of the same graph would produce — the oracle for
/// every catch-up assertion.
fn rebuild_scores(graph: &DirGraph) -> Vec<Vec<(i64, Option<f64>)>> {
    let mut rebuilt = graph.clone();
    build_text_index(&mut rebuilt, "Doc", "body", None).expect("rebuild");
    all_scores(&rebuilt)
}

fn round_trip(graph: DirGraph) -> Arc<DirGraph> {
    let mut arc = Arc::new(graph);
    prepare_kgl_write(&mut arc);
    let mut buf: Vec<u8> = Vec::new();
    write_kgl_to(&arc, &mut buf).unwrap();
    load_kgl_bytes(&buf).unwrap()
}

fn delete(graph: &mut DirGraph, id: i64) {
    run(graph, &format!("MATCH (d:Doc) WHERE d.id = {id} DELETE d"));
}

/// An index with holes on both sides — freed term ids in the dictionary and
/// freed node slots in the graph — comes back scoring identically.
///
/// The holes are the point. Deleting a document releases the term ids nothing
/// else uses, so the live index carries a free list and gaps in its `names`
/// vector; the round-trip writes the logical corpus and rebuilds compactly, so
/// the index that comes back is a *different* structure that has to answer
/// every query with the same number.
#[test]
fn a_holey_index_round_trips_with_identical_scores() {
    let mut graph = corpus_graph();
    build_text_index(&mut graph, "Doc", "body", None).expect("build");
    // Deleted after the build, so their documents are pruned and their terms
    // retired — the free list this test exists for.
    delete(&mut graph, 2);
    delete(&mut graph, 6);
    let before = all_scores(&graph);
    let documents = store(&graph).documents();

    let loaded = round_trip(graph);

    assert!(
        store(&loaded).validate().is_ok(),
        "a reloaded index must satisfy its own invariants: {:?}",
        store(&loaded).validate()
    );
    assert_eq!(store(&loaded).documents(), documents);
    assert_eq!(
        all_scores(&loaded),
        before,
        "a reloaded index must score exactly as the saved one did"
    );
    assert!(
        !store(&loaded).is_stale(&loaded),
        "nothing moved between the save and the load"
    );
}

/// The empty document is the one a postings-shaped payload loses: it appears
/// in no posting list, and dropping it would silently change BM25's `N` — and
/// therefore every other document's score, which is exactly what the score
/// assertion above would catch and this one names.
#[test]
fn an_empty_document_survives_the_round_trip() {
    let mut graph = corpus_graph();
    build_text_index(&mut graph, "Doc", "body", None).expect("build");
    let empty = graph.lookup_by_id("Doc", &Value::Int64(5)).unwrap();
    assert!(store(&graph).contains_node(empty), "'' is a document");
    let documents = store(&graph).documents();

    let loaded = round_trip(graph);

    assert_eq!(store(&loaded).documents(), documents);
    assert!(
        store(&loaded).contains_node(empty),
        "an empty document is indexed, not missing"
    );
}

/// The build-time facts a refresh cannot restate have to ride along: the
/// skipped count and the alias-resolved field.
#[test]
fn the_build_time_facts_ride_with_the_index() {
    let mut graph = corpus_graph();
    // A node of the type with no body at all — the skipped count this pins.
    run(&mut graph, "CREATE (:Doc {id: 7, title: 'doc-7'})");
    // Also indexed by the *title* alias, so one store's resolved field differs
    // from the key it is filed under.
    build_text_index(&mut graph, "Doc", "title", None).expect("build");
    build_text_index(&mut graph, "Doc", "body", None).expect("build");
    assert_eq!(store(&graph).skipped(), 1);

    let loaded = round_trip(graph);

    assert_eq!(store(&loaded).skipped(), 1, "a build-time count, carried");
    assert_eq!(store(&loaded).resolved_field(), "body");
    let by_title = loaded
        .text_indexes
        .get(&index_key("Doc", "title"))
        .expect("both indexes persist");
    assert_eq!(
        by_title.resolved_field(),
        "title",
        "the alias resolution the build performed, not a fresh one"
    );
}

/// A stale index must come back stale, and its outstanding delta must still
/// fold in to exactly what a rebuild would produce. Persisting the topology
/// alone would restore a half-covered index as a current one, and every
/// document written after the save would be silently unsearchable.
#[test]
fn the_freshness_state_round_trips_and_still_folds_in() {
    let mut graph = corpus_graph();
    build_text_index(&mut graph, "Doc", "body", None).expect("build");
    // A creation above the watermark (the gap) …
    run(
        &mut graph,
        "CREATE (:Doc {id: 8, title: 'doc-8', body: 'a later quick document'})",
    );
    // … and a write to an already-covered document (the dirty set).
    run(
        &mut graph,
        "MATCH (d:Doc) WHERE d.id = 3 SET d.body = 'rewritten turtles everywhere'",
    );
    let delta = store(&graph).delta_size(&graph);
    assert_eq!(delta, 2, "one creation, one in-place write");

    let loaded = round_trip(graph);

    assert!(store(&loaded).is_stale(&loaded));
    assert_eq!(
        store(&loaded).delta_size(&loaded),
        delta,
        "the outstanding delta is part of the index's state"
    );
    assert_eq!(
        store(&loaded).refresh(&loaded, "Doc"),
        delta,
        "the refresh re-reads exactly the slots the save recorded"
    );
    assert_eq!(
        all_scores(&loaded),
        rebuild_scores(&loaded),
        "a reloaded index caught up must be indistinguishable from a rebuilt one"
    );
    assert!(!store(&loaded).is_stale(&loaded));
}

/// The auto-refresh ceiling is per index and set by whoever built it; a reload
/// must not quietly restore the default.
#[test]
fn the_auto_refresh_ceiling_round_trips() {
    let mut graph = corpus_graph();
    build_text_index(&mut graph, "Doc", "body", Some(7)).expect("build");

    let loaded = round_trip(graph);

    assert_eq!(store(&loaded).auto_refresh_limit(), 7);
}

/// A file written before this section existed — modelled by a graph that never
/// built an index, which writes no section at all — loads with no index and no
/// error. That is also the byte-stability guarantee: the metadata key is
/// absent, so such a save is byte-for-byte what a pre-0.16.10 build wrote.
#[test]
fn a_file_without_the_section_loads_with_no_index() {
    let mut arc = Arc::new(corpus_graph());
    prepare_kgl_write(&mut arc);
    let mut buf: Vec<u8> = Vec::new();
    write_kgl_to(&arc, &mut buf).unwrap();

    let metadata_len = u32::from_le_bytes([buf[9], buf[10], buf[11], buf[12]]) as usize;
    let metadata: serde_json::Value = serde_json::from_slice(&buf[13..13 + metadata_len]).unwrap();
    assert!(
        metadata.get("text_index_compressed_size").is_none(),
        "an unindexed graph must write the bytes it wrote before this field existed"
    );

    let loaded = load_kgl_bytes(&buf).unwrap();
    assert!(loaded.text_indexes.is_empty());
    assert_eq!(loaded.graph.node_count(), bodies().len());
}

/// Equivalent corpora must serialize identically. The index's own maps are
/// hash maps and its term ids depend on the order documents arrived, so this
/// is a property of the *payload shape* (sorted terms, sorted slots), not
/// something the writer inherits.
#[test]
fn the_section_does_not_depend_on_index_internals() {
    let mut forward = corpus_graph();
    build_text_index(&mut forward, "Doc", "body", None).expect("build");

    // Same corpus, but the index is driven through the incremental path in a
    // different order, so its ids and hash-map layout differ.
    let mut churned = corpus_graph();
    build_text_index(&mut churned, "Doc", "body", None).expect("build");
    let nodes: Vec<_> = bodies()
        .into_iter()
        .rev()
        .map(|(id, _)| churned.lookup_by_id("Doc", &Value::Int64(id)).unwrap())
        .collect();
    let store = churned.text_indexes.get(&index_key("Doc", "body")).unwrap();
    for node in nodes {
        store.note_slot_changed(node);
    }
    store.refresh(&churned, "Doc");

    assert_eq!(
        encode_text_indexes(&forward).unwrap(),
        encode_text_indexes(&churned).unwrap(),
        "the payload records the logical corpus, not one index's internals"
    );
}

/// The payload is a rebuildable cache: a version this build cannot read, or a
/// mangled magic, is skipped silently — never attached, never a load failure.
#[test]
fn an_unreadable_payload_is_skipped_rather_than_attached() {
    let mut source = corpus_graph();
    build_text_index(&mut source, "Doc", "body", None).expect("build");
    let payload = encode_text_indexes(&source).unwrap().unwrap();

    let mut bumped = payload.clone();
    bumped[8] = bumped[8].wrapping_add(1);
    let mut destination = corpus_graph();
    decode_text_indexes(&bumped, &mut destination);
    assert!(
        destination.text_indexes.is_empty(),
        "an unknown payload version must be skipped"
    );

    let mut bad_magic = payload.clone();
    bad_magic[0] = b'X';
    decode_text_indexes(&bad_magic, &mut destination);
    assert!(destination.text_indexes.is_empty());

    let mut truncated = payload.clone();
    truncated.truncate(14);
    decode_text_indexes(&truncated, &mut destination);
    assert!(destination.text_indexes.is_empty());

    // …and the intact payload does attach, so the assertions above are not
    // vacuously green.
    decode_text_indexes(&payload, &mut destination);
    assert!(destination
        .text_indexes
        .contains_key(&index_key("Doc", "body")));
}

/// A version bump in a *saved file* has to survive the whole load path, not
/// just the decoder: the section is framed and digested like any other, so the
/// file still loads and only the index goes missing.
#[test]
fn a_saved_file_with_an_unknown_payload_version_still_loads() {
    let mut graph = corpus_graph();
    build_text_index(&mut graph, "Doc", "body", None).expect("build");
    let mut arc = Arc::new(graph);
    prepare_kgl_write(&mut arc);
    let mut buf: Vec<u8> = Vec::new();
    write_kgl_to(&arc, &mut buf).unwrap();

    // The section is compressed, so the version byte is patched in the payload
    // and the section re-framed: rewriting the compressed bytes in place would
    // fail the section digest, which is a *file damage* signal and would make
    // this test prove the wrong thing.
    let mut payload = encode_text_indexes(&arc).unwrap().unwrap();
    payload[8] = payload[8].wrapping_add(1);
    let patched = replace_text_section(&buf, &payload);

    let loaded = load_kgl_bytes(&patched).unwrap();
    assert!(
        loaded.text_indexes.is_empty(),
        "an index this build cannot read is absent, and the graph loads anyway"
    );
    assert_eq!(loaded.graph.node_count(), bodies().len());
}

/// Rewrite the trailing text-index section of a `.kgl` buffer with `payload`,
/// fixing up the metadata size and digest so only the *payload* is unusual.
fn replace_text_section(buf: &[u8], payload: &[u8]) -> Vec<u8> {
    let metadata_len = u32::from_le_bytes([buf[9], buf[10], buf[11], buf[12]]) as usize;
    let mut metadata: serde_json::Value =
        serde_json::from_slice(&buf[13..13 + metadata_len]).unwrap();
    let old_size = metadata["text_index_compressed_size"].as_u64().unwrap() as usize;
    let compressed = zstd_compress(payload).unwrap();
    metadata["text_index_compressed_size"] = serde_json::Value::from(compressed.len() as u64);
    metadata["section_digests"]["text_index"] =
        serde_json::Value::from(section_digest(&compressed));
    let metadata_json = serde_json::to_vec(&metadata).unwrap();

    let mut out = Vec::new();
    out.extend_from_slice(&buf[..9]);
    out.extend_from_slice(&(metadata_json.len() as u32).to_le_bytes());
    out.extend_from_slice(&metadata_json);
    out.extend_from_slice(&buf[13 + metadata_len..buf.len() - old_size]);
    out.extend_from_slice(&compressed);
    out
}

/// A payload whose recorded dirty set disagrees with its watermark is refused.
/// The tracker cannot produce that state — a slot at or above the watermark is
/// already in the gap — and a refresh would walk it twice.
#[test]
fn a_dirty_slot_above_the_watermark_is_refused() {
    let mut source = corpus_graph();
    build_text_index(&mut source, "Doc", "body", None).expect("build");
    let watermark = store(&source).freshness_state().watermark();
    let entry = PersistedTextIndex {
        node_type: "Doc".to_string(),
        property: "body".to_string(),
        resolved_field: "body".to_string(),
        skipped: 0,
        watermark,
        limit: 1000,
        dirty: vec![watermark],
        terms: vec![("fox".to_string(), vec![Posting { slot: 0, tf: 1 }])],
        empty_docs: Vec::new(),
    };

    let mut destination = corpus_graph();
    decode_text_indexes(&framed(&[entry]), &mut destination);

    assert!(destination.text_indexes.is_empty());
}

/// A corpus that does not describe a coherent index is refused rather than
/// attached: `from_terms` trusts nothing, and `validate` is the gate.
#[test]
fn an_incoherent_corpus_is_refused() {
    let entry = PersistedTextIndex {
        node_type: "Doc".to_string(),
        property: "body".to_string(),
        resolved_field: "body".to_string(),
        skipped: 0,
        watermark: 99,
        limit: 1000,
        dirty: Vec::new(),
        // Two postings for the same slot in one list: not ascending, so the
        // forward view the decoder derives cannot agree with it.
        terms: vec![(
            "fox".to_string(),
            vec![Posting { slot: 2, tf: 1 }, Posting { slot: 2, tf: 3 }],
        )],
        empty_docs: Vec::new(),
    };

    let mut destination = corpus_graph();
    decode_text_indexes(&framed(&[entry]), &mut destination);

    assert!(destination.text_indexes.is_empty());
}

/// An index over a node type this file does not carry is dropped: it could
/// never be queried, and listing it would advertise an index over nothing.
#[test]
fn an_index_over_an_absent_node_type_is_dropped() {
    let entry = PersistedTextIndex {
        node_type: "Ghost".to_string(),
        property: "body".to_string(),
        resolved_field: "body".to_string(),
        skipped: 0,
        watermark: 99,
        limit: 1000,
        dirty: Vec::new(),
        terms: vec![("fox".to_string(), vec![Posting { slot: 2, tf: 1 }])],
        empty_docs: Vec::new(),
    };

    let mut destination = corpus_graph();
    decode_text_indexes(&framed(&[entry]), &mut destination);

    assert!(destination.text_indexes.is_empty());
}

/// Frame owned entries the way `encode_text_indexes` frames borrowed ones —
/// the two serialize to the same bytes (postcard is positional).
fn framed(entries: &[PersistedTextIndex]) -> Vec<u8> {
    let body = codec_ser(serde_codec::CodecVersion::PostcardV1, &entries).unwrap();
    let mut payload = Vec::with_capacity(12 + body.len());
    payload.extend_from_slice(TEXT_INDEX_MAGIC);
    payload.extend_from_slice(&TEXT_INDEX_FORMAT_VERSION.to_le_bytes());
    payload.extend_from_slice(&body);
    payload
}

/// `from_terms` compacts: the free list a churned index accumulates is an
/// internal detail, so what comes back has no holes and still scores the same.
#[test]
fn the_round_trip_compacts_the_term_dictionary() {
    let mut graph = corpus_graph();
    build_text_index(&mut graph, "Doc", "body", None).expect("build");
    delete(&mut graph, 6); // retires "solitary", "hapax", "legomenon"
    let terms = store(&graph).terms();

    let loaded = round_trip(graph);

    assert_eq!(store(&loaded).terms(), terms, "same live vocabulary");
    let view = store(&loaded).read();
    assert_eq!(
        view.index().iter_terms().count(),
        terms,
        "and no freed ids left behind"
    );
}

/// The reuse hazard, across a save: a freed node slot handed to a new node
/// must not inherit the document the file recorded for it.
#[test]
fn a_slot_recycled_after_the_save_does_not_inherit_a_document() {
    let mut graph = corpus_graph();
    build_text_index(&mut graph, "Doc", "body", None).expect("build");
    let freed = graph.lookup_by_id("Doc", &Value::Int64(2)).unwrap();
    delete(&mut graph, 2);

    let mut loaded = round_trip(graph);
    let dir = Arc::make_mut(&mut loaded);
    assert!(
        !store(dir).contains_node(freed),
        "the deleted document must not come back"
    );

    run(
        dir,
        "CREATE (:Doc {id: 9, title: 'doc-9', body: 'unrelated content'})",
    );
    let reused = dir.lookup_by_id("Doc", &Value::Int64(9)).unwrap();
    assert_eq!(
        reused, freed,
        "petgraph hands the freed slot to the next node"
    );

    let view = store(dir).read();
    let prepared = view.prepare_query("quick brown marmoset");
    assert_eq!(
        view.score(reused, &prepared),
        None,
        "a recycled slot carries no ghost document"
    );
}
