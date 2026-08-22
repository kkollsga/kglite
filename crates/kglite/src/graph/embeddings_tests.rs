//! Unit tests for the lifted embedding-ingest primitives.

use super::*;
use crate::graph::schema::NodeData;
use crate::graph::storage::GraphWrite;
use std::collections::HashMap;

/// A graph of `Doc` nodes carrying a `summary` property.
fn docs(ids: &[i64]) -> DirGraph {
    let mut g = DirGraph::new();
    for &id in ids {
        let mut props = HashMap::new();
        props.insert("summary".to_string(), Value::String(format!("text {id}")));
        let nd = NodeData::new(
            Value::Int64(id),
            Value::String(format!("d{id}")),
            "Doc".to_string(),
            props,
            &mut g.interner,
        );
        let idx = GraphWrite::add_node(&mut g.graph, nd);
        g.type_indices.entry_or_default("Doc".to_string()).push(idx);
    }
    g.build_id_index("Doc");
    g
}

fn batch(entries: &[(i64, [f32; 2])]) -> Vec<(Value, Vec<f32>)> {
    entries
        .iter()
        .map(|(id, v)| (Value::Int64(*id), v.to_vec()))
        .collect()
}

fn store_of(g: &DirGraph) -> &EmbeddingStore {
    g.embeddings
        .get(&("Doc".to_string(), "summary_emb".to_string()))
        .expect("store")
}

#[test]
fn set_writes_the_store_and_bumps_the_version() {
    let mut g = docs(&[1, 2]);
    let before = g.version();
    let report = set_embeddings(
        &mut g,
        "Doc",
        "summary",
        None,
        batch(&[(1, [1.0, 0.0]), (2, [0.0, 1.0])]),
    )
    .unwrap();

    assert_eq!(
        report,
        EmbeddingIngestReport {
            embeddings_stored: 2,
            dimension: 2,
            skipped: 0,
            store_created: true,
        }
    );
    assert_eq!(store_of(&g).len(), 2);
    assert!(
        g.version() > before,
        "a non-empty write must bump the version — a receiver that decides \
         'did this write anything?' by comparing versions drops the write otherwise"
    );
}

#[test]
fn empty_batch_is_a_true_no_op_and_does_not_bump() {
    let mut g = docs(&[1]);
    let before = g.version();
    let empty: Vec<(Value, Vec<f32>)> = Vec::new();
    let report = set_embeddings(&mut g, "Doc", "summary", None, empty).unwrap();

    assert_eq!(report, EmbeddingIngestReport::default());
    assert!(g.embeddings.is_empty());
    assert_eq!(g.version(), before);
}

#[test]
fn unresolvable_ids_are_skipped_and_counted() {
    let mut g = docs(&[1]);
    let report = set_embeddings(
        &mut g,
        "Doc",
        "summary",
        None,
        batch(&[(1, [1.0, 0.0]), (99, [0.0, 1.0])]),
    )
    .unwrap();

    assert_eq!(report.embeddings_stored, 1);
    assert_eq!(report.skipped, 1);
}

/// Every id missing means nothing is written — including no empty store, and
/// no version bump for a call that stored nothing.
#[test]
fn all_ids_missing_writes_nothing() {
    let mut g = docs(&[1]);
    let before = g.version();
    let report =
        set_embeddings(&mut g, "Doc", "summary", None, batch(&[(99, [1.0, 0.0])])).unwrap();

    assert_eq!(report.embeddings_stored, 0);
    assert_eq!(report.dimension, 0);
    assert_eq!(report.skipped, 1);
    assert!(g.embeddings.is_empty());
    assert_eq!(g.version(), before);
}

#[test]
fn mismatched_dimensions_are_rejected_before_any_write() {
    let mut g = docs(&[1, 2]);
    let err = set_embeddings(
        &mut g,
        "Doc",
        "summary",
        None,
        vec![
            (Value::Int64(1), vec![1.0f32, 0.0]),
            (Value::Int64(2), vec![1.0f32, 0.0, 0.0]),
        ],
    )
    .unwrap_err();

    assert!(err.contains("Inconsistent embedding dimensions"), "{err}");
    assert!(
        g.embeddings.is_empty(),
        "validate-then-apply: a rejected batch leaves the graph untouched"
    );
}

#[test]
fn unknown_node_type_is_rejected() {
    let mut g = docs(&[1]);
    let err =
        set_embeddings(&mut g, "Ghost", "summary", None, batch(&[(1, [1.0, 0.0])])).unwrap_err();
    assert!(err.contains("does not exist"), "{err}");
}

/// The typo guard: passing the *store* name where the *column* name belongs.
#[test]
fn unknown_source_column_is_rejected() {
    let mut g = docs(&[1]);
    let err = set_embeddings(
        &mut g,
        "Doc",
        "summary_emb",
        None,
        batch(&[(1, [1.0, 0.0])]),
    )
    .unwrap_err();
    assert!(err.contains("not found on any 'Doc' node"), "{err}");
}

/// Unified with `set_embeddings` — `add_embeddings` used to accept a column
/// that exists on no node and quietly create an unreachable store.
#[test]
fn add_applies_the_same_source_column_check() {
    let mut g = docs(&[1]);
    let err = add_embeddings(
        &mut g,
        "Doc",
        "summary_emb",
        None,
        batch(&[(1, [1.0, 0.0])]),
    )
    .unwrap_err();
    assert!(err.contains("not found on any 'Doc' node"), "{err}");
    assert!(g.embeddings.is_empty());
}

#[test]
fn add_creates_then_extends_one_store() {
    let mut g = docs(&[1, 2]);
    let first = add_embeddings(&mut g, "Doc", "summary", None, batch(&[(1, [1.0, 0.0])])).unwrap();
    assert!(first.store_created);
    assert_eq!(first.embeddings_stored, 1);

    let second = add_embeddings(&mut g, "Doc", "summary", None, batch(&[(2, [0.0, 1.0])])).unwrap();
    assert!(!second.store_created);
    assert_eq!(second.embeddings_stored, 2, "the first batch survived");
    assert_eq!(g.embeddings.len(), 1);
}

#[test]
fn add_enforces_the_existing_store_dimension() {
    let mut g = docs(&[1, 2]);
    add_embeddings(&mut g, "Doc", "summary", None, batch(&[(1, [1.0, 0.0])])).unwrap();
    let err = add_embeddings(
        &mut g,
        "Doc",
        "summary",
        None,
        vec![(Value::Int64(2), vec![1.0f32, 0.0, 0.0])],
    )
    .unwrap_err();

    assert!(err.contains("store has 2 but got 3"), "{err}");
    assert_eq!(store_of(&g).len(), 1, "the rejected batch wrote nothing");
}

#[test]
fn set_replaces_the_store_rather_than_extending_it() {
    let mut g = docs(&[1, 2]);
    add_embeddings(&mut g, "Doc", "summary", None, batch(&[(1, [1.0, 0.0])])).unwrap();
    let report = set_embeddings(&mut g, "Doc", "summary", None, batch(&[(2, [0.0, 1.0])])).unwrap();

    assert_eq!(report.embeddings_stored, 1);
    assert_eq!(store_of(&g).len(), 1);
}

#[test]
fn metric_is_recorded_on_the_creating_call() {
    let mut g = docs(&[1, 2]);
    add_embeddings(
        &mut g,
        "Doc",
        "summary",
        Some("euclidean"),
        batch(&[(1, [1.0, 0.0])]),
    )
    .unwrap();
    assert_eq!(store_of(&g).metric.as_deref(), Some("euclidean"));

    // A later add extends the existing store, whose metric already stands.
    add_embeddings(
        &mut g,
        "Doc",
        "summary",
        Some("cosine"),
        batch(&[(2, [0.0, 1.0])]),
    )
    .unwrap();
    assert_eq!(store_of(&g).metric.as_deref(), Some("euclidean"));
}

#[test]
fn borrowed_slices_are_accepted_without_an_intermediate_copy() {
    let mut g = docs(&[1, 2]);
    let packed: Vec<f32> = vec![1.0, 0.0, 0.0, 1.0];
    let entries = [Value::Int64(1), Value::Int64(2)]
        .into_iter()
        .zip(packed.as_chunks::<2>().0.iter().map(|c| &c[..]));
    let report = set_embeddings(&mut g, "Doc", "summary", None, entries).unwrap();
    assert_eq!(report.embeddings_stored, 2);
}

#[test]
fn index_build_reports_defaults_and_the_resolved_metric() {
    let mut g = docs(&[1, 2, 3]);
    set_embeddings(
        &mut g,
        "Doc",
        "summary",
        Some("euclidean"),
        batch(&[(1, [1.0, 0.0]), (2, [0.0, 1.0]), (3, [0.5, 0.5])]),
    )
    .unwrap();

    let report = build_vector_index(&mut g, "Doc", "summary", None, None, None, None).unwrap();
    assert_eq!(report.indexed, 3);
    assert_eq!(
        report.metric, "euclidean",
        "the store's metric is inherited"
    );
    assert_eq!(report.m, HnswParams::default().m);
    assert!(store_of(&g).has_index());
}

#[test]
fn index_build_clamps_out_of_range_tuning() {
    let mut g = docs(&[1, 2]);
    set_embeddings(
        &mut g,
        "Doc",
        "summary",
        None,
        batch(&[(1, [1.0, 0.0]), (2, [0.0, 1.0])]),
    )
    .unwrap();
    let report =
        build_vector_index(&mut g, "Doc", "summary", Some(0), Some(0), Some(0), None).unwrap();
    assert_eq!(report.m, 2);
}

#[test]
fn a_vector_write_drops_the_index() {
    let mut g = docs(&[1, 2]);
    set_embeddings(
        &mut g,
        "Doc",
        "summary",
        None,
        batch(&[(1, [1.0, 0.0]), (2, [0.0, 1.0])]),
    )
    .unwrap();
    build_vector_index(&mut g, "Doc", "summary", None, None, None, None).unwrap();
    assert!(store_of(&g).has_index());

    add_embeddings(&mut g, "Doc", "summary", None, batch(&[(2, [0.3, 0.7])])).unwrap();
    assert!(
        !store_of(&g).has_index(),
        "an index over stale slots must not survive a vector write"
    );
}

#[test]
fn index_build_requires_a_store() {
    let mut g = docs(&[1]);
    let err = build_vector_index(&mut g, "Doc", "summary", None, None, None, None).unwrap_err();
    assert!(
        err.contains("No embedding store 'Doc.summary_emb'"),
        "{err}"
    );
}

/// Passing the *store* name where the text column belongs derives
/// `summary_emb_emb`, which can never exist. The error has to name the column
/// that would have worked, or the caller re-reads their own spelling as
/// correct.
#[test]
fn index_build_on_a_store_name_names_the_text_column() {
    let mut g = docs(&[1]);
    set_embeddings(&mut g, "Doc", "summary", None, batch(&[(1, [1.0, 0.0])])).unwrap();

    let err = build_vector_index(&mut g, "Doc", "summary_emb", None, None, None, None).unwrap_err();
    assert!(
        err.contains("No embedding store 'Doc.summary_emb_emb'"),
        "{err}"
    );
    assert!(err.contains("Did you mean 'summary'?"), "{err}");
    assert!(
        err.contains("build_vector_index() takes the text column"),
        "{err}"
    );
}

/// A genuinely unknown column has no suffix story to tell, so the error falls
/// back to what the type does have embedded.
#[test]
fn index_build_on_an_unknown_column_lists_the_embedded_columns() {
    let mut g = docs(&[1]);
    set_embeddings(&mut g, "Doc", "summary", None, batch(&[(1, [1.0, 0.0])])).unwrap();

    let err = build_vector_index(&mut g, "Doc", "nope", None, None, None, None).unwrap_err();
    assert!(err.contains("No embedding store 'Doc.nope_emb'"), "{err}");
    assert!(err.contains("summary"), "{err}");
}

#[test]
fn poincare_stays_on_the_exact_path() {
    let mut g = docs(&[1]);
    set_embeddings(
        &mut g,
        "Doc",
        "summary",
        Some("poincare"),
        batch(&[(1, [0.1, 0.2])]),
    )
    .unwrap();
    let err = build_vector_index(&mut g, "Doc", "summary", None, None, None, None).unwrap_err();
    assert!(err.contains("poincare"), "{err}");
}

#[test]
fn store_key_derives_the_emb_suffix_once() {
    assert_eq!(
        store_key("Doc", "summary"),
        ("Doc".to_string(), "summary_emb".to_string())
    );
}

#[test]
fn list_embeddings_projects_source_column_and_defaults_metric() {
    let mut g = docs(&[1, 2]);
    assert!(
        list_embeddings(&g).is_empty(),
        "a graph with no stores lists nothing"
    );

    set_embeddings(
        &mut g,
        "Doc",
        "summary",
        Some("dot_product"),
        batch(&[(1, [1.0, 0.0]), (2, [0.0, 1.0])]),
    )
    .unwrap();

    let listing = list_embeddings(&g);
    assert_eq!(
        listing,
        vec![EmbeddingStoreInfo {
            node_type: "Doc".to_string(),
            // both spellings: the source column this API takes, and the store
            // name Cypher's vector_score takes
            text_column: "summary".to_string(),
            store_name: "summary_emb".to_string(),
            dimension: 2,
            count: 2,
            metric: "dot_product".to_string(),
        }]
    );
}

#[test]
fn list_embeddings_defaults_an_unrecorded_metric_to_cosine() {
    let mut g = docs(&[1]);
    set_embeddings(&mut g, "Doc", "summary", None, batch(&[(1, [1.0, 0.0])])).unwrap();
    assert_eq!(list_embeddings(&g)[0].metric, "cosine");
}

// ---------------------------------------------------------------------------
// Identity-alias source columns
//
// `add_nodes(df, "Doc", "id", "name")` hoists the title column *out* of the
// property map and registers `name` as the type's title alias, so a source
// column the user still thinks of as `name` is neither `title` nor a live
// property. These pin that the ingest guard resolves it the way every read
// path does.
// ---------------------------------------------------------------------------

/// `docs()` with `title_alias` registered as the type's original title column
/// — the state `add_nodes(df, "Doc", "id", <title_alias>)` leaves behind.
fn docs_titled(ids: &[i64], title_alias: &str) -> DirGraph {
    let mut g = docs(ids);
    g.title_field_aliases_mut()
        .insert("Doc".to_string(), title_alias.to_string());
    g
}

/// `docs()` with `id_alias` registered as the type's original id column.
fn docs_ided(ids: &[i64], id_alias: &str) -> DirGraph {
    let mut g = docs(ids);
    g.id_field_aliases_mut()
        .insert("Doc".to_string(), id_alias.to_string());
    g
}

#[test]
fn a_per_type_title_alias_is_an_accepted_source_column() {
    let mut g = docs_titled(&[1, 2], "name");
    let report = set_embeddings(&mut g, "Doc", "name", None, batch(&[(1, [1.0, 0.0])])).unwrap();
    assert_eq!(report.embeddings_stored, 1);
}

#[test]
fn a_per_type_id_alias_is_an_accepted_source_column() {
    let mut g = docs_ided(&[1, 2], "doc_no");
    let report = set_embeddings(&mut g, "Doc", "doc_no", None, batch(&[(1, [1.0, 0.0])])).unwrap();
    assert_eq!(report.embeddings_stored, 1);
}

/// No alias map at all: `name` still resolves — structurally, to the title —
/// exactly as `MATCH (n:Doc) RETURN n.name` does.
#[test]
fn the_soft_alias_name_is_accepted_without_any_alias_map() {
    let mut g = docs(&[1]);
    assert!(g.title_field_aliases.is_empty() && g.id_field_aliases.is_empty());
    let report = set_embeddings(&mut g, "Doc", "name", None, batch(&[(1, [1.0, 0.0])])).unwrap();
    assert_eq!(report.embeddings_stored, 1);
}

#[test]
fn the_soft_alias_label_is_accepted() {
    let mut g = docs(&[1]);
    let report = set_embeddings(&mut g, "Doc", "label", None, batch(&[(1, [1.0, 0.0])])).unwrap();
    assert_eq!(report.embeddings_stored, 1);
}

#[test]
fn add_embeddings_accepts_the_same_identity_aliases() {
    let mut g = docs_titled(&[1, 2], "name");
    let report = add_embeddings(&mut g, "Doc", "name", None, batch(&[(1, [1.0, 0.0])])).unwrap();
    assert_eq!(report.embeddings_stored, 1);
}

/// **Store-key decision.** The store is keyed by the spelling the caller
/// passed — resolving `name` to `title` for the *read* never renames the
/// *store*. Pinned because the alternative (canonicalising the key) would
/// strand every store already written under the raw spelling: `add_nodes`'
/// `<col>_emb` ingest keys raw, so does every `.kgl` written before this
/// change, and Cypher's `text_score(n, col, q)` rewrite has no node type to
/// resolve with.
#[test]
fn the_store_is_keyed_by_the_spelling_the_caller_used() {
    let mut g = docs_titled(&[1], "name");
    set_embeddings(&mut g, "Doc", "name", None, batch(&[(1, [1.0, 0.0])])).unwrap();
    assert!(g
        .embeddings
        .contains_key(&("Doc".to_string(), "name_emb".to_string())));
    assert!(!g
        .embeddings
        .contains_key(&("Doc".to_string(), "title_emb".to_string())));
}

/// Accepting aliases must not degrade the typo guard into "anything goes".
#[test]
fn an_unknown_column_is_still_rejected_on_an_aliased_type() {
    let mut g = docs_titled(&[1], "name");
    let err =
        set_embeddings(&mut g, "Doc", "headline", None, batch(&[(1, [1.0, 0.0])])).unwrap_err();
    assert!(err.contains("not found on any 'Doc' node"), "{err}");
}

/// Aliases are per type: another type's title column is not this type's.
#[test]
fn another_types_title_alias_is_not_accepted() {
    let mut g = docs(&[1]);
    g.title_field_aliases_mut()
        .insert("Other".to_string(), "headline".to_string());
    let err =
        set_embeddings(&mut g, "Doc", "headline", None, batch(&[(1, [1.0, 0.0])])).unwrap_err();
    assert!(err.contains("not found on any 'Doc' node"), "{err}");
}

/// The store-name typo stays rejected even on a graph that has alias maps —
/// registering aliases must not make the `_emb` guard fall through.
#[test]
fn the_store_name_typo_is_still_rejected_on_an_aliased_type() {
    let mut g = docs_titled(&[1], "name");
    let err = set_embeddings(
        &mut g,
        "Doc",
        "summary_emb",
        None,
        batch(&[(1, [1.0, 0.0])]),
    )
    .unwrap_err();
    assert!(err.contains("not found on any 'Doc' node"), "{err}");
}
