//! `db.text.rank`: BM25 over a graph-selected scope.
//!
//! The acceptance property: for any scope, the procedure returns what a heap
//! text index built over just the scope's nodes returns for the same query
//! (same nodes, same order, same score to the bit). The oracle here builds
//! that index on a separate graph holding only the scope, so a statistic taken
//! from anywhere but the scope turns it red. A second set of tests pins
//! absolute numbers worked out by hand, which a bug shared with the heap
//! kernel cannot satisfy.

use crate::datatypes::Value;
use crate::graph::core::pattern_matching::text_column_scan::{take_text_meters, TextDecline};
use crate::graph::dir_graph::DirGraph;
use crate::graph::schema::{InternedKey, NodeData};
use crate::graph::session::{execute_mut, execute_read, ExecuteOptions};
use crate::graph::storage::{GraphRead, GraphWrite};
use crate::graph::text_indexes::{build_text_index, text_index_store};
use petgraph::graph::NodeIndex;
use std::collections::HashMap;

fn s(text: &str) -> Value {
    Value::String(text.to_string())
}

fn write(graph: &mut DirGraph, query: &str, params: &HashMap<String, Value>) {
    execute_mut(graph, query, &ExecuteOptions::eager(params))
        .unwrap_or_else(|e| panic!("{query}: {e}"));
}

fn id_of(value: &Value) -> i64 {
    match value {
        Value::Int64(i) => *i,
        Value::UniqueId(i) => i64::from(*i),
        other => panic!("not an id: {other:?}"),
    }
}

fn within_param(scope: &[NodeIndex]) -> HashMap<String, Value> {
    HashMap::from([(
        "w".to_string(),
        Value::List(
            scope
                .iter()
                .map(|n| Value::NodeRef(n.index() as u32))
                .collect(),
        ),
    )])
}

/// `(id, score)` per row, in the order the procedure returned them.
fn rank_with(
    graph: &DirGraph,
    config: &str,
    params: &HashMap<String, Value>,
) -> Result<Vec<(i64, f64)>, String> {
    let query =
        format!("CALL db.text.rank({config}) YIELD node, score RETURN node.id AS id, score");
    let rows = execute_read(graph, &query, &ExecuteOptions::eager(params))
        .map_err(|e| e.to_string())?
        .result
        .rows;
    Ok(rows
        .iter()
        .map(|r| match &r[1] {
            Value::Float64(f) => (id_of(&r[0]), *f),
            other => panic!("score is {other:?}"),
        })
        .collect())
}

fn rank(graph: &DirGraph, config: &str) -> Vec<(i64, f64)> {
    rank_with(graph, config, &HashMap::new()).unwrap_or_else(|e| panic!("{config}: {e}"))
}

// ── the oracle ───────────────────────────────────────────────────────

/// The text a value offers as a document, by the heap index's rule restated
/// here: a string is itself, a list of strings and nulls is its strings joined
/// by a space, and anything else is no document.
fn doc_of(value: &Value) -> Option<String> {
    match value {
        Value::String(t) => Some(t.clone()),
        Value::List(items) => {
            let mut parts = Vec::new();
            for item in items {
                match item {
                    Value::String(t) => parts.push(t.as_str()),
                    Value::Null => {}
                    _ => return None,
                }
            }
            Some(parts.join(" "))
        }
        _ => None,
    }
}

/// What a heap text index over just the scope's nodes returns, as
/// `(original id, score)`: the scope's live nodes of `Doc` in node order
/// become a fresh graph's nodes in the same order, so slots tie-break alike.
fn oracle(
    graph: &DirGraph,
    scope: Option<&[NodeIndex]>,
    query: &str,
    limit: usize,
) -> Vec<(i64, f64)> {
    let type_key = InternedKey::from_str("Doc");
    let mut members: Vec<NodeIndex> = match scope {
        None => graph.type_indices.get("Doc").map_or(vec![], |n| n.to_vec()),
        Some(scope) => scope
            .iter()
            .copied()
            .filter(|&n| {
                graph
                    .graph
                    .node_weight(n)
                    .is_some_and(|d| d.node_type == type_key)
            })
            .collect(),
    };
    members.sort_unstable();
    members.dedup();
    let key = InternedKey::from_str("t");
    let mut small = DirGraph::new();
    let mut original_ids = Vec::new();
    for node in members {
        let view = graph.graph.node_view(node).unwrap();
        let Some(text) = view
            .resolved_field("Doc", "t", key)
            .and_then(|v| doc_of(v.as_ref()))
        else {
            continue;
        };
        original_ids.push(id_of(view.id().as_ref()));
        let data = NodeData::new(
            Value::UniqueId(original_ids.len() as u32),
            Value::Null,
            "Doc".to_string(),
            HashMap::from([("t".to_string(), Value::String(text))]),
            &mut small.interner,
        );
        let idx = small.graph.add_node(data);
        small
            .type_indices
            .entry_or_default("Doc".to_string())
            .push(idx);
    }
    if original_ids.is_empty() || build_text_index(&mut small, "Doc", "t", None).is_err() {
        return Vec::new();
    }
    let index = text_index_store(&small, "Doc", "t").unwrap().read();
    let prepared = index.prepare_query(query);
    index
        .top_k(&prepared, limit)
        .into_iter()
        .map(|(n, score)| (original_ids[n.index()], score))
        .collect()
}

fn assert_same(got: &[(i64, f64)], want: &[(i64, f64)], context: &str) {
    let g: Vec<(i64, u64)> = got.iter().map(|(i, f)| (*i, f.to_bits())).collect();
    let w: Vec<(i64, u64)> = want.iter().map(|(i, f)| (*i, f.to_bits())).collect();
    assert_eq!(g, w, "{context}\n got  {got:?}\n want {want:?}");
}

// ── a seeded corpus ──────────────────────────────────────────────────

const WORDS: [&str; 16] = [
    "alpha",
    "Alpha",
    "ALPHA",
    "alphabet",
    "bravo",
    "ravo",
    "charlie",
    "straße",
    "İstanbul",
    "æøå",
    "Ærlig",
    "日本",
    "語",
    "delta",
    "it's",
    "x",
];
const SEPARATORS: [&str; 4] = [" ", ", ", "-", "\n"];
const QUERIES: [&str; 8] = [
    "alpha",
    "ravo",
    "Alpha bravo",
    "straße æøå",
    "istanbul charlie delta",
    "日本",
    "zzz",
    "the",
];

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

/// Text that always holds the word `the`, so one query term is present in
/// every document.
fn text_of(rng: &mut Rng) -> String {
    let mut text = String::from("the");
    for _ in 0..rng.below(7) {
        text.push_str(SEPARATORS[rng.below(SEPARATORS.len())]);
        text.push_str(WORDS[rng.below(WORDS.len())]);
    }
    text
}

/// 60 `Doc` nodes (ids 1..=60): mostly strings, some null, some empty, and
/// with `lists` some lists and non-text values. 12 `Other` nodes (ids 1001..).
fn corpus(seed: u64, lists: bool) -> DirGraph {
    let mut rng = Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
    let mut g = DirGraph::new();
    for id in 1..=60i64 {
        let value = match (rng.below(10), lists) {
            (0, _) => Value::Null,
            (1, _) => s(""),
            (2, true) => Value::List(vec![s(&text_of(&mut rng)), Value::Null, s("alpha tail")]),
            (3, true) => Value::List(vec![s("x"), Value::Int64(4)]),
            (4, true) => Value::Int64(7),
            _ => s(&text_of(&mut rng)),
        };
        let params = HashMap::from([("t".to_string(), value)]);
        write(
            &mut g,
            &format!("CREATE (:Doc {{id: {id}, t: $t}})"),
            &params,
        );
    }
    for id in 1001..=1012i64 {
        let params = HashMap::from([("t".to_string(), s(&text_of(&mut rng)))]);
        write(
            &mut g,
            &format!("CREATE (:Other {{id: {id}, t: $t}})"),
            &params,
        );
    }
    g
}

fn nodes_of(graph: &DirGraph, node_type: &str) -> Vec<NodeIndex> {
    let mut nodes = graph
        .type_indices
        .get(node_type)
        .map_or(vec![], |n| n.to_vec());
    nodes.sort_unstable();
    nodes
}

/// The scopes every graph state is checked under. `dead` are indices of nodes
/// that no longer exist.
fn scopes(
    graph: &DirGraph,
    dead: &[NodeIndex],
    rng: &mut Rng,
) -> Vec<(&'static str, Vec<NodeIndex>)> {
    let docs = nodes_of(graph, "Doc");
    let others = nodes_of(graph, "Other");
    let mut scattered: Vec<NodeIndex> = docs.iter().copied().step_by(3).collect();
    for i in (1..scattered.len()).rev() {
        scattered.swap(i, rng.below(i + 1));
    }
    let mut mixed: Vec<NodeIndex> = docs.iter().copied().skip(5).take(9).collect();
    mixed.extend(others.iter().copied().take(4));
    mixed.extend(dead.iter().copied());
    mixed.push(NodeIndex::new(1_000_000));
    mixed.extend(docs.iter().copied().skip(7).take(3));
    vec![
        ("empty", vec![]),
        ("single", docs.iter().copied().skip(4).take(1).collect()),
        ("run", docs.iter().copied().skip(10).take(15).collect()),
        ("scattered", scattered),
        ("whole type", docs.clone()),
        ("foreign dead repeated", mixed),
        ("only foreign", others),
    ]
}

/// Every scope against every query, against the heap oracle; the whole type
/// without `within` too. Returns how many calls the column driver served.
fn assert_acceptance(graph: &DirGraph, dead: &[NodeIndex], context: &str) -> usize {
    let mut rng = Rng(0xC0FFEE);
    let mut served = 0;
    for (name, scope) in scopes(graph, dead, &mut rng) {
        let params = within_param(&scope);
        for query in QUERIES {
            for limit in [usize::MAX, 3] {
                let limit_part = if limit == usize::MAX {
                    String::new()
                } else {
                    format!(", `limit`: {limit}")
                };
                let config = format!(
                    "{{type: 'Doc', property: 't', query: '{query}', within: $w{limit_part}}}"
                );
                take_text_meters();
                let got = rank_with(graph, &config, &params).unwrap();
                served += take_text_meters().iter().filter(|m| m.is_none()).count();
                let want = oracle(graph, Some(&scope), query, limit);
                assert_same(
                    &got,
                    &want,
                    &format!("{context} / {name} / '{query}' / {limit}"),
                );
            }
        }
    }
    for query in QUERIES {
        let config = format!("{{type: 'Doc', property: 't', query: '{query}'}}");
        let got = rank(graph, &config);
        assert_same(
            &got,
            &oracle(graph, None, query, usize::MAX),
            &format!("{context} / no within / '{query}'"),
        );
    }
    served
}

#[test]
fn a_scope_ranks_as_an_index_over_just_that_scope_would() {
    for seed in 1..=4u64 {
        let graph = corpus(seed, false);
        take_text_meters();
        let served = assert_acceptance(&graph, &[], &format!("seed {seed}"));
        assert!(served > 0, "the string column answers from the driver");
    }
}

#[test]
fn lists_and_non_text_values_rank_through_the_per_row_route_with_the_same_answer() {
    for seed in 1..=3u64 {
        let graph = corpus(seed, true);
        take_text_meters();
        let _ = rank(&graph, "{type: 'Doc', property: 't', query: 'alpha'}");
        assert_eq!(take_text_meters(), vec![Some(TextDecline::NotStringColumn)]);
        assert_acceptance(&graph, &[], &format!("lists, seed {seed}"));
    }
}

/// Edited, relocated, deleted, new and nulled rows; returns the dead indices.
fn churn(graph: &mut DirGraph) -> Vec<NodeIndex> {
    let before = nodes_of(graph, "Doc");
    for (id, text) in [
        (
            5,
            "the a replacement that is much longer than before, with alpha bravo bravo ALPHA",
        ),
        (6, "the ravo"),
        (7, "the alphabet soup alpha alpha alpha"),
    ] {
        let params = HashMap::from([("v".to_string(), s(text))]);
        write(
            graph,
            &format!("MATCH (n:Doc {{id: {id}}}) SET n.t = $v"),
            &params,
        );
    }
    write(
        graph,
        "MATCH (n:Doc) WHERE n.id % 9 = 0 DELETE n",
        &HashMap::new(),
    );
    write(
        graph,
        "MATCH (n:Doc {id: 8}) SET n.t = null",
        &HashMap::new(),
    );
    write(
        graph,
        "MATCH (n:Doc {id: 10}) SET n.t = ''",
        &HashMap::new(),
    );
    for (id, text) in [(2001, "the new alpha node"), (2002, "the bravo ravo")] {
        let params = HashMap::from([("t".to_string(), s(text))]);
        write(
            graph,
            &format!("CREATE (:Doc {{id: {id}, t: $t}})"),
            &params,
        );
    }
    let after = nodes_of(graph, "Doc");
    before.into_iter().filter(|n| !after.contains(n)).collect()
}

#[test]
fn edits_deletes_and_a_transaction_working_copy_change_no_answer() {
    use crate::graph::session::Session;
    let mut graph = corpus(7, false);
    let dead = churn(&mut graph);
    assert!(!dead.is_empty());
    take_text_meters();
    assert!(assert_acceptance(&graph, &dead, "after writes") > 0);

    let session = Session::new(graph);
    let mut tx = session.begin();
    let working = tx.working_mut().unwrap();
    let params = HashMap::from([("v".to_string(), s("the alpha inside the transaction"))]);
    write(working, "MATCH (n:Doc {id: 11}) SET n.t = $v", &params);
    write(
        working,
        "CREATE (:Doc {id: 3001, t: 'the ravo alpha in a new node'})",
        &HashMap::new(),
    );
    assert!(assert_acceptance(working, &dead, "transaction working copy") > 0);
}

#[test]
fn mapped_columns_with_edited_rows_change_no_answer() {
    use crate::graph::io::file::{
        load_file_with, save_graph_with_layout, with_floors, LoadOptions, MapPolicy, SaveLayout,
    };
    use crate::graph::storage::mode::StorageMode;
    let mut graph = corpus(3, false);
    // Filler carries the column past the size a save writes as a mappable section.
    for id in 5000..5040i64 {
        let params = HashMap::from([("t".to_string(), s(&format!("the {}", "z".repeat(8000))))]);
        write(
            &mut graph,
            &format!("CREATE (:Doc {{id: {id}, t: $t}})"),
            &params,
        );
    }
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("g.kgl");
    let mut arc = std::sync::Arc::new(graph);
    with_floors(10, 10, || {
        save_graph_with_layout(&mut arc, path.to_str().unwrap(), true, SaveLayout::Auto)
    })
    .unwrap();
    for storage in [StorageMode::Memory, StorageMode::Mapped] {
        let mut loaded = load_file_with(
            path.to_str().unwrap(),
            &LoadOptions::new()
                .with_storage(storage)
                .with_columns(MapPolicy::Mapped),
        )
        .unwrap();
        take_text_meters();
        assert!(
            assert_acceptance(&loaded, &[], "mapped, fresh") > 0,
            "{storage:?}"
        );
        let graph = std::sync::Arc::make_mut(&mut loaded);
        let params = HashMap::from([("v".to_string(), s("the a longer alpha ALPHA ravo value"))]);
        write(graph, "MATCH (n:Doc {id: 3}) SET n.t = $v", &params);
        write(
            graph,
            "MATCH (n:Doc {id: 4}) SET n.t = 'the ravo'",
            &HashMap::new(),
        );
        write(graph, "MATCH (n:Doc {id: 13}) DELETE n", &HashMap::new());
        let dead = Vec::new();
        assert!(
            assert_acceptance(&loaded, &dead, "mapped, edited") > 0,
            "{storage:?}"
        );
    }
}

// ── routes agree ─────────────────────────────────────────────────────

#[test]
fn a_declined_column_ranks_the_same_as_a_served_one() {
    let mut g = corpus(5, false);
    // The title holds the same text as `t`, and is not a column the driver serves.
    write(&mut g, "MATCH (n:Doc) SET n.title = n.t", &HashMap::new());
    let all = nodes_of(&g, "Doc");
    let params = within_param(&all[10..30]);
    for query in QUERIES {
        for within in ["", ", within: $w"] {
            let served = format!("{{type: 'Doc', property: 't', query: '{query}'{within}}}");
            let declined = format!("{{type: 'Doc', property: 'title', query: '{query}'{within}}}");
            take_text_meters();
            let a = rank_with(&g, &served, &params).unwrap();
            assert_eq!(take_text_meters(), vec![None], "{query}");
            let b = rank_with(&g, &declined, &params).unwrap();
            assert_eq!(
                take_text_meters(),
                vec![Some(TextDecline::IdentityField)],
                "{query}"
            );
            assert_same(&a, &b, &format!("served vs declined '{query}'{within}"));
        }
    }
    // A secondary label on a node of another type declines the driver and
    // adds nothing to the corpus: the label's carrier is not a `Doc`.
    let before = rank(&g, "{type: 'Doc', property: 't', query: 'alpha ravo'}");
    write(
        &mut g,
        "MATCH (m:Other {id: 1001}) SET m:Doc",
        &HashMap::new(),
    );
    take_text_meters();
    let after = rank(&g, "{type: 'Doc', property: 't', query: 'alpha ravo'}");
    assert_eq!(take_text_meters(), vec![Some(TextDecline::SecondaryLabel)]);
    assert_same(&before, &after, "secondary-label carrier");
}

// ── absolute values, worked by hand ──────────────────────────────────

fn hand_graph() -> DirGraph {
    let mut g = DirGraph::new();
    for (id, text) in [
        (1, Some("apple banana apple")),
        (2, Some("banana cherry")),
        (3, Some("apple")),
        (4, Some("cherry cherry cherry date")),
        (5, None),
        (6, Some("apple")),
    ] {
        let params = HashMap::from([("t".to_string(), text.map_or(Value::Null, s))]);
        write(
            &mut g,
            &format!("CREATE (:Doc {{id: {id}, t: $t}})"),
            &params,
        );
    }
    g
}

/// One term's contribution: `idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len / avgdl))`
/// with k1 = 1.2 and b = 0.75 written out.
fn part(idf: f64, tf: f64, len: f64, avgdl: f64) -> f64 {
    idf * (tf * 2.2) / (tf + 1.2 * (0.25 + 0.75 * len / avgdl))
}

fn close(got: f64, want: f64) {
    assert!(
        (got - want).abs() <= 1e-12 * want.abs(),
        "got {got}, want {want}"
    );
}

#[test]
fn a_hand_computed_corpus_scores_the_worked_numbers() {
    let g = hand_graph();
    // Five documents (id 5 is null and is not one): lengths 3, 2, 1, 4, 1;
    // average 11 / 5 = 2.2.
    let avgdl = 2.2;
    // apple: df 3 (ids 1, 3, 6), N = 5: idf = ln(1 + 2.5 / 3.5).
    let idf_apple = (1.0f64 + 2.5 / 3.5).ln();
    let rows = rank(&g, "{type: 'Doc', property: 't', query: 'Apple'}");
    let ids: Vec<i64> = rows.iter().map(|r| r.0).collect();
    // Shorter documents outrank the longer one with tf 2; ids 3 and 6 tie and
    // keep node order.
    assert_eq!(ids, vec![3, 6, 1]);
    close(rows[0].1, part(idf_apple, 1.0, 1.0, avgdl));
    close(rows[1].1, part(idf_apple, 1.0, 1.0, avgdl));
    close(rows[2].1, part(idf_apple, 2.0, 3.0, avgdl));
    assert_eq!(rows[0].1.to_bits(), rows[1].1.to_bits());

    // Two terms: scores add in the order the query names them.
    // cherry: df 2 (ids 2, 4): idf = ln(1 + 3.5 / 2.5).
    let idf_cherry = (1.0f64 + 3.5 / 2.5).ln();
    let rows = rank(
        &g,
        "{type: 'Doc', property: 't', query: 'cherry apple apple'}",
    );
    let want = [
        (4, part(idf_cherry, 3.0, 4.0, avgdl)),
        (6, part(idf_apple, 1.0, 1.0, avgdl)),
        (3, part(idf_apple, 1.0, 1.0, avgdl)),
        (2, part(idf_cherry, 1.0, 2.0, avgdl)),
        (1, part(idf_apple, 2.0, 3.0, avgdl)),
    ];
    let mut expected: Vec<(i64, f64)> = want.to_vec();
    expected.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    assert_eq!(
        rows.iter().map(|r| r.0).collect::<Vec<_>>(),
        expected.iter().map(|r| r.0).collect::<Vec<_>>()
    );
    for (got, want) in rows.iter().zip(&expected) {
        close(got.1, want.1);
    }
    // A document sharing no term (the null one, a banana-only one) has no row.
    assert!(rows.iter().all(|r| r.0 != 5));
}

#[test]
fn the_scope_is_the_corpus_and_a_term_is_a_token() {
    let g = hand_graph();
    let all = nodes_of(&g, "Doc");
    // Scope {1, 3}: N = 2, lengths 3 and 1, average 2, apple in both.
    let scope = vec![all[0], all[2]];
    let rows = rank_with(
        &g,
        "{type: 'Doc', property: 't', query: 'apple', within: $w}",
        &within_param(&scope),
    )
    .unwrap();
    let idf = (1.0f64 + 0.5 / 2.5).ln();
    assert_eq!(rows.iter().map(|r| r.0).collect::<Vec<_>>(), vec![3, 1]);
    close(rows[0].1, part(idf, 1.0, 1.0, 2.0));
    close(rows[1].1, part(idf, 2.0, 3.0, 2.0));
    // Scope {2, 3}: apple in one of two: idf = ln(1 + 1.5 / 1.5).
    let scope = vec![all[1], all[2]];
    let rows = rank_with(
        &g,
        "{type: 'Doc', property: 't', query: 'apple', within: $w}",
        &within_param(&scope),
    )
    .unwrap();
    assert_eq!(rows.len(), 1);
    close(rows[0].1, part(2.0f64.ln(), 1.0, 1.0, 1.5));
    // `app` is a prefix of a token, not a token.
    assert!(rank(&g, "{type: 'Doc', property: 't', query: 'app'}").is_empty());
}

#[test]
fn limit_order_and_empty_scopes() {
    let g = hand_graph();
    let all = rank(&g, "{type: 'Doc', property: 't', query: 'apple'}");
    assert_eq!(all.len(), 3);
    for k in 0..=4usize {
        let got = rank(
            &g,
            &format!("{{type: 'Doc', property: 't', query: 'apple', `limit`: {k}}}"),
        );
        assert_same(&got, &all[..k.min(3)], &format!("limit {k}"));
    }
    let empty = rank_with(
        &g,
        "{type: 'Doc', property: 't', query: 'apple', within: $w}",
        &within_param(&[]),
    )
    .unwrap();
    assert!(empty.is_empty());
    // A query with no token ranks nothing.
    assert!(rank(&g, "{type: 'Doc', property: 't', query: '!!! ??'}").is_empty());
    // A type or property nothing holds is an empty answer, as in db.text.search.
    assert!(rank(&g, "{type: 'Nope', property: 't', query: 'apple'}").is_empty());
    assert!(rank(&g, "{type: 'Doc', property: 'nope', query: 'apple'}").is_empty());
}

#[test]
fn a_bad_call_is_refused() {
    let g = hand_graph();
    for (config, needle) in [
        ("{type: 'Doc', property: 't'}", "missing parameter 'query'"),
        ("{property: 't', query: 'a'}", "missing parameter 'type'"),
        (
            "{type: 'Doc', property: 't', query: ''}",
            "non-empty string",
        ),
        (
            "{type: 'Doc', property: 't', query: 'a', mode: 'regex'}",
            "unknown parameter 'mode'",
        ),
        (
            "{type: 'Doc', property: 't', query: 'a', `limit`: -1}",
            "non-negative",
        ),
        (
            "{type: 'Doc', property: 't', query: 'a', within: 5}",
            "list of nodes",
        ),
        (
            "{type: 'Doc', property: 't', query: 'a', within: [1]}",
            "only nodes",
        ),
    ] {
        let err = rank_with(&g, config, &HashMap::new())
            .err()
            .unwrap_or_else(|| panic!("{config} should fail"));
        assert!(err.contains(needle), "{config}: {err}");
    }
}

#[test]
fn a_pattern_scope_through_collect_ranks_the_pattern_s_nodes() {
    let mut g = hand_graph();
    write(
        &mut g,
        "CREATE (:Folder {id: 1}), (:Folder {id: 2})",
        &HashMap::new(),
    );
    for (case, doc) in [(1, 1), (1, 3), (1, 4), (2, 2), (2, 6)] {
        write(
            &mut g,
            &format!(
                "MATCH (c:Folder {{id: {case}}}), (d:Doc {{id: {doc}}}) CREATE (c)-[:HAS]->(d)"
            ),
            &HashMap::new(),
        );
    }
    let rows = execute_read(
        &g,
        "MATCH (c:Folder {id: 1})-[:HAS]->(d:Doc) WITH collect(d) AS ds \
         CALL db.text.rank({type: 'Doc', property: 't', query: 'apple', within: ds}) \
         YIELD node, score RETURN node.id AS id, score",
        &ExecuteOptions::eager(&HashMap::new()),
    )
    .unwrap()
    .result
    .rows;
    // Documents 1, 3, 4: lengths 3, 1, 4, average 8/3; apple in two of three.
    let idf = (1.0f64 + 1.5 / 2.5).ln();
    let avgdl = 8.0 / 3.0;
    assert_eq!(rows.len(), 2);
    assert_eq!(id_of(&rows[0][0]), 3);
    assert_eq!(id_of(&rows[1][0]), 1);
    let score = |row: &Vec<Value>| match &row[1] {
        Value::Float64(f) => *f,
        other => panic!("{other:?}"),
    };
    close(score(&rows[0]), part(idf, 1.0, 1.0, avgdl));
    close(score(&rows[1]), part(idf, 2.0, 3.0, avgdl));
}

#[test]
fn an_expired_deadline_stops_the_ranking_with_the_timeout_error() {
    // More documents than the driver's polling stride, on both routes.
    let mut g = DirGraph::new();
    for id in 0..200i64 {
        write(
            &mut g,
            &format!("CREATE (:Doc {{id: {id}, t: 'the alpha {id}', title: 'the alpha {id}'}})"),
            &HashMap::new(),
        );
    }
    let params = HashMap::new();
    let mut opts = ExecuteOptions::eager(&params);
    opts.deadline = Some(std::time::Instant::now() - std::time::Duration::from_secs(1));
    for property in ["t", "title"] {
        let query = format!(
            "CALL db.text.rank({{type: 'Doc', property: '{property}', query: 'alpha'}}) \
             YIELD node RETURN node"
        );
        let err = execute_read(&g, &query, &opts).err().expect(&query);
        assert!(
            matches!(err, crate::error::KgError::CypherTimeout { .. }),
            "{query}: {err:?}"
        );
    }
}
