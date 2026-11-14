//! Text predicates through Cypher: golden rows for every form and mode, the
//! pass on against the pass off, and which route ran. A golden is an absolute
//! expected answer, so a mistake the two paths share cannot hide.

use super::text_column_scan::{take_text_meters, TextDecline};
use crate::datatypes::Value;
use crate::graph::dir_graph::DirGraph;
use crate::graph::session::{execute_mut, execute_read, ExecuteOptions};
use std::collections::{HashMap, HashSet};

const PASS: &str = "text_column_scan";

fn write(graph: &mut DirGraph, query: &str, params: &HashMap<String, Value>) {
    execute_mut(graph, query, &ExecuteOptions::eager(params))
        .unwrap_or_else(|e| panic!("{query}: {e}"));
}

fn s(text: &str) -> Value {
    Value::String(text.to_string())
}

fn golden_graph() -> DirGraph {
    let mut g = DirGraph::new();
    let rows: [(i64, Option<&str>); 10] = [
        (1, Some("alpha one")),
        (2, Some("Alpha two\nthird")),
        (3, Some("ALPHA THREE")),
        (4, Some("beta\nalpha gamma")),
        (5, Some("İstanbul")),
        (6, Some("straße STRASSE")),
        (7, None),
        (8, Some("alpha  gamma")),
        (9, Some("ünïcode ÅLPHA")),
        (10, Some("alphabet soup")),
    ];
    for (id, text) in rows {
        let t = text.map_or(Value::Null, s);
        let params = HashMap::from([("t".to_string(), t), ("u".to_string(), s("note"))]);
        write(
            &mut g,
            &format!("CREATE (:Doc {{id: {id}, t: $t, u: $u}})"),
            &params,
        );
    }
    // A column holding strings beside other kinds: the driver declines it.
    let mixed: [Value; 6] = [
        s("alpha"),
        Value::List(vec![s("alpha"), s("beta")]),
        Value::List(vec![s("alpha"), Value::Int64(5)]),
        Value::Int64(7),
        s("gamma alpha"),
        Value::List(vec![s("x"), s("y")]),
    ];
    for (i, value) in mixed.into_iter().enumerate() {
        let params = HashMap::from([("t".to_string(), value)]);
        write(
            &mut g,
            &format!("CREATE (:Mix {{id: {}, t: $t}})", i + 1),
            &params,
        );
    }
    g
}

fn ids(graph: &DirGraph, query: &str, disabled: &[&str]) -> Vec<i64> {
    let params = HashMap::new();
    let set: HashSet<String> = disabled.iter().map(|d| d.to_string()).collect();
    let mut opts = ExecuteOptions::eager(&params);
    opts.disabled_passes = Some(&set);
    let rows = execute_read(graph, query, &opts)
        .unwrap_or_else(|e| panic!("{query}: {e}"))
        .result
        .rows;
    rows.iter()
        .map(|r| match &r[0] {
            Value::Int64(i) => *i,
            Value::UniqueId(i) => i64::from(*i),
            other => panic!("{other:?}"),
        })
        .collect()
}

/// (query predicate, expected Doc ids). Every one is run with the pass on and
/// off and must give exactly these ids; `routed` says whether the pass is
/// expected to put the text scan in play.
const DOC_GOLDEN: &[(&str, &[i64], bool)] = &[
    ("n.t CONTAINS 'alpha'", &[1, 4, 8, 10], true),
    ("text_match(n.t, 'alpha')", &[1, 4, 8, 10], true),
    (
        "text_match(n.t, 'alpha', {ignore_case: true})",
        &[1, 2, 3, 4, 8, 10],
        true,
    ),
    (
        "text_match(n.t, 'alpha gamma', {mode: 'phrase'})",
        &[4, 8],
        true,
    ),
    (
        "text_match(n.t, 'beta alpha', {mode: 'phrase'})",
        &[4],
        true,
    ),
    (
        "text_match(n.t, '^alpha', {mode: 'regex'})",
        &[1, 8, 10],
        true,
    ),
    ("text_match(n.t, 'alpha$', {mode: 'regex'})", &[], true),
    ("text_match(n.t, 'gamma$', {mode: 'regex'})", &[4, 8], true),
    (
        "text_match(n.t, '\\\\balpha\\\\b', {mode: 'regex', ignore_case: true})",
        &[1, 2, 3, 4, 8],
        true,
    ),
    ("n.t =~ 'alpha.*'", &[1, 8, 10], true),
    ("n.t =~ '.*alpha.*'", &[1, 8, 10], true),
    ("n.t =~ 'alpha'", &[], true),
    ("text_match_regex(n.t, 'alp.a')", &[1, 4, 8, 10], true),
    ("toLower(n.t) CONTAINS 'alpha'", &[1, 2, 3, 4, 8, 10], true),
    ("toLower(n.t) CONTAINS 'i'", &[2, 5], true),
    ("toLower(n.t) CONTAINS 'strasse'", &[6], true),
    ("toLower(n.t) CONTAINS 'straße'", &[6], true),
    ("toLower(n.t) CONTAINS 'i\u{307}s'", &[5], true),
    ("toLower(n.t) CONTAINS 'Alpha'", &[], false),
    ("n.t STARTS WITH 'alpha'", &[1, 8, 10], false),
    ("n.t ENDS WITH 'gamma'", &[4, 8], false),
    ("n.t CONTAINS 'alpha' AND n.id > 4", &[8, 10], true),
    (
        "n.t CONTAINS 'alpha' AND text_match(n.u, 'note')",
        &[1, 4, 8, 10],
        true,
    ),
    ("NOT text_match(n.t, 'alpha')", &[2, 3, 5, 6, 9], false),
    (
        "text_match(n.t, 'alpha') OR n.id = 2",
        &[1, 2, 4, 8, 10],
        false,
    ),
    (
        "text_match(n.t, 'alpha', {ignore_case: true}) AND NOT n.id = 1",
        &[2, 3, 4, 8, 10],
        true,
    ),
];

#[test]
fn golden_rows_hold_with_the_pass_on_and_off_and_the_route_is_the_expected_one() {
    let graph = golden_graph();
    for (predicate, expected, routed) in DOC_GOLDEN {
        let query = format!("MATCH (n:Doc) WHERE {predicate} RETURN n.id AS id ORDER BY id");
        take_text_meters();
        let on = ids(&graph, &query, &[]);
        let meters = take_text_meters();
        let off = ids(&graph, &query, &[PASS]);
        assert_eq!(on, *expected, "pass on: {predicate}");
        assert_eq!(off, *expected, "pass off: {predicate}");
        assert!(
            take_text_meters().is_empty(),
            "the pass was off but the driver ran: {predicate}"
        );
        if *routed {
            assert!(
                !meters.is_empty() && meters.iter().all(Option::is_none),
                "{predicate}: expected the column scan to answer, saw {meters:?}"
            );
        } else {
            assert!(
                meters.is_empty(),
                "{predicate}: not a routed shape, saw {meters:?}"
            );
        }
    }
}

/// Non-string and list values: the driver declines the column and the
/// per-row path answers.
const MIX_GOLDEN: &[(&str, &[i64])] = &[
    ("text_match(n.t, 'alpha')", &[1, 2, 5]),
    ("n.t CONTAINS 'alpha'", &[1, 5]),
    ("text_match(n.t, 'alpha', {ignore_case: true})", &[1, 2, 5]),
    ("toLower(n.t) CONTAINS 'alpha'", &[1, 5]),
    ("text_match(n.t, 'x y', {mode: 'phrase'})", &[6]),
];

#[test]
fn a_column_with_lists_and_non_strings_declines_and_answers_per_row() {
    let graph = golden_graph();
    for (predicate, expected) in MIX_GOLDEN {
        let query = format!("MATCH (n:Mix) WHERE {predicate} RETURN n.id AS id ORDER BY id");
        take_text_meters();
        let on = ids(&graph, &query, &[]);
        let meters = take_text_meters();
        assert_eq!(on, *expected, "{predicate}");
        assert_eq!(ids(&graph, &query, &[PASS]), *expected, "off: {predicate}");
        assert_eq!(
            meters,
            vec![Some(TextDecline::NotStringColumn)],
            "{predicate}"
        );
    }
}

// ── what the pass does to the plan ───────────────────────────────────

fn plan(query: &str, disabled: &[&str]) -> String {
    use crate::graph::languages::cypher::parser::parse_cypher;
    use crate::graph::languages::cypher::planner::optimize_with_disabled;
    let mut parsed = parse_cypher(query).unwrap_or_else(|e| panic!("{query}: {e}"));
    let set: HashSet<String> = disabled.iter().map(|d| d.to_string()).collect();
    let params = HashMap::from([
        ("q".to_string(), s("alpha")),
        ("empty".to_string(), s("")),
        ("o".to_string(), Value::Null),
    ]);
    optimize_with_disabled(&mut parsed, &golden_graph(), &params, &set);
    format!("{:?}", parsed.clauses)
}

fn rewrites(query: &str) -> bool {
    plan(query, &[]) != plan(query, &[PASS])
}

#[test]
fn a_routed_shape_changes_the_plan_and_every_bail_shape_leaves_it_untouched() {
    for routed in [
        "MATCH (n:Doc) WHERE n.t CONTAINS 'alpha' RETURN n",
        "MATCH (n:Doc) WHERE n.t CONTAINS $q RETURN n",
        "MATCH (n:Doc) WHERE text_match(n.t, $q, {mode: 'phrase', ignore_case: true}) RETURN n",
        "MATCH (n:Doc) WHERE toLower(n.t) CONTAINS 'alpha' RETURN n",
        "MATCH (n:Doc) WHERE n.t =~ 'alpha.*' AND n.id > 2 RETURN n",
        "MATCH (n:Doc) WHERE text_match_regex(n.t, 'alp.a') RETURN n",
        "MATCH (n:Doc)-[:R]->(m) WHERE text_match(n.t, 'alpha') RETURN m",
    ] {
        assert!(rewrites(routed), "expected a rewrite: {routed}");
    }
    for bail in [
        // optional, negated, disjunctive
        "OPTIONAL MATCH (n:Doc) WHERE text_match(n.t, 'alpha') RETURN n",
        "MATCH (n:Doc) WHERE NOT text_match(n.t, 'alpha') RETURN n",
        "MATCH (n:Doc) WHERE text_match(n.t, 'alpha') OR n.id = 1 RETURN n",
        // non-constant, empty or invalid arguments
        "MATCH (n:Doc) WHERE text_match(n.t, n.u) RETURN n",
        "MATCH (n:Doc) WHERE text_match(n.t, $empty) RETURN n",
        "MATCH (n:Doc) WHERE text_match(n.t, 'alpha', $q) RETURN n",
        "MATCH (n:Doc) WHERE text_match(n.t, '(', {mode: 'regex'}) RETURN n",
        "MATCH (n:Doc) WHERE text_match(n.t, 'a', {mode: 'nope'}) RETURN n",
        // pattern shapes
        "MATCH (n) WHERE text_match(n.t, 'alpha') RETURN n",
        "MATCH (n:Doc:Other) WHERE text_match(n.t, 'alpha') RETURN n",
        "MATCH (n:Doc|Mix) WHERE text_match(n.t, 'alpha') RETURN n",
        "MATCH (m:Mix)-[:R]->(n:Doc) WHERE text_match(n.t, 'alpha') RETURN n",
        "MATCH (m:Mix) MATCH (n:Doc) WHERE text_match(m.t, 'alpha') RETURN n",
        // forms the pass does not route
        "MATCH (n:Doc) WHERE n.t STARTS WITH 'alpha' RETURN n",
        "MATCH (n:Doc) WHERE n.t ENDS WITH 'alpha' RETURN n",
        "MATCH (n:Doc) WHERE toUpper(n.t) CONTAINS 'ALPHA' RETURN n",
        "MATCH (n:Doc) WHERE toLower(n.t) CONTAINS 'Alpha' RETURN n",
        "MATCH (n:Doc) WHERE n.t CONTAINS '' RETURN n",
        "MATCH (n:Doc) WHERE n.t =~ 'a+' RETURN n",
        "MATCH (n:Doc) WHERE n.t =~ 'alpha|beta' RETURN n",
        "MATCH (n:Doc) WHERE text_match_regex(n.t, 'alpha', 'i') RETURN n",
        "MATCH (n:Doc) WHERE text_match(n.t, 'alpha') = false RETURN n",
        "MATCH (n:Doc) WHERE text_match(n.t || 'x', 'alpha') RETURN n",
    ] {
        assert!(!rewrites(bail), "expected no rewrite: {bail}");
    }
}

#[test]
fn an_indexed_anchor_keeps_the_index_as_the_candidate_source() {
    let mut graph = golden_graph();
    graph.create_index("Doc", "u");
    take_text_meters();
    let query =
        "MATCH (n:Doc {u: 'note'}) WHERE text_match(n.t, 'alpha') RETURN n.id AS id ORDER BY id";
    assert_eq!(ids(&graph, query, &[]), vec![1, 4, 8, 10]);
    assert!(
        take_text_meters().is_empty(),
        "the index served the pattern"
    );
}

// ── db.text.search ───────────────────────────────────────────────────

fn rows_of(graph: &DirGraph, query: &str) -> Vec<Vec<Value>> {
    let params = HashMap::new();
    execute_read(graph, query, &ExecuteOptions::eager(&params))
        .unwrap_or_else(|e| panic!("{query}: {e}"))
        .result
        .rows
}

fn int(i: i64) -> Value {
    Value::Int64(i)
}

fn hits(graph: &DirGraph, config: &str) -> Vec<Vec<Value>> {
    rows_of(
        graph,
        &format!(
            "CALL db.text.search({config}) YIELD node, offset, length \
             RETURN node.id AS id, offset, length ORDER BY id, offset"
        ),
    )
}

fn search_graph() -> DirGraph {
    let mut g = golden_graph();
    for (id, text) in [(11, "alpha alpha alpha"), (12, "éé alpha")] {
        let params = HashMap::from([("t".to_string(), s(text)), ("u".to_string(), s("x"))]);
        write(
            &mut g,
            &format!("CREATE (:Doc {{id: {id}, t: $t, u: $u}})"),
            &params,
        );
    }
    g
}

#[test]
fn the_procedure_yields_one_row_per_occurrence_with_byte_offsets() {
    let g = search_graph();
    take_text_meters();
    assert_eq!(
        hits(&g, "{type: 'Doc', property: 't', query: 'alpha'}"),
        vec![
            vec![int(1), int(0), int(5)],
            vec![int(4), int(5), int(5)],
            vec![int(8), int(0), int(5)],
            vec![int(10), int(0), int(5)],
            vec![int(11), int(0), int(5)],
            vec![int(11), int(6), int(5)],
            vec![int(11), int(12), int(5)],
            vec![int(12), int(5), int(5)],
        ]
    );
    assert_eq!(take_text_meters(), vec![None], "the column driver answered");
    let ci = hits(
        &g,
        "{type: 'Doc', property: 't', query: 'ALPHA', ignore_case: true, `limit`: 4}",
    );
    assert_eq!(
        ci,
        vec![
            vec![int(1), int(0), int(5)],
            vec![int(2), int(0), int(5)],
            vec![int(3), int(0), int(5)],
            vec![int(4), int(5), int(5)],
        ],
        "limit keeps the first hits in node order"
    );
    assert_eq!(
        hits(
            &g,
            "{type: 'Doc', property: 't', query: 'beta alpha', mode: 'phrase'}"
        ),
        vec![vec![int(4), int(0), int(10)]]
    );
}

#[test]
fn the_procedure_reports_character_offsets_and_snippets_on_character_boundaries() {
    let g = search_graph();
    let rows = rows_of(
        &g,
        "CALL db.text.search({type: 'Doc', property: 't', query: 'alpha', snippet_chars: 2}) \
         YIELD node, offset, char_offset, snippet, property \
         WHERE node.id = 12 RETURN offset, char_offset, snippet, property",
    );
    assert_eq!(
        rows,
        vec![vec![int(5), int(3), s("é alpha"), s("t")]],
        "bytes 5, characters 3; two characters of context each side"
    );
}

#[test]
fn within_scopes_the_search_to_the_given_nodes() {
    let g = search_graph();
    let rows = rows_of(
        &g,
        "MATCH (d:Doc) WHERE d.id IN [4, 1, 3, 4] WITH collect(d) AS ps \
         CALL db.text.search({type: 'Doc', property: 't', query: 'alpha', within: ps}) \
         YIELD node RETURN node.id AS id",
    );
    // Scope order, repeats once, only nodes holding a match.
    let got: Vec<i64> = rows
        .iter()
        .map(|r| match &r[0] {
            Value::Int64(i) => *i,
            other => panic!("{other:?}"),
        })
        .collect();
    let mut sorted = got.clone();
    sorted.sort_unstable();
    assert_eq!(sorted, vec![1, 4]);
}

#[test]
fn a_declined_column_gives_the_same_rows_through_the_per_row_path() {
    let mut g = search_graph();
    let served = hits(&g, "{type: 'Doc', property: 't', query: 'alpha'}");
    // A node of another type carrying the label makes the driver decline, and
    // the label's carriers become part of the answer, as `MATCH (n:Doc)` sees them.
    write(&mut g, "MATCH (m:Mix {id: 1}) SET m:Doc", &HashMap::new());
    take_text_meters();
    let fallback = hits(&g, "{type: 'Doc', property: 't', query: 'alpha'}");
    assert_eq!(take_text_meters(), vec![Some(TextDecline::SecondaryLabel)]);
    // Doc ids 1.. also exist as Mix id 1: compare through the node's type.
    let by_type = rows_of(
        &g,
        "CALL db.text.search({type: 'Doc', property: 't', query: 'alpha'}) \
         YIELD node, offset RETURN labels(node)[0] AS label, node.id AS id, offset ORDER BY label, id, offset",
    );
    let docs: Vec<Vec<Value>> = by_type
        .iter()
        .filter(|r| r[0] == s("Doc"))
        .map(|r| vec![r[1].clone(), r[2].clone()])
        .collect();
    let expected: Vec<Vec<Value>> = served
        .iter()
        .map(|r| vec![r[0].clone(), r[1].clone()])
        .collect();
    assert_eq!(docs, expected, "the Doc rows are the served rows");
    assert!(
        by_type.iter().any(|r| r[0] == s("Mix")),
        "the carrier is included"
    );
    assert!(fallback.len() > served.len(), "the carrier adds rows");
    // The title field is not a column the driver serves; the answer is still right.
    take_text_meters();
    let titled = rows_of(
        &search_graph(),
        "CALL db.text.search({type: 'Doc', property: 'title', query: '1'}) YIELD node RETURN count(node) AS c",
    );
    assert_eq!(take_text_meters(), vec![Some(TextDecline::IdentityField)]);
    assert!(matches!(titled[0][0], Value::Int64(c) if c > 0));
}

#[test]
fn the_procedure_refuses_a_bad_call() {
    let g = search_graph();
    let params = HashMap::new();
    for (config, needle) in [
        ("{type: 'Doc', property: 't'}", "missing parameter 'query'"),
        (
            "{type: 'Doc', property: 't', query: 'a', mood: 1}",
            "unknown parameter 'mood'",
        ),
        (
            "{type: 'Doc', property: 't', query: 'a', mode: 'fuzzy'}",
            "unknown mode",
        ),
        (
            "{type: 'Doc', property: 't', query: '(', mode: 'regex'}",
            "invalid pattern",
        ),
        (
            "{type: 'Doc', property: 't', query: 'a', `limit`: -1}",
            "non-negative",
        ),
        (
            "{type: 'Doc', property: 't', query: 'a', within: 5}",
            "list of nodes",
        ),
    ] {
        let err = execute_read(
            &g,
            &format!("CALL db.text.search({config}) YIELD node RETURN node"),
            &ExecuteOptions::eager(&params),
        )
        .err()
        .unwrap_or_else(|| panic!("{config} should fail"));
        assert!(err.to_string().contains(needle), "{config}: {err}");
    }
}

#[test]
fn an_expired_deadline_stops_the_scan_with_the_timeout_error() {
    let g = search_graph();
    let params = HashMap::new();
    let mut opts = ExecuteOptions::eager(&params);
    opts.deadline = Some(std::time::Instant::now() - std::time::Duration::from_secs(1));
    for query in [
        "MATCH (n:Doc) WHERE text_match(n.t, 'alpha') RETURN count(n)",
        "CALL db.text.search({type: 'Doc', property: 't', query: 'alpha'}) YIELD node RETURN node",
    ] {
        let err = execute_read(&g, query, &opts).err().expect(query);
        assert!(
            matches!(err, crate::error::KgError::CypherTimeout { .. }),
            "{query}: {err:?}"
        );
    }
}

// ── the pass on against the pass off ─────────────────────────────────

const WORDS: [&str; 14] = [
    "alpha", "Alpha", "ALPHA", "bravo", "charlie", "ß", "İ", "ſ", "delta ", "\n", "-", "é", "i",
    "ravo",
];

fn text_of(seed: u64) -> String {
    let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let mut text = String::new();
    for _ in 0..(x % 5) {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        text.push_str(WORDS[(x % WORDS.len() as u64) as usize]);
        if x.is_multiple_of(3) {
            text.push(' ');
        }
    }
    text
}

fn corpus(graph: &mut DirGraph, first: i64, count: i64) {
    for id in first..first + count {
        let params = HashMap::from([
            ("t".to_string(), s(&text_of(id as u64))),
            ("g".to_string(), Value::Int64(id % 7)),
        ]);
        write(
            graph,
            &format!("CREATE (:Item {{id: {id}, t: $t, g: $g}})"),
            &params,
        );
    }
}

const EQUIVALENCE_PREDICATES: &[&str] = &[
    "n.t CONTAINS 'alpha'",
    "n.t CONTAINS 'ravo'",
    "text_match(n.t, 'alpha', {ignore_case: true})",
    "text_match(n.t, 'alpha bravo', {mode: 'phrase', ignore_case: true})",
    "text_match(n.t, '^alpha', {mode: 'regex'})",
    "text_match(n.t, 'bravo|charlie', {mode: 'regex', ignore_case: true})",
    "n.t =~ '.*alpha.*'",
    "text_match_regex(n.t, 'alp.a')",
    "toLower(n.t) CONTAINS 'alpha'",
    "toLower(n.t) CONTAINS 'i'",
    "toLower(n.t) CONTAINS 'ss'",
    "toLower(n.t) CONTAINS 'ß'",
    "text_match(n.t, 'alpha') AND n.g = 3",
];

/// Every predicate, with the pass on and off, must give the same ids in the
/// same order; returns how many ran on the column driver.
fn assert_equivalent(graph: &DirGraph, context: &str) -> usize {
    let mut served = 0;
    for predicate in EQUIVALENCE_PREDICATES {
        let query = format!("MATCH (n:Item) WHERE {predicate} RETURN n.id AS id");
        take_text_meters();
        let on = ids(graph, &query, &[]);
        served += take_text_meters().iter().filter(|m| m.is_none()).count();
        let off = ids(graph, &query, &[PASS]);
        assert_eq!(on, off, "{context}: {predicate}");
    }
    served
}

#[test]
fn the_pass_changes_no_answer_across_writes_and_a_transaction_working_copy() {
    use crate::graph::session::Session;
    let mut graph = DirGraph::new();
    corpus(&mut graph, 1, 300);
    assert!(assert_equivalent(&graph, "fresh") >= EQUIVALENCE_PREDICATES.len());
    // Relocated rows, tombstones, new rows, nulls.
    for (id, text) in [
        (
            5,
            "a replacement that is longer, with alpha and ALPHA in it",
        ),
        (6, "ravo"),
    ] {
        let params = HashMap::from([("v".to_string(), s(text))]);
        write(
            &mut graph,
            &format!("MATCH (n:Item {{id: {id}}}) SET n.t = $v"),
            &params,
        );
    }
    write(
        &mut graph,
        "MATCH (n:Item) WHERE n.id % 11 = 0 DELETE n",
        &HashMap::new(),
    );
    write(
        &mut graph,
        "MATCH (n:Item {id: 7}) SET n.t = null",
        &HashMap::new(),
    );
    corpus(&mut graph, 1000, 30);
    assert!(assert_equivalent(&graph, "after writes") >= EQUIVALENCE_PREDICATES.len());

    let session = Session::new(graph);
    let mut tx = session.begin();
    let working = tx.working_mut().unwrap();
    let params = HashMap::from([("v".to_string(), s("alpha inside the transaction"))]);
    write(working, "MATCH (n:Item {id: 9}) SET n.t = $v", &params);
    write(
        working,
        "CREATE (:Item {id: 5000, t: 'alpha new', g: 3})",
        &HashMap::new(),
    );
    assert!(assert_equivalent(working, "transaction working copy") >= EQUIVALENCE_PREDICATES.len());
}

#[test]
fn the_pass_changes_no_answer_on_mapped_columns_with_edited_rows() {
    use crate::graph::io::file::{
        load_file_with, save_graph_with_layout, with_floors, LoadOptions, MapPolicy, SaveLayout,
    };
    use crate::graph::storage::mode::StorageMode;
    let mut graph = DirGraph::new();
    corpus(&mut graph, 1, 200);
    // Filler rows that carry the column past the size a save writes as a
    // mappable section.
    for id in 5000..5040i64 {
        let params = HashMap::from([("t".to_string(), s(&"z".repeat(8000)))]);
        write(
            &mut graph,
            &format!("CREATE (:Item {{id: {id}, t: $t, g: 1}})"),
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
        assert!(assert_equivalent(&loaded, "mapped, fresh") >= EQUIVALENCE_PREDICATES.len());
        let params = HashMap::from([("v".to_string(), s("a longer alpha ALPHA value, relocated"))]);
        let graph = std::sync::Arc::make_mut(&mut loaded);
        write(graph, "MATCH (n:Item {id: 3}) SET n.t = $v", &params);
        write(graph, "MATCH (n:Item {id: 4}) DELETE n", &HashMap::new());
        assert!(
            assert_equivalent(&loaded, "mapped, edited") >= EQUIVALENCE_PREDICATES.len(),
            "{storage:?}"
        );
    }
}
