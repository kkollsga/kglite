//! Lazy-eligibility, text_score, NDV-selectivity and top-K fusion planner
//! tests extracted from planner_tests.rs.

use super::*;
use crate::graph::languages::cypher::parser::parse_cypher;

/// The lazy-eligibility contract, pinned as a corpus.
///
/// `mark_lazy_eligibility` decides whether a result is returned deferred, and a
/// deferred result holds an `Arc<DirGraph>` for its whole life — which makes the
/// next write through the owning graph copy-on-write the entire graph. So this
/// gate is not merely a projection optimisation: it decides who pays an O(V+E)
/// fork. Pinning the exact shape keeps that reach visible and honest.
///
/// The surprising member is `WHERE`: `optimize` keeps a standalone
/// `Clause::Where` as a safety net even after pushing every predicate into the
/// MATCH (see `test_predicate_pushdown_simple`), and the eligibility walk has no
/// arm for it. So the same point lookup is eligible written with an inline map
/// and ineligible written with `WHERE`.
#[test]
fn lazy_eligibility_corpus() {
    fn is_lazy(q: &str) -> bool {
        let mut query = parse_cypher(q).unwrap();
        let graph = DirGraph::new();
        let params = HashMap::new();
        optimize(&mut query, &graph, &params);
        mark_lazy_eligibility(&mut query);
        query.clauses.iter().any(|c| match c {
            Clause::Return(r) => r.lazy_eligible,
            _ => false,
        })
    }

    // Eligible: bare property projections over an unfiltered or inline-filtered
    // MATCH, with nothing but SKIP/LIMIT after the RETURN.
    for q in [
        "MATCH (u:User) RETURN u.name",
        "MATCH (u:User {id: 1}) RETURN u.name, u.email",
        "MATCH (u:User {id: 1}) RETURN u.name AS name",
        "MATCH (u:User) RETURN u.name LIMIT 10",
        "MATCH (u:User)-[:OWNS]->(t:Task) RETURN u.name, t.title",
        "OPTIONAL MATCH (u:User) RETURN u.name",
    ] {
        assert!(is_lazy(q), "expected lazy-eligible: {q}");
    }

    // Ineligible. Each of these takes the eager path and therefore never pins
    // the graph, whatever its size.
    for q in [
        // A standalone WHERE survives optimisation and disqualifies.
        "MATCH (u:User) WHERE u.id = 1 RETURN u.name",
        // Whole-node returns resolve via NodeRef, not the lazy resolver.
        "MATCH (u:User) RETURN u",
        // Any non-PropertyAccess return item.
        "MATCH (u:User) RETURN u.age + 1",
        "MATCH (u:User) RETURN count(u)",
        // Ordering, dedup and multi-stage pipelines all disqualify.
        "MATCH (u:User) RETURN u.name ORDER BY u.name",
        "MATCH (u:User) RETURN DISTINCT u.name",
        "MATCH (u:User) WITH u.name AS n RETURN n",
        "UNWIND [1, 2] AS x RETURN x",
    ] {
        assert!(!is_lazy(q), "expected NOT lazy-eligible: {q}");
    }

    // The same lookup, two spellings, opposite classifications. Asserted as an
    // explicit pair because it is the least defensible part of the rule: the
    // two queries are semantically identical and a user has no way to know
    // which one they wrote. Whichever way the rule moves, these two should
    // arrive at the same answer — if a future change makes them agree, delete
    // this pair rather than "fixing" it.
    assert!(is_lazy("MATCH (u:User {id: 1}) RETURN u.name"));
    assert!(!is_lazy("MATCH (u:User) WHERE u.id = 1 RETURN u.name"));

    // The exact shapes the graph-pin benchmark relies on. Pinned here because
    // measuring the pin with an ineligible read reports no change and looks
    // like a working fix doing nothing.
    assert!(is_lazy(
        "MATCH (p:Person {id: 0}) RETURN p.name AS name, p.age AS age"
    ));
    assert!(!is_lazy(
        "MATCH (p:Person) WHERE p.id = 0 RETURN p.name AS name, p.age AS age"
    ));
    assert!(is_lazy(
        "MATCH (p:Person) RETURN p.name AS name, p.age AS age"
    ));
}

// ============================================================================
// text_score() — raw query vectors
// ============================================================================
//
// `text_score(n, col, q)` is `vector_score(n, '{col}_emb', q)` after this
// rewrite. A *vector*-shaped `q` therefore has nothing to embed: it passes
// straight through, `texts_to_embed` stays empty, and `execute` never
// consults an embedder (the embedder call is gated on a non-empty collect
// list). A *string*-shaped `q` stays text even when it looks like
// `"[1.0, 2.0]"` — that ambiguity is resolved in favour of text, and
// CYPHER.md says so.

fn rewrite_ts(
    query: &str,
    params: &HashMap<String, Value>,
) -> Result<(CypherQuery, Vec<(String, String)>), String> {
    let mut parsed = parse_cypher(query).unwrap();
    let rewrite = simplification::rewrite_text_score(&mut parsed, params)?;
    Ok((parsed, rewrite.texts_to_embed))
}

/// The `text_score(...)` call in the first RETURN item.
fn first_return_call(query: &CypherQuery) -> (&String, &Vec<Expression>) {
    for clause in &query.clauses {
        if let Clause::Return(r) = clause {
            if let Expression::FunctionCall { name, args, .. } = &r.items[0].expression {
                return (name, args);
            }
        }
    }
    panic!("expected a function call in the first RETURN item");
}

#[test]
fn test_text_score_list_parameter_passes_through() {
    let mut params = HashMap::new();
    params.insert(
        "q".to_string(),
        Value::List(vec![Value::Float64(1.0), Value::Float64(0.0)]),
    );
    let (query, texts) = rewrite_ts(
        "MATCH (n:Doc) RETURN text_score(n, 'summary', $q) AS s",
        &params,
    )
    .unwrap();

    assert!(texts.is_empty(), "a vector query must collect no text");

    let (name, args) = first_return_call(&query);
    assert_eq!(name, "vector_score");
    assert!(matches!(
        &args[1],
        Expression::Literal(Value::String(s)) if s == "summary_emb"
    ));
    // arg 2 is untouched — the caller's own parameter reaches vector_score.
    assert!(matches!(&args[2], Expression::Parameter(p) if p == "q"));
}

#[test]
fn test_text_score_list_literal_passes_through() {
    let params = HashMap::new();
    let (query, texts) = rewrite_ts(
        "MATCH (n:Doc) RETURN text_score(n, 'summary', [1.0, 0.0]) AS s",
        &params,
    )
    .unwrap();

    assert!(texts.is_empty());
    let (name, args) = first_return_call(&query);
    assert_eq!(name, "vector_score");
    assert!(matches!(
        &args[1],
        Expression::Literal(Value::String(s)) if s == "summary_emb"
    ));
    assert!(matches!(&args[2], Expression::ListLiteral(_)));
}

#[test]
fn test_text_score_metric_arg_survives_vector_passthrough() {
    let mut params = HashMap::new();
    params.insert(
        "q".to_string(),
        Value::List(vec![Value::Float64(1.0), Value::Float64(0.0)]),
    );
    let (query, texts) = rewrite_ts(
        "MATCH (n:Doc) RETURN text_score(n, 'summary', $q, 'euclidean') AS s",
        &params,
    )
    .unwrap();

    assert!(texts.is_empty());
    let (name, args) = first_return_call(&query);
    assert_eq!(name, "vector_score");
    assert_eq!(args.len(), 4);
    assert!(matches!(
        &args[3],
        Expression::Literal(Value::String(m)) if m == "euclidean"
    ));
}

#[test]
fn test_text_score_string_parameter_still_collects_text() {
    let mut params = HashMap::new();
    params.insert("q".to_string(), Value::String("hello".to_string()));
    let (query, texts) = rewrite_ts(
        "MATCH (n:Doc) RETURN text_score(n, 'summary', $q) AS s",
        &params,
    )
    .unwrap();

    assert_eq!(texts.len(), 1);
    assert_eq!(texts[0].1, "hello");
    let (name, args) = first_return_call(&query);
    assert_eq!(name, "vector_score");
    assert!(matches!(&args[2], Expression::Parameter(p) if p == &texts[0].0));
}

#[test]
fn test_text_score_json_shaped_string_stays_text() {
    // Locked decision: a string is query *text* in text_score, even when it
    // parses as a JSON vector. vector_score keeps the legacy JSON-string form.
    let mut params = HashMap::new();
    params.insert("q".to_string(), Value::String("[1.0, 0.0]".to_string()));
    let (_, texts) = rewrite_ts(
        "MATCH (n:Doc) RETURN text_score(n, 'summary', $q) AS s",
        &params,
    )
    .unwrap();
    assert_eq!(texts.len(), 1);
    assert_eq!(texts[0].1, "[1.0, 0.0]");
}

#[test]
fn test_text_score_rejects_non_string_non_list_parameter() {
    let mut params = HashMap::new();
    params.insert("q".to_string(), Value::Int64(7));
    let err = rewrite_ts(
        "MATCH (n:Doc) RETURN text_score(n, 'summary', $q) AS s",
        &params,
    )
    .unwrap_err();
    assert!(
        err.contains("must be a string or a list of numbers"),
        "unexpected error: {err}"
    );
}

#[test]
fn test_text_score_unknown_parameter_still_errors() {
    let params = HashMap::new();
    let err = rewrite_ts(
        "MATCH (n:Doc) RETURN text_score(n, 'summary', $q) AS s",
        &params,
    )
    .unwrap_err();
    assert!(err.contains("not found"), "unexpected error: {err}");
}

// ============================================================================
// NDV selectivity on identity fields
// ============================================================================
//
// `property_ndv` feeds `estimate_node_selectivity`'s `type_count / ndv`
// equality estimate. It used to read the property map only, which does not
// hold a type's `node_title_field` (`add_nodes` hoists that column into
// `NodeData.title`), so the scan found nothing, `.max(1)` reported NDV = 1,
// and the filter was scored *completely non-selective* — the planner then
// anchored on the other, larger end of the pattern. These pin the anchor
// choice, which the Cypher differential corpus cannot see (both plans return
// the same rows; only the cost differs).

/// `Doc` (the unfiltered end) outnumbers `Keyword` 3:1, and every `Keyword`
/// has a distinct title. `Keyword` is therefore the right anchor for a
/// `title`-equality filter (one node) and the wrong one only if the filter is
/// scored as matching the whole type.
fn title_anchor_graph() -> DirGraph {
    fn typed(graph: &mut DirGraph, node_type: &str, n: i64) {
        let rows: Vec<Vec<Value>> = (1..=n)
            .map(|i| {
                vec![
                    Value::Int64(i),
                    Value::String(format!("{}-{i}", node_type.to_lowercase())),
                ]
            })
            .collect();
        let df = crate::datatypes::DataFrame::from_cypher_rows(
            vec!["id".to_string(), "title".to_string()],
            rows,
        )
        .unwrap();
        crate::graph::mutation::maintain::add_nodes(
            graph,
            df,
            node_type.to_string(),
            "id".to_string(),
            Some("title".to_string()),
            None,
        )
        .unwrap();
    }
    let mut graph = DirGraph::new();
    typed(&mut graph, "Doc", 3000);
    typed(&mut graph, "Keyword", 1000);
    graph
}

fn optimized_start_variable(query: &str, graph: &DirGraph) -> String {
    let mut query = parse_cypher(query).unwrap();
    optimize(&mut query, graph, &HashMap::new());
    let m = query
        .clauses
        .iter()
        .find_map(|c| match c {
            Clause::Match(m) => Some(m),
            _ => None,
        })
        .expect("expected MATCH clause");
    match &m.patterns[0].elements[0] {
        PatternElement::Node(np) => np
            .variable
            .clone()
            .expect("start node should carry a variable"),
        _ => panic!("expected start node"),
    }
}

#[test]
fn test_ndv_counts_the_title_field() {
    let graph = title_anchor_graph();
    assert_eq!(
        graph.property_ndv("Keyword", "title"),
        Some(1000),
        "`title` is Keyword's node_title_field, so its distinct values live on \
         NodeData.title, not in the property map; reporting 1 (or None) makes \
         the planner score a title equality filter as non-selective"
    );
}

#[test]
fn test_title_equality_anchors_on_the_filtered_type() {
    let graph = title_anchor_graph();
    assert_eq!(
        optimized_start_variable(
            "MATCH (a:Doc)-[:MENTIONS]->(b:Keyword) WHERE b.title = 'keyword-7' RETURN a, b",
            &graph,
        ),
        "b",
        "a unique title equality selects one Keyword; anchoring on the 3000 \
         Docs instead means the filter was scored non-selective (NDV=1)"
    );
}

#[test]
fn test_title_in_list_anchors_on_the_filtered_type() {
    let graph = title_anchor_graph();
    assert_eq!(
        optimized_start_variable(
            "MATCH (a:Doc)-[:MENTIONS]->(b:Keyword) \
             WHERE b.title IN ['keyword-7', 'keyword-9'] RETURN a, b",
            &graph,
        ),
        "b",
        "PropertyMatcher::In reads the same NDV; two of 1000 distinct titles \
         is far more selective than a full Doc scan"
    );
}

/// The reporter's real shape: the filtered type names its identity columns
/// itself (`add_nodes(unique_id_field='term_id', node_title_field='term_name')`),
/// so `term_name` is a *registered alias* for the title field — the matcher
/// resolves it, and the statistic feeding the planner has to resolve it too.
fn aliased_identity_graph() -> DirGraph {
    let mut graph = DirGraph::new();
    let rows: Vec<Vec<Value>> = (1..=3000)
        .map(|i| vec![Value::Int64(i), Value::String(format!("doc-{i}"))])
        .collect();
    let df = crate::datatypes::DataFrame::from_cypher_rows(vec!["id".into(), "title".into()], rows)
        .unwrap();
    crate::graph::mutation::maintain::add_nodes(
        &mut graph,
        df,
        "Doc".to_string(),
        "id".to_string(),
        Some("title".to_string()),
        None,
    )
    .unwrap();

    let rows: Vec<Vec<Value>> = (1..=1000)
        .map(|i| vec![Value::Int64(i), Value::String(format!("term-{i}"))])
        .collect();
    let df = crate::datatypes::DataFrame::from_cypher_rows(
        vec!["term_id".into(), "term_name".into()],
        rows,
    )
    .unwrap();
    crate::graph::mutation::maintain::add_nodes(
        &mut graph,
        df,
        "Term".to_string(),
        "term_id".to_string(),
        Some("term_name".to_string()),
        None,
    )
    .unwrap();
    graph
}

#[test]
fn test_aliased_title_equality_anchors_on_the_filtered_type() {
    let graph = aliased_identity_graph();
    assert_eq!(
        graph.property_ndv("Term", "term_name"),
        Some(1000),
        "the statistic has to resolve the alias, not just the anchor it feeds"
    );
    assert_eq!(
        optimized_start_variable(
            "MATCH (a:Doc)-[:MENTIONS]->(b:Term) WHERE b.term_name = 'term-7' RETURN a, b",
            &graph,
        ),
        "b",
        "`term_name` is Term's registered title alias — the matcher resolves it \
         to the title field, so the NDV statistic must resolve it the same way"
    );
}

#[test]
fn test_aliased_id_equality_anchors_on_the_filtered_type() {
    let graph = aliased_identity_graph();
    assert_eq!(
        graph.property_ndv("Term", "term_id"),
        Some(1000),
        "the statistic has to resolve the alias, not just the anchor it feeds"
    );
    assert_eq!(
        optimized_start_variable(
            "MATCH (a:Doc)-[:MENTIONS]->(b:Term) WHERE b.term_id = 7 RETURN a, b",
            &graph,
        ),
        "b",
        "`term_id` is Term's registered id alias; only a literal `id` gets the \
         dedicated selectivity-1 path, so the alias has to come out of the NDV \
         statistic"
    );
}

#[test]
fn test_absent_property_is_no_information_not_zero_selectivity() {
    // The safety net behind the alias fix: when the scan finds *no* values at
    // all, "distinct = 0" must not collapse into "NDV = 1" — that reads as
    // `type_count / 1`, i.e. a filter that excludes nothing, and anchors the
    // join on the other, larger end. No information means fall back to the
    // flat heuristic.
    let graph = aliased_identity_graph();
    assert_eq!(
        graph.property_ndv("Term", "not_a_property"),
        None,
        "an empty scan is no information, not NDV=1"
    );
    assert_eq!(
        optimized_start_variable(
            "MATCH (a:Doc)-[:MENTIONS]->(b:Term) WHERE b.not_a_property = 'x' RETURN a, b",
            &graph,
        ),
        "b",
        "scanning the 1000 filtered Terms beats driving 3000 Docs through the \
         same filter, however unselective the estimate"
    );
}

// ============================================================================
// Multi-key top-K fusion
// ============================================================================

fn optimized_clauses(query: &str) -> Vec<Clause> {
    let mut parsed = parse_cypher(query).unwrap();
    let graph = DirGraph::new();
    let params = HashMap::new();
    optimize(&mut parsed, &graph, &params);
    parsed.clauses
}

fn node_scan_top_k_keys(query: &str) -> Option<Vec<FusedSortKey>> {
    optimized_clauses(query).into_iter().find_map(|c| match c {
        Clause::FusedNodeScanTopK { sort_keys, .. } => Some(sort_keys),
        _ => None,
    })
}

fn order_by_top_k_keys(query: &str) -> Option<Vec<FusedSortKey>> {
    optimized_clauses(query).into_iter().find_map(|c| match c {
        Clause::FusedOrderByTopK { sort_keys, .. } => Some(sort_keys),
        _ => None,
    })
}

#[test]
fn test_node_scan_top_k_fuses_multi_key_order_by() {
    // Before 0.15.14 the pass required exactly one ORDER BY item, so this
    // shape fell through to a full sort of every matching node.
    let keys = node_scan_top_k_keys(
        "MATCH (n:Item) RETURN n.title AS t ORDER BY n.p0 DESC, n.p1 ASC, n.p2 DESC LIMIT 10",
    )
    .expect("multi-key ORDER BY + LIMIT must fuse into FusedNodeScanTopK");
    assert_eq!(keys.len(), 3, "every ORDER BY item becomes a sort key");
    let directions: Vec<bool> = keys.iter().map(|k| k.ascending).collect();
    assert_eq!(
        directions,
        vec![false, true, false],
        "each key keeps its own direction"
    );
    let nulls: Vec<NullsPlacement> = keys.iter().map(|k| k.nulls).collect();
    assert_eq!(
        nulls,
        vec![
            NullsPlacement::First,
            NullsPlacement::Last,
            NullsPlacement::First
        ],
        "each key resolves its own default NULLS placement (DESC → First)"
    );
}

#[test]
fn test_top_k_keys_keep_explicit_nulls_placement() {
    let keys =
        node_scan_top_k_keys("MATCH (n:Item) RETURN n.title AS t ORDER BY n.p0 DESC NULLS LAST, n.p1 ASC NULLS FIRST LIMIT 5")
            .expect("explicit NULLS modifiers must still fuse");
    assert_eq!(
        keys.iter().map(|k| k.nulls).collect::<Vec<_>>(),
        vec![NullsPlacement::Last, NullsPlacement::First],
        "an explicit NULLS modifier overrides the direction default"
    );
}

#[test]
fn test_top_k_sort_key_written_as_a_return_alias_resolves_to_its_expression() {
    let keys = node_scan_top_k_keys(
        "MATCH (n:Item) RETURN n.p0 AS a, n.p1 AS b ORDER BY a, b DESC LIMIT 5",
    )
    .expect("ORDER BY over RETURN aliases must fuse");
    assert_eq!(keys.len(), 2);
    for (i, key) in keys.iter().enumerate() {
        assert!(
            matches!(&key.expression, Expression::PropertyAccess { .. }),
            "alias key {i} must be rewritten to the RETURN item's expression, \
             which is what the pre-projection scan can evaluate"
        );
        assert_eq!(
            key.return_item,
            Some(i),
            "the key remembers the RETURN item it projects"
        );
    }
}

#[test]
fn test_top_k_bails_when_a_sort_key_reads_an_alias_it_is_not_equal_to() {
    // `a` is only bound after projection, so `a + 1` evaluates to NULL in the
    // fused scan's row scope — fusing this returned zero rows before 0.15.14.
    assert!(
        node_scan_top_k_keys("MATCH (n:Item) RETURN n.p0 AS a ORDER BY a + 1 LIMIT 5").is_none(),
        "a computed expression over a RETURN alias must not fuse"
    );
    assert!(
        order_by_top_k_keys("MATCH (n:Item) RETURN n.p0 AS a ORDER BY a + 1 LIMIT 5").is_none(),
        "the generic pass must bail on the same shape"
    );
    // A RETURN item that is itself a reference to a sibling alias of the same
    // RETURN is unevaluable too — `x` only exists after this projection runs.
    assert!(
        order_by_top_k_keys("MATCH (n:Item) RETURN n.p0 AS x, x AS y ORDER BY y LIMIT 5").is_none(),
        "a matched RETURN item whose expression reads a sibling alias must bail"
    );
    // But an alias bound *upstream* by WITH is a real binding on the row, so
    // that shape stays fusable.
    assert!(
        order_by_top_k_keys("MATCH (n:Item) WITH n.p0 AS x RETURN x AS y ORDER BY y LIMIT 5")
            .is_some(),
        "an upstream WITH alias is bound before RETURN and must still fuse"
    );
}

#[test]
fn test_generic_top_k_fuses_multi_key_order_by() {
    let keys = order_by_top_k_keys(
        "MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN a.name AS n, b.age AS age \
         ORDER BY b.age DESC, a.name ASC LIMIT 10",
    )
    .expect("multi-key ORDER BY + LIMIT must fuse into FusedOrderByTopK");
    assert_eq!(keys.len(), 2);
    assert_eq!(
        keys.iter().map(|k| k.ascending).collect::<Vec<_>>(),
        vec![false, true],
        "mixed directions survive the rewrite"
    );
    // Written as properties rather than as the RETURN aliases, so they are
    // their own expressions and project nothing.
    assert!(keys.iter().all(|k| k.return_item.is_none()));

    let aliased = order_by_top_k_keys(
        "MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN a.name AS n, b.age AS age \
         ORDER BY age DESC, n ASC LIMIT 10",
    )
    .expect("the same shape written over RETURN aliases must fuse too");
    assert_eq!(
        aliased.iter().map(|k| k.return_item).collect::<Vec<_>>(),
        vec![Some(1), Some(0)],
        "each alias key remembers the RETURN item it projects"
    );
}

#[test]
fn test_top_k_still_bails_on_a_non_literal_limit() {
    assert!(
        node_scan_top_k_keys("MATCH (n:Item) RETURN n.title AS t ORDER BY n.p0, n.p1 LIMIT 1 + 1")
            .is_none(),
        "LIMIT must be a positive integer literal"
    );
}

// ── anchor_element_id ──────────────────────────────────────────────────────

/// The clause's resolved slot anchors after a full optimizer run.
fn anchors_of(query: &str, params: &HashMap<String, Value>) -> Vec<(String, usize)> {
    let mut parsed = parse_cypher(query).unwrap();
    let graph = DirGraph::new();
    optimize(&mut parsed, &graph, params);
    parsed
        .clauses
        .iter()
        .filter_map(|c| match c {
            Clause::Match(m) | Clause::OptionalMatch(m) => Some(&m.node_anchors),
            _ => None,
        })
        .flatten()
        .map(|(v, idx)| (v.clone(), idx.index()))
        .collect()
}

#[test]
fn test_element_id_anchor_literal_and_param_agree() {
    let no_params = HashMap::new();
    let params: HashMap<String, Value> =
        HashMap::from([("eid".to_string(), Value::String("7".into()))]);

    let literal = anchors_of("MATCH (v) WHERE elementId(v) = '7' RETURN v", &no_params);
    assert_eq!(literal, vec![("v".to_string(), 7)]);

    // The spelling a client actually sends — the round-tripped element_id as a
    // bound parameter — must resolve to the same anchor as the literal.
    assert_eq!(
        anchors_of("MATCH (v) WHERE elementId(v) = $eid RETURN v", &params),
        literal
    );
    // Commuted operands, and the integer spelling of the same slot.
    assert_eq!(
        anchors_of("MATCH (v) WHERE $eid = elementId(v) RETURN v", &params),
        literal
    );
    assert_eq!(
        anchors_of("MATCH (v) WHERE elementId(v) = 7 RETURN v", &no_params),
        literal
    );
}

#[test]
fn test_element_id_anchor_bails_on_non_conjunctive_and_unusable_values() {
    let no_params = HashMap::new();
    let params: HashMap<String, Value> =
        HashMap::from([("eid".to_string(), Value::String("7".into()))]);

    // A disjunct constrains nothing: every node is still a candidate.
    assert!(anchors_of(
        "MATCH (v) WHERE elementId(v) = $eid OR v.name = 'x' RETURN v",
        &params
    )
    .is_empty());
    assert!(anchors_of("MATCH (v) WHERE NOT elementId(v) = $eid RETURN v", &params).is_empty());
    // Not a slot: a name, a negative number, an unbound parameter.
    assert!(anchors_of("MATCH (v) WHERE elementId(v) = 'abc' RETURN v", &no_params).is_empty());
    assert!(anchors_of("MATCH (v) WHERE elementId(v) = -3 RETURN v", &no_params).is_empty());
    assert!(anchors_of("MATCH (v) WHERE elementId(v) = $eid RETURN v", &no_params).is_empty());
    // A variable this MATCH does not bind belongs to another clause's search
    // space, so the anchor is not this clause's to record.
    assert!(anchors_of(
        "MATCH (a) MATCH (b) WHERE elementId(a) = $eid RETURN b",
        &params
    )
    .is_empty());
}

#[test]
fn test_element_id_anchor_reads_a_conjunct_and_the_scoped_optional_where() {
    let params: HashMap<String, Value> =
        HashMap::from([("eid".to_string(), Value::String("2".into()))]);

    assert_eq!(
        anchors_of(
            "MATCH (v) WHERE v.name = 'x' AND elementId(v) = $eid RETURN v",
            &params
        ),
        vec![("v".to_string(), 2)],
        "the AND spine is descended"
    );
    assert_eq!(
        anchors_of(
            "MATCH (a:Person) OPTIONAL MATCH (v) WHERE elementId(v) = $eid RETURN v",
            &params
        ),
        vec![("v".to_string(), 2)],
        "OPTIONAL MATCH carries its WHERE inside the clause"
    );
}

#[test]
fn test_count_distinct_edge_var_is_not_fused() {
    // The fused DISTINCT path counts distinct *peer NodeIndices*, which is not
    // edge identity: two parallel a→b edges make `count(DISTINCT r)` 2 and the
    // peer-dedup answer 1. Both fusion entry points must decline the shape.
    let shapes = [
        "MATCH (a:N)-[r:R]->(b:N) RETURN a, count(DISTINCT r) AS c",
        "MATCH (a:N)-[r:R]->(b:N) WITH a, count(DISTINCT r) AS c RETURN a, c",
        // Anonymous other endpoint: the edge variable is the only non-group
        // variable in scope, so this shape reaches the same gate.
        "MATCH (a:N)<-[r:R]-() RETURN a, count(DISTINCT r) AS c",
    ];
    let graph = DirGraph::new();
    let params = HashMap::new();
    for source in shapes {
        let mut query = parse_cypher(source).unwrap();
        optimize(&mut query, &graph, &params);
        assert!(
            !query.clauses.iter().any(|clause| matches!(
                clause,
                Clause::FusedMatchReturnAggregate { .. } | Clause::FusedMatchWithAggregate { .. }
            )),
            "count(DISTINCT <edge var>) must not fuse to a distinct-peer count: {source}"
        );
    }
}

#[test]
fn test_count_of_edge_var_without_distinct_still_fuses() {
    // Control for `test_count_distinct_edge_var_is_not_fused`: the non-DISTINCT
    // edge count is what the fused edge-centric path actually computes.
    let mut query = parse_cypher("MATCH (a:N)-[r:R]->(b:N) RETURN a, count(r) AS c").unwrap();
    let graph = DirGraph::new();
    let params = HashMap::new();
    optimize(&mut query, &graph, &params);
    assert!(
        query.clauses.iter().any(|clause| matches!(
            clause,
            Clause::FusedMatchReturnAggregate {
                distinct_count: false,
                ..
            }
        )),
        "plain count(<edge var>) must keep fusing: {:#?}",
        query.clauses
    );
}

#[test]
fn test_push_limit_into_aggregate_bails_on_with_inline_filter() {
    // `execute_with` (and the streaming pipeline) project first and filter
    // after, so a capped group set drops groups the filter would have kept and
    // the LIMIT still had room for.
    let filtered = [
        "MATCH (n:T) WITH n.k AS k, collect(n.id) AS ids WHERE size(ids) > 1 LIMIT 5 RETURN k, ids",
        "MATCH (n:T) WITH n.k AS k, collect(n.id) AS ids HAVING size(ids) > 1 LIMIT 5 RETURN k, ids",
    ];
    let graph = DirGraph::new();
    let params = HashMap::new();
    for source in filtered {
        let mut query = parse_cypher(source).unwrap();
        optimize(&mut query, &graph, &params);
        for clause in &query.clauses {
            if let Clause::With(w) = clause {
                assert_eq!(
                    w.group_limit_hint, None,
                    "a filtered WITH must not carry a group cap: {source}"
                );
            }
        }
    }

    // Control: the same shape without the filter still gets the hint.
    let mut query =
        parse_cypher("MATCH (n:T) WITH n.k AS k, collect(n.id) AS ids LIMIT 5 RETURN k, ids")
            .unwrap();
    optimize(&mut query, &graph, &params);
    let hinted = query
        .clauses
        .iter()
        .any(|clause| matches!(clause, Clause::With(w) if w.group_limit_hint == Some(5)));
    assert!(hinted, "unfiltered WITH + LIMIT must still be hinted");
}
