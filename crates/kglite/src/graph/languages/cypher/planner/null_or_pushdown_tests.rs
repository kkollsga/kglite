//! Plan-shape goldens for pushing `x.p IS NULL OR x.p <op> c` and
//! `coalesce(x.p, c0) <op> c` into MATCH: the node matcher and the
//! relationship filter each must carry the exact predicate or nothing.

use super::*;
use crate::graph::core::pattern_matching::pattern::{PropOp, RelEdgePredicate};
use crate::graph::core::pattern_matching::{PatternElement, PropertyMatcher};
use crate::graph::languages::cypher::parser::parse_cypher;

fn planned(cypher: &str, params: &HashMap<String, Value>) -> CypherQuery {
    let mut query = parse_cypher(cypher).unwrap();
    optimize(&mut query, &DirGraph::new(), params);
    query
}

fn node_matcher(query: &CypherQuery, prop: &str) -> Option<PropertyMatcher> {
    query.clauses.iter().find_map(|clause| match clause {
        Clause::Match(m) => m.patterns[0].elements.iter().find_map(|el| match el {
            PatternElement::Node(np) => np.properties.as_ref()?.get(prop).cloned(),
            _ => None,
        }),
        _ => None,
    })
}

fn edge_filter(query: &CypherQuery) -> Option<RelEdgePredicate> {
    query.clauses.iter().find_map(|clause| match clause {
        Clause::Match(m) => m.patterns[0].elements.iter().find_map(|el| match el {
            PatternElement::Edge(edge) => edge.edge_filter.as_ref().map(|f| f.predicate.clone()),
            _ => None,
        }),
        _ => None,
    })
}

fn has_where(query: &CypherQuery) -> bool {
    query
        .clauses
        .iter()
        .any(|clause| matches!(clause, Clause::Where(_)))
}

fn is_null_or_ge(matcher: &Option<PropertyMatcher>, bound: i64) -> bool {
    matches!(
        matcher,
        Some(PropertyMatcher::NullOr(inner))
            if matches!(**inner, PropertyMatcher::GreaterOrEqual(Value::Int64(v)) if v == bound)
    )
}

#[test]
fn node_null_or_comparison_pushes_in_either_operand_order() {
    let t = HashMap::from([("t".to_string(), Value::Int64(150))]);
    for cypher in [
        "MATCH (n:S) WHERE n.vf <= 150 AND (n.vt IS NULL OR n.vt >= 150) RETURN n",
        "MATCH (n:S) WHERE n.vf <= 150 AND (150 <= n.vt OR n.vt IS NULL) RETURN n",
        "MATCH (n:S) WHERE n.vf <= 150 AND (n.vt IS NULL OR n.vt >= $t) RETURN n",
        "MATCH (n:S) WHERE n.vf <= 150 AND ($t <= n.vt OR n.vt IS NULL) RETURN n",
    ] {
        let query = planned(cypher, &t);
        assert!(
            is_null_or_ge(&node_matcher(&query, "vt"), 150),
            "{cypher}: {:?}",
            node_matcher(&query, "vt")
        );
        assert!(matches!(
            node_matcher(&query, "vf"),
            Some(PropertyMatcher::LessOrEqual(Value::Int64(150)))
        ));
    }
}

#[test]
fn node_null_or_with_unresolvable_value_is_not_pushed() {
    for params in [
        HashMap::new(),
        HashMap::from([("t".to_string(), Value::Null)]),
    ] {
        for cypher in [
            "MATCH (n:S) WHERE n.vf <= 150 AND (n.vt IS NULL OR n.vt >= $t) RETURN n",
            "MATCH (n:S) WHERE n.vf <= 150 AND coalesce(n.vt, 1000) >= $t RETURN n",
            "MATCH (n:S) WHERE n.vf <= 150 AND (n.vt IS NULL OR n.vt >= null) RETURN n",
        ] {
            let query = planned(cypher, &params);
            assert!(node_matcher(&query, "vt").is_none(), "{cypher} {params:?}");
            assert!(has_where(&query));
        }
    }
}

#[test]
fn node_coalesce_comparison_folds_its_default_at_plan_time() {
    // (query, true = `NullOr(>= 150)` / false = the plain comparison)
    for (cypher, null_or) in [
        (
            "MATCH (n:S) WHERE coalesce(n.vt, 1000) >= 150 RETURN n",
            true,
        ),
        (
            "MATCH (n:S) WHERE 150 <= coalesce(n.vt, 1000) RETURN n",
            true,
        ),
        ("MATCH (n:S) WHERE coalesce(n.vt, 0) >= 150 RETURN n", false),
        // An incomparable default makes the fold NULL, never true.
        (
            "MATCH (n:S) WHERE coalesce(n.vt, 'x') >= 150 RETURN n",
            false,
        ),
    ] {
        let got = node_matcher(&planned(cypher, &HashMap::new()), "vt");
        let expected = if null_or {
            is_null_or_ge(&got, 150)
        } else {
            matches!(
                got,
                Some(PropertyMatcher::GreaterOrEqual(Value::Int64(150)))
            )
        };
        assert!(expected, "{cypher}: {got:?}");
    }
    // A NULL default makes the fold NULL, never true: plain comparison.
    let got = node_matcher(
        &planned(
            "MATCH (n:S) WHERE coalesce(n.vt, null) <= 150 RETURN n",
            &HashMap::new(),
        ),
        "vt",
    );
    assert!(
        matches!(got, Some(PropertyMatcher::LessOrEqual(Value::Int64(150)))),
        "{got:?}"
    );
}

#[test]
fn node_shapes_that_are_not_the_null_or_rule_stay_in_where() {
    for cypher in [
        "MATCH (n:S) WHERE n.vt IS NOT NULL OR n.vt >= 150 RETURN n",
        "MATCH (n:S) WHERE n.vt IS NULL OR n.vf >= 150 RETURN n",
        "MATCH (n:S), (m:S) WHERE n.vt IS NULL OR m.vt >= 150 RETURN n",
        "MATCH (n:S) WHERE NOT (n.vt IS NULL OR n.vt >= 150) RETURN n",
        "MATCH (n:S) WHERE n.vt IS NULL OR n.vt = 150 RETURN n",
        "MATCH (n:S) WHERE n.vt IS NULL OR n.vt <> 150 RETURN n",
        "MATCH (n:S) WHERE coalesce(n.vt, n.vf, 0) >= 150 RETURN n",
        "MATCH (n:S) WHERE coalesce(n.vt, n.vf) >= 150 RETURN n",
        "MATCH (n:S) WHERE coalesce(n.vt, 1000) = 150 RETURN n",
    ] {
        let query = planned(cypher, &HashMap::new());
        assert!(
            node_matcher(&query, "vt").is_none(),
            "{cypher}: {:?}",
            node_matcher(&query, "vt")
        );
        assert!(has_where(&query), "{cypher}");
    }
}

#[test]
fn node_null_or_never_shares_a_property_with_another_matcher() {
    for cypher in [
        "MATCH (n:S) WHERE n.vt <= 300 AND (n.vt IS NULL OR n.vt >= 150) RETURN n",
        "MATCH (n:S) WHERE (n.vt IS NULL OR n.vt >= 150) AND n.vt <= 300 RETURN n",
        "MATCH (n:S {vt: 5}) WHERE n.vt IS NULL OR n.vt >= 150 RETURN n",
    ] {
        let query = planned(cypher, &HashMap::new());
        let Some(Clause::Where(w)) = query.clauses.iter().find(|c| matches!(c, Clause::Where(_)))
        else {
            panic!("{cypher}: the losing conjunct must stay in WHERE");
        };
        let residual = format!("{:?}", w.predicate);
        let expected_in_where = match node_matcher(&query, "vt") {
            Some(PropertyMatcher::NullOr(_)) => "LessThanEq",
            _ => "IsNull",
        };
        assert!(residual.contains(expected_in_where), "{cypher}: {residual}");
    }
}

#[test]
fn null_or_filter_does_not_reduce_the_start_node_estimate() {
    let query = parse_cypher("MATCH (n:Item) RETURN n").unwrap();
    let Clause::Match(match_clause) = &query.clauses[0] else {
        panic!("expected MATCH clause");
    };
    let PatternElement::Node(node) = &match_clause.patterns[0].elements[0] else {
        panic!("expected node pattern");
    };
    let mut node = node.clone();
    node.properties = Some(HashMap::from([(
        "vt".to_string(),
        PropertyMatcher::NullOr(Box::new(PropertyMatcher::GreaterOrEqual(Value::Int64(1)))),
    )]));
    let mut graph = DirGraph::new();
    graph
        .type_indices
        .entry_or_default("Item".to_string())
        .extend((0..100).map(petgraph::graph::NodeIndex::new));
    assert_eq!(join_order::estimate_node_selectivity(&node, &graph), 100);
}

fn is_rel_null_or_ge(pred: &RelEdgePredicate, bound: i64) -> bool {
    let RelEdgePredicate::Or(items) = pred else {
        return false;
    };
    items.len() == 2
        && items
            .iter()
            .any(|p| matches!(p, RelEdgePredicate::PropertyIsNull { prop } if prop == "vt"))
        && items.iter().any(|p| {
            matches!(
                p,
                RelEdgePredicate::Property { prop, op: PropOp::Ge, value: Value::Int64(v) }
                    if prop == "vt" && *v == bound
            )
        })
}

#[test]
fn relationship_null_or_comparison_is_consumed_by_the_edge_filter() {
    for cypher in [
        "MATCH (a:S)-[r:R]->(b) WHERE r.vf <= 150 AND (r.vt IS NULL OR r.vt >= 150) RETURN b",
        "MATCH (a:S)-[r:R]->(b) WHERE r.vf <= 150 AND (150 <= r.vt OR r.vt IS NULL) RETURN b",
        "MATCH (a:S)-[r:R]->(b) WHERE r.vf <= 150 AND coalesce(r.vt, 1000) >= 150 RETURN b",
    ] {
        let query = planned(cypher, &HashMap::new());
        let Some(RelEdgePredicate::And(items)) = edge_filter(&query) else {
            panic!("{cypher}: {:?}", edge_filter(&query));
        };
        assert_eq!(items.len(), 2, "{cypher}: {items:?}");
        assert!(
            items.iter().any(|p| is_rel_null_or_ge(p, 150)),
            "{cypher}: {items:?}"
        );
        assert!(!has_where(&query), "{cypher}: the WHERE is fully consumed");
    }
}

#[test]
fn relationship_is_null_leaves_push_on_their_own() {
    let query = planned(
        "MATCH (a:S)-[r:R]->(b) WHERE r.vt IS NULL RETURN b",
        &HashMap::new(),
    );
    assert!(matches!(
        edge_filter(&query),
        Some(RelEdgePredicate::PropertyIsNull { prop }) if prop == "vt"
    ));
    assert!(!has_where(&query));

    let query = planned(
        "MATCH (a:S)-[r:R]->(b) WHERE r.vt IS NOT NULL RETURN b",
        &HashMap::new(),
    );
    assert!(matches!(
        edge_filter(&query),
        Some(RelEdgePredicate::Not(inner))
            if matches!(&*inner, RelEdgePredicate::PropertyIsNull { prop } if prop == "vt")
    ));
    assert!(!has_where(&query));
}

#[test]
fn relationship_coalesce_folds_keep_their_null_row_answer() {
    // False fold: a NULL row is `false`, not unknown, so `NOT` keeps it.
    let query = planned(
        "MATCH (a:S)-[r:R]->(b) WHERE coalesce(r.vt, 0) >= 150 RETURN b",
        &HashMap::new(),
    );
    let Some(RelEdgePredicate::And(items)) = edge_filter(&query) else {
        panic!("{:?}", edge_filter(&query));
    };
    assert!(items.iter().any(|p| matches!(
        p,
        RelEdgePredicate::Not(inner)
            if matches!(&**inner, RelEdgePredicate::PropertyIsNull { prop } if prop == "vt")
    )));
    assert!(items
        .iter()
        .any(|p| matches!(p, RelEdgePredicate::Property { op: PropOp::Ge, .. })));

    // NULL fold: unknown for a NULL row, exactly as the plain comparison.
    let query = planned(
        "MATCH (a:S)-[r:R]->(b) WHERE coalesce(r.vt, null) <= 150 RETURN b",
        &HashMap::new(),
    );
    assert!(matches!(
        edge_filter(&query),
        Some(RelEdgePredicate::Property { op: PropOp::Le, .. })
    ));

    // Not the shape: stays in WHERE.
    for cypher in [
        "MATCH (a:S)-[r:R]->(b) WHERE coalesce(r.vt, r.vf) >= 150 RETURN b",
        "MATCH (a:S)-[r:R]->(b) WHERE coalesce(r.vt, 1000) >= $missing RETURN b",
    ] {
        let query = planned(cypher, &HashMap::new());
        assert!(edge_filter(&query).is_none(), "{cypher}");
        assert!(has_where(&query), "{cypher}");
    }
}
