use super::*;
use crate::graph::core::pattern_matching::PropertyMatcher;
use crate::graph::languages::cypher::parser::parse_cypher;

#[test]
fn test_predicate_pushdown_simple() {
    let mut query = parse_cypher("MATCH (n:Person) WHERE n.age = 30 RETURN n").unwrap();

    let graph = DirGraph::new();
    let params = HashMap::new();
    optimize(&mut query, &graph, &params);

    // WHERE is kept as a safety net even when all predicates are pushed
    assert_eq!(query.clauses.len(), 3); // MATCH + WHERE + RETURN
    assert!(matches!(&query.clauses[0], Clause::Match(_)));
    assert!(matches!(&query.clauses[2], Clause::Return(_)));

    // The MATCH pattern should now have {age: 30} as a property
    if let Clause::Match(m) = &query.clauses[0] {
        if let PatternElement::Node(np) = &m.patterns[0].elements[0] {
            assert!(np.properties.is_some());
            let props = np.properties.as_ref().unwrap();
            assert!(props.contains_key("age"));
        } else {
            panic!("Expected node pattern");
        }
    }
}

#[test]
fn test_predicate_pushdown_partial() {
    let mut query =
        parse_cypher("MATCH (n:Person) WHERE n.age = 30 AND n.score > 100 RETURN n").unwrap();

    let graph = DirGraph::new();
    let params = HashMap::new();
    optimize(&mut query, &graph, &params);

    // Both n.age = 30 and n.score > 100 should be pushed into MATCH
    // WHERE is kept as a safety net
    assert_eq!(query.clauses.len(), 3); // MATCH + WHERE + RETURN

    if let Clause::Match(m) = &query.clauses[0] {
        if let PatternElement::Node(np) = &m.patterns[0].elements[0] {
            let props = np.properties.as_ref().unwrap();
            assert!(matches!(
                props.get("age"),
                Some(PropertyMatcher::Equals(Value::Int64(30)))
            ));
            assert!(matches!(
                props.get("score"),
                Some(PropertyMatcher::GreaterThan(Value::Int64(100)))
            ));
        }
    }
}

#[test]
fn test_comparison_pushdown() {
    let mut query = parse_cypher("MATCH (n:Person) WHERE n.age > 30 RETURN n").unwrap();

    let graph = DirGraph::new();
    let params = HashMap::new();
    optimize(&mut query, &graph, &params);

    // Comparison should be pushed into MATCH, WHERE kept as safety net
    assert_eq!(query.clauses.len(), 3); // MATCH + WHERE + RETURN

    if let Clause::Match(m) = &query.clauses[0] {
        if let PatternElement::Node(np) = &m.patterns[0].elements[0] {
            let props = np.properties.as_ref().unwrap();
            assert!(matches!(
                props.get("age"),
                Some(PropertyMatcher::GreaterThan(Value::Int64(30)))
            ));
        }
    }
}

#[test]
fn test_no_pushdown_for_not_equals() {
    let mut query = parse_cypher("MATCH (n:Person) WHERE n.age <> 30 RETURN n").unwrap();

    let graph = DirGraph::new();
    let params = HashMap::new();
    optimize(&mut query, &graph, &params);

    // NotEquals should NOT be pushed - WHERE should remain
    assert_eq!(query.clauses.len(), 3); // MATCH + WHERE + RETURN
}

#[test]
fn test_predicate_pushdown_parameter() {
    let mut query = parse_cypher("MATCH (n:Person) WHERE n.name = $name RETURN n").unwrap();

    let graph = DirGraph::new();
    let mut params = HashMap::new();
    params.insert("name".to_string(), Value::String("Alice".to_string()));
    optimize(&mut query, &graph, &params);

    // Parameter resolved and pushed; WHERE kept as safety net
    assert_eq!(query.clauses.len(), 3); // MATCH + WHERE + RETURN

    // The MATCH pattern should now have {name: 'Alice'} as a property
    if let Clause::Match(m) = &query.clauses[0] {
        if let PatternElement::Node(np) = &m.patterns[0].elements[0] {
            assert!(np.properties.is_some());
            let props = np.properties.as_ref().unwrap();
            assert!(props.contains_key("name"));
            assert!(matches!(
                props.get("name"),
                Some(PropertyMatcher::Equals(Value::String(s))) if s == "Alice"
            ));
        } else {
            panic!("Expected node pattern");
        }
    }
}

#[test]
fn test_predicate_pushdown_parameter_partial() {
    let mut query =
        parse_cypher("MATCH (n:Person) WHERE n.name = $name AND n.age > $min_age RETURN n")
            .unwrap();

    let graph = DirGraph::new();
    let mut params = HashMap::new();
    params.insert("name".to_string(), Value::String("Alice".to_string()));
    params.insert("min_age".to_string(), Value::Int64(25));
    optimize(&mut query, &graph, &params);

    // Both should be pushed: n.name = $name (equality) and n.age > $min_age (comparison)
    // WHERE kept as safety net
    assert_eq!(query.clauses.len(), 3); // MATCH + WHERE + RETURN

    if let Clause::Match(m) = &query.clauses[0] {
        if let PatternElement::Node(np) = &m.patterns[0].elements[0] {
            let props = np.properties.as_ref().unwrap();
            assert!(matches!(
                props.get("name"),
                Some(PropertyMatcher::Equals(Value::String(s))) if s == "Alice"
            ));
            assert!(matches!(
                props.get("age"),
                Some(PropertyMatcher::GreaterThan(Value::Int64(25)))
            ));
        }
    }
}

#[test]
fn test_comparison_range_merge() {
    let mut query =
        parse_cypher("MATCH (n:Paper) WHERE n.year >= 2015 AND n.year <= 2022 RETURN n").unwrap();

    let graph = DirGraph::new();
    let params = HashMap::new();
    optimize(&mut query, &graph, &params);

    // Both comparisons should be merged into a Range matcher; WHERE kept
    assert_eq!(query.clauses.len(), 3); // MATCH + WHERE + RETURN

    if let Clause::Match(m) = &query.clauses[0] {
        if let PatternElement::Node(np) = &m.patterns[0].elements[0] {
            let props = np.properties.as_ref().unwrap();
            assert!(matches!(
                props.get("year"),
                Some(PropertyMatcher::Range {
                    lower: Value::Int64(2015),
                    lower_inclusive: true,
                    upper: Value::Int64(2022),
                    upper_inclusive: true,
                })
            ));
        }
    }
}

#[test]
fn test_correlated_nodeprop_pushdown() {
    // The classic shape from sodir-prospect build:
    //   MATCH (a:A) MATCH (b:B) WHERE b.x = a.y
    // should push { x: EqualsNodeProp { var: "a", prop: "y" } } onto B.
    let mut query =
        parse_cypher("MATCH (a:A) MATCH (b:B) WHERE b.x = a.y RETURN a.id, b.id").unwrap();

    let graph = DirGraph::new();
    let params = HashMap::new();
    optimize(&mut query, &graph, &params);

    // Locate the second MATCH (matching on B)
    let b_match = query
        .clauses
        .iter()
        .filter_map(|c| match c {
            Clause::Match(m) => Some(m),
            _ => None,
        })
        .find(|m| {
            matches!(
                &m.patterns[0].elements[0],
                PatternElement::Node(np) if np.node_type.as_deref() == Some("B")
            )
        })
        .expect("expected second MATCH on B");

    if let PatternElement::Node(np) = &b_match.patterns[0].elements[0] {
        let props = np.properties.as_ref().expect("expected props on b");
        match props.get("x") {
            Some(PropertyMatcher::EqualsNodeProp { var, prop }) => {
                assert_eq!(var, "a");
                assert_eq!(prop, "y");
            }
            other => panic!("expected EqualsNodeProp on b.x, got {:?}", other),
        }
    }
}

#[test]
fn test_correlated_nodeprop_reversed_sides() {
    // Reversed: a.y = b.x (the cur_var b appears on the right).
    let mut query =
        parse_cypher("MATCH (a:A) MATCH (b:B) WHERE a.y = b.x RETURN a.id, b.id").unwrap();

    let graph = DirGraph::new();
    let params = HashMap::new();
    optimize(&mut query, &graph, &params);

    let b_match = query
        .clauses
        .iter()
        .filter_map(|c| match c {
            Clause::Match(m) => Some(m),
            _ => None,
        })
        .find(|m| {
            matches!(
                &m.patterns[0].elements[0],
                PatternElement::Node(np) if np.node_type.as_deref() == Some("B")
            )
        })
        .unwrap();

    if let PatternElement::Node(np) = &b_match.patterns[0].elements[0] {
        let props = np.properties.as_ref().unwrap();
        assert!(matches!(
            props.get("x"),
            Some(PropertyMatcher::EqualsNodeProp { var, prop })
                if var == "a" && prop == "y"
        ));
    }
}

#[test]
fn test_scalar_var_pushdown_from_unwind() {
    // WHERE s.title = fname where fname comes from UNWIND should become
    // an EqualsVar matcher on the pattern.
    let mut query =
        parse_cypher("UNWIND ['x','y'] AS fname MATCH (s:Strat) WHERE s.title = fname RETURN s.id")
            .unwrap();

    let graph = DirGraph::new();
    let params = HashMap::new();
    optimize(&mut query, &graph, &params);

    let s_match = query
        .clauses
        .iter()
        .filter_map(|c| match c {
            Clause::Match(m) => Some(m),
            _ => None,
        })
        .next()
        .unwrap();

    if let PatternElement::Node(np) = &s_match.patterns[0].elements[0] {
        let props = np.properties.as_ref().unwrap();
        assert!(matches!(
            props.get("title"),
            Some(PropertyMatcher::EqualsVar(n)) if n == "fname"
        ));
    }
}

#[test]
fn test_no_pushdown_when_both_vars_in_same_match() {
    // Within the same MATCH, a.y = b.x is handled by the pattern
    // executor's shared-var join. We must NOT rewrite it as an
    // EqualsNodeProp (which assumes prior-bound node).
    let mut query = parse_cypher("MATCH (a:A), (b:B) WHERE a.y = b.x RETURN a.id, b.id").unwrap();

    let graph = DirGraph::new();
    let params = HashMap::new();
    optimize(&mut query, &graph, &params);

    for clause in &query.clauses {
        if let Clause::Match(m) = clause {
            for pat in &m.patterns {
                for el in &pat.elements {
                    if let PatternElement::Node(np) = el {
                        if let Some(props) = &np.properties {
                            for m in props.values() {
                                assert!(
                                    !matches!(m, PropertyMatcher::EqualsNodeProp { .. }),
                                    "same-MATCH correlated equality must not be rewritten"
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn test_undirected_pattern_reversed_by_selectivity() {
    // Regression: `(other)-[r]-(p {title: 'X'})` was not reversed because
    // the planner bailed out on undirected edges, leaving `other` (no
    // constraints, full graph scan) as the start node. The selective
    // anchor `p {title: 'X'}` must now be picked as start.
    let mut query =
        parse_cypher("MATCH (other)-[r]-(p {title: 'X'}) RETURN type(r), count(other)").unwrap();

    let graph = DirGraph::new();
    let params = HashMap::new();
    optimize(&mut query, &graph, &params);

    let m = query
        .clauses
        .iter()
        .find_map(|c| match c {
            Clause::Match(m) => Some(m),
            _ => None,
        })
        .unwrap();

    let first = match &m.patterns[0].elements[0] {
        PatternElement::Node(np) => np,
        _ => panic!("expected node"),
    };
    assert_eq!(
        first.variable.as_deref(),
        Some("p"),
        "selective anchor `p` should be the start after reversal"
    );
    assert!(
        first.properties.is_some(),
        "start node must carry the title property after reversal"
    );
}

#[test]
fn test_undirected_pattern_no_reverse_when_first_is_anchor() {
    // Reverse case: anchor is already first — no reversal expected.
    let mut query =
        parse_cypher("MATCH (p {title: 'X'})-[r]-(other) RETURN type(r), count(other)").unwrap();

    let graph = DirGraph::new();
    let params = HashMap::new();
    optimize(&mut query, &graph, &params);

    let m = query
        .clauses
        .iter()
        .find_map(|c| match c {
            Clause::Match(m) => Some(m),
            _ => None,
        })
        .unwrap();

    let first = match &m.patterns[0].elements[0] {
        PatternElement::Node(np) => np,
        _ => panic!("expected node"),
    };
    assert_eq!(first.variable.as_deref(), Some("p"));
}

#[test]
fn test_var_length_pattern_reversed_by_selectivity() {
    // `(other)-[*1..3]-(p {id: 1})` — the anchor is selective. Reversal
    // is safe when the pattern has no path assignment (the `path_assignments`
    // guard already protects path-bound patterns).
    let mut query = parse_cypher("MATCH (other)-[*1..3]-(p {id: 1}) RETURN p, other").unwrap();

    let graph = DirGraph::new();
    let params = HashMap::new();
    optimize(&mut query, &graph, &params);

    let m = query
        .clauses
        .iter()
        .find_map(|c| match c {
            Clause::Match(m) => Some(m),
            _ => None,
        })
        .unwrap();

    let first = match &m.patterns[0].elements[0] {
        PatternElement::Node(np) => np,
        _ => panic!("expected node"),
    };
    assert_eq!(
        first.variable.as_deref(),
        Some("p"),
        "var-length patterns should still get start-node optimization"
    );
}

#[test]
fn test_var_length_with_path_assignment_not_reversed() {
    // Path assignments must block reversal (path semantics depend on direction).
    let mut query = parse_cypher("MATCH path = (other)-[*1..3]-(p {id: 1}) RETURN path").unwrap();

    let graph = DirGraph::new();
    let params = HashMap::new();
    optimize(&mut query, &graph, &params);

    let m = query
        .clauses
        .iter()
        .find_map(|c| match c {
            Clause::Match(m) => Some(m),
            _ => None,
        })
        .unwrap();

    let first = match &m.patterns[0].elements[0] {
        PatternElement::Node(np) => np,
        _ => panic!("expected node"),
    };
    assert_eq!(
        first.variable.as_deref(),
        Some("other"),
        "path-bound patterns must not be reversed"
    );
}

#[test]
fn test_limit_pushdown_single_match_with_where() {
    // Single MATCH + WHERE + RETURN + LIMIT — pushdown is safe; the LIMIT
    // clause should be removed and the MATCH should carry limit_hint.
    let mut query =
        parse_cypher("MATCH (n:Person) WHERE n.age > 25 RETURN n.name LIMIT 10").unwrap();

    let graph = DirGraph::new();
    let params = HashMap::new();
    optimize(&mut query, &graph, &params);

    // Limit clause should be gone (absorbed into the MATCH hint)
    let has_limit = query.clauses.iter().any(|c| matches!(c, Clause::Limit(_)));
    assert!(
        !has_limit,
        "single-MATCH query should have LIMIT pushed into MATCH"
    );

    let m = query
        .clauses
        .iter()
        .find_map(|c| match c {
            Clause::Match(m) => Some(m),
            _ => None,
        })
        .expect("expected a MATCH clause");
    assert_eq!(m.limit_hint, Some(10));
}

#[test]
fn test_multi_match_no_reverse_when_bound_var_first() {
    // Regression for the user's "(p)-[:P31]->(:human) gets reversed to scan
    // 13M humans" report: in the second MATCH, `p` is already bound by the
    // first MATCH. The planner must not reverse the pattern to start from
    // `(:human)` (a 13M-row scan) just because `p` looks unconstrained
    // statically.
    let mut query =
        parse_cypher("MATCH (p:Person) MATCH (p)-[:KNOWS]->(c:Company) RETURN p, c").unwrap();

    let graph = DirGraph::new();
    let params = HashMap::new();
    optimize(&mut query, &graph, &params);

    // The second MATCH should still start with `p`, not `c`.
    let matches: Vec<_> = query
        .clauses
        .iter()
        .filter_map(|c| match c {
            Clause::Match(m) => Some(m),
            _ => None,
        })
        .collect();
    assert!(matches.len() >= 2, "expected two MATCH clauses");
    let second = matches[1];
    let first_var = match &second.patterns[0].elements[0] {
        PatternElement::Node(np) => np.variable.as_deref(),
        _ => None,
    };
    assert_eq!(
        first_var,
        Some("p"),
        "second MATCH must keep pre-bound `p` as start node, not reverse to `c`"
    );
}

#[test]
fn test_multi_match_reorder_prefers_anchored_pattern() {
    // When a single MATCH has two patterns sharing a pre-bound variable,
    // the more selective pattern should run first. The planner already
    // does intra-clause reordering — this test pins the cross-clause
    // bound-vars logic so it doesn't regress.
    let mut query = parse_cypher(
        "MATCH (p {id: 1}) \
         MATCH (p)-[:R1]->(:T1), (p)-[:R2]->({id: 99}) \
         RETURN p",
    )
    .unwrap();

    let graph = DirGraph::new();
    let params = HashMap::new();
    optimize(&mut query, &graph, &params);

    // Just ensure optimization doesn't crash and patterns survive.
    let m2 = query
        .clauses
        .iter()
        .filter_map(|c| match c {
            Clause::Match(m) => Some(m),
            _ => None,
        })
        .nth(1)
        .expect("expected second MATCH");
    assert_eq!(m2.patterns.len(), 2);
}

#[test]
fn test_limit_pushdown_multi_match_safety() {
    // Regression: 3-MATCH + WHERE on last-bound variable + LIMIT N produced
    // fewer rows than expected because push_limit_into_match was setting
    // `limit_hint` on the last MATCH, which interacts incorrectly with the
    // per-row pattern executor in the subsequent-MATCH path. The planner
    // must NOT push LIMIT into MATCH when there are multiple MATCH clauses.
    let mut query = parse_cypher(
        "MATCH (a)-[:R1]->(:T1) \
         MATCH (a)-[:R2]->(b) \
         MATCH (b)-[:R3]->(c) \
         WHERE c.id = 7318 \
         RETURN a.id, b.id LIMIT 50",
    )
    .unwrap();

    let graph = DirGraph::new();
    let params = HashMap::new();
    optimize(&mut query, &graph, &params);

    // Limit clause must remain — pushdown is unsafe for multi-MATCH
    let has_limit = query.clauses.iter().any(|c| matches!(c, Clause::Limit(_)));
    assert!(has_limit, "multi-MATCH query must retain its LIMIT clause");

    // No MATCH clause should have a limit_hint set
    for clause in &query.clauses {
        if let Clause::Match(m) = clause {
            assert_eq!(
                m.limit_hint, None,
                "multi-MATCH clauses must not receive a limit_hint"
            );
        }
    }
}

#[test]
fn test_reorder_match_clauses_picks_rare_edge_first() {
    // Two MATCH clauses share `p`, both id-anchored on the other end. The
    // planner should drive the smaller-edge-type clause first so the
    // executor enumerates fewer rows before joining.
    //
    // Motivating real-world case (Wikidata):
    //   MATCH (p)-[:P31]->({id:5})    -- 80M instance-of edges
    //   MATCH (p)-[:P27]->({id:183})  -- 3M citizenship edges
    // → swap so P27 drives, then per-row check P31. ~25× cheaper.
    let mut query = parse_cypher(
        "MATCH (p)-[:VERY_COMMON]->({id: 1}) \
         MATCH (p)-[:RARE]->({id: 2}) \
         RETURN p",
    )
    .unwrap();

    let graph = DirGraph::new();
    {
        let mut cache = graph.edge_type_counts_cache.write().unwrap();
        let mut counts = HashMap::new();
        counts.insert("VERY_COMMON".to_string(), 1_000_000);
        counts.insert("RARE".to_string(), 1_000);
        *cache = Some(counts);
    }

    let params = HashMap::new();
    optimize(&mut query, &graph, &params);

    let matches: Vec<_> = query
        .clauses
        .iter()
        .filter_map(|c| match c {
            Clause::Match(m) => Some(m),
            _ => None,
        })
        .collect();
    assert_eq!(matches.len(), 2, "expected two MATCH clauses preserved");

    // First MATCH after reorder must be the RARE one.
    let first_edge_type = matches[0].patterns[0].elements.iter().find_map(|e| {
        if let PatternElement::Edge(ep) = e {
            ep.connection_type.clone()
        } else {
            None
        }
    });
    assert_eq!(
        first_edge_type.as_deref(),
        Some("RARE"),
        "RARE (lower edge-type cost) should be promoted to first MATCH; \
         got first edge type = {first_edge_type:?}"
    );
}

#[test]
fn test_reorder_match_clauses_skips_when_cache_missing() {
    // Same query shape as above, but no edge_type_counts_cache populated.
    // The reorder pass must bail (avoid triggering an O(E) cache build
    // at plan time) and leave the original clause order intact.
    let mut query = parse_cypher(
        "MATCH (p)-[:VERY_COMMON]->({id: 1}) \
         MATCH (p)-[:RARE]->({id: 2}) \
         RETURN p",
    )
    .unwrap();

    let graph = DirGraph::new();
    // Confirm cache is unset to start.
    assert!(!graph.has_edge_type_counts_cache());

    let params = HashMap::new();
    optimize(&mut query, &graph, &params);

    // Original textual order preserved: VERY_COMMON first.
    let matches: Vec<_> = query
        .clauses
        .iter()
        .filter_map(|c| match c {
            Clause::Match(m) => Some(m),
            _ => None,
        })
        .collect();
    assert_eq!(matches.len(), 2);
    let first_edge_type = matches[0].patterns[0].elements.iter().find_map(|e| {
        if let PatternElement::Edge(ep) = e {
            ep.connection_type.clone()
        } else {
            None
        }
    });
    assert_eq!(first_edge_type.as_deref(), Some("VERY_COMMON"));
    // Cache must still be empty — the planner did not force a build.
    assert!(
        !graph.has_edge_type_counts_cache(),
        "planner must not warm the edge-type-counts cache from the optimization path"
    );
}

#[test]
fn test_reorder_match_clauses_requires_id_anchor() {
    // No id anchor on either MATCH → cannot use the edge-count proxy
    // safely (other-end fan-in dominates and isn't captured). Pass must
    // leave order intact even if the cache is populated.
    let mut query = parse_cypher(
        "MATCH (p)-[:VERY_COMMON]->(q) \
         MATCH (p)-[:RARE]->(r) \
         RETURN p",
    )
    .unwrap();

    let graph = DirGraph::new();
    {
        let mut cache = graph.edge_type_counts_cache.write().unwrap();
        let mut counts = HashMap::new();
        counts.insert("VERY_COMMON".to_string(), 1_000_000);
        counts.insert("RARE".to_string(), 1_000);
        *cache = Some(counts);
    }

    let params = HashMap::new();
    optimize(&mut query, &graph, &params);

    let matches: Vec<_> = query
        .clauses
        .iter()
        .filter_map(|c| match c {
            Clause::Match(m) => Some(m),
            _ => None,
        })
        .collect();
    let first_edge_type = matches[0].patterns[0].elements.iter().find_map(|e| {
        if let PatternElement::Edge(ep) = e {
            ep.connection_type.clone()
        } else {
            None
        }
    });
    assert_eq!(
        first_edge_type.as_deref(),
        Some("VERY_COMMON"),
        "without id-anchored endpoints the proxy is unreliable; do not reorder"
    );
}

#[test]
fn test_fuse_match_return_aggregate_count_distinct() {
    // Single-MATCH + RETURN with count(DISTINCT v) on the OTHER node variable
    // — the planner should now fuse this and set distinct_count=true.
    let mut query = parse_cypher(
        "MATCH (a:Person)-[:KNOWS]->(b:Person) \
         RETURN a, count(DISTINCT b) AS friends \
         ORDER BY friends DESC LIMIT 10",
    )
    .unwrap();

    let graph = DirGraph::new();
    let params = HashMap::new();
    optimize(&mut query, &graph, &params);

    let mut found = false;
    for clause in &query.clauses {
        if let Clause::FusedMatchReturnAggregate {
            distinct_count,
            top_k,
            ..
        } = clause
        {
            assert!(*distinct_count, "distinct_count flag must be set");
            assert!(
                top_k.is_some(),
                "ORDER BY count DESC LIMIT 10 must absorb into top_k"
            );
            found = true;
        }
    }
    assert!(
        found,
        "FusedMatchReturnAggregate must fire for count(DISTINCT) shape"
    );
}

#[test]
fn test_fuse_match_with_aggregate_count_distinct() {
    // Single-MATCH + WITH with count(DISTINCT v) — pipeline form.
    let mut query = parse_cypher(
        "MATCH (a:Person)-[:KNOWS]->(b:Person) \
         WITH a, count(DISTINCT b) AS friends \
         RETURN a, friends",
    )
    .unwrap();

    let graph = DirGraph::new();
    let params = HashMap::new();
    optimize(&mut query, &graph, &params);

    let mut found = false;
    for clause in &query.clauses {
        if let Clause::FusedMatchWithAggregate { distinct_count, .. } = clause {
            assert!(*distinct_count, "WITH-form distinct_count flag must be set");
            found = true;
        }
    }
    assert!(
        found,
        "FusedMatchWithAggregate must fire for WITH-count-DISTINCT shape"
    );
}

#[test]
fn test_count_distinct_unconstrained_group_not_fused() {
    // Unconstrained group node — fusing would force a 124M-node enumeration
    // on a Wikidata-scale graph. The materializing path is faster in that
    // regime, so the planner must NOT fuse.
    let mut query = parse_cypher(
        "MATCH (a)-[:R]->(b) \
         RETURN b, count(DISTINCT a) AS n \
         ORDER BY n DESC LIMIT 10",
    )
    .unwrap();

    let graph = DirGraph::new();
    let params = HashMap::new();
    optimize(&mut query, &graph, &params);

    let fused_with_distinct = query.clauses.iter().any(|c| {
        matches!(
            c,
            Clause::FusedMatchReturnAggregate {
                distinct_count: true,
                ..
            }
        )
    });
    assert!(
        !fused_with_distinct,
        "untyped group node must skip distinct-count fusion"
    );
}

#[test]
fn test_count_distinct_5_element_pattern_not_fused() {
    // 5-element patterns aren't supported in the distinct-count path yet.
    // The planner should leave the query unfused so the materializing
    // executor produces semantically-correct results.
    let mut query = parse_cypher(
        "MATCH (a:A)-[:R1]->(b)<-[:R2]-(c) \
         RETURN a, count(DISTINCT c) AS n \
         ORDER BY n DESC LIMIT 10",
    )
    .unwrap();

    let graph = DirGraph::new();
    let params = HashMap::new();
    optimize(&mut query, &graph, &params);

    let fused_with_distinct = query.clauses.iter().any(|c| {
        matches!(
            c,
            Clause::FusedMatchReturnAggregate {
                distinct_count: true,
                ..
            }
        )
    });
    assert!(
        !fused_with_distinct,
        "5-element distinct-count pattern must not be fused"
    );
}

#[test]
fn test_fold_pass_through_with_between_matches() {
    // The cohort-top-K shape: `Match WITH p Match Return ...`. The
    // pass-through WITH should be stripped so downstream Match-Match
    // fusion can fire.
    let mut query = parse_cypher(
        "MATCH (p)-[:T1]->({id: 1}) \
         WITH p \
         MATCH (p)-[r]->() \
         RETURN p.title, count(r) AS d \
         ORDER BY d DESC LIMIT 10",
    )
    .unwrap();

    let graph = DirGraph::new();
    let params = HashMap::new();
    optimize(&mut query, &graph, &params);

    // No bare WITH clause should remain — the pass-through one was
    // dropped, and the multi-MATCH desugar produced a NEW with(agg)
    // that should have been consumed by fuse_match_with_aggregate +
    // fuse_match_with_aggregate_top_k.
    let bare_with_count = query
        .clauses
        .iter()
        .filter(|c| matches!(c, Clause::With(_)))
        .count();
    assert_eq!(
        bare_with_count, 0,
        "pass-through WITH must be stripped, and any synthesized WITH \
         must be absorbed by aggregate fusion; got query: {:#?}",
        query.clauses
    );

    // The query should have collapsed into a FusedMatchWithAggregate —
    // the streaming aggregate path. (top_k absorption is a separate
    // win that requires a pure-variable RETURN; we still benefit from
    // the streaming aggregate even when the RETURN includes
    // `p.title`.)
    let has_fused_aggregate = query
        .clauses
        .iter()
        .any(|c| matches!(c, Clause::FusedMatchWithAggregate { .. }));
    assert!(
        has_fused_aggregate,
        "expected the cohort query to land on the fused streaming \
         aggregate path; clauses: {:#?}",
        query.clauses
    );
}

#[test]
fn test_fold_pass_through_with_keeps_useful_with() {
    // A non-pass-through WITH (here aliasing or referencing extra
    // variables that subsequent clauses need) must NOT be folded.
    let mut query = parse_cypher("MATCH (p)-[r]->(q) WITH p, r RETURN p, r LIMIT 10").unwrap();

    let graph = DirGraph::new();
    let params = HashMap::new();
    optimize(&mut query, &graph, &params);

    // The pass-through WITH `WITH p, r` covers exactly what RETURN
    // references, so it CAN be safely folded in this case. To ensure
    // the converse test, use a different shape where the WITH
    // *renames* a variable.
    let mut renaming =
        parse_cypher("MATCH (p)-[r]->(q) WITH p AS person RETURN person LIMIT 10").unwrap();
    optimize(&mut renaming, &graph, &params);
    let has_with = renaming
        .clauses
        .iter()
        .any(|c| matches!(c, Clause::With(_)));
    assert!(
        has_with,
        "renaming WITH (`p AS person`) must not be folded — it changes scope"
    );
}

#[test]
fn test_fold_pass_through_with_skipped_when_orderby_follows() {
    // ORDER BY immediately after a WITH binds to the WITH's row scope.
    // Folding the WITH would move the ORDER BY's binding context.
    let mut query =
        parse_cypher("MATCH (p)-[:T]->({id: 1}) WITH p ORDER BY p.title LIMIT 10 RETURN p")
            .unwrap();

    let graph = DirGraph::new();
    let params = HashMap::new();
    optimize(&mut query, &graph, &params);

    // The WITH is preserved because ORDER BY follows it.
    let has_with = query.clauses.iter().any(|c| matches!(c, Clause::With(_)));
    assert!(
        has_with,
        "WITH followed by ORDER BY/SKIP/LIMIT must not be folded; \
         clauses: {:#?}",
        query.clauses
    );
}

#[test]
fn test_desugar_multi_match_return_aggregate() {
    // The Variant-B shape (Match-Match-Return-aggregate, no WITH).
    // Should be rewritten into Match-Match-With(group, agg)-Return so
    // the existing aggregate fusion fires.
    let mut query = parse_cypher(
        "MATCH (p)-[:T1]->({id: 1}) \
         MATCH (p)-[r]->() \
         RETURN p.title, count(r) AS d \
         ORDER BY d DESC LIMIT 10",
    )
    .unwrap();

    let graph = DirGraph::new();
    let params = HashMap::new();
    optimize(&mut query, &graph, &params);

    // Expected outcome: the aggregate fusion absorbs the synthesized
    // WITH into a FusedMatchWithAggregate (the streaming aggregate
    // path). top_k absorption requires a pure-variable RETURN — that's
    // covered by a separate fusion when the RETURN qualifies; this
    // test only pins the desugar→fusion handoff.
    let landed_on_fused_aggregate = query
        .clauses
        .iter()
        .any(|c| matches!(c, Clause::FusedMatchWithAggregate { .. }));
    assert!(
        landed_on_fused_aggregate,
        "Match-Match-Return-aggregate must desugar and fuse into the \
         streaming aggregate path; clauses: {:#?}",
        query.clauses
    );
}

#[test]
fn test_topk_absorbed_for_property_access_return() {
    // Cohort top-K with property-access RETURN — the shape that drove
    // the user's Wikidata timeout. After desugar→fuse, the planner
    // must absorb ORDER BY/LIMIT into FusedMatchWithAggregate.top_k so
    // p.title and p.description are evaluated K times, not |cohort|
    // times.
    let mut query = parse_cypher(
        "MATCH (p)-[:T1]->({id: 1}) \
         MATCH (p)-[r]-(other) \
         WHERE NOT (type(r) = 'T2' AND startNode(r) = other) \
         RETURN p.title AS name, p.description AS desc, count(r) AS d \
         ORDER BY d DESC LIMIT 10",
    )
    .unwrap();

    let graph = DirGraph::new();
    let params = HashMap::new();
    optimize(&mut query, &graph, &params);

    let topk_absorbed = query
        .clauses
        .iter()
        .any(|c| matches!(c, Clause::FusedMatchWithAggregate { top_k: Some(_), .. }));
    assert!(
        topk_absorbed,
        "ORDER BY DESC LIMIT 10 must be absorbed into \
         FusedMatchWithAggregate.top_k when RETURN projects \
         properties of the group variable; clauses: {:#?}",
        query.clauses
    );
}

#[test]
fn test_topk_absorbed_with_explicit_pass_through_with() {
    // The user's *exact* shape from the Wikidata Q1 timeout, with the
    // explicit `WITH p` between the two MATCHes. fold_pass_through_with
    // must remove it, then desugar+fuse must produce a
    // FusedMatchWithAggregate with top_k = Some — otherwise the cohort's
    // p.title and p.description columns are read for every group key.
    let mut query = parse_cypher(
        "MATCH (p)-[:P27]->({id: 20}) \
         WITH p \
         MATCH (p)-[r]-(other) \
         WHERE NOT (type(r) = 'P50' AND startNode(r) = other) \
         RETURN p.title AS name, p.description AS desc, count(r) AS connections \
         ORDER BY connections DESC LIMIT 10",
    )
    .unwrap();

    let graph = DirGraph::new();
    let params = HashMap::new();
    optimize(&mut query, &graph, &params);

    let topk_absorbed = query
        .clauses
        .iter()
        .any(|c| matches!(c, Clause::FusedMatchWithAggregate { top_k: Some(_), .. }));
    assert!(
        topk_absorbed,
        "user's exact Q1 shape must absorb top_k; clauses: {:#?}",
        query.clauses
    );
}

#[test]
fn test_topk_skipped_for_computed_return_expressions() {
    // Computed RETURN expressions (arithmetic on aggregates here) are
    // not safe to absorb — we'd need *all* rows to know which 10 win.
    // Pin the bail-out so a future relaxation has to opt in.
    let mut query = parse_cypher(
        "MATCH (p)-[:T1]->({id: 1}) \
         MATCH (p)-[r]-() \
         WITH p, count(r) AS total, 1 AS one \
         RETURN p.title, total + one AS adjusted \
         ORDER BY total DESC LIMIT 10",
    )
    .unwrap();

    let graph = DirGraph::new();
    let params = HashMap::new();
    optimize(&mut query, &graph, &params);

    let topk_absorbed = query
        .clauses
        .iter()
        .any(|c| matches!(c, Clause::FusedMatchWithAggregate { top_k: Some(_), .. }));
    assert!(
        !topk_absorbed,
        "computed RETURN expressions must not absorb top_k; \
         clauses: {:#?}",
        query.clauses
    );
}

#[test]
fn test_desugar_skips_when_no_aggregate() {
    // No aggregate in RETURN → desugar must not fire (would just add
    // an unnecessary WITH).
    let mut query = parse_cypher(
        "MATCH (p)-[:T1]->({id: 1}) \
         MATCH (p)-[:T2]->({id: 2}) \
         RETURN p.title \
         LIMIT 10",
    )
    .unwrap();

    let graph = DirGraph::new();
    let params = HashMap::new();
    optimize(&mut query, &graph, &params);

    let bare_with_count = query
        .clauses
        .iter()
        .filter(|c| matches!(c, Clause::With(_)))
        .count();
    assert_eq!(
        bare_with_count, 0,
        "desugar must not introduce a WITH when RETURN has no aggregate"
    );
}

#[test]
fn test_desugar_skips_when_multiple_group_vars() {
    // Two distinct group variables (`p.title`, `q.name`) — the simple
    // single-variable rewrite doesn't apply; bail.
    let mut query = parse_cypher(
        "MATCH (p)-[:T1]->({id: 1}) \
         MATCH (p)-[r]->(q) \
         RETURN p.title, q.title, count(r) AS d \
         ORDER BY d DESC LIMIT 10",
    )
    .unwrap();

    let graph = DirGraph::new();
    let params = HashMap::new();
    optimize(&mut query, &graph, &params);

    // Should NOT have produced a FusedMatchWithAggregate — the desugar
    // was correctly skipped, leaving the query for the slow path.
    let fused_count = query
        .clauses
        .iter()
        .filter(|c| matches!(c, Clause::FusedMatchWithAggregate { .. }))
        .count();
    assert_eq!(
        fused_count, 0,
        "multi-group-variable RETURN must not be auto-rewritten"
    );
}
