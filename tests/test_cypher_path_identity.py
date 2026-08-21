"""Relationship identity and trail semantics for Cypher path bindings.

These cases are independently authored from the public language contract.  They
lock the two invariants path consumers rely on: each hop retains the exact
relationship matched, and one relationship cannot occur twice in one path.
"""

from kglite import KnowledgeGraph


def _parallel_path_graph() -> KnowledgeGraph:
    graph = KnowledgeGraph()
    graph.cypher("CREATE (:N {id: 1}), (:N {id: 2}), (:N {id: 3})")
    for tag in ("r1", "r2"):
        graph.cypher(f"MATCH (a:N {{id: 1}}), (b:N {{id: 2}}) CREATE (a)-[:R {{tag: '{tag}'}}]->(b)")
    for tag in ("s1", "s2"):
        graph.cypher(f"MATCH (b:N {{id: 2}}), (c:N {{id: 3}}) CREATE (b)-[:S {{tag: '{tag}'}}]->(c)")
    return graph


def test_fixed_path_retains_each_parallel_relationship():
    graph = _parallel_path_graph()
    rows = graph.cypher(
        "MATCH p=(a:N {id: 1})-[:R]->(b:N)-[:S]->(c:N {id: 3}) RETURN [r IN relationships(p) | r.tag] AS tags"
    ).to_list()

    assert sorted(tuple(row["tags"]) for row in rows) == [
        ("r1", "s1"),
        ("r1", "s2"),
        ("r2", "s1"),
        ("r2", "s2"),
    ]


def test_fixed_incoming_path_retains_relationship_identity_and_orientation():
    graph = _parallel_path_graph()
    rows = graph.cypher(
        "MATCH p=(b:N {id: 2})<-[r:R]-(a:N {id: 1}) "
        "RETURN id(r) AS bound_id, id(head(relationships(p))) AS path_id, "
        "relationships(p) AS relationships "
        "ORDER BY bound_id"
    ).to_list()

    assert len(rows) == 2
    assert all(row["path_id"] == row["bound_id"] for row in rows)
    assert all((row["relationships"][0]["start"], row["relationships"][0]["end"]) == (0, 1) for row in rows)


def test_fixed_path_cannot_reuse_one_relationship():
    graph = KnowledgeGraph()
    graph.cypher("CREATE (a:N {id: 1})")
    graph.cypher("MATCH (a:N {id: 1}) CREATE (a)-[:LOOP]->(a)")

    query = "MATCH p=(a:N {id: 1})-[:LOOP]->(b)-[:LOOP]->(c) RETURN count(*) AS paths"
    rows = graph.cypher(query).to_list()
    unfused = graph.cypher(query, disabled_passes=["fuse_match_return_aggregate"]).to_list()
    assert rows == [{"paths": 0}]
    assert unfused == rows


def test_variable_path_enumerates_parallel_relationships_exactly():
    graph = _parallel_path_graph()
    rows = graph.cypher(
        "MATCH p=(a:N {id: 1})-[:R*1..1]->(b:N {id: 2}) "
        "RETURN id(head(relationships(p))) AS rel_id, "
        "head(relationships(p)).tag AS tag ORDER BY tag"
    ).to_list()

    assert [row["tag"] for row in rows] == ["r1", "r2"]
    assert len({row["rel_id"] for row in rows}) == 2


def test_variable_path_cannot_reuse_relationship_but_may_repeat_nodes():
    graph = KnowledgeGraph()
    graph.cypher("CREATE (:N {id: 1}), (:N {id: 2})")
    graph.cypher("MATCH (a:N {id: 1}) CREATE (a)-[:LOOP]->(a)")
    graph.cypher("MATCH (a:N {id: 1}), (b:N {id: 2}) CREATE (a)-[:GO]->(b), (b)-[:BACK]->(a)")

    reused = graph.cypher("MATCH p=(a:N {id: 1})-[:LOOP*2..2]->(b) RETURN count(*) AS paths").to_list()
    repeated_node = graph.cypher(
        "MATCH p=(a:N {id: 1})-[:GO|BACK*2..2]->(b:N {id: 1}) RETURN [n IN nodes(p) | n.id] AS ids"
    ).to_list()

    assert reused == [{"paths": 0}]
    assert repeated_node == [{"ids": [1, 2, 1]}]


def test_undirected_variable_path_cannot_reverse_over_same_relationship():
    graph = KnowledgeGraph()
    graph.cypher("CREATE (:N {id: 1}), (:N {id: 2})")
    graph.cypher("MATCH (a:N {id: 1}), (b:N {id: 2}) CREATE (a)-[:R]->(b)")

    rows = graph.cypher("MATCH p=(a:N {id: 1})-[:R*2..2]-(b:N {id: 1}) RETURN count(*) AS paths").to_list()
    assert rows == [{"paths": 0}]


# ── The optimized (`needs_path_info = false`) expansion ──────────────
#
# Every test above binds `p = ...` or names the relationship variable, which
# forces the exact per-path expansion. The planner's `mark_fast_var_length_paths`
# swaps in a set-based BFS for `RETURN DISTINCT` shapes, and that BFS answered
# *distance* reachability — a different relation, and one that shipped wrong
# answers for a year. These are the same graphs and the same claims, spelled so
# the fast expansion is the thing under test.


def _applied_passes(graph: KnowledgeGraph, query: str) -> set[str]:
    return {
        row["operation"].removeprefix("OptimizerPass ")
        for row in graph.cypher(f"EXPLAIN {query}").to_list()
        if str(row["operation"]).startswith("OptimizerPass ")
    }


def test_fast_variable_path_engages_only_within_its_legal_window():
    """The gate itself: `min <= 1` marks, `min >= 2` never does.

    Without this the trail tests below could pass for the boring reason that
    the optimization stopped firing entirely.
    """
    graph = _parallel_path_graph()
    marked = "MATCH (a:N {id: 1})-[:R*1..3]->(b:N) RETURN DISTINCT b.id AS i"
    unmarked = "MATCH (a:N {id: 1})-[:R*2..3]->(b:N) RETURN DISTINCT b.id AS i"

    assert "mark_fast_var_length_paths" in _applied_passes(graph, marked)
    assert "mark_fast_var_length_paths" not in _applied_passes(graph, unmarked)


def test_fast_variable_path_cannot_reuse_relationship_but_may_repeat_nodes():
    graph = KnowledgeGraph()
    graph.cypher("CREATE (:N {id: 1}), (:N {id: 2})")
    graph.cypher("MATCH (a:N {id: 1}) CREATE (a)-[:LOOP]->(a)")
    graph.cypher("MATCH (a:N {id: 1}), (b:N {id: 2}) CREATE (a)-[:GO]->(b), (b)-[:BACK]->(a)")

    reused = graph.cypher("MATCH (a:N {id: 1})-[:LOOP*2..2]->(b:N) RETURN DISTINCT b.id AS i").to_list()
    repeated_node = graph.cypher("MATCH (a:N {id: 1})-[:GO|BACK*2..2]->(b:N) RETURN DISTINCT b.id AS i").to_list()
    # The source is a legal answer to its own segment when a closed trail
    # returns to it — here the single LOOP relationship (1 hop) and the
    # GO/BACK round trip (2 hops).
    source_on_closed_trail = graph.cypher("MATCH (a:N {id: 1})-[:LOOP*1..2]->(b:N) RETURN DISTINCT b.id AS i").to_list()

    assert reused == []
    assert repeated_node == [{"i": 1}]
    assert source_on_closed_trail == [{"i": 1}]


def test_fast_undirected_variable_path_cannot_reverse_over_same_relationship():
    graph = KnowledgeGraph()
    graph.cypher("CREATE (:N {id: 1}), (:N {id: 2})")
    graph.cypher("MATCH (a:N {id: 1}), (b:N {id: 2}) CREATE (a)-[:R]->(b)")

    # One relationship: walking back over it is not a trail, so 1 is not its
    # own answer and `*2..2` has none at all.
    assert graph.cypher("MATCH (a:N {id: 1})-[:R*2..2]-(b:N) RETURN DISTINCT b.id AS i").to_list() == []
    assert graph.cypher("MATCH (a:N {id: 1})-[:R*1..2]-(b:N) RETURN DISTINCT b.id AS i").to_list() == [{"i": 2}]


def test_fast_undirected_variable_path_returns_over_a_parallel_relationship():
    graph = _parallel_path_graph()

    # Two `:R` relationships between 1 and 2 do make a closed trail of length
    # 2, so 1 IS one of its own answers here — the case the single-relationship
    # graph above must not report.
    rows = graph.cypher("MATCH (a:N {id: 1})-[:R*1..2]-(b:N) RETURN DISTINCT b.id AS i").to_list()
    assert sorted(row["i"] for row in rows) == [1, 2]

    exact = graph.cypher("MATCH (a:N {id: 1})-[:R*1..1]->(b:N) RETURN DISTINCT b.id AS i").to_list()
    assert exact == [{"i": 2}]


def test_all_shortest_paths_distinguishes_parallel_relationships():
    graph = _parallel_path_graph()
    rows = graph.cypher(
        "MATCH p=allShortestPaths((a:N {id: 1})-[:R*]->(b:N {id: 2})) "
        "RETURN head(relationships(p)).tag AS tag ORDER BY tag"
    ).to_list()

    assert rows == [{"tag": "r1"}, {"tag": "r2"}]


def test_shortest_path_preserves_multi_type_filter():
    graph = KnowledgeGraph()
    graph.cypher("CREATE (:N {id: 1}), (:N {id: 2}), (:N {id: 3})")
    graph.cypher(
        "MATCH (a:N {id: 1}), (b:N {id: 2}), (c:N {id: 3}) CREATE (a)-[:A {tag: 'a'}]->(b), (b)-[:B {tag: 'b'}]->(c)"
    )

    rows = graph.cypher(
        "MATCH p=shortestPath((a:N {id: 1})-[:A|B*]->(c:N {id: 3})) RETURN [r IN relationships(p) | r.tag] AS tags"
    ).to_list()

    assert rows == [{"tags": ["a", "b"]}]


# ── distinct-target pushdown (part-6 phase V4) ───────────────────────────────
#
# A variable-length segment feeding a distinct-only consumer deduplicates by
# target *during* expansion, so a multi-seed k-hop never materialises one row
# per (source, target) pair. These pin the two things that could go wrong with
# it: an answer that is still per-source when the source escapes to the
# consumer, and a target skipped for traversal rather than for emission.


def _two_component_graph() -> KnowledgeGraph:
    """0→1→2→3→0, 4→5→6→4, plus the bridges 1→5 and 2→7, and 7→8→9→7.

    Three overlapping neighbourhoods, so several seeds reach targets each other
    already reached — the shape a global target dedup can get wrong.
    """
    graph = KnowledgeGraph()
    graph.cypher("CREATE " + ", ".join(f"(:P {{id: {i}}})" for i in range(10)))
    for source, target in [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 4),
        (1, 5),
        (7, 8),
        (8, 9),
        (9, 7),
        (2, 7),
    ]:
        graph.cypher(f"MATCH (a:P {{id: {source}}}), (b:P {{id: {target}}}) CREATE (a)-[:K]->(b)")
    return graph


def test_multi_seed_khop_counts_the_union_of_the_reachable_sets():
    graph = _two_component_graph()

    # Seed 0 reaches 1,2,5,3,7,6; seed 4 reaches 5,6,4. The union is seven
    # nodes — not the nine rows a per-source count would add up to.
    assert graph.cypher(
        "MATCH (p:P)-[:K*1..3]->(f:P) WHERE p.id IN [0, 4] RETURN count(DISTINCT f) AS c"
    ).to_list() == [{"c": 7}]
    assert sorted(
        row["i"]
        for row in graph.cypher("MATCH (p:P)-[:K*1..3]->(f:P) WHERE p.id IN [0, 4] RETURN DISTINCT f.id AS i").to_list()
    ) == [1, 2, 3, 4, 5, 6, 7]
    # `count(*)` counts paths, so it is NOT dedup-safe and must keep every one.
    assert graph.cypher("MATCH (p:P)-[:K*1..3]->(f:P) WHERE p.id IN [0, 4] RETURN count(*) AS c").to_list() == [
        {"c": 9}
    ]


def test_a_source_in_the_projection_keeps_the_count_per_source():
    """The control for the escape analysis: `p` in the projection forbids it.

    A global target dedup would answer this with one seed's count and an empty
    group for the other, or with the union split arbitrarily between them.
    """
    graph = _two_component_graph()

    rows = graph.cypher(
        "MATCH (p:P)-[:K*1..3]->(f:P) WHERE p.id IN [0, 4] RETURN p.id AS pid, count(DISTINCT f) AS c ORDER BY pid"
    ).to_list()

    assert rows == [{"pid": 0, "c": 6}, {"pid": 4, "c": 3}]


def test_a_seen_target_is_skipped_for_emission_but_still_traversed():
    """Seed 1's own targets are only reachable *through* seed 0's.

    Everything seed 1 adds (0, 4, 8) sits one hop past a node seed 0 already
    reached, so a dedup that pruned the BFS frontier instead of the emitted row
    would lose all three.
    """
    graph = _two_component_graph()

    assert graph.cypher(
        "MATCH (p:P)-[:K*1..3]->(f:P) WHERE p.id IN [0, 1] RETURN count(DISTINCT f) AS c"
    ).to_list() == [{"c": 9}]
    assert graph.cypher(
        "MATCH (p:P)-[:K*1..3]->(f:P) WHERE p.id IN [0, 1] RETURN p.id AS pid, count(DISTINCT f) AS c ORDER BY pid"
    ).to_list() == [{"pid": 0, "c": 6}, {"pid": 1, "c": 8}]


def test_comma_patterns_do_not_dedup_before_the_join():
    """Regression: the hint deduplicated the first pattern, then joined.

    `a1` and `a2` both reach `f`, but only `a2` has the `:M` relationship the
    second pattern needs. Keeping one arbitrary `a` per `f` picked `a1`, the
    join then rejected it, and `f` was lost: this answered `[]` where the
    same query without `DISTINCT` answers `[3]`.
    """
    graph = KnowledgeGraph()
    graph.cypher("CREATE " + ", ".join(f"(:P {{id: {i}}})" for i in (1, 2, 3, 4)))
    for source, target, rel in [(1, 3, "K"), (2, 3, "K"), (2, 4, "M")]:
        graph.cypher(f"MATCH (a:P {{id: {source}}}), (b:P {{id: {target}}}) CREATE (a)-[:{rel}]->(b)")

    assert graph.cypher("MATCH (a:P)-[:K]->(f:P), (a)-[:M]->(g:P) RETURN DISTINCT f.id AS i").to_list() == [{"i": 3}]
    assert graph.cypher("MATCH (a:P)-[:K]->(f:P), (a)-[:M]->(g:P) RETURN f.id AS i").to_list() == [{"i": 3}]


def test_a_representative_rejected_by_the_fused_where_does_not_lose_its_target():
    """The retry that makes matcher-level dedup sound.

    `a1` and `a2` both reach `f`, and only `a2` passes the predicate. The
    matcher deduplicates by `f` and keeps whichever match it saw first — `a1`'s
    — which the fused WHERE then rejects. Without the pass being redone without
    the dedup, `f` disappears: this answers `[]` where the same query without
    `DISTINCT` answers one row. `a.age + 0 > 10` is deliberately not pushable,
    so the matcher cannot have enforced it already.
    """
    graph = KnowledgeGraph()
    graph.cypher("CREATE (:P {id: 1, age: 1}), (:P {id: 2, age: 100}), (:P {id: 3, age: 0})")
    for source in (1, 2):
        graph.cypher(f"MATCH (a:P {{id: {source}}}), (b:P {{id: 3}}) CREATE (a)-[:K]->(b)")

    assert graph.cypher("MATCH (a:P)-[:K]->(f:P) WHERE a.age + 0 > 10 RETURN DISTINCT f.id AS i").to_list() == [
        {"i": 3}
    ]
    assert graph.cypher("MATCH (a:P)-[:K]->(f:P) WHERE a.age + 0 > 10 RETURN count(DISTINCT f) AS c").to_list() == [
        {"c": 1}
    ]
    assert graph.cypher("MATCH (a:P)-[:K]->(f:P) WHERE a.age + 0 > 10 RETURN f.id AS i").to_list() == [{"i": 3}]
