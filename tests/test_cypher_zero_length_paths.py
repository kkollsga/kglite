"""A `*0..` segment always yields its zero-length path.

openCypher: a variable-length pattern with `min = 0` yields the zero-length
path binding both endpoints to the same node, **independently of the
relationship type**. A relationship type constrains relationships, and a
zero-length path has none.

KGLite's `expand_from_node` short-circuited an unknown relationship type
before the variable-length dispatch, so `-[:ZZZ*0..2]->(b)` returned nothing
at all instead of the source node — and `OPTIONAL MATCH` produced a null row
rather than a real one. `shortestPath` carried an independent copy of the same
defect: its endpoint loop skipped `source == target` outright, so
`shortestPath((a)-[:K*0..]-(a))` answered "no path" even for a *known* type.
"""

from __future__ import annotations

import pytest

import kglite

PROFILES = pytest.mark.parametrize("disable_optimizer", [False, True])


@pytest.fixture
def linked_pair():
    """`(a:P {id: 1})-[:K]->(b:P {id: 2})` — one relationship, of one type."""
    graph = kglite.KnowledgeGraph()
    graph.cypher("CREATE (a:P {id: 1}), (b:P {id: 2}), (a)-[:K]->(b)")
    return graph


def ids(graph, query, disable_optimizer):
    rows = graph.cypher(query, disable_optimizer=disable_optimizer).to_list()
    return [next(iter(row.values())) for row in rows]


@PROFILES
@pytest.mark.parametrize(
    "pattern",
    [
        "(a:P {id: 1})-[:ZZZ*0..2]->(b)",
        "(a:P {id: 1})-[:ZZZ*0]->(b)",
        "(a:P {id: 1})-[:ZZZ*0..0]->(b)",
        "(a:P {id: 1})-[:ZZZ*0..2]-(b)",
        "(a:P {id: 1})<-[:ZZZ*0..2]-(b)",
        "(a:P {id: 1})-[:ZZZ|YYY*0..2]->(b)",
    ],
)
def test_zero_length_path_survives_an_unknown_relationship_type(linked_pair, pattern, disable_optimizer):
    assert ids(linked_pair, f"MATCH {pattern} RETURN b.id", disable_optimizer) == [1]


@PROFILES
def test_the_zero_length_path_of_an_unknown_type_has_no_relationships(linked_pair, disable_optimizer):
    rows = linked_pair.cypher(
        "MATCH p = (a:P {id: 1})-[r:ZZZ*0..2]->(b) RETURN b.id AS id, size(relationships(p)) AS n, length(p) AS len",
        disable_optimizer=disable_optimizer,
    ).to_list()
    assert rows == [{"id": 1, "n": 0, "len": 0}]


@PROFILES
@pytest.mark.parametrize(
    "pattern",
    [
        "(a:P {id: 1})-[:ZZZ*1..2]->(b)",
        "(a:P {id: 1})-[:ZZZ*1]->(b)",
        "(a:P {id: 1})-[:ZZZ*2..3]-(b)",
        "(a:P {id: 1})-[:ZZZ]->(b)",
    ],
)
def test_an_unknown_type_still_matches_nothing_from_one_hop_up(linked_pair, pattern, disable_optimizer):
    assert ids(linked_pair, f"MATCH {pattern} RETURN b.id", disable_optimizer) == []


@PROFILES
@pytest.mark.parametrize(
    "pattern",
    ["(a:P {id: 1})-[:K*0..2]->(b)", "(a:P {id: 1})-[*0..2]->(b)"],
)
def test_a_known_or_untyped_zero_length_segment_is_unchanged(linked_pair, pattern, disable_optimizer):
    assert sorted(ids(linked_pair, f"MATCH {pattern} RETURN b.id", disable_optimizer)) == [
        1,
        2,
    ]


@PROFILES
def test_optional_match_yields_the_zero_length_row_not_a_null(linked_pair, disable_optimizer):
    """The row exists, so `OPTIONAL MATCH` has nothing to pad: `b` is `a`."""
    rows = linked_pair.cypher(
        "MATCH (a:P {id: 1}) OPTIONAL MATCH (a)-[:ZZZ*0..2]->(b) RETURN b.id AS id",
        disable_optimizer=disable_optimizer,
    ).to_list()
    assert rows == [{"id": 1}]
    # `*1..` has no zero-length path, so that one really is padded.
    rows = linked_pair.cypher(
        "MATCH (a:P {id: 1}) OPTIONAL MATCH (a)-[:ZZZ*1..2]->(b) RETURN b.id AS id",
        disable_optimizer=disable_optimizer,
    ).to_list()
    assert rows == [{"id": None}]


@PROFILES
def test_the_zero_length_path_still_honours_the_target_pattern(linked_pair, disable_optimizer):
    linked_pair.cypher("CREATE (:Q {id: 9})")
    assert ids(linked_pair, "MATCH (a:P {id: 1})-[:ZZZ*0..2]->(b:Q) RETURN b.id", disable_optimizer) == []
    assert ids(linked_pair, "MATCH (a:P {id: 1})-[:ZZZ*0..2]->(b:P) RETURN b.id", disable_optimizer) == [1]


@PROFILES
def test_an_unknown_type_still_warns_and_names_the_zero_length_exception(linked_pair, capfd, disable_optimizer):
    """The warning stays, but its "returns no rows" claim is false for a
    zero-hop-capable segment, so that shape says what it actually does.
    """
    linked_pair.cypher("MATCH (a:P {id: 1})-[:ZZZ*1..2]->(b) RETURN b.id", disable_optimizer=disable_optimizer)
    err = capfd.readouterr().err
    assert "unknown relationship type 'ZZZ'" in err
    assert "returns no rows" in err

    linked_pair.cypher("MATCH (a:P {id: 1})-[:ZZZ*0..2]->(b) RETURN b.id", disable_optimizer=disable_optimizer)
    err = capfd.readouterr().err
    assert "unknown relationship type 'ZZZ'" in err
    assert "returns no rows" not in err
    assert "zero-length path" in err


# ──────────────────────────────────────────────────────────────────────────
# shortestPath — an independent copy of the same contract.
# ──────────────────────────────────────────────────────────────────────────


@PROFILES
@pytest.mark.parametrize(
    "pattern",
    [
        "shortestPath((a)-[:K*0..]-(a))",
        "shortestPath((a)-[*0..]-(a))",
        "shortestPath((a)-[:ZZZ*0..]-(a))",
        "shortestPath((a)-[:K*0..3]->(a))",
    ],
)
def test_shortest_path_answers_the_zero_length_path_to_the_same_node(linked_pair, pattern, disable_optimizer):
    rows = linked_pair.cypher(
        f"MATCH (a:P {{id: 1}}) MATCH p = {pattern} RETURN length(p) AS len, size(relationships(p)) AS rels",
        disable_optimizer=disable_optimizer,
    ).to_list()
    assert rows == [{"len": 0, "rels": 0}]


@PROFILES
@pytest.mark.parametrize(
    "pattern",
    [
        "shortestPath((a)-[:K*1..]-(a))",
        "shortestPath((a)-[*]-(a))",
        "shortestPath((a)-[:K*2..3]-(a))",
    ],
)
def test_shortest_path_still_refuses_a_min_one_walk_back_to_the_same_node(linked_pair, pattern, disable_optimizer):
    rows = linked_pair.cypher(
        f"MATCH (a:P {{id: 1}}) MATCH p = {pattern} RETURN length(p) AS len",
        disable_optimizer=disable_optimizer,
    ).to_list()
    assert rows == []


@PROFILES
def test_shortest_path_between_distinct_nodes_is_unchanged(linked_pair, disable_optimizer):
    rows = linked_pair.cypher(
        "MATCH (a:P {id: 1}), (c:P {id: 2}) MATCH p = shortestPath((a)-[:K*0..]-(c)) RETURN length(p) AS len",
        disable_optimizer=disable_optimizer,
    ).to_list()
    assert rows == [{"len": 1}]
