"""Cross-member parity for the Python shortest-path family.

The family has seven entry points (``shortest_path``, ``_ids``, ``_indices``,
``_length``, ``_lengths_batch``, ``are_connected``, ``all_paths``).  Before
this suite they answered *four different questions*: some honoured
``connection_types`` / ``via_types``, some silently ignored them, none could
express a direction.  A ``Person``-to-``Person`` "distance" was routinely
answered through a ``City``.

The oracle is Cypher.  ``shortestPath()`` is both direction-correct (the arrow
in the pattern) and filter-correct (the relationship type in the pattern), so
every Python cell that has an expressible Cypher spelling is asserted against
it rather than against a hand-computed number.  Cells Cypher cannot express
(``via_types``, the batch API's shape) carry exact-value regressions instead.

``node_type`` arguments are an **ID namespace** throughout the family — they
say which type to look the endpoint id up in, never which types the traversal
may route through.  That is ``via_types``' job, and several cases below pin
the distinction.
"""

from __future__ import annotations

import pandas as pd
import pytest

from kglite import KnowledgeGraph

# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------
#
#   Component 1                                  Component 2
#   -----------                                  -----------
#   (1 Alice) --KNOWS--> (2 Bob) --KNOWS--> (5 Erin) --KNOWS--> (4 Dave)
#                            ^                                     |
#                            +---------------KNOWS-----------------+
#                                                                  |
#   (3 Carol) --LIVES_IN--> (10 Oslo) <--LIVES_IN-- (4 Dave) ------+
#
#                                                (6 Frank) --LIVES_IN--> (11 Bergen)
#
# The shape is chosen so that every axis under test is *load-bearing*:
#
#  * 1 -> 4 undirected is 2 hops (1-2-4, using KNOWS 4->2 **backwards**) but
#    3 hops directed (1->2->5->4).  A direction bug changes the answer.
#  * 3 -> 4 is 2 hops through the City Oslo and has **no** KNOWS route.  A
#    dropped ``connection_types`` filter changes None into 2.
#  * Frank/Bergen are a second component: nothing reaches them.


@pytest.fixture(scope="module")
def path_graph() -> KnowledgeGraph:
    g = KnowledgeGraph()
    people = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6],
            "name": ["Alice", "Bob", "Carol", "Dave", "Erin", "Frank"],
        }
    )
    g.add_nodes(people, "Person", "id", "name")
    cities = pd.DataFrame({"id": [10, 11], "name": ["Oslo", "Bergen"]})
    g.add_nodes(cities, "City", "id", "name")

    knows = pd.DataFrame(
        {
            "src": [1, 4, 2, 5],
            "dst": [2, 2, 5, 4],
            "weight": [1.0, 1.0, 1.0, 1.0],
        }
    )
    g.add_connections(knows, "KNOWS", "Person", "src", "Person", "dst")

    lives = pd.DataFrame({"src": [3, 4, 6], "dst": [10, 10, 11], "weight": [1.0, 1.0, 1.0]})
    g.add_connections(lives, "LIVES_IN", "Person", "src", "City", "dst")
    return g


# ---------------------------------------------------------------------------
# The Cypher oracle
# ---------------------------------------------------------------------------

_ARROW = {"any": ("-", "-"), "outgoing": ("-", "->"), "incoming": ("<-", "-")}


def cypher_length(
    graph: KnowledgeGraph,
    src_id: int,
    dst_id: int,
    rel: str | None,
    direction: str,
) -> int | None:
    """Hop count of the Cypher ``shortestPath`` for the same question."""
    left, right = _ARROW[direction]
    rel_pat = f":{rel}" if rel else ""
    query = (
        f"MATCH (a:Person), (b:Person) "
        f"WHERE a.id = {src_id} AND b.id = {dst_id} "
        f"MATCH p = shortestPath((a){left}[{rel_pat}*..10]{right}(b)) "
        f"RETURN length(p) AS n"
    )
    rows = graph.cypher(query).to_dicts()
    return rows[0]["n"] if rows else None


# ---------------------------------------------------------------------------
# Pin-first: the default (no new kwargs) answers must never move
# ---------------------------------------------------------------------------


class TestDefaultsUnchanged:
    """Every default answer captured against the pre-S2 build."""

    def test_shortest_path_default(self, path_graph):
        r = path_graph.shortest_path("Person", 1, "Person", 4)
        assert r["length"] == 2
        assert [n["id"] for n in r["path"]] == [1, 2, 4]
        assert r["connections"] == ["KNOWS", "KNOWS"]

    def test_shortest_path_ids_default(self, path_graph):
        assert path_graph.shortest_path_ids("Person", 1, "Person", 4) == [1, 2, 4]

    def test_shortest_path_indices_default(self, path_graph):
        assert len(path_graph.shortest_path_indices("Person", 1, "Person", 4)) == 3

    def test_shortest_path_length_default(self, path_graph):
        assert path_graph.shortest_path_length("Person", 1, "Person", 4) == 2
        # Person->Person "distance" answered through a City, by design:
        # node_type is an id namespace, not a traversal restriction.
        assert path_graph.shortest_path_length("Person", 3, "Person", 4) == 2

    def test_shortest_path_lengths_batch_default(self, path_graph):
        assert path_graph.shortest_path_lengths_batch("Person", [(1, 4), (3, 4), (1, 6)]) == [
            2,
            2,
            None,
        ]

    def test_are_connected_default(self, path_graph):
        assert path_graph.are_connected("Person", 1, "Person", 4) is True
        assert path_graph.are_connected("Person", 1, "Person", 6) is False

    def test_all_paths_default(self, path_graph):
        paths = path_graph.all_paths("Person", 1, "Person", 4, max_hops=5)
        assert sorted(p["length"] for p in paths) == [2, 3]

    def test_weighted_default(self, path_graph):
        assert path_graph.shortest_path_length("Person", 1, "Person", 4, weight_property="weight") == 2.0

    def test_no_path_across_components(self, path_graph):
        assert path_graph.shortest_path("Person", 1, "Person", 6) is None
        assert path_graph.shortest_path_ids("Person", 1, "Person", 6) is None
        assert path_graph.shortest_path_indices("Person", 1, "Person", 6) is None
        assert path_graph.shortest_path_length("Person", 1, "Person", 6) is None
        assert path_graph.all_paths("Person", 1, "Person", 6) == []


# ---------------------------------------------------------------------------
# The matrix: every entry point x {no filter, connection_types} x direction
# ---------------------------------------------------------------------------

# (src, dst, rel, direction) -> the answer Cypher gives for the same question.
MATRIX = [
    (src, dst, rel, direction)
    for src, dst in [(1, 4), (4, 1), (3, 4), (1, 6)]
    for rel in [None, "KNOWS"]
    for direction in ["any", "outgoing", "incoming"]
]


def _cell_id(cell) -> str:
    src, dst, rel, direction = cell
    return f"{src}to{dst}-{rel or 'anyrel'}-{direction}"


@pytest.mark.parametrize("cell", MATRIX, ids=_cell_id)
def test_length_matches_cypher(path_graph, cell):
    src, dst, rel, direction = cell
    expected = cypher_length(path_graph, src, dst, rel, direction)
    kwargs = {"direction": direction}
    if rel:
        kwargs["connection_types"] = [rel]
    assert path_graph.shortest_path_length("Person", src, "Person", dst, **kwargs) == expected


@pytest.mark.parametrize("cell", MATRIX, ids=_cell_id)
def test_family_members_agree(path_graph, cell):
    """All seven members answer the same question for the same arguments."""
    src, dst, rel, direction = cell
    expected = cypher_length(path_graph, src, dst, rel, direction)
    kwargs = {"direction": direction}
    if rel:
        kwargs["connection_types"] = [rel]

    full = path_graph.shortest_path("Person", src, "Person", dst, **kwargs)
    assert (full["length"] if full else None) == expected

    ids = path_graph.shortest_path_ids("Person", src, "Person", dst, **kwargs)
    assert (len(ids) - 1 if ids else None) == expected

    idxs = path_graph.shortest_path_indices("Person", src, "Person", dst, **kwargs)
    assert (len(idxs) - 1 if idxs else None) == expected

    assert path_graph.shortest_path_length("Person", src, "Person", dst, **kwargs) == expected

    assert path_graph.shortest_path_lengths_batch("Person", [(src, dst)], **kwargs) == [expected]

    assert path_graph.are_connected("Person", src, "Person", dst, **kwargs) is (expected is not None)

    # all_paths enumerates every path; its minimum is the shortest.
    paths = path_graph.all_paths("Person", src, "Person", dst, max_hops=6, **kwargs)
    assert (min((p["length"] for p in paths), default=None)) == expected


# ---------------------------------------------------------------------------
# Exact-value regressions the oracle cannot express
# ---------------------------------------------------------------------------


class TestBatchHonoursFilters:
    """The batch API answered a Person-to-Person query through a City."""

    def test_batch_connection_types(self, path_graph):
        assert path_graph.shortest_path_lengths_batch("Person", [(3, 4)]) == [2]
        assert path_graph.shortest_path_lengths_batch("Person", [(3, 4)], connection_types=["KNOWS"]) == [None]

    def test_batch_via_types(self, path_graph):
        # via_types restricts the *intermediate* node types; the City hop dies.
        assert path_graph.shortest_path_lengths_batch("Person", [(3, 4)], via_types=["Person"]) == [None]
        assert path_graph.shortest_path_lengths_batch("Person", [(3, 4)], via_types=["Person", "City"]) == [2]

    def test_batch_direction(self, path_graph):
        assert path_graph.shortest_path_lengths_batch("Person", [(1, 4)]) == [2]
        assert path_graph.shortest_path_lengths_batch("Person", [(1, 4)], direction="outgoing") == [3]
        assert path_graph.shortest_path_lengths_batch("Person", [(1, 4)], direction="incoming") == [None]

    def test_batch_endpoint_outside_restricted_universe_is_none(self, path_graph):
        """A pair whose endpoint the filters exclude answers None, never errors."""
        # Carol (3) has only a LIVES_IN edge, so a KNOWS-only universe has no
        # vertex for her at all.
        assert path_graph.shortest_path_lengths_batch("Person", [(3, 1)], connection_types=["KNOWS"]) == [None]
        assert path_graph.shortest_path_lengths_batch("Person", [(1, 2), (3, 1)], connection_types=["KNOWS"]) == [
            1,
            None,
        ]

    def test_batch_self_pair(self, path_graph):
        assert path_graph.shortest_path_lengths_batch("Person", [(3, 3)], connection_types=["KNOWS"]) == [0]


class TestViaTypesFamilyWide:
    def test_via_types_blocks_the_city_hop(self, path_graph):
        assert path_graph.shortest_path_length("Person", 3, "Person", 4, via_types=["Person"]) is None
        assert path_graph.are_connected("Person", 3, "Person", 4, via_types=["Person"]) is False
        assert path_graph.shortest_path_length("Person", 3, "Person", 4, via_types=["Person", "City"]) == 2

    def test_via_types_does_not_constrain_endpoints(self, path_graph):
        """The endpoints are exempt from via_types — only the middle is filtered."""
        assert path_graph.shortest_path_length("Person", 3, "City", 10, via_types=["Person"]) == 1


class TestDirectionOnPathShape:
    def test_outgoing_path_has_no_backwards_edge(self, path_graph):
        """Undirected 1->4 uses KNOWS 4->2 backwards; outgoing must not."""
        undirected = path_graph.shortest_path("Person", 1, "Person", 4)
        assert [n["id"] for n in undirected["path"]] == [1, 2, 4]

        directed = path_graph.shortest_path("Person", 1, "Person", 4, direction="outgoing")
        assert [n["id"] for n in directed["path"]] == [1, 2, 5, 4]
        assert directed["connections"] == ["KNOWS", "KNOWS", "KNOWS"]

        # Every consecutive pair is a real forward edge.
        ids = [n["id"] for n in directed["path"]]
        for a, b in zip(ids, ids[1:]):
            rows = path_graph.cypher(
                f"MATCH (x:Person)-[r]->(y:Person) WHERE x.id = {a} AND y.id = {b} RETURN count(r) AS n"
            ).to_dicts()
            assert rows[0]["n"] >= 1, f"{a}->{b} is a backwards edge"

    def test_incoming_is_the_mirror_of_outgoing(self, path_graph):
        out = path_graph.shortest_path_ids("Person", 1, "Person", 4, direction="outgoing")
        inc = path_graph.shortest_path_ids("Person", 4, "Person", 1, direction="incoming")
        assert inc == list(reversed(out))

    def test_all_paths_direction(self, path_graph):
        undirected = path_graph.all_paths("Person", 1, "Person", 4, max_hops=5)
        assert sorted(p["length"] for p in undirected) == [2, 3]
        directed = path_graph.all_paths("Person", 1, "Person", 4, max_hops=5, direction="outgoing")
        assert sorted(p["length"] for p in directed) == [3]


class TestWeightedHonoursFilters:
    """weight_property silently discarded every filter it appeared to accept."""

    @pytest.fixture(scope="class")
    def weighted_graph(self) -> KnowledgeGraph:
        g = KnowledgeGraph()
        people = pd.DataFrame({"id": [1, 2, 3], "name": ["A", "B", "C"]})
        g.add_nodes(people, "Person", "id", "name")
        cities = pd.DataFrame({"id": [10], "name": ["Oslo"]})
        g.add_nodes(cities, "City", "id", "name")
        # A -> B -> C via KNOWS costs 10; A -> Oslo -> C via LIVES_IN costs 2.
        g.add_connections(
            pd.DataFrame({"src": [1, 2], "dst": [2, 3], "cost": [5.0, 5.0]}),
            "KNOWS",
            "Person",
            "src",
            "Person",
            "dst",
        )
        g.add_connections(
            pd.DataFrame({"src": [1, 3], "dst": [10, 10], "cost": [1.0, 1.0]}),
            "LIVES_IN",
            "Person",
            "src",
            "City",
            "dst",
        )
        return g

    def test_weighted_length_honours_connection_types(self, weighted_graph):
        assert weighted_graph.shortest_path_length("Person", 1, "Person", 3, weight_property="cost") == 2.0
        assert (
            weighted_graph.shortest_path_length(
                "Person", 1, "Person", 3, weight_property="cost", connection_types=["KNOWS"]
            )
            == 10.0
        )

    def test_weighted_length_honours_via_types(self, weighted_graph):
        assert (
            weighted_graph.shortest_path_length("Person", 1, "Person", 3, weight_property="cost", via_types=["Person"])
            == 10.0
        )

    def test_weighted_length_honours_direction(self, weighted_graph):
        # Both cheap edges point INTO Oslo, so an outgoing-only walk from A
        # cannot come back out of it; the KNOWS chain is the only directed route.
        assert (
            weighted_graph.shortest_path_length("Person", 1, "Person", 3, weight_property="cost", direction="outgoing")
            == 10.0
        )

    def test_weighted_path_honours_direction(self, weighted_graph):
        r = weighted_graph.shortest_path("Person", 1, "Person", 3, weight_property="cost")
        assert [n["id"] for n in r["path"]] == [1, 10, 3]
        r = weighted_graph.shortest_path("Person", 1, "Person", 3, weight_property="cost", direction="outgoing")
        assert [n["id"] for n in r["path"]] == [1, 2, 3]


class TestDirectionArgumentParsing:
    @pytest.mark.parametrize("alias", ["any", "both", None])
    def test_undirected_aliases(self, path_graph, alias):
        assert path_graph.shortest_path_length("Person", 1, "Person", 4, direction=alias) == 2

    @pytest.mark.parametrize("alias", ["outgoing", "out"])
    def test_outgoing_aliases(self, path_graph, alias):
        assert path_graph.shortest_path_length("Person", 1, "Person", 4, direction=alias) == 3

    @pytest.mark.parametrize("alias", ["incoming", "in"])
    def test_incoming_aliases(self, path_graph, alias):
        assert path_graph.shortest_path_length("Person", 1, "Person", 4, direction=alias) is None

    def test_bad_direction_is_an_error_not_a_silent_default(self, path_graph):
        with pytest.raises(Exception, match="Invalid direction"):
            path_graph.shortest_path_length("Person", 1, "Person", 4, direction="sideways")

    @pytest.mark.parametrize(
        "method",
        [
            "shortest_path",
            "shortest_path_ids",
            "shortest_path_indices",
            "shortest_path_length",
            "are_connected",
            "all_paths",
        ],
    )
    def test_every_member_rejects_a_bad_direction(self, path_graph, method):
        with pytest.raises(Exception, match="Invalid direction"):
            getattr(path_graph, method)("Person", 1, "Person", 4, direction="sideways")

    def test_batch_rejects_a_bad_direction(self, path_graph):
        with pytest.raises(Exception, match="Invalid direction"):
            path_graph.shortest_path_lengths_batch("Person", [(1, 4)], direction="sideways")


class TestTimeoutArgumentExists:
    """timeout_ms reached only three of the seven members."""

    @pytest.mark.parametrize(
        "call",
        [
            lambda g: g.shortest_path_length("Person", 1, "Person", 4, timeout_ms=5000),
            lambda g: g.shortest_path_lengths_batch("Person", [(1, 4)], timeout_ms=5000),
            lambda g: g.are_connected("Person", 1, "Person", 4, timeout_ms=5000),
        ],
    )
    def test_timeout_ms_accepted(self, path_graph, call):
        call(path_graph)


# ---------------------------------------------------------------------------
# One source, many targets (S3)
# ---------------------------------------------------------------------------
#
# The unreachable-representation contract, which is the whole reason this API
# has two shapes:
#
#   * target_ids given  -> an entry per REQUESTED id, None where unreachable.
#   * discovery mode    -> only what was reached; ABSENT means unreachable.
#
# Everything below is asserted against N single-pair ``shortest_path_length``
# calls on the same fixture, so the one-to-many member cannot drift from the
# pair member it exists to batch.

PEOPLE = [1, 2, 3, 4, 5, 6]

FROM_MATRIX = [
    (source, rel, direction)
    for source in [1, 3, 4, 6]
    for rel in [None, "KNOWS"]
    for direction in ["any", "outgoing", "incoming"]
]


def _from_id(cell) -> str:
    source, rel, direction = cell
    return f"from{source}-{rel or 'anyrel'}-{direction}"


def _kwargs(rel, direction, **extra):
    kwargs = {"direction": direction, **extra}
    if rel:
        kwargs["connection_types"] = [rel]
    return kwargs


class TestLengthsFromEqualsNSinglePairCalls:
    """The equality that defines the method: it is N pair calls, batched."""

    @pytest.mark.parametrize("cell", FROM_MATRIX, ids=_from_id)
    def test_discovery_mode_omits_unreachable(self, path_graph, cell):
        source, rel, direction = cell
        kwargs = _kwargs(rel, direction)
        got = path_graph.shortest_path_lengths_from("Person", source, "Person", **kwargs)
        expected = {
            target: path_graph.shortest_path_length("Person", source, "Person", target, **kwargs) for target in PEOPLE
        }
        # Absent == unreachable: the dict is the reachable half of `expected`.
        assert got == {k: v for k, v in expected.items() if v is not None}
        assert None not in got.values()

    @pytest.mark.parametrize("cell", FROM_MATRIX, ids=_from_id)
    def test_target_ids_mode_includes_none_for_unreachable(self, path_graph, cell):
        source, rel, direction = cell
        kwargs = _kwargs(rel, direction)
        got = path_graph.shortest_path_lengths_from("Person", source, "Person", PEOPLE, **kwargs)
        expected = {
            target: path_graph.shortest_path_length("Person", source, "Person", target, **kwargs) for target in PEOPLE
        }
        assert got == expected
        # Every requested id got an answer, even the unreachable ones.
        assert set(got) == set(PEOPLE)

    @pytest.mark.parametrize("cell", FROM_MATRIX, ids=_from_id)
    def test_via_types_matches_the_pair_member(self, path_graph, cell):
        source, rel, direction = cell
        kwargs = _kwargs(rel, direction, via_types=["Person"])
        got = path_graph.shortest_path_lengths_from("Person", source, "Person", PEOPLE, **kwargs)
        assert got == {
            target: path_graph.shortest_path_length("Person", source, "Person", target, **kwargs) for target in PEOPLE
        }

    @pytest.mark.parametrize("max_hops", [0, 1, 2, 3, 4, 10])
    def test_max_hops_truncates_rather_than_changing_the_answer(self, path_graph, max_hops):
        unbounded = path_graph.shortest_path_lengths_from("Person", 1, "Person")
        capped = path_graph.shortest_path_lengths_from("Person", 1, "Person", max_hops=max_hops)
        assert capped == {k: v for k, v in unbounded.items() if v <= max_hops}
        for target, hops in capped.items():
            assert hops == path_graph.shortest_path_length("Person", 1, "Person", target)

    def test_target_ids_beyond_max_hops_answer_none_not_absent(self, path_graph):
        """The bound is on the search; a requested id still gets an answer."""
        assert path_graph.shortest_path_lengths_from("Person", 1, "Person", [1, 2, 4], max_hops=1) == {
            1: 0,
            2: 1,
            4: None,
        }


class TestLengthsFromExactValues:
    def test_discovery_across_every_type(self, path_graph):
        # No target_type: every reached node, whatever its type. Oslo (10) is
        # 3 hops out, Carol (3) 4, and the Frank/Bergen component never appears.
        assert path_graph.shortest_path_lengths_from("Person", 1, max_hops=10) == {
            1: 0,
            2: 1,
            4: 2,
            5: 2,
            10: 3,
            3: 4,
        }

    def test_target_type_filters_the_result_not_the_walk(self, path_graph):
        """Carol is only reachable *through* the City — and still shows up."""
        people = path_graph.shortest_path_lengths_from("Person", 1, "Person")
        assert people == {1: 0, 2: 1, 4: 2, 5: 2, 3: 4}
        # Same walk, City-shaped answer.
        assert path_graph.shortest_path_lengths_from("Person", 1, "City") == {10: 3}
        # via_types is what restricts the walk: block the City and Carol dies,
        # while the City itself is still reported (a path end, not a hop).
        assert path_graph.shortest_path_lengths_from("Person", 1, max_hops=10, via_types=["Person"]) == {
            1: 0,
            2: 1,
            4: 2,
            5: 2,
            10: 3,
        }

    def test_source_is_present_at_zero(self, path_graph):
        assert path_graph.shortest_path_lengths_from("Person", 1, "Person")[1] == 0
        assert path_graph.shortest_path_lengths_from("Person", 1, "Person", [1]) == {1: 0}
        assert path_graph.shortest_path_lengths_from("Person", 1, max_hops=0) == {1: 0}
        # ...unless it is out of the result's scope.
        assert 1 not in path_graph.shortest_path_lengths_from("Person", 1, "City")

    def test_direction(self, path_graph):
        assert path_graph.shortest_path_lengths_from("Person", 1, "Person", direction="outgoing") == {
            1: 0,
            2: 1,
            5: 2,
            4: 3,
        }
        # Nothing points at Alice.
        assert path_graph.shortest_path_lengths_from("Person", 1, "Person", direction="incoming") == {1: 0}

    def test_connection_types(self, path_graph):
        assert path_graph.shortest_path_lengths_from("Person", 1, "Person", connection_types=["KNOWS"]) == {
            1: 0,
            2: 1,
            4: 2,
            5: 2,
        }

    def test_isolated_component(self, path_graph):
        assert path_graph.shortest_path_lengths_from("Person", 6, "Person") == {6: 0}
        assert path_graph.shortest_path_lengths_from("Person", 6, "Person", PEOPLE) == {
            1: None,
            2: None,
            3: None,
            4: None,
            5: None,
            6: 0,
        }

    def test_target_ids_namespace_defaults_to_source_type(self, path_graph):
        assert path_graph.shortest_path_lengths_from("Person", 1, target_ids=[4, 6]) == {4: 2, 6: None}
        # ...and target_type names it explicitly.
        assert path_graph.shortest_path_lengths_from("Person", 1, "City", [10, 11]) == {10: 3, 11: None}

    def test_key_order_is_the_request_order_then_distance_order(self, path_graph):
        assert list(path_graph.shortest_path_lengths_from("Person", 1, "Person", [5, 1, 6, 2])) == [5, 1, 6, 2]
        hops = list(path_graph.shortest_path_lengths_from("Person", 1, "Person").values())
        assert hops == sorted(hops)


class TestLengthsFromRejections:
    def test_unbounded_one_to_all_is_refused_by_name(self, path_graph):
        with pytest.raises(Exception) as exc:
            path_graph.shortest_path_lengths_from("Person", 1)
        message = str(exc.value)
        for bound in ("target_ids", "target_type", "max_hops"):
            assert bound in message, f"the refusal must name {bound}: {message}"

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"target_type": "Person"},
            {"target_ids": [2]},
            {"max_hops": 2},
        ],
    )
    def test_any_one_bound_is_enough(self, path_graph, kwargs):
        path_graph.shortest_path_lengths_from("Person", 1, **kwargs)

    def test_unknown_source_id(self, path_graph):
        with pytest.raises(Exception, match="Source node with id"):
            path_graph.shortest_path_lengths_from("Person", 999, "Person")

    def test_unknown_target_id(self, path_graph):
        with pytest.raises(Exception, match="Target node with id"):
            path_graph.shortest_path_lengths_from("Person", 1, "Person", [2, 999])

    def test_unknown_target_type(self, path_graph):
        with pytest.raises(Exception, match="Unknown target_type"):
            path_graph.shortest_path_lengths_from("Person", 1, "Perosn")

    def test_bad_direction(self, path_graph):
        with pytest.raises(Exception, match="Invalid direction"):
            path_graph.shortest_path_lengths_from("Person", 1, "Person", direction="sideways")

    def test_timeout_ms_accepted(self, path_graph):
        path_graph.shortest_path_lengths_from("Person", 1, "Person", timeout_ms=5000)

    def test_colliding_ids_across_types_raise_rather_than_collapse(self):
        """Ids are unique per type, not across types."""
        g = KnowledgeGraph()
        g.add_nodes(pd.DataFrame({"id": [1], "name": ["A"]}), "Person", "id", "name")
        g.add_nodes(pd.DataFrame({"id": [1], "name": ["Oslo"]}), "City", "id", "name")
        g.add_connections(pd.DataFrame({"s": [1], "t": [1]}), "LIVES_IN", "Person", "s", "City", "t")
        # Person 1 and City 1 both reached, both keyed `1`.
        with pytest.raises(Exception, match="sharing id"):
            g.shortest_path_lengths_from("Person", 1, max_hops=3)
        # Naming the namespace resolves it.
        assert g.shortest_path_lengths_from("Person", 1, "City", max_hops=3) == {1: 1}
        assert g.shortest_path_lengths_from("Person", 1, "Person", max_hops=3) == {1: 0}
