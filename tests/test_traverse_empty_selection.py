"""Traversing from an empty selection answers an empty selection.

`select(...).where(...)` that matches nothing used to make the next
`traverse()` raise `No source nodes available for traversal`, which broke any
per-group loop that meets an empty group (a province with no members). Every
other fluent step keeps chaining on an empty selection; `traverse` now does too.
An unknown relationship type is still an error.
"""

from __future__ import annotations

import pytest

import kglite


@pytest.fixture
def graph() -> kglite.KnowledgeGraph:
    g = kglite.KnowledgeGraph()
    g.cypher("CREATE (:M {name: 'a', since: date('2000-01-01')})-[:R {since: date('2000-01-01')}]->(:P {name: 'b'})")
    return g


def test_traverse_from_nothing_is_empty_and_chains(graph) -> None:
    empty = graph.select("M").where({"name": "zzz"})
    assert empty.len() == 0
    step = empty.traverse("R")
    assert step.len() == 0
    assert step.traverse("R", direction="incoming").len() == 0
    assert empty.traverse("R", where={"name": "b"}).len() == 0
    assert empty.traverse("R", direction="outgoing", during=("1990", "2020")).len() == 0


def test_the_non_empty_traversal_is_unchanged(graph) -> None:
    assert graph.select("M").traverse("R").len() == 1


def test_an_unknown_relationship_type_still_raises(graph) -> None:
    with pytest.raises(Exception, match="does not exist"):
        graph.select("M").where({"name": "zzz"}).traverse("NOPE")


@pytest.mark.parametrize(
    "method",
    [
        "contains",
        "intersects",
        {"type": "distance", "max_m": 5000},
        {"type": "cluster", "algorithm": "kmeans", "features": ["x"], "k": 2},
    ],
    ids=["contains", "intersects", "distance", "cluster"],
)
def test_compare_from_nothing_is_empty(graph, method) -> None:
    assert graph.select("M").where({"name": "zzz"}).compare("P", method).len() == 0


def test_compare_from_nothing_still_checks_its_arguments(graph) -> None:
    with pytest.raises(Exception, match="Unknown clustering algorithm"):
        graph.select("M").where({"name": "zzz"}).compare(
            "P", {"type": "cluster", "algorithm": "nope", "features": ["x"]}
        )
