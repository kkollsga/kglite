"""`valid_at` / `valid_during` refuse a bound property the type does not have.

A null bound is open, so a misspelled property name (`'validfrom'`) read null on
every row and answered the query as if that side were unbounded — silently,
with a plausible count. A property that no element of the type has is now an
error naming it, as `db.temporal.declare` already refused one; a property the
type has but one row leaves null stays open.
"""

from __future__ import annotations

import pandas as pd
import pytest

import kglite


@pytest.fixture
def graph() -> kglite.KnowledgeGraph:
    g = kglite.KnowledgeGraph()
    g.cypher(
        "CREATE (:M {id: 1, vf: date('2000-01-01'), vt: date('2010-01-01')}),"
        " (:M {id: 2, vf: date('2012-01-01')}), (:M {id: 3})"
    ).to_list()
    return g


@pytest.mark.parametrize("disable_optimizer", [False, True])
def test_a_null_bound_on_a_known_property_stays_open(graph, disable_optimizer) -> None:
    rows = graph.cypher(
        "MATCH (m:M) WHERE valid_at(m, date('2005-01-01'), 'vf', 'vt') RETURN m.id AS id ORDER BY id",
        disable_optimizer=disable_optimizer,
    ).to_list()
    assert rows == [{"id": 1}, {"id": 3}]


@pytest.mark.parametrize("disable_optimizer", [False, True])
@pytest.mark.parametrize(
    ("query", "missing"),
    [
        ("MATCH (m:M) WHERE valid_at(m, date('2005-01-01'), 'vfx', 'vt') RETURN count(*) AS c", "vfx"),
        ("MATCH (m:M) WHERE valid_at(m, date('2005-01-01'), 'vf', 'vtx') RETURN m.id AS id", "vtx"),
        (
            "MATCH (m:M) WHERE valid_during(m, date('2005-01-01'), date('2006-01-01'), 'validfrom', 'vt')"
            " RETURN count(*) AS c",
            "validfrom",
        ),
    ],
)
def test_a_property_no_node_has_is_refused(graph, query, missing, disable_optimizer) -> None:
    with pytest.raises(kglite.CypherExecutionError, match=f"property '{missing}' does not exist on node type 'M'"):
        graph.cypher(query, disable_optimizer=disable_optimizer).to_list()


def test_relationships_are_checked_and_declared_bounds_nobody_set_stay_open() -> None:
    g = kglite.KnowledgeGraph()
    g.add_nodes(pd.DataFrame({"id": ["a", "b"]}), "M", "id")
    # No period has ended yet: the valid_to column is all null, so no stored
    # relationship holds it — the declaration is what makes it known.
    g.add_relationships(
        pd.DataFrame({"s": ["a"], "t": ["b"], "vf": ["2000-01-01"], "vt": [None]}),
        "R",
        "M",
        "s",
        "M",
        "t",
        column_types={"vf": "validFrom", "vt": "validTo"},
    )
    ok = g.cypher("MATCH ()-[r:R]->() WHERE valid_at(r, date('2005-01-01'), 'vf', 'vt') RETURN count(*) AS c")
    assert ok.to_list() == [{"c": 1}]
    with pytest.raises(kglite.CypherExecutionError, match="property 'vtx' does not exist on relationship type 'R'"):
        g.cypher("MATCH ()-[r:R]->() WHERE valid_at(r, date('2005-01-01'), 'vf', 'vtx') RETURN count(*) AS c").to_list()
