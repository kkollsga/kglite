"""Result column names do not depend on the plan or on the row count.

A count short-circuit rendered an unaliased `count(*)` as `count(Star)` —
`MATCH (n) RETURN count(*)` and `RETURN type(r), count(*)` did, while a
filtered count said `count(*)`. And `RETURN *` over zero rows reported the
single column `*`, because the names it stands for were read off the first
row. Both are absolute: the unoptimised path shared the `*` one.
"""

from __future__ import annotations

import pytest

import kglite


@pytest.fixture
def graph() -> kglite.KnowledgeGraph:
    g = kglite.KnowledgeGraph()
    g.cypher("CREATE (:M {name: 'a'})-[:R]->(:P {name: 'b'})").to_list()
    return g


@pytest.mark.parametrize("disable_optimizer", [False, True])
@pytest.mark.parametrize(
    ("query", "columns"),
    [
        ("MATCH (n) RETURN count(*)", ["count(*)"]),
        ("MATCH (n:M) RETURN count(*)", ["count(*)"]),
        ("MATCH ()-[r]->() RETURN type(r), count(*)", ["type(r)", "count(*)"]),
        ("MATCH ()-[r]->() RETURN count(*)", ["count(*)"]),
        ("MATCH (n) RETURN labels(n)[0], count(*)", ["labels(n)[0]", "count(*)"]),
        ("MATCH (n:M) WHERE n.name = 'zzz' RETURN *", ["n"]),
        ("MATCH (n:M)-[r:R]->(m) WHERE n.name = 'zzz' RETURN *", ["n", "r", "m"]),
        ("MATCH (n:M) WHERE n.name = 'zzz' WITH n, 1 AS k RETURN *, k + 1 AS j", ["n", "k", "j"]),
        ("UNWIND [] AS x RETURN *", ["x"]),
    ],
)
def test_column_names(graph, query, columns, disable_optimizer) -> None:
    assert list(graph.cypher(query, disable_optimizer=disable_optimizer).columns) == columns
