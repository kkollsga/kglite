"""`push_limit_into_aggregate`'s group cap must not drop rows from groups it keeps.

The pass stamps `group_limit_hint = N` on a `RETURN`/`WITH` that groups,
aggregates and is followed by a bare `LIMIT N`. Both aggregation paths then
freeze their group set at N and skip later rows *that would open an N+1th
group* — rows for a group already collected must still reach its aggregate,
or the answer is silently short.

That contract broke for the shape the pass was written for. The group set is
keyed by a **surrogate**: a group key of the form `p.city` is held as the
bound `NodeIndex` and resolved to a value only after the row pass, so one
resolved group can be spread over arbitrarily many surrogates. Capping the
*surrogate* set therefore drops rows from groups that were already collected:
30 `:Person` nodes across 3 cities answered `count(*) = 5` for Oslo where 10
was the truth, and `collect(p.name)` returned 5 of the 10 names — with no
error and no flag set.

The fixtures below are sized past the cap on purpose: the cap engages only
once the row pass has opened more surrogate groups than the limit allows, so
a handful of nodes cannot see this at all.

Every query is run on four arms — the default plan, the materialized
aggregate (`streaming=False`), and both again with the pass disabled. The
pass-disabled arms are the oracle: they compute the same answer with no cap
in play, so an arm that disagrees with them has been capped wrongly.
"""

from __future__ import annotations

import pandas as pd
import pytest

import kglite

CITIES = ("Oslo", "Bergen", "Tromso")
PER_CITY = 10

#: Insertion order decides which groups a `LIMIT` keeps, so the expected rows
#: below are fixed by construction rather than by the engine's answer.
EXPECTED_FIRST = ("Oslo", PER_CITY)
EXPECTED_SECOND = ("Bergen", PER_CITY)

ARMS = {
    "default": {},
    "materialized": {"streaming": False},
    "hint-disabled": {"disabled_passes": ["push_limit_into_aggregate"]},
    "materialized+hint-disabled": {
        "streaming": False,
        "disabled_passes": ["push_limit_into_aggregate"],
    },
}

#: The projecting `WITH` keeps these off `fuse_node_scan_aggregate`, which
#: absorbs `MATCH (n:T) RETURN n.k, count(*) LIMIT n` whole and never consults
#: the hint — without it neither aggregation path runs and the cap is
#: unreachable.
NODEPROP_KEY = "MATCH (p:Person) WITH p, p.name AS nm RETURN p.city AS city, count(*) AS n LIMIT {n}"
EVAL_KEY = "MATCH (p:Person) WITH p.city AS c, p.name AS nm RETURN c AS city, count(*) AS n LIMIT {n}"
WITH_NODEPROP_KEY = "MATCH (p:Person) WITH p, p.name AS nm WITH p.city AS city, count(*) AS n LIMIT {n} RETURN city, n"
COLLECTING = (
    "MATCH (p:Person) WITH p, p.name AS nm RETURN p.city AS city, count(*) AS n, collect(p.name) AS names LIMIT {n}"
)

COUNTING_QUERIES = {
    "nodeprop_key": NODEPROP_KEY,
    "eval_key": EVAL_KEY,
    "with_nodeprop_key": WITH_NODEPROP_KEY,
}


@pytest.fixture
def city_graph() -> kglite.KnowledgeGraph:
    """30 `:Person` nodes, 3 cities, 10 each, round-robin so every city is
    represented in the first three rows and the cap cannot separate them."""
    rows = [{"nid": i, "name": f"P{i:02d}", "city": CITIES[i % len(CITIES)]} for i in range(len(CITIES) * PER_CITY)]
    graph = kglite.KnowledgeGraph()
    graph.add_nodes(pd.DataFrame(rows), "Person", "nid", "name", columns=["city"])
    return graph


@pytest.mark.parametrize("arm", sorted(ARMS))
@pytest.mark.parametrize("shape", sorted(COUNTING_QUERIES))
def test_group_cap_keeps_every_row_of_the_groups_it_keeps(city_graph, shape, arm):
    query = COUNTING_QUERIES[shape].format(n=1)
    rows = city_graph.cypher(query, **ARMS[arm]).to_list()
    assert rows == [{"city": EXPECTED_FIRST[0], "n": EXPECTED_FIRST[1]}]


@pytest.mark.parametrize("arm", sorted(ARMS))
@pytest.mark.parametrize("shape", sorted(COUNTING_QUERIES))
def test_group_cap_keeps_the_second_group_whole(city_graph, shape, arm):
    query = COUNTING_QUERIES[shape].format(n=2)
    rows = city_graph.cypher(query, **ARMS[arm]).to_list()
    assert rows == [
        {"city": EXPECTED_FIRST[0], "n": EXPECTED_FIRST[1]},
        {"city": EXPECTED_SECOND[0], "n": EXPECTED_SECOND[1]},
    ]


@pytest.mark.parametrize("arm", sorted(ARMS))
def test_group_cap_does_not_shorten_collect(city_graph, arm):
    """`collect()` disqualifies the streaming pipeline, so this shape reaches
    the materialized aggregate on every arm — the path the cap was written
    for, and the one the Wikidata hub-anchor case in the pass's own docstring
    lands on."""
    rows = city_graph.cypher(COLLECTING.format(n=1), **ARMS[arm]).to_list()
    assert len(rows) == 1
    assert rows[0]["city"] == EXPECTED_FIRST[0]
    assert rows[0]["n"] == PER_CITY
    assert sorted(rows[0]["names"]) == sorted(
        f"P{i:02d}" for i in range(len(CITIES) * PER_CITY) if CITIES[i % len(CITIES)] == EXPECTED_FIRST[0]
    )


@pytest.mark.parametrize("arm", sorted(ARMS))
def test_limit_above_group_count_returns_every_group(city_graph, arm):
    """A limit the group set never reaches must leave all three groups whole —
    the cap has to be a no-op here, not an off-by-one."""
    rows = city_graph.cypher(NODEPROP_KEY.format(n=10), **ARMS[arm]).to_list()
    assert sorted((row["city"], row["n"]) for row in rows) == sorted((city, PER_CITY) for city in CITIES)
