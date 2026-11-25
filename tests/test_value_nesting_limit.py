"""Runtime values nest at most 256 levels.

A query can build depth at runtime (``reduce`` with a wrapping body, repeated
``SET n.p = [n.p]``). Clone, Drop, comparison and every converter recurse once
per level, so an unbounded value overflowed the stack and killed the process.
Every construct that wraps a value in a new list or map now refuses to go past
256 levels with an ordinary query error. Depth 256 stays valid.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from kglite import KnowledgeGraph

LIMIT = 256


def depth_of(value) -> int:
    depth = 0
    while isinstance(value, list | dict):
        if isinstance(value, list):
            value = value[0]
        else:
            value = next(iter(value.values()))
        depth += 1
    return depth


@pytest.fixture
def graph() -> KnowledgeGraph:
    g = KnowledgeGraph()
    g.cypher("CREATE (:A {id: 1})-[:R]->(:B {id: 2})")
    return g


def _queries(n: int) -> dict[str, str]:
    """Each shape builds a value of depth ``n`` from a scalar."""
    return {
        "reduce_list": f"RETURN reduce(acc = 1, i IN range(1, {n}) | [acc]) AS x",
        "reduce_map": f"RETURN reduce(acc = 1, i IN range(1, {n}) | {{a: acc}}) AS x",
        "with_chain": "WITH 1 AS x " + "WITH [x] AS x " * n + "RETURN x",
        "collect_chain": "UNWIND [1] AS x " + "WITH collect(x) AS x " * n + "RETURN x",
        "comprehension": f"RETURN reduce(acc = 1, i IN range(1, {n}) | [a IN [acc] | a]) AS x",
        "pattern_comprehension": "MATCH (a:A) WITH a, 1 AS x " + "WITH a, [(a)-[:R]->(b) | x] AS x " * n + "RETURN x",
        "list_plus_element": f"RETURN reduce(acc = 1, i IN range(1, {n}) | [] + [acc]) AS x",
    }


SHAPES = list(_queries(1))


@pytest.mark.parametrize("shape", SHAPES)
def test_depth_limit_is_inclusive(graph, shape):
    rows = graph.cypher(_queries(LIMIT)[shape])
    assert depth_of(rows[0]["x"]) == LIMIT
    with pytest.raises(Exception, match=str(LIMIT)):
        graph.cypher(_queries(LIMIT + 1)[shape])


def test_repeated_set_property_wrap(graph):
    graph.cypher("MATCH (n:A) SET n.p = 1")
    for _ in range(LIMIT):
        graph.cypher("MATCH (n:A) SET n.p = [n.p]")
    assert depth_of(graph.cypher("MATCH (n:A) RETURN n.p AS x")[0]["x"]) == LIMIT
    with pytest.raises(Exception, match=str(LIMIT)):
        graph.cypher("MATCH (n:A) SET n.p = [n.p]")
    # The refused write left the stored value as it was.
    assert depth_of(graph.cypher("MATCH (n:A) RETURN n.p AS x")[0]["x"]) == LIMIT


def test_deep_reduce_does_not_kill_the_process():
    code = (
        "import kglite\n"
        "g = kglite.KnowledgeGraph()\n"
        "try:\n"
        "    g.cypher('RETURN reduce(acc = 1, i IN range(1, 100000) | [acc]) AS x')\n"
        "except Exception as e:\n"
        "    print('ERR', e)\n"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    assert "ERR" in done.stdout and "256" in done.stdout
