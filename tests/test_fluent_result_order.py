"""Fluent traversal results come back in a fixed order.

Children of each parent are in creation order and parents are in creation
order, in every storage mode. The order used to follow a per-process hash seed,
so two fresh processes over the same graph could list the same nodes
differently.
"""

import subprocess
import sys

import pytest

PROCESSES = 12

SCRIPT = """
import sys

mode, path = sys.argv[1], sys.argv[2]
if mode == "memory":
    g = kglite.KnowledgeGraph()
elif mode == "mapped":
    g = kglite.KnowledgeGraph(storage="mapped")
else:
    g = kglite.KnowledgeGraph(storage="disk", path=path)
# Creation order is deliberately not id order, and parents are created
# out of id order too.
g.cypher("CREATE (:P {id: 20, title: 'p20'}), (:P {id: 10, title: 'p10'}), (:P {id: 30, title: 'p30'})")
for pid, kids in ((10, (7, 3, 9, 1, 5)), (20, (8, 2, 6)), (30, (4,))):
    for k in kids:
        g.cypher(
            f"MATCH (p:P {{id: {pid}}}) CREATE (p)-[:HAS]->(:C {{id: {pid * 100 + k}, title: 'c{pid}_{k}'}})"
        )
print(g.select("P").traverse("HAS").titles())
print(g.select("P").traverse("HAS").traverse("HAS", direction="incoming").titles())
"""


def _run(mode, tmp_path, n):
    out = subprocess.run(
        [sys.executable, "-I", "-c", SCRIPT, mode, str(tmp_path / f"g{n}")],
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout


@pytest.mark.parametrize("mode", ["memory", "mapped", "disk"])
def test_traversal_titles_are_identical_across_fresh_processes(mode, tmp_path):
    outputs = {_run(mode, tmp_path, n) for n in range(PROCESSES)}
    assert len(outputs) == 1, outputs


@pytest.mark.parametrize("mode", ["memory", "mapped", "disk"])
def test_traversal_order_is_creation_order_per_parent_and_across_parents(mode, tmp_path):
    first = _run(mode, tmp_path, 0).splitlines()[0]
    expected = {
        "p20": ["c20_8", "c20_2", "c20_6"],
        "p10": ["c10_7", "c10_3", "c10_9", "c10_1", "c10_5"],
        "p30": ["c30_4"],
    }
    assert first == str(expected)
