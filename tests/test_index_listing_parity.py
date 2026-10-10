"""``has_index()`` and ``list_indexes()`` report an equality index in every
storage mode, including the persistent bundle a disk graph builds.

``create_index`` on a disk graph returns ``persistent: True``; both calls used
to answer as though no index existed.
"""

import pandas as pd
import pytest

import kglite

pytestmark = pytest.mark.parity

MODES = ["memory", "mapped", "disk", "disk_reopened"]


def _graph(mode, tmp_path):
    if mode == "memory":
        graph = kglite.KnowledgeGraph()
    elif mode == "mapped":
        graph = kglite.KnowledgeGraph(storage="mapped")
    else:
        graph = kglite.KnowledgeGraph(storage="disk", path=str(tmp_path / "g"))
    graph.add_nodes(
        pd.DataFrame({"nid": [1, 2, 3], "name": ["a", "b", "c"], "city": ["x", "y", "x"]}),
        "Person",
        "nid",
        "name",
    )
    return graph


@pytest.mark.parametrize("mode", MODES)
def test_an_index_is_reported_by_has_index_and_list_indexes(mode, tmp_path):
    graph = _graph(mode, tmp_path)
    assert graph.has_index("Person", "city") is False
    assert graph.list_indexes() == []
    info = graph.create_index("Person", "city")
    if mode in ("disk", "disk_reopened"):
        graph.save()
    if mode == "disk_reopened":
        del graph
        graph = kglite.load(str(tmp_path / "g"))
    assert graph.has_index("Person", "city") is True
    listed = [i for i in graph.list_indexes() if (i["node_type"], i["property"]) == ("Person", "city")]
    assert len(listed) == 1, graph.list_indexes()
    assert listed[0]["persistent"] is (mode in ("disk", "disk_reopened"))
    assert listed[0]["state"] == "ONLINE"
    if mode == "disk":
        assert info["persistent"] is True
    assert graph.has_index("Person", "name") is False
