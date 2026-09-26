"""`KnowledgeGraph(storage='disk', path=p).save()` saves back into `p`.

A disk graph lives in the directory it was built at, exactly as one opened with
`kglite.load(p)` does, but a bare `save()` refused with "needs a path". The
constructor now records that directory as the graph's origin.
"""

from __future__ import annotations

import pytest

import kglite


def test_a_disk_graph_saves_to_its_own_directory(tmp_path) -> None:
    path = str(tmp_path / "g")
    graph = kglite.KnowledgeGraph(storage="disk", path=path)
    graph.cypher("CREATE (:M {x: 1}), (:M {x: 2})").to_list()
    graph.save()
    assert kglite.load(path).cypher("MATCH (m:M) RETURN count(*) AS c").to_list() == [{"c": 2}]


def test_an_in_memory_graph_still_needs_a_path() -> None:
    with pytest.raises(ValueError, match="needs a path"):
        kglite.KnowledgeGraph().save()
