"""A value at the nesting ceiling (256 levels) survives save and load.

Values nested deeper than the ceiling cannot be built through the engine, so
the refusal of a too-deep stored value is pinned in the Rust tests, which
craft the stored bytes directly.
"""

import pytest

import kglite

DEPTH = 256


def _depth(value):
    depth = 0
    while isinstance(value, list):
        depth += 1
        value = value[0]
    return depth


@pytest.mark.parametrize("storage", ["default", "mapped"])
def test_ceiling_depth_property_round_trips(tmp_path, storage):
    kwargs = {} if storage == "default" else {"storage": storage}
    g = kglite.KnowledgeGraph(**kwargs)
    g.cypher(f"CREATE (:Doc {{id: 1, title: 'a', deep: reduce(acc = 0, i IN range(1, {DEPTH}) | [acc])}})")
    path = str(tmp_path / "deep.kgl")
    g.save(path)
    loaded = kglite.load(path)
    rows = loaded.cypher("MATCH (d:Doc) RETURN d.deep AS deep").to_list()
    assert _depth(rows[0]["deep"]) == DEPTH
