"""Native list properties (P3 — list ingestion as ColumnData::List).

A pandas column of Python lists is ingested as a real list-valued property,
not a stringified `"['x', 'y']"`. The load-bearing guarantee: `IN` tests
*membership* over the elements (true `'y' IN ['x','y']`), with no
false-positive substring match (`'xy'` is not a member), and `UNWIND`
yields the individual elements.
"""

import pandas as pd
import pytest

import kglite


@pytest.fixture
def kg_with_lists():
    kg = kglite.KnowledgeGraph()
    df = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "name": ["a", "b", "c"],
            "aliases": [["x", "y"], ["z"], []],
        }
    )
    kg.add_nodes(df, node_type="Person", unique_id_field="id")
    return kg


def test_list_stored_as_native_list(kg_with_lists):
    rows = kg_with_lists.cypher("MATCH (n:Person) RETURN n.id AS id, n.aliases AS al ORDER BY id").to_dicts()
    assert rows[0]["al"] == ["x", "y"]
    assert rows[1]["al"] == ["z"]
    assert rows[2]["al"] == []


def test_in_is_membership_not_substring(kg_with_lists):
    hit = kg_with_lists.cypher("MATCH (n:Person) WHERE 'y' IN n.aliases RETURN n.id AS id ORDER BY id").to_dicts()
    assert hit == [{"id": 1}]

    # 'xy' is a substring of the stringified list but NOT a member — the
    # whole point of native lists is that this returns nothing.
    miss = kg_with_lists.cypher("MATCH (n:Person) WHERE 'xy' IN n.aliases RETURN n.id AS id").to_dicts()
    assert miss == []


def test_unwind_over_stored_list(kg_with_lists):
    rows = kg_with_lists.cypher("MATCH (n:Person {id:1}) UNWIND n.aliases AS a RETURN a ORDER BY a").to_dicts()
    assert [r["a"] for r in rows] == ["x", "y"]


def test_explicit_list_column_type():
    kg = kglite.KnowledgeGraph()
    # Force list typing via column_types even though the cells look scalar-ish.
    df = pd.DataFrame({"id": [1], "tags": [["a", "b", "c"]]})
    kg.add_nodes(df, node_type="T", unique_id_field="id", column_types={"tags": "list"})
    rows = kg.cypher("MATCH (n:T) RETURN n.tags AS t").to_dicts()
    assert rows[0]["t"] == ["a", "b", "c"]


def test_mixed_int_list_roundtrips():
    kg = kglite.KnowledgeGraph()
    df = pd.DataFrame({"id": [1], "scores": [[1, 2, 3]]})
    kg.add_nodes(df, node_type="S", unique_id_field="id")
    rows = kg.cypher("MATCH (n:S) UNWIND n.scores AS s RETURN s ORDER BY s").to_dicts()
    assert [r["s"] for r in rows] == [1, 2, 3]


# ── Cross-mode parity (Phase 2): lists survive every storage backend and
#    the persistence round-trips (.kgl save/load + streaming-disk subset). ──

_EXPECTED = [{"id": 1, "al": ["x", "y"]}, {"id": 2, "al": ["z"]}]
_Q = "MATCH (n:Person) RETURN n.id AS id, n.aliases AS al ORDER BY id"


def _person_df():
    return pd.DataFrame({"id": [1, 2], "name": ["a", "b"], "aliases": [["x", "y"], ["z"]]})


@pytest.mark.parametrize("mode", ["memory", "mapped", "disk"])
def test_list_property_cross_mode_live(mode, tmp_path):
    if mode == "memory":
        kg = kglite.KnowledgeGraph()
    elif mode == "mapped":
        kg = kglite.KnowledgeGraph(storage="mapped")
    else:
        kg = kglite.KnowledgeGraph(storage="disk", path=str(tmp_path / "g"))
    kg.add_nodes(_person_df(), node_type="Person", unique_id_field="id")
    assert kg.cypher(_Q).to_dicts() == _EXPECTED


@pytest.mark.parametrize("mode", ["memory", "mapped", "disk"])
def test_list_property_save_reload(mode, tmp_path):
    if mode == "memory":
        kg = kglite.KnowledgeGraph()
    elif mode == "mapped":
        kg = kglite.KnowledgeGraph(storage="mapped")
    else:
        kg = kglite.KnowledgeGraph(storage="disk", path=str(tmp_path / "g"))
    kg.add_nodes(_person_df(), node_type="Person", unique_id_field="id")
    p = str(tmp_path / f"{mode}.kgl")
    kg.save(p)
    assert kglite.load(p).cypher(_Q).to_dicts() == _EXPECTED


def test_list_property_streaming_disk_subset(tmp_path):
    # save_subset on a disk-backed source takes the streaming-disk writer,
    # which marshals each property through the borrowed-value overflow path.
    # Before Phase 2 a list there serialized to NULL; now it round-trips.
    src = kglite.KnowledgeGraph(storage="disk", path=str(tmp_path / "src"))
    src.add_nodes(_person_df(), node_type="Person", unique_id_field="id")
    out = str(tmp_path / "subset.kgl")
    src.select("Person").save_subset(out)
    assert kglite.load(out).cypher(_Q).to_dicts() == _EXPECTED
