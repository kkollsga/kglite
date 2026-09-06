"""Output frames must preserve the exact cells before any dtype conversion."""

import datetime as dt

import pandas as pd
import pytest

import kglite
from tests.value_assertions import assert_value_equal


@pytest.mark.parametrize(
    "values",
    [
        [9007199254740993, None],
        [2**63 - 1, None],
        [-(2**63), None],
        [9007199254740993, 1.5],
        [1, True],
        [1, "1"],
        [[1, None], {"x": 2}],
        [dt.datetime(2025, 1, 2, 3, 4, 5, 123456), None],
    ],
)
@pytest.mark.parametrize("direct", [False, True])
def test_result_frame_preserves_exact_cells(values, direct):
    graph = kglite.KnowledgeGraph()
    query, params = "UNWIND $values AS v RETURN v", {"values": values}
    assert_value_equal(graph.cypher(query, params=params).column("v"), values)
    frame = graph.cypher(query, params=params, to_df=True) if direct else graph.cypher(query, params=params).to_df()
    actual = frame["v"].tolist()
    for given, result in zip(values, actual, strict=True):
        if given is None:
            assert pd.isna(result)
        elif type(given) is int:
            assert not isinstance(result, (float, bool))
            assert int(result) == given
        elif isinstance(given, dt.datetime):
            assert result == given
        else:
            assert_value_equal(result, given)


def test_table_frame_restores_values_before_dtype_and_after_save(tmp_path):
    graph = kglite.KnowledgeGraph()
    graph.cypher("CREATE (:N{id:1})")
    frame = pd.DataFrame(
        {
            "v": pd.Series([9007199254740993, None, -(2**63)], dtype="Int64"),
            "flag": pd.Series([True, None, False], dtype="boolean"),
        }
    )
    graph.set_table_property("N", 1, "items", frame)
    path = tmp_path / "table.kgl"
    graph.save(str(path))
    for owner in [graph, kglite.load(str(path))]:
        actual = owner.get_table_property("N", 1, "items")
        assert list(actual.columns) == ["v", "flag"]
        assert str(actual.v.dtype) == "Int64"
        assert int(actual.v.iloc[0]) == 9007199254740993
        assert int(actual.v.iloc[2]) == -(2**63)
        assert pd.isna(actual.v.iloc[1])
        assert str(actual.flag.dtype) == "boolean"


def test_fluent_frame_preserves_nullable_integer_property():
    graph = kglite.KnowledgeGraph()
    graph.cypher("CREATE (:N{id:1,v:$big}),(:N{id:2})", params={"big": 9007199254740993})
    frame = graph.select("N").to_df().sort_values("id")
    assert int(frame.v.iloc[0]) == 9007199254740993
    assert pd.isna(frame.v.iloc[1])


def test_empty_and_duplicate_result_columns_are_unchanged():
    graph = kglite.KnowledgeGraph()
    assert list(graph.cypher("UNWIND [] AS x RETURN x").to_df().columns) == ["x"]
    with pytest.raises(kglite.CypherSyntaxError, match="Multiple result columns"):
        graph.cypher("RETURN 1 AS x, 2 AS x").to_df()


@pytest.mark.parametrize("row_count", [0, 2])
def test_plain_empty_map_table_preserves_row_count(row_count):
    graph = kglite.KnowledgeGraph()
    rows = [{} for _ in range(row_count)]
    graph.cypher("CREATE (:N {id:1, items:$rows})", params={"rows": rows})
    assert_value_equal(graph.cypher("MATCH (n:N) RETURN n.items AS items").column("items"), [rows])
    actual = graph.get_table_property("N", 1, "items")
    assert actual.shape == (row_count, 0)
    assert list(actual.index) == list(range(row_count))


def test_table_setter_keeps_zero_column_refusal():
    graph = kglite.KnowledgeGraph()
    graph.cypher("CREATE (:N {id:1})")
    with pytest.raises(ValueError, match="DataFrame has no columns"):
        graph.set_table_property("N", 1, "items", pd.DataFrame(index=range(2)))
    assert graph.cypher("MATCH (n:N) RETURN n.items AS items").column("items") == [None]
