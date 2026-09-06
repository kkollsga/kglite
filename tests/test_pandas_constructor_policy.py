"""Binding-only dtype policy: exact inputs, optional import, and no lossy pre-inference."""

import subprocess
import sys

import pandas as pd
import pytest


@pytest.mark.parametrize(
    "values,dtype",
    [
        ([1, 2], "int64"),
        ([1.5, 2.5], "float64"),
        ([9007199254740993, None], "Int64"),
        ([9007199254740993, 1.5], "object"),
        ([1, True], "object"),
        ([1, "text"], "object"),
        ([1, [2]], "object"),
        ([1, {"x": 2}], "object"),
        ([-(2**63), None, 2**63 - 1], "Int64"),
        ([], "object"),
    ],
)
def test_constructor_selects_dtype_before_values_can_round(values, dtype):
    from kglite._pandas import dataframe

    actual = dataframe({"v": values})
    assert str(actual.v.dtype) == dtype
    for before, after in zip(values, actual.v.tolist(), strict=True):
        if before is None:
            assert pd.isna(after)
        elif type(before) is int:
            assert not isinstance(after, (bool, float))
            assert int(after) == before
        else:
            assert type(after) is type(before)
            assert after == before


def test_records_and_explicit_metadata_are_applied_from_exact_cells():
    from kglite._pandas import dataframe

    result = dataframe([{"v": 2**63 - 1, "b": True}, {}], columns=["b", "v"], dtypes={"v": "Int64", "b": "boolean"})
    assert list(result.columns) == ["b", "v"]
    assert int(result.v.iloc[0]) == 2**63 - 1
    assert pd.isna(result.v.iloc[1])


def test_import_does_not_require_or_import_pandas():
    command = "import sys; import kglite._pandas; assert 'pandas' not in sys.modules"
    result = subprocess.run([sys.executable, "-c", command], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "records,columns,rows",
    [([{}, {}], None, 2), ([{}, {}], [], 2), ([{"discard": 1}, {}], [], 2), ([], [], 0), ({}, [], 0)],
)
def test_zero_column_records_preserve_row_count(records, columns, rows):
    from kglite._pandas import dataframe

    result = dataframe(records, columns=columns)
    assert result.shape == (rows, 0)
    assert list(result.index) == list(range(rows))


@pytest.mark.parametrize("values", [[1.5, 2.5], [1, 2], [2**53 + 1, None], [1, True], [None, None], []])
def test_native_result_hints_avoid_reclassifying_exact_cells(monkeypatch, values):
    import kglite
    import kglite._pandas as helper

    def unexpected_scan(_values):
        pytest.fail("native result column was scanned again in Python")

    monkeypatch.setattr(helper, "_integer_dtype", unexpected_scan)
    frame = kglite.KnowledgeGraph().cypher("UNWIND $values AS v RETURN v", params={"values": values}).to_df()
    assert len(frame) == len(values)
    for expected, actual in zip(values, frame.v.tolist(), strict=True):
        if expected is None:
            assert pd.isna(actual)
        else:
            assert type(expected) is type(actual)
            assert expected == actual


def test_unhinted_columns_keep_exact_policy(monkeypatch):
    import kglite._pandas as helper

    original = helper._integer_dtype
    scanned = []

    def scan(values):
        scanned.append(values)
        return original(values)

    monkeypatch.setattr(helper, "_integer_dtype", scan)
    raw = {"a": [1.5, 2.5], "b": [2**53 + 1, None]}
    frame = helper.dataframe(raw, native_dtypes={"a": None})
    assert scanned == [raw["b"]]
    assert frame.a.tolist() == raw["a"]
    assert str(frame.b.dtype) == "Int64"
    assert int(frame.b.iloc[0]) == 2**53 + 1
    assert frame.b.iloc[1] is pd.NA
