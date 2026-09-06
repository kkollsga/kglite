"""The comparison gates must reject the representation losses they guard."""

import datetime

import pytest

from tests.test_cypher_clean_room_contract import test_independent_cypher_behavior as run_contract
from tests.test_cypher_differential import ORDERED_CASES, _normalize
from tests.value_assertions import FloatTolerance, assert_rows_equal, assert_value_equal, canonical_value


@pytest.mark.parametrize(
    "actual,expected",
    [
        ("1", 1),
        (True, 1),
        (1.0, 1),
        (1, True),
        ("None", None),
        ("[1, None]", [1, None]),
        ([2, 1], [1, 2]),
        ({"x": True}, {"x": 1}),
        (datetime.datetime(2025, 1, 2), datetime.datetime(2025, 1, 2, microsecond=123456)),
    ],
)
def test_typed_oracle_rejects_substitution(actual, expected):
    with pytest.raises(AssertionError):
        assert_value_equal(actual, expected)
    assert _normalize([{"v": actual}]) != _normalize([{"v": expected}])


@pytest.mark.parametrize("actual,expected", [(True, 1), (1.0, 1), ("None", None), ("[1, None]", [1, None])])
def test_absolute_runner_rejects_substitution(monkeypatch, tmp_path, actual, expected):
    class Result:
        def to_list(self):
            return [{"v": actual}]

    class Graph:
        def cypher(self, *args, **kwargs):
            return Result()

    monkeypatch.setattr("kglite.KnowledgeGraph", Graph)
    case = {"query": "RETURN 1 AS v", "expected": [{"v": expected}], "requirement": "typed value"}
    with pytest.raises(AssertionError):
        run_contract(case, tmp_path)


def test_bag_retains_duplicates_and_nested_order():
    assert_rows_equal([{"v": 2}, {"v": 1}], [{"v": 1}, {"v": 2}], order="bag")
    for actual, expected in [([{"v": 1}], [{"v": 1}, {"v": 1}]), ([{"v": [2, 1]}], [{"v": [1, 2]}])]:
        with pytest.raises(AssertionError):
            assert_rows_equal(actual, expected, order="bag")
    with pytest.raises(AssertionError):
        assert_rows_equal([{"v": 2}, {"v": 1}], [{"v": 1}, {"v": 2}], order="ordered")
    assert canonical_value({"a": 1, "b": 2}) == canonical_value({"b": 2, "a": 1})


def test_named_tolerance_does_not_erase_types():
    tolerance = FloatTolerance("one rounded measurement", absolute=0.01)
    assert_value_equal({"v": [1.001]}, {"v": [1.0]}, tolerance=tolerance)
    with pytest.raises(AssertionError):
        assert_value_equal(1, 1.0, tolerance=tolerance)
    with pytest.raises(AssertionError):
        assert_value_equal(1.001, 1.0)
    with pytest.raises(ValueError):
        FloatTolerance("", absolute=0.01)
    with pytest.raises(TypeError):
        canonical_value(object())


def test_ordered_differential_cases_exist():
    from tests.test_cypher_differential import DIFFERENTIAL_QUERIES

    assert ORDERED_CASES <= {case[0] for case in DIFFERENTIAL_QUERIES}


def test_clean_room_admits_only_the_exact_shared_assertion_module():
    from scripts.check_cypher_clean_room import ALLOWED_ASSERTION_IMPORTS, ALLOWED_RUNNER_IMPORTS, unexpected_imports

    assert unexpected_imports("from tests.value_assertions import assert_rows_equal", ALLOWED_RUNNER_IMPORTS) == set()
    assert unexpected_imports("from tests.external_cases import CASES", ALLOWED_RUNNER_IMPORTS) == {
        "tests.external_cases"
    }
    assert unexpected_imports("import kglite", ALLOWED_ASSERTION_IMPORTS) == {"kglite"}
