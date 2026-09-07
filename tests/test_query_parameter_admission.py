"""Query parameters refuse values that the engine cannot represent exactly."""

from decimal import Decimal
from fractions import Fraction
import math

import numpy as np
import pandas as pd
import pytest

import kglite


class FloatLike:
    def __float__(self):
        return 1.25


class IndexLike:
    def __index__(self):
        return 7


@pytest.fixture(params=["graph", "session", "frozen", "transaction"])
def query_surface(request):
    graph = kglite.KnowledgeGraph()
    transaction = None
    if request.param == "graph":
        surface = graph
    elif request.param == "session":
        surface = graph.session()
    elif request.param == "frozen":
        surface = graph.freeze()
    else:
        transaction = graph.begin_read()
        surface = transaction
    try:
        yield surface
    finally:
        if transaction is not None:
            transaction.rollback()


def roundtrip(surface, value):
    return surface.cypher("RETURN $value AS value", params={"value": value}).to_list()[0]["value"]


@pytest.mark.parametrize("value", [2**63, -(2**63) - 1])
def test_query_parameter_refuses_unrepresentable_python_integer(query_surface, value):
    with pytest.raises(OverflowError, match="value"):
        roundtrip(query_surface, value)


def test_query_parameter_refusal_names_nested_integer_path(query_surface):
    with pytest.raises(OverflowError, match=r"value.*\[0\].*items.*\[1\]"):
        roundtrip(query_surface, [{"items": [0, 2**63]}])


def test_query_parameter_refuses_unsupported_python_object(query_surface):
    with pytest.raises(TypeError, match=r"value.*object"):
        roundtrip(query_surface, object())


@pytest.mark.parametrize(
    "value",
    [Decimal("1.0000000000000000001"), Fraction(1, 3), FloatLike(), IndexLike()],
)
def test_query_parameter_refuses_implicit_numeric_coercion(query_surface, value):
    with pytest.raises(TypeError, match=rf"value.*{type(value).__name__}"):
        roundtrip(query_surface, value)


def test_query_parameter_refusal_names_nested_object_path(query_surface):
    with pytest.raises(TypeError, match=r"value.*\[0\].*items.*\[1\].*object"):
        roundtrip(query_surface, [{"items": [0, object()]}])


def test_refused_write_parameter_does_not_mutate_graph():
    graph = kglite.KnowledgeGraph()
    with pytest.raises(TypeError, match=r"value.*object"):
        graph.cypher("CREATE(:N {id: 1, value: $value})", params={"value": object()})
    assert graph.cypher("MATCH(n:N) RETURN count(*) AS n").to_list() == [{"n": 0}]


def test_query_parameter_accepts_exact_integer_boundaries_and_nested_values(query_surface):
    value = {
        "bounds": [-(2**63), 2**63 - 1],
        "ordinary": [None, True, 1.5, "Oslo", {"n": 7}],
    }
    assert roundtrip(query_surface, value) == value


def test_query_parameter_accepts_numpy_integer_boundaries(query_surface):
    value = [np.int64(-(2**63)), np.int64(2**63 - 1), np.uint64(2**63 - 1)]
    assert roundtrip(query_surface, value) == [-(2**63), 2**63 - 1, 2**63 - 1]


def test_query_parameter_accepts_representable_numpy_float_scalars(query_surface):
    value = [np.float16(1.5), np.float32(2.5), np.float64(3.5)]
    assert roundtrip(query_surface, value) == [1.5, 2.5, 3.5]


def test_query_parameter_refuses_unrepresentable_numpy_unsigned_integer(query_surface):
    with pytest.raises(OverflowError, match=r"value.*\[0\]"):
        roundtrip(query_surface, [np.uint64(2**63)])


def test_query_parameter_keeps_ndarray_and_nonfinite_float_support(query_surface):
    value = [np.array([[1, 2], [3, 4]]), math.nan, math.inf, -math.inf]
    actual = roundtrip(query_surface, value)
    assert actual[0] == [[1, 2], [3, 4]]
    assert math.isnan(actual[1])
    assert actual[2:] == [math.inf, -math.inf]


def test_declared_ingestion_keeps_tolerant_unsupported_value_policy():
    graph = kglite.KnowledgeGraph()
    with pytest.warns(UserWarning, match="stored as text"):
        graph.add_nodes(pd.DataFrame({"id": [1], "value": [object()]}), "N", "id")
    value = graph.cypher("MATCH(n:N) RETURN n.value AS value").to_list()[0]["value"]
    assert isinstance(value, str)
    assert value.startswith("<object object at 0x")


def test_a_timezone_aware_datetime_parameter_binds_as_naive_utc():
    """The declared conversion, pinned. KGLite's temporal values are zoneless,
    so a bound aware datetime normalises to UTC and loses its zone — the same
    rule the Cypher `datetime()` constructor already applies to an
    offset-bearing literal. This is Python's answer *only*: the Bolt server
    refuses the same parameter (`Neo.ClientError.Request.Invalid`) rather than
    converting it, because a PackStream zoned type is one a driver expects to
    round-trip. Both behaviours are correct for their surface, and the
    divergence is declared in `docs/python/value-projection.md` and
    `docs/operators/bolt-server.md` rather than unified."""
    import datetime

    aware = datetime.datetime(2024, 3, 9, 14, 30, 5, 123456, tzinfo=datetime.timezone(datetime.timedelta(hours=2)))
    graph = kglite.KnowledgeGraph()
    bound = graph.cypher("RETURN $v AS v", params={"v": aware}).to_list()[0]["v"]
    assert bound == datetime.datetime(2024, 3, 9, 12, 30, 5, 123456)
    assert bound.tzinfo is None, "a bound temporal must come back zoneless"

    # A naive datetime keeps its wall-clock value — the documented way to
    # preserve one.
    naive = datetime.datetime(2024, 3, 9, 14, 30, 5, 123456)
    assert graph.cypher("RETURN $v AS v", params={"v": naive}).to_list()[0]["v"] == naive
