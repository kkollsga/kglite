"""Lossless test oracles; transport adapters must preserve the raw value first."""

from __future__ import annotations

from dataclasses import dataclass
import datetime
import math
from typing import Literal


def canonical_value(value):
    """Tag primitive types; only map-key order is immaterial inside a value."""
    kind = type(value)
    if value is None:
        return ("null",)
    if kind in (bool, int, str):
        return (kind.__name__, value)
    if kind is float:
        # hex preserves finite bits, signed zero and infinity; NaN remains a
        # float token, never equivalent to NULL or the string "nan".
        return ("float", value.hex())
    if kind in (datetime.datetime, datetime.date):
        return (kind.__name__, value.isoformat())
    if kind in (list, tuple):
        return (kind.__name__, tuple(canonical_value(v) for v in value))
    if kind is dict:
        pairs = [(canonical_value(k), canonical_value(v)) for k, v in value.items()]
        return ("dict", tuple(sorted(pairs, key=lambda pair: repr(pair[0]))))
    raise TypeError(f"No lossless test adapter for {kind.__module__}.{kind.__qualname__}")


def canonical_rows(rows, *, order: Literal["ordered", "bag"]):
    """Bag comparison reorders outer rows only and retains every duplicate."""
    values = [canonical_value(row) for row in rows]
    if order == "ordered":
        return values
    if order == "bag":
        return sorted(values, key=repr)
    raise ValueError(f"Unknown row comparison mode: {order}")


@dataclass(frozen=True)
class FloatTolerance:
    """An explicitly named approximation for a particular ordered assertion."""

    reason: str
    relative: float = 0.0
    absolute: float = 0.0

    def __post_init__(self):
        if not self.reason.strip() or any(not math.isfinite(v) or v < 0 for v in (self.relative, self.absolute)):
            raise ValueError("Float tolerance needs a reason and finite nonnegative bounds")


def assert_value_equal(actual, expected, *, tolerance: FloatTolerance | None = None):
    if tolerance is None:
        assert canonical_value(actual) == canonical_value(expected), f"typed values differ: {actual!r} != {expected!r}"
        return
    assert type(actual) is type(expected), f"primitive types differ: {type(actual)} != {type(expected)}"
    if type(expected) is float:
        assert canonical_value(actual) == canonical_value(expected) or math.isclose(
            actual, expected, rel_tol=tolerance.relative, abs_tol=tolerance.absolute
        ), f"floats differ ({tolerance.reason}): {actual!r} != {expected!r}"
    elif type(expected) is dict:
        assert canonical_value(list(sorted(actual))) == canonical_value(list(sorted(expected)))
        for key in expected:
            assert_value_equal(actual[key], expected[key], tolerance=tolerance)
    elif type(expected) in (list, tuple):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            assert_value_equal(left, right, tolerance=tolerance)
    else:
        assert_value_equal(actual, expected)


def assert_rows_equal(actual, expected, *, order: Literal["ordered", "bag"]):
    assert canonical_rows(actual, order=order) == canonical_rows(expected, order=order), (
        f"typed {order} rows differ:\nactual: {actual!r}\nexpected: {expected!r}"
    )
