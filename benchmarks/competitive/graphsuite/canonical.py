"""Canonicalisation of group results for cross-backend parity checking.

Each group method returns an *actual result* (a set of ids, an aggregate
dict, a vector of path lengths, …) rather than a bare count. This module
turns any such result into:

- a stable **digest** (short hash) so two backends' results can be
  compared for exact equality regardless of ordering / float jitter, and
- a short **display** value (a count / scalar) for the timing table.

Floats are rounded so that e.g. two engines' `avg(age)` that differ only
in the last ULP still compare equal.
"""

from __future__ import annotations

import hashlib
from typing import Any

ROUND = 4


def _norm(x: Any) -> str:
    if x is None:
        return "∅"
    if isinstance(x, float):
        # whole floats collapse to int form so 4.0 == 4
        return str(int(x)) if x.is_integer() else f"{round(x, ROUND)}"
    if isinstance(x, bool):
        return str(int(x))
    if isinstance(x, (tuple, list)):
        return "(" + ",".join(_norm(e) for e in x) + ")"
    return str(x)


def serialise(value: Any) -> str:
    """Deterministic string form of a group result, independent of order."""
    if isinstance(value, (set, frozenset)):
        return "{" + ",".join(sorted(_norm(v) for v in value)) + "}"
    if isinstance(value, dict):
        return ";".join(f"{_norm(k)}={_norm(value[k])}" for k in sorted(value, key=_norm))
    if isinstance(value, (list, tuple)):
        # order-significant sequence (e.g. per-pair path lengths)
        return "[" + ",".join(_norm(v) for v in value) + "]"
    return _norm(value)


def digest(value: Any) -> str:
    return hashlib.sha1(serialise(value).encode()).hexdigest()[:12]


def display(value: Any) -> Any:
    """Short human value for the timing table (a count or the scalar)."""
    if isinstance(value, (set, frozenset, dict, list, tuple)):
        return len(value)
    return value
