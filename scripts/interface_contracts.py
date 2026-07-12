#!/usr/bin/env python3
"""Capture deterministic public-interface contracts for reviewable baselines."""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
PYTHON_BASELINE = ROOT / "tests" / "api-baselines" / "python-api.json"
PROTOCOL_MEMBERS = {"__enter__", "__exit__", "__iter__", "__len__", "__next__"}
NONCONSTRUCTIBLE = {"FrozenGraph", "ResultIter", "ResultView", "Session", "Transaction"}


def _signature(value: Any) -> str | None:
    try:
        return str(inspect.signature(value))
    except (TypeError, ValueError):
        return None


def _member_contract(cls: type, name: str, raw: Any) -> dict[str, Any]:
    resolved = getattr(cls, name)
    if isinstance(raw, property):
        return {"kind": "property"}
    signature = _signature(resolved)
    if signature is not None:
        return {"kind": "method", "signature": signature}
    return {"kind": "attribute", "type": type(resolved).__name__}


def capture_python_api() -> dict[str, Any]:
    """Return the runtime Python surface in a stable JSON-ready shape."""
    import kglite

    exports = list(kglite.__all__)
    objects: dict[str, Any] = {}
    for name in exports:
        value = getattr(kglite, name)
        if inspect.isclass(value):
            member_names = sorted(
                member for member in value.__dict__ if (not member.startswith("_") or member in PROTOCOL_MEMBERS)
            )
            objects[name] = {
                "kind": "class",
                "module": value.__module__,
                "bases": [base.__name__ for base in value.__bases__],
                "constructible": name not in NONCONSTRUCTIBLE,
                "constructor": _signature(value),
                "members": {member: _member_contract(value, member, value.__dict__[member]) for member in member_names},
            }
        elif callable(value):
            objects[name] = {"kind": "function", "signature": _signature(value)}
        else:
            objects[name] = {"kind": "constant", "type": type(value).__name__}

    non_exported_runtime = {
        name: type(value).__name__
        for name, value in sorted(vars(kglite).items())
        if not name.startswith("_") and name not in exports and not inspect.ismodule(value)
    }
    return {
        "schema_version": 1,
        "exports": exports,
        "non_exported_runtime": non_exported_runtime,
        "objects": objects,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help=f"write {PYTHON_BASELINE.relative_to(ROOT)}")
    args = parser.parse_args()
    rendered = json.dumps(capture_python_api(), indent=2, sort_keys=True) + "\n"
    if args.write:
        PYTHON_BASELINE.write_text(rendered)
        print(f"wrote {PYTHON_BASELINE.relative_to(ROOT)}")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
