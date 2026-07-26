"""Regenerate the Phase 10 golden snapshots.

Rebuild the deterministic golden graph on memory mode, run every seed
query from ``tests/golden/queries.py``, and write the serialised
outputs under ``tests/golden/snapshots/``.

Usage:
    python tests/golden/regenerate.py              # overwrite snapshots
    python tests/golden/regenerate.py --check      # exit 1 on drift

Only invoke when semantics change intentionally. The committed
snapshots are the pinned contract; CI compares against them via
``tests/test_golden.py``.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any

# Make "golden" package importable when this file runs as a script.
_TESTS_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

from golden.build_golden_graph import build_golden_graph  # noqa: E402
from golden.queries import CYPHER_QUERIES, FIND_QUERIES  # noqa: E402

from kglite import KnowledgeGraph  # noqa: E402

SNAPSHOTS_DIR = pathlib.Path(__file__).resolve().parent / "snapshots"


def _normalise_row(row: dict[str, Any]) -> dict[str, Any]:
    """Canonicalise rows for deterministic byte-diffs.

    - dicts sorted by key
    - floats quantised to 6 decimals (round-trip through repr())
    - datetime values stringified
    """
    out: dict[str, Any] = {}
    for k in sorted(row):
        v = row[k]
        if isinstance(v, float):
            out[k] = round(v, 6)
        elif hasattr(v, "isoformat"):
            out[k] = v.isoformat()
        else:
            out[k] = v
    return out


def _sort_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        (_normalise_row(r) for r in rows),
        key=lambda r: json.dumps(r, sort_keys=True, default=str),
    )


def _cypher_snapshot(kg: KnowledgeGraph, query: str) -> list[dict[str, Any]]:
    rows = [dict(r) for r in kg.cypher(query)]
    return _sort_rows(rows)


def _find_snapshot(kg: KnowledgeGraph, name: str, node_type: str | None) -> list[dict[str, Any]]:
    if node_type is None:
        rows = list(kg.find(name))
    else:
        rows = list(kg.find(name, node_type))
    return _sort_rows([dict(r) if isinstance(r, dict) else r for r in rows])


def _schema_snapshot(kg: KnowledgeGraph) -> dict[str, Any]:
    raw = kg.schema()
    # Normalise dict order for deterministic JSON.
    return json.loads(json.dumps(raw, default=str, sort_keys=True))


def generate_all() -> dict[pathlib.Path, str]:
    """Return {path: serialised_content} for every snapshot file."""
    kg = KnowledgeGraph()
    build_golden_graph(kg)

    outputs: dict[pathlib.Path, str] = {}

    outputs[SNAPSHOTS_DIR / "schema.json"] = json.dumps(_schema_snapshot(kg), indent=2, sort_keys=True) + "\n"

    for slug, cypher in CYPHER_QUERIES:
        outputs[SNAPSHOTS_DIR / f"cypher_{slug}.json"] = (
            json.dumps(
                {"query": cypher, "rows": _cypher_snapshot(kg, cypher)},
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )

    for slug, name, node_type in FIND_QUERIES:
        outputs[SNAPSHOTS_DIR / f"find_{slug}.json"] = (
            json.dumps(
                {
                    "name": name,
                    "node_type": node_type,
                    "rows": _find_snapshot(kg, name, node_type),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )

    return outputs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero if any snapshot would change (do not write).",
    )
    args = parser.parse_args()

    SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)

    outputs = generate_all()
    drift: list[pathlib.Path] = []

    for path, content in outputs.items():
        existing = path.read_text(encoding="utf-8") if path.exists() else None
        if existing == content:
            continue
        if args.check:
            drift.append(path)
        else:
            path.write_text(content, encoding="utf-8")
            print(f"wrote {path.relative_to(_TESTS_DIR.parent)}")

    if args.check and drift:
        print("Drift detected in:", file=sys.stderr)
        for p in drift:
            print(f"  {p.relative_to(_TESTS_DIR.parent)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
