"""Golden-fixture regression tests.

Rebuilds the Phase 10 golden graph on every storage mode, runs each
seed query, and asserts byte-identical output against the committed
snapshots under ``tests/golden/snapshots/``.

Intentional output changes: run ``python tests/golden/regenerate.py``
and commit the refreshed snapshots alongside the feature change.
"""

from __future__ import annotations

import pathlib

import pytest

from kglite import KnowledgeGraph
from tests.golden.build_golden_graph import build_golden_graph
from tests.golden.queries import CYPHER_QUERIES, FIND_QUERIES
from tests.golden.regenerate import (
    _cypher_snapshot,
    _find_snapshot,
    _schema_snapshot,
)

SNAPSHOTS_DIR = pathlib.Path(__file__).resolve().parent / "golden" / "snapshots"
STORAGE_MODES = ("memory", "mapped", "disk")


def _new_kg(mode: str, tmp_path) -> KnowledgeGraph:
    if mode == "memory":
        return KnowledgeGraph()
    if mode == "mapped":
        return KnowledgeGraph(storage="mapped")
    if mode == "disk":
        return KnowledgeGraph(storage="disk", path=str(tmp_path / "kg_disk"))
    raise ValueError(mode)


@pytest.fixture(scope="module")
def _memory_golden():
    """Build-once golden graph reused across memory-mode snapshot tests."""
    kg = KnowledgeGraph()
    build_golden_graph(kg)
    return kg


def _load_snapshot(name: str) -> str:
    return (SNAPSHOTS_DIR / name).read_text(encoding="utf-8")


@pytest.mark.parametrize("mode", STORAGE_MODES)
def test_schema_snapshot(mode, tmp_path):
    kg = _new_kg(mode, tmp_path)
    build_golden_graph(kg)
    import json

    got = json.dumps(_schema_snapshot(kg), indent=2, sort_keys=True) + "\n"
    assert got == _load_snapshot("schema.json"), (
        f"schema.json drift on mode={mode}. Run `python tests/golden/regenerate.py` to refresh if intentional."
    )


@pytest.mark.parametrize("mode", STORAGE_MODES)
@pytest.mark.parametrize("slug,cypher", CYPHER_QUERIES, ids=[slug for slug, _ in CYPHER_QUERIES])
def test_cypher_snapshot(mode, slug, cypher, tmp_path):
    kg = _new_kg(mode, tmp_path)
    build_golden_graph(kg)
    import json

    got = (
        json.dumps(
            {"query": cypher, "rows": _cypher_snapshot(kg, cypher)},
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    assert got == _load_snapshot(f"cypher_{slug}.json"), (
        f"cypher_{slug}.json drift on mode={mode}. Run `python tests/golden/regenerate.py` to refresh if intentional."
    )


@pytest.mark.parametrize("mode", STORAGE_MODES)
@pytest.mark.parametrize(
    "slug,name,node_type",
    FIND_QUERIES,
    ids=[slug for slug, _, _ in FIND_QUERIES],
)
def test_find_snapshot(mode, slug, name, node_type, tmp_path):
    kg = _new_kg(mode, tmp_path)
    build_golden_graph(kg)
    import json

    got = (
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
    assert got == _load_snapshot(f"find_{slug}.json"), (
        f"find_{slug}.json drift on mode={mode}. Run `python tests/golden/regenerate.py` to refresh if intentional."
    )
