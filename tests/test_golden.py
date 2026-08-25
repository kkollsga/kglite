"""Golden-fixture regression tests.

Rebuilds the Phase 10 golden graph on every storage mode, runs each
seed query, and asserts byte-identical output against the committed
snapshots under ``tests/golden/snapshots/``.

The BM25 and hybrid rankings are a second fixture over the 12-document text
corpus, asserted **in row order** — a ranking *is* the order. They run on the two
storage modes that can hold a text index; ``disk`` refuses to build one, and
that refusal is asserted here rather than skipped over.

Intentional output changes: run ``python tests/golden/regenerate.py``
and commit the refreshed snapshots alongside the feature change.
"""

from __future__ import annotations

import pathlib

import pytest

from kglite import KnowledgeGraph
from tests.golden.build_golden_graph import build_golden_graph
from tests.golden.build_text_corpus import build_text_corpus
from tests.golden.queries import BM25_QUERIES, CYPHER_QUERIES, FIND_QUERIES, HYBRID_QUERIES
from tests.golden.regenerate import (
    _cypher_snapshot,
    _find_snapshot,
    _ranked_snapshot,
    _schema_snapshot,
)

SNAPSHOTS_DIR = pathlib.Path(__file__).resolve().parent / "golden" / "snapshots"
STORAGE_MODES = ("memory", "mapped", "disk")
# `disk` is absent by design: build_text_index refuses there, and
# `test_bm25_refused_on_disk` pins the refusal.
TEXT_INDEX_MODES = ("memory", "mapped")


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


@pytest.mark.parametrize("mode", TEXT_INDEX_MODES)
@pytest.mark.parametrize("slug,cypher", BM25_QUERIES, ids=[slug for slug, _ in BM25_QUERIES])
def test_bm25_snapshot(mode, slug, cypher, tmp_path):
    kg = build_text_corpus(_new_kg(mode, tmp_path))
    import json

    got = (
        json.dumps(
            {"query": cypher, "rows": _ranked_snapshot(kg, cypher)},
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    assert got == _load_snapshot(f"bm25_{slug}.json"), (
        f"bm25_{slug}.json drift on mode={mode}. Run `python tests/golden/regenerate.py` to refresh if intentional."
    )


@pytest.mark.parametrize("mode", TEXT_INDEX_MODES)
@pytest.mark.parametrize("slug,cypher", HYBRID_QUERIES, ids=[slug for slug, _ in HYBRID_QUERIES])
def test_hybrid_snapshot(mode, slug, cypher, tmp_path):
    """The two lanes fused: same corpus, same row-order rule, and the same
    ranking on both storage modes that can hold a text index."""
    kg = build_text_corpus(_new_kg(mode, tmp_path))
    import json

    got = (
        json.dumps(
            {"query": cypher, "rows": _ranked_snapshot(kg, cypher)},
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    assert got == _load_snapshot(f"hybrid_{slug}.json"), (
        f"hybrid_{slug}.json drift on mode={mode}. Run `python tests/golden/regenerate.py` to refresh if intentional."
    )


def test_bm25_refused_on_disk(tmp_path):
    """The corpus builds; only the index refuses, and it names the way out."""
    kg = _new_kg("disk", tmp_path)
    with pytest.raises(ValueError, match="disk-backed graph"):
        build_text_corpus(kg)
