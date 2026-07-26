"""Execute KGLite's independently authored Cypher behavioral contract."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import kglite

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = json.loads((ROOT / "tests" / "cypher_contract" / "cases.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("case", MANIFEST["cases"], ids=lambda case: case["id"])
def test_independent_cypher_behavior(case, tmp_path):
    graph = kglite.KnowledgeGraph()
    for setup_query in case.get("setup", []):
        graph.cypher(setup_query).to_list()

    # `LOAD CSV` cases declare their input inline as `files: {name: content}`.
    # Each file is written under `tmp_path` (never into the repo) and the query
    # refers to it as `{csv_dir}/<name>`, substituted here — the only way a
    # declarative case can name a real path.
    query = case["query"]
    for name, content in case.get("files", {}).items():
        (tmp_path / name).write_text(content, encoding="utf-8")
    if "files" in case:
        query = query.replace("{csv_dir}", str(tmp_path))

    actual = graph.cypher(query, params=case.get("params")).to_list()
    assert actual == case["expected"], case["requirement"]


def test_clean_room_artifact_guard():
    from scripts.check_cypher_clean_room import validate

    assert validate() == []
