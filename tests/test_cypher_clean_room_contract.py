"""Execute KGLite's independently authored Cypher behavioral contract."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import kglite

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = json.loads((ROOT / "tests" / "cypher_contract" / "cases.json").read_text())


@pytest.mark.parametrize("case", MANIFEST["cases"], ids=lambda case: case["id"])
def test_independent_cypher_behavior(case):
    graph = kglite.KnowledgeGraph()
    for setup_query in case.get("setup", []):
        graph.cypher(setup_query).to_list()

    actual = graph.cypher(case["query"], params=case.get("params")).to_list()
    assert actual == case["expected"], case["requirement"]


def test_clean_room_artifact_guard():
    from scripts.check_cypher_clean_room import validate

    assert validate() == []
