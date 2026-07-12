"""Public dialect claims and extension naming stay executable and honest."""

from __future__ import annotations

import json
from pathlib import Path

import kglite

ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = ROOT / "tests" / "api-baselines" / "cypher-dialect.json"
DOC_PATH = ROOT / "CYPHER.md"
CLAIM_SURFACES = [
    DOC_PATH,
    ROOT / "docs" / "index.md",
    ROOT / "docs" / "operators" / "bolt-server.md",
    ROOT / "docs" / "operators" / "index.md",
    ROOT / "crates" / "kglite-bolt-server" / "README.md",
]


def _manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text())


def test_dialect_manifest_has_stable_unique_classifications():
    manifest = _manifest()
    allowed = set(manifest["status_definitions"])
    features = manifest["features"]
    extensions = manifest["extensions"]

    assert manifest["schema_version"] == 1
    assert "not a complete" in manifest["claim"]
    assert len({feature["id"] for feature in features}) == len(features)
    assert len({extension["id"] for extension in extensions}) == len(extensions)
    assert {feature["status"] for feature in features} <= allowed
    assert {"covered", "partial", "unsupported", "intentional_divergence"} <= {
        feature["status"] for feature in features
    }


def test_completed_alignment_work_is_not_described_as_future_work():
    manifest = _manifest()
    features = {feature["id"]: feature for feature in manifest["features"]}
    assert features["clause.where"]["status"] == "covered"
    assert features["path.values"]["status"] == "covered"
    rendered = json.dumps(manifest).lower()
    assert "being aligned" not in rendered
    assert "is tracked" not in rendered


def test_reference_links_the_machine_readable_contract_and_names_every_extension():
    document = DOC_PATH.read_text()
    manifest = _manifest()
    assert "tests/api-baselines/cypher-dialect.json" in document
    assert "not a complete openCypher or Neo4j implementation" in document
    for extension in manifest["extensions"]:
        assert extension["canonical"] in document, extension["id"]


def test_public_claim_surfaces_do_not_promise_complete_or_drop_in_compatibility():
    forbidden = ("Full Cypher", "plugs in unchanged", "absolute-correctness oracle")
    for path in CLAIM_SURFACES:
        document = path.read_text()
        assert not any(phrase in document for phrase in forbidden), path

    migration = (ROOT / "docs" / "python" / "migrations" / "neo4j-to-kglite.md").read_text()
    assert "`FOREACH (x IN list \\| ...)` | Supported" in migration


def test_namespaced_and_flat_extension_function_are_equivalent():
    graph = kglite.KnowledgeGraph()
    rows = graph.cypher(
        "RETURN text_normalize('  Hello, WORLD!  ') AS flat, KGLITE.TEXT_NORMALIZE('  Hello, WORLD!  ') AS namespaced"
    ).to_list()
    assert rows == [{"flat": "hello world", "namespaced": "hello world"}]


def test_namespaced_and_flat_extension_procedure_are_equivalent():
    graph = kglite.KnowledgeGraph()
    flat = graph.cypher(
        "CALL refresh_stats() YIELD src_type, edge_type, tgt_type, count RETURN src_type, edge_type, tgt_type, count"
    ).to_list()
    namespaced = graph.cypher(
        "CALL KGLITE.REFRESH_STATS() YIELD src_type, edge_type, tgt_type, count "
        "RETURN src_type, edge_type, tgt_type, count"
    ).to_list()
    assert namespaced == flat == []
