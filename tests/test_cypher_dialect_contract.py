"""Public dialect claims and extension naming stay executable and honest."""

from __future__ import annotations

import json
from pathlib import Path
import re
import subprocess
import sys

import pytest

import kglite

ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = ROOT / "tests" / "api-baselines" / "cypher-dialect.json"
CASES_PATH = ROOT / "tests" / "cypher_contract" / "cases.json"
RUNNER_PATH = ROOT / "tests" / "test_cypher_clean_room_contract.py"
DOC_PATH = ROOT / "CYPHER.md"
CLAIM_SURFACES = [
    DOC_PATH,
    ROOT / "docs" / "index.md",
    ROOT / "docs" / "concepts" / "cypher-conformance.md",
    ROOT / "docs" / "operators" / "bolt-server.md",
    ROOT / "docs" / "operators" / "index.md",
    ROOT / "crates" / "kglite-bolt-server" / "README.md",
]

# Statuses that claim executable behavior and must therefore cite behavioral
# cases from tests/cypher_contract/cases.json.
CASE_BACKED_STATUSES = {"covered", "partial"}

# Over-claiming phrases banned from public claim surfaces, matched
# case-insensitively and across line breaks. A match is tolerated only in an
# explicit negation ("not ..." shortly before it), which is how the sanctioned
# disclaimers are worded (e.g. "not a complete openCypher implementation").
FORBIDDEN_CLAIM_PHRASES = (
    "full cypher",
    "plugs in unchanged",
    "absolute-correctness oracle",
    "fully compatible",
    "drop-in replacement",
    "drop-in compatible",
    "complete opencypher",
    "full neo4j compatibility",
    "100% compatible",
)
NEGATION_WINDOW = 60


def _manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text())


def _corpus_case_ids() -> set[str]:
    return {case["id"] for case in json.loads(CASES_PATH.read_text())["cases"]}


def _referenced_case_ids(manifest: dict) -> set[str]:
    return {case_id for feature in manifest["features"] for case_id in feature.get("case_ids", [])}


def test_dialect_manifest_has_stable_unique_classifications():
    manifest = _manifest()
    allowed = set(manifest["status_definitions"])
    features = manifest["features"]
    extensions = manifest["extensions"]

    assert manifest["schema_version"] == 2
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


def test_every_case_backed_feature_cites_at_least_one_behavioral_case():
    manifest = _manifest()
    gaps = set(manifest.get("coverage_gaps", []))
    for feature in manifest["features"]:
        if feature["status"] not in CASE_BACKED_STATUSES or feature["id"] in gaps:
            continue
        case_ids = feature.get("case_ids", [])
        assert case_ids, (
            f"{feature['id']} claims {feature['status']!r} but cites no behavioral case; "
            "add case_ids or declare it in coverage_gaps"
        )
        assert len(set(case_ids)) == len(case_ids), f"{feature['id']} repeats a case_id"


def test_every_referenced_case_id_exists_in_the_contract_corpus():
    manifest = _manifest()
    missing = _referenced_case_ids(manifest) - _corpus_case_ids()
    assert not missing, f"case_ids not present in tests/cypher_contract/cases.json: {sorted(missing)}"


def test_declared_coverage_gaps_stay_visible_and_real():
    manifest = _manifest()
    gaps = manifest.get("coverage_gaps", [])
    case_backed = {feature["id"] for feature in manifest["features"] if feature["status"] in CASE_BACKED_STATUSES}
    unknown = set(gaps) - case_backed
    assert not unknown, f"coverage_gaps names non-case-backed features: {sorted(unknown)}"
    if gaps:
        pytest.xfail(f"declared behavioral-coverage gaps awaiting cases: {sorted(gaps)}")


def test_every_referenced_case_is_collected_by_the_contract_runner():
    manifest = _manifest()
    referenced = _referenced_case_ids(manifest)
    collection = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "-p",
            "no:cacheprovider",
            str(RUNNER_PATH),
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert collection.returncode == 0, collection.stdout + collection.stderr
    collected = set(re.findall(r"test_independent_cypher_behavior\[(.+?)\]", collection.stdout))
    missing = referenced - collected
    assert not missing, f"case_ids the contract runner never executes: {sorted(missing)}"


def test_provenance_guard_boundary_matches_reality():
    from scripts.check_cypher_clean_room import (
        PROVENANCE_GUARDED_PATHS,
        REVIEWED_SEMANTIC_SUITES,
    )

    for rel in PROVENANCE_GUARDED_PATHS + REVIEWED_SEMANTIC_SUITES:
        assert (ROOT / rel).exists(), f"conformance-surface manifest is stale: {rel}"
    assert set(PROVENANCE_GUARDED_PATHS).isdisjoint(REVIEWED_SEMANTIC_SUITES)
    assert "tests/cypher_contract" in PROVENANCE_GUARDED_PATHS
    assert "tests/test_cypher_clean_room_contract.py" in PROVENANCE_GUARDED_PATHS


def test_reference_links_the_machine_readable_contract_and_names_every_extension():
    document = DOC_PATH.read_text()
    manifest = _manifest()
    assert "tests/api-baselines/cypher-dialect.json" in document
    assert "not a complete openCypher or Neo4j implementation" in document
    for extension in manifest["extensions"]:
        assert extension["canonical"] in document, extension["id"]


def _overclaims(text: str) -> list[str]:
    lowered = text.lower()
    found = []
    for phrase in FORBIDDEN_CLAIM_PHRASES:
        pattern = re.escape(phrase).replace(r"\ ", r"\s+")
        for match in re.finditer(pattern, lowered):
            window = lowered[max(0, match.start() - NEGATION_WINDOW) : match.start()]
            if re.search(r"\bnot\b", window):
                continue
            found.append(phrase)
    return found


def test_public_claim_surfaces_do_not_promise_complete_or_drop_in_compatibility():
    for path in CLAIM_SURFACES:
        overclaims = _overclaims(path.read_text())
        assert not overclaims, f"{path}: over-claiming phrases {overclaims}"

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
