"""OKF ingestion — Tier 2: real-corpus integration.

Loads the three reference Open Knowledge Format bundles vendored from
GoogleCloudPlatform/knowledge-catalog (markdown-only, Apache-2.0) under
``tests/fixtures/okf/google/`` and asserts the loader ingests real-world OKF
correctly: expected graph shape, that every link resolves (no accidental
dangling), that section-inference yields typed edges on real data, that
``okf.source`` reads bodies on demand, and that the downstream analytics
(``CALL leiden`` / orphan detection) run cleanly over the result.

The bundles are frozen committed copies, so the captured counts are stable.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from kglite import okf

CORPUS = Path(__file__).parent / "fixtures" / "okf" / "google"

# Captured shape of the vendored bundles, with the default-on enrichment
# (concepts + Tag + Source nodes; link / TAGGED / CITES edges).
EXPECTED = {
    "stackoverflow": {"nodes": 133, "edges": 377},
    "ga4": {"nodes": 28, "edges": 47},
    "crypto_bitcoin": {"nodes": 26, "edges": 54},
}


def _build(name: str):
    return okf.build(str(CORPUS / name))


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_bundle_ingests_to_expected_shape(name):
    g = _build(name)
    n = g.cypher("MATCH (n) RETURN count(n) AS c").to_list()[0]["c"]
    e = g.cypher("MATCH ()-[r]->() RETURN count(r) AS c").to_list()[0]["c"]
    assert n == EXPECTED[name]["nodes"]
    assert e == EXPECTED[name]["edges"]


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_all_links_resolve_in_bundle(name):
    # The Google bundles are internally consistent — no link should dangle.
    g = _build(name)
    prov = g.cypher("MATCH (n {_provisional:true}) RETURN count(n) AS c").to_list()[0]["c"]
    assert prov == 0


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_leiden_runs_on_bundle(name):
    # Community detection (the OKF→GraphRAG indexing story) runs over real data.
    g = _build(name)
    c = g.cypher("CALL leiden() YIELD community RETURN count(DISTINCT community) AS c").to_list()[0]["c"]
    assert c >= 1


def test_stackoverflow_labels_and_dataset():
    g = _build("stackoverflow")
    labels = Counter(r["l"] for r in g.cypher("MATCH (n) RETURN labels(n)[0] AS l").to_list())
    # The public Stack Overflow catalog: one dataset, tables, and references.
    assert labels["BigQuery Dataset"] == 1
    assert labels["BigQuery Table"] >= 10
    assert labels["Reference"] >= 1


def test_section_inference_fires_on_real_data():
    # ga4 has a concept with a "# Joins" section linking another concept — proves
    # the edge-type ladder works beyond the synthetic golden fixtures.
    g = _build("ga4")
    joins = g.cypher("MATCH ()-[r:JOINS_WITH]->() RETURN count(r) AS c").to_list()[0]["c"]
    assert joins >= 1


def test_okf_source_reads_body_on_demand():
    # The on-demand body reader strips frontmatter.
    path = CORPUS / "stackoverflow" / "tables" / "posts_questions.md"
    body = okf.source(str(path))
    assert "# Schema" in body
    assert not body.lstrip().startswith("---"), "frontmatter should be stripped"
    # the frontmatter `resource:` key must not leak into the body
    assert "resource:" not in body.splitlines()[0]


def test_concepts_carry_file_path_pointer():
    # Partial ingestion: every concept keeps a file_path back to its source.
    # (Tag/Source nodes have no concept_id and are excluded.)
    g = _build("crypto_bitcoin")
    rows = g.cypher(
        "MATCH (n) WHERE n.concept_id IS NOT NULL AND n.file_path IS NULL "
        "AND coalesce(n._provisional, false) = false RETURN count(n) AS c"
    ).to_list()
    assert rows[0]["c"] == 0


def test_enrichment_densifies_and_improves_clustering():
    # Tag + Source nodes turn the sparse author-link graph into a dense,
    # well-clustering one. On stackoverflow this collapsed leiden from 19
    # fragmented communities (12 singletons) to a handful of real ones.
    g = _build("stackoverflow")
    from collections import Counter

    tagged = g.cypher("MATCH ()-[r:TAGGED]->() RETURN count(r) AS c").to_list()[0]["c"]
    cites = g.cypher("MATCH ()-[r:CITES]->() RETURN count(r) AS c").to_list()[0]["c"]
    assert tagged > 100 and cites > 10
    comm = Counter(r["c"] for r in g.cypher("CALL leiden() YIELD community RETURN community AS c").to_list())
    singletons = sum(1 for v in comm.values() if v == 1)
    assert len(comm) <= 10
    assert singletons <= 4
