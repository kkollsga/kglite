"""search_text / vector_search hit contract + `returning=` projection (Phase A).

Operator note (search hits + harvest N1): a hit should carry the full record so
no follow-up Cypher hydrate is needed — and a caller should be able to trim the
payload. This verifies:

- the DEFAULT hit carries id, title, type, score, AND every node property;
- `returning=[...]` trims a hit to id + score + the named fields;
- the contract holds across a save/load cycle (columnar storage) and to_df.
"""

import pandas as pd

import kglite


def _graph() -> kglite.KnowledgeGraph:
    g = kglite.KnowledgeGraph()
    g.add_nodes(
        pd.DataFrame(
            {
                "id": [1, 2, 3],
                "title": ["A", "B", "C"],
                "body": ["alpha", "beta", "gamma"],
                "year": [2001, 2002, 2003],
            }
        ),
        "Doc",
        "id",
        "title",
    )
    g.add_embeddings(
        "Doc",
        "summary",
        {1: [1.0, 0.0, 0.0, 0.0], 2: [0.0, 1.0, 0.0, 0.0], 3: [0.9, 0.1, 0.0, 0.0]},
    )
    return g


def test_default_hit_carries_all_fields():
    g = _graph()
    hits = g.select("Doc").vector_search("summary", [1.0, 0.0, 0.0, 0.0], top_k=3)
    top = hits[0]
    assert top["id"] == 1
    assert "score" in top
    # full record — no follow-up hydrate needed
    assert top["title"] == "A"
    assert top["type"] == "Doc"
    assert top["body"] == "alpha"
    assert top["year"] == 2001


def test_returning_trims_to_named_fields():
    g = _graph()
    hits = g.select("Doc").vector_search("summary", [1.0, 0.0, 0.0, 0.0], top_k=3, returning=["body"])
    top = hits[0]
    # id + score always retained (identity + rank)
    assert set(top.keys()) == {"id", "score", "body"}
    assert top["body"] == "alpha"


def test_returning_can_request_structural_fields():
    g = _graph()
    hits = g.select("Doc").vector_search("summary", [1.0, 0.0, 0.0, 0.0], top_k=1, returning=["title", "type"])
    assert set(hits[0].keys()) == {"id", "score", "title", "type"}


def test_returning_empty_list_is_id_and_score_only():
    g = _graph()
    hits = g.select("Doc").vector_search("summary", [1.0, 0.0, 0.0, 0.0], top_k=1, returning=[])
    assert set(hits[0].keys()) == {"id", "score"}


def test_returning_in_to_df():
    g = _graph()
    df = g.select("Doc").vector_search("summary", [1.0, 0.0, 0.0, 0.0], top_k=3, to_df=True, returning=["body"])
    assert set(df.columns) == {"id", "score", "body"}
    assert len(df) == 3


def test_full_hit_contract_survives_save_load(tmp_path):
    """The default 'all properties' contract must hold after a columnar
    save/reload — the case the operator hit where props went missing."""
    g = _graph()
    p = str(tmp_path / "g.kgl")
    g.save(p)
    g2 = kglite.load(p)
    hits = g2.select("Doc").vector_search("summary", [0.0, 1.0, 0.0, 0.0], top_k=1)
    top = hits[0]
    assert top["id"] == 2
    assert top["body"] == "beta"
    assert top["year"] == 2002
