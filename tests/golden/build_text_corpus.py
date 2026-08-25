"""Deterministic 12-document corpus for the BM25 golden rankings.

Small and hand-written on purpose: every ranking these documents produce has
to be explainable in one sentence, because the point of the golden snapshots is
to be *read* — the equivalence oracle and the property tests already cover
"the index agrees with a from-scratch BM25", which is a claim no human can
check by eye.

The vocabulary is engineered around three cases:

* ``ferrofluid`` appears in exactly one document, so it carries a large IDF and
  must beat a document that repeats a corpus-wide word.
* ``the`` / ``of`` / ``a`` appear nearly everywhere, so they must contribute
  almost nothing — this is what replaces a stopword list.
* Documents 11 and 12 are word-for-word permutations of each other, so a query
  matching both must break the tie deterministically, by node id.

Used by ``tests/golden/regenerate.py`` (snapshot refresh) and
``tests/test_golden.py`` (comparison).
"""

from __future__ import annotations

from kglite import KnowledgeGraph

# (doc_id, title, body). Order is the insertion order, and therefore the node
# order the tie-break falls back on.
DOCUMENTS: list[tuple[int, str, str]] = [
    (1, "d01", "the quick brown fox jumps over the lazy dog"),
    (2, "d02", "a quick brown fox is a fast fox and a clever fox"),
    (3, "d03", "the lazy dog sleeps in the sun all of the afternoon"),
    (4, "d04", "ferrofluid responds to a magnetic field"),
    (5, "d05", "the magnetic field of the earth protects the atmosphere"),
    (6, "d06", "a magnetic field is measured in tesla"),
    (7, "d07", "how does a plant convert light into sugar"),
    (8, "d08", "photosynthesis converts light into sugar inside a plant"),
    (9, "d09", "the sun is a star of the main sequence"),
    (10, "d10", "a star of the main sequence burns hydrogen in its core"),
    (11, "d11", "alpha beta gamma"),
    (12, "d12", "gamma beta alpha"),
]


def build_text_corpus(kg: KnowledgeGraph) -> KnowledgeGraph:
    """Create the corpus and build the BM25 index over ``Doc.body``."""
    kg.cypher(
        "UNWIND $docs AS d CREATE (:Doc {doc_id: d.doc_id, title: d.title, body: d.body})",
        params={"docs": [{"doc_id": doc_id, "title": title, "body": body} for doc_id, title, body in DOCUMENTS]},
    )
    kg.build_text_index("Doc", "body")
    return kg
