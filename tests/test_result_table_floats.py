"""Float precision in ``repr(ResultView)``.

The table used to render every float through ``{:.2}``. PageRank scores sum to
1 across the graph and every normalized centrality is a fraction, so past a
hundred nodes an entire score column printed as ``0.00`` — the same text as a
genuine zero, and the same text for the top-ranked node as for the bottom one.
The renderer now keeps three significant digits for finite non-zero floats
under 0.01; everything else is spelled exactly as before.
"""

from __future__ import annotations

import re

import kglite

# `0.00`, `-0.00` and any other all-zero fixed-point spelling.
ALL_ZERO = re.compile(r"^-?0\.0+$")


def _ring(n: int) -> kglite.KnowledgeGraph:
    g = kglite.KnowledgeGraph()
    g.cypher(f"UNWIND range(1, {n}) AS i CREATE (:N {{id: i, title: 'n' + toString(i)}})")
    g.cypher("MATCH (a:N), (b:N) WHERE b.id = a.id + 1 CREATE (a)-[:LINKS]->(b)")
    return g


def _score_cells(rendered: str) -> list[str]:
    """The last data column of a rendered table, one string per row."""
    cells = []
    for line in rendered.splitlines():
        if not line.startswith(("│", "|")):
            continue
        parts = [p.strip() for p in re.split(r"[│|┆]", line) if p.strip()]
        if parts and parts[-1] not in {"score", "…", "..."}:
            cells.append(parts[-1])
    return cells


def test_pagerank_scores_do_not_all_print_as_zero():
    scores = _score_cells(repr(_ring(300).pagerank()))
    assert scores, "no score cells parsed out of the table"
    assert not all(ALL_ZERO.match(s) for s in scores), f"every PageRank score collapsed to zero: {sorted(set(scores))}"
    # And they are not all the *same* rendering either — the table has to
    # separate the ranked nodes it was asked to rank.
    assert len(set(scores)) > 1, f"all scores rendered identically: {scores[0]}"


def test_small_float_keeps_three_significant_digits():
    g = kglite.KnowledgeGraph()
    rendered = repr(g.cypher("RETURN 0.0003 AS tiny, -0.00025 AS neg, 0.0 AS zed, 3.14159 AS pi"))
    assert "3.00e-4" in rendered, rendered
    assert "-2.50e-4" in rendered, rendered
    # A true zero and an ordinary float are spelled exactly as before.
    assert "0.00" in rendered, rendered
    assert "3.14" in rendered, rendered


def test_band_boundary_and_large_floats_are_unchanged():
    g = kglite.KnowledgeGraph()
    rendered = repr(g.cypher("RETURN 0.01 AS edge, 12345.678 AS big, 1.0/0.0 AS inf"))
    assert "0.01" in rendered, rendered
    assert "12345.68" in rendered, rendered
    assert "e-" not in rendered, f"scientific notation leaked outside the band:\n{rendered}"
