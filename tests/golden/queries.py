"""Seed queries pinned for the golden fixtures.

Any intentional output change must be re-committed via
``python tests/golden/regenerate.py``. Test-driven comparison lives in
``tests/test_golden.py``.

Two fixtures, two lists: :data:`CYPHER_QUERIES` / :data:`FIND_QUERIES` run
against the ~1,000-node social graph and are snapshotted **sorted**, because
what they pin is the row *set*. :data:`BM25_QUERIES` runs against the
12-document text corpus and is snapshotted **in row order**, because what it
pins is the ranking — sorting it would throw away the only thing it measures.
"""

from __future__ import annotations

# (slug, cypher). Slug becomes the filename stem (cypher_<slug>.json).
CYPHER_QUERIES: list[tuple[str, str]] = [
    (
        "count_by_type",
        "MATCH (n) RETURN labels(n)[0] AS label, count(n) AS c",
    ),
    (
        "count_by_connection",
        "MATCH ()-[r]->() RETURN type(r) AS rel, count(*) AS c",
    ),
    (
        "oldest_ten_people",
        "MATCH (p:Person) RETURN p.id AS id, p.age AS age ORDER BY p.age DESC, p.id ASC LIMIT 10",
    ),
    (
        "active_filter_count",
        "MATCH (p:Person) WHERE p.active = true RETURN count(p) AS c",
    ),
    (
        "salary_bucket",
        "MATCH (p:Person) WHERE p.salary >= 200000 "
        "RETURN p.id AS id, p.salary AS s ORDER BY p.salary DESC, p.id ASC LIMIT 20",
    ),
    (
        "companies_founded_before_1950",
        "MATCH (c:Company) WHERE c.founded < 1950 "
        "RETURN c.cid AS cid, c.founded AS founded ORDER BY c.founded ASC, c.cid ASC",
    ),
    (
        "person_to_company_degree_two",
        "MATCH (p:Person)-[:WORKS_AT]->(c:Company)-[:LOCATED_IN]->(pl:Place) "
        "WHERE pl.population > 500000 "
        "RETURN p.id AS pid, c.cid AS cid, pl.pid AS plid "
        "ORDER BY p.id, c.cid, pl.pid LIMIT 25",
    ),
    (
        "knows_two_hop",
        "MATCH (a:Person)-[:KNOWS]->(b:Person)-[:KNOWS]->(c:Person) "
        "WHERE a.id = 0 RETURN DISTINCT c.id AS cid ORDER BY cid LIMIT 15",
    ),
    (
        "place_population_stats",
        "MATCH (pl:Place) RETURN count(pl) AS n, min(pl.population) AS pmin, max(pl.population) AS pmax",
    ),
    (
        "top_companies_by_hires",
        "MATCH (c:Company)<-[:WORKS_AT]-(p:Person) "
        "RETURN c.cid AS cid, count(p) AS hires "
        "ORDER BY hires DESC, cid ASC LIMIT 10",
    ),
    # 0.9.0 §3 — datetime field accessors. Pin year/month/day
    # extraction from `joined_at` against the golden fixture so the
    # accessors don't drift across releases.
    (
        "joined_year_distribution",
        "MATCH (p:Person) WHERE p.joined_at IS NOT NULL RETURN p.joined_at.year AS y, count(p) AS n ORDER BY y, n",
    ),
    (
        "earliest_joiners_by_month",
        "MATCH (p:Person) WHERE p.joined_at IS NOT NULL "
        "RETURN p.id AS id, p.joined_at.year AS y, p.joined_at.month AS m, p.joined_at.day AS d "
        "ORDER BY p.joined_at, p.id ASC LIMIT 10",
    ),
]

# find(name, node_type=...) — code-entity-only in this build; always empty
# on the social graph, which still pins the API contract (empty list).
FIND_QUERIES: list[tuple[str, str, str | None]] = [
    ("alice_any", "Alice", None),
    ("function_none", "execute", "Function"),
    ("place_oslo", "Oslo", None),
    ("company_labs", "Labs", None),
    ("empty", "zzzzzzzzzz", None),
]


# (slug, cypher) over the text corpus in ``build_text_corpus.py``. Snapshotted
# in row order, with the scores, under ``bm25_<slug>.json``.
BM25_QUERIES: list[tuple[str, str]] = [
    # A term in one document out-weighs a term in three, however often the
    # query repeats the common one — this is BM25's IDF doing the job a
    # stopword list would otherwise be asked to do.
    (
        "rare_term_beats_common_terms",
        "MATCH (d:Doc) "
        "RETURN d.title AS title, text_bm25(d, 'body', 'ferrofluid magnetic field') AS score "
        "ORDER BY score DESC, title ASC LIMIT 5",
    ),
    # A natural question, mostly function words. The documents that answer it
    # must outrank the ones that merely share 'the', 'of', 'a' and 'sun'.
    (
        "stopword_heavy_question",
        "MATCH (d:Doc) "
        "RETURN d.title AS title, "
        "text_bm25(d, 'body', 'how does a plant convert the light of the sun into sugar') AS score "
        "ORDER BY score DESC, title ASC LIMIT 5",
    ),
    # d11 and d12 are permutations of each other and score identically. The
    # order below is the tie-break, and it must not depend on hash iteration.
    (
        "tie_break_is_deterministic",
        "MATCH (d:Doc) WHERE text_bm25(d, 'body', 'alpha beta gamma') > 0 "
        "RETURN d.title AS title, text_bm25(d, 'body', 'alpha beta gamma') AS score "
        "ORDER BY score DESC",
    ),
    # Term-frequency saturation plus length normalisation: d02 says 'fox'
    # three times in eleven words, d01 once in nine.
    (
        "term_frequency_saturation",
        "MATCH (d:Doc) "
        "RETURN d.title AS title, text_bm25(d, 'body', 'fox') AS score "
        "ORDER BY score DESC, title ASC LIMIT 3",
    ),
    # An indexed document sharing no query word scores 0.0, not null. The
    # snapshot is where that distinction is visible to a reader.
    (
        "no_shared_term_scores_zero",
        "MATCH (d:Doc) WHERE d.title = 'd03' RETURN d.title AS title, text_bm25(d, 'body', 'ferrofluid') AS score",
    ),
]
