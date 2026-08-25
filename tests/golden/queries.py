"""Seed queries pinned for the golden fixtures.

Any intentional output change must be re-committed via
``python tests/golden/regenerate.py``. Test-driven comparison lives in
``tests/test_golden.py``.

Two fixtures, two lists: :data:`CYPHER_QUERIES` / :data:`FIND_QUERIES` run
against the ~1,000-node social graph and are snapshotted **sorted**, because
what they pin is the row *set*. :data:`BM25_QUERIES` and
:data:`HYBRID_QUERIES` run against the 12-document text corpus and are
snapshotted **in row order**, because what they pin is the ranking — sorting
them would throw away the only thing they measure.
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


# (slug, cypher) over the same text corpus, fusing its two retrieval lanes.
# Snapshotted in row order under ``hybrid_<slug>.json``. Query vectors are
# literals rather than parameters because the snapshot runner passes no params
# — the vectors are the topic axes documented in ``build_text_corpus.py``.
HYBRID_QUERIES: list[tuple[str, str]] = [
    # d07 and d08 are the same topic in different words: only d08 contains
    # 'photosynthesis', so the keyword lane scores d07 0.0 while the vector
    # lane scores it 1.0. Fusing puts d07 above every document neither lane
    # liked — the single most important thing hybrid retrieval buys.
    (
        "vector_lane_rescues_a_keyword_miss",
        "MATCH (d:Doc) "
        "RETURN d.title AS title, "
        "text_bm25(d, 'body', 'photosynthesis') AS lexical, "
        "vector_score(d, 'body_emb', [0.0, 0.0, 1.0, 0.0]) AS semantic, "
        "score_fuse(text_bm25(d, 'body', 'photosynthesis'), "
        "vector_score(d, 'body_emb', [0.0, 0.0, 1.0, 0.0])) AS fused "
        "ORDER BY fused DESC, title ASC LIMIT 5",
    ),
    # Both fusions of the same two lanes, side by side, ordered by the
    # weighted one: 'magnetic field' is a lexical query while the query vector
    # points at astronomy, so equal weights hand the top row to BM25's
    # unbounded scale and a 1:9 tilt towards the vector lane hands it to a
    # document with no query word in it. The reorder between the two columns
    # is the point — weights are what stop one lane's scale from deciding.
    (
        "weights_reorder_two_disagreeing_lanes",
        "MATCH (d:Doc) "
        "RETURN d.title AS title, "
        "score_fuse(text_bm25(d, 'body', 'magnetic field'), "
        "vector_score(d, 'body_emb', [0.0, 0.0, 0.0, 1.0])) AS equal_weight, "
        "score_fuse(text_bm25(d, 'body', 'magnetic field'), "
        "vector_score(d, 'body_emb', [0.0, 0.0, 0.0, 1.0]), [0.1, 0.9]) AS vector_heavy "
        "ORDER BY vector_heavy DESC, title ASC LIMIT 5",
    ),
    # d11 and d12 are word-for-word permutations, so the keyword lane cannot
    # tell them apart — but d12 has no embedding. Its `fused` is its `lexical`
    # exactly: the absent lane leaves the average instead of scoring zero,
    # which is the whole of decision 5 in one snapshot.
    (
        "an_unembedded_row_keeps_its_lexical_score",
        "MATCH (d:Doc) WHERE text_bm25(d, 'body', 'alpha beta gamma') > 0 "
        "RETURN d.title AS title, "
        "text_bm25(d, 'body', 'alpha beta gamma') AS lexical, "
        "vector_score(d, 'body_emb', [1.0, 0.0, 0.0, 0.0]) AS semantic, "
        "score_fuse(text_bm25(d, 'body', 'alpha beta gamma'), "
        "vector_score(d, 'body_emb', [1.0, 0.0, 0.0, 0.0])) AS fused "
        "ORDER BY fused DESC, title ASC",
    ),
    # The Reciprocal Rank Fusion recipe CYPHER.md documents in place of an
    # rrf() scalar: rank each lane with a window function, fuse the
    # reciprocals. Pinned because it is a documented recipe, not an internal.
    (
        "reciprocal_rank_fusion_recipe",
        "MATCH (d:Doc) "
        "WITH d, rank() OVER (ORDER BY text_bm25(d, 'body', 'magnetic field') DESC) AS lex_rank, "
        "rank() OVER (ORDER BY vector_score(d, 'body_emb', [0.0, 1.0, 0.0, 0.0]) DESC) AS vec_rank "
        "RETURN d.title AS title, "
        "score_fuse(1.0 / (60 + lex_rank), 1.0 / (60 + vec_rank)) AS fused "
        "ORDER BY fused DESC, title ASC LIMIT 5",
    ),
]
