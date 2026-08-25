# Text Search and Hybrid Retrieval

KGLite has two ranked-retrieval lanes, and they find different things.

The **lexical lane** — a BM25 index built by `build_text_index()`, queried with
the Cypher scalar `text_bm25()` — ranks documents by word overlap, weighting
each word by how rare it is in the corpus. It finds the exact term: a product
code, a surname, an error string, a rare noun. It needs no model, no vectors and
no GPU, and it answers `'ORA-01555'` correctly the first time it sees it.

The **semantic lane** — embeddings plus `vector_score()` / `search_text()`, see
{doc}`semantic-search` — ranks by meaning. It finds the paraphrase that shares
no word with the query, and it does not care that the user typed "car" and the
document says "vehicle".

Neither is a superset of the other, and the failure modes are opposite: an
embedding blurs away the rare token that *is* the query, and a keyword index
cannot see a synonym. That is why the interesting configuration is **both, in
one query** — which in KGLite is one Cypher statement rather than two systems
and a merge step, because `score_fuse()` combines the lanes per row while
ordinary `MATCH` / `WHERE` does the graph part around them.

| You want… | Lane |
|---|---|
| Exact terms, codes, names, identifiers | lexical (`text_bm25`) |
| Paraphrase, synonymy, "something like this" | semantic (`vector_score`) |
| A short query typed by a human | both, fused |
| No embedding model in the deployment | lexical only |
| Non-text similarity (images, feature vectors) | semantic only |

This guide is the Python-side walk-through. The exact Cypher semantics of
`text_bm25()` and `score_fuse()` — argument forms, null rules, weights — live in
the [Cypher reference](../../reference/cypher-reference.md) under *Lexical
search* and *Fusing the lexical and semantic lanes*.

## Build the index

The index is **opt-in and explicit**, like `create_index()` and
`build_vector_index()`: nothing builds one for you, and a graph that never calls
`build_text_index()` never pays for one.

One index covers one `(node type, property)` pair.

```python
import kglite
import pandas as pd

articles = pd.DataFrame({
    "id": [1, 2, 3, 4],
    "title": [
        "Low-light photosynthesis",
        "Rust memory model",
        "Shade tolerance in ferns",
        "Chlorophyll assays",
    ],
    "body": [
        "Photosynthesis in low light depends on chlorophyll density.",
        "Ownership and borrowing govern memory in Rust.",
        "Ferns tolerate deep shade and low light for long periods.",
        "A chlorophyll assay measures pigment concentration.",
    ],
})

graph = kglite.KnowledgeGraph()
graph.add_nodes(articles, "Article", "id", "title")

report = graph.build_text_index("Article", "body")
print(report)
# {'indexed': 4, 'skipped': 0, 'terms': 26}
```

`indexed` is the number of documents in the corpus, `skipped` the nodes passed
over, and `terms` the size of the resulting vocabulary.

**What counts as a document.** Every node of the type whose property holds a
string — including the empty string, which indexes as a document with no terms
and still counts in the corpus statistics. A node whose property is absent or
holds a number, a list, or anything else non-string is **skipped** and counted
in `skipped`: BM25 indexes text, and a stringified number is not text.

**What counts as a word.** Runs of alphanumeric characters are terms and
everything else separates them; terms are lowercased per character. The rule is
unicode-aware — `Tromsø` is one token, not two — and it is the same rule
`text_normalize()` exposes, applied identically at build time and at query time
so the two cannot drift.

There is **no stemming and no stopword list**. That is deliberate rather than
unfinished: BM25's IDF term already discounts a word that appears in nearly
every document, statistically, without a per-language list to maintain or to be
wrong about. `'the'` in a query contributes almost nothing to the ranking
because almost every document has it. There is also no CJK segmentation — a run
of Han characters with no separator is one term.

**Memory.** Roughly **26 bytes per token** on a near-worst-case synthetic corpus
(2,000 documents of 40 tokens drawn from a 5,000-term vocabulary, so almost
every token is a distinct term — 2.08 MB for 80,000 tokens, measured). Real
prose repeats words inside a document, which collapses two postings into one
with a higher frequency, so it lands lower. Multiply your token count by 26
bytes for a ceiling: a corpus of 100,000 documents averaging 200 tokens is
20 million tokens, so budget half a gigabyte and expect to use less.

**Storage modes.** The default (in-memory) and `'mapped'` backends build the
index. The `'disk'` backend refuses, loudly:

```text
build_text_index('Article', 'body') is not supported on a disk-backed graph:
the BM25 index is heap-resident, and building one over a graph sized for the
disk backend is the memory cliff that backend exists to avoid. Use the default
(in-memory) or 'mapped' storage mode.
```

That is the point of the disk backend — it exists so a graph larger than RAM
stays queryable, and a heap-resident inverted index over it would undo that
silently.

`has_text_index(node_type, property)` reports whether one is built;
`drop_text_index(node_type, property)` removes it and returns whether there was
one.

## Rank with `text_bm25()`

`text_bm25(n, 'property', 'query text')` is an ordinary Cypher scalar returning
that row's BM25 relevance, so it composes with everything else in a query:

```python
rows = graph.cypher("""
    MATCH (a:Article)
    RETURN a.title AS title,
           text_bm25(a, 'body', $q) AS score
    ORDER BY score DESC
    LIMIT 10
""", params={"q": "low light chlorophyll"})

print(rows.to_df())
#                       title     score
# 0  Low-light photosynthesis  2.052358
# 1  Shade tolerance in ferns  1.239125
# 2        Chlorophyll assays  0.763694
# 3         Rust memory model  0.000000
```

Two results are deliberately different answers:

- **`0.0`** — the document was searched and shares no word with the query.
- **`null`** — the index holds no document for that row: it was created after
  the build and has not been caught up yet, or its property is absent or not a
  string.

Collapsing them would make an index that is quietly behind the graph look like
a corpus with no matches. Ties break by node id, so an unchanged corpus returns
the same order every time.

Calling `text_bm25()` on a `(type, property)` with no index is an **error**
naming `build_text_index` — never a silent column of nulls.

### The shape that gets the fast path

This exact shape —

> `RETURN … text_bm25(n, prop, q) AS score` · `ORDER BY score DESC` · `LIMIT k`,
> over a `MATCH` that binds the whole indexed node type

— plans as a single operator that asks the index's posting lists for their own
best *k* documents instead of scoring every row. Measured on a 100,000-document
synthetic corpus (release build, Apple Silicon, 2026-08-25, p50 over 200
rounds):

| Query | Per-row scoring | Postings top-k |
|---|---|---|
| Two terms, the rarer in ~30 documents | 30.8 ms | **7.5 ms** |
| Opens with a near-stopword (nothing to prune) | 25.2 ms | **21.4 ms** |

So a search over a six-figure corpus is single-digit to low-tens of
milliseconds, and the cost now follows the query's selectivity instead of being
flat. Both paths are exact: candidates are scored through the same kernel in the
same summation order, so the rows and their order are identical either way.

The operator hands the query back to ordinary per-row scoring whenever the index
cannot answer it alone — a `WHERE` that makes the rows a subset of the corpus,
`ORDER BY … ASC`, a per-row property or query argument, an index that has fallen
behind, or fewer matching documents than the `LIMIT` asks for. Those queries
answer *exactly* the same, just without the shortcut:

```python
rows = graph.cypher("""
    MATCH (a:Article)
    WHERE text_bm25(a, 'body', $q) > 0
    RETURN a.title AS title, text_bm25(a, 'body', $q) AS score
    ORDER BY score DESC
    LIMIT 5
""", params={"q": "chlorophyll"})

print(rows.to_df())
#                       title     score
# 0        Chlorophyll assays  0.763694
# 1  Low-light photosynthesis  0.684119
```

Filter first when the filter is what makes the query fast (a year, an author, a
one-hop traversal); leave the `WHERE` off and let the postings do the work when
you are ranking the whole corpus.

## The freshness contract

A text index does not follow writes. It **records** them and folds them in when
a query next reads it. That one sentence has three consequences worth stating
plainly, because they are the whole contract.

**Writes are never slowed.** Recording a creation is a comparison of one node
slot against a high-water mark — O(1) per bulk operation, nothing per row — so
`add_nodes()` into an indexed graph runs at the speed it would without one.
Edges never touch a text index at all. A graph with no text index pays a single
branch. This is not a tuning claim; it is why the design is watermark-based, and
there are committed benchmark cells that fail if bulk ingest into an indexed
graph diverges from the unindexed control.

**Queries catch up, up to a limit.** When a query reads the index, an
outstanding delta at or under that index's `auto_refresh_limit` (default 1,000
documents) is folded in first, inline, and the query sees fresh scores without
anyone calling anything. The limit is set to 1 below only so the *next* example
can cross it; leave it at the default in real code:

```python
small = kglite.KnowledgeGraph()
small.add_nodes(articles, "Article", "id", "title")
small.build_text_index("Article", "body", auto_refresh_limit=1)

small.cypher("CREATE (a:Article {id: 5, title: 'Moss', body: 'low light moss beds'})")
print(small.cypher("SHOW INDEXES").to_df()[["name", "type", "stale", "delta"]])
#            name      type  stale  delta
# 0  Article.body  FULLTEXT   True      1

rows = small.cypher("""
    MATCH (a:Article) RETURN a.title AS title, text_bm25(a, 'body', 'low light') AS score
    ORDER BY score DESC LIMIT 3
""")
print(rows.to_df())
#                       title     score
# 0                      Moss  1.307173
# 1  Low-light photosynthesis  1.018472
# 2  Shade tolerance in ferns  0.917187

print(rows.warnings)
# []
print(small.cypher("SHOW INDEXES").to_df()[["stale", "delta"]].to_dict("records"))
# [{'stale': False, 'delta': 0}]
```

The new node scored, nothing was rebuilt by hand, and the index came back clean.

**`auto_refresh_limit` is a document count, not a time budget** — and it is worth
knowing why, because it is the one place the honest number is not the flattering
one. Folding one document in splices into the posting list of *every* term that
document uses, and those lists grow with the corpus, so the per-document cost
rises with index size (measured: 0.08 ms per document over a 20,000-document
corpus, 0.4 ms over a 100,000-document one). Past roughly 1,500 documents,
folding costs more than rebuilding the index outright — so the catch-up
**rebuilds instead**. A refresh therefore costs the cheaper of the two and never
more than one rebuild, whatever you set the limit to; raising the limit far above
that crossover buys rebuilds, not an ever-slower fold.

**Over the limit, the index says so.** It serves what it has, scores the rows it
has no document for `null`, and attaches a warning naming the delta and the call
that fixes it:

```python
for i in range(6, 9):
    small.cypher(f"CREATE (a:Article {{id: {i}, title: 'Extra {i}', body: 'low light algae'}})")

rows = small.cypher("""
    MATCH (a:Article) RETURN a.title AS title, text_bm25(a, 'body', 'low light') AS score
    ORDER BY score DESC LIMIT 4
""")
print(rows.to_df()["score"].isna().sum())
# 3
for w in rows.warnings:
    print(w)
# text index 'Article.body' is stale: up to 3 documents are unindexed, over its
# auto_refresh_limit of 1 — those rows score null. Rebuild with
# build_text_index('Article', 'body').
```

A read-only graph gets the same treatment with the reason named, since a query
may not write to it:

```python
small.read_only(True)
rows = small.cypher("MATCH (a:Article) RETURN text_bm25(a, 'body', 'low light') AS score")
print(rows.warnings[0])
# text index 'Article.body' is stale: up to 3 documents are unindexed, and this
# graph is read-only, so a query cannot catch it up — those rows score null.
# Rebuild with build_text_index('Article', 'body').
small.read_only(False)
```

The point is that you never have to guess: a query that reads a behind-the-graph
index says so, in the result, with the number.

### Checking and repairing freshness

`SHOW INDEXES` reports `stale` and `delta` for every opt-in index. `delta` is an
upper bound on the documents a catch-up would re-read; both columns are `null`
on index kinds that are maintained on every write and have nothing to report.
`graph.schema()` lists which indexes exist (`'Article.body [text]'`) without the
freshness columns — use `SHOW INDEXES` when the question is how far behind.

Calling `build_text_index()` again rebuilds the index wholesale — the route back
from any delta, at any time. Omitting `auto_refresh_limit` on that call keeps
whatever the existing index used, so a rebuild does not quietly restore the
default.

Two things are **not** staleness:

- **Deletes.** Deleting a node prunes its document immediately, at the delete.
  It has to: the freed node slot is handed to the next node created, and an
  orphaned document would be inherited by it and score as a ghost.
- **`vacuum()`.** It renumbers every node, so it **drops** text indexes
  wholesale. Rebuild after vacuuming — the same rule HNSW indexes have always
  had. Auto-refreshing across a vacuum would be a hidden full rebuild inside
  whatever query happened to run next.

A text index is **saved with the graph**. `save()` writes it into the `.kgl` as
its own self-describing section carrying its resolved column, its refresh
ceiling and its staleness, and `kglite.load()` restores all of it — a reloaded
index that was stale is still stale by the same delta. The section is a
rebuildable cache, not a format break: a graph with no text index writes
byte-identical files to before, older files load unchanged, and a section a
build cannot read is skipped rather than refused (rebuild it in that case).

## Both lanes in one query

This is the reason the lexical lane exists here rather than in a separate
package. `score_fuse()` combines several ranked lanes into one number per row,
so keyword relevance and semantic similarity rank the same query in a single
statement — with the graph filters and traversals of ordinary Cypher around
them.

```python
graph.set_embeddings("Article", "body", {
    1: [1.0, 0.0],
    2: [0.0, 1.0],
    3: [0.9, 0.4],
    4: [0.8, 0.2],
})

hits = graph.cypher("""
    MATCH (a:Article)
    RETURN a.title AS title,
           score_fuse(text_bm25(a, 'body', $q),
                      vector_score(a, 'body_emb', $qv)) AS score
    ORDER BY score DESC
    LIMIT 3
""", params={"q": "low light", "qv": [1.0, 0.1]})

print(hits.to_df())
#                       title     score
# 0  Low-light photosynthesis  1.181638
# 1  Shade tolerance in ferns  1.094407
# 2        Chlorophyll assays  0.494731
```

(In a real deployment the vectors come from `embed_texts()` and an embedder —
see {doc}`semantic-search`. Two-dimensional literals are used here so the
example runs without a model.)

The lanes weigh equally by default. A **trailing list** weights them, in
argument order — relative, so `[3, 1]` and `[0.75, 0.25]` rank identically:

```python
hits = graph.cypher("""
    MATCH (a:Article)
    RETURN a.title AS title,
           score_fuse(text_bm25(a, 'body', $q),
                      vector_score(a, 'body_emb', $qv), [0.7, 0.3]) AS score
    ORDER BY score DESC
    LIMIT 3
""", params={"q": "low light", "qv": [1.0, 0.1]})
print(hits.to_df().head(1))
#                       title     score
# 0  Low-light photosynthesis  1.256278
```

**An absent lane leaves the average; it does not score zero.** A lane returns
`null` for a row it could not *see* — a document the text index has not caught
up with, a node with no stored embedding — and that row keeps the score of the
lanes that did run, with the absent lane's weight leaving the denominator too.
Zero would mean "this lane looked and found nothing", which would rank a
document one lane could not see below a document both lanes actively disliked.
The result is `null` only when every lane is absent. This is the rule that lets
hybrid retrieval work on a half-embedded corpus.

A wrong-length weights list, a negative weight and a non-numeric score are all
errors rather than a quietly different ranking.

### Reciprocal Rank Fusion

There is no `rrf()` scalar, because RRF works on each lane's **rank across the
whole result set** and a per-row scalar sees one row. It is two lines of what
already exists — window `rank()` in a `WITH`, then fuse the reciprocals:

```python
hits = graph.cypher("""
    MATCH (a:Article)
    WITH a, rank() OVER (ORDER BY text_bm25(a, 'body', $q) DESC) AS lex_rank,
            rank() OVER (ORDER BY vector_score(a, 'body_emb', $qv) DESC) AS vec_rank
    RETURN a.title AS title,
           score_fuse(1.0 / (60 + lex_rank), 1.0 / (60 + vec_rank)) AS score
    ORDER BY score DESC
    LIMIT 3
""", params={"q": "low light", "qv": [1.0, 0.1]})

print(hits.to_df().head(2))
#                       title     score
# 0  Low-light photosynthesis  0.016393
# 1  Shade tolerance in ferns  0.016001
```

Reach for RRF when the lanes' scores are on incomparable scales — BM25 is
unbounded, cosine is not — since ranks discard the magnitudes. Fuse the scores
directly when the magnitudes carry information you want. The
[Cypher reference](../../reference/cypher-reference.md) carries the same recipe
alongside the `score_fuse` semantics.

## The vector lane catches up the same way

Freshness is one shared mechanism, so an HNSW vector index behaves like a text
index where it can. Writing vectors after `build_vector_index()` no longer drops
the index: the outstanding vectors are recorded and folded in at query entry
while the delta is at or under `auto_refresh_limit`, a larger delta is served by
the exact scan (correct, and slower) until you rebuild or call
`refresh_vector_index()`, and both facts show up in `SHOW INDEXES`.

The one difference is what a *query* is allowed to do for you:

**Catch-up never embeds.** A node with no vector is not part of the delta — it
is counted in `SHOW INDEXES`' `unembedded` column and stays invisible to vector
search until you embed it. No query turns into an embedding run behind your
back; run `embed_texts()` when you mean to.

```python
vec = kglite.KnowledgeGraph()
vec.add_nodes(articles, "Article", "id", "title")
vec.set_embeddings("Article", "body", {1: [1.0, 0.0], 2: [0.0, 1.0], 3: [0.9, 0.4], 4: [0.8, 0.2]})
vec.build_vector_index("Article", "body")

vec.cypher("CREATE (a:Article {id: 9, title: 'New', body: 'low light lichen'})")
print(vec.cypher("SHOW INDEXES").to_df()[["type", "stale", "delta", "unembedded"]].to_dict("records"))
# [{'type': 'VECTOR', 'stale': False, 'delta': 0, 'unembedded': 1}]

vec.add_embeddings("Article", "body", {9: [0.6, 0.5]})
print(vec.cypher("SHOW INDEXES").to_df()[["stale", "delta", "unembedded"]].to_dict("records"))
# [{'stale': True, 'delta': 1, 'unembedded': 0}]

print(vec.refresh_vector_index("Article", "body"))
# 1
```

What still drops a vector index is a change to the slot layout it addresses:
deleting an embedded node, rolling that delete back, and `vacuum()`. Rebuild
after those.

## See also

- {doc}`semantic-search` — embedders, `embed_texts()`, HNSW tuning, and the
  recall numbers on hard corpora.
- [Cypher reference](../../reference/cypher-reference.md) — `text_bm25()`,
  `score_fuse()`, `SHOW INDEXES`, and the window functions RRF uses.
- {doc}`data-loading` — getting the documents in before you index them.
- {doc}`ai-agents` — exposing a retrieval graph to an agent.
