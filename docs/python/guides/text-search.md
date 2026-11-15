# Text Search and Hybrid Retrieval

KGLite has two ranked-retrieval lanes, and they find different things.

The **lexical lane** is a BM25 index built by `build_text_index()` and queried with the Cypher scalar `text_bm25()`.

- It ranks documents by word overlap, weighting each word by how rare it is in the corpus.
- It finds the exact term: a product code, a surname, an error string, a rare noun.
- It needs no model, no vectors and no GPU.
- It answers `'ORA-01555'` correctly the first time it sees it.

The **semantic lane** is embeddings plus `vector_score()` / `search_text()`; see {doc}`semantic-search`.

- It ranks by meaning.
- It finds the paraphrase that shares no word with the query.
- It does not care that the user typed "car" and the document says "vehicle".

Neither lane is a superset of the other, and the failure modes are opposite. An embedding blurs away the rare token that *is* the query. A keyword index cannot see a synonym.

The interesting configuration is therefore **both, in one query**. In KGLite that is one Cypher statement, not two systems and a merge step. `score_fuse()` combines the lanes per row, and ordinary `MATCH` / `WHERE` does the graph part around them.

| You want… | Lane |
|---|---|
| Exact terms, codes, names, identifiers | lexical (`text_bm25`) |
| Paraphrase, synonymy, "something like this" | semantic (`vector_score`) |
| A short query typed by a human | both, fused |
| No embedding model in the deployment | lexical only |
| Non-text similarity (images, feature vectors) | semantic only |

This guide is the Python-side walk-through. The exact Cypher semantics of `text_bm25()` and `score_fuse()` (argument forms, null rules, weights) live in the [Cypher reference](../../reference/cypher-reference.md) under *Lexical search* and *Fusing the lexical and semantic lanes*.

## Build the index

The index is **opt-in and explicit**, like `create_index()` and `build_vector_index()`. Nothing builds one for you, and a graph that never calls `build_text_index()` never pays for one.

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

The report has three keys:

- `indexed`: the number of documents in the corpus.
- `skipped`: the nodes passed over.
- `terms`: the size of the resulting vocabulary.

### What counts as a document

A string, or a list containing only strings and nulls, produces one document.

- List members are separated by spaces.
- Nulls are ignored.
- Repeated words retain their frequency.
- Empty strings and empty or all-null lists are empty documents. They count in corpus statistics.

An absent property, another type, or any non-string/non-null list member skips the whole document. It is counted in `skipped`. Values are never stringified.

### What counts as a word

Runs of alphanumeric characters are terms. Everything else separates them. Terms are lowercased per character.

The rule is unicode-aware: `Tromsø` is one token, not two. It is the same rule `text_normalize()` exposes. It is applied identically at build time and at query time, so the two cannot drift.

There is **no stemming and no stopword list**. That is deliberate, not unfinished.

- BM25's IDF term already discounts a word that appears in nearly every document, statistically. There is no per-language list to maintain or to be wrong about.
- `'the'` in a query contributes almost nothing to the ranking, because almost every document has it.
- There is no CJK segmentation. A run of Han characters with no separator is one term.

### Memory

The index costs roughly **26 bytes per token** on a near-worst-case synthetic corpus. That corpus has 2,000 documents of 40 tokens drawn from a 5,000-term vocabulary, so almost every token is a distinct term. Measured: 2.08 MB for 80,000 tokens.

Real prose repeats words inside a document. That collapses two postings into one with a higher frequency, so real corpora land lower.

Multiply your token count by 26 bytes for a ceiling. A corpus of 100,000 documents averaging 200 tokens is 20 million tokens, so budget half a gigabyte and expect to use less.

### Storage modes

The default (in-memory) and `'mapped'` backends build the index. The `'disk'` backend refuses, loudly:

```text
build_text_index('Article', 'body') is not supported on a disk-backed graph:
the BM25 index is heap-resident, and building one over a graph sized for the
disk backend is the memory cliff that backend exists to avoid. Use the default
(in-memory) or 'mapped' storage mode.
```

That is the point of the disk backend. It exists so a graph larger than RAM stays queryable, and a heap-resident inverted index over it would undo that silently.

### Checking and dropping

- `has_text_index(node_type, property)` reports whether one is built.
- `drop_text_index(node_type, property)` removes it and returns whether there was one.

## Rank with `text_bm25()`

`text_bm25(n, 'property', 'query text')` is an ordinary Cypher scalar. It returns that row's BM25 relevance, so it composes with everything else in a query:

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

- **`0.0`**: the document was searched and shares no word with the query.
- **`null`**: the index holds no document for that row. Either the row was created after the build and has not been caught up yet, or its property is absent or is neither text nor a list containing only text and nulls.

Collapsing them would make an index that is quietly behind the graph look like a corpus with no matches. Ties break by node id, so an unchanged corpus returns the same order every time.

Calling `text_bm25()` on a `(type, property)` with no index is an **error** naming `build_text_index`. It is never a silent column of nulls.

### The shape that gets the fast path

This exact shape gets a fast path:

> `RETURN … text_bm25(n, prop, q) AS score` · `ORDER BY score DESC` · `LIMIT k`,
> over a `MATCH` that binds the whole indexed node type

It plans as a single operator. The operator asks the index's posting lists for their own best *k* documents, instead of scoring every row.

Measured on a 100,000-document synthetic corpus (release build, Apple Silicon, 2026-08-25, p50 over 200 rounds):

| Query | Per-row scoring | Postings top-k |
|---|---|---|
| Two terms, the rarer in ~30 documents | 30.8 ms | **7.5 ms** |
| Opens with a near-stopword (nothing to prune) | 25.2 ms | **21.4 ms** |

A search over a six-figure corpus is therefore single-digit to low-tens of milliseconds. The cost follows the query's selectivity instead of being flat.

Both paths are exact. Candidates are scored through the same kernel in the same summation order, so the rows and their order are identical either way.

The operator hands the query back to ordinary per-row scoring whenever the index cannot answer it alone:

- a `WHERE` that makes the rows a subset of the corpus,
- `ORDER BY … ASC`,
- a per-row property or query argument,
- an index that has fallen behind.

Those queries answer *exactly* the same, just without the shortcut.

**Fewer matching documents than the `LIMIT` asks for is not one of them.** The operator keeps the documents it found and completes the answer from the corpus, since every remaining document scores `0.0`. That case used to fall back, which made the most selective queries the most expensive ones: 39.7 ms against 0.62 ms for the same query one `k` lower, on a 200,000-document corpus.

```python
rows = graph.cypher("""
    MATCH (a:Article)
    RETURN a.title AS title, text_bm25(a, 'body', $q) AS score
    ORDER BY score DESC
    LIMIT 5
""", params={"q": "chlorophyll"})

print(rows.to_df())
#                       title     score
# 0        Chlorophyll assays  0.763694
# 1  Low-light photosynthesis  0.684119
# 2         A history of moss  0.000000
# 3          Alpine transects  0.000000
# 4        Bogs of the Baltic  0.000000
```

`LIMIT k` means *k rows*. Two documents mention chlorophyll, so the answer is those two followed by three that do not, at `0.0`. To avoid the padding, ask for fewer rows, or drop the zero scores after the fact. A `WHERE` on the score costs the postings shortcut and buys nothing, since a zero already sorts last.

Choose where the filter goes:

- Filter first when the filter is what makes the query fast (a year, an author, a one-hop traversal).
- Leave the `WHERE` off and let the postings do the work when you are ranking the whole corpus.

## The freshness contract

A text index does not follow writes. It **records** them and folds them in when a query next reads it. That sentence has three consequences, and they are the whole contract.

### Writes are never slowed

Recording a creation is a comparison of one node slot against a high-water mark. That is O(1) per bulk operation and nothing per row, so `add_nodes()` into an indexed graph runs at the speed it would without one.

- A relationship write never touches a node text index.
- A [relationship text index](#relationship-text-indexes) on its own type and property records it, by the same rule.
- A graph with no text index pays a single branch.

This is not a tuning claim. It is why the design is watermark-based. Committed benchmark cells fail if bulk ingest into an indexed graph diverges from the unindexed control.

### Queries catch up, up to a limit

When a query reads the index, an outstanding delta at or under that index's `auto_refresh_limit` (default 1,000 documents) is folded in first, inline. The query sees fresh scores without anyone calling anything.

The limit is set to 1 below only so the *next* example can cross it. Leave it at the default in real code:

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

**`auto_refresh_limit` is a document count, not a time budget.** Catch-up cost also depends on corpus size and changed content.

- Fewer than 100 index-relevant changes use direct posting edits.
- Larger deltas merge affected posting lists in batches.
- Indexes below 20,000 documents rebuild above 1,500 changes.
- Larger indexes batch through 5,000 changes before rebuilding.

These conservative boundaries reflect the measured corpus sizes, and workloads vary. The limit still decides whether to refresh at all: a delta above it is served stale even when a batch would be fast. Raising the limit can put a full rebuild in the query.

### Over the limit, the index says so

Over the limit, the index serves what it has. It scores the rows it has no document for `null`. It attaches a warning naming the delta and the call that fixes it:

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

A read-only graph gets the same treatment, with the reason named, since a query may not write to it:

```python
small.read_only(True)
rows = small.cypher("MATCH (a:Article) RETURN text_bm25(a, 'body', 'low light') AS score")
print(rows.warnings[0])
# text index 'Article.body' is stale: up to 3 documents are unindexed, and this
# graph is read-only, so a query cannot catch it up — those rows score null.
# Rebuild with build_text_index('Article', 'body').
small.read_only(False)
```

You never have to guess. A query that reads a behind-the-graph index says so, in the result, with the number.

### Routing the warning

The warning rides the result: `rows.warnings`, and `rows.diagnostics["warnings"]`. It is echoed as a `warning: …` line on stderr by default, for relationship indexes and node indexes alike.

To route it through Python's `warnings` module instead, call `kglite.set_query_warning_policy("pywarn")` once per process. A filter can then silence it, or `-W error` can turn it into an exception:

```python
import warnings

kglite.set_query_warning_policy("pywarn")
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    small.cypher("MATCH (a:Article) RETURN text_bm25(a, 'body', 'low light') AS score")
print(caught[0].category.__name__)
# UserWarning
kglite.set_query_warning_policy("stderr")
```

`"stderr"` stays the default. A warning raised through `warnings` under `-W error` would turn an advisory into an exception out of `cypher()`.

### Checking and repairing freshness

`SHOW INDEXES` reports `stale` and `delta` for every opt-in index.

- `delta` is an upper bound on the documents a catch-up would re-read.
- Both columns are `null` on index kinds that are maintained on every write and have nothing to report.
- `graph.schema()` lists which indexes exist (`'Article.body [text]'`) without the freshness columns. Use `SHOW INDEXES` when the question is how far behind.

Calling `build_text_index()` again rebuilds the index wholesale. It is the route back from any delta, at any time. Omitting `auto_refresh_limit` on that call keeps whatever the existing index used, so a rebuild does not quietly restore the default.

Two things are **not** staleness:

- **Deletes.** Deleting a node prunes its document immediately, at the delete. It has to: the freed node slot is handed to the next node created, and an orphaned document would be inherited by it and score as a ghost.
- **`vacuum()`.** It renumbers every node, so it **drops** text indexes wholesale. Rebuild after vacuuming, the same rule HNSW indexes have always had. Auto-refreshing across a vacuum would be a hidden full rebuild inside whatever query happened to run next.

### Persistence and durability

A text index is **saved with the graph**. `save()` writes it into the `.kgl` as its own self-describing section. The section carries its resolved column, its refresh ceiling and its staleness. `kglite.load()` restores all of it, so a reloaded index that was stale is still stale by the same delta.

The section is a rebuildable cache, not a format break:

- A graph with no text index writes byte-identical files to before.
- Older files load unchanged.
- A section a build cannot read is skipped rather than refused. Rebuild it in that case.

A text index is recorded in the write-ahead log, on nodes and on relationships. On a durable graph (`kglite.open(path, durable=...)`), a build or drop is a logged write.

After a crash, the reopened graph replays the log and rebuilds the index from the recovered rows. A drop made since the last checkpoint is replayed too.

The log records the declaration (type, property, refresh limit), not the postings. Replay re-tokenizes the recovered rows, so recovery of a large indexed corpus costs one build.

A vector index is logged the same way.

## Exact text: phrases, regex and positions

Use `text_match()` when you need yes or no for an exact string, a phrase or a pattern. It has no score and needs no index.

```python
graph.cypher("MATCH (a:Article) WHERE text_match(a.body, 'low light') RETURN a.title")
graph.cypher("""
    MATCH (a:Article)
    WHERE text_match(a.body, 'low  light', {mode: 'phrase', ignore_case: true})
    RETURN a.title
""")
graph.cypher(r"""
    MATCH (a:Article)
    WHERE text_match(a.body, '\\bchlorophyll\\s+[ab]\\b', {mode: 'regex'})
    RETURN a.title
""")
```

The three modes are `literal` (a substring, the default), `phrase` (whitespace in the query matches any run of whitespace, newlines included) and `regex`. `ignore_case: true` folds case. A null value gives null, and so does a value that is not a string or a list of strings.

A string predicate on a single-label `MATCH` scans the property's column instead of reading one node at a time. This applies to `text_match()`, `CONTAINS`, `toLower(n.p) CONTAINS 'x'`, `=~` and `text_match_regex()`. The predicate still decides every row, so results do not change. `PROFILE` shows `[text column scan]` when it ran. The Cypher reference lists the exact shapes and the cases that fall back to the per-node path.

To get each occurrence with its position, call `db.text.search`:

```python
rows = graph.cypher("""
    CALL db.text.search({type: 'Article', property: 'body', query: 'low light',
                         ignore_case: true, snippet_chars: 30})
    YIELD node, offset, length, snippet
    RETURN node.title AS title, offset, length, snippet
""")
```

`offset` and `length` are byte positions in the property's current value, and `char_offset` counts characters. One row comes back per occurrence, in node order and then offset order. Pass `within` a list of nodes to search only those, for example the paragraphs of one document:

```python
graph.cypher("""
    MATCH (d:Doc {id: 1})-[:CONTAINS]->(p:Para)
    WITH collect(p) AS ps
    CALL db.text.search({type: 'Para', property: 'text', query: 'rollback', within: ps})
    YIELD node, offset, snippet
    RETURN node.id, offset, snippet
""")
```

### Rank inside a scope

`db.text.rank` ranks a set of nodes you select with a pattern, with no index. The scope is the corpus: the document count, average length and document frequencies all come from its nodes.

```python
rows = graph.cypher("""
    MATCH (d:Doc {id: 1})-[:CONTAINS]->(p:Para)
    WITH collect(p) AS ps
    CALL db.text.rank({type: 'Para', property: 'text', query: 'rollback segment',
                       within: ps, `limit`: 5})
    YIELD node, score
    RETURN node.id AS id, score
""")
```

The scores equal those of a text index built over only those paragraphs. They are not comparable with `text_bm25()` scores, whose statistics come from the whole index. Each call tokenizes the scope's text, so it suits scopes. Rank a whole large type with `text_bm25()` and an index.

### Search inside a selected part of the graph

Put the text test in the pattern's own `WHERE` and `RETURN`, so the pattern selects the rows and the test runs on each, with no list of nodes in between:

```python
rows = graph.cypher("""
    MATCH (k:Tag {id: $tag})<-[:TAGGED]-(p:Para)
    WHERE text_match(p.text, 'rollback', {ignore_case: true})
    RETURN p.id AS id, text_hits(p.text, 'rollback', {ignore_case: true}) AS hits
""", params={"tag": 7})
```

`text_hits(value, query [, options])` returns the matches inside one value as a list of `{offset, length, char_offset, snippet}` maps, the fields `db.text.search` yields. Its options are those of `text_match()` plus `snippet_chars` (default 40) and `max`. A value without text gives null; a value with text and no match gives `[]`.

Use `db.text.search ... within` when you already hold the nodes in a variable. Use `db.text.rank ... within` for scores computed from the selection alone, and `text_bm25()` with an index for scores from the whole corpus.

On a synthetic graph of 58,000 documents of about 500 bytes (release build), a scope of 4,907 documents costs 2.7 ms to select and `collect`; `db.text.search ... within` adds about 0.6 ms, and `text_match()` in the pattern's `WHERE` answers in 1.0 ms. The cost follows the scope, not the type.

Choose by what you need back:

| You need | Use |
|---|---|
| Rows that contain an exact string, phrase or pattern | `text_match()` |
| Where it matches, on rows a pattern selected | `text_hits()` |
| The occurrences with offsets and snippets | `db.text.search` |
| The best rows for a loose query, ranked | `text_bm25()` |
| The best rows among a chosen part of the graph, no index | `db.text.rank` |

## Relationship text indexes

A relationship property can carry a BM25 index too: evidence on a `SUPPORTS` edge, a quote on a `CITES` edge. The relationship index is the node index's twin. It has:

- the same document rule (a string, or a list of strings and nulls joined),
- the same scoring,
- the same freshness contract.

Its lifecycle lives in Cypher, as the relationship vector index's does, so every binding reaches it through `cypher()`:

```python
graph.cypher("""
    CALL db.relationship_text_index.build({type:'SUPPORTS', text_column:'evidence'})
    YIELD indexed, skipped, terms RETURN indexed, skipped, terms
""")

rows = graph.cypher("""
    MATCH (who:Claimant)-[r:SUPPORTS]->(c:Claim)
    RETURN who.name, c.title, text_bm25(r, 'evidence', 'water damage') AS score
    ORDER BY score DESC LIMIT 10
""").to_df()
```

`text_bm25(r, 'property', 'query')` scores a relationship bound by `MATCH`, or a relationship *value*. A value can come from `collect(r)[0]`, `UNWIND`, a `CALL { }` column, or the `relationship` column of `db.relationship_embeddings.query`. It returns:

- `0.0` for a relationship sharing no word with the query,
- `null` for one the index holds no document for,
- an error naming `db.relationship_text_index.build` when no index exists.

| Procedure | Yields |
|---|---|
| `db.relationship_text_index.build({type, text_column, auto_refresh_limit?})` | `indexed`, `skipped`, `terms` |
| `db.relationship_text_index.refresh({type, text_column})` | `refreshed` |
| `db.relationship_text_index.drop({type, text_column})` | `dropped` (`false` when there was no index) |
| `db.relationship_text_index.list({type?, text_column?})` | `entity`, `type`, `text_column`, `documents`, `terms`, `skipped`, `index_state`, `delta`, `auto_refresh_limit` |

### Freshness

These writes are folded in at the next query within `auto_refresh_limit`:

- `SET` and `REMOVE`,
- `CREATE` or `MERGE` of a relationship, including one that reuses a deleted relationship's storage slot,
- `add_relationships`.

Past the limit, the query serves what the index holds and warns, until `db.relationship_text_index.refresh` runs.

- A relationship whose text changed scores its old text.
- A relationship created since scores `null`.

A deleted relationship's document is removed at the delete. `list` and `SHOW INDEXES` report `stale` and `delta` exactly as for nodes.

### Rollback

`build`, `drop`, and any catch-up a statement triggered are undone when that statement fails. The index never keeps words from a write that did not happen.

### Naming and dropping

`SHOW INDEXES` lists the index as `relationship:SUPPORTS.evidence`, type `FULLTEXT`, `entityType` `RELATIONSHIP`.

`DROP INDEX relationship:SUPPORTS.evidence` removes every relationship structure under that name. That includes the BM25 index and a relationship vector index on the same property.

### Storage

- The index is saved in the `.kgl`, and a reloaded graph keeps it.
- `vacuum()` drops it. Rebuild after vacuuming.
- Memory and mapped storage only: a disk-backed graph refuses to build one, for the same reason it refuses a node text index.

Both relationship lanes fuse exactly as node lanes do:

```python
graph.cypher("""
    MATCH ()-[r:SUPPORTS]->()
    RETURN r, score_fuse(text_bm25(r, 'evidence', $q),
                         vector_score(r, 'evidence_emb', $qv)) AS score
    ORDER BY score DESC LIMIT 10
""", params={"q": "water damage", "qv": query_vector})
```

## Both lanes in one query

This is the reason the lexical lane exists here rather than in a separate package. `score_fuse()` combines several ranked lanes into one number per row. Keyword relevance and semantic similarity rank the same query in a single statement, with the graph filters and traversals of ordinary Cypher around them.

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

In a real deployment the vectors come from `embed_texts()` and an embedder; see {doc}`semantic-search`. This example uses two-dimensional literals so it runs without a model.

### Weighting the lanes

The lanes weigh equally by default. A **trailing list** weights them, in argument order. The weights are relative, so `[3, 1]` and `[0.75, 0.25]` rank identically:

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

A wrong-length weights list, a negative weight and a non-numeric score are all errors rather than a quietly different ranking.

### Absent lanes

**An absent lane leaves the average. It does not score zero.**

A lane returns `null` for a row it could not *see*: a document the text index has not caught up with, or a node with no stored embedding. That row keeps the score of the lanes that did run. The absent lane's weight leaves the denominator too. The result is `null` only when every lane is absent.

Zero would mean "this lane looked and found nothing". That would rank a document one lane could not see below a document both lanes actively disliked. This rule is what lets hybrid retrieval work on a half-embedded corpus.

### Reciprocal Rank Fusion

There is no `rrf()` scalar. RRF works on each lane's **rank across the whole result set**, and a per-row scalar sees one row. RRF is two lines of what already exists: window `rank()` in a `WITH`, then fuse the reciprocals:

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

Choose between RRF and direct fusion:

- Use RRF when the lanes' scores are on incomparable scales (BM25 is unbounded, cosine is not). Ranks discard the magnitudes.
- Fuse the scores directly when the magnitudes carry information you want.

The [Cypher reference](../../reference/cypher-reference.md) carries the same recipe alongside the `score_fuse` semantics.

## Request exact vector retrieval

Pass a final options map to either score function:

```cypher
MATCH (a:Article)
RETURN a.title AS title,
       vector_score(a, 'body_emb', $qv, {exact: true}) AS score
ORDER BY score DESC LIMIT 5
```

- The map may be a query parameter.
- An explicit metric goes before the map, for example `text_score(a, 'body', $query, 'cosine', {exact: true})`.
- `exact` must be boolean. Unknown options are rejected.
- Exact execution does not use or refresh HNSW.
- Without this option, a compatible index may narrow candidates approximately. A filter alone does not guarantee an exact scan.
- Omitted metrics come from each node type's actual embedding store.

### Checking the route taken

The returned view's `diagnostics["retrieval"]` reports the actual search route, including with `PROFILE`.

- `actual_mode="hnsw"` means approximate candidate selection ran.
- `actual_mode="quantized"` means a quantized scan copy narrowed the candidates; the returned scores are still exact.
- `actual_mode="exact"` includes a `fallback_reason` such as `forced_exact`, `no_index`, or `stale_index`.
- Records survive nested `CALL` and `UNION` queries.
- `EXPLAIN` reports requested policy without executing it.

## The vector lane catches up the same way

Freshness is one shared mechanism, so an HNSW vector index behaves like a text index where it can. Writing vectors after `build_vector_index()` no longer drops the index.

- The outstanding vectors are recorded and folded in at query entry, while the delta is at or under `auto_refresh_limit`.
- A larger delta is served by the exact scan (correct, and slower) until you rebuild or call `refresh_vector_index()`.
- Both facts show up in `SHOW INDEXES`.

The one difference is what a *query* is allowed to do for you.

**Catch-up never embeds.** A node with no vector is not part of the delta. It is counted in `SHOW INDEXES`' `unembedded` column and stays invisible to vector search until you embed it. No query turns into an embedding run behind your back. Run `embed_texts()` when you mean to.

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

- Deleting an embedded node.
- A `vacuum()` that compacts. It drops every vector index in the graph, node and relationship, including on types that saw no delete.

Rebuild after those. `refresh_vector_index()` refuses while no index is built. A delete that a failed statement or a rolled-back transaction undoes leaves the index in place.

## See also

- {doc}`semantic-search` — embedders, `embed_texts()`, HNSW tuning, and the recall numbers on hard corpora.
- [Cypher reference](../../reference/cypher-reference.md) — `text_bm25()`, `score_fuse()`, `SHOW INDEXES`, and the window functions RRF uses.
- {doc}`data-loading` — getting the documents in before you index them.
- {doc}`ai-agents` — exposing a retrieval graph to an agent.
