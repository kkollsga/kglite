# Cypher Queries

KGLite supports a substantial Cypher subset. This page covers the essentials — see the [full Cypher reference](../reference/cypher-reference.md) for complete documentation of every clause and function.

```{note}
**Label model:** Each node has one immutable **primary** type plus optional secondary labels (multi-label since 0.10.5). `CREATE (n:A:B)`, `SET n:B`, `REMOVE n:B`, and `MATCH (n:A:B)` all work; `labels(n)` returns a list with the primary type first. Change the primary type via `SET n.type = 'NewType'`.
```

## Basic Queries

```python
result = graph.cypher("""
    MATCH (p:Person)-[:KNOWS]->(f:Person)
    WHERE p.age > 30 AND f.city = 'Oslo'
    RETURN p.name AS person, f.name AS friend, p.age AS age
    ORDER BY p.age DESC
    LIMIT 10
""")

# Read queries → ResultView (iterate, index, or convert)
for row in result:
    print(f"{row['person']} knows {row['friend']}")

# Pass to_df=True for a DataFrame
df = graph.cypher("MATCH (n:Person) RETURN n.name, n.age ORDER BY n.age", to_df=True)
```

## Mutations

```python
# CREATE
result = graph.cypher("CREATE (n:Person {name: 'Alice', age: 30, city: 'Oslo'})")
print(result.stats['nodes_created'])  # 1

# SET
graph.cypher("MATCH (n:Person {name: 'Bob'}) SET n.age = 26")

# DELETE / DETACH DELETE
graph.cypher("MATCH (n:Person {name: 'Alice'}) DETACH DELETE n")

# MERGE
graph.cypher("""
    MERGE (n:Person {name: 'Alice'})
    ON CREATE SET n.created = 'today'
    ON MATCH SET n.updated = 'today'
""")
```

## Transactions

```python
with graph.begin() as tx:
    tx.cypher("CREATE (:Person {name: 'Alice', age: 30})")
    tx.cypher("CREATE (:Person {name: 'Bob', age: 25})")
    tx.cypher("""
        MATCH (a:Person {name: 'Alice'}), (b:Person {name: 'Bob'})
        CREATE (a)-[:KNOWS]->(b)
    """)
    # Commits on exit; rolls back on exception
```

## Parameters

```python
graph.cypher(
    "MATCH (n:Person) WHERE n.age > $min_age RETURN n.name, n.age",
    params={'min_age': 25}
)
```

## Tuning and diagnostics

Every read query carries lightweight diagnostics, and you can profile,
explain, bound, and even disable individual optimizer passes. This is the
machinery agents lean on to run untrusted queries safely and to explain why
a query returned what it did.

### Diagnostics (timing, timeouts, warnings)

```python
r = graph.cypher("MATCH (n:Person) RETURN n.name")
r.diagnostics
# {'elapsed_ms': 1, 'timed_out': False, 'timeout_ms': 180000, 'warnings': []}
```

The `warnings` list surfaces non-fatal advisories — most importantly a
`MATCH` against an unknown label or relationship type, which silently returns
zero rows. The same "did you mean?" hint interactive users see on stderr is
exposed here for programmatic / agent callers:

```python
r = graph.cypher("MATCH (n:Persn) RETURN n")   # typo
r.diagnostics["warnings"]
# ["MATCH references unknown node label 'Persn' — the graph has no such
#   type, so this pattern returns no rows. Did you mean 'Person'?"]
```

Surface `warnings` whenever an agent gets an empty result — it turns a silent
zero-row mystery into an actionable typo hint.

### Timeouts and row caps

```python
# Abort after 500 ms; rows reflect the partial set, diagnostics['timed_out'] is True
graph.cypher(long_query, timeout_ms=500)

# Cap the result set
graph.cypher(broad_query, max_rows=1000)

# Set graph-wide defaults (per-query args still override)
graph.set_default_timeout(30_000)
graph.set_default_max_rows(10_000)
```

In-memory graphs default to a generous deadline (shown in
`diagnostics['timeout_ms']`); pass `timeout_ms=0` to disable it. When a query
repeatedly nears its deadline, that's the signal to add an index or anchor the
pattern, not just to raise the budget.

### Interrupting a query (Ctrl-C)

A long-running **read** can be interrupted with `Ctrl-C` — it raises
`KeyboardInterrupt` and aborts promptly, rather than blocking until the
deadline. This works from a REPL or notebook (the interactive, single-query
case) on POSIX platforms, and applies to `KnowledgeGraph.cypher`,
`Session.cypher`, and `FrozenGraph.cypher`. The graph is left unchanged.

```python
# In a notebook: a runaway scan is now Ctrl-C-able
rows = graph.cypher("MATCH (a),(b),(c) RETURN count(*)", timeout_ms=0)
# ^ press Ctrl-C -> KeyboardInterrupt, instead of waiting
```

Interruption shares the engine's deadline checkpoints, so the same advice
applies: if you're routinely interrupting a query, anchor it or add an index.
In-place mutations (`CREATE` / `SET` / `DELETE` on a live graph) and
multi-statement transactions remain bounded by the deadline rather than
Ctrl-C. On non-POSIX platforms the deadline still applies; Ctrl-C mid-query
does not.

### EXPLAIN and PROFILE

```python
# EXPLAIN — show the optimized plan without running it
graph.cypher("EXPLAIN MATCH (n:Person) WHERE n.age > 25 RETURN n")

# PROFILE — run it and report per-clause row counts + timing
r = graph.cypher("PROFILE MATCH (n:Person) RETURN n.name")
r.profile
# [{'clause': 'Match :Person', 'rows_in': 0, 'rows_out': 2, 'elapsed_us': 1},
#  {'clause': 'Return', 'rows_in': 2, 'rows_out': 2, 'elapsed_us': 0}]
```

`rows_in` / `rows_out` per clause make it obvious where a query explodes
(a `Match` emitting far more rows than the next clause keeps is the usual
culprit — add a `WHERE` or an index upstream).

### Disabling optimizer passes (debugging)

If you suspect an optimizer pass changed results or regressed performance,
disable passes by name to bisect:

```python
kglite.cypher_pass_names()          # → ['fold_or_to_in', 'push_where_into_match.1', ...]
graph.cypher(query, disabled_passes=['fold_or_to_in'])
```

Comparing a query with and without a pass is the supported way to confirm a
planner bug before filing it.

## Semantic Search in Cypher

`text_score()` enables semantic search directly in Cypher — no
separate vector store, no manual join between vector hits and graph
state. Requires `set_embedder()` + `embed_texts()`:

```python
graph.cypher("""
    MATCH (n:Article)
    WHERE text_score(n, 'summary', 'machine learning') > 0.8
    RETURN n.title, text_score(n, 'summary', 'machine learning') AS score
    ORDER BY score DESC LIMIT 10
""")
```

### Why this matters

The same query handles three concerns in one round-trip:

1. **Semantic ranking** — `text_score()` returns a cosine-similarity
   score against the registered embedder.
2. **Structural filtering** — every Cypher clause is available
   alongside the score: `MATCH` patterns, `WHERE` predicates,
   property lookups, type filters.
3. **Graph traversal** — once you've found relevant nodes, traverse
   their neighbourhood in the same query.

Concretely, this query ranks chunks by semantic similarity, then
walks back to the parent document for provenance:

```python
graph.cypher("""
    MATCH (c:Chunk)-[:OF_PAGE]->(p:Page)<-[:HAS_PAGE]-(d:Document)
    WHERE text_score(c, 'text', $query) > 0.7
    RETURN d.title AS document,
           p.page_number AS page,
           c.text AS excerpt,
           text_score(c, 'text', $query) AS relevance
    ORDER BY relevance DESC
    LIMIT 20
""", params={"query": "deferred revenue recognition"})
```

A vector-DB + graph-DB combo would split this into two queries — a
top-k vector search returning IDs, then a separate graph query
joining on those IDs. With `text_score()` inside Cypher the planner
sees both halves at once, and the round-trip is one query.

### Filter cohorts before ranking

`text_score()` evaluates per row in the projected pipeline, so
upstream filters narrow the set you're scoring:

```python
graph.cypher("""
    MATCH (c:Chunk)-[:OF_PAGE]->(p:Page)<-[:HAS_PAGE]-(d:Document)
    WHERE d.year >= 2024 AND d.publisher = 'Q4'
    WITH c, d
    WHERE text_score(c, 'text', $query) > 0.7
    RETURN d.title, c.text, text_score(c, 'text', $query) AS score
    ORDER BY score DESC LIMIT 10
""", params={"query": "..."})
```

Cheap structural filters first → semantic scoring only on the
surviving cohort.

## Edge provenance via reified nodes

kglite enforces at-most-one edge per `(source, target, edge_type)`.
A second `add_connections` (or `MERGE`) for the same triple updates
the existing edge's properties rather than creating a parallel one.
That keeps the storage layer dense — but if you need to track *who
applied the edge, when, and why*, you need provenance per
application, not one shared property bag.

The pattern is to **reify the relationship as a node**. Instead of:

```cypher
(:Chunk)-[:TAGGED_AS {by_agent, applied_at}]->(:Tag)
```

…model the tagging itself as a node, with the tag and the agent as
edges off it:

```cypher
(:Chunk)-[:TAGGED_AS]->(:Tagging {by_agent, applied_at})-[:OF_TAG]->(:Tag)
```

Now each application is its own `Tagging` node — two agents tagging
the same chunk with the same tag produce two distinct `Tagging`
nodes carrying their own `by_agent` / `applied_at`. Query for the
tagging history of a chunk:

```python
graph.cypher("""
    MATCH (c:Chunk {id: $cid})-[:TAGGED_AS]->(t:Tagging)-[:OF_TAG]->(tag:Tag)
    RETURN tag.name AS tag,
           t.by_agent AS agent,
           t.applied_at AS when
    ORDER BY t.applied_at DESC
""", params={"cid": "chunk_42"})
```

The cost is one extra node per application + two edges where you'd
have one. The gain is unconstrained provenance + the ability to
attach additional context (confidence score, source, supersession
relationships) to each application.

Use reification when you need:

- Per-application metadata that differs across applications of the
  "same" relationship.
- An audit trail (when / who / why each application happened).
- The ability to delete or supersede individual applications
  without affecting others.

For one-shot relationships (a `Person` works at one `Company` —
attributes belong on the edge), the at-most-one constraint is
exactly what you want and reification adds noise.

## Multi-label nodes

A node has a **primary type** (set at creation, immutable via
label mutation) plus optional **secondary labels** added through
Cypher or the `add_label` pymethod. The primary type drives the
columnar storage layout; secondaries are a parallel index. Match
either kind transparently:

```cypher
CREATE (a:Agent:LLM:Reviewer {id: 'strict-1', model: 'sonnet'})

MATCH (n:Reviewer) RETURN n              -- secondary-only is fine
MATCH (n:Agent:Reviewer) RETURN n        -- AND-intersect across labels
MATCH (n) WHERE 'Reviewer' IN labels(n)  -- equivalent
```

Add or remove labels on existing nodes:

```cypher
MATCH (a:Agent {id: $id}) SET a:Verified            -- add one
MATCH (a:Agent {id: $id}) SET a:Verified:Reviewer    -- add several
MATCH (a:Agent {id: $id}) REMOVE a:Verified         -- remove one
```

The primary label is immutable through `SET`/`REMOVE`. To
retype a node, use the type property:

```cypher
MATCH (n:Article {id: $id}) SET n.type = 'BlogPost'
```

From Python, the same surface is available without Cypher:

```python
g.add_nodes(df, 'Agent', 'id', 'name', labels=['Reviewer'])
g.add_label('Agent', ['agent-7'], 'OnCall')
g.remove_label('Agent', ['agent-7'], 'OnCall')
```

### Use multi-label or subtype edges?

| If you want… | Use… |
|---|---|
| Classification tags (`Reviewer`, `Verified`, `Disputed`) | Multi-label |
| Hierarchy with shared properties (`Method` *is a* `Callable`) | Subtype edge `(:Method)-[:KIND_OF]->(:Callable)` |
| Per-application provenance | Reified `Tagging` node (see section above) |

## Count Subqueries

`count { ... }` evaluates an inline pattern and returns the number
of matches. Useful in `WITH` / `RETURN` to compute per-row degree
or filtered neighbour counts without a separate aggregating
sub-query:

```python
graph.cypher("""
    MATCH (p:Person)
    WITH p, count{ (p)-[:KNOWS]->() } AS friend_count
    WHERE friend_count > 5
    RETURN p.name, friend_count
    ORDER BY friend_count DESC LIMIT 20
""")
```

The pattern inside `count { … }` is independently bound — `p`
references the outer `MATCH`. Combine with typed relationships and
WHERE clauses inside the braces for finer control:

```python
graph.cypher("""
    MATCH (post:Post)
    RETURN post.title,
           count{ (post)<-[:LIKES]-(:User) } AS likes,
           count{ (post)<-[:COMMENTS_ON]-(c:Comment) WHERE c.flagged } AS flagged_comments
""")
```

## Supported Cypher Subset

| Category | Supported |
|----------|-----------|
| **Clauses** | `MATCH`, `OPTIONAL MATCH`, `WHERE`, `RETURN`, `WITH`, `ORDER BY`, `SKIP`, `LIMIT`, `UNWIND`, `UNION`/`UNION ALL`, `CALL { ... }` (read subqueries), `CREATE`, `SET`, `DELETE`, `DETACH DELETE`, `REMOVE`, `MERGE`, `EXPLAIN` |
| **Patterns** | Node `(n:Type)`, multi-label `(n:Type:Role)` (AND-intersect), relationship `-[:REL]->`, variable-length `*1..3`, undirected `-[:REL]-`, properties `{key: val}`, `p = shortestPath(...)` |
| **WHERE** | `=`, `<>`, `<`, `>`, `<=`, `>=`, `=~` (regex), `AND`, `OR`, `NOT`, `IS NULL`, `IS NOT NULL`, `IN [...]`, `CONTAINS`, `STARTS WITH`, `ENDS WITH`, `EXISTS { pattern }`, `EXISTS(( pattern ))` |
| **Subqueries** | `count{ pattern }` (degree / filtered neighbour counts), `EXISTS{ pattern }`, `CALL { ... }` read subqueries — uncorrelated (cartesian-combined) + correlated (`CALL { WITH p ... }`, per outer row); v1 excludes writes / `UNION` / unit subqueries in the body |
| **Functions** | `toUpper`, `toLower`, `toString`, `toInteger`, `toFloat`, `size`, `type`, `id`, `labels`, `coalesce`, `count`, `sum`, `avg`, `min`, `max`, `collect`, `std`, `text_score` |
| **Spatial** | `point`, `distance`, `contains`, `intersects`, `centroid`, `area`, `perimeter`, `latitude`, `longitude` |
| **Timeseries** | `ts_sum`, `ts_avg`, `ts_min`, `ts_max`, `ts_count`, `ts_at`, `ts_first`, `ts_last`, `ts_delta`, `ts_series` — date-string args |
| **CALL procedures** | Graph algorithms (`pagerank`, `betweenness`, `degree`, `closeness`, `louvain`, `label_propagation`, `connected_components`, `cluster`); structural validators (`orphan_node`, `self_loop`, `cycle_2step`, `missing_required_edge`, `missing_inbound_edge`, `duplicate_title`); code-tree (`affected_tests` for transitive test impact); planner (`refresh_stats` for cardinality cache refresh); `list_procedures` to enumerate. Map-syntax parameters: `CALL pagerank({damping_factor: 0.85})` |
| **Label mutation** | `SET n:Label`, `SET n:A:B` (multi), `REMOVE n:Label`, `CREATE (n:A:B)`; primary label is immutable via these — use `SET n.type = 'NewType'` to retype |
| **Not supported** | `FOREACH`, variable-length path filters |

## Structural-validator CALL procedures

Six procedures surface data-integrity gaps without writing
`WHERE NOT EXISTS` patterns yourself. Each binds `node` (or
`node_a, node_b`) — compose freely with WHERE / ORDER BY / LIMIT /
aggregation as you would any Cypher row.

| Procedure | What it finds | Required params |
|---|---|---|
| `orphan_node` | nodes with zero edges in any direction | `type` |
| `self_loop` | `(n)-[:edge]->(n)` self-loops | `type`, `edge` |
| `cycle_2step` | reciprocal pairs `a-[:edge]->b-[:edge]->a` | `type`, `edge` |
| `missing_required_edge` | nodes lacking outbound `edge` (direction-validated) | `type`, `edge` |
| `missing_inbound_edge` | nodes lacking inbound `edge` (direction-validated) | `type`, `edge` |
| `duplicate_title` | one row per node whose title is shared with another node of same type | `type` |

```cypher
// Standalone — find Wellbores with no production licence
CALL missing_required_edge({type: 'Wellbore', edge: 'IN_LICENCE'})
YIELD node
RETURN node.id, node.title

// Composed — cross-reference flagged nodes against a query result
MATCH (l:Licence {title: '057'})<-[:IN_LICENCE]-(w:Wellbore)
WITH collect(w.id) AS pl057
CALL missing_required_edge({type: 'Wellbore', edge: 'DRILLED_BY'}) YIELD node
WHERE node.id IN pl057
RETURN count(node) AS pl057_missing_drilled_by

// Aggregated duplicates — one row per group
CALL duplicate_title({type: 'Prospect'}) YIELD node
WITH node.title AS title, collect(node) AS dups
WITH title, size(dups) AS dup_count
WHERE dup_count > 1
RETURN title, dup_count
ORDER BY dup_count DESC LIMIT 20
```

`missing_required_edge` and `missing_inbound_edge` validate the
`(type, edge)` pair against the graph's actual schema before
iterating. Calling `missing_inbound_edge({type: 'Wellbore', edge:
'IN_LICENCE'})` — where `IN_LICENCE` flows Wellbore→Licence —
raises `DirectionMismatch` with a suggestion to use
`missing_required_edge` instead.

For per-procedure docs (params, examples), drill in:

```python
g.describe(cypher=['orphan_node'])
g.describe(cypher=['missing_required_edge'])
```

See the [full Cypher reference](../reference/cypher-reference.md) for detailed examples of every feature.
