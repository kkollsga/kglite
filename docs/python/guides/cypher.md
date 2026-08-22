# Cypher Queries

KGLite supports a substantial Cypher subset. This page covers the essentials — see the [full Cypher reference](../../reference/cypher-reference.md) for complete documentation of every clause and function.

```{note}
**Label model:** Each node has one immutable **primary** type plus optional
secondary labels. `CREATE (n:A:B)`, `SET n:B`, `REMOVE n:B`, and
`MATCH (n:A:B)` all work; `labels(n)` returns a list with the primary type
first. `SET n.type` writes an ordinary property—it does not retype the node.
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

Every query carries lightweight diagnostics, and you can profile,
explain, bound, hand one heavy query the whole machine, and even disable
individual optimizer passes. This is the machinery agents lean on to run
untrusted queries safely and to explain why a query returned what it did.

### Running one heavy query in parallel

An analytical query whose cost is *scanning* a large graph can use every core.
It is opt-in per call, because sequential is the right default for the common
case — many small queries, or many concurrent clients, where a fan-out costs
more than it saves:

```python
graph.cypher(
    "MATCH (p:Person) WHERE p.score > 0.5 RETURN p.city AS city, count(*) AS n",
    parallel=True,
)
```

`parallel=True` is a **hint, not an instruction**, and never a semantic change:
values, row order and group order are identical either way, so adding it cannot
break a caller that depends on the output.

**Reach for it on a scan, a filter, or an aggregate over a large graph.**
Measured release-mode on a 1M-node / 11M-edge graph, 10-core Apple Silicon
(4 performance + 6 efficiency cores), minimum of two agreeing runs:

| Query shape | Sequential | Parallel | |
|---|---|---|---|
| Scan + filter + `count(*)` | 34 ms | 6.5 ms | 5.2x |
| Scan + filter + grouped aggregate | 68 ms | 13 ms | 5.2x |
| Regex (`=~`) predicate | 41 ms | 6.9 ms | 6.0x |
| Grouped aggregation, 800k groups | 480 ms | 440 ms | 1.1x |
| Scan + filter + 792k-row projection | 132 ms | 122 ms | 1.1x |

The shape of that table *is* the guidance: **scanning parallelises, building
result rows does not.** Returning a million rows to Python is bound by the
allocator rather than by arithmetic, so the flag buys almost nothing there.
The "Parallel runtime" section of the
[full Cypher reference](../../reference/cypher-reference.md) lists operator by
operator what fans out and what deliberately stays sequential.

**A small query stays sequential however it is flagged.** Each operator gates
on the *real* candidate count, not a planner estimate: 20,000 rows for a
predicate the engine compiles, 5,000 when a per-row expression routes through
the interpreter. Both are measured crossovers, so there is no threshold to tune
and no penalty for flagging a query that turns out to be small.

Scope — worth knowing before you go looking for a win that is not there:

- **Per query, not per graph.** There is no `set_default_parallel()`; pass the
  keyword at the call sites you profiled. The CLI spells it
  `kglite query <graph> "<cypher>" --parallel`.
- **`KnowledgeGraph.cypher()` only.** `Session.cypher()`,
  `Transaction.cypher()` and `FrozenGraph.cypher()` execute sequentially, and
  so do the Bolt and MCP servers — deliberately, because a server's cores
  belong to its concurrent clients.
- **Memory and mapped graphs fan out; `storage="disk"` graphs and graphs with
  a spatial configuration ignore the flag** and run sequentially rather than
  refusing it, so portable code needs no branch.
- The worker pool is sized from the machine's available parallelism. Set
  `KGLITE_QUERY_THREADS=N` to pin it — useful when the process shares a box
  with something else that wants the cores.

### Diagnostics (timing, timeouts, warnings)

```python
r = graph.cypher("MATCH (n:Person) RETURN n.name")
r.diagnostics
# {'elapsed_ms': 1, 'timed_out': False, 'timeout_ms': 180000, 'warnings': []}
```

The `warnings` list surfaces non-fatal advisories — the query shapes that
return nothing useful without raising: a `MATCH` against an unknown label or
relationship type, a property reference (in `WHERE`, `RETURN`, `WITH` or
`ORDER BY`) that no node of that type carries, a relationship pattern pointing
the wrong way down a one-directional edge type, and a `WHERE` comparison that a
declared property type makes vacuous. The same "did you mean?" hint interactive
users see on stderr is exposed here for programmatic / agent callers, on every
kind of query (a mutation's `MATCH` can be typo'd too) and on repeat runs of a
query the plan cache is serving:

```python
r = graph.cypher("MATCH (n:Persn) RETURN n")   # typo
r.diagnostics["warnings"]
# ["MATCH references unknown node label 'Persn' — the graph has no such
#   type, so this pattern returns no rows. Did you mean 'Person'?"]

# With `CREATE CONSTRAINT ... REQUIRE p.age IS :: INTEGER` declared:
r = graph.cypher("MATCH (p:Person) WHERE p.age > 'forty' RETURN p")
r.diagnostics["warnings"]
# ["WHERE compares Person.age (declared INTEGER) with a STRING literal
#   'forty' — a cross-type ordering comparison is null in openCypher, so this
#   filters out every row."]
```

The declared-type family reads only DDL-declared property types (`IS :: T`),
never observed metadata, and stays silent on everything it cannot guarantee —
see the "Diagnostics" section of `CYPHER.md` for the full never-warn list. It
is a warning in every schema state; `lock_schema()` does not promote it.

Surface `warnings` whenever an agent gets an empty result — it turns a silent
zero-row mystery into an actionable typo hint.

### Timeouts and row caps

```python
# Abort after 500 ms; raises kglite.CypherTimeoutError (no partial result)
graph.cypher(long_query, timeout_ms=500)

# Cap intermediate rows and retained collection/work growth. Exceeding the
# cap raises an error. Use Session/Transaction for rollback-safe writes.
graph.cypher(broad_query, max_rows=1000)

# Set graph-wide defaults (per-query args still override)
graph.set_default_timeout(30_000)
graph.set_default_max_rows(10_000)
```

In-memory graphs default to a generous deadline (shown in
`diagnostics['timeout_ms']`); pass `timeout_ms=0` to disable it. When a query
repeatedly nears its deadline, that's the signal to add an index or anchor the
pattern, not just to raise the budget.

**There is a row ceiling even when you set none.** `max_rows` is opt-in, so a
query that expands without bound — a nested `UNWIND` cross-product is the
classic — used to materialize rows until the operating system killed the
process. kglite is *embedded*: the process it kills is your application. So a
query with no `max_rows` gets a backstop of **10,000,000 materialized rows or
retained collection items**, and crossing it raises:

```text
Query materialized 10000001 rows while executing UNWIND, exceeding the safety
ceiling of 10000000 rows that applies when no max_rows is set. Add a LIMIT
clause, or set an explicit max_rows (per query: max_rows=…; per graph or
session: set_default_max_rows(…)) to choose your own ceiling.
```

It is a last line of defence, not a planner hint: it sits at roughly twice the
largest row set any legitimate query in this project materializes without
`max_rows`, so reaching it means a query is running away rather than merely
being big. Two escape hatches, both explicit: pass `max_rows=N` on the query
(or `set_default_max_rows(N)`) to choose your own ceiling instead — a number
above 10M lifts the backstop as well as lowering it — or add a `LIMIT`. Work
whose memory cost is O(1) is exempt: a `count(*)` over a 100M-node mapped graph
charges 100M *work units* and allocates nothing, so it keeps answering.

The backstop is charged **as a clause builds its rows**, not only against the
finished set, so a pattern that expands explosively is refused while it expands
rather than after it has exhausted memory. The message names which expansion
overflowed — the `MATCH` itself, a comma-pattern join, an `OPTIONAL MATCH`, an
`EXISTS { … }` or a `COUNT { … }` subquery — so a runaway inside a subquery is
not reported as the outer clause's fault.
[How deep traversal behaves](#how-deep-traversal-behaves) is the shape this
matters most for.

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

## How deep traversal behaves

`-[:KNOWS*1..8]-` is an easy thing to write and a hard thing to predict, so
here is what the depth actually costs. The short version: **what bounds a deep
traversal is the question you asked, not the engine** — the reachability
questions stop growing with depth, and the path-*counting* ones cannot,
because counting paths is exponential by definition.

**Reachability is flat once the frontier saturates.** `count(DISTINCT b)`,
`RETURN DISTINCT b`, and `EXISTS { … }` over `(a)-[:R*1..k]-(b)` run a
per-source breadth-first search whose total work is one pass over the
reachable subgraph — a node is visited once, however many paths reach it. Once
every seed has exhausted its own component, raising `k` costs nothing.
Measured release-mode on a 10,000-node scale-free graph (~40,000 `KNOWS`
edges), 50 seed nodes, Apple M4:

| `k` | `count(DISTINCT b)` | distinct nodes reached |
|---|---|---|
| 3 | 1.9 ms | 7,690 |
| 5 | 16.1 ms | 10,000 (all of them) |
| 7 | 28.5 ms | 10,000 |
| 12 | 29.0 ms | 10,000 |

On a sparse 20,000-node chain, where nothing saturates within 12 hops, the
same shape is linear in the set it reaches: 0.167 µs per reached node.

**`EXISTS` is depth-independent** — 27.5–28.2 µs across `k = 1..12` on both of
those graphs — because the search stops at the first witness. Use it whenever
the question is *whether* something is reachable rather than what is.

**Point-to-point `shortestPath()` grows sub-linearly in distance**: on that
chain, 0.28 µs per pair at 2 hops against 0.81 µs at 12 — a 2.9x spread over a
6x increase in distance, because the search runs from both ends at once.

**Counting paths is a different question, and it is exponential.** `count(*)`
over a variable-length pattern counts *paths*, not nodes, and so does every
shape with a minimum hop count of 2 or more — openCypher's trail rule (no
relationship twice in one path) makes those genuinely per-path, and no index
removes work that the answer's own size demands. The same 10,000-node graph
above:

| Shape | Cost |
|---|---|
| `*1..4`, `count(DISTINCT b)` | 6.8 ms |
| `*2..4`, `count(DISTINCT b)` | 82 ms (968,336 paths enumerated) |
| `*2..5`, `count(DISTINCT b)` | 1.2 s |

If what you want is reachability, say so — `DISTINCT`, `EXISTS`, or a `LIMIT`
— and the BFS above is what runs. If you genuinely want the paths, expect the
cost to grow with branching to the power of the depth; kglite enumerates at
roughly 11.8M paths per second, and that rate is the whole budget you have.

Two limits are worth stating plainly rather than discovering:

- **An unbounded `*` means `*1..10` here.** `*` and `*N..` cap the upper bound
  at 10 hops as a runaway guard — a deliberate divergence from openCypher.
  Spell out `*1..20` when you mean deeper.
- **The 10,000,000-row backstop applies to the expansion itself.** A pattern
  that explodes is refused while it expands, not after it has finished
  materializing, so the error arrives in seconds instead of after the process
  has run out of memory. It is still seconds and gigabytes — reaching ten
  million held matches is not free — so for an open-ended shape set your own
  `max_rows` rather than relying on the backstop to be comfortable.

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

The primary label is immutable through `SET`/`REMOVE`. To change it, recreate
the node under the new primary type and migrate its properties and edges.
This query only writes a property and therefore does **not** retype the node:

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

## Supported Cypher surface

The machine-checked [Cypher reference](../../reference/cypher-reference.md)
is the authority for clauses, expressions, procedures, and intentional
divergences. In particular, updating `FOREACH` bodies are supported; do not
use older subset lists copied from release notes as a compatibility contract.

## Structural-validator CALL procedures

Fourteen procedures surface data-integrity gaps without writing
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

See the [full Cypher reference](../../reference/cypher-reference.md) for detailed examples of every feature.
