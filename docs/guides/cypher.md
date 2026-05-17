# Cypher Queries

KGLite supports a substantial Cypher subset. This page covers the essentials — see the [full Cypher reference](../reference/cypher-reference.md) for complete documentation of every clause and function.

```{note}
**Single-label note:** Each node has exactly one type. `labels(n)` returns a string, not a list. `SET n:OtherLabel` is not supported.
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

## Semantic Search in Cypher

`text_score()` enables semantic search directly in Cypher. Requires `set_embedder()` + `embed_texts()`:

```python
graph.cypher("""
    MATCH (n:Article)
    WHERE text_score(n, 'summary', 'machine learning') > 0.8
    RETURN n.title, text_score(n, 'summary', 'machine learning') AS score
    ORDER BY score DESC LIMIT 10
""")
```

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
| **Clauses** | `MATCH`, `OPTIONAL MATCH`, `WHERE`, `RETURN`, `WITH`, `ORDER BY`, `SKIP`, `LIMIT`, `UNWIND`, `UNION`/`UNION ALL`, `CREATE`, `SET`, `DELETE`, `DETACH DELETE`, `REMOVE`, `MERGE`, `EXPLAIN` |
| **Patterns** | Node `(n:Type)`, relationship `-[:REL]->`, variable-length `*1..3`, undirected `-[:REL]-`, properties `{key: val}`, `p = shortestPath(...)` |
| **WHERE** | `=`, `<>`, `<`, `>`, `<=`, `>=`, `=~` (regex), `AND`, `OR`, `NOT`, `IS NULL`, `IS NOT NULL`, `IN [...]`, `CONTAINS`, `STARTS WITH`, `ENDS WITH`, `EXISTS { pattern }`, `EXISTS(( pattern ))` |
| **Subqueries** | `count{ pattern }` (degree / filtered neighbour counts), `EXISTS{ pattern }` |
| **Functions** | `toUpper`, `toLower`, `toString`, `toInteger`, `toFloat`, `size`, `type`, `id`, `labels`, `coalesce`, `count`, `sum`, `avg`, `min`, `max`, `collect`, `std`, `text_score` |
| **Spatial** | `point`, `distance`, `contains`, `intersects`, `centroid`, `area`, `perimeter`, `latitude`, `longitude` |
| **Timeseries** | `ts_sum`, `ts_avg`, `ts_min`, `ts_max`, `ts_count`, `ts_at`, `ts_first`, `ts_last`, `ts_delta`, `ts_series` — date-string args |
| **CALL procedures** | Graph algorithms (`pagerank`, `betweenness`, `degree`, `closeness`, `louvain`, `label_propagation`, `connected_components`, `cluster`); structural validators (`orphan_node`, `self_loop`, `cycle_2step`, `missing_required_edge`, `missing_inbound_edge`, `duplicate_title`); code-tree (`affected_tests` for transitive test impact); planner (`refresh_stats` for cardinality cache refresh); `list_procedures` to enumerate. Map-syntax parameters: `CALL pagerank({damping_factor: 0.85})` |
| **Not supported** | `FOREACH`, variable-length path filters, `SET n:Label` (label mutation), multi-label |

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
