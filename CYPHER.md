# Cypher Reference

Full Cypher subset supported by KGLite. For a quick overview, see the [Cypher guide](https://kglite.readthedocs.io/en/latest/guides/cypher.html).

> **Single-label note:** Each node has exactly one type. `labels(n)` returns a string, not a list. `SET n:OtherLabel` is not supported.

---

## Basic Query

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

## WHERE Clause

```python
# Comparisons: =, <>, <, >, <=, >=
graph.cypher("MATCH (n:Product) WHERE n.price >= 500 RETURN n.title, n.price")

# Boolean operators: AND, OR, NOT
graph.cypher("MATCH (n:Person) WHERE n.age > 25 AND NOT n.city = 'Oslo' RETURN n.name")

# Null checks
graph.cypher("MATCH (n:Person) WHERE n.email IS NOT NULL RETURN n.name")

# String predicates: CONTAINS, STARTS WITH, ENDS WITH
graph.cypher("MATCH (n:Person) WHERE n.name CONTAINS 'ali' RETURN n.name")

# IN lists
graph.cypher("MATCH (n:Person) WHERE n.city IN ['Oslo', 'Bergen'] RETURN n.name")

# Regex matching with =~
graph.cypher("MATCH (n:Person) WHERE n.name =~ '(?i)^ali.*' RETURN n.name")
graph.cypher("MATCH (n:Person) WHERE n.email =~ '.*@example\\.com$' RETURN n.name")
```

## Relationship Properties

Relationships can have properties. Access them with `r.property` syntax:

```python
# Create relationships with properties
graph.cypher("""
    MATCH (p:Person {name: 'Alice'}), (m:Movie {title: 'Inception'})
    CREATE (p)-[:RATED {score: 5, comment: 'Excellent'}]->(m)
""")

# Access, filter, aggregate, sort by relationship properties
graph.cypher("MATCH (p)-[r:RATED]->(m) RETURN p.name, r.score, r.comment, type(r)")
graph.cypher("MATCH (p)-[r:RATED]->(m) WHERE r.score >= 4 RETURN p.name, m.title")
graph.cypher("MATCH (p)-[r:RATED]->(m) RETURN avg(r.score) AS avg_rating")
graph.cypher("MATCH ()-[r:RATED]->(m) RETURN m.title, r.score ORDER BY r.score DESC")
```

## Aggregation

```python
graph.cypher("MATCH (n:Person) RETURN n.city, count(*) AS population ORDER BY population DESC")
graph.cypher("MATCH (n:Person) RETURN avg(n.age) AS avg_age, min(n.age), max(n.age)")

# DISTINCT
graph.cypher("MATCH (n:Person) RETURN DISTINCT n.city")
graph.cypher("MATCH (n:Person) RETURN count(DISTINCT n.city) AS unique_cities")
```

## WITH Clause

```python
graph.cypher("""
    MATCH (p:Person)-[:KNOWS]->(f:Person)
    WITH p, count(f) AS friend_count
    WHERE friend_count > 3
    RETURN p.name, friend_count
    ORDER BY friend_count DESC
""")
```

## OPTIONAL MATCH

Left outer join — keeps rows even when no match:

```python
graph.cypher("""
    MATCH (p:Person)
    OPTIONAL MATCH (p)-[:KNOWS]->(f:Person)
    RETURN p.name, count(f) AS friends
""")
```

## Built-in Functions

| Function | Description |
|----------|-------------|
| `toUpper(expr)` | Convert to uppercase |
| `toLower(expr)` | Convert to lowercase |
| `toString(expr)` | Convert to string |
| `toInteger(expr)` | Convert to integer |
| `toFloat(expr)` | Convert to float |
| `size(expr)` | Length of string or list |
| `type(r)` | Relationship type |
| `id(n)` | Node ID |
| `labels(n)` | Node type (string, not list — single-label) |
| `keys(n)` / `keys(r)` | Property names of a node or relationship (as JSON list) |
| `date(str)` / `datetime(str)` | Parse date string to DateTime (`date('2020-01-15')`) |
| `coalesce(a, b, ...)` | First non-null argument |
| `range(start, end [, step])` | Generate integer list (inclusive); default step = 1 |
| `length(p)` | Path hop count |
| `nodes(p)` | Nodes in a path |
| `relationships(p)` | Relationships in a path |
| `split(str, delim)` | Split string into list |
| `replace(str, search, repl)` | Replace all occurrences |
| `substring(str, start [, len])` | Extract substring |
| `left(str, n)` / `right(str, n)` | First/last n characters |
| `trim(str)` | Remove leading/trailing whitespace |
| `ltrim(str)` / `rtrim(str)` | Left/right trim |
| `reverse(str)` | Reverse a string |
| `point(lat, lon)` | Create a geographic point |
| `distance(a, b)` | Geodesic distance (m); geometry-aware |
| `contains(a, b)` | Does a's geometry contain b? |
| `intersects(a, b)` | Do geometries intersect? |
| `centroid(n)` | Centroid of geometry → Point |
| `area(n)` | Geodesic area (m²) |
| `perimeter(n)` | Geodesic perimeter/length (m) |
| `latitude(point)` | Extract latitude from point |
| `longitude(point)` | Extract longitude from point |
| `valid_at(e, date, 'from', 'to')` | Temporal point-in-time filter (nodes or edges) |
| `valid_during(e, start, end, 'from', 'to')` | Temporal range overlap filter |
| `text_score(n, prop, query)` | Semantic similarity (auto-embeds query text; requires `set_embedder()`) |
| `text_score(n, prop, query, metric)` | With explicit metric (`'cosine'`, `'dot_product'`, `'euclidean'`) |
| `ts_sum(n.ch [, 'start'] [, 'end'])` | Sum of timeseries values (date-string range) |
| `ts_avg(n.ch [, 'start'] [, 'end'])` | Average of timeseries values |
| `ts_min(n.ch [, 'start'] [, 'end'])` | Minimum timeseries value |
| `ts_max(n.ch [, 'start'] [, 'end'])` | Maximum timeseries value |
| `ts_count(n.ch)` | Count of non-NaN timeseries values |
| `ts_at(n.ch, 'date')` | Exact timeseries key lookup |
| `ts_first(n.ch)` / `ts_last(n.ch)` | First / last non-NaN value |
| `ts_delta(n.ch, 'from', 'to')` | Value change between two time points |
| `ts_series(n.ch [, 'start'] [, 'end'])` | Extract series as `[{time, value}, ...]` |

## Spatial Functions

Built-in spatial functions for geographic queries. All node-aware functions auto-resolve geometry and location via [spatial types](https://kglite.readthedocs.io/en/latest/guides/spatial.html).

| Function | Returns | Description |
|----------|---------|-------------|
| `point(lat, lon)` | Point | Create a geographic point |
| `distance(a, b)` | Float (m) | Geodesic distance (WGS84); geometry-aware (0 if inside/touching) |
| `distance(lat1, lon1, lat2, lon2)` | Float (m) | Geodesic distance (4-arg shorthand) |
| `contains(a, b)` | Boolean | Does a's geometry contain b? (point-in-polygon or geometry containment) |
| `intersects(a, b)` | Boolean | Do geometries intersect? |
| `centroid(n)` | Point | Centroid of geometry (node or WKT string) |
| `area(n)` | Float (m²) | Geodesic area of polygon (node or WKT string) |
| `perimeter(n)` | Float (m) | Geodesic perimeter/length (node or WKT string) |
| `latitude(point)` | Float | Extract latitude component |
| `longitude(point)` | Float | Extract longitude component |

All functions accept both nodes (auto-resolved via spatial config) and raw values (WKT strings, Points).

> **Coordinate order:** `point(lat, lon)` uses **latitude-first** (geographic convention). WKT strings use **longitude-first** per OGC standard: `POLYGON((lon lat, lon lat, ...))`. These conventions differ — be careful when mixing them.

```python
# Node-aware spatial — with spatial config declared via column_types
graph.cypher("""
    MATCH (c:City), (a:Area)
    WHERE contains(a, c)
    RETURN c.name, a.name
""")

graph.cypher("""
    MATCH (a:Field), (b:Field)
    WHERE intersects(a, b) AND a <> b
    RETURN a.name, b.name
""")

graph.cypher("""
    MATCH (n:Field)
    RETURN n.name, area(n) AS area_m2, centroid(n) AS center
""")

# Geometry-aware distance
graph.cypher("""
    MATCH (a:Field), (b:Field) WHERE a <> b
    RETURN a.name, b.name, distance(a.geometry, b.geometry) AS dist
""")  # 0 if polygons touch, centroid distance otherwise

graph.cypher("""
    MATCH (n:Field)
    WHERE distance(point(60.5, 3.5), n.geometry) < 10000.0
    RETURN n.name
""")  # 0 if point inside polygon, closest boundary otherwise

# Distance filtering — cities within 100 km of Oslo
graph.cypher("""
    MATCH (n:City)
    WHERE distance(n, point(59.91, 10.75)) < 100000.0
    RETURN n.name
    ORDER BY distance(n, point(59.91, 10.75))
""")

# Aggregation with spatial
graph.cypher("""
    MATCH (a:Field), (b:Field) WHERE a <> b
    RETURN avg(distance(a, b)) AS avg_dist, std(distance(a, b)) AS std_dist
""")
```

## Temporal Functions

Date-range filtering on nodes and relationships with explicit field names.

| Function | Description |
|----------|-------------|
| `date(str)` / `datetime(str)` | Parse date string to DateTime value |
| `d.year`, `d.month`, `d.day` | Extract component from a DateTime value (use `WITH` to alias first) |
| `valid_at(entity, date, 'from_field', 'to_field')` | True if entity is active at a point in time |
| `valid_during(entity, start, end, 'from_field', 'to_field')` | True if entity's range overlaps the given interval |

**NULL semantics:** NULL `from` = valid since beginning. NULL `to` = still valid. Both NULL = always valid.

```python
# Nodes active at a point in time
graph.cypher("""
    MATCH (e:Employee)
    WHERE valid_at(e, '2020-06-15', 'hire_date', 'end_date')
    RETURN e.name
""")

# Relationships active at a point in time
graph.cypher("""
    MATCH (e:Employee)-[r:WORKS_AT]->(c:Company)
    WHERE valid_at(r, '2020-06-15', 'start_date', 'end_date')
    RETURN e.name, c.name
""")

# Overlap: entities active during a range
graph.cypher("""
    MATCH (r:Regulation)
    WHERE valid_during(r, '2020-01-01', '2022-12-31', 'effective_from', 'effective_to')
    RETURN r.name
""")

# Combine with other predicates
graph.cypher("""
    MATCH (e:Employee)-[r:WORKS_AT]->(c:Company {name: 'Acme'})
    WHERE valid_at(r, '2019-01-01', 'start_date', 'end_date')
    RETURN e.name ORDER BY e.name
""")

# Works with date() function too
graph.cypher("MATCH (e:Estimate) WHERE valid_at(e, date('2020-06-15'), 'date_from', 'date_to') RETURN count(*)")
```

## Math Functions

| Function | Description |
|----------|-------------|
| `abs(x)` | Absolute value |
| `ceil(x)` / `ceiling(x)` | Round up to integer |
| `floor(x)` | Round down to integer |
| `round(x)` | Round to nearest integer |
| `round(x, d)` | Round to `d` decimal places (e.g. `round(3.14159, 2)` → 3.14) |
| `sqrt(x)` | Square root |
| `sign(x)` | Sign: -1, 0, or 1 |
| `log(x)` / `ln(x)` | Natural logarithm (x must be > 0) |
| `log10(x)` | Base-10 logarithm (x must be > 0) |
| `exp(x)` | e^x |
| `pow(x, y)` / `power(x, y)` | x^y |
| `pi()` | π constant |
| `rand()` / `random()` | Random float [0, 1) |

## String Functions

| Function | Description |
|----------|-------------|
| `split(str, delim)` | Split string into list |
| `replace(str, search, repl)` | Replace all occurrences of `search` with `repl` |
| `substring(str, start [, len])` | Extract substring (0-indexed) |
| `left(str, n)` | First `n` characters |
| `right(str, n)` | Last `n` characters |
| `trim(str)` | Remove leading/trailing whitespace |
| `ltrim(str)` / `rtrim(str)` | Left/right trim |
| `reverse(str)` | Reverse a string |

> **Auto-coercion:** String functions accept non-string values (DateTime, numbers, booleans) and auto-convert them to strings. For example, `substring(date('2020-06-15'), 0, 4)` returns `"2020"`.

```python
graph.cypher("RETURN split('a,b,c', ',') AS parts")         # ["a", "b", "c"]
graph.cypher("RETURN replace('hello world', 'world', 'cypher') AS s")  # "hello cypher"
graph.cypher("RETURN substring('hello', 1, 3) AS s")        # "ell"
graph.cypher("RETURN left('hello', 2) AS l, right('hello', 2) AS r")  # "he", "lo"
```

## Arithmetic & String Concatenation

```python
graph.cypher("MATCH (n:Product) RETURN n.title, n.price * 1.25 AS price_with_tax")

# String concatenation with ||
graph.cypher("MATCH (n:Person) RETURN n.first || ' ' || n.last AS fullname")

# || auto-converts non-strings; null propagates
graph.cypher("RETURN 'block-' || 35 AS label")  # → "block-35"
```

## CASE Expressions

```python
# Generic form
graph.cypher("""
    MATCH (n:Person)
    RETURN n.name,
           CASE WHEN n.age >= 18 THEN 'adult' ELSE 'minor' END AS category
""")

# Simple form
graph.cypher("""
    MATCH (n:Person)
    RETURN n.name,
           CASE n.city WHEN 'Oslo' THEN 'capital' WHEN 'Bergen' THEN 'west coast' ELSE 'other' END AS region
""")
```

## List Comprehensions

`[x IN list WHERE predicate | expression]` syntax:

```python
# Map: double each number
graph.cypher("UNWIND [1] AS _ RETURN [x IN [1, 2, 3, 4, 5] | x * 2] AS doubled")
# [2, 4, 6, 8, 10]

# Filter only
graph.cypher("UNWIND [1] AS _ RETURN [x IN [1, 2, 3, 4, 5] WHERE x > 3] AS filtered")
# [4, 5]

# Filter + map
graph.cypher("UNWIND [1] AS _ RETURN [x IN [1, 2, 3, 4, 5] WHERE x > 3 | x * 2] AS result")
# [8, 10]

# With collect() — transform aggregated values
graph.cypher("""
    MATCH (p:Person)
    WITH collect(p.name) AS names
    RETURN [x IN names | toUpper(x)] AS upper_names
""")
```

> **Note:** List comprehensions require at least one row in the pipeline. Use `UNWIND [1] AS _` or a preceding `MATCH`/`WITH` to provide the row context.

## List Quantifier Predicates

`any(x IN list WHERE pred)`, `all(...)`, `none(...)`, `single(...)` — test list elements against a predicate:

| Function | Returns `true` when |
|----------|---------------------|
| `any(x IN list WHERE pred)` | At least one element satisfies the predicate |
| `all(x IN list WHERE pred)` | Every element satisfies the predicate |
| `none(x IN list WHERE pred)` | No element satisfies the predicate |
| `single(x IN list WHERE pred)` | Exactly one element satisfies the predicate |

```python
# any: at least one friend over 30
graph.cypher("""
    MATCH (p:Person)-[:KNOWS]->(f:Person)
    WITH p, collect(f.age) AS ages
    WHERE any(a IN ages WHERE a > 30)
    RETURN p.name
""")

# all: every item costs less than 100
graph.cypher("""
    MATCH (o:Order)-[:CONTAINS]->(i:Item)
    WITH o, collect(i.price) AS prices
    WHERE all(p IN prices WHERE p < 100)
    RETURN o.id
""")

# none / single
graph.cypher("RETURN none(x IN [1, 2, 3] WHERE x < 0) AS all_positive")   # true
graph.cypher("RETURN single(x IN [1, 2, 3] WHERE x = 2) AS has_one_two")  # true
```

Works in WHERE, RETURN, and WITH clauses.

## List Slicing

`expr[start..end]` syntax — slice lists with optional start/end bounds and negative indices:

```python
# Slice collected values
graph.cypher("""
    MATCH (p:Person)
    WITH collect(p.name) AS names
    RETURN names[0..3] AS first_three
""")

# Open-ended slices
graph.cypher("RETURN [1,2,3,4,5][2..] AS from_idx_2")    # [3, 4, 5]
graph.cypher("RETURN [1,2,3,4,5][..3] AS first_three")    # [1, 2, 3]

# Negative indices (from end)
graph.cypher("RETURN [1,2,3,4,5][-2..] AS last_two")      # [4, 5]
```

## Map Projections

`n {.prop1, .prop2, alias: expr}` syntax — select specific properties from a node:

```python
# Select only name and age (returns a dict per row)
graph.cypher("MATCH (p:Person) RETURN p {.name, .age} AS info")
# [{'info': {'name': 'Alice', 'age': 30}}, {'info': {'name': 'Bob', 'age': 25}}]

# Mix shorthand properties with computed values
graph.cypher("""
    MATCH (p:Person)-[:WORKS_AT]->(c:Company)
    RETURN p {.name, .age, company: c.name} AS info
""")

# System properties (id, type) work too
graph.cypher("MATCH (p:Person) RETURN p {.name, .type, .id} AS info LIMIT 1")
# [{'info': {'name': 'Alice', 'type': 'Person', 'id': 1}}]
```

## Map Literals

`{key: expr, key2: expr}` syntax — construct map objects in RETURN, WITH, or anywhere an expression is valid:

```python
# Build a map from node properties
graph.cypher("""
    MATCH (p:Person)
    RETURN {name: p.name, age: p.age} AS info
""")

# Computed values in map literals
graph.cypher("""
    MATCH (p:Person)
    RETURN {name: p.name, next_age: p.age + 1} AS info
""")

# Map literals in WITH
graph.cypher("WITH {x: 1, y: 2} AS point RETURN point")
```

## Parameters

```python
graph.cypher(
    "MATCH (n:Person) WHERE n.age > $min_age RETURN n.name, n.age",
    params={'min_age': 25}
)

# Parameters in inline pattern properties
graph.cypher(
    "MATCH (n:Person {name: $name}) RETURN n.age",
    params={'name': 'Alice'}
)

# Parameters with DataFrame output
df = graph.cypher(
    "MATCH (n:Person) WHERE n.age > $min_age RETURN n.name, n.age ORDER BY n.age",
    params={'min_age': 20}, to_df=True
)
```

## UNWIND

Expand a list into rows:

```python
graph.cypher("UNWIND [1, 2, 3] AS x RETURN x, x * 2 AS doubled")
```

## UNION

```python
graph.cypher("""
    MATCH (n:Person) WHERE n.city = 'Oslo' RETURN n.name AS name
    UNION
    MATCH (n:Person) WHERE n.age > 30 RETURN n.name AS name
""")
```

## Variable Binding in MATCH Patterns

Variables from `WITH` or `UNWIND` can be used as values in inline pattern properties:

```python
# Scalar variable in pattern property
graph.cypher("""
    WITH 'Oslo' AS city
    MATCH (p:Person {city: city})
    RETURN p.name
""")

# UNWIND + pattern variable — batch lookups
graph.cypher("""
    UNWIND ['Alice', 'Bob'] AS name
    MATCH (p:Person {name: name})
    RETURN p.name, p.age
    ORDER BY p.age
""")
```

## Variable-Length Paths

```python
# 1 to 3 hops
graph.cypher("MATCH (a:Person)-[:KNOWS*1..3]->(b:Person) WHERE a.name = 'Alice' RETURN b.name")

# Exact 2 hops
graph.cypher("MATCH (a:Person)-[:KNOWS*2]->(b:Person) RETURN a.name, b.name")
```

## WHERE EXISTS

Check for subpattern existence. Brace `{ }`, parenthesis `(( ))`, and inline pattern syntax are all supported:

```python
# Brace syntax
graph.cypher("MATCH (p:Person) WHERE EXISTS { (p)-[:KNOWS]->(:Person) } RETURN p.name")

# With optional MATCH keyword and WHERE clause inside
graph.cypher("""
    MATCH (p:Person)
    WHERE EXISTS { MATCH (p)-[:KNOWS]->(f:Person) WHERE f.age > 30 }
    RETURN p.name
""")

# Parenthesis syntax (equivalent)
graph.cypher("MATCH (p:Person) WHERE EXISTS((p)-[:KNOWS]->(:Person)) RETURN p.name")

# Inline pattern predicate (shorthand for EXISTS)
graph.cypher("MATCH (p:Person) WHERE (p)-[:KNOWS]->(:Person) RETURN p.name")

# Negation
graph.cypher("""
    MATCH (p:Person)
    WHERE NOT EXISTS { (p)-[:PURCHASED]->(:Product) }
    RETURN p.name
""")
```

## shortestPath()

BFS shortest path between two nodes. Supports directed (`->`) and undirected (`-`) syntax:

```python
# Directed — only follows edges in their defined direction
result = graph.cypher("""
    MATCH p = shortestPath((a:Person {name: 'Alice'})-[:KNOWS*..10]->(b:Person {name: 'Dave'}))
    RETURN length(p), nodes(p), relationships(p), a.name, b.name
""")

# Undirected — traverses edges in both directions (same as fluent API)
result = graph.cypher("""
    MATCH p = shortestPath((a:Person {name: 'Alice'})-[:KNOWS*..10]-(b:Person {name: 'Dave'}))
    RETURN length(p), nodes(p), relationships(p)
""")

# No path → empty list (not an error)
```

**Path functions:** `length(p)` returns hop count, `nodes(p)` returns node list, `relationships(p)` returns edge type list.

## CREATE / SET / DELETE / REMOVE / MERGE

```python
# CREATE — returns ResultView with .stats
result = graph.cypher("CREATE (n:Person {name: 'Alice', age: 30, city: 'Oslo'})")
print(result.stats['nodes_created'])  # 1

# CREATE relationship between existing nodes
graph.cypher("""
    MATCH (a:Person {name: 'Alice'}), (b:Person {name: 'Bob'})
    CREATE (a)-[:KNOWS]->(b)
""")

# SET — update properties
result = graph.cypher("MATCH (n:Person {name: 'Bob'}) SET n.age = 26, n.city = 'Stavanger'")
print(result.stats['properties_set'])  # 2

# DELETE — plain DELETE errors if node has relationships; DETACH removes all
graph.cypher("MATCH (n:Person {name: 'Alice'}) DETACH DELETE n")

# REMOVE — remove properties (id/type are immutable)
graph.cypher("MATCH (n:Person {name: 'Alice'}) REMOVE n.city")

# MERGE — match or create
graph.cypher("""
    MERGE (n:Person {name: 'Alice'})
    ON CREATE SET n.created = 'today'
    ON MATCH SET n.updated = 'today'
""")
```

## Transactions

Group multiple mutations into an atomic unit. On success, all changes apply; on exception, nothing changes.

```python
with graph.begin() as tx:
    tx.cypher("CREATE (:Person {name: 'Alice', age: 30})")
    tx.cypher("CREATE (:Person {name: 'Bob', age: 25})")
    tx.cypher("""
        MATCH (a:Person {name: 'Alice'}), (b:Person {name: 'Bob'})
        CREATE (a)-[:KNOWS]->(b)
    """)
    # Commits automatically when the block exits normally
    # Rolls back if an exception occurs

# Manual control:
tx = graph.begin()
tx.cypher("CREATE (:Person {name: 'Charlie'})")
tx.commit()   # or tx.rollback()
```

## DataFrame Output

```python
df = graph.cypher("""
    MATCH (p:Person)-[:KNOWS]->(f:Person)
    WITH p, count(f) AS friends
    RETURN p.name, p.city, friends
    ORDER BY friends DESC
""", to_df=True)
```

## EXPLAIN

Prefix any Cypher query with `EXPLAIN` to see the query plan without executing it.
Returns a `ResultView` with columns `[step, operation, estimated_rows]`:

```python
plan = graph.cypher("""
    EXPLAIN
    MATCH (p:Person)
    OPTIONAL MATCH (p)-[:KNOWS]->(f:Person)
    WITH p, count(f) AS friends
    RETURN p.name, friends
""")
for row in plan:
    print(row)
# {'step': 1, 'operation': 'Match :Person', 'estimated_rows': 500}
# {'step': 2, 'operation': 'FusedOptionalMatchAggregate', 'estimated_rows': 1}
# {'step': 3, 'operation': 'Projection (RETURN)', 'estimated_rows': None}
```

Cardinality estimates use `type_indices` counts when available, `None` otherwise.

## PROFILE

Prefix any Cypher query with `PROFILE` to execute AND collect per-clause statistics.
Returns a normal `ResultView` with results, plus a `.profile` property:

```python
result = graph.cypher("""
    PROFILE
    MATCH (p:Person)
    WHERE p.age > 30
    RETURN p.name, p.age
""")
# result contains the normal query results
for row in result:
    print(row)

# result.profile contains execution stats
for step in result.profile:
    print(step)
# {'clause': 'Match :Person', 'rows_in': 0, 'rows_out': 500, 'elapsed_us': 120}
# {'clause': 'Where', 'rows_in': 500, 'rows_out': 200, 'elapsed_us': 45}
# {'clause': 'Projection (RETURN)', 'rows_in': 200, 'rows_out': 200, 'elapsed_us': 30}
```

For non-profiled queries, `result.profile` is `None`.

## Timeseries Functions

Query time-indexed numeric data attached to nodes. All date arguments are strings (`'2020'`, `'2020-2'`, `'2020-2-15'`), and precision is validated against the data's resolution.

### Date-string syntax

| String | Depth | Matches resolution |
|--------|-------|--------------------|
| `'2020'` | year | year, month, day |
| `'2020-2'` | month | month, day |
| `'2020-2-15'` | day | day only |

**Precision rule:** Query depth must be ≤ data resolution for range functions (`ts_sum`, `ts_avg`, etc.). For exact-lookup functions (`ts_at`), query depth must equal the data resolution. Querying with day precision on month-resolution data produces an error.

### Functions

| Function | Arguments | Returns | Description |
|----------|-----------|---------|-------------|
| `ts_sum(n.channel)` | 1 | Float | Sum of all values |
| `ts_sum(n.channel, 'start')` | 2 | Float | Sum within prefix range |
| `ts_sum(n.channel, 'start', 'end')` | 3 | Float | Sum in range [start, end] inclusive |
| `ts_avg(n.channel [, 'start'] [, 'end'])` | 1-3 | Float | Average (same range rules as ts_sum) |
| `ts_min(n.channel [, 'start'] [, 'end'])` | 1-3 | Float | Minimum value in range |
| `ts_max(n.channel [, 'start'] [, 'end'])` | 1-3 | Float | Maximum value in range |
| `ts_count(n.channel)` | 1 | Integer | Count of non-NaN values |
| `ts_at(n.channel, 'date')` | 2 | Float/null | Exact key lookup (depth must match resolution) |
| `ts_first(n.channel)` | 1 | Float/null | First non-NaN value in series |
| `ts_last(n.channel)` | 1 | Float/null | Last non-NaN value in series |
| `ts_delta(n.channel, 'from', 'to')` | 3 | Float/null | Value at 'to' minus value at 'from' (prefix match) |
| `ts_series(n.channel [, 'start'] [, 'end'])` | 1-3 | List | Extract `[{time, value}, ...]` as JSON |

NaN values are skipped in all aggregation functions.

### Examples

```python
# Aggregate monthly data by year
graph.cypher("MATCH (f:Field) RETURN f.title, ts_sum(f.oil, '2020') AS prod")

# Range across months
graph.cypher("MATCH (f:Field) RETURN ts_avg(f.oil, '2020-1', '2020-6') AS h1_avg")

# Multi-year range
graph.cypher("MATCH (f:Field) RETURN ts_sum(f.oil, '2018', '2023') AS total")

# Exact month lookup
graph.cypher("MATCH (f:Field) RETURN ts_at(f.oil, '2020-3') AS march_prod")

# Change between two time points
graph.cypher("MATCH (f:Field) RETURN ts_delta(f.oil, '2019', '2021') AS change")

# Top producers
graph.cypher("""
    MATCH (f:Field)
    RETURN f.title, ts_sum(f.oil, '2020') AS prod
    ORDER BY prod DESC LIMIT 10
""")

# Filter by production threshold
graph.cypher("""
    MATCH (f:Field)
    WHERE ts_sum(f.oil, '2020') > 100.0
    RETURN f.title, ts_sum(f.oil, '2020') AS prod
""")

# Extract full series for plotting
graph.cypher("MATCH (f:Field {title: 'TROLL'}) RETURN ts_series(f.oil, '2015', '2020') AS data")

# Latest reading
graph.cypher("MATCH (s:Sensor) RETURN s.title, ts_last(s.temperature) AS latest")
```

### Precision validation

```python
# OK: year query on month data (coarser → aggregates all months)
graph.cypher("MATCH (f:Field) RETURN ts_sum(f.oil, '2020')")

# OK: month query on month data (exact match)
graph.cypher("MATCH (f:Field) RETURN ts_at(f.oil, '2020-3')")

# ERROR: day query on month data (finer than data resolution)
graph.cypher("MATCH (f:Field) RETURN ts_sum(f.oil, '2020-3-15')")
# → "Query precision 'day' (depth 3) exceeds data resolution 'month' (depth 2)"

# ERROR: year query with ts_at on month data (depth must match for exact lookup)
graph.cypher("MATCH (f:Field) RETURN ts_at(f.oil, '2020')")
# → "Exact lookup requires 2 date components for 'month' resolution, got 1"
```

## Supported Cypher Subset

| Category | Supported |
|----------|-----------|
| **Clauses** | `MATCH`, `OPTIONAL MATCH`, `WHERE`, `RETURN`, `WITH`, `ORDER BY`, `SKIP`, `LIMIT`, `UNWIND`, `UNION`/`UNION ALL`, `CREATE`, `SET`, `DELETE`, `DETACH DELETE`, `REMOVE`, `MERGE`, `EXPLAIN`, `PROFILE` |
| **Patterns** | Node `(n:Type)`, relationship `-[:REL]->`, variable-length `*1..3`, undirected `-[:REL]-`, properties `{key: val, key: $param, key: var}`, `p = shortestPath(...)` |
| **WHERE** | `=`, `<>`, `<`, `>`, `<=`, `>=`, `=~` (regex), `AND`, `OR`, `NOT`, `IS NULL`, `IS NOT NULL`, `IN [...]`, `CONTAINS`, `STARTS WITH`, `ENDS WITH`, `EXISTS { pattern WHERE ... }`, `EXISTS(( pattern ))`, inline pattern predicates, `any/all/none/single(x IN list WHERE ...)` |
| **RETURN** | `n.prop`, `r.prop`, `AS` aliases, `DISTINCT`, arithmetic `+`/`-`/`*`/`/`, string concat `\|\|`, map projections `n {.prop}`, map literals `{k: expr}`, list slicing `[i..j]` |
| **Aggregation** | `count(*)`, `count(expr)`, `sum`, `avg`/`mean`, `min`, `max`, `collect`, `std` |
| **Expressions** | `CASE WHEN...THEN...ELSE...END`, `$param`, `[x IN list WHERE ... \| expr]`, `any/all/none/single(...)` |
| **Functions** | `toUpper`, `toLower`, `toString`, `toInteger`, `toFloat`, `size`, `length`, `type`, `id`, `labels`, `keys`, `coalesce`, `date`/`datetime`, `range`, `nodes(p)`, `relationships(p)`, `round` |
| **String** | `split`, `replace`, `substring`, `left`, `right`, `trim`, `ltrim`, `rtrim`, `reverse` |
| **Math** | `abs`, `ceil`/`ceiling`, `floor`, `round`, `sqrt`, `sign`, `log`/`ln`, `log10`, `exp`, `pow`, `pi`, `rand` |
| **Spatial** | `point(lat, lon)`, `distance(a, b)`, `contains(a, b)`, `intersects(a, b)`, `centroid(n)`, `area(n)`, `perimeter(n)`, `latitude(point)`, `longitude(point)` |
| **Temporal** | `date(str)`/`datetime(str)`, `d.year`/`d.month`/`d.day`, `valid_at(entity, date, 'from', 'to')`, `valid_during(entity, start, end, 'from', 'to')` — NULL = open-ended |
| **Semantic** | `text_score(n, prop, query [, metric])` — auto-embeds query via `set_embedder()`, cosine/dot_product/euclidean |
| **Timeseries** | `ts_sum`, `ts_avg`, `ts_min`, `ts_max`, `ts_count`, `ts_at`, `ts_first`, `ts_last`, `ts_delta`, `ts_series` — date-string args with resolution validation |
| **Mutations** | `CREATE (n:Label {props})`, `CREATE (a)-[:TYPE]->(b)`, `SET n.prop = expr`, `DELETE`, `DETACH DELETE`, `REMOVE n.prop`, `MERGE ... ON CREATE SET ... ON MATCH SET` |
| **Procedures** | `CALL pagerank/betweenness/degree/closeness() YIELD node, score`, `CALL louvain/label_propagation() YIELD node, community`, `CALL connected_components() YIELD node, component`, `CALL cluster({method, ...}) YIELD node, cluster`, `CALL list_procedures()` |
| **Operators** | `+`, `-`, `*`, `/`, `\|\|` (string concat), `=~` (regex), `IN`, `STARTS WITH`, `ENDS WITH`, `CONTAINS`, `IS NULL`, `IS NOT NULL` |

## openCypher Compatibility Matrix

Clause-by-clause comparison with the openCypher specification.

### Clauses

| Clause | Status | Notes |
|--------|--------|-------|
| `MATCH` | Full | Node patterns, relationship patterns, variable-length paths, `shortestPath` |
| `OPTIONAL MATCH` | Full | Automatic fusion optimization with aggregation |
| `WHERE` | Full | All comparison, logical, string, and pattern operators |
| `RETURN` | Full | Aliases, `DISTINCT`, expressions, map projections |
| `WITH` | Full | Aggregation passthrough, grouping, chained subqueries |
| `ORDER BY` | Full | Multi-column, `ASC`/`DESC`, fused top-k optimization |
| `SKIP` / `LIMIT` | Full | |
| `UNWIND` | Full | List expansion, works with `collect()` round-trips |
| `UNION` / `UNION ALL` | Full | |
| `CREATE` | Full | Nodes, relationships, inline properties |
| `SET` | Full | `n.prop = expr`, `n += {map}` |
| `DELETE` / `DETACH DELETE` | Full | |
| `REMOVE` | Full | `REMOVE n.prop` — property removal |
| `MERGE` | Full | `ON CREATE SET`, `ON MATCH SET` |
| `EXPLAIN` | Full | Structured `ResultView` with cardinality estimates |
| `PROFILE` | Full | Execute + per-clause stats (rows_in, rows_out, elapsed_us) |
| `CALL ... YIELD` | Full | Built-in graph algorithm procedures |
| `FOREACH` | Not supported | Use `UNWIND` + `CREATE`/`SET` instead |
| `CALL {}` subqueries | Not supported | Use `WITH` chaining or multiple `cypher()` calls |
| `LOAD CSV` | Not supported | By design — use Python `pandas`/`csv` for better control |

### Expressions & Operators

| Feature | Status | Notes |
|---------|--------|-------|
| Arithmetic (`+`, `-`, `*`, `/`) | Full | |
| String concat (`\|\|`) | Full | Auto-converts non-strings |
| Comparison (`=`, `<>`, `<`, `>`, `<=`, `>=`) | Full | Three-valued logic (Null = false) |
| Boolean (`AND`, `OR`, `NOT`) | Full | |
| `IS NULL` / `IS NOT NULL` | Full | Also works as expressions in RETURN/WITH |
| `IN [list]` | Full | |
| `CONTAINS` / `STARTS WITH` / `ENDS WITH` | Full | |
| `=~` regex | Full | Compiled and cached per query |
| `CASE WHEN...THEN...ELSE...END` | Full | Simple and generic forms |
| Parameter references (`$param`) | Full | In WHERE, pattern properties, and expressions |
| List comprehensions (`[x IN list WHERE ... \| expr]`) | Full | |
| List slicing (`expr[start..end]`) | Full | Open-ended, negative indices |
| List quantifiers (`any/all/none/single(x IN list WHERE ...)`) | Full | |
| `EXISTS { pattern WHERE ... }` | Full | Brace `{}`, parenthesis `(( ))`, inline pattern, with WHERE |
| Map projections (`n {.prop1, .prop2}`) | Full | |
| Map literals (`{key: expr}`) | Full | |
| Variable binding in pattern properties | Full | `WITH val AS x MATCH ({prop: x})` |

### Scalar & Aggregation Functions

| Function | Status | Notes |
|----------|--------|-------|
| `count(*)`, `count(expr)` | Full | With `DISTINCT` support |
| `sum`, `avg`/`mean`, `min`, `max` | Full | |
| `collect` | Full | |
| `std` | Full | Standard deviation |
| `toUpper`, `toLower`, `toString` | Full | |
| `toInteger`, `toFloat` | Full | |
| `size`, `length` | Full | Strings, lists, and paths |
| `type(r)` | Full | Returns relationship type |
| `id(n)` | Full | Returns node id |
| `labels(n)` | Full | Returns single label (string, not list — single-label model) |
| `keys(n)` / `keys(r)` | Full | Returns property names as JSON list |
| `date(str)` / `datetime(str)` | Full | Parse date string to DateTime; `d.year`, `d.month`, `d.day` accessors |
| `coalesce` | Full | |
| `range(start, end [, step])` | Full | Inclusive integer range |
| `round(x [, precision])` | Full | |
| `nodes(p)`, `relationships(p)` | Full | Path decomposition |
| String functions | Full | `split`, `replace`, `substring`, `left`, `right`, `trim`, `ltrim`, `rtrim`, `reverse` — auto-coerce non-strings |
| Math functions | Full | `abs`, `ceil`, `floor`, `sqrt`, `sign`, `log`/`ln`, `log10`, `exp`, `pow`, `pi`, `rand` |
| Spatial functions | Full | `point`, `distance`, `contains`, `intersects`, `centroid`, `area`, `perimeter` |
| Temporal functions | Full | `valid_at`, `valid_during` — NULL = open-ended |

### Architectural Differences from Neo4j

| Feature | KGLite | Neo4j | Rationale |
|---------|--------|-------|-----------|
| Labels per node | Single label | Multiple | Simplifies indexing, type_indices are `HashMap<String, Vec<NodeIndex>>` |
| `labels(n)` return type | `String` | `List[String]` | Single-label model |
| `SET n:Label` | Not supported | Supported | Single-label model — change type via `SET n.type = 'NewType'` |
| Storage | In-memory (petgraph) | Disk-based | Embedded use case, explicit `save()`/`load()` |
| Transactions | Snapshot isolation + OCC | Full ACID | GIL serializes Python access; OCC catches conflicts |
| Indexing | Type indices + vector index | Schema indexes | Automatic type-based lookup, no manual `CREATE INDEX` |
| `LOAD CSV` | Not supported | Supported | Python ecosystem (pandas) preferred for data loading |
