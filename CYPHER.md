# Cypher Reference

The Cypher KGLite supports — broad openCypher coverage, all running
**in-process** (no server). For a quick overview, see the [Cypher guide](https://kglite.readthedocs.io/en/latest/guides/cypher.html).

> **Label model:** Each node has one immutable **primary** type plus optional secondary labels (multi-label since 0.10.5). `CREATE (n:A:B)`, `SET n:B`, `REMOVE n:B`, and `MATCH (n:A:B)` all work; `labels(n)` returns a list with the primary type first. Change the primary type via `SET n.type = 'NewType'` (`REMOVE n:Primary` errors deliberately).

## Feature coverage

If you're evaluating an embedded, Cypher-queryable graph, here's the
surface at a glance — most of what you'd reach for is here, in-process:

| Area | Supported |
|---|---|
| **Reading** | `MATCH`, `OPTIONAL MATCH`, `WHERE`, `RETURN`, `WITH`, `ORDER BY` / `SKIP` / `LIMIT`, `UNWIND`, `UNION` |
| **Writing** | `CREATE`, `MERGE` (+ `ON CREATE` / `ON MATCH SET`), `SET`, `DELETE` / `DETACH DELETE`, `REMOVE`, `FOREACH (x IN list \| …)` |
| **Subqueries** | `CALL { … }` (correlated + uncorrelated), `EXISTS { … }`, `COUNT { … }` |
| **Path finding** | variable-length `-[*1..n]->`, `shortestPath(…)`, `allShortestPaths(…)`, weighted shortest path (`CALL`) |
| **Predicates** | `=, <>, <, >, <=, >=`, `AND` / `OR` / `NOT`, `IS [NOT] NULL`, `IN`, `CONTAINS` / `STARTS WITH` / `ENDS WITH`, regex `=~` |
| **Expressions** | list comprehension `[x IN xs WHERE … \| …]`, `reduce(…)`, `CASE`, list/map literals, parameters `$p` |
| **Aggregation** | `count` / `sum` / `avg` / `min` / `max` / `collect` / `percentile_cont` / `mode` / `stdev` …, `DISTINCT`, `HAVING`, window functions (`OVER`, `PARTITION BY`, ranking) |
| **Procedures** (`CALL`) | centralities (pagerank, betweenness, closeness, degree), community (louvain, leiden, label propagation), components, k-core, clustering, `triangle_count` / `transitivity`, `eccentricity` / `diameter`, `ready_set` (dependency frontier), `shortest_path_length`, `kg_knn`, structural validators (`duplicate_title`, `cycle_2step`, `parallel_edges`, …) |
| **Vector + text** | `vector_score(…)` (HNSW index, exact fallback), `text_score(…)` (pluggable embedder) — hybrid semantic + structural in one query |
| **Spatial** | `point(…)`, `distance(…)`, `wkt_within` / `intersects`, buffer / hull / union, k-NN — see [Spatial](#spatial-functions) |
| **Temporal** | `date()` / `datetime()` / `localdatetime()`, `duration(…)`, `duration.between`, date arithmetic, `valid_at` / `valid_during` — see [Temporal](#temporal-functions) |
| **Value types** | int, float, string, bool, **date**, **timestamp** (date + time), duration, point, list, map, node, relationship, path |
| **Transactions** | multi-statement with snapshot isolation + rollback (`Session` / `Transaction`) |
| **Storage** | identical Cypher across in-memory, mmap, and on-disk modes (1B+ edges) |

Deliberately **not** supported (by design, not gaps): per-write
`UNIQUE` / `NOT NULL` / PRIMARY KEY constraints — uniqueness is a
load-time concern, handled by `MERGE`, the duplicate-id warning, and the
`duplicate_title` validator (see the data-integrity recipes), so it never
costs the in-memory write path.

### Node identity — use `id` as your primary key

`id` is the **indexed** identity property, and it accepts **strings as well as
integers**:

```python
graph.cypher("CREATE (:Memory {id: 'a3f9-uuid', text: 'hello'})")
graph.cypher("MATCH (n:Memory {id: 'a3f9-uuid'}) RETURN n.text")   # O(1), indexed
graph.cypher("MATCH (n:Memory) WHERE n.id IN $keys RETURN n", params={"keys": [...]})  # multi-probe
```

Put your application key in `id` — **not** a custom property. An anchored lookup
on `id` is O(1) in every storage mode; an arbitrary property (`mid`, `key`, …) is
**not indexed**, so `MATCH (n {mid: $k})` is a full label scan (linear in node
count). Two semantics to keep in mind:

- **No uniqueness constraint** (see above): `CREATE` does not reject a duplicate
  `id` — two `CREATE (:T {id: 'k'})` make two nodes. For primary-keyed writes use
  **`MERGE`, not `CREATE`** — `MERGE (:T {id: $k})` is idempotent.
- **Matching is type-exact**: `'42'` ≠ `42`. Keep id types consistent across
  writes and reads.

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

`SET` / `REMOVE` work on a relationship variable, so you can upsert edge
properties — including via `MERGE`:

```python
graph.cypher("MATCH (p)-[r:RATED]->(m) WHERE r.score < 3 SET r.flagged = true")
graph.cypher("MATCH (p)-[r:RATED]->(m) REMOVE r.comment")
# idempotent edge upsert:
graph.cypher("""
    MATCH (p:Person {id: $a}), (m:Movie {id: $b})
    MERGE (p)-[r:RATED]->(m) ON CREATE SET r.score = $s
""", params={"a": "p1", "b": "m1", "s": 5})
```

## Aggregation

```python
graph.cypher("MATCH (n:Person) RETURN n.city, count(*) AS population ORDER BY population DESC")
graph.cypher("MATCH (n:Person) RETURN avg(n.age) AS avg_age, min(n.age), max(n.age)")

# DISTINCT
graph.cypher("MATCH (n:Person) RETURN DISTINCT n.city")
graph.cypher("MATCH (n:Person) RETURN count(DISTINCT n.city) AS unique_cities")
```

| Aggregate | Description |
|---|---|
| `count(expr)` / `count(*)` / `count(DISTINCT expr)` | Row or value count |
| `sum(expr)` | Numeric sum (preserves Int64 when source is integer) |
| `avg(expr)` / `mean(expr)` / `average(expr)` | Arithmetic mean |
| `min(expr)` / `max(expr)` | Minimum / maximum |
| `collect(expr)` / `collect(DISTINCT expr)` | Gather values into a list |
| `std(expr)` / `stdev(expr)` | Sample standard deviation (n-1) |
| `variance(expr)` / `var_samp(expr)` | Sample variance (n-1) |
| `median(expr)` | Median value |
| `percentile_cont(expr, p)` | Continuous percentile (linear interpolation), `p ∈ [0,1]` |
| `percentile_disc(expr, p)` | Discrete percentile (nearest rank), `p ∈ [0,1]` |

```python
graph.cypher("MATCH (n:Person) RETURN median(n.age), percentile_cont(n.age, 0.9)")
graph.cypher("MATCH (n:Person) RETURN variance(n.age), std(n.age)")
```

## HAVING

Post-aggregation filter — use after RETURN or WITH with aggregates:

```python
graph.cypher("MATCH (n:Person) RETURN n.city, count(*) AS pop HAVING pop > 1000")
```

Also supported on WITH:

```python
graph.cypher("""
    MATCH (n:Person)
    WITH n.city AS city, count(*) AS pop HAVING pop > 100
    RETURN city, pop
""")
```

## Window Functions

Window functions compute values across partitions of the result set without collapsing rows.

| Function | Description |
|---|---|
| `row_number() OVER (...)` | Sequential number within partition |
| `rank() OVER (...)` | Rank with gaps for ties |
| `dense_rank() OVER (...)` | Rank without gaps for ties |

OVER clause: `OVER (PARTITION BY expr [, ...] ORDER BY expr [ASC|DESC] [, ...])`

PARTITION BY is optional (whole result set = one partition). ORDER BY is required.

```python
# Global ranking
graph.cypher("MATCH (n:Person) RETURN n.name, row_number() OVER (ORDER BY n.score DESC) AS rn")

# Rank within department
graph.cypher("""
    MATCH (n:Person)
    RETURN n.name, n.dept,
           rank() OVER (PARTITION BY n.dept ORDER BY n.score DESC) AS dept_rank
""")
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
| `labels(n)` | Node labels as a list, primary type first |
| `degree(n)` | Node's total edge count (in + out; a self-loop counts twice) — e.g. `WHERE degree(n) > 100` to find hubs |
| `inDegree(n)` / `outDegree(n)` | Node's incoming / outgoing edge count |
| `keys(n)` / `keys(r)` | Property names of a node or relationship (as JSON list) |
| `properties(n)` / `properties(r)` | Full property map of a node or relationship (as JSON map) |
| `start_node(r)` | Source node of a bound relationship; supports dotted access: `start_node(r).name` |
| `end_node(r)` | Target node of a bound relationship; supports dotted access: `end_node(r).name` |
| `date(str)` / `datetime(str)` | Parse date string to DateTime (`date('2020-01-15')`) |
| `date_diff(d1, d2)` | Days between two dates (`d1 - d2`); also supports `date - date` arithmetic |
| `coalesce(a, b, ...)` | First non-null argument |
| `range(start, end [, step])` | Generate integer list (inclusive); default step = 1 |
| `head(list)` / `last(list)` | First / last element of a list (returns `null` on empty) |
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
| `text_score(n, prop, query, metric)` | With explicit metric (`'cosine'`, `'dot_product'`, `'euclidean'`, `'poincare'`) |
| `vector_score(n, prop, vector [, metric])` | Semantic similarity against a pre-computed embedding vector (pass a list of floats directly, no `set_embedder()` needed) |
| `embedding_norm(n, prop)` | L2 norm of embedding vector (hierarchy depth in Poincaré space: 0=root, ~1=leaf) |
| `ts_sum(n.ch [, 'start'] [, 'end'])` | Sum of timeseries values (date-string range) |
| `ts_avg(n.ch [, 'start'] [, 'end'])` | Average of timeseries values |
| `ts_min(n.ch [, 'start'] [, 'end'])` | Minimum timeseries value |
| `ts_max(n.ch [, 'start'] [, 'end'])` | Maximum timeseries value |
| `ts_count(n.ch)` | Count of non-NaN timeseries values |
| `ts_at(n.ch, 'date')` | Exact timeseries key lookup |
| `ts_first(n.ch)` / `ts_last(n.ch)` | First / last non-NaN value |
| `ts_delta(n.ch, 'from', 'to')` | Value change between two time points |
| `ts_series(n.ch [, 'start'] [, 'end'])` | Extract series as `[{time, value}, ...]` |

### Hybrid retrieval (RAG) over a knowledge graph

`vector_score` / `text_score` compose with ordinary `WHERE` predicates and
traversal, so semantic retrieval and graph constraints run in **one query** —
no separate vector store + join. The graph filter and the similarity ranking
combine: filter first, rank the survivors by similarity, take the top *k*.

```cypher
// Top-3 'politics' articles most similar to a query embedding —
// the category filter is applied *before* ranking, so a highly-similar
// 'sports' article is correctly excluded.
MATCH (a:Article)
WHERE a.category = 'politics'
RETURN a.title, vector_score(a, 'summary_emb', $query_vec) AS score
ORDER BY score DESC
LIMIT 3
```

```cypher
// RAG with a graph hop: retrieve passages, then pull their source document
// and author in the same query (`text_score` auto-embeds the query string;
// requires set_embedder()).
MATCH (p:Passage)-[:OF_DOC]->(d:Document)-[:WRITTEN_BY]->(author:Person)
WHERE d.published_year >= 2020
RETURN p.text, d.title, author.name,
       text_score(p, 'text', 'how does photosynthesis work?') AS score
ORDER BY score DESC
LIMIT 5
```

The embedding store key is `{text_column}_emb` (set via
`set_embeddings(node_type, text_column, {id: vector})`), so embeddings set on
the `summary` column are scored as `vector_score(a, 'summary_emb', …)`.

> **`vector_score` takes the store name, `text_score` takes the raw column.**
> `vector_score` names the store directly — `'summary_emb'`. `text_score` names
> the source *column* — `'summary'` (it resolves to `summary_emb` and auto-embeds
> the query for you). That's why the example above uses `text_score(p, 'text', …)`
> (raw column `text`), not `'text_emb'`. The Python API (`embedding_info`,
> `vector_search`, `search_text`) likewise uses the raw column name throughout;
> only Cypher's `vector_score` is in store-name terms.

> **Index-accelerated top-k.** When an HNSW index is built
> (`build_vector_index`), a whole-corpus top-k —
> `RETURN vector_score(n, prop, q) AS s ORDER BY s DESC LIMIT k` (and the
> `text_score` form) — auto-uses it, the same opt-in approximate path the
> fluent API uses. Without an index, or for a selective `WHERE` that filters
> the candidates, scoring is the exact brute-force scan. So building an index
> speeds up "search the whole corpus by similarity"; a heavily-filtered query
> stays exact.

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

### Geometry primitives

Constructive operations on WKT geometries. All accept WKT strings, node variables (auto-resolved via spatial config), or `Point` values; all return WKT strings (or boolean / float as noted).

| Function | Returns | Description |
|---|---|---|
| `geom_buffer(geom, meters)` | WKT (MultiPolygon) | Planar buffer at the geometry's centroid latitude (geo crate native; degrades far from the centroid) |
| `geom_convex_hull(geoms)` | WKT (Polygon) | Convex hull over a list of geometries; also accepts variadic args |
| `geom_union(g1, g2)` | WKT (MultiPolygon) | Polygonal union; rectangles auto-converted |
| `geom_intersection(g1, g2)` | WKT (MultiPolygon) | Polygonal intersection (empty MultiPolygon when disjoint) |
| `geom_difference(g1, g2)` | WKT (MultiPolygon) | `g1 − g2` |
| `geom_is_valid(geom)` | Boolean | OGC-style validity check |
| `geom_length(geom)` | Float (m) | Geodesic length: LineString length, polygon perimeter (sum of rings), 0 for points |

```python
# Buffer a point by 5 km
graph.cypher("RETURN geom_buffer('POINT(10.7 59.9)', 5000) AS area")

# Hull of all city centroids
graph.cypher("""
    MATCH (c:City)
    WITH collect(c.geometry) AS shapes
    RETURN geom_convex_hull(shapes) AS catchment
""")

# Union of overlapping licence areas
graph.cypher("""
    MATCH (a:Licence), (b:Licence) WHERE a.id < b.id AND intersects(a, b)
    RETURN geom_union(a.geometry, b.geometry) AS merged
""")

# LineString length (perimeter is polygon-only)
graph.cypher("RETURN geom_length('LINESTRING(10.7 59.9, 5.3 60.4)') AS m")  # ≈ 305000
```

### k-nearest-neighbour

```cypher
CALL kg_knn({lat: 60.4, lon: 5.3, target_type: 'City', k: 5})
YIELD node, distance_m
RETURN node.title, round(distance_m / 1000.0, 1) AS km
```

Looks up the *k* nodes of `target_type` closest to `(lat, lon)` (geodesic). Uses the node's `location` config for point comparisons; falls back to geometry centroid when `location` isn't configured. Nodes without spatial config are skipped silently.

## Temporal Functions

Date-range filtering on nodes and relationships with explicit field names.

| Function | Description |
|----------|-------------|
| `date(str)` / `datetime(str)` | Parse date string to DateTime value |
| `datetime()` | Today's date (no-arg form) |
| `localdatetime()` | Local wall-clock datetime as ISO-8601 string (`YYYY-MM-DDTHH:MM:SS`); 1-arg form parses/normalises a string (NULL on bad input) |
| `localtime()` / `time()` | Local wall-clock time-of-day as `HH:MM:SS` string; 1-arg form parses/normalises a string (NULL on bad input) |
| `n.d.year`, `n.d.month`, `n.d.day` | Extract component from a DateTime property (chained accessor — works in `RETURN`, `WHERE`, `ORDER BY`) |
| `n.d.dayOfWeek`, `n.d.dayOfYear`, `n.d.epochSeconds` | Other temporal field accessors |
| `duration({days: N, months: M, ...})` | Build a Duration value (see [Duration semantics](#duration-semantics) below) |
| `duration.between(d1, d2)` | Day-delta between two DateTime values, returned as a Duration |
| `date + duration({days: N})` | Add a duration to a date |
| `date_diff(d1, d2)` | Days between two dates (legacy; same as `d2 - d1` returning Int64 directly) |
| `date + N` / `date - N` | Add/subtract N days (Int64 form, kept for backward compat) |
| `date - date` | Returns a Duration (was Int64 days pre-0.9.0) |
| `valid_at(entity, date, 'from_field', 'to_field')` | True if entity is active at a point in time |
| `valid_during(entity, start, end, 'from_field', 'to_field')` | True if entity's range overlaps the given interval |

**NULL semantics:** NULL `from` = valid since beginning. NULL `to` = still valid. Both NULL = always valid.

**`localdatetime()` / `localtime()` / `time()` return strings, not a temporal Value.** KGLite's `Value::DateTime` is date-only (`NaiveDate`), so there is no time-of-day Value variant to carry sub-day precision. Rather than silently dropping the time component, these functions emit ISO-8601 strings (`localdatetime()` → `YYYY-MM-DDTHH:MM:SS`, `localtime()`/`time()` → `HH:MM:SS`). The single-string-argument form validates and normalises its input, returning NULL on unparseable input (same contract as `datetime(str)`).

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

### Duration semantics

A `Duration` value carries three independent components:

| Component | Source                              | Units              |
|-----------|-------------------------------------|--------------------|
| `months`  | `years` + `months` from constructor | calendar months    |
| `days`    | `weeks` + `days` from constructor   | clock days         |
| `seconds` | `hours` + `minutes` + `seconds`     | clock seconds      |

**Components stay separate by design.** Calendar arithmetic
(`+ duration({months: 1})`) is fundamentally different from clock
arithmetic (`+ duration({days: 30})`) because months have variable
length. `duration({months: 1, days: 5}).months` returns `1`, not `35`.

This matches Neo4j and openCypher; it diverges from Postgres
`interval` (which collapses everything into a single combined value).
Users coming from Postgres will need to know.

#### `duration.between(d1, d2)`

Computes the day-delta between two `DateTime` values. **Months and
seconds are always 0** because `Value::DateTime` is currently
date-only (`NaiveDate`); a calendar-month-aware diff requires the
`Value::DateTime` → `NaiveDateTime` refactor (deferred).

```cypher
RETURN duration.between(date('2024-08-12'), date('2026-05-02')).days
// → 628

RETURN duration.between(date('2024-08-12'), date('2026-05-02')).months
// → 0   (NOT 20 — `between` only fills in `days`)
```

#### Composite accessors

`d.years = d.months / 12`, `d.minutes = d.seconds / 60`,
`d.hours = d.seconds / 3600`. These are integer-truncated convenience
views on the underlying components — derived, not stored.

```cypher
WITH duration({months: 26, days: 100}) AS d
RETURN d.months AS m, d.years AS y, d.days AS days
// → m=26, y=2 (26/12 truncated), days=100
```

#### `DateTime ± Duration`

Calendar months in the duration are approximated as 30 days for
`DateTime` arithmetic (the `Value::DateTime` precision limitation
again). For exact month-aware addition, use a literal date.

```cypher
RETURN date('2024-01-15') + duration({days: 30})  // → 2024-02-14
RETURN date('2024-01-15') + duration({months: 1}) // → 2024-02-14 (1*30 days), NOT 2024-02-15
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
| `randomUUID()` | Random RFC 4122 v4 UUID string |

### Trigonometric Functions

All take a numeric argument and return a Float64. Angles are in
radians (use `radians(x)` / `degrees(x)` to convert). NULL in → NULL
out; a non-numeric argument also yields NULL.

| Function | Description |
|----------|-------------|
| `sin(x)`, `cos(x)`, `tan(x)` | Trig functions (radians) |
| `asin(x)`, `acos(x)`, `atan(x)` | Inverse trig functions |
| `atan2(y, x)` | Quadrant-aware arctangent of `y/x` |
| `cot(x)` | Cotangent (`1 / tan(x)`) |
| `haversin(x)` | Half-versed-sine `(1 - cos(x)) / 2` (haversine distance) |
| `degrees(x)` | Radians → degrees |
| `radians(x)` | Degrees → radians |

```cypher
// Bearing between two points (radians)
RETURN atan2(0.5, 0.5)           // → 0.7853981633974483 (π/4)
RETURN degrees(pi())             // → 180.0
```

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

## Text Predicates

Lexical similarity and fuzzy-match primitives — useful for deduplication, alias matching, and free-text indexing without dropping to Python.

| Function | Returns | Description |
|---|---|---|
| `text_edit_distance(a, b)` | `Int64` | Levenshtein edit distance (UTF-8 aware) |
| `text_normalize(s)` | `String` | Lowercase, drop punctuation, collapse whitespace |
| `text_jaccard(a, b [, sep])` | `Float64` | Token-set Jaccard similarity (default separator: whitespace) |
| `text_ngrams(s, n)` | `List<String>` | Character n-grams |
| `text_contains_any(s, needles)` | `Boolean` | True if `s` contains any needle (variadic or list arg) |
| `text_starts_with_any(s, prefixes)` | `Boolean` | True if `s` starts with any prefix (variadic or list arg) |

```python
# Edit distance — Levenshtein
graph.cypher("RETURN text_edit_distance('kitten', 'sitting') AS d")  # 3

# Normalize before comparing
graph.cypher("RETURN text_normalize('  Hello, World!  ') AS s")  # "hello world"

# Jaccard similarity
graph.cypher("RETURN text_jaccard('a b c', 'b c d') AS j")  # 0.5

# Fuzzy dedup pipeline
graph.cypher("""
    MATCH (a:Person), (b:Person) WHERE a.id < b.id
    WITH a, b, text_edit_distance(
        text_normalize(a.title), text_normalize(b.title)
    ) AS d
    WHERE d <= 2 RETURN a.title, b.title, d
""")

# Multi-prefix / multi-substring filters
graph.cypher("MATCH (n) WHERE text_starts_with_any(n.title, ['Mr.', 'Dr.', 'Prof.']) RETURN n")
graph.cypher("MATCH (n) WHERE text_contains_any(n.body, 'urgent', 'critical') RETURN n")
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

## List Properties

Node properties can be **native lists**, not just scalars. When a pandas
column passed to `add_nodes` holds Python lists, it is ingested as a real
list property (auto-detected, or forced with `column_types={'col': 'list'}`)
— stored structurally rather than stringified. List properties behave like
list literals in every list operation:

```python
# aliases is a list property, e.g. ['Bob', 'Bobby']
graph.cypher("MATCH (n:Person) WHERE 'Bobby' IN n.aliases RETURN n.name")  # membership
graph.cypher("MATCH (n:Person) UNWIND n.aliases AS a RETURN n.name, a")    # explode
graph.cypher("MATCH (n:Person) RETURN size(n.aliases) AS n_aliases")       # length
```

`IN` over a list property is true *membership* — `'Bob' IN ['Bobby']` is
false (no substring matching).

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

## Reduce (List Fold)

`reduce(acc = init, x IN list | body)` — fold a list with an accumulator. The body is evaluated once per element with `acc` and `x` bound; the final accumulator value is returned.

```python
# Sum
graph.cypher("RETURN reduce(s = 0, x IN [1, 2, 3, 4, 5] | s + x) AS total")  # 15

# Concat
graph.cypher('RETURN reduce(s = "", x IN ["a", "b", "c"] | s + x) AS r')  # "abc"

# Max via CASE
graph.cypher("""
    RETURN reduce(m = 0, x IN [5, 3, 8, 1, 7] |
        CASE WHEN x > m THEN x ELSE m END
    ) AS max_val
""")  # 8

# Pair with collect()
graph.cypher("""
    MATCH (n:Person) WITH collect(n.age) AS ages
    RETURN reduce(s = 0, x IN ages | s + x) AS total
""")
```

## JSON Parsing

`parse_json(s)` (alias `from_json(s)`) parses a JSON string into a structured
value — an object becomes a map, an array a list, with scalars typed as
int / float / bool / string. Invalid JSON or a non-string argument returns
`null` (never an error). This lets you predicate over data that is *stored* as
a JSON string rather than as graph structure.

The code graph keeps `Function.parameters`, `Class.fields`, and
`Function.signature` as JSON (the columnar store holds scalars only), so
`parse_json` is how you query inside them:

```python
# Functions that take a parameter typed `Dataset`. Each parsed element is a
# map with keys name / type_annotation / default / kind.
graph.cypher("""
    MATCH (f:Function)
    WHERE any(p IN parse_json(f.parameters) WHERE p.type_annotation = 'Dataset')
    RETURN f.qualified_name
""")

# Index into the parsed structure with bracket subscript (works on lists and
# maps). To reach a map field after a list index, chain brackets — or bind the
# parsed value with WITH and use dot access (arr[0].name).
graph.cypher("RETURN parse_json('[{\"name\":\"x\"}]')[0]['name'] AS first")  # "x"
graph.cypher("RETURN parse_json('{\"a\":1}')['a'] AS a")                      # 1
```

Combine with `any` / `all` / list comprehensions to filter or project the
parsed elements.

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

## UNION / INTERSECT / EXCEPT

Set operators combine two queries with matching column shapes.

| Operator | Semantics | Duplicate handling |
|---|---|---|
| `UNION` | Rows from either side | Deduped |
| `UNION ALL` | Rows from either side | Duplicates kept |
| `INTERSECT` | Rows present in both sides | Always deduped |
| `EXCEPT` | Rows in left but not in right | Always deduped |

```python
# UNION — combine
graph.cypher("""
    MATCH (n:Person) WHERE n.city = 'Oslo' RETURN n.name AS name
    UNION
    MATCH (n:Person) WHERE n.age > 30 RETURN n.name AS name
""")

# INTERSECT — keep names that appear on both sides
graph.cypher("""
    MATCH (n:Person) WHERE n.city = 'Oslo' RETURN n.name AS name
    INTERSECT
    MATCH (n:Person) WHERE n.age > 30 RETURN n.name AS name
""")

# EXCEPT — Oslo residents minus everyone over 30
graph.cypher("""
    MATCH (n:Person) WHERE n.city = 'Oslo' RETURN n.name AS name
    EXCEPT
    MATCH (n:Person) WHERE n.age > 30 RETURN n.name AS name
""")
```

Set operators dedupe by the projected column values; column names must match between sides (positional fallback when they don't).

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

> **Property existence:** the Neo4j-legacy `exists(n.prop)` form for
> *property*-existence is **not** supported in KGLite. Use the modern
> `WHERE n.prop IS NOT NULL` / `WHERE n.prop IS NULL` instead — those
> are property-existence checks; `EXISTS { ... }` and `EXISTS((...))`
> are *pattern*-existence checks. Writing `exists(n.prop)` returns a
> parser error that points at the `IS NOT NULL` alternative.

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

### Weighted shortest path

The fluent `shortest_path()` accepts an optional `weight_property` that flips the search from BFS (hop count) to Dijkstra (sum of edge weights). Edges missing the property fall back to weight 1.0; negative weights cause the path to be reported as missing.

```python
# Cheapest path by edge.cost (a property on each edge)
result = graph.shortest_path(
    "Stop", "A", "Stop", "Z",
    weight_property="cost",
)
# {'path': [...], 'connections': [...], 'length': 3, 'weight': 4.7}

# Length-only variant returns float when weighted, int otherwise
graph.shortest_path_length("Stop", "A", "Stop", "Z", weight_property="cost")  # → 4.7
graph.shortest_path_length("Stop", "A", "Stop", "Z")                          # → 3
```

Same Louvain plumbing — `weight_property=None` falls back to BFS.

## `CALL { ... }` Subqueries

`CALL { ... }` nests a complete read sub-pipeline (`MATCH`/`WHERE`/`WITH`/`RETURN`, including nested `CALL { ... }`) and evaluates it as part of the outer query. It is the direct expression of post-aggregation enrichment shapes that otherwise require multiple `cypher()` calls or `WITH`-chaining workarounds that collapse the per-row cardinality you wanted to keep.

There are two forms, distinguished by whether the body imports outer variables.

### Uncorrelated — the body imports nothing

The subquery runs **exactly once**, independent of the outer row stream. Its result rows are **cartesian-producted** with the outer rows: an outer stream of *R* rows combined with a subquery returning *S* rows yields *R × S* rows.

```python
# Leading uncorrelated — no preceding clause, so R = 1 (one seed row):
# the result is simply the S rows the body returns.
graph.cypher("""
    CALL { MATCH (n:Person) RETURN count(n) AS total }
    RETURN total
""")

# Cartesian combine — each Company row is paired with the single
# subquery row, attaching the global person count to every company.
graph.cypher("""
    MATCH (c:Company)
    CALL { MATCH (n:Person) RETURN count(n) AS people }
    RETURN c.name AS company, people
""")
```

### Correlated — an importing `WITH` brings outer variables in

When the body's **first clause** is a `WITH` that lists outer variables, the subquery runs **once per outer row**, with those variables bound to that row's values. The subquery's result rows are joined back to *that* outer row.

```python
# The canonical per-row aggregate: count each person's friends without
# collapsing the person rows (a plain WITH ... count() would).
graph.cypher("""
    MATCH (p:Person)
    CALL {
        WITH p
        MATCH (p)-[:KNOWS]->(f)
        RETURN count(f) AS friend_count
    }
    RETURN p.name AS name, friend_count
""")

# Per-row top-K: keep each person's single oldest friend. ORDER BY and
# LIMIT inside the body apply independently per outer row.
graph.cypher("""
    MATCH (p:Person)
    CALL {
        WITH p
        MATCH (p)-[:KNOWS]->(f)
        RETURN f.name AS oldest ORDER BY f.age DESC LIMIT 1
    }
    RETURN p.name AS name, oldest
""")
```

### The importing `WITH` — bare variables only

The leading importing `WITH` may list **only plain variable references** — `WITH p`, `WITH p, c`. Projection, aliasing, aggregation, and a `WHERE` are all rejected in the importing position; re-project inside the body instead.

```python
# Rejected — aliasing in the importing WITH:
#   CALL { WITH p AS x  MATCH (x)-[:KNOWS]->(f) RETURN count(f) AS c }
# Rejected — projection / aggregation in the importing WITH:
#   CALL { WITH p.name AS n ... }
#   CALL { WITH p, count(*) ... }
# Correct — import the bare variable, re-project in the body:
graph.cypher("""
    MATCH (p:Person)
    CALL { WITH p RETURN p.name AS n }
    RETURN n
""")
```

Import is explicit and total: an outer variable is visible inside the body **iff** it appears in the importing `WITH`. A bare `MATCH (p)-[:KNOWS]->(f)` inside the body *without* `WITH p` treats `p` as a fresh, unbound pattern variable — not the outer `p`.

### Cardinality semantics

| Body shape | Per outer row | Outer row fate |
|---|---|---|
| Uncorrelated (any) | runs once, *S* rows | every outer row × *S* (cartesian) |
| Correlated, **non-aggregating** body returning *k* rows | runs per row | *k* output rows (inner join); **`k = 0` drops the outer row** |
| Correlated, **aggregating** body (`RETURN count(...)`, etc.) | runs per row | always exactly one row — `count` of an empty match is `0`, so the row **survives with the zero value** |

`CALL { ... }` is an **inner join**, not an optional one: a non-aggregating body that matches nothing for an outer row removes that row from the output. An aggregating body always returns one row, so those rows survive (e.g. `friend_count = 0`). A `NULL` import (e.g. an anchor that came from an upstream `OPTIONAL MATCH` miss) runs the body with the `NULL` binding — pattern matches against `NULL` produce no rows, so the same drop-vs-zero rule applies.

### Scoping

- **No outer leakage except via `RETURN`.** Variables introduced *inside* the body (`f` above) are not visible after the `CALL { ... }`. Only the columns named in the body's terminal `RETURN` escape, under their `RETURN` aliases.
- **Returned aliases must not collide with in-scope outer variables.** `CALL { ... RETURN p AS p }` when `p` is already bound outside is a compile error.
- **No auto-correlation.** Un-imported outer names are not silently visible inside the body (see the importing-`WITH` note above).

### v1 limitations

| Not supported (v1) | Why / workaround |
|---|---|
| Writes in the body (`CALL { ... CREATE/SET/DELETE ... }`) | Rejected at validation. Per-outer-row mutation + atomicity is deferred; do writes in a separate top-level clause. |
| Unit subquery (no terminal `RETURN`) | Deferred. The body must end in a `RETURN`. |
| `UNION` inside the body | Rejected at validation. Run separate queries and combine outside, or use a top-level `UNION`. |
| `CALL { ... } IN TRANSACTIONS` | Neo4j-server batching; no in-memory analogue. |
| `CALL (x) { ... }` scope-shorthand | Use the explicit `CALL { WITH x ... }` form. |

## Schema Introspection (`CALL db.*`)

Neo4j-compatible schema procedures for discovering what's in the graph
without leaving Cypher. Every Bolt client (cypher-shell, Neo4j Browser,
the Python `neo4j` driver) calls these to populate type palettes, drive
autocomplete, and surface index advisors.

| Procedure | YIELD columns | Returns |
|-----------|---------------|---------|
| `CALL db.labels()` | `label` | One row per node-type ("label") in the graph, sorted alphabetically |
| `CALL db.relationshipTypes()` | `relationshipType` | One row per connection-type ("relationship type") in the graph, sorted alphabetically |
| `CALL db.indexes()` | `name`, `type`, `entityType`, `labelsOrTypes`, `properties`, `state` | One row per index installed on the graph, sorted by `name` |
| `CALL db.propertyKeys()` | `propertyKey` | One row per declared property name (node + relationship), sorted alphabetically |
| `CALL db.schema()` | `nodeType`, `properties` | One row per node-type with its sorted list of property names — the in-language counterpart of Python `describe()` |

Procedure names are case-insensitive on dispatch (Neo4j convention
preserves camelCase in docs: `db.relationshipTypes`, not
`db.relationship_types`). YIELD columns are case-sensitive.

```python
# Enumerate node types
for row in graph.cypher("CALL db.labels() YIELD label RETURN label"):
    print(row["label"])

# Find relationship types matching a prefix
graph.cypher("""
    CALL db.relationshipTypes() YIELD relationshipType
    WHERE relationshipType STARTS WITH 'WORKS'
    RETURN relationshipType
""")

# Inspect indexes
for idx in graph.cypher("""
    CALL db.indexes() YIELD name, type, properties
    RETURN name, type, properties ORDER BY name
"""):
    print(f"{idx['name']:30}  type={idx['type']:9}  props={idx['properties']}")

# All property keys, and the per-type schema (no separate API needed)
graph.cypher("CALL db.propertyKeys() YIELD propertyKey RETURN propertyKey ORDER BY propertyKey")
for row in graph.cypher("CALL db.schema() YIELD nodeType, properties RETURN nodeType, properties"):
    print(f"{row['nodeType']}: {row['properties']}")
```

### `db.indexes()` column semantics

| Column | KGLite value |
|--------|--------------|
| `name` | `"<NodeType>.<property>"` (equality / range) or `"<NodeType>.(p1,p2,...)"` (composite) |
| `type` | `"PROPERTY"` for equality + composite indexes; `"RANGE"` for B-tree range indexes |
| `entityType` | Always `"NODE"` — relationship indexes are not yet supported |
| `labelsOrTypes` | `[node_type]` — single-element list |
| `properties` | `[property]` for equality/range; `[p1, p2, ...]` for composite |
| `state` | Always `"ONLINE"` — KGLite indexes are atomic, no `POPULATING` state |

**KGLite extension.** Neo4j collapses equality + range under a single
`type = "PROPERTY"`; KGLite distinguishes range indexes
(`type = "RANGE"`) because the planner uses the distinction — an
equality index can't serve a range query. Index advisors and tooling
that branch on `type` get the information they need without parsing
the `name` string.

### Cross-reference with the Python API

`db.indexes()` is the procedure form of the Python
`KnowledgeGraph.list_indexes()` method — both pull from the same
introspection helper, so output stays in sync. Use `db.indexes()`
from a Bolt client or inside a Cypher pipeline; use `list_indexes()`
from Python code where you'd rather have a Python list of dicts than
a `cypher()` result.

## Code-graph analysis

When the graph was built by `kglite.code_tree` (a parsed codebase), the
data needed for the analyses other tools ship as bespoke commands is
*already on the graph* — most are one query. The metrics are captured at
parse time (`branch_count`, `max_nesting`, `loc`) and the relationships
are first-class (`CALLS`, `REFERENCES_FN`, `USES_TYPE`, `EXTENDS`,
`IMPLEMENTS`).

### `CALL dead_code(...)` — unreferenced functions

```cypher
CALL dead_code() YIELD node
RETURN node.qualified_name AS fn, node.file_path AS file
ORDER BY file
```

Reports `Function` nodes with no inbound *use* edge — nothing `CALLS`
them, references them as a value (`REFERENCES_FN`), `HANDLES` them
(route), or `IMPLEMENTED_BY` (procedure), and no `DECORATES` participation.
Bundling all of those is the point: a naive `WHERE NOT (:Function)-[:CALLS]->(f)`
falsely flags callbacks, route handlers and decorated entry points.

Implicit entry points are excluded automatically: test functions, dunder
methods (`__x__`), and `main`. Options:

| Param | Default | Effect |
|---|---|---|
| `include_tests` | `false` | also report test functions |
| `exclude_public` | `false` | also drop `pub`/`public`/`export`/`exported` visibility (useful for Rust-style codebases; off by default because in Python every non-underscore name is nominally public) |

### Recipe queries (no procedure needed — the data is already there)

```cypher
-- Complexity hotspots (cyclomatic-style branch count is stored per fn)
MATCH (f:Function)
RETURN f.qualified_name, f.branch_count, f.max_nesting
ORDER BY f.branch_count DESC LIMIT 20

-- Blast radius: everything that (transitively) calls a target
MATCH (caller:Function)-[:CALLS*1..5]->(t:Function {name: 'parse'})
RETURN DISTINCT caller.qualified_name

-- God functions: large + high fan-in + high fan-out
MATCH (f:Function)
RETURN f.qualified_name, f.branch_count,
       size([(f)-[:CALLS]->() | 1]) AS fan_out,
       size([()-[:CALLS]->(f) | 1]) AS fan_in
ORDER BY fan_out + fan_in DESC LIMIT 20

-- Call-recursion cycles (strongly-connected components over CALLS)
CALL connected_components({node_type: 'Function', relationship: 'CALLS'})
YIELD node, component
RETURN component, collect(node.name) AS members
ORDER BY size(members) DESC
```

For test-impact analysis from a set of changed files, see
`CALL affected_tests({files: [...]})`.

### Edge confidence

Most edges are **extracted** — parsed facts (a `CALLS` edge is a real call
site). A few are **inferred** — best-effort heuristics, notably the
cross-language coupling edges (a client request matched to a server route by
path). Inferred edges carry `confidence = "inferred"`; extracted edges leave
the property unset. So:

```cypher
-- facts only (exclude heuristic edges)
MATCH (a)-[r:CALLS]->(b) WHERE r.confidence IS NULL RETURN a, b

-- just the heuristic cross-language couplings
MATCH (a)-[r]->(b) WHERE r.confidence = 'inferred' RETURN type(r), a.name, b.name
```

Inheritance-resolved `CALLS` edges stay **extracted** — they're pinned via
the type graph, not guessed, so they're facts, not heuristics.

## Scoping graph algorithms to a subgraph

The centrality (`pagerank`, `degree`, `betweenness`, `closeness`) and community
(`louvain`, `leiden`, `label_propagation`) procedures accept two optional
parameters that restrict the algorithm to a **property-filtered subgraph**,
so test / benchmark / external nodes don't pollute the result:

| Param | Meaning |
|-------|---------|
| `node_type` | string or list of node labels to include (e.g. `'Function'`) |
| `where` | a predicate over the node variable `n` — the same expression grammar as a `WHERE` clause |

```python
# PageRank over non-test, non-external functions only — the library's real hubs
graph.cypher("""
    CALL pagerank({node_type: 'Function', connection_types: 'CALLS',
                   where: 'n.is_test = false AND n.is_external = false'})
    YIELD node, score
    RETURN node.name, score ORDER BY score DESC LIMIT 15
""")

# Louvain over a subsystem, excluding benchmark code
graph.cypher("""
    CALL louvain({node_type: 'File', where: 'n.is_benchmark = false'})
    YIELD node, community
    RETURN community, count(*) AS size ORDER BY size DESC
""")
```

Only edges with **both** endpoints in scope are traversed, so scores and
communities reflect the subgraph, not the whole graph filtered afterward. An
explicit scope also lifts the large-graph refusal guard (you've bounded the
run yourself). Scoping is an **in-memory-only** feature; on disk/mapped graphs
the procedures reject `node_type` / `where` (filter with a preceding `MATCH`
instead).

## Dependency frontier — `CALL ready_set(...)`

Over a DAG on a chosen edge type, `ready_set` returns the nodes whose
dependencies are all satisfied — the "ready set" of a build / scheduling /
plan graph. A node's **dependencies** are its outgoing-edge neighbours, so
`(task)-[:DEPENDS_ON]->(dependency)` reads naturally: a task is ready once
every dependency it points to is *done*. "Done" is a predicate over the node
variable `n` (same grammar as `where`); a node already done is excluded, and a
root with no dependencies is ready as soon as it isn't done.

| Param | Meaning |
|-------|---------|
| `relationship` | the dependency edge type (string or list) |
| `done` | predicate over `n` marking a node satisfied, e.g. `'n.status = "done"'` |
| `node_type` | optional — limit which nodes are *emitted* (dependencies are followed regardless) |

`YIELD node, dependency_count` (how many dependencies the ready node had, all satisfied).

> **Scope to the type you care about.** A node with *no* outgoing-`E` edge is a
> root — vacuously "all dependencies satisfied" — so an unscoped `ready_set`
> over a sparse edge type also returns every unrelated node. To get e.g. "ready
> **tasks**", pass `node_type: 'Task'` so only that type is emitted (dependencies
> are still followed across types).

```python
# Which tasks can the agent pick up next?
graph.cypher("""
    CALL ready_set({relationship: 'DEPENDS_ON', done: 'n.status = "done"'})
    YIELD node, dependency_count
    RETURN node.id AS id, dependency_count AS deps ORDER BY id
""")
```

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

## Diagnostics

Every `cypher()` call attaches lightweight execution diagnostics to the returned `ResultView`. No prefix required, always on:

```python
result = graph.cypher("MATCH (n:Country {label: 'Norway'}) RETURN n.nid")
print(result.diagnostics)
# {'elapsed_ms': 3, 'timed_out': False, 'timeout_ms': 10000}
```

Keys:

- `elapsed_ms` — wall-clock duration in milliseconds.
- `timed_out` — `True` when the deadline fired (rows reflect the partial set).
- `timeout_ms` — the deadline that was in effect, or `None` when no deadline applied.

`timeout_ms` resolution: explicit `cypher(..., timeout_ms=N)` > `kg.set_default_timeout(ms)` > backend-aware default (Disk 10_000, Mapped 60_000, Memory none). Pass `timeout_ms=0` to disable the deadline entirely for one call.

## Indexes

Create an equality index on a `(node_type, property)` pair to accelerate `MATCH (n:T {prop: value})` and `WHERE n.prop = value` to O(log N):

```python
graph.create_index('Country', 'label')
# {'node_type': 'Country', 'property': 'label',
#  'unique_values': 5, 'persistent': true, 'created': true}
```

On a `storage='disk'` graph the index is **persistent** — written as four mmap'd files next to the CSR (`property_index_{type}_{property}_{meta,keys,offsets,ids}.bin`) and lazy-loaded on first query after reopen. No heap HashMap rebuild on `load()`. On in-memory graphs the existing `property_indices` HashMap is used (no change).

`describe()` annotates indexed properties so agents can see which columns hit the fast path before writing a query. String indexes support both equality and prefix; numeric indexes support equality only:

```xml
<prop name="label"  type="String" unique="5" indexed="eq,prefix" vals="Norway|..."/>
<prop name="year"   type="Int"    unique="20" indexed="eq"/>
```

### STARTS WITH pushdown

With a string index in place, `WHERE n.prop STARTS WITH 'x'` is pushed into the MATCH pattern and served by the prefix side of the sorted mmap:

```python
graph.cypher("MATCH (n:Country) WHERE n.label STARTS WITH 'O' RETURN n.nid")
# O(log N + k) where k is the number of matches
```

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

## Naming — identifiers, reserved words & structural accessors

### Reserved keywords as names (soft keywords)

Most reserved keywords can be used directly as a **relationship type**, **node
label**, or **property key** — the parser treats them as names in those
positions:

```cypher
CREATE (s:SourceDoc)-[:CONTAINS]->(c:Chunk)   // CONTAINS as a rel type
MATCH  (n:CONTAINS)                            // … as a label
CREATE (n:Doc {order: 1, in: true})            // … as property keys
RETURN n.contains, n.order                     // … and in property access
```

The soft set covers the operator / comparison / sort / set / mutation
keywords (`CONTAINS`, `IN`, `IS`, `NOT`, `STARTS`, `ENDS`, `ORDER`, `BY`,
`ASC`, `DESC`, `DISTINCT`, `ALL`, `MERGE`, `CREATE`, `DELETE`, `SET`,
`REMOVE`, `UNION`, …). A few keywords stay **reserved** to avoid ambiguity:
the clause-flow words (`MATCH`, `WHERE`, `RETURN`, `WITH`, `AND`, `OR`, …)
and the value keywords (`NULL`, `TRUE`, `FALSE`, `CASE`/`WHEN`/`END`,
`EXISTS`). For any reserved word, quote it with **backticks**:

```cypher
CREATE (n:Doc {`where`: 1, `null`: 'x'})
RETURN n.`where`
```

### Identifier charset & special characters (hyphens, dots, spaces)

A **bare** identifier (label, relationship type, property key, variable) must
match `[A-Za-z_][A-Za-z0-9_]*` — a letter or underscore followed by letters,
digits, or underscores. Anything outside that set — a hyphen, dot, space, or
leading digit — must be **backtick-quoted**. This matters most for
relationship types like `supports-claim` or `refines-idea`: written bare, the
`-` is parsed as the relationship-arrow token and you get a syntax error, so
backtick them:

```cypher
// Hyphenated / dotted / spaced rel types and labels — backtick them:
CREATE (a)-[:`supports-claim`]->(b)
MATCH  (a)-[r:`refines-idea`]->(b) RETURN a, b
MATCH  (n:`Legal Document`)       RETURN n
RETURN n.`dc.title`
```

The **string-typed APIs do not need escaping** — they take the type/label as a
plain string, so `add_connections(df, "supports-claim", …)`,
`add_nodes(df, "Legal Document", …)`, and `create_index("Doc", "dc.title")`
all accept arbitrary characters directly. Backticks are only a *Cypher-surface*
concern: escape when you name such a type/label/key inside a query, not when
you create it through the Python API.

### Structural accessors vs stored properties

Every node answers four convenience accessors:

| Accessor | Returns |
|----------|---------|
| `n.id` | the node's unique id (identity — always) |
| `n.title` | the node's title (identity — always) |
| `n.type` / `n.node_type` / `n.label` | the node's primary type string |
| `n.name` | the node's title |

`n.type` / `n.node_type` / `n.label` / `n.name` are **property-first**: if the
node stores a real property of that name, `n.<name>` returns the *stored
value*; the structural string is only the fallback when no such property
exists. So a `label` / `type` / `name` column loaded via `add_nodes` (or set
via `CREATE`) round-trips and reads back correctly. `id` and `title` are the
node's identity fields and always return the identity (no stored property can
shadow them). Use `labels(n)` for the label set and `id(n)` / `type(r)` for
the structural forms regardless of any same-named property.

### Identity (`id`) and prefixed-id datasets (`nid`)

`n.id` is the node's **unique identity** and behaves identically in every
storage mode (in-memory / mapped / disk). `CREATE (n {id: X})` and
`add_nodes(unique_id_field='id')` both make `X` the identity; `MATCH (n {id: X})`
finds it; it survives save → load. `id` is unique by convention — if duplicate
ids are created, `MATCH (n {id: X})` returns one node per id (a stderr warning
is emitted; use `MERGE` or dedupe the input). To audit a type for collisions
after the fact, `CALL duplicate_id({type: 'Artifact'}) YIELD node` yields every
node of that type whose id is shared (the identity-column sibling of
`duplicate_title`).

For datasets whose ids are a prefix + number (Wikidata `Q42`, `P31`, …), the
loader stores the **integer** as `id` (compact, identical across modes — disk
needs it at 100M-node scale) and the **string form** as the `nid` property.
Query by the string form via `{nid: 'Q42'}` (or by the integer via `{id: 42}`)
— `{id: 'Q42'}` does **not** match (ids are integers). `n.id → 42`,
`n.nid → 'Q42'`, in every mode.

## Supported Cypher Subset

| Category | Supported |
|----------|-----------|
| **Clauses** | `MATCH`, `OPTIONAL MATCH`, `WHERE`, `RETURN`, `WITH`, `ORDER BY`, `SKIP`, `LIMIT`, `UNWIND`, `UNION`/`UNION ALL`, `CALL { ... }` (read subqueries — uncorrelated + correlated), `CREATE`, `SET`, `DELETE`, `DETACH DELETE`, `REMOVE`, `MERGE`, `EXPLAIN`, `PROFILE` |
| **Patterns** | Node `(n:Type)`, relationship `-[:REL]->`, variable-length `*1..3`, undirected `-[:REL]-`, properties `{key: val, key: $param, key: var}`, `p = shortestPath(...)` |
| **WHERE** | `=`, `<>`, `<`, `>`, `<=`, `>=`, `=~` (regex), `AND`, `OR`, `NOT`, `IS NULL`, `IS NOT NULL`, `IN [...]`, `CONTAINS`, `STARTS WITH`, `ENDS WITH`, `EXISTS { pattern WHERE ... }`, `EXISTS(( pattern ))`, inline pattern predicates, `any/all/none/single(x IN list WHERE ...)` |
| **RETURN** | `n.prop`, `r.prop`, `AS` aliases, `DISTINCT`, arithmetic `+`/`-`/`*`/`/`, string concat `\|\|`, map projections `n {.prop}`, map literals `{k: expr}`, list slicing `[i..j]` |
| **Aggregation** | `count(*)`, `count(expr)`, `sum`, `avg`/`mean`, `min`, `max`, `collect`, `std` |
| **Expressions** | `CASE WHEN...THEN...ELSE...END`, `$param`, `[x IN list WHERE ... \| expr]`, `any/all/none/single(...)` |
| **Functions** | `toUpper`, `toLower`, `toString`, `toInteger`, `toFloat`, `size`, `length`, `type`, `id`, `labels`, `keys`, `coalesce`, `date`/`datetime`, `range`, `nodes(p)`, `relationships(p)`, `round` |
| **String** | `split`, `replace`, `substring`, `left`, `right`, `trim`, `ltrim`, `rtrim`, `reverse` |
| **Math** | `abs`, `ceil`/`ceiling`, `floor`, `round`, `sqrt`, `sign`, `log`/`ln`, `log10`, `exp`, `pow`, `pi`, `rand`, `randomUUID`, trig: `sin`/`cos`/`tan`/`asin`/`acos`/`atan`/`atan2`/`cot`/`haversin`/`degrees`/`radians` |
| **Spatial** | `point(lat, lon)`, `distance(a, b)`, `contains(a, b)`, `intersects(a, b)`, `centroid(n)`, `area(n)`, `perimeter(n)`, `latitude(point)`, `longitude(point)` |
| **Temporal** | `date(str)`/`datetime(str)`, `localdatetime()`/`localtime()`/`time()` (ISO strings), `date_diff(d1, d2)`, `date ± N` (days), `date - date` → int, `d.year`/`d.month`/`d.day`, `valid_at(...)`, `valid_during(...)` |
| **Semantic** | `text_score(n, prop, query [, metric])` — auto-embeds query via `set_embedder()`, cosine/dot_product/euclidean/poincare; `embedding_norm(n, prop)` — L2 norm (hierarchy depth) |
| **Timeseries** | `ts_sum`, `ts_avg`, `ts_min`, `ts_max`, `ts_count`, `ts_at`, `ts_first`, `ts_last`, `ts_delta`, `ts_series` — date-string args with resolution validation |
| **Mutations** | `CREATE (n:Label {props})`, `CREATE (a)-[:TYPE]->(b)`, `SET n.prop = expr`, `DELETE`, `DETACH DELETE`, `REMOVE n.prop`, `MERGE ... ON CREATE SET ... ON MATCH SET` |
| **Procedures** | `CALL pagerank/betweenness/degree/closeness() YIELD node, score`, `CALL louvain/leiden() YIELD node, community [, level]` (multilevel, hierarchical — `leiden` guarantees well-connected communities), `CALL label_propagation() YIELD node, community`, `CALL connected_components() YIELD node, component`, `CALL k_core/coreness() YIELD node, coreness`, `CALL clustering_coefficient() YIELD node, coefficient`, `CALL cluster({method, ...}) YIELD node, cluster`, `CALL affected_tests({files: [...], max_depth?}) YIELD test_file, depth` (0.9.34+, code-tree graphs), `CALL refresh_stats() YIELD src_type, edge_type, tgt_type, count` (0.9.35+, planner cardinality cache refresh), `CALL list_procedures()` |
| **Scoped algorithms** | `connected_components`, `k_core`/`coreness`, and `clustering_coefficient` accept an optional `{node_type, relationship}` map to run over a subgraph — e.g. `CALL k_core({node_type: 'Person', relationship: ['KNOWS', 'OWNS']})`. Each field is a string or list of strings; omit the map for the whole graph. Computed lazily over the live graph (identical across memory/mapped/disk modes). |
| **Schema** | `CALL db.labels() YIELD name`, `CALL db.relationshipTypes() YIELD name`, `CALL db.indexes() YIELD name, type, entityType, labelsOrTypes, properties, state` (0.10.0+, Bolt-compatible) |
| **Rule procedures** | `CALL orphan_node/self_loop/missing_required_edge/missing_inbound_edge/duplicate_title/duplicate_id/null_property({type[,edge\|property]}) YIELD node`, `CALL cycle_2step({type, edge}) YIELD node_a, node_b`, `CALL inverse_violation({rel_a, rel_b}) YIELD a, b`, `CALL transitivity_violation({rel}) YIELD a, b, c`, `CALL cardinality_violation({type, edge[, min, max]}) YIELD node, count`, `CALL type_domain_violation/type_range_violation({edge, expected_*}) YIELD source, target`, `CALL parallel_edges({edge}) YIELD a, b, count` |
| **Operators** | `+`, `-`, `*`, `/`, `\|\|` (string concat), `=~` (regex), `IN`, `STARTS WITH`, `ENDS WITH`, `CONTAINS`, `IS NULL`, `IS NOT NULL` |

## openCypher Compatibility Matrix

Clause-by-clause comparison with the openCypher specification.

### Clauses

| Clause | Status | Notes |
|--------|--------|-------|
| `MATCH` | Full | Node patterns, relationship patterns, variable-length paths, `shortestPath` |
| `OPTIONAL MATCH` | Full | Automatic fusion optimization with aggregation |
| `WHERE` | Full | All comparison, logical, string, and pattern operators |
| `RETURN` | Full | Aliases, `DISTINCT`, expressions, map projections, `HAVING` |
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
| `HAVING` | Full | Post-aggregation filter on `RETURN`/`WITH` |
| `CALL ... YIELD` | Full | Built-in graph algorithm procedures |
| `CALL { ... }` subqueries | Partial | Uncorrelated + correlated (importing `WITH`) read subqueries. v1 excludes writes in the body, unit (no-`RETURN`) subqueries, `UNION` inside the body, and `IN TRANSACTIONS`. See [`CALL { ... }` Subqueries](#call----subqueries) |
| `FOREACH` | Not supported | Use `UNWIND` + `CREATE`/`SET` instead |
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
| Window functions (`OVER`) | Full | `row_number()`, `rank()`, `dense_rank()` with `PARTITION BY`/`ORDER BY` |

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
| `labels(n)` | Full | Returns the label list, primary type first (multi-label since 0.10.5) |
| `keys(n)` / `keys(r)` | Full | Returns property names as JSON list |
| `date(str)` / `datetime(str)` | Full | Parse date string to DateTime; `d.year`, `d.month`, `d.day` accessors; `date ± N`, `date - date`, `date_diff()` |
| `coalesce` | Full | |
| `range(start, end [, step])` | Full | Inclusive integer range |
| `round(x [, precision])` | Full | |
| `nodes(p)`, `relationships(p)` | Full | Path decomposition |
| String functions | Full | `split`, `replace`, `substring`, `left`, `right`, `trim`, `ltrim`, `rtrim`, `reverse` — auto-coerce non-strings |
| Math functions | Full | `abs`, `ceil`, `floor`, `sqrt`, `sign`, `log`/`ln`, `log10`, `exp`, `pow`, `pi`, `rand`, `randomUUID` |
| Trig functions | Full | `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `atan2(y,x)`, `cot`, `haversin`, `degrees`, `radians` — radians; NULL/non-numeric → NULL |
| Spatial functions | Full | `point`, `distance`, `contains`, `intersects`, `centroid`, `area`, `perimeter` |
| Temporal functions | Full | `valid_at`, `valid_during` — NULL = open-ended; `localdatetime`/`localtime`/`time` → ISO strings |

### Architectural Differences from Neo4j

| Feature | KGLite | Neo4j | Rationale |
|---------|--------|-------|-----------|
| Labels per node | One primary type + secondary labels | Multiple equal labels | Primary type drives indexing (`type_indices`); secondary labels are additive (0.10.5+) |
| `labels(n)` return type | `List[String]` (primary first) | `List[String]` | Matches Neo4j since 0.10.5 |
| `SET n:Label` | Supported (adds a secondary label) | Supported | Primary type is immutable via label ops — change it with `SET n.type = 'NewType'` |
| Storage | In-memory (petgraph) | Disk-based | Embedded use case, explicit `save()`/`load()` |
| Transactions | Snapshot isolation + OCC | Full ACID | GIL serializes Python access; OCC catches conflicts |
| Indexing | Type indices + vector index | Schema indexes | Automatic type-based lookup, no manual `CREATE INDEX` |
| `LOAD CSV` | Not supported | Supported | Python ecosystem (pandas) preferred for data loading |
