# Cypher Reference

Full Cypher subset supported by KGLite. For a quick overview, see the [Cypher guide](https://kglite.readthedocs.io/en/latest/guides/cypher.html).

> **Label model:** Each node has one immutable **primary** type plus optional secondary labels (multi-label since 0.10.5). `CREATE (n:A:B)`, `SET n:B`, `REMOVE n:B`, and `MATCH (n:A:B)` all work; `labels(n)` returns a list with the primary type first. Change the primary type via `SET n.type = 'NewType'` (`REMOVE n:Primary` errors deliberately).

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

## Schema Introspection (`CALL db.*`)

Neo4j-compatible schema procedures for discovering what's in the graph
without leaving Cypher. Every Bolt client (cypher-shell, Neo4j Browser,
the Python `neo4j` driver) calls these to populate type palettes, drive
autocomplete, and surface index advisors.

| Procedure | YIELD columns | Returns |
|-----------|---------------|---------|
| `CALL db.labels()` | `name` | One row per node-type ("label") in the graph, sorted alphabetically |
| `CALL db.relationshipTypes()` | `name` | One row per connection-type ("relationship type") in the graph, sorted alphabetically |
| `CALL db.indexes()` | `name`, `type`, `entityType`, `labelsOrTypes`, `properties`, `state` | One row per index installed on the graph, sorted by `name` |

Procedure names are case-insensitive on dispatch (Neo4j convention
preserves camelCase in docs: `db.relationshipTypes`, not
`db.relationship_types`). YIELD columns are case-sensitive.

```python
# Enumerate node types
for row in graph.cypher("CALL db.labels() YIELD name RETURN name"):
    print(row["name"])

# Find relationship types matching a prefix
graph.cypher("""
    CALL db.relationshipTypes() YIELD name
    WHERE name STARTS WITH 'WORKS'
    RETURN name
""")

# Inspect indexes
for idx in graph.cypher("""
    CALL db.indexes() YIELD name, type, properties
    RETURN name, type, properties ORDER BY name
"""):
    print(f"{idx['name']:30}  type={idx['type']:9}  props={idx['properties']}")
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
is emitted; use `MERGE` or dedupe the input).

For datasets whose ids are a prefix + number (Wikidata `Q42`, `P31`, …), the
loader stores the **integer** as `id` (compact, identical across modes — disk
needs it at 100M-node scale) and the **string form** as the `nid` property.
Query by the string form via `{nid: 'Q42'}` (or by the integer via `{id: 42}`)
— `{id: 'Q42'}` does **not** match (ids are integers). `n.id → 42`,
`n.nid → 'Q42'`, in every mode.

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
| **Math** | `abs`, `ceil`/`ceiling`, `floor`, `round`, `sqrt`, `sign`, `log`/`ln`, `log10`, `exp`, `pow`, `pi`, `rand`, `randomUUID`, trig: `sin`/`cos`/`tan`/`asin`/`acos`/`atan`/`atan2`/`cot`/`haversin`/`degrees`/`radians` |
| **Spatial** | `point(lat, lon)`, `distance(a, b)`, `contains(a, b)`, `intersects(a, b)`, `centroid(n)`, `area(n)`, `perimeter(n)`, `latitude(point)`, `longitude(point)` |
| **Temporal** | `date(str)`/`datetime(str)`, `localdatetime()`/`localtime()`/`time()` (ISO strings), `date_diff(d1, d2)`, `date ± N` (days), `date - date` → int, `d.year`/`d.month`/`d.day`, `valid_at(...)`, `valid_during(...)` |
| **Semantic** | `text_score(n, prop, query [, metric])` — auto-embeds query via `set_embedder()`, cosine/dot_product/euclidean/poincare; `embedding_norm(n, prop)` — L2 norm (hierarchy depth) |
| **Timeseries** | `ts_sum`, `ts_avg`, `ts_min`, `ts_max`, `ts_count`, `ts_at`, `ts_first`, `ts_last`, `ts_delta`, `ts_series` — date-string args with resolution validation |
| **Mutations** | `CREATE (n:Label {props})`, `CREATE (a)-[:TYPE]->(b)`, `SET n.prop = expr`, `DELETE`, `DETACH DELETE`, `REMOVE n.prop`, `MERGE ... ON CREATE SET ... ON MATCH SET` |
| **Procedures** | `CALL pagerank/betweenness/degree/closeness() YIELD node, score`, `CALL louvain/label_propagation() YIELD node, community`, `CALL connected_components() YIELD node, component`, `CALL cluster({method, ...}) YIELD node, cluster`, `CALL affected_tests({files: [...], max_depth?}) YIELD test_file, depth` (0.9.34+, code-tree graphs), `CALL refresh_stats() YIELD src_type, edge_type, tgt_type, count` (0.9.35+, planner cardinality cache refresh), `CALL list_procedures()` |
| **Schema** | `CALL db.labels() YIELD name`, `CALL db.relationshipTypes() YIELD name`, `CALL db.indexes() YIELD name, type, entityType, labelsOrTypes, properties, state` (0.10.0+, Bolt-compatible) |
| **Rule procedures** | `CALL orphan_node/self_loop/missing_required_edge/missing_inbound_edge/duplicate_title/null_property({type[,edge\|property]}) YIELD node`, `CALL cycle_2step({type, edge}) YIELD node_a, node_b`, `CALL inverse_violation({rel_a, rel_b}) YIELD a, b`, `CALL transitivity_violation({rel}) YIELD a, b, c`, `CALL cardinality_violation({type, edge[, min, max]}) YIELD node, count`, `CALL type_domain_violation/type_range_violation({edge, expected_*}) YIELD source, target`, `CALL parallel_edges({edge}) YIELD a, b, count` |
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
