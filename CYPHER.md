# Cypher Reference

KGLite's independently implemented Cypher dialect runs **in-process** (no
server). It provides a tested openCypher-compatible subset plus explicitly
classified KGLite and GQL-style extensions. It is not a complete openCypher or
Neo4j implementation. For a quick overview, see the [Cypher guide](https://kglite.readthedocs.io/en/latest/python/guides/cypher.html).

**Start with:** [Getting Started](https://kglite.readthedocs.io/en/latest/python/getting-started.html) ·
[Cypher guide](https://kglite.readthedocs.io/en/latest/python/guides/cypher.html) ·
[dialect contract](#cypher-dialect-contract) · [Python API](https://kglite.readthedocs.io/en/latest/autoapi/kglite/index.html) ·
[Fluent API](https://kglite.readthedocs.io/en/latest/reference/fluent-api.html) ·
[0.13 → 0.14 migration](https://kglite.readthedocs.io/en/latest/python/migrations/0.13-to-0.14.html)

> **Label model:** Each node has one immutable **primary** type plus optional
> secondary labels. All of these forms work:
>
> - `CREATE (n:A:B)`, `SET n:B` and `REMOVE n:B`.
> - `MATCH (n:A:B)` matches nodes with both labels (AND).
> - `MATCH (n:A|B)` matches nodes with either label (OR). It is valid in
>   reading patterns only. Mixing `|` with a `:` chain is a parse error, and
>   `|` is refused in CREATE/MERGE/SET/REMOVE.
>
> `labels(n)` returns the primary type first.
>
> `SET n.type = 'NewType'` only writes a property. It does not retype the node.
> To change the primary type, migrate or recreate the node under the new schema.

## Feature coverage

This table lists the surface at a glance. Most of what you would reach for in
an embedded, Cypher-queryable graph runs in-process:

| Area | Supported |
|---|---|
| **Reading** | `MATCH`, `OPTIONAL MATCH`, `WHERE` / `FILTER`, `RETURN` / `FINISH`, `WITH`, `ORDER BY` / `SKIP` / `OFFSET` / `LIMIT`, `UNWIND`, `UNION` |
| **Writing** | `CREATE`, strict `INSERT`, `MERGE` (+ `ON CREATE` / `ON MATCH SET`), `SET`, `DELETE` / `NODETACH DELETE` / `DETACH DELETE`, `REMOVE`, `FOREACH (x IN list \| …)` |
| **Subqueries** | Per-row `CALL { … }`, `CALL (x, y) { … }`, `CALL (*) { … }`, `CALL () { … }`, `EXISTS { … }`, `COUNT { … }` |
| **Schema DDL** | `CREATE [RANGE] INDEX [name] [IF NOT EXISTS] FOR (n:L) ON (n.p, …)`, `DROP INDEX … [IF EXISTS]`, `SHOW INDEXES` — see [Cypher index DDL](#cypher-index-ddl) for the taxonomy mapping |
| **Constraint DDL** | `CREATE CONSTRAINT [name] [IF NOT EXISTS] FOR (n:L) REQUIRE n.p IS UNIQUE \| IS NOT NULL \| IS NODE KEY \| IS :: TYPE`, `FOR ()-[r:T]-() REQUIRE r.p IS NOT NULL \| IS :: TYPE`, `DROP CONSTRAINT … [IF EXISTS]`, `SHOW CONSTRAINTS` — enforced on every write path, see [Cypher constraint DDL](#cypher-constraint-ddl) |
| **Path finding** | variable-length `-[*1..n]->`, `shortestPath(…)`, `allShortestPaths(…)`, weighted shortest path (`CALL`) |
| **Predicates** | `=, <>, <, >, <=, >=`, `AND` / `OR` / `NOT`, `IS [NOT] NULL`, `IN`, `CONTAINS` / `STARTS WITH` / `ENDS WITH`, regex `=~` |
| **Expressions** | list comprehension `[x IN xs WHERE … \| …]`, `reduce(…)`, `CASE`, list/map literals, parameters `$p` |
| **Parameters** | values `$p`, and **names**: dynamic labels / relationship types `(n:$label)`, `(n:$(label))`, `-[:$type]->` — see [Parameters](#parameters) |
| **Aggregation** | `count` / `sum` / `avg` / `min` / `max` / `collect` / `percentile_cont` / `mode` / `stdev` …, `DISTINCT`, `HAVING`, window functions (`OVER`, `PARTITION BY`, ranking) |
| **Procedures** (`CALL`) | centralities (pagerank, betweenness, closeness, degree), community (louvain, leiden, label propagation), components, k-core, clustering, `triangle_count` / `transitivity`, `eccentricity` / `diameter`, `ready_set` (dependency frontier), `shortest_path_length`, `kg_knn`, structural validators (`duplicate_title`, `cycle_2step`, `parallel_edges`, …) |
| **Vector + text** | `vector_score(…)` (HNSW index, exact fallback), `text_score(…)` (query vector, or query text via a pluggable embedder), `text_bm25(…)` (lexical BM25 over `build_text_index`, or `db.relationship_text_index.build` for relationships), `score_fuse(…)` (combine the lanes into one score) — hybrid semantic, lexical and structural in one query |
| **Spatial** | `point(…)`, `distance(…)`, `wkt_within` / `intersects`, buffer / hull / union, k-NN — see [Spatial](#spatial-functions) |
| **Temporal** | `date()` / `datetime()` / `localdatetime()`, `duration(…)`, `duration.between`, date arithmetic, `valid_at` / `valid_during` — see [Temporal](#temporal-functions) |
| **Value types** | int, float, string, bool, **date**, **timestamp** (date + time), duration, point, list, map, node, relationship, path |
| **Transactions** | multi-statement with snapshot isolation + rollback (`Session` / `Transaction`) |
| **Change data capture** | opt-in change stream with stateless cursors — `CALL db.cdc.enable/status/current/earliest/query/disable`, see [Change data capture](#change-data-capture-call-dbcdc) |
| **Storage** | identical Cypher across in-memory, mmap, and on-disk modes (largest graph run: Wikidata, 124M nodes / 861M edges) |

`UNIQUE`, `NOT NULL`, NODE KEY and `IS :: TYPE` constraints **are** supported
per write and enforced on every write path, including the bulk loader. Declare
them with [constraint DDL](#cypher-constraint-ddl) or `define_schema`.
Relationships carry `NOT NULL` and `IS :: TYPE` too. A graph that declares no
constraint pays one `HashMap::is_empty` check per write, so the in-memory write
path is untouched by the feature existing.

### Node identity — use `id` as your primary key

`id` is the **indexed** identity property. It accepts **strings as well as
integers**:

```python
graph.cypher("CREATE (:Memory {id: 'a3f9-uuid', text: 'hello'})")
graph.cypher("MATCH (n:Memory {id: 'a3f9-uuid'}) RETURN n.text")   # indexed
graph.cypher("MATCH (n:Memory) WHERE n.id IN $keys RETURN n", params={"keys": [...]})  # multi-probe
```

Put your application key in `id`, **not** in a custom property. An anchored
lookup on `id` uses the storage mode's identity index. An arbitrary property
(`mid`, `key`, ...) is **not indexed**, so `MATCH (n {mid: $k})` is a full label
scan (linear in node count). Keep these semantics in mind:

- **Uniqueness is opt-in.** With no constraint declared, `CREATE` does not
  reject a duplicate `id`: two `CREATE (:T {id: 'k'})` make two nodes.
  - A duplicate-id warning joins the statement's warnings (`result.warnings`,
    echoed to stderr like any query warning). It appears at the write when the
    type's id index is built, as after `add_nodes`, and otherwise at the next id
    lookup.
  - A bulk loader has no statement to carry the warning, so its duplicates are
    reported on stderr.
  - To reject duplicates, declare the node type's primary key
    (`define_schema({'nodes': {'T': {'primary_key': 'id'}}})`). The second
    `CREATE` is then rejected.
  - Alternatively use **`MERGE` on the id alone, not `CREATE`**. `MERGE (:T {id: $k})`
    is idempotent either way.
  - A `MERGE` that also names a second label (`MERGE (:T:U {id: $k})`) or another
    property matches only a node carrying them. When the `T` node with that id
    lacks one, the `MERGE` is refused rather than creating a second node under
    the id, and the error names what differs: "MERGE would create a second :T
    node with id 'k'; the existing node lacks label :U — match on the id and add
    the label (`MERGE (n:T {id: 'k'}) SET n:U`), or use ON MATCH SET".
  - A `MERGE` naming no id, or an id the type does not hold, creates as before.
  - Constraint DDL is deliberately *not* the route here. `REQUIRE t.id IS UNIQUE`
    is refused, because `id` is a structural field rather than a stored property
    and the unique secondary index would never see the write. See
    [Cypher constraint DDL](#cypher-constraint-ddl).
- **Matching is type-exact.** `'42'` ≠ `42`. Keep id types consistent across
  writes and reads.
- **Property typos are caught where the type's shape is known.** A `CREATE` or
  `MERGE` node pattern naming a property no write to the type has named is
  rejected with a `Valid properties: …` / "did you mean?" hint. This is a
  deliberate typo-guard, active on an open schema.
  - It applies only where KGLite knows the shape. A type with no recorded
    properties yet, an unlabelled pattern, and relationship properties are all
    skipped. `SET n.newprop = …` stores a new property by design, which is how a
    type grows a column.
  - A property a write named counts as known even when every value written so
    far was NULL. Examples are an `add_nodes` column that was empty throughout,
    and a `CREATE` / `UNWIND … CREATE` map entry that evaluated to NULL. A later
    delivery carrying a value is accepted, open or locked.
  - Where nothing names its type yet (a Cypher NULL, an all-`None` object
    column), `schema()` records it as `Unknown` and `describe()` lists it with
    `coverage="0%"` until the first value types it. An `add_nodes` column with a
    dtype keeps that type.
  - A property the graph declares counts as known before any node carries it.
    That covers a `define_schema()` field, and the `from` / `to` bound of a
    [validity declaration](#validity-interval-declarations) on the label (on a
    secondary label, for the nodes that carry it).
  - Once any node of a type carries a property, it is known on the whole type,
    whichever label declared it.
  - `lock_schema()` extends the refusal to *reads* of a property no node of the
    type has (`WHERE`, `RETURN`, `WITH`, `ORDER BY`) and to unknown labels. See
    [Diagnostics](#diagnostics).
  - The bulk loaders (`add_nodes`/`add_connections`) deliberately bypass the
    lock. It gates the Cypher write path.

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

**Every result column needs a distinct name.** A `RETURN` or `WITH` that names
one column twice is rejected with *"Multiple result columns with the same name
are not supported"*, matching Neo4j. The rejected forms are:

- `RETURN 1 AS x, 2 AS x`
- `RETURN n.a AS x, n.b AS x`
- an unaliased `RETURN n.a, n.a`

A row is one name-keyed map, so two items sharing a name were never two
columns. The collision used to be silent and lost **both** values. Rename one
with `AS`. Names are case-sensitive, so `AS x` and `AS X` are two columns.

### Current clause spellings

KGLite accepts the Cypher 25 spellings `FILTER`, `OFFSET`, `NODETACH DELETE`,
`FINISH`, and `INSERT`:

```python
# FILTER is a standalone row filter; OFFSET is a SKIP synonym.
graph.cypher("MATCH (p:Person) FILTER p.active RETURN p.name OFFSET 10 LIMIT 5")

# FINISH ends a read or write pipeline and deliberately returns no rows.
graph.cypher("MATCH (p:Person) SET p.seen = true FINISH")

# NODETACH DELETE is explicit plain DELETE: connected nodes are still refused.
graph.cypher("MATCH (p:Person {id: $id}) NODETACH DELETE p", params={"id": 7})
```

- `FILTER predicate` has the row behavior of `WITH * WHERE predicate` without
  changing the current columns.
- `FINISH` must be terminal. Completed mutations remain visible, and its
  `ResultView` is empty.

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

# Regex matching with =~ — the pattern must match the WHOLE value
graph.cypher("MATCH (n:Person) WHERE n.name =~ '(?i)ali.*' RETURN n.name")
graph.cypher("MATCH (n:Person) WHERE n.email =~ '.*@example\\.com' RETURN n.name")

# `=~` is not a substring search: 'inactive' =~ 'active' is false.
# To search, say so with `.*` — or use CONTAINS / text_match_regex().
graph.cypher("MATCH (n:Person) WHERE n.name =~ '.*ali.*' RETURN n.name")
```

An inline property map is an equality filter written in the pattern. Its values
take the same expressions as a `CREATE` map: literals, `$params`, variables, and
computed values such as `row[0]`, `row['k']`, `toLower(name)` or `x + 1`. A
value that evaluates to `null` matches nothing. A value that fails to evaluate
is an error:

```python
graph.cypher("UNWIND $rows AS row MATCH (d:Doc {id: row[0]}) SET d.lines = row[1]",
             params={"rows": [["a", 10], ["b", 20]]})
```

### Generated filters: prefer `IN [...]` over long `OR` chains

Expression nesting is capped at **512 levels**, and every `OR` term adds one
level. A filter builder that emits one term per selected value
(`n.sku = 'a' OR n.sku = 'b' OR ...`) therefore stops parsing at ~512
selections:

```
Expression nesting exceeds 512 levels; simplify the query. ...
```

A list membership test is one level no matter how long the list is. The same
filter expressed with `IN` has no practical ceiling. It is also faster, because
it can be pushed into the MATCH and use an index:

```python
# Fragile above ~512 values, and slower below it
graph.cypher("MATCH (n:Product) WHERE " + " OR ".join(f"n.sku = '{s}'" for s in skus) + " RETURN n")

# Prefer this — one AST level for any number of values, and parameterised
graph.cypher("MATCH (n:Product) WHERE n.sku IN $skus RETURN n", params={"skus": skus})
```

The planner already rewrites `OR` chains of equalities on a *single* property
into `IN` for you. That happens only after the query parses, so it cannot rescue
a chain that is already too long. It does not fire for chains spanning different
properties. Generating `IN` directly is the robust habit.

### A stored `'["Oslo"]'` equals `'Oslo'`

A string property whose value is a **single-element JSON string list** compares
equal to that element's plain string. This holds in both directions and on
every route that decides equality:

- `=`, `<>` and `IN [...]`.
- Inline pattern properties (`MATCH (n {tag: 'Oslo'})`).
- The `WHERE`-clause scan predicates.
- The index-backed pushdown.

So `n.tag = 'Oslo'`, `n.tag = '["Oslo"]'` and `n.tag IN ['Oslo']` all match a
row storing either spelling, and `=`/`<>` always partition the rows between
them. The rule is one-element-only: `'["Oslo","Bergen"]'` is an ordinary string
and matches only itself. It applies to strings, never to real list values.

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

A stored property always wins, as it does on a node. A relationship with no
property of that name falls back to these values:

| Expression | Value |
|---|---|
| `r.type` (and `r.connection_type`, a synonym; so is the `<conn>` element in `describe()`) | The relationship type |
| `r.id` | `id(r)` |
| `r.start` / `r.start_id` | `id(startNode(r))` |
| `` r.`end` `` / `r.end_id` | `id(endNode(r))` (`end` is a reserved word, so it is backticked) |

A relationship that stores `type` or `id`, as graphs imported from LLM
extractors often do, reads those values back in every clause. The rule is the
same for a MATCH variable and for a relationship value from `collect`,
`UNWIND`, `relationships(p)` or `YIELD relationship`.

`type(r)`, `id(r)`, `startNode(r)` and `endNode(r)` always read the relationship
itself. A relationship pattern map (`-[r {type: 'x'}]->`) and `MERGE` match
stored properties only, so `{type: 'R'}` finds no relationship that merely *is*
an `R`. Write `WHERE r.type = 'R'` to include the fallback.

`SET` / `REMOVE` work on a relationship variable, so you can upsert edge
properties, including via `MERGE`:

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
| `sum(expr)` | Numeric sum; all-integer inputs produce an exact Int64 result or an overflow error if the final result is out of range |
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

### Result column names

An item with `AS` is named by its alias. An unaliased item of the final
`RETURN` is named by its source text as written, trimmed: `RETURN toInteger('3'), 1+2`
has the columns `toInteger('3')` and `1+2`. Casing, quotes and spacing are kept;
`//` comments are dropped. `WITH` items are referenced by their alias or
property form, not by source text.

### ORDER BY after an aggregate

An aggregating `RETURN` emits one row per group, so a sort key must have a
single value on that row. Two things do:

- **Projected columns**: an alias, or an unaliased item's expression form:
  `RETURN c.name AS company, count(p) ORDER BY count(p) DESC, company`.
- **Anything read by a grouping key**: every non-aggregate item is a grouping
  key, so any property of a variable it names is sortable even when it is not
  itself projected:
```python
# t is named by the grouping key t.title, so t.priority is sortable.
graph.cypher("""
    MATCH (t:Task)
    OPTIONAL MATCH (t)-[:X]->(c)
    RETURN t.title AS title, count(c) AS n
    ORDER BY t.priority DESC
""")
```

The value used is the group's first row. When several distinct nodes collapse
into one group (two `Task`s sharing a title), that representative is the one the
group met first. Project the sort key explicitly if you need it pinned.

Ordering by anything else is rejected rather than silently ignored:

```python
# ERROR: c collapses into count(c), so c.label has no value per group.
"... RETURN t.title AS title, count(c) AS n ORDER BY c.label DESC"

# ERROR: aggregate not projected. Project it, then order by the alias.
"... RETURN t.title AS title, count(*) AS n ORDER BY max(t.priority) DESC"

# ERROR: the aggregate is projected as `n` — order by `n`.
"... RETURN t.title AS title, count(*) AS n ORDER BY count(*) DESC"
```

A `RETURN` **without** aggregates keeps every binding on its rows, so ordering
by a non-projected expression is unrestricted there. `WITH` narrows scope to
what it projects, so a sort key after `WITH` must be one of its columns.

## Sort order

`ORDER BY` accepts `ASC` (default) or `DESC` per key, plus an explicit
`NULLS FIRST` / `NULLS LAST`. Without one, NULLs go **last ascending and first
descending**, the Neo4j 5 default.

A sort key can hold values of more than one type. Examples are a `CASE`
returning a number on some rows and a string on others, `coalesce` over
differently-typed properties, and a property read across two node types. Those
values are ordered **by type first**, following Neo4j 5's ranking. Ascending:

| | rank |
|---|---|
| lowest | map |
| | node |
| | relationship |
| | list |
| | path |
| | date / datetime |
| | duration |
| | point |
| | string |
| | boolean |
| highest | number |

NULL sorts after all of them ascending, subject to the `NULLS` placement above.
Within a rank:

- **Numbers** compare numerically across integers and floats, never "all
  integers, then all floats". They compare exactly, including integers past 2⁵³
  that no float can represent. `NaN` sorts above every other number.
- **Dates and datetimes** share one rank and compare chronologically, a date
  counting as midnight on that date.
  - The same rule governs `=`, `<>` and `IN`, so
    `datetime('2024-03-15T00:00:00') = date('2024-03-15')` is true. This is a
    deliberate divergence from openCypher, which makes every comparison between
    a date and a datetime null.
  - `DISTINCT` and grouping keys remain structural, so the two stay separate
    keys there.
- **A date or datetime against a string** compares by parsing the string. This
  holds for `=`, `<>` and `IN` exactly as for `<` and `>`.
  - A date parses from `YYYY-MM-DD` (and `YYYY/MM/DD`, `DD-MM-YYYY`,
    `MM/DD/YYYY`). A datetime also parses from `YYYY-MM-DDTHH:MM:SS`.
  - `n.valid_to = '1990-01-01'` finds the date a query returned as text.
  - A string that does not parse is another type family: `=` is false, `<>` is
    true and `<` is null, as openCypher answers every cross-type pair.
  - Comparing a parseable string is a divergence from openCypher, where `=`
    across the two types is false.
- **Lists** rank element by element, then by length. Ascending: `[1]`,
  `[1,1,9]`, `[1,2]`, `[2]`. This is the *sort* rank. The `<` operator has no
  rule for two lists (see below).
- **Maps** rank entry by entry in key order, then by size, with the same
  sort-only caveat.

```python
graph.cypher("UNWIND [3, 'b', 1, 'a'] AS v RETURN v ORDER BY v")
# → 'a', 'b', 1, 3      (strings rank below numbers)
```

`min()` and `max()` use this same order: `min` is the first value ascending and
`max` the last. They never depend on which row arrived first, and NULLs are
excluded from both. **This is one deliberate deviation from Neo4j**, which
applies a *different*, aggregate-specific rule to `min`/`max` on mixed input
(there, numbers rank below strings, which rank below lists). KGLite uses one
order everywhere, so `min(x)` always equals `x ORDER BY x ASC LIMIT 1`.

The same total order governs the fluent API's `sort=` fields. A node missing the
sort property is ordered as NULL.

**Sorting is total; comparing is not.** This ranking exists so `ORDER BY`, `min`
and `max` can place every pair. The `<`, `<=`, `>` and `>=` *operators* answer
`null` when no ordering rule relates the two values' types, which is
openCypher's rule.

- `'a' < 1`, `true < 1`, `[1] < 2`, `{a: 1} < 1` and a list against another list
  are all `null`, in both directions, even though `ORDER BY` places those same
  values confidently.
- A `null` keeps no row in either direction: `WHERE n.v < 5` and
  `WHERE NOT (n.v < 5)` both drop a string-valued `n.v`.
- Cross-type `=` is still `false` and `<>` still `true`.
- `NaN` is the one number that declines every ordering comparison without being
  cross-type. It answers `false` (never `null`) for all four operators,
  including against itself, while still sorting above every other number.

## HAVING

Post-aggregation filter. Use it after RETURN or WITH with aggregates:

```python
graph.cypher("MATCH (n:Person) RETURN n.city, count(*) AS pop HAVING pop > 1000")
```

`HAVING` is also supported on WITH:

```python
graph.cypher("""
    MATCH (n:Person)
    WITH n.city AS city, count(*) AS pop HAVING pop > 100
    RETURN city, pop
""")
```

## Window Functions

Window functions compute values across partitions of the result set without
collapsing rows.

| Function | Description |
|---|---|
| `row_number() OVER (...)` | Sequential number within partition |
| `rank() OVER (...)` | Rank with gaps for ties |
| `dense_rank() OVER (...)` | Rank without gaps for ties |

The OVER clause is `OVER (PARTITION BY expr [, ...] ORDER BY expr [ASC|DESC] [, ...])`.

- `PARTITION BY` is optional. Without it, the whole result set is one partition.
- `ORDER BY` is required.

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

`OPTIONAL MATCH` is a left outer join: it keeps rows even when nothing matches.

```python
graph.cypher("""
    MATCH (p:Person)
    OPTIONAL MATCH (p)-[:KNOWS]->(f:Person)
    RETURN p.name, count(f) AS friends
""")
```

### `WHERE` belongs to the `OPTIONAL MATCH` it follows

A `WHERE` written directly after an `OPTIONAL MATCH` is part of that clause's
pattern. Its predicate is applied **while looking for matches, not after**. A
row whose candidates all fail the predicate is therefore null-extended, not
deleted:

```python
graph.cypher("""
    MATCH (p:Person)
    OPTIONAL MATCH (p)-[:KNOWS]->(f:Person) WHERE f.age > 35
    RETURN p.name, f.name
""")
# every Person appears; f.name is NULL for those with no friend over 35
```

This holds whichever variables the predicate mentions, including one bound
before the `OPTIONAL MATCH`. `WHERE p.age > 35` decides whether the optional part
*matches*; it does not drop the person. To filter rows instead, put the
predicate where it filters: after a plain `MATCH`, or in a following
`WITH ... WHERE`, which deletes the null-extended rows:

```python
graph.cypher("""
    MATCH (p:Person)
    OPTIONAL MATCH (p)-[:KNOWS]->(f:Person)
    WITH p, f WHERE f.age > 35
    RETURN p.name, f.name
""")
# only Persons with a friend over 35
```

## Built-in Functions

| Function | Description |
|----------|-------------|
| `toUpper(expr)` | Convert to uppercase |
| `toLower(expr)` | Convert to lowercase |
| `toString(expr)` | Convert to string; `toString(null)` is `null` |
| `toInteger(expr)` | Convert to integer; a string must spell an integer (see below) |
| `toFloat(expr)` | Convert to float |
| `size(expr)` | Element count of a list, or **character** count of a string (not UTF-8 bytes) |
| `type(r)` | Relationship type |
| `id(n)` | The node's `id` **field** — the source data's own key, same value as `n.id`. Not an internal/engine identifier as in other Cypher implementations (see below) |
| `labels(n)` | Node labels as a list, primary type first |
| `degree(n)` | Node's total edge count (in + out; a self-loop counts twice) — e.g. `WHERE degree(n) > 100` to find hubs |
| `inDegree(n)` / `outDegree(n)` | Node's incoming / outgoing edge count |
| `keys(n)` / `keys(r)` / `keys(map)` | Sorted property names of a node or relationship, or entry names of a map — `keys(properties(n))` and `keys({a: 1, b: 2})` both work (as JSON list). A relationship's keys also include `type` |
| `properties(n)` / `properties(r)` | Full property map of a node or relationship (as JSON map); a relationship's map also carries its `type` |
| `start_node(r)` | Source node of a bound relationship; supports dotted access: `start_node(r).name` |
| `end_node(r)` | Target node of a bound relationship; supports dotted access: `end_node(r).name` |
| `date(str)` / `datetime(str)` | Parse a date / ISO-8601 datetime string (`date('2020-01-15')`, `datetime('2020-01-15T10:30:00Z')`), or build one from a map (`date({year: 2020, month: 1, day: 15})`) |
| `date_diff(d1, d2)` | Days between two dates (`d1 - d2`); also supports `date - date` arithmetic |
| `coalesce(a, b, ...)` | First non-null argument |
| `range(start, end [, step])` | Generate a checked inclusive integer list; default step = 1. Cardinality, the `max_work_units` budget, and a 256 MiB materialization ceiling are validated before allocation |
| `head(list)` / `last(list)` | First / last element of a list (returns `null` on empty) |
| `length(p)` | Path hop count; on a string or list, same as `size()` |
| `nodes(p)` | Nodes in a path |
| `relationships(p)` | Relationships in a path |
| `split(str, delim)` | Split string into list; an empty `delim` splits into characters |
| `replace(str, search, repl)` | Replace all occurrences |
| `substring(str, start [, len])` | Extract substring |
| `left(str, n)` / `right(str, n)` | First/last n characters |
| `trim(str)` | Remove leading/trailing whitespace |
| `ltrim(str)` / `rtrim(str)` | Left/right trim |
| `reverse(str)` | Reverse a string |
| `point(lat, lon)` / `point({latitude, longitude})` | Create a geographic point |
| `distance(a, b)` | Geodesic distance (m); geometry-aware |
| `contains(a, b)` | Does a's geometry contain b? |
| `intersects(a, b)` | Do geometries intersect? |
| `centroid(n)` | Centroid of geometry → Point |
| `area(n)` | Geodesic area (m²) |
| `perimeter(n)` | Geodesic perimeter/length (m) |
| `latitude(point)` | Extract latitude from point |
| `longitude(point)` | Extract longitude from point |
| `valid_at(e, date)` / `valid_at(e, date, 'from', 'to')` | Temporal point-in-time filter (nodes or edges); the short form reads the type's declared interval |
| `valid_during(e, start, end)` / `valid_during(e, start, end, 'from', 'to')` | Temporal range overlap filter |
| `valid_instant()` | The statement's valid-time instant (see [Temporal](#temporal-functions)) |
| `text_bm25(n, prop, query)` | Lexical (BM25) relevance of the node's — or relationship's — indexed text against a query string. Needs `build_text_index(node_type, property)` for a node, `CALL db.relationship_text_index.build({type, text_column})` for a relationship; `0.0` when the document shares no word with the query, `null` when the index has no document for that row |
| `text_score(n, prop, query)` | Semantic similarity. A **list** `query` is scored directly as your query vector; a **string** `query` is embedded first (requires `set_embedder()`) |
| `text_score(n, prop, query, metric)` | With explicit metric (`'cosine'`, `'dot_product'`, `'euclidean'`, `'poincare'`) |
| `vector_score(n, prop, vector [, metric] [, options])` | Semantic similarity against a pre-computed embedding vector (pass a list of floats directly, no `set_embedder()` needed) |
| `embedding_norm(n, prop)` | L2 norm of embedding vector (hierarchy depth in Poincaré space: 0=root, ~1=leaf) |
| `embedding(x, 'col_emb')` | The stored vector of a node or relationship as a list of floats; null when it has none, an error when its type has no such store. Compose it: `vector_score(m, 'col_emb', embedding(n, 'col_emb'))` is node-to-node (or relationship-to-relationship) similarity |
| `score_fuse(s1, s2, … [, weights])` | Fuse ranked-lane scores into one — the mean of the signals that are **present**, or a weighted mean with a trailing list. A lane that could not score the row (`null`, `NaN`, `inf`) drops out of the average together with its weight; `null` only when every lane is absent |
| `dot(a, b)` | Dot product of two list-valued vectors |
| `cosine(a, b)` | Cosine similarity of two list-valued vectors |
| `norm(a)` | Euclidean (L2) length of a list-valued vector |
| `ts_sum(n.ch [, 'start'] [, 'end'])` | Sum of timeseries values (date-string range) |
| `ts_avg(n.ch [, 'start'] [, 'end'])` | Average of timeseries values |
| `ts_min(n.ch [, 'start'] [, 'end'])` | Minimum timeseries value |
| `ts_max(n.ch [, 'start'] [, 'end'])` | Maximum timeseries value |
| `ts_count(n.ch)` | Count of non-NaN timeseries values |
| `ts_at(n.ch, 'date')` | Exact timeseries key lookup |
| `ts_first(n.ch)` / `ts_last(n.ch)` | First / last non-NaN value |
| `ts_delta(n.ch, 'from', 'to')` | Value change between two time points |
| `ts_series(n.ch [, 'start'] [, 'end'])` | Extract series as `[{time, value}, ...]` |

> **Divergence: `toInteger` on a string does not truncate.** A string argument
> must spell an *integer*. `toInteger('3')` is `3`, while `toInteger('3.7')` is
> `null` (Neo4j parses the float and truncates to `3`). So are `toInteger('abc')`
> and `toInteger(' 3 ')`: surrounding whitespace is not trimmed. A *numeric*
> argument does truncate as expected (`toInteger(3.7)` → `3`), so the portable
> spelling for a decimal string is `toInteger(toFloat(s))`. `toFloat('3.7')` is
> `3.7`, unaffected.

### Hybrid retrieval (RAG) over a knowledge graph

`vector_score` / `text_score` compose with ordinary `WHERE` predicates and
traversal, so semantic retrieval and graph constraints run in **one query**,
with no separate vector store and join. The graph filter and the similarity
ranking combine: filter first, rank the survivors by similarity, take the top
*k*.

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

The embedding store name is `{text_column}_emb` (set via
`set_embeddings(node_type, text_column, {id: vector})`). Embeddings set on the
`summary` column are therefore scored as `vector_score(a, 'summary_emb', …)`.

Stores are managed in Cypher at three tiers:

| Tier | Procedures | Notes |
|---|---|---|
| Specific embedding namespaces | `db.node_embeddings.*`, `db.relationship_embeddings.*` | The same nine procedures (`set`, `embed`, `list`, `remove`, `drop`, `build_index`, `refresh_index`, `drop_index`, `query`) with the same map parameters. One takes a list of nodes where the other holds relationships. |
| BM25 twins | `db.node_text_index.*`, `db.relationship_text_index.*` | `build`, `refresh`, `drop`, `list`. |
| Routers | `db.embeddings.*`, `db.text_index.*` | They take the union of both parameter sets plus `entity: 'node' \| 'relationship'` (a string literal) and run the specific procedure, **node by default**. |

A router refuses a parameter only the other entity takes, naming it:
``CALL db.embeddings.remove({type:'CITES', …, relationships: rs})`` → "`relationships`
belongs to entity:'relationship'".

```cypher
// Node vectors from a query: the store set_embeddings / embed_texts write.
MATCH (d:Doc) WHERE d.lang = 'en'
WITH collect(d) AS docs
CALL db.node_embeddings.embed({type: 'Doc', text_column: 'summary', nodes: docs})
YIELD embedded RETURN embedded

// Whole-store top-k, several node types merged (text: needs set_embedder()).
CALL db.embeddings.query({text_column: 'summary', text: 'graph databases', top_k: 5})
YIELD node, score, type RETURN type, node.title, score
```

`db.node_embeddings.set` takes `entries: [{node: n, vector: [...]}]` and
upserts through `add_embeddings`' path. Its writes, like every procedure here,
roll back with a statement that fails later.

Relationship embeddings use the same scoring functions and canonical store
names. Manage them from bound relationship values, so physical relationship IDs
never become an application identity:

```cypher
MATCH (claimant:Claimant)-[r:SUPPORTS]->(claim:Claim)
WHERE claim.status = 'open'
WITH collect(r) AS relationships
CALL db.relationship_embeddings.embed({
  type: 'SUPPORTS', text_column: 'evidence', relationships: relationships,
  mode: 'changed'
})
YIELD embedded, skipped, dimension, model
RETURN embedded, skipped, dimension, model
```

The `mode` parameter selects what `embed` writes:

| `mode` | Effect |
|---|---|
| `'missing'` | Generates only absent vectors. |
| `'changed'` | Also refreshes vectors whose source text changed. |
| `'all'` | Rebuilds the selected slice. A selected relationship with missing or non-string source text loses its old vector. |

Unselected vectors remain. A partial rebuild with another model therefore
reports `model: null` rather than relabeling a mixed store. A dimension change
is accepted only when the selection covers every vector already stored.
Selection is validated before the model callback, and the write is atomic. The
source text is read from live graph state, including a `SET r.evidence = ...`
earlier in the same statement.

`types: ['SUPPORTS', 'REFUTES']` in place of `type` embeds several types in one
call. Each listed type gets its own pass over the selected relationships of
that type, exactly as a separate call per type would. Every relationship in
`relationships` must be of a listed type, and `type` and `types` are mutually
exclusive. The call still yields one row: `embedded` and `skipped` are summed
over the types, and `dimension` and `model` are the value the passes share
(`null` when they differ). To embed every relationship of a type from Python,
without a selection, call `embed_relationship_texts(type, text_column, mode=…)`.

The mutating procedure must remain a top-level pipeline clause. A read-only
`CALL {}` subquery may return collected relationship values to an outer
top-level `db.relationship_embeddings.embed` call. A mutating call placed inside
the subquery or a `UNION` arm follows the existing write boundary and is
rejected before the model runs.

Manual management uses the same one-map shape. A `vector` may be a Python list
or a numpy row. A numeric 1-D or 2-D array parameter is read from its bytes, so
numpy rows bind fastest in a batched `UNWIND $batch … set` ingest:

```cypher
CALL db.relationship_embeddings.set({
  type:'SUPPORTS', text_column:'evidence',
  entries:[{relationship:r, vector:$vector}], metric:'cosine'
}) YIELD stored, dimension

CALL db.relationship_embeddings.remove({
  type:'SUPPORTS', text_column:'evidence', relationships:relationships
}) YIELD removed

CALL db.relationship_embeddings.drop({type:'SUPPORTS', text_column:'evidence'})
YIELD dropped
```

`set` upserts the relationships a query has bound. For a bulk load, with
vectors computed outside the graph for thousands of relationships, use one of
these Python calls instead:

- `add_relationship_embeddings(type, text_column, {(source_id, target_id):
  vector, …})` is the same upsert.
- `set_relationship_embeddings(…)` replaces the store as the node
  `set_embeddings` does.

The Rust equivalents are `kglite::api::embeddings::{add,set}_relationship_embeddings`.
Both calls address each relationship by its endpoint ids, take numpy rows, and
write without a per-row query pipeline.

`set` yields `stored` and `dimension`. `stored` is the number of vectors the
store holds after the call, not the number this call wrote, which no column
reports. `dimension` is the store's width. Called once per row, `stored` is
therefore a running total. The node `set_embeddings` / `add_embeddings` report
their `embeddings_stored` the same way.

Every `db.relationship_embeddings.*` procedure refuses a parameter it does not
read, naming the key and listing the ones it accepts. That includes the
per-entry map of `set`, whose keys are `relationship` and `vector`. A
misspelled option is an error, never a silently ignored one.

`set` and `embed` refuse a `text_column` that no relationship of `type` carries
(unless its store already exists), naming the properties the type does carry.
The node `set_embeddings()` / `embed_texts()` give the same refusal for an
unknown column. A misspelling therefore never creates an empty store. Store
manual vectors under a property the relationships have.

`db.relationship_embeddings.list({type?, text_column?})` reports:

- `entity`, `type`, `text_column` and the canonical `store`.
- `dimension`, `count`, `metric` and `model`.
- `index_state`, the pending `delta`, and the `unembedded` relationship count.

`delta` counts the stored vectors the index does not hold, so with no index
built it equals `count`.

In an ordinary `MATCH`, the scalar functions address a relationship store like
this:

- `vector_score(r, 'evidence_emb', $vector)` names the canonical store.
- `text_score(r, 'evidence', $text)` names the source property and embeds the
  query text.
- `embedding_norm(r, 'evidence_emb')` reads the stored vector's norm.
- `embedding(r, 'evidence_emb')` returns the vector itself, so
  `vector_score(r2, 'evidence_emb', embedding(r1, 'evidence_emb'))` scores one
  relationship against another.

Scored per row, they are exact and obey the surrounding graph filters.

The top-k shape is `RETURN … vector_score(r, …) AS s ORDER BY s DESC LIMIT k`,
or the same with `text_score`. It is served from the store, as it is for nodes:

- **Plain pattern.** A plain single-type pattern (no `WHERE`, no property maps)
  whose every relationship of the type is embedded goes straight to the store.
  It runs through HNSW when an online index serves the metric, and otherwise by
  an exact scan. A `WHERE vector_score(r, …) IS NOT NULL` filter keeps it there.
- **`WITH` before the sort.** `MATCH ()-[r:T]->() WITH r, vector_score(r, …)
  AS s ORDER BY s DESC LIMIT k RETURN startNode(r)…` is served the same way.
- **Undirected.** `(a)-[r:T]-(b)` uses the index too. It returns each
  relationship once per orientation, as the unfused pattern matches it.
- **Any other shape.** It scores its matched rows. With an online index this
  goes through HNSW with a 4× over-fetch filtered to those rows, falling back to
  the exact top-k when the filter underfills.
- **Several types.** A type alternation `()-[r:A|B]->()` or an untyped
  `()-[r]->()` (every relationship type the graph holds) is served the same
  way, per store, then merged into one top-k. This applies when every type in
  play carries the store.
  - HNSW answers only when every store's index is online. Otherwise every store
    is scanned exactly, so one answer never mixes approximate and exact
    candidates.
  - On the rows route, a store the filter covers only in part must still yield
    `k` candidates, or the query falls back to the exact top-k.
  - A type in play without the store raises the scalar's own error, as a node
    label without a store does:
    `vector_score(): no embedding 'p_emb' found for relationship type 'X'`, or
    `text_score(): no embedding for property 'p' on relationship type 'X'`.
  - Each relationship is scored under its own store's metric, exactly as the
    unfused query scores it row by row.

An HNSW answer is approximate. Pass `{exact:true}` as the final argument to
force the exact route.

- **Ties.** The answer equals the unoptimised pipeline's **up to ties**. When
  several rows score the same at the cut, which of them make the top `k`, and
  their order, can differ between the HNSW route and the exact or unfused one.
  The scores returned are the same.
- **Unembedded relationships.** A relationship with no vector scores `null` and
  so comes **first** under `DESC`. Add `WHERE vector_score(r, …) IS NOT NULL`,
  which is served from the store (see **Unembedded rows come first under
  `DESC`** below).
- **Diagnostics.** `diagnostics.retrieval` reports the route, with store
  `relationship:TYPE.property_emb`. When several stores were merged, it is a
  comma-separated list sorted by relationship type.
- **Turning it off.** `disabled_passes=['fuse_vector_score_order_limit']` turns
  the fusion off for nodes and relationships alike.

`db.relationship_embeddings.query` below ranks a whole store without a pattern.

The explicit whole-store lifecycle and query procedures are:

```cypher
CALL db.relationship_embeddings.build_index({
  type:'SUPPORTS', text_column:'evidence',
  m:16, ef_construction:200, ef_search:64, auto_refresh_limit:1000
}) YIELD indexed, metric, m

CALL db.relationship_embeddings.refresh_index({type:'SUPPORTS', text_column:'evidence'})
YIELD refreshed

CALL db.relationship_embeddings.drop_index({type:'SUPPORTS', text_column:'evidence'})
YIELD dropped

CALL db.relationship_embeddings.query({
  type:'SUPPORTS', text_column:'evidence', vector:$vector,
  top_k:10, exact:false
}) YIELD relationship, score, search_method
```

`refresh_index` folds pending changes into a built index and never builds one.
With no index it refuses, naming the `build_index` call, rather than yielding
`refreshed: 0`.

Deletes are why that matters:

- `SET` and `CREATE` leave an index `online`.
- Deleting an embedded relationship (`DELETE r`, or `DETACH DELETE` of either
  endpoint) drops it to `index_state: 'none'` until `build_index` runs again,
  exactly as deleting an embedded node drops the node index.
- A `vacuum()` that compacts drops **every** vector index in the graph, node and
  relationship, including those on types that saw no delete (a no-op on disk).
  Its result reports `tombstones_removed > 0`, whether you called it or
  auto-vacuum ran it after a large delete.
- Deleting a relationship the store holds no vector for leaves the index in
  place. So does a delete that a failed statement or a rolled-back transaction
  undoes.

Until the rebuild, `query` and the fused top-k answer by exact scan.

The query can also be text: `text:'…'` (a string literal or a `$parameter`
bound to a string) in place of `vector`. Before execution the statement embeds
it once with the registered embedder, exactly as `text_score` embeds its query,
and scores the resulting vector. The text route therefore ranks the same
relationships as passing that vector yourself:

```cypher
CALL db.relationship_embeddings.query({
  type:'SUPPORTS', text_column:'evidence', text:$question, top_k:10
}) YIELD relationship, score
```

**Several types at once.** `types:['SUPPORTS','REFUTES']` in place of `type`
ranks those stores together. Leaving out both `type` and `types` ranks every
relationship store for `text_column`. Each store answers on its own route (HNSW
when its index is online, exact otherwise) for its own `top_k`. The rows merge
into one `top_k`: score descending, then relationship type, then relationship
slot, so the order is deterministic. Each row also yields `type`, the hit's
relationship type, and its own `search_method`:

```cypher
CALL db.relationship_embeddings.query({
  types:['SUPPORTS','REFUTES'], text_column:'evidence', vector:$vector, top_k:10
}) YIELD relationship, score, type, search_method
```

A named type with no such store is refused by name. So are `type` together with
`types`, an empty `types`, and a `text_column` no store carries. Stores that
declare different metrics refuse the merge, naming both, because their scores
are not on one scale. Pass `metric` to score every store under one metric, or
query the types separately. `text:` works the same way across types.

`text` needs `set_embedder()` (or `ExecuteOptions::embedder` from Rust). Without
one the call is refused. These forms are also refused:

- `text` together with `vector`.
- A row-dependent text such as `WITH q AS t … {text: t}`. The query is embedded
  before any row exists, so it must be a statement constant.

As with `text_score`, the model is not compared with the one that built the
store.

Parameter defaults and routes:

- `top_k` defaults to 10, and `exact` defaults to false.
- `search_method` is `hnsw` only when HNSW served the query, otherwise `exact`.
  `exact:true` always bypasses the index.
- A requested metric that cannot use the installed index falls back to the
  exact route.
- `build_index` takes an optional `metric`. It becomes the store's metric when
  the store declares none, so a later `query` that names no metric resolves the
  one the index answers under. A `metric` that contradicts one the store
  already declares is refused, rather than producing an index the default query
  route cannot use.

The procedure ranks the complete declared relationship store before subsequent
clauses run, so `WHERE` after `YIELD` filters its top-k output. To constrain
ranking by endpoints or relationship properties, use filtered `MATCH` with the
exact scalar functions instead.

> **`vector_score` takes the store name, `text_score` takes the raw column.**
> `vector_score` names the store directly: `'summary_emb'`. `text_score` names
> the source *column*: `'summary'` (it resolves to `summary_emb`). That is why
> the example above uses `text_score(p, 'text', …)` (raw column `text`), not
> `'text_emb'`. The Python API (`embedding_info`, `vector_search`,
> `search_text`) likewise uses the raw column name throughout. Only Cypher's
> `vector_score` is in store-name terms.

> **Both scoring functions accept your own query vector.** `text_score` is
> `vector_score` after a plan-time rewrite, so the two differ only in how they
> name the column and in how they read the query argument. Give either one a
> list of numbers (a literal `[0.1, 0.2, …]` or a `$param` bound to a list) and
> it scores that list directly as the query vector, needing only the embedding
> store. That makes semantic scoring reachable from every binding: any language
> that can send a list parameter can query by vector.
>
> ```cypher
> // identical results; both need only the embedding store
> RETURN vector_score(n, 'summary_emb', $q) AS s   // store name
> RETURN text_score(n, 'summary', $q)     AS s     // raw column
> ```
>
> **The query argument's type selects how it is scored.**
>
> - In `text_score`, a list is scored as a vector and a string is scored as
>   text. `text_score(n, 'summary', [1.0, 2.0])` scores that two-element vector,
>   while `text_score(n, 'summary', '[1.0, 2.0]')` embeds the 10-character string
>   and requires `set_embedder()`.
> - `vector_score` reads a list as a vector and also parses a JSON-array string
>   as one (a legacy form kept for compatibility). Pass a list to have both
>   spellings agree.
> - A `$param` used as the query argument must be bound to a string or a list.
>   Plan-time validation reports the type of anything else.

> **The first argument may be a node or relationship *value*, not only a bound
> variable.** A node or relationship that reaches the scalar through
> `collect(n)` plus `UNWIND`, `head(...)`, a `CALL { }` column or `nodes(p)` /
> `relationships(p)` is resolved back to the entity it names. It is scored
> exactly as the binding is, for `vector_score`, `text_score` and
> `embedding_norm` alike. A value is a snapshot, so one whose entity has since
> been deleted, or whose slot has been reused by an entity of another type,
> scores `null` rather than scoring the new occupant.

> **Retrieval policy.** `vector_score` and `text_score` accept an optional final
> map: `{exact: true}` forces an exact scan without using or refreshing HNSW.
> Put it fourth when omitting the metric, or fifth after an explicit metric:
> `vector_score(n, 'summary_emb', $q, 'cosine', {exact: true})`. A
> parameter-bound map works too.
>
> - `exact` must be boolean and defaults to false.
> - Unknown options are errors.
> - An omitted metric uses each actual node type's store metric, defaulting to
>   cosine if the store has none.
>
> **Index-accelerated top-k.** With default policy and an HNSW index,
> `RETURN vector_score(n, prop, q) AS s ORDER BY s DESC LIMIT k` (or the
> `text_score` form) can narrow candidates approximately, then score those
> candidates exactly. These cases use exact execution:
>
> - ASC, and explicit NULLS LAST.
> - Row-dependent selectors, and mixed/duplicate or unembedded bindings.
> - Incompatible metrics, and unavailable indexes.
> - Filtered candidate underfill.
>
> Filters alone do not guarantee an exact scan: request `{exact: true}` when that
> matters. Invalid dimensions, metrics or options raise in filters as well as
> projected scores.
>
> **Unembedded rows come first under `DESC`.** `vector_score` and `text_score`
> are `null` for a node or relationship the store holds no vector for, and
> openCypher sorts `null` above every value. `ORDER BY s DESC LIMIT k` therefore
> fills its first rows with unembedded entities.
>
> - A node type with unembedded members is still answered from its store: the
>   null-scored nodes first, in the type's order, then the store's ranking. It
>   reports `fallback_reason: 'row_coverage'` only when all `k` rows are null or
>   the store's order differs from the type's.
> - A relationship type with unembedded members is answered by row scan and
>   reports `row_coverage`.
>
> To leave the unembedded rows out, filter them in the `MATCH`'s own `WHERE`:
>
> ```cypher
> MATCH (n:Doc) WHERE vector_score(n, 'body_emb', $q) IS NOT NULL
> RETURN n.title, vector_score(n, 'body_emb', $q) AS s ORDER BY s DESC LIMIT 10
>
> MATCH (a)-[r:SUPPORTS]->(b) WHERE vector_score(r, 'evidence_emb', $q) IS NOT NULL
> RETURN a.title, b.title, vector_score(r, 'evidence_emb', $q) AS s ORDER BY s DESC LIMIT 10
> ```
>
> The filter is served from the store at the cost of the store procedure, not by
> scoring every row. It uses HNSW once an index is online, with
> `diagnostics.retrieval` naming the store, for nodes and relationships alike
> (the `text_score` form too).
>
> - The same test written after the projection, `WITH … AS s WHERE s IS NOT NULL`,
>   leaves the fused route and scores every row exactly.
> - `ORDER BY s DESC NULLS LAST` also puts them last, by exact execution
>   (`ordering_requires_exact`).
> - `vector_search`, `search_text` and `db.relationship_embeddings.query` rank
>   the stored vectors only, so they never return an unembedded entity.

### Lexical search — `text_bm25`

`text_bm25(n, 'property', 'query text')` ranks a row by **Okapi BM25** against a
lexical index: word overlap, weighted by how rare each word is in the corpus and
normalised for document length. It needs no embedder and no vectors. It is the
keyword-search half of hybrid retrieval, and it finds the exact term (a product
code, a name, a rare noun) that an embedding blurs away.

The index is **opt-in and explicit**, like `build_vector_index`:

```python
graph.build_text_index("Article", "body")     # Python; every binding has it
```

In Cypher the same index is `CALL db.node_text_index.build({type: 'Article',
text_column: 'body'})`. It also has `refresh`, `drop` and `list`, and
`db.text_index.*` routes to it by default.

```cypher
// Rank the corpus, best first.
MATCH (a:Article)
RETURN a.title, text_bm25(a, 'body', 'photosynthesis in low light') AS score
ORDER BY score DESC LIMIT 10
```

```cypher
// It is an ordinary scalar: filter first, then rank the survivors.
MATCH (a:Article)-[:WRITTEN_BY]->(p:Person {name: 'Vera'})
WHERE a.year >= 2020
RETURN a.title, text_bm25(a, 'body', 'low light') AS score
ORDER BY score DESC LIMIT 5
```

Filter on the **graph** (a year, an author, a hop), never on the score. The
first form above, a bare `MATCH` over the whole indexed type with the score
projected and ordered, is the only shape the postings top-k operator can claim.
A `WHERE` makes the rows a subset of the corpus and sends the query back to
scoring every document. Measured over 50,000 documents with a term in 0.1% of
them, `WHERE text_bm25(…) > 0 … ORDER BY score DESC LIMIT 10` cost **53×** the
same query without the `WHERE` (6.1 ms vs 114 µs). The filter also buys
nothing: a document sharing no word with the query scores `0.0` and is already
last.

Semantics:

| Case | Result |
|------|--------|
| The row's document shares no word with the query | `0.0` |
| The index holds no document for that row (created after the build and not yet caught up, or its property is absent / not a string) | `null` |
| The query argument is `null` | `null` |
| No text index on the node's `(type, property)` | **error** naming `build_text_index` |

`0.0` and `null` are deliberately different answers. `0.0` says the document was
searched and did not match; `null` says it was not searched. Collapsing them
would make an index that is quietly behind the graph look like a corpus with no
matches.

Tokenization is the same on both sides: lowercase, split on anything that is
not a letter or a digit. There is **no stemming and no stopword list**. BM25's
IDF term already discounts a word that appears in nearly every document,
without a language-specific list to maintain or be wrong about. A repeated query
word is weighted once, so `'rust rust'` ranks exactly like `'rust'`.

Ties break by node id, so a query over unchanged data returns the same order
every time.

**Staleness is visible, never silent.** The index does not follow writes. It
records them and folds them in when a query next reads it, as long as the
outstanding delta is within that index's `auto_refresh_limit`. That is why a
node created after the build scores without anyone rebuilding.

Past the limit (or on a read-only graph, which a query may not write to), the
query serves what the index has, scores the rest `null`, and returns a warning
naming the delta and the rebuild call. `SHOW INDEXES` reports `stale` and
`delta` either way.

Folding a document in is not a constant-cost operation. It inserts into the
posting list of every term that document uses, and those lists grow with the
corpus, so `auto_refresh_limit` bounds a document *count*, not a duration. Past
roughly 1500 documents, folding costs more than rebuilding the index outright
and the catch-up rebuilds instead. A refresh costs the cheaper of the two and
never more than one rebuild, whatever the limit is set to.

### Lexical search over relationships — `db.relationship_text_index.*`

`text_bm25(r, 'property', 'query text')` ranks a relationship exactly as it
ranks a node, over a BM25 index on one relationship type's property. The
relationship index's lifecycle is in Cypher, like the relationship vector
index's, so every binding reaches it:

```cypher
CALL db.relationship_text_index.build({type:'SUPPORTS', text_column:'evidence', auto_refresh_limit:1000})
YIELD indexed, skipped, terms

CALL db.relationship_text_index.refresh({type:'SUPPORTS', text_column:'evidence'}) YIELD refreshed

CALL db.relationship_text_index.drop({type:'SUPPORTS', text_column:'evidence'}) YIELD dropped

CALL db.relationship_text_index.list({type:'SUPPORTS'})
YIELD entity, type, property, documents, terms, skipped, index_state, delta, auto_refresh_limit
```

```cypher
MATCH (c:Claimant)-[r:SUPPORTS]->(claim:Claim)
RETURN claim.title, text_bm25(r, 'evidence', 'water damage') AS score
ORDER BY score DESC LIMIT 10
```

| Procedure | Yields | Notes |
|---|---|---|
| `db.relationship_text_index.build({type, text_column, auto_refresh_limit?})` | `indexed`, `skipped`, `terms` | Builds or replaces the index. Same document rule as the node index (a string, or a list of strings/nulls joined); refuses an unknown relationship type and a property no relationship of the type carries as text |
| `db.relationship_text_index.refresh({type, text_column})` | `refreshed` | Folds in every change since the last build or refresh, whatever the limit; refuses a missing index |
| `db.relationship_text_index.drop({type, text_column})` | `dropped` | `false` when there was no such index |
| `db.relationship_text_index.list({type?, text_column?})` | `entity`, `type`, `text_column`, `documents`, `terms`, `skipped`, `index_state`, `delta`, `auto_refresh_limit` | Read-only; one row per index, sorted |

The semantics are the node lane's:

- `0.0` for no shared word.
- `null` for a relationship the index holds no document for.
- An error naming `db.relationship_text_index.build` when no index exists.
- Writes are folded in at the next query within `auto_refresh_limit`. They are
  `SET`, `REMOVE`, `CREATE`/`MERGE` of a relationship (including one that reuses
  a deleted relationship's storage slot), and `add_connections`.
- A deleted relationship's document is removed at the delete.

The first argument may be a `MATCH` binding or a relationship value
(`collect(r)[0]`, `UNWIND`, a `CALL { }` column, the `relationship` column of
`db.relationship_embeddings.query`). A binding deleted earlier in the same
statement scores `null`. `build`, `drop` and any catch-up a statement did are
undone when the statement fails.

`SHOW INDEXES` lists the index as `relationship:SUPPORTS.evidence`, type
`FULLTEXT`, entityType `RELATIONSHIP`.

- `DROP INDEX relationship:SUPPORTS.evidence` removes it, together with a
  relationship vector index on the same property. This is the node rule that
  one index name covers every structure registered under it.
- It persists in `.kgl` (a section that older readers skip) and is dropped by
  `vacuum()`.
- It is available in memory and mapped storage only: a disk-backed graph
  refuses to build one.
- Like the node text index, it is not recorded in the write-ahead log. On a
  durable graph, an index built after the last checkpoint is absent after
  reopen until rebuilt.

Hybrid ranking works over relationships unchanged:

```cypher
MATCH ()-[r:SUPPORTS]->()
RETURN r, score_fuse(text_bm25(r, 'evidence', $q), vector_score(r, 'evidence_emb', $qv)) AS score
ORDER BY score DESC LIMIT 10
```

### Fusing the lexical and semantic lanes — `score_fuse`

`score_fuse(s1, s2, …)` combines the scores of several ranked lanes into one
number, so a keyword lane and a semantic lane can rank the same query in a
single pass:

```cypher
MATCH (a:Article)
RETURN a.title,
       score_fuse(text_bm25(a, 'body', $q), vector_score(a, 'body_emb', $qv)) AS score
ORDER BY score DESC LIMIT 10
```

Each lane finds what the other misses. BM25 finds the exact term (a product
code, a surname, a rare noun) that an embedding blurs away, and the embedding
finds the paraphrase that shares no word with the query.

By default the lanes weigh equally. A **trailing list** weights them, in
argument order:

```cypher
// Lexical evidence counts for 70% of the score, semantic for 30%.
RETURN score_fuse(text_bm25(a, 'body', $q), vector_score(a, 'body_emb', $qv), [0.7, 0.3]) AS score
```

Any number of lanes fuse, and the weights are relative: `[3, 1]` and
`[0.75, 0.25]` produce the same ranking. Weights must be finite and `>= 0`, one
per score. A wrong-length list, a negative weight and a non-numeric score are
all errors rather than a quietly different ranking.

**An absent lane leaves the average; it does not score zero.** A lane reports
`null` (or `NaN`, or an infinity) for a row it *could not see*, such as a
document the text index has not caught up with or a node with no stored
embedding. That row keeps the score of the lanes that did run, and its weight
leaves the denominator with it. Zero would mean "this lane looked and found
nothing", which ranks a document one lane simply could not see below a document
both lanes actively disliked. `score_fuse` is `null` only when **every** lane is
absent. It reads no row state, so a call whose arguments are all constants is
folded once for the whole query rather than evaluated per row.

> **Where is `rrf()`?** Reciprocal Rank Fusion works on each lane's **rank**
> across the whole result set, and a per-row scalar sees only one row's scores.
> A function that claimed to do RRF row-by-row would be computing something
> else. Rank the lanes first with a window function, then fuse the reciprocals:
>
> ```cypher
> MATCH (a:Article)
> WITH a, rank() OVER (ORDER BY text_bm25(a, 'body', $q) DESC) AS lex_rank,
>         rank() OVER (ORDER BY vector_score(a, 'body_emb', $qv) DESC) AS vec_rank
> RETURN a.title, score_fuse(1.0 / (60 + lex_rank), 1.0 / (60 + vec_rank)) AS score
> ORDER BY score DESC LIMIT 10
> ```
>
> That is the whole of RRF, in the primitives that already exist. Reach for it
> when the lanes' scores are on incomparable scales (BM25 is unbounded, cosine
> is not), since ranks discard the magnitudes. Fuse the scores directly when the
> magnitudes carry information you want.

### Vector math over list properties — `dot` / `cosine` / `norm`

`vector_score` and `embedding_norm` read the **embedding store**
(`set_embeddings(...)`). `dot(a, b)`, `cosine(a, b)` and `norm(a)` read
**ordinary list-valued data** instead: a stored list property, a list literal, a
`$param` bound to a list, or a `collect()`. Use them when the vectors live in the
graph as data rather than in a registered store, or to compare two nodes'
vectors against each other.

```cypher
// Rank documents against a query vector held in a parameter.
MATCH (d:Doc)
RETURN d.title, cosine(d.vec, $q) AS score
ORDER BY score DESC LIMIT 10
```

```cypher
// Compare two nodes' own stored vectors — no store, no embedder.
MATCH (a:Doc {id: 1}), (b:Doc {id: 2})
RETURN dot(a.vec, b.vec) AS dot, cosine(a.vec, b.vec) AS cos, norm(a.vec) AS len
```

Semantics:

| Case | Result |
|------|--------|
| Any argument is `null` (including a missing property) | `null` |
| The two vectors have different lengths | **error** naming both lengths |
| An element is not a number (a string, a `null`, a bool) | **error** naming the vector and the position |
| An argument is not a list | **error** — reported even when the other argument is `null` |
| `cosine` where either vector has zero length | `null` (`0/0` is undefined) |
| `norm([])`, `dot([], [])` | `0.0` (the empty sum) |

A length mismatch is an error rather than `null`. A 384-dimension vector meeting
a 768-dimension one is a data bug, and a `null` would sit unremarked in a column
of otherwise plausible scores. This is the same reason Neo4j's
`vector.similarity.*` family only compares equal dimensions. A `null` *element*
is likewise not silently treated as `0.0` (Neo4j's GDS does substitute zero),
because a zeroed component changes the answer without changing its shape.

`cosine` returns `null` rather than `0.0` for a zero-length vector. That differs
from `vector_score`, which answers `0.0` there because a top-k ranking needs a
total order over every candidate.

A property whose value is a *bracketed string* (`'[3.0, 4.0]'`) is read as a
list here, exactly as `head()` / `last()` / `UNWIND` read it, so a graph that
stored its vectors as text answers too. `size()` / `length()` are the exception:
they measure a string as a string (see the dialect note under
[String Functions](#string-functions)).

## Spatial Functions

Built-in spatial functions for geographic queries. All node-aware functions
auto-resolve geometry and location via [spatial types](https://kglite.readthedocs.io/en/latest/python/guides/spatial.html).

| Function | Returns | Description |
|----------|---------|-------------|
| `point(lat, lon)` | Point | Create a geographic point |
| `point({latitude, longitude})` | Point | The same point from a map; `{x, y, crs: 'wgs-84'}` (or `srid: 4326`) with `x` the longitude also works |
| `distance(a, b)` | Float (m) | Geodesic distance (WGS84); geometry-aware (0 if inside/touching) |
| `distance(lat1, lon1, lat2, lon2)` | Float (m) | Geodesic distance (4-arg shorthand) |
| `contains(a, b)` | Boolean | Does a's geometry contain b? (point-in-polygon or geometry containment) |
| `intersects(a, b)` | Boolean | Do geometries intersect? |
| `centroid(n)` | Point | Centroid of geometry (node or WKT string) |
| `area(n)` | Float (m²) | Geodesic area of polygon (node or WKT string) |
| `perimeter(n)` | Float (m) | Geodesic perimeter/length (node or WKT string) |
| `latitude(point)` | Float | Extract latitude component |
| `longitude(point)` | Float | Extract longitude component |

All functions accept both nodes (auto-resolved via spatial config) and raw
values (WKT strings, Points).

> **Coordinate order:** `point(lat, lon)` uses **latitude-first** (geographic
> convention). WKT strings use **longitude-first** per OGC standard:
> `POLYGON((lon lat, lon lat, ...))`. These conventions differ, so be careful
> when mixing them.

Points are 2D WGS-84. These forms are an error rather than a geographic point
with the wrong distance semantics:

- A map naming a Cartesian point (`point({x: 1, y: 2})`, `crs: 'cartesian'`,
  `srid: 7203`).
- A map naming a 3D point (`height`, `z`, `wgs-84-3d`).

A `null` map or a `null` coordinate gives `null`.

```python
# Node-aware spatial — with spatial config declared via column_types
graph.cypher("""
    MATCH (c:City), (a:Area)
    WHERE contains(a, c)
    RETURN c.name, a.name
""")

graph.cypher("""
    MATCH (a:Project), (b:Project)
    WHERE intersects(a, b) AND a <> b
    RETURN a.name, b.name
""")

graph.cypher("""
    MATCH (n:Project)
    RETURN n.name, area(n) AS area_m2, centroid(n) AS center
""")

# Geometry-aware distance
graph.cypher("""
    MATCH (a:Project), (b:Project) WHERE a <> b
    RETURN a.name, b.name, distance(a.geometry, b.geometry) AS dist
""")  # 0 if polygons touch, centroid distance otherwise

graph.cypher("""
    MATCH (n:Project)
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
    MATCH (a:Project), (b:Project) WHERE a <> b
    RETURN avg(distance(a, b)) AS avg_dist, std(distance(a, b)) AS std_dist
""")
```

### Geometry primitives

Constructive operations on WKT geometries. All accept WKT strings, node
variables (auto-resolved via spatial config), or `Point` values. All return WKT
strings (or boolean / float as noted).

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

# Union of overlapping contract areas
graph.cypher("""
    MATCH (a:Contract), (b:Contract) WHERE a.id < b.id AND intersects(a, b)
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

`kg_knn` looks up the *k* nodes of `target_type` closest to `(lat, lon)`
(geodesic).

- It uses the node's `location` config for point comparisons.
- It falls back to the geometry centroid when `location` is not configured.
- Nodes without spatial config are skipped silently.

## Temporal Functions

This section covers date and datetime values, date arithmetic, and the
`valid_at` / `valid_during` validity tests. To ask a whole statement as of an
instant, prefix it with
[`FOR VALID_TIME AS OF`](#statement-context-for-valid_time-as-of). The
[valid-time guide](https://kglite.readthedocs.io/en/latest/python/guides/valid-time.html)
covers declaring intervals and modelling history.

| Function | Description |
|----------|-------------|
| `date()` | Today's date in UTC (no-arg form) |
| `date(str)` | Parse a date string to a DateTime (date-only) value: `'YYYY'`, `'YYYY-MM'`, `'YYYY-MM-DD'`, or ISO 8601 basic `'YYYYMMDD'`; any other string raises (a null argument gives null) |
| `date(datetime)` | The date part of a datetime value |
| `date({year, month, day})` | Build a date from integers (openCypher's map form): `date({year: y, month: 1, day: 1})`. `month` and `day` default to 1; an impossible date, an unknown key or a non-integer component raises; a null component gives null |
| `datetime({year, month, day, hour, minute, second, millisecond, microsecond, nanosecond})` | Build a zoneless datetime the same way; missing time fields are 0. A `timezone` key is refused |
| `datetime(str)` | Parse an ISO-8601 stamp to a Timestamp (date + time). Accepts `YYYY-MM-DD`, `…THH:MM`, `…THH:MM:SS[.fff]`, and a zoned `…Z` / `…±HH:MM`. **A zone is normalised to UTC**, since `Value::Timestamp` carries no zone. Sub-second digits are kept and compare: `datetime('…42.317')` is later than `datetime('…42')` (a Python `datetime` result carries them to the microsecond). An unparseable string raises; a null argument gives null |
| `datetime()` | Now in naive UTC (no-arg form) — the clock every stored datetime, validity instant and `auto_timestamp` uses |
| `localdatetime()` | Local wall-clock datetime; 1-arg form parses/normalises a string (raises on bad input). Unlike `datetime(str)` it keeps the wall-clock reading of a zoned input and drops only the zone label |
| `time()` | Time of day in UTC as an `HH:MM:SS` string (no-arg form); 1-arg form parses/normalises a string as `localtime(str)` does (raises on bad input) |
| `localtime()` | Local wall-clock time of day as an `HH:MM:SS` string; 1-arg form parses/normalises a string (raises on bad input) |
| `n.d.year`, `n.d.month`, `n.d.day` | Extract component from a DateTime property (chained accessor — works in `RETURN`, `WHERE`, `ORDER BY`) |
| `n.d.dayOfWeek`, `n.d.dayOfYear`, `n.d.epochSeconds` | Other temporal field accessors |
| `duration({days: N, months: M, ...})` | Build a Duration value (see [Duration semantics](#duration-semantics) below) |
| `duration.between(d1, d2)` | Difference between two date or timestamp values; whole days and remaining seconds are returned as a Duration |
| `add_days(date, n)` / `add_months(date, n)` / `add_years(date, n)` | Checked calendar shift; returns NULL when the requested date is outside the representable range |
| `date_truncate(date, unit)` | Start of `year`, `month`, `week`, or `day` |
| `date + duration({days: N})` | Add a duration to a date |
| `duration * integer` / `integer * duration` | Scale all duration components with checked arithmetic |
| `date_diff(d1, d2)` | Days between two dates (legacy; same as `d2 - d1` returning Int64 directly) |
| `date + N` / `date - N` | Add/subtract N days (Int64 form, kept for backward compat) |
| `date - date` | Returns a Duration (was Int64 days pre-0.9.0) |
| `valid_at(entity, date)` | True if entity is valid at a point in time under its type's declared bounds and convention (see [Validity-interval declarations](#validity-interval-declarations)); raises on a type with no declaration |
| `valid_during(entity, start, end)` | True if entity's declared interval overlaps `[start, end]` |
| `valid_at(entity, date, 'from_field', 'to_field')` | True if entity is active at a point in time — closed, unless the type's declaration names the same two properties, whose convention then applies |
| `valid_during(entity, start, end, 'from_field', 'to_field')` | True if entity's range overlaps the given interval, under the same rule |
| `valid_instant()` | The instant the statement's valid-time context reads: a date, or the datetime it was written as; today (UTC) under the default context. An error under `FOR VALID_TIME ALL`, in a write statement, and where no context applies (a graph with no declaration and no prefix). It does not count as a `valid_at()` call: a statement using it still gets the default context |

The entity may be a matched variable or a value: an item of `collect()`,
`UNWIND`, `nodes(p)`, `relationships(p)` or a variable-length relationship list.
Its type's declaration is read the same way either way.

**The functions are not the statement context.** `valid_at(n, date)` reads one
declaration: the node's primary type's, else its first declared secondary
label's (alphabetically). `valid_at(r, date)` tests the relationship's own
interval and not its endpoints.

- A relationship type holding several unkeyed declarations is read per row, by
  the first declaration whose bounds the relationship carries.
- A type declared only for other source types raises for a relationship out of
  any other source. The error names them: pass the bounds or declare that
  source. The statement context treats such a relationship as timeless.
- Each call filters only the element it names.

Under [`FOR VALID_TIME AS OF`](#statement-context-for-valid_time-as-of), and in
the fluent API's date context, the rules differ:

- A node must be valid under *every* declared label it carries.
- A relationship needs both endpoints valid.
- The ambiguous type is refused.
- Every element of the statement is filtered.

Use the functions for a second instant or for bounds no declaration names. Use
the context for "as of".

**NULL semantics:** NULL `from` = valid since beginning. NULL `to` = still
valid. Both NULL = always valid.

- A null entity (an unmatched `OPTIONAL MATCH`) gives null in every form, so
  `WHERE` drops the row.
- A property name that no element of the type has at all, such as a
  misspelling like `'validfrom'`, raises `CypherExecutionError` rather than
  reading as an open bound on every row.
- A property the type has but a row leaves null is open, as above.
- A relationship type's declared bound counts as known even before any
  relationship has a value for it.

**Query dates:** `valid_at` / `valid_during` take a `date()` or `datetime()`
value, or a string read as those functions read it. `'2009'` = `date('2009')` =
2009-01-01, and an offset in a datetime string is applied and normalised to UTC.
Any other value (an unparsable string, an integer, `null`) raises
`CypherExecutionError`. So does a stored bound that is not a date, a datetime or
an ISO string. A datetime bound compares exactly against a datetime and at date
grain against a date.

**Magnitude/error policy:** a well-typed date/calendar shift that lands outside
the representable chrono range returns NULL. These cases raise
`CypherExecutionError`:

- Invalid types.
- A zero range step.
- Range allocation/budget overflow.
- Duration construction or arithmetic that would overflow a stored component.

Values are never narrowed, wrapped, or silently truncated. Fractional or
non-finite Duration constructor components are rejected.

**The same policy covers plain integer arithmetic.** `+ - * / %` and unary `-`
on two integers raise `CypherExecutionError` when the result leaves the signed
64-bit range (`9223372036854775807 + 1`, `-9223372036854775808 / -1`).
**Integer division or modulo by zero raises** rather than answering NULL.
*Float* division by zero stays NULL. The IEEE answer is ±Infinity / NaN, which
no wire format kglite ships over can carry, so promoting it would move the
silence one layer out rather than remove it.

**`datetime()` and `localdatetime()` return timestamp values.**

- With no argument, `datetime()` (like `date()` and `time()`) reads the clock in
  UTC. `localdatetime()` (like `localtime()`) reads it in the process's local
  zone.
- An offset-bearing `datetime(str)` is normalized to naive UTC.
  `localdatetime(str)` keeps the local wall-clock reading and drops the zone.
- `localtime()` and `time()` return `HH:MM:SS` strings, because KGLite has no
  time-only value type.
- Each single-string form returns NULL on unparseable input.

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

### Validity-interval declarations

`db.temporal.declare` records which two properties bound a node label's or a
relationship type's validity interval. It also records whether the `to` day is
still valid (`convention: 'closed'`) or is the first day no longer valid
(`'half_open'`). The convention is required. A datetime `to` under `half_open`
excludes only from its own time, so a `to` after midnight leaves its day valid.

Cypher and the fluent API both read a declaration:

- The fluent `select()`, `valid_at()`, `valid_during()` and `traverse()` filters
  read it.
- In Cypher, `valid_at(m, date('2010-01-01'))` and `valid_during(m, start, end)`
  read the declared bounds and follow the declared convention.
- The four- and five-argument forms follow it too when the declaration names the
  same two properties. Only a property pair no declaration names is read closed.
- A relationship takes its source type's keyed declaration first, as the fluent
  filters do.

```cypher
CALL db.temporal.declare({node: 'FieldStatus', from: 'date_from', to: 'date_to', convention: 'closed'})
CALL db.temporal.declare({relationship: 'HAS_HOLDER', source_type: 'Project',
                          from: 'holder_from', to: 'holder_to', convention: 'half_open'})
  YIELD declared, rows, abutting_rows
CALL db.temporal.declare({node: 'Assignment', from: 'granted', to: 'ended', convention: 'closed',
                          empty_when: 'to_before_from'})
CALL db.temporal.undeclare({relationship: 'HAS_HOLDER', source_type: 'Project'}) YIELD undeclared
CALL db.temporal.declarations()
  YIELD kind, name, source_type, from, to, convention, empty_when, abutting_rows,
        ambiguous, empty_rows, unreadable_rows
```

- **Target.** Exactly one of `node` (a primary type or a secondary label) or
  `relationship`.
  - `source_type` narrows a relationship declaration to the relationships
    leaving nodes of that type.
  - A relationship uses its source's declaration first and the unkeyed one
    otherwise.
- **Validation.** The `from` property must exist on the target.
  - A `to` property that no row carries yet (every period still open) is
    accepted with a warning ("no row of … carries '…'; every row is open-ended
    until one is written"). The exception is a near miss of a property the
    target has, which is refused as a typo with a "Did you mean" hint.
  - Every stored bound must be NULL, a date, a datetime or an ISO string, with
    `from` not after `to`. The first row that fails is refused, naming the
    node's id or the relationship's endpoints.
  - `from == to` is a one-day interval under `closed`. Under `half_open` it is
    an **empty interval**, valid at no instant, which is accepted. The
    declaration warns once, counting such rows and naming the first, and counts
    them in `empty_rows`.
- **`empty_when: 'to_before_from'` (optional, `closed` only).** Under `closed`,
  a `to` before the `from` is refused as inverted.
  - With this option, a date `to` exactly one day before a date `from` (a
    version superseded the day it was registered) is instead an empty interval,
    treated like the `half_open` one. It is accepted, valid on no day, counted
    in `empty_rows`, and warned once ("N of M rows … have an empty interval
    under convention 'closed' with empty_when 'to_before_from' …").
  - A timestamp bound or a wider inversion is still refused.
  - `half_open` with the option is an error (it already holds `from == to` as
    empty).
  - `db.temporal.declarations()` reports it in the `empty_when` column (null
    when unset).
  - The saved file gains one optional key that an older build ignores,
    answering every query the same.
- **Later writes answer to the same rule.** A `CREATE`, `MERGE` or `SET`
  (including `SET n:Label` onto a declared label) refuses a row the declaration
  would refuse, with the same wording. So does an `add_nodes` /
  `add_relationships` / blueprint load onto a declared type.
  - The same holds for every writer that gives a node a declared label:
    `add_nodes(labels=[…])`, `add_label`, a blueprint's or `from_records` spec's
    `labels`, `extend()`'s label union, ontology materialisation and the
    materialised ancestors a new node is born with.
  - Cypher names the node's id or the relationship's endpoints and fails the
    statement. The statement rolls back, an `UNWIND … CREATE` included.
  - A load names the row by its 0-based position and writes nothing.
  - A `SET` is judged once its clause has applied every item, so
    `SET n.from = …, n.to = …` moves an interval in one step.
  - NULL bounds stay open.
  - A load judges an update row by the bounds it leaves, such as a row carrying
    only `to` against the stored `from`, under its conflict mode.
  - A fluent `update()`, `add_properties()` and the `store_as=` writers are
    judged the same way.
- **Empty intervals are kept.** A row a write leaves with `from == to` under
  `half_open` (an assignment created and cancelled on the same day) is written.
  - The statement reports one warning in `result.warnings` ("N of M rows written
    have an empty interval under convention 'half_open' … the first is node
    '…'").
  - A load emits the same text once as a `UserWarning`, naming the first row by
    its 0-based position.
  - Such a row is valid at no instant. No `FOR VALID_TIME AS OF` statement,
    `valid_at` / `valid_during` call, frozen view or fluent temporal filter
    returns it. A statement that reads every version (`FOR VALID_TIME ALL`: a
    lineage or as-known-at query) still reads it.
- **What the counts report.** `db.temporal.declarations()` counts, at the
  graph's current state:
  - `empty_rows`: rows valid at no instant (`from == to` under `half_open`,
    kept with a warning, or an inverted interval).
  - `unreadable_rows`: a bound that is not NULL, a date, a datetime or an ISO
    string.

  Every write refuses an inverted interval or an unreadable bound. Only an
  undeclare that hands a source's relationships to the unkeyed declaration, or
  a graph saved by an earlier version, which accepted such writes, leaves one. A bound that
  is not a date raises from the next `valid_at` / `valid_during` (or fluent
  temporal filter) that reads it, naming the node's id or the relationship's
  endpoints and the property. An inverted interval is valid on no date.
  Counting reads every row of the declared type once after each write.
- **Re-declaring** the same target with the same properties and convention is a
  no-op (`declared: false`). Different ones are refused until the target is
  undeclared.
- **`abutting_rows`** counts rows whose `to` equals another row's `from` within
  one entity at declare time. An entity is the relationships of one source node,
  or the node rows of a label that share an `id` (a blueprint sub-node with
  `parent_fk` groups by its parent edge, declared or generated).
  - Rows of different entities that share a date are not counted. Under
    `closed` the warning mentions them separately as possibly unrelated.
  - Under `closed` both such rows are valid on that day, so a non-zero count
    adds a query warning suggesting `half_open`.
  - On a disk-mode graph the count is skipped (NULL) for a node label above
    250,000 rows.
- **Saved with the graph.** Declarations, conventions and declare-time counts
  persist in a `.kgl` file. For older KGLite versions the file also records
  closed node declarations and each relationship type whose declarations are all
  closed and have no `source_type`. A type with a half-open or per-source
  declaration is left out of that record, so an older version reads it as
  undeclared rather than misreading it.
- **`ambiguous`** is `true` for a relationship type holding several declarations
  without a `source_type`, which only a graph saved by an older version can
  hold. Which one an edge uses depends on the order they were added in.
  Undeclare them and re-declare each with its `source_type`.
- **From Python.** `set_temporal()` and the loaders' `validFrom`/`validTo` column
  types declare through the same rules. They take an optional `convention` that
  keeps an existing declaration of the same properties and is `closed`
  otherwise. A blueprint declares through a spec's `temporal` key, which must
  name its convention.
  - A bulk load onto a declared relationship type never updates a stored
    relationship.
  - A row identical to one already between its endpoints is dropped. Identical
    means every property it holds is equal, with values compared by what they
    name (one day in any spelling is one value, a whole float equals its
    integer) and provenance stamps left out.
  - Any other row (a new period, a closing `to`, another value of any property)
    is a new, parallel relationship. This holds in every `conflict_handling`
    mode, and whether the rows arrive in one call or several.
  - To close or correct a stored period, `SET` or `DELETE` it in Cypher.

### Statement context: `FOR VALID_TIME AS OF`

A statement prefixed `FOR VALID_TIME AS OF <instant>` asks the whole query as of
that instant on the declared types. **A statement with no prefix on a graph that
declares validity runs as `FOR VALID_TIME AS OF date()`** (today, UTC).
`FOR VALID_TIME ALL` reads every version (see *The default and `ALL`* below).

- The prefix stands before or after `EXPLAIN` / `PROFILE`.
- The instant is a quoted ISO date or datetime, `$param`, `date(…)` /
  `datetime(…)` of a literal or parameter, or `date()` for today in UTC.
- `datetime()` with no argument is refused there. The statement resolves its
  instant more than once, and the clock moves between readings. Pass the moment
  as a `$param`.

Every binding's `valid_at` writes the same prefix:

- Python `KnowledgeGraph.cypher`, `Session.cypher` / `execute`,
  `Transaction.cypher` and `FrozenGraph.cypher`.
- The MCP `cypher_query`, `run_recipe_query` and named recipe tools.
- Java `ValidAt` on `query` / `queryResult` / `queryBatch`.

```cypher
FOR VALID_TIME AS OF date('2010-06-30')
MATCH (f:Project)-[l:HAS_HOLDER]->(c:Company) RETURN f.name, c.name
```

- **What it sees.** The statement answers as if the graph held only the elements
  valid at the instant.
  - A node passes when it is valid under every declared label it carries
    (primary or secondary), so `MATCH (n)` and `MATCH (n:Project)` agree.
  - A relationship passes when it is valid under the declaration keyed on its
    own source node's type (else its type's unkeyed one) **and** both its
    endpoints pass, named or not.
  - Undeclared labels and types are timeless.
  - A NULL or missing bound is open. A bound that is not a date, a datetime or
    an ISO string raises, naming the element.
  - An id seek (`{id: …}`) finds the version valid at the instant when several
    version nodes share the id, whichever numeric kind each stores it as. When
    more than one is valid, it takes the last in the type's node order, in every
    storage mode. A `.kgl` reload orders a type's nodes by slot, so after a
    deleted slot is reused that need not be the one created last. The seek
    reads no other id's bounds.
  - An inline-map value that is an expression (`{id: COUNT { (:Well) }}`) is
    evaluated under the context too.
- **Execution.** The instant is resolved once per execution. A `$param` or
  `date()` is read when the statement runs, so a cached plan never carries one.
  - These keep their fast routes under a context: counts (`count(*)` over a
    label, a type or the whole graph), node scans with an aggregate or
    `ORDER BY … LIMIT`, top-k over matched rows, retrieval top-k, `elementId`
    anchors, and the fused per-group aggregates over a one-hop pattern. The
    fused operators filter through the context's masks.
  - A longer pattern runs unfused, unless it is a chain of distinct types that
    the chain count operators take.
  - `PROFILE` runs the same way.
  - `EXPLAIN` leads the plan with a `ValidTimeContext` row naming the axis and
    the declared intervals the query can reach. The instant is resolved per
    execution, not planned.
- **Timeless exit.** When every declared type of the graph is valid in full at
  the instant ("as of today" on a graph that holds only current rows), the
  statement runs the plan of its text without the prefix, with every fast route,
  and returns the same rows. This is decided per execution, in memory and mapped
  storage. Disk storage keeps no endpoint index, so there the statement always
  runs guarded.
- **Paths and subqueries.**
  - A variable-length relationship (`-[:R*1..3]->`) crosses only valid
    relationships and valid intermediate nodes, so a path through an invalid
    node or relationship is not a match. A bound list (`-[rs:R*]->`) holds only
    valid relationships.
  - `shortestPath` and `allShortestPaths` search the valid elements only. The
    answer is the shortest valid route, which may be longer than the unguarded
    one, and a parallel relationship that is not valid is never on it.
  - `OPTIONAL MATCH` pads NULLs when every match is invalid.
  - `EXISTS { }`, `COUNT { }` and pattern comprehensions (`[(f)-->(c) | c.name]`)
    see only valid matches.
- **Retrieval.**
  - `text_bm25()` ranks with the statistics of the documents valid at the
    instant: their count, mean length and each term's document frequency. A
    document scores what it would in an index of the valid documents alone, and
    `ORDER BY text_bm25(…) DESC LIMIT k` keeps its index route.
  - A relationship document is valid when it and both its endpoints are, under
    every declared label they carry, whichever labels the statement names. Every
    statement at the instant therefore scores with the same statistics.
  - `vector_score(n, …)` top-k over `MATCH (n:Label)` ranks only the valid nodes'
    vectors, by one of two routes.
    - The exact route scores every valid vector (`diagnostics.retrieval`:
      `actual_mode: "exact"`, `fallback_reason: "exact_mask"`).
    - With a built HNSW index, a filtered search walks the index through every
      vector but returns only valid ones (`actual_mode: "hnsw_mask"`). It is one
      search with no over-fetch, and an invalid vector never comes back even
      when it is the nearest.
    - The search runs when the valid count weighted by the valid share
      (`valid² / stored`) reaches 1,500. `KGLITE_TEMPORAL_VECTOR_EXACT_MAX`
      replaces the threshold, and `1` sends every query with an index to the
      search. A selective instant is cheaper to score exactly, since the search's
      cost grows as the valid share shrinks.
    - A search gives way to the exact route in two cases:
      - It passes its step budget: fewer valid vectors than the index's
        `ef_search`, or valid vectors clustered away from the query
        (`fallback_reason: "exact_mask_visit_limit"`).
      - It walks every vector it can reach and finds fewer than `k` valid ones,
        some valid vector lying where the index's links do not lead
        (`"exact_mask_unreached"`).
  - `db.node_embeddings.query` / `db.relationship_embeddings.query` (and the
    `db.embeddings.query` router) rank only valid nodes, or valid relationships
    with both endpoints valid. They report `search_method: "exact_mask"` or
    `"hnsw_mask"`.
- **Graph algorithms.** `pagerank`, `betweenness`, `degree`, `closeness`,
  `louvain`, `leiden`, `label_propagation`, `connected_components`, `k_core`,
  `ready_set`, `clustering_coefficient`, `triangle_count`, `eccentricity` and
  `diameter` run on the valid slice: a copy of the elements valid at the
  instant, built once per instant and cached until the graph changes.
  - Their `{node_type, where}` scope applies within it.
  - `YIELD node` binds the graph's own node, so
    `CALL pagerank() YIELD node MATCH (node)-->(x)` reads the full graph as of
    the instant.
  - The statement is refused when it runs, naming the cap, in three cases: a
    slice over 128 MiB (`KGLITE_TEMPORAL_SLICE_MAX_BYTES`); in disk storage, a
    slice over 2,000,000 nodes and relationships
    (`KGLITE_TEMPORAL_DISK_SLICE_MAX_ELEMENTS`); or, in disk storage, a graph
    whose one-bit-per-element mask would pass 64 MiB
    (`KGLITE_TEMPORAL_DISK_MASK_MAX_BYTES`).
  - Disk storage evaluates the bounds of every element once per instant into
    that mask, for retrieval and the algorithms alike. Over the mask cap, the
    `vector_score` top-k route steps aside and the guarded match answers,
    testing only the nodes it reaches. The embedding query procedures and the
    algorithms are refused.
- **The echo.** The result's diagnostics carry a `temporal` object: Python
  `diagnostics["temporal"]`, the C / Java diagnostics JSON, the MCP `temporal:`
  line, Bolt's `kglite.temporal` summary key. Its fields:

  | Field | Meaning |
  |---|---|
  | `axis` | The temporal axis. |
  | `source` | `explicit`, `default`, `all`, or `skipped:write` / `skipped:procedure` / `skipped:valid_at` (see below). |
  | `instant` | The instant resolved (ISO; a datetime in naive UTC; `all` under `FOR VALID_TIME ALL`). |
  | `targets` | The declared labels the filter judges for the statement's patterns across every scope, without their bounds (`(:Well)`, `[:HOLDER]`, `[:HOLDER from :Project]`). It includes labels widened in through secondary labels: a node must be valid under every declared label it carries, so a `(:A)` pattern lists `(:B)` too when nodes it reaches can carry a declared `B`. A statement over types nothing declared can sit on lists none and runs unfiltered. |
  | `hidden` | Per target, how many rows it governs that are not valid at the instant, by that target's own bounds. A target answered by property guards has no entry. |
  | `endpoint_invalid` | Relationships valid by their own bounds but hidden because an endpoint node is not valid at the instant, in no `hidden` entry. Null when a target is answered by property guards. |
  | `route` | `guarded`; `plain` when the timeless exit ran; `view` through a `freeze(valid_at=…)` handle. |
  | `retrieval` | `exact_mask` / `hnsw_mask` for a `vector_score` top-k, else null. |
  | `slice` | A graph algorithm ran on the valid slice. |
  | `session_version` | The graph version answered. It is comparable within one process only, as it restarts on load. |

  A statement on a graph with no validity declaration has no `temporal` object
  in the serialized forms (`None` under the Python key). A statement that
  reaches no declared target reports route `plain`.
- **The default and `ALL`.** On a graph with a validity declaration, a statement
  with no prefix runs as `FOR VALID_TIME AS OF date()` in every binding. The
  engine adds the context when it lowers the statement, and the day is resolved
  per execution.
  - `FOR VALID_TIME ALL` reads every version: no filter, no write refusal. It is
    a no-op on a graph with no declaration, and refused inside a
    `freeze(valid_at=…)` view like any other context.
  - `valid_at='all'` on the bindings writes it, and `valid_at=None` means the
    default.
  - A graph with no declaration is untouched.
  - The runtime setting `set_valid_time_default('today' | 'all' | date)` (MCP
    `--valid-time-default`, manifest `extensions.valid_time.default`, Bolt and
    CLI flags) changes the instant the default reads.
- **Skip rules.** The default does not apply, and the statement reads every
  version, in three cases. Each is reported as `source: skipped:<reason>`:
  - A statement that writes (`skipped:write`). An explicit prefix on a write is
    still refused.
  - A procedure that is not valid-time aware, such as `refresh_stats`,
    `duplicate_id` and the `*_violation` audits (`skipped:procedure`). An
    explicit prefix still refuses it.
  - A statement that calls `valid_at()` / `valid_during()` itself
    (`skipped:valid_at`), so a query that chooses its own instants keeps its
    answer.

  Questions that join versions that never coexist, such as lineage over a
  successor relationship or a comparison of two instants, therefore run with
  `valid_at(x, d)` on the elements that need it, or under
  `FOR VALID_TIME ALL`. Under a context, the default one included, a hop is
  visible only when both its ends are valid at the one instant.
- **One context per statement.** A second prefix, or one inside a UNION arm or a
  `CALL { }` body, is a syntax error.
- **Refused** (under an explicit `AS OF`; `ALL` accepts the writing statement and
  the graph with no declaration):
  - An axis other than `VALID_TIME` (it parses, so a client can probe for
    support).
  - A graph with no validity declaration.
  - A writing statement.
  - Any other procedure that enumerates graph elements: the validation rules
    (`orphan_node`, …), `cluster`, `kg_knn`. Metadata procedures such as
    `db.labels()` and `db.temporal.declarations()` are fine.
  - `degree()` / `inDegree()` / `outDegree()` / `shortest_path_length()`. They
    read a node's relationships outside the pattern matcher. `COUNT { (n)--() }`
    and `MATCH p = shortestPath(…) RETURN length(p)` answer under a context.
    Under the default the message names both that and `FOR VALID_TIME ALL`, and
    the functions run under `ALL`.
  - A relationship type whose declarations are `ambiguous`, when the statement
    can reach it. The embedding query procedures and the algorithms reach every
    type, so any ambiguous type refuses them.

### Duration semantics

A `Duration` value carries three independent components:

| Component | Source                              | Units              |
|-----------|-------------------------------------|--------------------|
| `months`  | `years` + `months` from constructor | calendar months    |
| `days`    | `weeks` + `days` from constructor   | clock days         |
| `seconds` | `hours` + `minutes` + `seconds`     | clock seconds      |

Each constructor value is a whole number of its unit: `duration({seconds: 90.0})`
is accepted, and `duration({seconds: 1.5})` is refused. There is no sub-second
precision.

**Components stay separate by design.** Calendar arithmetic
(`+ duration({months: 1})`) is fundamentally different from clock arithmetic
(`+ duration({days: 30})`) because months have variable length.
`duration({months: 1, days: 5}).months` returns `1`, not `35`.

This matches Neo4j and openCypher. It diverges from Postgres `interval`, which
collapses everything into a single combined value. Users coming from Postgres
will need to know.

#### `duration.between(d1, d2)`

`duration.between` computes the difference between two date or timestamp values.
`months` is always 0. Whole days are stored in `days`, and any remaining sub-day
difference is stored in `seconds`.

```cypher
RETURN duration.between(date('2024-08-12'), date('2026-05-02')).days
// → 628

RETURN duration.between(date('2024-08-12'), date('2026-05-02')).months
// → 0   (NOT 20 — `between` only fills in `days`)
```

#### Composite accessors

`d.years = d.months / 12`, `d.minutes = d.seconds / 60` and
`d.hours = d.seconds / 3600`. These are integer-truncated convenience views on
the underlying components, derived and not stored.

```cypher
WITH duration({months: 26, days: 100}) AS d
RETURN d.months AS m, d.years AS y, d.days AS days
// → m=26, y=2 (26/12 truncated), days=100
```

#### `DateTime ± Duration`

A date or datetime plus or minus a duration applies the components in order:
calendar months (`years` + `months`), then days, then (for a datetime) the clock
seconds.

- A month shift keeps the day of the month and clamps to the last day of a
  shorter target month, exactly as `add_months()` / `add_years()` do (and as
  Neo4j does).
- A date has no time of day, so a duration's `seconds` do not move it.
- A result outside the representable calendar is `null`.

```cypher
RETURN date('2015-06-15') - duration({months: 11})   // → 2014-07-15
RETURN date('2024-01-15') + duration({months: 1})    // → 2024-02-15
RETURN date('2024-01-31') + duration({months: 1})    // → 2024-02-29 (clamped)
RETURN date('2016-02-29') + duration({years: 1})     // → 2017-02-28 (clamped)
RETURN date('2024-01-30') + duration({months: 1, days: 2}) // → 2024-03-02 (Feb 29, then +2 days)
RETURN date('2024-01-15') + duration({days: 30})     // → 2024-02-14
RETURN datetime('2024-01-31T10:30:00') + duration({months: 1, hours: 2}) // → 2024-02-29T12:30:00
```

Before 0.19.3 every month counted as 30 days, so
`date('2024-01-15') + duration({months: 1})` was `2024-02-14`. Use
`duration({days: 30})` for a fixed 30-day offset.

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

> **Divergence: undefined arithmetic returns null.** KGLite's Float64 value can
> represent NaN and infinities (time-series channels use NaN as a missing
> sentinel), but these math expressions do not produce a non-finite result. They
> return `null` rather than Neo4j's `NaN` / `Infinity`:
>
> - `sqrt(-1)`, `log(0)`, `log(-1)` and `log10(-1)` are `null`.
> - So is every float division or modulo by zero (`1.0 / 0.0`, `-1.0 / 0.0`,
>   `0.0 / 0.0`, `1.0 % 0.0`).
> - Null then propagates through the rest of the expression and through
>   comparisons, exactly like any other null.
> - *Integer* division by zero is the one case that raises instead
>   (`RETURN 1 / 0` → "Integer division by zero"), matching Neo4j.
>
> Guard with `coalesce(...)` or a `WHERE x > 0` filter where a number is
> required.

### Trigonometric Functions

All take a numeric argument and return a Float64. Angles are in radians (use
`radians(x)` / `degrees(x)` to convert). NULL in gives NULL out, and a
non-numeric argument also yields NULL.

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
| `split(str, delim)` | Split string into list; an empty `delim` splits into characters |
| `replace(str, search, repl)` | Replace all occurrences of `search` with `repl` |
| `substring(str, start [, len])` | Extract substring (0-indexed) |
| `left(str, n)` | First `n` characters |
| `right(str, n)` | Last `n` characters |
| `trim(str)` | Remove leading/trailing whitespace |
| `ltrim(str)` / `rtrim(str)` | Left/right trim |
| `reverse(str)` | Reverse a string |

> **Auto-coercion:** String functions accept non-string values (DateTime,
> numbers, booleans) and auto-convert them to strings. For example,
> `substring(date('2020-06-15'), 0, 4)` returns `"2020"`.

> **Characters, not bytes.** Every string function here is indexed by character,
> and so are `size()` / `length()`. `size('Tromsø')` is `6`, so
> `substring('Tromsø', size('Tromsø') - 1)` is `'ø'`.
>
> That holds for **every** string, including one delimited by brackets:
> `size('[1,2,3]')` is `7` and `size('[redacted]')` is `10`, matching Neo4j's
> `size(STRING)`. Only the argument's *type* decides: pass a real list and you
> get its element count. KGLite's other list-coercions on strings (`UNWIND`,
> indexing, `head`/`last`/`reverse`, `IN`) are unchanged and still read
> `'[1,2,3]'` as a three-element list.

> **Divergence: `split` with an empty delimiter.** openCypher does not define
> this case. KGLite splits into characters: `split('abc', '')` is
> `['a', 'b', 'c']`. An empty *original* stays `['']` for any delimiter, so
> `split('', '')` and `split('', ',')` both return `['']`.

```python
graph.cypher("RETURN split('a,b,c', ',') AS parts")         # ["a", "b", "c"]
graph.cypher("RETURN split('abc', '') AS chars")            # ["a", "b", "c"]
graph.cypher("RETURN replace('hello world', 'world', 'cypher') AS s")  # "hello cypher"
graph.cypher("RETURN substring('hello', 1, 3) AS s")        # "ell"
graph.cypher("RETURN left('hello', 2) AS l, right('hello', 2) AS r")  # "he", "lo"
```

## Text Predicates

These are lexical similarity and fuzzy-match primitives. They are useful for
deduplication, alias matching, and free-text indexing without dropping to
Python.

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

**Integer arithmetic is checked.** `+ - * / %` and unary `-` on two integers
raise `CypherExecutionError` when the result leaves the signed 64-bit range.
Integer division or modulo **by zero raises** rather than returning NULL.
Integer division truncates toward zero (`-7 / 2 → -3`) and only promotes to
float when an operand is a float. Float division by zero stays NULL. See
[Magnitude/error policy](#temporal-functions).

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

Node properties can be **native lists**, not just scalars. When a pandas column
passed to `add_nodes` holds Python lists, it is ingested as a real list
property. It is auto-detected, or forced with `column_types={'col': 'list'}`,
and stored structurally rather than stringified. List properties behave like
list literals in every list operation:

```python
# aliases is a list property, e.g. ['Bob', 'Bobby']
graph.cypher("MATCH (n:Person) WHERE 'Bobby' IN n.aliases RETURN n.name")  # membership
graph.cypher("MATCH (n:Person) UNWIND n.aliases AS a RETURN n.name, a")    # explode
graph.cypher("MATCH (n:Person) RETURN size(n.aliases) AS n_aliases")       # length
```

`IN` over a list property is true *membership*: `'Bob' IN ['Bobby']` is false
(no substring matching).

The list operators take a list.

- `x IN null`, `null[0]`, `null[0..2]`, `head(null)` and `last(null)` are `null`.
- Over any other value that is not a list (`'a' IN 'abc'`, `'abc'[0]`,
  `5[0..2]`, `head('abc')`), they are an error.
- So is `size()` of a value that is neither a list nor a string, and `length()`
  of one that is not a list, string or path.

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

Over `null` a comprehension is `null`. Over any other value that is not a list
(`[x IN 'abc' | x]`, `[x IN 5 | x]`) it is an error, not an empty list. The same
holds for the quantifiers and `reduce` below. (`UNWIND` of a non-list value is
that one value as one row.)

## Pattern Comprehensions

`[pattern WHERE predicate | expression]` gives one element per match of a
relationship pattern, evaluated for the current row:

```python
# The names of each person's friends
graph.cypher("MATCH (p:Person) RETURN p.name, [(p)-[:KNOWS]->(f) | f.name] AS friends")

# Filter the matches, read relationship properties
graph.cypher("""
    MATCH (p:Person)
    RETURN p.name, [(p)-[k:KNOWS]->(f) WHERE k.since < 2020 | f.name + ' ' + k.since] AS old
""")

# Degree, usable in WHERE / WITH / ORDER BY like any expression
graph.cypher("MATCH (p:Person) WHERE size([(p)--() | 1]) > 3 RETURN p.name")
```

Variables the row already binds (`p` above) are the row's values. The ones the
pattern introduces (`f`, `k`) are visible only inside the brackets.

- The pattern needs at least one relationship, and the `| expression`
  projection is required.
- A person with no match gets `[]`, as does a row whose correlated variable is
  `null`.
- The order of the elements is the match order, which is not specified.
- Naming the path binds it for each match, so `length(p)`, `nodes(p)` and
  `relationships(p)` work inside the brackets:
  `[p = (a)-[:KNOWS*1..2]->(f) | length(p)]`.

## List Quantifier Predicates

`any(x IN list WHERE pred)`, `all(...)`, `none(...)` and `single(...)` test list
elements against a predicate:

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

They work in WHERE, RETURN, and WITH clauses.

## Reduce (List Fold)

`reduce(acc = init, x IN list | body)` folds a list with an accumulator. The
body is evaluated once per element with `acc` and `x` bound, and the final
accumulator value is returned.

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
value. An object becomes a map and an array a list, with scalars typed as int /
float / bool / string. Invalid JSON or a non-string argument returns `null`
(never an error). This lets you predicate over data that is *stored* as a JSON
string rather than as graph structure.

A one-key object `{"$date": "2020-01-01"}`,
`{"$datetime": "2020-01-01T10:00:00+02:00"}` or
`{"$duration": {"months": 0, "days": 1, "seconds": 0}}` parses to that date,
datetime or duration, as query parameters do. With a malformed payload it stays
a map.

The code graph keeps `Function.parameters`, `Class.fields`, and
`Function.signature` as JSON (property columns hold scalars only), so
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

Combine it with `any` / `all` / list comprehensions to filter or project the
parsed elements.

## List Slicing

`expr[start..end]` slices lists with optional start/end bounds and negative
indices:

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

`n {.prop1, .prop2, alias: expr}` selects specific properties from a node:

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

`{key: expr, key2: expr}` constructs map objects in RETURN, WITH, or anywhere an
expression is valid:

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

A `$name` the query references and `params` does not supply is an error
(`Missing parameter: $name`), wherever it is written, the inline property map
included. It is never read as "matches nothing".

### Dynamic labels and relationship types

A parameter can also supply a **label or relationship type**, in both the bare
and the Neo4j 5 parenthesised spelling. `$label` and `$(label)` are the same
reference:

```python
graph.cypher("MATCH (n:$label) RETURN n.name", params={'label': 'Person'})
graph.cypher("MATCH (n:$(label)) RETURN n.name", params={'label': 'Person'})
graph.cypher("MATCH (a)-[:$type]->(b) RETURN b.name", params={'type': 'KNOWS'})

graph.cypher("CREATE (n:$label {id: 1})", params={'label': 'Person'})
graph.cypher("MATCH (n:Person {id: 1}) SET n:$label", params={'label': 'Employee'})
graph.cypher("MATCH (n:Person {id: 1}) REMOVE n:$label", params={'label': 'Employee'})
graph.cypher("MATCH (n) WHERE n:$label RETURN n.name", params={'label': 'Robot'})
```

It works anywhere a name is written:

- `MATCH`, `MERGE`, `CREATE`, `SET`, `REMOVE`.
- The `WHERE n:Label` predicate.
- Secondary labels (`(n:Person:$label)`).
- Type alternations (`-[:KNOWS|$type]->`).
- Inside `EXISTS { … }` / `COUNT { … }` / `CALL { … }`.

**Why it matters: there is now no position left that a caller has to escape.**
Building a query from user input used to mean splicing the label into the query
*text*, and every such caller owned an injection surface. As a parameter, the
value is a **name by construction**. It is bound into an already-parsed query,
so no spelling of it (backticks, braces, a closing parenthesis, another label)
can change the query's shape. A value that does not name an existing label
simply matches nothing, exactly as the literal spelling would:

```python
# Matches nothing. Not a syntax error, and not a different query.
graph.cypher("MATCH (n:$label) RETURN n",
             params={'label': "Person) RETURN n MATCH (m:Person"})
```

The parameter is bound before the query is planned, so a dynamic label plans and
performs exactly like the literal one. `EXPLAIN` of both is identical, and index
selection is unaffected.

Two limits, both deliberate:

- `$(...)` takes a **parameter name**, not a general expression. `$(row.label)`
  is a syntax error.
- The value must be a **string**. Neo4j's list forms (one parameter expanding
  into several labels, or into a type alternation) are rejected with an error
  rather than expanded. A missing parameter is an error too: a label has no
  sensible default, and matching nothing would hide the caller's bug.

## UNWIND

`UNWIND` expands a list into rows:

```python
graph.cypher("UNWIND [1, 2, 3] AS x RETURN x, x * 2 AS doubled")
```

`UNWIND` over a parameter is also the bulk-write route. `UNWIND $rows AS row
CREATE (…)` writes thousands of nodes in one statement, paying the per-statement
parse, plan, checkpoint and version-bump cost once instead of once per row.

## LOAD CSV

`LOAD CSV` reads a delimited file and pipes its rows through the rest of the
query, in the spelling other Cypher databases use, so a ported load script runs
unedited.

```
LOAD CSV [WITH HEADERS] FROM <source> AS <variable> [FIELDTERMINATOR <sep>]
```

```python
graph.cypher("""
    LOAD CSV WITH HEADERS FROM 'file:///data/people.csv' AS row
    CREATE (:Person {id: toInteger(row.id), name: row.name})
""")
```

**Row binding.**

- `WITH HEADERS` binds each record as a **map** keyed by the header row
  (`row.name`).
- Without it, each record binds as a zero-indexed **list** (`row[0]`), and the
  first line is data, not a header.
- Fields are always **strings**. CSV carries no types, and inferring them would
  corrupt leading-zero identifiers, so convert explicitly with
  `toInteger(row.n)` / `toFloat(row.x)`.
- An **empty field is `null`**, and a short row nulls its missing columns rather
  than failing the load.

**Position.** `LOAD CSV` must be the **first clause**. It is a row source, not a
transform, and the executor drives everything after it (see below).

**Sources.** `file://` URLs and plain local filesystem paths are supported.
`http(s)://` is **rejected with a message, never a syntax error**, because the
engine ships no HTTP client (network dependencies were removed in 0.14.x). Fetch
the file first, or download and parse it in your own code and pass the rows in
as a parameter. `FROM $path` works, so the location can be a parameter.

**Reading local files is a capability the caller is granted, not a given:**

| Caller | Default | Override |
|---|---|---|
| Python API, Rust library, CLI (in-process) | Allowed — the caller already has the host process's filesystem access | — |
| `kglite-bolt-server` (remote clients) | **Denied** | `--allow-csv-import <DIR>` confines imports to `DIR` after symlink resolution, so `..` segments and symlinks cannot escape it |
| `kglite-mcp-server` | **Denied** | none |

Without this gate, any client that could open a Bolt connection could run
`LOAD CSV FROM 'file:///etc/passwd'`. Server-mode graph databases generally gate
CSV import the same way: an allowed import directory, off by default.

### Memory: what streams and what does not

`LOAD CSV` reads the file in **1000-row batches** and runs the clauses that
follow once per batch, so peak memory does not scale with file size. Measured on
a row-local pipeline, a 5 MB and a 109 MB input both cost ~20 MB of resident
memory.

Batching is only equivalent to a single whole-file pass when every following
clause is **row-local**:

- `MATCH`, `WHERE`/`FILTER`, `UNWIND`, `CREATE`/`INSERT`, `MERGE`, `SET`.
- The delete/remove forms, `FOREACH`, and ordinary procedures.
- Non-aggregating `WITH`/`RETURN` (plus terminal `FINISH`).

That covers the ingest shape the clause exists for, and it streams at any file
size.

A clause that reasons over the **whole result** cannot be batched without
changing the answer (`RETURN count(*)` would report one count per batch). Such
clauses are any aggregate, `ORDER BY`, `SKIP`/`OFFSET`, `LIMIT`, `DISTINCT`, a
set operation, `cluster()`, or a `CALL` subquery. Those queries read the file
into a single pass instead, capped at **1,000,000 rows**. Past that, the query
fails naming the clause that forced it, rather than exhausting memory.
Restructure to a row-local pipeline, or aggregate outside the query.

**Not supported:** `CALL { ... } IN TRANSACTIONS` (and the older
`USING PERIODIC COMMIT`). Batching here is automatic and internal, so there is
no commit-interval to declare. A whole `LOAD CSV` statement commits once.

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

Set operators dedupe by the projected column values. Column names must match
between sides (positional fallback when they don't).
Unaliased items match on their internal form, so `toInteger('3')` and
`tointeger('3')` are the same column; the result uses the left side's spelling.

## Variable Binding in MATCH Patterns

Variables from `WITH` or `UNWIND` can be used as values in inline pattern
properties:

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

Open-ended forms (`*`, `*N..`) default the upper bound to **10 hops** as a
runaway-query guard. This is an intentional divergence from openCypher's
unbounded `*` (recorded as `pattern.var_length_default_cap` in the dialect
manifest).

- An explicit lower bound above 10 (`*11..`) raises the ceiling to that bound.
- A range whose minimum exceeds its maximum (`*5..2`) is a parse error.
- Spell out `*1..N` when you need more than 10 hops. A query with an open-ended
  form carries a warning saying so in `ResultView.warnings`.
- `shortestPath()` and `allShortestPaths()` are not capped (see below).

### The relationship variable is a list

A variable on a variable-length relationship binds the **list of relationships**
the segment walked, in walk order (openCypher semantics). Bind the path when you
need its nodes:

```python
# r is a list: size, indexing, comprehensions and quantifiers all apply
graph.cypher("""
    MATCH (a:Person {name: 'Alice'})-[r:KNOWS*1..3]->(b:Person)
    WHERE all(k IN r WHERE k.since >= 2015)
    RETURN b.name, size(r) AS hops, [k IN r | k.since] AS years
""")

# the nodes along the way come from a path variable
graph.cypher("MATCH p = (a:Person {name: 'Alice'})-[:KNOWS*1..3]->(b) RETURN nodes(p), relationships(p)")
```

Releases up to 0.18.0 bound `r` as a path map (`{nodes, relationships}`), so
`size(r)` was null and `all(k IN r WHERE …)` was always true. Code that read the
map's fields should bind `p = …` and use `nodes(p)` / `relationships(p)`.

### Trail semantics

A variable-length segment walks **trails**: no relationship may be used twice
within one `MATCH` clause. That rule spans the whole clause, so a sibling edge in
the same clause cannot re-bind a relationship the segment already walked. This
is openCypher's rule, and it is what makes an unbounded-looking pattern
terminate on a cyclic graph.

Two consequences worth knowing:

- **A node can be its own endpoint.** On a triangle `a-b-c-a`,
  `MATCH (a {id: 1})-[:KNOWS*1..3]-(b) RETURN DISTINCT b.id` returns `1, 2, 3`.
  The closed trail `a→b→c→a` uses three distinct relationships, so `a` is
  legitimately reachable from itself. It is *not* returned at `*1..2`, because
  stepping back along the edge you arrived on would reuse that relationship.
- **A trail is not a walk.** Engines that answer these patterns with a
  distance-style BFS (and SQL recursive CTEs written the obvious way) compute a
  different relation. A cross-engine comparison of a variable-length query can
  therefore disagree by a few nodes without either side being broken.

### What depth costs

Reachability shapes run one breadth-first pass over the reachable subgraph:
`count(DISTINCT …)`, `RETURN DISTINCT`, and `EXISTS { … }` over a segment whose
minimum is 0 or 1. Their cost flattens as soon as the frontier saturates, and
`EXISTS` is depth-independent (it stops at the first witness).

`count(*)`, and every shape whose minimum hop count is 2 or more, are per-*path*
by the trail rule above. They grow with branching to the power of the depth. The
guide's [How deep traversal behaves](https://kglite.readthedocs.io/en/latest/python/guides/cypher.html#how-deep-traversal-behaves)
section carries the measured curves.

When a pattern does explode, the **10,000,000-row backstop that applies when no
`max_work_units` is set** is charged against the expansion as it runs, not only
against the finished result. An open-ended traversal is therefore refused while
it expands rather than after it has exhausted memory. The error names which
expansion overflowed (the `MATCH` itself, a comma-pattern join, an
`OPTIONAL MATCH`, an `EXISTS { … }` or a `COUNT { … }` subquery) and both ways
to raise the ceiling.

An aggregate that streams its first `MATCH` (`streaming=True`, the default) is
the exception. The backstop bounds the matches one slice holds, not the rows it
passes through, so a 12-million-path aggregate answers streamed and raises with
`streaming=False`.

## WHERE EXISTS

`EXISTS` checks for subpattern existence. Brace `{ }`, parenthesis `(( ))`, and
inline pattern syntax are all supported:

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
> `WHERE n.prop IS NOT NULL` / `WHERE n.prop IS NULL` instead. Those are
> property-existence checks, while `EXISTS { ... }` and `EXISTS((...))` are
> *pattern*-existence checks. Writing `exists(n.prop)` returns a parser error
> that points at the `IS NOT NULL` alternative.

## shortestPath()

`shortestPath` finds the BFS shortest path between two nodes. It supports
directed (`->`) and undirected (`-`) syntax:

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

**Pattern shape and bounds.**

- The pattern is two nodes joined by exactly one relationship. A longer chain
  such as `(a)-[:R]->(m)-[:R*]->(b)` is a syntax error.
- A written maximum bounds the search: `*..3` and `*1..3` find no path longer
  than 3 hops, and a relationship without `*` is one hop.
- An open form (`*`, `*1..`) is unbounded here, unlike the 10-hop cap on a
  variable-length `MATCH`.
- The search visits each node at most once, and the query deadline still
  applies.
- The minimum must be 0 or 1. `*0..` admits the zero-length path from a node to
  itself, and `*2..5` is a syntax error, as in Neo4j.

An endpoint that an earlier clause bound is used as is. This covers a `MATCH`
binding and a node value from `WITH`, `UNWIND`, `startNode(r)` or a parameter.
The search runs once per input row, and a NULL endpoint yields no row.

**Path functions:**

- `length(p)` returns the hop count.
- `nodes(p)` returns full node values.
- `relationships(p)` returns full relationship values, in path order.
- Use `type(r)` when you need only each relationship type: `r.type` returns a
  stored `type` property when the relationship has one.

### Weighted shortest path

The fluent `shortest_path()` accepts an optional `weight_property` that flips
the search from BFS (hop count) to Dijkstra (sum of edge weights).

- Edges missing the property fall back to weight 1.0.
- Negative weights cause the path to be reported as missing.

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

This uses the same Louvain plumbing: `weight_property=None` falls back to BFS.

## Procedure calls and incoming rows

An ordinary `CALL procedure(...) YIELD ...` is a row operator. Parameter
expressions are evaluated against each incoming row, the procedure runs once for
that row, and each yielded row is inner-joined with the outer bindings.

- Multiple procedure rows multiply the outer row. No procedure rows drop it.
- A leading `CALL` receives one implicit seed row.
- An empty stream later in a query stays empty and does not invoke the
  procedure.

```python
graph.cypher("""
    UNWIND ['Person', 'Company'] AS kind
    CALL db.property_stats({node_type: kind, property: 'name'})
    YIELD value_count
    RETURN kind, value_count ORDER BY kind
""")
```

Outer bindings remain available after `CALL`. A yielded name that would
overwrite one is rejected. `cluster()` is the explicit exception: it consumes the
complete cohort bound by the preceding `MATCH`, because that row set is the data
being clustered.

## `CALL { ... }` read subqueries

A `CALL` subquery runs its read pipeline once for every incoming row, including
when it imports no variables. A leading subquery receives one implicit seed row,
and a later empty outer stream stays empty.

- Each row produced by the body is inner-joined with its own outer row. A body
  returning *k* rows emits *k* joined rows, and `k = 0` drops that outer row.
- An aggregating body such as `RETURN count(*)` still returns one row for an
  empty match, and therefore keeps the outer row with a zero.

```python
# Modern named scope: p is imported and remains visible throughout the body.
graph.cypher("""
    MATCH (p:Person)
    CALL (p) {
        MATCH (p)-[:KNOWS]->(f)
        RETURN count(f) AS friend_count
    }
    RETURN p.name AS name, friend_count
""")

# Empty scope imports nothing, but the body still executes once per company.
graph.cypher("""
    MATCH (c:Company)
    CALL () { MATCH (p:Person) RETURN count(p) AS people }
    RETURN c.name AS company, people
""")
```

### Scope forms

| Form | Imported outer variables | Scope inside the body |
|---|---|---|
| `CALL (p, q) { ... }` | The named variables | Global: imports survive later `WITH` clauses and enter every set-operation arm |
| `CALL (*) { ... }` | Every variable currently in scope | Global, as above |
| `CALL () { ... }` | None | No outer names are visible; the body still runs per input row |
| `CALL { WITH p, q ... }` | Bare names in the first `WITH` | Legacy import: each `UNION`/set arm must declare its own importing `WITH` |
| `CALL { ... }` without an importing `WITH` | None | Legacy no-import form; runs per input row like `CALL ()` |

Scope lists and importing `WITH` clauses accept only bare names:

- Modern scope lists accept bare, distinct variable names only. Aliases and
  expressions are rejected.
- Legacy importing `WITH` likewise accepts only bare variables: not aliases,
  expressions, aggregation, or a `WHERE` in the importing position.
- With modern scope, imported names stay visible after ordinary `WITH`
  projection or aggregation.
- With legacy scope, ordinary Cypher `WITH` scoping applies after the importing
  clause.

Unimported outer names are never auto-correlated. A name used in a pattern
inside an empty/no-import body is a fresh pattern variable, not the outer
binding.

### Set operations inside a subquery

Read subqueries support `UNION` and `UNION ALL`.

- Every arm starts from the same outer seed.
- A modern scope applies to all arms. A legacy import must be repeated at the
  start of each arm.
- All arms must return the same column names in the same order.
- KGLite also supports `INTERSECT` and `EXCEPT` in a read subquery as dialect
  extensions.

```python
graph.cypher("""
    MATCH (p:Person)
    CALL (p) {
        RETURN p.name AS value
        UNION ALL
        RETURN p.name + '!' AS value
    }
    RETURN value ORDER BY value
""")
```

### Result scope and restrictions

Only columns named by each arm's terminal `RETURN` leave the body. Internal
variables do not leak, and a returned name that collides with an outer binding
is rejected even when the outer stream is empty.

The body must be a read pipeline ending in `RETURN`. These are not supported:

- Writes.
- Unit subqueries (no terminal `RETURN`).
- `CALL { ... } IN TRANSACTIONS`.

Put writes in a top-level clause. Use the documented `LOAD CSV` batching or an
explicit transaction when you need a write batch.

## Schema Introspection (`CALL db.*`)

These Neo4j-compatible schema procedures discover what is in the graph without
leaving Cypher. Bolt clients (cypher-shell, Neo4j Browser, the Python `neo4j`
driver) call them to populate their type palettes and sidebars. `SHOW PROCEDURES`
feeds autocomplete.

| Procedure | YIELD columns | Returns |
|-----------|---------------|---------|
| `CALL db.labels()` | `label` | One row per node-type ("label") in the graph, sorted alphabetically |
| `CALL db.relationshipTypes()` | `relationshipType` | One row per connection-type ("relationship type") in the graph, sorted alphabetically |
| `CALL db.indexes()` | `name`, `type`, `entityType`, `labelsOrTypes`, `properties`, `state`, `stale`, `delta`, `unembedded` | One row per index installed on the graph, sorted by `name`. `stale`/`delta` are non-null on the opt-in kinds (`FULLTEXT`, `VECTOR`); `unembedded` on `VECTOR` alone. `SHOW INDEXES` returns the same rows — see [Cypher index DDL](#cypher-index-ddl) |
| `CALL db.constraints()` | `name`, `type`, `entityType`, `labelsOrTypes`, `properties`, `propertyType` | One row per declared constraint, sorted by `name`. `SHOW CONSTRAINTS` returns the same rows — see [Cypher constraint DDL](#cypher-constraint-ddl) |
| `CALL db.propertyKeys()` | `propertyKey` | One row per declared property name (node + relationship), sorted alphabetically |
| `CALL db.schema()` | `nodeType`, `properties` | One row per node-type with its sorted list of property names — the in-language counterpart of Python `describe()` |
| `CALL db.schema.visualization()` | `nodes`, `relationships` | One row: virtual nodes (one per label; `name`/`indexes`/`constraints` properties) and virtual relationships per observed (source label, type, target label) combination — what Neo4j Browser's schema tab renders |

Procedure names are case-insensitive on dispatch. Neo4j convention preserves
camelCase in docs: `db.relationshipTypes`, not `db.relationship_types`. YIELD
columns are case-sensitive. The other `db.*` family is the change stream; see
[Change data capture](#change-data-capture-call-dbcdc).

**Standalone CALL.** `YIELD` is optional when the `CALL` is the entire
statement. `CALL db.labels()` returns every declared column in declared order,
which is the form Neo4j clients and cypher-shell send. Combined with any other
clause, `YIELD` is required. Result columns always follow YIELD order
(alias-or-name), and a call that yields zero rows still reports its declared
columns.

**`YIELD *`.** `CALL db.temporal.declarations() YIELD * RETURN *` yields every
declared column of the procedure, in declared order. It is legal anywhere a
`YIELD` list is (followed by `WHERE`, `WITH`, `RETURN *`). It stands alone:
mixing it with named columns (`YIELD *, kind`) or aliasing it is an error, and an
unknown procedure keeps its usual error.

**`SHOW PROCEDURES [YIELD …]`** lists every procedure from the same registry
`CALL list_procedures()` reads. Neo4j's default columns are `name`,
`description`, `mode`, `worksOnSystem`. `signature` is yieldable.

**`SHOW FUNCTIONS [YIELD …]`** lists every scalar and aggregate function from a
registry whose every entry is test-verified to dispatch. The default columns are
`name`, `category`, `description`. `signature` and `aliases` are yieldable. IDEs
use both listings for autocomplete.

**Bolt server only:** `CALL dbms.components()`, `CALL dbms.showCurrentUser()`,
and `SHOW DATABASES` are answered by `kglite-bolt-server`. They report server
facts (identity, configured user, served database) and are not available
in-process. See the
[Bolt server guide](https://kglite.readthedocs.io/en/latest/operators/bolt-server.html).

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
| `name` | `"<NodeType>.<property>"` (equality / range / text) or `"<NodeType>.(p1,p2,...)"` (composite); `"relationship:<Type>.<property>"` for a relationship vector or BM25 index |
| `type` | `"PROPERTY"` for equality + composite indexes; `"RANGE"` for B-tree range indexes; `"FULLTEXT"` for a BM25 text index built by `build_text_index()`; `"VECTOR"` for an HNSW index built by `build_vector_index()` |
| `entityType` | `"NODE"`, or `"RELATIONSHIP"` for a relationship vector or BM25 index |
| `labelsOrTypes` | `[node_type]` — single-element list |
| `properties` | `[property]` for equality/range/text/vector; `[p1, p2, ...]` for composite |
| `state` | `"ONLINE"`, or `"DEFERRED"` on a graph loaded with `kglite.load(path, defer_index_rebuild=True)` — the index is declared but not yet built, and any write builds it. There is no `POPULATING` in between: a KGLite index is built atomically |

**KGLite extension.** Neo4j collapses equality and range under a single
`type = "PROPERTY"`. KGLite distinguishes range indexes (`type = "RANGE"`)
because the planner uses the distinction: an equality index can't serve a range
query. Index advisors and tooling that branch on `type` get the information they
need without parsing the `name` string.

- A BM25 text index reports Neo4j's `"FULLTEXT"`, which is what it is for a
  client reading the column. KGLite's is single-label and single-property, so
  `CREATE FULLTEXT INDEX` still refuses. Build one with
  `build_text_index(node_type, property)`.
- A built HNSW index reports `"VECTOR"` and is named for the *source column* it
  was built over, not the `_emb` store key. A type carrying both a text and a
  vector index over `body` therefore lists two rows under the same name and
  different types.
- Embedding stores with no index built over them are not indexes. They are
  reported by `list_embeddings()` instead.

### Cross-reference with the Python API

`db.indexes()` is the procedure form of the Python
`KnowledgeGraph.list_indexes()` method. Both pull from the same introspection
helper, so output stays in sync. Use `db.indexes()` from a Bolt client or inside
a Cypher pipeline. Use `list_indexes()` from Python code where you would rather
have a Python list of dicts than a `cypher()` result.

## Procedure reference

This table lists every procedure, its `YIELD` columns in declared order, and a
one-line description. It is generated from the procedure registry
(`CALL list_procedures()` and `SHOW PROCEDURES` read the same table).
Parameters and examples are in the sections above and in
`describe(cypher=['<topic>'])`.

<!-- BEGIN GENERATED: procedure-reference (scripts/render_docs_facts.py) -->

| Procedure | YIELD columns | Description |
|---|---|---|
| `pagerank` | `node`, `score` | Compute PageRank centrality for all nodes |
| `betweenness` | `node`, `score` | Compute betweenness centrality for all nodes |
| `degree` | `node`, `score` | Compute degree centrality for all nodes |
| `closeness` | `node`, `score` | Compute closeness centrality for all nodes |
| `louvain` | `node`, `community`, `level` | Detect communities using multilevel Louvain (hierarchical). |
| `leiden` | `node`, `community`, `level` | Detect communities using Leiden (multilevel, well-connected communities). |
| `label_propagation` | `node`, `community` | Detect communities using label propagation |
| `connected_components` | `node`, `component` | Find weakly connected components. |
| `k_core` | `node`, `coreness` | k-core decomposition (coreness per node). |
| `ready_set` | `node`, `dependency_count` | Dependency frontier: nodes whose {edge} prerequisites all satisfy the `done` predicate — the next actionable work items. |
| `clustering_coefficient` | `node`, `coefficient` | Local clustering coefficient per node (how interconnected its neighbours are). |
| `triangle_count` | `triangles`, `transitivity` | Global triangle count + transitivity (global clustering coefficient) for the whole graph. |
| `eccentricity` | `node`, `eccentricity` | Per-node eccentricity (longest shortest path to any node in its component). |
| `diameter` | `diameter` | Graph diameter (max eccentricity). |
| `cluster` | `node`, `cluster` | Cluster nodes by spatial location or numeric properties (DBSCAN/K-means). |
| `orphan_node` | `node` | Rule: nodes of {type} with zero matching edges (default: any edge, both directions). |
| `self_loop` | `node` | Rule: nodes of {type} with a self-loop via {edge} |
| `cycle_2step` | `node_a`, `node_b` | Rule: a-{edge}->b-{edge}->a pairs where both nodes are of {type} |
| `missing_required_edge` | `node`, `rule` | Rule: nodes of {type} with no outgoing edge of {edge} (direction-validated); no-arg form checks every ontology declaration |
| `missing_inbound_edge` | `node` | Rule: nodes of {type} with no incoming edge of {edge} (direction-validated) |
| `duplicate_title` | `node` | Rule: nodes of {type} whose title is shared with another node of the same type |
| `duplicate_id` | `node` | Rule: nodes of {type} whose id is shared with another node of the same type |
| `null_property` | `node` | Rule: nodes of {type} where {property} is missing, null, or empty |
| `outline` | `node`, `depth`, `parent_id`, `node_type`, `node_id_type`, `parent_type`, `parent_id_type`, `node_token`, `parent_token` | Projection: BFS spanning tree from node id {root} and optional {root_type} along {edge} — typed ids plus result-local identity tokens (render with kglite.outline) |
| `inverse_violation` | `a`, `b`, `rule` | Rule: (a)-[rel_a]->(b) without a matching (b)-[rel_b]->(a); no-arg form checks every ontology declaration |
| `transitivity_violation` | `a`, `b`, `c`, `rule` | Rule: (a)->(b)->(c) chains under {rel} where the direct (a)->(c) edge is absent; no-arg form checks every ontology declaration |
| `cardinality_violation` | `node`, `count`, `rule` | Rule: nodes of {type} whose outgoing-{edge} count is outside [min, max]; no-arg form checks every ontology declaration |
| `type_domain_violation` | `source`, `target`, `rule` | Rule: edges of {edge} whose source node is not of {expected_source} type; no-arg form checks every ontology declaration (a declared class widens to its descendants) |
| `type_range_violation` | `source`, `target`, `rule` | Rule: edges of {edge} whose target node is not of {expected_target} type; no-arg form checks every ontology declaration (a declared class widens to its descendants) |
| `table.upsert` | `action`, `rows` | Atomic keyed row upsert into a list-of-maps property: replaces the first row whose {key} cell matches row[key], else appends. |
| `table.delete` | `removed`, `rows` | Atomic keyed row delete from a list-of-maps property: removes every row whose {key} cell equals {value}. |
| `db.node_embeddings.set` | `stored`, `dimension` | Atomically upsert vectors for explicitly selected nodes (entries: [{node: n, vector: [...]}]) |
| `db.node_embeddings.embed` | `embedded`, `skipped`, `dimension`, `model` | Generate vectors for explicitly selected nodes (nodes: collect(n)) with the registered embedder |
| `db.node_embeddings.remove` | `removed` | Remove vectors from explicitly selected nodes |
| `db.node_embeddings.drop` | `dropped` | Drop one node embedding store |
| `db.node_embeddings.list` | `entity`, `type`, `text_column`, `store`, `dimension`, `count`, `metric`, `model`, `index_state`, `delta`, `unembedded` | List node embedding stores and their metadata |
| `db.node_embeddings.build_index` | `indexed`, `metric`, `m` | Build an HNSW index for one node embedding store |
| `db.node_embeddings.refresh_index` | `refreshed` | Refresh a node vector index from current stored vectors |
| `db.node_embeddings.drop_index` | `dropped` | Drop one node vector index while retaining its vectors |
| `db.node_embeddings.query` | `node`, `score`, `search_method`, `type` | Retrieve the nearest nodes from one or several whole embedding stores (type, types, or every store for text_column), merged into one top-k, by query vector or by text the registered embedder embeds |
| `db.ontology.declare` | `declared`, `warnings` | Declare the ontology (classes, relationships, closed_labels, enforcement): {ontology: <map or JSON string>} or the document's own keys. |
| `db.ontology.show` | `ontology`, `locked`, `enforcement` | The declared ontology as the canonical document (a map that db.ontology.declare({ontology: ...}) accepts back unchanged; Null when nothing is declared), locked (the operator's --ontology lock) and the store's default enforcement |
| `db.ontology.clear` | `cleared` | Remove the declared ontology entirely (managed labels are withdrawn). |
| `db.temporal.declare` | `declared`, `rows`, `abutting_rows` | Declare which two properties (from, to) bound a node label's or relationship type's validity interval, and whether the to day belongs to it (convention 'closed') or ends it ('half_open'), with an optional empty_when: 'to_before_from' under 'closed' that keeps a row whose to is the day before its from as an empty interval; validates every stored bound and counts rows whose end meets another row's start. |
| `db.temporal.undeclare` | `undeclared` | Remove one validity-interval declaration (node, or relationship with optional source_type). |
| `db.temporal.declarations` | `kind`, `name`, `source_type`, `from`, `to`, `convention`, `empty_when`, `abutting_rows`, `ambiguous`, `empty_rows`, `unreadable_rows` | List validity-interval declarations: kind, name, source type, bound properties, convention, empty_when, abutting rows counted at declare time, whether the relationship type is ambiguous (several unkeyed declarations from an older graph; re-declare them per source_type), and the rows a write since left with an empty interval (valid at no instant) or an unreadable bound, counted now |
| `db.node_text_index.build` | `indexed`, `skipped`, `terms` | Build (or rebuild) a BM25 text index over one node type's string property (text_column), for text_bm25(n, text_column, query) |
| `db.node_text_index.refresh` | `refreshed` | Fold every node change since the last build or refresh into a node text index |
| `db.node_text_index.drop` | `dropped` | Drop one node text index |
| `db.node_text_index.list` | `entity`, `type`, `text_column`, `documents`, `terms`, `skipped`, `index_state`, `delta`, `auto_refresh_limit` | List node text indexes and their freshness |
| `db.embeddings.set` | `stored`, `dimension` | Router: entity:'node' (default) runs db.node_embeddings.set, entity:'relationship' runs db.relationship_embeddings.set |
| `db.embeddings.embed` | `embedded`, `skipped`, `dimension`, `model` | Router: entity:'node' (default) runs db.node_embeddings.embed, entity:'relationship' runs db.relationship_embeddings.embed |
| `db.embeddings.remove` | `removed` | Router: entity:'node' (default) runs db.node_embeddings.remove, entity:'relationship' runs db.relationship_embeddings.remove |
| `db.embeddings.drop` | `dropped` | Router: entity:'node' (default) runs db.node_embeddings.drop, entity:'relationship' runs db.relationship_embeddings.drop |
| `db.embeddings.list` | `entity`, `type`, `text_column`, `store`, `dimension`, `count`, `metric`, `model`, `index_state`, `delta`, `unembedded` | Router: entity:'node' (default) runs db.node_embeddings.list, entity:'relationship' runs db.relationship_embeddings.list |
| `db.embeddings.build_index` | `indexed`, `metric`, `m` | Router: entity:'node' (default) runs db.node_embeddings.build_index, entity:'relationship' runs db.relationship_embeddings.build_index |
| `db.embeddings.refresh_index` | `refreshed` | Router: entity:'node' (default) runs db.node_embeddings.refresh_index, entity:'relationship' runs db.relationship_embeddings.refresh_index |
| `db.embeddings.drop_index` | `dropped` | Router: entity:'node' (default) runs db.node_embeddings.drop_index, entity:'relationship' runs db.relationship_embeddings.drop_index |
| `db.embeddings.query` | `node`, `score`, `search_method`, `type` | Router: entity:'node' (default) runs db.node_embeddings.query, entity:'relationship' runs db.relationship_embeddings.query (the relationship route yields relationship in place of node) |
| `db.text_index.build` | `indexed`, `skipped`, `terms` | Router: entity:'node' (default) runs db.node_text_index.build, entity:'relationship' runs db.relationship_text_index.build |
| `db.text_index.refresh` | `refreshed` | Router: entity:'node' (default) runs db.node_text_index.refresh, entity:'relationship' runs db.relationship_text_index.refresh |
| `db.text_index.drop` | `dropped` | Router: entity:'node' (default) runs db.node_text_index.drop, entity:'relationship' runs db.relationship_text_index.drop |
| `db.text_index.list` | `entity`, `type`, `text_column`, `documents`, `terms`, `skipped`, `index_state`, `delta`, `auto_refresh_limit` | Router: entity:'node' (default) runs db.node_text_index.list, entity:'relationship' runs db.relationship_text_index.list |
| `db.relationship_embeddings.set` | `stored`, `dimension` | Atomically upsert vectors for explicitly selected relationships |
| `db.relationship_embeddings.embed` | `embedded`, `skipped`, `dimension`, `model` | Generate vectors for explicitly selected relationships |
| `db.relationship_embeddings.remove` | `removed` | Remove vectors from explicitly selected relationships |
| `db.relationship_embeddings.drop` | `dropped` | Drop one relationship embedding store |
| `db.relationship_embeddings.list` | `entity`, `type`, `text_column`, `store`, `dimension`, `count`, `metric`, `model`, `index_state`, `delta`, `unembedded` | List declared relationship embedding stores and their metadata |
| `db.relationship_embeddings.build_index` | `indexed`, `metric`, `m` | Build an HNSW index for one relationship embedding store |
| `db.relationship_embeddings.refresh_index` | `refreshed` | Refresh a relationship vector index from current stored vectors |
| `db.relationship_embeddings.drop_index` | `dropped` | Drop one relationship vector index while retaining its vectors |
| `db.relationship_embeddings.query` | `relationship`, `score`, `search_method`, `type` | Retrieve the nearest relationships from one or several whole embedding stores (type, types, or every store for text_column), merged into one top-k, by query vector or by text the registered embedder embeds |
| `db.relationship_text_index.build` | `indexed`, `skipped`, `terms` | Build (or rebuild) a BM25 text index over one relationship type's string property (text_column), for text_bm25(r, text_column, query) |
| `db.relationship_text_index.refresh` | `refreshed` | Fold every relationship change since the last build or refresh into a relationship text index |
| `db.relationship_text_index.drop` | `dropped` | Drop one relationship text index |
| `db.relationship_text_index.list` | `entity`, `type`, `text_column`, `documents`, `terms`, `skipped`, `index_state`, `delta`, `auto_refresh_limit` | List relationship text indexes and their freshness |
| `ontology_audit` | `entity_kind`, `rule`, `severity`, `violations`, `exempted`, `total`, `pct`, `domain_class`, `property` | Scorecard: one row per declared node/edge ontology check, identified by entity_kind plus rule (violations, exempted, total, pct, declared severity). |
| `node_property_violation` | `class`, `check`, `node`, `property`, `properties` | Class property violations, including declared descendants by primary type. |
| `edge_property_violation` | `relationship`, `check`, `source`, `target`, `property`, `properties`, `exempt` | Rule: edges flagged by a declaration's required_properties (property absent/null) or property_types (present value of the wrong type) — the row listing behind those two ontology_audit counts. |
| `parallel_edges` | `a`, `b`, `count` | Rule: (a, b) pairs connected by more than one edge of {edge} |
| `kg_knn` | `node`, `distance_m` | Spatial: k nearest nodes of {target_type} to ({lat}, {lon}) |
| `affected_tests` | `test_file`, `depth` | Code graphs: test files reachable from changed files via inbound IMPORTS edges. |
| `rev_diff` | `bucket`, `type`, `qualified_name`, `name`, `file`, `line` | Multi-rev code graphs: added/removed/changed code entities between two revs {from, to}. |
| `dead_code` | `node` | Functions with no inbound use edge (CALLS / REFERENCES_FN / HANDLES / IMPLEMENTED_BY / DECORATES); excludes tests, dunder and main (pass exclude_public to also drop pub/exported, include_tests to keep tests) |
| `refresh_stats` | `src_type`, `edge_type`, `tgt_type`, `count` | Recompute the label-pair edge-count cache; one row per (src_type, edge_type, tgt_type) with its fresh count. |
| `list_procedures` | `name`, `description`, `yield_columns` | List all available procedures |
| `db.labels` | `label` | All node-type names ('labels') in the graph, sorted |
| `db.relationshipTypes` | `relationshipType` | All connection-type names ('relationship types') in the graph, sorted |
| `db.indexes` | `name`, `type`, `entityType`, `labelsOrTypes`, `properties`, `state`, `stale`, `delta`, `unembedded` | All indexes in the graph (equality, composite, range, text, vector), sorted by name |
| `db.constraints` | `name`, `type`, `entityType`, `labelsOrTypes`, `properties`, `propertyType` | All declared constraints (UNIQUENESS, NODE_KEY, NODE_PROPERTY_EXISTENCE, NODE_PROPERTY_TYPE, RELATIONSHIP_PROPERTY_EXISTENCE, RELATIONSHIP_PROPERTY_TYPE), sorted by name |
| `db.propertyKeys` | `propertyKey` | All property keys declared in the graph (node + relationship), sorted |
| `db.schema` | `nodeType`, `properties` | One row per node type with its sorted property-name list |
| `db.schema.visualization` | `nodes`, `relationships` | Schema graph for visualization: one row with virtual nodes (one per label; properties name/indexes/constraints) and virtual relationships (one per observed source-label/type/target-label combination). |
| `db.schema.nodeTypeProperties` | `nodeType`, `nodeLabels`, `propertyName`, `propertyTypes`, `mandatory` | Typed node schema: one row per (label, property) with propertyTypes and mandatory — the shape Neo4j clients load their data model from. |
| `db.schema.relTypeProperties` | `relType`, `propertyName`, `propertyTypes`, `mandatory` | Typed relationship schema: one row per (type, property) with propertyTypes and mandatory. |
| `apoc.meta.nodeTypeProperties` | `nodeType`, `nodeLabels`, `propertyName`, `propertyTypes`, `mandatory`, `propertyObservations`, `totalObservations` | APOC-compatibility shim over db.schema.nodeTypeProperties: same typed node schema under APOC's column set (adds totalObservations/propertyObservations). |
| `apoc.meta.relTypeProperties` | `relType`, `sourceNodeLabels`, `targetNodeLabels`, `propertyName`, `propertyTypes`, `mandatory`, `propertyObservations`, `totalObservations` | APOC-compatibility shim over db.schema.relTypeProperties, adding the endpoint columns (sourceNodeLabels/targetNodeLabels) the db.schema contract lacks - one row per observed (source, type, target) pairing. |
| `db.graph_stats` | `node_count`, `edge_count`, `label_count`, `relationship_type_count` | Per-graph summary: node, edge, label, and relationship-type counts. |
| `db.property_stats` | `value_count`, `null_count`, `distinct_count` | Per-(label, property) statistics: value, null, and distinct counts. |
| `db.property_uniqueness` | `is_unique`, `violation_count`, `distinct_count` | Uniqueness pre-flight for a (label, property): is it unique, and how many violations. |
| `db.cdc.enable` | `enabled`, `epoch`, `capacity`, `enrichment`, `cursor` | Start change data capture on this graph, or reconfigure a running log in place. |
| `db.cdc.disable` | `enabled`, `wasEnabled` | Stop change data capture and discard the log. |
| `db.cdc.status` | `enabled`, `epoch`, `capacity`, `enrichment`, `buffered`, `earliest`, `current` | How change data capture is configured on this graph, and how much it is holding. |
| `db.cdc.current` | `id` | Cursor addressing the newest published change — the position to start from to see only future changes |
| `db.cdc.earliest` | `id` | Cursor addressing the oldest change still retained — the position to resync from after a cursor expires |
| `db.cdc.query` | `id`, `seq`, `operation`, `elementType`, `nodeType`, `nodeId`, `relationshipType`, `srcType`, `srcId`, `tgtType`, `tgtId`, `state` | Changes published after a cursor, oldest first. |

<!-- END GENERATED: procedure-reference -->

## Change data capture (`CALL db.cdc.*`)

Change data capture is an opt-in stream of the changes the graph publishes, read
from Cypher. Every binding (Python, the CLI shell, a Bolt client, the C ABI) gets
it without a method of its own. Capture is **off until you ask for it**, and
nothing that happened before `enable` is in the log.

| Procedure | YIELD columns | Does |
|-----------|---------------|------|
| `CALL db.cdc.enable({capacity, enrichment})` | `enabled`, `epoch`, `capacity`, `enrichment`, `cursor` | Start capturing (or reconfigure a running log in place). Both arguments are optional and **every omitted one takes its default**: `capacity` is the retention in events (default 65536), `enrichment` is `'off'` (default) or `'full'`. `cursor` is the position to start consuming from |
| `CALL db.cdc.disable()` | `enabled`, `wasEnabled` | Stop capturing and discard the log. Idempotent; `wasEnabled` says whether it had been running |
| `CALL db.cdc.status()` | `enabled`, `epoch`, `capacity`, `enrichment`, `buffered`, `earliest`, `current` | How capture is configured and how much it is holding. The one read verb that **answers while capture is off** (`enabled: false`, every other column null) rather than failing |
| `CALL db.cdc.current()` | `id` | Cursor addressing the newest published change — start here to see only what happens *next* |
| `CALL db.cdc.earliest()` | `id` | Cursor addressing the oldest change still retained — start here to read the whole retained log |
| `CALL db.cdc.query({from, selectors, maxRows})` | `id`, `seq`, `operation`, `elementType`, `nodeType`, `nodeId`, `relationshipType`, `srcType`, `srcId`, `tgtType`, `tgtId`, `state` | Changes published *after* the `from` cursor, oldest first. Omit `from` to read everything retained. `selectors` filters (below); `maxRows` caps the rows returned |

```python
graph.cypher("CALL db.cdc.enable()")

# Consume from "now": remember the cursor, poll, remember again.
cursor = graph.cypher("CALL db.cdc.current()").to_dicts()[0]["id"]

graph.cypher("CREATE (:Person {id: 1, name: 'ann'})")
graph.cypher("MATCH (p:Person {id: 1}) SET p.name = 'anne'")

rows = graph.cypher(
    "CALL db.cdc.query({from: $c}) YIELD id, operation, nodeType, nodeId, state",
    params={"c": cursor},
).to_dicts()
for row in rows:
    print(row["operation"], row["nodeType"], row["nodeId"], row["state"])
# create Person 1 {'after': {'labels': [], 'properties': {'name': 'ann'}, 'title': 'ann'}, 'before': None}
# update Person 1 {'after': {'labels': [], 'properties': {'name': 'anne'}, 'title': 'anne'}, 'before': None}

cursor = rows[-1]["id"]  # the row's own id is the cursor for the next poll
```

**Rows.** `operation` is `create`, `update` or `delete`. `elementType` is `node`
or `relationship`.

- A node row carries `(nodeType, nodeId)` and null relationship columns.
- A relationship row carries `(relationshipType, srcType, srcId, tgtType, tgtId)`
  and null node columns.

These are logical identity, not an internal index, so a cursor's events stay
meaningful after a compaction.

**`state` is the pair `{before, after}`** (Neo4j's shape), and it is *always* a
map, so a consumer reads `state["after"]` without null-checking the container
first. Each half is the entity's image on that side of the commit, or `null`
where the log holds none:

- **`after`**: `{title, labels, properties}` for a node, `{properties}` for a
  relationship, and `null` for a delete, which left no entity behind.
- **`before`**: the same shape, under `enrichment: 'full'`, being what the
  commit found.
  - It is `null` for a create (there was nothing before it).
  - It is `null` in every row under the default `enrichment: 'off'`, which
    captures no pre-image.
  - A delete's `before` is the state it destroyed, the one event whose only
    informative half is this one.

Each half is null rather than an empty map on purpose: an empty map reads as "an
entity with no properties", which is a different fact.

**Enrichment.** `enable({enrichment: 'off' | 'full'})` selects how much state
each event carries. `'off'` (after-image only) is the default. `'full'` adds the
before-image.

- The argument is named after Neo4j's `txLogEnrichment` database option and takes
  two of its three values.
- `'diff'` is **refused by name**. A diff is computed *from* the full
  before-image, so it would save ring bytes rather than capture work, at the cost
  of a second event semantics for consumers to handle. A consumer that wants a
  diff computes one from a `'full'` event, where both sides are present.
- Changing the mode on a running log **keeps the epoch**, exactly as a resize
  does, so live cursors survive it.
- The change applies to *writes*, not to the log. The events already in the ring
  keep the shape they were captured with, so a consumer that switches to `'full'`
  starts seeing `before` from the next commit rather than retroactively.

**`before` is the state at the start of the commit**, not before the most recent
write. Three writes to one entity in one transaction publish one event, whose
`before` is what the transaction opened on and whose `after` is what it left. The
pair therefore answers "what did this commit change", which is the question a
mirror or an audit trail is asking. A label change is included on the node's
side: `before.labels` is the set the commit replaced.

### Selectors — filtering the stream

`selectors` is a **list of filter maps**, and an event is returned if **any one
of them** matches it. One query therefore serves "every delete, plus every change
to a `Person`". Within a single map every key must hold, so
`{nodeType: 'Person', operation: 'update'}` is "updates of Person".

```cypher
CALL db.cdc.query({
  from: $cursor,
  selectors: [{operation: 'delete'}, {nodeType: 'Person', operation: 'update'}],
  maxRows: 100
}) YIELD operation, nodeType, nodeId, state
```

| Key | Matches when | Example |
|-----|--------------|---------|
| `elementType` | the row's `elementType` equals it — `'node'` or `'relationship'` | `{elementType: 'relationship'}` |
| `operation` | the row's `operation` equals it — `'create'`, `'update'` or `'delete'` | `{operation: 'create'}` |
| `nodeType` | a **node** row whose `nodeType` equals it | `{nodeType: 'Person'}` |
| `relationshipType` | a **relationship** row whose type equals it | `{relationshipType: 'KNOWS'}` |
| `srcType` / `tgtType` | a relationship row whose endpoint type equals it (directional) | `{srcType: 'Person', tgtType: 'Company'}` |
| `nodeId` | a node row whose `nodeId` equals it | `{nodeId: 42}` |
| `srcId` / `tgtId` | a relationship row whose endpoint id equals it | `{srcId: 1, tgtId: 2}` |
| `labels` | a node carrying **all** the listed secondary labels | `{labels: ['Archived', 'Cold']}` |
| `changesTo` | **any** listed property differs across the commit — needs `enrichment: 'full'` | `{changesTo: ['email', 'phone']}` |

**The vocabulary is the columns'.** `operation` takes `create`/`update`/`delete`
and `elementType` takes `node`/`relationship`. These are the same strings the
rows report, *not* Neo4j's single-letter `c`/`u`/`d`. A selector that spelled a
concept differently from the column it filters would be a trap.

**`labels` is a conjunction, over secondary labels only.**
`{labels: ['Archived', 'Cold']}` selects nodes carrying both. Use two selectors
for "either". The primary type is not in this set, because that is `nodeType`'s
job, which keeps the key aligned with what `state.after.labels` reports. A
relationship has no labels, so a `labels` constraint never matches one.

**`changesTo` compares the two images**, so it requires `enrichment: 'full'`. On
an `'off'` log it is **refused**, naming the fix, rather than silently matching
every event that merely *has* the property.

- An absent image and an absent key read alike ("not there"), which makes the
  rule total across operations: a create matches a property it set, a delete one
  it had, an update one whose value moved.
- One caveat on a log switched from `'off'` to `'full'` mid-stream: events
  captured before the switch have no before-image, so every property they carry
  reads as newly set.

**Validation is strict, one level in.**

- An unknown key *inside* a selector map (`{nodeTyp: 'Person'}`) is refused
  rather than ignored. A silently-dropped constraint would widen the query to
  everything while looking like a filter.
- Wrong-typed values are refused by key name.
- `selectors: []` is the *absence* of filters and returns everything.
- `selectors: [{}]` is refused, because an empty map is a filter that constrains
  nothing and is almost always a selector built from an empty set of conditions.

**Filtering happens before the copy-out**, so an event you did not ask for is
never cloned out of the ring. `maxRows` is applied **after** filtering: it caps
the rows you receive, not the window they were drawn from. (The key is `maxRows`
and not `limit` because `LIMIT` is a reserved clause word that cannot be written
as a bare map key.)

**Polling with selectors: take the cursor first.** Rows keep the `id` they would
have had unfiltered. A cursor addresses the log, not your filtered view, which
is what lets two consumers with different selectors exchange cursors. The
consequence is that a filtered poll can return **zero rows while the log has
advanced**. A consumer that advanced its cursor from "the last row I received"
would then re-scan the same events forever. Take `db.cdc.current()` *before* the
query and adopt it after:

```python
cursor = graph.cypher("CALL db.cdc.current()").to_dicts()[0]["id"]
while True:
    nxt = graph.cypher("CALL db.cdc.current()").to_dicts()[0]["id"]
    rows = graph.cypher(
        "CALL db.cdc.query({from: $c, selectors: [{nodeType: 'Person'}]}) YIELD nodeId, state",
        params={"c": cursor},
    ).to_dicts()
    handle(rows)          # may legitimately be empty
    cursor = nxt          # advance regardless
```

**Cursors are opaque and exclusive.** Pass back the string you were given rather
than constructing one, because the encoding may change. Exclusive means a cursor
names what you have *already seen*. Polling with the last row's `id` therefore
never re-delivers it, and `current()` immediately followed by `query` returns
nothing.

**A cursor addresses one log, and the log says so.** Each log carries an
*epoch*. A cursor from a different epoch is refused rather than resolved against
different data.

- A new epoch is minted by `enable` on a graph that had capture off. That
  includes after `disable`, after a `.kgl` **load** (the log is process-local
  runtime state and is deliberately never saved, so a loaded graph starts with
  capture off), and on an independent copy.
- Re-`enable` on a *running* log only reconfigures it and **keeps** the epoch,
  so live consumers survive a capacity or enrichment change.

The three refusals are distinct because the fix differs:

| Refusal | Fix |
|---|---|
| A malformed cursor | Fix the call. |
| A foreign epoch | Re-acquire a cursor. |
| A cursor older than retention | Resync and accept the gap. |

**A save records where the running epoch got to**, so the wrong-epoch refusal in
the *next* process can say more than "that epoch is gone". Present a cursor from
the epoch the file was saved under, and the message names where it ended. It
also says whether you were caught up at that point or behind by a countable
number of changes that were never delivered. It is a diagnostic only. The log
itself is never persisted, so nothing is resumable, and the remedy is still to
resync from `earliest()` or `current()`. A wrong-epoch cursor the file knows
nothing about keeps the plain refusal, because nothing is known about it to
report.

**Retention is a bounded ring.** The oldest events are evicted as it fills, and
`earliest()` advances to match. A consumer that falls further behind than the
capacity gets a typed refusal naming both remedies: resync from `earliest()`, or
raise the retention with `CALL db.cdc.enable({capacity: <larger>})`. The answer
is never silently truncated.

**A change that was not committed never appears.** Events are derived from the
same write-capture buffer the write-ahead log uses, at the same commit
boundaries.

- A statement that failed, and a transaction that was rolled back, contribute
  *no event at all*, not a filtered-out one.
- A `Transaction` publishes its whole batch at `commit()`, and nothing before it.
- A held `ResultView` forces the next write to fork copy-on-write. The fork
  shares the one log, so that write publishes exactly once.

**Capture is not free while it is on.** Every mutation buffers an op, and an
enabled graph gives up the checkpoint-free-mutation fast path, so the cost tracks
ops written rather than statements run. Measured (release profile):

| Write | Overhead with capture on |
|---|---|
| bare `CREATE` | +33% |
| `MERGE` that creates | +8% |
| `SET` by id | +3-7% |
| `MERGE` that matches and writes nothing | **0%** |
| 1000-row `add_nodes` | +52% |
| 1000 `add_connections` | +82-88% |

The default ring holds **~50-65 MB of resident memory** once full at its 65 536
events. The spread is the event payload, not measurement slack: identity-only
changes sit near the bottom, 4-property nodes near the top. A graph that never
calls `db.cdc.enable()` is untouched by any of this.

`enrichment: 'full'` adds one whole-entity read per changed entity per commit.
It is not per write, because the image is taken at each entity's first touch, and
creates read nothing at all. Measured at **+2-5%** (median +4.5%) on 1000
autocommit `SET`s, which is the worst case for it: every write is its own commit
and so its own first touch.

An update or delete event then holds two images where it held one, so a ring of a
given capacity retains more bytes. That cost tracks the changed entity's property
payload rather than a fixed multiplier: **+0.7%** of ring memory when the changes
carry nothing beyond their identity, **+32%** (~82 MB) on 4-property nodes. The
default `capacity` does not change with the mode, and is the knob to lower if
that matters.

**Storage modes.** In-memory and `storage='mapped'` serve the stream
identically, durable or not. `storage='disk'` **refuses** `enable`. A disk graph
commits by publishing an immutable generation, so the per-commit write capture
this stream is derived from does not describe its change boundary, and serving it
would report a stream that silently missed writes.

### Divergences from Neo4j's `db.cdc.*`

KGLite matches the names its model has a concept for and drops the rest. An
absent column is better than an empty one, because a consumer writes code against
it.

| Neo4j | KGLite |
|-------|--------|
| `id` | Same meaning: the opaque cursor addressing a change |
| `seq` | **Wider.** Neo4j orders events *within* a transaction; KGLite's is the log-wide sequence, monotonic across commits, and it is what the cursor carries |
| `txId` | **Absent.** KGLite publishes at commit boundaries but assigns no durable transaction identity, so any value would be invented |
| `metadata` | **Absent.** Neo4j reports executing user, connection client and transaction start time; the engine holds none of that |
| `event` (nested map) | **Flattened into columns**, so `YIELD nodeType, operation` filters in Cypher without map traversal. Nesting survives only where it carries structure: `state` |
| `state: {before, after}` | **Same shape.** `after` is the post-commit image (null for a delete); `before` is the pre-commit one under `enrichment: 'full'` (null for a create, and null throughout under the default `'off'`) |
| `selectors` | **Same idea, KGLite's vocabulary.** A list of maps, ANY-matched. Keys name the columns (`operation: 'update'`, `elementType: 'node'`) rather than Neo4j's `select: 'n'` / `operation: 'c'`, and `labels` / `changesTo` are supported. `maxRows` is a KGLite addition |
| — | `db.cdc.enable` / `db.cdc.disable` / `db.cdc.status` have no Neo4j counterpart: enablement there is a database option (`ALTER DATABASE … SET OPTION txLogEnrichment`), whose `off` / `full` values `enable`'s `enrichment` argument mirrors — `diff` is refused, with the reason |

Arguments are passed as a **map** (`{capacity: N}`, `{from: cursor}`).
Positional arguments are a parse error.

## Code-graph analysis

When the graph is a parsed codebase (built by e.g. codingest), the data needed
for the analyses other tools ship as bespoke commands is *already on the graph*.
Most are one query. The metrics are captured at parse time (`branch_count`,
`max_nesting`, `loc`), and the relationships are first-class (`CALLS`,
`REFERENCES_FN`, `USES_TYPE`, `EXTENDS`, `IMPLEMENTS`).

### `CALL dead_code(...)` — unreferenced functions

```cypher
CALL dead_code() YIELD node
RETURN node.qualified_name AS fn, node.file_path AS file
ORDER BY file
```

`dead_code` reports `Function` nodes with no inbound *use* edge. A node counts as
used when any of these holds:

- Something `CALLS` it.
- Something references it as a value (`REFERENCES_FN`).
- A route `HANDLES` it.
- A procedure is `IMPLEMENTED_BY` it.
- It takes part in `DECORATES`.

Bundling all of those is the point: a naive
`WHERE NOT (:Function)-[:CALLS]->(f)` falsely flags callbacks, route handlers
and decorated entry points.

Implicit entry points are excluded automatically: test functions, dunder methods
(`__x__`), and `main`. Options:

| Param | Default | Effect |
|---|---|---|
| `include_tests` | `false` | also report test functions |
| `exclude_public` | `false` | also drop `pub`/`public`/`export`/`exported` visibility (useful for Rust-style codebases; off by default because in Python every non-underscore name is nominally public) |

### `CALL rev_diff({from, to})` — what changed between two revisions

For a **multi-rev code graph** built with a multi-rev codingest build (one graph
holding N git revisions; see the Python guide), `rev_diff` reports the code
entities `added`, `removed`, or `changed` between any two loaded revs:

```cypher
CALL rev_diff({from: 'v1', to: 'v2'})
YIELD bucket, type, qualified_name, name, file, line
RETURN bucket, type, qualified_name, file, line
ORDER BY bucket, qualified_name
```

Each merged node carries `revs: [str]` (the revisions it appears in) and
`rev_fp: [int]` (a per-rev shape fingerprint, positionally aligned with `revs`).
`rev_diff` reads those straight off each node:

- Present at `from` only: `removed`.
- Present at `to` only: `added`.
- Present at both with a divergent `rev_fp`: `changed`.

It is a pure set-and-fingerprint membership check. It reports *that* an entity
changed (and its current/newest value via the ordinary property columns), and
never re-parses source, matching the two-graph structural-diff contract.

| Param | Required | Effect |
|---|---|---|
| `from` | yes | the baseline rev label (as passed to `build(revs=[...])`) |
| `to` | yes | the comparison rev label |
| `node_type` | no | scope to one node type (string) or several (list) |

`rev_diff` yields `bucket` (`"added"`/`"removed"`/`"changed"`), `type`,
`qualified_name`, `name`, `file`, `line`.

- It errors clearly on a graph that isn't multi-rev (no `revs` property) or an
  unknown rev (listing the available revs).
- It is nodes-only in v1. Edge add/remove is a documented deferral.
- To scope an ordinary query to a single rev, use list membership directly:
  `MATCH (n:Function) WHERE 'v2' IN n.revs RETURN n`. An *unscoped* query spans
  all revs (an over-count trap).

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

### Freshness provenance & staleness (`auto_timestamp`)

Opt a node or connection type into engine-managed freshness stamping:

```python
g.define_schema({
    "nodes":       {"Task":  {"auto_timestamp": True}},
    "connections": {"LINKS": {"source": "N", "target": "N", "auto_timestamp": True}},
})
```

> **`define_schema` merges per node/connection type.** A type the call names
> takes the new declaration entire. A type it omits keeps its own, so declaring
> per module never withdraws another type's constraints. Pass `replace=True` for
> whole-schema replacement. It warns, naming each constraint it stops enforcing.
> Constraints declared with `CREATE CONSTRAINT` are unaffected by either mode.
> Drop them with `DROP CONSTRAINT`.

Every write to an opted-in type then stamps a reserved **`updated_at`** (a
`Timestamp` in naive UTC). The writes covered are Cypher
`CREATE`/`INSERT`/`MERGE`/`SET` and `add_nodes`/`add_connections`. Pass
`git_sha` / `modified_by` to record who/where too:

```python
g.cypher("MERGE (t:Task {id: $id}) SET t.status = 'done'",
         params={"id": "T1"}, git_sha=current_sha, modified_by="coding-agent")
```

These are **metadata, not data**. They are queryable directly (`n.updated_at`,
`n.git_sha`, `r.updated_at`) but hidden from `properties(n)` / `keys(n)` /
`RETURN n` / `describe()`. The engine owns them and only stamps opted-in types,
so other writes stay deterministic.

- On an opted-in type, the stamp replaces a user-written `updated_at` (on
  `CREATE`, `SET` and the bulk loaders alike), and a user-written `git_sha` /
  `modified_by` whenever the call passes its own.
- Because the engine writes them, none of the three can carry a constraint.
  `CREATE CONSTRAINT` and `define_schema` (`required`, `types`, `primary_key`,
  `unique`, `required_properties`, `property_types`) refuse them, whether or not
  the type has opted in yet.

**Staleness is a pure query: the engine never touches the filesystem.** Have the
*writer* stamp the linked file's state (`file_path`, `file_mtime` /
`content_hash`) when it writes the node. Drift is then a check:
```cypher
-- Nodes whose linked file changed since we last described it
MATCH (n:Artifact) WHERE n.updated_at < n.file_mtime
RETURN n.id, n.file_path, n.updated_at, n.file_mtime

-- Linked-path-gone: the writer stamps a flag (it owns fs access)
MATCH (n:Artifact) WHERE n.file_missing = true RETURN n.id, n.file_path

-- "As of which commit?" — group runtime nodes by the sha they were written at
MATCH (n:Task) WHERE n.git_sha IS NOT NULL
RETURN n.git_sha, count(n) ORDER BY count(n) DESC
```

Store `file_mtime` as a `Timestamp` (or ISO string) so it compares with
`updated_at`. For low-stakes drift you want to defer, mark your own
`stale = true` property and sweep later (`MATCH (n) WHERE n.stale = true ...`).

### Edge confidence

Most edges are **extracted**: parsed facts (a `CALLS` edge is a real call site).
A few are **inferred**: best-effort heuristics, notably the cross-language
coupling edges (a client request matched to a server route by path). Inferred
edges carry `confidence = "inferred"`. Extracted edges leave the property unset.
So:

```cypher
-- facts only (exclude heuristic edges)
MATCH (a)-[r:CALLS]->(b) WHERE r.confidence IS NULL RETURN a, b

-- just the heuristic cross-language couplings
MATCH (a)-[r]->(b) WHERE r.confidence = 'inferred' RETURN type(r), a.name, b.name
```

Inheritance-resolved `CALLS` edges stay **extracted**. They are pinned via the
type graph, not guessed, so they are facts, not heuristics.

## Scoping graph algorithms to a subgraph

The centrality (`pagerank`, `degree`, `betweenness`, `closeness`) and community
(`louvain`, `leiden`, `label_propagation`) procedures accept two optional
parameters. They restrict the algorithm to a **property-filtered subgraph**, so
test / benchmark / external nodes don't pollute the result:

| Param | Meaning |
|-------|---------|
| `node_type` | string or list of node labels to include (e.g. `'Function'`) |
| `where` | a predicate over the node variable `n`, with the same expression grammar as a `WHERE` clause |

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
communities reflect the subgraph, not the whole graph filtered afterward.

- An explicit scope also lifts the large-graph refusal guard (you've bounded the
  run yourself).
- Scoping is an **in-memory-only** feature. On disk/mapped graphs the procedures
  reject `node_type` / `where`; filter with a preceding `MATCH` instead.

## Dependency frontier — `CALL ready_set(...)`

Over a DAG on a chosen edge type, `ready_set` returns the nodes whose
dependencies are all satisfied: the "ready set" of a build / scheduling / plan
graph.

- A node's **dependencies** are its outgoing-edge neighbours, so
  `(task)-[:DEPENDS_ON]->(dependency)` reads naturally. A task is ready once
  every dependency it points to is *done*.
- "Done" is a predicate over the node variable `n` (same grammar as `where`).
- A node already done is excluded.
- A root with no dependencies is ready as soon as it isn't done.

| Param | Meaning |
|-------|---------|
| `relationship` | the dependency edge type (string or list) |
| `done` | predicate over `n` marking a node satisfied, e.g. `'n.status = "done"'` |
| `node_type` | optional — limit which nodes are *emitted* (dependencies are followed regardless) |

`YIELD node, dependency_count` returns the node and how many dependencies the
ready node had, all satisfied.

> **Scope to the type you care about.** A node with *no* outgoing-`E` edge is a
> root, and is vacuously "all dependencies satisfied". An unscoped `ready_set`
> over a sparse edge type therefore also returns every unrelated node. To get
> e.g. "ready **tasks**", pass `node_type: 'Task'` so only that type is emitted
> (dependencies are still followed across types).

```python
# Which tasks can the agent pick up next?
graph.cypher("""
    CALL ready_set({relationship: 'DEPENDS_ON', done: 'n.status = "done"'})
    YIELD node, dependency_count
    RETURN node.id AS id, dependency_count AS deps ORDER BY id
""")
```

## CREATE / INSERT / SET / DELETE / REMOVE / MERGE

A write clause's value expressions charge the statement's `max_work_units`
budget as a read's projection does. The clauses covered are `SET`, `CREATE`,
`MERGE` (its pattern and `ON CREATE` / `ON MATCH SET`) and a `FOREACH` list.

- The charge is cumulative across the statement's rows (a `range()` charges its
  length).
- As in a read, an expression that is the same on every row is folded once per
  clause. It is charged once, and every row gets the one value (`datetime()`
  included; `rand()` and `randomUUID()` stay per row).
- With no budget set, the 10,000,000 backstop applies.

```python
# CREATE — returns ResultView with .stats
result = graph.cypher("CREATE (n:Person {name: 'Alice', age: 30, city: 'Oslo'})")
print(result.stats['nodes_created'])  # 1

# CREATE relationship between existing nodes
graph.cypher("""
    MATCH (a:Person {name: 'Alice'}), (b:Person {name: 'Bob'})
    CREATE (a)-[:KNOWS]->(b)
""")

# Nodes and relationships in one statement. Endpoints need no variable name,
# and a variable introduced in one comma-separated part is a *reference* in
# every later part of the same CREATE (variable scope ends with the statement).
graph.cypher("CREATE (:Person {name: 'Carol'})-[:KNOWS]->(:Person {name: 'Dan'})")
graph.cypher("""
    CREATE (a:Person {name: 'Eve'}), (b:Person {name: 'Frank'}),
           (a)-[:KNOWS]->(b), (b)-[:KNOWS]->(a)
""")

# INSERT uses static labels/types. Multiple node labels use `&`.
graph.cypher("""
    INSERT (a IS Person&Actor {id: 1}), (b:Person {id: 2}),
           (a)-[r IS KNOWS {since: 2020}]->(b)
    RETURN labels(a), type(r), r.since
""")

# SET — update properties
result = graph.cypher("MATCH (n:Person {name: 'Bob'}) SET n.age = 26, n.city = 'Stavanger'")
print(result.stats['properties_set'])  # 2

# DELETE / NODETACH DELETE error if a node has relationships; DETACH removes all
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

`INSERT` is deliberately stricter than `CREATE`. It accepts static node labels
introduced by `:` or `IS`, with `&` between multiple labels, and one static,
directed relationship type. It rejects these forms:

- Dynamic labels/types.
- A dynamic property map.
- Path assignment.
- Colon-separated multiple labels.
- Relationship type alternation.
- Undirected or variable-length relationships.

Use `CREATE` when one of those CREATE-specific forms is intentional.

## Transactions

A transaction groups multiple mutations into an atomic unit. On success, all
changes apply. On exception, nothing changes.

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

Prefix any Cypher query with `EXPLAIN` to see the query plan without executing
it. It returns a `ResultView` with columns `[step, operation, estimated_rows]`:

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
# {'step': 4, 'operation': 'OptimizerPass fuse_optional_match_count', 'estimated_rows': None}
```

Cardinality estimates use `type_indices` counts when available, `None`
otherwise. After the physical plan rows, `EXPLAIN` appends an
`OptimizerPass <name>` row for every optimizer pass that changed the plan. These
rows make it possible to verify a particular rewrite directly. Disabling that
pass removes its tag.

A `ClosureProbe :Person (Student, Teacher)` row follows the `Match` row of any
node pattern whose ontology closure the matcher can answer from the member types'
indexes instead of a scan. This applies to a materialized supertype label in the
`closed` state, where every live descendant carries an index answering the
pattern's equality properties. The row names those members.

- No row means no probe: the label is `open`, a member is unindexed, or nothing
  is materialized. The match falls back to a (still correct) scan.
- A label written in `WHERE` position rather than in the pattern is an ordinary
  post-candidate predicate and never probes.
- The row's `estimated_rows` is `Null`. The probe returns the *value's* row
  count, which no static model knows.
- A value supplied as a parameter (`{p: $v}`) leaves the row off, because the
  plan renders before the parameter binds.

## PROFILE

Prefix any Cypher query with `PROFILE` to execute AND collect per-clause
statistics. It returns a normal `ResultView` with results, plus a `.profile`
property:

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

Every `cypher()` call attaches lightweight execution diagnostics to the returned
`ResultView`. No prefix is required, and they are always on:

```python
result = graph.cypher("MATCH (n:Country {label: 'Norway'}) RETURN n.nid")
print(result.diagnostics)
# {'elapsed_ms': 3, 'timeout_ms': 180000, 'row_limit': None, 'total_rows': None, 'warnings': []}
```

Keys:

- `elapsed_ms`: wall-clock duration in milliseconds.
- `timeout_ms`: the deadline that was in effect, or `None` when no deadline
  applied. A fired deadline raises `CypherTimeoutError`, so no partial
  `ResultView` is ever returned.
- `warnings`: non-fatal advisories about the query, empty for a clean one. Each
  is legal Cypher that quietly returns nothing useful, so it is a warning and not
  an error, with a "did you mean?" hint where one is genuinely close. There are
  five families:
  - A `MATCH` against an unknown node label or relationship type (zero rows).
  - A `WHERE` on a property no node of that type has. It is a null comparison,
    so every row is filtered out. An `IS NULL` test on it is true on every row
    instead, and is worded that way. Inside a function such as `coalesce`, or
    under an `OR`, `XOR` or negated `AND`, the reference does not decide, and the
    warning names the property without claiming a row count.
  - A `RETURN` / `WITH` / `ORDER BY` reading such a property (a silently all-null
    column). The sibling `n.name` still resolves, so the rows read as
    half-correct.
  - A relationship pattern pointing the wrong way, when every edge of that type
    runs the other way (zero rows).
  - A `WHERE` comparison a **declared** property type makes vacuous: `p.age >
    'forty'` where `p.age IS :: INTEGER` is null on every row. The other side may
    be a literal, a bound `$parameter`, or a second typed property.

  Warnings are populated on reads, mutations, `EXPLAIN`, and session/transaction
  queries alike, and mirrored to stderr for interactive users.

```python
result = graph.cypher("MATCH (n:Persn) RETURN n")   # typo
result.diagnostics["warnings"]
# ["MATCH references unknown node label 'Persn' — the graph has no such
#   type, so this pattern returns no rows. Did you mean 'Person'?"]

graph.cypher("MATCH (p:Port)-[:ARRIVES_AT]->(v:Voyage) RETURN p").diagnostics["warnings"]
# ["MATCH traverses 'ARRIVES_AT' as Port → Voyage, but every 'ARRIVES_AT'
#   relationship runs Voyage → Port — this pattern matches no edges.
#   Reverse the arrow?"]

graph.cypher("MATCH (v:Vessel) RETURN v.imo").diagnostics["warnings"]
# ["RETURN projects property 'imo' which no Vessel node has — every value
#   will be null."]

graph.cypher("MATCH (p:Person) WHERE p.age > 'forty' RETURN p").diagnostics["warnings"]
# ["WHERE compares Person.age (declared INTEGER) with a STRING literal
#   'forty' — a cross-type ordering comparison is null in openCypher, so this
#   filters out every row."]

graph.cypher("MATCH (p:Person) WHERE p.age > $cutoff RETURN p",
             params={"cutoff": "forty"}).diagnostics["warnings"]
# ["WHERE compares Person.age (declared INTEGER) with a STRING parameter
#   $cutoff ('forty') — a cross-type ordering comparison is null in
#   openCypher, so this filters out every row."]
```

The family classifies **both** operands. The one being compared against the typed
property may be a literal, a `$parameter` **the call actually bound**, or a
second property whose type is also known: `WHERE p.age > p.email` names both
sides and both types. `=~` joins `STARTS WITH` / `ENDS WITH` / `CONTAINS` as a
string predicate. It answers false for every non-`STRING` value whatever the
pattern is.

Two sources of type knowledge feed it, in this order:

1. **`REQUIRE p.x IS :: T`** (below). The write path enforces it, so a declared
   `INTEGER` property cannot come to hold a string, and "this comparison is
   cross-type" is a guarantee rather than a guess.
2. **`define_schema()` field types.** These are declared intent, checked only by
   `validate_schema()`, so the claim rests on the stored data honouring the
   declaration.

Where both cover a property **the declaration wins**, the same precedence the
constraint-vs-lock error messages follow. Each message quotes the declaration it
read *in that declaration's own words*: `declared INTEGER` for the constraint,
`schema-defined integer` for the schema definition. Every other type name in a
message is the engine's own, so the casing says which half of the sentence is
yours. Observed per-type metadata (last-write-wins, and not a declaration) is
consulted by neither.

Everything the family cannot place stays silent:

- Numeric properties against numeric literals (`INTEGER`/`FLOAT` are one
  comparison family, all nine value pairings intercomparable).
- `DATE`/`LOCAL DATETIME` against a string. The string is parsed, so whether the
  answer is null depends on its contents.
- `DURATION` and `POINT` (no comparison rules of their own).
- A schema field type it does not recognise.
- An **unbound** parameter.
- A property pair with one untyped side.
- Multi-label patterns, `WITH`-rebound variables and the built-in fields.

`<>` gets its own wording, because a cross-type `<>` is *true*: it matches every
row that has the property rather than filtering them out.

A *sparse* property never warns. `node_type_metadata` records a property as soon
as one node carries it, so only an absent one trips these: a typo, a field that
belongs to a different type, or a declared name nothing has written yet (a
`define_schema()` field, or a validity declaration's `from` / `to` bound). The
declared names still warn, because the all-null result is real. There is one
exception: a bare `IS NULL` test on one (the `WHERE n.valid_to IS NULL`
current-version idiom) is the expected answer and stays silent.

**Under `lock_schema()` the two absent-property families above are errors, not
warnings.** A locked schema is the opt-in "catch my typos" mechanism, and it
already rejected `MATCH (p:Person {agee: 1})` and `MATCH (p:Persn)`. The same
typo written as `WHERE p.agee = 1` or `RETURN p.agee` now raises `SchemaError`
with the same "did you mean?" hint, rather than returning an empty result or a
null column.

The lock checks the positions these warnings come from, and no others:

- In a `WHERE` (of `MATCH`, `OPTIONAL MATCH` or `WITH`) or a `FILTER`: a property
  tested directly (a comparison, `IN`, `IS [NOT] NULL`, `STARTS WITH` /
  `ENDS WITH` / `CONTAINS`, including through arithmetic or a function argument),
  and the `AND` / `OR` / `XOR` / `NOT` combinations of those.
- A property read by a `RETURN`, `WITH` or `ORDER BY` item.

A property read in a nested or derived position is neither refused nor warned
about:

- A predicate used as a value (`(p.agee = 1) IS NULL`, `p.agee IS NULL = true`).
- A `CASE`.
- A map key or list index on it (`p.agee.k`, `p.agee[0]`).
- A list predicate (`any(v IN … WHERE p.agee IS NULL)`).
- A `WHERE` inside `EXISTS { }` / `COUNT { }`.

**The declared-type family promotes too, but only from its enforced source.** A
mismatch against an `IS :: T` constraint raises `SchemaError` under a lock. An
example is `WHERE p.age > 'forty'` where `REQUIRE p.age IS :: INTEGER` is in
force. The error carries the same sentence the warning would have, and the same
`unlock_schema()` way out. The write path guarantees that declaration, so "no row
can answer this predicate" is a fact about the stored data.

- A mismatch read from a `define_schema()` field type is **not** promoted, in
  any schema state. Nothing enforces that declaration at write time, so a lock
  has no ground to reject the query on.
- A property pair promotes only when both of its sides are declared.
- A **bound parameter** does not block the promotion. The property side is still
  enforced, and the binding's type is a fact of that call. So `p.age > $cutoff`
  with a string bound raises, while the same statement with an integer bound
  runs.

`unlock_schema()` puts every one of them back to a warning. Unknown relationship
type and reversed arrow stay warnings in both states. Every conservatism above
still applies under the lock, so all of these are left alone:

- Sparse properties.
- Properties the same statement writes.
- Types with no recorded properties.
- Fields `define_schema()` declares, or a validity declaration names as a bound,
  but nothing has written yet.
- Multi-label patterns and `WITH`-rebound variables.
- The built-ins.
- Every comparison the runtime can actually answer.

`timeout_ms` resolution, in order of precedence: explicit
`cypher(..., timeout_ms=N)`, then `kg.set_default_timeout(ms)`, then a default
of 180,000 ms. Pass `timeout_ms=0` to disable the deadline for one call.

**The default is per-surface, and each surface declares its own:**

| Surface | Default deadline |
|---|---|
| Python API | 180,000 ms |
| MCP server (`cypher_query`'s `timeout_ms` argument) | 180,000 ms |
| CLI | None. It takes `--timeout-ms` per call. |
| Bolt server | None, per the Neo4j "absent `tx_timeout` means no timeout" wire contract. |

Polling and rollback:

- Expanding, aggregation, set-operation, subquery-join and procedure loops poll
  cooperatively.
- Mutation loops poll before every row, and a write clause's value expressions
  poll as a read's do. Before its commit phase, a write overruns its deadline by
  about one row's work.
- A statement that raises `CypherTimeoutError` has changed nothing, whether it
  ran through `KnowledgeGraph.cypher()`, `Session.execute()` or a `Transaction`.
  One that finishes its last clause after the deadline is rolled back. A
  statement that runs without a rollback checkpoint (a terminal `DELETE`, a
  single-node `CREATE`) checks the deadline once more just before it writes.
- Use a `Transaction` to roll back several statements as a group.

Some phases are not interruptible, and their own duration is not bounded by the
deadline:

- The removal phase of `DELETE` / `DETACH DELETE`. It starts only if the
  statement is not yet late, then runs to completion.
- The one-shot builds a statement can trigger: deferred index materialisation,
  the disk statement checkpoint, the temporal endpoint index and the lazy disk id
  index. The deadline is checked as soon as each build finishes.

## Parallel runtime

One heavy analytical query can use the whole machine. The parallel runtime is
off by default. Opt in per call:

```python
result = graph.cypher(
    "MATCH (p:Person) WHERE p.score > 0.5 RETURN p.city AS city, count(*) AS n",
    parallel=True,
)
```

`parallel=True` is a **hint, not an instruction**. It never changes an answer:
values, row order, and group order are identical either way. Only operators that
can partition deterministically use it. Each applies its own runtime gate on
candidate count and per-row cost, so a small query stays sequential however it
is flagged, and there is no threshold to tune.

What parallelises:

- **Scan + filter.** The node scan behind `MATCH (n:Label {prop: ...})` and
  `MATCH (n:Label) WHERE ...`. Partitions are contiguous candidate ranges
  concatenated in candidate order, so the bucket order of an un-`ORDER BY`'d
  `MATCH` is unchanged.
- **Fused scan aggregates.** `MATCH (n:Label) [WHERE ...] RETURN <keys>,
  <aggregates>` over `count` / `count(DISTINCT)` / `sum` / `avg` / `min` /
  `max`. Groups are emitted first-seen, as always.
- **Grouped aggregation, across groups.** Each group's aggregate is computed
  independently. Within a group nothing changes: rows are still folded in row
  order, so `collect` returns the same list and `median` / `mode` /
  `percentile_*` keep their per-group tie-breaks.
- **`ORDER BY` sort keys.** The key computation only. The sort itself is stable
  and stays sequential, so ties keep input order.

What stays sequential, and why:

- **Row construction and projection.** Building a result row allocates, and
  allocation does not share. Partitioning these was measured *slower* than not,
  so it is gated by row count rather than by the flag.
- **The streaming aggregate.** Partitioning it would reassociate `sum` and `avg`
  over floating point and move the last bit of the result. A number that depends
  on how many cores answered the query is not a number this engine will return.
- **Anything that would change an error.** A `LIMIT` pushed into a `MATCH`, a
  `max_work_units` budget, and a group-limited aggregation all have outcomes that
  depend on where the sequential scan stopped, so they opt out.

Scope:

- **Storage modes.** Memory and mapped fan out. Disk-mode graphs, and graphs with
  a spatial configuration, ignore the flag and run sequentially rather than
  refusing it, so code that runs against all three modes is unaffected.
- **Servers.** The Bolt and MCP servers never enable it. A server's cores belong
  to its concurrent clients, and turning it on there would trade across-query
  throughput for one query's latency.
- **Surfaces.** The keyword lives on `KnowledgeGraph.cypher()`.
  `Session.cypher()`, `Transaction.cypher()` and `FrozenGraph.cypher()` run
  sequentially. There is no graph-wide default to set.
- **CLI.** `kglite query <graph> "<cypher>" --parallel`.
- **Pool width.** One shared worker pool, sized from the machine's available
  parallelism and built lazily on the first fan-out. `KGLITE_QUERY_THREADS=N`
  pins it. Anything that does not parse to a positive integer is ignored.

The runtime gates are measured crossovers, not guesses. A scan or aggregate
region fans out at **20,000 candidate rows** when every per-row predicate
compiles, and at **5,000** when one routes through the interpreter. The
projection-shaped regions fan out at 4,096 rows. The count is the real candidate
count, never a planner estimate.

These figures were measured on a 1M-node / 11M-edge synthetic graph on a 10-core
Apple Silicon machine (4 performance + 6 efficiency cores), release build,
minimum of two agreeing runs. Numbers on other hardware will differ, and a
machine with fewer performance cores will see less:

| Query shape | Sequential | Parallel | |
|---|---|---|---|
| Scan + filter + `count(*)` | 34 ms | 6.5 ms | 5.2x |
| Scan + filter + grouped aggregate | 68 ms | 13 ms | 5.2x |
| Property-filtered scan | 7.6 ms | 1.5 ms | 5.0x |
| Interpreted text predicate | 30 ms | 4.6 ms | 6.5x |
| Regex (`=~`) predicate | 41 ms | 6.9 ms | 6.0x |
| Grouped aggregation, few groups | 356 ms | 280 ms | 1.3x |
| Grouped aggregation, 800k groups | 480 ms | 440 ms | 1.1x |
| `ORDER BY` over 800k rows | 400 ms | 360 ms | 1.1x |
| Scan + filter + 792k-row projection | 132 ms | 122 ms | 1.1x |

Shapes whose cost is *scanning* gain 5-6x. Shapes whose cost is *building rows*
gain almost nothing, because the second kind is bound by the allocator rather
than by arithmetic. Reach for `parallel=True` on an analytical scan or aggregate
over a large graph. It will not speed up returning a million rows to Python.

## Indexes

Create an equality index on a `(node_type, property)` pair so
`MATCH (n:T {prop: value})` and `WHERE n.prop = value` use an indexed lookup
instead of scanning the type:

```python
graph.create_index('Country', 'label')
# {'node_type': 'Country', 'property': 'label',
#  'unique_values': 5, 'persistent': true, 'created': true}
```

Persistence depends on the storage mode:

- **`storage='disk'`.** The index is **persistent**. It is written as four
  mmap'd, SHA-256-addressed files next to the CSR
  (`property_index_v2_{digest}_{meta,keys,offsets,ids}.bin`).
  - Versioned metadata stores and validates the exact type/property identity, so
    punctuation, Unicode, underscores, and case-distinct names cannot overwrite
    one another.
  - Indexes are lazy-loaded on first query after reopen.
  - Legacy filename-based bundles remain readable by exact request.
  - No heap HashMap rebuild occurs on `load()`.
- **In-memory graphs.** The existing `property_indices` HashMap is used (no
  change).

`describe()` annotates indexed properties so agents can see which columns hit the
fast path before writing a query. A string index supports both equality and
prefix **on a `storage='disk'` graph only**. Prefix is the sorted mmap's own
capability, and the in-memory hash index annotates `eq` alone because
`STARTS WITH` full-scans there. Numeric indexes support equality only:

```xml
<prop name="label"  type="String" unique="5" indexed="eq,prefix" vals="Norway|..."/>
<prop name="year"   type="Int"    unique="20" indexed="eq"/>
```

### STARTS WITH pushdown

With a `storage='disk'` string index in place, `WHERE n.prop STARTS WITH 'x'` is
pushed into the MATCH pattern and served by the prefix side of the sorted mmap:

```python
graph.cypher("MATCH (n:Country) WHERE n.label STARTS WITH 'O' RETURN n.nid")
# O(log N + k) where k is the number of matches
```

### Cypher index DDL

Neo4j 5 index DDL runs against KGLite, so a schema-setup script ports unedited.
What it *builds* is not the same thing, because the two engines have different
indexes. Read this table before assuming a statement did what it does in Neo4j.

```cypher
CREATE [RANGE] INDEX [name] [IF NOT EXISTS] FOR (n:Label) ON (n.prop [, n.prop2 ...])
DROP INDEX <name> [IF EXISTS]
DROP INDEX FOR (n:Label) ON (n.prop [, ...])        -- KGLite extension
SHOW [ALL] INDEX[ES]
```

#### What each statement builds

KGLite has three index structures, each serving a different predicate shape.
Neo4j has one `RANGE` index that serves all of them.

| KGLite structure | Serves | Equivalent API call |
|---|---|---|
| hash equality (`property_indices`) | `=`, `IN` | `create_index(label, prop)` |
| composite hash (`composite_indices`) | conjunctive `=` on all its properties | `create_composite_index(label, [p1, p2])` |
| B-tree range (`range_indices`) | `<`, `<=`, `>`, `>=`, ordering | `create_range_index(label, prop)` |

| Statement | Builds | `indexes_added` |
|---|---|---|
| `CREATE INDEX FOR (n:L) ON (n.p)` | one hash equality index | 1 |
| `CREATE INDEX FOR (n:L) ON (n.a, n.b)` | one composite index | 1 |
| `CREATE RANGE INDEX FOR (n:L) ON (n.p)` | a hash equality index **and** a B-tree range index | 2 |

The `RANGE` form builds two structures because that is what Neo4j's single
`RANGE` index serves. The bare form deliberately builds only the equality index.
This is where KGLite parts ways with Neo4j 5, for which bare and `RANGE` are
identical. Doing both for every `CREATE INDEX` in a ported script would silently
double index memory, and in-memory footprint is this engine's product. The
divergence costs performance, never correctness. If you need range acceleration,
write `CREATE RANGE INDEX`.

#### Index names are canonical, not yours

KGLite derives index names from what they index: `Label.property`, or
`Label.(a,b)` for a composite. A name in `CREATE INDEX <name> FOR …` is
**accepted but not stored**. The persisted `.kgl` index state is a list of
`(label, property)` key tuples, so there is nowhere to keep it.

A composite index is keyed by its property names **sorted**, so declaration order
is not part of its identity. `ON (n.city, n.age)` and `ON (n.age, n.city)` are
one index, named `L.(age,city)`, and either spelling drops it.

```cypher
CREATE INDEX person_email FOR (p:Person) ON (p.email);   -- index is created
SHOW INDEXES;                                            -- name is 'Person.email'
DROP INDEX person_email;                                 -- error, explains this
DROP INDEX Person.email;                                 -- works
DROP INDEX FOR (p:Person) ON (p.email);                   -- also works
```

`DROP INDEX` accepts the dotted canonical name unquoted, so `SHOW INDEXES` output
pastes straight in.

- `DROP INDEX <unknown name> IF EXISTS` is a no-op, truthfully, since no index
  carries that name.
- The bare form errors and spells the naming rule out, because "person_email
  doesn't exist" is baffling to someone who just created it under that name.
- The descriptor form (`DROP INDEX FOR …`) is a KGLite extension that sidesteps
  naming entirely.

#### Indexes on `name`, titles and ids

A property index serves the lookup that resolves to the same value, so these
indexes are read:

- `CREATE INDEX FOR (n:L) ON (n.name)`. A node with no stored `name` answers
  `n.name` with its title, and the index files it under that title. A stored
  `name` wins over the title, as it does in a scan.
- An index on `title`, on a type's declared title field (`node_title_field`),
  or on its declared id field.

`type`, `node_type` and `label` answer the node type for a node that stores none.
An index over them would hold a subset of what a query compares, so lookups scan
and `CREATE INDEX` returns a warning saying so. On a disk graph the index is a
snapshot: after a write it declines until `reindex()` or `save()`.

One canonical name can cover several structures. A property carrying a hash
index, a B-tree index and a BM25 text index shows three `SHOW INDEXES` rows
sharing a `name`, distinguished by `type` (`PROPERTY`, `RANGE`, `FULLTEXT`).
`DROP INDEX <name>` removes every structure under that name.

A relationship vector index and a relationship BM25 index are both named
`relationship:TYPE.property`, and that name pastes in unquoted too:

```cypher
DROP INDEX relationship:SUPPORTS.evidence;              -- works
DROP INDEX `relationship:SUPPORTS.evidence` IF EXISTS;  -- also works
```

`DROP INDEX` on that name drops every relationship structure under it: the HNSW
accelerator (keeping the vectors, exactly as the node vector arm does) and the
BM25 index. `db.relationship_embeddings.drop_index` and
`db.relationship_text_index.drop` are the same operations one structure at a
time.

- The prefix keeps a node label and a relationship type apart. A `Doc.text` node
  index and a `relationship:Doc.text` relationship index are therefore addressed,
  and dropped, independently.
- There is no descriptor form for a relationship index.
  `DROP INDEX FOR ()-[r:T]-() ON (r.p)` is rejected with the rest of relationship
  index DDL.

`IF EXISTS` is a no-op only when **nothing carries the name**. A name `SHOW
INDEXES` printed always drops something, or the statement errors. It never
reports success over an index that is still installed.

#### `SHOW INDEXES`

`SHOW INDEXES` is a read, so it works on a read-only graph. It returns the same
rows and columns as `CALL db.indexes()`:

| Column | Value |
|---|---|
| `name` | canonical name — `Label.property`, `Label.(a,b)`, or `relationship:TYPE.property` for a relationship index |
| `type` | `PROPERTY` (hash equality or composite), `RANGE` (B-tree), `FULLTEXT` (BM25 text index — see `build_text_index()`), or `VECTOR` (HNSW index over an embedding store — see `build_vector_index()`) |
| `entityType` | `NODE`, or `RELATIONSHIP` for a relationship vector or BM25 index |
| `labelsOrTypes` | single-element list holding the node or relationship type |
| `properties` | indexed property names, sorted for a composite |
| `state` | `ONLINE`, or `DEFERRED` for an index a `defer_index_rebuild=True` load has declared but not built (any write builds it). Nothing in between — KGLite builds indexes atomically |
| `stale` | whether the index is behind the graph. `null` on `PROPERTY` / `RANGE` rows, which are maintained on every write and have no staleness to report |
| `delta` | how many documents (or vectors) the index would re-read to catch up — an upper bound. `null` alongside a `null` `stale` |
| `unembedded` | `VECTOR` rows only: nodes (or relationships) of the type carrying no vector at all. `null` on every other row |

`stale` / `delta` are KGLite-specific. They describe the catch-up contract both
opt-in index kinds share: the index does not follow writes eagerly. It records
that they happened and folds them in at query entry while the delta stays under
that index's `auto_refresh_limit`, which is a document count, not a time budget.

- A `delta` above that limit is the signal to rebuild: with
  `build_text_index(...)`, or for a vector index with `build_vector_index(...)` /
  `refresh_vector_index(...)`.
- A text index catching up past its measured crossover (~1500 documents)
  rebuilds rather than folding, so no catch-up costs more than one rebuild.

The two kinds differ in what an over-limit delta *serves*.

- A stale text index returns `null` for the rows it has no document for, and
  warns.
- A stale **vector** index is simply stepped over. The query falls back to the
  exact scan, which is the oracle the approximate index is measured against, so a
  stale vector index costs latency and never accuracy.

`unembedded` is deliberately not folded into `delta`, because catch-up indexes
vectors that exist and never creates them. A node with no embedding stays
invisible to vector search until `embed_texts` / `set_embeddings` runs. A
`VECTOR` row appears only once an index is built. Embeddings on their own are
reported by `list_embeddings()`.

A node vector index is listed under its **source column** (`Doc.summary`), not
the store name (`Doc.summary_emb`). It therefore shares one canonical name with
that property's other indexes, and `DROP INDEX Doc.summary` removes it along with
them. That drops the accelerator only: the vectors are data, and DDL does not
delete them. A relationship vector index uses the unambiguous canonical name
`relationship:SUPPORTS.evidence`. It appears with `entityType: RELATIONSHIP`
through both `SHOW INDEXES` and `CALL db.indexes()`.

Neo4j 5 also returns `id`, `populationPercent`, `indexProvider`,
`owningConstraint`, `lastRead`, and `readCount`. KGLite has no equivalent state
for any of them, so they are **omitted** rather than filled with invented
values. For filtering and projection use `CALL db.indexes() YIELD …`. `SHOW
INDEXES` rejects `YIELD` / `WHERE` / `BRIEF` / `VERBOSE` rather than accepting a
filter it would ignore.

#### Forms that are rejected

Each of these forms fails with a `CypherExecutionError` naming the construct and
the route that works. It is never a syntax error, and never a no-op that reports
success.

| Statement | Why, and what to use |
|---|---|
| `CREATE TEXT INDEX` | Neo4j's TEXT index accelerates `CONTAINS` / `STARTS WITH` / `ENDS WITH`, which KGLite serves unindexed (a string index already gives prefix pushdown — see above). For *ranked* retrieval, build a BM25 index: `build_text_index(node_type, property)`, then `text_bm25()` |
| `CREATE FULLTEXT INDEX` | BM25 indexes exist, but Neo4j's FULLTEXT is multi-label, multi-property and name-addressed while KGLite's is one node type's one property, so they are created through `build_text_index(node_type, property)` and queried with `text_bm25()`. A built one *is* listed by `SHOW INDEXES` as type `FULLTEXT`, and `DROP INDEX Label.property` removes it |
| `CREATE POINT INDEX` | No point index. Spatial predicates and the spatial-join optimiser work on geometry properties without one |
| `CREATE VECTOR INDEX` | Vector indexes exist, but need an existing embedding store and HNSW build parameters, so they are created through `build_vector_index(...)`. A built one *is* listed by `SHOW INDEXES` as type `VECTOR`, and `DROP INDEX Label.column` removes it |
| `CREATE LOOKUP INDEX` | Label and relationship-type lookup is always indexed automatically (`type_indices`) |
| `CREATE INDEX FOR ()-[r:T]-() ON (r.p)` | No DDL form creates a relationship index. Relationship *vector* and *BM25 text* indexes do exist: built with `CALL db.relationship_embeddings.build_index(...)` / `CALL db.relationship_text_index.build(...)`, listed by `SHOW INDEXES` with `entityType: RELATIONSHIP`, and removed with `DROP INDEX relationship:T.p`. Every other relationship property is queryable, just scanned |
| `... OPTIONS { ... }` | No index providers or per-index configuration to apply |
| `CREATE RANGE INDEX ... ON (n.a, n.b)` | The B-tree is single-property. Use a composite equality index, or one `CREATE RANGE INDEX` per property |

#### On disk-backed graphs

`CREATE INDEX` on a `storage='disk'` graph builds the **persistent** mmap-backed
index. It is the same one `create_index(...)` builds there, not the in-memory
HashMap (which would need multiple GB of heap for a large type and be rebuilt on
every load). Two consequences:

- Persistent property indexes cover **string columns**. A statement that indexes
  nothing on a populated node type is **rejected** rather than reported as
  succeeding, so a numeric property gets an error naming the reason. Use an
  in-memory or mapped graph if you need to index a non-string property.
- `SHOW INDEXES` and `CALL db.indexes()` list the in-memory index structures, so
  a disk graph's persistent indexes do **not** appear there yet. The index is
  real and the planner consults it. Only this listing is incomplete. Re-running
  `create_index(...)` reports `created=False` for an installed persistent index.

`CREATE RANGE INDEX` and composite `CREATE INDEX` build in-memory structures in
every storage mode.

#### Guards

Schema is graph state, so index DDL is a **mutation**:

- It is blocked on a read-only graph (`read_only(True)`).
- It is blocked in a read-only transaction.
- It is rolled back with the rest of a failed statement.
- On a schema-locked graph, indexing an undeclared property is rejected. This is
  the same typo-guard writes get.
- A schema command is a standalone statement. It cannot follow another clause or
  appear inside a `CALL { }` body.
- An index belongs to one node type, so `write_scope=[...]` applies. `CREATE INDEX`
  / `DROP INDEX` on a type outside the whitelist is a scope violation.
- `SHOW INDEXES` is unaffected. A write scope restricts mutations, not
  visibility.

`indexes_added` and `indexes_removed` join `graph.last_mutation_stats`, mirroring
Neo4j's `indexesAdded` / `indexesRemoved` summary counters.

### Cypher constraint DDL

Neo4j 5 constraint DDL runs against KGLite and routes to **real per-write
enforcement**. It covers Cypher `CREATE` / `INSERT` / `MERGE` / `SET` / `REMOVE`
*and* the bulk loader (`add_nodes`, blueprints, `from_records`, WAL replay). A
declaration is not documentation: once it succeeds, a violating write is
rejected.

```cypher
CREATE CONSTRAINT [name] [IF NOT EXISTS] FOR (n:Label) REQUIRE n.prop IS UNIQUE
CREATE CONSTRAINT [name] [IF NOT EXISTS] FOR (n:Label) REQUIRE (n.a, n.b) IS UNIQUE
CREATE CONSTRAINT [name] [IF NOT EXISTS] FOR (n:Label) REQUIRE n.prop IS NOT NULL
CREATE CONSTRAINT [name] [IF NOT EXISTS] FOR (n:Label) REQUIRE n.prop IS NODE KEY
CREATE CONSTRAINT [name] [IF NOT EXISTS] FOR (n:Label) REQUIRE n.prop IS :: INTEGER
CREATE CONSTRAINT [name] [IF NOT EXISTS] FOR ()-[r:TYPE]-() REQUIRE r.prop IS NOT NULL
CREATE CONSTRAINT [name] [IF NOT EXISTS] FOR ()-[r:TYPE]-() REQUIRE r.prop IS :: INTEGER
DROP CONSTRAINT <name> [IF EXISTS]
SHOW CONSTRAINTS
```

The Neo4j 4 `ASSERT` spelling is accepted in place of `REQUIRE`, so a 4.x-era
script ports without edits. The optional `NODE` / `RELATIONSHIP` scope word before
`UNIQUE` / `KEY` is accepted too, but it must **agree** with the `FOR` pattern.
`FOR (n:Label) … IS RELATIONSHIP KEY` asks for two different constraints in one
statement, and is refused rather than resolved in favour of one half. Unscoped
`IS UNIQUE` / `IS KEY` mean "whatever this pattern targets" and are legal against
either.

#### What each form enforces

| Statement | Enforces | Backed by |
|---|---|---|
| `REQUIRE n.p IS UNIQUE` | no two nodes of the type share `p` | a single-occupant unique index |
| `REQUIRE (n.a, n.b) IS UNIQUE` | no two nodes share the *tuple* | one composite unique index |
| `REQUIRE n.p IS NOT NULL` | every node of the type has a non-null `p` | the node type's required-field list |
| `REQUIRE n.p IS NODE KEY` | both of the above, at once | a unique index **plus** a required field |
| `REQUIRE n.p IS :: TYPE` | every value written to `p` has the declared type | a declared per-property type, checked before the write lands |
| `REQUIRE r.p IS NOT NULL` (on `FOR ()-[r:T]-()`) | every relationship of the type has a non-null `p` | a declared per-connection-type presence rule |
| `REQUIRE r.p IS :: TYPE` (on `FOR ()-[r:T]-()`) | every value written to `r.p` has the declared type | a declared per-connection-type property type |

A constraint declared over several properties constrains the **combination**, not
each property. `REQUIRE (n.city, n.age) IS UNIQUE` permits many nodes in the same
city as long as no `(city, age)` pair repeats.

**NULL is exempt**, matching Neo4j. A node is outside a uniqueness constraint
unless *every* property in the tuple is present and non-null. So many nodes may
share "no email" while `email` is `UNIQUE`.

`IS NODE KEY` is served as the conjunction of uniqueness and presence. The two
halves are installed **atomically**: if the presence half cannot be declared, the
uniqueness half is rolled back, so a statement that reported failure has changed
nothing. `DROP CONSTRAINT` on a node key likewise withdraws both halves.

A node type's **declared primary key is not droppable through DDL.**
`define_schema({"nodes": {"User": {"primary_key": "email"}}})` shows up in
`SHOW CONSTRAINTS` as a `NODE_KEY` row. It is one declaration owned by the schema
rather than a DDL constraint with a store of its own, so a `DROP CONSTRAINT`
naming `User.email` is refused. `IF EXISTS` is refused too, because the key
exists and is enforced; it is simply not droppable here.

- Withdraw it by re-declaring the type without a key
  (`define_schema({"nodes": {"User": {}}})`) or with `clear_schema()`.
- Every other constraint on a keyed type, including a composite tuple that
  contains the key property, is its own declaration and drops normally.

Withdrawing the key withdraws **only** the key. A `CREATE CONSTRAINT … IS UNIQUE`
declared on the same property is a separate declaration that happens to share the
key's index.

- While both stand, the row reads `NODE_KEY` under the DDL name.
- Re-declaring the type without a key leaves that row in force as `UNIQUENESS`
  until `DROP CONSTRAINT` withdraws it.
- Declaring it the other way round is refused as a duplicate, because the key
  already enforces it. A property never carries two live uniqueness declarations.

#### Relationship constraints

A constraint on a relationship is written against a relationship pattern. It
serves the two kinds that do not depend on relationship identity:

```cypher
CREATE CONSTRAINT knows_since FOR ()-[r:KNOWS]-() REQUIRE r.since IS NOT NULL
CREATE CONSTRAINT FOR ()-[r:KNOWS]-() REQUIRE r.since IS :: INTEGER
DROP CONSTRAINT knows_since
```

Everything the node forms promise holds here. Declaring one **validates it
against every existing relationship of the type** and refuses, installing
nothing, if the data already violates it. Once installed, it is enforced on:

- `CREATE` / `INSERT` (and `MERGE`'s create branch).
- `SET r.p` in all three spellings (`SET r.p = v`, `SET r = {…}`,
  `SET r += {…}`).
- `REMOVE r.p`.
- The bulk `add_connections` / `replace_connections` loaders.

A refused write changes nothing: no relationship, no connection-type metadata,
and no entry in the change-capture stream. Declarations survive `save()` /
`load()`.

The pattern's direction and endpoints are ignored. `()-[r:T]-()`,
`()-[r:T]->()` and `()<-[r:T]-()` all declare the same constraint on the
connection type, exactly as they do for Neo4j.

A bulk row is judged on the state it will actually leave behind, which is not
always the row:

- Under `conflict_handling='preserve'`, a value the stored relationship already
  has is discarded and therefore never refused.
- Under `'sum'`, an addition that turns an integer into a float **is** refused
  even though both operands satisfy the constraint on their own.
- A frame is refused whole rather than row-by-row.
- `replace_connections` raises the refusal before its delete, so a rejected frame
  never costs the relationships already stored.

**`IS UNIQUE` and `IS RELATIONSHIP KEY` on a relationship are refused.** This is a
data-model gap, not a missing feature flag. The bulk loader deduplicates
`(type, source, target)` while Cypher `CREATE` freely makes parallel edges, so
KGLite has no single answer for when two relationships of a type are the same
one. A uniqueness declaration would mean different things depending on which
write path produced the data. The refusal names that reason.

#### Property-type constraints (`IS :: T`)

`REQUIRE n.prop IS :: TYPE` (and the equivalent `IS TYPED TYPE` spelling)
declares the type every value written to that property must have. It is checked
on the same three write paths as every other constraint, before the value lands.

The accepted type names are the Neo4j 5 names with an **exact** KGLite value
counterpart:

| Declared | Accepts |
|---|---|
| `BOOLEAN` | `true` / `false` |
| `STRING` | strings |
| `INTEGER` | integers, including auto-assigned node ids |
| `FLOAT` | floats **only** — an integer does not satisfy `FLOAT` |
| `DATE` | a calendar date (no time of day) |
| `LOCAL DATETIME` | a date *and* wall-clock time, with no timezone |
| `DURATION` | a duration value |
| `POINT` | a point value |

Anything else is **rejected by name**, and the error lists the names that do
work. That covers `LIST<...>`, unions, `ZONED DATETIME` / `LOCAL TIME` /
`ZONED TIME`, and decorated forms like `STRING NOT NULL`. This is deliberate:
KGLite has no timezone-aware or time-of-day-only value, so accepting
`ZONED DATETIME` would promise a type it cannot represent, and a constraint that
enforces something other than what was written is worse than one that was
refused. For those shapes use `define_schema()` + `validate_schema()`, or
`lock_schema()`.

**A type constraint is not an existence constraint.** As in Neo4j, `null` and an
absent property both satisfy it, and `REMOVE n.prop` is not a violation. Declare
`IS NOT NULL` alongside it when a value is also required:

```cypher
CREATE CONSTRAINT age_typed   FOR (p:Person) REQUIRE p.age IS :: INTEGER;
CREATE CONSTRAINT age_present FOR (p:Person) REQUIRE p.age IS NOT NULL;
```

A property carries at most one declared type. Re-declaring it with a different
one is rejected, so change it with `DROP CONSTRAINT` then a fresh declaration.
Where a declared type and a `lock_schema()`-recorded property type both cover a
property, **the declaration wins** and the error names the constraint you wrote.

#### Declaring against dirty data fails

A constraint the stored data already violates is **rejected**, and nothing is
installed. A constraint that silently exempts the rows already present would be
worse than no constraint at all. The error names an offending value:

```
cannot declare a UNIQUE constraint on Person.email: the existing data already has
2 duplicate values (for example 'email' = 'a@b.c'). Deduplicate the node type
before declaring the constraint.
```

Deduplicate (or populate, for `IS NOT NULL`) and retry.

#### Constraint names *are* stored

Constraint names are stored, which is the opposite of
[index names](#index-names-are-canonical-not-yours). This is deliberate: a ported
schema script almost always names its constraints and drops them by name, and
there is no descriptor form to fall back on.

```cypher
CREATE CONSTRAINT person_email_unique FOR (p:Person) REQUIRE p.email IS UNIQUE;
SHOW CONSTRAINTS;                        -- name is 'person_email_unique'
DROP CONSTRAINT person_email_unique;      -- works
```

A constraint declared **without** a name gets the canonical descriptor
`Label.property` (or `Label.(a, b)`). `DROP CONSTRAINT` accepts that spelling
too, so `SHOW CONSTRAINTS` output pastes straight in.

- Names are unique per graph. Reusing one for a different constraint is an error
  rather than a silent re-pointing.
- Names persist in the `.kgl` file and survive save/load. A name whose constraint
  has been dropped is discarded at save time, so it cannot resurrect.

The registry is a lookup aid, never the source of truth. The constraint lives in
the enforcement structure, so a missing name can cost addressability but never
enforcement.

#### `SHOW CONSTRAINTS`

`SHOW CONSTRAINTS` is a read, so it works on a read-only graph and is unaffected
by a write scope. It returns the same rows and columns as
`CALL db.constraints()`:

| Column | Value |
|---|---|
| `name` | the declared name, or the canonical descriptor when unnamed |
| `type` | `UNIQUENESS`, `NODE_KEY`, `NODE_PROPERTY_EXISTENCE`, `NODE_PROPERTY_TYPE`, `RELATIONSHIP_PROPERTY_EXISTENCE`, or `RELATIONSHIP_PROPERTY_TYPE` |
| `entityType` | `NODE` or `RELATIONSHIP` |
| `labelsOrTypes` | single-element list holding the node type — or the relationship type on a relationship row |
| `properties` | constrained property names, in declaration order |
| `propertyType` | the declared type on a `NODE_PROPERTY_TYPE` / `RELATIONSHIP_PROPERTY_TYPE` row, `null` on every other kind |

A node key is **one** row (`NODE_KEY`), not a uniqueness row plus an existence
row. A declared property type is its own row, never folded into another, so a
property can be `UNIQUE` **and** typed.

Neo4j 5 also returns `id` and `ownedIndex`. KGLite has no equivalent state for
either (a unique constraint *is* its index rather than owning a separate one), so
they are **omitted** rather than filled with invented values. For filtering and
projection use `CALL db.constraints() YIELD …`. `SHOW CONSTRAINTS` rejects
`YIELD` / `WHERE` / `BRIEF` / `VERBOSE` rather than accepting a filter it would
ignore.

#### Forms that are rejected

Each of these forms fails with a `CypherExecutionError` naming the construct and
the route that works. It is never a syntax error, and **never a success that
enforces nothing**.

| Statement | Why, and what to use |
|---|---|
| `REQUIRE n.p IS :: LIST<STRING>` | Only the type names with an exact KGLite value counterpart are accepted (see [Property-type constraints](#property-type-constraints-is--t)). Lists, unions, zoned temporal types and decorated forms are refused by name rather than approximated; the error lists the names that work. `validate_schema()` audits existing data against `define_schema`'s per-type `types` map, and `lock_schema()` rejects a write whose value disagrees with the recorded property type |
| `REQUIRE n.id IS UNIQUE` / `IS NODE KEY` | Uniqueness over the identity field — under **any** spelling that resolves to it, `id` itself or the node type's own id column (`person_id`) — is refused. `id` is a `NodeData` field, not an entry in the property map, so the write-path claim is never produced and the constraint would admit duplicates while reporting success. Declare the node type's primary key instead: `define_schema({'nodes': {'Person': {'primary_key': 'id'}}})` probes the per-type id index on every write path, and `MERGE` is the idempotent alternative to `CREATE`. Only `id` is affected — `title`, a column aliased to `title`, and ordinary properties all enforce correctly. `IS NOT NULL` on `id` **is** accepted, and genuinely enforced: every write path resolves an id before the check, so an *omitted* one satisfies the requirement, but an explicit `CREATE (:Person {id: null})` violates it — and the declaration itself is refused when existing rows already carry a null `id` |
| `FOR ()-[r:T]-() REQUIRE r.p IS UNIQUE` / `IS RELATIONSHIP KEY` | KGLite has no single answer for when two relationships of a type are the same one — the bulk loader deduplicates `(type, source, target)` while Cypher `CREATE` freely makes parallel edges — so a uniqueness declaration would mean different things depending on which write path produced the data. Presence and property type **are** served on relationships (see [Relationship constraints](#relationship-constraints)) |
| `REQUIRE …` with no properties | Nothing to constrain |
| `CREATE CONSTRAINT <name> …` reusing a live name | Names are unique per graph; drop the existing one or choose another name |

Refusing an *unmappable* type name, rather than approximating it, is the most
important decision in this surface. A `CREATE CONSTRAINT` that returns cleanly is
a promise users build data-integrity assumptions on. Keeping that promise honest
is worth an error.

#### Guards

Schema is graph state, so `CREATE CONSTRAINT` / `DROP CONSTRAINT` are
**mutations**:

- They are blocked on a read-only graph (`read_only(True)`).
- They are blocked in a read-only transaction.
- They are rolled back with the rest of a failed statement.
- On a schema-locked graph, constraining an undeclared node property is rejected.
  This is the same typo-guard writes get. On the relationship side, the lock
  gates the connection *type* but not the property, because the lock does not
  check edge property names on write either.
- A node constraint belongs to one node type, so `write_scope=[...]` applies to
  it.
- A **relationship** constraint is allowed under any write scope. Scopes name node
  types, and there is no relationship spelling for one to name.
- A schema command is a standalone statement. It cannot follow another clause or
  appear inside a `CALL { }` body.

`SHOW CONSTRAINTS` is unaffected by all of the above except the standalone rule.
A write scope restricts mutations, not visibility.

`constraints_added` and `constraints_removed` join `graph.last_mutation_stats`,
mirroring Neo4j's `constraintsAdded` / `constraintsRemoved` summary counters.
They count *constraints*, not the structures behind them, so `IS NODE KEY`
reports 1.

#### One caveat on error types

A constraint violation raised through Cypher arrives as `CypherExecutionError`,
not as the typed `ConstraintViolationError` that `define_schema` raises. The
Cypher executor's internal error channel is a string, so the structured violation
is rendered to text before any binding sees it. The *message* is stable and names
the constraint, the property, and the offending value, so match on that.
Enforcement is identical either way.

### Ontology (declared semantic layer)

The ontology is declared with the Python `define_ontology()` (classes with an
`is_a` forest, abstract supertypes, and relationship semantics), persisted with
the graph, and read-only from Cypher:

```cypher
SHOW ONTOLOGY
CALL ontology_audit() YIELD entity_kind, rule, severity, violations, exempted, total, pct, domain_class, property
CALL ontology_audit({by: 'domain_class'}) YIELD rule, domain_class, violations  -- per violating class
CALL ontology_audit({by: 'property'}) YIELD rule, property, violations, pct     -- per declared property
CALL type_domain_violation() YIELD source, target, rule   -- no-arg: checks every declaration
CALL edge_property_violation() YIELD relationship, check, source, target, property, properties, exempt
CALL node_property_violation() YIELD class, check, node, property, properties
```

`SHOW ONTOLOGY` returns one row per declared class and one per declared
relationship, and zero rows when nothing is declared.

- A class row has `kind`, `name`, `is_a`, `abstract`, `description`,
  `required_properties`, `property_types`, `enforcement`.
- A relationship row has `kind`, `name`, `domain`, `range`,
  `required_properties`, `property_types`, `enforcement`, `exempt`,
  `description`.

`ontology_audit()` is the scorecard: one row per declared check with its
violation count, denominator, percentage, and declared severity.

- `entity_kind` is `node` for class contracts and `edge` for relationship checks.
- Severity is `advisory` / `warn` / `error`, acted on by blueprint builds and
  reported everywhere else.
- `exempted` counts the rows a declaration's `exempt` classes excuse. They are
  left out of `violations` (and so out of the severity the gate acts on), and
  `violations + exempted` is everything the check flagged.

`{by: 'domain_class'}` answers the usual follow-up, *which* source types are
violating. It fans each rule's row out into one row per primary node type its
violations come from.

- `violations` and `pct` are then that class's share (they sum back to the rule's
  aggregate).
- `severity`, `exempted` and `total` keep their per-rule values on every fanned
  row.
- Exempted rows are left out, so a class whose every violation is excused gets no
  row at all.
- A rule with no violations to break down keeps its single aggregate row.
- Without the parameter, `domain_class` is Null on every row.

The domain-side class is:

- The edge's source, for relationship `domain` / `range` / `required_properties` /
  `property_types`.
- The node itself, for `required` / `cardinality`.
- The first bound node, for the pair/triple shapes (`inverse`, `symmetric`,
  `transitive`): the source of the edge or chain whose partner is missing.

Class property contracts partition by the violating node's primary type.

`{by: 'property'}` answers the other follow-up, *which fields* are missing. It
fans the `required_properties` and `property_types` rules into one row per
**declared** property: `violations` counts the nodes or edges failing that
property, `total` the rule's covered nodes or relationship edges, and `pct` the
share failing it. Every other rule keeps its aggregate row with a Null
`property`.

The two breakdowns are different kinds of answer, and mixing them up
double-counts.

- `domain_class` **partitions** a rule. Every violating row has exactly one source
  class, so the rows sum back to the aggregate.
- `property` is a **census**. A node or edge missing three declared properties is
  counted under all three, so the rows sum to *at least* the aggregate, and adding
  them up is not the rule's violation count.
- A declared property nothing fails still gets a row, at zero. "This field is
  complete" is what a census is for.
- Only one axis applies at a time. The column you did not ask for is Null.

`edge_property_violation()` lists the individual edges behind the
`required_properties` and `property_types` counts (the two declared checks with
no rule procedure of their own). It takes no arguments.

- It returns one row per flagged edge.
- `properties` lists every declared property the edge fails, and `property` is the
  first of them.
- `exempt` marks the rows an `exempt` declaration excuses. A relationship's row
  count for a check therefore equals that rule's `violations + exempted`, however
  many properties one edge fails.
- `UNWIND` the list to count per property.

`node_property_violation()` is the corresponding no-argument class-contract
drill-down.

- Each row names the declaring class, check, node and failed properties.
- Inherited contracts apply through the node's primary class ancestry.
- It has no edge endpoints or exemption column.
- Required properties reject missing/null values.
- Type checks use the shared property-type grammar, including outer
  `list`/`array`.

The six declaration-backed rule procedures (`type_domain_violation`,
`type_range_violation`, `missing_required_edge`, `cardinality_violation`,
`inverse_violation`, `transitivity_violation`) called with **no arguments**
iterate the declarations instead of erroring. Every row carries a `rule` column
naming the declaration it came from.

- A `domain`/`range` naming an **abstract class** widens to the class plus its
  declared descendants. This is the union-endpoint case
  (`MANAGED_BY from = six concrete types`) a flat schema cannot declare.
- Semantics are annotations, not axioms: the ontology never changes what a
  `MATCH` returns.
- `cardinality`/`required` describe *outgoing* edges of the domain type.
- The layer is deliberately distinct from `set_parent_type` (presentation
  ownership for `describe()` tiering).

`transitive: true` and `ancestry: true` are mutually exclusive annotations of the
same hierarchy shape, and they mean different things.

- `transitive` enrolls `transitivity_violation`, which audits a **stored**
  closure: every `a→b→c` needs a stored `a→c` edge. A taxonomy holding only parent
  pointers therefore reports 100% violations.
- `ancestry` enrolls no check. It records that the chain is meaningful and is
  walked with `*1..`, which is what a parent-pointer taxonomy (`STRAT_PARENT`,
  `wdt:P279`) is.

`materialize_ontology()` (Python) stamps each declared ancestor as a real
secondary label, after which `MATCH (p:Person)` matches every descendant with
ordinary label semantics. On such a label `EXPLAIN` marks the per-member index
probe with a [`ClosureProbe` row](#explain).

## Timeseries Functions

Timeseries functions query time-indexed numeric data attached to nodes.

- Date arguments are strings (`'2020'`, `'2020-2'`, `'2020-2-15'`), integers (a
  year), `date(…)` or `datetime(…)` values.
- A series' resolution is `year`, `month` or `day`.
- Each entry is keyed by the first day of its period.

### Date-string syntax

| String | Depth | Matches resolution |
|--------|-------|--------------------|
| `'2020'` | year | year, month, day |
| `'2020-2'` | month | month, day |
| `'2020-2-15'` | day | day only |

**Precision rule:**
- `ts_at` reads one entry. A key as fine as the series resolution or finer reads
  the period that *contains* it: on a month series `'2015-6-15'`,
  `date('2015-06-15')` and `datetime('2015-06-15T13:00:00')` all read June 2015. A
  key coarser than the resolution (`'2015'` or `2015` on a month series) names
  several periods and is an error.
- Range functions (`ts_sum`, `ts_avg`, `ts_min`, `ts_max`, `ts_count`,
  `ts_series`) count an entry when its key (the first day of its period) lies in
  `[start, end]`. A string or integer bound expands to its whole year or month. A
  day-level bound therefore does not cut a month in half. On a month series,
  `ts_sum(ch, date('2015-02-15'), date('2015-04-15'))` covers the March and April
  entries only (the February key, 02-01, lies before the start), while
  `ts_sum(ch, '2015-2', '2015-4')` covers February to April.
- `ts_delta` takes the first entry of the period each bound names. A bound finer
  than the resolution starts at the containing period.

### Functions

| Function | Arguments | Returns | Description |
|----------|-----------|---------|-------------|
| `ts_sum(n.channel)` | 1 | Float | Sum of all values; `0.0` when the window holds none (an empty window or an all-missing channel), where `ts_avg` / `ts_min` / `ts_max` give `null` and `ts_count` gives `0` |
| `ts_sum(n.channel, 'start')` | 2 | Float | Sum within prefix range |
| `ts_sum(n.channel, 'start', 'end')` | 3 | Float | Sum in range [start, end] inclusive |
| `ts_avg(n.channel [, 'start'] [, 'end'])` | 1-3 | Float | Average (same range rules as ts_sum) |
| `ts_min(n.channel [, 'start'] [, 'end'])` | 1-3 | Float | Minimum value in range |
| `ts_max(n.channel [, 'start'] [, 'end'])` | 1-3 | Float | Maximum value in range |
| `ts_count(n.channel)` | 1 | Integer | Count of non-NaN values |
| `ts_at(n.channel, 'date')` | 2 | Float/null | The entry of the period containing the key (a coarser key is an error) |
| `ts_first(n.channel)` | 1 | Float/null | First non-NaN value in series |
| `ts_last(n.channel)` | 1 | Float/null | Last non-NaN value in series |
| `ts_delta(n.channel, 'from', 'to')` | 3 | Float/null | Value at 'to' minus value at 'from' (prefix match) |
| `ts_series(n.channel [, 'start'] [, 'end'])` | 1-3 | List | Extract `[{time, value}, ...]` as JSON |

NaN values are skipped in all aggregation functions.

### Examples

```python
# Aggregate monthly data by year
graph.cypher("MATCH (f:Project) RETURN f.title, ts_sum(f.output, '2020') AS prod")

# Range across months
graph.cypher("MATCH (f:Project) RETURN ts_avg(f.output, '2020-1', '2020-6') AS h1_avg")

# Multi-year range
graph.cypher("MATCH (f:Project) RETURN ts_sum(f.output, '2018', '2023') AS total")

# Exact month lookup
graph.cypher("MATCH (f:Project) RETURN ts_at(f.output, '2020-3') AS march_prod")

# Change between two time points
graph.cypher("MATCH (f:Project) RETURN ts_delta(f.output, '2019', '2021') AS change")

# Top producers
graph.cypher("""
    MATCH (f:Project)
    RETURN f.title, ts_sum(f.output, '2020') AS prod
    ORDER BY prod DESC LIMIT 10
""")

# Filter by production threshold
graph.cypher("""
    MATCH (f:Project)
    WHERE ts_sum(f.output, '2020') > 100.0
    RETURN f.title, ts_sum(f.output, '2020') AS prod
""")

# Extract full series for plotting
graph.cypher("MATCH (f:Project {title: 'TUNDRA'}) RETURN ts_series(f.output, '2015', '2020') AS data")

# Latest reading
graph.cypher("MATCH (s:Sensor) RETURN s.title, ts_last(s.temperature) AS latest")
```

### Windows relative to the statement's instant

`valid_instant()` returns the instant of the statement's valid-time context, so a
series can be read relative to it without repeating the date:

```cypher
FOR VALID_TIME AS OF date('2015-06-15')
MATCH (p:Plant)
RETURN ts_at(p.output, valid_instant()) AS this_month,
       ts_sum(p.output, date_truncate(valid_instant(), 'year'), valid_instant()) AS year_to_date,
       ts_sum(p.output, add_months(date_truncate(valid_instant(), 'month'), -11), valid_instant()) AS trailing_12m
```

With no prefix on a graph that declares validity, the instant is today. Under
`FOR VALID_TIME ALL`, in a write, and on a graph with no context,
`valid_instant()` is an error. A datetime context returns a datetime.
`date_truncate()` and `add_months()` take dates, so wrap it as
`date(valid_instant())` first.

### Precision

```python
# OK: year query on month data (a range: aggregates all months of 2020)
graph.cypher("MATCH (f:Project) RETURN ts_sum(f.output, '2020')")

# OK: month, day, date and datetime keys on month data read the containing month
graph.cypher("MATCH (f:Project) RETURN ts_at(f.output, '2020-3')")
graph.cypher("MATCH (f:Project) RETURN ts_at(f.output, date('2020-03-15'))")

# ERROR: a year key with ts_at on month data names twelve entries
graph.cypher("MATCH (f:Project) RETURN ts_at(f.output, '2020')")
# → "ts_at() key precision 'year' is coarser than the series resolution 'month'; …"

# Range bounds are key-in-range: the 2020-02 entry (key 2020-02-01) is before 02-15
graph.cypher("MATCH (f:Project) RETURN ts_sum(f.output, date('2020-02-15'), date('2020-04-15'))")
```

## Naming — identifiers, reserved words & structural accessors

### Reserved keywords as names (soft keywords)

Most reserved keywords can be used directly as a **relationship type**, **node
label**, or **property key**. The parser treats them as names in those positions:

```cypher
CREATE (s:SourceDoc)-[:CONTAINS]->(c:Chunk)   // CONTAINS as a rel type
MATCH  (n:CONTAINS)                            // … as a label
CREATE (n:Doc {order: 1, in: true})            // … as property keys
RETURN n.contains, n.order                     // … and in property access
```

Keyword names are **case-preserving**: the stored name is the exact source
spelling. `{order: 1}` stores the key `order`, and `[:contains]` creates the
relationship type `contains`. Names are case-sensitive, so `n.ORDER` and
`n.order` are different keys. Releases up to 0.12.14 canonicalized keyword names
to uppercase, so a graph written by an older release may store `ORDER` where a
lowercase source now reads `order`. Backticks reach the old spelling.

The soft set covers the operator / comparison / sort / set / mutation keywords
(`CONTAINS`, `IN`, `IS`, `NOT`, `STARTS`, `ENDS`, `ORDER`, `BY`, `ASC`, `DESC`,
`DISTINCT`, `ALL`, `MERGE`, `CREATE`, `DELETE`, `SET`, `REMOVE`, `UNION`, …),
**and the value literals `TRUE`, `FALSE` and `NULL`**. openCypher spells a schema
name as `SchemaName = SymbolicName | ReservedWord`, so those three are legal
labels, relationship types and property keys:

```cypher
CREATE (:TRUE {null: 1})-[:FALSE]->(:Thing)
MATCH  (n:TRUE {null: 1})-[:FALSE]->() RETURN n.null
```

**Position decides.** The same word in a *value* position is still the literal,
and nothing about that changed: `{x: true}` is a boolean property,
`WHERE n.x = true` is a boolean comparison, and `RETURN null` is null. Only the
name positions read them as names:

- After the `:` of a label or relationship type.
- After the `|` of a type alternation.
- The key side of a property map.

The remaining reserved words are the clause-flow keywords (`MATCH`, `WHERE`,
`RETURN`, `WITH`, `AND`, `OR`, …) and the value-expression keywords
(`CASE`/`WHEN`/`END`, `EXISTS`). For any of those, quote it with **backticks**:

```cypher
CREATE (n:Doc {`where`: 1})
RETURN n.`where`
```

**Variables are the exception, in both directions.** `TRUE`, `FALSE` and `NULL`
are *not* accepted as bare variable names (`MATCH (true:Thing)` is a syntax
error). openCypher's `Variable = SymbolicName` excludes them, and a bare `true`
in an expression is the literal, so such a variable could never be read back.
Backticks make it an ordinary variable:

```cypher
MATCH (`true`:Thing) RETURN `true`.id
```

### Identifier charset & special characters (hyphens, dots, spaces)

A **bare** identifier (label, relationship type, property key, variable) must
match `[A-Za-z_][A-Za-z0-9_]*`: a letter or underscore followed by letters,
digits, or underscores. Anything outside that set (a hyphen, dot, space, or
leading digit) must be **backtick-quoted**. This matters most for relationship
types like `supports-claim` or `refines-idea`. Written bare, the `-` is parsed as
the relationship-arrow token and you get a syntax error, so backtick them:

```cypher
// Hyphenated / dotted / spaced rel types and labels — backtick them:
CREATE (a)-[:`supports-claim`]->(b)
MATCH  (a)-[r:`refines-idea`]->(b) RETURN a, b
MATCH  (n:`Legal Document`)       RETURN n
RETURN n.`dc.title`
```

**A backtick inside a quoted identifier is written as two backticks**
(openCypher's escape). Every name is therefore representable, and a quoted
identifier cannot be terminated early:

```cypher
CREATE (:`We``ird`)          // the label   We`ird
RETURN n.`dc``title` AS t    // the key     dc`title
```

This matters beyond exotic labels. **If you build a query by interpolating a
label, relationship type, property key, alias or variable, double every backtick
in it** (`name.replace("`", "``")`) before wrapping it in backticks. Without
doubling, a name carrying a backtick closes its own quote and the rest of it is
read as grammar. A caller passing ``Person`) DETACH DELETE n //`` as a label
emitted a query that deleted every `Person` and reported a count. Values never
need this: pass them as `params=` and they can never become syntax.

The **string-typed APIs do not need escaping**. They take the type/label as a
plain string, so `add_connections(df, "supports-claim", …)`,
`add_nodes(df, "Legal Document", …)`, and `create_index("Doc", "dc.title")` all
accept arbitrary characters directly. Backticks are only a *Cypher-surface*
concern: escape when you name such a type/label/key inside a query, not when you
create it through the Python API.

### Structural accessors vs stored properties

Every node answers four convenience accessors:

| Accessor | Returns |
|----------|---------|
| `n.id` | the node's unique id (identity — always) |
| `n.title` | the node's title (identity — always) |
| `n.type` / `n.node_type` / `n.label` | the node's primary type string |
| `n.name` | the node's title |

`n.type` / `n.node_type` / `n.label` / `n.name` are **property-first**. If the
node stores a real property of that name, `n.<name>` returns the *stored value*.
The structural string is only the fallback when no such property exists. So a
`label` / `type` / `name` column loaded via `add_nodes` (or set via `CREATE`)
round-trips and reads back correctly.

`id` and `title` are the node's identity fields and always return the identity (no
stored property can shadow them). Use `labels(n)` for the label set and `id(n)` /
`type(r)` for the structural forms regardless of any same-named property.

`CREATE` and `MERGE` take a node's title from its pattern in this order:

- On a type loaded with `node_title_field`, the declared title field. A
  different `title` value in the same pattern is refused, as two different
  values for the id and its declared spelling are.
- Otherwise `title`, then `name`. A pattern carrying both keeps both: `n.title`
  reads the title and `n.name` reads the stored `name`.
- With no title in the pattern, the fallback depends on the type:

  | Type | Title | Example |
  |---|---|---|
  | Titled by its ids: loaded by `add_nodes` with no `title` column and no `node_title_field` | The id | `12` |
  | Any other type, including one with text titles | `<Label>_<id>` | `Person_12` |

  A type counts as titled by its ids when it has no declared title field and
  its first node's title equals that node's id.

### Identity (`id`) and prefixed-id datasets (`nid`)

**`id(n)` reads the node's `id` field, not an internal identifier.** In Neo4j and
most other Cypher implementations, `id(n)` returns an engine-assigned integer
that has nothing to do with your data. In kglite it returns the source data's own
key, exactly what `n.id` returns. Two consequences worth planning around:

- It is only as stable across a rebuild as the source key is. A rebuild that
  renumbers your keys renumbers `id(n)`.
- It is **not** a positional index into anything. Treating the value as a row
  offset or an array position answers the wrong rows, silently, because the wrong
  rows are still real nodes.

Use it as a key, join on it, and let `labels(n)` / `type(r)` cover the structural
questions.

`n.id` is the node's **indexed logical identity** and behaves identically in every
storage mode (in-memory / mapped / disk). `CREATE (n {id: X})` and
`add_nodes(unique_id_field='id')` both make `X` the identity, `MATCH (n {id: X})`
finds it, and it survives save → load.

`id` is unique by convention. If duplicate ids are created, `MATCH (n {id: X})`
returns one node per id. The statement's warnings say so, so use `MERGE` on the
id alone or dedupe the input. A `MERGE` never forks an id.

Precisely, a lookup returns one node per **(type, id)**:

- An unlabeled `{id: X}` (or `WHERE id(n) = X`) anchors through every type's id
  index. It returns one node per type holding X, in deterministic order,
  identically for literal and `$param` spellings.
- With *intra-type* duplicates present, the anchored plan still answers one node
  per type, while `disable_optimizer=True` filters every duplicate. This is the
  documented one-per-id collapse, only observable on graphs already emitting the
  duplicate-id warning.

To audit a type for collisions after the fact,
`CALL duplicate_id({type: 'Artifact'}) YIELD node` yields every node of that type
whose id is shared (the identity-column sibling of `duplicate_title`).

A node type loaded with its own column names (`add_nodes(df, 'Person',
'person_id', 'person_name')`) accepts **either spelling** on the write path, not
just on reads.

- `CREATE (:Person {person_id: 99, person_name: 'Cleo'})` makes 99 the identity
  and `'Cleo'` the title. The values are *promoted* into those fields rather than
  also stored as properties, so `p.person_id` and `properties(p)['person_id']`
  cannot disagree.
- `MERGE` matches and creates through the same resolution.
- `SET` / `REMOVE` on the title spelling write the title. On the id spelling they
  are refused, because the identity is immutable under every spelling, exactly as
  `SET n.id` is.
- Giving one `CREATE` both `id` and the type's own id column with *different*
  values is refused rather than resolved one way.

**Durable graphs refuse a duplicate id.** With `durable=` (or a durable
`Session`), a `CREATE` whose `(type, id)` already exists fails. The write-ahead
log names every entity by that pair, so a second node under a live id cannot be
recovered: replay would merge the two and lose one. Use `MERGE` to upsert, pick a
distinct id, or declare a `primary_key` to get the same enforcement in every
mode. Without a log, the documented opt-in behaviour above is unchanged.

For datasets whose ids are a prefix + number (Wikidata `Q42`, `P31`, …), the
loader stores the **integer** as `id` (compact, identical across modes; disk needs
it at 100M-node scale) and the **string form** as the `nid` property. Query by the
string form via `{nid: 'Q42'}` (or by the integer via `{id: 42}`). `{id: 'Q42'}`
does **not** match, because ids are integers. `n.id → 42` and `n.nid → 'Q42'`, in
every mode.

### System labels (`KgliteSkill`, `KgliteRecipe`)

Two labels are reserved for content a graph carries **about itself** rather than
about its subject:

- `KgliteSkill` holds markdown methodology for an agent.
- `KgliteRecipe` holds one named, parameterised, read-only query the graph
  publishes for reuse.

Both are written and read from Python (`set_skill` / `set_recipe` and their
`list` / `get` / `delete` / `import` / `export` siblings). The guides are
[Authoring MCP skills](https://kglite.readthedocs.io/en/latest/python/guides/mcp-skills.html)
and the `extensions.cypher_recipes` section of the
[MCP servers guide](https://kglite.readthedocs.io/en/latest/python/guides/mcp-servers.html).

To Cypher they are **ordinary nodes**:

```cypher
MATCH (s:KgliteSkill) RETURN s.name, s.description ORDER BY s.name
MATCH (r:KgliteRecipe {recipe: 'code_review'}) RETURN r.name, r.parameters.type
CREATE (:KgliteSkill {name: 'wells', description: 'How to ask about wells.', body: '# Wells'})
MATCH (s:KgliteSkill {name: 'wells'}) DETACH DELETE s
```

They are also **hidden from every surface that enumerates node types**, and the
counts printed beside those listings are filtered to match. Storing a skill
therefore never changes what the graph reports itself to be about:

| Surface | Sees a `KgliteSkill` node? |
|---|---|
| `MATCH (n) RETURN count(n)` | **yes** — it is a node |
| `MATCH (s:KgliteSkill)` and every other label-anchored pattern | **yes** |
| `CALL db.labels()` | no |
| `node_types()`, `schema()`, `describe()`, `graph_overview()` | no |
| `save` / `load`, export-text, export-sqlite, diffs, digests | **yes** — it is data and it travels |

The convention is **not enforced**. Nothing stops a `CREATE` from putting an
arbitrary shape under either label, which is exactly why the Python writers
validate and why an MCP server re-validates each record at boot and skips the
ones that fail. Nothing stops you using the labels for something else either, but
they will be invisible to schema discovery, so don't.

## Selected syntax summary

This compact table is a non-exhaustive orientation aid. The versioned,
machine-checked source of truth is the
[Cypher Dialect Contract](#cypher-dialect-contract) below. Do not infer absence
from this shorter list.

| Category | Supported |
|----------|-----------|
| **Clauses** | `MATCH`, `OPTIONAL MATCH`, `WHERE`, `FILTER`, `RETURN`, `FINISH`, `WITH`, `ORDER BY`, `SKIP`/`OFFSET`, `LIMIT`, `UNWIND`, `UNION`/`UNION ALL`, scoped and legacy `CALL { ... }` read subqueries, `CREATE`, `INSERT`, `SET`, `DELETE`/`NODETACH DELETE`/`DETACH DELETE`, `REMOVE`, `MERGE`, `EXPLAIN`, `PROFILE` |
| **Schema DDL** | `CREATE INDEX`, `CREATE RANGE INDEX`, `DROP INDEX`, `SHOW INDEXES`, `CREATE CONSTRAINT`, `DROP CONSTRAINT`, `SHOW CONSTRAINTS` — standalone statements; the two `SHOW` forms are reads |
| **Patterns** | Node `(n:Type)`, relationship `-[:REL]->`, abbreviated `-->` / `--` / `<--`, variable-length `*1..3`, undirected `-[:REL]-`, properties `{key: val, key: $param, key: var}` or any expression (`{id: row[0]}`, `{n: x + 1}`), `p = shortestPath(...)` |
| **WHERE** | `=`, `<>`, `<`, `>`, `<=`, `>=`, `=~` (regex, full-string), `AND`, `OR`, `NOT`, `IS NULL`, `IS NOT NULL`, `IN [...]`, `CONTAINS`, `STARTS WITH`, `ENDS WITH`, `EXISTS { pattern WHERE ... }`, `EXISTS(( pattern ))`, inline pattern predicates, `any/all/none/single(x IN list WHERE ...)` |
| **RETURN** | `n.prop`, `r.prop`, `AS` aliases, `DISTINCT`, arithmetic `+`/`-`/`*`/`/`, string concat `\|\|`, map projections `n {.prop}`, map literals `{k: expr}`, list slicing `[i..j]` |
| **Aggregation** | `count(*)`, `count(expr)`, `sum`, `avg`/`mean`, `min`, `max`, `collect`, `std` |
| **Expressions** | `CASE WHEN...THEN...ELSE...END`, `$param`, `[x IN list WHERE ... \| expr]`, `any/all/none/single(...)` |
| **Functions** | `toUpper`, `toLower`, `toString`, `toInteger`, `toFloat`, `size`, `length`, `type`, `id`, `labels`, `keys`, `coalesce`, `date`/`datetime`, `range`, `nodes(p)`, `relationships(p)`, `round` |
| **String** | `split`, `replace`, `substring`, `left`, `right`, `trim`, `ltrim`, `rtrim`, `reverse` |
| **Math** | `abs`, `ceil`/`ceiling`, `floor`, `round`, `sqrt`, `sign`, `log`/`ln`, `log10`, `exp`, `pow`, `pi`, `rand`, `randomUUID`, trig: `sin`/`cos`/`tan`/`asin`/`acos`/`atan`/`atan2`/`cot`/`haversin`/`degrees`/`radians` |
| **Spatial** | `point(lat, lon)`, `point({latitude, longitude})`, `distance(a, b)`, `contains(a, b)`, `intersects(a, b)`, `centroid(n)`, `area(n)`, `perimeter(n)`, `latitude(point)`, `longitude(point)` |
| **Temporal** | `date(str)`, `datetime(str)`, `localdatetime()` (timestamp values), `localtime()`/`time()` (ISO strings), `duration.between(d1, d2)`, `date_diff(d1, d2)`, `date ± N` (days), `date - date` → duration, `d.year`/`d.month`/`d.day`, `valid_at(...)`, `valid_during(...)` |
| **Semantic** | `text_score(n, prop, query [, metric] [, options])` — scores a list `query` as a vector, embeds a string `query` via `set_embedder()`, cosine/dot_product/euclidean/poincare; `embedding_norm(n, prop)` — L2 norm (hierarchy depth) |
| **Timeseries** | `ts_sum`, `ts_avg`, `ts_min`, `ts_max`, `ts_count`, `ts_at`, `ts_first`, `ts_last`, `ts_delta`, `ts_series` — date-string args with resolution validation |
| **Mutations** | `CREATE (n:Label {props})`, strict `INSERT (n IS Label&Other {props})`, `CREATE`/`INSERT` relationships, `SET n.prop = expr`, `SET n += map`, `SET n = map`, `DELETE`, `NODETACH DELETE`, `DETACH DELETE`, `REMOVE n.prop`, `MERGE ... ON CREATE SET ... ON MATCH SET` |
| **Procedures** | `CALL pagerank/betweenness/degree/closeness() YIELD node, score`, `CALL louvain/leiden() YIELD node, community [, level]` (multilevel, hierarchical — `leiden` guarantees well-connected communities), `CALL label_propagation() YIELD node, community`, `CALL connected_components() YIELD node, component`, `CALL k_core/coreness() YIELD node, coreness`, `CALL clustering_coefficient() YIELD node, coefficient`, `CALL cluster({method, ...}) YIELD node, cluster`, `CALL affected_tests({files: [...], max_depth?}) YIELD test_file, depth` (code graphs), `CALL refresh_stats() YIELD src_type, edge_type, tgt_type, count` (planner cardinality cache refresh), `CALL list_procedures()` |
| **Scoped algorithms** | `connected_components`, `k_core`/`coreness`, and `clustering_coefficient` accept an optional `{node_type, relationship}` map to run over a subgraph — e.g. `CALL k_core({node_type: 'Person', relationship: ['KNOWS', 'OWNS']})`. Each field is a string or list of strings; omit the map for the whole graph. Computed lazily over the live graph (identical across memory/mapped/disk modes). |
| **Schema** | `CALL db.labels() YIELD label`, `CALL db.relationshipTypes() YIELD relationshipType`, `CALL db.indexes() YIELD name, type, entityType, labelsOrTypes, properties, state` |
| **Rule procedures** | `CALL orphan_node/self_loop/missing_required_edge/missing_inbound_edge/duplicate_title/duplicate_id/null_property({type[,edge\|property]}) YIELD node`, `CALL cycle_2step({type, edge}) YIELD node_a, node_b`, `CALL inverse_violation({rel_a, rel_b}) YIELD a, b`, `CALL transitivity_violation({rel}) YIELD a, b, c`, `CALL cardinality_violation({type, edge[, min, max]}) YIELD node, count`, `CALL type_domain_violation/type_range_violation({edge, expected_*}) YIELD source, target`, `CALL parallel_edges({edge}) YIELD a, b, count`, `CALL edge_property_violation() YIELD relationship, check, source, target, property, properties, exempt` (ontology declarations only) |
| **Operators** | `+`, `-`, `*`, `/`, `\|\|` (string concat), `=~` (regex, full-string), `IN`, `STARTS WITH`, `ENDS WITH`, `CONTAINS`, `IS NULL`, `IS NOT NULL` |

## Cypher Dialect Contract

The machine-readable source of truth is
[`tests/api-baselines/cypher-dialect.json`](https://github.com/kkollsga/kglite/blob/main/tests/api-baselines/cypher-dialect.json).
KGLite is not a complete openCypher or Neo4j implementation. The statuses below
mean:

| Status | Meaning |
|---|---|
| `Covered` | The stated behavior is implemented and locked by local, independently authored tests. It is not a claim that every grammar production or edge case from an external conformance suite is implemented. |
| `Partial` | A useful implementation with a known gap. |
| `Extension` | Outside the claimed openCypher-compatible subset. |

### Clauses

| Clause | Status | Notes |
|--------|--------|-------|
| `MATCH` | Partial | Node, relationship, variable-length, shortest-path, abbreviated (`-->`, `--`, `<--`), and relationship-unique trail patterns; not every openCypher pattern grammar form is implemented |
| `OPTIONAL MATCH` | Covered | Null-extending optional patterns |
| `WHERE` | Covered | Predicates preserve three-valued boolean, membership, and quantifier semantics |
| `FILTER` | Covered | Standalone row filter; equivalent to `WITH * WHERE predicate` without changing the projected columns |
| `RETURN` | Covered | Aliases, `DISTINCT`, expressions, and map projections |
| `FINISH` | Covered | Terminal clause that preserves completed side effects and returns no rows |
| `WITH` | Covered | Projection, grouping, standalone `WITH *`, and strict post-projection scope |
| `ORDER BY` | Covered | Multi-column, `ASC`/`DESC`, fused top-k optimization |
| `SKIP` / `OFFSET` / `LIMIT` | Covered | `OFFSET` is a `SKIP` synonym |
| `UNWIND` | Covered | List expansion, works with `collect()` round-trips |
| `UNION` / `UNION ALL` | Covered | |
| `CREATE` | Covered | Nodes, relationships, inline properties |
| `INSERT` | Covered | Cypher 25 static insertion: node labels use `&`; relationship types are singular and directed. Dynamic labels/types, dynamic property maps, path assignment, colon-separated multiple labels, undirected/variable-length relationships, and relationship type alternation are rejected |
| `SET` | Covered | Property/label assignment plus `n += map` merge and `n = map` replacement |
| `DELETE` / `NODETACH DELETE` / `DETACH DELETE` | Covered | `NODETACH DELETE` is the explicit spelling of plain `DELETE`; both reject deletion of a node that still has relationships |
| `REMOVE` | Covered | Property and secondary-label removal |
| `MERGE` | Covered | `ON CREATE SET`, `ON MATCH SET`, and pre-mutation null-property rejection |
| `EXPLAIN` | Extension | KGLite-specific structured plan output |
| `PROFILE` | Extension | KGLite-specific per-clause execution statistics |
| `HAVING` | Extension | Post-aggregation filter on `RETURN`/`WITH` |
| `CALL ... YIELD` | Extension | Namespaced KGLite procedures plus `db.*` discovery procedures; ordinary procedures run per input row and inner-join their yielded columns. `cluster()` is the explicit set-input exception |
| `CALL { ... }` subqueries | Partial | Per-input-row read subqueries with legacy importing `WITH` or modern `CALL (x, y)` / `CALL (*)` / `CALL ()` scope. `UNION`/`UNION ALL` bodies are supported; `INTERSECT`/`EXCEPT` there are KGLite extensions. Writes, unit bodies, and `IN TRANSACTIONS` remain unsupported |
| `FOREACH` | Covered | Updating bodies, including nested `FOREACH` |
| `LOAD CSV` | Partial | `LOAD CSV [WITH HEADERS] FROM <source> AS row [FIELDTERMINATOR <sep>]` over `file://` URLs and local paths, leading position only. Streams in batches for row-local pipelines; `http(s)://` and `IN TRANSACTIONS` are not supported. See [LOAD CSV](#load-csv) |

### Expressions & Operators

| Feature | Status | Notes |
|---------|--------|-------|
| Arithmetic (`+`, `-`, `*`, `/`) | Covered | Numeric arithmetic plus list/list and element/list composition |
| String concat (`\|\|`) | Extension | Auto-converts non-strings |
| Comparison (`=`, `<>`, `<`, `>`, `<=`, `>=`) | Partial | Scalar comparisons, null propagation and cross-type ordering follow openCypher: `<`/`<=`/`>`/`>=` are `null` between values no ordering rule relates, `=` is `false` and `<>` is `true`. Three deliberate divergences: a date equals midnight on that date, a date or datetime compares with a string by parsing it (both under Sort order), and lists and maps have no ordering rule of their own, so `[1] < [2]` is `null` rather than element-wise |
| Boolean (`AND`, `OR`, `XOR`, `NOT`) | Covered | Predicate and expression positions preserve three-valued results |
| `IS NULL` / `IS NOT NULL` | Covered | Also works as expressions in RETURN/WITH |
| `IN [list]` | Covered | Null operands and null-containing no-match lists preserve unknown; an **empty** list is `false` for every operand, `null` included |
| `CONTAINS` / `STARTS WITH` / `ENDS WITH` | Covered | |
| `=~` regex | Covered | Full-string match, per openCypher: the pattern must match the entire value, so `'inactive' =~ 'active'` is false. Wrap with `.*` to search, or use `text_match_regex()`, which is a search by design. Shares a process-wide FIFO cache with `text_match_regex()` (128 entries; 2 MiB compiled-program limit; misses compile outside the lock) |
| `CASE WHEN...THEN...ELSE...END` | Covered | Simple and generic forms |
| Parameter references (`$param`) | Covered | In WHERE, pattern properties, and expressions |
| List comprehensions (`[x IN list WHERE ... \| expr]`) | Covered | |
| List slicing (`expr[start..end]`) | Covered | Open-ended, negative indices |
| List quantifiers (`any/all/none/single(x IN list WHERE ...)`) | Covered | Decisive results short-circuit; otherwise unknown propagates |
| `EXISTS { pattern WHERE ... }` | Covered | Brace `{}`, parenthesis `(( ))`, inline pattern, with WHERE |
| Map projections (`n {.prop1, .prop2}`) | Covered | |
| Map literals (`{key: expr}`) | Covered | |
| Variable binding in pattern properties | Covered | `WITH val AS x MATCH ({prop: x})` |
| Window functions (`OVER`) | Extension | `row_number()`, `rank()`, `dense_rank()` with `PARTITION BY`/`ORDER BY` |

### Scalar & Aggregation Functions

| Function | Status | Notes |
|----------|--------|-------|
| `count(*)`, `count(expr)` | Covered | With `DISTINCT` support |
| `sum`, `avg`/`mean`, `min`, `max` | Covered | |
| `collect` | Covered | |
| `std` | Extension | Standard deviation helper |
| `toUpper`, `toLower`, `toString` | Covered | |
| `toInteger`, `toFloat` | Intentional divergence | `toFloat` is openCypher; `toInteger` of a *string* requires an integer spelling and yields null for `'3.7'`, where Neo4j truncates (see Built-in Functions). Numeric arguments truncate as in Neo4j |
| `size`, `length` | Covered | Strings (in characters, not bytes), lists, and paths |
| `type(r)` | Covered | Returns relationship type |
| `id(entity)` | Covered | KGLite logical node identity and stable relationship identity |
| `labels(n)` | Intentional divergence | Primary type first, then secondary labels |
| `keys(n)` / `keys(r)` / `keys(map)` | Covered | Returns property names, sorted; the map form covers `keys(properties(n))` |
| `date(str)` / `datetime(str)` | Partial | KGLite's temporal value model and documented arithmetic subset; a date and a datetime are intercomparable **and equatable**, a date being midnight on that date |
| `coalesce` | Covered | |
| `range(start, end [, step])` | Covered | Inclusive integer range |
| `round(x [, precision])` | Covered | |
| `nodes(p)`, `relationships(p)` | Covered | Exact node order, relationship identity, properties, and traversal direction are preserved for parallel and incoming paths |
| String functions | Covered | `split`, `replace`, `substring`, `left`, `right`, `trim`, `ltrim`, `rtrim`, `reverse`. All are character-indexed; `split` with an empty delimiter is a documented divergence (see String Functions) |
| Math functions | Intentional divergence | `abs`, `ceil`, `floor`, `sqrt`, `sign`, `log`/`ln`, `log10`, `exp`, `pow`, `pi`, `rand`, `randomUUID`. Undefined real-number results (`sqrt(-1)`, `log(0)`, `1.0/0.0`) are null rather than Neo4j's non-finite float; KGLite's Float64 value can otherwise represent NaN and infinities (see Math Functions) |
| Trig functions | Covered | `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `atan2(y,x)`, `cot`, `haversin`, `degrees`, `radians` |
| Spatial functions | Extension | KGLite's pragmatic `point`, geometry, containment, and distance model |
| Temporal functions | Extension | `valid_at`, `valid_during`, and KGLite temporal helpers |

### Extensions and canonical names

Custom callable features have canonical `kglite.*` spellings. Historical flat
names remain accepted and execute the same AST, so existing applications keep all
functionality, while new integrations can distinguish extensions from the
compatible subset.

- `kglite.pagerank and related procedures` (legacy: flat procedure names)
- `kglite.text_score / kglite.vector_score / kglite.score_fuse` (legacy: flat function names)
- `kglite.geom_* and KGLite spatial helpers`
- `kglite.ts_*`
- `kglite.text_*`
- `HAVING` and window functions using `OVER`
- `INTERSECT / EXCEPT`
- `EXPLAIN / PROFILE`

### Architectural Differences from Neo4j

| Feature | KGLite | Neo4j | Rationale |
|---------|--------|-------|-----------|
| Labels per node | One primary type + secondary labels | Multiple equal labels | Primary type drives indexing (`type_indices`); secondary labels are additive (0.10.5+) |
| `labels(n)` return type | `List[String]` (primary first) | `List[String]` | Matches Neo4j since 0.10.5 |
| `SET n:Label` | Supported (adds a secondary label) | Supported | Primary type is immutable; changing it requires node migration/recreation |
| Storage | In-memory, mmap-backed, or disk CSR | Disk-based | One Cypher engine spans all three embedded storage modes |
| Transactions | Snapshot isolation + OCC through `Session` / `Transaction`; durability depends on access mode | Server/embedded transaction management | Native session coordination is binding-independent; direct graph writes are in-place |
| Indexing | Three separate structures — hash equality, composite, B-tree range — plus automatic type indexes and vector indexes | One general `RANGE` index serving equality, range, and ordering | An equality index cannot serve a range predicate, so KGLite exposes the distinction that Neo4j collapses. `CREATE INDEX` / `DROP INDEX` / `SHOW INDEXES` are supported — see [Cypher index DDL](#cypher-index-ddl) for exactly what each statement builds |
| Index names | Canonical and derived: `Label.property`, `Label.(a,b)` | User-assigned, unique | A name in `CREATE INDEX <name> …` is accepted for script portability but not stored; the persisted `.kgl` index state is a list of `(label, property)` keys |
| Constraint DDL | `IS UNIQUE`, `IS NOT NULL`, `IS NODE KEY`, `IS :: TYPE` — enforced on every write path | `CREATE CONSTRAINT … IS UNIQUE / IS NOT NULL / IS NODE KEY / IS :: TYPE` | `IS :: TYPE` accepts the type names with an exact KGLite value counterpart (`BOOLEAN`, `STRING`, `INTEGER`, `FLOAT`, `DATE`, `LOCAL DATETIME`, `DURATION`, `POINT`); lists, unions and zoned temporal types are rejected by name rather than approximated. Relationship constraints cover `IS NOT NULL` and `IS :: TYPE`; `IS UNIQUE` / `IS RELATIONSHIP KEY` on a relationship are refused, because KGLite has no single answer for when two relationships of a type are the same one. See [Cypher constraint DDL](#cypher-constraint-ddl) |
| Constraint names | Stored, so `DROP CONSTRAINT <name>` works | User-assigned, unique per database | The opposite decision to index names above: a ported schema script almost always drops constraints by name, and the `.kgl` metadata section is JSON, so the field was free. A constraint declared without a name is addressable by its canonical descriptor |
| `LOAD CSV` source | Local files only — `file://` URLs and filesystem paths, gated by a per-caller capability | `file://` plus `http(s)://`, gated by an import-directory setting | The engine ships no HTTP client (network dependencies were removed in 0.14.x), so there is nothing to fetch a URL with. Filesystem access is granted per caller: on for in-process use, off for Bolt clients unless the server was started with `--allow-csv-import <DIR>` |


### Vector retrieval diagnostics

Ordinary query results and `PROFILE` carry `diagnostics.retrieval`: distinct
executed vector ranking routes. Each record has:

- `requested_policy`: `auto`, `exact`, or `per_row`.
- `actual_mode`: `hnsw` or `exact`.
- `fallback_reason`: reasons include `forced_exact`, `no_index`, `stale_index`, `metric_mismatch`,
  `filtered_underfill`, `row_coverage`, `row_dependent_selectors` and
  `ordering_requires_exact`.
- An optional `store`: `Type.embedding_property`, or
  `relationship:TYPE.embedding_property` for a relationship store. A missing store
  means that execution did not establish one common store.

Identical nested routes are coalesced; these records are not counters.
`EXPLAIN` describes the requested policy only, without running search or claiming
an actual route.

Empty inputs and `LIMIT 0` record no retrieval. Scalar scoring inside arbitrary
expressions remains exact and has no per-row telemetry. The surfaces expose the
records as follows:

- Python exposes them in `ResultView.diagnostics`.
- MCP appends them to result text and includes them in recipe diagnostics.
- C exposes `kglite_cypher_result_diagnostics_json` and batch diagnostics.
- Bolt includes `kglite.retrieval` in result summaries.
