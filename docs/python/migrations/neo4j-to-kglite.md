# Migrating from Neo4j to KGLite

This page is for a developer with an existing Neo4j database and/or
`neo4j`-driver code who wants to evaluate or adopt KGLite. It covers
where KGLite fits, the two migration paths (Bolt drop-in vs native
Python), how to lift your data across, and — the core of the guide —
where the Cypher dialect diverges.

KGLite ships a focused openCypher subset, **not a Neo4j drop-in
replacement**. Most read queries port unchanged; the divergences
below are the ones worth knowing before you commit.

## When KGLite fits — and when it doesn't

| | KGLite | Neo4j |
|---|---|---|
| Deployment | Embedded, in-process (`pip install kglite`) | Server (JVM) or embedded driver |
| Query language | Cypher (subset, see below) | Cypher (full) |
| Storage | `.kgl` file — in-mem · mmap · disk | Server store directory |
| Auth | None in-process; basic only via Bolt server | Full RBAC |
| Multi-database | No — one graph per process / per server | Yes (`USE db`) |
| Clustering / routing | No (single server) | Causal cluster, routing |
| Transactions | Snapshot isolation + OCC | Full ACID |
| Data model | One **primary** type per node + optional secondary labels | Arbitrary label sets |

**KGLite fits** when you want Cypher + Python ergonomics in one wheel:
analytics over a graph that fits on one machine, embedding a graph in
a Python app or notebook, shipping a queryable `.kgl` artifact, or
serving a read-mostly graph to LLM agents (KGLite bundles an MCP
server and a `describe()` schema). See the
[README comparison table](https://github.com/kkollsga/kglite#how-it-compares)
for the side-by-side against Kuzu / NetworkX / rustworkx / Neo4j
Embedded.

**KGLite does not fit** when you need server-mode RBAC, multiple
databases per instance, a causal cluster with routing, or full ACID
across long-lived multi-client write sessions. Those are Neo4j's
domain — KGLite is deliberately single-graph and embedded.

For positioning detail see
{doc}`../core-concepts` and the
[concepts index](../../concepts/index.md).

## Two migration paths

### Path A — Bolt server (drop-in for driver code)

`kglite-bolt-server` is a pure-Rust binary that speaks the
[Bolt v5 wire protocol](https://neo4j.com/docs/bolt/current/). Any
Neo4j-aware client — the official Python/JS/Java/Go drivers, Cypher
Shell, Neo4j Browser, LangChain's `Neo4jGraph` — connects with **no
consumer-side code changes** beyond the connection URL. See the
[Bolt server operator guide](../../operators/bolt-server.md).

```bash
cargo install kglite-bolt-server
kglite-bolt-server --graph my-graph.kgl --bind 127.0.0.1 --port 7687
```

Your driver code stays almost identical — just re-point the URI:

```python
# Before — against Neo4j
from neo4j import GraphDatabase
driver = GraphDatabase.driver("neo4j://prod-db:7687", auth=("neo4j", "secret"))

# After — against kglite-bolt-server
from neo4j import GraphDatabase
driver = GraphDatabase.driver("bolt://127.0.0.1:7687", auth=None)

with driver.session() as session:
    result = session.run(
        "MATCH (p:Person)-[:KNOWS]->(f) WHERE p.age > $min RETURN f.name",
        min=30,
    )
    for record in result:
        print(record["f.name"])
```

The query path is the *same* Cypher engine the Python API uses;
differential tests confirm row-for-row equivalence
(`tests/test_bolt_server_differential.py`). The Python driver is the
only client with automated regression coverage — other drivers use
the same Bolt v5 protocol and should work, but exercise them manually
first.

#### What carries over, and what does not

| Neo4j feature | Bolt server status |
|---|---|
| `bolt://` direct connections | Supported — **use this** |
| `neo4j://` routing URIs | Single-server routing table only; set `--advertise-addr` for reverse-proxy deployments. No real cluster. |
| Auth | `--auth basic` with `--auth-user` / `--auth-pass`; default `--auth none` accepts any LOGON. No RBAC, users, or roles. |
| TLS (`bolt+s://` / `neo4j+s://`) | Supported via `--tls-cert` + `--tls-key` |
| Read-only enforcement | `--readonly` rejects all mutations |
| Auto-commit **mutations** | **Not supported** — wrap `CREATE`/`SET`/`DELETE`/`MERGE` in explicit `BEGIN`/`COMMIT`. Auto-commit reads work. (Drivers wrap writes in a tx anyway.) |
| OCC on writes | Supported — stale-snapshot commits get `Neo.ClientError.Transaction.ConflictDetected`; retry client-side |
| Multi-database (`USE db`) | **Not supported** — single graph; `USE` is accepted but ignored |
| Causal consistency / bookmarks | **Not supported** — the `bookmark` field is not returned on COMMIT |
| Multi-statement queries (`;`-separated) | **Not supported** — one statement per `session.run`, or group with `BEGIN`/`COMMIT` |
| `db.labels()` / `db.relationshipTypes()` | Yield `label` / `relationshipType` (Neo4j-conventional names) over Bolt |

> The in-process Python API also exposes `kglite.to_neo4j(graph, uri, ...)`
> if you want to push a KGLite graph *into* a real Neo4j instance
> (batched `UNWIND`, optional `merge=True` upsert).

### Path B — native Python (`cypher()` directly)

If you control the calling code, skip the wire protocol entirely and
call `cypher()` in-process. No server, no socket, no driver — the
result is a `ResultView` you iterate, index, or convert with
`to_df=True`. See {doc}`../getting-started`.

The same query, both ways:

```python
# Bolt path — neo4j driver
with driver.session() as session:
    rows = list(session.run(
        "MATCH (p:Person) WHERE p.age > $min RETURN p.name AS name",
        min=30,
    ))

# Native path — kglite in-process
import kglite
graph = kglite.load("my-graph.kgl")
rows = list(graph.cypher(
    "MATCH (p:Person) WHERE p.age > $min RETURN p.name AS name",
    params={"min": 30},
))
```

Note the parameter syntax difference: the driver takes `**kwargs`
(or a `parameters=` dict); `cypher()` takes a `params=` dict. The
`$name` placeholders in the query string are identical.

## Getting data out of Neo4j into KGLite

The practical recipe: query Neo4j with the driver, pull rows into a
pandas DataFrame, and bulk-load with `add_nodes` / `add_connections`.

```python
import pandas as pd
import kglite
from neo4j import GraphDatabase

src = GraphDatabase.driver("neo4j://prod-db:7687", auth=("neo4j", "secret"))
graph = kglite.KnowledgeGraph()

# Nodes
with src.session() as s:
    people = pd.DataFrame([
        dict(r["p"]) for r in s.run("MATCH (p:Person) RETURN p")
    ])
graph.add_nodes(people, node_type="Person", unique_id_field="id",
                node_title_field="name")

# Relationships — return the endpoint ids, not the whole nodes
with src.session() as s:
    knows = pd.DataFrame([
        {"src": r["a"], "tgt": r["b"]}
        for r in s.run("MATCH (a:Person)-[:KNOWS]->(b:Person) "
                       "RETURN a.id AS a, b.id AS b")
    ])
graph.add_connections(knows, connection_type="KNOWS",
                      source_type="Person", source_id_field="src",
                      target_type="Person", target_id_field="tgt")

graph.save("my-graph.kgl")
```

`add_nodes` auto-detects string vs integer ids from the column dtype
and supports a `column_types=` override for spatial/temporal columns;
`add_connections` can take a Cypher `query=` instead of a DataFrame.
See the [data-loading guide](../guides/data-loading.md).

**Alternate route — APOC CSV export.** For large graphs, export from
Neo4j with `apoc.export.csv.all('graph.csv', {})` (or per-label
queries), then `pd.read_csv` → `add_nodes` / `add_connections`. KGLite
has no `LOAD CSV` — by design, pandas/`csv` give you better control
over typing and cleanup.

## Cypher dialect divergence

The tables below are the heart of the guide. KGLite's supported
surface is documented in full in
[CYPHER.md](https://github.com/kkollsga/kglite/blob/main/CYPHER.md);
this section lists only where it diverges from Neo4j. Conformance is
spot-checked against a live Neo4j via `scripts/cypher_conformance.py`
(see {doc}`../../concepts/cypher-conformance`).

### Data model — labels and node identity

| Neo4j form | KGLite status | Workaround / note |
|---|---|---|
| Arbitrary label sets | One **primary** type + optional secondary labels (since 0.10.5) | `CREATE (n:A:B)`, `SET n:B`, `REMOVE n:B`, `MATCH (n:A:B)` all work; `labels(n)` returns a **list**, primary first |
| Retype a node by swapping labels | Primary type is immutable via label ops | `SET n.type = 'NewType'`; `REMOVE n:Primary` errors deliberately |
| Per-row label assignment at load | `add_nodes(labels=[...])` applies uniform secondary labels to the batch | `g.add_label(node_type, ids, label)` for batches after load |
| `id(n)` returns an internal integer | `id(n)` / `n.id` returns the node's **identity** | See identity note below |

> **Note:** Neo4j docs and some older KGLite material describe KGLite
> as "single-label" with `labels(n)` returning a string. That changed
> in 0.10.5 — multi-label is native and `labels(n)` returns a list.

#### Node identity (`id`) — the 0.10.10 model

As of 0.10.10, `n.id` is the node's **unique identity** and behaves
identically in every storage mode.

| Aspect | KGLite behaviour |
|---|---|
| `CREATE (n {id: X})` | Honours `X` as the identity (string / int / float; survives save → load) |
| Prefixed-id datasets (Wikidata `Q42`) | Loader stores the **integer** as `id` (`n.id == 42`) and the string form as the `nid` property (`n.nid == 'Q42'`) |
| Lookup by string id | `{nid: 'Q42'}` (a plain indexed string-property lookup); `{id: 'Q42'}` does **not** match — ids are integers |
| Duplicate ids | `MATCH (n {id: X})` returns one node per id; a rate-limited warning is emitted at index build. Use `MERGE` or dedupe input. |

This is a **breaking** change from earlier releases for prefixed-id
data — see the
[0.10.10 CHANGELOG entry](https://github.com/kkollsga/kglite/blob/main/CHANGELOG.md).

### Missing language constructs

Verified absent against 0.10.14:

| Neo4j construct | KGLite status | Workaround |
|---|---|---|
| `FOREACH (x IN list \| ...)` | Not supported | `UNWIND list AS x` then `CREATE`/`SET` |
| `CALL { ... CREATE/SET/DELETE ... }` (writes in body) | Not supported (v1) | Do writes in a separate top-level clause; read subqueries **are** supported (see below) |
| `CALL { ... UNION ... }` (UNION inside body) | Not supported (v1) | Top-level `UNION`, or combine separate `cypher()` results |
| Unit `CALL { ... }` (no terminal `RETURN`) | Not supported (v1) | Body must end in `RETURN` |
| `CALL { ... } IN TRANSACTIONS` | Not supported | Server batching; no in-memory analogue |
| Pattern comprehensions `[(n)-->(m) \| m]` | Not supported | `MATCH`/`OPTIONAL MATCH` + `collect()` |
| Quantified path patterns `((a)-->(b))+` | Not supported | Variable-length paths `-[:R*1..3]->` (supported) |
| `allShortestPaths(...)` | Not supported | `shortestPath(...)` (supported) returns one path |
| `LOAD CSV` | Not supported (by design) | pandas / `csv` → `add_nodes` |
| `exists(n.prop)` (property existence) | Not supported | `WHERE n.prop IS NOT NULL` / `IS NULL` |
| `exists((pattern))` in `RETURN` | Not supported as a `RETURN` expression | `EXISTS { pattern }` / inline pattern predicate in `WHERE` |
| `CREATE INDEX FOR (n:L) ON ...` (DDL) | Not supported | Python `graph.create_index(type, prop)` / `create_range_index(...)` |

### Constructs that DO work (worth confirming)

These port unchanged from Neo4j and are easy to assume missing:

- `MERGE ... ON CREATE SET ... ON MATCH SET` — match-or-create.
- Variable-length paths `-[:KNOWS*1..3]->`, `shortestPath(...)`.
- `WHERE EXISTS { pattern WHERE ... }` (pattern-existence), inline
  pattern predicates, `any/all/none/single(x IN list WHERE ...)`.
- `CALL { ... }` **read** subqueries — both uncorrelated (`CALL {
  MATCH ... RETURN ... }`, cartesian-combined with the outer rows)
  and correlated (`CALL { WITH p MATCH (p)-->... RETURN ... }`, run
  per outer row). The importing `WITH` lists **bare variables only**.
  Aggregating bodies preserve the outer row with a zero value; non-
  aggregating bodies inner-join (zero matches drops the row). v1
  caveats: no writes / `UNION` / unit subqueries in the body, no
  `IN TRANSACTIONS`. See
  [CYPHER.md → `CALL { ... }` Subqueries](https://github.com/kkollsga/kglite/blob/main/CYPHER.md#call----subqueries).
- List comprehensions `[x IN list WHERE p \| expr]`, `reduce(...)`,
  list slicing `xs[1..3]`, map projections `n {.a, .b}`, map literals.
- Map subscript `m['key']` and **dynamic property access** `n[key]`
  where `key` is a variable.
- Window functions `row_number()/rank()/dense_rank() OVER (...)`,
  `UNION`/`INTERSECT`/`EXCEPT`, `HAVING`.

### Recently added functions (new — verify your version ≥ 0.10.x)

These work today and may not appear in older comparison material:

| Function | Form |
|---|---|
| Trig family | `sin`/`cos`/`tan`/`asin`/`acos`/`atan`/`cot`/`haversin`/`degrees`/`radians` (radians) |
| `atan2(y, x)` | Quadrant-aware arctangent |
| `randomUUID()` | RFC 4122 v4 UUID string |
| `localdatetime()` / `localtime()` / `time()` | Return ISO-8601 **strings** (`Value::DateTime` is date-only — see note) |
| `m['key']` | Map subscript |
| `n[key]` | Dynamic property access (variable key) |

> `localdatetime()`/`localtime()`/`time()` return strings, not a
> temporal Value, because KGLite's `Value::DateTime` carries no
> time-of-day component. The 1-arg form validates/normalises a string
> and returns `NULL` on bad input.

## Function coverage

KGLite covers the common scalar / string / math / aggregation /
temporal / spatial families. Rather than duplicate them, see the
function tables in
[CYPHER.md](https://github.com/kkollsga/kglite/blob/main/CYPHER.md)
(Built-in, String, Math, Spatial, Temporal, Timeseries, Text
predicates, plus the openCypher compatibility matrix).

Notable **absent** functions a Neo4j user will miss (verified against
0.10.14):

| Neo4j | KGLite status | Note / workaround |
|---|---|---|
| `apoc.*` (all) | Not supported | No APOC library; use Python or built-in functions |
| `point({latitude, longitude})` | Map form not supported | KGLite uses `point(lat, lon)` (**latitude-first**); WKT strings are longitude-first per OGC |
| `point.distance(a, b)` | Use top-level `distance(a, b)` | Geodesic (WGS84); also `contains`, `intersects`, `centroid`, `area`, `perimeter`, geometry primitives (`geom_*`) — all present |
| `duration('P1Y2M')` (ISO-8601) | Map form only | `duration({years: 1, months: 2})`; `duration.between(d1, d2)` fills `days` only (date-only `DateTime`) |
| `timestamp()` | Not supported | `datetime()` (date-only); `localdatetime()` for a wall-clock string |
| `toBoolean(...)` | Not supported | `CASE` / Python-side coercion |
| Calendar-aware month diffs | Approximated (months ≈ 30 days in `DateTime ± Duration`) | Use literal dates for exact month arithmetic — see CYPHER.md "Duration semantics" |

KGLite also adds functions Neo4j lacks — semantic search
(`text_score`/`vector_score`), timeseries (`ts_*`), fuzzy text
predicates (`text_edit_distance`, `text_jaccard`), and graph-algorithm
procedures (`CALL pagerank/louvain/...`). See CYPHER.md.

### `EXPLAIN` / `PROFILE`

Both are supported but the shape differs from Neo4j's plan tree:

- `EXPLAIN <query>` returns a `ResultView` with rows
  `[step, operation, estimated_rows]` — a flat, ordered step list, not
  a nested operator tree.
- `PROFILE <query>` executes the query (you get the real results) and
  attaches per-clause stats on `result.profile`
  (`[clause, rows_in, rows_out, elapsed_us]`).

Every `cypher()` call also attaches lightweight `result.diagnostics`
(`elapsed_ms`, `timed_out`, `timeout_ms`) with no prefix required.

## Operational differences

| Concern | Neo4j | KGLite |
|---|---|---|
| Persistence | Live server store directory | A `.kgl` file; explicit `graph.save(path)` / `kglite.load(path)` |
| Backup | `neo4j-admin dump` / online backup | Copy the `.kgl` file (or `save_subset(path)` for a slice) |
| Concurrency | Server-managed sessions, ACID | Reads parallelize (GIL released via `py.detach()`); mutations serialize via copy-on-write; OCC on transactions |
| Cross-process access | Native (server) | Embedded — use the Bolt server as the coordination point for multi-process |
| Schema DDL | `CREATE INDEX` / `CREATE CONSTRAINT` Cypher | No DDL Cypher. Programmatic only: `create_index(type, prop)`, `create_range_index(...)`, `list_indexes()`, `drop_index(...)`. Type indices are automatic. |
| Migrations | Versioned migration tools | None — you own schema evolution in Python load code |

Indexes are maintained automatically across Cypher mutations
(`CREATE`/`SET`/`REMOVE`/`DELETE`/`MERGE`). On disk-backed graphs
property indexes are persisted next to the store; on in-memory graphs
they live in a HashMap. See the Indexes section of
[CYPHER.md](https://github.com/kkollsga/kglite/blob/main/CYPHER.md).

For the transaction model (snapshot isolation, OCC, last-writer-wins,
per-call cost) see {doc}`../transactions`; for the concurrency
contract see {doc}`../../concepts/concurrency`.

## See also

- {doc}`../getting-started` — install, build a graph, run Cypher.
- {doc}`../core-concepts` — nodes, relationships, storage modes.
- {doc}`../transactions` — `begin()` / `commit()` / OCC.
- [Bolt server operator guide](../../operators/bolt-server.md).
- {doc}`../../concepts/cypher-conformance` — how the Neo4j oracle works.
- [CYPHER.md](https://github.com/kkollsga/kglite/blob/main/CYPHER.md) — full supported Cypher reference.
