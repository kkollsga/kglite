# Import and Export

## Saving and Loading

```python
graph.save("my_graph.kgl")          # atomic (temp + rename) + fsync by default
graph.save("my_graph.kgl", fsync=False)   # skip the flush for speed (still atomic)
loaded_graph = kglite.load("my_graph.kgl")
```

`save()` is **atomic and crash-safe**: it writes to a sibling temp file and
atomically renames it over the target, so a crash mid-save can't leave a torn
`.kgl` — a reader always sees the old file or the complete new one. With
`fsync=True` (default) the file + directory are flushed before returning; pass
`fsync=False` to skip that for speed. `load()` raises a typed
`kglite.FileFormatError` on a corrupt file (see [Threading](#threading) and the
{doc}`durable apps guide </python/guides/durable-apps>`).

Save files (`.kgl`) use an explicitly versioned binary container. Current
files use RGF v6, an explicit Postcard codec tag, and core-data version 3.
Current readers accept RGF v6 and RGF v5. RGF v4/bincode and older containers
are rejected with an explicit migration/rebuild instruction; use kglite 0.13.4
as the conversion bridge described in the
[0.13 → 0.14 migration guide](../migrations/0.13-to-0.14.md#convert-persisted-data-before-upgrading).
A `.kgl` is the only complete KGLite backup because portable exports
intentionally omit some engine-specific state.

### `open()` — load-or-create lifecycle

For an app that persists to one file, `kglite.open(path)` is the ergonomic
entry point: it loads the graph if the file exists and creates a fresh one if
it doesn't, and the returned graph **remembers the path**.

```python
g = kglite.open("app.kgl")          # loads if present, else creates
g.cypher("CREATE (:Person {name: 'Alice'})")
g.save()                             # no path needed — writes back to app.kgl
```

Use it as a context manager to auto-save on clean exit:

```python
with kglite.open("app.kgl") as g:
    g.cypher("CREATE (:Person {name: 'Bob'})")
# snapshotted to app.kgl on block exit
```

- `save()` with no argument writes to the remembered path; passing a path
  (`save("other.kgl")`) updates the remembered target ("save as"). A graph built
  in memory with no path raises `ValueError` if you call `save()` with no path.
- `kglite.load(path)` also remembers its path, so bare `save()` works after a load.
- The context manager **skips the save if the block raised** — the on-disk file
  keeps its last good state. `close()` persists explicitly.

> **Auto-save-on-close is not what makes this crash-safe.** The clean-exit
> checkpoint writes nothing on a hard crash (`kill -9`, power loss). Crash
> safety comes from the write-ahead log, which is on by default — see below.

### Crash-safe writes (write-ahead log, on by default)

`open()` makes every committed mutation survive a hard crash. Each mutation is
appended to a `<path>-wal` sidecar and `fsync`'d **before the call returns**; on
open, any WAL frames are replayed onto the loaded checkpoint to recover work
committed since the last `save()`.

```python
with kglite.open("app.kgl") as g:
    g.cypher("CREATE (:Person {id: 1, name: 'Alice'})")
    # committed + fsync'd to app.kgl-wal here — survives kill -9

# A later run recovers automatically, even after a crash with no save():
g = kglite.open("app.kgl")
g.cypher("MATCH (p:Person) RETURN p.name")   # -> Alice
```

- `save()` writes a full `.kgl` checkpoint and truncates the WAL. A durable
  graph that was *never* saved still recovers entirely from its WAL.
- The log is idempotent (identity-keyed upsert/remove ops, per-frame CRC), so a
  torn trailing frame from a crash mid-append is discarded and recovery is safe.
- Supported for the in-memory default and `storage="mapped"`.
  `storage="disk"` opens non-durable (its commit boundary is a generation
  publish, not a log) and uses explicit-`save()` checkpoints.
- **`durable=False` opts out** of logging entirely — the right choice for bulk
  loading and for graphs rebuildable from source data. If you want to keep the
  log but not the per-commit barrier, `durable="normal"` costs roughly what an
  unlogged write costs and still loses nothing to a crashing process. Reads
  never pay for the capture path at any level.

## Export Formats

```python
graph.export('my_graph.graphml', format='graphml')  # Gephi, yEd
graph.export('my_graph.gexf', format='gexf')        # Gephi native
graph.export('my_graph.json', format='d3')           # D3.js
graph.export('my_graph.csv', format='csv')           # creates _nodes.csv + _edges.csv
graph.export('my_graph.sql', format='sqlite')        # SQLite SQL script

graphml_string = graph.export_string(format='graphml')
```

The format is inferred from the extension when you omit it, so
`graph.export('out.sql')` is enough.

## Export to SQLite — the no-lock-in exit

Your data should never be trapped in KGLite. `format='sqlite'` writes a
**SQLite-dialect SQL script** that the stock `sqlite3` CLI turns into a real,
queryable relational database:

```python
graph.export('dump.sql')
```

```bash
sqlite3 mygraph.db < dump.sql
sqlite3 mygraph.db "SELECT count(*) FROM Person"
```

Or straight from the command line, no Python involved:

```bash
kglite export-sqlite mygraph.kgl dump.sql
kglite export-sqlite mygraph.kgl | sqlite3 mygraph.db   # or pipe it
```

**The mapping.** Each node type becomes a table with `id`, `title`, and one
column per property the type uses. Each connection type becomes a link table
with `source_type`, `source_id`, `target_type`, `target_id`, and one column per
edge property. So the graph is queryable as ordinary SQL joins:

```sql
SELECT p.title, c.title, w.since
FROM WORKS_AT w
JOIN Person  p ON p.id = w.source_id
JOIN Company c ON c.id = w.target_id;
```

**Why a script and not a `.db` file.** Writing a `.db` directly would mean
linking a SQLite C library into KGLite. A text dump reaches the same
destination with **zero added dependencies**, and it is also diffable,
greppable, and ingestible by Postgres/DuckDB/MySQL after minor edits. Keeping
the dependency out is worth one extra `sqlite3` invocation.

**What to expect from the translation.** Graphs and relational tables do not
model everything the same way, so a few choices are worth knowing:

| Aspect | Behaviour | Why |
|---|---|---|
| `id` columns | Indexed, **not** `PRIMARY KEY` | KGLite warns about duplicate ids rather than rejecting them, so a graph may legitimately hold two nodes of a type sharing an id. A primary key would abort the ingest halfway. |
| Link tables | No foreign keys | They reference `(type, id)` pairs across several node tables, which SQL foreign keys cannot express. |
| Endpoint types | Stored as columns | One connection type can join several type pairs, so the endpoint type is data, not schema. |
| Booleans | `INTEGER` 0/1 | SQLite has no boolean type. |
| Missing properties | `NULL` | Distinguishable from a genuine empty string. |
| Floats | Full round-trip precision | Non-finite values (`NaN`, `±Inf`) become `NULL`, which SQLite cannot represent. |
| Mixed-type columns | `TEXT` | Column affinity is inferred from the values present; `INTEGER` widens to `REAL`, anything mixed collapses to `TEXT`. |
| Points, durations, lists, maps | JSON text | No relational counterpart; JSON keeps them readable rather than pretending they are native columns. |
| `updated_at` / `git_sha` / `modified_by` | Omitted | Engine write-provenance metadata, not your data. |

Output is deterministic — the same graph always produces byte-identical SQL —
so a dump can be committed and diffed.

### Parquet

KGLite does **not** export Parquet directly, and this is a deliberate scope
decision rather than a gap. Doing it in Rust means taking on the
`arrow` + `parquet` dependency tree — measured at **+35 crates** on top of the
CLI's 143 for a minimal Arrow-backed writer, and +336 lines of `Cargo.lock`.
KGLite has been steadily *shedding* dependencies (309 → 171 across 0.14.x), and
`arrow` in particular is a library this project already keeps at arm's length:
kglite pins its bundled allocator to mimalloc v2 specifically to survive being
imported alongside `pyarrow`.

Parquet is a format you can reach with tools you already have. Pick whichever
end of the pipe you prefer.

**One table, from a query.** The dependency lives in your environment, where
you control it:

```python
graph.cypher("MATCH (p:Person) RETURN p.id, p.title, p.age").to_df().to_parquet('people.parquet')
```

**The whole graph, no Python at all.** The SQLite dump above already turns
every node type and connection type into a table, and DuckDB writes Parquet
from those tables directly:

```bash
kglite export-sqlite mygraph.kgl dump.sql
sqlite3 mygraph.db < dump.sql
duckdb -c "INSTALL sqlite; LOAD sqlite; ATTACH 'mygraph.db' AS g (TYPE sqlite);
           COPY (SELECT * FROM g.Person)   TO 'Person.parquet'   (FORMAT PARQUET);
           COPY (SELECT * FROM g.WORKS_AT) TO 'WORKS_AT.parquet' (FORMAT PARQUET);"
```

**The whole graph, from Python.** `to_df()` covers nodes; edges come from a
Cypher query in the same link-table shape the SQLite export uses. Drive both
off the graph's own type lists rather than hand-writing a query per type:

```python
import pathlib
import pandas as pd

def export_parquet(graph, out_dir):
    out = pathlib.Path(out_dir)
    (out / 'nodes').mkdir(parents=True, exist_ok=True)
    (out / 'edges').mkdir(exist_ok=True)

    for node_type in graph.node_types:
        graph.select(node_type).to_df().to_parquet(out / 'nodes' / f'{node_type}.parquet')

    for conn in graph.connection_types():
        ct = conn['type']
        df = graph.cypher(f"""
            MATCH (a)-[r:{ct}]->(b)
            RETURN labels(a)[0] AS source_type, a.id AS source_id,
                   labels(b)[0] AS target_type, b.id AS target_id,
                   properties(r) AS props
        """).to_df()
        props = pd.json_normalize(df.pop('props')).drop(columns=['type'], errors='ignore')
        pd.concat([df, props], axis=1).to_parquet(out / 'edges' / f'{ct}.parquet')
```

This produces `nodes/Person.parquet` with columns `type, title, id, age, city`
and `edges/WORKS_AT.parquet` with `source_type, source_id, target_type,
target_id, salary, since` — the same mapping the SQLite table/link-table export
uses. Properties absent on a given node or edge become nulls, and integers,
floats, booleans and datetimes keep their types through the round-trip.

Two things to know when you write your own variant:

- **`properties(r)` includes the connection type under a `type` key**; drop it
  (as above) or it becomes a redundant column.
- **A node's `id`, `title` and `type` come from its canonical identity**, not
  from its property bag. If a node also stores a property under one of those
  names, the canonical value wins and the column is not repeated — a
  duplicated column name would make `to_parquet()` fail outright.

For a whole-graph dump without a DataFrame,
{meth}`~kglite.KnowledgeGraph.export_csv` writes one CSV per node and
connection type plus a `blueprint.json` for re-import, and SQLite (above)
covers the "give me a real database" case.

## Back up before upgrading

The `.kgl` file (and `to_bytes()`) is a **versioned binary cache**, not a
forever-stable archive. KGLite occasionally hard-breaks the on-disk format
across pre-1.0 minor versions, and a newer
binary will **refuse** an older file rather than silently misread it. If you
still have the original source (CSV, DataFrame, dataset loader), you just
rebuild. If you *don't*, you want a portable copy made **before** you upgrade.

Keep the original source/build recipe whenever possible, and copy the `.kgl`
before upgrading. For node/edge/property recovery, also make a portable CSV
export explicitly from the complete graph rather than the current selection:

```python
# Under the version that can still open the graph:
graph.export_csv('backup/', selection_only=False)

# Later, on any version — rebuild the full graph from the portable copy:
import kglite
graph = kglite.from_blueprint('backup/blueprint.json')
```

CSV/blueprint preserves ordinary nodes, edges, and scalar properties, but it is
**not a full-graph backup**: secondary labels, embeddings/vector indexes,
timeseries stores, configured indexes/schema, and some structured value types
are omitted or degraded. Recreate those from their source after import. If the
current fluent selection is intentional, omit `selection_only=False` and treat
the result as a subgraph export.

## NetworkX Interop

Round-trip with [NetworkX](https://networkx.org/) for graph algorithms.
KGLite is a directed multigraph with typed nodes/edges, so the lossless
target is `networkx.MultiDiGraph`: each node's `id` is the networkx node
key (with `node_type`, `title`, and every property as node attributes),
and the first edge's `connection_type` is its edge key. Additional parallel
edges with the same endpoints and type receive collision-safe composite keys,
while every edge retains a `connection_type` attribute.

`to_networkx()` preserves same-type parallel edges. The inverse bulk importer
uses KGLite's endpoint-plus-type DataFrame identity, so importing such a
NetworkX graph collapses duplicates with identical endpoints and type.

Requires the `networkx` extra: `pip install kglite[networkx]`.

```python
import networkx as nx

# Export, run an algorithm, write the scores back.
nxg = graph.to_networkx()              # -> nx.MultiDiGraph
scores = nx.pagerank(nxg)               # {node_id: rank} (pagerank needs scipy)

import pandas as pd
df = pd.DataFrame(
    [{'id': nid, 'pagerank': rank} for nid, rank in scores.items()]
)
# Update existing nodes in place (matched by id), or with Cypher SET:
graph.add_nodes(df, 'Person', 'id', conflict_handling='update')
# graph.cypher("MATCH (n) WHERE n.id = $id SET n.pagerank = $r", ...)

# Import a plain networkx graph (defaults applied where attrs are absent).
g2 = kglite.from_networkx(nxg, default_node_type='Node', default_edge_type='RELATED')
```

`from_networkx()` accepts `Graph` / `DiGraph` / `MultiGraph` /
`MultiDiGraph`; undirected edges become a single directed edge each.
`to_networkx()` exports the full graph (the active selection is ignored
in v1).

## Neo4j Export

Push a graph (or the active selection) to a live Neo4j database over Bolt,
using batched `UNWIND` writes. Requires the `neo4j` driver:
`pip install neo4j`.

```python
import kglite

g = kglite.load("graph.kgl")
report = kglite.to_neo4j(
    g,
    "bolt://localhost:7687",
    auth=("neo4j", "password"),
    clear=False,    # set True to wipe the target DB first
    merge=False,    # set True for MERGE (upsert) instead of CREATE
    batch_size=5000,
)
# {'nodes_created': ..., 'relationships_created': ..., 'elapsed': ..., 'database': 'neo4j'}
```

Pass `selection_only=True` to export just the current selection (otherwise
the full graph is written). Use `merge=True` for idempotent re-runs against
an existing dataset; `clear=True` for a clean reload.

## Merging Graphs (multi-source ingest)

`extend()` folds one in-memory graph into another in place — the native
alternative to round-tripping through CSV when you build a graph
incrementally from several sources or merge two loaded `.kgl` files.

```python
g1 = kglite.load("source_a.kgl")
g2 = kglite.load("source_b.kgl")

report = g1.extend(g2)              # g2 folded into g1; g2 untouched
report = g1.extend(g2, "preserve")  # on conflict, existing g1 values win
```

Node identity is `(node_type, id)`. The `conflict_handling` argument shares
the `add_nodes` vocabulary — `'update'` (default, *other* wins), `'replace'`,
`'skip'`, `'preserve'` (existing wins), `'sum'` (adds numeric **edge**
properties). Secondary labels are unioned (never removed); edges dedup on
`(connection_type, source, target)` so a merge never silently doubles shared
edges. Scope limits (v1): **in-memory storage only**, and **embeddings are
not merged** — re-run `set_embeddings` / `add_embeddings` after the merge.

## Subgraph Extraction

```python
subgraph = (
    graph.select('Company')
    .where({'title': 'Acme Corp'})
    .expand(hops=2)
    .to_subgraph()
)
subgraph.export('acme_network.graphml', format='graphml')
```

## Embedding Snapshots

Export embeddings to a standalone `.kgle` file so they survive graph rebuilds. Embeddings are keyed by node ID — import resolves IDs against the current graph, skipping any that no longer exist.

```python
# Export all embeddings
stats = graph.export_embeddings("embeddings.kgle")
# {'stores': 2, 'embeddings': 5000}

# Export only specific node types
graph.export_embeddings("embeddings.kgle", ["Article"])

# Export specific (node_type, property) pairs
graph.export_embeddings("embeddings.kgle", {
    "Article": ["summary", "title"],
    "Author": [],                     # all embedding properties for Author
})

# Import into a fresh graph — matches by (node_type, node_id)
graph2 = kglite.KnowledgeGraph()
graph2.add_nodes(articles_df, 'Article', 'id', 'title')
result = graph2.import_embeddings("embeddings.kgle")
# {'stores': 2, 'imported': 4800, 'skipped': 200}
```

## Schema and Indexes

### Schema Definition

```python
graph.define_schema({
    'nodes': {
        'Prospect': {
            'required': ['npdid_prospect', 'prospect_name'],
            'optional': ['prospect_status'],
            'types': {'npdid_prospect': 'integer', 'prospect_name': 'string'}
        }
    },
    'connections': {
        'HAS_ESTIMATE': {'source': 'Prospect', 'target': 'ProspectEstimate'}
    }
})

errors = graph.validate_schema()
schema = graph.schema_text()
```

### Indexes

Two index types:

| Method | Accelerates | Use for |
|--------|-------------|---------|
| `create_index()` | Equality (`= value`) | Exact lookups |
| `create_range_index()` | Range (`>`, `<`, `>=`, `<=`) | Numeric/date filtering |

Both also accelerate Cypher `WHERE` clauses. Composite indexes support multi-property equality.

```python
graph.create_index('Prospect', 'prospect_geoprovince')        # equality index
graph.create_range_index('Person', 'age')                      # B-Tree range index
graph.create_composite_index('Person', ['city', 'age'])        # composite equality

graph.list_indexes()
graph.drop_index('Prospect', 'prospect_geoprovince')
```

Indexes are maintained automatically by all mutation operations.

## Performance Tips

1. **Batch operations** — add nodes/connections in batches, not individually
2. **Specify columns** — only include columns you need to reduce memory
3. **Filter by type first** — `select()` before `where()` for narrower scans
4. **Create indexes** — on frequently filtered equality conditions (~3x on 100k+ nodes)
5. **Use lightweight methods** — `len()`, `indices()`, `node()` skip property materialization
6. **Cypher LIMIT** — use `LIMIT` to avoid scanning entire result sets

## Threading

The Python GIL is released during heavy Rust operations, allowing other Python threads to run concurrently:

| Operation | GIL Released? | Notes |
|-----------|:---:|-------|
| `save()` | Yes | Serialization + compression + file write |
| `load()` | Yes | File read + decompression + deserialization |
| `cypher()` (reads) | Yes | Query parsing, optimization, and execution |
| `vector_search()` | Yes | Similarity computation (uses rayon internally) |
| `search_text()` | Partial | Model embedding needs GIL; vector search releases it |
| `add_nodes()` | No | DataFrame conversion requires GIL throughout |
| `cypher()` (mutations) | No | Must hold exclusive lock on graph |

A `KnowledgeGraph` is single-owner: concurrent reads are fine, but a read that
overlaps a mutation on the same instance raises a `RuntimeError`. For lock-free
concurrent reads across threads, serve from an immutable `graph.freeze()`
snapshot (see {doc}`/concepts/concurrency`).

## Serialize to/from bytes

`save()`/`load()` go through a filesystem path. To own the write — push to object
storage, a socket, a checksum — serialize the whole graph to a `.kgl` **byte
buffer** instead:

```python
blob = graph.to_bytes()              # bytes (the same format save() writes)
graph = kglite.from_bytes(blob)      # round-trips; raises FileFormatError if corrupt
```

In-memory / mapped graphs only (a disk-mode graph is a directory, not a stream).

## Human-readable diffs (`to_text` + git `textconv`)

A `.kgl` is a compressed binary blob, so `git diff` shows `Binary files differ`
— useless for reviewing a change to a graph (e.g. an agent-built planning
graph). `to_text()` projects the whole graph to a **deterministic, readable**
form (nodes grouped by type + sorted by id, edges sorted by endpoints), stable
across insert order *and* across save/load:

```python
print(graph.to_text())
# Task (2 node(s))
#   t1 | Plan the API | status=done
#   t2 | Write tests | status=todo
#
# edges (1)
#   (t1)-[BLOCKS]->(t2)
```

Reserved provenance keys (`updated_at`/`git_sha`) are omitted so per-write
metadata churn doesn't swamp the diff.

**Wire it into git** so `git diff` / PR review renders `.kgl` files readably
(the CLI ships an `export-text` subcommand — `pip install kglite`):

```bash
git config diff.kglite.textconv "kglite export-text"
echo "*.kgl diff=kglite" >> .gitattributes
```

Now `git diff path/to/graph.kgl` shows real content changes. For an explicit
before/after delta of two files:

```bash
kglite diff old.kgl new.kgl
# -t2 | Write tests | status=todo
# +t2 | Write tests | status=done    # a changed node shows as a -/+ pair
```

## Graph Maintenance

After heavy mutation workloads (DELETE, REMOVE), internal storage accumulates tombstones. Monitor with `graph_info()`.

```python
info = graph.graph_info()
# {'node_count': 950, 'node_capacity': 1000, 'node_tombstones': 50,
#  'edge_count': 2800, 'edge_capacity': 3000, 'edge_tombstones': 200,
#  'fragmentation_ratio': 0.05, 'auto_vacuum_threshold': 0.3,
#  'auto_vacuums_run': 2, ...}

if info['fragmentation_ratio'] > 0.3 or info['edge_tombstones'] > 0:
    result = graph.vacuum()
    print(f"Reclaimed {result['tombstones_removed']} node slots, "
          f"{result['edge_tombstones_removed']} edge slots")
```

`vacuum()` rebuilds the graph with contiguous indices and rebuilds all indexes.
The current selection is **carried through** it: surviving nodes keep their
place at their new indices, deleted ones drop out, and after a traversal a
group whose parent was deleted is dropped whole.

`fragmentation_ratio` is node-shaped. A workload that deletes only
relationships leaves it at `0.0` and shows up in `edge_tombstones` instead —
auto-vacuum takes the worst of node slots, edge slots and dead property-column
rows, so it fires on any of the three.

## Common Gotchas

- **One primary type per node.** Secondary labels (multi-label, 0.10.5+) are preserved; `labels(n)` returns a list, primary type first.
- **`id` and `title` are canonical.** `add_nodes(unique_id_field='user_id')` stores the column as `id`. The original name works as an alias.
- **Save files use a versioned binary format.** They are portable across
  supported OS/architectures, but a pre-1.0 minor release may require rebuild.
- **Indexes:** `create_index()` accelerates equality only. For range queries, use `create_range_index()`.
- **Flat vs. grouped results.** After traversal with multiple parents, `titles()` and `collect()` return grouped dicts.
- **Persistence is explicit unless lifecycle helpers are used.** `save()` is
  manual on a plain graph; `open()` remembers a path and clean context-manager
  exit saves, while `open()` is write-ahead logged by default.
