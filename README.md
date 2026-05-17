# KGLite — Lightweight Knowledge Graph for Python

[![PyPI version](https://img.shields.io/pypi/v/kglite)](https://pypi.org/project/kglite/)
[![Python versions](https://img.shields.io/pypi/pyversions/kglite)](https://pypi.org/project/kglite/)
[![License: MIT](https://img.shields.io/pypi/l/kglite)](https://github.com/kkollsga/kglite/blob/main/LICENSE)
[![Docs](https://img.shields.io/readthedocs/kglite)](https://kglite.readthedocs.io)

KGLite is an embedded knowledge graph for Python: `pip install`, no
server, no setup. It speaks Cypher, loads pandas DataFrames, and
ships with the connective tissue for AI agents — an MCP server so
Claude / Cursor / any MCP-capable LLM can query your graph as a
tool, a `describe()` method that emits a compact XML schema for
system prompts, and a `code_tree` parser that turns any source
directory into a graph of functions, classes, calls, imports,
decorators, and web-framework routes across 13 languages.

Three storage modes scale from in-memory (millisecond queries on
small graphs) to mmap-backed on disk (1 B+ edges, Wikidata-scale).
Bundled dataset wrappers turn `pip install kglite` into a queryable
Wikidata or petroleum-domain graph in one line.

## Why KGLite?

- **Built for LLM agents** — `describe()` XML schema, bundled MCP
  server, an agent-oriented query surface (`cypher()`,
  `graph.select(...).traverse(...)`), and structural validators
  (`CALL orphan_node({type: ...}) YIELD node`) for data-integrity
  checks that compose with the rest of Cypher.
- **One-line public datasets** — `wikidata.open(path)` and
  `sodir.open(path)` handle fetch, parallel build, and caching;
  re-runs reload the cached graph instantly.
- **Codebase → graph in one line** — `kglite.code_tree.build(".")`
  parses Python, Rust, TypeScript, JavaScript, Go, Java, C#, C, C++,
  Swift, PHP, HTML, and CSS into `Function` / `Class` / `Module` /
  `Route` / `Element` (HTML headings + sections + forms) /
  `Selector` (CSS rules) nodes with `CALLS` / `DEFINES` / `IMPORTS` /
  `DECORATES` / `HANDLES` / `HAS_CHILD` edges. Web-framework route
  detection ships for Flask, FastAPI, and Django. HTML god-file
  workflows are first-class: inline `<script>` blocks are parsed as
  JS so their functions get CALLS edges, and the document outline
  surfaces as Element nodes connected via HAS_CHILD. `kg.explore(query)`
  does one-call codebase exploration; `CALL affected_tests({files: [...]})`
  returns transitively-impacted test files.
- **Scales without leaving Python** — in-memory for prototyping,
  mmap-backed for notebook-scale, disk-mode CSR for graphs too
  large for RAM. Same API across modes.
- **Query with Cypher** — `MATCH`, `MERGE`, `OPTIONAL MATCH`,
  aggregations, parameters, semantic search via `text_score()`.
- **DataFrames in, DataFrames out** — bulk-load nodes and edges
  from pandas with `add_nodes` / `add_connections`, query results
  back as DataFrames. End-to-end walkthrough in the
  [Data Loading guide](https://kglite.readthedocs.io/en/latest/guides/data-loading.html).

## Quick Start

```bash
pip install kglite
```

```python
import pandas as pd
import kglite

# Three storage modes — pick by graph size:
#   default (in-memory)   — small/medium graphs, fastest queries
#   storage="mapped"      — mmap columns, RAM-friendly as you grow
#   storage="disk", path=…  — 100M+ nodes, Wikidata-scale, loaded lazily
graph = kglite.KnowledgeGraph()

# Bulk-load nodes from a DataFrame (also: add_nodes_bulk, from_blueprint,
# load_ntriples, or Cypher CREATE for ad-hoc inserts).
people = pd.DataFrame({
    "id":   ["alice", "bob", "eve"],
    "name": ["Alice", "Bob", "Eve"],
    "age":  [28, 35, 41],
    "city": ["Oslo", "Bergen", "Trondheim"],
})
graph.add_nodes(people, node_type="Person", unique_id_field="id", node_title_field="name")

# Bulk-load relationships the same way (also: add_connections_bulk,
# add_connections_from_source for auto-filter by loaded types).
knows = pd.DataFrame({"src": ["alice", "bob"], "tgt": ["bob", "eve"]})
graph.add_connections(knows, connection_type="KNOWS",
                      source_type="Person", source_id_field="src",
                      target_type="Person", target_id_field="tgt")

# Query — returns a ResultView (lazy; data stays in Rust until accessed).
result = graph.cypher("""
    MATCH (p:Person) WHERE p.age > 30
    RETURN p.name AS name, p.city AS city
    ORDER BY p.age DESC
""")
for row in result:
    print(row['name'], row['city'])

# Or get a pandas DataFrame directly.
df = graph.cypher("MATCH (p:Person) RETURN p.name, p.age ORDER BY p.age", to_df=True)

# Persist to disk and reload.
graph.save("my_graph.kgl")
loaded = kglite.load("my_graph.kgl")
```

## Try it instantly: ready-to-query datasets

Two bundled wrappers turn well-known public sources into queryable
graphs without writing a loader. Each call handles the *fetch +
build + cache* cycle, returns a `KnowledgeGraph` you can `cypher()`
against, and respects a per-dataset cooldown so re-running just
loads the cached graph in seconds. KGLite is independent of the
upstream organisations — see each module docstring for
non-affiliation notes.

### Wikidata

Single-stream `latest-truthy.nt.bz2` from
[dumps.wikimedia.org](https://dumps.wikimedia.org/wikidatawiki/entities/) —
parallel-decoded with a bit-level block scanner, parsed, built into a
queryable graph in one call:

```python
from kglite.datasets import wikidata

g = wikidata.open("/data/wd")                                    # full graph
g = wikidata.open("/data/wd", entity_limit_millions=100)         # 100M slice
g = wikidata.open("/data/wd", storage="memory",                  # in-memory, fast tests
                  entity_limit_millions=10)
```

### Sodir (Norwegian Offshore Directorate)

Petroleum-domain graph from the public ArcGIS REST FeatureServer at
[factmaps.sodir.no](https://factmaps.sodir.no/api/rest/services/DataService) —
33 baseline node types (Field, Wellbore, Discovery, Licence,
Stratigraphy, …), ~480 k nodes, parallel-fetched and built in
seconds:

```python
from kglite.datasets import sodir

g = sodir.open("/data/sodir")  # in-memory by default; ~30s first run
g = sodir.open("/data/sodir", complement_blueprint="my_extras.json")  # extend
```

Two-tier cooldown — cheap row-count probes every 14 days; full
per-dataset re-fetch every 30 days. Add a *complement blueprint* to
extend the baseline (new node types, custom edges) without touching
the canonical schema; the file is persisted into the workdir on
first use and auto-loaded after.

## Use Cases

### Agentic AI — memory and tool use

Give an LLM a structured memory it can query. `describe()` emits a
compact XML schema that fits in a system prompt, and the bundled MCP
server exposes the whole graph as a Cypher tool — drop-in for Claude,
Cursor, or any MCP-capable agent.

```python
xml = graph.describe()                            # schema for the agent's context
prompt = f"You have a knowledge graph:\n{xml}\nAnswer via graph.cypher()."
```

```bash
# Or serve the whole graph over MCP. `kglite-mcp-server` is shipped
# inside the wheel as a Python console-script entry point — no Rust
# toolchain needed, no PyO3 env vars, no conda env handling.
pip install 'kglite[mcp]'
kglite-mcp-server --graph path/to/graph.kgl
```

> **Migrating from a 0.9.18 or 0.9.19 install?** No YAML changes
> needed. `pip install --upgrade 'kglite[mcp]'` and you're done.
> The 0.9.20 release retired the bundled Rust binary in favour of a
> Python entry point, which removes the per-Python-version wheel
> matrix and the install_name_tool / patchelf / mold complexity that
> came with it. Cypher execution still happens in the Rust extension
> module under the GIL release inside `cypher()`, so performance is
> unchanged.

Drop a `<basename>_mcp.yaml` next to the graph to auto-extend the
tool surface — `source_root:` for read/grep/list over your source
files, inline Cypher templates as named tools, `extensions.embedder`
for `text_score()`, `extensions.cypher_preprocessor` for query
rewriting. No fork required for most customisation. See
[the MCP guide](https://kglite.readthedocs.io/en/latest/guides/mcp-servers.html).

### Codebase analysis

Parse Python, Rust, TypeScript, Go, Java, C#, and C++ into a graph of
functions, classes, calls, and imports. Trace who-calls-what, find
dead code, and review structure without leaving your editor. Pairs
naturally with the MCP server so an agent can reason over your repo.

```python
from kglite.code_tree import build

graph = build(".")                                # parse current directory
graph.cypher("""
    MATCH (f:Function)-[:CALLS]->(g:Function)
    RETURN g.name, count(f) AS callers
    ORDER BY callers DESC LIMIT 10
""")
```

### RAG retrieval

Store documents, chunks, and entities together as one graph. Combine
`text_score()` semantic similarity with Cypher structure — hybrid
retrieval in one query, no second vector DB.

```python
graph.cypher("""
    MATCH (c:Chunk)-[:IN_DOC]->(d:Document)
    RETURN c.text, d.title,
           text_score(c.embedding, $query_vec) AS score
    ORDER BY score DESC LIMIT 5
""", params={"query_vec": query_embedding})
```

### Data exploration and analysis

Load CSVs or DataFrames, walk relationships, run graph algorithms
(shortest path, centrality, community detection), and export — all
from a notebook.

```python
graph.add_nodes(users_df, node_type="User", unique_id_field="user_id", node_title_field="name")
graph.cypher("""
    MATCH path = shortestPath((a:User {name:'Alice'})-[*]-(b:User {name:'Eve'}))
    RETURN path
""")
```

### Structural validators — surface data-integrity gaps in one query

Six built-in `CALL` procedures find the gaps that aren't visible
from normal queries: nodes with zero edges, missing-required-edge
violations, two-step cycles, duplicate titles, more. They compose
with the rest of Cypher — feed the output into `WHERE`, `ORDER BY`,
or downstream aggregation in a single pass.

```python
# Wellbores in our sodir graph that lack a production licence
graph.cypher("""
    CALL missing_required_edge({type: 'Wellbore', edge: 'IN_LICENCE'}) YIELD node
    RETURN node.id, node.title
""")  # 502 violations on the Sodir April-2026 snapshot

# Cross-reference flagged IDs against any query result, in one Cypher pass
graph.cypher("""
    MATCH (l:Licence {title: '057'})<-[:IN_LICENCE]-(w:Wellbore)
    WITH collect(w.id) AS pl057
    CALL missing_required_edge({type: 'Wellbore', edge: 'DRILLED_BY'}) YIELD node
    WHERE node.id IN pl057
    RETURN count(*) AS pl057_missing_drilled_by
""")
```

`missing_required_edge` and `missing_inbound_edge` validate the
`(type, edge)` direction against the graph's actual schema and
refuse to execute when misused. See
[`docs/guides/cypher.md`](https://kglite.readthedocs.io/en/latest/guides/cypher.html#structural-validator-call-procedures)
for the full procedure list.

## Examples

The [`examples/`](https://github.com/kkollsga/kglite/tree/main/examples)
directory has runnable, self-contained artifacts covering each of the
use cases above:

- **[`open_source_workspace_mcp.yaml`](https://github.com/kkollsga/kglite/blob/main/examples/open_source_workspace_mcp.yaml)**
  — annotated workspace-mode manifest for the
  github-clone-tracker pattern: agent calls
  `repo_management('org/repo')`, kglite clones the repo and builds
  its code-tree graph, queries flow against the active clone. Drop
  the file into your workspace directory as `workspace_mcp.yaml`
  and run `kglite-mcp-server --workspace /path/to/dir/`. Walked
  through in
  [the workspace manifest example](https://kglite.readthedocs.io/en/latest/examples/manifest_workspace.html).
- **[`legal_graph.py`](https://github.com/kkollsga/kglite/blob/main/examples/legal_graph.py)**
  — end-to-end `add_nodes` / `add_connections` from pandas DataFrames,
  covering laws, regulations, and court decisions with citation
  relationships. The imperative-API alternative when you're building
  the graph itself, not configuring a server on top.
- **[`code_graph.py`](https://github.com/kkollsga/kglite/blob/main/examples/code_graph.py)**
  — build a code knowledge graph from a source directory via
  `code_tree.build`. Produces `Function`, `Class`, `Module`, `File`
  nodes with `CALLS`, `DEFINES`, `IMPORTS` edges.
- **[`spatial_graph.py`](https://github.com/kkollsga/kglite/blob/main/examples/spatial_graph.py)**
  — declarative CSV→graph loading via a JSON blueprint; regions,
  facilities, and sensors with lat/lon coordinates and pipeline-path
  traversal queries.
- **[`crates/kglite-mcp-server/`](https://github.com/kkollsga/kglite/tree/main/crates/kglite-mcp-server)**
  — Rust-native single-binary MCP server (built on rmcp + the
  [mcp-methods] framework). Reach for it when the manifest doesn't
  express what you need; the binary is the reference for layering
  domain-specific tools on top of the generic source / GitHub /
  workspace surface.

[mcp-methods]: https://github.com/kkollsga/mcp-methods

For Wikidata- and Sodir-scale builds, see the [Public datasets](#public-datasets)
section above — `kglite.datasets.wikidata.open(...)` and
`kglite.datasets.sodir.open(...)` cover those workflows in one call.

## Benchmarks

KGLite builds and queries Wikidata-scale graphs on a laptop.
Measured with
[`bench/wiki_benchmark.py`](https://github.com/kkollsga/kglite/blob/main/bench/wiki_benchmark.py)
on an M-series MacBook.

**Ingest** — full pipeline from compressed N-Triples to a queryable graph:

| dataset   | triples | nodes  | edges  | ingest  | throughput       | peak RAM |
|-----------|--------:|-------:|-------:|--------:|------------------|---------:|
| wiki100m  |  100 M  |  938 K |  748 K |   29 s  | 3.4 M triples/s  |  1.3 GB  |
| wiki500m  |  500 M  |  5.6 M |  6.7 M |  157 s  | 3.2 M triples/s  |  5.2 GB  |
| wiki1000m |    1 B  | 14.7 M | 15.4 M |  395 s  | 2.5 M triples/s  |  7.0 GB  |

Reloading a saved 1 B-triple graph from disk (7 GB on-disk): **3.5 s**.

**Query latency on the 1 B-triple graph** (mapped storage). Type names
match the labels Wikidata ships per language — with `languages=["en"]`
(the default), `Q5` is renamed to `human`:

| Cypher                                                              |     wall |
|---------------------------------------------------------------------|---------:|
| `MATCH (n)-[:P31]->(:human) RETURN count(n)` — typed aggregation    |   0.5 ms |
| `MATCH (a)-[:P31]->(b)-[:P279]->(c) LIMIT 10` — 2-hop typed         |   0.9 ms |
| `MATCH (a)-[:P31]->(b {nid:'Q64'}) RETURN a LIMIT 20` — pivot       |     1 ms |
| `MATCH (a)-[:P31]->(:human)` `MATCH (a)-[:P27]->(c) LIMIT 10` — join |   44 ms |

Disk and mapped storage track within 1 % on build; mapped wins on
query shapes backed by its in-memory inverted index, disk wins on
unbounded typed traversals by staying on sorted-CSR mmap I/O.

No server, no tuning, same Python process as your code.

## Key Features

| Feature | Description |
|---|---|
| **Cypher queries** | MATCH, CREATE, SET, DELETE, MERGE, UNION/INTERSECT/EXCEPT, aggregations (incl. `median`, `percentile_cont`, `variance`), `reduce()`, ORDER BY, LIMIT, SKIP |
| **Semantic search** | Vector embeddings + `text_score()` for similarity ranking |
| **Text predicates** | `text_edit_distance`, `text_normalize`, `text_jaccard`, `text_ngrams`, `text_contains_any` / `text_starts_with_any` for fuzzy match |
| **Graph algorithms** | Shortest path (BFS or Dijkstra via `weight_property`), centrality, community detection, clustering |
| **Structural validators** | 14 `CALL` procedures: `orphan_node`, `missing_required_edge`, `cycle_2step`, `inverse_violation`, `transitivity_violation`, `cardinality_violation`, `parallel_edges`, `null_property`, `type_domain/range_violation`, etc. — agent-discoverable integrity checks composable with normal Cypher |
| **Spatial** | Coordinates, WKT geometry, distance + containment, geometry primitives (`geom_buffer`, `geom_convex_hull`, `geom_union/intersection/difference`, `geom_is_valid`, `geom_length`), `kg_knn` k-nearest-neighbour |
| **Timeseries** | Time-indexed data with `ts_*()` Cypher functions |
| **Bulk loading** | Fluent API (`add_nodes` / `add_connections`) for DataFrames |
| **Blueprints** | Declarative CSV-to-graph loading via JSON config |
| **Import/Export** | Save/load snapshots, GraphML, CSV export |
| **AI integration** | `describe()` introspection, MCP server, agent prompts |
| **Code analysis** | Parse codebases via tree-sitter (`kglite.code_tree`) |

## Documentation

Full docs at **[kglite.readthedocs.io](https://kglite.readthedocs.io)**:

- [Getting Started](https://kglite.readthedocs.io/en/latest/getting-started.html) — installation, first graph, core concepts
- [Cypher Guide](https://kglite.readthedocs.io/en/latest/guides/cypher.html) — queries, mutations, parameters
- [Semantic Search](https://kglite.readthedocs.io/en/latest/guides/semantic-search.html) — embeddings, vector search
- [AI Agents](https://kglite.readthedocs.io/en/latest/guides/ai-agents.html) — MCP server, `describe()`, agent prompts
- [API Reference](https://kglite.readthedocs.io/en/latest/autoapi/kglite/index.html) — full auto-generated reference

## Requirements

Python 3.10+ (CPython) | macOS (ARM), Linux (x86_64/aarch64), Windows (x86_64) | `pandas >= 1.5`

## License

MIT — see [LICENSE](https://github.com/kkollsga/kglite/blob/main/LICENSE) for details.
