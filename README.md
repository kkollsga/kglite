# KGLite — Knowledge graph for Python, built for LLM agents

[![PyPI version](https://img.shields.io/pypi/v/kglite)](https://pypi.org/project/kglite/)
[![Python versions](https://img.shields.io/pypi/pyversions/kglite)](https://pypi.org/project/kglite/)
[![License: MIT](https://img.shields.io/pypi/l/kglite)](https://github.com/kkollsga/kglite/blob/main/LICENSE)
[![Docs](https://img.shields.io/readthedocs/kglite)](https://kglite.readthedocs.io)

KGLite is an embedded, Cypher-queryable knowledge graph for Python,
built so you can hand it to an LLM agent. `pip install kglite` and
point `kglite.code_tree.build(".")` at any source directory — your
first queryable graph in seconds. It ships with a bundled MCP server,
a `describe()` method that emits a system-prompt-shaped schema, and
structural validators that compose with Cypher.

> ### 🚀 See it end-to-end: codebase → Claude in ~50 lines
>
> [**`examples/codebase_to_claude_mcp.ipynb`**](https://github.com/kkollsga/kglite/blob/main/examples/codebase_to_claude_mcp.ipynb)
> clones a GitHub repo, parses it into a code knowledge graph, runs a
> few Cypher queries, then registers a workspace MCP server in Claude
> Desktop. Closes with a screenshot of Claude calling `repo_management`
> → `graph_overview` → `cypher_query` against the live graph.

> ### 🏦 Or: SEC filings as a knowledge graph, in one call
>
> ```python
> from kglite.datasets.sec import SEC
> g = SEC.fetch("./sec", "13F-HR", "TSLA", years=2,
>               user_agent="Your Name your@email.com")
> ```
>
> `SEC.fetch` names the forms, the companies, and a span — then
> downloads from SEC EDGAR (with a progress bar) and hands back a
> Cypher-queryable graph: Form 4 insider transactions, 13F
> institutional holdings, SC 13D activist stakes, DEF 14A board
> composition, 8-K material events. Every fact is a typed node, and
> the same person is one `:Person` across every form. `SEC.open` is
> the full-control entry point — XBRL financials, Exhibit 21
> subsidiaries, the full ~14M-filing index since 1993. Public-domain
> data (US Govt work). **→
> [SEC guide](https://kglite.readthedocs.io/en/latest/guides/sec.html).**

## Use cases

KGLite is shape-agnostic — the agent-facing surface is the same
whether the graph holds your legal precedents, a Wikidata slice,
your SQL warehouse, a RAG corpus, or a parsed codebase.

- 🏦 **SEC EDGAR in one call.** `SEC.fetch(path, "13F-HR", "TSLA",
  years=2, user_agent="...")` builds a US-public-company knowledge
  graph from the SEC's free data: companies, filings, insider
  transactions (Form 4), institutional holdings (13F), activist
  stakes (SC 13D), board composition (DEF 14A), 8-K material events
  — with XBRL financials and Exhibit 21 subsidiary trees a flag away
  via `SEC.open`. Facts are typed nodes; the same person is one
  `:Person` across every form. Three-tier `raw` / `processed` /
  `graph` cache that never re-fetches. **→
  [SEC guide](https://kglite.readthedocs.io/en/latest/guides/sec.html).**
- 🏛️ **Domain knowledge for agents.** Legal precedents + citations,
  regulatory rules, medical ontologies, manufacturing BOMs, scientific
  catalogues — anything with structure becomes a queryable graph an
  MCP-capable agent can reason over. See the
  [legal-graph example](https://github.com/kkollsga/kglite/blob/main/examples/legal_graph.py)
  for a Norwegian-Supreme-Court walk-through (laws + decisions +
  citation edges + judge metadata).
- 📊 **Business data → queryable graph.** Any tabular source — SQL,
  CSV, Parquet, REST API responses, pandas DataFrames — goes straight
  in via `add_nodes(df, ...)` and `add_connections(df, ...)`. Layer a
  graph on top of your warehouse and the agent reasons over the
  relationships without you writing a server. **→
  [Data Loading guide](https://kglite.readthedocs.io/en/latest/guides/data-loading.html).**
- 🌐 **Public datasets, one line.** `wikidata.open(path)` and
  `sodir.open(path)` handle the *fetch + build + cache* cycle. Run
  Cypher queries on a billion-edge Wikidata graph from a 16 GB
  laptop — mapped/disk storage means you can operate and query
  datasets that won't fit in RAM. **→ See
  [Bundled datasets](#bundled-datasets) below.**
- 📚 **RAG with structure.** Documents, chunks, entities, and the
  edges between them in one graph. Combine `text_score()` vector
  similarity with Cypher traversal — *"find court cases semantically
  similar to my fact pattern, then walk one hop to related
  precedents"* — hybrid retrieval in one query, no second vector DB.
  **→ [Semantic Search guide](https://kglite.readthedocs.io/en/latest/guides/semantic-search.html).**
- 📂 **Codebase analysis.** `kglite.code_tree.build(".")` parses 13
  languages into Function / Class / Module / Route nodes with
  web-framework route detection (Flask, FastAPI, Django). See the
  [notebook above](https://github.com/kkollsga/kglite/blob/main/examples/codebase_to_claude_mcp.ipynb)
  for the full code → Claude Desktop workflow. **→
  [Code analysis guide](https://kglite.readthedocs.io/en/latest/guides/code-tree.html).**

## Why Cypher?

A question every investor asks: *which insiders are selling, and at
what price?* Against raw SEC XML you parse 1000s of Form 4 documents,
join on issuer CIK, filter by transaction code. Against a graph it's
one query:

```cypher
-- Insider sells at Apple (CIK 320193), most recent first
MATCH (c:Company {cik: 320193})-[:HAS_INSIDER]->(p:Person)
      <-[:OF_PERSON]-(t:Transaction {transaction_code: 'S'})
RETURN p.display_name, t.transaction_date, t.shares, t.price_per_share
ORDER BY t.transaction_date DESC LIMIT 10
```

Three node types (`Company`, `Person`, `Transaction`), two edge
types (`HAS_INSIDER`, `OF_PERSON`), pattern-matched and joined in
one expression. The same shape composes into harder questions —
swap `:HAS_INSIDER` for `:HOLDS` and you're walking institutional
positions; add `:SERVES_ON_BOARD` and you're checking who's an
insider AND a director. Cypher pays off most when the data has
real structure and your questions traverse it.

## How it compares

|                                            | KGLite                            | Kuzu                       | NetworkX           | rustworkx          | Neo4j Embedded         |
|--------------------------------------------|-----------------------------------|----------------------------|--------------------|--------------------|------------------------|
| **Install**                                | `pip install kglite`              | `pip install kuzu`         | `pip install networkx` | `pip install rustworkx` | JVM + Java deps  |
| **Query language**                         | Cypher (subset)                   | Cypher (full)              | Python API         | Python API         | Cypher (full)          |
| **Storage**                                | in-mem · mmap · disk (1B+ edges)  | in-mem · disk (columnar)   | in-mem             | in-mem             | in-mem · disk (JVM)    |
| **Bulk-load from pandas**                  | one-liner                         | via Arrow                  | manual             | manual             | via driver             |
| **Bundled MCP server for LLM agents**      | ✅                                 | —                          | —                  | —                  | —                      |
| **`describe()` schema for LLM prompts**    | ✅                                 | —                          | —                  | —                  | —                      |
| **Codebase → graph parser**                | 13 languages, route detection     | —                          | —                  | —                  | —                      |
| **Bundled public datasets**                | SEC EDGAR, Wikidata, Sodir        | —                          | toy graphs only    | —                  | —                      |
| **License**                                | MIT                               | MIT                        | BSD-3              | Apache-2           | GPLv3                  |

**Pick KGLite** when you want Cypher + Python ergonomics + LLM-agent
plumbing in one wheel. **Pick Kuzu** for full openCypher coverage and
analytical OLAP throughput. **Pick NetworkX** when you need its
enormous graph-algorithm library and your data fits in RAM. **Pick
rustworkx** when you want NetworkX's API in Rust with no query
language. **Pick Neo4j Embedded** when you've standardised on
server-mode Cypher and want the in-process driver for tests.

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

# Bulk-load nodes from a DataFrame.
people = pd.DataFrame({
    "id":   ["alice", "bob", "eve"],
    "name": ["Alice", "Bob", "Eve"],
    "age":  [28, 35, 41],
    "city": ["Oslo", "Bergen", "Trondheim"],
})
graph.add_nodes(people, node_type="Person", unique_id_field="id", node_title_field="name")

# Bulk-load relationships the same way.
knows = pd.DataFrame({"src": ["alice", "bob"], "tgt": ["bob", "eve"]})
graph.add_connections(knows, connection_type="KNOWS",
                      source_type="Person", source_id_field="src",
                      target_type="Person", target_id_field="tgt")

# Query — returns a ResultView (lazy; data stays in Rust until accessed).
for row in graph.cypher("""
    MATCH (p:Person) WHERE p.age > 30
    RETURN p.name AS name, p.city AS city
    ORDER BY p.age DESC
"""):
    print(row['name'], row['city'])

# Or get a pandas DataFrame directly.
df = graph.cypher("MATCH (p:Person) RETURN p.name, p.age ORDER BY p.age", to_df=True)

# Persist to disk and reload.
graph.save("my_graph.kgl")
loaded = kglite.load("my_graph.kgl")
```

**→ [Getting Started guide](https://kglite.readthedocs.io/en/latest/getting-started.html) ·
[Cypher reference](https://kglite.readthedocs.io/en/latest/guides/cypher.html) ·
[API reference](https://kglite.readthedocs.io/en/latest/autoapi/kglite/index.html).**

## Serve it to an agent

Three levels of effort, three levels of capability.

### 1. One command — any `.kgl` becomes an MCP server

```bash
kglite-mcp-server --graph path/to/graph.kgl
```

The server exposes `cypher_query`, `graph_overview`, schema
introspection, structural validators, and source-file tools over MCP
stdio. Drop it into Claude Desktop / Cursor / any MCP-capable client
and your graph is queryable. Works on every graph kglite can build —
your own, Wikidata, Sodir, code-tree.

### 2. Customise with a YAML manifest

Drop `<basename>_mcp.yaml` next to the graph (e.g. `wikidata_mcp.yaml`
beside `wikidata.kgl`) and the server auto-loads it at boot.

```yaml
name: Wikidata Explorer
source_root: /path/to/related/source        # exposes read/grep/list
extensions:
  embedder: { kind: fastembed, model: bge-small }   # enables text_score()
  csv_http_server: true                              # bulk CSV exports
tools:                                               # inline parameterised Cypher
  - name: who_invented
    cypher: |
      MATCH (i:Q5)-[:P61]->(t {label:$thing})
      RETURN i.label LIMIT 5
```

No fork required for most customisation. **→
[MCP server guide](https://kglite.readthedocs.io/en/latest/guides/mcp-servers.html).**

### 3. Teach the agent with bundled skills

Markdown skill files (`<basename>.skills/*.md`) ship methodology for
each tool. The agent reads `cypher_query.md` at session start to learn
your schema conventions, `read_code_source.md` to know when to drill
into source vs. query the graph, etc. Three layers compose:
kglite-bundled defaults + your project's `.skills/` overrides +
operator-declared domain packs. Skills with `applies_when:` predicates
only activate when the graph contains the relevant node types — so a
non-code graph never sees `read_code_source` methodology.

Net effect: the agent comes pre-loaded with how to use your graph,
rather than discovering it through trial-and-error. **→
[AI Agents guide](https://kglite.readthedocs.io/en/latest/guides/ai-agents.html).**

## Bundled datasets

Three wrappers turn well-known public sources into queryable graphs
without writing a loader. Each handles the *fetch + build + cache*
cycle, returns a `KnowledgeGraph` you can `cypher()` against, and
respects a per-dataset cooldown so re-running just reloads the cached
graph in seconds. KGLite is independent of the upstream
organisations — see each module docstring for non-affiliation notes.
**→ [Datasets guide](https://kglite.readthedocs.io/en/latest/guides/datasets.html).**

### SEC EDGAR

US-public-company knowledge graph from the SEC's free public data —
all 14M historical filings + per-filing payload parsing for Form 4
(insider transactions), 13F-HR (institutional holdings), SC 13D
(activist stakes), DEF 14A (board composition), XBRL company facts
(financial metrics), 10-K Exhibit 21 (subsidiaries), 8-K cover pages
(material event Item codes):

```python
from kglite.datasets.sec import SEC

# SEC.fetch — name the forms, the companies, a span; get a graph back.
g = SEC.fetch("/data/sec", ["4", "8-K", "DEF 14A"], ["AAPL", "TSLA"],
              years=2, user_agent="Your Name your@email.com")

# SEC.open — full control: separate filing-index vs. payload spans,
# storage mode, and the include_* flags (XBRL financials, Exhibit 21
# subsidiaries).
g = SEC.open("/data/sec", years=10, detailed=2,
             user_agent="Your Name your@email.com")

# Full universe — drop `companies`; auto-escalates to mode="disk".
g = SEC.open("/data/sec", years="all", detailed=5,
             user_agent="Your Name your@email.com")
```

Two dozen-plus typed node types — Company, Person, Filing,
InsiderTransaction, Holding, InstitutionalHolding, CorporateEvent,
Compensation, Role, MetricFact, Subsidiary and more — wired by typed
edges, every fact node tracing back to its source filing. Three-tier
`raw` / `processed` / `graph/{mode}` cache
— `raw` is immutable, `processed` regenerates only when its `raw`
source changes, `graph/{mode}/` reuses on reopen unless
`force_rebuild=True`. SEC's 10 req/s fair-access policy is enforced
by an internal token-bucket rate limiter; the `user_agent` arg is
mandatory (SEC returns 403 without it).

Source data is public domain (US Govt work) — redistribute the built
`.kgl` however you like. **→
[SEC guide](https://kglite.readthedocs.io/en/latest/guides/sec.html).**

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
Stratigraphy, …), ~480 k nodes, parallel-fetched and built in seconds:

```python
from kglite.datasets import sodir

g = sodir.open("/data/sodir")  # in-memory by default; ~30s first run
g = sodir.open("/data/sodir", complement_blueprint="my_extras.json")  # extend baseline
```

Two-tier cooldown — cheap row-count probes every 14 days; full
per-dataset re-fetch every 30 days. Add a *complement blueprint* to
extend the baseline (new node types, custom edges) without touching
the canonical schema.

## Recipes

Short patterns for the most-common shapes. Each is self-contained.

### Hybrid semantic + structural retrieval

Combine vector similarity (`text_score()`) with Cypher pattern
matching in one query:

```python
graph.cypher("""
    MATCH (c:Chunk)-[:IN_DOC]->(d:Document)
    RETURN c.text, d.title,
           text_score(c.embedding, $query_vec) AS score
    ORDER BY score DESC LIMIT 5
""", params={"query_vec": query_embedding})
```

Vector embeddings via `pip install 'kglite[embed]'` (adds fastembed +
onnxruntime). **→ [Semantic Search guide](https://kglite.readthedocs.io/en/latest/guides/semantic-search.html).**

### Structural validators — surface data-integrity gaps

Fourteen built-in `CALL` procedures find the gaps that aren't visible
from normal queries: orphan nodes, missing-required-edge violations,
two-step cycles, duplicate titles, parallel edges, cardinality
violations, more. They compose with the rest of Cypher.

```python
# Wellbores in our sodir graph that lack a production licence
graph.cypher("""
    CALL missing_required_edge({type: 'Wellbore', edge: 'IN_LICENCE'}) YIELD node
    RETURN node.id, node.title
""")
```

`missing_required_edge` and `missing_inbound_edge` validate the
`(type, edge)` direction against the graph's actual schema and refuse
to execute when misused. **→ Full procedure list in the
[Cypher reference](https://kglite.readthedocs.io/en/latest/guides/cypher.html#structural-validator-call-procedures).**

### Graph algorithms

Shortest path (BFS or Dijkstra), centrality, community detection,
clustering — all in Cypher:

```python
graph.cypher("""
    MATCH path = shortestPath((a:User {name:'Alice'})-[*]-(b:User {name:'Eve'}))
    RETURN path
""")
```

**→ [Graph algorithms guide](https://kglite.readthedocs.io/en/latest/guides/graph-algorithms.html) ·
[Traversal patterns](https://kglite.readthedocs.io/en/latest/guides/traversal-hierarchy.html) ·
[Recipes index](https://kglite.readthedocs.io/en/latest/guides/recipes.html).**

## Examples

The [`examples/`](https://github.com/kkollsga/kglite/tree/main/examples)
directory has runnable, self-contained artifacts:

- **[`codebase_to_claude_mcp.ipynb`](https://github.com/kkollsga/kglite/blob/main/examples/codebase_to_claude_mcp.ipynb)**
  — clone a famous open-source repo, parse it into a code knowledge
  graph, register a workspace MCP server in Claude Desktop. End-to-end
  in ~50 lines.
- **[`open_source_workspace_mcp.yaml`](https://github.com/kkollsga/kglite/blob/main/examples/open_source_workspace_mcp.yaml)**
  — annotated workspace-mode manifest for the github-clone-tracker
  pattern. Walked through in the
  [workspace manifest example](https://kglite.readthedocs.io/en/latest/examples/manifest_workspace.html).
- **[`legal_graph.py`](https://github.com/kkollsga/kglite/blob/main/examples/legal_graph.py)**
  — end-to-end `add_nodes` / `add_connections` from pandas DataFrames,
  covering laws, regulations, court decisions with citation edges.
- **[`code_graph.py`](https://github.com/kkollsga/kglite/blob/main/examples/code_graph.py)**
  — build a code knowledge graph from a source directory via
  `code_tree.build`.
- **[`spatial_graph.py`](https://github.com/kkollsga/kglite/blob/main/examples/spatial_graph.py)**
  — declarative CSV→graph loading via a JSON blueprint; lat/lon
  coordinates and pipeline-path traversal queries.
- **[`crates/kglite-mcp-server/`](https://github.com/kkollsga/kglite/tree/main/crates/kglite-mcp-server)**
  — Rust-native single-binary MCP server (built on rmcp + the
  [mcp-methods] framework). Reach for it when the manifest doesn't
  express what you need; the binary is the reference for layering
  domain-specific tools on top of the generic surface.

[mcp-methods]: https://github.com/kkollsga/mcp-methods

## Benchmarks

KGLite builds and queries Wikidata-scale graphs on a laptop. Measured
with [`bench/wiki_benchmark.py`](https://github.com/kkollsga/kglite/blob/main/bench/wiki_benchmark.py)
on an M-series MacBook.

**Ingest** — full pipeline from compressed N-Triples to a queryable graph:

| dataset   | triples | nodes  | edges  | ingest  | throughput       | peak RAM |
|-----------|--------:|-------:|-------:|--------:|------------------|---------:|
| wiki100m  |  100 M  |  938 K |  748 K |   29 s  | 3.4 M triples/s  |  1.3 GB  |
| wiki500m  |  500 M  |  5.6 M |  6.7 M |  157 s  | 3.2 M triples/s  |  5.2 GB  |
| wiki1000m |    1 B  | 14.7 M | 15.4 M |  395 s  | 2.5 M triples/s  |  7.0 GB  |

Reloading a saved 1 B-triple graph from disk (7 GB on-disk): **3.5 s**.

**Query latency on the 1 B-triple graph** (mapped storage):

| Cypher                                                              |     wall |
|---------------------------------------------------------------------|---------:|
| `MATCH (n)-[:P31]->(:human) RETURN count(n)` — typed aggregation    |   0.5 ms |
| `MATCH (a)-[:P31]->(b)-[:P279]->(c) LIMIT 10` — 2-hop typed         |   0.9 ms |
| `MATCH (a)-[:P31]->(b {nid:'Q64'}) RETURN a LIMIT 20` — pivot       |     1 ms |
| `MATCH (a)-[:P31]->(:human)` `MATCH (a)-[:P27]->(c) LIMIT 10` — join |   44 ms |

Disk and mapped storage build at the same speed; mapped wins on
small-result queries (in-memory inverted index), disk wins on
unbounded typed traversals (sorted-CSR mmap I/O). No server, no
tuning, same Python process as your code.

## Key Features

Quick reference. Each links into the appropriate guide.

| Feature | Description |
|---|---|
| **[Cypher](https://kglite.readthedocs.io/en/latest/guides/cypher.html)** | MATCH, CREATE, SET, DELETE, MERGE, UNION/INTERSECT/EXCEPT, aggregations (incl. `median`, `percentile_cont`, `variance`), `reduce()`, ORDER BY, LIMIT, SKIP |
| **[Semantic search](https://kglite.readthedocs.io/en/latest/guides/semantic-search.html)** | Vector embeddings + `text_score()` for similarity ranking. Opt-in via `pip install 'kglite[embed]'`. |
| **Text predicates** | `text_edit_distance`, `text_normalize`, `text_jaccard`, `text_ngrams`, `text_contains_any` / `text_starts_with_any` |
| **[Graph algorithms](https://kglite.readthedocs.io/en/latest/guides/graph-algorithms.html)** | Shortest path (BFS or Dijkstra), centrality, community detection, clustering |
| **Structural validators** | 14 `CALL` procedures: `orphan_node`, `missing_required_edge`, `cycle_2step`, `inverse_violation`, `cardinality_violation`, `parallel_edges`, `null_property`, more — agent-discoverable integrity checks composable with Cypher |
| **[Spatial](https://kglite.readthedocs.io/en/latest/guides/spatial.html)** | Coordinates, WKT geometry, distance + containment, geometry primitives (`geom_buffer`, `geom_convex_hull`, `geom_union/intersection/difference`, `geom_is_valid`, `geom_length`), `kg_knn` k-nearest-neighbour |
| **[Timeseries](https://kglite.readthedocs.io/en/latest/guides/timeseries.html)** | Time-indexed data with `ts_*()` Cypher functions |
| **[Bulk loading](https://kglite.readthedocs.io/en/latest/guides/data-loading.html)** | `add_nodes` / `add_connections` for DataFrames |
| **[Blueprints](https://kglite.readthedocs.io/en/latest/guides/blueprints.html)** | Declarative CSV-to-graph loading via JSON config |
| **[Import/Export](https://kglite.readthedocs.io/en/latest/guides/import-export.html)** | Save/load snapshots (`.kgl`), GraphML, CSV export |
| **[AI integration](https://kglite.readthedocs.io/en/latest/guides/ai-agents.html)** | `describe()` introspection, MCP server, agent prompts |
| **[Code analysis](https://kglite.readthedocs.io/en/latest/guides/code-tree.html)** | 13-language tree-sitter parser (`kglite.code_tree`) — functions, classes, calls, imports, web-framework routes |

## Documentation

Full docs at **[kglite.readthedocs.io](https://kglite.readthedocs.io)**:

**Getting started**
- [Getting Started](https://kglite.readthedocs.io/en/latest/getting-started.html) — installation, first graph, core concepts
- [Querying overview](https://kglite.readthedocs.io/en/latest/guides/querying.html) — Cypher vs fluent API, when to reach for which
- [Recipes index](https://kglite.readthedocs.io/en/latest/guides/recipes.html) — copy-paste patterns for common shapes

**Querying**
- [Cypher Guide](https://kglite.readthedocs.io/en/latest/guides/cypher.html) — MATCH, MERGE, mutations, parameters, validators
- [Traversal & hierarchy](https://kglite.readthedocs.io/en/latest/guides/traversal-hierarchy.html) — variable-length paths, tree walks
- [Graph algorithms](https://kglite.readthedocs.io/en/latest/guides/graph-algorithms.html) — shortest path, PageRank, community detection
- [Semantic Search](https://kglite.readthedocs.io/en/latest/guides/semantic-search.html) — embeddings, vector search, hybrid retrieval

**Loading data**
- [Data Loading](https://kglite.readthedocs.io/en/latest/guides/data-loading.html) — DataFrames in, DataFrames out
- [Blueprints](https://kglite.readthedocs.io/en/latest/guides/blueprints.html) — declarative CSV→graph via JSON config
- [Datasets](https://kglite.readthedocs.io/en/latest/guides/datasets.html) — Wikidata + Sodir wrappers
- [Code analysis](https://kglite.readthedocs.io/en/latest/guides/code-tree.html) — `code_tree.build`, framework route detection
- [Import / Export](https://kglite.readthedocs.io/en/latest/guides/import-export.html) — `.kgl` snapshots, GraphML, CSV

**Domain features**
- [Spatial](https://kglite.readthedocs.io/en/latest/guides/spatial.html) — WKT geometry, lat/lon, k-nearest-neighbour
- [Timeseries](https://kglite.readthedocs.io/en/latest/guides/timeseries.html) — time-indexed values, `ts_*()` functions

**Agent integration**
- [AI Agents](https://kglite.readthedocs.io/en/latest/guides/ai-agents.html) — MCP server, `describe()`, agent prompts
- [MCP server config](https://kglite.readthedocs.io/en/latest/guides/mcp-servers.html) — manifests, skills, extensions

**Reference**
- [API Reference](https://kglite.readthedocs.io/en/latest/autoapi/kglite/index.html) — full auto-generated reference

## Requirements

Python 3.10+ (CPython) | macOS (ARM), Linux (x86_64/aarch64), Windows (x86_64) | `pandas >= 1.5`

## License

MIT — see [LICENSE](https://github.com/kkollsga/kglite/blob/main/LICENSE) for details.
