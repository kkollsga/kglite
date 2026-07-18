# KGLite — Knowledge graph for Python, built for LLM agents

[![PyPI version](https://img.shields.io/pypi/v/kglite)](https://pypi.org/project/kglite/)
[![Python versions](https://img.shields.io/pypi/pyversions/kglite)](https://pypi.org/project/kglite/)
[![crates.io](https://img.shields.io/crates/v/kglite)](https://crates.io/crates/kglite)
[![docs.rs](https://img.shields.io/docsrs/kglite)](https://docs.rs/kglite)
[![License: MIT](https://img.shields.io/pypi/l/kglite)](https://github.com/kkollsga/kglite/blob/main/LICENSE)
[![Docs](https://img.shields.io/readthedocs/kglite)](https://kglite.readthedocs.io)

KGLite is an embedded, Cypher-queryable knowledge graph for Python and Rust,
built so the same graph can serve an application, an analyst, or an LLM agent.
The Python wheel has no required Python runtime dependencies; the graph engine
runs in-process without an external database service. The distribution also
includes a CLI and MCP server, prompt-shaped `describe()` introspection, and
structural validators that compose with Cypher.

## Start here

**Install → build one graph → query it:**

```bash
pip install kglite
```

```python
import kglite

graph = kglite.from_records({"nodes": [{
    "type": "Person", "id_field": "id", "title_field": "name",
    "records": [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}],
}]})
print(graph.cypher("MATCH (p:Person) RETURN p.name ORDER BY p.name").to_dicts())
```

Choose the path that matches what you are doing:

- **[Getting Started](https://kglite.readthedocs.io/en/latest/python/getting-started.html)** — install, first graph, storage choices
- **[Python API](https://kglite.readthedocs.io/en/latest/autoapi/kglite/index.html)** · **[Cypher](https://kglite.readthedocs.io/en/latest/reference/cypher-reference.html)** · **[Fluent API](https://kglite.readthedocs.io/en/latest/reference/fluent-api.html)**
- **[MCP and agents](https://kglite.readthedocs.io/en/latest/python/guides/mcp-servers.html)** · **[Rust](https://kglite.readthedocs.io/en/latest/rust/index.html)** · **[Operators](https://kglite.readthedocs.io/en/latest/operators/index.html)**
- **[0.13 → 0.14 migration](https://kglite.readthedocs.io/en/latest/python/migrations/0.13-to-0.14.html)** · **[all documentation](https://kglite.readthedocs.io)**

For DataFrame loading, install the optional pandas integration with
`pip install "kglite[pandas]"`; the complete walkthrough is in
**[Quick Start](#quick-start)**.

> kglite is a **pure-Rust knowledge graph engine**
> ([`crates/kglite`](https://github.com/kkollsga/kglite/tree/main/crates/kglite))
> packaged for Python via `pip install kglite`. The interactive shell,
> Bolt-server, and MCP-server binaries are sibling Rust crates wrapping
> the same engine. If you want kglite as a Rust library — without the
> Python wheel in your build — see **[Use from Rust](#use-from-rust)** below.

> **Interactive shell.** `pip install kglite` also gives you the `kglite`
> command — a `sqlite3`-style REPL: `kglite app.kgl` opens a Cypher prompt with
> `.import`, `.dump`, `.schema`, multi-line input, and tab-completion. For a
> standalone CLI-only install, use `pip install kglite-cli` or `cargo install
> kglite-cli`.

## Ecosystem

kglite is the engine. Two companion projects build graphs it serves — each
released and versioned on its own cadence:

- **[kglite](https://github.com/kkollsga/kglite)** — the embedded Cypher
  knowledge-graph engine (this project): graph + Cypher + fluent API + bundled
  MCP server.
- **[codingest](https://codingest.readthedocs.io)** — parses codebases into
  code graphs (14 languages, web-framework route detection). Build with it,
  query the `.kgl` here. Requires kglite ≥ 0.14.
- **[kglite-datasets](https://kglite-datasets.readthedocs.io)** —
  fetch-build-cache loaders for public registries (SEC EDGAR, Wikidata, Sodir).
- **[sonagram](https://sonagram.readthedocs.io)** — turns a local music
  library into a kglite knowledge graph via sonara audio analysis (tempo,
  energy, mood, key); AI agents curate playlists over it through a simple
  bundled skill and CLI (`pip install sonagram`).

**Upgrading from 0.13?** The code-graph builder and dataset loaders moved out
of the wheel in 0.14 — see the
[0.13 → 0.14 migration guide](https://kglite.readthedocs.io/en/latest/python/migrations/0.13-to-0.14.html).
Pin back anytime with `pip install "kglite<0.14"`.

## Use cases

The same agent-facing surface works whether the graph holds legal
precedents, a Wikidata slice, a SQL warehouse, a RAG corpus, or a
parsed codebase.

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
  [Data Loading guide](https://kglite.readthedocs.io/en/latest/python/guides/data-loading.html).**
- 🌐 **Public datasets.** Pre-packaged loaders for SEC EDGAR, Wikidata,
  and Sodir live in the companion **kglite-datasets** project — they
  handle the *fetch + build + cache* cycle and return a queryable
  `KnowledgeGraph`. kglite's mapped and disk storage then query graphs
  that don't fit in RAM — a billion-edge Wikidata graph on a 16 GB
  laptop. **→ See [Public datasets](#public-datasets) below.**
- 📚 **RAG with structure.** Documents, chunks, entities, and the
  edges between them in one graph. Combine `text_score()` vector
  similarity with Cypher traversal — *"find court cases semantically
  similar to my fact pattern, then walk one hop to related
  precedents"* — hybrid retrieval in one query, no second vector DB.
  Scale to large corpora with an opt-in HNSW index
  (`build_vector_index()`).
  **→ [Semantic Search guide](https://kglite.readthedocs.io/en/latest/python/guides/semantic-search.html).**
- 📂 **Codebase analysis.** The [codingest](https://github.com/kkollsga/codingest) builder parses 14
  languages into Function / Class / Module / Route nodes with
  web-framework route detection (Flask, FastAPI, Django). Build from
  any git revision, or merge several into one multi-revision graph for
  structural diffs (multi-rev builds). kglite serves and queries those graphs. **The
  builder and the code → Claude Desktop workflow live in the codingest
  project.**
- 🤝 **A shared graph as an agent contract.** One `.kgl` can be the
  two-way contract between collaborating agents (e.g. a *research* agent
  that batch-rebuilds specs and *coding* agents that plan and mutate
  status live). The primitives that make this safe are first-class:
  **ownership layers** (`define_schema(layer='managed'|'runtime')` +
  `add_nodes(managed_reload=True)` so a rebuild provably can't clobber
  agent-owned nodes), **role-scoped writes**
  (`cypher(..., write_scope=[...])` rejects out-of-scope CREATE/SET), a
  verbatim **instructions slot** at the top of `describe()`
  (`set_instructions(text)`), **native list properties**, JSON-native
  ingestion (`from_records(spec)`), and a **dependency frontier**
  (`CALL ready_set(...)`) to find the next actionable work. Keep the
  graph general — these are small, opt-in building blocks, not a baked-in
  workflow.
- 🧠 **Markdown knowledge bases & agent memory.** `kglite.okf.build(dir)`
  ingests an [Open Knowledge Format](https://github.com/GoogleCloudPlatform/knowledge-catalog)
  bundle — or a Claude memory dir, skills folder, or Obsidian vault — into a
  graph: frontmatter → node properties, markdown links → typed edges. Then
  cluster it (`CALL leiden`), find orphaned or stale notes, and surface dangling
  references — the query engine OKF itself doesn't ship. **→
  [OKF guide](https://kglite.readthedocs.io/en/latest/python/guides/okf.html).**

## Why Cypher?

Questions over connected data — *which insiders sold this stock, who
sits on two boards, what cites this case* — are pattern matches. In
SQL they become multi-table joins; in Cypher the pattern is the
query:

```cypher
-- Insider sells, most recent first
MATCH (t:InsiderTransaction {direction: 'sale'})-[:BY_INSIDER]->(p:Person)
MATCH (t)-[:IN_COMPANY]->(c:Company)
RETURN p.title, c.title, t.shares, t.price_per_share
ORDER BY t.transaction_date DESC LIMIT 10
```

Cypher pays off most when the data has real structure and your
questions traverse it.

## How it compares

|                                            | KGLite                            | [LadybugDB](https://ladybugdb.com/) (formerly Kuzu) | NetworkX           | rustworkx          | Neo4j Embedded         |
|--------------------------------------------|-----------------------------------|-----------------------------------------------------|--------------------|--------------------|------------------------|
| **Install**                                | `pip install kglite`              | `pip install ladybug`                               | `pip install networkx` | `pip install rustworkx` | JVM + Java deps  |
| **Query language**                         | Cypher ([broad coverage](CYPHER.md#feature-coverage)) | Cypher                              | Python API         | Python API         | Cypher (full)          |
| **Storage**                                | in-mem · mmap · disk (1B+ edges)  | in-mem · disk (columnar)                            | in-mem             | in-mem             | in-mem · disk (JVM)    |
| **Bulk-load from pandas**                  | one-liner                         | via Arrow                                           | manual             | manual             | via driver             |
| **MCP server for LLM agents**              | bundled in the `kglite` wheel     | [separate `mcp-server-ladybug` install](https://github.com/LadybugDB/mcp-server-ladybug) | — | — | — |
| **`describe()` schema for LLM prompts**    | ✅                                 | —                                                   | —                  | —                  | —                      |
| **Embeddable in Rust** (no Python in build) | pure-Rust [`kglite`](https://crates.io/crates/kglite) crate | [`lbug`](https://crates.io/crates/lbug) bindings to the C++ engine | — | ✅ | — |
| **License**                                | MIT                               | MIT                                                 | BSD-3              | Apache-2           | GPLv3                  |

**Pick KGLite** when you want one embedded package that combines Python and
pure-Rust Cypher APIs with a bundled MCP binary, prompt-shaped `describe()`, and
agent-contract primitives: role-scoped writes (`write_scope`), ownership layers,
`set_instructions`, and `CALL ready_set(...)` — with companion projects
([codingest](https://codingest.readthedocs.io),
[kglite-datasets](https://kglite-datasets.readthedocs.io)) that build code and
public-registry graphs it serves. **Pick LadybugDB** when columnar analytical scans and
its broader language ecosystem are the priority; it also provides Rust
bindings and a separately installed MCP server. **Pick NetworkX** when you need
its enormous graph-algorithm library and your data fits in RAM. **Pick
rustworkx** when you want a Rust-backed Python graph API with no query language.
**Pick Neo4j Embedded** when you've standardised on server-mode Cypher and want
the in-process driver for tests.

📊 **[Benchmarks →](BENCHMARKS.md)** — wall-to-wall time per topic (load,
filter/aggregate, traversal, pathfinding, algorithms, mutations) against
other embedded graph engines, NetworkX, rustworkx, igraph, and DuckDB on
one shared synthetic graph. Reproduce with `python benchmarks/benchmark.py`.

## Quick Start

```bash
# Python (the headline distribution path)
pip install kglite

# Optional extras
pip install 'kglite[pandas]'   # DataFrame loading used in the walkthrough below
pip install fastembed            # (or sentence-transformers) embedding models for text_score() — bring your own
pip install 'kglite[neo4j]'      # Neo4j Python driver for Bolt-server tests
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

# Query — returns a ResultView; eligible projections stay lazy until accessed.
for row in graph.cypher("""
    MATCH (p:Person) WHERE p.age > 30
    RETURN p.name AS name, p.city AS city
    ORDER BY p.age DESC
"""):
    print(row['name'], row['city'])

# Or get a pandas DataFrame directly.
df = graph.cypher("MATCH (p:Person) RETURN p.name, p.age ORDER BY p.age", to_df=True)

# Persist to disk and reload. save() is atomic + fsync by default (crash-safe —
# no torn file); load() raises a typed kglite.FileFormatError on a corrupt file.
graph.save("my_graph.kgl")
loaded = kglite.load("my_graph.kgl")

# Or serialize to/from bytes (no filesystem path):
blob = graph.to_bytes(); loaded = kglite.from_bytes(blob)

# Share read-only across threads with an immutable, lock-free snapshot:
snapshot = graph.freeze()        # concurrent snapshot.cypher(...) from many threads

# No data yet? Generate a realistic demo graph in one line (bundled, no extra deps):
demo = kglite.graphgen("medium")               # ~25k nodes, ready to query
# kglite.graphgen("huge", out="/tmp/g")        # stream millions of nodes to CSV, bounded memory
```

**→ [Getting Started guide](https://kglite.readthedocs.io/en/latest/python/getting-started.html) ·
[Cypher reference](https://kglite.readthedocs.io/en/latest/python/guides/cypher.html) ·
[API reference](https://kglite.readthedocs.io/en/latest/autoapi/kglite/index.html).**

Prefer a runnable file? [`examples/csv_to_graph.py`](https://github.com/kkollsga/kglite/blob/main/examples/csv_to_graph.py)
loads real CSVs end to end.

## Serve it to an agent

Use the KGLite MCP server when you want a graph kept warm across many calls,
with typed graph-query and lifecycle tools. Code-graph construction, repository
cloning, and code-watch workflows belong to **codingest-mcp**, which embeds the
same KGLite graph-serving surface.

### One command — any `.kgl` becomes an MCP server

```bash
kglite-mcp-server --graph path/to/graph.kgl
```

The server exposes `cypher_query`, `graph_overview`, schema introspection, and
structural validators over MCP stdio. When a valid `source_root` is configured,
it also exposes source-file read/search tools. Drop it into Claude Desktop,
Cursor, or another MCP-capable client and any KGLite graph is queryable.

When you register it, point `command` at the **absolute path** to the
binary (`/abs/path/to/venv/bin/kglite-mcp-server`), not a bare name — a
bare command can silently launch an older PATH-shadowing install. Then
confirm it with `kglite-mcp-server --selftest --graph path/to/graph.kgl`,
which drives a real handshake and prints green/red per capability.

**Two ready-made code-intelligence recipes** ship in
[`examples/`](examples/) — both build code graphs, so run them under
**codingest-mcp** (it embeds this same tool surface and injects the builder):

- **Clone-and-explore GitHub repos** —
  [`open_source_workspace_mcp.yaml`](examples/open_source_workspace_mcp.yaml):
  the agent calls `repo_management('org/repo')` to clone and build a
  code graph on demand.
- **Review a local directory** —
  [`local_code_review_mcp.yaml`](examples/local_code_review_mcp.yaml):
  point it at a checked-out tree, `set_root_dir(path)` to swap roots,
  watch-mode auto-rebuild.

### Customise with a YAML manifest

Drop `<basename>_mcp.yaml` next to the graph (e.g. `wikidata_mcp.yaml`
beside `wikidata.kgl`) and the server auto-loads it at boot.

```yaml
name: Wikidata Explorer
source_root: /path/to/related/source        # exposes read/grep/list
skills: true                                # load bundled + project tool guidance
trust:
  allow_embedder: true
extensions:
  embedder: { library: fastembed, model: BAAI/bge-small-en-v1.5 }  # enables text_score()
  csv_http_server: true                              # bulk CSV exports
tools:                                               # inline parameterised Cypher
  - name: who_invented
    cypher: |
      MATCH (i:Q5)-[:P61]->(t {label:$thing})
      RETURN i.label LIMIT 5
```

No fork required for most customisation. **→
[MCP server guide](https://kglite.readthedocs.io/en/latest/python/guides/mcp-servers.html).**

### Teach the MCP agent with bundled tool skills

With `skills: true`, Markdown skill files (`<basename>.skills/*.md`) provide
methodology for each tool. The agent reads `cypher_query.md` to learn
your schema conventions, `read_code_source.md` to know when to drill
into source vs. query the graph, etc. Three layers compose:
kglite-bundled defaults + your project's `.skills/` overrides +
operator-declared domain packs. Skills with `applies_when:` predicates
only activate when the graph contains the relevant node types — so a
non-code graph never sees `read_code_source` methodology.

Net effect: the agent comes pre-loaded with how to use your graph,
rather than discovering it through trial-and-error. **→
[AI Agents guide](https://kglite.readthedocs.io/en/latest/python/guides/ai-agents.html).**

## Public datasets

Pre-packaged loaders that turn well-known public sources into queryable
graphs — **SEC EDGAR** filings (insider transactions, institutional
holdings, board composition, XBRL financials), **Wikidata** (the full
`latest-truthy` RDF dump, parallel-decoded and built into a billion-edge
graph), and **Sodir** (Norwegian Offshore Directorate petroleum data) —
live in the companion **[kglite-datasets](https://kglite-datasets.readthedocs.io)**
project. Install it separately with `pip install kglite-datasets`; its Python
package supplies the dataset-specific loaders while KGLite supplies the graph:

```python
import kglite_datasets  # choose a loader from the companion documentation
```

Each loader handles the
*fetch + build + cache* cycle and returns a `KnowledgeGraph` you can
`cypher()` against; kglite serves and queries the graphs they produce.
The core graph engine does not require network access; fetching public data is
an explicit companion-project operation.

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

Vector embeddings via a bring-your-own embedder — `pip install fastembed` (or
`sentence-transformers`) and pass it to `g.set_embedder(...)`. **→ [Semantic Search guide](https://kglite.readthedocs.io/en/latest/python/guides/semantic-search.html).**

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
[Cypher reference](https://kglite.readthedocs.io/en/latest/python/guides/cypher.html#structural-validator-call-procedures).**

### Graph algorithms

Shortest path (BFS or Dijkstra), centrality, community detection,
clustering — all in Cypher:

```python
graph.cypher("""
    MATCH path = shortestPath((a:User {name:'Alice'})-[*]-(b:User {name:'Eve'}))
    RETURN path
""")
```

**→ [Graph algorithms guide](https://kglite.readthedocs.io/en/latest/python/guides/graph-algorithms.html) ·
[Traversal patterns](https://kglite.readthedocs.io/en/latest/python/guides/traversal-hierarchy.html) ·
[Recipes index](https://kglite.readthedocs.io/en/latest/python/guides/recipes.html).**

## Use from Rust

The same engine is available as a pure-Rust crate — embed it in a
Rust binary without the Python wheel in your build:

```toml
# Cargo.toml
[dependencies]
kglite = "0.14"
```

```rust
use kglite::api::{io::load_file, session, Value};
use std::collections::HashMap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let graph = load_file("my_graph.kgl")?;     // same .kgl as Python writes
    let params = HashMap::new();
    let opts = session::ExecuteOptions::eager(&params);
    let outcome = session::execute_read(
        &graph,
        "MATCH (p:Person) RETURN p.name LIMIT 5",
        &opts,
    )?;
    for row in &outcome.result.rows {
        if let Some(Value::String(name)) = row.first() {
            println!("{}", name);
        }
    }
    Ok(())
}
```

Zero PyO3 in the dependency tree:
`cargo tree -p your-crate | rg pyo3` → empty.

- **[Rust quickstart](https://kglite.readthedocs.io/en/latest/rust/index.html)**
  — load + query + transaction examples.
- **[Embedding guide](https://kglite.readthedocs.io/en/latest/rust/embedding.html)**
  — workspace layout, the `kglite::api::*` surface, cgo / napi /
  JNI sketches.
- **[Session abstraction](https://kglite.readthedocs.io/en/latest/rust/session.html)**
  — binding-implementer reference for the canonical Cypher pipeline.
- **[API reference (docs.rs)](https://docs.rs/kglite)** — per-symbol Rust API docs.

The Bolt server (`crates/kglite-bolt-server`) and the Rust MCP
server (`crates/kglite-mcp-server`) are standalone binaries built
on the same engine — see the
[Operators guide](https://kglite.readthedocs.io/en/latest/operators/bolt-server.html)
for deployment.

For **non-Rust language bindings** (Go via cgo, JavaScript via napi,
JVM via JNI, .NET via P/Invoke), the
[`crates/kglite-c`](https://github.com/kkollsga/kglite/tree/main/crates/kglite-c)
crate exposes the engine through a stable C ABI covering lifecycle, sessions,
Cypher, results, persistence, and embedders, plus a cbindgen-generated
`kglite.h`. See
[`docs/rust/c-abi.md`](https://kglite.readthedocs.io/en/latest/rust/c-abi.html)
for the design and
[`docs/rust/implementing-a-binding.md`](https://kglite.readthedocs.io/en/latest/rust/implementing-a-binding.html)
for cgo / napi / JNI worked examples.

## Examples

The [`examples/`](https://github.com/kkollsga/kglite/tree/main/examples)
directory has runnable, self-contained artifacts:

- **[`open_source_workspace_mcp.yaml`](https://github.com/kkollsga/kglite/blob/main/examples/open_source_workspace_mcp.yaml)**
  — annotated workspace-mode manifest for the github-clone-tracker
  pattern. Walked through in the
  [workspace manifest example](https://kglite.readthedocs.io/en/latest/python/examples/manifest_workspace.html).
- **[`csv_to_graph.py`](https://github.com/kkollsga/kglite/blob/main/examples/csv_to_graph.py)**
  — minimal `pd.read_csv` → `add_nodes` / `add_connections` walkthrough
  on a tiny org chart, with a few Cypher queries. The fastest way in.
- **[`incremental_update.py`](https://github.com/kkollsga/kglite/blob/main/examples/incremental_update.py)**
  — merge a second data snapshot into an existing graph with
  `add_nodes(conflict_handling='update')`.
- **[`legal_graph.py`](https://github.com/kkollsga/kglite/blob/main/examples/legal_graph.py)**
  — end-to-end `add_nodes` / `add_connections` from pandas DataFrames,
  covering laws, regulations, court decisions with citation edges.
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

Reproducible, versioned comparisons live in **[BENCHMARKS.md](BENCHMARKS.md)**.
Run the public harness with `python benchmarks/benchmark.py`; maintainer-only
storage and release-regression probes live under `tests/benchmarks/`.

## Key Features

Quick reference. Each links into the appropriate guide.

| Feature | Description |
|---|---|
| **[Cypher](https://kglite.readthedocs.io/en/latest/python/guides/cypher.html)** | MATCH, CREATE, SET, DELETE, MERGE, UNION/INTERSECT/EXCEPT, aggregations (incl. `median`, `percentile_cont`, `variance`), `reduce()`, ORDER BY, LIMIT, SKIP |
| **[Semantic search](https://kglite.readthedocs.io/en/latest/python/guides/semantic-search.html)** | Vector embeddings + `text_score()` for similarity ranking. Bring your own embedder (`pip install fastembed` or `sentence-transformers`). |
| **Text predicates** | `text_edit_distance`, `text_normalize`, `text_jaccard`, `text_ngrams`, `text_contains_any` / `text_starts_with_any` |
| **[Graph algorithms](https://kglite.readthedocs.io/en/latest/python/guides/graph-algorithms.html)** | Shortest path (BFS or Dijkstra), centrality, community detection, clustering |
| **Structural validators** | 14 `CALL` procedures: `orphan_node`, `missing_required_edge`, `cycle_2step`, `inverse_violation`, `cardinality_violation`, `parallel_edges`, `null_property`, more — agent-discoverable integrity checks composable with Cypher |
| **[Spatial](https://kglite.readthedocs.io/en/latest/python/guides/spatial.html)** | Coordinates, WKT geometry, distance + containment, `kg_knn` k-nearest-neighbour. Pragmatic primitives, not a full GIS stack. |
| **[Timeseries](https://kglite.readthedocs.io/en/latest/python/guides/timeseries.html)** | Time-indexed values with `ts_*()` Cypher functions. For graphs whose nodes carry value-over-time series. |
| **[Bulk loading](https://kglite.readthedocs.io/en/latest/python/guides/data-loading.html)** | `add_nodes` / `add_connections` for DataFrames |
| **[Blueprints](https://kglite.readthedocs.io/en/latest/python/guides/blueprints.html)** | Declarative CSV-to-graph loading via JSON config |
| **[Import/Export](https://kglite.readthedocs.io/en/latest/python/guides/import-export.html)** | Save/load snapshots (`.kgl`), GraphML, CSV export |
| **[AI integration](https://kglite.readthedocs.io/en/latest/python/guides/ai-agents.html)** | `describe()` introspection, MCP server, agent prompts |
| **Code analysis** | serve + query 14-language code graphs built by the codingest project — functions, classes, calls, imports, web-framework routes |
| **[OKF ingestion](https://kglite.readthedocs.io/en/latest/python/guides/okf.html)** | Markdown + YAML-frontmatter bundles (`kglite.okf`) — Open Knowledge Format, Claude memory dirs, skills, Obsidian vaults → frontmatter as properties, links as typed edges |
| **Public dataset loaders** | Fetch-build-cache loaders for public sources — SEC EDGAR filings, Wikidata, Sodir (Norwegian Offshore Directorate) — live in the companion kglite-datasets project; each returns a queryable `KnowledgeGraph` kglite serves |

## Documentation

Full docs at **[kglite.readthedocs.io](https://kglite.readthedocs.io)**
— five tracks by audience.

**[Python track](https://kglite.readthedocs.io/en/latest/python/index.html)** — `pip install kglite`
- [Getting Started](https://kglite.readthedocs.io/en/latest/python/getting-started.html) — installation, first graph, core concepts
- [Cypher Guide](https://kglite.readthedocs.io/en/latest/python/guides/cypher.html) — MATCH, MERGE, mutations, parameters, validators
- [Data Loading](https://kglite.readthedocs.io/en/latest/python/guides/data-loading.html) — DataFrames in, DataFrames out
- [Graph algorithms](https://kglite.readthedocs.io/en/latest/python/guides/graph-algorithms.html) — shortest path, PageRank, community detection
- [Semantic Search](https://kglite.readthedocs.io/en/latest/python/guides/semantic-search.html) — embeddings, vector search, hybrid retrieval
- [OKF ingestion](https://kglite.readthedocs.io/en/latest/python/guides/okf.html) — `okf.build`, markdown knowledge bases & agent memory
- [MCP server config](https://kglite.readthedocs.io/en/latest/python/guides/mcp-servers.html) — manifests, skills, extensions
- [Spatial](https://kglite.readthedocs.io/en/latest/python/guides/spatial.html) · [Timeseries](https://kglite.readthedocs.io/en/latest/python/guides/timeseries.html) · [Blueprints](https://kglite.readthedocs.io/en/latest/python/guides/blueprints.html) · [Import/Export](https://kglite.readthedocs.io/en/latest/python/guides/import-export.html) · [Traversal & hierarchy](https://kglite.readthedocs.io/en/latest/python/guides/traversal-hierarchy.html) · [AI Agents](https://kglite.readthedocs.io/en/latest/python/guides/ai-agents.html)
- [Recipes index](https://kglite.readthedocs.io/en/latest/python/guides/recipes.html) — copy-paste patterns for common shapes

**[Rust track](https://kglite.readthedocs.io/en/latest/rust/index.html)** — `cargo add kglite`
- [Rust quickstart](https://kglite.readthedocs.io/en/latest/rust/index.html) — load, query, transactions
- [Embedding kglite](https://kglite.readthedocs.io/en/latest/rust/embedding.html) — surface tour, language-binding sketches
- [Session abstraction](https://kglite.readthedocs.io/en/latest/rust/session.html) — pipeline + CoW transactions
- [API manifest](https://kglite.readthedocs.io/en/latest/rust/api-reference.html) + [per-symbol docs.rs](https://docs.rs/kglite)

**[Operators](https://kglite.readthedocs.io/en/latest/operators/index.html)** — running the protocol servers
- [Bolt server](https://kglite.readthedocs.io/en/latest/operators/bolt-server.html) — Neo4j wire compat for cluster-aware drivers

**Reference** — cross-binding
- [Cypher reference](https://kglite.readthedocs.io/en/latest/reference/cypher-reference.html) — the supported Cypher subset
- [Fluent API reference](https://kglite.readthedocs.io/en/latest/reference/fluent-api.html) — programmatic graph construction
- [Python API (auto)](https://kglite.readthedocs.io/en/latest/autoapi/kglite/index.html) — auto-generated from stubs

**[Concepts](https://kglite.readthedocs.io/en/latest/concepts/index.html)** — architecture + contributor docs
- [Architecture](https://kglite.readthedocs.io/en/latest/concepts/architecture.html) · [Design decisions](https://kglite.readthedocs.io/en/latest/concepts/design-decisions.html) · [Cypher conformance](https://kglite.readthedocs.io/en/latest/concepts/cypher-conformance.html) · [Concurrency](https://kglite.readthedocs.io/en/latest/concepts/concurrency.html)

## Requirements

CPython 3.10+ | macOS (arm64/x86_64), Linux (glibc/musl; x86_64 and
best-effort aarch64), Windows (x86_64). The base wheel has no Python runtime
dependencies; integrations install their named extras. See the
[artifact support policy](https://kglite.readthedocs.io/en/latest/python/platform-support.html)
for the tested/build-only tiers, libc floors, PyPy status, and source-build
fallback.

## Stability

KGLite is beta software and remains pre-1.0. Patch releases preserve public
source APIs; a 0.x minor release may make an intentional breaking source-API
change when it is documented with a migration path. Saved graph files have a
separate format lifecycle: a release either reads an older format or refuses it
with an explicit rebuild/migration error. See the current
[0.13 → 0.14 migration guide](https://kglite.readthedocs.io/en/latest/python/migrations/0.13-to-0.14.html)
and [CHANGELOG.md](https://github.com/kkollsga/kglite/blob/main/CHANGELOG.md).
Storage parity and differential Cypher oracles run on every change.

## License

MIT — see [LICENSE](https://github.com/kkollsga/kglite/blob/main/LICENSE) for details.
