# KGLite: a knowledge graph for Python, built for LLM agents

[![PyPI version](https://img.shields.io/pypi/v/kglite)](https://pypi.org/project/kglite/)
[![Python versions](https://img.shields.io/pypi/pyversions/kglite)](https://pypi.org/project/kglite/)
[![crates.io](https://img.shields.io/crates/v/kglite)](https://crates.io/crates/kglite)
[![docs.rs](https://img.shields.io/docsrs/kglite)](https://docs.rs/kglite)
[![License: MIT](https://img.shields.io/pypi/l/kglite)](https://github.com/kkollsga/kglite/blob/main/LICENSE)
[![Docs](https://img.shields.io/readthedocs/kglite)](https://kglite.readthedocs.io)

KGLite is an embedded, Cypher-queryable knowledge graph for Python and Rust,
built so the same graph can serve an application, an analyst, or an LLM agent.
The Python wheel has no required Python runtime dependencies; the graph engine
runs in-process without an external database service. Every crate ships under
MIT. If you are embedding a graph engine in something you distribute, see
**[Licensing and embedded distribution](#licensing-and-embedded-distribution)**.

## Quick Start

```bash
pip install kglite             # the DataFrame walk-through below assumes pandas
pip install fastembed          # (or sentence-transformers) bring-your-own embedder for text_score()
```

```python
import pandas as pd
import kglite

# Three storage modes, picked by graph size: default (in-memory) is fastest,
# storage="mapped" mmaps columns as you grow, storage="disk", path=… goes to
# 100M+ nodes, Wikidata-scale, loaded lazily.
graph = kglite.KnowledgeGraph()

# Bulk-load nodes from a DataFrame.
people = pd.DataFrame({
    "id":   ["alice", "bob", "eve"],   "name": ["Alice", "Bob", "Eve"],
    "age":  [28, 35, 41],              "city": ["Oslo", "Bergen", "Trondheim"],
})
graph.add_nodes(people, node_type="Person", unique_id_field="id", node_title_field="name")

# Bulk-load relationships the same way.
knows = pd.DataFrame({"src": ["alice", "bob"], "tgt": ["bob", "eve"]})
graph.add_connections(knows, connection_type="KNOWS",
                      source_type="Person", source_id_field="src",
                      target_type="Person", target_id_field="tgt")

# Query: returns a ResultView; eligible projections stay lazy until accessed.
for row in graph.cypher("""
    MATCH (p:Person) WHERE p.age > 30
    RETURN p.name AS name, p.city AS city
    ORDER BY p.age DESC
"""):
    print(row['name'], row['city'])

# Or get a pandas DataFrame directly.
df = graph.cypher("MATCH (p:Person) RETURN p.name, p.age ORDER BY p.age", to_df=True)

# Persist and reload. save() is atomic + fsync by default (crash-safe, no torn
# file); load() raises a typed kglite.FileFormatError on a corrupt file.
graph.save("my_graph.kgl")
loaded = kglite.load("my_graph.kgl")
blob = graph.to_bytes(); loaded = kglite.from_bytes(blob)   # or without a path

# Immutable, lock-free snapshot: concurrent snapshot.cypher(...) from many threads.
snapshot = graph.freeze()

# No data yet? A realistic demo graph in one line (bundled, no extra deps):
demo = kglite.graphgen("medium")               # ~25k nodes, ready to query
```

Then hand the same file to an agent. The MCP server is bundled in the wheel:

```bash
kglite-mcp-server --graph my_graph.kgl
```

**→ [MCP servers guide](https://kglite.readthedocs.io/en/latest/python/guides/mcp-servers.html) ·
[CLI guide](https://kglite.readthedocs.io/en/latest/operators/cli.html).** Prefer a
runnable file? [`examples/csv_to_graph.py`](https://github.com/kkollsga/kglite/blob/main/examples/csv_to_graph.py)
loads real CSVs end to end.

Two guides cover most first sessions:

- **[Getting Started](https://kglite.readthedocs.io/en/latest/python/getting-started.html)**: install, first graph, storage choices
- **[AI agents](https://kglite.readthedocs.io/en/latest/python/guides/ai-agents.html)**: hand a graph to an LLM agent: `describe()` prompts, semantic search, MCP

Everything else is linked where it comes up, and the
[Documentation](#documentation) section at the bottom indexes all five tracks.

## What makes it different

Three things the graph does that you would otherwise build yourself.

**`describe()`: progressive-disclosure schema for LLM context windows.** One
call returns a schema sized for a prompt, not for a DBA: the inventory switches
between four detail tiers as the graph's type count grows (full inline detail
under 16 core types, a compact listing, a top-50 listing, then a statistical
summary with a search hint), each type carrying size, complexity, and capability
flags (`ts`, `geo`, `loc`, `vec` for timeseries, geometry, location, and
embeddings). The declared ontology's `is_a` class forest comes with it, and on
graphs small enough to sample, so do join-candidate hints: unconnected types
sharing an identically-named, type-compatible property with overlapping values.
Serve it over MCP with `skills: true` and the tool arrives with methodology
attached, gated by `applies_when` predicates to what the graph actually contains
(a non-code graph never sees code-tool guidance), so *the agent comes pre-loaded
with how to use your graph rather than discovering it through trial-and-error.*
**→ [AI Agents guide](https://kglite.readthedocs.io/en/latest/python/guides/ai-agents.html).**

**As-of queries over history.** Load history tables as dated edges and lifecycle
windows as node properties, and one symmetric idiom answers *"how did the world
look on ⟨date⟩?"* for nodes and relationships alike:

```cypher
MATCH (l:Licence)-[r:HAS_OPERATOR]->(c:Company)
WHERE valid_at(l, '1999-06-30', 'existsFrom', 'existsTo')
  AND valid_at(r, '1999-06-30', 'validFrom',  'validTo')
RETURN c.title
```

Move the date and the answer moves with it: the operator of record in 1999, not
today's. A null or missing bound is open-ended, so an edge with no end date is
still current and an entity carrying no dates at all always matches;
`valid_during(entity, start, end, from, to)` is the interval-overlap sibling.
**→ [Timeseries and temporal guide](https://kglite.readthedocs.io/en/latest/python/guides/timeseries.html).**

**A declared ontology that gates the build.** `define_ontology()` records what
must hold: domain and range over an `is_a` class forest, required edge
properties, property types, cardinality. Each check carries its own enforcement
level, so a document referenced from a blueprint fails the build on an
`error`-level breach, with every violated rule counted and no output graph
written, while `CALL ontology_audit()` scores the same declarations against a
live graph. *Observe → fix → enforce* is configuration, not code review.
**→ [Ontology guide](https://kglite.readthedocs.io/en/latest/python/guides/ontology.html).**

## Serve it to an agent

### One command turns any current `.kgl` into an MCP server

```bash
kglite-mcp-server --graph path/to/graph.kgl
```

Reach for it when you want a graph kept warm across many calls. The server
exposes `cypher_query`, `graph_overview`, schema introspection, and structural
validators over MCP stdio, plus source-file read/search tools when a valid
`source_root` is configured. Drop it into Claude Desktop, Cursor, or another
MCP-capable client and any KGLite graph is queryable. Code-graph construction,
repository cloning, and code-watch workflows belong to **codingest-mcp**, which
embeds this same graph-serving surface.

When you register it, point `command` at the **absolute path** to the binary
(`/abs/path/to/venv/bin/kglite-mcp-server`), not a bare name: a bare command can
silently launch an older PATH-shadowing install. Then confirm it with
`kglite-mcp-server --selftest --graph path/to/graph.kgl`, which drives a real
handshake and prints green/red per capability.

Two ready-made code-intelligence recipes ship in [`examples/`](examples/); run
both under codingest-mcp:
[`open_source_workspace_mcp.yaml`](examples/open_source_workspace_mcp.yaml)
(`repo_management('org/repo')` clones and builds a code graph on demand) and
[`local_code_review_mcp.yaml`](examples/local_code_review_mcp.yaml)
(`set_root_dir(path)` swaps roots, watch-mode auto-rebuilds).

**→ [MCP server operations](https://kglite.readthedocs.io/en/latest/operators/mcp-server.html).**

### Customise with a YAML manifest

Drop `<basename>_mcp.yaml` next to the graph (e.g. `wikidata_mcp.yaml` beside
`wikidata.kgl`) and the server auto-loads it at boot.

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

`skills: true` composes three layers of per-tool methodology (kglite-bundled
defaults, your project's `<basename>.skills/*.md` overrides, and
operator-declared domain packs), so no fork is required for most customisation.
**→ [MCP server guide](https://kglite.readthedocs.io/en/latest/python/guides/mcp-servers.html).**

## Use cases

The same agent-facing surface works whether the graph holds legal precedents, a
Wikidata slice, a SQL warehouse, a RAG corpus, or a parsed codebase.

- 🏛️ **Domain knowledge for agents.** Legal precedents + citations, regulatory
  rules, medical ontologies, manufacturing BOMs, scientific catalogues: anything
  with structure becomes a queryable graph an MCP-capable agent can reason over.
  See the [legal-graph example](https://github.com/kkollsga/kglite/blob/main/examples/legal_graph.py)
  for a Norwegian-Supreme-Court walk-through.
- 📊 **Business data → queryable graph.** Any tabular source (SQL, CSV, Parquet,
  REST API responses, pandas DataFrames) goes straight in via `add_nodes(df,
  ...)` and `add_connections(df, ...)`. Layer a graph on your warehouse and the
  agent reasons over the relationships without you writing a server. **→
  [Data Loading guide](https://kglite.readthedocs.io/en/latest/python/guides/data-loading.html).**
- 🌐 **Public datasets.** Loaders for **SEC EDGAR** filings, **Wikidata** (the
  full `latest-truthy` RDF dump), and **Sodir** petroleum data live in
  [kglite-datasets](https://kglite-datasets.readthedocs.io), each handling the
  *fetch + build + cache* cycle; kglite's mapped and disk storage then query
  graphs that don't fit in RAM, up to the 124M-node / 861M-edge Wikidata graph
  on a 16 GB laptop. The core engine itself needs no network access.
- 📚 **RAG with structure.** Documents, chunks, entities, and the edges between
  them in one graph. Combine `text_score()` vector similarity with Cypher
  traversal (*"find court cases semantically similar to my fact pattern, then
  walk one hop to related precedents"*): hybrid retrieval in one query, no second
  vector DB, scaling with an opt-in HNSW index (`build_vector_index()`).
  **→ [Semantic Search guide](https://kglite.readthedocs.io/en/latest/python/guides/semantic-search.html).**
- 🔎 **Keyword and meaning in one ranking.** An opt-in BM25 lexical index
  (`build_text_index()` + `text_bm25()`) finds the exact term an embedding blurs
  away, and `score_fuse()` blends it with the vector lane in a single Cypher
  query, with no second search service and no merge step in your code.
  **→ [Text Search guide](https://kglite.readthedocs.io/en/latest/python/guides/text-search.html).**
- 📂 **Codebase analysis.** The
  [codingest](https://github.com/kkollsga/codingest) builder parses 14 languages
  into Function / Class / Module / Route nodes with web-framework route
  detection (Flask, FastAPI, Django), from any git revision or several merged
  into one multi-revision graph for structural diffs. kglite serves and queries
  those graphs; **the builder lives in the codingest project.**
- 🤝 **A shared graph as an agent contract.** One `.kgl` as the two-way contract
  between collaborating agents: **ownership layers** (`define_schema(layer=…)` +
  `add_nodes(managed_reload=True)`) separate batch-rebuilt types from live
  agent-mutated ones, **role-scoped writes** (`cypher(..., write_scope=[...])`)
  fence what each agent may touch, a verbatim **instructions slot**
  (`set_instructions`) leads `describe()`, and `CALL ready_set(...)` hands out
  the next actionable work. These are opt-in guards, not an enforced perimeter;
  the exact boundaries, and what each does *not* cover, are in the
  **[MCP servers guide](https://kglite.readthedocs.io/en/latest/python/guides/mcp-servers.html)**.
- 🧠 **Markdown knowledge bases & agent memory.** `kglite.okf.build(dir)` ingests
  an [Open Knowledge Format](https://github.com/GoogleCloudPlatform/knowledge-catalog)
  bundle (or a Claude memory dir, skills folder, or Obsidian vault) into a
  graph: frontmatter → node properties, markdown links → typed edges. Then
  cluster it (`CALL leiden`), find stale notes, surface dangling references: the
  query engine OKF itself doesn't ship. **→
  [OKF guide](https://kglite.readthedocs.io/en/latest/python/guides/okf.html).**

**Why Cypher?** Questions over connected data (*which insiders sold this stock,
who sits on two boards, what cites this case*) are pattern matches. In SQL they
become multi-table joins; in Cypher the pattern *is* the query, and it pays off
most when the data has real structure and your questions traverse it:

```cypher
-- Insider sells, most recent first
MATCH (t:InsiderTransaction {direction: 'sale'})-[:BY_INSIDER]->(p:Person)
MATCH (t)-[:IN_COMPANY]->(c:Company)
RETURN p.title, c.title, t.shares, t.price_per_share
ORDER BY t.transaction_date DESC LIMIT 10
```

**→ [Cypher guide](https://kglite.readthedocs.io/en/latest/python/guides/cypher.html) ·
[Cypher reference](https://kglite.readthedocs.io/en/latest/reference/cypher-reference.html).**

## One engine, seven doorways

Every wrapper drives the same engine over the same `.kgl` files with the same
Cypher. Pick the doorway that matches your stack; a graph built through any of
them is readable through all of them.

| Doorway | Get it | Docs |
|---|---|---|
| **Python**: the primary binding, with DataFrames in/out, fluent API, embeddings | `pip install kglite` | [Getting started](https://kglite.readthedocs.io/en/latest/python/getting-started.html) · [Python track](https://kglite.readthedocs.io/en/latest/python/index.html) |
| **Rust**: embed the engine directly; sessions, CoW transactions | `cargo add kglite` | [Rust track](https://kglite.readthedocs.io/en/latest/rust/index.html) · [docs.rs](https://docs.rs/kglite) |
| **Java**: Panama/FFM binding, natives for 4 platforms bundled | Maven Central `io.github.kkollsga:kglite` | [kglite-java README](https://github.com/kkollsga/kglite/tree/main/kglite-java) |
| **C ABI**: stable `kglite.h` for any other language (Go, JS, .NET, …) | [`crates/kglite-c`](https://github.com/kkollsga/kglite/tree/main/crates/kglite-c) | [C ABI design](https://kglite.readthedocs.io/en/latest/rust/c-abi.html) · [implementing a binding](https://kglite.readthedocs.io/en/latest/rust/implementing-a-binding.html) |
| **CLI**: shell/scripts/JSONL agent loops over a `.kgl` | bundled in the wheel, or `pip install kglite-cli` / `cargo install kglite-cli` | [CLI guide](https://kglite.readthedocs.io/en/latest/operators/cli.html) |
| **Bolt server**: Bolt v5 front-end for Neo4j wire-compatible drivers | `cargo install kglite-bolt-server` | [Bolt server](https://kglite.readthedocs.io/en/latest/operators/bolt-server.html) |
| **MCP server**: serve a graph to AI agents as tools + skills | bundled with the wheel: `kglite-mcp-server --graph <graph>.kgl` | [MCP config guide](https://kglite.readthedocs.io/en/latest/python/guides/mcp-servers.html) · [operators page](https://kglite.readthedocs.io/en/latest/operators/mcp-server.html) |

The engine itself is a pure-Rust crate
([`crates/kglite`](https://github.com/kkollsga/kglite/tree/main/crates/kglite))
packaged for Python via `pip install kglite`; the shell, Bolt-server, and
MCP-server binaries are sibling crates wrapping it. See
**[Use from Rust](#use-from-rust)** to build against it without the wheel. The
wheel also installs the `kglite` command, a `sqlite3`-style REPL: `kglite app.kgl`
opens a Cypher prompt with `.import`, `.dump`, `.schema`, multi-line input, and
tab-completion. The
[operators index](https://kglite.readthedocs.io/en/latest/operators/index.html)
has a decision table for the server-shaped doorways.

## Ecosystem

kglite is the engine. Three companion projects build graphs it serves, each
released and versioned on its own cadence:

- **[codingest](https://codingest.readthedocs.io)** parses codebases into
  code graphs (14 languages, web-framework route detection). Build with it,
  query the `.kgl` here.
- **[kglite-datasets](https://kglite-datasets.readthedocs.io)** carries
  fetch-build-cache loaders for public registries (SEC EDGAR, Wikidata, Sodir).
- **[sonagram](https://sonagram.readthedocs.io)** turns a local music
  library into a kglite knowledge graph via sonara audio analysis (tempo,
  energy, mood, key); AI agents curate playlists over it through a bundled
  skill and CLI (`pip install sonagram`).

## How it compares

|                                            | KGLite                            | [LadybugDB](https://ladybugdb.com/) (formerly Kuzu) | NetworkX           | rustworkx          | Neo4j Embedded         |
|--------------------------------------------|-----------------------------------|-----------------------------------------------------|--------------------|--------------------|------------------------|
| **Install**                                | `pip install kglite`              | `pip install ladybug`                               | `pip install networkx` | `pip install rustworkx` | JVM + Java deps  |
| **Query language**                         | Cypher ([broad coverage](CYPHER.md#feature-coverage)) | Cypher                              | Python API         | Python API         | Cypher (full)          |
| **Storage**                                | in-mem · mmap · disk (tested to 861M edges) | in-mem · disk (columnar)                            | in-mem             | in-mem             | in-mem · disk (JVM)    |
| **Bulk-load from pandas**                  | one-liner                         | via Arrow                                           | manual             | manual             | via driver             |
| **MCP server for LLM agents**              | bundled in the `kglite` wheel     | [separate `mcp-server-ladybug` install](https://github.com/LadybugDB/mcp-server-ladybug) | no | no | no |
| **`describe()` schema for LLM prompts**    | ✅                                 | no                                                  | no                 | no                 | no                     |
| **Declared semantics + data-quality gate** | ✅ (`define_ontology`, audit scorecard, build gate) | typed schema pins edge endpoints | no         | no                 | constraint DDL         |
| **As-of temporal filtering**               | ✅ (`valid_at` on nodes + edges)   | manual                                              | manual             | manual             | manual                 |
| **Embeddable in Rust** (no Python in build) | pure-Rust [`kglite`](https://crates.io/crates/kglite) crate | [`lbug`](https://crates.io/crates/lbug) bindings to the C++ engine | no | ✅ | no |
| **License**                                | MIT                               | MIT                                                 | BSD-3              | Apache-2           | GPLv3                  |

("manual" = expressible in application code or a `WHERE` clause, but no engine
primitive. LadybugDB's rel tables pin each edge's endpoint types at DDL and
Neo4j's constraints cover uniqueness and existence; neither declares domain,
range, or cardinality over a class forest, nor gates a build on the result.)

**Pick KGLite** when you want one embedded package combining Python and
pure-Rust Cypher APIs with a bundled MCP binary, prompt-shaped `describe()`,
agent-contract primitives (role-scoped writes, ownership layers,
`set_instructions`, `CALL ready_set(...)`), a **declared ontology** with
build-time data-quality gates and audit scorecards, and **as-of temporal
filtering** (`valid_at`) over dated edges and lifecycle windows, plus companion
projects that build code and public-registry graphs it serves. **Pick
LadybugDB** when columnar analytical scans and its broader language ecosystem
are the priority; it also provides Rust bindings and a separately installed MCP
server. **Pick NetworkX** when you need its enormous graph-algorithm library and
your data fits in RAM. **Pick rustworkx** when you want a Rust-backed Python
graph API with no query language. **Pick Neo4j Embedded** when you've
standardised on server-mode Cypher and want the in-process driver for tests.

📊 **[Benchmarks →](BENCHMARKS.md)**: wall-to-wall time per topic (load,
filter/aggregate, traversal, pathfinding, algorithms, mutations) against other
embedded graph engines, NetworkX, rustworkx, igraph, and DuckDB on one shared
synthetic graph. Reproduce with `python benchmarks/benchmark.py`; maintainer-only
storage and release-regression probes live under `tests/benchmarks/`.

## Primary store, or derived index?

Two shapes, both supported, with different guarantees. Knowing which one you are
building saves a lot of argument later.

- **Derived index**: the authoritative copy lives elsewhere (a warehouse, an
  API, a repo) and the graph is a rebuildable projection you query. Most kglite
  deployments are this, and it is the cheapest correct answer. **→ [Derived
  index guide](https://kglite.readthedocs.io/en/latest/python/guides/derived-index.html).**
- **Primary store**: the graph *is* the authoritative copy, with crash-safe
  `open()` for the in-memory and `mapped` backends (`disk` checkpoints on `save()`),
  atomic statements, snapshot isolation for readers, and UNIQUE / NOT NULL /
  node-key constraints enforced on every write path including the bulk loaders.
  One process owns the writes; the scope statement lists the limits rather than
  softening them. **→ [Primary store: scope and
  limits](https://kglite.readthedocs.io/en/latest/python/guides/primary-store.html).**

## Licensing and embedded distribution

**kglite is MIT-licensed throughout: every crate in the workspace ships under
MIT.** No separate commercial tier, no development/production distinction, and no
copyleft obligation attached to shipping it: if you can use kglite, you can
distribute it inside your own product.

One honest qualification about the *default* build: the optional `fastembed`
backend is off by default everywhere, so neither the published wheel nor the
default MCP-server binary contains it, and a `--features fastembed` build pulls
one transitive MPL-2.0 crate (`option-ext`, four dependencies down). The reviewed
policy is in [dependency licences](https://kglite.readthedocs.io/en/latest/explanation/dependency-licenses.html).

## Recipes

Short patterns for the most-common shapes. Each is self-contained.

### Hybrid semantic + structural retrieval

Vector similarity (`text_score()`) and Cypher pattern matching in one query,
with a bring-your-own embedder passed to `g.set_embedder(...)`:

```python
graph.cypher("""
    MATCH (c:Chunk)-[:IN_DOC]->(d:Document)
    RETURN c.text, d.title, text_score(c.embedding, $query_vec) AS score
    ORDER BY score DESC LIMIT 5
""", params={"query_vec": query_embedding})
```

**→ [Semantic Search guide](https://kglite.readthedocs.io/en/latest/python/guides/semantic-search.html).**

### Structural validators: surface data-integrity gaps

Fifteen built-in `CALL` procedures find the gaps normal queries don't show:
orphan nodes, missing required edges, two-step cycles, duplicate titles,
parallel edges, cardinality violations, more.

```python
# Wellbores in our sodir graph that lack a production licence
graph.cypher("""
    CALL missing_required_edge({type: 'Wellbore', edge: 'IN_LICENCE'}) YIELD node
    RETURN node.id, node.title
""")
```

`missing_required_edge` and `missing_inbound_edge` validate the `(type, edge)`
direction against the graph's actual schema and refuse to execute when misused.
**→ [Full procedure list](https://kglite.readthedocs.io/en/latest/python/guides/cypher.html#structural-validator-call-procedures).**

### Graph algorithms

Shortest path (BFS or Dijkstra), centrality, community detection, and clustering
are Cypher-callable: `shortestPath((a)-[*]-(b))`, `CALL leiden`, `CALL
pagerank`. **→ [Graph algorithms guide](https://kglite.readthedocs.io/en/latest/python/guides/graph-algorithms.html) ·
[Traversal patterns](https://kglite.readthedocs.io/en/latest/python/guides/traversal-hierarchy.html) ·
[Recipes index](https://kglite.readthedocs.io/en/latest/python/guides/recipes.html).**

## Use from Rust

The same engine is available as a pure-Rust crate. Embed it in a Rust binary
without the Python wheel in your build:

```toml
# Cargo.toml
[dependencies]
kglite = "0.16"
```

```rust
use kglite::api::{io::load_file, session, Value};
use std::collections::HashMap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let graph = load_file("my_graph.kgl")?;     // same .kgl as Python writes
    let params = HashMap::new();
    let opts = session::ExecuteOptions::eager(&params);
    let query = "MATCH (p:Person) RETURN p.name LIMIT 5";
    let outcome = session::execute_read(&graph, query, &opts)?;
    for row in &outcome.result.rows {
        if let Some(Value::String(name)) = row.first() {
            println!("{name}");
        }
    }
    Ok(())
}
```

Zero PyO3 in the dependency tree: `cargo tree -p your-crate | rg pyo3` → empty.
The Bolt server (`crates/kglite-bolt-server`) and the Rust MCP server
(`crates/kglite-mcp-server`) are standalone binaries on the same engine.
**→ [Rust quickstart](https://kglite.readthedocs.io/en/latest/rust/index.html) ·
[embedding guide](https://kglite.readthedocs.io/en/latest/rust/embedding.html) ·
[session abstraction](https://kglite.readthedocs.io/en/latest/rust/session.html) ·
[docs.rs](https://docs.rs/kglite) ·
[Operators guide](https://kglite.readthedocs.io/en/latest/operators/bolt-server.html).**

For **Java**, an official binding is on Maven Central:
`io.github.kkollsga:kglite` (Panama/FFM over the C ABI, natives bundled; see
[`kglite-java/README.md`](https://github.com/kkollsga/kglite/tree/main/kglite-java)).
For **other non-Rust bindings** (Go via cgo, JavaScript via napi, .NET via
P/Invoke),
[`crates/kglite-c`](https://github.com/kkollsga/kglite/tree/main/crates/kglite-c)
exposes the engine through a stable C ABI covering lifecycle, sessions, Cypher,
results, persistence, and embedders, plus a cbindgen-generated `kglite.h`.
**→ [C ABI design](https://kglite.readthedocs.io/en/latest/rust/c-abi.html) ·
[implementing a binding](https://kglite.readthedocs.io/en/latest/rust/implementing-a-binding.html)
(cgo / napi / JNI worked examples).**

## Examples

The [`examples/`](https://github.com/kkollsga/kglite/tree/main/examples)
directory has runnable, self-contained artifacts:

- **[`csv_to_graph.py`](https://github.com/kkollsga/kglite/blob/main/examples/csv_to_graph.py)**:
  `pd.read_csv` → `add_nodes` / `add_connections` on a tiny org chart. The
  fastest way in.
- **[`legal_graph.py`](https://github.com/kkollsga/kglite/blob/main/examples/legal_graph.py)**:
  end-to-end pandas → graph with laws, regulations, court decisions, citation edges.
- **[`incremental_update.py`](https://github.com/kkollsga/kglite/blob/main/examples/incremental_update.py)**:
  merge a second snapshot with `add_nodes(conflict_handling='update')`.
- **[`spatial_graph.py`](https://github.com/kkollsga/kglite/blob/main/examples/spatial_graph.py)**:
  declarative CSV→graph loading via a JSON blueprint; lat/lon coordinates and
  pipeline-path traversal.
- **[`crates/kglite-mcp-server/`](https://github.com/kkollsga/kglite/tree/main/crates/kglite-mcp-server)**:
  a Rust-native single-binary MCP server (rmcp + the [mcp-methods] framework),
  the reference for layering domain-specific tools when a manifest isn't enough.

**→ [Recipes index](https://kglite.readthedocs.io/en/latest/python/guides/recipes.html).**

[mcp-methods]: https://github.com/kkollsga/mcp-methods

## Documentation

Full docs at **[kglite.readthedocs.io](https://kglite.readthedocs.io)**, in five
tracks by audience, each with its own index:

- **[Python](https://kglite.readthedocs.io/en/latest/python/index.html)**
  (`pip install kglite`):
  [getting started](https://kglite.readthedocs.io/en/latest/python/getting-started.html),
  then one guide per subject.
  *Load*: [data loading](https://kglite.readthedocs.io/en/latest/python/guides/data-loading.html) ·
  [inline records](https://kglite.readthedocs.io/en/latest/python/guides/inline-records.html) ·
  [blueprints](https://kglite.readthedocs.io/en/latest/python/guides/blueprints.html) ·
  [import/export](https://kglite.readthedocs.io/en/latest/python/guides/import-export.html) ·
  [structured data](https://kglite.readthedocs.io/en/latest/python/guides/structured-data.html) ·
  [schema migrations](https://kglite.readthedocs.io/en/latest/python/guides/schema-migrations.html).
  *Query*: [Cypher](https://kglite.readthedocs.io/en/latest/python/guides/cypher.html) ·
  [fluent API](https://kglite.readthedocs.io/en/latest/python/guides/querying.html) ·
  [traversal and hierarchy](https://kglite.readthedocs.io/en/latest/python/guides/traversal-hierarchy.html) ·
  [graph algorithms](https://kglite.readthedocs.io/en/latest/python/guides/graph-algorithms.html) ·
  [semantic search](https://kglite.readthedocs.io/en/latest/python/guides/semantic-search.html) ·
  [text search](https://kglite.readthedocs.io/en/latest/python/guides/text-search.html) ·
  [spatial](https://kglite.readthedocs.io/en/latest/python/guides/spatial.html) ·
  [timeseries](https://kglite.readthedocs.io/en/latest/python/guides/timeseries.html) ·
  [ontology](https://kglite.readthedocs.io/en/latest/python/guides/ontology.html) ·
  [recipes](https://kglite.readthedocs.io/en/latest/python/guides/recipes.html).
  *Ship*: [durable apps](https://kglite.readthedocs.io/en/latest/python/guides/durable-apps.html) ·
  [derived index](https://kglite.readthedocs.io/en/latest/python/guides/derived-index.html) ·
  [primary store](https://kglite.readthedocs.io/en/latest/python/guides/primary-store.html) ·
  [OKF ingestion](https://kglite.readthedocs.io/en/latest/python/guides/okf.html) ·
  [AI agents](https://kglite.readthedocs.io/en/latest/python/guides/ai-agents.html) ·
  [MCP servers](https://kglite.readthedocs.io/en/latest/python/guides/mcp-servers.html) ·
  [MCP skills](https://kglite.readthedocs.io/en/latest/python/guides/mcp-skills.html).
- **[Rust](https://kglite.readthedocs.io/en/latest/rust/index.html)**
  (`cargo add kglite`): quickstart, embedding, sessions, C ABI ·
  [docs.rs](https://docs.rs/kglite).
- **[Operators](https://kglite.readthedocs.io/en/latest/operators/index.html)**:
  running the Bolt, MCP, and CLI front-ends.
- **[Reference](https://kglite.readthedocs.io/en/latest/reference/cypher-reference.html)**:
  the supported Cypher subset, the
  [fluent API](https://kglite.readthedocs.io/en/latest/reference/fluent-api.html),
  the [auto-generated Python API](https://kglite.readthedocs.io/en/latest/autoapi/kglite/index.html).
- **[Concepts](https://kglite.readthedocs.io/en/latest/concepts/index.html)**:
  architecture, design decisions, Cypher conformance, concurrency.

Quick reference to the feature set; each row links into the appropriate guide.

| Feature | Description |
|---|---|
| **[Cypher](https://kglite.readthedocs.io/en/latest/python/guides/cypher.html)** | MATCH, CREATE, SET, DELETE, MERGE, UNION/INTERSECT/EXCEPT, aggregations (incl. `median`, `percentile_cont`, `variance`), `reduce()`, ORDER BY, LIMIT, SKIP |
| **Label model** | One immutable primary type per node plus optional secondary labels: `CREATE (n:A:B)`, `SET n:B`, `REMOVE n:B`, and `labels(n)` returns the list (primary first). Details in the [Cypher reference](CYPHER.md) callout. |
| **Text predicates** | `text_edit_distance`, `text_normalize`, `text_jaccard`, `text_ngrams`, `text_contains_any` / `text_starts_with_any` |
| **[Ontology](https://kglite.readthedocs.io/en/latest/python/guides/ontology.html)** | Declared semantic layer: `is_a` class forest + relationship semantics (`define_ontology`), `SHOW ONTOLOGY`, no-arg validators, `CALL ontology_audit()` scorecard, blueprint data-quality gate, opt-in materialization. Annotations, not axioms: SKOS in spirit, never OWL. |
| **Temporal** | `valid_at()` / `valid_during()` as-of and interval-overlap filtering on nodes and relationships (null bounds are open-ended), `date()`/`datetime()`, `date_diff()`, date arithmetic |
| **[Structured data](https://kglite.readthedocs.io/en/latest/python/guides/structured-data.html)** | DataFrame table properties (`set_table_property`/`get_table_property`), declared `list<map{...}>` shapes with indexed error paths, atomic nested `SET o.items[2].qty = 8`, `table.upsert`/`table.delete`, `attach_rows`. |
| **[Spatial](https://kglite.readthedocs.io/en/latest/python/guides/spatial.html)** | Coordinates, WKT geometry, distance + containment, `kg_knn` k-nearest-neighbour. Pragmatic primitives, not a full GIS stack. |
| **[Timeseries](https://kglite.readthedocs.io/en/latest/python/guides/timeseries.html)** | Time-indexed values with `ts_*()` Cypher functions. For graphs whose nodes carry value-over-time series. |
| **[Blueprints](https://kglite.readthedocs.io/en/latest/python/guides/blueprints.html)** | Declarative CSV-to-graph loading via JSON config |
| **[Import/Export](https://kglite.readthedocs.io/en/latest/python/guides/import-export.html)** | Save/load snapshots (`.kgl`), GraphML, CSV export |

## Requirements

CPython 3.10+ | macOS (arm64/x86_64), Linux (glibc/musl; x86_64 and best-effort
aarch64), Windows (x86_64). The base wheel has no Python runtime dependencies;
integrations install their named extras. See the
[artifact support policy](https://kglite.readthedocs.io/en/latest/python/platform-support.html)
for tested/build-only tiers, libc floors, PyPy status, and source-build fallback.

## Stability

KGLite is beta software and remains pre-1.0. Patch releases preserve public
source APIs; a 0.x minor release may make an intentional breaking source-API
change when it is documented with a migration path. Saved graph files have a
separate format lifecycle: a release either reads an older format or refuses it
with an explicit rebuild/migration error; see
[CHANGELOG.md](https://github.com/kkollsga/kglite/blob/main/CHANGELOG.md).

Every change runs a cross-storage parity matrix and a differential Cypher
corpus: the same query must return the same rows on the in-memory, mmap, and
disk backends, and again with every optimiser pass disabled.
**→ [Cypher conformance](https://kglite.readthedocs.io/en/latest/concepts/cypher-conformance.html).**

## License

MIT. See [LICENSE](https://github.com/kkollsga/kglite/blob/main/LICENSE), and
[Licensing and embedded distribution](#licensing-and-embedded-distribution) for
what that means when kglite ships inside a product you distribute.
