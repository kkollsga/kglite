# Getting Started

## Installation

```bash
pip install kglite
```

Optional extras:

```bash
pip install "kglite[mcp]"        # ship the graph as an MCP server for Claude / Cursor / agents
pip install "kglite[code-tree]"  # parse Python / Rust / TS / Go / Java / C++ codebases into a graph
pip install "kglite[neo4j]"      # round-trip with Neo4j
```

## Quick Start — DataFrames in, queries out

The day-1 workflow is *not* writing CREATE statements one node at a
time — that path exists, but it isn't how anyone loads real data.
Shape the data as a flat DataFrame (one row per node, one row per
edge) and bulk-load it:

```python
import pandas as pd
import kglite

graph = kglite.KnowledgeGraph()

# Nodes — one row per node, columns become properties.
people_df = pd.DataFrame({
    "user_id": [1001, 1002, 1003],
    "name":    ["Alice", "Bob", "Charlie"],
    "age":     [28, 35, 42],
    "city":    ["Oslo", "Bergen", "Oslo"],
})
graph.add_nodes(
    data=people_df,
    node_type="Person",
    unique_id_field="user_id",
    node_title_field="name",
)

# Edges — one row per edge, columns name the endpoints.
edges_df = pd.DataFrame({"src": [1001, 1002], "tgt": [1002, 1003]})
graph.add_connections(
    data=edges_df,
    connection_type="KNOWS",
    source_type="Person",
    source_id_field="src",
    target_type="Person",
    target_id_field="tgt",
)

# Query — Cypher result is a ResultView (lazy; data stays in Rust).
result = graph.cypher("""
    MATCH (p:Person) WHERE p.age > 30
    RETURN p.name AS name, p.city AS city
    ORDER BY p.age DESC
""")
for row in result:
    print(row["name"], row["city"])

# Or pull a pandas DataFrame back out.
df = graph.cypher(
    "MATCH (p:Person) RETURN p.name, p.age ORDER BY p.age",
    to_df=True,
)

# Persist + reload.
graph.save("my_graph.kgl")
loaded = kglite.load("my_graph.kgl")
```

That's the loop: shape DataFrames → `add_nodes` / `add_connections` →
Cypher → save. {doc}`guides/data-loading` covers conflict handling
(`update` / `replace` / `skip` / `preserve` / `sum`), incremental
loads, hierarchies, and N-Triples / CSV ingest.

### Ad-hoc inserts

For interactive tinkering or single-node tweaks, plain Cypher works:

```python
graph.cypher("CREATE (:Person {name: 'Dana', age: 24, city: 'Trondheim'})")
graph.cypher("""
    MATCH (a:Person {name: 'Dana'}), (b:Person {name: 'Alice'})
    CREATE (a)-[:KNOWS]->(b)
""")
```

For thousands of rows, `add_nodes` / `add_connections` is 50–100×
faster — every Cypher CREATE goes through the parser; the bulk path
goes straight to the columnar store.

## Serve it to an AI agent

With the `[mcp]` extra installed, expose any `.kgl` file to Claude /
Cursor / any MCP-capable agent in one command:

```bash
pip install "kglite[mcp]"
kglite-mcp-server --graph my_graph.kgl
```

Two tools out of the box (`graph_overview` for schema discovery,
`cypher_query` for execution). Add a sibling `<basename>_mcp.yaml`
file with `source_root: ./data` and you get **five** tools — three
sandboxed file-access tools (`read_source` / `grep` / `list_source`)
register automatically. See {doc}`guides/mcp-servers`.

## Loading a Public Dataset

KGLite ships one-call wrappers for well-known public sources. Each
handles download, caching, cooldown, and graph build:

```python
from kglite.datasets import wikidata, sodir

# Wikidata: parallel-decoded multistream bz2 → disk-cached graph
g = wikidata.open("/data/wd")                              # full graph
g = wikidata.open("/data/wd", entity_limit_millions=100)   # 100M slice

# Sodir: petroleum-domain graph, in-memory by default
g = sodir.open("/data/sodir")
```

Re-running just loads the cached graph — sub-second on Wikidata
slices, ~2 s for the full Sodir graph. See {doc}`guides/datasets`
for the full API including cooldown semantics, complement
blueprints, and parallel-fetch tuning.

## Next Steps

- {doc}`guides/index` — guide index ranked by what you're trying to do.
- {doc}`guides/data-loading` — full DataFrame walkthrough, conflict
  handling, hierarchies.
- {doc}`guides/cypher` — full Cypher coverage, parameters, count
  subqueries, semantic search.
- {doc}`guides/mcp-servers` — bundled CLI, manifest customisation,
  source-file tools.
- {doc}`core-concepts` — storage modes, return types, the
  fluent / Cypher split.
