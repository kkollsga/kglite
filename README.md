# KGLite

[![PyPI version](https://img.shields.io/pypi/v/kglite)](https://pypi.org/project/kglite/)
[![Python versions](https://img.shields.io/pypi/pyversions/kglite)](https://pypi.org/project/kglite/)
[![License: MIT](https://img.shields.io/pypi/l/kglite)](https://github.com/kkollsga/kglite/blob/main/LICENSE)

A knowledge graph that runs inside your Python process. Load data, query with Cypher, do semantic search — no server, no setup, no infrastructure.

> **Two APIs:** Use **Cypher** for querying, mutations, and semantic search. Use the **fluent API** (`add_nodes` / `add_connections`) for bulk-loading DataFrames. Most agent and application code only needs `cypher()`.

| | |
|---|---|
| Embedded, in-process | No server, no network; `import` and go |
| In-memory | Persistence via `save()`/`load()` snapshots |
| Cypher subset | Querying + mutations + `text_score()` for semantic search |
| Single-label nodes | Each node has exactly one type |
| Fluent bulk loading | Import DataFrames with `add_nodes()` / `add_connections()` |

**Requirements:** Python 3.10+ (CPython) | macOS (ARM/Intel), Linux (x86_64/aarch64), Windows (x86_64) | `pandas >= 1.5`

```bash
pip install kglite
```

---

## Quick Start — [Cypher guide](https://kglite.readthedocs.io/en/latest/guides/cypher.html)

```python
import kglite

graph = kglite.KnowledgeGraph()

# Create nodes and relationships
graph.cypher("CREATE (:Person {name: 'Alice', age: 28, city: 'Oslo'})")
graph.cypher("CREATE (:Person {name: 'Bob', age: 35, city: 'Bergen'})")
graph.cypher("""
    MATCH (a:Person {name: 'Alice'}), (b:Person {name: 'Bob'})
    CREATE (a)-[:KNOWS]->(b)
""")

# Query — returns a ResultView (lazy; data stays in Rust until accessed)
result = graph.cypher("""
    MATCH (p:Person) WHERE p.age > 30
    RETURN p.name AS name, p.city AS city
    ORDER BY p.age DESC
""")
for row in result:
    print(row['name'], row['city'])

# Or get a pandas DataFrame
df = graph.cypher("MATCH (p:Person) RETURN p.name, p.age ORDER BY p.age", to_df=True)

# Persist to disk and reload
graph.save("my_graph.kgl")
loaded = kglite.load("my_graph.kgl")
```

### Bulk Loading from DataFrames — [docs](https://kglite.readthedocs.io/en/latest/guides/data-loading.html)

```python
import pandas as pd

users_df = pd.DataFrame({
    'user_id': [1001, 1002, 1003],
    'name': ['Alice', 'Bob', 'Charlie'],
    'age': [28, 35, 42]
})
graph.add_nodes(data=users_df, node_type='User', unique_id_field='user_id', node_title_field='name')
```

### Blueprint Loading (CSV → Graph) — [docs](https://kglite.readthedocs.io/en/latest/guides/blueprints.html)

```python
import kglite

# Define a blueprint.json mapping CSVs to nodes and connections:
# {
#   "settings": {"root": "./data"},
#   "nodes": {
#     "Person": {
#       "csv": "persons.csv", "pk": "person_id", "title": "name",
#       "properties": {"age": "int", "city": "string"},
#       "connections": {
#         "junction_edges": {
#           "KNOWS": {"csv": "knows.csv", "source_fk": "person_id",
#                     "target": "Person", "target_fk": "friend_id"}
#         }
#       }
#     }
#   }
# }

graph = kglite.from_blueprint("blueprint.json")
```

### Code Review (Parse a Codebase) — [docs](https://kglite.readthedocs.io/en/latest/guides/code-tree.html)

```python
from kglite.code_tree import build

graph = build(".")  # auto-detects pyproject.toml / Cargo.toml

# Find the most-called functions
graph.cypher("""
    MATCH (caller:Function)-[:CALLS]->(f:Function)
    RETURN f.name AS function, count(caller) AS callers
    ORDER BY callers DESC LIMIT 10
""")

# Explore code
graph.find("execute")              # search by name
graph.source("execute_query")      # read source code
graph.context("KnowledgeGraph")    # see struct with methods
```

### AI Agent Integration — [docs](https://kglite.readthedocs.io/en/latest/guides/ai-agents.html)

```python
xml = graph.describe()  # progressive-disclosure schema for agents
prompt = f"You have a knowledge graph:\n{xml}\nAnswer using graph.cypher()."
```

---

## Documentation

Full documentation is available at **[kglite.readthedocs.io](https://kglite.readthedocs.io)**.

| Topic | Description |
|---|---|
| [Getting Started](https://kglite.readthedocs.io/en/latest/getting-started.html) | Installation, quick start, DataFrame loading |
| [Core Concepts](https://kglite.readthedocs.io/en/latest/core-concepts.html) | Nodes, relationships, selections, return types |
| [Cypher Guide](https://kglite.readthedocs.io/en/latest/guides/cypher.html) | Queries, mutations, transactions, parameters |
| [Cypher Reference](https://kglite.readthedocs.io/en/latest/reference/cypher-reference.html) | Full reference for every clause and function |
| [Data Loading](https://kglite.readthedocs.io/en/latest/guides/data-loading.html) | Fluent API, conflict handling, batch updates |
| [Blueprints](https://kglite.readthedocs.io/en/latest/guides/blueprints.html) | Declarative CSV-to-graph loading via JSON |
| [Querying](https://kglite.readthedocs.io/en/latest/guides/querying.html) | Filtering, traversal, schema introspection |
| [Fluent API Reference](https://kglite.readthedocs.io/en/latest/reference/fluent-api.html) | Full reference for every fluent method |
| [Semantic Search](https://kglite.readthedocs.io/en/latest/guides/semantic-search.html) | Embeddings, vector search, `text_score()` |
| [AI Agents](https://kglite.readthedocs.io/en/latest/guides/ai-agents.html) | MCP server, `describe()`, agent prompts |
| [Spatial](https://kglite.readthedocs.io/en/latest/guides/spatial.html) | Coordinates, geometry, distance, containment |
| [Timeseries](https://kglite.readthedocs.io/en/latest/guides/timeseries.html) | Time-indexed data, `ts_*()` Cypher functions |
| [Graph Algorithms](https://kglite.readthedocs.io/en/latest/guides/graph-algorithms.html) | Shortest path, centrality, community detection, clustering |
| [Import & Export](https://kglite.readthedocs.io/en/latest/guides/import-export.html) | Save/load, GraphML, CSV, indexes, performance |
| [Code Tree](https://kglite.readthedocs.io/en/latest/guides/code-tree.html) | Parse codebases into knowledge graphs |
| [API Reference](https://kglite.readthedocs.io/en/latest/autoapi/kglite/index.html) | Auto-generated from type stubs |

---

## License

MIT — see [LICENSE](https://github.com/kkollsga/kglite/blob/main/LICENSE) for details.
