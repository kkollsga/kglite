# Using with AI Agents

KGLite is designed to work as a self-contained knowledge layer for AI agents. No external database, no server process, no network — just a Python object with a Cypher interface that an agent can query directly.

## The Idea

1. **Load or build a graph** from your data (DataFrames, CSVs, APIs)
2. **Give the agent `describe()`** — a progressive-disclosure XML schema that scales from tiny to massive graphs
3. **The agent writes Cypher queries** using `graph.cypher()` — no other API to learn
4. **Semantic search works natively** — `text_score()` in Cypher, backed by any embedding model you wrap

No vector database, no graph database, no infrastructure. The graph lives in memory and persists to a single `.kgl` file.

## Quick Setup

```python
xml = graph.describe()  # inventory overview — types, connections, Cypher extensions
prompt = f"You have a knowledge graph:\n{xml}\nAnswer the user's question using graph.cypher()."
```

## MCP Server

Expose the graph to any MCP-compatible agent (Claude, etc.) with a thin server. See the [MCP Servers guide](mcp-servers.md) for a complete walkthrough — server setup, tool patterns, FORMAT CSV export, security, and a copy-paste template.

## Adding Semantic Search (5-Minute Setup)

Semantic search lets agents find nodes by meaning, not just exact property matches:

```python
# 1. Wrap any embedding model (local or remote)
class Embedder:
    dimension = 384
    def embed(self, texts: list[str]) -> list[list[float]]:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        return model.encode(texts).tolist()

# 2. Register it on the graph
graph.set_embedder(Embedder())

# 3. Embed a text column (one-time, incremental on re-run)
graph.embed_texts("Article", "summary")

# 4. Now agents can search by meaning in Cypher — no extra API
graph.cypher("""
    MATCH (a:Article)
    WHERE text_score(a, 'summary', 'climate policy') > 0.5
    RETURN a.title, text_score(a, 'summary', 'climate policy') AS score
    ORDER BY score DESC LIMIT 10
""")
```

The model wrapper works with any provider — OpenAI, Cohere, local sentence-transformers, Ollama. See [Semantic Search](semantic-search.md) for the full API.

## Tips for Agent Prompts

1. **Start with `describe()`** — gives the agent an inventory of types with capability flags, connection map, and non-standard Cypher extensions
2. **Drill into types with `describe(types=['Field'])`** — shows properties, connections, timeseries/spatial config, supporting children, and sample nodes
3. **Use `properties(type)`** for deeper column discovery — shows types, nullability, unique counts, and sample values
4. **Use `sample(type, n=3)`** before writing queries — lets the agent see real data shapes
5. **Prefer Cypher** over the fluent API in agent contexts — closer to natural language, easier for LLMs to generate
6. **Use parameters** (`params={'x': val}`) to prevent injection when passing user input to queries
7. **ResultView is lazy** — agents can call `len(result)` to check row count without converting all rows

## Structural Validators

Six native Cypher procedures surface data-integrity gaps without the agent having to write the underlying `WHERE NOT EXISTS` patterns. They appear in `describe()` (in the `<rules hint="..."/>` extension and in `describe(cypher=True)`) so the agent can discover them inline with the schema.

| Procedure | What it finds |
|---|---|
| `orphan_node({type})` | nodes with zero edges in any direction |
| `self_loop({type, edge})` | self-loops via the given edge |
| `cycle_2step({type, edge})` | reciprocal pairs `a-[:edge]->b-[:edge]->a` |
| `missing_required_edge({type, edge})` | direction-validated outbound check |
| `missing_inbound_edge({type, edge})` | direction-validated inbound check |
| `duplicate_title({type})` | nodes whose title is shared with another node of the same type |

Each binds `node` (or `node_a, node_b` for `cycle_2step`), so the agent can compose with WHERE / ORDER BY / aggregation in a single Cypher pass — including the cross-reference workflow where flagged IDs are checked against another query's results:

```python
graph.cypher("""
    MATCH (l:Licence {title: '057'})<-[:IN_LICENCE]-(w:Wellbore)
    WITH collect(w.id) AS pl057
    CALL missing_required_edge({type: 'Wellbore', edge: 'DRILLED_BY'}) YIELD node
    WHERE node.id IN pl057
    RETURN count(*) AS pl057_missing_drilled_by
""")
```

`missing_required_edge` and `missing_inbound_edge` validate the `(type, edge)` direction against the graph's actual schema and raise `DirectionMismatch` with a fix-suggesting message when the agent picks the wrong one (e.g. asking for inbound `IN_LICENCE` on a `Wellbore` when the edge flows outward). See [Cypher → Structural-validator CALL procedures](cypher.md#structural-validator-call-procedures) for the full surface and per-procedure docs via `describe(cypher=['orphan_node'])`.

## What `describe()` Returns

- **Inventory mode** (`describe()`): node types as compact descriptors `TypeName[size,complexity,flags]` sorted by count, connection map, Cypher extensions. Core/supporting type tiers hide child types behind `+N` suffixes. For small graphs (≤15 types), full detail is inlined automatically. The `<extensions>` block carries `<algorithms>` and `<rules>` hint lines pointing the agent at the available `CALL` procedures (graph algorithms + structural validators).
- **Focused mode** (`describe(types=['Field'])`): detailed properties with types, connection topology, timeseries/spatial config, supporting children, and sample nodes.
- **Cypher reference** (`describe(cypher=True)`): full language reference including all supported clauses, operators, built-in functions, predicates, and procedures (including the six structural validators). Drill into a single procedure with `describe(cypher=['orphan_node'])`.

### Reading the XML

A few attributes worth knowing when you paste `describe()` into a prompt:

- **`kglite_version="…"`** on the root `<graph>` element (0.9.37+).
  The KGLite version that produced the XML, sourced at compile time
  from the running binary. Useful when the local `pip install kglite`
  is on a different version from the MCP-server-side binary — the
  schema you're reading is from the server's binary, not your local
  one. Surface this if you see a schema/query mismatch.
- **`id_alias="…"` / `title_alias="…"`** on a `<type>` element. Set
  when `add_nodes(...)` was called with a `unique_id_field` other
  than `"id"` (e.g. `"npdid"`) or a `node_title_field` other than
  `"title"` (e.g. `"prospect_name"`). The alias tells the agent
  that `n.npdid` and `n.id` resolve to the same field, so it can
  use whichever name appears in the source data without wondering
  which one Cypher will accept. Both forms work in `MATCH` /
  `WHERE`. Result rows always come back keyed under the canonical
  `id` / `title`.
- **`sample_truncate`** (call-site knob). Sample values, sample
  node titles, and sample edge attributes get truncated at 40
  chars by default to keep prompts compact. Pass
  `describe(sample_truncate=None)` to emit them in full when you
  have the context budget, or e.g. `sample_truncate=120` for a
  middle ground. The knob only affects rendering; stored data is
  always full-precision and accessible via Cypher.
