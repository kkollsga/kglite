# Using with AI Agents

KGLite is a self-contained knowledge layer for AI agents. In-process use needs no external database or server. The graph is a Python object with a Cypher interface that an agent can query directly. Network-backed data sources and optional model downloads remain explicit choices.

## The Idea

1. **Load or build a graph** from your data (DataFrames, CSVs, APIs)
2. **Give the agent `describe()`** — a progressive-disclosure XML schema that scales from tiny to massive graphs
3. **The agent writes Cypher queries** using `graph.cypher()` — no other API to learn
4. **Semantic search works natively** — `text_score()` in Cypher, backed by any embedding model you wrap

You need no vector database, no graph database and no infrastructure. The graph lives in memory and persists to a single `.kgl` file.

## Quick Setup

```python
xml = graph.describe()  # inventory overview — types, connections, Cypher extensions
prompt = f"You have a knowledge graph:\n{xml}\nAnswer the user's question using graph.cypher()."
```

## MCP Server

A thin server exposes the graph to any MCP-compatible agent (Claude, etc.). The [MCP Servers guide](mcp-servers.md) gives a complete walkthrough: server setup, tool patterns, FORMAT CSV export, security, and a copy-paste template.

For large results, start with a bounded preview. Then retrieve the exact retained row or nested value the preview identifies. [Bounded agent responses](../../operators/agent-responses.md) documents the MCP action-composition flow and CLI follow-up commands.

Expansion does not broaden a Cypher `LIMIT`. It only reads the completed retained result.

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

The model wrapper works with any provider: OpenAI, Cohere, local sentence-transformers, Ollama. See [Semantic Search](semantic-search.md) for the full API.

## Tips for Agent Prompts

1. **Start with `describe()`**. It gives the agent an inventory of types with capability flags, a connection map, and non-standard Cypher extensions.
2. **Drill into types with `describe(types=['Project'])`**. It shows properties, connections, timeseries/spatial config, supporting children, and sample nodes.
3. **Use `properties(type)`** for deeper column discovery. It shows types, nullability, unique counts, and sample values.
4. **Use `sample(type, n=3)`** before writing queries. The agent sees real data shapes.
5. **Prefer Cypher** over the fluent API in agent contexts. It is closer to natural language and easier for LLMs to generate.
6. **Use parameters** (`params={'x': val}`) to prevent injection when passing user input to queries.
7. **ResultView is lazy.** Agents can call `len(result)` to check the row count without converting all rows.

## Structural Validators

Fifteen native Cypher procedures surface data-integrity gaps. The agent does not have to write the underlying `WHERE NOT EXISTS` patterns.

The table lists six common procedures. `describe()` lists the full set, in the `<rules hint="..."/>` extension and in `describe(cypher=True)`.

| Procedure | What it finds |
|---|---|
| `orphan_node({type})` | nodes with zero edges in any direction |
| `self_loop({type, edge})` | self-loops via the given edge |
| `cycle_2step({type, edge})` | reciprocal pairs `a-[:edge]->b-[:edge]->a` |
| `missing_required_edge({type, edge})` | direction-validated outbound check |
| `missing_inbound_edge({type, edge})` | direction-validated inbound check |
| `duplicate_title({type})` | nodes whose title is shared with another node of the same type |

Each procedure binds `node` (or `node_a, node_b` for `cycle_2step`). The agent can compose with WHERE / ORDER BY / aggregation in a single Cypher pass. This includes the cross-reference workflow, where flagged IDs are checked against another query's results:

```python
graph.cypher("""
    MATCH (l:Contract {title: '057'})<-[:IN_CONTRACT]-(w:Site)
    WITH collect(w.id) AS c057
    CALL missing_required_edge({type: 'Site', edge: 'BUILT_BY'}) YIELD node
    WHERE node.id IN c057
    RETURN count(*) AS c057_missing_built_by
""")
```

`missing_required_edge` and `missing_inbound_edge` validate the `(type, edge)` direction against the graph's actual schema. If the agent picks the wrong direction, they raise `DirectionMismatch` with a fix-suggesting message. An example is asking for inbound `IN_CONTRACT` on a `Site` when the edge flows outward.

See [Cypher → Structural-validator CALL procedures](cypher.md#structural-validator-call-procedures) for more examples. Use `describe(cypher=['orphan_node'])` for per-procedure details.

## What `describe()` Returns

`describe()` has three modes.

- **Inventory mode** (`describe()`): node types as compact descriptors `TypeName[size,complexity,flags]` sorted by count, a connection map, and Cypher extensions.
  - Core/supporting type tiers hide child types behind `+N` suffixes.
  - For small graphs (≤15 types), full detail is inlined automatically.
  - The `<extensions>` block carries `<algorithms>` and `<rules>` hint lines. They point the agent at the available `CALL` procedures (graph algorithms + structural validators).
- **Focused mode** (`describe(types=['Project'])`): detailed properties with types, connection topology, timeseries/spatial config, supporting children, and sample nodes.
- **Cypher reference** (`describe(cypher=True)`): the full language reference, including supported clauses, operators, built-in functions, predicates, and the structural-validator catalogue. Drill into a single procedure with `describe(cypher=['orphan_node'])`.

### Reading the XML

These attributes matter when you paste `describe()` into a prompt.

- **`kglite_version="…"`** on the root `<graph>` element (0.9.37+).
  - It is the KGLite version that produced the XML, sourced at compile time from the running binary.
  - The local `pip install kglite` can be on a different version from the MCP-server-side binary. The schema you read comes from the server's binary, not your local one.
  - Surface this if you see a schema/query mismatch.
- **`id_alias="…"` / `title_alias="…"`** on a `<type>` element.
  - They are set when `add_nodes(...)` was called with a `unique_id_field` other than `"id"` (e.g. `"person_id"`) or a `node_title_field` other than `"title"` (e.g. `"proposal_name"`).
  - The alias tells the agent that `n.person_id` and `n.id` resolve to the same field. The agent can use whichever name appears in the source data.
  - Both forms work in `MATCH` / `WHERE`. Result rows always come back keyed under the canonical `id` / `title`.
- **`revs="…"`** on the `<active_graph …/>` header (MCP code-graph sessions).
  - It is present only when the server built a **multi-revision** code graph: `repo_management(name, revs=N|[list])` (github) or `set_root_dir(path, revs=…)` (local). It names the loaded rev-set, e.g. `revs="v1.0,v2.0,HEAD"`.
  - It signals that one graph holds every listed revision. An **unscoped** query counts the union across revs and **over-counts**. `MATCH (n:Function) RETURN count(n)` is not "functions at HEAD" but "functions across all revs".
  - Scope a query to a single rev with list membership on the per-node `revs` property (`MATCH (n:Function) WHERE 'v2.0' IN n.revs RETURN n`).
  - Use `CALL rev_diff({from, to})` for added / removed / changed deltas between two revs.
  - When `revs` is absent the graph is single-revision, and plain unscoped queries are exact.
  - `describe()` also lists the loaded revs and teaches the same `WHERE '<rev>' IN n.revs` idiom.
- **`sample_truncate`** (call-site knob).
  - Sample values, sample node titles, and sample edge attributes are truncated at 40 chars by default to keep prompts compact.
  - Pass `describe(sample_truncate=None)` to emit them in full when you have the context budget, or e.g. `sample_truncate=120` for a middle ground.
  - The knob only affects rendering. Stored data is always full-precision and accessible via Cypher.
