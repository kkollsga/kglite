# Example: query rewriting with `extensions.cypher_preprocessor`

Rewrite the agent's Cypher before every `cypher_query` and
`tools[].cypher` invocation — without a bespoke MCP server. Two shapes,
both gated by `trust.allow_query_preprocessor: true`:

- **`rules:`** — ordered regex substitutions (replacement supports `$1`
  backrefs). Zero code; covers most id/token normalisation.
- **`command:`** — a subprocess: the query goes in on stdin, the
  rewritten query comes back on stdout. Arbitrary logic, any language.
  Runs with the manifest's directory as its cwd.

The motivating use case is **Wikidata Q-number rewriting**: the graph
stores entity ids as integers (`42`), but LLMs naturally type the
Wikidata-native `'Q42'`. (kglite deliberately stopped auto-coercing
prefixed ids in 0.10.10 — it was a false-positive risk — so this
per-graph coercion belongs in a per-graph preprocessor.)

> **Replaces a custom FastMCP server.** Before 0.10.26 the only way to
> express pre-query rewriting was a bespoke FastMCP server built on the
> `mcp` + `mcp_methods` wheels. With `cypher_preprocessor` it's a few
> lines of manifest (or a tiny stdin→stdout script) on the standard
> `kglite-mcp-server` binary.

## Option A — declarative `rules:` (the common case)

```yaml
# wikidata_mcp.yaml — co-located with wikidata.kgl
name: Wikidata
trust:
  allow_query_preprocessor: true     # gate; mirrors trust.allow_embedder
extensions:
  cypher_preprocessor:
    rules:
      - pattern: "'Q(\\d+)'"          # 'Q42' → 42  (strip quotes + prefix)
        replace: "$1"
overview_prefix: |
  ## Identifiers
  - Q-numbers stored as integers in `id`. `{id: 42}` and `{nid: 'Q42'}`
    both work — the latter is rewritten before execution.
```

Rules apply in order; each is a Rust `regex` pattern with a `$1`-style
replacement. This single rule already handles `{nid: 'Q42'}`,
`WHERE n.id = 'Q42'`, and `nid IN ['Q42','Q64']` — every quoted
`'Q<digits>'` becomes the bare integer.

## Option B — `command:` (arbitrary logic)

When regex isn't enough, pipe the query through a script. It reads the
query on stdin and prints the rewritten query on stdout:

```yaml
extensions:
  cypher_preprocessor:
    command: ["./wikidata_rewrite.py"]   # relative to the manifest dir
```

```python
#!/usr/bin/env python3
# wikidata_rewrite.py  (chmod +x)
import re
import sys

query = sys.stdin.read()
query = re.sub(r"(['\"])Q(\d+)\1", r"\2", query)   # 'Q42' → 42
# ...any further transformations...
sys.stdout.write(query)
```

A non-zero exit (or anything on stderr with a failure status) surfaces
to the agent as a `Cypher error:` — the original query is **not** run.
The command's working directory is the manifest's parent, so relative
paths and data files resolve predictably.

## What the agent experiences

```cypher
-- The agent writes the natural form:
MATCH (n {nid: 'Q42'}) RETURN n.label
-- The preprocessor rewrites it to:
MATCH (n {nid: 42}) RETURN n.label
-- ...which is what kglite's Cypher engine executes.
```

Both `cypher_query` and any manifest `tools[].cypher` template go
through the preprocessor; non-Cypher tools (`graph_overview`,
`read_source`, `grep`, …) are untouched.

## Trust gate

`extensions.cypher_preprocessor` requires `trust.allow_query_preprocessor:
true`. Without it the server **fails to boot**:

```
ERROR: extensions.cypher_preprocessor requires trust.allow_query_preprocessor: true
```

Mirrors `extensions.embedder` ↔ `trust.allow_embedder` — a preprocessor
rewrites every query (and `command:` runs a subprocess), so operators
opt in explicitly and can audit all dynamic hooks under `trust:`.

## Other shapes that fit

- **Date normalisation** — `31.12.2020` → `2020-12-31` (a `rules:` regex).
- **Multi-tenant scoping** — inject a `WHERE n.tenant_id = …` clause
  (a `command:` script, since it's structural).
- **Query shortcuts** — expand a `RECENT(7d)` macro into a datetime
  predicate.

Rule of thumb: pure string/regex rewriting → `rules:`; anything needing
real logic → `command:`; a fixed parameterised lookup → not a
preprocessor at all, use `tools[].cypher`.
