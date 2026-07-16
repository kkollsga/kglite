# CLI

The main `kglite` wheel includes the `kglite` command for working with `.kgl`
graph files without starting a server. It has two modes:

- one-shot commands for scripts and agents
- an interactive Cypher shell for humans

Install the Python API and CLI together:

```bash
pip install kglite
```

For a standalone CLI-only installation:

```bash
pip install kglite-cli
# or build the libpython-free binary from crates.io
cargo install kglite-cli
```

Both routes expose the same Rust CLI implementation. Do not install both into
one environment because they provide the same `kglite` command name.

## Code-Review Skill

Install the bundled Agent Skill for both Codex and Claude Code:

```bash
kglite skill install
```

Use `--host codex` or `--host claude` to choose one, `--project` for repository
scope, and `--dry-run` to inspect destinations. Re-running replaces the managed
artifact idempotently; `kglite skill uninstall` removes only directories marked
as CLI-managed.

The skill drives the CLI directly. Build a working-tree graph, or a graph that
spans a committed base and head revision:

```bash
# Code-graph builds moved to the codingest project (its CLI builds the .kgl):
# see the codingest README. Example shape:
#   codingest build . --output .kglite/code-review.kgl
#   codingest status --output .kglite/code-review.kgl
```

`build` writes a metadata sidecar with the source/revision fingerprint.
`status` reports `fresh`, `stale`, or `missing` without loading the graph. The
review workflow still calls `describe` before Cypher and verifies structural
results against exact source lines.

## One-Shot Commands

Run a read-only Cypher query and exit:

```bash
kglite query app.kgl "MATCH (n:Person) RETURN n.name AS name" --format json
```

Run a write statement and save the graph:

```bash
kglite write app.kgl "CREATE (:Task {id:'t1', status:'todo'})" \
  --save \
  --write-scope Task \
  --git-sha abc123 \
  --modified-by agent
```

`--write-scope` restricts writes to the listed node and relationship
types. `--git-sha` and `--modified-by` stamp provenance on
`auto_timestamp` types.

Inspect a dependency frontier:

```bash
kglite ready-set app.kgl \
  --done 'n.status = "done"' \
  --node-type Task \
  --format csv
```

Print the agent-oriented graph description:

```bash
kglite describe app.kgl
kglite describe app.kgl --types Task
kglite describe app.kgl --cypher
kglite describe app.kgl --connections
```

`describe` returns the same XML schema document exposed by the Python API
and MCP server, including focused views for labels, Cypher support, and
connection types.

## Agent Sessions

Use `session` when an agent needs multiple operations against the same
graph. The process keeps one graph loaded in memory and accepts JSONL
requests on stdin:

```bash
kglite session app.kgl --format json
```

Example request stream:

```json
{"op":"describe","types":["Task"]}
{"id":"w1","op":"write","query":"CREATE (:Task {id:'t1', status:'todo'})"}
{"id":"q1","op":"query","query":"MATCH (t:Task) RETURN count(t) AS n","format":"json"}
{"op":"save"}
{"op":"exit"}
```

Responses echo `id` when provided. In JSON mode, `query` and `write`
return typed `rows`; table and CSV modes return rendered `output`.

For focused descriptions, agents can use compact or explicit object
forms:

```json
{"op":"describe","connections":true}
{"op":"describe","connections":["KNOWS"]}
{"op":"describe","connections":{"detail":"overview"}}
{"op":"describe","connections":{"types":["KNOWS"]}}
```

The same object style works for `types`, `cypher`, and `fluent` detail
selectors where applicable.

## Interactive Shell

Open the shell with a graph path:

```bash
kglite app.kgl
```

Run with no path for a scratch in-memory graph:

```bash
kglite
```

Cypher statements execute when terminated by `;`, so a query can span
multiple lines. Dot-commands execute on Enter. Tab completion covers
dot-commands and graph labels.

Common dot-commands:

- `.help` — list commands
- `.quit` / `.exit` — leave the shell
- `.labels` / `.rels` / `.schema` / `.indexes` — inspect schema
- `.mode table|csv|json` — set output format
- `.import <file.csv> <NodeType> [--id <col>] [--title <col>]` — import CSV rows as nodes
- `.dump <dir>` — export CSV files plus a `blueprint.json`
- `.read <file>` — run Cypher statements from a file
- `.save [path]` — save the graph to a `.kgl` file
- `.timing on|off` — show query wall time

`Ctrl-C` cancels a running query. `Ctrl-D` exits.

## Other Commands

`export-text` prints the deterministic text projection used by git
textconv:

```bash
kglite export-text app.kgl
```

`diff` compares two graph projections:

```bash
kglite diff before.kgl after.kgl
```
