# kglite-cli

The interactive Cypher shell for [kglite](https://github.com/kkollsga/kglite)
knowledge graphs — the `sqlite3`-style REPL for `.kgl` files.

## Install

```console
$ pip install kglite-cli      # ships the `kglite` binary on PATH
# or
$ cargo install kglite-cli    # build from source (needs a Rust toolchain)
```

`kglite-cli` is a standalone binary distribution — installing it gives you the
`kglite` command. It's independent of the `kglite` Python library; install
either or both.

## Use

Run one query and exit:

```console
$ kglite query app.kgl "MATCH (n:Person) RETURN n.name AS name" --format json
[
  {
    "name": "Alice"
  }
]
```

Run a scoped write and persist it:

```console
$ kglite write app.kgl "CREATE (:Task {id:'t1', status:'todo'})" \
    --save --write-scope Task --git-sha abc123 --modified-by agent
```

Inspect a dependency frontier:

```console
$ kglite ready-set app.kgl --done 'n.status = "done"' --node-type Task --format csv
```

Ask for the agent-oriented graph description:

```console
$ kglite describe app.kgl
$ kglite describe app.kgl --types Task
$ kglite describe app.kgl --cypher
```

Keep one graph loaded for an agent loop:

```console
$ kglite session app.kgl --format json
{"op":"describe","types":["Task"]}
{"op":"write","query":"CREATE (:Task {id:'t1', status:'todo'})"}
{"op":"query","query":"MATCH (t:Task) RETURN count(t) AS n","format":"json"}
{"op":"save"}
{"op":"exit"}
```

Or open the interactive shell:

```console
$ kglite app.kgl
kglite shell — app.kgl
Type .help for commands, .quit to exit.
kglite> MATCH (n:Person) RETURN n.name AS name LIMIT 3;
name
----
Alice
Bob
Carol
(3 rows)
kglite> .quit
```

Run with no path for a scratch in-memory graph (`$ kglite`). Pure-Rust single
binary over `kglite::api::*` — no Python, no server.

A Cypher statement runs when terminated by `;`, so it can span multiple lines;
dot-commands run on Enter. Tab completes dot-commands and the graph's labels.

## Commands

Non-interactive commands:

- `query <graph.kgl> <cypher> [--format table|csv|json]` — run a read-only Cypher query
- `write <graph.kgl> <cypher> [--format table|csv|json] [--save]` — run a write-capable Cypher statement
- `write --write-scope A,B --git-sha <sha> --modified-by <actor>` — restrict writes and stamp provenance on `auto_timestamp` types
- `ready-set <graph.kgl> --done <predicate> [--relationship DEPENDS_ON] [--node-type T]` — print `CALL ready_set(...)`
- `describe <graph.kgl> [--types T] [--cypher] [--connections]` — print the XML `describe()` document for agents
- `session <graph.kgl>` — process JSONL requests against one in-memory graph (`query`, `write`, `describe`, `save`, `exit`)
- `export-text <graph.kgl>` — print the deterministic text projection used by git textconv
- `diff <a.kgl> <b.kgl>` — compare two graph text projections

Interactive dot-commands:

- `.help` — list commands
- `.quit` / `.exit` — leave the shell
- `.labels` / `.rels` / `.schema` / `.indexes` — schema introspection
- `.mode table|csv|json` — set the output format
- `.import <file.csv> <NodeType> [--id <col>] [--title <col>]` — load a CSV as nodes
- `.dump <dir>` — export a portable CSV + `blueprint.json` copy
  (reload with `kglite.from_blueprint(...)`)
- `.read <file>` — run the Cypher statements in a file
- `.save [path]` — write the graph to a `.kgl` file
- `.timing on|off` — show query wall-time after each statement

Anything else is executed as Cypher. **Ctrl-C** cancels a running query;
**Ctrl-D** exits.
