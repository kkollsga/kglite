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
