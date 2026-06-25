# kglite-cli

The interactive Cypher shell for [kglite](https://github.com/kkollsga/kglite)
knowledge graphs — the `sqlite3`-style REPL for `.kgl` files.

```console
$ kglite app.kgl
kglite shell — app.kgl
Type .help for commands, .quit to exit.
kglite> MATCH (n:Person) RETURN n.name AS name LIMIT 3
name
----
Alice
Bob
Carol
(3 rows)
kglite> .quit
```

Run with no path for a scratch in-memory graph:

```console
$ kglite
```

Pure-Rust single binary over `kglite::api::*` — no Python, no server.
`cargo install kglite-cli` installs the `kglite` binary.

## Commands

- `.help` — list commands
- `.quit` / `.exit` — leave the shell
- `.labels` / `.rels` / `.schema` / `.indexes` — schema introspection
- `.mode table|csv|json` — set the output format
- `.dump <dir>` — export a portable CSV + `blueprint.json` copy
  (reload with `kglite.from_blueprint(...)`)
- `.read <file>` — run the Cypher statements in a file
- `.save [path]` — write the graph to a `.kgl` file

Anything else is executed as Cypher. **Ctrl-C** cancels a running query;
**Ctrl-D** exits. (`.import` is not yet supported — there's no `LOAD CSV`; use
`.read` or `kglite.from_blueprint()`.)
