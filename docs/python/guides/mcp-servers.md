# MCP Servers

> [Model Context Protocol](https://modelcontextprotocol.io/) is the
> protocol Claude / Cursor / agentic CLIs use to call tools. Your
> KGLite graph becomes a server that speaks it over stdin/stdout, and
> the agent gets Cypher access to your data through ordinary tool
> calls — no API to learn, no infrastructure to manage.

`kglite-mcp-server` is a **single, pure-Rust server** built on the
[mcp-methods] framework (rmcp + manifest-driven tool registration; no
Python runtime, no libpython link). It exposes your graph as
`graph_overview` + `cypher_query` over MCP stdio. For project-specific
tools — semantic search, source-file access, parameterised Cypher
lookups, query preprocessing — drop a YAML manifest next to your graph
and the server picks it up automatically. **No fork required for most
customisation.**

> **0.10.26:** the server is reachable two ways, both running the
> identical Rust implementation. `pip install kglite` bundles it *inside*
> the wheel (statically linked into the extension, sharing the one engine
> — no separate wheel, no duplicated engine) and exposes the
> `kglite-mcp-server` command via a thin console-script shim;
> `cargo install kglite-mcp-server` gives the same command as a
> standalone libpython-free binary. (Through 0.10.24 the wheel shipped a
> *Python* server; 0.10.25 retired it for cargo-only to stop two
> implementations drifting; 0.10.26 brought the command back to `pip` as
> the bundled Rust server.)

[mcp-methods]: https://github.com/kkollsga/mcp-methods

## Quick Start

### 1. Install

```bash
pip install kglite          # ships the kglite-mcp-server command in the wheel
# — or, for a standalone binary with no Python at all:
cargo install kglite-mcp-server
```

Either way the `kglite-mcp-server` command lands on PATH running the same
Rust server. Run `kglite-mcp-server --help` to confirm.

For semantic search (`text_score()`) in the server, name an embedding engine in
the manifest `extensions.embedder` block — you provide the `library` and the
`model`, and install that library:

- **pip wheel** → a Python library: `library: sentence-transformers` (`pip
  install sentence-transformers` — has `bge-m3` + all of HuggingFace) or
  `library: fastembed` (`pip install fastembed` — light, but no `bge-m3`).
- **standalone cargo binary** → `library: fastembed-rs` +
  `cargo install kglite-mcp-server --features fastembed` (no Python in the
  deployment; has `bge-m3`).

See the [embedder example](../examples/manifest_with_embedder.md). Note the two
fastembeds are *separate* libraries with different catalogs (bge-m3 is in
fastembed-rs + sentence-transformers, **not** fastembed-py), and the runtime
model must match the one the graph was embedded with.

### 2. Point it at a graph file

```bash
kglite-mcp-server --graph /path/to/my_graph.kgl
```

The server speaks MCP over stdio and exposes three tools out of the box:

- `graph_overview(...)` — wraps `describe()` for progressive schema
  disclosure (types, connections, Cypher reference).
- `cypher_query(query)` — runs any Cypher query; inline result up to
  15 rows, append `FORMAT CSV` for a localhost-served file export.
- `ping(message?)` — liveness probe; echoes the message or returns `pong`.

Want semantic search (`text_score()` inside Cypher) or source-file
access tools? Drop a manifest — see step 4 below or the
[Customising with a manifest](#customising-with-a-manifest) section.

### Agent graph workbench (opt-in writes)

Servers are read-only by default. Add `--writable` when the agent must mutate
or switch graphs:

```bash
# Open an existing graph with mutation + lifecycle tools.
kglite-mcp-server --graph /data/work.kgl --writable

# Create a missing graph explicitly in one of the three storage modes.
kglite-mcp-server --graph /data/new.kgl --storage memory --writable
# --storage mapped|disk may point at a directory-backed graph instead.
```

Writable mode registers mutation-capable `cypher_query` plus `load_graph`,
`create_graph`, and `save_graph_as`; persistence to the active target is also
available. `--storage` is a creation choice, not a silent conversion of an
existing graph. Keep the default read-only mode for untrusted clients, and use
`write_scope`/schema controls when writes should be type-scoped.

### 3. Register with Claude Desktop

Add to your Claude Desktop config (`~/Library/Application Support/Claude/claude_desktop_config.json` on macOS):

```json
{
  "mcpServers": {
    "my-graph": {
      "command": "/abs/path/to/your/venv/bin/kglite-mcp-server",
      "args": ["--graph", "/abs/path/to/my_graph.kgl"]
    }
  }
}
```

**Use the absolute path to the binary in `command`, not a bare
`kglite-mcp-server`.** A bare command is resolved against `$PATH`, and if
an older install sits earlier on `$PATH` (a stray `cargo install`, a Conda
base env, a previous editable build) the client silently launches *that*
one — a stale server that may register a different tool set or lazy-load
tools your client then can't see. There's no error; the tools just quietly
differ. Point `command` at the exact binary you mean (`which
kglite-mcp-server` inside your active env prints it). See
[Operator notes](#operator-notes) on multi-install PATH order.

For Claude Code, add to `.claude/settings.json` with the same shape.
The agent can now call `graph_overview()` to learn the schema and
`cypher_query()` to query.

```{important}
**Restart after any config change.** The manifest and the client's MCP
config are read **once, at server boot**. If you edit this JSON, the
manifest YAML, or a `.env`, the running server won't pick it up — fully
restart Claude Desktop / your MCP client (or the standalone process) so it
re-reads them. A surprising number of "my change had no effect" reports are
just this.
```

### Verify your setup

Because a misconfigured server fails *silently* — missing tools, github
tools hidden for lack of a token, a stale PATH-shadowing binary, or "No
active graph" — the absence of errors doesn't mean it's working. Run the
built-in self-test to get a positive green/red answer:

```bash
kglite-mcp-server --selftest --graph /abs/path/to/my_graph.kgl
# …or with a manifest / workspace:
kglite-mcp-server --selftest --mcp-config /abs/path/to/manifest.yaml
```

It re-spawns the server with the *same* flags, drives a real MCP handshake
(`initialize` → `tools/list` → activate → `cypher_query`), and prints one
line per capability:

```
kglite-mcp-server --selftest  (mode: single-graph)
  ✓ server initializes: serverInfo.name = KGLite (single-graph)
  ✓ graph tools registered: cypher_query + graph_overview present
  – github tools: none registered (no GITHUB_TOKEN reachable, or disabled)
  ✓ graph hydrates: MATCH (n) RETURN count(n) → 1 row(s):
Selftest PASSED — the server is configured correctly.
```

It exits non-zero if any check fails, so you can also wire it into a
deployment or CI smoke gate. Pass the *absolute path to the binary you
registered* (per the caveat above) so the self-test exercises the same
server your client launches.

### 4. (Optional) Add a manifest for more tools

Drop a sibling YAML file next to your graph and you get three more
tools without writing any Python:

```yaml
# my_graph_mcp.yaml
source_root: ./data
```

That auto-registers `read_source`, `grep`, and `list_source` over the
`./data` directory (sandboxed, ripgrep-backed, gitignore-aware). Cypher
narrows the search at the graph level; the agent follows up with
`read_source` for the top hits or `grep` for context the graph didn't
lift. Full reference is in
[Customising with a manifest](#customising-with-a-manifest) below.

## Customising with a manifest

A **manifest** is a YAML file that sits next to your graph and tells
`kglite-mcp-server` to register additional tools. Drop a file named
`<graph_basename>_mcp.yaml` alongside your graph and it loads
automatically:

```
demo.kgl
demo_mcp.yaml      ← auto-detected sibling
```

Or point at any path with `--mcp-config`:

```bash
kglite-mcp-server --graph demo.kgl --mcp-config /path/to/manifest.yaml
```

A manifest can declare several kinds of additions, all optional:

| Section | What it does | Trust |
|---|---|---|
| `source_root:` / `source_roots:` | Auto-registers `read_source` / `grep` / `list_source` over the directory tree | None — read-only |
| `tools[].cypher` | Parameterised Cypher templates as named MCP tools | None — read-only |
| `extensions.embedder` | Registers an embedder so `text_score()` works inside Cypher | `trust.allow_embedder: true` |
| `extensions.csv_http_server` | Localhost listener that serves `FORMAT CSV` exports as URLs | None |
| `extensions.value_codecs` | Position-scoped literal conversions (`'Q42'↔42`) bound to a property, applied after parsing | none (declarative; presence is opt-in) |
| `workspace:` | Bind a local directory (or clone-and-track GitHub repos) as the active source root | None |
| `builtins.save_graph: true` | Registers `save_graph` so the agent can persist mutations | None |

### `source_root:` — first-class source-file access

Most knowledge graphs index *something* — a codebase, a JSON corpus,
scraped documents. The agent flow is almost always: Cypher narrows
the search, source read for the top hits, occasional grep for
context that didn't make it into the graph. Wire it in with one line:

```yaml
# demo_mcp.yaml
source_root: ./data
```

`./data` is resolved relative to the yaml file's directory, so a
manifest in `/proj/demo_mcp.yaml` exposes `/proj/data`. Use `../`
to point at a sibling directory:

```yaml
source_root: ../scrape
```

For multi-root setups, use `source_roots:`:

```yaml
source_roots:
  - ./data
  - ../shared/lookups
```

This auto-registers three tools, all sandboxed to the configured
roots:

**`read_source(file_path, ...)`** — read a file relative to the
source root. Use `grep="pattern"` to filter to matching lines
instead of dumping everything (essential for large files — agents
can search a 50 MB JSON without exhausting context).

| Parameter | Type | Default | Notes |
|---|---|---|---|
| `file_path` | string | (required) | Relative to a configured source root. |
| `start_line` / `end_line` | int / int | `1` / EOF | 1-indexed line slice. |
| `grep` | string | `None` | Filter to lines matching this regex. |
| `grep_context` | int | `2` | Lines of context around each match. |
| `max_matches` | int | (none) | Cap matches when `grep` is set. |
| `max_chars` | int | (none) | Cap output size. |

**`grep(pattern, ...)`** — regex search across all files in the
source roots. Backed by ripgrep crates, gitignore-aware by default.

| Parameter | Type | Default | Notes |
|---|---|---|---|
| `pattern` | string | (required) | Regex pattern. |
| `glob` | string | `*` | File-name glob filter. |
| `context` | int | `0` | Lines of context around matches. |
| `max_results` | int | `50` | Cap result count. |
| `case_insensitive` | bool | `false` | Toggle case sensitivity. |

**`list_source(...)`** — tree-formatted directory listing under
the first source root.

| Parameter | Type | Default | Notes |
|---|---|---|---|
| `path` | string | `.` | Directory relative to source root. |
| `depth` | int | `1` | Tree depth; `2+` is recursive. |
| `glob` | string | `None` | Filter entries by name. |
| `dirs_only` | bool | `false` | Hide files; directories only. |

All path resolution is sandboxed — `..` traversal that escapes the
configured roots is rejected.

### `tools:` — inline Cypher tools

Declare Cypher templates as named MCP tools. Each entry becomes a
top-level tool the agent can call by name with typed parameters:

```yaml
tools:
  - name: similar_sessions
    description: Top-k semantically similar sessions for a session id.
    parameters:
      type: object
      properties:
        session_id:
          type: string
        top_k:
          type: integer
          default: 5
      required: [session_id]
    cypher: |
      MATCH (s:Session {id: $session_id})-[r:SIMILAR_TO]->(t:Session)
      RETURN t.id AS id, t.title AS title, r.score AS score
      ORDER BY score DESC LIMIT $top_k
```

The agent sees `similar_sessions(session_id, top_k=5)` as a regular
MCP tool. Param names in the Cypher (`$session_id`, `$top_k`) bind
to the JSON Schema properties at call time.

**Validation runs at server startup**, not at agent call time:

- Every `$param` in the Cypher must appear in `parameters.properties`
- The schema itself must be valid JSON Schema (Draft 2020-12)

Typos surface at boot with a clear error pointing at the yaml file —
not 30 seconds into a conversation.

Manifest Cypher tools cap output at 15 rows / 2k chars. For full
result exports, agents use the bundled `cypher_query` with
`FORMAT CSV`.

### `extensions.embedder` — semantic search inside Cypher

Wire bge-m3 (or any fastembed-catalog model) so `text_score()` works
inside `cypher_query`. Loading model code is explicit and trust-gated:

```yaml
trust:
  allow_embedder: true
extensions:
  embedder:
    library: sentence-transformers
    model: BAAI/bge-m3
```

Worked example at
{doc}`../examples/manifest_with_embedder`. Reference under
[`extensions:` schema reference](#extensions-schema-reference) below.

### `extensions.value_codecs` — convert literals in/out

Map the agent's natural input onto your stored types — and read it back in
the form the agent typed — for one declared property at a time. Bound to a
property and applied **after parsing** (never as raw-text substitution), so
it's position-scoped and can't mangle unrelated literals. Three kinds:

- **`prefix`** — strip/add a fixed prefix (Wikidata `'Q42'↔42`, `gene:BRCA1`).
- **`map`** — a fixed bijective lookup table (enum `'active'↔1`).
- **`regex`** — full-match rewrite of the literal (date `'31.12.2020'→'2020-12-31'`).

Decode runs on query-side literals in the property's position; encode runs on
direct result-column projections of it. No trust gate — a codec is pure
declarative data transformation. Worked example at
{doc}`../examples/manifest_value_codecs`. Reference below.

> Replaces `extensions.cypher_preprocessor` (removed in 0.10.27) — that hook
> rewrote raw query *text* before parsing, which could corrupt string
> literals / RETURN aliases. `value_codecs` does the conversion at a safe,
> post-parse, position-scoped site instead.

### Top-level fields

```yaml
name: My Graph                        # Server display name (optional)
instructions: |                       # Replaces default instructions (optional)
  Custom prompt shown to the agent at server-info time.
skills: true                          # Turn on the skill system (see below)
source_root: ./data                   # OR source_roots: [./data, ../alt]
trust:
  allow_embedder: true                # Required when extensions.embedder exists.
builtins:
  save_graph: false                   # Default false — gate write-back tool.
  temp_cleanup: on_overview           # Wipe temp/ on every bare graph_overview().
extensions:                           # kglite-specific addons (see matrix below).
  embedder:
    library: sentence-transformers    # or fastembed (py) / fastembed-rs (cargo)
    model: BAAI/bge-m3
  csv_http_server:
    port: 8765
    dir: temp/
tools:
  - name: ...                         # See sections above
```

Anything else fails fast at load time with the offending key
listed.

### `skills:` — teach agents how to use the tools

`skills: true` turns on the **skill system**: bundled and operator-authored
markdown that injects per-tool and cross-tool methodology (and TRIGGER/SKIP
routing) directly into tool descriptions, gated per-graph. Reach for this
instead of stuffing everything into `instructions:` — skills re-surface in
`tools/list`, attach to specific tools, and stay silent on graphs they don't
fit. Drop files into a `<basename>.skills/` directory beside the manifest.

The full authoring spec — frontmatter schema, `applies_when` gating, the
three text channels, size limits — is its own guide: {doc}`mcp-skills`.

### Common boot errors

The manifest is validated before `mcp.run()` is called, so most
configuration mistakes surface as a one-line `ERROR:` to stderr at
startup with a non-zero exit code. The recurring ones:

| Error message | What it means | Fix |
|---|---|---|
| `ERROR: <path>: unknown top-level keys: ['foo']` | Typo or unsupported key in manifest. | Compare against the [top-level field list](#top-level-fields). |
| `ERROR: <path>: source root './data' resolves to '/abs/.../data' which is not an existing directory` | The path is relative-to-yaml; it didn't land on a real directory. | Check the path; create the directory; or use `source_roots:` if you have multiple. |
| `ERROR: <path>: cypher tool 'foo': cypher references $params ['bar'] not declared in parameters.properties` | A `$param` in the Cypher template isn't in the JSON Schema. | Add it under `parameters.properties` (and to `required:` if it's mandatory). |
| `ERROR: <path>: cypher tool 'foo': invalid parameters schema: ...` | The `parameters:` block isn't valid JSON Schema (Draft 2020-12). | Check `type`, nested types in `properties`, and `required:` list. |
| `ERROR: --mcp-config path does not exist: <path>` | Explicit `--mcp-config` value points at a missing file. | Check the path. Sibling auto-detect is `<basename>_mcp.yaml`. |
| `ERROR: extensions.value_codecs ... is not bijective` | A `map` codec has two keys mapping to the same value, so encode is ambiguous. | Make the `map:` one-to-one. |
| `ERROR: value_codecs[i].match ... is not a valid regex` | A `regex` codec's `match` doesn't compile. | Fix the regex (anchor it for a full match). |

Exit code 3 is reserved for manifest / validation errors; exit 1 for
graph-file-not-found and for missing runtime dependencies. Wrapping
scripts can branch on those.

## End-to-end example: a conference catalog graph

A graph indexing conference sessions, speakers, and companies, with
embedding-derived similarity edges between sessions. Manifest
co-locates with the graph file and the source data:

```
conference/
├── conference.kgl
├── conference_mcp.yaml          ← auto-detected
└── data/
    ├── sessions/
    │   └── classified.json
    └── speakers/
```

```yaml
# conference_mcp.yaml
name: Conference Graph
instructions: |
  Conference catalog — sessions, speakers, companies, plus
  similarity edges between sessions. Use cypher_query for
  structured questions, read_source/grep for raw JSON in ./data,
  similar_sessions for embedding-based recommendations,
  session_detail for the full session record by id.

source_root: ./data

tools:
  - name: similar_sessions
    description: Top-k semantically similar sessions for a session id.
    parameters:
      type: object
      properties:
        session_id: {type: string}
        top_k:      {type: integer, default: 5}
      required: [session_id]
    cypher: |
      MATCH (s:Session {id: $session_id})-[r:SIMILAR_TO]->(t:Session)
      RETURN t.id AS id, t.title AS title, r.score AS score
      ORDER BY score DESC LIMIT $top_k

  - name: session_detail
    description: Full record for a session by id.
    parameters:
      type: object
      properties:
        session_id: {type: string}
      required: [session_id]
    cypher: |
      MATCH (s:Session {id: $session_id})
      OPTIONAL MATCH (s)-[:PRESENTED_BY]->(speaker:Speaker)
      OPTIONAL MATCH (speaker)-[:WORKS_AT]->(company:Company)
      RETURN s, collect(DISTINCT speaker) AS speakers,
             collect(DISTINCT company) AS companies
```

Run with:

```bash
kglite-mcp-server --graph conference.kgl
```

Tools registered (visible in any MCP-aware agent):

- `graph_overview`, `cypher_query`, `ping` — core graph tools
- `read_code_source`, `explore` — code-graph-aware tools
- `read_source`, `grep`, `list_source` — from `source_root`
- `similar_sessions` — inline Cypher
- `session_detail` — inline Cypher

The exact list is mode-dependent. `save_graph` is registered only when the
manifest opts in with `builtins.save_graph: true` or the server runs with
`--writable`; write-enabled workbench mode also adds graph lifecycle tools.
For mapping the agent's input onto your stored types (Wikidata
`'Q42'↔42`, enum codes, date formats), see
{doc}`../examples/manifest_value_codecs`. For full Rust integration, see
**Building a downstream binary** below.

## Building a downstream binary

When manifest Cypher templates aren't enough — domain logic needs to share the
active graph, materialize files, or conditionally register tools — embed the
`kglite-mcp-server` library and add tools through `ServerExtensions`. KGLite
still owns graph/Cypher/source tools, manifests, skills, watchers, and stdio;
the downstream binary owns only its domain methods.

The shape:

```rust
use kglite_mcp_server::{run_with_extensions, ServerExtensions};

fn main() -> anyhow::Result<()> {
    let extensions = ServerExtensions::new().with_domain_tools(|registry| {
        let graph = registry.graph_state().clone();
        registry.register_typed_tool::<MyArgs, _>(
            "my_tool",
            "What the domain tool does.",
            move |args| graph.with_context(|context| {
                my_domain_logic(context.graph(), context.root(), args)
            }).unwrap_or_else(|| "no active graph".to_string()),
        )
    });
    run_with_extensions(std::env::args_os(), extensions)
}
```

The registry rejects names already owned by KGLite or manifest tools. Use
`DomainGraphState::with_context` when the result needs both graph data and its
identity: the borrowed graph, save target, and source root come from one
active-slot snapshot. Keep that callback short and read-only. The registry also
offers `register_route` for a custom rmcp `ToolRoute` while preserving the same
collision check. See the compiling
[`domain_tools.rs`](https://github.com/kkollsga/kglite/blob/main/crates/kglite-mcp-server/examples/domain_tools.rs)
example. If you need to replace KGLite tools or change its stdio transport,
build a separate server directly on
[`mcp-methods`](https://crates.io/crates/mcp-methods); that is a server fork,
not domain-tool composition.

When deciding between a manifest and a composed downstream binary:

| Need | Manifest | Downstream binary |
|---|---|---|
| Read-only tools (Cypher templates, source access) | ✅ | overkill |
| Executable Rust/domain logic | ❌ | ✅ |
| Tool registration conditional on active graph | ❌ | ✅ |
| Custom rmcp transports / middleware | ❌ | separate server |
| Replacing `cypher_query` / `graph_overview` | ❌ | ❌ |

Most projects never need a downstream binary.

## Built-in patterns

### `FORMAT CSV` export

When agents need full result sets (not just 15 rows), they append
`FORMAT CSV` to the query. The Rust binary saves it to a temp file
and serves it over a localhost HTTP server with CORS — agents can
generate HTML artifacts that `fetch()` the CSV at runtime instead
of hardcoding thousands of rows into the artifact source.

### Mode banner — tell the agent which conditional tools are registered

Whichever CLI mode the server is in
(`--graph` / `--workspace` / `--watch` / `--source-root` / bare /
local-workspace via manifest), the server prepends a per-mode
**banner** to two surfaces:

- the `instructions` block returned during MCP `initialize` (read
  once at handshake), and
- the bare `graph_overview()` response preamble (re-read on every
  call, survives context aging on long sessions).

The banner names every conditional tool — both the registered ones
and the unregistered ones — so the agent can see at a glance whether
`repo_management`, `set_root_dir`, or `save_graph` are available
without trial-calling each one. Example for workspace mode:

```
[kglite-mode] workspace (clone-and-activate)
- repo_management: registered. Start with:
    repo_management()             — list known repos
    repo_management('org/repo')   — clone + activate
- cypher_query / graph_overview: registered (operate on the active repo's graph).
- save_graph / set_root_dir: not in this mode.
```

The `[kglite-mode]` marker identifies the segment for downstream
tooling. Operator-declared `instructions:` / `overview_prefix:` text
follows the banner unchanged.

### Multi-revision code graphs

Both activation paths take an optional `revs` argument that builds the
code graph across several git revisions instead of just the latest:

- **github mode:** `repo_management('org/repo', revs=5)` (last 5 release
  tags + `HEAD`) or `repo_management('org/repo', revs=['v1.0', 'v2.0', 'HEAD'])`.
- **local mode:** `set_root_dir('/path', revs=5)` or `set_root_dir('/path', revs=[…])`.

An integer means "the last N release tags (`git tag --sort=-v:refname`)
plus `HEAD`"; a list is passed through as explicit revspecs, oldest →
newest. Omitting `revs` is the unchanged single-revision build.

The result is **one merged graph**, not N graphs: each entity is stored
once, carrying a `revs: [str]` list (the revisions it appears in) and a
`rev_fp: [int]` shape fingerprint (positionally aligned with `revs`).
Ordinary properties report the newest rev an entity appears in
(newest-wins), so plain Cypher reads `HEAD`'s value. The active-graph
header names the loaded set — `<active_graph … revs="v1.0,v2.0,HEAD"/>` —
and the activation message teaches the scoping idiom below.

Because the graph spans every rev, an **unscoped** query counts the
union across revs (an over-count trap). Scope to a single rev with list
membership, and use `CALL rev_diff` for deltas:

```cypher
-- Everything present in v2.0 (scoped — no over-count)
MATCH (n:Function) WHERE 'v2.0' IN n.revs RETURN n.name

-- What changed between two revs (added / removed / changed)
CALL rev_diff({from: 'v1.0', to: 'HEAD'})
YIELD bucket, type, qualified_name, name, file, line
RETURN bucket, type, qualified_name, file, line
```

See the [Cypher reference](../../reference/cypher-reference.md) `rev_diff` entry and the
codingest project for the full build semantics.

**Two operator caveats:**

- **Older clones may lack tags.** Pre-0.3.49 mcp-methods cloned
  depth-1/tag-less, so `revs=N` (tag-based) finds nothing to resolve.
  `repo_management('org/repo', update=True)` fetches tags; a full
  re-clone restores complete history. Fresh clones under 0.3.49+ bring
  tags automatically.
- **Activation cost scales with rev count.** Each rev is a full parse,
  so a `revs=N` build costs ≈ N × a single build. On a large repo,
  start small (`revs=2`–`3`) before loading a deep history.

### Mutable graphs

`save_graph` is built in: when the manifest sets `builtins.save_graph: true`
(single-graph mode), the tool registers automatically and persists
post-mutation graph state to the source `.kgl` path. The mode banner
above flips its `save_graph` line from "not registered (read-only)"
to "registered. Call to persist CREATE / SET / DELETE mutations."
when this is on.

### Semantic search (`text_score()`)

`extensions.embedder` in the manifest registers an embedder so
`text_score()` works inside Cypher. The agent can then write:

```cypher
MATCH (a:Article)
WHERE text_score(a, 'summary', 'renewable energy') > 0.4
RETURN a.title, text_score(a, 'summary', 'renewable energy') AS score
ORDER BY score DESC LIMIT 10
```

Full schema in the
[`extensions:` schema reference](#extensions-schema-reference)
below; worked example at
{doc}`../examples/manifest_with_embedder`. The embedder protocol
itself is in [Semantic Search](semantic-search.md).

### Security

- **Read-only mode** rejects mutations at the Cypher level — set via
  `graph.read_only(True)` before binding to the server, or use
  single-graph mode with `builtins.save_graph: false` (the default).
- **Path traversal** is blocked by the framework's source tools: the
  bundled `read_source` / `grep` / `list_source` canonicalise every
  path against the configured `source_root` before any I/O.

**Query parameters** — when passing user input to Cypher, use
`params` to prevent injection:

```python
graph.cypher("MATCH (n) WHERE n.name = $name RETURN n", params={"name": user_input})
```

## Deployment shapes

### Small / medium graphs (`.kgl` file)

The Quick Start path — graph fits in memory, load via
`kglite.load(path)` or save via `g.save(path)`, point the CLI at
the resulting `.kgl` file. Default storage. No special config.
Suitable up to ~10M nodes on a developer laptop, larger on a
beefier host.

### Large graphs (disk-backed)

For graphs that don't fit in memory or take too long to deserialise
on every boot, build a disk-backed graph once and point the CLI at
its directory:

```python
# One-off ingestion (e.g. from Wikidata's truthy.nt.bz2 dump):
import kglite

# Streams the dump straight into a disk-backed graph in
# `/data/wikidata-graph/`.
g = kglite.KnowledgeGraph(storage="disk", path="/data/wikidata-graph/")
g.load_ntriples("latest-truthy.nt.bz2", languages=["en"], verbose=True)
```

(The pre-packaged dataset loaders — SEC EDGAR, Sodir, Wikidata —
live in the separate kglite-datasets project and wrap this same
`load_ntriples` path with download/cooldown/resume; kglite loads
the graphs they produce.)

Then run the MCP server against the directory (not a single file):

```bash
kglite-mcp-server --graph /data/wikidata-graph/
```

The CLI's `--graph` validator accepts both shapes — a `.kgl` file
OR a directory containing `disk_graph_meta.json` (the disk-graph
sentinel). For your own data, the API is
`kglite.KnowledgeGraph(storage="disk", path="/data/graph/")` for
the constructor and `g.add_nodes(...) / g.add_connections(...)`
for population.

Manifests work the same way for both shapes. For example,
`wikidata_mcp.yaml` next to (or pointed at via `--mcp-config` for)
the graph dir:

```yaml
name: Wikidata
extensions:
  value_codecs:
    - property: id          # integer-keyed column
      kind: prefix
      prefix: "Q"           # decode 'Q42' → 42 ; encode 42 → 'Q42'
      stored_type: int
```

See {doc}`../examples/manifest_value_codecs` for the full Wikidata example
(Q-number ↔ integer, plus the `map` and `regex` kinds).

## Troubleshooting

Common post-boot pitfalls, grouped by symptom.

### `github_issues` says "could not auto-detect from git remote"

`GITHUB_TOKEN` (or `GH_TOKEN`) isn't in the server's environment.
The token is loaded from:

1. The process environment when the server boots.
2. The manifest's `env_file:` path (explicit).
3. A `.env` file discovered by walking up from the active mode's
   directory.

Existing process env never gets overwritten by the `.env` file. To
verify: the server logs `loaded env file: <path>` on stderr when
it finds a `.env`. Absence of that line means no `.env` was
discovered, and process env is what's in effect.

### `text_score()` returns 0.0 for every node

The embedder isn't bound. Causes, in order of likelihood:

- The manifest didn't declare `extensions.embedder`. See
  [`extensions:` schema reference](#extensions-schema-reference).
- The model couldn't download (network issue) or load (out-of-
  memory). Look for tracebacks in the server's stderr at boot.
- The property being scored doesn't exist on the matched nodes.
  `text_score(n, 'summary', 'query')` returns 0.0 when `n.summary`
  is null. Use `WHERE n.summary IS NOT NULL` to filter first.

### Warm `text_score()` is slow (seconds, not milliseconds)

bge-m3's cool-down may have released the ONNX session. The default
`cooldown` is 900 seconds (15 min) — set
`extensions.embedder.cooldown: 0` in the manifest to keep the
session resident forever (heavy-use mode), or pick a larger value
matching your usage pattern. See
{doc}`../examples/manifest_with_embedder` for the tradeoff table.

### Conda environment lifts an old `kglite-mcp-server`

If `which kglite-mcp-server` resolves outside your active env,
your shell PATH is finding an older install (typically from a
prior `cargo install` or a different conda env). Drop the old
install (`rm $(which kglite-mcp-server)` from outside the active
env) or activate the right env explicitly.

### Server boots but `tools/list` shows fewer tools than expected

The [tool gating matrix](#tool-gating) shows the conditions each
tool needs to register. Most common cases:

- `repo_management` missing — repository cloning is a `codingest-mcp`
  workspace feature. `set_root_dir` requires `workspace.kind: local`.
- `read_source` / `grep` / `list_source` missing — no source root
  is configured (no `source_root:` in the manifest, no `--source-root`
  CLI flag, and `--graph` parent auto-bind didn't fire).
- `github_issues` / `github_api` missing — no `GITHUB_TOKEN` in env.
- `save_graph` missing — you're not in `--graph` mode OR the
  manifest doesn't set `builtins.save_graph: true`.

### PyPI says "No matching distribution found" immediately after a release

PyPI's `simple/` index lags the JSON metadata by ~few minutes
after publish. Workaround:

```bash
pip install --index-url https://pypi.org/simple/ --no-cache-dir 'kglite==X.Y.Z'
```

Or wait a few minutes. This is a PyPI mirror-cache behaviour, not
a kglite packaging issue.

## Reference

The full programmable surface of `kglite-mcp-server`, with the
"what's documented enough that an agent or operator can rely on
it?" stance: anything in this section is treated as a contract.

### Mode × YAML-field acceptance matrix

Which manifest key takes effect in which CLI mode. "—" means the
key parses cleanly but has no behavioural effect in that mode (the
same YAML can move between modes without edits). The graph file is
the discriminator for `--graph` / `--workspace` / `--watch` /
`--source-root` / bare:

| Manifest key | `--graph` | `--workspace` | `--watch` | `--source-root` | bare (no graph) |
|---|---|---|---|---|---|
| `name`, `instructions`, `overview_prefix` | ✓ | ✓ | ✓ | ✓ | ✓ |
| `source_root` / `source_roots` | ✓ (overrides parent-of-`.kgl`) | — | — | ✓ (canonical) | ✓ |
| `env_file` | ✓ | ✓ | ✓ | ✓ | ✓ |
| `workspace.kind: local` + `workspace.root: <dir>` | — | — | — | — | promotes into local-workspace mode |
| `workspace.watch: true` | — | — | ✓ (auto-rebuild) | — | ✓ when `workspace.kind: local` |
| `tools[].cypher` | ✓ | ✓ (per active repo) | ✓ | — (no graph) | — |
| `trust.allow_embedder` | parsed, required by `extensions.embedder` | parsed, required by matching extension | parsed, required by matching extension | parsed (no graph) | parsed (no graph) |
| `builtins.save_graph: true` | ✓ (registers `save_graph`) | — (multiple graphs) | — | — | — |
| `builtins.temp_cleanup: on_overview` | ✓ | ✓ | ✓ | ✓ | ✓ |
| `extensions.embedder` | ✓ | ✓ (per active repo) | ✓ | — (no graph) | — |
| `extensions.csv_http_server` | ✓ | ✓ | ✓ | ✓ | ✓ |
| `extensions.value_codecs` | ✓ | ✓ | ✓ | — (no graph) | — |
| `extensions.<other>` (passthrough) | parsed, opaque to framework | parsed, opaque | parsed, opaque | parsed, opaque | parsed, opaque |

Unknown keys at the top level (or under `builtins:` / `workspace:` /
`trust:` / `tools[]`) fail validation at boot with a
non-zero exit and an `ERROR: <path>: unknown ... keys: [...]`
message. Keys under `extensions:` are deliberately unvalidated —
they're the downstream-binary passthrough zone.

### Tool gating

Which tool registers, under what conditions. `tools/list` only ever
shows what's registered, so this also answers "what set of tools
will my agent see?"

| Tool | Registered when | Notes |
|---|---|---|
| `cypher_query` | always | Returns inline rows or CSV URL — see "Tool response formats". |
| `graph_overview` | always | Always available even with no graph: returns the no-graph message. |
| `ping` | always | Liveness probe. |
| `read_code_source` | always | Requires an active graph at call time (returns the no-graph message otherwise). |
| `save_graph` | `--graph` mode AND `builtins.save_graph: true` | Other modes have no single graph to save back to. |
| `read_source` / `grep` / `list_source` | a source root is configured (`--source-root`, `--graph` parent auto-bind, manifest `source_root:`, or active workspace repo) | All three register together; never registered independently. |
| `repo_management` | `codingest-mcp --workspace` clone-tracker mode | Not registered in local-workspace mode; use `set_root_dir` there. |
| `set_root_dir` | `workspace.kind: local` only | Sandboxed against the manifest-declared `workspace.root` for the lifetime of the server. |
| `github_issues` / `github_api` | `GITHUB_TOKEN` (or `GH_TOKEN`) reachable at boot | Token loaded from process env, walk-up `.env`, or explicit `env_file:`. Tools are registered together; never one without the other. |
| Manifest `tools[].cypher` entries | the manifest declares them AND the mode supports cypher (anything but `--source-root` and bare) | Tool names cannot collide with the built-ins above. |

### Tool response formats

Bundled-tool response shapes are treated as version-stable contracts
across patch releases — they're tagged below per stability. Manifest
`tools[].cypher` responses inherit `cypher_query`'s format.

| Tool | Response shape | Stability |
|---|---|---|
| `ping` | `<message>` (default `pong`) | Stable. |
| `cypher_query` (inline) | `<N> row(s)[ (showing first 15)]:\n<TAB-joined column names>\n<TAB-joined repr'd values per row>\n` | Stable post-0.9.22 (the 0.9.21 row-formatter regression is the canonical "this is now a contract" event). |
| `cypher_query FORMAT CSV` with `csv_http_server` | `FORMAT CSV: <N> row(s) written to <url>\nFetch with: curl <url>` | Stable. |
| `cypher_query FORMAT CSV` without `csv_http_server` | Inline CSV body. | Stable. |
| `cypher_query` errors | `Cypher error: <engine message>` | Stable. |
| `graph_overview` | XML schema (see `describe()` output) — types / connections / cypher panes depending on args. | Stable; the XML shape is the canonical agent-facing format. |
| `read_source` | First line: `<path>  (lines X-Y of Z)`, body lines: `   <lineno>: <text>`. Truncation footer when `max_chars` trips: `... (truncated)`. | Stable. |
| `read_source` (path errors) | `Error: path '<path>' does not exist or access denied.` | Stable. |
| `grep` | `<path>:<line>:<text>` for matches, `<path>-<line>-<text>` for context lines. | Stable. |
| `list_source` | Tree-formatted directory listing relative to the primary source root. | Stable. |
| `read_code_source` | First line: `// <qualified_name> (<path>:<start>-<end>)`, body lines: `   <lineno>: <text>`. | Stable. |
| `save_graph` | `Saved <path> (<N> nodes, <M> edges).` (or `Saved <path>.` when schema unavailable). | Stable. |
| `save_graph` (no graph) | `save_graph requires --graph mode (no source path bound).` | Stable. |
| `repo_management` (list) | `<N> live repo(s):\n  <repo>[ [active]]  (<count> access[es], last <when>)` | Stable. |
| `repo_management` (activate) | `Cloned 'org/repo' at <path>.` / `Updated 'org/repo' at <path>.` / `Activated (already up to date) 'org/repo' at <path>.` | Stable. |
| `set_root_dir` (success) | `Active root set to <absolute_path>.` | Stable. |
| `set_root_dir` (escape) | `Error: path '<path>' escapes the workspace root.` | Stable. |
| `github_issues` (FETCH) | Issue/PR/discussion body with `cb_N` / `patch_N` / `comment_N` / `review_N` placeholders for collapsed elements. Drill down with `element_id=<placeholder>`. | Stable. |
| `github_issues` (LIST/SEARCH) | `<N> discussions in org/repo (<state>):` then per-line summary. | Stable. |
| `github_api` | Pretty-printed JSON body, truncated to `truncate_at` chars (default 80 000). | Stable. |
| (any tool, no active graph) | `No active graph. Pass --graph X.kgl, or activate one via repo_management('org/repo').` | Stable. |

If a future release needs to change a stable shape, that's a breaking
change tracked in the `CHANGELOG.md` "Changed" section (not "Fixed")
and the version bumps minor, not patch.

### `extensions:` schema reference

The `extensions:` block is the kglite-specific addon namespace. The
keys validated below are first-class — they have parser-level
validation, default values, and contracts. Anything else under
`extensions.*` is opaque passthrough.

Machine-readable JSON Schema (Draft 2020-12) for each first-class
block lives under [`docs/schemas/extensions/`][schemas-dir] in the
repo:

- [`csv_http_server.json`][schema-csv]
- [`embedder.json`][schema-embedder]
- [`value_codecs.json`][schema-value-codecs]

The schemas are anchored to the Python parsers by the regression
test `tests/test_extensions_schemas.py` — any drift between
"what the parser accepts" and "what the schema accepts" surfaces
as a test failure on the next CI run.

[schemas-dir]: https://github.com/kkollsga/kglite/tree/main/docs/schemas/extensions
[schema-csv]: https://github.com/kkollsga/kglite/blob/main/docs/schemas/extensions/csv_http_server.json
[schema-embedder]: https://github.com/kkollsga/kglite/blob/main/docs/schemas/extensions/embedder.json
[schema-value-codecs]: https://github.com/kkollsga/kglite/blob/main/docs/schemas/extensions/value_codecs.json

#### `extensions.embedder`

Registers an embedder so `text_score()` works inside Cypher.

```yaml
extensions:
  embedder:
    library: sentence-transformers  # the engine; host (Python/Rust) inferred from it
    model: BAAI/bge-m3              # required (passed to the library)
    # cooldown: 900                 # fastembed-rs only; seconds (default 900). 0 = never release.
```

| Field | Type | Default | Constraint |
|---|---|---|---|
| `library` | string | `fastembed` | `fastembed` / `sentence-transformers` (Python, wheel) · `fastembed-rs` (Rust, cargo). |
| `model` | string | (required) | Passed to the chosen library; must be in *its* catalog. |
| `factory` | string | — | `module:attr` returning an `EmbeddingModel` — any custom Python embedder. |
| `cooldown` | int | 900 | `fastembed-rs` only; `0` disables auto-release. |

The legacy `embedder:` block (top-level, 0.9.17 and earlier) is
parsed by the framework but ignored — use `extensions.embedder:` with
an in-catalog model (the server's Rust fastembed backend, built via
`--features fastembed`).

#### `extensions.csv_http_server`

Spawns a localhost HTTP listener (loopback only) that serves CSV
exports produced by `cypher_query ... FORMAT CSV`.

```yaml
extensions:
  csv_http_server:
    port: 8765                      # optional; default 8765
    dir: temp/                      # optional; default temp/ (relative to manifest)
    cors_origin: "*"                # optional; default "*"
```

Also accepts shorthand:

```yaml
extensions:
  csv_http_server: true             # defaults — port 8765, dir temp/
  # or
  csv_http_server: false            # explicitly disabled (same as absent)
```

| Field | Type | Default | Constraint |
|---|---|---|---|
| `port` | int | 8765 | `0 ≤ port ≤ 65535`. |
| `dir` | string | `temp` | Path; resolved against the manifest's parent directory. |
| `cors_origin` | string | `"*"` | Sent in `Access-Control-Allow-Origin`. Use a specific origin for tighter security. |

Only GETs of flat filenames inside `dir` are served. No directory
listings, no write surface from the HTTP layer (writes only come
from the Cypher executor via `FORMAT CSV`).

#### `extensions.value_codecs` (0.10.27+)

A list of operator-declared literal codecs, each bound to a stored property.
Query-side literals in that property's position are decoded before execution;
direct result-column projections of it are encoded back. Applied **after
parsing** (never as raw-text substitution), for `cypher_query` and
`tools[].cypher` only — not `graph_overview`, `read_source`, etc.

```yaml
extensions:
  value_codecs:
    - property: id
      kind: prefix            # prefix | map | regex
      prefix: "Q"             # 'Q42' ↔ 42
      stored_type: int        # int (default) | float | str
    - property: status
      kind: map
      map: { active: 1, archived: 2 }    # must be bijective
    - property: event_date
      kind: regex
      match: '^(\d{2})\.(\d{2})\.(\d{4})$'
      decode: '$3-$2-$1'
      encode: { match: '^(\d{4})-(\d{2})-(\d{2})$', replace: '$3.$2.$1' }  # optional
```

| Field | Type | Default | Constraint |
|---|---|---|---|
| `property` | string | (required) | Stored column the codec governs. |
| `kind` | string | (required) | `prefix` \| `map` \| `regex`. |
| `prefix` | string | (required for `prefix`) | Stripped on decode, added on encode. |
| `stored_type` | string | `int` | `int` \| `float` \| `str` (for `prefix`). |
| `map` | mapping | (required for `map`) | string → value; must be bijective. |
| `match` / `decode` | string | (required for `regex`) | Full-match regex + replacement template. |
| `encode` | `{match, replace}` | none | Optional reverse for `regex`. |

No trust gate — a codec is pure declarative data transformation (no code
execution). A malformed block (bad regex, non-bijective map) is a boot error.

#### `extensions.<other>` (passthrough)

Any other key under `extensions:` parses cleanly and is preserved on
the loaded `Manifest.extensions` dict. The framework does not
validate inner shape. Downstream consumers (kglite-mcp-server, your
own server binaries) read whatever they need from this map.

### `tools[].cypher` template reference

Manifest entries shaped like

```yaml
tools:
  - name: <identifier>
    description: <agent-visible explanation>
    parameters: <JSON Schema object>
    cypher: |
      <Cypher template with $param placeholders>
```

become first-class MCP tools. Behaviour:

**Name** — must match `^[a-zA-Z_][a-zA-Z0-9_]*$`. Cannot collide
with built-in tool names (`cypher_query` / `graph_overview` etc.).

**`$param` substitution** — Cypher templates pass through to
`graph.cypher(query, params=args)` unchanged. The kglite Cypher
engine does typed parameter binding (no string interpolation) — the
JSON value of `args[$name]` becomes a typed value at the
`MATCH (n {field: $name})` site, so injection is impossible by
construction. The agent supplies values per the JSON Schema; kglite
binds them in-engine.

**JSON Schema flavour** — `parameters:` accepts the subset of
JSON Schema (draft 2020-12) that the MCP SDK supports for tool
input. Practically: `type`, `properties`, `required`, `default`,
`description`, `enum`, `items` (for arrays), `minimum`/`maximum`,
`minLength`/`maxLength`, `pattern`. Nested objects work; bring the
schema's complexity in proportion to the tool's parameter
complexity.

**Parameter validation** — the MCP client enforces schema validation
before the tool is dispatched. A type mismatch (string supplied for
an `integer` field) raises an MCP-level error before
`graph.cypher()` runs; the agent receives a structured tool error
rather than a Cypher error.

**Tool errors** — if `graph.cypher()` raises, the response body is
`Cypher error: <engine message>` (the same envelope as
`cypher_query`). Empty result sets render as `No results.`.

**`FORMAT CSV` inheritance** — manifest cypher tools share the
formatting path with `cypher_query`. Append `FORMAT CSV` inside the
template (or `$_csv_format` if you want to gate it on a parameter),
and the tool's output follows the same inline-vs-URL behaviour
documented under "Tool response formats."

**Boot-time validation** — every `$param` named in the template
must appear in `parameters.properties`. Mismatch fails at boot,
not at agent call time.

Worked examples — see the `docs/python/examples/manifest_*.md` pages
(`manifest_cypher_tool`, `manifest_value_codecs`, `manifest_with_embedder`,
`manifest_workspace`).

### Embedder `library` × model catalog

The valid `model:` values depend on the `library:` you pick — the catalogs are
**not** shared:

| `library:` | Catalog | `bge-m3`? |
|---|---|---|
| `sentence-transformers` (pip) | any HuggingFace embedding model | ✅ |
| `fastembed` (pip, fastembed-py) | `bge-*-en-v1.5`, `bge-small-zh-v1.5`, `multilingual-e5-large`, `all-MiniLM-L6-v2`, … (`TextEmbedding.list_supported_models()`) | ❌ |
| `fastembed-rs` (cargo) | `bge-m3`, `bge-{small,base,large}-en-v1.5`, `multilingual-e5-{large,base}`, `all-MiniLM-L6-v2` | ✅ |
| `factory: mod:attr` | whatever your builder loads | — |

**`bge-m3` is in fastembed-rs and sentence-transformers, but not fastembed-py**
— if you set `library: fastembed, model: BAAI/bge-m3` the server fails to boot
(fastembed-py rejects the unknown model). And the runtime model must match the
one the graph was embedded with, or `text_score()` rankings are meaningless.

fastembed (both ports) caches ONNX weights at `~/.cache/fastembed/`;
sentence-transformers uses the HuggingFace cache. First call downloads.

Adding support for a model outside the curated Python libraries doesn't need a
kglite change — use `factory: module:attr` pointing at your own builder.

### Path resolution and manifest discovery

**Relative paths in manifests resolve against the manifest's own
directory.** This applies to `source_root`, `env_file`, every
entry in `source_roots`, `workspace.root`, and
`extensions.csv_http_server.dir`. The rule is unconditional:
no path is interpreted relative to `cwd` unless explicitly
absolute.

Manifest discovery order:

1. `--mcp-config <path>` — explicit path; absolute or
   resolved against cwd.
2. `--graph X.kgl` — auto-detects `<dirname>/<basename>_mcp.yaml`
   next to the graph file (the "sibling" pattern).
3. `--workspace DIR` / `--watch DIR` — auto-detects
   `DIR/workspace_mcp.yaml`.
4. `--source-root` / bare — no auto-detection. Pass `--mcp-config`
   explicitly if you want a manifest.

`.env` discovery order:

1. Manifest `env_file: <path>` — explicit; absolute or relative to
   manifest dir.
2. Otherwise walks upward from the mode path (or cwd in bare mode)
   looking for a `.env` file. Loads the first one found.

Existing process env vars are never overwritten by `.env` —
`GITHUB_TOKEN=...` in your shell wins over the file.

### Operator notes

#### PyPI simple-index lag after publish

After a `kglite` release publishes to PyPI, the `simple/` index
that `pip install` consults can lag the JSON metadata by ~few
minutes. The first `pip install kglite==X.Y.Z` after publish may
return `No matching distribution found`. Workaround:

```bash
pip install --index-url https://pypi.org/simple/ --no-cache-dir 'kglite==X.Y.Z'
```

The `--index-url` forces a direct fetch (some mirrors cache
longer); `--no-cache-dir` bypasses pip's local cache. Wait a few
minutes if you'd rather not pass flags — the lag is consistent.

This is a PyPI / mirror-cache behaviour, not a kglite packaging
problem.

#### Conda + multiple Pythons

`pip install kglite` against a conda env's Python (`conda
activate myenv && pip install kglite`) Just Works — no `PYO3_PYTHON=`,
no `install_name_tool` patching. It also installs the `kglite-mcp-server`
command into that env (the bundled Rust server). If you *also* ran
`cargo install kglite-mcp-server`, both land on PATH — `which
kglite-mcp-server` confirms which install you're running (they run the
same server, so it rarely matters).

#### Watch mode rebuild costs

`workspace.watch: true` + `--watch DIR` rebuilds the code graph under
`codingest-mcp`; the generic `kglite-mcp-server` intentionally has no builder
on every debounced file change (500 ms default debounce). For source
trees over 100k LoC this costs a few seconds per rebuild. The
rebuild runs on a background thread; queries against the previous
graph keep working until the new graph atomically swaps in.

## Migrations

Pre-0.9.20 operators upgrading from a bundled-binary install: see
{doc}`../migrations/mcp-pre-0.9.20` for the 0.9.17→0.9.18 (Python
embedder + tools[].python removal, csv_http_server introduction)
and 0.9.19→0.9.20 (bundled-binary → Python entry point) migration
notes.

## Worked examples

End-to-end manifest snippets, each focused on one feature:

```{toctree}
:maxdepth: 1

../examples/manifest_cypher_tool
../examples/manifest_with_embedder
../examples/manifest_workspace
../examples/manifest_value_codecs
```
