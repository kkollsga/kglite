# Examples: workspace mode (local + github-clone-tracker)

`kglite-mcp-server --workspace <dir>` runs the server with a workspace
backing it. Two flavours, picked by the manifest's `workspace.kind`
field:

- **`workspace.kind: local`** — a fixed local source directory, watch
  mode for auto-rebuild on file changes, `set_root_dir(path)` to swap
  between sibling subdirectories without restarting. Best for
  code-review against a checked-out project tree.
- **No `workspace:` block (default github-clone-tracker)** — the agent
  calls `repo_management('org/repo')` to clone repos into the
  workspace; kglite builds a code-tree graph over each; queries flow
  against the active one. Best for exploring open-source codebases on
  demand.

Both share the same source tools (`read_source` / `grep` /
`list_source`), the same `read_code_source` qualified-name lookup,
and the same trust gates for extensions.

## Variant 1 — `workspace.kind: local` + `watch`

Bind a local source directory as the active source root for source
tools (`read_source` / `grep` / `list_source`), build a code-tree
graph over it, and auto-rebuild on file changes. The agent can swap
between sibling project directories with `set_root_dir(path)`
without restarting the server.

## Manifest

```yaml
# code_review_mcp.yaml — under /Volumes/EksternalHome/Koding/
name: Code Review
instructions: |
  Code review against any project under Koding/. The active root
  starts at the directory below; swap with set_root_dir(path).

workspace:
  kind: local
  root: /Volumes/EksternalHome/Koding   # the sandbox boundary
  watch: true                            # auto-rebuild on file changes
```

## What gets registered

`tools/list` returns:

```
- cypher_query
- graph_overview
- read_code_source
- ping
- read_source / grep / list_source
- repo_management              # local-mode also gets this for rebuild dispatch
- set_root_dir                 # local-mode only
- github_issues / github_api   # if GITHUB_TOKEN is in env
```

## The flow

1. Boot: the server canonicalises `/Volumes/.../Koding/` and binds
   it as the active root. `read_source` / `grep` / `list_source`
   sandbox to it.
2. The watcher starts on the root with a 500 ms debounce.
3. First agent action: `cypher_query` against the auto-built
   code-tree graph (modules, functions, calls, imports).
4. Agent narrows interest to a specific file → calls `read_source`.
5. Agent decides to switch to a different project under Koding/:
   `set_root_dir("/Volumes/EksternalHome/Koding/Rust/KGLite")`.
   The active root atomically swaps; source tools rebind; the watch
   handle moves to the new root.
6. On any file change inside the new root, the watcher fires (after
   the 500 ms debounce window), the code-tree rebuilds on a
   background thread, and the new graph atomically swaps in. Queries
   against the previous graph keep working until the swap.

## Sandbox boundary

`workspace.root` is the manifest-declared root. The sandbox check on
`set_root_dir(path)` validates against this immutable boundary, not
the current active root. After `set_root_dir(/.../Rust/KGLite)`,
a subsequent `set_root_dir(/.../MCP servers)` works because both
paths are under `workspace.root`:

```
{"name":"set_root_dir","arguments":{"path":"/Volumes/EksternalHome/Koding/Rust/KGLite"}}
→ "Active root set to /Volumes/EksternalHome/Koding/Rust/KGLite."   ✓

{"name":"set_root_dir","arguments":{"path":"/Volumes/EksternalHome/Koding/MCP servers"}}
→ "Active root set to /Volumes/EksternalHome/Koding/MCP servers."   ✓ (sibling swap, no restart)

{"name":"set_root_dir","arguments":{"path":"/tmp"}}
→ "Error: path '/tmp' escapes the workspace root."                 ✓ (outside the sandbox)
```

## `repo_management` in local mode

In local-workspace mode `repo_management` does not accept a repo
name (there's nothing to clone). It accepts:

```text
{"update": true}                        — re-fingerprint + rebuild
{"update": true, "force_rebuild": true} — rebuild regardless of fingerprint
```

`force_rebuild: true` bypasses the SHA gating that's otherwise on
top of the post-activate hook — useful after upgrading kglite
itself or after a code-tree-builder change.

## github_issues integration

If the project under the active root has a `.git/config` with a
GitHub remote, `github_issues` auto-resolves the `repo_name`
argument from the active root's git config. Agents can call

```json
{"name": "github_issues", "arguments": {"limit": 5, "state": "all"}}
```

without supplying `repo_name`; the active workspace repo is used as
the fallback. The token still has to be in env / walked-up `.env`.

## Performance notes

- Code-tree rebuild is incremental for small files (a single-file
  edit is fast). Whole-tree rebuilds on large projects (>100k LoC)
  can take a few seconds — the debounce + background thread keep
  the agent unblocked during the rebuild.
- Atomic swap means agents never see a half-built graph. Queries
  against the in-flight graph complete; the next query after the
  swap sees the new graph.
- The `.mcp-workspace/` directory inside `workspace.root` stores
  inventory + last-built SHA. Safe to delete — the next boot
  re-seeds it.

## Variant 2 — github-clone-tracker (no `workspace:` block)

Run the server with `--workspace <dir>` (no `workspace.kind: local`
in the manifest) and the agent gets `repo_management` for cloning,
plus the standard source tools against the active clone. A copy-
paste-ready manifest lives at
[`examples/open_source_workspace_mcp.yaml`](https://github.com/kkollsga/kglite/blob/main/examples/open_source_workspace_mcp.yaml).

### Deployment

```bash
mkdir /path/to/my-workspace/
cp examples/open_source_workspace_mcp.yaml /path/to/my-workspace/workspace_mcp.yaml
kglite-mcp-server --workspace /path/to/my-workspace/
```

The filename inside the workspace dir MUST be `workspace_mcp.yaml` —
that's the name the CLI auto-detects when given `--workspace <dir>`.

### What the manifest enables

Looking at the example file inline:

```yaml
name: Open Source Explorer
env_file: ../.env             # walks up to find GITHUB_TOKEN for github_*
builtins:
  temp_cleanup: on_overview
extensions:
  csv_http_server: true       # FORMAT CSV → localhost URL (CORS-enabled)
instructions: |
  Code intelligence for open-source GitHub repositories ...
  FIRST STEP: call repo_management('org/repo') to clone + build ...
overview_prefix: |
  ## Two read paths
  ... (sticky context shown on bare graph_overview())
tools:
  - bundled: repo_management
    description: |              # per-tool guidance lives with the tool
      Manage cloned repositories. FIRST STEP before any other tool.
      Common invocations: ... (full text in the file)
  - bundled: cypher_query
    description: |
      Run a Cypher query against the active repo's knowledge graph.
      Append `FORMAT CSV` for a CSV-exported result over localhost ...
  - bundled: read_source
    description: |
      Read a file slice from the active repo. Prefer
      read_code_source for graph-indexed symbols ...
  - bundled: grep
    description: |
      .gitignore-aware regex search across the active repo's source.
      Reach for cypher first for symbol lookups ...
```

Key choices:

- **`env_file: ../.env`** — manifest paths are manifest-relative, so
  this walks up one level from the workspace dir to find the `.env`.
  Loads `GITHUB_TOKEN` for `github_issues` + `github_api`. Without a
  token those tools don't register at boot.
- **`builtins.temp_cleanup: on_overview`** — wipes the temp/ dir on
  every bare `graph_overview()` call. With `csv_http_server` enabled,
  every `FORMAT CSV` export writes a file to temp/; without cleanup
  they accumulate.
- **`extensions.csv_http_server: true`** — enables the localhost
  listener so `FORMAT CSV` exports return URLs the agent can fetch
  (instead of inlining the CSV body, which blows out for large
  results). Loopback-only, CORS-enabled.
- **`instructions:`** — server-wide first-message orientation. Kept
  short: the project pitch + the FIRST STEP framing. The detailed
  per-tool guidance lives on the tools themselves (see
  `tools[].bundled:` overrides below). Splitting reduces the
  one-time-read context the agent has to retain through a long
  session.
- **`overview_prefix:`** — sticky context prepended to bare
  `graph_overview()` output. Skipped for focused drill-downs
  (`graph_overview(types=...)` etc.) so it doesn't bloat every
  response. Good place for the "two-read-paths" mental model.
- **`tools[].bundled:` overrides** (kglite 0.9.27 / mcp-methods
  0.3.31+) — replace the agent-facing `description:` for specific
  bundled tools. The example overrides four:
  - `repo_management` — its description carries the "FIRST STEP +
    five common invocations" guidance that used to live in the
    global `instructions:` blob. Now it rides `tools/list` next
    to the tool's schema, so the agent sees the guidance every
    time it considers which tool to call.
  - `cypher_query` — description teaches the `FORMAT CSV` →
    localhost URL pattern.
  - `read_source` — disambiguates against `read_code_source`
    (prefer the latter for symbols the graph already indexed) and
    pins `file_path` semantics to the active repo's clone path.
  - `grep` — frames the regex-search workhorse against cypher
    (cypher first for symbols; grep for plain text). Same
    principle as the other three: per-tool guidance attached to
    the tool, not buried in a session-wide blob.

  Override `description` works on any of the 12 bundled tools
  (`cypher_query`, `graph_overview`, `ping`, `read_code_source`,
  `save_graph`, `read_source`, `grep`, `list_source`,
  `repo_management`, `set_root_dir`, `github_issues`,
  `github_api`). Adding `hidden: true` drops a tool from
  `tools/list` AND rejects calls — useful for narrowing the agent
  surface when a bundled tool doesn't fit your deployment.
  Unknown tool names fail at boot with the full valid catalogue
  listed in the error message.

### What gets registered

For `--workspace <dir>` mode (no `workspace.kind: local`):

```
- cypher_query
- graph_overview
- read_code_source
- ping
- repo_management              # github clone tracker
- read_source / grep / list_source   # against the active repo
- github_issues / github_api   # if GITHUB_TOKEN was loaded
```

Note: `set_root_dir` is **only** in the `workspace.kind: local`
variant above. `repo_management` is **only** in github-workspace
mode. The two modes are mutually exclusive.

### Lifecycle

- First call: `repo_management('pydata/xarray')` does a shallow
  `git clone --depth 1`, kglite builds the code-tree graph,
  graph is active for subsequent queries.
- Subsequent calls: `repo_management('pydata/xarray')` again
  short-circuits to "already cloned, activated."
  `repo_management(update=True)` does `git fetch --depth 1
  origin` + reset to remote; graph rebuilds only if the SHA
  moved (gated by `last_built_sha`).
- Idle sweep: repos untouched for `--stale-after-days` (default
  7) get tombstoned in `inventory.json`. The active repo is
  exempt. Sweep happens on the next `repo_management` call.
- Force rebuild: `repo_management(update=True, force_rebuild=True)`
  bypasses the SHA gate — useful after a kglite upgrade.
