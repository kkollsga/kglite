# MCP server: migrating from 0.6.x to 0.9.x

This page is for operators upgrading custom Python MCP servers
written against pre-0.9 kglite (typically 0.6.x – 0.8.x) to the
bundled `kglite-mcp-server` shipped from 0.9.20 onwards. The arc is
large — 0.6 to 0.9 covers a full re-platforming of the MCP surface
— so this guide covers the differences as a single jump rather than
release-by-release.

If you're on 0.9.19 or later and want the smaller hops, see
{doc}`mcp-pre-0.9.20`.

## High-level shift

**Before (0.6.x – 0.8.x):** each MCP server was a hand-written
Python script using the `mcp` SDK directly. The server registered
tools like `cypher_query`, `graph_overview`, `read_source`,
`grep_source`, etc., wired them to a `KnowledgeGraph` instance, and
ran its own watcher / embedder / CSV-export logic. Operators
typically maintained five-plus custom Python servers, each ~500–
1000 LOC.

**After (0.9.20+):** one bundled `kglite-mcp-server` binary (a Python
entry point since 0.9.20) is driven entirely by a YAML manifest. The
operator's custom Python is gone; configuration lives in the
manifest. Five custom Python servers retired across the 0.9.16 →
0.9.27 arc.

## Install

```bash
pip install --upgrade 'kglite[mcp]'
```

The `[mcp]` extras pulls everything the server needs: `mcp`,
`pyyaml`, `fastembed`, `aiohttp`, `watchdog`. There is no separate
`[code-tree]` extras — tree-sitter grammars are bundled into the
Rust extension and load with kglite itself.

After install, `kglite-mcp-server` is on PATH.

## Translation cheat-sheet

If you're staring at a custom Python MCP server and trying to pick
the equivalent flag, this maps the most common patterns:

| Old pattern in your custom script                          | New mode                                |
|------------------------------------------------------------|-----------------------------------------|
| Loads a `.kgl` file at boot, serves it read-mostly         | `--graph path/to/X.kgl`                 |
| Loads a disk-backed graph directory (`storage="disk"`)     | `--graph path/to/dir/` (auto-detected via `disk_graph_meta.json`) |
| Clones GitHub repos at runtime and builds code-trees       | `--workspace path/to/workspace_dir/`    |
| Uses `set_root_dir` to swap between sibling project dirs   | `--workspace path/` + manifest `workspace.kind: local`  |
| Watches a single fixed directory and rebuilds on change    | `--watch path/`                         |
| Exposes only `read_source` / `grep` / `list_source` (no graph) | `--source-root path/`               |
| Reads `.env` walked-up from the manifest                   | `env_file:` field in your manifest (manifest-relative) |
| Spawns its own CSV-over-HTTP listener                      | `extensions.csv_http_server: true` (auto-port since 0.9.29) |
| Wires a custom embedder for `text_score`                   | `extensions.embedder.backend: fastembed` (or pre-compute via `g.embed_texts(...)` at build time) |

A complete custom server typically maps onto one mode + a manifest
with one or two `extensions:` entries. The "extra Python" you used
to hand-write — listener setup, embedder wiring, watcher debounce,
CSV write paths — all moves into manifest declarations.

## .kgl format compatibility

`.kgl` files written by 0.6.x – 0.8.x **cannot be loaded by 0.9.x**.
The serialisation format changed across the storage-backend
refactor (DirGraph traits, disk-backed mode, mapped-mode columnar
layout) and graphs need to be rebuilt under 0.9.x. The error you
get loading an old file is explicit ("unsupported format version"
or similar) rather than silent corruption — that part works as
intended.

If your build pipeline lives in your own scripts, re-running them
under the new kglite produces a fresh 0.9.x-compatible `.kgl`. If
you don't have the build pipeline anymore, fluent-API or Cypher
`CREATE` against an empty `KnowledgeGraph()` is the supported way
to rebuild from source data — see the
[Graph construction guide](https://kglite.readthedocs.io/en/latest/guides/graph-construction.html).

## Five operating modes

The bundled server picks its mode from CLI flags (and `workspace.kind`
when a manifest is present):

| Flag                      | Mode kind          | What it does                                                                 |
|---------------------------|--------------------|------------------------------------------------------------------------------|
| `--graph X.kgl` or dir    | `graph`            | Static graph from a `.kgl` file or disk-backed graph directory.              |
| `--workspace DIR`         | `workspace`        | Github-clone-tracker: agent calls `repo_management('org/repo')` to clone.    |
| `--workspace DIR` + manifest `workspace.kind: local` | `local_workspace` | Fixed local source root with sibling-swap via `set_root_dir`; auto-rebuild on file changes if `watch: true`. |
| `--watch DIR`             | `watch`            | Single directory, auto-rebuild on file changes. Like `local_workspace` but no swap-between-siblings.        |
| `--source-root DIR`       | `source_root`      | Source-tools-only binding (read-only `read_source` / `grep` / `list_source`); no graph.                     |

Manifest auto-detection: if `--graph X.kgl` is passed and
`X_mcp.yaml` sits next to it, the manifest loads automatically. For
`--workspace DIR`, the server looks for `workspace_mcp.yaml` first
inside the directory, then (since 0.9.29 / mcp-methods 0.3.33) one
level up — but only when the parent manifest opts in via
`workspace.applies_to`. This lets operators use either of these
layouts:

```text
# Layout 1: manifest inside the workspace dir (no opt-in needed)
workspace_dir/
├── workspace_mcp.yaml
└── repos/                  # clones land here

# Layout 2: manifest as a sibling (since 0.9.29)
my_project/
├── workspace_mcp.yaml      # declares `workspace.applies_to: ./*`
└── repos/                  # --workspace points here
```

For layout 2, the parent manifest must declare which child directory
(or directories) it covers:

```yaml
workspace:
  kind: github
  applies_to: ./*            # any direct child (most permissive)
  # or:
  applies_to: ./repos        # literal: only "repos" matches
  # or:
  applies_to: ./prod-*       # glob: matches "prod-api", "prod-web", etc.
  # or:
  applies_to: [./repos, ./clones]   # list form
```

Without the `applies_to` opt-in, the parent-walk discovery is
refused — a deliberate safety property to prevent silent-wrong-
manifest if the operator points `--workspace` at any unrelated
sibling directory under a workspace-manifest parent. The framework
emits a `tracing` log when it considers and rejects a parent-walk
match (visible if you set `RUST_LOG=mcp_methods=debug`).

Pattern syntax follows the `globset` crate (`*`, `?`, `[abc]`).
Multi-segment paths and `..` are rejected at manifest parse time
with explicit error messages.

## Tool surface differences from pre-0.9.x

| Pre-0.9.x tool name             | 0.9.20+ tool name(s)                                  | Notes |
|---------------------------------|--------------------------------------------------------|-------|
| `read_source` (file + symbol)   | `read_source` (file slice) + `read_code_source` (qualified-name lookup) | Split into two tools. `read_source` reads by `file_path` + optional line range + grep filter. `read_code_source` resolves a `qualified_name` against the graph's `Function` / `Class` / `Method` nodes and returns the source slice with `file_path:line` headers. |
| `grep_source`                   | `grep`                                                 | Renamed; arguments unchanged (`pattern`, `glob`, `context`, `max_results`, `case_insensitive`). .gitignore-aware. |
| `list_source`                   | `list_source`                                          | Same name, same arguments. |
| `ripgrep` (custom servers)      | `grep`                                                 | `ripgrep` is not in the bundled catalogue. `grep` is the regex-search workhorse (powered by ripgrep crates under the hood). If your custom binary registers a tool named `ripgrep`, the bundled server's `tools[].bundled:` override mechanism won't bind to it — they're distinct identifiers. |
| `cypher_query`                  | `cypher_query`                                         | Same name. Now supports `FORMAT CSV` suffix returning a localhost URL when `extensions.csv_http_server: true` is set in the manifest. |
| `graph_overview`                | `graph_overview`                                       | Same name. `connections=true` for connection details, `types=['Function']` to drill in. |
| `save_graph`                    | `save_graph` (gated on `builtins.save_graph: true`)    | Off by default — operators must opt in via manifest. Pre-0.9.x always exposed it. |
| `repo_management`               | `repo_management` (workspace + local-workspace modes)  | Same surface: name / delete / update / force_rebuild. The bundled version uses the validated mcp-methods Rust path (clone-and-track, inventory, atomic root swap). |
| `set_root_dir`                  | `set_root_dir` (local-workspace mode only)             | Same surface. Sibling swap under the workspace root sandbox. |
| (custom github wrapper)         | `github_issues` / `github_api`                         | Built-in; auto-registers when `GITHUB_TOKEN` (or `GH_TOKEN`) is in env. |

## Embedder transition

Pre-0.9.x custom servers typically loaded `sentence-transformers`
directly, with torch + MPS for GPU acceleration on Apple Silicon:

```python
# Pre-0.9.x pattern
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("BAAI/bge-m3", device="mps")
class CustomEmbedder:
    def embed(self, texts: list[str]) -> list[list[float]]:
        return model.encode(texts).tolist()
```

0.9.20+ uses `fastembed` (ONNX runtime) as the default backend:

```yaml
# 0.9.20+ pattern — declared in the manifest
extensions:
  embedder:
    backend: fastembed
    model: BAAI/bge-m3
    cooldown: 1200   # idle-evict after 20 min, reload on next use
```

The cache directory is `~/.cache/fastembed/` (same as pre-0.9.x
direct fastembed installs). First call downloads the ONNX weights;
subsequent calls reuse the cached copy.

**Performance note**: fastembed's ONNX runtime is CPU-bound on
default install (no `coreml`/`mps` execution provider). On Apple
Silicon this is a measurable change from torch+MPS in your custom
server — bge-m3 inference goes from ~50 ms/query (MPS) to ~200–400
ms/query (CPU). If query-time embedding is on your hot path, decide
whether to:

1. Accept the CPU latency (simpler ops; reasonable for low-volume
   exploration servers).
2. Pre-compute embeddings via `g.embed_texts("Type", "prop")` and
   store them on the graph — then `text_score` is a vector lookup,
   not an inference call.
3. Keep your custom embedder. Pass a Python class instance
   implementing the embedder duck-type through `PyEmbedderAdapter`
   when constructing the server programmatically. (Not currently
   wired into the manifest schema — file an issue if you need it
   surfaced via YAML.)

For most operators, option 2 is the right answer: pre-compute via
`g.embed_texts(...)` at graph-build time, save with `g.save(...)`,
and `text_score` at query time is microseconds rather than hundreds
of milliseconds.

## Manifest cheat-sheet

### `--graph` mode (static graph)

`prospect_mcp.yaml` next to `prospect.kgl`:

```yaml
name: Prospect
instructions: |
  Code intelligence for the prospect project. Use cypher_query first
  to locate entities, then read_code_source(qualified_name=...) for
  the slice.
overview_prefix: |
  ## Two read paths
  - read_code_source for graph-indexed symbols.
  - read_source for files without graph nodes.
```

### `--workspace` (github clone-tracker)

`workspace_mcp.yaml` inside the workspace directory. Full example:
[`examples/open_source_workspace_mcp.yaml`](https://github.com/kkollsga/kglite/blob/main/examples/open_source_workspace_mcp.yaml).

### `workspace.kind: local`

```yaml
name: Code Review
workspace:
  kind: local
  root: /Volumes/.../Koding   # sandbox boundary
  watch: true                 # auto-rebuild on file changes
```

### `--watch` (single dir)

No manifest required for the simplest case:

```bash
kglite-mcp-server --watch /path/to/project
```

Pair with a sibling `project_mcp.yaml` for custom instructions /
cypher tools / extensions.

## Tool overrides (0.9.27+)

The `tools[].bundled:` block (added in 0.9.27 / mcp-methods 0.3.31)
lets you customise the agent-facing surface of bundled tools without
declaring them inline. Useful for replacing the boilerplate
guidance pre-0.9.x servers carried in the global `instructions:` blob:

```yaml
tools:
  - bundled: repo_management
    description: |
      FIRST STEP: call repo_management('org/repo') to clone.
      ... (the call patterns you used to teach in instructions)

  - bundled: ping
    hidden: true                # drop from tools/list + reject calls
```

The 12 bundled tools: `cypher_query`, `graph_overview`, `ping`,
`read_code_source`, `save_graph`, `read_source`, `grep`,
`list_source`, `repo_management`, `set_root_dir`, `github_issues`,
`github_api`. Unknown names fail at boot with the valid catalogue
listed in stderr.

## Common migration gotchas

- **`workspace` mode didn't build graphs before 0.9.28.** Pre-0.9.28
  `repo_management('org/repo')` would clone the repo but the
  workspace's `post_activate` hook was never wired in `server.py`,
  so no code-tree graph was built. Fixed in 0.9.28: the hook now
  fires on every activate and rebuilds the graph + rebinds
  `source_roots`. If you tried `--workspace` before 0.9.28 and gave
  up, retry with 0.9.28+.
- **`kglite.code_tree` attribute access was broken before 0.9.28**
  in the mcp_server path — `kglite/__init__.py` doesn't import the
  submodule at module load. The fix is to use `from kglite import
  code_tree` instead of `kglite.code_tree.X` (the mcp_server's
  `build_code_tree` does this now). External callers writing custom
  embedders or graph builders should follow the same pattern.
- **`save_graph` is opt-in.** Pre-0.9.x servers always exposed it;
  the bundled server gates on `builtins.save_graph: true`. Without
  that flag the tool doesn't register at all.
- **Manifest paths are manifest-relative.** `env_file: ../.env`
  walks one directory up from the manifest, NOT from the CWD where
  the server was launched. This bit operators expecting CWD-relative
  resolution.
- **Auto-detected manifests.** `kglite-mcp-server --graph X.kgl`
  looks for `X_mcp.yaml` next to the .kgl. `kglite-mcp-server
  --workspace DIR` looks for `workspace_mcp.yaml` inside the
  workspace dir. If you'd prefer not to auto-load, pass
  `--no-config` (always wins over auto-detection).

## Reference

- {doc}`../guides/mcp-servers` — end-to-end setup guide for new
  servers (post-migration default).
- {doc}`../examples/manifest_workspace` — workspace + watch mode
  walkthroughs with the operator-modelled `open_source_workspace_mcp.yaml`.
- [CHANGELOG.md](https://github.com/kkollsga/kglite/blob/main/CHANGELOG.md)
  — full release-by-release notes if you need to verify exactly
  which version introduced a specific change.
