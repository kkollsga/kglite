# Changelog

All notable changes to KGLite will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.9.34] — 2026-05-17

Code-graph expansion release: closes the feature gap with
colbymchenry/codegraph in six commits — File→File IMPORTS edges,
the `affected_tests` Cypher procedure, DECORATES edges, web-framework
route extraction (Flask/FastAPI/Django), the `explore()` one-call
codebase tool (pymethod + MCP), and a minimal Swift parser.

Release-mode benchmark snapshot on a synthetic Flask-shaped package
(saved as `release_0934` under `.benchmarks/`):

| Surface | Median |
|---|---|
| `CALL affected_tests` | 2.3 µs |
| `MATCH (Function)-[:DECORATES]->(Function)` | 6.0 µs |
| `MATCH (Route)-[:HANDLES]->(Function)` | 8.8 µs |
| `kg.explore(query)` with source slicing | 17.4 µs |
| `code_tree.build()` end-to-end | 600 µs |

Core in-memory benchmarks (`tests/benchmarks/test_bench_core.py`)
were re-run release-mode against 0.9.33 and show no unambiguous
regressions in code paths we changed. `add_nodes` / `add_connections`
/ `save_v3` all measured within noise of baseline; the Cypher core
medians shifted on some queries but well within 1 StdDev of the
baseline. Future regressions in any of these surfaces will surface
via `make bench-compare` against the saved `release_0934` baseline.

### Added (code_tree — Swift)

- **Swift parser.** New `src/code_tree/parsers/swift.rs` via
  `tree-sitter-swift = "0.7.2"`. Coverage in 0.9.34:

  - `class` / `struct` / `actor` / `enum` declarations — emitted as
    Class/Struct nodes with kind tagged from the grammar's
    `declaration_kind` field (Swift's grammar collapses all five
    into one AST node).
  - `protocol` declarations → Interface nodes with `kind="protocol"`.
  - Top-level and method `func` declarations → Function nodes with
    HAS_METHOD edges from their owning type.
  - `import Foundation` → FileInfo.imports → File→Module IMPORTS.
  - Visibility (`public` / `internal` / `fileprivate` / `private`).
  - CALLS edges resolve via the existing 5-tier name resolver.

  Follow-up scope (separate PRs): `extension` IMPLEMENTS edges,
  `init` / `subscript` / computed properties, `@objc` / `@MainActor`
  attributes as decorators, `async` / `throws` flags. The parser
  structure leaves clean slots for each.

### Added (code_tree)

- **File → File IMPORTS edges.** Sibling to the existing File → Module
  IMPORTS, resolved via a `module_path → file_path` reverse index using
  the same longest-prefix walk as the module resolver. Multiple imports
  from one source to the same target collapse into a single edge whose
  `import_count` property records the multiplicity. Enables direct
  file-level impact analysis in one Cypher hop —
  `MATCH (changed:File {path: 'src/foo.py'})<-[:IMPORTS*1..]-(impacted:File)`
  — without joining through Module nodes.

### Added (explore)

- **`KnowledgeGraph.explore(query)` and matching `explore` MCP tool.**
  One-call codebase exploration over a code-tree graph: lexically
  ranks Function/Class/Interface/Struct/Trait/Protocol/Enum nodes
  against a free-text query (name > signature > docstring), takes
  the top `max_entities`, 2-hop traverses CALLS / USES_TYPE /
  HAS_METHOD / DEFINES / REFERENCES_FN, and returns a markdown
  report with entry points, a relationship-map neighborhood, and
  grouped source slices for the entry points. Designed for the
  "how does X work in this codebase" Explore-agent question that
  would otherwise turn into chained grep + read calls — closes the
  feature gap with colbymchenry/codegraph's `codegraph_explore` tool
  while composing existing primitives rather than building a parallel
  index.

  Pyfunction signature:
  ```python
  kg.explore(query, max_entities=10, max_depth=2,
             include_source=True, source_roots=None)
  ```
  MCP tool ships with bundled methodology
  (`kglite/mcp_server/skills/explore.md`) gated on
  `graph_has_node_type: [Function, Class]` so it only activates on
  code-tree graphs.

### Added (code_tree — routes)

- **Web-framework Route extraction.** New `Route` node type plus
  `Route -[HANDLES]-> Function` edges, synthesized from decorators and
  `urlpatterns` constants. Three frameworks in v1: **Flask** (`@app.route`,
  `@app.get`/`@app.post`/..., blueprints), **FastAPI** (`@router.get`,
  `@app.get`, all HTTP verbs), and **Django** (`urlpatterns = [path('x/', view)]`
  in any urls.py-shaped file). `@app.route` decorators with `methods=[...]`
  fan out to one Route per method so `WHERE r.method = 'DELETE'` queries
  match correctly. Express, Axum, Rails, Spring, Laravel land as follow-up
  PRs — they need parser-side capture of call arguments which the
  parser model doesn't preserve today; the per-framework module layout
  under `builder/routes/` makes each subsequent framework a single-file
  addition.

  ```cypher
  MATCH (r:Route)-[:HANDLES]->(f:Function)
  WHERE r.framework = 'fastapi' AND r.method = 'POST'
  RETURN r.path, f.qualified_name
  ```

- **`urlpatterns` is now extracted as a top-level constant on Python
  files.** The constant-extraction filter previously required
  SCREAMING_SNAKE_CASE; a narrow framework-allowlist now also lets
  Django's lowercase `urlpatterns` through so the route extractor can
  read the list literal. Other lowercase names remain filtered out.

### Added (code_tree — DECORATES)

- **Function → Function DECORATES edges.** The Python/TS/Java/C# parsers
  already extracted `FunctionInfo.decorators` as raw strings; a new
  builder pass now resolves each to a target Function via the same
  bare-name lookup CALLS uses. Strips call-args (`@app.route('/x')` →
  `app.route`) and the namespace prefix (`functools.wraps` → `wraps`).
  Edge property `decorator_name` preserves the original literal so
  downstream queries don't have to re-parse the Function.decorators
  property. Unresolved (third-party) and ambiguous decorators are
  silently dropped — same stance as the call-edge resolver.

  ```cypher
  MATCH (d:Function)-[:DECORATES]->(f:Function)
  WHERE d.name = 'cache' RETURN f.qualified_name
  ```

### Added (Cypher)

- **`CALL affected_tests({files: [...], max_depth?}) YIELD test_file, depth`**
  — given a seed set of changed file paths, BFS over inbound IMPORTS
  edges and yield reachable File nodes whose `is_test` property is
  true. Either yield column is optional individually (the common case
  is `YIELD test_file` to get just a list of paths). Builds directly
  on the 0.9.34 File → File IMPORTS edges; closes parity with the
  `codegraph affected` CLI feature from colbymchenry/codegraph but as
  pure Cypher rather than a separate command.

  ```cypher
  CALL affected_tests({files: ['src/utils.rs', 'src/api.rs']})
  YIELD test_file
  RETURN test_file ORDER BY test_file
  ```

## [0.9.33] — 2026-05-14

mcp-methods 0.3.37 adopted both operator-reported fixes from the
0.9.31 deployment audit, including kglite's stop-gap full-body
skill-inject shape verbatim as the framework canonical. This is a
pin-bump release: the Rust binary path picks up the canonical
behaviour automatically; the Python entry's `_apply_skill_hint`
stays — same shape, no longer "stop-gap", now aligned with the
framework's `serve_prompts` auto-inject pass.

Folds in the unpublished 0.9.32 work (overview_prefix plumbing +
auto-inject full-body) so the cumulative diff against 0.9.31 is
one coherent release rather than two adjacent versions on PyPI.

### Changed

- **mcp-methods pin bumped to 0.3.37** (was 0.3.36). Picks up:
  - The framework's `serve_prompts` auto-inject pass now embeds
    the full skill body under a `## Methodology` header instead
    of the dangling `[See prompts/get NAME ...]` pointer.
    Operators running the standalone Rust binary
    (`crates/kglite-mcp-server/`) get the canonical behaviour
    without any kglite-side change.
  - `ResolvedRegistry.parse_warnings()` Rust getter and
    `SkillRegistry.parse_warnings` Python getter for the
    silent-skill-drop visibility (mcp-methods bug 1). The
    framework's `tracing::warn!` channel also continues to
    fire. Operator-visible boot summary integration in the
    Python entry queued for 0.9.34 (the PyPI publish of the
    0.3.37 wheel is still in flight at release-cut time).

### Fixed (MCP server)

These fixes were drafted as 0.9.32 commits but folded into 0.9.33
so a single coherent cut publishes to PyPI:

- **`overview_prefix:` from the manifest is now prepended to bare
  `graph_overview()` output.** Pre-0.9.32 the field was parsed by
  `manifest.py` but never read by `tools.py::run_overview` in the
  Python entry path. The FastMCP path at
  `mcp_methods/fastmcp/_overview.py` had honoured it correctly;
  kglite's Python entry didn't. Operators authoring documented
  `overview_prefix:` blocks were getting silently-dropped content.
  `run_overview` now accepts an optional `overview_prefix` keyword,
  prepended only on bare-overview calls (no `types=...` /
  `connections=...` / `cypher=...` drill-down args), matching the
  framework's behaviour and the documented contract.
- **`_apply_skill_hint` injects the full skill body, not a dangling
  `prompts/get` pointer.** Operator empirically confirmed that
  agents in Claude Code, Claude Desktop, Cursor, and Continue
  don't expose `prompts/get` to the model — the MCP `prompts/*`
  plane was designed for human slash commands in chat UIs, not
  agentic retrieval. The pre-0.9.33 bracketed pointer
  (`[See prompts/get NAME for full methodology.]`) was a dangling
  reference in those clients. 0.9.33 embeds the skill body under
  a `## Methodology` header in the matching tool's description so
  it reaches the agent via `tools/list`, which every MCP client
  exposes. Capped at the framework's 16 KB hard limit / 4 KB soft
  target. Operators can still set `auto_inject_hint: false`
  per-skill to suppress the embed.
- **`mcp-methods` Python wheel added to `[mcp]` extras** (`pyproject.toml`).
  Without the framework's Python wheel, `SkillRegistry.from_manifest`
  is unavailable and `skills_loader.py::load_framework_skills`
  silently returns the empty list — project-layer
  `<basename>.skills/` overrides and operator-declared domain packs
  silently don't load. CI surfaced the gap when test_o3 (the
  `auto_inject_hint: false` escape-hatch test) failed against an
  environment without the wheel manually installed.

### Added (regression tests)

- **O1**: `overview_prefix:` is prepended to bare `graph_overview()`.
- **O2**: `overview_prefix:` is NOT prepended to drill-down calls
  (`types=[...]` etc.).
- **O3**: `auto_inject_hint: false` per-skill suppresses the body
  embed in the matching tool's description.
- **O4**: Re-calling `list_tools` doesn't double-inject (the
  `## Methodology` header is the idempotency marker).
- **SK4** updated to assert the full-body embed semantics.

85/85 mcp Python tests green (was 81 in 0.9.31).

### Sequencing notes

The mcp-methods 0.3.37 wheel publish to PyPI is still in flight at
release-cut time. CI's `pip install kglite[mcp]` resolves
`mcp-methods>=0.3.36` against PyPI's current state — once 0.3.37
lands there, `pip install --upgrade` on kglite will pull it
through. The Cargo dep already resolves to 0.3.37 on crates.io, so
the Rust binary path (and the wheel's bundled Rust extension) get
the canonical inject behaviour today.

### Yanked

The 0.9.32 git commits (`50a41c6...54bd095`) describe the same
fixes that ship in 0.9.33; 0.9.32 was never published to PyPI.
Its CI workflow was cancelled in favour of the single 0.9.33 cut
so PyPI's release history stays clean. Operators who pulled
0.9.32 wheels from the in-flight build artifacts (if any) should
upgrade directly to 0.9.33.

## [0.9.32] — 2026-05-14 — unpublished

Reserved version: the `Cargo.toml` line was bumped to `0.9.32`
mid-day on 2026-05-14 in commits `50a41c6` and `54bd095`. CI was
cancelled mid-flight in favour of a single 0.9.33 cut bundling
the same fixes plus the 0.3.37 framework pin. No 0.9.32 wheel
exists on PyPI. See 0.9.33 above for the actual operator-facing
changes.

## [0.9.31] — 2026-05-14

Two same-day operator bug reports from the 0.9.31 deployment.
Both confirmed root-cause; one is kglite-side and shipped here,
the other is a framework-design issue forwarded to mcp-methods
with our reading of the trade-offs.

### Fixed (MCP server)

- **`overview_prefix:` from the manifest is now prepended to bare
  `graph_overview()` output.** Pre-0.9.32 the field was parsed by
  `manifest.py` but never read by `tools.py::run_overview` in the
  Python entry path. The FastMCP path at
  `mcp_methods/fastmcp/_overview.py` had honoured it correctly;
  kglite's Python entry didn't. Operators authoring documented
  `overview_prefix:` blocks were getting silently-dropped content.
  `run_overview` now accepts an optional `overview_prefix` keyword,
  prepended only on bare-overview calls (no `types=...` /
  `connections=...` / `cypher=...` drill-down args), matching the
  framework's behaviour and the documented contract.
- **`_apply_skill_hint` injects the full skill body, not a dangling
  `prompts/get` pointer.** Operator empirically confirmed that
  agents in Claude Code, Claude Desktop, Cursor, and Continue
  don't expose `prompts/get` to the model — the MCP `prompts/*`
  plane was designed for human slash commands in chat UIs, not
  agentic retrieval. The pre-0.9.32 bracketed pointer
  (`[See prompts/get NAME for full methodology.]`) was a dangling
  reference in those clients. 0.9.32 embeds the skill body under
  a `## Methodology` header in the matching tool's description so
  it reaches the agent via `tools/list`, which every MCP client
  exposes. Capped at the framework's 16 KB hard limit / 4 KB soft
  target. Operators can still set `auto_inject_hint: false`
  per-skill to suppress the embed (useful for clients that DO
  expose prompts/get, or where context cost matters more than
  reachability).

  This is a kglite Python-entry stop-gap; the framework's
  canonical fix is being discussed with the mcp-methods maintainer
  (their `serve_prompts` auto-inject pass would benefit from the
  same upgrade for the Rust binary path + every other framework
  consumer). When that lands, this kglite-side implementation
  will converge.

### Added (regression tests)

- **O1**: `overview_prefix:` is prepended to bare `graph_overview()`.
- **O2**: `overview_prefix:` is NOT prepended to drill-down calls
  (`types=[...]` etc.).
- **O3**: `auto_inject_hint: false` per-skill suppresses the body
  embed in the matching tool's description.
- **O4**: Re-calling `list_tools` doesn't double-inject (the
  `## Methodology` header is the idempotency marker).
- **SK4** updated to assert the full-body embed semantics
  (previously asserted the dangling-pointer shape).

85/85 mcp Python tests green (was 81 in 0.9.31).

### Acknowledged but not yet fixed (forwarded to mcp-methods)

- **`SkillRegistry.from_manifest` silently drops files with YAML
  frontmatter parse errors.** Operator hit this with a
  colon-in-value in an unquoted `description:` field; spent 25
  minutes debugging because no log line surfaces. Framework owns
  the parser; mcp-methods inbox has the bug report with operator-
  ranked fixes (log.warning on each skipped file; return
  parse_warnings on the registry; scaffold helper hints).
- **Skills via `prompts/get` are unreachable in real MCP clients.**
  Forwarded as a framework-wide design issue with the operator's
  ranked fixes (auto-inject full body — adopted here as a
  stop-gap; expose `get_skill` as a tool; document the
  limitation). Maintainer decides the canonical shape; we'll
  converge once they ship.

## [0.9.31] — 2026-05-14

Skills-aware MCP. Ships kglite-authored methodology for the four
custom tools (`cypher_query`, `graph_overview`, `save_graph`,
`read_code_source`) plus the wiring to compose them with framework
defaults + operator-side layers + predicate-gated filtering. Opt-in
per manifest via `skills: true` (or a path list); existing
deployments without that declaration see no behavioural change.

### Added (MCP server)

- **Four bundled skills** under `kglite/mcp_server/skills/`,
  authored against mcp-methods 0.3.35's `writing-effective-skills.md`
  guide (TRIGGER/SKIP descriptions, Overview → Quick Reference →
  Common Pitfalls → When wrong body anatomy, ~150-220 lines each).
  Shipped as Python package data; `include_str!`'d into the Rust
  binary at `crates/kglite-mcp-server/src/main.rs`. One source of
  truth across both shipping paths.
- **`SkillRegistry` wiring in the standalone Rust binary**
  (`crates/kglite-mcp-server/src/main.rs`): `add_bundled` for each
  of the four kglite skills, `merge_framework_defaults`,
  `auto_detect_project_layer`, `layer_dirs(manifest.skills)`,
  predicate evaluator (`KglitePredicateEvaluator` consults
  `graph_state.has_node_type` / `has_property` for the
  `graph_has_node_type:` / `graph_has_property:` clauses),
  `finalise`. Wired into `serve_prompts(&registry, &mut server)`
  before the stdio loop.
- **Python entry point prompts handlers** at
  `kglite/mcp_server/server.py`. The lowlevel `mcp.server.Server`
  surface doesn't have the framework's FastMCP-shaped
  `register_skills_as_prompts` helper, so we hand-roll
  `@server.list_prompts()` and `@server.get_prompt()` backed by
  `skills_loader.build_active_skill_set(...)`. Predicate gating is
  re-evaluated at request time so post-boot graph state changes
  (workspace activation, watch rebuild) reflect immediately.
- **`kglite/mcp_server/skills_loader.py`** — minimal frontmatter
  parser, `Skill`/`AppliesWhen` dataclasses, three-layer merge
  (kglite-bundled + framework + operator), and runtime
  `applies_when:` evaluation. Lives entirely in Python; talks to
  the framework via `mcp_methods.SkillRegistry.from_manifest(...)`
  for the framework+operator layers.
- **Auto-inject hint pass.** When a skill's `name` matches a
  registered tool and `auto_inject_hint: true` (default), the tool's
  `description` gains a `[See prompts/get <name> for full
  methodology.]` pointer in `tools/list`. Agents that scan tools
  first still discover the methodology surface.
- **`Manifest.skills` field** on the Python dataclass
  (`kglite/mcp_server/manifest.py`) parsed from the framework's
  polymorphic JSON shape (`false` / array of `true`/path entries).

### Changed

- **mcp-methods pin bumped to 0.3.36** (was 0.3.34). Picks up
  `applies_when:` predicate gating (0.3.36) and skills foundation
  (0.3.35). Both shipped by the maintainer in response to our
  design feedback within the same day; no design changes during
  review.

### Added (regression tests)

- **SK1**: Manifest without `skills:` exposes no prompts (opt-in
  property; pre-0.9.31 behaviour preserved by default).
- **SK2**: `skills: true` exposes kglite-bundled skills via
  `prompts/list`. read_code_source filtered out via
  applies_when on non-code graph fixtures.
- **SK3**: `prompts/get cypher_query` returns the bundled
  markdown body with description.
- **SK4**: Auto-inject hint appends `[See prompts/get ...]` to
  matching tool descriptions when skills are enabled.
- **SK5**: `read_code_source` skill ACTIVE on a code-tree graph
  (Function/Class present); proves the predicate evaluator
  consults live graph state.
- **SK6**: `prompts/get` with unknown name returns a clean
  JSON-RPC error.

81/81 mcp Python tests green (was 75 in 0.9.30). Standalone Rust
binary builds clean; `cargo fmt --check` + `cargo clippy -- -D
warnings` clean across the workspace.

### Example

`examples/open_source_workspace_mcp.yaml` opts in with `skills:
true` and carries explanatory comments covering the three value
shapes (`true` / single path / list form) plus the `applies_when:`
predicate behaviour for the legal / o&g / code deployment shape.

## [0.9.30] — 2026-05-14

Operator-reported friction from the 0.9.29 deployment audit:
agent-facing schema clarity (Item 2), MCP tool-search round-trip
counts (Item 1), and identical-tool-surface ambiguity across
multiple kglite servers (Item 3). All four items fixed in one
release.

### Fixed (code_tree)

- **`module` property is now populated on every code-tree entity
  type, not just File and Module.** Operator reported
  `MATCH (f:Function) WHERE f.module STARTS WITH 'xarray.core'
  RETURN f` returned zero rows — the property only existed on
  File/Module nodes. The code_tree builder now looks up each
  Function/Class/Constant/Enum/Interface/Trait/Protocol/Struct's
  file_path in a file → module_path map and populates a `module`
  property derived from the parent file's module. Module nodes
  also get a `module` alias of their `qualified_name` for cross-
  type uniformity. Result: `WHERE n.module STARTS WITH '...'`
  works against any node label without branching.

### Added (introspection)

- **`<prop sample="..." />` attribute on high-cardinality
  properties in `describe()` / `graph_overview` output.**
  Pre-0.9.30 the schema XML showed `vals="..."` for properties
  with ≤15 unique values (low-cardinality enums); high-
  cardinality properties (docstring, signature, file_path with
  hundreds of values) showed only `unique=N` with no example.
  Now one example value is emitted as a `sample="..."` attribute
  whenever `vals=` would be omitted, so the agent always sees
  what the property *looks like* (e.g. `signature` becomes
  `sample="def to_list (self) -> list[dict[str, ..."` instead
  of just `unique="16"`). Same logic applied to edge property
  stats in `<connections>` blocks.

### Added (MCP server)

- **Auto-injected ToolSearch batch-load hint into the server's
  `instructions:` field.** Operator reported deferred-tool loaders
  (Claude Code's ToolSearch) gating each tool family per-tool,
  forcing N round trips to load N tools from one server. The
  server now prepends a one-paragraph hint to operator-declared
  `instructions:` describing the
  `ToolSearch(query='+<server-slug>', max_results=20)` batch-load
  pattern (one round trip per server). Idempotent: composing
  twice does not duplicate the hint. Operators who want full
  control can include the literal marker `[kglite-batch-load-hint]`
  in their instructions text to suppress auto-injection.
- **`tools[].bundled: <name>` overrides accept a `rename:` field
  (mcp-methods 0.3.34+).** Operator reported that running three
  kglite servers exposing identical bundled surfaces produced
  six near-identical entries in ToolSearch results, ambiguous
  to rank. `rename:` lets operators expose a bundled tool under
  a per-deployment name (e.g. `legal_cypher_query`,
  `prospect_cypher_query`) while the canonical handler still
  runs the body. Composes with the existing `description:` and
  `hidden:` overrides. Boot-time validation refuses renames
  that shadow another bundled tool, another rename, or a
  manifest-declared cypher tool.

### Changed

- **mcp-methods pin bumped to 0.3.34** (was 0.3.33). Picks up
  the `tools[].bundled: rename:` extension. The framework
  patch was applied locally to mcp-methods, tested (16/16
  bundled tests green; +6 new entries covering rename
  validation and JSON shape), and proposed to the maintainer
  via inbox note.

### Added (regression tests)

- **B14**: `bundled: cypher_query` + `rename: legal_cypher_query`
  exposes the renamed identifier in `tools/list` and removes the
  canonical name from the listing.
- **B15**: Call to renamed tool dispatches through the canonical
  handler (proves rename isn't visible-only).
- **B16**: Renaming a bundled tool to a name that shadows
  another bundled tool fails at boot with a clear collision
  error.
- **B17**: Renaming to a name that shadows a cypher tool fails
  at boot.
- **I1 / I2 / I3**: Batch-load hint appears in composed
  instructions, idempotent across repeated composition,
  suppressible by operator-supplied marker.
- **S1**: `module` property is populated on Function / Class /
  Constant / Module / File nodes uniformly (operator's literal
  reproducer).
- **S2**: `describe()` emits `sample="..."` for high-cardinality
  properties.

75/75 mcp tests green (was 66 in 0.9.29). 603/603 Rust unit
tests green.

## [0.9.29] — 2026-05-14

Two operator-reported fixes from the post-0.9.28 deployment audit:
a hardcoded port-collision default that made parallel server boot
impossible, and a workspace manifest layout that forced explicit
`--mcp-config` for the natural folder shape. The second is fixed
upstream in mcp-methods 0.3.33 via an opt-in `workspace.applies_to`
declaration — kglite 0.9.29 bumps the pin to pick it up.

### Fixed (MCP server)

- **`csv_http_server` default port changed from 8765 to 0
  (OS-assigned).** When Claude Desktop launches multiple
  kglite-mcp-servers concurrently at startup, the first server
  to boot used to grab port 8765 and every subsequent server
  crashed with `OSError: address already in use` — surfaced to
  the user as "Server disconnected" with no actionable detail.
  The default now binds to port 0 (kernel-assigned); the actual
  bound port is captured back into the config so `url_for()`
  produces correct URLs (via `runner.addresses`, aiohttp's
  public API for this). Operators who need a stable port for
  external integrations can still set `port: 9000` explicitly.
- **Workspace manifest auto-discovery walks one level up when
  the parent manifest opts in via `workspace.applies_to`.**
  Operators with the natural layout

  ```text
  open_source/
  ├── workspace_mcp.yaml      # declares `workspace.applies_to: ./*`
  └── repos/                  # --workspace points here
  ```

  no longer have to pass `--mcp-config` explicitly. The opt-in
  declaration accepts a literal name (`./repos`), a glob pattern
  (`./prod-*`), or a list of patterns (`[./repos, ./clones]`).
  Patterns match the workspace dir's basename; the parent walk
  is bounded to one level. Without `applies_to` declared, the
  parent-walk is refused — a deliberate safety property to
  prevent silent-wrong-manifest if `--workspace` points at any
  unrelated sibling under a workspace-manifest parent.

  Implementation lives upstream in mcp-methods 0.3.33's
  `server::manifest::find_workspace_manifest`. kglite's Python
  wrapper is now a thin pass-through; the unconditional
  parent-walk fallback we briefly considered for the wrapper
  was abandoned after the maintainer flagged its silent-wrong-
  manifest failure mode.

### Changed

- **mcp-methods pin bumped to 0.3.33** (was 0.3.31). Picks up
  the `workspace.applies_to` opt-in (above) plus the cumulative
  framework changes from 0.3.32 (initial `applies_to` design as
  a single literal, superseded by 0.3.33's glob + list shape).
- **`WorkspaceCfg` dataclass gains `applies_to: str | list[str]
  | None`** (`kglite/mcp_server/manifest.py`) — parsed from the
  framework's polymorphic JSON shape and passed through to
  consumers verbatim.

### Added (MCP server)

- **C9 / C10 / E6 / E7 regression tests.** C9:
  `csv_http_server: true` produces a URL with a non-zero port.
  C10: two concurrent servers booted with the default both come
  up alive without port collision. E6: parent-walk discovery
  with `applies_to: ./*` resolves the parent manifest. E7
  (safety property): parent-walk is refused when the parent
  manifest has no `applies_to` AND when an `applies_to` literal
  doesn't match the child's basename.
- **Migration guide extended** (`docs/migrations/mcp-0.6-to-0.9.md`):
  new "Translation cheat-sheet" section maps common pre-0.9.x
  patterns to the new mode flags ("loads a .kgl at boot →
  `--graph`", "uses set_root_dir → `workspace.kind: local`",
  etc.). New ".kgl format compatibility" section calls out
  that 0.6.x – 0.8.x graphs cannot be loaded by 0.9.x and must
  be rebuilt. Workspace manifest auto-detection section
  describes `applies_to` opt-in.
- **Example manifest updated** (`examples/open_source_workspace_mcp.yaml`):
  declares `workspace.applies_to: ./*` to demonstrate the
  layout-B (manifest beside workspace dir) pattern. Header
  comment shows both layouts side-by-side.

## [0.9.28] — 2026-05-14

Fixes three bugs the mcp-servers operator surfaced in their 0.6.18 →
0.9.27 deployment-verification audit: workspace mode wasn't actually
building code-tree graphs on activate, `local_workspace` mode booted
with an empty graph, and `kglite.code_tree` attribute-chain access
raised at runtime. Two of the four servers they were migrating
couldn't work end-to-end before this release; they can now.

### Fixed (MCP server)

- **`--workspace` mode now actually builds graphs on activate.** The
  workspace's `post_activate` hook was registered as a Python wrapper
  in 0.9.24 but never wired into `_build_server` — so
  `repo_management('org/repo')` would clone the repo, no code-tree
  build would fire, and the next `cypher_query` returned
  `No active graph.` The hook is now wired and fires on both
  `repo_management` activate and `set_root_dir`. Triggers
  `graph_state.build_code_tree(active_path)` + `source_roots[:] =
  [active_path]` so source tools (`read_source`, `grep`,
  `list_source`) target the active clone.
- **`workspace.kind: local` mode builds the code-tree at boot.**
  Previously local-workspace booted with an empty graph until the
  agent issued the first `set_root_dir`. Mirrors watch mode's
  boot-time `build_code_tree(mode_path)` so the first
  `cypher_query` against a freshly-booted local-workspace server
  sees a populated graph.
- **`kglite.code_tree` attribute-chain access works again.**
  `kglite/mcp_server/tools.py::GraphState.build_code_tree` was
  calling `kglite.code_tree.build(...)` as an attribute chain on the
  kglite package, but `kglite/__init__.py` doesn't import the
  submodule eagerly — so the call raised `AttributeError` at runtime
  the first time the workspace post-activate hook tried to fire.
  Now uses `from kglite import code_tree` to force the submodule
  load, with a clean error if the bundled tree-sitter grammars
  aren't available.

### Added (MCP server)

- **Migration guide for operators upgrading from 0.6.x – 0.8.x**
  (`docs/migrations/mcp-0.6-to-0.9.md`). Covers the shift from
  custom Python MCP scripts to the bundled manifest-driven
  `kglite-mcp-server`: operating modes, tool surface differences
  (`read_source` split, `grep_source` → `grep`, `ripgrep` is not
  a bundled name), embedder transition
  (sentence-transformers/torch/MPS → fastembed/ONNX), manifest
  cheat-sheets, and common gotchas. Linked from `docs/index.md`.
- **F6/F7/F8 regression tests.** F6: `local_workspace` mode builds
  code-tree at boot. F7: `set_root_dir(child)` rebuilds the
  code-tree for the new root via the post-activate hook. F8:
  `from kglite import code_tree` loads successfully. 62/62 mcp
  tests green (was 59 in 0.9.27).

## [0.9.27] — 2026-05-13

Picks up the `tools[].bundled:` override shape from mcp-methods
0.3.31 and the cross-binary `repo_management` gating fix that
landed in the same framework release. Closes a customisation gap
that had been forcing operators to stuff per-tool guidance into
the global `instructions:` block.

### Added

- **`tools[].bundled:` override shape** — manifests can now
  customise the agent-facing surface of bundled tools without
  declaring them inline. Two override types:
  - `description: "..."` — replaces the bundled tool's default
    agent-facing description (what shows in `tools/list`).
    Lets operators teach agents that `repo_management` is the
    FIRST STEP, or that `cypher_query` returns a specific
    dataset shape, without burying the guidance in the global
    instructions blob.
  - `hidden: true` — drops the tool from `tools/list` AND rejects
    direct call attempts with `Error: tool 'X' is hidden by
    manifest configuration.` Useful for narrowing the agent
    surface (e.g. hiding `ping` on a production server, or
    suppressing source tools when the auto-bound `source_root`
    is wider than the operator wants).

  Both validate against the kglite bundled-tool catalogue at
  boot: a typo in the `bundled:` name exits 3 with
  `ERROR: unknown bundled tool name(s) ... Valid names: [...]`
  listing the full catalogue.

  Example:

  ```yaml
  tools:
    - bundled: repo_management
      description: |
        FIRST STEP for this server. Call repo_management('org/repo')
        to clone + build a repo before any other tool.

    - bundled: ping
      hidden: true                # narrow the agent surface

    - name: similar_sessions      # existing cypher-tool shape unchanged
      cypher: ...
  ```

  Cypher tools (`tools[].cypher` entries) carry their own
  description in the manifest entry and are NOT affected by
  bundled overrides — those apply to the fixed bundled catalogue
  only (`cypher_query`, `graph_overview`, `ping`,
  `read_code_source`, `save_graph`, `read_source`, `grep`,
  `list_source`, `repo_management`, `set_root_dir`,
  `github_issues`, `github_api`).

- **Four regression tests** in
  `tests/test_mcp_server_python_entry.py`:
  - **B10** — `bundled: cypher_query` with `description:` appears
    in `tools/list` with the override text.
  - **B11** — `bundled: ping` with `hidden: true` drops `ping`
    from `tools/list` while leaving other tools intact.
  - **B12** — calling a hidden bundled tool by name returns the
    `hidden by manifest configuration` error rather than
    falling through to "unknown tool."
  - **B13** — an unknown bundled name (`bundled: cipher_query`)
    fails at boot with an error listing the valid catalogue.

  103/103 mcp + extensions_schemas tests green (was 99 in 0.9.26
  + 4 new).

### Changed

- **mcp-methods pin: 0.3.30 → 0.3.31** (auto-resolved via
  Cargo.toml's `version = "0.3"` constraint; Cargo.lock locks
  the exact 0.3.31). The framework half of the bundled-override
  work landed in 0.3.31 alongside our implementation; we did the
  Rust changes ourselves and the maintainer reviewed + released
  with no revisions. The same release also fixed the
  `repo_management` cross-binary gating drift we'd flagged from
  the operator's post-0.9.25 verification — `mcp-server` (bare
  framework) and `kglite-mcp-server` now register the tool with
  the same gating rules.

### Internal

- `BUNDLED_TOOL_NAMES` frozenset at `kglite/mcp_server/server.py`
  defines the catalogue against which `tools[].bundled:` names
  are validated. Adding or removing a bundled tool requires
  updating this set; manifest overrides will surface "unknown
  bundled tool" errors otherwise.

## [0.9.26] — 2026-05-13

Operator-driven release combining a CLI fix that unblocks the
wikidata pure-YAML migration, the Cat G-N fixture acceptance
that closes the 0.9.16 → 0.9.25 arc, a disk-storage write-path
guard that turns silent data loss into a clean error, and a
significant docs sweep across `docs/guides/mcp-servers.md`.

### Fixed

- **`kglite-mcp-server --graph` now accepts disk-backed graph
  directories**, not just single `.kgl` files. Pre-0.9.26 the
  validator (`server.py::_validate_mode_paths`) used
  `Path.is_file()` which silently rejected any directory — even
  though `kglite.load(path)` (the Python API) accepts both shapes
  fine. The error message even read "does not exist" when the
  path was demonstrably a valid graph directory, which was
  misleading. The new validator accepts a path if EITHER it's a
  regular file (the `.kgl` case) OR a directory containing the
  `disk_graph_meta.json` sentinel (the disk-graph case, same
  marker the Rust loader at `src/graph/io/file.rs::load_file`
  uses). Reported by the mcp-servers operator after they shipped
  the wikidata preprocessor migration against 0.9.25; the bug
  blocked the last 84 lines of `wikidata_mcp_server.py` from
  being deleted (their disk-backed 124M-node Wikidata graph
  couldn't boot via the CLI). Anyone deploying a
  `storage="disk"` graph (the documented kglite path for
  >50M nodes) hit this immediately.
- **Cypher `CREATE` / `MERGE` on `storage="disk"` graphs now
  fails loudly** (returns a clear `Cypher error` pointing at
  `add_nodes()` / `to_disk()` workarounds) instead of silently
  succeeding-with-no-data. Pre-0.9.26 the disk `add_node` path
  only stored a slot (type + row_id) and dropped the
  `NodeData.properties` / `title` / `id` fields, so `CREATE
  (:Marker {title: 'x'})` against a disk graph completed without
  error but every property and the auto-title vanished — both
  in-memory (the column store was never told) and after
  save/reload. Discovered while writing the B6 regression test;
  not on any reported issue list but a silent-corruption class
  worth surfacing. Affects only `CREATE` and the create-path of
  `MERGE`; `SET` / `DELETE` on disk-backed graphs work
  correctly. The proper disk write-path implementation is on
  the roadmap.

- **Cypher `REMOVE n.prop` on `storage="disk"` graphs now works
  correctly.** Discovered as a silent no-op during the
  disk-CREATE guard investigation: the disk staged-write flush
  (`flush_node_mut_cache`) only persisted property keys *present*
  in the staged Map, so a bare `properties.remove(key)` from
  `NodeData::remove_property` left the column store untouched
  and reads returned the original value. Fix: a new
  `NodeData::clear_property` helper inserts `Value::Null` for
  the key instead, which the flush writes through to the column
  store. `execute_remove` now routes to `clear_property` on
  disk-backed graphs via an `is_disk()` branch; memory and
  mapped backends keep the prior in-place `remove_property`
  behaviour (no change). Verified by B9 (regression test) +
  parity with the documented `SET n.prop = null` path.
- **`DiskGraph::node_weight` debug-assertion no longer fires on
  false-positive cases.** The 0.9.0 Cluster 6 hygiene check at
  `disk/graph.rs::node_weight` previously fired whenever
  `node_mut_cache` had ANY entry for the index being read. That
  included `PropertyStorage::Columnar { row_id, .. }` scratch
  entries left by `batch.rs::flush_chunk` (the `add_nodes`
  path) — those are "already persisted via full-Arc
  replacement, safe to discard" and not a missed-flush concern.
  The check now filters to non-empty `PropertyStorage::Map`
  entries only (the actual Cypher-style staged writes that
  WOULD be shadowed by a column-store read). Removes the
  warning noise that appeared during normal `maturin develop`
  test runs. Debug-only assertion; never appeared in release
  builds.

### Added

- **B6 / B7 / B8 / B9 regression tests** in
  `tests/test_mcp_server_python_entry.py`:
  - B6 — `--graph <disk-graph-dir>` boots the server and serves
    a `cypher_query` against the persisted nodes (built via
    `add_nodes()`, the supported disk-mode write path).
  - B7 — `--graph <arbitrary-directory-without-meta>` is still
    rejected with the new error message, so the validator
    isn't too permissive.
  - B8 — disk-mode Cypher `CREATE` / `MERGE` returns the new
    loud-failure error pointing at `add_nodes` / `to_disk`,
    not a silent no-op.
  - B9 — disk-mode Cypher `SET` and `DELETE` still work normally
    (the guard is narrow by design; existing mutation paths
    must not regress).
- **Cat J / K / L test fixtures and 8 forward-looking tests** —
  the mcp-servers operator delivered the fixture bundle that was
  the last open thread from the 0.9.16 → 0.9.25 arc. Four tiny
  `.kgl` files (5.9 KB total) + paired manifests under
  `tests/fixtures/{spatial_graph,timeseries_graph,graph_with_orphans,graph_with_duplicates}.kgl`,
  with the fixture catalog at `tests/fixtures/CAT_G_N_FIXTURES.md`
  documenting what each one anchors. The wired tests:
  - **J1-J3** (spatial Cypher) — `contains(area, point(lat, lon))`,
    `centroid(polygon)`, query-side `point(...)` literal lookup.
  - **K1-K3** (timeseries Cypher) — `ts_sum(channel, 'YYYY')`,
    `ts_at(channel, 'YYYY-M')`, and `ts_sum` across multiple
    matched nodes. Asserts against the random-seeded values
    in the fixture (TROLL oil 2019 sums to 1563.55; March 2019
    spot is 177.12).
  - **L1-L2** (procedures) — `CALL orphan_node({type:'Wellbore'})`
    returns 3 isolated nodes; `CALL duplicate_title({type:'Prospect'})`
    returns 4 duplicate-set members across two pairs.
  97/97 mcp + schemas tests green (was 89 in this release before
  the fixture wire-up).

### Changed

- **`docs/guides/mcp-servers.md` — significant sweep on top of
  yesterday's six accuracy fixes** (`51436df`). 0.9.26 adds:
  - Quick Start renumbered 1–4 (was 1, 2, 2½, 3); manifest
    teaser moved after Claude registration so the install-and-
    point happy path reads top-to-bottom.
  - Stale `--embedder` CLI-flag reference (line 58) removed —
    that flag doesn't exist; the supported path is
    `extensions.embedder` in a manifest.
  - "Custom embedders" subsection under Built-in patterns
    rewritten as a 4-line pointer to the
    `extensions.embedder` reference + worked example (was
    using the removed-in-0.9.18 `embedder: { module, class }`
    shape with `--trust-tools`).
  - `read_source` / `grep` / `list_source` parameter docs
    reformatted from dense single-line prose to per-tool
    parameter tables.
  - New "Deployment shapes" section with a "Large graphs
    (disk-backed)" subsection covering the canonical Wikidata-
    scale flow (ntriples loader → `storage="disk"` → CLI
    pointed at the directory).
  - New "Known limitations" section documenting the disk-
    `CREATE`/`MERGE` refusal, the disk-`REMOVE` silent no-op,
    and the `repo_management` cross-binary gating drift between
    `kglite-mcp-server` and the bare `mcp-server` CLI.
  - New "Troubleshooting" section gathering common post-boot
    pitfalls (GITHUB_TOKEN discovery, `text_score()` returning
    zero, warm-call slowness, conda PATH shadowing, tools
    missing from `tools/list`, PyPI simple-index lag).
  - Pre-0.9.20 migration sections (90 LoC) moved to
    `docs/migrations/mcp-pre-0.9.20.md`, leaving a one-line
    pointer in the main guide.
- **`mcp-methods` dependency switched from git+rev to crates.io**
  (commit `f53e8f1`, also between releases). mcp-methods 0.3.30
  was the first crates.io publish; library binary surface is
  functionally identical to 0.3.29's `71f7ba6`. The switch is
  cosmetic Cargo.toml tidy — no behaviour change. Cargo.lock
  locks the exact version for reproducible builds.
- **`mcp-methods` dependency switched from git+rev to crates.io**
  (commit `f53e8f1`, also between releases). mcp-methods 0.3.30
  was the first crates.io publish; library binary surface is
  functionally identical to 0.3.29's `71f7ba6`. The switch is
  cosmetic Cargo.toml tidy — no behaviour change. Cargo.lock
  locks the exact version for reproducible builds.

## [0.9.25] — 2026-05-12

Doc + feature release driven entirely by the mcp-servers operator's
end-of-arc audit (`inbox/read/2026-05-12-from-mcp-servers-end-of-arc-audit.md`).
The operator flagged that 0.9.24 was "genuinely solid" but they'd
hesitate to recommend `kglite-mcp-server` to a third party because
the docs left them inbox-thread-dependent for edge cases. 0.9.25
addresses every gap in the audit (eight reference doc sections + four
worked examples + machine-readable JSON schemas) and ships the one
feature that retires their last custom Python MCP server
(`extensions.cypher_preprocessor`).

### Added

- **`extensions.cypher_preprocessor`** — manifest-declarable Python
  hook that fires before every `cypher_query` and `tools[].cypher`
  invocation. The hook can rewrite the query string and/or params
  before they reach `graph.cypher(...)`. Gated by
  `trust.allow_query_preprocessor: true`. The motivating use case
  is Wikidata Q-number rewriting (`{nid: 'Q42'}` →
  `{id: 42}` against the integer-id graph), but the hook generalises
  to date normalisation, multi-tenant scoping, parameter validation,
  and any "rewrite agent input before query execution" shape that
  pure-declarative regex can't express. Class-based loaders thread
  `kwargs:` through to `__init__`; free-function loaders work for
  state-free rewriters. Boot-time errors (trust gate, missing module,
  missing class/function) exit 3 with the operator-facing message;
  runtime exceptions surface as `preprocessor: <message>` in the
  tool body without leaking a traceback. ~50 LOC implementation in
  `kglite/mcp_server/preprocessor.py`; 9 regression tests
  (`test_o1`-`test_o9` in `tests/test_mcp_server_python_entry.py`)
  cover the full contract from in-process unit dispatch to
  end-to-end YAML round-trip through MCP stdio.
- **Eight reference doc sections in `docs/guides/mcp-servers.md`** —
  fills every gap the operator's end-of-arc audit called out:
  mode × YAML-field acceptance matrix; tool gating rules; tool
  response formats (with stability tags so the 0.9.21 row-formatter
  regression class can't recur silently); `extensions:` schema
  reference; `tools[].cypher` template reference ($param semantics,
  JSON Schema flavour, error envelope, FORMAT CSV inheritance);
  embedder backend × model catalog; path resolution + manifest
  discovery rules; operator notes (pip-index lag workaround,
  conda guidance, watch-mode rebuild costs).
- **Four worked manifest examples under `docs/examples/`** —
  `manifest_cypher_tool.md`, `manifest_with_embedder.md`,
  `manifest_workspace.md`, `manifest_cypher_preprocessor.md`. Wired
  into the docs guide via a toctree.
- **Machine-readable JSON Schema (Draft 2020-12)** for each
  first-class `extensions.*` block under `docs/schemas/extensions/`.
  Linked from the reference docs. Anchored to the Python parsers
  by `tests/test_extensions_schemas.py` (44 tests) — schema/parser
  drift fails loudly in CI.
- **`Manifest.trust` dataclass field** — `allow_python_tools`,
  `allow_embedder`, `allow_query_preprocessor` populated from
  `mcp_methods::server::Manifest::to_json()` output. Available to
  the rest of `kglite/mcp_server/` (and to tests).

### Changed

- **mcp-methods pin: 0.3.28 → 0.3.29** (rev `1ba9469` → `71f7ba6`).
  Adds `allow_query_preprocessor` to `ALLOWED_TRUST_KEYS` and
  `TrustConfig`, plus emits it under the `trust` object in
  `Manifest::to_json()`. Non-breaking JSON shape addition.
- **`kglite.mcp_server.tools.run_cypher` signature** — adds an
  optional `preprocessor: Preprocessor | None = None` parameter.
  Existing callers (every prior release plus all current internal
  call sites) work unchanged via the default. Same for
  `kglite.mcp_server.cypher_tools.call_cypher_tool`.

### Internal

- Pre-release suite expanded to 39 default-mode tests (was 34):
  added the 9 new cypher_preprocessor tests (O1-O9) plus 44 schema
  drift tests in `tests/test_extensions_schemas.py`.
- W1 watch-callback assertion loosened to accept either the changed
  file path or the parent directory — macOS FSEvents coalesces
  depending on rate, and the contract we care about is "the
  callback receives a `list[str]` within the debounce window," not
  the path-granularity decision the OS makes.

## [0.9.24] — 2026-05-12

Architectural cleanup: kglite's MCP server framework is now a thin
shim over `mcp-methods` Rust rather than a parallel Python
re-implementation. ~600 LOC of Python deleted; the validated Rust
behaviour replaces it. As a side effect, the 0.9.23 set_root_dir
sandbox-narrowing regression the operator flagged is fixed by
construction.

### Fixed
- **`set_root_dir` no longer narrows the sandbox with each swap.**
  The pre-0.9.24 Python `Workspace.set_root_dir_tool` mutated
  `self.root` after each successful swap, so the next sandbox check
  compared against the narrower active root rather than the
  manifest's declared `workspace.root`. After one swap, lateral
  swaps to sibling projects under the configured root failed with
  "escapes the workspace root." Fix: the new wrapper inherits
  `mcp_methods::server::Workspace`'s atomic-swap RwLock + immutable
  configured `workspace_dir`, so the sandbox check always validates
  against the manifest's declared root. Workspaces remain swappable
  to any sibling under the configured root for the lifetime of the
  server — no restart required.

### Changed
- **`kglite/mcp_server/{manifest,workspace,watch}.py` are now thin
  pyo3 wrappers** around `mcp_methods::server::{Manifest,Workspace,
  watch_dir}`. The Python surface stays the same (`Manifest`
  dataclass, `Workspace.root` / `.kind` / `.repo_management_tool` /
  `.set_root_dir_tool`, `watch.start`), but the implementation is
  ~600 LOC of Rust behind a single passthrough each. The Python
  manifest dataclass is populated from `Manifest::to_json()` (new
  in mcp-methods 0.3.27) so field drift between the framework and
  downstream consumers is a non-issue.
- **`.env` walk-up delegated to mcp-methods Rust**
  (`load_env_walk`). Same parse rules (skip blanks / `#` comments,
  strip outer quotes, no-overwrite-existing-env), same result —
  one fewer parallel implementation to keep in sync. Explicit
  `env_file:` paths still loaded inline.
- **File watcher uses `notify-debouncer-mini` via Rust** instead of
  the pure-Python `watchdog` + threading debounce. The
  `watchdog` extra remains in `[mcp]` optional-deps for now —
  downstream tooling may depend on it — but kglite's own watch path
  no longer uses it.
- **Pin: `mcp-methods` rev `1ba9469` (0.3.28).** Three same-day
  releases against this cleanup: 0.3.26 (three-crate split), 0.3.27
  (`Manifest::to_json`), 0.3.28 (local-mode `set_root_dir` no longer
  clobbers `active_repo_path`). Two of those bumped specifically to
  unblock the 0.9.24 pyo3 wrapper; the third was a bug found during
  the wrap pass.

### Added
- **`kglite._mcp_internal.{Manifest, Workspace, WatchHandle,
  start_watch, load_env_walk}`** — pyo3 wrappers around
  `mcp_methods::server::*`. Internal surface (the public Python
  entry point is still `kglite.mcp_server.server:main`), but
  importable for tests and for downstream tools that want the
  validated mcp-methods behaviour without a Python re-implementation.
- **4 new regression tests** anchoring the pyo3-wrapper boundary:
  F4 (sandbox lateral swap — operator's bug repro), E5 (manifest
  extensions passthrough — recursive `serde_json::Value` → dict),
  F5 (workspace post-activate hook GIL dispatch), W1 (watch
  callback receives changed paths). 34/34 mcp_server tests green
  (was 30/30).

### Internal
- `anyhow` added as a direct dep — required by
  `mcp_methods::server::PostActivateHook`'s `Result<(), anyhow::Error>`
  signature.
- `MEMORY.md` of "thin Python shim" intent updated for future
  sessions (see `CLAUDE.md`'s new "Standard plan procedure" section
  for the per-phase commit rhythm we followed for 0.9.24).

## [0.9.23] — 2026-05-12

### Fixed
- **`extensions.csv_http_server` returned HTTP 500 on every GET.**
  aiohttp's `web.Response` rejects `content_type` strings that
  contain a charset directive — we were passing
  `"text/csv; charset=utf-8"`. Fix: pass `content_type="text/csv"` and
  `charset="utf-8"` as separate kwargs. Operator workaround was to
  disable the csv_http_server block; 0.9.23 makes it usable again.
- **`github_issues` now auto-defaults to the workspace's active repo
  when `repo_name` isn't supplied.** Previously `repo_management(name)`
  activated the repo correctly but `github_issues` without
  `repo_name` hit the "could not auto-detect from git remote" error
  path. Workspaces now track `active_repo` and the github_issues
  dispatcher uses it as the fallback. Agents no longer need to
  repeat `repo_name='org/repo'` on every call.
- **`set_root_dir` now actually rebinds the source tools.** Previously
  the tool registered, the workspace's `root` field updated, but the
  `source_roots` list captured at server build time wasn't refreshed
  — so `list_source` / `grep` / `read_source` kept hitting the old
  root. Tests/test_f2 was designed to catch exactly this and did.
- **`BgeM3Embedder.unload()` is now a no-op** (formerly dropped the
  ONNX session). kglite's `kg_core.rs::cypher` does
  `load → embed → unload` around every text_score call, so dropping
  the session meant every cypher paid the full ~1s ORT session init.
  Warm-call latency drops from ~1.1s to ~50ms; ~2 GB RAM stays
  resident while the embedder is in use.

### Added
- **27-test pre-release suite per the operator's spec.** Every test
  maps to a specific bug from the 0.9.16 → 0.9.22 release arc.
  Categories: A (install/boot), B (per-mode tool registration),
  C (tool output content), D (embedder + semantic search), E
  (manifest/.env), F (workspace state propagation). The
  tool-output content assertions (Cat C) are the gate that would
  have caught 0.9.21's row formatter and 0.9.22's csv_http 500
  before release.
- **bge-m3 cool-down timer**: configurable via
  `extensions.embedder.cooldown` (default 900s = 15 min;
  `0` = never release). Active sessions hold the ONNX session
  resident for fast queries; long-idle servers release ~2 GB of
  RAM. Cool-down check fires on each `embed()` call — no background
  threads.
- `BgeM3Embedder.release()` — explicit counterpart to the no-op
  `unload()`. Drops the ORT session + tokenizer when the caller
  really wants the memory back.
- `tests/fixtures/build_tiny_graph.py` — programmatic 50-node-per-
  type fixture (Person + Company + Article) with semantically
  clustered article bodies (quantum / baking / programming) for
  embedder relevance tests.

### Changed
- **`extensions.embedder` YAML now accepts `cooldown:`** (seconds).
  Falls through to the BgeM3Embedder for `BAAI/bge-m3`; FastEmbed
  adapter ignores the field for other models (their lifecycle
  follows fastembed-python's defaults).

## [0.9.22] — 2026-05-12

### Fixed
- **`cypher_query` row formatter now returns row values, not column
  names.** 0.9.21 regression: `_format_inline` in
  `kglite/mcp_server/tools.py` iterated `for v in row` against a dict,
  yielding the column names as values. Every non-CSV `cypher_query`
  call produced rows like `'f.name'\t'f.line_number'` instead of the
  actual data. Operator caught it on redeploy and rolled back to
  0.9.18. Fix: index the row dict by column (`row[col]`) so the
  preview shows real values. The Rust-side Cypher engine + the
  `FORMAT CSV` path were always correct — only the inline preview
  formatter was wrong.

### Added
- `test_cypher_query_returns_actual_row_data` integration test —
  asserts the inline preview contains the computed value (e.g. `2`)
  and **not** the column name (`'sum'`). The 0.9.21 regression class
  ("tool registers but returns garbage") can no longer reach release
  without breaking the build. Same shape as the existing per-tool
  content assertions (ping returns `pong`, read_source returns the
  file slice, grep returns matches).

## [0.9.21] — 2026-05-12

Fixes the two 0.9.20 regressions the operator caught on redeploy:
8 of 11 tools were silently missing, and bge-m3 embedder was broken
because fastembed-python doesn't carry that model in its catalog.

### Added
- **All 8 framework tools restored**: `ping`, `read_source`, `grep`,
  `list_source`, `repo_management`, `set_root_dir`, `github_issues`,
  `github_api`. Same output format as the 0.9.18 binary. Implementation
  comes from the pure-Rust `mcp-methods` crate (0.3.26+, three-crate
  split with zero pyo3 in the library half) wrapped via pyo3 in
  `src/mcp_tools.rs` and exposed as `kglite._mcp_internal`. The
  Python `kglite.mcp_server.server` entry point dispatches each
  tool to the wrapped Rust function — `cypher()`-style GIL release
  preserves the original Rust performance.
- **`BgeM3Embedder`** (`kglite/mcp_server/bge_m3.py`): direct
  onnxruntime + huggingface_hub implementation for `BAAI/bge-m3`
  because fastembed-python's catalog doesn't include it. Same ONNX
  weights as fastembed-rs, same CLS pooling, same
  `~/.cache/fastembed/` cache directory — operator's existing
  downloaded weights reused without re-download. Other models
  (bge-small/base/large, all-MiniLM-L6-v2, multilingual-e5) continue
  through fastembed-python.
- **CI gate against tool-surface regression**:
  `tests/test_mcp_server_python_entry.py` boots the server in every
  supported mode and asserts `tools/list` matches
  `tests/fixtures/tool_baseline.json` exactly. Any added or removed
  tool fails the build. Adopted in response to the 0.9.20 failure mode
  where the regression was caught after release.

### Changed
- `kglite::api` Rust facade adds `mcp-methods 0.3.26` as a curated
  dep (`default-features = false, features = ["server"]`, no pyo3 in
  its tree). Downstream Rust consumers can use `mcp_methods::*`
  directly without going through us.
- `extensions.embedder` dispatcher routes `BAAI/bge-m3` to the new
  `BgeM3Embedder`; all other models continue to fastembed-python.
- Auto-bind manifest's directory as fallback source root when no
  `source_root:` is declared. Matches the 0.9.18 binary's behaviour
  — without this fallback, sodir-style manifests (no source_root)
  silently lost `read_source`/`grep`/`list_source`.

### Fixed
- 0.9.20's tool-surface regression (8 missing tools per manifest).
- 0.9.20's bge-m3 catalog regression (`text_score()` broken on every
  embedder-enabled deployment).

## [0.9.20] — 2026-05-11

### Changed
- **`kglite-mcp-server` is now a Python console-script entry point,
  not a bundled Rust binary.** The 0.9.18/0.9.19 binary bundling
  forced a 12-wheel (3 OS × 4 Python) build matrix because any
  Rust binary that transitively depends on pyo3 links libpython at
  a specific version — no abi3 escape for binaries. The Python
  entry point bypasses that entirely. Wheel matrix back to 3 abi3
  wheels per release (same as pre-0.9.18). Performance unchanged:
  kglite's Python `cypher()` already releases the GIL inside
  `py.detach()`, so the wrapping layer is sub-microsecond.
- The 0.9.18 conda install_name regression and the 0.9.19
  `install_name_tool` / `patchelf` / mold post-build surgery are
  gone — there's no binary to mis-link.
- Wheel deps: install via `pip install 'kglite[mcp]'` to pull the
  server-time deps (mcp, pyyaml, fastembed, aiohttp, watchdog).
  Plain `pip install kglite` skips them — for users who just want
  the graph engine.

### Removed
- `kglite/_bin/` directory inside the wheel (was per-Python binary
  drop site).
- `kglite/_cli.py` (was the launcher that exec'd the bundled
  binary).
- Per-Python-version wheel build matrix axis.
- `install_name_tool` / `patchelf` / mold steps from the CI
  workflow.
- `[project.optional-dependencies] embeddings` (torch +
  sentence-transformers) — superseded by `[mcp]` which uses
  fastembed natively.

### Internal
- `crates/kglite-mcp-server/` stays in the repo for direct Rust
  consumers (Wikidata-scale deployments, Docker-vendored binaries)
  but is no longer bundled into the wheel. Build via
  `cargo build -p kglite-mcp-server` if you want it.

## [0.9.19] — 2026-05-11

### Changed
- **Wheel build is substantially faster.** Three workflow changes:
  - Drop fastembed's `image-models` default feature (jpeg/png/webp
    decoders we don't use) — ~3-4 min/wheel saved.
  - Add `Swatinem/rust-cache` to the wheel-build workflow with a
    per-target shared key so most deps (mcp-methods, hyper,
    fastembed, tokio) are reused across the four Python-version
    cells within each OS. Warm builds reuse the cache.
  - Use the `mold` linker on Linux. The bundled ld spent ~1-2 min
    linking the cdylib + binary at end-of-build; mold does the same
    work in ~10s. macOS and Windows keep their platform linkers.

### Fixed
- **`pip install kglite` now works on conda Python.** The 0.9.18 wheel
  shipped the bundled `kglite-mcp-server` binary with an absolute
  install_name pointing at `/Library/Frameworks/Python.framework/...`
  (the actions/setup-python build path); conda installations don't
  have that path and the binary failed to launch with a dyld error.
  The wheel-build workflow now rewrites the install_name to
  `@rpath/libpython3.X.dylib` and adds an rpath relative to the
  binary's wheel install location so dyld finds the env-local
  libpython under conda, venv, virtualenv, and Python.org installs
  uniformly. Linux gets the same treatment via `patchelf --set-rpath
  '$ORIGIN/../../../..'`.
- **`builtins.temp_cleanup: on_overview` now actually wipes the
  configured directory.** The 0.9.18 implementation hardcoded the
  cleanup target to `./temp` (cwd-relative) which only worked when
  the server was launched from the manifest's parent directory.
  0.9.19 resolves the temp directory against the manifest base — and
  reuses `extensions.csv_http_server.dir` when configured, so the
  same place CSVs are written is also the place that gets swept.
- **`FORMAT CSV` row-count status no longer reports `0 row(s) written`
  for queries with `LIMIT N`.** The status counter read
  `result.rows.len()`, which is empty when the planner's lazy
  materialisation kicks in — even though the CSV body has the right
  data. The count now comes from the CSV body itself.

## [0.9.18] — 2026-05-11

### Changed
- **MCP server is now pure-Rust at the source level.** `kglite-mcp-server`
  no longer calls PyO3 anywhere — every tool handler goes through the
  new `kglite::api` façade (Cypher pipeline, `compute_description`,
  `build_code_tree`, `source_location`). The mcp-methods Python
  feature is off, so the framework's Python tool surface isn't on
  the binary's dep graph either. The bundled binary still links to
  libpython transitively through kglite's own PyO3 layer (kglite is a
  Python library — that's by design), so the wheel matrix stays at
  3 OS × 4 Python = 12 wheels per release.
- **Embedder backend switched to fastembed-rs.** The framework-level
  `embedder:` Python factory is gone. Configure with
  `extensions.embedder: { backend: fastembed, model: BAAI/bge-m3 }`
  instead — bge-m3, bge-small/base/large-en-v1.5, all-MiniLM-L6-v2,
  and multilingual-e5 supported out of the box. ONNX weights are
  downloaded to `~/.cache/fastembed/` on first use; no `torch` /
  `sentence-transformers` install needed.

### Added
- `kglite::api` — curated Rust façade for downstream binaries. Exposes
  `KnowledgeGraph` + `DirGraph` + `Embedder` + `FastEmbedAdapter` +
  the Cypher parse/plan/execute surface + `compute_description` +
  `compute_schema` + `load_file` + `build_code_tree` +
  `SourceLocation`/`SourceLookup`.
- `extensions.csv_http_server` — opt-in localhost HTTP listener that
  serves `FORMAT CSV` exports as URLs instead of inline strings.
  Useful for million-row exports that would blow the MCP response
  budget. Bound to 127.0.0.1, path-traversal hardened, no write
  surface, CORS-enabled.
- `KnowledgeGraph::set_embedder_native(Arc<dyn Embedder>)` — pure-Rust
  counterpart to the `set_embedder` pymethod; lets downstream Rust
  binaries bind embedders without a `Py<PyAny>`.
- `KnowledgeGraph::source_location(name, node_type)` — pure-Rust
  counterpart to `graph.source()` used by the `read_code_source` tool.

### Removed
- `tools[].python` manifest entries — Python tool hooks no longer
  loadable. Move tool logic into a `tools[].cypher` template or a
  downstream Rust binary that embeds the kglite crate directly.
- `embedder:` top-level manifest key — replaced by
  `extensions.embedder:` (see Changed).
- Pre-0.9.18 install-UX workarounds (`PYO3_PYTHON=`,
  `install_name_tool -add_rpath`, conda-env symlinks) are no longer
  needed; `pip install kglite` ships `kglite-mcp-server` on `PATH`
  directly.

## [0.9.17] — 2026-05-11

### Added

- **`read_code_source(qualified_name=...)` MCP tool** — kglite-side
  companion to the framework's `read_source(file_path=...)`. Resolves a
  fully-qualified entity name through the active graph's
  `graph.source()` (which uses the code-tree node attributes), then
  reads the corresponding file slice from the configured source root(s).
  Equivalent to cypher → graph.source → read_source in a single MCP
  call. Same `start_line` / `end_line` / `grep` / `max_chars` filters
  as `read_source`. Restores the qualified-name flow operators relied
  on pre-0.9.14; reported by the MCP-servers operator after the
  0.9.14 framework take-over trimmed `read_source` to `file_path`-only.
- Boot-summary line on stderr now names the `.env` file actually
  loaded (or reports `(no .env found)` when walk-up came up empty),
  closing the gap between "token missing" and "token present but
  unreadable".

### Fixed

- **`kglite-mcp-server` now actually loads `.env` files.** The
  shim's `main.rs` never invoked mcp-methods'
  `load_env_for_mode()` — so the framework's walk-up + `env_file:`
  YAML key support, although present in mcp-methods 0.3.22+, never
  fired under the kglite binary. Operators ran into "GITHUB_TOKEN
  not set" with a `.env` one directory up from their workspace,
  which should have been auto-discovered. Now wired: walk-up from
  `--graph` parent / `--source-root` / `--workspace` /
  `--watch` / `workspace.kind: local` root / cwd-in-bare, with
  explicit `env_file:` in the manifest as override.
- **`embedding_diagnostics()` now sees columnar properties.** The
  0.9.16 implementation iterated `NodeData::property_iter()`, which
  yields nothing for `PropertyStorage::Columnar` — the variant nodes
  use after save+reload. As a result, diagnostics on a freshly-loaded
  graph reported `nodes_with_property: 0` for properties that
  actually existed (Cypher `WHERE x IS NOT NULL` confirmed), flipping
  the status to `store_orphan` on a healthy steady-state graph. Same
  root cause hid the `embeddable` status when a `node_type` filter
  was passed for a type with a string property but no store yet.
  Fixed by switching to `properties_cloned()`, which dispatches
  across all `PropertyStorage` variants. Two new regression tests
  cover the save+reload and filter-by-type paths.

### Documentation

- **Python linkage policy for `kglite-mcp-server`** — PyO3 picks one
  interpreter at build time. New section in `docs/guides/mcp-servers.md`
  ("Where does the binary find Python? — read this before
  `pip install`") covers the discovery one-liners (`otool -L` /
  `ldd`) and the `PYO3_PYTHON=...` install-time override. README has
  a short callout pointing to the long version. Reported after an
  operator landed 2 GB of `torch` in base conda because the binary
  linked to base Python rather than the sub-env where they pip'd
  `kglite`.

## [0.9.16] — 2026-05-10

### Added

- **YAML `tools[].cypher` entries are now wired into MCP.** The
  `kglite-mcp-server` shim adds a `cypher_tools` module that
  registers each manifest-declared parameterised Cypher tool as a
  first-class MCP tool, dispatching to `graph.cypher(template,
  params=args)` on the active graph. Closes the gap that left all
  five of the MCP-servers project's production manifests'
  `tools:` sections invisible to agents after the Python
  `kglite.mcp_server` was retired. Schema is taken from the YAML
  `parameters:` block when present, otherwise an empty object schema.
- **`manifest.workspace.kind: local` is now honored.** The shim
  promotes a manifest-declared local workspace into a new internal
  `Mode::LocalWorkspace` before mode-specific binding, with
  `set_root_dir` registered for runtime root swap and an optional
  debounced watch loop on `watch: true`. Manifest declaration wins
  over the `--workspace` CLI flag, mirroring the framework's own
  binary. Lets users retire `code_review_mcp_server.py`-style
  custom Python servers in favour of a YAML manifest.
- **`graph.embedding_diagnostics(node_type=None)`** — companion to
  `list_embeddings()` that surfaces per-`(node_type, text_column)`
  coverage with three states: `"embedded"` (store and property both
  present), `"embeddable"` (property present, no store), and
  `"store_orphan"` (store present, no node has the property — the
  symptom an `import_embeddings()` warning indicates). Use it after
  a silent-drop warning to see which stores are affected.
- Type stubs for `import_embeddings()` and `export_embeddings()` —
  both methods existed but were missing from `kglite/__init__.pyi`.
- Documentation of the `code_tree` qualified-name format per
  language with a stability commitment within minor releases —
  `docs/guides/code-tree.md`.
- Recipe: `SET` → `add_properties` migration for hub aggregations,
  with a worked example showing `Agg.count()` / `Spatial.distance()`
  helpers replacing imperative-Cypher `WITH ... SET ...` chains —
  `docs/guides/recipes.md`.
- End-to-end smoke suite for `kglite-mcp-server` over JSON-RPC stdio
  (`tests/test_mcp_server_smoke.py`) — 25 tests covering every tool
  the binary exposes (`cypher_query`, `graph_overview`, `save_graph`,
  `read_source`, `grep`, `list_source`, `github_issues`, `github_api`,
  `set_root_dir`, `ping`, plus YAML-declared parameterised Cypher
  tools). Auto-skips when the binary isn't built; runs in ~3 s.

### Changed

- **`kglite-mcp-server` now pins mcp-methods 0.3.23** (rev `e45a282`,
  bumped from 0.3.21). Brings, in order: `.env` auto-loading,
  GitHub-tool drill-down via `element_id`, honest tool listing gated
  on `GITHUB_TOKEN`, `inventory.json` `last_built_sha` + auto-rebuild
  gating on `repo_management(update=True)`, framework parsing of
  `workspace.kind: local`, the `mcp_methods.fastmcp` Python helper
  submodule (`register_overview` / `register_cypher_query` /
  `register_source_tools` / `register_save_graph` / `serve_csv_via_http`),
  a public `build_tool_attr` for downstream cypher-tool registration,
  and an empty-string filter in `auth_token`. Existing YAML manifests
  parse unchanged — every schema addition is optional.
- The framework's embedder factory now hands back an
  `Arc<EmbedderHandle>` (load/unload/embed/touch + idle tracking)
  instead of a raw `Py<PyAny>`. The shim extracts the underlying
  Python instance via `handle.instance()` and binds that to the
  active graph; kglite's per-batch `set_embedder` lifecycle drives
  the same instance the framework's idle-watch task observes.
- `README.md` migration note: replaces the old `pip install
  "kglite[mcp]"` flow with `cargo install --path crates/kglite-mcp-server`.

### Fixed

- **`import_embeddings()` no longer silently drops mismatched files.**
  When `imported == 0` but the `.kgle` file contained data, or when
  a per-type store had zero matches, the call now emits a
  `UserWarning` describing the mismatch (file path, counts, likely
  cause). The result dict gains a `dropped_stores` key so callers
  can detect partial-drop cases programmatically. Reported via the
  MCP-servers wishlist after a 7 MB embedding file silently became
  `{stores: 0, imported: 0, skipped: 1923}` against a graph whose
  `code_tree` qualified-name format had drifted.
- **`save_disk` no longer fails with `OSError: Invalid argument
  (os error 22)` on disk-backed graphs.** The 0.9.15 unified
  mega-file writer had an early-return gate that required *both*
  `total_bytes == 0` *and* `unhandled.is_empty()` — but unhandled
  types only need sidecar fallback, never bytes in the mega-file.
  With non-zero `unhandled` and zero planned bytes, the code fell
  through to `mmap::map_mut` on a 0-byte file, which returns EINVAL
  on every Unix. Triggered on every fresh disk graph
  (`KnowledgeGraph(storage="disk", ...).add_nodes(...).save(...)`) —
  the entire `tests/test_disk_property_index.py` suite was failing.

## [0.9.15] — 2026-05-10

### Added

- `KnowledgeGraph._save_subset_induced_by_edge_type(path, edge_types)`
  — variant of `_save_subset_filtered_by_edge_type` that produces the
  *induced* subgraph: the kept-node set is still derived from edges
  matching `edge_types` (Pass A), but the output keeps **every** edge
  between any two kept nodes, not just the filter edge type. On the
  Wikidata `articles_authors` carve this expands the result from a
  single P50 layer to ~174 M edges across 20 distinct types
  (P2860 citations, P2093 stated authors, P98 editor, etc.) while
  still pinning the node set to "articles + their authors". Disk
  source only.

### Fixed

- **Streaming subgraph carve now round-trips non-schema properties.**
  The disk-to-disk `save_subset_streaming_disk` writer dropped any
  property whose key wasn't in the type's schema — Wikidata stores
  most low-cardinality properties (P356 DOI, P577 publication date,
  P304 page numbers, …) in a per-row overflow bag, and those were
  silently lost on save. `TypeWriter` now accumulates an overflow
  blob in the same wire format the source uses
  (`[u16 num_entries] + [u64 key | u8 type_tag | value]`), and
  `RowVisitor` routes non-schema keys into it instead of dropping
  them. `ColumnStore::replace_overflow_bag` is the new setter.

- **Saved DiskGraphs now load with mmap-fast-path speed.** A graph
  built in memory and persisted via `save_disk` previously emitted
  per-type zstd sidecars under `columns/<type>/columns.zst`, which
  the loader rebuilt eagerly on every open — ~70 s on a 17 M-node
  carve vs. ~150 ms for the same data when produced by the ntriples
  builder. `save_disk` now emits the unified `seg_000/columns.bin`
  mega-file format the loader's mmap fast path consumes (new
  `crate::graph::io::unified_columns` module), so saved subgraphs
  load in tens of milliseconds. Existing sidecar-format graphs still
  load via the legacy path.

### Performance

- Legacy sidecar column loader (still used for pre-mega-file files)
  parallelised via rayon — read + zstd decode + `load_packed` now
  run per-type concurrently. ~2.3× faster on a 16-core machine for
  the rare case of opening a sidecar-format graph.

## [0.9.14] — 2026-05-09

### Added

- **`kglite-mcp-server` is now a Rust-native single binary**, built
  on top of the `mcp-server` framework (rmcp + manifest-driven tool
  registration) shipped from the sibling `mcp-methods` workspace.
  The binary lives at `crates/kglite-mcp-server/`. Full mode coverage
  matches the previous Python server: `--graph X.kgl`, `--workspace
  DIR`, `--watch DIR`, `--source-root DIR`, plus bare framework. The
  manifest YAML schema is unchanged, so any `<basename>_mcp.yaml`
  written for the Python server boots unchanged on the new binary.
- Workspace mode auto-builds a code-tree graph for each cloned repo
  via a `PostActivateHook` calling `kglite.code_tree.build()`; watch
  mode re-runs the same path on debounced file changes; embedder
  factories declared in the manifest are bound to the active graph
  via `graph.set_embedder()`.

### Removed

- **`kglite/mcp_server/` Python package + the `kglite[mcp]` extras
  dependency group + the `kglite-mcp-server` console script entry +
  `examples/mcp_server.py` + 11 `tests/test_mcp_*.py` modules**
  (~4,150 LoC). All replaced by the Rust binary above. The manifest
  schema and tool surface are 1:1 compatible — agents see the same
  `cypher_query` / `graph_overview` / `save_graph` / `read_source` /
  `grep` / `list_source` / `github_issues` / `github_api` /
  `repo_management` / `ping` tools as before.
- `mcp` optional-dependency group removed from `pyproject.toml`. To
  install the new server: `cargo install --path crates/kglite-mcp-server`
  from a kglite source clone.

### Changed

- `kglite` is now a Cargo workspace (root crate + `crates/kglite-mcp-server`).
  The Python wheel build via maturin is unchanged; a new
  `python-extension` Cargo feature gates `pyo3/extension-module` so
  `cargo build` can share the rlib with the new sibling binary.
- `CLAUDE.md` "When changing a `#[pymethods]` function" checklist
  step 4 now points at `crates/kglite-mcp-server/src/tools.rs`
  (instead of the deleted `examples/mcp_server.py`).

## [0.9.13] — 2026-05-09

### Added

- **`kglite-mcp-server --workspace DIR`: multi-graph workspace mode.**
  Boots the server without a graph; the agent activates one with
  `repo_management('org/repo')`, which clones the GitHub repo, builds
  a code-graph via `kglite.code_tree.build`, and pins it as the active
  graph for `cypher_query` / `graph_overview` / `read_source` / `grep`
  / `list_source`. Inventory tracks `last_accessed` / `access_count`
  per repo in `<workspace>/inventory.json`. Idle repos auto-sweep
  after `--stale-after-days` (default 7); the active repo is exempt
  and stale entries preserve their access history. Layout:
  `<workspace>/{repos,graphs,temp,inventory.json,workspace_mcp.yaml}`.
- **Manifest `embedder:` section** for project-supplied embedder
  factories. Declare `module: ./embedder.py` + `class: GraphEmbedder`
  + `kwargs: {...}` and the CLI imports + instantiates via
  `Class(**kwargs)` and binds with `graph.set_embedder()`. Trust-gated
  by `trust.allow_embedder: true` plus `--trust-tools` (both signals
  required, mirrors the `python:` tool gate). Replaces the
  always-loaded `--embedder MODEL_NAME` shortcut for users who need
  cooldown-based unload (e.g. BAAI/bge-m3 on consumer hardware).
- **Manifest `overview_prefix:` field.** Sticky preamble prepended to
  `graph.describe()` output on bare `graph_overview()` calls. Skipped
  for focused drill-downs (`types=[...]`, `connections=[...]`,
  `cypher=[...]`) so they stay terse. Lets agents re-discover
  load-bearing context (validator hints, baseline counts, hidden
  invariants) deep into a session without competing with the
  conversation for slot in the system instructions.
- **Manifest `builtins:` section** for pre-blessed tools that don't
  need `--trust-tools`. `save_graph: true` registers a `save_graph()`
  MCP tool that calls `graph.save(graph_path)` — for persisting
  CREATE/SET/DELETE Cypher mutations. `temp_cleanup: on_overview`
  clears the CSV-export `temp/` directory on bare `graph_overview()`
  calls so it doesn't grow unbounded across long sessions; `never`
  (default) keeps the existing behaviour.
- **`read_source(qualified_name=...)`** for code-aware servers. When
  the bound graph carries `qualified_name` + `file_path` properties on
  code nodes, the agent can pass a name like `MyClass.my_method` and
  the tool resolves through `graph.source()` to a file slice in one
  round-trip (was: cypher-then-read, two round-trips). Suffix
  fallback handles short bare names (e.g. `helper_fn`) via Cypher
  `ENDS WITH` against `qualified_name`. Available in both single-
  graph and workspace modes.

### Changed

- `kglite-mcp-server` `--graph` and `--workspace` are now mutually
  exclusive flags. Default behaviour (no flag, no `graph.kgl` in cwd)
  is unchanged — single-graph mode looking for `./graph.kgl`.
- `examples/conference_graph_mcp.yaml` annotated with the new
  manifest fields (`overview_prefix`, `builtins`, `embedder`,
  `trust.allow_embedder`).

## [0.9.12] — 2026-05-09

### Added

- **`KnowledgeGraph.save_subset(path)` on the fluent selection chain.**
  Equivalent to `kg.to_subgraph().save(path)` in a single call —
  produces an independent v3 binary file that reloads via
  `kglite.load(path)` (or `load(path, storage='disk')` for disk mode).
  All edges between selected nodes are included; node and edge
  properties round-trip byte-for-byte. Works on any source storage
  mode.
- **`_save_subset_filtered_by_edge_type(path, [edge_types])` —
  disk-to-disk streaming subgraph filter** (Wikidata-scale path).
  Single-pass over the source's `edge_endpoints.bin` builds a kept-
  nodes bitset; a per-type `TypeWriter` then streams kept rows
  directly to dest column files via `BufWriter`s — no intermediate
  in-memory `ColumnStore`, no chunk-and-merge step. End-to-end on the
  full Wikidata graph (124M nodes / 861M edges, P50 + endpoints)
  produces a 17,364,495-node / 35,448,243-edge subgraph in 349s wall
  time (down from 550s on the v1 in-memory path). Working set stays
  in the hundreds of MB regardless of subset size — peak RSS is
  largely soft mmap pages.
- **`BorrowedValue<'a>` zero-copy view of `Value`** in
  `datatypes::values`. `String(&'a str)` borrows from the source
  buffer (typically an mmap region) instead of cloning into a
  `String`. Used by the streaming subgraph filter; available as a
  general read-path primitive. Convert with `to_value()`.
- **`MmapColumnStore::id_borrowed` / `title_borrowed` /
  `try_for_each_property_borrowed`** — allocation-free reads. The
  property visitor decodes overflow-bag bincode entries in place
  and yields `BorrowedValue::String` views into the mmap; previously
  every overflow row allocated a `Vec<(InternedKey, Value)>` plus a
  `String` per entry. ColumnStore wrapper delegates to mmap_store
  for disk graphs.

### Changed

- **`MmapColumnStore::read_str` skips UTF-8 validation.** Source
  bytes were always written through `String::as_bytes()` (Rust's
  UTF-8 invariant), so the `from_utf8` validator was walking ~25 GB
  of source data per Wikidata save for nothing. `from_utf8_unchecked`
  is now used. Saves ~70 s of wall time on the streaming subgraph
  filter; no observable behavior change.
- **Streaming subgraph filter uses borrowed-value writes
  end-to-end.** `TypeWriter::push_row_borrowed` accepts
  `BorrowedValue<'_>` and writes `&[u8]` straight to per-column
  `BufWriter`s without ever materializing `Value::String`. Combined
  with the read-side borrowing above, the Wikidata save phase drops
  from 550s to 349s (-37%). Node-walk sub-phase: 446s → 241s
  (-46%); the `props` portion 330s → 145s (-56%); `id+title` 87s →
  17s (-80%). Output is byte-identical to the prior path —
  `tests/test_subgraph_streaming.py`'s round-trip equality suite
  (14 cases) stays green.
- **Sub-phase timers in `save_subset_streaming_disk` gated on
  `KGLITE_STREAMING_TIMING=1`.** Off by default (zero overhead);
  when enabled, prints per-phase wall times plus per-million-row
  progress so a future optimizer can iterate in 30-second chunks
  rather than 10-minute round-trips. Useful for finding bottlenecks
  without committing to a full bench cycle.

### Changed

- **Edge-driven group-by aggregations with a typed target node now use
  the fast `lookup_peer_counts` path.** Queries shaped
  `MATCH (a)-[:E]->(b:T) RETURN b, count(a) [ORDER BY count(a) DESC LIMIT k]`
  were correctly routed to `FusedMatchReturnAggregate` but BOTH executor
  branches (top-K and non-top-K) bailed when the planner reversed the
  pattern to start at the typed node — the resulting `group_elem_idx == 0`
  short-circuit forced the slow node-centric scan. The fast path now
  detects "group is semantic target" via `(group_elem_idx, edge_direction)`
  and applies a binary-search type filter against `type_indices[T]`
  (sorted by construction).
  On Wikidata:
    - museums-by-works (with ORDER BY): 15 s → 108 ms (140×)
    - most-eponymed-globally: 122 s timeout → 169 ms (~720×)
    - top-influencers: 122 s timeout → 26 ms (~4700×)
    - typed-target without ORDER BY (LIMIT only): 13.3 s → 110 ms (~120×)
    - untyped-target without ORDER BY (LIMIT only): 14.3 s → 535 ms (~27×)
  No on-disk format change; all-Wikidata graph rebuild not required.
  Differential-test queries `edge_groupby_typed_target_top_k` and
  `edge_groupby_typed_target_no_orderby` added to the corpus to gate
  both branches.
- **`MATCH...WITH count(...)` aggregations now use the fast path
  post-pattern-reversal too.** The third instance of the same
  position-only `group_elem_idx == 2` bug was in
  `try_fast_with_aggregate_via_histogram` (the executor's fast path
  for `FusedMatchWithAggregate`). After
  `optimize_pattern_start_node` reverses `(a)-[:E]->(b:T) WITH b,
  count(a)` to start at the typed node, group_elem_idx becomes 0
  and the histogram path silently bailed despite
  `lookup_peer_counts` serving both shapes. Same direction-aware
  predicate fix as the RETURN-aggregate paths, plus
  `count(<edge-var>)` now fuses through the WITH-aggregate gate too.
  c7's `MATCH (n)<-[r]-() WITH n, count(r)` now reaches
  `FusedMatchWithAggregate` (still bounded by the absence of a
  global-in-degree histogram for untyped edges; that's a follow-up
  workstream). Differential test
  `edge_groupby_match_with_aggregate_typed_target` added.
- **`count(<edge-variable>)` now fuses into `FusedMatchReturnAggregate`.**
  `MATCH (paper)<-[r:CITES]-(citing) RETURN paper.title, count(r)` is
  the natural shape for the Wikidata citation graph, but the gate at
  `fuse_match_return_aggregate` only accepted `count(<other-node-var>)`
  — `count(r)` for the edge variable bailed silently. Semantically
  equivalent for a 3-element pattern (each edge is one peer binding),
  so the fix accepts both. On Wikidata: most-cited scholarly articles
  drops from 198s timeout to 28s (~7×, bounded by `lookup_peer_counts`
  HashMap construction over P2860's hundreds-of-millions of edges).
  Differential test `edge_groupby_count_edge_variable` added.
- **`ORDER BY <agg-expr>` now fuses equivalently to `ORDER BY <alias>`.**
  `fuse_match_return_aggregate`'s top-K absorption matched only
  `ORDER BY <alias-name>` (a Variable expression matching a RETURN
  alias). Writing the same query as `ORDER BY count(x)` (an expression
  duplicating a RETURN item's expression) left ORDER BY + LIMIT
  unfused in the pipeline, so the fused MATCH-RETURN-aggregate
  produced every distinct peer's row (245k for `:P138` on Wikidata)
  before downstream OrderBy + Limit trimmed to k. On Wikidata this
  cost 8 s vs the alias-form's 175 ms for the same query. Absorption
  now also matches via `expression_to_column_name` so both forms
  fuse. Differential test
  `edge_groupby_orderby_expression_form` added.
- **Group-by-source aggregations now use a fast path too.** Queries
  shaped `MATCH (h:T)-[:E]->(other) RETURN h, count(other) ORDER BY
  count(other) DESC LIMIT k` (e.g. "humans with most awards") were
  hitting the slow node-centric scan, which on Wikidata's
  `:human` (13.4M nodes) timed out at 30s/75s with random mmap reads
  thrashing the page cache. The fast path now detects "group is
  semantic source" via `(group_elem_idx, edge_direction)` —
  `(0, Outgoing)` for the user-written form, `(2, Incoming)` for the
  post-reversal form — and computes source-keyed counts on the fly via
  `count_edges_grouped_by_peer(.., Direction::Incoming)`, a sequential
  scan of `edge_endpoints`. Sequential I/O is the right shape for this
  workload (see `feedback_disk_io_patterns.md`). On Wikidata,
  `humans-with-most-awards` drops from 30-75s timeout to 54s answer —
  bounded by the sequential edge-scan I/O ceiling, not by query-engine
  inefficiency. Smaller graphs (social_graph) see sub-millisecond
  results. Differential test
  `edge_groupby_source_typed` added.

## [0.9.11] — 2026-05-07

### Docs

- **Getting Started rewritten** to lead with bulk-load
  (`add_nodes` / `add_connections` from DataFrames) instead of
  three single-row `cypher("CREATE ...")` statements. The old
  ordering misrepresented the day-1 workflow — every real project
  loads data through the columnar path. Single-CREATE demoted to
  an "Ad-hoc inserts" callout. Adds the missing
  `pip install "kglite[mcp]"` line and a preview of the bundled
  CLI + `source_root:` one-liner.
- **New audience-ranked guide index** at `docs/guides/index.md`
  groups the 14 how-to guides by intent: load-bearing path
  (data-loading → cypher → mcp-servers), domain-specific
  (code-tree, spatial, timeseries, etc.), power-user, and
  "if you want to know why". Sidebar toctree reordered to match.
- **MCP Servers guide polished**: 3-line "What's MCP?" intro
  with link to modelcontextprotocol.io; "Five tools from one yaml
  line" preview pulled into the Quick Start so the source_root:
  ROI lands before the Claude Desktop config; new "Common boot
  errors" subsection with eight error→fix mappings + exit-code
  reference; two new rows in the manifest-vs-fork decision table
  for the 15-row output cap and FORMAT CSV constraints.
- **README examples reordered**: `conference_graph_mcp.yaml`
  promoted to the first example as the canonical zero-Python
  starter post-0.9.10. `legal_graph.py` reframed as the
  imperative-API alternative; `mcp_server.py` demoted to
  fork-only-when-manifest-can't.
- **`recipes.md` "Top-K Nodes by Centrality"** now shows the
  `CALL pagerank() YIELD node, score` Cypher form alongside the
  inherent `graph.pagerank(top_k=10)` Python form — manifest /
  MCP / agent contexts all reach KGLite through `cypher()`, so
  the Cypher form is the agent-friendly default.

## [0.9.10] — 2026-05-07

### Added

- **YAML manifest for `kglite-mcp-server`.** Drop a
  `<graph_basename>_mcp.yaml` next to your graph file (or pass
  `--mcp-config FILE`) and the bundled CLI auto-loads it at
  startup. Three tiers, all optional:
  - `source_root: ./data` (or `source_roots: [./a, ../b]`)
    auto-registers `read_source` / `grep` / `list_source` tools
    sandboxed to those directories. Backed by the
    [`mcp-methods`](https://github.com/kkollsga/mcp-methods)
    Rust-extension package — ripgrep crates, gitignore-aware,
    parallel walker, with internal grep so agents can search
    files too large to dump into context. Paths resolve relative
    to the yaml's directory; `../` is allowed.
  - `tools: cypher: |` blocks register parameterised Cypher as
    named MCP tools. JSON Schema `parameters:` drives the
    synthesised input schema, every `$param` reference is
    validated at server startup against the schema, and
    function signatures get built dynamically so FastMCP's
    introspection produces clean tool schemas on the wire.
  - `tools: python: ./tools.py` + `function: name` loads custom
    Python hooks. Two-signal trust gate: requires both
    `trust.allow_python_tools: true` in the yaml AND
    `--trust-tools` on the CLI. Either alone refuses to load.
- **`mcp-methods` and `PyYAML` added to the `[mcp]` extras** —
  `pip install "kglite[mcp]"` now pulls them automatically.
- **`--mcp-config FILE`** explicit-override flag and
  **`--trust-tools`** Python-hook authorisation flag added to
  `kglite-mcp-server`.
- **`name:` and `instructions:` manifest fields** override the
  default FastMCP server-info values when set.

### Docs

- MCP Servers guide rewritten around the manifest as the primary
  customisation path. Forking `examples/mcp_server.py` is now
  framed as the escape hatch for needs the manifest can't cover
  (custom CSV-export logic, FastMCP middleware, alternative
  transports). Includes a complete manifest example for a
  conference-catalog graph.
- `KnowledgeGraph.explain_mcp()` (the agent-facing XML quickstart)
  rewritten for the bundled CLI + manifest path. Previously
  recommended forking a server file and pointed agents at the wrong
  install / import.
- New `examples/conference_graph_mcp.yaml` — copy-paste-ready
  reference manifest demonstrating all three tiers (`source_root`,
  inline cypher, python hooks) with comments explaining each.

## [0.9.9] — 2026-05-07

### Added

- **`kglite-mcp-server` console script.** The MCP server that
  exposes any `.kgl` graph as a Cypher tool now ships as part
  of the package — `pip install "kglite[mcp]"` and run
  `kglite-mcp-server --graph my.kgl`. Same surface as
  `examples/mcp_server.py`, which is now a thin wrapper around
  the new `kglite.mcp_server.main` entry point and lives on as
  the fork-this template for adding custom tools.

### Changed

- `add_nodes` and `add_connections` now emit a `UserWarning`
  whenever the report flags **any** errors, not just when rows
  were skipped. Previously, follow-up loads with type mismatches
  set `has_errors=True` on the report but stayed silent; you had
  to inspect `last_report()` to notice. Silent partial successes
  were a recurring footgun.

### Docs

- New "Loading in passes" section in the Data Loading guide
  covering the second-`add_nodes` contract (static-then-
  timeseries, schema-then-enrichment), what carries over between
  calls, and a `conflict_handling` cheatsheet.
- New "Hierarchies" section disambiguating `set_parent_type`
  (type-level disclosure for `describe()`) from explicit
  `PARENT_OF`-style edges (instance-level tree structure that
  Cypher `*` walks). They look similar; they aren't.
- MCP Servers guide rewritten to lead with the bundled CLI:
  `pip install "kglite[mcp]"` → `kglite-mcp-server --graph X.kgl`.
  Claude Desktop / Claude Code configs use `"command":
  "kglite-mcp-server"` directly. Tutorial body is now framed as
  the customisation path for forks of `examples/mcp_server.py`.
- `add_nodes` / `add_connections` reference sections in the Data
  Loading guide reframed as parameter tables (no longer redundant
  with the walkthrough).

## [0.9.8] — 2026-05-07

### Fixed — `add_nodes` no longer clobbers the title alias on follow-up calls

A repeated `add_nodes(...)` on an existing node type without
`node_title_field` (the canonical pattern when layering
timeseries onto static rows, the example in the docstring)
silently rebound `title_field_aliases[node_type]` to the
`unique_id_field`. Later Cypher queries for `s.id` then resolved
to the stored title, returning the title string in place of the
id. The only visible signal was `title_alias="id"` in
`describe()` output.

The alias map is now written only when the caller explicitly
passes `node_title_field`. Surfaced by a real-world graph build
where the "static rows once, timeseries on top" pattern hit it.

### Added — `describe(sample_truncate=…)` to control title truncation in the XML

Sample values, sample node titles, and sample edge attributes
emitted by `describe()` get truncated at 40 chars by default to
keep prompts compact. Pass `describe(sample_truncate=None)` to
emit them in full when you want full titles in an LLM context
and have the budget for it; pass an integer for a custom
threshold. The knob only affects rendering — stored data is
always full-precision and accessible via Cypher.

### Docs

- New "End-to-end walkthrough" section at the top of the Data
  Loading guide — shape tables → `add_nodes` → `add_connections`
  → Cypher → save/load — so the README's "DataFrames in" pitch
  lands on a single connected story instead of scattered
  reference snippets.
- AI Agents guide now documents the `id_alias` / `title_alias`
  attributes on `<type>` elements and the new `sample_truncate`
  knob.

## [0.9.7] — 2026-05-04

## [0.9.7] — 2026-05-04

### Jupyter ergonomics — `wikidata.open()` is now process-cached

`wikidata.open(workdir)` previously did a fresh disk-graph load
(~350 MB in-memory state on the 124M-node truthy graph) on every
call, even when the same workdir had already been opened in the
same process. Repeating the call in a Jupyter notebook (the typical
"rerun-cell" workflow) accumulated RSS until the kernel ran out of
room and started swapping or hung after a dozen iterations.

`open()` now holds a process-local cache keyed by
`(canonical workdir path, entity_limit_millions)` →
`(KnowledgeGraph, disk_graph_meta.mtime)`. Cache hits return the
same `KnowledgeGraph` instance the prior call handed back. The
cache invalidates automatically when:

- the on-disk graph is rebuilt (mtime advances)
- `force_rebuild=True` is passed
- the user calls the new `wikidata.cache_clear()` (mirrors
  `functools.lru_cache`'s pattern; returns count of entries
  dropped)

Memory-mode opens skip the cache entirely — they're meant to be
reproducible rebuilds.

Verified: 3 × `wikidata.open(WORKDIR)` in one process → 426 MB
once, then flat. Same instance returned (`g1 is g2 is g3`).

### Examples — `examples/wikidata_disk.py` rewritten

Replaces the 259-line build-plus-bench harness with a 35-line
realistic walkthrough: download/load the dump via
`wikidata.datasets.wikidata.open()`, print graph size + a
"name+type lookup → awards" demo for Albert Einstein. Each step
shows its wall time so users see what each operation costs. Full
benchmark version preserved in `dev-documentation/`.

### Known issue (not fixed in 0.9.7)

`MATCH ({nid: $param})-[:T]->()` on the 124M-node Wikidata graph
runs ~12,000× slower than the literal-form
`MATCH ({nid: 'Q937'})-[:T]->()` (~65 seconds vs ~5 ms) and
allocates ~3 GB of RSS per call. The index-lookup planner pass
treats `Expression::Parameter` as a non-indexable predicate when
the property name is the global id alias. Workaround: inline the
literal value (Cypher injection-safe when the value came from the
graph itself) or fold into a single multi-MATCH query that anchors
once on a typed node pattern. Tracked for a future release.

585 cargo, 2345 pytest, 97/97 parity, lint clean.

## [0.9.6] — 2026-05-03

### Cypher correctness fix — `collect()[slice]` over `OPTIONAL MATCH` raised a spurious aggregate-context error

Cypher of the shape

```cypher
MATCH (n {id: $id})
OPTIONAL MATCH (n)-[:T]-(x)
WITH n, collect(DISTINCT x.title)[0..3] AS first_three
RETURN n.title, first_three
```

failed at runtime with `Aggregate function 'collect' cannot be used
outside of RETURN/WITH`, even though the `collect` call was clearly
inside a `WITH` projection. The same expression on a non-`OPTIONAL`
`MATCH` worked fine. Caused users to rewrite the query as
`WITH ... collect(...) AS xs RETURN xs[0..3]`, which is identical
semantically but unobvious as a workaround.

**Root cause.** `aggregates_only_count` in
`planner/fusion.rs` — the gate that decides whether the OPTIONAL-MATCH
count fusion can absorb a projection — recursed into arithmetic and
function-call nodes but fell through to `_ => true` on `ListSlice`,
`IndexAccess`, `ListComprehension`, and `Case`. So
`collect(x)[0..3]` (a `ListSlice` wrapping a `FunctionCall`) was
wrongly classified as "all aggregates inside are count-shaped" and
the count-fusion accepted it. The fused executor then ran
`evaluate_expression` per row on the substituted-but-still-
containing-`collect` projection, and the runtime correctly rejected
the per-row aggregate call.

**Fix.** `aggregates_only_count` now recurses into the same wrapper
expression variants that `ast::is_aggregate_expression` walks — slice,
index, list comprehension, case, expression-property-access,
map-literal. `collect()[…]`, `collect()[i]`, `collect()` inside a
`CASE`, and other "aggregate inside a wrapper" shapes all bail
fusion correctly and route through the materialised aggregate
evaluator.

The same fix incidentally closes a related class of broken queries:
`sum(x.prop)`, `min(...)`, `max(...)`, etc. wrapped by
`ListSlice`/`IndexAccess`/etc. over an `OPTIONAL MATCH` were also
silently broken pre-0.9.6 — never explicitly tested but caught by
the new corpus entries.

Differential corpus regressions (`tests/test_cypher_differential.py`):
`collect_slice_over_optional`, `collect_index_over_optional`,
`sum_over_optional`.

### Cypher perf fix — `LIMIT N` not pushed into grouping aggregator

Hub-anchored `OPTIONAL MATCH + collect/aggregate + LIMIT N` queries
on the 124M-node Wikidata graph were unnecessarily slow:

```cypher
MATCH (x)-[:P31]->(hub {nid: 'Q11424'})        -- film hub, 340k inbound
OPTIONAL MATCH (x)-[:P27]->(country)
RETURN x.title AS x, collect(DISTINCT country.title) AS countries
LIMIT 15
```

Materialised the full 340k MATCH expansion + 340k OPTIONAL P27
expansions + 309k group buckets, then truncated to 15 at the very
end. Cold: 64s. Warm: 547ms.

**Root cause.** The materialised aggregator drained every group
key before any downstream `LIMIT` clause looked at the rows.
There was no path for a literal `LIMIT N` to inform the grouping
loop that it could stop creating new groups after `N` distinct
keys.

**Fix.** New planner pass `push_limit_into_aggregate` (registered
in `PASSES` between `push_limit_into_match` and
`push_distinct_into_match`). When the projection clause has both
group keys and aggregates AND the next clause is a literal
`LIMIT N` (no intervening `ORDER BY`, no `DISTINCT`, no `HAVING`),
the pass stamps a `group_limit_hint` on the `ReturnClause` /
`WithClause`. The aggregator then uses a 2× safety margin during
the surrogate-key grouping pass (NodeIndex→Value collisions can
collapse groups during the resolve step) and truncates to the
exact `N` after resolve. Rows for already-collected keys
continue to feed their aggregates so `collect()` / `sum()`
complete correctly for the surviving groups.

ORDER BY between projection and LIMIT correctly disables the
optimisation — needed every group to find the top N. The
existing LIMIT clause stays in the plan as a hard safety cap.

**Verification on the 124M-node graph (warm steady state):**

| State | Latency | Profile shape |
|-------|---------|---------------|
| Pre-fix  | 547 ms | `Return rows_in=340688 rows_out=309004 → Limit 15` |
| Post-fix | 257 ms | `Return rows_in=340688 rows_out=15 → Limit 15` |

The remaining 257ms is genuine MATCH + OPTIONAL fanout work
(340k inbound `:P31` + 340k OPTIONAL `:P27` expansions). True
streaming through MATCH itself would need a larger refactor and
is left for a future release.

Differential corpus regressions: `limit_into_aggregate_collect`,
`limit_into_aggregate_count`, `limit_with_order_by_no_pushdown`.

### Investigated, not fixed — first-MATCH cold-mmap warmup spike

User reported ~700ms latency on the first `cypher` call after
process start against the same 124M-node graph; subsequent queries
sub-100ms. Reproduced locally at ~1010ms on the same graph; <5ms
on a 16M-node graph regardless of whether we'd touched it that
session. Not a code bug — the first MATCH page-faults the
`id_indices`, `node_slots`, and column-store mmap regions for the
queried type, and the cost scales with graph size + how cold the
OS page cache is.

Existing mitigation: `KGLITE_PREFETCH=1` env var triggers
`madvise(MADV_WILLNEED)` against hot regions at load time, paid
upfront instead of on the first user query. Recommended for MCP
servers that load multi-GB graphs at startup and want predictable
first-query latency.

585 cargo, 2345 pytest, 97/97 parity, lint clean.

## [0.9.5] — 2026-05-02

### Disk-mode regression fix — `register_connection_type` flipped the conn-type cache into a half-built state

On a disk graph that's been loaded from disk (`kglite.load(path)`),
the very first `add_connections` call after load broke MATCH queries
on every other edge type. `MATCH ()-[:OF_DISCOVERY]->()` returned 0
rows even though the edges were still on disk and unanchored
`MATCH ()-[r]->() RETURN type(r)` still listed them. Surfaced on
the Sodir prospect graph's load-then-enhance flow: the moment the
re-enhance pass added its first new edge type, every subsequent
FK-edge join (`OF_DISCOVERY`, `OF_FIELD`, `OF_PROSPECT`, `IN_PLAY`,
`HAS_DEPOSIT_PROSPECT`, `HAS_PLAY`) silently produced 0 rows. The
single-pass `build_sodir_graph.py --storage disk` flow was unaffected
because the cache was in the right state during initial build.

**Root cause.** `DirGraph.connection_types` is a `HashSet<InternedKey>`
that powers the O(1) fast path of `has_connection_type`. The set is
built from `connection_type_metadata.keys()` via
`build_connection_types_cache()` — but that function was only called
from `read_graph_v3` (the `.kgl` v3 loader), never from
`load_disk_dir` (the disk-graph loader). On a freshly-loaded disk
graph the set was therefore empty, and `has_connection_type` correctly
fell through to the metadata-fallback branch which returns true for
every edge type in the loaded metadata.

`register_connection_type(NEW)` then unconditionally inserted NEW
into the empty set. The next `has_connection_type(EXISTING)` call
hit the fast path: "set non-empty? consult cache" — which now only
contained NEW, so it returned false for every existing edge type.
The pattern matcher's early-exit ("skip iteration when the conn type
doesn't exist") fired for every typed MATCH on existing edges.

**Fix.** Two complementary patches:

1. `load_disk_dir` now calls `build_connection_types_cache()` after
   loading metadata, mirroring the v3 loader. Keeps the cache
   authoritative throughout the lifetime of any loaded disk graph.

2. `register_connection_type` lazy-builds the cache from
   `connection_type_metadata` when called against an empty set.
   Closes the same hole defensively for any future code path that
   could leave the cache empty before calling it.

**Verification.** New `bench/cross_mode_pipeline.py` harness runs an
enhance-style sequence (load → create_index → add_connections of a
new edge type → SET → re-read baseline queries at every step) on
legal + sodir × {memory, mapped, disk}. Pre-fix:

    legal disk:  step3_after_add_connections → cites_total: 0 (baseline 592305)
    legal disk:  step3_after_add_connections → section_of_total: 0 (baseline 23585)
    sodir disk:  step3_after_add_connections → of_discovery_join: 0 (baseline 107)
    sodir disk:  step3_after_add_connections → of_field_join: 0 (baseline 2280)
    sodir disk:  step3_after_add_connections → of_prospect_join: 0 (baseline 21857)

Post-fix all per-step counts match baseline across all three modes
on both graphs. Re-running the Sodir `enhance(g)` on a loaded disk
graph (which was the user-facing surface of this bug) now matches
the standalone-build counts: 97 discoveries enriched, 139 fields,
126 production profiles, 4345 prospects tagged, 48 plays calibrated,
2406 prospects calibrated.

Regression test:
`tests/test_disk_mutation_roundtrip.py::test_register_new_conn_type_preserves_existing_type_lookups`
loads a small disk graph with WORKS_AT edges, runs `add_connections`
of a new FRIENDS_WITH type, and asserts MATCH on WORKS_AT still
finds all 5 edges. Test fails with the fix reverted.

### Test harness — pipeline-shaped consistency

`bench/cross_mode_pipeline.py` (new). Closes the gap that let the
sequence-of-operations bug class slip through the earlier single-op
harnesses (`cross_mode_table.py`, `cross_mode_consistency.py`):
record per-step row counts during a multi-op pipeline, compare
each step to a fresh-load baseline, flag the exact step + query
where any cell drifts. Same code shape that caught the bug in this
release; held in `bench/` (gitignored) for ad-hoc use.

585 cargo, 2338 pytest, 97/97 parity, lint clean.

## [0.9.4] — 2026-05-02

### Disk- and mapped-mode regression fix — Cypher SET silently no-oped on mmap-backed ColumnStores

Cypher `SET` on nodes whose property storage was an mmap-backed
`ColumnStore` (the path `load_ntriples` always takes for mapped /
disk targets, regardless of input size) reported success at the
clause boundary — `MATCH (a) SET a.x = 1 RETURN count(a)` returned
the expected `count = 1` — but a follow-up `RETURN a.x` came back
`None`. The clause matched, the writer reported success, but the
write was invisible on read. Same pattern for `SET a.title = …`:
the title column held the old value forever.

**Root cause.** `ColumnStore::from_mmap_store` builds a store with
`schema = TypeSchema::new()` (empty), `columns = Vec::new()`,
`title_column = None`, and `mmap_store = Some(...)` carrying the
real data. Every read method (`get`, `get_title`, `str_prop_eq`,
`row_properties`) short-circuited to the mmap-backed read at the
top of the function — but `set` and `set_title` wrote into the
local `self.columns` / `self.title_column` fields the readers
bypassed. The fix to 0.9.2's disk-mode SET visibility regression
(per-clause `flush_pending_writes` + `sync_column_stores_from_disk`)
landed the writes in the right place; readers just couldn't see
them once the store was mmap-backed.

The bug only surfaced for `load_ntriples`-built graphs: `add_nodes`
followed by save+reload produces ColumnStores whose blobs reload as
in-memory `columns: Vec<TypedColumn>` with `mmap_store: None`, so
the read short-circuit never fires there. Sodir's
`from_blueprint` build → save → reload path passes the existing
SET-roundtrip parity tests for the same reason.

**Fix.** `get` / `get_title` / `str_prop_eq` / `row_properties` now
consult the in-memory overlay first and only fall through to the
mmap-backed read when no override exists for that (row, key).
`set_title` lazy-promotes the mmap-backed title column into a
Mixed in-memory column on first override, so the dense title column
isn't allocated up-front (avoiding the multi-million-row
materialisation that an eager promotion would force on Wikidata).

**Verification on the cross-mode consistency harness**
(`bench/cross_mode_consistency.py`, runs the same edit + read
queries against every (graph, mode) cell and SHA-256s the rows):

| graph | mode coverage | pre-fix `verify_edit` digest | post-fix |
|-------|---------------|------------------------------|----------|
| legal      | memory + mapped + disk | identical | identical ✓ |
| sodir      | memory + mapped + disk | identical | identical ✓ |
| wiki100m   | memory + mapped + disk | memory=`278f…9516`, mapped/disk=∅ | all `278f…9516` ✓ |
| wiki500m   | mapped + disk          | mapped/disk=∅ vs absent canonical | both `278f…9516` ✓ |

Read-only queries (simple / medium / complex) were already
identical pre-fix; this release closes the SET-visibility gap.

Regression test `test_cypher_set_visible_on_mmap_backed_columnstore`
in `tests/test_disk_mutation_roundtrip.py` builds a tiny inline
`.nt` with 5 Q-entities, loads it under mapped + disk, applies SET
to both a new property name AND `title`, and asserts the writes are
visible. Test fails with the fix reverted (confirmed by
`git stash`-ing `column_store.rs` only).

585 cargo, 2337 pytest, 97/97 parity, lint clean.

## [0.9.3] — 2026-05-02

### Disk-mode regression fix — parallel projection data race on `node_arena`

Disk-mode queries that reached `node_weight` materialization through
the Cypher executor's projection phase produced **non-deterministic
results across runs**, ranging from silently wrong row counts (Bug A
in the 0.9.2 disk regression report — ~13% of `NEAREST_AFEX_HUB`
edges and ~2% of `IN_AFEX_AREA` edges silently dropped on the Sodir
prospect graph) to use-after-free segfaults with `BUG: InternedKey N
not found in StringInterner` lines on stderr (Bug B in the same
report).

**Root cause.** `DiskGraph::node_arena` was an
`UnsafeCell<Vec<NodeData>>` and its module-level SAFETY block claimed
the single-threaded-query contract guaranteed exclusive access. In
reality the Cypher executor's projection phase
(`return_clause::project_row`) runs `evaluate_expression` under
`par_iter_mut` once `result_set.rows.len() >= RAYON_THRESHOLD` (256),
and any expression that reaches `node_weight` — spatial functions
(`centroid` / `contains` / `distance`), the spatial-fallback branch
of `resolve_property`, the `NodeRef`-in-projected branch — pushed
onto the unguarded `Vec` from sibling Rayon tasks. A `push` that
triggered realloc invalidated the `&NodeData` references already
returned to other tasks; downstream reads either (a) saw the wrong
row's properties (Bug A — sometimes the polygon parsed cleanly to a
near-but-different centroid and the row simply missed its hub) or
(b) followed a dangling pointer into the freed allocation (Bug B —
sometimes the dangling slot decoded to an `InternedKey` the interner
didn't know, surfacing the BUG line; with worse timing, SIGSEGV).

The non-determinism explains why the fresh build → save → reload →
read flow showed different counts than the in-process build: the
in-process flow used the warm in-memory column stores, while the
load-then-read flow re-materialized through `node_weight` under the
parallel projection.

**Fix.** `node_arena` is now `Mutex<Vec<Box<NodeData>>>`, mirroring
the long-standing pattern used by `edge_arena`. The Box gives stable
heap pointers that survive Vec growth; the Mutex serialises pushes.
The `&NodeData` references handed back are valid for the lifetime of
`&self` because the arena is only cleared via `clear_arenas`
(`&mut self`) or `reset_arenas` (called between top-level queries).

**Verification.** Fresh disk build of the Sodir prospect graph
(557k nodes) now produces edge counts that match the in-memory
build byte-for-byte:

| Edge type             | default | disk (pre) | disk (post) |
|-----------------------|---------|------------|-------------|
| `IN_AFEX_AREA`        |    886  |       866  |        886  |
| `NEAREST_AFEX_HUB`    |  5,881  |     5,134  |      5,881  |
| `IN_BLOCK`            |  7,983  |     7,983  |      7,983  |
| `IN_STRUCTURAL_ELEMENT`|  6,763  |     6,763  |      6,763  |

The `load → enhance → save` workflow that was reported as
segfaulting on disk now completes cleanly with no `BUG: InternedKey`
lines on stderr.

Two regression tests cover the race surface:
`test_disk_parallel_projection_node_weight_is_race_free` (8 repeated
runs of `centroid` projection on a 600-node disk graph must report
identical counts) and
`test_disk_parallel_projection_no_interner_corruption` (asserts no
`BUG: InternedKey N not found` lines appear on stderr across
repeated parallel projections).

585 cargo, 2333 pytest, 97/97 parity, lint clean.

## [0.9.2] — 2026-05-02

### Disk-mode regression fix — property visibility after blueprint build

Disk-mode `from_blueprint(...)` left freshly-built graphs in a
state where every Cypher property read returned NULL.
`MATCH (p) RETURN count(p)` worked (slot enumeration), but
`RETURN p.x` for any property — even `id` and `title` — returned
None. The Sodir prospect-graph build on disk surfaced the failure
shape: every Stage 2.x derived edge was 0, every Stage 3 SET
cascade collapsed because the read-back of the previous stage's
writes returned 0 rows. Single-process build → enhance → save was
unusable on disk; default and mapped modes were unaffected.

Two related sync gaps between `DirGraph.column_stores` and
`DiskGraph.column_stores`:

- **batch.rs deferred-columnar pass after creates.** Disk's
  `add_nodes` populated `graph.column_stores` (DirGraph-side) and
  updated each slot's `row_id`, but never mirrored to
  `disk_graph.column_stores` (where disk reads through
  `node_weight` / `get_node_id` / `get_node_title`). The existing
  `sync_disk_column_stores()` only fired when the chunk had
  UPDATEs, never on creates-only. Fix: after the deferred-columnar
  loop, call `graph.sync_disk_column_stores()` when disk-backed.

- **write.rs after Cypher mutation flushes.** Each SET / REMOVE /
  MERGE clause calls `flush_pending_writes` (0.8.41) which drains
  `node_mut_cache` into `disk_graph.column_stores`. But
  `graph.column_stores` stayed stale. A subsequent `add_nodes`
  (which now calls `sync_disk_column_stores` per the first fix)
  would clobber disk's post-flush state with the pre-flush
  DirGraph snapshot — silently losing the SET's effects on the
  multi-stage `SET → add_nodes → read` pipeline. Fix: after every
  `flush_pending_writes` (per-clause + end-of-query), also call
  `graph.sync_column_stores_from_disk()` to mirror the post-flush
  disk state back to DirGraph.

**Verification on the Sodir prospect-graph build (557k nodes, 47
edge types):**

- All Stage 2.x derived edges populated (was all 0): IN_BLOCK
  7,983, IN_STRUCTURAL_ELEMENT 6,763, IN_AFEX_AREA 779,
  NEAREST_AFEX_HUB 5,560.
- All Stage 7 derived edges populated (was all 0):
  HC_IN_FORMATION 850, ENCLOSES 10,588, PLAY_HAS_FORMATION 248,
  DRILLED_IN_PLAY 2,627.
- Stage 3.6 value_basis distribution matches the in-memory
  baseline byte-for-byte: estimate=2754, realized=546, dry=1437,
  deflagged=1577, unscored=460. Pre-fix every prospect collapsed
  into `unscored`.

All three storage modes — default, mapped, disk — now produce
identical results on a 16-case integration smoke test that exercises
every 0.9.x gate item plus the multi-stage SET / `add_connections`
/ save+reload pipeline.

585 cargo, 2333 pytest, 97/97 parity, lint clean.

## [0.9.1] — 2026-05-02

### Breaking change in 0.9.0 (Rust crate API) — late-breaking notice

`MemoryGraph` and `MappedGraph` no longer implement
`std::ops::DerefMut` to their inner `StableDiGraph`. Code that
mutated through auto-deref now fails to compile and must use one
of:

- the explicit accessor: `g.inner_mut().add_node(data)`, OR
- trait dispatch: `use kglite::graph::storage::GraphWrite;
  g.add_node(data)`.

**Why**: auto-deref-via-DerefMut shadowed `GraphWrite::add_node`
(and peer mutation methods), causing calls to bypass
`MappedGraph::invalidate_property_index()` that the trait impl
runs first. Removing `DerefMut` converts a silent stale-index bug
into a compile-time error. Read-only `Deref` is retained — read
calls (`g.node_count()`, etc.) are unchanged.

**Python users**: no impact. The PyO3 boundary is unchanged.

This was already in 0.9.0 (commit `bcb0bb7`, "chore(storage): drop
DerefMut on Memory/MappedGraph") under "Hygiene & test coverage" —
this entry simply re-flags it for downstream Rust embedders who
might miss it under that subhead.

### Blueprint diagnostics

- **`from_blueprint(verbose=True)` reports actual graph edge counts.**
  Pre-fix, the verbose log printed an accumulated input-row count
  from the blueprint pipeline; with default Update conflict
  handling, repeated `(src, tgt)` pairs across multiple section
  ingests over-counted vs `MATCH ()-[r]->() RETURN count(r)`. Easy
  to mistake for a save regression. Now queries
  `graph.get_edge_type_counts()` (the post-build truth) and reports
  those numbers; when input and graph counts diverge, the line is
  annotated `[T]: N edges (M input rows, K deduped)` so users see
  both. Backward-compatible for zero-duplicate datasets.
- **Warning capture documented.** Blueprint warnings (target node
  not found, null FK, etc.) hit stderr by default. Python's
  standard `logging.captureWarnings(True)` routes the Rust-emitted
  UserWarnings into the `py.warnings` logger where any standard
  handler (file, rotating, stream) can catch them — no `2>&1`
  shell redirect needed. The `from_blueprint` docstring now
  documents the pattern with a copy-paste-ready file-capture
  example. Pinned by `test_logging_capture_warnings_pipeline`.

### Documentation

- **CYPHER.md** gained a "Duration semantics" subsection explaining
  the calendar-vs-clock component split (months/years stay
  separate from days/hours/minutes/seconds), with worked examples
  for `duration.between()` and `DateTime ± Duration`. Postgres
  `interval` users will want this. Pinned by an anchor test
  (`test_duration_between_cypher_md_example`).

## [0.9.0] — 2026-05-02

### Cypher dialect — gate items

- **§5 Integer division (Neo4j-standard).** `1967 / 10` now returns
  `196` (truncated `Int64`), matching openCypher / Neo4j. Float
  promotion only when at least one operand is a float. Negatives
  truncate toward zero (`-7 / 2 → -3`). Modulo behavior preserved.
- **§2 NULLS FIRST / NULLS LAST in ORDER BY.** New
  `ast::NullsPlacement` + parser support. Default placement is
  Neo4j 5+: NULLS LAST for ASC, NULLS FIRST for DESC. Plumbed
  through both the in-memory sort path and the `heap_top_k`
  streaming operator.
- **§3 Stable date function set + Cluster 2 proper Value::Duration.**
  Datetime field accessors `.year/.month/.day/.dayOfWeek/.dayOfYear/.epochSeconds`
  on `Value::DateTime`. New `Value::Duration { months: i32, days: i32,
  seconds: i64 }` variant — calendar units (months/years) and clock
  units (days/hours/minutes/seconds) stay separate, so
  `duration({months: 1, days: 5}).months` returns 1 (not 35
  collapsed to days). `duration()` constructor, `duration.between()`,
  `DateTime ± Duration`, `Duration ± Duration` arithmetic. Sub-day
  precision wired in `seconds`; `Value::DateTime` is still
  `NaiveDate`, so DateTime + Duration discards the seconds
  component for now (Cluster 1 deferred). Duration variant is the
  LAST enum variant — old `.kgl` files load unchanged.
- **§4 Polygon-vs-polygon `contains()` in WHERE.** The fast-path
  spatial filter for `MATCH (a), (b) WHERE contains(a, b)` now
  handles geometry-vs-geometry when neither side has a `Location`
  point. Pre-fix the path silently returned `false` for every
  outer-contains-inner match in polygon-only graphs. Bundled
  MULTIPOLYGON dedupe — single boolean answer per `(a, b)` pair
  regardless of how many components match.
- **§1 Better Cypher error messages.** Every parse error now
  carries `(line N col M)` plus a single-line source excerpt with
  a `^` caret. Position is byte-precise — the tokenizer attaches
  char offsets to every token; parser threads them through;
  `format_parse_error` walks `input.chars()` to compute (line,
  col) on the error path. New `intent_level_rewrite` hook in
  `parser/mod.rs` for "feature not yet implemented" detection
  (currently empty — all named candidates parse successfully).
- **§6 size() over pattern expressions.** `size((:A)-[:R]->(:B))`,
  `size((a)-[:R]->(:B))` (per-row binding), `size((:A)-[:R]->(:B)) >= 2`
  in WHERE. Wraps the existing 0.8.16 count-subquery code path.
  Refactored `parse_exists_patterns` into
  `parse_pattern_subquery_patterns` with a caller-supplied
  delimiter (RBrace for EXISTS/count, RParen for size).

### Hygiene & test coverage

- **Cluster 6: dropped `DerefMut` on `MemoryGraph` / `MappedGraph`.**
  Auto-deref-via-DerefMut shadowed `GraphWrite` trait methods, so
  e.g. `g.add_node(data)` on `&mut MappedGraph` reached petgraph's
  inherent method directly, bypassing
  `MappedGraph::invalidate_property_index()` that the trait impl
  runs first. Removing DerefMut forces explicit `.inner_mut()` or
  trait dispatch — compile-time catch instead of silently-stale-
  index runtime bug. Read-only `Deref` retained.
- **Cluster 6: `node_weight_mut` staging contract documented.**
  Trait method on `GraphWrite` now carries explicit doc that disk
  buffers writes in `node_mut_cache` and callers must call
  `flush_pending_writes()` before any subsequent `&self` read.
  Debug-only assertion in `DiskGraph::node_weight` warns when a
  staged write is shadowed by a read (catches future code paths
  that forget the flush).
- **Cluster 7: deep-traversal path-materialization coverage.** New
  `test_long_chain_traversal_path_materialization_100_hops`
  exercises actual path enumeration across memory + mapped + disk
  via `RETURN b.id` instead of the planner-short-circuited
  `RETURN count(b)` shape that the prior 1,000-hop test used.
- **Cluster 7: datetime accessor golden round-trip.** Two new
  queries in the golden corpus pin `joined_at.year/.month/.day`
  extraction against the social-graph fixture so §3 accessor
  behaviour can't drift unnoticed.

### Internal — pre-existing parity audits cleared

- `.kgl` v3 fixture digest updated (no format change —
  `CURRENT_FORMAT_VERSION` still 3). Fixture-graph hash drifted
  across the 0.8.x → 0.9.0 line as save-path / interner
  refinements landed; backward compatibility verified by loading
  pre-0.9.0 `.kgl` files cleanly.
- `GraphBackend` enum-match audit whitelist refreshed for the
  pre-existing leaks (`column_builder.rs`, `match_clause.rs`,
  `blueprint.rs`, `indexes.rs`).
- Binary-size baseline reset to 0.9.0 (~22.4 MB) with a +10%
  gate. Phase 4 baseline (6.67 MB) was 3.4× off the current build
  — accumulated growth from multi-mode storage + spatial +
  timeseries + code-tree + MCP + Cypher dialect work.
- `god_file_gate`: `match_clause.rs` (2,679 lines) and `fusion.rs`
  (2,923 lines) documented in `GOD_FILE_EXCEPTIONS` with concrete
  0.9.x split plans.
- `mod_rs_purity`: caps bumped on `executor/`, `parser/`,
  `planner/` `mod.rs` files to match the dispatch surface they
  carry; 0.9.x cleanup intent documented.

## [0.8.41] — 2026-05-02

### Cypher executor — Bug 8 followup

- **Fix: SET on an in-memory graph dropped new properties on save.**
  The 0.8.39 master-path fix that routed Columnar SET writes through
  `graph.column_stores` to dodge the per-node Arc-clone storm computed
  the property's `InternedKey` via `from_str()` (just hashing) without
  registering the source string in `graph.interner`. As a result,
  Cypher SET that introduced a *new* property name on an in-memory or
  mapped graph survived in-memory queries but vanished on save+reload,
  printing `BUG: InternedKey N not found in StringInterner` to stderr
  and silently corrupting the saved file. Now registers via
  `graph.interner.get_or_intern(property)` before borrowing
  `column_stores`. Disk mode is unaffected (gated path, separate
  pre-existing bug for that backend).
- **Defense in depth: debug-only invariant in `write_graph_v3`.** Walks
  every `column_store`'s schema before serialization and panics with a
  clear, actionable message if any `InternedKey` doesn't resolve in
  `graph.interner`. Catches the entire class of bug ("writer
  synthesizes an InternedKey without first registering the source
  string") at write time, not at load time on the user's machine. Zero
  release-build cost.
- **Regression**: `tests/test_disk_mutation_roundtrip.py::test_cypher_set_new_property_persists_through_save_reload`
  parameterised over `memory` + `mapped` + `disk` (all three modes
  now covered after the disk-side fix below).

### Cypher executor — disk SET visibility

- **Fix: Cypher SET on a disk-backed graph appeared to no-op on
  in-session reads** until the next `&mut self` op (e.g. `save()`)
  flushed the staged writes. Disk's `node_weight_mut` stages writes in
  `node_mut_cache` to dodge the `Arc<ColumnStore>` share-clone storm
  per row; `node_weight` (the read path) reads `column_stores`
  directly and ignored the cache. Save+reload happened to recover the
  data because `clear_arenas` runs as part of save, but every
  in-session read between SET and save returned the pre-SET value —
  silently corrupting any analysis that read SET-staged columns
  back. Affects both new-property SET (`SET n.value_score = …`) and
  existing-property SET (`SET n.age = n.age + 100`). Same query
  shape: `MATCH … SET … RETURN n.prop` returned NULLs for the just-
  written column.
- **Fix shape**: new `GraphWrite::flush_pending_writes(&mut self)`
  trait method, default no-op, overridden on the disk backend to
  call its existing `clear_arenas` (which already does the
  clone-apply-replace flush of `node_mut_cache` /
  `edge_mut_cache` into `column_stores` / `edge_properties`).
  `execute_mutable` calls it after every write clause (SET / REMOVE /
  MERGE) and once at end-of-query so any trailing RETURN's property
  projection — and any next-query read — sees the writes. Memory
  and mapped backends are unaffected (their `node_weight_mut`
  mutates `StableDiGraph` in place, so reads see writes
  immediately).
- **Doesn't affect Sodir** — the prospect graph runs in-memory — but
  unblocks the disk-mode "SET as cached column" pattern for any
  future workstream that uses storage="disk".

## [0.8.40] — 2026-05-01

### Cypher planner

- **Spatial-join fusion: multi-MATCH + `centroid()` probe.** Extends
  `fuse_spatial_join` to recognise the
  `MATCH (a:T1) MATCH (b:T2) WHERE contains(b, centroid(a))` shape (or
  the inverse `contains(a, centroid(b))` — the call decides which MATCH
  is container vs. probe, not pattern position). Previously only the
  single-MATCH cartesian form fired; the multi-MATCH form fell back to
  cross-product + post-filter. Sodir's `IN_STRUCTURAL_ELEMENT` enrichment
  (Prospect → StructuralElement via centroid) drops from ~2 s to ~0.6 s
  on a 6,775-prospect graph; any project doing point-in-polygon
  enrichment via `centroid()` benefits.
- New `Clause::SpatialJoin::probe_kind: SpatialProbeKind` carries
  whether the probe-side point comes from the spatial-config `location`
  (single-MATCH cartesian) or the centroid of the probe's geometry
  (multi-MATCH centroid). The spatial-join executor honours the kind
  when sourcing the per-probe point.
- An optional pre-WHERE between the two MATCHes (e.g.
  `MATCH (p:Prospect) WHERE p.wkt_geometry IS NOT NULL MATCH (s:StructuralElement) ...`)
  is folded into the SpatialJoin's residual predicate so per-pattern
  filters still apply after fusion.
- Regression coverage: `tests/test_cypher_spatial.py::TestSpatialJoin`
  picks up `test_multi_match_with_centroid_probe` (correctness vs. the
  brute-force two-pattern path) and `test_multi_match_centroid_fires_fusion`
  (EXPLAIN must show `SpatialJoin`).

## [0.8.39] — 2026-05-01

### Cypher executor

- **Fixed scalar projection from spatial-function results
  (`centroid(n).latitude` and `WITH centroid(n) AS c RETURN
  c.latitude`).** Previously returned the entire `{latitude,
  longitude}` dict — or `Null` on in-memory graphs — instead of the
  float. Property access on `Value::Point` now extracts the named
  field via a new `point_field()` helper, applied in both
  `Expression::ExprPropertyAccess` and the `resolve_property`
  projected fallback. Accepted aliases: `latitude`/`lat`/`y` and
  `longitude`/`lon`/`lng`/`long`/`x`. The canonical
  `point(centroid(n).lat, centroid(n).lon)` idiom now composes in a
  single Cypher query.
- **Fixed spatial-predicate failure on partial-coverage typed sets.**
  When a typed node set has spatial config but only a fraction of
  rows have geometry data populated (real-world example: 312 of 469
  AfexAreas in the Sodir graph have `wkt_geometry IS NULL`),
  `WHERE contains(a, point(lat, lon))` errored on the missing rows
  instead of treating them as predicate-false. `contains()` now
  NULL-propagates: row-level missing geometry returns
  `Boolean(false)` so the predicate filters the row out cleanly.
  Same NULL-propagation for `centroid()` / `area()` / `perimeter()`
  (return `Value::Null`). `intersects()` retains the loud error for
  the type-level "no spatial config anywhere" case (preserves
  existing diagnostic test coverage).
- **Fixed superlinear `SET` cost on typed nodes with shared
  columnar storage; OOM at ~1k rows on the Sodir Prospect set.**
  Per-node `Arc::make_mut(store).set(...)` cloned the entire shared
  `ColumnStore` on every write — for 6,775 Prospect nodes sharing
  one store, refcount=N+1 meant a full clone per row, giving
  O(N²) work and ~11 GB transient allocations. SET now routes
  Columnar writes through `graph.column_stores[type]` once per
  batch, then refreshes per-node `Arc<ColumnStore>` handles in a
  single end-of-statement sweep. Verified on the real Sodir graph:
  100 → 4 ms (was 0.81 s, 200×), 500 → 4.4 ms (was 13.1 s,
  3000×), 1000 / 2000 → 3 ms (were OOM). Linear scaling restored.
  Disk-mode graphs use a separate write path and are gated out
  via `graph.graph.is_disk()`.

### Fluent API

- **`create_connections()` now emits a `UserWarning` when called on
  a chained graph view.** The fluent `g.select(...).traverse(...)
  .create_connections(...)` pattern returns a NEW `KnowledgeGraph`
  whose mutations live on a temporary clone (Arc COW); discarding
  the return loses the writes. The warning fires when
  `Arc::strong_count(self.inner) > 1` and points to the two
  workarounds: capture the return (`g = g.select(...)
  .create_connections(...)`) or use the equivalent
  `add_connections(data=cypher_result, ...)`. Docstring also
  updated. Proper structural fix (shared interior mutability)
  deferred — would touch ~hundreds of read-path call sites.

## [0.8.38] — 2026-05-01

### code_tree

- **Re-add `_build` to the ignored-dirs list.** Caught while
  perf-testing 0.8.37 against the kglite repo itself: Sphinx
  `docs/_build/_static/*.js` was getting indexed (26 JS files), even
  though the build artifacts aren't user source. The leading-
  underscore convention is a strong signal of "tool-generated build
  output" (Sphinx, mkdocs, mdBook), distinct from the more ambiguous
  `build` / `dist` / `out` (which stay off the list because
  `dist/bundle.js` may be the user's webpack output they want
  flagged-as-too-large). Verified against `code_tree.build()` on
  the kglite repo: 337 files → 310 files, matching the 0.8.34
  baseline exactly. Build time 0.21s.

## [0.8.37] — 2026-05-01

### code_tree

- **Fixed `code_tree.build()` over-indexing on projects with nested
  `.venv` / `node_modules` / `target` / etc.** Reported by an MCP
  consumer whose codebase graph ballooned 7,620 → 70,605 nodes after
  upgrading to 0.8.36. Root cause: the mixed-language safety net
  added in 0.8.36 walked first-level subdirs recursively to detect
  undeclared languages — but the walk filter for ignored directory
  names (`.venv`, `node_modules`, `target`, `__pycache__`,
  `site-packages`, `venv`, `env`, plus all `.dot` dirs) only applied
  at the top level. A single C-extension source inside a nested
  `.venv` (e.g. numpy in a subprojects venv) attracted the parent
  dir as a supplemental source root and then `parse_directory`
  walked the full venv. The filter is now applied at every depth in
  both the safety-net detection walk and the actual parser walk.
- **Conservative trim of the ignored-names list.** `build`, `dist`,
  `out`, `_build` removed — those names are tooling-dependent (e.g.
  `dist/bundle.js` is sometimes the user's webpack output they want
  indexed-and-flagged as too-large rather than excluded outright).
  Use `max_loc_per_file` to handle oversized build artifacts.

## [0.8.36] — 2026-05-01

### Cypher planner

- **Fixed `mark_fast_var_length_paths` per-target row drop** — closes
  the lone open xfail in the differential harness. The pass set
  `needs_path_info=false` on any unnamed variable-length edge, which
  triggered a target-node-deduping BFS that returned fewer rows than
  Cypher's per-path semantic (e.g., 2 rows where Neo4j returns 3).
  The fix gates the pass on `downstream_is_dedup_safe` — the next
  RETURN/WITH must be `DISTINCT` or its projections must be entirely
  dedup-safe aggregates (`min/max/count(DISTINCT)/collect(DISTINCT)`).
  Plain `RETURN q.name` over var-length now uses the slow per-path
  BFS (correct); users who want the fast path opt in via `RETURN
  DISTINCT q.name` or `count(DISTINCT q)`.
- **Differential harness corpus extended to ~95 query shapes** plus 9
  mutations + 26 per-pass bisection tests. Probed three additional
  rounds (CALL/list-comp/path-ops/multi-WITH/HAVING/coalesce/CASE-in-
  agg/expr-filter/etc.) and surfaced no further divergences after
  fixing the var-length pass. The corpus also now includes
  `var_length_no_var_per_path`, `var_length_no_var_distinct`, and
  `var_length_no_var_count_distinct` as permanent regression tests
  for the fix.
- **Performance cleanups in the new code paths.** `optimize()` now
  returns a process-lifetime empty `HashSet<String>` via `OnceLock`
  instead of allocating a fresh `HashSet::new()` on every call; the
  PyAPI's `cypher()` short-circuits the disabled-passes set
  construction when both `disable_optimizer=False` and
  `disabled_passes=None` (the default), bypassing the validation
  loop entirely on the hot path.
- **Third debug-mode IR invariant**: literal `LIMIT` and `SKIP`
  values must be non-negative. Catches passes that synthesize a
  literal limit hint (e.g. fusion top-K) and forget to clamp at
  zero. Zero release cost (`#[cfg(debug_assertions)]`-gated).
- **Makefile bench targets force `maturin develop --release`.**
  Saved baselines are release-built; running `make bench-compare`
  against a dev build previously showed false ~15× regressions
  across every benchmark. New baseline `0007_post_robustness_pass`
  saved with all the planner refactor + bug fixes in place.
- **Optimizer is now a registry of named passes (`kglite.cypher_pass_names()`).**
  The 25-pass orchestrator at `src/graph/languages/cypher/planner/mod.rs`
  has been refactored from a 40-line inline body into a single
  `const PASSES: &[(&str, PassFn)]` source of truth. Each pass has a
  stable name and a doc-comment with precondition / pattern / rewrite
  / why-bail. Adding a new pass is now: write the impl, write a
  one-line wrapper, register it in `PASSES`, add a corpus entry — no
  hidden ordering dependencies.
- **New `cypher(disable_optimizer=True, disabled_passes=[...])` kwargs.**
  Diagnostic / testing knob: skip every optimizer pass, or skip a
  specific subset by name. Validated against the registry — typos
  raise `ValueError`. Used by the new differential test harness and
  the bisection script.
- **Fixed `push_limit_into_match` multi-pattern row drop.** A query
  with a single MATCH containing multiple comma-separated patterns
  plus WHERE plus LIMIT (e.g. self-joins:
  `MATCH (p)-[:T]->(q), (p)-[:T]->(r) WHERE q <> r RETURN ... LIMIT 5`)
  silently dropped rows. The 0.8.27 fix narrowed the pushdown to
  single-MATCH but didn't check single-pattern; the pattern executor's
  `max_matches` hint applied per-pattern and the cartesian
  cross-product fell short of the requested LIMIT. The pass now also
  bails when the MATCH has more than one pattern.
- **Fixed `fuse_node_scan_top_k` empty-result on alias-sorted top-K.**
  Queries of the form `MATCH (p:T) RETURN <expr> AS h ORDER BY h LIMIT k`
  silently produced zero rows when the ORDER BY referenced a RETURN
  alias — the fused executor's sort-key evaluator only knows graph
  variables, not RETURN-alias bindings. The pass now bails when the
  sort expression references any RETURN alias, falling back to the
  materializing path which handles aliases correctly.
- **Fixed `desugar_multi_match_return_aggregate` over-grouping bug**
  surfaced by the new differential harness on first run.
  `MATCH (p:Person) MATCH (c:Company) RETURN p.city, count(c)` was
  producing 20 rows of `(city, n=5)` instead of 4 rows of
  `(city, n=25)`: the rewrite introduced a `WITH p, count(c)` (group
  by source variable) when the user's RETURN was grouping by
  `p.city`. The rewrite now generates `WITH p.city AS <internal>,
  count(c) AS n` so GROUP BY matches Cypher's standard semantic
  (the set of non-aggregate RETURN expressions).
- **New differential test harness `tests/test_cypher_differential.py`.**
  Every query in a curated corpus runs twice (optimized vs.
  optimizer-off) and asserts identical rows. Includes 9 mutation
  tests that compare the cypher result *and* the post-mutation graph
  state (node + edge counts) across the two modes. Surfaced 4
  divergences across two probing rounds, all now fixed and tracked as
  permanent regression tests:
    - `desugar_multi_match_return_aggregate` over-grouping
      (fixed, see above)
    - `push_limit_into_match` multi-pattern row drop (fixed)
    - `fuse_node_scan_top_k` empty-result on alias-sorted top-K (fixed)
    - `mark_fast_var_length_paths` per-target vs. per-path semantics
      (lone `KNOWN_DIVERGENT` entry; pending design call).
- **New `scripts/cypher_pass_bisect.py`.** Given a query that diverges,
  runs each pass disabled in isolation and reports which pass's
  absence resolves the divergence. Works against `.kgl` files or
  `tests/conftest.py` fixtures.
- **Debug-mode IR invariant checks** run after every pass in debug
  builds. Catches passes that produce empty MATCH patterns, empty
  RETURN/WITH item lists, or splice clauses after a terminal RETURN.
  Zero cost in release.

### code_tree

- **Manifest discovery: declared-package strategies and mixed-language
  safety net.** `read_pyproject` was previously the only Python source-root
  finder, and it only matched the `<name>/__init__.py` / `src/<name>/__init__.py`
  conventions. Several common configurations silently parsed only a slice
  of the repo:
  - **Workspace path collisions:** Cargo workspaces with two crates each
    containing `src/lib.rs` collapsed to a single `File` node because every
    parser stripped `rel_path` against its per-root walk directory. Paths
    are now project-root-relative (any two source roots that share a
    same-named file at matching depth survive as distinct nodes).
  - **Explicit `[tool.poetry].packages` declarations** with a custom `from`
    directory (e.g. `from = "lib"`) are now respected. Previously these
    only worked by accident when the project also lacked a conventionally-
    placed package — adding a stub `<name>/__init__.py` would silently
    suppress the lib/ packages.
  - **`[tool.setuptools].packages = [...]`** explicit lists, including
    dotted names resolved against `[tool.setuptools].package_dir`.
  - **`[tool.setuptools.packages.find].where = [...]`** is honoured —
    each `where` directory becomes a source root.
  - **`[tool.hatch.build.targets.wheel].packages = [...]`** is honoured.
  - **`[tool.poetry].name`** is now a valid name source. Pure-poetry
    pyprojects without a `[project]` table previously left `name` as the
    parent directory, breaking name-keyed package discovery.
  - **Mixed-language safety net:** when a pyproject finds a Python package
    next to first-level directories that contain code in a language NOT
    declared by the manifest (e.g. tooling pyproject + huge `src/*.c`),
    those directories are auto-supplemented as source roots labelled
    `auto:<dirname>`. The "undeclared language" gate keeps it surgical:
    sibling `.py` directories are not pulled in for pure-Python repos.

  The manifest module was also refactored into a list of small named
  strategy fns (one per declaration shape) instead of one 150-line
  function — adding a new build backend is now one fn, not a new branch.

## [0.8.35] — 2026-05-01

### Ingestion

- **`add_nodes(nullable_int_downcast=True)` recovers integer columns
  that pandas auto-promoted to float64.** Pandas turns nullable int
  columns into `float64` whenever nulls are present, which surfaces in
  queries as `"2.0"` instead of `2`. The new opt-in flag scans Float64
  columns post-ingestion: when every non-null value is integer-valued
  and within `i64` range, the column is downcast to `Int64`. Default
  `False` so existing callers see no change.

### code_tree

- **`code_tree.build(max_loc_per_file=N)` skips oversized files.**
  Files whose newline count exceeds `N` are recorded as `File`
  nodes with `skip_reason="too_large"` but never sent to the parser.
  Default `None` preserves existing behavior. Targets autogenerated
  multi-thousand-LOC files (dotnet/runtime's JIT regression tests at
  85k+ LOC each) that dominate parse time without contributing
  structural information. Also threaded through `repo_tree(...)`.
- **Module nodes with purely-numeric names are no longer synthesized.**
  Repos with numeric directory components (dotnet/runtime's
  `tests/JIT/Regression/Runtime_<bug-id>/...` test layout, in
  particular) used to produce thousands of `Module {title="125042"}`
  nodes when the parser fell back to file-path-derived module names.
  `build_modules` now skips path segments that are pure ASCII digits
  while keeping legitimate alphanumeric ancestors and descendants.

### Cypher engine

- **Map-typed list-comprehensions now access fields correctly.**
  `[x IN collect({h: a.title, k: km}) WHERE x.k = min_km | x.h]` used
  to silently drop every row because `Value` has no Map variant — the
  collected items round-tripped through a JSON-encoded `Value::String`,
  and `x.k` returned the entire string, never matching the aggregated
  `min_km`. Property access on a map-shaped projected string now parses
  the map on demand and returns the field. `parse_value_token` and
  `extract_map_field` were factored out of `parse_list_value`.
- **Spatial functions infer config from conventional property names.**
  `intersects()` / `contains()` / `centroid()` / `distance()` etc. now
  accept nodes that store WKT under `wkt_geometry` / `geometry` / `geom`
  / `wkt`, or lat/lon under `latitude`+`longitude` / `lat`+`lon`, even
  when no `SpatialConfig` was registered at ingestion. The fallback
  inference is per-query and never mutates `graph.spatial_configs`;
  explicit configs always win. Also: clearer error message when a
  spatial argument truly can't be resolved.
- **Cross-MATCH equality joins on non-id properties now run in O(N+M).**
  When a subsequent MATCH joins on a non-canonical property (e.g.
  `WITH a.key AS k MATCH (b) WHERE b.key = k`), the executor builds a
  query-local hash index over the target type once and probes it per
  outer row, instead of running the full pattern matcher per row. Only
  fires when the outer row count is at least 64 and no persistent
  property index already covers the type+property; below that, the
  existing path runs unchanged. Internal API: new
  `executor::transient_index::TransientEqIndex`. A 5,000×5,000 join
  that previously degraded to ~25M property reads now finishes in tens
  of milliseconds.

## [0.8.34] — 2026-05-01

### Code-tree graph quality

- **`File DEFINES Function` now includes methods.** The previous
  `if !is_method` filter dropped every C# method (the language has no
  top-level functions), so an entire codebase of methods looked
  edge-less when joined through File. Class HAS_METHOD edges still
  carry the logical hierarchy.
- **Kind-aware target resolution for IMPLEMENTS / EXTENDS.** When a
  bare base-type name matches multiple namespaces, `implements` now
  prefers Interface candidates and `extends` prefers class-like
  candidates. On dotnet/runtime the `Class -[IMPLEMENTS]-> Class`
  noise (mis-typed because of name collisions) dropped from 1,869 to
  90 rows, while the correct `Class -[IMPLEMENTS]-> Interface` rows
  rose from 447 to 7,696.
- **Auto-reroute `extends → implements` when the target is an
  Interface.** Fixes the C# parser's "first base is always extends"
  assumption for `class Foo : IDisposable` (no base class).
- **`using`-directive scope as a CALLS resolution tier.** Calls like
  `Assert.True` now pin to the `Assert` class actually imported by
  the caller's file. On dotnet/runtime, `Xunit.Assert.True` collapses
  from four collision-cloned entries (~11 k each, false equals) to a
  single 11 k entry; `IDisposable` implementer count goes from 0 to
  236, `IEnumerable<T>` from 0 to 305, `IEquatable` from 0 to 469.
- **C# `get_base_types` captures every secondary base.** The hardcoded
  list of accepted node kinds dropped any base type whose grammar
  kind wasn't on it (in practice every base after the first), so
  `class Foo : Bar, IDisposable` lost the IDisposable edge entirely.
- **C# generic args are stripped from base type names** so
  `IEnumerable<int>` resolves against the `IEnumerable` index entry.
- **`is_test` propagates from File to defined Functions.** Previously
  `meta_bool(f, "is_test")` returned `false` for every Function in
  every language except Rust `#[test]`, so the codebase-level test
  filter was unusable.
- **`build(save_to=...)` now persists the full property graph.** The
  build path skipped the `prepare_save` + `enable_columnar` steps
  `KnowledgeGraph.save()` does, so everything except `id`/`title`/
  `type` was stripped from the file. Round-tripped graphs now match
  in-memory ones.

### Performance

- `code_tree.build` is ~15% faster on polyglot codebases. Two changes:
  the orchestrator walks the source tree once and partitions files by
  language instead of re-walking per parser (was N+1 traversals — 8 walks
  of ~57k entries on dotnet/runtime); and a byte-level aho-corasick
  pre-check skips the full-AST comment walk for files that contain no
  TODO/FIXME/HACK/etc. keywords at all (the vast majority). On
  dotnet/runtime: 20.3 s → 17.4 s wall-clock, mostly from the C# parse
  phase (12.8 s → 11.2 s).
- New per-language and per-phase timings printed under `verbose=True`.

### Fixed

- `code_tree.build` no longer SIGBUSes on deeply-nested expressions in
  source files (e.g. dotnet/runtime's `JIT/Regression/JitBlue/GitHub_10215.cs`,
  a regression test that is literally a chain of thousands of `+`
  operators). Tree-sitter is recursive-descent and was overflowing the
  rayon worker thread stack (~2 MB on macOS); parsers now share a
  dedicated rayon pool with a 16 MB stack, so pathological-but-valid
  inputs parse cleanly across all languages.

## [0.8.33] — 2026-04-30

### Tooling

- New `bench/bench_cohort_cold.py` — spawns a fresh Python subprocess
  per (query, iteration) and runs `sudo purge` between each so the
  OS page cache is dropped before the kglite mmap is re-faulted.
  Settles "is plan X actually faster than plan Y?" questions that
  warm-cache timings can't answer.
- Planner regression test pinning that the user's exact cohort top-K
  shape (with explicit `WITH p` between MATCHes) absorbs `top_k`
  after the fold + desugar + fuse pipeline.

## [0.8.32] — 2026-04-30

### Performance — cohort top-K with PropertyAccess RETURN now absorbs LIMIT

`fuse_match_with_aggregate_top_k` previously required every RETURN
item to be a plain alias of a WITH-projected column. Cohort queries
of the form

```cypher
MATCH (p)-[:P27]->({id: 20})
WITH p
MATCH (p)-[r]-(other)
WHERE NOT (type(r) = 'P50' AND startNode(r) = other)
RETURN p.title, p.description, count(r) AS d
ORDER BY d DESC LIMIT 10
```

include `p.title` and `p.description` (PropertyAccess) in the RETURN,
so the absorber bailed and the fused operator emitted *every* cohort
row before LIMIT — paying property-column I/O for ~73K Norwegians on
Wikidata even though only 10 survived. The relaxed gate now accepts
PropertyAccess on the WITH's group variable (the executor already
preserves `node_bindings[group_var]` for K-winner rows). Warm-cache
runtime on the Norwegians cohort fell from ≈0.9s to ≈0.4s; cold runs
that previously exceeded the MCP 20s ceiling now stay under it.

## [0.8.31] — 2026-04-30

### Performance — fused OPTIONAL MATCH widens to derived aggregates and edge-var counts

Two follow-on fixes after the 0.8.30 release. Both close gaps in
`fuse_optional_match_aggregate` that kept the cohort-style query in
the issue from picking up its existing fast path:

- **Edge-variable counts now fuse.** The fusion gate's local-binding
  set was built from `collect_pattern_variables`, which only returns
  *node* variables. `count(r)` over an OPTIONAL MATCH edge variable
  failed the local-to-OPT check and fell back to the materialized
  per-row expansion. Replaced with a local walk that includes edge
  variables in the OPTIONAL pattern's binding set, minus any names
  already bound by prior MATCH/WITH/UNWIND.

- **Derived `total - count(rp)` expressions now fuse.** Previously
  the gate accepted only pure `count(...)` aggregates; arithmetic
  involving them blocked fusion entirely. The gate now recognizes
  any expression whose only aggregate is `count(...)` and the
  executor substitutes the per-row count into each `count(...)`
  sub-tree before evaluating the surrounding arithmetic. Same row
  cost as the pure-count path.

- **Output columns now reflect the fused operator's own RETURN/WITH
  items.** Previously the fused operator silently inherited the
  upstream's column names, so a downstream consumer reading
  `row["p50_in"]` got a `KeyError`. Visible only when the OPTIONAL
  MATCH adds new RETURN columns the upstream WITH didn't have.

`startNode(r)` / `endNode(r)` returning the matcher's anchor side
instead of the actual graph endpoints (caught while testing) now
look up the edge endpoints via `edge_index`. This was a pre-existing
bug, exposed by Phase 3 actually exercising the predicate.

**Wikidata cohort impact** (warm cache, 124M-node graph; queries
from the issue's "narrow then enrich" report):
- fm1 (`WHERE NOT (type(r) = 'P50' AND startNode(r) = other)`):
  656ms → 624ms (~unchanged; the win was already in 0.8.30)
- fm2 (post-aggregate OPTIONAL MATCH + `total - count(rp)`):
  1290ms → **403ms** (~3.2× faster, fully fused)

## [0.8.30] — 2026-04-30

### Performance — relationship-predicate pushdown (Phase 3)

Pushes `WHERE` sub-predicates that reference only the edge variable
(and the structural peer endpoint of that edge in the pattern) into
the matcher's expansion loop, *before* per-edge bindings allocate.
Selective edge filters that previously forced the materialized 100M+
row path now run during expansion via the disk CSR sorted-edge-type
binary search.

- **`extract_pushable_rel_predicates`** in
  `src/graph/languages/cypher/planner/rel_predicate_pushdown.rs`
  recognizes `type(r) = 'X'` / `type(r) IN […]`, `r.<prop> OP <lit>`
  for `=`/`<>`/`<`/`<=`/`>`/`>=`, and `startNode(r) = peer` /
  `endNode(r) = peer` against the structural peer in the same
  pattern. AND/OR/NOT compositions of those leaves push as a unit
  when the entire subtree is pushable; partial pushdowns over OR /
  NOT correctly leave the predicate alone.
- **`EdgePattern::edge_filter`** carries the compiled
  `RelEdgePredicate` to the matcher. The hot loop in
  `pattern_matching/matcher.rs` evaluates it after the existing
  connection-type check and before the property check — single
  branch-predicted `if let Some` for the no-filter path.
- **Fused count phase honors the filter.** The fused
  `FusedMatch{Return,With}Aggregate` operators run their own count
  loop via `try_count_simple_pattern` /
  `try_count_distinct_peers`; both now apply the inline filter
  during edge iteration so fused queries with a pushed predicate
  produce correct results without falling back to the materialized
  path. `try_fast_with_aggregate_via_histogram` bails when a filter
  is set (the histogram counts every edge of a type by definition).

`startNode(r)` / `endNode(r)` previously returned the
`EdgeBinding`'s pattern-anchor side instead of the actual graph
source/target — silently wrong when the planner anchored on the
right-hand pattern endpoint and walked incoming edges. Fixed by
looking up the edge endpoints via `edge_index`.

**Wikidata cohort impact** (warm cache, 124M-node graph):
- baseline `MATCH (p:Q20-citizen)-[r]-()` aggregate: 224ms
- `WHERE type(r) = 'P19'` (selective): 47s → **667ms** (~70× faster)
- `WHERE type(r) IN ['P19', 'P569', 'P570']`: 47s → **670ms**
- `WHERE NOT (type(r) = 'P50' AND startNode(r) = other)`: 60s → **656ms** (~90× faster, correct result)

### Performance — streaming aggregate + heap top-K (Phase 1)

First slice of a multi-phase rework that lifts streaming primitives
into the Cypher generic execution path. Falling out of the fused fast
path used to cost ~1000× on cohort-scale queries; this slice closes
the gap on shapes that combine `WITH(group, agg)` with `ORDER BY …
LIMIT k` decorations the existing fused operators don't cover.

- **`RowStream` operator pipeline** in
  `src/graph/languages/cypher/executor/stream/`. The driver in
  `executor::execute` tries to absorb a contiguous clause run
  (`WITH/RETURN(group, agg) [→ ORDER BY → LIMIT]`) into a single
  streaming pipeline before the materialized executor sees the
  clauses. On no match the driver falls through with the input
  `ResultSet` unchanged.

- **`StreamingAggregate`**: hash aggregate that builds per-group state
  inline as upstream rows arrive — same I/O profile as the
  materialized path (NodeIndex surrogate keys, deferred property
  reads, re-bucket-by-resolved-value at finalization). Supports
  `count(*)`, `count[(DISTINCT) expr]`,
  `sum/avg/min/max[(DISTINCT) expr]`. Other aggregates (`collect`,
  `std`, percentiles, arithmetic on aggregates) bail to the
  materialized executor unchanged.

- **`HeapTopK`**: `BinaryHeap` of capacity K replaces the full sort +
  truncate path for streaming pipelines that end in `ORDER BY <expr>
  [ASC|DESC] LIMIT k`. O(n log k) instead of O(n log n).

- **`streaming` kwarg on `kg.cypher`** (default `True`). Pass
  `streaming=False` to force the materialized executor — useful for
  parity debugging.

Phases 2-4 will widen recognized shapes (multi-MATCH, OPTIONAL MATCH
streaming, post-aggregate `WHERE`), inline relationship predicates
into pattern expansion, and retire the shape-matched fused operators
once benchmarks confirm streaming parity.

## [0.8.29] — 2026-04-30

### Performance — cohort + multi-MATCH planner improvements

Three planner passes turn cohort top-K and multi-MATCH joins on
Wikidata-scale graphs from "borderline timing out" into "sub-second
warm." Validated end-to-end on the 124M-node / 861M-edge Wikidata
graph; no regressions on point lookups, 1-hop / 2-hop, aggregates, or
load.

- **`reorder_match_clauses`** — orders consecutive id-anchored MATCH
  clauses by edge-type total-count cost. Drives from the rarer side
  first.
  ```cypher
  MATCH (p)-[:P31]->({id:5}) MATCH (p)-[:P27]->({id:183}) RETURN p.title LIMIT 20
  ```
  458s cold / 497s warm → 49s cold / **0.5s warm**.

- **`fold_pass_through_with`** — strips a `WITH x [, y, ...]` clause
  that's a pure projection (no DISTINCT/aggregate/WHERE/ORDER BY) when
  every variable referenced downstream is in the projection list. Lets
  later fusion passes see a contiguous Match-Match span when the user
  wrote `Match WITH p Match …`.

- **`desugar_multi_match_return_aggregate`** — rewrites
  `Match-Match-Return(group, aggregate)` into
  `Match-Match-With(group_var, aggregate)-Return(project)`. Lets the
  existing aggregate fusion fire on the natural "RETURN with
  aggregate" form.

Together, the two simplification passes turn cohort top-K queries
from per-row materialization into the streaming aggregate path:
- Norwegians outgoing-degree top 10: 34s → **0.07s** (490×)
- Norwegians total-degree top 10: 38s → **1.35s** (28×)

The reorder pass is gated to avoid in-memory regressions
(edge-type-counts cache must be populated, id-anchors required,
shared variable required). The simplifications are pure AST rewrites —
no executor changes, O(1) planner overhead.

## [0.8.28] — 2026-04-30

### Performance — slice-built graph load (round 2)

Continuation of the disk-graph load optimisation that landed earlier in
this version. Three further changes drop slice-built Wikidata graph
loads from seconds to ~100 ms:

- **`metadata.json` heavy fields → binary sidecars.** The two
  HashMap-of-HashMap fields (`node_type_metadata`,
  `connection_type_metadata`) move into dedicated
  `node_type_metadata.bin.zst` and `connection_type_metadata.bin.zst`
  files with a hand-rolled length-prefixed format. On the 1B-triple
  Wikidata slice, `metadata.json` shrinks from 5.0 MB to 23 KB, and
  the parse drops from ~1 s to ~10 ms. (The custom `Serialize`/
  `Deserialize` impls on `ConnectionTypeInfo` make bincode
  round-tripping unsafe — hence the hand-rolled binary.)

- **`type_connectivity_cache` is lazy.** Skipped both the
  cartesian-product derive in `apply_to` (clones tens of millions of
  String triples on slice-built graphs) and the eager
  `type_connectivity.bin.zst` read at load. Existing read sites in
  `introspection/describe.rs` already fall through to a bounded edge
  scan when the cache is missing; first `describe()` triggers a
  `compute_type_connectivity` populate. Set
  `KGLITE_EAGER_TYPE_CONNECTIVITY=1` to opt back into eager loading
  for workloads that immediately call `describe()`.

- **`apply_to_with(graph, derive_type_connectivity: bool)`.** Splits
  the implicit derive out of the metadata-application path so the
  caller can opt out. The original `apply_to` is now a thin wrapper
  that defaults to `true` for the in-memory `.kgl` load path (which
  has no separate sidecar).

**Round-2 measurements** (warm load, M2 macOS, external SSD):

| graph        | nodes | round-1 | round-2 | total speedup vs original |
|--------------|-------|---------|---------|---------------------------|
| graph_500.0  | 6 M   | 770 ms  | **62 ms**  | (built fresh) |
| graph_1000.0 | 16 M  | 4.89 s  | **113 ms** | 43× |
| main wikidata| 124 M | 178 ms  | **179 ms** | 51× vs 9.08 s baseline |

The remaining 100-180 ms is dominated by `column_stores_load`
(72-124 ms) — a per-type ColumnStore wrapper construction that's
already mmap-backed.

### Fixed — seg_000 corruption on save of legacy disk graphs

Pre-existing bug exposed by the migration round-trip benchmark: re-saving
a disk graph whose `disk_graph_meta.json` carries `sealed_nodes_bound: 0`
(the serde default for pre-phase-8 graphs) AND has a non-empty
`seg_manifest.json` would call `seal_to_new_segment` with `tail_lo=0`,
`tail_hi=node_count`. That writes a fresh empty `seg_001` AND truncates
`seg_000/out_offsets.bin` and `seg_000/in_offsets.bin` to one entry via
`reconcile_seg0_csr` — the on-disk CSR loses every edge offset, and the
graph reloads with zero traversable edges (the edge data files survive,
but without offsets nothing is reachable).

`DiskGraph::load_from_dir` now bumps `sealed_nodes_bound` to `node_count`
when it detects the legacy zero with a populated segment manifest. Fresh
phase-8+ graphs persist the correct watermark, so the bump is a no-op
for them.

### Performance — disk-graph load

`kglite.load(path)` on the 124M-node Wikidata graph drops from ~9.0s to
~5.5s warm-cache today, and to ~1s once the graph is re-saved with this
version. Three changes:

- **`prefetch_hot_regions` no longer runs by default.** It called
  `madvise(MADV_WILLNEED)` on the 1.9 GB out_offsets + in_offsets arrays.
  On macOS that syscall synchronously schedules readahead and blocks even
  on warm pages — costing ~2.7s of every load on Wikidata-scale graphs.
  The kernel pages in offsets on first query anyway, so the upfront cost
  was a tax with no payoff for typical anchored-MATCH workloads. Set
  `KGLITE_PREFETCH=1` to opt in for first-query latency-sensitive use
  cases.

- **`id_indices` is now mmap-resident.** New raw `id_indices.bin`
  layout: header + sorted-by-type-key directory + per-type sorted u32
  keys / u32 NodeIndex arrays. Lookups are O(log N) binary search on a
  cache-friendly contiguous slice instead of O(1) HashMap probe with a
  cache miss; cost in practice is parity (~50-100 ns either way for
  13M-entry types). Eliminates the 124M `HashMap::insert` rebuild that
  cost ~5.3s on Wikidata. New struct `IdIndexStore` in
  `storage/disk/id_index.rs` with overlay for post-load mutations.

- **`type_indices` is now mmap-resident.** New raw `type_indices.bin`
  layout: header + directory + contiguous `[u32]` slices per type. Reads
  return a `TypeNodesRef` view that yields `NodeIndex` either directly
  from the overlay `Vec` or by reinterpreting the mmap'd `u32` slice.
  Eliminates the 124M `Vec::push` rebuild that cost ~890ms on Wikidata.
  New struct `TypeIndexStore` in `storage/disk/type_index.rs` with
  overlay for post-load mutations (delete paths promote to overlay on
  first mutation).

- **Sub-stage instrumentation** for `DiskGraph::load_from_dir`. Set
  `KGLITE_LOAD_TIMING=1` and stages emit `[TIMING] dg.<name> dur_ms=N`
  lines to stderr (segment_csr, edge_properties, overflow_edges,
  segment_manifest). Useful for measuring the next round of load
  optimizations.

**Backward compatibility.** Loaders try the new `id_indices.bin` /
`type_indices.bin` first, fall back to the old `.bin.zst` legacy
formats, then to a node_slots scan. Existing graphs continue to load —
slower until re-saved. Re-save once with `g.save(path)` to migrate.

**RSS reduction.** Wikidata peak RSS post-load drops from 3.6 GB to
~500 MB (the Integer-variant id_indices + type_indices were the bulk
of heap). General-variant id_indices entries still materialize on
first access.

## [0.8.27] — 2026-04-29

### Changed

- **Default Cypher query timeout is now 180_000 ms (3 min) for every storage
  mode** (memory, mapped, disk). Previously memory had no default deadline,
  mapped was 60s and disk was 10s. The old per-mode defaults left memory
  queries unbounded and disk's 10s ceiling tripped legitimate cold queries
  on large graphs. Override per-call with `timeout_ms=N` (or `0` to disable),
  or globally via `set_default_timeout(ms)`.

### Fixed

- **Cypher property-anchored single-MATCH fusion returned empty results.**
  `MATCH (m {id: X})<-[:R]-(p) RETURN m.title, count(p)` (and the directed
  / DISTINCT / undirected variants) returned zero rows on any graph with
  matching data. `try_count_simple_pattern` bailed with `Ok(None)` when
  the bound node carried property filters; the fused executor's
  `count_for_node` closures `.unwrap_or(0)` that None into a zero count,
  which the row-skip guard then dropped. The bail-out has been removed —
  the bound `NodeIndex` already satisfies its property filter by virtue of
  being selected upstream, so re-checking is unnecessary. Also fixed an
  in-memory-backend miss in `try_count_distinct_peers` (the new helper
  added in the count(DISTINCT) work this release): `edges_directed_filtered`
  is a hint on the in-memory backend and returns every edge regardless of
  connection type, so the function now post-filters by `connection_type`.
- **Cypher fusion machinery now accepts `count(DISTINCT v)` on a node
  variable.** Previously rejected at fusion time, forcing all distinct-count
  queries — the canonical "top-N by relationship count" Cypher shape —
  through the materializing executor (full intermediate cross-product, then
  group-by, then sort). The planner now propagates a `distinct_count` flag
  into `FusedMatchReturnAggregate` and `FusedMatchWithAggregate`; the
  executor uses a per-group `HashSet<NodeIndex>` of peers instead of an
  edge counter, naturally collapsing multi-edges. Edge-centric fast paths
  (which count edges, not distinct peers) are bypassed in distinct mode.
  Fusion is gated on the group node being type- or property-constrained,
  since the fused per-node enumeration only beats the materializing path
  when the group set is small (the materializing path's single sequential
  edge scan wins on unconstrained 124M-node groups). Documented limitation:
  3+ MATCH queries still don't fuse and continue to use the materializing
  path.
- **Cypher planner's selectivity estimator now considers variables bound by
  earlier clauses.** Previously `MATCH (p:Person) MATCH (p)-[:KNOWS]->(c:Type)`
  treated `(p)` as statically unconstrained — worst-possible selectivity — and
  reversed the second pattern to start scanning all `:Type` nodes (millions on
  large graphs) instead of expanding from the pre-bound `p`. The estimator
  now walks clauses in order, accumulates node variables introduced by each
  MATCH/OPTIONAL MATCH, and treats already-bound variables as selectivity 1
  (effectively-anchored). On a 124M-node Wikidata graph, a 3-MATCH query
  with two `{id: …}` anchors now completes in ~1s (was 20s+ timeout).
- **Cypher LIMIT pushdown was unsafe for multi-MATCH queries with WHERE on a
  late-bound variable.** `MATCH a MATCH b MATCH c WHERE c.id = X RETURN ... LIMIT N`
  was rewritten to push `limit_hint = N` into the last MATCH clause. The
  per-row pattern executor's `max_matches = remaining` then interacted
  incorrectly with the outer row loop, causing fewer matching rows than
  expected to surface (e.g. `LIMIT 10` returning 8, `LIMIT 5` returning 3 — or
  zero rows on Wikidata-scale graphs where the WHERE bucket happened to be at
  the tail of the row stream). The planner now only pushes LIMIT into MATCH
  for queries with a single MATCH/OPTIONAL MATCH clause; multi-MATCH queries
  retain LIMIT as a separate clause and apply it after the full pattern
  matching completes. Reproduced and confirmed against the user's
  124M-node Wikidata disk graph (Issue 1 of the 2026-04-29 bug report).
- **Cypher planner now reverses undirected and variable-length patterns by
  selectivity.** `MATCH (other)-[r]-(p {title: 'X'})` previously left
  `other` (no constraints) as the start node, causing a full-graph scan
  with edge expansion from every node. The over-conservative bail-out on
  `EdgeDirection::Both` and `var_length` has been removed — `Both` reverses
  to `Both` (no semantic change) and var-length reversal is symmetric for
  patterns without a path assignment (path-bound patterns are already
  protected by a separate guard). On a 124M-node Wikidata graph this turns
  a multi-minute scan into a sub-second anchored lookup.

## [0.8.26] — 2026-04-28

### Fixed

- **`code_tree` Rust parser missed calls inside macro invocations.** Calls
  inside `format!`, `vec!`, `json!`, `Err(format!(…))`, custom derive
  macros, etc. were silently dropped because tree-sitter-rust represents
  them as `identifier` + `token_tree` siblings rather than `call_expression`
  nodes. The walker now dives into `macro_invocation` token-trees and
  reconstructs synthetic call sites. Resolves the dominant source of
  false-positive orphan-function reports on Rust codebases.
- **`code_tree` Rust parser dropped turbofish call expressions.** Calls of
  the form `path::with::<T>(...)` are wrapped in a `generic_function`
  AST node that the previous match arm didn't handle, so e.g.
  `reconcile_seg0_csr::<DiskNodeSlot>(arg)` produced no CALLS edge. The
  type arguments are now stripped and the inner identifier/scoped path is
  recursively recorded.
- **`code_tree` Rust parser misresolved `Self::method(...)` calls.** The
  `"Self"` segment was emitted as an explicit receiver hint, which matched
  no function's owner type and broke disambiguation for non-unique method
  names. The parser now strips the `Self::` prefix so the resolver's
  implicit caller-owner hint kicks in, yielding the same behaviour as
  bare `self.method()` calls inside an impl block.
- **`code_tree` IMPLEMENTS edge schema excluded `Enum -> Trait`.** A Rust
  enum implementing a trait (e.g. `impl Clone for GraphBackend`) yielded
  no `IMPLEMENTS` edge because the IMPLEMENTS routing only mapped Class /
  Struct sources. `Enum` is now a recognised source label.

### Removed

- **macOS x86_64 wheels.** `x86_64-apple-darwin` is no longer built or
  published to PyPI. Apple Silicon (`aarch64-apple-darwin`) remains. Intel
  Mac users on existing installs are unaffected; new installs will need to
  build from source.

## [0.8.25] — 2026-04-27

### Fixed

- **`code_tree.build` silently parsed only `tests/` for repos with a tooling-only
  `pyproject.toml`.** Manifests that declared no primary source roots (e.g.
  llama.cpp's `pyproject.toml` for poetry-managed scripts, with no
  `<name>/__init__.py` package and no maturin) yielded `source_roots = []`
  and `test_roots = ["tests"]`. The builder then parsed only `tests/`, set
  `parsed_any = true`, and skipped the whole-repo fallback — silently
  producing an undersized graph (e.g. 58 files instead of thousands).
  When a manifest declares zero source roots, the builder now logs the
  situation in verbose mode and falls through to the whole-repo scan
  instead of trusting the test-only roots.

## [0.8.24] — 2026-04-27

C++ parser robustness round driven by analysing nlohmann/json (24% → 12%
signature fallback) and llama.cpp's `src/` (32% → 5.9% fallback, function
count 742 → 1,263 because previously-`unknown`-named functions can now be
called targets, CALLS edges 352 → 2,971).

### Fixed

- **C-style `struct T *` / `enum T` / `union T` parameter types** —
  `struct_specifier`, `enum_specifier`, `union_specifier`, `class_specifier`,
  `sized_type_specifier` were missing from C++ `extract_parameters`'s
  type-recognition list. C-heavy headers like llama.cpp's
  `void llama_grammar_free(struct llama_grammar * grammar)` were losing the
  parameter type entirely (`type_annotation=None`).

- **Out-of-class C++ method names** — `bool Foo::bar() const` produces a
  `qualified_identifier` (`Foo::bar`) child in tree-sitter-cpp, which the
  existing `get_name` walk missed. Now drills into `qualified_identifier`
  and returns the trailing segment (`bar`).

- **Reference-return functions** — `T & foo()` wraps the real
  `function_declarator` in a `reference_declarator`. `parse_function` now
  unwraps it just like it already did for `pointer_declarator`. Functions
  returning references no longer show as `name="unknown"`.

- **C++ template-typed methods, destructors, qualifier-stripped return types,
  and macro-decorated constructors.** Four targeted parser fixes that drop
  nlohmann/json's signature-fallback rate from 24% → 12% and capture 39
  previously-missed template methods.
  - `template_type` and `qualified_identifier` now in `TYPE_NODES` so generic
    return types like `iteration_proxy<int>` and `std::vector<T>` are captured.
  - `type_qualifier`, `storage_class_specifier`, `virtual_specifier` skipped
    in `get_return_type` — fixes `constexpr int foo()` returning "constexpr"
    instead of "int".
  - `destructor_name` recognized in `get_name` — `~Widget` now returns
    `"~Widget"` instead of `"unknown"`.
  - In-class `field_declaration` items containing `function_declarator` are
    routed to `parse_function` (template-typed methods were being treated as
    fields). New `find_buried_function_declarator` walker unwraps
    `parenthesized_declarator` wrappers that tree-sitter-cpp emits around
    macro-decorated constructors (e.g. `JSON_HEDLEY_NON_NULL(3) Foo(int x)`)
    so the real `function_declarator` and its parameters are captured.

## [0.8.23] — 2026-04-27

C++ macro-aware parsing, Go method return-type fix, and receiver attribution
across Go and Rust. Cross-library validation: testify signature-fallback
49% → 0%, KGLite self-graph USES_TYPE edges 2078 → 3197 (+54%), spdlog
macro names eliminated from top return-type list.

### Added

- **`ParameterKind::Receiver`** — Go method receivers (`(c *Call)`) and Rust
  `&self`/`&mut self`/`self` are now captured as structured parameters with
  `kind: "receiver"`, distinct from positional/variadic/kw_variadic. Excluded
  from `param_count` (receivers aren't user-supplied arguments). Cypher
  consumers can filter via `parameters` JSON column.

- **`USES_TYPE` `position="receiver"`** — receivers contribute USES_TYPE edges
  with their own position label. A method `(c *Call) Once() *Call` (receiver +
  return) collapses to `position="both"` consistent with existing aggregation.
  On testify, this drops `position="signature"` fallback from 49% → 0% and
  raises total USES_TYPE edges 251 → 365 (+45%). On KGLite self-graph,
  USES_TYPE edges go 2078 → 3197 (+54%) — Rust `&self` methods now surface
  their owner type as a receiver USES_TYPE edge.

### Fixed

- **C++ parser ignores macro decorators** (`SPDLOG_INLINE`, `FMT_API`,
  `FMT_BEGIN_NAMESPACE`, `Q_INVOKABLE`, etc.). Without this, tree-sitter-cpp
  parses `SPDLOG_INLINE void foo()` so that `SPDLOG_INLINE` looks like a
  type and `foo` becomes the return type — producing `name="unknown"` and
  `return_type="SPDLOG_INLINE"`. Heuristic in `parsers/shared.rs::looks_like_macro_decorator`
  matches all-caps identifiers (length ≥ 2, optional underscores/digits) and
  is applied in `cpp.rs::get_return_type`, `get_name`, and parameter
  extraction. `get_return_type` also recovers from tree-sitter `ERROR` wrappers
  around primitive type keywords (`void`, `int`, `bool`, etc.) — common when
  a macro decorator has confused the parser. On spdlog, macro names are
  eliminated from the top return-type frequency list.

## [0.8.22] — 2026-04-27

A code-graph quality round driven by analysing KGLite's own self-graph and
closing every concrete gap that surfaced. Five new node/edge primitives,
seven new properties on existing nodes, and one small dead-code cleanup.

### Added

- **`BINDS` edges — Python wrapper to Rust pymethod.** Closes the cross-language
  gap where `kglite.KnowledgeGraph.add_nodes` (the Python class method) and
  `crate::graph::pyapi::*::KnowledgeGraph::add_nodes` (the Rust `#[pymethods]`
  impl) lived as disconnected `Function` nodes. The resolver indexes Rust
  functions with `is_pymethod = true` by `(parent_struct_short_name,
  method_name)` and emits `Function -[BINDS]-> Function` for each Python
  method that finds a unique match. Cypher: `MATCH (py)-[:BINDS]->(rs)
  -[:CALLS*]->(impl)` traces a request from the Python entry point to deep
  Rust impl. On the KGLite codebase: ~184 BINDS edges; closes the false-positive
  dead-code finding for `load_ntriples` and other pyapi-exposed functions.

- **Promoted metadata flags as typed Function/Class properties.** Eight booleans
  (`is_pymethod`, `is_pymodule`, `is_ffi`, `is_static`, `is_abstract`,
  `is_property`, `is_classmethod`) plus the `ffi_kind` string are now
  Function-node columns; `is_pyclass` is a Class/Struct column. Replaces
  `f.metadata.get("is_pymethod") == true` JSON-parsing gymnastics with
  `MATCH (f:Function {is_pymethod: true})` direct filters.

- **`USES_TYPE` edges carry a `position` property** (`parameter` | `return` |
  `both` | `signature`). Distinguishes consumers from producers — a function
  that takes `Widget` as a parameter and a function that returns `Widget` no
  longer collapse to the same edge shape. Aggregated per `(function, type)`
  so a single transformation `fn f(w: Widget) -> Widget` emits one edge with
  `position: "both"`. Cypher: `WHERE r.position IN ['parameter','both']` to
  find consumers; `IN ['return','both']` for producers.

- **`Module HAS_FILE File` edges** — closes the natural top-down walk from
  Module → File → Function. Was string-prefix gymnastics on `qualified_name`;
  now `MATCH (m:Module)-[:HAS_SUBMODULE*0..]->(:Module)-[:HAS_FILE]->(f:File)
  -[:DEFINES]->(fn:Function)` returns "what's in this module" in one query.
  Edge name avoids `CONTAINS` (a reserved Cypher keyword for substring matching).

- **`Procedure` nodes — annotation-driven, language-agnostic.** Functions
  whose docstring/leading comment contains `@procedure: NAME` (or
  `@cypher_procedure: NAME`) at the start of a line synthesize a `Procedure`
  node with an `IMPLEMENTED_BY` edge to the function. A single function can
  carry multiple annotations to register under aliases (e.g. both
  `betweenness` and `betweenness_centrality` dispatching to the same impl).
  Generic mechanism for surfacing project-specific registries (Cypher CALL
  procedures, RPC method catalogs, command-bus dispatchers) as first-class
  graph entities. Anchored to line start so prose mentions in docs/tests
  don't false-positive.

- **Annotated all 22 KGLite Cypher CALL procedures** in
  `src/graph/algorithms/graph_algorithms.rs`,
  `src/graph/languages/cypher/executor/rule_procedures.rs`, and
  `executor/call_clause.rs::execute_call_cluster`. Activates the `Procedure`
  node mechanism on the KGLite self-graph: `MATCH (p:Procedure {name: 'pagerank'})
  -[:IMPLEMENTED_BY]->(f:Function) RETURN f.qualified_name` resolves a Cypher
  procedure name to its Rust impl in one query. 27 Procedure nodes (including
  aliases) → 22 implementing functions.

- **Function complexity counters** — `branch_count`, `param_count`,
  `max_nesting`, `is_recursive` now populate on every `Function` node
  produced by `kglite.code_tree.build(...)`. Computed from the
  tree-sitter AST in the same walk that gathers `CALLS` edges, so
  there's no extra parse pass. Per-language branch tables in
  `parsers/shared.rs` cover `if`/`for`/`while`/`case`/`catch`/ternary
  and short-circuit `&&`/`||` forms (cyclomatic-style). Enables direct
  Cypher queries for high-complexity hotspots:

  ```cypher
  MATCH (f:Function)
  WHERE f.branch_count > 30 AND NOT EXISTS { ()-[:CALLS]->(f) }
  RETURN f.qualified_name, f.branch_count, f.max_nesting
  ORDER BY f.branch_count DESC
  ```

- **Generated and minified file skipping during ingestion.** The
  builder now content-sniffs each source file's first 2 KiB before
  dispatching to the per-language parser. Files matching codegen
  markers (`auto-generated`, `DO NOT EDIT`, `code generated by`,
  `<auto-generated>`, `@generated`) are skipped, as are minified
  bundles (one extreme line, or average line width above 500 chars
  across the first 50 lines). Skipped files emit only a `File` node
  with `skip_reason: "generated"` or `skip_reason: "minified"` — no
  Function/Class/Constant nodes — so phantom CALLS edges from
  protobuf stubs and webpack bundles no longer pollute the graph.

  ```cypher
  MATCH (f:File) WHERE f.skip_reason IS NOT NULL
  RETURN f.path, f.skip_reason
  ```

- **Structured `parameters` on `Function` nodes** — JSON-serialised
  list of `{name, type_annotation, default, kind}` per declared
  parameter, with `kind ∈ {positional, variadic, kw_variadic}`.
  Implicit receivers (`self`/`cls`/`&self`/`&mut self`) are excluded.
  Promoting parameters out of the signature string also extends
  `USES_TYPE` resolution: parameter type annotations are now scanned
  alongside the signature and return type, so a function that takes a
  `Widget` argument but doesn't return one now emits the expected
  `Function -[USES_TYPE]-> Widget` edge.

## [0.8.21] — 2026-04-27

Code-analysis tooling round. Closes seven issues filed against KGLite's
own MCP server after a self-analysis session surfaced them — all
visible to users running `kglite.code_tree.build(...)` or
`g.cypher("CALL ...")` against a Rust codebase.

### Added

- **`REFERENCES_FN` edge type — Function → Function** for bare or
  scoped identifiers passed as arguments to higher-order calls
  (`iter.and_then(some_fn)`, `Option::map(my_helper)`). Distinct from
  `CALLS` because the referenced function isn't necessarily invoked
  at the reference site. Dead-code analysis can union the two:

  ```cypher
  MATCH (f:Function)
  WHERE NOT EXISTS { ()-[:CALLS]->(f) }
    AND NOT EXISTS { ()-[:REFERENCES_FN]->(f) }
  RETURN f.qualified_name
  ```

- **`REFERENCES` edge type — Function → Constant** for bare or scoped
  identifiers in function bodies that resolve to a known constant.
  The Rust parser uses `SCREAMING_SNAKE_CASE` as the parse-time
  filter, so local variables don't pollute the edge set. Enables
  detecting unreferenced constants directly from the graph rather
  than via ripgrep.

- **`orphan_node` accepts `link_type` and `direction` parameters.**
  The default behaviour (zero edges in any direction) is unchanged;
  the new params let queries express "no inbound matching edge of a
  specific connection type" — the natural shape for "functions never
  called", "files never imported", etc.:

  ```cypher
  CALL orphan_node({type: 'Function', link_type: 'CALLS', direction: 'in'})
  YIELD node RETURN node.qualified_name
  ```

- **`EXISTS { MATCH ... MATCH ... [WHERE ...] }` multi-clause
  subqueries.** The bare-pattern form already worked; the full
  subquery form with multiple `MATCH` clauses (sharing variables)
  and a `WHERE` predicate evaluated against the merged bindings now
  parses and executes. Multi-hop existence checks no longer have to
  be rewritten as `MATCH ... WITH collect(...) AS xs ... AND NOT y IN xs`.

- **`Project.crate_type` column** captured from `[lib] crate-type` in
  `Cargo.toml`. Lets downstream queries distinguish a regular `lib`
  crate (where `pub fn` is a real export) from a `cdylib` PyO3 crate
  (where only `#[pyfunction]` / `#[pymethods]` matter).

- **`Function.is_test` column** surfaced as a queryable property on
  Function nodes (previously only stored in metadata).

### Fixed

- **`CALLS` edges now include calls inside closure bodies.**
  `closure_expression` was on the parser's NESTED_SCOPES skip-list,
  so `.map(|x| foo(x))` / `.and_then(|x| bar(x))` produced zero
  CALLS edges to the inner function. Closures are expressions in
  Rust, not items — calls inside them belong to the enclosing
  function semantically.

- **`self.method()` receiver-type disambiguation.** When the same
  method name exists on multiple structs, a bare `self.method()`
  call inside a method of `Foo` now narrows to `Foo::method` ahead
  of `Bar::method`, even when both are candidates and live in
  different files. Uses the caller's owner short name as an implicit
  receiver hint when no explicit one is present.

- **`is_test` propagates into inline `#[cfg(test)] mod tests` blocks.**
  Previously only `#[test]` / `#[bench]` annotated functions were
  flagged; helpers inside the test mod weren't, inflating every
  dead-code query against a Rust codebase. Files literally named
  `tests.rs` are also flagged at the file level.

- **`CALL` rule procedures list accepted parameters in error
  messages.** Missing-required-parameter errors now show the full
  schema (required + optional names), so first-time use of a
  procedure doesn't cost three error rounds before guessing the
  parameter name.

### Removed

- **`ARCHITECTURE.md`.** A refactor-time artifact from the 0.8.0
  storage refactor; framing was past-tense and the parity test that
  validated its file-path references was removed alongside it. The
  three other parity gates (god-file cap, unsafe-SAFETY comments,
  mod.rs purity) are evergreen and stay.

- **Dead disk-build infrastructure** — `block_pool.rs`,
  `block_column.rs`, `memory/build_column_store.rs`,
  `AsyncPropertyLogWriter`, plus a sweep of unused methods, fields,
  and enum variants. v3 disk pipeline replaced these; ~3000 lines net.

- **Legacy benchmark suites** — `test_nx_comparison.py` (NetworkX
  comparison, required scipy that wasn't in the venv) and
  `test_performance.py` (used the old `result["stats"][...]`
  subscript API, every Cypher-mutation test failed). Superseded by
  `test_bench_core.py` and `test_bench_memory.py`.

The "completeness round" — six phases that round out the Cypher and
Fluent surface so no primitive a user would reasonably expect is
missing. Every domain (legal, code, sodir, Wikidata) gets value from
each addition; none are domain-specific.

### Added

- **Cypher INTERSECT / EXCEPT (Phase 6 of the completeness round).**
  Cypher now exposes the standard set operators:

  ```cypher
  MATCH (n:Person) WHERE n.city = 'Oslo' RETURN n.name AS name
  INTERSECT
  MATCH (n:Person) WHERE n.age > 30 RETURN n.name AS name

  MATCH (n:Person) WHERE n.city = 'Oslo' RETURN n.name AS name
  EXCEPT
  MATCH (n:Person) WHERE n.age > 30 RETURN n.name AS name
  ```

  `INTERSECT` keeps rows present in both sides; `EXCEPT` keeps rows in
  left but not in right. Both always dedupe, matching SQL/openCypher
  conventions. Internals: new `SetOpKind` enum on `UnionClause` (the
  same `Clause::Union` variant carries all three operators). The
  executor dispatches on `kind`: UNION uses the existing concat-and-
  dedup path; INTERSECT/EXCEPT pre-build a row-hash set from the
  right-side result and filter the left.

  Brings Cypher in line with the fluent-API set ops (`union`,
  `intersection`, `difference`, `symmetric_difference`).

- **Geospatial primitives (Phase 5 of the completeness round).** Round
  out the spatial surface with the standard GIS toolkit operations.

  Cypher scalar functions on WKT/node geometries:
  - `geom_buffer(geom, meters)` — planar buffer.
  - `geom_convex_hull(geoms)` — variadic or list arg.
  - `geom_union(g1, g2)` / `geom_intersection(g1, g2)` /
    `geom_difference(g1, g2)` — boolean ops.
  - `geom_is_valid(geom)` — OGC validity.
  - `geom_length(geom)` — geodesic length for LineStrings; perimeter
    for polygons (sum of rings); 0 for points.

  Cypher CALL procedure:
  - `kg_knn({lat, lon, target_type, k})` YIELD `node, distance_m` —
    *k* nearest nodes of a target type to a coordinate (geodesic;
    location-first, falls back to geometry centroid).

  ```cypher
  CALL kg_knn({lat: 60.4, lon: 5.3, target_type: 'City', k: 5})
  YIELD node, distance_m
  RETURN node.title, round(distance_m / 1000.0, 1) AS km
  ```

  Backed by `geo = "0.33"` (`Buffer`, `BooleanOps`, `ConvexHull`,
  `Validation`, `LengthMeasurable` traits). New helpers in
  `src/graph/features/spatial.rs`; new `geom_arg` resolver in
  `executor/expression.rs` accepts WKT strings, Points, and
  spatial-configured node/property variables.

- **Weighted shortest path (Phase 4 of the completeness round).**
  `graph.shortest_path()` and `graph.shortest_path_length()` now
  accept an optional `weight_property` parameter. When set, the
  search switches from BFS (hop count) to Dijkstra (sum of edge
  weights). Edges missing the property fall back to weight 1.0
  (matching Louvain's existing weighted-adjacency convention);
  negative weights cause the path to be reported as missing.

  ```python
  result = graph.shortest_path(
      "Stop", "A", "Stop", "Z",
      weight_property="cost",
  )
  # {'path': [...], 'connections': [...], 'length': 3, 'weight': 4.7}

  graph.shortest_path_length(
      "Stop", "A", "Stop", "Z",
      weight_property="cost",
  )  # → 4.7 (float; int when unweighted)
  ```

  Internals: new `shortest_path_weighted()` and
  `shortest_path_cost_weighted()` in `algorithms/graph_algorithms.rs`,
  Dijkstra with a `BinaryHeap<State>` keyed on `(distance, node_idx)`.
  The existing BFS path remains the default — no overhead for
  unweighted callers.

- **Structural validators v2 — rule packs extension (Phase 3 of the
  completeness round).** Seven new CALL procedures complete the
  rule-pack family from 0.8.19 by covering *n*-ary and declarative
  checks:

  - `inverse_violation({rel_a, rel_b}) YIELD a, b` — declared-inverse
    relations not symmetric (e.g. `parent_of` without matching
    `child_of`).
  - `transitivity_violation({rel}) YIELD a, b, c` — `(a)-[rel]->(b)
    -[rel]->(c)` chains where the direct `(a)-[rel]->(c)` is absent.
    Generalizes the OCTF subclass-fold audit pattern.
  - `cardinality_violation({type, edge[, min, max]}) YIELD node, count`
    — declarative cardinality. Setting `max:1` catches functional-
    property violations; `min:1` catches missing-required-edge.
  - `type_domain_violation({edge, expected_source}) YIELD source, target`
  - `type_range_violation({edge, expected_target}) YIELD source, target`
    — schema integrity checks on edge endpoints.
  - `parallel_edges({edge}) YIELD a, b, count` — pairs connected by
    more than one edge of the same type (almost always an ETL bug).
  - `null_property({type, property}) YIELD node` — property side of
    `missing_required_edge`.

  All seven follow the existing rule-procedure pattern and surface
  via `CALL list_procedures()` and `describe(cypher=True)`.

- **Lexical text predicates (Phase 2 of the completeness round).** Six
  string-similarity primitives now expressible in Cypher without
  dropping to Python:

  - `text_edit_distance(a, b)` — Levenshtein, UTF-8 aware (uses
    minimum-row DP for O(min(n,m)) memory).
  - `text_normalize(s)` — lowercase, drop punctuation, collapse
    whitespace. The thing every fuzzy-match pipeline reaches for first.
  - `text_jaccard(a, b [, sep])` — token-set Jaccard, default whitespace
    separator.
  - `text_ngrams(s, n)` — character n-grams as a list.
  - `text_contains_any(s, needles)` / `text_starts_with_any(s, prefixes)`
    — variadic or list-argument forms; short-circuit on first match.

  ```cypher
  MATCH (a:Person), (b:Person) WHERE a.id < b.id
  WITH a, b, text_edit_distance(
      text_normalize(a.title), text_normalize(b.title)
  ) AS d
  WHERE d <= 2 RETURN a.title, b.title, d
  ```

- **Expression-engine fundamentals (Phase 1 of the completeness round).**
  The Cypher engine gains the standard scalar/aggregate/list-fold
  primitives that were missing:

  - `properties(n)` / `properties(r)` — full property map of a node or
    relationship (returns a JSON-formatted map; works alongside
    `keys()`).
  - `start_node(r)` / `end_node(r)` — endpoint access on a bound edge
    variable. `start_node(r).name` works via the existing dotted
    property accessor.
  - `reduce(acc = init, x IN list | body)` — list fold with
    accumulator. New `Expression::Reduce` AST variant; mirrors
    openCypher.
  - `percentile_cont(expr, p)` — continuous percentile via linear
    interpolation; `p ∈ [0,1]`.
  - `percentile_disc(expr, p)` — discrete percentile via nearest rank.
  - `median(expr)` — sugar for `percentile_cont(expr, 0.5)`.
  - `variance(expr)` / `var_samp(expr)` — sample variance, n-1
    denominator (matching the existing `std` convention).

  ```cypher
  MATCH (n:Person)
  RETURN median(n.age), percentile_cont(n.age, 0.9), variance(n.age)

  MATCH (n:Person) WITH collect(n.age) AS ages
  RETURN reduce(s = 0, x IN ages | s + x) AS total
  ```

### Changed

- **Example MCP server simplified to two tools.**
  `examples/mcp_server.py` now exposes only `graph_overview` and
  `cypher_query`. The convenience tools (`search`, `find_entity`,
  `read_source`, `entity_context`, `bug_report`) are removed — every
  operation is reachable from Cypher via `MATCH (n) WHERE n.title =
  $text` etc., and the docstring shows the equivalent patterns. Same
  simplification philosophy as the 0.8.19 rule-procedure refactor:
  lean on the Cypher surface.

## [0.8.19] — 2026-04-26

### Changed

- **Rule packs rebuilt as native Cypher CALL procedures.** The
  Python-layer `kglite.rules` package (`g.rules.run(...)`,
  `RuleReport`, YAML packs, ~1,200 lines) is removed; six
  structural-validator procedures live inside the Cypher engine
  alongside `pagerank` / `connected_components`:

  ```cypher
  CALL orphan_node({type: 'Wellbore'}) YIELD node RETURN node
  CALL missing_required_edge({type: 'Wellbore', edge: 'IN_LICENCE'}) YIELD node ...
  CALL missing_inbound_edge({type: 'Discovery', edge: 'IN_DISCOVERY'}) YIELD node ...
  CALL self_loop({type: 'Person', edge: 'KNOWS'}) YIELD node ...
  CALL cycle_2step({type: 'Person', edge: 'KNOWS'}) YIELD node_a, node_b ...
  CALL duplicate_title({type: 'Prospect'}) YIELD node ...
  ```

  Direct graph iteration in Rust replaces the YAML→Cypher→parse round
  trip — single rule on sodir (564k nodes) runs in **<2 ms** vs.
  ~5 ms for the legacy Python pack runner. Composability with
  surrounding Cypher (WHERE / ORDER BY / aggregation) collapses the
  previous two-step `rules_run + cypher_query` flow into a single
  pass.

  Direction validation, anchored type-by-type iteration, and the
  `DirectionMismatch` error survive — ported to Rust. Same agent
  protection without the parallel API.

  Discovery surface: rule procedures appear in
  `describe(cypher=True)` topic list, in the
  `<rules hint="..."/>` extension hint of `describe()`, and in
  `CALL list_procedures() YIELD name`. Per-procedure docs via
  `describe(cypher=['orphan_node'])`. No `<rule_packs>` block. No
  opt-in `advertise()` function. No separate `rules_run` MCP tool —
  agents invoke via `cypher_query`.

  **Breaking change.** Code using `g.rules.run(...)` or any
  `kglite.rules.*` import from 0.8.16–0.8.18 must migrate to the
  CALL syntax. The migration is mechanical: one `CALL` per rule
  with map-syntax parameters and `YIELD node` (or `YIELD node_a,
  node_b` for `cycle_2step`).

  Removed: `kglite/rules/` package, `g.rules` accessor on
  `KnowledgeGraph`, `Rule`/`RulePack`/`RuleReport`/`_RulesAccessor`
  classes, `kglite.rules.advertise()`, `_set_default_rule_pack_xml`
  PyO3 function, `_set_rule_pack_xml` PyO3 method, `rule_packs_xml`
  field on `KnowledgeGraph`, `inject_rule_packs` helper in
  describe.rs, the `<rule_packs>` block in `describe()`, the
  `rules_run` MCP tool from `examples/mcp_server.py` and
  `prospect_mcp_server.py`, `pyyaml>=6.0` runtime dependency.

## [0.8.18] — 2026-04-26

### Changed

- **Rule-pack discovery via `describe()` is now opt-in.** A fresh
  `kglite.load(...)` produces a `describe()` with no `<rule_packs>`
  block — graphs that don't use rule packs incur no agent-facing
  noise. Activation is explicit:
  - `g.rules.run(...)` or `g.rules.load(...)` activates per-graph
    advertising (existing behaviour, unchanged).
  - New `kglite.rules.advertise()` publishes a module-level default
    visible to every subsequent `describe()` across all graphs. Use
    this for MCP servers that expose a rule-pack tool. Idempotent.
  - The `examples/mcp_server.py` `rules_run` tool is now commented
    out; the file documents how to re-enable it for users who want
    rule packs in their MCP surface. The default MCP example is
    rule-pack-free.
- **Per-rule Cypher timeout via `default_timeout_ms`.** Rules can
  declare a YAML-level `default_timeout_ms` and the runner passes it
  as the `timeout_ms` to `g.cypher()` per rule. A caller-supplied
  `timeout_ms` to `g.rules.run(...)` always wins. Lets full-Wikidata
  users set realistic budgets on rules that scan dense node types
  (13M humans, 45M scholarly articles) without affecting the global
  graph timeout.
- **Rule-pack `describe()` integration moved into Rust.** Slice 1.1
  shipped agent-discovery via a Python monkey-patch that wrapped
  `KnowledgeGraph.describe`, called the Rust method (preserved as
  `_describe_native`), and post-processed the XML to splice in a
  `<rule_packs>` block. That dispatch is now native: `describe()`
  is the Rust method again. Per-instance pack XML lives on the
  `KnowledgeGraph` struct (`Mutex<Option<String>>`); a module-level
  default holds the cold bundled-pack inventory. Python's role
  shrinks to rendering the XML on pack `load()` / `run()` and
  pushing it via the new `_set_rule_pack_xml` PyO3 method (and the
  module-level `_set_default_rule_pack_xml`). User-visible XML and
  behaviour are byte-compatible; the wrapper indirection and its
  per-call `str.rfind`/slice/concat are gone.

### Fixed

- **`LIMIT` was applied before `WHERE` filtering, returning fewer rows
  than expected.** A query like
  `MATCH (n:T) WHERE NOT EXISTS { (n)-[:E]->() } RETURN n.id LIMIT 5`
  could return 0 rows when the first 5 candidate nodes all failed the
  WHERE predicate. Root cause: the planner pushed the LIMIT hint into
  `PatternExecutor`, which capped *candidates* before the inline
  WHERE filter ran. Fix: skip the limit hint at pattern-execution
  time when an inline WHERE is present, and apply LIMIT after the
  WHERE filter (as the surrounding executor already attempts to).
  Affects any filtered query with LIMIT, not just `NOT EXISTS`.

### Added

- **Rule packs** — agent-discoverable structural validators. New
  `g.rules` sub-namespace exposes `list()`, `load()`, `run()`, and
  `describe()` for named YAML packs that compile to Cypher and emit a
  structured `RuleReport`. The bundled `structural_integrity` pack
  (v1.1) ships six universal cross-graph rules: orphan nodes,
  self-loops, short cycles, missing-required-edge (outbound),
  missing-inbound-edge, and duplicate titles. `g.describe()` surfaces
  a `<rule_packs>` block so agents discover packs through the same
  XML they consume for schema. See
  [`docs/guides/rules.md`](docs/guides/rules.md). Reports are lazy:
  `.summary` returns counts without materialising rows, and runs are
  cached per `(pack_name, params, graph)`. New runtime dependency:
  `pyyaml>=6.0`.
- **Rule-pack ergonomics:**
  - `summary["any_truncated"]` — top-level boolean so agents can
    one-glance check if any rule hit its LIMIT.
  - `report.is_suspect(node_id)` — O(1) cross-reference helper that
    returns `[(rule_name, severity), ...]` for rules that flagged the
    node. Built lazily; accepts string or int ids.
  - `g.rules.list()` now reads bundled-YAML headers lazily so cold
    inventory shows real version + rule_count + description (no
    placeholders) before any pack has been loaded.
  - Optional `usage_hint:` field on a pack — surfaced via
    `g.rules.describe(name)` and as an XML attribute in
    `g.describe()` so agents can read "use this pack when…" guidance
    inline with the schema.
  - `to_markdown()` truncates list-typed cells (e.g. the `ids`
    column in `duplicate_title`) to 3 elements + " (+N more)" so
    agent-pasted output stays readable.
  - **Direction-aware `missing_*_edge` rules.** New optional
    `validates_direction:` rule field (`"outbound"` or `"inbound"`).
    The runner inspects `g.connection_types()` and refuses to execute
    when the `(type, edge)` pair flows the wrong way in the graph's
    actual schema, surfacing a `DirectionMismatch` error that
    suggests the right rule. The bundled `missing_required_edge` and
    `missing_inbound_edge` opt in. Prevents trivial rule firing where
    e.g. asking for incoming `IN_LICENCE` on a `Wellbore` would have
    matched every wellbore meaninglessly.

## [0.8.17] — 2026-04-26

### Performance

- **Two-MATCH count fusion: top-K-by-degree filtered queries now run
  ~20× faster.** The shape
  `MATCH (w)-[:T]->(b {nid:'X'}) MATCH (w)-[r]-() WITH ...
  count(r) ... ORDER BY count DESC LIMIT k`
  used to materialise one row per edge for every group key (e.g. 4 M
  edge rows for 416 k Wikidata writers — 494 s on the full graph).
  The aggregation-fusion pass at
  `src/graph/languages/cypher/planner/fusion.rs` now also recognises
  `[Match, Match, With(count)]` and folds it into a single
  `FusedMatchWithAggregate` whose secondary pattern drives the
  per-group-key degree count via the existing `count_edges_filtered`
  fast-path. Measured: top-10-by-degree on Wikidata writers
  **494 s → 24 s (20×)**. The remaining time is one degree lookup
  per group key (832 k mmap reads) — further wins live in storage,
  out of scope for this session.
- **`count_edges_filtered` fast-path now handles undirected `[r]-`
  edges.** Previously the fast-path returned `None` for
  `EdgeDirection::Both`, forcing the slow per-edge enumeration. It
  now sums incoming + outgoing `count_edges_filtered` calls — the
  canonical "total degree" pattern. Both the new two-MATCH fusion and
  the existing single-MATCH `WITH count` benefit.
- **Per-group-key count phase in `execute_fused_match_with_aggregate`
  now runs in parallel** above 4 096 group keys. Each
  `count_edges_filtered` call is a read-only mmap lookup, so rayon's
  par_iter overlaps the per-call I/O instead of serialising it.
  Measured on the same Wikidata top-10-by-degree query
  (124 M nodes / 861 M edges): **24 s → 8.5 s (~2.5×)** on top of
  the fusion win. End-to-end wall on `top_writers.py` is now 73 s,
  vs. 510 s before any of this session's work (~7× total).
- **Top-K hint absorbed into `FusedMatchWithAggregate`.** A new
  planner pass `fuse_match_with_aggregate_top_k` recognises the
  shape `[FusedMatchWithAggregate, Return, OrderBy(count_alias),
  Limit(k)]` (where the RETURN is a pure pass-through projection)
  and pushes the K-bound into the fused stage. The executor sorts
  by count first and then evaluates the group-key projection
  expressions only for the K winners — saves N×P
  `evaluate_expression` calls when N is large and K is small. The
  Wikidata top-10-by-degree query goes from materialising 416 k
  rows to 10; per-row property reads stop being the tail cost
  (modest 5% gain on top of the parallel-count win, but principled:
  "only do necessary work" — projection-heavy queries get a much
  larger benefit).
- **Lazy RETURN — defer per-row property evaluation until Python
  reads each cell.** The planner's new `mark_lazy_eligibility`
  pass annotates the terminal RETURN with `lazy_eligible = true`
  when the query is `MATCH … (WHERE …) RETURN <prop access>` and
  there's no downstream operator that needs row values
  (DISTINCT/HAVING/ORDER BY/aggregate/WITH/UNWIND/CALL/UNION/
  mutation all force the eager path). The executor skips
  `execute_return_projection`'s per-row loop and hands the
  pending rows + return items to the Python `ResultView` via a
  side-channel `LazyResultDescriptor`. `ResultView` materialises
  cells on access (memoised via a `Mutex<Vec<Option<…>>>` so
  repeat reads are free), and `__len__` becomes O(1). Measured
  on the same Wikidata script: the find-writers query
  (`MATCH … RETURN nid, title`, used only for `len()` in the
  caller) **57 s → 35 s (~1.6×)**. End-to-end on
  `top_writers.py` is now **49 s** — down from the original
  **510 s** before this session — **~10× total**.

### Performance

- **Phase 1 N-Triples loader is ~1.7× faster.** Steady-state on
  Wikidata's `latest-truthy.nt.bz2` went from ~2.4 M tri/s to
  **~4.1 M tri/s** (`--size 50` build dropped 20.83 s → 15.96 s; full
  Wikidata Phase 1 projects from ~2 h to **~70 min**). Profiling with
  `samply` showed the loader thread spending ~32% of CPU in
  `libsystem_malloc` and ~10.7% in `core::str::pattern::TwoWaySearcher`
  (used by `str::find`). Four targeted changes:
  - **Byte-level `parse_line`** (`src/graph/io/ntriples/parser.rs`):
    swap `line.find("> ")` for `memchr::memchr(b'>')`. URIs in
    N-triples cannot contain `>`, so a single byte scan is sufficient.
  - **`EntityAccumulator` capacity preallocation**
    (`HashMap::with_capacity(32)`, `Vec::with_capacity(8)`):
    eliminates `RawVecInner::finish_grow` reallocs in the per-entity
    accumulator.
  - **`scratch_props` reuse in `flush_entity`**: hoist the
    `Vec<(InternedKey, Value)>` out of the function so the alloc cost
    is paid once per build instead of per entity.
  - **`mimalloc` as global allocator** (`src/lib.rs`): pure Rust /
    build-time-only dependency; ~10% wall-time win on the loader on
    top of the parser-side changes.
- **Reader-thread channel batches 50k → 200k.** Reduces per-batch
  sync overhead 4× without growing peak RSS meaningfully.

### Added

- **Block-level parallel decoder for single-stream `.bz2` files.** Wikidata
  ships `latest-truthy.nt.bz2` as a single bz2 stream, so the existing
  stream-level scanner in `parallel_bz2.rs` was falling through to a
  single-threaded `MultiBzDecoder` (~1 M triples/s ceiling). The new
  single-stream path delegates to `bzip2_rs::ParallelDecoderReader`
  (paolobarbolini/bzip2-rs, MIT/Apache-2.0), which finds bit-aligned
  block magics inside one stream and decodes blocks on rayon workers.
  Measured Phase 1 throughput on Wikidata: **~1.0 M tri/s → ~3.3 M tri/s
  (3.3× speedup)** at the same memory ceiling. Multistream files still
  use the existing stream-level path. Pinned to a git rev because the
  published 0.1.2 crate ships an older `Cargo.toml` without the `rayon`
  feature flag.
- **Phase 1 progress bar now shows ETA when `max_entities` is set.**
  When the caller has set an entity cap, the loader emits the bar
  position as `entities_created` against `total = max_entities`, so
  tqdm can compute ETA from the entity rate. Without a cap the bar
  still tracks triples (no total → no ETA, just rate). The unused
  counter ships in the event's `fields` dict either way.
- **Ctrl+C cancellation of `load_ntriples` builds.** Phase 1 runs
  inside `py.detach()` (GIL released so Python heartbeat threads can
  run), which previously meant SIGINT couldn't reach Python until the
  Rust call returned — i.e. Ctrl+C did nothing during a multi-hour
  Wikidata build. The progress sink now reacquires the GIL on every
  update event and calls `Python::check_signals`; a pending SIGINT
  flows back through a new `Cancelled` marker on `ProgressSink::emit`,
  unwinds the loader cleanly, and surfaces as `KeyboardInterrupt` on
  the Python side. Cancellation requires a `progress=` callback (which
  is the default in the bench script and dataset wrappers).

### Changed

- **`bench/wikidata_e2e.py` CLI overhaul.** `--progress` is now the
  default (use `--legacy-progress` for the old `[Phase X]` stderr
  output). Removed `--quiet` — wrapper-level status messages
  (cooldown checks, cache hits) print regardless; the loader's
  per-phase eplog lines are auto-silenced when tqdm is active so they
  don't fight the bar. Renamed `--size` to `--entities-m` to make it
  clear the cap is in millions of *entities*, not triples (`--size`
  remains as a deprecated alias).
- **`kglite.datasets.wikidata.open` auto-silences the loader when
  `progress=` is set.** Wrapper-level `verbose=True` controls the
  cache-hit / cooldown messages; loader `verbose` is forced off when
  a progress callback is wired so tqdm owns the terminal.

### Added (continued)

- **Structured build-phase progress callback for `load_ntriples`.** New
  `progress=` kwarg accepts a Python callable that receives one dict
  per phase event (`start` / `update` / `complete`) for each of
  `phase1` (streaming), `phase1b` (columnar build), `phase2` (edges),
  `phase3` (CSR), and `finalising`. Phase 1 fires updates every 5M
  triples (decoupled from the 60s stderr gate) so a UI driven by the
  callback stays live. Errors raised by the callback are swallowed so
  a broken UI cannot kill a multi-hour build.
  Pure-Rust trait `ProgressSink` lives in
  `src/graph/io/ntriples/mod.rs`; the PyO3 adapter that translates a
  Python callable into a sink lives in `src/graph/pyapi/kg_core.rs`,
  keeping the loader free of `pyo3` types.
- **`kglite.progress.TqdmBuildProgress`** — drop-in tqdm-backed
  reporter. One bar per phase, with RSS (via `psutil`) and per-phase
  counters in the postfix. `pip install tqdm psutil` to use.
- **`bench/wikidata_e2e.py --progress`** — opt-in flag that wires
  `TqdmBuildProgress` into the e2e benchmark.

### Fixed

- **`load_ntriples` no longer panics with `slice index starts at A
  but ends at B` on large disk/mapped builds.** Wikidata builds past
  ~450 M triples crashed in `MmapColumnStore::read_str` on reload.
  Root cause: `flush_entity` for entities whose ID didn't parse as a
  Q-code wrote `Value::String(acc.id)` into the `nid` column, which
  flipped that column's `id_is_string=true`. Subsequent entities with
  `Value::UniqueId` left their string offsets uninitialised (zero), so
  reload hit `start > end` decoding them. Fix: in disk/mapped mode,
  skip entities whose ID is not a parseable Q-code at the top of
  `flush_entity` (these were unreachable in the canonical Wikidata
  query surface anyway). In-memory mode is unaffected.
- **Cypher `DETACH DELETE` no longer breaks subsequent typed-edge
  traversals.** Pre-fix, after a Cypher `DETACH DELETE`, fluent
  `g.select(t).traverse(conn_type, ...)` (and any `make_traversal`
  caller) would throw *"Connection type 'X' does not exist in
  graph"* even when `X` still had millions of live edges. The
  Cypher executor's `execute_delete` invalidated the
  `edge_type_counts_cache` but left the `connection_types`
  HashSet alone — `has_connection_type()` consults the HashSet
  first and returned a stale negative. Fix: clear the
  `connection_types` cache on Cypher delete, and add a final
  fall-through in `has_connection_type()` to the disk backend's
  authoritative `conn_type_index_*` arrays.
  Surfaced by `bench/benchmark_full.py` against every disk row.

### Performance

- **Parallel multistream `.bz2` decoder for `load_ntriples` —
  closes the gap with `.zst`.** Wikidata / pbzip2 dumps are a
  concatenation of independent bz2 streams; the previous
  `bzip2::read::MultiBzDecoder` walked them sequentially on a
  single core. New `parallel_bz2::open()` (in
  `src/graph/io/ntriples/parallel_bz2.rs`) scans the file for
  `BZh[1-9]` + 6-byte block magic, dispatches streams to a
  worker pool sized by a memory budget (256 MB default, after
  pbzip2's `NumBufferedBlocksMax`), and re-orders the
  decompressed chunks behind a single `Read` surface. Single-
  stream `.bz2` files take a fast path through `MultiBzDecoder`
  with no thread-pool overhead. Workers join before
  `load_ntriples` exits Phase 1, so no parallelism leaks into
  the Phase-2/3 rayon pool.
  Measured: wiki100m bz2 **99 s → 34 s (2.9×)**, wiki200m
  **199 s → 71 s (2.8×)**. bz2/zst ratio 3.4× → 1.19×.
- **`enable_columnar()` is now idempotent on the already-columnar
  fast path.** Previously, every `g.save()` re-ran the full
  per-node columnar rebuild — even when the graph was already
  columnar and unmodified. At wiki100m memory mode this cost
  ~257 s of pure waste on consecutive saves. Now `enable_columnar`
  walks every node once (O(N) cheap matches) and short-circuits
  if all nodes are `PropertyStorage::Columnar` AND their
  `Arc<ColumnStore>` matches `graph.column_stores` for the type
  (the Arc-pointer check catches the common
  `add_nodes(conflict_handling="update")` fork pattern that would
  otherwise lose updates on save). Measured wiki5m: consecutive
  `save()` 455 ms → **177 ms (2.6×)**.

### Changed

- **`verbose=True` on `load_ntriples` is now phase-oriented.** Previous
  output was a mix of `[T+30s]` timestamps and ad-hoc sub-step prints;
  Phase 2/3 output in particular was developer-grade noise. The new
  output is a small set of `[Phase N]` gate messages — open/close
  pairs around each major stage of the build:
  ```
  [Phase 1] Streaming and parsing N-triples (...)
  [Phase 1] 12.3M triples, 2.8M entities, 8.5M edges buffered — 205k triples/s
  [Phase 1] Complete: ... in 47m18s
  [Phase 1b] Building columnar storage (...)
  [Phase 1b] Complete in 8m42s
  [Phase 2] Creating edges
  [Phase 2] Complete: ... edges in 11m22s
  [Phase 3] Building CSR edge index
  [Phase 3] Complete in 2m04s
  [Finalising] Building auxiliary indexes + saving metadata
  [Finalising] Complete in 32s
  [Build] Total elapsed: 1h09m54s
  ```
  Sub-step timings (CSR step 1/4, peer-count histogram, mmap layout,
  per-type flush logs, interner save timings, Q-code resolution
  timings, …) move behind `KGLITE_BUILD_DEBUG=1`. The legacy
  `KGLITE_CSR_VERBOSE` env var is replaced by `KGLITE_BUILD_DEBUG`
  (one flag for all build sub-step output).

### Added

- **`kglite.datasets.sodir.open(workdir, ...)` — one-call lifecycle for
  Sodir factmaps petroleum data.** Resolves CSVs from the public
  ArcGIS FeatureServer at
  `https://factmaps.sodir.no/api/rest/services/DataService`, applies
  the FK pre-processing the existing build script does, and builds
  the graph via the packaged blueprint. Default storage is
  ``memory`` — Sodir is small enough that disk caching adds little
  on top of CSV caching:

      g = sodir.open("/data/sodir")  # memory; index_cooldown_days=14, dataset_cooldown_days=30
      g = sodir.open("/data/sodir", storage="disk")  # opt-in for cross-process reuse

  Workdir layout: `csv/` (fetched datasets, flat), `sodir_index.json`
  (per-dataset row count + timestamps), `graph/` (disk-mode only).
  Index sweep cheaply re-checks remote row counts every 14 days; only
  changed datasets re-download. Hard cooldown forces full per-dataset
  refresh every 30 days even if counts match.

  **Complement blueprints**: pass
  ``complement_blueprint=path/to/extra.json`` to add new node types
  / edges on top of the packaged baseline. The file is persisted to
  ``workdir/blueprint_complement.json`` on first call and auto-loaded
  on subsequent calls. Pass ``use_complement=False`` to skip it for a
  single call, or ``sodir.remove_complement(workdir)`` to drop it
  permanently. Deep merge with **base-wins on key collisions by
  default** (the packaged baseline tracks the canonical Sodir REST
  catalog and stays authoritative); set
  ``complement_overrides=True`` to flip when the complement should
  win.

  The blueprint walker auto-detects which datasets are referenced and
  fetches only those — adding new node types to the blueprint
  triggers fetches the next time `open()` runs. The packaged
  baseline ships only the 33 node types whose CSVs are fetchable
  from REST (no sideloaded prospect / play / ocean data); use a
  complement to layer those in.

  **Parallel fetcher**: ``workers`` parameter (default 4) drives a
  thread-pool that pulls dataset jobs off a shared backlog. tqdm
  progress bar replaces per-dataset prints so verbose output stays
  one line tall. Geometry handler is defensive against empty
  ``coordinates: []`` (some pre-1970 Sodir wellbores) — drops those
  features' geometry without aborting the fetch. KGLite is independent of Sodir / the
  Norwegian Offshore Directorate; see module docstring. Catalog
  (LAYERS / TABLES / FACTMAPS_LAYERS) vendored from
  `kkollsga/factpages-py`.

- **`kglite.datasets.wikidata.open(workdir, ...)` — one-call lifecycle
  for Wikidata `latest-truthy` graphs.** Resolves the dump (download,
  resume `.part`, refresh on cooldown), builds the disk or in-memory
  graph, and returns it. Subsequent calls cache-hit on the saved
  graph at `workdir/graph[_<N>m]/`. Two storage backends:

      g_full = wikidata.open("/data/wd")                        # disk graph
      g_100m = wikidata.open("/data/wd", entity_limit_millions=100)
      g_mem  = wikidata.open("/data/wd", storage="memory",      # rebuild every call
                              entity_limit_millions=10)

  Sized slices (`entity_limit_millions=100/200/...`) live alongside
  the full graph (`graph_100m/`, `graph_200m/`, `graph/`) and all
  share the same `latest-truthy.nt.bz2` dump under `workdir`.

  Also exports `fetch_truthy(workdir, cooldown_days=31)` for the
  dump-only path. KGLite is independent of the Wikimedia Foundation;
  see module docstring.

- **`load_ntriples` releases the GIL.** Multi-minute loads no longer
  block Python threads — heartbeat / progress monitors and other
  background workers run on schedule throughout the build. Required
  for the new minute-cadence reporting in
  `examples/wikidata_disk.py`.
- **`bench/benchmark_full.py`** — full-stack lifecycle benchmark
  (build / save / load / mutate / resave / Cypher / fluent) across
  every storage mode × Wikidata subset. Wide-pivot CSV output —
  one row per (run, mode, dataset). Errors preserved both in the
  `errors` column (truncated, semicolon-separated) and in a
  sidecar `bench/benchmark_full.errors.log` (full text,
  tab-separated for `cut`-friendly inspection).
- **`bench/results.py`** — pandas-backed analysis tool over the
  bench CSV. Three commands: `latest` (most recent measurement
  per cell), `trends` (per-run time-series for filtered cells),
  `deltas` (consecutive-run deltas — surfaces regressions). Use
  `--mode` / `--dataset` / `--cols` filters.
- **`--languages` CLI flag on `bench/wiki_benchmark.py` and
  `bench/api_benchmark.py`** (default `en`). Matches the existing
  flag on `bench/wikidata_e2e.py`. Threads through subprocess
  scenarios via argv (wiki_benchmark) and `KGLITE_BENCH_LANGUAGES`
  env (api_benchmark, which uses positional args). Pass
  `--languages ""` to keep all languages, but the canonical query
  suite expects English type names like `:human` and will return
  zero rows otherwise.
- **Post-build SANITY PROBE in `bench/wikidata_e2e.py`.** Quick Q42
  title/description lookup + top-5 type histogram printed after
  every build, even with `--no-queries`. Catches language-filter or
  auto-type-rename regressions immediately, before query-suite time.

## [0.8.15] — 2026-04-25

### Performance

- **Mapped-mode property index — `MATCH (n:Type {prop: val})` in O(log N).**
  `MappedGraph` now carries a lazy per-`(node_type, property)` and
  cross-type property index alongside the 0.8.15 conn_type index. On
  first `lookup_by_property_eq` / `lookup_by_property_prefix` /
  `*_any_type` hit the backend iterates nodes once, emits a sorted
  `(key, NodeIndex)` array — same layout as disk's persistent
  `PropertyIndex` — and caches it behind an `Arc<RwLock<…>>`;
  subsequent queries binary-search that array. Alias handling
  matches disk (reads from `node.title` for `title`/`label`/`name`
  and `node.id` for `id`/`nid`/`qid` so the `add_nodes(...,
  node_title_field=...)` pipeline "just works"). Invalidated on
  `add_node`/`remove_node`/`node_weight_mut`. Measured on a 938 k
  node / 212 k Q5 subset (wiki100m Wikidata humans):
  `MATCH (n:Q5 {title: 'Douglas Adams'})` — **37 ms first call
  (builds the index), 0.1 ms warm** (index hit). Prefix scans stay
  in the single-digit-ms range. Correctness pinned by
  `tests/test_mapped_property_index.py` against both memory and disk.

### Added

- **Cypher `count { <pattern> }` subquery expression.** `count { ... }`
  in `WITH` / `RETURN` / `ORDER BY` / `WHERE` now evaluates to the
  number of matches of the inner pattern, scoped to the outer row's
  bindings. Previously the parser rejected the shape with
  *"Expected property name or .property in map projection, got
  Some(LParen)"* because the identifier-followed-by-brace dispatch
  routed to map projection (`n { .prop1, .prop2 }`) unconditionally;
  the parser now special-cases `count` and routes to a new
  `parse_count_subquery` that mirrors the existing `EXISTS { ... }`
  grammar. New AST variant `Expression::CountSubquery`; the executor
  runs the pattern via the shared `PatternExecutor`, bindings-
  compatible with the outer row, with optional inline `WHERE`.
  Parity across all three storage modes verified by
  `tests/test_cypher_count_subquery.py`. Cypher shapes like
  `WITH a, count{(a)-[:REL]->()} AS n` now work out of the box.

### Performance

- **Mapped-mode query acceleration — lazy per-connection-type index.**
  `MappedGraph` was a bare `StableDiGraph` wrapper with none of the
  inverted indexes that make disk-mode fast (`conn_type_index_*`,
  `peer_count_*`, CSR sorted by type). Cypher queries that depend on
  those structures on disk did full-graph scans on mapped — 2-10×
  slower than disk despite every byte being in RAM. 0.8.15 adds a lazy
  `MappedTypeIndex` populated on first typed-edge query per connection
  type: CSR-style sorted source lists, per-peer count histograms, and
  per-source edge slices. Overrides `sources_for_conn_type_bounded`,
  `lookup_peer_counts`, and `count_edges_grouped_by_peer` on the
  mapped `GraphRead` impl; `filter_by_connection` (powering
  `where_connected`) now hoists the source list into a `HashSet` once
  per call instead of probing `edges_directed_filtered` per node.
  Measured on wiki1000m (1 B triples):
  - Cypher `P31 class counts`: **150 ms → 0.5 ms (300×)**.
  - Cypher `2-hop P31 + P279`: **180 ms → 0.9 ms (200×)**.
  - Cypher `P31 LIMIT 50`: **5.6 s → 0.8-1.1 s (5-7×)**, now **beats
    disk at every subset** (disk wiki1000m was 1.40 s).
  - Cypher `Q5 (human) lookup`: 77 ms → 50-54 ms (1.5×).
  - Fluent `traverse P31 out unlimited`: unchanged (74 ms; bare
    petgraph scan is already optimal for sparse-degree nodes and
    avoids the ~100 ns/call index-lookup overhead).
  Correctness preserved: same row counts across storage modes on
  every benchmarked query. Index is built lazily per conn_type,
  amortised across subsequent queries of the same type, and
  invalidated on any edge mutation.

- **Mapped-mode `load_ntriples` routes through the disk fast path.**
  Previously `storage="mapped"` fell through to `DirGraph::enable_columnar`,
  which iterates every node once, clones each property map into a `Vec`,
  and pushes row-by-row into per-column `MmapOrVec` instances that grow
  via `set_len` + remap; each schema extension additionally triggered
  `Arc::make_mut` store clones. On wiki50m (377 k nodes / 282 k edges)
  this ran at ~430 k triples/s with a 5.1 GB peak RSS. Mapped now shares
  the disk path's property-log + single-`columns.bin` pipeline:
  properties stream to a zstd-compressed log during Phase 1, Phase 1b
  replays the log once into a pre-allocated mmap, and a new second-pass
  links each node's `PropertyStorage` to the shared
  `Arc<ColumnStore>` by row_id. Measured (bench/wiki_benchmark_mapped):
  - wiki50m build: **116 s → 13.5 s (8.6× faster)**, peak RSS
    **5073 MB → 828 MB (6× less memory)**.
  - wiki100m build: **? → 29 s**, peak RSS **? → 1.5 GB** — now
    within 1% of disk-mode build time.
  - Same Cypher + fluent rowsets across modes (round-trip oracle in
    `tests/test_incremental_columnar.py::TestNTriplesColumnar`).
  Memory-mode N-Triples load (`storage=default`) is unchanged — still
  goes through the non-columnar `PropertyStorage::Map/Compact` path.

- **Fluent traverse + `where_connected` use CSR-filtered edge iterator.**
  `core/traversal.rs` (`make_traversal_fast`, `make_traversal_full`) and
  `core/filtering.rs` (`filter_by_connection`, i.e. `.where_connected()`)
  previously called `graph.edges_directed(node, dir)` and post-filtered on
  `connection_type`. On disk-mode graphs with `csr_sorted_by_type=true`
  (the `merge_sort` algo that the `wikidata_disk.py` example uses), we now
  pass the connection key into `edges_directed_filtered` so the DiskEdges
  iterator can binary-search the CSR range — O(log D) instead of O(D) plus
  per-edge `EdgeData` materialisation. This is the same fast path the
  Cypher executor has used since 0.8.0 and targets the shape that the
  Wikidata fluent suite regressed on: `select("Entity").traverse("P31",
  limit=100)` was ~71 s, `.where_connected("P31")` was ~887 s, and
  `traverse P31 in limit 50` was ~2171 s. Heap backends ignore the hint;
  correctness is preserved by the existing post-filter.

### Changed

- **`load_ntriples` verbose output is no longer per-type.** Phase 1b used
  to print one `N dense cols, M overflow cols` line per type with any
  overflow and one `overflow bag X MB for N sparse cols` line per type
  with a non-empty overflow bag. On Wikidata this was ~90 k lines. Both
  are now collapsed into a single Phase 1b summary per pass (`columns — N
  dense, M overflow across K types with sparse cols` and `overflow bags —
  X.X MB across N types, M sparse cols total`).
- **Load-phase progress lines throttled to time (≥ 15 s), not bucket.**
  The loader used to emit a progress line every 5 M triples, which was
  ~2× per second on a fast machine and scrolled the interesting phase-
  timing lines off screen. Now the 5 M bucket is still a cheap fast-loop
  counter but the line only fires when ≥ 15 s have passed since the last
  one. Lines now prefix `[T+NNNNs]` so they interleave cleanly with the
  existing phase-timing output.

### Fixed

- **`examples/wikidata_disk.py`** — the fluent cases called `.where_(...)`,
  but the PyO3 binding exposes the method as `.where(...)`; on a large
  graph this masked the real traversal slowness behind an
  `AttributeError`. Expanded the Cypher/fluent suites with 23 + 15 more
  diverse queries (typed 1-hop, 2-hop chains, parameter binding,
  `ORDER BY` on bounded scans, and string-prefix/contains filters). The
  fluent suite now introspects via `g.node_type_counts()` and skips
  unavailable types instead of raising.

## [0.8.14] — 2026-04-24

### Performance — disk-graph `kglite.load()` fast-load series

Four independent on-disk format changes aimed at the serde overhead
that dominates `kglite.load()` on large disk-mode graphs. Profiling
the 124 M-node, 863 M-edge Wikidata graph
(`wikidata_disk_graph_0.8.11`, 81 GB on disk) showed zstd
decompression + mmap setup account for only ~5–6 s of a ~77 s cold
load; the remaining ~70 s is serde rebuild cost on three bulk
structures, plus a 266 MB JSON array inside `metadata.json`. Each
format change replaces that cost with flat packed slices + exact
`HashMap::with_capacity` sizing — same in-memory representation,
zero consumer surface change.

- **`type_connectivity` out of `metadata.json` into a packed binary.**
  On the 81 GB graph this field was 266 MB of a 415 MB JSON file
  (3,176,503 `ConnectivityTriple` entries). The new
  `type_connectivity.bin.zst` at the graph root is:

  ```
  [ 0.. 8]  magic       = b"KGLTCN1\0"
  [ 8..12]  version     = u32 LE (= 1)
  [12..16]  num_entries = u32 LE
  [16..n*32+16]  entries: (u64 src_key, u64 conn_key, u64 tgt_key, u64 count) × n
  ```

  Keys are interner hashes (`InternedKey::as_u64()`). Disk-mode save
  strips the field from metadata.json; in-memory `.kgl` saves keep
  embedding it for single-file portability.

- **`type_indices.bin.zst` flat CSR binary, interner-keyed.** Replaces
  bincode `HashMap<String, Vec<NodeIndex>>` with three packed slices:

  ```
  [ 0.. 8]  magic       = b"KGLTIDX1"
  [ 8..12]  version     = u32 LE (= 1)
  [12..16]  num_types   = u32 LE
  [16..24]  total_nodes = u64 LE
  [24..24 + 8·num_types]        type_keys: [u64]
  [next..next + 8·(num_types+1)]  offsets:  [u64]   (CSR)
  [next..next + 4·total_nodes]   nodes:    [u32]
  ```

  HashMap capacity is sized exactly from `num_types`, and each type's
  `Vec<NodeIndex>` is built from a contiguous u32 slice rather than
  bincode's per-field serde calls.

- **`id_indices.bin.zst` per-variant flat binary.** Replaces bincode
  `HashMap<String, TypeIdIndex>` with:

  ```
  [ 0.. 8]  magic     = b"KGLIIDX1"
  [ 8..12]  version   = u32 LE (= 1)
  [12..16]  num_types = u32 LE
  per-type block:
    [ 0.. 8]  type_key:    u64 LE
    [ 8.. 9]  variant_tag: u8  (0 = Integer, 1 = General)
    [ 9..16]  padding:     [u8; 7]
    [16..24]  num_entries: u64 LE
    payload:
      Integer (tag=0):  keys: [u32], node_idxs: [u32]
      General (tag=1):  blob_len: u64 + bincode HashMap<Value, NodeIndex>
  ```

  The Integer variant dominates Wikidata-style graphs (Q-number ids
  strip to u32), so the bulk of the 997 MB decompressed bincode blob
  collapses to two flat u32 arrays per type.

- **`interner.bin.zst` replaces `interner.json`.** The hash→string
  JSON map becomes a zstd-compressed bincode `Vec<String>` of just
  the originals; hashes are re-derived deterministically on load via
  `get_or_intern`. On the 81 GB graph this drops from ~7 MB JSON to
  ~3 MB bincode and eliminates one JSON parse on the critical path.

- **`KGLITE_LOAD_TIMING=1` stage instrumentation.** Gated per-stage
  wall-clock timing in `load_disk_dir`; off by default (zero
  overhead). Emits one `[TIMING] stage=<name> dur_ms=<ms>` line to
  stderr per major phase (`metadata_json`, `interner_load`,
  `disk_graph_load`, `type_indices_load`, `column_stores_load`,
  `id_indices_load`, `type_connectivity_load`). Used as the
  measurement harness for the four format changes.

**Backward compatibility.** All four loaders fall back to the old
format on a missing file or magic-byte mismatch:

- Missing `type_connectivity.bin.zst` → loader reads embedded JSON
  from `metadata.json`, then derives from `connection_type_metadata`.
- `type_indices.bin.zst` without `KGLTIDX1` magic → old bincode
  `HashMap<String, Vec<NodeIndex>>` path.
- `id_indices.bin.zst` without `KGLIIDX1` magic → old bincode
  `HashMap<String, TypeIdIndex>` path.
- Missing `interner.bin.zst` → old `interner.json` path.

Graphs saved by 0.8.11 and 0.8.12 continue to load without a rewrite.
Re-saving an old-format graph with 0.8.13 produces all four new files
automatically.

**Non-goals / out of scope.** No change to query execution, pattern
matcher, mutation paths, `node_mut_cache` / F1 / F2 flow, segmented
CSR layout (`seg_NNN/`), or the `KGLCOLv1` sidecar format. In-memory
representation of every touched structure is byte-identical to
0.8.12 — zero possibility of query-side regression.

## [0.8.12] — 2026-04-24

### Fixed

Seven disk-backend correctness fixes stacked on top of the 0.8.11
segmented-CSR foundation. Five cover latent save/reload regressions
that slipped through 0.8.11's phase-1–8 coverage; two (F1 + F2) close
the pre-existing Cypher `SET` and `DETACH DELETE` mutation holes on
disk graphs.

- **`save_disk` no longer compacts overflow away before seal.**
  The previous `dg.has_overflow()` gate unconditionally compacted
  before `save_to_dir`, which cleared `overflow_out`/`overflow_in`
  and made the phase-6 seal path see empty overflow. Every edge
  added between saves was silently lost on reload. Gating the
  compact on "won't take the seal path" (manifest empty OR no
  tail above the sealed watermark) preserves overflow for seal
  and keeps the compact-rewrite semantics for the non-seal case.

- **Segment-local seals merge `conn_type_index_sources` with global ids.**
  `write_conn_type_index` walks the segment's segment-local
  `out_offsets` (indices 0..tail_len) and stored those local
  indices as source ids. Reload's merge needed to shift each
  entry by `node_lo[seg]` for segment-local seals (full-range
  seals already store global ids). Without this, post-reload
  `MATCH (a)-[:T]->(b) RETURN a.id, b.id` returned no rows even
  though `count(*)` via the histogram reported the correct total.

- **Compact-rewrite after a prior seal cleans up stale `seg_NNN`.**
  When `save_to_dir` falls to compact-rewrite (tombstones, edits,
  or pure edge mutations between existing nodes), it now removes
  every `seg_NNN > 0` under the target dir before rewriting
  `seg_000`. Without this, `enumerate_segment_dirs` picked up the
  stale sealed segments on reload and concat'd them against the
  fresh seg_000, double-counting nodes and edges.

- **Compact-rewrite persists heap-backed core arrays.**
  `reconcile_seg0_csr` (called inside seal) replaces
  `self.{node_slots, out_offsets, in_offsets, edge_endpoints}`
  with heap-backed `MmapOrVec::Heap` copies. The same-dir
  compact-rewrite previously relied on mmap persistence and
  skipped explicit writes, which left the on-disk files at the
  pre-seal trimmed sizes; reload errored with "File too small".
  `save_to_file` is now called unconditionally for every core
  array — it handles both backings.

- **New types added via `add_nodes` persist on disk save.**
  `DirGraph::save_disk` gated the per-type
  `columns/<type>/columns.zst` sidecar write on the absence of
  `columns.bin`. For disk graphs built via `load_ntriples`,
  `columns.bin` is always present, so the sidecar branch was dead
  code and every type added after the initial build lost its
  column data on reload (properties read back as `None`). Save
  now reads `columns_meta.json`/`.bin.zst` to identify the types
  already covered by `columns.bin` and emits sidecars for the
  remainder. Load path additively walks `columns/` after the mmap
  fast-path to pick them up.

- **Cypher `SET n.prop = X` on disk-backed graphs persists through save + reload.**
  Pre-fix, `DiskGraph::node_weight_mut` materialised a `NodeData`
  into `self.node_arena`; the Cypher executor mutated that arena
  copy; `clear_arenas` dropped it without writing back to the
  canonical `ColumnStore`. Affected SET + save + reload was a silent
  no-op for persistence across 0.8.10 and 0.8.11.

  Fix: mirror the proven `batch.rs::flush_chunk` full-`Arc`
  replacement pattern for exact-row mutations.
  `DiskGraph::node_weight_mut` now stages writes in a
  `node_mut_cache` (Map-backed `NodeData`); `clear_arenas` groups
  cached entries by type, deep-clones each affected `ColumnStore`
  **once**, applies every staged title / property write + DELETE
  tombstone to the clone, and replaces both
  `DiskGraph.column_stores[ty]` and (via
  `DirGraph::sync_column_stores_from_disk`)
  `DirGraph.column_stores[ty]` atomically. Avoids the
  `Arc::make_mut` → per-row clone + Arc divergence that doomed the
  earlier attempt. Title writes diff against the current stored
  value before calling `set_title` so that `TypedColumn::Str`'s
  in-place-update offset-corruption bug (pre-existing) doesn't
  trigger on unchanged titles.

- **`DETACH DELETE` on disk preserves surviving nodes' property
  values across save + reload.** Pre-fix, a disk-graph delete cycle
  corrupted `title` reads (garbage bytes) and returned `None` for
  some `age` values on reload. The surviving count and id set were
  always correct — only the column-store-backed property columns
  were affected. Root cause was in the sidecar load path, not in
  mutation routing: `load_column_sidecars` derived `row_count` from
  `type_indices[type].len()` (live rows only), while the sidecar
  blob retains tombstoned rows alongside live ones. The mismatch
  made `ColumnStore::load_packed` walk column blobs at the wrong
  offsets and decode offset bytes as string data.

  Fix: the sidecar `columns.zst` file now starts with an 8-byte
  `KGLCOLv1` magic tag followed by `ColumnStore::row_count` (u32
  LE) before the existing `write_packed` payload, and the loader
  uses that stored count. Old-format sidecars (no magic tag) fall
  through to the `type_indices.len()` derivation for backward
  compat — best effort for legacy graphs, correct for any graph
  saved by 0.8.12+. Locked by
  `test_detach_delete_property_persistence_disk` (was `_xfail`).

### Deferred to 0.8.13+

- **Planner pruning using `SegmentManifest`.** Summaries are
  populated and persisted since 0.8.11; the pattern matcher
  doesn't yet consult them. Initial exploration showed the win
  is small under the current concat-at-load architecture (typed-
  edge queries at wiki1000m already run at sub-ms when the
  histogram path applies, and concat'd reads are uniform).
  Kept as a future option for 200-segment workloads once those
  exist in practice.

## [0.8.11] — 2026-04-23

### Added (disk-graph-improvement-plan PR1, phases 1–8)

This release lands the segmented-CSR foundation on the disk backend
and then fills in the incremental-save and auxiliary-index work on
top of it. Net result: write amplification on incremental ingest
drops from 5–25× to ~2× (target from
`dev-documentation/disk-graph-improvement-plan.md`) while load/save
on Wikidata-scale graphs now beats 0.8.10 across every subset. Every
existing `.kgl` directory still loads byte-for-byte identically.

- **Segment manifest (`seg_manifest.json`).** On-disk JSON listing
  per-segment `node_id_range`, `edge_count`, `conn_types`,
  `node_type_counts`, and `indexed_prop_ranges` summaries. Future
  planner pruning consults this before scanning. Today populated
  as a single-segment descriptor on every save. New module
  `src/graph/storage/disk/segment_summary.rs`.

- **Segmented CSR directory layout (`seg_NNN/`).** The CSR
  binaries, ColumnStore, and per-(type,prop) property indexes now
  live under a per-segment subdirectory. Graph-level metadata
  (`disk_graph_meta.json`, `seg_manifest.json`, DirGraph
  metadata) stays at the graph root. Gated by
  `csr_layout_version` (`#[serde(default)] == 0`) so legacy flat-
  layout `.kgl` directories still load.

- **`segment_subdir(id)` + `enumerate_segment_dirs(root)`.** The
  directory name is now id-parameterised, and load walks a sorted
  `seg_NNN/` enumeration instead of a hardcoded path.

- **Multi-segment read path.** `SegmentCsr` bundles one segment's
  six core CSR arrays (`node_slots`, `out_offsets`, `out_edges`,
  `in_offsets`, `in_edges`, `edge_endpoints`);
  `concat_segment_csrs` stitches them by shifting segment-local
  `edge_idx` onto combined `edge_endpoints`, concatenating
  per-segment `node_slots` and `edge_endpoints`, and welding the
  offset arrays. Single-segment load stays on the direct-mmap
  path for zero overhead vs 0.8.10.

- **Multi-segment write path (`DiskGraph::seal_to_new_segment`).**
  Flushes the still-mutable tail
  (`[sealed_nodes_bound, node_count)` + overflow edges between
  those nodes) to a fresh `seg_NNN/` — with per-segment
  `conn_type_index_*`, `peer_count_*`, and `edge_prop_*`
  alongside the core CSR — appends a `SegmentSummary` to the
  manifest, clears consumed overflow, advances the watermark,
  and rewrites `disk_graph_meta.json`.

- **Full-range-offset mode for cross-segment seal.** The seal
  path accepts overflow whose source or target is below the
  watermark by emitting offsets that span every global node id
  rather than only the new segment's tail. `concat_segment_csrs`
  distinguishes the two modes per segment via
  `out_offsets.len() > node_slots.len() + 1` and unions
  contributions per-node. Lets general incremental ingest (not
  just new-nodes-only batches) take the seal path.

- **Automatic incremental save.** `save_to_dir` now delegates
  to `seal_to_new_segment` whenever a graph has a prior segment
  manifest and a tail above the watermark — the typical
  incremental-ingest shape. Second save on a 10-chunk build
  produces 10 segments instead of rewriting the entire tree
  each time.

- **Per-segment auxiliary indexes survive seal+reload.**
  Multi-segment reload now merges `conn_type_index_*`,
  `peer_count_*`, and `edge_prop_*` across all segments so
  typed-edge matches, `edge_weight()`, and `peer_count`-backed
  aggregates return correct results on sealed segments.

### Performance

- **Save/load regression on Wikidata-scale graphs undone.**
  0.8.11's initial segmented-CSR work regressed `save`/`load`
  6–22× on wiki100m–wiki500m because the
  `dir.join("columns.bin").exists()` guard in `DirGraph` and
  `io::file` didn't know about the phase-4 `seg_000/`
  relocation. Checking both locations restores the 0.8.10
  baseline — and beats it once the rest of the phase work lands
  (wiki500m build −23 %, save −10 %, load −12 % vs 0.8.10;
  wiki100m build −22 %, save −16 %, load −15 %).

- **`MATCH ()-[:T]->(c) RETURN c, count(*)` aggregations
  sub-millisecond at every scale.** The fused MATCH+RETURN
  aggregate path was running
  `PatternExecutor::execute(MATCH (c))` unconditionally to
  enumerate the group target before dispatching to the
  histogram top-K fast path — for an untyped group target,
  a 14.7 M-node full-graph scan on wiki1000m that the fast
  path never reads. Enumeration is now deferred to the one
  node-centric fallback branch that actually needs it.
  `P31 class counts` on wiki1000m: 3702 ms → 0.3 ms (12 300×).
  Same fix applied to `FusedMatchWithAggregate`:
  `WITH P27 count` on wiki1000m 5387 ms → 13 ms (408×), on
  wiki500m 426 ms → 6 ms (71×).

- **Edge-centric fast path for
  `MATCH (src:T1)-[:T]->(tgt) WITH tgt, count(src)`.** Phase
  3 routes this shape through the pre-built
  `peer_count_histogram` when the source-type filter is a
  no-op, or walks `conn_type_index` source lists otherwise —
  both O(|T-sources|) instead of O(|all nodes| × in-degree).
  On wiki500m the query drops from 1210 ms to 445 ms (the
  result is a stepping stone; the full win lands via the
  histogram-routing fix above).

- **Multi-key ORDER BY LIMIT on aggregated counts.** Phase 4
  extends the fused MATCH+RETURN aggregate path so
  `ORDER BY k DESC, c.title` no longer disables fusion.
  Fusion sets a `candidate_emit` descriptor; the executor
  picks the primary-key threshold via a heap and emits the
  qualifying superset (any tie-breaking on secondary keys
  happens in the unchanged downstream `OrderBy + Limit`).
  `P31 class counts` warm-cache on wiki500m dropped from
  1477 ms to 450 ms as a secondary effect, before the
  group-target-scan fix made it sub-millisecond.

### Fixed

- **CSR mmap files now trim in-place on `save_to_dir`.**
  `MmapOrVec::mapped(path, cap)` has a 64-element minimum, so
  small graphs left trailing zeros on disk. The single-segment
  load path masked this by using `meta.*_len`, but the new
  multi-segment load path relies on file-size inference. All six
  core CSR arrays now pass through `save_to_file` on the
  same-dir save path, triggering the `file.set_len(byte_len)`
  truncation. Same bug pattern as the 0.8.10 conn_type_index trim.

## [0.8.10] — 2026-04-20

### Performance

- **GROUP BY aggregation defers property materialization.** Queries of the
  shape `RETURN x.prop, count(*)` now hash by NodeIndex during the per-row
  pass and resolve the property once per resulting group, rather than once
  per input row. For high-fanout aggregations on disk graphs (e.g., walking
  Wikidata's 439K country=Norway entities and grouping by their `instance_of`
  type) this drops O(rows) random-I/O column reads to O(distinct groups).
  Cypher semantics are preserved by re-bucketing on resolved values after —
  two distinct nodes that share a property value still collapse into one
  group. Implementation in `src/graph/languages/cypher/executor/return_clause.rs`.

### Fixed

- **OPTIONAL MATCH + RETURN with PropertyAccess group keys no longer silently
  returns NULL groups.** The fused OPTIONAL MATCH + aggregation path
  evaluated group-key expressions against the source row (pre-OPTIONAL),
  so a query like `OPTIONAL MATCH (p)-[:OWNS]->(pet) RETURN pet.name,
  count(*)` would resolve `pet.name` to NULL for all rows, collapsing every
  result into one wrong group. The fusion check now rejects PropertyAccess
  on variables only bound by the OPTIONAL MATCH itself, falling through to
  the correct (non-fused) aggregation path. `is_fusable_return_clause` in
  `src/graph/languages/cypher/planner/fusion.rs` now takes the OPTIONAL
  MATCH variable set and rejects matching property accesses.

- **Multi-MATCH re-bind no longer full-scans the graph.** When a second
  MATCH clause re-bound a variable from a prior clause
  (`MATCH (f {id: X}) MATCH (f)-[:R]->(c)`), the pattern matcher's
  inverted-index fast path ignored the existing binding and returned
  every source node for the edge type — 20s+ timeouts on Wikidata-scale
  graphs. The fast path now skips when the first node is already bound,
  falling through to `find_matching_nodes` which resolves the variable
  to a single node. `{Gjøa, Norway}` goes from >20s timeout to ~36ms
  on the 124M-node Wikidata graph.

### Changed

- **Graph algorithm procedures (`CALL pagerank/degree/betweenness/
  closeness/louvain/label_propagation/connected_components`) now error
  on timeout instead of silently returning partial results.** Algorithm
  signatures changed to `Result<_, String>`; the `break`-on-deadline
  branches now return `Err`, and the new `algorithm_timeout_err()`
  message points users at `timeout_ms=N` / `timeout_ms=0`. Fixes silent
  half-converged results that looked successful.
- **`CALL` on graphs over 2M nodes now refuses unscoped procedure
  runs up front.** Prior to this, `CALL degree()` on Wikidata (124M
  nodes) ignored its `_deadline` parameter entirely and ran for
  minutes — long enough to exhaust MCP transport timeouts and appear
  to wedge the server. The new guard errors in <1ms with "would scan
  the whole graph. Subgraph scoping is not yet supported — try a
  smaller graph, or pass timeout_ms=0 to override this guard."
- **`degree_centrality` and `weakly_connected_components` now honor
  the 20s Cypher deadline.** Both previously ignored deadlines (the
  former via an unused `_deadline` parameter, the latter by not
  accepting one). Periodic checks every ~1M edges.

## [0.8.9] — 2026-04-20

### Changed

- **Streaming label journal replaces in-memory `label_cache` during
  `load_ntriples`.** The previous `HashMap<u32, String>` grew to ~10GB
  on Wikidata's 124M entities, collapsing streaming throughput from
  1.8M triples/s to 450K/s via swap pressure. Labels now spill to a
  sequential on-disk journal (`{spill_dir}/labels.bin`) — zero heap
  growth during Phase 1. The post-Phase-1 rename pass reads the
  journal once, filtering to the ~tens-of-thousands of Q-numbers
  that actually appear as type names (~3MB final footprint). New
  module: `src/graph/io/ntriples/label_spill.rs`.

### Fixed

- **Typed `MATCH (n:Type {title: 'X'})` now takes the cross-type
  global-index fast path.** Previously only untyped patterns consulted
  the global index; typed patterns fell through to a full-type scan —
  10–14s (and frequent timeouts) on 13M-row types like Wikidata
  `human`. The matcher now consults the global index and post-filters
  by `node_type_of(idx)`, dropping `MATCH (n:human {title: 'Barack
  Obama'})` from 14s to ~25ms. Alias-aware (title↔label↔name).
- **Per-type `{nid: ...}` / `{qid: ...}` anchors hit the id index.**
  Both typed and untyped paths previously only checked the literal
  `"id"` key, so alias queries fell through to full scans. Now
  `id`/`nid`/`qid` all anchor via the same per-type id_index.
- **String-form id anchors (`{nid: 'Q76'}`) hit the id index.**
  `TypeIdIndex::get` now coerces `"Q76"` → `UniqueId(76)` by stripping
  the leading alpha prefix. Works for any `[A-Za-z]+[0-9]+` id
  scheme (Wikidata Q-codes, P-codes, E-codes, ...). Previously the
  lookup fell through to a full-type scan, so
  `MATCH (a:human {nid: 'Q76'})-[r]-(b:human {nid: 'Q13133'})`
  dropped from ~14s to ~300ms on Wikidata. Also fixes the
  correctness bug where `MATCH (a {id: 'Q76'})` silently returned
  `0` rows instead of the matching node.

## [0.8.8] — 2026-04-19

### Fixed

- **EXISTS inline-property filters on target nodes were silently
  dropped.** `WHERE EXISTS { (a)-[:REL]->({id: 20}) }` used the fast
  path's `get_property("id")` which missed the special id_column,
  producing silent zero-row results even when the pattern genuinely
  matched. Ported the same alias resolution that
  `node_matches_properties` uses — `title`/`name`/`label`/`id`/`type`
  all route to the right column via `resolve_alias`. The fast path
  now behaves identically to the slow path for these literal-property
  checks. Regression tests added to `test_where_exists.py`.

### Added

- **Cross-type global property index.** New `create_global_index(property)`
  builds a single mmap'd sorted-string index covering every live node,
  not just one type. On a disk graph, `save()` now auto-builds a global
  title index so `MATCH (n {title: 'X'})` — without a type label — is
  O(log N) out of the box. Solves the "title-to-ID without guessing
  the type" problem that agents hit repeatedly on Wikidata-scale graphs.
  Files: `global_index_{property}_{meta,keys,offsets,ids}.bin`.
- **`g.search(text, property='title', limit=10)` helper.** Returns
  the top-k nodes whose `property` matches `text` (exact match first,
  then prefix fallback) as `[{id, type, title, id_value}]`. Backed by
  the global index. Also exposed as a new MCP tool so agents can skip
  the "guess the type" ceremony entirely: `search('Equinor')` returns
  the right Q-number without `MATCH` or a type label.
- **Alias-aware cross-type lookups.** When the untyped matcher sees
  `{title: 'X'}` and the literal `title` index doesn't exist, it also
  tries the hardcoded title family (`title/label/name`) AND any
  per-type aliases declared via `node_title_field=` at `add_nodes`
  time. Same for `id/nid/qid`. An agent who built the index as
  `create_global_index('label')` still hits the fast path when
  querying `{title: 'X'}`, and vice versa. Derivation is automatic
  from the graph's existing schema — no new config API.

### Changed

- **`save_disk` auto-builds the global title index.** Every call to
  `save()` on a disk graph now produces `global_index_title_*.bin`
  files. Adds a one-pass sweep over `node_slots` at save time —
  negligible on small graphs, ~single-digit minutes on Wikidata-scale
  (124M nodes). Opt-out: delete the files after save.

## [0.8.7] — 2026-04-19

### Added

- **`WHERE n.prop STARTS WITH 'prefix'` now pushes down into the MATCH
  pattern** and routes through the persistent disk prefix index when
  available. New `PropertyMatcher::StartsWith(String)` variant, new
  `apply_prefix_to_patterns` helper in
  `src/graph/languages/cypher/planner/index_selection.rs`, new path in
  `matcher.rs::try_index_lookup` that calls
  `GraphRead::lookup_by_property_prefix`. String indexes are annotated
  `indexed="eq,prefix"` in `describe()` output (previously just `eq`);
  numeric indexes remain `indexed="eq"` only.
- **Deadline polling inside unanchored matcher scans.** Three hot
  loops in `matcher.rs` that used `.filter().collect()` over
  13M+-node type lists now poll the deadline every 4096 rows (via a
  new `check_scan_deadline()` helper with a structured hint message).
  Worst-case overshoot past the deadline drops from 20-60+ s to under
  a few ms. Other scan loops (variable-length paths, CSR edge
  counting, column stats) already polled; this closes the final gaps.
- **MCP `cypher_query` tool accepts `timeout_ms`.** `examples/mcp_server.py`'s
  tool signature now exposes the override so agents can deliberately
  extend or disable the deadline (`timeout_ms=0`) per call after an
  EXPLAIN confirms the plan is anchored. Previously the MCP agent was
  stuck with the backend-aware default.
- **`ResultView.diagnostics` — lightweight execution diagnostics.** Every
  `cypher()` call now attaches an always-on diagnostics dict to the
  returned `ResultView` with `elapsed_ms`, `timed_out`, and the
  `timeout_ms` that was in effect. Gives agents immediate feedback on
  query cost and timeout state without requiring `PROFILE`. The field
  is ``None`` for mutation paths, EXPLAIN, and transaction queries.
- **`describe()` indexed-property annotations.** Properties covered by
  an index (in-memory `property_indices` *or* the new persistent disk
  `PropertyIndex`) are now emitted with an `indexed="eq"` attribute in
  the `<properties>` detail block. A new `<indexing>` hint inside
  `<extensions>` explains the annotation and reminds agents to prefer
  anchored queries over unanchored scans on disk-backed graphs. New
  helper `DirGraph::has_any_index(node_type, property)` consolidates
  the "in-memory or persistent" check.
- **Persistent disk-backed property index.** `create_index('T', 'label')`
  on a `storage='disk'` graph now writes four mmap'd files
  (`property_index_{type}_{property}_{meta,keys,offsets,ids}.bin`)
  next to the CSR instead of rebuilding a `HashMap<Value, Vec<NodeIndex>>`
  on every `load()`. The previous in-memory path consumed ~1-3 GB of
  heap on 13M-row types and made `create_index` effectively unusable
  on Wikidata-scale disk graphs. The new persistent index is lazy-loaded
  on first query after reopen, keys are sorted lexicographically (so
  both equality and prefix can share the same structure), and the
  Cypher planner consults it via a new `GraphRead::lookup_by_property_eq`
  trait method. `MATCH (n {label: 'X'})` now hits the index on disk in
  O(log N + k). Supports string columns and title aliases
  (`node_title_field` at `add_nodes` time — `label`, `name`, etc.).
  Numeric equality and `STARTS WITH` pushdown are follow-ups. In-memory
  graphs are unchanged (keep the existing `property_indices` HashMap).
  The `create_index` return dict grows a `persistent: bool` field
  indicating whether the disk path was taken.
- **Cypher schema validation** at plan time — catches typos in
  pattern-literal property names (`MATCH (n:Person {agee: 30})`) before
  the executor commits to a scan. Returns a `Did you mean 'age'?` hint.
  Runs in O(clauses) against `node_type_metadata`; skipped when a graph
  has no declared schema. Pattern-literal properties are the only v1
  target — unknown node types, connection types, and WHERE/RETURN
  `n.prop` accesses deliberately pass through (existence-check queries
  and virtual columns would otherwise false-positive). Phase 3 will
  surface those as non-fatal diagnostics.

### Changed

- **`cypher()` default timeout is now backend-aware.** Disk-backed
  graphs default to 10 s, Mapped to 60 s, Memory to no deadline. Users
  can override per-call via `timeout_ms=N` or globally via
  `set_default_timeout(ms)`. The documented escape hatch
  `timeout_ms=0` disables the deadline entirely. Previously,
  disk-backed queries without an explicit `timeout_ms` ran until the
  harness killed them; the new default returns a structured timeout
  error after 10 s with hints pointing at anchoring / index usage.
  (Also applies to transaction-level `cypher()`.)
- **Cypher timeout error message now carries remediation hints.**
  Replaces the bare string `Query timed out` with guidance on
  anchoring queries, raising `timeout_ms`, or using the `timeout_ms=0`
  escape hatch.
- **`set_default_timeout(None)` behaviour updated.** Passing `None`
  now falls through to the backend-aware default rather than meaning
  "no timeout". Pass `0` for the old behaviour explicitly.

## [0.8.6] — 2026-04-19

### Performance

- **`describe(connections=['T'])` fast path on disk graphs.** Rewrote
  `write_connections_detail` to use the persisted `conn_type_index_*`
  inverted index instead of three full `edge_references()` sweeps. The
  previous path materialised every visited edge into a per-query
  `edge_arena` that was never cleared mid-call, growing VSZ linearly
  with scanned edges — on Wikidata (863 M edges) a single
  `describe(connections=['P31'])` call was SIGKILLed by the kernel
  after exhausting VM. The new path:
  - Reads pair counts from `type_connectivity_cache` when populated
    (zero edge I/O).
  - Skips the property-stats scan entirely when the connection type's
    metadata declares no edge properties.
  - Walks only matching edges via the inverted index, capped at two
    samples via an early-exit callback.
  - Measured on Wikidata (`wikidata_disk_graph_p12rebuild`, 122 M
    nodes, 863 M edges, cold page cache):
    - `describe(connections=['P170'])` (1.3 M edges): 108 s → **0.24 s**
      (~450× faster; previous in-flight code held VSZ at +27 GB
      after 90 s without completing).
    - `describe(connections=['P31'])` (122 M edges): **0.25 s** (was
      SIGKILLed by OOM killer before this change).
    - `describe(connections=True)` unchanged at ~0.15 s.

### Changed

- **`describe(connections=['T'])` pair-breakdown now capped at 50
  entries by default** (sorted by count desc), overridable via a new
  `max_pairs` keyword argument. Wide fan-out connection types like
  Wikidata's `P31` have tens of thousands of distinct
  `(src_type, tgt_type)` pairs — P31 alone has 191 k — which produced
  ~13 MB of XML that overshot typical MCP response budgets. The cap
  emits `<endpoints total="N" shown="…">` plus a trailing
  `<more pairs="…" edges="…"/>` marker so agents see both the
  dominant relationships and the exact size of the tail. P31 output
  drops 13 MB → ~4 KB by default; pass `max_pairs=500` (or similar)
  to drill into the full distribution on demand.

### Added

- `GraphBackend::for_each_edge_of_conn_type` — monomorphic closure
  iterator yielding `(src, tgt, edge_idx, properties)` per match. On
  disk uses the inverted index and never materialises `EdgeData`; on
  Memory/Mapped filters petgraph's resident `edge_references`. The
  callback returns `bool` so callers can stop after a bounded prefix.
- `DiskGraph::edge_properties_at(edge_idx)` — borrow an edge's
  property slice without going through the `materialize_edge` arena.
- `describe(..., max_pairs=<int>)` keyword argument — controls the
  pair-breakdown cap described above. `None` (default) resolves to 50.

## [0.8.5] — 2026-04-19

Internal: test coverage, SAFETY docs, storage module reorganization.

## [0.8.4] — 2026-04-19

### Performance

- **Correlated-equality pushdown in the Cypher planner.** `WHERE cur.prop =
  prior.other_prop` — where `prior` is a node bound by an earlier MATCH —
  now pushes onto the current MATCH's pattern as a new `EqualsNodeProp`
  matcher that the executor resolves per-row via the bound node's
  property. When the probe-side property is indexed, the pattern executor
  then picks an indexed lookup instead of scanning all nodes of that
  type. Also pushes `cur.prop = scalar_var` (where `scalar_var` is
  projected by a prior WITH/UNWIND) as `EqualsVar`. WHERE stays as a
  safety-net filter. Fallback: unchanged behavior when no index exists.
- **`add_connections(query=...)` now runs the planner.** Previously, the
  query path in `add_connections` went straight from parse → execute,
  skipping the entire planner — so no pushdowns (equality, IN,
  comparison), no spatial-join fusion, no LIMIT/DISTINCT pushdown. It
  now calls `cypher::optimize` like `g.cypher()` does. Combined with
  the correlated-equality pushdown, the Sodir prospect graph's derived
  connections (Phase 7) now build **~8.5× faster** — 29.6 s → 3.5 s:
  - 7a HC_IN_FORMATION (3 UNION ranks): 11.0 s → 1.2 s (~9×)
  - 7b/7c StructuralElement ENCLOSES: 0.6 s → 0.05 s (fuse_spatial_join
    now fires here)
  - 7d PLAY_HAS_FORMATION (primary + fallback): 17.3 s → 1.5 s (~11×)

## [0.8.3] — 2026-04-19

### Performance

- **Spatial-join operator for `MATCH (s:A), (w:B) WHERE contains(s, w)`.**
  A new planner pass (`fuse_spatial_join`) rewrites this two-pattern
  containment shape into `Clause::SpatialJoin`, bypassing the cartesian
  product. The executor builds an R-tree over the container side (via
  the new `rstar` dependency), iterates the probe side once, and emits
  only matching (container, probe) pairs — `O((N+M) log N + K)` rather
  than `O(N·M)`. Speedups on `tests/bench_spatial.py` (release build):
  - `contains 500K pairs` (500 polygons × 1 K points): 86.96 ms → 0.52 ms (**~167×**)
  - `contains 2.6M prospect_shape` (263 complex polygons × 10 K points):
    480.51 ms → 3.32 ms (**~145×**)
  - `contains 100K pairs`: 17.65 ms → 0.55 ms (~32×)
  - Complex polygons (50 vertices): 18.29 ms → 0.24 ms (~76×)

  Fires when both types have `SpatialConfig` (container needs `geometry`,
  probe needs `location`), the two patterns are disjoint typed nodes
  with no edges, and the WHERE is `contains(var, var)` optionally ANDed
  with a residual predicate. Other shapes (`NOT contains`, constant-point
  `contains(a, point(…))`, intra-pattern edges, three-plus patterns,
  disjunctions) fall back to the existing per-row fast path unchanged.

## [0.8.2] — 2026-04-19

### Changed

- **Blueprint loader rewritten in Rust.** `kglite.from_blueprint()` now runs
  entirely in a new `src/graph/blueprint/` module (schema + CSV reader +
  filter DSL + geometry + timeseries + build orchestrator). `pandas` is no
  longer touched during ingestion — CSVs are parsed with the `csv` crate
  straight into the internal columnar `DataFrame`, then handed to
  `mutation::maintain::add_nodes` / `add_connections`.
  - Every parallelisable phase is pipelined: CSV pre-parse, per-spec
    prep (filter + geometry + typed-column build), FK edge DataFrames,
    and junction edge DataFrames all run across threads via `rayon`.
    Only the graph mutation calls (`add_nodes` / `add_connections`)
    stay serial, because the graph is `&mut`. GeoJSON→WKT centroid
    extraction is also parallelised per row.
  - The Python shim (`kglite/blueprint/__init__.py`, ~60 lines) now
    only handles optional save + schema lock on top of the native
    build. The old 831-line `kglite/blueprint/loader.py` is deleted.
  - Sodir blueprint (564 K nodes, 759 K edges): **9.87 s → 1.6 s** (~6×).
  - Node / edge counts match the previous Python loader exactly (parity
    verified per-type across all 90 node types and all 93 edge types).
  - New runtime deps: `csv`, `geojson`, `indexmap` (the last so node /
    sub-node iteration preserves blueprint JSON order, which in turn
    keeps edge counts byte-identical to the old loader).
  - Set `KGLITE_BLUEPRINT_PROFILE=1` for a per-phase / per-sub-phase
    ms breakdown on stderr.

## [0.8.1] — 2026-04-19

### Changed

- **`code_tree` rewritten in Rust.** The polyglot codebase parser previously
  implemented in Python (`kglite/code_tree/*.py`, ~7,500 LOC) is now a
  first-class Rust module (`src/code_tree/`) exposed via PyO3. All eight
  language parsers (Python, Rust, TypeScript/JavaScript, Go, Java, C#, C,
  C++) plus the builder and manifest readers run natively. Tree-sitter
  grammars are bundled into the native extension — no optional dependency
  needed. **`pip install kglite[code-tree]` is no longer required**; the
  `[code-tree]` extras entry has been removed.
- **abi3 wheel — one wheel per platform, Python 3.10+.** PyO3's
  `abi3-py310` stable-ABI target is now enabled, collapsing the CI wheel
  matrix from 20 wheels (5 Python versions × 4 platforms) to 4. Users on
  any Python ≥ 3.10 install the same wheel.
- **Parallel parsing via rayon + thread-local parsers.** File-level
  parsing runs across CPU cores with one tree-sitter `Parser` per thread
  (via `thread_local!`) — no `Mutex` contention.
- **Parallel CALLS-edge tier resolution.** The 5-tier name-matching pass
  (84 K functions → 200 K edges) now runs in parallel via rayon; each
  function's edges are independent.
- **Aho-Corasick for USES_TYPE.** The multi-pattern type-name scan
  replaces a giant regex alternation with an Aho-Corasick automaton,
  yielding ~2.5× faster `USES_TYPE` edge building on Java-scale corpora.
- **End-to-end performance on real repos:**
  - duckdb C++ (2,805 files): **29 s (Python) → 0.63 s (Rust)** — ~46×
  - neo4j Java (7,966 files, 84 K functions): **crashed in Python →
    1.69 s in Rust**
  - KGLite mixed Py+Rust (248 files): 0.17 s
- **New `code_tree` module shape.** `kglite/code_tree/__init__.py` is a
  4-line shim importing from the native `kglite._kglite_code_tree`
  submodule. The previous Python modules under `kglite/code_tree/` have
  been removed.

### Fixed

- **`build()` no longer crashes on pure-Java repos** (e.g. `neo4j/neo4j`)
  with `Source type 'Struct' does not exist in graph`. Edge routing now
  picks source/target node types per-row from the graph schema rather
  than defaulting to hardcoded names.

## [0.8.0] — 2026-04-18

Internal-only storage-architecture refactor plus a handful of disk-mode
bug fixes and large performance wins. **No Python API signature changes.**
`kglite/__init__.pyi` signatures are byte-identical to v0.7.17 (`git diff
v0.7.17 HEAD -- kglite/__init__.pyi` contains only docstring additions).
Users upgrading from 0.7.x will see no behavioural differences other
than the fixes and performance gains listed below.

### Fixed

- **Concurrent `load_ntriples` calls no longer wipe each other's spill
  directories.** The previous cleanup logic deleted *all other*
  `kglite_build_*` directories in `/tmp` at every ingest start. Two
  `load_ntriples` calls running at the same time (e.g. a long Wikidata
  build and a small test suite) would kill each other's property-log
  files, causing the in-flight build to crash at Phase 1b with `No
  such file or directory`. The cleanup now only removes spill dirs
  whose contents haven't been modified in the last hour, so active
  builds are always safe.
- **`save()` on Wikidata-scale disk graphs no longer appears to hang.**
  On large disk graphs (e.g. 124 M nodes / 88 K types from N-Triples
  ingest), `save()` was iterating `self.column_stores` and writing
  each type's columnar data as a separate `columns/<type>/columns.zst`
  zstd file — a multi-hour serial loop, redundant because the v3
  single-file `columns.bin` (written during Phase 1b of the N-Triples
  builder) already contains everything the loader needs. `save_disk`
  now skips the per-type loop when `columns.bin` exists on disk, and
  reduces to the metadata flush it was always meant to be.
  **Measured on Wikidata (124 230 686 nodes, 862 810 243 edges,
  88 931 types): `save()` went from ≥ 60 min → 5.52 s.** In-memory
  graphs persisted to disk still write the per-type files as before
  (that path never produces `columns.bin`).
- **Disk-mode `add_nodes(conflict_handling="update")` now applies
  property updates.** Previously on disk graphs, re-inserting an
  existing node via `add_nodes(..., conflict_handling="update")`
  silently dropped the new values — `node_weight_mut` materialised
  `NodeData` into a per-query arena that `clear_arenas` discarded
  before the next read, so the mutation never reached
  `DiskGraph::column_stores` where reads happen. The batch-update path
  now mutates the per-type column store directly via `Arc::make_mut`
  and re-syncs with `sync_disk_column_stores` at the end of the chunk.
  Memory and mapped graphs are unaffected (they already worked).
- **Disk-mode `add_nodes(conflict_handling="replace")` now clears
  omitted properties.** Same root cause as above; Replace now nulls
  out every previously-set property on the row before writing the new
  set, matching the `PropertyStorage::replace_all` semantics of the
  heap backends.
- **Disk-mode `MERGE` edges are visible to subsequent `MATCH`
  queries.** `DiskGraph` used to default `defer_csr = true` so every
  `add_edge` on a fresh graph queued into `pending_edges`, which
  `edges_directed` never reads. One-off Cypher mutations now route
  directly to the overflow buffer (visible immediately); bulk loaders
  (`add_connections`, ntriples) still use the pending+rebuild path via
  `build_csr_from_pending`.

### Performance

- **Cypher query primitives faster across the board vs v0.7.17** (N=20
  trials, macOS dev box). Memory and mapped modes both win:
  - `pattern_match` at 10 k nodes: −60 %
  - `two_hop_10x`: −24 %
  - `describe()`: −21 % (memory), −24 % (mapped)
  - `pagerank`: −17 %
  - `find_20x`: −13 % (mapped)
  - Construction sweep (1 k / 10 k / 50 k nodes): −11 % to −22 %
  - No memory-mode query regressed above the +5 % gate; only four cells
    flagged under 5 % (`find_20x_memory` +4.7 %, `simple_filter` /
    `multi_predicate` minor noise).
- **N-Triples disk-graph build is 2.5 % faster on Wikidata.** Added
  `#[inline(always)]` on the hot `GraphBackend` → `GraphRead`/`GraphWrite`
  trampolines (`node_type_of`, `edge_endpoint_keys`, `edge_endpoints`,
  `node_weight`) and a new closure-based
  `GraphBackend::for_each_edge_endpoint_key` that bypasses the
  boxed-iterator virtual-dispatch on hot edge iteration. Phase 1b
  (columnar write) −86 s, Phase 2 (edge creation) −32 s, Phase 3
  (CSR build) −55 s on a 7.65 B-triple / 862.8 M-edge build. Total
  `load_ntriples`: 4747 s → 4627 s.
- **`rebuild_caches()` is 28 % faster on large disk graphs.** Two
  fixes: (a) `compute_type_connectivity` is now Rayon-parallel on the
  disk backend — shards the edge range across all cores and merges
  per-shard HashMaps serially, matching `build_peer_count_histogram`'s
  pattern; (b) removed a `madvise(DONT_NEED)` call at the end of
  `build_peer_count_histogram` that was evicting the 13.8 GB
  `edge_endpoints` from page cache right before
  `compute_type_connectivity` had to re-read it. Also reordered
  `rebuild_caches` to run `compute_type_connectivity` first so its
  sequential sweep warms the cache for the histogram builder. Measured
  on Wikidata (862.8 M edges): 235 s → 169 s. Memory and mapped modes
  unaffected (serial path retained).

### Changed

- **Deterministic `.kgl` v3 saves.** `save()` now produces byte-identical
  output for identical graphs regardless of per-process HashMap
  randomisation. `write_graph_v3` iterates `column_stores` in sorted
  order and canonicalises the metadata JSON (object keys sorted). Old
  `.kgl` files load unchanged — the format on the wire is a strict
  subset of the previous format's possible outputs. Enables byte-level
  golden-hash format-drift tests.
- **`ConnectionTypeInfo` serialises with sorted keys.** `source_types`
  and `target_types` (HashSet<String>) and `property_types`
  (HashMap<String, String>) now emit in lexicographic order, hardening
  the v3 golden-hash invariant for fixtures richer than single-element
  sets. Existing `.kgl` files load unchanged.

### Changed (internal, not user-visible)

- **Internal reorganization — `src/graph/` split into domain
  subdirectories.** Code previously flat in `src/graph/` now lives
  under `algorithms/`, `languages/cypher/`, `features/`,
  `introspection/`, `io/`, `mutation/`, `pyapi/`, `core/` (shared
  primitives, was `query/`), and `storage/`. `storage/` further splits
  into `memory/`, `mapped/`, and `disk/` per-backend folders. Pure
  file moves via `git mv` (rename similarity 97–100 %; git blame
  preserved). Filenames cleaned of redundant prefixes / suffixes
  (`pymethods_*` → `*`, `filtering_methods` → `filtering`, etc.). See
  `ARCHITECTURE.md` for the final layout.
- **Every `.rs` under `src/graph/` is now at or under the 2,500-line
  hard cap.** The Phase 9 split carved nine god files (12,144-line
  `executor.rs` down through the 2,610-line `pattern_matching.rs`)
  into themed submodules. `GOD_FILE_EXCEPTIONS` is empty;
  `test_god_file_gate` passes unconditionally.
- **`MappedGraph` promoted to a distinct struct** (was a type alias
  for `MemoryGraph` pre-Phase 5). Per-backend `impl GraphRead` /
  `impl GraphWrite` land in `src/graph/storage/impls.rs`, setting up
  future backend-specific optimizations without breaking callers.
- **`RecordingGraph<G>` ships as a Rust-only validation wrapper.**
  Generic over any `G: GraphRead`, logs every read-path method call.
  Used internally to prove the architecture is actually open/closed —
  adding a new backend is a 3-src-file change. Not exposed to Python.
  See `docs/adding-a-storage-backend.md` for the worked example.
- **Testing envelope hardened.** New parity tests cover zero-node /
  single-edge / 1 000-hop / Unicode / type-promotion / null-NaN /
  100 000-row cypher results across memory / mapped / disk
  (`tests/test_edge_cases_parity.py`). Golden-fixture regression suite
  (`tests/test_golden.py` + `tests/golden/`) pins byte-exact output
  for a deterministic 1 000-node / 3 000-edge graph across every
  storage mode. New `@pytest.mark.stress` tier for the 30 GB mapped
  bench and 10 k-hop traversal.
- **Unsafe-block hygiene.** All 40 `unsafe { ... }` blocks in `src/`
  carry `// SAFETY:` justifications. A module-level invariants block
  at the top of `src/graph/storage/mapped/mmap_vec.rs` documents the
  shared mmap safety contract.
- **Python API docstring clarification.** The `find()` docstring now
  warns that it searches only code-entity node types (`Function`,
  `Struct`, `Class`, `Enum`, `Trait`, `Protocol`, `Interface`,
  `Module`, `Constant`). The signature is unchanged.
- **Deprecated `TempDir::into_path()` calls** migrated to
  `TempDir::keep()` per tempfile 3.14+ API.
- **`pub type Graph = GraphBackend` alias dropped.** Every call site
  uses `GraphBackend` directly. Removes a hygiene wart flagged in the
  Phase 9 report-out.
- **RecordingGraph audit methods (`log`, `log_len`, `drain_log`) are
  now `#[cfg(test)]` rather than `#[allow(dead_code)]`.** Release
  builds no longer compile these helpers at all.

## [0.7.17] — 2026-04-17

### Added
- **Python 3.14 wheels**. CI test matrix and `build_wheels.yml` now cover
  3.14 across Linux/macOS (Intel + arm64)/Windows, alongside 3.10–3.13.
  Full test suite passes on 3.14 (1758 tests, same as 3.12 minus the
  optional `code-tree` tests that require tree-sitter wheels). pyo3 0.28
  (shipped in 0.7.16) enables this via `ABI3_MAX_MINOR = 14`.

## [0.7.16] — 2026-04-17

CI-fix release on top of 0.7.15. No functional changes to the Cypher
engine; dependency bumps + clippy 1.95 compatibility only.

### Dependencies
- **pyo3 0.27 → 0.28**, **geo 0.29 → 0.33**, **wkt 0.11 → 0.14**,
  **bzip2 0.5 → 0.6**. API changes absorbed: `#[pyclass(skip_from_py_object)]`
  on `KnowledgeGraph` (pyo3 0.28 opt-in); `Geodesic` is now a static value
  (call as `Geodesic.distance(...)` / `length(&Geodesic)`) with the
  `LengthMeasurable` trait imported from `geo::line_measures`.
- Clippy 1.95 compat: `sort_by` → `sort_by_key(Reverse)`, collapsed `if`/
  `match` guard patterns, `file_len.checked_div(elem_size)`, removed
  redundant `.into_iter()` in `IntoIterator` args.

## [0.7.15] — 2026-04-17

### Added
- **`WHERE n:Label` predicate**. Cypher now supports label checks as boolean
  predicates (not just MATCH-level filters). Composes with `AND`/`OR`/`NOT`
  and chained `n:A:B` form (`n:A AND n:B`). Example:
  `MATCH (n) WHERE n:Person OR n:Org RETURN count(n)`.
- **`Value::as_str() -> Option<&str>`**. Borrowing companion to the existing
  `as_string()`. Prefer when ownership is not required — avoids the per-call
  `String` clone.

### Changed
- **Function names lowercased at parse time** instead of per-row during
  dispatch. Every Cypher scalar/aggregate dispatch used to call
  `.to_lowercase()` on the function name each time it evaluated a row
  (21+ sites); names are now normalized once in `parse_function_call` and
  compared directly. Pure CPU win on function-heavy queries.
- **`count(DISTINCT n)` uses typed identity sets** — `HashSet<usize>` keyed
  on node/edge indices (with a `HashSet<Value>` fallback for non-binding
  expressions) instead of per-row `format!("n:{}", idx.index())` string
  formatting. ~20–26% faster on DISTINCT-count queries.
- **`substring()` skips intermediate `Vec<char>`** — uses `chars().skip(start)
  .take(len).collect()` instead of materializing the full char vector.
  ~10–18% faster on substring-heavy queries.
- **Zero-allocation property iterators**. `PropertyStorage::keys()` and
  `::iter()` return explicit `PropertyKeyIter` / `PropertyIter` enums instead
  of `Box<dyn Iterator>`. Saves one heap allocation per `keys(n)` /
  `RETURN n {.*}` / property-scan call. ~10% faster on `keys(n)` over
  all nodes.

### Fixed
- **`HAVING` with aggregate expressions**. `HAVING count(m) > 1` was silently
  returning zero rows when the RETURN item was aliased (`count(m) AS c`).
  Root cause: the aggregate function call fell through to per-row scalar
  dispatch, which errored with "Aggregate function cannot be used outside
  of RETURN/WITH", and the error was swallowed by `unwrap_or(false)`,
  dropping every row. Now `HAVING count(m)` and `HAVING c` both resolve
  to the pre-computed aggregate value regardless of aliasing. Unaliased,
  `DISTINCT`, and no-group-by forms all covered.
- **`rand()` / `random()` correctness under tight loops**. The previous
  SystemTime-per-call seeding could return identical values for adjacent
  rows when the system clock resolved two calls to the same nanosecond,
  and constant folding could collapse `rand()` to a single value for the
  whole query. Replaced with a thread-local xorshift64 PRNG, seeded once
  per thread with a splitmix64-avalanched counter (so parallel Rayon
  workers don't collide), and marked as row-dependent so it bypasses
  constant folding. Also uses the top 53 bits of state for full f64
  mantissa precision.

## [0.7.14] — 2026-04-17

### Added
- **Per-(conn_type, peer) edge-count histogram** as a persistent disk cache. Built once at CSR-build time (parallelised via Rayon, single sequential scan of `edge_endpoints.bin`), stored as three flat `peer_count_*.bin` files. Unanchored aggregate queries like `MATCH (a)-[:TYPE]->(b) RETURN b.title, count(a) ORDER BY cnt DESC LIMIT N` now return in ~ms instead of scanning the full 13 GB `edge_endpoints` array. Rebuildable on existing disk graphs via `g.rebuild_caches()` without a full graph rebuild.
- **`FusedCountAnchoredEdges` planner rule + executor**. `MATCH (var)-[r:TYPE?]->({id: V}) RETURN count(var)` (and the three symmetric variants) is now fused into O(log D) CSR offset arithmetic. The anchor is resolved to a `NodeIndex` at plan time via `graph.id_indices`. Combined with the tombstone short-circuit (below) this turns hub-node count queries (e.g. ~40 M incoming edges on Q5) from 100 s TIMEOUTs into sub-second lookups.
- **Tombstone-free short-circuit in `count_edges_filtered`**. When no nodes/edges have been removed and no peer-type filter is set, the function returns `end - start + overflow_count` directly after binary-searching for the connection-type range — skipping the per-edge tombstone check on hot hubs. Adds a `has_tombstones: bool` flag to `DiskGraph` and `DiskGraphMeta` (defaults to conservative `true` on legacy graphs so correctness is preserved; new builds flip it false).
- **Bounded `sources_for_conn_type`**. `DiskGraph::sources_for_conn_type_bounded(conn_type, max)` stops copying source node IDs after `max` entries, avoiding the ~400 MB eager heap allocation on cold-cache `LIMIT`-bounded pattern-matching queries. `pattern_matching.rs` now passes the `source_cap` through so e.g. `LIMIT 10` queries only read 1 000 sources from `conn_type_index_sources.bin` on first access.
- **`FusedCountTypedEdge` uses cached edge-type counts**. A one-liner that had been missed in v0.7.12: `MATCH (_)-[:TYPE]->(_) RETURN count(*)` now returns `edge_type_counts[TYPE]` in O(1) instead of scanning `edge_weights()` (64 s → sub-millisecond on Wikidata's 862 M edges).
- **`rebuild_caches` refreshes the peer-count histogram** on existing disk graphs, so users don't need to rebuild from scratch to get the v0.7.14 aggregate speedups.

### Fixed
- **DataFrame / blueprint disk builds now rebuild indexes at save time**. Previously, the first `add_connections` batch triggered a CSR build (via `ensure_disk_edges_built`) which wrote `conn_type_index` and `peer_count_histogram` reflecting only that first batch's edges. Subsequent batches added edges to overflow but never refreshed those indexes. Fix: `save_disk` now calls `compact()` once when overflow has accumulated, merging overflow back into CSR and rebuilding the indexes from all live edges. The per-batch `ensure_disk_edges_built` is now a no-op for overflow purposes (no O(E²) cost during multi-batch builds).
- **`lookup_peer_counts` returns `None` on type miss**. Previously returned `Some(empty_map)`, which blocked the caller from falling back to the sequential-scan path when the histogram was stale. Now returns `None` so callers see a clean cache miss.
- **Deadline checks in anchored-count paths**. `try_count_simple_pattern` / `count_edges_filtered` now accept an `Option<Instant>` deadline and check it every 1 M iterations. Closes the bypass that let `Q5_count_P31_incoming` run to 100 s past the 20 s default timeout.
- **Deadline check in `expand_var_length_fast` inner loop**. The outer queue loop was already checked every 512 pops, but the per-edge inner loop was unbounded — a single hub expansion could process 100 M+ edges without checking. Added an inner check every 1 M iterations.
- **Benchmark metric: Wikidata `unanchored_P31_count` now returns in 0.7 ms** (was 64 s with a wrong answer on cold deadline checks), `Q5_count_P31_incoming` 615 ms (was 100 s TIMEOUT), `Q5_incoming_all_count` 670 ms (was 20 s TIMEOUT), `cross_type_limited` 3 ms (was 2.5 s), `limit_10_P31` 10 ms cold-cache (was 2.6 s).

## [0.7.12] — 2026-04-16

### Added
- **Parallel Phase 3 CSR build**: The per-node `out_edges` sort-by-connection-type and the `conn_type_index` inverted-index build are now Rayon-parallelised. On a 124 M-node / 862 M-edge Wikidata build, this cuts combined CSR-build wall-clock on the parallelised portions from ~1000 s serial to ~100-200 s on 8+ P-cores. Build output is bit-identical to the serial version (index source lists are sorted post-reduce for determinism).
- **Deadline enforcement on long edge scans**: `count_edges_grouped_by_peer` (used by fused aggregate top-K and streaming HAVING paths) now accepts an optional deadline and checks it every 1 M edges. Pattern-matching's parallel expansion short-circuits when any thread detects a timeout. Together these stop unanchored aggregate queries from running unbounded past the default 20 s timeout.
- **Disk mode iterative updates**: Loaded disk graphs now support `add_connections()` — new edges go directly to overflow and are immediately visible to queries without CSR rebuild.
- **`compact()` method**: Merges overflow edges back into CSR arrays via full rebuild. Call after accumulating significant overflow (e.g., >10% of edges) to restore optimal query performance.
- **Connection-type inverted index for overflow**: `sources_for_conn_type()` now includes nodes with overflow edges, so Cypher queries on new edge types work immediately.
- **Partitioned CSR build parity**: Out-edges now sorted by connection type (enables binary search), and connection-type inverted index built for both CSR algorithms.

### Fixed
- **Streaming HAVING aggregate no longer OOMs**: `MATCH ...-[...]->(...) RETURN group_key, count(...) HAVING ... ORDER BY ...` (without LIMIT) used to materialise all edge rows before grouping — a 10 GB materialisation on Wikidata-scale graphs that triggered macOS OOM kill. The planner now fuses this shape into `FusedMatchReturnAggregate`; the executor's non-top-k path uses edge-centric `count_edges_grouped_by_peer` and applies HAVING post-aggregation on the small group-by map. On 16 GB hosts, queries that previously SIGKILL'd the Python process now return a clean "Query timed out" error.
- **Benchmark CSV preserved across crashes**: `bench/benchmark_wikidata_cypher.py` now streams each query's result to the CSV row-by-row with per-row flushes, instead of batching the write at the end. SIGKILL / OOM / Ctrl-C mid-run leaves every completed row on disk.
- **Mmap lifecycle during CSR build**: CSR build now writes to a temporary directory, then atomically swaps files into place. Fixes panics on large DataFrame builds where CSR output overwrote mmap'd files.
- **Overflow edges missing from `edges_directed`**: The `edges_directed_filtered_iter` iterator now correctly includes overflow edges (was passing `None` for overflow parameter).
- **Column store corruption on save→load→save cycle**: `write_packed()` now handles mmap-backed column stores (from loaded disk graphs) by materializing data from the MmapColumnStore. Also skips writing empty schema columns that duplicate id/title columns.
- **`defer_csr` not reset after CSR build**: After the first CSR build, `defer_csr` stayed `true`, causing all subsequent `add_edge()` calls to route to `pending_edges` instead of overflow. Each CSR rebuild then lost all previous edges. Fixed by setting `defer_csr = false` in `build_csr_from_pending()`.
- **`edge_weight_mut` for disk mode**: Implemented mutable edge property access for disk graphs, required by `add_connections` with duplicate edge handling (e.g., blueprint builds with temporal edge properties).
- **Disk graph `save_to_dir` missing metadata**: `disk_graph_meta.json` and conn_type_index files were only written to `data_dir`, not to `target_dir` when saving to a different directory. Fixed `save_to_dir` to write metadata to the target.
- **N-Triples mapped mode used compact IDs**: Mapped mode incorrectly used disk-style compact integer IDs instead of string IDs. Now matches memory mode behavior.
- **`enable_columnar()` title column mismatch**: Columnar nodes with missing titles in old stores got no title pushed, causing title column length < row_count. Save→load then failed with "blob too small". Fixed by always using node.title as fallback.
- **`edge_weight_mut` arena offset bug**: The flush logic assumed all edge_weight_mut entries were contiguous at the end of the arena, but read-only `edge_weight` calls interspersed between writes caused wrong offsets. Replaced arena-based tracking with a dedicated `edge_mut_cache` HashMap.
- **N-Triples mapped mode used compact edge path**: `use_compact` was true for mapped mode, sending it through `create_edges_compact()` instead of `create_edges_strings()`. Mapped now uses the memory-mode path for everything.
- **`InternedKey` hash is now deterministic across processes**: `InternedKey::from_str` previously used `DefaultHasher` (SipHash with a per-process random seed). Since `DiskNodeSlot.node_type` persists this as raw u64 on disk and the loader resolves it via the freshly-built interner's hashes, disk graphs built in one process couldn't be reliably loaded in another. Replaced with FNV-1a 64-bit (zero-alloc, zero new deps, deterministic). **Breaking change for existing disk graphs** saved with an older kglite: their `node_type` u64 values were hashed with a random SipHash seed and will not resolve against the new interner. Rebuild affected disk graphs.
- **Disk mode save/load loses embeddings, timeseries_store, and parent_types**: `save_disk` and `load_disk_dir` only persisted the `FileMetadata` struct, which didn't include `parent_types` and omitted embeddings/timeseries entirely. Describe() output on reloaded disk graphs was missing the "core vs supporting" tier split and `<embeddings>` section. Fix: added `parent_types` to `FileMetadata`, and save/load `embeddings.bin.zst` and `timeseries.bin.zst` alongside the other disk artifacts.
- **`describe()` non-deterministic across processes**: `compute_join_candidates` iterated `node_type_metadata` (HashMap) and broke `sort_by` ties with insertion order. Different HashMap `RandomState` seeds produced different candidate orderings, making checksums unstable. Property iteration is now sorted by name, and the candidate sort uses `(overlap desc, left_type, right_type, left_prop)` as a stable key.
- **Disk mode DataFrame/blueprint builds: wrong node titles/properties after multiple types**: `batch_operations` assigned `DiskNodeSlot.row_id` by slot index (set in `add_node`) instead of the per-type column store row returned by `push_row`. Pass 2 tried to fix this via `node_weight_mut`, but that call materializes into an arena that gets cleared on the next call — so the correction never persisted. Once a second node type was added, slot indices diverged from column store rows, causing `n.title`/`n.id` to read wrong rows (and `None` for out-of-bounds slots). Fix: batch_operations now also calls `DiskGraph::update_row_id` after each deferred assignment. Raises `api_benchmark.py` from 38/51 to 49/51 across all 3 modes.
- **Column store schema rebuild drops titles**: When batch_operations rebuilds a column store due to schema growth, titles for existing rows could be lost if `get_title()` returned None. Fixed by always pushing Null fallback.
- **`save_disk()` now persists `type_indices.bin.zst` and `id_indices.bin.zst`**: Previously only written by the N-Triples builder, DataFrame/blueprint-built disk graphs now also persist these files for correct and fast reload.
- **`write_packed()` preserves all schema columns**: Empty schema columns are now written with null padding instead of being skipped, ensuring lossless metadata round-trip through save→load cycles.

## [0.7.10] - 2026-04-16

### Added
- **Connection-type inverted index**: Built during CSR construction, maps edge types to source node IDs. Enables instant lookup of "which nodes have P31 outgoing edges" for unanchored edge queries. Cold-cache `MATCH (a)-[:P31]->(b) LIMIT 50` improved from 14.5s to 4.6s.
- **madvise hints for edge scans**: Sequential/DontNeed advisories on edge_endpoints during full-graph aggregation to reduce page cache pollution.

### Changed
- **FusedNodeScanTopK**: New fused clause for `MATCH (n:Type) RETURN n.prop ORDER BY n.prop LIMIT K` — single-pass scan with inline top-K selection, avoids materializing all rows. String sort keys supported.
- **Streaming top-K for FusedMatchReturnAggregate**: Iterates group nodes directly from type_indices instead of materializing all PatternMatch objects.
- **Edge-centric aggregation**: For untyped group nodes, scans edge_endpoints sequentially with HashMap accumulation instead of per-node iteration.
- **Lightweight peer iteration**: `expand_from_node` skips edge_endpoints reads when edge variable is unnamed (disk-only, reduces I/O by ~50%).

## [0.7.9] - 2026-04-16

### Changed
- **Zero-allocation edge counting**: `count()` queries on edge patterns use a new fast path that iterates CSR edges without materializing `EdgeData`. With sorted CSR, uses binary search to narrow to matching edge type. Result: "count instances of City" dropped from 2.3s to 37ms (63x faster).
- **WHERE-MATCH fusion**: The executor detects MATCH followed by WHERE and evaluates the WHERE predicate inline during pattern expansion. Non-matching rows are skipped immediately, and expansion stops after finding exactly LIMIT matching rows. Previously stuck queries (>10 min) now complete within timeout.
- **LIMIT push-down through WHERE**: Extended `push_limit_into_match` to handle `MATCH → WHERE → RETURN → LIMIT` pattern. The executor enforces exact LIMIT during fused WHERE evaluation.
- **Pre-computed edge type counts**: Edge type counts are computed during CSR build (zero overhead — counted inline during endpoint materialization). Persisted to metadata so `FusedCountEdgesByType` is O(1) on reload.

### Fixed
- **Wikidata type merge**: Q-code types (e.g., "Q5") now properly merge into human-readable labels ("human") during N-Triples build. Previously, when both "Q5" and "human" existed as types, the merge was skipped — now indices and column stores are merged correctly.
- **Column store key remapping**: Property log entries with old Q-code InternedKeys are remapped to merged label keys during Phase 1b, ensuring column stores have correct data after type merges.

## [0.7.8] - 2026-04-15

### Added
- **`set_default_timeout(timeout_ms)`**: Set a default per-query timeout (milliseconds) applied to all `cypher()` calls. Per-query `timeout_ms` overrides it.
- **`set_default_max_rows(max_rows)`**: Set a default cap on intermediate result rows. Queries exceeding this return an error with guidance to add LIMIT. Per-query `max_rows` overrides it.
- **`cypher(max_rows=N)`**: Per-query max rows limit parameter.

### Changed
- **Cypher LIMIT push-down**: Tightened source candidate cap from 10,000× to 100× the LIMIT value. Queries like `MATCH (n:Type)-[:EDGE]->(m) RETURN ... LIMIT 10` on large types are now ~100x faster (avoids allocating the full type index).
- **Cypher pattern start-node optimization**: Improved selectivity estimation for property-filtered nodes (equality filters now estimate /100 instead of /10). Lowered reversal threshold from 10× to 5×. Queries with filters on the target node (e.g., `WHERE b.prop = 'X'`) are now 2-3× faster.
- **DiskGraph edge iteration**: DiskEdges iterator now reads CSR edges lazily from the mmap instead of pre-collecting into a Vec. Eliminates O(degree) allocation per iterator — critical for high-degree nodes at Wikidata scale.
- **DiskGraph direct columnar property access**: Property checks in Cypher WHERE clauses and pattern matching now read individual column values directly from the ColumnStore on disk graphs, bypassing full `NodeData` materialization. Eliminates arena allocation and unnecessary id/title reads — ~3x fewer mmap reads per property check.
- **DiskGraph CSR sorted by connection type**: CSR edges are now sorted by `(node, connection_type)` during build. Edge-type filtering uses binary search instead of linear scan — O(log D + matching) instead of O(D) for high-degree nodes. Metadata flag `csr_sorted_by_type` ensures backward compatibility with older graphs.
- **Fused aggregation with WHERE clauses**: `FusedNodeScanAggregate` now activates for queries with property filters (e.g., `MATCH (n:Entity) WHERE n.pop > 1M RETURN n.continent, count(n)`). `FusedMatchReturnAggregate` now supports property filters on the unbound (counted) node. Both avoid materializing intermediate result rows.

## [0.7.7] - 2026-04-15

### Added
- **Schema locking**: `lock_schema()` / `unlock_schema()` enforce the graph's known schema on Cypher mutations (CREATE, SET, MERGE). Invalid writes return descriptive errors with "did you mean?" suggestions via edit-distance matching. Works on any graph — locks against `node_type_metadata` and `connection_type_metadata`.
- **`from_blueprint(lock_schema=True)`**: Convenience parameter to lock the schema immediately after blueprint loading.
- **`schema_locked` property**: Check whether the schema is currently locked.
- **`describe()` schema-locked notice**: When schema is locked, `describe()` includes a `<schema-locked>` element so agents know writes will be validated.

## [0.7.6] - 2026-04-12

### Fixed
- **Silent data loss on incremental save/load**: Loading a `.kgl` file, adding or updating nodes, then saving and loading again would silently lose properties for the new/updated nodes. The v3 column writer now always consolidates all node properties (Compact, Map, and Columnar) into column stores before writing.
- **Corrupt `.kgl` file on re-save**: Simply loading and re-saving a `.kgl` file (with no changes) could produce a corrupt file that failed to load with `blob too small for offsets`. The v3 column loader was building the ColumnStore schema from `node_type_metadata` (which includes id/title fields) instead of from the column section metadata (which only has property columns), creating empty placeholder columns that corrupted on write.
- **`enable_columnar()` dropped Columnar nodes on rebuild**: When rebuilding column stores, nodes already using Columnar storage were skipped, then their old stores were replaced — losing their properties. Now reads properties from old Columnar stores during rebuild and preserves mapped-mode id/title columns.

## [0.7.5] - 2026-04-10

### Added
- **`describe()` extreme-scale support**: Adaptive output for graphs with thousands or millions of types. Four scale tiers: Small (≤15 types, inline detail), Medium (16-200, compact listing), Large (201-5000, top-50 + search hint), Extreme (5001+, statistical summary).
- **`describe(type_search='...')`**: Find types by name with 1-layer neighborhood fan-out. Returns matching types with their connections plus connected types — enables domain discovery in a single call.
- **`rebuild_caches()`**: Force computation of type connectivity, edge type counts, and connection endpoint types in a single O(E) pass. Caches are persisted by `save()` and restored by `load()`.
- **Type connectivity cache**: Pre-computed type-level graph `(src_type, conn_type, tgt_type, count)` triples. Makes `type_search` and `describe(types=[...])` instant on any scale.
- **Lazy connectivity compute**: For Large/Extreme graphs, type connectivity is computed on first `type_search` call and cached for the session.

### Changed
- **`describe()` on Wikidata**: Output reduced from 2.9MB/508s to 3KB/0.15s. `type_search` with warm cache is sub-millisecond (was 2082s).
- **Performance guards**: Sampled neighbor schema for types >50K nodes, skip join candidates for >200 types, bounded error messages for large graphs.
- **Connection overview capping**: `describe(connections=True)` caps at 50 connection types for graphs with >500 connection types.
- **Empty endpoint resolution**: Disk-imported graphs with empty `connection_type_metadata` endpoints get source/target types resolved via bounded edge scan.
- **CSR build: in_edges merge sort**: Replaced scatter-write with merge sort for in_edges (1407s → 259s, 5.4× faster). Power-law target distributions caused page cache thrashing with scatter; merge sort uses only sequential I/O.
- **CSR build: zero-fill elimination**: `mapped_zeroed` creates mmap files at full size without writing zeros — OS lazy-fills pages on demand. Saves 13.8 GB of writes.
- **CSR build: edge_endpoints reuse**: Steps 3-4 read from edge_endpoints (written in step 1) instead of re-reading pending_edges. Eliminates one redundant 13.8 GB copy.
- **Wikidata build: 93 min → 73 min** (Phase 3 CSR: 1842s → 631s).
- **Cypher backtick type names**: `MATCH (n:\`programming language\`)` now works. Pattern parser re-adds backticks when reconstructing identifiers with spaces.
- **Q-code type resolution**: Post-Phase-1 pass resolves raw Q-code type names (e.g., Q13442814 → "scholarly article") using the complete label cache. Single sequential scan of node_slots.
- **NTriples type connectivity**: Type connectivity triples accumulated inline during edge creation, eliminating separate O(E) rebuild pass for freshly built graphs.
- **`get_edge_type_counts()` memory-safe**: Fallback path uses `edge_endpoint_keys()` (mmap reads) instead of `edge_weights()` (which materialized all EdgeData → OOM on disk graphs).

## [0.7.4] - 2026-04-08

### Changed
- **CsrEdge 16 → 8 bytes**: Removed `conn_type` from CSR edge records. Connection type stored only in `EdgeEndpoints`. Saves ~14 GB on Wikidata (out_edges + in_edges halved).
- **MergeSortEntry 24 → 12 bytes**: Removed `conn_type` from sort entries. 2× more edges per sort chunk during CSR build.
- **Edge conn_type pre-filter**: `DiskEdges` iterator checks `edge_endpoints` before `materialize_edge()`, skipping arena allocation and property HashMap lookup for non-matching edges.
- **Arena clearing at query boundaries**: `reset_arenas()` called at start of every Cypher execution. Prevents unbounded memory growth across queries (was the OOM cause on Wikidata).
- **`node_type_of()` — zero-materialization type check**: Reads directly from mmap'd `node_slots` (16-byte struct). Used in all Cypher executor fast paths and pattern matching hot loops instead of `node_weight()`.
- **Edge properties fast path**: `materialize_edge()` skips HashMap lookup when `edge_properties` is empty (common for Wikidata — 862M edges, zero properties).
- **Source node cap with LIMIT**: Multi-hop patterns with LIMIT N only allocate PatternMatch objects for `N × 10,000` source nodes instead of the full type.
- **`expand_from_node` limit propagation**: Edge expansion stops after collecting enough results instead of eagerly materializing all matching edges.
- **`id_indices` built on load**: Disk graphs build id_indices from column stores during load (no node materialization). Enables O(1) cross-type id lookup.
- **`lookup_by_id_normalized` trusts id_indices**: When id_indices exist for a type, the O(1) lookup result is trusted without falling through to linear scan.

### Added
- `DiskGraph::node_type_of()` — O(1) node type lookup from mmap'd node_slots.
- `DiskGraph::reset_arenas()` — public arena clearing for query boundary use.
- `DiskGraph::edges_directed_filtered_iter()` — pre-filtered edge iteration by connection type.
- `GraphBackend::node_type_of()`, `edges_directed_filtered()`, `reset_arenas()` — backend-agnostic wrappers.
- `DirGraph::build_id_index_from_columns()` — builds id_indices directly from mmap'd column stores without node materialization.
- `WHERE id(n) = X` pushdown in planner — converts `id()` function calls to inline `{id: X}` pattern properties.
- Cross-type id lookup in `find_matching_nodes` — untyped `{id: X}` patterns try all types via id_indices (O(types) × O(1)).
- `estimate_node_selectivity` returns 1 for any `{id: X}` pattern regardless of type.

### Fixed
- **Typed edge queries on disk graphs returning 0 rows**: `has_connection_type()` returned false when `connection_type_metadata` was empty (disk graphs skip O(types²) metadata). Fixed by falling back to interner check.
- **N-Triples build not registering connection type names**: Added lightweight connection type metadata registration (names only, no type×type matrix).

## [0.7.3] - 2026-04-08

### Changed
- **Single-file mmap column storage**: Column stores written to a single `columns.bin` file with mmap-backed reads. Replaces per-type `columns/<type>/columns.zst` layout. Near-instant load via mmap (no decompression).
- **Property log (disk mode)**: Phase 1 serializes properties to a zstd-compressed log file instead of building ColumnStores in-memory. Phase 1b replays the log to build columns in bulk — avoids O(n²) column rebuilds.
- **Partitioned CSR build**: Default CSR algorithm switched to hash-partitioned (Kuzu pattern). Merge-sort still available via `KGLITE_CSR_ALGO=merge_sort`.
- **File-backed pending edges**: `pending_edges` buffer uses mmap-backed `MmapOrVec` instead of heap `Vec`, avoiding ~14 GB heap allocation at Wikidata scale.
- **Auto-typing from P31**: N-Triples loader automatically derives node types from `P31` (instance-of) values, resolving Q-codes to labels. Entities without P31 default to "Entity".
- **Sparse property overflow**: Properties with <5% fill rate stored in a compact overflow bag instead of dense columns, reducing file size for wide schemas.

### Added
- `MmapColumnStore` — mmap-backed column reader for disk mode.
- `BuildColumnStore` — direct column writer that streams to the mmap file.
- `PropertyLogWriter`/`PropertyLogReader` — zstd-compressed property spill log for disk builds.
- `BlockPool`/`BlockColumn` — block-allocated typed column storage.
- `TypeBuildMeta` — per-type metadata for build-time column schema discovery.
- `MmapOrVec::load_mapped_region()`, `from_vec()`, `as_mut_bytes()` — new helpers for region-mapped and bulk byte access.
- `DiskGraph::update_row_id()` — fix per-type row_id mapping after column conversion.
- `ColumnStore::from_mmap_store()`, `from_raw_columns()` — constructors for mmap-backed and direct-built stores.
- `ColumnStore` id/title column accessors now work in disk mode.

### Fixed
- **code_tree stack overflow**: `extract_comment_annotations` switched from recursive to iterative traversal, fixing crashes on deeply nested ASTs.

## [0.7.2] - 2026-04-07

### Fixed
- **code_tree stack overflow**: `extract_comment_annotations` switched from recursive to iterative traversal, fixing crashes on deeply nested ASTs.

## [0.7.1] - 2026-04-06

### Changed
- **CSR build: external merge sort** (DuckDB-inspired). Replaced random-I/O scatter with external merge sort — sort chunks in memory, merge sequentially. All disk I/O is sequential. Phase 3 at Wikidata scale (862M edges, 16 GB RAM): ~16 min vs 90+ min previously.
- **Disk graph auto-persistence**: CSR arrays and metadata written directly to graph dir during build. No separate `save()` step needed. Mutations (`add_node`, `add_edge`, etc.) auto-flush metadata.
- **Disk graph raw storage**: Save/load uses raw `.bin` files (direct mmap) instead of zstd compression. Load is near-instant (mmap, no decompression). Legacy `.bin.zst` files still supported for loading.
- **Mmap-backed edge buffer**: N-Triples loader streams edges to mmap during Phase 1 (0 heap for edge buffer). Eliminates 13.8 GB heap allocation at Wikidata scale.

### Fixed
- **Memory leak in N-Triples loader**: `edge_buffer` (13.8 GB at Wikidata scale) was kept alive during Phase 3 CSR build, doubling peak memory. Now dropped immediately after Phase 2.
- **Disk thrashing during CSR build**: Random writes to mmap caused SSD thrashing. All writes are now sequential.
- **Temp file cleanup**: CSR build temp files cleaned up immediately after merge. Drop impl flushes metadata as safety net.

## [0.7.0] - 2026-04-05

### Added
- **Disk storage mode**: `KnowledgeGraph(storage="disk", path="./my_graph")` — fully disk-backed graph for very large datasets (100M+ nodes, 1B+ edges). Data lives on disk via mmap, using ~10% of equivalent in-memory RAM. The directory IS the graph — no separate save step needed.
- **GraphBackend abstraction**: Unified API across InMemory (petgraph), Mapped, and Disk backends. All Cypher queries, fluent API, and graph algorithms work identically across all three storage modes.
- **CSR edge storage**: Disk mode uses cache-friendly Compressed Sparse Row format. 3-4x faster than default on WHERE filters, SELECT, and SET operations at 100k scale.
- **zstd N-Triples support**: `load_ntriples()` now accepts `.nt.zst` files — 30x faster decompression than bz2.
- **`enable_disk_mode()`** method: Convert existing in-memory graph to disk-backed CSR.
- **`path` parameter** on constructor: Required for `storage="disk"`.

### Changed
- **Mapped mode**: Fixed O(n²) Arc clone bug — 50-300x faster `add_nodes` in mapped mode.
- **N-Triples loader**: 81x faster via bulk columnar conversion, pipeline parallelism, zero-copy parsing, byte-level filtering, and dense Vec edge lookup.

### Fixed
- Schema extension bug in mapped mode incremental `add_nodes`.
- `add_connections()` in disk mode auto-builds CSR so queries work immediately.

## [0.6.18] - 2026-03-30

### Fixed
- **Cypher LIMIT**: 16x faster multi-hop traversals with LIMIT. `MATCH (a)-[:R]->(b)-[:R]->(c) RETURN ... LIMIT 20` now pushes the limit into the pattern matcher — early termination at the last hop, overcommit budgets at intermediate hops. Benchmarks show parity with Neo4j on 2-hop queries.

## [0.6.17] - 2026-03-30

### Added
- `kglite.to_neo4j(graph, uri, ...)` — push graph data directly to a Neo4j database using batched UNWIND operations. Supports `clear`/`merge` modes, selection export, and verbose progress. Requires the `neo4j` package (`pip install neo4j` or `pip install kglite[neo4j]`).
- **ResultView**: Polars-style table display — `repr()` and `print()` now show a bordered table with column headers. Large results show first 10 + last 5 rows with `…` separator.
- **ResultView**: Improved `help(ResultView)` with quick-reference cheat sheet and examples on all methods.

### Fixed
- **code_tree**: Parse output (`Found N files`) now respects `verbose=False` — silent by default.

## [0.6.16] - 2026-03-30

### Changed
- **ResultView**: Polars-style table display — `repr()` and `print()` now show a bordered table with column headers instead of `ResultView(N rows, columns=[...])`. Large results show first 10 + last 5 rows with `…` separator.
- **ResultView**: Improved `help(ResultView)` with quick-reference cheat sheet and examples on all methods.
- **code_tree**: Parse output (`Found N files`) now respects `verbose=False` — silent by default.

## [0.6.15] - 2026-03-30

### Added
- `kglite.repo_tree(repo)` / `code_tree.repo_tree(repo)` — clone a GitHub repository and build a knowledge graph in one call. Cloned files are cleaned up by default; pass `clone_to=` to keep them locally. Supports private repos via `token=` or `GITHUB_TOKEN` env var.

### Fixed
- **code_tree**: Auto-create stub nodes for external base classes, enums, and traits referenced in EXTENDS, IMPLEMENTS, and HAS_METHOD edges — eliminates all "rows skipped: node not found" warnings during graph building.

## [0.6.12] - 2026-03-30

### Fixed
- **BUG-21**: Window functions (`row_number`, `rank`, `dense_rank`) crash with "Window function must appear in RETURN/WITH clause" when query has `WITH` aggregation + `ORDER BY` + `LIMIT`. The planner's `fuse_order_by_top_k` optimization now skips fusion when RETURN contains window functions.

### Changed
- Extracted window function execution into `window.rs` module (~240 lines out of executor.rs)
- Moved `is_aggregate_expression` / `is_window_expression` from executor.rs to ast.rs for cross-module reuse

## [0.6.11] - 2026-03-29

### Fixed

- **19 Cypher engine bugs resolved** — systematic fix of all bugs discovered via legal knowledge graph testing (BUG-01 through BUG-20, except BUG-04 which requires large-graph validation).

#### Critical — Silent wrong results
- **BUG-01**: Equality filter + GROUP BY no longer returns empty results. WHERE clause is now preserved after predicate pushdown to guarantee correctness when fusion fails.
- **BUG-02**: ORDER BY + LIMIT preserves integer types. `count()`, `size()`, `sum()` on integers no longer convert to float through the top-K heap path.
- **BUG-03**: HAVING clause is now propagated when the planner converts RETURN to WITH in fused optional-match aggregation.
- **BUG-05**: `RETURN *` expands to all bound variables (nodes, edges, paths, projected) instead of returning `{'*': 1}`.
- **BUG-06**: Path variable on explicit multi-hop patterns (`p = (a)-[]->(b)-[]->(c)`) now captures all intermediate nodes and relationships. `length(p)`, `nodes(p)`, `relationships(p)` return correct results.
- **BUG-17**: `MATCH (n) WHERE n.type = 'X'` on unlabeled nodes now works. Pattern matcher recognizes `type`/`node_type`/`label` as virtual properties.
- **BUG-18**: `labels()` returns consistent list format in both plain RETURN and GROUP BY contexts. Single-element list comparison (`labels(n) = 'Person'`) now works.

#### High — Errors on valid syntax
- **BUG-07**: `stDev()` / `stdev()` recognized as alias for `std()` aggregate function.
- **BUG-08**: `datetime('2024-03-15T10:30:00')` parses correctly instead of crashing on the time portion.
- **BUG-09**: `date()` returns null on invalid input (`''`, `'2016-00-00'`, `'2016-13-01'`) instead of crashing.
- **BUG-10**: `date('...').year`, `.month`, `.day` property access on function results now works.
- **BUG-11**: `[:TYPE1|TYPE2|TYPE3]` pipe syntax for multiple relationship types in MATCH patterns.
- **BUG-12**: `XOR` logical operator implemented with correct precedence (between OR and AND).
- **BUG-13**: `%` modulo operator implemented for both integer and float operands.
- **BUG-14**: `head()` and `last()` list functions implemented.
- **BUG-15**: `IN` operator accepts variable references, parameters, and function results — not just literal `[...]` lists.

#### Medium — Less common patterns
- **BUG-16**: Boolean/comparison expressions (`STARTS WITH`, `CONTAINS`, `>`, `=~`, etc.) work in RETURN/WITH clauses, evaluating to boolean values.
- **BUG-19**: `null = null` and `null <> null` return null (Cypher three-valued logic) instead of syntax error.
- **BUG-20**: Map all-properties projection `n {.*}` supported.

### Added

- **`Expression::PredicateExpr`** — AST variant bridging the expression/predicate boundary, enabling boolean predicates in RETURN/WITH items.
- **`Expression::ExprPropertyAccess`** — property access on arbitrary expression results (e.g. `date().year`).
- **`Expression::Modulo`** — modulo arithmetic operator.
- **`Predicate::Xor`** — exclusive-or logical operator.
- **`Predicate::InExpression`** — IN with runtime-evaluated list expressions.
- **`MapProjectionItem::AllProperties`** — wildcard map projection.
- **`EdgePattern.connection_types`** — multi-type edge matching for pipe syntax.
- **Performance benchmark suite** (`bench/benchmark_bugs.py`) — 70 targeted benchmarks covering all affected code paths, with CSV output for version-to-version comparison.

## [0.6.10] - 2026-03-29

### Fixed

- **Multi-MATCH empty propagation** — when the first MATCH in a multi-MATCH query returns 0 rows, subsequent MATCH/OPTIONAL MATCH clauses now correctly return 0 rows instead of matching against the entire graph.
- **Planner fusion guard** — MATCH fusion optimizations (FusedNodeScanAggregate, FusedMatchReturnAggregate, FusedMatchWithAggregate) are now restricted to first-clause position, preventing incorrect results when fused clauses ignored pipeline state from prior clauses.

### Changed

- **Retired legacy pytest/ test suite** — migrated unique test coverage (edge cases, subgraph extraction, pattern matching property filters, connection aggregation, connector API) into the official tests/ suite. Test count grew from 1,573 to 1,609.

## [0.6.9] - 2026-03-22

### Added

- **`'poincare'` distance metric** — new metric for `vector_search()`, `text_score()`, `compare()`, and `search_text()`. Computes hyperbolic distance in the Poincaré ball model, ideal for hierarchical data (taxonomies, ontologies). Based on Nickel & Kiela (2017).
- **`embedding_norm()` Cypher function** — returns the L2 norm of a node's embedding vector. In Poincaré embeddings, norm encodes hierarchy depth (0 = root/general, ~1 = leaf/specific).
- **Stored metric on embeddings** — `set_embeddings(..., metric='poincare')` stores the intended distance metric alongside vectors. Queries default to the stored metric when no explicit `metric=` is passed.

## [0.6.8] - 2026-03-19

### Added

- **`compare()` method** — dedicated API for spatial, semantic, and clustering operations. Replaces the overloaded `traverse(..., method=...)` pattern with a clearer `compare(target_type, method)` signature.
- **`collect_grouped()` method** — materialise nodes grouped by parent type as a dict. `collect()` now always returns a flat `ResultView`.
- **`Agg` helper class** — discoverable aggregation expression builders for `add_properties()`: `Agg.count()`, `Agg.sum(prop)`, `Agg.mean(prop)`, `Agg.min(prop)`, `Agg.max(prop)`, `Agg.std(prop)`, `Agg.collect(prop)`.
- **`Spatial` helper class** — spatial compute expression builders for `add_properties()`: `Spatial.distance()`, `Spatial.area()`, `Spatial.perimeter()`, `Spatial.centroid_lat()`, `Spatial.centroid_lon()`.
- **Traversal hierarchy guide** — new conceptual documentation explaining levels, property enrichment, and grouped collection.

### Breaking

- **`traverse()` no longer accepts `method=`** — use `compare(target_type, method)` instead.
- **`collect()` no longer accepts `parent_type`, `parent_info`, `flatten_single_parent`, or `indices`** — use `collect_grouped(group_by)` for grouped output. `collect()` always returns `ResultView`.

## [0.6.7] - 2026-03-18

### Performance

- **31% faster `.kgl` load** — large files are now memory-mapped directly instead of buffered read; small columns (< 256 KB) skip temp file creation and load into heap.
- **28% faster Cypher queries** — `PropertyStorage::get_value()` returns `Value` directly, avoiding `Cow` wrapping/unwrapping overhead on every property access.
- **Zero-alloc string column access** — `TypedColumn::get_str()` returns `&str` slices into mmap'd data without heap allocation, benefiting all WHERE string comparisons.
- **23% faster save** — reduced overhead from mmap threshold optimizations.

## [0.6.6] - 2026-03-18

### Breaking

- **`.kgl` format upgraded to v3** — files saved with older versions (v1/v2) cannot be loaded; rebuild the graph from source data and re-save.
- **`save_mmap()` and `kglite.load_mmap()` removed** — the v3 `.kgl` format replaces the mmap directory format with a single shareable file that supports larger-than-RAM loading.
- `save()` now leaves the graph in columnar mode after saving (previously restored non-columnar state). This avoids an expensive O(N×P) disable step.

### Added

- **v3 unified columnar file format** — `save()` now writes a single `.kgl` file with separated topology and per-type columnar sections (zstd-compressed). On load, column sections are decompressed to temp files and memory-mapped, keeping peak memory to topology + one type's data at a time.
- `save()` automatically enables columnar storage if not already active — no need to call `enable_columnar()` before saving.
- Loaded v3 files are always columnar (`is_columnar` returns `True`).

### Fixed

- **Temp directory leak** — `/tmp/kglite_v3_*` and `/tmp/kglite_spill_*` directories created during `load()` and `enable_columnar()` are now automatically cleaned up when the graph is dropped.
- Reduced save-side memory usage by eliminating double buffering in column packing.

### Removed

- `save_mmap(path)` method — use `save(path)` instead.
- `kglite.load_mmap(path)` function — use `kglite.load(path)` instead.
- v1 and v2 `.kgl` format support (load and save).
- Dead code: `StringInterner::len()`.

## [0.6.5] - 2026-03-18

### Added

- **Columnar property storage** — `enable_columnar()` / `disable_columnar()` convert node properties to per-type column stores, reducing memory usage for homogeneous typed columns (int64, float64, string, etc.). `is_columnar` property reports current storage mode.
- **Memory-mapped directory format** — `save_mmap(path)` / `kglite.load_mmap(path)` persist graphs as mmap-backed column files, enabling instant startup and out-of-core (larger-than-RAM) workloads. Directory layout: `manifest.json` + `topology.zst` + per-type column files.
- **Automatic memory-pressure spill** — `set_memory_limit(limit_bytes)` configures a heap-byte threshold; `enable_columnar()` automatically spills the largest column stores to disk when the limit is exceeded. `graph_info()` now reports `columnar_heap_bytes`, `columnar_is_mapped`, and `memory_limit`.
- **`unspill()`** — move mmap-backed columnar data back to heap memory (e.g., after deleting nodes to free space).
- **`memmap2` dependency** for memory-mapped file I/O.
- Columnar and mmap benchmarks in `test_bench_core.py` (5 new benchmarks).
- Comprehensive test suite for columnar storage and mmap format (28 new Python tests, 30+ new Rust tests).

### Fixed

- **`vacuum()` now rebuilds columnar stores** — previously, deleting nodes left orphaned rows in columnar storage that were never reclaimed. Now `vacuum()` (and auto-vacuum) automatically rebuilds column stores from only live nodes, eliminating the memory leak.
- `graph_info()` reports `columnar_total_rows` and `columnar_live_rows` for diagnosing columnar fragmentation.
- Boolean columns now correctly persist in mmap directory format (`from_type_str` now matches `"boolean"` in addition to `"bool"`).

### Performance

- 4-11x speedup for columnar/mmap operations: eliminated unnecessary full graph clone in `save_mmap()`, bulk memcpy in `materialize_to_heap()`, async flush, aligned pointer reads, direct push in `push_row()`, and skipped UTF-8 re-validation for string columns.

## [0.6.1] - 2026-03-08

### Changed

- **`describe()` default output now shows edge property names** in the `<connections>` section, improving agent discoverability of edge data without requiring `describe(connections=True)`.
- Improved hint text in describe output to guide agents toward `describe(connections=['CONN_TYPE'])` for edge property stats.
- `write_connections_overview` now reuses pre-computed metadata instead of scanning all edges (performance improvement).

## [0.6.0] - 2026-03-07

### Added

- **Python linting (ruff)** — format + lint enforcement for all Python files. `make lint` now checks both Rust and Python. `make fmt-py` auto-fixes.
- **Coverage reporting** — pytest-cov + Codecov integration in CI (informational, not blocking). `make cov` for local reports.
- **Stubtest** — `mypy.stubtest` verifies `.pyi` stubs match the compiled Rust extension. Runs in CI (py3.12). `make stubtest` for local checks.
- **Property-based testing** — Hypothesis tests for graph invariants (node count, filter correctness, index transparency, Cypher-fluent parity, delete consistency, sort correctness, type roundtrip).
- **Historical benchmark tracking** — pytest-benchmark with `github-action-benchmark` for performance regression detection. `make bench-save` / `make bench-compare` for local use.
- **Diátaxis documentation** — restructured docs into Tutorials, How-to Guides, Explanation, and Reference sections. New architecture and design-decisions explanation pages.
- **GitHub scaffolding** — issue templates (YAML forms), PR template, dependabot, security policy, `.editorconfig`, `.codecov.yml`.
- **PEP 561 `py.typed` marker** — type checkers now recognize KGLite's type stubs.
- **`connection_types` parameter** on `betweenness_centrality()`, `pagerank()`, `degree_centrality()` (stub fix — parameter existed at runtime).
- **`titles_only` parameter** on `connected_components()` (stub fix).
- **`timeout_ms` parameter** on `cypher()` (stub fix).

### Changed

- **Tree-sitter is now an optional dependency** — `pip install kglite[code-tree]` for codebase parsing. Core install reduced to just `pandas`.
- **README rewritten** as a keyword-optimized landing page for discoverability.
- Benchmarks CI job now runs on every push to main (was manual dispatch only).

## [0.5.88] - 2026-03-04

### Added

- **MCP Servers guide** — new docs page covering server setup, core tools, FORMAT CSV export, security, semantic search, and a minimal template

## [0.5.87] - 2026-03-04

### Added

- **`FORMAT CSV` Cypher clause** — append `FORMAT CSV` to any query to get results as a CSV string instead of a ResultView. Good for large data transfers and token-efficient output in MCP servers.

## [0.5.86] - 2026-03-03

### Added

- **`add_connections` query mode** — `add_connections(None, ..., query='MATCH ... RETURN ...')` creates edges from Cypher query results instead of a DataFrame. `extra_properties=` stamps static properties onto every edge.
- **`'sum'` conflict handling mode** — `conflict_handling='sum'` adds numeric edge properties on conflict (Int64+Int64, Float64+Float64, mixed promotes to Float64). Non-numeric properties overwrite like `'update'`. For nodes, `'sum'` behaves identically to `'update'`.

### Fixed

- **`add_connections` query-mode param validation** — `columns`, `skip_columns`, and `column_types` now raise `ValueError` in query mode (previously silently ignored)
- **`describe()` incomplete `add_connections` signature** — now shows `query`, `extra_properties`, `conflict_handling` params and query-mode example

## [0.5.84] - 2026-03-03

### Fixed

- **Cypher edge traversal without ORDER BY** — queries like `MATCH (a)-[r:REL]->(b) RETURN ... LIMIT N` returned wrong row count, NULL target/edge properties, and ignored LIMIT. Root cause: `push_limit_into_match` pushed LIMIT into the pattern executor for edge patterns, causing early termination before edge expansion. Now only pushes for node-only patterns.
- **`create_connections()` silently creating 0 edges** — two sub-bugs: (1) `ConnectionBatchProcessor.flush_chunk` used `find_edge()` which matches ANY edge type, so creating PERSON_AT edges would update existing WORKS_AT edges instead. Now uses type-aware `edges_connecting` lookup. (2) Parent map in `maintain_graph::create_connections` used `HashMap<NodeIndex, NodeIndex>` (single parent per child), losing multi-parent relationships. Now uses `Vec<NodeIndex>` per child and iterates group parents directly.
- **`describe(fluent=['loading'])` wrong parameter name** — documented `properties=` for `add_connections()`, actual parameter is `columns=`
- **`traverse()` with `method='contains'` ignoring `target_type=`** — when spatial method was specified, `target_type=` keyword was ignored and only the first positional arg was used as target type. Now prefers explicit `target_type=` over positional arg.
- **`geometry_contains_geometry` missing combinations** — added `(MultiPolygon, LineString)` and `(MultiPolygon, MultiPolygon)` match arms that previously fell through to `false`

## [0.5.83] - 2026-03-03

### Added

- **`fold_or_to_in` optimizer pass** — folds `WHERE n.x = 'A' OR n.x = 'B' OR n.x = 'C'` into `WHERE n.x IN ['A', 'B', 'C']` for pushdown and index acceleration
- **`InLiteralSet` AST node** — pre-evaluated literal IN with HashSet for O(1) membership testing instead of per-row list evaluation
- **TypeSchema-based fast property key discovery** — `to_df()`, `ResultView`, and `describe()` use TypeSchema for O(1) key lookup when all nodes share a type (>50 nodes)
- **Sampled property stats** — `describe()` and `properties()` sample large types (>1000 nodes) for faster response
- **`StringInterner::try_resolve()`** — fallible key resolution for TypeSchema-based paths
- **`rebuild_type_indices_and_compact` metadata fallback** — scans nodes to build TypeSchemas when metadata is empty (loaded from file)

### Fixed

- **FusedMatchReturnAggregate output columns** — built from return clause items instead of reusing pre-existing columns, fixing wrong column names in fused aggregation results
- **FusedMatchReturnAggregate top-k sort order** — removed erroneous `top.reverse()` calls that inverted DESC/ASC order in `ORDER BY ... LIMIT` queries
- **FusedMatchReturnAggregate zero-count rows** — exclude nodes with zero matching edges (MATCH semantics require at least one match)
- **Path binding variable lookup** — path assignments now find the correct variable-length edge variable instead of grabbing the first available binding
- **UNWIND null produces zero rows** — `UNWIND null AS x` now correctly produces no rows per Cypher spec instead of emitting a null row
- **InLiteralSet cross-type equality** — `WHERE n.id IN [1, 2, 3]` now matches float values via `values_equal` fallback
- **NULL = NULL returns false in WHERE** — implements Cypher three-valued logic where NULL comparisons are falsy; grouping/DISTINCT unaffected
- **Property push-down no longer overwrites** — `apply_property_to_patterns` uses `entry().or_insert()` to preserve earlier matchers
- **Pattern reversal skips path assignments** — `optimize_pattern_start_node` no longer reverses patterns bound to path variables
- **Fuse guard: HAVING clause** — `fuse_match_return_aggregate` bails out when HAVING is present
- **Fuse guard: vector score aggregation** — `fuse_vector_score_order_limit` bails out when return items contain aggregate functions
- **Fuse guard: bidirectional edge count** — `fuse_count_short_circuits` skips undirected patterns that could produce wrong counts
- **Fuse guard: dead SKIP check removed** — `fuse_order_by_top_k` no longer checks wrong clause index for SKIP
- **Parallel expansion error propagation** — errors in rayon parallel edge expansion are now propagated instead of silently returning empty results
- **Variable-length paths min_hops=0** — source node is now yielded at depth 0 when `min_hops=0` (e.g., `[*0..2]`)
- **Parallel distinct target dedup** — parallel expansion path now applies `distinct_target_var` deduplication matching the serial path
- **Unterminated string/backtick detection** — tokenizer now returns errors for unclosed string literals and backtick identifiers
- **String reconstruction preserves escapes** — `CypherToken::StringLit` re-escapes quotes and backslashes during reconstruction

## [0.5.82] - 2026-03-03

### Changed

- **zstd compression for save/load** — replaced gzip level 3 with zstd level 1; Save 9.5s → 1.1s (8.6×), Load 2.3s → 1.0s (2.2×), file 7% smaller. Backward-compatible: old gzip files load transparently
- **Vectorized pandas series extraction** — `convert_pandas_series()` uses `series.tolist()` + `PyList.get_item()` instead of per-cell `Series.get_item()`, plus batch `extract::<Vec<Option<T>>>()` for Float64/Boolean/String. Build 24.7s → 19.3s
- **Fast lookup constructors** — `TypeLookup::from_id_indices()` and `CombinedTypeLookup::from_id_indices()` reuse pre-built `DirGraph.id_indices` instead of scanning all nodes
- **Skip edge existence check on initial load** — `ConnectionBatchProcessor.skip_existence_check` flag bypasses `find_edge()` when no edges of that type exist yet
- **Pre-interned property keys** — intern column name strings once before the row loop, use `Vec<(InternedKey, Value)>` instead of per-row `HashMap<String, Value>` for node creation
- **Single-pass load finalize** — `rebuild_type_indices_and_compact()` combines type index rebuild + Map→Compact property conversion in one pass, with TypeSchemas built from metadata instead of scanning nodes
- **Zero-alloc InternedKey deserialization** — custom serde Visitor hashes borrowed `&str` from the decompressed buffer, eliminating ~5.6M String allocations per load
- **Remove unnecessary `.copy()` on first CSV read** in blueprint loader

## [0.5.81] - 2026-03-02

### Added

- **Comparison pushdown into MATCH** — `WHERE n.prop > val` (and `>=`, `<`, `<=`) is now pushed from WHERE into MATCH patterns, filtering during node scan instead of post-expansion. Includes range merging (`year >= 2015 AND year <= 2022` → single Range matcher). Benchmark: filtered 2-hop query 109ms → 14ms (7.6×), property filter 2.5ms → 0.8ms (3×)
- **Range index acceleration** — pushed comparisons now use `create_range_index()` B-Tree indexes via `lookup_range()` for O(log N + k) scans instead of O(N) type scans
- **Reverse fused aggregation** — `MATCH (:A)-[:REL]->(b:B) RETURN b.prop, count(*)` (group by target node) now fuses into a single pass like source-node grouping. In-degree benchmark: 26ms → 9ms (2.8×)
- **EXISTS/NOT EXISTS fast path** — direct edge-existence check for simple EXISTS patterns instead of instantiating a full PatternExecutor per row. NOT EXISTS: 2372ms → 0.3ms (7400×)
- **FusedMatchReturnAggregate top-k** — BinaryHeap-based top-k selection during edge counting, avoiding full materialization + sort. In-degree top-20: 10.5ms → 5.0ms
- **FusedOrderByTopK external sort expression** — ORDER BY on expressions not in RETURN items now fuses into the top-k heap, projecting only surviving rows. UNION per-arm: 5.4ms → 2.3ms
- **FusedNodeScanAggregate** — single-pass node scan with inline accumulators (count/sum/avg/min/max) for `MATCH (n:Type) RETURN group_keys, aggs(...)`, avoiding intermediate ResultRow creation
- **FusedMatchWithAggregate** — fuse `MATCH...WITH count()` into single pass (same as MATCH+RETURN fusion but for pipeline continuation)
- **DISTINCT push-down into MATCH** — when `RETURN DISTINCT` references a single node variable, pre-deduplicate by NodeIndex during pattern matching. Includes intermediate-hop dedup for anonymous nodes. Filtered 2-hop DISTINCT: 15ms → 10ms
- **UNION hash-based dedup** — replace `HashSet<Vec<Value>>` with hash-of-values approach for UNION (non-ALL) deduplication
- **35-query DuckDB/SQLite comparison benchmark** (`bench_graph_traversal.py`)

## [0.5.80] - 2026-03-02

### Added

- **`closeness_centrality(sample_size=…)`** — stride-based node sampling for closeness centrality, matching the existing betweenness pattern; reduces O(N²) to O(k×(N+E)) for approximate results on large graphs
- **`copy()` / `__copy__` / `__deepcopy__`** — deep-copy a `KnowledgeGraph` in memory without disk I/O, useful for running mutations on an independent copy

### Changed

- **`compute_property_stats` value-set cap** — stop cloning values into the uniqueness `HashSet` once `max_values+1` entries are collected, avoiding O(N) clones for high-cardinality properties
- **Closeness centrality Cypher `CALL`** — `CALL closeness({sample_size: 100})` now supported alongside `normalized` and `connection_types`
- **Regex cache in fluent filtering** — pre-compile `Regex` patterns before filter loops (was compiling per-node); `fluent_where_regex` 302 ms → 1 ms
- **Single-pass property stats** — replaced O(N×P) two-pass scan with O(N×avg_props) single-pass accumulator
- **Pre-computed neighbor schemas** — `describe()` scans all edges once instead of per-type

## [0.5.79] - 2026-03-02

### Added

- **Window functions** — `row_number()`, `rank()`, `dense_rank()` with `OVER (PARTITION BY ... ORDER BY ...)` syntax for ranking within result partitions
- **HAVING clause** — post-aggregation filtering on RETURN and WITH (`RETURN n.type, count(*) AS cnt HAVING cnt > 5`)
- **Date arithmetic** — DateTime ± Int64 (add/subtract days), DateTime − DateTime (days between), `date_diff()` function
- **Window function performance** — pre-computed column names, constant folding, OVER spec deduplication, rayon parallelism, fast path for unpartitioned windows

## [0.5.78] - 2026-03-02

### Changed

- **Betweenness BFS inner loop** — merged redundant `dist[w_idx]` loads into cached `if/else if` branch, eliminating a second memory access per edge in both parallel and sequential paths
- **Pre-intern connection types in algorithms** — betweenness, pagerank, degree, closeness, louvain, and label propagation now pre-intern connection type filters once per call instead of hashing per-edge
- **Adjacency list dedup** — undirected adjacency lists are now sorted and deduplicated to prevent double-counting from bidirectional edges (A→B + B→A)
- **3-way traversal benchmark** — added DuckDB (columnar/vectorized) alongside SQLite and KGLite with optimized batch queries

## [0.5.77] - 2026-03-02

### Changed

- **Edge data optimization** — `EdgeData.connection_type` changed from `String` (24 bytes) to `InternedKey` (8 bytes), reducing per-edge overhead by 16 bytes
- **Edge properties compacted** — `EdgeData.properties` changed from `HashMap<InternedKey, Value>` (48 bytes) to `Vec<(InternedKey, Value)>` (24 bytes), saving 24 bytes per edge
- **BFS connection type comparison** — pre-intern connection type before edge loops for `u64 == u64` comparison instead of string equality
- **Static slice in BFS** — `expand_from_node` changed `vec![Direction]` heap allocation to `&[Direction]` static slice
- **Save/load performance** — save time -70% (2,253 → 682 ms), load time -93% (1,676 → 119 ms) on 50k node / 150k edge benchmark
- **Deep traversal speedup** — 8-20 hop citation queries 16-28% faster from interned comparison and eliminated heap allocations

## [0.5.76] - 2026-03-01

### Changed

- **BFS traversal optimization** — replaced `HashSet` visited set with `Vec<bool>` for cache-friendly O(1) lookups during variable-length path expansion
- **Skip redundant node type checks** — planner now marks edges where the connection type guarantees the target node type, avoiding unnecessary `node_weight()` loads during BFS
- **Skip edge data cloning** — unnamed edge variables no longer clone `connection_type` and `properties`, eliminating thousands of heap allocations per traversal
- **DISTINCT dedup optimization** — uses `Value` hash keys instead of `format_value_compact()` string allocation per row

### Added

- **Graph traversal benchmark suite** — SQLite recursive CTE vs KGLite across 15 query types (citation chains, shortest path, reachability, triangles, neighborhood aggregation)

## [0.5.75] - 2026-03-01

### Added

- **`keys()` function** — `keys(n)` / `keys(r)` returns property names of nodes and relationships as a JSON list
- **Math functions** — `log`/`ln`, `log10`, `exp`, `pow`/`power`, `pi`, `rand`/`random` (previously documented but not implemented)
- **`datetime()` alias** — `datetime('2020-01-15')` works identically to `date()`
- **DateTime property accessors** — `d.year`, `d.month`, `d.day` on DateTime values (via WITH alias)
- **Scientific notation** — tokenizer now parses `1e6`, `1.5e-3`, `2E+10` as float literals

### Fixed

- **String function auto-coercion** — `substring`, `left`, `right`, `split`, `replace`, `trim`, `reverse` now auto-coerce DateTime/numeric/boolean values to strings instead of returning NULL
- **`describe()` algorithm hint** — fixed misleading `YIELD node, score|community|cluster` that didn't mention `component`; now shows which yield name belongs to which procedure
- **Spatial coordinate order note** — added documentation clarifying WKT uses (longitude latitude) while `point()` uses (latitude, longitude)

## [0.5.74] - 2026-03-01

### Added

- **Multi-hop traversal benchmarks** — scale-free graph benchmarks at 1K/10K/50K/100K nodes with hop depths 1–8, comparable to TuringDB/Neo4j multi-hop benchmarks
- **Blueprint documentation** — standalone guide page with step-by-step walkthrough, real CSV examples, and troubleshooting

### Changed

- **Variable-length path BFS** — global dedup mode skips path tracking when path info isn't needed (no `p = ...` assignment, no named edge variable), reducing memory and redundant exploration (~4x faster)
- **WHERE IN predicate pushdown** — `WHERE n.id IN [list]` is now pushed into the MATCH pattern and resolved via id-index O(1) lookups instead of post-filtering all nodes (~1,400x faster on 10K 8-hop traversals)

## [0.5.73] - 2026-02-27

### Changed

- **README** — added blueprint loading and code review examples to Quick Start, doc links on each section
- **CLAUDE.md** — simplified and consolidated conventions

## [0.5.72] - 2026-02-27

### Added

- **Documentation site** — Sphinx + Furo docs with auto-generated API reference from `.pyi` stubs, hosted on Read the Docs. Guide pages for Cypher, data loading, querying, semantic search, spatial, timeseries, graph algorithms, import/export, AI agents, and code tree.

## [0.5.71] - 2026-02-27

### Added

- **`traverse()` API improvements:**
  - `target_type` parameter — filter targets to specific node type(s): `traverse('OF_FIELD', direction='incoming', target_type='ProductionProfile')` or `target_type=['ProductionProfile', 'FieldReserves']`
  - `where` parameter — alias for `filter_target`, consistent with the fluent API: `traverse('HAS_LICENSEE', where={'title': 'Equinor'})`
  - `where_connection` parameter — alias for `filter_connection`: `traverse('RATED', where_connection={'score': {'>': 4}})`
  - `help(g.traverse)` now shows a comprehensive docstring with args, examples, and usage patterns
- **Temporal awareness** — first-class support for time-dependent nodes and connections:
  - Declare temporal columns via `column_types={"fldLicenseeFrom": "validFrom", "fldLicenseeTo": "validTo"}` on `add_nodes()` or `add_connections()` — auto-configures temporal filtering behind the scenes (same pattern as spatial `"geometry"` / `"location.lat"`)
  - `date("2013")` sets a temporal context for the entire chain — all subsequent `select()` and `traverse()` calls filter to that date instead of today
  - `date("2010", "2015")` — range mode: include everything valid at any point during the period (overlap check)
  - `date("all")` — disable temporal filtering entirely (show all records regardless of validity dates)
  - `select()` auto-filters temporal nodes to "currently valid" (or the `date()` context). Pass `temporal=False` to include all historic records
  - `traverse()` auto-filters temporal connections to "currently valid". Override with `at="2015"`, `during=("2010", "2020")`, or `temporal=False`
  - `valid_at()` / `valid_during()` auto-detect field names from temporal config; NULL `date_to` treated as "still active"
  - Display (`sample()`, `collect()`) filters connection summaries to temporally valid edges
  - `describe()` includes `temporal_from`/`temporal_to` attributes on configured types and connections
  - Blueprint loader: use `"validFrom"` / `"validTo"` property types to auto-configure temporal filtering
  - `set_temporal(type_name, valid_from, valid_to)` available as low-level API for manual configuration
  - Temporal configs persist through `save()`/`load()` round-trips
- **`show(columns, limit=200)`** — compact display of selected nodes with chosen properties. Single-level shows `Type(val1, val2)` per line; after `traverse()` walks the full chain as `Type1(vals) -> Type2(vals) -> Type3(vals)`. Resolves field aliases and truncates long values

## [0.5.70] - 2026-02-26

### Added

- **`to_str(limit=50)`** — format current selection as a human-readable string with `[Type] title (id: x)` headers and indented properties
- **`print(ResultView)` smart formatting** — `ResultView.__str__` uses multiline card format (properties + connection arrows) for ≤3 rows, compact one-liner for >3. Connections show direction with `◆` as the current node: `◆ --WORKS_AT--> Company(id, title)` for outgoing, `Person(id, title) --WORKS_AT--> ◆` for incoming. Long values (WKT geometries, etc.) are truncated with middle ellipsis
- **`sample()` selection-aware** — `sample()` now works on the current selection (`graph.select('Person').sample(3)`) in addition to the existing `sample('Person', 3)` form
- **`head()`/`tail()` preserve connections** — slicing a ResultView carries connection summaries through

## [0.5.67] - 2026-02-26

### Changed

- **BREAKING: Fluent API method renames** — modernized the fluent API surface to match common query DSL conventions:
  - `type_filter()` → `select()`
  - `filter()` → `where()`
  - `filter_any()` → `where_any()`
  - `filter_orphans()` → `where_orphans()`
  - `has_connection()` → `where_connected()`
  - `max_nodes()` → `limit()`
  - `get_nodes()` → `collect()`
  - `node_count()` → `len()` (also adds `__len__` for `len(graph)`)
  - `id_values()` → `ids()`
  - `max_nodes=` parameter → `limit=` everywhere (select, where, traverse, collect, etc.)
- **BREAKING: Retrieval method renames** — dropped inconsistent `get_` prefix and shortened verbose methods:
  - `get_titles()` → `titles()`
  - `get_connections()` → `connections()`
  - `get_degrees()` → `degrees()`
  - `get_bounds()` → `bounds()`
  - `get_centroid()` → `centroid()`
  - `get_selection()` → `selection()`
  - `get_schema()` → `schema_text()`
  - `get_schema_definition()` → `schema_definition()`
  - `get_last_report()` → `last_report()`
  - `get_operation_index()` → `operation_index()`
  - `get_report_history()` → `report_history()`
  - `get_spatial()` → `spatial()`
  - `get_timeseries()` → `timeseries()`
  - `get_time_index()` → `time_index()`
  - `get_timeseries_config()` → `timeseries_config()`
  - `get_embeddings()` → `embeddings()`
  - `get_embedding()` → `embedding()`
  - `get_node_by_id()` → `node()`
  - `children_properties_to_list()` → `collect_children()` (also `filter=` param → `where=`)

### Removed

- **`get_ids()`** — removed; use `ids()` for flat ID list or `collect()` for full node dicts

## [0.5.66] - 2026-02-26

### Changed

- **Blueprint loader output** — quiet by default (only warnings/errors + summary); verbose mode for per-type detail. Warnings from `add_connections` skips are now tracked in the loader instead of surfacing as raw `UserWarning`s
- **Blueprint settings** — `root` renamed to `input_root`, `output` split into `output_path` (optional directory) + `output_file` (filename or relative path with `../` support). Old keys still accepted for backwards compatibility

### Fixed

- **Float→Int ID coercion** — FK columns with nullable integers (read as float64 by pandas, e.g. `260.0`) are now auto-coerced to int before edge matching. The Rust lookup layer also gained Float64 → Int64/UniqueId fallback as a safety net
- **Timeseries FK edge filtering** — FK edges for timeseries node types now apply the same time-component filter as node creation (e.g. dropping month=0 aggregate rows), preventing "source node not found" warnings for carriers that only have aggregate data

## [0.5.65] - 2026-02-26

### Added

- **`FLUENT.md`** — comprehensive fluent API reference documenting all method-chaining operations: data loading, selection & filtering, spatial, temporal, timeseries, vector search, traversal, algorithms, set operations, indexes, transactions, export, and a fluent-vs-Cypher feature matrix
- **`create_connections()`** — renamed from `selection_to_new_connections` with new capabilities: `properties` dict copies node properties onto new edges (e.g. `properties={'B': ['score']}`), `source_type`/`target_type` override which traversal levels to connect (defaults to first→last level)
- **Comparison-based `traverse(method=...)`** — discover relationships without pre-existing edges. Five methods: `'contains'` (spatial containment), `'intersects'` (geometry overlap), `'distance'` (geodesic proximity), `'text_score'` (semantic similarity via embeddings), `'cluster'` (kmeans/dbscan grouping). `method` accepts a string shorthand (`method='contains'`) or a dict with settings (`method={'type': 'distance', 'max_m': 5000, 'resolve': 'centroid'}`). The `resolve` key controls polygon geometry interpretation: `'centroid'` (force geometry centroid), `'closest'` (nearest boundary point), `'geometry'` (full polygon shape). Produces the same selection hierarchy as edge-based traversal, so all downstream methods work unchanged
- **`add_properties()`** — enrich selected nodes with properties from ancestor nodes in the traversal chain. Supports copy (`['name']`), copy-all (`[]`), rename (`{'new': 'old'}`), aggregate expressions (`'count(*)'`, `'mean(depth)'`, `'sum(production)'`, `'min()'`, `'max()'`, `'std()'`, `'collect()'`), and spatial compute (`'distance'`, `'area'`, `'perimeter'`, `'centroid_lat'`, `'centroid_lon'`)

### Changed

- **`selection_to_new_connections` → `create_connections`** — renamed for brevity. Now defaults to connecting the top-level ancestor to leaf nodes (was parent→child at last level only)

## [0.5.64] - 2026-02-25

### Added

- **List quantifier predicates** — `any(x IN list WHERE pred)`, `all(...)`, `none(...)`, `single(...)` for filtering over lists in WHERE, RETURN, and WITH clauses
- **Exploration hints in `describe()`** — inventory views now surface disconnected node types and join candidates (property value overlaps between unconnected type pairs) to suggest enrichment opportunities
- **Temporal Cypher functions** — `valid_at(entity, date, 'from_field', 'to_field')` and `valid_during(entity, start, end, 'from_field', 'to_field')` for date-range filtering on both nodes and relationships in WHERE clauses. NULL fields treated as open-ended boundaries

### Changed

- **Rewritten examples** — new domain examples: `legal_graph.py` (index-based loading), `code_graph.py` (code tree parsing), `spatial_graph.py` (blueprint loading), `mcp_server.py` (generic MCP server with auto-detected code tools)

## [0.5.63] - 2026-02-25

### Added

- **`export_csv(path)`** — bulk export to organized CSV directory tree with one file per node type and connection type, sub-node nesting, full properties, and a `blueprint.json` for round-trip re-import via `from_blueprint()`
- **Variable binding in MATCH pattern properties** — bare variables from `WITH`/`UNWIND` can now be used in inline pattern properties: `WITH "Oslo" AS city MATCH (n:Person {city: city}) RETURN n`
- **Map literals in Cypher expressions** — `{key: expr, key2: expr}` syntax in `RETURN`/`WITH` for constructing map objects: `RETURN {name: n.name, age: n.age} AS m`
- **WHERE clause inside EXISTS subqueries** — `EXISTS { MATCH (n:Type) WHERE n.prop = expr }` now supports arbitrary WHERE predicates including cross-scope variable references and regex

### Changed

- **Cypher query performance** — eliminated `type_indices` Vec clone on every MATCH (iterate by reference), move-on-last-match optimization to reduce row cloning in joins, pre-allocated result vectors, eliminated unnecessary clone in composite index lookups
- **MERGE index acceleration** — MERGE now uses `id_indices`, `property_indices`, and `composite_indices` for O(1) pattern matching instead of linear scan through all nodes of a type. Orders-of-magnitude faster for batch `UNWIND + MERGE` workloads
- **UNWIND/MERGE clone reduction** — UNWIND moves (instead of cloning) the row for the last unwound item; MERGE iterates source rows by value to avoid per-row cloning

## [0.5.61] - 2026-02-24

### Added

- **PROFILE** prefix for Cypher queries — executes query and collects per-clause statistics (rows_in, rows_out, elapsed_us). Access via `result.profile`
- **Structured EXPLAIN** — `EXPLAIN` now returns a `ResultView` with columns `[step, operation, estimated_rows]` instead of a plain string. Cardinality estimates use type_indices counts
- **Read-only transactions** — `begin_read()` creates an O(1) Arc-backed snapshot (zero memory overhead). Mutations are rejected
- **Optimistic concurrency control** — `commit()` detects graph modifications since `begin()` and raises `RuntimeError` on conflict
- **Transaction timeout** — `begin(timeout_ms=...)` and `begin_read(timeout_ms=...)` set a deadline for all operations within the transaction
- `Transaction.is_read_only` property
- `describe(cypher=['EXPLAIN'])` and `describe(cypher=['PROFILE'])` topic detail pages
- Expanded `<limitations>` section in `describe(cypher=True)` with workarounds for unsupported features
- openCypher compatibility matrix in CYPHER.md

## [0.5.60] - 2026-02-24

### Added

- `describe(cypher=True)` tier 1 hint now highlights KGLite-specific features (||, =~, coalesce, CALL procedures, distance/contains)
- `describe(cypher=True)` tier 2 includes `<not_supported>` section and spatial functions group
- `describe()` overview connection map includes `count` attribute per connection type
- `describe()` connections hint only shown when graph has edges
- `describe(cypher=['spatial'])` topic with distance, contains, intersects, centroid, area, perimeter docs

## [0.5.59] - 2026-02-24

### Added

- `bug_report(query, result, expected, description)` — file Cypher bug reports to `reported_bugs.md`. Timestamped, version-tagged entries prepended to top of file. Input sanitised against HTML/code injection
- `KnowledgeGraph.explain_mcp()` — static method returning a self-contained XML quickstart for setting up a KGLite MCP server (server template, core/optional tools, Claude registration config)

### Fixed

- `collect(node)[0].property` now returns the actual property value instead of the node's title. Previously, `WITH f, collect(fr)[0] AS lr RETURN lr.oil` would return the node title for every property access. Node identity is now preserved through collect→index→WITH pipelines via internal `Value::NodeRef` references

## [0.5.58] - 2026-02-24

### Added

- `CALL cluster()` procedure — general-purpose clustering via Cypher. Supports DBSCAN and K-means methods. Reads nodes from preceding MATCH clause. Spatial mode auto-detects lat/lon from `set_spatial()` config with geometry centroid fallback; property mode clusters on explicit numeric properties with optional normalization. YIELD node, cluster (noise = -1 for DBSCAN)
- `round(x, decimals)` — optional second argument for decimal precision (e.g. `round(3.14159, 2)` → 3.14). Backward compatible: `round(x)` still rounds to integer
- `||` string concatenation operator — concatenates values in expressions (e.g. `n.first || ' ' || n.last`). Null propagates. Non-string values auto-converted
- `describe(cypher=True)` — 3-tier Cypher language reference: compact `<cypher hint/>` in overview (tier 1), full clause/operator/function/procedure listing with `cypher=True` (tier 2), detailed docs with params and examples via `cypher=['cluster','MATCH',...]` (tier 3)
- `describe(connections=True)` — connection type progressive disclosure: overview with `connections=True` (all types, counts, endpoints, property names), deep-dive with `connections=['BELONGS_TO']` (per-pair counts, property stats, sample edges)

## [0.5.56] - 2026-02-23

### Added

- `near_point_m()` — geodesic distance filter in meters (SI units), replaces `near_point_km()` and `near_point_km_from_wkt()`
- Geometry centroid fallback: fluent API spatial methods (`near_point_m`, `within_bounds`, `get_bounds`, `get_centroid`) now fall back to WKT geometry centroid when lat/lon fields are missing but a geometry is configured via `set_spatial` or `column_types`

### Changed

- Cypher `distance(a, b)` returns Null (instead of erroring) when a node has no spatial data, so `WHERE distance(a, b) < X` simply filters those nodes out
- Cypher comparison operators (`<`, `<=`, `>`, `>=`) now follow three-valued logic: comparisons involving Null evaluate to false (previously Null sorted as less-than-everything)

### Removed

- `near_point_km()` — use `near_point_m()` with meters instead (e.g. `max_distance_m=50_000.0` for 50 km)
- `near_point_km_from_wkt()` — subsumed by `near_point_m()` which auto-falls back to geometry centroid

## [0.5.55] - 2026-02-23

### Changed

- Cypher spatial functions now return SI units: `distance()` → meters, `area()` → m², `perimeter()` → meters (were km/km²). Distance uses WGS84 geodesic (Karney algorithm) instead of spherical haversine

### Removed

- `agent_describe()` — replaced by `describe()`. Migration: `graph.agent_describe()` → `graph.describe()`, `graph.agent_describe(detail='full')` → `graph.describe()` (auto-selects detail level)

## [0.5.54] - 2026-02-23

### Added

- `describe(types=None)` — progressive disclosure schema description for AI agents. Inventory mode returns node types grouped by size with property complexity markers and capability flags, connection map, and Cypher extensions. Focused mode (`types=['Field']`) returns detailed properties, connections, timeseries/spatial config, and sample nodes. Automatically inlines full detail for graphs with ≤15 types
- `set_parent_type(node_type, parent_type)` — declare a node type as a supporting child of a core type. Supporting types are hidden from the `describe()` inventory and appear in the `<supporting>` section when the parent is inspected. The `from_blueprint()` loader auto-sets parent types for sub-nodes
- Cypher math functions: `abs()`, `ceil()` / `ceiling()`, `floor()`, `round()`, `sqrt()`, `sign()` — work with Int64 and Float64 values, propagate Null
- String coercion on `+` operator: when one operand is a string, the other is automatically converted (e.g. `2024 + '-06'` → `'2024-06'`). Null still propagates

### Changed

- `describe()` inventory now uses compact descriptor format `TypeName[size,complexity,flags]` instead of size bands. Types listed as flat comma-separated list sorted by count descending. Core types with supporting children show `+N` suffix. Capability flags from supporting types bubble up to their parent descriptor
- `describe()` now shows a `<read-only>` notice listing unsupported Cypher write commands (CREATE, SET, DELETE, REMOVE, MERGE) when the graph is in read-only mode

## [0.5.53] - 2026-02-23

### Added

- `from_blueprint()` — build a complete KnowledgeGraph from a JSON blueprint and CSV files. Supports core nodes, sub-nodes, FK edges, junction edges, timeseries, geometry conversion, filters, manual nodes (from FK values), and auto-generated IDs
- Cypher `date()` function — converts date strings to DateTime values: `date('2020-01-15')`
- `property_types` on blueprint junction edges for automatic type conversion (e.g. epoch millis → DateTime)
- Temporal join support: `ts_*()` functions accept DateTime edge properties and null values as date range arguments
- Cypher `IS NULL` / `IS NOT NULL` now supported as expressions in RETURN/WITH (e.g. `RETURN x IS NULL AS flag`)
- `agent_describe(detail, include_fluent)` — optional detail level adapts output to graph complexity. Graphs with >15 types auto-select compact mode (~5-8x smaller output). Fluent API docs excluded by default (opt-in via `include_fluent=True`)

### Changed

- **Performance**: `agent_describe()` 27x faster (1.3s → 48ms) via property index fast path and scan capping
- **Performance**: `MATCH (n) RETURN count(n)` short-circuits to O(1) via `FusedCountAll` (was ~266ms, now sub-ms)
- **Performance**: `MATCH (n) RETURN n.type, count(n)` short-circuits to O(types) via `FusedCountByType` (was ~727ms, now sub-ms)
- **Performance**: `MATCH ()-[r]->() RETURN type(r), count(*)` short-circuits to O(E) single-pass via `FusedCountEdgesByType` (was ~822ms, now ~3ms)
- **Performance**: `MATCH (n:Type) RETURN count(n)` short-circuits to O(1) via `FusedCountTypedNode` (reads type index length directly)
- **Performance**: `MATCH ()-[r:Type]->() RETURN count(*)` short-circuits via `FusedCountTypedEdge` (single-pass edge filter)
- **Performance**: Edge type counts cached in DirGraph with lazy invalidation on mutations
- **Performance**: Multi-hop fused aggregation for 5-element patterns (e.g. `MATCH (a)-[]->(b)<-[]-(c) RETURN a.x, count(*)`) traverses without materializing intermediate rows
- **Performance**: Regex `=~` operator caches compiled patterns per query execution (compile once, match many)
- **Performance**: PageRank uses pull-based iteration with rayon parallelization for large graphs (3-4x speedup)
- **Performance**: Louvain community detection precomputes loop-invariant division terms
- Timeseries keys stored as `NaiveDate` instead of composite integer arrays (`Vec<Vec<i64>>`)
- `set_time_index()` now accepts date strings (`['2020-01', '2020-02']`) in addition to integer lists
- `get_time_index()` returns ISO date strings (`['2020-01-01', '2020-02-01']`) instead of integer lists
- `get_timeseries()` keys returned as ISO date strings
- `ts_series()` output uses ISO date strings for time keys (e.g. `"2020-01-01"` instead of `[2020, 1]`)
- Null date arguments to `ts_*()` treated as open-ended ranges (no bound)
- Timeseries data format bumped (v2); legacy files skip timeseries loading with a warning

### Fixed

- `MATCH (a)-[]->(b) RETURN count(*)` with all-aggregate RETURN (no group keys) now correctly returns a single row instead of per-node rows
- `ORDER BY` on DateTime properties with `LIMIT` now returns correct results (FusedOrderByTopK optimization extended to handle DateTime, UniqueId, and Boolean sort keys)
- `ORDER BY` on String/Point properties with `LIMIT` now falls back to standard sort instead of returning empty results

## [0.5.52] - 2026-02-22

### Added

- `add_nodes()` now accepts a `timeseries` parameter for inline timeseries loading from flat DataFrames — automatically deduplicates rows per ID and attaches time-indexed channels
- Timeseries resolution extended to support `hour` (depth 4) and `minute` (depth 5) granularity
- `parse_date_string` now handles `'yyyy-mm-dd hh:mm'` and ISO `'yyyy-mm-ddThh:mm'` formats
- **Timeseries support**: per-node time-indexed data channels with resolution-aware date-string queries
- `set_timeseries()` with `resolution` ("year", "month", "day"), `units`, and `bin_type` metadata
- `set_time_index()` / `add_ts_channel()` for per-node timeseries construction
- `add_timeseries()` for bulk DataFrame ingestion with FK-based node matching and resolution validation
- `get_timeseries()` / `get_time_index()` for data extraction with date-string range filters
- Cypher `ts_*()` functions with date-string arguments: `ts_sum(f.oil, '2020')`, `ts_avg(f.oil, '2020-2', '2020-6')`, etc.
- Query precision validation: errors when query detail exceeds data resolution (e.g. `'2020-2-15'` on month data)
- Channel units (e.g. "MSm3", "°C") and bin type ("total", "mean", "sample") metadata
- Timeseries data persisted as a separate section in `.kgl` files (backward compatible)
- `agent_describe()` includes timeseries metadata, resolution, units, and function reference
- Cypher `range(start, end [, step])` function — generates integer lists for use with `UNWIND`

## [0.5.51] - 2026-02-21

### Added

- Fluent API: `filter()` now supports `regex` (or `=~`) operator for pattern matching, e.g. `filter({'name': {'regex': '^A.*'}})`
- Fluent API: `filter()` now supports negated operators: `not_contains`, `not_starts_with`, `not_ends_with`, `not_in`, `not_regex`
- Fluent API: `filter_any()` method for OR logic — keeps nodes matching any of the provided condition sets
- Fluent API: `offset(n)` method for pagination — combine with `max_nodes()` for page-based queries
- Fluent API: `has_connection(type, direction)` method — filter nodes by edge existence without changing the selection target
- Fluent API: `count(group_by='prop')` and `statistics('prop', group_by='prop')` — group by arbitrary property instead of parent hierarchy

## [0.5.50] - 2026-02-21

### Added

- Shapely/geopandas integration for spatial methods — `intersects_geometry()` and `wkt_centroid()` now accept shapely geometry objects as input in addition to WKT strings
- `as_shapely=True` parameter on `get_centroid()`, `get_bounds()`, and `wkt_centroid()` to return shapely geometry objects instead of dicts
- `ResultView.to_gdf()` — converts lazy results to a geopandas GeoDataFrame, parsing a WKT column into shapely geometries with optional CRS
- Spatial type system via `column_types` in `add_nodes()` — declare `location.lat`/`location.lon`, `geometry`, `point.<name>.lat`/`.lon`, and `shape.<name>` types for auto-resolution in Cypher and fluent API methods
- `set_spatial()` / `get_spatial()` for retroactive spatial configuration
- Cypher `distance(a, b)` now auto-resolves via spatial config (location preferred, geometry centroid fallback)
- Virtual spatial properties in Cypher: `n.location` → Point, `n.geometry` → WKT, `n.<point_name>` → Point, `n.<shape_name>` → WKT
- Spatial methods (`within_bounds`, `near_point_km`, `get_bounds`, `get_centroid`, etc.) auto-resolve field names from spatial config when not explicitly provided
- Node-aware spatial Cypher functions: `contains(a, b)`, `intersects(a, b)`, `centroid(n)`, `area(n)`, `perimeter(n)` — auto-resolve geometry via spatial config, also accept WKT strings
- Geometry-aware `distance()` — `distance(a.geometry, b.geometry)` returns 0 if touching; `distance(point(...), n.geometry)` returns 0 if inside, closest boundary distance otherwise

### Removed

- Cypher functions `wkt_contains()`, `wkt_intersects()`, `wkt_centroid()` — replaced by node-aware `contains()`, `intersects()`, `centroid()` which also accept raw WKT strings

### Fixed

- Betweenness centrality now uses undirected BFS — previously only traversed outgoing edges, causing nodes bridging communities via incoming edges to get zero scores

### Performance

- `RETURN ... ORDER BY expr LIMIT k` fused into single-pass top-k heap — O(n log k) instead of O(n log n) sort + O(n) full projection. **5.4x speedup** on `distance()` ORDER BY LIMIT queries (1M pairs: 2627ms → 486ms)
- `WHERE contains(a, b)` fast path (`ContainsFilterSpec`) — extracts contains() patterns and evaluates directly from spatial cache, bypassing expression evaluator chain
- Spatial Cypher functions 6-8x faster for contains/intersects via per-node spatial cache + bounding box pre-filter:
  - Per-node cache (`NodeSpatialData`): resolves each node's spatial data once per query, cached for all cross-product rows (N×M → N+M lookups)
  - Bounding box pre-filter: computes `geo::Rect` alongside cached geometry; rejects non-overlapping pairs in O(1) before expensive polygon tests
  - `resolve_spatial()` skips redundant expression evaluation for Variable/PropertyAccess — goes directly to cached node data
- Spatial resolution uses WKT geometry cache for centroid fallback path — previously re-parsed WKT on every row
- `intersects()` and `centroid()` avoid deep-cloning `Arc<Geometry>` — use references directly
- `geometry_contains_geometry()` uses `geo::Contains` trait instead of point-by-point boundary check

## [0.5.49] - 2026-02-20

### Added

- Python type stub (`.pyi`) files now included in code graph — enables graph coverage of stub-only packages, compiled extensions, and authoritative type contracts

### Fixed

- Cypher parser now accepts reserved words (e.g. `optional`, `match`, `type`) as alias names after `AS` — previously failed with "Expected alias name after AS"
- Betweenness centrality `sample_size` now uses stride-based sampling across the full node range — previously sampled only the first k nodes, which could be non-participating node types (Module/Class) yielding all-zero scores

## [0.5.46] - 2026-02-20

### Fixed

- Decorator property stored as JSON array instead of comma-separated string — fixes fragmentation of decorators with comma-containing arguments (e.g. `@functools.wraps(func, assigned=(...))`)
- `is_test`, `is_async`, `is_method` boolean properties now explicitly `false` on non-matching entities instead of `null` — enables `WHERE f.is_test = false` queries
- Dynamic project versions (setuptools-scm etc.) now stored as `"dynamic"` instead of `null` on the Project node
- CALLS edges now scope-aware — calls inside nested functions, lambdas, and closures are no longer attributed to the enclosing function (fixes over-counted fan-out in all 7 language parsers)
- `collect(x)[0..N]`, `count(x) + 1` and other aggregate-wrapping expressions in RETURN now work — previously errored with "Aggregate function cannot be used outside of RETURN/WITH"
- `size(collect(...))` and other non-aggregate functions wrapping aggregates now evaluate correctly — previously silently returned `null` because the expression was misclassified as non-aggregate

## [0.5.43] - 2026-02-20

### Added

- List slicing in Cypher: `expr[start..end]`, `expr[..end]`, `expr[start..]` — works on `collect()` results and list literals, supports negative indices

### Fixed

- `size()` and `length()` functions on lists now return element count instead of JSON string length — e.g. `size(collect(n.name))` returns 5 instead of 29
- Duplicate nodes when test directory overlaps with source root (e.g. `root/tests/` inside `root/`) — test roots already covered by a parent source root are now skipped, with `is_test` flags applied to the existing entities instead
- Duplicate Dependency ID collision when same package appears in multiple optional groups — IDs now include the group name (e.g. `matplotlib::viz`)

## [0.5.42] - 2026-02-19

### Added

- `connection_types` parameter for `louvain` and `label_propagation` procedures — filter edges by type, matching the existing support in centrality algorithms

### Fixed

- `CALL pagerank({connection_types: ['CALLS']})` list literal syntax now works correctly — was silently serialized as JSON string causing zero edge matches and uniform scores
- Document list comprehension patterns as unsupported in Cypher reference

## [0.5.41] - 2026-02-19

### Added

- Cypher string functions: `split(str, delim)`, `replace(str, search, repl)`, `substring(str, start [, len])`, `left(str, n)`, `right(str, n)`, `trim(str)`, `ltrim(str)`, `rtrim(str)`, `reverse(str)`

### Fixed

- Duplicate File nodes when source and test roots overlap in code_tree (e.g. `xarray/` source root containing `xarray/tests/` + separate test root)
- Empty `Module.path` properties for declared submodules in code_tree — now resolved from parsed files or inferred from parent directory
- Boolean properties (`is_test`, `is_abstract`, `is_async`, etc.) stored as string `'True'` instead of actual booleans — improved pandas `object` dtype detection to recognize boolean-only columns

## [0.5.39] - 2026-02-19

### Added

- `read_only(True/False)` method to disable Cypher mutations (CREATE, SET, DELETE, REMOVE, MERGE). When enabled, `agent_describe()` omits mutation documentation, simplifying the agent interface for read-only use cases

## [0.5.38] - 2026-02-19

### Added

- Cypher `CALL procedure({params}) YIELD columns` for graph algorithms: pagerank, betweenness, degree, closeness, louvain, label_propagation, connected_components. YIELD `node` is a node binding enabling `node.title`, `node.type` etc. in downstream WHERE/RETURN/ORDER BY clauses
- Inline pattern predicates in WHERE clauses — `WHERE (a)-[:REL]->(b)` and `WHERE NOT (a)-[:REL]->(b)` now work as shorthand for `EXISTS { ... }`, matching standard Cypher behavior
- `CALL list_procedures() YIELD name, description, yield_columns` — introspection procedure listing all available graph algorithm procedures with their parameters and descriptions

### Changed

- `build()` now includes test directories by default (`include_tests=True`)
- CALL procedure error message now hints at the correct map syntax when keyword arguments are used instead of `{key: value}` maps

### Fixed

- CALLS edge resolution in code_tree now uses tiered scope-aware matching (same owner > same file > same language > global) instead of flat bare-name lookup — eliminates false cross-class and cross-language edges
- Rust parser now detects test files at the File level (`_test.rs`, `test_*`, `tests/`, `benches/` conventions) — previously only function-level `#[test]` attributes were detected, leaving File nodes untagged

## [0.5.36] - 2026-02-18

### Changed

- Split `mod.rs` (6,742 LOC) into 5 thematic `#[pymethods]` files: algorithms, export, indexes, spatial, vector — mod.rs reduced to 4,005 LOC
- Enabled PyO3 `multiple-pymethods` feature for multi-file `#[pymethods]` blocks
- Documented transaction isolation semantics (snapshot isolation, last-writer-wins)

### Fixed

- `[n IN nodes(p) | n.name]` now correctly extracts node properties in list comprehensions over path functions — previously returned serialized JSON fragments instead of property values
- `parse_list_value` is now brace-aware — splits at top-level commas only, preserving JSON objects and nested structures
- `EXISTS { MATCH (pattern) }` syntax now accepted — the optional `MATCH` keyword inside EXISTS braces is silently skipped, matching standard Cypher behavior

## [0.5.35] - 2026-02-18

### Added

- CALLS edges now carry `call_lines` and `call_count` properties — line numbers where each call occurs in the caller function
- Comment annotation extraction (TODO/FIXME/HACK/NOTE/etc.) for all non-Rust parsers (Python, TypeScript, JavaScript, Java, Go, C, C++, C#)
- Test file detection (`is_test`) for all parsers based on language naming conventions
- Generic/type parameter extraction for Go 1.18+ and Python 3.12+ (PEP 695) parsers

## [0.5.34] - 2026-02-18

### Added

- `toc(file_path)` method: get a table of contents for any source file — all code entities sorted by line number with a type summary
- `find()` now accepts `match_type` parameter: `"exact"` (default), `"contains"` (case-insensitive substring), `"starts_with"` (case-insensitive prefix)
- `file_toc` MCP tool in `examples/mcp_server.py` for file-level exploration
- `find_entity` MCP tool now supports `match_type` parameter
- Qualified name format documented in `agent_describe()` output (Rust: `crate::module::Type::method`, Python: `package.module.Class.method`)
- Block doc comment support (`/** */`) in Rust parser — previously only `///` line comments were captured
- `call_trace` MCP tool in `examples/mcp_server.py` for tracing function call chains (outgoing/incoming, configurable depth)
- Call trace Cypher pattern documented in `agent_describe()` output
- CHANGELOG.md, CONTRIBUTING.md, and CLAUDE.md for project governance

### Changed

- Doc comments added to all critical Rust structs (`KnowledgeGraph`, `DirGraph`, `CypherExecutor`, `PatternExecutor`, `CypherParser`, and 15+ supporting types)
- Rust parser now captures all `use` declarations, not just `crate::` prefixed imports
- MCP tool descriptions improved with workflow guidance (`graph_overview` says "ALWAYS call this first", `cypher_query` mentions label-optional MATCH, etc.)
- GitHub Release workflow now uses CHANGELOG.md content instead of auto-generated notes

## [0.5.31] - 2025-05-15

### Added

- `find(name, node_type=None)` method: search code entities by name across all types
- `source(name)` method: resolve entity names to file paths and line ranges (supports single string or list)
- `context(name, hops=None)` method: get full neighborhood of a code entity grouped by relationship type
- `find_entity`, `read_source`, `entity_context` MCP tools in `examples/mcp_server.py`
- Label-optional MATCH documented in `agent_describe()` — `MATCH (n {name: 'x'})` searches all node types

### Changed

- Code entity helpers (`find`, `context`) moved from Python (`kglite/code_tree/helpers.py`) to native Rust methods for performance
- `agent_describe()` now conditionally shows code entity methods and notes when code entities are present in the graph

### Removed

- `kglite/code_tree/helpers.py` — replaced by native Rust methods on `KnowledgeGraph`

## [0.5.28] - 2025-05-10

### Added

- Manifest-based building: `build(".")` auto-detects `pyproject.toml` / `Cargo.toml` and reads project metadata (name, version, dependencies)
- `Project` and `Dependency` node types with `DEPENDS_ON` and `HAS_SOURCE` edges
- `USES_TYPE` edges: Function → type references in signatures
- `EXPOSES` edges: FFI boundary tracking (PyO3 modules → exposed items)

### Fixed

- Various code tree parser fixes for Rust trait implementations and method resolution

## [0.5.22] - 2025-04-28

### Added

- `kglite.code_tree` module: parse multi-language codebases into knowledge graphs using tree-sitter
- Supported languages: Rust, Python, TypeScript, JavaScript, Go, Java, C++, C#
- Node types: File, Module, Function, Struct, Class, Enum, Trait, Protocol, Interface, Constant
- Edge types: DEFINES, CALLS, HAS_METHOD, HAS_SUBMODULE, IMPLEMENTS, EXTENDS, IMPORTS
- Embedding export support

---

*For versions prior to 0.5.22, see [GitHub Releases](https://github.com/kkollsga/kglite/releases).*

[0.5.35]: https://github.com/kkollsga/kglite/compare/v0.5.34...v0.5.35
[0.5.34]: https://github.com/kkollsga/kglite/compare/v0.5.31...v0.5.34
[0.5.31]: https://github.com/kkollsga/kglite/compare/v0.5.28...v0.5.31
[0.5.28]: https://github.com/kkollsga/kglite/compare/v0.5.22...v0.5.28
[0.5.22]: https://github.com/kkollsga/kglite/releases/tag/v0.5.22
