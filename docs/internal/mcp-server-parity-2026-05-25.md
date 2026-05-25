# KGLite MCP Server Feature-Parity Audit — 2026-05-25

**Purpose:** Phase 4 of the "prepare kglite for future-language
wrappers" work-stream. Per the recalibrated framing
(`docs/internal/consider-for-future.md`), this is an **audit, not a
consolidation**. Both MCP servers continue to ship.

KGLite has two MCP servers:

- **Python** at `kglite/mcp_server/` — ships in the `pip install
  kglite[mcp]` wheel, runs as the `kglite-mcp-server` console
  script. The wheel's canonical MCP host for Jupyter / Python users.
- **Rust** at `crates/kglite-mcp-server/` — ships as the
  `cargo install kglite-mcp-server` binary, pure-Rust (zero PyO3 in
  the dep tree). The canonical MCP host for Python-free deployments.

Goal of this audit: produce a feature matrix that surfaces every
meaningful divergence, classify each as "should converge", "accept
as design intent", or "bug", and write the audience-map so future
maintenance has a clear ranking for which server is the right
target for any given change.

---

## Section 1 — Source inventory

### Python (`kglite/mcp_server/`) — 3548 LOC across 15 modules

| File | LOC | Purpose |
|---|---|---|
| `server.py` | 1280 | Main entry point, CLI parser, MCP server assembly, tool registration |
| `tools.py` | 257 | Core kglite tools: `cypher_query`, `graph_overview`, `save_graph` |
| `skills_loader.py` | 305 | Skill discovery, frontmatter parsing, `applies_when` filtering, `prompts/*` handlers |
| `manifest.py` | 274 | YAML manifest parsing wrapper (delegates to `mcp-methods` Rust crate via `_mcp_internal`) |
| `claude_config.py` | 275 | Claude Desktop / Code config mutation helpers (add/edit/list MCP entries) |
| `code_source.py` | 205 | `read_code_source` tool: qualified-name → file path resolution |
| `bge_m3.py` | 204 | Direct ONNX runner for BGE-M3 embeddings (BAAI/bge-m3 only) |
| `csv_http.py` | 196 | CSV-over-HTTP server (aiohttp), OS-assigned port, CORS-enabled |
| `preprocessor.py` | 167 | Query-rewrite pipeline (user-supplied logic) |
| `workspace.py` | 159 | Workspace handle (thin wrapper over `mcp-methods::Workspace`) |
| `embedder.py` | 106 | FastEmbedAdapter wrapper (lazy-load / unload of ONNX models) |
| `cypher_tools.py` | 64 | YAML-declared Cypher-as-tool builder |
| `watch.py` | 38 | File watcher for code-tree rebuild |
| `__init__.py` / `__main__.py` | 18 | Module docstring, console-script stub |

### Rust (`crates/kglite-mcp-server/src/`) — 1974 LOC across 6 modules

| File | LOC | Purpose |
|---|---|---|
| `main.rs` | 627 | CLI parser (clap), mode dispatch, manifest load, embedder init, server boot |
| `tools.rs` | 606 | Core kglite tools: `cypher_query`, `graph_overview`, `save_graph` |
| `csv_http.rs` | 292 | CSV-over-HTTP listener (hyper 1.x + `TcpListener`), OS-assigned port |
| `code_source.rs` | 190 | `read_code_source` tool: qualified-name → file path resolution |
| `explore.rs` | 145 | `explore` tool: lexical FTS + 2-hop neighborhood → markdown report |
| `cypher_tools.rs` | 114 | YAML-declared Cypher-as-tool builder |

**Why Python is 1.8× larger.** Python re-implements pieces Rust
delegates to `mcp-methods` (the framework crate): a `skills_loader`
module, a `manifest` wrapper, a `claude_config` helper. The Rust
binary leans on `mcp-methods` for all of those, which is why its
own source can be smaller.

### CLI surface (identical between Python and Rust)

```
kglite-mcp-server [OPTIONS]
  --graph PATH              Load .kgl at boot (graph mode)
  --source-root PATH        Source-root mode (no graph)
  --workspace PATH          Workspace mode (GitHub clone-tracker)
  --watch PATH              Watch mode (rebuild code-tree on file changes)
  --mcp-config PATH         Explicit manifest path
  --name STR                Server name override
  --stale-after-days N      Workspace refresh interval (default 7)
  --trust-tools             Legacy flag (no-op since 0.9.20 / 0.10.1)
```

### Configuration sources (identical)

CLI flags → YAML manifest → `.env` file (walk-up search from mode
dir) → environment variables. Both use `mcp-methods 0.3+` for the
manifest schema; YAML blocks are: `extensions`, `tools`, `skills`,
`builtins`, `workspace`.

---

## Section 2 — Tool inventory

| Tool | Python | Rust | Notes |
|---|---|---|---|
| `cypher_query` | ✓ | ✓ | Identical schema; both return 15-row preview or CSV append |
| `graph_overview` | ✓ | ✓ | Identical: `types[]`, `connections` (bool or array), `cypher` (bool or array); both prepend `[kglite-mode]` banner |
| `read_code_source` | ✓ | ✓ | Identical: `qualified_name`, `node_type` (optional); Python uses `graph.source_location()`, Rust uses `KnowledgeGraph::source_location()` |
| `save_graph` | ✓ (opt-in) | ✓ (opt-in) | Both require `builtins.save_graph: true` in manifest; both persist via `kglite::api::save_graph` (Rust) or `kglite.save` (Python) |
| `ping` | ✓ | ✓ | Liveness probe; optional `message` echo |
| `read_source` | ✓ (framework) | ✓ (framework) | Generic file read with slicing + grep; via `mcp-methods` in both |
| `grep` | ✓ (framework) | ✓ (framework) | Ripgrep wrapper; via `mcp-methods` |
| `list_source` | ✓ (framework) | ✓ (framework) | Directory tree listing; via `mcp-methods` |
| `github_issues` | ✓ (if `GITHUB_TOKEN`) | ✓ (framework) | Issue/PR/discussion search + drill-down; via `mcp-methods` in both |
| `github_api` | ✓ (if `GITHUB_TOKEN`) | ✓ (framework) | Generic GitHub REST GET; via `mcp-methods` |
| `repo_management` | ✓ (workspace mode) | ✓ (workspace mode) | Clone + activate + list + update + delete |
| `set_root_dir` | ✓ (local-workspace mode) | ✓ (local-workspace mode) | Rebind active directory; only in `workspace.kind: local` manifests |
| **`explore`** | **✗** | **✓** | **Lexical FTS + 2-hop neighborhood traversal → markdown report. Rust-native (kglite::api::explore_markdown). The one real tool gap.** |

12 of 13 tools have identical JSON schemas between the two servers.
The one divergence is `explore` (Rust-only).

---

## Section 3 — Operating modes

| Mode | Python | Rust | CLI | Use case |
|---|---|---|---|---|
| Graph | ✓ | ✓ | `--graph /path/to.kgl` | Open a pre-built graph; expose query + overview + code lookup tools |
| Workspace | ✓ | ✓ | `--workspace /path/to/repos` | Multi-repo clone + activate; `repo_management` tool active |
| Watch | ✓ | ✓ | `--watch /path/to/source` | Single source directory; rebuild code-tree on file changes |
| Source-root | ✓ | ✓ | `--source-root /path` | Generic file tree (no graph); only `read_source` / `grep` / `list_source` |
| Local-workspace | ✓ | ✓ | YAML `workspace.kind: local; root: PATH` | Fixed directory binding; `set_root_dir` tool active |
| Bare | ✓ | ✓ | (no mode flag) | Framework tools + manifest-declared Cypher tools only |

All six modes are identical between Python and Rust. Mode dispatch
is in `server.py::_dispatch_mode` (Python) and `main.rs::main`
(Rust); both branch on the same CLI flag → mode mapping.

---

## Section 4 — Skills

### Bundled skills (identical 5 in both)

- `cypher_query.md`
- `graph_overview.md`
- `save_graph.md`
- `read_code_source.md`
- `explore.md` ⚠️ *(Python ships the skill description but lacks
  the underlying tool — the skill file exists in
  `kglite/mcp_server/skills/explore.md` for consistency with the
  Rust binary's skill set, but `applies_when: tool_registered:
  explore` filtering prevents it from being injected when the tool
  isn't present.)*

### Loading mechanism

- **Python:** `skills_loader.py` (305 LOC) implements parsing +
  `applies_when` filtering + `prompts/list` + `prompts/get`
  handlers for the lowlevel MCP Server shape. Frontmatter
  (`---`...`---`) carries `applies_when` predicates:
  `graph_has_node_type`, `graph_has_property`, `tool_registered`,
  `extension_enabled`.
- **Rust:** Delegates to `mcp-methods::SkillRegistry` (a Rust
  crate that already parses bundled + manifest-declared skills
  for the FastMCP shape).

### Auto-inject

Both servers inject the skill body into the tool description when
`auto_inject_hint=True` (default). Python enforces a 16 KB hard
ceiling per skill (`skills_loader.py:_MAX_SKILL_LEN`); Rust likely
has the same via `mcp-methods`.

### Opt-in / opt-out

Manifest `skills: true` (or a list of paths) enables the loader.
Both servers default to disabled in bare mode, enabled in graph /
workspace / watch / source-root modes.

---

## Section 5 — Embedder support

| Aspect | Python | Rust |
|---|---|---|
| Backend | `fastembed` (Python pkg) | `fastembed-rs` (Cargo) |
| Bundled with default install? | ✓ (in `[mcp]` extras) | ✗ (requires `--features fastembed`) |
| Models supported | 8 (BAAI/bge-m3, bge-{small,base,large}, all-MiniLM, multilingual-e5-{large,base}) | Same 8 |
| Direct ONNX runner | ✓ (`bge_m3.py` for BGE-M3 specifically) | — |
| Lazy load | ✓ (load on first `embed()`, unload after cooldown) | ✗ (loaded at boot, stays resident) |
| Cooldown | Configurable, default 900s | — |
| `text_score()` availability | Always | Only when binary compiled with `fastembed` feature |
| Error when missing | — | `"extensions.embedder.backend = \"fastembed\" requires this binary to be built with the \`fastembed\` feature enabled..."` (`main.rs:~581`) |
| Manifest schema | `extensions.embedder: { backend, model, cooldown }` | `extensions.embedder: { backend, model }` (no cooldown field) |

### Why this divergence exists

Rust's `ort-sys` (the ONNX runtime crate that `fastembed-rs`
depends on) has a flaky upstream binary download from
`parcel.pyke.io` that caused CI breakage in the 0.10.1 release
cycle (documented in `feedback_crates_io_publish_gotchas.md`).
Making `fastembed` optional ships a smaller binary (~30 MB vs
~150 MB) that builds reliably for users who don't need
`text_score()`.

---

## Section 6 — Multi-graph routing

**Neither server natively serves multiple named graphs from a
single process.** Both have a single active-graph slot:

- Python: `GraphState._active: ActiveGraph | None` (protected by
  `threading.Lock`)
- Rust: `GraphState.inner: Arc<RwLock<Option<ActiveGraph>>>`

Workspace mode allows runtime *swapping* of the active graph via
`repo_management('org/repo')` or `set_root_dir(path)`, but only one
graph is active at any moment.

The multi-graph pattern observed in this session's MCP server
instructions (`legal`, `open-source`, `sodir-prospect` as named
graphs in Claude Code) is **server multiplexing**, not within-
server multi-graph: each of those is a separate MCP server
instance registered in the Claude config, each with its own
process and graph.

---

## Section 7 — Network protocols

| Protocol | Python | Rust | Notes |
|---|---|---|---|
| stdio | ✓ | ✓ | Primary MCP transport |
| CSV HTTP | ✓ (aiohttp) | ✓ (hyper 1.x + `TcpListener`) | Optional via `extensions.csv_http_server` in manifest; both bind 127.0.0.1, OS-assigned port (default 0); CORS-enabled; flat-file GET-only |
| TCP graph port | ✗ | ✗ | Not in scope (use Bolt server) |
| SSE | ✗ | ✗ | No streaming responses |
| WebSocket | ✗ | ✗ | Not supported |

CSV HTTP config schema is identical:

```yaml
extensions:
  csv_http_server: true              # defaults: port=0, dir=temp/
  csv_http_server:
    port: 9000
    dir: my-csv-output/
    cors_origin: "https://my.app"
```

---

## Section 8 — Logging

| | Python | Rust |
|---|---|---|
| Backend | stdlib `logging` | `tracing` crate |
| Setup | `_setup_logging()` in `server.py:118` | `init_tracing()` in `main.rs:151` (delegates to `mcp_methods::server::init_tracing`) |
| Level | `INFO` (hardcoded) | From `RUST_LOG` env var |
| Format | `%(asctime)s %(name)s %(levelname)s %(message)s` | tracing-subscriber default + structured fields |
| Stream | `sys.stderr` | stderr by default; subscriber-configurable |
| Per-module loggers | `kglite.mcp_server.*` namespace | Tracing spans with key-value context |

Both write to stderr (stdout is reserved for MCP protocol traffic).
The Rust side gets structured-logging benefits; the Python side is
simpler / older but adequate for the use cases.

---

## Section 9 — Error handling

Both servers use the same pattern: **tool errors surface as plain-
text in the tool body, not as MCP-level errors.**

```python
# Python: tools.py:116-117
except Exception as e:
    return f"Cypher error: {e}"
```

```rust
// Rust: tools.rs:245
let pre_parsed = kglite::api::cypher::parse_cypher(query)
    .map_err(|e| e.to_string())?;
```

Cypher errors flow through verbatim. Mutation rejections are
pre-emptive (Rust: `tools.rs:246–250`). Both leave the MCP
protocol layer clean (every tool returns success; the error
message is the body text). This is a deliberate, aligned design
choice — keeping it that way.

---

## Section 10 — Feature matrix (executive summary)

| Feature | Python | Rust | Verdict |
|---|---|---|---|
| `cypher_query` | ✓ | ✓ | Aligned |
| `graph_overview` | ✓ | ✓ | Aligned |
| `read_code_source` | ✓ | ✓ | Aligned |
| `save_graph` (opt-in) | ✓ | ✓ | Aligned |
| `ping` | ✓ | ✓ | Aligned |
| `read_source` | ✓ | ✓ | Aligned |
| `grep` | ✓ | ✓ | Aligned |
| `list_source` | ✓ | ✓ | Aligned |
| `github_issues` | ✓ | ✓ | Aligned |
| `github_api` | ✓ | ✓ | Aligned |
| `repo_management` | ✓ | ✓ | Aligned |
| `set_root_dir` | ✓ | ✓ | Aligned |
| **`explore`** | **✗** | **✓** | **Should converge — port to Python (M)** |
| All 6 modes | ✓ | ✓ | Aligned |
| Skills (5 bundled) | ✓ | ✓ | Aligned |
| Skills auto-inject | ✓ | ✓ | Aligned |
| Manifest YAML | ✓ | ✓ | Aligned |
| CSV HTTP | ✓ | ✓ | Aligned |
| File watcher | ✓ | ✓ | Aligned |
| Embedder (fastembed) | ✓ default ON | ✓ default OFF | Accept divergence (design intent) |
| Embedder lazy-load | ✓ | ✗ | Should converge — Rust adds (M, blocked on `ort-sys` capability) |
| `text_score()` Cypher | ✓ | ✓ (if `--features fastembed`) | Conditional parity |
| Multi-graph (within process) | ✗ | ✗ | Neither — server multiplexing is the pattern |
| Logging backend | stdlib | tracing | Accept divergence (ecosystem idiom) |
| Error formatting | plain text in body | plain text in body | Aligned |

---

## Section 11 — Recommendations

### A. Should converge — port `explore` to Python (Medium)

**The one real tool gap.** Python users get less if they choose the
wheel-bundled server over the Rust binary for code-navigation work.

Effort estimate: 200-300 LOC in Python. The underlying logic is in
`kglite::api::explore_markdown` (in core), exposed via the wheel's
PyO3 surface. The Python tool wrapper is a ~80-line shim: parse
arguments, call `kg.explore(query, ...)`, return the markdown.
Plus registering it in `server.py::_register_tools`.

The skill file already exists at
`kglite/mcp_server/skills/explore.md` (currently gated by
`applies_when: tool_registered: explore` so it doesn't surface
today).

Tracked in `docs/internal/consider-for-future.md`.

### B. Should converge — Rust embedder lazy-load (Medium, conditional)

Python's embedder unloads ONNX after a configurable cooldown
(default 900s), freeing ~1 GB of resident memory for servers that
hit `text_score()` infrequently. Rust's embedder loads at boot and
stays resident.

**Blocked on `ort-sys` capability.** Lazy-load requires explicit
drop + reload of the ONNX session; verify `ort-sys`'s API surface
supports session lifecycle control before scoping. If it doesn't,
document the trade-off and stop.

Tracked in `docs/internal/consider-for-future.md`.

### C. Accept divergence — `fastembed` default OFF in Rust

Correct for the cargo-install audience: smaller binary, faster
build, no flaky ONNX runtime CDN download. Python's default ON is
correct for the wheel audience (users expect `text_score()` to
just work).

No action.

### D. Accept divergence — logging backends

`tracing` is the Rust ecosystem standard. Python's stdlib `logging`
is adequate for the use cases. Both write to stderr.

No action.

### E. Monitor — Python's `skills_loader.py` complexity

Python's 305-line skills loader exists because the Python server
runs against the lowlevel MCP Server (not FastMCP). If the
underlying Python MCP library shifts to support FastMCP-style
skills natively, this module could shrink significantly. Worth
re-evaluating when the Python MCP SDK evolves.

Tracked informally.

---

## Section 12 — Audience map

### Python MCP server — for the wheel

**Audience:** Jupyter users, Python-first teams, anyone who already
has Python installed and wants kglite as one of the wheel-bundled
tools they `pip install`.

**When to choose:**
- You're already running Python (Jupyter, FastAPI, conda) and want
  the MCP server alongside the kglite Python API in the same
  environment.
- You're building/customizing embedders in Python (the wheel's
  embedder API accepts Python classes).
- You need to ship to non-Rust environments (conda channels,
  managed Python services).

### Rust MCP server — for `cargo install`

**Audience:** DevOps, polyglot teams, containerized deployments,
teams that don't want a Python runtime in the loop.

**When to choose:**
- You want a self-contained binary (`cargo install kglite-mcp-server`,
  or `cargo binstall` for pre-built).
- You're deploying to Alpine / distroless containers where Python
  is not desired.
- You need the `explore` tool today (Python doesn't have it yet —
  see Recommendation A).
- You're integrating with Rust-based CI / deployment tooling and
  want one consistent ecosystem.

### When the choice doesn't matter

Both servers expose the same 12 tools (modulo `explore`) with
identical schemas, the same 6 modes, the same manifest YAML, the
same CSV HTTP transport, and roughly equivalent observability.
For a deployment that just needs "an MCP server that talks to a
`.kgl` file" — either works. Pick the one that fits your runtime.

---

## Appendix — Survey methodology

Phase 4 audit performed via an Explore agent against:

- `kglite/mcp_server/*.py` — 15 files, end-to-end on the small
  ones (under 200 LOC) + structural read on `server.py`
- `crates/kglite-mcp-server/src/*.rs` — 6 files
- Tool registration sites: grepped both codebases for tool
  decorators / `add_tool` calls
- CLI flags: read both `clap` and `argparse` definitions
- Configuration sources: grep for `mcp-methods` + manifest schema

Specific file:line citations are in the audit; this report is
ready to update via additions-only edits if any claim proves
inaccurate.

**End of audit.**
