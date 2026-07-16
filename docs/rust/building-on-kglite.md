# Building on kglite

The out-of-the-box playbook for a **producer** — a library that computes domain
data (SEC filings, a parsed codebase, an audio analysis, a distilled PDF) and
turns it into a queryable kglite graph. This page tells a third party how to
connect their library to kglite, what contract they're building against, and how
to test it.

If you're *querying* an existing graph rather than producing one, the
[Rust guide](index.md) and the [Python guide](../python/index.md) are your
starting points instead.

## The one-question chooser

There is a single question that picks your tier:

> **Does your build logic need to query the graph while it is being built?**

- **No** — the graph is a pure function of your source data → **P3
  (engine-free).** You compute nodes and edges and hand kglite a structured
  description; kglite builds the graph. This is the default, and where most
  producers land.
- **Yes** — a build pass has to read back what earlier passes wrote (resolving
  references, deduping against existing nodes, cross-linking) → **P1
  (embedded engine).** You link the `kglite` crate, build the graph natively,
  and hand it off as a `.kgl` file.

| | P3 — engine-free | P1 — embedded engine |
|---|---|---|
| kglite crate dependency | none | links `kglite` |
| Query graph mid-build? | no | yes |
| Handoff | `blueprint.json`+CSV / inline records / `add_nodes` | save `.kgl` → `load` |
| Version coupling | python `kglite>=X` (input-format floor) | crate pin + format-floor rule + pin hygiene |
| Producer language | any | Rust (until P2) |
| Wheel carries engine? | no | yes |
| Reference | kglite-datasets | codingest |

When unsure, start at P3. You only pay for P1 if you can name the build pass that
must query the half-built graph.

## The seam inventory — what you build against

Everything a producer depends on is one of five seams. Each has a defined
stability posture; kglite's CI locks against accidental drift on all of them.

| Seam | What it is | Stability |
|---|---|---|
| **Engine facade** — `kglite::api::*` | The curated Rust surface: `DirGraph`, `Value`, `session::*`, `io::{save_graph, load_file}`, error types, `code_entities`. | Exact-baseline-locked in CI (cargo-public-api, pinned nightly). Additive within a minor line; deliberate breaks ship on a MINOR bump with a migration guide. See the [API reference](api-reference.md). |
| **MCP server library** — `kglite-mcp-server` | `run`, `run_with_embedder_factory`, `run_with_code_tree_hooks`, `CodeTreeHooks`. The seam a domain MCP server builds on. | Public-API baseline + hook-semantics unit tests. Same MINOR-break posture as the engine facade. |
| **`.kgl` file format** | The persisted graph format that P1 handoff and all persistence use. | Versioned (`v3`, `v4`, …). Readers stay backward-compatible or refuse an old format with a clear rebuild message; a format bump lands with its decoder. |
| **Python top-level** — `kglite.*` | `kglite.load`, `kglite.from_blueprint`, `kglite.from_records`, `KnowledgeGraph` methods. The P3 entry points and the P1 handoff target. | Contract-tested + stubtest against `kglite/__init__.pyi`. |
| **C ABI** — `include/kglite.h` | The `extern "C"` surface for non-Rust bindings. | cbindgen header-drift check in CI; see the [C ABI guide](c-abi.md). |

One caveat CI cannot lock for you: **version pins across repos.** Your producer
pins `kglite` (and, for an MCP producer, `kglite-mcp-server`) to a minor line,
and a P1 producer must keep transitive pins — notably `rmcp` / `rmcp-macros` —
in lockstep with kglite's. Re-check these at every kglite bump; nothing
machine-enforces them from a single repo.

## P3 recipe — engine-free

You emit a structured description of your graph; kglite builds it. Three input
shapes, all public API:

**Blueprint + CSVs.** Describe the graph once in `blueprint.json` — node types,
primary keys, titles, properties, and connections (FK edges, timeseries) — with
each node type pointing at a CSV. Then:

```python
import kglite
g = kglite.from_blueprint("blueprint.json")
g.cypher_query("MATCH (c:Company) RETURN c.name LIMIT 5")
```

```json
{
  "nodes": {
    "Company": {
      "csv": "processed/company.csv",
      "pk": "cik",
      "title": "name",
      "properties": { "name": "string", "sic": "string" },
      "connections": {
        "fk_edges": { "IN_INDUSTRY": { "target": "SicCode", "fk": "sic" } }
      }
    },
    "SicCode": { "csv": "processed/sic.csv", "pk": "sic", "title": "description",
      "properties": { "description": "string" } }
  }
}
```

**Inline records** — `kglite.from_records(spec)` — carry nodes and connections
inline as JSON instead of pointing at CSVs; column types are inferred and array
values become native list properties. This is the no-CSV-on-disk / agent-authored
path.

**Imperative** — `KnowledgeGraph.add_nodes` / `add_connections` (and the `_bulk`
variants) build a graph node-by-node from Python.

Your library links **zero** kglite crate code; its only kglite tie is the wheel's
runtime floor, `kglite>=X`, chosen for the input format you emit. The living
template is **kglite-datasets** (`pip install kglite-datasets`): its SEC loader
computes a 34-CSV `processed/` layout, describes it with one `blueprint.json`,
and calls `from_blueprint` — engine-free end to end. Full blueprint semantics are
in the [blueprint guide](../python/guides/blueprints.md).

## P1 recipe — embedded engine

Link the crate, build a `DirGraph`, hand off through `.kgl`:

```toml
# Cargo.toml
[dependencies]
kglite = "0.14"   # a version whose reader understands the format you write
```

```rust
use kglite::api::io::save_graph;

let mut graph = build_my_graph()?;          // your builder; may query `graph` mid-build
save_graph(&mut graph, "out.kgl")           // → the handoff artifact
    .map_err(anyhow::Error::msg)?;
```

The Python handoff pattern (see codingest for the reference): run the pure-Rust
builder with the GIL released, `save_graph` to a `.kgl` (a temp file when the
caller gave no path, deleted once the load completes), then `py.import("kglite")`
and call its top-level `load(path)` — the returned object is a real
`kglite.KnowledgeGraph`, so every downstream kglite API works unchanged.

**The format-floor rule.** Your declared floor `kglite>=X` must name a version
whose *reader* understands the format your linked engine *writes*. If your crate
links an engine that writes `.kgl` v4, `kglite>=X` must be a version that reads
v4 — otherwise the handoff `load()` fails at runtime for exactly the users who
took the floor literally.

**Pin hygiene.** Re-check your kglite pin — and transitive pins that must move in
lockstep with it (`rmcp` / `rmcp-macros`) — at every kglite bump. There is a
measured cost to the round-trip P1 pays and P3 avoids: roughly 12% on
parse-heavy builds, up to ~50% on very fast builds (serialization is fixed-cost
against graph size, so it dominates a cheap build).

The reference is **codingest** (`cargo add codingest` / `pip install
codingest`) — its resolution passes query the half-built graph, which is exactly
why it is P1. See [embedding.md](embedding.md) and
[implementing-a-binding.md](implementing-a-binding.md) for the full embedder
surface.

## MCP recipe

A producer that wants an MCP server wraps
`kglite_mcp_server::run_with_code_tree_hooks`. The seam is *code-tree-named* but
*generically shaped* — a triple of hooks:

- **build**: `path → graph` (single build),
- **build_revs**: `path → graph` over multiple git revisions (the hook owns rev
  canonicalization), and
- **is_code_file**: the watch predicate — is a change to this path
  build-relevant?

```rust
use kglite_mcp_server::CodeTreeHooks;

fn main() -> anyhow::Result<()> {
    let hooks = CodeTreeHooks {
        build: Box::new(|dir, include_docs| my_builder::build(dir, include_docs)),
        build_revs: Box::new(|dir, revs, include_docs| {
            let revs = my_builder::dedup_revs(revs);
            let graph = my_builder::build_revs(dir, &revs, include_docs)?;
            Ok((graph, revs))
        }),
        is_code_file: Box::new(|p| my_builder::language_for_path(p).is_some()),
    };
    kglite_mcp_server::run_with_code_tree_hooks(std::env::args_os(), Some(hooks))
}
```

The server crate owns the entire tool surface, Cypher pipeline, `set_root_dir`
activation, and file watching; you inject only the builder. **Drop-in property:**
your server takes the *same flags* as the kglite MCP server — operators switch
the binary, not their config. codingest-mcp is the ~40-line reference `main`.

## Testing pattern

- **Golden-digest parity, frozen.** While a reference producer exists, freeze a
  golden `.kgl` digest of a fixture graph and assert it in CI. A build change
  that shifts the digest is either a bug or an intended graph-shape change that
  must re-bless the golden in the same commit. (kglite's own writer side is
  pinned by `test_phase4_parity.py::GOLDEN_V3_DIGEST`, refreshed per release;
  your producer freezes its own goldens the same way.)
- **Offline-first gates.** A producer's default test suite must run with no
  network — fetchers hit cached fixtures, not live registries. Gate any
  network-touching test behind an explicit marker so the common `make test` path
  stays hermetic.

## Domain math at build time

**Bake domain computation into node properties and edges at build time; keep the
Cypher layer generic.** Bucket mappings, edge kinds, segment structure, block
hierarchy — compute them in your producer and store them as graph data. Do not
push domain logic into kglite.

A helper graduates *into* kglite Cypher only when it is **domain-independent**
(sequence/array math, date helpers, graph algorithms, statistics) **and a second
domain wants it** — the same use-case test the
[boundary principle](boundary-principle.md) applies to lifts. There is no UDF
plugin mechanism: producers do not register custom Cypher functions. Compute at
build time, store, and let generic Cypher read it back.

## Where to go next

- **[API reference](api-reference.md)** — the `kglite::api::*` inventory and the
  stability policy that governs it.
- **[Embedding kglite](embedding.md)** / **[implementing a binding](implementing-a-binding.md)** — the full P1 embedder surface.
- **[Boundary principle](boundary-principle.md)** — why the engine stays generic
  and domain math lives in producers.
- **[C ABI](c-abi.md)** — the non-Rust producer surface.
</content>
