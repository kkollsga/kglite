# Embedding kglite in a Rust binary

This document is written for **Rust embedders**: anyone who wants
to use kglite's graph engine directly from a Rust binary without
the Python wheel in their build. If you're a Python user
(`pip install kglite`), you don't need to read this — `import
kglite` already wraps everything for you.

## The polars-style split

After Phase G (2026-05-24) kglite follows the same architectural
pattern as polars / pydantic-core / many published pyo3 projects:

| Crate | Purpose | Has PyO3? |
|---|---|---|
| `kglite-core` (`crates/kglite/`) | Pure-Rust engine. Publishable on crates.io. | **No** |
| `kglite-py` (workspace root, will move to `crates/kglite-py/` in G.4) | PyO3 wrapper. Built by maturin into the wheel. | Yes |
| `kglite-bolt-server` (`crates/kglite-bolt-server/`) | Bolt v5.x protocol binary. Wraps kglite-core. | No |
| `kglite-mcp-server` (`crates/kglite-mcp-server/`) | MCP protocol binary. Wraps kglite-core (presently via root crate; cleanup pending). | No |

The end-state design that any future binding (Go via cgo,
TypeScript via napi, JVM via JNI) follows: a sibling crate that
depends on `kglite-core` and adapts its API to the target
language's idioms. No changes to the core are required.

## Quick start

Add `kglite-core` (eventually published as `kglite`) to your `Cargo.toml`:

```toml
[dependencies]
# Pre-publish: path dependency from within the workspace.
kglite-core = { path = "../kglite/crates/kglite" }
# Post-publish: crates.io coordinate.
# kglite = "0.11"
```

Then load a `.kgl` file written by any kglite binding and query it:

```rust
use kglite_core::api::{load_file, session, Value};
use std::collections::HashMap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load a graph file written by ANY kglite binding —
    // Python's `kg.save("graph.kgl")`, the bolt-server's
    // `CALL db.checkpoint("graph.kgl")`, etc. The on-disk
    // .kgl format is the portable cross-binding contract.
    let graph = load_file("graph.kgl")?;

    // Run a Cypher query through the canonical pipeline.
    // Same path Python / Bolt / MCP all flow through (Phase E).
    let params = HashMap::new();
    let opts = session::ExecuteOptions {
        params: &params,
        deadline: None,
        max_rows: None,
        lazy_eligible: false,
        disabled_passes: None,
        embedder: None,
    };
    let outcome = session::execute_read(
        &graph,
        "MATCH (n:Person) RETURN n.name LIMIT 10",
        &opts,
    )?;

    for row in &outcome.result.rows {
        if let Some(Value::String(name)) = row.first() {
            println!("{}", name);
        }
    }
    Ok(())
}
```

Verify the build has zero pyo3:

```bash
cargo tree -p your-crate | grep pyo3   # → (empty)
```

See `crates/kglite/examples/embedded_*.rs` for runnable
end-to-end examples (`embedded_basic` reads a `.kgl`,
`embedded_session` demonstrates OCC transactions,
`embedded_blueprint` builds a graph from a Rust source tree
via `build_code_tree`).

## The stable API surface

`kglite_core::api::*` is the curated surface that gets semver
guarantees. Everything else (`kglite_core::graph::*`,
`kglite_core::datatypes::*`, etc.) is an implementation detail
that may move between minor releases.

### Engine types

```rust
use kglite_core::api::{DirGraph, Value, KgError, KgErrorCode};
use kglite_core::api::{NodeValue, PathValue, RelValue};
use kglite_core::api::Embedder;
```

- **`DirGraph`** — the in-memory graph. Built from blueprint,
  loaded from a `.kgl`, or constructed via the code_tree
  builder. Owned by your binding's "graph handle" type.
- **`Value`** — every value a Cypher query can return. Variants
  include scalars (`Int64`, `Float64`, `String`, `Bool`,
  `NaiveDate`), compound (`List`, `Map`), and graph-specific
  (`Node`, `Relationship`, `Path`).
- **`KgError`** — typed error enum (16 variants) every engine
  function can return. Map to your binding's error idiom at the
  boundary.
- **`Embedder`** trait — pluggable text-embedding backend. Bind
  via `kglite_core::api::FastEmbedAdapter` (with the
  `fastembed` feature) or implement your own.

### Cypher pipeline

```rust
use kglite_core::api::cypher::{parse_cypher, CypherExecutor, validate_schema};
use kglite_core::api::cypher::{is_mutation_query, generate_explain_result};
use kglite_core::api::cypher::{mark_lazy_eligibility, rewrite_text_score, planner};
```

Use these if you're building a custom Cypher pipeline (e.g. a
custom GraphQL adapter that compiles to Cypher). For the canonical
pipeline, **use `session` instead**.

### Session (canonical query + transaction surface)

```rust
use kglite_core::api::session::{Session, Transaction, CommitOutcome};
use kglite_core::api::session::{ExecuteOptions, execute_read, execute_mut};
```

This is the "single source of truth" added in Phase E. All
bindings flow through it. Cypher pipeline orchestration +
snapshot/working CoW + OCC live here exactly once.

- `execute_read(&dir, query, &opts)` — run a read query against
  `&DirGraph`.
- `execute_mut(&mut dir, query, &opts)` — run a mutation
  against `&mut DirGraph`.
- `Session::new(dir)` + `session.begin()` / `session.commit(tx,
  true)` — the snapshot/working CoW transaction model. OCC is
  opt-in per commit; pass `true` for production semantics.
- `CommitOutcome::{NoWritesNoOp, Committed { new_version },
  ConflictDetected { current_version, base_version }}` — what
  your binding maps to its consumer-facing error type.

See `docs/explanation/session.md` for the full session
abstraction guide.

### Dataset loaders (feature-gated)

```toml
kglite-core = { features = ["sec", "sodir", "wikidata"] }
```

```rust
use kglite_core::datasets::sec::{SecClient, fetch_quarterly_master_idx};
use kglite_core::datasets::sodir::{ArcGISClient, fetch_all};
use kglite_core::datasets::wikidata::{ensure_dump};
```

Each loader is opt-in via its Cargo feature so you only pay for
what you use. Polars-io pattern.

### `KnowledgeGraph` is NOT in the core

The Python-facing `KnowledgeGraph` struct (with its `selection`,
`reports`, `temporal_context`, `embedder`, etc. state) lives in
the `kglite-py` wrapper crate because it's binding-ergonomic
state — the kind of thing each binding wants to model in its own
language's idioms. Embedders should:

- Hold a `DirGraph` (or `Arc<DirGraph>` for cheap clones)
  directly.
- Hold an optional `Arc<dyn Embedder>` if they need text_score.
- Bundle their own per-binding ergonomics (selection
  history, default timeouts, format conversion).

Your binding's "graph handle" type is a wrapper around these
two values + whatever your language wants on top. The bolt-server
crate is a working example — it wraps `Session` (which owns the
`Arc<DirGraph>`) and adds Bolt protocol state.

## The `.kgl` file format is portable

A `.kgl` written by any kglite binding loads cleanly in any
other:

- Python `kg.save("graph.kgl")` → Bolt server reads via
  `kglite_core::api::load_file(path)`
- Rust embedder `kglite_core::api::save_graph(&mut arc, path)`
  → Python loads via `kglite.load("graph.kgl")`
- Future Go binding writes → TypeScript binding reads, etc.

The format is versioned (`load_v3`, `load_v4`, …). Format bumps
are coordinated with kglite's minor release cycle and tracked
via `tests/test_phase4_parity.py::GOLDEN_V3_DIGEST` etc. (see
CLAUDE.md → "Captured-constant refresh at release time").

The format does NOT bundle binding-ergonomic state (Python's
selection cache, default timeouts, etc.). Each binding sets
those fresh on load.

## Wrapping kglite-core in a new language

The path for a new language binding (Go, TypeScript, JVM) is:

1. **Create a new sibling crate** — `crates/kglite-go/`, or
   wherever your binding's natural home is.
2. **Depend on kglite-core** — `kglite-core = { path =
   "../kglite", features = ["sec", "sodir"] }` (opt in to the
   datasets you need).
3. **Author your bridge** — cgo / napi / JNI handles that
   marshal between your language's types and `kglite_core::api::*`.
4. **Wrap the binding-ergonomic state** in your language's
   idiomatic style. (For Go: a `Graph` struct holding `*C.DirGraph`
   + a metrics/logger; for TS: a `Graph` class wrapping
   `napi::Reference` + a Promise-returning API.)

The hard part — the Cypher pipeline, the CoW transaction model,
the OCC commit — is solved once, in `kglite-core::api::session`.
Each binding only owns the marshalling layer.

### cgo sketch (Go)

```go
package kglite

/*
#cgo LDFLAGS: -lkglite_c
#include "kglite.h"
*/
import "C"
import "runtime"

type Graph struct {
    h *C.KgliteDirGraph
}

func Load(path string) (*Graph, error) {
    cPath := C.CString(path)
    defer C.free(unsafe.Pointer(cPath))
    h := C.kglite_load_file(cPath)
    if h == nil {
        return nil, errors.New("load failed")
    }
    g := &Graph{h: h}
    runtime.SetFinalizer(g, func(g *Graph) { C.kglite_dir_graph_drop(g.h) })
    return g, nil
}

func (g *Graph) Cypher(query string) (*ResultSet, error) {
    cQuery := C.CString(query)
    defer C.free(unsafe.Pointer(cQuery))
    res := C.kglite_session_execute_read(g.h, cQuery)
    return resultSetFromC(res), nil
}
```

This would wrap a `kglite-c` crate (Phase H follow-up) that
exposes `kglite_core::api::session::execute_read` etc. through a
C ABI. The bridge is mechanical; no new core development.

### napi sketch (TypeScript)

```typescript
import { loadFile, executeRead } from './kglite-napi';

const graph = loadFile('graph.kgl');
const result = await executeRead(graph, 'MATCH (n) RETURN count(n)');
console.log(result.rows[0][0]); // node count
```

Same pattern: a `kglite-napi` crate wraps `kglite-core` for the
napi runtime; the TS API is a thin handle + Promise wrapper.

### JNI sketch (Java / Kotlin)

```java
import io.kglite.Graph;

Graph g = Graph.load("graph.kgl");
ResultSet rs = g.cypher("MATCH (n) RETURN count(n)");
System.out.println(rs.getInt(0, 0));
```

Same pattern: a `kglite-jni` crate; Java API thin wrapper.

## What's stable vs internal

| Item | Stability |
|---|---|
| `kglite_core::api::*` | **Semver-stable** within a minor. Breaking changes are announced + version-bumped. |
| `kglite_core::datasets::{sec,sodir,wikidata}::*` (feature-gated) | Stable — these were standalone published crates pre-G.3a; their API surface is preserved post-merge. |
| `kglite_core::error::*` | Stable (KgError variants may grow but won't be removed without a major bump). |
| `kglite_core::graph::*` (raw module path) | **Internal**. Subject to reorganization. Always go through `api::*` re-exports. |
| `kglite_core::datatypes::*` (raw module path) | Internal — use `api::{Value, NodeValue, PathValue, RelValue}`. |
| Any `pub(crate)` item promoted to `pub` for visibility (Phase G.3a) | **Unstable** — these were opened up for the wrapper crate's needs. Subject to retraction once accessor methods are designed. |

If you depend on something outside `api::*`, you're on your own
for minor-version compatibility.

## See also

- `docs/explanation/session.md` — full session/transaction
  abstraction reference (Phase E).
- `docs/explanation/transactions.md` — Python-API-flavored
  transaction guide.
- `docs/explanation/bolt-server.md` — Bolt server operator guide
  (an example of a sibling-crate binding).
- `CYPHER.md` — Cypher language reference.
- `bolt_implementation.md` — Phase E + Phase G design docs.
