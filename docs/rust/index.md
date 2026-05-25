# Rust guide

Embed the kglite engine in a Rust binary — without the Python
wheel in your build. The `kglite` crate is a pure-Rust knowledge
graph engine (zero pyo3 in the dep tree); the Python wheel is a
separate PyO3 wrapper crate that maturin builds on top.

If you're using the Python wheel (`pip install kglite`), the
[Python guide](../python/index.md) is for you. This track is for
*Rust* embedders.

## Quick start

```toml
# Cargo.toml
[dependencies]
# Path dep within this workspace; post-publish:
#   kglite = "0.10"
kglite = { path = "../kglite/crates/kglite" }
```

```rust
use kglite::api::{load_file, session, Value};
use std::collections::HashMap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load any .kgl file — same format that Python's
    // `kg.save("graph.kgl")` writes.
    let graph = load_file("graph.kgl")?;

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

Verify your build has zero pyo3:

```bash
cargo tree -p your-crate | grep pyo3   # → (empty)
```

## Transactions

The `Session` / `Transaction` types wrap the snapshot/working CoW
+ optimistic concurrency control. Pattern: begin, mutate the
working copy, commit. On a concurrent-writer conflict, the second
commit returns `CommitOutcome::ConflictDetected` and the binding
surfaces it to its caller as a retryable error.

```rust
use kglite::api::session::{CommitOutcome, ExecuteOptions, Session};
use kglite::api::DirGraph;
use std::collections::HashMap;
use std::sync::Arc;

let session = Arc::new(Session::new(DirGraph::new()));
let params: HashMap<String, kglite::api::Value> = HashMap::new();
let opts = ExecuteOptions {
    params: &params,
    deadline: None,
    max_rows: None,
    lazy_eligible: false,
    disabled_passes: None,
    embedder: None,
};

let mut tx = session.begin();
kglite::api::session::execute_mut(
    tx.working_mut()?,
    "CREATE (:Person {id: 1, name: 'Alice'})",
    &opts,
)?;

match session.commit(tx, /* check_occ = */ true) {
    CommitOutcome::Committed { new_version } => {
        println!("committed at version {}", new_version);
    }
    CommitOutcome::ConflictDetected { current_version, base_version } => {
        // Retry: re-begin against a fresh snapshot.
        eprintln!("conflict: base={} current={}", base_version, current_version);
    }
    CommitOutcome::NoWritesNoOp => {} // nothing to do
}
```

See [embedding.md](embedding.md) for the full embedder guide and
[session.md](session.md) for the binding-implementer reference on
the session/transaction primitives.

## Examples

Three runnable examples ship with the crate:

```bash
cargo run -p kglite --example embedded_basic -- path/to/graph.kgl
cargo run -p kglite --example embedded_session
cargo run -p kglite --example embedded_blueprint
```

- `embedded_basic.rs` — load a `.kgl`, run a Cypher query.
  Smallest possible embedder.
- `embedded_session.rs` — two concurrent transactions; OCC catches
  the conflict.
- `embedded_blueprint.rs` — parse the kglite source tree itself
  via `code_tree::build_code_tree`, then query the resulting
  graph.

All three are pyo3-free; `cargo tree -p kglite --example
embedded_basic | grep pyo3` returns empty.

## Where to go next

- **[Embedding kglite](embedding.md)** — full embedder guide:
  workspace layout, the `kglite::api::*` surface tour, the
  `.kgl` portability story, sketches for cgo / napi / JNI
  wrappers if you're building a binding in another language.
- **[Session abstraction](session.md)** — binding-implementer
  reference for the canonical Cypher pipeline + CoW transaction
  model.
- **[Implementing a binding](implementing-a-binding.md)** —
  deep-dive companion to `embedding.md` for anyone publishing a
  new-language binding: bridge-layer choice, full `KgErrorCode`
  mapping table, Embedder trait walkthrough, dataset-wrapping
  patterns, binding-side cookbook. Includes cgo / napi / JNI
  worked examples calling the shipped C ABI.
- **[C ABI (`kglite-c`)](c-abi.md)** — the design conventions for
  the C ABI crate that non-Rust bindings (Go, JS, JVM, .NET, …)
  consume. Naming rules, opaque-handle pattern, errno-style
  errors, JSON-at-boundary, sync-only ABI, versioning.
- **[API reference](api-reference.md)** — manifest of the stable
  `kglite::api::*` items + semver guarantees. Full per-symbol
  docs (post-publish) at [docs.rs/kglite](https://docs.rs/kglite).
- **[Cypher reference](../reference/cypher-reference.md)** —
  the Cypher subset kglite supports. Same syntax across all
  bindings.

```{toctree}
:hidden:

embedding
session
implementing-a-binding
c-abi
api-reference
```
