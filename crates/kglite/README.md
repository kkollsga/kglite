# kglite

[![crates.io](https://img.shields.io/crates/v/kglite)](https://crates.io/crates/kglite)
[![docs.rs](https://img.shields.io/docsrs/kglite)](https://docs.rs/kglite)
[![License: MIT](https://img.shields.io/crates/l/kglite)](https://github.com/kkollsga/kglite/blob/main/LICENSE)

**Pure-Rust knowledge graph engine** — Cypher pipeline,
snapshot/working CoW transactions, columnar / mmap / disk storage
backends, optional RDF / OKF format loaders. Pre-packaged domain
dataset loaders (SEC EDGAR, Sodir, Wikidata) live in the separate
kglite-datasets project. Zero PyO3 in the dependency tree; embed
directly from any Rust binary.

> Looking for the Python wheel? `pip install kglite` — the wheel
> is a separate PyO3 wrapper (`kglite-py`) built on top of this
> crate. See the [workspace README] for the Python story; this
> page is the crates.io-side documentation.

[workspace README]: https://github.com/kkollsga/kglite#kglite--knowledge-graph-for-python-built-for-llm-agents

## Quick start

```toml
[dependencies]
kglite = "0.10"
```

```rust
use kglite::api::{load_file, session, Value};
use std::collections::HashMap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load any .kgl file — same format the Python wheel writes.
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

Verify no Python dep leaked in:

```bash
cargo tree -p your-crate | grep pyo3   # → (empty)
```

## What's in it

| Surface | Purpose |
|---|---|
| `kglite::api::DirGraph` | The in-memory graph. Owned by your binding's graph handle. |
| `kglite::api::Value` | The Cypher value type — scalars, `List`, `Map`, `Node`, `Relationship`, `Path`. |
| `kglite::api::KgError`, `KgErrorCode` | Typed error enum (16 variants) for binding-friendly error mapping. |
| `kglite::api::session::Session` / `Transaction` | Snapshot/working CoW transaction model with optimistic concurrency control. |
| `kglite::api::session::execute_read` / `execute_mut` | The canonical Cypher pipeline — parse, validate, optimise, execute. |
| `kglite::api::cypher::*` | Lower-level pipeline primitives for building custom orchestrations. |
| `kglite::api::load_file` / `save_graph` | `.kgl` portable graph snapshots — copy, share, reload across bindings. |
| `kglite::api::compute_description` / `compute_schema` | Schema introspection: XML for LLM system prompts, structured types for programmatic use. |

## Transactions

The `Session` / `Transaction` types wrap the snapshot/working CoW
+ optimistic concurrency control. Pattern: begin, mutate, commit.
On a concurrent-writer conflict, the second commit returns
`CommitOutcome::ConflictDetected` and the binding surfaces it to
its caller as a retryable error.

```rust
use kglite::api::session::{CommitOutcome, ExecuteOptions, Session};
use kglite::api::DirGraph;
use std::collections::HashMap;
use std::sync::Arc;

let session = Arc::new(Session::new(DirGraph::new()));
let params: HashMap<String, kglite::api::Value> = HashMap::new();
let opts = ExecuteOptions {
    params: &params, deadline: None, max_rows: None,
    lazy_eligible: false, disabled_passes: None, embedder: None,
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
        eprintln!("conflict: base={} current={}", base_version, current_version);
    }
    CommitOutcome::NoWritesNoOp => {}
}
```

## Examples

Three runnable examples ship with the crate:

```bash
cargo run --example embedded_basic -- graph.kgl
cargo run --example embedded_session
cargo run --example embedded_blueprint
```

- `embedded_basic` — load + query. Smallest embedder.
- `embedded_session` — two concurrent transactions; OCC catches
  the conflict.
- `embedded_blueprint` — parse the kglite source tree itself,
  query the resulting graph.

## Feature flags

Polars-io style: opt in only to what you use.

| Feature | What it pulls in |
|---|---|
| `default` | The engine. No optional loaders. (Domain dataset loaders live in the kglite-datasets project; code-graph building in the codingest crate.) |
| `rdf` | RDF loader (Turtle / N-Triples / N-Quads / TriG via oxttl). |
| `okf` | Open Knowledge Format bundle loader (markdown + YAML frontmatter). |
| `fastembed` | Rust-native ONNX embedder for `text_score()` semantic search. |

```toml
[dependencies]
kglite = { version = "0.13", features = ["rdf", "okf"] }
```

## Documentation

- **[Rust quickstart](https://kglite.readthedocs.io/en/latest/rust/index.html)**
  — load + query + transaction examples.
- **[Embedding guide](https://kglite.readthedocs.io/en/latest/rust/embedding.html)**
  — workspace layout, the `kglite::api::*` surface tour, sketches
  for cgo / napi / JNI wrappers if you're building a binding in
  another language.
- **[Session abstraction](https://kglite.readthedocs.io/en/latest/rust/session.html)**
  — binding-implementer reference for the canonical Cypher pipeline
  + CoW transaction model.
- **[API manifest](https://kglite.readthedocs.io/en/latest/rust/api-reference.html)**
  — curated inventory of `kglite::api::*` items + semver rules.
- **Per-symbol docs at [docs.rs/kglite](https://docs.rs/kglite).**

The full kglite docs site at
[kglite.readthedocs.io](https://kglite.readthedocs.io) has more
on the Cypher subset, design rationale, and the protocol-server
binaries that ship alongside this crate.

## Semver

`kglite::api::*` items get semver guarantees within a minor
release. Anything outside that surface — `kglite::graph::*`,
`kglite::datatypes::*`, raw module paths — is internal and may
move freely between minor releases.

## License

MIT — see [LICENSE](https://github.com/kkollsga/kglite/blob/main/LICENSE).
