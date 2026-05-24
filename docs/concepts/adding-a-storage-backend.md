# Adding a storage backend

This guide walks through adding a new storage backend to KGLite. The
example used here is `RecordingGraph`, the validation wrapper shipped
in Phase 6 of the 0.8.0 refactor — a "real" backend built on top of
the public trait surface that proves the architecture is actually
open/closed.

## TL;DR

Adding a backend is a **3-src-file change**:

1. **Own file** — `src/graph/storage/<name>.rs`: a struct + `impl GraphRead for Name` (+ optionally `impl GraphWrite for Name`).
2. **Enum variant** — `src/graph/storage/backend.rs`: a new `GraphBackend::Name(...)` arm in the dispatch enum.
3. **Re-export** — `src/graph/storage/mod.rs`: `pub mod <name>; pub use <name>::Name;`.

Plus a parity test in `tests/test_phaseN_parity.py` that wraps your
backend in the existing cross-mode oracle.

## Pattern overview

KGLite's storage layer is anchored on two traits:

- `GraphRead` — read-side API (counts, per-node properties, iteration,
  neighbour lookup, backend-kind predicates, edge accessors).
- `GraphWrite: GraphRead` — mutations (add/remove node/edge,
  `node_weight_mut`, `edge_weight_mut`, `update_row_id`).

Both live in `src/graph/storage/mod.rs`. Iterator methods use generic
associated types (GATs) — see the "GATs and object-safety" section of
the trait doc. The trait is **not object-safe**; `&dyn GraphRead` does
not compile. All consumers take `&impl GraphRead`.

The `GraphBackend` enum in `storage/backend.rs` is a **4-arm dispatcher** that
routes to the per-backend `impl GraphRead` / `impl GraphWrite` impls
(currently `MemoryGraph`, `MappedGraph`, `DiskGraph`, and
`RecordingGraph` as the wrapping variant).

## Worked example: RecordingGraph

RecordingGraph is a thin wrapper backend that logs every `GraphRead`
method call to a `Mutex<Vec<&'static str>>` while forwarding every
call to an inner backend. It's generic over any `G: GraphRead`, so
you can wrap a Memory / Mapped / Disk graph and get a read-audit
log for free. It's Rust-only (no Python constructor reaches it) — the
parity matrix for it lives in `src/graph/storage/recording.rs::tests`.

### 1. Write the backend module

File: `src/graph/storage/recording.rs`.

```rust
use crate::graph::storage::{GraphRead, GraphWrite};
use std::sync::Mutex;

pub struct RecordingGraph<G> {
    pub(crate) inner: G,
    log: Mutex<Vec<&'static str>>,
}

impl<G: GraphRead> RecordingGraph<G> {
    pub fn new(inner: G) -> Self {
        RecordingGraph {
            inner,
            log: Mutex::new(Vec::new()),
        }
    }

    pub fn log(&self) -> Vec<&'static str> {
        self.log.lock().unwrap().clone()
    }
}

impl<G: GraphRead> GraphRead for RecordingGraph<G> {
    // Associated types forward directly — GATs just thread through.
    type NodeIndicesIter<'a> = G::NodeIndicesIter<'a> where Self: 'a;
    type EdgesIter<'a> = G::EdgesIter<'a> where Self: 'a;
    // ...one per iterator-returning method...

    fn node_count(&self) -> usize {
        self.log.lock().unwrap().push("node_count");
        self.inner.node_count()
    }

    // ...etc for every trait method. The body is always:
    //   1. push the method name into the log
    //   2. forward to `self.inner`
}

// GraphWrite passthrough (mutations don't log — scope decision).
impl<G: GraphWrite> GraphWrite for RecordingGraph<G> {
    fn add_node(&mut self, data: NodeData) -> NodeIndex {
        self.inner.add_node(data)
    }
    // ...etc...
}
```

Key points:

- **Generic over `G`** — you don't have to know the concrete backend at compile time.
- **GATs forward directly** — `type NodeIndicesIter<'a> = G::NodeIndicesIter<'a>`.
- **`Mutex` not `RefCell`** — PyO3 requires the outer `KnowledgeGraph` class to be `Send`, and `RefCell: !Send`.
- **`log` is `Vec<&'static str>`, not a typed `ReadOp` enum** — simpler, equally testable.

### 2. Add the enum variant

File: `src/graph/storage/backend.rs`.

```rust
pub enum GraphBackend {
    Memory(MemoryGraph),
    Mapped(MappedGraph),
    Disk(Box<DiskGraph>),
    // New variant, boxed to avoid infinite recursion on the generic
    // (GraphBackend contains a RecordingGraph<GraphBackend>).
    Recording(Box<RecordingGraph<GraphBackend>>),
}
```

And extend every `impl GraphRead for GraphBackend` method with the
fourth arm:

```rust
impl GraphRead for GraphBackend {
    fn node_count(&self) -> usize {
        match self {
            Self::Memory(g) => GraphRead::node_count(g),
            Self::Mapped(g) => GraphRead::node_count(g),
            Self::Disk(g) => GraphRead::node_count(g.as_ref()),
            Self::Recording(rg) => GraphRead::node_count(rg.as_ref()),
        }
    }
    // ...
}
```

The `is_memory` / `is_mapped` / `is_disk` helpers should **look through
the wrapper** — `GraphBackend::Recording(wrapping Memory)` should
report `is_memory() == true` so downstream call sites that gate on
backend kind continue to work:

```rust
fn is_memory(&self) -> bool {
    match self {
        Self::Memory(_) => true,
        Self::Recording(rg) => GraphRead::is_memory(rg.as_ref()),
        _ => false,
    }
}
```

### 3. Register the module

File: `src/graph/storage/mod.rs`.

```rust
pub mod recording;
pub use recording::RecordingGraph;
```

That's the whole PR surface. Total LoC: ~500 (the wrapper is mostly
mechanical forwarding).

## Testing

Every new backend should pass the **cross-backend parity oracle**.
Phases 1–6 each ship a `tests/test_phaseN_parity.py` file; the
relevant one for your phase asserts behaviour identity across backends
for a curated query set. A new backend joins the parametrization:

```python
@pytest.fixture(params=["memory", "mapped", "disk", "my_new_backend"])
def kg(request):
    ...
```

For Rust-only backends like `RecordingGraph` (no Python constructor),
the tests live inline in the backend's own `.rs` file under
`#[cfg(test)] mod tests`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn logs_node_count() {
        let mem = MemoryGraph::default();
        let rg = RecordingGraph::new(mem);
        let _ = rg.node_count();
        assert_eq!(rg.log(), vec!["node_count"]);
    }

    // ...one per GraphRead method + one cross-backend parity matrix...
}
```

`RecordingGraph` ships with 13 unit tests covering the full trait
surface on each of the three production backends it can wrap.

## File-count budget

Phase 6's gate (`tests/test_phase6_parity.py::test_file_count_budget`)
enforces that adding a wrapper backend touches at most 3 src files:

- own file (`src/graph/storage/recording.rs`)
- enum + dispatch (`src/graph/storage/backend.rs`)
- re-export (`src/graph/storage/mod.rs`)

Anything more than that means either (a) the new backend couldn't
express itself in the existing trait surface (add the method to the
trait + implement for every backend, not just yours), or (b) the
backend leaked concrete-type matches somewhere outside the dispatch
layer. Both require design revisions.

## Reading more

- `src/graph/storage/mod.rs` — the `GraphRead` / `GraphWrite` trait surface.
- `src/graph/storage/recording.rs` — the worked example from this guide.
- `src/graph/storage/impls.rs` — the three production backends' trait impls (Memory / Mapped / Disk).
- `src/graph/storage/backend.rs` — the 4-arm `GraphBackend` dispatcher.
- `dev-documentation/todo.md` Phase 6 Report-out (gitignored, repo-checkout only) — lessons learned from RecordingGraph's implementation.
