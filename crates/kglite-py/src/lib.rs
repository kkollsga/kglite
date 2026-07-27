// src/lib.rs

// kglite-py is the PyO3 wrapper over the kglite engine crate
// (aliased as `kglite_core` in this crate's source — see
// Cargo.toml for the `package = "kglite"` indirection). The
// local modules expose only the curated engine API plus PyO3-specific
// wrapper concerns.

// mimalloc as the global allocator. samply profile of the N-Triples
// build showed libsystem_malloc accounting for ~32% of loader-thread
// CPU time. mimalloc is consistently faster than macOS's default
// allocator on small-object-heavy workloads (Strings, HashMaps, Vecs
// in the parser hot loop). Pure Rust dependency — no system dep, just
// a slightly larger build artifact.
//
// Pinned to the mimalloc v2 series (`features = ["v2"]` in Cargo.toml),
// NOT the default v3. Why (2026-07-08): a process that co-loads
// `pyarrow==24.0.0` and `kglite` SIGSEGVs at interpreter teardown when
// BOTH ship mimalloc v3. A three-mimalloc census of the crashing wheel
// found the culprit pair — kglite's own v3 (this `#[global_allocator]`,
// statically linked) and the v3 copy CPython 3.14 vendors into libarrow;
// their independent thread-heap teardowns
// (`_mi_theap_collect_retired`, lldb-confirmed inside libarrow's copy)
// collide. v2 coexists with the v3 copy cleanly (verified 0×5 both
// import orders). Dropping mimalloc entirely also fixes the crash but
// costs the allocator win outright, and swapping to jemalloc or the
// system allocator measured a 30-62% in-memory regression on the tracked
// core benchmarks (2026-07-08) — unacceptable per the in-memory-wins
// protocol. v2 keeps nearly all the throughput (core query benches flat;
// ~3-4% on parse-heavy loads) while resolving the teardown clash.
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use pyo3::prelude::*;
mod datatypes;
mod error_py;
mod graph;
mod graphgen;
mod okf;
mod util;

// The pyo3 wrapper depends on the kglite engine for everything
// non-Python. Re-export the engine's `error` module so existing
// `crate::error::*` paths in pyapi/, error_py.rs, the datatypes
// shims, etc. resolve unchanged.
pub use kglite_core::error;

use graph::pyapi::blueprint::{from_blueprint_rust, from_records_rust};
use graph::pyapi::frozen::FrozenGraph;
use graph::pyapi::result_view::{ResultIter, ResultView};
use graph::pyapi::session::Session;
use graph::{KnowledgeGraph, Transaction};
use kglite_core::api::io::load_file;

/// Curated Rust-side façade for downstream binaries (notably
/// `kglite-mcp-server`). This module is the **only** stable Rust
/// API the kglite-py wrapper promises to keep — the underlying
/// `pub mod graph` is public for tooling
/// but their internals can move between minor releases. New
/// consumers should import from `kglite::api::*` (or
/// `kglite_core::api::*` from inside this crate's source, where
/// the dep is aliased); breakage there is a semver concern.
///
/// The Python API (`#[pymethods]` on `KnowledgeGraph`, etc.) is
/// independent — it stays as the wheel's primary surface.
pub mod api {
    pub use crate::datatypes::Value;
    // Per-variant carriers for the Value enum's compound shapes. Phase A.1
    // added `Value::Node` / `Relationship` / `Path` carrying these struct
    // types; downstream Rust consumers (kglite-bolt-server's value adapter,
    // and future Arrow/Polars exporters) want to pattern-match into them
    // without re-deriving accessors.
    pub use crate::datatypes::values::{NodeValue, PathValue, RelValue};
    // Typed error surface — Phase A.2 added KgError + KgErrorCode for the
    // Python boundary; Phase C.6 (bolt-server) consumes them to map onto
    // Neo4j `Neo.ClientError.*` wire codes via `BoltError::Query`.
    pub use crate::error::{KgError, KgErrorCode};
    #[cfg(feature = "fastembed")]
    pub use crate::graph::embedder::fastembed::FastEmbedAdapter;
    pub use crate::graph::embedder::Embedder;
    pub use crate::graph::KnowledgeGraph;
    pub use kglite_core::api::code_entities::{SourceLocation, SourceLookup};
    pub use kglite_core::api::introspection::compute_description;
    pub use kglite_core::api::introspection::compute_schema;
    pub use kglite_core::api::introspection::SchemaOverview;
    pub use kglite_core::api::introspection::{ConnectionDetail, CypherDetail, FluentDetail};
    pub use kglite_core::api::io::{load_file, save_graph};
    pub use kglite_core::api::DirGraph;
    pub use kglite_core::api::{explore_markdown, ExploreOptions};

    /// Cypher parser + planner + executor surface. Downstream Rust
    /// consumers (notably `kglite-mcp-server`) build their own
    /// parse → rewrite_text_score → optimize → execute pipeline using
    /// these items; the Python boundary in
    /// `src/graph/pyapi/kg_core.rs::cypher` is the canonical example.
    ///
    /// **For new consumers, prefer [`session`]** — it bundles the
    /// canonical pipeline + transaction CoW into a single surface
    /// so future drift between bindings is impossible. This raw
    /// `cypher` re-export stays public for callers that need to
    /// reach into specific passes (planner introspection,
    /// custom-disabled-pass sets, etc.).
    pub mod cypher {
        pub use crate::graph::languages::cypher::execute_mutable;
        pub use crate::graph::languages::cypher::generate_explain_result;
        pub use crate::graph::languages::cypher::is_mutation_query;
        pub use crate::graph::languages::cypher::parse_cypher;
        pub use crate::graph::languages::cypher::planner;
        pub use crate::graph::languages::cypher::planner::mark_lazy_eligibility;
        pub use crate::graph::languages::cypher::planner::schema_check::validate_schema;
        pub use crate::graph::languages::cypher::planner::simplification::rewrite_text_score;
        pub use crate::graph::languages::cypher::CypherExecutor;
        pub use crate::graph::languages::cypher::CypherQuery;
        pub use crate::graph::languages::cypher::CypherResult;
        pub use crate::graph::languages::cypher::OutputFormat;
    }

    /// Canonical query + transaction surface. Single source of truth
    /// for the Cypher pipeline (parse → validate → rewrite → optimize
    /// → execute) and the snapshot/working CoW transaction model.
    /// All bindings (pyapi, mcp-server, bolt-server, future Go/TS/JVM)
    /// wrap this module's types and free functions.
    ///
    /// See `docs/rust/session.md` for the operator-facing
    /// guide and `bolt_implementation.md` Phase E for the rationale.
    pub mod session {
        pub use kglite_core::api::session::{
            execute_mut, execute_read, CommitOutcome, ExecuteOptions, ExecuteOutcome, Session,
            Transaction,
        };
    }
}

/// Read-only accessor for the underlying [`DirGraph`] of a
/// [`api::KnowledgeGraph`]. The struct field is private; this method
/// gives downstream Rust binaries a stable handle to plug into the
/// planner / executor surface in [`api::cypher`].
impl crate::graph::KnowledgeGraph {
    pub fn dir(&self) -> &std::sync::Arc<kglite_core::api::DirGraph> {
        &self.inner
    }
}

/// Map a load failure (`load_file` / `load_kgl_bytes`, which return
/// `io::Error`) to a *classifiable* typed exception, so callers can reliably
/// distinguish "this `.kgl` is corrupt → rebuild from source" (`FileFormatError`)
/// from "it isn't there" (`FileError`) or a genuine IO fault (`FileIoError`),
/// instead of catching a broad `IOError`. A load that fails for any reason
/// other than not-found / permission is treated as a format/corruption error
/// (bad magic, truncated section, version mismatch, compression/codec failure).
fn load_err_to_pyerr(e: std::io::Error, path: Option<&str>) -> PyErr {
    use std::io::ErrorKind;
    let pb = || std::path::PathBuf::from(path.unwrap_or(""));
    let kg = match e.kind() {
        ErrorKind::NotFound => crate::error::KgError::FileNotFound(pb()),
        ErrorKind::PermissionDenied => crate::error::KgError::FileIo(e),
        _ => crate::error::KgError::FileFormat {
            path: pb(),
            message: e.to_string(),
        },
    };
    crate::error_py::kg_to_pyerr(kg)
}

#[pyfunction]
fn load(py: Python<'_>, path: String) -> PyResult<KnowledgeGraph> {
    py.detach(|| load_file(&path))
        .map(|inner| {
            let mut kg = KnowledgeGraph::from_arc(inner);
            kg.lifecycle.source_path = Some(std::path::PathBuf::from(&path));
            kg
        })
        .map_err(|e| load_err_to_pyerr(e, Some(&path)))
}

/// Load an RDF file into a fresh in-memory graph and return it.
///
/// Dispatches on the file extension: `.ttl` → Turtle, `.nt` → N-Triples,
/// `.nq` → N-Quads, `.trig` → TriG. The RDF→property-graph fold is:
/// object literals become typed node properties, resource objects become
/// edges, and `rdf:type` sets the node label (first wins; any extra types
/// are kept in an `rdf_types` list property). Predicate and type IRIs are
/// CURIE-compacted using the document's own `@prefix` declarations plus a
/// well-known prefix table; the full subject IRI is kept in each node's
/// `uri` property and `n.id` is a dense integer.
///
/// Keyword args: `languages` (keep only literals in these language tags),
/// `label_predicates` (IRIs whose literal sets the node title; defaults to
/// `rdfs:label`), `keep_full_iris` (skip CURIE compaction), `default_type`
/// (node type for subjects without `rdf:type`; defaults to `"Resource"`),
/// `max_triples` (stop after N).
#[pyfunction]
#[pyo3(signature = (path, *, languages=None, label_predicates=None, keep_full_iris=false, default_type=None, max_triples=None))]
fn load_rdf(
    py: Python<'_>,
    path: String,
    languages: Option<Vec<String>>,
    label_predicates: Option<Vec<String>>,
    keep_full_iris: bool,
    default_type: Option<String>,
    max_triples: Option<u64>,
) -> PyResult<KnowledgeGraph> {
    use std::collections::HashSet;

    let config = kglite_core::api::io::RdfConfig {
        languages: languages.map(|v| v.into_iter().collect::<HashSet<_>>()),
        label_predicates: label_predicates
            .unwrap_or_else(|| vec!["http://www.w3.org/2000/01/rdf-schema#label".to_string()]),
        keep_full_iris,
        default_type: default_type.unwrap_or_else(|| "Resource".to_string()),
        max_triples,
    };

    let mut graph = kglite_core::api::DirGraph::new();
    py.detach(|| kglite_core::api::io::load_rdf(&mut graph, &path, &config))
        .map_err(|e| {
            if e.starts_with("Cannot open") {
                PyErr::new::<pyo3::exceptions::PyFileNotFoundError, _>(e)
            } else {
                PyErr::new::<pyo3::exceptions::PyValueError, _>(e)
            }
        })?;
    Ok(KnowledgeGraph::from_arc(std::sync::Arc::new(graph)))
}

/// Load a saved graph at `path` directly as a thread-safe [`Session`] — the
/// one-call shortcut for the concurrent-serving case (equivalent to
/// `kglite.load(path).session()`).
///
/// Share the returned `Session` across a thread pool: `cypher()` reads run
/// lock-free, `execute()` writes serialize (and compose), and `cursor()` hands
/// each thread its own per-thread fluent handle. The file must already exist.
///
/// For embedding-backed semantic search (`text_score()` over a query string),
/// register the model first via the `KnowledgeGraph` path:
/// `g = kglite.load(path); g.set_embedder(model); s = g.session()`.
#[pyfunction]
fn open_session(py: Python<'_>, path: String) -> PyResult<Session> {
    let inner = py
        .detach(|| load_file(&path))
        .map_err(|e| load_err_to_pyerr(e, Some(&path)))?;
    Ok(Session::from_arc(inner, None))
}

/// Load an in-memory graph from a `.kgl` byte buffer produced by
/// `graph.to_bytes()` — the in-memory counterpart of `kglite.load(path)`.
/// The returned graph has no `source_path` (it didn't come from a file),
/// so a bare `save()` will ask for an explicit path. A corrupt/truncated
/// or non-`.kgl` buffer raises a classifiable error (bad magic / truncated
/// section), distinct from a successful empty graph.
#[pyfunction]
fn from_bytes(py: Python<'_>, data: &[u8]) -> PyResult<KnowledgeGraph> {
    py.detach(|| kglite_core::api::io::load_kgl_bytes(data))
        .map(KnowledgeGraph::from_arc)
        .map_err(|e| load_err_to_pyerr(e, None))
}

/// Open a graph at `path`, loading it if the file/directory exists or
/// creating a fresh one if it doesn't (load-or-create) — the embedded-DB
/// lifecycle entry point. The returned graph remembers `path`, so a later
/// bare `save()` (or the context-manager auto-save-on-close) writes back to
/// it without re-specifying the target.
///
/// `storage` (`"mapped"` / `"disk"`) applies only when *creating* a new graph.
/// Opening an existing path uses the mode the load produces, which is not
/// necessarily the mode that wrote it: a `.kgl` checkpoint records no storage
/// mode, so it always loads as `"memory"`, while a disk graph is a directory
/// and always loads as `"disk"`. Passing a `storage=` that disagrees with the
/// loaded backend raises `kglite.ArgumentError` rather than being ignored — a
/// silently-downgraded mode is indistinguishable from success, and in
/// particular `open(path, storage="mapped")` gives a genuinely mapped graph
/// only on the call that *creates* the file.
///
/// **Durability differs between the two ways to build a graph, and the
/// difference is structural.** `kglite.open()` attaches a WAL sidecar next to
/// `path` and defaults to `durable="full"`. The `KnowledgeGraph(...)`
/// constructor has no `durable` argument at all and is never durable: it
/// produces a *detached* graph with no `source_path`, so there is no location
/// for a log to live. `KnowledgeGraph(storage="mapped")` is therefore
/// mapped-and-unlogged, while `open(path, storage="mapped")` on a fresh path is
/// mapped-and-logged.
///
/// The `durable=` argument: a level name, a bool, or `None`.
///
/// `True`/`False` are accepted *spellings* of `"full"`/`"off"`, not a second
/// code path — every form normalises to one [`DurabilityLevel`] here, at the
/// single point where Python vocabulary becomes engine vocabulary, and
/// nothing downstream knows which spelling was used.
/// Normalise the `durable=` argument into one [`DurabilityLevel`].
///
/// A plain function rather than a `FromPyObject` impl: the trait's shape has
/// moved across PyO3 releases, and nothing here needs to participate in
/// generic extraction — `open` is the only caller, and it wants to own the
/// error messages anyway.
fn durable_level_from_arg(
    ob: &Bound<'_, PyAny>,
) -> PyResult<kglite_core::api::durable::DurabilityLevel> {
    use kglite_core::api::durable::DurabilityLevel;
    // `bool` extraction is strict in PyO3 (it requires an actual `PyBool`),
    // so this cannot swallow `1` or a string.
    if let Ok(flag) = ob.extract::<bool>() {
        return Ok(if flag {
            DurabilityLevel::Full
        } else {
            DurabilityLevel::Off
        });
    }
    let name: String = ob.extract().map_err(|_| {
        PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
            "durable must be a bool or one of {:?} (True is 'full', False is 'off')",
            DurabilityLevel::NAMES,
        ))
    })?;
    DurabilityLevel::from_name(&name).ok_or_else(|| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "unknown durability level {name:?}. Valid levels are {:?}: \
             'full' survives power loss (one barrier per commit), \
             'normal' survives the process dying but not an OS crash or \
             power loss, 'off' keeps no log and relies on save().",
            DurabilityLevel::NAMES,
        ))
    })
}

/// Durability is **on by default**: each committed mutation is `fsync`'d to a
/// `<path>-wal` sidecar before the call returns, and on open any WAL frames are
/// replayed onto the loaded checkpoint to recover work committed since the last
/// `save()`. This is the point of an embedded database — without it a hard
/// crash loses everything since the last explicit save.
///
/// `durable` names **what a committed mutation survives**, using SQLite's
/// `synchronous` vocabulary. Guarantees, not syscalls — the syscall differs by
/// platform, the guarantee does not:
///
/// - `"full"` (also spelled `True`) — survives **power loss**. One barrier per
///   commit; on macOS that is `F_FULLFSYNC`, which is a stronger guarantee than
///   SQLite's own default gives.
/// - `"normal"` — survives the **process** dying (`SIGKILL`, an unhandled
///   panic, an OOM-kill), because the frame is in the kernel's page cache
///   before the call returns. An OS crash or power loss loses commits made
///   since the last `save()`. No barrier per commit, so it costs
///   essentially nothing beyond the write itself. Call `sync()` to take an
///   explicit power-safe point without a full checkpoint.
/// - `"off"` (also spelled `False`) — no log at all. The graph still remembers
///   `path` for `save()`, which is measurably faster for write-heavy bulk
///   loading and is the right choice when the graph is rebuildable from source
///   data.
/// - `None` (the default) — `"full"`, except on `storage="disk"` where it
///   resolves to `"off"` (see below) rather than raising.
///
/// **The rungs are not uniform across storage modes: `storage="disk"` supports
/// only `"off"`, and any explicit request to log raises `ValueError`** — a disk
/// graph commits by publishing an immutable generation rather than by logging a
/// write, so the blocker is structural and not a matter of barrier strength.
///
/// `lock` (default `true`) takes the cross-process single-writer lease for the
/// life of the returned graph. It is on by default because the alternative is
/// silent data loss, not an error: two processes that open one path both build
/// a complete snapshot and the last `save()` wins. `false` opts out for callers
/// who coordinate writers themselves. Readers never take the lease — that is
/// what `load` / `open_session` are for.
#[pyfunction]
#[pyo3(signature = (path, *, storage=None, durable=None, lock=true))]
fn open(
    py: Python<'_>,
    path: String,
    storage: Option<&str>,
    durable: Option<Bound<'_, PyAny>>,
    lock: bool,
) -> PyResult<KnowledgeGraph> {
    use kglite_core::api::durable::DurabilityLevel;
    use kglite_core::api::GraphRead;

    // Take write ownership *before* reading a byte. The window that loses a
    // writer's work is open-to-save, not save itself: two processes that both
    // load, both mutate, and both save produce two full snapshots, and the
    // second one published wins outright. Locking at save time would be too
    // late to notice.
    //
    // Fail-fast (`Duration::ZERO`) rather than waiting: `open()` is called on
    // request paths and in worker startup, where a blocked-for-30s open is a
    // worse failure than a clear error. A caller that genuinely wants to queue
    // can retry around the error.
    let writer_lease = if lock {
        Some(
            py.detach(|| {
                kglite_core::api::io::GraphWriterLease::acquire(
                    std::path::Path::new(&path),
                    std::time::Duration::ZERO,
                )
            })
            .map_err(|e| {
                // The engine's message is binding-neutral by design, so the
                // Python-specific way out is appended here rather than baked
                // into core. Most callers who hit this wanted to *read* a
                // graph someone else is writing, and naming the call that
                // does that turns a refusal into an answer.
                crate::error_py::kg_to_pyerr(crate::error::KgError::FileIo(std::io::Error::new(
                    e.kind(),
                    format!(
                        "{e} To read this graph while another process writes it, use \
                         kglite.load(path) or kglite.open_session(path), which take no \
                         lease. Pass kglite.open(..., lock=False) only if you are \
                         coordinating writers yourself.",
                    ),
                )))
            })?,
        )
    } else {
        None
    };

    // Parse the mode up front, on *both* branches. Validating it only inside
    // `construct` meant an unknown spelling was silently accepted whenever the
    // path happened to exist already.
    let requested_mode = storage
        .map(|mode_str| {
            kglite_core::api::storage::StorageMode::parse(mode_str)
                .map_err(|e| crate::error_py::kg_to_pyerr(crate::error::KgError::Argument(e)))
        })
        .transpose()?;

    let existed = std::path::Path::new(&path).exists();
    let mut kg = if existed {
        py.detach(|| load_file(&path))
            .map(KnowledgeGraph::from_arc)
            .map_err(|e| load_err_to_pyerr(e, Some(&path)))?
    } else {
        KnowledgeGraph::construct(storage, Some(&path))?
    };
    // An existing path is *loaded*, and the load decides the backend — so a
    // `storage=` that disagrees with what came back cannot be honoured. Refuse
    // instead of ignoring it: silently handing back a different mode than the
    // caller asked for is indistinguishable from success, and has already
    // invalidated at least one mapped-vs-memory comparison where both arms
    // unknowingly ran on the memory backend.
    if existed {
        if let Some(requested) = requested_mode {
            let actual = actual_storage_mode(&kg);
            if requested != actual {
                // `ArgumentError`, matching `KnowledgeGraph(storage="invalid")`
                // and the parse failure above — one error class for one kwarg.
                // (The neighbouring disk-plus-`durable` refusal raises
                // `ValueError`; that is about `durable=`, not about this.)
                let message = format!(
                    "cannot open existing graph {path:?} with storage={:?}: it loaded as {:?}. \
                     `storage=` selects a backend for a graph being *created*; an existing \
                     path is loaded, and the load decides the backend. A `.kgl` checkpoint \
                     does not record which mode wrote it, so loading one always yields \
                     'memory'; a disk graph is a directory and always yields 'disk'. {} \
                     To accept whatever the file provides, omit `storage=`.",
                    requested.as_str(),
                    actual.as_str(),
                    storage_mode_remedy(requested),
                );
                return Err(crate::error_py::kg_to_pyerr(
                    crate::error::KgError::Argument(message),
                ));
            }
        }
    }
    kg.lifecycle.source_path = Some(std::path::PathBuf::from(&path));
    kg.lifecycle.writer_lease = writer_lease;
    // Resolved after the graph exists, because the mode of an *existing* path
    // comes from the file, not from the `storage` argument.
    let level = match durable.as_ref() {
        Some(ob) => durable_level_from_arg(ob)?,
        // Default stays the strongest level the mode supports. Disk keeps no
        // log at any level, so it resolves to `off` instead of raising —
        // unchanged from when `durable` was a plain tri-state bool.
        None if kg.inner.graph.is_disk() => DurabilityLevel::Off,
        None => DurabilityLevel::Full,
    };
    if level.logs() {
        setup_durable(&mut kg, &path, level)?;
    }
    Ok(kg)
}

/// The storage mode `kg` actually ended up in, for comparison against a
/// caller's `storage=` request.
fn actual_storage_mode(kg: &KnowledgeGraph) -> kglite_core::api::storage::StorageMode {
    use kglite_core::api::storage::StorageMode;
    use kglite_core::api::GraphRead;
    if kg.inner.graph.is_disk() {
        StorageMode::Disk
    } else if kg.inner.graph.is_mapped() {
        StorageMode::Mapped
    } else {
        StorageMode::Memory
    }
}

/// The actionable half of the "mode cannot be honoured on open" error: what to
/// do instead, per requested mode.
///
/// `mapped` is the blunt one. There is no load-into-an-existing-graph call and
/// no saved-graph-to-mapped conversion, so the only route is to build a mapped
/// graph and re-ingest from the original source data. That is worth saying
/// plainly rather than implying a conversion exists.
fn storage_mode_remedy(requested: kglite_core::api::storage::StorageMode) -> &'static str {
    use kglite_core::api::storage::StorageMode;
    match requested {
        StorageMode::Mapped => {
            "Mapped mode cannot be recovered from a saved graph: build one with \
             `kglite.KnowledgeGraph(storage=\"mapped\")` and re-ingest from your source data, \
             or delete the path and let this call create it."
        }
        StorageMode::Disk => {
            "To move a loaded graph onto disk storage, call `g.enable_disk_mode()` after \
             opening it, or delete the path and let this call create it."
        }
        StorageMode::Memory => {
            "To get a memory-backed graph, open a `.kgl` file rather than a disk-graph \
             directory."
        }
    }
}

/// Turn `kg` into a durable graph: replay any WAL frames committed since
/// the last checkpoint onto the loaded graph, wrap its backend in the
/// write-capture layer, and open the WAL for append.
///
/// Storage-mode-agnostic by construction: the capture wrapper wraps the
/// `GraphBackend` enum rather than a concrete backend, and both memory and
/// mapped graphs mutate the same heap `StableDiGraph` underneath, so one
/// capture path covers both. Disk is refused — see the error below.
fn setup_durable(
    kg: &mut KnowledgeGraph,
    path: &str,
    level: kglite_core::api::durable::DurabilityLevel,
) -> PyResult<()> {
    use kglite_core::api::GraphRead;
    // `wal` stays a below-api reach (durable-transaction internals) —
    // deferred to a high-level durable-transaction api lift (roadmap Piece 2).
    use kglite_core::api::durable as wal;

    if kg.inner.graph.is_disk() {
        // `ValueError`, not one of the typed `kglite.*Error` classes, and that
        // is deliberate: `error_py`'s taxonomy reserves the typed hierarchy for
        // *engine* failures and keeps "argument-shape" rejections in their
        // built-in family. An unsupported `durable` + `storage` combination is
        // rejected at the Python API boundary before any engine work starts,
        // so it belongs in the built-in family — as the pre-existing
        // `test_durable_rejects_disk_mode` contract already fixed.
        //
        // Every logging level is refused here, not just `full`. The blocker is
        // structural rather than a matter of barrier strength: there is no WAL
        // for disk mode at any level, so `normal` would be exactly as
        // unimplementable as `full`. Naming the requested level keeps the
        // message honest for a caller who reasonably assumed the rungs were
        // uniform across storage modes.
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "durable={:?} is not supported for storage='disk' (only 'off' is). \
             A disk graph commits by publishing an immutable generation, so its \
             durability boundary is the generation publish, not a logical \
             write-ahead log: a replayed WAL frame and a published generation \
             can each describe the same commit, and reconciling them needs a \
             generation-aware log this release does not have. Use save() \
             checkpoints for disk graphs, or storage='mapped' / the in-memory \
             default if you need per-commit crash safety.",
            level.name(),
        )));
    }
    let sync = level
        .sync_mode()
        .expect("caller checks level.logs() before setup_durable");

    let wpath = wal::wal_path(std::path::Path::new(path));
    // Read (do not truncate) any frames committed since the last checkpoint.
    let frames = wal::recover(&wpath)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;

    // Replay onto the (unwrapped) loaded graph, then wrap so subsequent
    // mutations are captured. Replaying before wrapping keeps the replay's
    // own GraphWrite calls out of the capture buffer.
    let dir = crate::graph::get_graph_mut(&mut kg.inner);
    let max_lsn = kglite_core::api::durable::apply_frames(dir, &frames, 0)
        .map_err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>)?;
    KnowledgeGraph::wrap_backend_for_durability(dir);

    let walh = wal::Wal::open(wpath, sync)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
    kg.lifecycle.durable = Some(crate::graph::DurableState {
        wal: walh,
        next_lsn: max_lsn + 1,
        level,
    });
    Ok(())
}

/// Names of every Cypher optimizer pass, in execution order. Useful for
/// the `disabled_passes=` kwarg of `KnowledgeGraph.cypher()` and for
/// bisection scripts. The list is the source of truth — names that
/// aren't here will be rejected by `cypher(..., disabled_passes=[...])`.
#[pyfunction]
fn cypher_pass_names() -> Vec<String> {
    kglite_core::api::cypher::planner::all_pass_names()
}

/// Run the shared KGLite CLI in-process and block until it exits.
///
/// The standalone `kglite-cli` binary and the `kglite` console script bundled
/// in this wheel both call the same pure-Rust library. `argv` excludes the
/// program name, which is synthesized here for clap. The Python shim owns only
/// console-script error formatting; all command behavior remains Rust-side.
#[pyfunction]
fn _run_cli(py: Python<'_>, argv: Vec<String>) -> PyResult<()> {
    let mut full = Vec::with_capacity(argv.len() + 1);
    full.push("kglite".to_string());
    full.extend(argv);
    py.detach(|| kglite_cli::run(full))
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{e:#}")))
}

/// Run the bundled MCP server in-process and block until it exits.
///
/// This is the exact same server as the standalone `kglite-mcp-server`
/// binary — it lives in the `kglite-mcp-server` *library* (pure Rust, no
/// libpython link) and is statically linked into this wheel, sharing the
/// one `kglite` engine. The `kglite-mcp-server` console script (a thin
/// `kglite/mcp_server.py` shim) is the public entry point; it forwards
/// `sys.argv[1:]` here.
///
/// `argv` is the argument vector **without** the program name; clap
/// expects `argv[0]` to be the program name, so we synthesise it. The
/// server serves over stdio and runs its own tokio runtime, so this
/// blocks for the process lifetime — `py.detach` releases the GIL for
/// the entire run (the Python process simply *becomes* the MCP server).
///
/// `embedder_factory`, when given, is a Python callable
/// `factory(config_json: str) -> EmbeddingModel`, where `config_json` is the
/// manifest's whole `extensions.embedder` object. It is invoked **only** for a
/// Python-hosted embedder library (`library: fastembed` / `sentence-transformers`
/// / a `factory:` escape — anything that isn't `fastembed-rs`). The factory
/// (`kglite._mcp_embed`) picks the library, builds the model, and returns an
/// `EmbeddingModel`; the server wraps it in a `PyEmbedderAdapter` (GIL
/// re-acquired just for the per-query embed) so `text_score()` runs against it
/// with no Rust toolchain. The standalone cargo binary supplies no factory, so
/// a Python library errors there (use `library: fastembed-rs`).
#[cfg(feature = "mcp-server")]
#[pyfunction]
#[pyo3(signature = (argv, embedder_factory=None))]
fn _run_mcp_server(
    py: Python<'_>,
    argv: Vec<String>,
    embedder_factory: Option<Py<PyAny>>,
) -> PyResult<()> {
    let mut full = Vec::with_capacity(argv.len() + 1);
    full.push("kglite-mcp-server".to_string());
    full.extend(argv);

    // Bridge the Python factory into the libpython-free server library as a
    // Rust closure producing an `Arc<dyn Embedder>`. The closure re-acquires
    // the GIL (`Python::attach`) only when the server actually calls it — at
    // boot, if the manifest declares a Python embedder library. The argument is
    // the `extensions.embedder` config as JSON, so Python owns library choice.
    let factory: Option<kglite_mcp_server::PyEmbedderFactory> = embedder_factory.map(|f| {
        Box::new(move |config_json: &str| -> Result<std::sync::Arc<dyn kglite_core::api::Embedder>, String> {
            Python::attach(|py| {
                let instance = f
                    .call1(py, (config_json,))
                    .map_err(|e| format!("embedder factory raised: {e}"))?;
                let adapter = graph::embedder::py_adapter::PyEmbedderAdapter::new(py, instance)
                    .map_err(|e| {
                        format!("embedder factory returned an object missing the EmbeddingModel protocol (need `dimension` + `embed`): {e}")
                    })?;
                Ok(std::sync::Arc::new(adapter) as std::sync::Arc<dyn kglite_core::api::Embedder>)
            })
        }) as kglite_mcp_server::PyEmbedderFactory
    });

    py.detach(|| kglite_mcp_server::run_with_embedder_factory(full, factory))
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{e:#}")))
}

// `gil_used = false` declares this module compatible with the free-threaded
// (no-GIL, 3.14t) build — PyO3 0.29 makes this opt-in. The engine's shared
// types are already `Send + Sync` (the concurrent `Session` / `FrozenGraph`
// path is built on that), and the read/write entry points release the GIL via
// `EnterKg`, so the module doesn't rely on the GIL as an implicit lock.
#[pymodule(gil_used = false)]
fn kglite(py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(load, m)?)?;
    m.add_function(wrap_pyfunction!(load_rdf, m)?)?;
    m.add_function(wrap_pyfunction!(open_session, m)?)?;
    m.add_function(wrap_pyfunction!(from_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(open, m)?)?;
    m.add_function(wrap_pyfunction!(from_blueprint_rust, m)?)?;
    m.add_function(wrap_pyfunction!(from_records_rust, m)?)?;
    m.add_function(wrap_pyfunction!(cypher_pass_names, m)?)?;
    m.add_function(wrap_pyfunction!(_run_cli, m)?)?;
    #[cfg(feature = "mcp-server")]
    m.add_function(wrap_pyfunction!(_run_mcp_server, m)?)?;
    m.add_class::<KnowledgeGraph>()?;
    m.add_class::<FrozenGraph>()?;
    m.add_class::<Session>()?;
    m.add_class::<Transaction>()?;
    m.add_class::<ResultView>()?;
    m.add_class::<ResultIter>()?;
    // Phase A.2 / C1 — typed exception class hierarchy. Every kglite
    // error surfaces as `kglite.KgError` or a more specific subclass.
    error_py::register(py, m)?;
    graphgen::register(py, m)?;
    okf::pyapi::register(py, m)?;
    Ok(())
}
