//! Streaming N-Triples loader with bz2/gz/plain support.
//!
//! Designed for Wikidata truthy dumps but works with any N-Triples file.
//!
//! Single-pass algorithm:
//!   1. Stream-decompress → parse lines → filter by predicate/language
//!   2. Accumulate properties per subject; flush node on subject change
//!   3. Buffer edges (source_id, target_id, predicate)
//!   4. After EOF: create all buffered edges, skip if target missing
//!
//! Split:
//! - [`parser`] — triple AST (Subject/Predicate/Object/EdgeBuffer), line
//!   parser, XSD literal → Value coercion, language filters.
//! - [`writer`] — edge-creation dispatch (compact, strings, qnum map).
//! - [`loader`] — `load_ntriples` entry point + per-entity flush.
//! - [`column_builder`] — direct-write columnar build pipeline +
//!   `ColumnTypeMeta` (saved to disk for mmap reload).

use std::collections::{HashMap, HashSet};

pub mod column_builder;
pub mod label_spill;
pub mod loader;
pub mod parallel_bz2;
pub mod parser;
pub mod writer;

pub use column_builder::{ColMapEntry, ColumnTypeMeta, FixedColMeta, RegionMeta, StrColMeta};
pub use loader::load_ntriples;

/// Per-counter value carried by [`ProgressEvent`]. Pure Rust — the
/// pyapi layer maps this into Python types when (and only when)
/// adapting a Python callback into a [`ProgressSink`]. `F64` and
/// `Str` are reserved for sub-step labels emitted from Phase 1b / 2 / 3
/// once those phases get internal progress reporting.
pub enum ProgressValue<'a> {
    U64(u64),
    #[allow(dead_code)] // Reserved for sub-step labels in Phase 1b/2/3.
    F64(f64),
    #[allow(dead_code)] // Reserved for sub-step labels in Phase 1b/2/3.
    Str(&'a str),
}

/// Structured progress event emitted by `load_ntriples` at phase
/// boundaries and within the streaming loop. Phases: `"phase1"`,
/// `"phase1b"`, `"phase2"`, `"phase3"`, `"finalising"`.
pub enum ProgressEvent<'a> {
    Start {
        phase: &'a str,
        label: &'a str,
        total: Option<u64>,
        unit: &'a str,
    },
    Update {
        phase: &'a str,
        current: u64,
        fields: &'a [(&'a str, ProgressValue<'a>)],
    },
    Complete {
        phase: &'a str,
        elapsed_s: f64,
        fields: &'a [(&'a str, ProgressValue<'a>)],
    },
}

/// Marker for cooperative cancellation. Returned by [`ProgressSink::emit`]
/// when the caller (typically the pyapi layer reacting to a Python
/// `KeyboardInterrupt`) wants the loader to stop. The loader threads
/// this back up as a clean error, which the pyapi wrapper translates
/// into `PyKeyboardInterrupt`.
pub struct Cancelled;

/// Sink for [`ProgressEvent`]s. Implementations must be `Send + Sync`
/// because the loader fires events from worker contexts. The pyapi
/// layer provides a Python-callable adapter; pure-Rust callers can
/// implement this trait directly.
///
/// Returning `Err(Cancelled)` from `emit` requests the loader stop at
/// the next safe point. Implementations that never want to cancel can
/// always return `Ok(())`.
pub trait ProgressSink: Send + Sync {
    fn emit(&self, event: ProgressEvent<'_>) -> Result<(), Cancelled>;
}

/// Stats returned after loading.
pub struct NTriplesStats {
    pub triples_scanned: u64,
    pub entities_created: u64,
    pub edges_created: u64,
    pub edges_skipped: u64, // target entity not in graph
    pub seconds: f64,
}

/// Configuration for the loader.
pub struct NTriplesConfig {
    /// Only import these Wikidata predicates (P-codes). None = all.
    pub predicates: Option<HashSet<String>>,
    /// Only keep literals in these languages. None = all.
    pub languages: Option<HashSet<String>>,
    /// Map P31 target Q-code → human-readable node type name.
    pub node_types: HashMap<String, String>,
    /// Map P-code → human-readable predicate label.
    pub predicate_labels: HashMap<String, String>,
    /// Stop after this many entities. None = no limit.
    pub max_entities: Option<usize>,
    /// Stop after this many triples scanned (entity + non-entity lines
    /// alike). None = no limit. Applied alongside `max_entities` —
    /// whichever fires first wins. Useful for benchmarking against the
    /// raw bz2 dump where you want a deterministic, format-agnostic
    /// stopping point.
    pub max_triples: Option<u64>,
    /// Print progress to stderr.
    pub verbose: bool,
    /// Automatically derive node types from P31 values, resolving Q-codes to
    /// labels when known. Entities without P31 default to "Entity".
    /// When true, node_types mappings still take priority.
    pub auto_type: bool,
    /// Optional sink for structured per-phase progress events. The pyapi
    /// layer wraps a Python callable into a [`ProgressSink`]; pure-Rust
    /// callers (benches, tests) implement the trait directly.
    pub progress: Option<Box<dyn ProgressSink>>,
}
