//! kglite — pure-Rust knowledge graph engine.
//!
//! Cypher pipeline, snapshot/working CoW transactions, columnar /
//! mmap / disk storage backends, optional dataset loaders
//! (SEC EDGAR, Sodir, Wikidata). The Python wheel
//! (`pip install kglite`) is built by the sibling `kglite-py`
//! crate; the Bolt and MCP protocol servers are separate
//! workspace binaries.
//!
//! ## Public API
//!
//! Downstream Rust consumers (the Python wheel, the bolt and
//! mcp server binaries, future Go/TypeScript/JVM bindings)
//! should depend on the curated [`api`] module — those items
//! get semver guarantees. Anything else is an implementation
//! detail.
//!
//! See `docs/rust/embedding.md` for the embedder guide.

// Phase A.2 / C2 — crate-wide allow for clippy::result_large_err.
// `KgError` is intentionally rich (16 variants spanning Cypher /
// schema / IO / argument validation) so its size pushes past clippy's
// default 128-byte threshold. Boxing the error variant in every
// `Result<T, KgError>` would add an allocation per error path for no
// real benefit — error paths aren't hot. Standard pattern for crates
// with a unified typed error.
#![allow(clippy::result_large_err)]
// Phase G.3a — when the engine moved from a cdylib (root crate) into
// an rlib (this crate), clippy's lint set widened because more items
// became reachable across crate boundaries. The noisy lints below
// were tolerated in the root crate's pub(crate) surface; they're
// follow-up cleanup tasks but not blockers. Two of them
// (hidden_glob_reexports, private_interfaces) directly reflect the
// rushed wide-public visibility bumps in G.3a — proper accessor
// methods would resolve them.
#![allow(clippy::new_without_default)]
#![allow(clippy::len_without_is_empty)]
#![allow(private_interfaces)]
#![allow(hidden_glob_reexports)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::should_implement_trait)]
#![allow(clippy::result_unit_err)]

pub mod code_tree;
pub mod datasets;
pub mod datatypes;
pub mod error;
pub mod graph;
#[cfg(feature = "okf")]
pub mod okf;
pub mod param;

/// Curated stable Rust API. Downstream consumers should depend on
/// items here, not on the underlying module structure (which may
/// move between minor releases).
pub mod api {
    pub use crate::code_tree::builder::run_with_options as build_code_tree;
    /// Map a file path to its `code_tree` language identifier, or
    /// `None` if no parser handles the file. Bindings use this to
    /// decide whether a filesystem event is graph-relevant — only
    /// changes to files this returns `Some` for can change what
    /// `build_code_tree` produces.
    pub use crate::code_tree::parsers::language_for_path;
    pub use crate::datatypes::values::{NodeValue, PathValue, RelValue};
    pub use crate::datatypes::Value;
    pub use crate::error::{KgError, KgErrorCode};
    pub use crate::graph::dir_graph::DirGraph;
    #[cfg(feature = "fastembed")]
    pub use crate::graph::embedder::fastembed::FastEmbedAdapter;
    pub use crate::graph::embedder::Embedder;
    pub use crate::graph::explore::{explore_markdown, ExploreOptions};
    // Inline timeseries config types (lifted from kglite-py in 0.10.1).
    pub use crate::graph::features::timeseries::{InlineTimeseriesConfig, TimeSpec};
    // Thin pure-Rust graph handle for embedders + the free function
    // backing it. The wheel crate (`kglite-py`) defines its own,
    // Python-flavored `KnowledgeGraph` separately — same name,
    // different audience (`pip install kglite` users), polars-style.
    //
    // `infer_selection_node_type` is NOT re-exported here: it depends
    // on `CowSelection`, a wheel-only-external-consumer type. When
    // the Selection concept gets lifted to a stable api type, both
    // should land in api together. The wheel reaches the function
    // directly via `kglite_core::graph::handle::infer_selection_node_type`.
    pub use crate::graph::handle::{
        discover_property_keys_from_data, source_location, KnowledgeGraph,
    };
    // `Arc<DirGraph>` → `&mut DirGraph` + version bump (lifted in 0.10.1).
    pub use crate::graph::dir_graph::make_dir_graph_mut;
    pub use crate::graph::introspection::describe::compute_description;
    pub use crate::graph::introspection::schema_overview::compute_schema;
    pub use crate::graph::introspection::SchemaOverview;
    pub use crate::graph::introspection::{ConnectionDetail, CypherDetail, FluentDetail};
    pub use crate::graph::io::file::{load_file, save_graph};
    pub use crate::graph::{SourceLocation, SourceLookup};

    /// Parameter-shape helpers for bindings — wire-shaped values
    /// (JSON / protobuf-map / etc.) → `kglite::api::Value`. Future
    /// REST / gRPC bindings shouldn't re-implement the JSON
    /// dispatch each time; this re-export hands them the canonical
    /// converter.
    pub mod param {
        pub use crate::param::json_value_to_kglite_value;
    }

    /// Blueprint loader + builder — declarative graph construction
    /// from a YAML/JSON spec + a directory of CSVs. The wheel's
    /// `from_blueprint` is a thin ergonomics wrapper around
    /// [`load_blueprint_file`] + [`build`]; future bindings (Go,
    /// JS, JVM, …) call these directly.
    pub mod blueprint {
        pub use crate::graph::blueprint::build::{build, BuildReport, FlatSpec};
        pub use crate::graph::blueprint::schema::{
            load_blueprint_file, AggregateEdge, Blueprint, CalendarLink, ComputeOp, Connections,
            FkEdge, JunctionEdge, NodeSpec, Settings, TimeKey, TimeseriesSpec,
        };
    }

    /// Cypher parser + planner + executor primitives. Downstream
    /// consumers can build their own custom Cypher pipelines using
    /// these items; for the canonical pipeline see [`session`].
    pub mod cypher {
        pub use crate::graph::languages::cypher::ast::CypherQuery;
        pub use crate::graph::languages::cypher::ast::OutputFormat;
        pub use crate::graph::languages::cypher::executor::write::execute_mutable;
        pub use crate::graph::languages::cypher::executor::CypherExecutor;
        pub use crate::graph::languages::cypher::generate_explain_result;
        pub use crate::graph::languages::cypher::is_mutation_query;
        pub use crate::graph::languages::cypher::parse_with_mutation_check;
        pub use crate::graph::languages::cypher::parser::parse_cypher;
        pub use crate::graph::languages::cypher::planner;
        pub use crate::graph::languages::cypher::planner::mark_lazy_eligibility;
        pub use crate::graph::languages::cypher::planner::schema_check::validate_schema;
        pub use crate::graph::languages::cypher::planner::simplification::rewrite_text_score;
        pub use crate::graph::languages::cypher::result::CypherResult;
    }

    /// Canonical query + transaction surface — single source of
    /// truth for the Cypher pipeline + snapshot/working CoW
    /// transaction model. See `docs/explanation/session.md`.
    pub mod session {
        pub use crate::graph::session::{
            execute_mut, execute_read, resolve_noderefs, CommitOutcome, ExecuteOptions,
            ExecuteOutcome, Session, Transaction,
        };
    }

    /// Dataset fetch + extract building blocks for bindings that
    /// want to wrap SEC EDGAR, Sodir (Norwegian Continental Shelf),
    /// or Wikidata. Each submodule re-exports the same surface the
    /// Python wheel uses today via `_sec_internal` / `_sodir_internal`
    /// / `_wikidata_internal`; future Go / JS / JVM bindings consume
    /// the same items here through the stable api namespace.
    ///
    /// **Lifecycle orchestration is NOT in core.** The "fetch what's
    /// missing, build if cache is stale, return a ready-to-query
    /// graph" loop lives in each binding's wrapper (the Python
    /// wheel's wrappers at `kglite/datasets/*/wrapper.py` are the
    /// reference implementation). The engine ships the building
    /// blocks; bindings compose them in their own language idiom.
    /// See `docs/rust/implementing-a-binding.md` → "Wrapping a
    /// dataset for your binding" for the pattern.
    ///
    /// **All `fetch_*` entry points are `async`.** Bindings need a
    /// tokio runtime to drive them. The Python wheel builds one
    /// per call via `pyo3-async-runtimes`; a Rust binary can spin
    /// one up via `tokio::runtime::Builder`.
    pub mod datasets {
        /// SEC EDGAR — quarterly filings index, bulk submissions
        /// archive, per-form fetchers (Form 3/4/5, 13F, 8-K, SC 13D/G,
        /// DEF 14A, Form 144, Exhibit 21, XBRL company facts).
        #[cfg(feature = "sec")]
        pub mod sec {
            // Workdir layout + storage mode picker
            pub use crate::datasets::sec::{
                pick_storage_mode, predict_graph_size_gb, SliceSpec, StorageMode, Workdir,
                YearRange,
            };
            // Error type + the crate's Result alias
            pub use crate::datasets::sec::{Result, SecError};
            // HTTP client + fetch entry points (all async)
            pub use crate::datasets::sec::{
                fetch_13f_info_table, fetch_company_facts, fetch_company_submission,
                fetch_company_tickers, fetch_exhibit21_attachment, fetch_filing_primary_doc,
                fetch_form4_filing, fetch_quarterly_master_idx, fetch_submissions_bulk, FetchMode,
                SecClient,
            };
            // Sync wrappers (single-thread tokio runtime per call).
            // For bindings that don't manage their own async runtime.
            pub use crate::datasets::sec::{
                fetch_13f_info_table_blocking, fetch_company_facts_blocking,
                fetch_company_submission_blocking, fetch_company_tickers_blocking,
                fetch_exhibit21_attachment_blocking, fetch_filing_primary_doc_blocking,
                fetch_form4_filing_blocking, fetch_quarterly_master_idx_blocking,
                fetch_submissions_bulk_blocking,
            };
            // Form-type → per-filing-fetcher bucket mapping. Lifts
            // the wheel's `_FORM_BUCKETS` + `_resolve_fetch_buckets`
            // into core so every binding gets the same form-string
            // table without re-implementing it.
            pub use crate::datasets::sec::{
                all_buckets, resolve_fetch_buckets, SecFormBucket, ALL_BUCKETS, LEAN_FETCH_BUCKETS,
            };
            // Per-filing dispatch planning — reads filing_index.csv,
            // applies company / year / form-type filters, groups by
            // bucket. Every binding then drives its own execution
            // loop over the plan. Lifted from the wheel's
            // `_dispatch_per_filing_fetches` (CSV-reading + filtering
            // + grouping half) in the 2026-05-25 binding prep.
            pub use crate::datasets::sec::{
                prepare_dispatch_plan, DispatchPlan, DispatchScope, FilingTask,
            };
            // SEC company_tickers.json parser — turns the published
            // JSON into a `TICKER → CIK` HashMap for bindings that
            // accept string tickers from their users.
            pub use crate::datasets::sec::parse_tickers_json;
            // Extract pipeline (parses raw/ → processed/ CSVs)
            pub use crate::datasets::sec::{run_all, ExtractReport};
        }

        /// Sodir — Norwegian Continental Shelf petroleum data
        /// (fields, wells, prospects, licences, …) via the
        /// ArcGIS FactMaps REST API.
        #[cfg(feature = "sodir")]
        pub mod sodir {
            pub use crate::datasets::sodir::ArcGISClient;
            pub use crate::datasets::sodir::{Result, SodirError};
            pub use crate::datasets::sodir::{StorageMode, Workdir};
            // Single async fetch entry — pulls all referenced datasets
            // into csv/, applies preprocessing, returns the report.
            // The `*_blocking` variant for bindings without an async
            // runtime spins up a single-thread tokio runtime per call.
            pub use crate::datasets::sodir::{fetch_all, fetch_all_blocking, FetchAllReport};
            // Blueprint utilities the wheel composes with from_blueprint
            pub use crate::datasets::sodir::{datasets_used_by_blueprint, merge_blueprint_json};
        }

        /// Wikidata — resumable download of the
        /// `latest-truthy.nt.bz2` RDF dump. Building the graph from
        /// the dump is a separate concern (the wheel uses
        /// `KnowledgeGraph::load_ntriples`); this surface is the
        /// dump-management half only.
        #[cfg(feature = "wikidata")]
        pub mod wikidata {
            pub use crate::datasets::wikidata::Workdir;
            pub use crate::datasets::wikidata::{
                ensure_dump, ensure_dump_blocking, remote_last_modified,
                remote_last_modified_blocking,
            };
            pub use crate::datasets::wikidata::{Result, WikidataError};
            // Mirror config constants — bindings can read these to
            // tell users what file they'll end up with.
            pub use crate::datasets::wikidata::{DUMP_FILE, DUMP_URL};
            // Cache-freshness decision tree — every binding's
            // `open()` flow asks the same questions; the decision
            // lives in core, but each binding handles the outcome
            // (verbose prints, process-local cache hits, etc.) in
            // its own idiom. Lifted from `kglite/datasets/wikidata.py`
            // in the 2026-05-25 dataset-wrapper prep.
            pub use crate::datasets::wikidata::{
                age_days, decide, file_mtime_utc, read_remote_mtime_from_source_meta,
                CacheDecision, FreshnessInputs,
            };
        }
    }
}
