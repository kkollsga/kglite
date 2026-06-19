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
pub mod graphgen;
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
    /// Streaming synthetic-graph generator — `generate_to_dir(&config, dir)`
    /// streams the benchmark/demo graph as CSVs + a manifest in bounded memory.
    /// Surfaced through the wheel as `kglite.graphgen(...)`.
    pub use crate::graphgen::{generate_to_dir as graphgen, GraphGenConfig, GraphGenStats};
    // Inline timeseries config types (lifted from kglite-py in 0.10.1).
    pub use crate::graph::features::timeseries::{InlineTimeseriesConfig, TimeSpec};
    // Thin pure-Rust graph handle for embedders + the free function
    // backing it. The wheel crate (`kglite-py`) defines its own,
    // Python-flavored `KnowledgeGraph` separately — same name,
    // different audience (`pip install kglite` users), polars-style.
    //
    // `infer_selection_node_type` infers the node type of a selection's
    // current level; it takes `&CowSelection`, so it landed here in
    // Piece 3b alongside the Selection api-type lift (Piece 3a).
    //
    // `resolve_code_entity` + `CODE_TYPES` are the code-tree graph
    // helpers (resolve a `Type::method` qualified name to a node;
    // the canonical set of code-entity node-type labels) — generic
    // for any binding that ships the code-tree parser. Lifted in the
    // api-sealing soft-seal (roadmap Piece 1).
    pub use crate::graph::handle::{
        discover_property_keys_from_data, infer_selection_node_type, resolve_code_entity,
        source_location, KnowledgeGraph, CODE_TYPES,
    };
    /// The fluent **selection** data model — the cursor state threaded
    /// through the fluent query chain (and through Selection-scoped
    /// capabilities like `algorithms::vector_search`, `mutation`
    /// set-ops/subgraph, and the spatial predicates). `CowSelection` is
    /// the Arc copy-on-write wrapper a binding holds as its cursor;
    /// `CurrentSelection` is the underlying level/plan state; `PlanStep`
    /// is an `explain()` plan entry. Pure core types (petgraph node
    /// indices and hash maps), no binding coupling. Lifted in roadmap
    /// Piece 3a as the foundation for the fluent api surface. The
    /// high-level fluent chain operations are consolidated into core and
    /// exposed in Piece 3c; the fine-grained `core::*` primitives stay
    /// internal.
    pub use crate::graph::schema::{
        CowSelection, CurrentSelection, PlanStep, SelectionLevel, SelectionOperation,
    };
    /// Interned property-/type-key handle (a transparent `u64` newtype).
    /// Bindings doing low-level direct graph access bridge between string
    /// keys and the engine's interned ids via `InternedKey::from_str(..)` /
    /// `.as_u64()`. Lifted in roadmap Piece 2.
    pub use crate::graph::storage::interner::InternedKey;
    /// The canonical graph read trait — node/edge/property accessors
    /// shared by every storage backend. Non-object-safe (GATs on the
    /// iterator-returning methods), so consumers take `&impl GraphRead`,
    /// never `&dyn`. Lifted for cross-binding read access (roadmap Piece 1).
    pub use crate::graph::storage::GraphRead;
    // `Arc<DirGraph>` → `&mut DirGraph` + version bump (lifted in 0.10.1).
    pub use crate::graph::dir_graph::make_dir_graph_mut;
    pub use crate::graph::introspection::describe::compute_description;
    /// Structured mutation reports — what a write touched (nodes/edges
    /// created/updated/deleted, per operation). Every binding surfaces
    /// these after a mutating call; lifted for cross-binding result
    /// reporting (roadmap Piece 1; the per-op `NodeOperationReport` /
    /// `ConnectionOperationReport` return types added in Piece 2 alongside
    /// the bulk-mutation functions that produce them).
    pub use crate::graph::introspection::reporting::{
        ConnectionOperationReport, NodeOperationReport, OperationReport, OperationReports,
    };
    pub use crate::graph::introspection::schema_overview::compute_schema;
    pub use crate::graph::introspection::SchemaOverview;
    pub use crate::graph::introspection::{ConnectionDetail, CypherDetail, FluentDetail};
    pub use crate::graph::io::file::{
        load_file, load_kgl_bytes, save_graph, write_kgl, write_kgl_to, write_kgl_with,
    };
    pub use crate::graph::{SourceLocation, SourceLookup};

    /// Parameter-shape helpers for bindings — wire-shaped values
    /// (JSON / protobuf-map / etc.) ↔ `kglite::api::Value`. Future
    /// REST / gRPC bindings shouldn't re-implement the JSON dispatch
    /// each time; these re-exports hand them the canonical converters
    /// for both directions: `json_value_to_kglite_value` (inbound
    /// params) and `kglite_value_to_json` (outbound result cells, in
    /// natural untagged JSON).
    pub mod param {
        pub use crate::param::{json_value_to_kglite_value, kglite_value_to_json};
    }

    /// Bulk graph construction + maintenance. `add_edges_from_specs` is
    /// the DataFrame-free edge-ingest path that non-Python bindings use
    /// (the C ABI's `create_edges_batch` wraps it); the DataFrame-based
    /// `add_nodes` / `add_connections` / `replace_connections` are the
    /// Rust-side bulk-ingest path (polars `DataFrame` in, operation report
    /// out). `update_node_properties`, `purge_provisional_nodes`, and
    /// `extend_graph` (merge one graph into another) round out the
    /// generic, non-Selection mutation surface. Lifted in roadmap Piece 2.
    /// `create_connections` (edge-create between the two ends of a
    /// selection) lifted in Piece 3b once `CurrentSelection` reached api.
    pub mod mutation {
        pub use crate::graph::mutation::extend::{extend_graph, ExtendReport};
        pub use crate::graph::mutation::maintain::{
            add_connections, add_edges_from_specs, add_nodes, create_connections,
            purge_provisional_nodes, replace_connections, update_node_properties, EdgeSpec,
            EdgeSpecReport,
        };
    }

    /// Selection-scoped operations — selection set algebra
    /// (`union`/`intersection`/`difference`/`symmetric_difference`) and
    /// subgraph extract / expand / stats. These take `&CurrentSelection`
    /// (now an api type, roadmap Piece 3a) and are the building blocks the
    /// fluent chain composes.
    ///
    /// The bulk of this module (Piece 3c) is the **shared selection-based
    /// query-primitive layer** — `core::graph::core::*`, which CLAUDE.md
    /// describes as "pattern matching, filtering, traversal … used by both
    /// Cypher and the fluent API." Each op takes `(&DirGraph, &mut
    /// CurrentSelection, …already-marshalled params)` and mutates the
    /// selection in place; a binding building a fluent surface composes
    /// these directly (the wheel's `kg_fluent` / `kg_introspection` PyO3
    /// methods marshal Python args, then call straight into here). The
    /// primitives stay *defined* in `core::graph::core`; this is their
    /// curated, stable re-export surface. (A future refinement could hoist
    /// the small amount of per-method branching — `select`'s
    /// include-secondary / temporal logic, `traverse`'s temporal precedence
    /// — into higher-level ops, but the primitives below are already the
    /// correctly-grained shared operations, not glue to hide.)
    pub mod fluent {
        // Selection set algebra + subgraph (Piece 3b).
        pub use crate::graph::mutation::set_ops::{
            difference_selections, intersection_selections, symmetric_difference_selections,
            union_selections,
        };
        pub use crate::graph::mutation::subgraph::{
            expand_selection, extract_subgraph, get_subgraph_stats, SubgraphStats,
        };
        // Filtering / sorting / pagination over a selection.
        pub use crate::graph::core::filtering::{
            filter_by_connection, filter_nodes, filter_nodes_any, filter_nodes_by_label,
            filter_orphan_nodes, limit_nodes_per_group, offset_nodes, sort_nodes,
        };
        // Traversal (parent→child level expansion) + its config/filter types.
        pub use crate::graph::core::traversal::{
            format_for_dictionary, format_for_storage, get_children_properties,
            make_comparison_traversal, make_traversal, MethodConfig, TemporalEdgeFilter,
        };
        // Per-level calculations / equation evaluation / counts.
        pub use crate::graph::core::calculations::{
            count_nodes_by_parent, count_nodes_in_level, process_equation, store_count_results,
            EvaluationResult, StatResult,
        };
        // Node/connection/property retrieval from a selection + result types.
        pub use crate::graph::core::data_retrieval::{
            format_unique_values_for_storage, get_connections, get_nodes, get_property_values,
            get_unique_values, LevelConnections, LevelNodes, LevelValues, UniqueValues,
        };
        // Aggregate statistics over selected nodes.
        pub use crate::graph::core::statistics::{
            calculate_property_stats, collect_selected_nodes, get_parent_child_pairs, PropertyStats,
        };
        // Pattern-match execution (shared with Cypher MATCH).
        pub use crate::graph::core::pattern_matching::{
            parse_pattern, MatchBinding, PatternExecutor, PatternMatch,
        };
        // Compact value formatting for fluent result shaping.
        pub use crate::graph::core::value_operations::format_value_compact;
    }

    /// Graph algorithms — pathfinding, components, centrality, community
    /// detection (the typed, direct-call surface). Every binding that
    /// exposes a typed `shortest_path()` / `pagerank()` / `louvain()`
    /// method reaches these; they all take `&DirGraph` + plain params and
    /// return the result structs below. (Per-query algorithm access is
    /// also available through Cypher procedures; this is the typed-result
    /// path for bindings that want structs, not result rows.) Lifted in
    /// api-sealing roadmap Piece 2 (`vector_search` + `VectorSearchResult`
    /// added in Piece 3b once `CurrentSelection` was lifted to api — vector
    /// search is scoped to a selection).
    pub mod algorithms {
        pub use crate::graph::algorithms::graph_algorithms::{
            all_paths, are_connected, betweenness_centrality, closeness_centrality,
            connected_components, degree_centrality, get_node_info, get_path_connections,
            label_propagation, louvain_communities, node_degree, pagerank, shortest_path,
            shortest_path_cost, shortest_path_cost_batch, shortest_path_cost_weighted,
            shortest_path_weighted, weakly_connected_components, CentralityResult, CommunityResult,
            PathNodeInfo, PathResult,
        };
        pub use crate::graph::algorithms::hnsw::HnswParams;
        pub use crate::graph::algorithms::vector::{
            vector_search, DistanceMetric, VectorSearchResult,
        };
    }

    /// Timeseries date/query helpers — the pure date-parsing and
    /// range-finding utilities behind inline timeseries support.
    /// `parse_date_query` ("2013" / "2010..2015" → `NaiveDate` +
    /// `DatePrecision`), `expand_end`, `date_from_ymd`, `find_range`, and
    /// the validators are plain functions every binding's date handling
    /// reaches; `TimeseriesConfig` / `NodeTimeseries` are the config/data
    /// types. Lifted in roadmap Piece 2. (The KG-construction-level
    /// `InlineTimeseriesConfig` / `TimeSpec` live in the api root.)
    pub mod timeseries {
        pub use crate::graph::features::timeseries::{
            date_from_ymd, expand_end, find_range, parse_date_query, validate_channel_length,
            validate_keys_sorted, validate_resolution, DatePrecision, NodeTimeseries,
            TimeseriesConfig,
        };
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
        /// Operator-declared value codecs — position-scoped, bidirectional
        /// literal conversions (`'Q42'` ↔ `42`) bound to a property. Bindings
        /// build a `Vec<ValueCodec>` (e.g. from a YAML manifest) and pass it via
        /// `session::ExecuteOptions::value_codecs`. See `value_codec` module
        /// docs for the safety model.
        pub use crate::graph::languages::cypher::value_codec::{CodecKind, StoredType, ValueCodec};
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
