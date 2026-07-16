//! kglite — pure-Rust knowledge graph engine.
//!
//! Cypher pipeline, snapshot/working CoW transactions, columnar /
//! mmap / disk storage backends, and optional format loaders (RDF,
//! OKF). Pre-packaged domain dataset loaders live in the separate
//! kglite-datasets project. The Python wheel (`pip install kglite`)
//! is built by the sibling `kglite-py` crate; the Bolt and MCP
//! protocol servers are separate workspace binaries.
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

pub mod datatypes;
pub mod error;
// Engine internals — sealed behind the curated `api` facade (roadmap Piece 4).
// `pub(crate)` so no downstream crate can reach `kglite::graph::*` directly;
// the `api` re-exports below still resolve (re-exporting a `pub` item out of a
// `pub(crate)` module is legal). A CI grep (`scripts/check_api_chokepoint.sh`)
// keeps the wrapper crates honest.
pub(crate) mod graph;
pub mod graphgen;
#[cfg(feature = "okf")]
pub mod okf;
pub mod param;
pub(crate) mod serde_codec;

#[cfg(test)]
mod bincode_wire_contract_tests;

/// Curated stable Rust API. Downstream consumers should depend on
/// items here, not on the underlying module structure (which may
/// move between minor releases).
pub mod api {
    // ── Root prelude ──────────────────────────────────────────────────────
    // The root holds only the cross-cutting *data model* (the types every
    // binding speaks) + a couple of standalone top-level capabilities
    // (`graphgen`, `explore_markdown`). Everything else is clustered into a
    // submodule by concern: `param`, `mutation`, `fluent`, `algorithms`,
    // `timeseries`, `introspection`, `io`, `blueprint`,
    // `cypher`, `session`. Per-cluster items live in exactly one
    // place (no root↔submodule duplication).
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
    // Thin pure-Rust graph handle for embedders + the free function
    // backing it. The wheel crate (`kglite-py`) defines its own,
    // Python-flavored `KnowledgeGraph` separately — same name,
    // different audience (`pip install kglite` users), polars-style.
    //
    // `infer_selection_node_type` infers the node type of a selection's
    // current level; it takes `&CowSelection`, so it landed here in
    // Piece 3b alongside the Selection api-type lift (Piece 3a).
    //
    // (The code-tree handle helpers `resolve_code_entity` / `CODE_TYPES` /
    // `source_location` live in `api::code_entities`.)
    pub use crate::graph::handle::{
        discover_property_keys_from_data, infer_selection_node_type, KnowledgeGraph,
    };
    /// Core schema data types — the node record (`NodeData`), the projected
    /// `NodeInfo`, geo/temporal validity configs (`SpatialConfig` /
    /// `TemporalConfig`), and the declarative schema-definition +
    /// validation types. Generic across bindings; lifted in roadmap
    /// Piece 3 cleanup.
    pub use crate::graph::schema::{
        parse_spatial_column_types_from_pairs, parse_temporal_column_types_from_pairs,
        ConnectionSchemaDefinition, NodeData, NodeInfo, NodeSchemaDefinition, SchemaDefinition,
        SpatialConfig, TemporalConfig, ValidationError,
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
    /// Interned property-/type-key handle (`InternedKey`, a transparent
    /// `u64` newtype) + the `StringInterner` that mints them. Bindings
    /// doing low-level direct graph access bridge string keys ↔ interned
    /// ids via `InternedKey::from_str(..)` / `.as_u64()`. Lifted in roadmap
    /// Piece 2 / Piece 3 cleanup.
    pub use crate::graph::storage::interner::{InternedKey, InternerCollision, StringInterner};
    /// The canonical graph read trait — node/edge/property accessors
    /// shared by every storage backend. Non-object-safe (GATs on the
    /// iterator-returning methods), so consumers take `&impl GraphRead`,
    /// never `&dyn`. Lifted for cross-binding read access (roadmap Piece 1).
    pub use crate::graph::storage::GraphRead;
    /// The temporal query context (`At` / `During` / `Today` / `All`) — the
    /// as-of filter a binding's cursor carries for temporal-validity
    /// auto-filtering. Lifted in roadmap Piece 4.
    pub use crate::graph::TemporalContext;
    // `Arc<DirGraph>` → `&mut DirGraph` + version bump (lifted in 0.10.1).
    pub use crate::graph::handle::make_dir_graph_mut;
    // (Mutation reports → `api::mutation`; schema introspection /
    // `SchemaOverview` / detail enums → `api::introspection`; `.kgl`
    // load/save → `api::io`; `SourceLocation`/`SourceLookup` →
    // `api::code_entities`.)

    /// Parameter-shape helpers for bindings — wire-shaped values
    /// (JSON / protobuf-map / etc.) ↔ `kglite::api::Value`. Future
    /// REST / gRPC bindings shouldn't re-implement the JSON dispatch
    /// each time; these re-exports hand them the canonical converters
    /// for both directions: `json_value_to_kglite_value` (inbound
    /// params) and `kglite_value_to_json` (outbound result cells, in
    /// natural untagged JSON).
    pub mod param {
        pub use crate::param::{
            json_object_to_value_map, json_value_to_kglite_value, kglite_value_to_json,
        };
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
        /// Structured mutation reports — what a write touched (nodes/edges
        /// created/updated/deleted, per operation). Returned by the mutation
        /// functions above; every binding surfaces them after a mutating call.
        pub use crate::graph::introspection::reporting::{
            ConnectionOperationReport, NodeOperationReport, OperationReport, OperationReports,
        };
        pub use crate::graph::mutation::extend::{extend_graph, ExtendReport};
        pub use crate::graph::mutation::maintain::{
            add_connections, add_edges_from_specs, add_nodes, add_properties, create_connections,
            purge_provisional_nodes, replace_connections, update_node_properties, EdgeSpec,
            EdgeSpecReport, PropertySpec,
        };
        /// Validate a graph against a `SchemaDefinition` (Piece 3 cleanup).
        pub use crate::graph::mutation::validation::validate_graph;
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
            calculate_grouped_property_stats, calculate_property_stats, collect_selected_nodes,
            get_parent_child_pairs, GroupedPropertyStats, PropertyStats,
        };
        // Pattern-match execution (shared with Cypher MATCH).
        pub use crate::graph::core::pattern_matching::{
            parse_pattern, MatchBinding, PatternExecutor, PatternMatch,
        };
        // Compact value formatting for fluent result shaping.
        pub use crate::graph::core::value_operations::format_value_compact;
        // Spatial predicates over a selection (geo filters / centroids /
        // bounds). Selection-scoped — lifted in Piece 3 cleanup now that
        // CurrentSelection is an api type.
        pub use crate::graph::features::spatial::{
            calculate_centroid, contains_point, get_bounds, intersects_geometry, near_point,
            near_point_m, within_bounds, wkt_centroid,
        };
        // Temporal validity predicates (per NodeData + TemporalConfig).
        pub use crate::graph::features::temporal::{
            node_is_temporally_valid, node_overlaps_range, node_passes_context,
        };
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
            shortest_path_weighted, weakly_connected_components, AllPathsOptions,
            CentralityOptions, CentralityResult, CommunityOptions, CommunityResult,
            DegreeCentralityOptions, LabelPropagationOptions, PagerankOptions, PathNodeInfo,
            PathOptions, PathResult,
        };
        pub use crate::graph::algorithms::hnsw::HnswParams;
        pub use crate::graph::algorithms::vector::{
            vector_search, DistanceMetric, VectorSearchOptions, VectorSearchResult,
        };
        pub use crate::graph::algorithms::Interrupt;
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
            validate_keys_sorted, validate_resolution, DatePrecision, InlineTimeseriesConfig,
            NodeTimeseries, TimeSpec, TimeseriesConfig,
        };
    }

    /// Schema/graph introspection — the compute primitives behind
    /// `describe()` / schema overview (connectivity, per-type stats,
    /// neighbor schema) + the detail-level enums + a bug-report writer.
    /// The typed schema-discovery surface every binding builds its
    /// agent-facing schema from. Lifted in roadmap Piece 3 cleanup.
    pub mod introspection {
        pub use crate::graph::introspection::bug_report::write_bug_report;
        /// Debug-string helpers (schema / selection dumps) for diagnostics.
        pub use crate::graph::introspection::debugging;
        pub use crate::graph::introspection::describe::{compute_description, mcp_quickstart};
        pub use crate::graph::introspection::schema_overview::{
            compute_connection_type_stats, compute_neighbors_schema, compute_property_stats,
            compute_schema,
        };
        pub use crate::graph::introspection::{
            compute_type_connectivity, derive_edge_counts_from_triples, schema_overview_to_json,
            ConnectionDetail, ConnectionTypeStats, CypherDetail, FluentDetail, SchemaOverview,
            EXACT_PROPERTY_STATS_MAX_NODES,
        };
    }

    /// Graph I/O: `.kgl` load/save, format exporters (GraphML / GEXF /
    /// D3-JSON / CSV), the N-Triples (RDF) streaming loader + progress
    /// callbacks, embedding-vector file export/import, and streaming
    /// disk subset export.
    pub mod io {
        pub use crate::graph::io::export::{
            to_csv, to_csv_dir, to_d3_json, to_gexf, to_graphml, to_text,
        };
        /// Embedding-vector file export / import.
        pub use crate::graph::io::file::{
            export_embeddings_to_file, import_embeddings_from_file, EmbeddingExportFilter,
            ImportStats,
        };
        /// `.kgl` load / save (the canonical persistence format).
        pub use crate::graph::io::file::{
            load_file, load_kgl_bytes, prepare_save, save_graph, save_graph_with, write_kgl,
            write_kgl_to, write_kgl_with,
        };
        pub use crate::graph::io::ntriples::{
            load_ntriples, Cancelled, NTriplesConfig, ProgressEvent, ProgressSink, ProgressValue,
        };
        pub use crate::graph::io::open::{
            open_or_create_graph, GraphFileIdentity, GraphWriterLease, OpenDisposition,
            OpenGraphResult,
        };
        /// General-purpose RDF loader (Turtle / N-Triples / N-Quads /
        /// TriG). Gated behind the `rdf` Cargo feature.
        #[cfg(feature = "rdf")]
        pub use crate::graph::io::rdf::{load_rdf, RdfConfig, RdfStats};
        /// Streaming disk subset export (bounded-memory subgraph save).
        pub use crate::graph::mutation::subgraph_streaming::{
            pass_a_scan, pass_a_scan_to_file, save_subset, save_subset_streaming_disk, RankIndex,
            SubsetSpec,
        };
    }

    /// Storage backend configuration — the in-memory / mmap / disk backends
    /// (`GraphBackend` + `DiskGraph` / `MappedGraph` constructors), the
    /// per-type lookup, and the embedding store. CLAUDE.md designates
    /// storage-backend configuration a direct-api concern; these let a
    /// binding open / inspect a graph in a specific storage mode and manage
    /// embeddings. Lifted in roadmap Piece 4 (the hard-seal gateway).
    pub mod storage {
        pub use crate::graph::schema::EmbeddingStore;
        pub use crate::graph::storage::backend::GraphBackend;
        pub use crate::graph::storage::disk::graph::DiskGraph;
        pub use crate::graph::storage::lookups::TypeLookup;
        /// The cross-binding create-in-mode builder: resolve a mode string to
        /// a [`StorageMode`] and build a fresh graph in that backend. Shared by
        /// the wheel (`storage='mapped'/'disk'`), the bolt/mcp servers
        /// (`--storage`), and the C ABI (`kglite_graph_new_in_mode`).
        pub use crate::graph::storage::mode::{new_dir_graph_in_mode, StorageMode};
        pub use crate::graph::storage::MappedGraph;
    }

    /// Durable transactions — the write-ahead log (append / recover / replay)
    /// and the write-capture recording layer behind a binding's `durable()`
    /// feature. The in-process WAL mechanism (distinct from the checkpoint
    /// save in `io`). Lifted in roadmap Piece 4.
    pub mod durable {
        pub use crate::graph::mutation::wal_replay::apply_frames;
        pub use crate::graph::storage::recording::{resolve_ops, RecordingGraph};
        pub use crate::graph::wal::{recover, wal_path, Wal, WalFrame};
    }

    /// Code-entity read surface — resolve / locate / contextualize entities
    /// (`Type::method` helpers + source-location types) on any graph with
    /// the code schema (Function/Class/… nodes carrying `file_path`/`line`).
    /// Defined on the graph handle, independent of the builder: graphs built
    /// by an external builder (codingest) get the same surface.
    pub mod code_entities {
        pub use crate::graph::handle::{
            code_entity_context, find_code_entities, resolve_code_entity, source_location,
            CodeContextLookup, CodeEntityContext, CodeEntityMatch, CODE_TYPES,
        };
        pub use crate::graph::{SourceLocation, SourceLookup};
    }

    /// Blueprint loader + builder — declarative graph construction
    /// from a YAML/JSON spec + a directory of CSVs. The wheel's
    /// `from_blueprint` is a thin ergonomics wrapper around
    /// [`load_blueprint_file`] + [`build`]; future bindings (Go,
    /// JS, JVM, …) call these directly.
    pub mod blueprint {
        pub use crate::graph::blueprint::build::{build, BuildReport, FlatSpec};
        pub use crate::graph::blueprint::json_records::{from_records, RecordsReport};
        pub use crate::graph::blueprint::schema::{
            load_blueprint_file, AggregateEdge, Blueprint, CalendarLink, ComputeOp, Connections,
            FkEdge, JunctionEdge, NodeSpec, Settings, TimeKey, TimeseriesSpec,
        };
    }

    /// Cypher parser + planner + executor primitives. Downstream
    /// consumers can build their own custom Cypher pipelines using
    /// these items; for the canonical pipeline see [`session`].
    pub mod cypher {
        pub use crate::graph::languages::cypher::ast::{
            CypherQuery, Expression, OutputFormat, ReturnItem,
        };
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
        pub use crate::graph::languages::cypher::result::{
            materialise_lazy, materialise_lazy_range, materialise_lazy_row, CypherResult,
            LazyResultDescriptor,
        };
        /// Operator-declared value codecs — position-scoped, bidirectional
        /// literal conversions (`'Q42'` ↔ `42`) bound to a property. Bindings
        /// build a `Vec<ValueCodec>` (e.g. from a YAML manifest) and pass it via
        /// `session::ExecuteOptions::value_codecs`. See `value_codec` module
        /// docs for the safety model.
        pub use crate::graph::languages::cypher::value_codec::{CodecKind, StoredType, ValueCodec};
        // Specific Cypher-pipeline items a binding implementing a native
        // `cypher()` method (the wheel) reaches. Exposed INDIVIDUALLY — not as
        // whole `ast`/`executor`/`parser`/`result` submodules — so the rest of
        // the executor/parser internals stay un-exported and the optimizer can
        // keep inlining the per-query hot path. (Re-exporting the whole
        // executor module measurably regressed cypher micro-query latency by
        // ~60% on tiny graphs — roadmap Piece 4 perf follow-up.)
        pub use crate::graph::languages::cypher::executor::helpers::{
            resolve_edge_property, resolve_node_property,
        };
        pub use crate::graph::languages::cypher::optimize;
        pub use crate::graph::languages::cypher::planner::schema_check::collect_unknown_pattern_warnings;
        pub use crate::graph::languages::cypher::result::{
            ClauseStats, EdgeBinding, MutationStats, QueryDiagnostics, ResultRow,
        };
    }

    /// Canonical query + transaction surface — single source of
    /// truth for the Cypher pipeline + snapshot/working CoW
    /// transaction model. See `docs/rust/session.md`.
    pub mod session {
        pub use crate::graph::session::{
            execute_mut, execute_read, resolve_noderefs, CommitOutcome, ExecuteOptions,
            ExecuteOutcome, Session, Transaction,
        };
    }
}
