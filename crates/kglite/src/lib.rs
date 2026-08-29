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
//! Downstream Rust consumers (the Python wheel, the C ABI, the
//! CLI, the bolt and mcp server binaries) should depend on the
//! curated [`api`] module — it is the documented surface, locked
//! against accidental drift by a CI API baseline. Pre-1.0 that is
//! not a no-break promise: any release, **including a patch**, may
//! ship a documented breaking change, announced in `CHANGELOG.md`.
//! Anything outside [`api`] is an implementation detail. Non-Rust
//! bindings (Java today) reach that same surface through the C ABI.
//!
//! See `docs/rust/embedding.md` for the embedder guide.

pub mod datatypes;
pub mod error;
// Engine internals — sealed behind the curated `api` facade.
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

/// Curated stable Rust API. Downstream consumers should depend on
/// items here, not on the underlying module structure (which may
/// move between minor releases).
pub mod api {
    // ── Root prelude ──────────────────────────────────────────────────────
    // The root holds only the cross-cutting *data model* (the types every
    // binding speaks) plus the standalone top-level capabilities; everything
    // else is clustered into a submodule by concern. Each item lives in
    // exactly one of them — no root↔submodule duplication.
    pub use crate::datatypes::values::{NodeValue, PathValue, RelValue};
    /// The property container `Value::Map`, `NodeValue::properties`,
    /// `RelValue::properties` and `mutation::ColumnData::Map` all carry: an
    /// `Arc`'d flat map, sorted by key and duplicate-free (every read binary-
    /// searches on that, and `Eq`/`Ord`/`Hash`/`Serialize` rely on it for
    /// determinism). Keys are plain `String`.
    pub use crate::datatypes::PropMap;
    pub use crate::datatypes::Value;
    pub use crate::error::{KgError, KgErrorCode};
    pub use crate::graph::dir_graph::DirGraph;
    /// The storage-health report `DirGraph::graph_info` returns: live vs
    /// tombstoned node and edge slots, columnar row and heap totals, index
    /// counts, and the two mmap flags. `fragmentation_ratio` is the node-slot
    /// reading only — the auto-vacuum trigger takes the worst of all three
    /// garbage populations.
    pub use crate::graph::dir_graph::GraphInfo;
    /// The old→new node-index mapping `DirGraph::vacuum` returns.
    pub use crate::graph::dir_graph::NodeRemap;
    #[cfg(feature = "fastembed")]
    pub use crate::graph::embedder::fastembed::FastEmbedAdapter;
    pub use crate::graph::embedder::Embedder;
    pub use crate::graph::explore::{explore_markdown, ExploreOptions};
    /// Streaming synthetic-graph generator — `generate_to_dir(&config, dir)`
    /// streams the benchmark/demo graph as CSVs + a manifest in bounded memory.
    /// Surfaced through the wheel as `kglite.graphgen(...)`.
    pub use crate::graphgen::{generate_to_dir as graphgen, GraphGenConfig, GraphGenStats};
    /// The petgraph types this API's own signatures already speak.
    ///
    /// `NodeIndex` and `EdgeIndex` are the slot handles every `GraphRead` /
    /// `GraphWrite` call takes and returns; `Direction` is the in/out argument
    /// on every adjacency call (`edges_directed`, `count_edges_filtered`,
    /// `fluent::filter_by_connection`). Re-exported so a consumer need not add
    /// its own `petgraph` dependency pinned to *the same major* the engine
    /// links — a mismatch there is a type error at the call site, not a version
    /// warning, so the coupling is the engine's to carry.
    pub use petgraph::graph::{EdgeIndex, NodeIndex};
    pub use petgraph::Direction;
    // Thin pure-Rust graph handle for embedders + the free functions backing
    // it. The wheel crate (`kglite-py`) defines its own, Python-flavored
    // `KnowledgeGraph` separately — same name, different audience
    // (`pip install kglite` users), polars-style.
    /// The declared semantic layer (classes with an `is_a` forest +
    /// relationship semantics) and its one external dialect — same
    /// chokepoint posture as `schema_from_value`: Python dicts and C-ABI
    /// JSON parse through the identical grammar. Annotations, not axioms.
    pub use crate::graph::dir_graph::ontology_apply::MaterializedLabel;
    pub use crate::graph::handle::{
        discover_property_keys_excluding, discover_property_keys_from_data,
        infer_selection_node_type, is_canonical_node_column, KnowledgeGraph,
        CANONICAL_NODE_COLUMNS,
    };
    pub use crate::graph::ontology::{
        ontology_from_json, ontology_from_value, CardinalityDecl, ClassDecl, Enforcement,
        ManagedLabelState, OntologyStore, RelationshipDecl, MAX_ONTOLOGY_CLASSES,
    };
    /// Core schema data types — the node and edge records (`NodeData` /
    /// `EdgeData`), the projected `NodeInfo`, geo/temporal validity configs
    /// (`SpatialConfig` / `TemporalConfig`), and the declarative
    /// schema-definition + validation types. Generic across bindings.
    /// `EdgeData` is included because `GraphWrite::add_edge` and
    /// `DiskGraph::from_stable_digraph` name it in public signatures, so it
    /// must be publicly nameable too.
    pub use crate::graph::schema::{
        parse_spatial_column_types_from_pairs, parse_temporal_column_types_from_pairs,
        ConnectionSchemaDefinition, EdgeData, NodeData, NodeInfo, NodeSchemaDefinition,
        SchemaDefinition, SchemaInstall, SpatialConfig, TemporalConfig, ValidationError,
    };
    /// The fluent **selection** data model — the cursor state threaded
    /// through the fluent query chain (and through Selection-scoped
    /// capabilities like `algorithms::vector_search`, `fluent`
    /// set-ops/subgraph, and the spatial predicates). `CowSelection` is
    /// the Arc copy-on-write wrapper a binding holds as its cursor;
    /// `CurrentSelection` is the underlying level/plan state; `PlanStep`
    /// is an `explain()` plan entry. Pure core types (petgraph node
    /// indices and hash maps), no binding coupling. The operations that
    /// consume them live in `api::fluent`.
    pub use crate::graph::schema::{
        CowSelection, CurrentSelection, PlanStep, SelectionLevel, SelectionOperation,
    };
    /// The single external schema **dialect** — the `{"nodes": {...},
    /// "connections": {...}}` shape users write — and its parser. Every
    /// binding's `define_schema` routes through here rather than hand-walking
    /// its own dict: the Python wheel converts its dict to a [`Value`] and
    /// calls `schema_from_value`, the C ABI's `kglite_define_schema` parses
    /// JSON with `schema_from_json`. Keeping one grammar matters most for the
    /// C ABI, where a published signature can never change within a major, so
    /// a second dialect would be permanent. `SchemaParseErrorKind` lets a
    /// binding raise its own conventional exception class.
    pub use crate::graph::schema_json::{
        schema_from_json, schema_from_value, SchemaParseError, SchemaParseErrorKind,
    };
    /// Arena guard for direct `GraphRead` traversals on disk-backed graphs.
    /// Acquire via [`DirGraph::begin_read_pass`] and keep alive while
    /// borrowed node/edge weights live; `None` on memory/mapped backends.
    pub use crate::graph::storage::disk::graph::DiskQueryGuard;
    /// Interned property-/type-key handle (`InternedKey`, a transparent
    /// `u64` newtype) + the `StringInterner` that mints them.
    /// `InternedKey::from_str(..)` computes the hash **without registering
    /// the name** — safe for lookups and removals, but a key that reaches a
    /// *write* unregistered reads back in-session and then breaks
    /// enumeration and persistence (the name cannot be resolved). For
    /// writes, use `DirGraph::set_node_property` (registers for you) or
    /// register via `StringInterner::try_get_or_intern` first.
    pub use crate::graph::storage::interner::{InternedKey, InternerCollision, StringInterner};
    /// The canonical graph read trait — node/edge/property accessors
    /// shared by every storage backend. Non-object-safe (GATs on the
    /// iterator-returning methods), so consumers take `&impl GraphRead`,
    /// never `&dyn`. Lifted for cross-binding read access.
    pub use crate::graph::storage::GraphRead;
    /// The canonical graph write trait (`GraphWrite: GraphRead`) —
    /// storage-variant-routed mutation, including `set_node_property` and its
    /// siblings. Non-object-safe like `GraphRead`: consumers take
    /// `&mut impl GraphWrite`, never `&mut dyn`. Implemented by the storage
    /// backends — reach it as `graph.graph.set_node_property(..)` on a
    /// `DirGraph` (the `graph` field is public), the same call the Cypher
    /// `SET` executor makes. Bridge string keys via
    /// `StringInterner::try_get_or_intern` (`DirGraph::interner` is public).
    pub use crate::graph::storage::GraphWrite;
    /// The authoritative read handle for one node's properties.
    ///
    /// A `&NodeData` is one *replica* of a columnar type's column store;
    /// `NodeView` is the store the storage backend answers with, and its
    /// enumeration methods are complete for every storage variant.
    /// Obtain one from `GraphRead::node_view` / `DirGraph::node_view`; do not
    /// hold it across a `Python::attach` boundary — resolve to owned values
    /// first. See `crates/kglite/src/graph/storage/node_view.rs`.
    pub use crate::graph::storage::NodeView;
    /// Structured-data support over the list/map substrate: table-property
    /// fidelity metadata + declared property shapes (`list<map{...}>`).
    pub use crate::graph::tables::{
        parse_property_shape, table_meta_key, PropertyShape, ScalarShape, TablePropertyMeta,
    };
    /// The temporal query context (`At` / `During` / `Today` / `All`) — the
    /// as-of filter a binding's cursor carries for temporal-validity
    /// auto-filtering.
    pub use crate::graph::TemporalContext;
    // `Arc<DirGraph>` → `&mut DirGraph` + version bump.
    pub use crate::graph::handle::make_dir_graph_mut;

    /// The item type of every [`GraphRead`] edge iterator (`EdgesIter`,
    /// `EdgeReferencesIter`, `EdgesConnectingIter`), so a consumer that stores
    /// or returns one can name it. Its inherent `source()` / `target()` /
    /// `id()` / `weight()` mirror `petgraph::visit::EdgeRef`, so no trait
    /// import is needed. Prefer `connection_type()` over
    /// `weight().connection_type` in traversals: on the disk backend `weight()`
    /// materialises the [`EdgeData`], `connection_type()` reads the CSR
    /// endpoint table.
    pub use crate::graph::core::iterators::GraphEdgeRef;

    /// Parameter-shape helpers for bindings — wire-shaped values
    /// (JSON / protobuf-map / etc.) ↔ `kglite::api::Value`. The
    /// canonical converters both ways, so no binding re-implements the
    /// JSON dispatch: `json_value_to_kglite_value` (inbound params) and
    /// `kglite_value_to_json` (outbound result cells, in natural
    /// untagged JSON).
    pub mod param {
        pub use crate::param::{
            json_object_to_value_map, json_value_to_kglite_value, kglite_value_to_json,
        };
    }

    /// Bulk graph construction + maintenance. `add_edges_from_specs` is
    /// the DataFrame-free edge-ingest path that non-Python bindings use
    /// (the C ABI's `kglite_create_edges_batch` wraps it); the DataFrame-based
    /// `add_nodes` / `add_connections` / `replace_connections` are the
    /// Rust-side bulk-ingest path (`DataFrame` in, operation report out).
    /// That [`mutation::DataFrame`] is kglite's own columnar container, built on
    /// `Value` — kglite does not depend on polars. `update_node_properties`,
    /// `purge_provisional_nodes`, and
    /// `extend_graph` (merge one graph into another) round out the
    /// generic, non-Selection mutation surface. `create_connections`
    /// (edge-create between the two ends of a selection) is here too, since
    /// `CurrentSelection` is itself an api type.
    pub mod mutation {
        /// The bulk-ingest container and its column vocabulary.
        /// `DataFrame::new` declares columns by [`ColumnType`] and
        /// `DataFrame::add_column` fills one from the matching [`ColumnData`]
        /// variant, so all three are needed to build a frame at all —
        /// `from_cypher_rows` is the only route that skips them.
        pub use crate::datatypes::values::{ColumnData, ColumnType, DataFrame};
        /// Structured mutation reports — what a write touched (nodes/edges
        /// created/updated/deleted, per operation). Returned by the mutation
        /// functions above; every binding surfaces them after a mutating call.
        pub use crate::graph::introspection::reporting::{
            ConnectionOperationReport, NodeOperationReport, OperationReport, OperationReports,
        };
        pub use crate::graph::mutation::add_properties::{add_properties, PropertySpec};
        pub use crate::graph::mutation::extend::{extend_graph, ExtendReport};
        // `AddPropertiesReport` is deliberately not re-exported: it was not part
        // of the public surface before this module was split out, and the API
        // baseline pins that surface.
        pub use crate::graph::mutation::maintain::{
            add_connections, add_edges_from_specs, add_nodes, create_connections,
            purge_provisional_nodes, replace_connections, update_node_properties, EdgeSpec,
            EdgeSpecReport,
        };
        /// Validate a graph against a `SchemaDefinition`.
        pub use crate::graph::mutation::validation::validate_graph;
    }

    /// Selection-scoped operations — selection set algebra
    /// (`union`/`intersection`/`difference`/`symmetric_difference`),
    /// subgraph extract / expand / stats, and the **shared selection-based
    /// query-primitive layer** used by both Cypher and the fluent API. Each op
    /// takes `(&DirGraph, &mut CurrentSelection, …already-marshalled params)`
    /// and mutates the selection in place; a binding building a fluent surface
    /// composes these directly (the wheel's `kg_fluent` / `kg_introspection`
    /// PyO3 methods marshal Python args, then call straight into here). The
    /// `core::*` primitives stay *defined* in `crate::graph::core`; this is
    /// their curated, stable re-export surface.
    pub mod fluent {
        // Selection set algebra + subgraph.
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
            format_unique_values_for_storage, get_connections, get_node_degrees, get_nodes,
            get_property_values, get_unique_values, LevelConnections, LevelNodes, LevelValues,
            UniqueValues,
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
        pub use crate::graph::core::value_operations::format_value_compact;
        // Spatial predicates over a selection (geo filters / centroids /
        // bounds).
        pub use crate::graph::features::spatial::{
            calculate_centroid, contains_point, get_bounds, intersects_geometry, near_point,
            near_point_m, within_bounds, wkt_centroid,
        };
        // Temporal validity predicates (per NodeData + TemporalConfig).
        pub use crate::graph::features::temporal::{
            node_is_temporally_valid, node_overlaps_range, node_passes_context,
        };
    }

    /// Embedding ingest + vector-index construction — how a binding gets
    /// vectors *into* a graph. `set_embeddings` replaces a store,
    /// `add_embeddings` upserts into one, `build_vector_index` builds the
    /// HNSW index that accelerates whole-corpus top-k, `store_name` /
    /// `store_key` are the one place the `"{text_column}_emb"` store name is
    /// minted, and `resolve_source_column` is the one place a source column is
    /// validated and resolved (identity aliases included) — a binding that
    /// reads the column's text itself goes through it so the validating half
    /// and the reading half cannot disagree. Each ingest
    /// call validates every id and dimension before it touches a store and
    /// bumps the graph version on a non-empty write, so it is all-or-nothing
    /// under a plain `&mut DirGraph`.
    ///
    /// Querying by vector needs no surface here: `vector_score` and
    /// `text_score` both take a caller-supplied query vector through
    /// `cypher_query`.
    pub mod embeddings {
        pub use crate::graph::embeddings::{
            add_embeddings, build_vector_index, drop_vector_index, has_vector_index,
            list_embeddings, list_vector_indexes, refresh_vector_index, resolve_source_column,
            set_embeddings, store_key, store_name, EmbeddingIngestReport, EmbeddingStoreInfo,
            VectorIndexReport, VectorIndexStatus,
        };
    }

    /// Lexical (BM25) text-index lifecycle — how a binding gets a text index
    /// *built*. `build_text_index` indexes one node type's string property,
    /// `drop_text_index` removes it, `has_text_index` / `list_text_indexes`
    /// report what exists, and `index_key` is the one place the
    /// `(node_type, property)` key is minted. Building is explicit and
    /// idempotent: a rebuild is the same call again.
    ///
    /// Querying needs no surface here — ranking is a Cypher function, so every
    /// binding reaches it through `cypher_query`. `TextIndexStore`'s own
    /// `prepare_query` / `score` / `top_k` are the direct-call companions for a
    /// Rust embedder that would rather not go through Cypher; `PreparedQuery`
    /// is exported because those signatures name it (with `QueryTerm` /
    /// `TermId`, which its own accessor names in turn), and `TextIndexRead`
    /// because a scoring loop that *can* hold one view across
    /// prepare-and-score should, rather than re-locking per row. A caller that
    /// cannot — the Cypher scalar's executor is `Sync` and a read guard is not
    /// `Send` — re-locks per row and watches `TextIndexStore::generation`
    /// instead. `text_index_store` resolves one by key.
    ///
    /// `refresh_text_index` is the catch-up driver: a query that is about to
    /// read an index asks `TextIndexStore::can_auto_refresh` whether the
    /// outstanding delta is small enough to fold in, and calls this if it is.
    /// It takes `&DirGraph` — catching up happens on the read path, and the
    /// index is behind its own lock for exactly that reason. Folding a
    /// document is not a constant: it splices into postings lists that grow
    /// with the corpus, so a refresh past the measured crossover rebuilds the
    /// index instead, and the worst case is one rebuild rather than
    /// delta x corpus.
    pub mod text_indexes {
        pub use crate::graph::algorithms::text_index::bm25::{PreparedQuery, QueryTerm};
        pub use crate::graph::algorithms::text_index::TermId;
        pub use crate::graph::index_freshness::DEFAULT_AUTO_REFRESH_LIMIT;
        pub use crate::graph::text_indexes::{
            build_text_index, drop_text_index, has_text_index, index_key, list_text_indexes,
            refresh_text_index, text_index_store, TextIndexRead, TextIndexReport, TextIndexStore,
        };
    }

    /// Graph algorithms — pathfinding, components, centrality, community
    /// detection. The typed, direct-call surface: each takes `&DirGraph` +
    /// plain params and returns a result struct, for bindings that want
    /// structs rather than result rows. (The same algorithms are reachable
    /// per-query as Cypher procedures.) `vector_search` +
    /// `VectorSearchResult` are here too — vector search is scoped to a
    /// selection.
    pub mod algorithms {
        pub use crate::graph::algorithms::graph_algorithms::{
            all_paths, are_connected, are_connected_with, betweenness_centrality,
            closeness_centrality, connected_components, degree_centrality, get_node_info,
            get_path_connections, label_propagation, leiden_communities, louvain_communities,
            node_degree, pagerank, shortest_path, shortest_path_cost, shortest_path_cost_batch,
            shortest_path_cost_batch_with, shortest_path_cost_weighted, shortest_path_cost_with,
            shortest_path_costs_from, shortest_path_weighted, weakly_connected_components,
            AllPathsOptions, CentralityOptions, CentralityResult, CommunityOptions,
            CommunityResult, DegreeCentralityOptions, EdgeDir, LabelPropagationOptions,
            PagerankOptions, PathNodeInfo, PathOptions, PathResult,
        };
        pub use crate::graph::algorithms::hnsw::HnswParams;
        pub use crate::graph::algorithms::vector::{
            vector_search, DistanceMetric, VectorSearchOptions, VectorSearchResult,
        };
        pub use crate::graph::algorithms::Interrupt;
    }

    /// Timeseries date/query helpers — the pure date-parsing and
    /// range-finding utilities behind inline timeseries support.
    /// `parse_date_query` maps a user string ("2013" / "2010..2015") to a
    /// `NaiveDate` + `DatePrecision`; `TimeseriesConfig` / `NodeTimeseries`
    /// are the config/data types.
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
    /// agent-facing schema from.
    pub mod introspection {
        pub use crate::graph::introspection::bug_report::write_bug_report;
        /// What [`derive_edge_counts_from_triples`] folds a triple list into:
        /// per-edge-type totals plus each type's endpoint sets, so a caller
        /// that already holds the triples needs no second scan for either.
        pub use crate::graph::introspection::connectivity::DerivedEdgeStats;
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
        /// Core-type-count tier classification — the four ranges `describe()`
        /// adapts its output by (supporting types, those with a parent, are not
        /// counted). Exported so a consumer that renders a graph reads the
        /// thresholds from here instead of copying them, which drifts silently.
        pub use crate::graph::introspection::{graph_scale, GraphScale};
        /// Result types of the `compute_*` functions above:
        /// `NodeTypeOverview` is one [`SchemaOverview::node_types`] entry,
        /// `NeighborsSchema` / `NeighborConnection` are
        /// [`compute_neighbors_schema`]'s answer, and `PropertyStatInfo` is
        /// [`compute_property_stats`]'.
        pub use crate::graph::introspection::{
            NeighborConnection, NeighborsSchema, NodeTypeOverview, PropertyStatInfo,
        };
        /// One row of the type-level cardinality graph —
        /// `(src)-[conn]->(tgt)` and its edge count.
        /// [`compute_type_connectivity`] computes them in one O(E) pass;
        /// `DirGraph::get_or_compute_type_connectivity` serves the copy
        /// persisted in the `.kgl`.
        pub use crate::graph::schema::ConnectivityTriple;
    }

    /// Graph I/O: `.kgl` load/save, format exporters (GraphML / GEXF /
    /// D3-JSON / CSV), the N-Triples (RDF) streaming loader + progress
    /// callbacks, embedding-vector file export/import, and streaming
    /// disk subset export.
    pub mod io {
        pub use crate::graph::io::export::{
            to_csv, to_csv_dir, to_d3_json, to_gexf, to_graphml, to_text,
        };
        /// Dependency-free relational exit: a deterministic SQLite-dialect SQL
        /// script (`sqlite3 out.db < dump.sql`). Node types become tables,
        /// connection types become link tables.
        pub use crate::graph::io::export_sql::to_sqlite_dump;
        /// Everything a `.kgl` write needs done to the graph before its bytes
        /// exist: metadata stamp plus the column-consolidation pass whose row
        /// order *is* the file's node binding. A binding that wants the bytes
        /// rather than a file calls this and then `write_kgl_to`; `save_graph`
        /// runs it internally.
        pub use crate::graph::io::file::prepare_kgl_write;
        /// Embedding-vector file export / import.
        pub use crate::graph::io::file::{
            export_embeddings_to_file, import_embeddings_from_file, EmbeddingExportFilter,
            ImportStats,
        };
        /// `.kgl` load / save (the canonical persistence format). `save_graph`
        /// and `save_graph_with` are the single save dispatch and report
        /// `SaveError`, whose `Refused` variant is a save declined *before*
        /// the path was touched — a write-ahead sidecar beside the target
        /// holds commits this checkpoint would strand. Bindings map that to
        /// their own class for a bad request, not to an I/O failure.
        ///
        /// `materialize_disk_graph` is the convert-and-publish composite behind
        /// `enable_disk_mode(path=...)`: it runs the same guard, converts the
        /// graph to disk storage *inside* the destination, and publishes it, so
        /// the live handle ends on the generation a fresh open would read.
        pub use crate::graph::io::file::{
            load_file, load_kgl_bytes, materialize_disk_graph, prepare_save, save_graph,
            save_graph_with, write_kgl, write_kgl_to, write_kgl_with, SaveError,
        };
        pub use crate::graph::io::ntriples::{
            load_ntriples, Cancelled, NTriplesConfig, ProgressEvent, ProgressSink, ProgressValue,
        };
        /// `open_or_create_graph` treats its mode as a *creation default* and
        /// never touches an existing graph's own mode;
        /// `open_or_create_graph_in_mode` treats it as the caller's explicit
        /// request and converts (or refuses) on an existing graph too. A
        /// binding that took the mode from a user wants the latter.
        pub use crate::graph::io::open::{
            open_or_create_graph, open_or_create_graph_in_mode, GraphFileIdentity,
            GraphWriterLease, LeaseHolder, LeaseRefusal, OpenDisposition, OpenGraphResult,
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
        /// The persisted-format version numbers this build reads and writes:
        /// [`KGL_FORMAT_VERSION`] is the `.kgl` snapshot format stamped into new
        /// saves, [`WAL_FORMAT_VERSION`] the write-ahead-log frame format, and
        /// [`MIN_READABLE_WAL_FORMAT_VERSION`] the oldest WAL frame format this
        /// build can replay. All three are distinct from the engine SemVer a
        /// binding reads via the ABI version probe — they describe the on-disk
        /// format lifecycle, not the library version. Exposed so a binding can
        /// report the storage format it operates against.
        pub use crate::graph::schema::KGL_FORMAT_VERSION;
        pub use crate::graph::wal::{MIN_READABLE_WAL_FORMAT_VERSION, WAL_FORMAT_VERSION};
    }

    /// Storage backend configuration — the in-memory / mmap / disk backends
    /// (`GraphBackend` + `DiskGraph` / `MappedGraph` constructors), the
    /// per-type lookup, and the embedding store. CLAUDE.md designates
    /// storage-backend configuration a direct-api concern.
    pub mod storage {
        pub use crate::graph::schema::EmbeddingStore;
        pub use crate::graph::storage::backend::GraphBackend;
        pub use crate::graph::storage::disk::graph::DiskGraph;
        pub use crate::graph::storage::lookups::TypeLookup;
        /// The cross-binding create-in-mode builder: resolve a mode string to
        /// a [`StorageMode`] and build a fresh graph in that backend. Shared by
        /// the wheel (`storage='mapped'/'disk'`), the bolt/mcp servers
        /// (`--storage`), and the C ABI (`kglite_graph_new_in_mode`).
        /// `live_storage_mode` answers "which mode is this graph actually in?"
        /// — the classification every binding needs after an open — and
        /// `convert_dir_graph_to_mode` is the explicit switch between the two
        /// portable backends, refusing the disk directions structurally.
        pub use crate::graph::storage::mode::{
            convert_dir_graph_to_mode, live_storage_mode, new_dir_graph_in_mode, StorageMode,
        };
        pub use crate::graph::storage::MappedGraph;
    }

    /// Change data capture — the opt-in in-process change stream a binding
    /// exposes through `db.cdc.*`, plus the commit-boundary drain any owner of
    /// a bare `DirGraph` must call for its commits to be published (the same
    /// obligation the durable paths carry for `flush_wal`). Cypher-first: a
    /// binding needs none of this to *read* the stream, only to say where its
    /// commit boundaries are.
    pub mod cdc {
        pub use crate::graph::cdc::{
            disable, drain_at_commit, enable, needs_before_images, parse_selectors,
            publish_drained, read, status, CdcChange, CdcEnrichment, CdcEvent, CdcEventKind,
            CdcHandle, CdcHandoff, CdcLog, CdcSelector, CdcStatus, EdgeState, NodeState,
            DEFAULT_CAPACITY, MAX_CAPACITY,
        };
    }

    /// Durable transactions — the write-ahead log (append / recover / replay)
    /// and the write-capture recording layer behind a binding's `durable()`
    /// feature. The in-process WAL mechanism (distinct from the checkpoint
    /// save in `io`).
    pub mod durable {
        /// Binding-agnostic durable-open + checkpoint orchestration: the
        /// recover→replay→wrap→append ordering every owner of a log performs at
        /// open (`open_log`, which also enforces the unconditional
        /// recovery-on-open refusal), and the two halves of the four-step
        /// checkpoint that bracket a binding's own save. `ensure_recovered` is
        /// that same refusal for an opener that attaches no log at all, and is
        /// already applied by `io::open_or_create_graph`.
        pub use crate::graph::durability::{
            checkpoint_epilogue, checkpoint_prologue, ensure_recovered, open_log, DurableOpenError,
        };
        pub use crate::graph::mutation::wal_replay::apply_frames;
        pub use crate::graph::storage::recording::{
            resolve_ops, wrap_for_durability, BeforeImage, CaptureOrigin, RawOp, RecordingGraph,
        };
        pub use crate::graph::wal::{recover, wal_path, DurabilityLevel, SyncMode, Wal, WalFrame};
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
    /// `from_blueprint` and the C ABI's `kglite_blueprint_build` are both
    /// thin wrappers around [`load_blueprint_file`] + [`build`].
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
        /// Bind label / relationship-type positions written as parameters
        /// (`MATCH (n:$label)`) before validation and planning. `session`
        /// runs this for every statement it prepares; a binding that drives
        /// the parse → optimize → execute steps itself must call it too, or
        /// a parameterised label silently matches nothing.
        pub use crate::graph::languages::cypher::dynamic_labels;
        pub use crate::graph::languages::cypher::executor::write::execute_mutable;
        pub use crate::graph::languages::cypher::executor::CypherExecutor;
        pub use crate::graph::languages::cypher::generate_explain_result;
        pub use crate::graph::languages::cypher::is_mutation_query;
        pub use crate::graph::languages::cypher::parameter_names;
        /// Parse a Cypher statement into a `CypherQuery`.
        ///
        /// This must stay the **cached** parser
        /// (`parse_cache::parse_cypher_cached`), the same one
        /// `session::execute` uses: a repeated statement is a hash lookup plus
        /// an AST clone rather than a full re-parse. Pointing it at the raw
        /// `parser::parse_cypher` makes every binding that pre-parses to
        /// classify a statement (the wheel does it on `cypher` /
        /// `Session.cypher` / `Transaction.cypher` / `frozen`) pay a second
        /// full parse per call — measured at 25% of a small parameterised
        /// query.
        pub use crate::graph::languages::cypher::parse_cypher;
        pub use crate::graph::languages::cypher::parse_with_mutation_check;
        pub use crate::graph::languages::cypher::planner;
        pub use crate::graph::languages::cypher::planner::mark_lazy_eligibility;
        pub use crate::graph::languages::cypher::planner::schema_check::validate_schema;
        /// Where query-warning *echoes* go, process-wide. The structured
        /// `QueryDiagnostics::warnings` channel is unconditional and
        /// unaffected; this only moves the `warning:` line off stderr, for a
        /// binding that presents warnings itself.
        pub use crate::graph::languages::cypher::planner::schema_check::{
            query_warning_sink, set_query_warning_sink, QueryWarningSink,
        };
        pub use crate::graph::languages::cypher::planner::simplification::rewrite_text_score;
        pub use crate::graph::languages::cypher::query_features;
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
        pub use crate::graph::languages::cypher::QueryFeatures;
        // Specific Cypher-pipeline items a binding implementing a native
        // `cypher()` method (the wheel) reaches. Exposed INDIVIDUALLY — not as
        // whole `ast`/`executor`/`parser`/`result` submodules — so the rest of
        // the executor/parser internals stay un-exported and the optimizer can
        // keep inlining the per-query hot path. (Re-exporting the whole
        // executor module measurably regressed cypher micro-query latency by
        // ~60% on tiny graphs.)
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
        /// `LOAD CSV` filesystem capability. Every binding decides what its
        /// callers get; see `ExecuteOptions::csv_import`.
        pub use crate::graph::languages::cypher::executor::load_csv::CsvImportPolicy;
        pub use crate::graph::session::{
            execute_mut, execute_read, resolve_noderefs, CommitOutcome, ExecuteOptions,
            ExecuteOutcome, Session, Transaction, QUERY_THREAD_STACK_SIZE,
        };
    }
}
