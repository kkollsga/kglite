# Changelog

All notable changes to KGLite will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.5.74] - 2026-03-01

### Added

- **Multi-hop traversal benchmarks** — scale-free graph benchmarks at 1K/10K/50K/100K nodes with hop depths 1–8, comparable to TuringDB/Neo4j multi-hop benchmarks
- **Blueprint documentation** — standalone guide page with step-by-step walkthrough, real CSV examples, and troubleshooting

### Changed

- **Variable-length path BFS** — global dedup mode skips path tracking when path info isn't needed (no `p = ...` assignment, no named edge variable), reducing memory and redundant exploration (~4x faster)
- **WHERE IN predicate pushdown** — `WHERE n.id IN [list]` is now pushed into the MATCH pattern and resolved via id-index O(1) lookups instead of post-filtering all nodes (~1,400x faster on 10K 8-hop traversals)

## [0.5.73] - 2026-02-27

### Changed

- **README** — added blueprint loading and code review examples to Quick Start, doc links on each section
- **CLAUDE.md** — simplified and consolidated conventions

## [0.5.72] - 2026-02-27

### Added

- **Documentation site** — Sphinx + Furo docs with auto-generated API reference from `.pyi` stubs, hosted on Read the Docs. Guide pages for Cypher, data loading, querying, semantic search, spatial, timeseries, graph algorithms, import/export, AI agents, and code tree.

## [0.5.71] - 2026-02-27

### Added

- **`traverse()` API improvements:**
  - `target_type` parameter — filter targets to specific node type(s): `traverse('OF_FIELD', direction='incoming', target_type='ProductionProfile')` or `target_type=['ProductionProfile', 'FieldReserves']`
  - `where` parameter — alias for `filter_target`, consistent with the fluent API: `traverse('HAS_LICENSEE', where={'title': 'Equinor'})`
  - `where_connection` parameter — alias for `filter_connection`: `traverse('RATED', where_connection={'score': {'>': 4}})`
  - `help(g.traverse)` now shows a comprehensive docstring with args, examples, and usage patterns
- **Temporal awareness** — first-class support for time-dependent nodes and connections:
  - Declare temporal columns via `column_types={"fldLicenseeFrom": "validFrom", "fldLicenseeTo": "validTo"}` on `add_nodes()` or `add_connections()` — auto-configures temporal filtering behind the scenes (same pattern as spatial `"geometry"` / `"location.lat"`)
  - `date("2013")` sets a temporal context for the entire chain — all subsequent `select()` and `traverse()` calls filter to that date instead of today
  - `date("2010", "2015")` — range mode: include everything valid at any point during the period (overlap check)
  - `date("all")` — disable temporal filtering entirely (show all records regardless of validity dates)
  - `select()` auto-filters temporal nodes to "currently valid" (or the `date()` context). Pass `temporal=False` to include all historic records
  - `traverse()` auto-filters temporal connections to "currently valid". Override with `at="2015"`, `during=("2010", "2020")`, or `temporal=False`
  - `valid_at()` / `valid_during()` auto-detect field names from temporal config; NULL `date_to` treated as "still active"
  - Display (`sample()`, `collect()`) filters connection summaries to temporally valid edges
  - `describe()` includes `temporal_from`/`temporal_to` attributes on configured types and connections
  - Blueprint loader: use `"validFrom"` / `"validTo"` property types to auto-configure temporal filtering
  - `set_temporal(type_name, valid_from, valid_to)` available as low-level API for manual configuration
  - Temporal configs persist through `save()`/`load()` round-trips
- **`show(columns, limit=200)`** — compact display of selected nodes with chosen properties. Single-level shows `Type(val1, val2)` per line; after `traverse()` walks the full chain as `Type1(vals) -> Type2(vals) -> Type3(vals)`. Resolves field aliases and truncates long values

## [0.5.70] - 2026-02-26

### Added

- **`to_str(limit=50)`** — format current selection as a human-readable string with `[Type] title (id: x)` headers and indented properties
- **`print(ResultView)` smart formatting** — `ResultView.__str__` uses multiline card format (properties + connection arrows) for ≤3 rows, compact one-liner for >3. Connections show direction with `◆` as the current node: `◆ --WORKS_AT--> Company(id, title)` for outgoing, `Person(id, title) --WORKS_AT--> ◆` for incoming. Long values (WKT geometries, etc.) are truncated with middle ellipsis
- **`sample()` selection-aware** — `sample()` now works on the current selection (`graph.select('Person').sample(3)`) in addition to the existing `sample('Person', 3)` form
- **`head()`/`tail()` preserve connections** — slicing a ResultView carries connection summaries through

## [0.5.67] - 2026-02-26

### Changed

- **BREAKING: Fluent API method renames** — modernized the fluent API surface to match common query DSL conventions:
  - `type_filter()` → `select()`
  - `filter()` → `where()`
  - `filter_any()` → `where_any()`
  - `filter_orphans()` → `where_orphans()`
  - `has_connection()` → `where_connected()`
  - `max_nodes()` → `limit()`
  - `get_nodes()` → `collect()`
  - `node_count()` → `len()` (also adds `__len__` for `len(graph)`)
  - `id_values()` → `ids()`
  - `max_nodes=` parameter → `limit=` everywhere (select, where, traverse, collect, etc.)
- **BREAKING: Retrieval method renames** — dropped inconsistent `get_` prefix and shortened verbose methods:
  - `get_titles()` → `titles()`
  - `get_connections()` → `connections()`
  - `get_degrees()` → `degrees()`
  - `get_bounds()` → `bounds()`
  - `get_centroid()` → `centroid()`
  - `get_selection()` → `selection()`
  - `get_schema()` → `schema_text()`
  - `get_schema_definition()` → `schema_definition()`
  - `get_last_report()` → `last_report()`
  - `get_operation_index()` → `operation_index()`
  - `get_report_history()` → `report_history()`
  - `get_spatial()` → `spatial()`
  - `get_timeseries()` → `timeseries()`
  - `get_time_index()` → `time_index()`
  - `get_timeseries_config()` → `timeseries_config()`
  - `get_embeddings()` → `embeddings()`
  - `get_embedding()` → `embedding()`
  - `get_node_by_id()` → `node()`
  - `children_properties_to_list()` → `collect_children()` (also `filter=` param → `where=`)

### Removed

- **`get_ids()`** — removed; use `ids()` for flat ID list or `collect()` for full node dicts

## [0.5.66] - 2026-02-26

### Changed

- **Blueprint loader output** — quiet by default (only warnings/errors + summary); verbose mode for per-type detail. Warnings from `add_connections` skips are now tracked in the loader instead of surfacing as raw `UserWarning`s
- **Blueprint settings** — `root` renamed to `input_root`, `output` split into `output_path` (optional directory) + `output_file` (filename or relative path with `../` support). Old keys still accepted for backwards compatibility

### Fixed

- **Float→Int ID coercion** — FK columns with nullable integers (read as float64 by pandas, e.g. `260.0`) are now auto-coerced to int before edge matching. The Rust lookup layer also gained Float64 → Int64/UniqueId fallback as a safety net
- **Timeseries FK edge filtering** — FK edges for timeseries node types now apply the same time-component filter as node creation (e.g. dropping month=0 aggregate rows), preventing "source node not found" warnings for carriers that only have aggregate data

## [0.5.65] - 2026-02-26

### Added

- **`FLUENT.md`** — comprehensive fluent API reference documenting all method-chaining operations: data loading, selection & filtering, spatial, temporal, timeseries, vector search, traversal, algorithms, set operations, indexes, transactions, export, and a fluent-vs-Cypher feature matrix
- **`create_connections()`** — renamed from `selection_to_new_connections` with new capabilities: `properties` dict copies node properties onto new edges (e.g. `properties={'B': ['score']}`), `source_type`/`target_type` override which traversal levels to connect (defaults to first→last level)
- **Comparison-based `traverse(method=...)`** — discover relationships without pre-existing edges. Five methods: `'contains'` (spatial containment), `'intersects'` (geometry overlap), `'distance'` (geodesic proximity), `'text_score'` (semantic similarity via embeddings), `'cluster'` (kmeans/dbscan grouping). `method` accepts a string shorthand (`method='contains'`) or a dict with settings (`method={'type': 'distance', 'max_m': 5000, 'resolve': 'centroid'}`). The `resolve` key controls polygon geometry interpretation: `'centroid'` (force geometry centroid), `'closest'` (nearest boundary point), `'geometry'` (full polygon shape). Produces the same selection hierarchy as edge-based traversal, so all downstream methods work unchanged
- **`add_properties()`** — enrich selected nodes with properties from ancestor nodes in the traversal chain. Supports copy (`['name']`), copy-all (`[]`), rename (`{'new': 'old'}`), aggregate expressions (`'count(*)'`, `'mean(depth)'`, `'sum(production)'`, `'min()'`, `'max()'`, `'std()'`, `'collect()'`), and spatial compute (`'distance'`, `'area'`, `'perimeter'`, `'centroid_lat'`, `'centroid_lon'`)

### Changed

- **`selection_to_new_connections` → `create_connections`** — renamed for brevity. Now defaults to connecting the top-level ancestor to leaf nodes (was parent→child at last level only)

## [0.5.64] - 2026-02-25

### Added

- **List quantifier predicates** — `any(x IN list WHERE pred)`, `all(...)`, `none(...)`, `single(...)` for filtering over lists in WHERE, RETURN, and WITH clauses
- **Exploration hints in `describe()`** — inventory views now surface disconnected node types and join candidates (property value overlaps between unconnected type pairs) to suggest enrichment opportunities
- **Temporal Cypher functions** — `valid_at(entity, date, 'from_field', 'to_field')` and `valid_during(entity, start, end, 'from_field', 'to_field')` for date-range filtering on both nodes and relationships in WHERE clauses. NULL fields treated as open-ended boundaries

### Changed

- **Rewritten examples** — new domain examples: `legal_graph.py` (index-based loading), `code_graph.py` (code tree parsing), `spatial_graph.py` (blueprint loading), `mcp_server.py` (generic MCP server with auto-detected code tools)

## [0.5.63] - 2026-02-25

### Added

- **`export_csv(path)`** — bulk export to organized CSV directory tree with one file per node type and connection type, sub-node nesting, full properties, and a `blueprint.json` for round-trip re-import via `from_blueprint()`
- **Variable binding in MATCH pattern properties** — bare variables from `WITH`/`UNWIND` can now be used in inline pattern properties: `WITH "Oslo" AS city MATCH (n:Person {city: city}) RETURN n`
- **Map literals in Cypher expressions** — `{key: expr, key2: expr}` syntax in `RETURN`/`WITH` for constructing map objects: `RETURN {name: n.name, age: n.age} AS m`
- **WHERE clause inside EXISTS subqueries** — `EXISTS { MATCH (n:Type) WHERE n.prop = expr }` now supports arbitrary WHERE predicates including cross-scope variable references and regex

### Changed

- **Cypher query performance** — eliminated `type_indices` Vec clone on every MATCH (iterate by reference), move-on-last-match optimization to reduce row cloning in joins, pre-allocated result vectors, eliminated unnecessary clone in composite index lookups
- **MERGE index acceleration** — MERGE now uses `id_indices`, `property_indices`, and `composite_indices` for O(1) pattern matching instead of linear scan through all nodes of a type. Orders-of-magnitude faster for batch `UNWIND + MERGE` workloads
- **UNWIND/MERGE clone reduction** — UNWIND moves (instead of cloning) the row for the last unwound item; MERGE iterates source rows by value to avoid per-row cloning

## [0.5.61] - 2026-02-24

### Added

- **PROFILE** prefix for Cypher queries — executes query and collects per-clause statistics (rows_in, rows_out, elapsed_us). Access via `result.profile`
- **Structured EXPLAIN** — `EXPLAIN` now returns a `ResultView` with columns `[step, operation, estimated_rows]` instead of a plain string. Cardinality estimates use type_indices counts
- **Read-only transactions** — `begin_read()` creates an O(1) Arc-backed snapshot (zero memory overhead). Mutations are rejected
- **Optimistic concurrency control** — `commit()` detects graph modifications since `begin()` and raises `RuntimeError` on conflict
- **Transaction timeout** — `begin(timeout_ms=...)` and `begin_read(timeout_ms=...)` set a deadline for all operations within the transaction
- `Transaction.is_read_only` property
- `describe(cypher=['EXPLAIN'])` and `describe(cypher=['PROFILE'])` topic detail pages
- Expanded `<limitations>` section in `describe(cypher=True)` with workarounds for unsupported features
- openCypher compatibility matrix in CYPHER.md

## [0.5.60] - 2026-02-24

### Added

- `describe(cypher=True)` tier 1 hint now highlights KGLite-specific features (||, =~, coalesce, CALL procedures, distance/contains)
- `describe(cypher=True)` tier 2 includes `<not_supported>` section and spatial functions group
- `describe()` overview connection map includes `count` attribute per connection type
- `describe()` connections hint only shown when graph has edges
- `describe(cypher=['spatial'])` topic with distance, contains, intersects, centroid, area, perimeter docs

## [0.5.59] - 2026-02-24

### Added

- `bug_report(query, result, expected, description)` — file Cypher bug reports to `reported_bugs.md`. Timestamped, version-tagged entries prepended to top of file. Input sanitised against HTML/code injection
- `KnowledgeGraph.explain_mcp()` — static method returning a self-contained XML quickstart for setting up a KGLite MCP server (server template, core/optional tools, Claude registration config)

### Fixed

- `collect(node)[0].property` now returns the actual property value instead of the node's title. Previously, `WITH f, collect(fr)[0] AS lr RETURN lr.oil` would return the node title for every property access. Node identity is now preserved through collect→index→WITH pipelines via internal `Value::NodeRef` references

## [0.5.58] - 2026-02-24

### Added

- `CALL cluster()` procedure — general-purpose clustering via Cypher. Supports DBSCAN and K-means methods. Reads nodes from preceding MATCH clause. Spatial mode auto-detects lat/lon from `set_spatial()` config with geometry centroid fallback; property mode clusters on explicit numeric properties with optional normalization. YIELD node, cluster (noise = -1 for DBSCAN)
- `round(x, decimals)` — optional second argument for decimal precision (e.g. `round(3.14159, 2)` → 3.14). Backward compatible: `round(x)` still rounds to integer
- `||` string concatenation operator — concatenates values in expressions (e.g. `n.first || ' ' || n.last`). Null propagates. Non-string values auto-converted
- `describe(cypher=True)` — 3-tier Cypher language reference: compact `<cypher hint/>` in overview (tier 1), full clause/operator/function/procedure listing with `cypher=True` (tier 2), detailed docs with params and examples via `cypher=['cluster','MATCH',...]` (tier 3)
- `describe(connections=True)` — connection type progressive disclosure: overview with `connections=True` (all types, counts, endpoints, property names), deep-dive with `connections=['BELONGS_TO']` (per-pair counts, property stats, sample edges)

## [0.5.56] - 2026-02-23

### Added

- `near_point_m()` — geodesic distance filter in meters (SI units), replaces `near_point_km()` and `near_point_km_from_wkt()`
- Geometry centroid fallback: fluent API spatial methods (`near_point_m`, `within_bounds`, `get_bounds`, `get_centroid`) now fall back to WKT geometry centroid when lat/lon fields are missing but a geometry is configured via `set_spatial` or `column_types`

### Changed

- Cypher `distance(a, b)` returns Null (instead of erroring) when a node has no spatial data, so `WHERE distance(a, b) < X` simply filters those nodes out
- Cypher comparison operators (`<`, `<=`, `>`, `>=`) now follow three-valued logic: comparisons involving Null evaluate to false (previously Null sorted as less-than-everything)

### Removed

- `near_point_km()` — use `near_point_m()` with meters instead (e.g. `max_distance_m=50_000.0` for 50 km)
- `near_point_km_from_wkt()` — subsumed by `near_point_m()` which auto-falls back to geometry centroid

## [0.5.55] - 2026-02-23

### Changed

- Cypher spatial functions now return SI units: `distance()` → meters, `area()` → m², `perimeter()` → meters (were km/km²). Distance uses WGS84 geodesic (Karney algorithm) instead of spherical haversine

### Removed

- `agent_describe()` — replaced by `describe()`. Migration: `graph.agent_describe()` → `graph.describe()`, `graph.agent_describe(detail='full')` → `graph.describe()` (auto-selects detail level)

## [0.5.54] - 2026-02-23

### Added

- `describe(types=None)` — progressive disclosure schema description for AI agents. Inventory mode returns node types grouped by size with property complexity markers and capability flags, connection map, and Cypher extensions. Focused mode (`types=['Field']`) returns detailed properties, connections, timeseries/spatial config, and sample nodes. Automatically inlines full detail for graphs with ≤15 types
- `set_parent_type(node_type, parent_type)` — declare a node type as a supporting child of a core type. Supporting types are hidden from the `describe()` inventory and appear in the `<supporting>` section when the parent is inspected. The `from_blueprint()` loader auto-sets parent types for sub-nodes
- Cypher math functions: `abs()`, `ceil()` / `ceiling()`, `floor()`, `round()`, `sqrt()`, `sign()` — work with Int64 and Float64 values, propagate Null
- String coercion on `+` operator: when one operand is a string, the other is automatically converted (e.g. `2024 + '-06'` → `'2024-06'`). Null still propagates

### Changed

- `describe()` inventory now uses compact descriptor format `TypeName[size,complexity,flags]` instead of size bands. Types listed as flat comma-separated list sorted by count descending. Core types with supporting children show `+N` suffix. Capability flags from supporting types bubble up to their parent descriptor
- `describe()` now shows a `<read-only>` notice listing unsupported Cypher write commands (CREATE, SET, DELETE, REMOVE, MERGE) when the graph is in read-only mode

## [0.5.53] - 2026-02-23

### Added

- `from_blueprint()` — build a complete KnowledgeGraph from a JSON blueprint and CSV files. Supports core nodes, sub-nodes, FK edges, junction edges, timeseries, geometry conversion, filters, manual nodes (from FK values), and auto-generated IDs
- Cypher `date()` function — converts date strings to DateTime values: `date('2020-01-15')`
- `property_types` on blueprint junction edges for automatic type conversion (e.g. epoch millis → DateTime)
- Temporal join support: `ts_*()` functions accept DateTime edge properties and null values as date range arguments
- Cypher `IS NULL` / `IS NOT NULL` now supported as expressions in RETURN/WITH (e.g. `RETURN x IS NULL AS flag`)
- `agent_describe(detail, include_fluent)` — optional detail level adapts output to graph complexity. Graphs with >15 types auto-select compact mode (~5-8x smaller output). Fluent API docs excluded by default (opt-in via `include_fluent=True`)

### Changed

- **Performance**: `agent_describe()` 27x faster (1.3s → 48ms) via property index fast path and scan capping
- **Performance**: `MATCH (n) RETURN count(n)` short-circuits to O(1) via `FusedCountAll` (was ~266ms, now sub-ms)
- **Performance**: `MATCH (n) RETURN n.type, count(n)` short-circuits to O(types) via `FusedCountByType` (was ~727ms, now sub-ms)
- **Performance**: `MATCH ()-[r]->() RETURN type(r), count(*)` short-circuits to O(E) single-pass via `FusedCountEdgesByType` (was ~822ms, now ~3ms)
- **Performance**: `MATCH (n:Type) RETURN count(n)` short-circuits to O(1) via `FusedCountTypedNode` (reads type index length directly)
- **Performance**: `MATCH ()-[r:Type]->() RETURN count(*)` short-circuits via `FusedCountTypedEdge` (single-pass edge filter)
- **Performance**: Edge type counts cached in DirGraph with lazy invalidation on mutations
- **Performance**: Multi-hop fused aggregation for 5-element patterns (e.g. `MATCH (a)-[]->(b)<-[]-(c) RETURN a.x, count(*)`) traverses without materializing intermediate rows
- **Performance**: Regex `=~` operator caches compiled patterns per query execution (compile once, match many)
- **Performance**: PageRank uses pull-based iteration with rayon parallelization for large graphs (3-4x speedup)
- **Performance**: Louvain community detection precomputes loop-invariant division terms
- Timeseries keys stored as `NaiveDate` instead of composite integer arrays (`Vec<Vec<i64>>`)
- `set_time_index()` now accepts date strings (`['2020-01', '2020-02']`) in addition to integer lists
- `get_time_index()` returns ISO date strings (`['2020-01-01', '2020-02-01']`) instead of integer lists
- `get_timeseries()` keys returned as ISO date strings
- `ts_series()` output uses ISO date strings for time keys (e.g. `"2020-01-01"` instead of `[2020, 1]`)
- Null date arguments to `ts_*()` treated as open-ended ranges (no bound)
- Timeseries data format bumped (v2); legacy files skip timeseries loading with a warning

### Fixed

- `MATCH (a)-[]->(b) RETURN count(*)` with all-aggregate RETURN (no group keys) now correctly returns a single row instead of per-node rows
- `ORDER BY` on DateTime properties with `LIMIT` now returns correct results (FusedOrderByTopK optimization extended to handle DateTime, UniqueId, and Boolean sort keys)
- `ORDER BY` on String/Point properties with `LIMIT` now falls back to standard sort instead of returning empty results

## [0.5.52] - 2026-02-22

### Added

- `add_nodes()` now accepts a `timeseries` parameter for inline timeseries loading from flat DataFrames — automatically deduplicates rows per ID and attaches time-indexed channels
- Timeseries resolution extended to support `hour` (depth 4) and `minute` (depth 5) granularity
- `parse_date_string` now handles `'yyyy-mm-dd hh:mm'` and ISO `'yyyy-mm-ddThh:mm'` formats
- **Timeseries support**: per-node time-indexed data channels with resolution-aware date-string queries
- `set_timeseries()` with `resolution` ("year", "month", "day"), `units`, and `bin_type` metadata
- `set_time_index()` / `add_ts_channel()` for per-node timeseries construction
- `add_timeseries()` for bulk DataFrame ingestion with FK-based node matching and resolution validation
- `get_timeseries()` / `get_time_index()` for data extraction with date-string range filters
- Cypher `ts_*()` functions with date-string arguments: `ts_sum(f.oil, '2020')`, `ts_avg(f.oil, '2020-2', '2020-6')`, etc.
- Query precision validation: errors when query detail exceeds data resolution (e.g. `'2020-2-15'` on month data)
- Channel units (e.g. "MSm3", "°C") and bin type ("total", "mean", "sample") metadata
- Timeseries data persisted as a separate section in `.kgl` files (backward compatible)
- `agent_describe()` includes timeseries metadata, resolution, units, and function reference
- Cypher `range(start, end [, step])` function — generates integer lists for use with `UNWIND`

## [0.5.51] - 2026-02-21

### Added

- Fluent API: `filter()` now supports `regex` (or `=~`) operator for pattern matching, e.g. `filter({'name': {'regex': '^A.*'}})`
- Fluent API: `filter()` now supports negated operators: `not_contains`, `not_starts_with`, `not_ends_with`, `not_in`, `not_regex`
- Fluent API: `filter_any()` method for OR logic — keeps nodes matching any of the provided condition sets
- Fluent API: `offset(n)` method for pagination — combine with `max_nodes()` for page-based queries
- Fluent API: `has_connection(type, direction)` method — filter nodes by edge existence without changing the selection target
- Fluent API: `count(group_by='prop')` and `statistics('prop', group_by='prop')` — group by arbitrary property instead of parent hierarchy

## [0.5.50] - 2026-02-21

### Added

- Shapely/geopandas integration for spatial methods — `intersects_geometry()` and `wkt_centroid()` now accept shapely geometry objects as input in addition to WKT strings
- `as_shapely=True` parameter on `get_centroid()`, `get_bounds()`, and `wkt_centroid()` to return shapely geometry objects instead of dicts
- `ResultView.to_gdf()` — converts lazy results to a geopandas GeoDataFrame, parsing a WKT column into shapely geometries with optional CRS
- Spatial type system via `column_types` in `add_nodes()` — declare `location.lat`/`location.lon`, `geometry`, `point.<name>.lat`/`.lon`, and `shape.<name>` types for auto-resolution in Cypher and fluent API methods
- `set_spatial()` / `get_spatial()` for retroactive spatial configuration
- Cypher `distance(a, b)` now auto-resolves via spatial config (location preferred, geometry centroid fallback)
- Virtual spatial properties in Cypher: `n.location` → Point, `n.geometry` → WKT, `n.<point_name>` → Point, `n.<shape_name>` → WKT
- Spatial methods (`within_bounds`, `near_point_km`, `get_bounds`, `get_centroid`, etc.) auto-resolve field names from spatial config when not explicitly provided
- Node-aware spatial Cypher functions: `contains(a, b)`, `intersects(a, b)`, `centroid(n)`, `area(n)`, `perimeter(n)` — auto-resolve geometry via spatial config, also accept WKT strings
- Geometry-aware `distance()` — `distance(a.geometry, b.geometry)` returns 0 if touching; `distance(point(...), n.geometry)` returns 0 if inside, closest boundary distance otherwise

### Removed

- Cypher functions `wkt_contains()`, `wkt_intersects()`, `wkt_centroid()` — replaced by node-aware `contains()`, `intersects()`, `centroid()` which also accept raw WKT strings

### Fixed

- Betweenness centrality now uses undirected BFS — previously only traversed outgoing edges, causing nodes bridging communities via incoming edges to get zero scores

### Performance

- `RETURN ... ORDER BY expr LIMIT k` fused into single-pass top-k heap — O(n log k) instead of O(n log n) sort + O(n) full projection. **5.4x speedup** on `distance()` ORDER BY LIMIT queries (1M pairs: 2627ms → 486ms)
- `WHERE contains(a, b)` fast path (`ContainsFilterSpec`) — extracts contains() patterns and evaluates directly from spatial cache, bypassing expression evaluator chain
- Spatial Cypher functions 6-8x faster for contains/intersects via per-node spatial cache + bounding box pre-filter:
  - Per-node cache (`NodeSpatialData`): resolves each node's spatial data once per query, cached for all cross-product rows (N×M → N+M lookups)
  - Bounding box pre-filter: computes `geo::Rect` alongside cached geometry; rejects non-overlapping pairs in O(1) before expensive polygon tests
  - `resolve_spatial()` skips redundant expression evaluation for Variable/PropertyAccess — goes directly to cached node data
- Spatial resolution uses WKT geometry cache for centroid fallback path — previously re-parsed WKT on every row
- `intersects()` and `centroid()` avoid deep-cloning `Arc<Geometry>` — use references directly
- `geometry_contains_geometry()` uses `geo::Contains` trait instead of point-by-point boundary check

## [0.5.49] - 2026-02-20

### Added

- Python type stub (`.pyi`) files now included in code graph — enables graph coverage of stub-only packages, compiled extensions, and authoritative type contracts

### Fixed

- Cypher parser now accepts reserved words (e.g. `optional`, `match`, `type`) as alias names after `AS` — previously failed with "Expected alias name after AS"
- Betweenness centrality `sample_size` now uses stride-based sampling across the full node range — previously sampled only the first k nodes, which could be non-participating node types (Module/Class) yielding all-zero scores

## [0.5.46] - 2026-02-20

### Fixed

- Decorator property stored as JSON array instead of comma-separated string — fixes fragmentation of decorators with comma-containing arguments (e.g. `@functools.wraps(func, assigned=(...))`)
- `is_test`, `is_async`, `is_method` boolean properties now explicitly `false` on non-matching entities instead of `null` — enables `WHERE f.is_test = false` queries
- Dynamic project versions (setuptools-scm etc.) now stored as `"dynamic"` instead of `null` on the Project node
- CALLS edges now scope-aware — calls inside nested functions, lambdas, and closures are no longer attributed to the enclosing function (fixes over-counted fan-out in all 7 language parsers)
- `collect(x)[0..N]`, `count(x) + 1` and other aggregate-wrapping expressions in RETURN now work — previously errored with "Aggregate function cannot be used outside of RETURN/WITH"
- `size(collect(...))` and other non-aggregate functions wrapping aggregates now evaluate correctly — previously silently returned `null` because the expression was misclassified as non-aggregate

## [0.5.43] - 2026-02-20

### Added

- List slicing in Cypher: `expr[start..end]`, `expr[..end]`, `expr[start..]` — works on `collect()` results and list literals, supports negative indices

### Fixed

- `size()` and `length()` functions on lists now return element count instead of JSON string length — e.g. `size(collect(n.name))` returns 5 instead of 29
- Duplicate nodes when test directory overlaps with source root (e.g. `root/tests/` inside `root/`) — test roots already covered by a parent source root are now skipped, with `is_test` flags applied to the existing entities instead
- Duplicate Dependency ID collision when same package appears in multiple optional groups — IDs now include the group name (e.g. `matplotlib::viz`)

## [0.5.42] - 2026-02-19

### Added

- `connection_types` parameter for `louvain` and `label_propagation` procedures — filter edges by type, matching the existing support in centrality algorithms

### Fixed

- `CALL pagerank({connection_types: ['CALLS']})` list literal syntax now works correctly — was silently serialized as JSON string causing zero edge matches and uniform scores
- Document list comprehension patterns as unsupported in Cypher reference

## [0.5.41] - 2026-02-19

### Added

- Cypher string functions: `split(str, delim)`, `replace(str, search, repl)`, `substring(str, start [, len])`, `left(str, n)`, `right(str, n)`, `trim(str)`, `ltrim(str)`, `rtrim(str)`, `reverse(str)`

### Fixed

- Duplicate File nodes when source and test roots overlap in code_tree (e.g. `xarray/` source root containing `xarray/tests/` + separate test root)
- Empty `Module.path` properties for declared submodules in code_tree — now resolved from parsed files or inferred from parent directory
- Boolean properties (`is_test`, `is_abstract`, `is_async`, etc.) stored as string `'True'` instead of actual booleans — improved pandas `object` dtype detection to recognize boolean-only columns

## [0.5.39] - 2026-02-19

### Added

- `read_only(True/False)` method to disable Cypher mutations (CREATE, SET, DELETE, REMOVE, MERGE). When enabled, `agent_describe()` omits mutation documentation, simplifying the agent interface for read-only use cases

## [0.5.38] - 2026-02-19

### Added

- Cypher `CALL procedure({params}) YIELD columns` for graph algorithms: pagerank, betweenness, degree, closeness, louvain, label_propagation, connected_components. YIELD `node` is a node binding enabling `node.title`, `node.type` etc. in downstream WHERE/RETURN/ORDER BY clauses
- Inline pattern predicates in WHERE clauses — `WHERE (a)-[:REL]->(b)` and `WHERE NOT (a)-[:REL]->(b)` now work as shorthand for `EXISTS { ... }`, matching standard Cypher behavior
- `CALL list_procedures() YIELD name, description, yield_columns` — introspection procedure listing all available graph algorithm procedures with their parameters and descriptions

### Changed

- `build()` now includes test directories by default (`include_tests=True`)
- CALL procedure error message now hints at the correct map syntax when keyword arguments are used instead of `{key: value}` maps

### Fixed

- CALLS edge resolution in code_tree now uses tiered scope-aware matching (same owner > same file > same language > global) instead of flat bare-name lookup — eliminates false cross-class and cross-language edges
- Rust parser now detects test files at the File level (`_test.rs`, `test_*`, `tests/`, `benches/` conventions) — previously only function-level `#[test]` attributes were detected, leaving File nodes untagged

## [0.5.36] - 2026-02-18

### Changed

- Split `mod.rs` (6,742 LOC) into 5 thematic `#[pymethods]` files: algorithms, export, indexes, spatial, vector — mod.rs reduced to 4,005 LOC
- Enabled PyO3 `multiple-pymethods` feature for multi-file `#[pymethods]` blocks
- Documented transaction isolation semantics (snapshot isolation, last-writer-wins)

### Fixed

- `[n IN nodes(p) | n.name]` now correctly extracts node properties in list comprehensions over path functions — previously returned serialized JSON fragments instead of property values
- `parse_list_value` is now brace-aware — splits at top-level commas only, preserving JSON objects and nested structures
- `EXISTS { MATCH (pattern) }` syntax now accepted — the optional `MATCH` keyword inside EXISTS braces is silently skipped, matching standard Cypher behavior

## [0.5.35] - 2026-02-18

### Added

- CALLS edges now carry `call_lines` and `call_count` properties — line numbers where each call occurs in the caller function
- Comment annotation extraction (TODO/FIXME/HACK/NOTE/etc.) for all non-Rust parsers (Python, TypeScript, JavaScript, Java, Go, C, C++, C#)
- Test file detection (`is_test`) for all parsers based on language naming conventions
- Generic/type parameter extraction for Go 1.18+ and Python 3.12+ (PEP 695) parsers

## [0.5.34] - 2026-02-18

### Added

- `toc(file_path)` method: get a table of contents for any source file — all code entities sorted by line number with a type summary
- `find()` now accepts `match_type` parameter: `"exact"` (default), `"contains"` (case-insensitive substring), `"starts_with"` (case-insensitive prefix)
- `file_toc` MCP tool in `examples/mcp_server.py` for file-level exploration
- `find_entity` MCP tool now supports `match_type` parameter
- Qualified name format documented in `agent_describe()` output (Rust: `crate::module::Type::method`, Python: `package.module.Class.method`)
- Block doc comment support (`/** */`) in Rust parser — previously only `///` line comments were captured
- `call_trace` MCP tool in `examples/mcp_server.py` for tracing function call chains (outgoing/incoming, configurable depth)
- Call trace Cypher pattern documented in `agent_describe()` output
- CHANGELOG.md, CONTRIBUTING.md, and CLAUDE.md for project governance

### Changed

- Doc comments added to all critical Rust structs (`KnowledgeGraph`, `DirGraph`, `CypherExecutor`, `PatternExecutor`, `CypherParser`, and 15+ supporting types)
- Rust parser now captures all `use` declarations, not just `crate::` prefixed imports
- MCP tool descriptions improved with workflow guidance (`graph_overview` says "ALWAYS call this first", `cypher_query` mentions label-optional MATCH, etc.)
- GitHub Release workflow now uses CHANGELOG.md content instead of auto-generated notes

## [0.5.31] - 2025-05-15

### Added

- `find(name, node_type=None)` method: search code entities by name across all types
- `source(name)` method: resolve entity names to file paths and line ranges (supports single string or list)
- `context(name, hops=None)` method: get full neighborhood of a code entity grouped by relationship type
- `find_entity`, `read_source`, `entity_context` MCP tools in `examples/mcp_server.py`
- Label-optional MATCH documented in `agent_describe()` — `MATCH (n {name: 'x'})` searches all node types

### Changed

- Code entity helpers (`find`, `context`) moved from Python (`kglite/code_tree/helpers.py`) to native Rust methods for performance
- `agent_describe()` now conditionally shows code entity methods and notes when code entities are present in the graph

### Removed

- `kglite/code_tree/helpers.py` — replaced by native Rust methods on `KnowledgeGraph`

## [0.5.28] - 2025-05-10

### Added

- Manifest-based building: `build(".")` auto-detects `pyproject.toml` / `Cargo.toml` and reads project metadata (name, version, dependencies)
- `Project` and `Dependency` node types with `DEPENDS_ON` and `HAS_SOURCE` edges
- `USES_TYPE` edges: Function → type references in signatures
- `EXPOSES` edges: FFI boundary tracking (PyO3 modules → exposed items)

### Fixed

- Various code tree parser fixes for Rust trait implementations and method resolution

## [0.5.22] - 2025-04-28

### Added

- `kglite.code_tree` module: parse multi-language codebases into knowledge graphs using tree-sitter
- Supported languages: Rust, Python, TypeScript, JavaScript, Go, Java, C++, C#
- Node types: File, Module, Function, Struct, Class, Enum, Trait, Protocol, Interface, Constant
- Edge types: DEFINES, CALLS, HAS_METHOD, HAS_SUBMODULE, IMPLEMENTS, EXTENDS, IMPORTS
- Embedding export support

---

*For versions prior to 0.5.22, see [GitHub Releases](https://github.com/kkollsga/kglite/releases).*

[0.5.35]: https://github.com/kkollsga/kglite/compare/v0.5.34...v0.5.35
[0.5.34]: https://github.com/kkollsga/kglite/compare/v0.5.31...v0.5.34
[0.5.31]: https://github.com/kkollsga/kglite/compare/v0.5.28...v0.5.31
[0.5.28]: https://github.com/kkollsga/kglite/compare/v0.5.22...v0.5.28
[0.5.22]: https://github.com/kkollsga/kglite/releases/tag/v0.5.22
