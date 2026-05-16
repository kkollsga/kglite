"""Type stubs for kglite — a high-performance knowledge graph library."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional, Protocol, Union, overload, runtime_checkable

import pandas as pd

__version__: str

@runtime_checkable
class EmbeddingModel(Protocol):
    """Protocol for embedding models passed to ``embed_texts`` / ``search_text``.

    **Required** — ``dimension`` and ``embed()`` must be present.

    **Optional** — ``load()`` and ``unload()`` are called automatically if present:

    - ``load()`` is called before each ``embed_texts()`` / ``search_text()`` call.
    - ``unload()`` is called after each call completes (even on error).

    This lets models manage heavyweight resources (GPU memory, large weights)
    on demand.  A common pattern is to implement a cooldown in ``unload()``
    so the model stays warm across rapid successive calls but eventually
    releases memory after a period of inactivity.

    Example::

        import threading
        from sentence_transformers import SentenceTransformer

        class Embedder:
            def __init__(self, model_name="all-MiniLM-L6-v2"):
                self._model_name = model_name
                self._model = None
                self._timer = None
                self.dimension = 384  # known ahead of time, or set in load()

            def load(self):
                if self._timer:
                    self._timer.cancel()
                    self._timer = None
                if self._model is None:
                    self._model = SentenceTransformer(self._model_name)
                    self.dimension = self._model.get_sentence_embedding_dimension()

            def unload(self, cooldown=60):
                def _release():
                    self._model = None
                    self._timer = None
                self._timer = threading.Timer(cooldown, _release)
                self._timer.start()

            def embed(self, texts: list[str]) -> list[list[float]]:
                return self._model.encode(texts).tolist()
    """

    @property
    def dimension(self) -> int:
        """The dimensionality of the embedding vectors."""
        ...

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts, returning one vector per text."""
        ...

    def load(self) -> None:
        """(Optional) Load model weights / allocate resources.

        Called automatically before ``embed()`` in ``embed_texts()`` and
        ``search_text()``.  If not defined, this step is skipped.
        """
        ...

    def unload(self) -> None:
        """(Optional) Release model weights / free resources.

        Called automatically after ``embed_texts()`` and ``search_text()``
        complete (including on error).  A common pattern is to start a
        cooldown timer here instead of releasing immediately.
        """
        ...

class Agg:
    """Aggregation expression builders for ``add_properties()``.

    Each method returns the string expression that ``add_properties()``
    already understands, making the DSL discoverable via autocomplete
    instead of requiring users to know the string syntax.

    Example::

        from kglite import Agg

        graph.select('Well').traverse('HAS_BLOCK').add_properties({
            'Block': {'well_count': Agg.count(), 'avg_depth': Agg.mean('depth')}
        })

    Equivalent to the raw string form::

        graph.select('Well').traverse('HAS_BLOCK').add_properties({
            'Block': {'well_count': 'count(*)', 'avg_depth': 'mean(depth)'}
        })
    """

    @staticmethod
    def count() -> str:
        """Count leaf nodes per ancestor — returns ``'count(*)'``."""
        ...

    @staticmethod
    def sum(prop: str) -> str:
        """Sum a numeric property across leaves — returns ``'sum(prop)'``."""
        ...

    @staticmethod
    def mean(prop: str) -> str:
        """Arithmetic mean of a numeric property — returns ``'mean(prop)'``."""
        ...

    @staticmethod
    def min(prop: str) -> str:
        """Minimum value of a numeric property — returns ``'min(prop)'``."""
        ...

    @staticmethod
    def max(prop: str) -> str:
        """Maximum value of a numeric property — returns ``'max(prop)'``."""
        ...

    @staticmethod
    def std(prop: str) -> str:
        """Sample standard deviation — returns ``'std(prop)'``."""
        ...

    @staticmethod
    def collect(prop: str) -> str:
        """Comma-separated string of values — returns ``'collect(prop)'``."""
        ...

class Spatial:
    """Spatial compute expression builders for ``add_properties()``.

    Each method returns the string keyword that ``add_properties()``
    understands for spatial computations between leaf and ancestor nodes.

    Example::

        from kglite import Spatial

        graph.select('Well').compare('Structure', 'contains') \\
            .add_properties({
                'Well': {'dist': Spatial.distance(), 'a': Spatial.area()}
            })

    Equivalent to::

        graph.select('Well').compare('Structure', 'contains') \\
            .add_properties({
                'Well': {'dist': 'distance', 'a': 'area'}
            })
    """

    @staticmethod
    def distance() -> str:
        """Geodesic distance between leaf and ancestor (meters) — returns ``'distance'``."""
        ...

    @staticmethod
    def area() -> str:
        """Area of ancestor geometry (square meters) — returns ``'area'``."""
        ...

    @staticmethod
    def perimeter() -> str:
        """Perimeter of ancestor geometry (meters) — returns ``'perimeter'``."""
        ...

    @staticmethod
    def centroid_lat() -> str:
        """Latitude of ancestor geometry centroid — returns ``'centroid_lat'``."""
        ...

    @staticmethod
    def centroid_lon() -> str:
        """Longitude of ancestor geometry centroid — returns ``'centroid_lon'``."""
        ...

class ResultIter:
    """Iterator for ResultView. Converts one row per step."""

    def __iter__(self) -> ResultIter: ...
    def __next__(self) -> dict[str, Any]: ...

class ResultView:
    """Lazy result container — data stays in Rust until accessed from Python.

    Returned by ``cypher()``, centrality methods, ``collect()`` (flat),
    and ``sample()``.

    Data is only converted to Python objects when you actually access rows
    (via iteration, indexing, ``to_list()``, or ``to_df()``). This makes
    ``cypher()`` calls fast even for large result sets — the cost is deferred
    to when you consume the data.

    Supports:
      - ``len(result)`` — row count (O(1), no conversion)
      - ``bool(result)`` — True if non-empty
      - ``result[i]`` — single row as dict (converts that row only)
      - ``for row in result`` — iterate rows as dicts (one at a time)
      - ``result.head(n)`` / ``result.tail(n)`` — first/last n rows as a new ResultView
      - ``result.to_list()`` — all rows as ``list[dict]`` (full conversion)
      - ``result.to_df()`` — pandas DataFrame (full conversion)
      - ``result.columns`` — column names
      - ``result.stats`` — mutation stats (CREATE/SET/DELETE queries only)
    """

    @property
    def columns(self) -> list[str]:
        """Column names."""
        ...

    @property
    def stats(self) -> Optional[dict[str, int]]:
        """Mutation statistics, or ``None`` for read queries / non-cypher results."""
        ...

    @property
    def profile(self) -> Optional[list[dict[str, Any]]]:
        """PROFILE execution statistics, or ``None`` for non-profiled queries.

        Each dict has keys: ``clause`` (str), ``rows_in`` (int),
        ``rows_out`` (int), ``elapsed_us`` (int).

        Only populated when the query is prefixed with ``PROFILE``.
        """
        ...

    @property
    def diagnostics(self) -> Optional[dict[str, Any]]:
        """Lightweight execution diagnostics for this query, or ``None``
        for mutation paths / EXPLAIN / transactions.

        Returned dict keys:

        - ``elapsed_ms`` (int): wall-clock query duration in milliseconds.
        - ``timed_out`` (bool): ``True`` when the deadline fired (result
          rows reflect the partial set before cancellation).
        - ``timeout_ms`` (Optional[int]): the deadline that was in effect,
          or ``None`` when no deadline applied (memory graphs by default,
          or any call with ``timeout_ms=0``).

        Use this to tune ``timeout_ms`` or move toward anchored queries
        when your query repeatedly approaches the deadline.
        """
        ...

    def head(self, n: int = 5) -> ResultView:
        """Return a new ResultView with the first *n* rows (default 5)."""
        ...

    def tail(self, n: int = 5) -> ResultView:
        """Return a new ResultView with the last *n* rows (default 5)."""
        ...

    def to_list(self) -> list[dict[str, Any]]:
        """Convert all rows to a Python list of dicts (full materialization)."""
        ...

    def to_df(self) -> pd.DataFrame:
        """Convert to a pandas DataFrame."""
        ...

    def to_gdf(
        self,
        geometry_column: str = "geometry",
        crs: Optional[str] = None,
    ) -> Any:
        """Convert to a GeoDataFrame with a geometry column parsed from WKT.

        Materializes the data as a DataFrame, then converts the specified
        WKT string column into shapely geometries and returns a
        ``geopandas.GeoDataFrame``.

        Args:
            geometry_column: Column containing WKT strings. Default ``'geometry'``.
            crs: Coordinate reference system (e.g. ``'EPSG:4326'``), or ``None``.

        Returns:
            A ``geopandas.GeoDataFrame``.

        Raises:
            ImportError: If geopandas is not installed.
        """
        ...

    def __len__(self) -> int: ...
    def __bool__(self) -> bool: ...
    def __getitem__(self, index: int, /) -> dict[str, Any]: ...
    def __iter__(self) -> ResultIter: ...
    def __repr__(self) -> str: ...
    def __str__(self) -> str:
        """Vertical card format: one key-value per line, rows separated by blank lines."""
        ...

def load(path: str) -> KnowledgeGraph:
    """Load a graph from a binary file previously saved with ``save()``.

    Args:
        path: Path to the ``.kgl`` file.

    Returns:
        A new KnowledgeGraph with the loaded data.
    """
    ...

def cypher_pass_names() -> list[str]:
    """Names of every Cypher optimizer pass, in execution order.

    Pass names are stable identifiers for the diagnostic
    ``KnowledgeGraph.cypher(disabled_passes=[...])`` kwarg and for
    bisection scripts. Use this list as the source of truth — names not
    in it will be rejected by ``cypher(...)``.

    Returns:
        List of pass names in pipeline order.

    Example::

        passes = kglite.cypher_pass_names()
        # Bisect: disable each pass in turn, find the divergence.
        for name in passes:
            naive = g.cypher(query, disabled_passes=[name]).to_list()
            if naive != optimized:
                print(f"Divergence introduced by `{name}`")
                break
    """
    ...

def from_blueprint(
    blueprint_path: Union[str, Path],
    *,
    verbose: bool = False,
    save: bool = True,
    lock_schema: bool = False,
    storage: str = "default",
    path: Optional[str] = None,
) -> KnowledgeGraph:
    """Build a KnowledgeGraph from a JSON blueprint and CSV files.

    The blueprint JSON describes all node types, properties, connections,
    timeseries, and data sources. CSV paths in the blueprint are resolved
    relative to ``settings.root``.

    **Diagnostic output**:

    - **Progress** (per-type counts, summary line) prints to **stdout**
      when ``verbose=True``. Edge counts come from the live graph, not
      the input-row tally — so they match
      ``MATCH ()-[r]->() RETURN count(r)``. When dedupe collapses input
      rows, the line is annotated as
      ``[T]: N edges (M input rows, K deduped)``.
    - **Warnings** (target node not found, null FK, type mismatch, etc.)
      are emitted as Python ``UserWarning`` objects (originating in Rust
      via PyO3). By default they reach **stderr**.

    To capture warnings to a file, use the standard Python pattern::

        import logging
        logging.captureWarnings(True)
        logging.basicConfig(
            filename="blueprint.log",
            level=logging.WARNING,
            format="%(asctime)s %(levelname)s %(message)s",
        )
        graph = kglite.from_blueprint("blueprint.json", verbose=True)
        # All warnings now in blueprint.log instead of stderr.

    Args:
        blueprint_path: Path to the blueprint JSON file.
        verbose: If True, print progress information during loading.
        save: If True and the blueprint specifies an ``output`` path,
            save the graph to that path after building.
        lock_schema: If True, lock the schema after loading. Cypher
            mutations will be validated against the blueprint's types
            and properties.

    Returns:
        A new KnowledgeGraph populated from the blueprint.

    Raises:
        FileNotFoundError: If the blueprint file is missing.
        ValueError: If the blueprint JSON is malformed.

    Example::

        import kglite

        graph = kglite.from_blueprint("blueprint.json", verbose=True)
    """
    ...

def to_neo4j(
    graph: KnowledgeGraph,
    uri: str,
    *,
    auth: Optional[tuple[str, str]] = None,
    database: str = "neo4j",
    batch_size: int = 5000,
    clear: bool = False,
    merge: bool = False,
    selection_only: Optional[bool] = None,
    verbose: bool = False,
) -> dict[str, Any]:
    """Push graph data to a Neo4j database.

    Extracts all nodes and edges (or the current selection) and writes
    them to Neo4j using batched ``UNWIND`` operations for performance.

    Requires the ``neo4j`` package: ``pip install neo4j``.

    Args:
        graph: The KnowledgeGraph to export.
        uri: Neo4j connection URI (e.g. ``"bolt://localhost:7687"``).
        auth: Tuple of ``(username, password)``. ``None`` for no auth.
        database: Neo4j database name (default ``"neo4j"``).
        batch_size: Nodes/relationships per UNWIND batch (default 5000).
        clear: If ``True``, delete all existing data before import.
        merge: If ``True``, use MERGE instead of CREATE (upsert semantics).
        selection_only: If ``True``, export only selected nodes.
            Default: auto-detect from active selection.
        verbose: Print progress information.

    Returns:
        Summary dict with ``nodes_created``, ``relationships_created``,
        ``constraints_created``, ``elapsed``, ``database``.

    Example::

        import kglite

        g = kglite.load("graph.kgl")
        kglite.to_neo4j(g, "bolt://localhost:7687", auth=("neo4j", "password"))
    """
    ...

class KnowledgeGraph:
    """A high-performance knowledge graph with typed nodes, connections, and
    a fluent query API backed by Rust.
    """

    # ====================================================================
    # Constructor
    # ====================================================================

    def __new__(
        cls,
        *,
        storage: str | None = None,
        path: str | None = None,
    ) -> "KnowledgeGraph":
        """Create an empty KnowledgeGraph.

        Args:
            storage: Storage mode. ``None`` (default) uses heap-resident
                storage, optimal for small-to-medium graphs. ``"mapped"``
                uses mmap-backed columnar storage from the start, designed
                for large graphs that may approach or exceed available RAM.
                ``"disk"`` uses fully disk-backed storage for very large
                graphs (100M+ nodes). Requires ``path``.
            path: Directory path for disk-mode storage. Required when
                ``storage="disk"``. The directory IS the graph — data is
                written directly to disk via mmap. Load with
                ``kglite.load(path)``.
        """
        ...

    # ====================================================================
    # Properties
    # ====================================================================

    @property
    def node_types(self) -> list[str]:
        """List of node type names present in the graph."""
        ...

    @property
    def last_mutation_stats(self) -> Optional[dict[str, int]]:
        """Mutation statistics from the last Cypher mutation query.

        Returns ``None`` if no mutation has been executed yet.
        Keys: ``nodes_created``, ``relationships_created``, ``properties_set``,
        ``nodes_deleted``, ``relationships_deleted``, ``properties_removed``.
        """
        ...

    @property
    def is_columnar(self) -> bool:
        """Whether node properties are stored in columnar format.

        Returns ``True`` if ``enable_columnar()`` has been called (or the graph
        was loaded from a v3 ``.kgl`` file).
        """
        ...

    # ====================================================================
    # Data Loading
    # ====================================================================

    def add_nodes(
        self,
        data: pd.DataFrame,
        node_type: str,
        unique_id_field: str,
        node_title_field: Optional[str] = None,
        columns: Optional[list[str]] = None,
        conflict_handling: Optional[str] = None,
        skip_columns: Optional[list[str]] = None,
        column_types: Optional[dict[str, str]] = None,
        timeseries: Optional[dict[str, Any]] = None,
        nullable_int_downcast: bool = False,
    ) -> dict[str, Any]:
        """Add nodes from a DataFrame.

        String and integer IDs are auto-detected from the DataFrame dtype.
        Non-contiguous DataFrame indexes (e.g. from filtering) are handled
        automatically.

        When ``timeseries`` is provided, the DataFrame may contain multiple
        rows per unique ID (one per time step). Rows are deduplicated
        automatically — the first occurrence per ID provides static node
        properties, and all rows contribute to the timeseries channels.

        Args:
            data: DataFrame containing node data.
            node_type: Label for this set of nodes (e.g. ``'Person'``).
            unique_id_field: Column used as unique identifier.
            node_title_field: Column used as display title. Defaults to ``unique_id_field``.
            columns: Whitelist of columns to include. ``None`` = all.
            conflict_handling: ``'update'`` (default), ``'replace'``, ``'skip'``,
                ``'preserve'``, or ``'sum'``. ``'sum'`` acts as ``'update'`` for nodes.
            skip_columns: Columns to exclude.
            column_types: Override column dtypes ``{'col': 'string'|'integer'|'float'|'datetime'|'uniqueid'}``.
                Also supports spatial types: ``'location.lat'``, ``'location.lon'``,
                ``'geometry'``, ``'point.<name>.lat'``, ``'point.<name>.lon'``,
                ``'shape.<name>'``.
            nullable_int_downcast: When ``True``, Float64 columns whose non-null
                values are all integer-valued (e.g. ``pd.NA``-bearing ints that
                pandas auto-promoted to float64) are silently downcast to Int64.
                Default ``False`` — explicit opt-in protects existing callers.
            timeseries: Inline timeseries configuration dict with keys:

                - ``time`` (required): column name containing date strings
                  (``'yyyy-mm'``, ``'yyyy-mm-dd'``, ``'yyyy-mm-dd hh:mm'``),
                  or a dict mapping ``year``/``month``/``day``/``hour``/``minute``
                  to column names (e.g. ``{'year': 'ar', 'month': 'maned'}``).
                - ``channels`` (required): list of column names for timeseries
                  data (e.g. ``['oil', 'gas', 'condensate']``).
                - ``resolution`` (optional): ``'year'``, ``'month'``, ``'day'``,
                  ``'hour'``, or ``'minute'``. Auto-detected from time format if omitted.
                - ``units`` (optional): dict mapping channel names to unit strings
                  (e.g. ``{'oil': 'MSm3'}``).

        Returns:
            Operation report dict with keys ``nodes_created``,
            ``nodes_updated``, ``nodes_skipped``, ``processing_time_ms``,
            ``has_errors``, and optionally ``errors`` with skip reasons.

        Example::

            graph.add_nodes(df, 'Production', 'field_id', 'field_name',
                timeseries={
                    'time': 'date',
                    'channels': ['oil', 'gas', 'condensate', 'oe'],
                })
        """
        ...

    def add_connections(
        self,
        data: Optional[pd.DataFrame],
        connection_type: str,
        source_type: str,
        source_id_field: str,
        target_type: str,
        target_id_field: str,
        source_title_field: Optional[str] = None,
        target_title_field: Optional[str] = None,
        columns: Optional[list[str]] = None,
        skip_columns: Optional[list[str]] = None,
        conflict_handling: Optional[str] = None,
        column_types: Optional[dict[str, str]] = None,
        query: Optional[str] = None,
        extra_properties: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Add connections (edges) between existing nodes.

        Two modes — supply **either** ``data`` (a pandas DataFrame) **or**
        ``query`` (a Cypher string whose RETURN columns provide source/target IDs).

        Example (from DataFrame)::

            graph.add_connections(df, 'KNOWS', 'Person', 'src_id', 'Person', 'tgt_id')

        Example (from Cypher query)::

            graph.add_connections(
                None, 'ENCLOSES', 'Play', 'play_id', 'StructuralElement', 'struct_id',
                query=\"\"\"
                    MATCH (p:Play), (s:StructuralElement)
                    WHERE contains(p, s)
                    RETURN DISTINCT p.id AS play_id, s.id AS struct_id
                \"\"\",
            )

        Example (query with extra properties)::

            graph.add_connections(
                None, 'HC_IN_FORMATION', 'Discovery', 'src', 'Stratigraphy', 'tgt',
                query='MATCH ... RETURN d.id AS src, s.id AS tgt',
                extra_properties={'hc_rank': 1},
            )

        Args:
            data: DataFrame containing edge data, or ``None`` when using ``query``.
            connection_type: Label for this edge type (e.g. ``'KNOWS'``).
            source_type: Node type of source nodes.
            source_id_field: Column with source node IDs (must appear in DataFrame or query RETURN).
            target_type: Node type of target nodes.
            target_id_field: Column with target node IDs (must appear in DataFrame or query RETURN).
            source_title_field: Optional title column for source nodes.
            target_title_field: Optional title column for target nodes.
            columns: Whitelist of property columns to include (data mode only).
            skip_columns: Columns to exclude (data mode only).
            conflict_handling: ``'update'`` (default), ``'replace'``, ``'skip'``,
                ``'preserve'``, or ``'sum'``. ``'sum'`` adds numeric edge properties
                (Int64+Int64, Float64+Float64; mixed promotes to Float64).
                Non-numeric properties overwrite like ``'update'``.
            column_types: Override column dtypes (data mode only).
            query: Cypher query string (alternative to ``data``). Must be a
                read-only query whose RETURN clause includes columns matching
                ``source_id_field`` and ``target_id_field``.
            extra_properties: Dict of static properties to add to every edge
                created from the query results (query mode only).

        Returns:
            Operation report dict with ``connections_created``, ``connections_skipped``, etc.
        """
        ...

    def add_nodes_bulk(self, nodes: list[dict[str, Any]]) -> dict[str, int]:
        """Add multiple node types at once.

        Each dict in *nodes* must contain ``node_type``, ``unique_id_field``,
        ``node_title_field``, and ``data`` (a DataFrame).

        Returns:
            Mapping of ``node_type`` to count of nodes added.
        """
        ...

    def add_connections_bulk(self, connections: list[dict[str, Any]]) -> dict[str, int]:
        """Add multiple connection types at once.

        Each dict must contain ``source_type``, ``target_type``,
        ``connection_name``, and ``data`` (DataFrame with ``source_id``/``target_id`` columns).

        Returns:
            Mapping of ``connection_name`` to count of connections created.
        """
        ...

    def add_connections_from_source(self, connections: list[dict[str, Any]]) -> dict[str, int]:
        """Add connections, auto-filtering to types already loaded in the graph.

        Same spec format as :meth:`add_connections_bulk`, but silently skips
        connection specs whose source or target type is not in the graph.

        Returns:
            Mapping of ``connection_name`` to count of connections created.
        """
        ...

    # ====================================================================
    # Selection & Filtering
    # ====================================================================

    def select(
        self,
        node_type: str,
        sort: Optional[Union[str, list[tuple[str, bool]]]] = None,
        limit: Optional[int] = None,
        temporal: Optional[bool] = None,
    ) -> KnowledgeGraph:
        """Select all nodes of a given type.

        When a temporal config exists for this node type (via ``set_temporal()``),
        nodes are auto-filtered to those valid at the reference date (today or
        ``date()`` context). Pass ``temporal=False`` to include all nodes.

        Args:
            node_type: The node type to select (e.g. ``'Person'``).
            sort: Optional sort spec — a property name or list of ``(field, ascending)`` tuples.
            limit: Limit the number of selected nodes.
            temporal: Override temporal filtering. ``None`` = auto (filter if configured),
                ``False`` = disable, ``True`` = require (error if not configured).

        Returns:
            A new KnowledgeGraph with the filtered selection.
        """
        ...

    def where(
        self,
        conditions: dict[str, Any],
        sort: Optional[Union[str, list[tuple[str, bool]]]] = None,
        limit: Optional[int] = None,
    ) -> KnowledgeGraph:
        """Filter the current selection by property conditions.

        Conditions support exact match, comparison operators
        (``'>'``, ``'<'``, ``'>='``, ``'<='``), ``'in'``, ``'is_null'``,
        ``'is_not_null'``, ``'contains'``, ``'starts_with'``, ``'ends_with'``,
        ``'regex'`` (or ``'=~'``), and negated variants: ``'not_contains'``,
        ``'not_starts_with'``, ``'not_ends_with'``, ``'not_in'``, ``'not_regex'``.

        Example::

            graph.select('Person').where({
                'age': {'>': 25},
                'city': 'Oslo',
                'name': {'regex': '^A.*'},
                'status': {'not_in': ['inactive', 'banned']},
            })

        Returns:
            A new KnowledgeGraph with the filtered selection.
        """
        ...

    def where_any(
        self,
        conditions: list[dict[str, Any]],
        sort: Optional[Union[str, list[tuple[str, bool]]]] = None,
        limit: Optional[int] = None,
    ) -> KnowledgeGraph:
        """Filter the current selection with OR logic across multiple condition sets.

        Each dict in *conditions* is a set of AND conditions (same as ``where()``).
        A node is kept if it matches **any** of the condition sets.

        Args:
            conditions: List of condition dicts. Must contain at least one.
            sort: Optional sort spec.
            limit: Limit the number of selected nodes.

        Returns:
            A new KnowledgeGraph with the filtered selection.

        Example::

            graph.select('Person').where_any([
                {'city': 'Oslo'},
                {'city': 'Bergen'},
            ])

        Raises:
            ValueError: If *conditions* is empty.
        """
        ...

    def where_orphans(
        self,
        include_orphans: Optional[bool] = None,
        sort: Optional[Union[str, list[tuple[str, bool]]]] = None,
        limit: Optional[int] = None,
    ) -> KnowledgeGraph:
        """Filter nodes based on whether they have connections.

        Args:
            include_orphans: If ``True``, keep only orphan (disconnected) nodes.
                If ``False``, keep only connected nodes. Default ``True``.
            sort: Optional sort spec.
            limit: Limit the number of selected nodes.

        Returns:
            A new KnowledgeGraph with the filtered selection.
        """
        ...

    def sort(
        self,
        sort: Union[str, list[tuple[str, bool]]],
        ascending: Optional[bool] = None,
    ) -> KnowledgeGraph:
        """Sort the current selection.

        Args:
            sort: Property name (string) or list of ``(field, ascending)`` tuples.
            ascending: Direction when *sort* is a single string. Default ``True``.

        Returns:
            A new KnowledgeGraph with the sorted selection.
        """
        ...

    def limit(self, max_per_group: int) -> KnowledgeGraph:
        """Limit the number of nodes per parent group.

        Args:
            max_per_group: Maximum number of nodes to keep per group.

        Returns:
            A new KnowledgeGraph with the limited selection.
        """
        ...

    def offset(self, n: int) -> KnowledgeGraph:
        """Skip the first *n* nodes per parent group (pagination).

        Combine with ``limit()`` for pagination:
        ``graph.sort('name').offset(20).limit(10)``

        Args:
            n: Number of nodes to skip.

        Returns:
            A new KnowledgeGraph with the offset selection.
        """
        ...

    def where_connected(
        self,
        connection_type: str,
        direction: Optional[str] = None,
    ) -> KnowledgeGraph:
        """Filter nodes that have at least one connection of the given type.

        Keeps only nodes from the current selection that participate in
        edges of the specified type and direction.

        Args:
            connection_type: Edge type to check (e.g. ``'KNOWS'``).
            direction: ``'outgoing'``, ``'incoming'``, or ``'any'`` (default).

        Returns:
            A new KnowledgeGraph with only connected nodes.

        Raises:
            ValueError: If *direction* is not one of the valid values.
        """
        ...

    def valid_at(
        self,
        date: Optional[str] = None,
        date_from_field: Optional[str] = None,
        date_to_field: Optional[str] = None,
    ) -> KnowledgeGraph:
        """Filter nodes valid at a specific date.

        Keeps nodes where ``date_from <= date <= date_to``.

        If field names are not specified, auto-detects from ``set_temporal()`` config.
        If *date* is not specified, uses the ``date()`` context or today.

        Args:
            date: Date string (e.g. ``'2024-01-15'``). Defaults to reference date or today.
            date_from_field: Name of the start-date property. Auto-detected if temporal config exists.
            date_to_field: Name of the end-date property. Auto-detected if temporal config exists.

        Returns:
            A new KnowledgeGraph with the filtered selection.
        """
        ...

    def valid_during(
        self,
        start_date: str,
        end_date: str,
        date_from_field: Optional[str] = None,
        date_to_field: Optional[str] = None,
    ) -> KnowledgeGraph:
        """Filter nodes whose validity period overlaps a date range.

        If field names are not specified, auto-detects from ``set_temporal()`` config.

        Args:
            start_date: Start of the query range.
            end_date: End of the query range.
            date_from_field: Name of the start-date property. Auto-detected if temporal config exists.
            date_to_field: Name of the end-date property. Auto-detected if temporal config exists.

        Returns:
            A new KnowledgeGraph with the filtered selection.
        """
        ...

    # ====================================================================
    # Update
    # ====================================================================

    def update(
        self,
        properties: dict[str, Any],
        keep_selection: Optional[bool] = None,
    ) -> dict[str, Any]:
        """Batch-update properties on all selected nodes.

        Args:
            properties: Mapping of property names to new values.
            keep_selection: Preserve the current selection in the returned graph. Default ``False``.

        Returns:
            Dict with ``graph`` (updated KnowledgeGraph), ``nodes_updated`` (int),
            and ``report_index`` (int).
        """
        ...

    # ====================================================================
    # Data Retrieval
    # ====================================================================

    def collect(
        self,
        limit: Optional[int] = None,
    ) -> ResultView:
        """Materialise selected nodes as a flat ``ResultView``.

        For grouped output by parent type, use ``collect_grouped()`` instead.

        Args:
            limit: Maximum number of nodes to return.

        Returns:
            A ``ResultView`` containing ``id``, ``title``, ``type``,
            and all stored properties for each selected node.
        """
        ...

    def collect_grouped(
        self,
        group_by: str,
        *,
        parent_info: bool = False,
        flatten_single_parent: bool = True,
        limit: Optional[int] = None,
    ) -> dict[str, Any]:
        """Materialise selected nodes grouped by a parent type in the
        traversal hierarchy.

        Args:
            group_by: Parent node type to group by (must exist in the
                traversal chain).
            parent_info: Include parent metadata (``type``, ``id``,
                ``title``) in each group. Default ``False``.
            flatten_single_parent: If only one parent group exists,
                return a flat list instead of a single-key dict.
                Default ``True``.
            limit: Maximum number of nodes to return.

        Returns:
            Dict mapping parent title → list of node dicts. If
            ``flatten_single_parent`` is ``True`` and there is only one
            parent, returns a flat list.

        Examples::

            # Group wells by their parent field
            graph.select('Field').traverse('HAS_WELL') \\
                .collect_grouped('Field')
            # → {'TROLL': [...], 'EKOFISK': [...]}

            # Include parent metadata
            graph.select('Field').traverse('HAS_WELL') \\
                .collect_grouped('Field', parent_info=True)
        """
        ...

    def to_df(
        self,
        *,
        include_type: bool = True,
        include_id: bool = True,
    ) -> pd.DataFrame:
        """Export current selection as a pandas DataFrame.

        Each node becomes a row with columns for title, type, id, and all
        properties. Missing properties across different node types become None.

        Args:
            include_type: Include ``type`` column. Default ``True``.
            include_id: Include ``id`` column. Default ``True``.

        Returns:
            DataFrame with one row per selected node.
        """
        ...

    def to_str(self, limit: int = 50) -> str:
        """Format the current selection as a human-readable string.

        Each node is printed as a block with ``[Type] title (id: x)``
        header and indented properties, one per line.

        Args:
            limit: Maximum number of nodes to show. Default ``50``.
        """
        ...

    def show(
        self,
        columns: list[str] | None = None,
        limit: int = 200,
    ) -> str:
        """Display selected nodes with specific properties in a compact format.

        Single level (no traversals): one node per line as ``Type(val1, val2)``.
        Multi-level (after traverse): walks the full chain as
        ``Type1(vals) -> Type2(vals) -> Type3(vals)``.

        Args:
            columns: Property names to include. Default ``["id", "title"]``.
            limit: Maximum output lines. Default ``200``.

        Example::

            print(graph.select("Discovery").show(["id", "title"]))
            # Discovery(123, Johan Sverdrup)

            print(graph.select("Discovery")
                .traverse("IN_FIELD")
                .traverse("DISCOVERY_WELLBORE")
                .show(["id"]))
            # Discovery(123) -> Field(456) -> Wellbore(789)
        """
        ...

    def len(self) -> int:
        """Count selected nodes without materialising them.

        Much faster than ``len(collect())``. Also available via ``len(graph)``.
        """
        ...

    def __len__(self) -> int: ...
    def indices(self) -> list[int]:
        """Return raw graph indices for selected nodes."""
        ...

    def ids(self) -> list[Any]:
        """Return a flat list of ID values from the current selection.

        The lightest retrieval method — no dict wrapping.
        """
        ...

    def node(self, node_type: str, node_id: Any) -> Optional[dict[str, Any]]:
        """Look up a single node by type and ID. O(1) via hash index.

        Args:
            node_type: The node type (e.g. ``'User'``).
            node_id: The unique ID value.

        Returns:
            Node property dict, or ``None`` if not found.
        """
        ...

    def find(
        self,
        name: str,
        node_type: Optional[str] = None,
        match_type: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Find code entities by name, with disambiguation context.

        ⚠ **Code-entity search only.** ``find()`` searches nodes of type
        ``Function``, ``Struct``, ``Class``, ``Enum``, ``Trait``,
        ``Protocol``, ``Interface``, ``Module``, or ``Constant`` — the
        types produced by ``kglite.code_tree``. On graphs that don't
        contain these types (e.g. a social graph with ``Person`` nodes),
        ``find()`` returns an empty list. For general name lookup on
        other node types, use ``select(type).where({"name": ...})`` or
        ``cypher("MATCH (n:Type) WHERE n.name = $n RETURN n",
        params={"n": ...})``.

        Args:
            name: Entity name to search for (e.g. ``"execute"``).
            node_type: Optional filter — only search this node type
                (e.g. ``"Function"``, ``"Struct"``).
            match_type: Matching strategy: ``"exact"`` (default),
                ``"contains"`` (case-insensitive substring), or
                ``"starts_with"`` (case-insensitive prefix).

        Returns:
            List of dicts with: type, name, qualified_name, file_path,
            line_number, and optionally signature and visibility.
        """
        ...

    @overload
    def source(
        self,
        name: str,
        node_type: Optional[str] = None,
    ) -> dict[str, Any]:
        """Get the source location of one or more code entities.

        Resolves names or qualified names to code entities and returns
        file paths and line ranges.

        Args:
            name: Entity name, qualified name, or list of names.
            node_type: Optional node type hint.

        Returns:
            Single name: dict with ``file_path``, ``line_number``,
            ``end_line``, ``line_count``, ``name``, ``qualified_name``,
            ``type``, ``signature``.
            List of names: list of such dicts.
        """
        ...
    @overload
    def source(
        self,
        name: list[str],
        node_type: Optional[str] = None,
    ) -> list[dict[str, Any]]: ...
    def context(
        self,
        name: str,
        node_type: Optional[str] = None,
        hops: Optional[int] = None,
    ) -> dict[str, Any]:
        """Get the full neighborhood of a code entity.

        Returns the node's properties and all related entities grouped by
        relationship type. If the name is ambiguous, returns the matches
        so you can refine with a qualified name.

        Args:
            name: Entity name or qualified name.
            node_type: Optional node type hint.
            hops: Max traversal depth (default 1).

        Returns:
            Dict with ``"node"`` (properties), ``"defined_in"`` (file path),
            and relationship groups (e.g. ``"HAS_METHOD"``, ``"CALLS"``,
            ``"called_by"``).
        """
        ...

    def toc(
        self,
        file_path: str,
    ) -> dict[str, Any]:
        """Get a table of contents for a file — all code entities defined in it.

        Returns entities sorted by line number with a type summary.

        Args:
            file_path: Path of the file (the File node's id/path).

        Returns:
            Dict with ``"file"`` (path), ``"entities"`` (list of entity dicts
            sorted by line_number, each with type, name, qualified_name,
            line_number, end_line, and optionally signature), and ``"summary"``
            (dict of type name to count).
        """
        ...

    def build_id_indices(self, node_types: Optional[list[str]] = None) -> None:
        """Pre-build ID lookup indices for fast :meth:`node` calls.

        Args:
            node_types: Types to index. ``None`` indexes all types.
        """
        ...

    def node_type_counts(self) -> dict[str, int]:
        """Get node counts per type without materialising nodes.

        Returns:
            Dict mapping node type name to count.
        """
        ...

    # Graph Maintenance
    # -----------------

    def reindex(self) -> None:
        """Rebuild all indexes from the current graph state.

        Reconstructs type_indices, property_indices, and composite_indices by
        scanning all live nodes. Clears lazy caches (id_indices, connection_types)
        so they rebuild on next access.

        Use after bulk mutations (especially Cypher DELETE/REMOVE) to ensure
        index consistency.

        Example::

            graph.reindex()
        """
        ...

    def vacuum(self) -> dict[str, Any]:
        """Compact the graph by removing tombstones left by node/edge deletions.

        With StableDiGraph, deletions leave holes in the internal storage.
        Over time, this wastes memory and degrades iteration performance.
        ``vacuum()`` rebuilds the graph with contiguous indices, then rebuilds
        all indexes. If columnar storage is active, column stores are also
        rebuilt to eliminate orphaned rows from deleted nodes.

        **Important**: This resets the current selection since node indices change.
        Call this between query chains, not in the middle of one.

        Returns:
            dict with keys:
                - ``nodes_remapped``: Number of nodes that were remapped
                - ``tombstones_removed``: Number of tombstone slots reclaimed
                - ``columnar_rebuilt``: Whether columnar stores were rebuilt

        Example::

            info = graph.graph_info()
            if info['fragmentation_ratio'] > 0.3:
                result = graph.vacuum()
                print(f"Reclaimed {result['tombstones_removed']} slots")
        """
        ...

    def set_auto_vacuum(self, threshold: float | None) -> None:
        """Configure automatic vacuum after DELETE operations.

        When enabled, the graph automatically compacts itself after Cypher DELETE
        operations if the fragmentation ratio exceeds the threshold and there are
        more than 100 tombstones.

        Args:
            threshold: A float between 0.0 and 1.0, or ``None`` to disable.
                Default is ``0.3`` (30% fragmentation triggers vacuum).

        Example::

            graph.set_auto_vacuum(0.2)   # more aggressive — vacuum at 20%
            graph.set_auto_vacuum(None)  # disable auto-vacuum
            graph.set_auto_vacuum(0.3)   # restore default
        """
        ...

    def read_only(self, enabled: bool | None = None) -> bool:
        """Set or query read-only mode for the Cypher layer.

        When enabled, all Cypher mutation queries (CREATE, SET, DELETE, REMOVE,
        MERGE) are rejected, and ``describe()`` omits mutation docs.

        Args:
            enabled: If ``True``, enable read-only mode. If ``False``, disable.
                If omitted, return the current state without changing it.

        Returns:
            The current read-only state (after applying the change, if any).

        Example::

            graph.read_only(True)   # lock the graph
            graph.read_only()       # -> True
            graph.read_only(False)  # unlock
        """
        ...

    def lock_schema(self) -> KnowledgeGraph:
        """Lock the schema: future Cypher mutations must conform to current types.

        When locked, CREATE, SET, and MERGE operations are validated against
        the graph's known node types, connection types, and property types.
        Invalid writes return descriptive errors with 'did you mean?' suggestions.

        Returns:
            Self for method chaining.

        Example::

            graph.lock_schema()
            graph.cypher("CREATE (p:Typo {name: 'x'})")  # raises RuntimeError

        See Also:
            :meth:`unlock_schema`, :attr:`schema_locked`
        """
        ...

    def unlock_schema(self) -> KnowledgeGraph:
        """Unlock the schema: allow any Cypher mutations without validation.

        Returns:
            Self for method chaining.
        """
        ...

    @property
    def schema_locked(self) -> bool:
        """Whether the schema is currently locked."""
        ...

    def graph_info(self) -> dict[str, Any]:
        """Get diagnostic information about graph storage health.

        Returns a dictionary with storage metrics useful for deciding when
        to call :meth:`vacuum` or :meth:`reindex`.

        Returns:
            dict with keys:
                - ``node_count``: Number of live nodes
                - ``node_capacity``: Upper bound of node indices (includes tombstones)
                - ``node_tombstones``: Number of wasted slots from deletions
                - ``edge_count``: Number of live edges
                - ``fragmentation_ratio``: Ratio of wasted storage (0.0 = clean)
                - ``type_count``: Number of distinct node types
                - ``property_index_count``: Number of single-property indexes
                - ``composite_index_count``: Number of composite indexes
                - ``columnar_heap_bytes``: Heap-resident bytes in columnar stores
                - ``columnar_is_mapped``: Whether any columnar data is file-backed
                - ``memory_limit``: Configured memory limit (None if unset)
                - ``columnar_total_rows``: Total rows in columnar stores (includes orphaned)
                - ``columnar_live_rows``: Rows backed by live nodes

        Example::

            info = graph.graph_info()
            if info['fragmentation_ratio'] > 0.3:
                graph.vacuum()
        """
        ...

    def connections(
        self,
        indices: Optional[list[int]] = None,
        parent_info: Optional[bool] = None,
        include_node_properties: Optional[bool] = None,
        flatten_single_parent: bool = True,
    ) -> dict[str, Any]:
        """Get connections for selected nodes.

        Args:
            indices: Specific node indices to query.
            parent_info: Include parent info in output.
            include_node_properties: Include properties of connected nodes. Default ``True``.
            flatten_single_parent: Flatten when only one parent. Default ``True``.

        Returns:
            Nested dict ``{title: {node_id, type, incoming, outgoing}}``.
        """
        ...

    def titles(
        self,
        limit: Optional[int] = None,
        indices: Optional[list[int]] = None,
        flatten_single_parent: Optional[bool] = None,
    ) -> Union[list[str], dict[str, list[str]]]:
        """Get titles of selected nodes.

        Without traversal (single parent group), returns a flat list of titles.
        After traversal (multiple parent groups), returns ``{parent_title: [titles]}``.

        Args:
            flatten_single_parent: Flatten single-group results to a list. Default ``True``.

        Returns:
            ``list[str]`` when flattened, ``dict[str, list[str]]`` when grouped.
        """
        ...

    def explain(self) -> str:
        """Return a human-readable execution plan for the current query chain.

        Example output::

            SELECT Person (500 nodes) -> WHERE (42 nodes)
        """
        ...

    def get_properties(
        self,
        properties: list[str],
        limit: Optional[int] = None,
        indices: Optional[list[int]] = None,
        flatten_single_parent: Optional[bool] = None,
    ) -> Union[list[tuple[Any, ...]], dict[str, list[tuple[Any, ...]]]]:
        """Get specific properties for selected nodes.

        Without traversal (single parent group), returns a flat list of tuples.
        After traversal (multiple parent groups), returns ``{parent_title: [tuples]}``.

        Args:
            properties: List of property names to retrieve.
            limit: Maximum number of nodes.
            indices: Specific node indices.
            flatten_single_parent: Flatten single-group results to a list. Default ``True``.

        Returns:
            ``list[tuple]`` when flattened, ``dict[str, list[tuple]]`` when grouped.
        """
        ...

    def unique_values(
        self,
        property: str,
        group_by_parent: Optional[bool] = None,
        level_index: Optional[int] = None,
        indices: Optional[list[int]] = None,
        store_as: Optional[str] = None,
        max_length: Optional[int] = None,
        keep_selection: Optional[bool] = None,
    ) -> Any:
        """Get unique values of a property, optionally storing results.

        Args:
            property: Property name to extract unique values from.
            group_by_parent: Group by parent node. Default ``True``.
            level_index: Target level in the selection hierarchy.
            indices: Specific node indices.
            store_as: If set, stores comma-separated unique values as this property on parents.
            max_length: Max string length when storing.
            keep_selection: Preserve selection after store. Default ``False``.

        Returns:
            Dict of unique values per parent, or a KnowledgeGraph if ``store_as`` is set.
        """
        ...

    # ====================================================================
    # Traversal
    # ====================================================================

    def traverse(
        self,
        connection_type: str,
        level_index: Optional[int] = None,
        direction: Optional[str] = None,
        filter_target: Optional[dict[str, Any]] = None,
        filter_connection: Optional[dict[str, Any]] = None,
        sort_target: Optional[Union[str, list[tuple[str, bool]]]] = None,
        limit: Optional[int] = None,
        new_level: Optional[bool] = None,
        at: Optional[str] = None,
        during: Optional[tuple[str, str]] = None,
        temporal: Optional[bool] = None,
        target_type: Optional[Union[str, list[str]]] = None,
        where: Optional[dict[str, Any]] = None,
        where_connection: Optional[dict[str, Any]] = None,
    ) -> KnowledgeGraph:
        """Traverse connections to discover related nodes by following graph edges.

        For spatial, semantic, or clustering operations, use ``compare()`` instead.

        Args:
            connection_type: Edge type to follow (e.g. ``'HAS_LICENSEE'``).
            direction: ``'outgoing'``, ``'incoming'``, or ``None`` (both).
            target_type: Filter targets to specific node type(s). Accepts a
                string or list of strings. Useful when a connection type
                connects to multiple node types.
            where: Property conditions for **target nodes** — same operators
                as ``.where()`` (``'>'``, ``'contains'``, ``'in'``, etc.).
            where_connection: Property conditions for **edge properties**.
            sort_target: Sort targets per source. Field name or
                ``[(field, ascending)]`` list.
            limit: Max target nodes per source.
            at: Temporal point-in-time filter (e.g. ``'2005'``).
            during: Temporal range filter (e.g. ``('2000', '2010')``).
            temporal: Override temporal filtering. ``False`` = disable.
            level_index: Source level in the hierarchy (advanced).
            new_level: Add targets as new hierarchy level. Default ``True``.
            filter_target: Alias for ``where``.
            filter_connection: Alias for ``where_connection``.

        Returns:
            A new KnowledgeGraph with traversal results selected.

        Examples::

            # Follow edges
            graph.select('Field').traverse('HAS_LICENSEE')

            # Filter to specific target type
            graph.select('Field').traverse('OF_FIELD', direction='incoming',
                target_type='ProductionProfile')

            # Multiple target types
            graph.select('Field').traverse('OF_FIELD', direction='incoming',
                target_type=['ProductionProfile', 'FieldReserves'])

            # Filter target node properties
            graph.select('Field').traverse('HAS_LICENSEE',
                where={'title': 'Equinor Energy AS'})

            # Filter edge properties
            graph.select('Person').traverse('RATED',
                where_connection={'score': {'>': 4}})

            # Temporal filtering
            graph.select('Field').traverse('HAS_LICENSEE', at='2005')
            graph.select('Field').traverse('HAS_LICENSEE',
                during=('2000', '2010'))
        """
        ...

    def compare(
        self,
        target_type: Union[str, list[str]],
        method: Union[str, dict[str, Any]],
        *,
        filter: Optional[dict[str, Any]] = None,
        sort: Optional[Union[str, list[tuple[str, bool]]]] = None,
        limit: Optional[int] = None,
        level_index: Optional[int] = None,
        new_level: Optional[bool] = None,
    ) -> KnowledgeGraph:
        """Compare selected nodes against a target type using spatial, semantic,
        or clustering methods.

        Args:
            target_type: Node type to compare against (e.g. ``'Well'``).
            method: Comparison method — a string shorthand or a dict with
                settings:

                **Spatial methods:**

                - ``'contains'`` — point-in-polygon or polygon containment
                - ``'intersects'`` — polygon-polygon intersection
                - ``{'type': 'distance', 'max_m': 5000}`` — geodesic distance

                **Semantic methods:**

                - ``{'type': 'text_score', 'property': 'name'}`` — embedding similarity
                - ``{'type': 'text_score', 'threshold': 0.7}`` — with threshold
                - ``{'type': 'text_score', 'metric': 'poincare'}`` — Poincaré distance (hierarchical data)

                **Clustering methods:**

                - ``{'type': 'cluster', 'k': 5}`` — K-means clustering
                - ``{'type': 'cluster', 'algorithm': 'dbscan', 'eps': 0.5}`` — DBSCAN

                **Common dict keys:**

                - ``resolve``: ``'centroid'``, ``'closest'``, or ``'geometry'``
                - ``max_m``: Maximum distance in meters (distance method)
                - ``threshold``: Minimum similarity score (semantic methods)
                - ``k``: Number of clusters (K-means)
                - ``features``: Properties to cluster on
            filter: Property conditions for target nodes.
            sort: Sort results. Field name or ``[(field, ascending)]`` list.
            limit: Max target nodes per source.
            level_index: Source level in the hierarchy (advanced).
            new_level: Add targets as new hierarchy level. Default ``True``.

        Returns:
            A new KnowledgeGraph with comparison results selected.

        Examples::

            # Spatial containment: find wells inside structures
            graph.select('Structure').compare('Well', 'contains')

            # Distance: wells within 5km
            graph.select('Well').compare('Well',
                {'type': 'distance', 'max_m': 5000})

            # Semantic similarity
            graph.select('Document').compare('Document',
                {'type': 'text_score', 'property': 'summary', 'threshold': 0.7})

            # Clustering
            graph.select('Well').compare('Well',
                {'type': 'cluster', 'k': 5, 'features': ['latitude', 'longitude']})
        """
        ...

    def create_connections(
        self,
        connection_type: str,
        keep_selection: Optional[bool] = None,
        conflict_handling: Optional[str] = None,
        properties: Optional[dict[str, list[str]]] = None,
        source_type: Optional[str] = None,
        target_type: Optional[str] = None,
    ) -> KnowledgeGraph:
        """Create connections from the traversal hierarchy.

        By default, creates edges from the top-level ancestor (first
        traversal level) to the leaf nodes (last level). Use
        ``source_type`` / ``target_type`` to choose different levels.

        Args:
            connection_type: Label for the new edges (e.g. ``'A_TO_C'``).
            keep_selection: Preserve selection. Default ``False``.
            conflict_handling: ``'update'`` (default), ``'replace'``, ``'skip'``,
                ``'preserve'``, or ``'sum'``.
            properties: Copy properties from intermediate nodes onto the new
                edges. Dict mapping node type to property names:
                ``{'TypeB': ['score', 'weight']}``.
                An empty list copies all properties from that type.
            source_type: Node type to use as source (default: first level).
            target_type: Node type to use as target (default: last level).

        Returns:
            A new KnowledgeGraph with the connections added.

        Example::

            # After traversal A → B → C, create direct A → C edges
            # with B's 'score' property copied onto each edge
            graph.select('A') \\
                .traverse('REL_AB') \\
                .traverse('REL_BC') \\
                .create_connections('A_TO_C',
                    properties={'B': ['score']})
        """
        ...

    def add_properties(
        self,
        properties: dict[str, Union[list[str], dict[str, str]]],
        keep_selection: Optional[bool] = None,
    ) -> KnowledgeGraph:
        """Enrich selected nodes with properties from ancestor nodes in the traversal chain.

        Copies, renames, aggregates, or computes spatial properties from nodes
        at other levels of the selection hierarchy onto the current leaf nodes.

        Args:
            properties: Dict mapping source node type → property spec:

                - ``{'B': ['name', 'status']}`` — copy listed properties as-is
                - ``{'B': []}`` — copy all properties from B
                - ``{'B': {'new_name': 'old_name'}}`` — copy with rename
                - ``{'B': {'avg_depth': 'mean(depth)'}}`` — aggregate functions:
                  ``count(*)``, ``sum(prop)``, ``mean(prop)``, ``min(prop)``,
                  ``max(prop)``, ``std(prop)``, ``collect(prop)``
                - ``{'B': {'dist': 'distance'}}`` — spatial compute:
                  ``distance``, ``area``, ``perimeter``, ``centroid_lat``, ``centroid_lon``

            keep_selection: Preserve current selection. Default ``True``.

        Returns:
            A new KnowledgeGraph with the properties added to selected nodes.

        Examples::

            # Copy structure name onto wells
            graph.select('Structure').compare('Well', 'contains') \\
                .add_properties({'Structure': ['name', 'status']})

            # Rename properties
            graph.select('Structure').compare('Well', 'contains') \\
                .add_properties({'Structure': {'struct_name': 'name'}})

            # Aggregate with Agg helpers (discoverable via autocomplete)
            from kglite import Agg, Spatial
            graph.select('Structure').compare('Well', 'contains') \\
                .add_properties({'Well': {
                    'well_count': Agg.count(),
                    'avg_depth': Agg.mean('depth'),
                }})

            # Spatial compute with Spatial helpers
            graph.select('Structure').compare('Well', 'contains') \\
                .add_properties({'Structure': {
                    'dist_to_center': Spatial.distance(),
                    'parent_area': Spatial.area(),
                }})
        """
        ...

    def collect_children(
        self,
        property: Optional[str] = None,
        where: Optional[dict[str, Any]] = None,
        sort: Optional[Union[str, list[tuple[str, bool]]]] = None,
        limit: Optional[int] = None,
        store_as: Optional[str] = None,
        max_length: Optional[int] = None,
        keep_selection: Optional[bool] = None,
    ) -> Any:
        """Collect child-node property values into comma-separated lists.

        Args:
            property: Child property to collect. Default ``'title'``.
            where: Filter conditions for children.
            sort: Sort children.
            limit: Limit children per parent.
            store_as: If set, stores the list as this property on parent nodes.
            max_length: Max string length when storing.
            keep_selection: Preserve selection after store. Default ``False``.

        Returns:
            Dict of ``{parent_title: 'val1, val2, ...'}`` or a KnowledgeGraph
            if ``store_as`` is set.
        """
        ...

    # ====================================================================
    # Statistics & Calculations
    # ====================================================================

    def statistics(
        self,
        property: str,
        level_index: Optional[int] = None,
        group_by: Optional[str] = None,
    ) -> Any:
        """Compute descriptive statistics for a numeric property.

        Returns per-parent stats including count, mean, std, min, max, sum.

        Args:
            property: Numeric property name.
            level_index: Target level in the hierarchy.
            group_by: Group results by this property instead of by parent.
                Returns ``{group_value: {count, sum, mean, min, max, std}}``.
        """
        ...

    def calculate(
        self,
        expression: str,
        level_index: Optional[int] = None,
        store_as: Optional[str] = None,
        keep_selection: Optional[bool] = None,
        aggregate_connections: Optional[bool] = None,
    ) -> Any:
        """Evaluate a mathematical expression on selected nodes.

        Supports property references, arithmetic operators, and aggregate
        functions (``sum``, ``mean``, ``std``, ``min``, ``max``, ``count``).

        Args:
            expression: Expression string, e.g. ``'price * quantity'`` or ``'mean(age)'``.
            level_index: Target level in the hierarchy.
            store_as: If set, stores results as this property on nodes.
            keep_selection: Preserve selection after store. Default ``False``.
            aggregate_connections: Aggregate over connected nodes.

        Returns:
            Computation results, or a KnowledgeGraph if ``store_as`` is set.
        """
        ...

    def count(
        self,
        level_index: Optional[int] = None,
        group_by_parent: Optional[bool] = None,
        store_as: Optional[str] = None,
        keep_selection: Optional[bool] = None,
        group_by: Optional[str] = None,
    ) -> Any:
        """Count nodes, optionally grouped by parent or by a property.

        Args:
            level_index: Target level in the hierarchy.
            group_by_parent: Group counts by parent node.
            store_as: Store count as a property on parent nodes.
            keep_selection: Preserve selection after store. Default ``False``.
            group_by: Group counts by this property instead of by parent.
                Returns ``{group_value: count}``.

        Returns:
            An integer count, grouped counts, or a KnowledgeGraph if ``store_as`` is set.
        """
        ...

    # ====================================================================
    # Debugging & Introspection
    # ====================================================================

    def schema_text(self) -> str:
        """Return a text summary of the graph schema (node types, connections)."""
        ...

    def set_parent_type(self, node_type: str, parent_type: str) -> None:
        """Declare a node type as a supporting child of a parent type.

        Supporting types are hidden from the ``describe()`` inventory and
        instead appear in the ``<supporting>`` section when the parent type
        is inspected.  Their capabilities (timeseries, spatial, etc.) bubble
        up to the parent descriptor.

        Args:
            node_type: The supporting (child) node type.
            parent_type: The core (parent) node type.

        Raises:
            ValueError: If either type does not exist in the graph.

        Example::

            graph.set_parent_type('ProductionProfile', 'Field')
            graph.set_parent_type('FieldReserves', 'Field')
        """
        ...

    def describe(
        self,
        types: list[str] | None = None,
        type_search: str | None = None,
        connections: bool | list[str] | None = None,
        cypher: bool | list[str] | None = None,
        fluent: bool | list[str] | None = None,
        max_pairs: int | None = None,
        sample_truncate: int | None = 40,
    ) -> str:
        """Return an XML description of this graph for AI agents.

        Five independent axes for progressive disclosure:

        **Node types** (``types`` parameter):

        - ``describe()`` — Inventory overview with compact type descriptors
          and connections with edge property names. Adapts to graph scale:
          small graphs get full inline detail, extreme-scale graphs get
          a statistical summary with search hints.
        - ``describe(types=['Field', 'Well'])`` — Focused detail for
          specific types with properties, connections, and samples.

        **Type search** (``type_search`` parameter):

        - ``describe(type_search='software')`` — Find types by name
          (case-insensitive substring match) with neighborhood fan-out.
          Returns matching types with their connections, plus one layer
          of connected types. Ideal for exploring large/extreme-scale
          graphs where listing all types is impractical.

        **Connections** (``connections`` parameter):

        - ``describe(connections=True)`` — All connection types with count,
          source/target node types, and property names.
        - ``describe(connections=['BELONGS_TO'])`` — Deep-dive with per-pair
          counts, property stats with sample values, and sample edges.
          Use this to discover what data edges carry.

        **Cypher** (``cypher`` parameter):

        - ``describe(cypher=True)`` — Compact Cypher reference: all clauses,
          operators, functions, and procedures with 1-line descriptions.
        - ``describe(cypher=['cluster', 'MATCH'])`` — Detailed docs with
          parameters and examples for specific topics.

        **Fluent API** (``fluent`` parameter):

        - ``describe(fluent=True)`` — Compact fluent API reference: all
          methods grouped by area with signatures and descriptions.
        - ``describe(fluent=['traverse', 'where', 'spatial'])`` — Detailed
          docs with parameters and examples for specific topics.

        When ``type_search``, ``connections``, ``cypher``, or ``fluent``
        is set, only those tracks are returned (no node inventory).

        Args:
            types: Node type names for focused detail.
            type_search: Case-insensitive substring pattern to search type
                names. Returns matching types with connections + 1 layer
                of connected types for neighborhood discovery.
            connections: True for overview, list for deep-dive into specific types.
            cypher: True for compact reference, list for detailed topic docs.
            fluent: True for compact reference, list for detailed topic docs.
            max_pairs: Cap on ``(src_type, tgt_type)`` pairs rendered in the
                ``describe(connections=['T'])`` deep-dive. Defaults to 50.
                Sorted by count desc, so the head covers the dominant
                relationships; a trailing ``<more pairs="…" edges="…"/>``
                marker reports the hidden tail. Raise to drill into
                wide fan-out connection types (e.g. Wikidata ``P31`` has
                191k distinct pairs).
            sample_truncate: Max chars for string values in ``vals=``
                attributes, sample-node ids/titles, and sample-edge
                attributes. Defaults to 40. Pass ``None`` to disable
                truncation entirely — useful when you want full titles
                in an LLM prompt and have context-window budget for it.
                Has no effect on stored data; only the rendered XML.

        Raises:
            ValueError: If any type/connection/topic is not found.
            TypeError: If connections, cypher, or fluent has wrong type.
        """
        ...

    def explore(
        self,
        query: str,
        max_entities: int = 10,
        max_depth: int = 2,
        include_source: bool = True,
        source_roots: list[str] | None = None,
    ) -> str:
        """One-call codebase exploration over a code-tree graph.

        Lexically ranks ``Function`` / ``Class`` / ``Interface`` /
        ``Struct`` / ``Trait`` / ``Protocol`` / ``Enum`` nodes against
        ``query`` (matched against name + signature + docstring), takes
        the top ``max_entities``, 2-hop traverses CALLS / USES_TYPE /
        HAS_METHOD / DEFINES / REFERENCES_FN, and returns a markdown
        report with entry points, a relationship map, and grouped source
        slices for the entry points.

        Designed for the "how does X work in this codebase" question
        that would otherwise require a chain of grep + read calls.
        Composes FTS + traversal + source-slicing into a single Rust-
        side call.

        Args:
            query: Free-text topic. Matched against name + signature +
                docstring. Exact name matches rank highest.
            max_entities: Top N entry points after ranking (default 10).
            max_depth: Hops for the neighborhood traversal (default 2).
            include_source: Whether to include source slices for entry
                points (default ``True``). Set False for a smaller,
                faster response when the entity list is all you need.
            source_roots: Filesystem roots to resolve ``file_path``
                properties against. Files matched literally are tried
                first; roots are searched in order. Default: cwd only.

        Returns:
            A markdown string with ``## Query`` / ``## Entry points`` /
            ``## Related`` / ``## Source`` sections. Empty queries and
            graphs with no matching entities return a clear no-match
            message rather than raising.
        """
        ...

    def bug_report(
        self,
        query: str,
        result: str,
        expected: str,
        description: str,
        path: str | None = None,
    ) -> str:
        """File a Cypher bug report to ``reported_bugs.md``.

        Appends a timestamped, version-tagged report to the top of the file
        (creating it if needed).  All inputs are sanitised against code
        injection (HTML tags, ``javascript:`` URIs, triple-backtick breakout).

        Args:
            query: The Cypher query that triggered the bug.
            result: The actual result you got.
            expected: The result you expected.
            description: Free-text explanation.
            path: Optional file path (default: ``reported_bugs.md`` in cwd).

        Returns:
            Confirmation message with the file path.

        Raises:
            IOError: If the file cannot be written.
        """
        ...

    @staticmethod
    def explain_mcp() -> str:
        """Return a self-contained XML quickstart for setting up a KGLite MCP server.

        Covers the bundled ``kglite-mcp-server`` console script (the default
        on-ramp — no fork required), the YAML manifest for adding custom
        tools (``source_root:`` for file access, inline Cypher tools, and
        trust-gated Python hooks), and Claude Desktop / Claude Code
        registration config.

        Example::

            print(KnowledgeGraph.explain_mcp())
        """
        ...

    def selection(self) -> str:
        """Return a text summary of the current selection state."""
        ...

    def clear(self) -> None:
        """Clear the current selection (resets to empty)."""
        ...

    # ====================================================================
    # Copy / Clone
    # ====================================================================

    def copy(self) -> "KnowledgeGraph":
        """Create an independent deep copy of this graph.

        Returns a new ``KnowledgeGraph`` that shares no mutable state with
        the original.  Useful for running mutations without affecting the
        source graph.
        """
        ...

    def __copy__(self) -> "KnowledgeGraph": ...
    def __deepcopy__(self, memo: Any) -> "KnowledgeGraph": ...

    # ====================================================================
    # Schema Introspection
    # ====================================================================

    def rebuild_caches(self) -> None:
        """Force recomputation of internal caches (edge type counts, etc.).

        Call once after bulk mutations to warm the cache before
        ``save()`` or ``describe()``. The cache is persisted by
        ``save()`` and restored by ``load()``, so this only needs
        to be called once after building or mutating a graph.

        Example::

            g = KnowledgeGraph()
            g.add_nodes(...)
            g.add_connections(...)
            g.rebuild_caches()   # one-time O(E) pass
            g.save("graph.kgl")  # persists warm cache
        """
        ...

    def schema(self) -> dict[str, Any]:
        """Return a full schema overview of the graph.

        Returns:
            Dict with keys:
                - ``node_types``: ``{type_name: {count, properties: {name: type_str}}}``
                - ``connection_types``: ``{conn_name: {count, source_types: list, target_types: list}}``
                - ``indexes``: list of ``"Type.property"`` strings
                - ``node_count``: total nodes
                - ``edge_count``: total edges

        Note:
            Scans all edges once (O(m)) to compute accurate connection type stats.
        """
        ...

    def connection_types(self) -> list[dict[str, Any]]:
        """Return all connection types with counts and endpoint type sets.

        Returns:
            List of dicts with ``type``, ``count``, ``source_types``, ``target_types``.
        """
        ...

    def properties(self, node_type: str, max_values: int = 20) -> dict[str, dict[str, Any]]:
        """Return property statistics for a node type.

        Only properties that exist on at least one node are included.

        Args:
            node_type: The node type to inspect.
            max_values: Include ``values`` list when unique count <= this
                threshold. Set to 0 to never include values. Default: 20.

        Returns:
            Dict mapping property name to stats dict with keys:
                - ``type``: type string (e.g. ``'str'``, ``'int'``, ``'float'``)
                - ``non_null``: count of non-null values
                - ``unique``: count of distinct values
                - ``values``: (optional) sorted list of values when unique count <= max_values

        Raises:
            KeyError: If node_type does not exist.
        """
        ...

    def neighbors_schema(self, node_type: str) -> dict[str, list[dict[str, Any]]]:
        """Return connection topology for a node type.

        Args:
            node_type: The node type to inspect.

        Returns:
            Dict with:
                - ``outgoing``: list of ``{connection_type, target_type, count}``
                - ``incoming``: list of ``{connection_type, source_type, count}``

        Raises:
            KeyError: If node_type does not exist.
        """
        ...

    @overload
    def sample(self, node_type: str, n: int = 5) -> ResultView:
        """Return a quick sample of nodes.

        Can be called as:

        - ``sample("Person")`` — sample 5 nodes of the given type
        - ``sample("Person", 10)`` — sample 10 of that type
        - ``sample(3)`` — sample 3 nodes from the current selection
        - ``sample()`` — sample 5 nodes from the current selection

        Args:
            node_type_or_n: A node type (str) or sample count (int).
            n: Sample count when first arg is a node type. Default ``5``.

        Returns:
            :class:`ResultView` with sampled node rows.

        Raises:
            KeyError: If the given node type does not exist.
            ValueError: If no selection and no node type given.
        """
        ...
    @overload
    def sample(self, n: int = 5) -> ResultView: ...
    def indexes(self) -> list[dict[str, Any]]:
        """Return a unified list of all indexes.

        Returns:
            List of dicts, each with:
                - ``node_type``: the indexed node type
                - ``property``: property name (for equality indexes)
                - ``properties``: list of property names (for composite indexes)
                - ``type``: ``'equality'`` or ``'composite'``
        """
        ...

    # ====================================================================
    # Columnar Storage
    # ====================================================================

    def enable_disk_mode(self) -> None:
        """Convert the graph to disk-backed storage mode.

        Enables columnar storage first (if not already), then builds
        CSR (Compressed Sparse Row) edge arrays on disk. Nodes stay
        in memory (~40 bytes each), edges are mmap'd from disk.

        This reduces memory usage to ~10% of the in-memory graph for
        edge-heavy graphs. Best called after all data is loaded.

        All query methods (Cypher, fluent API, algorithms) work
        identically in disk mode.

        Example::

            graph.enable_disk_mode()
        """
        ...

    def enable_columnar(self) -> None:
        """Convert node properties to columnar storage.

        Properties are moved from per-node storage into per-type column
        stores, reducing memory usage for homogeneous typed columns
        (int64, float64, string, etc.). Automatically compacts properties
        first if not already compacted.

        Example::

            graph.enable_columnar()
            assert graph.is_columnar
        """
        ...

    def disable_columnar(self) -> None:
        """Convert columnar properties back to compact per-node storage.

        This is the inverse of :meth:`enable_columnar`. Useful before
        saving to ``.kgl`` format or when columnar storage is no longer
        needed.
        """
        ...

    def unspill(self) -> None:
        """Move mmap-backed columnar data back to heap memory.

        Useful after deleting nodes when you want data back in RAM for
        faster access. Internally rebuilds columnar stores from scratch
        with the memory limit temporarily suspended to prevent re-spilling.

        No-op if the graph is not in columnar mode.

        Example::

            graph.unspill()
            info = graph.graph_info()
            assert not info['columnar_is_mapped']
        """
        ...

    def set_memory_limit(self, limit_bytes: int | None, spill_dir: str | None = None) -> None:
        """Configure automatic memory-pressure spill for columnar storage.

        When a memory limit is set, :meth:`enable_columnar` will
        automatically spill the largest column stores to temporary files
        on disk when total heap usage exceeds the limit.

        Args:
            limit_bytes: Maximum heap bytes for columnar data, or ``None``
                to disable the limit.
            spill_dir: Directory for spill files. Defaults to system temp dir.

        Example::

            graph.set_memory_limit(500_000_000)  # 500 MB limit
            graph.enable_columnar()  # auto-spills if over limit
            graph.set_memory_limit(None)  # disable limit
        """
        ...

    # ====================================================================
    # Import
    # ====================================================================

    def load_ntriples(
        self,
        path: str,
        *,
        predicates: list[str] | None = None,
        languages: list[str] | None = None,
        node_types: dict[str, str] | None = None,
        predicate_labels: dict[str, str] | None = None,
        max_entities: int | None = None,
        max_triples: int | None = None,
        verbose: bool = False,
        progress: Callable[[dict], None] | None = None,
    ) -> dict:
        """Load an N-Triples file into the graph.

        Streams the file (supports ``.bz2``, ``.gz``, or plain ``.nt``)
        and converts RDF triples into a property graph. Designed for
        Wikidata truthy dumps but works with any N-Triples file.

        **RDF → property graph mapping:**

        - Each unique ``Q``-entity subject becomes a **node**.
        - ``rdfs:label`` (language-filtered) → **node title**.
        - ``schema:description`` → ``description`` property.
        - ``prop/direct/P*`` with a ``Q``-entity object → **edge**.
        - ``prop/direct/P*`` with a literal object → **node property**.
        - ``P31`` (instance of) determines the **node type** via *node_types*.

        Args:
            path: Path to the N-Triples file (``.nt``, ``.nt.bz2``, ``.nt.gz``).
            predicates: Wikidata P-codes to import (e.g. ``["P31", "P279"]``).
                ``None`` imports all predicates. Labels and descriptions are
                always imported regardless of this filter.
            languages: Language codes for label/description literals
                (e.g. ``["en"]``). ``None`` keeps all languages.
            node_types: Maps a P31 target Q-code to a human-readable node type
                name (e.g. ``{"Q5": "Person", "Q6256": "Country"}``).
                Entities whose P31 value is not in this map use the raw Q-code
                as their type. Entities without P31 get type ``"Entity"``.
            predicate_labels: Maps P-codes to human-readable edge/property
                names (e.g. ``{"P31": "instance_of", "P17": "country"}``).
                Unmapped predicates use the raw P-code.
            max_entities: Stop after creating this many nodes. Useful for
                exploratory loading of large dumps.
            max_triples: Stop after scanning this many triples (entity and
                non-entity lines alike). Applied alongside ``max_entities``;
                whichever fires first wins. Useful for benchmarking the
                bz2/gz/plain decompression + parser pipeline at a fixed
                wall-time-friendly slice.
            verbose: Print progress to stderr every 5M triples.
            progress: Optional callable receiving structured per-phase
                events. The callback is invoked with a single dict whose
                ``"kind"`` is ``"start"``, ``"update"``, or ``"complete"``
                and whose ``"phase"`` is one of ``"phase1"``, ``"phase1b"``,
                ``"phase2"``, ``"phase3"``, ``"finalising"``. See
                ``kglite.progress.TqdmBuildProgress`` for a tqdm-backed
                reporter. Errors raised by the callback are swallowed.

        Returns:
            A dict with load statistics::

                {"entities": int, "edges": int, "edges_skipped": int,
                 "triples_scanned": int, "seconds": float}

        Example::

            graph = KnowledgeGraph()
            stats = graph.load_ntriples(
                "latest-truthy.nt.bz2",
                predicates=["P31", "P279", "P17", "P106"],
                languages=["en"],
                node_types={"Q5": "Person", "Q6256": "Country"},
                predicate_labels={"P31": "instance_of", "P17": "country"},
                max_entities=1_000_000,
                verbose=True,
            )
        """
        ...

    # ====================================================================
    # Persistence
    # ====================================================================

    def save(self, path: str) -> None:
        """Serialise the graph to disk.

        For default/mapped modes: saves to a ``.kgl`` v3 binary file.
        For disk mode: saves to a directory containing CSR files and
        compressed node/edge data. The directory IS the saved graph.

        Uses columnar storage internally for efficient compression and
        larger-than-RAM loading. If the graph is not already in columnar
        mode, ``save()`` enables it automatically (the graph stays columnar
        after the call).

        Load it back with :func:`kglite.load` (accepts both files and
        directories).

        Args:
            path: Output file path (typically ``*.kgl``).
        """
        ...

    def compact(self) -> int:
        """Compact a disk-mode graph: merge overflow edges back into CSR arrays.

        Overflow edges accumulate when edges are added after the initial CSR
        build (e.g., after loading a graph and adding new connections).
        Compaction rebuilds the CSR to include all overflow edges, restoring
        optimal query performance.

        Returns:
            Number of overflow edges that were merged. Returns 0 if there
            are no overflow edges or the graph is not in disk mode.
        """
        ...

    # ====================================================================
    # Operation Reports
    # ====================================================================

    def set_default_timeout(self, timeout_ms: Optional[int] = None) -> None:
        """Set a default query timeout (milliseconds) for all cypher() calls.

        - ``None`` (default): fall through to the built-in default
          of 180_000 ms (3 min) for every storage mode.
        - ``0``: disable the deadline for every query unless per-call
          ``timeout_ms`` overrides.
        - Positive integer: use as the default.

        Per-query ``timeout_ms`` always overrides this setting.
        """
        ...

    def get_default_timeout(self) -> Optional[int]:
        """Get the current default query timeout in milliseconds, or None."""
        ...

    def set_default_max_rows(self, max_rows: Optional[int] = None) -> None:
        """Set a default max rows limit for all cypher() calls.

        Queries producing more intermediate rows than this will raise an error.
        Pass None to disable (default). Per-query ``max_rows`` overrides this.
        """
        ...

    def get_default_max_rows(self) -> Optional[int]:
        """Get the current default max rows limit, or None."""
        ...

    def last_report(self) -> dict[str, Any]:
        """Get the most recent operation report as a dict.

        Returns an empty dict if no operations have been performed.
        """
        ...

    def operation_index(self) -> int:
        """Get the sequential index of the last operation."""
        ...

    def report_history(self) -> list[dict[str, Any]]:
        """Get all operation reports as a list of dicts."""
        ...

    # ====================================================================
    # Set Operations
    # ====================================================================

    def union(self, other: KnowledgeGraph) -> KnowledgeGraph:
        """Combine selections from both graphs (set union).

        Returns:
            A new KnowledgeGraph with nodes from either selection.
        """
        ...

    def intersection(self, other: KnowledgeGraph) -> KnowledgeGraph:
        """Keep only nodes present in both selections (set intersection).

        Returns:
            A new KnowledgeGraph with only shared nodes.
        """
        ...

    def difference(self, other: KnowledgeGraph) -> KnowledgeGraph:
        """Keep nodes in ``self`` but not in ``other`` (set difference).

        Returns:
            A new KnowledgeGraph with the difference.
        """
        ...

    def symmetric_difference(self, other: KnowledgeGraph) -> KnowledgeGraph:
        """Keep nodes in exactly one of the selections (symmetric difference).

        Returns:
            A new KnowledgeGraph with nodes exclusive to each side.
        """
        ...

    # ====================================================================
    # Schema Definition & Validation
    # ====================================================================

    def define_schema(self, schema_dict: dict[str, Any]) -> KnowledgeGraph:
        """Define the expected schema for the graph.

        Args:
            schema_dict: Schema definition with ``nodes`` and ``connections`` keys.
                See the Rust docstring for full structure.

        Returns:
            Self with schema defined.
        """
        ...

    def validate_schema(self, strict: Optional[bool] = None) -> list[dict[str, Any]]:
        """Validate the graph against the defined schema.

        Args:
            strict: Report undefined types in the graph. Default ``False``.

        Returns:
            List of validation error dicts. Empty list means valid.
        """
        ...

    def has_schema(self) -> bool:
        """Check if a schema has been defined."""
        ...

    def clear_schema(self) -> KnowledgeGraph:
        """Remove the schema definition from the graph."""
        ...

    def schema_definition(self) -> Optional[dict[str, Any]]:
        """Get the current schema definition as a dict, or ``None``."""
        ...

    # ====================================================================
    # Graph Algorithms — Path Finding & Connectivity
    # ====================================================================

    def shortest_path(
        self,
        source_type: str,
        source_id: Any,
        target_type: str,
        target_id: Any,
        connection_types: Optional[list[str]] = None,
        via_types: Optional[list[str]] = None,
        weight_property: Optional[str] = None,
        timeout_ms: Optional[int] = None,
    ) -> Optional[dict[str, Any]]:
        """Find the shortest path between two nodes.

        Args:
            source_type: Source node type.
            source_id: Source node ID.
            target_type: Target node type.
            target_id: Target node ID.
            connection_types: Only traverse edges of these types. Default all.
            via_types: Only traverse through nodes of these types. Default all.
            weight_property: Edge property to use as cost. When set, the
                search uses Dijkstra and minimises total weight; when
                ``None``, BFS minimises hop count. Edges missing the
                property fall back to weight 1.0 (matches Louvain's
                weighted-adjacency convention). Negative weights cause
                the path to be reported as missing.
            timeout_ms: Abort after this many milliseconds and return ``None``.

        Returns:
            Dict with ``path`` (list of node info dicts), ``connections``
            (list of edge types), and ``length`` (hop count). When
            ``weight_property`` is set, also includes ``weight`` (sum
            of edge weights). ``None`` if no path exists or timeout is
            reached.
        """
        ...

    def shortest_path_length(
        self,
        source_type: str,
        source_id: Any,
        target_type: str,
        target_id: Any,
        weight_property: Optional[str] = None,
    ) -> Optional[Union[int, float]]:
        """Get just the cost of the shortest path.

        Faster than :meth:`shortest_path` when you only need the distance.
        Does not support ``connection_types`` or ``via_types`` filtering.

        Args:
            weight_property: When set, uses Dijkstra and returns total
                weight (float). When ``None``, BFS returns hop count (int).

        Returns:
            Hop count (int) or total weight (float), or ``None`` if no
            path exists.
        """
        ...

    def shortest_path_ids(
        self,
        source_type: str,
        source_id: Any,
        target_type: str,
        target_id: Any,
        connection_types: Optional[list[str]] = None,
        via_types: Optional[list[str]] = None,
        timeout_ms: Optional[int] = None,
    ) -> Optional[list[Any]]:
        """Get node IDs along the shortest path.

        Args:
            connection_types: Only traverse edges of these types. Default all.
            via_types: Only traverse through nodes of these types. Default all.
            timeout_ms: Abort after this many milliseconds and return ``None``.

        Returns:
            List of node IDs, or ``None`` if no path exists or timeout is reached.
        """
        ...

    def shortest_path_indices(
        self,
        source_type: str,
        source_id: Any,
        target_type: str,
        target_id: Any,
        connection_types: Optional[list[str]] = None,
        via_types: Optional[list[str]] = None,
        timeout_ms: Optional[int] = None,
    ) -> Optional[list[int]]:
        """Get raw graph indices along the shortest path.

        Fastest path query — no node data lookup.

        Args:
            connection_types: Only traverse edges of these types. Default all.
            via_types: Only traverse through nodes of these types. Default all.
            timeout_ms: Abort after this many milliseconds and return ``None``.

        Returns:
            List of integer indices, or ``None`` if no path exists or timeout is reached.
        """
        ...

    def all_paths(
        self,
        source_type: str,
        source_id: Any,
        target_type: str,
        target_id: Any,
        max_hops: Optional[int] = None,
        max_results: Optional[int] = None,
        connection_types: Optional[list[str]] = None,
        via_types: Optional[list[str]] = None,
        timeout_ms: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        """Find all paths between two nodes.

        Args:
            source_type: Source node type.
            source_id: Source node ID.
            target_type: Target node type.
            target_id: Target node ID.
            max_hops: Maximum path length. Default ``5``.
            max_results: Stop after finding this many paths. Default unlimited.
                Use to prevent OOM on dense graphs.
            connection_types: Only traverse edges of these types. Default all.
            via_types: Only traverse through nodes of these types. Default all.
            timeout_ms: Abort after this many milliseconds, returning partial results.

        Returns:
            List of path dicts, each with ``path``, ``connections``, ``length``.
        """
        ...

    def connected_components(
        self, weak: Optional[bool] = None, titles_only: Optional[bool] = None
    ) -> list[list[dict[str, Any]]]:
        """Find connected components in the graph.

        Args:
            weak: If ``True`` (default), find weakly connected components.
                If ``False``, find strongly connected components.
            titles_only: If ``True``, return lists of node titles instead of full dicts.

        Returns:
            List of components (largest first), each a list of node info dicts.
        """
        ...

    def are_connected(
        self,
        source_type: str,
        source_id: Any,
        target_type: str,
        target_id: Any,
    ) -> bool:
        """Check if two nodes are connected (directly or indirectly)."""
        ...

    def degrees(self) -> dict[str, int]:
        """Get connection count for each selected node.

        Returns:
            ``{node_title: degree}``.
        """
        ...

    # ====================================================================
    # Centrality Algorithms
    # ====================================================================

    def betweenness_centrality(
        self,
        normalized: Optional[bool] = None,
        sample_size: Optional[int] = None,
        connection_types: Optional[Union[str, list[str]]] = None,
        top_k: Optional[int] = None,
        as_dict: Optional[bool] = None,
        timeout_ms: Optional[int] = None,
        to_df: Optional[bool] = None,
    ) -> Union[ResultView, dict[Any, float], pd.DataFrame]:
        """Calculate betweenness centrality.

        Args:
            normalized: Normalise scores to ``[0, 1]``. Default ``True``.
            sample_size: Sample source nodes for faster computation on large graphs.
            connection_types: Only traverse these relationship types (str or list).
            top_k: Return only the top *K* nodes.
            as_dict: Return ``{id: score}`` dict instead of list of dicts.
            timeout_ms: Abort after this many milliseconds, returning partial results.
            to_df: Return a pandas DataFrame with columns ``type``, ``title``, ``id``, ``score``.

        Returns:
            List of dicts with ``type``, ``title``, ``id``, ``score``,
            sorted by score descending. Or a DataFrame if ``to_df=True``.
        """
        ...

    def pagerank(
        self,
        damping_factor: Optional[float] = None,
        max_iterations: Optional[int] = None,
        tolerance: Optional[float] = None,
        connection_types: Optional[Union[str, list[str]]] = None,
        top_k: Optional[int] = None,
        as_dict: Optional[bool] = None,
        timeout_ms: Optional[int] = None,
        to_df: Optional[bool] = None,
    ) -> Union[ResultView, dict[Any, float], pd.DataFrame]:
        """Calculate PageRank centrality.

        Args:
            damping_factor: Probability of following a link. Default ``0.85``.
            max_iterations: Maximum iterations. Default ``100``.
            tolerance: Convergence threshold. Default ``1e-6``.
            connection_types: Only traverse these relationship types (str or list).
            top_k: Return only the top *K* nodes.
            as_dict: Return ``{id: score}`` dict instead of list of dicts.
            timeout_ms: Abort after this many milliseconds, returning partial results.
            to_df: Return a pandas DataFrame with columns ``type``, ``title``, ``id``, ``score``.

        Returns:
            List of dicts with ``type``, ``title``, ``id``, ``score``.
            Or a DataFrame if ``to_df=True``.
        """
        ...

    def degree_centrality(
        self,
        normalized: Optional[bool] = None,
        connection_types: Optional[Union[str, list[str]]] = None,
        top_k: Optional[int] = None,
        as_dict: Optional[bool] = None,
        timeout_ms: Optional[int] = None,
        to_df: Optional[bool] = None,
    ) -> Union[ResultView, dict[Any, float], pd.DataFrame]:
        """Calculate degree centrality.

        Args:
            normalized: Normalise by ``(n-1)``. Default ``True``.
            connection_types: Only count these relationship types (str or list).
            top_k: Return only the top *K* nodes.
            as_dict: Return ``{id: score}`` dict instead of list of dicts.
            timeout_ms: Abort after this many milliseconds, returning partial results.
            to_df: Return a pandas DataFrame with columns ``type``, ``title``, ``id``, ``score``.

        Returns:
            List of dicts with ``type``, ``title``, ``id``, ``score``.
            Or a DataFrame if ``to_df=True``.
        """
        ...

    def closeness_centrality(
        self,
        normalized: Optional[bool] = None,
        sample_size: Optional[int] = None,
        connection_types: Optional[Union[str, list[str]]] = None,
        top_k: Optional[int] = None,
        as_dict: Optional[bool] = None,
        timeout_ms: Optional[int] = None,
        to_df: Optional[bool] = None,
    ) -> Union[ResultView, dict[Any, float], pd.DataFrame]:
        """Calculate closeness centrality.

        Args:
            normalized: Adjust for disconnected components. Default ``True``.
            sample_size: Approximate by sampling *N* source nodes (faster for large graphs).
                If ``None``, uses all nodes.
            connection_types: Filter to specific relationship types.
            top_k: Return only the top *K* nodes.
            as_dict: Return ``{id: score}`` dict instead of list of dicts.
            timeout_ms: Abort after this many milliseconds, returning partial results.
            to_df: Return a pandas DataFrame with columns ``type``, ``title``, ``id``, ``score``.

        Returns:
            List of dicts with ``type``, ``title``, ``id``, ``score``.
        """
        ...

    # ====================================================================
    # Community Detection
    # ====================================================================

    def louvain_communities(
        self,
        weight_property: Optional[str] = None,
        resolution: Optional[float] = None,
        connection_types: Optional[list[str]] = None,
        timeout_ms: Optional[int] = None,
    ) -> dict[str, Any]:
        """Detect communities using the Louvain algorithm.

        Args:
            weight_property: Edge property to use as weight. Default all edges weight ``1.0``.
            resolution: Resolution parameter (higher = more communities). Default ``1.0``.
            connection_types: Only consider edges of these types. Default all edge types.
            timeout_ms: Abort after this many milliseconds, returning partial results.

        Returns:
            Dict with ``communities`` (dict of community_id to member list),
            ``modularity``, and ``num_communities``.
        """
        ...

    def label_propagation(
        self,
        max_iterations: Optional[int] = None,
        connection_types: Optional[list[str]] = None,
        timeout_ms: Optional[int] = None,
    ) -> dict[str, Any]:
        """Detect communities using label propagation.

        Args:
            max_iterations: Maximum iterations. Default ``100``.
            connection_types: Only consider edges of these types. Default all edge types.
            timeout_ms: Abort after this many milliseconds, returning partial results.

        Returns:
            Dict with ``communities``, ``modularity``, and ``num_communities``.
        """
        ...

    # ====================================================================
    # Subgraph Extraction
    # ====================================================================

    def expand(self, hops: Optional[int] = None) -> KnowledgeGraph:
        """Expand the selection by *N* hops (breadth-first, undirected).

        Args:
            hops: Number of hops to expand. Default ``1``.

        Returns:
            A new KnowledgeGraph with the expanded selection.
        """
        ...

    def to_subgraph(self) -> KnowledgeGraph:
        """Extract selected nodes into a new independent graph.

        The new graph contains only selected nodes and the edges between them.
        """
        ...

    def subgraph_stats(self) -> dict[str, Any]:
        """Get statistics about the subgraph that would be extracted.

        Returns:
            Dict with ``node_count``, ``edge_count``, ``node_types``, ``connection_types``.
        """
        ...

    def save_subset(self, path: str) -> None:
        """Save the current selection as an independent subgraph file.

        Equivalent to ``kg.to_subgraph().save(path)`` in a single call.
        Output is a v3 binary file that reloads via ``kglite.load(path)``
        (or ``load(path, storage='disk')`` for disk mode). All edges
        between selected nodes are included; node and edge properties
        round-trip byte-for-byte.

        Args:
            path: Destination path for the subgraph file.

        Example:
            >>> kg.select("Article").expand(hops=1, type="AUTHORED_BY").save_subset(
            ...     "articles_with_authors.kgl"
            ... )
        """
        ...

    # ====================================================================
    # Export
    # ====================================================================

    def export(
        self,
        path: str,
        format: Optional[str] = None,
        selection_only: Optional[bool] = None,
    ) -> None:
        """Export the graph to a file.

        Supported formats: ``graphml``, ``gexf``, ``d3``/``json``, ``csv``.
        Format is inferred from the file extension if not specified.

        Args:
            path: Output file path.
            format: Export format. Default: inferred from extension.
            selection_only: Export only selected nodes. Default: ``True`` if selection exists.
        """
        ...

    def export_csv(
        self,
        path: str,
        selection_only: Optional[bool] = None,
        verbose: bool = False,
    ) -> dict[str, Any]:
        """Export to an organized CSV directory tree with blueprint.

        Creates a directory structure with one CSV per node type and one
        per connection type, plus a ``blueprint.json`` for round-trip
        re-import via :func:`from_blueprint`.

        Output structure::

            path/
            ├── nodes/
            │   ├── Person.csv
            │   ├── Company.csv
            │   └── Person/          # sub-nodes nested under parent
            │       └── Skill.csv
            ├── connections/
            │   ├── WORKS_AT.csv
            │   └── KNOWS.csv
            └── blueprint.json

        Node CSVs have columns: ``id``, ``title``, then all properties.
        Connection CSVs: ``source_id``, ``source_type``, ``target_id``,
        ``target_type``, then edge properties.

        Only connections where **both** endpoints are in the selection
        are exported.

        Args:
            path: Output directory (created if it doesn't exist).
            selection_only: Export only selected nodes. Default:
                ``True`` if a selection exists, ``False`` otherwise.
            verbose: Print progress information during export.

        Returns:
            Summary dict with keys ``output_dir``, ``nodes`` (type → count),
            ``connections`` (type → count), ``files_written``.

        Example::

            graph.export_csv('output/')
            graph.select('Person').export_csv('output/', verbose=True)
        """
        ...

    def export_string(
        self,
        format: str,
        selection_only: Optional[bool] = None,
    ) -> str:
        """Export the graph to a string.

        Supported formats: ``graphml``, ``gexf``, ``d3``/``json``.

        Args:
            format: Export format.
            selection_only: Export only selected nodes.
        """
        ...

    # ====================================================================
    # Property Indexes
    # ====================================================================

    def create_index(self, node_type: str, property: str) -> dict[str, Any]:
        """Create an index on a property for O(1) equality filter lookups.

        Indexes are automatically maintained by Cypher mutations
        (CREATE, SET, REMOVE, DELETE, MERGE).

        Args:
            node_type: Node type to index.
            property: Property name to index.

        Returns:
            Dict with ``type``, ``property``, ``unique_values``, ``created``.
        """
        ...

    def drop_index(self, node_type: str, property: str) -> bool:
        """Remove an index. Returns ``True`` if it existed."""
        ...

    def create_global_index(self, property: str) -> dict[str, Any]:
        """Build a cross-type global index on `property`.

        Unlike ``create_index``, which is keyed by ``(node_type, property)``,
        this indexes EVERY node whose value at `property` is a non-empty
        string — regardless of node type. Enables:

        - ``MATCH (n {label: 'Norway'})`` — untyped Cypher lookups, routed
          through the global index in O(log N).
        - ``graph.search(text)`` — top-k helper that returns the nodes
          whose label (or any indexed property) matches.

        Disk-backed graphs only. On memory/mapped graphs this is a no-op
        that returns ``unique_values=0`` — per-type ``create_index`` already
        covers the use case at in-memory scale.

        Returns:
            Dict with ``property``, ``unique_values`` (count of nodes
            indexed), and ``created``.
        """
        ...

    def search(
        self,
        text: str,
        *,
        property: str = "title",
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Find nodes matching ``text`` on ``property`` (default 'title').

        Tries exact match first, then prefix match. Returns up to ``limit``
        results as dicts with ``id`` (node index), ``type``, ``title``, and
        ``id_value`` (the node's id-field value, e.g. a Wikidata Q-number).

        Requires ``create_global_index(property)`` to have been run on a
        disk-backed graph. Returns an empty list otherwise.

        Example::

            graph.create_global_index('label')
            hits = graph.search('Norway')
            # [{'id': 12345, 'type': 'country', 'title': 'Norway',
            #   'id_value': 'Q20'}]
        """
        ...

    def list_indexes(self) -> list[dict[str, str]]:
        """List all indexes. Each dict has ``type`` and ``property``."""
        ...

    def has_index(self, node_type: str, property: str) -> bool:
        """Check if an index exists."""
        ...

    def index_stats(self, node_type: str, property: str) -> Optional[dict[str, Any]]:
        """Get statistics for an index. Returns ``None`` if not found."""
        ...

    def rebuild_indexes(self) -> int:
        """Rebuild all indexes. Returns the number of indexes rebuilt."""
        ...

    # ====================================================================
    # Range Indexes (B-Tree)
    # ====================================================================

    def create_range_index(self, node_type: str, property: str) -> dict[str, Any]:
        """Create a range index (B-Tree) on a property for efficient range queries.

        Enables fast ``>``, ``>=``, ``<``, ``<=``, and ``BETWEEN`` queries
        in ``filter()`` calls.

        Args:
            node_type: Node type to index.
            property: Property name to index.

        Returns:
            Dict with ``type``, ``property``, ``unique_values``, ``created``.

        Example::

            graph.create_range_index('Person', 'age')
            old = graph.select('Person').where({'age': {'>': 60}}).collect()
        """
        ...

    def drop_range_index(self, node_type: str, property: str) -> bool:
        """Remove a range index. Returns ``True`` if it existed."""
        ...

    # ====================================================================
    # Composite Indexes
    # ====================================================================

    def create_composite_index(
        self,
        node_type: str,
        properties: list[str],
    ) -> dict[str, Any]:
        """Create a composite index on multiple properties.

        Args:
            node_type: Node type to index.
            properties: List of property names for the composite key.

        Returns:
            Dict with ``type``, ``properties``, ``unique_combinations``.
        """
        ...

    def drop_composite_index(self, node_type: str, properties: list[str]) -> bool:
        """Remove a composite index. Returns ``True`` if it existed."""
        ...

    def list_composite_indexes(self) -> list[dict[str, Any]]:
        """List all composite indexes."""
        ...

    def has_composite_index(self, node_type: str, properties: list[str]) -> bool:
        """Check if a composite index exists."""
        ...

    def composite_index_stats(
        self,
        node_type: str,
        properties: list[str],
    ) -> Optional[dict[str, Any]]:
        """Get statistics for a composite index. Returns ``None`` if not found."""
        ...

    # ====================================================================
    # Pattern Matching & Cypher
    # ====================================================================

    def match_pattern(
        self,
        pattern: str,
        max_matches: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        """Match a Cypher-like pattern against the graph.

        Supports node patterns ``(a:Type {prop: val})``, directed edges
        ``-[:TYPE]->``, ``<-[:TYPE]-``, and undirected ``-[:TYPE]-``.

        Args:
            pattern: Pattern string, e.g. ``'(a:Person)-[:KNOWS]->(b:Person)'``.
            max_matches: Maximum results to return.

        Returns:
            List of match dicts with variable bindings.
        """
        ...

    def cypher(
        self,
        query: str,
        *,
        to_df: bool = False,
        params: Optional[dict[str, Any]] = None,
        timeout_ms: Optional[int] = None,
        max_rows: Optional[int] = None,
        streaming: bool = True,
        disable_optimizer: bool = False,
        disabled_passes: Optional[list[str]] = None,
    ) -> Union[ResultView, pd.DataFrame, str]:
        """Execute a Cypher query.

        Supports MATCH, WHERE, RETURN, ORDER BY, LIMIT, SKIP, WITH,
        OPTIONAL MATCH, UNWIND, UNION, CREATE, SET, DELETE, DETACH DELETE,
        REMOVE, MERGE (with ON CREATE SET / ON MATCH SET), HAVING,
        CASE expressions, WHERE EXISTS, shortestPath(), list comprehensions,
        CALL...YIELD (graph algorithms: pagerank, betweenness, degree,
        closeness, louvain, label_propagation, connected_components),
        parameters ($param), ``!=`` operator, aggregation functions,
        window functions (``row_number()``, ``rank()``, ``dense_rank()``
        with ``OVER (PARTITION BY ... ORDER BY ...)``), and date arithmetic
        (``date + N``, ``date - date``, ``date_diff(d1, d2)``).

        Mutation queries (CREATE, SET, DELETE, REMOVE, MERGE) store
        statistics on ``graph.last_mutation_stats`` with keys
        ``nodes_created``, ``relationships_created``, ``properties_set``,
        ``nodes_deleted``, ``relationships_deleted``, ``properties_removed``.

        Each call is atomic: if any clause fails, the graph is unchanged.
        Property and composite indexes are automatically maintained.

        **FORMAT CSV**: Append ``FORMAT CSV`` to any query to get results as
        a CSV string instead of a ResultView. Good for large result transfers
        and token-efficient LLM consumption in MCP servers.

        Before execution, the query is validated against the graph schema
        (known node types, connection types, and properties). Unknown
        identifiers raise ``ValueError`` with a ``Did you mean '...'?``
        suggestion — catches typos before any scan runs.

        Args:
            query: Cypher query string.
            to_df: If ``True``, return a pandas DataFrame.
            params: Optional parameter dict for ``$param`` substitution.
            timeout_ms: Deadline in milliseconds. If omitted, uses
                ``set_default_timeout()`` when set, otherwise the
                built-in default of 180_000 ms (3 min). Pass ``0`` to
                disable the deadline for this call.
            max_rows: Cap on intermediate rows; defaults to
                ``set_default_max_rows()``.
            streaming: When ``True`` (default), the executor absorbs
                compatible clause runs (currently
                ``WITH/RETURN(group, agg) [ORDER BY ... LIMIT k]``) into
                a streaming pipeline that builds aggregate state inline
                and replaces full sort + truncate with a heap-pruned
                top-K (O(n log k) instead of O(n log n)). Pass ``False``
                to force the materialized executor — useful for
                debugging parity issues; behaviour should otherwise be
                identical.
            disable_optimizer: When ``True``, run the query with **all**
                optimizer passes skipped — schema validation still
                applies, but no predicate pushdown, fusion, reordering,
                or LIMIT-pushdown happens. Diagnostic / testing knob;
                normal use should leave this ``False``. Used by the
                differential test harness to assert optimized and naive
                executions produce identical results.
            disabled_passes: Skip a specific subset of optimizer passes
                by name. Names must come from
                :func:`kglite.cypher_pass_names()` — typos raise
                ``ValueError``. Useful for bisecting which pass
                introduces a divergence.

        Returns:
            ResultView by default, DataFrame when ``to_df=True``,
            or CSV string when the query ends with ``FORMAT CSV``.

        Example::

            rows = graph.cypher('''
                MATCH (p:Person)-[:KNOWS]->(f:Person)
                WHERE p.age > $min_age
                RETURN p.name, count(f) AS friends
                ORDER BY friends DESC LIMIT 10
            ''', params={'min_age': 25})
            for row in rows:
                print(row['name'], row['friends'])

            # As DataFrame
            df = graph.cypher('MATCH (n:Person) RETURN n.name, n.age', to_df=True)

            # As CSV string (good for large data transfers)
            csv = graph.cypher('MATCH (n:Person) RETURN n.name, n.age FORMAT CSV')

            # CREATE nodes and edges
            graph.cypher("CREATE (n:Person {name: 'Alice', age: 30})")
            print(graph.last_mutation_stats['nodes_created'])  # 1

            # SET properties
            graph.cypher('''
                MATCH (n:Person) WHERE n.name = 'Alice'
                SET n.city = 'Oslo', n.age = 31
            ''')

            # Semantic search with text_score (requires set_embedder + embed_texts)
            results = graph.cypher('''
                MATCH (n:Article)
                RETURN n.title,
                       text_score(n, 'summary', 'machine learning') AS score
                ORDER BY score DESC LIMIT 10
            ''', to_df=True)

            # text_score with parameter
            graph.cypher('''
                MATCH (n:Article)
                WHERE text_score(n, 'summary', $query) > 0.8
                RETURN n.title
            ''', params={'query': 'artificial intelligence'})

            # CALL graph algorithms
            top = graph.cypher('''
                CALL pagerank() YIELD node, score
                RETURN node.title, score
                ORDER BY score DESC LIMIT 10
            ''')

            # Community detection
            graph.cypher('''
                CALL louvain() YIELD node, community
                RETURN community, count(*) AS size
                ORDER BY size DESC
            ''')
        """
        ...

    # ====================================================================
    # Temporal
    # ====================================================================

    def set_temporal(
        self,
        type_name: str,
        valid_from: str,
        valid_to: str,
    ) -> None:
        """Configure temporal validity for a node type or connection type.

        After configuration, ``select()`` auto-filters temporal nodes and
        ``traverse()`` auto-filters temporal connections to "current" (today
        or the ``date()`` context).

        Auto-detects whether *type_name* is a node type or connection type.

        Args:
            type_name: Node type (e.g. ``'FieldStatus'``) or connection type
                (e.g. ``'HAS_LICENSEE'``).
            valid_from: Property name holding the start date.
            valid_to: Property name holding the end date.

        Raises:
            ValueError: If *type_name* is not a known node or connection type.
        """
        ...

    def date(
        self,
        date_str: Optional[str] = None,
        end_str: Optional[str] = None,
    ) -> KnowledgeGraph:
        """Set the temporal context for auto-filtering.

        Returns a new KnowledgeGraph. All subsequent ``select()`` and
        ``traverse()`` calls on the returned graph use this context for
        temporal filtering.

        Modes:
            - ``date('2013')`` — point-in-time (valid at 2013-01-01).
            - ``date('2010', '2015')`` — range: include everything valid at
              any point during 2010-01-01 to 2015-12-31 (overlap check).
            - ``date('all')`` — disable temporal filtering entirely.
            - ``date()`` — reset to today (default).

        Args:
            date_str: Date string, ``'all'``, or ``None`` to reset.
            end_str: Optional end date for range mode. End dates expand to
                period end (``'2015'`` → 2015-12-31, ``'2015-06'`` → 2015-06-30).

        Returns:
            A new KnowledgeGraph with the given temporal context.
        """
        ...

    # ====================================================================
    # Spatial / Geometry
    # ====================================================================

    def set_spatial(
        self,
        node_type: str,
        *,
        location: Optional[tuple[str, str]] = None,
        geometry: Optional[str] = None,
        points: Optional[dict[str, tuple[str, str]]] = None,
        shapes: Optional[dict[str, str]] = None,
    ) -> None:
        """Configure spatial properties for a node type.

        Args:
            node_type: The node type to configure.
            location: Primary lat/lon pair as ``(lat_field, lon_field)``. At most one per type.
            geometry: Primary WKT geometry field name. At most one per type.
            points: Named lat/lon points as ``{name: (lat_field, lon_field)}``.
            shapes: Named WKT shape fields as ``{name: field_name}``.
        """
        ...

    def spatial(
        self,
        node_type: Optional[str] = None,
    ) -> Optional[dict[str, Any]]:
        """Get spatial configuration for a node type or all types.

        Args:
            node_type: If given, return config for this type only. Otherwise return all.

        Returns:
            Dict with spatial config, or ``None`` if not configured.
        """
        ...

    def within_bounds(
        self,
        min_lat: float,
        max_lat: float,
        min_lon: float,
        max_lon: float,
        lat_field: Optional[str] = None,
        lon_field: Optional[str] = None,
    ) -> KnowledgeGraph:
        """Filter nodes within a geographic bounding box.

        Args:
            min_lat: South bound latitude.
            max_lat: North bound latitude.
            min_lon: West bound longitude.
            max_lon: East bound longitude.
            lat_field: Latitude property name. Default ``'latitude'``.
            lon_field: Longitude property name. Default ``'longitude'``.

        Returns:
            A new KnowledgeGraph with only nodes in the bounding box.
        """
        ...

    def near_point(
        self,
        center_lat: float,
        center_lon: float,
        max_distance: float,
        lat_field: Optional[str] = None,
        lon_field: Optional[str] = None,
    ) -> KnowledgeGraph:
        """Filter nodes within a distance (in degrees) of a point.

        Args:
            center_lat: Center latitude.
            center_lon: Center longitude.
            max_distance: Maximum distance in degrees.
            lat_field: Latitude property name. Default ``'latitude'``.
            lon_field: Longitude property name. Default ``'longitude'``.

        Returns:
            A new KnowledgeGraph with nearby nodes.
        """
        ...

    def near_point_m(
        self,
        center_lat: float,
        center_lon: float,
        max_distance_m: float,
        lat_field: Optional[str] = None,
        lon_field: Optional[str] = None,
    ) -> KnowledgeGraph:
        """Filter nodes within a distance (in meters) using geodesic calculation.

        Uses WGS84 ellipsoid for accurate Earth-surface distances.
        Falls back to geometry centroid when lat/lon fields are missing
        but a WKT geometry is configured via ``set_spatial``.

        Args:
            center_lat: Center latitude.
            center_lon: Center longitude.
            max_distance_m: Maximum distance in meters.
            lat_field: Latitude property name. Default from spatial config or ``'latitude'``.
            lon_field: Longitude property name. Default from spatial config or ``'longitude'``.

        Returns:
            A new KnowledgeGraph with nearby nodes.
        """
        ...

    def contains_point(
        self,
        lat: float,
        lon: float,
        geometry_field: Optional[str] = None,
    ) -> KnowledgeGraph:
        """Filter nodes whose WKT polygon contains a point.

        Args:
            lat: Query point latitude.
            lon: Query point longitude.
            geometry_field: WKT geometry property name. Default ``'geometry'``.

        Returns:
            A new KnowledgeGraph with containing nodes.
        """
        ...

    def intersects_geometry(
        self,
        query_wkt: Union[str, Any],
        geometry_field: Optional[str] = None,
    ) -> KnowledgeGraph:
        """Filter nodes whose geometry intersects a WKT geometry.

        Args:
            query_wkt: WKT string or shapely geometry object.
            geometry_field: Geometry property name. Default ``'geometry'``.

        Returns:
            A new KnowledgeGraph with intersecting nodes.
        """
        ...

    def bounds(
        self,
        lat_field: Optional[str] = None,
        lon_field: Optional[str] = None,
        as_shapely: bool = False,
    ) -> Optional[Union[dict[str, float], Any]]:
        """Get geographic bounds of selected nodes.

        Args:
            lat_field: Latitude property name. Default ``'latitude'``.
            lon_field: Longitude property name. Default ``'longitude'``.
            as_shapely: If ``True``, return a ``shapely.geometry.Polygon``
                (box) instead of a dict.

        Returns:
            Dict with ``min_lat``, ``max_lat``, ``min_lon``, ``max_lon``,
            or a shapely box polygon when ``as_shapely=True``,
            or ``None`` if no valid coordinates found.
        """
        ...

    def centroid(
        self,
        lat_field: Optional[str] = None,
        lon_field: Optional[str] = None,
        as_shapely: bool = False,
    ) -> Optional[Union[dict[str, float], Any]]:
        """Get the geographic centroid (average lat/lon) of selected nodes.

        Args:
            lat_field: Latitude property name. Default ``'latitude'``.
            lon_field: Longitude property name. Default ``'longitude'``.
            as_shapely: If ``True``, return a ``shapely.geometry.Point``
                instead of a dict.

        Returns:
            Dict with ``latitude`` and ``longitude``, or a shapely Point
            when ``as_shapely=True``, or ``None``.
        """
        ...

    def wkt_centroid(
        self,
        wkt_string: Union[str, Any],
        as_shapely: bool = False,
    ) -> Union[dict[str, float], Any]:
        """Calculate the centroid of a WKT geometry string.

        Args:
            wkt_string: WKT geometry string or shapely geometry object.
            as_shapely: If ``True``, return a ``shapely.geometry.Point``
                instead of a dict.

        Returns:
            Dict with ``latitude`` and ``longitude``, or a shapely Point
            when ``as_shapely=True``.
        """
        ...

    # ── Timeseries ─────────────────────────────────────────────────────────

    def set_timeseries(
        self,
        node_type: str,
        *,
        resolution: str,
        channels: Optional[list[str]] = None,
        units: Optional[dict[str, str]] = None,
        bin_type: Optional[str] = None,
    ) -> None:
        """Configure timeseries metadata for a node type.

        Args:
            node_type: The node type to configure.
            resolution: Time granularity — ``'year'``, ``'month'``, or ``'day'``.
                Determines key depth (year=1, month=2, day=3).
            channels: Optional list of known channel names.
            units: Optional map of channel name to unit string,
                e.g. ``{'oil': 'MSm3', 'temperature': '°C'}``.
            bin_type: What values represent — ``'total'``, ``'mean'``,
                or ``'sample'``. None if unspecified.
        """
        ...

    def timeseries_config(
        self,
        node_type: Optional[str] = None,
    ) -> Optional[dict[str, Any]]:
        """Get timeseries configuration for a node type or all types.

        Returns a dict with ``resolution``, ``channels``, ``units``, ``bin_type``.
        """
        ...

    def set_time_index(
        self,
        node_id: Any,
        keys: Union[list[str], list[list[int]]],
    ) -> None:
        """Set the sorted time index for a specific node.

        If the node already has a timeseries, this replaces its time index
        and clears all channels.

        Args:
            node_id: The node's unique ID.
            keys: Sorted list of date strings (e.g. ``['2020-01', '2020-02']``)
                or composite integer keys for backwards compat
                (e.g. ``[[2020, 1], [2020, 2]]``).
        """
        ...

    def add_ts_channel(
        self,
        node_id: Any,
        channel_name: str,
        values: list[float],
    ) -> None:
        """Add a timeseries channel to a node.

        The node must already have a time index set (via ``set_time_index``
        or ``add_timeseries``). The values length must match the time index
        length. Use ``float('nan')`` for missing values.

        Args:
            node_id: The node's unique ID.
            channel_name: Channel name (e.g. ``'oil'``, ``'temperature'``).
            values: Float values aligned with the time index.
        """
        ...

    def add_timeseries(
        self,
        node_type: str,
        *,
        data: Any,
        fk: str,
        time_key: list[str],
        channels: Union[dict[str, str], list[str]],
        resolution: Optional[str] = None,
        units: Optional[dict[str, str]] = None,
    ) -> dict[str, Any]:
        """Bulk-load timeseries data from a DataFrame.

        Groups rows by ``fk``, sorts by ``time_key``, and attaches the
        resulting timeseries to matching nodes (found by node ID).

        Time keys are combined into NaiveDate internally:
        - Single column: parsed as date strings (``'2020-06'``)
        - Multiple columns: combined as year + month [+ day] → NaiveDate

        Args:
            node_type: Target node type.
            data: Source DataFrame.
            fk: Foreign key column in ``data`` linking to node IDs.
            time_key: Column(s) for time keys. If single column, values
                are parsed as date strings. If multiple, combined as
                year + month [+ day].
            channels: Either a list of column names (used as channel names)
                or a dict mapping ``{channel_name: column_name}``.
            resolution: Time granularity (``'year'``, ``'month'``, ``'day'``).
                Auto-detected from time_key count if not specified.
            units: Optional channel→unit map, merged into config.

        Returns:
            Summary: ``{'nodes_loaded': N, 'total_records': M, 'total_rows': R}``.
        """
        ...

    def timeseries(
        self,
        node_id: Any,
        channel: Optional[str] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> Optional[dict[str, Any]]:
        """Extract timeseries data for a node.

        If ``channel`` is given, returns ``{'keys': [...], 'values': [...]}``.
        Otherwise returns ``{'keys': [...], 'channels': {'name': [...], ...}}``.

        Args:
            node_id: The node's unique ID.
            channel: Optional channel name to extract.
            start: Optional range start as date string (e.g. ``'2020'``, ``'2020-2'``).
            end: Optional range end as date string.

        Returns:
            Dict with keys and channel data, or ``None`` if no timeseries.
        """
        ...

    def time_index(
        self,
        node_id: Any,
    ) -> Optional[list[str]]:
        """Get the time index for a node as ISO date strings, or ``None``."""
        ...

    # ── Embedding / Vector Search ──────────────────────────────────────────

    def set_embeddings(
        self,
        node_type: str,
        text_column: str,
        embeddings: dict[Any, list[float]],
        metric: Optional[str] = None,
    ) -> dict[str, int]:
        """Store embeddings for nodes of the given type.

        Embeddings are stored separately from regular node properties and are
        invisible to ``collect()``, ``to_df()``, and other property-based APIs.
        The embedding store key is auto-derived as ``{text_column}_emb``.

        Validates that ``text_column`` exists as a property on the node type
        (builtins ``id``, ``title``, ``type`` are always accepted).

        Args:
            node_type: The node type (e.g. ``'Article'``).
            text_column: Source text column name (e.g. ``'summary'``).
            embeddings: Dict mapping node IDs to embedding vectors.
            metric: Default distance metric for this store. Used when no metric
                is specified at query time. ``'cosine'`` (default), ``'dot_product'``,
                ``'euclidean'``, or ``'poincare'``. Persisted with ``save()``.

        Returns:
            Dict with ``embeddings_stored``, ``dimension``, and ``skipped``.
        """
        ...

    def vector_search(
        self,
        text_column: str,
        query_vector: list[float],
        top_k: int = 10,
        metric: str = "cosine",
        to_df: bool = False,
    ) -> list[dict[str, Any]] | pd.DataFrame:
        """Vector similarity search within the current selection.

        Searches for nodes most similar to the query vector among the currently
        selected nodes. Results are ordered by similarity (most similar first).

        Args:
            text_column: Source text column name (e.g. ``'summary'``).
            query_vector: The query embedding vector.
            top_k: Number of results to return (default 10).
            metric: ``'cosine'``, ``'dot_product'``, ``'euclidean'``, or ``'poincare'``.
                If omitted, uses the metric stored via ``set_embeddings(metric=...)``,
                or falls back to ``'cosine'``.
            to_df: If ``True``, return a pandas DataFrame instead of list of dicts.

        Returns:
            List of dicts with ``id``, ``title``, ``type``, ``score``, and all
            node properties. Or a DataFrame if ``to_df=True``.

        Example::

            results = (graph
                .select('Article')
                .where({'category': 'politics'})
                .vector_search('summary', query_vec, top_k=10))
        """
        ...

    def list_embeddings(self) -> list[dict[str, Any]]:
        """List all embedding stores in the graph.

        Returns:
            List of dicts with ``node_type``, ``text_column``, ``dimension``, ``count``.
        """
        ...

    def embedding_diagnostics(self, node_type: Optional[str] = None) -> list[dict[str, Any]]:
        """Diagnose embedding coverage per (node_type, text_column).

        Companion to ``list_embeddings()``. Surfaces three states:

        - ``"embedded"``: a store exists and at least one node has the
          underlying property.
        - ``"embeddable"``: nodes have a string-typed property but no
          embedding store has been created or restored.
        - ``"store_orphan"``: a store exists but no node in the current
          graph has the underlying property — the symptom
          ``import_embeddings()`` warns about when keys mismatch.

        Use this after a ``UserWarning`` from ``import_embeddings()`` to
        see *which* stores were dropped or partially mapped.

        Args:
            node_type: Optional filter. When set, only that node type is
                scanned. When ``None``, every type in the graph is scanned
                — may be expensive on graphs with millions of nodes.

        Returns:
            List of dicts with: ``node_type``, ``text_column``,
            ``embedding_key`` (= ``f"{text_column}_emb"``),
            ``nodes_with_property``, ``nodes_embedded``,
            ``dimension`` (or ``None``), ``metric`` (or ``None``),
            and ``status``.
        """
        ...

    def export_embeddings(
        self,
        path: str,
        node_types: Union[list[str], dict[str, list[str]], None] = None,
    ) -> dict[str, int]:
        """Export embeddings to a standalone ``.kgle`` file, keyed by node ID.

        Args:
            path: Output ``.kgle`` file path.
            node_types: Optional filter.

                - ``None`` (default): export all stores.
                - ``list[str]``: only stores whose ``node_type`` is in the list.
                - ``dict[str, list[str]]``: per-type list of text columns to export.

        Returns:
            Dict with ``stores`` (count of stores written) and
            ``embeddings`` (total embedding vectors written).
        """
        ...

    def import_embeddings(self, path: str) -> dict[str, int]:
        """Import embeddings from a ``.kgle`` file.

        Matches embeddings to nodes by ``(node_type, node_id)``. Embeddings
        whose node ID doesn't exist in the current graph are skipped.

        When all embeddings — or whole per-type stores — fail to match,
        a ``UserWarning`` is emitted to surface the silent-drop case. This
        most commonly indicates the ``.kgle`` file was exported from a
        different graph, or that the node ID schema has drifted (e.g. the
        ``code_tree`` qualified-name format changed across kglite versions).

        Args:
            path: Path to a ``.kgle`` file previously created by
                ``export_embeddings()``.

        Returns:
            Dict with:

            - ``stores``: number of stores actually inserted.
            - ``imported``: total embedding vectors matched to current nodes.
            - ``skipped``: total entries in the file that didn't match.
            - ``dropped_stores``: number of per-type stores in the file that
              contained entries but had zero matches (so the store was
              not inserted).
        """
        ...

    def remove_embeddings(self, node_type: str, text_column: str) -> None:
        """Remove an embedding store.

        Args:
            node_type: The node type.
            text_column: Source text column name (e.g. ``'summary'``).
        """
        ...

    @overload
    def embeddings(self, node_type: str, text_column: str) -> dict[Any, list[float]]:
        """Retrieve all embeddings for a node type.

        Args:
            node_type: The node type (e.g. 'Article').
            text_column: Source text column name (e.g. 'summary').

        Returns:
            Dict mapping node IDs to embedding vectors.
        """
        ...

    @overload
    def embeddings(self, text_column: str) -> dict[Any, list[float]]:
        """Retrieve embeddings for nodes in the current selection.

        Args:
            text_column: Source text column name (e.g. 'summary').

        Returns:
            Dict mapping node IDs to embedding vectors.
        """
        ...

    def embedding(self, node_type: str, text_column: str, node_id: Any) -> list[float] | None:
        """Retrieve a single node's embedding vector.

        Args:
            node_type: The node type (e.g. 'Article').
            text_column: Source text column name (e.g. 'summary').
            node_id: The node ID to look up.

        Returns:
            The embedding vector as a list of floats, or None if not found.
        """
        ...

    def set_embedder(self, model: EmbeddingModel) -> None:
        """Register an embedding model on the graph.

        After calling this, ``embed_texts()`` and ``search_text()`` use the
        registered model automatically.  The model is **not** serialized —
        call ``set_embedder()`` again after deserializing.

        If the model has optional ``load()`` / ``unload()`` methods, they are
        called automatically around each embedding operation.

        Args:
            model: An embedding model with ``dimension`` and ``embed()`` — see
                :class:`EmbeddingModel`.

        Example::

            g.set_embedder(my_model)
        """
        ...

    def embed_texts(
        self,
        node_type: str,
        text_column: str,
        batch_size: int = 256,
        show_progress: bool = True,
        replace: bool = False,
    ) -> dict[str, int]:
        """Embed a text column for all nodes of a given type.

        Uses the model registered via ``set_embedder()``.  Reads each node's
        ``text_column`` property, calls ``model.embed()`` in batches, and
        stores the resulting vectors as ``{text_column}_emb``.
        Nodes with missing or non-string text are skipped.

        By default, nodes that already have an embedding are skipped.
        Pass ``replace=True`` to re-embed everything.

        Shows a tqdm progress bar by default (requires ``tqdm``).

        Args:
            node_type: The node type to embed (e.g. ``'Article'``).
            text_column: The node property containing text to embed.
            batch_size: Number of texts per ``model.embed()`` call (default 256).
            show_progress: Show a tqdm progress bar (default ``True``).
                Silently falls back to no bar if ``tqdm`` is not installed.
            replace: Re-embed all nodes, even those with existing embeddings
                (default ``False``).

        Returns:
            Dict with ``embedded``, ``skipped``, ``skipped_existing``, and ``dimension``.

        Example::

            g.set_embedder(my_model)
            g.embed_texts("Article", "summary")
            # Embedding Article.summary: 100%|████████| 1000/1000 [00:05<00:00]

            # Add new articles, then re-run — only new ones get embedded:
            g.embed_texts("Article", "summary")  # skips already-embedded nodes
        """
        ...

    def search_text(
        self,
        text_column: str,
        query: str,
        top_k: int = 10,
        metric: str = "cosine",
        to_df: bool = False,
    ) -> list[dict[str, Any]] | pd.DataFrame:
        """Search embeddings using a text query.

        Uses the model registered via ``set_embedder()`` to embed the query,
        then performs vector search within the current selection.  Refer to
        the text column name (e.g. ``"summary"``); the graph resolves it to
        ``"summary_emb"`` internally.

        Args:
            text_column: Text column whose embeddings to search (e.g. ``'summary'``).
            query: The text query to search for.
            top_k: Number of results (default 10).
            metric: ``'cosine'`` (default), ``'dot_product'``, ``'euclidean'``, or ``'poincare'``.
            to_df: If True, return a pandas DataFrame.

        Returns:
            Same format as ``vector_search()`` — list of dicts or DataFrame.

        Example::

            results = g.select("Article").search_text(
                "summary", "find AI articles", top_k=10
            )
        """
        ...

    def begin(self, timeout_ms: Optional[int] = None) -> Transaction:
        """Begin a read-write transaction — returns a Transaction with a working copy.

        Creates a snapshot (deep-clone) of the current graph state. All mutations
        within the transaction are isolated until ``commit()`` is called. If rolled
        back (or dropped without committing), no changes are applied.

        Uses optimistic concurrency control: ``commit()`` will raise
        ``RuntimeError`` if the graph was modified by another transaction since
        ``begin()`` was called.

        Args:
            timeout_ms: Optional transaction-level timeout in milliseconds.
                If set, all operations after the deadline raise ``TimeoutError``.

        Can be used as a context manager::

            with graph.begin() as tx:
                tx.cypher("CREATE (n:Person {name: 'Alice', age: 30})")
                tx.cypher("CREATE (n:Person {name: 'Bob', age: 25})")
                # auto-commits on success, auto-rollbacks on exception
        """
        ...

    def begin_read(self, timeout_ms: Optional[int] = None) -> Transaction:
        """Begin a read-only transaction — O(1) cost, zero memory overhead.

        Returns a Transaction backed by an Arc reference to the current graph
        state. Mutations (CREATE, SET, DELETE, REMOVE, MERGE) are rejected.

        Ideal for concurrent read-heavy workloads (e.g. MCP server agents)
        where you want a consistent snapshot without the cost of a full clone.

        Args:
            timeout_ms: Optional transaction-level timeout in milliseconds.

        Can be used as a context manager::

            with graph.begin_read() as tx:
                result = tx.cypher("MATCH (n:Person) RETURN n.name")
                # auto-closes on exit (no commit needed)
        """
        ...

class Transaction:
    """An isolated transaction on a KnowledgeGraph.

    Created via :meth:`KnowledgeGraph.begin` (read-write) or
    :meth:`KnowledgeGraph.begin_read` (read-only).

    Read-write transactions:
        - **Snapshot isolation**: ``begin()`` clones the entire graph.
        - **Write isolation**: mutations modify only the working copy.
        - **Optimistic concurrency control**: ``commit()`` checks that the
          graph version hasn't changed since ``begin()``. If another transaction
          committed in between, ``RuntimeError`` is raised.
        - **Commit**: replaces the original graph's data atomically.

    Read-only transactions (``begin_read()``):
        - O(1) creation cost (Arc reference, no deep clone).
        - Mutations are rejected with ``RuntimeError``.
        - ``commit()`` is a no-op; ``rollback()`` releases the snapshot.
    """

    @property
    def is_read_only(self) -> bool:
        """Whether this is a read-only transaction."""
        ...

    def cypher(
        self,
        query: str,
        params: dict[str, Any] | None = None,
        to_df: bool = False,
        timeout_ms: Optional[int] = None,
    ) -> ResultView | pd.DataFrame:
        """Execute a Cypher query within this transaction.

        Same interface as :meth:`KnowledgeGraph.cypher` but operates on
        the transaction's working copy (or Arc snapshot for read-only).

        Args:
            query: Cypher query string. Supports ``EXPLAIN`` and ``PROFILE`` prefixes.
            params: Optional query parameters.
            to_df: If True, return a pandas DataFrame.
            timeout_ms: Per-query timeout in milliseconds (merged with transaction deadline).
        """
        ...

    def commit(self) -> None:
        """Commit the transaction — apply all changes to the original graph.

        For read-only transactions, this is a no-op.
        After commit, the transaction cannot be used again.

        Raises:
            RuntimeError: If the graph was modified since ``begin()`` (OCC conflict).
        """
        ...

    def rollback(self) -> None:
        """Roll back the transaction — discard all changes.

        After rollback, the transaction cannot be used again.
        """
        ...

    def __enter__(self) -> Transaction: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any | None,
    ) -> bool: ...
