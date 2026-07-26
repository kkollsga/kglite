// src/graph/pyapi/result_view.rs
// Lazy ResultView — Polars-style result container.
// Data stays in Rust and converts to Python only on access.
//
// Moved from src/graph/languages/cypher/result_view.rs in Phase 8 to bring
// all #[pyclass] definitions under pyapi/. The Cypher-internal preprocessing
// logic stays in `languages/cypher/py_convert.rs`; we import it here.

use crate::datatypes::values::Value;
use crate::graph::languages::cypher::py_convert::{
    preprocess_values_owned, preprocessed_result_to_dataframe, preprocessed_value_to_py,
    stats_to_py, PreProcessedValue,
};
use crate::graph::languages::cypher::{
    ClauseStats, CypherResult, LazyResultDescriptor, MutationStats, QueryDiagnostics,
};
use crate::graph::pyapi::result_table::format_table;
use kglite_core::api::algorithms::CentralityResult;
use kglite_core::api::{DirGraph, NodeData};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PySlice};
use pyo3::IntoPyObjectExt;
use std::borrow::Cow;
use std::collections::HashSet;
use std::sync::{Arc, Mutex};

/// Lazy result container — data stays in Rust until you access it.
///
/// Returned by ``cypher()``, centrality methods, ``collect()`` (flat),
/// and ``sample()``.
///
/// Data is only converted to Python objects when you actually access rows
/// (via iteration, indexing, ``to_list()``, or ``to_df()``).  This makes
/// ``cypher()`` calls fast even for large result sets — the cost is
/// deferred to when you consume the data.
///
/// Quick reference::
///
/// ```text
/// r = g.cypher("MATCH (n:Person) RETURN n.name, n.age ORDER BY n.age")
///
/// len(r)           # row count (O(1), no conversion)
/// bool(r)          # True if non-empty
/// r[0]             # single row as dict  {'n.name': 'Alice', 'n.age': 30}
/// r[-1]            # last row
/// r[1:3]           # slice → new ResultView
/// r.columns        # column names
/// r.head(5)        # first 5 rows → new ResultView
/// r.tail(5)        # last 5 rows → new ResultView
/// r.to_list()      # all rows as list[dict]
/// r.to_dicts()     # alias for to_list() (polars/pandas naming)
/// r.one()          # first row as dict, or None if empty
/// r.scalar()       # first column of first row, or None if empty
/// r.column("n.age")# all values for one column as a list
/// r.to_df()        # pandas DataFrame
/// r.to_gdf()       # GeoDataFrame (requires geopandas)
/// r.stats          # mutation stats (CREATE/SET/DELETE only)
/// r.profile        # PROFILE stats (only with "PROFILE MATCH ...")
///
/// for row in r:    # iterate rows as dicts (one at a time)
///     print(row)
/// ```
///
/// Indexing is row-wise (`r[i]` / `r[1:3]`); `r["col"]` is unsupported (only
/// `"columns"` / `"rows"` are valid string keys). For a single column use the
/// explicit accessor `r.column("col")` (returns a `list`), or `r.scalar()` /
/// `r.one()` for the first cell / first row of small results.
/// Lazy row backing — set when the planner flagged a terminal RETURN as
/// `lazy_eligible`. The executor stops short of evaluating per-row
/// projection expressions; `LazyRows` carries the unresolved
/// `ResultRow.node_bindings` plus the `RETURN` items so cells materialise
/// on demand. The graph `Arc` keeps the storage alive for the lifetime of
/// the view; the `Mutex<Vec<...>>` caches materialised rows so repeat
/// access doesn't re-read from disk.
/// Cell budget (`rows × columns`) at or below which a lazy-eligible result is
/// materialised up front instead of holding the graph open — see
/// [`ResultView::from_cypher_result_with_graph`].
///
/// **This is paid by every eligible read, so it is sized to disappear into
/// noise, not to maximise coverage.** Materialising is a win when the caller
/// consumes the rows (one batched pass beats per-row resolution by 12–15%
/// across every size measured) and dead loss when it does not — a query whose
/// rows are never read, or read only via `len()`, now does work it previously
/// skipped. Since a read is far more common than a read held across a write,
/// the tax on the former has to be invisible before the saving on the latter
/// counts for anything.
///
/// Measured on a 100k-node graph, `MATCH (n:Item) RETURN n.title, n.value
/// LIMIT k`, release build, never consumed — the shape the tracked
/// `test_bench_cypher_match` / `test_bench_columnar_cypher_match` benchmarks
/// run, and the shape that caught an earlier 4096-row version of this constant
/// as a 2× regression:
///
/// | cells | cost |
/// |---|---|
/// | 32 (k=16)  | +1.6% |
/// | 64 (k=32)  | +8.0% |
/// | 128 (k=64) | +18.6% |
/// | 200 (k=100)| +28.5% |
///
/// 32 is the last point that stays inside run-to-run noise. Budgeting in
/// *cells* rather than rows keeps that guarantee when a RETURN is wide: cost
/// tracks property reads, so 16 rows × 2 columns and 1 row × 32 columns are
/// the same work and neither should be treated as cheaper than the other.
///
/// The coverage this buys is small in rows and precisely the shape that
/// matters: a read-modify-write handler reads one entity, or a handful, and
/// holds it across the write. For that case the saving is the entire fork —
/// tens of milliseconds on a large graph against well under a microsecond of
/// eager work. Bigger results stay deferred and keep the pin; that residual is
/// documented on `ResultView` rather than paid for by every reader.
const EAGER_MATERIALISE_MAX_CELLS: usize = 32;

struct LazyRows {
    descriptor: LazyResultDescriptor,
    graph: Arc<DirGraph>,
    /// Memoised materialisation. `cache[i]` is `Some(row)` once
    /// `materialise_row(i)` has run for that row index. Mutex (rather than
    /// RwLock or cell) keeps it Send + Sync, which #[pyclass] requires.
    cache: Mutex<Vec<Option<Vec<PreProcessedValue>>>>,
}

#[pyclass(name = "ResultView", module = "kglite", frozen)]
pub struct ResultView {
    columns: Vec<String>,
    rows: Vec<Vec<PreProcessedValue>>,
    stats: Option<MutationStats>,
    profile: Option<Vec<ClauseStats>>,
    diagnostics: Option<QueryDiagnostics>,
    /// When `Some`, `rows` is empty and reads route through `lazy`. The
    /// executor flagged the RETURN as `lazy_eligible` and the receiver
    /// hands cells back row-by-row from `pending`.
    lazy: Option<LazyRows>,
}

// ========================================================================
// Rust-only constructors (not exposed to Python)
// ========================================================================

impl ResultView {
    /// Cypher read path: data already preprocessed during py.detach (GIL-free).
    /// O(1) — just moves owned data into the struct.
    /// Cypher read path: data already preprocessed during py.detach
    /// (GIL-free). `diagnostics` attaches elapsed-time / timeout
    /// bookkeeping for agent feedback; pass `None` when not applicable
    /// (mutation paths, centrality results).
    pub fn from_preprocessed(
        columns: Vec<String>,
        rows: Vec<Vec<PreProcessedValue>>,
        stats: Option<MutationStats>,
        profile: Option<Vec<ClauseStats>>,
        diagnostics: Option<QueryDiagnostics>,
    ) -> Self {
        ResultView {
            columns,
            rows,
            stats,
            profile,
            diagnostics,
            lazy: None,
        }
    }

    /// Cypher mutation path + Transaction: takes a CypherResult and preprocesses values.
    pub fn from_cypher_result(result: CypherResult) -> Self {
        let rows = preprocess_values_owned(result.rows);
        ResultView {
            columns: result.columns,
            rows,
            stats: result.stats,
            profile: result.profile,
            diagnostics: result.diagnostics,
            lazy: None,
        }
    }

    /// Cypher read path with lazy materialisation. When `result.lazy` is
    /// `Some`, the executor flagged the RETURN as `lazy_eligible` and
    /// skipped per-row property evaluation; this constructor stashes the
    /// pending rows and graph reference so cells materialise on demand at
    /// the Python boundary. Falls back to the eager
    /// `from_preprocessed`-equivalent path when `result.lazy` is `None`.
    ///
    /// **Small results materialise immediately and keep no graph reference.**
    /// A retained lazy view pins the `Arc<DirGraph>` it was built from, so the
    /// next write through the owning `KnowledgeGraph` finds a shared Arc and
    /// `Arc::make_mut` deep-clones the entire graph — every node, edge, index
    /// and embedding. That turns the ordinary read-then-write request handler
    /// (`rows = g.cypher(...)` then `g.cypher("... SET ...")`, with `rows`
    /// still in scope) into a whole-graph copy *per request*.
    ///
    /// Deferral only earns that risk when most rows are never consumed, which
    /// needs a large result to matter. Within the cell budget the projection is
    /// bounded and cheap, and paying it up front is strictly better than
    /// risking an O(V+E) fork — so the view drops the `Arc` and becomes eager.
    /// Materialisation failures fall through to the lazy path deliberately, so
    /// an error still surfaces at access time exactly as before.
    pub fn from_cypher_result_with_graph(result: CypherResult, graph: Arc<DirGraph>) -> Self {
        if let Some(lazy_desc) = result.lazy {
            let n = lazy_desc.len();
            // Saturating: a pathological column count must fall through to the
            // deferred path, never wrap into a small product and look cheap.
            let cells = n.saturating_mul(result.columns.len());
            if cells <= EAGER_MATERIALISE_MAX_CELLS {
                if let Ok(rows) =
                    kglite_core::api::cypher::materialise_lazy_range(&lazy_desc, &graph, 0..n)
                {
                    return ResultView {
                        columns: result.columns,
                        rows: preprocess_values_owned(rows),
                        stats: result.stats,
                        profile: result.profile,
                        diagnostics: result.diagnostics,
                        lazy: None,
                    };
                }
            }
            let cache = Mutex::new((0..n).map(|_| None).collect::<Vec<_>>());
            return ResultView {
                columns: result.columns,
                rows: Vec::new(),
                stats: result.stats,
                profile: result.profile,
                diagnostics: result.diagnostics,
                lazy: Some(LazyRows {
                    descriptor: lazy_desc,
                    graph,
                    cache,
                }),
            };
        }
        Self::from_cypher_result(result)
    }

    /// Centrality methods: resolves node_idx → NodeData lookups, builds rows.
    /// Pure Rust, no GIL needed.
    pub fn from_centrality(
        graph: &DirGraph,
        results: Vec<CentralityResult>,
        top_k: Option<usize>,
    ) -> Self {
        let _arena_guard = graph.begin_read_pass(); // disk arena guard (no-op on memory/mapped)
        let limit = top_k.unwrap_or(results.len());
        let columns = vec!["type".into(), "title".into(), "id".into(), "score".into()];

        let rows: Vec<Vec<PreProcessedValue>> = results
            .into_iter()
            .take(limit)
            .filter_map(|r| {
                graph.get_node(r.node_idx).map(|node| {
                    vec![
                        PreProcessedValue::Plain(Value::String(
                            node.node_type_str(&graph.interner).to_string(),
                        )),
                        PreProcessedValue::Plain(node.title().into_owned()),
                        PreProcessedValue::Plain(node.id().into_owned()),
                        PreProcessedValue::Plain(Value::Float64(r.score)),
                    ]
                })
            })
            .collect();

        ResultView {
            columns,
            rows,
            stats: None,
            profile: None,
            diagnostics: None,
            lazy: None,
        }
    }

    /// Discover property keys by scanning all nodes (fallback path).
    ///
    /// Excludes keys naming a canonical identity column — `from_nodes_with_graph`
    /// leads every row with `type`/`title`/`id` from the node header, so a
    /// stored property of the same name would repeat the column and shadow the
    /// canonical value.
    fn discover_property_keys(
        nodes: &[&NodeData],
        interner: &kglite_core::api::StringInterner,
    ) -> Vec<String> {
        let mut seen: HashSet<&str> = HashSet::new();
        let mut keys: Vec<String> = Vec::new();
        for node in nodes {
            for key in node.property_keys(interner) {
                if kglite_core::api::is_canonical_node_column(key) {
                    continue;
                }
                if seen.insert(key) {
                    keys.push(key.to_string());
                }
            }
        }
        keys.sort();
        keys
    }

    /// collect / sample with graph access: nodes + connection summaries.
    pub fn from_nodes_with_graph(
        graph: &DirGraph,
        node_indices: &[petgraph::graph::NodeIndex],
    ) -> Self {
        let _arena_guard = graph.begin_read_pass(); // disk arena guard (no-op on memory/mapped)
        let nodes_vec: Vec<&NodeData> = node_indices
            .iter()
            .filter_map(|&idx| graph.get_node(idx))
            .collect();

        // Compute union of property keys.
        // Fast path: if all nodes share a type, use TypeSchema (O(1) key discovery).
        let prop_keys: Vec<String> = if nodes_vec.len() > 50 {
            let first_type = nodes_vec[0].node_type;
            let all_same_type = nodes_vec.iter().all(|n| n.node_type == first_type);
            if all_same_type {
                let first_type_str = graph.interner.resolve(first_type);
                if let Some(schema) = graph.type_schemas.get(first_type_str) {
                    let mut keys: Vec<String> = schema
                        .iter()
                        .filter_map(|(_, ik)| {
                            graph
                                .interner
                                .try_resolve(ik)
                                .filter(|s| !kglite_core::api::is_canonical_node_column(s))
                                .map(|s| s.to_string())
                        })
                        .collect();
                    keys.sort();
                    keys
                } else {
                    Self::discover_property_keys(&nodes_vec, &graph.interner)
                }
            } else {
                Self::discover_property_keys(&nodes_vec, &graph.interner)
            }
        } else {
            Self::discover_property_keys(&nodes_vec, &graph.interner)
        };

        let mut columns = vec!["type".into(), "title".into(), "id".into()];
        columns.extend(prop_keys.iter().cloned());

        let rows: Vec<Vec<PreProcessedValue>> = nodes_vec
            .iter()
            .map(|node| {
                let mut row = vec![
                    PreProcessedValue::Plain(Value::String(
                        node.node_type_str(&graph.interner).to_string(),
                    )),
                    PreProcessedValue::Plain(node.title().into_owned()),
                    PreProcessedValue::Plain(node.id().into_owned()),
                ];
                for key in &prop_keys {
                    row.push(PreProcessedValue::Plain(
                        node.get_property(key)
                            .map(Cow::into_owned)
                            .unwrap_or(Value::Null),
                    ));
                }
                row
            })
            .collect();

        ResultView {
            columns,
            rows,
            stats: None,
            profile: None,
            diagnostics: None,
            lazy: None,
        }
    }

    /// Number of rows — handles both eager (`self.rows`) and lazy
    /// (`self.lazy.pending`) backings without forcing materialisation.
    fn effective_len(&self) -> usize {
        if let Some(lz) = &self.lazy {
            return lz.descriptor.len();
        }
        self.rows.len()
    }

    /// Materialise row `index` for the lazy backing, evaluating each
    /// `RETURN` item against the row's stashed `node_bindings` /
    /// `edge_bindings`. Caches the result so repeat access is O(1) after
    /// the first call. Caller already holds the cache lock; we re-lock
    /// inside to keep the borrow scope tight.
    fn materialise_lazy_row(&self, index: usize) -> PyResult<Vec<PreProcessedValue>> {
        let lz = self
            .lazy
            .as_ref()
            .expect("materialise_lazy_row called without lazy backing");
        // Quick read-after-write check: if cache hit, return clone.
        if let Some(row) = lz.cache.lock().unwrap().get(index).and_then(|c| c.clone()) {
            return Ok(row);
        }
        let cells: Vec<PreProcessedValue> =
            kglite_core::api::cypher::materialise_lazy_row(&lz.descriptor, &lz.graph, index)
                .map_err(crate::error_py::kg_to_pyerr)?
                .into_iter()
                .map(PreProcessedValue::Plain)
                .collect();
        // Cache for repeat access.
        if let Some(slot) = lz.cache.lock().unwrap().get_mut(index) {
            *slot = Some(cells.clone());
        }
        Ok(cells)
    }

    /// Materialise a contiguous lazy range with one provenance check and one
    /// cache lock per batch. The core still holds a separate disk arena guard
    /// for every row, preserving bounded arena lifetime under large reads.
    fn materialise_lazy_range(
        &self,
        range: std::ops::Range<usize>,
    ) -> PyResult<Vec<Vec<PreProcessedValue>>> {
        let lz = self
            .lazy
            .as_ref()
            .expect("materialise_lazy_range called without lazy backing");
        {
            let cache = lz.cache.lock().unwrap();
            if !range.is_empty() && range.clone().all(|index| cache[index].is_some()) {
                return Ok(range.map(|index| cache[index].clone().unwrap()).collect());
            }
        }

        let start = range.start;
        let cells = kglite_core::api::cypher::materialise_lazy_range(
            &lz.descriptor,
            &lz.graph,
            range.clone(),
        )
        .map_err(crate::error_py::kg_to_pyerr)?;
        let processed: Vec<Vec<PreProcessedValue>> = cells
            .into_iter()
            .map(|row| row.into_iter().map(PreProcessedValue::Plain).collect())
            .collect();

        let mut cache = lz.cache.lock().unwrap();
        for (offset, row) in processed.into_iter().enumerate() {
            let slot = &mut cache[start + offset];
            if slot.is_none() {
                *slot = Some(row);
            }
        }
        Ok(range.map(|index| cache[index].clone().unwrap()).collect())
    }

    /// Force materialisation of every lazy row (or return the existing
    /// eager rows) — used by `to_df` which consumes every cell. Public to
    /// the crate so `kg_core::cypher` can build a DataFrame from a lazy
    /// result without re-implementing the resolver.
    pub fn materialise_all(&self) -> PyResult<Vec<Vec<PreProcessedValue>>> {
        if self.lazy.is_some() {
            return self.materialise_lazy_range(0..self.effective_len());
        }
        Ok(self.rows.clone())
    }

    /// Owned-clone of the column names; used in DataFrame construction
    /// where the consumer needs an owned slice.
    pub fn columns_owned(&self) -> Vec<String> {
        self.columns.clone()
    }

    /// Convert a single row to a Python dict. Used by __getitem__ and __iter__.
    fn row_to_py(&self, py: Python<'_>, index: usize) -> PyResult<Py<PyAny>> {
        let owned;
        let row: &Vec<PreProcessedValue> = if self.lazy.is_some() {
            owned = self.materialise_lazy_row(index)?;
            &owned
        } else {
            &self.rows[index]
        };
        let dict = PyDict::new(py);
        for (i, col) in self.columns.iter().enumerate() {
            if let Some(pv) = row.get(i) {
                dict.set_item(col, preprocessed_value_to_py(py, pv)?)?;
            } else {
                dict.set_item(col, py.None())?;
            }
        }
        Ok(dict.into_any().unbind())
    }

    /// Row → dict using pre-interned column-name keys. The bulk `to_list`
    /// path interns the column names ONCE and reuses them for every row,
    /// instead of re-creating the same Python strings per cell (the old
    /// `set_item(col, …)` allocated a fresh key string for every cell —
    /// rows × cols allocations of a handful of distinct names).
    fn row_to_py_keyed(
        &self,
        py: Python<'_>,
        index: usize,
        keys: &[pyo3::Bound<'_, pyo3::types::PyString>],
    ) -> PyResult<Py<PyAny>> {
        let owned;
        let row: &Vec<PreProcessedValue> = if self.lazy.is_some() {
            owned = self.materialise_lazy_row(index)?;
            &owned
        } else {
            &self.rows[index]
        };
        Self::row_values_to_py_keyed(py, row, keys)
    }

    fn row_values_to_py_keyed(
        py: Python<'_>,
        row: &[PreProcessedValue],
        keys: &[pyo3::Bound<'_, pyo3::types::PyString>],
    ) -> PyResult<Py<PyAny>> {
        let dict = PyDict::new(py);
        for (i, key) in keys.iter().enumerate() {
            if let Some(pv) = row.get(i) {
                dict.set_item(key, preprocessed_value_to_py(py, pv)?)?;
            } else {
                dict.set_item(key, py.None())?;
            }
        }
        Ok(dict.into_any().unbind())
    }
}

// ========================================================================
// Python protocol
// ========================================================================

#[pymethods]
impl ResultView {
    fn __len__(&self) -> usize {
        self.effective_len()
    }

    fn __bool__(&self) -> bool {
        self.effective_len() > 0
    }

    fn __getitem__(&self, py: Python<'_>, key: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        // String key access — dict-like interface for 'columns' and 'rows'
        if let Ok(skey) = key.extract::<String>() {
            match skey.as_str() {
                "columns" => return self.columns(py),
                "rows" => {
                    let rows: Vec<Py<PyAny>> = (0..self.effective_len())
                        .map(|i| self.row_to_py(py, i))
                        .collect::<Result<_, _>>()?;
                    return rows.into_py_any(py);
                }
                _ => {
                    return Err(pyo3::exceptions::PyKeyError::new_err(skey));
                }
            }
        }
        if let Ok(idx) = key.extract::<isize>() {
            // Integer indexing — returns a single row as dict
            let len = self.effective_len() as isize;
            let actual = if idx < 0 { len + idx } else { idx };
            if actual < 0 || actual >= len {
                return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                    "index {} out of range for ResultView with {} rows",
                    idx,
                    self.effective_len()
                )));
            }
            self.row_to_py(py, actual as usize)
        } else if let Ok(slice) = key.cast::<PySlice>() {
            // Slice indexing — returns a new ResultView. Lazy slicing
            // materialises the K rows and returns an eager sub-view; this
            // is intentional — the caller asked for a snapshot subset, so
            // forcing materialisation keeps the contract simple.
            let len = self.effective_len();
            let indices = slice.indices(len as isize)?;
            let mut sliced_rows = Vec::new();
            let mut i = indices.start;
            while (indices.step > 0 && i < indices.stop) || (indices.step < 0 && i > indices.stop) {
                if i >= 0 && (i as usize) < len {
                    if self.lazy.is_some() {
                        sliced_rows.push(self.materialise_lazy_row(i as usize)?);
                    } else {
                        sliced_rows.push(self.rows[i as usize].clone());
                    }
                }
                i += indices.step;
            }
            Py::new(
                py,
                ResultView {
                    columns: self.columns.clone(),
                    rows: sliced_rows,
                    stats: None,
                    profile: None,
                    diagnostics: None,
                    lazy: None,
                },
            )
            .map(|v| v.into_any())
        } else {
            Err(pyo3::exceptions::PyTypeError::new_err(
                "indices must be integers, slices, or string keys ('columns', 'rows')",
            ))
        }
    }

    fn __iter__(slf: Py<Self>) -> ResultIter {
        ResultIter {
            view: slf,
            index: 0,
        }
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        // Materialise lazy rows for printing; format_table needs concrete
        // PreProcessedValues. For very large lazy results, callers should
        // use head()/tail() instead of repr.
        if self.lazy.is_some() {
            let rows = self.materialise_all()?;
            return Ok(format_table(py, &self.columns, &rows));
        }
        Ok(format_table(py, &self.columns, &self.rows))
    }

    fn __str__(&self, py: Python<'_>) -> PyResult<String> {
        self.__repr__(py)
    }

    /// Column names as a list of strings.
    ///
    /// Example::
    ///
    /// ```text
    /// r = g.cypher("MATCH (n) RETURN n.name, n.age")
    /// r.columns   # ['n.name', 'n.age']
    /// ```
    #[getter]
    fn columns(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.columns.clone().into_py_any(py)
    }

    /// Mutation statistics (CREATE/SET/DELETE queries), or None for reads.
    ///
    /// Returns a dict with keys like ``nodes_created``, ``properties_set``,
    /// ``relationships_created``, etc.
    #[getter]
    fn stats(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match &self.stats {
            Some(s) => stats_to_py(py, s).map(|d| d.into_any().unbind()),
            None => Ok(py.None()),
        }
    }

    /// PROFILE execution statistics, or None for non-profiled queries.
    /// Returns a list of dicts with keys: clause, rows_in, rows_out, elapsed_us.
    #[getter]
    fn profile(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match &self.profile {
            Some(steps) => {
                let list = pyo3::types::PyList::empty(py);
                for step in steps {
                    let dict = PyDict::new(py);
                    dict.set_item("clause", &step.clause_name)?;
                    dict.set_item("rows_in", step.rows_in)?;
                    dict.set_item("rows_out", step.rows_out)?;
                    dict.set_item("elapsed_us", step.elapsed_us)?;
                    list.append(dict)?;
                }
                Ok(list.into_any().unbind())
            }
            None => Ok(py.None()),
        }
    }

    /// Lightweight execution diagnostics, or None when the backend
    /// didn't populate them (mutation queries, EXPLAIN, transaction
    /// paths).
    ///
    /// Returns a dict with ``elapsed_ms`` (wall-clock query duration),
    /// ``timed_out`` (True when the deadline fired), and ``timeout_ms``
    /// (the deadline that was in effect, or None). Use this to tune
    /// ``timeout_ms`` or move toward anchored queries when queries
    /// approach the deadline.
    #[getter]
    fn diagnostics(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match &self.diagnostics {
            Some(d) => {
                let dict = PyDict::new(py);
                dict.set_item("elapsed_ms", d.elapsed_ms)?;
                dict.set_item("timed_out", d.timed_out)?;
                match d.timeout_ms {
                    Some(ms) => dict.set_item("timeout_ms", ms)?,
                    None => dict.set_item("timeout_ms", py.None())?,
                }
                dict.set_item("warnings", d.warnings.clone())?;
                Ok(dict.into_any().unbind())
            }
            None => Ok(py.None()),
        }
    }

    /// Materialize all rows as a list of dicts.
    ///
    /// Example::
    ///
    /// ```text
    /// r.to_list()  # [{'name': 'Alice', 'age': 30}, ...]
    /// ```
    fn to_list(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let list = pyo3::types::PyList::empty(py);
        // Intern the column-name dict keys ONCE and reuse them for every row
        // (Phase 2: avoids re-creating the same Python strings per cell).
        let keys: Vec<pyo3::Bound<'_, pyo3::types::PyString>> = self
            .columns
            .iter()
            .map(|c| pyo3::types::PyString::intern(py, c))
            .collect();
        if self.lazy.is_some() {
            for row in self.materialise_all()? {
                list.append(Self::row_values_to_py_keyed(py, &row, &keys)?)?;
            }
        } else {
            for i in 0..self.rows.len() {
                list.append(self.row_to_py_keyed(py, i, &keys)?)?;
            }
        }
        Ok(list.into_any().unbind())
    }

    /// Alias for ``to_list()`` — materialize all rows as a list of dicts.
    ///
    /// Provided for callers coming from polars (``.to_dicts()``) or pandas
    /// (``.to_dict(orient="records")``), where the row-wise dict accessor
    /// carries this name. Identical behaviour to ``to_list()``.
    fn to_dicts(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.to_list(py)
    }

    /// First row as a dict, or None if the result is empty.
    ///
    /// Does not materialize the whole result set — only the first row is
    /// converted (the same row-materialization path as ``r[0]``).
    ///
    /// Example::
    ///
    /// ```text
    /// row = g.cypher("MATCH (n:Person {id: 1}) RETURN n.name, n.age").one()
    /// # {'n.name': 'Alice', 'n.age': 30}  or  None if no match
    /// ```
    fn one(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        if self.effective_len() == 0 {
            return Ok(py.None());
        }
        self.row_to_py(py, 0)
    }

    /// First column of the first row, or None if the result is empty.
    ///
    /// The "first column" is decided by the query's ``RETURN`` order (the
    /// same order as ``columns``). Convenient for aggregate queries::
    ///
    /// ```text
    /// n = g.cypher("MATCH (n:Person) RETURN count(n)").scalar()  # an int
    /// ```
    ///
    /// Only the first cell of the first row is materialized — the rest of the
    /// result set is never converted.
    fn scalar(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        if self.effective_len() == 0 || self.columns.is_empty() {
            return Ok(py.None());
        }
        let owned;
        let row: &Vec<PreProcessedValue> = if self.lazy.is_some() {
            owned = self.materialise_lazy_row(0)?;
            &owned
        } else {
            &self.rows[0]
        };
        match row.first() {
            Some(pv) => preprocessed_value_to_py(py, pv),
            None => Ok(py.None()),
        }
    }

    /// All values for one named column, as a list (no DataFrame conversion).
    ///
    /// Raises ``KeyError`` for an unknown column name, listing the available
    /// columns. This is the explicit column accessor — row indexing
    /// (``r[i]``) stays integer-only.
    ///
    /// Example::
    ///
    /// ```text
    /// r = g.cypher("MATCH (n:Person) RETURN n.name, n.age")
    /// r.column("n.name")   # ['Alice', 'Bob', ...]
    /// ```
    fn column(&self, py: Python<'_>, name: &str) -> PyResult<Py<PyAny>> {
        let col_idx = match self.columns.iter().position(|c| c == name) {
            Some(i) => i,
            None => {
                let available = self
                    .columns
                    .iter()
                    .map(|c| format!("'{c}'"))
                    .collect::<Vec<_>>()
                    .join(", ");
                return Err(pyo3::exceptions::PyKeyError::new_err(format!(
                    "no column {name:?} in ResultView; available columns: [{available}]"
                )));
            }
        };
        let list = pyo3::types::PyList::empty(py);
        let lazy_rows = self
            .lazy
            .is_some()
            .then(|| self.materialise_all())
            .transpose()?;
        for i in 0..self.effective_len() {
            let row: &Vec<PreProcessedValue> = if let Some(rows) = &lazy_rows {
                &rows[i]
            } else {
                &self.rows[i]
            };
            match row.get(col_idx) {
                Some(pv) => list.append(preprocessed_value_to_py(py, pv)?)?,
                None => list.append(py.None())?,
            }
        }
        Ok(list.into_any().unbind())
    }

    /// First *n* rows as a new ResultView (default 5). Data stays lazy.
    ///
    /// Example::
    ///
    /// ```text
    /// r.head()     # first 5 rows
    /// r.head(10)   # first 10 rows
    /// ```
    #[pyo3(signature = (n=5))]
    fn head(&self, n: usize) -> PyResult<Self> {
        let total = self.effective_len();
        let take = n.min(total);
        // For lazy results we materialise the first `take` rows so the
        // returned view is concrete (head() callers commonly print or
        // sample). The lazy → eager conversion is paid once.
        let rows: Vec<Vec<PreProcessedValue>> = if self.lazy.is_some() {
            self.materialise_lazy_range(0..take)?
        } else {
            self.rows[..take].to_vec()
        };
        Ok(ResultView {
            columns: self.columns.clone(),
            rows,
            stats: None,
            profile: None,
            diagnostics: None,
            lazy: None,
        })
    }

    /// Last *n* rows as a new ResultView (default 5). Data stays lazy.
    ///
    /// Example::
    ///
    /// ```text
    /// r.tail()     # last 5 rows
    /// r.tail(10)   # last 10 rows
    /// ```
    #[pyo3(signature = (n=5))]
    fn tail(&self, n: usize) -> PyResult<Self> {
        let len = self.effective_len();
        let start = len.saturating_sub(n);
        let rows: Vec<Vec<PreProcessedValue>> = if self.lazy.is_some() {
            self.materialise_lazy_range(start..len)?
        } else {
            self.rows[start..].to_vec()
        };
        Ok(ResultView {
            columns: self.columns.clone(),
            rows,
            stats: None,
            profile: None,
            diagnostics: None,
            lazy: None,
        })
    }

    /// Materialize as a pandas DataFrame.
    ///
    /// Example::
    ///
    /// ```text
    /// df = r.to_df()
    /// df.plot(x='year', y='count')
    /// ```
    fn to_df(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        if self.lazy.is_some() {
            // DataFrame consumes every cell — force lazy materialisation.
            let materialised = self.materialise_all()?;
            return preprocessed_result_to_dataframe(py, &self.columns, &materialised);
        }
        preprocessed_result_to_dataframe(py, &self.columns, &self.rows)
    }

    /// Convert to a GeoDataFrame with a geometry column parsed from WKT.
    ///
    /// Materializes the data as a pandas DataFrame, then converts the
    /// specified WKT string column into shapely geometries and returns
    /// a geopandas GeoDataFrame.
    ///
    /// Args:
    ///     geometry_column: Name of the column containing WKT strings (default: 'geometry')
    ///     crs: Coordinate reference system (e.g. 'EPSG:4326'), or None
    ///
    /// Returns:
    ///     A geopandas GeoDataFrame
    #[pyo3(signature = (geometry_column="geometry", crs=None))]
    fn to_gdf(
        &self,
        py: Python<'_>,
        geometry_column: &str,
        crs: Option<&str>,
    ) -> PyResult<Py<PyAny>> {
        let df = if self.lazy.is_some() {
            let materialised = self.materialise_all()?;
            preprocessed_result_to_dataframe(py, &self.columns, &materialised)?
        } else {
            preprocessed_result_to_dataframe(py, &self.columns, &self.rows)?
        };

        let gpd = py.import("geopandas").map_err(|_| {
            PyErr::new::<pyo3::exceptions::PyImportError, _>(
                "geopandas is required for to_gdf(). Install it with: pip install geopandas",
            )
        })?;

        // gpd.GeoSeries.from_wkt(df[geometry_column])
        let geo_series_cls = gpd.getattr("GeoSeries")?;
        let wkt_col = df.call_method1(py, "__getitem__", (geometry_column,))?;
        let geo_series = geo_series_cls.call_method1("from_wkt", (wkt_col,))?;

        // df[geometry_column] = geo_series
        df.call_method1(py, "__setitem__", (geometry_column, geo_series))?;

        // gpd.GeoDataFrame(df, geometry=geometry_column, crs=crs)
        let kwargs = PyDict::new(py);
        kwargs.set_item("geometry", geometry_column)?;
        if let Some(crs_val) = crs {
            kwargs.set_item("crs", crs_val)?;
        }
        let gdf_cls = gpd.getattr("GeoDataFrame")?;
        let gdf = gdf_cls.call((df,), Some(&kwargs))?;
        Ok(gdf.unbind())
    }
}

// ========================================================================
// ResultIter — lazy iterator over ResultView rows
// ========================================================================

/// Iterator for ResultView. Converts one row per __next__ call.
#[pyclass(name = "ResultIter", module = "kglite")]
pub struct ResultIter {
    view: Py<ResultView>,
    index: usize,
}

#[pymethods]
impl ResultIter {
    fn __iter__(slf: Py<Self>) -> Py<Self> {
        slf
    }

    fn __next__(&mut self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        let view = self.view.borrow(py);
        if self.index >= view.effective_len() {
            return Ok(None);
        }
        let result = view.row_to_py(py, self.index)?;
        self.index += 1;
        Ok(Some(result))
    }
}
