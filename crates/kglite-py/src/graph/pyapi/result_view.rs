// src/graph/pyapi/result_view.rs
// Lazy ResultView — Polars-style result container.
// Data stays in Rust and converts to Python only on access.
//
// Moved from src/graph/languages/cypher/result_view.rs in Phase 8 to bring
// all #[pyclass] definitions under pyapi/. The Cypher-internal preprocessing
// logic stays in `languages/cypher/py_convert.rs`; we import it here.

use crate::datatypes::values::Value;
use crate::graph::algorithms::graph_algorithms::CentralityResult;
use crate::graph::languages::cypher::ast::{Expression, ReturnItem};
use crate::graph::languages::cypher::py_convert::{
    preprocess_values_owned, preprocessed_result_to_dataframe, preprocessed_value_to_py,
    stats_to_py, PreProcessedValue,
};
use crate::graph::languages::cypher::result::{
    ClauseStats, CypherResult, MutationStats, QueryDiagnostics, ResultRow,
};
use crate::graph::schema::{DirGraph, NodeData};
use crate::graph::storage::GraphRead;
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
struct LazyRows {
    pending: Vec<ResultRow>,
    return_items: Vec<ReturnItem>,
    graph: Arc<DirGraph>,
    /// Memoised materialisation. `cache[i]` is `Some(row)` once
    /// `materialise_row(i)` has run for that row index. Mutex (rather than
    /// RwLock or cell) keeps it Send + Sync, which #[pyclass] requires.
    cache: Mutex<Vec<Option<Vec<PreProcessedValue>>>>,
}

#[pyclass(name = "ResultView")]
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
    pub fn from_cypher_result_with_graph(result: CypherResult, graph: Arc<DirGraph>) -> Self {
        if let Some(lazy_desc) = result.lazy {
            let n = lazy_desc.pending_rows.len();
            let cache = Mutex::new((0..n).map(|_| None).collect::<Vec<_>>());
            return ResultView {
                columns: result.columns,
                rows: Vec::new(),
                stats: result.stats,
                profile: result.profile,
                diagnostics: result.diagnostics,
                lazy: Some(LazyRows {
                    pending: lazy_desc.pending_rows,
                    return_items: lazy_desc.return_items,
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
    fn discover_property_keys(
        nodes: &[&NodeData],
        interner: &crate::graph::schema::StringInterner,
    ) -> Vec<String> {
        let mut seen: HashSet<&str> = HashSet::new();
        let mut keys: Vec<String> = Vec::new();
        for node in nodes {
            for key in node.property_keys(interner) {
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
                        .filter_map(|(_, ik)| graph.interner.try_resolve(ik).map(|s| s.to_string()))
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
            return lz.pending.len();
        }
        self.rows.len()
    }

    /// Materialise row `index` for the lazy backing, evaluating each
    /// `RETURN` item against the row's stashed `node_bindings` /
    /// `edge_bindings`. Caches the result so repeat access is O(1) after
    /// the first call. Caller already holds the cache lock; we re-lock
    /// inside to keep the borrow scope tight.
    fn materialise_lazy_row(&self, index: usize) -> Vec<PreProcessedValue> {
        let lz = self
            .lazy
            .as_ref()
            .expect("materialise_lazy_row called without lazy backing");
        // Quick read-after-write check: if cache hit, return clone.
        if let Some(row) = lz.cache.lock().unwrap().get(index).and_then(|c| c.clone()) {
            return row;
        }
        let pending_row = &lz.pending[index];
        let cells: Vec<PreProcessedValue> = lz
            .return_items
            .iter()
            .map(|item| {
                let val = match &item.expression {
                    Expression::PropertyAccess { variable, property } => {
                        if let Some(&node_idx) = pending_row.node_bindings.get(variable) {
                            resolve_node_property_lazy(&lz.graph, node_idx, property)
                        } else if let Some(eb) = pending_row.edge_bindings.get(variable) {
                            // Edge property access — fall through to a generic
                            // "best effort" via the edge data; rare on lazy
                            // path because RETURN items here are usually
                            // node-property reads, but keep it correct.
                            resolve_edge_property_lazy(&lz.graph, eb, property)
                        } else if let Some(v) = pending_row.projected.get(variable) {
                            v.clone()
                        } else {
                            Value::Null
                        }
                    }
                    Expression::Variable(v) => {
                        if let Some(&node_idx) = pending_row.node_bindings.get(v) {
                            resolve_node_full_lazy(&lz.graph, node_idx)
                        } else if let Some(val) = pending_row.projected.get(v) {
                            val.clone()
                        } else {
                            Value::Null
                        }
                    }
                    _ => Value::Null,
                };
                PreProcessedValue::Plain(val)
            })
            .collect();
        // Cache for repeat access.
        if let Some(slot) = lz.cache.lock().unwrap().get_mut(index) {
            *slot = Some(cells.clone());
        }
        cells
    }

    /// Force materialisation of every lazy row (or return the existing
    /// eager rows) — used by `to_df` which consumes every cell. Public to
    /// the crate so `kg_core::cypher` can build a DataFrame from a lazy
    /// result without re-implementing the resolver.
    pub fn materialise_all(&self) -> Vec<Vec<PreProcessedValue>> {
        if self.lazy.is_some() {
            return (0..self.effective_len())
                .map(|i| self.materialise_lazy_row(i))
                .collect();
        }
        self.rows.clone()
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
            owned = self.materialise_lazy_row(index);
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
}

/// Resolve `node_idx . property` against the graph for the lazy path.
/// Delegates to the same `resolve_node_property` helper the eager
/// executor uses, so virtual properties (spatial location/geometry,
/// named points, etc.) and per-type aliases work identically.
fn resolve_node_property_lazy(
    graph: &DirGraph,
    idx: petgraph::graph::NodeIndex,
    prop: &str,
) -> Value {
    use crate::graph::languages::cypher::executor::helpers::resolve_node_property;
    if let Some(node) = graph.graph.node_weight(idx) {
        return resolve_node_property(node, prop, graph);
    }
    Value::Null
}

fn resolve_edge_property_lazy(
    graph: &DirGraph,
    eb: &crate::graph::languages::cypher::result::EdgeBinding,
    prop: &str,
) -> Value {
    use crate::graph::languages::cypher::executor::helpers::resolve_edge_property;
    resolve_edge_property(graph, eb, prop)
}

fn resolve_node_full_lazy(graph: &DirGraph, idx: petgraph::graph::NodeIndex) -> Value {
    // Whole-node Variable access (e.g. `RETURN n` not `n.prop`). The lazy
    // planner pass currently rejects this shape, so the path is defensive
    // only — return the node id.
    graph.graph.get_node_id(idx).unwrap_or(Value::Null)
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
                        sliced_rows.push(self.materialise_lazy_row(i as usize));
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

    fn __repr__(&self) -> String {
        // Materialise lazy rows for printing; format_table needs concrete
        // PreProcessedValues. For very large lazy results, callers should
        // use head()/tail() instead of repr.
        if self.lazy.is_some() {
            let rows: Vec<Vec<PreProcessedValue>> = (0..self.effective_len())
                .map(|i| self.materialise_lazy_row(i))
                .collect();
            return format_table(&self.columns, &rows);
        }
        format_table(&self.columns, &self.rows)
    }

    fn __str__(&self) -> String {
        self.__repr__()
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
        for i in 0..self.effective_len() {
            list.append(self.row_to_py(py, i)?)?;
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
            owned = self.materialise_lazy_row(0);
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
        for i in 0..self.effective_len() {
            let owned;
            let row: &Vec<PreProcessedValue> = if self.lazy.is_some() {
                owned = self.materialise_lazy_row(i);
                &owned
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
    fn head(&self, n: usize) -> Self {
        let total = self.effective_len();
        let take = n.min(total);
        // For lazy results we materialise the first `take` rows so the
        // returned view is concrete (head() callers commonly print or
        // sample). The lazy → eager conversion is paid once.
        let rows: Vec<Vec<PreProcessedValue>> = if self.lazy.is_some() {
            (0..take).map(|i| self.materialise_lazy_row(i)).collect()
        } else {
            self.rows[..take].to_vec()
        };
        ResultView {
            columns: self.columns.clone(),
            rows,
            stats: None,
            profile: None,
            diagnostics: None,
            lazy: None,
        }
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
    fn tail(&self, n: usize) -> Self {
        let len = self.effective_len();
        let start = len.saturating_sub(n);
        let rows: Vec<Vec<PreProcessedValue>> = if self.lazy.is_some() {
            (start..len).map(|i| self.materialise_lazy_row(i)).collect()
        } else {
            self.rows[start..].to_vec()
        };
        ResultView {
            columns: self.columns.clone(),
            rows,
            stats: None,
            profile: None,
            diagnostics: None,
            lazy: None,
        }
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
            let materialised = self.materialise_all();
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
            let materialised = self.materialise_all();
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
#[pyclass(name = "ResultIter")]
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

// ========================================================================
// Pretty-print formatting for ResultView
// ========================================================================

fn format_preprocessed_value(pv: &PreProcessedValue) -> String {
    // Phase A.1 / C7a — ParsedJson variant deleted; only Plain remains.
    match pv {
        PreProcessedValue::Plain(v) => crate::datatypes::values::format_value(v),
    }
}

/// Format a ResultView as a Polars-style table.
///
/// Shows `shape: (rows, cols)` header, a bordered table with column names,
/// and for large results shows the first and last rows with `…` in between.
fn format_table(columns: &[String], rows: &[Vec<PreProcessedValue>]) -> String {
    if rows.is_empty() {
        return format!("shape: (0, {})\n(empty)", columns.len());
    }

    let n = rows.len();
    let max_col_width = 30;
    let max_display_rows = 20;

    // Decide which rows to show
    let (show_head, show_tail, truncated) = if n <= max_display_rows {
        (n, 0, false)
    } else {
        (10, 5, true)
    };

    // Format all visible cell values
    let mut formatted: Vec<Vec<String>> = Vec::new();
    for row in rows.iter().take(show_head) {
        formatted.push(
            row.iter()
                .map(|v| truncate_middle(&format_preprocessed_value(v), max_col_width))
                .collect(),
        );
    }
    if truncated {
        for row in rows.iter().skip(n - show_tail) {
            formatted.push(
                row.iter()
                    .map(|v| truncate_middle(&format_preprocessed_value(v), max_col_width))
                    .collect(),
            );
        }
    }

    // Compute column widths (header vs data)
    let num_cols = columns.len();
    let mut widths: Vec<usize> = columns.iter().map(|c| c.len()).collect();
    for row in &formatted {
        for (j, cell) in row.iter().enumerate() {
            if j < num_cols {
                widths[j] = widths[j].max(cell.len());
            }
        }
    }
    if truncated {
        // Ensure columns are wide enough for "…"
        for w in &mut widths {
            *w = (*w).max(1);
        }
    }

    let mut buf = String::with_capacity(n * 100);

    // Shape header
    buf.push_str(&format!("shape: ({}, {})\n", n, num_cols));

    // Top border: ┌──────┬──────┐
    buf.push('┌');
    for (j, w) in widths.iter().enumerate() {
        if j > 0 {
            buf.push('┬');
        }
        for _ in 0..(w + 2) {
            buf.push('─');
        }
    }
    buf.push_str("┐\n");

    // Header row: │ col1 ┆ col2 │
    buf.push('│');
    for (j, col) in columns.iter().enumerate() {
        if j > 0 {
            buf.push_str(" ┆");
        }
        buf.push_str(&format!(" {:width$}", col, width = widths[j]));
    }
    buf.push_str(" │\n");

    // Separator: ╞══════╪══════╡
    buf.push('╞');
    for (j, w) in widths.iter().enumerate() {
        if j > 0 {
            buf.push('╪');
        }
        for _ in 0..(w + 2) {
            buf.push('═');
        }
    }
    buf.push_str("╡\n");

    // Data rows (head)
    for row in &formatted[..show_head] {
        buf.push('│');
        for (j, w) in widths.iter().enumerate() {
            if j > 0 {
                buf.push_str(" ┆");
            }
            let cell = row.get(j).map(|s| s.as_str()).unwrap_or("");
            buf.push_str(&format!(" {:width$}", cell, width = *w));
        }
        buf.push_str(" │\n");
    }

    // Truncation row: │ …    ┆ …    │
    if truncated {
        buf.push('│');
        for (j, w) in widths.iter().enumerate() {
            if j > 0 {
                buf.push_str(" ┆");
            }
            buf.push_str(&format!(" {:width$}", "…", width = *w));
        }
        buf.push_str(" │\n");

        // Tail rows
        for row in &formatted[show_head..] {
            buf.push('│');
            for (j, w) in widths.iter().enumerate() {
                if j > 0 {
                    buf.push_str(" ┆");
                }
                let cell = row.get(j).map(|s| s.as_str()).unwrap_or("");
                buf.push_str(&format!(" {:width$}", cell, width = *w));
            }
            buf.push_str(" │\n");
        }
    }

    // Bottom border: └──────┴──────┘
    buf.push('└');
    for (j, w) in widths.iter().enumerate() {
        if j > 0 {
            buf.push('┴');
        }
        for _ in 0..(w + 2) {
            buf.push('─');
        }
    }
    buf.push_str("┘\n");

    buf
}

/// Truncate a string in the middle if it exceeds `max_len`, keeping both ends visible.
fn truncate_middle(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        return s.to_string();
    }
    let keep = (max_len - 5) / 2; // 5 chars for " ... "
    format!("{} ... {}", &s[..keep], &s[s.len() - keep..])
}
