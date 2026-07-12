// src/graph/cypher/result.rs
// Result types for the Cypher query pipeline

use crate::datatypes::values::Value;
use petgraph::graph::{EdgeIndex, NodeIndex};

use crate::graph::core::pattern_matching::PathHop;
use crate::graph::storage::GraphRead;
use std::collections::HashMap;

// ============================================================================
// Bindings — compact ordered map for small variable counts
// ============================================================================

/// Compact ordered map using `Vec<(String, V)>` with linear search.
/// Faster than HashMap for typical Cypher variable counts (1-8 entries):
/// no hashing overhead, cache-friendly sequential access, cheaper clone
/// (one Vec allocation vs HashMap bucket array + entries).
#[derive(Debug, Clone, Default)]
pub struct Bindings<V> {
    entries: Vec<(String, V)>,
}

impl<V> Bindings<V> {
    pub fn new() -> Self {
        Bindings {
            entries: Vec::new(),
        }
    }

    pub fn with_capacity(cap: usize) -> Self {
        Bindings {
            entries: Vec::with_capacity(cap),
        }
    }

    pub fn get(&self, key: &str) -> Option<&V> {
        self.entries.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }

    pub fn get_mut(&mut self, key: &str) -> Option<&mut V> {
        self.entries
            .iter_mut()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v)
    }

    /// Upsert: update if key exists, push if not.
    pub fn insert(&mut self, key: String, val: V) {
        if let Some(entry) = self.entries.iter_mut().find(|(k, _)| *k == key) {
            entry.1 = val;
        } else {
            self.entries.push((key, val));
        }
    }

    pub fn contains_key(&self, key: &str) -> bool {
        self.entries.iter().any(|(k, _)| k == key)
    }

    pub fn keys(&self) -> impl Iterator<Item = &String> {
        self.entries.iter().map(|(k, _)| k)
    }

    pub fn iter(&self) -> impl Iterator<Item = (&String, &V)> {
        self.entries.iter().map(|(k, v)| (k, v))
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Remove a key and return its value (move, no clone).
    pub fn remove(&mut self, key: &str) -> Option<V> {
        if let Some(pos) = self.entries.iter().position(|(k, _)| k == key) {
            Some(self.entries.swap_remove(pos).1)
        } else {
            None
        }
    }

    /// Convert to HashMap for interop with pattern_matching pre_bindings.
    pub fn to_hashmap(&self) -> HashMap<String, V>
    where
        V: Clone,
    {
        self.entries
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect()
    }
}

impl<V> IntoIterator for Bindings<V> {
    type Item = (String, V);
    type IntoIter = std::vec::IntoIter<(String, V)>;
    fn into_iter(self) -> Self::IntoIter {
        self.entries.into_iter()
    }
}

impl<'a, V> IntoIterator for &'a Bindings<V> {
    type Item = &'a (String, V);
    type IntoIter = std::slice::Iter<'a, (String, V)>;
    fn into_iter(self) -> Self::IntoIter {
        self.entries.iter()
    }
}

// ============================================================================
// Pipeline Result Types
// ============================================================================

/// A single row in the pipeline result set.
/// During execution, rows carry lightweight NodeIndex/EdgeIndex references.
/// Properties are resolved on-demand from the graph.
#[derive(Debug, Clone)]
pub struct ResultRow {
    /// Node variable bindings: variable_name -> NodeIndex
    pub node_bindings: Bindings<NodeIndex>,
    /// Edge variable bindings: variable_name -> EdgeBinding (source, target, edge_index)
    pub edge_bindings: Bindings<EdgeBinding>,
    /// Variable-length path bindings
    pub path_bindings: Bindings<PathBinding>,
    /// Projected values from WITH/RETURN
    pub projected: Bindings<Value>,
}

/// Lightweight edge binding — stores only indices, no cloned data.
/// Edge properties are resolved on-demand from the graph via edge_index.
#[derive(Debug, Clone, Copy)]
pub struct EdgeBinding {
    pub source: NodeIndex,
    pub target: NodeIndex,
    pub edge_index: EdgeIndex,
}

/// Variable-length path binding
#[derive(Debug, Clone)]
pub struct PathBinding {
    pub source: NodeIndex,
    pub hops: usize,
    pub path: Vec<PathHop>,
}

impl ResultRow {
    pub fn new() -> Self {
        ResultRow {
            node_bindings: Bindings::new(),
            edge_bindings: Bindings::new(),
            path_bindings: Bindings::new(),
            projected: Bindings::new(),
        }
    }

    /// Pre-sized constructor to avoid reallocation.
    pub fn with_capacity(nodes: usize, edges: usize, projected: usize) -> Self {
        ResultRow {
            node_bindings: Bindings::with_capacity(nodes),
            edge_bindings: Bindings::with_capacity(edges),
            path_bindings: Bindings::new(),
            projected: Bindings::with_capacity(projected),
        }
    }

    /// Create a row with only projected values (for aggregation results)
    pub fn from_projected(projected: Bindings<Value>) -> Self {
        ResultRow {
            node_bindings: Bindings::new(),
            edge_bindings: Bindings::new(),
            path_bindings: Bindings::new(),
            projected,
        }
    }
}

/// The result set flowing through the pipeline
#[derive(Debug)]
pub struct ResultSet {
    pub rows: Vec<ResultRow>,
    /// Column names in output order (populated by RETURN)
    pub columns: Vec<String>,
    /// Set when the executor's RETURN clause skipped per-row projection
    /// (because the planner flagged it `lazy_eligible`). `finalize_result`
    /// reads this to emit a lazy `CypherResult` instead of materialising
    /// every cell. Cleared on any clause that consumes row values.
    pub lazy_return_items: Option<Vec<super::ast::ReturnItem>>,
}

impl ResultSet {
    pub fn new() -> Self {
        ResultSet {
            rows: Vec::new(),
            columns: Vec::new(),
            lazy_return_items: None,
        }
    }
}

// ============================================================================
// Final Output
// ============================================================================

/// Per-clause execution statistics collected during PROFILE mode.
#[derive(Debug, Clone)]
pub struct ClauseStats {
    pub clause_name: String,
    pub rows_in: usize,
    pub rows_out: usize,
    pub elapsed_us: u64,
}

/// Mutation statistics returned from CREATE/SET/DELETE queries
#[derive(Debug, Clone, Default)]
pub struct MutationStats {
    pub nodes_created: usize,
    pub relationships_created: usize,
    pub properties_set: usize,
    pub nodes_deleted: usize,
    pub relationships_deleted: usize,
    pub properties_removed: usize,
}

/// Lightweight diagnostics attached to every `CypherResult` — gives
/// agents the signal they need to iterate without relying on PROFILE.
///
/// Populated unconditionally. The cost is a single `Instant::now()` call
/// pair and a handful of counter bumps, so always-on.
#[derive(Debug, Clone, Default)]
pub struct QueryDiagnostics {
    /// Wall-clock time spent executing the query (parse + plan + execute).
    pub elapsed_ms: u64,
    /// True when the deadline fired during execution. When set, the
    /// result rows are the partial set materialised before cancellation.
    pub timed_out: bool,
    /// Deadline that was in effect for this query, in milliseconds.
    /// `None` when no deadline applied.
    pub timeout_ms: Option<u64>,
    /// Non-fatal advisory warnings about this query — e.g. a MATCH that
    /// references an unknown node label or relationship type (almost always a
    /// typo), with a "did you mean?" hint. Empty for a clean query. Lets
    /// programmatic / agent callers see the same signal interactive users get
    /// on stderr.
    pub warnings: Vec<String>,
}

/// Side-channel lazy-evaluation descriptor. When set on a `CypherResult`,
/// `rows` is empty and the receiver should resolve cells on demand from
/// `pending_rows` + `return_items` against the graph (held by ResultView's
/// graph reference). Set only when the planner has flagged the terminal
/// RETURN as `lazy_eligible` and Python's downstream consumer supports
/// lazy materialisation (the standard ResultView path does).
#[derive(Debug)]
pub struct LazyResultDescriptor {
    pending_rows: Vec<ResultRow>,
    return_items: Vec<super::ast::ReturnItem>,
    graph_id: u64,
    graph_version: u64,
}

impl LazyResultDescriptor {
    pub(crate) fn new(
        pending_rows: Vec<ResultRow>,
        return_items: Vec<super::ast::ReturnItem>,
        graph: &crate::graph::dir_graph::DirGraph,
    ) -> Self {
        Self {
            pending_rows,
            return_items,
            graph_id: graph.graph_id(),
            graph_version: graph.version(),
        }
    }

    /// Number of unresolved rows held by this descriptor.
    pub fn len(&self) -> usize {
        self.pending_rows.len()
    }

    /// Whether this descriptor contains no unresolved rows.
    pub fn is_empty(&self) -> bool {
        self.pending_rows.is_empty()
    }

    fn validate_graph(
        &self,
        graph: &crate::graph::dir_graph::DirGraph,
    ) -> Result<(), crate::error::KgError> {
        if graph.graph_id() != self.graph_id {
            return Err(crate::error::KgError::InvalidArgument {
                argument: "graph".to_string(),
                expected: format!("graph id {}", self.graph_id),
                found: format!("graph id {}", graph.graph_id()),
            });
        }
        if graph.version() != self.graph_version {
            return Err(crate::error::KgError::InvalidArgument {
                argument: "graph".to_string(),
                expected: format!("snapshot version {}", self.graph_version),
                found: format!("snapshot version {}", graph.version()),
            });
        }
        Ok(())
    }
}

/// Materialise one pending lazy-result row against `graph`.
///
/// Returns engine [`Value`]s; native/wire conversion remains a binding concern.
pub fn materialise_lazy_row(
    descriptor: &LazyResultDescriptor,
    graph: &crate::graph::dir_graph::DirGraph,
    index: usize,
) -> Result<Vec<Value>, crate::error::KgError> {
    descriptor.validate_graph(graph)?;
    let pending_row = descriptor.pending_rows.get(index).ok_or_else(|| {
        crate::error::KgError::InvalidArgument {
            argument: "index".to_string(),
            expected: format!("an index below {}", descriptor.len()),
            found: index.to_string(),
        }
    })?;
    // Disk node/edge materialization returns arena-backed references. Keep
    // the arena generation alive until every cell in this row is owned.
    let _arena_guard = graph.graph.begin_query();
    Ok(materialise_lazy_row_inner(
        pending_row,
        &descriptor.return_items,
        graph,
    ))
}

fn materialise_lazy_row_inner(
    pending_row: &ResultRow,
    return_items: &[super::ast::ReturnItem],
    graph: &crate::graph::dir_graph::DirGraph,
) -> Vec<Value> {
    return_items
        .iter()
        .map(|item| match &item.expression {
            super::ast::Expression::PropertyAccess { variable, property } => {
                if let Some(&node_idx) = pending_row.node_bindings.get(variable) {
                    graph
                        .graph
                        .node_weight(node_idx)
                        .map(|node| {
                            super::executor::helpers::resolve_node_property(node, property, graph)
                        })
                        .unwrap_or(Value::Null)
                } else if let Some(edge) = pending_row.edge_bindings.get(variable) {
                    super::executor::helpers::resolve_edge_property(graph, edge, property)
                } else {
                    pending_row
                        .projected
                        .get(variable)
                        .cloned()
                        .unwrap_or(Value::Null)
                }
            }
            super::ast::Expression::Variable(variable) => {
                if let Some(&node_idx) = pending_row.node_bindings.get(variable) {
                    // Whole-node lazy projection is currently planner-ineligible;
                    // retain the wrapper's defensive id fallback.
                    graph.graph.get_node_id(node_idx).unwrap_or(Value::Null)
                } else {
                    pending_row
                        .projected
                        .get(variable)
                        .cloned()
                        .unwrap_or(Value::Null)
                }
            }
            _ => Value::Null,
        })
        .collect()
}

/// Materialise every row in a lazy descriptor against `graph`.
///
/// C-ABI and server consumers can call this before serialising an executor
/// result whose [`CypherResult::lazy`] field is populated.
pub fn materialise_lazy(
    descriptor: &LazyResultDescriptor,
    graph: &crate::graph::dir_graph::DirGraph,
) -> Result<Vec<Vec<Value>>, crate::error::KgError> {
    descriptor.validate_graph(graph)?;
    let mut rows = Vec::with_capacity(descriptor.len());
    for pending_row in &descriptor.pending_rows {
        // One guard per row bounds sequential disk-arena growth. Each cell is
        // converted to an owned Value before the guard drops.
        let _arena_guard = graph.graph.begin_query();
        rows.push(materialise_lazy_row_inner(
            pending_row,
            &descriptor.return_items,
            graph,
        ));
    }
    Ok(rows)
}

/// Final query result returned to Python
#[derive(Debug)]
pub struct CypherResult {
    pub columns: Vec<String>,
    pub rows: Vec<Vec<Value>>,
    pub stats: Option<MutationStats>,
    pub profile: Option<Vec<ClauseStats>>,
    pub diagnostics: Option<QueryDiagnostics>,
    /// Set when the receiver should evaluate row cells lazily; in that
    /// case `rows` is empty.
    pub lazy: Option<LazyResultDescriptor>,
}

impl CypherResult {
    pub fn empty() -> Self {
        CypherResult {
            columns: Vec::new(),
            rows: Vec::new(),
            stats: None,
            profile: None,
            diagnostics: None,
            lazy: None,
        }
    }

    /// Serialize the result as a CSV string.
    pub fn to_csv(&self) -> String {
        let mut buf = String::new();
        // Header
        for (i, col) in self.columns.iter().enumerate() {
            if i > 0 {
                buf.push(',');
            }
            csv_field(&mut buf, col);
        }
        buf.push('\n');
        // Rows
        for row in &self.rows {
            for (i, val) in row.iter().enumerate() {
                if i > 0 {
                    buf.push(',');
                }
                csv_value(&mut buf, val);
            }
            buf.push('\n');
        }
        buf
    }
}

/// Write a CSV field, quoting if it contains comma, quote, or newline.
fn csv_field(buf: &mut String, s: &str) {
    if s.contains(',') || s.contains('"') || s.contains('\n') {
        buf.push('"');
        for c in s.chars() {
            if c == '"' {
                buf.push('"');
            }
            buf.push(c);
        }
        buf.push('"');
    } else {
        buf.push_str(s);
    }
}

/// Write a Value as a CSV field.
fn csv_value(buf: &mut String, val: &Value) {
    match val {
        Value::Null => {} // empty cell
        Value::String(s) => csv_field(buf, s),
        Value::Int64(n) => {
            use std::fmt::Write;
            let _ = write!(buf, "{}", n);
        }
        Value::Float64(f) => {
            use std::fmt::Write;
            let _ = write!(buf, "{}", f);
        }
        Value::Boolean(b) => buf.push_str(if *b { "true" } else { "false" }),
        Value::UniqueId(u) => {
            use std::fmt::Write;
            let _ = write!(buf, "{}", u);
        }
        Value::DateTime(d) => buf.push_str(&d.format("%Y-%m-%d").to_string()),
        Value::Timestamp(d) => buf.push_str(&d.format("%Y-%m-%dT%H:%M:%S").to_string()),
        Value::Point { lat, lon } => {
            use std::fmt::Write;
            let _ = write!(buf, "POINT({} {})", lon, lat);
        }
        Value::Duration {
            months,
            days,
            seconds,
        } => {
            use std::fmt::Write;
            let _ = write!(buf, "duration(M={}, D={}, S={})", months, days, seconds);
        }
        Value::NodeRef(idx) => {
            use std::fmt::Write;
            let _ = write!(buf, "{}", idx);
        }
        // Phase A.1 — collection / graph-entity variants are CSV-
        // serialised as JSON-ish strings (delegate to format_value
        // and quote-escape via csv_field).
        Value::List(_)
        | Value::Map(_)
        | Value::Node(_)
        | Value::Relationship(_)
        | Value::Path(_) => {
            let s = crate::datatypes::values::format_value(val);
            csv_field(buf, &s);
        }
    }
}

#[cfg(test)]
mod lazy_materialisation_tests {
    use super::*;
    use crate::graph::session::{execute_mut, execute_read, ExecuteOptions};
    use crate::graph::storage::mode::{new_dir_graph_in_mode, StorageMode};
    use std::collections::HashMap;
    use std::sync::{Arc, Barrier};

    fn graph_in_mode(
        mode: StorageMode,
    ) -> (crate::graph::dir_graph::DirGraph, Option<tempfile::TempDir>) {
        let temp = (mode == StorageMode::Disk)
            .then(tempfile::tempdir)
            .transpose()
            .unwrap();
        let mut graph = new_dir_graph_in_mode(mode, temp.as_ref().map(|dir| dir.path())).unwrap();
        let params = HashMap::new();
        execute_mut(
            &mut graph,
            "CREATE (:Person {id:1, title:'Alice', name:'Alice', age:30}), \
             (:Person {id:2, title:'Bob', name:'Bob', age:25})",
            &ExecuteOptions::eager(&params),
        )
        .expect("fixture CREATE");
        (graph, temp)
    }

    fn execute_lazy(graph: &crate::graph::dir_graph::DirGraph) -> LazyResultDescriptor {
        let params = HashMap::new();
        let mut opts = ExecuteOptions::eager(&params);
        opts.lazy_eligible = true;
        let mut outcome =
            execute_read(graph, "MATCH (n:Person) RETURN n.name, n.age", &opts).expect("lazy read");
        outcome.result.lazy.take().expect("lazy descriptor")
    }

    #[test]
    fn lazy_matches_eager_across_storage_modes() {
        for mode in [StorageMode::Memory, StorageMode::Mapped, StorageMode::Disk] {
            let (graph, _temp) = graph_in_mode(mode);
            let params = HashMap::new();
            let eager = execute_read(
                &graph,
                "MATCH (n:Person) RETURN n.name, n.age",
                &ExecuteOptions::eager(&params),
            )
            .expect("eager read")
            .result
            .rows;
            let descriptor = execute_lazy(&graph);
            assert_eq!(materialise_lazy(&descriptor, &graph).unwrap(), eager);
        }
    }

    #[test]
    fn invalid_index_wrong_graph_and_stale_snapshot_are_typed_errors() {
        let (mut graph, _temp) = graph_in_mode(StorageMode::Memory);
        let descriptor = execute_lazy(&graph);
        assert!(matches!(
            materialise_lazy_row(&descriptor, &graph, descriptor.len()),
            Err(crate::error::KgError::InvalidArgument { ref argument, .. }) if argument == "index"
        ));

        let (other, _other_temp) = graph_in_mode(StorageMode::Memory);
        assert!(matches!(
            materialise_lazy(&descriptor, &other),
            Err(crate::error::KgError::InvalidArgument { ref argument, .. }) if argument == "graph"
        ));

        graph.bump_version();
        assert!(matches!(
            materialise_lazy(&descriptor, &graph),
            Err(crate::error::KgError::InvalidArgument { ref argument, .. }) if argument == "graph"
        ));
    }

    #[test]
    fn disk_lazy_access_survives_executor_drop_and_intervening_query() {
        let (graph, _temp) = graph_in_mode(StorageMode::Disk);
        let descriptor = execute_lazy(&graph);
        let params = HashMap::new();
        execute_read(
            &graph,
            "MATCH (n:Person) RETURN count(n)",
            &ExecuteOptions::eager(&params),
        )
        .expect("intervening query");
        assert_eq!(
            materialise_lazy_row(&descriptor, &graph, 0).unwrap(),
            vec![Value::String("Alice".into()), Value::Int64(30)]
        );
    }

    #[test]
    fn concurrent_disk_lazy_reads_remain_exact() {
        let (graph, _temp) = graph_in_mode(StorageMode::Disk);
        let graph = Arc::new(graph);
        let descriptor = Arc::new(execute_lazy(&graph));
        let barrier = Arc::new(Barrier::new(3));

        let materializer = {
            let graph = Arc::clone(&graph);
            let descriptor = Arc::clone(&descriptor);
            let barrier = Arc::clone(&barrier);
            std::thread::spawn(move || {
                barrier.wait();
                for _ in 0..250 {
                    assert_eq!(
                        materialise_lazy_row(&descriptor, &graph, 1).unwrap(),
                        vec![Value::String("Bob".into()), Value::Int64(25)]
                    );
                }
            })
        };
        let querier = {
            let graph = Arc::clone(&graph);
            let barrier = Arc::clone(&barrier);
            std::thread::spawn(move || {
                let params = HashMap::new();
                barrier.wait();
                for _ in 0..250 {
                    execute_read(
                        &graph,
                        "MATCH (n:Person) RETURN n.title LIMIT 1",
                        &ExecuteOptions::eager(&params),
                    )
                    .expect("concurrent query");
                }
            })
        };
        barrier.wait();
        materializer.join().expect("materializer thread");
        querier.join().expect("query thread");
    }
}
