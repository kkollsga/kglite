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

    pub fn values_mut(&mut self) -> impl Iterator<Item = &mut V> {
        self.entries.iter_mut().map(|(_, value)| value)
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

    /// Keep only the entries `carried` names, re-keyed to the name each is
    /// carried under. `carried` is `(output name, source name)`.
    ///
    /// This is a `WITH` projection's effect on identity bindings: the clause
    /// is a scope barrier, so a name it does not project stops being bound,
    /// and `WITH n AS m` moves the binding to `m` rather than copying it.
    /// A source name absent from the map (an `OPTIONAL MATCH` miss) stays
    /// absent — it is not carried as a null.
    pub fn restrict_renamed(&mut self, carried: &[(String, String)])
    where
        V: Clone,
    {
        if carried.is_empty() {
            self.entries.clear();
            return;
        }
        // The common shape — every binding projected under its own name, in
        // order — needs no rebuild at all. Anything else falls through.
        if self.entries.len() == carried.len()
            && self
                .entries
                .iter()
                .zip(carried)
                .all(|((key, _), (out, src))| key == src && src == out)
        {
            return;
        }
        let mut kept = Vec::with_capacity(carried.len().min(self.entries.len()));
        for (out, src) in carried {
            if let Some(value) = self.get(src) {
                kept.push((out.clone(), value.clone()));
            }
        }
        self.entries = kept;
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
    pub incarnation: Option<crate::datatypes::values::RelationshipIncarnation>,
}

/// A walked sequence of hops: a path variable's value, or the segment a
/// variable-length relationship variable (`[r*1..3]`) walked.
#[derive(Debug, Clone)]
pub struct PathBinding {
    /// True for a variable-length relationship variable's segment, which
    /// reads as the *list* of its relationships (openCypher), not as a path.
    /// The list is built only when the variable is read, so a named segment
    /// nothing reads costs no more than an anonymous one.
    pub relationship_list: bool,
    pub source: NodeIndex,
    pub hops: usize,
    pub path: Vec<PathHop>,
    /// This statement's relationship identity token for each hop, captured
    /// when the hop was bound — the path's counterpart to
    /// [`EdgeBinding::incarnation`], and load-bearing for the same reason: a
    /// hop names a storage *slot*, and a `DELETE` + `CREATE` inside one write
    /// statement can hand that slot to a different relationship. Reading the
    /// slot's token at *use* time silently re-pointed the path at the
    /// replacement; the bind-time token makes the retired hop read as stale.
    ///
    /// `None` for a path built without identity tracking — every read
    /// statement, where no slot can be retired mid-statement — so the read
    /// path allocates nothing per path binding. Per-hop `None` inside a
    /// tracked path means "bound by a clause that had no token yet";
    /// `mutation_support::stamp_relationships` fills those at clause end,
    /// exactly as it stamps edge bindings created by CREATE/MERGE.
    pub hop_incarnations: Option<Vec<Option<crate::datatypes::values::RelationshipIncarnation>>>,
}

impl PathBinding {
    /// The bind-time token for hop `index`, or `None` when this path carries
    /// no identity tracking.
    pub fn hop_incarnation(
        &self,
        index: usize,
    ) -> Option<crate::datatypes::values::RelationshipIncarnation> {
        self.hop_incarnations
            .as_ref()
            .and_then(|tokens| tokens.get(index).copied().flatten())
    }
}

impl ResultRow {
    // Keep the established constructor-only result-row API stable in this hardening pass.
    #[allow(clippy::new_without_default)]
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
    /// Indexes installed by `CREATE INDEX`. Counts KGLite index *structures*,
    /// so `CREATE RANGE INDEX` reports 2 — a hash equality index plus a B-tree
    /// range index, which together provide what Neo4j's single RANGE index
    /// does. Named after Neo4j's `indexesAdded` summary counter.
    pub indexes_added: usize,
    /// Indexes removed by `DROP INDEX`, counted the same way.
    pub indexes_removed: usize,
    /// Constraints declared by `CREATE CONSTRAINT`. Counts *constraints*, not
    /// the structures behind them, so `IS NODE KEY` reports 1 even though
    /// KGLite serves it as uniqueness plus presence. Named after Neo4j's
    /// `constraintsAdded` summary counter.
    pub constraints_added: usize,
    /// Constraints withdrawn by `DROP CONSTRAINT`, counted the same way.
    pub constraints_removed: usize,
}

/// Lightweight diagnostics attached to every `CypherResult` — gives
/// agents the signal they need to iterate without relying on PROFILE.
///
/// Populated unconditionally. Timing uses one clock pair; retrieval evidence
/// is collected per operator, without synchronization on scalar scoring rows.
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct QueryDiagnostics {
    /// Distinct vector retrieval routes actually executed, including nested queries.
    /// Empty for EXPLAIN, empty inputs and queries without a vector retrieval operator.
    pub retrieval: Vec<RetrievalDiagnostics>,
    /// Wall-clock time spent executing the query (parse + plan + execute).
    pub elapsed_ms: u64,
    /// Deadline that was in effect for this query, in milliseconds.
    /// `None` when no deadline applied.
    pub timeout_ms: Option<u64>,
    /// Non-fatal advisory warnings about this query — e.g. a MATCH that
    /// references an unknown node label or relationship type (almost always a
    /// typo), with a "did you mean?" hint. Empty for a clean query. Lets
    /// programmatic / agent callers see the same signal interactive users get
    /// on stderr.
    ///
    /// A `row_limit` truncation adds one here too, so a caller that reads only
    /// the warnings still learns its result was cut.
    pub warnings: Vec<String>,
    /// The `row_limit` retention cap that was in force, echoed back. `None`
    /// when the caller set no cap.
    ///
    /// Set whether or not the cap bit, so a caller can distinguish "no cap
    /// applied" from "capped, and the result fitted".
    pub row_limit: Option<usize>,
    /// Rows the query produced *before* `row_limit` truncated it.
    ///
    /// `Some` **only** when truncation actually happened — so
    /// `total_rows.is_some()` is the truncation signal, and the count is the
    /// exact pre-truncation total on every execution path (eager rows, the
    /// lazy descriptor, and a mutation's RETURN), never an estimate or a
    /// lower bound. `None` both when no cap applied and when the result fitted
    /// inside one.
    pub total_rows: Option<u64>,
    /// How a statement under `FOR VALID_TIME AS OF` (or a binding's
    /// `valid_at=`) was answered. `None`, and left out of the serialized form,
    /// for a statement without a context.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temporal: Option<Box<TemporalDiagnostics>>,
}

/// The valid-time echo on [`QueryDiagnostics::temporal`]: where the context
/// came from, the instant it resolved to, what it filtered and by which
/// route.
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize)]
pub struct TemporalDiagnostics {
    /// The context's axis, `VALID_TIME`.
    pub axis: String,
    /// Where the context came from: `default` (the graph declares validity
    /// and the statement wrote no context, so it ran as of today in UTC),
    /// `explicit` (`FOR VALID_TIME AS OF` or a binding's `valid_at=`), `all`
    /// (`FOR VALID_TIME ALL`), or `skipped:<reason>` when the default did not
    /// apply and the statement read every version: `write`, `procedure` (not
    /// valid-time aware), or `valid_at` (the statement names its own
    /// instants).
    pub source: String,
    /// The instant this execution resolved, ISO 8601: `2009-06-30` for a
    /// date, `2009-06-30T12:00:00` (naive UTC) for a datetime; `all` when the
    /// statement reads every version.
    pub instant: String,
    /// The declared labels the filter judges for the statement's patterns,
    /// including labels widened in through secondary labels (a `(:A)`
    /// pattern lists `(:B)` when nodes it may reach can carry a declared
    /// `B`), merged across every scope and without bounds (EXPLAIN shows only
    /// the top scope, with bounds): `(:Well)`, `[:HOLDER]`,
    /// `[:HOLDER from :Project]`.
    pub targets: Vec<String>,
    /// Per target in `targets`, how many of the rows it governs the context
    /// hides at the instant, judged by that target's own bounds: `(:Project)`
    /// counts the projects not valid then, `[:FUNDED_BY from :Project]` the
    /// relationships not valid then. A target answered by property guards
    /// (Disk mode, an unreadable bound, the index byte cap) has no entry.
    pub hidden: std::collections::BTreeMap<String, usize>,
    /// Relationships that are valid by their own bounds but hidden because
    /// an endpoint node is not valid at the instant. These are in no
    /// `hidden` entry: a funding period that starts before its project is
    /// approved shows up here. `None` when a target of the statement is
    /// answered by property guards, so the nodes' validity cannot be counted.
    pub endpoint_invalid: Option<usize>,
    /// `guarded`: the statement ran under the filter (EXPLAIN renders that
    /// plan). `plain`: nothing was filtered — every declared target was
    /// timeless at the instant, the statement reaches none, or it reads every
    /// version — so it ran its plain plan.
    /// `view`: it ran through a valid-time view (`freeze(valid_at=)`).
    pub route: String,
    /// `exact_mask` when a vector retrieval scored the valid vectors exactly
    /// (the retrieval record's `fallback_reason` says why), `hnsw_mask` when
    /// a filtered search of its HNSW index returned the nearest valid ones,
    /// `quantized_mask` when the quantized copy's best valid candidates were
    /// re-scored exactly; `None` without a vector retrieval.
    pub retrieval: Option<String>,
    /// Whether a graph algorithm ran on the valid slice.
    pub slice: bool,
    /// The graph version the answer was computed at. Session-scoped: it
    /// restarts when a graph is loaded, so it compares answers within one
    /// process, not across saves.
    pub session_version: u64,
}

/// One executed vector retrieval route. Repeated identical routes are coalesced;
/// this is execution metadata, not a per-row or per-call counter.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct RetrievalDiagnostics {
    /// `auto`, `exact`, or `per_row` when options remain evaluated per row.
    pub requested_policy: String,
    /// `hnsw` for approximate candidate selection, `quantized` when a
    /// quantized copy narrowed the candidates before an exact re-score,
    /// otherwise `exact`; under a `FOR VALID_TIME AS OF` context, `hnsw_mask`
    /// when a filtered search of the index returned the valid vectors nearest
    /// the query, `quantized_mask` for the copy's valid candidates.
    pub actual_mode: String,
    /// Why an exact route ran (`exact_mask`: the valid vectors, scored
    /// exactly; `exact_mask_visit_limit`: the same, after the filtered search
    /// passed its step budget; `exact_mask_unreached`: the same, after it
    /// reached fewer valid vectors than `k` or than exist, whichever is
    /// smaller); absent when HNSW served the result.
    pub fallback_reason: Option<String>,
    /// `Type.embedding_property` when a single store was established; a
    /// comma-separated list, in type order, when a relationship top-k merged
    /// several stores (`relationship:A.p_emb,relationship:B.p_emb`).
    pub store: Option<String>,
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

    // KgError carries provenance details; boxing it would only burden an error path.
    #[allow(clippy::result_large_err)]
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
// KgError deliberately carries structured context; boxing it would change the public result type.
#[allow(clippy::result_large_err)]
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
    let mut row = materialise_lazy_row_inner(pending_row, &descriptor.return_items, graph);
    clear_published_row(&mut row);
    Ok(row)
}

fn materialise_lazy_row_inner(
    pending_row: &ResultRow,
    return_items: &[super::ast::ReturnItem],
    graph: &crate::graph::dir_graph::DirGraph,
) -> Vec<Value> {
    let mut row: Vec<Value> = return_items
        .iter()
        .map(|item| match &item.expression {
            super::ast::Expression::PropertyAccess { variable, property } => {
                if let Some(&node_idx) = pending_row.node_bindings.get(variable) {
                    graph
                        .graph
                        .node_view(node_idx)
                        .map(|node| {
                            super::executor::helpers::resolve_node_property(&node, property, graph)
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
        .collect();
    crate::graph::session::resolve_noderefs(&graph.graph, std::slice::from_mut(&mut row));
    row
}

/// Materialise every row in a lazy descriptor against `graph`.
///
/// C-ABI and server consumers can call this before serialising an executor
/// result whose [`CypherResult::lazy`] field is populated.
// KgError deliberately carries structured context; boxing it would change the public result type.
#[allow(clippy::result_large_err)]
pub fn materialise_lazy(
    descriptor: &LazyResultDescriptor,
    graph: &crate::graph::dir_graph::DirGraph,
) -> Result<Vec<Vec<Value>>, crate::error::KgError> {
    materialise_lazy_range(descriptor, graph, 0..descriptor.len())
}

/// Materialise a contiguous range of pending lazy-result rows.
///
/// Provenance and bounds are checked once for the whole batch. Disk arena
/// lifetime is still scoped independently to each row, so large sequential
/// reads retain the bounded-memory guarantee of [`materialise_lazy_row`].
// KgError deliberately carries structured context; boxing it would change the public result type.
#[allow(clippy::result_large_err)]
pub fn materialise_lazy_range(
    descriptor: &LazyResultDescriptor,
    graph: &crate::graph::dir_graph::DirGraph,
    range: std::ops::Range<usize>,
) -> Result<Vec<Vec<Value>>, crate::error::KgError> {
    descriptor.validate_graph(graph)?;
    if range.start > range.end || range.end > descriptor.len() {
        return Err(crate::error::KgError::InvalidArgument {
            argument: "range".to_string(),
            expected: format!("a range within 0..{}", descriptor.len()),
            found: format!("{}..{}", range.start, range.end),
        });
    }
    let mut rows = Vec::with_capacity(range.len());
    for pending_row in &descriptor.pending_rows[range] {
        // One guard per row bounds sequential disk-arena growth. Each cell is
        // converted to an owned Value before the guard drops.
        let _arena_guard = graph.graph.begin_query();
        let mut row = materialise_lazy_row_inner(pending_row, &descriptor.return_items, graph);
        clear_published_row(&mut row);
        rows.push(row);
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

/// Write a CSV field, quoting if it contains comma, quote, CR, or LF.
fn csv_field(buf: &mut String, s: &str) {
    if s.contains(',') || s.contains('"') || s.contains('\n') || s.contains('\r') {
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
    csv_field(buf, &crate::param::kglite_value_to_csv_text(val));
}

#[cfg(test)]
mod lazy_materialisation_tests {
    use super::*;
    use crate::graph::session::{execute_mut, execute_read, ExecuteOptions};
    use crate::graph::storage::disk::temp_owner::TempGraphDir;
    use crate::graph::storage::mode::{new_dir_graph_in_mode, StorageMode};
    use std::collections::HashMap;
    use std::sync::{Arc, Barrier};

    /// Fixture graph plus the temp directory it maps into (disk mode only).
    ///
    /// Field order is the contract: `graph` is declared first, so it drops
    /// first, and `temp`'s guard then asserts that no handle survived. The
    /// previous `(DirGraph, Option<TempDir>)` tuple had the opposite drop
    /// order — bindings in a `let (graph, _temp) = …` pattern drop
    /// right-to-left, so the directory was destroyed while the disk backend
    /// still held mmaps into it. On Unix that unlink is silent, so nothing
    /// failed; `TempGraphDir` makes the ordering itself the assertion.
    struct GraphFixture {
        graph: Arc<crate::graph::dir_graph::DirGraph>,
        /// Held only for its `Drop`: it asserts the graph above is gone.
        _temp: Option<TempGraphDir>,
    }

    impl GraphFixture {
        fn graph(&self) -> &crate::graph::dir_graph::DirGraph {
            &self.graph
        }

        /// A counted handle for the concurrency test. `watch_arc` probes the
        /// `Arc`'s strong count, so a clone outliving the directory — leaked
        /// into a thread, or held by a stray binding — fails the test.
        fn shared(&self) -> Arc<crate::graph::dir_graph::DirGraph> {
            Arc::clone(&self.graph)
        }

        fn graph_mut(&mut self) -> &mut crate::graph::dir_graph::DirGraph {
            Arc::get_mut(&mut self.graph).expect("fixture graph has outstanding clones")
        }
    }

    fn graph_in_mode(mode: StorageMode) -> GraphFixture {
        let temp = (mode == StorageMode::Disk).then(TempGraphDir::new);
        let mut graph = new_dir_graph_in_mode(mode, temp.as_ref().map(|dir| dir.path())).unwrap();
        let params = HashMap::new();
        execute_mut(
            &mut graph,
            "CREATE (:Person {id:1, title:'Alice', name:'Alice', age:30}), \
             (:Person {id:2, title:'Bob', name:'Bob', age:25})",
            &ExecuteOptions::eager(&params),
        )
        .expect("fixture CREATE");
        let graph = Arc::new(graph);
        if let Some(dir) = temp.as_ref() {
            dir.watch_arc("disk-mode DirGraph", &graph);
        }
        GraphFixture { graph, _temp: temp }
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
            let fixture = graph_in_mode(mode);
            let graph = fixture.graph();
            let params = HashMap::new();
            let eager = execute_read(
                graph,
                "MATCH (n:Person) RETURN n.name, n.age",
                &ExecuteOptions::eager(&params),
            )
            .expect("eager read")
            .result
            .rows;
            let descriptor = execute_lazy(graph);
            assert_eq!(materialise_lazy(&descriptor, graph).unwrap(), eager);
        }
    }

    #[test]
    fn invalid_index_wrong_graph_and_stale_snapshot_are_typed_errors() {
        let mut fixture = graph_in_mode(StorageMode::Memory);
        let descriptor = execute_lazy(fixture.graph());
        assert!(matches!(
            materialise_lazy_row(&descriptor, fixture.graph(), descriptor.len()),
            Err(crate::error::KgError::InvalidArgument { ref argument, .. }) if argument == "index"
        ));

        let other = graph_in_mode(StorageMode::Memory);
        assert!(matches!(
            materialise_lazy(&descriptor, other.graph()),
            Err(crate::error::KgError::InvalidArgument { ref argument, .. }) if argument == "graph"
        ));

        fixture.graph_mut().bump_version();
        assert!(matches!(
            materialise_lazy(&descriptor, fixture.graph()),
            Err(crate::error::KgError::InvalidArgument { ref argument, .. }) if argument == "graph"
        ));
    }

    #[test]
    fn lazy_range_materialises_subset_and_rejects_invalid_bounds() {
        let fixture = graph_in_mode(StorageMode::Memory);
        let graph = fixture.graph();
        let descriptor = execute_lazy(graph);
        assert_eq!(
            materialise_lazy_range(&descriptor, graph, 1..2).unwrap(),
            vec![vec![Value::String("Bob".into()), Value::Int64(25)]]
        );
        assert!(matches!(
            materialise_lazy_range(&descriptor, graph, 0..3),
            Err(crate::error::KgError::InvalidArgument { ref argument, .. }) if argument == "range"
        ));
    }

    #[test]
    fn disk_lazy_access_survives_executor_drop_and_intervening_query() {
        let fixture = graph_in_mode(StorageMode::Disk);
        let graph = fixture.graph();
        let descriptor = execute_lazy(graph);
        let params = HashMap::new();
        execute_read(
            graph,
            "MATCH (n:Person) RETURN count(n)",
            &ExecuteOptions::eager(&params),
        )
        .expect("intervening query");
        assert_eq!(
            materialise_lazy_row(&descriptor, graph, 0).unwrap(),
            vec![Value::String("Alice".into()), Value::Int64(30)]
        );
    }

    #[test]
    fn concurrent_disk_lazy_reads_remain_exact() {
        let fixture = graph_in_mode(StorageMode::Disk);
        // Counted handle, not an incidental re-bind: `watch_arc` fails the
        // test if any clone (including one left in a thread) outlives the
        // temp directory, so this test's safety no longer rests on the
        // accident of a later `let` shadowing the fixture binding.
        let graph = fixture.shared();
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

#[cfg(test)]
mod fractional_timestamp_contract_tests {
    use super::*;
    fn stamp(text: &str) -> Value {
        Value::Timestamp(
            chrono::NaiveDateTime::parse_from_str(text, "%Y-%m-%dT%H:%M:%S%.f").unwrap(),
        )
    }
    #[test]
    fn fractional_timestamp_text_keeps_exact_value_and_whole_second_spelling() {
        for text in [
            "2025-01-02T03:04:05",
            "2025-01-02T03:04:05.123456789",
            "1969-12-31T23:59:59.500",
        ] {
            let mut actual = String::new();
            csv_value(&mut actual, &stamp(text));
            assert_eq!(actual, text);
        }
    }
}

#[cfg(test)]
mod nested_timestamp_csv_tests {
    use super::*;
    #[test]
    fn nested_csv_retains_fraction_and_existing_field_escaping() {
        let stamp = Value::Timestamp(
            chrono::NaiveDateTime::parse_from_str(
                "2025-01-02T03:04:05.123456789",
                "%Y-%m-%dT%H:%M:%S%.f",
            )
            .unwrap(),
        );
        let mut actual = String::new();
        csv_value(&mut actual, &Value::List(vec![stamp]));
        assert_eq!(actual, "\"[\"\"2025-01-02T03:04:05.123456789\"\"]\"");
    }

    #[test]
    fn nested_csv_retains_float_precision() {
        let mut actual = String::new();
        csv_value(
            &mut actual,
            &Value::List(vec![Value::Float64(1.234_567_89)]),
        );
        assert_eq!(actual, "[1.23456789]");
    }
}

pub(crate) fn clear_published_relationship_incarnations(result: &mut CypherResult) {
    for row in &mut result.rows {
        clear_published_row(row);
    }
}

fn clear_published_row(row: &mut [Value]) {
    for value in row {
        value.clear_relationship_incarnations();
    }
}

#[cfg(test)]
mod diagnostics_size_tests {
    use super::*;

    /// Every statement moves its diagnostics through the session: the
    /// valid-time echo is boxed so a statement without a context carries one
    /// pointer for it, not the whole echo.
    #[test]
    fn the_temporal_echo_costs_one_pointer() {
        assert!(std::mem::size_of::<QueryDiagnostics>() <= 112);
    }
}
