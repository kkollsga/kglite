//! Thin pure-Rust graph handle for Rust embedders.
//!
//! Bridges `Arc<DirGraph>` (the engine) and the minimal set of
//! convenience methods that protocol-server binaries
//! (`kglite-mcp-server`, `kglite-bolt-server`) and other Rust
//! embedders need without taking on the wheel crate's full
//! Python-flavored state (selection / reports / mutation stats /
//! temporal context / default timeout / default max rows).
//!
//! This is the **Rust-side** `KnowledgeGraph`. The Python-side
//! `KnowledgeGraph` (the `#[pyclass]` wrapper backing
//! `pip install kglite`'s `import kglite`) lives in the
//! `kglite-py` crate at `crates/kglite-py/src/graph/mod.rs`. Two
//! types named `KnowledgeGraph` exist in distinct crates with
//! distinct audiences; mirrors the polars precedent
//! (`polars::DataFrame` vs `polars.DataFrame`).
//!
//! The heavy logic — `source_location` + `resolve_code_entity` —
//! lives as free functions in this module so the wheel's full
//! `KnowledgeGraph` can delegate to the same implementation,
//! keeping the single source of truth in `kglite` (the core).

use std::sync::Arc;

use petgraph::graph::NodeIndex;

use crate::datatypes::values::{raw_string, Value};
use crate::graph::dir_graph::DirGraph;
use crate::graph::embedder::Embedder;
use crate::graph::schema;
use crate::graph::storage::GraphRead;
use crate::graph::{SourceLocation, SourceLookup};

/// Code-entity node types used by `source_location` / `resolve_code_entity`
/// when the caller doesn't specify a `node_type`. Matches what the
/// `code_tree` parser emits — language-specific subsets (Rust:
/// `Struct`/`Enum`/`Trait`; Python: `Class`/`Mixin`/`Protocol`; etc.)
/// are all listed so a single search covers every supported source
/// language.
pub const CODE_TYPES: &[&str] = &[
    "Function",
    "Struct",
    "Class",
    "Mixin",
    "Enum",
    "Trait",
    "Protocol",
    "Interface",
    "Module",
    "Constant",
];

/// Name-matching strategy for [`find_code_entities`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CodeEntityMatch {
    Exact,
    Contains,
    StartsWith,
}

/// Search code-entity type indices by `name` or `title`.
///
/// This is the binding-neutral scan behind the wheel's `find()` method. It
/// returns typed [`schema::NodeInfo`] values; dict/object marshalling remains
/// in the consuming wrapper.
pub fn find_code_entities(
    dir: &Arc<DirGraph>,
    name: &str,
    node_type: Option<&str>,
    match_type: CodeEntityMatch,
) -> Vec<schema::NodeInfo> {
    let _arena_guard = dir.graph.begin_query();
    let name_lower = name.to_lowercase();
    let name_value = Value::String(name.to_string());
    let types_to_search: Vec<&str> = match node_type {
        Some(nt) => vec![nt],
        None => CODE_TYPES.to_vec(),
    };

    let mut results = Vec::new();
    for node_type in types_to_search {
        let Some(indices) = dir.type_indices.get(node_type) else {
            continue;
        };
        for index in indices.iter() {
            let Some(node) = dir.get_node(index) else {
                continue;
            };
            // `title` is a primary NodeData field, not an ordinary property.
            // Resolve it explicitly: `field_*_ci("title")` only covers the
            // property store and silently missed titles extracted at load.
            let title = node.title();
            let title_string = match &*title {
                Value::String(value) => Some(value.as_str()),
                _ => None,
            };
            let matches = match match_type {
                CodeEntityMatch::Contains => {
                    node.field_contains_ci("name", &name_lower)
                        || title_string
                            .is_some_and(|value| value.to_lowercase().contains(&name_lower))
                }
                CodeEntityMatch::StartsWith => {
                    node.field_starts_with_ci("name", &name_lower)
                        || title_string
                            .is_some_and(|value| value.to_lowercase().starts_with(&name_lower))
                }
                CodeEntityMatch::Exact => {
                    node.get_field_ref("name")
                        .is_some_and(|value| *value == name_value)
                        || *title == name_value
                }
            };
            if matches {
                results.push(node.to_node_info(&dir.interner));
            }
        }
    }
    results
}

/// Resolved code-entity neighborhood, kept directional for neutral bindings.
#[derive(Debug)]
pub struct CodeEntityContext {
    pub node: schema::NodeInfo,
    pub defined_in: Option<String>,
    pub outgoing: std::collections::HashMap<String, Vec<schema::NodeInfo>>,
    pub incoming: std::collections::HashMap<String, Vec<schema::NodeInfo>>,
}

/// Outcome of resolving a code entity for [`code_entity_context`].
#[derive(Debug)]
pub enum CodeContextLookup {
    Found(Box<CodeEntityContext>),
    Ambiguous(Vec<schema::NodeInfo>),
    NotFound,
}

/// Resolve a code entity and collect its neighborhood up to `hops` away.
///
/// The traversal and edge-type grouping are shared engine logic. Bindings may
/// flatten or rename the directional groups to suit their native result shape.
pub fn code_entity_context(
    dir: &Arc<DirGraph>,
    name: &str,
    node_type: Option<&str>,
    hops: usize,
) -> CodeContextLookup {
    let _arena_guard = dir.graph.begin_query();
    let (resolved, matches) = resolve_code_entity(dir, name, node_type);
    let Some(target_idx) = resolved else {
        return if matches.is_empty() {
            CodeContextLookup::NotFound
        } else {
            CodeContextLookup::Ambiguous(matches.into_iter().map(|(_, info)| info).collect())
        };
    };
    let Some(target_node) = dir.get_node(target_idx) else {
        return CodeContextLookup::NotFound;
    };

    let neighbor_indices = if hops <= 1 {
        let mut neighbors = std::collections::HashSet::new();
        for edge in dir
            .graph
            .edges_directed(target_idx, petgraph::Direction::Outgoing)
        {
            neighbors.insert(edge.target());
        }
        for edge in dir
            .graph
            .edges_directed(target_idx, petgraph::Direction::Incoming)
        {
            neighbors.insert(edge.source());
        }
        neighbors
    } else {
        let mut visited = std::collections::HashSet::from([target_idx]);
        let mut frontier = std::collections::HashSet::from([target_idx]);
        for _ in 0..hops {
            let mut next_frontier = std::collections::HashSet::new();
            for &node in &frontier {
                for neighbor in dir.graph.neighbors_undirected(node) {
                    if visited.insert(neighbor) {
                        next_frontier.insert(neighbor);
                    }
                }
            }
            if next_frontier.is_empty() {
                break;
            }
            frontier = next_frontier;
        }
        visited.remove(&target_idx);
        visited
    };

    let mut outgoing_indices: std::collections::HashMap<String, Vec<NodeIndex>> =
        std::collections::HashMap::new();
    let mut incoming_indices: std::collections::HashMap<String, Vec<NodeIndex>> =
        std::collections::HashMap::new();
    for edge in dir
        .graph
        .edges_directed(target_idx, petgraph::Direction::Outgoing)
    {
        let target = edge.target();
        if hops <= 1 || neighbor_indices.contains(&target) {
            outgoing_indices
                .entry(edge.weight().connection_type_str(&dir.interner).to_string())
                .or_default()
                .push(target);
        }
    }
    for edge in dir
        .graph
        .edges_directed(target_idx, petgraph::Direction::Incoming)
    {
        let source = edge.source();
        if hops <= 1 || neighbor_indices.contains(&source) {
            incoming_indices
                .entry(edge.weight().connection_type_str(&dir.interner).to_string())
                .or_default()
                .push(source);
        }
    }
    if hops > 1 {
        for &node_idx in &neighbor_indices {
            for edge in dir
                .graph
                .edges_directed(node_idx, petgraph::Direction::Outgoing)
            {
                let target = edge.target();
                if target != target_idx && neighbor_indices.contains(&target) {
                    outgoing_indices
                        .entry(edge.weight().connection_type_str(&dir.interner).to_string())
                        .or_default()
                        .push(target);
                }
            }
        }
    }

    let materialise_groups = |groups: std::collections::HashMap<String, Vec<NodeIndex>>| {
        groups
            .into_iter()
            .map(|(edge_type, indices)| {
                let mut seen = std::collections::HashSet::new();
                let nodes = indices
                    .into_iter()
                    .filter(|index| seen.insert(*index))
                    .filter_map(|index| dir.get_node(index))
                    .map(|node| node.to_node_info(&dir.interner))
                    .collect();
                (edge_type, nodes)
            })
            .collect()
    };

    CodeContextLookup::Found(Box::new(CodeEntityContext {
        node: target_node.to_node_info(&dir.interner),
        defined_in: match target_node.get_field_ref("file_path").as_deref() {
            Some(Value::String(path)) => Some(path.clone()),
            _ => None,
        },
        outgoing: materialise_groups(outgoing_indices),
        incoming: materialise_groups(incoming_indices),
    }))
}

/// Resolve a name (or qualified-name suffix) to a single code-entity
/// `NodeIndex`. Returns `(Some(idx), Vec::new())` for an unambiguous
/// match, `(None, matches)` when 0 or >1 candidates matched.
///
/// Lookup order:
/// 1. Exact match on `node.id()` (the qualified name, e.g.
///    `crate::graph::languages::cypher::executor::CypherExecutor::execute_single_clause`)
/// 2. Suffix match on `node.id()` if `name` contains `::`
///    (e.g. `CypherExecutor::execute_single_clause` matches the above)
/// 3. Exact match on `node.get_field_ref("name")` or
///    `node.get_field_ref("title")` — bare-name fallback
///
/// When `node_type` is `None`, searches across every entry in
/// [`CODE_TYPES`]; otherwise restricted to the single type.
pub fn resolve_code_entity(
    dir: &Arc<DirGraph>,
    name: &str,
    node_type: Option<&str>,
) -> (Option<NodeIndex>, Vec<(NodeIndex, schema::NodeInfo)>) {
    // Arena guard: disk-backed node reads materialize into the query arena
    // (protocol in disk/graph.rs); no-op on memory/mapped.
    let _arena_guard = dir.graph.begin_query();
    let name_val = Value::String(name.to_string());
    let types_to_search: Vec<&str> = match node_type {
        Some(nt) => vec![nt],
        None => CODE_TYPES.to_vec(),
    };

    // Try qualified_name (stored as "id") exact match first
    for nt in &types_to_search {
        if let Some(indices) = dir.type_indices.get(nt) {
            for idx in indices.iter() {
                if let Some(node) = dir.get_node(idx) {
                    if *node.id() == name_val {
                        return (Some(idx), Vec::new());
                    }
                }
            }
        }
    }

    // Try qualified_name suffix match (e.g. "CypherExecutor::execute_single_clause"
    // matches "crate::graph::languages::cypher::executor::CypherExecutor::execute_single_clause")
    if name.contains("::") {
        let suffix = format!("::{}", name);
        let mut matches: Vec<(NodeIndex, schema::NodeInfo)> = Vec::new();
        for nt in &types_to_search {
            if let Some(indices) = dir.type_indices.get(nt) {
                for idx in indices.iter() {
                    if let Some(node) = dir.get_node(idx) {
                        if let Value::String(qn) = &*node.id() {
                            if qn.ends_with(&suffix) {
                                matches.push((idx, node.to_node_info(&dir.interner)));
                            }
                        }
                    }
                }
            }
        }
        if matches.len() == 1 {
            return (Some(matches[0].0), matches);
        } else if !matches.is_empty() {
            return (None, matches);
        }
    }

    // Fall back to name/title search
    let mut matches: Vec<(NodeIndex, schema::NodeInfo)> = Vec::new();
    for nt in &types_to_search {
        if let Some(indices) = dir.type_indices.get(nt) {
            for idx in indices.iter() {
                if let Some(node) = dir.get_node(idx) {
                    let name_match = node
                        .get_field_ref("name")
                        .map(|v| *v == name_val)
                        .unwrap_or(false)
                        || node
                            .get_field_ref("title")
                            .map(|v| *v == name_val)
                            .unwrap_or(false);
                    if name_match {
                        matches.push((idx, node.to_node_info(&dir.interner)));
                    }
                }
            }
        }
    }

    if matches.len() == 1 {
        (Some(matches[0].0), matches)
    } else {
        (None, matches)
    }
}

/// Infer the node type of the current (latest level) selection by
/// sampling the first node. Returns `None` if the selection is empty
/// or the node disappeared.
///
/// **Not re-exported through `kglite::api`** — it takes a
/// `&CowSelection`, which is currently only used externally by the
/// Python wheel's fluent-API surface. A future binding cannot
/// meaningfully call this without first lifting the `Selection`
/// concept to be a stable api type. When that happens, both should
/// move to api together. The wheel reaches this directly via
/// `kglite_core::graph::handle::infer_selection_node_type` for now
/// (see `crates/kglite-py/src/graph/mod.rs`).
pub fn infer_selection_node_type(
    selection: &crate::graph::schema::CowSelection,
    dir: &Arc<DirGraph>,
) -> Option<String> {
    let level_idx = selection.get_level_count().saturating_sub(1);
    let level = selection.get_level(level_idx)?;
    let first_idx = level.iter_node_indices().next()?;
    // Arena guard: node_weight materializes on the disk backend (protocol
    // in disk/graph.rs); no-op on memory/mapped.
    let _arena_guard = dir.graph.begin_query();
    dir.graph
        .node_weight(first_idx)
        .map(|n| n.node_type_str(&dir.interner).to_string())
}

/// Discover all unique property keys across a slice of typed nodes.
/// Returns sorted, de-duplicated key names — useful for any
/// row-oriented exporter (CSV, Parquet, DataFrame, JSON-lines) that
/// needs a stable column-name set without scanning the entire graph
/// schema. The function takes only core types (`NodeData`,
/// `StringInterner`) so every binding's table-export path can call
/// it directly.
pub fn discover_property_keys_from_data(
    nodes: &[(&str, &crate::graph::schema::NodeData)],
    interner: &crate::graph::schema::StringInterner,
) -> Vec<String> {
    let mut seen = std::collections::HashSet::new();
    let mut keys = Vec::new();
    for (_, node) in nodes {
        for key in node.property_keys(interner) {
            if seen.insert(key.to_string()) {
                keys.push(key.to_string());
            }
        }
    }
    keys.sort();
    keys
}

/// Look up the source-file location for a code-entity node.
///
/// Drives the `read_code_source` MCP tool's "qualified_name →
/// (file_path, line_number, end_line, signature)" mapping. The
/// returned [`SourceLookup`] enum distinguishes a unique match
/// ([`SourceLookup::Found`]) from ambiguous candidates
/// ([`SourceLookup::Ambiguous`] with qualified-name suggestions)
/// from a miss ([`SourceLookup::NotFound`]).
///
/// All optional fields on [`SourceLocation`] mirror the
/// corresponding node fields. Graphs built from non-code-tree
/// sources (e.g. a hand-built `code_tree::build` output, or a
/// manually-constructed graph) may have fewer populated.
pub fn source_location(dir: &Arc<DirGraph>, name: &str, node_type: Option<&str>) -> SourceLookup {
    // Arena guard: disk-backed node reads materialize into the query arena
    // (protocol in disk/graph.rs); no-op on memory/mapped.
    let _arena_guard = dir.graph.begin_query();
    let (resolved, matches) = resolve_code_entity(dir, name, node_type);

    if let Some(target_idx) = resolved {
        let node = match dir.get_node(target_idx) {
            Some(n) => n,
            None => return SourceLookup::NotFound,
        };
        let type_name = node.get_node_type_ref(&dir.interner).to_string();
        let entity_name = raw_string(&node.title());
        let qname = raw_string(&node.id());
        let file_path = node.get_field_ref("file_path").as_deref().map(raw_string);
        let line_number = node
            .get_field_ref("line_number")
            .as_deref()
            .and_then(|v| match v {
                Value::Int64(n) => Some(*n),
                _ => None,
            });
        let end_line = node
            .get_field_ref("end_line")
            .as_deref()
            .and_then(|v| match v {
                Value::Int64(n) => Some(*n),
                _ => None,
            });
        let signature = node.get_field_ref("signature").as_deref().map(raw_string);
        SourceLookup::Found(SourceLocation {
            type_name,
            name: entity_name,
            qualified_name: qname,
            file_path,
            line_number,
            end_line,
            signature,
        })
    } else if matches.is_empty() {
        SourceLookup::NotFound
    } else {
        let qnames: Vec<String> = matches
            .iter()
            .map(|(_, info)| raw_string(&info.id))
            .collect();
        SourceLookup::Ambiguous(qnames)
    }
}

/// Thin pure-Rust graph handle. Holds an `Arc<DirGraph>` plus an
/// optional [`Embedder`] for `text_score()` queries. For Rust
/// embedders (mcp-server, bolt-server, third-party binaries) that
/// don't need the Python wheel's full state.
///
/// The Python wheel's `KnowledgeGraph` (in `kglite-py`) has the
/// same name but adds wheel-API state (selection, reports,
/// mutation stats, temporal context, default timeout / max-rows).
/// The two types don't share a definition; pick whichever fits
/// your audience.
pub struct KnowledgeGraph {
    inner: Arc<DirGraph>,
    embedder: Option<Arc<dyn Embedder>>,
}

impl KnowledgeGraph {
    /// Wrap an existing `Arc<DirGraph>` (e.g. one returned by
    /// [`crate::graph::io::file::load_file`] or
    /// [`crate::code_tree::builder::run_with_options`]) into a
    /// `KnowledgeGraph` handle with no embedder set.
    pub fn from_arc(inner: Arc<DirGraph>) -> Self {
        KnowledgeGraph {
            inner,
            embedder: None,
        }
    }

    /// Borrow the underlying `Arc<DirGraph>`. Use this to reach
    /// the engine surface (`compute_schema`, `execute_read`,
    /// `compute_description`, ...) which all take `&DirGraph`.
    pub fn dir(&self) -> &Arc<DirGraph> {
        &self.inner
    }

    /// Mutable borrow of the underlying `Arc<DirGraph>` — the write
    /// counterpart of [`dir`](Self::dir). Pair with
    /// [`make_dir_graph_mut`] to obtain a `&mut DirGraph` for the
    /// mutation surface (`execute_mut`, …). Used by bindings that hold a
    /// long-lived handle and mutate it in place (e.g. the write-enabled
    /// MCP server), so the mutation lands on *this* handle's graph rather
    /// than a detached clone.
    pub fn dir_mut(&mut self) -> &mut Arc<DirGraph> {
        &mut self.inner
    }

    /// Bind an embedder implementing the [`Embedder`] trait — used
    /// by `text_score()` Cypher to map text queries onto stored
    /// vectors. Replaces any previously-bound embedder. Callers
    /// that wrap a Python embedder object should construct an
    /// adapter in the wheel crate; pure-Rust callers can pass
    /// e.g. `Arc::new(FastEmbedAdapter::new("bge-small")?)`.
    pub fn set_embedder_native(&mut self, embedder: Arc<dyn Embedder>) {
        self.embedder = Some(embedder);
    }

    /// Access the active embedder, if any. Returns `None` until
    /// [`set_embedder_native`](Self::set_embedder_native) has been
    /// called.
    pub fn embedder(&self) -> Option<&Arc<dyn Embedder>> {
        self.embedder.as_ref()
    }

    /// Look up the source-file location for a code-entity node by
    /// name (or qualified-name suffix). Delegates to the
    /// [`source_location`] free function so the wheel crate's
    /// `KnowledgeGraph` can share the same implementation.
    pub fn source_location(&self, name: &str, node_type: Option<&str>) -> SourceLookup {
        source_location(&self.inner, name, node_type)
    }
}

/// Get a `&mut DirGraph` from an `Arc<DirGraph>` and bump the version
/// counter. Wraps [`Arc::make_mut`] (which clones the inner `DirGraph`
/// if other strong refs exist) plus the canonical post-mutation version
/// increment that downstream OCC commit-checks + the plan cache rely on.
///
/// Lifted from the wheel crate in 0.10.1 so bindings + embedders that
/// hold an `Arc<DirGraph>` and want to mutate it have a single,
/// consistent entry point. Re-exported as `kglite::api::make_dir_graph_mut`.
/// (Homed here rather than in `dir_graph.rs` to keep that file under the
/// god-file ceiling.)
///
/// **Warning:** If other `Arc<DirGraph>` references exist (e.g. a
/// snapshot held by an open transaction, or a clone held by a still-
/// alive `ResultView`), this deep-clones the entire graph — every
/// node, edge, and index. Mutation in a read-heavy workload is fine,
/// but a lingering reference can cause an unexpected memory spike on
/// the first write.
pub fn make_dir_graph_mut(arc: &mut Arc<DirGraph>) -> &mut DirGraph {
    let graph = Arc::make_mut(arc);
    graph.bump_version();
    graph
}

#[cfg(test)]
mod boundary_lift_tests {
    use super::*;
    use crate::graph::session::{execute_mut, ExecuteOptions};
    use std::collections::HashMap;

    fn code_graph() -> Arc<DirGraph> {
        let mut graph = DirGraph::new();
        let params = HashMap::new();
        execute_mut(
            &mut graph,
            "CREATE (a:Function {id:'mod::alpha', title:'alpha', name:'alpha', file_path:'src/a.rs'}), \
             (b:Function {id:'mod::beta', title:'BetaWorker', name:'beta', file_path:'src/b.rs'}), \
             (c:Function {id:'mod::gamma', title:'gamma', name:'gamma', file_path:'src/c.rs'}), \
             (f:File {id:'src/a.rs', title:'src/a.rs'})",
            &ExecuteOptions::eager(&params),
        )
        .expect("fixture nodes");
        execute_mut(
            &mut graph,
            "MATCH (a:Function {id:'mod::alpha'}), (b:Function {id:'mod::beta'}), \
             (c:Function {id:'mod::gamma'}), (f:File {id:'src/a.rs'}) \
             CREATE (a)-[:CALLS]->(b), (b)-[:CALLS]->(c), (f)-[:DEFINES]->(a)",
            &ExecuteOptions::eager(&params),
        )
        .expect("fixture edges");
        Arc::new(graph)
    }

    #[test]
    fn find_code_entities_supports_match_modes_and_type_filter() {
        let graph = code_graph();
        let exact = find_code_entities(&graph, "alpha", Some("Function"), CodeEntityMatch::Exact);
        assert_eq!(exact.len(), 1);
        assert_eq!(exact[0].id, Value::String("mod::alpha".into()));

        let contains = find_code_entities(&graph, "et", None, CodeEntityMatch::Contains);
        assert_eq!(contains.len(), 1);
        assert_eq!(contains[0].id, Value::String("mod::beta".into()));

        let starts_with = find_code_entities(&graph, "bet", None, CodeEntityMatch::StartsWith);
        assert_eq!(starts_with.len(), 1);
        assert_eq!(starts_with[0].id, Value::String("mod::beta".into()));
    }

    #[test]
    fn code_entity_context_groups_directional_multi_hop_neighbors() {
        let graph = code_graph();
        let CodeContextLookup::Found(context) =
            code_entity_context(&graph, "alpha", Some("Function"), 2)
        else {
            panic!("expected resolved context");
        };
        assert_eq!(context.defined_in.as_deref(), Some("src/a.rs"));
        let calls = &context.outgoing["CALLS"];
        assert_eq!(calls.len(), 2);
        assert!(calls
            .iter()
            .any(|node| node.id == Value::String("mod::beta".into())));
        assert!(calls
            .iter()
            .any(|node| node.id == Value::String("mod::gamma".into())));
        assert_eq!(context.incoming["DEFINES"].len(), 1);
    }

    #[test]
    fn code_entity_context_distinguishes_miss_from_ambiguity() {
        let mut graph = match Arc::try_unwrap(code_graph()) {
            Ok(graph) => graph,
            Err(_) => panic!("expected sole graph owner"),
        };
        let params = HashMap::new();
        execute_mut(
            &mut graph,
            "CREATE (:Function {id:'other::alpha', title:'alpha', name:'alpha', file_path:'other.rs'})",
            &ExecuteOptions::eager(&params),
        )
        .expect("add ambiguous entity");
        let graph = Arc::new(graph);
        assert!(matches!(
            code_entity_context(&graph, "alpha", Some("Function"), 1),
            CodeContextLookup::Ambiguous(matches) if matches.len() == 2
        ));
        assert!(matches!(
            code_entity_context(&graph, "missing", None, 1),
            CodeContextLookup::NotFound
        ));
    }
}
