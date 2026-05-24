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
/// or the node disappeared. Used by introspection helpers that need
/// to know "what kind of nodes did we just collect" without forcing
/// the caller to track it. Lifted from kglite-py in 0.10.1.
pub fn infer_selection_node_type(
    selection: &crate::graph::schema::CowSelection,
    dir: &Arc<DirGraph>,
) -> Option<String> {
    let level_idx = selection.get_level_count().saturating_sub(1);
    let level = selection.get_level(level_idx)?;
    let first_idx = level.iter_node_indices().next()?;
    dir.graph
        .node_weight(first_idx)
        .map(|n| n.node_type_str(&dir.interner).to_string())
}

/// Discover all unique property keys across a slice of typed nodes.
/// Returns sorted, de-duplicated key names. Used by DataFrame-style
/// exporters that need a stable column-name set without scanning the
/// entire graph schema. Lifted from kglite-py in 0.10.1.
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
