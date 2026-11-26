//! `ExportManifest` (`kglite-export/1`): what an export must carry beside the
//! rows for a reader to rebuild the graph exactly — the valid-time
//! declarations, secondary labels, parent types, the kglite type of every
//! property column, and the id and title kinds of every node type.
//!
//! One manifest, several encodings: the CSV export writes it as
//! `manifest.json` beside `blueprint.json`; other formats embed the same JSON.
//! [`ExportManifest::apply_declarations`] is the reverse direction every
//! importer calls once its rows are loaded.
//!
//! JSON schema (all keys always present, maps sorted by key):
//! ```json
//! {
//!   "format": "kglite-export/1",
//!   "node_types": {
//!     "Person": {
//!       "count": 3,
//!       "id_kind": "Int64",
//!       "title_kind": "String",
//!       "labels": ["Employee"],
//!       "partial_labels": [],
//!       "parent": null,
//!       "properties": {"name": "String", "hired": "DateTime"}
//!     }
//!   },
//!   "relationship_types": {
//!     "WORKS_AT": {
//!       "count": 2,
//!       "sources": {
//!         "Person": {"count": 2, "targets": ["Plant"], "properties": {"since": "Timestamp"}}
//!       }
//!     }
//!   },
//!   "temporal": [
//!     {"target": "node", "name": "Person", "source_type": null,
//!      "from": "hired", "to": "left", "convention": "closed"}
//!   ]
//! }
//! ```
//! A column kind is one of `UniqueId`, `Int64`, `Float64`, `String`, `Boolean`,
//! `DateTime` (a date), `Timestamp`, `Point`, `Duration`, `List`, `Map`,
//! `Null` (no row holds a value) or `Mixed` (rows of different kinds; a
//! text-only encoding degrades such a column to its display strings).
//! `labels` are the secondary labels every node of the type carries;
//! `partial_labels` are carried by only some of its nodes, which a per-type
//! label declaration cannot express.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use petgraph::graph::NodeIndex;
use serde::{Deserialize, Serialize};

use crate::datatypes::values::Value;
use crate::graph::diagnostics::{Diagnostic, DiagnosticGroup};
use crate::graph::dir_graph::DirGraph;
use crate::graph::features::temporal::{
    declare_loaded, declared, IntervalConvention, TemporalTarget,
};
use crate::graph::schema::CurrentSelection;
use crate::graph::storage::{GraphRead, GraphWrite};

/// The `format` value of this manifest version.
pub const MANIFEST_FORMAT: &str = "kglite-export/1";

/// The kglite type of one property column, id or title.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ColumnKind {
    UniqueId,
    Int64,
    Float64,
    String,
    Boolean,
    DateTime,
    Timestamp,
    Point,
    Duration,
    LocalTime,
    Time,
    ZonedDateTime,
    List,
    Map,
    /// No row holds a value.
    Null,
    /// Rows of different kinds.
    Mixed,
}

impl ColumnKind {
    /// The kind of one value; `None` for a value that says nothing (null).
    pub fn of(value: &Value) -> Option<Self> {
        Some(match value {
            Value::UniqueId(_) => Self::UniqueId,
            Value::Int64(_) => Self::Int64,
            Value::Float64(_) => Self::Float64,
            Value::String(_) => Self::String,
            Value::Boolean(_) => Self::Boolean,
            Value::DateTime(_) => Self::DateTime,
            Value::Timestamp(_) => Self::Timestamp,
            Value::Point { .. } => Self::Point,
            Value::Duration { .. } => Self::Duration,
            Value::LocalTime(_) => Self::LocalTime,
            Value::Time { .. } => Self::Time,
            Value::ZonedDateTime { .. } => Self::ZonedDateTime,
            Value::List(_) => Self::List,
            Value::Map(_) => Self::Map,
            Value::Null | Value::NodeRef(_) => return None,
            Value::Node(_) | Value::Relationship(_) | Value::Path(_) => Self::Mixed,
        })
    }

    /// Fold `other` in: equal kinds stay, `UniqueId` and `Int64` are one
    /// integer kind (ids compare equal across them), anything else mixes.
    pub fn merge(self, other: Self) -> Self {
        match (self, other) {
            (a, b) if a == b => a,
            (Self::Null, b) => b,
            (a, Self::Null) => a,
            (Self::UniqueId, Self::Int64) | (Self::Int64, Self::UniqueId) => Self::Int64,
            _ => Self::Mixed,
        }
    }
}

/// Accumulates the kind of one column over its values.
#[derive(Clone, Copy, Debug)]
pub struct KindTally(ColumnKind);

impl Default for KindTally {
    fn default() -> Self {
        Self(ColumnKind::Null)
    }
}

impl KindTally {
    pub fn observe(&mut self, value: &Value) {
        if let Some(kind) = ColumnKind::of(value) {
            self.0 = self.0.merge(kind);
        }
    }

    pub fn kind(self) -> ColumnKind {
        self.0
    }
}

/// One valid-time declaration.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeclarationEntry {
    /// `"node"` (a label) or `"relationship"` (a type).
    pub target: String,
    pub name: String,
    /// For a relationship: the source type a keyed declaration covers.
    pub source_type: Option<String>,
    pub from: String,
    pub to: String,
    /// `"closed"` or `"half_open"`.
    pub convention: String,
}

impl DeclarationEntry {
    fn target(&self) -> TemporalTarget {
        if self.target == "node" {
            TemporalTarget::Node(self.name.clone())
        } else {
            TemporalTarget::Relationship {
                rel_type: self.name.clone(),
                source_type: self.source_type.clone(),
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct NodeTypeManifest {
    pub count: usize,
    pub id_kind: ColumnKind,
    pub title_kind: ColumnKind,
    pub labels: Vec<String>,
    pub partial_labels: Vec<String>,
    pub parent: Option<String>,
    pub properties: BTreeMap<String, ColumnKind>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RelSourceManifest {
    pub count: usize,
    pub targets: Vec<String>,
    pub properties: BTreeMap<String, ColumnKind>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RelTypeManifest {
    pub count: usize,
    pub sources: BTreeMap<String, RelSourceManifest>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExportManifest {
    pub format: String,
    pub node_types: BTreeMap<String, NodeTypeManifest>,
    pub relationship_types: BTreeMap<String, RelTypeManifest>,
    pub temporal: Vec<DeclarationEntry>,
}

/// The nodes an export covers: every node, or a selection's last level.
pub(crate) struct Scope {
    selected: Option<HashSet<NodeIndex>>,
    ordered: Option<Vec<NodeIndex>>,
}

impl Scope {
    pub(crate) fn new(graph: &DirGraph, selection: Option<&CurrentSelection>) -> Self {
        match selection {
            None => Self {
                selected: None,
                ordered: None,
            },
            Some(_) => {
                let ordered = super::selected_node_indices(graph, selection);
                Self {
                    selected: Some(ordered.iter().copied().collect()),
                    ordered: Some(ordered),
                }
            }
        }
    }

    pub(crate) fn contains(&self, idx: NodeIndex) -> bool {
        self.selected.as_ref().is_none_or(|s| s.contains(&idx))
    }

    /// Visit every covered node in node order.
    pub(crate) fn for_each_node(
        &self,
        graph: &DirGraph,
        mut visit: impl FnMut(NodeIndex) -> Result<(), String>,
    ) -> Result<(), String> {
        match &self.ordered {
            Some(list) => list.iter().try_for_each(|&i| visit(i)),
            None => graph.graph.node_indices().try_for_each(visit),
        }
    }
}

impl ExportManifest {
    /// Scan the graph (or a selection) once and describe it.
    pub fn build(
        graph: &DirGraph,
        selection: Option<&CurrentSelection>,
        parent_types: &HashMap<String, String>,
    ) -> Result<Self, String> {
        let scope = Scope::new(graph, selection);
        Self::build_in(graph, &scope, parent_types)
    }

    pub(crate) fn build_in(
        graph: &DirGraph,
        scope: &Scope,
        parent_types: &HashMap<String, String>,
    ) -> Result<Self, String> {
        struct NodeAcc {
            count: usize,
            id: KindTally,
            title: KindTally,
            props: BTreeMap<String, KindTally>,
        }
        let mut nodes: BTreeMap<String, NodeAcc> = BTreeMap::new();
        scope.for_each_node(graph, |idx| {
            let Some(node) = graph.graph.node_view(idx) else {
                return Ok(());
            };
            let acc = nodes
                .entry(node.node_type_str(&graph.interner).to_string())
                .or_insert_with(|| NodeAcc {
                    count: 0,
                    id: KindTally::default(),
                    title: KindTally::default(),
                    props: BTreeMap::new(),
                });
            acc.count += 1;
            acc.id.observe(&node.id());
            acc.title.observe(&node.title());
            for key in node.property_keys(&graph.interner) {
                // `id` and `title` are the node's own header columns; neither
                // export has a structural `type` column on a node row, so a
                // property named `type` is ordinary user data.
                if key == "id" || key == "title" {
                    continue;
                }
                let tally = acc.props.entry(key.to_string()).or_default();
                if let Some(value) = node.get_property(key) {
                    tally.observe(&value);
                }
            }
            Ok(())
        })?;

        let mut label_counts: BTreeMap<(String, String), usize> = BTreeMap::new();
        if graph.has_secondary_labels {
            for (key, bucket) in &graph.secondary_label_index {
                let label = graph.interner.resolve(*key).to_string();
                for &idx in bucket {
                    if !scope.contains(idx) {
                        continue;
                    }
                    if let Some(ty) = graph.graph.node_type_of(idx) {
                        *label_counts
                            .entry((graph.interner.resolve(ty).to_string(), label.clone()))
                            .or_default() += 1;
                    }
                }
            }
        }

        let mut node_types = BTreeMap::new();
        for (name, acc) in nodes {
            let mut labels = Vec::new();
            let mut partial = Vec::new();
            for ((ty, label), n) in &label_counts {
                if ty == &name {
                    if *n == acc.count {
                        labels.push(label.clone());
                    } else {
                        partial.push(label.clone());
                    }
                }
            }
            node_types.insert(
                name.clone(),
                NodeTypeManifest {
                    count: acc.count,
                    id_kind: acc.id.kind(),
                    title_kind: acc.title.kind(),
                    labels,
                    partial_labels: partial,
                    parent: parent_types.get(&name).cloned(),
                    properties: acc.props.into_iter().map(|(k, t)| (k, t.kind())).collect(),
                },
            );
        }

        let relationship_types = Self::scan_relationships(graph, scope)?;
        let mut temporal: Vec<DeclarationEntry> = declared(graph)
            .into_iter()
            .map(|info| match info.target {
                TemporalTarget::Node(label) => DeclarationEntry {
                    target: "node".into(),
                    name: label,
                    source_type: None,
                    from: info.config.valid_from,
                    to: info.config.valid_to,
                    convention: info.config.convention.as_str().into(),
                },
                TemporalTarget::Relationship {
                    rel_type,
                    source_type,
                } => DeclarationEntry {
                    target: "relationship".into(),
                    name: rel_type,
                    source_type,
                    from: info.config.valid_from,
                    to: info.config.valid_to,
                    convention: info.config.convention.as_str().into(),
                },
            })
            .collect();
        temporal.sort_by(|a, b| {
            (&a.target, &a.name, &a.source_type).cmp(&(&b.target, &b.name, &b.source_type))
        });

        Ok(ExportManifest {
            format: MANIFEST_FORMAT.to_string(),
            node_types,
            relationship_types,
            temporal,
        })
    }

    fn scan_relationships(
        graph: &DirGraph,
        scope: &Scope,
    ) -> Result<BTreeMap<String, RelTypeManifest>, String> {
        #[derive(Default)]
        struct SourceAcc {
            count: usize,
            targets: BTreeSet<String>,
            props: BTreeMap<String, KindTally>,
        }
        let mut acc: BTreeMap<(String, String), SourceAcc> = BTreeMap::new();
        scope.for_each_node(graph, |source| {
            let Some(src) = graph.graph.node_view(source) else {
                return Ok(());
            };
            let source_type = src.node_type_str(&graph.interner).to_string();
            for edge in graph.graph.edges(source) {
                let target = edge.target();
                if !scope.contains(target) {
                    continue;
                }
                let Some(tgt) = graph.graph.node_view(target) else {
                    continue;
                };
                let w = edge.weight();
                let entry = acc
                    .entry((
                        w.connection_type_str(&graph.interner).to_string(),
                        source_type.clone(),
                    ))
                    .or_default();
                entry.count += 1;
                entry
                    .targets
                    .insert(tgt.node_type_str(&graph.interner).to_string());
                for (key, value) in w.properties_cloned(&graph.interner) {
                    entry.props.entry(key).or_default().observe(&value);
                }
            }
            Ok(())
        })?;
        let mut out: BTreeMap<String, RelTypeManifest> = BTreeMap::new();
        for ((rel, source), a) in acc {
            let entry = out.entry(rel).or_insert_with(|| RelTypeManifest {
                count: 0,
                sources: BTreeMap::new(),
            });
            entry.count += a.count;
            entry.sources.insert(
                source,
                RelSourceManifest {
                    count: a.count,
                    targets: a.targets.into_iter().collect(),
                    properties: a.props.into_iter().map(|(k, t)| (k, t.kind())).collect(),
                },
            );
        }
        Ok(out)
    }

    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).expect("a manifest serializes")
    }

    pub fn from_json(text: &str) -> Result<Self, String> {
        let manifest: Self =
            serde_json::from_str(text).map_err(|e| format!("Invalid export manifest: {e}"))?;
        if manifest.format != MANIFEST_FORMAT {
            return Err(format!(
                "Unsupported export manifest format '{}' (this build reads '{}')",
                manifest.format, MANIFEST_FORMAT
            ));
        }
        Ok(manifest)
    }

    /// Give the nodes of every type whose titles mix kinds their own typed
    /// title back: the CSV export writes such a title as its JSON spelling and
    /// the blueprint reads the column as text. Returns the titles restored.
    pub fn restore_mixed_titles(&self, graph: &mut DirGraph) -> usize {
        let mut restored = 0;
        for (name, info) in &self.node_types {
            if info.title_kind != ColumnKind::Mixed {
                continue;
            }
            let Some(nodes) = graph.type_indices.get(name) else {
                continue;
            };
            let indices: Vec<NodeIndex> = nodes.iter().collect();
            let mut decoded: Vec<(NodeIndex, Value)> = Vec::new();
            {
                for idx in indices {
                    let Some(node) = graph.graph.node_view(idx) else {
                        continue;
                    };
                    if let Value::String(text) = &*node.title() {
                        if let Ok(json) = serde_json::from_str(text) {
                            decoded.push((idx, super::typed_json::from_json(&json)));
                        }
                    }
                }
            }
            restored += decoded.len();
            for (idx, title) in decoded {
                graph.graph.set_node_title(idx, title);
            }
        }
        restored
    }

    /// Declare every valid-time declaration on a graph whose rows are loaded.
    /// Identical declarations already in place are no-ops. A declaration whose
    /// target holds no rows is skipped and named in the returned warnings; a
    /// refusal (an unreadable bound, a conflicting declaration) is an error.
    pub fn apply_declarations(&self, graph: &mut DirGraph) -> Result<Vec<Diagnostic>, String> {
        let mut warnings = Vec::new();
        for entry in &self.temporal {
            let target = entry.target();
            let convention = IntervalConvention::parse(&entry.convention).ok_or_else(|| {
                format!(
                    "Invalid export manifest: convention '{}' for {}",
                    entry.convention,
                    target.describe()
                )
            })?;
            let present = match &target {
                TemporalTarget::Node(label) => {
                    graph.has_node_type(label)
                        || graph
                            .secondary_label_index
                            .keys()
                            .any(|key| graph.interner.resolve(*key) == label.as_str())
                }
                TemporalTarget::Relationship { rel_type, .. } => {
                    graph.connection_type_metadata.contains_key(rel_type)
                }
            };
            if !present {
                warnings.push(Diagnostic::new(
                    DiagnosticGroup::Declarations,
                    "interval_not_declared",
                    format!(
                        "{}: no rows were loaded, so no validity interval is declared",
                        target.describe()
                    ),
                ));
                continue;
            }
            let report = declare_loaded(
                graph,
                &target,
                &entry.from,
                &entry.to,
                convention,
                &[entry.from.as_str(), entry.to.as_str()],
            )
            .map_err(|e| format!("{}: the declaration is refused: {e}", target.describe()))?;
            warnings.extend(report.diagnostics);
        }
        Ok(warnings)
    }
}

#[cfg(test)]
#[path = "manifest_tests.rs"]
mod manifest_tests;
