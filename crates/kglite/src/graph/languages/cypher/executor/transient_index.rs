//! Query-local equality hash index for cross-MATCH joins on non-id properties.
//!
//! Built once per `execute_match` call when the heuristic detects a single
//! typed-node pattern carrying exactly one `EqualsVar` / `EqualsNodeProp`
//! matcher. Probed per outer row, replacing N×M property scans with O(N+M)
//! work. Mirrors the per-query R-tree pattern in [`super::spatial_join`].
//!
//! The index is dropped when the executor goes out of scope; it never
//! mutates [`DirGraph::property_indices`].

use super::helpers::resolve_node_property;
use super::ResultRow;
use crate::datatypes::values::Value;
use crate::graph::core::pattern_matching::{NodePattern, Pattern, PatternElement, PropertyMatcher};
use crate::graph::schema::DirGraph;
use crate::graph::storage::GraphRead;
use petgraph::graph::NodeIndex;
use std::collections::HashMap;

/// Activation threshold: only build the index when there are at least this
/// many outer rows. Below it, the per-row pattern execution is already
/// cheap enough that the build cost doesn't pay back.
pub(super) const TRANSIENT_INDEX_THRESHOLD: usize = 64;

/// Per-query equality index over a single typed property.
pub(super) struct TransientEqIndex {
    /// Pattern variable bound by an index probe (e.g. `"pg"`).
    pub(super) bind_var: String,
    /// Node type the index covers (e.g. `"Prospect"`).
    #[allow(dead_code)]
    pub(super) node_type: String,
    /// Property used as the equality key (e.g. `"prospect_number"`).
    #[allow(dead_code)]
    pub(super) property: String,
    /// How to resolve the per-row probe value.
    pub(super) resolution: ProbeResolution,
    /// Built index: property value → matching `NodeIndex`(es).
    pub(super) by_value: HashMap<Value, Vec<NodeIndex>>,
}

/// How to read the probe value from a row.
pub(super) enum ProbeResolution {
    /// Resolved from `row.projected[var]` — pushed by the planner from
    /// `WITH x AS pnum MATCH (n {prop: pnum})` style joins.
    Projected(String),
    /// Resolved by reading `row.node_bindings[var]`'s property `prop`.
    /// Pushed from correlated `MATCH (a) MATCH (b) WHERE b.x = a.y`.
    NodeProp { var: String, prop: String },
}

impl TransientEqIndex {
    /// Try to build a transient index for `pattern`. Returns `None` when:
    /// - the pattern shape doesn't qualify (not a single typed node, more
    ///   than one matcher, etc.),
    /// - the existing-row count is below the threshold,
    /// - a persistent index already covers `(node_type, property)`,
    /// - or the type has no live nodes.
    pub(super) fn try_build(
        graph: &DirGraph,
        pattern: &Pattern,
        existing_row_count: usize,
    ) -> Option<TransientEqIndex> {
        if existing_row_count < TRANSIENT_INDEX_THRESHOLD {
            return None;
        }
        let np = extract_single_node_pattern(pattern)?;
        let node_type = np.node_type.as_deref()?.to_string();
        let bind_var = np.variable.as_deref()?.to_string();
        let props = np.properties.as_ref()?;
        if props.len() != 1 {
            return None;
        }
        let (property, matcher) = props.iter().next()?;
        let resolution = match matcher {
            PropertyMatcher::EqualsVar(name) => ProbeResolution::Projected(name.clone()),
            PropertyMatcher::EqualsNodeProp { var, prop } => ProbeResolution::NodeProp {
                var: var.clone(),
                prop: prop.clone(),
            },
            _ => return None,
        };
        // Don't double-build when a persistent index already exists.
        if graph.has_any_index(&node_type, property) {
            return None;
        }
        let nodes = graph.type_indices.get(&node_type)?;
        if nodes.is_empty() {
            return None;
        }
        let mut by_value: HashMap<Value, Vec<NodeIndex>> = HashMap::with_capacity(nodes.len());
        for idx in nodes.iter() {
            if let Some(node) = graph.graph.node_weight(idx) {
                let val = resolve_node_property(node, property, graph);
                if !matches!(val, Value::Null) {
                    by_value.entry(val).or_default().push(idx);
                }
            }
        }
        Some(TransientEqIndex {
            bind_var,
            node_type,
            property: property.clone(),
            resolution,
            by_value,
        })
    }

    /// Resolve the probe value for this row. `None` means "no candidates":
    /// either the variable is missing or the value is null (Cypher
    /// equality with null never matches).
    pub(super) fn probe_value(&self, row: &ResultRow, graph: &DirGraph) -> Option<Value> {
        let value = match &self.resolution {
            ProbeResolution::Projected(var) => row.projected.get(var.as_str()).cloned()?,
            ProbeResolution::NodeProp { var, prop } => {
                let idx = row.node_bindings.get(var.as_str())?;
                let node = graph.graph.node_weight(*idx)?;
                resolve_node_property(node, prop, graph)
            }
        };
        if matches!(value, Value::Null) {
            None
        } else {
            Some(value)
        }
    }

    /// Look up matching node indices by value. Empty slice if no match.
    pub(super) fn lookup(&self, value: &Value) -> &[NodeIndex] {
        self.by_value
            .get(value)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }
}

fn extract_single_node_pattern(pattern: &Pattern) -> Option<&NodePattern> {
    if pattern.elements.len() != 1 {
        return None;
    }
    match &pattern.elements[0] {
        PatternElement::Node(np) => Some(np),
        _ => None,
    }
}
