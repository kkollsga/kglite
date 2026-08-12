// src/graph/statistics.rs
use crate::datatypes::values::Value;
use crate::graph::schema::{CurrentSelection, DirGraph, InternedKey};
use petgraph::graph::NodeIndex;
use std::borrow::Cow;
use std::collections::{HashMap, HashSet};

#[derive(Debug)]
pub struct ParentChildPair {
    pub parent: Option<NodeIndex>,
    pub children: Vec<NodeIndex>,
}

pub fn get_parent_child_pairs(
    selection: &CurrentSelection,
    level_index: Option<usize>,
) -> Vec<ParentChildPair> {
    // If no level specified, use the deepest level
    let target_level = level_index.unwrap_or_else(|| selection.get_level_count().saturating_sub(1));

    // Return empty vec if level doesn't exist
    if target_level >= selection.get_level_count() {
        return Vec::new();
    }

    let level = selection
        .get_level(target_level)
        .expect("Level index was already checked");

    // If the level has no selections, return empty vec
    if level.is_empty() {
        return Vec::new();
    }

    // If we have parent-child pairs, return them
    if level.iter_groups().any(|(parent, _)| parent.is_some()) {
        level
            .iter_groups()
            .map(|(parent, children)| ParentChildPair {
                parent: *parent,
                children: children.clone(),
            })
            .collect()
    } else {
        // For root level or standalone selections, create a single pair with no parent
        vec![ParentChildPair {
            parent: None,
            children: level.get_all_nodes(),
        }]
    }
}

/// Collect all selected node indices from the specified level (flattened).
pub fn collect_selected_nodes(
    selection: &CurrentSelection,
    level_index: Option<usize>,
) -> Vec<NodeIndex> {
    let target_level = level_index.unwrap_or_else(|| selection.get_level_count().saturating_sub(1));
    selection
        .get_level(target_level)
        .map(|level| level.get_all_nodes())
        .unwrap_or_default()
}

/// Numeric statistics for one group in [`calculate_grouped_property_stats`].
#[derive(Debug, Clone, PartialEq)]
pub struct GroupedPropertyStats {
    pub count: usize,
    pub sum: Option<f64>,
    pub mean: Option<f64>,
    pub min: Option<f64>,
    pub max: Option<f64>,
    /// Sample standard deviation (`n - 1` denominator), absent for 0/1 values.
    pub std: Option<f64>,
}

/// Calculate numeric property statistics grouped by another selected-node property.
///
/// This backs the existing fluent `statistics(..., group_by=...)` operation.
/// It accepts only core graph/selection types and returns typed Rust data;
/// bindings retain responsibility for dict/object marshalling.
pub fn calculate_grouped_property_stats(
    graph: &DirGraph,
    selection: &CurrentSelection,
    property: &str,
    group_by: &str,
    level_index: Option<usize>,
) -> HashMap<String, GroupedPropertyStats> {
    let _arena_guard = graph.graph.begin_query();
    let mut grouped_values: HashMap<String, Vec<f64>> = HashMap::new();
    for index in collect_selected_nodes(selection, level_index) {
        let Some(node) = graph.node_view(index) else {
            continue;
        };
        let key = match resolved_stat_field(graph, node, group_by).as_deref() {
            Some(Value::String(value)) => value.clone(),
            Some(Value::Int64(value)) => value.to_string(),
            Some(value) => format!("{:?}", value),
            None => "null".to_string(),
        };
        let values = grouped_values.entry(key).or_default();
        let numeric = resolved_stat_field(graph, node, property)
            .as_deref()
            .and_then(|value| match value {
                Value::Int64(value) => Some(*value as f64),
                Value::Float64(value) => Some(*value),
                Value::UniqueId(value) => Some(*value as f64),
                _ => None,
            });
        if let Some(value) = numeric {
            values.push(value);
        }
    }

    grouped_values
        .into_iter()
        .map(|(key, values)| {
            let count = values.len();
            let stats = if count == 0 {
                GroupedPropertyStats {
                    count,
                    sum: None,
                    mean: None,
                    min: None,
                    max: None,
                    std: None,
                }
            } else {
                let sum = values.iter().sum::<f64>();
                let mean = sum / count as f64;
                let variance = (count > 1).then(|| {
                    values
                        .iter()
                        .map(|value| (value - mean).powi(2))
                        .sum::<f64>()
                        / (count - 1) as f64
                });
                GroupedPropertyStats {
                    count,
                    sum: Some(sum),
                    mean: Some(mean),
                    min: Some(values.iter().copied().fold(f64::INFINITY, f64::min)),
                    max: Some(values.iter().copied().fold(f64::NEG_INFINITY, f64::max)),
                    std: variance.map(f64::sqrt),
                }
            };
            (key, stats)
        })
        .collect()
}

#[derive(Debug)]
pub struct PropertyStats {
    pub parent_idx: Option<NodeIndex>,
    pub parent_type: Option<String>,
    pub parent_title: Option<Value>,
    pub parent_id: Option<Value>,
    pub property_name: String,
    pub value_type: String,
    pub count: usize,
    pub children: usize,
    pub sum: Option<f64>,
    pub avg: Option<f64>,
    pub min: Option<f64>,
    pub max: Option<f64>,
    pub valid_count: usize,
    pub is_numeric: bool,
}

impl PropertyStats {
    fn new(parent_idx: Option<NodeIndex>, graph: &DirGraph, property: &str) -> Self {
        let (parent_type, parent_title, parent_id) = parent_idx
            .and_then(|idx| graph.node_view(idx))
            .map(|node| {
                (
                    Some(node.node_type_str(&graph.interner).to_string()),
                    Some(node.title().into_owned()),
                    Some(node.id().into_owned()),
                )
            })
            .unwrap_or((None, None, None));

        PropertyStats {
            parent_idx,
            parent_type,
            parent_title,
            parent_id,
            property_name: property.to_string(),
            value_type: "unknown".to_string(),
            count: 0,
            children: 0,
            sum: None,
            avg: None,
            min: None,
            max: None,
            valid_count: 0,
            is_numeric: false,
        }
    }

    fn finalize(&mut self) {
        if self.is_numeric {
            if let Some(sum) = self.sum {
                if self.valid_count > 0 {
                    self.avg = Some(sum / self.valid_count as f64);
                }
            }
        }
    }
}

pub fn calculate_property_stats(
    graph: &DirGraph,
    pairs: &[ParentChildPair],
    property: &str,
) -> Vec<PropertyStats> {
    // Arena guard: disk-backed node/edge reads materialize into the query
    // arena (protocol in disk/graph.rs); no-op on memory/mapped.
    let _arena_guard = graph.graph.begin_query();
    pairs
        .iter()
        .map(|pair| {
            let mut stats = PropertyStats::new(pair.parent, graph, property);
            calculate_stats_for_nodes(graph, &pair.children, property, &mut stats);
            stats.finalize();
            stats
        })
        .collect()
}

fn calculate_stats_for_nodes(
    graph: &DirGraph,
    nodes: &[NodeIndex],
    property: &str,
    stats: &mut PropertyStats,
) {
    stats.count = nodes.len();
    stats.children = nodes.len();

    let mut found_numeric = false;
    let mut sum = 0.0;
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    let mut valid_numeric_count = 0;
    let mut seen_types = HashSet::new();

    for &node_idx in nodes {
        if let Some(node) = graph.node_view(node_idx) {
            if let Some(value) = resolved_stat_field(graph, node, property) {
                match &*value {
                    Value::Null => continue,
                    Value::String(s) if s.is_empty() => continue,
                    _ => {
                        stats.valid_count += 1;
                        seen_types.insert(match &*value {
                            Value::String(_) => "string",
                            Value::Int64(_) => "int64",
                            Value::Float64(_) => "float64",
                            Value::Boolean(_) => "boolean",
                            Value::DateTime(_) => "datetime",
                            Value::Timestamp(_) => "timestamp",
                            Value::UniqueId(_) => "unique_id",
                            Value::Point { .. } => "point",
                            Value::Duration { .. } => "duration",
                            Value::Null => "null",
                            Value::NodeRef(_) => "noderef",
                            // Phase A.1 — collection / graph-entity variants
                            // shouldn't appear as stored property values
                            // (they're query-result-time), but classify
                            // defensively if they do.
                            Value::List(_) => "list",
                            Value::Map(_) => "map",
                            Value::Node(_) => "node",
                            Value::Relationship(_) => "relationship",
                            Value::Path(_) => "path",
                        });
                    }
                }

                if let Some(num) = try_convert_to_float(&value) {
                    found_numeric = true;
                    sum += num;
                    min = min.min(num);
                    max = max.max(num);
                    valid_numeric_count += 1;
                }
            }
        }
    }

    // Set value type based on seen values
    stats.value_type = if seen_types.is_empty() {
        "null".to_string()
    } else if seen_types.len() == 1 {
        seen_types.into_iter().next().unwrap().to_string()
    } else {
        "mixed".to_string()
    };

    if found_numeric && valid_numeric_count > 0 {
        stats.is_numeric = true;
        stats.sum = Some(sum);
        stats.min = Some(min);
        stats.max = Some(max);
    } else {
        stats.is_numeric = false;
        stats.sum = None;
        stats.min = None;
        stats.max = None;
        stats.avg = None;
    }
}

fn try_convert_to_float(value: &Value) -> Option<f64> {
    match value {
        Value::Int64(i) => Some(*i as f64),
        Value::Float64(f) => Some(*f),
        Value::String(s) => s.parse::<f64>().ok(),
        Value::UniqueId(u) => Some(*u as f64),
        _ => None,
    }
}

/// Read `property` off `node` the way a filter on it would.
///
/// Both statistics entry points come through here, so the grouped and
/// ungrouped surfaces cannot disagree about what a field name means. The
/// resolution is [`NodeView::resolved_field`]'s — a type's `unique_id_field` /
/// `node_title_field` map onto the identity columns, then a stored property,
/// then the structural soft alias. `calculate_property_stats` used to read the
/// property map alone, so `.statistics()` on a type's own id/title column (in
/// any spelling) reported `valid_count = 0` and no numeric stats.
fn resolved_stat_field<'a>(
    graph: &'a DirGraph,
    node: crate::graph::storage::NodeView<'a>,
    property: &str,
) -> Option<Cow<'a, Value>> {
    let node_type = node.node_type_str(&graph.interner);
    let field = graph.resolve_alias(node_type, property);
    node.resolved_field(node_type, field, InternedKey::from_str(field))
}

#[cfg(test)]
mod identity_field_tests {
    use super::*;
    use crate::datatypes::DataFrame;
    use crate::graph::mutation::maintain::add_nodes;

    /// A type whose identity columns are named `term_id` / `term_name`, so
    /// `add_nodes` hoists them onto `NodeData.id` / `NodeData.title` and
    /// registers the alias. Nothing named `term_id`, `term_name`, `id` or
    /// `title` remains in the property map — a read that skips alias
    /// resolution finds nothing at all.
    fn aliased_graph() -> DirGraph {
        let rows: Vec<Vec<Value>> = (1..=5)
            .map(|i| vec![Value::Int64(i), Value::String(format!("term-{i}"))])
            .collect();
        let df =
            DataFrame::from_cypher_rows(vec!["term_id".to_string(), "term_name".to_string()], rows)
                .expect("fixture frame");
        let mut graph = DirGraph::new();
        add_nodes(
            &mut graph,
            df,
            "Term".to_string(),
            "term_id".to_string(),
            Some("term_name".to_string()),
            None,
        )
        .expect("fixture add_nodes");
        graph
    }

    fn stats_for(graph: &DirGraph, property: &str) -> PropertyStats {
        let children: Vec<_> = graph
            .type_indices
            .get("Term")
            .expect("Term type index")
            .iter()
            .collect();
        let pairs = vec![ParentChildPair {
            parent: None,
            children,
        }];
        calculate_property_stats(graph, &pairs, property)
            .pop()
            .expect("one group")
    }

    /// `.statistics("term_id")` and `.statistics("id")` must both see the
    /// identity column. Reading the property map alone reports
    /// `valid_count = 0` with no numeric stats — silently wrong, not an error.
    #[test]
    fn statistics_resolve_the_types_id_field() {
        let graph = aliased_graph();
        for spelling in ["term_id", "id"] {
            let stats = stats_for(&graph, spelling);
            assert_eq!(stats.count, 5, "{spelling}: node count");
            assert_eq!(
                stats.valid_count, 5,
                "{spelling}: every node carries the id field"
            );
            assert!(stats.is_numeric, "{spelling}: id values are integers");
            assert_eq!(stats.min, Some(1.0), "{spelling}: min");
            assert_eq!(stats.max, Some(5.0), "{spelling}: max");
            assert_eq!(stats.sum, Some(15.0), "{spelling}: sum");
            assert_eq!(stats.avg, Some(3.0), "{spelling}: avg");
        }
    }

    /// Same for the title field, which is non-numeric: the fix is visible as
    /// `valid_count` and the value type, not as min/max.
    #[test]
    fn statistics_resolve_the_types_title_field() {
        let graph = aliased_graph();
        for spelling in ["term_name", "title"] {
            let stats = stats_for(&graph, spelling);
            assert_eq!(stats.count, 5, "{spelling}: node count");
            assert_eq!(
                stats.valid_count, 5,
                "{spelling}: every node carries the title field"
            );
            assert_eq!(stats.value_type, "string", "{spelling}: value type");
        }
    }

    /// A genuinely absent property still reports nothing — the fix must not
    /// invent values.
    #[test]
    fn statistics_on_an_absent_property_stay_empty() {
        let graph = aliased_graph();
        let stats = stats_for(&graph, "no_such_property");
        assert_eq!(stats.count, 5);
        assert_eq!(stats.valid_count, 0);
        assert_eq!(stats.value_type, "null");
        assert_eq!(stats.min, None);
        assert_eq!(stats.max, None);
    }

    /// The grouped sibling resolves the same way — one graph, one field, two
    /// entry points, one answer.
    #[test]
    fn grouped_statistics_resolve_the_types_id_field() {
        let graph = aliased_graph();
        let indices: Vec<_> = graph
            .type_indices
            .get("Term")
            .expect("Term type index")
            .iter()
            .collect();
        let mut selection = CurrentSelection::new();
        selection
            .get_level_mut(0)
            .expect("root level")
            .add_selection(None, indices);
        let grouped =
            calculate_grouped_property_stats(&graph, &selection, "term_id", "term_name", None);
        assert_eq!(grouped.len(), 5, "one group per distinct title");
        let one = &grouped["term-1"];
        assert_eq!(one.count, 1);
        assert_eq!(one.min, Some(1.0));
        assert_eq!(one.max, Some(1.0));
    }
}

#[cfg(test)]
mod grouped_tests {
    use super::*;
    use crate::graph::session::{execute_mut, ExecuteOptions};

    #[test]
    fn grouped_statistics_preserve_empty_groups_and_sample_stddev() {
        let mut graph = DirGraph::new();
        let params = HashMap::new();
        execute_mut(
            &mut graph,
            "CREATE (:Item {id:1, title:'one', team:'A', score:1}), \
             (:Item {id:2, title:'two', team:'A', score:3}), \
             (:Item {id:3, title:'three', team:'B', score:'n/a'})",
            &ExecuteOptions::eager(&params),
        )
        .expect("fixture CREATE");
        let indices: Vec<_> = graph
            .type_indices
            .get("Item")
            .expect("Item type index")
            .iter()
            .collect();
        let mut selection = CurrentSelection::new();
        selection
            .get_level_mut(0)
            .expect("root level")
            .add_selection(None, indices);

        let stats = calculate_grouped_property_stats(&graph, &selection, "score", "team", None);
        assert_eq!(
            stats["A"],
            GroupedPropertyStats {
                count: 2,
                sum: Some(4.0),
                mean: Some(2.0),
                min: Some(1.0),
                max: Some(3.0),
                std: Some(2.0_f64.sqrt()),
            }
        );
        assert_eq!(stats["B"].count, 0);
        assert_eq!(stats["B"].sum, None);
    }
}
