//! MERGE's match arm: does the pattern already exist?
//!
//! Only the *lookup* lives here — a miss is reported to [`super::write`]'s
//! `execute_merge`, which then runs the pattern through CREATE.

use super::identity_fields::merge_expected_props;
use super::CypherExecutor;
use crate::datatypes::values::Value;
use crate::graph::languages::cypher::ast::{
    CreateEdgeDirection, CreateEdgePattern, CreateElement, CreateNodePattern, CreatePattern,
};
use crate::graph::languages::cypher::result::{EdgeBinding, ResultRow};
use crate::graph::schema::{canonical_id, CompositeValue, DirGraph, EdgeData, InternedKey};
use crate::graph::storage::GraphRead;
use petgraph::graph::NodeIndex;

/// Every binding the pattern already has: one entry per matching node or
/// relationship, in the order the store yields them. Empty when the pattern
/// does not exist (the caller then CREATEs it). MERGE returns one row per
/// match, so a caller that stopped at the first would drop the rest.
pub(super) fn try_match_merge_pattern(
    graph: &DirGraph,
    pattern: &CreatePattern,
    row: &ResultRow,
    ctx: &super::write::WriteClauseCtx<'_>,
) -> Result<Vec<ResultRow>, String> {
    let executor = super::write::row_evaluator(graph, ctx);

    match pattern.elements.len() {
        // Node-only MERGE: (var:Label {key: val, ...})
        1 => match &pattern.elements[0] {
            CreateElement::Node(node_pat) => match_node_pattern(graph, &executor, node_pat, row),
            _ => Err("MERGE pattern must start with a node".to_string()),
        },
        // Relationship MERGE: (a)-[r:TYPE]->(b)
        3 => match_relationship_pattern(graph, &executor, pattern, row),
        _ => Err("MERGE supports single-node or single-edge patterns only".to_string()),
    }
}

fn match_node_pattern(
    graph: &DirGraph,
    executor: &CypherExecutor<'_>,
    node_pat: &CreateNodePattern,
    row: &ResultRow,
) -> Result<Vec<ResultRow>, String> {
    // If variable is already bound from prior MATCH — or carried in as a
    // projected node value — it's already matched. Without the projected arm
    // `UNWIND collect(n) AS x MERGE (x)` reached the CREATE arm and duplicated
    // the node it was handed.
    if let Some(ref var) = node_pat.variable {
        if let Some(existing_idx) =
            row.node_bindings
                .get(var)
                .copied()
                .or(super::projected_targets::projected_node_target(
                    graph, row, var,
                )?)
        {
            if graph.graph.node_view(existing_idx).is_some() {
                let mut result_row = ResultRow::new();
                result_row.node_bindings.insert(var.clone(), existing_idx);
                return Ok(vec![result_row]);
            }
        }
    }

    let Some(label) = node_pat.label.as_deref() else {
        return match_unlabelled_node(
            graph,
            node_pat,
            &merge_expected_props(executor, node_pat, row, graph)?,
        );
    };

    // The id/property/composite indexes and `type_indices` are keyed by
    // PRIMARY type. If `label` also occurs as a secondary label on some node,
    // those structures miss the secondary-labelled candidates and would
    // falsely report "no match" → MERGE creates a duplicate. In that case skip
    // the index short-circuits and scan the full primary∪secondary candidate
    // set (`nodes_with_label`). The common case (label has no secondary
    // occurrences) keeps every index fast path.
    let label_has_secondary = graph.has_secondary_labels
        && graph
            .secondary_label_index
            .contains_key(&InternedKey::from_str(label));

    let expected_props = merge_expected_props(executor, node_pat, row, graph)?;
    // `MERGE (n:A:B {…})` matches only a node carrying every label, as `MATCH`
    // does; the indexes and `nodes_with_label` answer for the primary alone.
    let wanted = Candidate {
        props: &expected_props,
        extra_labels: &node_pat.extra_labels,
    };

    if !label_has_secondary {
        match probe_node_indexes(graph, label, &wanted) {
            IndexProbe::Matched(found) => return Ok(node_result_rows(node_pat, &found)),
            IndexProbe::NoMatch => return refuse_duplicate_id(graph, label, &wanted),
            IndexProbe::Unindexed => {}
        }
    }

    // Fall back to linear scan (no index covers the pattern, or `label` has
    // secondary occurrences). `nodes_with_label` unions primary + secondary
    // candidates (and is the identical `type_indices` clone when no secondary
    // labels exist).
    let found: Vec<NodeIndex> = graph
        .nodes_with_label(label)
        .into_iter()
        .filter(|&idx| wanted.matches(graph, idx))
        .collect();
    if found.is_empty() {
        return refuse_duplicate_id(graph, label, &wanted);
    }
    Ok(node_result_rows(node_pat, &found))
}

/// A label-less node pattern matches every node, whatever its type, as
/// `MATCH (n {…})` does. No per-type index applies, so this scans; a miss falls
/// to the CREATE arm, which types the new node `Node`.
fn match_unlabelled_node(
    graph: &DirGraph,
    node_pat: &CreateNodePattern,
    expected_props: &[(&str, Value)],
) -> Result<Vec<ResultRow>, String> {
    let wanted = Candidate {
        props: expected_props,
        extra_labels: &node_pat.extra_labels,
    };
    let found: Vec<NodeIndex> = graph
        .graph
        .node_indices()
        .filter(|&idx| wanted.matches(graph, idx))
        .collect();
    if found.is_empty() {
        return refuse_duplicate_id(graph, "Node", &wanted);
    }
    Ok(node_result_rows(node_pat, &found))
}

/// The miss a node MERGE reports once no node matched: `Ok(vec![])` — create
/// it, an empty list — unless the pattern names an id a `label` node already holds. The id
/// is the node's identity, so creating would give the type a second node
/// under it; that node matched on its id but not on the pattern's other
/// labels or properties, and the refusal names which.
fn refuse_duplicate_id(
    graph: &DirGraph,
    label: &str,
    wanted: &Candidate<'_>,
) -> Result<Vec<ResultRow>, String> {
    let Some((_, id)) = wanted.props.iter().find(|(key, _)| *key == "id") else {
        return Ok(Vec::new());
    };
    let Some(existing) = graph.lookup_by_id_readonly(label, id) else {
        return Ok(Vec::new());
    };
    let id = match id {
        Value::String(text) => format!("'{text}'"),
        other => crate::graph::core::value_operations::format_value_compact(other),
    };
    let missing: Vec<&str> = wanted
        .extra_labels
        .iter()
        .filter(|l| !graph.node_has_label(existing, InternedKey::from_str(l)))
        .map(String::as_str)
        .collect();
    let lead = format!("MERGE would create a second :{label} node with id {id}");
    if let Some(first) = missing.first() {
        let lacks: Vec<String> = missing.iter().map(|l| format!(":{l}")).collect();
        return Err(format!(
            "{lead}; the existing node lacks label {} — match on the id and add the label \
             (`MERGE (n:{label} {{id: {id}}}) SET n:{first}`), or use ON MATCH SET",
            lacks.join(", ")
        ));
    }
    let differing: Vec<&str> = wanted
        .props
        .iter()
        .filter(|(key, value)| {
            *key != "id" && !node_matches_all(graph, existing, &[(*key, value.clone())])
        })
        .map(|(key, _)| *key)
        .collect();
    let (named, first) = match differing.first() {
        Some(first) => (differing.join(", "), *first),
        None => ("its properties".to_string(), "property"),
    };
    Err(format!(
        "{lead}; the existing node differs in {named} — match on the id alone and set the \
         properties (`MERGE (n:{label} {{id: {id}}}) SET n.{first} = …`), or use ON MATCH SET"
    ))
}

/// What the index short-circuits settled. `NoMatch` is authoritative — the
/// index that answered covers the whole pattern, so no scan can add a match —
/// while `Unindexed` means no index applies and the caller must scan.
enum IndexProbe {
    Matched(Vec<NodeIndex>),
    NoMatch,
    Unindexed,
}

/// What a MERGE node pattern requires of an existing node: every property
/// value, and every label after the first.
struct Candidate<'a> {
    props: &'a [(&'a str, Value)],
    extra_labels: &'a [String],
}

impl Candidate<'_> {
    fn matches(&self, graph: &DirGraph, idx: NodeIndex) -> bool {
        node_matches_all(graph, idx, self.props) && self.carries_labels(graph, idx)
    }

    /// The probe verdict for an index's candidate list: every candidate that
    /// matches, or the authoritative `NoMatch`.
    fn probe_result(&self, graph: &DirGraph, candidates: &[NodeIndex]) -> IndexProbe {
        let found: Vec<NodeIndex> = candidates
            .iter()
            .copied()
            .filter(|&idx| self.matches(graph, idx))
            .collect();
        if found.is_empty() {
            IndexProbe::NoMatch
        } else {
            IndexProbe::Matched(found)
        }
    }

    /// The pattern's values for `names`, in that order; `None` when the
    /// pattern leaves one of them unnamed or names it null.
    fn tuple_for(&self, names: &[String]) -> Option<Vec<Value>> {
        names
            .iter()
            .map(|name| {
                self.props
                    .iter()
                    .find(|(key, _)| key == name)
                    .map(|(_, value)| value.clone())
                    .filter(|value| !matches!(value, Value::Null))
            })
            .collect()
    }

    fn carries_labels(&self, graph: &DirGraph, idx: NodeIndex) -> bool {
        self.extra_labels
            .iter()
            .all(|label| graph.node_has_label(idx, InternedKey::from_str(label)))
    }
}

fn probe_node_indexes(graph: &DirGraph, label: &str, wanted: &Candidate<'_>) -> IndexProbe {
    let expected_props = wanted.props;
    // 1. If pattern contains "id" property, use O(1) id_index lookup
    if let Some((_, id_value)) = expected_props.iter().find(|(k, _)| *k == "id") {
        if let Some(idx) = graph.lookup_by_id_readonly(label, id_value) {
            // The id lookup already matched the only property; skip re-reading it.
            let props_match =
                expected_props.len() == 1 || node_matches_all(graph, idx, expected_props);
            if props_match && wanted.carries_labels(graph, idx) {
                return IndexProbe::Matched(vec![idx]);
            }
        }
        return IndexProbe::NoMatch;
    }

    // 1b. A declared unique tuple the pattern names in full has one holder.
    if let Some(probe) = probe_unique_index(graph, label, wanted) {
        return probe;
    }

    // 2. Single non-id property: try property index.
    // Probed under the pattern's own spelling — `lookup_by_index` resolves a
    // type's registered title/id alias itself. The old `name`/`title` →
    // `title` remap was not that: on a type whose `name` is an ordinary stored
    // property distinct from the title, it asked the title index a question
    // about `name` and MERGE created a duplicate of the node it missed.
    if expected_props.len() == 1 {
        let (key, ref value) = expected_props[0];
        if let Some(candidates) = graph.lookup_by_index(label, key, value) {
            return wanted.probe_result(graph, &candidates);
        }
        // No index — fall through to linear scan
    }

    // 3. Multi-property: try composite index
    if expected_props.len() >= 2 {
        // Composite lookup excludes id/name/title: they use special storage,
        // not ordinary property columns.
        let mut indexable: Vec<(&str, &Value)> = expected_props
            .iter()
            .filter(|(k, _)| *k != "id" && *k != "name" && *k != "title")
            .map(|(k, v)| (*k, v))
            .collect();
        if indexable.len() >= 2 {
            indexable.sort_by(|a, b| a.0.cmp(b.0));
            let names: Vec<String> = indexable.iter().map(|(k, _)| k.to_string()).collect();
            let values: Vec<Value> = indexable.iter().map(|(_, v)| (*v).clone()).collect();
            if let Some(candidates) = graph.lookup_by_composite_predicate(label, &names, &values) {
                return wanted.probe_result(graph, &candidates);
            }
        }
    }

    IndexProbe::Unindexed
}

/// The verdict of a unique constraint on `label` whose every property the
/// pattern names, or `None` when no such constraint answers. The index keys
/// tuples by exact `Value`, so a hit is one node (kept only if the whole
/// pattern matches it); a miss is authoritative only for text and boolean
/// values, where exact equality is Cypher equality. An integer or float
/// pattern value may equal a stored value of the other numeric type, which the
/// index files apart, so a numeric miss is left to the scan.
fn probe_unique_index(graph: &DirGraph, label: &str, wanted: &Candidate<'_>) -> Option<IndexProbe> {
    if !graph.has_unique_constraints() {
        return None;
    }
    for key in &graph.unique_constraint_keys {
        if key.0 != label {
            continue;
        }
        let Some(tuple) = wanted.tuple_for(&key.1) else {
            continue;
        };
        if let Some(idx) = graph.unique_occupant(key, &CompositeValue(tuple.clone())) {
            if wanted.matches(graph, idx) {
                return Some(IndexProbe::Matched(vec![idx]));
            }
        } else if tuple
            .iter()
            .all(|v| matches!(v, Value::String(_) | Value::Boolean(_)))
        {
            return Some(IndexProbe::NoMatch);
        }
    }
    None
}

fn node_matches_all(graph: &DirGraph, idx: NodeIndex, props: &[(&str, Value)]) -> bool {
    let Some(node) = graph.graph.node_view(idx) else {
        return false;
    };
    let node_type = node.node_type_str(&graph.interner);
    props.iter().all(|(key, expected)| {
        let value = node.resolved_field(node_type, key, InternedKey::from_str(key));
        value.as_deref().is_some_and(|value| {
            if *key == "id" {
                // Identity matching keeps the id index's normalization — a
                // loaded `UniqueId(1)` is the id `1` — so a pattern naming the
                // id beside other properties matches the node the id alone
                // does; ordinary properties use Cypher predicate equality.
                canonical_id(value) == canonical_id(expected)
            } else {
                crate::graph::core::filtering::values_equal(value, expected)
            }
        })
    })
}

fn node_result_rows(node_pat: &CreateNodePattern, found: &[NodeIndex]) -> Vec<ResultRow> {
    found
        .iter()
        .map(|&idx| {
            let mut result_row = ResultRow::new();
            if let Some(ref var) = node_pat.variable {
                result_row.node_bindings.insert(var.clone(), idx);
            }
            result_row
        })
        .collect()
}

fn match_relationship_pattern(
    graph: &DirGraph,
    executor: &CypherExecutor<'_>,
    pattern: &CreatePattern,
    row: &ResultRow,
) -> Result<Vec<ResultRow>, String> {
    let source_var = create_node_variable(&pattern.elements[0]);
    let target_var = create_node_variable(&pattern.elements[2]);

    let source_idx = merge_endpoint(graph, row, source_var)?
        .ok_or("MERGE path: source node must be bound by prior MATCH")?;
    let target_idx = merge_endpoint(graph, row, target_var)?
        .ok_or("MERGE path: target node must be bound by prior MATCH")?;

    let CreateElement::Edge(edge_pat) = &pattern.elements[1] else {
        return Err("Expected edge in MERGE path pattern".to_string());
    };

    let (actual_src, actual_tgt) = match edge_pat.direction {
        CreateEdgeDirection::Outgoing => (source_idx, target_idx),
        CreateEdgeDirection::Incoming => (target_idx, source_idx),
    };

    let interned_ct = InternedKey::from_str(&edge_pat.connection_type);
    // The pattern's relationship properties are part of the match, exactly as
    // the node arm's are. Without them, parallel members between the same
    // endpoints are indistinguishable and `edges_directed(..).find(..)` binds
    // whichever one adjacency yields first — so `MERGE (a)-[r:T {k:0}]->(b)`
    // could bind the `{k:1}` member and run ON MATCH SET against it, and a
    // pattern matching no member never reached the CREATE arm.
    let expected_props = merge_expected_edge_props(executor, edge_pat, row, graph)?;
    Ok(graph
        .graph
        .edges_directed(actual_src, petgraph::Direction::Outgoing)
        .filter(|e| {
            e.target() == actual_tgt
                && e.connection_type() == interned_ct
                && edge_matches_all(&e.weight(), &expected_props)
        })
        .map(|edge_ref| {
            let mut result_row = ResultRow::new();
            if let Some(ref var) = edge_pat.variable {
                result_row.edge_bindings.insert(
                    var.clone(),
                    EdgeBinding {
                        incarnation: None,
                        source: actual_src,
                        target: actual_tgt,
                        edge_index: edge_ref.id(),
                    },
                );
            }
            result_row
        })
        .collect())
}

/// One endpoint of a relationship MERGE: a live `node_bindings` entry, or the
/// node a projected value names. `None` means the endpoint is not bound at
/// all, which the caller reports — a relationship MERGE cannot invent one.
fn merge_endpoint(
    graph: &DirGraph,
    row: &ResultRow,
    var: Option<&str>,
) -> Result<Option<NodeIndex>, String> {
    let Some(var) = var else {
        return Ok(None);
    };
    if let Some(&idx) = row.node_bindings.get(var) {
        return Ok(Some(idx));
    }
    super::projected_targets::projected_node_target(graph, row, var)
}

fn create_node_variable(element: &CreateElement) -> Option<&str> {
    match element {
        CreateElement::Node(np) => np.variable.as_deref(),
        _ => None,
    }
}

/// The property values a relationship MERGE pattern expects, in the pattern's
/// own spelling.
///
/// The node counterpart ([`merge_expected_props`]) canonicalises each key
/// through the type's identity aliases; relationships have no identity columns,
/// so a key here always names an ordinary stored property. Endpoint references
/// are snapshotted for the same reason the node path snapshots them: stored
/// properties hold resolved values, so an unresolved one would never compare
/// equal to what is on the edge.
fn merge_expected_edge_props<'p>(
    executor: &CypherExecutor<'_>,
    edge_pat: &'p CreateEdgePattern,
    row: &ResultRow,
    graph: &DirGraph,
) -> Result<Vec<(&'p str, Value)>, String> {
    edge_pat
        .properties
        .iter()
        .map(|(key, expr)| {
            let mut value = executor.evaluate_expression(expr, row)?;
            crate::datatypes::value_depth::check_value(&value)?;
            crate::graph::session::snapshot_property_values(
                &graph.graph,
                std::iter::once(&mut value),
            );
            Ok((key.as_str(), value))
        })
        .collect()
}

/// Whether a candidate edge carries every property the MERGE pattern spelled,
/// under the same Cypher predicate equality the node arm applies. A property
/// the edge does not carry never matches.
fn edge_matches_all(edge: &EdgeData, expected: &[(&str, Value)]) -> bool {
    expected.iter().all(|(key, value)| {
        edge.get_property(key)
            .is_some_and(|stored| crate::graph::core::filtering::values_equal(stored, value))
    })
}
