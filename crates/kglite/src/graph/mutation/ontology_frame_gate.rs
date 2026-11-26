//! The node write gate for bulk loaders: a whole frame is judged against the
//! enforced ontology rules before the loader writes anything, so a refusal
//! leaves the graph as it was.
//!
//! Rows are judged on the node each will *leave* under the call's conflict
//! mode — incoming cells over stored ones for `update`/`sum`, stored over
//! incoming for `preserve`, incoming alone for `replace` or a new node — and
//! `skip` leaves an existing node unjudged because it is not written. The
//! verdicts come from the same [`TypeRules`] the Cypher statement-end judge
//! runs, so the two surfaces cannot disagree.
//!
//! A row marked `_provisional` is an auto-vivified relationship endpoint
//! holding only its id; its property rules are deferred exactly as NOT NULL
//! defers them, and the later row that promotes it is judged in full.

use std::collections::HashMap;

use petgraph::graph::NodeIndex;

use crate::datatypes::{DataFrame, Value};
use crate::graph::diagnostics::{Diagnostic, DiagnosticGroup};
use crate::graph::mutation::batch::ConflictHandling;
use crate::graph::ontology::node_gate::{Tally, TypeRules};
use crate::graph::ontology::predicates::stored_property_value;
use crate::graph::schema::PROVISIONAL_KEY;
use crate::graph::schema::{soft_alias_fallback, DirGraph, InternedKey, SoftAliasFallback};
use crate::graph::storage::GraphRead;

/// Where a frame keeps the identity of the nodes it loads.
pub(super) struct FrameShape<'a> {
    pub id_idx: usize,
    pub title_idx: usize,
    pub id_field: &'a str,
    pub title_field: &'a str,
    pub derived_titles: Option<&'a [Value]>,
}

/// Judge `df` as an `add_nodes` of `node_type` under `mode`. `Err` refuses
/// the whole call (typed violation parked on the graph); `Ok` carries the
/// `warn`-level findings as advisories for the call's report.
pub(super) fn gate_node_frame(
    graph: &mut DirGraph,
    node_type: &str,
    df: &DataFrame,
    shape: &FrameShape<'_>,
    mode: ConflictHandling,
) -> Result<Vec<Diagnostic>, String> {
    if !graph.ontology_node_gate {
        return Ok(Vec::new());
    }
    let rules = TypeRules::build(&graph.ontology, node_type);
    if rules.is_empty() {
        return Ok(Vec::new());
    }
    graph.build_id_index(node_type);
    let mut tally = Tally::default();
    {
        judge_rows(graph, node_type, df, shape, mode, &rules, &mut tally);
    }
    Ok(as_diagnostics(graph.settle_ontology_tally(tally)?))
}

/// `warn`-level lines as advisories for a loader report.
pub(super) fn as_diagnostics(lines: Vec<String>) -> Vec<Diagnostic> {
    lines
        .into_iter()
        .map(|line| Diagnostic::new(DiagnosticGroup::DataQuality, "ontology_warning", line))
        .collect()
}

/// Judge nodes about to receive property writes, on the node each will
/// become: the writes are laid over the stored node. `Err` refuses the whole
/// call; `Ok` carries the `warn`-level lines. For the writers that change
/// existing nodes (`update`, `store_as=`, `add_properties`).
pub(super) fn gate_node_updates<'v>(
    graph: &mut DirGraph,
    updates: impl IntoIterator<Item = (NodeIndex, Vec<(&'v str, &'v Value)>)>,
) -> Result<Vec<String>, String> {
    if !graph.ontology_node_gate {
        return Ok(Vec::new());
    }
    let mut tally = Tally::default();
    {
        let graph: &DirGraph = graph;
        let mut by_type: HashMap<InternedKey, Option<TypeRules>> = HashMap::new();
        for (idx, writes) in updates {
            let Some(view) = graph.graph.node_view(idx) else {
                continue;
            };
            let type_key = view.node_type();
            let node_type = graph.interner.resolve(type_key);
            let rules = by_type.entry(type_key).or_insert_with(|| {
                let rules = TypeRules::build(&graph.ontology, node_type);
                (!rules.is_empty()).then_some(rules)
            });
            let Some(rules) = rules else {
                continue;
            };
            rules.judge_label(node_type, &mut tally);
            let stub = matches!(
                view.get(InternedKey::from_str(PROVISIONAL_KEY)).as_deref(),
                Some(Value::Boolean(true))
            );
            if rules.has_property_rules() && !stub {
                rules.judge_values(
                    node_type,
                    |property| {
                        let field = graph.resolve_alias(node_type, property);
                        let written = writes
                            .iter()
                            .find(|(name, _)| graph.resolve_alias(node_type, name) == field);
                        match written {
                            Some((_, value)) => Some((*value).clone()),
                            None => stored_property_value(graph, &view, node_type, property),
                        }
                    },
                    &mut tally,
                );
            }
            if tally.is_refused() {
                break;
            }
        }
    }
    graph.settle_ontology_tally(tally)
}

fn judge_rows(
    graph: &DirGraph,
    node_type: &str,
    df: &DataFrame,
    shape: &FrameShape<'_>,
    mode: ConflictHandling,
    rules: &TypeRules,
    tally: &mut Tally,
) {
    let provisional_col = df.get_column_index(PROVISIONAL_KEY);
    for row in 0..df.row_count() {
        // A row the loader skips is not written, so it is not judged.
        let Some(id) = df.get_value_by_index(row, shape.id_idx) else {
            continue;
        };
        if matches!(id, Value::Null) {
            continue;
        }
        let existing = graph.id_indices.lookup(node_type, &id);
        if existing.is_some() && mode == ConflictHandling::Skip {
            continue;
        }
        rules.judge_label(node_type, tally);
        if tally.is_refused() {
            return;
        }
        let stub = provisional_col
            .and_then(|col| df.get_value_by_index(row, col))
            .is_some_and(|v| matches!(v, Value::Boolean(true)));
        if !rules.has_property_rules() || stub {
            continue;
        }
        let title = shape
            .derived_titles
            .and_then(|titles| titles.get(row).cloned())
            .or_else(|| df.get_value_by_index(row, shape.title_idx))
            .unwrap_or(Value::Null);
        let title_supplied = shape.title_field != shape.id_field;
        rules.judge_values(
            node_type,
            |property| {
                let field = graph.resolve_alias(node_type, property);
                let incoming = if field == "id" || property == shape.id_field {
                    Some(id.clone())
                } else if field == "title" || property == shape.title_field {
                    // Without a title column the loader mints the title from
                    // the id, which supplies nothing.
                    title_supplied.then(|| title.clone())
                } else {
                    df.get_column_index(property)
                        .and_then(|col| df.get_value_by_index(row, col))
                };
                let incoming = incoming.filter(|v| !matches!(v, Value::Null));
                let stored =
                    || existing.and_then(|idx| stored_value(graph, idx, node_type, property));
                let merged = match (existing, mode) {
                    (None, _) | (Some(_), ConflictHandling::Replace) => incoming,
                    (Some(_), ConflictHandling::Preserve) => stored().or(incoming),
                    (Some(_), _) => incoming.or_else(stored),
                };
                // The structural fallback a stored node's read applies, minus
                // the title one: a minted title never satisfies a rule.
                merged.or_else(|| match soft_alias_fallback(field) {
                    Some(SoftAliasFallback::TypeString) => {
                        Some(Value::String(node_type.to_string()))
                    }
                    _ => None,
                })
            },
            tally,
        );
        if tally.is_refused() {
            return;
        }
    }
}

/// A stored node's non-null value for `property`, read as a stored-node
/// predicate reads it.
fn stored_value(
    graph: &DirGraph,
    idx: petgraph::graph::NodeIndex,
    node_type: &str,
    property: &str,
) -> Option<Value> {
    let view = graph.graph.node_view(idx)?;
    stored_property_value(graph, &view, node_type, property).filter(|v| !matches!(v, Value::Null))
}
