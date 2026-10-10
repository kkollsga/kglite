//! Stored-title admission for the DataFrame batch paths.

use crate::datatypes::{DataFrame, Value};
use crate::graph::constraints::UniqueConstraintKey;
use crate::graph::diagnostics::{Diagnostic, DiagnosticGroup};
use crate::graph::features::temporal::{check_edge_load, coerce_edge_bounds, EmptyIntervals};
use crate::graph::mutation::endpoints::{
    resolve_endpoints, title_column_indices, ResolvedEndpoints,
};
use crate::graph::schema::{CompositeValue, DirGraph, PROVISIONAL_KEY};
use crate::graph::storage::GraphWrite;
use petgraph::graph::NodeIndex;
use std::collections::HashMap;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum PendingTitleOwner {
    Existing(NodeIndex),
    Deferred(String, Value),
}

struct PendingTitleUpdate {
    owner: PendingTitleOwner,
    node_type: String,
    node_idx: Option<NodeIndex>,
    id: Value,
    value: Value,
}

struct ConnectionTitleInput<'a> {
    frame: &'a DataFrame,
    matched: &'a [(usize, NodeIndex, NodeIndex)],
    deferred: &'a [(usize, Value, Value)],
    source_type: &'a str,
    target_type: &'a str,
    source_id_idx: usize,
    target_id_idx: usize,
    titles: &'a ConnectionTitles,
}

pub(super) struct ConnectionAdmissionFields<'a> {
    /// Judged against its declared validity interval, if any, before the
    /// endpoints are resolved.
    pub connection_type: &'a str,
    pub source_type: &'a str,
    pub source_id: &'a str,
    pub source_title: Option<&'a str>,
    pub target_type: &'a str,
    pub target_id: &'a str,
    pub target_title: Option<&'a str>,
}

pub(super) struct ConnectionTitles {
    source_idx: Option<usize>,
    target_idx: Option<usize>,
    source_derived: Option<Vec<Value>>,
    target_derived: Option<Vec<Value>>,
}

impl ConnectionTitles {
    fn value(&self, frame: &DataFrame, source: bool, row: usize) -> Option<Value> {
        let (index, derived) = if source {
            (self.source_idx, self.source_derived.as_deref())
        } else {
            (self.target_idx, self.target_derived.as_deref())
        };
        derived
            .and_then(|values| values.get(row).cloned())
            .or_else(|| index.and_then(|column| frame.get_value_by_index(row, column)))
    }

    pub(super) fn apply(
        &self,
        graph: &mut DirGraph,
        node_types: (&str, &str),
        endpoints: (NodeIndex, NodeIndex),
        row: usize,
        frame: &DataFrame,
    ) {
        if let Some(title) = self.source_idx.and_then(|_| self.value(frame, true, row)) {
            GraphWrite::set_node_title(&mut graph.graph, endpoints.0, title);
            crate::graph::index_freshness::write_hooks::note_property_written(
                graph,
                endpoints.0,
                node_types.0,
                Some("title"),
            );
        }
        if let Some(title) = self.target_idx.and_then(|_| self.value(frame, false, row)) {
            GraphWrite::set_node_title(&mut graph.graph, endpoints.1, title);
            crate::graph::index_freshness::write_hooks::note_property_written(
                graph,
                endpoints.1,
                node_types.1,
                Some("title"),
            );
        }
    }

    pub(super) fn refresh_indexes(
        &self,
        graph: &mut DirGraph,
        source_type: &str,
        target_type: &str,
    ) {
        if self.source_idx.is_some() {
            graph.refresh_indexes_for_type(source_type);
        }
        if self.target_idx.is_some() && target_type != source_type {
            graph.refresh_indexes_for_type(target_type);
        }
    }
}

pub(super) fn snapshot_node_titles(
    graph: &DirGraph,
    frame: &mut DataFrame,
    id_field: &str,
    title_field: &str,
    title_idx: usize,
) -> Option<Vec<Value>> {
    crate::graph::session::snapshot_dataframe_properties(&graph.graph, frame, &[id_field]);
    (title_field == id_field).then(|| snapshot_column(graph, frame, title_idx))
}

/// The column a node load takes its titles from, and whether those titles
/// overwrite an existing node's on update.
///
/// An explicit `node_title_field` wins. Without one, a type that already
/// declares a title field keeps it: the frame's `title` column is an ordinary
/// property there (reported by [`shadowed_identity_columns`]), and a frame
/// without the declared column titles new nodes by id and leaves existing
/// titles alone — titles never come from an undeclared column. Rebinding to
/// `title` made `n.title` and the declared spelling answer the `title` values
/// while `properties(n)` still held the declared column's.
///
/// On an undeclared type, a frame carrying a `title` column titles its nodes
/// from it: falling back to the id column there stored the `title` values as a
/// property every read shadows with the id, so `n.title` answered the id and
/// the column was silently lost.
pub(super) fn resolve_node_title_field(
    graph: &DirGraph,
    frame: &DataFrame,
    node_type: &str,
    unique_id_field: &str,
    node_title_field: Option<String>,
) -> (String, bool) {
    if let Some(field) = node_title_field {
        return (field, true);
    }
    match graph.title_field_aliases.get(node_type) {
        Some(declared) if frame.get_column_index(declared).is_some() => (declared.clone(), true),
        Some(_) => (unique_id_field.to_string(), false),
        None if unique_id_field != "title" && frame.get_column_index("title").is_some() => {
            ("title".to_string(), true)
        }
        None => (unique_id_field.to_string(), false),
    }
}

/// One advisory per `id` / `title` column the load cannot store readably,
/// because the identity field of that name comes from a different column.
///
/// `n.id` and `n.title` always read the node's identity fields, so a property
/// column spelled `id` or `title` beside a different `unique_id_field` /
/// title column is stored but answers nowhere — a silent loss the caller
/// should hear about. A column that is the *other* identity field's source —
/// an `id` column the title is read from, or a `title` column the id is read
/// from — answers through that field, so it is not reported.
///
/// On a type with a declared title field, the advice differs: passing
/// `node_title_field='title'` would not move the declared spelling, so the
/// column is named as the ordinary property it is stored as.
pub(super) fn shadowed_identity_columns(
    graph: &DirGraph,
    frame: &DataFrame,
    node_type: &str,
    unique_id_field: &str,
    title_field: &str,
) -> Vec<Diagnostic> {
    let declared_title = graph.title_field_aliases.get(node_type).map(String::as_str);
    [
        ("id", unique_id_field, title_field, "unique_id_field"),
        ("title", title_field, unique_id_field, "node_title_field"),
    ]
    .into_iter()
    .filter(|(field, source, other, _)| {
        source != field && other != field && frame.get_column_index(field).is_some()
    })
    .map(|(field, source, _, parameter)| {
        let message = if field == "title" && declared_title.is_some() {
            let declared = declared_title.unwrap_or_default();
            format!(
                "add_nodes: node type '{node_type}' declares '{declared}' as its title \
                 field, so its 'title' column is not readable — it is stored as an \
                 ordinary property, and n.title answers with the '{source}' value. \
                 Rename the 'title' column to keep it readable."
            )
        } else {
            format!(
                "add_nodes: node type '{node_type}' takes its {field} from column \
                 '{source}', so its '{field}' column is not readable — n.{field} answers \
                 with the '{source}' value. Rename the '{field}' column, or pass \
                 {parameter}='{field}'."
            )
        };
        Diagnostic::new(
            DiagnosticGroup::DataShape,
            "identity_column_shadowed",
            message,
        )
    })
    .collect()
}

pub(super) fn node_property_columns(
    frame: &DataFrame,
    id_field: &str,
    title_field: &str,
) -> Vec<(String, usize)> {
    frame
        .get_column_names()
        .into_iter()
        .filter(|name| name != id_field && name != title_field)
        .filter_map(|name| frame.get_column_index(&name).map(|index| (name, index)))
        .collect()
}

pub(super) fn prepare_connection_admission(
    graph: &mut DirGraph,
    frame: &mut DataFrame,
    fields: ConnectionAdmissionFields<'_>,
) -> Result<(ResolvedEndpoints, ConnectionTitles, EmptyIntervals), String> {
    coerce_edge_bounds(graph, fields.connection_type, fields.source_type, frame);
    let empty = check_edge_load(graph, fields.connection_type, fields.source_type, frame)?;
    let source_id_idx = frame
        .get_column_index(fields.source_id)
        .ok_or_else(|| format!("Source ID column '{}' not found", fields.source_id))?;
    let target_id_idx = frame
        .get_column_index(fields.target_id)
        .ok_or_else(|| format!("Target ID column '{}' not found", fields.target_id))?;
    let (source_idx, target_idx) =
        title_column_indices(frame, fields.source_title, fields.target_title);

    crate::graph::session::snapshot_dataframe_properties(
        &graph.graph,
        frame,
        &[fields.source_id, fields.target_id],
    );
    let titles = ConnectionTitles {
        source_idx,
        target_idx,
        source_derived: source_idx
            .filter(|index| *index == source_id_idx || *index == target_id_idx)
            .map(|index| snapshot_column(graph, frame, index)),
        target_derived: target_idx
            .filter(|index| *index == source_id_idx || *index == target_id_idx)
            .map(|index| snapshot_column(graph, frame, index)),
    };
    let resolved = resolve_endpoints(
        graph,
        frame,
        fields.source_type,
        fields.target_type,
        source_id_idx,
        target_id_idx,
    )?;
    gate_connection_title_updates(
        graph,
        ConnectionTitleInput {
            frame,
            matched: &resolved.matched,
            deferred: &resolved.deferred,
            source_type: fields.source_type,
            target_type: fields.target_type,
            source_id_idx,
            target_id_idx,
            titles: &titles,
        },
    )?;
    Ok((resolved, titles, empty))
}

fn snapshot_column(graph: &DirGraph, frame: &DataFrame, column: usize) -> Vec<Value> {
    let mut values: Vec<Value> = (0..frame.row_count())
        .map(|row| frame.get_value_by_index(row, column).unwrap_or(Value::Null))
        .collect();
    crate::graph::session::snapshot_property_values(&graph.graph, values.iter_mut());
    values
}

fn gate_connection_title_updates(
    graph: &mut DirGraph,
    input: ConnectionTitleInput<'_>,
) -> Result<(), String> {
    if input.titles.source_idx.is_none() && input.titles.target_idx.is_none() {
        return Ok(());
    }
    let updates = collect_connection_title_updates(&input);
    let mut batch_claims = HashMap::new();
    for update in updates {
        if let Some(shape) = graph.shape_for(&update.node_type, "title") {
            shape.check("title", &update.value)?;
        }
        gate_pending_title_update(graph, &update, &mut batch_claims)?;
    }
    Ok(())
}

fn collect_connection_title_updates(input: &ConnectionTitleInput<'_>) -> Vec<PendingTitleUpdate> {
    let mut updates = Vec::<PendingTitleUpdate>::new();
    let mut positions = HashMap::<PendingTitleOwner, usize>::new();
    let mut remember = |update: PendingTitleUpdate| {
        if let Some(position) = positions.get(&update.owner).copied() {
            updates[position] = update;
        } else {
            positions.insert(update.owner.clone(), updates.len());
            updates.push(update);
        }
    };
    let value_at = |row: usize, column: usize| {
        input
            .frame
            .get_value_by_index(row, column)
            .unwrap_or(Value::Null)
    };

    for (row, source, target) in input.matched {
        if input.titles.source_idx.is_some() {
            remember(PendingTitleUpdate {
                owner: PendingTitleOwner::Existing(*source),
                node_type: input.source_type.to_string(),
                node_idx: Some(*source),
                id: value_at(*row, input.source_id_idx),
                value: input
                    .titles
                    .value(input.frame, true, *row)
                    .unwrap_or(Value::Null),
            });
        }
        if input.titles.target_idx.is_some() {
            remember(PendingTitleUpdate {
                owner: PendingTitleOwner::Existing(*target),
                node_type: input.target_type.to_string(),
                node_idx: Some(*target),
                id: value_at(*row, input.target_id_idx),
                value: input
                    .titles
                    .value(input.frame, false, *row)
                    .unwrap_or(Value::Null),
            });
        }
    }
    for (row, source_id, target_id) in input.deferred {
        if input.titles.source_idx.is_some() {
            remember(PendingTitleUpdate {
                owner: PendingTitleOwner::Deferred(
                    input.source_type.to_string(),
                    source_id.clone(),
                ),
                node_type: input.source_type.to_string(),
                node_idx: None,
                id: source_id.clone(),
                value: input
                    .titles
                    .value(input.frame, true, *row)
                    .unwrap_or(Value::Null),
            });
        }
        if input.titles.target_idx.is_some() {
            remember(PendingTitleUpdate {
                owner: PendingTitleOwner::Deferred(
                    input.target_type.to_string(),
                    target_id.clone(),
                ),
                node_type: input.target_type.to_string(),
                node_idx: None,
                id: target_id.clone(),
                value: input
                    .titles
                    .value(input.frame, false, *row)
                    .unwrap_or(Value::Null),
            });
        }
    }
    updates
}

fn gate_pending_title_update(
    graph: &mut DirGraph,
    update: &PendingTitleUpdate,
    batch_claims: &mut HashMap<(UniqueConstraintKey, CompositeValue), PendingTitleOwner>,
) -> Result<(), String> {
    let claims = if let Some(node_idx) = update.node_idx {
        graph
            .plan_property_write(&update.node_type, node_idx, "title", Some(&update.value))
            .map(|plan| plan.claim)
            .map_err(|violation| violation.to_string())?
    } else {
        let typed = graph.check_property_types(&update.node_type, |property| {
            deferred_title_field(graph, update, property)
        });
        if let Err(violation) = typed {
            return Err(graph.record_constraint_violation(*violation));
        }
        let required = graph.check_required_fields(&update.node_type, |property| {
            deferred_title_field(graph, update, property)
        });
        if let Err(violation) = required {
            return Err(graph.record_constraint_violation(*violation));
        }
        let claims = graph.unique_claims(&update.node_type, |property| {
            deferred_title_field(graph, update, property)
        });
        graph
            .check_unique_claims(&claims, None)
            .map_err(|violation| graph.record_constraint_violation(*violation))?;
        claims
    };
    for claim in claims {
        let key = (claim.key.clone(), claim.value.clone());
        if let Some(owner) = batch_claims.insert(key, update.owner.clone()) {
            if owner != update.owner {
                let violation = graph.unique_batch_conflict(&claim);
                return Err(graph.record_constraint_violation(violation));
            }
        }
    }
    Ok(())
}

fn deferred_title_field(
    graph: &DirGraph,
    update: &PendingTitleUpdate,
    property: &str,
) -> Option<Value> {
    match graph.resolve_alias(&update.node_type, property) {
        "id" => Some(update.id.clone()),
        "title" => (!matches!(update.value, Value::Null)).then(|| update.value.clone()),
        PROVISIONAL_KEY => Some(Value::Boolean(true)),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use crate::datatypes::{DataFrame, Value};
    use crate::graph::mutation::maintain::add_nodes;
    use crate::graph::schema::DirGraph;

    fn load(columns: &[&str], row: Vec<Value>, id: &str, title: Option<&str>) -> Vec<String> {
        let mut graph = DirGraph::new();
        let names = columns.iter().map(|c| c.to_string()).collect();
        let df = DataFrame::from_cypher_rows(names, vec![row]).unwrap();
        let report = add_nodes(
            &mut graph,
            df,
            "P".into(),
            id.into(),
            title.map(str::to_string),
            None,
        )
        .unwrap();
        report
            .diagnostics
            .iter()
            .filter(|d| d.kind == "identity_column_shadowed")
            .map(|d| d.message.clone())
            .collect()
    }

    #[test]
    fn a_shadowed_id_or_title_column_is_reported() {
        let s = |v: &str| Value::String(v.into());
        let shadowed = load(
            &["pid", "id", "name", "title"],
            vec![Value::Int64(10), Value::Int64(99), s("Nan"), s("Ann")],
            "pid",
            Some("name"),
        );
        assert_eq!(shadowed.len(), 2, "{shadowed:?}");
        assert!(shadowed[0].contains("its 'id' column is not readable"));
        assert!(shadowed[1].contains("its 'title' column is not readable"));
        // With no title field named, the title column is the title.
        let defaulted = load(
            &["id", "title"],
            vec![Value::Int64(1), s("Ann")],
            "id",
            None,
        );
        assert!(defaulted.is_empty(), "{defaulted:?}");
    }
}
