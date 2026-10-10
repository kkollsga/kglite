//! The row check a write onto a declared type runs: the rule a declaration
//! validates its stored rows by ([`validate::check_row`]) — every bound NULL,
//! a date, a datetime or an ISO string, and no interval inverted — applied to
//! the rows a load or a Cypher write leaves, and worded as the declaration
//! words it. A row whose interval is empty (`half_open`, or `closed` with `empty_when`) is written, and
//! counted into the one warning ([`EmptyIntervals`]) its load or statement
//! reports. A type with no declaration costs one map lookup.
//!
//! A loader row is judged before anything is written, by the values it will
//! leave under its conflict mode; a Cypher `CREATE` before its insert; a
//! Cypher `SET` once its clause has applied every item, so `SET n.vf = …,
//! n.vt = …` is judged as the pair it writes, and the statement's rollback
//! undoes a refused one.

use petgraph::graph::{EdgeIndex, NodeIndex};

use super::validate::{self, check_row, Emptiness, EmptyIntervals};
use crate::datatypes::values::{DataFrame, Value};
use crate::graph::core::value_operations::format_value_compact;
use crate::graph::dir_graph::DirGraph;
use crate::graph::mutation::batch::ConflictHandling;
use crate::graph::schema::{InternedKey, TemporalConfig};
use crate::graph::storage::GraphRead;

/// [`check_row`] for a write, completed with the element's name: whether
/// the row's interval is empty.
fn judge(
    from: &Value,
    to: &Value,
    config: &TemporalConfig,
    name: impl FnOnce() -> String,
) -> Result<Emptiness, String> {
    if suspended() {
        return Ok(Emptiness::default());
    }
    check_row(from, to, config)
        .map(|bounds| Emptiness::of(bounds, config))
        .map_err(|reason| format!("{}, {reason}", name()))
}

#[cfg(test)]
thread_local! {
    static SUSPENDED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

#[cfg(test)]
fn suspended() -> bool {
    SUSPENDED.with(std::cell::Cell::get)
}

#[cfg(not(test))]
fn suspended() -> bool {
    false
}

/// Run `f` with the check off on this thread — the state a graph saved by
/// an earlier version, which accepted any write onto a declared type, can
/// hold. The tests of how the readers treat such rows build them here.
#[cfg(test)]
pub(crate) fn unchecked<T>(f: impl FnOnce() -> T) -> T {
    SUSPENDED.with(|flag| flag.set(true));
    let out = f();
    SUSPENDED.with(|flag| flag.set(false));
    out
}

/// The declarations governing a `rel_type` relationship from a node of
/// `source_type`: the source's keyed declaration, else every unkeyed one —
/// the declarations a declare-time walk validates that relationship under.
pub(super) fn edge_configs_for<'g>(
    graph: &'g DirGraph,
    rel_type: &str,
    source_type: Option<&str>,
) -> Vec<&'g TemporalConfig> {
    let configs = graph.temporal.edges(rel_type);
    if let Some(keyed) = source_type.and_then(|source| {
        configs
            .iter()
            .find(|c| c.source_type.as_deref() == Some(source))
    }) {
        return vec![keyed];
    }
    configs.iter().filter(|c| c.source_type.is_none()).collect()
}

/// The declarations on the labels node `idx` carries: its primary type's,
/// then its secondary labels' — built only when the graph has any, so the
/// common node allocates nothing.
fn node_configs(graph: &DirGraph, idx: NodeIndex) -> impl Iterator<Item = &TemporalConfig> {
    let declared =
        |key: InternedKey| resolve(graph, key).and_then(|label| graph.temporal.node(label));
    let (primary, secondary) = if graph.temporal.has_node_declarations() {
        let primary = graph.graph.node_type_of(idx).and_then(declared);
        let secondary: Vec<&TemporalConfig> = if graph.has_secondary_labels {
            graph
                .secondary_labels(idx)
                .into_iter()
                .filter_map(declared)
                .collect()
        } else {
            Vec::new()
        };
        (primary, secondary)
    } else {
        (None, Vec::new())
    };
    primary.into_iter().chain(secondary)
}

fn resolve(graph: &DirGraph, key: InternedKey) -> Option<&str> {
    graph.interner.try_resolve(key)
}

fn node_name(graph: &DirGraph, idx: NodeIndex) -> String {
    graph
        .graph
        .get_node_id(idx)
        .map_or_else(|| "?".to_string(), |id| format_value_compact(&id))
}

fn edge_name(graph: &DirGraph, rel_type: &str, source: NodeIndex, target: NodeIndex) -> String {
    format!(
        "{rel_type} relationship from node '{}' to node '{}'",
        node_name(graph, source),
        node_name(graph, target)
    )
}

/// Whether writing `property` onto relationship type `rel_type` can move a
/// declared bound.
pub(crate) fn edge_property_is_bound(graph: &DirGraph, rel_type: &str, property: &str) -> bool {
    graph
        .temporal
        .edges(rel_type)
        .iter()
        .any(|c| c.valid_from == property || c.valid_to == property)
}

/// Refuse the stored state of node `idx` when it breaks a declaration on one
/// of its labels, naming the node as a declaration does, and count it into
/// `empty`.
pub(crate) fn check_stored_node(
    graph: &DirGraph,
    idx: NodeIndex,
    empty: &mut EmptyIntervals,
) -> Result<(), String> {
    let name = || format!("node '{}'", node_name(graph, idx));
    let mut row_empty = Emptiness::default();
    for config in node_configs(graph, idx) {
        let from = validate::node_bound(graph, idx, &config.valid_from);
        let to = validate::node_bound(graph, idx, &config.valid_to);
        row_empty |= judge(&from, &to, config, name)?;
    }
    empty.note(row_empty, name);
    Ok(())
}

/// Refuse writing `values` (property, new value) onto node `idx` when the
/// node's end state would break a declaration on one of its labels, and count
/// it into `empty`. All of `values` are judged together, so a write that moves
/// both bounds is judged as the pair it leaves — the `SET n.vf = …, n.vt = …`
/// rule. A declaration none of `values` binds is left as the node holds it.
/// Judged before anything is written, so a refusal leaves the graph unchanged.
pub(crate) fn check_node_update(
    graph: &DirGraph,
    idx: NodeIndex,
    values: &[(&str, &Value)],
    empty: &mut EmptyIntervals,
) -> Result<(), String> {
    let name = || format!("node '{}'", node_name(graph, idx));
    let read = |property: &str| {
        values
            .iter()
            .rev()
            .find(|(written, _)| *written == property)
            .map_or_else(
                || validate::node_bound(graph, idx, property),
                |(_, value)| (*value).clone(),
            )
    };
    let mut declared = false;
    let mut row_empty = Emptiness::default();
    for config in node_configs(graph, idx) {
        let moves_a_bound = values
            .iter()
            .any(|(written, _)| *written == config.valid_from || *written == config.valid_to);
        if !moves_a_bound {
            continue;
        }
        declared = true;
        let (from, to) = (read(&config.valid_from), read(&config.valid_to));
        row_empty |= judge(&from, &to, config, name)?;
    }
    if declared {
        empty.note(row_empty, name);
    }
    Ok(())
}

/// Refuse the stored state of relationship `edge` when it breaks the
/// declaration governing it, naming its endpoints as a declaration does, and
/// count it into `empty`.
pub(crate) fn check_stored_edge(
    graph: &DirGraph,
    edge: EdgeIndex,
    empty: &mut EmptyIntervals,
) -> Result<(), String> {
    let Some(rel_key) = graph.graph.edge_weight(edge).map(|e| e.connection_type) else {
        return Ok(());
    };
    let Some(rel_type) = resolve(graph, rel_key) else {
        return Ok(());
    };
    if graph.temporal.edges(rel_type).is_empty() {
        return Ok(());
    }
    let Some((source, target)) = graph.graph.edge_endpoints(edge) else {
        return Ok(());
    };
    let name = || edge_name(graph, rel_type, source, target);
    let mut row_empty = Emptiness::default();
    let source_type = graph
        .graph
        .node_type_of(source)
        .and_then(|key| resolve(graph, key));
    let configs = edge_configs_for(graph, rel_type, source_type);
    if configs.is_empty() {
        return Ok(());
    }
    for config in configs {
        let from = validate::edge_bound(graph, edge, InternedKey::from_str(&config.valid_from));
        let to = validate::edge_bound(graph, edge, InternedKey::from_str(&config.valid_to));
        row_empty |= judge(&from, &to, config, name)?;
    }
    empty.note(row_empty, name);
    Ok(())
}

/// Refuse a node a Cypher `CREATE` is about to insert with `labels`, when the
/// bounds `read` gives break a declaration on one of them, and count it into
/// `empty` when one is declared. `id` names it.
pub(crate) fn check_new_node<'l>(
    graph: &DirGraph,
    labels: impl IntoIterator<Item = &'l str>,
    (id, read): (&Value, impl Fn(&str) -> Option<Value>),
    empty: &mut EmptyIntervals,
) -> Result<(), String> {
    if !graph.temporal.has_node_declarations() {
        return Ok(());
    }
    let name = || format!("node '{}'", format_value_compact(id));
    let mut declared = false;
    let mut row_empty = Emptiness::default();
    for config in labels.into_iter().filter_map(|l| graph.temporal.node(l)) {
        declared = true;
        let from = read(&config.valid_from).unwrap_or(Value::Null);
        let to = read(&config.valid_to).unwrap_or(Value::Null);
        row_empty |= judge(&from, &to, config, name)?;
    }
    if declared {
        empty.note(row_empty, name);
    }
    Ok(())
}

/// Refuse a `rel_type` relationship a Cypher `CREATE` is about to insert from
/// `source` to `target`, when the bounds `read` gives break its declaration,
/// and count it into `empty` when the type is declared.
pub(crate) fn check_new_edge(
    graph: &DirGraph,
    rel_type: &str,
    (source, target): (NodeIndex, NodeIndex),
    read: impl Fn(&str) -> Option<Value>,
    empty: &mut EmptyIntervals,
) -> Result<(), String> {
    if graph.temporal.edges(rel_type).is_empty() {
        return Ok(());
    }
    let source_type = graph
        .graph
        .node_type_of(source)
        .and_then(|key| resolve(graph, key));
    let name = || edge_name(graph, rel_type, source, target);
    let configs = edge_configs_for(graph, rel_type, source_type);
    if configs.is_empty() {
        return Ok(());
    }
    let mut row_empty = Emptiness::default();
    for config in configs {
        let from = read(&config.valid_from).unwrap_or(Value::Null);
        let to = read(&config.valid_to).unwrap_or(Value::Null);
        row_empty |= judge(&from, &to, config, name)?;
    }
    empty.note(row_empty, name);
    Ok(())
}

/// The value a loaded row leaves in one bound of a node under `mode`: a new
/// node takes the row's; an existing one merges it with what it holds, as
/// the batch applies it (a NULL cell writes nothing).
fn loaded_bound(
    graph: &DirGraph,
    existing: Option<NodeIndex>,
    mode: ConflictHandling,
    property: &str,
    cell: Value,
) -> Value {
    let Some(idx) = existing else {
        return cell;
    };
    let stored = || validate::node_bound(graph, idx, property);
    match mode {
        ConflictHandling::Replace => cell,
        ConflictHandling::Update | ConflictHandling::Sum | ConflictHandling::Skip => {
            if matches!(cell, Value::Null) {
                stored()
            } else {
                cell
            }
        }
        ConflictHandling::Preserve => {
            let held = stored();
            if matches!(held, Value::Null) {
                cell
            } else {
                held
            }
        }
    }
}

/// The labels a new `node_type` node is born with besides its type: the
/// declared ontology ancestors a materialized ontology stamps on it.
pub(crate) fn ancestor_labels<'g>(graph: &'g DirGraph, node_type: &str) -> Vec<&'g str> {
    if graph.managed_labels.is_empty() || graph.suppress_ontology_stamp {
        return Vec::new();
    }
    graph
        .ontology_ancestors_of(InternedKey::from_str(node_type))
        .iter()
        .filter_map(|&key| resolve(graph, key))
        .collect()
}

/// One declaration a load's rows answer to: the load's own type's, or that of
/// a label the load stamps (`stamped` — an existing node that lacks it gains
/// it, so its stored bounds answer too).
struct LoadConfig<'g> {
    config: &'g TemporalConfig,
    label: InternedKey,
    columns: (Option<usize>, Option<usize>),
    stamped: bool,
}

/// The declarations of `node_type` and of every label a load onto it stamps
/// — `labels`, then the type's ontology ancestors — each once.
fn load_configs<'g>(
    graph: &'g DirGraph,
    node_type: &str,
    labels: &[&str],
    frame: &DataFrame,
) -> Vec<LoadConfig<'g>> {
    let ancestors = ancestor_labels(graph, node_type);
    let names = std::iter::once(node_type)
        .chain(labels.iter().copied())
        .chain(ancestors);
    let mut configs: Vec<LoadConfig<'g>> = Vec::new();
    for (position, name) in names.enumerate() {
        let Some(config) = graph.temporal.node(name) else {
            continue;
        };
        if configs.iter().any(|c| std::ptr::eq(c.config, config)) {
            continue;
        }
        configs.push(LoadConfig {
            config,
            label: InternedKey::from_str(name),
            columns: (
                frame.get_column_index(&config.valid_from),
                frame.get_column_index(&config.valid_to),
            ),
            stamped: position > 0,
        });
    }
    configs
}

/// Refuse an `add_nodes` load onto `node_type` — stamping `labels` besides
/// its type — whose rows would leave a node breaking a declaration on the
/// labels it ends up with, naming the first by its position in the load, and
/// count the rows it leaves with an empty interval. Runs before the load
/// writes. A row is judged by the bounds it leaves under `mode`; a
/// declaration whose bounds the load does not carry is read only for an
/// existing node that gains its label.
pub(crate) fn check_node_load(
    graph: &DirGraph,
    node_type: &str,
    frame: &DataFrame,
    id_column: usize,
    (mode, labels): (ConflictHandling, &[&str]),
) -> Result<EmptyIntervals, String> {
    let mut empty = EmptyIntervals::default();
    if !graph.temporal.has_node_declarations() || suspended() {
        return Ok(empty);
    }
    let configs = load_configs(graph, node_type, labels, frame);
    // A secondary label an existing node carries can be declared although
    // the load's type is not.
    let secondary = graph.has_secondary_labels;
    let carried = |c: &LoadConfig<'_>| c.columns.0.is_some() || c.columns.1.is_some();
    if !secondary && !configs.iter().any(|c| c.stamped || carried(c)) {
        return Ok(empty);
    }
    let cell = |row: usize, column: Option<usize>| {
        column
            .and_then(|column| frame.get_value_by_index(row, column))
            .unwrap_or(Value::Null)
    };
    for row in 0..frame.row_count() {
        let id = match frame.get_value_by_index(row, id_column) {
            Some(Value::Null) | None => continue,
            Some(id) => id,
        };
        let row_name = || format!("row {row} (0-based) of the load");
        let mut row_empty = Emptiness::default();
        // Looked up only when a stored bound can show through, so the
        // common row — both bounds present, a mode that writes them — costs
        // no id lookup.
        let mut existing: Option<Option<NodeIndex>> = None;
        let mut lookup =
            || *existing.get_or_insert_with(|| graph.id_indices.lookup(node_type, &id));
        for load in &configs {
            let (from, to) = (cell(row, load.columns.0), cell(row, load.columns.1));
            let decided = match mode {
                ConflictHandling::Replace => true,
                ConflictHandling::Update | ConflictHandling::Sum => {
                    !matches!(from, Value::Null) && !matches!(to, Value::Null)
                }
                ConflictHandling::Skip | ConflictHandling::Preserve => false,
            };
            if decided && load.columns.0.is_some() && load.columns.1.is_some() {
                row_empty |= judge(&from, &to, load.config, row_name)?;
                continue;
            }
            let Some(idx) = lookup() else {
                row_empty |= judge(&from, &to, load.config, row_name)?;
                continue;
            };
            let gains = load.stamped && !graph.node_has_label(idx, load.label);
            if !gains && (mode == ConflictHandling::Skip || !carried(load)) {
                continue;
            }
            let (from, to) = if mode == ConflictHandling::Skip {
                (
                    validate::node_bound(graph, idx, &load.config.valid_from),
                    validate::node_bound(graph, idx, &load.config.valid_to),
                )
            } else {
                (
                    loaded_bound(graph, Some(idx), mode, &load.config.valid_from, from),
                    loaded_bound(graph, Some(idx), mode, &load.config.valid_to, to),
                )
            };
            row_empty |= judge(&from, &to, load.config, row_name)?;
        }
        if secondary && mode != ConflictHandling::Skip {
            if let Some(idx) = lookup() {
                row_empty |= check_secondary_labels(graph, &configs, (idx, mode), frame, row)?;
            }
        }
        empty.note(row_empty, row_name);
    }
    Ok(empty)
}

/// The declarations on the secondary labels an existing node `idx` already
/// carries, other than `configs`, judged on the bounds row `row` leaves it.
fn check_secondary_labels(
    graph: &DirGraph,
    configs: &[LoadConfig<'_>],
    (idx, mode): (NodeIndex, ConflictHandling),
    frame: &DataFrame,
    row: usize,
) -> Result<Emptiness, String> {
    let cell = |column: Option<usize>| {
        column
            .and_then(|column| frame.get_value_by_index(row, column))
            .unwrap_or(Value::Null)
    };
    let row_name = || format!("row {row} (0-based) of the load");
    let mut row_empty = Emptiness::default();
    for config in node_configs(graph, idx) {
        if configs.iter().any(|c| std::ptr::eq(c.config, config)) {
            continue;
        }
        let from_column = frame.get_column_index(&config.valid_from);
        let to_column = frame.get_column_index(&config.valid_to);
        if from_column.is_none() && to_column.is_none() {
            continue;
        }
        let from = loaded_bound(
            graph,
            Some(idx),
            mode,
            &config.valid_from,
            cell(from_column),
        );
        let to = loaded_bound(graph, Some(idx), mode, &config.valid_to, cell(to_column));
        row_empty |= judge(&from, &to, config, row_name)?;
    }
    Ok(row_empty)
}

/// Refuse an `add_nodes` load onto `node_type` that also stamps `labels` on
/// every row, before anything is written, when a row would leave a node
/// breaking a declaration on the labels it ends up with — the check
/// `add_nodes` itself runs, widened to the labels a binding stamps after it.
/// `conflict_handling` is the load's. `Ok` carries the warning for the rows
/// the load leaves with an empty interval — the one `add_nodes` reports for
/// its own type, widened the same way.
pub fn check_labelled_load(
    graph: &mut DirGraph,
    frame: &DataFrame,
    (node_type, unique_id_field): (&str, &str),
    conflict_handling: Option<&str>,
    labels: &[&str],
) -> Result<Option<String>, String> {
    if !graph.temporal.has_node_declarations() {
        return Ok(None);
    }
    let mode = crate::graph::mutation::maintain::parse_conflict_mode(conflict_handling)?;
    let Some(id_column) = frame.get_column_index(unique_id_field) else {
        return Ok(None);
    };
    graph.build_id_index(node_type);
    check_node_load(graph, node_type, frame, id_column, (mode, labels)).map(|e| e.warning())
}

/// Refuse stamping `label` on `nodes` when it is declared and a node that
/// does not carry it yet holds bounds its declaration refuses — the rule
/// and wording of Cypher `SET n:Label`, naming the node. `Ok` carries the
/// warning for the nodes gaining it whose interval is empty.
pub fn check_label_stamp(
    graph: &DirGraph,
    nodes: &[NodeIndex],
    label: &str,
) -> Result<Option<String>, String> {
    let Some(config) = graph.temporal.node(label) else {
        return Ok(None);
    };
    let key = InternedKey::from_str(label);
    let _arena_guard = graph.graph.begin_query();
    let mut empty = EmptyIntervals::default();
    for &idx in nodes {
        if graph.graph.node_type_of(idx) == Some(key) || graph.node_has_label(idx, key) {
            continue;
        }
        let from = validate::node_bound(graph, idx, &config.valid_from);
        let to = validate::node_bound(graph, idx, &config.valid_to);
        let name = || format!("node '{}'", node_name(graph, idx));
        let row_empty = judge(&from, &to, config, name)?;
        empty.note(row_empty, name);
    }
    Ok(empty.warning())
}

/// Refuse giving the `node_type` node with id `id` — written (or merged,
/// under `mode`) with the properties `read` gives — the declared `labels`
/// among `labels`, when it would break one, naming the node, and count it
/// into `empty` when a label it gains is declared. The pre-write check for a
/// writer that stamps labels per node (`extend`).
pub(crate) fn check_labelled_node(
    graph: &DirGraph,
    (node_type, id): (&str, &Value),
    read: impl Fn(&str) -> Option<Value>,
    (labels, mode): (&[String], ConflictHandling),
    empty: &mut EmptyIntervals,
) -> Result<(), String> {
    if !graph.temporal.has_node_declarations() {
        return Ok(());
    }
    let name = || format!("node '{}'", format_value_compact(id));
    let mut judged = false;
    let mut row_empty = Emptiness::default();
    let existing = graph.lookup_by_id_normalized(node_type, id);
    for label in labels {
        let Some(config) = graph.temporal.node(label) else {
            continue;
        };
        if existing.is_some_and(|idx| graph.node_has_label(idx, InternedKey::from_str(label))) {
            continue;
        }
        let bound = |property: &str| {
            let cell = read(property).unwrap_or(Value::Null);
            match (existing, mode) {
                (Some(idx), ConflictHandling::Skip) => validate::node_bound(graph, idx, property),
                _ => loaded_bound(graph, existing, mode, property, cell),
            }
        };
        let (from, to) = (bound(&config.valid_from), bound(&config.valid_to));
        judged = true;
        row_empty |= judge(&from, &to, config, name)?;
    }
    if judged {
        empty.note(row_empty, name);
    }
    Ok(())
}

/// Refuse an `add_relationships` load of `rel_type` from `source_type` whose
/// rows break the declaration governing them, naming the first by its
/// position in the load, and count its rows with an empty interval. A
/// declared type's rows are versions — each is stored as it is or matches an
/// identical stored one — so the row is what the relationship will hold.
pub(crate) fn check_edge_load(
    graph: &DirGraph,
    rel_type: &str,
    source_type: &str,
    frame: &DataFrame,
) -> Result<EmptyIntervals, String> {
    if graph.temporal.edges(rel_type).is_empty() || suspended() {
        return Ok(EmptyIntervals::default());
    }
    validate::check_frame(frame, &edge_configs_for(graph, rel_type, Some(source_type)))
}

/// Refuse edge rows already resolved to their endpoints — the shape
/// `add_edges_from_specs` and `create_connections` hold, row `i` of
/// `endpoints` reading `properties[i]` — when a row breaks the declaration
/// governing it, naming its endpoints, and count those with an empty
/// interval.
pub(crate) fn check_edge_rows(
    graph: &DirGraph,
    rel_type: &str,
    endpoints: &[(usize, NodeIndex, NodeIndex)],
    properties: &[Vec<(InternedKey, Value)>],
) -> Result<EmptyIntervals, String> {
    let mut empty = EmptyIntervals::default();
    if graph.temporal.edges(rel_type).is_empty() {
        return Ok(empty);
    }
    for &(row, source, target) in endpoints {
        let Some(row) = properties.get(row) else {
            continue;
        };
        let read = |property: &str| {
            let key = InternedKey::from_str(property);
            row.iter().find(|(k, _)| *k == key).map(|(_, v)| v.clone())
        };
        check_new_edge(graph, rel_type, (source, target), read, &mut empty)?;
    }
    Ok(empty)
}
