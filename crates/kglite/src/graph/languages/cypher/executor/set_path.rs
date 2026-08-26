//! Nested SET l-value application (`SET o.line_items[2].qty = 8`) — the
//! read-modify-write half of the structured-data support.
//!
//! The write side stays whole-value: `apply_path` rebuilds the property's
//! value with one cell replaced and hands it back to the normal property-SET
//! path, so shape validation re-checks the whole collection and WAL/CDC
//! carry the whole value. Errors name the path travelled so far
//! (`line_items[7]: index out of bounds (list has 3 rows)`).

use crate::datatypes::values::Value;
use crate::datatypes::PropMap;
use crate::graph::languages::cypher::ast::SetPathStep;
use crate::graph::languages::cypher::result::ResultRow;

use super::CypherExecutor;

/// The executor-facing entry: evaluate the path's index expressions, read
/// the property's current value, and rebuild it with the addressed cell
/// replaced. The caller writes the result through the normal property-SET
/// path, so shape gates re-validate the whole collection and WAL/CDC carry
/// the whole value — atomic per statement, replay-correct by construction,
/// and no lost-update window between an application-level read and write.
pub(super) fn read_modify(
    graph: &crate::graph::dir_graph::DirGraph,
    node_idx: petgraph::graph::NodeIndex,
    property: &str,
    steps: &[SetPathStep],
    new_value: Value,
    params: &std::collections::HashMap<String, Value>,
    row: &ResultRow,
) -> Result<Value, String> {
    let resolved = {
        let executor = CypherExecutor::with_params(graph, params, None);
        resolve_steps(&executor, row, steps)?
    };
    let current = {
        use crate::graph::storage::GraphRead;
        let _guard = graph.graph.begin_query();
        let key = crate::graph::schema::InternedKey::from_str(property);
        graph
            .graph
            .get_node_property(node_idx, key)
            .unwrap_or(Value::Null)
    };
    apply_path(property, current, &resolved, new_value)
}

/// A path step with its index expression already evaluated.
pub(super) enum ResolvedStep {
    Field(String),
    Index(i64),
}

pub(super) fn resolve_steps(
    executor: &CypherExecutor<'_>,
    row: &ResultRow,
    steps: &[SetPathStep],
) -> Result<Vec<ResolvedStep>, String> {
    steps
        .iter()
        .map(|step| match step {
            SetPathStep::Field(name) => Ok(ResolvedStep::Field(name.clone())),
            SetPathStep::Index(expr) => match executor.evaluate_expression(expr, row)? {
                Value::Int64(i) => Ok(ResolvedStep::Index(i)),
                other => Err(format!(
                    "SET path index must be an integer, got {}",
                    other.type_name()
                )),
            },
        })
        .collect()
}

/// Rebuild `current` with the cell at `steps` replaced by `new_value`.
pub(super) fn apply_path(
    property: &str,
    current: Value,
    steps: &[ResolvedStep],
    new_value: Value,
) -> Result<Value, String> {
    descend(property.to_string(), current, steps, new_value)
}

fn descend(
    path: String,
    current: Value,
    steps: &[ResolvedStep],
    new_value: Value,
) -> Result<Value, String> {
    let Some((head, rest)) = steps.split_first() else {
        return Ok(new_value);
    };
    match head {
        ResolvedStep::Index(raw) => {
            let Value::List(mut items) = current else {
                return Err(format!(
                    "{path}: expected a list to index into, found {}",
                    current.type_name()
                ));
            };
            // Negative indexes count from the end, matching read-side
            // IndexAccess semantics.
            let len = items.len() as i64;
            let idx = if *raw < 0 { len + raw } else { *raw };
            if idx < 0 || idx >= len {
                return Err(format!(
                    "{path}[{raw}]: index out of bounds (list has {len} entries)"
                ));
            }
            let idx = idx as usize;
            let child = std::mem::replace(&mut items[idx], Value::Null);
            items[idx] = descend(format!("{path}[{raw}]"), child, rest, new_value)?;
            Ok(Value::List(items))
        }
        ResolvedStep::Field(name) => {
            let map = match current {
                Value::Map(map) => map,
                // Setting a field on a missing/null intermediate creates the
                // map — `SET o.metadata.status = 'x'` works on a node that
                // never had `metadata` (openCypher-adjacent upsert shape).
                Value::Null => PropMap::from_pairs(Vec::new()),
                other => {
                    return Err(format!(
                        "{path}: expected a map for '.{name}', found {}",
                        other.type_name()
                    ));
                }
            };
            let mut pairs: Vec<(crate::datatypes::PropKey, Value)> =
                map.iter().map(|(k, v)| (k.into(), v.clone())).collect();
            let child = map.get(name).cloned().unwrap_or(Value::Null);
            let updated = descend(format!("{path}.{name}"), child, rest, new_value)?;
            match pairs
                .iter_mut()
                .find(|(k, _)| AsRef::<str>::as_ref(k) == name.as_str())
            {
                Some(slot) => slot.1 = updated,
                None => pairs.push((name.as_str().into(), updated)),
            }
            Ok(Value::Map(PropMap::from_pairs(pairs)))
        }
    }
}
