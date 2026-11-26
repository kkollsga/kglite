//! Writing a node's `id` through Cypher.
//!
//! `id` is the node's identity, not a stored property, so a `SET` cannot move
//! it. Two cases are still legitimate, and both come from drivers that treat
//! `id` as an ordinary key (GraphBridge's `CREATE (n:L) SET n = $properties`
//! and its `SET n.id = n.id` lock):
//!
//! - **Setting the id to the value it already has** is a no-op, whichever
//!   spelling the statement uses (`SET n.id`, `SET n += {id: …}`,
//!   `SET n = {id: …, …}`). [`set_existing_id`] accepts exactly that.
//! - **Naming the id of a node the statement is creating.** `CREATE (n:L) SET
//!   n = $props` reads the id out of `$props`. The id is known before the node
//!   exists, so [`supply_id_from_set`] hands it to the `CREATE` instead of
//!   minting one and re-keying the node afterwards: the node is born with its
//!   id, and the id index, the title, the unique-constraint check, rollback and
//!   the write-ahead log all see an ordinary create.
//!
//! Re-keying a node that already exists stays refused.

use std::collections::HashMap;

use super::identity_fields::IdentityAliases;
use super::write::{row_evaluator, WriteClauseCtx};
use crate::datatypes::prop_map::PropMap;
use crate::datatypes::values::Value;
use crate::graph::core::filtering::values_equal;
use crate::graph::languages::cypher::ast::{
    Clause, CreateNodePattern, Expression, SetClause, SetItem,
};
use crate::graph::languages::cypher::result::ResultRow;
use crate::graph::schema::DirGraph;
use crate::graph::storage::GraphRead;
use petgraph::graph::NodeIndex;

/// The SET clause a CREATE may take node ids from: the one directly after it.
pub(super) fn set_after_create<'a>(
    clause: &Clause,
    next: Option<&'a Clause>,
) -> Option<&'a SetClause> {
    match (clause, next) {
        (Clause::Create(_), Some(Clause::Set(set))) => Some(set),
        _ => None,
    }
}

/// Accept `SET <id spelling> = value` on an existing node when `value` is its
/// current id; refuse any other value.
pub(super) fn set_existing_id(
    graph: &DirGraph,
    node_idx: NodeIndex,
    property: &str,
    value: &Value,
) -> Result<(), String> {
    let (current, node_type) = {
        let node_type = graph
            .graph
            .node_type_of(node_idx)
            .and_then(|key| graph.interner.try_resolve(key))
            .unwrap_or("?");
        (GraphRead::get_node_id(&graph.graph, node_idx), node_type)
    };
    if current.as_ref().is_some_and(|id| values_equal(id, value)) {
        return Ok(());
    }
    Err(format!(
        "Cannot SET node id — it is immutable ('{property}' is the id field of node type \
         '{node_type}', so it names the identity). A node takes its id when it is created: \
         in the CREATE pattern, or in a SET that directly follows the CREATE \
         (`CREATE (n:L) SET n = $props`). Setting the id to the value it already has is \
         accepted."
    ))
}

/// Give a node being created the id the SET clause that directly follows its
/// `CREATE` names for it, when the pattern names none.
///
/// Only an expression that cannot differ between now and the SET is read
/// ([`is_stable`]): the SET later checks the node's id against its own value,
/// so a re-evaluated `randomUUID()` would disagree with the id used here. An
/// expression that fails to evaluate, or yields no id, supplies nothing and
/// the SET meets the minted id as it always did. When several items name an id
/// the last one wins, as sequential SET items do.
pub(super) fn supply_id_from_set(
    graph: &DirGraph,
    ctx: &WriteClauseCtx<'_>,
    (node_pat, row): (&CreateNodePattern, &ResultRow),
    aliases: &IdentityAliases,
    properties: &mut HashMap<String, Value>,
) {
    let (Some(set), Some(var)) = (ctx.next_set, node_pat.variable.as_deref()) else {
        return;
    };
    if properties.keys().any(|key| aliases.canonical(key) == "id") {
        return;
    }
    let mut named: Option<Value> = None;
    for item in &set.items {
        match item {
            SetItem::Property {
                variable,
                property,
                path,
                expression,
            } if variable == var
                && path.is_empty()
                && aliases.canonical(property) == "id"
                && is_stable(expression, var) =>
            {
                if let Ok(value) = row_evaluator(graph, ctx).evaluate_expression(expression, row) {
                    named = Some(value).filter(|v| !matches!(v, Value::Null)).or(named);
                }
            }
            SetItem::Map {
                variable,
                expression,
                ..
            } if variable == var && is_stable(expression, var) => {
                // A parameter or row variable is probed in place; only another
                // expression is evaluated, which materialises its map.
                let held = match expression {
                    Expression::Parameter(name) => ctx.params.get(name),
                    Expression::Variable(name) => row.projected.get(name),
                    _ => None,
                };
                let id = match held {
                    Some(Value::Map(map)) => id_in_map(map, aliases),
                    Some(_) => None,
                    None => match row_evaluator(graph, ctx).evaluate_expression(expression, row) {
                        Ok(Value::Map(map)) => id_in_map(&map, aliases),
                        _ => None,
                    },
                };
                named = id.or(named);
            }
            _ => {}
        }
    }
    if let Some(id) = named {
        properties.insert("id".to_string(), id);
    }
}

/// The non-null id a map names, under `id` or the type's own id spelling.
fn id_in_map(map: &PropMap, aliases: &IdentityAliases) -> Option<Value> {
    let alias = aliases.id_field().and_then(|name| map.get(name));
    map.get("id")
        .or(alias)
        .filter(|v| !matches!(v, Value::Null))
        .cloned()
}

/// Whether `expr` yields the same value when evaluated again for the same
/// row, and does not read the node `var` the CREATE has not made yet.
fn is_stable(expr: &Expression, var: &str) -> bool {
    match expr {
        Expression::Parameter(_) | Expression::Literal(_) => true,
        Expression::Variable(name) => name != var,
        Expression::PropertyAccess { variable, .. } => variable != var,
        Expression::MapLiteral(entries) => entries.iter().all(|(_, e)| is_stable(e, var)),
        Expression::ListLiteral(items) => items.iter().all(|e| is_stable(e, var)),
        Expression::ExprPropertyAccess { expr, .. } => is_stable(expr, var),
        Expression::IndexAccess { expr, index } => is_stable(expr, var) && is_stable(index, var),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn property(variable: &str) -> Expression {
        Expression::PropertyAccess {
            variable: variable.to_string(),
            property: "id".to_string(),
        }
    }

    #[test]
    fn parameters_literals_and_other_rows_variables_are_stable() {
        let map = Expression::MapLiteral(vec![
            ("id".to_string(), Expression::Parameter("p".to_string())),
            ("x".to_string(), Expression::Literal(Value::Int64(1))),
        ]);
        assert!(is_stable(&map, "n"));
        assert!(is_stable(&Expression::Variable("row".to_string()), "n"));
        assert!(is_stable(&property("row"), "n"));
    }

    #[test]
    fn the_new_node_and_function_calls_are_not_stable() {
        assert!(!is_stable(&Expression::Variable("n".to_string()), "n"));
        assert!(!is_stable(&property("n"), "n"));
        let uuid = Expression::FunctionCall {
            name: "randomuuid".to_string(),
            args: Vec::new(),
            distinct: false,
        };
        assert!(!is_stable(&uuid, "n"));
        let nested = Expression::MapLiteral(vec![("id".to_string(), uuid)]);
        assert!(!is_stable(&nested, "n"));
    }
}
