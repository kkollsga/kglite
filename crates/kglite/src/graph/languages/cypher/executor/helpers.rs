//! Shared executor helpers — expression-to-string, predicate-to-string,
//! value comparison, arithmetic, type coercion, property resolution, and
//! CALL parameter extractors.

use super::super::ast::*;
use super::super::result::*;
use crate::datatypes::values::Value;
use crate::graph::schema::{soft_alias_fallback, DirGraph, InternedKey, SoftAliasFallback};
use crate::graph::storage::{GraphRead, NodeView};
use std::collections::{HashMap, HashSet};

// Re-export the ast aggregate helpers so downstream code can refer to them
// via the executor namespace (backward compatibility with pre-split code).
pub use super::super::ast::is_aggregate_expression;

// ============================================================================
// Helper Functions
// ============================================================================

// is_aggregate_expression and is_window_expression are re-exported above.

/// Variables a grouped aggregation still pins down on its output rows.
///
/// Every non-aggregate projection item is a grouping key, so any variable it
/// reads has one representative value per group. Variables that occur *only*
/// inside aggregate arguments do not — `count(c)` collapses many `c`s into a
/// scalar — and are deliberately excluded.
///
/// The three aggregation implementations
/// ([`super::CypherExecutor::execute_return_with_aggregation`],
/// [`super::stream::aggregate::apply`], and
/// [`super::CypherExecutor::execute_fused_optional_match_aggregate`]) all
/// rebuild their output rows from scratch, so they must agree on this set or
/// the same query orders differently depending on which one the planner picked.
pub(crate) fn grouping_variables(items: &[ReturnItem]) -> HashSet<String> {
    use crate::graph::languages::cypher::planner::simplification::collect_expression_refs;

    let mut vars = HashSet::new();
    for item in items {
        if !is_aggregate_expression(&item.expression) {
            collect_expression_refs(&item.expression, &mut vars);
        }
    }
    vars
}

/// Carry a group's representative node/edge/path bindings onto an aggregation
/// output row.
///
/// `source` is the group's first input row; `vars` comes from
/// [`grouping_variables`]. Without this, `ORDER BY t.priority` after
/// `RETURN t.title AS title, count(c) AS n` evaluates against a row that has
/// no `t` binding: every sort key resolves to NULL, the keys all tie, and the
/// stable sort silently hands back insertion order. Downstream
/// MATCH/OPTIONAL MATCH clauses need the same bindings to re-anchor patterns.
pub(crate) fn carry_group_bindings(
    vars: &HashSet<String>,
    source: &ResultRow,
    target: &mut ResultRow,
) {
    for var in vars {
        if let Some(&idx) = source.node_bindings.get(var) {
            target.node_bindings.insert(var.clone(), idx);
        }
        if let Some(edge) = source.edge_bindings.get(var) {
            target.edge_bindings.insert(var.clone(), *edge);
        }
        if let Some(path) = source.path_bindings.get(var) {
            target.path_bindings.insert(var.clone(), path.clone());
        }
    }
}

/// Augment each row's `projected` with an expression-keyed copy of every
/// aggregate return item, so HAVING predicates like `count(m) > 1` can
/// resolve even when the RETURN item is aliased (`count(m) AS c`).
/// Without this, the aliased aggregate is stored only under `c` and a
/// HAVING reference to `count(m)` would fall through to scalar dispatch
/// (which errors for aggregates and gets swallowed by unwrap_or(false)).
pub(super) fn augment_rows_with_aggregate_keys(rows: &mut [ResultRow], items: &[ReturnItem]) {
    for item in items {
        if !is_aggregate_expression(&item.expression) {
            continue;
        }
        let alias_key = return_item_column_name(item);
        let expr_key = expression_to_string(&item.expression);
        if alias_key == expr_key {
            continue;
        }
        for row in rows.iter_mut() {
            if row.projected.contains_key(&expr_key) {
                continue;
            }
            if let Some(val) = row.projected.get(&alias_key).cloned() {
                row.projected.insert(expr_key.clone(), val);
            }
        }
    }
}

/// Get the column name for a return item
pub fn return_item_column_name(item: &ReturnItem) -> String {
    if let Some(ref alias) = item.alias {
        alias.clone()
    } else {
        expression_to_string(&item.expression)
    }
}

/// Convert an expression to its string representation (for column naming).
///
/// This rendering *is* the identity the executor uses to resolve an unaliased
/// projection, so `schema_check` compares ORDER BY keys against RETURN items
/// with it — matching what the runtime can actually resolve.
pub(crate) fn expression_to_string(expr: &Expression) -> String {
    match expr {
        Expression::PropertyAccess { variable, property } => format!("{}.{}", variable, property),
        Expression::Variable(name) => name.clone(),
        Expression::Literal(val) => format_value_compact(val),
        Expression::FunctionCall {
            name,
            args,
            distinct,
        } => {
            let args_str: Vec<String> = args.iter().map(expression_to_string).collect();
            if *distinct {
                format!("{}(DISTINCT {})", name, args_str.join(", "))
            } else {
                format!("{}({})", name, args_str.join(", "))
            }
        }
        Expression::Star => "*".to_string(),
        Expression::Add(l, r) => {
            format!("{} + {}", expression_to_string(l), expression_to_string(r))
        }
        Expression::Subtract(l, r) => {
            format!("{} - {}", expression_to_string(l), expression_to_string(r))
        }
        Expression::Multiply(l, r) => {
            format!("{} * {}", expression_to_string(l), expression_to_string(r))
        }
        Expression::Divide(l, r) => {
            format!("{} / {}", expression_to_string(l), expression_to_string(r))
        }
        Expression::Modulo(l, r) => {
            format!("{} % {}", expression_to_string(l), expression_to_string(r))
        }
        Expression::Concat(l, r) => {
            format!("{} || {}", expression_to_string(l), expression_to_string(r))
        }
        Expression::Negate(inner) => format!("-{}", expression_to_string(inner)),
        Expression::ListLiteral(items) => {
            let items_str: Vec<String> = items.iter().map(expression_to_string).collect();
            format!("[{}]", items_str.join(", "))
        }
        Expression::Case { .. } => "CASE".to_string(),
        Expression::Parameter(name) => format!("${}", name),
        Expression::ListComprehension {
            variable,
            list_expr,
            filter,
            map_expr,
        } => {
            let mut result = format!("[{} IN {}", variable, expression_to_string(list_expr));
            if filter.is_some() {
                result.push_str(" WHERE ...");
            }
            if let Some(ref expr) = map_expr {
                result.push_str(&format!(" | {}", expression_to_string(expr)));
            }
            result.push(']');
            result
        }
        Expression::IndexAccess { expr, index } => {
            format!(
                "{}[{}]",
                expression_to_string(expr),
                expression_to_string(index)
            )
        }
        Expression::ListSlice { expr, start, end } => {
            let s = start
                .as_ref()
                .map_or(String::new(), |e| expression_to_string(e));
            let e = end
                .as_ref()
                .map_or(String::new(), |e| expression_to_string(e));
            format!("{}[{}..{}]", expression_to_string(expr), s, e)
        }
        Expression::MapProjection { variable, items } => {
            let items_str: Vec<String> = items
                .iter()
                .map(|item| match item {
                    MapProjectionItem::Property(prop) => format!(".{}", prop),
                    MapProjectionItem::AllProperties => ".*".to_string(),
                    MapProjectionItem::Alias { key, expr } => {
                        format!("{}: {}", key, expression_to_string(expr))
                    }
                })
                .collect();
            format!("{} {{{}}}", variable, items_str.join(", "))
        }
        Expression::MapLiteral(entries) => {
            let items_str: Vec<String> = entries
                .iter()
                .map(|(key, expr)| format!("{}: {}", key, expression_to_string(expr)))
                .collect();
            format!("{{{}}}", items_str.join(", "))
        }
        Expression::IsNull(inner) => format!("{} IS NULL", expression_to_string(inner)),
        Expression::IsNotNull(inner) => format!("{} IS NOT NULL", expression_to_string(inner)),
        Expression::QuantifiedList {
            quantifier,
            variable,
            list_expr,
            ..
        } => {
            let qname = match quantifier {
                ListQuantifier::Any => "any",
                ListQuantifier::All => "all",
                ListQuantifier::None => "none",
                ListQuantifier::Single => "single",
            };
            format!(
                "{}({} IN {} WHERE ...)",
                qname,
                variable,
                expression_to_string(list_expr)
            )
        }
        Expression::WindowFunction {
            name,
            partition_by,
            order_by,
        } => {
            let mut s = format!("{}() OVER (", name);
            if !partition_by.is_empty() {
                s.push_str("PARTITION BY ");
                let parts: Vec<String> = partition_by.iter().map(expression_to_string).collect();
                s.push_str(&parts.join(", "));
                if !order_by.is_empty() {
                    s.push(' ');
                }
            }
            if !order_by.is_empty() {
                s.push_str("ORDER BY ");
                let parts: Vec<String> = order_by
                    .iter()
                    .map(|item| {
                        let dir = if item.ascending { "" } else { " DESC" };
                        format!("{}{}", expression_to_string(&item.expression), dir)
                    })
                    .collect();
                s.push_str(&parts.join(", "));
            }
            s.push(')');
            s
        }
        Expression::PredicateExpr(pred) => predicate_to_string(pred),
        Expression::ExprPropertyAccess { expr, property } => {
            format!("{}.{}", expression_to_string(expr), property)
        }
        Expression::CountSubquery { .. } => {
            // Used as a column label (e.g. in a `WITH ... AS alias` projection).
            // Patterns don't render usefully here, so emit the subquery marker.
            "count{...}".to_string()
        }
        Expression::Reduce {
            accumulator,
            variable,
            list_expr,
            ..
        } => format!(
            "reduce({} = ..., {} IN {} | ...)",
            accumulator,
            variable,
            expression_to_string(list_expr)
        ),
    }
}

/// Convert a predicate to its string representation (for column naming)
pub(super) fn predicate_to_string(pred: &Predicate) -> String {
    match pred {
        Predicate::Comparison {
            left,
            operator,
            right,
        } => {
            let op_str = match operator {
                ComparisonOp::Equals => "=",
                ComparisonOp::NotEquals => "<>",
                ComparisonOp::LessThan => "<",
                ComparisonOp::LessThanEq => "<=",
                ComparisonOp::GreaterThan => ">",
                ComparisonOp::GreaterThanEq => ">=",
                ComparisonOp::RegexMatch => "=~",
            };
            format!(
                "{} {} {}",
                expression_to_string(left),
                op_str,
                expression_to_string(right)
            )
        }
        Predicate::StartsWith { expr, pattern } => {
            format!(
                "{} STARTS WITH {}",
                expression_to_string(expr),
                expression_to_string(pattern)
            )
        }
        Predicate::EndsWith { expr, pattern } => {
            format!(
                "{} ENDS WITH {}",
                expression_to_string(expr),
                expression_to_string(pattern)
            )
        }
        Predicate::Contains { expr, pattern } => {
            format!(
                "{} CONTAINS {}",
                expression_to_string(expr),
                expression_to_string(pattern)
            )
        }
        Predicate::LabelCheck {
            variable, label, ..
        } => format!("{}:{}", variable, label),
        _ => "predicate(...)".to_string(),
    }
}

/// Evaluate a comparison using existing filtering infrastructure
pub(super) fn evaluate_comparison(
    left: &Value,
    op: &ComparisonOp,
    right: &Value,
) -> Result<bool, String> {
    // Three-valued logic: comparisons involving Null propagate Null → false
    // (except IS NULL / IS NOT NULL which are handled elsewhere, and
    // Equals/NotEquals which handle Null explicitly via values_equal).
    match op {
        ComparisonOp::Equals => Ok(crate::graph::core::filtering::values_equal(left, right)),
        ComparisonOp::NotEquals => Ok(!crate::graph::core::filtering::values_equal(left, right)),
        _ if matches!(left, Value::Null) || matches!(right, Value::Null) => Ok(false),
        ComparisonOp::LessThan => Ok(crate::graph::core::filtering::compare_values(left, right)
            == Some(std::cmp::Ordering::Less)),
        ComparisonOp::LessThanEq => Ok(matches!(
            crate::graph::core::filtering::compare_values(left, right),
            Some(std::cmp::Ordering::Less) | Some(std::cmp::Ordering::Equal)
        )),
        ComparisonOp::GreaterThan => Ok(crate::graph::core::filtering::compare_values(left, right)
            == Some(std::cmp::Ordering::Greater)),
        ComparisonOp::GreaterThanEq => Ok(matches!(
            crate::graph::core::filtering::compare_values(left, right),
            Some(std::cmp::Ordering::Greater) | Some(std::cmp::Ordering::Equal)
        )),
        ComparisonOp::RegexMatch => match (left, right) {
            (Value::String(text), Value::String(pattern)) => {
                // Borrowed, not cloned: this runs once per row, and cloning
                // the shared `Arc` here was measured to cost more than the
                // match itself under the parallel runtime.
                super::regex_cache::with_compiled(pattern, |re| re.is_match(text))
                    .map_err(|e| format!("Invalid regular expression '{}': {}", pattern, e))
            }
            _ => Ok(false),
        },
    }
}

/// Resolve a node property, returning an owned Value directly.
/// Uses `get_property_value()` to avoid Cow wrapping/unwrapping overhead.
pub fn resolve_node_property(node: NodeView<'_>, property: &str, graph: &DirGraph) -> Value {
    let node_type_str = node.node_type_str(&graph.interner);
    let resolved = graph.resolve_alias(node_type_str, property);
    resolve_node_property_resolved(node, resolved, InternedKey::from_str(resolved), graph)
}

/// Hot-path variant of [`resolve_node_property`] for when the caller has
/// already established that `property` is **not** a registered id-/title-
/// field alias for this node's type (via
/// `CypherExecutor::property_might_be_alias`). The resolved name then
/// equals `property` verbatim, so we skip `resolve_alias` entirely — its
/// two `String`-keyed HashMap lookups are the dominant per-row cost on
/// alias-bearing graphs. Results are identical to `resolve_node_property`
/// for any non-alias property.
pub fn resolve_node_property_unaliased(
    node: NodeView<'_>,
    property: &str,
    graph: &DirGraph,
) -> Value {
    resolve_node_property_resolved(node, property, InternedKey::from_str(property), graph)
}

/// [`resolve_node_property_unaliased`] for a caller that has already interned
/// the resolved name. `InternedKey::from_str` hashes the property name, and a
/// scan that reads the same property from every row was paying that hash per
/// row rather than per query.
pub fn resolve_node_property_keyed(
    node: NodeView<'_>,
    property: &str,
    key: InternedKey,
    graph: &DirGraph,
) -> Value {
    resolve_node_property_resolved(node, property, key, graph)
}

/// Shared tail of [`resolve_node_property`] / [`resolve_node_property_unaliased`]:
/// turn an already-resolved canonical property name into an owned `Value`,
/// honouring the `id` / `title` virtuals, stored-property-wins (KG-1), the
/// soft-alias fallbacks, and spatial virtual properties. `node_type_str` is
/// resolved lazily — only the soft-alias / spatial branches need it.
///
/// `key` must be `InternedKey::from_str(resolved)`; it is a parameter only so
/// a per-row caller can hoist the hash.
#[inline]
fn resolve_node_property_resolved(
    node: NodeView<'_>,
    resolved: &str,
    key: InternedKey,
    graph: &DirGraph,
) -> Value {
    debug_assert_eq!(key, InternedKey::from_str(resolved));
    match resolved {
        "id" => node.id().into_owned(),
        "title" => node.title().into_owned(),
        _ => {
            // Stored property wins (covers a user property named `label`,
            // `type`, `node_type`, `name`, … — KG-1).
            if let Some(val) = node.get_value(key) {
                return val;
            }
            let node_type_str = node.node_type_str(&graph.interner);
            // No stored property — fall back to the structural convenience
            // for the soft aliases.
            if let Some(fb) = soft_alias_fallback(resolved) {
                return match fb {
                    SoftAliasFallback::Title => node.title().into_owned(),
                    SoftAliasFallback::TypeString => Value::String(node_type_str.to_string()),
                };
            }
            // Fall through to spatial virtual properties only if not found
            if let Some(config) = graph.get_spatial_config(node_type_str) {
                if resolved == "location" {
                    if let Some((lat_f, lon_f)) = &config.location {
                        let lat = crate::graph::core::value_operations::value_to_f64(
                            node.get_property(lat_f).as_deref().unwrap_or(&Value::Null),
                        );
                        let lon = crate::graph::core::value_operations::value_to_f64(
                            node.get_property(lon_f).as_deref().unwrap_or(&Value::Null),
                        );
                        if let (Some(lat), Some(lon)) = (lat, lon) {
                            return Value::Point { lat, lon };
                        }
                    }
                }
                if resolved == "geometry" {
                    if let Some(geom_f) = &config.geometry {
                        if let Some(val) = node.get_property_value(geom_f) {
                            return val;
                        }
                    }
                }
                if let Some((lat_f, lon_f)) = config.points.get(resolved) {
                    let lat = crate::graph::core::value_operations::value_to_f64(
                        node.get_property(lat_f).as_deref().unwrap_or(&Value::Null),
                    );
                    let lon = crate::graph::core::value_operations::value_to_f64(
                        node.get_property(lon_f).as_deref().unwrap_or(&Value::Null),
                    );
                    if let (Some(lat), Some(lon)) = (lat, lon) {
                        return Value::Point { lat, lon };
                    }
                }
                if let Some(shape_f) = config.shapes.get(resolved) {
                    if let Some(val) = node.get_property_value(shape_f) {
                        return val;
                    }
                }
            }
            Value::Null
        }
    }
}

/// Resolve a property from an EdgeBinding by looking up the graph
pub fn resolve_edge_property(graph: &DirGraph, edge: &EdgeBinding, property: &str) -> Value {
    let g = &graph.graph;
    if let Some(edge_data) = g.edge_weight(edge.edge_index) {
        match property {
            "type" | "connection_type" => {
                Value::String(edge_data.connection_type_str(&graph.interner).to_string())
            }
            _ => edge_data
                .get_property(property)
                .cloned()
                .unwrap_or(Value::Null),
        }
    } else {
        Value::Null
    }
}

/// Convert a NodeData to a representative Value (title string)
pub(super) fn node_to_map_value(node: NodeView<'_>) -> Value {
    node.title().into_owned()
}

/// What a node materialisation collects.
///
/// `keys(n)` and `properties(n)` must agree on the key set exactly — the whole
/// contract of `keys(n)` is that it equals `keys(properties(n))` — so the two
/// share one collection pass ([`collect_node_properties`]) and differ only in
/// what they keep from it. A `keys(n)` that walked the node on its own would be
/// a second copy of the null-omission, soft-alias and completion rules, free to
/// drift from the first.
trait PropertySink {
    /// `false` when the sink discards values, letting the collector skip work
    /// whose only product is a value it would drop.
    const NEEDS_VALUES: bool;

    /// Record a key whose value is unconditional (the id/title/type virtuals);
    /// `value` is not called when the sink discards values.
    fn insert_with(&mut self, key: &str, value: impl FnOnce() -> Value);

    /// Record a key whose value is already in hand.
    fn insert(&mut self, key: &str, value: Value);

    fn contains(&self, key: &str) -> bool;

    /// Absorb every stored property of the row, skipping the virtuals and the
    /// reserved provenance keys. Returns `true` when a stored key could not be
    /// resolved back to a name — the one case
    /// [`complete_from_type_schema`] cannot narrow.
    fn absorb_stored(
        &mut self,
        node: crate::graph::storage::NodeView<'_>,
        graph: &DirGraph,
    ) -> bool;
}

/// Is this a key the enumeration loop must not surface as a user property?
#[inline]
fn is_virtual_or_reserved(key: &str) -> bool {
    key == "id"
        || key == "title"
        || key == "type"
        || crate::graph::schema::is_reserved_provenance_key(key)
}

impl PropertySink for std::collections::BTreeMap<String, Value> {
    const NEEDS_VALUES: bool = true;

    #[inline]
    fn insert_with(&mut self, key: &str, value: impl FnOnce() -> Value) {
        self.insert(key.to_string(), value());
    }

    #[inline]
    fn insert(&mut self, key: &str, value: Value) {
        std::collections::BTreeMap::insert(self, key.to_string(), value);
    }

    #[inline]
    fn contains(&self, key: &str) -> bool {
        self.contains_key(key)
    }

    fn absorb_stored(
        &mut self,
        node: crate::graph::storage::NodeView<'_>,
        graph: &DirGraph,
    ) -> bool {
        // Resolved key by key rather than through `property_pairs_named`: that
        // allocates a `String` for every key including the four this loop
        // drops, and it silently discards a pair whose key the interner cannot
        // resolve — which is the one case the completion pass cannot derive
        // from the type's schema.
        let mut unresolved_key = false;
        for (ik, val) in node.property_pairs() {
            let Some(key) = graph.interner.try_resolve(ik) else {
                unresolved_key = true;
                continue;
            };
            if is_virtual_or_reserved(key) {
                continue;
            }
            std::collections::BTreeMap::insert(self, key.to_string(), val);
        }
        unresolved_key
    }
}

/// The names-only sink behind `keys(n)`. A `BTreeSet` so the emitted order is
/// the sorted, de-duplicated order `BTreeMap::into_keys` produced.
#[derive(Default)]
struct KeySink(std::collections::BTreeSet<String>);

impl PropertySink for KeySink {
    const NEEDS_VALUES: bool = false;

    #[inline]
    fn insert_with(&mut self, key: &str, _value: impl FnOnce() -> Value) {
        self.0.insert(key.to_string());
    }

    #[inline]
    fn insert(&mut self, key: &str, _value: Value) {
        self.0.insert(key.to_string());
    }

    #[inline]
    fn contains(&self, key: &str) -> bool {
        self.0.contains(key)
    }

    fn absorb_stored(
        &mut self,
        node: crate::graph::storage::NodeView<'_>,
        graph: &DirGraph,
    ) -> bool {
        let mut unresolved_key = false;
        for ik in node.property_key_set() {
            let Some(key) = graph.interner.try_resolve(ik) else {
                unresolved_key = true;
                continue;
            };
            if is_virtual_or_reserved(key) {
                continue;
            }
            self.0.insert(key.to_string());
        }
        unresolved_key
    }
}

/// Insert the hoisted id/title column back under its original df-column
/// name (e.g. `npdid`, `prospect_name`), skipping the three reserved
/// virtuals and any key already materialised. `value` is only evaluated
/// when an alias actually needs inserting.
#[inline]
fn insert_field_alias<S: PropertySink>(
    properties: &mut S,
    alias: Option<&String>,
    value: impl FnOnce() -> Value,
) {
    let Some(alias) = alias else { return };
    if alias == "id" || alias == "title" || alias == "type" {
        return;
    }
    if properties.contains(alias) {
        return;
    }
    // Match the old metadata-pass behaviour: a Null resolution is omitted,
    // not inserted (preserves the null-omission rule in returned nodes).
    let v = value();
    if !matches!(v, Value::Null) {
        properties.insert(alias, v);
    }
}

/// Phase A.1 / C2 — materialise a graph node into an owned
/// [`NodeValue`] suitable for `Value::Node`.
///
/// Returns `None` if the index doesn't resolve (defensive — callers
/// pass indices from active bindings, so this should always succeed).
///
/// The `id` field uses the petgraph NodeIndex as a stable internal
/// identity (mirrors Neo4j's INT64 node identity in Bolt). The
/// user-set `id` field, if any, is preserved inside `properties.id`.
pub(crate) fn materialize_node_value(
    idx: petgraph::graph::NodeIndex,
    graph: &crate::graph::DirGraph,
) -> Option<crate::datatypes::values::NodeValue> {
    // The returned value owns everything it needs, so on the disk backend the
    // record must not be parked in the query arena — a projection over N rows
    // would retain N records for the rest of the query
    // (storage/disk/query_arena.rs). Heap backends have no arena and keep the
    // direct borrow, which measures faster than routing through a closure.
    if graph.graph.is_disk() {
        let data = graph.graph.owned_node_data(idx)?;
        let store = data.properties.columnar_row_id().and_then(|row_id| {
            graph
                .graph
                .column_store(data.node_type)
                .map(|store| (&**store, row_id))
        });
        let node = crate::graph::storage::NodeView::new(&data, store);
        return Some(node_value_from_view(idx, node, graph));
    }
    let node = graph.graph.node_view(idx)?;
    Some(node_value_from_view(idx, node, graph))
}

/// The sorted, de-duplicated key set of [`materialize_node_value`]'s property
/// map — without building a single `Value` that only the map would have kept.
///
/// `keys(n)` is defined as `keys(properties(n))`, so this runs the *same*
/// collection pass through a names-only sink rather than walking the node
/// again: on a 30-column type the map route allocated 34 tree nodes and cloned
/// 30 values per node to then throw all of them away.
pub(crate) fn materialize_node_keys(
    idx: petgraph::graph::NodeIndex,
    graph: &crate::graph::DirGraph,
) -> Option<Vec<String>> {
    // Same arena discipline as `materialize_node_value` (disk records must not
    // outlive the call).
    if graph.graph.is_disk() {
        let data = graph.graph.owned_node_data(idx)?;
        let store = data.properties.columnar_row_id().and_then(|row_id| {
            graph
                .graph
                .column_store(data.node_type)
                .map(|store| (&**store, row_id))
        });
        let node = crate::graph::storage::NodeView::new(&data, store);
        return Some(node_keys_from_view(node, graph));
    }
    let node = graph.graph.node_view(idx)?;
    Some(node_keys_from_view(node, graph))
}

fn node_keys_from_view(
    node: crate::graph::storage::NodeView<'_>,
    graph: &crate::graph::DirGraph,
) -> Vec<String> {
    let node_type = node.node_type_str(&graph.interner).to_string();
    let mut keys = KeySink::default();
    collect_node_properties(&mut keys, node, &node_type, graph);
    keys.0.into_iter().collect()
}

/// The one walk that decides which properties a materialised node carries:
/// the three virtuals, every stored property, the id/title column aliases, and
/// the columnar completion pass. Shared by the value and names-only sinks.
fn collect_node_properties<S: PropertySink>(
    sink: &mut S,
    node: crate::graph::storage::NodeView<'_>,
    node_type: &str,
    graph: &crate::graph::DirGraph,
) {
    // Include the three virtual builtins so consumers always see
    // id/title/type, matching what n.id / n.title / n.type would
    // resolve to via the alias machinery.
    sink.insert_with("id", || node.id().into_owned());
    sink.insert_with("title", || node.title().into_owned());
    sink.insert_with("type", || Value::String(node_type.to_string()));
    // `type` is a soft alias, not a hard virtual: a stored property named
    // "type" wins over the structural type string (KG-1), matching what
    // `n.type` resolves to via `resolve_node_property_resolved`. `id` and
    // `title` are genuine virtuals — the canonical identity always wins —
    // so only `type` gets the stored-shadow check. It can only *replace* the
    // value under a key already present, so a names-only sink skips it.
    if S::NEEDS_VALUES {
        if let Some(stored) = node.get_property_value("type") {
            sink.insert("type", stored);
        }
    }
    // Then every user-set property the node carries. Reserved provenance keys
    // (updated_at, …) are engine metadata, not user data — omit them from the
    // materialised value so they stay out of keys()/properties()/RETURN n[.*]
    // (direct `n.updated_at` still resolves via the property path).
    // Complete for every storage variant: `NodeView` enumeration reads the
    // node's column store on saved graphs, where `NodeData::property_iter`
    // used to yield nothing (D1 defect 1). The columnar completion pass
    // further down stays — it recovers schema-declared properties that the
    // row itself does not carry.
    let unresolved_key = sink.absorb_stored(node, graph);
    // Cross-backend property completion.
    //
    // The loop above yields every stored property on every backend. The ONE
    // thing it cannot yield is
    // a column the loader hoisted out of `properties` into the dedicated
    // `id` / `title` fields: when `add_nodes` is given a `unique_id_field` /
    // `node_title_field` whose name isn't literally "id"/"title", that
    // original column name is recorded in `{id,title}_field_aliases` and the
    // value lives in `node.id()` / `node.title()`, NOT in `properties`.
    // `RETURN n` must surface it back under the original column name (e.g.
    // `Person.name` -> title). That's at most two O(1) map lookups — far
    // cheaper than the former per-node walk over every metadata key with a
    // `resolve_node_property` (alias + spatial) call each, which for the
    // in-memory backend could only ever re-discover these same two columns.
    insert_field_alias(sink, graph.title_field_aliases.get(node_type), || {
        node.title().into_owned()
    });
    insert_field_alias(sink, graph.id_field_aliases.get(node_type), || {
        node.id().into_owned()
    });
    // Columnar completion: a property the *type* declares but this row does
    // not carry as a stored column (a spatial virtual, a soft structural
    // alias) is recovered through `resolve_node_property`. Keys the loop above
    // already inserted are skipped, so this only ever adds.
    if node.properties_are_columnar() {
        if let Some(type_meta) = graph.get_node_type_metadata(node_type) {
            complete_from_type_schema(sink, node, node_type, type_meta, graph, unresolved_key);
        }
    }
}

/// Owned [`NodeValue`] from a borrowed view. Split out of
/// [`materialize_node_value`] so the view's lifetime ends with the call.
fn node_value_from_view(
    idx: petgraph::graph::NodeIndex,
    node: crate::graph::storage::NodeView<'_>,
    graph: &crate::graph::DirGraph,
) -> crate::datatypes::values::NodeValue {
    use crate::datatypes::values::NodeValue;
    use std::collections::BTreeMap;
    let node_type = node.node_type_str(&graph.interner).to_string();
    let mut properties: BTreeMap<String, Value> = BTreeMap::new();
    collect_node_properties(&mut properties, node, &node_type, graph);
    // Full label set (primary + secondaries), not just the primary type —
    // so a materialised node (RETURN n, collect(n)[0], …) carries the same
    // labels `MATCH (n:Sec)` / `labels(n)` see. `node_labels` reads the
    // canonical secondary-label index.
    let labels: Vec<String> = graph
        .node_labels(idx)
        .iter()
        .map(|k| graph.interner.resolve(*k).to_string())
        .collect();
    let labels = if labels.is_empty() {
        vec![node_type]
    } else {
        labels
    };
    NodeValue {
        id: idx.index() as u32,
        labels,
        properties,
    }
}

/// Recover the properties this row does not store but its **type** declares.
///
/// # Why this is not a walk over the declared keys
///
/// It used to be: for every key in `node_type_metadata[type]`, a full
/// [`resolve_node_property`] (alias resolve → intern → store probe → soft-alias
/// → spatial config) whose result was discarded whenever it came back `Null`.
/// On a 30-declared / 5-populated type that is 25 discarded resolutions per
/// materialised node — ~810 ns/node, and the pass runs for `RETURN n`,
/// `properties(n)`, `keys(n)`, `n {.*}`, export and `describe`.
///
/// Which keys can actually produce something is a per-**type** fact. For a
/// declared key `p` that this row does not already carry, `resolve_node_property`
/// answers in exactly four ways:
///
/// 1. `p` is the type's `unique_id_field` / `node_title_field` alias → the
///    identity value. Already inserted (with the same null-omission rule) by
///    `insert_field_alias` above, which does not require `p` to be declared, so
///    it is a strict superset of what this pass would add.
/// 2. `p` is a soft structural alias (`name`, `label`, `node_type` — `type` is
///    excluded by the filter below) → the title / type string.
/// 3. `p` names a spatial virtual of this type → the synthesized Point / WKT.
/// 4. Otherwise → the stored column value, or `Null`. A *stored* value is
///    already in `properties`: `NodeView::property_pairs` enumerates every
///    non-null column of the row, so anything `get_value` could find the loop
///    above already inserted. A `Null` inserts nothing. Either way: no-op.
///
/// So only cases 2 and 3 need visiting, and both come from a fixed, tiny
/// candidate set. Each candidate is still evaluated by `resolve_node_property`
/// itself — the *set* narrows, the resolution does not, so precedence between
/// stored value, soft alias and spatial virtual stays exactly where it was.
///
/// `unresolved_key` is case 4's one escape hatch: a column whose interned key
/// the interner cannot resolve back to a name (a store attached from another
/// graph) is dropped by the enumeration loop, so for that row a declared key
/// *can* still be recovered from the store. The caller reports it and the walk
/// falls back to the full declared set — correct, and off the common path.
fn complete_from_type_schema<S: PropertySink>(
    properties: &mut S,
    node: crate::graph::storage::NodeView<'_>,
    node_type: &str,
    type_meta: &HashMap<String, String>,
    graph: &crate::graph::DirGraph,
    unresolved_key: bool,
) {
    let complete = |properties: &mut S, prop_name: &String| {
        if is_virtual_or_reserved(prop_name) {
            return;
        }
        if properties.contains(prop_name) {
            return;
        }
        let val = resolve_node_property(node, prop_name, graph);
        if !matches!(val, Value::Null) {
            properties.insert(prop_name, val);
        }
    };

    if unresolved_key {
        for prop_name in type_meta.keys() {
            complete(properties, prop_name);
        }
        return;
    }

    // Case 2 — the soft structural aliases, if the type declares them.
    for candidate in crate::graph::schema::SOFT_ALIAS_NAMES {
        if let Some((declared, _)) = type_meta.get_key_value(candidate) {
            complete(properties, declared);
        }
    }
    // Case 3 — the type's spatial virtuals, if it declares a property under a
    // virtual's name. (A type with no spatial config declares none of them, and
    // the whole graph usually has no spatial config at all.)
    if graph.spatial_configs.is_empty() {
        return;
    }
    let Some(config) = graph.get_spatial_config(node_type) else {
        return;
    };
    let named = config
        .points
        .keys()
        .chain(config.shapes.keys())
        .map(String::as_str);
    for candidate in ["location", "geometry"].into_iter().chain(named) {
        if let Some((declared, _)) = type_meta.get_key_value(candidate) {
            complete(properties, declared);
        }
    }
}

/// Phase A.1 / C2 — materialise an edge into an owned [`RelValue`]
/// suitable for `Value::Relationship`. Mirrors `materialize_node_value`.
pub(crate) fn materialize_rel_value(
    edge_idx: petgraph::graph::EdgeIndex,
    graph: &crate::graph::DirGraph,
) -> Option<crate::datatypes::values::RelValue> {
    use crate::datatypes::values::RelValue;
    use std::collections::BTreeMap;
    let edge_data = graph.graph.edge_weight(edge_idx)?;
    let (src, dst) = graph.graph.edge_endpoints(edge_idx)?;
    let mut properties: BTreeMap<String, Value> = BTreeMap::new();
    for key in edge_data.property_keys(&graph.interner) {
        // Reserved provenance keys are engine metadata — kept out of the
        // materialised edge value (direct `r.updated_at` still resolves).
        if crate::graph::schema::is_reserved_provenance_key(key) {
            continue;
        }
        if let Some(val) = edge_data.get_property(key) {
            properties.insert(key.to_string(), val.clone());
        }
    }
    Some(RelValue {
        id: edge_idx.index() as u32,
        start_id: src.index() as u32,
        end_id: dst.index() as u32,
        rel_type: edge_data.connection_type_str(&graph.interner).to_string(),
        properties,
    })
}

/// Phase A.1 / C2 — materialise a variable-length [`PathBinding`]
/// into an owned [`PathValue`] suitable for `Value::Path`.
///
/// Every hop carries its exact edge slot, so parallel relationships and
/// incoming/undirected traversal retain the relationship actually matched.
pub(crate) fn materialize_path_value(
    path: &super::PathBinding,
    graph: &crate::graph::DirGraph,
) -> crate::datatypes::values::PathValue {
    use crate::datatypes::values::PathValue;
    let mut nodes = Vec::with_capacity(path.path.len() + 1);
    let mut rels = Vec::with_capacity(path.path.len());

    if let Some(src_node) = materialize_node_value(path.source, graph) {
        nodes.push(src_node);
    }
    for hop in &path.path {
        if let Some(rel) = materialize_rel_value(hop.edge, graph) {
            rels.push(rel);
        }
        if let Some(node) = materialize_node_value(hop.node, graph) {
            nodes.push(node);
        }
    }
    PathValue { nodes, rels }
}

/// Resolve a string-keyed subscript (`container[key]`) against a map-like
/// value: a `Value::Map`, a node's properties, or a relationship's
/// properties. Per openCypher / Neo4j semantics a missing key resolves to
/// `Value::Null` (never an error), and subscripting a non-map value also
/// yields `Value::Null`.
pub(super) fn map_subscript(container: &Value, key: &str) -> Value {
    match container {
        Value::Map(map) => map.get(key).cloned().unwrap_or(Value::Null),
        Value::Node(node) => node.properties.get(key).cloned().unwrap_or(Value::Null),
        Value::Relationship(rel) => rel.properties.get(key).cloned().unwrap_or(Value::Null),
        _ => Value::Null,
    }
}

/// Parse a list value.
///
/// Phase A.1 / C2 — fast path for native `Value::List`: just clone the
/// items, no parsing needed. The legacy `Value::String("[a, b, c]")`
/// path stays for back-compat with any remaining JSON-string-producing
/// sites and for parameters / literals that come in as strings. Once
/// every producer emits native lists (C4 removes PreProcessedValue;
/// later cleanups retire the string path), this function can shrink
/// to just the List arm.
/// Index a container value with a (possibly negative) integer, cloning **only**
/// the selected element. A native `Value::List` is indexed by reference — the
/// whole list is never cloned, which is what makes `list[i]` O(1) in list
/// length on the hot vector-scoring path rather than O(len) per access. The
/// legacy stringified-list form still parses first; anything else, and any
/// out-of-range index, is `Value::Null` — matching `parse_list_value` + index.
pub(in crate::graph::languages::cypher) fn index_into_value(
    container: &Value,
    integer_index: i64,
) -> Value {
    match container {
        Value::List(items) => index_list_slice(items, integer_index),
        Value::String(_) => index_list_slice(&parse_list_value(container), integer_index),
        _ => Value::Null,
    }
}

/// Shared bounds/negative-index logic for `list[i]`. A negative index counts
/// from the end; anything still out of range yields `Value::Null`.
#[inline]
fn index_list_slice(items: &[Value], integer_index: i64) -> Value {
    let len = items.len() as i64;
    let actual_index = if integer_index < 0 {
        len + integer_index
    } else {
        integer_index
    };
    if actual_index >= 0 && (actual_index as usize) < items.len() {
        items[actual_index as usize].clone()
    } else {
        Value::Null
    }
}

pub(in crate::graph::languages::cypher) fn parse_list_value(val: &Value) -> Vec<Value> {
    match val {
        Value::List(items) => items.clone(),
        Value::String(s) => {
            let trimmed = s.trim();
            if !trimmed.starts_with('[') || !trimmed.ends_with(']') {
                return vec![];
            }
            let inner = &trimmed[1..trimmed.len() - 1];
            if inner.is_empty() {
                return vec![];
            }
            // Split at top-level commas, respecting nesting
            let items = split_top_level_commas(inner);
            items.into_iter().map(parse_value_token).collect()
        }
        _ => vec![],
    }
}

/// Parse a single value token (the same grammar as items inside a
/// formatted list/map). Recognizes integers, floats, booleans, null, the
/// `__nref:N` node-reference sentinel, and quoted strings with `\\`/`\"`
/// escapes. Anything else is returned as a `Value::String` after
/// stripping outer quotes. Mirrors the per-item logic that
/// [`parse_list_value`] used to inline.
pub(super) fn parse_value_token(s: &str) -> Value {
    let trimmed = s.trim();
    if trimmed.is_empty() {
        return Value::Null;
    }
    if let Ok(i) = trimmed.parse::<i64>() {
        return Value::Int64(i);
    }
    if let Ok(f) = trimmed.parse::<f64>() {
        return Value::Float64(f);
    }
    match trimmed {
        "true" => return Value::Boolean(true),
        "false" => return Value::Boolean(false),
        "null" => return Value::Null,
        _ => {}
    }
    // Quoted string: strip the quotes and unescape `\\`/`\"`.
    let bytes = trimmed.as_bytes();
    if bytes.len() >= 2
        && (bytes[0] == b'"' || bytes[0] == b'\'')
        && bytes[bytes.len() - 1] == bytes[0]
    {
        let inner = &trimmed[1..trimmed.len() - 1];
        if let Some(idx_str) = inner.strip_prefix("__nref:") {
            if let Ok(idx) = idx_str.parse::<u32>() {
                return Value::NodeRef(idx);
            }
        }
        // Unescape: `\\` → `\`, `\"` → `"`
        let mut out = String::with_capacity(inner.len());
        let mut chars = inner.chars();
        while let Some(c) = chars.next() {
            if c == '\\' {
                if let Some(next) = chars.next() {
                    out.push(next);
                    continue;
                }
            }
            out.push(c);
        }
        return Value::String(out);
    }
    // Bare token (e.g. an unquoted identifier from format_value_compact).
    Value::String(trimmed.to_string())
}

/// Extract the value of `key` from a map-shaped string of the form
/// `{"k1": v1, "k2": v2, ...}` (the executor's own JSON-style serialization
/// of a [`super::Expression::MapLiteral`]). Returns `None` if the input doesn't
/// look like a map or the key isn't present. The grammar is
/// closed (always emitted by the executor itself), so a small ad-hoc
/// parser is enough — no `serde_json` dependency.
pub(super) fn extract_map_field(s: &str, key: &str) -> Option<Value> {
    let trimmed = s.trim();
    let bytes = trimmed.as_bytes();
    if bytes.len() < 2 || bytes[0] != b'{' || bytes[bytes.len() - 1] != b'}' {
        return None;
    }
    let inner = &trimmed[1..trimmed.len() - 1];
    if inner.trim().is_empty() {
        return None;
    }
    for entry in split_top_level_commas(inner) {
        // Split at the FIRST top-level colon — strings inside the value
        // may contain colons, so we have to respect quoting + nesting.
        let colon_pos = first_top_level_colon(entry)?;
        let raw_key = entry[..colon_pos].trim();
        let raw_val = entry[colon_pos + 1..].trim();
        // Keys are always quoted strings in the formatted output.
        if let Value::String(parsed_key) = parse_value_token(raw_key) {
            if parsed_key == key {
                return Some(parse_value_token(raw_val));
            }
        }
    }
    None
}

/// Pull a named field out of a `Value::Point { lat, lon }` produced by
/// `centroid()`, `point()`, etc. Accepts the canonical Cypher names
/// (`latitude`/`longitude`) plus their short aliases (`lat`/`lon`,
/// `x`/`y`) that some users reach for. Returns `Value::Null` for
/// unknown fields or non-Point inputs.
pub(super) fn point_field(val: &Value, property: &str) -> Value {
    if let Value::Point { lat, lon } = val {
        return match property {
            "latitude" | "lat" | "y" => Value::Float64(*lat),
            "longitude" | "lon" | "lng" | "long" | "x" => Value::Float64(*lon),
            _ => Value::Null,
        };
    }
    Value::Null
}

/// Index of the first top-level `:` in a slice (zero brace/bracket/quote
/// nesting). `None` if no such colon exists.
fn first_top_level_colon(s: &str) -> Option<usize> {
    let mut depth = 0i32;
    let mut in_quotes = false;
    let mut quote_char = '"';
    for (i, ch) in s.char_indices() {
        match ch {
            '"' | '\'' if !in_quotes => {
                in_quotes = true;
                quote_char = ch;
            }
            c if in_quotes && c == quote_char => {
                let bytes = s.as_bytes();
                if i == 0 || bytes[i - 1] != b'\\' {
                    in_quotes = false;
                }
            }
            '{' | '[' | '(' if !in_quotes => depth += 1,
            '}' | ']' | ')' if !in_quotes => depth -= 1,
            ':' if !in_quotes && depth == 0 => return Some(i),
            _ => {}
        }
    }
    None
}

/// Split a string at commas that are not inside braces, brackets, or quotes.
pub(super) fn split_top_level_commas(s: &str) -> Vec<&str> {
    let mut items = Vec::new();
    let mut depth = 0i32; // tracks {}, [], ()
    let mut in_quotes = false;
    let mut quote_char = '"';
    let mut start = 0;

    for (i, ch) in s.char_indices() {
        match ch {
            '"' | '\'' if !in_quotes => {
                in_quotes = true;
                quote_char = ch;
            }
            c if in_quotes && c == quote_char => {
                // Check for escaped quote
                let bytes = s.as_bytes();
                if i == 0 || bytes[i - 1] != b'\\' {
                    in_quotes = false;
                }
            }
            '{' | '[' | '(' if !in_quotes => depth += 1,
            '}' | ']' | ')' if !in_quotes => depth -= 1,
            ',' if !in_quotes && depth == 0 => {
                items.push(&s[start..i]);
                start = i + 1;
            }
            _ => {}
        }
    }
    items.push(&s[start..]);
    items
}

// Delegate to shared value_operations module
pub(super) fn format_value_compact(val: &Value) -> String {
    crate::graph::core::value_operations::format_value_compact(val)
}
pub(super) fn value_to_f64(val: &Value) -> Option<f64> {
    crate::graph::core::value_operations::value_to_f64(val)
}

/// Auto-coerce non-string types (DateTime, Int64, Float64, Boolean) to String
/// for use in string functions. Null stays Null.
pub(super) fn coerce_to_string(val: Value) -> Value {
    match &val {
        Value::String(_) | Value::Null => val,
        _ => Value::String(format_value_compact(&val)),
    }
}

/// Levenshtein edit distance between two UTF-8 strings.
/// Two-row dynamic programming, O(min(n,m)) memory.
pub(super) fn levenshtein(a: &str, b: &str) -> usize {
    if a == b {
        return 0;
    }
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    if a_chars.is_empty() {
        return b_chars.len();
    }
    if b_chars.is_empty() {
        return a_chars.len();
    }
    // Use the shorter string for the row dimension to minimise memory.
    let (short, long) = if a_chars.len() <= b_chars.len() {
        (&a_chars, &b_chars)
    } else {
        (&b_chars, &a_chars)
    };
    let mut prev: Vec<usize> = (0..=short.len()).collect();
    let mut curr: Vec<usize> = vec![0; short.len() + 1];
    for (i, lc) in long.iter().enumerate() {
        curr[0] = i + 1;
        for (j, sc) in short.iter().enumerate() {
            let cost = if lc == sc { 0 } else { 1 };
            curr[j + 1] = (curr[j] + 1).min(prev[j + 1] + 1).min(prev[j] + cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[short.len()]
}

/// Parse a JSON-style float list string "[1.0, 2.0, 3.0]" into Vec<f32>.
pub(super) fn parse_json_float_list(s: &str) -> Result<Vec<f32>, String> {
    let trimmed = s.trim();
    if !trimmed.starts_with('[') || !trimmed.ends_with(']') {
        return Err("vector_score(): query vector must be a list like [1.0, 2.0, ...]".into());
    }
    let inner = &trimmed[1..trimmed.len() - 1];
    if inner.is_empty() {
        return Ok(Vec::new());
    }
    inner
        .split(',')
        .map(|item| {
            item.trim()
                .parse::<f32>()
                .map_err(|_| format!("vector_score(): cannot parse '{}' as a number", item.trim()))
        })
        .collect()
}
#[cfg(test)]
pub(super) fn arithmetic_add(a: &Value, b: &Value) -> Value {
    crate::graph::core::value_operations::arithmetic_add(a, b)
}
#[cfg(test)]
pub(super) fn arithmetic_sub(a: &Value, b: &Value) -> Value {
    crate::graph::core::value_operations::arithmetic_sub(a, b)
}
#[cfg(test)]
pub(super) fn arithmetic_mul(a: &Value, b: &Value) -> Value {
    crate::graph::core::value_operations::arithmetic_mul(a, b)
}
pub(super) fn arithmetic_div(a: &Value, b: &Value) -> Result<Value, String> {
    crate::graph::core::value_operations::arithmetic_div_checked(a, b)
}
pub(super) fn arithmetic_mod(a: &Value, b: &Value) -> Result<Value, String> {
    crate::graph::core::value_operations::arithmetic_mod_checked(a, b)
}
pub(super) fn arithmetic_negate(a: &Value) -> Result<Value, String> {
    crate::graph::core::value_operations::arithmetic_negate_checked(a)
}
pub(super) fn to_integer(val: &Value) -> Value {
    crate::graph::core::value_operations::to_integer(val)
}
pub(super) fn as_i64(val: &Value) -> Result<i64, String> {
    match val {
        Value::Int64(n) => Ok(*n),
        Value::Float64(f) => Ok(*f as i64),
        Value::String(s) => s
            .parse::<i64>()
            .map_err(|_| format!("Cannot convert '{}' to integer", s)),
        _ => Err(format!("Expected integer, got {:?}", val)),
    }
}
pub(super) fn to_float(val: &Value) -> Value {
    crate::graph::core::value_operations::to_float(val)
}
pub(super) fn parse_value_string(s: &str) -> Value {
    crate::graph::core::value_operations::parse_value_string(s)
}

/// Split a list string like "[1, 2, [3, 4], 5]" into top-level items,
/// respecting nested brackets and quoted strings. Returns inner items
/// as string slices. Empty list "[]" returns empty vec.
pub(super) fn split_list_top_level(s: &str) -> Vec<&str> {
    let inner = &s[1..s.len() - 1]; // strip outer []
    if inner.trim().is_empty() {
        return Vec::new();
    }
    let mut items = Vec::new();
    let mut depth = 0i32;
    let mut in_string = false;
    let mut escape = false;
    let mut start = 0;

    for (i, ch) in inner.char_indices() {
        if escape {
            escape = false;
            continue;
        }
        match ch {
            '\\' if in_string => {
                escape = true;
            }
            '"' | '\'' => {
                in_string = !in_string;
            }
            '[' | '{' if !in_string => {
                depth += 1;
            }
            ']' | '}' if !in_string => {
                depth -= 1;
            }
            ',' if !in_string && depth == 0 => {
                items.push(inner[start..i].trim());
                start = i + 1;
            }
            _ => {}
        }
    }
    // Last item
    let last = inner[start..].trim();
    if !last.is_empty() {
        items.push(last);
    }
    items
}

// ============================================================================
// CALL parameter helpers
// ============================================================================

pub(super) fn call_param_f64(params: &HashMap<String, Value>, key: &str, default: f64) -> f64 {
    params
        .get(key)
        .map(|v| match v {
            Value::Float64(f) => *f,
            Value::Int64(i) => *i as f64,
            _ => default,
        })
        .unwrap_or(default)
}

pub(super) fn call_param_usize(
    params: &HashMap<String, Value>,
    key: &str,
    default: usize,
) -> usize {
    params
        .get(key)
        .map(|v| match v {
            Value::Int64(i) => *i as usize,
            Value::Float64(f) => *f as usize,
            _ => default,
        })
        .unwrap_or(default)
}

pub(super) fn call_param_bool(params: &HashMap<String, Value>, key: &str, default: bool) -> bool {
    params
        .get(key)
        .map(|v| match v {
            Value::Boolean(b) => *b,
            _ => default,
        })
        .unwrap_or(default)
}

pub(super) fn call_param_opt_usize(params: &HashMap<String, Value>, key: &str) -> Option<usize> {
    params.get(key).and_then(|v| match v {
        Value::Int64(i) => Some(*i as usize),
        _ => None,
    })
}

pub(super) fn call_param_opt_string(params: &HashMap<String, Value>, key: &str) -> Option<String> {
    params.get(key).and_then(|v| match v {
        Value::String(s) => Some(s.clone()),
        _ => None,
    })
}

pub(super) fn call_param_string_list(
    params: &HashMap<String, Value>,
    key: &str,
) -> Option<Vec<String>> {
    params.get(key).and_then(|v| match v {
        // Phase A.1 / C4 — native Value::List from list literals
        // (`connection_types: ['CALLS', 'IMPORTS']`).
        Value::List(items) => {
            let strs: Vec<String> = items
                .iter()
                .filter_map(|item| match item {
                    Value::String(s) => Some(s.clone()),
                    _ => None,
                })
                .collect();
            if strs.is_empty() {
                None
            } else {
                Some(strs)
            }
        }
        Value::String(s) => {
            if s.starts_with('[') {
                // Legacy JSON-string list (kept as fallback for
                // parameters/literals that come in as strings).
                let items = parse_list_value(v);
                if items.is_empty() {
                    return None;
                }
                Some(
                    items
                        .into_iter()
                        .filter_map(|item| match item {
                            Value::String(s) => Some(s),
                            _ => None,
                        })
                        .collect(),
                )
            } else {
                Some(vec![s.clone()])
            }
        }
        _ => None,
    })
}

/// Look up a required string CALL param. Returns `None` if the param
/// is absent or non-string; callers turn `None` into an error.
pub(super) fn call_param_string(params: &HashMap<String, Value>, key: &str) -> Option<String> {
    params.get(key).and_then(|v| match v {
        Value::String(s) => Some(s.clone()),
        _ => None,
    })
}

/// CALL procedure helper: look up a YIELD column's user-facing alias
/// (returns the alias if `YIELD name AS alias` was used, or the bare
/// column name when no alias). Returns `None` if the column wasn't
/// listed in the YIELD clause — caller can skip emitting it.
///
/// Consolidated 0.9.53 from `affected_tests.rs` and `refresh_stats.rs`.
pub(super) fn yield_alias(yield_items: &[YieldItem], expected: &str) -> Option<String> {
    yield_items
        .iter()
        .find(|y| y.name == expected)
        .map(|item| item.alias.clone().unwrap_or_else(|| expected.to_string()))
}

#[cfg(test)]
#[path = "node_record_golden_tests.rs"]
mod node_record_golden_tests;
