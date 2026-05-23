//! Shared executor helpers — expression-to-string, predicate-to-string,
//! value comparison, arithmetic, type coercion, property resolution, and
//! CALL parameter extractors.

use super::super::ast::*;
use super::super::result::*;
use crate::datatypes::values::Value;
use crate::graph::schema::{DirGraph, NodeData};
use crate::graph::storage::GraphRead;
use std::collections::HashMap;
use std::sync::RwLock;

// Re-export the ast aggregate helpers so downstream code can refer to them
// via the executor namespace (backward compatibility with pre-split code).
pub use super::super::ast::is_aggregate_expression;

// ============================================================================
// Helper Functions
// ============================================================================

// is_aggregate_expression and is_window_expression are re-exported above.

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

/// Convert an expression to its string representation (for column naming)
pub(super) fn expression_to_string(expr: &Expression) -> String {
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
        Predicate::LabelCheck { variable, label } => format!("{}:{}", variable, label),
        _ => "predicate(...)".to_string(),
    }
}

/// Evaluate a comparison using existing filtering infrastructure
pub(super) fn evaluate_comparison(
    left: &Value,
    op: &ComparisonOp,
    right: &Value,
    regex_cache: Option<&RwLock<HashMap<String, regex::Regex>>>,
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
                // Try cached regex first
                if let Some(cache) = regex_cache {
                    {
                        let read = cache.read().unwrap();
                        if let Some(re) = read.get(pattern.as_str()) {
                            return Ok(re.is_match(text));
                        }
                    }
                    let re = regex::Regex::new(pattern)
                        .map_err(|e| format!("Invalid regular expression '{}': {}", pattern, e))?;
                    let result = re.is_match(text);
                    cache.write().unwrap().insert(pattern.clone(), re);
                    Ok(result)
                } else {
                    match regex::Regex::new(pattern) {
                        Ok(re) => Ok(re.is_match(text)),
                        Err(e) => Err(format!("Invalid regular expression '{}': {}", pattern, e)),
                    }
                }
            }
            _ => Ok(false),
        },
    }
}

/// Resolve a node property, returning an owned Value directly.
/// Uses `get_property_value()` to avoid Cow wrapping/unwrapping overhead.
pub(crate) fn resolve_node_property(node: &NodeData, property: &str, graph: &DirGraph) -> Value {
    let node_type_str = node.node_type_str(&graph.interner);
    let resolved = graph.resolve_alias(node_type_str, property);
    match resolved {
        "id" => node.id().into_owned(),
        "title" | "name" => node.title().into_owned(),
        "type" | "node_type" | "label" => Value::String(node_type_str.to_string()),
        _ => {
            if let Some(val) = node.get_property_value(resolved) {
                return val;
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
pub(crate) fn resolve_edge_property(graph: &DirGraph, edge: &EdgeBinding, property: &str) -> Value {
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
pub(super) fn node_to_map_value(node: &NodeData) -> Value {
    node.title().into_owned()
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
    use crate::datatypes::values::NodeValue;
    use std::collections::BTreeMap;
    let node = graph.graph.node_weight(idx)?;
    let node_type = node.node_type_str(&graph.interner).to_string();
    let mut properties: BTreeMap<String, Value> = BTreeMap::new();
    // Include the three virtual builtins so consumers always see
    // id/title/type, matching what n.id / n.title / n.type would
    // resolve to via the alias machinery.
    properties.insert("id".to_string(), node.id().into_owned());
    properties.insert("title".to_string(), node.title().into_owned());
    properties.insert("type".to_string(), Value::String(node_type.clone()));
    // Then every user-set property the node carries.
    //
    // Cross-backend discovery: in memory mode NodeData.property_iter
    // returns every property. In disk/mapped mode the property values
    // live in the column store, NOT in NodeData — property_iter
    // returns nothing useful there. Use node_type_metadata as the
    // schema source for "which properties does this type have", then
    // read each via `resolve_node_property` (which knows the
    // backend-specific access path).
    for (key, val) in node.property_iter(&graph.interner) {
        if key == "id" || key == "title" || key == "type" {
            continue;
        }
        properties.insert(key.to_string(), val.clone());
    }
    if let Some(type_meta) = graph.get_node_type_metadata(&node_type) {
        for prop_name in type_meta.keys() {
            if prop_name == "id" || prop_name == "title" || prop_name == "type" {
                continue;
            }
            if properties.contains_key(prop_name) {
                continue;
            }
            let val = resolve_node_property(node, prop_name, graph);
            if !matches!(val, Value::Null) {
                properties.insert(prop_name.clone(), val);
            }
        }
    }
    Some(NodeValue {
        id: idx.index() as u32,
        labels: vec![node_type],
        properties,
    })
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
        if let Some(val) = edge_data.get_property(key) {
            properties.insert(key.to_string(), val.clone());
        }
    }
    Some(RelValue {
        id: edge_idx.index() as u32,
        start_id: src.index() as u32,
        end_id: dst.index() as u32,
        rel_type: edge_data
            .connection_type_str(&graph.interner)
            .to_string(),
        properties,
    })
}

/// Phase A.1 / C2 — materialise a variable-length [`PathBinding`]
/// into an owned [`PathValue`] suitable for `Value::Path`.
///
/// The path stores `(NodeIndex, connection_type_string)` per hop
/// rather than `EdgeIndex`, so we look up each edge via
/// `petgraph::find_edge(prev, curr)` to recover its properties. If
/// multiple parallel edges exist between two nodes, we pick the first
/// — same indeterminacy as the rest of the variable-length matcher.
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
    let mut prev_idx = path.source;
    for (node_idx, _conn_type) in &path.path {
        if let Some(edge_idx) = graph.graph.find_edge(prev_idx, *node_idx) {
            if let Some(rel) = materialize_rel_value(edge_idx, graph) {
                rels.push(rel);
            }
        }
        if let Some(node) = materialize_node_value(*node_idx, graph) {
            nodes.push(node);
        }
        prev_idx = *node_idx;
    }
    PathValue { nodes, rels }
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
pub(super) fn parse_list_value(val: &Value) -> Vec<Value> {
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
/// `{"k1": v1, "k2": v2, ...}` produced by [`format_value_json`] over a
/// [`super::Expression::MapLiteral`]. Returns `None` if the input doesn't
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
/// JSON-safe value formatting: strings are quoted, others are as-is.
/// Used for list serialization so py_convert can parse via json.loads.
pub(super) fn format_value_json(val: &Value) -> String {
    match val {
        Value::String(s) => format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\"")),
        Value::Null => "null".to_string(),
        Value::Boolean(b) => if *b { "true" } else { "false" }.to_string(),
        Value::NodeRef(idx) => format!("\"__nref:{}\"", idx),
        _ => format_value_compact(val),
    }
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
pub(super) fn arithmetic_add(a: &Value, b: &Value) -> Value {
    crate::graph::core::value_operations::arithmetic_add(a, b)
}
pub(super) fn arithmetic_sub(a: &Value, b: &Value) -> Value {
    crate::graph::core::value_operations::arithmetic_sub(a, b)
}
pub(super) fn arithmetic_mul(a: &Value, b: &Value) -> Value {
    crate::graph::core::value_operations::arithmetic_mul(a, b)
}
pub(super) fn arithmetic_div(a: &Value, b: &Value) -> Value {
    crate::graph::core::value_operations::arithmetic_div(a, b)
}
pub(super) fn arithmetic_mod(a: &Value, b: &Value) -> Value {
    crate::graph::core::value_operations::arithmetic_mod(a, b)
}
pub(super) fn arithmetic_negate(a: &Value) -> Value {
    crate::graph::core::value_operations::arithmetic_negate(a)
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
