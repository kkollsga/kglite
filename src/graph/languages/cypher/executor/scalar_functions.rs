//! Cypher executor — scalar (non-aggregate) function dispatch.
//!
//! Split out of `expression.rs` to keep that file under the 2,500-line cap.
//! Lives in a sibling `impl<'a> CypherExecutor<'a> {}` block.

use super::helpers::*;
use super::*;
use crate::datatypes::values::Value;
use crate::graph::algorithms::vector as vs;
use crate::graph::storage::GraphRead;

/// Shared error suffix when a spatial function arg can't be resolved to a
/// geometry or point. Names the conventional property names that the
/// fallback inference (in `build_node_spatial_data`) accepts so users have
/// a quick fix. Also surfaced from `resolve_spatial` when a node has no
/// registered spatial config and no inferable conventional fields.
pub(super) const SPATIAL_RESOLUTION_HELP: &str =
    "spatial argument did not resolve to a geometry or point. \
Either pass column_types={'<col>': 'geometry'} (or 'location.lat'/'location.lon') during \
add_nodes(), or store the data under a conventional property name (wkt_geometry, geometry, \
geom, or wkt for WKT; latitude+longitude or lat+lon for points).";

impl<'a> CypherExecutor<'a> {
    /// Evaluate scalar (non-aggregate) functions
    pub(super) fn evaluate_scalar_function(
        &self,
        name: &str,
        args: &[Expression],
        row: &ResultRow,
    ) -> Result<Value, String> {
        match name {
            "toupper" | "touppercase" => {
                let val = self.evaluate_expression(&args[0], row)?;
                match val {
                    Value::String(s) => Ok(Value::String(s.to_uppercase())),
                    _ => Ok(Value::Null),
                }
            }
            "tolower" | "tolowercase" => {
                let val = self.evaluate_expression(&args[0], row)?;
                match val {
                    Value::String(s) => Ok(Value::String(s.to_lowercase())),
                    _ => Ok(Value::Null),
                }
            }
            "tostring" => {
                let val = self.evaluate_expression(&args[0], row)?;
                Ok(Value::String(format_value_compact(&val)))
            }
            "tointeger" | "toint" => {
                let val = self.evaluate_expression(&args[0], row)?;
                Ok(to_integer(&val))
            }
            "tofloat" => {
                let val = self.evaluate_expression(&args[0], row)?;
                Ok(to_float(&val))
            }
            "date" => {
                if args.len() != 1 {
                    return Err("date() requires 1 argument: date('2020-01-15')".into());
                }
                let val = self.evaluate_expression(&args[0], row)?;
                match val {
                    Value::String(s) => {
                        // Return Null on invalid input instead of crashing (BUG-09)
                        match crate::graph::features::timeseries::parse_date_query(&s) {
                            Ok((d, _)) => Ok(Value::DateTime(d)),
                            Err(_) => Ok(Value::Null),
                        }
                    }
                    Value::DateTime(_) => Ok(val),
                    Value::Null => Ok(Value::Null),
                    _ => Err(format!("date() argument must be a string, got {:?}", val)),
                }
            }
            "datetime" => {
                // 0-arg form returns "now" (today's date — Value::DateTime
                // is NaiveDate, so subsecond precision is dropped).
                // 0.9.0 §3.
                if args.is_empty() {
                    return Ok(Value::DateTime(chrono::Local::now().date_naive()));
                }
                if args.len() != 1 {
                    return Err(
                        "datetime() requires 0 or 1 argument: datetime() or datetime('2024-03-15T10:30:00')".into(),
                    );
                }
                let val = self.evaluate_expression(&args[0], row)?;
                match val {
                    Value::String(s) => {
                        // Try parsing as ISO datetime with T separator
                        if s.contains('T') {
                            let date_part = s.split('T').next().unwrap_or("");
                            match crate::graph::features::timeseries::parse_date_query(date_part) {
                                Ok((d, _)) => Ok(Value::DateTime(d)),
                                Err(_) => Ok(Value::Null),
                            }
                        } else {
                            // Fallback: try as plain date
                            match crate::graph::features::timeseries::parse_date_query(&s) {
                                Ok((d, _)) => Ok(Value::DateTime(d)),
                                Err(_) => Ok(Value::Null),
                            }
                        }
                    }
                    Value::DateTime(_) => Ok(val),
                    Value::Null => Ok(Value::Null),
                    _ => Err(format!(
                        "datetime() argument must be a string, got {:?}",
                        val
                    )),
                }
            }
            "date_diff" | "datediff" => {
                if args.len() != 2 {
                    return Err("date_diff() requires 2 date arguments".into());
                }
                let a = self.evaluate_expression(&args[0], row)?;
                let b = self.evaluate_expression(&args[1], row)?;
                match (&a, &b) {
                    (Value::DateTime(d1), Value::DateTime(d2)) => {
                        Ok(Value::Int64((*d1 - *d2).num_days()))
                    }
                    (Value::Null, _) | (_, Value::Null) => Ok(Value::Null),
                    _ => Err("date_diff() arguments must be dates".into()),
                }
            }
            // 0.9.0 §3 / Cluster 2 — proper Value::Duration variant.
            // Calendar units (months/years) and clock units
            // (days/hours/minutes/seconds) are kept separate in the
            // value, so `duration({months: 1, days: 5}).months` returns
            // 1, not 35. Sub-day precision is wired in `seconds` —
            // DateTime + Duration discards the seconds component for
            // now because Value::DateTime is still NaiveDate (Cluster
            // 1, deferred).
            "duration" => {
                if args.len() != 1 {
                    return Err("duration() requires 1 map argument: duration({days: N})".into());
                }
                if let Expression::MapLiteral(entries) = &args[0] {
                    let mut months: i64 = 0;
                    let mut days: i64 = 0;
                    let mut seconds: i64 = 0;
                    for (key, expr) in entries {
                        let v = self.evaluate_expression(expr, row)?;
                        let n = match v {
                            Value::Int64(n) => n,
                            Value::Float64(f) => f as i64,
                            Value::Null => 0,
                            _ => {
                                return Err(format!("duration({{{key}: ...}}) expects a number"));
                            }
                        };
                        match key.as_str() {
                            "years" => months += n * 12,
                            "months" => months += n,
                            "weeks" => days += n * 7,
                            "days" => days += n,
                            "hours" => seconds += n * 3600,
                            "minutes" => seconds += n * 60,
                            "seconds" => seconds += n,
                            other => {
                                return Err(format!(
                                    "duration(): unknown key '{other}' (expected years/months/weeks/days/hours/minutes/seconds)"
                                ));
                            }
                        }
                    }
                    Ok(Value::Duration {
                        months: months as i32,
                        days: days as i32,
                        seconds,
                    })
                } else {
                    Err("duration() requires a map literal: duration({days: N})".into())
                }
            }
            "duration.between" => {
                if args.len() != 2 {
                    return Err("duration.between() requires 2 datetime arguments".into());
                }
                let a = self.evaluate_expression(&args[0], row)?;
                let b = self.evaluate_expression(&args[1], row)?;
                match (&a, &b) {
                    (Value::DateTime(d1), Value::DateTime(d2)) => {
                        // Whole-day delta carried in `days`. Months and
                        // seconds are 0 — Value::DateTime is date-only.
                        Ok(Value::Duration {
                            months: 0,
                            days: (*d2 - *d1).num_days() as i32,
                            seconds: 0,
                        })
                    }
                    (Value::Null, _) | (_, Value::Null) => Ok(Value::Null),
                    _ => Err("duration.between() arguments must be datetime values".into()),
                }
            }
            "size" => {
                // Phase A.1 / C2 — native Value::List fast path;
                // string fallback stays for legacy collect-as-JSON
                // and parameter-passed lists.
                let val = self.evaluate_expression(&args[0], row)?;
                match val {
                    Value::List(items) => Ok(Value::Int64(items.len() as i64)),
                    Value::Map(m) => Ok(Value::Int64(m.len() as i64)),
                    Value::String(s) => {
                        if s.starts_with('[') && s.ends_with(']') {
                            let items = parse_list_value(&Value::String(s));
                            Ok(Value::Int64(items.len() as i64))
                        } else {
                            Ok(Value::Int64(s.len() as i64))
                        }
                    }
                    _ => Ok(Value::Null),
                }
            }
            "length" => {
                // length(p) for paths, length(s) for strings, length(list) for lists
                if let Some(Expression::Variable(var)) = args.first() {
                    if let Some(path) = row.path_bindings.get(var) {
                        return Ok(Value::Int64(path.hops as i64));
                    }
                }
                let val = self.evaluate_expression(&args[0], row)?;
                match val {
                    // Phase A.1 / C2 — native Value::List/Path/Map paths.
                    Value::List(items) => Ok(Value::Int64(items.len() as i64)),
                    Value::Map(m) => Ok(Value::Int64(m.len() as i64)),
                    Value::Path(p) => Ok(Value::Int64(p.rels.len() as i64)),
                    Value::String(s) => {
                        if s.starts_with('[') && s.ends_with(']') {
                            let items = parse_list_value(&Value::String(s));
                            Ok(Value::Int64(items.len() as i64))
                        } else {
                            Ok(Value::Int64(s.len() as i64))
                        }
                    }
                    _ => Ok(Value::Null),
                }
            }
            "nodes" => {
                // nodes(p) returns the list of nodes in a path
                // (source + intermediates + target).
                //
                // Phase A.1 / C2 — native `Value::List(Vec<Value::Node>)`.
                // Each element is a full NodeValue (id, labels, properties)
                // mirroring what `RETURN n` would emit. Replaces the
                // pre-A.1 JSON-string list of dicts.
                if let Some(Expression::Variable(var)) = args.first() {
                    if let Some(path) = row.path_bindings.get(var) {
                        let mut items: Vec<Value> = Vec::with_capacity(path.path.len() + 1);
                        if let Some(src) = materialize_node_value(path.source, self.graph) {
                            items.push(Value::Node(Box::new(src)));
                        }
                        for (node_idx, _conn_type) in &path.path {
                            if let Some(node) = materialize_node_value(*node_idx, self.graph) {
                                items.push(Value::Node(Box::new(node)));
                            }
                        }
                        return Ok(Value::List(items));
                    }
                }
                Ok(Value::Null)
            }
            "relationships" | "rels" => {
                // relationships(p) — list of relationships in a path.
                //
                // Phase A.1 / C2 — native `Value::List(Vec<Value::Relationship>)`.
                // Each element is a full RelValue (id, start, end, type,
                // properties), recovered by walking the path's
                // (node_idx, _) pairs and looking up the connecting
                // edge between consecutive nodes (mirrors
                // materialize_path_value's hop-recovery).
                if let Some(Expression::Variable(var)) = args.first() {
                    if let Some(path) = row.path_bindings.get(var) {
                        let mut items: Vec<Value> = Vec::with_capacity(path.path.len());
                        let mut prev_idx = path.source;
                        for (node_idx, _conn_type) in &path.path {
                            if let Some(edge_idx) =
                                self.graph.graph.find_edge(prev_idx, *node_idx)
                            {
                                if let Some(rel) = materialize_rel_value(edge_idx, self.graph) {
                                    items.push(Value::Relationship(Box::new(rel)));
                                }
                            }
                            prev_idx = *node_idx;
                        }
                        return Ok(Value::List(items));
                    }
                }
                Ok(Value::Null)
            }
            "type" => {
                // type(r) returns the relationship type
                if let Some(Expression::Variable(var)) = args.first() {
                    if let Some(edge) = row.edge_bindings.get(var) {
                        if let Some(edge_data) = {
                            let g = &self.graph.graph;
                            g.edge_weight(edge.edge_index)
                        } {
                            return Ok(Value::String(
                                edge_data
                                    .connection_type_str(&self.graph.interner)
                                    .to_string(),
                            ));
                        }
                    }
                }
                Ok(Value::Null)
            }
            "id" => {
                // id(n) returns the node id
                if let Some(Expression::Variable(var)) = args.first() {
                    if let Some(&idx) = row.node_bindings.get(var) {
                        if let Some(node) = self.graph.graph.node_weight(idx) {
                            return Ok(resolve_node_property(node, "id", self.graph));
                        }
                    }
                }
                Ok(Value::Null)
            }
            "labels" => {
                // labels(n) returns the list of node labels.
                //
                // Phase A.1 / C2 — flipped the Track C swap-point:
                // emits native `Value::List(Vec<Value::String>)` now
                // that the variant exists. KGLite is still single-
                // label so the list has one element. When multi-label
                // lands (ROADMAP §5), this returns N elements with
                // no consumer changes required.
                if let Some(Expression::Variable(var)) = args.first() {
                    if let Some(&idx) = row.node_bindings.get(var) {
                        if let Some(node) = self.graph.graph.node_weight(idx) {
                            let node_type = node.get_node_type_ref(&self.graph.interner);
                            return Ok(Value::List(vec![Value::String(node_type.to_string())]));
                        }
                    }
                }
                Ok(Value::Null)
            }
            "keys" => {
                // keys(n) or keys(r) — return property names as a list.
                //
                // Phase A.1 / C2 — native `Value::List(Vec<Value::String>)`.
                // Includes both the virtual aliases (`id`, `title`, `type`)
                // and the user's original column names (the unique-id-field
                // and title-field passed to add_nodes), so the output
                // matches what `n.<name>` would resolve at query time.
                if let Some(Expression::Variable(var)) = args.first() {
                    if let Some(&idx) = row.node_bindings.get(var) {
                        if let Some(node) = self.graph.graph.node_weight(idx) {
                            let node_type = node.node_type_str(&self.graph.interner);
                            let mut keys: Vec<String> =
                                vec!["id".to_string(), "title".to_string(), "type".to_string()];
                            if let Some(id_alias) = self.graph.id_field_aliases.get(node_type) {
                                keys.push(id_alias.clone());
                            }
                            if let Some(title_alias) = self.graph.title_field_aliases.get(node_type)
                            {
                                keys.push(title_alias.clone());
                            }
                            keys.extend(node.property_keys(&self.graph.interner).map(String::from));
                            keys.sort();
                            keys.dedup();
                            return Ok(Value::List(
                                keys.into_iter().map(Value::String).collect(),
                            ));
                        }
                    }
                    if let Some(edge) = row.edge_bindings.get(var) {
                        if let Some(edge_data) = {
                            let g = &self.graph.graph;
                            g.edge_weight(edge.edge_index)
                        } {
                            let mut keys: Vec<String> = vec!["type".to_string()];
                            keys.extend(
                                edge_data
                                    .property_keys(&self.graph.interner)
                                    .map(String::from),
                            );
                            keys.sort();
                            return Ok(Value::List(
                                keys.into_iter().map(Value::String).collect(),
                            ));
                        }
                    }
                }
                Ok(Value::Null)
            }
            "coalesce" => {
                // coalesce(expr1, expr2, ...) returns first non-null
                for arg in args {
                    let val = self.evaluate_expression(arg, row)?;
                    if !matches!(val, Value::Null) {
                        return Ok(val);
                    }
                }
                Ok(Value::Null)
            }
            "properties" => {
                // properties(n) / properties(r) → native Value::Map.
                //
                // Phase A.1 / C2 — emits `Value::Map(BTreeMap)` directly.
                // For nodes, includes the virtual id/title/type plus
                // every user-set property (mirrors materialize_node_value).
                // For relationships, includes `type` + every user-set
                // edge property.
                if args.len() != 1 {
                    return Err("properties() requires 1 argument: a node or relationship".into());
                }
                if let Expression::Variable(var) = &args[0] {
                    if let Some(&idx) = row.node_bindings.get(var.as_str()) {
                        if let Some(node) = self.graph.graph.node_weight(idx) {
                            let mut props: std::collections::BTreeMap<String, Value> =
                                std::collections::BTreeMap::new();
                            for &builtin in &["id", "title", "type"] {
                                let val = resolve_node_property(node, builtin, self.graph);
                                if !matches!(val, Value::Null) {
                                    props.insert(builtin.to_string(), val);
                                }
                            }
                            for key in node.property_keys(&self.graph.interner) {
                                if key == "id" || key == "title" || key == "type" {
                                    continue;
                                }
                                let val = resolve_node_property(node, key, self.graph);
                                props.insert(key.to_string(), val);
                            }
                            return Ok(Value::Map(props));
                        }
                    }
                    if let Some(edge) = row.edge_bindings.get(var.as_str()) {
                        if let Some(edge_data) = {
                            let g = &self.graph.graph;
                            g.edge_weight(edge.edge_index)
                        } {
                            let mut props: std::collections::BTreeMap<String, Value> =
                                std::collections::BTreeMap::new();
                            props.insert(
                                "type".to_string(),
                                Value::String(
                                    edge_data
                                        .connection_type_str(&self.graph.interner)
                                        .to_string(),
                                ),
                            );
                            for key in edge_data.property_keys(&self.graph.interner) {
                                if let Some(val) = edge_data.get_property(key) {
                                    props.insert(key.to_string(), val.clone());
                                }
                            }
                            return Ok(Value::Map(props));
                        }
                    }
                }
                Ok(Value::Null)
            }
            "start_node" | "startnode" => {
                // start_node(r) / startNode(r) → source node of the
                // bound relationship in the graph. Look up via
                // `edge_index` rather than `EdgeBinding.source` —
                // the binding stores the pattern's left endpoint,
                // which is *not* the same as the edge's graph source
                // when the matcher anchored on the right endpoint and
                // walked incoming.
                if let Some(Expression::Variable(var)) = args.first() {
                    if let Some(edge) = row.edge_bindings.get(var.as_str()) {
                        if let Some((src, _)) = self.graph.graph.edge_endpoints(edge.edge_index) {
                            return Ok(Value::NodeRef(src.index() as u32));
                        }
                    }
                }
                Ok(Value::Null)
            }
            "end_node" | "endnode" => {
                // end_node(r) / endNode(r) → target node of the
                // bound relationship in the graph. See `start_node`
                // above for the reason we go through `edge_index`.
                if let Some(Expression::Variable(var)) = args.first() {
                    if let Some(edge) = row.edge_bindings.get(var.as_str()) {
                        if let Some((_, tgt)) = self.graph.graph.edge_endpoints(edge.edge_index) {
                            return Ok(Value::NodeRef(tgt.index() as u32));
                        }
                    }
                }
                Ok(Value::Null)
            }
            // ── Text predicates (0.8.20) ──────────────────────────────
            "text_edit_distance" => {
                if args.len() != 2 {
                    return Err("text_edit_distance() requires 2 arguments".into());
                }
                let a = coerce_to_string(self.evaluate_expression(&args[0], row)?);
                let b = coerce_to_string(self.evaluate_expression(&args[1], row)?);
                match (&a, &b) {
                    (Value::String(s1), Value::String(s2)) => {
                        Ok(Value::Int64(levenshtein(s1, s2) as i64))
                    }
                    _ => Ok(Value::Null),
                }
            }
            "text_normalize" => {
                if args.len() != 1 {
                    return Err("text_normalize() requires 1 argument".into());
                }
                let val = coerce_to_string(self.evaluate_expression(&args[0], row)?);
                match val {
                    Value::String(s) => {
                        let mut out = String::with_capacity(s.len());
                        let mut last_space = true;
                        for c in s.chars() {
                            if c.is_alphanumeric() {
                                for lc in c.to_lowercase() {
                                    out.push(lc);
                                }
                                last_space = false;
                            } else if c.is_whitespace() && !last_space {
                                out.push(' ');
                                last_space = true;
                            }
                            // punctuation: drop
                        }
                        Ok(Value::String(out.trim().to_string()))
                    }
                    _ => Ok(Value::Null),
                }
            }
            "text_jaccard" => {
                if args.len() < 2 || args.len() > 3 {
                    return Err(
                        "text_jaccard() requires 2-3 arguments: (a, b [, separator])".into(),
                    );
                }
                let a = coerce_to_string(self.evaluate_expression(&args[0], row)?);
                let b = coerce_to_string(self.evaluate_expression(&args[1], row)?);
                let sep = if args.len() == 3 {
                    match self.evaluate_expression(&args[2], row)? {
                        Value::String(s) => Some(s),
                        _ => return Err("text_jaccard(): separator must be a string".into()),
                    }
                } else {
                    None
                };
                match (&a, &b) {
                    (Value::String(s1), Value::String(s2)) => {
                        let tokenize = |s: &str| -> std::collections::HashSet<String> {
                            match &sep {
                                Some(d) => s.split(d.as_str()).map(|t| t.to_string()).collect(),
                                None => s.split_whitespace().map(|t| t.to_string()).collect(),
                            }
                        };
                        let set_a = tokenize(s1);
                        let set_b = tokenize(s2);
                        if set_a.is_empty() && set_b.is_empty() {
                            return Ok(Value::Float64(1.0));
                        }
                        let inter = set_a.intersection(&set_b).count() as f64;
                        let union = set_a.union(&set_b).count() as f64;
                        Ok(Value::Float64(inter / union))
                    }
                    _ => Ok(Value::Null),
                }
            }
            "text_ngrams" => {
                // Phase A.1 / C4 — native Value::List of Value::String.
                if args.len() != 2 {
                    return Err("text_ngrams() requires 2 arguments: (string, n)".into());
                }
                let s_val = coerce_to_string(self.evaluate_expression(&args[0], row)?);
                let n_val = self.evaluate_expression(&args[1], row)?;
                match (&s_val, &n_val) {
                    (Value::String(s), Value::Int64(n)) => {
                        let n = *n as usize;
                        if n == 0 {
                            return Err("text_ngrams(): n must be ≥ 1".into());
                        }
                        let chars: Vec<char> = s.chars().collect();
                        let mut grams: Vec<Value> = Vec::new();
                        if chars.len() >= n {
                            for i in 0..=chars.len() - n {
                                let gram: String = chars[i..i + n].iter().collect();
                                grams.push(Value::String(gram));
                            }
                        }
                        Ok(Value::List(grams))
                    }
                    (Value::Null, _) | (_, Value::Null) => Ok(Value::Null),
                    _ => Ok(Value::Null),
                }
            }
            "text_contains_any" => {
                if args.is_empty() {
                    return Err("text_contains_any() requires at least 1 argument".into());
                }
                let s_val = coerce_to_string(self.evaluate_expression(&args[0], row)?);
                let s = match &s_val {
                    Value::String(s) => s.clone(),
                    _ => return Ok(Value::Null),
                };
                // Phase A.1 / C4 — accept native Value::List, legacy
                // JSON-string list, single-string second arg, or
                // variadic remaining args.
                if args.len() == 2 {
                    let list_val = self.evaluate_expression(&args[1], row)?;
                    if let Value::List(needles) = &list_val {
                        for needle in needles {
                            if let Value::String(n) = needle {
                                if s.contains(n.as_str()) {
                                    return Ok(Value::Boolean(true));
                                }
                            }
                        }
                        return Ok(Value::Boolean(false));
                    }
                    if let Value::String(ref ls) = list_val {
                        if ls.starts_with('[') && ls.ends_with(']') {
                            let needles = parse_list_value(&list_val);
                            for needle in needles {
                                if let Value::String(n) = needle {
                                    if s.contains(n.as_str()) {
                                        return Ok(Value::Boolean(true));
                                    }
                                }
                            }
                            return Ok(Value::Boolean(false));
                        }
                        if s.contains(ls.as_str()) {
                            return Ok(Value::Boolean(true));
                        }
                        return Ok(Value::Boolean(false));
                    }
                }
                for arg in &args[1..] {
                    let needle = self.evaluate_expression(arg, row)?;
                    if let Value::String(n) = needle {
                        if s.contains(n.as_str()) {
                            return Ok(Value::Boolean(true));
                        }
                    }
                }
                Ok(Value::Boolean(false))
            }
            "text_starts_with_any" => {
                if args.is_empty() {
                    return Err("text_starts_with_any() requires at least 1 argument".into());
                }
                let s_val = coerce_to_string(self.evaluate_expression(&args[0], row)?);
                let s = match &s_val {
                    Value::String(s) => s.clone(),
                    _ => return Ok(Value::Null),
                };
                // Phase A.1 / C4 — same native-list handling as
                // text_contains_any.
                if args.len() == 2 {
                    let list_val = self.evaluate_expression(&args[1], row)?;
                    if let Value::List(prefixes) = &list_val {
                        for prefix in prefixes {
                            if let Value::String(p) = prefix {
                                if s.starts_with(p.as_str()) {
                                    return Ok(Value::Boolean(true));
                                }
                            }
                        }
                        return Ok(Value::Boolean(false));
                    }
                    if let Value::String(ref ls) = list_val {
                        if ls.starts_with('[') && ls.ends_with(']') {
                            let prefixes = parse_list_value(&list_val);
                            for prefix in prefixes {
                                if let Value::String(p) = prefix {
                                    if s.starts_with(p.as_str()) {
                                        return Ok(Value::Boolean(true));
                                    }
                                }
                            }
                            return Ok(Value::Boolean(false));
                        }
                        if s.starts_with(ls.as_str()) {
                            return Ok(Value::Boolean(true));
                        }
                        return Ok(Value::Boolean(false));
                    }
                }
                for arg in &args[1..] {
                    let prefix = self.evaluate_expression(arg, row)?;
                    if let Value::String(p) = prefix {
                        if s.starts_with(p.as_str()) {
                            return Ok(Value::Boolean(true));
                        }
                    }
                }
                Ok(Value::Boolean(false))
            }
            // ── String functions ──────────────────────────────────
            "split" => {
                if args.len() != 2 {
                    return Err("split() requires 2 arguments: string, delimiter".into());
                }
                let str_val = coerce_to_string(self.evaluate_expression(&args[0], row)?);
                let delim_val = self.evaluate_expression(&args[1], row)?;
                match (&str_val, &delim_val) {
                    (Value::String(s), Value::String(delim)) => {
                        let parts: Vec<String> = s
                            .split(delim.as_str())
                            .map(|p| {
                                format!("\"{}\"", p.replace('\\', "\\\\").replace('"', "\\\""))
                            })
                            .collect();
                        Ok(Value::String(format!("[{}]", parts.join(", "))))
                    }
                    _ => Ok(Value::Null),
                }
            }
            "replace" => {
                if args.len() != 3 {
                    return Err(
                        "replace() requires 3 arguments: string, search, replacement".into(),
                    );
                }
                let str_val = coerce_to_string(self.evaluate_expression(&args[0], row)?);
                let search_val = self.evaluate_expression(&args[1], row)?;
                let replace_val = self.evaluate_expression(&args[2], row)?;
                match (&str_val, &search_val, &replace_val) {
                    (Value::String(s), Value::String(search), Value::String(replacement)) => Ok(
                        Value::String(s.replace(search.as_str(), replacement.as_str())),
                    ),
                    _ => Ok(Value::Null),
                }
            }
            "substring" => {
                if args.len() < 2 || args.len() > 3 {
                    return Err(
                        "substring() requires 2-3 arguments: string, start [, length]".into(),
                    );
                }
                let str_val = coerce_to_string(self.evaluate_expression(&args[0], row)?);
                let start_val = self.evaluate_expression(&args[1], row)?;
                match (&str_val, &start_val) {
                    (Value::String(s), Value::Int64(start)) => {
                        let start_idx = (*start).max(0) as usize;
                        let substr: String = if args.len() == 3 {
                            let len_val = self.evaluate_expression(&args[2], row)?;
                            match len_val {
                                Value::Int64(len) => {
                                    let take = (len).max(0) as usize;
                                    s.chars().skip(start_idx).take(take).collect()
                                }
                                _ => return Ok(Value::Null),
                            }
                        } else {
                            s.chars().skip(start_idx).collect()
                        };
                        Ok(Value::String(substr))
                    }
                    _ => Ok(Value::Null),
                }
            }
            "left" => {
                if args.len() != 2 {
                    return Err("left() requires 2 arguments: string, length".into());
                }
                let str_val = coerce_to_string(self.evaluate_expression(&args[0], row)?);
                let len_val = self.evaluate_expression(&args[1], row)?;
                match (&str_val, &len_val) {
                    (Value::String(s), Value::Int64(len)) => {
                        let result: String = s.chars().take(*len as usize).collect();
                        Ok(Value::String(result))
                    }
                    _ => Ok(Value::Null),
                }
            }
            "right" => {
                if args.len() != 2 {
                    return Err("right() requires 2 arguments: string, length".into());
                }
                let str_val = coerce_to_string(self.evaluate_expression(&args[0], row)?);
                let len_val = self.evaluate_expression(&args[1], row)?;
                match (&str_val, &len_val) {
                    (Value::String(s), Value::Int64(len)) => {
                        let char_count = s.chars().count();
                        let skip = char_count.saturating_sub(*len as usize);
                        let result: String = s.chars().skip(skip).collect();
                        Ok(Value::String(result))
                    }
                    _ => Ok(Value::Null),
                }
            }
            "trim" | "btrim" => {
                if args.len() != 1 {
                    return Err("trim() requires 1 argument: string".into());
                }
                let val = coerce_to_string(self.evaluate_expression(&args[0], row)?);
                match val {
                    Value::String(s) => Ok(Value::String(s.trim().to_string())),
                    _ => Ok(Value::Null),
                }
            }
            "ltrim" => {
                if args.len() != 1 {
                    return Err("ltrim() requires 1 argument: string".into());
                }
                let val = coerce_to_string(self.evaluate_expression(&args[0], row)?);
                match val {
                    Value::String(s) => Ok(Value::String(s.trim_start().to_string())),
                    _ => Ok(Value::Null),
                }
            }
            "rtrim" => {
                if args.len() != 1 {
                    return Err("rtrim() requires 1 argument: string".into());
                }
                let val = coerce_to_string(self.evaluate_expression(&args[0], row)?);
                match val {
                    Value::String(s) => Ok(Value::String(s.trim_end().to_string())),
                    _ => Ok(Value::Null),
                }
            }
            "reverse" => {
                if args.len() != 1 {
                    return Err("reverse() requires 1 argument: string".into());
                }
                let val = coerce_to_string(self.evaluate_expression(&args[0], row)?);
                match val {
                    Value::String(s) => Ok(Value::String(s.chars().rev().collect())),
                    _ => Ok(Value::Null),
                }
            }
            // ── List functions ────────────────────────────────────
            "head" => {
                if args.len() != 1 {
                    return Err("head() requires 1 argument".into());
                }
                let val = self.evaluate_expression(&args[0], row)?;
                let items = parse_list_value(&val);
                Ok(items.into_iter().next().unwrap_or(Value::Null))
            }
            "last" => {
                if args.len() != 1 {
                    return Err("last() requires 1 argument".into());
                }
                let val = self.evaluate_expression(&args[0], row)?;
                let items = parse_list_value(&val);
                Ok(items.into_iter().last().unwrap_or(Value::Null))
            }
            // ── Spatial functions ─────────────────────────────────
            "point" => {
                if args.len() != 2 {
                    return Err("point() requires 2 arguments: lat, lon".into());
                }
                let lat = crate::graph::core::value_operations::value_to_f64(
                    &self.evaluate_expression(&args[0], row)?,
                )
                .ok_or("point(): lat must be numeric")?;
                let lon = crate::graph::core::value_operations::value_to_f64(
                    &self.evaluate_expression(&args[1], row)?,
                )
                .ok_or("point(): lon must be numeric")?;
                Ok(Value::Point { lat, lon })
            }
            "distance" => match args.len() {
                2 => {
                    // Resolve via spatial config — prefer_geometry=false so bare
                    // variables resolve as Points; explicit .geometry resolves as Geometry
                    let r1 = self.resolve_spatial(&args[0], row, false)?;
                    let r2 = self.resolve_spatial(&args[1], row, false)?;
                    match (r1, r2) {
                        (
                            Some(ResolvedSpatial::Point(lat1, lon1)),
                            Some(ResolvedSpatial::Point(lat2, lon2)),
                        ) => Ok(Value::Float64(
                            crate::graph::features::spatial::geodesic_distance(
                                lat1, lon1, lat2, lon2,
                            ),
                        )),
                        (
                            Some(ResolvedSpatial::Point(lat, lon)),
                            Some(ResolvedSpatial::Geometry(g, _)),
                        )
                        | (
                            Some(ResolvedSpatial::Geometry(g, _)),
                            Some(ResolvedSpatial::Point(lat, lon)),
                        ) => Ok(Value::Float64(
                            crate::graph::features::spatial::point_to_geometry_distance_m(
                                lat, lon, &g,
                            )?,
                        )),
                        (
                            Some(ResolvedSpatial::Geometry(g1, _)),
                            Some(ResolvedSpatial::Geometry(g2, _)),
                        ) => Ok(Value::Float64(
                            crate::graph::features::spatial::geometry_to_geometry_distance_m(
                                &g1, &g2,
                            )?,
                        )),
                        // One or both sides have no spatial data (e.g. node
                        // exists but geometry field is NULL) → propagate Null
                        // so WHERE distance(a, b) < X simply filters them out.
                        _ => Ok(Value::Null),
                    }
                }
                4 => {
                    let lat1 = crate::graph::core::value_operations::value_to_f64(
                        &self.evaluate_expression(&args[0], row)?,
                    )
                    .ok_or("distance(): args must be numeric")?;
                    let lon1 = crate::graph::core::value_operations::value_to_f64(
                        &self.evaluate_expression(&args[1], row)?,
                    )
                    .ok_or("distance(): args must be numeric")?;
                    let lat2 = crate::graph::core::value_operations::value_to_f64(
                        &self.evaluate_expression(&args[2], row)?,
                    )
                    .ok_or("distance(): args must be numeric")?;
                    let lon2 = crate::graph::core::value_operations::value_to_f64(
                        &self.evaluate_expression(&args[3], row)?,
                    )
                    .ok_or("distance(): args must be numeric")?;
                    Ok(Value::Float64(
                        crate::graph::features::spatial::geodesic_distance(lat1, lon1, lat2, lon2),
                    ))
                }
                _ => Err(
                    "distance() requires 2 (Point, Point) or 4 (lat1, lon1, lat2, lon2) arguments"
                        .into(),
                ),
            },
            // ── Node-aware spatial functions ──────────────────────────
            "contains" => {
                if args.len() != 2 {
                    return Err("contains() requires 2 arguments".into());
                }
                // Arg 1: must be a geometry (the container).
                // When the arg is a node-bound variable but that specific
                // node has no geometry (e.g. partial coverage in a typed
                // set — real-world: 312/469 AfexAreas have no
                // wkt_geometry), treat the predicate as false for this
                // row instead of erroring out the whole query. Matches
                // Cypher's NULL-propagation semantics: missing data ≠ true.
                let resolved1 = match self.resolve_spatial(&args[0], row, true)? {
                    Some(r) => r,
                    None => return Ok(Value::Boolean(false)),
                };
                let (geom, bbox1) = match &resolved1 {
                    ResolvedSpatial::Geometry(g, bbox) => (g, bbox),
                    ResolvedSpatial::Point(_, _) => {
                        return Err("contains(): first arg must be a geometry, not a point".into());
                    }
                };
                // Arg 2: prefer point for the contained item (point-in-polygon).
                // Same NULL-propagation: missing target → predicate false.
                let resolved2 = match self.resolve_spatial(&args[1], row, false)? {
                    Some(r) => r,
                    None => return Ok(Value::Boolean(false)),
                };

                match &resolved2 {
                    ResolvedSpatial::Point(lat, lon) => {
                        // Bbox pre-filter: if the point is outside the container's bbox,
                        // it cannot be inside the polygon. This is O(1) vs O(n_vertices).
                        if let Some(bb) = bbox1 {
                            let pt = geo::Coord { x: *lon, y: *lat };
                            if !bb.min().x.le(&pt.x)
                                || !bb.max().x.ge(&pt.x)
                                || !bb.min().y.le(&pt.y)
                                || !bb.max().y.ge(&pt.y)
                            {
                                return Ok(Value::Boolean(false));
                            }
                        }
                        let pt = geo::Point::new(*lon, *lat);
                        Ok(Value::Boolean(
                            crate::graph::features::spatial::geometry_contains_point(geom, &pt),
                        ))
                    }
                    ResolvedSpatial::Geometry(g2, bbox2) => {
                        // Bbox pre-filter: if bboxes don't overlap, containment is impossible
                        if let (Some(bb1), Some(bb2)) = (bbox1, bbox2) {
                            if bb1.max().x < bb2.min().x
                                || bb2.max().x < bb1.min().x
                                || bb1.max().y < bb2.min().y
                                || bb2.max().y < bb1.min().y
                            {
                                return Ok(Value::Boolean(false));
                            }
                        }
                        Ok(Value::Boolean(
                            crate::graph::features::spatial::geometry_contains_geometry(geom, g2),
                        ))
                    }
                }
            }
            "intersects" => {
                if args.len() != 2 {
                    return Err("intersects() requires 2 arguments".into());
                }
                let r1 = self
                    .resolve_spatial(&args[0], row, true)?
                    .ok_or(SPATIAL_RESOLUTION_HELP)?;
                let r2 = self
                    .resolve_spatial(&args[1], row, true)?
                    .ok_or(SPATIAL_RESOLUTION_HELP)?;
                // Dispatch without cloning — use Arc references where possible
                let result = match (&r1, &r2) {
                    (
                        ResolvedSpatial::Geometry(g1, bbox1),
                        ResolvedSpatial::Geometry(g2, bbox2),
                    ) => {
                        // Bbox pre-filter: if bboxes don't overlap, no intersection possible
                        if let (Some(bb1), Some(bb2)) = (bbox1, bbox2) {
                            if bb1.max().x < bb2.min().x
                                || bb2.max().x < bb1.min().x
                                || bb1.max().y < bb2.min().y
                                || bb2.max().y < bb1.min().y
                            {
                                return Ok(Value::Boolean(false));
                            }
                        }
                        crate::graph::features::spatial::geometries_intersect(g1, g2)
                    }
                    (ResolvedSpatial::Point(lat, lon), ResolvedSpatial::Geometry(g, bbox)) => {
                        // Bbox pre-filter for point-vs-geometry
                        if let Some(bb) = bbox {
                            if *lon < bb.min().x
                                || *lon > bb.max().x
                                || *lat < bb.min().y
                                || *lat > bb.max().y
                            {
                                return Ok(Value::Boolean(false));
                            }
                        }
                        let pt = geo::Geometry::Point(geo::Point::new(*lon, *lat));
                        crate::graph::features::spatial::geometries_intersect(&pt, g)
                    }
                    (ResolvedSpatial::Geometry(g, bbox), ResolvedSpatial::Point(lat, lon)) => {
                        if let Some(bb) = bbox {
                            if *lon < bb.min().x
                                || *lon > bb.max().x
                                || *lat < bb.min().y
                                || *lat > bb.max().y
                            {
                                return Ok(Value::Boolean(false));
                            }
                        }
                        let pt = geo::Geometry::Point(geo::Point::new(*lon, *lat));
                        crate::graph::features::spatial::geometries_intersect(g, &pt)
                    }
                    (ResolvedSpatial::Point(lat1, lon1), ResolvedSpatial::Point(lat2, lon2)) => {
                        lat1 == lat2 && lon1 == lon2
                    }
                };
                Ok(Value::Boolean(result))
            }
            "centroid" => {
                if args.len() != 1 {
                    return Err("centroid() requires 1 argument".into());
                }
                // NULL-propagate: scalar functions on missing geometry
                // return Value::Null so downstream WHERE/IS NOT NULL can
                // filter cleanly without erroring the whole query.
                let resolved = match self.resolve_spatial(&args[0], row, true)? {
                    Some(r) => r,
                    None => return Ok(Value::Null),
                };
                match &resolved {
                    ResolvedSpatial::Point(lat, lon) => Ok(Value::Point {
                        lat: *lat,
                        lon: *lon,
                    }),
                    ResolvedSpatial::Geometry(g, _) => {
                        let (lat, lon) = crate::graph::features::spatial::geometry_centroid(g)?;
                        Ok(Value::Point { lat, lon })
                    }
                }
            }
            "area" => {
                if args.len() != 1 {
                    return Err("area() requires 1 argument".into());
                }
                let resolved = match self.resolve_spatial(&args[0], row, true)? {
                    Some(r) => r,
                    None => return Ok(Value::Null),
                };
                match &resolved {
                    ResolvedSpatial::Geometry(g, _) => Ok(Value::Float64(
                        crate::graph::features::spatial::geometry_area_m2(g)?,
                    )),
                    ResolvedSpatial::Point(_, _) => {
                        Err("area(): arg must be a polygon geometry, not a point".into())
                    }
                }
            }
            "perimeter" => {
                if args.len() != 1 {
                    return Err("perimeter() requires 1 argument".into());
                }
                let resolved = match self.resolve_spatial(&args[0], row, true)? {
                    Some(r) => r,
                    None => return Ok(Value::Null),
                };
                match &resolved {
                    ResolvedSpatial::Geometry(g, _) => Ok(Value::Float64(
                        crate::graph::features::spatial::geometry_perimeter_m(g)?,
                    )),
                    ResolvedSpatial::Point(_, _) => {
                        Err("perimeter(): arg must be a geometry, not a point".into())
                    }
                }
            }
            "latitude" => {
                if args.len() != 1 {
                    return Err("latitude() requires 1 argument".into());
                }
                match self.evaluate_expression(&args[0], row)? {
                    Value::Point { lat, .. } => Ok(Value::Float64(lat)),
                    _ => Err("latitude() requires a Point argument".into()),
                }
            }
            "longitude" => {
                if args.len() != 1 {
                    return Err("longitude() requires 1 argument".into());
                }
                match self.evaluate_expression(&args[0], row)? {
                    Value::Point { lon, .. } => Ok(Value::Float64(lon)),
                    _ => Err("longitude() requires a Point argument".into()),
                }
            }
            // ── Geometry primitives (0.8.20) ──────────────────────────
            "geom_buffer" => {
                if args.len() != 2 {
                    return Err("geom_buffer() requires 2 arguments: (geom, meters)".into());
                }
                let geom = match self.geom_arg(&args[0], row)? {
                    Some(g) => g,
                    None => return Ok(Value::Null),
                };
                let meters = crate::graph::core::value_operations::value_to_f64(
                    &self.evaluate_expression(&args[1], row)?,
                )
                .ok_or("geom_buffer(): second argument must be numeric (meters)")?;
                let result = crate::graph::features::spatial::geometry_buffer(&geom, meters)?;
                Ok(Value::String(
                    crate::graph::features::spatial::geometry_to_wkt(&result),
                ))
            }
            "geom_convex_hull" => {
                if args.is_empty() {
                    return Err("geom_convex_hull() requires at least 1 argument".into());
                }
                let mut geoms: Vec<geo::Geometry<f64>> = Vec::new();
                // Single list argument: parse list of WKT strings.
                // Phase A.1 / C4 — native Value::List path.
                if args.len() == 1 {
                    let val = self.evaluate_expression(&args[0], row)?;
                    if let Value::List(items) = &val {
                        for item in items {
                            if let Value::String(wkt) = item {
                                if let Ok(g) = crate::graph::features::spatial::parse_wkt(wkt) {
                                    geoms.push(g);
                                }
                            }
                        }
                    } else if let Value::String(ref s) = val {
                        if s.starts_with('[') && s.ends_with(']') {
                            for item in parse_list_value(&val) {
                                if let Value::String(wkt) = item {
                                    if let Ok(g) = crate::graph::features::spatial::parse_wkt(&wkt)
                                    {
                                        geoms.push(g);
                                    }
                                }
                            }
                        }
                    }
                }
                if geoms.is_empty() {
                    for arg in args {
                        if let Some(g) = self.geom_arg(arg, row)? {
                            geoms.push(g);
                        }
                    }
                }
                if geoms.is_empty() {
                    return Ok(Value::Null);
                }
                let hull = crate::graph::features::spatial::geometries_convex_hull(&geoms)?;
                Ok(Value::String(
                    crate::graph::features::spatial::geometry_to_wkt(&hull),
                ))
            }
            "geom_union" | "geom_intersection" | "geom_difference" => {
                if args.len() != 2 {
                    return Err(format!("{name}() requires 2 arguments: (g1, g2)"));
                }
                let g1 = match self.geom_arg(&args[0], row)? {
                    Some(g) => g,
                    None => return Ok(Value::Null),
                };
                let g2 = match self.geom_arg(&args[1], row)? {
                    Some(g) => g,
                    None => return Ok(Value::Null),
                };
                let result = match name {
                    "geom_union" => crate::graph::features::spatial::geometry_union(&g1, &g2)?,
                    "geom_intersection" => {
                        crate::graph::features::spatial::geometry_intersection(&g1, &g2)?
                    }
                    "geom_difference" => {
                        crate::graph::features::spatial::geometry_difference(&g1, &g2)?
                    }
                    _ => unreachable!(),
                };
                Ok(Value::String(
                    crate::graph::features::spatial::geometry_to_wkt(&result),
                ))
            }
            "geom_is_valid" => {
                if args.len() != 1 {
                    return Err("geom_is_valid() requires 1 argument".into());
                }
                let geom = match self.geom_arg(&args[0], row)? {
                    Some(g) => g,
                    None => return Ok(Value::Null),
                };
                Ok(Value::Boolean(
                    crate::graph::features::spatial::geometry_is_valid(&geom),
                ))
            }
            "geom_length" => {
                if args.len() != 1 {
                    return Err("geom_length() requires 1 argument".into());
                }
                let geom = match self.geom_arg(&args[0], row)? {
                    Some(g) => g,
                    None => return Ok(Value::Null),
                };
                Ok(Value::Float64(
                    crate::graph::features::spatial::geometry_length_m(&geom),
                ))
            }
            // vector_score(node, embedding_property, query_vector [, metric])
            // Returns the similarity score (f32→f64) for the node's embedding vs query vector.
            //
            // Performance: The constant arguments (property name, query vector, metric) are
            // parsed once on the first call and cached in self.vs_cache. Subsequent rows
            // skip JSON parsing, String allocation, and metric dispatch entirely.
            "vector_score" => {
                if args.len() < 3 || args.len() > 4 {
                    return Err(
                        "vector_score() requires 3-4 arguments: (node, property, query_vector [, metric])"
                            .into(),
                    );
                }

                // Arg 0: node variable → resolve to NodeIndex (changes per row)
                let node_idx = match &args[0] {
                    Expression::Variable(var) => match row.node_bindings.get(var) {
                        Some(&idx) => idx,
                        None => return Ok(Value::Null),
                    },
                    _ => {
                        return Err("vector_score(): first argument must be a node variable".into())
                    }
                };

                // Get or initialize cache — constant args parsed once, reused for all rows
                let c = match self.vs_cache.get() {
                    Some(c) => c,
                    None => {
                        let prop_name = match self.evaluate_expression(&args[1], row)? {
                            Value::String(s) => s,
                            _ => return Err(
                                "vector_score(): second argument must be a string property name"
                                    .into(),
                            ),
                        };
                        let query_vec = self.extract_float_list(&args[2], row)?;
                        // Resolve metric: explicit arg > stored metric > cosine default
                        let metric_name = if args.len() > 3 {
                            match self.evaluate_expression(&args[3], row)? {
                                Value::String(s) => s,
                                _ => "cosine".to_string(),
                            }
                        } else {
                            // Look up stored metric from the embedding store
                            self.graph
                                .embeddings
                                .iter()
                                .find(|((_, pn), _)| pn == &prop_name)
                                .and_then(|(_, store)| store.metric.clone())
                                .unwrap_or_else(|| "cosine".to_string())
                        };
                        let similarity_fn = match metric_name.as_str() {
                            "cosine" => vs::cosine_similarity as fn(&[f32], &[f32]) -> f32,
                            "dot_product" => vs::dot_product,
                            "euclidean" => vs::neg_euclidean_distance,
                            "poincare" => vs::neg_poincare_distance,
                            other => {
                                return Err(format!(
                                    "vector_score(): unknown metric '{}'. Use 'cosine', 'dot_product', 'euclidean', or 'poincare'.",
                                    other
                                ))
                            }
                        };
                        let _ = self.vs_cache.set(VectorScoreCache {
                            prop_name,
                            query_vec,
                            similarity_fn,
                        });
                        self.vs_cache.get().unwrap()
                    }
                };

                // Per-row: look up node type → embedding store → compute similarity
                let node_type = match self.graph.graph.node_weight(node_idx) {
                    Some(n) => n.node_type_str(&self.graph.interner),
                    None => return Ok(Value::Null),
                };

                let store = match self.graph.embedding_store(node_type, &c.prop_name) {
                    Some(s) => s,
                    None => {
                        return Err(format!(
                            "vector_score(): no embedding '{}' found for node type '{}'",
                            c.prop_name, node_type
                        ))
                    }
                };

                if c.query_vec.len() != store.dimension {
                    return Err(format!(
                        "vector_score(): query vector dimension {} does not match embedding dimension {}",
                        c.query_vec.len(),
                        store.dimension
                    ));
                }

                match store.get_embedding(node_idx.index()) {
                    Some(embedding) => {
                        let score = (c.similarity_fn)(&c.query_vec, embedding);
                        Ok(Value::Float64(score as f64))
                    }
                    None => Ok(Value::Null),
                }
            }
            // ── Timeseries functions ──────────────────────────────────────
            "ts_at" => {
                if args.len() != 2 {
                    return Err("ts_at() requires 2 arguments: (n.channel, '2020-2')".into());
                }
                let (ts, channel, _config) = self.resolve_timeseries_channel(&args[0], row)?;
                let date_arg = self.resolve_ts_date_arg(&args[1], row)?;
                match date_arg {
                    Some((date, _prec)) => {
                        match crate::graph::features::timeseries::find_key_index(&ts.keys, date) {
                            Some(idx) => {
                                let v = channel[idx];
                                if v.is_finite() {
                                    Ok(Value::Float64(v))
                                } else {
                                    Ok(Value::Null)
                                }
                            }
                            None => Ok(Value::Null),
                        }
                    }
                    None => Ok(Value::Null), // null date → null
                }
            }
            "ts_sum" | "ts_avg" | "ts_min" | "ts_max" | "ts_count" => {
                if args.is_empty() || args.len() > 3 {
                    return Err(format!(
                        "{}() requires 1-3 arguments: (n.channel [, 'start'] [, 'end'])",
                        name
                    ));
                }
                let (ts, channel, _config) = self.resolve_timeseries_channel(&args[0], row)?;
                let (lo, hi) = self.resolve_ts_range(ts, &args[1..], row)?;
                let slice = &channel[lo..hi];
                match name {
                    "ts_sum" => Ok(Value::Float64(crate::graph::features::timeseries::ts_sum(
                        slice,
                    ))),
                    "ts_avg" => {
                        let v = crate::graph::features::timeseries::ts_avg(slice);
                        if v.is_nan() {
                            Ok(Value::Null)
                        } else {
                            Ok(Value::Float64(v))
                        }
                    }
                    "ts_min" => {
                        let v = crate::graph::features::timeseries::ts_min(slice);
                        if v.is_infinite() {
                            Ok(Value::Null)
                        } else {
                            Ok(Value::Float64(v))
                        }
                    }
                    "ts_max" => {
                        let v = crate::graph::features::timeseries::ts_max(slice);
                        if v.is_infinite() {
                            Ok(Value::Null)
                        } else {
                            Ok(Value::Float64(v))
                        }
                    }
                    "ts_count" => Ok(Value::Int64(crate::graph::features::timeseries::ts_count(
                        slice,
                    ) as i64)),
                    _ => unreachable!(),
                }
            }
            "ts_first" => {
                if args.len() != 1 {
                    return Err("ts_first() requires 1 argument: (n.channel)".into());
                }
                let (_, channel, _) = self.resolve_timeseries_channel(&args[0], row)?;
                match channel.iter().find(|v| v.is_finite()) {
                    Some(&v) => Ok(Value::Float64(v)),
                    None => Ok(Value::Null),
                }
            }
            "ts_last" => {
                if args.len() != 1 {
                    return Err("ts_last() requires 1 argument: (n.channel)".into());
                }
                let (_, channel, _) = self.resolve_timeseries_channel(&args[0], row)?;
                match channel.iter().rev().find(|v| v.is_finite()) {
                    Some(&v) => Ok(Value::Float64(v)),
                    None => Ok(Value::Null),
                }
            }
            "ts_delta" => {
                if args.len() != 3 {
                    return Err(
                        "ts_delta() requires 3 arguments: (n.channel, '2019-12', '2021-1')".into(),
                    );
                }
                let (ts, channel, _config) = self.resolve_timeseries_channel(&args[0], row)?;
                let a1 = self.resolve_ts_date_arg(&args[1], row)?;
                let a2 = self.resolve_ts_date_arg(&args[2], row)?;
                let v1 = a1.and_then(|(date, prec)| {
                    let end = crate::graph::features::timeseries::expand_end(date, prec);
                    let (lo, hi) = crate::graph::features::timeseries::find_range(
                        &ts.keys,
                        Some(date),
                        Some(end),
                    );
                    if lo < hi { Some(channel[lo]) } else { None }.filter(|v| v.is_finite())
                });
                let v2 = a2.and_then(|(date, prec)| {
                    let end = crate::graph::features::timeseries::expand_end(date, prec);
                    let (lo, hi) = crate::graph::features::timeseries::find_range(
                        &ts.keys,
                        Some(date),
                        Some(end),
                    );
                    if lo < hi { Some(channel[lo]) } else { None }.filter(|v| v.is_finite())
                });
                match (v1, v2) {
                    (Some(a), Some(b)) => Ok(Value::Float64(b - a)),
                    _ => Ok(Value::Null),
                }
            }
            "ts_series" => {
                // Phase A.1 / C4 — native Value::List of Value::Map.
                // Each entry: {"time": <date-str>, "value": <float|null>}.
                if args.is_empty() || args.len() > 3 {
                    return Err(
                        "ts_series() requires 1-3 arguments: (n.channel [, 'start'] [, 'end'])"
                            .into(),
                    );
                }
                let (ts, channel, _config) = self.resolve_timeseries_channel(&args[0], row)?;
                let (lo, hi) = self.resolve_ts_range(ts, &args[1..], row)?;
                let mut entries: Vec<Value> = Vec::with_capacity(hi - lo);
                for (date, &val) in ts.keys[lo..hi].iter().zip(&channel[lo..hi]) {
                    let mut entry: std::collections::BTreeMap<String, Value> =
                        std::collections::BTreeMap::new();
                    entry.insert("time".to_string(), Value::String(date.to_string()));
                    entry.insert(
                        "value".to_string(),
                        if val.is_finite() {
                            Value::Float64(val)
                        } else {
                            Value::Null
                        },
                    );
                    entries.push(Value::Map(entry));
                }
                Ok(Value::List(entries))
            }
            // ── List functions ────────────────────────────────────
            "range" => {
                if args.len() < 2 || args.len() > 3 {
                    return Err(
                        "range() requires 2 or 3 arguments: range(start, end[, step])".into(),
                    );
                }
                let start = as_i64(&self.evaluate_expression(&args[0], row)?)?;
                let end = as_i64(&self.evaluate_expression(&args[1], row)?)?;
                let step = if args.len() == 3 {
                    let s = as_i64(&self.evaluate_expression(&args[2], row)?)?;
                    if s == 0 {
                        return Err("range() step must not be zero".into());
                    }
                    s
                } else {
                    1
                };
                // Phase A.1 / C4 — native Value::List of Value::Int64.
                let mut vals: Vec<Value> = Vec::new();
                let mut cur = start;
                if step > 0 {
                    while cur <= end {
                        vals.push(Value::Int64(cur));
                        cur += step;
                    }
                } else {
                    while cur >= end {
                        vals.push(Value::Int64(cur));
                        cur += step;
                    }
                }
                Ok(Value::List(vals))
            }

            // ── Numeric math functions ──────────────────────────
            "abs" => {
                let val = self.evaluate_expression(&args[0], row)?;
                match val {
                    Value::Int64(n) => Ok(Value::Int64(n.abs())),
                    Value::Float64(f) => Ok(Value::Float64(f.abs())),
                    Value::Null => Ok(Value::Null),
                    _ => match value_to_f64(&val) {
                        Some(f) => Ok(Value::Float64(f.abs())),
                        None => Ok(Value::Null),
                    },
                }
            }
            "ceil" | "ceiling" => {
                let val = self.evaluate_expression(&args[0], row)?;
                match val {
                    Value::Null => Ok(Value::Null),
                    _ => match value_to_f64(&val) {
                        Some(f) => Ok(Value::Float64(f.ceil())),
                        None => Ok(Value::Null),
                    },
                }
            }
            "floor" => {
                let val = self.evaluate_expression(&args[0], row)?;
                match val {
                    Value::Null => Ok(Value::Null),
                    _ => match value_to_f64(&val) {
                        Some(f) => Ok(Value::Float64(f.floor())),
                        None => Ok(Value::Null),
                    },
                }
            }
            "round" => {
                let val = self.evaluate_expression(&args[0], row)?;
                match val {
                    Value::Null => Ok(Value::Null),
                    _ => match value_to_f64(&val) {
                        Some(f) => {
                            if args.len() >= 2 {
                                let prec = self.evaluate_expression(&args[1], row)?;
                                let d = match &prec {
                                    Value::Int64(i) => *i as i32,
                                    Value::Float64(fl) => *fl as i32,
                                    _ => 0,
                                };
                                let factor = 10f64.powi(d);
                                Ok(Value::Float64((f * factor).round() / factor))
                            } else {
                                Ok(Value::Float64(f.round()))
                            }
                        }
                        None => Ok(Value::Null),
                    },
                }
            }
            "sqrt" => {
                let val = self.evaluate_expression(&args[0], row)?;
                match val {
                    Value::Null => Ok(Value::Null),
                    _ => match value_to_f64(&val) {
                        Some(f) if f >= 0.0 => Ok(Value::Float64(f.sqrt())),
                        _ => Ok(Value::Null),
                    },
                }
            }
            "sign" => {
                let val = self.evaluate_expression(&args[0], row)?;
                match val {
                    Value::Null => Ok(Value::Null),
                    _ => match value_to_f64(&val) {
                        Some(f) if f > 0.0 => Ok(Value::Int64(1)),
                        Some(f) if f < 0.0 => Ok(Value::Int64(-1)),
                        Some(_) => Ok(Value::Int64(0)),
                        None => Ok(Value::Null),
                    },
                }
            }
            "log" | "ln" => {
                let val = self.evaluate_expression(&args[0], row)?;
                match val {
                    Value::Null => Ok(Value::Null),
                    _ => match value_to_f64(&val) {
                        Some(f) if f > 0.0 => Ok(Value::Float64(f.ln())),
                        _ => Ok(Value::Null),
                    },
                }
            }
            "log10" => {
                let val = self.evaluate_expression(&args[0], row)?;
                match val {
                    Value::Null => Ok(Value::Null),
                    _ => match value_to_f64(&val) {
                        Some(f) if f > 0.0 => Ok(Value::Float64(f.log10())),
                        _ => Ok(Value::Null),
                    },
                }
            }
            "exp" => {
                let val = self.evaluate_expression(&args[0], row)?;
                match val {
                    Value::Null => Ok(Value::Null),
                    _ => match value_to_f64(&val) {
                        Some(f) => Ok(Value::Float64(f.exp())),
                        None => Ok(Value::Null),
                    },
                }
            }
            "pow" | "power" => {
                if args.len() != 2 {
                    return Err("pow() requires 2 arguments: base, exponent".into());
                }
                let base_val = self.evaluate_expression(&args[0], row)?;
                let exp_val = self.evaluate_expression(&args[1], row)?;
                match (value_to_f64(&base_val), value_to_f64(&exp_val)) {
                    (Some(base), Some(exp)) => Ok(Value::Float64(base.powf(exp))),
                    _ => Ok(Value::Null),
                }
            }
            "pi" => Ok(Value::Float64(std::f64::consts::PI)),
            "rand" | "random" => {
                // Thread-local xorshift64 PRNG. Seeded once per thread from
                // SystemTime mixed with a monotonic per-thread counter;
                // subsequent calls just advance the state. Avoids per-call
                // SystemTime::now() overhead and guarantees distinct values
                // within a tight per-row loop. The counter splat ensures
                // parallel rayon workers don't collide on the same nanosecond.
                use std::cell::Cell;
                use std::sync::atomic::{AtomicU64, Ordering};
                use std::time::SystemTime;
                static THREAD_COUNTER: AtomicU64 = AtomicU64::new(0);
                thread_local! {
                    static XORSHIFT_STATE: Cell<u64> = {
                        let nanos = SystemTime::now()
                            .duration_since(SystemTime::UNIX_EPOCH)
                            .unwrap_or_default()
                            .as_nanos() as u64;
                        let counter = THREAD_COUNTER.fetch_add(1, Ordering::Relaxed);
                        // Mix counter via splitmix64-ish avalanche so adjacent
                        // thread IDs produce well-separated seeds.
                        let mut seed = nanos.wrapping_add(counter.wrapping_mul(0x9E37_79B9_7F4A_7C15));
                        seed ^= seed >> 30;
                        seed = seed.wrapping_mul(0xBF58_476D_1CE4_E5B9);
                        seed ^= seed >> 27;
                        seed = seed.wrapping_mul(0x94D0_49BB_1331_11EB);
                        seed ^= seed >> 31;
                        Cell::new(seed | 1)
                    };
                }
                let val = XORSHIFT_STATE.with(|state| {
                    let mut x = state.get();
                    x ^= x << 13;
                    x ^= x >> 7;
                    x ^= x << 17;
                    state.set(x);
                    // Use top 53 bits → f64 mantissa to avoid precision loss.
                    ((x >> 11) as f64) / ((1u64 << 53) as f64)
                });
                Ok(Value::Float64(val))
            }

            // ── Temporal filtering functions ──────────────────────────────
            "valid_at" => {
                // valid_at(entity, date, 'from_field', 'to_field') → Boolean
                // True when entity.from_field <= date AND entity.to_field >= date.
                // NULL fields = open-ended (always pass).
                if args.len() != 4 {
                    return Err(
                        "valid_at() requires 4 arguments: (entity, date, from_field, to_field)"
                            .into(),
                    );
                }
                let var_name =
                    match &args[0] {
                        Expression::Variable(v) => v,
                        _ => return Err(
                            "valid_at(): first argument must be a node or relationship variable"
                                .into(),
                        ),
                    };
                let date_val = self.evaluate_expression(&args[1], row)?;
                let from_field = match self.evaluate_expression(&args[2], row)? {
                    Value::String(s) => s,
                    _ => return Err("valid_at(): from_field (3rd arg) must be a string".into()),
                };
                let to_field = match self.evaluate_expression(&args[3], row)? {
                    Value::String(s) => s,
                    _ => return Err("valid_at(): to_field (4th arg) must be a string".into()),
                };
                let from_val = self.resolve_property(var_name, &from_field, row)?;
                let to_val = self.resolve_property(var_name, &to_field, row)?;
                // NULL = open-ended boundary
                let from_ok = match &from_val {
                    Value::Null => true,
                    _ => {
                        evaluate_comparison(&from_val, &ComparisonOp::LessThanEq, &date_val, None)?
                    }
                };
                let to_ok = match &to_val {
                    Value::Null => true,
                    _ => {
                        evaluate_comparison(&to_val, &ComparisonOp::GreaterThanEq, &date_val, None)?
                    }
                };
                Ok(Value::Boolean(from_ok && to_ok))
            }
            "valid_during" => {
                // valid_during(entity, start, end, 'from_field', 'to_field') → Boolean
                // Overlap: entity.from_field <= end AND entity.to_field >= start.
                // NULL fields = open-ended (always pass).
                if args.len() != 5 {
                    return Err(
                        "valid_during() requires 5 arguments: (entity, start, end, from_field, to_field)"
                            .into(),
                    );
                }
                let var_name = match &args[0] {
                    Expression::Variable(v) => v,
                    _ => return Err(
                        "valid_during(): first argument must be a node or relationship variable"
                            .into(),
                    ),
                };
                let start_val = self.evaluate_expression(&args[1], row)?;
                let end_val = self.evaluate_expression(&args[2], row)?;
                let from_field = match self.evaluate_expression(&args[3], row)? {
                    Value::String(s) => s,
                    _ => return Err("valid_during(): from_field (4th arg) must be a string".into()),
                };
                let to_field = match self.evaluate_expression(&args[4], row)? {
                    Value::String(s) => s,
                    _ => return Err("valid_during(): to_field (5th arg) must be a string".into()),
                };
                let from_val = self.resolve_property(var_name, &from_field, row)?;
                let to_val = self.resolve_property(var_name, &to_field, row)?;
                // Overlap: entity.from <= query_end AND entity.to >= query_start
                let from_ok = match &from_val {
                    Value::Null => true,
                    _ => evaluate_comparison(&from_val, &ComparisonOp::LessThanEq, &end_val, None)?,
                };
                let to_ok = match &to_val {
                    Value::Null => true,
                    _ => evaluate_comparison(
                        &to_val,
                        &ComparisonOp::GreaterThanEq,
                        &start_val,
                        None,
                    )?,
                };
                Ok(Value::Boolean(from_ok && to_ok))
            }

            // Aggregate functions should not be evaluated per-row
            "count" | "sum" | "avg" | "min" | "max" | "collect" | "mean" | "std" | "stdev" => {
                Err(format!(
                    "Aggregate function '{}' cannot be used outside of RETURN/WITH",
                    name
                ))
            }
            // embedding_norm(node, property) → Float64
            // Returns the L2 norm of the node's embedding vector.
            // Useful for inferring hierarchy depth in Poincaré embeddings
            // (norm close to 0 = root/general, norm close to 1 = leaf/specific).
            "embedding_norm" => {
                if args.len() != 2 {
                    return Err("embedding_norm() requires 2 arguments: (node, property)".into());
                }
                let node_idx = match &args[0] {
                    Expression::Variable(var) => match row.node_bindings.get(var) {
                        Some(&idx) => idx,
                        None => return Ok(Value::Null),
                    },
                    _ => {
                        return Err(
                            "embedding_norm(): first argument must be a node variable".into()
                        )
                    }
                };
                let prop_name = match self.evaluate_expression(&args[1], row)? {
                    Value::String(s) => s,
                    _ => {
                        return Err(
                            "embedding_norm(): second argument must be a string property name"
                                .into(),
                        )
                    }
                };
                let node_type = match self.graph.graph.node_weight(node_idx) {
                    Some(n) => n.node_type_str(&self.graph.interner),
                    None => return Ok(Value::Null),
                };
                let store = match self.graph.embedding_store(node_type, &prop_name) {
                    Some(s) => s,
                    None => {
                        return Err(format!(
                            "embedding_norm(): no embedding '{}' found for node type '{}'",
                            prop_name, node_type
                        ))
                    }
                };
                match store.get_embedding(node_idx.index()) {
                    Some(emb) => {
                        let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
                        Ok(Value::Float64(norm as f64))
                    }
                    None => Ok(Value::Null),
                }
            }
            "text_score" => Err(
                "text_score() requires set_embedder(). Call g.set_embedder(model) first."
                    .to_string(),
            ),
            _ => Err(format!("Unknown function: {}", name)),
        }
    }

    // ── Timeseries helpers ─────────────────────────────────────────────

    /// Resolve the first argument of a ts_*() function into the node's timeseries
    /// data, the specific channel's values, and the timeseries config.
    /// The argument must be a PropertyAccess (e.g. `f.oil`).
    pub(super) fn resolve_timeseries_channel<'b>(
        &'b self,
        expr: &Expression,
        row: &ResultRow,
    ) -> Result<
        (
            &'b crate::graph::features::timeseries::NodeTimeseries,
            &'b [f64],
            &'b crate::graph::features::timeseries::TimeseriesConfig,
        ),
        String,
    > {
        let (variable, property) = match expr {
            Expression::PropertyAccess { variable, property } => (variable, property),
            _ => {
                return Err(
                    "ts_*() first argument must be a property access (e.g. n.channel)".into(),
                )
            }
        };
        let node_idx = row
            .node_bindings
            .get(variable)
            .ok_or_else(|| format!("ts_*(): variable '{}' is not bound to a node", variable))?;
        let ts = self
            .graph
            .get_node_timeseries(node_idx.index())
            .ok_or_else(|| format!("ts_*(): node '{}' has no timeseries data", variable))?;
        let channel = ts.channels.get(property.as_str()).ok_or_else(|| {
            let available: Vec<&str> = ts.channels.keys().map(|s| s.as_str()).collect();
            format!(
                "ts_*(): channel '{}' not found on node '{}'. Available: {:?}",
                property, variable, available
            )
        })?;
        // Look up the config for this node type
        let node = self
            .graph
            .graph
            .node_weight(*node_idx)
            .ok_or("ts_*(): node not found in graph")?;
        let node_type_str = node.node_type_str(&self.graph.interner);
        let config = self
            .graph
            .timeseries_configs
            .get(node_type_str)
            .ok_or_else(|| {
                format!(
                    "ts_*(): no timeseries config for node type '{}'",
                    node_type_str
                )
            })?;
        Ok((ts, channel, config))
    }

    /// Parse a date argument from a ts_*() function call.
    /// Accepts string date queries, integer years, DateTime values, and Null.
    pub(super) fn resolve_ts_date_arg(
        &self,
        expr: &Expression,
        row: &ResultRow,
    ) -> Result<
        Option<(
            chrono::NaiveDate,
            crate::graph::features::timeseries::DatePrecision,
        )>,
        String,
    > {
        let v = self.evaluate_expression(expr, row)?;
        match &v {
            Value::String(s) => crate::graph::features::timeseries::parse_date_query(s).map(Some),
            Value::Int64(year) => {
                let date = chrono::NaiveDate::from_ymd_opt(*year as i32, 1, 1)
                    .ok_or_else(|| format!("ts_*() invalid year: {}", year))?;
                Ok(Some((
                    date,
                    crate::graph::features::timeseries::DatePrecision::Year,
                )))
            }
            Value::DateTime(date) => Ok(Some((
                *date,
                crate::graph::features::timeseries::DatePrecision::Day,
            ))),
            Value::Null => Ok(None),
            _ => Err(format!(
                "ts_*() date argument must be a string, integer, date, or null, got {:?}",
                v
            )),
        }
    }

    /// Resolve 0-2 range arguments into a `(start_idx, end_idx)` slice range.
    pub(super) fn resolve_ts_range(
        &self,
        ts: &crate::graph::features::timeseries::NodeTimeseries,
        range_args: &[Expression],
        row: &ResultRow,
    ) -> Result<(usize, usize), String> {
        if range_args.is_empty() {
            return Ok((0, ts.keys.len()));
        }

        let first = self.resolve_ts_date_arg(&range_args[0], row)?;

        if range_args.len() >= 2 {
            // Two-arg range: [start, end]
            let second = self.resolve_ts_date_arg(&range_args[1], row)?;
            let start = first.map(|(d, _)| d);
            let end =
                second.map(|(d, prec)| crate::graph::features::timeseries::expand_end(d, prec));
            Ok(crate::graph::features::timeseries::find_range(
                &ts.keys, start, end,
            ))
        } else {
            // Single arg: expand to full precision range
            match first {
                Some((date, prec)) => {
                    let end = crate::graph::features::timeseries::expand_end(date, prec);
                    Ok(crate::graph::features::timeseries::find_range(
                        &ts.keys,
                        Some(date),
                        Some(end),
                    ))
                }
                None => Ok((0, ts.keys.len())), // null = no bounds
            }
        }
    }

    /// Extract a Vec<f32> from an expression that is either a ListLiteral or a JSON string.
    pub(super) fn extract_float_list(
        &self,
        expr: &Expression,
        row: &ResultRow,
    ) -> Result<Vec<f32>, String> {
        match expr {
            Expression::ListLiteral(items) => {
                let mut result = Vec::with_capacity(items.len());
                for item in items {
                    match self.evaluate_expression(item, row)? {
                        Value::Float64(f) => result.push(f as f32),
                        Value::Int64(i) => result.push(i as f32),
                        other => {
                            return Err(format!(
                                "vector_score(): query vector elements must be numeric, got {:?}",
                                other
                            ))
                        }
                    }
                }
                Ok(result)
            }
            _ => {
                // Evaluate; accept Value::List (post-A.1 native shape)
                // or Value::String (legacy JSON string).
                let val = self.evaluate_expression(expr, row)?;
                match val {
                    Value::List(items) => {
                        let mut result = Vec::with_capacity(items.len());
                        for item in &items {
                            match item {
                                Value::Float64(f) => result.push(*f as f32),
                                Value::Int64(i) => result.push(*i as f32),
                                other => {
                                    return Err(format!(
                                        "vector_score(): query vector elements must be numeric, got {:?}",
                                        other
                                    ))
                                }
                            }
                        }
                        Ok(result)
                    }
                    Value::String(s) => parse_json_float_list(&s),
                    _ => Err("vector_score(): query vector must be a list of numbers".into()),
                }
            }
        }
    }

    // ========================================================================
    // RETURN
    // ========================================================================
}
