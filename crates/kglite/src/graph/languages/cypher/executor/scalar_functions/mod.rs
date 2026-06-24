//! Cypher executor — scalar (non-aggregate) function dispatch.
//!
//! `evaluate_scalar_function` is a thin dispatcher that delegates to the
//! per-category submodules (`string`, `numeric`, `temporal`, `graph`,
//! `spatial`, `collection`, `timeseries`, `utility`); shared free helpers
//! live in `shared`. Lives in a sibling `impl<'a> CypherExecutor<'a> {}` block.
use super::helpers::*;
use super::*;
use crate::datatypes::values::Value;

mod collection;
mod graph;
mod numeric;
mod shared;
mod spatial;
mod string;
mod temporal;
mod timeseries;
mod utility;

use shared::*;

impl<'a> CypherExecutor<'a> {
    /// Evaluate `localdatetime()` / `localtime()` / `time()`. No-arg form
    /// returns the local wall-clock "now" as an ISO-8601 string; the
    /// single-string form validates/normalises and returns `Null` on
    /// unparseable input (mirrors `datetime()`'s Null-on-bad-input
    /// contract). Any other arity/type is an error.
    fn eval_local_temporal(
        &self,
        args: &[Expression],
        row: &ResultRow,
        kind: LocalTemporalKind,
    ) -> Result<Value, String> {
        use chrono::{NaiveTime, Timelike};
        if args.is_empty() {
            let now = chrono::Local::now();
            return match kind {
                // localdatetime() → full date+time at second precision.
                LocalTemporalKind::DateTime => Ok(Value::Timestamp(
                    now.naive_local()
                        .with_nanosecond(0)
                        .unwrap_or(now.naive_local()),
                )),
                // localtime() stays a string — there is no time-of-day Value variant.
                LocalTemporalKind::Time => Ok(Value::String(now.format("%H:%M:%S").to_string())),
            };
        }
        if args.len() != 1 {
            return Err("local temporal functions take 0 or 1 string argument".into());
        }
        let val = self.evaluate_expression(&args[0], row)?;
        let s = match val {
            Value::String(s) => s,
            Value::Null => return Ok(Value::Null),
            _ => return Err("local temporal argument must be a string".into()),
        };
        match kind {
            LocalTemporalKind::DateTime => {
                // Accept full ISO datetime, or a bare date (midnight).
                // Returns a Value::Timestamp (date + time, second precision).
                if let Ok(dt) = chrono::NaiveDateTime::parse_from_str(&s, "%Y-%m-%dT%H:%M:%S") {
                    Ok(Value::Timestamp(dt))
                } else if let Ok(d) =
                    chrono::NaiveDate::parse_from_str(s.split('T').next().unwrap_or(&s), "%Y-%m-%d")
                {
                    Ok(Value::Timestamp(d.and_hms_opt(0, 0, 0).unwrap_or_default()))
                } else {
                    Ok(Value::Null)
                }
            }
            LocalTemporalKind::Time => {
                // Accept HH:MM:SS or HH:MM.
                if let Ok(t) = NaiveTime::parse_from_str(&s, "%H:%M:%S") {
                    Ok(Value::String(format!(
                        "{:02}:{:02}:{:02}",
                        t.hour(),
                        t.minute(),
                        t.second()
                    )))
                } else if let Ok(t) = NaiveTime::parse_from_str(&s, "%H:%M") {
                    Ok(Value::String(format!(
                        "{:02}:{:02}:{:02}",
                        t.hour(),
                        t.minute(),
                        t.second()
                    )))
                } else {
                    Ok(Value::Null)
                }
            }
        }
    }

    /// Resolve a function argument that denotes a node to its live
    /// `NodeIndex`. Handles a bound node variable (fast path) AND a node
    /// arriving as a `Value::NodeRef` — the shape that `collect(a)[0]`,
    /// `head(collect(a))`, and `WITH … AS x` projection preserve. Without
    /// the NodeRef arm, `labels()` / `keys()` / `properties()` / `id()` on
    /// a collected node silently returned NULL: the node value was intact
    /// (property access and `RETURN` worked) but these functions only
    /// consulted `node_bindings`.
    fn node_arg_index(
        &self,
        arg: &Expression,
        row: &ResultRow,
    ) -> Option<petgraph::graph::NodeIndex> {
        if let Expression::Variable(var) = arg {
            if let Some(&idx) = row.node_bindings.get(var.as_str()) {
                return Some(idx);
            }
        }
        match self.evaluate_expression(arg, row).ok()? {
            Value::NodeRef(i) => Some(petgraph::graph::NodeIndex::new(i as usize)),
            _ => None,
        }
    }

    /// Evaluate scalar (non-aggregate) functions by delegating to the
    /// per-category modules in order. Each `eval_*_fn` returns `Ok(None)`
    /// when it does not own `name`, so the first owner wins.
    pub(super) fn evaluate_scalar_function(
        &self,
        name: &str,
        args: &[Expression],
        row: &ResultRow,
    ) -> Result<Value, String> {
        if let Some(v) = self.eval_string_fn(name, args, row)? {
            return Ok(v);
        }
        if let Some(v) = self.eval_numeric_fn(name, args, row)? {
            return Ok(v);
        }
        if let Some(v) = self.eval_temporal_fn(name, args, row)? {
            return Ok(v);
        }
        if let Some(v) = self.eval_graph_fn(name, args, row)? {
            return Ok(v);
        }
        if let Some(v) = self.eval_collection_fn(name, args, row)? {
            return Ok(v);
        }
        if let Some(v) = self.eval_spatial_fn(name, args, row)? {
            return Ok(v);
        }
        if let Some(v) = self.eval_timeseries_fn(name, args, row)? {
            return Ok(v);
        }
        if let Some(v) = self.eval_utility_fn(name, args, row)? {
            return Ok(v);
        }
        Err(format!("Unknown function: {}", name))
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
