//! Cypher executor — scalar (non-aggregate) function dispatch.
//!
//! `evaluate_scalar_function` is a thin dispatcher that delegates to the
//! per-category submodules (`string`, `numeric`, `temporal`, `graph`,
//! `spatial`, `collection`, `conversion`, `timeseries`, `utility`); shared free helpers
//! live in `shared`. Lives in a sibling `impl<'a> CypherExecutor<'a> {}` block.
use super::helpers::*;
use super::*;
use crate::datatypes::values::Value;

mod collection;
mod conversion;
mod edge_text;
mod embedding_readout;
mod function_registry;
mod graph;
mod numeric;
mod shared;
mod spatial;
mod string;
mod temporal;
pub(in crate::graph::languages::cypher::executor) mod temporal_accessors;
mod temporal_duration;
mod temporal_iso;
mod temporal_map;
mod temporal_time;
mod temporal_truncate;
mod temporal_zoned;
mod text_hits;
mod text_match;
mod timeseries;
mod utility;
pub(super) mod validity;
mod vector;

pub(super) use function_registry::{FunctionSpec, FUNCTIONS};

/// Parse a datetime string exactly as `datetime(s)` does: an offset is
/// applied, normalising the reading to UTC. `None` when `datetime(s)` is null.
pub(crate) fn parse_datetime_utc(text: &str) -> Option<chrono::NaiveDateTime> {
    shared::parse_iso_datetime(text).map(|parsed| parsed.utc())
}

/// A `datetime` text as a value: an instant with its offset when the text names
/// one, else the local reading. The reader behind the `$datetime` parameter tag.
pub(crate) fn parse_datetime_value(text: &str) -> Option<Value> {
    let parsed = shared::parse_iso_datetime(text)?;
    match parsed.offset {
        None => Some(Value::Timestamp(parsed.local)),
        Some(offset) => {
            let offset_secs = offset.local_minus_utc();
            (offset_secs.abs() <= crate::datatypes::time_of_day::MAX_OFFSET_SECS)
                .then(|| Value::zoned(parsed.utc(), offset_secs))
        }
    }
}

impl<'a> CypherExecutor<'a> {
    #[cfg(test)]
    pub(super) fn test_eval_collection_fn(
        &self,
        name: &str,
        args: &[Expression],
        row: &ResultRow,
    ) -> Result<Option<Value>, String> {
        self.eval_collection_fn(name, args, row)
    }

    /// Evaluate `localdatetime()`. The no-arg form reads the statement clock on the local wall;
    /// the single-string form validates/normalises, returns `Null` for a null
    /// argument and raises on an unparseable string (as `date()`/`datetime()`
    /// do). Any other arity/type is an error.
    fn eval_localdatetime(&self, args: &[Expression], row: &ResultRow) -> Result<Value, String> {
        use chrono::NaiveTime;
        if args.is_empty() {
            return Ok(Value::Timestamp(self.budget.statement_clock().local()));
        }
        if args.len() != 1 {
            return Err("localdatetime() takes 0 or 1 argument".into());
        }
        let s = match self.evaluate_expression(&args[0], row)? {
            Value::String(s) => s,
            Value::Null => return Ok(Value::Null),
            Value::Map(map) => {
                return temporal_map::temporal_from_map(
                    temporal_map::MapTarget::LocalDateTime,
                    &map,
                );
            }
            Value::Timestamp(t) => return Ok(Value::Timestamp(t)),
            // The wall clock the zoned value was written in.
            Value::ZonedDateTime {
                utc, offset_secs, ..
            } => {
                return Ok(Value::Timestamp(
                    crate::datatypes::zoned_datetime::wall_clock(utc, offset_secs),
                ))
            }
            Value::DateTime(d) => return Ok(Value::Timestamp(d.and_time(NaiveTime::MIN))),
            _ => {
                return Err(
                    "localdatetime() argument must be a string, a map or a temporal value".into(),
                )
            }
        };
        // The ISO calendar/clock spellings first; then the zoned forms, read
        // as `local` rather than `utc`: this function names the *local*
        // reading, so an offset-bearing input keeps its wall clock and only
        // loses the zone label — the opposite of `datetime()`, which
        // normalises. The time is never dropped.
        temporal_iso::parse_local_datetime_text(&s)
            .map(Value::Timestamp)
            .or_else(|| shared::parse_iso_datetime(&s).map(|parsed| Value::Timestamp(parsed.local)))
            .ok_or_else(|| format!("localdatetime(): cannot parse '{s}' as an ISO-8601 datetime"))
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

    /// Probe hook for the `function_registry` drift gate: the same dispatch
    /// path a query takes, reachable from the sibling test module.
    #[cfg(test)]
    pub(super) fn test_evaluate_scalar_function(
        &self,
        name: &str,
        args: &[Expression],
        row: &ResultRow,
    ) -> Result<Value, String> {
        self.evaluate_scalar_function(name, args, row)
    }

    /// Evaluate scalar (non-aggregate) functions by delegating to the
    /// per-category modules in order. Each `eval_*_fn` returns `Ok(None)`
    /// when it does not own `name`, so the first owner wins.
    ///
    /// **Adding a `match` arm to a category module means adding an entry to
    /// [`function_registry::FUNCTIONS`]** — that table is what `SHOW
    /// FUNCTIONS` lists, and a client's autocomplete only knows what it
    /// advertises. The registry→dispatcher direction is enforced (every
    /// registered name is executed against this chain by
    /// `function_registry::tests::every_registry_name_dispatches`); this
    /// direction is not, and cannot be, so it is a rule rather than a gate.
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
        if let Some(v) = self.eval_vector_fn(name, args, row)? {
            return Ok(v);
        }
        if let Some(v) = self.eval_embedding_readout_fn(name, args, row)? {
            return Ok(v);
        }
        if let Some(v) = self.eval_utility_fn(name, args, row)? {
            return Ok(v);
        }
        if let Some(v) = self.eval_conversion_fn(name, args, row)? {
            return Ok(v);
        }
        Err(format!("Unknown function: {}", name))
    }

    // ── Timeseries helpers ─────────────────────────────────────────────

    /// Resolve the first argument of a ts_*() function into the node's timeseries
    /// data, the specific channel's values, and the timeseries config.
    /// The argument must be a PropertyAccess (e.g. `f.output`).
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
            .node_view(*node_idx)
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
    /// Accepts string date queries, integer years, date and datetime values (a zoned one at its UTC day), and Null.
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
            Value::Timestamp(t) | Value::ZonedDateTime { utc: t, .. } => Ok(Some((
                t.date(),
                crate::graph::features::timeseries::DatePrecision::Day,
            ))),
            Value::Null => Ok(None),
            _ => Err(format!(
                "ts_*() date argument must be a string, integer, date, datetime, or null, got {:?}",
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
                // Evaluate; accept Value::List (the native shape)
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
                    Value::String(s) => {
                        parse_json_float_list(&s).map_err(|e| format!("vector_score(): {e}"))
                    }
                    _ => Err("vector_score(): query vector must be a list of numbers".into()),
                }
            }
        }
    }

    // ========================================================================
    // RETURN
    // ========================================================================
}

/// Checked first-argument fetch for the category dispatchers. Pre-fix,
/// 26 single-argument functions indexed `&args[0]` unguarded, so a zero-arg
/// call — `RETURN size()` — panicked with an index-out-of-bounds instead of
/// erroring (measured 2026-08-15 across numeric/string/collection; over Bolt
/// a panic aborts the connection). The parser permits empty argument lists,
/// so the guard belongs here.
pub(super) fn first_arg<'e>(name: &str, args: &'e [Expression]) -> Result<&'e Expression, String> {
    args.first()
        .ok_or_else(|| format!("{name}() requires at least one argument"))
}
