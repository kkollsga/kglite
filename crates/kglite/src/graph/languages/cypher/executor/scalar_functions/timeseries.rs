//! Cypher scalar functions — timeseries category. Split out of the monolithic
//! `evaluate_scalar_function` dispatcher; arms are verbatim. Routed from
//! `super::evaluate_scalar_function`; returns `Ok(None)` when `name` is not
//! one of this category's functions so the dispatcher tries the next.
use super::super::*;
use crate::datatypes::values::Value;

impl<'a> CypherExecutor<'a> {
    pub(super) fn eval_timeseries_fn(
        &self,
        name: &str,
        args: &[Expression],
        row: &ResultRow,
    ) -> Result<Option<Value>, String> {
        let result: Result<Value, String> = match name {
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
            _ => return Ok(None),
        };
        result.map(Some)
    }
}
