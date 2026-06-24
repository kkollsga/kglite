//! Cypher scalar functions — temporal category. Split out of the monolithic
//! `evaluate_scalar_function` dispatcher; arms are verbatim. Routed from
//! `super::evaluate_scalar_function`; returns `Ok(None)` when `name` is not
//! one of this category's functions so the dispatcher tries the next.
use super::super::*;
use super::shared::*;
use crate::datatypes::values::Value;

impl<'a> CypherExecutor<'a> {
    pub(super) fn eval_temporal_fn(
        &self,
        name: &str,
        args: &[Expression],
        row: &ResultRow,
    ) -> Result<Option<Value>, String> {
        let result: Result<Value, String> = match name {
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
                // Full date + time at second precision (Value::Timestamp).
                // 0-arg form returns local "now"; a bare date parses to
                // midnight. 0.12 Cluster 1 (was date-only via DateTime).
                if args.is_empty() {
                    use chrono::Timelike;
                    let now = chrono::Local::now().naive_local();
                    return Ok(Some(Value::Timestamp(
                        now.with_nanosecond(0).unwrap_or(now),
                    )));
                }
                if args.len() != 1 {
                    return Err(
                        "datetime() requires 0 or 1 argument: datetime() or datetime('2024-03-15T10:30:00')".into(),
                    );
                }
                let val = self.evaluate_expression(&args[0], row)?;
                match val {
                    Value::String(s) => {
                        // Full ISO datetime, else a bare date at midnight.
                        let parsed = chrono::NaiveDateTime::parse_from_str(&s, "%Y-%m-%dT%H:%M:%S")
                            .ok()
                            .or_else(|| {
                                let date_part = s.split('T').next().unwrap_or(&s);
                                crate::graph::features::timeseries::parse_date_query(date_part)
                                    .ok()
                                    .and_then(|(d, _)| d.and_hms_opt(0, 0, 0))
                            });
                        Ok(parsed.map_or(Value::Null, Value::Timestamp))
                    }
                    Value::Timestamp(_) => Ok(val),
                    Value::DateTime(d) => {
                        Ok(Value::Timestamp(d.and_hms_opt(0, 0, 0).unwrap_or_default()))
                    }
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
                if matches!(a, Value::Null) || matches!(b, Value::Null) {
                    return Ok(Some(Value::Null));
                }
                // Accepts date() and datetime() operands (and a mix).
                match (coerce_naive_datetime(&a), coerce_naive_datetime(&b)) {
                    (Some(d1), Some(d2)) => Ok(Value::Int64((d1 - d2).num_days())),
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
                if matches!(a, Value::Null) || matches!(b, Value::Null) {
                    return Ok(Some(Value::Null));
                }
                // Accepts date() and datetime() operands (and a mix). Whole
                // days go in `days`; any remaining sub-day delta (when a
                // Timestamp is involved) is carried in `seconds`.
                match (coerce_naive_datetime(&a), coerce_naive_datetime(&b)) {
                    (Some(start), Some(end)) => {
                        let total_secs = (end - start).num_seconds();
                        Ok(Value::Duration {
                            months: 0,
                            days: (total_secs / 86_400) as i32,
                            seconds: total_secs % 86_400,
                        })
                    }
                    _ => Err("duration.between() arguments must be datetime values".into()),
                }
            }
            // Temporal arithmetic (2026-05-25 broad-scan lift).
            // Real use case: "events scheduled in the next N days":
            //   MATCH (e:Event) WHERE e.date <= add_days(date(), 30)
            // Date math via chrono — NaiveDate handles month/year
            // arithmetic correctly (Feb 28 + 1 year = Feb 28; Jan 31
            // + 1 month = Feb 28/29 depending on leap year).
            "add_days" => {
                if args.len() != 2 {
                    return Err("add_days() requires 2 args: add_days(date, n_days)".into());
                }
                let d = self.evaluate_expression(&args[0], row)?;
                let n = self.evaluate_expression(&args[1], row)?;
                match (&d, &n) {
                    (Value::DateTime(date), Value::Int64(n)) => {
                        match date.checked_add_signed(chrono::Duration::days(*n)) {
                            Some(out) => Ok(Value::DateTime(out)),
                            None => Ok(Value::Null),
                        }
                    }
                    (Value::Null, _) | (_, Value::Null) => Ok(Value::Null),
                    _ => Err("add_days() expects (date, integer)".into()),
                }
            }
            "add_months" => {
                if args.len() != 2 {
                    return Err("add_months() requires 2 args: add_months(date, n_months)".into());
                }
                let d = self.evaluate_expression(&args[0], row)?;
                let n = self.evaluate_expression(&args[1], row)?;
                match (&d, &n) {
                    (Value::DateTime(date), Value::Int64(n)) => {
                        let result = if *n >= 0 {
                            date.checked_add_months(chrono::Months::new(*n as u32))
                        } else {
                            date.checked_sub_months(chrono::Months::new((-*n) as u32))
                        };
                        match result {
                            Some(out) => Ok(Value::DateTime(out)),
                            None => Ok(Value::Null),
                        }
                    }
                    (Value::Null, _) | (_, Value::Null) => Ok(Value::Null),
                    _ => Err("add_months() expects (date, integer)".into()),
                }
            }
            "add_years" => {
                if args.len() != 2 {
                    return Err("add_years() requires 2 args: add_years(date, n_years)".into());
                }
                let d = self.evaluate_expression(&args[0], row)?;
                let n = self.evaluate_expression(&args[1], row)?;
                match (&d, &n) {
                    (Value::DateTime(date), Value::Int64(n)) => {
                        // 12 months per year — chrono's Months handles
                        // leap-year edge case (Feb 29 + 1 year → Feb 28).
                        let months_delta = n.saturating_mul(12);
                        let result = if months_delta >= 0 {
                            date.checked_add_months(chrono::Months::new(months_delta as u32))
                        } else {
                            date.checked_sub_months(chrono::Months::new((-months_delta) as u32))
                        };
                        match result {
                            Some(out) => Ok(Value::DateTime(out)),
                            None => Ok(Value::Null),
                        }
                    }
                    (Value::Null, _) | (_, Value::Null) => Ok(Value::Null),
                    _ => Err("add_years() expects (date, integer)".into()),
                }
            }
            // date_truncate(date, 'unit') — round down to the start of
            // a calendar period. Real use case: group analytics by
            // week/month: `RETURN date_truncate(e.ts, 'month'), count(e)`.
            "date_truncate" => {
                use chrono::{Datelike, NaiveDate};
                if args.len() != 2 {
                    return Err(
                        "date_truncate() requires 2 args: date_truncate(date, 'year'|'month'|'week'|'day')".into()
                    );
                }
                let d = self.evaluate_expression(&args[0], row)?;
                let unit = self.evaluate_expression(&args[1], row)?;
                let (date, unit_str) = match (&d, &unit) {
                    (Value::DateTime(date), Value::String(u)) => (*date, u.as_str()),
                    (Value::Null, _) | (_, Value::Null) => return Ok(Some(Value::Null)),
                    _ => return Err("date_truncate() expects (date, string)".into()),
                };
                let truncated = match unit_str {
                    "year" | "years" => NaiveDate::from_ymd_opt(date.year(), 1, 1),
                    "month" | "months" => NaiveDate::from_ymd_opt(date.year(), date.month(), 1),
                    "week" | "weeks" => {
                        // ISO week starts Monday. Subtract weekday-1 days.
                        let dow = date.weekday().num_days_from_monday() as i64;
                        date.checked_sub_signed(chrono::Duration::days(dow))
                    }
                    "day" | "days" => Some(date),
                    other => {
                        return Err(format!(
                            "date_truncate() unit must be year/month/week/day, got '{other}'"
                        ));
                    }
                };
                Ok(truncated.map(Value::DateTime).unwrap_or(Value::Null))
            }
            "localdatetime" => self.eval_local_temporal(args, row, LocalTemporalKind::DateTime),
            "localtime" => self.eval_local_temporal(args, row, LocalTemporalKind::Time),
            "time" => self.eval_local_temporal(args, row, LocalTemporalKind::Time),
            _ => return Ok(None),
        };
        result.map(Some)
    }
}
