//! Cypher scalar functions — collection category. Split out of the monolithic
//! `evaluate_scalar_function` dispatcher; arms are verbatim. Routed from
//! `super::evaluate_scalar_function`; returns `Ok(None)` when `name` is not
//! one of this category's functions so the dispatcher tries the next.
use super::super::helpers::*;
use super::super::*;
use crate::datatypes::values::Value;

impl<'a> CypherExecutor<'a> {
    pub(super) fn eval_collection_fn(
        &self,
        name: &str,
        args: &[Expression],
        row: &ResultRow,
    ) -> Result<Option<Value>, String> {
        let result: Result<Value, String> = match name {
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
                        return Ok(Some(Value::Int64(path.hops as i64)));
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
            "coalesce" => {
                // coalesce(expr1, expr2, ...) returns first non-null
                for arg in args {
                    let val = self.evaluate_expression(arg, row)?;
                    if !matches!(val, Value::Null) {
                        return Ok(Some(val));
                    }
                }
                Ok(Value::Null)
            }
            "reverse" => {
                if args.len() != 1 {
                    return Err("reverse() requires 1 argument".into());
                }
                match self.evaluate_expression(&args[0], row)? {
                    // Cypher reverse() on a list reverses its elements.
                    Value::List(mut items) => {
                        items.reverse();
                        Ok(Value::List(items))
                    }
                    Value::Null => Ok(Value::Null),
                    other => {
                        let s = match coerce_to_string(other) {
                            Value::String(s) => s,
                            _ => return Ok(Some(Value::Null)),
                        };
                        let trimmed = s.trim();
                        if trimmed.starts_with('[') && trimmed.ends_with(']') {
                            // A bracketed string is a list, consistent with
                            // head/last/size (parse_list_value) — reverse elements.
                            let mut items = parse_list_value(&Value::String(s));
                            items.reverse();
                            Ok(Value::List(items))
                        } else {
                            // Otherwise reverse characters.
                            Ok(Value::String(s.chars().rev().collect()))
                        }
                    }
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
            _ => return Ok(None),
        };
        result.map(Some)
    }
}
