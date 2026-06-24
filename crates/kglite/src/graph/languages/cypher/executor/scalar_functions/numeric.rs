//! Cypher scalar functions — numeric category. Split out of the monolithic
//! `evaluate_scalar_function` dispatcher; arms are verbatim. Routed from
//! `super::evaluate_scalar_function`; returns `Ok(None)` when `name` is not
//! one of this category's functions so the dispatcher tries the next.
use super::super::helpers::*;
use super::super::*;
use crate::datatypes::values::Value;

impl<'a> CypherExecutor<'a> {
    pub(super) fn eval_numeric_fn(
        &self,
        name: &str,
        args: &[Expression],
        row: &ResultRow,
    ) -> Result<Option<Value>, String> {
        let result: Result<Value, String> = match name {
            "tointeger" | "toint" => {
                let val = self.evaluate_expression(&args[0], row)?;
                Ok(to_integer(&val))
            }
            "tofloat" => {
                let val = self.evaluate_expression(&args[0], row)?;
                Ok(to_float(&val))
            }
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
            // ── Trigonometric / angular math ──────────────────────────
            // Real use cases: geospatial bearing/heading math and
            // embedding-vector angle computations done server-side in
            // Cypher. All take a numeric arg, return Float64. Null in →
            // null out; non-numeric (and not coercible) → Null. Mirrors
            // the sqrt/abs arms exactly: `value_to_f64` does the coercion,
            // `Value::Null` short-circuits before coercion.
            "sin" | "cos" | "tan" | "asin" | "acos" | "atan" | "cot" | "haversin" | "degrees"
            | "radians" => {
                let val = self.evaluate_expression(&args[0], row)?;
                match val {
                    Value::Null => Ok(Value::Null),
                    _ => match value_to_f64(&val) {
                        Some(f) => {
                            let out = match name {
                                "sin" => f.sin(),
                                "cos" => f.cos(),
                                "tan" => f.tan(),
                                "asin" => f.asin(),
                                "acos" => f.acos(),
                                "atan" => f.atan(),
                                "cot" => 1.0 / f.tan(),
                                // haversin(x) = (1 - cos(x)) / 2 — the
                                // half-versed-sine used by the haversine
                                // great-circle distance formula.
                                "haversin" => (1.0 - f.cos()) / 2.0,
                                "degrees" => f.to_degrees(),
                                "radians" => f.to_radians(),
                                _ => unreachable!(),
                            };
                            Ok(Value::Float64(out))
                        }
                        None => Ok(Value::Null),
                    },
                }
            }
            // atan2(y, x) — two-arg arctangent, quadrant-aware. Real use
            // case: bearing between two geographic points. Either arg
            // Null → Null; either non-numeric → Null.
            "atan2" => {
                if args.len() != 2 {
                    return Err("atan2() requires 2 arguments: atan2(y, x)".into());
                }
                let y_val = self.evaluate_expression(&args[0], row)?;
                let x_val = self.evaluate_expression(&args[1], row)?;
                match (&y_val, &x_val) {
                    (Value::Null, _) | (_, Value::Null) => Ok(Value::Null),
                    _ => match (value_to_f64(&y_val), value_to_f64(&x_val)) {
                        (Some(y), Some(x)) => Ok(Value::Float64(y.atan2(x))),
                        _ => Ok(Value::Null),
                    },
                }
            }
            // randomUUID() — RFC 4122 version-4 UUID string. Non-
            // deterministic; classified alongside rand() in
            // `is_row_independent` (where_clause.rs) so constant folding
            // never collapses it to a single value across rows. No `uuid`
            // crate dependency — we draw 128 random bits from the same
            // thread-local xorshift64 PRNG that rand() uses (two u64
            // draws), then stamp the version (4) and variant (10xx) bits
            // per the v4 layout. Registered under the lowercased key
            // `randomuuid`; the canonical Cypher spelling is randomUUID().
            _ => return Ok(None),
        };
        result.map(Some)
    }
}
