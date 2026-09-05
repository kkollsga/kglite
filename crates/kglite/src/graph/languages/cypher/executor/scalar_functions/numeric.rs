//! Cypher scalar numeric functions.
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
                let val = self.evaluate_expression(super::first_arg(name, args)?, row)?;
                Ok(to_integer(&val))
            }
            "tofloat" => {
                let val = self.evaluate_expression(super::first_arg(name, args)?, row)?;
                Ok(to_float(&val))
            }
            "abs" => {
                let val = self.evaluate_expression(super::first_arg(name, args)?, row)?;
                match val {
                    Value::Int64(n) => n
                        .checked_abs()
                        .map(Value::Int64)
                        .ok_or_else(|| "Integer overflow in abs()".to_string()),
                    Value::Float64(f) => Ok(Value::Float64(f.abs())),
                    Value::Null => Ok(Value::Null),
                    _ => match value_to_f64(&val) {
                        Some(f) => Ok(Value::Float64(f.abs())),
                        None => Ok(Value::Null),
                    },
                }
            }
            "ceil" | "ceiling" => {
                let val = self.evaluate_expression(super::first_arg(name, args)?, row)?;
                match val {
                    Value::Null => Ok(Value::Null),
                    _ => match value_to_f64(&val) {
                        Some(f) => Ok(Value::Float64(f.ceil())),
                        None => Ok(Value::Null),
                    },
                }
            }
            "floor" => {
                let val = self.evaluate_expression(super::first_arg(name, args)?, row)?;
                match val {
                    Value::Null => Ok(Value::Null),
                    _ => match value_to_f64(&val) {
                        Some(f) => Ok(Value::Float64(f.floor())),
                        None => Ok(Value::Null),
                    },
                }
            }
            "round" => {
                let val = self.evaluate_expression(super::first_arg(name, args)?, row)?;
                match val {
                    Value::Null => Ok(Value::Null),
                    _ => match value_to_f64(&val) {
                        Some(f) => {
                            if args.len() >= 2 {
                                let prec = self.evaluate_expression(&args[1], row)?;
                                Ok(Value::Float64(round_decimal(f, decimal_precision(&prec))))
                            } else {
                                Ok(Value::Float64(f.round()))
                            }
                        }
                        None => Ok(Value::Null),
                    },
                }
            }
            "sqrt" => {
                let val = self.evaluate_expression(super::first_arg(name, args)?, row)?;
                match val {
                    Value::Null => Ok(Value::Null),
                    _ => match value_to_f64(&val) {
                        Some(f) if f >= 0.0 => Ok(Value::Float64(f.sqrt())),
                        _ => Ok(Value::Null),
                    },
                }
            }
            "sign" => {
                let val = self.evaluate_expression(super::first_arg(name, args)?, row)?;
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
                let val = self.evaluate_expression(super::first_arg(name, args)?, row)?;
                match val {
                    Value::Null => Ok(Value::Null),
                    _ => match value_to_f64(&val) {
                        Some(f) if f > 0.0 => Ok(Value::Float64(f.ln())),
                        _ => Ok(Value::Null),
                    },
                }
            }
            "log10" => {
                let val = self.evaluate_expression(super::first_arg(name, args)?, row)?;
                match val {
                    Value::Null => Ok(Value::Null),
                    _ => match value_to_f64(&val) {
                        Some(f) if f > 0.0 => Ok(Value::Float64(f.log10())),
                        _ => Ok(Value::Null),
                    },
                }
            }
            "exp" => {
                let val = self.evaluate_expression(super::first_arg(name, args)?, row)?;
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
                let base_val = self.evaluate_expression(super::first_arg(name, args)?, row)?;
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
                let val = self.evaluate_expression(super::first_arg(name, args)?, row)?;
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
                let y_val = self.evaluate_expression(super::first_arg(name, args)?, row)?;
                let x_val = self.evaluate_expression(&args[1], row)?;
                match (&y_val, &x_val) {
                    (Value::Null, _) | (_, Value::Null) => Ok(Value::Null),
                    _ => match (value_to_f64(&y_val), value_to_f64(&x_val)) {
                        (Some(y), Some(x)) => Ok(Value::Float64(y.atan2(x))),
                        _ => Ok(Value::Null),
                    },
                }
            }
            // randomUUID() lives in utility.rs (its doc comment travels
            // with the arm there).
            _ => return Ok(None),
        };
        result.map(Some)
    }
}

// Clamp before narrowing, keeping fractional truncation and nonnumeric/default
// precision behavior. Beyond these bounds decimal rounding cannot change a
// finite f64 except to signed zero at the negative end.
fn decimal_precision(value: &Value) -> i32 {
    match value {
        Value::Int64(value) => (*value).clamp(-309, 324) as i32,
        Value::Float64(value) => value.clamp(-309.0, 324.0) as i32,
        _ => 0,
    }
}

fn round_decimal(value: f64, precision: i32) -> f64 {
    if !value.is_finite() || precision >= 324 {
        return value;
    }
    if precision <= -309 {
        return 0.0_f64.copysign(value);
    }
    if precision < 0 {
        let factor = 10f64.powi(-precision);
        return (value / factor).round() * factor;
    }
    // 10^309 overflows, but subnormals still round at precisions 309..323.
    // Split the scale rather than creating an infinite factor. If the scaled
    // value overflows, the quantum is already below this input's f64 spacing.
    let base = 10f64.powi(precision.min(308));
    let extra = 10f64.powi((precision - 308).max(0));
    let scaled = value * base * extra;
    if !scaled.is_finite() {
        value
    } else {
        scaled.round() / extra / base
    }
}
