//! Runtime validation of recipe variable instances.

use std::cmp::Ordering;
use std::fmt;

use serde_json::{Number, Value};

use super::schema::{SchemaNode, ValueType};

impl SchemaNode {
    pub(super) fn validate(
        &self,
        value: &Value,
        path: &str,
        check_enum: bool,
        issues: &mut Vec<VariableIssue>,
    ) {
        if !self.matches_type(value) {
            let expected = self
                .types
                .iter()
                .map(|kind| kind.display())
                .collect::<Vec<_>>()
                .join(" | ");
            issues.push(VariableIssue::new(
                path,
                VariableIssueKind::WrongType,
                format!("{path} must have type {expected}"),
            ));
            return;
        }

        if check_enum
            && self
                .enum_values
                .as_ref()
                .is_some_and(|values| !values.iter().any(|item| json_equal(item, value)))
        {
            issues.push(VariableIssue::new(
                path,
                VariableIssueKind::Enum,
                format!("{path} is not one of the allowed enum values"),
            ));
        }

        self.validate_number(value, path, issues);
        self.validate_array(value, path, issues);
        self.validate_object(value, path, issues);
    }

    fn validate_number(&self, value: &Value, path: &str, issues: &mut Vec<VariableIssue>) {
        let Some(number) = value.as_number() else {
            return;
        };
        if self
            .minimum
            .as_ref()
            .is_some_and(|minimum| number_cmp(number, minimum) == Some(Ordering::Less))
        {
            issues.push(VariableIssue::new(
                path,
                VariableIssueKind::Minimum,
                format!("{path} is below minimum"),
            ));
        }
        if self
            .maximum
            .as_ref()
            .is_some_and(|maximum| number_cmp(number, maximum) == Some(Ordering::Greater))
        {
            issues.push(VariableIssue::new(
                path,
                VariableIssueKind::Maximum,
                format!("{path} exceeds maximum"),
            ));
        }
    }

    fn validate_array(&self, value: &Value, path: &str, issues: &mut Vec<VariableIssue>) {
        let Some(array) = value.as_array() else {
            return;
        };
        if let Some(minimum) = self.min_items {
            if array.len() < minimum {
                issues.push(VariableIssue::new(
                    path,
                    VariableIssueKind::MinItems,
                    format!("{path} has fewer than {minimum} items"),
                ));
            }
        }
        if let Some(maximum) = self.max_items {
            if array.len() > maximum {
                issues.push(VariableIssue::new(
                    path,
                    VariableIssueKind::MaxItems,
                    format!("{path} has more than {maximum} items"),
                ));
            }
        }
        if let Some(items) = self.items.as_ref() {
            for (index, item) in array.iter().enumerate() {
                items.validate(item, &format!("{path}[{index}]"), true, issues);
            }
        }
    }

    fn validate_object(&self, value: &Value, path: &str, issues: &mut Vec<VariableIssue>) {
        let Some(object) = value.as_object() else {
            return;
        };
        for required in &self.required {
            if !object.contains_key(required) {
                let item_path = child_path(path, required);
                issues.push(VariableIssue::new(
                    &item_path,
                    VariableIssueKind::Missing,
                    format!("{item_path} is required"),
                ));
            }
        }
        for (name, item) in object {
            let item_path = child_path(path, name);
            if let Some(schema) = self.properties.get(name) {
                schema.validate(item, &item_path, true, issues);
            } else if self.additional_properties == Some(false) {
                issues.push(VariableIssue::new(
                    &item_path,
                    VariableIssueKind::Unknown,
                    format!("{item_path} is not an allowed property"),
                ));
            }
        }
    }

    fn matches_type(&self, value: &Value) -> bool {
        self.types.iter().any(|kind| match kind {
            ValueType::Null => value.is_null(),
            ValueType::Boolean => value.is_boolean(),
            ValueType::Object => value.is_object(),
            ValueType::Array => value.is_array(),
            ValueType::Number => value.is_number(),
            ValueType::Integer => value.as_number().is_some_and(is_json_integer),
            ValueType::String => value.is_string(),
        })
    }
}

/// JSON Schema's `integer` is mathematical, not tied to the lexical spelling:
/// `1` and `1.0` are both integers. KGLite can represent exact integer values
/// only through signed 64-bit range, so integral floats use the same bound.
fn is_json_integer(number: &Number) -> bool {
    if number.as_i64().is_some() {
        return true;
    }
    if number.as_u64().is_some() {
        // The only u64 values not already exposed by as_i64 exceed i64::MAX.
        return false;
    }
    number.as_f64().is_some_and(is_integral_i64_float)
}

fn is_integral_i64_float(value: f64) -> bool {
    const I64_EXCLUSIVE_UPPER: f64 = 9_223_372_036_854_775_808.0;
    value.is_finite()
        && value.fract() == 0.0
        && value >= i64::MIN as f64
        && value < I64_EXCLUSIVE_UPPER
}

/// Stable categories used to construct `invalid_variables` details later.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum VariableIssueKind {
    Missing,
    Unknown,
    WrongType,
    IntegerRange,
    Enum,
    Minimum,
    Maximum,
    MinItems,
    MaxItems,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct VariableIssue {
    pub(crate) path: String,
    pub(crate) kind: VariableIssueKind,
    pub(crate) message: String,
}

impl VariableIssue {
    fn new(path: &str, kind: VariableIssueKind, message: String) -> Self {
        Self {
            path: path.to_string(),
            kind,
            message,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct VariablesValidationError {
    pub(crate) issues: Vec<VariableIssue>,
}

impl fmt::Display for VariablesValidationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let messages = self
            .issues
            .iter()
            .map(|issue| issue.message.as_str())
            .collect::<Vec<_>>()
            .join("; ");
        write!(formatter, "{messages}")
    }
}

impl std::error::Error for VariablesValidationError {}

pub(super) fn validate_exact_i64_recursive(
    value: &Value,
    path: &str,
    issues: &mut Vec<VariableIssue>,
) {
    match value {
        Value::Number(number) if number.as_i64().is_none() && number.as_u64().is_some() => {
            issues.push(VariableIssue::new(
                path,
                VariableIssueKind::IntegerRange,
                format!("{path} integer is outside KGLite's exact signed 64-bit range"),
            ));
        }
        Value::Array(items) => {
            for (index, item) in items.iter().enumerate() {
                validate_exact_i64_recursive(item, &format!("{path}[{index}]"), issues);
            }
        }
        Value::Object(map) => {
            for (name, item) in map {
                validate_exact_i64_recursive(item, &child_path(path, name), issues);
            }
        }
        _ => {}
    }
}

fn child_path(parent: &str, name: &str) -> String {
    format!("{parent}.{name}")
}

pub(super) fn number_cmp(left: &Number, right: &Number) -> Option<Ordering> {
    match (integer_as_i128(left), integer_as_i128(right)) {
        (Some(left), Some(right)) => Some(left.cmp(&right)),
        (Some(left), None) => compare_integer_to_float(left, right.as_f64()?),
        (None, Some(right)) => {
            compare_integer_to_float(right, left.as_f64()?).map(Ordering::reverse)
        }
        (None, None) => left.as_f64()?.partial_cmp(&right.as_f64()?),
    }
}

/// Compare an exact JSON integer with a finite JSON float without first
/// rounding the integer through `f64` (which loses adjacent values above
/// 2^53). JSON numbers cannot represent NaN/inf, but keep the guard explicit.
fn compare_integer_to_float(integer: i128, float: f64) -> Option<Ordering> {
    if !float.is_finite() {
        return None;
    }
    if float >= i128::MAX as f64 {
        return Some(Ordering::Less);
    }
    if float <= i128::MIN as f64 {
        return Some(Ordering::Greater);
    }

    let truncated = float.trunc() as i128;
    match integer.cmp(&truncated) {
        Ordering::Equal if float.fract().is_sign_positive() && float.fract() != 0.0 => {
            Some(Ordering::Less)
        }
        Ordering::Equal if float.fract().is_sign_negative() && float.fract() != 0.0 => {
            Some(Ordering::Greater)
        }
        ordering => Some(ordering),
    }
}

fn integer_as_i128(number: &Number) -> Option<i128> {
    number
        .as_i64()
        .map(i128::from)
        .or_else(|| number.as_u64().map(i128::from))
}

pub(super) fn json_equal(left: &Value, right: &Value) -> bool {
    match (left, right) {
        (Value::Number(left), Value::Number(right)) => {
            number_cmp(left, right) == Some(Ordering::Equal)
        }
        (Value::Array(left), Value::Array(right)) => {
            left.len() == right.len()
                && left
                    .iter()
                    .zip(right)
                    .all(|(left, right)| json_equal(left, right))
        }
        (Value::Object(left), Value::Object(right)) => {
            left.len() == right.len()
                && left
                    .iter()
                    .all(|(key, left)| right.get(key).is_some_and(|right| json_equal(left, right)))
        }
        _ => left == right,
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use kglite::api::Value as KgliteValue;
    use serde_json::{json, Map, Value};

    use super::*;
    use crate::recipe_queries::ParameterSchema;

    fn compile(properties: Value, required: Value) -> ParameterSchema {
        ParameterSchema::compile_root(
            &json!({
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": false
            }),
            &["value".to_string()],
        )
        .unwrap()
    }

    #[test]
    fn nullable_type_arrays_accept_null_and_the_declared_value() {
        let schema = compile(
            json!({"value": {"type": ["string", "null"]}}),
            json!(["value"]),
        );
        schema
            .validate_variables(&Map::from_iter([("value".into(), Value::Null)]))
            .unwrap();
        schema
            .validate_variables(&Map::from_iter([("value".into(), json!("name"))]))
            .unwrap();
        let error = schema
            .validate_variables(&Map::from_iter([("value".into(), json!(3))]))
            .unwrap_err();
        assert_eq!(error.issues[0].kind, VariableIssueKind::WrongType);
    }

    #[test]
    fn nested_arrays_objects_enum_and_bounds_are_validated() {
        let schema = compile(
            json!({
                "value": {
                    "type": "array", "minItems": 1, "maxItems": 2,
                    "items": {
                        "type": "object",
                        "properties": {
                            "score": {"type": "integer", "minimum": 0, "maximum": 10},
                            "kind": {"type": "string", "enum": ["unit", "integration"]}
                        },
                        "required": ["score", "kind"],
                        "additionalProperties": false
                    }
                }
            }),
            json!(["value"]),
        );
        schema
            .validate_variables(&Map::from_iter([(
                "value".into(),
                json!([{"score": 7, "kind": "unit"}]),
            )]))
            .unwrap();

        let error = schema
            .validate_variables(&Map::from_iter([(
                "value".into(),
                json!([{"score": 11, "kind": "other", "extra": true}]),
            )]))
            .unwrap_err();
        let kinds: BTreeSet<_> = error.issues.iter().map(|issue| issue.kind).collect();
        assert!(kinds.contains(&VariableIssueKind::Maximum));
        assert!(kinds.contains(&VariableIssueKind::Enum));
        assert!(kinds.contains(&VariableIssueKind::Unknown));
    }

    #[test]
    fn reports_missing_and_unknown_top_level_variables() {
        let schema = compile(json!({"value": {"type": "string"}}), json!(["value"]));
        let error = schema
            .validate_variables(&Map::from_iter([("other".into(), json!("x"))]))
            .unwrap_err();
        assert_eq!(error.issues.len(), 2);
        assert!(error
            .issues
            .iter()
            .any(|issue| issue.kind == VariableIssueKind::Missing));
        assert!(error
            .issues
            .iter()
            .any(|issue| issue.kind == VariableIssueKind::Unknown));
    }

    #[test]
    fn integer_bounds_above_f64_exact_range_are_compared_without_rounding() {
        let schema = compile(
            json!({"value": {"type": "integer", "maximum": 9007199254740992.0}}),
            json!(["value"]),
        );
        let error = schema
            .validate_variables(&Map::from_iter([(
                "value".into(),
                json!(9007199254740993_i64),
            )]))
            .unwrap_err();
        assert!(error
            .issues
            .iter()
            .any(|issue| issue.kind == VariableIssueKind::Maximum));
    }

    #[test]
    fn integer_type_accepts_integral_floats_without_changing_conversion_semantics() {
        let schema = compile(json!({"value": {"type": "integer"}}), json!(["value"]));
        let variables = Map::from_iter([("value".into(), json!(1.0))]);
        schema.validate_variables(&variables).unwrap();
        assert!(matches!(
            kglite::api::param::json_value_to_kglite_value(&variables["value"]),
            KgliteValue::Float64(value) if value == 1.0
        ));

        for rejected in [json!(1.5), json!(1e100), json!(9223372036854775808.0)] {
            let error = schema
                .validate_variables(&Map::from_iter([("value".into(), rejected)]))
                .unwrap_err();
            assert!(error
                .issues
                .iter()
                .any(|issue| issue.kind == VariableIssueKind::WrongType));
        }

        for accepted in [
            json!(i64::MIN),
            json!(i64::MAX),
            json!(-9223372036854775808.0),
        ] {
            schema
                .validate_variables(&Map::from_iter([("value".into(), accepted)]))
                .unwrap();
        }
        assert!(!is_integral_i64_float(f64::NAN));
        assert!(!is_integral_i64_float(f64::INFINITY));
    }

    #[test]
    fn enum_numeric_equality_recurses_through_arrays_and_objects() {
        let duplicate = ParameterSchema::compile_root(
            &json!({
                "type": "object",
                "properties": {
                    "value": {
                        "type": "object",
                        "enum": [
                            {"nested": [1, {"score": 2.0}]},
                            {"nested": [1.0, {"score": 2}]}
                        ]
                    }
                },
                "required": ["value"],
                "additionalProperties": false
            }),
            &["value".to_string()],
        )
        .unwrap_err();
        assert!(format!("{duplicate:#}").contains("duplicate value"));

        let schema = compile(
            json!({
                "value": {
                    "type": "object",
                    "enum": [{"nested": [1.0, {"score": 2}]}]
                }
            }),
            json!(["value"]),
        );
        schema
            .validate_variables(&Map::from_iter([(
                "value".into(),
                json!({"nested": [1, {"score": 2.0}]}),
            )]))
            .unwrap();
        let error = schema
            .validate_variables(&Map::from_iter([(
                "value".into(),
                json!({"nested": [1, {"score": 3}]}),
            )]))
            .unwrap_err();
        assert!(error
            .issues
            .iter()
            .any(|issue| issue.kind == VariableIssueKind::Enum));
    }

    #[test]
    fn rejects_integers_outside_i64_at_boot_and_runtime_without_f64_conversion() {
        let overflow = json!(9223372036854775808_u64);
        let schema_error = ParameterSchema::compile_root(
            &json!({
                "type": "object",
                "properties": {"value": {"type": "integer", "enum": [overflow.clone()]}},
                "required": ["value"],
                "additionalProperties": false
            }),
            &["value".to_string()],
        )
        .unwrap_err();
        assert!(format!("{schema_error:#}").contains("signed 64-bit"));

        let schema = compile(json!({"value": {"type": "number"}}), json!(["value"]));
        let runtime_error = schema
            .validate_variables(&Map::from_iter([("value".into(), overflow)]))
            .unwrap_err();
        assert!(runtime_error
            .issues
            .iter()
            .any(|issue| issue.kind == VariableIssueKind::IntegerRange));
    }

    #[test]
    fn rejects_unsupported_keywords_and_keyword_collisions() {
        let unsupported = ParameterSchema::compile_root(
            &json!({
                "type": "object",
                "properties": {"value": {"type": "string", "pattern": "x"}},
                "required": ["value"],
                "additionalProperties": false
            }),
            &["value".to_string()],
        )
        .unwrap_err();
        assert!(format!("{unsupported:#}").contains("unsupported JSON Schema keywords"));

        let duplicate_required = ParameterSchema::compile_root(
            &json!({
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value", "value"],
                "additionalProperties": false
            }),
            &["value".to_string()],
        )
        .unwrap_err();
        assert!(format!("{duplicate_required:#}").contains("duplicate property"));
    }

    #[test]
    fn root_is_closed_and_every_parameter_is_required() {
        for raw in [
            json!({"type": "object", "properties": {}, "required": []}),
            json!({"type": "object", "additionalProperties": false}),
            json!({
                "type": ["object", "null"], "properties": {}, "required": [],
                "additionalProperties": false
            }),
            json!({
                "type": "object", "properties": {"value": {"type": "string"}},
                "required": [], "additionalProperties": false
            }),
        ] {
            assert!(ParameterSchema::compile_root(&raw, &[]).is_err());
        }
    }
}
