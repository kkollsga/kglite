//! Goldens for the three-valued answer of `<`, `<=`, `>`, `>=`.
//!
//! Two rules meet here and are easy to conflate:
//!
//! * **No ordering rule for this pair of types → `null`** (openCypher). The
//!   engine used to collapse that to `false`, which made `NOT (a < b)` answer
//!   `true` and keep rows Neo4j drops.
//! * **NaN → `false`** (IEEE / Neo4j). A NaN is a *number*; the pair has an
//!   ordering rule, the value simply declines every one of them. CYPHER.md's
//!   ordering section already declares NaN sortable, so this must not become
//!   `null` when the arm above does.

use super::helpers::evaluate_comparison_tristate;
use crate::datatypes::values::Value;
use crate::graph::languages::cypher::ast::ComparisonOp;
use chrono::NaiveDate;

const ORDERING_OPS: [ComparisonOp; 4] = [
    ComparisonOp::LessThan,
    ComparisonOp::LessThanEq,
    ComparisonOp::GreaterThan,
    ComparisonOp::GreaterThanEq,
];

fn eval(left: &Value, op: &ComparisonOp, right: &Value) -> Option<bool> {
    evaluate_comparison_tristate(left, op, right).expect("comparison must not error")
}

fn map_value(key: &str, value: i64) -> Value {
    let mut map = crate::datatypes::prop_map::PropMap::new();
    map.insert(key.to_string(), Value::Int64(value));
    Value::Map(map)
}

/// Pairs whose two values belong to families no ordering rule relates.
fn incomparable_pairs() -> Vec<(Value, Value)> {
    vec![
        (Value::Int64(1), Value::String("a".into())),
        (Value::Float64(1.5), Value::String("a".into())),
        (Value::Boolean(true), Value::Int64(1)),
        (Value::List(vec![Value::Int64(1)]), Value::Int64(2)),
        (map_value("a", 1), Value::Int64(1)),
        // A date against a string that is not a date: the parse fails, so the
        // pair has no rule either.
        (
            Value::DateTime(NaiveDate::from_ymd_opt(2024, 3, 15).unwrap()),
            Value::String("not-a-date".into()),
        ),
        // Composite ordering: Neo4j answers `null` for list-vs-list and
        // map-vs-map `<` too, and `compare_values` has no arm for them.
        (
            Value::List(vec![Value::Int64(1)]),
            Value::List(vec![Value::Int64(2)]),
        ),
        (map_value("a", 1), map_value("a", 2)),
    ]
}

#[test]
fn an_ordering_comparison_without_a_rule_is_null_in_both_directions() {
    for (left, right) in incomparable_pairs() {
        for op in ORDERING_OPS {
            assert_eq!(
                eval(&left, &op, &right),
                None,
                "{left:?} {op:?} {right:?} must be null"
            );
            assert_eq!(
                eval(&right, &op, &left),
                None,
                "{right:?} {op:?} {left:?} must be null"
            );
        }
    }
}

#[test]
fn equality_across_types_stays_two_valued() {
    // openCypher: `=` is false and `<>` is true across type families — only
    // the *ordering* operators went null.
    for (left, right) in incomparable_pairs() {
        assert_eq!(eval(&left, &ComparisonOp::Equals, &right), Some(false));
        assert_eq!(eval(&left, &ComparisonOp::NotEquals, &right), Some(true));
        assert_eq!(eval(&right, &ComparisonOp::Equals, &left), Some(false));
        assert_eq!(eval(&right, &ComparisonOp::NotEquals, &left), Some(true));
    }
}

#[test]
fn nan_answers_false_not_null() {
    let nan = Value::Float64(f64::NAN);
    for other in [
        Value::Int64(1),
        Value::Float64(1.0),
        Value::Float64(f64::NAN),
        Value::UniqueId(7),
    ] {
        for op in ORDERING_OPS {
            assert_eq!(
                eval(&nan, &op, &other),
                Some(false),
                "NaN {op:?} {other:?} must be false"
            );
            assert_eq!(
                eval(&other, &op, &nan),
                Some(false),
                "{other:?} {op:?} NaN must be false"
            );
        }
    }
}

#[test]
fn a_null_operand_is_still_null() {
    for op in ORDERING_OPS {
        assert_eq!(eval(&Value::Null, &op, &Value::Int64(1)), None);
        assert_eq!(eval(&Value::Int64(1), &op, &Value::Null), None);
    }
}

#[test]
fn same_family_ordering_is_unchanged() {
    for (left, right, less, less_eq, greater, greater_eq) in [
        (Value::Int64(1), Value::Int64(2), true, true, false, false),
        (Value::Int64(2), Value::Int64(2), false, true, false, true),
        (
            Value::Float64(1.5),
            Value::Int64(2),
            true,
            true,
            false,
            false,
        ),
        (
            Value::String("a".into()),
            Value::String("b".into()),
            true,
            true,
            false,
            false,
        ),
        (
            Value::Boolean(false),
            Value::Boolean(true),
            true,
            true,
            false,
            false,
        ),
        (
            Value::DateTime(NaiveDate::from_ymd_opt(2024, 3, 15).unwrap()),
            Value::String("2024-03-16".into()),
            true,
            true,
            false,
            false,
        ),
    ] {
        assert_eq!(
            eval(&left, &ComparisonOp::LessThan, &right),
            Some(less),
            "{left:?} < {right:?}"
        );
        assert_eq!(
            eval(&left, &ComparisonOp::LessThanEq, &right),
            Some(less_eq)
        );
        assert_eq!(
            eval(&left, &ComparisonOp::GreaterThan, &right),
            Some(greater)
        );
        assert_eq!(
            eval(&left, &ComparisonOp::GreaterThanEq, &right),
            Some(greater_eq)
        );
    }
}
