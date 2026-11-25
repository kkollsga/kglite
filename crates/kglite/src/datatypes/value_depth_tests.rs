use super::*;
use crate::datatypes::PropMap;

fn nested_list(depth: usize) -> Value {
    let mut v = Value::Int64(1);
    for _ in 0..depth {
        v = Value::List(vec![v]);
    }
    v
}

#[test]
fn scalars_have_depth_zero_and_each_list_adds_one() {
    assert_eq!(container_depth(&Value::Int64(1), MAX_VALUE_DEPTH), 0);
    assert_eq!(container_depth(&nested_list(1), MAX_VALUE_DEPTH), 1);
    assert_eq!(container_depth(&nested_list(7), MAX_VALUE_DEPTH), 7);
    let wide = Value::List(vec![Value::Int64(1), nested_list(3), Value::Null]);
    assert_eq!(container_depth(&wide, MAX_VALUE_DEPTH), 4);
}

#[test]
fn a_map_counts_as_a_level() {
    let map = PropMap::from_iter([("a".to_string(), nested_list(2))]);
    assert_eq!(container_depth(&Value::Map(map), MAX_VALUE_DEPTH), 3);
}

#[test]
fn the_walk_stops_past_the_limit() {
    let deep = nested_list(MAX_VALUE_DEPTH + 50);
    assert_eq!(container_depth(&deep, 10), 11);
}

#[test]
fn wrapping_is_allowed_up_to_the_ceiling_and_refused_past_it() {
    let at_limit_minus_one = nested_list(MAX_VALUE_DEPTH - 1);
    assert!(check_wrap([&at_limit_minus_one]).is_ok());
    let at_limit = nested_list(MAX_VALUE_DEPTH);
    let err = check_wrap([&at_limit]).unwrap_err();
    assert!(err.contains("256"), "{err}");
    assert!(check_wrap([&Value::Int64(1), &Value::Null]).is_ok());
}

#[test]
fn a_whole_value_is_checked_against_the_ceiling() {
    assert!(check_value(&nested_list(MAX_VALUE_DEPTH)).is_ok());
    assert!(check_value(&nested_list(MAX_VALUE_DEPTH + 1)).is_err());
    assert!(check_value(&Value::String("x".into())).is_ok());
}
