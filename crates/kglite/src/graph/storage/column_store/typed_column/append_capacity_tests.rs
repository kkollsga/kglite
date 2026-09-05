use super::*;
use crate::graph::storage::mapped::mmap_vec::{fail_next, FailurePoint};

fn equal(actual: Option<Value>, expected: &Value) {
    let actual = actual.unwrap_or(Value::Null);
    if let (Value::Float64(a), Value::Float64(b)) = (&actual, expected) {
        assert_eq!(a.to_bits(), b.to_bits());
    } else {
        assert_eq!(actual, *expected);
    }
}

#[test]
fn append_copy_preserves_scalar_bits_nulls_and_source() {
    for value in [
        Value::Int64(i64::MIN),
        Value::Float64(f64::from_bits(0x7ff8_0000_0000_0042)),
        Value::UniqueId(u32::MAX),
        Value::Boolean(true),
        Value::DateTime(NaiveDate::from_ymd_opt(1960, 2, 29).unwrap()),
        Value::String("λ🙂".into()),
    ] {
        let mut base = TypedColumn::for_value(&value);
        let expected: Vec<Value> = (0..16)
            .map(|i| {
                if i % 3 == 0 {
                    Value::Null
                } else {
                    value.clone()
                }
            })
            .collect();
        for v in &expected {
            base.push(v).unwrap();
        }
        let original = Arc::new(base);
        let mut copied = Arc::clone(&original);
        for incoming in [&value, &Value::Null] {
            TypedColumn::make_mut_for_append(&mut copied, incoming)
                .push(incoming)
                .unwrap();
        }
        for (i, v) in expected.iter().enumerate() {
            equal(original.get(i as u32), v);
            equal(copied.get(i as u32), v);
        }
        equal(copied.get(16), &value);
        equal(copied.get(17), &Value::Null);
        assert_eq!(original.len(), 16);
    }
}

#[test]
fn shared_mapped_string_append_preserves_relocated_values_and_mapping() {
    let dir = tempfile::tempdir().unwrap();
    let mut base = TypedColumn::for_value(&Value::String("a".into()));
    for _ in 0..16 {
        base.push(&Value::String("a".into())).unwrap();
    }
    base.set(3, &Value::String("relocated-long-λ".into()))
        .unwrap();
    base.materialize_to_file(dir.path(), "text").unwrap();
    assert!(base.is_mapped());
    let original = Arc::new(base);
    let mut copied = Arc::clone(&original);
    let incoming = Value::String("🙂 appended".into());
    TypedColumn::make_mut_for_append(&mut copied, &incoming)
        .push(&incoming)
        .unwrap();
    assert!(original.is_mapped());
    assert!(!copied.is_mapped());
    for i in 0..16 {
        let expected = Value::String(if i == 3 { "relocated-long-λ" } else { "a" }.into());
        equal(original.get(i), &expected);
        equal(copied.get(i), &expected);
    }
    equal(copied.get(16), &incoming);
    assert_eq!(original.len(), 16);
}

#[test]
fn append_fallback_preserves_type_mismatch_and_failed_push_state() {
    let mut base = TypedColumn::for_value(&Value::Int64(7));
    for _ in 0..16 {
        base.push(&Value::Int64(7)).unwrap();
    }
    assert!(base
        .try_clone_for_append(&Value::String("demote".into()))
        .is_none());
    let original = Arc::new(base);
    let mut copied = Arc::clone(&original);
    let col = TypedColumn::make_mut_for_append(&mut copied, &Value::Int64(8));
    fail_next(FailurePoint::HeapReserve);
    assert!(col.push(&Value::Int64(8)).is_err());
    assert_eq!(col.len(), 16);
    for i in 0..16 {
        equal(col.get(i), &Value::Int64(7));
        equal(original.get(i), &Value::Int64(7));
    }
    col.push(&Value::Int64(8)).unwrap();
    equal(col.get(16), &Value::Int64(8));
    let mixed = TypedColumn::Mixed {
        data: vec![Value::Int64(1), Value::String("x".into())],
    };
    assert!(mixed.try_clone_for_append(&Value::Null).is_none());
    let old = Arc::new(mixed);
    let mut copy = Arc::clone(&old);
    TypedColumn::make_mut_for_append(&mut copy, &Value::Null).push_null();
    equal(copy.get(0), &Value::Int64(1));
    equal(copy.get(1), &Value::String("x".into()));
    equal(copy.get(2), &Value::Null);
    assert_eq!(old.len(), 2);
}

#[test]
fn oversized_reserve_declines_without_touching_source() {
    let values = MmapOrVec::from_vec(vec![i64::MIN, 7]);
    assert!(values.try_clone_for_append(usize::MAX).is_none());
    assert_eq!(values.as_slice(), &[i64::MIN, 7]);
    let mut bytes = MmapBytes::new();
    bytes.extend(b"abc").unwrap();
    assert!(bytes.try_clone_for_append(usize::MAX).is_none());
    assert_eq!(bytes.slice(0, 3), b"abc");
}
