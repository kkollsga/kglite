//! Unit tests for `column_store.rs`. Split out as a sibling file via
//! `#[path]` so the production file stays under the 2500-line god-file
//! gate (test_phase7_parity.py::test_god_file_gate).

#![allow(clippy::approx_constant)]

use super::*;

fn make_schema_and_meta() -> (Arc<TypeSchema>, HashMap<String, String>, StringInterner) {
    let mut interner = StringInterner::new();
    let keys = ["name", "age", "salary", "active", "joined"];
    let interned: Vec<InternedKey> = keys.iter().map(|k| interner.get_or_intern(k)).collect();

    let schema = Arc::new(TypeSchema::from_keys(interned));

    let mut meta = HashMap::new();
    meta.insert("name".to_string(), "string".to_string());
    meta.insert("age".to_string(), "int64".to_string());
    meta.insert("salary".to_string(), "float64".to_string());
    meta.insert("active".to_string(), "bool".to_string());
    meta.insert("joined".to_string(), "date".to_string());

    (schema, meta, interner)
}

#[test]
fn test_typed_column_int64_roundtrip() {
    let mut col = TypedColumn::from_type_str("int64");
    assert!(col.push(&Value::Int64(42)).is_ok());
    assert!(col.push(&Value::Int64(-7)).is_ok());
    assert!(col.push(&Value::Null).is_ok());

    assert_eq!(col.get(0), Some(Value::Int64(42)));
    assert_eq!(col.get(1), Some(Value::Int64(-7)));
    assert_eq!(col.get(2), None); // null
    assert_eq!(col.get(3), None); // out of bounds
    assert_eq!(col.len(), 3);
}

#[test]
fn test_typed_column_float64_with_int_promotion() {
    let mut col = TypedColumn::from_type_str("float64");
    assert!(col.push(&Value::Float64(3.14)).is_ok());
    assert!(col.push(&Value::Int64(42)).is_ok()); // int→float promotion
    assert_eq!(col.get(0), Some(Value::Float64(3.14)));
    assert_eq!(col.get(1), Some(Value::Float64(42.0)));
}

#[test]
fn test_typed_column_string_roundtrip() {
    let mut col = TypedColumn::from_type_str("string");
    assert!(col.push(&Value::String("hello".into())).is_ok());
    assert!(col.push(&Value::String("world".into())).is_ok());
    assert!(col.push(&Value::Null).is_ok());
    assert!(col.push(&Value::String("".into())).is_ok());

    assert_eq!(col.get(0), Some(Value::String("hello".into())));
    assert_eq!(col.get(1), Some(Value::String("world".into())));
    assert_eq!(col.get(2), None);
    assert_eq!(col.get(3), Some(Value::String("".into())));
    assert_eq!(col.len(), 4);
}

#[test]
fn test_typed_column_bool_roundtrip() {
    let mut col = TypedColumn::from_type_str("bool");
    assert!(col.push(&Value::Boolean(true)).is_ok());
    assert!(col.push(&Value::Boolean(false)).is_ok());
    assert_eq!(col.get(0), Some(Value::Boolean(true)));
    assert_eq!(col.get(1), Some(Value::Boolean(false)));
}

#[test]
fn test_typed_column_date_roundtrip() {
    let mut col = TypedColumn::from_type_str("date");
    let d = NaiveDate::from_ymd_opt(2024, 3, 15).unwrap();
    assert!(col.push(&Value::DateTime(d)).is_ok());
    assert!(col.push(&Value::Null).is_ok());
    assert_eq!(col.get(0), Some(Value::DateTime(d)));
    assert_eq!(col.get(1), None);
}

#[test]
fn test_typed_column_uniqueid_roundtrip() {
    let mut col = TypedColumn::from_type_str("uniqueid");
    assert!(col.push(&Value::UniqueId(100)).is_ok());
    assert_eq!(col.get(0), Some(Value::UniqueId(100)));
}

#[test]
fn test_typed_column_mixed_fallback() {
    let mut col = TypedColumn::from_type_str("mixed");
    assert!(col.push(&Value::Int64(1)).is_ok());
    assert!(col.push(&Value::String("hello".into())).is_ok());
    assert!(col.push(&Value::Boolean(true)).is_ok());
    assert_eq!(col.get(0), Some(Value::Int64(1)));
    assert_eq!(col.get(1), Some(Value::String("hello".into())));
    assert_eq!(col.get(2), Some(Value::Boolean(true)));
}

#[test]
fn test_typed_column_type_mismatch_rejected() {
    let mut col = TypedColumn::from_type_str("int64");
    assert!(col.push(&Value::String("oops".into())).is_err());
}

#[test]
fn test_column_store_basic_roundtrip() {
    let (schema, meta, interner) = make_schema_and_meta();
    let mut store = ColumnStore::new(schema, &meta, &interner);

    let name_key = InternedKey::from_str("name");
    let age_key = InternedKey::from_str("age");
    let salary_key = InternedKey::from_str("salary");

    let row0 = store.push_row(&[
        (name_key, Value::String("Alice".into())),
        (age_key, Value::Int64(30)),
        (salary_key, Value::Float64(75000.0)),
    ]);
    assert_eq!(row0, 0);

    let row1 = store.push_row(&[
        (name_key, Value::String("Bob".into())),
        (age_key, Value::Int64(25)),
    ]);
    assert_eq!(row1, 1);

    assert_eq!(store.get(0, name_key), Some(Value::String("Alice".into())));
    assert_eq!(store.get(0, age_key), Some(Value::Int64(30)));
    assert_eq!(store.get(0, salary_key), Some(Value::Float64(75000.0)));

    assert_eq!(store.get(1, name_key), Some(Value::String("Bob".into())));
    assert_eq!(store.get(1, age_key), Some(Value::Int64(25)));
    assert_eq!(store.get(1, salary_key), None); // null

    assert_eq!(store.row_count(), 2);
    assert_eq!(store.live_count(), 2);
}

#[test]
fn test_column_store_property_update() {
    let (schema, meta, interner) = make_schema_and_meta();
    let mut store = ColumnStore::new(schema, &meta, &interner);

    let name_key = InternedKey::from_str("name");
    let age_key = InternedKey::from_str("age");

    store.push_row(&[
        (name_key, Value::String("Alice".into())),
        (age_key, Value::Int64(30)),
    ]);

    // Update age
    assert!(store.set(0, age_key, &Value::Int64(31), None));
    assert_eq!(store.get(0, age_key), Some(Value::Int64(31)));

    // Update name
    assert!(store.set(0, name_key, &Value::String("Alicia".into()), None));
    assert_eq!(store.get(0, name_key), Some(Value::String("Alicia".into())));
}

#[test]
fn test_column_store_schema_extension() {
    let (schema, meta, interner) = make_schema_and_meta();
    let mut store = ColumnStore::new(schema, &meta, &interner);

    let name_key = InternedKey::from_str("name");
    let new_key = InternedKey::from_str("email");

    store.push_row(&[(name_key, Value::String("Alice".into()))]);

    // Set a property that doesn't exist in the schema yet
    assert!(store.set(
        0,
        new_key,
        &Value::String("alice@example.com".into()),
        Some("string")
    ));
    assert_eq!(
        store.get(0, new_key),
        Some(Value::String("alice@example.com".into()))
    );
}

#[test]
fn test_column_store_tombstone() {
    let (schema, meta, interner) = make_schema_and_meta();
    let mut store = ColumnStore::new(schema, &meta, &interner);

    let name_key = InternedKey::from_str("name");
    store.push_row(&[(name_key, Value::String("Alice".into()))]);
    store.push_row(&[(name_key, Value::String("Bob".into()))]);

    store.tombstone(0);
    assert_eq!(store.get(0, name_key), None);
    assert_eq!(store.get(1, name_key), Some(Value::String("Bob".into())));
    assert_eq!(store.live_count(), 1);
}

#[test]
fn test_column_store_row_properties() {
    let (schema, meta, interner) = make_schema_and_meta();
    let mut store = ColumnStore::new(schema, &meta, &interner);

    let name_key = InternedKey::from_str("name");
    let age_key = InternedKey::from_str("age");

    store.push_row(&[
        (name_key, Value::String("Alice".into())),
        (age_key, Value::Int64(30)),
    ]);

    let props = store.row_properties(0);
    assert_eq!(props.len(), 2);

    let map = store.row_properties_map(0, &interner);
    assert_eq!(map.get("name"), Some(&Value::String("Alice".into())));
    assert_eq!(map.get("age"), Some(&Value::Int64(30)));
}

#[test]
fn test_column_store_demote_to_mixed() {
    let (schema, meta, interner) = make_schema_and_meta();
    let mut store = ColumnStore::new(schema, &meta, &interner);

    let age_key = InternedKey::from_str("age");

    // Push an int64 row
    store.push_row(&[(age_key, Value::Int64(30))]);

    // Now try to set a string into an int64 column — should demote to Mixed
    assert!(store.set(0, age_key, &Value::String("thirty".into()), None));
    assert_eq!(store.get(0, age_key), Some(Value::String("thirty".into())));
}

#[test]
fn test_column_store_new_mixed() {
    let mut interner = StringInterner::new();
    let keys = vec![interner.get_or_intern("a"), interner.get_or_intern("b")];
    let schema = Arc::new(TypeSchema::from_keys(keys));
    let mut store = ColumnStore::new_mixed(schema);

    let a_key = InternedKey::from_str("a");
    let b_key = InternedKey::from_str("b");

    store.push_row(&[
        (a_key, Value::Int64(1)),
        (b_key, Value::String("hello".into())),
    ]);

    assert_eq!(store.get(0, a_key), Some(Value::Int64(1)));
    assert_eq!(store.get(0, b_key), Some(Value::String("hello".into())));
}

#[test]
fn test_column_store_materialize_roundtrip() {
    let dir = tempfile::TempDir::new().unwrap();
    let (schema, meta, interner) = make_schema_and_meta();
    let mut store = ColumnStore::new(schema, &meta, &interner);

    let name_key = InternedKey::from_str("name");
    let age_key = InternedKey::from_str("age");
    let salary_key = InternedKey::from_str("salary");
    let active_key = InternedKey::from_str("active");

    store.push_row(&[
        (name_key, Value::String("Alice".into())),
        (age_key, Value::Int64(30)),
        (salary_key, Value::Float64(75000.0)),
        (active_key, Value::Boolean(true)),
    ]);
    store.push_row(&[
        (name_key, Value::String("Bob".into())),
        (age_key, Value::Int64(25)),
        (salary_key, Value::Float64(50000.0)),
        (active_key, Value::Boolean(false)),
    ]);

    // Materialize to files
    store.materialize_to_files(dir.path(), &interner).unwrap();
    assert!(store.is_mapped());

    // Verify data still accessible
    assert_eq!(store.get(0, name_key), Some(Value::String("Alice".into())));
    assert_eq!(store.get(0, age_key), Some(Value::Int64(30)));
    assert_eq!(store.get(1, salary_key), Some(Value::Float64(50000.0)));
    assert_eq!(store.get(1, active_key), Some(Value::Boolean(false)));

    // Convert back to heap
    store.materialize_to_heap();
    assert!(!store.is_mapped());
    assert_eq!(store.get(0, name_key), Some(Value::String("Alice".into())));
    assert_eq!(store.get(1, age_key), Some(Value::Int64(25)));
}

#[test]
fn test_overflow_value_list_roundtrip() {
    // Native list properties must survive the overflow-bag wire format
    // (tag 8 = u32 length prefix + bincode(Vec<Value>)). Before the
    // 0.11.x fix this serialized as null and the list was silently lost.
    let mut interner = StringInterner::new();
    let key = interner.get_or_intern("aliases");

    let list = Value::List(vec![
        Value::String("x".into()),
        Value::String("y".into()),
        Value::Int64(7),
    ]);

    let mut buf = 1u16.to_le_bytes().to_vec(); // entry-count header
    ColumnStore::serialize_overflow_value(&mut buf, key, &list);

    let decoded = ColumnStore::decode_overflow_blob(&buf);
    assert_eq!(decoded.len(), 1);
    assert_eq!(decoded[0].0, key);
    assert_eq!(decoded[0].1, list);
}

#[test]
fn test_overflow_blob_mixed_scalars_and_list_roundtrip() {
    // A list entry sandwiched between scalars must not desync the reader
    // (skip_overflow_value has to advance past the tag-8 blob correctly).
    let mut interner = StringInterner::new();
    let k_a = interner.get_or_intern("a");
    let k_list = interner.get_or_intern("tags");
    let k_b = interner.get_or_intern("b");

    let entries = [
        (k_a, Value::Int64(1)),
        (k_list, Value::List(vec![Value::String("red".into())])),
        (k_b, Value::Boolean(true)),
    ];
    let mut buf = (entries.len() as u16).to_le_bytes().to_vec(); // entry-count header
    for (k, v) in &entries {
        ColumnStore::serialize_overflow_value(&mut buf, *k, v);
    }

    let decoded = ColumnStore::decode_overflow_blob(&buf);
    assert_eq!(decoded.len(), 3);
    assert_eq!(decoded[0], (k_a, Value::Int64(1)));
    assert_eq!(
        decoded[1],
        (k_list, Value::List(vec![Value::String("red".into())]))
    );
    assert_eq!(decoded[2], (k_b, Value::Boolean(true)));
}
