//! Unit tests for `column_store.rs`. Split out as a sibling file via
//! `#[path]` so the production file stays under the centralized 2500-line
//! source-quality cap.

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
fn packed_save_null_pads_partial_typed_columns_to_row_count() {
    let mut interner = StringInterner::new();
    let age = interner.get_or_intern("age");
    let schema = Arc::new(TypeSchema::from_keys(vec![age]));
    let meta = HashMap::from([("age".to_string(), "int64".to_string())]);
    let mut store = ColumnStore::new(schema.clone(), &meta, &interner);
    store.push_row(&[(age, Value::Int64(42))]);

    // Reproduce a partially materialized mutation overlay: the logical store
    // has a second row while this typed column still contains only its base row.
    store.row_count = 2;
    store.tombstones.push(false);

    let packed = store.write_packed(&interner).unwrap();
    let loaded = ColumnStore::load_packed(schema, &meta, &interner, &packed, 2, None).unwrap();
    assert_eq!(loaded.get(0, age), Some(Value::Int64(42)));
    assert_eq!(loaded.get(1, age), None);
}

fn decode_misaligned<T: PackedElement>(encoded: &[u8], len: usize) -> Vec<T> {
    let alignment = std::mem::align_of::<T>();
    assert!(
        alignment > 1,
        "fixture requires a type with nontrivial alignment"
    );
    let mut framed = vec![0xff; encoded.len() + alignment];
    let base = framed.as_ptr() as usize;
    let offset = (0..alignment)
        .find(|offset| !(base + offset).is_multiple_of(alignment))
        .expect("nontrivial alignment must have a misaligned offset");
    framed[offset..offset + encoded.len()].copy_from_slice(encoded);
    let bytes = &framed[offset..offset + encoded.len()];
    assert_ne!(
        bytes.as_ptr() as usize % std::mem::align_of::<T>(),
        0,
        "fixture must exercise a genuinely misaligned slice"
    );
    ColumnStore::load_typed_vec::<T>(bytes, len, None, "fixture", "bin")
        .unwrap()
        .to_vec()
}

#[test]
fn packed_primitives_decode_from_misaligned_little_endian_bytes() {
    assert_eq!(
        decode_misaligned::<u32>(&0x7856_3412_u32.to_le_bytes(), 1),
        vec![0x7856_3412]
    );
    assert_eq!(
        decode_misaligned::<u64>(&0xfedc_ba98_7654_3210_u64.to_le_bytes(), 1),
        vec![0xfedc_ba98_7654_3210]
    );
    assert_eq!(
        decode_misaligned::<i32>(&(-12_345_678_i32).to_le_bytes(), 1),
        vec![-12_345_678]
    );
    assert_eq!(
        decode_misaligned::<i64>(&(-1_234_567_890_123_i64).to_le_bytes(), 1),
        vec![-1_234_567_890_123]
    );
    assert_eq!(
        decode_misaligned::<f64>(&1234.5_f64.to_le_bytes(), 1),
        vec![1234.5]
    );
}

#[test]
fn packed_primitive_writer_uses_little_endian_wire_order() {
    let mut bytes = Vec::new();
    write_packed_values(&MmapOrVec::from_vec(vec![0x7856_3412_u32]), &mut bytes).unwrap();
    write_packed_values(
        &MmapOrVec::from_vec(vec![0xfedc_ba98_7654_3210_u64]),
        &mut bytes,
    )
    .unwrap();
    write_packed_values(&MmapOrVec::from_vec(vec![-12_345_678_i32]), &mut bytes).unwrap();
    write_packed_values(
        &MmapOrVec::from_vec(vec![-1_234_567_890_123_i64]),
        &mut bytes,
    )
    .unwrap();
    write_packed_values(&MmapOrVec::from_vec(vec![1234.5_f64]), &mut bytes).unwrap();

    let expected = [
        0x7856_3412_u32.to_le_bytes().as_slice(),
        0xfedc_ba98_7654_3210_u64.to_le_bytes().as_slice(),
        (-12_345_678_i32).to_le_bytes().as_slice(),
        (-1_234_567_890_123_i64).to_le_bytes().as_slice(),
        1234.5_f64.to_le_bytes().as_slice(),
    ]
    .concat();
    assert_eq!(bytes, expected);
}

#[test]
fn packed_primitive_decoder_rejects_partial_or_trailing_elements() {
    for invalid in [&[1, 2, 3][..], &[1, 2, 3, 4, 5][..]] {
        let err = ColumnStore::load_typed_vec::<u32>(invalid, 1, None, "bad", "u32").unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
    }
}

#[test]
fn packed_column_store_round_trips_every_typed_column() {
    let (schema, meta, interner) = make_schema_and_meta();
    let mut store = ColumnStore::new(Arc::clone(&schema), &meta, &interner);
    let date = NaiveDate::from_ymd_opt(2025, 7, 10).unwrap();
    let values = [
        (
            InternedKey::from_str("name"),
            Value::String("Ålesund".into()),
        ),
        (InternedKey::from_str("age"), Value::Int64(-42)),
        (InternedKey::from_str("salary"), Value::Float64(1234.5)),
        (InternedKey::from_str("active"), Value::Boolean(true)),
        (InternedKey::from_str("joined"), Value::DateTime(date)),
    ];
    store.push_row(&values);
    store.replace_id_column(TypedColumn::UniqueId {
        data: MmapOrVec::from_vec(vec![0xfedc_ba98]),
        nulls: MmapOrVec::from_vec(vec![0]),
    });
    store.push_title(&Value::String("Typed row".into()));

    let packed = store.write_packed(&interner).unwrap();
    let loaded = ColumnStore::load_packed(schema, &meta, &interner, &packed, 1, None).unwrap();

    for (key, expected) in values {
        assert_eq!(loaded.get(0, key), Some(expected));
    }
    assert_eq!(loaded.get_id(0), Some(Value::UniqueId(0xfedc_ba98)));
    assert_eq!(loaded.get_title(0), Some(Value::String("Typed row".into())));
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
fn packed_mixed_columns_round_trip_with_current_codec() {
    let mut interner = StringInterner::new();
    let key = interner.get_or_intern("payload");
    let schema = Arc::new(TypeSchema::from_keys(vec![key]));
    let mut store = ColumnStore::new_mixed(schema.clone());
    store.push_row(&[(key, Value::String("legacy-or-current".into()))]);

    let codec = crate::serde_codec::CURRENT_CODEC;
    let packed = store.write_packed_with_codec(&interner, codec).unwrap();
    let loaded = ColumnStore::load_packed_with_codec(
        schema,
        &HashMap::new(),
        &interner,
        &packed,
        1,
        None,
        codec,
    )
    .unwrap();
    assert_eq!(
        loaded.get(0, key),
        Some(Value::String("legacy-or-current".into()))
    );
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

/// The `row_properties` shape that shipped before the second dense pass was
/// removed: two passes over the schema, the second one filling blanks the first
/// left. Kept here as the equivalence reference — see
/// `row_properties_matches_forced_second_pass`.
fn row_properties_forced_second_pass(
    store: &ColumnStore,
    row_id: u32,
) -> Vec<(InternedKey, Value)> {
    if row_id >= store.row_count
        || store
            .tombstones
            .get(row_id as usize)
            .copied()
            .unwrap_or(false)
    {
        return Vec::new();
    }
    let mut result = Vec::new();
    let mut seen: std::collections::HashSet<InternedKey> = std::collections::HashSet::new();
    for (slot, ik) in store.schema.iter() {
        if let Some(val) = store.columns.get(slot as usize).and_then(|c| c.get(row_id)) {
            seen.insert(ik);
            result.push((ik, val));
        }
    }
    if let Some(ref ms) = store.mmap_store {
        for (ik, val) in ms.row_properties(row_id) {
            if !seen.contains(&ik) {
                result.push((ik, val));
            }
        }
        return result;
    }
    for (slot, ik) in store.schema.iter() {
        if seen.contains(&ik) {
            continue;
        }
        if let Some(val) = store.columns.get(slot as usize).and_then(|c| c.get(row_id)) {
            result.push((ik, val));
        }
    }
    result.extend(store.overflow_row_properties(row_id));
    result
}

/// `row_properties` drops the second dense pass on the non-mmap path (it could
/// never emit a row the first pass had not already emitted) and builds the
/// `seen` set only for the mmap merge. This pins the output against the old
/// two-pass shape over every row class: fully populated, partially null,
/// carrying a schema key that has no value on any row, tombstoned, out of
/// range, and with an overflow bag installed.
#[test]
fn row_properties_matches_forced_second_pass() {
    use crate::graph::storage::mapped::mmap_vec::{MmapBytes, MmapOrVec};

    let (schema, meta, interner) = make_schema_and_meta();
    let mut store = ColumnStore::new(schema, &meta, &interner);

    let name_key = InternedKey::from_str("name");
    let age_key = InternedKey::from_str("age");
    let salary_key = InternedKey::from_str("salary");
    // `joined` and `active` stay unset on every row: schema keys whose column
    // answers `None` for all of them — precisely the case the removed second
    // pass claimed to cover.
    store.push_row(&[
        (name_key, Value::String("Alice".into())),
        (age_key, Value::Int64(30)),
        (salary_key, Value::Float64(1.5)),
    ]);
    store.push_row(&[(name_key, Value::String("Bob".into()))]);
    store.push_row(&[(age_key, Value::Int64(41))]);
    // A key added after the fact extends the schema and backfills nulls.
    let nickname_key = InternedKey::from_str("nickname");
    assert!(store.set(
        1,
        nickname_key,
        &Value::String("Bobby".into()),
        Some("string")
    ));
    store.tombstone(2);

    // Overflow bag: row 0 carries a sparse property, row 1 none.
    let mut data = Vec::new();
    let mut offsets: Vec<u64> = vec![0];
    let mut blob = 1u16.to_le_bytes().to_vec();
    crate::graph::storage::overflow::encode_value(
        &mut blob,
        InternedKey::from_str("sparse"),
        &Value::Int64(9),
    );
    data.extend_from_slice(&blob);
    offsets.push(data.len() as u64);
    for _ in 1..store.row_count {
        offsets.push(data.len() as u64);
    }
    let mut data_bytes = MmapBytes::new();
    data_bytes.extend(&data).expect("overflow data");
    store.replace_overflow_bag(MmapOrVec::from_vec(offsets), data_bytes);

    let schema_len = store.schema.iter().count();
    let row0 = store.row_properties(0);
    assert!(
        row0.len() < schema_len,
        "fixture must leave at least one schema key unset on row 0 \
         (schema {schema_len}, row {row0:?})"
    );
    assert!(
        row0.iter()
            .any(|(k, _)| *k == InternedKey::from_str("sparse")),
        "fixture must exercise the overflow bag: {row0:?}"
    );

    for row_id in 0..=store.row_count {
        assert_eq!(
            store.row_properties(row_id),
            row_properties_forced_second_pass(&store, row_id),
            "row {row_id} diverged from the two-pass reference"
        );
    }
}

#[test]
fn test_overflow_value_list_roundtrip() {
    // Native list properties must survive the overflow-bag wire format
    // (tag 9 = u32 length prefix + Postcard(Vec<Value>)). Before the
    // 0.11.x fix this serialized as null and the list was silently lost.
    let mut interner = StringInterner::new();
    let key = interner.get_or_intern("aliases");

    let list = Value::List(vec![
        Value::String("x".into()),
        Value::String("y".into()),
        Value::Int64(7),
    ]);

    let mut buf = 1u16.to_le_bytes().to_vec(); // entry-count header
    crate::graph::storage::overflow::encode_value(&mut buf, key, &list);

    let decoded = crate::graph::storage::overflow::decode_blob(&buf);
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
        crate::graph::storage::overflow::encode_value(&mut buf, *k, v);
    }

    let decoded = crate::graph::storage::overflow::decode_blob(&buf);
    assert_eq!(decoded.len(), 3);
    assert_eq!(decoded[0], (k_a, Value::Int64(1)));
    assert_eq!(
        decoded[1],
        (k_list, Value::List(vec![Value::String("red".into())]))
    );
    assert_eq!(decoded[2], (k_b, Value::Boolean(true)));
}

fn one_packed_column(name: &str, tag: &str, blob: &[u8]) -> Vec<u8> {
    let mut packed = 1u32.to_le_bytes().to_vec();
    packed.extend_from_slice(&(name.len() as u16).to_le_bytes());
    packed.extend_from_slice(name.as_bytes());
    packed.extend_from_slice(&(tag.len() as u16).to_le_bytes());
    packed.extend_from_slice(tag.as_bytes());
    packed.extend_from_slice(&(blob.len() as u64).to_le_bytes());
    packed.extend_from_slice(blob);
    packed
}

fn load_single_column(name: &str, tag: &str, blob: &[u8], rows: u32) -> io::Result<ColumnStore> {
    let mut interner = StringInterner::new();
    let key = interner.get_or_intern(name);
    let schema = Arc::new(TypeSchema::from_keys(vec![key]));
    let meta = HashMap::from([(name.to_string(), tag.to_string())]);
    ColumnStore::load_packed(
        schema,
        &meta,
        &interner,
        &one_packed_column(name, tag, blob),
        rows,
        None,
    )
}

#[test]
fn packed_strings_reject_invalid_offsets_and_utf8() {
    let mut non_monotonic = Vec::new();
    for offset in [0u64, 2, 1] {
        non_monotonic.extend_from_slice(&offset.to_le_bytes());
    }
    non_monotonic.extend_from_slice(b"x");
    non_monotonic.extend_from_slice(&[0, 0]);
    let err = load_single_column("name", "string", &non_monotonic, 2).unwrap_err();
    assert_eq!(err.kind(), io::ErrorKind::InvalidData);

    let mut invalid_utf8 = Vec::new();
    for offset in [0u64, 1] {
        invalid_utf8.extend_from_slice(&offset.to_le_bytes());
    }
    invalid_utf8.extend_from_slice(&[0xff, 0]);
    let err = load_single_column("name", "string", &invalid_utf8, 1).unwrap_err();
    assert_eq!(err.kind(), io::ErrorKind::InvalidData);
}

#[test]
fn packed_column_names_cannot_escape_temp_root() {
    let root = tempfile::tempdir().unwrap();
    let load_root = root.path().join("load");
    std::fs::create_dir(&load_root).unwrap();
    let rows = (MMAP_THRESHOLD / 8 + 1) as u32;
    let mut blob = vec![0u8; rows as usize * 8];
    blob.extend(std::iter::repeat_n(0, rows as usize));

    let name = "../../escaped";
    let mut interner = StringInterner::new();
    let key = interner.get_or_intern(name);
    let schema = Arc::new(TypeSchema::from_keys(vec![key]));
    let meta = HashMap::from([(name.to_string(), "int64".to_string())]);
    ColumnStore::load_packed(
        schema,
        &meta,
        &interner,
        &one_packed_column(name, "int64", &blob),
        rows,
        Some(&load_root),
    )
    .unwrap();

    assert!(!root.path().join("escaped.i64").exists());
    for entry in std::fs::read_dir(&load_root).unwrap() {
        let file_name = entry.unwrap().file_name().to_string_lossy().into_owned();
        assert!(
            file_name.starts_with("column_"),
            "unexpected temp file {file_name}"
        );
    }
}
