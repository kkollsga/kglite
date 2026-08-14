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
    let packed = store
        .write_packed_with_codec(
            &interner,
            codec,
            crate::graph::storage::packed_codec::IntColumnEncoding::Raw,
        )
        .unwrap();
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

// ── Mapped columns stay mapped across writes (Phase 4 contract) ─────────────
//
// `set_memory_limit` is the only bound a caller can place on the columnar
// heap. A spilled column that comes back to the heap the first time it is
// written removes that bound silently — the limit is not re-enforced anywhere
// after a statement, so the heap only ever grows. `MmapOrVec::set` writes
// through `map_mut` into the backing file, and every writable mapped column
// lives in a process-owned spill/temp directory that is removed on drop
// (never a user's `.kgl`), so the write belongs in the file.

/// A `set` on a file-backed column writes through the mapping instead of
/// pulling the column onto the heap, and the new byte reaches the file.
#[test]
fn mapped_column_set_writes_through_to_the_file() {
    let dir = tempfile::tempdir().unwrap();
    let (schema, meta, interner) = make_schema_and_meta();
    let mut store = ColumnStore::new(schema, &meta, &interner);

    let age_key = InternedKey::from_str("age");
    let name_key = InternedKey::from_str("name");
    for i in 0..128i64 {
        store.push_row(&[
            (name_key, Value::String(format!("n{i}"))),
            (age_key, Value::Int64(i)),
        ]);
    }
    store.materialize_to_files(dir.path(), &interner).unwrap();
    assert!(store.is_mapped(), "precondition: the fixture must spill");
    let mapped_heap = store.heap_bytes();

    assert!(store.set(7, age_key, &Value::Int64(4242), None));

    assert_eq!(store.get(7, age_key), Some(Value::Int64(4242)));
    assert!(
        store.is_mapped(),
        "a single-cell SET un-mapped the store: set_memory_limit's bound is gone"
    );
    assert_eq!(
        store.heap_bytes(),
        mapped_heap,
        "the touched column was copied onto the heap instead of written through"
    );

    // The write is in the file, not only in a private overlay: read the raw
    // i64 image back off disk.
    let bytes = std::fs::read(store.spill_subdir(dir.path()).join("age.i64")).unwrap();
    let cell = i64::from_le_bytes(bytes[7 * 8..8 * 8].try_into().unwrap());
    assert_eq!(cell, 4242, "the mapped write never reached the spill file");
}

/// Restoring a cell pre-image (what statement rollback does) travels the same
/// route as the write it undoes, so a rollback on a spilled graph is
/// symmetric and does not un-map it either.
#[test]
fn mapped_column_cell_restore_is_symmetric() {
    let dir = tempfile::tempdir().unwrap();
    let (schema, meta, interner) = make_schema_and_meta();
    let mut store = ColumnStore::new(schema, &meta, &interner);

    let age_key = InternedKey::from_str("age");
    let name_key = InternedKey::from_str("name");
    for i in 0..128i64 {
        store.push_row(&[
            (name_key, Value::String(format!("n{i}"))),
            (age_key, Value::Int64(i)),
        ]);
    }
    store.materialize_to_files(dir.path(), &interner).unwrap();
    let mapped_heap = store.heap_bytes();
    let prior = store.get(9, age_key);

    store.set(9, age_key, &Value::Int64(-1), None);
    // Undo: the journal replays the pre-image through the same `set`.
    store.set(9, age_key, prior.as_ref().unwrap_or(&Value::Null), None);

    assert_eq!(store.get(9, age_key), Some(Value::Int64(9)));
    assert!(store.is_mapped(), "rollback un-mapped the store");
    assert_eq!(store.heap_bytes(), mapped_heap);
}

/// String columns park updates in the `relocated` overlay rather than
/// rewriting `offsets`, so they must also keep their mapping — the overlay is
/// the bounded per-changed-cell residue, not a whole-column copy.
#[test]
fn mapped_str_column_set_keeps_its_mapping() {
    let dir = tempfile::tempdir().unwrap();
    let (schema, meta, interner) = make_schema_and_meta();
    let mut store = ColumnStore::new(schema, &meta, &interner);

    let name_key = InternedKey::from_str("name");
    for i in 0..128i64 {
        store.push_row(&[(name_key, Value::String(format!("name-{i}")))]);
    }
    store.materialize_to_files(dir.path(), &interner).unwrap();
    assert!(store.is_mapped());
    let mapped_heap = store.heap_bytes();

    store.set(3, name_key, &Value::String("rewritten".into()), None);

    assert_eq!(
        store.get(3, name_key),
        Some(Value::String("rewritten".into()))
    );
    assert!(store.is_mapped());
    assert!(
        store.heap_bytes() <= mapped_heap + 64,
        "a one-cell string SET grew the heap by {} bytes",
        store.heap_bytes() - mapped_heap
    );
}

/// A held reader sharing the store's `Arc` is isolated from a mapped
/// write-through: `Arc::make_mut` hands the *writer* a fresh clone, and
/// `MmapOrVec::clone` always yields a `Heap` buffer, so the writer never
/// touches the bytes the reader is mapping. Without that, an in-place mapped
/// write would mutate a page a frozen graph or a held result view is still
/// iterating.
///
/// The residue this pins in passing: it is the *writer* that leaves the
/// mapping behind, not the reader. A first write under a held view therefore
/// un-maps the master and moves it onto the heap — the pre-existing
/// fork-copy cost (D2), unchanged by the write-through, and the one shape in
/// which `set_memory_limit` still loses its bound.
#[test]
fn a_held_reader_is_isolated_from_a_mapped_write_through() {
    let dir = tempfile::tempdir().unwrap();
    let (schema, meta, interner) = make_schema_and_meta();
    let mut store = ColumnStore::new(schema, &meta, &interner);

    let age_key = InternedKey::from_str("age");
    for i in 0..128i64 {
        store.push_row(&[(age_key, Value::Int64(i))]);
    }
    store.materialize_to_files(dir.path(), &interner).unwrap();

    let mut master = Arc::new(store);
    let held = Arc::clone(&master);

    Arc::make_mut(&mut master).set(5, age_key, &Value::Int64(999), None);

    assert_eq!(master.get(5, age_key), Some(Value::Int64(999)));
    assert_eq!(
        held.get(5, age_key),
        Some(Value::Int64(5)),
        "the mapped write reached a held reader's bytes"
    );
    assert!(!Arc::ptr_eq(&master, &held));
    assert!(
        held.is_mapped(),
        "the reader kept the mapping; the writer took the heap copy"
    );
    assert!(
        !master.is_mapped(),
        "the fork copy is heap-backed — this is the D2 fork cost, pinned here \
         so a future change that shares the mapping across the fork has to \
         come past this assertion"
    );
}

// ── Schema growth without a rebuild (Phase 5(i)) ─────────────────────────────
//
// `push_row` used to build its slot→value lookup from `self.schema.slot(key)`
// and drop every key the lookup missed — silently, with no error and no
// counter. Its callers compensated by *rebuilding the whole store* whenever the
// registered type schema had grown: `ensure_column_store_for_push` and the
// mapped arm of `BatchProcessor::flush_chunk` both re-pushed every existing row
// into a fresh store, O(rows x cols) per newly-seen key. A stream that widens
// its key set over time therefore paid quadratically for the privilege, and a
// caller that pushed a key without registering it first simply lost the data.

#[test]
fn push_row_appends_a_column_for_a_key_the_schema_has_never_seen() {
    let (schema, meta, interner) = make_schema_and_meta();
    let mut store = ColumnStore::new(schema, &meta, &interner);

    let name_key = InternedKey::from_str("name");
    let fresh_key = InternedKey::from_str("nickname");
    let width = store.column_count();

    store.push_row(&[(name_key, Value::String("Alice".into()))]);
    store.push_row(&[
        (name_key, Value::String("Bob".into())),
        (fresh_key, Value::String("Bobby".into())),
    ]);

    // (a) no data loss — the value the schema had no slot for survives.
    assert_eq!(
        store.get(1, fresh_key),
        Some(Value::String("Bobby".into())),
        "push_row dropped a value whose key was not in the store's schema"
    );
    // The row that predates the column reads as absent, not as garbage.
    assert_eq!(store.get(0, fresh_key), None);
    assert_eq!(store.get(0, name_key), Some(Value::String("Alice".into())));
    assert_eq!(store.get(1, name_key), Some(Value::String("Bob".into())));

    // Exactly one column was appended, and the pre-existing slots still name
    // the same columns (the precondition `restore_schema` undoes by truncating).
    assert_eq!(store.column_count(), width + 1);
    assert_eq!(store.slot(name_key), Some(0));
    assert_eq!(store.slot(fresh_key), Some(width as u16));

    // The appended column is typed from the value in hand, not `Mixed`.
    assert_eq!(store.column_type_str(width), Some("string"));
}

#[test]
fn push_row_types_an_appended_column_from_the_value_in_hand() {
    let (schema, meta, interner) = make_schema_and_meta();
    let mut store = ColumnStore::new(schema, &meta, &interner);
    let width = store.column_count();

    store.push_row(&[(InternedKey::from_str("score"), Value::Int64(7))]);
    store.push_row(&[(InternedKey::from_str("ratio"), Value::Float64(0.5))]);
    store.push_row(&[(InternedKey::from_str("flag"), Value::Boolean(true))]);
    // No type evidence — `Mixed` is the honest answer, not a guess.
    store.push_row(&[(InternedKey::from_str("blank"), Value::Null)]);

    assert_eq!(store.column_type_str(width), Some("int64"));
    assert_eq!(store.column_type_str(width + 1), Some("float64"));
    assert_eq!(store.column_type_str(width + 2), Some("bool"));
    assert_eq!(store.column_type_str(width + 3), Some("mixed"));
}

#[test]
fn growing_the_key_set_over_a_stream_pushes_each_row_exactly_once() {
    // The amortized-O(1) contract, measured rather than argued. Before the
    // append path this test read 1 + 2 + ... + rows-per-key pushes, because
    // every newly-seen key re-pushed the whole store.
    let mut interner = StringInterner::new();
    let base = interner.get_or_intern("base");
    let schema = Arc::new(TypeSchema::from_keys(vec![base]));
    let mut store = ColumnStore::new(schema, &HashMap::new(), &interner);

    const ROWS: usize = 200;
    let keys: Vec<InternedKey> = (0..ROWS)
        .map(|i| interner.get_or_intern(&format!("k{i}")))
        .collect();

    crate::graph::storage::column_store::reset_column_store_row_pushes();
    for (i, key) in keys.iter().enumerate() {
        store.push_row(&[
            (base, Value::Int64(i as i64)),
            (*key, Value::Int64(i as i64)),
        ]);
    }
    assert_eq!(
        crate::graph::storage::column_store::column_store_row_pushes(),
        ROWS,
        "a widening key set re-pushed existing rows: the store is being rebuilt \
         per new key instead of appending one column"
    );

    // ... and every row still reads back exactly what it carried.
    for (i, key) in keys.iter().enumerate() {
        assert_eq!(store.get(i as u32, base), Some(Value::Int64(i as i64)));
        assert_eq!(store.get(i as u32, *key), Some(Value::Int64(i as i64)));
        if i > 0 {
            assert_eq!(store.get(i as u32 - 1, *key), None);
        }
    }
}

// ── Typed columns at creation (Phase 5(ii)) ──────────────────────────────────

#[test]
fn set_types_a_new_column_from_the_value_when_metadata_is_silent() {
    let (schema, meta, interner) = make_schema_and_meta();
    let mut store = ColumnStore::new(schema, &meta, &interner);
    let width = store.column_count();

    store.push_row(&[(InternedKey::from_str("name"), Value::String("A".into()))]);
    // Every production write site passes `type_meta: None` today; the column
    // must still come out typed, because `Mixed` cannot be spilled.
    assert!(store.set(0, InternedKey::from_str("hits"), &Value::Int64(3), None));

    assert_eq!(store.column_type_str(width), Some("int64"));
    assert_eq!(
        store.get(0, InternedKey::from_str("hits")),
        Some(Value::Int64(3))
    );
}

#[test]
fn set_prefers_declared_metadata_over_the_value_in_hand() {
    // A `float64` property whose first written value happens to be an integer
    // must not create an `Int64` column that the next 0.5 demotes to `Mixed`.
    let (schema, meta, interner) = make_schema_and_meta();
    let mut store = ColumnStore::new(schema, &meta, &interner);
    let width = store.column_count();

    store.push_row(&[(InternedKey::from_str("name"), Value::String("A".into()))]);
    let key = InternedKey::from_str("rate");
    assert!(store.set(0, key, &Value::Int64(1), Some("float64")));
    assert_eq!(store.column_type_str(width), Some("float64"));

    assert!(store.set(0, key, &Value::Float64(0.5), None));
    assert_eq!(store.column_type_str(width), Some("float64"));
    assert_eq!(store.get(0, key), Some(Value::Float64(0.5)));
}

#[test]
fn a_type_mismatched_set_still_demotes_to_mixed() {
    // Correctness over memory: typing the column at creation must not turn a
    // heterogeneous property into a lost or coerced write.
    let (schema, meta, interner) = make_schema_and_meta();
    let mut store = ColumnStore::new(schema, &meta, &interner);
    let width = store.column_count();

    store.push_row(&[(InternedKey::from_str("name"), Value::String("A".into()))]);
    store.push_row(&[(InternedKey::from_str("name"), Value::String("B".into()))]);

    let key = InternedKey::from_str("mixedish");
    assert!(store.set(0, key, &Value::Int64(1), None));
    assert_eq!(store.column_type_str(width), Some("int64"));
    assert!(store.set(1, key, &Value::String("two".into()), None));
    assert_eq!(store.column_type_str(width), Some("mixed"));

    assert_eq!(store.get(0, key), Some(Value::Int64(1)));
    assert_eq!(store.get(1, key), Some(Value::String("two".into())));
}

#[test]
fn the_id_column_is_typed_from_the_first_id_pushed() {
    let (schema, meta, interner) = make_schema_and_meta();

    let mut ints = ColumnStore::new(Arc::clone(&schema), &meta, &interner);
    for i in 0..4i64 {
        ints.push_id(&Value::Int64(i));
        ints.push_row(&[]);
    }
    assert_eq!(
        ints.id_type_str(),
        Some("int64"),
        "the id column is still born Mixed — 32 B/row that cannot be spilled"
    );
    for i in 0..4i64 {
        assert_eq!(ints.get_id(i as u32), Some(Value::Int64(i)));
    }

    let mut strs = ColumnStore::new(Arc::clone(&schema), &meta, &interner);
    strs.push_id(&Value::String("Q42".into()));
    strs.push_row(&[]);
    assert_eq!(strs.id_type_str(), Some("string"));
    assert_eq!(strs.get_id(0), Some(Value::String("Q42".into())));

    // Heterogeneous ids still land in `Mixed`, without losing a value.
    let mut mixed = ColumnStore::new(schema, &meta, &interner);
    mixed.push_id(&Value::Int64(1));
    mixed.push_row(&[]);
    mixed.push_id(&Value::String("Q7".into()));
    mixed.push_row(&[]);
    assert_eq!(mixed.id_type_str(), Some("mixed"));
    assert_eq!(mixed.get_id(0), Some(Value::Int64(1)));
    assert_eq!(mixed.get_id(1), Some(Value::String("Q7".into())));
}

#[test]
fn a_typed_id_column_spills_to_a_file() {
    // The point of typing it: `materialize_to_file` is a no-op for `Mixed`, so
    // a Mixed id column is a permanent heap floor that `set_memory_limit`
    // cannot touch.
    let dir = tempfile::tempdir().expect("tempdir");
    let (schema, meta, interner) = make_schema_and_meta();
    let mut store = ColumnStore::new(schema, &meta, &interner);
    for i in 0..64i64 {
        store.push_id(&Value::Int64(i));
        store.push_row(&[(InternedKey::from_str("age"), Value::Int64(i))]);
    }
    let before = store.heap_bytes();
    store
        .materialize_to_files(dir.path(), &interner)
        .expect("materialize");

    assert!(
        store.spill_subdir(dir.path()).join("__id__.i64").exists(),
        "no __id__ file written"
    );
    assert!(
        store.heap_bytes() < before,
        "spilling reclaimed nothing: {before} -> {}",
        store.heap_bytes()
    );
    for i in 0..64i64 {
        assert_eq!(store.get_id(i as u32), Some(Value::Int64(i)));
    }
}

/// Two stores that came from one `clone()` must not spill into the same files.
///
/// A graph's spill directory is copied by every clone of it — `copy()`, a
/// transaction fork, a held view — so a per-*type* spill path puts two live
/// stores on the same bytes. Because a spilled column is read back through its
/// file mapping, the loser then reads the winner's values: a copy's write
/// showing up in the original, which is the failure
/// `test_phase5_parity.py::test_graph_copy_cow_correctness_mapped` sees from
/// the outside. Asserted here on the primitive, where the mechanism is.
#[test]
fn a_cloned_store_spills_to_its_own_files() {
    let dir = tempfile::tempdir().unwrap();
    let mut interner = StringInterner::new();
    let age = interner.get_or_intern("age");
    let schema = Arc::new(TypeSchema::from_keys(vec![age]));
    let meta = HashMap::from([("age".to_string(), "int64".to_string())]);

    let mut original = ColumnStore::new(schema, &meta, &interner);
    for i in 0..64i64 {
        original.push_row(&[(age, Value::Int64(i))]);
    }
    let mut copy = original.clone();

    original
        .materialize_to_files(dir.path(), &interner)
        .unwrap();
    // The copy diverges *after* the clone and spills to the same root.
    copy.set(7, age, &Value::Int64(4242), None);
    copy.materialize_to_files(dir.path(), &interner).unwrap();

    assert_ne!(
        original.spill_subdir(dir.path()),
        copy.spill_subdir(dir.path()),
        "a clone must draw its own spill path, or the two stores overwrite \
         each other's columns"
    );
    assert_eq!(
        original.get(7, age),
        Some(Value::Int64(7)),
        "the copy's spill overwrote the original's column file"
    );
    assert_eq!(copy.get(7, age), Some(Value::Int64(4242)));
}
