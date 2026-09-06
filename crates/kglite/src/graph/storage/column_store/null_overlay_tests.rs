//! Exact merged-value and persistence oracles for immutable-base overrides.
use super::*;
use crate::graph::io::ntriples::ColumnTypeMeta;
use crate::graph::io::unified_columns::write_unified_columns;
use memmap2::MmapOptions;

struct Fixture {
    store: ColumnStore,
    interner: StringInterner,
    // Declared after store so its file mappings drop before cleanup on Windows.
    _directory: Option<tempfile::TempDir>,
}

fn key(name: &str) -> InternedKey {
    InternedKey::from_str(name)
}

fn fixture(mapped: bool, pure_overflow: bool) -> Fixture {
    let mut interner = StringInterner::new();
    for name in ["dense", "sparse", "keep", "fresh", "later"] {
        interner.get_or_intern(name);
    }
    let mut store = ColumnStore::new(Arc::new(TypeSchema::new()), &HashMap::new(), &interner);
    let mut offsets = vec![0];
    let mut bytes = Vec::new();
    for row in 0..3 {
        store.push_id(&Value::Int64(row));
        store.push_title(&Value::String(format!("row{row}")));
        let values = if pure_overflow {
            Vec::new()
        } else {
            vec![(key("dense"), Value::String(format!("dense{row}")))]
        };
        store.push_row(&values);
        bytes.extend_from_slice(&2u16.to_le_bytes());
        crate::graph::storage::overflow::encode_value(
            &mut bytes,
            key("sparse"),
            &Value::String(format!("sparse{row}")),
        );
        crate::graph::storage::overflow::encode_value(
            &mut bytes,
            key("keep"),
            &Value::Int64(row + 10),
        );
        offsets.push(bytes.len() as u64);
    }
    let mut blob = MmapBytes::new();
    blob.extend(&bytes).unwrap();
    store.replace_overflow_bag(MmapOrVec::from_vec(offsets), blob);
    if !mapped {
        return Fixture {
            store,
            interner,
            _directory: None,
        };
    }
    map_fixture(store, interner)
}

fn map_fixture(store: ColumnStore, interner: StringInterner) -> Fixture {
    let directory = tempfile::tempdir().unwrap();
    let stores = HashMap::from([("T".to_string(), Arc::new(store))]);
    let written = write_unified_columns(directory.path(), &stores, &interner).unwrap();
    assert!(written.written.contains("T"));
    let metas: Vec<ColumnTypeMeta> = serde_json::from_slice(
        &std::fs::read(directory.path().join("seg_000/columns_meta.json")).unwrap(),
    )
    .unwrap();
    let file = std::fs::File::open(directory.path().join("seg_000/columns.bin")).unwrap();
    // SAFETY: this fixture owns the immutable file and retains its directory.
    let map = unsafe { MmapOptions::new().map_copy(&file).unwrap() };
    let meta = metas
        .into_iter()
        .find(|meta| meta.type_name == "T")
        .unwrap();
    Fixture {
        store: ColumnStore::from_mmap_store(Arc::new(meta.to_mmap_store(Arc::new(map)))),
        interner,
        _directory: Some(directory),
    }
}

fn expected_row(row: u32, dense: bool, sparse: Option<Value>) -> Vec<(InternedKey, Value)> {
    let mut result = vec![(key("keep"), Value::Int64(row as i64 + 10))];
    if dense {
        result.push((key("dense"), Value::String(format!("dense{row}"))));
    }
    if let Some(value) = sparse {
        result.push((key("sparse"), value));
    }
    result
}

fn assert_row(store: &ColumnStore, row: u32, mut expected: Vec<(InternedKey, Value)>) {
    expected.sort_unstable_by_key(|(key, _)| key.as_u64());
    let mut properties = store.row_properties(row);
    properties.sort_unstable_by_key(|(key, _)| key.as_u64());
    assert_eq!(
        properties, expected,
        "row {row}: duplicates or stale values"
    );
    let mut keys = store.row_property_keys(row);
    keys.sort_unstable_by_key(|key| key.as_u64());
    assert_eq!(
        keys,
        expected.iter().map(|(key, _)| *key).collect::<Vec<_>>()
    );
    assert_eq!(store.row_property_count(row), expected.len());
    let mut borrowed = Vec::new();
    store
        .try_for_each_property_borrowed(row, |key, value| {
            borrowed.push((key, value.to_value()));
            Ok::<(), ()>(())
        })
        .unwrap();
    borrowed.sort_unstable_by_key(|(key, _)| key.as_u64());
    assert_eq!(borrowed, expected);
    for name in ["dense", "sparse", "keep", "fresh", "later"] {
        let key = key(name);
        let value = expected
            .iter()
            .find(|(k, _)| *k == key)
            .map(|(_, value)| value.clone());
        assert_eq!(store.get(row, key), value, "row {row} key {name}");
        assert_eq!(
            store.get_cow(row, key).map(std::borrow::Cow::into_owned),
            value
        );
        assert_eq!(store.contains_value(row, key), value.is_some());
        let string = format!("sparse{row}");
        assert_eq!(
            store.str_prop_eq(row, key, &string),
            value
                .as_ref()
                .map(|value| matches!(value, Value::String(s) if s == &string))
        );
        match (&value, store.str_field(row, key)) {
            (None, StrField::Absent) => {}
            (Some(Value::String(expected)), StrField::Str(actual)) => {
                assert_eq!(actual.as_ref(), expected)
            }
            (Some(_), StrField::NotString) => {}
            other => panic!("string field mismatch: {other:?}"),
        }
    }
}

fn packed(store: &ColumnStore, interner: &StringInterner) -> ColumnStore {
    let bytes = store.write_packed(interner).unwrap();
    ColumnStore::load_packed(
        store.schema_arc(),
        &HashMap::new(),
        interner,
        &bytes,
        store.row_count(),
        None,
    )
    .unwrap()
}

#[test]
fn explicit_null_over_base_preserves_untouched_rows_all_readers_and_packed_output() {
    for mapped in [false, true] {
        let mut fixture = fixture(mapped, false);
        let before = fixture.store.clone();
        assert!(fixture.store.set(0, key("sparse"), &Value::Null, None));
        let slot = fixture.store.slot(key("sparse")).unwrap();
        assert!(fixture.store.set_at_slot(0, slot, &Value::Null));
        for row in 0..3 {
            let sparse = (row != 0).then(|| Value::String(format!("sparse{row}")));
            assert_row(&fixture.store, row, expected_row(row, true, sparse));
            assert_row(
                &before,
                row,
                expected_row(row, true, Some(Value::String(format!("sparse{row}")))),
            );
        }
        let loaded = packed(&fixture.store, &fixture.interner);
        for row in 0..3 {
            assert_row(
                &loaded,
                row,
                expected_row(
                    row,
                    true,
                    (row != 0).then(|| Value::String(format!("sparse{row}"))),
                ),
            );
        }
        assert!(fixture
            .store
            .set_at_slot(0, slot, &Value::String("replacement".into())));
        assert_row(
            &fixture.store,
            0,
            expected_row(0, true, Some(Value::String("replacement".into()))),
        );
        assert!(fixture.store.set_at_slot(0, slot, &Value::Null));
        assert_row(&fixture.store, 0, expected_row(0, true, None));
    }
}

#[test]
fn pure_overflow_borrowing_and_overlay_only_key_save_do_not_drop_properties() {
    for mapped in [false, true] {
        let mut fixture = fixture(mapped, true);
        for row in 0..3 {
            assert_row(
                &fixture.store,
                row,
                expected_row(row, false, Some(Value::String(format!("sparse{row}")))),
            );
        }
        assert!(fixture.store.set(0, key("sparse"), &Value::Null, None));
        assert!(fixture
            .store
            .set(0, key("fresh"), &Value::String("added".into()), None));
        // Save before any merged read: enumeration must not be needed to repair state.
        let loaded = packed(&fixture.store, &fixture.interner);
        let mut expected = expected_row(0, false, None);
        expected.push((key("fresh"), Value::String("added".into())));
        assert_row(&loaded, 0, expected);
        assert_row(
            &loaded,
            1,
            expected_row(1, false, Some(Value::String("sparse1".into()))),
        );
    }
}

#[test]
fn null_override_clone_and_row_schema_rollback_do_not_leak_to_reused_cells() {
    let mut fixture = fixture(false, false);
    let schema = fixture.store.schema_arc();
    let columns = fixture.store.column_count();
    assert!(fixture.store.set(0, key("sparse"), &Value::Null, None));
    let cleared = fixture.store.clone();
    // Equivalent to ordinary cell replay before undoing appended schema slots.
    assert!(fixture
        .store
        .set(0, key("sparse"), &Value::String("sparse0".into()), None));
    fixture.store.restore_schema(schema, columns);
    assert_row(
        &fixture.store,
        0,
        expected_row(0, true, Some(Value::String("sparse0".into()))),
    );
    assert_row(&cleared, 0, expected_row(0, true, None));
    let row = fixture
        .store
        .push_row(&[(key("sparse"), Value::String("tail".into()))]);
    assert!(fixture.store.set(row, key("sparse"), &Value::Null, None));
    fixture.store.truncate_rows(row);
    let reused = fixture
        .store
        .push_row(&[(key("sparse"), Value::String("new".into()))]);
    assert_eq!(reused, row);
    assert_eq!(
        fixture.store.get(reused, key("sparse")),
        Some(Value::String("new".into()))
    );
    // A discarded schema slot must not donate its clear to a different new key.
    let schema = fixture.store.schema_arc();
    let columns = fixture.store.column_count();
    assert!(fixture.store.set(1, key("fresh"), &Value::Null, None));
    fixture.store.restore_schema(schema, columns);
    assert!(fixture.store.set(0, key("keep"), &Value::Int64(77), None));
    // The reused slot now names another base property. A leaked row1 marker
    // would hide its untouched overflow value even though row0 was written.
    assert_eq!(fixture.store.get(1, key("keep")), Some(Value::Int64(11)));
}

#[test]
fn bulk_column_replacement_on_overflow_base_marks_explicit_nulls_only() {
    let mut fixture = fixture(false, false);
    assert!(fixture
        .store
        .set(0, key("sparse"), &Value::String("override".into()), None));
    let mut columns: Vec<TypedColumn> = fixture.store.columns_ref().cloned().collect();
    let slot = fixture.store.slot(key("sparse")).unwrap() as usize;
    columns[slot].set(0, &Value::Null).unwrap();
    // Covered NULL cells supplied to a bulk replacement are explicit replacements.
    fixture.store.replace_columns(columns);
    assert_row(&fixture.store, 0, expected_row(0, true, None));
}

#[test]
fn untouched_mmap_packed_output_deduplicates_promoted_overflow_values() {
    let mut base = fixture(false, false);
    // Only row0 gets a dense cell; rows1/2 retain their ordinary overflow values.
    assert!(base
        .store
        .set(0, key("sparse"), &Value::String("sparse0".into()), None));
    let mapped = map_fixture(base.store, base.interner);
    assert_eq!(mapped.store.column_count(), 0);
    assert!(mapped.store.has_mmap_base());
    for row in 0..3 {
        assert_row(
            &mapped.store,
            row,
            expected_row(row, true, Some(Value::String(format!("sparse{row}")))),
        );
    }
    let loaded = packed(&mapped.store, &mapped.interner);
    for row in 0..3 {
        assert_eq!(
            loaded.get_overflow_property(row, key("sparse")),
            None,
            "emitted dense values own this key"
        );
        assert_row(
            &loaded,
            row,
            expected_row(row, true, Some(Value::String(format!("sparse{row}")))),
        );
    }
}

#[test]
fn dense_nulls_need_no_override_state_and_shrinking_discards_old_clears() {
    let mut ordinary = fixture(false, false);
    ordinary.store.overflow_offsets = None;
    ordinary.store.overflow_data = None;
    let slot = ordinary.store.slot(key("dense")).unwrap();
    assert!(ordinary.store.set_at_slot(0, slot, &Value::Null));
    assert_eq!(ordinary.store.get(0, key("dense")), None);
    assert!(ordinary.store.null_overrides.is_none());
    assert_eq!(
        ordinary.store.get(1, key("dense")),
        Some(Value::String("dense1".into()))
    );

    let mut base = fixture(false, false);
    assert!(base.store.set(2, key("sparse"), &Value::Null, None));
    let before = base.store.clone();
    assert!(!base.store.set_at_slot(3, 0, &Value::Null));
    assert!(!base.store.set_at_slot(0, u16::MAX, &Value::Null));
    assert_eq!(base.store.null_overrides, before.null_overrides);
    base.store.set_row_count(2);
    base.store.set_row_count(3);
    assert_eq!(
        base.store.get(2, key("sparse")),
        Some(Value::String("sparse2".into()))
    );
    assert_eq!(before.get(2, key("sparse")), None);
}

#[test]
fn fixed_null_cells_inherit_overflow_for_string_predicates() {
    let mut base = fixture(false, false);
    assert!(base.store.set(0, key("fresh"), &Value::Int64(1), None));
    assert!(base.store.set(0, key("sparse"), &Value::Int64(0), None));
    let mapped = map_fixture(base.store, base.interner);
    assert_eq!(mapped.store.str_prop_eq(1, key("fresh"), "x"), None);
    assert_eq!(
        mapped.store.str_prop_eq(1, key("sparse"), "sparse1"),
        Some(true)
    );
    assert_eq!(mapped.store.str_prop_eq(0, key("sparse"), "0"), Some(false));
    assert_eq!(mapped.store.str_prop_eq(3, key("sparse"), "x"), None);
}

#[test]
fn mapped_title_overrides_remain_authoritative_for_borrowed_reads() {
    let mut mapped = fixture(true, false);
    let held = mapped.store.clone();
    assert!(mapped.store.set_title(0, &Value::Null));
    assert_eq!(mapped.store.get_title(0), None);
    assert!(matches!(mapped.store.title_field(0), StrField::Absent));
    assert_eq!(mapped.store.title_borrowed(0), None);
    assert_eq!(mapped.store.title_borrowed(1), Some("row1"));
    assert_eq!(held.title_borrowed(0), Some("row0"));
    assert!(mapped.store.set_title(0, &Value::String("changed".into())));
    assert_eq!(mapped.store.title_borrowed(0), Some("changed"));
    let loaded = packed(&mapped.store, &mapped.interner);
    assert_eq!(loaded.get_title(0), Some(Value::String("changed".into())));
    assert_eq!(loaded.get_title(1), Some(Value::String("row1".into())));
}
