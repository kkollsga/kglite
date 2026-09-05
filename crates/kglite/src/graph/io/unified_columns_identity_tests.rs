use super::*;
use crate::datatypes::values::{BorrowedValue, Value};
use crate::graph::schema::TypeSchema;
use memmap2::MmapOptions;

fn store(ids: &[Value]) -> ColumnStore {
    let mut store = ColumnStore::new(
        Arc::new(TypeSchema::new()),
        &HashMap::new(),
        &StringInterner::new(),
    );
    for id in ids {
        store.push_id(id);
        store.push_title(&Value::String("row".into()));
        store.push_row(&[]);
    }
    store
}

fn mapped(store: ColumnStore) -> (ColumnTypeMeta, MmapMut) {
    let dir = tempfile::tempdir().unwrap();
    // The extra type supplies bytes when the tested column has zero rows.
    let stores = HashMap::from([
        ("Subject".into(), Arc::new(store)),
        (
            "Control".into(),
            Arc::new(self::store(&[Value::UniqueId(7)])),
        ),
    ]);
    let result = write_unified_columns(dir.path(), &stores, &StringInterner::new()).unwrap();
    assert!(result.written.contains("Subject"));
    let metas: Vec<ColumnTypeMeta> = serde_json::from_slice(
        &std::fs::read(dir.path().join("seg_000/columns_meta.json")).unwrap(),
    )
    .unwrap();
    let file = File::open(dir.path().join("seg_000/columns.bin")).unwrap();
    // SAFETY: the test owns this immutable file; the private map survives its unlink.
    let mmap = unsafe { MmapOptions::new().map_copy(&file).unwrap() };
    (
        metas
            .into_iter()
            .find(|m| m.type_name == "Subject")
            .unwrap(),
        mmap,
    )
}

#[test]
fn unified_integer_identity_reads_preserve_width_sign_nulls_and_borrowing() {
    let values = [
        Value::Int64(i64::MIN),
        Value::Int64(-1),
        Value::Null,
        Value::Int64(i32::MAX as i64 + 1),
        Value::Int64(u32::MAX as i64 + 1),
        Value::Int64(i64::MAX),
    ];
    let (meta, mmap) = mapped(store(&values));
    assert_eq!(meta.id_data.len, values.len() * 8);
    let loaded = ColumnStore::from_mmap_store(Arc::new(meta.to_mmap_store(Arc::new(mmap))));
    assert_eq!(loaded.id_type_str(), Some("int64"));
    for (i, expected) in values.iter().enumerate() {
        let expected = (!matches!(expected, Value::Null)).then_some(expected.clone());
        assert_eq!(loaded.get_id(i as u32), expected);
        assert_eq!(
            loaded.id_borrowed(i as u32).map(BorrowedValue::to_value),
            expected
        );
        if expected.is_some() {
            assert!(matches!(
                loaded.id_borrowed(i as u32),
                Some(BorrowedValue::Int64(_))
            ));
        }
    }
}

#[test]
fn unified_compact_ids_keep_native_u32_representation() {
    let values = [Value::UniqueId(0), Value::UniqueId(u32::MAX), Value::Null];
    let (meta, mmap) = mapped(store(&values));
    assert_eq!(meta.id_data.len, values.len() * 4);
    let loaded = ColumnStore::from_mmap_store(Arc::new(meta.to_mmap_store(Arc::new(mmap))));
    assert_eq!(loaded.id_type_str(), Some("uniqueid"));
    assert!(matches!(loaded.get_id(1), Some(Value::UniqueId(u32::MAX))));
    assert!(matches!(
        loaded.id_borrowed(1),
        Some(BorrowedValue::UniqueId(u32::MAX))
    ));
    assert!(loaded.get_id(2).is_none());
}

#[test]
fn empty_missing_and_all_null_fixed_ids_never_create_zero_identities() {
    let mut empty = store(&[Value::Int64(1)]);
    empty.truncate_rows(0);
    let (meta, mmap) = mapped(empty);
    assert_eq!(meta.id_data.len, 0);
    let loaded = meta.to_mmap_store(Arc::new(mmap));
    assert!(loaded.get_id(0).is_none());
    assert!(loaded.id_borrowed(0).is_none());

    let mut missing = store(&[]);
    missing.push_title(&Value::String("no id".into()));
    missing.push_row(&[]);
    let (meta, mmap) = mapped(missing);
    let loaded = meta.to_mmap_store(Arc::new(mmap));
    assert!(loaded.get_id(0).is_none());
    assert!(loaded.id_borrowed(0).is_none());

    let (meta, mut mmap) = mapped(store(&[Value::Int64(1), Value::Int64(2)]));
    // Valid fixed-width payload with every row marked null still carries its width.
    mmap[meta.id_nulls.offset..meta.id_nulls.offset + meta.id_nulls.len].fill(1);
    let loaded = meta.to_mmap_store(Arc::new(mmap));
    for row in 0..2 {
        assert!(loaded.get_id(row).is_none());
        assert!(loaded.id_borrowed(row).is_none());
    }
}

#[test]
fn unsupported_identity_columns_require_lossless_sidecars() {
    for value in [
        Value::Float64(1.5),
        Value::Boolean(true),
        Value::DateTime(chrono::NaiveDate::from_ymd_opt(2020, 1, 1).unwrap()),
    ] {
        let dir = tempfile::tempdir().unwrap();
        let stores = HashMap::from([("Subject".into(), Arc::new(store(&[value])))]);
        let result = write_unified_columns(dir.path(), &stores, &StringInterner::new()).unwrap();
        assert!(result.unhandled.contains("Subject"));
        assert!(result.written.is_empty());
        assert!(!dir.path().join("seg_000/columns.bin").exists());
    }
}
