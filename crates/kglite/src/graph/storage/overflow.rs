//! Overflow-bag value codec — the single source of truth for the
//! per-row property "overflow" wire format shared by the heap
//! [`ColumnStore`](crate::graph::storage::column_store::ColumnStore)
//! and the mmap-backed
//! [`MmapColumnStore`](crate::graph::storage::mapped::column_store::MmapColumnStore).
//!
//! Historically the encoder and decoder were written twice (owned +
//! borrowed, heap + mapped) and drifted: the mapped borrowed decoder
//! was missing the Timestamp tag and dropped every remaining property
//! of a row on the first unknown tag. Centralising the tag table and
//! all encode/decode/skip logic here makes that class of drift
//! impossible.
//!
//! ## Wire format
//!
//! Per row blob: `[num_entries: u16 LE]` then repeated per entry:
//!
//! ```text
//! [key: u64 LE][tag: u8][payload: tag-specific]
//! ```
//!
//! | tag | value      | payload                                  |
//! |-----|------------|------------------------------------------|
//! | 0   | Null       | (empty)                                  |
//! | 1   | Int64      | 8 bytes i64 LE                           |
//! | 2   | Float64    | 8 bytes f64 LE                           |
//! | 3   | UniqueId   | 4 bytes u32 LE                           |
//! | 4   | Boolean    | 1 byte                                   |
//! | 5   | DateTime   | 4 bytes i32 LE (days since unix epoch)   |
//! | 6   | String     | u32 LE length + UTF-8 bytes              |
//! | 7   | Timestamp  | 8 bytes i64 LE (seconds since unix epoch)|
//! | 8   | Reserved   | u32 LE length + retired pre-0.14 payload |
//! | 9   | List       | u32 LE length + Postcard(`Vec<Value>`)   |
//! | 10  | Map        | u32 LE length + Postcard(`BTreeMap`)     |
//!
//! **Forward-compat rule:** any *future* tag MUST use a `u32 LE`
//! length prefix + payload, so that older readers can skip an unknown
//! tag without losing the rest of the row (see [`skip_value`]).
//!
//! ## Deliberately non-storable values
//!
//! `Node`, `Relationship`, and `Path` are query-result-time values, while
//! `Duration` and `NodeRef` are query-time-only; the encoder writes them as
//! the Null tag. `Point` is encoded as its string form `"lat,lon"` (tag 6),
//! matching the legacy writers.

use crate::datatypes::values::{BorrowedValue, Value};
use crate::graph::schema::InternedKey;
use chrono::NaiveDate;

const UNIX_EPOCH_DATE: NaiveDate = match NaiveDate::from_ymd_opt(1970, 1, 1) {
    Some(d) => d,
    None => unreachable!(),
};

pub const TAG_NULL: u8 = 0;
pub const TAG_INT64: u8 = 1;
pub const TAG_FLOAT64: u8 = 2;
pub const TAG_UNIQUE_ID: u8 = 3;
pub const TAG_BOOL: u8 = 4;
pub const TAG_DATE: u8 = 5;
pub const TAG_STRING: u8 = 6;
pub const TAG_TIMESTAMP: u8 = 7;
/// Retired pre-0.14 list payload tag. Kept reserved so scanners can skip it
/// without misaligning later properties; it is never decoded or written.
pub const TAG_LIST: u8 = 8;
pub const TAG_LIST_POSTCARD: u8 = 9;
pub const TAG_MAP_POSTCARD: u8 = 10;

/// Highest tag this build knows how to *decode*. Tags above this are
/// skipped via the length-prefix forward-compat rule (module docs).
pub const MAX_KNOWN_TAG: u8 = TAG_MAP_POSTCARD;

// ─── Encoders ────────────────────────────────────────────────────────────────

/// Serialize one `(key, value)` entry into overflow wire format,
/// appending to `buf`. The owned-`Value` encoder.
pub fn encode_value(buf: &mut Vec<u8>, key: InternedKey, value: &Value) {
    buf.extend_from_slice(&key.as_u64().to_le_bytes());
    match value {
        Value::Null => buf.push(TAG_NULL),
        Value::Int64(v) => {
            buf.push(TAG_INT64);
            buf.extend_from_slice(&v.to_le_bytes());
        }
        Value::Float64(v) => {
            buf.push(TAG_FLOAT64);
            buf.extend_from_slice(&v.to_le_bytes());
        }
        Value::UniqueId(v) => {
            buf.push(TAG_UNIQUE_ID);
            buf.extend_from_slice(&v.to_le_bytes());
        }
        Value::Boolean(v) => {
            buf.push(TAG_BOOL);
            buf.push(*v as u8);
        }
        Value::DateTime(d) => {
            buf.push(TAG_DATE);
            let days = (*d - UNIX_EPOCH_DATE).num_days() as i32;
            buf.extend_from_slice(&days.to_le_bytes());
        }
        Value::Timestamp(dt) => {
            buf.push(TAG_TIMESTAMP);
            let epoch = UNIX_EPOCH_DATE.and_hms_opt(0, 0, 0).unwrap_or_default();
            let secs = (*dt - epoch).num_seconds();
            buf.extend_from_slice(&secs.to_le_bytes());
        }
        Value::String(s) => encode_str(buf, s),
        // Point is serialized as its "lat,lon" string form (legacy
        // convention — kept for wire-format stability).
        Value::Point { lat, lon } => encode_str(buf, &format!("{},{}", lat, lon)),
        // NodeRef is transient; Duration is query-time-only (Cluster 2).
        // Both are written as null.
        Value::NodeRef(_) | Value::Duration { .. } => buf.push(TAG_NULL),
        // Lists are persistable property values. Tag 8 is retired and
        // current writes always use the Postcard tag.
        Value::List(items) => encode_list(buf, items),
        Value::Map(entries) => encode_map(buf, entries),
        // Graph-entity variants are query-result-time values; they don't
        // belong in the overflow property bag and are written as null.
        Value::Node(_) | Value::Relationship(_) | Value::Path(_) => {
            buf.push(TAG_NULL);
        }
    }
}

/// Borrowed-value counterpart of [`encode_value`] for hot-path callers
/// that already hold a `BorrowedValue` (avoids the clone into `Value`).
/// Produces byte-identical output for every representable variant.
pub fn encode_value_borrowed(buf: &mut Vec<u8>, key: InternedKey, value: &BorrowedValue<'_>) {
    buf.extend_from_slice(&key.as_u64().to_le_bytes());
    match value {
        BorrowedValue::Null => buf.push(TAG_NULL),
        BorrowedValue::Int64(v) => {
            buf.push(TAG_INT64);
            buf.extend_from_slice(&v.to_le_bytes());
        }
        BorrowedValue::Float64(v) => {
            buf.push(TAG_FLOAT64);
            buf.extend_from_slice(&v.to_le_bytes());
        }
        BorrowedValue::UniqueId(v) => {
            buf.push(TAG_UNIQUE_ID);
            buf.extend_from_slice(&v.to_le_bytes());
        }
        BorrowedValue::Boolean(v) => {
            buf.push(TAG_BOOL);
            buf.push(*v as u8);
        }
        BorrowedValue::DateTime(d) => {
            buf.push(TAG_DATE);
            let days = (*d - UNIX_EPOCH_DATE).num_days() as i32;
            buf.extend_from_slice(&days.to_le_bytes());
        }
        BorrowedValue::Timestamp(dt) => {
            buf.push(TAG_TIMESTAMP);
            let epoch = UNIX_EPOCH_DATE.and_hms_opt(0, 0, 0).unwrap_or_default();
            let secs = (*dt - epoch).num_seconds();
            buf.extend_from_slice(&secs.to_le_bytes());
        }
        BorrowedValue::String(s) => encode_str(buf, s),
        BorrowedValue::List(items) => encode_list(buf, items),
        BorrowedValue::Map(entries) => encode_map(buf, entries),
    }
}

#[inline]
fn encode_str(buf: &mut Vec<u8>, s: &str) {
    buf.push(TAG_STRING);
    buf.extend_from_slice(&(s.len() as u32).to_le_bytes());
    buf.extend_from_slice(s.as_bytes());
}

#[inline]
fn encode_list(buf: &mut Vec<u8>, items: &[Value]) {
    match crate::serde_codec::encode_versioned(
        crate::serde_codec::CURRENT_CODEC,
        items,
        u32::MAX as u64,
    ) {
        Ok(bytes) => {
            buf.push(TAG_LIST_POSTCARD);
            buf.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
            buf.extend_from_slice(&bytes);
        }
        // Serialisation should never fail for a Vec<Value>; if it
        // somehow does, fall back to null rather than corrupting the
        // blob.
        Err(_) => buf.push(TAG_NULL),
    }
}

#[inline]
fn encode_map(buf: &mut Vec<u8>, entries: &std::collections::BTreeMap<String, Value>) {
    match crate::serde_codec::encode_versioned(
        crate::serde_codec::CURRENT_CODEC,
        entries,
        u32::MAX as u64,
    ) {
        Ok(bytes) => {
            buf.push(TAG_MAP_POSTCARD);
            buf.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
            buf.extend_from_slice(&bytes);
        }
        // `Value` maps are serializable. Preserve the existing infallible
        // overflow-writer contract if resource limits are ever exceeded.
        Err(_) => buf.push(TAG_NULL),
    }
}

// ─── Decoders ────────────────────────────────────────────────────────────────

/// Read a single *known-tag* value from `blob` at `*pos`, advancing
/// `*pos` past it. Returns `None` when the payload is truncated or the
/// tag is unknown (callers use [`skip_value`] for unknown tags).
pub fn read_value(blob: &[u8], pos: &mut usize, type_tag: u8) -> Option<Value> {
    match type_tag {
        TAG_NULL => Some(Value::Null),
        TAG_INT64 => {
            if *pos + 8 > blob.len() {
                return None;
            }
            let v = i64::from_le_bytes(blob[*pos..*pos + 8].try_into().ok()?);
            *pos += 8;
            Some(Value::Int64(v))
        }
        TAG_FLOAT64 => {
            if *pos + 8 > blob.len() {
                return None;
            }
            let v = f64::from_le_bytes(blob[*pos..*pos + 8].try_into().ok()?);
            *pos += 8;
            Some(Value::Float64(v))
        }
        TAG_UNIQUE_ID => {
            if *pos + 4 > blob.len() {
                return None;
            }
            let v = u32::from_le_bytes(blob[*pos..*pos + 4].try_into().ok()?);
            *pos += 4;
            Some(Value::UniqueId(v))
        }
        TAG_BOOL => {
            if *pos + 1 > blob.len() {
                return None;
            }
            let v = blob[*pos] != 0;
            *pos += 1;
            Some(Value::Boolean(v))
        }
        TAG_DATE => {
            if *pos + 4 > blob.len() {
                return None;
            }
            let days = i32::from_le_bytes(blob[*pos..*pos + 4].try_into().ok()?);
            *pos += 4;
            Some(Value::DateTime(
                UNIX_EPOCH_DATE + chrono::Duration::days(days as i64),
            ))
        }
        TAG_STRING => {
            let bytes = read_len_prefixed(blob, pos)?;
            // Lossy decode matches the long-standing owned-reader
            // behaviour: a malformed byte sequence yields the
            // U+FFFD-substituted form rather than dropping the row.
            Some(Value::String(String::from_utf8_lossy(bytes).into_owned()))
        }
        TAG_TIMESTAMP => {
            if *pos + 8 > blob.len() {
                return None;
            }
            let secs = i64::from_le_bytes(blob[*pos..*pos + 8].try_into().ok()?);
            *pos += 8;
            let epoch = UNIX_EPOCH_DATE.and_hms_opt(0, 0, 0)?;
            Some(Value::Timestamp(epoch + chrono::Duration::seconds(secs)))
        }
        TAG_LIST_POSTCARD => {
            let bytes = read_len_prefixed(blob, pos)?;
            let limits = crate::serde_codec::DecodeLimits::new(u32::MAX as u64, bytes.len() as u64);
            let items: Vec<Value> = crate::serde_codec::decode_exact_with(
                crate::serde_codec::CodecVersion::PostcardV1,
                bytes,
                bytes.len() as u64,
                limits,
            )
            .ok()?;
            Some(Value::List(items))
        }
        TAG_MAP_POSTCARD => {
            let bytes = read_len_prefixed(blob, pos)?;
            let limits = crate::serde_codec::DecodeLimits::new(u32::MAX as u64, bytes.len() as u64);
            let entries: std::collections::BTreeMap<String, Value> =
                crate::serde_codec::decode_exact_with(
                    crate::serde_codec::CodecVersion::PostcardV1,
                    bytes,
                    bytes.len() as u64,
                    limits,
                )
                .ok()?;
            Some(Value::Map(entries))
        }
        _ => None,
    }
}

/// Read a `u32 LE` length prefix + that many payload bytes, advancing
/// `*pos` past both. `None` when truncated.
#[inline]
fn read_len_prefixed<'b>(blob: &'b [u8], pos: &mut usize) -> Option<&'b [u8]> {
    if *pos + 4 > blob.len() {
        return None;
    }
    let len = u32::from_le_bytes(blob[*pos..*pos + 4].try_into().ok()?) as usize;
    *pos += 4;
    let end = pos.checked_add(len)?;
    if end > blob.len() {
        return None;
    }
    let bytes = &blob[*pos..end];
    *pos = end;
    Some(bytes)
}

/// Skip one value (known **or unknown** tag) without decoding it,
/// advancing `*pos` past its payload. Unknown tags (> [`MAX_KNOWN_TAG`])
/// follow the forward-compat rule — u32 length prefix + payload — so a
/// reader older than the writer loses only the one entry it can't
/// understand, never the rest of the row.
///
/// Returns `false` when the blob is truncated (the caller must stop
/// scanning this row — position can no longer be trusted).
#[must_use]
pub fn skip_value(blob: &[u8], pos: &mut usize, type_tag: u8) -> bool {
    let fixed = match type_tag {
        TAG_NULL => 0,
        TAG_INT64 | TAG_FLOAT64 | TAG_TIMESTAMP => 8,
        TAG_UNIQUE_ID | TAG_DATE => 4,
        TAG_BOOL => 1,
        // Length-prefixed: String, List, and every future tag.
        _ => return read_len_prefixed(blob, pos).is_some(),
    };
    if *pos + fixed > blob.len() {
        return false;
    }
    *pos += fixed;
    true
}

/// Scan a row blob for a specific key. Returns the decoded value if
/// the key is present and decodable.
pub fn scan_blob(blob: &[u8], key: InternedKey) -> Option<Value> {
    let target = key.as_u64();
    let mut found = None;
    let _ = for_each_raw(blob, |entry_key, tag, blob, pos| {
        if entry_key == target && tag <= MAX_KNOWN_TAG && tag != TAG_LIST {
            found = read_value(blob, pos, tag);
            return Some(()); // stop scanning
        }
        if !skip_value(blob, pos, tag) {
            return Some(()); // truncated — stop
        }
        None
    });
    found
}

/// Decode all entries from a row blob into owned `(key, Value)` pairs.
/// Unknown tags are skipped (not dropped-with-the-rest); a truncated
/// tail ends the scan with the entries decoded so far.
pub fn decode_blob(blob: &[u8]) -> Vec<(InternedKey, Value)> {
    let mut result = Vec::new();
    let _ = for_each_raw(blob, |entry_key, tag, blob, pos| {
        if tag <= MAX_KNOWN_TAG && tag != TAG_LIST {
            match read_value(blob, pos, tag) {
                Some(val) => result.push((InternedKey::from_u64(entry_key), val)),
                None => return Some(()), // truncated payload — stop
            }
        } else if !skip_value(blob, pos, tag) {
            return Some(()); // truncated unknown entry — stop
        }
        None
    });
    result
}

/// Visit each entry of a row blob as a zero/low-copy
/// [`BorrowedValue`]: strings borrow the blob bytes directly (after a
/// UTF-8 *check* — corrupt bytes are decoded lossily, matching
/// [`read_value`]), lists and maps allocate transient containers that live
/// across the callback. Unknown tags are skipped via the
/// forward-compat length prefix; a truncated tail ends the visit.
///
/// Stops early at the first `Err` returned by `f`.
pub fn try_for_each_borrowed<F, E>(blob: &[u8], mut f: F) -> Result<(), E>
where
    F: FnMut(InternedKey, BorrowedValue<'_>) -> Result<(), E>,
{
    for_each_raw(blob, |entry_key, tag, blob, pos| {
        let key = InternedKey::from_u64(entry_key);
        match tag {
            TAG_NULL => {
                if let Err(e) = f(key, BorrowedValue::Null) {
                    return Some(Err(e));
                }
            }
            TAG_INT64 => {
                if *pos + 8 > blob.len() {
                    return Some(Ok(()));
                }
                let v = i64::from_le_bytes(blob[*pos..*pos + 8].try_into().unwrap_or([0; 8]));
                *pos += 8;
                if let Err(e) = f(key, BorrowedValue::Int64(v)) {
                    return Some(Err(e));
                }
            }
            TAG_FLOAT64 => {
                if *pos + 8 > blob.len() {
                    return Some(Ok(()));
                }
                let v = f64::from_le_bytes(blob[*pos..*pos + 8].try_into().unwrap_or([0; 8]));
                *pos += 8;
                if let Err(e) = f(key, BorrowedValue::Float64(v)) {
                    return Some(Err(e));
                }
            }
            TAG_UNIQUE_ID => {
                if *pos + 4 > blob.len() {
                    return Some(Ok(()));
                }
                let v = u32::from_le_bytes(blob[*pos..*pos + 4].try_into().unwrap_or([0; 4]));
                *pos += 4;
                if let Err(e) = f(key, BorrowedValue::UniqueId(v)) {
                    return Some(Err(e));
                }
            }
            TAG_BOOL => {
                if *pos + 1 > blob.len() {
                    return Some(Ok(()));
                }
                let v = blob[*pos] != 0;
                *pos += 1;
                if let Err(e) = f(key, BorrowedValue::Boolean(v)) {
                    return Some(Err(e));
                }
            }
            TAG_DATE => {
                if *pos + 4 > blob.len() {
                    return Some(Ok(()));
                }
                let days = i32::from_le_bytes(blob[*pos..*pos + 4].try_into().unwrap_or([0; 4]));
                *pos += 4;
                let d = UNIX_EPOCH_DATE + chrono::Duration::days(days as i64);
                if let Err(e) = f(key, BorrowedValue::DateTime(d)) {
                    return Some(Err(e));
                }
            }
            TAG_STRING => {
                let Some(bytes) = read_len_prefixed(blob, pos) else {
                    return Some(Ok(()));
                };
                // Checked conversion: overflow blobs normally hold
                // valid UTF-8 (written from `String::as_bytes`), but
                // the bytes come from disk — validate rather than
                // `from_utf8_unchecked` (UB on corrupt input). The
                // rare invalid case decodes lossily, matching the
                // owned reader.
                match std::str::from_utf8(bytes) {
                    Ok(s) => {
                        if let Err(e) = f(key, BorrowedValue::String(s)) {
                            return Some(Err(e));
                        }
                    }
                    Err(_) => {
                        let owned = String::from_utf8_lossy(bytes);
                        if let Err(e) = f(key, BorrowedValue::String(&owned)) {
                            return Some(Err(e));
                        }
                    }
                }
            }
            TAG_TIMESTAMP => {
                if *pos + 8 > blob.len() {
                    return Some(Ok(()));
                }
                let secs = i64::from_le_bytes(blob[*pos..*pos + 8].try_into().unwrap_or([0; 8]));
                *pos += 8;
                let Some(epoch) = UNIX_EPOCH_DATE.and_hms_opt(0, 0, 0) else {
                    return Some(Ok(()));
                };
                let ts = epoch + chrono::Duration::seconds(secs);
                if let Err(e) = f(key, BorrowedValue::Timestamp(ts)) {
                    return Some(Err(e));
                }
            }
            TAG_LIST_POSTCARD => {
                let Some(bytes) = read_len_prefixed(blob, pos) else {
                    return Some(Ok(()));
                };
                // Lists can't borrow the blob — the codec must allocate —
                // but the owned `Vec` lives across the synchronous
                // `f()` call, so a borrowed slice into it is valid.
                let limits =
                    crate::serde_codec::DecodeLimits::new(u32::MAX as u64, bytes.len() as u64);
                let items: Vec<Value> = match crate::serde_codec::decode_exact_with(
                    crate::serde_codec::CodecVersion::PostcardV1,
                    bytes,
                    bytes.len() as u64,
                    limits,
                ) {
                    Ok(v) => v,
                    Err(_) => return Some(Ok(())), // corrupt payload — stop
                };
                if let Err(e) = f(key, BorrowedValue::List(&items)) {
                    return Some(Err(e));
                }
            }
            TAG_MAP_POSTCARD => {
                let Some(bytes) = read_len_prefixed(blob, pos) else {
                    return Some(Ok(()));
                };
                let limits =
                    crate::serde_codec::DecodeLimits::new(u32::MAX as u64, bytes.len() as u64);
                let entries: std::collections::BTreeMap<String, Value> =
                    match crate::serde_codec::decode_exact_with(
                        crate::serde_codec::CodecVersion::PostcardV1,
                        bytes,
                        bytes.len() as u64,
                        limits,
                    ) {
                        Ok(v) => v,
                        Err(_) => return Some(Ok(())),
                    };
                if let Err(e) = f(key, BorrowedValue::Map(&entries)) {
                    return Some(Err(e));
                }
            }
            _ => {
                // Unknown (future) tag: length-prefixed by the
                // forward-compat rule. Skip it, keep the rest of the
                // row's properties.
                if !skip_value(blob, pos, tag) {
                    return Some(Ok(()));
                }
            }
        }
        None
    })
    .unwrap_or(Ok(()))
}

/// Shared entry-walk driver: parses `[num_entries]` then per entry the
/// `[key][tag]` header, handing `(key, tag, blob, pos)` to `step`.
/// `step` must advance `*pos` past the entry payload; returning
/// `Some(r)` short-circuits the walk with `r`.
fn for_each_raw<R>(
    blob: &[u8],
    mut step: impl FnMut(u64, u8, &[u8], &mut usize) -> Option<R>,
) -> Option<R> {
    if blob.len() < 2 {
        return None;
    }
    let num_entries = u16::from_le_bytes([blob[0], blob[1]]) as usize;
    let mut pos = 2;
    for _ in 0..num_entries {
        if pos + 9 > blob.len() {
            break; // truncated entry header
        }
        let entry_key = u64::from_le_bytes(blob[pos..pos + 8].try_into().unwrap_or([0; 8]));
        let type_tag = blob[pos + 8];
        pos += 9;
        if let Some(r) = step(entry_key, type_tag, blob, &mut pos) {
            return Some(r);
        }
    }
    None
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn key(n: u64) -> InternedKey {
        InternedKey::from_u64(n)
    }

    fn blob_of(entries: &[(InternedKey, Value)]) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&(entries.len() as u16).to_le_bytes());
        for (k, v) in entries {
            encode_value(&mut buf, *k, v);
        }
        buf
    }

    #[test]
    fn owned_round_trip_all_tags() {
        let ts = UNIX_EPOCH_DATE.and_hms_opt(0, 0, 0).unwrap() + chrono::Duration::seconds(12345);
        let entries = vec![
            (key(1), Value::Null),
            (key(2), Value::Int64(-7)),
            (key(3), Value::Float64(2.5)),
            (key(4), Value::UniqueId(42)),
            (key(5), Value::Boolean(true)),
            (
                key(6),
                Value::DateTime(NaiveDate::from_ymd_opt(2020, 6, 1).unwrap()),
            ),
            (key(7), Value::String("hello".into())),
            (key(8), Value::Timestamp(ts)),
            (
                key(9),
                Value::List(vec![Value::Int64(1), Value::String("x".into())]),
            ),
            (
                key(10),
                Value::Map(std::collections::BTreeMap::from([
                    ("count".into(), Value::Int64(2)),
                    (
                        "nested".into(),
                        Value::List(vec![Value::Map(std::collections::BTreeMap::from([(
                            "ok".into(),
                            Value::Boolean(true),
                        )]))]),
                    ),
                ])),
            ),
        ];
        let blob = blob_of(&entries);
        let mut encoded_list = Vec::new();
        encode_value(
            &mut encoded_list,
            key(9),
            &Value::List(vec![Value::Int64(1), Value::String("x".into())]),
        );
        assert_eq!(encoded_list[8], TAG_LIST_POSTCARD);
        assert_eq!(decode_blob(&blob), entries);
        // Per-key scan agrees.
        for (k, v) in &entries {
            assert_eq!(scan_blob(&blob, *k).as_ref(), Some(v));
        }
    }

    #[test]
    fn retired_list_tag_is_skipped_without_losing_later_properties() {
        let payload = [1u8, 2, 3];
        let mut blob = Vec::new();
        blob.extend_from_slice(&2u16.to_le_bytes());
        blob.extend_from_slice(&key(1).as_u64().to_le_bytes());
        blob.push(TAG_LIST);
        blob.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        blob.extend_from_slice(&payload);
        encode_value(&mut blob, key(2), &Value::Int64(7));

        assert_eq!(scan_blob(&blob, key(1)), None);
        assert_eq!(scan_blob(&blob, key(2)), Some(Value::Int64(7)));
        assert_eq!(decode_blob(&blob), vec![(key(2), Value::Int64(7))]);
        let mut seen = Vec::new();
        try_for_each_borrowed(&blob, |_, value| -> Result<(), ()> {
            seen.push(value.to_value());
            Ok(())
        })
        .unwrap();
        assert_eq!(seen, vec![Value::Int64(7)]);
    }

    #[test]
    fn borrowed_round_trip_includes_timestamp() {
        // Regression: the mapped borrowed decoder was missing tag 7 —
        // a Timestamp in the overflow bag ended the row scan.
        let ts = UNIX_EPOCH_DATE.and_hms_opt(0, 0, 0).unwrap() + chrono::Duration::seconds(9999);
        let entries = vec![
            (key(1), Value::Timestamp(ts)),
            (key(2), Value::String("after-the-timestamp".into())),
        ];
        let blob = blob_of(&entries);
        let mut seen: Vec<(InternedKey, Value)> = Vec::new();
        try_for_each_borrowed(&blob, |k, bv| -> Result<(), ()> {
            seen.push((k, bv.to_value()));
            Ok(())
        })
        .unwrap();
        assert_eq!(seen, entries);
    }

    #[test]
    fn borrowed_encoder_matches_owned_for_timestamp() {
        let ts = UNIX_EPOCH_DATE.and_hms_opt(0, 0, 0).unwrap() + chrono::Duration::seconds(4242);
        let mut owned = Vec::new();
        encode_value(&mut owned, key(9), &Value::Timestamp(ts));
        let mut borrowed = Vec::new();
        encode_value_borrowed(&mut borrowed, key(9), &BorrowedValue::Timestamp(ts));
        assert_eq!(owned, borrowed);
    }

    #[test]
    fn borrowed_encoder_and_decoder_preserve_nested_map() {
        let entries = std::collections::BTreeMap::from([
            ("name".into(), Value::String("map".into())),
            (
                "items".into(),
                Value::List(vec![Value::Map(std::collections::BTreeMap::from([(
                    "answer".into(),
                    Value::Int64(42),
                )]))]),
            ),
        ]);
        let expected = Value::Map(entries.clone());

        let mut owned = Vec::new();
        encode_value(&mut owned, key(9), &expected);
        let mut borrowed = Vec::new();
        encode_value_borrowed(&mut borrowed, key(9), &BorrowedValue::Map(&entries));

        assert_eq!(owned, borrowed);
        assert_eq!(owned[8], TAG_MAP_POSTCARD);
        let payload_len = u32::from_le_bytes(owned[9..13].try_into().unwrap()) as usize;
        assert_eq!(owned.len(), 13 + payload_len);

        let blob = blob_of(&[(key(9), expected.clone())]);
        assert_eq!(decode_blob(&blob), vec![(key(9), expected.clone())]);
        let mut seen = Vec::new();
        try_for_each_borrowed(&blob, |k, value| -> Result<(), ()> {
            seen.push((k, value.to_value()));
            Ok(())
        })
        .unwrap();
        assert_eq!(seen, vec![(key(9), expected)]);
    }

    #[test]
    fn corrupt_map_payload_ends_scan_without_fabricating_a_value() {
        let mut blob = Vec::new();
        blob.extend_from_slice(&2u16.to_le_bytes());
        blob.extend_from_slice(&key(1).as_u64().to_le_bytes());
        blob.push(TAG_MAP_POSTCARD);
        blob.extend_from_slice(&1u32.to_le_bytes());
        blob.push(0xff);
        encode_value(&mut blob, key(2), &Value::String("not-reached".into()));

        assert!(decode_blob(&blob).is_empty());
        assert_eq!(scan_blob(&blob, key(1)), None);
        let mut seen = Vec::new();
        try_for_each_borrowed(&blob, |k, value| -> Result<(), ()> {
            seen.push((k, value.to_value()));
            Ok(())
        })
        .unwrap();
        assert!(seen.is_empty());
    }

    /// One unknown (future, length-prefixed) tag must not drop the
    /// rest of the row's properties — in ANY decoder.
    #[test]
    fn unknown_tag_is_skipped_not_row_fatal() {
        let mut blob = Vec::new();
        blob.extend_from_slice(&2u16.to_le_bytes());
        // Entry 1: artificially-unknown tag 200 with a 3-byte payload.
        blob.extend_from_slice(&key(1).as_u64().to_le_bytes());
        blob.push(200);
        blob.extend_from_slice(&3u32.to_le_bytes());
        blob.extend_from_slice(b"xyz");
        // Entry 2: a good string property AFTER the unknown one.
        encode_value(&mut blob, key(2), &Value::String("kept".into()));

        // Owned full decode keeps the good property.
        assert_eq!(
            decode_blob(&blob),
            vec![(key(2), Value::String("kept".into()))]
        );
        // Key scan finds it past the unknown tag.
        assert_eq!(scan_blob(&blob, key(2)), Some(Value::String("kept".into())));
        // Borrowed visitor sees it too.
        let mut seen = Vec::new();
        try_for_each_borrowed(&blob, |k, bv| -> Result<(), ()> {
            seen.push((k, bv.to_value()));
            Ok(())
        })
        .unwrap();
        assert_eq!(seen, vec![(key(2), Value::String("kept".into()))]);
    }

    #[test]
    fn truncated_tail_ends_scan_gracefully() {
        let entries = vec![
            (key(1), Value::Int64(5)),
            (key(2), Value::String("will-be-torn".into())),
        ];
        let mut blob = blob_of(&entries);
        blob.truncate(blob.len() - 4); // tear the string payload
        assert_eq!(decode_blob(&blob), vec![(key(1), Value::Int64(5))]);
        let mut seen = Vec::new();
        try_for_each_borrowed(&blob, |k, bv| -> Result<(), ()> {
            seen.push((k, bv.to_value()));
            Ok(())
        })
        .unwrap();
        assert_eq!(seen, vec![(key(1), Value::Int64(5))]);
    }

    #[test]
    fn invalid_utf8_string_decodes_lossily_in_both_paths() {
        let mut blob = Vec::new();
        blob.extend_from_slice(&1u16.to_le_bytes());
        blob.extend_from_slice(&key(1).as_u64().to_le_bytes());
        blob.push(TAG_STRING);
        blob.extend_from_slice(&2u32.to_le_bytes());
        blob.extend_from_slice(&[0xFF, 0xFE]); // invalid UTF-8
        let owned = decode_blob(&blob);
        assert_eq!(owned.len(), 1);
        assert!(matches!(&owned[0].1, Value::String(s) if s.contains('\u{FFFD}')));
        let mut seen = Vec::new();
        try_for_each_borrowed(&blob, |k, bv| -> Result<(), ()> {
            seen.push((k, bv.to_value()));
            Ok(())
        })
        .unwrap();
        assert_eq!(seen, owned);
    }

    #[test]
    fn transient_values_encode_as_null_tag() {
        // Query-time values remain transient and therefore use the null
        // tag in the overflow property bag.
        let mut buf = Vec::new();
        encode_value(&mut buf, key(1), &Value::NodeRef(7));
        assert_eq!(buf[8], TAG_NULL);
        assert_eq!(buf.len(), 9); // key + tag only, no payload
    }
}
