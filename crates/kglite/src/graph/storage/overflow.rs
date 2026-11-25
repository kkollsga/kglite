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
//! | 7   | Timestamp  | 8 bytes i64 LE (whole seconds, read-only)|
//! | 8   | Reserved   | u32 LE length + retired pre-0.14 payload |
//! | 9   | List       | u32 LE length + Postcard(`Vec<Value>`)   |
//! | 10  | Map        | u32 LE length + Postcard(`BTreeMap`)     |
//! | 11  | Timestamp  | u32 LE length (12) + i64 LE seconds + u32 LE nanoseconds |
//! | 12  | ZonedDateTime | u32 LE length (16) + i64 LE epoch seconds + u32 LE nanoseconds + i32 LE offset seconds |
//! | 13  | LocalTime  | u32 LE length (8) + u64 LE nanoseconds of day |
//! | 14  | Time       | u32 LE length (12) + u64 LE nanoseconds of day + i32 LE offset seconds |
//! | 15  | Duration   | u32 LE length (20) + i32 months + i32 days + i64 seconds + i32 nanoseconds, all LE |
//! | 16  | ZonedDateTime in a named zone | u32 LE length (16 + n) + the tag-12 payload + n UTF-8 bytes of the IANA zone name |
//!
//! Tag 16 stores the zone by **name**, so the file does not depend on the tz
//! database's internal numbering. A reader without the zone (no `iana-zones`
//! feature, or a name its database lacks) reads the stored offset as a fixed
//! offset.
//!
//! Tag 15 (`Duration`) is defensive: the bag is written only by the N-Triples
//! builder, the streaming subgraph writer and legacy migration, none of which
//! produces a duration today. Before it existed the encoder wrote a `Duration`
//! as Null, so any future writer that routed one here would have lost it.
//!
//! Tag 7 truncated a timestamp to whole seconds. It is still decoded (a bag an
//! earlier build wrote keeps its values) but never written: tag 11 carries the
//! full `NaiveDateTime`, sub-microsecond digits and a leap second included.
//!
//! **Forward-compat rule:** any *future* tag MUST use a `u32 LE`
//! length prefix + payload, so that older readers can skip an unknown
//! tag without losing the rest of the row (see [`skip_value`]).
//!
//! ## Deliberately non-storable values
//!
//! `Node`, `Relationship`, and `Path` are query-result-time values, while
//! `NodeRef` is query-time-only; the encoder writes it as the Null tag. `Point` is encoded as its string form `"lat,lon"` (tag 6),
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
/// A timestamp to the nanosecond; supersedes [`TAG_TIMESTAMP`], which is read
/// but no longer written.
pub const TAG_TIMESTAMP_EXACT: u8 = 11;

/// An instant with an offset: epoch seconds, nanoseconds, then offset seconds.
pub const TAG_ZONED_DATETIME: u8 = 12;
/// A time of day without an offset: nanoseconds since midnight.
pub const TAG_LOCAL_TIME: u8 = 13;
/// A time of day with an offset: nanoseconds since midnight, then offset seconds.
pub const TAG_TIME: u8 = 14;
/// A duration with a sub-second part (and every whole-second one): months, days,
/// seconds, nanoseconds.
pub const TAG_DURATION: u8 = 15;
/// An instant in a named IANA zone: the [`TAG_ZONED_DATETIME`] payload followed
/// by the zone name.
pub const TAG_ZONED_DATETIME_NAMED: u8 = 16;

/// Length of a [`TAG_DURATION`] payload: `i32` + `i32` + `i64` + `i32`.
const DURATION_LEN: usize = 20;

/// Highest tag this build knows how to *decode*. Tags above this are
/// skipped via the length-prefix forward-compat rule (module docs).
pub const MAX_KNOWN_TAG: u8 = TAG_ZONED_DATETIME_NAMED;

/// Whether `read_value` can decode `tag`. Tag 8 is retired: it sits inside
/// `..=MAX_KNOWN_TAG` yet must be skipped, not read.
#[inline]
fn is_decodable_tag(tag: u8) -> bool {
    tag <= MAX_KNOWN_TAG && tag != TAG_LIST
}

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
        Value::Timestamp(dt) => encode_timestamp(buf, dt),
        Value::String(s) => encode_str(buf, s),
        // Point is serialized as its "lat,lon" string form (legacy
        // convention — kept for wire-format stability).
        Value::Point { lat, lon } => encode_str(buf, &format!("{},{}", lat, lon)),
        Value::Duration {
            months,
            days,
            seconds,
            nanos,
        } => encode_duration(buf, *months, *days, *seconds, *nanos),
        Value::LocalTime(n) => encode_local_time(buf, *n),
        Value::Time { nanos, offset_secs } => encode_time(buf, *nanos, *offset_secs),
        Value::ZonedDateTime {
            utc,
            offset_secs,
            zone,
        } => encode_zoned(buf, utc, *offset_secs, *zone),
        // NodeRef is transient and is written as null.
        Value::NodeRef(_) => buf.push(TAG_NULL),
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
        BorrowedValue::Timestamp(dt) => encode_timestamp(buf, dt),
        BorrowedValue::String(s) => encode_str(buf, s),
        BorrowedValue::List(items) => encode_list(buf, items),
        BorrowedValue::Map(entries) => encode_map(buf, entries),
        BorrowedValue::Duration {
            months,
            days,
            seconds,
            nanos,
        } => encode_duration(buf, *months, *days, *seconds, *nanos),
        BorrowedValue::LocalTime(n) => encode_local_time(buf, *n),
        BorrowedValue::Time { nanos, offset_secs } => encode_time(buf, *nanos, *offset_secs),
        BorrowedValue::ZonedDateTime {
            utc,
            offset_secs,
            zone,
        } => encode_zoned(buf, utc, *offset_secs, *zone),
    }
}

/// Length of a [`TAG_LOCAL_TIME`] payload: `u64`.
const LOCAL_TIME_LEN: usize = 8;
/// Length of a [`TAG_TIME`] payload: `u64` + `i32`.
const TIME_LEN: usize = 12;
/// Length of a [`TAG_ZONED_DATETIME`] payload: `i64` + `u32` + `i32`.
const ZONED_LEN: usize = 16;

fn encode_zoned(
    buf: &mut Vec<u8>,
    utc: &chrono::NaiveDateTime,
    offset_secs: i32,
    zone: Option<crate::datatypes::time_zone::ZoneId>,
) {
    let (secs, nanos) = crate::datatypes::zoned_datetime::epoch_parts(*utc);
    let name = zone.map_or("", |zone| zone.name());
    buf.push(if zone.is_some() {
        TAG_ZONED_DATETIME_NAMED
    } else {
        TAG_ZONED_DATETIME
    });
    buf.extend_from_slice(&((ZONED_LEN + name.len()) as u32).to_le_bytes());
    buf.extend_from_slice(&secs.to_le_bytes());
    buf.extend_from_slice(&nanos.to_le_bytes());
    buf.extend_from_slice(&offset_secs.to_le_bytes());
    buf.extend_from_slice(name.as_bytes());
}

/// A [`TAG_ZONED_DATETIME_NAMED`] payload: the fixed form and the zone name. A
/// name this build knows takes its offset from the zone's rules at the instant;
/// one it does not know keeps the stored offset and no zone.
fn decode_zoned_named(
    payload: &[u8],
) -> Option<(
    chrono::NaiveDateTime,
    i32,
    Option<crate::datatypes::time_zone::ZoneId>,
)> {
    if payload.len() <= ZONED_LEN {
        return None;
    }
    let (utc, stored) = decode_zoned(&payload[..ZONED_LEN])?;
    let name = std::str::from_utf8(&payload[ZONED_LEN..]).ok()?;
    Some(match crate::datatypes::time_zone::ZoneId::from_name(name) {
        Some(zone) => (utc, zone.offset_at(utc), Some(zone)),
        None => (utc, stored, None),
    })
}

/// `None` for a payload of the wrong length, an instant out of range or an
/// offset past ±18:00: corrupt, not a value to fabricate.
fn decode_zoned(payload: &[u8]) -> Option<(chrono::NaiveDateTime, i32)> {
    if payload.len() != ZONED_LEN {
        return None;
    }
    let secs = i64::from_le_bytes(payload[0..8].try_into().ok()?);
    let nanos = u32::from_le_bytes(payload[8..12].try_into().ok()?);
    let offset = i32::from_le_bytes(payload[12..16].try_into().ok()?);
    if offset.abs() > crate::datatypes::time_of_day::MAX_OFFSET_SECS {
        return None;
    }
    crate::datatypes::zoned_datetime::from_epoch(secs, nanos).map(|utc| (utc, offset))
}

fn encode_local_time(buf: &mut Vec<u8>, nanos: u64) {
    buf.push(TAG_LOCAL_TIME);
    buf.extend_from_slice(&(LOCAL_TIME_LEN as u32).to_le_bytes());
    buf.extend_from_slice(&nanos.to_le_bytes());
}

fn encode_time(buf: &mut Vec<u8>, nanos: u64, offset_secs: i32) {
    buf.push(TAG_TIME);
    buf.extend_from_slice(&(TIME_LEN as u32).to_le_bytes());
    buf.extend_from_slice(&nanos.to_le_bytes());
    buf.extend_from_slice(&offset_secs.to_le_bytes());
}

/// `None` for a payload of the wrong length or a clock reading past the end of
/// the day: corrupt, not a value to fabricate.
fn decode_local_time(payload: &[u8]) -> Option<u64> {
    if payload.len() != LOCAL_TIME_LEN {
        return None;
    }
    let nanos = u64::from_le_bytes(payload.try_into().ok()?);
    crate::datatypes::time_of_day::is_valid_nanos(nanos).then_some(nanos)
}

/// As [`decode_local_time`], and an offset outside ±18:00 is corrupt too.
fn decode_time(payload: &[u8]) -> Option<(u64, i32)> {
    if payload.len() != TIME_LEN {
        return None;
    }
    let nanos = u64::from_le_bytes(payload[..8].try_into().ok()?);
    let offset = i32::from_le_bytes(payload[8..].try_into().ok()?);
    (crate::datatypes::time_of_day::is_valid_nanos(nanos)
        && offset.abs() <= crate::datatypes::time_of_day::MAX_OFFSET_SECS)
        .then_some((nanos, offset))
}

/// The borrowed value of a length-prefixed temporal payload, `None` when the
/// payload is corrupt for its tag.
fn decode_fixed_temporal(tag: u8, payload: &[u8]) -> Option<BorrowedValue<'static>> {
    Some(match tag {
        TAG_TIMESTAMP_EXACT => BorrowedValue::Timestamp(decode_timestamp_exact(payload)?),
        TAG_DURATION => {
            let (months, days, seconds, nanos) = decode_duration(payload)?;
            BorrowedValue::Duration {
                months,
                days,
                seconds,
                nanos,
            }
        }
        TAG_ZONED_DATETIME => {
            let (utc, offset_secs) = decode_zoned(payload)?;
            BorrowedValue::ZonedDateTime {
                utc,
                offset_secs,
                zone: None,
            }
        }
        TAG_ZONED_DATETIME_NAMED => {
            let (utc, offset_secs, zone) = decode_zoned_named(payload)?;
            BorrowedValue::ZonedDateTime {
                utc,
                offset_secs,
                zone,
            }
        }
        TAG_LOCAL_TIME => BorrowedValue::LocalTime(decode_local_time(payload)?),
        TAG_TIME => {
            let (nanos, offset_secs) = decode_time(payload)?;
            BorrowedValue::Time { nanos, offset_secs }
        }
        _ => return None,
    })
}

fn encode_duration(buf: &mut Vec<u8>, months: i32, days: i32, seconds: i64, nanos: i32) {
    buf.push(TAG_DURATION);
    buf.extend_from_slice(&(DURATION_LEN as u32).to_le_bytes());
    buf.extend_from_slice(&months.to_le_bytes());
    buf.extend_from_slice(&days.to_le_bytes());
    buf.extend_from_slice(&seconds.to_le_bytes());
    buf.extend_from_slice(&nanos.to_le_bytes());
}

/// `None` for a payload of the wrong length or a sub-second part outside
/// `0..1e9`: corrupt, not a value to fabricate.
fn decode_duration(payload: &[u8]) -> Option<(i32, i32, i64, i32)> {
    if payload.len() != DURATION_LEN {
        return None;
    }
    let months = i32::from_le_bytes(payload[0..4].try_into().ok()?);
    let days = i32::from_le_bytes(payload[4..8].try_into().ok()?);
    let seconds = i64::from_le_bytes(payload[8..16].try_into().ok()?);
    let nanos = i32::from_le_bytes(payload[16..20].try_into().ok()?);
    (0..1_000_000_000)
        .contains(&nanos)
        .then_some((months, days, seconds, nanos))
}

/// Length of a [`TAG_TIMESTAMP_EXACT`] payload: seconds (`i64`) + nanoseconds (`u32`).
const TIMESTAMP_EXACT_LEN: usize = 12;

fn encode_timestamp(buf: &mut Vec<u8>, dt: &chrono::NaiveDateTime) {
    use chrono::Timelike;
    buf.push(TAG_TIMESTAMP_EXACT);
    buf.extend_from_slice(&(TIMESTAMP_EXACT_LEN as u32).to_le_bytes());
    // `timestamp()` is the floor to the second, and `nanosecond()` the
    // remainder (at most 1_999_999_999: a leap second is stored as itself).
    buf.extend_from_slice(&dt.and_utc().timestamp().to_le_bytes());
    buf.extend_from_slice(&dt.nanosecond().to_le_bytes());
}

fn decode_timestamp_exact(payload: &[u8]) -> Option<chrono::NaiveDateTime> {
    if payload.len() != TIMESTAMP_EXACT_LEN {
        return None;
    }
    let secs = i64::from_le_bytes(payload[..8].try_into().ok()?);
    let nanos = u32::from_le_bytes(payload[8..].try_into().ok()?);
    chrono::DateTime::from_timestamp(secs, nanos).map(|dt| dt.naive_utc())
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
fn encode_map(buf: &mut Vec<u8>, entries: &crate::datatypes::PropMap) {
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
        TAG_TIMESTAMP_EXACT => {
            let bytes = read_len_prefixed(blob, pos)?;
            Some(Value::Timestamp(decode_timestamp_exact(bytes)?))
        }
        TAG_DURATION => {
            let bytes = read_len_prefixed(blob, pos)?;
            let (months, days, seconds, nanos) = decode_duration(bytes)?;
            Some(Value::Duration {
                months,
                days,
                seconds,
                nanos,
            })
        }
        TAG_ZONED_DATETIME => {
            let bytes = read_len_prefixed(blob, pos)?;
            let (utc, offset_secs) = decode_zoned(bytes)?;
            Some(Value::zoned(utc, offset_secs))
        }
        TAG_ZONED_DATETIME_NAMED => {
            let bytes = read_len_prefixed(blob, pos)?;
            let (utc, offset_secs, zone) = decode_zoned_named(bytes)?;
            Some(Value::ZonedDateTime {
                utc,
                offset_secs,
                zone,
            })
        }
        TAG_LOCAL_TIME => {
            let bytes = read_len_prefixed(blob, pos)?;
            Some(Value::LocalTime(decode_local_time(bytes)?))
        }
        TAG_TIME => {
            let bytes = read_len_prefixed(blob, pos)?;
            let (nanos, offset_secs) = decode_time(bytes)?;
            Some(Value::Time { nanos, offset_secs })
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
            .map_err(note_undecodable)
            .ok()?;
            Some(Value::List(items))
        }
        TAG_MAP_POSTCARD => {
            let bytes = read_len_prefixed(blob, pos)?;
            let limits = crate::serde_codec::DecodeLimits::new(u32::MAX as u64, bytes.len() as u64);
            let entries: crate::datatypes::PropMap = crate::serde_codec::decode_exact_with(
                crate::serde_codec::CodecVersion::PostcardV1,
                bytes,
                bytes.len() as u64,
                limits,
            )
            .map_err(note_undecodable)
            .ok()?;
            Some(Value::Map(entries))
        }
        _ => None,
    }
}

/// Report a list or map that could not be decoded from a row blob.
///
/// The row readers return `Option`/`Vec` and have no error channel, so an
/// undecodable value reads as absent. A value nested past the ceiling is a
/// real stored value the engine now refuses, not noise, so say so once per
/// process instead of letting the property vanish silently.
fn note_undecodable(error: crate::serde_codec::CodecError) {
    static WARNED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if error
        .to_string()
        .contains(&crate::datatypes::value_depth::depth_error())
        && !WARNED.swap(true, std::sync::atomic::Ordering::Relaxed)
    {
        eprintln!(
            "[kglite] WARN a stored list or map is nested deeper than {} levels and reads as \
             absent; rebuild the graph from source with the value flattened",
            crate::datatypes::value_depth::MAX_VALUE_DEPTH
        );
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
        if entry_key == target && is_decodable_tag(tag) {
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
        if is_decodable_tag(tag) {
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
            TAG_TIMESTAMP_EXACT
            | TAG_ZONED_DATETIME
            | TAG_ZONED_DATETIME_NAMED
            | TAG_DURATION
            | TAG_LOCAL_TIME
            | TAG_TIME => {
                let Some(bytes) = read_len_prefixed(blob, pos) else {
                    return Some(Ok(()));
                };
                // A payload that is not its tag's value is corrupt: end the scan
                // without fabricating one, as a corrupt list or map does.
                let Some(value) = decode_fixed_temporal(tag, bytes) else {
                    return Some(Ok(()));
                };
                if let Err(e) = f(key, value) {
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
                    Err(e) => {
                        note_undecodable(e);
                        return Some(Ok(())); // corrupt payload — stop
                    }
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
                let entries: crate::datatypes::PropMap =
                    match crate::serde_codec::decode_exact_with(
                        crate::serde_codec::CodecVersion::PostcardV1,
                        bytes,
                        bytes.len() as u64,
                        limits,
                    ) {
                        Ok(v) => v,
                        Err(e) => {
                            note_undecodable(e);
                            return Some(Ok(()));
                        }
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

/// Whether any entry of a row blob decodes to a value `hit` accepts, visiting
/// only the entries that can nest other values (List, Map).
///
/// Every other decodable tag yields a scalar, and the encoder has written a
/// top-level `NodeRef` as [`TAG_NULL`] since 0.13.0, so a value holding a
/// `NodeRef` can come out of a blob only inside a List or Map payload. The
/// retired [`TAG_LIST`] and unknown tags are never decoded ([`decode_blob`]),
/// so nothing a reader sees can come from them either.
pub(crate) fn blob_any_nested_value(blob: &[u8], mut hit: impl FnMut(&Value) -> bool) -> bool {
    for_each_raw(blob, |_, tag, blob, pos| {
        if tag == TAG_LIST_POSTCARD || tag == TAG_MAP_POSTCARD {
            match read_value(blob, pos, tag) {
                Some(value) if hit(&value) => return Some(true),
                Some(_) => {}
                None => return Some(false), // truncated payload ends the row
            }
        } else if !skip_value(blob, pos, tag) {
            return Some(false);
        }
        None
    })
    .unwrap_or(false)
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

/// Append a row while retaining selected entries byte-for-byte, including future tags.
pub(crate) fn append_filtered_blob(
    blob: &[u8],
    output: &mut Vec<u8>,
    mut keep: impl FnMut(InternedKey) -> bool,
) {
    if blob.is_empty() {
        return;
    }
    let header = output.len();
    output.extend_from_slice(&0u16.to_le_bytes());
    let mut count = 0u16;
    for_each_raw(blob, |key, tag, bytes, pos| {
        let start = *pos - 9;
        if !skip_value(bytes, pos, tag) {
            return Some(());
        }
        if keep(InternedKey::from_u64(key)) {
            output.extend_from_slice(&bytes[start..*pos]);
            count += 1;
        }
        None
    });
    output[header..header + 2].copy_from_slice(&count.to_le_bytes());
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
                Value::Map(crate::datatypes::PropMap::from_iter([
                    ("count", Value::Int64(2)),
                    (
                        "nested",
                        Value::List(vec![Value::Map(crate::datatypes::PropMap::from_iter([(
                            "ok",
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

    fn timestamps_under_test() -> Vec<chrono::NaiveDateTime> {
        let day = NaiveDate::from_ymd_opt(2010, 1, 1).unwrap();
        vec![
            day.and_hms_micro_opt(0, 0, 1, 250_000).unwrap(),
            day.and_hms_nano_opt(0, 0, 1, 250_000_123).unwrap(),
            // A leap second is stored as itself, not folded into the next second.
            NaiveDate::from_ymd_opt(2016, 12, 31)
                .unwrap()
                .and_hms_nano_opt(23, 59, 59, 1_500_000_000)
                .unwrap(),
            // Before the epoch, with a fraction: the seconds are a floor.
            NaiveDate::from_ymd_opt(1969, 12, 31)
                .unwrap()
                .and_hms_micro_opt(23, 59, 59, 999_999)
                .unwrap(),
            NaiveDate::from_ymd_opt(1, 1, 1)
                .unwrap()
                .and_hms_opt(0, 0, 0)
                .unwrap(),
            NaiveDate::from_ymd_opt(9999, 12, 31)
                .unwrap()
                .and_hms_nano_opt(23, 59, 59, 999_999_999)
                .unwrap(),
        ]
    }

    /// The bag used to store a timestamp as whole seconds, dropping every
    /// microsecond of a value that a typed column would have kept.
    #[test]
    fn a_timestamp_keeps_its_fraction_through_every_decode_path() {
        for ts in timestamps_under_test() {
            let entries = vec![
                (key(1), Value::Timestamp(ts)),
                (key(2), Value::String("after".into())),
            ];
            let blob = blob_of(&entries);
            assert_eq!(decode_blob(&blob), entries, "{ts}");
            assert_eq!(scan_blob(&blob, key(1)), Some(Value::Timestamp(ts)), "{ts}");
            let mut seen = Vec::new();
            try_for_each_borrowed(&blob, |k, bv| -> Result<(), ()> {
                seen.push((k, bv.to_value()));
                Ok(())
            })
            .unwrap();
            assert_eq!(seen, entries, "{ts}");
            let mut borrowed = Vec::new();
            encode_value_borrowed(&mut borrowed, key(1), &BorrowedValue::Timestamp(ts));
            let mut owned = Vec::new();
            encode_value(&mut owned, key(1), &Value::Timestamp(ts));
            assert_eq!(borrowed, owned, "{ts}");
            assert_eq!(owned[8], TAG_TIMESTAMP_EXACT, "no tag-7 write survives");
        }
    }

    /// A bag an earlier build wrote holds tag 7 (whole seconds); it still reads,
    /// and the entries after it are not lost.
    #[test]
    fn a_whole_second_timestamp_written_by_an_earlier_build_still_reads() {
        let mut blob = Vec::new();
        blob.extend_from_slice(&2u16.to_le_bytes());
        blob.extend_from_slice(&key(1).as_u64().to_le_bytes());
        blob.push(TAG_TIMESTAMP);
        blob.extend_from_slice(&1_262_304_001i64.to_le_bytes());
        encode_value(&mut blob, key(2), &Value::Int64(7));
        let expected = NaiveDate::from_ymd_opt(2010, 1, 1)
            .unwrap()
            .and_hms_opt(0, 0, 1)
            .unwrap();
        let want = vec![
            (key(1), Value::Timestamp(expected)),
            (key(2), Value::Int64(7)),
        ];
        assert_eq!(decode_blob(&blob), want);
        assert_eq!(scan_blob(&blob, key(1)), Some(Value::Timestamp(expected)));
        let mut seen = Vec::new();
        try_for_each_borrowed(&blob, |k, bv| -> Result<(), ()> {
            seen.push((k, bv.to_value()));
            Ok(())
        })
        .unwrap();
        assert_eq!(seen, want);
    }

    #[test]
    fn a_malformed_exact_timestamp_ends_the_scan_without_inventing_a_value() {
        let mut blob = Vec::new();
        blob.extend_from_slice(&2u16.to_le_bytes());
        blob.extend_from_slice(&key(1).as_u64().to_le_bytes());
        blob.push(TAG_TIMESTAMP_EXACT);
        blob.extend_from_slice(&5u32.to_le_bytes());
        blob.extend_from_slice(&[1, 2, 3, 4, 5]);
        encode_value(&mut blob, key(2), &Value::Int64(7));
        assert_eq!(decode_blob(&blob), Vec::new());
        assert_eq!(scan_blob(&blob, key(1)), None);
        let mut seen = 0;
        try_for_each_borrowed(&blob, |_, _| -> Result<(), ()> {
            seen += 1;
            Ok(())
        })
        .unwrap();
        assert_eq!(seen, 0);
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
        let entries = crate::datatypes::PropMap::from_iter([
            ("name", Value::String("map".into())),
            (
                "items",
                Value::List(vec![Value::Map(crate::datatypes::PropMap::from_iter([(
                    "answer",
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
    fn filtered_rows_preserve_retained_encoded_entries() {
        let retained = [
            (key(1), Value::String("Grüße\nkept".into())),
            (key(3), Value::List(vec![Value::Int64(9007199254740993)])),
        ];
        let mut blob = blob_of(&[
            retained[0].clone(),
            (key(2), Value::String("removed".into())),
            retained[1].clone(),
        ]);
        // A length-prefixed future entry follows the existing forward-compatible format.
        blob[..2].copy_from_slice(&4u16.to_le_bytes());
        let mut future = key(4).as_u64().to_le_bytes().to_vec();
        future.push(MAX_KNOWN_TAG + 1);
        future.extend_from_slice(&3u32.to_le_bytes());
        future.extend_from_slice(b"new");
        blob.extend_from_slice(&future);
        let mut output = Vec::new();
        append_filtered_blob(&blob, &mut output, |candidate| candidate != key(2));
        let mut expected = blob_of(&retained);
        expected[..2].copy_from_slice(&3u16.to_le_bytes());
        expected.extend_from_slice(&future);
        assert_eq!(output, expected);
        assert_eq!(decode_blob(&output), retained);
        let mut unchanged = Vec::new();
        append_filtered_blob(&blob, &mut unchanged, |_| true);
        assert_eq!(unchanged, blob);
        let mut empty = Vec::new();
        append_filtered_blob(&blob, &mut empty, |_| false);
        assert_eq!(empty, 0u16.to_le_bytes());
    }

    /// A duration is a persistable value: it used to be written as Null.
    #[test]
    fn durations_round_trip_through_every_reader() {
        let durations = [
            Value::Duration {
                months: -1,
                days: 2,
                seconds: -3,
                nanos: 0,
            },
            Value::Duration {
                months: i32::MAX,
                days: i32::MIN,
                seconds: i64::MIN,
                nanos: 999_999_999,
            },
        ];
        for (i, d) in durations.iter().enumerate() {
            let entries = vec![
                (key(1), Value::Int64(9)),
                (key(2), d.clone()),
                (key(3), Value::Null),
            ];
            let blob = blob_of(&entries);
            assert_eq!(decode_blob(&blob), entries, "case {i}: decode_blob");
            assert_eq!(
                scan_blob(&blob, key(2)),
                Some(d.clone()),
                "case {i}: scan_blob"
            );
            let mut seen = Vec::new();
            try_for_each_borrowed::<_, ()>(&blob, |k, v| {
                seen.push((k, v.to_value()));
                Ok(())
            })
            .unwrap();
            assert_eq!(seen, entries, "case {i}: borrowed visit");
            // The borrowed encoder produces the same bytes as the owned one.
            let mut owned = Vec::new();
            encode_value(&mut owned, key(2), d);
            let mut borrowed = Vec::new();
            encode_value_borrowed(&mut borrowed, key(2), &BorrowedValue::of(d).unwrap());
            assert_eq!(owned, borrowed, "case {i}");
            assert_eq!(owned[8], TAG_DURATION);
            assert_eq!(owned.len(), 8 + 1 + 4 + DURATION_LEN);
        }
    }

    /// A tag past `MAX_KNOWN_TAG` is length-prefixed, so a reader that cannot
    /// decode it steps over the one entry and keeps the rest of the row; tag 15
    /// with a corrupt payload ends the scan rather than fabricating a duration.
    #[test]
    fn unknown_temporal_tags_are_skipped_and_a_corrupt_duration_is_not_invented() {
        let keep = [(key(1), Value::Int64(1)), (key(3), Value::Int64(3))];
        let reserved = MAX_KNOWN_TAG + 1;
        {
            let mut blob = blob_of(&[keep[0].clone()]);
            blob[..2].copy_from_slice(&3u16.to_le_bytes());
            blob.extend_from_slice(&key(2).as_u64().to_le_bytes());
            blob.push(reserved);
            blob.extend_from_slice(&5u32.to_le_bytes());
            blob.extend_from_slice(b"abcde");
            encode_value(&mut blob, key(3), &keep[1].1);
            assert_eq!(decode_blob(&blob), keep, "tag {reserved}");
            assert_eq!(scan_blob(&blob, key(3)), Some(Value::Int64(3)));
            assert_eq!(scan_blob(&blob, key(2)), None);
        }
        // Wrong-length and out-of-range payloads decode to nothing.
        for payload in [vec![0u8; 19], {
            let mut p = vec![0u8; 16];
            p.extend_from_slice(&1_000_000_000i32.to_le_bytes());
            p
        }] {
            let mut blob = 1u16.to_le_bytes().to_vec();
            blob.extend_from_slice(&key(1).as_u64().to_le_bytes());
            blob.push(TAG_DURATION);
            blob.extend_from_slice(&(payload.len() as u32).to_le_bytes());
            blob.extend_from_slice(&payload);
            assert_eq!(decode_blob(&blob), vec![]);
        }
    }

    /// A named-zone instant is stored under tag 16 by NAME: it survives every
    /// reader with its zone, and a bag whose name the build does not know reads as
    /// the stored fixed offset (this is also what a build without `iana-zones`
    /// does with every tag-16 entry).
    #[cfg(feature = "iana-zones")]
    #[test]
    fn a_named_zone_round_trips_by_name_through_every_reader() {
        let utc = crate::datatypes::zoned_datetime::from_epoch(1_437_511_232, 142_000_000).unwrap();
        let zone = crate::datatypes::time_zone::ZoneId::from_name("Europe/Oslo").unwrap();
        let value = Value::zoned_in(utc, zone);
        let entries = vec![(key(1), Value::Int64(9)), (key(2), value.clone())];
        let blob = blob_of(&entries);
        let zone_of = |v: &Value| match v {
            Value::ZonedDateTime {
                offset_secs, zone, ..
            } => Some((*offset_secs, zone.map(|z| z.name()))),
            _ => None,
        };
        let want = Some((7200, Some("Europe/Oslo")));
        assert_eq!(zone_of(&scan_blob(&blob, key(2)).unwrap()), want);
        assert_eq!(zone_of(&decode_blob(&blob)[1].1), want);
        let mut seen = None;
        try_for_each_borrowed::<_, ()>(&blob, |k, b| {
            if k == key(2) {
                seen = zone_of(&b.to_value());
            }
            Ok(())
        })
        .unwrap();
        assert_eq!(seen, want);
        let mut owned = Vec::new();
        encode_value(&mut owned, key(2), &value);
        let mut borrowed = Vec::new();
        encode_value_borrowed(&mut borrowed, key(2), &BorrowedValue::of(&value).unwrap());
        assert_eq!(owned, borrowed);
        // The tag is 16 and the payload is the tag-12 payload plus the name.
        assert_eq!(owned[8], TAG_ZONED_DATETIME_NAMED);
        assert_eq!(&owned[13 + ZONED_LEN..], b"Europe/Oslo");
        // A name this database lacks keeps the stored offset and no zone.
        let mut forged = blob.clone();
        let at = forged.len() - 1;
        forged[at] = b'x';
        assert_eq!(
            zone_of(&scan_blob(&forged, key(2)).unwrap()),
            Some((7200, None))
        );
    }

    /// A tag-16 entry read by a build with no tz database is its fixed offset.
    #[cfg(not(feature = "iana-zones"))]
    #[test]
    fn a_named_zone_entry_reads_as_its_fixed_offset_without_the_feature() {
        let utc = crate::datatypes::zoned_datetime::from_epoch(1_437_511_232, 142_000_000).unwrap();
        let mut fixed = Vec::new();
        encode_value(&mut fixed, key(2), &Value::zoned(utc, 7200));
        // Rewrite the tag-12 entry into the tag-16 form with a name appended.
        let mut named = fixed[..8].to_vec();
        named.push(TAG_ZONED_DATETIME_NAMED);
        named.extend_from_slice(&((ZONED_LEN + 11) as u32).to_le_bytes());
        named.extend_from_slice(&fixed[13..]);
        named.extend_from_slice(b"Europe/Oslo");
        let mut blob = 1u16.to_le_bytes().to_vec();
        blob.extend_from_slice(&named);
        let Some(Value::ZonedDateTime {
            offset_secs,
            zone,
            utc: got,
        }) = scan_blob(&blob, key(2))
        else {
            panic!("tag 16 decodes a zoned datetime");
        };
        assert_eq!((got, offset_secs, zone), (utc, 7200, None));
    }

    /// LocalTime and Time are persistable: each round-trips through every
    /// reader, the borrowed encoder matches the owned one byte for byte, and a
    /// payload past the end of the day (or an offset past 18 h) ends the scan.
    #[test]
    fn times_of_day_round_trip_through_every_reader() {
        let values = [
            Value::LocalTime(0),
            Value::LocalTime(86_399_999_999_999),
            Value::Time {
                nanos: 36_000_500_000_000,
                offset_secs: 3_600,
            },
            Value::Time {
                nanos: 1,
                offset_secs: -64_800,
            },
            Value::ZonedDateTime {
                utc: crate::datatypes::zoned_datetime::from_epoch(1_437_511_232, 142_000_001)
                    .unwrap(),
                offset_secs: 3_600,
                zone: None,
            },
            Value::ZonedDateTime {
                utc: crate::datatypes::zoned_datetime::from_epoch(-62_135_596_800, 0).unwrap(),
                offset_secs: -64_800,
                zone: None,
            },
        ];
        for (i, v) in values.iter().enumerate() {
            let entries = vec![
                (key(1), Value::Int64(9)),
                (key(2), v.clone()),
                (key(3), Value::Null),
            ];
            let blob = blob_of(&entries);
            assert_eq!(decode_blob(&blob), entries, "case {i}: decode_blob");
            assert_eq!(scan_blob(&blob, key(2)), Some(v.clone()), "case {i}");
            let mut seen = Vec::new();
            try_for_each_borrowed::<_, ()>(&blob, |k, b| {
                seen.push((k, b.to_value()));
                Ok(())
            })
            .unwrap();
            assert_eq!(seen, entries, "case {i}: borrowed visit");
            let mut owned = Vec::new();
            encode_value(&mut owned, key(2), v);
            let mut borrowed = Vec::new();
            encode_value_borrowed(&mut borrowed, key(2), &BorrowedValue::of(v).unwrap());
            assert_eq!(owned, borrowed, "case {i}");
        }
        for (tag, payload) in [
            (TAG_LOCAL_TIME, 86_400_000_000_000u64.to_le_bytes().to_vec()),
            (TAG_LOCAL_TIME, vec![0u8; 7]),
            (TAG_TIME, {
                let mut p = 0u64.to_le_bytes().to_vec();
                p.extend_from_slice(&64_801i32.to_le_bytes());
                p
            }),
            (TAG_ZONED_DATETIME, vec![0u8; 15]),
            (TAG_ZONED_DATETIME, {
                let mut p = 0i64.to_le_bytes().to_vec();
                p.extend_from_slice(&0u32.to_le_bytes());
                p.extend_from_slice(&64_801i32.to_le_bytes());
                p
            }),
            (TAG_ZONED_DATETIME, {
                let mut p = 0i64.to_le_bytes().to_vec();
                p.extend_from_slice(&1_000_000_000u32.to_le_bytes());
                p.extend_from_slice(&0i32.to_le_bytes());
                p
            }),
        ] {
            let mut blob = 1u16.to_le_bytes().to_vec();
            blob.extend_from_slice(&key(1).as_u64().to_le_bytes());
            blob.push(tag);
            blob.extend_from_slice(&(payload.len() as u32).to_le_bytes());
            blob.extend_from_slice(&payload);
            assert_eq!(decode_blob(&blob), vec![]);
        }
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
