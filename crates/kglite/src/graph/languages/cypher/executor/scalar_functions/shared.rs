//! Shared constants and free helpers for the scalar-function modules.
use super::super::helpers::format_value_compact;
use crate::datatypes::values::Value;

/// Length of a string argument to `size()` / `length()`.
///
/// Strings are measured in **characters**, not UTF-8 bytes, so the answer
/// agrees with the char-indexed `substring()` / `left()` / `right()`:
/// `size('Tromsø')` is 6, and `substring('Tromsø', size('Tromsø') - 1)` is
/// `'ø'`. Counting bytes made the two families disagree on every
/// non-ASCII string.
///
/// **Every** string is measured this way, including bracket-delimited text,
/// which matches Neo4j's `size(STRING)`. Reporting the element count of the
/// list a `'[...]'` string parses to made `size('[redacted]')` 1 and
/// `size('[]')` 0 — neither the characters nor the bytes of anything the
/// caller wrote. The rest of the legacy collect-as-JSON family (`UNWIND`,
/// indexing, `head`/`last`/`reverse`, `IN`) still coerces that shape; that
/// cutover is tracked separately and does not reach this function.
pub(super) fn string_scalar_length(s: String) -> i64 {
    s.chars().count() as i64
}

/// `toString(value)` — the compact string form, with **null in, null out**.
///
/// Formatting a null produced the *string* `'null'`, which made an absent
/// value indistinguishable from a present one and, being non-null, survived
/// `coalesce(toString(x), 'default')` — the very call that exists to supply
/// a default for a missing value.
pub(super) fn to_string_or_null(value: Value) -> Value {
    match value {
        Value::Null => Value::Null,
        other => Value::String(format_value_compact(&other)),
    }
}

/// `split(original, delimiter)` — the element list, as native `Value`s.
///
/// An **empty delimiter** splits into characters. Rust's `str::split`
/// yields a phantom leading and trailing empty element for that case
/// (`split('a', '')` → `['', 'a', '']`), which is an artefact of the Rust
/// API rather than a Cypher answer. Neo4j's behaviour for an empty
/// delimiter is not specified in its manual, so kglite pins the
/// per-character reading — the same one JavaScript's `String.split('')`
/// gives — and documents it as a dialect note in `CYPHER.md`.
///
/// An empty *original* stays `['']` whatever the delimiter, matching the
/// non-empty-delimiter case (`split('', ',')` → `['']`).
pub(super) fn split_string(s: &str, delim: &str) -> Vec<Value> {
    if s.is_empty() {
        return vec![Value::String(String::new())];
    }
    if delim.is_empty() {
        return s.chars().map(|c| Value::String(c.to_string())).collect();
    }
    s.split(delim)
        .map(|p| Value::String(p.to_string()))
        .collect()
}

/// Shared error suffix when a spatial function arg can't be resolved to a
/// geometry or point. Names the conventional property names that the
/// fallback inference (in `build_node_spatial_data`) accepts so users have
/// a quick fix. Also surfaced from `resolve_spatial` when a node has no
/// registered spatial config and no inferable conventional fields.
pub(super) const SPATIAL_RESOLUTION_HELP: &str =
    "spatial argument did not resolve to a geometry or point. \
Either pass column_types={'<col>': 'geometry'} (or 'location.lat'/'location.lon') during \
add_nodes(), or store the data under a conventional property name (wkt_geometry, geometry, \
geom, or wkt for WKT; latitude+longitude or lat+lon for points).";

/// Recursively convert a parsed `serde_json::Value` into a kglite `Value`.
/// Objects become `Value::Map`, arrays `Value::List`; integers that fit i64
/// stay `Int64`, and every other number becomes `Float64` — including one
/// past i64 (`9223372036854775808` folds to `9.223372036854776e18`). This is
/// the tolerant policy; query parameters are the strict path.
///
/// **A number outside finite `f64` never reaches here under the shipped
/// features.** Without `arbitrary_precision`, serde_json refuses `1e400` at
/// parse time ("number out of range"), so `parse_json()` takes its
/// invalid-JSON branch and yields `Null` for the *whole document*, not for
/// the one field — pinned by `non_finite_number_token_fails_the_parse` below
/// and by `tests/test_cypher_parse_json.py`. The `is_finite` filter guards
/// the case where a downstream crate turns `arbitrary_precision` on through
/// feature unification and such a token does get parsed.
///
/// Backs the `parse_json()` Cypher function.
pub(super) fn json_to_value(j: &serde_json::Value) -> Value {
    match j {
        serde_json::Value::Null => Value::Null,
        serde_json::Value::Bool(b) => Value::Boolean(*b),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Value::Int64(i)
            } else {
                n.as_f64()
                    .filter(|value| value.is_finite())
                    .map_or(Value::Null, Value::Float64)
            }
        }
        serde_json::Value::String(s) => Value::String(s.clone()),
        serde_json::Value::Array(a) => Value::List(a.iter().map(json_to_value).collect()),
        serde_json::Value::Object(o) => Value::Map(
            o.iter()
                .map(|(k, v)| (k.clone(), json_to_value(v)))
                .collect(),
        ),
    }
}

/// One parsed ISO-8601 datetime: the wall-clock reading exactly as written,
/// plus the UTC offset the string carried (`None` when it carried none).
pub(super) struct ParsedIsoDateTime {
    pub local: chrono::NaiveDateTime,
    pub offset: Option<chrono::FixedOffset>,
}

impl ParsedIsoDateTime {
    /// The reading normalised to UTC. Identical to `local` for a zone-less
    /// string, which is why a naive input round-trips unchanged.
    pub fn utc(&self) -> chrono::NaiveDateTime {
        match self.offset {
            Some(offset) => self.local - offset,
            None => self.local,
        }
    }
}

/// Parse an ISO-8601 datetime string, retaining fractional seconds.
///
/// Accepts, in order: an offset-bearing RFC 3339 stamp (`…Z`, `…+02:00`),
/// a zone-less `YYYY-MM-DDTHH:MM:SS` with optional fractional seconds, a
/// zone-less `YYYY-MM-DDTHH:MM`, and finally a bare date (midnight).
/// The returned Timestamp retains the precision represented by Chrono.
///
/// **The bare-date fallback only fires for a string with no time part.** It
/// used to be reached by splitting any input on `'T'` and re-parsing the date
/// half, so every form this function did not recognise — every zoned stamp
/// included — silently answered midnight: `datetime('2024-01-15T10:30:00Z')`
/// returned `2024-01-15T00:00:00`, dropping both the time and the zone with no
/// diagnostic. An unrecognised stamp that *has* a time part now returns `None`,
/// which the callers surface as `Null`, matching their documented
/// Null-on-unparseable contract.
pub(super) fn parse_iso_datetime(s: &str) -> Option<ParsedIsoDateTime> {
    let trimmed = s.trim();

    if let Ok(zoned) = chrono::DateTime::parse_from_rfc3339(trimmed) {
        return Some(ParsedIsoDateTime {
            local: zoned.naive_local(),
            offset: Some(*zoned.offset()),
        });
    }
    // chrono's `%Y` refuses an *unsigned* year outside 0..=9999 but accepts a
    // signed one, and `NaiveDateTime` represents far more than four digits of
    // year. Retry a wide bare year with the sign it wants rather than reject a
    // representable stamp — `datetime('10000-01-01T00:00:00')` is a real value.
    let signed;
    let widened = match trimmed.split_once('-') {
        Some((year, _)) if year.len() > 4 && year.chars().all(|c| c.is_ascii_digit()) => {
            signed = format!("+{trimmed}");
            Some(signed.as_str())
        }
        _ => None,
    };
    for candidate in [Some(trimmed), widened].into_iter().flatten() {
        for format in ["%Y-%m-%dT%H:%M:%S%.f", "%Y-%m-%dT%H:%M"] {
            if let Ok(naive) = chrono::NaiveDateTime::parse_from_str(candidate, format) {
                return Some(ParsedIsoDateTime {
                    local: naive,
                    offset: None,
                });
            }
        }
    }
    if trimmed.contains('T') {
        return None;
    }
    crate::graph::features::timeseries::parse_date_query(trimmed)
        .ok()
        .and_then(|(date, _)| date.and_hms_opt(0, 0, 0))
        .map(|local| ParsedIsoDateTime {
            local,
            offset: None,
        })
}

/// Which wall-clock "now" shape a `local*`/`time` function produces.
/// DateTime emits a Timestamp; time-only forms emit strings because
/// KGLite has no standalone time-of-day Value variant.
#[derive(Clone, Copy)]
pub(super) enum LocalTemporalKind {
    /// `localdatetime()` → `YYYY-MM-DDTHH:MM:SS` (no offset).
    DateTime,
    /// `localtime()` / `time()` → `HH:MM:SS`.
    Time,
}

/// Advance the thread-local xorshift64 PRNG one step and return the
/// raw 64-bit state. Shared by `rand()`/`random()` and `randomUUID()`.
///
/// Seeded once per thread from SystemTime mixed with a monotonic
/// per-thread counter; subsequent calls just advance the state. Avoids
/// per-call `SystemTime::now()` overhead and guarantees distinct values
/// within a tight per-row loop. The counter splat ensures parallel
/// rayon workers don't collide on the same nanosecond.
pub(super) fn next_random_u64() -> u64 {
    use std::cell::Cell;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::SystemTime;
    static THREAD_COUNTER: AtomicU64 = AtomicU64::new(0);
    thread_local! {
        static XORSHIFT_STATE: Cell<u64> = {
            let nanos = SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos() as u64;
            let counter = THREAD_COUNTER.fetch_add(1, Ordering::Relaxed);
            // Mix counter via splitmix64-ish avalanche so adjacent
            // thread IDs produce well-separated seeds.
            let mut seed = nanos.wrapping_add(counter.wrapping_mul(0x9E37_79B9_7F4A_7C15));
            seed ^= seed >> 30;
            seed = seed.wrapping_mul(0xBF58_476D_1CE4_E5B9);
            seed ^= seed >> 27;
            seed = seed.wrapping_mul(0x94D0_49BB_1331_11EB);
            seed ^= seed >> 31;
            Cell::new(seed | 1)
        };
    }
    XORSHIFT_STATE.with(|state| {
        let mut x = state.get();
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        state.set(x);
        x
    })
}

/// Draw 128 random bits as two u64 halves (for `randomUUID()`).
pub(super) fn next_random_u128_halves() -> (u64, u64) {
    (next_random_u64(), next_random_u64())
}

/// Coerce a temporal Value to `NaiveDateTime` for cross-type temporal
/// arithmetic (`date_diff`, `duration.between`). A date-only `DateTime`
/// is treated as midnight, so mixing `date()` and `datetime()` operands
/// works. Returns `None` for non-temporal values.
pub(super) fn coerce_naive_datetime(v: &Value) -> Option<chrono::NaiveDateTime> {
    match v {
        Value::Timestamp(dt) => Some(*dt),
        Value::DateTime(d) => d.and_hms_opt(0, 0, 0),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::json_to_value;
    use crate::datatypes::values::Value;

    /// The premise of `json_to_value`'s doc: a non-finite number token is
    /// refused by the parser, so the `is_finite` filter is unreachable and
    /// `parse_json()` nulls the whole document. A red here means
    /// `arbitrary_precision` got unified in and the filter is live again —
    /// the doc, not the filter, is what has to change.
    #[test]
    fn non_finite_number_token_fails_the_parse() {
        for text in [r#"{"a": 1e400}"#, "[1e400]", "-1e400"] {
            let parsed = serde_json::from_str::<serde_json::Value>(text);
            assert!(parsed.is_err(), "{text} unexpectedly parsed: {parsed:?}");
        }
    }

    #[test]
    fn parse_json_numbers_keep_int64_only_when_exact() {
        let parsed: serde_json::Value = serde_json::from_str(
            r#"{"a": 1e308, "c": 9223372036854775807, "d": 9223372036854775808}"#,
        )
        .expect("finite tokens parse");
        let Value::Map(map) = json_to_value(&parsed) else {
            panic!("object must convert to a map");
        };
        assert_eq!(map.get("a"), Some(&Value::Float64(1e308)));
        assert_eq!(map.get("c"), Some(&Value::Int64(i64::MAX)));
        // Past i64 the tolerant converter folds to f64 rather than nulling.
        assert_eq!(map.get("d"), Some(&Value::Float64(9.223372036854776e18)));
    }
}
