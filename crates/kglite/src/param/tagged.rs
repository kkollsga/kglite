//! Tagged JSON objects for query parameter types JSON has no spelling for.
//!
//! An object with exactly one key, `$date`, `$datetime` or `$duration`, is a
//! typed value rather than a map. The payloads are the shapes
//! [`super::kglite_value_to_json`] renders those types as, so a result cell
//! wrapped in its tag reads back as the value it came from:
//!
//! - `{"$date": "2020-01-01"}` → `Value::DateTime`, parsed as `date()` parses
//! - `{"$datetime": "2020-01-01T08:00:00+02:00"}` → `Value::Timestamp`, parsed
//!   as `datetime()` parses (an offset is applied, normalising to UTC)
//! - `{"$duration": {"months": 0, "days": 1, "seconds": 0}}` → `Value::Duration`
//!   (each field optional, default 0; no other field allowed)
//!
//! Any other object, including one holding a tag key beside other keys, stays
//! an ordinary map.

use crate::datatypes::values::Value;

/// `None` when `map` is not a tagged object; `Some(None)` when it is one whose
/// payload is invalid.
pub(super) fn decode(map: &serde_json::Map<String, serde_json::Value>) -> Option<Option<Value>> {
    if map.len() != 1 {
        return None;
    }
    let (key, payload) = map.iter().next()?;
    Some(match key.as_str() {
        "$date" => payload.as_str().and_then(|text| {
            crate::graph::features::timeseries::parse_date_query(text)
                .ok()
                .map(|(date, _)| Value::DateTime(date))
        }),
        "$datetime" => payload.as_str().and_then(|text| {
            crate::graph::languages::cypher::executor::scalar_functions::parse_datetime_utc(text)
                .map(Value::Timestamp)
        }),
        "$duration" => decode_duration(payload),
        _ => return None,
    })
}

pub(super) fn decode_duration(payload: &serde_json::Value) -> Option<Value> {
    let fields = payload.as_object()?;
    let (mut months, mut days, mut seconds) = (0i32, 0i32, 0i64);
    for (name, value) in fields {
        let value = value.as_i64()?;
        match name.as_str() {
            "months" => months = i32::try_from(value).ok()?,
            "days" => days = i32::try_from(value).ok()?,
            "seconds" => seconds = value,
            _ => return None,
        }
    }
    Some(Value::Duration {
        months,
        days,
        seconds,
    })
}

#[cfg(test)]
#[path = "tagged_tests.rs"]
mod tests;
