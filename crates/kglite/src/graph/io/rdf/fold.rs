//! Typed-literal → [`Value`] coercion for the RDF fold.
//!
//! Maps an XSD (and GeoSPARQL `wktLiteral`) datatype IRI + lexical form
//! to the closest native [`Value`] variant, falling back to
//! `Value::String` whenever a value doesn't parse cleanly. This is the
//! RDF-side analogue of the N-Triples loader's `typed_literal_to_value`,
//! kept separate so the RDF loader's richer date/timestamp/point
//! coercion never disturbs the Wikidata-tuned path.

use crate::datatypes::values::Value;
use chrono::{DateTime, NaiveDate, NaiveDateTime};

// XSD namespace + the leaf datatypes we special-case.
const XSD: &str = "http://www.w3.org/2001/XMLSchema#";
const GEO_WKT: &str = "http://www.opengis.net/ont/geosparql#wktLiteral";

/// Coerce a typed literal to a native [`Value`]. `datatype_iri` is the
/// full datatype IRI (e.g. `http://www.w3.org/2001/XMLSchema#integer`).
pub(super) fn datatype_to_value(value: &str, datatype_iri: &str) -> Value {
    // Strip the XSD namespace once; everything past it is the local name.
    if let Some(local) = datatype_iri.strip_prefix(XSD) {
        return xsd_to_value(value, local);
    }
    if datatype_iri == GEO_WKT {
        return wkt_to_value(value);
    }
    // xsd:string, rdf:langString, and any unknown datatype → string.
    Value::String(value.to_string())
}

/// Dispatch on the XSD local name (the part after the `#`).
fn xsd_to_value(value: &str, local: &str) -> Value {
    match local {
        // The integer family. xsd:integer plus the bounded / derived
        // integer types all map to i64 when they parse.
        "integer" | "int" | "long" | "short" | "byte" | "nonNegativeInteger"
        | "nonPositiveInteger" | "negativeInteger" | "positiveInteger" | "unsignedLong"
        | "unsignedInt" | "unsignedShort" | "unsignedByte" => value
            .trim_start_matches('+')
            .parse::<i64>()
            .map(Value::Int64)
            .unwrap_or_else(|_| Value::String(value.to_string())),

        // decimal: prefer an exact i64, else f64.
        "decimal" => {
            let v = value.trim_start_matches('+');
            if let Ok(i) = v.parse::<i64>() {
                Value::Int64(i)
            } else if let Ok(f) = v.parse::<f64>() {
                Value::Float64(f)
            } else {
                Value::String(value.to_string())
            }
        }

        "double" | "float" => value
            .parse::<f64>()
            .map(Value::Float64)
            .unwrap_or_else(|_| Value::String(value.to_string())),

        "boolean" => match value {
            "true" | "1" => Value::Boolean(true),
            "false" | "0" => Value::Boolean(false),
            _ => Value::String(value.to_string()),
        },

        "date" => parse_xsd_date(value)
            .map(Value::DateTime)
            .unwrap_or_else(|| Value::String(value.to_string())),

        "dateTime" => parse_xsd_datetime(value),

        _ => Value::String(value.to_string()),
    }
}

/// Parse an `xsd:date`, tolerating a trailing `Z` or timezone offset
/// (`2020-01-01Z`, `2020-01-01+02:00`). Falls back to the first 10
/// characters (`%Y-%m-%d`) when the suffix-stripped form still fails.
fn parse_xsd_date(value: &str) -> Option<NaiveDate> {
    let base = strip_tz_suffix(value);
    if let Ok(d) = NaiveDate::parse_from_str(base, "%Y-%m-%d") {
        return Some(d);
    }
    if value.len() >= 10 {
        if let Ok(d) = NaiveDate::parse_from_str(&value[..10], "%Y-%m-%d") {
            return Some(d);
        }
    }
    None
}

/// Parse an `xsd:dateTime` to a [`Value::Timestamp`]. Tries RFC 3339
/// first (handles fractional seconds + offsets), then a plain
/// `%Y-%m-%dT%H:%M:%S` after stripping fractional seconds and any
/// timezone suffix. If only the date portion parses, returns a
/// [`Value::DateTime`]; otherwise falls back to a string.
fn parse_xsd_datetime(value: &str) -> Value {
    // RFC 3339 covers `2020-01-01T12:30:00Z` and offset forms.
    if let Ok(dt) = DateTime::parse_from_rfc3339(value) {
        return Value::Timestamp(dt.naive_utc());
    }
    // Strip timezone + fractional seconds, then parse the wall clock.
    let base = strip_tz_suffix(value);
    let base = base.split('.').next().unwrap_or(base);
    if let Ok(dt) = NaiveDateTime::parse_from_str(base, "%Y-%m-%dT%H:%M:%S") {
        return Value::Timestamp(dt);
    }
    // Degrade to a date if that's all we have.
    if let Some(d) = parse_xsd_date(value) {
        return Value::DateTime(d);
    }
    Value::String(value.to_string())
}

/// Drop a trailing `Z` or `±HH:MM` timezone offset from a date/datetime
/// lexical form. Only touches the offset region (after the time, or the
/// whole string for a bare date) so a `-` inside the date isn't eaten.
fn strip_tz_suffix(value: &str) -> &str {
    if let Some(stripped) = value.strip_suffix('Z') {
        return stripped;
    }
    // A timezone offset is `+HH:MM` / `-HH:MM` at the very end. Look for
    // the last `+`, or a `-` that sits past the date (index >= 10), to
    // avoid clipping the `-` separators inside `YYYY-MM-DD`.
    if let Some(pos) = value.rfind('+') {
        return &value[..pos];
    }
    if let Some(pos) = value.rfind('-') {
        if pos >= 10 {
            return &value[..pos];
        }
    }
    value
}

/// Parse a GeoSPARQL `wktLiteral`. Only `POINT(lon lat)` maps to a
/// native [`Value::Point`] (WKT axis order is lon-then-lat); any other
/// geometry is kept as its WKT string for downstream tools.
fn wkt_to_value(value: &str) -> Value {
    // A wktLiteral may carry a leading `<srs-uri>` CRS prefix; skip it.
    let body = value.trim();
    let body = if body.starts_with('<') {
        match body.find('>') {
            Some(end) => body[end + 1..].trim_start(),
            None => return Value::String(value.to_string()),
        }
    } else {
        body
    };

    let upper = body.to_ascii_uppercase();
    if let Some(rest) = upper.strip_prefix("POINT") {
        // Locate the original-case parenthesised coordinate pair so we
        // parse the real digits (uppercasing left them intact anyway).
        if let (Some(open), Some(close)) = (body.find('('), body.rfind(')')) {
            if open < close {
                let inner = body[open + 1..close].trim();
                let mut parts = inner.split_whitespace();
                if let (Some(lon_s), Some(lat_s), None) = (parts.next(), parts.next(), parts.next())
                {
                    if let (Ok(lon), Ok(lat)) = (lon_s.parse::<f64>(), lat_s.parse::<f64>()) {
                        return Value::Point { lat, lon };
                    }
                }
            }
        }
        // `rest` is unused beyond confirming the POINT tag; bind to _.
        let _ = rest;
    }
    Value::String(value.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn xsd(local: &str) -> String {
        format!("{}{}", XSD, local)
    }

    #[test]
    fn integers_and_decimals() {
        assert_eq!(datatype_to_value("42", &xsd("integer")), Value::Int64(42));
        assert_eq!(datatype_to_value("+7", &xsd("integer")), Value::Int64(7));
        assert_eq!(datatype_to_value("3", &xsd("decimal")), Value::Int64(3));
        assert_eq!(
            datatype_to_value("3.5", &xsd("decimal")),
            Value::Float64(3.5)
        );
        assert_eq!(
            datatype_to_value("notanum", &xsd("integer")),
            Value::String("notanum".to_string())
        );
    }

    #[test]
    fn floats_and_booleans() {
        assert_eq!(
            datatype_to_value("1.5e2", &xsd("double")),
            Value::Float64(150.0)
        );
        assert_eq!(
            datatype_to_value("true", &xsd("boolean")),
            Value::Boolean(true)
        );
        assert_eq!(
            datatype_to_value("0", &xsd("boolean")),
            Value::Boolean(false)
        );
    }

    #[test]
    fn dates_and_datetimes() {
        assert_eq!(
            datatype_to_value("2020-01-15", &xsd("date")),
            Value::DateTime(NaiveDate::from_ymd_opt(2020, 1, 15).unwrap())
        );
        assert_eq!(
            datatype_to_value("2020-01-15Z", &xsd("date")),
            Value::DateTime(NaiveDate::from_ymd_opt(2020, 1, 15).unwrap())
        );
        match datatype_to_value("2020-01-15T08:30:00Z", &xsd("dateTime")) {
            Value::Timestamp(_) => {}
            other => panic!("expected Timestamp, got {:?}", other),
        }
        match datatype_to_value("2020-01-15T08:30:00.123+02:00", &xsd("dateTime")) {
            Value::Timestamp(_) => {}
            other => panic!("expected Timestamp, got {:?}", other),
        }
    }

    #[test]
    fn wkt_point() {
        match datatype_to_value("POINT(10.5 59.9)", GEO_WKT) {
            Value::Point { lat, lon } => {
                assert_eq!(lon, 10.5);
                assert_eq!(lat, 59.9);
            }
            other => panic!("expected Point, got {:?}", other),
        }
        // Non-POINT geometry stays a string.
        assert!(matches!(
            datatype_to_value("LINESTRING(0 0, 1 1)", GEO_WKT),
            Value::String(_)
        ));
    }

    #[test]
    fn unknown_and_string() {
        assert_eq!(
            datatype_to_value("hello", &xsd("string")),
            Value::String("hello".to_string())
        );
        assert_eq!(
            datatype_to_value("x", "http://example.org/custom"),
            Value::String("x".to_string())
        );
    }
}
