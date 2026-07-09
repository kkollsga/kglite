// src/datatypes/type_conversions.rs
use chrono::{NaiveDate, NaiveDateTime};
use pyo3::prelude::*;
use pyo3::Bound;

pub fn to_u32(value: &Bound<'_, PyAny>) -> Option<u32> {
    if value.is_none() {
        return None;
    }
    if let Ok(val) = value.extract::<u32>() {
        return Some(val);
    }
    if let Ok(val) = value.extract::<i64>() {
        if val >= 0 && val <= u32::MAX as i64 {
            return Some(val as u32);
        }
    }
    if let Ok(val) = value.extract::<f64>() {
        if val.fract() == 0.0 && val >= 0.0 && val <= u32::MAX as f64 {
            return Some(val as u32);
        }
    }
    if let Ok(s) = value.extract::<String>() {
        if let Ok(val) = s.parse::<u32>() {
            return Some(val);
        }
        if let Ok(val) = s.parse::<f64>() {
            if val.fract() == 0.0 && val >= 0.0 && val <= u32::MAX as f64 {
                return Some(val as u32);
            }
        }
    }
    None
}

pub fn to_i64(value: &Bound<'_, PyAny>) -> Option<i64> {
    if value.is_none() {
        return None;
    }
    if let Ok(val) = value.extract::<i64>() {
        return Some(val);
    }
    if let Ok(val) = value.extract::<f64>() {
        if val.fract() == 0.0 && val >= i64::MIN as f64 && val <= i64::MAX as f64 {
            return Some(val as i64);
        }
    }
    if let Ok(s) = value.extract::<String>() {
        if let Ok(val) = s.parse::<i64>() {
            return Some(val);
        }
        if let Ok(val) = s.parse::<f64>() {
            if val.fract() == 0.0 && val >= i64::MIN as f64 && val <= i64::MAX as f64 {
                return Some(val as i64);
            }
        }
    }
    None
}

pub fn to_f64(value: &Bound<'_, PyAny>) -> Option<f64> {
    if value.is_none() {
        return None;
    }
    if let Ok(val) = value.extract::<f64>() {
        if val.is_nan() {
            return None;
        }
        return Some(val);
    }
    if let Ok(val) = value.extract::<i64>() {
        return Some(val as f64);
    }
    if let Ok(s) = value.extract::<String>() {
        if let Ok(val) = s.parse::<f64>() {
            if val.is_nan() {
                return None;
            }
            return Some(val);
        }
    }
    None
}

pub fn to_datetime(value: &Bound<'_, PyAny>) -> Option<NaiveDate> {
    if value.is_none() {
        return None;
    }

    // Try to extract as Python datetime/date first via attribute access
    // (abi3-compatible: PyDateAccess is not part of the stable ABI).
    Python::attach(|_py| {
        if let (Ok(y), Ok(m), Ok(d)) = (
            value.getattr("year").and_then(|v| v.extract::<i32>()),
            value.getattr("month").and_then(|v| v.extract::<u32>()),
            value.getattr("day").and_then(|v| v.extract::<u32>()),
        ) {
            if let Some(date) = NaiveDate::from_ymd_opt(y, m, d) {
                return Some(date);
            }
        }

        // Try to parse string dates (ISO format: YYYY-MM-DD)
        if let Ok(s) = value.extract::<String>() {
            // Try ISO format first (YYYY-MM-DD)
            if let Ok(date) = NaiveDate::parse_from_str(&s, "%Y-%m-%d") {
                return Some(date);
            }
            // Try with slashes (YYYY/MM/DD)
            if let Ok(date) = NaiveDate::parse_from_str(&s, "%Y/%m/%d") {
                return Some(date);
            }
            // Try DD-MM-YYYY
            if let Ok(date) = NaiveDate::parse_from_str(&s, "%d-%m-%Y") {
                return Some(date);
            }
            // Try MM/DD/YYYY
            if let Ok(date) = NaiveDate::parse_from_str(&s, "%m/%d/%Y") {
                return Some(date);
            }
        }

        None
    })
}

/// Convert a Python/pandas datetime-like value to a full-precision
/// `NaiveDateTime` (date + time-of-day). Used by the `Timestamp` column path so
/// a `datetime64` cell with a nonzero time isn't truncated to date-only.
pub fn to_timestamp(value: &Bound<'_, PyAny>) -> Option<NaiveDateTime> {
    if value.is_none() {
        return None;
    }

    // pandas.Timestamp / datetime.datetime subclass datetime, so pyo3's chrono
    // conversion handles them directly.
    if let Ok(dt) = value.extract::<NaiveDateTime>() {
        return Some(dt);
    }

    Python::attach(|_py| {
        // Attribute-based fallback (abi3-safe: no PyDateAccess).
        if let (Ok(y), Ok(mo), Ok(d)) = (
            value.getattr("year").and_then(|v| v.extract::<i32>()),
            value.getattr("month").and_then(|v| v.extract::<u32>()),
            value.getattr("day").and_then(|v| v.extract::<u32>()),
        ) {
            let h = value
                .getattr("hour")
                .and_then(|v| v.extract::<u32>())
                .unwrap_or(0);
            let mi = value
                .getattr("minute")
                .and_then(|v| v.extract::<u32>())
                .unwrap_or(0);
            let s = value
                .getattr("second")
                .and_then(|v| v.extract::<u32>())
                .unwrap_or(0);
            let us = value
                .getattr("microsecond")
                .and_then(|v| v.extract::<u32>())
                .unwrap_or(0);
            if let Some(date) = NaiveDate::from_ymd_opt(y, mo, d) {
                if let Some(dt) = date.and_hms_micro_opt(h, mi, s, us) {
                    return Some(dt);
                }
            }
        }

        // ISO string fallback (with and without fractional seconds / 'T').
        if let Ok(st) = value.extract::<String>() {
            for fmt in [
                "%Y-%m-%dT%H:%M:%S%.f",
                "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%d %H:%M:%S%.f",
                "%Y-%m-%d %H:%M:%S",
            ] {
                if let Ok(dt) = NaiveDateTime::parse_from_str(&st, fmt) {
                    return Some(dt);
                }
            }
            // Date-only string → midnight.
            if let Ok(date) = NaiveDate::parse_from_str(&st, "%Y-%m-%d") {
                return date.and_hms_opt(0, 0, 0);
            }
        }

        None
    })
}

pub fn to_bool(value: &Bound<'_, PyAny>) -> Option<bool> {
    if value.is_none() {
        return None;
    }
    if let Ok(b) = value.extract::<bool>() {
        return Some(b);
    }
    if let Ok(s) = value.str() {
        match s.to_string().to_lowercase().as_str() {
            "true" | "1" | "yes" | "t" | "y" => return Some(true),
            "false" | "0" | "no" | "f" | "n" => return Some(false),
            _ => return None,
        }
    }
    None
}
