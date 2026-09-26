//! The validity-interval evaluator: how a query instant is read and how it is
//! compared with an element's `from`/`to` bounds. Cypher `valid_at` /
//! `valid_during` and the fluent temporal filters all answer through it, so a
//! bound stored as a date, a datetime or an ISO string gives one answer
//! whichever surface asks.

use crate::datatypes::values::Value;
use crate::graph::property_types::value_type_name;
use chrono::{NaiveDate, NaiveDateTime, NaiveTime};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::fmt;

/// A point on the time line, at the grain it was written in.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Instant {
    Date(NaiveDate),
    /// Naive UTC: an offset the source carried has already been applied.
    Timestamp(NaiveDateTime),
}

impl Instant {
    pub(crate) fn date(self) -> NaiveDate {
        match self {
            Instant::Date(d) => d,
            Instant::Timestamp(ts) => ts.date(),
        }
    }

    /// Chronological order. Two timestamps compare exactly; when either side
    /// is a date the comparison is at date grain, so a date bound covers its
    /// whole day.
    pub(crate) fn chrono_cmp(self, other: Instant) -> Ordering {
        match (self, other) {
            (Instant::Timestamp(a), Instant::Timestamp(b)) => a.cmp(&b),
            (a, b) => a.date().cmp(&b.date()),
        }
    }
}

/// Whether the `to` bound belongs to the interval.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IntervalConvention {
    /// `[from, to]`: the `to` day is the last valid day.
    #[default]
    Closed,
    /// `[from, to)`: a date `to` is the first day no longer valid; a datetime
    /// `to` is the first instant, so its day is still valid when it ends after
    /// midnight.
    HalfOpen,
}

impl IntervalConvention {
    /// The spelling a declaration takes and reports: `closed` / `half_open`.
    pub fn as_str(self) -> &'static str {
        match self {
            IntervalConvention::Closed => "closed",
            IntervalConvention::HalfOpen => "half_open",
        }
    }

    /// Read [`Self::as_str`]'s spelling back; `None` for anything else.
    pub fn parse(text: &str) -> Option<Self> {
        match text {
            "closed" => Some(IntervalConvention::Closed),
            "half_open" => Some(IntervalConvention::HalfOpen),
            _ => None,
        }
    }

    pub(crate) fn is_closed(&self) -> bool {
        *self == IntervalConvention::Closed
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum BoundSide {
    From,
    To,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum TemporalError {
    /// The query instant is not a date, a datetime, or a string that
    /// `date()` / `datetime()` can read.
    Instant { found: &'static str, shown: String },
    /// A stored bound is not NULL, a date, a datetime or a readable ISO
    /// string.
    Bound {
        side: BoundSide,
        found: &'static str,
        shown: String,
    },
}

impl fmt::Display for TemporalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TemporalError::Instant { found, shown } => {
                write!(f, "{shown} ({found}) is not a date or datetime")
            }
            TemporalError::Bound { side, found, shown } => {
                let side = match side {
                    BoundSide::From => "from",
                    BoundSide::To => "to",
                };
                write!(
                    f,
                    "the {side} bound {shown} ({found}) is not a date, a datetime or an ISO date string"
                )
            }
        }
    }
}

pub(crate) fn shown(value: &Value) -> String {
    match value {
        Value::String(s) => format!("'{s}'"),
        other => crate::graph::core::value_operations::format_value_compact(other),
    }
}

/// Read a string as `date()` reads it when it has no time part, and as
/// `datetime()` reads it otherwise.
fn parse_instant_str(text: &str) -> Option<Instant> {
    if let Ok((date, _)) = crate::graph::features::timeseries::parse_date_query(text) {
        return Some(Instant::Date(date));
    }
    crate::graph::languages::cypher::executor::scalar_functions::parse_datetime_utc(text)
        .map(Instant::Timestamp)
}

/// The query instant. A date or datetime value is taken as is; a string is
/// read by the `date()` parser when it has no time part (`'2009'` is
/// 2009-01-01) and by the `datetime()` parser otherwise (an offset is applied,
/// normalised to UTC). Anything else, NULL included, is an error.
pub(crate) fn parse_instant(value: &Value) -> Result<Instant, TemporalError> {
    let parsed = match value {
        Value::DateTime(d) => Some(Instant::Date(*d)),
        Value::Timestamp(ts) => Some(Instant::Timestamp(*ts)),
        Value::String(s) => parse_instant_str(s),
        _ => None,
    };
    parsed.ok_or_else(|| TemporalError::Instant {
        found: value_type_name(value),
        shown: shown(value),
    })
}

/// A stored bound: `None` for NULL (open), otherwise the instant it holds.
fn parse_bound(value: &Value, side: BoundSide) -> Result<Option<Instant>, TemporalError> {
    let parsed = match value {
        Value::Null => return Ok(None),
        Value::DateTime(d) => Some(Instant::Date(*d)),
        Value::Timestamp(ts) => Some(Instant::Timestamp(*ts)),
        Value::String(s) => parse_instant_str(s),
        _ => None,
    };
    match parsed {
        Some(instant) => Ok(Some(instant)),
        None => Err(TemporalError::Bound {
            side,
            found: value_type_name(value),
            shown: shown(value),
        }),
    }
}

/// Both stored bounds, `from` first; the first unreadable one is the error.
pub(crate) fn parse_bounds(
    from: &Value,
    to: &Value,
) -> Result<(Option<Instant>, Option<Instant>), TemporalError> {
    Ok((
        parse_bound(from, BoundSide::From)?,
        parse_bound(to, BoundSide::To)?,
    ))
}

/// `from <= t` and `t <= to` (`t < to` when half-open, as [`end_admits`]
/// reads it). NULL bounds are open.
/// Both bounds are read before either is compared, so a bad bound errors
/// whatever the instant.
pub(crate) fn interval_contains(
    from: &Value,
    to: &Value,
    instant: Instant,
    convention: IntervalConvention,
) -> Result<bool, TemporalError> {
    let (from, to) = parse_bounds(from, to)?;
    Ok(starts_by(from, instant) && ends_after(to, instant, convention))
}

/// Whether the element's interval shares an instant with the closed query
/// range `[a, b]`. An empty interval — inverted, or `from == to` under
/// half-open, which a write after the declaration can leave — shares none, as
/// [`interval_contains`] finds it valid on no date.
pub(crate) fn interval_overlaps(
    from: &Value,
    to: &Value,
    a: Instant,
    b: Instant,
    convention: IntervalConvention,
) -> Result<bool, TemporalError> {
    let (from, to) = parse_bounds(from, to)?;
    let non_empty = match (from, to) {
        (Some(from), Some(to)) => end_admits(to, from, convention),
        _ => true,
    };
    Ok(non_empty && starts_by(from, b) && ends_after(to, a, convention))
}

fn starts_by(from: Option<Instant>, t: Instant) -> bool {
    from.is_none_or(|f| f.chrono_cmp(t) != Ordering::Greater)
}

fn ends_after(to: Option<Instant>, t: Instant, convention: IntervalConvention) -> bool {
    to.is_none_or(|end| end_admits(end, t, convention))
}

/// Whether the end bound `end` still admits `t`: `t <= end` closed, `t < end`
/// half-open, at [`Instant::chrono_cmp`]'s grain. One exception: half-open, a
/// timestamp end against a date `t` is compared with `t`'s midnight exactly,
/// so `[.., 2009-06-30T20:00)` holds part of 06-30 and is valid on it, while
/// `[.., 2009-06-30T00:00)` holds none of it.
pub(crate) fn end_admits(end: Instant, t: Instant, convention: IntervalConvention) -> bool {
    match (convention, end, t) {
        (IntervalConvention::Closed, _, _) => end.chrono_cmp(t) != Ordering::Less,
        (IntervalConvention::HalfOpen, Instant::Timestamp(end), Instant::Date(day)) => {
            end > day.and_time(NaiveTime::MIN)
        }
        (IntervalConvention::HalfOpen, _, _) => end.chrono_cmp(t) == Ordering::Greater,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_empty_interval_overlaps_no_range() {
        let (a, b) = (
            Instant::Date(NaiveDate::from_ymd_opt(1900, 1, 1).unwrap()),
            Instant::Date(NaiveDate::from_ymd_opt(2100, 1, 1).unwrap()),
        );
        let closed = IntervalConvention::Closed;
        let half_open = IntervalConvention::HalfOpen;
        let inverted = (d("2005-01-01"), d("1990-01-01"));
        assert!(!interval_overlaps(&inverted.0, &inverted.1, a, b, closed).unwrap());
        assert!(!interval_overlaps(&inverted.0, &inverted.1, a, b, half_open).unwrap());
        let one_day = (d("2005-01-01"), d("2005-01-01"));
        assert!(interval_overlaps(&one_day.0, &one_day.1, a, b, closed).unwrap());
        assert!(!interval_overlaps(&one_day.0, &one_day.1, a, b, half_open).unwrap());
        // A datetime end later on the start day leaves part of that day.
        let part_day = (d("2005-01-01"), ts("2005-01-01T20:00"));
        assert!(interval_overlaps(&part_day.0, &part_day.1, a, b, half_open).unwrap());
        assert!(interval_overlaps(&Value::Null, &inverted.1, a, b, half_open).unwrap());
    }

    fn d(s: &str) -> Value {
        Value::DateTime(NaiveDate::parse_from_str(s, "%Y-%m-%d").unwrap())
    }

    fn ts(s: &str) -> Value {
        Value::Timestamp(NaiveDateTime::parse_from_str(s, "%Y-%m-%dT%H:%M").unwrap())
    }

    fn s(text: &str) -> Value {
        Value::String(text.to_string())
    }

    fn at(v: Value) -> Instant {
        parse_instant(&v).unwrap()
    }

    fn contains(from: &Value, to: &Value, t: Value, c: IntervalConvention) -> bool {
        interval_contains(from, to, at(t), c).unwrap()
    }

    const CLOSED: IntervalConvention = IntervalConvention::Closed;
    const HALF_OPEN: IntervalConvention = IntervalConvention::HalfOpen;

    #[test]
    fn date_only_strings_read_as_date_does() {
        let jan1 = Instant::Date(NaiveDate::from_ymd_opt(2009, 1, 1).unwrap());
        assert_eq!(parse_instant(&s("2009")), Ok(jan1));
        assert_eq!(
            parse_instant(&s("2009-06")),
            Ok(Instant::Date(NaiveDate::from_ymd_opt(2009, 6, 1).unwrap()))
        );
        assert_eq!(
            parse_instant(&s("2009-06-30")),
            Ok(Instant::Date(NaiveDate::from_ymd_opt(2009, 6, 30).unwrap()))
        );
        assert_eq!(parse_instant(&d("2009-01-01")), Ok(jan1));
    }

    #[test]
    fn a_string_offset_is_applied_not_dropped() {
        let got = parse_instant(&s("2009-06-30T01:00:00+02:00")).unwrap();
        assert_eq!(got, at(ts("2009-06-29T23:00")));
        let minutes = parse_instant(&s("2009-06-30T01:00+02:00")).unwrap();
        assert_eq!(minutes, got);
    }

    #[test]
    fn unreadable_instants_are_errors() {
        for bad in [s("garbage"), Value::Int64(2009), Value::Null, s("")] {
            assert!(
                matches!(parse_instant(&bad), Err(TemporalError::Instant { .. })),
                "{bad:?}"
            );
        }
        assert_eq!(
            parse_instant(&Value::Int64(2009)),
            Err(TemporalError::Instant {
                found: "INTEGER",
                shown: "2009".into()
            })
        );
    }

    #[test]
    fn null_bounds_are_open() {
        assert!(contains(&Value::Null, &Value::Null, s("1900"), CLOSED));
        assert!(contains(&d("2005-01-01"), &Value::Null, s("2100"), CLOSED));
        assert!(!contains(&d("2005-01-01"), &Value::Null, s("2004"), CLOSED));
    }

    #[test]
    fn every_bound_kind_answers_alike() {
        let bounds = [
            (d("2005-01-01"), d("2012-12-31")),
            (s("2005-01-01"), s("2012-12-31")),
            (ts("2005-01-01T00:00"), ts("2012-12-31T12:00")),
            (s("2005"), s("2012-12-31T12:00:00")),
        ];
        for (from, to) in &bounds {
            assert!(contains(from, to, s("2009"), CLOSED), "{from:?}");
            assert!(contains(from, to, s("2005"), CLOSED), "{from:?}");
            assert!(!contains(from, to, s("2004-12-31"), CLOSED), "{from:?}");
            assert!(!contains(from, to, s("2013"), CLOSED), "{from:?}");
        }
    }

    #[test]
    fn timestamp_bound_is_exact_against_a_timestamp_and_daily_against_a_date() {
        let from = d("2000-01-01");
        let to = ts("2009-06-29T23:30");
        assert!(contains(&from, &to, ts("2009-06-29T23:00"), CLOSED));
        assert!(!contains(
            &from,
            &ts("2009-06-29T22:30"),
            ts("2009-06-29T23:00"),
            CLOSED
        ));
        // A date instant compares at date grain: 22:30 on the 29th still
        // covers the 29th.
        assert!(contains(
            &from,
            &ts("2009-06-29T22:30"),
            d("2009-06-29"),
            CLOSED
        ));
        assert!(!contains(
            &from,
            &ts("2009-06-29T22:30"),
            d("2009-06-30"),
            CLOSED
        ));
    }

    #[test]
    fn date_bound_covers_its_whole_day_against_a_timestamp() {
        let from = d("2000-01-01");
        let to = d("2009-06-29");
        assert!(contains(&from, &to, ts("2009-06-29T23:00"), CLOSED));
        assert!(!contains(&from, &to, ts("2009-06-30T00:00"), CLOSED));
        assert!(contains(
            &d("2009-06-29"),
            &Value::Null,
            ts("2009-06-29T00:00"),
            CLOSED
        ));
    }

    #[test]
    fn closed_and_half_open_differ_only_on_the_to_day() {
        let from = d("2009-01-01");
        let to = d("2009-06-30");
        assert!(contains(&from, &to, s("2009-06-30"), CLOSED));
        assert!(!contains(&from, &to, s("2009-06-30"), HALF_OPEN));
        assert!(contains(&from, &to, s("2009-06-29"), HALF_OPEN));
        // The from day belongs to the interval under both conventions.
        assert!(contains(&from, &to, s("2009-01-01"), HALF_OPEN));
        assert!(!contains(&from, &to, s("2008-12-31"), HALF_OPEN));
        let ts_to = ts("2009-06-30T12:00");
        assert!(contains(&from, &ts_to, ts("2009-06-30T11:59"), HALF_OPEN));
        assert!(!contains(&from, &ts_to, ts("2009-06-30T12:00"), HALF_OPEN));
        assert!(contains(&from, &ts_to, ts("2009-06-30T12:00"), CLOSED));
    }

    #[test]
    fn a_half_open_timestamp_end_is_exact_against_a_dates_midnight() {
        // The interval exists only on 06-30, so it is valid on 06-30.
        let from = ts("2009-06-30T08:00");
        let to = ts("2009-06-30T20:00");
        assert!(contains(&from, &to, d("2009-06-30"), HALF_OPEN));
        assert!(!contains(&from, &to, d("2009-07-01"), HALF_OPEN));
        assert!(!contains(&from, &to, d("2009-06-29"), HALF_OPEN));
        // A date from with a mid-day end is valid on its day; an end at
        // midnight holds none of that day.
        assert!(contains(
            &d("2009-06-01"),
            &ts("2009-06-30T12:00"),
            d("2009-06-30"),
            HALF_OPEN
        ));
        assert!(!contains(
            &d("2009-06-01"),
            &ts("2009-06-30T00:00"),
            d("2009-06-30"),
            HALF_OPEN
        ));
        // A string timestamp end reads the same.
        assert!(contains(
            &from,
            &s("2009-06-30T20:00:00"),
            d("2009-06-30"),
            HALF_OPEN
        ));
        // Every other pairing keeps the date-grain rule.
        assert!(!contains(
            &from,
            &d("2009-06-30"),
            d("2009-06-30"),
            HALF_OPEN
        ));
        assert!(contains(&from, &to, d("2009-06-30"), CLOSED));
    }

    #[test]
    fn a_half_open_timestamp_end_overlaps_a_range_starting_on_its_day() {
        let overlaps = |to: Value, a: &str| {
            interval_overlaps(
                &d("2009-06-01"),
                &to,
                at(s(a)),
                at(s("2009-07-10")),
                HALF_OPEN,
            )
            .unwrap()
        };
        assert!(overlaps(ts("2009-06-30T12:00"), "2009-06-30"));
        assert!(!overlaps(ts("2009-06-30T00:00"), "2009-06-30"));
        assert!(!overlaps(ts("2009-06-30T12:00"), "2009-07-01"));
    }

    #[test]
    fn overlap_is_closed_on_the_query_range() {
        let from = d("2009-01-01");
        let to = d("2009-06-30");
        let overlaps =
            |a: &str, b: &str, c| interval_overlaps(&from, &to, at(s(a)), at(s(b)), c).unwrap();
        assert!(overlaps("2000", "2009", CLOSED));
        assert!(!overlaps("2000", "2008-12-31", CLOSED));
        assert!(overlaps("2009-06-30", "2010", CLOSED));
        assert!(!overlaps("2009-06-30", "2010", HALF_OPEN));
        assert!(!overlaps("2009-07-01", "2010", CLOSED));
        assert!(interval_overlaps(
            &Value::Null,
            &Value::Null,
            at(s("1900")),
            at(s("1901")),
            CLOSED
        )
        .unwrap());
    }

    #[test]
    fn unreadable_bounds_are_errors_whatever_the_instant() {
        let err = interval_contains(&s("someday"), &d("2012-12-31"), at(s("2009")), CLOSED);
        assert_eq!(
            err,
            Err(TemporalError::Bound {
                side: BoundSide::From,
                found: "STRING",
                shown: "'someday'".into()
            })
        );
        // The from bound alone already excludes the instant; the bad to bound
        // is still reported.
        let err = interval_contains(&d("2010-01-01"), &Value::Int64(2012), at(s("2009")), CLOSED);
        assert!(matches!(
            err,
            Err(TemporalError::Bound {
                side: BoundSide::To,
                found: "INTEGER",
                ..
            })
        ));
    }
}
