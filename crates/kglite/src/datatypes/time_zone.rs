//! Named IANA time zones for [`Value::ZonedDateTime`](super::Value::ZonedDateTime).
//!
//! [`ZoneId`] is kglite's own handle: a 2-byte index into the bundled tz
//! database, so the public API never names the database crate and the same
//! type exists with the `iana-zones` feature off. Without the feature a
//! [`ZoneId`] cannot be built ([`ZoneId::from_name`] answers `None`), and a
//! stored named-zone value decodes as its fixed offset.
//!
//! The persisted form of a zone is its **name** (stable across database
//! versions), never the index. The index is only meaningful inside one build.
use chrono::NaiveDateTime;

#[cfg(feature = "iana-zones")]
type Repr = std::num::NonZeroU16;
#[cfg(not(feature = "iana-zones"))]
type Repr = std::convert::Infallible;

/// Whether this build carries the IANA time-zone database.
pub const IANA_ZONES: bool = cfg!(feature = "iana-zones");

/// A named time zone (`Europe/Oslo`). Equality is zone identity; two
/// [`Value::ZonedDateTime`](super::Value::ZonedDateTime) values are still equal
/// by instant whatever their zones.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct ZoneId(Repr);

/// How a wall-clock reading sits in a zone's rules.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum LocalFit {
    /// Exactly one instant has this reading.
    Exact,
    /// A clock change repeated the reading; the earlier offset (or the
    /// preferred one, when it is one of the two) was taken.
    Overlap,
    /// A clock change skipped the reading; it was moved forward by the length
    /// of the gap.
    Gap,
}

/// An instant and the offset a zone has at it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct ZoneInstant {
    pub utc: NaiveDateTime,
    pub offset_secs: i32,
    pub fit: LocalFit,
}

impl ZoneId {
    /// The zone called `name` (`Europe/Oslo`, `UTC`, `US/Pacific`); `None` for
    /// an unknown name, and always `None` without the `iana-zones` feature.
    pub fn from_name(name: &str) -> Option<ZoneId> {
        imp::from_name(name)
    }

    /// The IANA name this zone was looked up by.
    pub fn name(self) -> &'static str {
        imp::name(self)
    }

    /// The UTC offset in seconds the zone has at the instant `utc`.
    pub fn offset_at(self, utc: NaiveDateTime) -> i32 {
        imp::offset_at(self, utc)
    }

    /// The instant a wall-clock reading names in this zone.
    ///
    /// A reading in a **gap** (clocks went forward) is shifted forward by the
    /// length of the gap, so `02:30` on the night Oslo skips 02:00-03:00 is
    /// `03:30+02:00`. A reading in an **overlap** (clocks went back) takes the
    /// earlier instant, i.e. the offset from before the change, unless
    /// `preferred_offset` is the other one. `None` when the instant is out of
    /// the representable range.
    pub fn resolve_local(
        self,
        wall: NaiveDateTime,
        preferred_offset: Option<i32>,
    ) -> Option<ZoneInstant> {
        imp::resolve_local(self, wall, preferred_offset)
    }

    /// The same instant seen in this zone.
    pub fn at_instant(self, utc: NaiveDateTime) -> ZoneInstant {
        ZoneInstant {
            utc,
            offset_secs: self.offset_at(utc),
            fit: LocalFit::Exact,
        }
    }
}

#[cfg(feature = "iana-zones")]
mod imp {
    use super::{LocalFit, ZoneId, ZoneInstant};
    use chrono::{NaiveDateTime, TimeDelta};
    use jiff::civil;
    use jiff::tz::{AmbiguousOffset, TimeZone, TimeZoneDatabase};
    use jiff::Timestamp;
    use std::num::NonZeroU16;
    use std::sync::OnceLock;

    /// Every zone name the bundled database knows, sorted for an exact
    /// (case-sensitive) lookup, each with its rules parsed on first use.
    /// A `ZoneId` is `index + 1` into this table. The bundled database is
    /// named explicitly so the answer never depends on the host's zoneinfo.
    struct Table {
        db: TimeZoneDatabase,
        zones: Vec<(String, OnceLock<TimeZone>)>,
    }

    fn table() -> &'static Table {
        static TABLE: OnceLock<Table> = OnceLock::new();
        TABLE.get_or_init(|| {
            let db = TimeZoneDatabase::bundled();
            let mut names: Vec<String> = db.available().map(|n| n.as_str().to_owned()).collect();
            names.sort_unstable();
            names.dedup();
            let zones = names.into_iter().map(|n| (n, OnceLock::new())).collect();
            Table { db, zones }
        })
    }

    fn tz(id: ZoneId) -> &'static TimeZone {
        let table = table();
        let (name, zone) = &table.zones[usize::from(id.0.get()) - 1];
        zone.get_or_init(|| {
            // Every listed name resolves; UTC is the defensive fallback.
            table.db.get(name).unwrap_or(TimeZone::UTC)
        })
    }

    pub(super) fn from_name(name: &str) -> Option<ZoneId> {
        let zones = &table().zones;
        let index = zones.binary_search_by(|(n, _)| n.as_str().cmp(name)).ok()?;
        let index = u16::try_from(index).ok()?.checked_add(1)?;
        NonZeroU16::new(index).map(ZoneId)
    }

    pub(super) fn name(id: ZoneId) -> &'static str {
        &table().zones[usize::from(id.0.get()) - 1].0
    }

    /// `utc` as a jiff instant, clamped into jiff's ±9999-year range (the
    /// rules at the far ends are fixed offsets, so the clamp answers with the
    /// offset the zone has there). A leap second (`nanos >= 1e9`) reads as the
    /// last nanosecond of its second.
    fn instant(utc: NaiveDateTime) -> Timestamp {
        let utc = utc.and_utc();
        let nanos = utc.timestamp_subsec_nanos().min(999_999_999) as i32;
        Timestamp::new(utc.timestamp(), nanos).unwrap_or(if utc.timestamp() < 0 {
            Timestamp::MIN
        } else {
            Timestamp::MAX
        })
    }

    pub(super) fn offset_at(id: ZoneId, utc: NaiveDateTime) -> i32 {
        tz(id).to_offset(instant(utc)).seconds()
    }

    fn civil(wall: NaiveDateTime) -> Option<civil::DateTime> {
        use chrono::{Datelike, Timelike};
        civil::DateTime::new(
            i16::try_from(wall.year()).ok()?,
            wall.month() as i8,
            wall.day() as i8,
            wall.hour() as i8,
            wall.minute() as i8,
            wall.second() as i8,
            wall.nanosecond().min(999_999_999) as i32,
        )
        .ok()
    }

    pub(super) fn resolve_local(
        id: ZoneId,
        wall: NaiveDateTime,
        preferred: Option<i32>,
    ) -> Option<ZoneInstant> {
        let at = |offset_secs: i32, fit| -> Option<ZoneInstant> {
            let utc = wall.checked_sub_signed(TimeDelta::try_seconds(i64::from(offset_secs))?)?;
            Some(ZoneInstant {
                utc,
                offset_secs,
                fit,
            })
        };
        let Some(civil) = civil(wall) else {
            // Outside jiff's range the zone's rules are a fixed offset; read
            // the reading with the offset at the clamped end.
            let offset = offset_at(id, wall);
            return at(offset, LocalFit::Exact);
        };
        match tz(id).to_ambiguous_timestamp(civil).offset() {
            AmbiguousOffset::Unambiguous { offset } => at(offset.seconds(), LocalFit::Exact),
            AmbiguousOffset::Fold { before, after } => {
                let (earlier, later) = (before.seconds(), after.seconds());
                let offset = if preferred == Some(later) {
                    later
                } else {
                    earlier
                };
                at(offset, LocalFit::Overlap)
            }
            AmbiguousOffset::Gap { before, after } => {
                // A skipped reading: read it with the offset from before the
                // change, which lands after the change, and show it with the
                // offset from after (the wall clock moves forward by the gap).
                let instant = at(before.seconds(), LocalFit::Gap)?;
                Some(ZoneInstant {
                    offset_secs: after.seconds(),
                    ..instant
                })
            }
        }
    }

    #[cfg(test)]
    pub(super) fn variant_count() -> usize {
        table().zones.len()
    }
}

#[cfg(not(feature = "iana-zones"))]
mod imp {
    use super::{ZoneId, ZoneInstant};
    use chrono::NaiveDateTime;

    pub(super) fn from_name(_name: &str) -> Option<ZoneId> {
        None
    }

    // Without the feature no `ZoneId` exists, so these cannot run.
    pub(super) fn name(id: ZoneId) -> &'static str {
        match id.0 {}
    }

    pub(super) fn offset_at(id: ZoneId, _utc: NaiveDateTime) -> i32 {
        match id.0 {}
    }

    pub(super) fn resolve_local(
        id: ZoneId,
        _wall: NaiveDateTime,
        _preferred: Option<i32>,
    ) -> Option<ZoneInstant> {
        match id.0 {}
    }
}

#[cfg(all(test, feature = "iana-zones"))]
mod tests {
    use super::*;
    use chrono::NaiveDate;

    fn wall(y: i32, mo: u32, d: u32, h: u32, mi: u32) -> NaiveDateTime {
        NaiveDate::from_ymd_opt(y, mo, d)
            .unwrap()
            .and_hms_opt(h, mi, 0)
            .unwrap()
    }

    fn zone(name: &str) -> ZoneId {
        ZoneId::from_name(name).unwrap()
    }

    #[test]
    fn every_variant_round_trips_through_its_name() {
        for i in 1..=imp::variant_count() {
            let id = ZoneId(std::num::NonZeroU16::new(i as u16).unwrap());
            assert_eq!(ZoneId::from_name(id.name()), Some(id), "{}", id.name());
        }
        assert_eq!(ZoneId::from_name("Mars/Olympus"), None);
        assert_eq!(ZoneId::from_name(""), None);
        assert_eq!(zone("US/Pacific").name(), "US/Pacific");
    }

    // Oslo: UTC+1 until 2026-03-29 01:00Z, UTC+2 until 2026-10-25 01:00Z.
    #[test]
    fn oslo_spring_forward_shifts_a_skipped_reading_by_the_gap() {
        let oslo = zone("Europe/Oslo");
        // 02:30 does not exist on 2026-03-29 (02:00 -> 03:00): it becomes
        // 03:30+02:00, the instant 01:30Z.
        let r = oslo.resolve_local(wall(2026, 3, 29, 2, 30), None).unwrap();
        assert_eq!(r.fit, LocalFit::Gap);
        assert_eq!(r.offset_secs, 7200);
        assert_eq!(r.utc, wall(2026, 3, 29, 1, 30));
        // The reading just before and just after the gap are exact.
        let before = oslo.resolve_local(wall(2026, 3, 29, 1, 59), None).unwrap();
        assert_eq!((before.fit, before.offset_secs), (LocalFit::Exact, 3600));
        let after = oslo.resolve_local(wall(2026, 3, 29, 3, 0), None).unwrap();
        assert_eq!((after.fit, after.offset_secs), (LocalFit::Exact, 7200));
        assert_eq!(after.utc, wall(2026, 3, 29, 1, 0));
    }

    #[test]
    fn oslo_fall_back_takes_the_earlier_offset_unless_preferred() {
        let oslo = zone("Europe/Oslo");
        // 02:30 happens twice on 2026-10-25 (+02:00 at 00:30Z, +01:00 at 01:30Z).
        let first = oslo.resolve_local(wall(2026, 10, 25, 2, 30), None).unwrap();
        assert_eq!(first.fit, LocalFit::Overlap);
        assert_eq!(
            (first.offset_secs, first.utc),
            (7200, wall(2026, 10, 25, 0, 30))
        );
        let second = oslo
            .resolve_local(wall(2026, 10, 25, 2, 30), Some(3600))
            .unwrap();
        assert_eq!(
            (second.offset_secs, second.utc),
            (3600, wall(2026, 10, 25, 1, 30))
        );
        // A preference that is neither offset falls back to the earlier one.
        let other = oslo
            .resolve_local(wall(2026, 10, 25, 2, 30), Some(0))
            .unwrap();
        assert_eq!(other.offset_secs, 7200);
    }

    // Sydney (southern hemisphere): UTC+11 until 2026-04-05 03:00 local
    // (2026-04-04 16:00Z), then UTC+10 until 2026-10-04 02:00 local
    // (2026-10-03 16:00Z), then UTC+11.
    #[test]
    fn sydney_runs_the_changes_in_the_other_season() {
        let sydney = zone("Australia/Sydney");
        // Autumn 2026-04-05: 02:30 happens twice; the earlier is +11:00.
        let r = sydney.resolve_local(wall(2026, 4, 5, 2, 30), None).unwrap();
        assert_eq!((r.fit, r.offset_secs), (LocalFit::Overlap, 39_600));
        assert_eq!(r.utc, wall(2026, 4, 4, 15, 30));
        // Spring 2026-10-04: 02:30 is skipped (02:00 -> 03:00), becoming 03:30+11:00.
        let g = sydney
            .resolve_local(wall(2026, 10, 4, 2, 30), None)
            .unwrap();
        assert_eq!((g.fit, g.offset_secs), (LocalFit::Gap, 39_600));
        assert_eq!(g.utc, wall(2026, 10, 3, 16, 30));
    }

    #[test]
    fn a_whole_skipped_day_moves_forward_a_day() {
        // Samoa jumped from 2011-12-29 23:59:59 (-10:00) to 2011-12-31 00:00 (+14:00).
        let apia = zone("Pacific/Apia");
        let r = apia.resolve_local(wall(2011, 12, 30, 12, 0), None).unwrap();
        assert_eq!((r.fit, r.offset_secs), (LocalFit::Gap, 14 * 3600));
        assert_eq!(r.utc, wall(2011, 12, 30, 22, 0));
    }

    // The database's own lookup ignores case; names stay exact, as they were
    // before the database changed, so `[europe/oslo]` is still unknown.
    #[test]
    fn zone_names_are_case_sensitive() {
        assert!(ZoneId::from_name("Europe/Oslo").is_some());
        assert_eq!(ZoneId::from_name("europe/oslo"), None);
        assert_eq!(ZoneId::from_name("EUROPE/OSLO"), None);
    }

    // The database covers years -9999..=9999; past that a zone answers with
    // the fixed offset at the end of its range instead of failing, and a leap
    // second reads as the end of its second.
    #[test]
    fn readings_outside_the_database_range_still_resolve() {
        let oslo = zone("Europe/Oslo");
        // Before 1895 Oslo kept local mean time, +00:53:28.
        for (year, expected) in [
            (-20_000, &[3208][..]),
            (10_000, &[3600, 7200]),
            (200_000, &[3600, 7200]),
        ] {
            let utc = wall(year, 6, 1, 12, 0);
            let offset = oslo.offset_at(utc);
            assert!(expected.contains(&offset), "{year}: {offset}");
            let r = oslo.resolve_local(utc, None).unwrap();
            assert_eq!(r.offset_secs, offset, "{year}");
        }
        let leap = NaiveDate::from_ymd_opt(2016, 12, 31)
            .unwrap()
            .and_hms_nano_opt(23, 59, 59, 1_500_000_000)
            .unwrap();
        assert_eq!(zone("UTC").offset_at(leap), 0);
        let r = oslo.resolve_local(leap, None).unwrap();
        assert_eq!((r.fit, r.offset_secs), (LocalFit::Exact, 3600));
    }

    #[test]
    fn offset_at_follows_the_instant() {
        let oslo = zone("Europe/Oslo");
        assert_eq!(oslo.offset_at(wall(2026, 3, 29, 0, 59)), 3600);
        assert_eq!(oslo.offset_at(wall(2026, 3, 29, 1, 0)), 7200);
        assert_eq!(zone("UTC").offset_at(wall(2026, 6, 1, 0, 0)), 0);
    }
}
