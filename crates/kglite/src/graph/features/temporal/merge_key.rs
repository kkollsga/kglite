//! The start a bulk load keys a declared temporal relationship on. Every
//! site that decides whether a row merges into a stored relationship — the
//! batch flush, the `skip` pre-check and the relationship-constraint gate —
//! reads the start through [`StartKey::read`], so they agree on which rows are
//! the same period.

use std::borrow::Borrow;

use chrono::NaiveTime;

use super::eval::{parse_instant, Instant};
use crate::datatypes::values::Value;
use crate::graph::schema::{InternedKey, TemporalConfig};

/// The `from` properties rows of one declared relationship type key on, in
/// declaration order: one for a source-keyed or single declaration, several
/// for a legacy type holding more than one unkeyed declaration. A row keys on
/// the first of them it carries.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct StartKey(Box<[InternedKey]>);

/// A row's or stored relationship's start as the merge key compares it: the
/// `from` property it was read from and the value, `None` when none is
/// carried or all are NULL.
pub(crate) type Start = Option<(InternedKey, StartValue)>;

/// A start value, normalised so one instant compares equal however it was
/// written: a date, a datetime at midnight and an ISO string of either are
/// the same day; any other datetime is exact. A value the evaluator cannot
/// read keys on itself — rows loaded onto a declared type are not validated.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) enum StartValue {
    At(Instant),
    Raw(Value),
}

impl StartValue {
    /// `None` for NULL.
    pub(crate) fn of(value: &Value) -> Option<Self> {
        if matches!(value, Value::Null) {
            return None;
        }
        Some(match parse_instant(value) {
            Ok(Instant::Timestamp(ts)) if ts.time() == NaiveTime::MIN => {
                StartValue::At(Instant::Date(ts.date()))
            }
            Ok(instant) => StartValue::At(instant),
            Err(_) => StartValue::Raw(value.clone()),
        })
    }
}

impl StartKey {
    pub(super) fn of<'c>(configs: impl IntoIterator<Item = &'c TemporalConfig>) -> Option<Self> {
        let mut keys: Vec<InternedKey> = Vec::new();
        for config in configs {
            let key = InternedKey::from_str(&config.valid_from);
            if !keys.contains(&key) {
                keys.push(key);
            }
        }
        (!keys.is_empty()).then(|| StartKey(keys.into_boxed_slice()))
    }

    /// The `from` properties, in the order a row is keyed by.
    pub(crate) fn keys(&self) -> &[InternedKey] {
        &self.0
    }

    /// The start `cell` gives: the first key whose value is present and not
    /// NULL.
    pub(crate) fn read<V: Borrow<Value>>(
        &self,
        mut cell: impl FnMut(InternedKey) -> Option<V>,
    ) -> Start {
        self.0.iter().find_map(|&key| {
            let value = cell(key)?;
            StartValue::of(value.borrow()).map(|start| (key, start))
        })
    }

    /// The start of a row given as interned property pairs.
    pub(crate) fn of_properties(&self, properties: &[(InternedKey, Value)]) -> Start {
        self.read(|key| {
            properties
                .iter()
                .find(|(k, _)| *k == key)
                .map(|(_, value)| value)
        })
    }
}
