//! Temporal validity filtering of nodes and edges for the fluent API. Every
//! check delegates to [`eval`], the evaluator Cypher `valid_at` /
//! `valid_during` use, so bounds stored as dates, datetimes or ISO strings
//! answer the same on every surface. A bound that holds anything else is an
//! error naming the element and the property, not a pass.

pub(crate) mod declarations;
#[cfg(test)]
mod declarations_tests;
pub(crate) mod eval;
mod loader;
#[cfg(test)]
mod loader_tests;
mod merge_key;
#[cfg(test)]
mod merge_key_tests;
pub(crate) mod persist;
#[cfg(test)]
mod persist_tests;
mod validate;

pub(crate) use declarations::merge_start_key;
pub use declarations::{
    declare, declare_loaded, edge_configs, list, node_config, undeclare, DeclarationInfo,
    DeclareReport, TemporalTarget, DISK_NODE_ABUTMENT_CAP,
};
pub use eval::IntervalConvention;
pub(crate) use loader::{adopt_declarations, settle_adopted, withdraw_adopted};
pub use loader::{declare_defaulted, declare_from_column_types, LoadDeclaration};
pub(crate) use merge_key::{Start, StartKey};

use crate::datatypes::values::Value;
use crate::graph::schema::{InternedKey, TemporalConfig};
use chrono::NaiveDate;
use eval::{BoundSide, Instant, TemporalError};

/// `property 'vf': the from bound … is not a date …` — the element prefix is
/// the caller's, since only it knows which element the properties belong to.
fn describe(err: TemporalError, config: &TemporalConfig) -> String {
    let property = match &err {
        TemporalError::Bound {
            side: BoundSide::From,
            ..
        } => &config.valid_from,
        TemporalError::Bound {
            side: BoundSide::To,
            ..
        } => &config.valid_to,
        TemporalError::Instant { .. } => return err.to_string(),
    };
    format!("property '{property}': {err}")
}

fn node_error(
    node: crate::graph::storage::NodeView<'_>,
    err: TemporalError,
    config: &TemporalConfig,
) -> String {
    let id = crate::graph::core::value_operations::format_value_compact(&node.id());
    format!("node '{id}', {}", describe(err, config))
}

fn property<'p>(properties: &'p [(InternedKey, Value)], name: &str) -> &'p Value {
    let key = InternedKey::from_str(name);
    properties
        .iter()
        .find(|(k, _)| *k == key)
        .map_or(&Value::Null, |(_, v)| v)
}

/// Whether edge properties are valid at `reference`: `valid_from <= reference
/// <= valid_to`, a NULL or missing bound open.
pub fn is_temporally_valid(
    properties: &[(InternedKey, Value)],
    config: &TemporalConfig,
    reference: &NaiveDate,
) -> Result<bool, String> {
    eval::interval_contains(
        property(properties, &config.valid_from),
        property(properties, &config.valid_to),
        Instant::Date(*reference),
        config.convention,
    )
    .map_err(|e| describe(e, config))
}

/// Whether a node is valid at `reference`. Bounds are read through
/// `get_field_ref()`, so `id` / `title` fields work as bounds too.
pub fn node_is_temporally_valid(
    node: crate::graph::storage::NodeView<'_>,
    config: &TemporalConfig,
    reference: &NaiveDate,
) -> Result<bool, String> {
    let from = node.get_field_ref(&config.valid_from);
    let to = node.get_field_ref(&config.valid_to);
    eval::interval_contains(
        from.as_deref().unwrap_or(&Value::Null),
        to.as_deref().unwrap_or(&Value::Null),
        Instant::Date(*reference),
        config.convention,
    )
    .map_err(|e| node_error(node, e, config))
}

/// Whether edge properties' validity overlaps `[start, end]`.
pub fn overlaps_range(
    properties: &[(InternedKey, Value)],
    config: &TemporalConfig,
    start: &NaiveDate,
    end: &NaiveDate,
) -> Result<bool, String> {
    eval::interval_overlaps(
        property(properties, &config.valid_from),
        property(properties, &config.valid_to),
        Instant::Date(*start),
        Instant::Date(*end),
        config.convention,
    )
    .map_err(|e| describe(e, config))
}

/// Whether a node's validity overlaps `[start, end]`.
pub fn node_overlaps_range(
    node: crate::graph::storage::NodeView<'_>,
    config: &TemporalConfig,
    start: &NaiveDate,
    end: &NaiveDate,
) -> Result<bool, String> {
    let from = node.get_field_ref(&config.valid_from);
    let to = node.get_field_ref(&config.valid_to);
    eval::interval_overlaps(
        from.as_deref().unwrap_or(&Value::Null),
        to.as_deref().unwrap_or(&Value::Null),
        Instant::Date(*start),
        Instant::Date(*end),
        config.convention,
    )
    .map_err(|e| node_error(node, e, config))
}

/// The config an edge leaving a node of type `source` is filtered by: that
/// source's keyed config when it has one, otherwise the first unkeyed config
/// whose `valid_from` or `valid_to` field the edge carries. `None` — the edge
/// is not temporal — when neither applies, including a keyed config whose
/// fields the edge does not carry.
fn matching_config<'c>(
    properties: &[(InternedKey, Value)],
    configs: &'c [TemporalConfig],
    source: Option<InternedKey>,
) -> Option<&'c TemporalConfig> {
    let carries = |config: &TemporalConfig| {
        let from_key = InternedKey::from_str(&config.valid_from);
        let to_key = InternedKey::from_str(&config.valid_to);
        properties
            .iter()
            .any(|(k, _)| *k == from_key || *k == to_key)
    };
    let keyed = configs.iter().find(|config| {
        config
            .source_type
            .as_deref()
            .is_some_and(|keyed| source == Some(InternedKey::from_str(keyed)))
    });
    match keyed {
        Some(config) => carries(config).then_some(config),
        None => configs
            .iter()
            .find(|config| config.source_type.is_none() && carries(config)),
    }
}

/// [`is_temporally_valid`] under the config `matching_config` picks for the
/// edge; `true` when none applies.
pub fn is_temporally_valid_multi(
    properties: &[(InternedKey, Value)],
    configs: &[TemporalConfig],
    source: Option<InternedKey>,
    reference: &NaiveDate,
) -> Result<bool, String> {
    match matching_config(properties, configs, source) {
        Some(config) => is_temporally_valid(properties, config, reference),
        None => Ok(true),
    }
}

/// [`overlaps_range`] under the same config choice as
/// [`is_temporally_valid_multi`].
pub fn overlaps_range_multi(
    properties: &[(InternedKey, Value)],
    configs: &[TemporalConfig],
    source: Option<InternedKey>,
    start: &NaiveDate,
    end: &NaiveDate,
) -> Result<bool, String> {
    match matching_config(properties, configs, source) {
        Some(config) => overlaps_range(properties, config, start, end),
        None => Ok(true),
    }
}

/// Whether a node passes the fluent temporal context; `All` passes everything.
pub fn node_passes_context(
    node: crate::graph::storage::NodeView<'_>,
    config: &TemporalConfig,
    context: &crate::graph::TemporalContext,
) -> Result<bool, String> {
    use crate::graph::TemporalContext;
    match context {
        TemporalContext::All => Ok(true),
        TemporalContext::Today => {
            let today = chrono::Local::now().date_naive();
            node_is_temporally_valid(node, config, &today)
        }
        TemporalContext::At(d) => node_is_temporally_valid(node, config, d),
        TemporalContext::During(start, end) => node_overlaps_range(node, config, start, end),
    }
}
