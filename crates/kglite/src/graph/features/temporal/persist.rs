//! How declarations travel through the `.kgl` metadata block.
//!
//! Every declaration is written as one entry of the `temporal_declarations`
//! key, the only key a current build reads when it is present. The two keys
//! older builds read, `temporal_node_configs` and `temporal_edge_configs`,
//! carry a mirror of the declarations an older build applies as this one
//! does: a closed node config, and a relationship type whose configs are all
//! closed and unkeyed, written as the whole list in order, each without the
//! fields older builds do not know. Both builds apply the first unkeyed
//! config whose properties an edge carries, so an ambiguous list means the
//! same to either. A type with a half-open or a source-keyed config is left
//! out: an older build reads every config as closed, so a half-open one would
//! keep each interval's end day, and it knows no source types, so a keyed one
//! would reach other sources' edges. A type it sees as undeclared is filtered
//! by nothing instead.
//!
//! A file without the new key (written before it existed) is read from the
//! legacy keys. That is read-compatibility for persisted data, kept as long
//! as such files can exist. A legacy relationship list repeating one config
//! reads as that config once; one holding several different configs reads as
//! all of them, in order, and the type is ambiguous (see
//! [`TemporalDeclarations::is_ambiguous`]).

use std::collections::HashMap;

use serde::{Deserialize, Deserializer, Serialize};

use super::declarations::{TemporalDeclarations, TemporalTarget};
use super::eval::IntervalConvention;
use crate::graph::schema::TemporalConfig;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Kind {
    Node,
    Relationship,
}

/// One entry of the `temporal_declarations` key:
/// `{"kind", "name", "source_type"?, "from", "to", "convention",
/// "abutting_rows"?}`. A field added later must be optional with a default,
/// so a file stays readable in both directions without a format change.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub(crate) struct PersistedDeclaration {
    kind: Kind,
    name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    source_type: Option<String>,
    from: String,
    to: String,
    convention: IntervalConvention,
    /// The declare-time count, so a loaded graph reports what the
    /// declaration reported.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    abutting_rows: Option<usize>,
}

/// Read the `temporal_declarations` list, dropping an entry this build
/// cannot read (a convention or kind a newer build added) rather than
/// refusing the whole file: a type this build cannot interpret stays
/// undeclared here.
pub(crate) fn lenient_entries<'de, D>(
    deserializer: D,
) -> Result<Vec<PersistedDeclaration>, D::Error>
where
    D: Deserializer<'de>,
{
    let raw = Vec::<serde_json::Value>::deserialize(deserializer)?;
    Ok(raw
        .into_iter()
        .filter_map(|entry| serde_json::from_value(entry).ok())
        .collect())
}

/// The two legacy metadata maps: node label → config, relationship type →
/// configs.
pub(crate) type LegacyMaps = (
    HashMap<String, TemporalConfig>,
    HashMap<String, Vec<TemporalConfig>>,
);

fn legacy_config(from: &str, to: &str) -> TemporalConfig {
    TemporalConfig {
        valid_from: from.to_string(),
        valid_to: to.to_string(),
        convention: IntervalConvention::Closed,
        source_type: None,
    }
}

impl TemporalDeclarations {
    /// Every declaration in [`super::list`] order, which is deterministic, so
    /// the same store always writes the same bytes.
    pub(crate) fn to_persisted(&self) -> Vec<PersistedDeclaration> {
        self.entries()
            .into_iter()
            .map(|info| {
                let (kind, name, source_type) = match info.target {
                    TemporalTarget::Node(label) => (Kind::Node, label, None),
                    TemporalTarget::Relationship {
                        rel_type,
                        source_type,
                    } => (Kind::Relationship, rel_type, source_type),
                };
                PersistedDeclaration {
                    kind,
                    name,
                    source_type,
                    from: info.config.valid_from,
                    to: info.config.valid_to,
                    convention: info.config.convention,
                    abutting_rows: info.abutting_rows,
                }
            })
            .collect()
    }

    /// The legacy maps an older build reads, holding only what it applies as
    /// this build does (module docs).
    pub(crate) fn legacy_mirror(&self) -> LegacyMaps {
        let nodes = self
            .nodes
            .iter()
            .filter(|(_, config)| config.convention.is_closed())
            .map(|(label, config)| {
                (
                    label.clone(),
                    legacy_config(&config.valid_from, &config.valid_to),
                )
            })
            .collect();
        let edges = self
            .edges
            .iter()
            .filter(|(_, configs)| {
                configs
                    .iter()
                    .all(|config| config.convention.is_closed() && config.source_type.is_none())
            })
            .map(|(rel_type, configs)| {
                let legacy = configs
                    .iter()
                    .map(|config| legacy_config(&config.valid_from, &config.valid_to))
                    .collect();
                (rel_type.clone(), legacy)
            })
            .collect();
        (nodes, edges)
    }

    /// Rebuild the store from a file: the `temporal_declarations` entries
    /// when the file has any, else the legacy maps.
    pub(crate) fn from_file(entries: Vec<PersistedDeclaration>, legacy: LegacyMaps) -> Self {
        if entries.is_empty() {
            return Self::from_legacy(legacy);
        }
        let mut store = TemporalDeclarations::default();
        for entry in entries {
            let config = TemporalConfig {
                valid_from: entry.from,
                valid_to: entry.to,
                convention: entry.convention,
                source_type: match entry.kind {
                    Kind::Node => None,
                    Kind::Relationship => entry.source_type,
                },
            };
            let target = match entry.kind {
                Kind::Node => TemporalTarget::Node(entry.name.clone()),
                Kind::Relationship => TemporalTarget::Relationship {
                    rel_type: entry.name.clone(),
                    source_type: config.source_type.clone(),
                },
            };
            match entry.kind {
                Kind::Node => {
                    store.nodes.insert(entry.name, config);
                }
                Kind::Relationship => {
                    let configs = store.edges.entry(entry.name).or_default();
                    if !configs.contains(&config) {
                        configs.push(config);
                    }
                }
            }
            if let Some(count) = entry.abutting_rows {
                store.abutting.insert(target, count);
            }
        }
        store
    }

    fn from_legacy((nodes, edges): LegacyMaps) -> Self {
        let mut store = TemporalDeclarations {
            nodes,
            ..TemporalDeclarations::default()
        };
        for (rel_type, configs) in edges {
            for config in configs {
                store.legacy_push_edge(rel_type.clone(), config);
            }
        }
        store
    }
}
