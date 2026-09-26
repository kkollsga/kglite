//! The temporal declaration store: which two properties bound each node
//! label's or relationship type's validity interval, and under which
//! convention.
//!
//! Relationship configs are kept as an ordered list per type because the
//! fluent `traverse()` picks the first config whose properties an edge carries
//! (see [`super::is_temporally_valid_multi`]); insertion order is therefore
//! part of the answer. How the store is saved and loaded is `persist.rs`.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use super::eval::IntervalConvention;
use super::merge_key::StartKey;
use super::validate::{self, Walk};
use crate::graph::dir_graph::DirGraph;
use crate::graph::schema::TemporalConfig;

/// What a declaration is about.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum TemporalTarget {
    /// A node label: a primary type or a secondary label.
    Node(String),
    /// A relationship type, optionally only the relationships leaving nodes of
    /// `source_type`. One type may carry several source-keyed declarations
    /// when its sources store their bounds under different properties. A
    /// relationship takes its source's keyed declaration first, and the
    /// unkeyed one only when its source has none.
    Relationship {
        rel_type: String,
        source_type: Option<String>,
    },
}

impl TemporalTarget {
    /// `node label 'X'`, `relationship type 'R'`, or `relationship type 'R'
    /// from source type 'S'` — the phrase every message names a target by.
    pub(crate) fn describe(&self) -> String {
        match self {
            TemporalTarget::Node(label) => format!("node label '{label}'"),
            TemporalTarget::Relationship {
                rel_type,
                source_type: None,
            } => format!("relationship type '{rel_type}'"),
            TemporalTarget::Relationship {
                rel_type,
                source_type: Some(source),
            } => format!("relationship type '{rel_type}' from source type '{source}'"),
        }
    }

    fn source_type(&self) -> Option<&str> {
        match self {
            TemporalTarget::Node(_) => None,
            TemporalTarget::Relationship { source_type, .. } => source_type.as_deref(),
        }
    }
}

/// The outcome of [`declare`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeclareReport {
    /// `false` when an identical declaration was already in place for the same
    /// key (the node label, or the relationship type plus its source type).
    pub changed: bool,
    /// Rows validated. 0 for a no-op, which validates nothing.
    pub rows: usize,
    /// Rows whose `to` bound equals another row's `from` bound within the same
    /// label (nodes) or the same source node (relationships), counted at
    /// declare time. `None` when not counted: a no-op, or a disk-mode label
    /// above [`DISK_NODE_ABUTMENT_CAP`] rows.
    pub abutting_rows: Option<usize>,
    /// The advisory a closed declaration with abutting rows earns.
    pub warning: Option<String>,
}

/// One entry of [`list`].
#[derive(Clone, Debug, PartialEq)]
pub struct DeclarationInfo {
    pub target: TemporalTarget,
    pub config: TemporalConfig,
    /// The declare-time count [`DeclareReport::abutting_rows`] reported,
    /// kept across save and load; `None` for a config a loader or
    /// `set_temporal` wrote, and for one read from a file written before
    /// declarations were saved.
    pub abutting_rows: Option<usize>,
    /// The relationship type holds several different configs none of which
    /// names a source type — possible only for configs written before source
    /// types existed, or by `set_temporal` — so which one applies to an edge
    /// depends on the order they were added in. Re-declare them per
    /// `source_type` to resolve it. Always `false` for a node label.
    pub ambiguous: bool,
}

/// The largest disk-mode node label whose abutting rows are counted. Counting
/// holds every row's bounds at once, and disk mode keeps no heap structure
/// that grows with the graph beyond a fixed ceiling; above it the declaration
/// is still validated in full and the count is reported as not computed.
pub const DISK_NODE_ABUTMENT_CAP: usize = 250_000;

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub(crate) struct TemporalDeclarations {
    pub(super) nodes: HashMap<String, TemporalConfig>,
    pub(super) edges: HashMap<String, Vec<TemporalConfig>>,
    /// Declare-time counts. Saved with the declarations by `persist.rs`, not
    /// through this derive.
    #[serde(skip)]
    pub(super) abutting: HashMap<TemporalTarget, usize>,
}

/// What a declaration does to the store once validated.
pub(super) enum Change {
    Unchanged,
    Insert,
}

/// Nodes first, then by name; within a relationship type the source-keyed
/// declarations (by source) before the unkeyed one — the order a lookup
/// tries them in.
fn lookup_order(target: &TemporalTarget) -> (u8, &str, bool, Option<&str>) {
    match target {
        TemporalTarget::Node(label) => (0, label, false, None),
        TemporalTarget::Relationship {
            rel_type,
            source_type,
        } => (1, rel_type, source_type.is_none(), source_type.as_deref()),
    }
}

fn same_properties(a: &TemporalConfig, b: &TemporalConfig) -> bool {
    a.valid_from == b.valid_from && a.valid_to == b.valid_to
}

fn same_interval(a: &TemporalConfig, b: &TemporalConfig) -> bool {
    same_properties(a, b) && a.convention == b.convention
}

impl TemporalDeclarations {
    pub(crate) fn node(&self, label: &str) -> Option<&TemporalConfig> {
        self.nodes.get(label)
    }

    /// Every config of `rel_type`, in declaration order; empty when none.
    pub(crate) fn edges(&self, rel_type: &str) -> &[TemporalConfig] {
        self.edges.get(rel_type).map_or(&[], Vec::as_slice)
    }

    /// Whether `rel_type` holds more than one unkeyed config (see
    /// [`DeclarationInfo::ambiguous`]).
    pub(crate) fn is_ambiguous(&self, rel_type: &str) -> bool {
        self.edges(rel_type)
            .iter()
            .filter(|c| c.source_type.is_none())
            .nth(1)
            .is_some()
    }

    pub(crate) fn abutting(&self, target: &TemporalTarget) -> Option<usize> {
        self.abutting.get(target).copied()
    }

    /// The key a load of `target`'s rows declares under (see
    /// `loader::declare_from_column_types`): a source-keyed relationship
    /// target becomes unkeyed unless its source has a declaration of its own
    /// or the unkeyed one names other properties.
    pub(super) fn load_target(
        &self,
        target: TemporalTarget,
        valid_from: &str,
        valid_to: &str,
    ) -> TemporalTarget {
        let TemporalTarget::Relationship {
            rel_type,
            source_type: Some(source),
        } = target
        else {
            return target;
        };
        let configs = self.edges(&rel_type);
        let keyed = configs
            .iter()
            .any(|c| c.source_type.as_deref() == Some(source.as_str()));
        let fallbacks = || configs.iter().filter(|c| c.source_type.is_none());
        let other_fallback = fallbacks().next().is_some()
            && !fallbacks().any(|c| c.valid_from == valid_from && c.valid_to == valid_to);
        TemporalTarget::Relationship {
            rel_type,
            source_type: (keyed || other_fallback).then_some(source),
        }
    }

    /// The convention a declaration of `valid_from`/`valid_to` for `target`
    /// takes when its caller names none: that of a declaration of the same
    /// key already naming the same properties, else closed.
    pub(super) fn default_convention(
        &self,
        target: &TemporalTarget,
        valid_from: &str,
        valid_to: &str,
    ) -> IntervalConvention {
        let same = |c: &&TemporalConfig| c.valid_from == valid_from && c.valid_to == valid_to;
        let existing = match target {
            TemporalTarget::Node(label) => self.nodes.get(label).filter(same),
            TemporalTarget::Relationship {
                rel_type,
                source_type,
            } => self
                .edges(rel_type)
                .iter()
                .filter(|c| c.source_type == *source_type)
                .find(same),
        };
        existing.map_or(IntervalConvention::Closed, |c| c.convention)
    }

    /// Decide what declaring `config` for `target` does: nothing (an identical
    /// declaration holds the same key), an insert, or a conflict error. The
    /// key is the node label, or the relationship type plus its source type —
    /// an unkeyed relationship declaration is its own key, the fallback for
    /// sources without one, so it never conflicts with a keyed one.
    pub(super) fn change_for(
        &self,
        target: &TemporalTarget,
        config: &TemporalConfig,
    ) -> Result<Change, String> {
        let same_key: Vec<&TemporalConfig> = match target {
            TemporalTarget::Node(label) => self.nodes.get(label).into_iter().collect(),
            TemporalTarget::Relationship {
                rel_type,
                source_type,
            } => self
                .edges(rel_type)
                .iter()
                .filter(|c| c.source_type == *source_type)
                .collect(),
        };
        // A legacy list can hold several distinct unkeyed configs; one
        // identical to the declaration makes it a no-op.
        match same_key.first() {
            None => Ok(Change::Insert),
            Some(_) if same_key.iter().any(|c| same_interval(c, config)) => Ok(Change::Unchanged),
            Some(existing) => Err(format!(
                "{} is already declared with from '{}', to '{}', convention '{}'. \
                 Undeclare it first to change it.",
                target.describe(),
                existing.valid_from,
                existing.valid_to,
                existing.convention.as_str()
            )),
        }
    }

    /// Source types with a keyed declaration of `rel_type` — the sources an
    /// unkeyed declaration does not apply to.
    pub(crate) fn keyed_sources(&self, rel_type: &str) -> Vec<&str> {
        self.edges(rel_type)
            .iter()
            .filter_map(|c| c.source_type.as_deref())
            .collect()
    }

    pub(super) fn insert(
        &mut self,
        target: &TemporalTarget,
        config: TemporalConfig,
        abutting: Option<usize>,
    ) {
        match target {
            TemporalTarget::Node(label) => {
                self.nodes.insert(label.clone(), config);
            }
            TemporalTarget::Relationship { rel_type, .. } => {
                self.edges.entry(rel_type.clone()).or_default().push(config);
            }
        }
        match abutting {
            Some(count) => self.abutting.insert(target.clone(), count),
            None => self.abutting.remove(target),
        };
    }

    pub(super) fn remove(&mut self, target: &TemporalTarget) -> bool {
        self.abutting.remove(target);
        match target {
            TemporalTarget::Node(label) => self.nodes.remove(label).is_some(),
            TemporalTarget::Relationship {
                rel_type,
                source_type,
            } => {
                let Some(configs) = self.edges.get_mut(rel_type) else {
                    return false;
                };
                let before = configs.len();
                configs.retain(|c| c.source_type != *source_type);
                let removed = configs.len() != before;
                if configs.is_empty() {
                    self.edges.remove(rel_type);
                }
                removed
            }
        }
    }

    pub(super) fn entries(&self) -> Vec<DeclarationInfo> {
        let nodes = self
            .nodes
            .iter()
            .map(|(label, config)| (TemporalTarget::Node(label.clone()), config));
        let edges = self.edges.iter().flat_map(|(rel_type, configs)| {
            configs.iter().map(move |config| {
                (
                    TemporalTarget::Relationship {
                        rel_type: rel_type.clone(),
                        source_type: config.source_type.clone(),
                    },
                    config,
                )
            })
        });
        let mut out: Vec<DeclarationInfo> = nodes
            .chain(edges)
            .map(|(target, config)| DeclarationInfo {
                abutting_rows: self.abutting(&target),
                ambiguous: match &target {
                    TemporalTarget::Relationship {
                        rel_type,
                        source_type: None,
                    } => self.is_ambiguous(rel_type),
                    _ => false,
                },
                target,
                config: config.clone(),
            })
            .collect();
        out.sort_by(|a, b| lookup_order(&a.target).cmp(&lookup_order(&b.target)));
        out
    }
}

/// Declare which two properties bound `target`'s validity interval, and
/// whether the `to` day belongs to it.
///
/// Both properties must exist on the target. Every stored bound is read under
/// the rule `valid_at` uses; the first one that is not NULL, a date, a
/// datetime or an ISO date string is refused, naming its element, and so is a
/// row whose interval is inverted (`from > to`, or `from == to` under
/// half-open, which would be empty). Re-declaring an identical interval is a
/// no-op; a different one for the same target is refused. A real change bumps
/// the graph version.
pub fn declare(
    graph: &mut DirGraph,
    target: &TemporalTarget,
    valid_from: &str,
    valid_to: &str,
    convention: IntervalConvention,
) -> Result<DeclareReport, String> {
    declare_loaded(graph, target, valid_from, valid_to, convention, &[])
}

/// [`declare`] for a loader that has just written the bound columns. The
/// schema records a property only once some row holds a value for it, so a
/// column the load wrote entirely NULL (every period still open) is absent
/// from it; naming it in `written` lets it count as existing. A manual
/// declaration has no such list, so a mistyped property is still refused.
pub fn declare_loaded(
    graph: &mut DirGraph,
    target: &TemporalTarget,
    valid_from: &str,
    valid_to: &str,
    convention: IntervalConvention,
    written: &[&str],
) -> Result<DeclareReport, String> {
    let config = TemporalConfig {
        valid_from: valid_from.to_string(),
        valid_to: valid_to.to_string(),
        convention,
        source_type: target.source_type().map(str::to_string),
    };
    validate::check_target(graph, target)?;
    let change = graph.temporal.change_for(target, &config)?;
    if matches!(change, Change::Unchanged) {
        return Ok(DeclareReport {
            changed: false,
            rows: 0,
            abutting_rows: None,
            warning: None,
        });
    }
    let Walk { rows, abutting } = validate::walk(graph, target, &config, written)?;
    let warning = match abutting {
        Some(count) if count > 0 && convention == IntervalConvention::Closed => {
            Some(validate::abutment_warning(target, count, rows))
        }
        _ => None,
    };
    record_insert(graph, target, config, abutting);
    graph.bump_version();
    Ok(DeclareReport {
        changed: true,
        rows,
        abutting_rows: abutting,
        warning,
    })
}

/// Remove `target`'s declaration. `false`, and no version bump, when there
/// was none. An unkeyed relationship target removes only the unkeyed
/// declaration, never a source-keyed one.
pub fn undeclare(graph: &mut DirGraph, target: &TemporalTarget) -> bool {
    let removed = record_remove(graph, target);
    if removed {
        graph.bump_version();
    }
    removed
}

/// Install `config` for `target`, journaling the change. Every route that
/// writes the store goes through this or [`record_remove`], so a durable graph
/// recovers what a crash before the next checkpoint would otherwise lose.
pub(super) fn record_insert(
    graph: &mut DirGraph,
    target: &TemporalTarget,
    config: TemporalConfig,
    abutting: Option<usize>,
) {
    graph.note_temporal_declaration(target, Some((&config, abutting)));
    graph.temporal.insert(target, config, abutting);
}

/// Remove `target`'s declaration, journaling the withdrawal when there was
/// one to remove.
pub(super) fn record_remove(graph: &mut DirGraph, target: &TemporalTarget) -> bool {
    let removed = graph.temporal.remove(target);
    if removed {
        graph.note_temporal_declaration(target, None);
    }
    removed
}

/// Every declaration: nodes by label, then relationship types by name, each
/// type's source-keyed declarations (by source) before its unkeyed one — the
/// order a relationship's lookup tries them in.
pub fn list(graph: &DirGraph) -> Vec<DeclarationInfo> {
    graph.temporal.entries()
}

/// The config declared for node label `label`, if any.
pub fn node_config<'g>(graph: &'g DirGraph, label: &str) -> Option<&'g TemporalConfig> {
    graph.temporal.node(label)
}

/// Every config declared for relationship type `rel_type`, in declaration
/// order — the order the fluent `traverse()` filter tries them in.
pub fn edge_configs<'g>(graph: &'g DirGraph, rel_type: &str) -> &'g [TemporalConfig] {
    graph.temporal.edges(rel_type)
}

/// The `from` properties a bulk load of `rel_type` relationships from
/// `source_type` adds to its merge key, so that a row starting a different
/// period between the same endpoints becomes a parallel relationship instead
/// of merging into the stored one: the source's keyed declaration's, else
/// every unkeyed declaration's in declaration order (a legacy type can hold
/// several; each row keys on the first it carries). `None` as the source takes
/// only the unkeyed declarations, and `None` is returned when none applies.
pub(crate) fn merge_start_key(
    graph: &DirGraph,
    rel_type: &str,
    source_type: Option<&str>,
) -> Option<StartKey> {
    let configs = graph.temporal.edges(rel_type);
    let keyed = source_type.and_then(|source| {
        configs
            .iter()
            .find(|c| c.source_type.as_deref() == Some(source))
    });
    match keyed {
        Some(config) => StartKey::of([config]),
        None => StartKey::of(configs.iter().filter(|c| c.source_type.is_none())),
    }
}
