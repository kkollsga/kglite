//! Declarations the bulk writers make: `set_temporal` and a load whose
//! `validFrom`/`validTo` column types name the bounds, which may leave the
//! convention out and are checked before the load writes, and `extend`, which
//! carries another graph's declarations over with its rows.

use super::declarations::{
    self, record_insert, record_remove, Change, DeclarationInfo, DeclareReport, TemporalTarget,
};
use super::eval::IntervalConvention;
use super::validate;
use crate::datatypes::values::DataFrame;
use crate::graph::dir_graph::DirGraph;
use crate::graph::schema::TemporalConfig;

/// [`declarations::declare`] with an optional convention. Without one, a
/// declaration of the same key already naming the same properties keeps its
/// convention (so the call is a no-op), and a new declaration is closed.
pub fn declare_defaulted(
    graph: &mut DirGraph,
    target: &TemporalTarget,
    valid_from: &str,
    valid_to: &str,
    convention: Option<IntervalConvention>,
) -> Result<DeclareReport, String> {
    let convention = convention.unwrap_or_else(|| {
        graph
            .temporal
            .default_convention(target, valid_from, valid_to)
    });
    declarations::declare(graph, target, valid_from, valid_to, convention)
}

/// A bulk load's declaration, between [`declare_from_column_types`] (before
/// the load writes) and [`LoadDeclaration::finish`] (after it has).
#[must_use = "finish the declaration after the load, or abandon it when the load fails"]
pub struct LoadDeclaration {
    target: TemporalTarget,
    config: TemporalConfig,
    /// The bound columns the load carries.
    written: Vec<String>,
    stage: Stage,
}

enum Stage {
    /// Declared already — before the load, or identically by an earlier one.
    Declared(DeclareReport),
    /// Installed unvalidated because the load creates the target's first
    /// rows: the merge key applies to the load, and `finish` validates.
    Installed,
}

/// Start the declaration a load's `validFrom`/`validTo` column types ask for,
/// before the load writes `frame`. `convention` defaults as in
/// [`declare_defaulted`].
///
/// An identical declaration makes the whole thing a no-op: rows written onto
/// a declared target are not validated. Otherwise, before anything is
/// written, a conflicting declaration of the same key is refused, and so is a
/// row of `frame` holding an unreadable, inverted or empty interval (named by
/// its position). A target that already exists is then declared at once —
/// its stored rows validated; a target whose first rows the load writes is
/// installed unvalidated and validated by [`LoadDeclaration::finish`]. Either
/// way the load itself merges under the declared relationship key.
///
/// A relationship load names its source type. It declares the type-wide
/// (unkeyed) declaration — the one every build before source types applied —
/// unless the source already has its own, or the type-wide one names other
/// properties; then it declares for its source.
pub fn declare_from_column_types(
    graph: &mut DirGraph,
    target: TemporalTarget,
    valid_from: &str,
    valid_to: &str,
    convention: Option<IntervalConvention>,
    frame: &DataFrame,
) -> Result<LoadDeclaration, String> {
    let target = graph.temporal.load_target(target, valid_from, valid_to);
    let convention = convention.unwrap_or_else(|| {
        graph
            .temporal
            .default_convention(&target, valid_from, valid_to)
    });
    let config = TemporalConfig {
        valid_from: valid_from.to_string(),
        valid_to: valid_to.to_string(),
        convention,
        source_type: match &target {
            TemporalTarget::Node(_) => None,
            TemporalTarget::Relationship { source_type, .. } => source_type.clone(),
        },
    };
    let written = [valid_from, valid_to]
        .into_iter()
        .filter(|name| frame.verify_column(name))
        .map(str::to_string)
        .collect();
    let mut load = LoadDeclaration {
        target,
        config,
        written,
        stage: Stage::Installed,
    };
    if matches!(
        graph.temporal.change_for(&load.target, &load.config)?,
        Change::Unchanged
    ) {
        load.stage = Stage::Declared(DeclareReport {
            changed: false,
            rows: 0,
            abutting_rows: None,
            warning: None,
        });
        return Ok(load);
    }
    validate::check_frame(frame, &load.config)?;
    if validate::check_target(graph, &load.target).is_ok() {
        load.stage = Stage::Declared(load.declare(graph)?);
    } else {
        record_insert(graph, &load.target, load.config.clone(), None);
    }
    Ok(load)
}

impl LoadDeclaration {
    fn declare(&self, graph: &mut DirGraph) -> Result<DeclareReport, String> {
        let written: Vec<&str> = self.written.iter().map(String::as_str).collect();
        declarations::declare_loaded(
            graph,
            &self.target,
            &self.config.valid_from,
            &self.config.valid_to,
            self.config.convention,
            &written,
        )
    }

    /// Complete the declaration once the load has written: validate an
    /// installed one against every row the target now holds. One the rows
    /// refuse is removed.
    pub fn finish(self, graph: &mut DirGraph) -> Result<DeclareReport, String> {
        match self.stage {
            Stage::Declared(report) => Ok(report),
            Stage::Installed => {
                record_remove(graph, &self.target);
                self.declare(graph)
            }
        }
    }

    /// Withdraw what [`declare_from_column_types`] added, for a load that
    /// then failed, so a refused load leaves no declaration behind.
    pub fn abandon(self, graph: &mut DirGraph) {
        match self.stage {
            Stage::Declared(DeclareReport { changed: true, .. }) => {
                declarations::undeclare(graph, &self.target);
            }
            Stage::Declared(_) => {}
            Stage::Installed => {
                record_remove(graph, &self.target);
            }
        }
    }
}

/// Install another graph's declarations (`list` of it) ahead of the rows
/// they govern, so the merge that brings those rows keys each declared
/// relationship type on its `from` bound. Nothing is validated yet — the
/// rows have not arrived; [`settle_adopted`] validates the returned entries
/// once they have. An entry the graph already holds is left as it is; one that
/// conflicts with the graph's own declaration of the same key is not copied,
/// and says so in `errors`.
pub(crate) fn adopt_declarations(
    graph: &mut DirGraph,
    declarations: Vec<DeclarationInfo>,
    errors: &mut Vec<String>,
) -> Vec<DeclarationInfo> {
    let mut adopted = Vec::new();
    for info in declarations {
        match graph.temporal.change_for(&info.target, &info.config) {
            Ok(Change::Unchanged) => {}
            Ok(Change::Insert) => {
                record_insert(graph, &info.target, info.config.clone(), None);
                adopted.push(info);
            }
            Err(reason) => errors.push(format!(
                "the temporal declaration of {} was not copied: {reason}",
                info.target.describe()
            )),
        }
    }
    adopted
}

/// Remove what [`adopt_declarations`] installed, for a merge that failed
/// before its rows could validate them: an unvalidated adoption never outlives
/// the call.
pub(crate) fn withdraw_adopted(graph: &mut DirGraph, adopted: Vec<DeclarationInfo>) {
    for info in adopted {
        record_remove(graph, &info.target);
    }
}

/// Validate what [`adopt_declarations`] installed, now that the rows are in:
/// each is declared afresh, counting its abutting rows, and one the rows
/// refuse is removed, saying so in `errors`.
pub(crate) fn settle_adopted(
    graph: &mut DirGraph,
    adopted: Vec<DeclarationInfo>,
    errors: &mut Vec<String>,
) {
    for info in adopted {
        record_remove(graph, &info.target);
        let config = &info.config;
        let written = [config.valid_from.as_str(), config.valid_to.as_str()];
        if let Err(reason) = declarations::declare_loaded(
            graph,
            &info.target,
            &config.valid_from,
            &config.valid_to,
            config.convention,
            &written,
        ) {
            errors.push(format!(
                "the temporal declaration of {} was not copied: {reason}",
                info.target.describe()
            ));
        }
    }
}
