//! Python query policy, captured when a handle is derived from another.
//!
//! The policy struct itself, the resolution rule and the 180 s default live in
//! [`kglite_core::api::session`] — the MCP server applies the same default, and
//! a second copy of the constant is how two surfaces come to disagree about
//! what "the default" is. What stays here is the Python-flavoured half: the
//! `set_default_*` capture on `KnowledgeGraph` and the derived-handle
//! constructor.
use crate::graph::embedder::Embedder;
use crate::graph::{CursorState, GraphLifecycle, KnowledgeGraph};
use kglite_core::api::{CowSelection, DirGraph};
use std::sync::Arc;

pub(crate) use kglite_core::api::session::{deadline_from, QueryDefaults};

impl KnowledgeGraph {
    /// The captured policy this handle runs its own queries under, and the
    /// one a derived handle inherits.
    pub(crate) fn query_defaults(&self) -> QueryDefaults {
        QueryDefaults {
            timeout_ms: self.default_timeout_ms,
            max_work_units: self.default_max_work_units,
            row_limit: self.default_row_limit,
        }
    }

    /// Write a captured policy onto a handle. The wheel keeps the three
    /// fields flat on `KnowledgeGraph` (they back the `set_default_*`
    /// pymethods), so this is the Python-side half of the core struct.
    pub(crate) fn apply_query_defaults(&mut self, defaults: QueryDefaults) {
        self.default_timeout_ms = defaults.timeout_ms;
        self.default_max_work_units = defaults.max_work_units;
        self.default_row_limit = defaults.row_limit;
    }

    /// The one constructor for a handle derived from this one. A derived
    /// handle is the same caller under a new name, so it carries the captured
    /// query policy — building the struct literal by hand instead is how
    /// `to_subgraph` came to answer a thousand rows under a ten-row cap.
    /// Graphs that are *not* derived (`kglite.open`, a blueprint load) keep
    /// their own literal and start from the unset defaults.
    pub(crate) fn derive_handle(
        &self,
        inner: Arc<DirGraph>,
        cursor: CursorState,
        embedder: Option<Arc<dyn Embedder>>,
        lifecycle: GraphLifecycle,
    ) -> Self {
        let mut derived = KnowledgeGraph {
            inner,
            cursor,
            embedder,
            default_timeout_ms: None,
            default_max_work_units: None,
            default_row_limit: None,
            lifecycle,
        };
        derived.apply_query_defaults(self.query_defaults());
        derived
    }

    /// A handle onto the same graph for an operation that has already landed
    /// on `self.inner`: the reports and temporal context follow, the selection
    /// only when the caller asked to keep it, and the lifecycle is detached so
    /// `self` stays the owner of the durability state.
    pub(crate) fn detached_view(&self, keep_selection: bool) -> Self {
        self.derive_handle(
            self.inner.clone(),
            CursorState {
                selection: if keep_selection {
                    self.cursor.selection.clone()
                } else {
                    CowSelection::new()
                },
                reports: self.cursor.reports.clone(),
                last_mutation_stats: None,
                temporal_context: self.cursor.temporal_context.clone(),
            },
            self.embedder.as_ref().map(Arc::clone),
            self.detached_view_lifecycle(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The resolution rule itself is tested in the core; this pins that the
    /// wheel's capture reaches it — a `set_default_timeout(0)` must stay "no
    /// deadline" rather than collapsing into "inherit the 180 s default".
    #[test]
    fn a_captured_zero_timeout_survives_the_core_resolution() {
        let policy = QueryDefaults {
            timeout_ms: Some(0),
            max_work_units: None,
            row_limit: Some(2),
        };
        let inherited = policy.resolve(None, None, None);
        assert!(inherited.deadline.is_none());
        assert_eq!(inherited.row_limit, Some(2));
        assert!(deadline_from(Some(0)).is_none());
    }
}
