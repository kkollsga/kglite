//! Python query policy, captured when a handle is derived from another.
use crate::graph::embedder::Embedder;
use crate::graph::{CursorState, GraphLifecycle, KnowledgeGraph};
use kglite_core::api::{CowSelection, DirGraph};
use std::sync::Arc;
use std::time::{Duration, Instant};

const DEFAULT_TIMEOUT_MS: u64 = 180_000;

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct QueryDefaults {
    pub(crate) timeout_ms: Option<u64>,
    pub(crate) max_work_units: Option<usize>,
    pub(crate) row_limit: Option<usize>,
}

pub(crate) struct ResolvedQueryOptions {
    pub(crate) timeout_ms: Option<u64>,
    pub(crate) deadline: Option<Instant>,
    pub(crate) max_work_units: Option<usize>,
    pub(crate) row_limit: Option<usize>,
}

pub(crate) fn deadline_from(timeout_ms: Option<u64>) -> Option<Instant> {
    timeout_ms
        .filter(|ms| *ms != 0)
        .map(|ms| Instant::now() + Duration::from_millis(ms))
}

impl QueryDefaults {
    pub(crate) fn resolve(
        self,
        timeout_ms: Option<u64>,
        max_work_units: Option<usize>,
        row_limit: Option<usize>,
    ) -> ResolvedQueryOptions {
        let timeout_ms = timeout_ms.or(self.timeout_ms).or(Some(DEFAULT_TIMEOUT_MS));
        ResolvedQueryOptions {
            timeout_ms: timeout_ms.filter(|ms| *ms != 0),
            deadline: deadline_from(timeout_ms),
            max_work_units: max_work_units.or(self.max_work_units),
            row_limit: row_limit.or(self.row_limit),
        }
    }

    pub(crate) fn apply_to(self, graph: &mut KnowledgeGraph) {
        graph.default_timeout_ms = self.timeout_ms;
        graph.default_max_work_units = self.max_work_units;
        graph.default_row_limit = self.row_limit;
    }
}

impl KnowledgeGraph {
    pub(crate) fn query_defaults(&self) -> QueryDefaults {
        QueryDefaults {
            timeout_ms: self.default_timeout_ms,
            max_work_units: self.default_max_work_units,
            row_limit: self.default_row_limit,
        }
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
        self.query_defaults().apply_to(&mut derived);
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

    #[test]
    fn literal_zero_and_optional_inheritance_have_distinct_meanings() {
        let policy = QueryDefaults {
            timeout_ms: Some(0),
            max_work_units: Some(1),
            row_limit: Some(2),
        };
        let inherited = policy.resolve(None, None, None);
        assert!(inherited.deadline.is_none());
        assert_eq!(inherited.max_work_units, Some(1));
        assert_eq!(inherited.row_limit, Some(2));
        let explicit = policy.resolve(Some(5), Some(0), Some(0));
        assert!(explicit.deadline.is_some());
        assert_eq!(explicit.max_work_units, Some(0));
        assert_eq!(explicit.row_limit, Some(0));
        assert!(QueryDefaults::default()
            .resolve(None, None, None)
            .deadline
            .is_some());
        assert!(deadline_from(Some(0)).is_none());
    }
}
