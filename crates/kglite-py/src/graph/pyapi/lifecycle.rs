//! Binding ownership epochs and the transition to a detached mutable graph.
use crate::graph::{GraphLifecycle, KnowledgeGraph};
use pyo3::prelude::*;
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc,
};

#[derive(Clone)]
pub(crate) struct SourceAuthority {
    epoch: Arc<AtomicU64>,
    captured: u64,
}

impl SourceAuthority {
    pub(crate) fn ended(&self) -> bool {
        self.epoch.load(Ordering::Acquire) != self.captured
    }
}

impl GraphLifecycle {
    pub(crate) fn epoch(&self) -> u64 {
        self.ownership_epoch.load(Ordering::Acquire)
    }

    pub(crate) fn durable_authority(&self) -> Option<SourceAuthority> {
        self.in_durable_lineage().then(|| SourceAuthority {
            epoch: Arc::clone(&self.ownership_epoch),
            captured: self.epoch(),
        })
    }
}

impl KnowledgeGraph {
    pub(crate) fn detached_view_lifecycle(&self) -> GraphLifecycle {
        GraphLifecycle::detached_from(&self.lifecycle, self.inner.cdc_enabled())
    }

    /// Preparation is allocation-only. The old graph, its WAL and writer lease
    /// remain intact until the replacement is ready; the CDC epoch remains unchanged.
    pub(crate) fn end_persistence_ownership(&mut self, py: Python<'_>) {
        let detached = self.inner.detached_persistence_snapshot();
        py.detach(|| self.inner.end_persistence_authority());
        self.inner = Arc::new(detached);
        self.lifecycle
            .ownership_epoch
            .fetch_add(1, Ordering::AcqRel);
        self.lifecycle.source_path = None;
        self.lifecycle.durable = None;
        self.lifecycle.orphaned_from_durable = false;
        self.lifecycle.writer_lease = None;
    }
}
