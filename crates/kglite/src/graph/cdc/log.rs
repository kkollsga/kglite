//! The in-process change log: a bounded ring of [`CdcEvent`]s plus the
//! `(epoch, seq)` addressing a stateless consumer cursor is built from.

use super::event::{CdcEvent, PendingEvent};
use std::collections::VecDeque;

/// Source of process-unique CDC epochs. Starts at 1 so 0 can serve as a
/// sentinel, and is never reused — the same shape as
/// [`next_graph_id`](crate::graph::dir_graph::next_graph_id), and for the same
/// reason: an identity a consumer holds must not be reissued to different
/// data.
static NEXT_EPOCH: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

fn next_epoch() -> u64 {
    NEXT_EPOCH.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}

/// Default ring capacity, in events, when `enable` is given none.
///
/// Order-64k is the plan's settled magnitude: large enough that a consumer
/// polling on a human timescale does not miss events on a busy graph, small
/// enough to stay a bounded, opt-in cost — an event is roughly the size of the
/// entity it describes, so a wide-row workload should expect tens of MB at
/// this capacity and lower it if that is not wanted.
pub const DEFAULT_CAPACITY: usize = 65_536;

/// Ceiling on a caller-supplied capacity. The knob reaches users through a
/// Cypher procedure argument, so it needs a bound that is not "whatever the
/// allocator will give you".
pub const MAX_CAPACITY: usize = 10_000_000;

/// How much of an entity's state each event carries.
///
/// The knob exists because before-images are not free: capturing one means
/// reading the whole entity at first touch in a commit, and keeping it in the
/// ring alongside the after-image. A consumer that only mirrors current state
/// should not pay for a `before` it will not read, so capture is off unless
/// asked for — the same posture the stream itself takes.
///
/// Neo4j's equivalent is the `txLogEnrichment` database option, whose third
/// value (`DIFF`) KGLite deliberately does not offer: a diff still requires
/// the full before-image to compute, so it would save ring bytes at the cost
/// of a second event semantics for consumers to handle. See
/// `db.cdc.enable`'s refusal prose.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CdcEnrichment {
    /// After-image only: `state.before` is always null.
    #[default]
    Off,
    /// Before *and* after images for every change the mode can capture.
    ///
    /// The mode is recorded and reported as soon as it is selected; the
    /// before-image capture it turns on is the next piece of this work, so
    /// `state.before` reads null under `Full` as well until that lands.
    Full,
}

impl CdcEnrichment {
    /// Stable lowercase wire name (`"off"` / `"full"`), as the Cypher surface
    /// spells it in both directions.
    pub fn as_str(&self) -> &'static str {
        match self {
            CdcEnrichment::Off => "off",
            CdcEnrichment::Full => "full",
        }
    }
}

/// A snapshot of the log's addressing state — what `db.cdc.current()` /
/// `db.cdc.earliest()` report and what a status surface prints.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CdcStatus {
    /// Process-unique identity of this log. A cursor carrying a different
    /// epoch addresses a log that no longer exists (or never did) and must be
    /// refused rather than silently reinterpreted.
    pub epoch: u64,
    /// Configured ring capacity in events.
    pub capacity: usize,
    /// How much state each event carries — see [`CdcEnrichment`].
    pub enrichment: CdcEnrichment,
    /// Events currently retained.
    pub buffered: usize,
    /// Sequence number of the oldest retained event; equals `current + 1`
    /// when nothing is retained ("nothing to read, start here").
    pub earliest: u64,
    /// Sequence number of the newest published event; 0 before the first.
    pub current: u64,
}

/// Bounded ring of published change events.
///
/// **Reads are public, writes are not.** A binding reaches this through
/// [`DirGraph::cdc_log`](crate::graph::dir_graph::DirGraph::cdc_log) and may
/// ask it anything — epoch, watermarks, retained events — but cannot append,
/// resize or construct one. Events enter only through
/// [`publish_drained`](super::publish_drained), from a drained capture buffer,
/// which is what makes "every event describes a committed change" a property
/// of the type rather than a convention callers are asked to respect.
///
/// **Not persisted, by construction.** The log lives behind a `#[serde(skip)]`
/// field (see [`DirGraph::cdc`](crate::graph::dir_graph::DirGraph)), so a
/// `.kgl` save writes none of it and a load starts a new epoch. That is a
/// design decision, not an omission: persisting it would grow the file without
/// bound and would hand a cursor from one process to a copy of the data in
/// another, where the same `seq` means something else.
#[derive(Debug)]
pub struct CdcLog {
    epoch: u64,
    /// Sequence number the next published event will carry. Starts at 1, so a
    /// cursor of 0 addresses "before everything".
    next_seq: u64,
    capacity: usize,
    enrichment: CdcEnrichment,
    events: VecDeque<CdcEvent>,
}

impl CdcLog {
    /// A fresh, empty log with a new process-unique epoch.
    pub(crate) fn new(capacity: usize, enrichment: CdcEnrichment) -> Self {
        Self {
            epoch: next_epoch(),
            next_seq: 1,
            capacity: capacity.max(1),
            enrichment,
            events: VecDeque::new(),
        }
    }

    /// This log's process-unique epoch.
    pub fn epoch(&self) -> u64 {
        self.epoch
    }

    /// Configured capacity in events.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// How much state this log captures per change.
    pub fn enrichment(&self) -> CdcEnrichment {
        self.enrichment
    }

    /// Sequence number of the newest published event, or 0 before the first.
    pub fn current(&self) -> u64 {
        self.next_seq - 1
    }

    /// Sequence number of the oldest *retained* event — the earliest position
    /// a cursor can still read from. Advances as eviction discards events, and
    /// equals `current() + 1` while nothing is retained.
    pub fn earliest(&self) -> u64 {
        self.events.front().map_or(self.next_seq, |event| event.seq)
    }

    /// Events currently retained.
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// Whether the ring is empty.
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    /// Addressing snapshot.
    pub fn status(&self) -> CdcStatus {
        CdcStatus {
            epoch: self.epoch,
            capacity: self.capacity,
            enrichment: self.enrichment,
            buffered: self.events.len(),
            earliest: self.earliest(),
            current: self.current(),
        }
    }

    /// Publish one commit's events, assigning sequence numbers and evicting
    /// the oldest to stay within capacity.
    pub(crate) fn append(&mut self, pending: Vec<PendingEvent>) {
        for event in pending {
            self.events.push_back(CdcEvent {
                seq: self.next_seq,
                kind: event.kind,
                change: event.change,
            });
            self.next_seq += 1;
            // `while`, not `if`: a capacity lowered by a re-`enable` can leave
            // the ring over its bound with nothing appended yet.
            while self.events.len() > self.capacity {
                self.events.pop_front();
            }
        }
    }

    /// Reconfigure the ring in place, keeping the epoch and the newest events.
    ///
    /// Keeping the epoch is what makes a re-`enable` non-destructive to
    /// consumers: their cursors stay valid, and a shrink is reported by
    /// [`earliest`](Self::earliest) advancing, exactly as ordinary eviction is.
    /// An enrichment change keeps it too — the events already in the ring are
    /// still the same events, addressed the same way; only what the *next*
    /// capture records changes.
    pub(crate) fn reconfigure(&mut self, capacity: usize, enrichment: CdcEnrichment) {
        self.capacity = capacity.max(1);
        self.enrichment = enrichment;
        while self.events.len() > self.capacity {
            self.events.pop_front();
        }
    }

    /// Events with `seq > from`, oldest first, capped at `limit` when given.
    ///
    /// The cursor is *exclusive* — a consumer passes back the `seq` it last
    /// saw — so `from = 0` reads everything retained.
    pub fn since(&self, from: u64, limit: Option<usize>) -> Vec<&CdcEvent> {
        let iter = self.events.iter().filter(move |event| event.seq > from);
        match limit {
            Some(limit) => iter.take(limit).collect(),
            None => iter.collect(),
        }
    }
}
