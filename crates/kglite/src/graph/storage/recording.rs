//! Write-capture backend — [`RecordingGraph`].
//!
//! Introduced in Phase 6 of the 0.8.0 storage refactor as a read-logging
//! validation wrapper; **repurposed in the Stage 1 durability work** into
//! the production write-capture seam for the WAL. `RecordingGraph<G>`
//! wraps any `G: GraphRead`/`GraphWrite`, forwards every call, and — on
//! the six mutation methods only — buffers a [`RawOp`] describing the
//! change. Reads forward with **zero overhead** (no logging), so a
//! durable graph pays the wrapper cost only on writes.
//!
//! It drives the [`crate::graph::schema::GraphBackend::Recording`] enum
//! variant: a durable graph's backend is
//! `Recording(Box<RecordingGraph<GraphBackend>>)`, so every `GraphWrite`
//! call from the Cypher executor, the fluent/batch mutation paths, and
//! bulk load funnels through this one seam — validated against the call
//! graph (no path mutates the inner `StableDiGraph` around the trait).
//!
//! ## Why raw ops, resolved later
//!
//! The backend stores **interned** node-type / property keys; resolving
//! them to strings needs the `DirGraph`'s `StringInterner`, which the
//! backend does not own. So writes buffer *raw* ops keyed by
//! `NodeIndex`/`EdgeIndex` + `InternedKey`, and [`resolve_ops`] — run at
//! flush, where the interner is in scope — turns them into the
//! string-keyed, identity-keyed [`crate::graph::wal::MutationOp`]s the
//! WAL persists. Upserts are captured as a placeholder index and
//! resolved against the *final* post-batch node/edge state (so an
//! add-then-SET collapses to one upsert; an add-then-remove drops the
//! upsert and keeps the remove). Removes capture their logical
//! `(type, id)` *before* the entry vanishes.
//!
//! ## `Send + Sync` without a `Mutex`
//!
//! All six mutation methods take `&mut self`, and reads no longer record,
//! so the op buffer is a plain `Vec<RawOp>` mutated only through `&mut`.
//! That keeps `RecordingGraph` `Send + Sync` (required for the PyO3
//! `KnowledgeGraph` class) with no lock on the hot path.

use crate::datatypes::Value;
use crate::graph::schema::{EdgeData, InternedKey, NodeData, StringInterner};
use crate::graph::storage::{GraphRead, GraphWrite};
use crate::graph::wal::MutationOp;
use petgraph::graph::{EdgeIndex, NodeIndex};
use petgraph::Direction;
use std::collections::HashMap;
use std::time::Instant;

/// Which `GraphWrite` method produced an upsert: the entity came into
/// existence (`add_node` / `add_edge`) or an existing one was mutated.
///
/// The WAL does not care — [`MutationOp::UpsertNode`] is add-or-replace either
/// way, which is what makes replay idempotent — so this is **in-memory capture
/// state only** and never reaches a [`WalFrame`](crate::graph::wal::WalFrame).
/// Change data capture *does* care: "created" and "updated" are different
/// events to a consumer, and the method boundary is the only place that knows
/// which one happened. Keeping the marker here rather than in `MutationOp`
/// is what lets CDC distinguish them without moving `WAL_FORMAT_VERSION`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CaptureOrigin {
    /// `add_node` / `add_edge` — the entity did not exist before this op.
    Create,
    /// A property/title/label write on an entity that already existed.
    Update,
}

/// One entity's state as a commit found it — the raw half of change data
/// capture's `before` image.
///
/// **Interner-keyed, like every other `RawOp` payload**, because the backend
/// this is captured in does not own the `StringInterner`; resolution happens
/// where `RemoveNode`'s type key is already resolved, at drain time.
///
/// Captured **at first touch** in a batch and never overwritten, so the image
/// is the entity's state at the *start of the commit* rather than before the
/// most recent write. That is the correct answer for a multi-statement
/// transaction, whose consumer wants "what did this transaction change",
/// not "what did its last statement change".
#[derive(Debug, Clone, PartialEq)]
pub struct BeforeImage {
    /// The entity's title. `Value::Null` for an edge, which has none.
    pub title: Value,
    pub properties: Vec<(InternedKey, Value)>,
    /// Secondary labels, when the capturing site could see them.
    ///
    /// `None` is not "no labels" — it is **"not captured here"**, and the two
    /// must not be confused. Labels live in `DirGraph::secondary_label_index`,
    /// one layer above this backend, so a property write captured inside the
    /// wrapper cannot read them. The label choke point fills this in when a
    /// label edit happens in the same commit; a `None` that survives to drain
    /// time means no label edit occurred, so the *final* label set is also the
    /// commit-start one and the resolver may use it.
    pub labels: Option<Vec<String>>,
}

/// Identity of an entity within one capture batch, for first-touch dedup.
///
/// The storage address, matching `cdc::event`'s collapse key and for the same
/// reason: it is stable for the batch's lifetime and cheap to hash, while
/// logical identity is `PartialEq`-only by design (it carries floats).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum BeforeSlot {
    Node(u32),
    Edge(u32),
}

/// A buffered, unresolved mutation. Keyed by petgraph index (for
/// upserts, resolved against the final graph state at flush) or by the
/// pre-removal logical identity (for removes, since the entry is gone by
/// flush time). Turned into a [`MutationOp`] by [`resolve_ops`].
///
/// ## Why the before-image rides inside the op
///
/// It could have been a side table drained alongside the buffer. Inlining it
/// keeps `take_ops`/`truncate_ops`/`resolve_ops` — and therefore all six drain
/// sites — working on exactly one sequence: rollback truncates images with the
/// ops that carried them, and a drain cannot hand out ops whose images it left
/// behind. The `Box` keeps the enum small for the overwhelmingly common case
/// (capture off, or a create), where the field is one null pointer.
#[derive(Debug, Clone, PartialEq)]
pub enum RawOp {
    /// A node was added or property-mutated. Resolve its full final
    /// state at flush; drop if the node was later removed in the batch.
    ///
    /// The before-image is `Some` only on the **first** op that touched this
    /// node in the batch, and only under `capture_before`.
    UpsertNode(NodeIndex, CaptureOrigin, Option<Box<BeforeImage>>),
    /// A node was removed. Its logical identity, captured before removal,
    /// with its full state when before-images are on.
    RemoveNode {
        node_type: InternedKey,
        id: Value,
        before: Option<Box<BeforeImage>>,
    },
    /// An edge was added or property-mutated. Resolve at flush.
    UpsertEdge(EdgeIndex, CaptureOrigin, Option<Box<BeforeImage>>),
    /// An edge was removed. Logical identity captured before removal.
    RemoveEdge {
        conn_type: InternedKey,
        src_type: InternedKey,
        src_id: Value,
        tgt_type: InternedKey,
        tgt_id: Value,
        before: Option<Box<BeforeImage>>,
    },
    /// A node's secondary-label set changed. Like the upserts this carries
    /// only the index and is resolved against final state, so several
    /// `SET n:A SET n:B` in one batch collapse to one op holding both.
    /// Dropped at resolve time if the node was later removed.
    SetNodeLabels(NodeIndex, Option<Box<BeforeImage>>),
}

/// Where an entity's first-touch image lives in the op buffer.
///
/// Two indices, because a delete of an entity the commit already wrote to
/// puts a **copy** of the image on its own op while leaving the original in
/// place — see [`RecordingGraph::copy_node_image`]. `first_at` is the original
/// and never moves; `latest_at` is the newest op carrying a copy, which is the
/// one whose event will actually be published for a delete.
///
/// Keeping both is what makes rollback recoverable: dropping the copy restores
/// `latest_at` to `first_at` rather than losing the slot's image entirely.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ImageSite {
    first_at: usize,
    latest_at: usize,
}

impl ImageSite {
    #[inline]
    fn at(at: usize) -> Self {
        Self {
            first_at: at,
            latest_at: at,
        }
    }

    /// Every op holding a copy of this slot's image, without duplicates.
    #[inline]
    fn carriers(&self) -> impl Iterator<Item = usize> {
        let second = (self.latest_at != self.first_at).then_some(self.latest_at);
        std::iter::once(self.first_at).chain(second)
    }
}

/// The before-image slot of an op, for the ops that carry one.
#[inline]
fn op_image(op: &RawOp) -> Option<&BeforeImage> {
    match op {
        RawOp::UpsertNode(_, _, before)
        | RawOp::UpsertEdge(_, _, before)
        | RawOp::SetNodeLabels(_, before)
        | RawOp::RemoveNode { before, .. }
        | RawOp::RemoveEdge { before, .. } => before.as_deref(),
    }
}

#[inline]
fn op_image_mut(op: &mut RawOp) -> Option<&mut BeforeImage> {
    match op {
        RawOp::UpsertNode(_, _, before)
        | RawOp::UpsertEdge(_, _, before)
        | RawOp::SetNodeLabels(_, before)
        | RawOp::RemoveNode { before, .. }
        | RawOp::RemoveEdge { before, .. } => before.as_deref_mut(),
    }
}

/// A node index as a dedup slot. `u32` because petgraph indices are `u32`
/// underneath and the map is per-batch — halving its key width matters more
/// than the cast does.
#[inline]
fn node_slot(idx: NodeIndex) -> u32 {
    idx.index() as u32
}

#[inline]
fn edge_slot(idx: EdgeIndex) -> u32 {
    idx.index() as u32
}

/// Wrapper that captures write invocations on `G` as [`RawOp`]s while
/// forwarding every `GraphRead`/`GraphWrite` method to it. See the
/// module docs.
#[derive(Debug, Default)]
pub struct RecordingGraph<G: GraphRead> {
    inner: G,
    ops: Vec<RawOp>,
    /// Whether a **write-ahead log owner** installed this wrapper.
    ///
    /// The wrapper has two consumers now: durability (which drains the buffer
    /// into WAL frames) and change data capture (which derives events from the
    /// same buffer). Only the first claims ownership, and three decisions read
    /// that claim rather than the mere presence of the wrapper:
    ///
    /// 1. [`crate::graph::durability::open_log`] refuses a second durable
    ///    owner — the refusal must fire for another *log* owner, not for a CDC
    ///    consumer, or enabling CDC would lock the graph out of durability.
    /// 2. The Cypher create path refuses a duplicate `(type, id)` on a durable
    ///    graph, because the log cannot represent two nodes under one identity.
    ///    Enabling CDC must not silently impose that refusal on graphs that
    ///    keep no log.
    /// 3. The commit-boundary drain hands the ops to the log owner when there
    ///    is one, and discards them after publishing when there is not — which
    ///    is what keeps the buffer bounded on a CDC-only graph.
    ///
    /// Preserved by `Clone` (a fork of a durable graph is still under durable
    /// ownership) and never serialized.
    wal_owner: bool,
    /// Whether writes capture a before-image.
    ///
    /// Off by default and set only by `cdc::enable` under
    /// `CdcEnrichment::Full`, so a durable-only graph — the common wrapped
    /// case — pays one predictable bool test per write and never a read.
    capture_before: bool,
    /// First touch per entity in this batch: slot -> index in `ops` of the op
    /// that carries the entity's image.
    ///
    /// Both halves matter. It makes the image the **commit-start** state
    /// (later writes to the same entity find an entry and capture nothing),
    /// and it caps the read cost at one whole-entity read per changed entity
    /// per commit rather than one per write.
    before_touched: HashMap<BeforeSlot, ImageSite>,
    /// An image a side-channel choke point read *before* its write, waiting
    /// for that write's op to claim it.
    ///
    /// Offered rather than pushed, for two reasons. A choke point that turns
    /// out to write nothing (a `REMOVE n:Label` for a label the node lacks)
    /// must leave no trace — pushing an op there would invent a change and
    /// break the no-phantom invariant. And the write's own op is already
    /// coming, so pushing a second one would double every label edit in the
    /// write-ahead log.
    pending_before: Option<(BeforeSlot, Box<BeforeImage>)>,
    /// Sites dropped by [`RecordingGraph::forget_slot`], with the buffer
    /// length at the moment they were dropped, so a rollback can put them
    /// back.
    ///
    /// Needed because statement rollback restores a deleted node by *adding*
    /// it again (`dir_graph::rollback::undo_node_removed` →
    /// `GraphWrite::add_node`), and that runs **before** `truncate_ops`. The
    /// add looks exactly like a create landing on a reused index, so the
    /// forget fires and the entity's commit-start image becomes unreachable —
    /// permanently, because truncation cannot put back what an earlier step
    /// removed. Empty whenever nothing was forgotten, which is almost always.
    forgotten_sites: Vec<(usize, BeforeSlot, ImageSite)>,
}

impl<G: GraphRead> RecordingGraph<G> {
    /// Wrap `inner` in a fresh-buffer `RecordingGraph` that no write-ahead log
    /// owns. The durable path calls [`claim_wal_ownership`](Self::claim_wal_ownership)
    /// on top; see the [`wal_owner`](Self::is_wal_owner) contract.
    #[inline]
    pub fn new(inner: G) -> Self {
        Self {
            inner,
            ops: Vec::new(),
            wal_owner: false,
            capture_before: false,
            before_touched: HashMap::new(),
            pending_before: None,
            forgotten_sites: Vec::new(),
        }
    }

    /// Mark this wrapper as owned by a write-ahead log. Idempotent, and never
    /// released: ownership ends with the graph.
    #[inline]
    pub(crate) fn claim_wal_ownership(&mut self) {
        self.wal_owner = true;
    }

    /// Whether a write-ahead log owns this wrapper's buffer. See the field
    /// docs for the three decisions that read it.
    #[inline]
    pub fn is_wal_owner(&self) -> bool {
        self.wal_owner
    }

    /// Borrow the wrapped backend.
    #[inline]
    pub fn inner(&self) -> &G {
        &self.inner
    }

    /// Mutable borrow of the wrapped backend (for mode-switch / teardown
    /// paths that need the raw inner without recording).
    #[inline]
    pub fn inner_mut(&mut self) -> &mut G {
        &mut self.inner
    }

    /// Whether this wrapper captures before-images.
    #[inline]
    pub fn captures_before(&self) -> bool {
        self.capture_before
    }

    /// Turn before-image capture on or off.
    ///
    /// Takes effect from the next write; images already buffered are left
    /// alone, which is what a consumer of the current batch expects — the
    /// events it is about to read were captured under the old setting and
    /// re-reading the graph now could not reconstruct them anyway.
    #[inline]
    pub(crate) fn set_capture_before(&mut self, on: bool) {
        self.capture_before = on;
    }

    /// Whether `slot` still needs its first-touch image in this batch.
    ///
    /// Public so the two **side-channel choke points** (the columnar master
    /// write and the label index, both of which mutate outside `GraphWrite`)
    /// can ask before paying for a read they may not need. See
    /// [`Self::note_node_before`].
    #[inline]
    pub fn needs_node_before(&self, idx: NodeIndex) -> bool {
        self.capture_before
            && !self
                .before_touched
                .contains_key(&BeforeSlot::Node(node_slot(idx)))
    }

    /// Record that node `idx` was upserted by a mutation that wrote through
    /// a side channel (the columnar master `ColumnStore`) and so bypassed
    /// the recorded `GraphWrite::node_weight_mut`. Resolved at flush like any
    /// other [`RawOp::UpsertNode`].
    #[inline]
    pub fn note_node_upsert(&mut self, idx: NodeIndex) {
        self.push_node_upsert(idx, CaptureOrigin::Update);
    }

    /// Record an entity's pre-write state from a **side-channel choke point**
    /// — a site that mutates outside the `GraphWrite` seam and therefore has
    /// to hand the image in rather than let this wrapper read it.
    ///
    /// Two such sites exist, and both must call this *before* their write:
    /// the columnar master store (`executor::columnar_write`) and the
    /// secondary-label index (`dir_graph::labels`). Calling it after would
    /// record the post-write state under the name `before`, which is worse
    /// than recording nothing.
    ///
    /// Ignored when the slot already has an image, so the first-touch rule
    /// holds across the mixture of in-wrapper and choke-point captures.
    pub fn note_node_before(&mut self, idx: NodeIndex, image: BeforeImage) {
        if !self.capture_before {
            return;
        }
        let slot = BeforeSlot::Node(node_slot(idx));
        if self.before_touched.contains_key(&slot) {
            return;
        }
        self.pending_before = Some((slot, Box::new(image)));
    }

    /// Fill in the label half of a node's already-captured image.
    ///
    /// The label choke point calls this when the node was first touched by a
    /// *property* write, which could not see labels. The set it passes is
    /// still the commit-start one: this is the commit's first label edit on
    /// the node, or the image would already carry labels.
    pub fn backfill_node_before_labels(&mut self, idx: NodeIndex, labels: Vec<String>) {
        if !self.capture_before {
            return;
        }
        let Some(&site) = self.before_touched.get(&BeforeSlot::Node(node_slot(idx))) else {
            return;
        };
        // Every op holding a copy, not just one: a delete carries its own copy
        // of the image alongside the original, and either may be the one that
        // ends up published.
        for at in site.carriers() {
            let Some(image) = self.ops.get_mut(at).and_then(op_image_mut) else {
                continue;
            };
            if image.labels.is_none() {
                image.labels = Some(labels.clone());
            }
        }
    }

    /// Note that the op about to be pushed at `at` carries `slot`'s image.
    ///
    /// A first touch creates the site; a later *copy* only advances
    /// `latest_at`, leaving `first_at` — and the op it names — untouched, so a
    /// rollback of the copy has somewhere to fall back to.
    #[inline]
    fn record_image_site(&mut self, slot: BeforeSlot, at: usize) {
        self.before_touched
            .entry(slot)
            .and_modify(|site| site.latest_at = at)
            .or_insert_with(|| ImageSite::at(at));
    }

    /// Push an upsert op, capturing the node's first-touch image with it.
    fn push_node_upsert(&mut self, idx: NodeIndex, origin: CaptureOrigin) {
        let before = self.take_node_image(idx);
        if before.is_some() {
            self.record_image_site(BeforeSlot::Node(node_slot(idx)), self.ops.len());
        }
        self.ops.push(RawOp::UpsertNode(idx, origin, before));
    }

    /// Push an edge upsert op, capturing the edge's first-touch image with it.
    fn push_edge_upsert(&mut self, idx: EdgeIndex, origin: CaptureOrigin) {
        let before = self.take_edge_image(idx);
        if before.is_some() {
            self.record_image_site(BeforeSlot::Edge(edge_slot(idx)), self.ops.len());
        }
        self.ops.push(RawOp::UpsertEdge(idx, origin, before));
    }

    /// Copy an already-captured image onto a delete's own op.
    ///
    /// For a delete of an entity this commit *already* wrote to: first-touch
    /// dedup put the image on an earlier op, and that op is normally dropped
    /// at resolve time because the entity no longer exists — so the delete,
    /// the one event whose *only* informative half is `before`, would carry
    /// nothing.
    ///
    /// **Copied, not moved.** Moving it was a defect: a statement that deletes
    /// and then fails is rolled back, the node comes *back*, and the earlier
    /// op is published after all — with the image gone, because `truncate_ops`
    /// can drop the delete's op but cannot put back what that op took. The
    /// image is one entity's payload and is read-only once captured, so
    /// duplicating it is the cheap half of that trade.
    fn copy_node_image(&mut self, idx: NodeIndex) -> Option<Box<BeforeImage>> {
        let site = *self.before_touched.get(&BeforeSlot::Node(node_slot(idx)))?;
        op_image(self.ops.get(site.first_at)?)
            .cloned()
            .map(Box::new)
    }

    fn copy_edge_image(&mut self, idx: EdgeIndex) -> Option<Box<BeforeImage>> {
        let site = *self.before_touched.get(&BeforeSlot::Edge(edge_slot(idx)))?;
        op_image(self.ops.get(site.first_at)?)
            .cloned()
            .map(Box::new)
    }

    /// Forget any image held for `slot`, because the index now addresses a
    /// different entity.
    ///
    /// petgraph reuses a freed index, so a create can land on a slot whose
    /// deleted predecessor still owns a dedup entry. Leaving it would let a
    /// later label backfill write the new entity's labels into the old
    /// entity's image.
    #[inline]
    fn forget_slot(&mut self, slot: BeforeSlot) {
        if !self.capture_before {
            return;
        }
        if let Some(site) = self.before_touched.remove(&slot) {
            // Remembered, not discarded: this add may be a rollback restoring
            // the very entity whose image the site points at.
            self.forgotten_sites.push((self.ops.len(), slot, site));
        }
    }

    /// The node's current state as a before-image, or `None` when capture is
    /// off, the batch already imaged it, or the node does not exist.
    ///
    /// Reads the whole entity, which is the cost the `Full` enrichment mode
    /// buys: once per changed entity per commit, never once per write.
    ///
    /// Measured at **+2-3%** wall time on 1000 autocommit `SET`s (release
    /// profile, min of 7 rounds, two agreeing runs, 2026-08-19). That shape is
    /// the worst case for this read: every write is its own commit and so its
    /// own first touch, so the dedup below never gets to amortise anything.
    fn take_node_image(&mut self, idx: NodeIndex) -> Option<Box<BeforeImage>> {
        if !self.needs_node_before(idx) {
            return None;
        }
        // A choke point that already read the pre-write state wins: by the
        // time this runs its write has landed, so reading here would report
        // the new value under the name `before`. That is the failure mode the
        // offer exists to prevent.
        if let Some(offered) = self.claim_pending(BeforeSlot::Node(node_slot(idx))) {
            return Some(offered);
        }
        let view = self.inner.node_view(idx)?;
        Some(Box::new(BeforeImage {
            title: view.title().into_owned(),
            properties: view.property_pairs(),
            // Not visible from here — `DirGraph`'s label choke point fills it
            // in if the commit touches labels. See the field docs.
            labels: None,
        }))
    }

    /// The edge's current state as a before-image. Edges carry no title and
    /// no labels, so the image is its property set.
    /// Claim an offered image if it is for `slot`.
    ///
    /// An offer for a *different* slot is dropped rather than kept: it belongs
    /// to a write that never reached its op, and holding it would risk
    /// attaching it to some later write of that entity — by which time it
    /// would no longer be a pre-write image.
    fn claim_pending(&mut self, slot: BeforeSlot) -> Option<Box<BeforeImage>> {
        let (offered_slot, image) = self.pending_before.take()?;
        (offered_slot == slot).then_some(image)
    }

    fn take_edge_image(&mut self, idx: EdgeIndex) -> Option<Box<BeforeImage>> {
        if !self.capture_before
            || self
                .before_touched
                .contains_key(&BeforeSlot::Edge(edge_slot(idx)))
        {
            return None;
        }
        let edge = self.inner.edge_weight(idx)?;
        Some(Box::new(BeforeImage {
            title: Value::Null,
            properties: edge.properties.clone(),
            labels: None,
        }))
    }

    /// Record that node `idx`'s secondary labels changed. Labels live in
    /// `DirGraph::secondary_label_index`, above this backend, so no
    /// `GraphWrite` call carries them — the label choke points call this
    /// instead. Resolved at flush like any other index-keyed op.
    #[inline]
    pub fn note_node_labels(&mut self, idx: NodeIndex) {
        let before = self.take_node_image(idx);
        if before.is_some() {
            self.record_image_site(BeforeSlot::Node(node_slot(idx)), self.ops.len());
        }
        self.ops.push(RawOp::SetNodeLabels(idx, before));
    }

    /// Drain the buffered raw ops, leaving the buffer empty. Called at
    /// each commit/flush before [`resolve_ops`].
    #[inline]
    pub fn take_ops(&mut self) -> Vec<RawOp> {
        // The dedup map addresses positions in the buffer being handed out,
        // so it dies with it — the next batch starts its own first touches.
        self.before_touched.clear();
        self.pending_before = None;
        self.forgotten_sites.clear();
        std::mem::take(&mut self.ops)
    }

    /// Number of buffered (undrained) raw ops.
    #[inline]
    pub fn ops_len(&self) -> usize {
        self.ops.len()
    }

    /// Drop buffered ops past `len`.
    ///
    /// Used by statement rollback: a failed mutation's writes are undone in
    /// the graph, so the ops describing them must not survive into the next
    /// WAL flush — [`resolve_ops`] reads *final* state, so a stale upsert op
    /// would resolve against the restored node and persist a mutation that
    /// never committed. Truncating to the pre-statement length is precise: it
    /// discards this statement's ops and keeps any earlier unflushed ones.
    ///
    /// Crate-internal, unlike [`ops_len`](Self::ops_len): reading the capture
    /// depth is safe for anyone, but truncating the buffer is only correct when
    /// paired with the matching graph-state undo, which `GraphBackend`'s
    /// rollback path is the sole site able to guarantee. Keeping it off the
    /// public surface means no binding can drop committed ops on the floor.
    #[inline]
    pub(crate) fn truncate_ops(&mut self, len: usize) {
        self.ops.truncate(len);
        self.pending_before = None;
        // Two different outcomes, and conflating them was a defect:
        //
        // - the **original** capture is gone → the slot has no image left, so
        //   the next write to that entity captures a fresh first-touch one, of
        //   the restored state the surviving ops describe;
        // - only a later **copy** is gone (a rolled-back delete) → the
        //   original survives, so the site falls back to it. Forgetting the
        //   slot here instead would let the next write re-capture, recording
        //   whatever an earlier *surviving* statement had already written as
        //   though it were the commit-start state.
        //
        // One pass, and the map is per-batch, so rollback stays cheap.
        self.before_touched.retain(|_, site| {
            if site.first_at >= len {
                return false;
            }
            if site.latest_at >= len {
                site.latest_at = site.first_at;
            }
            true
        });
        // Undo the forgets this statement made. The rollback's own
        // `add_node` fired them while putting a deleted entity back, so the
        // slot still means what it did before the statement ran — and its
        // image is still on a surviving op.
        while let Some(&(at, slot, site)) = self.forgotten_sites.last() {
            if at < len {
                break;
            }
            self.forgotten_sites.pop();
            if site.first_at < len {
                let recovered = ImageSite {
                    first_at: site.first_at,
                    // The later copy may itself be one of the truncated ops.
                    latest_at: if site.latest_at < len {
                        site.latest_at
                    } else {
                        site.first_at
                    },
                };
                self.before_touched.entry(slot).or_insert(recovered);
            }
        }
    }
}

impl<G: GraphRead + Clone> Clone for RecordingGraph<G> {
    #[inline]
    fn clone(&self) -> Self {
        // A clone starts with an empty op buffer. In the CoW transaction
        // model the buffer is always drained after each mutation, so it
        // is empty at clone time anyway; resetting makes that an
        // invariant the clone can rely on rather than a coincidence.
        Self {
            inner: self.inner.clone(),
            ops: Vec::new(),
            // A fork of a full-capture graph still captures: its commit
            // publishes into the same log through the shared `Arc`.
            capture_before: self.capture_before,
            before_touched: HashMap::new(),
            pending_before: None,
            forgotten_sites: Vec::new(),
            // Ownership follows the data: a transaction fork or a copy-on-write
            // view of a durably-owned graph is still durably owned, and its
            // commit drains into the same log.
            wal_owner: self.wal_owner,
        }
    }
}

/// Wrap `dir`'s backend in the [`RecordingGraph`] write-capture layer so every
/// mutation that crosses the `GraphWrite` seam is buffered for the WAL.
/// Idempotent — an already-wrapped graph is left alone.
///
/// Storage-mode-agnostic by construction: the wrapper wraps the
/// [`GraphBackend`](crate::graph::storage::backend::GraphBackend) *enum* rather
/// than a concrete backend, so one capture path
/// covers memory and mapped alike. (Disk is refused a rung earlier, by the
/// durable-open path, because it has no logical WAL at all.)
///
/// Lifted out of the wheel: `Session::open_durable` and every non-Rust binding
/// that opens a durable graph need exactly this, byte for byte, so it is core
/// rather than per-binding (CLAUDE.md's boundary principle — "anything two or
/// more wrappers would write identically belongs in `kglite::api`").
pub fn wrap_for_durability(dir: &mut crate::graph::dir_graph::DirGraph) {
    dir.graph.wrap_for_durability();
}

/// Resolve buffered [`RawOp`]s into string-keyed, identity-keyed
/// [`MutationOp`]s, reading final node/edge state from `graph` and
/// resolving interned keys through `interner`. Upserts whose node/edge
/// no longer exists (removed later in the same batch) are dropped — the
/// corresponding remove op already captures the final state.
///
/// `secondary_labels` yields a node's current secondary labels by name,
/// ordered as `DirGraph::node_labels` orders them. It is a callback rather
/// than a field read because labels are not backend state at all: they live
/// in `DirGraph::secondary_label_index`, one layer above the `graph` this
/// function is given. Callers pass `DirGraph::secondary_label_names`.
pub fn resolve_ops(
    raw: &[RawOp],
    graph: &impl GraphRead,
    interner: &StringInterner,
    secondary_labels: impl Fn(NodeIndex) -> Vec<String>,
) -> Vec<MutationOp> {
    let mut out = Vec::with_capacity(raw.len());
    for op in raw {
        match op {
            RawOp::UpsertNode(idx, _, _) => {
                if let Some(nd) = graph.node_view(*idx) {
                    out.push(MutationOp::UpsertNode {
                        node_type: nd.node_type_str(interner).to_string(),
                        id: nd.id().into_owned(),
                        title: nd.title().into_owned(),
                        properties: nd.properties_cloned(interner).into_iter().collect(),
                    });
                }
            }
            RawOp::RemoveNode { node_type, id, .. } => {
                out.push(MutationOp::RemoveNode {
                    node_type: interner.resolve(*node_type).to_string(),
                    id: id.clone(),
                });
            }
            RawOp::UpsertEdge(eidx, _, _) => {
                if let (Some((a, b)), Some(ed)) =
                    (graph.edge_endpoints(*eidx), graph.edge_weight(*eidx))
                {
                    if let (Some(src), Some(tgt)) = (
                        logical_node(graph, a, interner),
                        logical_node(graph, b, interner),
                    ) {
                        out.push(MutationOp::UpsertEdge {
                            conn_type: ed.connection_type_str(interner).to_string(),
                            src_type: src.0,
                            src_id: src.1,
                            tgt_type: tgt.0,
                            tgt_id: tgt.1,
                            properties: ed.properties_cloned(interner).into_iter().collect(),
                        });
                    }
                }
            }
            RawOp::RemoveEdge {
                conn_type,
                src_type,
                src_id,
                tgt_type,
                tgt_id,
                ..
            } => {
                out.push(MutationOp::RemoveEdge {
                    conn_type: interner.resolve(*conn_type).to_string(),
                    src_type: interner.resolve(*src_type).to_string(),
                    src_id: src_id.clone(),
                    tgt_type: interner.resolve(*tgt_type).to_string(),
                    tgt_id: tgt_id.clone(),
                });
            }
            RawOp::SetNodeLabels(idx, _) => {
                // Resolved against final state, so repeated `SET n:A`/`SET
                // n:B` in one batch collapse into a single whole-set op. A
                // node removed later in the batch yields `None` and is
                // dropped — its `RemoveNode` already carries the outcome.
                if let Some((node_type, id)) = logical_node(graph, *idx, interner) {
                    out.push(MutationOp::SetNodeLabels {
                        node_type,
                        id,
                        labels: secondary_labels(*idx),
                    });
                }
            }
        }
    }
    out
}

/// Resolve a node index to its logical `(node_type, id)`, or `None` if
/// the node is gone.
fn logical_node(
    graph: &impl GraphRead,
    idx: NodeIndex,
    interner: &StringInterner,
) -> Option<(String, Value)> {
    let nd = graph.node_view(idx)?;
    Some((nd.node_type_str(interner).to_string(), nd.id().into_owned()))
}

// `Serialize` forwards to the inner backend verbatim — the op buffer is
// transient capture state and intentionally does not persist. For
// `RecordingGraph<GraphBackend>` wrapping a `Disk` variant this lands
// on the existing Disk-serialization error path, which is the correct
// behaviour.
impl<G: GraphRead + serde::Serialize> serde::Serialize for RecordingGraph<G> {
    fn serialize<S: serde::Serializer>(&self, ser: S) -> Result<S::Ok, S::Error> {
        self.inner.serialize(ser)
    }
}

impl<'de, G> serde::Deserialize<'de> for RecordingGraph<G>
where
    G: GraphRead + serde::Deserialize<'de>,
{
    fn deserialize<D: serde::Deserializer<'de>>(de: D) -> Result<Self, D::Error> {
        G::deserialize(de).map(Self::new)
    }
}

// ─────────────────────────────────────────────────────────────────────
// GraphRead — log every call, forward to `self.inner`.
// ─────────────────────────────────────────────────────────────────────

impl<G: GraphRead> GraphRead for RecordingGraph<G> {
    type NodeIndicesIter<'a>
        = G::NodeIndicesIter<'a>
    where
        Self: 'a;
    type EdgeIndicesIter<'a>
        = G::EdgeIndicesIter<'a>
    where
        Self: 'a;
    type EdgesIter<'a>
        = G::EdgesIter<'a>
    where
        Self: 'a;
    type EdgeReferencesIter<'a>
        = G::EdgeReferencesIter<'a>
    where
        Self: 'a;
    type EdgesConnectingIter<'a>
        = G::EdgesConnectingIter<'a>
    where
        Self: 'a;
    type NeighborsIter<'a>
        = G::NeighborsIter<'a>
    where
        Self: 'a;

    #[inline]
    fn column_store(
        &self,
        type_key: crate::graph::schema::InternedKey,
    ) -> Option<&std::sync::Arc<crate::graph::storage::column_store::ColumnStore>> {
        self.inner.column_store(type_key)
    }

    fn column_stores_iter(
        &self,
    ) -> Box<
        dyn Iterator<
                Item = (
                    crate::graph::schema::InternedKey,
                    &std::sync::Arc<crate::graph::storage::column_store::ColumnStore>,
                ),
            > + '_,
    > {
        self.inner.column_stores_iter()
    }

    #[inline]
    fn node_count(&self) -> usize {
        self.inner.node_count()
    }

    #[inline]
    fn edge_count(&self) -> usize {
        self.inner.edge_count()
    }

    #[inline]
    fn node_bound(&self) -> usize {
        self.inner.node_bound()
    }

    #[inline]
    fn edge_bound(&self) -> usize {
        self.inner.edge_bound()
    }

    #[inline]
    fn is_memory(&self) -> bool {
        self.inner.is_memory()
    }

    #[inline]
    fn is_mapped(&self) -> bool {
        self.inner.is_mapped()
    }

    #[inline]
    fn is_disk(&self) -> bool {
        self.inner.is_disk()
    }

    #[inline]
    fn node_type_of(&self, idx: NodeIndex) -> Option<InternedKey> {
        self.inner.node_type_of(idx)
    }

    #[inline]
    fn node_labels_of(&self, idx: NodeIndex) -> Vec<InternedKey> {
        self.inner.node_labels_of(idx)
    }

    #[inline]
    fn node_weight(&self, idx: NodeIndex) -> Option<&NodeData> {
        self.inner.node_weight(idx)
    }

    #[inline]
    fn get_node_property(&self, idx: NodeIndex, key: InternedKey) -> Option<Value> {
        self.inner.get_node_property(idx, key)
    }

    #[inline]
    fn get_node_id(&self, idx: NodeIndex) -> Option<Value> {
        self.inner.get_node_id(idx)
    }

    #[inline]
    fn get_node_title(&self, idx: NodeIndex) -> Option<Value> {
        self.inner.get_node_title(idx)
    }

    #[inline]
    fn str_prop_eq(&self, idx: NodeIndex, key: InternedKey, target: &str) -> Option<bool> {
        self.inner.str_prop_eq(idx, key, target)
    }

    #[inline]
    fn node_indices(&self) -> Self::NodeIndicesIter<'_> {
        self.inner.node_indices()
    }

    #[inline]
    fn edge_indices(&self) -> Self::EdgeIndicesIter<'_> {
        self.inner.edge_indices()
    }

    #[inline]
    fn edge_references(&self) -> Self::EdgeReferencesIter<'_> {
        self.inner.edge_references()
    }

    #[inline]
    fn edge_weights<'a>(&'a self) -> Box<dyn Iterator<Item = &'a EdgeData> + 'a> {
        self.inner.edge_weights()
    }

    #[inline]
    fn edges_directed(&self, idx: NodeIndex, dir: Direction) -> Self::EdgesIter<'_> {
        self.inner.edges_directed(idx, dir)
    }

    #[inline]
    fn edges(&self, idx: NodeIndex) -> Self::EdgesIter<'_> {
        self.inner.edges(idx)
    }

    #[inline]
    fn edges_directed_filtered(
        &self,
        idx: NodeIndex,
        dir: Direction,
        conn_type_filter: Option<InternedKey>,
    ) -> Self::EdgesIter<'_> {
        self.inner
            .edges_directed_filtered(idx, dir, conn_type_filter)
    }

    #[inline]
    fn edges_connecting(&self, a: NodeIndex, b: NodeIndex) -> Self::EdgesConnectingIter<'_> {
        self.inner.edges_connecting(a, b)
    }

    #[inline]
    fn edge_weight(&self, idx: EdgeIndex) -> Option<&EdgeData> {
        self.inner.edge_weight(idx)
    }

    #[inline]
    fn find_edge(&self, a: NodeIndex, b: NodeIndex) -> Option<EdgeIndex> {
        self.inner.find_edge(a, b)
    }

    #[inline]
    fn edge_endpoints(&self, idx: EdgeIndex) -> Option<(NodeIndex, NodeIndex)> {
        self.inner.edge_endpoints(idx)
    }

    #[inline]
    fn edge_endpoint_keys<'a>(
        &'a self,
    ) -> Box<dyn Iterator<Item = (NodeIndex, NodeIndex, InternedKey)> + 'a> {
        self.inner.edge_endpoint_keys()
    }

    #[inline]
    fn neighbors_directed(&self, idx: NodeIndex, dir: Direction) -> Self::NeighborsIter<'_> {
        self.inner.neighbors_directed(idx, dir)
    }

    #[inline]
    fn neighbors_undirected(&self, idx: NodeIndex) -> Self::NeighborsIter<'_> {
        self.inner.neighbors_undirected(idx)
    }

    #[inline]
    fn sources_for_conn_type_bounded(
        &self,
        conn_type: InternedKey,
        max: Option<usize>,
    ) -> Option<Vec<u32>> {
        self.inner.sources_for_conn_type_bounded(conn_type, max)
    }

    #[inline]
    fn lookup_peer_counts(&self, conn_type: InternedKey) -> Option<HashMap<u32, i64>> {
        self.inner.lookup_peer_counts(conn_type)
    }

    #[inline]
    fn lookup_by_property_eq(
        &self,
        node_type: &str,
        property: &str,
        value: &str,
    ) -> Option<Vec<NodeIndex>> {
        self.inner.lookup_by_property_eq(node_type, property, value)
    }

    #[inline]
    fn lookup_by_property_prefix(
        &self,
        node_type: &str,
        property: &str,
        prefix: &str,
        limit: usize,
    ) -> Option<Vec<NodeIndex>> {
        self.inner
            .lookup_by_property_prefix(node_type, property, prefix, limit)
    }

    #[inline]
    fn lookup_by_property_eq_any_type(
        &self,
        property: &str,
        value: &str,
    ) -> Option<Vec<NodeIndex>> {
        self.inner.lookup_by_property_eq_any_type(property, value)
    }

    #[inline]
    fn lookup_by_property_prefix_any_type(
        &self,
        property: &str,
        prefix: &str,
        limit: usize,
    ) -> Option<Vec<NodeIndex>> {
        self.inner
            .lookup_by_property_prefix_any_type(property, prefix, limit)
    }

    #[inline]
    fn count_edges_grouped_by_peer(
        &self,
        conn_type: InternedKey,
        dir: Direction,
        deadline: Option<Instant>,
    ) -> Result<HashMap<u32, i64>, String> {
        self.inner
            .count_edges_grouped_by_peer(conn_type, dir, deadline)
    }

    #[inline]
    fn count_edges_filtered(
        &self,
        node: NodeIndex,
        dir: Direction,
        conn_type: Option<InternedKey>,
        other_node_type: Option<InternedKey>,
        deadline: Option<Instant>,
    ) -> Result<usize, String> {
        self.inner
            .count_edges_filtered(node, dir, conn_type, other_node_type, deadline)
    }

    #[inline]
    fn iter_peers_filtered<'a>(
        &'a self,
        node: NodeIndex,
        dir: Direction,
        conn_type: Option<u64>,
    ) -> Box<dyn Iterator<Item = (NodeIndex, EdgeIndex)> + 'a> {
        self.inner.iter_peers_filtered(node, dir, conn_type)
    }

    #[inline]
    fn reset_arenas(&self) {
        self.inner.reset_arenas();
    }
}

// ─────────────────────────────────────────────────────────────────────
// GraphWrite — forward to the inner backend AND buffer a RawOp. This is
// the WAL capture seam; see the module docs.
// ─────────────────────────────────────────────────────────────────────

impl<G: GraphWrite> GraphWrite for RecordingGraph<G> {
    // ── Column-store ownership: delegated, never recorded ──
    //
    // Installing or replacing a type's store is storage bookkeeping — the
    // logical mutation is the property write that follows, and recording the
    // install as well would replay it twice.
    #[inline]
    fn install_column_store(
        &mut self,
        type_key: crate::graph::schema::InternedKey,
        store: std::sync::Arc<crate::graph::storage::column_store::ColumnStore>,
    ) {
        self.inner.install_column_store(type_key, store);
    }

    #[inline]
    fn column_store_mut(
        &mut self,
        type_key: crate::graph::schema::InternedKey,
    ) -> Option<&mut std::sync::Arc<crate::graph::storage::column_store::ColumnStore>> {
        self.inner.column_store_mut(type_key)
    }

    #[inline]
    fn take_column_store(
        &mut self,
        type_key: crate::graph::schema::InternedKey,
    ) -> Option<std::sync::Arc<crate::graph::storage::column_store::ColumnStore>> {
        self.inner.take_column_store(type_key)
    }

    #[inline]
    fn clear_column_stores(&mut self) {
        self.inner.clear_column_stores();
    }

    // ── Node property writes: recorded, like `node_weight_mut` ──
    //
    // These *are* logical mutations. Recording an `UpsertNode` placeholder and
    // resolving the node's final state at flush matches what `node_weight_mut`
    // does, and is what keeps a columnar `SET` in the WAL now that it no longer
    // passes through `node_weight_mut` at all.
    #[inline]
    fn set_node_property(
        &mut self,
        idx: NodeIndex,
        key: crate::graph::schema::InternedKey,
        value: Value,
    ) {
        if self.inner.node_weight(idx).is_some() {
            // Before the inner call, so the image is genuinely pre-write.
            self.push_node_upsert(idx, CaptureOrigin::Update);
        }
        self.inner.set_node_property(idx, key, value);
    }

    /// A title write is a logical mutation like any other property write, and
    /// on a columnar node it no longer passes through `node_weight_mut` — so
    /// it is recorded here or it never reaches the log.
    #[inline]
    fn set_node_title(&mut self, idx: NodeIndex, value: Value) {
        if self.inner.node_weight(idx).is_some() {
            // Before the inner call, so the image is genuinely pre-write.
            self.push_node_upsert(idx, CaptureOrigin::Update);
        }
        self.inner.set_node_title(idx, value);
    }

    #[inline]
    fn set_node_property_if_absent(
        &mut self,
        idx: NodeIndex,
        key: crate::graph::schema::InternedKey,
        value: Value,
    ) {
        if self.inner.node_weight(idx).is_some() {
            // Before the inner call, so the image is genuinely pre-write.
            self.push_node_upsert(idx, CaptureOrigin::Update);
        }
        self.inner.set_node_property_if_absent(idx, key, value);
    }

    #[inline]
    fn remove_node_property(
        &mut self,
        idx: NodeIndex,
        key: crate::graph::schema::InternedKey,
    ) -> Option<Value> {
        if self.inner.node_weight(idx).is_some() {
            // Before the inner call, so the image is genuinely pre-write.
            self.push_node_upsert(idx, CaptureOrigin::Update);
        }
        self.inner.remove_node_property(idx, key)
    }

    #[inline]
    fn clear_node_property(
        &mut self,
        idx: NodeIndex,
        key: crate::graph::schema::InternedKey,
    ) -> Option<Value> {
        if self.inner.node_weight(idx).is_some() {
            // Before the inner call, so the image is genuinely pre-write.
            self.push_node_upsert(idx, CaptureOrigin::Update);
        }
        self.inner.clear_node_property(idx, key)
    }

    #[inline]
    fn replace_node_properties(
        &mut self,
        idx: NodeIndex,
        pairs: Vec<(crate::graph::schema::InternedKey, Value)>,
    ) {
        if self.inner.node_weight(idx).is_some() {
            // Before the inner call, so the image is genuinely pre-write.
            self.push_node_upsert(idx, CaptureOrigin::Update);
        }
        self.inner.replace_node_properties(idx, pairs);
    }

    #[inline]
    fn node_weight_mut(&mut self, idx: NodeIndex) -> Option<&mut NodeData> {
        // The caller mutates the returned &mut after this returns, so we
        // can't see the change here — record a placeholder and resolve the
        // node's final state at flush. The existence check uses an
        // immutable read that ends before the push (the returned mut borrow
        // would otherwise hold all of `self`). Only record when the node
        // exists (a None borrow changes nothing).
        if self.inner.node_weight(idx).is_some() {
            // Before the inner call, so the image is genuinely pre-write.
            self.push_node_upsert(idx, CaptureOrigin::Update);
        }
        self.inner.node_weight_mut(idx)
    }

    #[inline]
    fn node_weight_mut_silent(&mut self, idx: NodeIndex) -> Option<&mut NodeData> {
        // Bypass recording — internal bookkeeping (columnar handle refresh),
        // not a logical mutation. See the trait method docs. Delegated to the
        // inner backend's *silent* method rather than its recorded one so the
        // silence composes: `MemoryGraph` also skips undo-journal capture
        // here, and wrapping it must not put that capture back.
        self.inner.node_weight_mut_silent(idx)
    }

    #[inline]
    fn edge_weight_mut(&mut self, idx: EdgeIndex) -> Option<&mut EdgeData> {
        if self.inner.edge_weight(idx).is_some() {
            self.push_edge_upsert(idx, CaptureOrigin::Update);
        }
        self.inner.edge_weight_mut(idx)
    }

    #[inline]
    fn add_node(&mut self, data: NodeData) -> NodeIndex {
        let idx = self.inner.add_node(data);
        self.forget_slot(BeforeSlot::Node(node_slot(idx)));
        // A create has nothing before it; the `None` is the fact, not a gap.
        self.ops
            .push(RawOp::UpsertNode(idx, CaptureOrigin::Create, None));
        idx
    }

    #[inline]
    fn remove_node(&mut self, idx: NodeIndex) -> Option<NodeData> {
        // Capture the logical identity before the node vanishes.
        let identity = self
            .inner
            .node_type_of(idx)
            .zip(self.inner.get_node_id(idx));
        // A delete is the one change whose *only* informative half is the
        // before-image, so it is read here — the last moment the node exists.
        let before = self
            .take_node_image(idx)
            .or_else(|| self.copy_node_image(idx));
        let removed = self.inner.remove_node(idx);
        if removed.is_some() {
            if let Some((node_type, id)) = identity {
                if before.is_some() {
                    self.record_image_site(BeforeSlot::Node(node_slot(idx)), self.ops.len());
                }
                self.ops.push(RawOp::RemoveNode {
                    node_type,
                    id,
                    before,
                });
            }
        }
        removed
    }

    #[inline]
    fn add_edge(&mut self, a: NodeIndex, b: NodeIndex, data: EdgeData) -> EdgeIndex {
        let eidx = self.inner.add_edge(a, b, data);
        self.forget_slot(BeforeSlot::Edge(edge_slot(eidx)));
        self.ops
            .push(RawOp::UpsertEdge(eidx, CaptureOrigin::Create, None));
        eidx
    }

    #[inline]
    fn remove_edge(&mut self, idx: EdgeIndex) -> Option<EdgeData> {
        // Capture conn type + both endpoints' logical identity before the
        // edge vanishes.
        let identity = self.inner.edge_endpoints(idx).and_then(|(a, b)| {
            let conn_type = self.inner.edge_weight(idx)?.connection_type;
            let (src_type, src_id) = (self.inner.node_type_of(a)?, self.inner.get_node_id(a)?);
            let (tgt_type, tgt_id) = (self.inner.node_type_of(b)?, self.inner.get_node_id(b)?);
            Some((conn_type, src_type, src_id, tgt_type, tgt_id))
        });
        let before = self
            .take_edge_image(idx)
            .or_else(|| self.copy_edge_image(idx));
        let removed = self.inner.remove_edge(idx);
        if removed.is_some() {
            if let Some((conn_type, src_type, src_id, tgt_type, tgt_id)) = identity {
                if before.is_some() {
                    self.record_image_site(BeforeSlot::Edge(edge_slot(idx)), self.ops.len());
                }
                self.ops.push(RawOp::RemoveEdge {
                    conn_type,
                    src_type,
                    src_id,
                    tgt_type,
                    tgt_id,
                    before,
                });
            }
        }
        removed
    }

    #[inline]
    fn update_row_id(&mut self, node_idx: NodeIndex, row_id: u32) {
        self.inner.update_row_id(node_idx, row_id);
    }

    #[inline]
    fn flush_pending_writes(&mut self) {
        self.inner.flush_pending_writes();
    }
}

// ─────────────────────────────────────────────────────────────────────
// In-source parity tests — the Phase 6 "parity matrix run against
// RecordingGraph(MemoryGraph) / RecordingGraph(MappedGraph) /
// RecordingGraph(DiskGraph)" crunch-point.
//
// Exercises the GraphBackend::Recording enum dispatcher end-to-end so
// the new variant is not dead code.
// ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::schema::{EdgeData, GraphBackend, MappedGraph, MemoryGraph, StringInterner};
    use crate::graph::storage::disk::graph::DiskGraph;
    use std::collections::HashMap;
    use tempfile::TempDir;

    // ── fixtures ─────────────────────────────────────────────────────

    fn make_memory_backend(interner: &mut StringInterner) -> GraphBackend {
        let mut g = MemoryGraph::new();
        let a = g.add_node(NodeData::new(
            Value::UniqueId(1),
            Value::String("Alice".to_string()),
            "Person".to_string(),
            {
                let mut p = HashMap::new();
                p.insert("age".to_string(), Value::Int64(30));
                p
            },
            interner,
        ));
        let b = g.add_node(NodeData::new(
            Value::UniqueId(2),
            Value::String("Bob".to_string()),
            "Person".to_string(),
            HashMap::new(),
            interner,
        ));
        g.add_edge(
            a,
            b,
            EdgeData::new("KNOWS".to_string(), HashMap::new(), interner),
        );
        GraphBackend::Memory(std::sync::Arc::new(g))
    }

    fn make_mapped_backend(interner: &mut StringInterner) -> GraphBackend {
        // Mapped backend has identical shape to Memory at this stage;
        // difference is trait-impl identity, which is what we test.
        let mut g = MappedGraph::new();
        let a = g.add_node(NodeData::new(
            Value::UniqueId(1),
            Value::String("Alice".to_string()),
            "Person".to_string(),
            HashMap::new(),
            interner,
        ));
        let b = g.add_node(NodeData::new(
            Value::UniqueId(2),
            Value::String("Bob".to_string()),
            "Person".to_string(),
            HashMap::new(),
            interner,
        ));
        g.add_edge(
            a,
            b,
            EdgeData::new("KNOWS".to_string(), HashMap::new(), interner),
        );
        GraphBackend::Mapped(std::sync::Arc::new(g))
    }

    fn make_disk_backend(dir: &TempDir) -> GraphBackend {
        let dg = DiskGraph::new_at_path(dir.path()).expect("create disk graph");
        GraphBackend::Disk(Box::new(dg))
    }

    // ── helpers ──────────────────────────────────────────────────────

    fn collect_read_surface(g: &impl GraphRead) -> (usize, usize, usize) {
        let nc = g.node_count();
        let ec = g.edge_count();
        let nb = g.node_bound();
        // Iterator methods: exercise them to confirm the GAT associated
        // types line up, then discard.
        let _ = g.node_indices().count();
        let _ = g.edge_indices().count();
        let _ = g.edge_references().count();
        (nc, ec, nb)
    }

    /// `resolve_ops` for the majority of tests, which drive a bare
    /// `RecordingGraph` with no `DirGraph` above it and therefore no label
    /// index. Named rather than inlined so "this graph has no secondary
    /// labels" is an assertion about the fixture, not an anonymous
    /// `|_| vec![]`.
    fn resolve_unlabelled(
        raw: &[RawOp],
        graph: &impl GraphRead,
        interner: &StringInterner,
    ) -> Vec<MutationOp> {
        resolve_ops(raw, graph, interner, |_| Vec::new())
    }

    // ── write capture + resolution ───────────────────────────────────

    #[test]
    fn reads_do_not_capture() {
        let mut interner = StringInterner::new();
        let rg: RecordingGraph<GraphBackend> =
            RecordingGraph::new(make_memory_backend(&mut interner));
        let _ = rg.node_count();
        let _ = rg.edge_count();
        let _ = rg.node_weight(NodeIndex::new(0));
        let _ = rg
            .edges_directed(NodeIndex::new(0), Direction::Outgoing)
            .count();
        assert_eq!(rg.ops_len(), 0, "reads must not buffer any ops");
    }

    #[test]
    fn captures_add_node_and_edge_as_upserts() {
        let mut interner = StringInterner::new();
        let mut rg: RecordingGraph<GraphBackend> = RecordingGraph::new(GraphBackend::new());
        let a = rg.add_node(NodeData::new(
            Value::Int64(1),
            Value::String("Alice".into()),
            "Person".into(),
            HashMap::from([("age".to_string(), Value::Int64(30))]),
            &mut interner,
        ));
        let b = rg.add_node(NodeData::new(
            Value::Int64(2),
            Value::String("Bob".into()),
            "Person".into(),
            HashMap::new(),
            &mut interner,
        ));
        rg.add_edge(
            a,
            b,
            EdgeData::new("KNOWS".into(), HashMap::new(), &mut interner),
        );

        let raw = rg.take_ops();
        assert_eq!(rg.ops_len(), 0, "take_ops empties the buffer");
        let ops = resolve_unlabelled(&raw, &rg, &interner);
        assert_eq!(
            ops,
            vec![
                MutationOp::UpsertNode {
                    node_type: "Person".into(),
                    id: Value::Int64(1),
                    title: Value::String("Alice".into()),
                    properties: vec![("age".into(), Value::Int64(30))],
                },
                MutationOp::UpsertNode {
                    node_type: "Person".into(),
                    id: Value::Int64(2),
                    title: Value::String("Bob".into()),
                    properties: vec![],
                },
                MutationOp::UpsertEdge {
                    conn_type: "KNOWS".into(),
                    src_type: "Person".into(),
                    src_id: Value::Int64(1),
                    tgt_type: "Person".into(),
                    tgt_id: Value::Int64(2),
                    properties: vec![],
                },
            ]
        );
    }

    #[test]
    fn captures_set_as_node_upsert_with_final_state() {
        let mut interner = StringInterner::new();
        let mut rg: RecordingGraph<GraphBackend> = RecordingGraph::new(GraphBackend::new());
        let a = rg.add_node(NodeData::new(
            Value::Int64(1),
            Value::String("Alice".into()),
            "Person".into(),
            HashMap::from([("age".to_string(), Value::Int64(30))]),
            &mut interner,
        ));
        let _ = rg.take_ops(); // drain the add
                               // SET age = 41 via the mutable-borrow path.
        let age_key = interner.get_or_intern("age");
        rg.set_node_property(a, age_key, Value::Int64(41));
        let raw = rg.take_ops();
        let ops = resolve_unlabelled(&raw, &rg, &interner);
        // Resolves to the node's FINAL state (age = 41), not a delta.
        assert_eq!(
            ops,
            vec![MutationOp::UpsertNode {
                node_type: "Person".into(),
                id: Value::Int64(1),
                title: Value::String("Alice".into()),
                properties: vec![("age".into(), Value::Int64(41))],
            }]
        );
    }

    #[test]
    fn captures_remove_node_by_logical_identity() {
        let mut interner = StringInterner::new();
        let backend = make_memory_backend(&mut interner);
        let mut rg: RecordingGraph<GraphBackend> = RecordingGraph::new(backend);
        let removed = rg.remove_node(NodeIndex::new(0));
        assert!(removed.is_some());
        let raw = rg.take_ops();
        let ops = resolve_unlabelled(&raw, &rg, &interner);
        assert_eq!(
            ops,
            vec![MutationOp::RemoveNode {
                node_type: "Person".into(),
                id: Value::UniqueId(1),
            }]
        );
    }

    #[test]
    fn captures_remove_edge_by_logical_identity() {
        let mut interner = StringInterner::new();
        let backend = make_memory_backend(&mut interner);
        let mut rg: RecordingGraph<GraphBackend> = RecordingGraph::new(backend);
        let removed = rg.remove_edge(EdgeIndex::new(0));
        assert!(removed.is_some());
        let raw = rg.take_ops();
        let ops = resolve_unlabelled(&raw, &rg, &interner);
        assert_eq!(
            ops,
            vec![MutationOp::RemoveEdge {
                conn_type: "KNOWS".into(),
                src_type: "Person".into(),
                src_id: Value::UniqueId(1),
                tgt_type: "Person".into(),
                tgt_id: Value::UniqueId(2),
            }]
        );
    }

    #[test]
    fn add_then_remove_in_batch_drops_the_upsert() {
        let mut interner = StringInterner::new();
        let mut rg: RecordingGraph<GraphBackend> = RecordingGraph::new(GraphBackend::new());
        let a = rg.add_node(NodeData::new(
            Value::Int64(7),
            Value::String("Ghost".into()),
            "Person".into(),
            HashMap::new(),
            &mut interner,
        ));
        rg.remove_node(a);
        let raw = rg.take_ops();
        let ops = resolve_unlabelled(&raw, &rg, &interner);
        // The UpsertNode placeholder resolves to None (node gone); only
        // the RemoveNode survives — replay reaches the right final state.
        assert_eq!(
            ops,
            vec![MutationOp::RemoveNode {
                node_type: "Person".into(),
                id: Value::Int64(7),
            }]
        );
    }

    /// A label change produces one whole-set op resolved against final
    /// state, so repeated notes for the same node collapse instead of
    /// logging an add-per-label.
    #[test]
    fn captures_label_changes_as_one_whole_set_op() {
        let mut interner = StringInterner::new();
        let backend = make_memory_backend(&mut interner);
        let mut rg: RecordingGraph<GraphBackend> = RecordingGraph::new(backend);
        let node = NodeIndex::new(0);
        rg.note_node_labels(node);
        rg.note_node_labels(node);
        let raw = rg.take_ops();
        assert_eq!(raw.len(), 2, "each choke-point call buffers a raw op");

        let ops = resolve_ops(&raw, &rg, &interner, |idx| {
            assert_eq!(idx, node);
            vec!["Employee".to_string(), "Manager".to_string()]
        });
        let expected = MutationOp::SetNodeLabels {
            node_type: "Person".into(),
            id: Value::UniqueId(1),
            labels: vec!["Employee".to_string(), "Manager".to_string()],
        };
        assert_eq!(
            ops,
            vec![expected.clone(), expected],
            "both resolve to the same final set — replay is idempotent"
        );
    }

    /// A node labelled and then removed in one batch must not log a label
    /// op naming a node that no longer exists.
    #[test]
    fn label_op_for_a_removed_node_is_dropped() {
        let mut interner = StringInterner::new();
        let mut rg: RecordingGraph<GraphBackend> = RecordingGraph::new(GraphBackend::new());
        let a = rg.add_node(NodeData::new(
            Value::Int64(7),
            Value::String("Ghost".into()),
            "Person".into(),
            HashMap::new(),
            &mut interner,
        ));
        rg.note_node_labels(a);
        rg.remove_node(a);
        let raw = rg.take_ops();
        let ops = resolve_ops(&raw, &rg, &interner, |_| vec!["Employee".to_string()]);
        assert_eq!(
            ops,
            vec![MutationOp::RemoveNode {
                node_type: "Person".into(),
                id: Value::Int64(7),
            }],
            "no SetNodeLabels for a node the batch deleted"
        );
    }

    // ── parity: identity vs unwrapped backend ────────────────────────

    #[test]
    fn recording_trait_parity_memory() {
        let mut a_interner = StringInterner::new();
        let backend_a = make_memory_backend(&mut a_interner);
        let mut b_interner = StringInterner::new();
        let backend_b = make_memory_backend(&mut b_interner);

        let rg: RecordingGraph<GraphBackend> = RecordingGraph::new(backend_b);

        assert_eq!(collect_read_surface(&backend_a), collect_read_surface(&rg));
    }

    #[test]
    fn recording_trait_parity_mapped() {
        let mut a_interner = StringInterner::new();
        let backend_a = make_mapped_backend(&mut a_interner);
        let mut b_interner = StringInterner::new();
        let backend_b = make_mapped_backend(&mut b_interner);

        let rg: RecordingGraph<GraphBackend> = RecordingGraph::new(backend_b);

        assert_eq!(collect_read_surface(&backend_a), collect_read_surface(&rg));
    }

    #[test]
    fn recording_trait_parity_disk() {
        let dir_a = TempDir::new().expect("tempdir");
        let dir_b = TempDir::new().expect("tempdir");
        let backend_a = make_disk_backend(&dir_a);
        let backend_b = make_disk_backend(&dir_b);

        let rg: RecordingGraph<GraphBackend> = RecordingGraph::new(backend_b);

        assert_eq!(collect_read_surface(&backend_a), collect_read_surface(&rg));
    }

    // ── GraphWrite passthrough ────────────────────────────────────────

    #[test]
    fn recording_write_passthrough_memory() {
        let mut interner = StringInterner::new();
        let backend = make_memory_backend(&mut interner);
        let n0 = backend.node_count();
        let e0 = backend.edge_count();

        let mut rg: RecordingGraph<GraphBackend> = RecordingGraph::new(backend);
        let new_node = NodeData::new(
            Value::UniqueId(3),
            Value::String("Carol".to_string()),
            "Person".to_string(),
            HashMap::new(),
            &mut interner,
        );
        let idx = rg.add_node(new_node);
        rg.add_edge(
            NodeIndex::new(0),
            idx,
            EdgeData::new("KNOWS".to_string(), HashMap::new(), &mut interner),
        );

        assert_eq!(rg.node_count(), n0 + 1);
        assert_eq!(rg.edge_count(), e0 + 1);
    }

    // ── is_* predicates forward through the wrapper ──────────────────

    #[test]
    fn recording_is_predicates_forward() {
        let mut interner = StringInterner::new();

        let mem = RecordingGraph::new(make_memory_backend(&mut interner));
        assert!(mem.is_memory());
        assert!(!mem.is_mapped());
        assert!(!mem.is_disk());

        let mut interner2 = StringInterner::new();
        let mapped = RecordingGraph::new(make_mapped_backend(&mut interner2));
        assert!(!mapped.is_memory());
        assert!(mapped.is_mapped());
        assert!(!mapped.is_disk());

        let dir = TempDir::new().expect("tempdir");
        let disk = RecordingGraph::new(make_disk_backend(&dir));
        assert!(!disk.is_memory());
        assert!(!disk.is_mapped());
        assert!(disk.is_disk());
    }

    // ── GraphBackend::Recording variant drives the dispatcher ────────

    #[test]
    fn enum_variant_dispatches_reads_through_recording_layer() {
        let mut interner = StringInterner::new();
        let inner = make_memory_backend(&mut interner);
        let expected_nc = inner.node_count();
        let expected_ec = inner.edge_count();

        let wrapped = GraphBackend::Recording(Box::new(RecordingGraph::new(inner)));

        // Every trait call goes through:
        //   GraphBackend::Recording dispatcher arm
        //   → RecordingGraph<GraphBackend>::node_count (logs + delegates)
        //     → GraphBackend::Memory dispatcher arm
        //       → MemoryGraph::node_count
        assert_eq!(wrapped.node_count(), expected_nc);
        assert_eq!(wrapped.edge_count(), expected_ec);
        assert!(!wrapped.is_disk());
        assert!(wrapped.is_memory());

        let idx0 = NodeIndex::new(0);
        assert!(wrapped.node_weight(idx0).is_some());
        assert_eq!(
            wrapped.edges_directed(idx0, Direction::Outgoing).count(),
            1,
            "KNOWS edge should appear through the recording layer"
        );

        // Reads through the enum dispatcher capture nothing.
        let GraphBackend::Recording(rg) = wrapped else {
            unreachable!()
        };
        assert_eq!(
            rg.ops_len(),
            0,
            "reads through the dispatcher must not capture"
        );
    }

    #[test]
    fn enum_variant_captures_writes_through_dispatcher() {
        let mut interner = StringInterner::new();
        let mut wrapped =
            GraphBackend::Recording(Box::new(RecordingGraph::new(GraphBackend::new())));
        // A write through the enum dispatcher reaches the recording layer.
        wrapped.add_node(NodeData::new(
            Value::Int64(1),
            Value::String("Alice".into()),
            "Person".into(),
            HashMap::new(),
            &mut interner,
        ));
        let GraphBackend::Recording(rg) = &mut wrapped else {
            unreachable!()
        };
        assert_eq!(
            rg.ops_len(),
            1,
            "add_node through the dispatcher is captured"
        );
    }

    #[test]
    fn enum_variant_round_trips_every_backend() {
        // Memory
        let mut i1 = StringInterner::new();
        let wrapped_mem =
            GraphBackend::Recording(Box::new(RecordingGraph::new(make_memory_backend(&mut i1))));
        assert!(wrapped_mem.is_memory());
        assert_eq!(wrapped_mem.node_count(), 2);

        // Mapped
        let mut i2 = StringInterner::new();
        let wrapped_mapped =
            GraphBackend::Recording(Box::new(RecordingGraph::new(make_mapped_backend(&mut i2))));
        assert!(wrapped_mapped.is_mapped());
        assert_eq!(wrapped_mapped.node_count(), 2);

        // Disk
        let dir = TempDir::new().expect("tempdir");
        let wrapped_disk =
            GraphBackend::Recording(Box::new(RecordingGraph::new(make_disk_backend(&dir))));
        assert!(wrapped_disk.is_disk());
        assert_eq!(wrapped_disk.node_count(), 0);
    }

    // ── buffer semantics: take + clone ───────────────────────────────

    #[test]
    fn take_ops_empties_the_buffer() {
        let mut interner = StringInterner::new();
        let mut rg: RecordingGraph<GraphBackend> = RecordingGraph::new(GraphBackend::new());
        rg.add_node(NodeData::new(
            Value::Int64(1),
            Value::String("A".into()),
            "T".into(),
            HashMap::new(),
            &mut interner,
        ));
        assert_eq!(rg.ops_len(), 1);
        let drained = rg.take_ops();
        assert_eq!(drained.len(), 1);
        assert_eq!(rg.ops_len(), 0);
    }

    #[test]
    fn silent_mut_does_not_record_but_note_upsert_does() {
        let mut interner = StringInterner::new();
        let mut rg: RecordingGraph<GraphBackend> =
            RecordingGraph::new(make_memory_backend(&mut interner));
        // The recorded path captures.
        let _ = rg.node_weight_mut(NodeIndex::new(0));
        assert_eq!(rg.ops_len(), 1);
        let _ = rg.take_ops();
        // The silent path (columnar handle-refresh bookkeeping) must NOT.
        let _ = rg.node_weight_mut_silent(NodeIndex::new(0));
        assert_eq!(rg.ops_len(), 0, "node_weight_mut_silent must not capture");
        // Side-channel writes (columnar master) record explicitly.
        rg.note_node_upsert(NodeIndex::new(0));
        assert_eq!(rg.ops_len(), 1);
    }

    #[test]
    fn clone_starts_with_empty_op_buffer() {
        let mut interner = StringInterner::new();
        let mut rg: RecordingGraph<GraphBackend> = RecordingGraph::new(GraphBackend::new());
        rg.add_node(NodeData::new(
            Value::Int64(1),
            Value::String("A".into()),
            "T".into(),
            HashMap::new(),
            &mut interner,
        ));
        let rg2 = rg.clone();
        assert_eq!(rg.ops_len(), 1);
        assert_eq!(rg2.ops_len(), 0, "a clone starts with a fresh op buffer");
    }

    // ── Edge iterator semantics forward correctly ────────────────────

    #[test]
    fn edge_references_forward_through_recording() {
        let mut interner = StringInterner::new();
        let backend = make_memory_backend(&mut interner);
        let rg: RecordingGraph<GraphBackend> = RecordingGraph::new(backend);
        let edges: Vec<_> = rg
            .edge_references()
            .map(|er| (er.source().index(), er.target().index()))
            .collect();
        assert_eq!(edges, vec![(0, 1)]);
    }
}
