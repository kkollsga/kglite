//! Write-ahead log for durable in-memory graphs.
//!
//! A `.kgl-wal` sidecar holds an append-only sequence of **logical**
//! mutation frames. Each committed mutation operation appends one
//! [`WalFrame`] — a batch of [`MutationOp`]s tagged with a log-sequence
//! number (LSN) — and makes it durable to the degree the configured
//! [`DurabilityLevel`] promises. On open, the engine loads the `.kgl`
//! checkpoint snapshot, then replays every WAL frame with
//! `lsn > DirGraph::checkpoint_lsn` to recover work committed since the
//! last checkpoint. A checkpoint (a full `.kgl` save) truncates the WAL
//! and stamps the LSN it consumed up to into the `.kgl`.
//!
//! The LSN is a **counter owned by the log**, not the graph `version`:
//! the writing binding hands out `next_lsn` and increments it (see
//! `KnowledgeGraph::flush_wal`), and it never restarts at a checkpoint
//! (see [`WalFrame::lsn`]). Graph `version` advances on work that is
//! never logged, and is not a log position.
//!
//! This module owns only the **on-disk format**: the op schema, the
//! frame envelope, and crash-safe read/write. Capture (translating
//! `GraphWrite` calls into ops) and replay (applying ops to a
//! `DirGraph`) live in their own modules — kept separate so the format
//! can be tested in isolation.
//!
//! ## Logical, identity-keyed ops
//!
//! Ops are keyed by **stable logical identity**, never by petgraph
//! `NodeIndex`/`EdgeIndex` (which do not survive checkpoint load or
//! compaction). A node is `(node_type, id)`; an edge is
//! `(conn_type, src, tgt)` identifies a parallel-edge group, not one edge.
//! v4 persists its complete ordered property-map sequence, including equal
//! maps, and full node state with explicit incarnation resets. Legacy v2/v3
//! edge upserts/removes retain their original single-edge semantics.
//! Idempotence means replaying a
//! frame twice is harmless — important for crash recovery, where the last
//! frame before a crash may or may not have been applied to the snapshot.
//!
//! ## Crash safety of the format
//!
//! A frame is `[len: u32 LE][crc32: u32 LE][payload: codec(WalFrame)]`,
//! emitted by a **single** `write_all` (see [`append_frame`]).
//! The v2/v3/v4 file headers select Postcard for every frame. Older headers
//! are rejected before any payload or torn-tail handling.
//! A crash mid-append leaves a torn trailing frame; [`read_frames`] stops
//! at the first short read or CRC mismatch and returns every frame up to
//! it. A torn frame is therefore *discarded*, never half-applied — the
//! atomic unit of durability is the whole frame.
//!
//! Torn-tail handling does **not** depend on `fsync`: `fsync` controls
//! *when* bytes reach stable storage, not whether a write is atomic, so a
//! torn frame has always been possible and has always been discarded. That
//! is what lets the barrier be a per-level choice without touching recovery.
//!
//! ## Durability levels
//!
//! [`DurabilityLevel`] names what a committed mutation survives; the WAL
//! itself only cares about the derived [`SyncMode`] — `Full` barriers every
//! frame, `Normal` hands it to the page cache without one, `Off` keeps no log
//! at all. The per-level guarantees are on [`DurabilityLevel`]'s variants.
//!
//! Under `Normal` an OS crash can lose an arbitrary suffix of the log, but
//! never a *hole*: [`read_frames`] stops at the first frame it cannot
//! verify, so recovery always yields a **prefix**. Frames are per-commit
//! and replay is idempotent, so a prefix is a valid earlier state rather
//! than a corrupt one.
//!
//! One invariant this places on the *caller*: a checkpoint must not
//! truncate frames that are still only in the page cache, or replaying the
//! surviving prefix could revert data the checkpoint already holds. Call
//! [`Wal::sync`] before checkpointing — see its docs.

use std::fs::{File, OpenOptions};
use std::io::{self, BufReader, Read, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::datatypes::Value;

#[path = "wal_edge_embeddings.rs"]
mod edge_embeddings;
pub use edge_embeddings::{
    EdgeEmbeddingGroupDigest, EdgeEmbeddingStoreState, EdgeGroupEmbeddingPatchWal,
    EdgeGroupMemberPatchWal, EdgeGroupStoreWalState, EdgeVectorCellPatchWal, EdgeVectorWalState,
};

/// File magic for a kglite WAL sidecar: `KWAL`.
pub const WAL_MAGIC: [u8; 4] = *b"KWAL";

/// On-disk WAL format version *written* by this build. Bumped when the
/// frame payload gains anything an older build could not parse; the WAL is
/// a within-version recovery artefact (truncated at every checkpoint), not
/// a long-term archival format like `.kgl`.
///
/// **v2 → v3** appended [`MutationOp::SetNodeLabels`] to the op enum.
/// Postcard tags enum variants by index, so every v2 op (tags 0–3) encodes
/// byte-identically under v3 — a v2 WAL is a *strict subset* of v3 and is
/// read exactly, without a compat mirror of the old schema (see
/// [`MIN_READABLE_WAL_FORMAT_VERSION`]). The version byte still moves,
/// because the reverse direction is not safe: a v3 WAL handed to a
/// v2-writing build would hit an unknown tag, and that build's recovery
/// treats an unparseable payload as a torn tail — it would *silently
/// discard* committed frames. The header bump converts that silent data
/// loss into the loud "unsupported WAL format version" refusal such a
/// build already implements.
///
/// **v3 → v4** appends full-node and parallel-group state tags 5 and 6.
/// Prior tags and payloads are unchanged. The header is upgraded before a
/// new writer appends, so a v2/v3-only reader refuses rather than dropping
/// unfamiliar committed operations as an undecodable tail.
///
/// **v4 → v5** appends [`MutationOp::SetTypeFieldAliases`] as tag 7. Same
/// shape as every bump before it: tags 0–6 encode byte-identically, so a
/// v2/v3/v4 WAL is a strict subset read exactly, while the header moves
/// because a v4-writing build meeting tag 7 would treat the frame as a torn
/// tail and silently drop committed work.
///
/// **v5 → v6** appends the remaining above-the-backend *declarations* as tags
/// 8–13: [`MutationOp::SetTypeParent`], [`MutationOp::SetOntology`],
/// [`MutationOp::SetSchemaVersion`], [`MutationOp::SetSpatialConfig`],
/// [`MutationOp::SetPropertyIndex`] and [`MutationOp::SetConstraint`]. Tag 7
/// closed the identity-spelling half of that class; these close the rest, so a
/// pre-checkpoint crash no longer silently drops a parent-type map, an
/// ontology, a schema stamp, a spatial declaration, a user index or a
/// constraint. Tags 0–7 are untouched and every older WAL stays a strict
/// subset; the header moves for the same reason as every bump before it.
///
/// **v6 → v7** appends the two remaining above-the-backend classes as tags
/// 14–17: [`MutationOp::SetNodeTimeseries`],
/// [`MutationOp::SetTimeseriesConfig`], [`MutationOp::SetEmbeddings`] and
/// [`MutationOp::SetVectorIndex`]. These are *bulk payloads* rather than
/// declarations — a node's whole date index and channel set, a store's whole
/// vector buffer — but they occupied the same blind spot: a crash before the
/// first checkpoint recovered every row while `timeseries()` and
/// `list_embeddings()` answered as if the load had never happened. Tags 0–13
/// are untouched and every older WAL stays a strict subset; the header moves
/// for the same reason as every bump before it.
///
/// **v7 → v8** appends relationship embedding metadata, full-group fallback,
/// and relative-group patch records as tags 18–20. Tags 0–17 remain unchanged.
///
/// **v8 → v9** appends the relationship HNSW declaration as tag 21. Without
/// the header bump a v8 reader could mistake the unknown trailing tag for a
/// torn tail and silently discard a committed declaration.
pub const WAL_FORMAT_VERSION: u8 = 9;

/// Oldest WAL format this build can replay. Frames from any version in
/// `MIN_READABLE_WAL_FORMAT_VERSION..=WAL_FORMAT_VERSION` decode with the
/// current [`MutationOp`] schema; see [`WAL_FORMAT_VERSION`] for why that
/// is sound rather than a shim. Reading these is deliberate
/// format-lifecycle handling: a WAL that outlived the build that wrote it
/// is exactly the crash-recovery case durability exists for, so an
/// upgraded binary must recover it, not discard it.
pub const MIN_READABLE_WAL_FORMAT_VERSION: u8 = 2;

const MAX_WAL_FRAME_BYTES: u64 = u32::MAX as u64;

#[path = "wal_durability.rs"]
mod durability;
pub use durability::{DurabilityLevel, SyncMode};

/// One logical, identity-keyed mutation. See the module docs for why
/// the state-changing shapes are idempotent upserts.
///
/// **Variant order is on-disk format.** Postcard tags variants by
/// declaration index, so a new op must be *appended* — inserting one
/// renumbers its successors and silently misparses every existing WAL.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MutationOp {
    /// Add-or-replace a node identified by `(node_type, id)` with the
    /// full given title + property set.
    UpsertNode {
        node_type: String,
        id: Value,
        title: Value,
        properties: Vec<(String, Value)>,
    },
    /// Remove the node identified by `(node_type, id)`, if present.
    RemoveNode { node_type: String, id: Value },
    /// Add-or-replace the edge `(conn_type, src, tgt)` with the full
    /// given property set. Endpoints are named by their logical
    /// `(node_type, id)`.
    UpsertEdge {
        conn_type: String,
        src_type: String,
        src_id: Value,
        tgt_type: String,
        tgt_id: Value,
        properties: Vec<(String, Value)>,
    },
    /// Remove the edge `(conn_type, src, tgt)`, if present.
    RemoveEdge {
        conn_type: String,
        src_type: String,
        src_id: Value,
        tgt_type: String,
        tgt_id: Value,
    },
    /// Replace the **secondary** labels of `(node_type, id)` with exactly
    /// `labels` (the primary type is `node_type` and is never listed).
    ///
    /// A node's secondary labels live in `DirGraph::secondary_label_index`,
    /// *above* the storage backend — `NodeData` carries none — so they are
    /// invisible to the `GraphWrite` capture seam that produces
    /// [`MutationOp::UpsertNode`]. Without this op a `:Label` added by
    /// `CREATE (n:A:B)` / `SET n:B` was lost on WAL replay while every
    /// property survived. Labels are therefore captured at their own choke
    /// point ([`crate::graph::dir_graph::DirGraph::add_node_label`] and its
    /// remove sibling) and carried as a whole set, which keeps the op
    /// idempotent like every other: replaying it twice, or over a
    /// checkpoint that already holds some of the labels, converges on the
    /// same state.
    ///
    /// Ordered by label name, matching `DirGraph::node_labels`, so a
    /// recovered graph reports labels in the same order as the graph that
    /// logged them.
    SetNodeLabels {
        node_type: String,
        id: Value,
        labels: Vec<String>,
    },
    /// Complete v4 node state. Reset severs the checkpoint incarnation first.
    ReplaceNodeState {
        node_type: String,
        id: Value,
        title: Value,
        properties: Vec<(String, Value)>,
        labels: Vec<String>,
        reset: bool,
    },
    /// Complete v4 parallel group, including identical property maps.
    /// Empty edges removes the group; nonempty groups require live endpoints.
    ReplaceEdgeGroup {
        conn_type: String,
        src_type: String,
        src_id: Value,
        tgt_type: String,
        tgt_id: Value,
        edges: Vec<Vec<(String, Value)>>,
    },
    /// Declare the column spellings a node type's identity fields answer to:
    /// the `unique_id_field` / `node_title_field` an `add_nodes` call named.
    ///
    /// These live in `DirGraph::id_field_aliases` / `title_field_aliases`,
    /// *above* the storage backend — the same position secondary labels
    /// occupy — so no `GraphWrite` call describes one and the capture seam
    /// that produces [`MutationOp::UpsertNode`] cannot infer it. Without this
    /// op a crash before the first checkpoint recovered every value under the
    /// canonical `id`/`title` while losing the name the caller reads them by:
    /// `n.uid` came back null and `{uid: …}` raised a schema error, and the
    /// recovered app's next `save()` truncated the log and made that
    /// permanent.
    ///
    /// `None` means **leave the existing declaration alone**, never "clear
    /// it". That distinction is the `should_update_title` guard at the
    /// `add_nodes` choke point: a follow-up call with `node_title_field=None`
    /// must not rebind the title spelling to the id column. Both fields
    /// carry the caller's spelling only when it differs from the canonical
    /// name, so an op naming neither is never emitted.
    SetTypeFieldAliases {
        node_type: String,
        id_field: Option<String>,
        title_field: Option<String>,
    },
    /// Declare `node_type` a supporting child of `parent_type`, or withdraw
    /// the declaration when `parent_type` is `None`.
    ///
    /// `DirGraph::parent_types` is presentation ownership — it decides which
    /// types `describe()` hides behind a `<supporting>` section — and lives
    /// above the storage backend, so nothing in the capture seam describes it.
    SetTypeParent {
        node_type: String,
        parent_type: Option<String>,
    },
    /// Replace the declared semantic layer wholesale. An empty store is
    /// `clear_ontology`, which is why this carries no `Option`: "no ontology"
    /// is a value the store can hold, and a whole-store replace is what
    /// `define_ontology` does, so replaying it twice converges.
    ///
    /// The payload is the serialized `OntologyStore`, not the user's
    /// declaration document, so it replays through the load-time install
    /// (assign + `rebuild_ontology_closures`) rather than through
    /// `define_ontology`'s graph-aware checks — see the install site for why
    /// re-running those against recovered rows would refuse a valid log.
    ///
    /// **JSON, not the struct.** Frames are postcard, which is not
    /// self-describing: it deserializes a fixed field sequence, so a struct
    /// whose fields carry `skip_serializing_if` — as every level of the
    /// ontology store does — writes fewer fields than it reads back and the
    /// whole frame decodes as a torn tail. JSON also keeps the WAL's on-disk
    /// shape independent of the store's field list, which is what
    /// [`WAL_FORMAT_VERSION`] would otherwise have to move for.
    SetOntology { document: String },
    /// Stamp the caller's own data-model revision (`set_schema_version`).
    /// The engine never interprets it, so replay is an unconditional
    /// last-writer-wins assignment.
    SetSchemaVersion { version: u32 },
    /// Replace `node_type`'s spatial field declaration — which columns hold
    /// lat/lon pairs and WKT geometries. Whole-config replace, matching
    /// `set_spatial`, which is insert-or-replace per type.
    ///
    /// JSON for the same reason [`MutationOp::SetOntology`] is.
    SetSpatialConfig { node_type: String, config: String },
    /// Declare (`present`) or withdraw a user index on `node_type`.
    ///
    /// `properties` carries one name for an equality or range index and the
    /// declared tuple for a composite one. Only the *declaration* travels:
    /// replay rebuilds the structure from the recovered rows through the same
    /// routed builders the `.kgl` loader uses, so the frame stays small
    /// however large the type is.
    SetPropertyIndex {
        node_type: String,
        properties: Vec<String>,
        kind: PropertyIndexKind,
        present: bool,
    },
    /// Declare (`present`) or withdraw a `CREATE CONSTRAINT` declaration.
    ///
    /// One op carries the whole family — node and relationship, all four
    /// kinds — because `DROP CONSTRAINT` withdraws by name and has to name
    /// exactly what the declaration installed. `declared_type` is set only for
    /// [`ConstraintKind::PropertyType`]; `name` only when the author gave one.
    SetConstraint {
        name: Option<String>,
        entity: crate::graph::constraints::EntityKind,
        kind: crate::graph::constraints::ConstraintKind,
        entity_type: String,
        properties: Vec<String>,
        declared_type: Option<crate::graph::property_types::DeclaredType>,
        present: bool,
    },
    /// The whole timeseries — sorted date index and every channel — of the
    /// node `(node_type, id)`, as it stands after the write.
    ///
    /// `DirGraph::timeseries_store` is keyed by `NodeIndex.index()`, a
    /// *physical* slot that a later delete hands to a different node, so the
    /// op is keyed logically like every other: replay resolves the id against
    /// the recovered rows. Whole-payload rather than a delta because that is
    /// what each writer produces — `set_time_index` replaces the index and
    /// clears the channels, `add_ts_channel` rewrites one channel of a series
    /// it has in hand — and it keeps the op idempotent.
    ///
    /// **This is the WAL's largest frame shape.** A 365-key × 3-channel node
    /// is ~13 KB, so a 10 000-node bulk load logs one ~129 MB frame, which
    /// `append_frame_bounded` assembles into a single `Vec` before its one
    /// `write_all` (deliberately — a one-write frame cannot be torn). Under
    /// the 4 GiB format cap, and ~3.6× cheaper per source row than the node
    /// rows the log already carries, but a real transient memory cost.
    SetNodeTimeseries {
        node_type: String,
        id: Value,
        timeseries: crate::graph::features::timeseries::NodeTimeseries,
    },
    /// Replace `node_type`'s timeseries declaration — resolution, known
    /// channels, units and bin semantics. Insert-or-replace per type, matching
    /// every writer of `DirGraph::timeseries_configs`.
    ///
    /// JSON for the same reason [`MutationOp::SetOntology`] is: three of
    /// `TimeseriesConfig`'s four fields carry `skip_serializing_if`, so under
    /// postcard — which is not self-describing and reads a fixed field
    /// sequence — the struct would write fewer fields than it reads back and
    /// the whole frame would decode as a torn tail.
    SetTimeseriesConfig { node_type: String, config: String },
    /// Vectors for the store `(node_type, "{text_column}_emb")`, with the
    /// provenance that makes them answerable.
    ///
    /// `model_id` and the per-entry text hash ride **in this op**, never
    /// reconstructed: a replay that restored vectors with `model_id: None`
    /// would silently break `embedding_info()` and turn
    /// `embed_texts(mode='changed')` into a full re-embed of the corpus.
    ///
    /// Entries are `(node id, vector, source-text hash)` and are keyed
    /// logically for the same reason [`MutationOp::SetNodeTimeseries`] is —
    /// the store addresses `NodeIndex.index()`, which a delete reuses.
    /// `mode` says how the op relates to what the log already carries for this
    /// store, which is what lets an incremental `add_embeddings` log its own
    /// batch instead of the whole store each time.
    SetEmbeddings {
        node_type: String,
        text_column: String,
        dimension: usize,
        metric: Option<String>,
        model_id: Option<String>,
        entries: Vec<(Value, Vec<f32>, Option<u64>)>,
        mode: EmbeddingWrite,
    },
    /// Declare (`present`) or withdraw the HNSW index over
    /// `(node_type, "{text_column}_emb")`, with the parameters it was built
    /// from.
    ///
    /// Only the *declaration* travels. The index addresses **store slots**,
    /// which replay renumbers, and `io/file/vector_persistence.rs` already
    /// states that it is a rebuildable cache and never a correctness
    /// dependency — so replay rebuilds the topology from the replayed vectors
    /// through `build_vector_index`, exactly as it rebuilds property indexes
    /// from recovered rows.
    SetVectorIndex {
        node_type: String,
        text_column: String,
        metric: Option<String>,
        m: Option<usize>,
        ef_construction: Option<usize>,
        ef_search: Option<usize>,
        auto_refresh_limit: Option<usize>,
        present: bool,
    },
    /// Complete metadata state for one relationship embedding store.
    ///
    /// `Present` includes a declared store with zero vectors; `Absent` drops
    /// it and its index declaration. Every field is complete state, so
    /// `model_id: None` explicitly clears generated provenance.
    SetEdgeEmbeddingStore {
        conn_type: String,
        text_column: String,
        state: EdgeEmbeddingStoreState,
    },
    /// Complete relationship-vector state for one final ordered parallel group.
    ///
    /// The paired [`Self::ReplaceEdgeGroup`] supplies the final property maps.
    /// `stores` supplies one cell per final member for every declared embedding
    /// store of `conn_type`, including explicit `None` cells.
    ReplaceEdgeGroupEmbeddings {
        conn_type: String,
        src_type: String,
        src_id: Value,
        tgt_type: String,
        tgt_id: Value,
        member_count: usize,
        stores: Vec<EdgeGroupStoreWalState>,
    },
    /// Relative relationship-vector state paired with ordered final topology.
    PatchEdgeGroupEmbeddings {
        conn_type: String,
        src_type: String,
        src_id: Value,
        tgt_type: String,
        tgt_id: Value,
        patch: EdgeGroupEmbeddingPatchWal,
    },
    /// Declare or withdraw the rebuildable HNSW index for one relationship
    /// embedding store. The topology is rebuilt from recovered vectors.
    SetEdgeVectorIndex {
        conn_type: String,
        text_column: String,
        metric: Option<String>,
        m: Option<usize>,
        ef_construction: Option<usize>,
        ef_search: Option<usize>,
        auto_refresh_limit: Option<usize>,
        present: bool,
    },
}

/// How a [`MutationOp::SetEmbeddings`] relates to the vectors the log already
/// carries for its store.
///
/// **Variant order is on-disk format**, for the same reason [`MutationOp`]'s
/// is: postcard tags by declaration index. Append only.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EmbeddingWrite {
    /// `set_embeddings` / `embed_texts` / an import: these are the store's
    /// vectors, so anything logged for it earlier is superseded.
    Replace,
    /// `add_embeddings`: this batch joins what the store already holds. Logged
    /// as the batch rather than the whole store, so an n-batch ingest costs
    /// O(n) bytes rather than O(n²).
    Upsert,
    /// `remove_embeddings`: the store is gone, entries empty.
    Withdraw,
}

/// Which user-index structure a [`MutationOp::SetPropertyIndex`] declares.
///
/// **Variant order is on-disk format**, for the same reason [`MutationOp`]'s
/// is: postcard tags by declaration index. Append only.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PropertyIndexKind {
    /// `create_index` — hash equality lookup on one property.
    Equality,
    /// `create_range_index` — ordered lookup on one property.
    Range,
    /// `create_composite_index` — one index over a property tuple.
    Composite,
}

/// One committed mutation operation: the ops it produced, tagged with a
/// log-sequence number.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WalFrame {
    /// Log-sequence number, issued by the writer's own monotonic counter —
    /// **not** the graph `version` (see the module docs for the replay rule).
    ///
    /// The counter must never restart at a checkpoint: a restarted LSN would
    /// be reused by a post-checkpoint frame, making a stale pre-checkpoint
    /// frame indistinguishable from a fresh one.
    pub lsn: u64,
    /// The logical ops this commit produced, in application order.
    pub ops: Vec<MutationOp>,
}

// ─────────────────────────────────────────────────────────────────────
// CRC32 (IEEE 802.3, polynomial 0xEDB88320)
// ─────────────────────────────────────────────────────────────────────

/// CRC32 (IEEE) of `data`.
///
/// The single CRC32 in this crate: the per-frame integrity check here, and
/// the per-section digest over a `.kgl`'s compressed bytes
/// (`graph::io::file::section_digest`). Deterministic across processes and
/// builds (unlike `DefaultHasher`), which the torn-frame check relies on.
///
/// Backed by `crc32fast`, which dispatches to the CPU's CRC instructions
/// (aarch64 `crc32*`, x86 `pclmulqdq`) and falls back to a software table
/// elsewhere. The values are identical to the hand-rolled table this
/// replaced — `crc32_matches_known_vector` pins them — so digests written
/// by any previous build still verify, and digests written here still verify
/// on one. It replaced that table because the software path runs at
/// ~0.5 GB/s: on a 180 MB `.kgl` that is ~360 ms added to every load, which
/// is what 0.16.6 shipped. The accelerated path costs ~14 ms for the same
/// bytes.
pub fn crc32(data: &[u8]) -> u32 {
    let mut hasher = crc32fast::Hasher::new();
    hasher.update(data);
    hasher.finalize()
}

/// Write the WAL file header (magic + format version) to a freshly
/// created/truncated WAL. Call once before any [`append_frame`].
pub fn write_header(w: &mut impl Write) -> io::Result<()> {
    write_header_version(w, WAL_FORMAT_VERSION)
}

fn write_header_version(w: &mut impl Write, version: u8) -> io::Result<()> {
    w.write_all(&WAL_MAGIC)?;
    w.write_all(&[version])?;
    Ok(())
}

/// Append one frame: `[len][crc][payload]`. The caller is responsible
/// for `fsync`/`flush` after the append to make it durable — this fn
/// only writes the bytes (so a batch of frames can share one fsync if
/// the caller wants).
///
/// The prefix and payload are assembled into one buffer and emitted with a
/// **single** `write_all`. That removes two syscalls from the per-commit
/// path and — more importantly for [`DurabilityLevel::Normal`] — shrinks the
/// window in which a process death can leave a torn frame: a `write(2)`
/// cannot be interrupted partway by `SIGKILL`, so a frame that fits in one
/// write is either wholly in the page cache or wholly absent. A short write
/// is still possible in principle, so the length/CRC torn-tail check remains
/// the authority rather than an optimisation.
pub fn append_frame(w: &mut impl Write, frame: &WalFrame) -> io::Result<()> {
    append_frame_with_codec(w, frame, crate::serde_codec::CURRENT_CODEC)
}

fn append_frame_with_codec(
    w: &mut impl Write,
    frame: &WalFrame,
    codec: crate::serde_codec::CodecVersion,
) -> io::Result<()> {
    append_frame_bounded(w, frame, codec, MAX_WAL_FRAME_BYTES)
}

// One envelope writer; the explicit bound lets tests reject a full group
// without allocating a 4 GiB fixture. Production always passes the format cap.
fn append_frame_bounded(
    w: &mut impl Write,
    frame: &WalFrame,
    codec: crate::serde_codec::CodecVersion,
    limit: u64,
) -> io::Result<()> {
    let payload = crate::serde_codec::encode_versioned(codec, frame, limit)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    let len = u32::try_from(payload.len())
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "WAL frame exceeds 4 GiB"))?;
    let crc = crc32(&payload);
    let mut framed = Vec::with_capacity(8 + payload.len());
    framed.extend_from_slice(&len.to_le_bytes());
    framed.extend_from_slice(&crc.to_le_bytes());
    framed.extend_from_slice(&payload);
    w.write_all(&framed)?;
    Ok(())
}

/// Read a fixed-size buffer, mapping a clean OR partial EOF to `None`
/// (both end the frame stream). Any other I/O error propagates.
fn read_exact_opt(r: &mut impl Read, buf: &mut [u8]) -> io::Result<Option<()>> {
    match r.read_exact(buf) {
        Ok(()) => Ok(Some(())),
        Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => Ok(None),
        Err(e) => Err(e),
    }
}

/// Read and validate the WAL header. Returns the format version, or an
/// error if the magic is wrong. An empty reader (0 bytes) is an error —
/// a WAL file should always carry at least a header.
pub fn read_header(r: &mut impl Read) -> io::Result<u8> {
    let mut magic = [0u8; 4];
    r.read_exact(&mut magic)?;
    if magic != WAL_MAGIC {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "not a kglite WAL file (bad magic)",
        ));
    }
    let mut ver = [0u8; 1];
    r.read_exact(&mut ver)?;
    Ok(ver[0])
}

/// Read every intact frame from `r`, which must be positioned at the
/// start of the file; `stream_len` is the total byte length of the
/// stream (file size). Reads and validates the header, then frames
/// until a clean EOF or the first torn/corrupt frame (short read,
/// over-long declared length, or CRC mismatch) — that frame and
/// anything after it are discarded, modelling a crash mid-append.
/// Returns the recovered frames in file order.
///
/// `stream_len` bounds the per-frame allocation: a corrupt length
/// prefix can otherwise ask for up to 4 GiB *before* the short read is
/// detected. A declared length larger than the bytes remaining in the
/// stream is provably torn/corrupt and stops recovery without
/// allocating.
///
/// When recovery stops before consuming the whole stream, a one-line
/// warning naming how many frames were recovered and the byte offset of
/// the bad frame goes to stderr, so the loss is not silent. It
/// distinguishes a torn tail from mid-file damage, which call for opposite
/// responses — see [`recovery_diagnostic`].
pub fn read_frames(r: impl Read, stream_len: u64) -> io::Result<Vec<WalFrame>> {
    let (frames, diagnostic) = read_frames_diagnosed(r, stream_len)?;
    if let Some(message) = diagnostic {
        eprintln!("{message}");
    }
    Ok(frames)
}

/// [`read_frames`], handing back the stderr line instead of printing it.
///
/// The wording is the whole point of the diagnostic, and a test cannot capture
/// this process's stderr — so the one place that decides *which* wording a
/// given file earns is reachable from a test, rather than re-derived by one.
fn read_frames_diagnosed(
    r: impl Read,
    stream_len: u64,
) -> io::Result<(Vec<WalFrame>, Option<String>)> {
    let read = scan_frames(r, stream_len)?;
    Ok((read.frames, read.diagnostic))
}

#[derive(Clone, Copy)]
struct ResumePoint {
    version: u8,
    stream_len: u64,
    valid_bytes: u64,
}

struct FrameScan {
    frames: Vec<WalFrame>,
    diagnostic: Option<String>,
    resume: ResumePoint,
    non_tail_damage: bool,
}

impl FrameScan {
    fn ensure_appendable(&self) -> io::Result<()> {
        if self.non_tail_damage {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "WAL corruption at byte offset {} ends before EOF ({} bytes); refusing to append \
                     or truncate non-tail damage. Recover the sidecar or move it aside explicitly.",
                    self.resume.valid_bytes, self.resume.stream_len
                ),
            ));
        }
        Ok(())
    }
}

fn scan_frames(mut r: impl Read, stream_len: u64) -> io::Result<FrameScan> {
    let version = read_header(&mut r)?;
    let codec = wal_codec(version)?;

    let header_len = (WAL_MAGIC.len() + 1) as u64;
    let mut consumed: u64 = header_len;
    let mut frames = Vec::new();
    let stopped_at = loop {
        match read_frame_step(&mut r, stream_len, consumed, codec)? {
            FrameStep::Frame(frame, frame_len) => {
                frames.push(frame);
                consumed += frame_len;
            }
            // A clean EOF is the normal end and says nothing.
            FrameStep::Eof => break None,
            FrameStep::Torn => break Some((consumed, None)),
            // The length prefix survived, so the next frame boundary is
            // known and the bytes past this frame can be probed.
            FrameStep::Corrupt(frame_len) => break Some((consumed, Some(consumed + frame_len))),
        }
    };
    // A corrupt frame with a known end before EOF is not a trailing frame,
    // even if the following bytes do not decode. Never search payload bytes
    // for a guessed boundary; a torn prefix supplies no next boundary at all.
    let non_tail_damage = stopped_at
        .and_then(|(_, next)| next)
        .is_some_and(|next| next < stream_len);
    let diagnostic = stopped_at.map(|(offset, resume)| {
        let trailing = resume.map_or(0, |next| {
            count_intact_frames(&mut r, stream_len, next, codec)
        });
        recovery_diagnostic(offset, stream_len, frames.len(), trailing)
    });
    Ok(FrameScan {
        frames,
        diagnostic,
        resume: ResumePoint {
            version,
            stream_len,
            valid_bytes: consumed,
        },
        non_tail_damage,
    })
}

/// What the bytes at one position in the frame walk turned out to be.
///
/// Split out of [`read_frames`] so the *probe* below walks frames by exactly
/// the same rules recovery does — a probe with its own parser would answer a
/// question about a format it only approximates.
enum FrameStep {
    /// A complete frame: CRC matched and the payload decoded. Carries the
    /// frame and its total on-disk length (header + payload).
    Frame(WalFrame, u64),
    /// The stream ended exactly on a frame boundary — the normal end.
    Eof,
    /// The framing itself is unusable from here: a partial header, a
    /// zero-filled hole, a declared length past the end of the file, or a
    /// short payload. There is no trustworthy next-frame boundary, so nothing
    /// beyond this point can be probed.
    Torn,
    /// The frame's header was intact but its *contents* were not (CRC
    /// mismatch or an undecodable payload). Carries the frame's total on-disk
    /// length, which locates the following frame.
    Corrupt(u64),
}

/// Read one frame's worth of bytes at `frame_start`, classifying what is there.
fn read_frame_step(
    r: &mut impl Read,
    stream_len: u64,
    frame_start: u64,
    codec: crate::serde_codec::CodecVersion,
) -> io::Result<FrameStep> {
    let mut len_buf = [0u8; 4];
    if read_exact_opt(r, &mut len_buf)?.is_none() {
        // Clean EOF or torn length prefix. Only the partial prefix is a
        // failure; landing exactly on the end of the file is the normal end.
        return Ok(if frame_start == stream_len {
            FrameStep::Eof
        } else {
            FrameStep::Torn
        });
    }
    let mut crc_buf = [0u8; 4];
    if read_exact_opt(r, &mut crc_buf)?.is_none() {
        return Ok(FrameStep::Torn); // torn: length present, crc missing
    }
    let after_header = frame_start + 8;
    let len = u32::from_le_bytes(len_buf) as u64;
    let expected_crc = u32::from_le_bytes(crc_buf);

    if len == 0 {
        // A run of zero bytes — the shape an OS crash leaves when a
        // file's length was extended but its data block never reached
        // the platter, which `DurabilityLevel::Normal` makes reachable.
        // `crc32(b"") == 0`, so a zero prefix would otherwise pass the
        // CRC check as a "valid" empty frame and reach the decoder.
        // `append_frame` can never emit one (the smallest real payload
        // is a two-byte Postcard `lsn` + `ops` pair), so treat it as the
        // torn tail it is — by intent, rather than relying on the
        // decoder to reject it. Deliberately `Torn` and not `Corrupt`: a
        // hole says nothing about where the next frame starts, so the bytes
        // after it must not be probed as if they were one.
        return Ok(FrameStep::Torn);
    }
    if len > stream_len.saturating_sub(after_header) {
        // Declared length exceeds the bytes that exist — torn or
        // corrupt prefix. Stop WITHOUT allocating `len` bytes.
        return Ok(FrameStep::Torn);
    }
    let mut payload = vec![0u8; len as usize];
    if read_exact_opt(r, &mut payload)?.is_none() {
        return Ok(FrameStep::Torn); // torn: payload short
    }
    let frame_len = 8 + len;
    if crc32(&payload) != expected_crc {
        return Ok(FrameStep::Corrupt(frame_len));
    }
    let limits = crate::serde_codec::DecodeLimits::new(MAX_WAL_FRAME_BYTES, len);
    match crate::serde_codec::decode_exact_with::<WalFrame>(codec, &payload, len, limits) {
        Ok(frame) => Ok(FrameStep::Frame(frame, frame_len)),
        Err(_) => Ok(FrameStep::Corrupt(frame_len)),
    }
}

/// How many complete frames sit after a corrupt one, purely to tell the
/// operator which failure they have.
///
/// **Diagnostic only — the frames are still discarded.** A frame's meaning
/// depends on every frame before it having been applied, so recovery cannot
/// resume past a gap; what it *can* do is stop calling the result a crash
/// tail when the file plainly continues. Any I/O failure while probing ends
/// the count, because a diagnostic must never turn into a second failure.
fn count_intact_frames(
    r: &mut impl Read,
    stream_len: u64,
    mut consumed: u64,
    codec: crate::serde_codec::CodecVersion,
) -> usize {
    let mut count = 0;
    while let Ok(FrameStep::Frame(_, frame_len)) = read_frame_step(r, stream_len, consumed, codec) {
        count += 1;
        consumed += frame_len;
    }
    count
}

/// The stderr line [`read_frames`] prints when recovery stopped early.
///
/// Two failures wear the same stop: a **torn tail**, which is what a crash
/// mid-commit leaves and costs nothing, and **mid-file damage**, where frames
/// the writer completed sit after the bad one and are being thrown away.
/// `trailing` (frames that still decode after the corrupt one) is what
/// separates them; reporting the second as routine would file silently
/// discarded committed work as expected.
fn recovery_diagnostic(offset: u64, stream_len: u64, recovered: usize, trailing: usize) -> String {
    if trailing == 0 {
        return format!(
            "[kglite] WAL recovery stopped at a torn/corrupt frame at byte offset {offset} \
             (of {stream_len}); recovered {recovered} intact frame(s) before it. This is expected \
             after a crash mid-commit; the torn tail is discarded from recovered state. A writer repairs only a trailing \
             frame before appending; it refuses damage with a known following frame boundary."
        );
    }
    let discarded = stream_len.saturating_sub(offset);
    format!(
        "[kglite] WAL recovery stopped at a corrupt frame at byte offset {offset} \
         (of {stream_len}); recovered {recovered} intact frame(s) before it. At least \
         {trailing} later frame(s) still decode cleanly, and all {discarded} byte(s) from \
         the stop point to the end of the file are discarded: a frame's effect depends on \
         every frame before it, so the log cannot be trusted past the corruption. This looks \
         like mid-file damage rather than a crash tail — committed work is being dropped. \
         Check the storage this log lives on, and treat the last checkpoint plus the \
         {recovered} recovered frame(s) as the surviving state."
    )
}

/// Codec for a WAL header version, or an error naming what this build can
/// read. Every version in `MIN_READABLE..=CURRENT` shares one codec and one
/// op schema — see [`WAL_FORMAT_VERSION`].
fn wal_codec(version: u8) -> io::Result<crate::serde_codec::CodecVersion> {
    match version {
        MIN_READABLE_WAL_FORMAT_VERSION..=WAL_FORMAT_VERSION => {
            Ok(crate::serde_codec::CodecVersion::PostcardV1)
        }
        1 => Err(crate::graph::io::file::pre_014_bincode_error(
            "WAL format v1",
        )),
        _ => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "unsupported WAL format version {version} (this build reads \
                 v{MIN_READABLE_WAL_FORMAT_VERSION}-v{WAL_FORMAT_VERSION}). \
                 A WAL newer than the binary cannot be replayed safely: open \
                 the graph with a matching kglite build to recover it, or \
                 delete the '-wal' sidecar to discard work committed since \
                 the last save() checkpoint."
            ),
        )),
    }
}

/// The sidecar WAL path for a `.kgl` checkpoint file: `<path>-wal`. Keeps
/// the WAL adjacent to its checkpoint so one is never found without the
/// other being locatable.
pub fn wal_path(checkpoint: &Path) -> PathBuf {
    let mut s = checkpoint.as_os_str().to_owned();
    s.push("-wal");
    PathBuf::from(s)
}

/// Read every intact frame from the WAL at `path` for crash recovery.
/// A missing file yields no frames (a graph that was never mutated since
/// its checkpoint). Stops at the first torn/corrupt frame (see
/// [`read_frames`]).
pub fn recover(path: &Path) -> io::Result<Vec<WalFrame>> {
    match File::open(path) {
        Ok(f) => {
            let len = f.metadata()?.len();
            read_frames(BufReader::new(f), len)
        }
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(Vec::new()),
        Err(e) => Err(e),
    }
}

/// Recovery plus its verified append boundary, kept internal so callers
/// cannot manufacture a truncation point. The durable owner holds its writer
/// lease from this scan through `open_recovered`.
pub(crate) struct WalRecovery {
    pub(crate) frames: Vec<WalFrame>,
    resume: Option<RecoveredBoundary>,
}

enum AppendBoundary {
    Unscanned,
    Missing,
    Recovered(RecoveredBoundary),
}

impl AppendBoundary {
    fn recovered(&self) -> Option<&RecoveredBoundary> {
        match self {
            Self::Recovered(recovered) => Some(recovered),
            Self::Unscanned | Self::Missing => None,
        }
    }
}

struct RecoveredBoundary {
    point: ResumePoint,
    source: File,
    modified: std::time::SystemTime,
}

/// Compare opened files, retaining the source handle so its identity cannot be
/// recycled between recovery and append preparation.
fn same_open_file(left: &File, right: &File) -> io::Result<bool> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        let left = left.metadata()?;
        let right = right.metadata()?;
        Ok(left.dev() == right.dev() && left.ino() == right.ino())
    }
    #[cfg(windows)]
    {
        Ok(same_file::Handle::from_file(left.try_clone()?)?
            == same_file::Handle::from_file(right.try_clone()?)?)
    }
    #[cfg(not(any(unix, windows)))]
    {
        let _ = (left, right);
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "WAL file identity verification is unsupported on this platform",
        ))
    }
}

fn verify_recovered_file(file: &File, recovered: &RecoveredBoundary) -> io::Result<()> {
    let metadata = file.metadata()?;
    if !same_open_file(file, &recovered.source)?
        || metadata.len() != recovered.point.stream_len
        || metadata.modified()? != recovered.modified
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "WAL identity or contents changed after recovery; refusing to append",
        ));
    }
    Ok(())
}

pub(crate) fn recover_for_append(path: &Path) -> io::Result<WalRecovery> {
    let file = match File::open(path) {
        Ok(file) => file,
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            return Ok(WalRecovery {
                frames: Vec::new(),
                resume: None,
            });
        }
        Err(error) => return Err(error),
    };
    let metadata = file.metadata()?;
    let read = scan_frames(BufReader::new(&file), metadata.len())?;
    read.ensure_appendable()?;
    if let Some(message) = read.diagnostic {
        eprintln!("{message}");
    }
    let recovered = RecoveredBoundary {
        point: read.resume,
        source: file,
        modified: metadata.modified()?,
    };
    verify_recovered_file(&recovered.source, &recovered)?;
    Ok(WalRecovery {
        frames: read.frames,
        resume: Some(recovered),
    })
}

/// Best-effort fsync of a file's parent directory, so a freshly created
/// file's directory entry survives an OS/power crash (mirrors the
/// directory-fsync step of `io/file.rs::write_kgl_with`). Errors are
/// ignored: some filesystems don't support directory fsync, and the
/// file's own contents are already synced.
fn sync_parent_dir(path: &Path) {
    if let Some(dir) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        if let Ok(dirfile) = File::open(dir) {
            let _ = dirfile.sync_all();
        }
    }
}

/// Truncate a WAL to nothing and lay down a fresh header, `fsync`ing the
/// result. The caller supplies a handle opened for ordinary writing — never
/// the append handle (see [`prepare_wal_file`]).
fn truncate_to_header(file: &mut File) -> io::Result<()> {
    use std::io::{Seek, SeekFrom};
    file.set_len(0)?;
    // The read that classified the header left the cursor mid-file. Without
    // this seek an ordinary (non-append) write would land at that offset and
    // leave a hole in front of the header.
    file.seek(SeekFrom::Start(0))?;
    write_header(file)?;
    file.sync_all()
}

/// Validate the WAL at `path`, creating or repairing its header as needed, so
/// that [`Wal::open`] can take an append handle over a file already known to
/// be well-formed.
///
/// All header maintenance happens here, on an ordinary read/write handle, and
/// finishes before the append handle exists. **An append handle is not a
/// general-purpose write handle.** Rust maps `OpenOptions::append(true)` to
/// `FILE_GENERIC_WRITE & !FILE_WRITE_DATA` on Windows — deliberately dropping
/// the very right that truncation and in-place rewrites require — and an
/// append handle ignores seeks on write on every platform. Repairing a torn
/// header through the append handle is what POSIX tolerates and Windows does
/// not.
///
/// The classification rules applied below are documented on [`Wal::open`].
fn prepare_wal_file(path: &Path, boundary: &AppendBoundary) -> io::Result<File> {
    use std::io::{Seek, SeekFrom};
    let recovered = boundary.recovered();
    let header_len = (WAL_MAGIC.len() + 1) as u64;
    let mut file = OpenOptions::new()
        .create(matches!(boundary, AppendBoundary::Unscanned))
        .create_new(matches!(boundary, AppendBoundary::Missing))
        .read(true)
        .write(true)
        .truncate(false)
        .open(path)?;
    if let Some(recovered) = recovered {
        verify_recovered_file(&file, recovered)?;
    }
    let file_len = file.metadata()?.len();
    if file_len == 0 {
        write_header(&mut file)?;
        file.sync_all()?;
        sync_parent_dir(path);
        return Ok(file);
    }

    let mut header = [0u8; 5];
    let read_len = file_len.min(header_len) as usize;
    file.read_exact(&mut header[..read_len])?;
    let magic_ok = read_len >= WAL_MAGIC.len() && header[..4] == WAL_MAGIC;

    if file_len < header_len || (!magic_ok && file_len == header_len) {
        truncate_to_header(&mut file)?;
        return Ok(file);
    }
    if !magic_ok {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "{} is not a kglite WAL file (bad magic) and is not empty; \
                 refusing to overwrite it. Move the file aside if it is stale.",
                path.display()
            ),
        ));
    }

    // Reject an unreadable version before appending to it; the codec lookup
    // owns the actionable message.
    wal_codec(header[4])?;
    let point = match recovered {
        Some(recovered) if recovered.point.version == header[4] => recovered.point,
        Some(_) => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "WAL version changed after recovery; refusing to append",
            ))
        }
        None => {
            file.seek(SeekFrom::Start(0))?;
            let read = scan_frames(BufReader::new(&mut file), file_len)?;
            read.ensure_appendable()?;
            read.resume
        }
    };
    repair_tail(&file, point)?;
    if header[4] != WAL_FORMAT_VERSION {
        // A readable older version. We are about to append current-format
        // frames, so the header must advertise the newer version or a future
        // reader would parse the new frames under the old schema. Rewriting
        // the byte is lossless precisely because the older format is a subset
        // (see `WAL_FORMAT_VERSION`): the frames already in the file are valid
        // current-format frames, and the per-frame CRCs cover payloads only,
        // not the header.
        file.seek(SeekFrom::Start(WAL_MAGIC.len() as u64))?;
        file.write_all(&[WAL_FORMAT_VERSION])?;
        file.sync_data()?;
    }
    Ok(file)
}

/// Truncation is synced before an append handle exists, even at Normal:
/// later acknowledged frames must never sit behind a resurrected old tail.
fn repair_tail(file: &File, point: ResumePoint) -> io::Result<()> {
    if point.valid_bytes < point.stream_len {
        file.set_len(point.valid_bytes)?;
        file.sync_all()?;
    }
    Ok(())
}

/// An open, append-only WAL file. Session-scoped (one per open graph
/// file) — it owns a `File` handle, so it lives *outside* the CoW-cloned
/// `DirGraph` (which must stay `Clone`). Each [`append`](Self::append)
/// writes a frame, and under [`SyncMode::Barrier`] also flushes it to
/// stable storage, making the committed mutation durable before the call
/// returns.
///
/// The handle is deliberately **unbuffered** — `file` is a bare [`File`],
/// never a `BufWriter`. That is what makes [`SyncMode::PageCache`] mean
/// anything: the bytes are in the kernel's page cache by the time `append`
/// returns, so they outlive the process even without a barrier. Wrapping
/// this in a userspace buffer would silently downgrade
/// [`DurabilityLevel::Normal`] to "survives nothing".
#[derive(Debug)]
pub struct Wal {
    file: File,
    path: PathBuf,
    sync: SyncMode,
}

impl Wal {
    /// Open the WAL at `path` for appending, creating it with a fresh
    /// header if absent. Verified frames are preserved; an unreadable trailing
    /// frame is truncated and synced before appending. A corrupt frame ending
    /// before EOF is refused. Call [`recover`] first if its frames need replay.
    ///
    /// The header is validated on open. A file too short to hold a full
    /// header, or a header-sized file with the wrong magic, can never
    /// contain a frame — it is the residue of a crash between `create`
    /// and the header `fsync` — so it is truncated and re-initialised in
    /// place. A *longer* file with a bad magic could be somebody's data:
    /// that errors loudly instead of destroying it. A header naming a
    /// version this build cannot read (pre-0.14 v1, or anything newer than
    /// [`WAL_FORMAT_VERSION`]) is rejected before a single frame is
    /// appended; a *readable* older version is upgraded in place, since the
    /// frames already present parse under the current schema unchanged.
    ///
    /// `sync` fixes the per-append durability behaviour for the life of the
    /// handle; see [`SyncMode`]. Header and tail repair always barrier
    /// regardless of the level — a WAL whose header might not exist after a
    /// crash could not be recovered at all, and it is paid once per open
    /// rather than once per commit.
    pub fn open(path: PathBuf, sync: SyncMode) -> io::Result<Self> {
        Self::open_at_boundary(path, sync, AppendBoundary::Unscanned)
    }

    pub(crate) fn open_recovered(
        path: PathBuf,
        sync: SyncMode,
        recovered: WalRecovery,
    ) -> io::Result<Self> {
        Self::open_at_boundary(
            path,
            sync,
            recovered
                .resume
                .map_or(AppendBoundary::Missing, AppendBoundary::Recovered),
        )
    }

    fn open_at_boundary(
        path: PathBuf,
        sync: SyncMode,
        boundary: AppendBoundary,
    ) -> io::Result<Self> {
        // Maintenance uses a read/write handle: append handles cannot portably
        // truncate or seek-write. Reuse durable open's scan under its lease.
        let maintained = prepare_wal_file(&path, &boundary)?;
        let file = OpenOptions::new().read(true).append(true).open(&path)?;
        if !same_open_file(&file, &maintained)? {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "WAL identity changed before append open; refusing to append",
            ));
        }
        Ok(Self { file, path, sync })
    }

    /// Append one frame — the commit point.
    ///
    /// Under [`SyncMode::Barrier`] this returns only after the bytes are on
    /// stable storage; under [`SyncMode::PageCache`] once the kernel has
    /// them.
    pub fn append(&mut self, frame: &WalFrame) -> io::Result<()> {
        append_frame(&mut self.file, frame)?;
        self.file.flush()?;
        if self.sync == SyncMode::Barrier {
            self.file.sync_data()?;
        }
        Ok(())
    }

    /// Flush every frame appended so far to stable storage — the barrier
    /// that [`SyncMode::Barrier`] performs on every commit, on demand.
    ///
    /// Two callers, and both matter:
    ///
    /// 1. **Before a checkpoint.** A checkpoint truncates the log, so the
    ///    frames it folds in must already be on disk. If they are not, an OS
    ///    crash in the window between writing the checkpoint and truncating
    ///    the log can leave a *prefix* of the frames, and replaying that
    ///    prefix over the newer checkpoint would revert data the checkpoint
    ///    already holds. Under [`SyncMode::Barrier`] the frames are on disk
    ///    already and this is the no-op it looks like; under
    ///    [`SyncMode::PageCache`] it is load-bearing.
    /// 2. **On user demand.** It is the only way a `Normal` graph can reach
    ///    power-safety at a granularity finer than a whole checkpoint —
    ///    "flush at end of request", "flush before shutdown".
    pub fn sync(&mut self) -> io::Result<()> {
        self.file.flush()?;
        self.file.sync_data()
    }

    /// Reset to an empty WAL (header only), `fsync`ing the truncation.
    /// Called after a checkpoint (a full `.kgl` save) has folded every
    /// frame into the snapshot, so the log can start fresh.
    pub fn reset(&mut self) -> io::Result<()> {
        // Truncation and header rewrite need a dedicated read/write handle
        // (see `prepare_wal_file`). `self.file` stays usable afterwards —
        // append mode resolves the end of the file at write time, so the next
        // frame lands straight after the fresh header.
        let mut file = OpenOptions::new().read(true).write(true).open(&self.path)?;
        truncate_to_header(&mut file)
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

#[cfg(test)]
#[path = "wal_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "wal_tail_tests.rs"]
mod tail_tests;

#[cfg(test)]
#[path = "wal_v4_tests.rs"]
mod v4_tests;
