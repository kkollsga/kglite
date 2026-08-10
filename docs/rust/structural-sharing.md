# Structural sharing: what a write costs while a reader is alive

`DirGraph` is shared as `Arc<DirGraph>`. A lazy `ResultView`, a `freeze()`, a
`Session`, an open `Transaction` and every fluent-derived handle each hold one,
and a write needs `&mut DirGraph` — so the writer forks.

That fork used to be a deep copy of the whole graph: every node, every edge,
every index. On a 1M-node graph it cost **36.3 ms and a second 668.8 MB of
resident memory**, against a 3 µs write with nothing held, and it fired on
`rows = g.cypher(...)` followed by `g.cypher("… SET …")` — no threads, no
snapshot API, no explicit copy. This page is the durable record of what
replaced it.

**The fork is now O(changes).** Held-view first write, 1M nodes, mean of the
timed write with the reference re-acquired untimed every round:

| graph shape | before | after |
|---|---:|---:|
| plain | 36,338 µs | **4.6 µs** |
| saved / columnar | ~17,000 µs | **4.0 µs** |
| 2 property + 1 composite + 1 range index | ~180,000 µs | **~97 µs** (all of it the range index — see [Limits](#limits)) |
| resident growth, 20 writes under a held view, 1M | +668.8 MB | **+0.0 MB** |

---

## Why writer-side, and not the obvious alternative

The attractive design is reader-side MVCC: keep one graph, let the snapshot read
through an undo chain, and the live graph pays nothing. The journal already
captures exactly the pre-images such a read would need.

**It is not expressible in safe Rust.** The snapshot holds `Arc<DirGraph>` and
reads it as `&DirGraph` while the writer needs `&mut DirGraph` to the same
allocation. That is aliasing UB. The only escapes are a lock on the read path —
categorically over the in-memory budget in the `MATCH` loop — or a query-duration
read guard, which makes writes *block* on a held Python object: a deadlock
hazard traded for a latency cliff.

**Rust's aliasing rules force copy-on-write to be writer-side.** That is the
structural fact behind everything below, and it is the answer to "why not just
version the reads".

---

## The mechanism

The reader's `Arc<DirGraph>` is left byte-for-byte untouched. The writer builds a
graph whose data-scale fields share the parent's allocations plus a small delta,
and folds the delta back the moment the base becomes uniquely owned again.

Four fields carry the graph-sized state, and each is layered:

| field | layer | shape |
|---|---|---|
| `graph` (the backend) | `GraphBackend::Forked` — `storage/forked.rs` | base `Arc<MemoryGraph>` + per-node/edge copy-on-write maps and tombstones |
| `id_indices` | `storage/disk/id_index_layer.rs` | per type: `Owned` or `Layered { base: Arc<TypeEntry>, delta }`, recursive chain, deletions as `NodeIndex::end()` tombstones |
| `type_indices` | `storage/disk/type_index_layer.rs` | per type: `Vec<Arc<Vec<NodeIndex>>>`, **append-only**, last level writable |
| `property_indices`, `composite_indices` | `dir_graph/index_layer.rs` | per index: `Vec<Arc<HashMap<K, Option<Vec<NodeIndex>>>>>`, `None` = tombstone, bucket-granular copy-on-write |

Three of the four are the same idea — a stack of shared immutable levels whose
tail is writable — and the differences are dictated by each field's access
pattern, not by taste:

* **`type_indices` is append-only** because a `CREATE`'s only edit is a push, and
  the realistic shape is one type holding nearly every node. A per-bucket
  copy-on-write would copy a million-entry `Vec` on the first write and win
  nothing.
* **The user index families are bucket-granular** because they are point-keyed:
  a statement touches two buckets, not the index. Their delta bucket is a *full
  copy of the merged bucket*, which is what keeps the journal's reversals
  correct unchanged (see [Invariants](#invariants)).
* **`id_indices` splits inside `Clone`** through its existing `RwLock`, because
  it can: it hands out values, not borrowed slices. The other layers cannot take
  a lock — their reads return `&[NodeIndex]` — so they keep *every* level behind
  an `Arc`, including the one being written, and the writer discovers the fork
  lazily at its next write through `Arc::get_mut`.

### The one thing that must not be done

**A `share()` that merges before it can hand back one immutable value is
O(N), and it fires on every fork that follows a write** — which is the founding
defect's own shape, since a read-then-write loop re-takes a view every
iteration. This was measured, not reasoned about: it held the 1M cell at 4.1 ms
with every other part of the design correct and every test green.

Its twin: **compaction written as "materialise into a fresh structure" is also
O(N)**, because materialising clones the base first. It reads as a fold and
behaves as the deep clone the design removes, moved one write later. It showed
up as a +289% regression on the *dropped-view control*, not on the cell under
test.

Both are avoided the same way — the base is moved out of its `Arc`
(`Arc::try_unwrap` under a `get_mut` guard), never copied — and both are pinned
by tests that assert on pointer identity rather than on content.

---

## Invariants

**Slot identity.** Statement rollback guarantees a node or edge returns on the
exact `NodeIndex`/`EdgeIndex` it vacated, and those indices are the keys of every
index structure. `StableGraph` reuses free-list slots and offers no
index-controlled insertion, so the overlay must *predict* what `add_node` will
return and reproduce it at fold-back. `storage/slot_mirror.rs` mirrors petgraph's
two free lists as LIFO stacks and refuses to predict — rather than guessing —
for a graph whose free-list order is not observable (one adopted by `from_graph`
or restored by serde, unless it provably has no holes). Unsynced means *slower*,
never wrong. A `debug_assert` validates the prediction on every insert the test
suites perform.

**The journal reverses into the delta, never the base.** Every `UndoEntry` is
keyed on an index and replayed through the write path; on a forked graph that
replay must land in the overlay. If any of it reached the shared base, the
reader's snapshot would silently acquire a rolled-back write — no error, no
crash. Two specific re-pointings:

* `BucketAppended` on a node-type bucket is reversed by editing the **writable
  tail**, and *refuses* rather than guessing when the entry is not there (the
  caller falls back to a flattening retain: slower, still correct). A statement's
  appends are all in the tail because a fork needs `&DirGraph` while the writer
  holds `&mut DirGraph`, so no fork can interleave with a statement.
* `BucketRemoved` re-inserts at a recorded **position**, and
  `BucketAppended` on a user index drops the **last** occurrence so a
  pre-statement occurrence of the same node is spared. Both work unchanged
  because a materialised delta bucket *is* the merged bucket the position was
  measured against.

**`supports_undo_journal()` must stay true on the forked backend.** If it were
false, every statement taken while a view is held would fall back to a
whole-graph clone checkpoint — an O(V+E) copy *per statement* instead of one
per fork, i.e. the fix introducing a worse cliff than the defect.

**Depth caps.** Every layer bounds its stack at **32** levels and flattens once
at the cap. A stack only grows while a reader is held *continuously across
writes*; any write with nothing shared folds it back. The cap is not removable:
it is what bounds memory (one retained delta per level) and read-miss depth. Its
value is measured, not chosen — at 8 the flatten put a ~5x-median spike into one
round in eight and dominated the *mean* of the held-view cell; at 32 the worst
case is ~2x the median. Raising it further tunes against one benchmark's hold
window rather than against a mechanism.

**Fork-private caches.** `edge_type_counts_cache` and `type_connectivity_cache`
are `ForkPrivateCache`: no `Arc`, and `Clone` returns an *empty* cache. They used
to be `Arc<RwLock<…>>` shared by a plain clone, which was a real wrong-observable
— a snapshot holder reported the *writer's* edge-type counts as its own. Making
the aliasing structurally impossible beats avoiding it. `wkt_cache` (a pure
function of its key) and `property_ndv_cache` (version-tagged, and only a planner
estimate) stay shared deliberately. `peer_counts` and the mapped lazy indexes are
likewise reset on fork: correct-but-cold beats a cache shared with a reader's
snapshot.

**Compaction contract.** Fold-back runs at write entry, which is the earliest
moment the writer can observe the reader's departure — `Arc::get_mut` succeeding
*is* that observation. So "hold a view, write, drop the view, write again"
self-heals on the very next write. A fold must be O(delta) and must decline
while any other holder is alive; the `Arc::get_mut` gate is the whole
enforcement.

**`g.copy()` forks *from* the source.** The source's own backend and the copy's
base then become the same allocation while the source is still uniquely owned,
so writing through it would edit a backend the copy is reading.
`ensure_writable()` at write entry turns the source into an overlay too — one
`Arc::get_mut` probe in the steady state.

---

## Limits

These are real and deliberate; none of them is a defect.

**Adjacency edits flatten the overlay.** `add_edge`, `remove_node` and
`remove_edge` rewrite existing nodes' petgraph adjacency, which the overlay
cannot express without chaining base⊕overlay behind all six iterator types.
They flatten instead — one whole-graph copy, the pre-D2 cost, paid **once per
fork rather than once per statement**. Accepting that boundary is what let the
overlay skip adjacency chaining entirely. A statement *rollback* that removes a
node it created flattens for the same reason. Extending the overlay to adjacency
is a well-defined follow-up.

**A continuously held reader pays an amortised flatten.** In a loop that
re-takes a view before every write, compaction never fires and the depth cap
does. On a graph with large user indexes that is ~`|index| / 32` per fork —
measured at ~4.3 ms per write on a 1M indexed graph, against a 97 µs median
round. Read plainly: the median improves ~1,500x and the mean ~32x. The flatten
cannot be made cheaper, because it must copy a base that a reader holds.

**`range_indices` is not layered.** It is a `BTreeMap` per index, and
`lookup_range` needs ordered iteration, which a level stack can only serve by
k-way merging across levels — a different mechanism, not an incidental
extension. It is therefore the whole remaining fork cost on an indexed graph:
~90 µs for an index over ~1,000 distinct values, and **O(distinct values)**, so
a range index over a high-cardinality property costs proportionally more.

**`unique_indices`, `embeddings` and `timeseries_store` are not layered
either.** `unique_indices` holds one entry per node of every constrained type;
`embeddings` is linear in dimension (6.9 ms at d=64 on 1M nodes, so ≈41 ms at
d=384, and more once an HNSW index exists, whose `links` allocate per node per
layer). A graph carrying those still pays them on every fork.

**`Mapped` stays on the deep-clone path**, explicitly.

**The rollback pre-image clone is a different cost and is unchanged by this
work.** It is not the fork: it fires once per write *statement* on a columnar
type, with no reader held. Measured at
**≈ (N / 100,000) × (368 + 41 × columns) µs** — linear in both axes, ≈8.6 ms per
statement at 1M × 12 columns, and ≥97% of the write above N = 25,000. On a
never-saved graph the term is exactly zero. The overlay neither improves nor
worsens it: it is the statement checkpoint capturing a pre-image, not the fork
copying a graph, and the two are independent. Anyone reading a slow write on a
saved graph should check this term before suspecting the fork.

---

## Observing it

`GraphBackend::is_forked()` is public as a diagnostic, and bindings expose it
(`kglite._backend_is_forked` in Python). It is the one cheap, non-timing
observable that separates the three states this design moves between: flat,
forked, and folded back. Regression tests assert on it rather than on timings —
`False` where a fork is expected means whole-graph-clone semantics returned;
`True` where flat is expected means compaction stopped folding.

Engine-side the oracle is `BACKEND_CLONE_NODES` (test-only), which counts nodes
*actually copied*, so it distinguishes a genuine deep copy from the O(1) clone of
an intentionally emptied backend.
