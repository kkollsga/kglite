# Durable embedded apps

This guide covers running KGLite as the **embedded database behind an
application** — the open → mutate → reopen lifecycle, persistence on close,
and crash-safe durable writes via a write-ahead log (WAL).

If you only build a graph, query it, and throw it away, you don't need any of
this — `KnowledgeGraph()` plus {doc}`data-loading` is enough. Reach for this
guide when the graph is *long-lived state* your app reopens across runs:
an agent's memory, a knowledge base that accretes facts, a service that
accepts writes between restarts.

## The lifecycle entry points

| Call | What it does |
|---|---|
| `kglite.open(path)` | **Load-or-create.** Opens the graph at `path` if it exists, creates a fresh one bound to `path` if it doesn't. The database-style entry point. |
| `kglite.load(path)` | Load an existing `.kgl` file (or disk-mode directory). Raises `kglite.FileError` if missing, `kglite.FileFormatError` if corrupt (see below). |
| `g.save(path=None, *, fsync=True)` | Write a full checkpoint, **atomically and durably**. With no `path`, saves back to the remembered path. |
| `g.to_bytes()` / `kglite.from_bytes(data)` | Serialize/deserialize the graph to/from a `.kgl` **byte buffer** — own the write (object storage, a pipe, a checksum) instead of a filesystem path. |
| `g.close()` | Persist to the remembered path. The graph stays usable afterwards. |
| `with kglite.open(...) as g:` | Auto-saves on clean block exit; **skips** the save if the block raises, preserving the last good file. |

**Every `save()` is atomic and torn-proof**, even in non-durable mode: it writes
to a sibling temp file and atomically renames it over the target, so a crash
mid-save can never leave a half-written `.kgl` — a reader always sees the old
file or the complete new one. With `fsync=True` (default) the file and its
directory are flushed to physical storage before returning; pass `fsync=False`
to skip that flush for speed in a hot loop (still atomic). This removes the
temp-file + `os.replace` + dir-fsync dance consumers used to hand-roll.

**Corrupt-file detection is typed.** `load()` / `from_bytes()` raise
`kglite.FileFormatError` (a subclass of `kglite.KgError`) on a corrupt,
truncated, or wrong-format input, and `kglite.FileError` on a missing file — so
a disposable-cache consumer can branch "corrupt → rebuild from source" vs
"missing → create new" cleanly, without a broad `except IOError`.

The thread that ties these together is the **remembered path**: `open()` and
`load()` record where the graph came from, so a later bare `save()` — or the
context manager's auto-save — writes back without you re-specifying the target.

```python
import kglite

# First run: file doesn't exist → fresh graph, bound to "app.kgl".
with kglite.open("app.kgl") as g:
    g.cypher("CREATE (:Person {id: 1, name: 'Alice'})")
# clean exit → auto-saved to app.kgl

# Next run: file exists → loaded back.
with kglite.open("app.kgl") as g:
    g.cypher("CREATE (:Person {id: 2, name: 'Bob'})")
    print(g.cypher("MATCH (p:Person) RETURN count(p) AS n").scalar())  # 2
```

## The default: crash-safe

`open()` is durable by default. Every committed mutation is `fsync`'d before the
call returns, so a mutation that has returned survives a hard crash.

```python
g = kglite.open("app.kgl")
g.cypher("CREATE (:Order {id: 1001, total: 49.90})")   # fsync'd before this returns
```

You get this without asking for it because it is what makes an embedded database
trustworthy: the alternative default silently loses every write since the last
explicit `save()` whenever a process dies.

The default is the *strongest* level, not the only one. If power-loss safety is
more than your application needs, `durable="normal"` keeps the log and drops
only the per-commit barrier — see [Choosing a durability
level](#choosing-a-durability-level).

## Choosing a durability level

`durable` names **what a committed mutation survives**. It uses SQLite's
`synchronous` vocabulary, and the levels are stated as guarantees rather than
as syscalls — the syscall differs by platform, the guarantee does not.

| `durable=` | A committed mutation survives… | Per-commit cost |
|---|---|---|
| `"full"` (or `True`) — **default** | process crash, OS crash, **power loss** | one barrier |
| `"normal"` | **process crash** — `kill -9`, panic, OOM-kill | no barrier |
| `"off"` (or `False`) | nothing since the last `save()` | no log |

`True` and `False` are accepted spellings of `"full"` and `"off"`, so existing
code keeps working unchanged.

### `"normal"` — the process-crash level

```python
g = kglite.open("app.kgl", durable="normal")
g.cypher("CREATE (:Order {id: 1001, total: 49.90})")   # logged, not barriered
```

The frame is handed to the kernel with a plain write before the call returns.
**The page cache belongs to the kernel, not to your process**, so the commit
survives your process dying by any means — an uncaught exception, `kill -9`,
the OOM killer. What it does not survive is the *kernel* dying: an OS crash or
a power cut loses commits made since the last `save()`.

That is the right trade for most applications, because a crashing process is
the failure that actually happens and a power cut is the one you keep backups
for. It is also the level to reach for when per-commit barrier latency is
shaping your write throughput — `"normal"` writes the same log frame and skips
only the barrier.

### `"off"` — no log at all

When the graph is rebuildable from source data — a bulk load, a derived index,
a scratch analysis — logging buys nothing:

```python
g = kglite.open("kb.kgl", durable="off")
g.add_nodes(df, node_type="Topic", unique_id_field="id")   # nothing logged
g.save()          # one explicit checkpoint at the end
g.close()
```

What that is **not**: crash-safe. A snapshot is written only when *you* call
`save()`/`close()` or the context manager exits cleanly. If the process is
killed mid-session (`kill -9`, power loss, an unhandled crash before the next
`save()`), the work since the last checkpoint is gone.

### Taking a power-safe point on demand: `sync()`

`"normal"` skips the per-commit barrier — but you can take that barrier
whenever it matters:

```python
g = kglite.open("app.kgl", durable="normal")

def handle_request(payload):
    g.cypher("CREATE (:Event $props)", params={"props": payload})

handle_request(...)
g.sync()      # everything committed so far now survives power loss too
```

`sync()` writes **no checkpoint** and truncates nothing — it only makes the
existing log durable, which is why it is the right granularity for "flush at
the end of a request" or "flush before shutdown". A full `save()` republishes
the entire graph and is far more expensive.

Under `"full"` it returns immediately (every commit was already barriered). On
a graph with no log it raises `ValueError` rather than silently doing nothing,
because a caller who believes they bought power-safety and got nothing is the
failure that costs data.

## How durability works

With durability on, every committed mutation is appended to a
`<path>-wal` sidecar file and `fsync`'d to stable storage **before the call
returns**. A mutation that has returned is guaranteed to survive a hard crash.

How it fits together:

- **Each mutation** → one WAL frame, written before the call returns. Under
  `"full"` the frame is also barriered to stable storage per commit; that
  barrier is the durability cost, and it bounds write latency by device
  latency rather than engine speed (see "Cost and tuning" below). Under
  `"normal"` the frame is written but not barriered.
- **`save()`** → writes a full checkpoint (`.kgl`) and **truncates the WAL**.
  The checkpoint is the new baseline; the WAL starts empty again.
- **`open(...)`** → loads the last checkpoint, then **replays**
  any WAL frames written since it, reconstructing the exact committed state —
  including work that was never checkpointed because the process crashed.

So the on-disk state is always "last checkpoint + replayable tail", and reopen
folds the two back together automatically.

### Crash recovery in practice

```python
import os

# Process A — commits, then dies hard before any save().
g = kglite.open("app.kgl", durable=True)
g.cypher("CREATE (:Person {id: 1, name: 'Alice'})")   # committed + fsync'd
g.cypher("CREATE (:Person {id: 2, name: 'Bob'})")     # committed + fsync'd
os._exit(1)   # hard crash — no save(), no clean close

# Process B — reopen recovers both, from the WAL.
g = kglite.open("app.kgl", durable=True)
assert g.cypher("MATCH (p:Person) RETURN count(p) AS n").scalar() == 2
g.save()   # checkpoint: fold the WAL into a fresh .kgl, truncate the log
```

Both rows survive the crash even though `save()` was never called in process A —
they were `fsync`'d to the WAL at commit time, and reopen replayed them.

## Choosing the mode

KGLite has three persistence postures for an embedded app. Pick by what you're
optimising for:

| You want… | Use | Trade-off |
|---|---|---|
| Every committed write to survive a hard crash | `open(path)` (the default) | One barrier per commit; reopen is O(graph) (loads the whole graph). |
| Committed writes to survive a crashing *process*, cheaply | `open(path, durable="normal")` | No barrier per commit; an OS crash or power cut loses work since the last `save()`. Call `sync()` for a power-safe point. |
| Maximum write throughput on rebuildable data | `open(path, durable="off")` | Nothing logged; a crash loses work since the last checkpoint. |
| Crash safety on a graph that outgrows RAM | `open(path, storage="mapped")` | Same per-commit WAL guarantee; property columns spill to mmap. |
| 100 M+ nodes (Wikidata-scale), cheap cold-open | `open(path, storage="disk")` | Paged mmap, lazy load; **no per-commit WAL** — durability is your `save()` calls. |

The first three are **in-memory** — the whole graph lives in RAM, which is what
makes traversal and multi-hop queries fast. Durability adds crash-safety on
top of that model without changing the in-memory read path.

**If your app is simply growing, reach for `mapped`, not `disk`.** `mapped`
is the larger-than-RAM mode that keeps this guide's guarantee: it is durable by
default and its crash recovery is kill-9 tested alongside in-memory. `disk` is
a different trade — a Wikidata-scale exploration mode whose commit boundary is
an explicit `save()`, not a logged write (see the Limitations below). Choosing
`disk` because a graph got big means giving up per-commit crash safety you
did not have to give up.

## Serving concurrent reads

A `KnowledgeGraph` is single-owner — don't share one instance across threads
while a thread mutates it (that raises a clear `RuntimeError`). For a read-heavy
server, take an immutable snapshot with `g.freeze()` → a `FrozenGraph` that
shares the data via an O(1) clone and serves `cypher()` from many threads at
once, lock-free. When the data changes, build/reload, `freeze()` again, and swap
the snapshot in. See {doc}`/concepts/concurrency` for the full model.

```python
snapshot = g.freeze()
# hand `snapshot` to N reader threads — concurrent, lock-free
snapshot.cypher("MATCH (o:Order) RETURN count(o)")
```

**Durability and shared concurrent writes don't combine in one handle.** A
`Session` (`graph.session()` / `kglite.open_session(...)`) serves shared reads
and serialized writes, but its `execute()` writes land on a working copy visible
only through that session — reachable by neither the log nor the owning graph's
`save()`. A `Session` write against a durable graph therefore **raises**, rather
than applying a mutation nothing can persist; reads are unaffected. For a
durable app, keep writes on the durable `KnowledgeGraph` itself (there they are
serialized and `fsync`'d) and use `freeze()` snapshots for concurrent reads.
Reach for `Session` writes with `durable=False`, when you need shared concurrent
writes but **not** durability. See {doc}`/concepts/concurrency` for the full
model.

## Cost and tuning

- **`"full"` is barrier-bound, not engine-bound.** A workload of many small
  committed transactions spends its time waiting on the disk to confirm each
  barrier, not in KGLite. This is the price of power-loss safety and is
  inherent to any WAL database. The cost scales with the *number* of commits
  and with device latency, not with graph size, and reads pay nothing at all.
- **`"normal"` is the level to try before you reach for `"off"`.** It writes
  the same log frame and skips only the barrier, so it costs roughly what an
  unlogged write costs while still losing nothing to a crashing process. If
  you were about to disable durability purely for write throughput, this is
  almost always the better answer — and `sync()` gives you power-safe points
  wherever you actually need them.
- **On macOS, `"full"` buys more than SQLite's default does.** KGLite's
  barrier is `F_FULLFSYNC`, which flushes the drive's own write cache;
  SQLite's default `synchronous=FULL` issues a plain `fsync`, which on macOS
  does not. The guarantees are therefore not the same thing measured
  differently — KGLite's default is the stronger one, and it costs
  accordingly.
- **Batch where you can.** One `cypher()` that creates 1,000 nodes is one
  `fsync`; 1,000 separate `cypher()` calls are 1,000 `fsync`s. Group related
  mutations into a single statement (or a transaction — see
  {doc}`/python/transactions`) when they logically commit together.
- **Checkpoint to bound recovery time.** Reopen replays every WAL frame since
  the last `save()`. Replay is fast (frames are folded into net per-entity state
  and the index rebuilt once), but a periodic `save()` keeps the WAL short and
  recovery near-instant for write-heavy, rarely-restarted services.

## Limitations

- **Not available for `storage="disk"`.** A disk graph commits by publishing an
  immutable generation, so its durability boundary is that publish rather than a
  logical log; reconciling a replayed frame against a published generation needs
  a generation-aware log this release does not have. `open(path,
  storage="disk")` therefore opens **non-durable**, and **both `durable="full"`
  and `durable="normal"` raise `ValueError`** rather than pretending — the
  levels are not uniform across storage modes, because the blocker here is the
  commit boundary itself and not barrier strength. `storage="disk"` supports
  only `durable="off"`. The in-memory default and `storage="mapped"` support
  every level — if you want crash safety on a graph that outgrew RAM, `mapped`
  is the answer.

  What disk mode *does* guarantee is worth stating exactly, because it is
  stronger than "no crash safety" and is kill-9 tested
  (`crates/kglite/tests/disk_crash_guarantee.rs`):

  > A crash loses exactly the mutations made since the last `save()`, and
  > nothing else. The graph reopens at the last published generation, complete
  > and uncorrupted — never at a partially-written one.

  Between `save()` calls, disk-mode mutations live only in the process's heap
  overlay; nothing is written, so nothing can be half-written. The publish
  itself is crash-atomic: the staged snapshot is `fsync`'d, renamed into place,
  and only then does an atomically-replaced `CURRENT` pointer select it, so a
  crash mid-publish leaves the previous generation selected. No acknowledged
  commit is ever lost — the acknowledgement point is your `save()` call.

  **Budget for the checkpoint's cost before you sprinkle `save()` calls.**
  Every disk `save()` writes a complete new generation and the superseded ones
  are retained, so *N* checkpoints leave *N* full copies of the graph on disk.
  That is deliberate — readers hold their generation mmap'd and must not have
  it deleted underneath them — but there is no retention policy yet, so
  checkpoint on a schedule you have the disk budget for, and prune old
  `generations/gen_*` directories yourself once no reader is using them.
- **Some state is checkpoint-only.** The log describes nodes, edges, and
  labels. Schema and config metadata, user-created indexes, embeddings, and
  timeseries have no log entry, so they are persisted by `save()` rather than
  recovered by replay. Call `save()` after changing them if a crash must not
  lose them.
- **A `with` block is not a transaction.** Each mutation commits as it runs, so
  an exception inside the block does not undo mutations that already returned —
  they are recovered on the next `open()`. Use `begin()` when you want
  discard-on-error ({doc}`/python/transactions`).

## See also

- {doc}`/python/transactions` — `begin()` / `commit()` / `rollback()`,
  snapshot isolation, and how the Bolt server consumes the same surface.
- {doc}`/python/core-concepts` — the memory / mapped / disk storage modes.
- {doc}`data-loading` — bulk-loading the seed data an app starts from.
