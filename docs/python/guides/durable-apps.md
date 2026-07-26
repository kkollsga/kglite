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

## Opting out: `durable=False`

Durability costs one `fsync` per commit. When the graph is rebuildable from
source data — a bulk load, a derived index, a scratch analysis — that cost buys
nothing, and `durable=False` gives you the older snapshot-on-close behaviour:

```python
g = kglite.open("kb.kgl", durable=False)
g.add_nodes(df, node_type="Topic", unique_id_field="id")   # no fsync per write
g.save()          # one explicit checkpoint at the end
g.close()
```

What that is **not**: crash-safe. A snapshot is written only when *you* call
`save()`/`close()` or the context manager exits cleanly. If the process is
killed mid-session (`kill -9`, power loss, an unhandled crash before the next
`save()`), the work since the last checkpoint is gone.

## How durability works

With durability on, every committed mutation is appended to a
`<path>-wal` sidecar file and `fsync`'d to stable storage **before the call
returns**. A mutation that has returned is guaranteed to survive a hard crash.

How it fits together:

- **Each mutation** → one WAL frame, `fsync`'d per commit. This is the
  durability cost: durable writes are bounded by `fsync` latency, not by engine
  speed (see "Cost and tuning" below).
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
| Every committed write to survive a hard crash | `open(path)` (the default) | One `fsync` per commit; reopen is O(graph) (loads the whole graph). |
| Maximum write throughput on rebuildable data | `open(path, durable=False)` | No `fsync` per write; a crash loses work since the last checkpoint. |
| Graphs larger than RAM, cheap reopen | `open(path, storage="disk")` | Paged mmap, lazy load; not a crash-safe-per-write WAL mode. |

The first two are **in-memory** — the whole graph lives in RAM, which is what
makes traversal and multi-hop queries fast. Durability adds crash-safety on
top of that model without changing the in-memory read path. `storage="disk"`
(see {doc}`/python/core-concepts`) is the separate answer for *larger-than-RAM*
graphs and cheap cold-open; it is not combined with the WAL.

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

- **Durability is `fsync`-bound, not engine-bound.** A workload of many
  small committed transactions spends its time waiting on the disk to confirm
  each `fsync`, not in KGLite. `durable=False` does the same logical work
  far faster precisely because it skips the per-commit `fsync`. This is the
  price of crash-safety and is inherent to any WAL database. The cost scales
  with the *number* of commits and with device latency, not with graph size,
  and reads pay nothing at all.
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
  storage="disk")` therefore opens **non-durable**, and passing an explicit
  `durable=True` raises `ValueError` rather than pretending. Use `save()`
  checkpoints there. The in-memory default and `storage="mapped"` are both
  fully durable.
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
