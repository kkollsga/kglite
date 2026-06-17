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

## The default: "feels like a database", checkpoint on close

Plain `open()` (without `durable=True`) gives you ergonomic persistence: open,
mutate, close → your work is on disk. This is the right default for the common
case where an app does a batch of work and exits cleanly.

```python
g = kglite.open("kb.kgl")
g.cypher("MERGE (:Topic {id: 'graphs', label: 'Graph theory'})")
g.save()          # explicit checkpoint, back to kb.kgl
# ... more work ...
g.close()         # final checkpoint
```

What this is **not**: crash-safe. A snapshot is written only when *you* call
`save()`/`close()` or the context manager exits cleanly. If the process is
killed mid-session (`kill -9`, power loss, an unhandled crash before the next
`save()`), the work since the last checkpoint is gone. For many apps that's
fine — checkpoint often, accept losing the current batch on a crash.

When losing the in-flight batch is *not* acceptable, use durable mode.

## Crash-safe writes: `durable=True`

```python
g = kglite.open("app.kgl", durable=True)
g.cypher("CREATE (:Order {id: 1001, total: 49.90})")   # fsync'd before this returns
```

With `durable=True`, every committed Cypher mutation is appended to a
`<path>-wal` sidecar file and `fsync`'d to stable storage **before the call
returns**. A mutation that has returned is guaranteed to survive a hard crash.

How it fits together:

- **Each mutation** → one WAL frame, `fsync`'d per commit. This is the
  durability cost: durable writes are bounded by `fsync` latency, not by engine
  speed (see "Cost and tuning" below).
- **`save()`** → writes a full checkpoint (`.kgl`) and **truncates the WAL**.
  The checkpoint is the new baseline; the WAL starts empty again.
- **`open(..., durable=True)`** → loads the last checkpoint, then **replays**
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
| Fast, all-in-RAM; lose the current batch on a crash is acceptable | `open(path)` (non-durable) | No `fsync` per write; crash loses work since the last checkpoint. |
| Every committed write to survive a hard crash | `open(path, durable=True)` | One `fsync` per commit; reopen is O(graph) (loads the whole graph). |
| Graphs larger than RAM, cheap reopen | `open(path, storage="disk")` | Paged mmap, lazy load; not a crash-safe-per-write WAL mode. |

The first two are **in-memory** — the whole graph lives in RAM, which is what
makes traversal and multi-hop queries fast. `durable=True` adds crash-safety on
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

## Cost and tuning

- **`durable=True` is `fsync`-bound, not engine-bound.** A workload of many
  small committed transactions spends its time waiting on the disk to confirm
  each `fsync`, not in KGLite. The non-durable mode does the same logical work
  far faster precisely because it skips the per-commit `fsync`. This is the
  price of crash-safety and is inherent to any WAL database.
- **Batch where you can.** One `cypher()` that creates 1,000 nodes is one
  `fsync`; 1,000 separate `cypher()` calls are 1,000 `fsync`s. Group related
  mutations into a single statement (or a transaction — see
  {doc}`/python/transactions`) when they logically commit together.
- **Checkpoint to bound recovery time.** Reopen replays every WAL frame since
  the last `save()`. Replay is fast (frames are folded into net per-entity state
  and the index rebuilt once), but a periodic `save()` keeps the WAL short and
  recovery near-instant for write-heavy, rarely-restarted services.

## Limitations

- **In-memory only this release.** `durable=True` with `storage="mapped"` or
  `storage="disk"` raises `ValueError`. Crash-safe durable writes apply to the
  in-memory model; use `storage="disk"` for the larger-than-RAM case (without
  per-write WAL durability).
- **Durability is per *committed* mutation.** A statement that errors out
  commits nothing. For multi-statement atomicity, wrap the work in a
  transaction ({doc}`/python/transactions`) and commit once.

## See also

- {doc}`/python/transactions` — `begin()` / `commit()` / `rollback()`,
  snapshot isolation, and how the Bolt server consumes the same surface.
- {doc}`/python/core-concepts` — the memory / mapped / disk storage modes.
- {doc}`data-loading` — bulk-loading the seed data an app starts from.
