# C ABI

`kglite-c` is the supported boundary for non-Rust bindings (cgo, JNI,
P/Invoke, Swift, and similar FFIs). Rust embedders should call
`kglite::api::*` directly. The generated
[`kglite.h`](https://github.com/kkollsga/kglite/blob/main/crates/kglite-c/include/kglite.h)
is the exact symbol/signature authority; this page explains ownership and use.

## Build and versioning

```bash
cargo build -p kglite-c --release
```

The workspace version is lockstep across the engine, Python wrapper, C ABI,
servers, and CLI. `kglite_abi_version()` derives major/minor/patch from that
package version. Header drift is CI-gated; regenerate through the crate build,
never edit `kglite.h` by hand.

Precompiled C ABI libraries are not currently attached to releases. Build the
library from the matching workspace/crate source and package it for the target
platform alongside your binding.

## Status and error ownership

Every fallible call returns `KgliteStatusCode`:

```c
KgliteGraph *graph = NULL;
const char *error = NULL;
KgliteStatusCode status = kglite_load_file("graph.kgl", &graph, &error);
if (status != KGLITE_STATUS_CODE_OK) {
    fprintf(stderr, "%s\n", error ? error : "unknown kglite error");
    kglite_free_string(error);
    return 1;
}
```

Engine codes are `KGLITE_STATUS_CODE_CYPHER_SYNTAX` through
`KGLITE_STATUS_CODE_ONTOLOGY_VIOLATION` (1–23). Boundary-only failures use 100+ such as
`INVALID_UTF8` and `NULL_POINTER`. Output handles/messages are reset before
validation, and any returned error string is Rust-owned until freed with
`kglite_free_string`.

Name a code with `kglite_status_code_name_static`, which returns a `'static`
pointer into the library's own data: no allocation, and **never free it**. A
binding that renders the name on every error — the usual shape, since the name
goes into the exception it raises — should prefer it. `kglite_status_code_name`
returns the same text as an owned copy that *must* be freed with
`kglite_free_string`; it predates the static form and stays for callers that
would rather free one uniform kind of string. Pick one per call site; freeing
the static pointer is undefined behaviour. Both return null for
`KGLITE_STATUS_CODE_OK`.

## Opaque handles

`KgliteGraph`, `KgliteSession`, `KgliteCypherResult`, `KgliteEmbedder`, and
`KgliteWriterLease` are opaque. Create/load them only through exported
constructors and release them with their matching `*_free` function. Null-safe
free functions simplify error paths. Never copy/dereference the structs or free
Rust memory with the host allocator.

## Lifecycle and persistence

The header exposes:

- graph creation by storage mode, `.kgl`/RDF loading, graph generation, and
  blueprint construction;
- open-format exits: `kglite_export_csv` (lossless CSV tree) and
  `kglite_export_rdf` (RDF 1.2 N-Quads / TriG, `rdf` feature), each writing the
  whole graph and returning a JSON summary, and `kglite_load_rdf_with_options`
  (`kglite_load_rdf` plus `language_maps`). A session consumes its graph handle,
  so `kglite_session_export_csv` and `kglite_session_export_rdf` export a
  consistent snapshot of a live session instead;
- `kglite_open_or_create_graph_in_mode`, which opens or creates a path in an
  explicit mode (null mode = honour what the checkpoint recorded) and reports
  any conversion through `out_converted_from`. It can write: a mode with a
  missing path creates the graph, and a differing mode converts it;
- `kglite_load_file`, the read-only open: it never creates the path, never
  converts, takes no lease and writes nothing (a missing path is
  `FILE_NOT_FOUND`). A binding's read-only open uses this symbol and checks
  the mode from `kglite_graph_storage_mode` itself; no separate read-only
  symbol exists;
- `kglite_graph_storage_mode`, which reports the mode a graph handle is running
  in *right now* as an owned `"memory"` / `"mapped"` / `"disk"` string. It
  borrows the handle, so call it before `kglite_session_new` consumes it.
  `out_converted_from` above answers "what was it before?" and is null whenever
  nothing changed. This answers "what is it now?" unconditionally. That is the
  question after a creation, after an unspecified-mode open, or when asserting
  the mode you asked for is the mode you got;
- `kglite_writer_lease_acquire` / `kglite_writer_lease_free`. **Any caller that
  may save to a path must hold the lease across the whole read-modify-save
  interval; readers take none.** Two processes that both open, mutate, and save
  one path each publish a complete snapshot and the later one silently wins, so
  locking at save time is already too late. `timeout_ms = 0` fails fast, and a
  refusal (`KGLITE_STATUS_CODE_WRITER_LEASE_HELD`) names the holding process.
  The full write cycle is: acquire the lease → `kglite_open_or_create_graph_in_mode`
  → `kglite_session_new` → `kglite_session_execute_mut` → `kglite_session_save`
  → free the session → free the lease;
- `kglite_open_session`, the durable open. One call takes the writer lease,
  opens or creates the graph, and replays the write-ahead log. It returns a
  session whose commits are logged at the chosen level.
  - Options are a JSON object: `storage`, `durability` (`full` | `normal` |
    `off`), `lock_timeout_ms`, `valid_time_default`, `create_if_missing`,
    `auto_checkpoint_wal_mib`. An unknown key is `INVALID_ARGUMENT`.
  - `auto_checkpoint_wal_mib` (default 16, `0` disables) folds an oversized log
    into the checkpoint with an online checkpoint, run inline on the thread of
    the commit that crossed the bound. Other threads keep committing; that
    call takes the checkpoint's time.
  - `lock_timeout_ms = -1` takes no lease and opens read-only. Nothing is
    created, converted, logged or written, and every write is `READ_ONLY` (24).
  - A contended lease is `WRITER_LEASE_HELD` (102). The holder is in
    `kglite_last_error_details_json`.
  - The session owns the lease. `kglite_session_free` releases it.
    `kglite_session_close` checkpoints unsaved changes first; it is idempotent
    and the handle is still freed separately.
  - `kglite_session_sync` flushes the log, the power-safe point at `normal`.
    On a session with no log it returns `NOT_DURABLE` (25).
    `kglite_session_checkpoint` writes the opened path unless nothing changed.
  - Writes that bypass the log are refused with `DURABILITY_FAILED` on a logged
    session: schema, text and vector indexes, embedding ingest. `execute_mut`,
    `execute_mut_batch`, `create_edges_batch` and ontology declaration are
    logged.
  - A session from `kglite_session_new` has no log. Persist it with
    `kglite_session_save`;
- atomic/durable save, byte serialization, and schema JSON;
- `kglite_session_backup`, an online single-file `.kgl` snapshot of a session's
  published graph that does not stall writers. It takes no lease and writes no
  `-wal` sidecar. Pass `live_path` (the file the graph was opened from, or null)
  so a destination aliasing it is refused; a disk-mode graph is refused too.
  Both refusals report `KGLITE_STATUS_CODE_FILE_IO` with the reason in the
  message. The report is a JSON object (`path`, `bytes`, `nodes`,
  `relationships`, `graph_version`, `lsn` or null, `lock_hold_ms`,
  `elapsed_ms`) freed with `kglite_free_string`;
- `kglite_session_define_ontology` and `kglite_session_clear_ontology`, which
  declare and remove the session graph's ontology. The document is the JSON
  `define_ontology()` takes. A declaration at `warn` or `error` then binds every
  write through the session: a refused write returns
  `KGLITE_STATUS_CODE_ONTOLOGY_VIOLATION` (22) with the rule, type and property
  in the message, and `kglite_session_execute_mut_batch` rolls back the whole
  batch. Declaring over stored data that breaks an `error` rule also returns 22;
  `out_warnings_json` then holds the per-rule report (`rule`, `entity`,
  `entity_type`, `property`, `count`). On success it holds the `warn`-level
  findings. After any 22, `kglite_last_error_details_json` returns the
  structured fields (`rule`, `entity`, `entity_type`, `property`, `report`) as
  JSON for the failing call on the calling thread, so a caller does not parse
  the message. The declaration is durable once it is committed to a session
  opened by `kglite_open_session` (the write-ahead log holds it), or once
  `kglite_session_save` runs on any other session;
- `kglite_session_save`, the checkpoint for a graph that has been moved into a
  session. `kglite_session_new` takes ownership of the graph handle, so a graph
  mutated through `kglite_session_execute_mut` is persisted from the session
  rather than from the (now-consumed) graph handle; `kglite_save_graph` stays
  the entry point for a graph that was never moved into one. The save writes
  through the session's own graph — it never copies the graph to checkpoint it
  — and is serialized against concurrent mutations on that session. The graph
  handle is consumed **only on `KGLITE_STATUS_CODE_OK`**. A failed
  `kglite_session_new` leaves ownership with the caller, who must still
  `kglite_graph_free` it;
- session construction plus read/mutation execution. `_opts` takes a timeout
  and a work budget. `_ex` takes a versioned `KgliteExecuteOptions` block that
  adds a result-row cap. The cap truncates instead of failing and reports
  `row_limit` and `total_rows` in the result's diagnostics. The block carries
  its own `struct_size`. The library reads only that many bytes and treats
  later fields as zero, so a field appended later is additive;
- read and mutation batches, including atomic edge batches;
- JSON result metadata/rows, memory statistics, and embedder binding.

`.kgl` is the cross-binding handoff format. The current writer emits RGF
v7/Postcard and the current reader accepts v7, v6 and v5. RGF v4/bincode and
older containers are rejected with a clear migration/rebuild message; convert
them with kglite 0.13.4 before crossing the C boundary. A v7 file cannot be
read by kglite 0.19.0 or earlier, so a prebuilt consumer must be rebuilt
against this engine before it is handed one.

## Sessions and transactions

Use `kglite_session_execute_read[_opts|_ex]` for reads and
`kglite_session_execute_mut[_opts|_ex]` for auto-committed mutations. Mutation
batches commit atomically.

### Explicit transactions

| Call | Effect |
| --- | --- |
| `kglite_session_begin(session, read_only, &tx, &err)` | Takes a snapshot and returns an owned `KgliteTx`. |
| `kglite_tx_execute(tx, query, params, options, &result, &err)` | Runs one statement; `options` is the `KgliteExecuteOptions` block. |
| `kglite_tx_commit(tx, &err)` | Publishes the writes atomically, then finishes the transaction. |
| `kglite_tx_rollback(tx)` | Discards the writes and finishes the transaction. |
| `kglite_tx_free(tx)` | Frees the handle. An open transaction is rolled back, never committed. |

- A statement sees the transaction's own earlier writes. Other readers and
  transactions see none of them until commit.
- A failed statement is rolled back on its own; the transaction stays open.
- A commit that loses to another writer returns
  `KGLITE_STATUS_CODE_TRANSACTION_CONFLICT` and applies nothing. Begin a new
  transaction and redo the work.
- On a durable session a commit is logged before it is published, like Bolt's
  `COMMIT`. A log failure is `KGLITE_STATUS_CODE_DURABILITY_FAILED` and applies
  nothing.
- A read-only transaction reads one fixed snapshot and refuses writes with
  `KGLITE_STATUS_CODE_READ_ONLY`. A read-write `begin` on a read-only session
  is refused the same way.
- After commit or rollback the transaction is finished. `execute` and `commit`
  on it return `KGLITE_STATUS_CODE_INVALID_ARGUMENT`; `rollback` returns OK.
- A transaction is single-threaded: never call its functions concurrently.
  The session stays usable from other threads, and several transactions on one
  session run independently.
- Free every transaction before freeing or closing its session.

### Cancelling a running query

| Call | Effect |
| --- | --- |
| `kglite_cancel_token_new(&token)` | Creates an owned `KgliteCancelToken`. |
| `kglite_cancel_token_cancel(token)` | Asks every call carrying the token to stop. Callable from any thread. |
| `kglite_cancel_token_free(token)` | Frees the handle. |

- Attach the token through `KgliteExecuteOptions.cancel`, the field after
  `reserved`. It is read only when `struct_size` covers it, so a caller built
  before the field existed is unaffected. It applies to
  `kglite_session_execute_read_ex`, `kglite_session_execute_mut_ex` and
  `kglite_tx_execute`.
- A stopped query returns `KGLITE_STATUS_CODE_CANCELLED` (17) at its next
  check. A cancelled write publishes nothing.
- A cancelled token stays cancelled. Make one token per query you may want to
  stop.
- Each call takes its own reference to the token before it starts, so
  `kglite_cancel_token_free` is safe while the call runs. Never pass a freed
  handle to a new call or to `cancel`, and never race `cancel` against `free`
  on the same handle.

Query parameter JSON is checked recursively before execution. Integer tokens
must fit signed 64-bit; decimal or exponent tokens must fit a finite 64-bit
float. A refusal returns `KGLITE_STATUS_CODE_INVALID_ARGUMENT`, leaves the
result output null, and, when the caller supplies `out_error_msg`, returns an
owned message naming the nested parameter path. Free that message with
`kglite_free_string`. This applies to single, options and batch query calls;
edge/property ingestion keeps its declared tolerant conversion policy.

JSON has no date type, so a date or datetime parameter is a one-key tagged
object: `{"$date": "2020-01-01"}` binds a date (parsed as `date()` parses),
`{"$datetime": "2020-01-01T10:00:00+02:00"}` binds a datetime (parsed as
`datetime()` parses; an offset is applied, normalising to UTC), and
`{"$duration": {"months": 0, "days": 1, "seconds": 0}}` binds a duration. The
payloads are the shapes result rows render those types as, so a cell read back
and wrapped in its tag matches the stored value. A bare string stays a string.
An object with any other key, or with a tag key beside other keys, is an
ordinary map. A malformed payload is refused like an unrepresentable number.

A point parameter is `{"$point": {"lat": 60.1, "lon": 5.2}}`; each coordinate
is a number or a `$float` tag. `{"$map": {...}}` binds the map `{...}` itself,
the escape for a map whose only key is a tag name.

JSON has no NaN or infinity, so a non-finite float parameter is the tagged
object `{"$float": "NaN"}`, `{"$float": "inf"}` or `{"$float": "-inf"}`. Any
other payload is refused like an unrepresentable number. A bare JSON number
never carries them, and a parameter is never turned into `null`. `-0.0` is an
ordinary number and keeps its sign.

Every JSON input decodes the same tags: `kglite_create_edges_batch` edge
properties and endpoint ids, recipe record parameters, `from_records` records
and Cypher `parse_json()`. Those tolerant paths keep a tagged object whose
payload is malformed as an ordinary map instead of refusing it.

## Result access

Results remain owned by `KgliteCypherResult` until
`kglite_cypher_result_free`. Column and row helpers return JSON strings for
portable decoding in the host language. Copy/parse data before freeing the
result, and free every independently returned string with
`kglite_free_string`.

Result rows use natural JSON by default, as every earlier release did. A date
and a datetime are strings, a duration is `{"months", "days", "seconds"}`, a
point is `{"latitude", "longitude"}`, and a non-finite float is `null`. Call
`kglite_session_set_result_encoding(session, 1)` (`KGLITE_RESULT_ENCODING_TAGGED`)
to render those values as the tags a parameter accepts, so a cell read back and
bound again is unchanged.

- `{"$date": "2020-01-02"}`
- `{"$datetime": "2020-01-02T03:04:05.250"}`, with no zone
- `{"$duration": {"months": 0, "days": 1, "seconds": 0}}`
- `{"$point": {"lat": 60.1, "lon": 5.2}}`
- `{"$float": "NaN" | "inf" | "-inf"}`
- `{"$map": {...}}`, which wraps a map whose only key is itself a tag name, so
  it is not read as that tag

The setting covers `kglite_cypher_result_rows_json` and the rows of both batch
results, for every result the session produces afterwards. It nests at any
depth, including node and relationship properties and a point's coordinates.
Strings, integers, ids, finite floats and booleans are the same in both
encodings. Value 0 is `KGLITE_RESULT_ENCODING_NATURAL`; any other value except
1 is `INVALID_ARGUMENT`. The Java binding turns the tagged encoding on and
decodes the tags to `LocalDate`, `LocalDateTime`, `KgliteDuration`, `Point` and
`Double`.

Query warnings (an unknown label or relationship type, a row-cap truncation)
arrive only in `kglite_cypher_result_diagnostics_json`'s `warnings` array. The
library does not print them to the host process's stderr.

A `PROFILE` query's diagnostics also carry a `profile` array: one
`{"clause", "rows_in", "rows_out", "elapsed_us"}` object per executed clause,
in execution order. The key is absent for an unprofiled query. Batch results
carry the same object under each statement's `diagnostics`.

### Reading a large result in batches

`kglite_session_cursor_open` runs a read query and returns a `KgliteCursor`.
`kglite_cursor_next_batch(cursor, max_rows, &rows_json, &error)` returns up to
`max_rows` rows as a JSON array in the same encoding as
`kglite_cypher_result_rows_json`. An empty array means the cursor is exhausted.

- **Memory:** a plain `MATCH ... RETURN <expressions>` is produced as it is
  pulled, so a caller that drops each batch holds a few batches whatever the
  row count. `kglite_cursor_streamed` returns `true` for that shape.
- **Everything else:** `ORDER BY`, `DISTINCT`, aggregation, `UNION`, several
  clauses, a `row_limit`, `max_work_units` and a disk graph are built whole by
  the engine, exactly as `kglite_session_execute_read` builds them. The cursor
  then slices the finished rows, and `kglite_cursor_streamed` returns `false`.
- **Snapshot:** the cursor reads the graph as it was at open and keeps it alive
  until `kglite_cursor_free`. A later commit does not change the cursor, and
  freeing the session does not invalidate it.
- **Errors:** a parse or planning error is returned by `open`. An execution
  error, a cancellation or a timeout is returned once by `next_batch`, and the
  cursor then reports exhausted.
- **Limits:** the `KgliteExecuteOptions` timeout and cancel token apply for the
  cursor's life, including between batches.
- **Not carried:** the diagnostics JSON of a result. Schema warnings are not
  available from a cursor.

## Binding checklist

1. Validate UTF-8 and nullability before calls.
2. Map all status codes, including `CANCELLED`; preserve the message/code.
3. Wrap opaque handles in deterministic finalizers plus explicit close/free.
4. Keep async/runtime/logging/iteration style in the binding; the core is sync.
5. Test null outputs, double-free-safe cleanup paths, malformed JSON/UTF-8,
   timeout/budget failures, and concurrent session use.
6. Compile against the generated header and run the C-ABI integration/header
   drift checks for every release.

See [Implementing a binding](implementing-a-binding.md) for the architectural
boundary and [Session abstraction](session.md) for the native Rust pipeline.
