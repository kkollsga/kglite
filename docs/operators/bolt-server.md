# `kglite-bolt-server` — Neo4j Bolt v5 protocol server

The bolt-server is a pure-Rust binary that exposes a loaded kglite
`.kgl` graph over the [Bolt v5 wire protocol](https://neo4j.com/docs/bolt/current/).
Any Neo4j-aware client — the official Python/JS/Java/Go drivers,
Cypher Shell, Neo4j Browser, BloodHound, LangChain's `Neo4jGraph` —
can connect and run Cypher queries with **zero consumer-side
changes**.

## Architecture

The Bolt server is a **thin async transport** over the same sync
Cypher pipeline the Python API uses:

```
neo4j driver  ──Bolt v5 over TCP──▶  boltr crate (server crate
                                     handles handshake, message
                                     framing, PackStream, session
                                     state machine)
                                              │
                                              ▼
                                     KgliteBackend (impl BoltBackend)
                                              │
                                              │  ┌──────────────────┐
                                              ├──▶ kglite::api::cypher
                                              │   parse → validate →
                                              │   optimize → execute
                                              │
                                              │   (same code path as
                                              │    Python cypher())
                                              │
                                              ▼
                                     Arc<Mutex<Arc<DirGraph>>>
                                     + per-tx working copies
```

Three things live in bolt-server but not in the Python API:

- **`Arc<Mutex<Arc<DirGraph>>>`** for the shared graph — so commits
  can swap the inner Arc.
- **Per-session transaction state** in a `HashMap<TransactionHandle,
  Arc<Mutex<TxState>>>` mirroring the snapshot/working CoW shape the
  Python `Transaction` class uses.
- **`async fn` glue** to satisfy boltr's `BoltBackend` trait.

The Cypher pipeline itself is the same. Both `cypher()` and bolt-
server's `execute()` call into `kglite::api::cypher::parse_cypher`,
`validate_schema`, `rewrite_text_score`, `optimize_with_disabled`,
`is_mutation_query`, `CypherExecutor::with_params(...).execute()`,
and `execute_mutable()`. Differential testing against the
27-query corpus confirms row-for-row equivalence
(`tests/test_bolt_server_differential.py`).

## CLI reference

```
kglite-bolt-server --graph PATH [OPTIONS]
```

| Flag | Default | Meaning |
|---|---|---|
| `--graph PATH` | (required) | Path to a `.kgl` graph file to serve |
| `--bind ADDR` | `127.0.0.1` | Interface to bind |
| `--port N` | `7687` | TCP port (Neo4j default) |
| `--readonly` | off | Reject all mutations: auto-commit + explicit `BEGIN` both error with `Neo.ClientError.Security.Forbidden` |
| `--auth {none\|basic}` | `none` | `none` accepts any LOGON; `basic` validates against `--auth-user` / `--auth-pass` |
| `--auth-user STR` | — | Username for `--auth basic` |
| `--auth-pass STR` | — | Password for `--auth basic` |
| `--idle-timeout SECS` | (disabled) | Per-session idle timeout — boltr reaps idle sessions, calling `close_session` (which rolls back any in-flight tx) |
| `--max-sessions N` | `256` | Max concurrent Bolt sessions |
| `--max-message-size BYTES` | `16777216` (16 MiB) | Reject Bolt messages exceeding this size — protects against memory exhaustion from pathologically large queries |

Example:

```bash
kglite-bolt-server \
  --graph my-graph.kgl \
  --bind 0.0.0.0 --port 7687 \
  --auth basic --auth-user neo4j --auth-pass secret \
  --idle-timeout 300 \
  --readonly
```

## Connection URLs

- **`bolt://host:port`** — direct connection. **Use this.**
- **`neo4j://host:port`** — *routed* connection (cluster-aware).
  Rejected: returns `Neo.ClientError.Request.Invalid` with the
  message `routing not supported by kglite-bolt-server — connect
  with bolt:// (direct) rather than neo4j:// (routed)`.

## Auth modes

| Mode | Behavior |
|---|---|
| `--auth none` (default) | boltr's LOGON handler accepts any credentials. Drivers sending `auth=None` or `auth=("anything", "anything")` both succeed. |
| `--auth basic` | `BasicAuthValidator` checks `scheme == "basic"` and credentials against `--auth-user` / `--auth-pass`. Mismatch → `Neo.ClientError.Security.Unauthorized` → driver raises `AuthError`. |

## Tracing / observability

The server uses `tracing` for structured logs. Filter via `RUST_LOG`:

```bash
# Default: info-level for our crate, warn-level for boltr.
RUST_LOG=kglite_bolt_server=info,boltr=warn kglite-bolt-server ...

# Verbose: per-session create/configure/close + per-tx begin/commit/rollback.
RUST_LOG=kglite_bolt_server=debug kglite-bolt-server ...

# Quiet: errors only.
RUST_LOG=kglite_bolt_server=warn kglite-bolt-server ...
```

Each log line carries structured fields (session_id, tx, etc.) for
filtering downstream.

## Known limitations

- **No auto-commit mutations.** Mutations (`CREATE` / `SET` /
  `DELETE` / `MERGE`) must be wrapped in an explicit `BEGIN` /
  `COMMIT`. Auto-commit reads work fine. (Drivers always wrap writes
  in BEGIN/COMMIT in practice; supporting auto-commit mutations adds
  surface for no real win.)
- **Single-graph only.** No multi-database support. `USE db_name`
  and per-session database switching via `configure_session` are
  accepted but ignored.
- **No causal consistency / bookmarks.** Each session sees a
  consistent snapshot during a transaction; the `bookmark` field is
  not returned on COMMIT.

Formerly listed here, now supported: **OCC version checking on
commit** (Phase E.4 — conflicting concurrent commits are rejected
with a retryable error instead of last-writer-wins), **TLS** via
`--tls-cert` / `--tls-key` (Phase F), **`neo4j://` routing URIs**
via a single-server routing table (`--advertise-addr`), and
Neo4j-canonical **`CALL db.labels() YIELD label`** /
`db.relationshipTypes() YIELD relationshipType` column names.

## Driver compatibility matrix

| Driver | Status | Notes |
|---|---|---|
| `neo4j` (Python ≥ 5.0) | ✓ Verified — 226+ tests in `tests/test_bolt_server_*.py` | 6.1.0 actively tested in CI |
| Cypher Shell | Untested | Should work; uses Java driver internally |
| Neo4j Browser | Untested | Configure browser to point at `bolt://localhost:7687`; expect it to display the graph but specific Browser features (e.g. graph view, query plan visualisation, db.* discovery dialogs) may stumble on the divergences above |
| LangChain `Neo4jGraph` | Untested | Uses Python driver; should work for basic Cypher |
| BloodHound | Untested | Should work for Cypher-level queries; BloodHound-specific Neo4j features (APOC procs etc.) NOT supported |
| Java / JS / Go drivers | Untested | All use the same Bolt v5 protocol; should work but not exercised |

The Python driver path is the **only one with automated regression
coverage**. Other drivers are likely to work but exercise them
manually before relying on them.

## Common error symptoms

| Symptom | Cause | Fix |
|---|---|---|
| `ServiceUnavailable: Failed to establish connection` | Server not running, or wrong port | Check `--bind` / `--port`; verify with `nc -z host port` |
| `AuthError: invalid username or password` | `--auth basic` is on; driver sent wrong creds | Match driver's `auth=(...)` against the server's `--auth-user` / `--auth-pass` |
| `ClientError: routing not supported ...` | Driver used `neo4j://` URI scheme | Switch to `bolt://` |
| `ClientError: server is read-only — ...` | `--readonly` is on | Either start server without `--readonly` or use a read-only query |
| `ClientError: empty Cypher query` | Whitespace-only or empty string sent | Check the query string isn't being silently truncated upstream |
| `ClientError: multi-statement queries not supported` | Cypher with a `;` separator | Split into separate `session.run` calls, or use `BEGIN`/`COMMIT` to group |
| `ClientError: Cypher syntax error at line X` | Parser rejected the query | Standard Cypher syntax fix |
| `ClientError: non-finite Float parameter: NaN` | Parameter was `float('nan')` or `inf` | Send `None` (translates to Cypher `NULL`) if absence is what you mean |
| `DatabaseError: ...` (no specific code) | Cypher executor returned an error not yet typed by the string heuristic | Check server tracing logs for the underlying error; the message in the FAILURE response is unchanged from the original kglite error string |
| Connection drops with no error | Tokio task panicked (RA-2 + RA-3 should prevent this; if you see it, file a bug) | Run with `RUST_LOG=debug` to capture the panic site |

## Performance shape

Numbers from `tests/benchmarks/test_bench_bolt.py` on M4 Apple
Silicon (release build, debug-build kglite extension), one bolt-
server, one driver, fresh 10k-Person + 30k-KNOWS graph:

| Benchmark | Min latency | Notes |
|---|---|---|
| `tx_commit_no_writes` | ~77 µs | BEGIN + commit() with zero mutations. Arc clone + handle mint + cleanup. |
| `connect_and_run` | ~278 µs | Full handshake + LOGON + RUN + GOODBYE for one query. Amortizes to negligible across many queries per session. |
| `run_overhead_vs_direct` (100 scalar rows) | ~583 µs | One small RUN+PULL on an already-open driver session. Compare to direct-Python `cypher()` baseline (`test_bench_return_node_rel_node_100`, ~292 µs in B.3) → Bolt wire tax ≈ +290 µs. |
| `tx_commit_with_100_writes` | ~8.8 ms | BEGIN + 100 CREATEs + COMMIT (≈88 µs per CREATE inside a tx). |
| `pull_10k_scalars` | ~63 ms | 10k Person.name rows. Boltr's PULL pagination + driver-side materialization. |
| `pull_10k_nodes` | ~167 ms | 10k full Node structs (Phase A.1 + C.4 path). The 2.6× ratio over `pull_10k_scalars` is the per-row cost of full Node packStream encoding. |

**Practical advice:**

- For **request/response patterns** (a driver sending one query and
  reading the result): expect ~300 µs to ~5 ms latency depending on
  query complexity.
- For **bulk read**: a 10k-row pull takes 60-170 ms. If the same
  data fits in memory you may want to do it in-process via the
  Python API (no wire tax at all).
- For **bulk write**: BEGIN + many CREATEs + COMMIT is the right
  pattern; ≈88 µs per CREATE means 100k inserts is roughly 9
  seconds.
- For **many small sessions**: amortize by holding the driver
  connection open (the neo4j driver pools sessions). A fresh
  `verify_connectivity` + RUN + close costs ~280 µs; reuse is much
  cheaper.

## See also

- [`docs/python/transactions.md`](transactions.md) — how the
  Python `Transaction` class works, including the OCC + CoW pattern
  that bolt-server mirrors.
- [`docs/concepts/concurrency.md`](concurrency.md) — the
  underlying `Arc<DirGraph>` + GIL-release model.
- [`bolt_implementation.md`](../../bolt_implementation.md) — Phase
  plan and status, including the boltr v0.2 dependency rationale.
- [`tests/test_bolt_server_*.py`](../../tests) — the 226+ tests
  exercising the server (smoke / correctness / transactions /
  concurrency / robustness / differential).
