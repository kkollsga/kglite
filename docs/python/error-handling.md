# Error handling

KGLite exposes a typed Python exception hierarchy for engine, Cypher, schema,
transaction, and storage failures. Catch the narrowest class you can recover
from; catch `kglite.KgError` when every KGLite engine failure has the same
handling policy.

## Exception hierarchy

```text
Exception
└── kglite.KgError
    ├── kglite.CypherError
    │   ├── kglite.CypherSyntaxError
    │   ├── kglite.CypherTimeoutError
    │   ├── kglite.CypherExecutionError
    │   └── kglite.CypherTypeMismatchError
    ├── kglite.SchemaError
    ├── kglite.ValidationError
    ├── kglite.ExprError
    ├── kglite.ConstraintError
    │   ├── kglite.ConstraintViolationError
    │   └── kglite.ConstraintCreationError
    ├── kglite.TransactionConflictError
    ├── kglite.NodeNotFoundError
    ├── kglite.ConnectionNotFoundError
    ├── kglite.PropertyNotFoundError
    ├── kglite.FileError
    ├── kglite.FileFormatError
    ├── kglite.FileIoError
    ├── kglite.LoadMemoryLimitError
    ├── kglite.ArgumentError
    ├── kglite.MissingArgumentError
    ├── kglite.InternerCollisionError
    └── kglite.InternalError
```

`CypherSyntaxError` always has `.line` and `.col` attributes (either may be
`None`). `CypherExecutionError` has them when the executor can identify the
source position. Timeout messages report the elapsed and configured limit.

## Stable codes

Every instance carries `.code`, a stable classifier string — branch on that
rather than on message prose, which is free to improve between releases:

```python
try:
    graph.cypher(query)
except kglite.KgError as exc:
    log.warning("kglite failed", extra={"kglite_code": exc.code})
```

`.code` is also readable on the concrete classes themselves
(`kglite.ConstraintViolationError.code == "ConstraintViolation"`), so a
dispatch table can be built up front. It is `None` on the three abstract bases
— `KgError`, `CypherError`, `ConstraintError` — which each span several codes.
The same strings appear as `KGLITE_STATUS_*` in the C ABI and drive the Bolt
`Neo.*` status mapping, so one code means the same thing in every binding.

## Constraint violations

A write that breaks a declared UNIQUE / NOT NULL / NODE KEY / `IS :: TYPE`
constraint raises `ConstraintViolationError` — from **every** write path,
`cypher()` and the bulk loaders alike. Relationship constraints
(`FOR ()-[r:T]-() REQUIRE r.p IS NOT NULL` / `IS :: TYPE`) raise the same
exception, with a message written in relationship words. Declaring a constraint the stored data already violates is a
different problem with a different fix, so it raises the sibling
`ConstraintCreationError`; both subclass `ConstraintError`.

```python
try:
    graph.cypher("CREATE (u:User {email: $email})", params={"email": email})
except kglite.ConstraintViolationError:
    raise Conflict("that email is already registered")
```

The message names the constraint, the property, and the offending value, so it
is worth logging — but the type and `.code` are the contract.

## Transaction conflicts

`Transaction.commit()` raises `TransactionConflictError` when the graph moved
since `begin()`. Nothing was applied, so the fix is to re-run the work against
a fresh `begin()` — see {doc}`transactions` for `retry_on_conflict`, which is
that loop.

```python
try:
    tx.commit()
except kglite.TransactionConflictError:
    ...  # rebuild the transaction and try again
```

## Catching errors

```python
import kglite

try:
    result = graph.cypher(query, params=params, timeout_ms=30_000)
except kglite.CypherSyntaxError as exc:
    print(f"invalid query at {exc.line}:{exc.col}: {exc}")
except kglite.CypherTimeoutError:
    print("rewrite, scope, or explicitly increase the deadline")
except kglite.CypherError as exc:
    print(f"query failed: {exc}")
```

A timed-out Cypher query raises `CypherTimeoutError`; it does not return a
partial `ResultView`. For rollback-safe mutations, execute the query through
a {doc}`Transaction or Session <transactions>` rather than directly on
`KnowledgeGraph`.

For a broad engine boundary:

```python
try:
    graph = kglite.load("graph.kgl")
    rows = graph.cypher(query)
except kglite.KgError as exc:
    log.error("KGLite operation failed: %s", exc)
```

## Built-in Python exceptions

`KgError` is not a wrapper around every Python failure. Python-facing
protocols retain conventional exceptions:

| Situation | Exception |
|---|---|
| Missing result column or mapping key | `KeyError` |
| Invalid Python-side value or unsupported wrapper mode | `ValueError` |
| Wrong Python object or argument shape | `TypeError` |
| Wrapper-side path opening | `FileNotFoundError` where documented |
| Borrow or object-lifecycle conflict | `RuntimeError` |
| User cancellation with Ctrl-C | `KeyboardInterrupt` |

`KeyboardInterrupt` is deliberately outside `KgError`; an interrupt is a user
action, not a query fault. Catch it separately if the application needs
cleanup:

```python
try:
    graph.cypher(long_read, timeout_ms=0)
except KeyboardInterrupt:
    print("cancelled")
```

## Loading and recovery

Load failures are classifiable: a missing engine-managed path raises
`FileError`; malformed, truncated, or unsupported saved data raises
`FileFormatError`; other I/O failures raise `FileIoError`.

A fourth case is not a failure of the file at all. `kglite.load(path,
max_load_mb=N)` — and the process-wide `KGLITE_MAX_LOAD_MB` — refuse a load
whose estimated peak memory is over the ceiling, *before* decompressing
anything, and raise `LoadMemoryLimitError`. The graph is valid; this process
cannot afford it. Rebuilding would not help, which is exactly why it is its own
class: raise the ceiling, pass `defer_index_rebuild=True` (usually the largest
term), or load it somewhere with more memory. `kglite.estimate_load_memory(path)`
returns the same estimate as a dict of named terms, so a caller can decide for
itself rather than setting a ceiling.

```python
budget_mb = 512
try:
    graph = kglite.load("large.kgl", max_load_mb=budget_mb)
except kglite.LoadMemoryLimitError:
    # The index rebuild is the term usually worth dropping.
    graph = kglite.load("large.kgl", max_load_mb=budget_mb, defer_index_rebuild=True)
```

The ceiling compares an *estimate* read from the file's metadata head, not a
measurement, and it errs high on purpose — so a ceiling set close to a graph's
real cost can refuse a load that would have fitted. Set it where a failure is
what you want (a serving process that must not be killed by a file it did not
choose), not as a tight budget.

The write half classifies the same way: a `save()`, `sync()` or `to_bytes()`
that fails on I/O — a full disk, a read-only directory, a failing device —
raises `FileIoError` too, so `except kglite.KgError` covers both directions of
the file lifecycle. A `save()` *refused* before it touched the path (no
remembered path, or a write-ahead sidecar running ahead of the target) is a
`ValueError` instead: nothing was written, and the fix is a different argument
rather than a different disk.

```python
try:
    graph = kglite.load("cache.kgl")
except kglite.FileError:
    graph = rebuild_from_source()
except kglite.FileFormatError:
    graph = rebuild_from_source()
```

A CSV export is an interoperability view, not a byte-for-byte graph backup:
labels, schema, indexes, embeddings, time series, and some structured values
are not fully preserved. Keep the original source or a tested rebuild path;
see {doc}`guides/import-export` for the exact persistence and export contract.

## Concurrency conflicts

Direct `KnowledgeGraph` objects follow Python ownership and borrow rules. For
shared readers and writers, use `graph.session()`. A transaction commits with
optimistic concurrency control; a stale snapshot raises a typed `KgError`
instead of silently overwriting a newer commit. See {doc}`transactions` and
{doc}`/concepts/concurrency`.

## Other bindings

Rust code matches on `KgError` or the stable classifier `KgErrorCode`. The
classifier also supplies canonical HTTP and Neo4j/Bolt status codes; each
binding still owns its response shape and lifecycle. The C ABI exposes the
corresponding `KGLITE_STATUS_*` codes declared in the generated header. See
{doc}`/rust/c-abi` for ownership and status details.

`InternalError` represents a broken KGLite invariant. It is not a recoverable
user-input condition; report it with the complete message and a minimal
reproducer.

## See also

- {doc}`Python API reference <../autoapi/index>` — method-specific exceptions.
- {doc}`transactions` — rollback, optimistic commits, and shared sessions.
- {doc}`guides/cypher` — query deadlines, row caps, and diagnostics.
- {doc}`/rust/api-reference` — Rust error and execution-option boundary.
- {doc}`/rust/c-abi` — non-Rust binding status codes.
