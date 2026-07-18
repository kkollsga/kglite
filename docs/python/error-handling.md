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
    ├── kglite.NodeNotFoundError
    ├── kglite.ConnectionNotFoundError
    ├── kglite.PropertyNotFoundError
    ├── kglite.FileError
    ├── kglite.FileFormatError
    ├── kglite.FileIoError
    ├── kglite.ArgumentError
    ├── kglite.MissingArgumentError
    ├── kglite.InternerCollisionError
    └── kglite.InternalError
```

`CypherSyntaxError` always has `.line` and `.col` attributes (either may be
`None`). `CypherExecutionError` has them when the executor can identify the
source position. Timeout messages report the elapsed and configured limit.

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
