# Error handling — typed exceptions

> Companion to [`bolt_implementation.md`](https://github.com/kkollsga/kglite/blob/main/bolt_implementation.md)
> Phase A.2. Reference for Python consumers catching kglite-raised
> exceptions, and for binding implementers (Bolt server in Phase C.6,
> future Arrow / language-binding work) mapping typed errors to
> wire-protocol shapes.

This page documents the `kglite.KgError` exception hierarchy: what
each class means, what raises it, how to catch it, and the trade-off
with the pre-A.2 built-in-exception surface.

## The hierarchy

Every kglite-raised exception is a subclass of `kglite.KgError`. The
class chain mirrors the Rust `KgError` enum at `src/error.rs`:

```text
Exception
└── kglite.KgError                          (universal base)
    ├── kglite.CypherError                   (Cypher pipeline base)
    │   ├── kglite.CypherSyntaxError         — parser/tokenizer rejection
    │   ├── kglite.CypherTimeoutError        — exceeded timeout_ms
    │   ├── kglite.CypherExecutionError      — executor failure
    │   └── kglite.CypherTypeMismatchError   — type mismatch in expression
    ├── kglite.SchemaError                   — pattern-literal schema check
    ├── kglite.ValidationError                — structural validation
    ├── kglite.ExprError                      — blueprint expression
    ├── kglite.NodeNotFoundError              — node lookup miss
    ├── kglite.ConnectionNotFoundError        — edge-type lookup miss
    ├── kglite.PropertyNotFoundError          — property lookup miss
    ├── kglite.FileError                      — file not found
    ├── kglite.FileFormatError                — malformed .kgl / blueprint
    ├── kglite.FileIoError                    — permission, mid-read EOF
    ├── kglite.ArgumentError                  — bad arg precondition
    ├── kglite.MissingArgumentError           — required arg not passed
    ├── kglite.InternerCollisionError         — persisted name-key collision; write rejected
    └── kglite.InternalError                  — invariant violation (bug)
```

### The one exception that is *not* a `KgError`: Ctrl-C

Interrupting a long-running read with `Ctrl-C` raises the **builtin
`KeyboardInterrupt`**, not a `kglite.KgError` subclass — by design, an
interrupt is a user action, not a query fault. (Internally the engine
raises `KgError::Cancelled`; the Python boundary maps it to
`KeyboardInterrupt`.) So `except kglite.KgError` will **not** swallow a
Ctrl-C — catch it separately if you need to:

```python
try:
    g.cypher(long_running_read, timeout_ms=0)
except KeyboardInterrupt:
    print("interrupted by user")   # graph is unchanged
except kglite.CypherError as e:
    print(f"query failed: {e}")
```

A deadline timeout is different — that *is* a query fault and surfaces as
`kglite.CypherTimeoutError` (a `KgError`). See the Cypher guide's
"Interrupting a query" section for the behaviour and platform notes.

## How to catch

**Specific:** when you care which kind of error fired:

```python
import kglite

try:
    g.cypher("MATCH x RETURN y INVALID")
except kglite.CypherSyntaxError as e:
    print(f"parse error: {e}")  # message includes "line N, col M: ..."
```

**Cypher-family base:** catch any Cypher-related error:

```python
try:
    g.cypher("MATCH (n:Person) WHERE n.age > $missing RETURN n")
except kglite.CypherError:
    # catches CypherSyntaxError, CypherTimeoutError,
    # CypherExecutionError, CypherTypeMismatchError
    ...
```

**Universal:** catch any kglite-raised error regardless of category:

```python
try:
    g.cypher(query)
    g.add_nodes(df, ...)
    g.save("graph.kgl")
except kglite.KgError as e:
    log.error("kglite failed: %s", e)
```

`kglite.KgError` is `Exception`-derived, so a bare `except Exception:`
still works as the last-resort net.

## Load failures: corrupt vs missing (disposable-cache branch)

`kglite.load(path)` and `kglite.from_bytes(data)` raise **typed, classifiable**
errors (0.11.0): `kglite.FileFormatError` on a corrupt / truncated / wrong-format
input, and `kglite.FileError` on a missing file. A consumer treating the `.kgl`
as a rebuildable cache can branch cleanly:

```python
try:
    g = kglite.load("cache.kgl")
except kglite.FileError:
    g = build_from_source()          # missing → build fresh
except kglite.FileFormatError:
    g = build_from_source()          # corrupt/old format → rebuild, don't trust it
```

If the original source may not be available at recovery time, keep a
format-stable backup so `FileFormatError` is always survivable — see
[Back up before upgrading](guides/import-export.md#back-up-before-upgrading-the-format-stable-escape-hatch)
(`export_csv()` → `from_blueprint()`).

## Sharing a graph across threads

A `KnowledgeGraph` is single-owner. If one thread mutates it (`add_nodes`,
`embed_texts`, a `CREATE`/`SET`/`DELETE` query, `save`, …) while another touches
the same object, the second call raises a clear `RuntimeError` — never a panic
or silent corruption. Give each worker its own `g.copy()`, share a read-only
`g.freeze()` snapshot for concurrent reads, or — for shared reads **and** writes
— use `g.session()` (see {doc}`/concepts/concurrency`).

## Migration from pre-A.2

Before A.2, kglite raised the Python built-in exceptions directly:
`ValueError`, `RuntimeError`, `KeyError`, `IOError`, `TypeError`. A.2
replaces these with typed subclasses descending from `KgError`. The
trade-off: **existing `except ValueError:` / `except RuntimeError:` /
`except KeyError:` catches no longer match** kglite-raised
exceptions.

PyO3's `create_exception!` macro is single-inheritance — combining
`KgError` as a base AND `ValueError` as an additional base would
require Python-level multiple-inheritance which PyO3 doesn't support
cleanly. The plan chose consistency-first: every kglite error is
`isinstance(e, kglite.KgError)`.

**Migration mapping** (most common cases):

| Pre-A.2 | A.2+ |
|---|---|
| `except ValueError:` | `except kglite.KgError:` or specific subclass |
| `except RuntimeError:` | `except kglite.CypherExecutionError:` (or `KgError`) |
| `except KeyError:` | `except kglite.NodeNotFoundError:` / `PropertyNotFoundError:` |
| `except FileNotFoundError:` | `except kglite.FileError:` |
| `except TypeError:` | `except kglite.ArgumentError:` |

For the "I don't care which kglite error, just any error from kglite"
case, `except kglite.KgError:` is the new canonical form.

## What's in the message

Every typed exception's `str(e)` includes the most actionable
diagnostic information for that error class:

| Class | Message includes |
|---|---|
| `CypherSyntaxError` | `at line N, col M:` prefix, the bad token, a caret excerpt of the offending source line |
| `CypherTimeoutError` | `elapsed_ms`, `limit_ms` |
| `CypherExecutionError` | optional `(line, col)`, the operator/function name when known |
| `SchemaError` | the unknown property name, the type that didn't have it, "did you mean?" hints when close |
| `ValidationError` | per-variant context (missing field, expected vs actual type, etc.) |
| `NodeNotFoundError` | `node_type`, `id` |
| `ArgumentError` | the violating argument, what was expected, what was found |

## For binding implementers

If you're writing a binding that consumes Rust-side `Result<T, KgError>`
from `kglite::api::cypher::*` — the Bolt server (`crates/kglite-bolt-server`,
Phase C.6), a future Arrow exporter, a JNI bridge — your error
mapping layer takes `KgError` and produces the consumer-specific
shape.

The `KgErrorCode` enum (at `src/error.rs`) gives you a `Copy + Eq +
Hash` classifier suitable for match-dispatch tables. For Bolt's
FAILURE-code mapping (Phase C.6), the table looks like:

```rust
use kglite::error::KgErrorCode;

fn neo4j_code(code: KgErrorCode) -> &'static str {
    match code {
        KgErrorCode::CypherSyntax => "Neo.ClientError.Statement.SyntaxError",
        KgErrorCode::CypherTimeout => "Neo.ClientError.Transaction.TransactionTimedOut",
        KgErrorCode::CypherExecution => "Neo.ClientError.Statement.ExecutionFailed",
        KgErrorCode::CypherTypeMismatch => "Neo.ClientError.Statement.TypeError",
        KgErrorCode::Schema => "Neo.ClientError.Schema.SchemaRuleAccessFailed",
        KgErrorCode::Validation => "Neo.ClientError.Schema.ConstraintValidationFailed",
        KgErrorCode::NodeNotFound => "Neo.ClientError.Statement.EntityNotFound",
        KgErrorCode::FileNotFound | KgErrorCode::FileFormat | KgErrorCode::FileIo
            => "Neo.DatabaseError.General.UnknownError",
        KgErrorCode::InvalidArgument | KgErrorCode::MissingArgument
            => "Neo.ClientError.Statement.ArgumentError",
        // ... etc.
    }
}
```

The `KgError::position()` accessor returns `Option<(line, col)>` for
errors that have source position info (currently `CypherSyntax` and
optionally `CypherExecution`). Bolt's FAILURE message includes this
as the `position` field when present.

## Internal errors

`kglite.InternalError` is the only class that should never appear in
end-user code paths. It's reserved for invariant violations — places
where an upstream check guaranteed something, the invariant broke,
and the code chose `return Err(Internal { ... })` rather than
`unwrap()` panic. Phase A.2 / C4 (the A.4 fold-in) replaced ~11
executor unwraps with `.expect("invariant: ...")` panics rather than
typed errors when the invariant was genuinely upheld; sites where the
invariant might plausibly fail under malformed input got typed
`KgError::Internal { message, location }` returns.

If you see `kglite.InternalError` in production, file a bug — the
`location` field names the source site so we can find it fast.

## See also

- `src/error.rs` — the Rust `KgError` enum + `KgErrorCode` definitions.
- `src/error_py.rs` — `create_exception!` declarations + the
  canonical `From<KgError> for PyErr` boundary conversion.
- `kglite/__init__.pyi` — Python stub declarations matching the
  hierarchy here.
- `tests/test_error_types.py` — canonical pinning suite (54 tests
  covering hierarchy, cross-mode behaviour, diagnostic quality).
- `bolt_implementation.md` Phase C.6 — Bolt FAILURE-code mapping
  (consumes the table shape sketched above).
- `docs/python/value-projection.md` — Phase A.1 companion;
  shape-and-value answer to A.2's error-and-type answer.
