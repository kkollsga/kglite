# Value projection — how RETURN materialises

> Reference for reviewers of the executor/projection code and implementers of
> bindings or structured-result adapters.

This page documents what happens between a Cypher `RETURN` clause and
a Python dict in your hand. It explains the `Value` enum, the
distinction between transient and materialised graph values, the
serialised `.kgl` format, and the contract that bindings consume.

## The `Value` enum

Defined at `crates/kglite/src/datatypes/values.rs`. Sixteen variants today:

| Variant | What it carries | Bolt PackStream analogue |
|---|---|---|
| `Null` | nothing | NULL (`0xC0`) |
| `Boolean(bool)` | true / false | BOOLEAN (`0xC2`/`0xC3`) |
| `Int64(i64)` | signed 64-bit integer | INT (sized) |
| `Float64(f64)` | IEEE 754 double | FLOAT (`0xC1`) |
| `UniqueId(u32)` | node-id-style integer | INT (cast to i64) |
| `String(String)` | UTF-8 | STRING (sized) |
| `DateTime(NaiveDate)` | calendar date | Struct `0x44` (Date) |
| `Timestamp(NaiveDateTime)` | date + time-of-day, second precision | local date-time structure |
| `Point { lat, lon }` | 2D geographical point | Struct `0x58` (Point2D, srid=4326) |
| `Duration { months, days, seconds }` | Neo4j-shape calendar duration | Struct `0x45` (Duration) |
| **`NodeRef(u32)`** | **transient internal handle (see below)** | — never crosses the boundary |
| **`Node(Box<NodeValue>)`** | materialised node `(id, labels, properties)` | Struct `0x4E` (Node) |
| **`Relationship(Box<RelValue>)`** | materialised rel `(id, start, end, type, properties)` | Struct `0x52` (Relationship) |
| **`Path(Box<PathValue>)`** | materialised path `{nodes, relationships}` | Struct `0x50` (Path) |
| **`List(Vec<Value>)`** | ordered heterogeneous list | LIST (sized) |
| **`Map(BTreeMap<String, Value>)`** | string-keyed map (deterministic order) | MAP (sized) |

Persistence uses the explicitly versioned RGF v5/Postcard container. The
current reader accepts v5/Postcard and rejects v4/bincode and older containers
with a clear migration/rebuild error.

## `NodeRef` vs `Node` — transient vs materialised

`Value::NodeRef(u32)` and `Value::Node(Box<NodeValue>)` look similar
but serve different roles in the executor:

- **`NodeRef(idx)`** — a transient internal handle carrying the
  petgraph `NodeIndex`. Used by intermediate stages (`WITH`,
  `UNWIND`, `collect()` inputs) to preserve node identity without
  cloning property data. Never user-visible; never persisted.
- **`Node(Box<NodeValue>)`** — a materialised graph value with the
  full `(id, labels, properties)` triple. Built at *projection
  time* when the executor needs to hand a node value to a consumer:
  `RETURN n`, `collect(n)`, `nodes(p)`, the `WITH n` chain that
  carries n forward, etc.

The transition happens in `evaluate_expression(Expression::Variable)`
at `crates/kglite/src/graph/languages/cypher/executor/expression.rs`:

```rust
if let Some(&idx) = row.node_bindings.get(name) {
    if let Some(node_value) = materialize_node_value(idx, self.graph) {
        return Ok(Value::Node(Box::new(node_value)));
    }
    // Tombstone path — DELETE-then-RETURN-in-same-query
    return Ok(Value::Node(Box::new(NodeValue { id: idx.index() as u32, labels: vec![], properties: BTreeMap::new() })));
}
```

The tombstone arm preserves Cypher's "count-of-matched-rows" semantics
across `MATCH ... DELETE n RETURN count(n)` — the binding survives
deletion but materialising the node would return `None`; the tombstone
keeps `count(n)` non-Null without faking data.

## The projection flow

End-to-end for `MATCH (n:Person {id: 'alice'}) RETURN n`:

```
parser           AST: RETURN Variable("n")
  │
  ▼
planner          MATCH binds n → NodeIndex 42 in result_row.node_bindings
                 RETURN n is a per-row projection expression
  │
  ▼
executor         per row:
  │                evaluate_expression(Variable("n"), row)
  │                  → row.node_bindings.get("n") = Some(42)
  │                  → materialize_node_value(42, graph)
  │                      = NodeValue { id: 42, labels: ["Person"],
  │                                    properties: BTreeMap {...} }
  │                  → Value::Node(Box::new(node_value))
  │                projected.insert("n", Value::Node(...))
  │
  ▼
CypherResult     rows: [[ Value::Node(...) ]]
  │
  ▼
Python boundary  preprocess_values_owned wraps each Value in
                 PreProcessedValue::Plain
  │
  ▼
py_out::value_to_py(py, &Value::Node(node_val)) →
   PyDict { "id": 42, "labels": ["Person"], "properties": {...} }
```

The `materialize_node_value` helper (`executor/helpers.rs`) is the
canonical entry point. It's **backend-aware**: in memory mode it reads
properties via `NodeData::property_iter`; in mapped / disk modes
properties live in the column store, so it walks `graph
.get_node_type_metadata(node_type)` and reads each property via
`resolve_node_property` (which knows the column-aware path). The
parametrised tests in `tests/test_value_variants.py` run every
projection assertion 3× (memory / mapped / disk) to keep the
behaviours in lockstep.

## At the Python boundary

`crates/kglite-py/src/datatypes/py_out.rs::value_to_py` recursively converts the
sixteen `Value` variants to Python objects. The shapes consumers
see:

```python
>>> g.cypher("MATCH (n:Person) RETURN n LIMIT 1").to_list()
[{"n": {"id": 42, "labels": ["Person"], "properties": {"id": "alice", "title": "Alice", "type": "Person", "age": 30, "city": "Oslo"}}}]

>>> g.cypher("MATCH ()-[r:KNOWS]->() RETURN r LIMIT 1").to_list()
[{"r": {"id": 0, "start": 42, "end": 43, "type": "KNOWS", "properties": {"since": 2015}}}]

>>> g.cypher("MATCH p = shortestPath((a:Person)-[*]->(b:Person)) RETURN p LIMIT 1").to_list()
[{"p": {"nodes": [...], "relationships": [...]}}]

>>> g.cypher("MATCH (n:Person) RETURN labels(n) AS L LIMIT 1").to_list()
[{"L": ["Person"]}]                  # native list, not '["Person"]'

>>> g.cypher("MATCH (n:Person) RETURN properties(n) AS P LIMIT 1").to_list()
[{"P": {"id": "alice", "title": "Alice", "age": 30, "city": "Oslo"}}]
```

`RETURN n` and `labels(n)` expose native structured values. Bindings should not
infer types by parsing JSON-looking strings.

The lazy `ResultView` path is preserved: when the planner flags a
terminal `RETURN` as `lazy_eligible`, per-cell materialisation runs
on Python access (cached via `Mutex<Vec<Option<Vec<PreProcessedValue>>>>`).
Node projections materialise one `Box<NodeValue>` per accessed cell; release
performance baselines cover this path.

## In `.kgl` files

The current `.kgl` format is an RGF v5 binary container:

```
[0..4]    Magic: b"RGF\x05"
[4]       codec tag: Postcard
[...]     core_data_version (currently 3)
[8..12]   metadata_length: u32 LE
[12..N]   JSON metadata (column schemas, section sizes, all config)
[section] topology.zst — graph structure without node properties
[section] columns_<Type>.zst — packed property columns per type
[section] embeddings.zst (optional)
[section] timeseries.zst (optional)
[section] secondary labels / vector-index metadata (optional)
```

`Value` serialises via `serde` with a discriminant tagged by
variant position. The order in `crates/kglite/src/datatypes/values.rs` is
intentionally stable for the first 9 variants (Null=0 .. Duration=8)
so future enum changes append at the end (Timestamp is discriminant 15).
The container and codec tags make compatibility explicit: the current reader
selects v5/Postcard by header and refuses v4/bincode or older containers rather
than guessing. Kglite 0.13.4 is the conversion bridge for pre-0.14 artifacts.

The `tests/test_phase4_parity.py::test_kgl_v3_golden_hash` byte-level
tripwire fires on any drift in the saved layout; the
`test_kgl_v3_file_rejected_with_clear_error` test pins the
hard-break error message.

## For binding implementers

If you're writing a binding that consumes `CypherResult` from Rust —
the Bolt server (`crates/kglite-bolt-server`), an Arrow
exporter, a Polars adapter, a JNI bridge, anything that reads
`Vec<Vec<Value>>` and produces a downstream shape — your value-mapping
layer is responsible for all 16 variants.

The reference table at the top of this page maps each variant to its
Bolt PackStream analogue. For other targets:

- **Arrow / Polars**: scalars map to their typed columns directly.
  `List` → a `LargeList` column; `Map` → a `Struct` column when the
  key set is fixed at the column level, otherwise a `Map` column.
  `Node` / `Relationship` / `Path` → either a `Struct` column or a
  JSON-string fallback (the agent ecosystem expects dict-shaped output;
  the structured form is preferred where the target supports it).
- **C ABI** (`kglite-c`, shipped 0.10.3): Cypher result rows
  serialize as JSON-string blobs via
  `kglite_cypher_result_rows_json`. Each binding parses the JSON
  with its language's stdlib (Go's `encoding/json`, JS's
  `JSON.parse`, etc.) — same row-shape rules apply. A future v2
  may add a tagged-union accessor for performance-critical
  row-by-row consumption; the JSON-at-boundary path is fine for
  the common-case query sizes. See
  [`docs/rust/c-abi.md`](../rust/c-abi.md).

The `Value::type_name() -> &'static str` method (at
`crates/kglite/src/datatypes/values.rs`) returns the canonical PascalCase variant
name — useful for binding-side dispatch tables. The `impl Display`
gives a debug-shaped string suitable for log lines and error
messages; for wire serialisation, use the per-binding mapping
explicitly.

## What this is NOT

- **Not the public API surface.** That's `kglite::api::*`
  (`crates/kglite/src/lib.rs`).
- **Not a serialisation spec for the `.kgl` format.** The versioned binary
  container and zstd layout details live in
  `crates/kglite/src/graph/io/file.rs`; this doc just names the format-version
  boundaries.
- **Not a guide to writing new scalar functions.** That's covered
  inline in `crates/kglite/src/graph/languages/cypher/executor/scalar_functions/`
  — search for the pattern of an existing function and mirror it.

## See also

- `docs/concepts/multi-label-rationale.md` — multi-label nodes
  shipped in 0.10.5. `labels()` now returns `[primary, ...secondaries]`.
- `docs/concepts/design-decisions.md` — the "why" behind the
  embedded design, the primary-type label model, and the LLM-agent surface.
- `tests/test_value_variants.py` — the canonical pinning suite for
  every shape this doc describes. When in doubt, search this file.
