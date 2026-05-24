# Multi-Label Nodes — Rationale for the Current Single-Label Design

KGLite gives every node exactly one label. `labels(n)` returns a
single-element list. `SET n:Label` / `REMOVE n:Label` are not
supported; the workaround documented in `graph_overview(cypher=True)`
is `SET n.type = 'NewType'` to change a node's type. This is a
deliberate design choice, not a limitation awaiting a fix.

This document captures *why*, what a green-light implementation would
cost, and the smaller helpers that lower the future cost without
committing to the full work. It is the canonical place to start if
the question "should KGLite support multi-label nodes?" gets opened
again.

## Verdict

**Defer.** The current value proposition (lightweight embedded engine
where in-memory perf wins, with disk modes as add-ons for large-graph
exploration) is at direct odds with the multi-label data model.
Columnar storage is keyed by node type; the type index is a binary
mmap-friendly CSR keyed by a single `InternedKey`. Multi-label support
isn't a feature addition — it's a redesign of the data layer.

No active workstream (sodir prospect graph, code-tree, legal data,
Wikidata) is blocked on this. Wikidata-style "X is a Human and a
Politician" is already encoded via `(n)-[:INSTANCE_OF]->(:Q5)` —
which is how Wikidata itself stores it. Code-tree "Method is a
Callable" is solvable with a `KIND_OF` edge plus a small planner
rewrite, well under the cost of multi-label storage.

If a future consumer materializes that genuinely needs multi-label,
re-open this decision. The intermediate helpers in the [Stepping
stones](#stepping-stones) section lower the future cost without
committing to the full implementation.

## Where the design has anticipated this

Nowhere. The codebase commits to single-label at every layer:

| Layer | File | Coupling |
|---|---|---|
| Node data | `src/graph/schema.rs:1240` | `NodeData.node_type: InternedKey` (singular). Four constructors all take a single type. |
| Type index | `src/graph/storage/disk/type_index.rs:20-27` | 24-byte directory entry per `type_key` (u64); one node ↔ one type. |
| Columnar storage | `src/graph/storage/column_store.rs` | One `ColumnStore` per node type; `PropertyStorage::Columnar { store, row_id }` is keyed by the node's single type. |
| Type-keyed property schemas | `src/graph/schema.rs:1685` | `add_node_schema(node_type, schema)` assumes 1:1 mapping. |
| Three storage backends | `src/graph/storage/{disk,mapped}/`, `src/graph/storage/mod.rs` | `GraphRead::node_type_of(idx) -> Option<InternedKey>` returns one key. |
| Cypher planner | `src/graph/languages/cypher/planner/index_selection.rs` | Cost model has no notion of partial type membership. |
| `labels()` builtin | `src/graph/languages/cypher/executor/scalar_functions.rs:334` | Emits a single-element JSON-string list. (Tightened in 0.9.52 + Track-C marker.) |
| `LabelCheck` predicate | `src/graph/languages/cypher/executor/where_clause.rs:LabelCheck` | Compares against the single primary type. |
| PyO3 surface | `src/graph/pyapi/kg_core.rs`, `kglite/__init__.pyi` | `node_type: str` (~50 method signatures). |
| MCP surface | `crates/kglite-mcp-server/src/tools.rs` | `has_node_type`, `has_property` are single-type predicates. |
| Save / load (v3) | `src/graph/storage/disk/graph_persist.rs` | No secondary-label section in the file metadata. |
| Introspection XML | `src/graph/introspection/topics.rs:1340` | Explicitly documents multi-label as **not supported**; lists `SET n.type = 'NewType'` as the workaround. |

`introspection/topics.rs:1340` is the most explicit signal — it's not a
TODO note, it's a design statement that gets shipped to MCP agents on
every `graph_overview(cypher=True)` call.

## What a green-light implementation would touch

Estimated scope, by layer:

| Layer | Files (representative) | Change shape |
|---|---|---|
| Node storage | `schema.rs` (4 constructors), all `NodeData::new_*` callers | Add `secondary_labels: SmallVec<[InternedKey; N]>` + `has_secondary: bool` graph-level fast-skip. |
| Type index | `storage/disk/type_index.rs`, `core/pattern_matching/pattern.rs`, `planner/index_selection.rs` | Add a secondary type index parallel to the primary. Bump the binary format version. |
| Columnar storage | `storage/column_store.rs`, `schema.rs::PropertyStorage` | **Keep primary as the columnar grouping key.** Secondary labels do not get their own columnar layout — they're purely an index over nodes that happen to also wear a label. No schema union; primary columnar layout unchanged. |
| Three storage backends | `storage/{disk,mapped}/`, `storage/mod.rs` (`GraphRead`) | New trait method `node_types_of(idx) -> &[InternedKey]`. Today's single-label shim returns a 1-element slice; multi-label returns primary + secondaries. |
| Cypher planner | `planner/index_selection.rs`, `planner/fusion.rs`, `planner/join_order.rs` | Multi-type cost model: `MATCH (n:Label)` consults the union of primary and secondary indexes. |
| Cypher executor | `scalar_functions.rs` (`labels`), `where_clause.rs` (`LabelCheck`) | Both swap to `Value::List` once that variant exists. |
| Value type | `src/datatypes/values.rs` | Add `Value::List(Vec<Value>)`. **This is the headline prerequisite** — every list-producing site (literal lists, `labels()`, `nodes(p)`, `relationships(p)`, `collect()`, list comprehensions, list slicing) round-trips through `Value::String` today. Migrating them all is a separate, independently-valuable effort. |
| PyO3 + MCP surface | `pyapi/kg_core.rs`, `kglite/__init__.pyi`, `kglite-mcp-server/src/tools.rs` | ~50 method signatures move from `node_type: str` to `node_type: str` (primary) + `secondary_labels: list[str]` or single combined list. |
| Save / load | `storage/disk/graph_persist.rs` | New section `node_secondary_labels.bin`. Versioned format bump; old files load with empty secondary set. |
| Tests | `test_storage_parity.py`, `test_phase{1..7}_parity.py`, `test_cypher.py` (~150 label assertions) | Every site that asserts `node_types == {Person: N}` becomes membership-aware. |
| Introspection | `topics.rs:1340` | Flip the limitation note. |

Realistic budget: 2-3 weeks of focused work, including the test
migration. **Effort: XL.** **Risk: High** — the most likely
post-implementation bug class is a mutation site that updates the
primary index but forgets the secondary index. Mitigation:
single-choke-point API for `add_label` / `remove_label` that owns
both updates.

## Stepping stones

Three additive things that are independently valuable and lower
the cost of multi-label if it ever lands. Doing any of them does
not commit to Track C.

### 1. `Value::List(Vec<Value>)` — the single biggest prerequisite

Today every list value round-trips through `Value::String` (a
JSON-encoded string). `labels()`, `nodes(p)`, `collect()`, list
literals, list comprehensions all do this; consumers like
`parse_list_value` (`executor/helpers.rs`) JSON-parse on the way
in. The encoding is fragile (the hand-rolled escape in `labels()`
was a real bug, fixed in 0.9.52) and prevents native list operations.

Adding `Value::List` and migrating list producers + consumers is
a 1-day effort that:

- removes the JSON round-trip from list operations entirely;
- separates the encoding fix from the data-model fix (the
  fingerprint-1 outcome of multi-label work);
- closes the brittleness without locking in any data-model decision.

Swap points are already annotated in 0.9.52:
`scalar_functions.rs::"labels"` and `helpers.rs::parse_list_value`.

### 2. Subtype-edge planner rewrite

Most of the Wikidata "X is a Human and a Politician" ergonomic
can be served without multi-label storage. Add an opt-in subtype
hierarchy (a `:SUBTYPE_OF` edge connecting `:Politician` to
`:Human`) and a planner rewrite that lowers `MATCH (n:Label)` to
`(:Subtype)-[:INSTANCE_OF*0..]->(:Label)` for nodes that opt in.

This buys the ergonomic without redesigning storage. It's an
additive planner pass that fits the existing `PASSES &[...]` in
`planner/mod.rs`. Estimated 2-3 days.

### 3. `GraphRead::node_types_of(idx) -> &[InternedKey]`

A trait method that today returns a 1-element slice (the primary
type wrapped in a slice). Lets every consumer migrate from the
singular `node_type_of(idx) -> Option<InternedKey>` shape without
churn when (if) multi-label lands. A no-op refactor on its own,
but it's the API surface every Track-C consumer would have to
touch — landing it now spreads the migration cost across many
small PRs instead of one XL one.

Estimated half a day.

## Cheaper alternative for the known motivating use cases

| Use case | Cheap encoding |
|---|---|
| Wikidata "X is a Human and a Politician" | `(n)-[:INSTANCE_OF]->(:Q5)` — already in the data. The planner rewrite (stepping stone #2) closes the ergonomic gap. |
| Code-tree "Method is also a Callable" | `(:Method)-[:KIND_OF]->(:Callable)` plus the same planner rewrite. |
| Imported graphs assuming multi-label | Not a known consumer. If one appears, that's the trigger for Track C. |

## Test invariants any Track C implementation must hit

If multi-label is eventually built, these are the test points that
catch the index-consistency bug class:

- **Parity oracles** (`test_storage_parity.py`, `test_phase{1..7}_parity.py`)
  agree across all three backends on `node_types_of(idx)` for nodes
  with multiple labels.
- **Label mutation idempotence**: `SET n:Label; REMOVE n:Label;
  SET n:Label` ends in the same index state as `SET n:Label` once.
- **Columnar layout**: a multi-label node's properties stay routed
  by its primary type's columnar store. The secondary index entry
  is purely metadata — no columnar duplication.
- **Save / load round-trip**: an old `.kgl` v3 file loads with an
  empty secondary set per node; a new file written under multi-label
  loads identically.
- **Introspection XML**: the `topics.rs:1340` limitation note is
  removed (or inverted) and `graph_overview(cypher=True)` reflects
  the new capability.

## Pointers

- `src/graph/introspection/topics.rs:1340` — the current limitation
  note; should link back to this document via a doc-attribute or
  comment when Track C is opened for discussion.
- 0.9.52 commit history — Phase 2 (`labels()` hygiene) added the
  swap-point markers at the two call sites a Track-C implementation
  would touch first.
- The investigation that produced this document is summarised in
  the commit `docs: multi-label rationale (Track C deferred)`.
