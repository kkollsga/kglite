# Multi-Label Nodes

Shipped in **0.10.5**. Every node has a primary type (immutable,
set at creation) plus an optional list of secondary labels added
via `SET n:Label` / `CREATE (n:A:B)` / `g.add_label(...)`.
`labels(n)` returns the full list, primary first.

```cypher
CREATE (n:Agent:LLM:Reviewer {id: 'agent-strict', model: 'sonnet'})

MATCH (a:Reviewer) RETURN a              -- any reviewer
MATCH (a:Agent:Reviewer) RETURN a        -- AND-intersect
MATCH (n) WHERE 'Reviewer' IN labels(n)  -- equivalent
```

The trigger condition this document was originally written for —
*"If a future consumer materializes that genuinely needs
multi-label, re-open this decision"* — was met by `kglite-docs`
2026-05-28 (agent role taxonomies, `:Chunk:NeedsOcr` status
labels, cross-type predicates).

## What landed

| Surface | Shape |
|---|---|
| Storage | `NodeData.extra_labels: Vec<InternedKey>` (`#[serde(default)]` for back-compat with 0.10.4 saves) |
| Index | `DirGraph.secondary_label_index: HashMap<InternedKey, Vec<NodeIndex>>` + `has_secondary_labels: bool` fast-skip |
| Choke-point API | `DirGraph::add_node_label` / `remove_node_label` / `node_labels` — both indexes always update together |
| `GraphRead` trait | `node_labels_of(idx) -> Vec<InternedKey>` |
| Cypher MATCH | `(n:A:B:C)` parser loops colons; executor AND-intersects |
| Cypher mutation | `SET n:Label`, `REMOVE n:Label`, `CREATE (n:A:B)`, MERGE; multi-colon syntax (`SET n:A:B`) parses as multiple items |
| Python pymethods | `g.add_label(node_type, ids, label)` / `g.remove_label(...)` for direct batch ops |
| `add_nodes` kwarg | `labels: list[str]` applies uniform secondary labels to every row in the batch |
| `labels(n)` | Returns `[primary, ...secondaries]` in insertion order (was single-element list since 0.9.52) |
| Save / load | Bincode-default field; existing 0.10.4 files load with empty `extra_labels`; round-trips preserve secondaries |
| Test surface | `tests/test_multi_label.py` covers CREATE, SET, REMOVE, MATCH AND-intersect, idempotence, primary-removal-error, save+load, the pymethods, and the `add_nodes(labels=...)` kwarg |

## Why a primary type

Each node still has a load-bearing primary type — set at creation,
immutable via label mutation. The primary type drives the
type-indexed columnar storage (one `ColumnStore` per primary
type), the `type_indices` mmap-friendly CSR, and the per-type
property schema. Secondary labels are a parallel index over those
same nodes, not a second columnar layout.

To retype a node, set the `type` property: `SET n.type =
'NewType'`. Removing the primary label via `REMOVE n:Primary`
errors deliberately.

## Performance

Single-label workloads pay zero overhead. The `has_secondary_labels`
flag (a single `bool` checked on every label-keyed read) gates
the secondary index scan; when no node uses secondary labels, every
read takes the original hot path. The Sodir, Wikidata, and
code-tree workloads — none of which use secondary labels — show
no perf regression vs 0.10.4 on the tracked benchmarks.

## Cheaper alternatives we considered (and still recommend for some cases)

The pre-0.10.5 design assumed users would model "X is a Human and
a Politician" via subtype edges. That pattern is still valid and
preferred for **hierarchical** classifications:

| Use case | Modelling |
|---|---|
| Wikidata "X is a Human and a Politician" | `(n)-[:INSTANCE_OF]->(:Q5)` — already how Wikidata encodes it. |
| Code-tree "Method is a Callable" | `(:Method)-[:KIND_OF]->(:Callable)` — explicit relationship. |
| Per-application provenance (which agent tagged it, when) | Reify the relationship as a `(:Tagging)` node (see the `Cypher` guide). |
| Truly multi-label (agent role, status enum, lifecycle stage) | Multi-label, now native. |

The choice is between "labels as classification tags" (multi-label,
new in 0.10.5) and "labels as type hierarchy" (subtype edges, still
preferred when the taxonomy is hierarchical or when label-specific
properties matter).

## History

This document was originally written to defer Track C until a
real consumer materialised — single-label was a deliberate design
choice, not a TODO. `kglite-docs` 2026-05-28 (agent role
taxonomies + status-as-label) was that trigger; the work shipped
in 0.10.5 across six commits (storage foundation → read-side
Cypher → mutation surface → `add_nodes` kwarg → docs → release).

The three "stepping stones" identified in the original deferral:

1. **`Value::List(Vec<Value>)`** — shipped 0.10.0 Phase A.
2. **Subtype-edge planner rewrite** — not built; the
   classifications-as-label use case kglite-docs actually needs
   doesn't require it.
3. **`GraphRead::node_types_of`** — landed as `node_labels_of` in
   the Phase 1 commit of this release.

Tests covering the index-consistency invariant the original
rationale called out live in `tests/test_multi_label.py` and the
inline `multi_label_tests` module in `crates/kglite/src/graph/dir_graph.rs`.
