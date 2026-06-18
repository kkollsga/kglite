# graphsuite — multi-library graph benchmark

A reproducible, extensible benchmark that runs **15 workload groups**
against many graph backends on a single synthetic knowledge graph, and
accumulates results in a datafile you can keep adding libraries and runs
to over time.

## What it compares

| key | backend | surface |
|---|---|---|
| `kglite-cypher` | kglite, in-memory | Cypher (`g.cypher`) |
| `kglite-fluent` | kglite, in-memory | fluent `select/where/traverse` |
| `kglite-mapped` | kglite, mmap-backed columnar | Cypher |
| `kglite-disk`   | kglite, fully disk-backed | Cypher |
| `kglite-bolt`   | kglite, in-memory, over the wire | Bolt protocol (neo4j driver) |
| `networkx`      | NetworkX (pure Python) | native API |
| `duckdb`        | DuckDB (relational/SQL) | SQL + recursive CTEs |
| `kuzu`          | Kùzu (embedded graph DB) | Cypher |
| `rustworkx`     | rustworkx (Rust graph algos) | native API |
| `igraph`        | python-igraph (C graph algos) | native API |
| `neo4j`         | Neo4j server | Bolt (opt-in, see below) |

The five kglite rows exercise large parts of kglite's surface: bulk
load, the Cypher planner/executor (filter, aggregation, variable-length
traversal, `shortestPath`, cyclic pattern match, mutations), the fluent
builder (`select/where/traverse/statistics`), all three storage modes,
and the Bolt server.

## The 15 groups

`build`, `node_scan`, `point_lookup`, `property_filter`,
`group_aggregation`, `one_hop`, `two_hop`, `three_hop`,
`filtered_traversal`, `deep_traversal` (DEPENDS_ON closure),
`shortest_path`, `pattern_match` (Person→Company→Project→Person
triangle), `degree_topk`, `connected_components`, `mutations`.

Run `python -m benchmarks.competitive.graphsuite.run --list` for the
one-line description of each.

## The dataset

`dataset.py` generates a heterogeneous org/social knowledge graph —
Person / Company / Project / Skill / City nodes; KNOWS / WORKS_AT /
CONTRIBUTES_TO / HAS_SKILL / OWNS / DEPENDS_ON / LOCATED_IN edges — with
scalar properties for filtering/aggregation, a dense KNOWS subgraph for
multi-hop traversal, and a DEPENDS_ON DAG for deep traversal. Scale
presets: `small` (~2.5k nodes), `medium` (~24k nodes, default), `large`
(~120k nodes). Generation is deterministic (seeded), and a frozen set of
**query parameters** (seed node ids, filter values, shortest-path pairs)
is shared by every backend so all of them run the *same* queries.

## Fairness notes

- **Same logical result per group.** Each group prints a sanity value (a
  count) next to its timing; these line up across backends, so a
  comparison reflects equal work. K-hop groups count *distinct nodes
  reachable within k hops* from the seed set — these agree to <1% across
  engines; the only delta is walk-vs-trail handling of paths that briefly
  return to a seed (kuzu/duckdb/fluent allow it, the trail engines don't),
  which shifts a handful of seed nodes in or out of the count. We keep
  each engine's idiomatic form rather than bolting on a `NOT IN $seeds`
  filter, because that filter would distort the *timing* (it hits the
  same `IN $list` planner cost noted below) far more than the <1% count
  nuance it would erase.
- **Idiomatic per backend.** Cypher engines use variable-length patterns
  and `shortestPath`; the algorithm libraries use BFS / `descendants` /
  `connected_components`; SQL uses recursive CTEs and joins. Each backend
  is written the way a competent user of *that* tool would write it.
- **Honest skips.** A backend skips a group it can't express well rather
  than faking it: kglite/kuzu/neo4j skip `connected_components` (no
  native WCC); DuckDB skips `shortest_path` + `connected_components`
  (impractical in pure SQL); rustworkx/igraph skip `pattern_match` (no
  relational surface); the fluent API skips `shortest_path`,
  `pattern_match`, `degree_topk`. Skips show as `skip` in the report.
- **Full-dataset build.** The property-graph stores load the entire
  dataset; the algorithm libraries load the subgraphs they operate on.
- **Deeper hops use smaller seed sets** (200 → 50 → 20) to keep
  variable-length expansion tractable; every backend uses the same seeds
  per group, so within-group comparisons stay fair.

### A kglite finding surfaced while building this — now fixed

While building this suite, `MATCH (p:Person)-[:KNOWS]-(f) WHERE p.id IN
$ids` measured **~240× slower** than the index-anchored
`UNWIND $ids AS sid MATCH (p:Person {id:sid})-[:KNOWS]-(f)` form: the
planner did not use the `id IN $param` predicate as a scan anchor, so it
expanded KNOWS for *all* persons and filtered afterwards.

This was **fixed in the planner** (`index_selection.rs`): `WHERE x.prop
IN $param` (an `InExpression` whose RHS resolves to a list) now pushes an
`IN` matcher into the MATCH pattern — anchoring on the id index when the
property is `id` — and rewrites the surviving WHERE to the O(1)
`InLiteralSet` form. The `WHERE p.id IN $ids` shape dropped from ~89 ms
to ~1.2 ms (1-hop) and ~266 ms to ~1.3 ms (2-hop) at the small scale,
matching the hand-anchored form. (Trigger query added to the differential
corpus as `id_in_param_anchored`.) The suite still uses the UNWIND form
as the idiomatic baseline; both are now fast.

## Running

```bash
source .venv/bin/activate          # or use .venv/bin/python directly
maturin develop --release          # kglite numbers must be a release build

python -m benchmarks.competitive.graphsuite.run                  # all libs, medium
python -m benchmarks.competitive.graphsuite.run --scale small    # quick
python -m benchmarks.competitive.graphsuite.run --libs kglite-cypher,kuzu,duckdb
python -m benchmarks.competitive.graphsuite.run --report-only    # re-render datafile
python -m benchmarks.competitive.graphsuite.run --list           # libs + groups
```

The `kglite-bolt` row needs the release bolt binary at
`target/release/kglite-bolt-server` (`make build-bolt-server`).

To include **Neo4j**, start a server and point the adapter at it:

```bash
export GRAPHSUITE_NEO4J_URI=bolt://localhost:7687
export GRAPHSUITE_NEO4J_USER=neo4j GRAPHSUITE_NEO4J_PASSWORD=yourpass
python -m benchmarks.competitive.graphsuite.run --libs neo4j
```

Without `GRAPHSUITE_NEO4J_URI` the Neo4j row reports unavailable and is
skipped (there is no Python-embedded Neo4j; the embedded API is JVM-only).

## Results datafile

`results.json` is append-only. Each invocation adds one **run per
library**, tagged with `library`, `version`, `run_date`, the dataset
`signature`, the machine, and per-group `{min_s, median_s, reps, sanity,
status}`. Re-run any time — to add a new library, refresh a library after
an upgrade, or record a new machine — and old runs are preserved.
`report.py` renders the most recent run per library for a given dataset
signature as the combined-time-per-group matrix.

## Methodology

Each group method bundles its operations; the reported number is the
**combined wall-time** of the whole group, taken as the **min** over a
few repeats (repeat count adapts to per-run cost so the suite stays
bounded — sub-0.4s groups get the full repeat count, multi-second groups
run once). `build` is measured once (twice for cheap builds, keeping the
min). For trustworthy kglite numbers, build the wheel with
`maturin develop --release`.

## Adding a library

1. Add `ad_<lib>.py` with a `class …(Adapter)` (see `base.py`): set
   `name`, implement `build()`, override the `g_*` group methods you can
   support, `raise Skip("reason")` for the rest, return a sanity value
   that matches the other backends.
2. Register it in `run.py`'s `REGISTRY`.
3. `python -m benchmarks.competitive.graphsuite.run --libs <lib>`.
