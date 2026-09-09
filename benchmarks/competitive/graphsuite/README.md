# graphsuite — multi-library graph benchmark

A reproducible, extensible benchmark that runs **26 workload groups**
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
| `ladybug`       | LadybugDB (embedded graph DB) | Cypher |
| `rustworkx`     | rustworkx (Rust graph algos) | native API |
| `igraph`        | python-igraph (C graph algos) | native API |
| `neo4j`         | Neo4j server | Bolt (opt-in, see below) |

The five kglite rows exercise large parts of kglite's surface: bulk
load, the Cypher planner/executor (filter, aggregation, variable-length
traversal, `shortestPath`, cyclic pattern match, mutations), the fluent
surface (`select/where/traverse/statistics`, plus `shortest_path`,
`match_pattern`, `degree_centrality` and `vector_search`), all three
storage modes, and the Bolt server.

## The 26 groups

`build`, `node_scan`, `point_lookup`, `edge_scan`, `property_filter`,
`range_filter`, `group_aggregation`, `year_aggregation`, `one_hop`,
`two_hop`, `three_hop`, `filtered_traversal`, `deep_traversal`
(DEPENDS_ON closure), `score_filtered_traversal`, `shortest_path`,
`pattern_match` (Person→Company→Project→Person triangle),
`industry_aggregation`, `two_step_join`, `degree_topk`,
`connected_components`, `louvain`, `degree_filter`, `vector_knn`,
`geo_within`, `bulk_update`, `mutations`.

`base.py`'s `GROUPS` list is the single source of truth for the set and
its order.

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
is shared by every adapter. Each adapter expresses the corresponding workload
through its own data model and idiomatic API; query text and some semantics
therefore differ.

## Fairness notes

- **Results are visible, not presumed equal.** Each group prints a sanity
  value and digest next to its timing. `--verify` lists every cross-adapter
  difference. K-hop groups aim to count *distinct nodes reachable within k
  hops* from the seed set, but engines differ in how a path may return to
  its own seed. A walk-semantics engine (LadybugDB/DuckDB/fluent) may step back
  along the edge it arrived on, so every seed with a neighbour re-enters the
  count at two hops. A trail engine (kglite's Cypher path, neo4j) may not
  reuse a relationship, so a seed re-enters only when the return goes round
  a cycle on different edges — which shifts a handful of seed nodes in or
  out of the count. Other recorded differences must be reviewed before a
  capture is described as like-for-like. We keep each engine's idiomatic form rather than
  bolting on a `NOT IN $seeds` filter, because that filter would distort
  the *timing* (it hits the same `IN $list` planner cost noted below) far
  more than the <1% count nuance it would erase.
- **Idiomatic per backend.** Cypher engines use variable-length patterns
  and `shortestPath`; the algorithm libraries use BFS / `descendants` /
  `connected_components`; SQL uses recursive CTEs and joins. Each backend
  is written the way a competent user of *that* tool would write it.
- **Missing means not exercised.** An adapter skips a group when this harness
  profile has no maintained implementation for it. That is not a claim that
  the underlying product lacks the feature. The fluent kglite column skips
  exactly four — `edge_scan`
  (no scan primitive; `label_pair_counts()` is a cached cardinality
  snapshot, so timing it would be a fake win), `two_step_join`
  (`traverse()` returns a node set, not path rows), and
  `connected_components` / `louvain` (the Python binding takes no
  node-type scope, so it would answer a different question about a
  different universe). Skips show as `skip` in the report. A skip is a
  claim about the surface and is re-derived, not inherited: the four
  above are what survived the 2026-08-25 re-derivation of twelve.
- **The fluent `mutations` cell measures less work than the others.**
  kglite's Python surface has no node/edge delete outside Cypher, so the
  fluent Mutations cell runs create + connect + update where every Cypher
  column also runs a `DETACH DELETE`, and it reports the created count
  rather than the deleted one. Read that one cell as a *substitution*,
  not a like-for-like time. `mutations` is excluded from the
  cross-backend result-parity check for the same reason.
- **Construction is not cross-kind ranked.** Property-graph stores load the
  full dataset while some algorithm adapters build only the subgraphs they
  operate on. The public headline therefore omits construction time.
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
uv run --no-sync maturin develop --release
make build-bolt-server

.venv/bin/python -m benchmarks.competitive.graphsuite.run                  # default adapters, medium
.venv/bin/python -m benchmarks.competitive.graphsuite.run --scale small    # exploratory quick run
.venv/bin/python -m benchmarks.competitive.graphsuite.run --libs kglite-cypher,ladybug,duckdb
.venv/bin/python -m benchmarks.competitive.graphsuite.run --report-only
.venv/bin/python -m benchmarks.competitive.graphsuite.run --list
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
`report.py` renders the most recent raw run per library for investigation.
The public generator additionally requires one clean capture id containing an
error-free row for every requested adapter. An incomplete invocation remains
raw history but cannot silently borrow old rows for publication.

## Methodology

Each group method bundles its operations; the reported number is the
**combined wall-time** of the whole group, taken as the **min** over a
few repeats (repeat count adapts to per-run cost so the suite stays
bounded — sub-0.4s groups get the full repeat count, multi-second groups
run once). `build` is measured once (twice for cheap builds, keeping the
minimum). There is no warm-up. The public report labels this protocol
directly; values above 10 seconds are retained and labelled rather than called
timeouts. Kglite measurements require the release build shown above.

## Adding a library

1. Add `ad_<lib>.py` with a `class …(Adapter)` (see `base.py`): set
   `name`, implement `build()`, override the `g_*` group methods you can
   support, `raise Skip("reason")` for the rest, return a sanity value
   that matches the other backends.
2. Register it in `run.py`'s `REGISTRY`.
3. `python -m benchmarks.competitive.graphsuite.run --libs <lib>`.
