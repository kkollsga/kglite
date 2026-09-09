# KGLite benchmarks

> **Unqualified historical snapshot.** These rows predate publication qualification. They may combine different invocations or dirty source revisions and must not be described as a current capture.

Minimum observed wall time for the workloads each adapter exercised on one seed-deterministic synthetic graph. Lower is better within a row; `not exercised` is not a claim about the underlying product's capabilities.

**Dataset:** 25,333 nodes · 280,774 edges (Person/Company/Project/Skill/City), scale `medium`.

| Workload topic | kglite | DuckDB | igraph | Kùzu (historical) | Neo4j (native) | NetworkX | rustworkx |
|---|---|---|---|---|---|---|---|
| Node/edge scans and ID lookup | 2.4ms | 1.7ms | **14µs** | 4.5ms | 4.7ms | 2.5ms | 241µs |
| Property filters and aggregation | 3.7ms | **1.4ms** | 4.2ms | 3.4ms | 64.2ms | 5.4ms | 3.6ms |
| Reachability and traversal | 16.8ms | 12.9ms | **2.9ms** | 3.51s (5/6 exercised) | 272.0ms | 8.0ms | 8.5ms |
| Shortest-path queries | **261µs** | not exercised | 22.5ms | 38.3ms | 12.3ms | 482µs | 88.1ms |
| Typed graph joins and aggregation | 7.3ms | **1.9ms** | not exercised | 3.4ms | 25.8ms | 117.5ms | not exercised |
| Degree and connected-components operations | 23.0ms | 3.3ms (2/3 exercised) | **2.0ms** | 10.3ms (2/3 exercised) | 39.4ms (2/3 exercised) | 9.1ms | 8.6ms |
| Louvain community detection | **133.9ms** | not exercised | 1.41s | not exercised | not exercised | 11.17s (measured >10s) | not exercised |
| Updates and create/delete batch | 4.3ms | 2.4ms | 6.8ms | 143.6ms | 63.6ms | 673µs | **149µs** |
| Exact vector scoring | **4.2ms** | not exercised | not exercised | not exercised | not exercised | not exercised | not exercised |
| Latitude/longitude bounding-box filter | **17µs** | 73µs | not exercised | 252µs | 647µs | not exercised | not exercised |

Bold marks the fastest complete, non-slow measurement in that row. Partial cells sum only the named executed groups; missing work is not estimated. Values above 10 seconds remain in the measurement and are labelled, not treated as timeouts. There is deliberately no grand total across unequal workload coverage.

Adapters receive the same generated input records and shared query parameters, but use different data models and idiomatic APIs. Their operations are corresponding workloads, not necessarily identical query text or identical execution semantics. The detailed parity report records result differences instead of treating every cross-library difference as a failure.

### Workloads exercised by these adapters

This is coverage of the recorded adapter implementations, not a product capability matrix.

| Workload topic | kglite | DuckDB | igraph | Kùzu (historical) | Neo4j (native) | NetworkX | rustworkx |
|---|---|---|---|---|---|---|---|
| Node/edge scans and ID lookup | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Property filters and aggregation | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Reachability and traversal | ✓ | ✓ | ✓ | 5/6 exercised | ✓ | ✓ | ✓ |
| Shortest-path queries | ✓ | not exercised | ✓ | ✓ | ✓ | ✓ | ✓ |
| Typed graph joins and aggregation | ✓ | ✓ | not exercised | ✓ | ✓ | ✓ | not exercised |
| Degree and connected-components operations | ✓ | 2/3 exercised | ✓ | 2/3 exercised | 2/3 exercised | ✓ | ✓ |
| Louvain community detection | ✓ | not exercised | ✓ | not exercised | not exercised | ✓ | not exercised |
| Updates and create/delete batch | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Exact vector scoring | ✓ | not exercised | not exercised | not exercised | not exercised | not exercised | not exercised |
| Latitude/longitude bounding-box filter | ✓ | ✓ | not exercised | ✓ | ✓ | not exercised | not exercised |

### What's being compared

- **kglite** — Cypher graph engine, in-memory
- **DuckDB** — SQL/relational database
- **igraph** — C-backed graph library
- **Kùzu (historical)** — archived Cypher graph database, embedded
- **Neo4j (native)** — Cypher graph database, native server
- **NetworkX** — pure-Python graph library
- **rustworkx** — Rust-backed graph library

Graph-construction time is omitted from the cross-kind headline because these adapters materialise different internal representations and indices. Compare workload rows individually and consult the raw per-group report for detail.

### kglite storage modes and protocols

| Workload topic | kglite | kglite (mapped) | kglite (disk) | kglite (fluent) | kglite (Bolt) |
|---|---|---|---|---|---|
| Graph construction | 39.7ms | 44.2ms | 88.3ms | **38.2ms** | 221.5ms |
| Node/edge scans and ID lookup | **2.4ms** | 2.4ms | 3.7ms | 444µs (2/3 exercised) | 171.9ms |
| Property filters and aggregation | 3.7ms | **3.6ms** | 6.9ms | 6.2ms | 28.6ms |
| Reachability and traversal | **16.8ms** | 17.4ms | 18.0ms | 46.6ms | 219.1ms |
| Shortest-path queries | **261µs** | 273µs | 319µs | 22.0ms | 3.0ms |
| Typed graph joins and aggregation | **7.3ms** | 8.0ms | 10.1ms | 45.1ms (2/3 exercised) | 9.8ms |
| Degree and connected-components operations | **23.0ms** | 23.3ms | 70.4ms | 5.7ms (2/3 exercised) | 37.2ms |
| Louvain community detection | **133.9ms** | 134.6ms | 134.3ms | not exercised | 137.8ms |
| Updates and create/delete batch | **4.3ms** | 4.6ms | 10.2ms | 82.3ms | 20.8ms |
| Exact vector scoring | 4.2ms | 4.3ms | 11.5ms | **129µs** | 5.7ms |
| Latitude/longitude bounding-box filter | **17µs** | 17µs | 22µs | 21µs | 329µs |

### Scaling

The headline uses the `medium` graph so every recorded adapter can complete a useful subset. For the separate historical load-and-first-query study of disk-backed modes, see [`benchmarks/competitive/largescale/`](benchmarks/competitive/largescale/README.md).

## Reproduce

```bash
uv pip install --python .venv/bin/python ladybug networkx rustworkx python-igraph duckdb neo4j
uv run --no-sync maturin develop --release
make build-bolt-server
.venv/bin/python benchmarks/benchmark.py
```

The public command publishes only a clean, complete, error-free invocation containing every requested adapter. See [`graphsuite/README.md`](benchmarks/competitive/graphsuite/README.md) for opt-in native and Docker server profiles.

### Measurement protocol

Each query group records the minimum observed wall time from an adaptive 1–5 repetitions; mutation groups use at most two. Graph construction uses one measurement, or two for inexpensive non-Bolt builds. There is no warm-up phase. These labels describe this harness exactly; they are not confidence intervals or enforced timeouts.

Versions used for the selected rows:

_kglite 0.16.9, DuckDB 1.5.3, igraph 1.0.0, Kùzu (historical) 0.11.3, Neo4j (native) 2026.02.3, NetworkX 3.6.1, rustworkx 0.17.1_

Run on macOS-26.3-arm64-arm-64bit-Mach-O · Python 3.14.3.

### Capture provenance

- Publication qualification: `legacy / unqualified`
- Results schema: `2`
- Harness: `graphsuite` v3
- Dataset signature: `medium-s1234-n25333-e280774`
- Selected run timestamps: `2026-08-25T18:21:09+02:00` through `2026-08-25T18:41:06+02:00`
- Capture id: `not recorded`
- Source commit: `4822c840edb8aa3c5ea3761172f2d1a26f19e859, e2df280492ef7d5e6de8673047823784bb5c2458` (dirty: `true`)
- Base repeat policy: `5`
- Dataset seed: `1234`
- Raw metadata authority: `benchmarks/competitive/graphsuite/results.json`.
