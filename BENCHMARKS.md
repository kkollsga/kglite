# KGLite benchmarks

Minimum observed wall time for the workloads each adapter exercised on one seed-deterministic synthetic graph. Lower is better within a row; `not exercised` is not a claim about the underlying product's capabilities.

**Dataset:** 25,333 nodes · 280,774 edges (Person/Company/Project/Skill/City), scale `medium`.

| Workload topic | kglite | DuckDB | igraph | LadybugDB | Neo4j (native) | NetworkX | rustworkx |
|---|---|---|---|---|---|---|---|
| Node/edge scans and ID lookup | 2.6ms | 1.7ms | **14µs** | 4.3ms | 5.2ms | 3.2ms | 246µs |
| Property filters and aggregation | 3.9ms | **1.4ms** | 4.1ms | 3.4ms | 60.2ms | 5.7ms | 3.7ms |
| Reachability and traversal | 17.8ms | 13.4ms | **3.0ms** | 35.02s (measured >10s) | 279.1ms | 8.2ms | 8.4ms |
| Shortest-path queries | **318µs** | not exercised | 24.3ms | 207.1ms | 13.3ms | 504µs | 89.9ms |
| Typed graph joins and aggregation | 7.7ms | **1.9ms** | not exercised | 4.2ms | 28.0ms | 119.2ms | not exercised |
| Degree and connected-components operations | 17.3ms | 3.3ms (2/3 exercised) | **2.0ms** | 10.7ms (2/3 exercised) | 40.6ms (2/3 exercised) | 10.3ms | 7.8ms |
| Louvain community detection | **138.7ms** | not exercised | 1.22s | not exercised | not exercised | 11.99s (measured >10s) | not exercised |
| Updates and create/delete batch | 4.3ms | 2.5ms | 6.6ms | 122.7ms | 68.6ms | 693µs | **124µs** |
| Exact vector scoring | **4.6ms** | not exercised | not exercised | not exercised | not exercised | not exercised | not exercised |
| Latitude/longitude bounding-box filter | **17µs** | 78µs | not exercised | 324µs | 674µs | not exercised | not exercised |

Bold marks the fastest complete, non-slow measurement in that row. Partial cells sum only the named executed groups; missing work is not estimated. Values above 10 seconds remain in the measurement and are labelled, not treated as timeouts. There is deliberately no grand total across unequal workload coverage.

Adapters receive the same generated input records and shared query parameters, but use different data models and idiomatic APIs. Their operations are corresponding workloads, not necessarily identical query text or identical execution semantics. The detailed parity report records result differences instead of treating every cross-library difference as a failure.

### Workloads exercised by these adapters

This is coverage of the recorded adapter implementations, not a product capability matrix.

| Workload topic | kglite | DuckDB | igraph | LadybugDB | Neo4j (native) | NetworkX | rustworkx |
|---|---|---|---|---|---|---|---|
| Node/edge scans and ID lookup | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Property filters and aggregation | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Reachability and traversal | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
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
- **LadybugDB** — Cypher graph database, embedded
- **Neo4j (native)** — Cypher graph database, native server
- **NetworkX** — pure-Python graph library
- **rustworkx** — Rust-backed graph library

Graph-construction time is omitted from the cross-kind headline because these adapters materialise different internal representations and indices. Compare workload rows individually and consult the raw per-group report for detail.

### kglite storage modes and protocols

| Workload topic | kglite | kglite (mapped) | kglite (disk) | kglite (fluent) | kglite (Bolt) |
|---|---|---|---|---|---|
| Graph construction | **42.2ms** | 49.2ms | 88.5ms | 42.2ms | 245.9ms |
| Node/edge scans and ID lookup | 2.6ms | **2.6ms** | 4.3ms | 519µs (2/3 exercised) | 173.7ms |
| Property filters and aggregation | 3.9ms | **3.7ms** | 7.4ms | 6.2ms | 29.5ms |
| Reachability and traversal | **17.8ms** | 18.2ms | 19.2ms | 51.8ms | 219.9ms |
| Shortest-path queries | 318µs | **297µs** | 350µs | 23.9ms | 2.8ms |
| Typed graph joins and aggregation | **7.7ms** | 7.9ms | 10.5ms | 47.4ms (2/3 exercised) | 9.9ms |
| Degree and connected-components operations | **17.3ms** | 25.5ms | 18.1ms | 6.3ms (2/3 exercised) | 28.5ms |
| Louvain community detection | 138.7ms | **138.2ms** | 153.8ms | not exercised | 145.1ms |
| Updates and create/delete batch | **4.3ms** | 4.6ms | 50.5ms | 93.9ms | 22.5ms |
| Exact vector scoring | 4.6ms | 4.4ms | 13.3ms | **120µs** | 6.0ms |
| Latitude/longitude bounding-box filter | **17µs** | 17µs | 25µs | 24µs | 277µs |

### Scaling

The headline uses the `medium` graph selected above. For the separate historical load-and-first-query study of disk-backed modes, see [`benchmarks/competitive/largescale/`](benchmarks/competitive/largescale/README.md).

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

_kglite 0.17.3, DuckDB 1.5.5, igraph 1.0.0, LadybugDB 0.17.0, Neo4j (native) 2026.02.3, NetworkX 3.6.1, rustworkx 0.18.1_

Run on macOS-26.6.2-arm64-arm-64bit-Mach-O · Python 3.14.3.

### Capture provenance

- Publication qualification: `qualified`
- Results schema: `2`
- Results file writer: `graphsuite` v3
- Dataset signature: `medium-s1234-n25333-e280774`
- Selected run timestamps: `2026-09-11T16:23:48+02:00` through `2026-09-11T16:25:07+02:00`
- Selected capture harness: `v3`
- Capture id: `b3c21298286d4ea0bf25ab663fd2e57d`
- Source commit: `ea06b81042aa16429d7c4dfbb6d89d4dafa180d2` (dirty: `false`)
- Base repeat policy: `5`
- Dataset seed: `1234`
- Raw metadata authority: `benchmarks/competitive/graphsuite/results.json`.
