# KGLite benchmarks

Minimum observed wall time for the workloads each adapter exercised on one seed-deterministic synthetic graph. Lower is better within a row; `not exercised` is not a claim about the underlying product's capabilities.

**Dataset:** 25,333 nodes · 266,944 edges (Person/Company/Project/Skill/City), scale `medium`.

| Workload topic | kglite | DuckDB | igraph | LadybugDB | Neo4j (native) | NetworkX | rustworkx |
|---|---|---|---|---|---|---|---|
| Node/edge scans and ID lookup | 2.9ms | 1.8ms | **16µs** | 4.7ms | 5.4ms | 4.7ms | 275µs |
| Property filters and aggregation | 4.0ms | **1.4ms** | 4.7ms | 3.9ms | 83.7ms | 6.3ms | 4.1ms |
| Reachability and traversal | 11.0ms | 13.2ms | **2.0ms** | 16.62s (measured >10s) | 211.9ms | 4.8ms | 5.2ms |
| Shortest-path queries | **2.8ms** | not exercised | 118.5ms | 1.15s | 74.2ms | 5.3ms | 350.9ms |
| Typed graph joins and aggregation | 12.5ms | **2.0ms** | not exercised | 4.6ms | 46.9ms | 124.8ms | not exercised |
| Degree and connected-components operations | 25.0ms | 3.0ms (2/3 exercised) | **2.0ms** | 11.5ms (2/3 exercised) | 53.0ms (2/3 exercised) | 20.4ms | 5.8ms |
| Louvain community detection | **17.8ms** | not exercised | 87.4ms | not exercised | not exercised | 1.94s | not exercised |
| Updates and create/delete batch | 2.7ms | 2.9ms | 6.1ms | 87.8ms | 53.7ms | 391µs | **130µs** |
| Exact vector scoring | **5.6ms** | not exercised | not exercised | not exercised | not exercised | not exercised | not exercised |
| Latitude/longitude bounding-box filter | **21µs** | 76µs | not exercised | 339µs | 731µs | not exercised | not exercised |

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
| Graph construction | 73.5ms | 79.1ms | 138.1ms | **72.9ms** | 285.6ms |
| Node/edge scans and ID lookup | 2.9ms | **2.8ms** | 5.0ms | 1.2ms (2/3 exercised) | 190.0ms |
| Property filters and aggregation | 4.0ms | **3.9ms** | 7.3ms | 7.7ms | 31.6ms |
| Reachability and traversal | **11.0ms** | 11.1ms | 12.4ms | 48.0ms | 156.9ms |
| Shortest-path queries | **2.8ms** | 3.2ms | 2.9ms | 123.1ms | 23.6ms |
| Typed graph joins and aggregation | **12.5ms** | 14.0ms | 20.3ms | 62.1ms (2/3 exercised) | 14.9ms |
| Degree and connected-components operations | **25.0ms** | 25.9ms | 61.2ms | 6.9ms (2/3 exercised) | 49.2ms |
| Louvain community detection | 17.8ms | 18.2ms | **17.7ms** | not exercised | 21.7ms |
| Updates and create/delete batch | **2.7ms** | 3.0ms | 61.8ms | 60.5ms | 25.7ms |
| Exact vector scoring | 5.6ms | 5.2ms | 13.7ms | **125µs** | 7.5ms |
| Latitude/longitude bounding-box filter | **21µs** | 21µs | 27µs | 23µs | 370µs |

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

_kglite 0.17.1, DuckDB 1.5.5, igraph 1.0.0, LadybugDB 0.17.0, Neo4j (native) 2026.07.1, NetworkX 3.6.1, rustworkx 0.18.1_

Run on macOS-26.6.2-arm64-arm-64bit-Mach-O · Python 3.14.3.

### Capture provenance

- Publication qualification: `qualified`
- Results schema: `2`
- Results file writer: `graphsuite` v3
- Dataset signature: `medium-s1234-n25333-e266944`
- Selected run timestamps: `2026-09-09T09:44:30+02:00` through `2026-09-09T09:45:27+02:00`
- Selected capture harness: `v3`
- Capture id: `97eaae7115a14aacb13d16c5de97c74c`
- Source commit: `bf6c7bbf07e747e851254f5a0e45d2fe3995144b` (dirty: `false`)
- Base repeat policy: `5`
- Dataset seed: `1234`
- Raw metadata authority: `benchmarks/competitive/graphsuite/results.json`.
