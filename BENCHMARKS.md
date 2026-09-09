# KGLite benchmarks

Minimum observed wall time for the workloads each adapter exercised on one seed-deterministic synthetic graph. Lower is better within a row; `not exercised` is not a claim about the underlying product's capabilities.

**Dataset:** 25,333 nodes · 280,774 edges (Person/Company/Project/Skill/City), scale `medium`.

| Workload topic | kglite | DuckDB | igraph | LadybugDB | Neo4j (native) | NetworkX | rustworkx |
|---|---|---|---|---|---|---|---|
| Node/edge scans and ID lookup | 2.6ms | 1.8ms | **15µs** | 4.0ms | 5.2ms | 3.1ms | 244µs |
| Property filters and aggregation | 3.9ms | **1.4ms** | 4.1ms | 3.5ms | 62.9ms | 5.6ms | 3.7ms |
| Reachability and traversal | 19.7ms | 14.3ms | **3.1ms** | 38.93s (measured >10s) | 656.5ms | 8.3ms | 8.5ms |
| Shortest-path queries | **323µs** | not exercised | 25.6ms | 212.6ms | 16.6ms | 508µs | 93.7ms |
| Typed graph joins and aggregation | 8.7ms | **2.0ms** | not exercised | 4.1ms | 36.6ms | 127.3ms | not exercised |
| Degree and connected-components operations | 25.7ms | 3.4ms (2/3 exercised) | **2.0ms** | 10.6ms (2/3 exercised) | 54.5ms (2/3 exercised) | 11.3ms | 8.1ms |
| Louvain community detection | **150.6ms** | not exercised | 1.44s | not exercised | not exercised | 13.66s (measured >10s) | not exercised |
| Updates and create/delete batch | 4.4ms | 2.8ms | 7.1ms | 102.3ms | 80.1ms | 712µs | **127µs** |
| Exact vector scoring | **5.1ms** | not exercised | not exercised | not exercised | not exercised | not exercised | not exercised |
| Latitude/longitude bounding-box filter | **18µs** | 78µs | not exercised | 142µs | 740µs | not exercised | not exercised |

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
| Graph construction | 44.5ms | 50.2ms | 96.8ms | **43.2ms** | 234.4ms |
| Node/edge scans and ID lookup | **2.6ms** | 2.6ms | 4.1ms | 496µs (2/3 exercised) | 180.7ms |
| Property filters and aggregation | 3.9ms | **3.8ms** | 7.2ms | 6.7ms | 31.0ms |
| Reachability and traversal | **19.7ms** | 20.4ms | 20.6ms | 49.4ms | 230.8ms |
| Shortest-path queries | 323µs | **302µs** | 332µs | 23.3ms | 3.2ms |
| Typed graph joins and aggregation | 8.7ms | **8.7ms** | 11.6ms | 51.2ms (2/3 exercised) | 11.6ms |
| Degree and connected-components operations | **25.7ms** | 27.0ms | 69.4ms | 6.5ms (2/3 exercised) | 51.4ms |
| Louvain community detection | **150.6ms** | 151.7ms | 151.8ms | not exercised | 152.4ms |
| Updates and create/delete batch | **4.4ms** | 4.7ms | 50.2ms | 90.2ms | 23.7ms |
| Exact vector scoring | 5.1ms | 5.2ms | 13.4ms | **134µs** | 6.6ms |
| Latitude/longitude bounding-box filter | **18µs** | 19µs | 27µs | 22µs | 347µs |

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

_kglite 0.17.1, DuckDB 1.5.5, igraph 1.0.0, LadybugDB 0.20.3, Neo4j (native) 2026.07.1, NetworkX 3.6.1, rustworkx 0.18.1_

Run on macOS-26.6.2-arm64-arm-64bit-Mach-O · Python 3.14.3.

### Capture provenance

- Publication qualification: `qualified`
- Results schema: `2`
- Results file writer: `graphsuite` v3
- Dataset signature: `medium-s1234-n25333-e280774`
- Selected run timestamps: `2026-09-09T10:08:10+02:00` through `2026-09-09T10:09:38+02:00`
- Selected capture harness: `v3`
- Capture id: `d8051e1d57e54d00b01319a435bfd96d`
- Source commit: `4f4841288382d01531f39c4d0e5dabea4923c042` (dirty: `false`)
- Base repeat policy: `5`
- Dataset seed: `1234`
- Raw metadata authority: `benchmarks/competitive/graphsuite/results.json`.
