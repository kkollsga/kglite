# tests/benchmarks/internal/

Internal, maintainer-facing performance scripts — **not** the customer-facing
comparison (that's [`benchmarks/`](../../../benchmarks/) → `BENCHMARKS.md`),
and **not** the CI perf gates (those are the `test_bench_*.py` files one level
up in [`tests/benchmarks/`](../)).

These are standalone, run-on-demand profiling / exploration tools used while
developing kglite — ingest-throughput baselines, Cypher scalability sweeps,
storage-mode comparisons, `describe()` cost, Wikidata-scale and legal-graph
probes. They're kept here (out of the marketing folder) for reference; they
aren't collected by pytest (no `test_` prefix) and aren't run in CI.

Each is invoked directly, e.g.:

```bash
python tests/benchmarks/internal/api_benchmark.py
python tests/benchmarks/internal/wiki_benchmark.py
```

| Script | Probes |
|---|---|
| `api_benchmark.py` | Bulk-ingest throughput (uses `sodir_graph_config.json`; streams `ingest_baseline.csv`). |
| `bench_graph_traversal.py`, `bench_algorithms_quick.py` | Traversal + graph-algorithm micro-benchmarks. |
| `benchmark_cypher_scalability.py` | Cypher planner/executor scaling. |
| `benchmark_storage_modes.py` | memory vs mapped vs disk. |
| `benchmark_describe.py`, `benchmark_wikidata_describe.py` | `describe()` cost. |
| `benchmark_wikidata_cypher.py`, `wiki_benchmark.py` | Wikidata-scale load/query (appends `wiki_benchmark.csv`). |
| `benchmark_legal_graph.py` | Legal-graph scenario. |
| `bench_save_subset.py` | Subset-save cost. |
| `bench_postcard_persistence.py` | Cross-version portable/disk/WAL/property-log persistence and artifact sizes. |

Some target datasets or scales that aren't wired into a one-command flow; treat
them as dev tools, and prefer `benchmarks/benchmark.py` for any
externally-shareable numbers.
