# benchmarks/

Public, reproducible performance comparisons of kglite vs other graph
libraries. This folder is for **showing how kglite performs**; the
performance *gates* that guard against regressions live under
[`tests/benchmarks/`](../tests/benchmarks/) (run by CI), and the internal /
exploratory one-off scripts moved to
[`tests/benchmarks/internal/`](../tests/benchmarks/internal/).

## One command

```bash
pip install kglite kuzu networkx rustworkx igraph duckdb   # competitors are optional
python benchmarks/benchmark.py                              # → writes BENCHMARKS.md
```

This stages a seed-deterministic synthetic graph with the **bundled**
`kglite.graphgen` (no Rust toolchain needed — it ships in the wheel), runs
every installed backend across a suite of workloads on those identical
bytes, and (re)writes the top-level **[`BENCHMARKS.md`](../BENCHMARKS.md)** —
the topic-summed comparison table you can link to. `--scale large` for a
bigger graph; `--report-only` to just rebuild the table from saved results.

## Layout

| Path | What |
|---|---|
| `benchmark.py` | The one-command entry point above. |
| `competitive/graphsuite/` | The engine: 15 workload groups × many backends (kglite ×5 modes, Kùzu, Neo4j, DuckDB, NetworkX, rustworkx, igraph) on one shared graph; accumulates `results.json`. `marketing.py` renders `BENCHMARKS.md`; `report.py` prints the fine-grained per-group matrix. |
| `competitive/largescale/`, `competitive/embedded_app/`, `competitive/nornic/` | Scenario / larger-than-RAM benchmarks. Local-only (gitignored); `graphsuite` is the tracked public suite. |
| `graphgen/` | The standalone Rust CLI form of the generator. The generator now lives in the engine (`crates/kglite/src/graphgen/`) and is bundled in the wheel as **`kglite.graphgen(...)`** — what `benchmark.py` uses, so no `cargo` is required. This CLI is just a thin direct-streaming front-end. |

## Fairness

Every engine loads identical data and runs the *same* seed-derived queries;
each workload prints a result digest so backends are checked to be doing
equal work. Native algorithm libraries (NetworkX / rustworkx / igraph) have
no query language, engine-level filtering, or persistence — they're
in-process toolkits, not databases — so read the table with the "what's
being compared" legend in mind.
