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
| `competitive/graphsuite/` | The engine: 26 workload groups across 9 categories (scan/filter/aggregate/traversal/pathfinding/multi-type queries/graph algorithms/community detection/mutations + vector search + geospatial) × many backends (kglite ×5 modes + Bolt-in-Docker, Kùzu, Neo4j ×2 deploy flavors, DuckDB, NetworkX, rustworkx, igraph) on one shared graph; accumulates `results.json`. `marketing.py` renders `BENCHMARKS.md`; `report.py` prints the fine-grained per-group matrix; `neo4j_server.py` auto-provisions the Neo4j flavors. |
| `competitive/largescale/`, `competitive/embedded_app/` | Scenario / larger-than-RAM benchmarks. Local-only (gitignored); `graphsuite` is the tracked public suite. |

The synthetic-graph generator lives in the engine core
(`crates/kglite/src/graphgen/`) and ships in the wheel as
**`kglite.graphgen(...)`** — what `benchmark.py` stages from, so no `cargo` is
required. (It used to also exist as a standalone `benchmarks/graphgen/` CLI
crate; that duplicate was removed once the generator was bundled in 0.11.2.)

## Opt-in backends (Neo4j, kglite over Bolt-in-Docker)

A few backends are **opt-in** — they're heavy (start an external server /
build a container) so a default run stays fast and dependency-free. Each
skips cleanly with an explanation when its prerequisite is missing, so
adding them never breaks a plain run. Request them via `--libs`:

```bash
# Neo4j — two deployment flavors, both auto-provisioned and torn down:
python -m benchmarks.competitive.graphsuite.run --libs neo4j-native   # local server (needs Java + the `neo4j` CLI) — higher performance
python -m benchmarks.competitive.graphsuite.run --libs neo4j-docker   # neo4j:5-community container (needs a running Docker daemon)

# kglite served over the Bolt wire protocol from a container — the
# containerised peer to `kglite-bolt` and a like-for-like vs neo4j-docker:
python -m benchmarks.competitive.graphsuite.run --libs kglite-bolt-docker   # needs Docker; builds crates/kglite-bolt-server/Dockerfile once
```

`neo4j-native` launches the installed Neo4j against a throwaway config
(temp data dir, free Bolt port, HTTP off) and is the fair "Neo4j at its
best" number; `neo4j-docker` is the portable baseline (on macOS it pays
the Docker-VM tax). In the public table the strongest Neo4j run collapses
into a single **Neo4j** column; `report.py` shows every variant. The
provisioning lives in `competitive/graphsuite/neo4j_server.py`.

## Fairness

Every engine loads identical data and runs the *same* seed-derived queries;
each workload prints a result digest so backends are checked to be doing
equal work. Native algorithm libraries (NetworkX / rustworkx / igraph) have
no query language, engine-level filtering, or persistence — they're
in-process toolkits, not databases — so read the table with the "what's
being compared" legend in mind.
