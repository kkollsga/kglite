# graphgen — streaming synthetic graph generator

A pure-Rust, zero-dependency generator that emits benchmark property graphs
at **any scale in bounded memory**. It streams one CSV per node/edge type plus
a `manifest.json` (schema + seed-derived query params) to an output directory.
Every engine (kùzu `COPY FROM`, kglite `add_nodes`) loads the *same* bytes, so
cross-engine result-parity holds by construction.

## Why it exists

The Python `graphsuite` generator builds the whole graph as lists-of-dicts →
pandas in RAM, so it caps out around 100k nodes — useless for larger-than-memory
benchmarks. `graphgen` never holds the graph in memory: it generates and writes
row-by-row. Measured: **6.3M nodes + 70M edges in 4.0 s at 14 MB peak RSS**.
A 50M-person graph (`--scale xhuge`, ~700M edges, ~13 GB CSV) generates in the
same ~14 MB of RAM.

## Build

```bash
cd benchmarks/graphgen && cargo build --release   # standalone crate, not in the workspace
```

## Usage

```bash
./target/release/graphgen --out DIR [--scale NAME | --persons N] \
    [--seed S] [--knows-per K] [--degree-dist uniform|zipf] [--zipf-exp E]
```

| Flag | Default | Meaning |
|---|---|---|
| `--out DIR` | (required) | Output directory for CSVs + `manifest.json` |
| `--scale` | — | `tiny`(1k) `small`(2k) `medium`(20k) `large`(100k) `huge`(5M) `xhuge`(50M) persons |
| `--persons N` | — | Exact person count (overrides `--scale`) |
| `--seed S` | 1234 | Deterministic seed |
| `--knows-per K` | 8 | Avg KNOWS out-degree per person |
| `--degree-dist` | `zipf` | `uniform` or `zipf` — zipf creates high-degree **hubs** (realistic; makes k-hop traversal explode, which is the interesting benchmark axis) |
| `--zipf-exp E` | 1.6 | Skew exponent (>1 ⇒ stronger hubs) |

Other node/edge counts scale off persons with the same ratios as `graphsuite`
(`Company = persons/25`, `Project = persons/5`, `Skill = persons/60`,
`City = persons/100`).

## Schema

Five node types (`Person`, `Company`, `Project`, `Skill`, `City`) and seven edge
types (`KNOWS`, `WORKS_AT`, `CONTRIBUTES_TO`, `HAS_SKILL`, `OWNS`, `DEPENDS_ON`,
`LOCATED_IN`) — identical to `graphsuite/dataset.py`, so the existing kùzu and
kglite adapters and query suite consume the output unchanged. `DEPENDS_ON` is a
DAG (downstream gid < source gid). Node `gid`s are contiguous global ids by type.

## Output

```
DIR/
  Person.csv Company.csv Project.csv Skill.csv City.csv     # nodes (with headers)
  KNOWS.csv WORKS_AT.csv ... LOCATED_IN.csv                 # edges: src,dst
  manifest.json                                             # schema, counts, ranges, query params
```

`manifest.json` carries the seed-derived, *valid* query params (lookup ids,
seed persons/projects for traversal, shortest-path pairs, filter constants) so a
load+query harness runs identical queries across every engine.
