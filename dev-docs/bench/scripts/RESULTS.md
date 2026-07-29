# In-memory graph-library benchmark

Single script: `graph_bench.py`. One objective table: rows = benchmarks
(30 tests + load + memory), columns = libraries, plus a row-wise `sum` column.
Add a library = add an `Adapter` subclass + register it in `ADAPTERS`.

```bash
python graph_bench.py --scale 100000                 # default in-memory libs
python graph_bench.py --scale 100000 --csv t.csv     # also write CSV
python graph_bench.py --libs kglite,kuzu,igraph      # subset
python graph_bench.py --libs ...,memgraph            # needs Docker (Bolt)
```

## Method
- Default set runs **in-memory / in-process** (no wire protocol): kglite,
  turingdb, kuzu, networkx, rustworkx, igraph.
- **Byte-identical data** for every library; Cypher engines (kglite, turingdb,
  kuzu) get **byte-identical Cypher**. Pure-Python libs implement each benchmark
  natively. Each library in its **own subprocess** → clean peak-RSS. Latency =
  **min over timed rounds after warmup**.
- **Validated**: all six return identical answers on every benchmark.
- `n/a` = engine lacks that feature (turingdb OSS = count only, no aggregations).
  `—` = not applicable (e.g. sum cell on a header). The `sum` column adds the
  library values for a row (skips n/a).

## Result — 110,000 nodes / 830,162 edges (degree 8), macOS / 16 GB

| Benchmark | Unit | kglite | turingdb | kuzu | networkx | rustworkx | igraph | sum |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Load time | s | 0.161 | 1.328 | 0.448 | 2.112 | 0.469 | 0.764 | 5.282 |
| Memory — resident | MB | 311.8 | 306.4 | 330.9 | 606.0 | 357.2 | 361.5 | 2273.8 |
| Memory — peak | MB | 982.0 | 1448.4 | 350.0 | 618.4 | 357.2 | 361.5 | 4117.5 |
| Count all nodes | ms | 0.032 | 0.080 | 0.219 | 0.000 | 0.000 | 0.000 | 0.331 |
| Count Person | ms | 0.032 | 0.094 | 0.217 | 0.000 | 0.000 | 0.000 | 0.344 |
| Count Crime | ms | 0.032 | 0.059 | 0.161 | 0.000 | 0.000 | 0.000 | 0.252 |
| Count KNOWS edges | ms | 0.032 | 15.482 | 5.096 | 0.000 | 0.000 | 0.000 | 20.610 |
| Count PARTY_TO edges | ms | 0.031 | 6.763 | 0.627 | 0.000 | 0.000 | 0.000 | 7.422 |
| Filter age>50 | ms | 16.424 | 2.314 | 0.297 | 2.103 | 2.117 | 2.108 | 25.364 |
| Filter age 30–40 | ms | 6.503 | 1.916 | 0.356 | 1.795 | 1.834 | 1.838 | 14.242 |
| Filter compound (age+surname) | ms | 1.807 | 2.858 | 1.295 | 1.648 | 1.660 | 1.629 | 10.898 |
| Lookup unindexed (surname) | ms | 1.969 | 1.676 | 1.250 | 1.292 | 1.294 | 1.344 | 8.826 |
| Lookup indexed (name) | ms | 0.043 | 1.468 | 0.980 | 0.000 | 0.000 | 0.000 | 2.492 |
| Aggregate avg(age) | ms | 3.301 | n/a | 0.454 | 1.902 | 1.896 | 1.888 | 9.440 |
| Aggregate min(age) | ms | 3.320 | n/a | 0.238 | 2.116 | 2.105 | 2.093 | 9.872 |
| Aggregate max(age) | ms | 3.291 | n/a | 0.246 | 2.096 | 2.114 | 2.089 | 9.835 |
| Aggregate sum(age) | ms | 3.424 | n/a | 0.448 | 1.890 | 1.855 | 1.878 | 9.495 |
| Count distinct surnames | ms | 12.350 | n/a | 1.822 | 1.285 | 1.234 | 1.150 | 17.841 |
| Group by surname | ms | 27.161 | n/a | 2.046 | 4.046 | 4.017 | 3.877 | 41.147 |
| Scan + materialize | ms | 26.265 | 7.995 | 5.370 | 3.864 | 3.955 | 3.904 | 51.354 |
| Top-10 by age | ms | 3.186 | 6.299 | 1.097 | 12.661 | 12.617 | 12.567 | 48.428 |
| Top-100 by age | ms | 3.886 | 6.314 | 1.140 | 7.752 | 7.793 | 7.526 | 34.411 |
| Traversal PARTY_TO fwd | ms | 8.834 | 7.446 | 0.562 | 0.000 | 0.000 | 0.000 | 16.842 |
| Traversal PARTY_TO rev | ms | 14.512 | 1.517 | 1.917 | 0.000 | 0.000 | 0.000 | 17.947 |
| Seed → PARTY_TO | ms | 0.033 | 1.425 | 1.072 | 0.607 | 0.578 | 0.576 | 4.291 |
| Seed → KNOWS neighbours | ms | 0.036 | 1.406 | 1.533 | 0.000 | 0.000 | 0.000 | 2.976 |
| Surname → PARTY_TO | ms | 1.902 | 1.794 | 1.657 | 2.298 | 2.280 | 2.270 | 12.202 |
| Traversal 1-hop | ms | 0.036 | 1.426 | 1.063 | 0.000 | 0.000 | 0.000 | 2.525 |
| Traversal 2-hop | ms | 0.054 | 1.450 | 2.300 | 0.001 | 0.001 | 0.001 | 3.807 |
| Traversal 3-hop | ms | 0.219 | 1.453 | 7.709 | 0.007 | 0.007 | 0.006 | 9.401 |
| Traversal 4-hop | ms | 1.737 | 1.850 | 15.225 | 0.054 | 0.061 | 0.054 | 18.981 |
| Traversal 5-hop | ms | 21.637 | 6.038 | 23.988 | 0.402 | 0.444 | 0.418 | 52.927 |
| Traversal 6-hop | ms | 159.270 | 37.628 | 35.356 | 8.227 | 8.135 | 7.624 | 256.240 |

## Notes
- Pure-Python libs (networkx/rustworkx/igraph) have no query layer — ops are
  hand-coded, so count/walk rows show ~0 (native O(1) / raw adjacency) while
  Python-bound rows (materialize, top-N, group-by) carry their cost. Their query
  rows converge (work is Python-bound); they differ on **load** (rustworkx/igraph
  3–5× faster than networkx) and **memory** (networkx ~1.7× heavier).
- This corpus is scan/filter/traversal heavy. rustworkx & igraph's compiled
  graph-algorithm strength (shortest path, PageRank, centrality) is not exercised.
- `memgraph` is registered but excluded (Bolt server, needs Docker; network
  transport → flagged in the `Transport` row when enabled via `--libs ...,memgraph`).
- The `sum` column is a per-row total across libraries (n/a skipped), so rows with
  an n/a sum fewer libraries.
