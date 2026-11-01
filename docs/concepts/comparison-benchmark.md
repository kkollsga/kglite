# Comparison benchmark

This page describes how the README's "How it compares" table measures memory
use and performance.

The README's memory use and performance rows run the same test in a fresh
process per engine. Each process loads the graph and runs a fixed query mix six times.

- **Memory use** is the process's private memory after the first pass. Private
  memory is macOS `phys_footprint`.
- **Performance** is the end-to-end time of one warm pass over the whole mix,
  best of passes two to six. It includes building the result each API returns.
- **Graphs:** each paper has an id, a title and three other properties.

  | size   | `Paper` | `Author` | `CITES` | `WROTE` |
  |--------|---------|----------|---------|---------|
  | medium | 500k    | 50k      | 5M      | 1M      |
  | large  | 1M      | 100k     | 10M     | 2M      |

- **Query mix:** 50 point lookups, 50 one-hop and 10 two-hop expansions, and one
  filtered count over all papers. The mix is the same at both sizes, and every
  engine returns the same count.
- **Loading:** each engine opens its own on-disk store. KGLite loads a `.kgl`
  file with `storage="mapped"` or `"memory"`. LadybugDB opens read-only. Neo4j
  opens through the embedded Java API. NetworkX and rustworkx have no on-disk
  store, so they build from Parquet in the process.
- **Runtime included:** Python with the engine imported is 13-16 MB, Python
  with pandas 64 MB, and the JVM 47 MB.
- **File pages excluded:** mapped KGLite and LadybugDB read file pages that the
  OS can drop under pressure. Tools that report RSS count those pages. Mapped
  KGLite shows 150 / 257 MB RSS and LadybugDB 101 / 111 MB.
- **NetworkX on the large graph** does not fit in RAM on the test machine.
  macOS compresses part of it, so only 3.8 GB of the 7.7 GB is resident.
- **Where the time goes.** KGLite answers the 110 lookups and expansions in
  about 0.4 ms and the filtered count in 0.27 / 0.53 ms, from its column
  kernel. LadybugDB's columnar scan runs the count in 1.4 / 2.0 ms, and its
  lookups and expansions take 3.1-3.5 ms.
- **The first pass is slower.** Right after loading, Neo4j takes 0.8 / 1.1 s
  while it compiles the queries. The other engines take 15-125 ms.
- **Versions:** KGLite 0.21.5, LadybugDB 0.22.1, NetworkX 3.7, rustworkx 0.18.1,
  and Neo4j 2026.02.3 on JDK 21. Python 3.14 on an Apple M4 with 16 GB, with
  other work running (load average about 6; about 10 for the KGLite 0.21.5
  performance cells, measured from its release build before publishing).
- **Repeats:** every configuration ran twice. A memory cell is the higher of the
  two runs, which agreed within 12 MB. A performance cell is the faster run;
  the two runs agreed within 16 %.
