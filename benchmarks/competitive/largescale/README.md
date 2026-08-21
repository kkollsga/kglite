# largescale — staged-dataset benchmark (kùzu vs kglite-mapped vs kglite-disk)

Loads a graph staged on disk by the bundled `kglite.graphgen` streaming
generator into each engine and times **load + a query suite**. All three
engines read the *same* staged CSVs, so results are directly comparable and the
parity check is meaningful.

Unlike the in-RAM `graphsuite` micro-benchmark, this targets the regime that
matters for big graphs: kglite `mapped`/`disk` are mmap-backed columnar, kùzu is
paged/buffer-managed — all page from disk rather than holding the graph in heap.
kglite loads are chunked (`--chunk`, default 2M rows) so loading stays
bounded-memory at any scale; kùzu `COPY FROM` is natively bounded.

```bash
python -c "import kglite; kglite.graphgen('large', out='/tmp/g_large')"
python -m benchmarks.competitive.largescale.bench /tmp/g_large
```

Needs `pandas` and `kuzu` installed alongside kglite (`--engines mapped,disk`
skips kùzu). The harness is exception-safe: a query that blows an engine's
memory (kùzu's buffer pool OOMs on hub-heavy traversal) is recorded as `ERR` and
the run continues, rather than aborting.

## Methodology — read this before quoting a number

Each phase is timed **once** per engine, immediately after that engine loaded
the data. This is a load-and-first-query harness, not a repeated-rounds
micro-benchmark: there is no warm-up, no repeat count, and no min/median over
rounds. Order-of-magnitude gaps (the 20×–6000× rows below) survive that
weakness comfortably; a gap under ~2× does not — re-run before quoting one.

The harness prints the kglite and kùzu versions it ran, because a table of
numbers without them is not a capture anyone can compare against later.

## Results

**Capture provenance.** Apple M-series laptop, 16 GB RAM, release build,
2026-06-13 — kglite at the development tree that became **v0.10.18** (just
after the cyclic-pattern optimisations: matcher `target_hint` + planner
`reorder_cyclic_pattern_edges`). The kùzu version of that run was not recorded;
the harness did not print engine versions at the time. Machine load during the
capture was not recorded either.

The staged dataset itself is not byte-identical to a fresh run either: the
generator has drifted slightly (a `small` graph is 28,115 edges today against
the 28,083 recorded below).

**These numbers are historical.** The engine has moved a long way since
v0.10.18 — loader, columnar storage shape, and write path especially. Treat the
table as the shape of the result, not as current performance, and re-run the
harness for a number you intend to rely on.

Wall time, lower is better. Dataset is `graphgen`'s zipf-skewed social graph
(hubs present — the realistic structure that makes k-hop traversal explode).

### large — 126,666 nodes · 1,403,822 edges

| phase | kùzu | kglite-mapped | kglite-disk | best |
|---|--:|--:|--:|---|
| load | 555 ms | 614 ms | 1.39 s | **kùzu** (disk pays load) |
| point_lookup (500 by id) | 20.5 ms | 0.9 ms | 0.9 ms | **kglite ~23×** |
| property_filter | 6.2 ms | 7.6 ms | 4.3 ms | ~tie (disk wins) |
| group_aggregate | 6.4 ms | 4.8 ms | 7.0 ms | ~tie |
| one_hop | 140.9 ms | 1.6 ms | 1.7 ms | **kglite ~83×** |
| **three_hop** | **90.5 s** | 15.2 ms | 14.7 ms | **kglite ~6000×** |
| deep_dag (`*1..15`) | 4.08 s | 0.6 ms | 0.6 ms | **kglite ~6800×** |
| **pattern_match** (4-way cyclic join) | **17.0 ms** | 37.5 ms | 34.4 ms | **kùzu ~2×** |

> kùzu's `three_hop` is 90.5 s on a hub graph — it materialises the exploding
> path set; kglite's global-dedup BFS did it in 15 ms. This is the canonical
> embedded-app query, and it's where kùzu falls over at scale.

### medium — 25,333 nodes · 280,775 edges

| phase | kùzu | kglite-mapped | kglite-disk |
|---|--:|--:|--:|
| load | 350 ms | 95 ms | 124 ms |
| point_lookup | 9.1 ms | 0.9 ms | 0.8 ms |
| property_filter | 2.5 ms | 1.7 ms | 1.7 ms |
| group_aggregate | 2.5 ms | 1.0 ms | 1.6 ms |
| one_hop | 29.2 ms | 1.6 ms | 1.7 ms |
| three_hop | 33.3 s | 12.4 ms | 11.7 ms |
| deep_dag | 451.7 ms | 0.4 ms | 0.4 ms |
| **pattern_match** (tied) | 6.1 ms | 7.6 ms | **6.2 ms** |

### small — 2,533 nodes · 28,083 edges

| phase | kùzu | kglite-mapped | kglite-disk |
|---|--:|--:|--:|
| load | 310 ms | 11.9 ms | 13.4 ms |
| point_lookup | 10.3 ms | 0.8 ms | 0.7 ms |
| property_filter | 2.4 ms | 0.2 ms | 0.2 ms |
| group_aggregate | 1.6 ms | 0.1 ms | 0.2 ms |
| one_hop | 5.5 ms | 1.7 ms | 1.6 ms |
| three_hop | 4.28 s | 4.2 ms | 4.4 ms |
| deep_dag | 20.5 ms | 0.2 ms | 0.1 ms |
| **pattern_match** (kglite wins) | 1.5 ms | 0.9 ms | **0.6 ms** |

## Parity

Every phase also reports its result *value* per engine, and the run prints
`[ok]` or `[DIFF]` against the first engine — a benchmark whose engines answer
different questions is measuring nothing.

One known `[DIFF]`: the generator's `lookup_ids` list repeats ids, and
`MATCH (n:Person) WHERE n.id IN $ids RETURN count(n)` counts matched *nodes* in
kùzu but one row *per matching list entry* in kglite (500 vs 446 at `small`,
measured with kglite 0.16.5 / kùzu 0.11.3). kglite's is the wrong answer — a
MATCH binds each node once, so a repeated value in a `WHERE ... IN` list must
not duplicate the row — and it is a defect in the id fast path, not a
definition. Re-run after any matcher change: that line should read `[ok]`.
The other six phases agree exactly.

## What this shows

**kglite dominates the interactive / traversal path, by widening margins.**
one_hop, three_hop, deep_dag, point_lookup are 20×–6000× faster, and kùzu takes
*seconds* where kglite stays in single-digit ms. This is the in-memory-first
row/traversal model paying off.

**The 4-way cyclic `pattern_match` is the one query kùzu's engine wins — and the
gap is now mostly closed.** Before the cyclic-reorder work it was ~16–21 ms at
medium (kùzu ~6 ms). The matcher `target_hint` (cycle-close = O(1) check) +
planner re-rooting (start the cycle at its most-selective node) brought it to:

- **small — kglite wins** (0.6 ms vs 1.5 ms),
- **medium — tied** (disk 6.2 ms ≈ kùzu 6.1 ms),
- **large — kùzu ~2×** (17 ms vs 34–37 ms).

The residual at large is kùzu's vectorised + factorised join executing the large
intermediate set faster — a deliberate design cliff: kglite does not rebuild a
columnar vectorised join engine. The planner pass minimises the *start* and
*intermediate* size; it can't vectorise the join itself.

**mapped vs disk track each other on query perf.** Disk's only real cost is
**load** (1.39 s vs mapped 614 ms at large) — the CSR + columnar build. Pick disk
for larger-than-RAM graphs and cheap reopen; mapped when it fits and load speed
matters.

**Scoreboard (large):** kglite won 6 of 8 phases (often categorically), tied 2
analytical scans, and lost load-time (disk) and `pattern_match` (~2×). For an
embedded graph app doing durable writes + interactive multi-hop reads, kglite is
the stronger choice; kùzu's edge here is narrow — vectorised analytical joins at
scale.

## Larger-than-memory

The 16 GB test box fits `large` in RAM, so these numbers stress paging without
strictly exceeding RAM. For a true >RAM run, generate `xhuge` (~50M persons,
~700M edges, ~13 GB of CSV) and point the harness at it — kglite mapped/disk and
kùzu all page from disk. Expect kùzu to struggle on the traversal phases: it
already spends 90 s on `three_hop` at `large`, and a query that exhausts its
buffer pool is recorded as `ERR` rather than aborting the run.
