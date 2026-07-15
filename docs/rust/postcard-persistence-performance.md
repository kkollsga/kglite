# Postcard persistence performance gate

Measured 2026-07-15 on an Apple M4 Mac mini. This is the release gate for the
bincode-to-Postcard persistence migration; it is not a general product
benchmark.

## Method

The candidate was built as the packaged release wheel from commit `8b6307ed`.
The reference was the published `kglite==0.13.3` wheel. Both ran from isolated
Python 3.12.9 environments outside the checkout, with identical pandas 2.3.3
and numpy 2.5.1 dependencies. Each timed persistence cell ran five rounds and
the minimum is the primary result. Runs were separated by a 30-second thermal
settle; the candidate was repeated twice and the reference once when disk-save
and WAL-recovery samples looked noisy.

The approved plan also named the Phase-3 codec-boundary commit as a reference.
It was not rebuilt: the repository performance protocol forbids source-building
an old revision for A/B evidence. Phase 3 was byte-neutral and still used the
same active bincode writers, so the published 0.13.3 wheel is the reproducible
bincode reference. The standalone harness is
`tests/benchmarks/internal/bench_postcard_persistence.py`.

## Results

The 20,000-node / 60,000-edge fixture deliberately uses small integer IDs,
groups, lengths, edge ranks, and list values.

| Cell | Published 0.13.3 | Postcard candidate | Result |
|---|---:|---:|---:|
| Portable `.kgl` size | 414,484 B | 258,872 B | **37.5% smaller** |
| Portable save, min | 8.04–8.16 ms | 5.33–5.36 ms | **33–35% faster** |
| Portable load + count, min | 7.96–8.21 ms | 7.63–7.89 ms | 1–7% faster |
| Disk directory size | 6,037,357–6,037,359 B | 5,077,021–5,077,022 B | **15.9% smaller** |
| Disk save, min | 38.93–41.49 ms | 39.50–43.46 ms | overlapping; no material change |
| Disk open + count, min | 2.43–2.44 ms | 2.30–2.33 ms | 4–6% faster |
| WAL size, 1,500 mutations | 160,895 B | 61,704 B | **61.7% smaller** |
| WAL append, min | 5.94–5.95 s | 5.73–5.83 s | 2–4% faster |
| WAL recovery, median | 2.38–2.62 ms | 2.17–2.32 ms | no regression |
| N-Triples disk ingest, min | 86.46–88.19 ms | 85.33–88.08 ms | overlapping; no material change |
| Process peak RSS | 214,958,080 B | 214,614,016 B | flat (−0.16%) |

The first WAL baseline produced a single 1.13 ms minimum while its median was
2.38 ms. The required replay produced a 2.43 ms minimum / 2.62 ms median,
confirming that the first minimum was an outlier rather than a Postcard
regression. Disk-save repeats likewise crossed in both directions; no stable
slowdown reproduced.

Edge-property scans, overflow-list projection, and the small in-memory filter
canaries overlapped across the repeated runs. The repository's standardized
release suite provided the stronger in-memory gate: all 27 benchmarks in
`tests/benchmarks/test_bench_core.py` were at least as fast as the committed
0.13.3 baseline by minimum time. The range was 1.2% faster
(`return_node_10k`) to 22.8% faster (`save_v3`); no regression was observed.

## Commands

```bash
maturin develop --release
maturin build --release --out /tmp/kglite-postcard-dist

uv venv /tmp/kglite-postcard-bench-envs/baseline --python 3.12
uv pip install --python /tmp/kglite-postcard-bench-envs/baseline/bin/python \
  'kglite==0.13.3' pandas==2.3.3

uv venv /tmp/kglite-postcard-bench-envs/candidate --python 3.12
uv pip install --python /tmp/kglite-postcard-bench-envs/candidate/bin/python \
  /tmp/kglite-postcard-dist/kglite-0.13.3-cp310-abi3-macosx_11_0_arm64.whl \
  pandas==2.3.3

# Run from /tmp; repeat with each interpreter and a 30-second settle.
python /absolute/path/to/bench_postcard_persistence.py \
  --output result.json --scale 20000 --rounds 5 \
  --wal-mutations 1500 --ntriples-entities 20000

pytest tests/benchmarks/test_bench_core.py -m benchmark --benchmark-only \
  --benchmark-min-rounds=100 --benchmark-warmup=on \
  --benchmark-warmup-iterations=20 --benchmark-json=result.json
```

## Decision

The migration passes the performance gate. It materially reduces every
codec-sensitive artifact measured, does not increase peak memory, and shows no
repeatable in-memory, disk-open, disk-save, WAL, property-log, edge-property,
or overflow-path regression.
