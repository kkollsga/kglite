# Benchmark capture qualification

Versioned JSON files preserve raw measurements. `current.json` and
`current.linux.json` identify the latest captured workload; the per-platform
`references` pointers in `qualifications.json` select approved comparison data.
Capturing a release alone does not advance those pointers.

Each decision records the filename, SHA-256, operating system/architecture,
status and evidence. Changing registered bytes invalidates their decision.
Rejected captures cannot become candidates by being renamed. A rejected
candidate fails without substituting another capture. Missing eligible data,
platform identity or usable measurements also fails without a performance verdict.

After `make refresh-release-constants`, inspect the pending capture against
unchanged-path controls and repeat the measurements. Record machine load and
the statistic used; investigate disagreements and repeat near-threshold results.
Then explicitly record the decision, for example:

```sh
uv run --no-sync python scripts/benchmark_qualification.py \
  tests/benchmarks/baselines/0_17_0.json --status accepted --promote \
  --evidence "Two agreeing release runs; controls stable; see release measurement record"
```

Use `--status rejected` with the actual reason for an unsuitable capture.
Never rewrite historical measurements to make them pass. A qualification retry
preserves the existing capture; promotion requires an accepted versioned file.
`compare_bench.py` resolves current aliases through the approved pointer.
`check_perf_anchor.py` chooses the chronological release distance first, then
walks backwards over unsuitable references, reporting exclusions. It cannot
compare a candidate against itself or a later release.

The macOS 0.16.22 and 0.16.23 captures remain unchanged but rejected because
wheel/control measurements established load-related inflation. The approved
macOS reference is 0.16.19.

`reference_0_16_23.linux.json` preserves the existing 36-cell Linux secondary CI
reference, including its provenance and limitations: eleven cells are provisional
scaled estimates, not Linux measurements. Acceptance preserves that secondary
check; it does not convert estimates into observations. The primary same-runner
CI comparison remains authoritative. `promote_linux_benchmark.py` validates fresh
Linux wheel provenance and workload coverage before promoting replacement data.
