"""Cross-storage-mode result-parity gate for the graphsuite benchmark.

Every kglite storage mode (memory / mapped / disk) runs the *same* Cypher
workloads through the shared planner+executor; for any benchmark group all
three can run, the result digest MUST be byte-for-byte identical. A
divergence here is a kglite correctness regression, not a benchmark
artifact.

This is the gate that would have caught the 0.11.2 storage-mode bugs found
while building the benchmark suite:
- the `UNWIND $list MATCH (n {id:i})` transient-index bug (>64 → 0 rows), and
- the vector-kNN storage-mode divergence (an ordering bug exposed by running
  a read group after the mutating groups).

Only kglite is exercised (no competitor libs), so it runs in the default
`make test` gate. The benchmark's own `report.render_parity()` does the
broader cross-engine check when the optional backends are installed.
"""

from __future__ import annotations

from benchmarks.competitive.graphsuite import canonical
from benchmarks.competitive.graphsuite import dataset as dm
from benchmarks.competitive.graphsuite.ad_kglite import KgliteCypher, KgliteDisk, KgliteMapped
from benchmarks.competitive.graphsuite.base import GROUPS, Skip


def _group_digests(adapter_cls, ds) -> dict[str, str]:
    """Run every group of one adapter, returning {group_id: result_digest}.
    Skipped groups are omitted; an unexpected error fails the test loudly."""
    a = adapter_cls()
    a.build(ds)
    out: dict[str, str] = {}
    try:
        for gid, _desc, method in GROUPS:
            if method is None:  # 'build' — timed separately, no result
                continue
            try:
                out[gid] = canonical.digest(getattr(a, method)(ds))
            except Skip:
                pass
    finally:
        try:
            a.teardown()
        except Exception:
            pass
    return out


def test_kglite_storage_mode_result_parity():
    ds = dm.generate("small", 1234)
    reference = _group_digests(KgliteCypher, ds)
    assert reference, "no kglite groups ran — benchmark wiring broken"
    for cls in (KgliteMapped, KgliteDisk):
        got = _group_digests(cls, ds)
        for gid, dig in reference.items():
            if gid in got:
                assert got[gid] == dig, (
                    f"{cls.__name__} diverges from KgliteCypher on group '{gid}' "
                    f"({got[gid]} != {dig}) — kglite storage-mode correctness regression"
                )
