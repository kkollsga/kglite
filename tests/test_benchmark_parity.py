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
from benchmarks.competitive.graphsuite.ad_kglite import KgliteCypher, KgliteDisk, KgliteFluent, KgliteMapped
from benchmarks.competitive.graphsuite.base import GROUPS, Skip
import pytest


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


# Groups whose fluent form is a same-semantics restatement of the Cypher cell
# (re-derived and measured 2026-08-25). Anything absent is deliberately out.
FLUENT_PARITY_GROUPS = (
    "range_filter",
    "degree_filter",
    "degree_topk",
    "shortest_path",
    "pattern_match",
    "industry_aggregation",
    "geo_within",
    "vector_knn",
)


def test_fluent_matches_cypher_on_the_groups_it_claims():
    """Allowlisted fluent-vs-Cypher digest gate — deliberately not the full loop.

    The two surfaces are not required to agree everywhere: the fluent k-hop
    groups accumulate a `traverse()` walk while Cypher's `[:KNOWS*1..k]` walks
    a trail, so a seed with a neighbour re-enters its own 2-hop set on one side
    and not the other. That divergence is intended, documented in the suite
    README's fairness notes, and reported as INFO rather than an error by
    `report.render_parity` — folding KgliteFluent into the storage-mode test
    above wholesale would turn a deliberate difference red.

    What is gated is the list above: every group there was published as a
    `Skip`, i.e. as a capability gap in BENCHMARKS.md, until it was shown to
    produce the Cypher column's exact result. This test is what keeps that
    claim true.
    """
    ds = dm.generate("small", 1234)
    cypher = _group_digests(KgliteCypher, ds)
    fluent = _group_digests(KgliteFluent, ds)
    for gid in FLUENT_PARITY_GROUPS:
        assert gid in fluent, f"KgliteFluent stopped running '{gid}' — the table would republish it as a capability gap"
        assert fluent[gid] == cypher[gid], (
            f"KgliteFluent diverges from KgliteCypher on group '{gid}' "
            f"({fluent[gid]} != {cypher[gid]}) — the benchmark's fluent column would publish a different result"
        )


def test_bolt_adapter_pays_the_first_exec_assessment_before_anything_is_timed(monkeypatch):
    """`KgliteBolt.available()` must execute the server binary once, untimed.

    macOS assesses a binary's code signature the first time that *inode* runs.
    Measured 2026-08-25: a freshly linked `kglite-bolt-server` spawns in
    332-540 ms and in 110-112 ms on every exec after; `build` is the only
    group that spawns it, and `run_library` refuses to repeat `build` for bolt
    keys, so the whole assessment lands inside a single published number. It
    is what made the 0.16.9 capture's Bolt "Bulk load" read 530.9 ms against a
    true ~246 ms. Deleting the pre-warm republishes that inflation silently,
    which is what this test exists to prevent.
    """
    import subprocess

    from benchmarks.competitive.graphsuite import ad_kglite

    pytest.importorskip("neo4j", reason="the bolt adapter reports unavailable without the driver")
    from tests.conftest import _BOLT_BINARY

    if not _BOLT_BINARY.exists():
        pytest.skip(f"bolt binary not built at {_BOLT_BINARY}")

    execed: list[str] = []
    real_run = subprocess.run

    def spy(cmd, *args, **kwargs):
        execed.append(str(cmd[0]))
        return real_run(cmd, *args, **kwargs)

    monkeypatch.setattr(ad_kglite.subprocess, "run", spy)
    ok, reason = ad_kglite.KgliteBolt().available()
    assert ok, reason
    assert str(_BOLT_BINARY) in execed, (
        "KgliteBolt.available() no longer execs the server binary — the first-exec "
        "code-signature assessment will be billed to the timed 'build' group"
    )
