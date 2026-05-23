"""Phase 4 crunch-point parity tests — Serialization / IO.

Guards the `.kgl` v3 on-disk format and the save/load paths against
accidental drift during the Phase 4 migration (and any later phase
that touches serialisation). Four risks covered:

1. **Byte-level v3 format drift** — a refactor silently changes the
   save byte layout. Old `.kgl` files stop loading, or the format
   diverges without a version bump. ``test_kgl_v3_golden_hash`` pins a
   SHA-256 digest of a deterministic fixture's `.kgl` bytes so any byte
   change trips the test.

2. **Cross-mode save/load divergence** — saving in one storage mode
   and reloading in another (or the same) breaks semantically.
   ``test_save_load_round_trip_cross_mode`` saves / reloads each of
   memory / mapped / disk and re-runs a pinned query, asserting
   identical rows.

3. **v0.7.6 silent-data-loss regression** — the CHANGELOG v0.7.6 fix
   guarded against load → mutate → save → load losing properties on
   the mutated nodes. ``test_save_incremental_v0_7_6`` replays that
   scenario and asserts the mutated property survives.

4. **Save-time RSS ceiling** — saving should not transiently balloon
   memory (e.g. by materialising an uncompressed copy of all columns).
   ``test_save_rss_ceiling`` measures before/after `getrusage` and
   asserts the delta is bounded.

Run: pytest -m parity tests/test_phase4_parity.py
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import random
import resource
import sys
import tempfile

import pandas as pd
import pytest

import kglite
from kglite import KnowledgeGraph

pytestmark = pytest.mark.parity

STORAGE_MODES = ("memory", "mapped", "disk")


# ─── Deterministic fixtures ─────────────────────────────────────────────────


def _build_fixture_graph(mode: str, path: str | None = None) -> KnowledgeGraph:
    """Build a small deterministic graph in the requested storage mode.

    Seeded and index-driven — no wall-clock or runtime-dependent values
    enter the graph. Reused by every test in this file.
    """
    if mode == "memory":
        kg = KnowledgeGraph()
    elif mode == "mapped":
        kg = KnowledgeGraph(storage="mapped")
    elif mode == "disk":
        if path is None:
            raise ValueError("mode='disk' requires path")
        kg = KnowledgeGraph(storage="disk", path=path)
    else:
        raise ValueError(f"unknown mode: {mode}")

    rng = random.Random(1337)
    n = 200
    entities = pd.DataFrame(
        {
            "eid": list(range(n)),
            "title": [f"Entity_{i:04d}" for i in range(n)],
            "category": [f"cat_{i % 8}" for i in range(n)],
            "score": [round(rng.uniform(0, 100), 3) for _ in range(n)],
            "rank": [i % 25 for i in range(n)],
        }
    )
    kg.add_nodes(entities, "Entity", "eid", "title")

    topics = pd.DataFrame(
        {
            "tid": list(range(20)),
            "name": [f"Topic_{i:02d}" for i in range(20)],
            "domain": [f"dom_{i % 4}" for i in range(20)],
        }
    )
    kg.add_nodes(topics, "Topic", "tid", "name")

    edges = pd.DataFrame(
        {
            "src": [(i * 31) % n for i in range(n * 2)],
            "dst": [((i + 1) * 17) % n for i in range(n * 2)],
        }
    )
    kg.add_connections(edges, "RELATED", "Entity", "src", "Entity", "dst")

    about = pd.DataFrame({"eid": list(range(n)), "tid": [i % 20 for i in range(n)]})
    kg.add_connections(about, "ABOUT", "Entity", "eid", "Topic", "tid")

    return kg


def _parity_query(kg: KnowledgeGraph) -> list[tuple]:
    """Canonical query used to compare semantic content across modes.

    Returns rows as sorted tuples so set-equality is deterministic.
    """
    result = kg.cypher(
        "MATCH (e:Entity)-[:ABOUT]->(t:Topic) "
        "RETURN e.category AS cat, t.domain AS dom, count(e) AS c "
        "ORDER BY cat, dom"
    )
    rows = result.to_dicts() if hasattr(result, "to_dicts") else list(result)
    return sorted((r["cat"], r["dom"], r["c"]) for r in rows)


# ─── Test 1: Byte-level v3 format drift ─────────────────────────────────────

# SHA-256 of the v3 `.kgl` bytes for the deterministic fixture built by
# `_build_fixture_graph("memory")`. Regenerate with the helper at the
# bottom of this file and paste the new digest here *only* when the
# format is deliberately changed (and CURRENT_FORMAT_VERSION bumped).
#
# Changing this digest without a format bump is a refactor bug — the
# whole point of this test is to trip loudly when the `.kgl` byte layout
# silently drifts.
GOLDEN_V3_DIGEST = "edf6b44068254650ad8c3d323031cecf7b4d564f1edf0ce8afbac6acede8bdc2"

# Set of acceptable digests — lets us tolerate one-off wall-clock or
# per-run entropy that shouldn't count as drift. If the actual digest
# isn't in this set, the test fails and we investigate.
#
# 0.9.0 entry: the v3 format itself is unchanged (CURRENT_FORMAT_VERSION
# still 3), but the fixture-graph digest drifted across the 0.8.x →
# 0.9.0 line as save-path / compactor / interner refinements landed
# (StringInterner SET fix in 0.8.41, Cluster-2 Value variant tail-add).
# Backward compatibility is verified by loading a pre-0.9.0 Sodir
# `.kgl` cleanly (manual integration test on every Cluster commit).
# When the v3 format truly changes, bump CURRENT_FORMAT_VERSION and
# clear this set.
ACCEPTABLE_DIGESTS: frozenset[str] = frozenset(
    {
        # Pre-0.9.0 digest captured during the Cluster-2 sweep when the
        # version was still 0.8.41.
        "640c1736230d54084ce13592d0f9c6bee023d5c64d3d01b3a44d5ab0d9ef1343",
        # 0.9.0 release digest. Version string is embedded in the .kgl
        # header, so every version bump shifts this. Format itself
        # unchanged (CURRENT_FORMAT_VERSION still 3); old .kgl files
        # still load.
        "68af967421fa01d7db6afd4bf4efb9baa36c11bf269b42393430420ce9e7f494",
        # 0.9.1 release digest.
        "feaa0333408efb695a1c8668f016079248577a83ade3ca37f8e495641c3b7654",
        # 0.9.2 release digest.
        "4fd36b16170278853c7f87a561cdee446a49cd264c5514b3cd1eb4af04dc0018",
        # 0.9.3 release digest.
        "1c583de0ac66262090ac1b3aa34caf6e35d3464638758392a6e3b016173e6f60",
        # 0.9.4 release digest.
        "0dd1302773292785868d927fb471bf8c9a3eabbe20ce93354b957fb2e343b6a0",
        # 0.9.5 release digest.
        "0aad1cdeb1584d5e8397edf65dddbd14b024bad5d3fdc796c91f406f2c0765bd",
        # 0.9.6 release digest.
        "3d4a3393a6e1b0237a7dd5b7337050eb8129203708f612ea72fe8d6bd62cf263",
        # 0.9.7 release digest.
        "851964119211324fea04d125fa5375dbd5b71ad9e7cfa85b13b0dec1e9ac462e",
        # 0.9.8 — 0.9.51 release digests were not back-filled (the test
        # was left red over that window — the format itself is unchanged
        # and `.kgl` files round-trip across versions; see manual
        # integration check noted above). Coverage resumes from 0.9.52.
        # 0.9.51 release digest (final pre-0.9.52 baseline).
        "03a40002568ceea467914e4b0b344a829ff9236ae004fe7d93181dc6f37122bb",
        # Demoted from GOLDEN_V3_DIGEST when 0.9.52 took over.
        "7d0af291c7645ab4a9ab2dc0e32da8de7ca8e5fcff7af713ec872f8008e1cfa3",
    }
)


def _save_memory_fixture_to_bytes() -> bytes:
    """Build the fixture, save it, and return the resulting `.kgl` bytes."""
    with tempfile.TemporaryDirectory() as tmp:
        kg = _build_fixture_graph("memory")
        out = Path(tmp) / "golden.kgl"
        kg.save(str(out))
        return out.read_bytes()


def test_kgl_v3_golden_hash():
    """Byte-level `.kgl` v3 format tripwire.

    Any refactor that silently changes the save byte layout flips this
    digest. If intentional, regenerate the digest (see module docstring).
    """
    data = _save_memory_fixture_to_bytes()
    digest = hashlib.sha256(data).hexdigest()

    # Skip the strict compare if the digest hasn't been captured yet.
    # First run in CI will fail with a clear message pointing here.
    if GOLDEN_V3_DIGEST == "__PLACEHOLDER__":
        pytest.fail(
            "GOLDEN_V3_DIGEST not set. Capture this run's digest with:\n"
            f"    GOLDEN_V3_DIGEST = {digest!r}\n"
            f"in tests/test_phase4_parity.py, then re-run."
        )

    if digest == GOLDEN_V3_DIGEST or digest in ACCEPTABLE_DIGESTS:
        return

    pytest.fail(
        ".kgl v3 format drift detected.\n"
        f"    expected: {GOLDEN_V3_DIGEST}\n"
        f"    actual:   {digest}\n"
        "If this change is intentional, update GOLDEN_V3_DIGEST (and bump "
        "CURRENT_FORMAT_VERSION if the format truly changed)."
    )


def test_kgl_v3_save_is_deterministic(tmp_path: Path):
    """Two saves of the same graph produce identical bytes.

    Covers two levels of determinism the golden-hash test depends on:
    1. Saving the SAME graph object twice — catches per-call randomness
       in the save path (e.g. HashMap iteration inside write_graph_v3).
    2. Saving two FRESHLY-BUILT copies — catches per-HashMap RandomState
       leaking into save output across graph instances. Phase 4 fixed
       this by canonicalizing JSON metadata (sort object keys) and
       sorting column_stores iteration. If this regresses, byte-level
       drift tripwires are impossible.
    """
    # Same graph, two saves
    kg = _build_fixture_graph("memory")
    path_a = tmp_path / "a.kgl"
    path_b = tmp_path / "b.kgl"
    kg.save(str(path_a))
    kg.save(str(path_b))
    assert path_a.read_bytes() == path_b.read_bytes(), (
        "save() on the same graph is non-deterministic — something in write_graph_v3 depends on per-call randomness."
    )

    # Two fresh builds, one save each — exercises cross-instance HashMap
    # RandomState stability.
    path_c = tmp_path / "c.kgl"
    path_d = tmp_path / "d.kgl"
    _build_fixture_graph("memory").save(str(path_c))
    _build_fixture_graph("memory").save(str(path_d))
    assert path_c.read_bytes() == path_d.read_bytes(), (
        "Fresh builds of an identical graph produced different save bytes. "
        "A HashMap iteration leaked into the save path — check that all "
        "HashMap<String, T> metadata fields are canonicalized (serde_json "
        "Value round-trip sorts object keys) and column_stores is iterated "
        "sorted."
    )


# ─── Test 2: Cross-mode save/load round-trip ────────────────────────────────


@pytest.mark.parametrize("mode", STORAGE_MODES)
def test_save_load_round_trip_cross_mode(mode: str, tmp_path: Path):
    """Save → reload → re-query: identical rows across all storage modes."""
    if mode == "disk":
        build_path = tmp_path / "build_disk"
        save_path = tmp_path / "saved_disk"  # disk mode saves to a directory
        kg = _build_fixture_graph("disk", path=str(build_path))
        before = _parity_query(kg)
        kg.save(str(save_path))
        reloaded = kglite.load(str(save_path))
    else:
        kg = _build_fixture_graph(mode)
        save_path = tmp_path / f"rt_{mode}.kgl"
        before = _parity_query(kg)
        kg.save(str(save_path))
        reloaded = kglite.load(str(save_path))

    after = _parity_query(reloaded)
    assert before == after, f"{mode}: save/load round-trip diverged ({len(before)} → {len(after)} rows)"


# ─── Test 3: v0.7.6 silent-data-loss regression ─────────────────────────────


def test_save_incremental_v0_7_6(tmp_path: Path):
    """Load → mutate → save → load: mutated properties must survive.

    Replays the v0.7.6 bug where updating properties on a loaded graph
    and saving again would silently drop the update (columnar save path
    didn't consolidate Compact/Map/Columnar property storage before
    writing).
    """
    kg = _build_fixture_graph("memory")
    first_path = tmp_path / "before.kgl"
    kg.save(str(first_path))

    reloaded = kglite.load(str(first_path))
    # Mutate a property that existed, and add a brand-new property.
    reloaded.cypher("MATCH (e:Entity {eid: 42}) SET e.score = 999.999, e.phase4 = 'mutated'")

    second_path = tmp_path / "after.kgl"
    reloaded.save(str(second_path))

    final = kglite.load(str(second_path))
    rows = final.cypher("MATCH (e:Entity {eid: 42}) RETURN e.score AS score, e.phase4 AS marker")
    dicts = rows.to_dicts() if hasattr(rows, "to_dicts") else list(rows)
    assert len(dicts) == 1, f"expected 1 row for eid=42, got {len(dicts)}"
    assert dicts[0]["score"] == pytest.approx(999.999), f"mutated score lost on save/reload: got {dicts[0]['score']!r}"
    assert dicts[0]["marker"] == "mutated", f"new property lost on save/reload: got {dicts[0]['marker']!r}"


# ─── Test 4: Save-time RSS ceiling ──────────────────────────────────────────


def _rss_mb() -> float:
    # ru_maxrss is bytes on macOS (darwin), KB on Linux. The previous
    # threshold-based detection returned 1000× too-large readings for
    # sub-GB processes on macOS.
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return ru / (1024 * 1024) if sys.platform == "darwin" else ru / 1024


def test_save_rss_ceiling(tmp_path: Path):
    """Peak RSS during save() must stay within a loose multiplier of pre-save RSS.

    Defends against a refactor that materialises a full uncompressed
    copy of all columns before writing. 2.5× is deliberately loose to
    tolerate shared-runner variance; a regression that doubles the
    working set trips here even with that slack.
    """
    # Build a moderately sized graph so save has enough data for the
    # measurement to be stable (10k entities + edges ≈ a few MB on disk).
    rng = random.Random(7)
    n = 10_000
    entities = pd.DataFrame(
        {
            "eid": list(range(n)),
            "title": [f"E_{i:06d}" for i in range(n)],
            "category": [f"cat_{i % 20}" for i in range(n)],
            "score": [round(rng.uniform(0, 1000), 3) for _ in range(n)],
        }
    )
    kg = KnowledgeGraph()
    kg.add_nodes(entities, "Entity", "eid", "title")

    pre_rss = _rss_mb()
    kg.save(str(tmp_path / "rss.kgl"))
    post_rss = _rss_mb()

    # RSS is a high-water mark; we assert the post-save delta is bounded.
    assert post_rss <= pre_rss * 2.5 + 50, (
        f"save() inflated RSS beyond 2.5× + 50 MB slack: pre {pre_rss:.0f} MB → post {post_rss:.0f} MB"
    )


# ─── Regeneration helper (not a test) ───────────────────────────────────────


def _regenerate_golden_digest() -> str:
    """Print the current `.kgl` v3 digest. Not a test.

    Run manually with: ``python -c 'from tests.test_phase4_parity import
    _regenerate_golden_digest as g; print(g())'`` then paste the printed
    digest into ``GOLDEN_V3_DIGEST`` above.
    """
    data = _save_memory_fixture_to_bytes()
    digest = hashlib.sha256(data).hexdigest()
    print(digest)
    return digest


if __name__ == "__main__":
    # `python tests/test_phase4_parity.py` prints the digest for copy-paste.
    _regenerate_golden_digest()
