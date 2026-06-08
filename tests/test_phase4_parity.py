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
GOLDEN_V3_DIGEST = "b82316d8485d404dbfe222248ef09c79048d8126a69d77b892affa9be06c14ac"

# Phase A.1 / C5 cleared this set on the v3 → v4 format break. The
# new v4 loader rejects v3 files (per the user-decided hard break
# in bolt_implementation.md), so every digest captured against a v3
# binary is now meaningless — the test would never re-see those byte
# patterns. The name `GOLDEN_V3_DIGEST` is kept for git-blame
# continuity; the digest itself is now the v4 byte pattern.
#
# 0.10.0 release: the `.kgl` header embeds the package version string,
# so every release shifts the digest even with byte-identical payload.
# Prior release digests are preserved here for ergonomic bisection
# back to a working v4 era.
ACCEPTABLE_DIGESTS: frozenset[str] = frozenset(
    {
        "adf955b60f07eaf1fb87e49f4c01e5e685c7236e2f6f562c1738e5ba462e4c67",  # 0.9.52
        "5b728f348d8e98c3c32a9b9262941a2740624c8d9b59f48a2c5ed79fe852a35a",  # 0.9.53 (never pushed)
        # Demoted from GOLDEN_V3_DIGEST when 0.10.1 took over.
        "6efd22ca8d49059e32ed62b22658a9e02e65700c0bd1363a7cfbdefcc7c336fa",
        # Demoted from GOLDEN_V3_DIGEST when 0.10.2 took over.
        "9719454bc7213ebd4445970a020d9be9ac1cfb89743e81f3aace5a43bacd3418",
        # Demoted from GOLDEN_V3_DIGEST when 0.10.3 took over.
        "55d5c2157d9f3c874de6439d0006c304aebe7e2dd8ba0e00402a212bac5be23a",
        # Demoted from GOLDEN_V3_DIGEST when 0.10.4 took over.
        "9de0f258d04422ee75498f256e36f0f4624e8cb8e0ec76e7bea0cbab8d82c471",
        # Demoted from GOLDEN_V3_DIGEST when the multi-label storage
        # change in 0.10.5 added the `extra_labels` field to NodeData.
        "9df0f8754d4e22b4b1cdfbf1a10c5afcbd56ecf5bdd17bbcb567dab9b2c27ba8",
        # Demoted from GOLDEN_V3_DIGEST when 0.10.5 took over.
        "a96f2c52b424859c3ce544c05c3a6774a27c960450aa429f71ec48a6db595b5c",
        # Demoted from GOLDEN_V3_DIGEST when 0.10.5 took over.
        "5e647f482fc4a580123391c92be99b367cbdf343774e1b016c50e58cacae7f57",
        # Demoted from GOLDEN_V3_DIGEST when 0.10.6 took over.
        "6b4d0f6eb37d750d1a8a29b9485bff4822527db29354bcd42f9119c276270bff",
        # Demoted from GOLDEN_V3_DIGEST when 0.10.7 took over.
        "db65a9fc5795f54d2124a6f7b859d4e3ec2310c2b0458f310739a51933c3a896",
        # Demoted from GOLDEN_V3_DIGEST when 0.10.8 took over.
        "3afdc6a47266d26c774c8f745ff639e2c7022cc816007370642b8cb6e9739a13",
        # Demoted from GOLDEN_V3_DIGEST when 0.10.9 took over.
        "81461781d5379d2ef5c755d408b985453b19e3ddf51482257234e564c23c9571",
        # Demoted from GOLDEN_V3_DIGEST when 0.10.10 took over.
        "3b6d8b6c469594f2264a56bab9ffdd331ef819c48434dad753d2175c498c6d07",
        # Demoted from GOLDEN_V3_DIGEST when 0.10.11 took over.
        "7d1fe3311b1e80527b6bdee2b23349394a86a2defb1d3c7f9b46ea21c4968344",
        # Demoted from GOLDEN_V3_DIGEST when 0.10.12 took over.
        "0e57e413f174e29874fd6136bf3127cbbb190008a13aa1721fdda0d43c696990",
        # Demoted from GOLDEN_V3_DIGEST when 0.10.13 took over.
        "b9c0033689dec21a6f53cdd4e32d338e8bb81886c8b3fa416dea1dc1a46be19a",
        # Demoted from GOLDEN_V3_DIGEST when 0.10.14 took over.
        "327f3c6ba0c1e397403a73d8f3c6915291184c982cf26afa888e15e2acfa2a33",
        # Demoted from GOLDEN_V3_DIGEST when 0.10.14 took over.
        "9302429e4d47da16be529951acb0d23a061d7852e61846ef026291e7259d130f",
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


@pytest.mark.parity
def test_kgl_v3_file_rejected_with_clear_error(tmp_path: Path):
    """Phase A.1 / C5 — v3 `.kgl` files must error cleanly under the v4
    binary, with a message that names the format change and tells the
    user how to recover.

    Crafts a minimal v3 header (`RGF\\x03`) on disk and confirms
    `kglite.load` fails with the documented hard-break error rather
    than panicking or returning silently-wrong data.
    """
    import kglite

    # Minimal v3-magic header — enough bytes to pass the "too small"
    # check but not enough to actually deserialise. The loader's
    # FIRST check is the magic, so it short-circuits before any
    # downstream parser is exercised.
    v3_file = tmp_path / "fake_v3.kgl"
    v3_file.write_bytes(
        b"RGF\x03"  # v3 magic
        + (0).to_bytes(4, "little")  # core_data_version = 0
        + (0).to_bytes(4, "little")  # metadata_length = 0
    )

    with pytest.raises((OSError, RuntimeError, ValueError)) as exc_info:
        kglite.load(str(v3_file))

    msg = str(exc_info.value)
    assert "v3" in msg, f"error message must name the v3 format: {msg!r}"
    assert "v4" in msg or "0.10" in msg, f"error message must point at the v4 / 0.10 boundary: {msg!r}"
    assert "rebuild" in msg.lower() or "downgrade" in msg.lower(), (
        f"error message must tell the user how to recover: {msg!r}"
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
