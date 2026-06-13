"""Mutation + save + reload roundtrip parity across memory / mapped / disk.

Locks in the 0.8.11 / 0.8.12 fixes that make `save → reload` preserve:
  - column_stores for types added post-`load_ntriples` via `add_nodes`
    (bug E; previously the columns.bin guard in `DirGraph::save_disk`
    dropped sidecars, so reloaded graphs returned `None` for every
    property on nodes of the new type),
  - cross-segment edges produced by seal,
  - CSR arrays when heap-backed by `reconcile_seg0_csr` (bug D),
  - clean segment layout (no stale `seg_NNN > 0` after compact-rewrite —
    bug C).

**Covered**: Cypher `SET n.prop = X` on disk-backed graphs now
persists through save + reload (bug F1 fix — node_mut_cache
clone-apply-replace flush in `DiskGraph::clear_arenas`). Asserted by
`test_cypher_set_persists_through_save_reload`.

**Covered**: DETACH DELETE on disk preserves surviving nodes' full
property values through save + reload (bug F2 fix — sidecar format
prefixes `row_count` so the loader uses the true stored row count
instead of `type_indices.len()` which undercounts tombstoned rows).
Asserted by `test_detach_delete_property_persistence` (parametrized
across every storage mode, including disk).

Run: pytest tests/test_disk_mutation_roundtrip.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import kglite
from kglite import KnowledgeGraph

STORAGE_MODES = ("memory", "mapped", "disk")


def _new_kg(mode: str, path: str | None = None) -> KnowledgeGraph:
    if mode == "memory":
        return KnowledgeGraph()
    if mode == "mapped":
        return KnowledgeGraph(storage="mapped")
    if mode == "disk":
        assert path is not None
        return KnowledgeGraph(storage="disk", path=path)
    raise ValueError(mode)


def _save_and_reload(kg: KnowledgeGraph, save_path: str, mode: str) -> KnowledgeGraph:
    kg.save(save_path)
    if mode == "memory":
        return kglite.load(save_path)
    # mapped + disk reload via the load entrypoint — the directory
    # carries the mode info.
    return kglite.load(save_path)


def _rows(rv) -> list[dict]:
    return rv.to_list() if hasattr(rv, "to_list") else list(rv)


@pytest.mark.parametrize("mode", STORAGE_MODES)
def test_add_nodes_new_type_survives_save_reload(mode, tmp_path):
    """add_nodes with a new type → save → reload → properties must
    round-trip. Bug E regression."""
    graph_path = str(tmp_path / f"g_{mode}")
    save_path = str(tmp_path / f"saved_{mode}") if mode != "disk" else graph_path
    kg = _new_kg(mode, path=graph_path if mode == "disk" else None)

    df = pd.DataFrame([{"id": f"N_{i}", "title": f"Title {i}", "age": 20 + i} for i in range(5)])
    kg.add_nodes(df, "NewEntity", "id", "title")

    pre_rows = _rows(
        kg.cypher(
            "MATCH (n:NewEntity) RETURN n.id, n.title, n.age ORDER BY n.id",
            timeout_ms=10_000,
        )
    )
    assert len(pre_rows) == 5
    assert pre_rows[0]["n.id"] == "N_0"
    assert pre_rows[0]["n.title"] == "Title 0"
    assert pre_rows[0]["n.age"] == 20

    reloaded = _save_and_reload(kg, save_path, mode)
    post_rows = _rows(
        reloaded.cypher(
            "MATCH (n:NewEntity) RETURN n.id, n.title, n.age ORDER BY n.id",
            timeout_ms=10_000,
        )
    )
    assert post_rows == pre_rows, f"{mode}: property roundtrip mismatch"


@pytest.mark.parametrize("mode", STORAGE_MODES)
def test_add_edges_survive_save_reload(mode, tmp_path):
    """add_connections for a new edge type → save → reload → edges
    must be enumerable post-reload."""
    graph_path = str(tmp_path / f"g_{mode}")
    kg = _new_kg(mode, path=graph_path if mode == "disk" else None)

    nodes_df = pd.DataFrame([{"id": f"N_{i}", "title": f"N{i}"} for i in range(4)])
    kg.add_nodes(nodes_df, "Person", "id", "title")

    edges_df = pd.DataFrame(
        [
            {"src": "N_0", "tgt": "N_1"},
            {"src": "N_1", "tgt": "N_2"},
            {"src": "N_2", "tgt": "N_3"},
        ]
    )
    kg.add_connections(edges_df, "KNOWS", "Person", "src", "Person", "tgt")

    pre_count = _rows(
        kg.cypher(
            "MATCH (:Person)-[:KNOWS]->(:Person) RETURN count(*) AS n",
            timeout_ms=10_000,
        )
    )[0]["n"]
    assert pre_count == 3

    reloaded = _save_and_reload(kg, graph_path, mode)
    post_count = _rows(
        reloaded.cypher(
            "MATCH (:Person)-[:KNOWS]->(:Person) RETURN count(*) AS n",
            timeout_ms=10_000,
        )
    )[0]["n"]
    assert post_count == 3, f"{mode}: edge count regressed after reload"


@pytest.mark.parametrize("mode", STORAGE_MODES)
def test_detach_delete_count_survives_save_reload(mode, tmp_path):
    """DETACH DELETE → save → reload → surviving count and ids must
    round-trip. Full property persistence is covered by the in-memory
    modes here; see `test_detach_delete_property_persistence_*` for
    the mode-specific property coverage."""
    graph_path = str(tmp_path / f"g_{mode}")
    kg = _new_kg(mode, path=graph_path if mode == "disk" else None)

    nodes_df = pd.DataFrame([{"id": f"P_{i}", "title": f"P{i}", "age": 20 + i} for i in range(10)])
    kg.add_nodes(nodes_df, "Thing", "id", "title")

    deleted = _rows(
        kg.cypher(
            "MATCH (n:Thing) WHERE n.id >= 'P_5' DETACH DELETE n RETURN count(n) AS n",
            timeout_ms=10_000,
        )
    )[0]["n"]
    assert deleted == 5

    reloaded = _save_and_reload(kg, graph_path, mode)
    post = _rows(
        reloaded.cypher(
            "MATCH (n:Thing) RETURN n.id ORDER BY n.id",
            timeout_ms=10_000,
        )
    )
    assert [r["n.id"] for r in post] == [f"P_{i}" for i in range(5)], (
        f"{mode}: surviving id set diverged after delete + reload"
    )


@pytest.mark.parametrize("mode", STORAGE_MODES)
def test_detach_delete_property_persistence(mode, tmp_path):
    """DETACH DELETE preserves surviving nodes' full property values across
    every storage mode, including after save/reload.

    Disk-mode regression (Bug F2): the sidecar load path derived
    `row_count` from `type_indices[type].len()` — which after a DELETE
    reflects only the live rows, while the sidecar blob still contains the
    tombstoned-but-retained rows. The mismatch made `load_packed` read
    column blobs at the wrong offsets and return garbage titles / null
    ages. Fix: the sidecar format prefixes the stored
    `ColumnStore::row_count` with a `KGLCOLv1` magic tag so the loader uses
    the true row count regardless of live-vs-stored divergence."""
    graph_path = str(tmp_path / f"g_{mode}")
    kg = _new_kg(mode, path=graph_path if mode == "disk" else None)
    kg.add_nodes(
        pd.DataFrame([{"id": f"P_{i}", "title": f"P{i}", "age": 20 + i} for i in range(10)]),
        "Thing",
        "id",
        "title",
    )
    kg.cypher(
        "MATCH (n:Thing) WHERE n.id >= 'P_5' DETACH DELETE n",
        timeout_ms=10_000,
    )
    save_path = graph_path if mode == "disk" else str(tmp_path / f"saved_{mode}")
    reloaded = _save_and_reload(kg, save_path, mode)
    post = _rows(
        reloaded.cypher(
            "MATCH (n:Thing) RETURN n.id, n.title, n.age ORDER BY n.id",
            timeout_ms=10_000,
        )
    )
    for r in post:
        idx = int(r["n.id"][2:])
        assert r["n.title"] == f"P{idx}", f"{mode}: title mismatch for {r['n.id']}"
        assert r["n.age"] == 20 + idx, f"{mode}: age mismatch for {r['n.id']}"


@pytest.mark.parametrize("mode", STORAGE_MODES)
def test_seal_then_compact_rewrite_cleans_stale_segs(mode, tmp_path):
    """Disk-specific regression: subsequent save after a seal must
    either seal or compact-rewrite without leaving stale seg_NNN dirs.
    Memory/mapped run as controls (trivially pass).
    Bug C regression."""
    graph_path = str(tmp_path / f"g_{mode}")
    kg = _new_kg(mode, path=graph_path if mode == "disk" else None)

    # Initial save.
    kg.add_nodes(
        pd.DataFrame([{"id": f"A_{i}", "title": f"A{i}"} for i in range(3)]),
        "A",
        "id",
        "title",
    )
    if mode == "disk":
        kg.save(graph_path)

    # Tail: add nodes + intra-tail edges — triggers seal on disk.
    kg.add_nodes(
        pd.DataFrame([{"id": f"B_{i}", "title": f"B{i}"} for i in range(3)]),
        "B",
        "id",
        "title",
    )
    kg.add_connections(
        pd.DataFrame([{"src": "B_0", "tgt": "B_1"}, {"src": "B_1", "tgt": "B_2"}]),
        "T",
        "B",
        "src",
        "B",
        "tgt",
    )
    if mode == "disk":
        kg.save(graph_path)

    # Now mutate existing data via add_connections (no new nodes).
    # On disk this falls to compact-rewrite; must clean the stale
    # seg_001 and persist heap-backed arrays.
    kg.add_connections(
        pd.DataFrame([{"src": "A_0", "tgt": "A_1"}]),
        "T2",
        "A",
        "src",
        "A",
        "tgt",
    )

    save_path = graph_path if mode == "disk" else str(tmp_path / f"saved_{mode}")
    reloaded = _save_and_reload(kg, save_path, mode)

    # All three edges must survive.
    t_count = _rows(reloaded.cypher("MATCH ()-[:T]->() RETURN count(*) AS n", timeout_ms=10_000))[0]["n"]
    t2_count = _rows(reloaded.cypher("MATCH ()-[:T2]->() RETURN count(*) AS n", timeout_ms=10_000))[0]["n"]
    assert t_count == 2, f"{mode}: :T edge count regressed"
    assert t2_count == 1, f"{mode}: :T2 edge count regressed"

    # On disk, confirm no stale seg_NNN > 0 remains.
    if mode == "disk":
        seg_dirs = sorted(p.name for p in Path(graph_path).iterdir() if p.is_dir() and p.name.startswith("seg_"))
        assert seg_dirs == ["seg_000"], f"compact-rewrite must remove stale segments; found {seg_dirs}"


def test_cypher_set_persists_through_save_reload(tmp_path):
    """Cypher SET on disk-backed graphs persists through save + reload.

    Locked in by the `node_mut_cache` clone-apply-replace flush in
    `DiskGraph::clear_arenas` + the DirGraph resync in
    `save_disk` (bug F1 fix). Previously this path was a no-op for
    persistence across 0.8.10 and 0.8.11; the mutation landed in an
    arena copy that `clear_arenas` dropped."""
    graph_path = str(tmp_path / "g_disk")
    kg = KnowledgeGraph(storage="disk", path=graph_path)
    kg.add_nodes(
        pd.DataFrame([{"id": f"P_{i}", "title": f"P{i}", "age": 20 + i} for i in range(5)]),
        "Person",
        "id",
        "title",
    )
    kg.cypher(
        "MATCH (n:Person) WHERE n.age < 23 SET n.age = n.age + 100",
        timeout_ms=10_000,
    )
    kg.save(graph_path)
    del kg
    reloaded = kglite.load(graph_path)
    post = _rows(
        reloaded.cypher(
            "MATCH (n:Person) RETURN n.id, n.title, n.age ORDER BY n.id",
            timeout_ms=10_000,
        )
    )
    assert [r["n.id"] for r in post] == [f"P_{i}" for i in range(5)]
    assert [r["n.title"] for r in post] == [f"P{i}" for i in range(5)]
    # P_0, P_1, P_2 match the WHERE clause and get +100; P_3, P_4 keep
    # their original ages.
    assert [r["n.age"] for r in post] == [120, 121, 122, 23, 24]


@pytest.mark.parametrize("mode", STORAGE_MODES)
def test_cypher_set_new_property_persists_through_save_reload(mode, tmp_path):
    """Cypher SET that introduces a *brand-new* property name must
    register the key in the graph's StringInterner so that save() can
    resolve it back to a string at serialize time, AND must be visible
    to subsequent reads in the same session.

    Regression for two bugs:

    - 0.8.39/0.8.40 in-memory master-path bug: the Bug-8 fix routed
      Columnar SET writes through the graph-level `column_stores`
      master to dodge the per-node Arc-clone storm, but the master path
      computed `InternedKey::from_str(property)` without registering
      the string with `graph.interner`. Symptom on Sodir-scale graphs:
      every SET-introduced property survived in-memory but vanished
      after save+reload, accompanied by
      `BUG: InternedKey N not found in StringInterner` on stderr.

    - Disk-mode in-memory-read regression: SET on a disk-backed graph
      stages writes in `node_mut_cache`, which `node_weight` (the read
      path) ignores. Subsequent reads silently returned the pre-SET
      values until the next `&mut self` op (e.g. save) drained the
      cache. Save+reload happened to recover the data because
      `clear_arenas` runs as part of save, but in-session reads went
      stale. Fixed by flushing pending writes after every mutation
      clause via `GraphWrite::flush_pending_writes`, which dispatches
      to `clear_arenas` on the disk backend.
    """
    graph_path = str(tmp_path / f"g_{mode}")
    save_path = str(tmp_path / f"saved_{mode}") if mode != "disk" else graph_path
    kg = _new_kg(mode, path=graph_path if mode == "disk" else None)

    kg.add_nodes(
        pd.DataFrame([{"id": f"P_{i}", "title": f"P{i}", "age": 20 + i} for i in range(5)]),
        "Person",
        "id",
        "title",
    )
    # value_score is a NEW property (not in the columns at add_nodes
    # time). The master-path Bug-8 fix path is the trigger.
    kg.cypher(
        "MATCH (n:Person) WHERE n.age >= 22 SET n.value_score = n.age * 10",
        timeout_ms=10_000,
    )
    # Sanity: in-memory read works.
    pre = _rows(
        kg.cypher(
            "MATCH (n:Person) WHERE n.value_score IS NOT NULL RETURN n.id, n.value_score ORDER BY n.id",
            timeout_ms=10_000,
        )
    )
    assert [(r["n.id"], r["n.value_score"]) for r in pre] == [
        ("P_2", 220),
        ("P_3", 230),
        ("P_4", 240),
    ]

    reloaded = _save_and_reload(kg, save_path, mode)
    post = _rows(
        reloaded.cypher(
            "MATCH (n:Person) WHERE n.value_score IS NOT NULL RETURN n.id, n.value_score ORDER BY n.id",
            timeout_ms=10_000,
        )
    )
    assert [(r["n.id"], r["n.value_score"]) for r in post] == [
        ("P_2", 220),
        ("P_3", 230),
        ("P_4", 240),
    ], f"value_score lost on save+reload in {mode} mode"


def test_disk_parallel_projection_node_weight_is_race_free(tmp_path):
    """Spatial-projection over a disk graph must be deterministic across
    repeated runs.

    Pre-0.9.3 disk regression: `DiskGraph::node_arena` was an
    `UnsafeCell<Vec<NodeData>>` that the docstring claimed was
    single-threaded only. In reality, the Cypher executor's projection
    phase (`return_clause::project_row`) runs `evaluate_expression`
    under `par_iter_mut` once `result_set.rows.len() >= 256`, and any
    expression that reaches the `node_weight` materialization path —
    spatial functions like `centroid` / `contains`, plus the
    spatial-fallback branch of `resolve_property` — concurrently
    pushed onto the unguarded `Vec`. A push that triggered realloc
    invalidated `&NodeData` references held by sibling Rayon tasks,
    leaking into either silent wrong-row reads (Bug A in the 0.9.2
    disk-mode regression report — ~13% NEAREST_AFEX_HUB and ~2%
    IN_AFEX_AREA edges silently dropped on the Sodir prospect graph)
    or use-after-free segfaults (Bug B in the same report — `BUG:
    InternedKey N not found in StringInterner` on stderr).

    Fix: `node_arena` is now `Mutex<Vec<Box<NodeData>>>`, mirroring
    the long-standing `edge_arena` pattern. The Box gives stable
    heap pointers across Vec growth and the Mutex serialises pushes.

    This test creates a disk graph with > 256 polygon-bearing nodes
    (the Rayon threshold) and runs a `centroid` projection 8 times,
    asserting identical row counts. Pre-fix: counts varied run-to-run
    by 1–3 rows. Post-fix: deterministic.
    """
    graph_path = str(tmp_path / "g_disk_race")
    kg = KnowledgeGraph(storage="disk", path=graph_path)

    # 600 tiny squares — well above the RAYON_THRESHOLD of 256.
    # Each polygon is offset so centroids are all distinct.
    rows = []
    for i in range(600):
        cx = -180.0 + (i * 0.5)
        cy = -45.0 + (i * 0.05)
        wkt = f"POLYGON(({cx} {cy}, {cx + 0.1} {cy}, {cx + 0.1} {cy + 0.1}, {cx} {cy + 0.1}, {cx} {cy}))"
        rows.append({"id": f"P_{i}", "title": f"P{i}", "wkt_geometry": wkt})

    kg.add_nodes(
        pd.DataFrame(rows),
        "Region",
        "id",
        "title",
        column_types={"wkt_geometry": "geometry"},
    )
    kg.save(graph_path)

    reloaded = kglite.load(graph_path)
    counts: list[int] = []
    for _ in range(8):
        out = _rows(
            reloaded.cypher(
                "MATCH (r:Region) WHERE r.wkt_geometry IS NOT NULL WITH r, centroid(r) AS c RETURN count(c) AS n"
            )
        )
        counts.append(out[0]["n"])
    # All 8 runs must report the same count. Pre-fix: ~1-3 row variance.
    assert all(c == counts[0] for c in counts), f"centroid() projection over disk graph is non-deterministic: {counts}"
    # And the count itself must equal the input row count — losing any
    # rows here would mean a parallel write into the spatial cache
    # raced and a sibling task observed a stale entry.
    assert counts[0] == 600, f"expected 600 centroids, got {counts}"


def test_disk_parallel_projection_no_interner_corruption(tmp_path, capfd):
    """Same race surface as `…_node_weight_is_race_free`, but watching
    a different victim: `&NodeData` carries an `InternedKey`-tagged
    Columnar property storage. When the Vec realloc invalidated the
    reference, downstream `interner.resolve(key)` would sometimes hit
    a key the interner didn't know — surfacing as `BUG: InternedKey N
    not found in StringInterner` on stderr (or, with bad timing, a
    SIGSEGV). The fix removes the race; the test asserts no such
    BUG line ever appears across many repeated parallel projections.
    """
    graph_path = str(tmp_path / "g_disk_interner_race")
    kg = KnowledgeGraph(storage="disk", path=graph_path)

    rows = []
    for i in range(600):
        cx = -180.0 + (i * 0.5)
        cy = -45.0 + (i * 0.05)
        wkt = f"POLYGON(({cx} {cy}, {cx + 0.1} {cy}, {cx + 0.1} {cy + 0.1}, {cx} {cy + 0.1}, {cx} {cy}))"
        rows.append({"id": f"P_{i}", "title": f"P{i}", "wkt_geometry": wkt})

    kg.add_nodes(
        pd.DataFrame(rows),
        "Region",
        "id",
        "title",
        column_types={"wkt_geometry": "geometry"},
    )
    kg.save(graph_path)

    # Reload and hammer the projection a few times. The Rust-side
    # `eprintln!` for "BUG: InternedKey N not found" goes through C
    # stdio (stderr), which capfd captures.
    reloaded = kglite.load(graph_path)
    for _ in range(8):
        list(
            reloaded.cypher(
                "MATCH (r:Region) WHERE r.wkt_geometry IS NOT NULL WITH r, centroid(r) AS c WHERE c IS NULL RETURN r.id"
            )
        )
    out, err = capfd.readouterr()
    bug_lines = [ln for ln in err.splitlines() if "BUG: InternedKey" in ln and "not found" in ln]
    assert not bug_lines, f"Disk parallel projection still racing on the StringInterner: {bug_lines[:3]}"


@pytest.mark.parametrize("mode", ["mapped", "disk"])
def test_cypher_set_visible_on_mmap_backed_columnstore(mode, tmp_path):
    """Cypher SET on a mmap-backed ColumnStore must be visible on
    subsequent reads.

    Pre-0.9.4 Bug C: when a ColumnStore is constructed via
    `from_mmap_store` — the path `load_ntriples` always takes for
    mapped / disk targets — every read method (`get`, `get_title`,
    `str_prop_eq`, `row_properties`) short-circuited to the mmap-backed
    store at the top of the function. But `set()` and `set_title()`
    wrote into `self.columns` / `self.title_column` — fields the read
    short-circuit bypassed. So `MATCH (a) SET a.x = 1 RETURN count(a)`
    returned the expected `count = 1` (proving the SET matched + the
    writer reported success), but a follow-up `RETURN a.x` returned
    `None`. Surfaced by the cross-mode consistency check on a
    Wikidata-style graph: memory mode reported the SET marker
    correctly, mapped + disk reported the property as NULL.

    Fix: `get` / `get_title` / `str_prop_eq` / `row_properties` consult
    the in-memory overlay first and only fall through to the mmap-backed
    read when no override exists. `set_title` lazily promotes the mmap-
    backed title column into a Mixed in-memory column on first override
    so the dense title column doesn't have to be allocated up-front.

    `add_nodes` + save + reload doesn't take the from_mmap_store path
    today (column blobs reload as in-memory `columns: Vec<TypedColumn>`
    with `mmap_store: None`), so the bug requires `load_ntriples` to
    reproduce. The test writes a tiny `.nt` inline, loads it into
    mapped + disk, applies SET, and reads back.
    """
    nt_path = tmp_path / "tiny.nt"
    # 5 Q-entities each with a title (rdfs:label@en) + P31 instance-of
    # so the type rename surfaces them. load_ntriples in mapped / disk
    # mode populates ColumnStores via from_mmap_store regardless of
    # input size — that's enough to trigger Bug C.
    nt_lines = []
    for i in range(1, 6):
        nt_lines.append(
            f'<http://www.wikidata.org/entity/Q{i}> <http://www.w3.org/2000/01/rdf-schema#label> "orig{i}"@en .'
        )
        nt_lines.append(
            f"<http://www.wikidata.org/entity/Q{i}> "
            f"<http://www.wikidata.org/prop/direct/P31> "
            f"<http://www.wikidata.org/entity/Q5> ."
        )
    nt_path.write_text("\n".join(nt_lines) + "\n")

    if mode == "mapped":
        kg = KnowledgeGraph(storage="mapped")
    else:
        kg = KnowledgeGraph(storage="disk", path=str(tmp_path / "g_disk"))
    kg.load_ntriples(str(nt_path), languages=["en"], verbose=False)

    pre = _rows(kg.cypher("MATCH (a {nid: 'Q3'}) RETURN a.title AS t"))
    assert pre == [{"t": "orig3"}], f"{mode}: load_ntriples didn't materialise Q3: {pre}"

    # SET an existing property (title — set_title's lazy-promotion
    # branch) AND a new property name (bench_marker — the schema-
    # extension branch in `set`). The SET clause itself reports
    # success either way; pre-fix the regression was only on the read.
    rep = _rows(kg.cypher("MATCH (a {nid: 'Q3'}) SET a.title = 'updated', a.bench_marker = 1 RETURN count(a) AS n"))
    assert rep == [{"n": 1}], f"{mode}: SET clause didn't match Q3: {rep}"

    rb = _rows(kg.cypher("MATCH (a {nid: 'Q3'}) RETURN a.title AS t, a.bench_marker AS m"))
    assert rb == [{"t": "updated", "m": 1}], f"{mode}: SET visibility regression on mmap-backed ColumnStore: {rb}"

    # Sibling row must be untouched (overlay must not leak across rows).
    other = _rows(kg.cypher("MATCH (a {nid: 'Q4'}) RETURN a.title AS t, a.bench_marker AS m"))
    assert other == [{"t": "orig4", "m": None}], f"{mode}: SET overlay leaked into a non-target row: {other}"


def test_register_new_conn_type_preserves_existing_type_lookups(tmp_path):
    """add_connections of a brand-new edge type on a *loaded* disk graph
    must not break MATCH queries on existing edge types.

    Pre-0.9.5 Bug C (separate from the SET-visibility bug also called
    Bug C in the prior release report — naming collision). After
    `kglite.load(disk_graph_path)`, the `connection_types` HashSet
    cache was empty (the disk loader, unlike `load_v3`, never called
    `build_connection_types_cache`). `has_connection_type("EXISTING")`
    fell through to the `connection_type_metadata` branch and
    correctly returned True.

    The moment any code path called `register_connection_type("NEW")`
    — e.g. via `add_connections` — the `connection_types` set went
    from empty to {NEW}, which flipped `has_connection_type` into the
    "use cache" fast path. From that point on,
    `has_connection_type("EXISTING")` returned False, and every typed
    MATCH (`MATCH (a)-[:EXISTING]->(b)`) returned 0 rows because the
    pattern matcher's early-exit "skip iteration when the conn type
    doesn't exist" check fired. Surfaced on the Sodir
    load-then-enhance flow: stages that traverse FK edges
    (`OF_DISCOVERY`, `OF_FIELD`, `OF_PROSPECT`, `IN_PLAY`) all
    silently produced 0 rows after the first `add_connections` of the
    re-enhance pass.

    Fix: backfill the cache from `connection_type_metadata` in
    `load_disk_dir`, AND make `register_connection_type` lazy-build
    the cache when called against an empty set with non-empty metadata.
    Either layer alone closes the bug; both keeps the cache
    authoritative for any code path.
    """
    graph_path = str(tmp_path / "g_disk")
    kg = KnowledgeGraph(storage="disk", path=graph_path)
    # Two node types + one edge type. Save + reload to mimic the
    # production "kglite.load(saved_graph)" workflow.
    kg.add_nodes(
        pd.DataFrame([{"id": f"P_{i}", "title": f"P{i}"} for i in range(5)]),
        "Person",
        "id",
        "title",
    )
    kg.add_nodes(
        pd.DataFrame([{"id": f"C_{i}", "title": f"C{i}"} for i in range(3)]),
        "Company",
        "id",
        "title",
    )
    kg.add_connections(
        pd.DataFrame([{"src": f"P_{i}", "tgt": "C_0"} for i in range(5)]),
        "WORKS_AT",
        "Person",
        "src",
        "Company",
        "tgt",
    )
    kg.save(graph_path)
    reloaded = kglite.load(graph_path)

    # Sanity: existing edge type findable on a fresh load.
    pre = _rows(reloaded.cypher("MATCH (p:Person)-[:WORKS_AT]->(c:Company) RETURN count(*) AS n"))
    assert pre == [{"n": 5}], f"existing edge type not findable post-load: {pre}"

    # Trigger: register a brand-new edge type. Pre-fix, this flipped
    # has_connection_type from "metadata fallback" to "cache lookup"
    # mode, and the cache only contained the new key.
    reloaded.add_connections(
        pd.DataFrame([{"src": "P_0", "tgt": "P_1"}]),
        "FRIENDS_WITH",
        "Person",
        "src",
        "Person",
        "tgt",
    )

    # Existing edge type must still be findable.
    post = _rows(reloaded.cypher("MATCH (p:Person)-[:WORKS_AT]->(c:Company) RETURN count(*) AS n"))
    assert post == [{"n": 5}], f"adding a new edge type broke MATCH on the existing edge type: {post}"

    # And the new edge is also findable.
    new_edges = _rows(reloaded.cypher("MATCH (a:Person)-[:FRIENDS_WITH]->(b:Person) RETURN count(*) AS n"))
    assert new_edges == [{"n": 1}], f"new edge type lost: {new_edges}"

    # And the unanchored read is consistent (this never broke pre-fix
    # but covering it lets a single test catch any regression in either
    # the cache-fast-path or the metadata-fallback branch).
    by_type = _rows(reloaded.cypher("MATCH ()-[r]->() RETURN type(r) AS t, count(*) AS n ORDER BY t"))
    assert by_type == [
        {"t": "FRIENDS_WITH", "n": 1},
        {"t": "WORKS_AT", "n": 5},
    ], f"per-type edge counts diverged after register: {by_type}"
