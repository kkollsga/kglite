"""Silent-property-loss round-trip matrix — permanent regression suite.

This suite systematically classifies how every kind of property VALUE fares
across every ingestion path, both targets (node / edge property), every
storage mode, and persistence (in-memory vs `.kgl` save→reload). It exists to
lock down a known bug *class*: cases where a property value silently
**degrades** (read back as ``None`` or a stringified scalar) instead of either
round-tripping exactly or failing loudly at write time.

The contract for every cell is binary:

    A write either (a) round-trips the value exactly, or (b) raises loudly at
    write time. A write that *succeeds* but reads back ``None`` / a degraded
    representation is a FAILING cell and is marked ``xfail(strict=True)`` with
    a reason naming the responsible layer.

Strict xfails are deliberate: the day a fix lands, the xfail turns into an
XPASS and *fails the suite*, forcing the reason string (and this file) to be
updated in the same change. That is how the matrix stays honest.

## Findings (2026-07-09, updated as fixes landed)

Fixed (the cells below now round-trip and pass — kept as passing rows):

* **Top-level dict/map via `add_nodes` / `add_connections`** — was silently
  **stringified** (``{'k': 1}`` → ``"{'k': 1}"``). Fixed: an object column
  whose first cell is a dict is typed ``Map`` and each cell converts via the
  recursive ``py_value_to_value``. (`datatypes/py_in.rs`.)
* **Top-level dict/map via `from_records`** — was silently **``None``**. Fixed:
  ``DataFrame::from_cypher_rows`` type inference now maps ``Value::Map`` →
  ``ColumnType::Map`` (``json_to_value`` already builds the map). (`values.rs`.)
* **Python ``datetime`` via `add_nodes` / `add_connections`** — was silently
  **truncated to date-only**. Fixed: a ``datetime64`` column with any nonzero
  time-of-day is typed ``Timestamp`` (full precision); a pure-midnight column
  stays date-only ``DateTime``. (`datatypes/py_in.rs`.)

* **Chained-dot into a map — ``n.m.k``** — was returning ``None`` while bracket
  subscript ``n.m['k']`` worked (node and edge). Fixed: ``ExprPropertyAccess``
  now has a ``Value::Map`` arm mirroring ``map_subscript``
  (`executor/expression.rs`). Was Fact #2 in
  ``dev-docs/plans/rev-aware-code-graphs.md`` (B.2 design).

No reachable *silent*-degradation cells remain. Loud-not-silent (acceptable
half of the contract):

* **`add_connections` now matches `add_nodes` default property semantics.**
  Omitting ``columns=`` preserves every non-skipped DataFrame column; an
  explicit ``columns=[...]`` remains a whitelist and ``skip_columns=[...]``
  remains an exclusion filter.

Notes on the two "known" bugs this suite was seeded from:

1. *"Node ``Value::Map`` props saved as NULL to ``.kgl``"*
   (`column_store.rs:1073`). This encoder (``serialize_overflow_value``) is
   **not reached** by any Cypher-CREATE-then-save path in any storage mode
   (memory / mapped / disk), nor by ``to_subgraph`` — those all preserve maps
   (see ``test_map_survives_kgl_all_storage_modes``, a passing guard). The
   overflow-bag encoder is exercised only by the columnar bulk builders
   (RDF / n-triples / subgraph-streaming writer), so the bug is **latent**:
   real in code, dormant for the common Python surface. If a future change
   routes user maps through that encoder, the passing guard here becomes the
   canary.
2. *"`add_connections` drops list edge columns"* is fixed: default,
   whitelist, and exclusion semantics are covered below and through all
   storage modes.
"""

from __future__ import annotations

import datetime
import warnings

import numpy as np
import pandas as pd
import pytest

import kglite

# ── Value kinds ─────────────────────────────────────────────────────────────
_DATE = datetime.date(2020, 1, 2)
_DT = datetime.datetime(2020, 1, 2, 3, 4, 5)
_POINT_READBACK = {"latitude": 1.0, "longitude": 2.0}

# name -> (python value | None if not python-constructible, cypher literal,
#          expected readback)
KINDS: dict[str, tuple[object, str, object]] = {
    "str": ("hello", "'hello'", "hello"),
    "int": (42, "42", 42),
    "float": (3.5, "3.5", 3.5),
    "bool": (True, "true", True),
    "none": (None, "null", None),
    "date": (_DATE, "date('2020-01-02')", _DATE),
    "datetime": (_DT, "datetime('2020-01-02T03:04:05')", _DT),
    "point": (None, "point(1.0,2.0)", _POINT_READBACK),
    "list_str": (["a", "b", "c"], "['a','b','c']", ["a", "b", "c"]),
    "list_int": ([1, 2, 3], "[1,2,3]", [1, 2, 3]),
    "nested_list": ([[1, 2], [3, 4]], "[[1,2],[3,4]]", [[1, 2], [3, 4]]),
    "map": ({"k": 1, "j": "x"}, "{k:1, j:'x'}", {"k": 1, "j": "x"}),
    "dict_in_list": ([{"a": 1}, {"a": 2}], "[{a:1},{a:2}]", [{"a": 1}, {"a": 2}]),
    "list_in_dict": ({"xs": [1, 2, 3]}, "{xs:[1,2,3]}", {"xs": [1, 2, 3]}),
}

# Kinds with no Python value literal: only reachable through Cypher point().
_CYPHER_ONLY = {"point"}
# Kinds that from_records cannot carry as a Python object (JSON has no date).
_JSON_UNSERIALISABLE = {"date", "datetime"}
# Collection kinds need an object-dtype pandas column with per-cell assignment;
# scalars use their natural inferred dtype (an object column of scalars is a
# pandas footgun that stringifies ints/floats/datetimes — not a value-kind bug).
_COLLECTION = {"list_str", "list_int", "nested_list", "map", "dict_in_list", "list_in_dict"}


def _matches(expected: object, got: object) -> bool:
    """Tolerant value comparison for round-trip assertions.

    Temporal values may read back as ISO strings or as date/datetime objects
    depending on the path; either is a faithful round-trip of the *value*.
    """
    if isinstance(expected, datetime.datetime):
        return got == expected or str(got).startswith("2020-01-02T03:04:05")
    if isinstance(expected, datetime.date):
        return got == expected or str(got).startswith("2020-01-02")
    return expected == got


# ── Storage-mode helper ─────────────────────────────────────────────────────
def _new_kg(mode: str, tmp_path) -> kglite.KnowledgeGraph:
    if mode == "memory":
        return kglite.KnowledgeGraph()
    if mode == "mapped":
        return kglite.KnowledgeGraph(storage="mapped")
    if mode == "disk":
        return kglite.KnowledgeGraph(storage="disk", path=str(tmp_path / "diskg"))
    raise ValueError(mode)


# ── Node ingestion paths ────────────────────────────────────────────────────
def _one_col_series(val: object) -> pd.Series:
    """A one-row object-dtype Series holding a single collection value.

    Used for list/map columns, which must be object-dtype with the collection
    placed per-cell so pandas doesn't try to broaden it into a typed column.

    Built via a numpy object array rather than ``s.iloc[0] = val`` — on pandas
    3.x the latter *broadcasts* a dict into a ``pd.Series`` cell (label
    alignment), so a ``{'k': 1}`` value would never reach ingestion as a dict.
    A numpy object array stores the value verbatim, matching what a real user
    gets from ``pd.DataFrame({'p': [{'k': 1}]})`` or ``df['p'] = [{'k': 1}]``.
    """
    arr = np.empty(1, dtype=object)
    arr[0] = val
    return pd.Series(arr)


def _prop_series(kind: str, val: object) -> pd.Series:
    """Build the ``p`` column for ``kind`` using its natural pandas dtype.

    Scalars get their inferred dtype (int64 / float64 / datetime64 / bool /
    object-of-str); collections get an object column via :func:`_one_col_series`.
    """
    if kind in _COLLECTION:
        return _one_col_series(val)
    return pd.Series([val])


def ingest_node(kind: str, path: str) -> kglite.KnowledgeGraph:
    """Create one ``:N {id:1, p:<value>}`` node via ``path``. May raise."""
    val, lit, _ = KINDS[kind]
    kg = kglite.KnowledgeGraph()
    if path == "add_nodes":
        df = pd.DataFrame({"id": [1]})
        df["p"] = _prop_series(kind, val)
        kg.add_nodes(df, "N", "id")
    elif path == "from_records":
        spec = {"nodes": [{"type": "N", "id_field": "id", "records": [{"id": 1, "p": val}]}]}
        kg = kglite.from_records(spec)
    elif path == "cypher_literal":
        kg.cypher(f"CREATE (n:N {{id:1, p:{lit}}})")
    elif path == "cypher_set":
        kg.cypher("CREATE (n:N {id:1})")
        if kind in _CYPHER_ONLY:
            kg.cypher(f"MATCH (n:N {{id:1}}) SET n.p = {lit}")
        else:
            kg.cypher("MATCH (n:N {id:1}) SET n.p = $p", params={"p": val})
    elif path == "params_create":
        kg.cypher("CREATE (n:N {id:1, p:$p})", params={"p": val})
    else:
        raise ValueError(path)
    return kg


def read_node(kg: kglite.KnowledgeGraph) -> object:
    rows = kg.cypher("MATCH (n:N {id:1}) RETURN n.p AS p").to_dicts()
    return rows[0]["p"] if rows else "__NOROW__"


# ── Edge ingestion paths ────────────────────────────────────────────────────
def ingest_edge(kind: str, path: str) -> kglite.KnowledgeGraph:
    """Create one ``(:N)-[:R {p:<value>}]->(:N)`` edge via ``path``. May raise."""
    val, lit, _ = KINDS[kind]
    kg = kglite.KnowledgeGraph()
    if path != "from_records":
        kg.cypher("CREATE (:N {id:1}) CREATE (:N {id:2})")
    if path == "add_connections":
        df = pd.DataFrame({"s": [1], "t": [2]})
        df["p"] = _prop_series(kind, val)
        kg.add_connections(df, "R", "N", "s", "N", "t", columns=["p"])
    elif path == "from_records":
        spec = {
            "nodes": [{"type": "N", "id_field": "id", "records": [{"id": 1}, {"id": 2}]}],
            "connections": [
                {
                    "type": "R",
                    "source_type": "N",
                    "source_id_field": "s",
                    "target_type": "N",
                    "target_id_field": "t",
                    "records": [{"s": 1, "t": 2, "p": val}],
                }
            ],
        }
        kg = kglite.from_records(spec)
    elif path == "cypher_literal":
        kg.cypher(f"MATCH (a:N {{id:1}}),(b:N {{id:2}}) CREATE (a)-[r:R {{p:{lit}}}]->(b)")
    elif path == "cypher_set":
        kg.cypher("MATCH (a:N {id:1}),(b:N {id:2}) CREATE (a)-[:R]->(b)")
        if kind in _CYPHER_ONLY:
            kg.cypher(f"MATCH ()-[r:R]->() SET r.p = {lit}")
        else:
            kg.cypher("MATCH ()-[r:R]->() SET r.p = $p", params={"p": val})
    elif path == "params_create":
        kg.cypher(
            "MATCH (a:N {id:1}),(b:N {id:2}) CREATE (a)-[r:R {p:$p}]->(b)",
            params={"p": val},
        )
    else:
        raise ValueError(path)
    return kg


def read_edge(kg: kglite.KnowledgeGraph) -> object:
    rows = kg.cypher("MATCH ()-[r:R]->() RETURN r.p AS p").to_dicts()
    return rows[0]["p"] if rows else "__NOROW__"


# ── Outcome tables ──────────────────────────────────────────────────────────
# (kind, path) -> "ok" | ("xfail", reason) | ("skip", reason) | ("raises", Exc)
_RAISES_JSON = ("raises", TypeError)

NODE_PATHS = ["add_nodes", "from_records", "cypher_literal", "cypher_set", "params_create"]
EDGE_PATHS = ["add_connections", "from_records", "cypher_literal", "cypher_set", "params_create"]


def _node_outcome(kind: str, path: str):
    if kind in _CYPHER_ONLY and path not in ("cypher_literal", "cypher_set"):
        return ("skip", "Point has no Python literal; only creatable via cypher point()")
    if path == "from_records" and kind in _JSON_UNSERIALISABLE:
        return _RAISES_JSON
    return "ok"


def _edge_outcome(kind: str, path: str):
    if kind in _CYPHER_ONLY and path not in ("cypher_literal", "cypher_set"):
        return ("skip", "Point has no Python literal; only creatable via cypher point()")
    if path == "from_records" and kind in _JSON_UNSERIALISABLE:
        return _RAISES_JSON
    return "ok"


def _build_params(paths, outcome_fn):
    params = []
    for kind in KINDS:
        for path in paths:
            outcome = outcome_fn(kind, path)
            marks = []
            if isinstance(outcome, tuple) and outcome[0] == "xfail":
                marks = [pytest.mark.xfail(strict=True, reason=outcome[1])]
            elif isinstance(outcome, tuple) and outcome[0] == "skip":
                marks = [pytest.mark.skip(reason=outcome[1])]
            params.append(pytest.param(kind, path, id=f"{kind}-{path}", marks=marks))
    return params


# ── In-memory round-trip matrix ─────────────────────────────────────────────
@pytest.mark.parametrize("kind,path", _build_params(NODE_PATHS, _node_outcome))
def test_node_value_roundtrip_in_memory(kind, path):
    outcome = _node_outcome(kind, path)
    if isinstance(outcome, tuple) and outcome[0] == "raises":
        with pytest.raises(outcome[1]):
            ingest_node(kind, path)
        return
    kg = ingest_node(kind, path)
    got = read_node(kg)
    _, _, expected = KINDS[kind]
    assert _matches(expected, got), f"{kind} via {path}: expected {expected!r}, got {got!r}"


@pytest.mark.parametrize("kind,path", _build_params(EDGE_PATHS, _edge_outcome))
def test_edge_value_roundtrip_in_memory(kind, path):
    outcome = _edge_outcome(kind, path)
    if isinstance(outcome, tuple) and outcome[0] == "raises":
        with pytest.raises(outcome[1]):
            ingest_edge(kind, path)
        return
    kg = ingest_edge(kind, path)
    got = read_edge(kg)
    _, _, expected = KINDS[kind]
    assert _matches(expected, got), f"{kind} via {path}: expected {expected!r}, got {got!r}"


# ── .kgl persistence round-trip ─────────────────────────────────────────────
# Ingest via the most-preserving path (cypher literal), save, reload, re-read.
# Every kind that survives in-memory MUST also survive .kgl — this is the
# passing guard proving that maps/lists/point are NOT lost on save through the
# default in-memory serialization (counter-evidence to the literal reading of
# the "Map saved as NULL to .kgl" report).
_PERSIST_KINDS = list(KINDS.keys())


@pytest.mark.parametrize("kind", _PERSIST_KINDS)
def test_node_value_survives_kgl(kind, tmp_path):
    kg = ingest_node(kind, "cypher_literal")
    p = str(tmp_path / f"node_{kind}.kgl")
    kg.save(p)
    got = read_node(kglite.load(p))
    _, _, expected = KINDS[kind]
    assert _matches(expected, got), f"{kind}: .kgl reload expected {expected!r}, got {got!r}"


@pytest.mark.parametrize("kind", _PERSIST_KINDS)
def test_edge_value_survives_kgl(kind, tmp_path):
    kg = ingest_edge(kind, "cypher_literal")
    p = str(tmp_path / f"edge_{kind}.kgl")
    kg.save(p)
    got = read_edge(kglite.load(p))
    _, _, expected = KINDS[kind]
    assert _matches(expected, got), f"{kind}: .kgl reload expected {expected!r}, got {got!r}"


# ── Cross-storage-mode parity for collections (list / map) ──────────────────
# Locks in that native collections survive live reads AND .kgl save/reload on
# every storage backend — including the mapped/disk columnar modes whose
# overflow-bag encoder (column_store.rs:1073) drops maps in the bulk-build
# path. These pass because the Cypher-write path keeps Values in the in-memory
# overlay, not the overflow bag; if that ever changes, these fail first.
_COLLECTION_KINDS = ["list_str", "list_int", "nested_list", "map", "dict_in_list", "list_in_dict"]


@pytest.mark.parametrize("mode", ["memory", "mapped", "disk"])
@pytest.mark.parametrize("kind", _COLLECTION_KINDS)
def test_collection_survives_storage_modes(kind, mode, tmp_path):
    val, _, expected = KINDS[kind]
    kg = _new_kg(mode, tmp_path)
    kg.cypher("CREATE (n:N {id:1, p:$p})", params={"p": val})
    live = read_node(kg)
    assert _matches(expected, live), f"{kind}/{mode} live: expected {expected!r}, got {live!r}"
    p = str(tmp_path / f"{mode}_{kind}.kgl")
    kg.save(p)
    after = read_node(kglite.load(p))
    assert _matches(expected, after), f"{kind}/{mode} .kgl: expected {expected!r}, got {after!r}"


def test_map_survives_kgl_all_storage_modes(tmp_path):
    """Canary guard for the latent column_store.rs:1073 overflow-bag bug.

    A node map survives ``.kgl`` save/reload on memory, mapped, AND disk. The
    overflow-bag encoder that writes maps as NULL is only reached by the
    columnar bulk builders (RDF / streaming), not this path. If a future change
    routes Cypher-created maps through that encoder, this guard fails and points
    at the regression.
    """
    for mode in ("memory", "mapped", "disk"):
        sub = tmp_path / mode
        sub.mkdir()
        kg = _new_kg(mode, sub)
        kg.cypher("CREATE (n:N {id:1, m:$m})", params={"m": {"k": "v", "n": 7}})
        p = str(sub / "g.kgl")
        kg.save(p)
        got = kglite.load(p).cypher("MATCH (n:N {id:1}) RETURN n.m AS m").to_dicts()[0]["m"]
        assert got == {"k": "v", "n": 7}, f"{mode}: map lost on .kgl reload → {got!r}"


# ── add_connections property-selection semantics ───────────────────────────
def test_add_connections_without_columns_preserves_list():
    """The default edge ingest preserves properties, matching add_nodes."""
    kg = kglite.KnowledgeGraph()
    kg.cypher("CREATE (:N {id:1}) CREATE (:N {id:2})")
    df = pd.DataFrame({"s": [1], "t": [2]})
    df["tags"] = _one_col_series(["a", "b"])
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        kg.add_connections(df, "R", "N", "s", "N", "t")  # no columns= whitelist
    got = kg.cypher("MATCH ()-[r:R]->() RETURN r.tags AS tags").to_dicts()[0]["tags"]
    assert got == ["a", "b"]
    assert not [w for w in captured if issubclass(w.category, UserWarning)]


def test_add_connections_with_columns_preserves_list():
    """The companion to the above: whitelisting the list column round-trips it.

    Proves the storage/Cypher edge-list path is sound (B.2a) — the loss is
    purely the missing-whitelist asymmetry, not a storage defect.
    """
    kg = kglite.KnowledgeGraph()
    kg.cypher("CREATE (:N {id:1}) CREATE (:N {id:2})")
    df = pd.DataFrame({"s": [1], "t": [2]})
    df["tags"] = _one_col_series(["a", "b"])
    kg.add_connections(df, "R", "N", "s", "N", "t", columns=["tags"])
    got = kg.cypher("MATCH ()-[r:R]->() RETURN r.tags AS tags").to_dicts()[0]["tags"]
    assert got == ["a", "b"]


def test_add_connections_columns_and_skip_columns_filter_exactly():
    kg = kglite.KnowledgeGraph()
    kg.cypher("CREATE (:N {id:1}) CREATE (:N {id:2}) CREATE (:N {id:3}) CREATE (:N {id:4})")
    df = pd.DataFrame({"s": [1], "t": [2], "keep": [10], "drop": [20]})
    kg.add_connections(df, "WHITELIST", "N", "s", "N", "t", columns=["keep"])
    kg.add_connections(df.assign(t=3), "EXCLUDE", "N", "s", "N", "t", skip_columns=["drop"])
    kg.add_connections(
        df.assign(t=4),
        "COMBINED",
        "N",
        "s",
        "N",
        "t",
        columns=["keep", "drop"],
        skip_columns=["drop"],
    )

    rows = kg.cypher("MATCH ()-[r]->() RETURN type(r) AS type, r.keep AS keep, r.drop AS drop ORDER BY type").to_dicts()
    assert rows == [
        {"type": "COMBINED", "keep": 10, "drop": None},
        {"type": "EXCLUDE", "keep": 10, "drop": None},
        {"type": "WHITELIST", "keep": 10, "drop": None},
    ]


@pytest.mark.parametrize("mode", ["memory", "mapped", "disk"])
def test_default_edge_properties_survive_storage_and_reload(mode, tmp_path):
    sub = tmp_path / mode
    sub.mkdir()
    kg = _new_kg(mode, sub)
    kg.cypher("CREATE (:N {id:1}) CREATE (:N {id:2})")
    df = pd.DataFrame({"s": [1], "t": [2], "weight": [3.5]})
    df["tags"] = _one_col_series(["a", "b"])
    df["meta"] = _one_col_series({"rank": 7, "flags": [True, False]})
    kg.add_connections(df, "R", "N", "s", "N", "t")

    query = "MATCH ()-[r:R]->() RETURN r.weight AS weight, r.tags AS tags, r.meta AS meta"
    expected = {"weight": 3.5, "tags": ["a", "b"], "meta": {"rank": 7, "flags": [True, False]}}
    assert kg.cypher(query).to_dicts() == [expected]

    saved = str(sub / "roundtrip.kgl")
    kg.save(saved)
    assert kglite.load(saved).cypher(query).to_dicts() == [expected]


def test_add_nodes_keeps_list_column_without_whitelist():
    """add_nodes and add_connections both keep default property columns."""
    kg = kglite.KnowledgeGraph()
    df = pd.DataFrame({"id": [1]})
    df["tags"] = _one_col_series(["a", "b"])
    kg.add_nodes(df, "N", "id")
    got = kg.cypher("MATCH (n:N) RETURN n.tags AS tags").to_dicts()[0]["tags"]
    assert got == ["a", "b"]


# ── Chained-dot into a map now works (was Fact #2 silent-loss) ──────────────
def test_chained_dot_into_node_map_works():
    kg = kglite.KnowledgeGraph()
    kg.cypher("CREATE (n:N {id:1, m:$m})", params={"m": {"k": "v"}})
    got = kg.cypher("MATCH (n:N) RETURN n.m.k AS x").to_dicts()[0]["x"]
    assert got == "v", f"chained-dot n.m.k degraded to {got!r}"


def test_chained_dot_into_edge_map_works():
    kg = kglite.KnowledgeGraph()
    kg.cypher("CREATE (:N {id:1}) CREATE (:N {id:2})")
    kg.cypher("MATCH (a:N {id:1}),(b:N {id:2}) CREATE (a)-[r:R {m:{k:'v'}}]->(b)")
    got = kg.cypher("MATCH ()-[r:R]->() RETURN r.m.k AS x").to_dicts()[0]["x"]
    assert got == "v", f"chained-dot r.m.k degraded to {got!r}"


def test_bracket_subscript_into_map_works():
    """The companion reader: bracket subscript into a map works on this build."""
    kg = kglite.KnowledgeGraph()
    kg.cypher("CREATE (n:N {id:1, m:$m})", params={"m": {"k": "v"}})
    got = kg.cypher("MATCH (n:N) RETURN n.m['k'] AS x").to_dicts()[0]["x"]
    assert got == "v"


# ── from_records loud-error contract for Python temporals ───────────────────
@pytest.mark.parametrize("val", [_DATE, _DT], ids=["date", "datetime"])
def test_from_records_rejects_python_temporal_loudly(val):
    """A Python date/datetime in a from_records spec fails loudly (JSON has no
    date type). This is the *acceptable* half of the contract — a loud error at
    write time, not a silent degradation."""
    spec = {"nodes": [{"type": "N", "id_field": "id", "records": [{"id": 1, "p": val}]}]}
    with pytest.raises(TypeError):
        kglite.from_records(spec)


# ── Point round-trip (spatial value) ────────────────────────────────────────
def test_point_roundtrips_through_kgl(tmp_path):
    kg = kglite.KnowledgeGraph()
    kg.cypher("CREATE (n:N {id:1, loc:point(1.0,2.0)})")
    p = str(tmp_path / "point.kgl")
    kg.save(p)
    got = kglite.load(p).cypher("MATCH (n:N) RETURN n.loc AS x").to_dicts()[0]["x"]
    assert got == _POINT_READBACK
