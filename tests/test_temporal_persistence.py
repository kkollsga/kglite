"""Validity-interval declarations in a saved graph: what a `.kgl` records for
this build (`temporal_declarations`) and for older ones (the legacy
`temporal_node_configs` / `temporal_edge_configs` keys), and how files that
carry only the legacy keys load.

The legacy fixtures under ``tests/fixtures/temporal_legacy/`` were written by
the published 0.18.1 wheel; ``tests/fixtures/build_temporal_legacy_fixtures.py``
regenerates them and documents the isolated-interpreter procedure. A
fixture that stops loading is a finding, not a regeneration prompt.

Red proof: before declarations had their own key, a save wrote half-open and
source-keyed configs into the legacy keys (with fields an older reader
ignores, so it read them as closed and unkeyed), dropped the declare-time
counts, and a legacy list read back with its duplicates and no
``ambiguous`` column.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import shutil
import struct

import pytest

import kglite

FIXTURES = Path(__file__).parent / "fixtures" / "temporal_legacy"
LEGACY_KEYS = ("temporal_node_configs", "temporal_edge_configs")
DECLARATIONS = (
    "CALL db.temporal.declarations() "
    "YIELD kind, name, source_type, from, to, convention, abutting_rows, ambiguous "
    "RETURN kind, name, source_type, from, to, convention, abutting_rows, ambiguous"
)


def _builder():
    """The fixture generator's graph builders, so a current build can make
    exactly the graph 0.18.1 saved."""
    spec = importlib.util.spec_from_file_location(
        "build_temporal_legacy_fixtures", FIXTURES.parent / "build_temporal_legacy_fixtures.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _metadata_text(path: Path) -> str:
    raw = path.read_bytes()
    (length,) = struct.unpack_from("<I", raw, 9)
    return raw[13 : 13 + length].decode()


def _raw_key(metadata: str, name: str) -> str | None:
    """The exact JSON text a top-level metadata key holds, or None."""
    tag = f'"{name}":'
    if tag not in metadata:
        return None
    start = metadata.index(tag) + len(tag)
    _, end = json.JSONDecoder().raw_decode(metadata, start)
    return metadata[start:end]


def _declarations(g):
    return g.cypher(DECLARATIONS).to_list()


def _load_copy(name: str, tmp_path: Path):
    copy = tmp_path / f"{name}.kgl"
    shutil.copy(FIXTURES / f"{name}.kgl", copy)
    return kglite.load(str(copy))


@pytest.fixture
def licensees():
    """Fields and licences hold HAS_LICENSEE periods under different
    properties; OPERATES periods use one pair for both sources."""
    g = kglite.KnowledgeGraph()
    g.cypher(
        """
        CREATE (f:Field {id: 1, title: 'F1', vf: '2000-01-01', vt: '2010-12-31'}),
               (l:Licence {id: 10, title: 'L10'}),
               (c:Company {id: 100, title: 'Acme'}),
               (f)-[:HAS_LICENSEE {ff: '2000-01-01', ft: '2009-12-31'}]->(c),
               (l)-[:HAS_LICENSEE {lf: '1990-01-01', lt: '1999-12-31'}]->(c),
               (f)-[:OPERATES {of: '2001-01-01', ot: '2002-01-01'}]->(c),
               (l)-[:OPERATES {of: '2003-01-01', ot: null}]->(c)
        """
    )
    return g


class TestLegacyKeysForOlderReaders:
    def test_closed_only_graph_writes_the_legacy_bytes_0_18_1_wrote(self, tmp_path):
        builder = _builder()
        g = builder._base()
        builder._connect(g, {"lic_from": "validFrom", "lic_to": "validTo"})
        out = tmp_path / "closed.kgl"
        g.save(str(out))
        ours, theirs = _metadata_text(out), _metadata_text(FIXTURES / "closed.kgl")
        for key in LEGACY_KEYS:
            assert _raw_key(ours, key) == _raw_key(theirs, key), key
        assert _raw_key(ours, "temporal_declarations") is not None

    def test_per_source_and_half_open_declarations_stay_out_of_the_legacy_keys(self, licensees, tmp_path):
        for spec in (
            "{node: 'Field', from: 'vf', to: 'vt', convention: 'half_open'}",
            "{relationship: 'HAS_LICENSEE', source_type: 'Field', from: 'ff', to: 'ft', convention: 'closed'}",
            "{relationship: 'HAS_LICENSEE', source_type: 'Licence', from: 'lf', to: 'lt', convention: 'closed'}",
            "{relationship: 'OPERATES', source_type: 'Field', from: 'of', to: 'ot', convention: 'closed'}",
            "{relationship: 'OPERATES', source_type: 'Licence', from: 'of', to: 'ot', convention: 'closed'}",
        ):
            licensees.cypher(f"CALL db.temporal.declare({spec})")
        out = tmp_path / "declared.kgl"
        licensees.save(str(out))
        metadata = _metadata_text(out)
        assert _raw_key(metadata, "temporal_node_configs") == "{}"
        # Keyed configs stay out even when their properties agree (OPERATES).
        assert _raw_key(metadata, "temporal_edge_configs") == "{}"
        declared = json.loads(_raw_key(metadata, "temporal_declarations"))
        assert {"kind": "node", "name": "Field", "convention": "half_open"}.items() <= declared[0].items()
        assert [d.get("source_type") for d in declared if d["name"] == "HAS_LICENSEE"] == ["Field", "Licence"]

    def test_a_graph_without_declarations_has_no_new_key(self, licensees, tmp_path):
        out = tmp_path / "plain.kgl"
        licensees.save(str(out))
        metadata = _metadata_text(out)
        assert _raw_key(metadata, "temporal_declarations") is None
        assert [_raw_key(metadata, key) for key in LEGACY_KEYS] == ["{}", "{}"]


class TestLegacyFilesLoad:
    def test_single_configs(self, tmp_path):
        g = _load_copy("closed", tmp_path)
        assert _declarations(g) == [
            {
                "kind": "node",
                "name": "Field",
                "source_type": None,
                "from": "vf",
                "to": "vt",
                "convention": "closed",
                "abutting_rows": None,
                "ambiguous": False,
            },
            {
                "kind": "relationship",
                "name": "HAS_LICENSEE",
                "source_type": None,
                "from": "lic_from",
                "to": "lic_to",
                "convention": "closed",
                "abutting_rows": None,
                "ambiguous": False,
            },
        ]
        # The fluent filter still applies it: Beta's licence (2006-, open) is
        # current in 2008, Alpha's (2001-2003) is not.
        found = g.select("Field").valid_at("2008-06-01").traverse("HAS_LICENSEE").collect()
        assert sorted(n["title"] for n in found) == ["Globex"]

    def test_identical_duplicates_read_as_one(self, tmp_path):
        g = _load_copy("identical_duplicates", tmp_path)
        rels = [d for d in _declarations(g) if d["kind"] == "relationship"]
        assert [(d["from"], d["to"], d["ambiguous"]) for d in rels] == [("lic_from", "lic_to", False)]

    def test_distinct_duplicates_load_ambiguous_with_both_kept(self, tmp_path):
        g = _load_copy("distinct_duplicates", tmp_path)
        rels = [d for d in _declarations(g) if d["kind"] == "relationship"]
        assert [(d["from"], d["source_type"], d["ambiguous"]) for d in rels] == [
            ("lic_from", None, True),
            ("other_from", None, True),
        ]
        conn = next(line for line in g.describe().splitlines() if "HAS_LICENSEE" in line and "temporal" in line)
        assert 'temporal_ambiguous="true"' in conn
        # Re-saved, the legacy key holds both in order, as 0.18.1 wrote it.
        resaved = tmp_path / "resaved.kgl"
        g.save(str(resaved))
        theirs = _metadata_text(FIXTURES / "distinct_duplicates.kgl")
        assert _raw_key(_metadata_text(resaved), "temporal_edge_configs") == _raw_key(theirs, "temporal_edge_configs")
        # Re-declaring per source resolves it.
        g.cypher("CALL db.temporal.undeclare({relationship: 'HAS_LICENSEE'})")
        g.cypher(
            "CALL db.temporal.declare({relationship: 'HAS_LICENSEE', source_type: 'Field', "
            "from: 'lic_from', to: 'lic_to', convention: 'closed'})"
        )
        assert [d["ambiguous"] for d in _declarations(g) if d["kind"] == "relationship"] == [False]


@pytest.mark.parity
@pytest.mark.parametrize("mode", ("memory", "mapped", "disk"))
def test_temporal_declarations_round_trip(mode, tmp_path):
    kwargs = {"memory": {}, "mapped": {"storage": "mapped"}, "disk": {"storage": "disk", "path": str(tmp_path / "b")}}
    g = kglite.KnowledgeGraph(**kwargs[mode])
    g.cypher(
        """
        CREATE (f:Field {id: 1, title: 'F1', vf: '2000-01-01', vt: '2010-12-31'}),
               (l:Licence {id: 10, title: 'L10'}),
               (c:Company {id: 100, title: 'Acme'}),
               (f)-[:HAS_LICENSEE {ff: '2000-01-01', ft: '2009-12-31'}]->(c),
               (l)-[:HAS_LICENSEE {lf: '1990-01-01', lt: '1999-12-31'}]->(c)
        """
    )
    for spec in (
        "{node: 'Field', from: 'vf', to: 'vt', convention: 'closed'}",
        "{relationship: 'HAS_LICENSEE', source_type: 'Field', from: 'ff', to: 'ft', convention: 'closed'}",
        "{relationship: 'HAS_LICENSEE', source_type: 'Licence', from: 'lf', to: 'lt', convention: 'half_open'}",
    ):
        g.cypher(f"CALL db.temporal.declare({spec})")
    before = _declarations(g)
    assert [d["convention"] for d in before] == ["closed", "closed", "half_open"]
    assert all(d["abutting_rows"] == 0 for d in before)
    out = tmp_path / ("saved" if mode == "disk" else "saved.kgl")
    g.save(str(out))
    loaded = kglite.load(str(out))
    assert _declarations(loaded) == before
    assert loaded.describe() == g.describe()
