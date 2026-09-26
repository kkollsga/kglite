"""Edges find string-keyed nodes by their zero-padded codes.

A blueprint that declares its `pk` as `"string"` keeps `"0001"` as the node
id. The id columns of its `fk_edges` and `junction_edges` were still typed by
inference, which read the same cell as the integer 1: no node matched, and
every row vivified an unbounded stub node (valid on every date, under a
temporal declaration) instead of connecting. An id column that refers to a
string-keyed node type is now read as text, on the buffered and the streamed
path alike.
"""

from __future__ import annotations

import json
import warnings

import pytest

import kglite


def _write(tmp_path, name: str, text: str) -> None:
    (tmp_path / name).write_text(text, encoding="utf-8")


@pytest.fixture
def blueprint_path(tmp_path):
    _write(
        tmp_path,
        "munis.csv",
        "code,name,successor,valid_from,valid_to\n"
        "0001,Adorp,0053,,1990-01-01\n"
        "0053,Winsum,,1990-01-01,2019-01-01\n"
        "0990,Oldtown,0053,,1990-01-01\n",
    )
    _write(tmp_path, "provinces.csv", "pid,name\nPV-GR,Groningen\n")
    _write(
        tmp_path,
        "in_province.csv",
        "src,dst,valid_from,valid_to\n0001,PV-GR,,1990-01-01\n0053,PV-GR,1990-01-01,\n",
    )
    _write(tmp_path, "cities.csv", "wid,name\nWP1,Adorp dorp\n")
    _write(tmp_path, "in_muni.csv", "src,dst\nWP1,0001\n")
    blueprint = {
        "settings": {"root": str(tmp_path)},
        "nodes": {
            "Province": {"csv": "provinces.csv", "pk": "pid", "title": "name"},
            "Municipality": {
                "csv": "munis.csv",
                "pk": "code",
                "title": "name",
                "properties": {
                    "code": "string",
                    "successor": "string",
                    "valid_from": "date",
                    "valid_to": "date",
                },
                "temporal": {"from": "valid_from", "to": "valid_to", "convention": "half_open"},
                "connections": {
                    "fk_edges": {"MERGED_INTO": {"target": "Municipality", "fk": "successor"}},
                    "junction_edges": {
                        "IN_PROVINCE": {
                            "csv": "in_province.csv",
                            "source_fk": "src",
                            "target": "Province",
                            "target_fk": "dst",
                            "properties": ["valid_from", "valid_to"],
                            "property_types": {"valid_from": "date", "valid_to": "date"},
                        }
                    },
                },
            },
            "Woonplaats": {
                "csv": "cities.csv",
                "pk": "wid",
                "title": "name",
                "connections": {
                    "junction_edges": {
                        "IN_MUNICIPALITY": {
                            "csv": "in_muni.csv",
                            "source_fk": "src",
                            "target": "Municipality",
                            "target_fk": "dst",
                        }
                    }
                },
            },
        },
    }
    path = tmp_path / "bp.json"
    path.write_text(json.dumps(blueprint), encoding="utf-8")
    return str(path)


@pytest.mark.parametrize("streamed", [False, True], ids=["buffered", "streamed"])
def test_zero_padded_codes_connect_without_stubs(blueprint_path, streamed, monkeypatch) -> None:
    if streamed:
        monkeypatch.setenv("KGLITE_BLUEPRINT_STREAMING_THRESHOLD_MB", "0")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        graph = kglite.from_blueprint(blueprint_path)
    stub_warnings = [str(w.message) for w in caught if "stub" in str(w.message)]
    assert stub_warnings == []

    assert graph.cypher("MATCH (m:Municipality) RETURN m.id AS id ORDER BY id").to_list() == [
        {"id": "0001"},
        {"id": "0053"},
        {"id": "0990"},
    ]
    assert graph.cypher(
        "MATCH (m:Municipality)-[:MERGED_INTO]->(s) RETURN m.id AS m, s.id AS s ORDER BY m"
    ).to_list() == [{"m": "0001", "s": "0053"}, {"m": "0990", "s": "0053"}]
    assert graph.cypher(
        "MATCH (m:Municipality)-[:IN_PROVINCE]->(p) RETURN m.id AS m, p.id AS p ORDER BY m"
    ).to_list() == [{"m": "0001", "p": "PV-GR"}, {"m": "0053", "p": "PV-GR"}]
    assert graph.cypher("MATCH (w:Woonplaats)-[:IN_MUNICIPALITY]->(m) RETURN w.id AS w, m.id AS m").to_list() == [
        {"w": "WP1", "m": "0001"}
    ]
    # The as-of view sees the three real rows only — no open-interval stubs.
    assert graph.date("1995-06-01").select("Municipality").len() == 1
