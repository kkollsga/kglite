"""Validity intervals a blueprint declares through a spec's ``temporal`` key:
``{"from": ..., "to": ..., "convention": "closed" | "half_open"}`` on a node
spec, an ``fk_edges`` entry or a ``junction_edges`` entry.

Red proof: before the key existed the parser ignored it with an unknown-key
warning, nothing was declared, and every golden below saw both the expired
and the current row.
"""

import json
import warnings

import pandas as pd
import pytest

import kglite
from kglite.blueprint import from_blueprint

DECLARATIONS = (
    "CALL db.temporal.declarations() "
    "YIELD kind, name, source_type, from, to, convention "
    "RETURN kind, name, source_type, from, to, convention"
)

CLOSED = {"from": "vf", "to": "vt", "convention": "closed"}


def _declarations(g):
    rows = g.cypher(DECLARATIONS).to_list()
    return sorted((r["kind"], r["name"], r["source_type"], r["from"], r["to"], r["convention"]) for r in rows)


def _titles(selection):
    return sorted(row["title"] for row in selection.collect())


def _write(tmp_path, tables, bp):
    for name, frame in tables.items():
        frame.to_csv(tmp_path / name, index=False)
    bp = {"settings": {"root": str(tmp_path)}, **bp}
    path = tmp_path / "blueprint.json"
    path.write_text(json.dumps(bp), encoding="utf-8")
    return path


def _build(blueprint_path, **kwargs):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        graph = from_blueprint(blueprint_path, save=False, **kwargs)
    return graph, [str(w.message) for w in caught]


# ── The operator/status fixture (companies, one field, its statuses) ─────


def _mk3_tables():
    return {
        "company.csv": pd.DataFrame({"cid": [1, 2], "name": ["Statoil", "Equinor"]}),
        "field.csv": pd.DataFrame({"fid": [10], "fname": ["Gullfaks"]}),
        "status.csv": pd.DataFrame(
            {
                "sid": [1, 2],
                "fid": [10, 10],
                "status": ["Approved", "Producing"],
                "sf": ["1980-01-01", "1987-01-01"],
                "st": ["1986-12-31", None],
            }
        ),
        "fop.csv": pd.DataFrame(
            {
                "fid": [10, 10],
                "cid": [1, 2],
                "vf": ["2001-01-01", "2018-05-16"],
                "vt": ["2018-05-15", None],
            }
        ),
    }


def _mk3_blueprint(status_temporal, operator_temporal):
    status = {
        "csv": "status.csv",
        "pk": "sid",
        "title": "status",
        "properties": {"sf": "validFrom", "st": "validTo"},
    }
    operator = {
        "csv": "fop.csv",
        "source_fk": "fid",
        "target": "Company",
        "target_fk": "cid",
        "properties": ["vf", "vt"],
        "property_types": {"vf": "validFrom", "vt": "validTo"},
    }
    if status_temporal is not None:
        status["temporal"] = status_temporal
    if operator_temporal is not None:
        operator["temporal"] = operator_temporal
    return {
        "nodes": {
            "Company": {"csv": "company.csv", "pk": "cid", "title": "name"},
            "Field": {
                "csv": "field.csv",
                "pk": "fid",
                "title": "fname",
                "connections": {"junction_edges": {"HAS_OPERATOR": operator}},
            },
            "Status": status,
        }
    }


STATUS_CLOSED = {"from": "sf", "to": "st", "convention": "closed"}


class TestOperatorGolden:
    @pytest.mark.parametrize("storage", ["default", "mapped", "disk"])
    def test_declared_intervals_filter_every_read(self, tmp_path, storage):
        path = _write(tmp_path, _mk3_tables(), _mk3_blueprint(STATUS_CLOSED, CLOSED))
        kwargs = {"storage": storage}
        if storage == "disk":
            kwargs["path"] = str(tmp_path / "disk")
        g, messages = _build(path, **kwargs)
        assert not [m for m in messages if "unknown key" in m], messages

        assert _titles(g.date("2009-06-30").select("Field").traverse("HAS_OPERATOR")) == ["Statoil"]
        assert _titles(g.select("Status")) == ["Producing"]
        assert _titles(g.date("1985-01-01").select("Status")) == ["Approved"]
        assert _declarations(g) == [
            ("node", "Status", None, "sf", "st", "closed"),
            ("relationship", "HAS_OPERATOR", "Field", "vf", "vt", "closed"),
        ]
        described = g.describe()
        assert 'temporal_from="sf" temporal_to="st"' in described
        assert 'temporal_from="vf" temporal_to="vt" temporal_source="Field"' in described

    def test_a_streamed_node_spec_declares_too(self, tmp_path, monkeypatch):
        monkeypatch.setenv("KGLITE_BLUEPRINT_STREAMING_THRESHOLD_MB", "0")
        path = _write(tmp_path, _mk3_tables(), _mk3_blueprint(STATUS_CLOSED, CLOSED))
        g, _ = _build(path)
        assert _titles(g.select("Status")) == ["Producing"]
        assert ("node", "Status", None, "sf", "st", "closed") in _declarations(g)

    def test_role_types_alone_declare_nothing_and_warn(self, tmp_path):
        path = _write(tmp_path, _mk3_tables(), _mk3_blueprint(None, None))
        g, messages = _build(path)
        assert _declarations(g) == []
        assert _titles(g.select("Status")) == ["Approved", "Producing"]
        assert "temporal_from" not in g.describe()
        typed_only = [m for m in messages if "only type the column" in m]
        assert len(typed_only) == 2, messages
        assert any(m.startswith("node 'Status'") and '"from": "sf", "to": "st"' in m for m in typed_only), typed_only
        assert any(m.startswith("junction 'HAS_OPERATOR' (node 'Field')") for m in typed_only), typed_only

    def test_a_key_without_a_convention_declares_nothing_and_warns(self, tmp_path):
        no_convention = {"from": "sf", "to": "st"}
        path = _write(tmp_path, _mk3_tables(), _mk3_blueprint(no_convention, None))
        g, messages = _build(path)
        assert _declarations(g) == []
        assert "temporal_from" not in g.describe()
        assert any(m.startswith("node 'Status'") and "names no convention" in m for m in messages), messages

    def test_an_unknown_convention_fails_the_build(self, tmp_path):
        spec = {**STATUS_CLOSED, "convention": "half-open"}
        path = _write(tmp_path, _mk3_tables(), _mk3_blueprint(spec, None))
        with pytest.raises(Exception, match="convention 'half-open' is not one of"):
            from_blueprint(path, save=False)


# ── Two node types writing one relationship type ─────────────────────────


class TestTwoSourceRelationship:
    """Fields and licences both hold licensee periods, the licences under
    renamed bounds. Each source keeps its own declaration."""

    def _path(self, tmp_path):
        tables = {
            "company.csv": pd.DataFrame({"cid": [1, 2], "name": ["Statoil", "Equinor"]}),
            "field.csv": pd.DataFrame({"fid": [10], "fname": ["Gullfaks"]}),
            "licence.csv": pd.DataFrame({"lid": [50], "lname": ["PL050"]}),
            "field_lic.csv": pd.DataFrame(
                {"fid": [10, 10], "cid": [1, 2], "vf": ["2001-01-01", "2005-01-01"], "vt": ["2004-12-31", None]}
            ),
            "lic_lic.csv": pd.DataFrame(
                {"lid": [50, 50], "cid": [1, 2], "vf": ["1990-01-01", "2000-01-01"], "vt": ["1999-12-31", None]}
            ),
        }
        bp = {
            "nodes": {
                "Company": {"csv": "company.csv", "pk": "cid", "title": "name"},
                "Field": {
                    "csv": "field.csv",
                    "pk": "fid",
                    "title": "fname",
                    "connections": {
                        "junction_edges": {
                            "HAS_LICENSEE": {
                                "csv": "field_lic.csv",
                                "source_fk": "fid",
                                "target": "Company",
                                "target_fk": "cid",
                                "properties": ["vf", "vt"],
                                "property_types": {"vf": "validFrom", "vt": "validTo"},
                                "temporal": CLOSED,
                            }
                        }
                    },
                },
                "Licence": {
                    "csv": "licence.csv",
                    "pk": "lid",
                    "title": "lname",
                    "connections": {
                        "junction_edges": {
                            "HAS_LICENSEE": {
                                "csv": "lic_lic.csv",
                                "source_fk": "lid",
                                "target": "Company",
                                "target_fk": "cid",
                                "properties": ["vf", "vt"],
                                "property_types": {"vf": "validFrom", "vt": "validTo"},
                                "rename": {"vf": "lic_from", "vt": "lic_to"},
                                "temporal": {"from": "lic_from", "to": "lic_to", "convention": "closed"},
                            }
                        }
                    },
                },
            }
        }
        return _write(tmp_path, tables, bp)

    def test_each_source_declares_under_its_stored_names(self, tmp_path):
        g, _ = _build(self._path(tmp_path))
        assert _declarations(g) == [
            ("relationship", "HAS_LICENSEE", "Field", "vf", "vt", "closed"),
            ("relationship", "HAS_LICENSEE", "Licence", "lic_from", "lic_to", "closed"),
        ]
        at_2003 = g.date("2003-01-01")
        assert _titles(at_2003.select("Field").traverse("HAS_LICENSEE")) == ["Statoil"]
        assert _titles(at_2003.select("Licence").traverse("HAS_LICENSEE")) == ["Equinor"]
        at_1995 = g.date("1995-01-01")
        assert _titles(at_1995.select("Licence").traverse("HAS_LICENSEE")) == ["Statoil"]

    def test_repeated_pairs_from_both_sources_keep_every_period(self, tmp_path):
        """Each source repeats one endpoint pair. Every source node type's
        first load of the relationship type owns its rows, so the licence's
        three periods survive beside the field's two — none folds onto
        another, and no merged, inverted interval exists to refuse."""
        path = self._path(tmp_path)
        pd.DataFrame(
            {"fid": [10, 10], "cid": [1, 1], "vf": ["2001-01-01", "2005-01-01"], "vt": ["2004-12-31", None]}
        ).to_csv(tmp_path / "field_lic.csv", index=False)
        pd.DataFrame(
            {
                "lid": [50, 50, 50],
                "cid": [1, 1, 1],
                "vf": ["2001-01-01", "2004-01-01", "2007-01-01"],
                "vt": ["2003-12-31", "2006-12-31", None],
            }
        ).to_csv(tmp_path / "lic_lic.csv", index=False)
        g, _ = _build(path)
        assert _declarations(g) == [
            ("relationship", "HAS_LICENSEE", "Field", "vf", "vt", "closed"),
            ("relationship", "HAS_LICENSEE", "Licence", "lic_from", "lic_to", "closed"),
        ]

        def periods(source_type, lo, hi):
            rows = g.cypher(
                f"MATCH (:{source_type})-[r:HAS_LICENSEE]->(:Company) RETURN r.{lo} AS vf, r.{hi} AS vt"
            ).to_list()
            found = [(str(r["vf"]), None if r["vt"] is None else str(r["vt"])) for r in rows]
            return sorted(found, key=lambda p: (p[0], p[1] or ""))

        assert periods("Field", "vf", "vt") == [("2001-01-01", "2004-12-31"), ("2005-01-01", None)]
        assert periods("Licence", "lic_from", "lic_to") == [
            ("2001-01-01", "2003-12-31"),
            ("2004-01-01", "2006-12-31"),
            ("2007-01-01", None),
        ]
        assert _titles(g.date("2005-06-30").select("Licence").traverse("HAS_LICENSEE")) == ["Statoil"]


# ── A filtered subset of a declared type ─────────────────────────────────


def test_a_filter_into_type_inherits_the_temporal_key_with_its_properties(tmp_path):
    """``filter`` with ``into`` copies the source spec — properties,
    connections and their ``temporal`` keys alike — so the subset declares the
    same intervals, under its own name and as its own source type."""
    tables = {
        "company.csv": pd.DataFrame({"cid": [10], "name": ["Acme"]}),
        "status.csv": pd.DataFrame(
            {
                "sid": [1, 2, 3],
                "status": ["Approved", "Producing", "Producing"],
                "sf": ["1980-01-01", "1987-01-01", "1990-01-01"],
                "st": ["1986-12-31", None, "1995-12-31"],
                "cid": [10, 10, 10],
            }
        ),
    }
    bp = {
        "nodes": {
            "Company": {"csv": "company.csv", "pk": "cid", "title": "name"},
            "Status": {
                "csv": "status.csv",
                "pk": "sid",
                "title": "status",
                "properties": {"sf": "validFrom", "st": "validTo"},
                "temporal": {"from": "sf", "to": "st", "convention": "closed"},
                "connections": {
                    "fk_edges": {
                        "OF_CO": {
                            "target": "Company",
                            "fk": "cid",
                            "properties": ["sf", "st"],
                            "property_types": {"sf": "validFrom", "st": "validTo"},
                            "temporal": {"from": "sf", "to": "st", "convention": "closed"},
                        }
                    }
                },
            },
        },
        "compute": [{"op": "filter", "from": "Status", "into": "Producing", "where": "status == 'Producing'"}],
    }
    g, _ = _build(_write(tmp_path, tables, bp))
    assert _declarations(g) == [
        ("node", "Producing", None, "sf", "st", "closed"),
        ("node", "Status", None, "sf", "st", "closed"),
        ("relationship", "OF_CO", "Producing", "sf", "st", "closed"),
        ("relationship", "OF_CO", "Status", "sf", "st", "closed"),
    ]
    counts = {day: len(g.date(day).select("Producing").collect()) for day in ("1985-06-30", "1992-06-30", "1997-06-30")}
    assert counts == {"1985-06-30": 0, "1992-06-30": 2, "1997-06-30": 1}


# ── Rename, conventions, persistence, refusals ───────────────────────────


def _staff_tables(left_on=("2012-12-31", None)):
    return {
        "org.csv": pd.DataFrame({"oid": [1, 2], "oname": ["Acme", "Globex"]}),
        "person.csv": pd.DataFrame(
            {
                "pid": [7, 8],
                "pname": ["Ada", "Bo"],
                "oid": [1, 2],
                "hired_on": ["2010-01-01", "2013-01-01"],
                "left_on": list(left_on),
            }
        ),
    }


def _staff_blueprint(temporal):
    return {
        "nodes": {
            "Org": {"csv": "org.csv", "pk": "oid", "title": "oname"},
            "Person": {
                "csv": "person.csv",
                "pk": "pid",
                "title": "pname",
                "connections": {
                    "fk_edges": {
                        "WORKS_AT": {
                            "target": "Org",
                            "fk": "oid",
                            "properties": ["hired_on", "left_on"],
                            "property_types": {"hired_on": "validFrom", "left_on": "validTo"},
                            "rename": {"hired_on": "start", "left_on": "end"},
                            "temporal": temporal,
                        }
                    }
                },
            },
        }
    }


class TestRenameAndConvention:
    def test_an_fk_edge_declares_under_the_renamed_property(self, tmp_path):
        spec = {"from": "start", "to": "end", "convention": "closed"}
        g, _ = _build(_write(tmp_path, _staff_tables(), _staff_blueprint(spec)))
        assert _declarations(g) == [("relationship", "WORKS_AT", "Person", "start", "end", "closed")]
        assert _titles(g.date("2011-06-01").select("Person").traverse("WORKS_AT")) == ["Acme"]
        assert _titles(g.date("2014-06-01").select("Person").traverse("WORKS_AT")) == ["Globex"]

    def test_the_csv_name_of_a_renamed_bound_is_refused_with_the_stored_one(self, tmp_path):
        spec = {"from": "hired_on", "to": "end", "convention": "closed"}
        path = _write(tmp_path, _staff_tables(), _staff_blueprint(spec))
        with pytest.raises(Exception, match="'hired_on' is renamed to 'start'"):
            from_blueprint(path, save=False)

    def test_a_bound_every_row_left_open_still_declares(self, tmp_path):
        spec = {"from": "start", "to": "end", "convention": "closed"}
        path = _write(tmp_path, _staff_tables(left_on=(None, None)), _staff_blueprint(spec))
        g, _ = _build(path)
        assert _declarations(g) == [("relationship", "WORKS_AT", "Person", "start", "end", "closed")]

    def test_half_open_survives_save_and_load(self, tmp_path):
        spec = {"from": "start", "to": "end", "convention": "half_open"}
        tables = _staff_tables(left_on=("2013-01-01", None))
        g, _ = _build(_write(tmp_path, tables, _staff_blueprint(spec)))
        assert 'temporal_from="start" temporal_to="end" temporal_convention="half_open"' in g.describe()
        saved = tmp_path / "staff.kgl"
        g.save(str(saved))
        loaded = kglite.load(str(saved))
        assert _declarations(loaded) == [("relationship", "WORKS_AT", "Person", "start", "end", "half_open")]
        # Under half-open, Ada's `end` day is the first day she no longer works there.
        ada = loaded.date("2013-01-01").select("Person").where({"title": "Ada"})
        assert _titles(ada.traverse("WORKS_AT")) == []
        ada = loaded.date("2012-12-31").select("Person").where({"title": "Ada"})
        assert _titles(ada.traverse("WORKS_AT")) == ["Acme"]


class TestDirtyBounds:
    def test_an_inverted_relationship_interval_fails_the_build(self, tmp_path):
        spec = {"from": "start", "to": "end", "convention": "closed"}
        tables = _staff_tables(left_on=("2009-01-01", None))
        path = _write(tmp_path, tables, _staff_blueprint(spec))
        with pytest.raises(
            Exception, match=r"fk_edge 'WORKS_AT' \(node 'Person'\).*from node '7' to node '1'.*after the to bound"
        ):
            from_blueprint(path, save=False)

    def test_an_unreadable_node_bound_fails_the_build_naming_the_row(self, tmp_path):
        tables = _mk3_tables()
        tables["status.csv"]["st"] = ["1986-12-31", "someday"]
        bp = _mk3_blueprint(STATUS_CLOSED, None)
        bp["nodes"]["Status"]["properties"] = {}
        path = _write(tmp_path, tables, bp)
        with pytest.raises(Exception, match=r"node 'Status'.*node '2', property 'st'.*'someday'"):
            from_blueprint(path, save=False)
