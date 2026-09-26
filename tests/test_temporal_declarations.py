"""Validity-interval declarations: ``db.temporal.declare`` / ``undeclare`` /
``declarations`` and what a declaration changes in the fluent filters and in
``describe()``.

Red proof: before the procedures existed every call here failed as an unknown
procedure.
"""

from pathlib import Path
import shutil
import warnings

import pandas as pd
import pytest

import kglite

DECLARATIONS = (
    "CALL db.temporal.declarations() "
    "YIELD kind, name, source_type, from, to, convention, abutting_rows "
    "RETURN kind, name, source_type, from, to, convention, abutting_rows"
)


def _declarations(g):
    return g.cypher(DECLARATIONS).to_list()


def _declare(g, spec):
    return g.cypher(f"CALL db.temporal.declare({spec})")


@pytest.fixture
def licensees():
    """Fields and licences both hold HAS_LICENSEE periods, under different
    property names. Field 1's second period starts the day its first ends."""
    g = kglite.KnowledgeGraph()
    g.cypher(
        """
        CREATE (f1:Field {id: 1, title: 'F1'}), (f2:Field {id: 2, title: 'F2'}),
               (l:Licence {id: 10, title: 'L10'}),
               (a:Company {id: 100, title: 'Alpha'}), (b:Company {id: 200, title: 'Beta'}),
               (f1)-[:HAS_LICENSEE {ff: '2000-01-01', ft: '2009-12-31'}]->(a),
               (f1)-[:HAS_LICENSEE {ff: '2009-12-31', ft: null}]->(b),
               (f2)-[:HAS_LICENSEE {ff: '2009-12-31', ft: '2011-01-01'}]->(a),
               (l)-[:HAS_LICENSEE {lf: '1990-01-01', lt: '1999-12-31'}]->(b)
        """
    )
    return g


@pytest.fixture
def statuses():
    g = kglite.KnowledgeGraph()
    g.cypher(
        """
        UNWIND [
          {id: 1, title: 'Producing', vf: '2000-01-01', vt: '2010-06-01'},
          {id: 2, title: 'Shut down', vf: '2010-06-01', vt: null}
        ] AS r CREATE (:Status {id: r.id, title: r.title, vf: r.vf, vt: r.vt})
        """
    )
    return g


class TestDeclareAndList:
    def test_node_round_trip(self, statuses):
        rows = _declare(statuses, "{node: 'Status', from: 'vf', to: 'vt', convention: 'half_open'}").to_list()
        assert rows == [{"declared": True, "rows": 2, "abutting_rows": 1}]
        assert _declarations(statuses) == [
            {
                "kind": "node",
                "name": "Status",
                "source_type": None,
                "from": "vf",
                "to": "vt",
                "convention": "half_open",
                "abutting_rows": 1,
            }
        ]
        undeclared = statuses.cypher("CALL db.temporal.undeclare({node: 'Status'}) YIELD undeclared RETURN undeclared")
        assert undeclared.to_list() == [{"undeclared": True}]
        assert _declarations(statuses) == []

    def test_two_source_keys_list_separately(self, licensees):
        _declare(
            licensees,
            "{relationship: 'HAS_LICENSEE', source_type: 'Field', from: 'ff', to: 'ft', convention: 'half_open'}",
        )
        _declare(
            licensees,
            "{relationship: 'HAS_LICENSEE', source_type: 'Licence', from: 'lf', to: 'lt', convention: 'closed'}",
        )
        assert [
            (r["source_type"], r["from"], r["convention"], r["abutting_rows"]) for r in _declarations(licensees)
        ] == [
            ("Field", "ff", "half_open", 1),
            ("Licence", "lf", "closed", 0),
        ]

    def test_secondary_label_target(self):
        g = kglite.KnowledgeGraph()
        g.cypher(
            "CREATE (:Status:Tracked {id: 1, vf: '2000', vt: '2001'}), (:Other:Tracked {id: 2, vf: '2003', vt: null})"
        )
        rows = _declare(g, "{node: 'Tracked', from: 'vf', to: 'vt', convention: 'closed'}").to_list()
        assert rows == [{"declared": True, "rows": 2, "abutting_rows": 0}]
        assert [(r["kind"], r["name"]) for r in _declarations(g)] == [("node", "Tracked")]

    def test_identical_redeclare_is_a_no_op(self, statuses):
        spec = "{node: 'Status', from: 'vf', to: 'vt', convention: 'half_open'}"
        _declare(statuses, spec)
        assert _declare(statuses, spec).to_list() == [{"declared": False, "rows": 0, "abutting_rows": None}]

    def test_unkeyed_declaration_is_stored_beside_keyed_ones(self, licensees):
        _declare(
            licensees,
            "{relationship: 'HAS_LICENSEE', source_type: 'Field', from: 'ff', to: 'ft', convention: 'half_open'}",
        )
        rows = _declare(licensees, "{relationship: 'HAS_LICENSEE', from: 'lf', to: 'lt', convention: 'closed'}")
        # Only the Licence relationship falls back to it; Field's are keyed.
        assert rows.to_list() == [{"declared": True, "rows": 1, "abutting_rows": 0}]
        assert [(r["source_type"], r["from"]) for r in _declarations(licensees)] == [("Field", "ff"), (None, "lf")]


class TestRefusals:
    def test_missing_target_kind(self, statuses):
        with pytest.raises(Exception, match="name the target kind"):
            _declare(statuses, "{from: 'vf', to: 'vt', convention: 'closed'}")

    def test_missing_convention(self, statuses):
        with pytest.raises(Exception, match="convention"):
            _declare(statuses, "{node: 'Status', from: 'vf', to: 'vt'}")

    def test_bad_property_name(self, statuses):
        with pytest.raises(Exception, match="property 'valid_to' does not exist on node label 'Status'"):
            _declare(statuses, "{node: 'Status', from: 'vf', to: 'valid_to', convention: 'closed'}")

    def test_dirty_bound_names_the_node(self, statuses):
        statuses.cypher("CREATE (:Status {id: 9, title: 'Odd', vf: 'someday', vt: null})")
        with pytest.raises(Exception, match=r"node '9', property 'vf'.*'someday'"):
            _declare(statuses, "{node: 'Status', from: 'vf', to: 'vt', convention: 'closed'}")
        assert _declarations(statuses) == []

    def test_dirty_bound_names_the_relationship_endpoints(self, licensees):
        licensees.cypher("MATCH (f:Field {id: 2}), (c:Company {id: 200}) CREATE (f)-[:HAS_LICENSEE {ff: 2009}]->(c)")
        with pytest.raises(Exception, match="HAS_LICENSEE relationship from node '2' to node '200'"):
            _declare(
                licensees,
                "{relationship: 'HAS_LICENSEE', source_type: 'Field', from: 'ff', to: 'ft', convention: 'closed'}",
            )

    def test_inverted_row_names_the_node(self, statuses):
        statuses.cypher("CREATE (:Status {id: 7, title: 'Backwards', vf: '2012-01-01', vt: '2011-01-01'})")
        with pytest.raises(Exception, match="node '7'.*is after the to bound"):
            _declare(statuses, "{node: 'Status', from: 'vf', to: 'vt', convention: 'closed'}")

    def test_conflicting_redeclare(self, statuses):
        _declare(statuses, "{node: 'Status', from: 'vf', to: 'vt', convention: 'half_open'}")
        with pytest.raises(Exception, match="already declared"):
            _declare(statuses, "{node: 'Status', from: 'vf', to: 'vt', convention: 'closed'}")

    def test_conflict_is_per_key(self, licensees):
        keyed = "{relationship: 'HAS_LICENSEE', source_type: 'Field', from: 'ff', to: 'ft', convention: 'closed'}"
        _declare(licensees, keyed)
        _declare(licensees, "{relationship: 'HAS_LICENSEE', from: 'ff', to: 'ft', convention: 'half_open'}")
        with pytest.raises(Exception, match="from source type 'Field' is already declared"):
            _declare(licensees, keyed.replace("'closed'", "'half_open'"))
        with pytest.raises(Exception, match="relationship type 'HAS_LICENSEE' is already declared"):
            _declare(licensees, "{relationship: 'HAS_LICENSEE', from: 'ff', to: 'ft', convention: 'closed'}")


class TestAbutmentWarning:
    def test_closed_declaration_warns_in_result_warnings(self, statuses):
        result = _declare(statuses, "{node: 'Status', from: 'vf', to: 'vt', convention: 'closed'}")
        assert result.warnings == [
            "1 of 2 rows of node label 'Status' end on the day another row of the same label begins; "
            "under convention 'closed' both rows are valid on that day. If an end bound is its "
            "successor's start, declare the interval with convention: 'half_open'."
        ]

    def test_half_open_declaration_does_not_warn(self, statuses):
        result = _declare(statuses, "{node: 'Status', from: 'vf', to: 'vt', convention: 'half_open'}")
        assert result.warnings == []


class TestHalfOpenFluent:
    """A half-open declaration answers the fluent filters half-open: the to day
    is the successor's, not the predecessor's."""

    def test_select_on_the_boundary_day(self, statuses):
        _declare(statuses, "{node: 'Status', from: 'vf', to: 'vt', convention: 'half_open'}")
        assert [r["title"] for r in statuses.date("2010-06-01").select("Status").collect()] == ["Shut down"]
        assert [r["title"] for r in statuses.date("2010-05-31").select("Status").collect()] == ["Producing"]

    def test_select_on_the_boundary_day_when_closed(self, statuses):
        _declare(statuses, "{node: 'Status', from: 'vf', to: 'vt', convention: 'closed'}")
        titles = sorted(r["title"] for r in statuses.date("2010-06-01").select("Status").collect())
        assert titles == ["Producing", "Shut down"]

    def test_valid_at_uses_the_declared_convention(self, statuses):
        _declare(statuses, "{node: 'Status', from: 'vf', to: 'vt', convention: 'half_open'}")
        got = statuses.select("Status", temporal=False).valid_at("2010-06-01").collect()
        assert [r["title"] for r in got] == ["Shut down"]

    def test_traverse_uses_each_source_keys_convention(self, licensees):
        _declare(
            licensees,
            "{relationship: 'HAS_LICENSEE', source_type: 'Field', from: 'ff', to: 'ft', convention: 'half_open'}",
        )
        field_one = licensees.select("Field").where({"title": "F1"})
        assert _titles(field_one.traverse("HAS_LICENSEE", at="2009-12-31")) == ["Beta"]
        assert _titles(field_one.traverse("HAS_LICENSEE", at="2009-12-30")) == ["Alpha"]


class TestHalfOpenTimestampEnd:
    """Under half-open a datetime ``to`` excludes only from its own time, so an
    interval ending mid-day is valid on that day: the fluent filters compare it
    with the day's midnight, not at date grain."""

    @pytest.fixture
    def shifts(self):
        g = kglite.KnowledgeGraph()
        g.cypher(
            """
            CREATE (:Shift {id: 1, title: 'Day', vf: datetime('2009-06-30T08:00'),
                            vt: datetime('2009-06-30T20:00')}),
                   (:Shift {id: 2, title: 'Month', vf: date('2009-06-01'),
                            vt: datetime('2009-06-30T12:00')}),
                   (:Shift {id: 3, title: 'Midnight', vf: date('2009-06-01'),
                            vt: datetime('2009-06-30T00:00')})
            """
        )
        _declare(g, "{node: 'Shift', from: 'vf', to: 'vt', convention: 'half_open'}")
        return g

    def test_valid_on_its_own_day(self, shifts):
        assert _titles(shifts.date("2009-06-30").select("Shift")) == ["Day", "Month"]
        assert _titles(shifts.date("2009-07-01").select("Shift")) == []

    def test_valid_during_its_own_day(self, shifts):
        got = shifts.select("Shift", temporal=False).valid_during("2009-06-30", "2009-06-30")
        assert _titles(got) == ["Day", "Month"]

    def test_overlaps_a_range_starting_on_its_day(self, shifts):
        got = shifts.select("Shift", temporal=False).valid_during("2009-06-30", "2009-07-10")
        assert _titles(got) == ["Day", "Month"]

    def test_declare_accepts_a_mid_day_end_and_refuses_a_midnight_one(self):
        g = kglite.KnowledgeGraph()
        g.cypher("CREATE (:S {id: 1, vf: date('2009-06-30'), vt: datetime('2009-06-30T18:00')})")
        _declare(g, "{node: 'S', from: 'vf', to: 'vt', convention: 'half_open'}")
        g.cypher("CREATE (:T {id: 1, vf: date('2009-06-30'), vt: datetime('2009-06-30T00:00')})")
        with pytest.raises(Exception, match="an empty interval"):
            _declare(g, "{node: 'T', from: 'vf', to: 'vt', convention: 'half_open'}")


class TestSetTemporalBesideDeclaration:
    """``set_temporal`` naming the properties a declaration already bounds keeps
    that declaration, convention included."""

    def test_relationship_declaration_is_kept(self, licensees):
        _declare(licensees, "{relationship: 'HAS_LICENSEE', from: 'ff', to: 'ft', convention: 'half_open'}")
        licensees.set_temporal("HAS_LICENSEE", "ff", "ft")
        rows = _declarations(licensees)
        assert [(r["name"], r["source_type"], r["convention"]) for r in rows] == [("HAS_LICENSEE", None, "half_open")]
        field_one = licensees.select("Field").where({"title": "F1"})
        assert _titles(field_one.traverse("HAS_LICENSEE", at="2009-12-31")) == ["Beta"]
        undeclared = licensees.cypher(
            "CALL db.temporal.undeclare({relationship: 'HAS_LICENSEE'}) YIELD undeclared RETURN undeclared"
        ).to_list()
        assert undeclared == [{"undeclared": True}]
        assert _declarations(licensees) == []

    def test_node_declaration_is_kept(self, statuses):
        _declare(statuses, "{node: 'Status', from: 'vf', to: 'vt', convention: 'half_open'}")
        statuses.set_temporal("Status", "vf", "vt")
        assert [r["convention"] for r in _declarations(statuses)] == ["half_open"]
        assert [r["title"] for r in statuses.date("2010-06-01").select("Status").collect()] == ["Shut down"]


class TestLookupOrder:
    """A relationship takes its source's keyed declaration, and the unkeyed
    one only when its source has none."""

    def test_keyed_source_and_fallback(self, licensees):
        licensees.cypher(
            "MATCH (l:Licence), (a:Company {id: 100}) "
            "CREATE (l)-[:HAS_LICENSEE {ff: '2000-01-01', ft: '2009-12-31'}]->(a)"
        )
        # Both declarations read ff/ft; only the convention differs, so the
        # to day tells which one a relationship was filtered by. Licence's
        # lf/lt relationship to Beta carries neither and always passes.
        _declare(licensees, "{relationship: 'HAS_LICENSEE', from: 'ff', to: 'ft', convention: 'closed'}")
        _declare(
            licensees,
            "{relationship: 'HAS_LICENSEE', source_type: 'Field', from: 'ff', to: 'ft', convention: 'half_open'}",
        )
        field_one = licensees.select("Field").where({"title": "F1"})
        assert _titles(field_one.traverse("HAS_LICENSEE", at="2009-12-31")) == ["Beta"]
        licence = licensees.select("Licence")
        assert _titles(licence.traverse("HAS_LICENSEE", at="2009-12-31")) == ["Alpha", "Beta"]
        assert _titles(licence.traverse("HAS_LICENSEE", at="2010-01-01")) == ["Beta"]


def _titles(selection):
    return sorted(row["title"] for row in selection.collect())


class TestDescribe:
    def test_node_declaration_attributes(self, statuses):
        _declare(statuses, "{node: 'Status', from: 'vf', to: 'vt', convention: 'half_open'}")
        assert (
            'temporal_from="vf" temporal_to="vt" temporal_convention="half_open" temporal_abutting="1"'
            in statuses.describe()
        )

    def test_several_relationship_declarations_print_once_each(self, licensees):
        _declare(
            licensees,
            "{relationship: 'HAS_LICENSEE', source_type: 'Field', from: 'ff', to: 'ft', convention: 'half_open'}",
        )
        _declare(
            licensees,
            "{relationship: 'HAS_LICENSEE', source_type: 'Licence', from: 'lf', to: 'lt', convention: 'closed'}",
        )
        xml = licensees.describe()
        assert 'temporal="Field: ff..ft half_open abutting=1; Licence: lf..lt abutting=0"' in xml
        assert "temporal_from=" not in xml

    def test_fallback_prints_last_as_other_sources(self, licensees):
        _declare(licensees, "{relationship: 'HAS_LICENSEE', from: 'lf', to: 'lt', convention: 'closed'}")
        _declare(
            licensees,
            "{relationship: 'HAS_LICENSEE', source_type: 'Field', from: 'ff', to: 'ft', convention: 'closed'}",
        )
        assert 'temporal="Field: ff..ft abutting=1; other sources: lf..lt abutting=0"' in licensees.describe()

    def test_single_keyed_declaration_attributes(self, licensees):
        _declare(
            licensees,
            "{relationship: 'HAS_LICENSEE', source_type: 'Field', from: 'ff', to: 'ft', convention: 'closed'}",
        )
        assert (
            'temporal_from="ff" temporal_to="ft" temporal_source="Field" temporal_abutting="1"' in licensees.describe()
        )


# ── Loaders and set_temporal declare through the same store ──────────────

PERIOD_TYPES = {"vf": "validFrom", "vt": "validTo"}


def _docs():
    g = kglite.KnowledgeGraph()
    g.add_nodes(pd.DataFrame({"id": [1, 2], "title": ["D1", "D2"]}), "Doc", "id", "title")
    return g


def _link(g, rows, mode=None, column_types=PERIOD_TYPES, **kwargs):
    frame = pd.DataFrame(rows, columns=["src", "tgt", "vf", "vt"])
    return g.add_connections(
        frame,
        "IN",
        "Doc",
        "src",
        "Doc",
        "tgt",
        conflict_handling=mode,
        column_types=column_types,
        **kwargs,
    )


def _periods(g):
    rows = g.cypher("MATCH (:Doc)-[r:IN]->(:Doc) RETURN r.vf AS vf, r.vt AS vt ORDER BY vf").to_list()
    return [(str(r["vf"])[:10], None if r["vt"] is None else str(r["vt"])[:10]) for r in rows]


def _counts(report):
    return report["connections_created"], report["connections_updated"]


class TestLoaderDeclarations:
    def test_repeated_and_chunked_loads_leave_one_declaration(self):
        g = _docs()
        _link(g, [(1, 2, "2000-01-01", "2005-01-01")])
        _link(g, [(2, 1, "2001-01-01", None)])
        _link(g, [(1, 1, "2002-01-01", None)])
        rows = _declarations(g)
        assert [(r["kind"], r["name"], r["source_type"], r["convention"]) for r in rows] == [
            ("relationship", "IN", None, "closed")
        ]

    def test_half_open_convention_round_trips(self):
        g = _docs()
        _link(g, [(1, 2, "2000-01-01", "2005-01-01")], convention="half_open")
        assert [r["convention"] for r in _declarations(g)] == ["half_open"]
        # A later load that names no convention keeps it.
        _link(g, [(1, 2, "2005-01-01", None)])
        assert [r["convention"] for r in _declarations(g)] == ["half_open"]

    def test_a_conflicting_convention_is_refused_before_the_load(self):
        g = _docs()
        _link(g, [(1, 2, "2000-01-01", "2005-01-01")])
        with pytest.raises(kglite.ArgumentError, match="already declared"):
            _link(g, [(1, 2, "2010-01-01", None)], convention="half_open")
        assert _periods(g) == [("2000-01-01", "2005-01-01")]

    def test_an_inverted_row_is_refused_naming_the_row(self):
        g = _docs()
        with pytest.raises(kglite.ArgumentError, match=r"row 1 \(0-based\) of the load.*is after the to bound"):
            _link(g, [(1, 2, "2000-01-01", "2005-01-01"), (2, 1, "2010-01-01", "2009-01-01")])
        assert _periods(g) == []
        assert _declarations(g) == []

    def test_a_stored_dirty_bound_refuses_the_load(self):
        g = _docs()
        g.cypher("MATCH (a:Doc {id: 1}), (b:Doc {id: 2}) CREATE (a)-[:IN {vf: 'someday'}]->(b)")
        with pytest.raises(kglite.ArgumentError, match="someday"):
            _link(g, [(2, 1, "2000-01-01", None)])
        assert _declarations(g) == []
        assert len(_periods(g)) == 1

    def test_convention_needs_validity_column_types(self):
        g = _docs()
        with pytest.raises(ValueError, match="validFrom/validTo"):
            _link(g, [(1, 2, "2000-01-01", None)], column_types=None, convention="closed")
        with pytest.raises(kglite.ArgumentError, match="'closed' or 'half_open'"):
            _link(g, [(1, 2, "2000-01-01", None)], convention="open")

    def test_the_abutment_advisory_reaches_the_loader_caller(self):
        g = _docs()
        with pytest.warns(UserWarning, match="half_open"):
            _link(g, [(1, 2, "2000-01-01", "2005-01-01"), (1, 1, "2005-01-01", None)])
        g = _docs()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _link(g, [(1, 2, "2000-01-01", "2005-01-01"), (1, 1, "2005-01-01", None)], convention="half_open")

    def test_add_nodes_refused_by_its_timeseries_declares_nothing(self):
        """Red proof: the timeseries cells were read after the declaration was
        installed, so the refused call left a declaration over a type with no
        rows, and a later ``set_temporal`` was refused as "already declared"."""
        g = kglite.KnowledgeGraph()
        frame = pd.DataFrame(
            {
                "id": [1, 1],
                "vf": ["2000-01-01", "2000-01-01"],
                "vt": [None, None],
                "t": ["2020-01", "garbage"],
                "v": [1.0, 2.0],
            }
        )
        with pytest.raises(kglite.ArgumentError):
            g.add_nodes(frame, "P", "id", column_types=PERIOD_TYPES, timeseries={"time": "t", "channels": ["v"]})
        assert _declarations(g) == []
        g.add_nodes(pd.DataFrame({"id": [1], "a": ["2000-01-01"], "b": [None]}), "P", "id")
        g.set_temporal("P", "a", "b")
        assert [r["from"] for r in _declarations(g)] == ["a"]

    def test_add_nodes_declares_and_warns(self):
        g = kglite.KnowledgeGraph()
        frame = pd.DataFrame({"id": [1, 2], "vf": ["2000-01-01", "2005-01-01"], "vt": ["2005-01-01", None]})
        with pytest.warns(UserWarning, match="node label 'Status'"):
            g.add_nodes(frame, "Status", "id", column_types=PERIOD_TYPES)
        g2 = kglite.KnowledgeGraph()
        g2.add_nodes(frame, "Status", "id", column_types=PERIOD_TYPES, convention="half_open")
        assert [(r["kind"], r["convention"]) for r in _declarations(g2)] == [("node", "half_open")]


class TestSetTemporalDeclares:
    def test_a_name_that_is_both_kinds_declares_the_node_type(self):
        g = kglite.KnowledgeGraph()
        g.cypher(
            "CREATE (:Link {id: 1, a: '2000-01-01', b: '2001-01-01'})"
            "-[:Link {a: '2000-01-01', b: '2001-01-01'}]->(:Link {id: 2})"
        )
        g.set_temporal("Link", "a", "b")
        assert [r["kind"] for r in _declarations(g)] == ["node"]

    def test_a_relationship_only_name_and_a_source_type(self):
        g = kglite.KnowledgeGraph()
        g.cypher(
            "CREATE (:A {id: 1})"
            "-[:R {a: '2000-01-01', b: '2001-01-01', c: '2002-01-01', d: '2003-01-01'}]->(:B {id: 2})"
        )
        g.set_temporal("R", "a", "b")
        g.set_temporal("R", "c", "d", convention="half_open", source_type="A")
        rows = _declarations(g)
        assert [(r["kind"], r["source_type"], r["from"], r["convention"]) for r in rows] == [
            ("relationship", "A", "c", "half_open"),
            ("relationship", None, "a", "closed"),
        ]
        with pytest.raises(kglite.ArgumentError, match="source_type applies to a relationship type"):
            g.set_temporal("A", "a", "b", source_type="A")

    def test_a_different_redeclare_is_refused(self, statuses):
        with pytest.warns(UserWarning, match="1 of 2 rows"):
            statuses.set_temporal("Status", "vf", "vt")
        with pytest.raises(kglite.ArgumentError, match="already declared"):
            statuses.set_temporal("Status", "vf", "vt", convention="half_open")

    def test_a_missing_property_is_refused(self, statuses):
        with pytest.raises(kglite.ArgumentError, match="nope"):
            statuses.set_temporal("Status", "nope", "vt")


MODES = [None, "update", "replace", "preserve", "skip", "sum"]


class TestDeclaredMergeKey:
    """A later period between the same endpoints is a parallel relationship on
    a declared type, and merges into the stored one on an undeclared type."""

    @pytest.mark.parametrize("mode", MODES)
    def test_declared_type_keeps_both_periods(self, mode):
        g = _docs()
        _link(g, [(1, 2, "2000-01-01", "2005-01-01")])
        report = _link(g, [(1, 2, "2010-01-01", None)], mode=mode)
        assert _counts(report) == (1, 0)
        assert _periods(g) == [("2000-01-01", "2005-01-01"), ("2010-01-01", None)]

    @pytest.mark.parametrize(
        ("mode", "period", "counts"),
        [
            (None, ("2010-01-01", "2005-01-01"), (0, 1)),
            ("update", ("2010-01-01", "2005-01-01"), (0, 1)),
            ("replace", ("2010-01-01", None), (0, 1)),
            ("preserve", ("2000-01-01", "2005-01-01"), (0, 1)),
            ("skip", ("2000-01-01", "2005-01-01"), (0, 0)),
            ("sum", ("2010-01-01", "2005-01-01"), (0, 1)),
        ],
    )
    def test_undeclared_type_merges_on_the_endpoints(self, mode, period, counts):
        g = _docs()
        _link(g, [(1, 2, "2000-01-01", "2005-01-01")], column_types=None)
        report = _link(g, [(1, 2, "2010-01-01", None)], mode=mode, column_types=None)
        assert _counts(report) == counts
        assert _periods(g) == [period]

    def test_the_same_start_closes_the_open_period(self):
        g = _docs()
        _link(g, [(1, 2, "2000-01-01", "2005-01-01"), (1, 2, "2010-01-01", None)])
        report = _link(g, [(1, 2, "2010-01-01", "2015-01-01")])
        assert _counts(report) == (0, 1)
        assert _periods(g) == [("2000-01-01", "2005-01-01"), ("2010-01-01", "2015-01-01")]

    def test_a_corrected_start_is_a_new_relationship(self):
        g = _docs()
        _link(g, [(1, 2, "2000-01-01", "2005-01-01")])
        report = _link(g, [(1, 2, "2000-02-01", "2005-01-01")])
        assert _counts(report) == (1, 0)
        g.cypher("MATCH (:Doc)-[r:IN {vf: date('2000-01-01')}]->(:Doc) DELETE r")
        assert _periods(g) == [("2000-02-01", "2005-01-01")]

    def test_the_constraint_gate_uses_the_same_key(self):
        g = _docs()
        frame = pd.DataFrame({"src": [1], "tgt": [2], "vf": ["2000-01-01"], "vt": ["2005-01-01"], "x": [7]})
        g.add_connections(frame, "IN", "Doc", "src", "Doc", "tgt", column_types=PERIOD_TYPES)
        g.cypher("CREATE CONSTRAINT FOR ()-[r:IN]-() REQUIRE r.x IS NOT NULL")
        with pytest.raises(kglite.ConstraintViolationError):
            _link(g, [(1, 2, "2010-01-01", None)])
        assert _periods(g) == [("2000-01-01", "2005-01-01")]
        report = _link(g, [(1, 2, "2000-01-01", "2006-01-01")])
        assert _counts(report) == (0, 1)


class TestTheMergeKeyReadsTheStart:
    """One start written as a date, a midnight datetime or an ISO string is
    one period: re-loading it in another spelling merges into the stored
    relationship instead of adding a parallel one.

    Red proof: the key compared raw values, so each case below created a
    second relationship for the same period."""

    @staticmethod
    def _cypher_period(start):
        g = _docs()
        g.cypher(
            f"MATCH (a:Doc {{id: 1}}), (b:Doc {{id: 2}}) CREATE (a)-[:IN {{vf: {start}, vt: date('2005-01-01')}}]->(b)"
        )
        g.set_temporal("IN", "vf", "vt")
        return g

    def test_a_datetime64_column_reloaded_as_dates_merges(self):
        g = _docs()
        g.add_nodes(pd.DataFrame({"id": [3], "title": ["D3"]}), "Doc", "id", "title")
        # One row with a time of day types the whole column as datetimes, so
        # 1 -> 2 stores a midnight datetime.
        first = pd.DataFrame(
            {
                "src": [1, 2],
                "tgt": [2, 3],
                "vf": pd.to_datetime(["2009-01-01 00:00", "2010-05-05 12:30"]),
                "vt": pd.to_datetime([None, "2011-01-01"]),
            }
        )
        g.add_connections(first, "IN", "Doc", "src", "Doc", "tgt")
        g.set_temporal("IN", "vf", "vt")
        # Midnight-only: typed as dates.
        closing = pd.DataFrame(
            {"src": [1], "tgt": [2], "vf": pd.to_datetime(["2009-01-01"]), "vt": pd.to_datetime(["2012-01-01"])}
        )
        report = g.add_connections(closing, "IN", "Doc", "src", "Doc", "tgt")
        assert _counts(report) == (0, 1)
        rows = g.cypher("MATCH (:Doc {id: 1})-[r:IN]->(:Doc {id: 2}) RETURN r.vt AS vt").to_list()
        assert [str(r["vt"])[:10] for r in rows] == ["2012-01-01"]

    @pytest.mark.parametrize(
        "stored", ["date('2000-01-01')", "datetime('2000-01-01T00:00:00')", "'2000-01-01'", "'2000-01-01T00:00:00Z'"]
    )
    @pytest.mark.parametrize("mode", MODES)
    def test_every_spelling_of_one_start_merges(self, stored, mode):
        g = self._cypher_period(stored)
        report = _link(g, [(1, 2, "2000-01-01", "2006-01-01")], mode=mode)
        assert _counts(report) == ((0, 0) if mode == "skip" else (0, 1))
        assert len(_periods(g)) == 1

    def test_a_datetime_within_the_day_is_its_own_start(self):
        g = self._cypher_period("datetime('2000-01-01T12:30:00')")
        report = _link(g, [(1, 2, "2000-01-01", "2006-01-01")])
        assert _counts(report) == (1, 0)
        assert len(_periods(g)) == 2

    def test_the_constraint_gate_reads_the_start_as_the_loader_does(self):
        g = _docs()
        g.cypher(
            "MATCH (a:Doc {id: 1}), (b:Doc {id: 2}) "
            "CREATE (a)-[:IN {vf: datetime('2000-01-01T00:00:00'), vt: date('2005-01-01'), x: 7}]->(b)"
        )
        g.set_temporal("IN", "vf", "vt")
        g.cypher("CREATE CONSTRAINT FOR ()-[r:IN]-() REQUIRE r.x IS NOT NULL")
        report = _link(g, [(1, 2, "2000-01-01", "2006-01-01")])
        assert _counts(report) == (0, 1)
        with pytest.raises(kglite.ConstraintViolationError, match=r"IN\.x"):
            _link(g, [(1, 2, "2000-01-02", None)])
        assert len(_periods(g)) == 1

    def test_extend_merges_a_period_spelled_differently_in_each_graph(self):
        target = self._cypher_period("datetime('2000-01-01T00:00:00')")
        source = self._cypher_period("date('2000-01-01')")
        report = target.extend(source)
        assert (report["edges_created"], report["edges_updated"]) == (0, 1)
        assert len(_periods(target)) == 1


class TestCreateRelationshipsKeysBySource:
    """``create_relationships()`` without ``source_type=`` keys each edge on
    the declaration covering its own source type.

    Red proof: without ``source_type=`` only an unkeyed declaration was
    consulted, so a second period between the same pair merged into the
    first."""

    @staticmethod
    def _graph():
        g = kglite.KnowledgeGraph()
        g.cypher(
            """
            CREATE (a:A {id: 1, title: 'A1'}), (b:B {id: 10, title: 'B10'}),
                   (c:C {id: 100, title: 'C100', vf: date('2010-01-01')}),
                   (a)-[:AB]->(b), (b)-[:BC]->(c),
                   (a)-[:R {vf: date('2000-01-01'), vt: date('2005-01-01')}]->(c)
            """
        )
        g.set_temporal("R", "vf", "vt", source_type="A")
        return g

    @pytest.mark.parametrize("source_type", [None, "A"])
    def test_a_new_period_is_a_new_relationship(self, source_type):
        g = (
            self._graph()
            .select("A")
            .traverse("AB")
            .traverse("BC")
            .create_relationships("R", properties={"C": ["vf"]}, source_type=source_type)
        )
        rows = g.cypher("MATCH (:A)-[r:R]->(:C) RETURN r.vf AS vf ORDER BY vf").to_list()
        assert [str(r["vf"])[:10] for r in rows] == ["2000-01-01", "2010-01-01"]

    def test_mixed_sources_each_key_on_their_own_declaration(self):
        """A source level holding two node types keys each edge on its own
        type's declaration: both edges restate their stored period, so both
        merge. Keying either on the other type's ``from`` reads it as absent
        and adds a parallel relationship."""

        def build():
            g = kglite.KnowledgeGraph()
            g.cypher(
                """
                CREATE (a:A {id: 1, title: 'A1'}), (x:X:A {id: 2, title: 'X2'}),
                       (b:B {id: 10, title: 'B10'}),
                       (c:C {id: 100, title: 'C100', vf: date('2000-01-01'), xf: date('2000-01-01')}),
                       (a)-[:AB]->(b), (x)-[:AB]->(b), (b)-[:BC]->(c),
                       (a)-[:R {vf: date('2000-01-01'), vt: date('2005-01-01')}]->(c),
                       (x)-[:R {xf: date('2000-01-01'), xt: date('2005-01-01')}]->(c)
                """
            )
            g.set_temporal("R", "vf", "vt", source_type="A")
            g.set_temporal("R", "xf", "xt", source_type="X")
            return g

        g = (
            build()
            .select("A", include_secondary=True)
            .traverse("AB")
            .traverse("BC")
            .create_relationships("R", properties={"C": ["vf", "xf"]})
        )
        rows = g.cypher("MATCH (s)-[r:R]->(:C) RETURN s.title AS s, count(r) AS n ORDER BY s").to_list()
        assert rows == [{"s": "A1", "n": 1}, {"s": "X2", "n": 1}]


class TestAnAmbiguousTypeKeysOnTheDeclarationARowCarries:
    """A legacy type holding two unkeyed declarations keys each row on the
    first declaration whose ``from`` it carries.

    Red proof: every row keyed on the first declaration's ``from``, so rows
    bounded by the second declaration keyed on nothing and collapsed."""

    def test_second_declaration_periods_stay_apart(self, tmp_path):
        fixture = Path(__file__).parent / "fixtures" / "temporal_legacy" / "distinct_duplicates.kgl"
        copy = tmp_path / "distinct_duplicates.kgl"
        shutil.copy(fixture, copy)
        g = kglite.load(str(copy))

        def other(start):
            frame = pd.DataFrame({"field": [1], "company": [10], "other_from": pd.to_datetime([start])})
            return g.add_connections(frame, "HAS_LICENSEE", "Field", "field", "Company", "company")

        assert _counts(other("2015-01-01")) == (1, 0)
        assert _counts(other("2020-01-01")) == (1, 0)
        assert _counts(other("2020-01-01")) == (0, 1)
        rows = g.cypher("MATCH (:Field {id: 1})-[r:HAS_LICENSEE]->(:Company {id: 10}) RETURN count(r) AS n").to_list()
        assert rows == [{"n": 3}]
