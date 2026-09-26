"""Validity-interval declarations: ``db.temporal.declare`` / ``undeclare`` /
``declarations`` and what a declaration changes in the fluent filters and in
``describe()``.

Red proof: before the procedures existed every call here failed as an unknown
procedure.
"""

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
