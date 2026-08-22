"""Tests for schema locking: lock_schema(), unlock_schema(), schema_locked."""

import pytest

import kglite

# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_graph():
    """Build a small graph with Person and Paper types + AUTHORED edges."""
    g = kglite.KnowledgeGraph()
    import pandas as pd

    persons = pd.DataFrame({"pid": [1, 2], "name": ["Alice", "Bob"], "age": [30, 25]})
    g.add_nodes(persons, "Person", "pid", node_title_field="name")

    papers = pd.DataFrame({"doi": ["10.1", "10.2"], "title": ["Paper A", "Paper B"], "year": [2020, 2021]})
    g.add_nodes(papers, "Paper", "doi", node_title_field="title")

    edges = pd.DataFrame({"pid": [1, 2], "doi": ["10.1", "10.2"]})
    g.add_connections(edges, "AUTHORED", "Person", "pid", "Paper", "doi")

    return g


# ── API basics ───────────────────────────────────────────────────────────────


class TestSchemaLockAPI:
    def test_default_unlocked(self):
        g = kglite.KnowledgeGraph()
        assert g.schema_locked is False

    def test_lock_unlock_toggle(self):
        g = _make_graph()
        g.lock_schema()
        assert g.schema_locked is True
        g.unlock_schema()
        assert g.schema_locked is False

    def test_lock_returns_self(self):
        g = _make_graph()
        result = g.lock_schema()
        assert result is not None  # returns Self for chaining


# ── CREATE node validation ───────────────────────────────────────────────────


class TestCreateNodeValidation:
    def test_create_valid_node(self):
        g = _make_graph()
        g.lock_schema()
        g.cypher("CREATE (p:Person {name: 'Carol', age: 35})")

    def test_create_unknown_node_type(self):
        g = _make_graph()
        g.lock_schema()
        with pytest.raises(kglite.KgError, match="Unknown node type 'Persom'"):
            g.cypher("CREATE (p:Persom {name: 'x'})")

    def test_create_unknown_type_suggests(self):
        g = _make_graph()
        g.lock_schema()
        with pytest.raises(kglite.KgError, match="Did you mean 'Person'"):
            g.cypher("CREATE (p:Persom {name: 'x'})")

    def test_create_unknown_property(self):
        g = _make_graph()
        g.lock_schema()
        with pytest.raises(kglite.KgError, match="Unknown property 'agee' on Person"):
            g.cypher("CREATE (p:Person {name: 'x', agee: 30})")

    def test_create_unknown_property_suggests(self):
        g = _make_graph()
        g.lock_schema()
        with pytest.raises(kglite.KgError, match="Did you mean 'age'"):
            g.cypher("CREATE (p:Person {name: 'x', agee: 30})")

    def test_create_type_mismatch(self):
        g = _make_graph()
        g.lock_schema()
        with pytest.raises(kglite.KgError, match="expects integer, got string"):
            g.cypher("CREATE (p:Person {name: 'x', age: 'thirty'})")


# ── CREATE edge validation ───────────────────────────────────────────────────


class TestCreateEdgeValidation:
    def test_create_valid_edge(self):
        g = _make_graph()
        g.lock_schema()
        g.cypher("""
            MATCH (p:Person {name: 'Alice'}), (pa:Paper {title: 'Paper B'})
            CREATE (p)-[:AUTHORED]->(pa)
        """)

    def test_create_unknown_edge_type(self):
        g = _make_graph()
        g.lock_schema()
        with pytest.raises(kglite.KgError, match="Unknown edge type 'WRITES'"):
            g.cypher("""
                MATCH (p:Person {name: 'Alice'}), (pa:Paper {title: 'Paper B'})
                CREATE (p)-[:WRITES]->(pa)
            """)

    def test_create_invalid_edge_endpoints(self):
        g = _make_graph()
        g.lock_schema()
        with pytest.raises(kglite.KgError, match="AUTHORED edges connect"):
            g.cypher("""
                MATCH (a:Paper {title: 'Paper A'}), (b:Paper {title: 'Paper B'})
                CREATE (a)-[:AUTHORED]->(b)
            """)


# ── SET validation ───────────────────────────────────────────────────────────


class TestSetValidation:
    def test_set_valid_property(self):
        g = _make_graph()
        g.lock_schema()
        g.cypher("MATCH (p:Person {name: 'Alice'}) SET p.age = 31")

    def test_set_unknown_property(self):
        g = _make_graph()
        g.lock_schema()
        with pytest.raises(kglite.KgError, match="Unknown property 'salary' on Person"):
            g.cypher("MATCH (p:Person {name: 'Alice'}) SET p.salary = 100000")

    def test_set_type_mismatch(self):
        g = _make_graph()
        g.lock_schema()
        with pytest.raises(kglite.KgError, match="expects integer, got string"):
            g.cypher("MATCH (p:Person {name: 'Alice'}) SET p.age = 'old'")

    def test_set_title_always_allowed(self):
        g = _make_graph()
        g.lock_schema()
        g.cypher("MATCH (p:Person {name: 'Alice'}) SET p.title = 'Dr.'")

    def test_set_name_always_allowed(self):
        g = _make_graph()
        g.lock_schema()
        g.cypher("MATCH (p:Person {name: 'Alice'}) SET p.name = 'Alicia'")


# ── MERGE validation ────────────────────────────────────────────────────────


class TestMergeValidation:
    def test_merge_valid_existing(self):
        g = _make_graph()
        g.lock_schema()
        # Should match existing node — no creation needed
        g.cypher("MERGE (p:Person {name: 'Alice'})")

    def test_merge_valid_new(self):
        g = _make_graph()
        g.lock_schema()
        # Should create new node — valid type and properties
        g.cypher("MERGE (p:Person {name: 'Zara', age: 28})")

    def test_merge_unknown_type(self):
        g = _make_graph()
        g.lock_schema()
        with pytest.raises(kglite.KgError, match="Unknown node type 'Auther'"):
            g.cypher("MERGE (a:Auther {name: 'x'})")


# ── Unlock behavior ─────────────────────────────────────────────────────────


class TestUnlock:
    def test_unlock_allows_unknown_type(self):
        g = _make_graph()
        g.lock_schema()
        g.unlock_schema()
        # Should succeed now — schema unlocked
        g.cypher("CREATE (x:NewType {name: 'anything'})")

    def test_valid_reads_always_allowed(self):
        g = _make_graph()
        g.lock_schema()
        # A read against a *known* label is never blocked by the lock — the
        # lock rejects typos, it does not gate reading.
        result = g.cypher("MATCH (p:Person) RETURN p.name")
        assert len(result) == 2

    def test_delete_always_allowed(self):
        g = _make_graph()
        g.lock_schema()
        # DELETE should work even when schema locked
        g.cypher("MATCH (p:Person {name: 'Bob'}) DETACH DELETE p")


# ── Read-side label validation ───────────────────────────────────────────────
#
# `lock_schema()` is the opt-in "catch my typos" mechanism. It has always
# rejected an unknown *property* and an unknown *node type* on writes, but a
# typo'd label in a MATCH used to return `[]` with no error — and an empty
# result set reads as "no matching data" rather than "you made a mistake", so
# it survives review and reaches production. These lock the symmetry in.


class TestMatchLabelValidation:
    def test_unknown_label_in_match_raises(self):
        g = _make_graph()
        g.lock_schema()
        with pytest.raises(kglite.SchemaError, match="Unknown node type 'Persom'"):
            g.cypher("MATCH (p:Persom) RETURN p")

    def test_error_enumerates_the_valid_labels(self):
        # Mirrors the unknown-property message, which lists valid properties.
        g = _make_graph()
        g.lock_schema()
        with pytest.raises(kglite.SchemaError, match=r"Valid types: Paper, Person"):
            g.cypher("MATCH (p:Persom) RETURN p")

    def test_error_suggests_the_near_miss(self):
        g = _make_graph()
        g.lock_schema()
        with pytest.raises(kglite.SchemaError, match="Did you mean 'Person'"):
            g.cypher("MATCH (p:Persom) RETURN p")

    def test_error_carries_the_schema_code(self):
        # Applications branch on `.code`, not on message prose.
        g = _make_graph()
        g.lock_schema()
        with pytest.raises(kglite.SchemaError) as exc:
            g.cypher("MATCH (p:Persom) RETURN p")
        assert exc.value.code == "Schema"

    @pytest.mark.parametrize(
        "query",
        [
            # Every clause that can carry a label — covering only MATCH would
            # recreate the same asymmetry one level down.
            "MATCH (p:Persom) RETURN p",
            "MATCH (p:Person) OPTIONAL MATCH (q:Persom) RETURN p, q",
            "MATCH (p:Person) WHERE EXISTS { MATCH (q:Persom) } RETURN p",
            "CALL { MATCH (q:Persom) RETURN q } RETURN q",
            "MATCH (p:Person) RETURN p.name AS n UNION MATCH (q:Persom) RETURN q.name AS n",
            "MATCH (n:Person:Persom) RETURN n",
            "MERGE (q:Persom {name: 'x'})",
        ],
    )
    def test_every_label_carrying_clause_is_covered(self, query):
        g = _make_graph()
        g.lock_schema()
        with pytest.raises(kglite.KgError, match="Unknown node type 'Persom'"):
            g.cypher(query)

    @pytest.mark.parametrize(
        "query",
        [
            "MATCH (p:Person) RETURN p",
            "MATCH (p:Person) OPTIONAL MATCH (q:Paper) RETURN p, q",
            "MATCH (p:Person) WHERE EXISTS { MATCH (q:Paper) } RETURN p",
            "CALL { MATCH (q:Paper) RETURN q } RETURN q",
            "MATCH (p:Person) RETURN p.name AS n UNION MATCH (q:Paper) RETURN q.title AS n",
            "MATCH (n) RETURN n",
            "MATCH (p:Person)-[:AUTHORED]->(q:Paper) RETURN p, q",
        ],
    )
    def test_known_labels_are_never_rejected(self, query):
        g = _make_graph()
        g.lock_schema()
        g.cypher(query)  # must not raise

    def test_unlocking_restores_the_zero_row_idiom(self):
        g = _make_graph()
        g.lock_schema()
        g.unlock_schema()
        assert len(g.cypher("MATCH (p:Persom) RETURN p")) == 0


class TestOpenSchemaUnchanged:
    """The schemaless default is the product — it must be untouched."""

    @pytest.mark.parametrize(
        ("query", "rows"),
        [
            ("MATCH (p:Nonexistent) RETURN p", 0),
            ("CALL { MATCH (q:Nonexistent) RETURN q } RETURN q", 0),
            (
                "MATCH (p:Person) RETURN p.name AS n UNION MATCH (q:Nonexistent) RETURN q.name AS n",
                2,
            ),
            # OPTIONAL MATCH keeps the left rows with a null right side —
            # still no error, which is the point.
            ("MATCH (p:Person) OPTIONAL MATCH (q:Nonexistent) RETURN p, q", 2),
        ],
    )
    def test_unknown_label_does_not_error(self, query, rows):
        # The existence-check idiom: an unknown label is legal Cypher on an
        # open schema and simply matches nothing.
        g = _make_graph()
        assert g.schema_locked is False
        assert len(g.cypher(query)) == rows

    def test_create_of_a_brand_new_label_still_works(self):
        g = _make_graph()
        g.cypher("CREATE (x:BrandNewType {name: 'anything'})")
        assert len(g.cypher("MATCH (x:BrandNewType) RETURN x")) == 1

    def test_merge_of_a_brand_new_label_still_works(self):
        g = _make_graph()
        g.cypher("MERGE (x:AnotherNewType {name: 'anything'})")
        assert len(g.cypher("MATCH (x:AnotherNewType) RETURN x")) == 1


# ── Read-side property validation ────────────────────────────────────────────
#
# The label symmetry above closed one half of the read side. The other half is
# the property: `MATCH (p:Person) WHERE p.agee = 1` returns `[]` and
# `RETURN p.agee` returns a column of nulls next to correct-looking siblings —
# both indistinguishable from a legitimate empty or sparse result, and both a
# warning only. Under the lock they are errors; unlocked they are unchanged.


class TestMatchPropertyValidation:
    def test_absent_property_in_return_raises(self):
        g = _make_graph()
        g.lock_schema()
        with pytest.raises(kglite.SchemaError, match="Did you mean 'age'"):
            g.cypher("MATCH (p:Person) RETURN p.agee")

    def test_absent_property_in_where_raises(self):
        g = _make_graph()
        g.lock_schema()
        with pytest.raises(kglite.SchemaError) as exc:
            g.cypher("MATCH (p:Person) WHERE p.agee = 1 RETURN p")
        message = str(exc.value)
        assert "Unknown property 'agee' on Person, referenced in WHERE" in message
        # Actionable without a trip to describe(): the valid set and the way out.
        assert "Valid properties: age" in message
        assert "unlock_schema()" in message

    def test_error_carries_the_schema_code(self):
        g = _make_graph()
        g.lock_schema()
        with pytest.raises(kglite.SchemaError) as exc:
            g.cypher("MATCH (p:Person) RETURN p.agee")
        assert exc.value.code == "Schema"

    @pytest.mark.parametrize(
        "query",
        [
            "MATCH (p:Person) WHERE p.agee = 1 RETURN p",
            "MATCH (p:Person) RETURN p.agee",
            "MATCH (p:Person) WITH p.agee AS a RETURN a",
            "MATCH (p:Person) RETURN p ORDER BY p.agee",
            # A mutation's selector goes through the same prepare step.
            "MATCH (p:Person) WHERE p.agee = 1 SET p.age = 99",
            # EXPLAIN is rejected too, exactly as it already was for a typo'd
            # label — the check runs before the plan is rendered.
            "EXPLAIN MATCH (p:Person) RETURN p.agee",
        ],
    )
    def test_every_reading_clause_is_covered(self, query):
        g = _make_graph()
        g.lock_schema()
        with pytest.raises(kglite.SchemaError, match="agee"):
            g.cypher(query)

    @pytest.mark.parametrize(
        "query",
        [
            # A real property, and the built-ins, are never rejected.
            "MATCH (p:Person) RETURN p.age, p.id, p.title, p.name",
            # No label to reason from.
            "MATCH (n) WHERE n.whatever = 1 RETURN n",
            # `WITH n AS m` rebinds a var the label map never tracked.
            "MATCH (n:Person) WITH n AS m RETURN m.agee",
            # A property of a *different* known type is still not this type's.
            "MATCH (a:Paper) RETURN a.year",
        ],
    )
    def test_valid_reads_are_never_rejected(self, query):
        g = _make_graph()
        g.lock_schema()
        g.cypher(query)

    def test_a_sparse_property_is_not_a_typo(self):
        """One node carrying a property makes it known, however many are null."""
        import pandas as pd

        g = _make_graph()
        g.add_nodes(
            pd.DataFrame({"pid": [3], "name": ["Cara"], "age": [41], "nickname": ["C"]}),
            "Person",
            "pid",
            node_title_field="name",
        )
        g.lock_schema()
        rows = g.cypher("MATCH (p:Person) RETURN p.nickname").to_list()
        assert sum(1 for r in rows if r["p.nickname"] is None) >= 2

    def test_the_session_and_transaction_paths_share_the_rule(self):
        g = _make_graph()
        g.lock_schema()
        session = g.session()
        with pytest.raises(kglite.SchemaError, match="agee"):
            session.cypher("MATCH (p:Person) RETURN p.agee")
        with pytest.raises(kglite.SchemaError, match="agee"):
            session.execute("MATCH (p:Person) WHERE p.agee = 1 SET p.age = 99")

        tx = g.begin()
        try:
            with pytest.raises(kglite.SchemaError, match="agee"):
                tx.cypher("MATCH (p:Person) RETURN p.agee")
        finally:
            tx.rollback()

    def test_locking_after_the_plan_was_cached_still_raises(self):
        """The plan cache must not carry a pre-lock verdict past the lock."""
        g = _make_graph()
        query = "MATCH (p:Person) RETURN p.agee"
        for _ in range(2):
            assert len(g.cypher(query)) == 2
        g.lock_schema()
        with pytest.raises(kglite.SchemaError, match="agee"):
            g.cypher(query)
        g.unlock_schema()
        assert len(g.cypher(query)) == 2

    def test_unlocked_returns_nulls_with_a_warning_instead(self):
        """The pair, side by side: the default is unchanged and still explains itself."""
        g = _make_graph()
        assert g.schema_locked is False
        result = g.cypher("MATCH (p:Person) RETURN p.name, p.agee")
        assert [row["p.agee"] for row in result.to_list()] == [None, None]
        assert any("Did you mean 'age'?" in w for w in result.warnings), result.warnings

    def test_a_reversed_arrow_stays_a_warning_under_the_lock(self):
        g = _make_graph()
        g.lock_schema()
        result = g.cypher("MATCH (a:Paper)-[:AUTHORED]->(p:Person) RETURN p")
        assert len(result) == 0
        assert any("Reverse the arrow?" in w for w in result.warnings), result.warnings

    def test_an_unknown_relationship_type_stays_a_warning_under_the_lock(self):
        g = _make_graph()
        g.lock_schema()
        result = g.cypher("MATCH (p:Person)-[:AUTHRED]->(a:Paper) RETURN p")
        assert len(result) == 0
        assert any("AUTHRED" in w for w in result.warnings), result.warnings


# ── Declared-type mismatches under the lock ──────────────────────────────────
#
# The third promotion, and the one with a boundary inside it: a comparison an
# `IS :: T` constraint makes vacuous is an error under a lock (the write path
# enforces the declaration), while the same claim from `define_schema()` is a
# warning in every schema state (nothing enforces it). Every case below is a
# pair, because a build that promoted both halves would pass any single one.


def _declared_graph():
    """`_make_graph()` with a DDL declaration behind `Person.age`."""
    g = _make_graph()
    g.cypher("CREATE CONSTRAINT person_age_typed FOR (p:Person) REQUIRE p.age IS :: INTEGER")
    return g


def _schema_defined_graph():
    """The same claim from `define_schema()` — declared intent, unenforced."""
    g = _make_graph()
    g.define_schema({"nodes": {"Person": {"types": {"age": "integer"}}}})
    return g


CROSS_TYPE = "MATCH (p:Person) WHERE p.age > 'forty' RETURN p.name"


class TestDeclaredTypeMismatch:
    def test_a_declared_type_mismatch_raises_under_the_lock(self):
        g = _declared_graph()
        g.lock_schema()
        with pytest.raises(kglite.SchemaError) as exc:
            g.cypher(CROSS_TYPE)
        message = str(exc.value)
        # Both type names, so the reader sees the mismatch itself and not just
        # that something was wrong.
        assert "Person.age (declared INTEGER)" in message
        assert "STRING literal 'forty'" in message
        assert "unlock_schema()" in message

    def test_a_schema_defined_type_mismatch_still_only_warns(self):
        """The twin, same query text: it runs, and it explains itself."""
        g = _schema_defined_graph()
        g.lock_schema()
        rv = g.cypher(CROSS_TYPE)
        assert rv.to_list() == []  # the comparison really is vacuous
        assert any("Person.age (schema-defined integer)" in w for w in rv.warnings), rv.warnings
        # ...and a query the same declaration makes *true* everywhere returns
        # its rows, so "runs" is not just "did not raise".
        assert len(g.cypher("MATCH (p:Person) WHERE p.age <> 'forty' RETURN p.name")) == 2

    def test_unlocking_restores_the_warning(self):
        g = _declared_graph()
        g.lock_schema()
        with pytest.raises(kglite.SchemaError):
            g.cypher(CROSS_TYPE)
        g.unlock_schema()
        rv = g.cypher(CROSS_TYPE)
        assert any("declared INTEGER" in w for w in rv.warnings), rv.warnings

    def test_a_bound_parameter_promotes_with_the_property_it_meets(self):
        """The mistake the query text cannot show — and the reason the verdict
        is per call: the same statement with an integer bound is fine."""
        g = _declared_graph()
        g.lock_schema()
        query = "MATCH (p:Person) WHERE p.age > $cutoff RETURN p.name"
        with pytest.raises(kglite.SchemaError, match=r"STRING parameter \$cutoff"):
            g.cypher(query, params={"cutoff": "forty"})
        assert len(g.cypher(query, params={"cutoff": 26})) == 1

    def test_a_well_typed_comparison_is_never_rejected(self):
        g = _declared_graph()
        g.lock_schema()
        assert len(g.cypher("MATCH (p:Person) WHERE p.age > 20 RETURN p")) == 2
        assert len(g.cypher("MATCH (p:Person) WHERE p.age > 20.5 RETURN p")) == 2
        # No declaration behind the property: observed metadata is not a source.
        g.cypher("MATCH (a:Paper) WHERE a.year > 'x' RETURN a")

    def test_locking_after_the_plan_was_cached_still_raises(self):
        """A plan primed while the schema was open must not outrun the lock."""
        g = _declared_graph()
        for _ in range(2):
            assert g.cypher(CROSS_TYPE).to_list() == []
        g.lock_schema()
        with pytest.raises(kglite.SchemaError, match="declared INTEGER"):
            g.cypher(CROSS_TYPE)

    def test_the_session_and_transaction_paths_share_the_rule(self):
        g = _declared_graph()
        g.lock_schema()
        session = g.session()
        with pytest.raises(kglite.SchemaError, match="declared INTEGER"):
            session.cypher(CROSS_TYPE)
        with pytest.raises(kglite.SchemaError, match="declared INTEGER"):
            session.execute("MATCH (p:Person) WHERE p.age > 'forty' SET p.age = 99")

        tx = g.begin()
        try:
            with pytest.raises(kglite.SchemaError, match="declared INTEGER"):
                tx.cypher(CROSS_TYPE)
        finally:
            tx.rollback()


# ── Introspection ────────────────────────────────────────────────────────────


class TestIntrospection:
    def test_describe_shows_schema_locked(self):
        g = _make_graph()
        g.lock_schema()
        desc = g.describe()
        assert "schema-locked" in desc

    def test_describe_no_notice_when_unlocked(self):
        g = _make_graph()
        desc = g.describe()
        assert "schema-locked" not in desc
