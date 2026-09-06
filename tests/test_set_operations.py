"""Tests for set operations: union, intersection, difference, symmetric_difference."""

import pytest

import kglite

SET_OPERATIONS = ("union", "intersection", "difference", "symmetric_difference")
STORAGE_MODES = ("memory", "mapped", "disk")


def _graph(mode, tmp_path, name):
    path = str(tmp_path / name) if mode == "disk" else None
    return kglite.KnowledgeGraph(storage=mode, path=path)


def _selection_ids(selection):
    return sorted(row["id"] for row in selection.collect().to_list())


@pytest.mark.parametrize("mode", STORAGE_MODES)
@pytest.mark.parametrize("operation", SET_OPERATIONS)
def test_set_operations_reject_unrelated_graph_views(mode, operation, tmp_path):
    left_graph = _graph(mode, tmp_path, f"{mode}-left")
    right_graph = _graph(mode, tmp_path, f"{mode}-right")
    left_graph.cypher("CREATE (:N {id: 1, title: 'Alpha'})")
    right_graph.cypher("CREATE (:N {id: 2, title: 'Beta'})")

    with pytest.raises(ValueError, match="same immutable graph view"):
        getattr(left_graph.select("N"), operation)(right_graph.select("N"))


@pytest.mark.parametrize("mode", STORAGE_MODES)
@pytest.mark.parametrize(
    ("operation", "expected"),
    [
        ("union", [1, 2, 3]),
        ("intersection", [2]),
        ("difference", [1]),
        ("symmetric_difference", [1, 3]),
    ],
)
def test_set_operations_accept_same_immutable_view(mode, operation, expected, tmp_path):
    graph = _graph(mode, tmp_path, f"{mode}-same-view")
    graph.cypher("CREATE (:N {id: 1, title: 'One'}), (:N {id: 2, title: 'Two'}), (:N {id: 3, title: 'Three'})")
    left = graph.select("N").where({"id": {"<": 3}})
    right = graph.select("N").where({"id": {">": 1}})

    assert _selection_ids(getattr(left, operation)(right)) == expected


def test_set_operations_reject_explicit_copy():
    graph = kglite.KnowledgeGraph()
    graph.cypher("CREATE (:N {id: 1, title: 'One'})")

    with pytest.raises(ValueError, match="same immutable graph view"):
        graph.select("N").union(graph.copy().select("N"))


@pytest.mark.parametrize("mode", STORAGE_MODES)
def test_set_operations_reject_view_retained_before_mutation_and_vacuum(mode, tmp_path):
    graph = _graph(mode, tmp_path, f"{mode}-remapped")
    graph.cypher("CREATE (:N {id: 1, title: 'One'}), (:N {id: 2, title: 'Two'}), (:N {id: 3, title: 'Three'})")
    before = graph.select("N")
    graph.cypher("MATCH (n:N {id: 1}) DETACH DELETE n")
    graph.vacuum()
    after = graph.select("N")

    with pytest.raises(ValueError, match="same immutable graph view"):
        before.union(after)


def test_set_operations_reject_equal_step_sibling_divergence():
    graph = kglite.KnowledgeGraph()
    graph.cypher("CREATE (:N {id: 1, title: 'Original'})")
    left_owner = graph.select("N")
    right_owner = graph.select("N")

    # Each sibling performs one write from the same base. A lineage/version
    # approximation can call these compatible even though their data differs.
    left_owner.cypher("CREATE (:N {id: 2, title: 'Left'})")
    right_owner.cypher("CREATE (:N {id: 3, title: 'Right'})")

    with pytest.raises(ValueError, match="same immutable graph view"):
        left_owner.select("N").union(right_owner.select("N"))


class TestUnion:
    def test_union_basic(self, social_graph):
        oslo = social_graph.select("Person").where({"city": "Oslo"})
        bergen = social_graph.select("Person").where({"city": "Bergen"})
        combined = oslo.union(bergen)
        assert combined.len() == oslo.len() + bergen.len()

    def test_union_with_overlap(self, social_graph):
        young = social_graph.select("Person").where({"age": {"<": 30}})
        oslo = social_graph.select("Person").where({"city": "Oslo"})
        combined = young.union(oslo)
        assert combined.len() >= max(young.len(), oslo.len())

    def test_union_with_self(self, social_graph):
        oslo = social_graph.select("Person").where({"city": "Oslo"})
        combined = oslo.union(oslo)
        assert combined.len() == oslo.len()

    def test_union_with_empty(self, social_graph):
        oslo = social_graph.select("Person").where({"city": "Oslo"})
        empty = social_graph.select("NonExistent")
        combined = oslo.union(empty)
        assert combined.len() == oslo.len()


class TestIntersection:
    def test_intersection_basic(self, social_graph):
        oslo = social_graph.select("Person").where({"city": "Oslo"})
        old = social_graph.select("Person").where({"age": {">": 35}})
        result = oslo.intersection(old)
        nodes = result.collect()
        for n in nodes:
            assert n["city"] == "Oslo"
            assert n["age"] > 35

    def test_intersection_with_empty(self, social_graph):
        oslo = social_graph.select("Person").where({"city": "Oslo"})
        empty = social_graph.select("NonExistent")
        result = oslo.intersection(empty)
        assert result.len() == 0


class TestDifference:
    def test_difference_basic(self, social_graph):
        all_people = social_graph.select("Person")
        oslo = social_graph.select("Person").where({"city": "Oslo"})
        non_oslo = all_people.difference(oslo)
        nodes = non_oslo.collect()
        for n in nodes:
            assert n["city"] != "Oslo"

    def test_difference_with_self(self, social_graph):
        oslo = social_graph.select("Person").where({"city": "Oslo"})
        result = oslo.difference(oslo)
        assert result.len() == 0


class TestSymmetricDifference:
    def test_symmetric_difference(self, social_graph):
        oslo = social_graph.select("Person").where({"city": "Oslo"})
        young = social_graph.select("Person").where({"age": {"<": 30}})
        result = oslo.symmetric_difference(young)
        # XOR: in one but not both
        intersection_count = oslo.intersection(young).len()
        expected = oslo.len() + young.len() - 2 * intersection_count
        assert result.len() == expected


class TestChaining:
    def test_union_then_intersection(self, social_graph):
        oslo = social_graph.select("Person").where({"city": "Oslo"})
        bergen = social_graph.select("Person").where({"city": "Bergen"})
        old = social_graph.select("Person").where({"age": {">": 35}})
        result = oslo.union(bergen).intersection(old)
        assert result.len() >= 0


class TestCypherUnionColumnNames:
    """Cypher UNION arms must have matching return column names (Neo4j rule).
    A mismatch previously yielded silent NULL rows; now it errors."""

    def test_mismatched_names_error(self):
        import pytest

        import kglite

        kg = kglite.KnowledgeGraph()
        kg.cypher("CREATE (:Q {id: 'q1'})")
        kg.cypher("CREATE (:D {id: 'd1'})")
        with pytest.raises(Exception, match="same return column names"):
            kg.cypher("MATCH (q:Q) RETURN q.id AS a UNION MATCH (d:D) RETURN d.id AS x").to_dicts()

    def test_matching_names_ok(self):
        import kglite

        kg = kglite.KnowledgeGraph()
        kg.cypher("CREATE (:Q {id: 'q1'})")
        kg.cypher("CREATE (:D {id: 'd1'})")
        rows = kg.cypher("MATCH (q:Q) RETURN q.id AS a UNION MATCH (d:D) RETURN d.id AS a").to_dicts()
        assert {r["a"] for r in rows} == {"q1", "d1"}
        # No null rows leaked in.
        assert all(r["a"] is not None for r in rows)
