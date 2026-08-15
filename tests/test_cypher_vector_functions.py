"""`dot` / `cosine` / `norm` over list-valued data, from the Python surface.

The exhaustive semantic goldens live next to the implementation
(`crates/kglite/src/graph/languages/cypher/executor/tests/vectors.rs`); this
file covers what only the wheel can show — a list arriving as a `$param`, a
list column stored through `add_nodes`, NULL reaching Python as `None`, and the
error arms surfacing as kglite exceptions rather than silent nulls.

Red proof: against 0.16.0 every query here failed with
``Unknown function: dot`` / ``cosine`` / ``norm``.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

import kglite


@pytest.fixture
def docs() -> kglite.KnowledgeGraph:
    """Docs carrying a genuinely list-valued `vec` column.

    `text` stores the same vector as bracketed text — the legacy shape
    `size()` / `head()` also read. `zero` is the undefined-cosine control.
    """
    g = kglite.KnowledgeGraph()
    g.add_nodes(
        pd.DataFrame(
            {
                "doc_id": [1, 2, 3, 4, 5],
                "title": ["axis", "diag", "opposite", "zero", "text"],
                "vec": [[1.0, 0.0], [1.0, 1.0], [-1.0, 0.0], [0.0, 0.0], None],
                "textvec": [None, None, None, None, "[3.0, 4.0]"],
            }
        ),
        "Doc",
        "doc_id",
        "title",
        columns=["vec", "textvec"],
    )
    return g


def one(graph: kglite.KnowledgeGraph, query: str, **kwargs):
    rows = graph.cypher(query, **kwargs).to_list()
    assert len(rows) == 1, rows
    return next(iter(rows[0].values()))


class TestArithmetic:
    def test_hand_computed_values(self):
        g = kglite.KnowledgeGraph()
        # 1*4 + 2*5 + 3*6 = 32
        assert one(g, "RETURN dot([1, 2, 3], [4, 5, 6]) AS d") == 32.0
        # sqrt(9 + 16) = 5
        assert one(g, "RETURN norm([3, 4]) AS n") == 5.0
        # orthogonal
        assert one(g, "RETURN cosine([1, 0], [0, 1]) AS c") == 0.0
        # 1/sqrt(2)
        assert math.isclose(one(g, "RETURN cosine([1, 0], [1, 1]) AS c"), 1 / math.sqrt(2), rel_tol=1e-12)

    def test_a_parameter_bound_list_is_a_vector(self):
        # The route every non-Python binding also takes: the query vector
        # arrives as a parameter, not a literal.
        g = kglite.KnowledgeGraph()
        assert one(g, "RETURN dot($a, $b) AS d", params={"a": [1, 2], "b": [3, 4]}) == 11.0
        assert math.isclose(
            one(g, "RETURN cosine($a, $b) AS c", params={"a": [3.0, 4.0], "b": [1.0, 0.0]}),
            0.6,
            rel_tol=1e-12,
        )


class TestStoredListProperties:
    def test_ranking_by_similarity_to_a_query_vector(self, docs):
        rows = docs.cypher(
            "MATCH (d:Doc) WHERE d.vec IS NOT NULL RETURN d.title AS t, cosine(d.vec, $q) AS s ORDER BY s DESC, t",
            params={"q": [1.0, 0.0]},
        ).to_list()
        # axis is identical (1.0), diag is 45° (1/sqrt 2), opposite is -1.
        # `zero`'s cosine is undefined, so it is a NULL — and NULL is the
        # greatest value in the total order, which puts it first under DESC.
        assert [r["t"] for r in rows] == ["zero", "axis", "diag", "opposite"]
        assert rows[0]["s"] is None
        assert rows[1]["s"] == 1.0
        assert math.isclose(rows[2]["s"], 1 / math.sqrt(2), rel_tol=1e-12)
        assert rows[3]["s"] == -1.0

    def test_norm_of_a_stored_list(self, docs):
        assert one(docs, "MATCH (d:Doc {title: 'diag'}) RETURN norm(d.vec) AS n") == pytest.approx(math.sqrt(2))

    def test_two_stored_vectors_compared_against_each_other(self, docs):
        assert (
            one(
                docs,
                "MATCH (a:Doc {title: 'axis'}), (b:Doc {title: 'opposite'}) RETURN dot(a.vec, b.vec) AS d",
            )
            == -1.0
        )

    def test_a_bracketed_string_property_reads_as_a_list(self, docs):
        # sqrt(9 + 16) = 5 — the legacy JSON-text shape answers like a list.
        assert one(docs, "MATCH (d:Doc {title: 'text'}) RETURN norm(d.textvec) AS n") == 5.0

    def test_a_missing_property_is_none_not_an_error(self, docs):
        # A partially-vectorised corpus must still return its rows.
        rows = docs.cypher("MATCH (d:Doc) RETURN d.title AS t, norm(d.vec) AS n ORDER BY t").to_list()
        assert {r["t"] for r in rows if r["n"] is None} == {"text"}


class TestNullAndErrorArms:
    def test_null_propagates_to_none(self):
        g = kglite.KnowledgeGraph()
        for query in [
            "RETURN dot(null, [1, 2]) AS x",
            "RETURN cosine([1, 2], null) AS x",
            "RETURN norm(null) AS x",
        ]:
            assert one(g, query) is None, query

    def test_cosine_of_a_zero_vector_is_none(self):
        g = kglite.KnowledgeGraph()
        assert one(g, "RETURN cosine([0, 0], [1, 2]) AS c") is None
        # dot and norm are defined there and stay numbers.
        assert one(g, "RETURN dot([0, 0], [1, 2]) AS d") == 0.0
        assert one(g, "RETURN norm([0, 0]) AS n") == 0.0

    @pytest.mark.parametrize(
        "query,needle",
        [
            ("RETURN dot([1, 2], [1, 2, 3])", "same length"),
            ("RETURN cosine([1], [1, 2])", "same length"),
            ("RETURN dot([1, 'x'], [1, 2])", "element 1 must be a number"),
            ("RETURN norm([1, null])", "element 1 must be a number"),
            ("RETURN norm('not a list')", "list of numbers"),
            ("RETURN dot(7, [1, 2])", "list of numbers"),
            ("RETURN norm([1], [2])", "1 argument"),
            ("RETURN dot([1])", "2 arguments"),
        ],
    )
    def test_the_error_arms_raise_rather_than_returning_null(self, query, needle):
        # A length mismatch or a non-numeric element is a data bug; a silent
        # null would sit unremarked in a column of plausible scores.
        g = kglite.KnowledgeGraph()
        with pytest.raises(kglite.KgError) as excinfo:
            g.cypher(query)
        assert needle in str(excinfo.value)


class TestComposition:
    def test_the_functions_work_in_where_order_by_and_with(self, docs):
        rows = docs.cypher(
            "MATCH (d:Doc) WHERE d.vec IS NOT NULL "
            "WITH d, dot(d.vec, [1, 0]) AS s WHERE s > 0 "
            "RETURN d.title AS t ORDER BY s DESC"
        ).to_list()
        assert [r["t"] for r in rows] == ["axis", "diag"]

    def test_it_composes_with_collect(self):
        g = kglite.KnowledgeGraph()
        g.add_nodes(
            pd.DataFrame({"id": [1, 2, 3], "title": ["a", "b", "c"], "x": [3.0, 4.0, 0.0]}),
            "P",
            "id",
            "title",
            columns=["x"],
        )
        # collect() yields a list, so norm() reads it: sqrt(9 + 16 + 0) = 5.
        assert one(g, "MATCH (p:P) WITH collect(p.x) AS xs RETURN norm(xs) AS n") == 5.0
