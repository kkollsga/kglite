"""Tests for cypher() — full Cypher query pipeline."""

import pandas as pd
import pytest

import kglite
from kglite import KnowledgeGraph


@pytest.fixture
def cypher_graph():
    """Graph optimized for Cypher tests."""
    graph = KnowledgeGraph()

    people = pd.DataFrame(
        {
            "person_id": [1, 2, 3, 4, 5],
            "name": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
            "age": [30, 25, 35, 28, 40],
            "city": ["Oslo", "Bergen", "Oslo", "Bergen", "Oslo"],
            "salary": [70000, 55000, 80000, 65000, 90000],
            "email": ["alice@test.com", None, "charlie@test.com", None, "eve@test.com"],
        }
    )
    graph.add_nodes(people, "Person", "person_id", "name")

    products = pd.DataFrame(
        {
            "product_id": [101, 102, 103],
            "name": ["Laptop", "Phone", "Tablet"],
            "price": [999.99, 699.99, 349.99],
        }
    )
    graph.add_nodes(products, "Product", "product_id", "name")

    knows = pd.DataFrame(
        {
            "from_id": [1, 1, 2, 3, 4],
            "to_id": [2, 3, 3, 4, 5],
        }
    )
    graph.add_connections(knows, "KNOWS", "Person", "from_id", "Person", "to_id")

    purchased = pd.DataFrame(
        {
            "person_id": [1, 1, 2, 3],
            "product_id": [101, 102, 103, 101],
        }
    )
    graph.add_connections(purchased, "PURCHASED", "Person", "person_id", "Product", "product_id")

    return graph


class TestBasicQueries:
    def test_simple_match_return(self, cypher_graph):
        rows = cypher_graph.cypher("MATCH (n:Person) RETURN n.title")
        assert len(rows) == 5
        assert "n.title" in rows[0]

    def test_match_with_alias(self, cypher_graph):
        rows = cypher_graph.cypher("MATCH (n:Person) RETURN n.title AS name")
        assert "name" in rows[0]
        names = {r["name"] for r in rows}
        assert "Alice" in names

    def test_edge_pattern(self, cypher_graph):
        rows = cypher_graph.cypher("MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN a.title, b.title")
        assert len(rows) == 5

    def test_multi_hop(self, cypher_graph):
        rows = cypher_graph.cypher("MATCH (a:Person)-[:PURCHASED]->(p:Product) RETURN a.title, p.title")
        assert len(rows) == 4

    def test_cross_type(self, cypher_graph):
        rows = cypher_graph.cypher(
            "MATCH (p:Person)-[:PURCHASED]->(pr:Product) RETURN p.title AS person, pr.title AS product"
        )
        assert len(rows) == 4

    def test_case_insensitive_keywords(self, cypher_graph):
        rows = cypher_graph.cypher("match (n:Person) return n.title")
        assert len(rows) == 5


class TestWhereClause:
    def test_comparison_gt(self, cypher_graph):
        rows = cypher_graph.cypher("MATCH (n:Person) WHERE n.age > 30 RETURN n.title, n.age")
        for row in rows:
            assert row["n.age"] > 30

    def test_equality(self, cypher_graph):
        rows = cypher_graph.cypher("MATCH (n:Person) WHERE n.city = 'Oslo' RETURN n.title")
        assert len(rows) == 3

    def test_not_equals(self, cypher_graph):
        rows = cypher_graph.cypher("MATCH (n:Person) WHERE n.city <> 'Oslo' RETURN n.title")
        assert len(rows) == 2

    def test_and(self, cypher_graph):
        rows = cypher_graph.cypher("MATCH (n:Person) WHERE n.age > 25 AND n.city = 'Oslo' RETURN n.title")
        for row in rows:
            assert row["n.title"] in ["Alice", "Charlie", "Eve"]

    def test_or(self, cypher_graph):
        rows = cypher_graph.cypher("MATCH (n:Person) WHERE n.age < 26 OR n.age > 39 RETURN n.title")
        names = {r["n.title"] for r in rows}
        assert "Bob" in names
        assert "Eve" in names

    def test_not(self, cypher_graph):
        rows = cypher_graph.cypher("MATCH (n:Person) WHERE NOT n.city = 'Oslo' RETURN n.title")
        assert len(rows) == 2

    def test_is_null(self, cypher_graph):
        rows = cypher_graph.cypher("MATCH (n:Person) WHERE n.email IS NULL RETURN n.title")
        assert len(rows) == 2

    def test_is_not_null(self, cypher_graph):
        rows = cypher_graph.cypher("MATCH (n:Person) WHERE n.email IS NOT NULL RETURN n.title")
        assert len(rows) == 3

    def test_in_list(self, cypher_graph):
        rows = cypher_graph.cypher("MATCH (n:Person) WHERE n.city IN ['Oslo', 'Bergen'] RETURN n.title")
        assert len(rows) == 5

    def test_contains(self, cypher_graph):
        rows = cypher_graph.cypher("MATCH (n:Person) WHERE n.title CONTAINS 'li' RETURN n.title")
        names = {r["n.title"] for r in rows}
        assert "Alice" in names
        assert "Charlie" in names

    def test_starts_with(self, cypher_graph):
        rows = cypher_graph.cypher("MATCH (n:Person) WHERE n.title STARTS WITH 'A' RETURN n.title")
        assert len(rows) == 1

    def test_ends_with(self, cypher_graph):
        rows = cypher_graph.cypher("MATCH (n:Person) WHERE n.title ENDS WITH 'e' RETURN n.title")
        names = {r["n.title"] for r in rows}
        assert "Alice" in names
        assert "Eve" in names
        assert "Charlie" in names


class TestOrderByLimitSkip:
    def test_order_by_asc(self, cypher_graph):
        rows = cypher_graph.cypher("MATCH (n:Person) RETURN n.title, n.age ORDER BY n.age")
        ages = [r["n.age"] for r in rows]
        assert ages == sorted(ages)

    def test_order_by_desc(self, cypher_graph):
        rows = cypher_graph.cypher("MATCH (n:Person) RETURN n.title, n.age ORDER BY n.age DESC")
        ages = [r["n.age"] for r in rows]
        assert ages == sorted(ages, reverse=True)

    def test_limit(self, cypher_graph):
        rows = cypher_graph.cypher("MATCH (n:Person) RETURN n.title LIMIT 3")
        assert len(rows) == 3

    def test_skip(self, cypher_graph):
        rows = cypher_graph.cypher("MATCH (n:Person) RETURN n.title ORDER BY n.age SKIP 2")
        assert len(rows) == 3

    def test_skip_and_limit(self, cypher_graph):
        rows = cypher_graph.cypher("MATCH (n:Person) RETURN n.title, n.age ORDER BY n.age SKIP 1 LIMIT 2")
        assert len(rows) == 2
        ages = [r["n.age"] for r in rows]
        assert ages == [28, 30]


class TestAggregation:
    def test_count_star(self, cypher_graph):
        rows = cypher_graph.cypher("MATCH (n:Person) RETURN count(*) AS total")
        assert rows[0]["total"] == 5

    def test_count_with_grouping(self, cypher_graph):
        rows = cypher_graph.cypher("MATCH (n:Person) RETURN n.city AS city, count(*) AS cnt")
        cities = {r["city"]: r["cnt"] for r in rows}
        assert cities["Oslo"] == 3
        assert cities["Bergen"] == 2

    def test_sum(self, cypher_graph):
        rows = cypher_graph.cypher("MATCH (n:Person) RETURN sum(n.salary) AS total")
        assert rows[0]["total"] == 360000

    def test_avg(self, cypher_graph):
        rows = cypher_graph.cypher("MATCH (n:Person) RETURN avg(n.age) AS avg_age")
        assert abs(rows[0]["avg_age"] - 31.6) < 0.1

    def test_min_max(self, cypher_graph):
        rows = cypher_graph.cypher("MATCH (n:Person) RETURN min(n.age) AS youngest, max(n.age) AS oldest")
        assert rows[0]["youngest"] == 25
        assert rows[0]["oldest"] == 40

    def test_distinct(self, cypher_graph):
        rows = cypher_graph.cypher("MATCH (n:Person) RETURN DISTINCT n.city")
        assert len(rows) == 2

    def test_count_distinct(self, cypher_graph):
        rows = cypher_graph.cypher("MATCH (n:Person) RETURN count(DISTINCT n.city) AS unique_cities")
        assert rows[0]["unique_cities"] == 2


class TestWithClause:
    def test_with_basic(self, cypher_graph):
        rows = cypher_graph.cypher("""
            MATCH (p:Person)-[:KNOWS]->(f:Person)
            WITH p, count(f) AS friend_count
            RETURN p.title, friend_count
            ORDER BY friend_count DESC
        """)
        assert len(rows) > 0
        counts = [r["friend_count"] for r in rows]
        assert counts == sorted(counts, reverse=True)


class TestOptionalMatch:
    def test_optional_match_basic(self, cypher_graph):
        rows = cypher_graph.cypher("""
            MATCH (p:Person)
            OPTIONAL MATCH (p)-[:PURCHASED]->(pr:Product)
            RETURN p.title, count(pr) AS purchases
        """)
        assert len(rows) == 5  # All persons, even without purchases


class TestExpressions:
    def test_arithmetic(self, cypher_graph):
        rows = cypher_graph.cypher("MATCH (n:Product) RETURN n.title, n.price * 1.25 AS with_tax")
        for row in rows:
            assert row["with_tax"] > row.get("n.price", 0) or "with_tax" in row

    def test_coalesce(self, cypher_graph):
        rows = cypher_graph.cypher("MATCH (n:Person) RETURN n.title, coalesce(n.email, 'no email') AS contact")
        for row in rows:
            assert row["contact"] != "" or row["contact"] is not None

    def test_predicate_pushdown(self, cypher_graph):
        """Predicate pushdown should produce same results as without."""
        rows = cypher_graph.cypher("MATCH (n:Person) WHERE n.age = 30 RETURN n.title")
        assert len(rows) == 1
        assert rows[0]["n.title"] == "Alice"


class TestEmptyResults:
    def test_no_matching_nodes(self, cypher_graph):
        rows = cypher_graph.cypher("MATCH (n:NonExistent) RETURN n.title")
        assert len(rows) == 0

    def test_where_eliminates_all(self, cypher_graph):
        rows = cypher_graph.cypher("MATCH (n:Person) WHERE n.age > 100 RETURN n.title")
        assert len(rows) == 0


class TestSyntaxErrors:
    def test_invalid_query(self, cypher_graph):
        # Phase A.2 / C2 — parse_cypher raises typed CypherSyntaxError
        # (was ValueError pre-A.2). Catchable via either the specific
        # class or the universal kglite.KgError base.
        import kglite

        with pytest.raises(kglite.CypherSyntaxError):
            cypher_graph.cypher("NOT A VALID QUERY")


class TestCaseExpressions:
    """Tests for CASE WHEN ... THEN ... ELSE ... END expressions."""

    def test_generic_case(self, cypher_graph):
        """CASE WHEN predicate THEN result ELSE default END."""
        rows = cypher_graph.cypher("""
            MATCH (n:Person)
            RETURN n.name AS name,
                   CASE WHEN n.age >= 30 THEN 'senior' ELSE 'junior' END AS level
            ORDER BY n.name
        """)
        assert len(rows) == 5
        alice = next(r for r in rows if r["name"] == "Alice")
        assert alice["level"] == "senior"  # age 30
        bob = next(r for r in rows if r["name"] == "Bob")
        assert bob["level"] == "junior"  # age 25

    def test_simple_case(self, cypher_graph):
        """CASE expr WHEN val THEN result ... END."""
        rows = cypher_graph.cypher("""
            MATCH (n:Person)
            RETURN n.name AS name,
                   CASE n.city WHEN 'Oslo' THEN 'capital' WHEN 'Bergen' THEN 'west' ELSE 'other' END AS region
            ORDER BY n.name
        """)
        alice = next(r for r in rows if r["name"] == "Alice")
        assert alice["region"] == "capital"
        bob = next(r for r in rows if r["name"] == "Bob")
        assert bob["region"] == "west"

    def test_case_no_else_returns_null(self, cypher_graph):
        """CASE without ELSE returns null when no match."""
        rows = cypher_graph.cypher("""
            MATCH (n:Person)
            RETURN n.name AS name,
                   CASE n.city WHEN 'Trondheim' THEN 'found' END AS status
            ORDER BY n.name
        """)
        # No one lives in Trondheim, so all should be null
        for row in rows:
            assert row["status"] is None

    def test_case_multiple_when(self, cypher_graph):
        """CASE with multiple WHEN clauses — first match wins."""
        rows = cypher_graph.cypher("""
            MATCH (n:Person)
            RETURN n.name AS name,
                   CASE
                       WHEN n.age >= 40 THEN 'veteran'
                       WHEN n.age >= 30 THEN 'experienced'
                       ELSE 'newcomer'
                   END AS tier
            ORDER BY n.name
        """)
        eve = next(r for r in rows if r["name"] == "Eve")
        assert eve["tier"] == "veteran"  # age 40 — first match wins
        alice = next(r for r in rows if r["name"] == "Alice")
        assert alice["tier"] == "experienced"  # age 30
        bob = next(r for r in rows if r["name"] == "Bob")
        assert bob["tier"] == "newcomer"  # age 25


class TestCaseEdgeCases:
    """Edge cases for CASE expressions: nulls, WHERE usage, nesting."""

    def test_case_in_where(self, cypher_graph):
        """CASE expression used inside WHERE clause."""
        rows = cypher_graph.cypher("""
            MATCH (n:Person)
            WHERE CASE WHEN n.age >= 30 THEN true ELSE false END
            RETURN n.name AS name
            ORDER BY n.name
        """)
        names = [r["name"] for r in rows]
        assert "Alice" in names  # age 30
        assert "Eve" in names  # age 40
        assert "Bob" not in names  # age 25

    def test_case_with_null_property(self, cypher_graph):
        """CASE on a property that might be null."""
        cypher_graph.cypher("CREATE (:Person {name: 'NullAge'})")
        rows = cypher_graph.cypher("""
            MATCH (n:Person {name: 'NullAge'})
            RETURN CASE WHEN n.age IS NULL THEN 'unknown' ELSE 'known' END AS status
        """)
        assert len(rows) == 1
        assert rows[0]["status"] == "unknown"

    def test_case_no_else_no_match_returns_null(self, cypher_graph):
        """CASE without ELSE returns null when no WHEN matches."""
        rows = cypher_graph.cypher("""
            MATCH (n:Person)
            RETURN n.name AS name,
                   CASE WHEN n.age > 100 THEN 'centenarian' END AS label
            ORDER BY n.name
        """)
        for row in rows:
            assert row["label"] is None

    # CASE result/operand positions parse at the full expression tower —
    # comparisons, boolean operators, EXISTS subqueries, and pattern
    # expressions all work in THEN / ELSE / WHEN / operand positions.
    # (Regression: results parsed below the comparison level, so
    # `THEN 1 < 2` and `THEN (n)-[:R]->()` were syntax errors.)

    def test_case_comparison_in_then(self, cypher_graph):
        rows = cypher_graph.cypher("MATCH (n:Person {id: 1}) RETURN CASE WHEN true THEN 1 < 2 ELSE false END AS v")
        assert [r["v"] for r in rows] == [True]

    def test_case_comparison_in_else(self, cypher_graph):
        rows = cypher_graph.cypher("MATCH (n:Person {id: 1}) RETURN CASE WHEN false THEN false ELSE 2 > 3 END AS v")
        assert [r["v"] for r in rows] == [False]

    def test_case_exists_subquery_in_then(self, cypher_graph):
        rows = cypher_graph.cypher(
            "MATCH (n:Person {id: 1}) RETURN CASE WHEN true THEN EXISTS { (n)-[:KNOWS]->() } ELSE false END AS v"
        )
        assert [r["v"] for r in rows] == [True]

    def test_case_pattern_expression_in_then(self, cypher_graph):
        rows = cypher_graph.cypher(
            "MATCH (n:Person {id: 1}) RETURN CASE WHEN true THEN (n)-[:KNOWS]->() ELSE false END AS v"
        )
        assert [r["v"] for r in rows] == [True]

    def test_case_pattern_expression_in_else(self, cypher_graph):
        # Eve (id 5) has no outgoing KNOWS edge.
        rows = cypher_graph.cypher(
            "MATCH (n:Person {id: 5}) RETURN CASE WHEN false THEN true ELSE (n)-[:KNOWS]->() END AS v"
        )
        assert [r["v"] for r in rows] == [False]

    def test_case_exists_subquery_in_when(self, cypher_graph):
        rows = cypher_graph.cypher(
            "MATCH (n:Person {id: 1}) RETURN CASE WHEN EXISTS { (n)-[:KNOWS]->() } THEN 'y' ELSE 'n' END AS v"
        )
        assert [r["v"] for r in rows] == ["y"]

    def test_case_boolean_operand_position(self, cypher_graph):
        rows = cypher_graph.cypher(
            "MATCH (n:Person {id: 1}) RETURN CASE n.age >= 30 WHEN true THEN 'senior' ELSE 'junior' END AS v"
        )
        assert [r["v"] for r in rows] == ["senior"]


class TestAbbreviatedEdgePatterns:
    """openCypher abbreviated relationship patterns: -->, --, <--.

    (Regression: the pattern parser rejected them with "expected '['".)
    Fixture shape: KNOWS 1→2, 1→3, 2→3, 3→4, 4→5; PURCHASED 1→101,
    1→102, 2→103, 3→101.
    """

    def test_abbreviated_outgoing(self, cypher_graph):
        rows = cypher_graph.cypher("MATCH (a:Person {id: 1})-->(x) RETURN count(x) AS c")
        assert rows[0]["c"] == 4  # KNOWS ×2 + PURCHASED ×2

    def test_abbreviated_undirected(self, cypher_graph):
        rows = cypher_graph.cypher("MATCH (b:Person {id: 2})--(y) RETURN count(y) AS c")
        assert rows[0]["c"] == 3  # in: KNOWS 1→2; out: KNOWS 2→3, PURCHASED 2→103

    def test_abbreviated_incoming(self, cypher_graph):
        rows = cypher_graph.cypher("MATCH (b:Person {id: 2})<--(a) RETURN count(a) AS c")
        assert rows[0]["c"] == 1  # KNOWS 1→2

    def test_abbreviated_matches_bracketed_equivalent(self, cypher_graph):
        abbrev = cypher_graph.cypher("MATCH (a:Person)-->(x) RETURN count(*) AS c")[0]["c"]
        bracketed = cypher_graph.cypher("MATCH (a:Person)-[]->(x) RETURN count(*) AS c")[0]["c"]
        assert abbrev == bracketed == 9  # 5 KNOWS + 4 PURCHASED

    def test_abbreviated_multi_hop(self, cypher_graph):
        rows = cypher_graph.cypher("MATCH (a:Person {id: 1})-->(b:Person)-->(c:Person) RETURN count(*) AS c")
        # 1→2→3 and 1→3→4
        assert rows[0]["c"] == 2

    def test_abbreviated_in_optional_match(self, cypher_graph):
        rows = cypher_graph.cypher("MATCH (e:Person {id: 5}) OPTIONAL MATCH (e)-->(m) RETURN count(m) AS c")
        assert rows[0]["c"] == 0  # Eve has no outgoing edges; row survives

    def test_abbreviated_in_exists(self, cypher_graph):
        rows = cypher_graph.cypher("MATCH (p:Person) WHERE EXISTS { (p)-->() } RETURN count(p) AS c")
        assert rows[0]["c"] == 4  # everyone but Eve

    def test_abbreviated_pattern_expression_in_where(self, cypher_graph):
        rows = cypher_graph.cypher("MATCH (p:Person) WHERE (p)-->() RETURN count(p) AS c")
        assert rows[0]["c"] == 4

    def test_abbreviated_pattern_expression_in_return(self, cypher_graph):
        rows = cypher_graph.cypher("MATCH (p:Person {id: 5}) RETURN (p)-->() AS has_out, (p)<--() AS has_in")
        assert rows[0]["has_out"] is False
        assert rows[0]["has_in"] is True

    def test_double_arrow_rejected(self, cypher_graph):
        with pytest.raises(kglite.CypherError):
            cypher_graph.cypher("MATCH (a)<-->(b) RETURN count(*) AS c")

    def test_single_dash_still_rejected_in_match(self, cypher_graph):
        with pytest.raises(kglite.CypherError):
            cypher_graph.cypher("MATCH (a)-(b) RETURN count(*) AS c")


class TestAggregateArgumentErrors:
    """Aggregate ARGUMENT evaluation errors must propagate on every
    execution path (fused node-scan, streaming, materialized) instead of
    being swallowed into Null. Legitimate null semantics (property on a
    node that lacks it) stay non-erroring.
    """

    PATH_KWARGS = (
        {},  # optimizer + streaming (fused node-scan aggregate)
        {"disable_optimizer": True},  # streaming aggregate
        {"streaming": False},  # optimizer, no streaming
        {"streaming": False, "disable_optimizer": True},  # materialized
    )

    @pytest.mark.parametrize("kwargs", PATH_KWARGS)
    def test_missing_parameter_in_aggregate_argument_errors(self, cypher_graph, kwargs):
        with pytest.raises(kglite.CypherError, match="Missing parameter"):
            cypher_graph.cypher("MATCH (n:Person) RETURN sum($missing) AS s", **kwargs)

    @pytest.mark.parametrize("kwargs", PATH_KWARGS)
    def test_missing_parameter_in_compound_aggregate_argument_errors(self, cypher_graph, kwargs):
        with pytest.raises(kglite.CypherError, match="Missing parameter"):
            cypher_graph.cypher(
                "MATCH (n:Person) RETURN n.city AS city, sum(n.age + $missing) AS s",
                **kwargs,
            )

    @pytest.mark.parametrize("kwargs", PATH_KWARGS)
    def test_missing_parameter_after_with_errors(self, cypher_graph, kwargs):
        with pytest.raises(kglite.CypherError, match="Missing parameter"):
            cypher_graph.cypher("MATCH (n:Person) WITH n RETURN sum($missing) AS s", **kwargs)

    @pytest.mark.parametrize("kwargs", PATH_KWARGS)
    def test_absent_property_stays_null_not_error(self, cypher_graph, kwargs):
        rows = cypher_graph.cypher("MATCH (n:Person) RETURN sum(n.absent) AS s, count(n.absent) AS c", **kwargs)
        assert rows[0]["s"] == 0
        assert rows[0]["c"] == 0


class TestParameters:
    """Tests for $param parameter substitution."""

    def test_single_parameter(self, cypher_graph):
        rows = cypher_graph.cypher(
            "MATCH (n:Person) WHERE n.age > $min_age RETURN n.name AS name ORDER BY n.name", params={"min_age": 30}
        )
        names = [r["name"] for r in rows]
        assert "Charlie" in names  # age 35
        assert "Eve" in names  # age 40
        assert "Alice" not in names  # age 30, not > 30
        assert "Bob" not in names  # age 25

    def test_multiple_parameters(self, cypher_graph):
        rows = cypher_graph.cypher(
            "MATCH (n:Person) WHERE n.city = $city AND n.age > $age RETURN n.name AS name",
            params={"city": "Oslo", "age": 30},
        )
        names = [r["name"] for r in rows]
        assert "Charlie" in names  # Oslo, age 35
        assert "Eve" in names  # Oslo, age 40
        assert "Alice" not in names  # Oslo, age 30 (not > 30)

    def test_missing_parameter_error(self, cypher_graph):
        with pytest.raises(kglite.KgError, match="Missing parameter"):
            cypher_graph.cypher("MATCH (n:Person) WHERE n.age > $nonexistent RETURN n.name")

    def test_parameter_with_to_df(self, cypher_graph):
        df = cypher_graph.cypher(
            "MATCH (n:Person) WHERE n.age >= $min_age RETURN n.name AS name, n.age AS age ORDER BY n.age",
            params={"min_age": 35},
            to_df=True,
        )
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2  # Charlie (35) and Eve (40)
        assert list(df["name"]) == ["Charlie", "Eve"]

    def test_string_parameter(self, cypher_graph):
        rows = cypher_graph.cypher(
            "MATCH (n:Person) WHERE n.city = $city RETURN n.name AS name ORDER BY n.name", params={"city": "Bergen"}
        )
        names = [r["name"] for r in rows]
        assert names == ["Bob", "Diana"]

    def test_parameter_in_return(self, cypher_graph):
        rows = cypher_graph.cypher(
            "MATCH (n:Person) RETURN n.name AS name, $label AS category ORDER BY n.name LIMIT 1",
            params={"label": "person"},
        )
        assert rows[0]["category"] == "person"


class TestExistingFeatures:
    """Tests for already-implemented features to ensure coverage."""

    def test_unwind(self, cypher_graph):
        rows = cypher_graph.cypher("UNWIND [1, 2, 3] AS x RETURN x")
        assert len(rows) == 3

    def test_union(self, cypher_graph):
        rows = cypher_graph.cypher("""
            MATCH (n:Person) WHERE n.city = 'Oslo' RETURN n.name AS name
            UNION
            MATCH (n:Person) WHERE n.age > 35 RETURN n.name AS name
        """)
        names = {r["name"] for r in rows}
        # Oslo: Alice, Charlie, Eve; age > 35: Eve; UNION deduplicates
        assert names == {"Alice", "Charlie", "Eve"}

    def test_union_all(self, cypher_graph):
        rows = cypher_graph.cypher("""
            MATCH (n:Person) WHERE n.city = 'Oslo' RETURN n.name AS name
            UNION ALL
            MATCH (n:Person) WHERE n.age > 35 RETURN n.name AS name
        """)
        names = [r["name"] for r in rows]
        # Oslo: Alice, Charlie, Eve; age > 35: Eve; UNION ALL keeps duplicates
        assert len(names) == 4  # 3 + 1 (Eve appears twice)

    def test_intersect(self, cypher_graph):
        rows = cypher_graph.cypher("""
            MATCH (n:Person) WHERE n.city = 'Oslo' RETURN n.name AS name
            INTERSECT
            MATCH (n:Person) WHERE n.age > 35 RETURN n.name AS name
        """)
        names = {r["name"] for r in rows}
        # Oslo: Alice, Charlie, Eve; age > 35: Eve; intersection = {Eve}
        assert names == {"Eve"}

    def test_intersect_disjoint(self, cypher_graph):
        rows = cypher_graph.cypher("""
            MATCH (n:Person) WHERE n.name = 'Alice' RETURN n.name AS name
            INTERSECT
            MATCH (n:Person) WHERE n.name = 'Bob' RETURN n.name AS name
        """)
        assert list(rows) == []

    def test_except(self, cypher_graph):
        rows = cypher_graph.cypher("""
            MATCH (n:Person) WHERE n.city = 'Oslo' RETURN n.name AS name
            EXCEPT
            MATCH (n:Person) WHERE n.age > 35 RETURN n.name AS name
        """)
        names = {r["name"] for r in rows}
        # Oslo: Alice, Charlie, Eve; minus age > 35 (Eve) = {Alice, Charlie}
        assert names == {"Alice", "Charlie"}

    def test_except_when_right_empty(self, cypher_graph):
        rows = cypher_graph.cypher("""
            MATCH (n:Person) WHERE n.city = 'Oslo' RETURN n.name AS name
            EXCEPT
            MATCH (n:Person) WHERE n.name = 'Nonexistent' RETURN n.name AS name
        """)
        names = {r["name"] for r in rows}
        assert names == {"Alice", "Charlie", "Eve"}

    def test_var_length_path(self, cypher_graph):
        rows = cypher_graph.cypher("""
            MATCH (a:Person)-[:KNOWS*1..2]->(b:Person)
            WHERE a.name = 'Alice'
            RETURN DISTINCT b.name AS friend
        """)
        names = {r["friend"] for r in rows}
        # Alice->Bob, Alice->Charlie, Bob->Charlie, Charlie->Diana
        assert "Bob" in names
        assert "Charlie" in names

    def test_coalesce_function(self, cypher_graph):
        rows = cypher_graph.cypher("""
            MATCH (n:Person)
            RETURN n.name AS name, coalesce(n.email, 'no email') AS contact
            ORDER BY n.name
        """)
        bob = next(r for r in rows if r["name"] == "Bob")
        assert bob["contact"] == "no email"
        alice = next(r for r in rows if r["name"] == "Alice")
        assert alice["contact"] == "alice@test.com"

    def test_collect_aggregate(self, cypher_graph):
        rows = cypher_graph.cypher("""
            MATCH (n:Person)
            RETURN n.city AS city, collect(n.name) AS names
            ORDER BY city
        """)
        assert len(rows) == 2  # Bergen and Oslo


class TestCreateClause:
    """Tests for CREATE clause — node and edge creation via Cypher."""

    def test_create_node(self, cypher_graph):
        """CREATE (n:City {name: 'Trondheim'}) creates a new node."""
        before_cnt = cypher_graph.cypher("MATCH (n:City) RETURN count(*) AS cnt")[0]["cnt"]

        cypher_graph.cypher("CREATE (n:City {name: 'Trondheim'})")
        stats = cypher_graph.last_mutation_stats
        assert stats is not None
        assert stats["nodes_created"] == 1

        after_cnt = cypher_graph.cypher("MATCH (n:City) RETURN count(*) AS cnt")[0]["cnt"]
        assert after_cnt == before_cnt + 1

    def test_create_node_with_properties(self, cypher_graph):
        """CREATE with multiple properties stores them on the node."""
        cypher_graph.cypher("CREATE (n:Person {name: 'Frank', age: 45, city: 'Trondheim'})")
        rows = cypher_graph.cypher("MATCH (n:Person) WHERE n.name = 'Frank' RETURN n.name, n.age, n.city")
        assert len(rows) == 1
        row = rows[0]
        assert row["n.name"] == "Frank"
        assert row["n.age"] == 45
        assert row["n.city"] == "Trondheim"

    def test_create_edge_after_match(self, cypher_graph):
        """MATCH (a) MATCH (b) CREATE (a)-[:REL]->(b) creates an edge."""
        cypher_graph.cypher("""
            MATCH (a:Person) WHERE a.name = 'Alice'
            MATCH (b:Person) WHERE b.name = 'Eve'
            CREATE (a)-[:FRIENDS]->(b)
        """)
        assert cypher_graph.last_mutation_stats["relationships_created"] == 1

        # Verify the edge exists
        check = cypher_graph.cypher("""
            MATCH (a:Person)-[:FRIENDS]->(b:Person)
            RETURN a.name AS src, b.name AS tgt
        """)
        assert len(check) == 1
        assert check[0]["src"] == "Alice"
        assert check[0]["tgt"] == "Eve"

    def test_create_path(self, cypher_graph):
        """CREATE (a:X)-[:R]->(b:Y) creates both nodes and the edge."""
        cypher_graph.cypher("CREATE (a:Team {name: 'Alpha'})-[:MEMBER]->(b:Team {name: 'Beta'})")
        stats = cypher_graph.last_mutation_stats
        assert stats["nodes_created"] == 2
        assert stats["relationships_created"] == 1

    def test_create_with_params(self, cypher_graph):
        """CREATE with $param substitution for property values."""
        cypher_graph.cypher("CREATE (n:Person {name: $name, age: $age})", params={"name": "Grace", "age": 29})
        assert cypher_graph.last_mutation_stats["nodes_created"] == 1

        rows = cypher_graph.cypher("MATCH (n:Person) WHERE n.name = 'Grace' RETURN n.age")
        assert len(rows) == 1
        assert rows[0]["n.age"] == 29

    def test_create_return_created_node(self, cypher_graph):
        """CREATE ... RETURN should return data about created nodes."""
        result = cypher_graph.cypher("CREATE (n:Animal {name: 'Rex', species: 'Dog'}) RETURN n.name, n.species")
        assert result.stats is not None
        assert len(result) == 1
        assert result[0]["n.name"] == "Rex"
        assert result[0]["n.species"] == "Dog"


class TestSetClause:
    """Tests for SET clause — property updates via Cypher."""

    def test_set_property(self, cypher_graph):
        """SET n.prop = value updates a property."""
        cypher_graph.cypher("""
            MATCH (n:Person) WHERE n.name = 'Alice'
            SET n.city = 'Trondheim'
        """)
        rows = cypher_graph.cypher("MATCH (n:Person) WHERE n.name = 'Alice' RETURN n.city")
        assert rows[0]["n.city"] == "Trondheim"

    def test_set_multiple_properties(self, cypher_graph):
        """SET n.a = x, n.b = y updates multiple properties."""
        cypher_graph.cypher("""
            MATCH (n:Person) WHERE n.name = 'Bob'
            SET n.city = 'Stavanger', n.age = 26
        """)
        assert cypher_graph.last_mutation_stats["properties_set"] == 2

        rows = cypher_graph.cypher("MATCH (n:Person) WHERE n.name = 'Bob' RETURN n.city, n.age")
        row = rows[0]
        assert row["n.city"] == "Stavanger"
        assert row["n.age"] == 26

    def test_set_title(self, cypher_graph):
        """SET n.name = 'X' updates the node title."""
        cypher_graph.cypher("""
            MATCH (n:Person) WHERE n.name = 'Charlie'
            SET n.name = 'Charles'
        """)
        rows = cypher_graph.cypher("MATCH (n:Person) WHERE n.name = 'Charles' RETURN n.name, n.title")
        assert len(rows) == 1
        assert rows[0]["n.name"] == "Charles"
        assert rows[0]["n.title"] == "Charles"

    def test_set_id_error(self, cypher_graph):
        """SET n.id = x should raise an error (id is immutable)."""
        with pytest.raises(kglite.KgError):
            cypher_graph.cypher("""
                MATCH (n:Person) WHERE n.name = 'Alice'
                SET n.id = 999
            """)

    def test_set_with_expression(self, cypher_graph):
        """SET n.prop = expression (arithmetic)."""
        cypher_graph.cypher("""
            MATCH (n:Person) WHERE n.name = 'Alice'
            SET n.age = 30 + 1
        """)
        rows = cypher_graph.cypher("MATCH (n:Person) WHERE n.name = 'Alice' RETURN n.age")
        assert rows[0]["n.age"] == 31


class TestMutationStats:
    """Tests that CREATE and SET return proper mutation statistics."""

    def test_create_returns_stats(self, cypher_graph):
        """CREATE stores stats in last_mutation_stats."""
        cypher_graph.cypher("CREATE (a:X {name: 'A'})-[:R]->(b:X {name: 'B'})")
        stats = cypher_graph.last_mutation_stats
        assert stats is not None
        assert stats["nodes_created"] == 2
        assert stats["relationships_created"] == 1
        assert stats["properties_set"] == 0  # properties on CREATE don't count as SET

    def test_set_returns_stats(self, cypher_graph):
        """SET stores stats in last_mutation_stats with properties_set count."""
        cypher_graph.cypher("""
            MATCH (n:Person) WHERE n.name = 'Alice'
            SET n.city = 'Drammen', n.salary = 75000
        """)
        stats = cypher_graph.last_mutation_stats
        assert stats is not None
        assert stats["properties_set"] == 2
        assert stats["nodes_created"] == 0

    def test_read_query_no_stats(self):
        """Read-only queries should not update last_mutation_stats."""
        fresh = KnowledgeGraph()
        people = pd.DataFrame({"id": [1], "name": ["Alice"]})
        fresh.add_nodes(people, "Person", "id", "name")
        fresh.cypher("MATCH (n:Person) RETURN n.name")
        assert fresh.last_mutation_stats is None

    def test_delete_returns_stats(self, cypher_graph):
        """DELETE stores deletion stats in last_mutation_stats."""
        cypher_graph.cypher("""
            MATCH (n:Person) WHERE n.name = 'Eve'
            DETACH DELETE n
        """)
        stats = cypher_graph.last_mutation_stats
        assert stats is not None
        assert stats["nodes_deleted"] == 1

    def test_remove_returns_stats(self, cypher_graph):
        """REMOVE stores properties_removed in last_mutation_stats."""
        cypher_graph.cypher("""
            MATCH (n:Person) WHERE n.name = 'Alice'
            REMOVE n.age
        """)
        stats = cypher_graph.last_mutation_stats
        assert stats is not None
        assert stats["properties_removed"] == 1


class TestDeleteClause:
    """Tests for DELETE clause — node and edge deletion via Cypher."""

    def test_detach_delete_node(self, cypher_graph):
        """DETACH DELETE removes a node and its edges."""
        before_cnt = cypher_graph.cypher("MATCH (n:Person) RETURN count(*) AS cnt")[0]["cnt"]

        cypher_graph.cypher("""
            MATCH (n:Person) WHERE n.name = 'Eve'
            DETACH DELETE n
        """)
        assert cypher_graph.last_mutation_stats["nodes_deleted"] == 1

        after_cnt = cypher_graph.cypher("MATCH (n:Person) RETURN count(*) AS cnt")[0]["cnt"]
        assert after_cnt == before_cnt - 1

    def test_detach_delete_node_with_edges(self, cypher_graph):
        """DETACH DELETE removes connected edges too."""
        cypher_graph.cypher("""
            MATCH (n:Person) WHERE n.name = 'Alice'
            DETACH DELETE n
        """)
        stats = cypher_graph.last_mutation_stats
        assert stats["nodes_deleted"] == 1
        assert stats["relationships_deleted"] > 0

    def test_delete_node_error_has_edges(self, cypher_graph):
        """Plain DELETE on a node with edges should error."""
        with pytest.raises(kglite.KgError, match="DETACH DELETE"):
            cypher_graph.cypher("""
                MATCH (n:Person) WHERE n.name = 'Alice'
                DELETE n
            """)

    def test_delete_relationship(self, cypher_graph):
        """DELETE r removes a relationship but keeps nodes."""
        before_persons = cypher_graph.cypher("MATCH (n:Person) RETURN count(*) AS cnt")[0]["cnt"]

        cypher_graph.cypher("""
            MATCH (a:Person)-[r:KNOWS]->(b:Person)
            DELETE r
        """)
        assert cypher_graph.last_mutation_stats["relationships_deleted"] > 0

        # Nodes should still be there
        after_persons = cypher_graph.cypher("MATCH (n:Person) RETURN count(*) AS cnt")[0]["cnt"]
        assert after_persons == before_persons


class TestRemoveClause:
    """Tests for REMOVE clause — property removal via Cypher."""

    def test_remove_property(self, cypher_graph):
        """REMOVE n.prop deletes the property from the node."""
        cypher_graph.cypher("""
            MATCH (n:Person) WHERE n.name = 'Alice'
            REMOVE n.age
        """)
        rows = cypher_graph.cypher("MATCH (n:Person) WHERE n.name = 'Alice' RETURN n.age AS age")
        assert rows[0]["age"] is None

    def test_remove_multiple_properties(self, cypher_graph):
        """REMOVE n.a, n.b removes multiple properties."""
        cypher_graph.cypher("""
            MATCH (n:Person) WHERE n.name = 'Alice'
            REMOVE n.age, n.city
        """)
        assert cypher_graph.last_mutation_stats["properties_removed"] == 2

    def test_remove_nonexistent_is_noop(self, cypher_graph):
        """REMOVE on a non-existent property is a no-op."""
        cypher_graph.cypher("""
            MATCH (n:Person) WHERE n.name = 'Alice'
            REMOVE n.nonexistent
        """)
        assert cypher_graph.last_mutation_stats["properties_removed"] == 0


class TestMergeClause:
    """Tests for MERGE clause — match-or-create via Cypher."""

    def test_merge_creates_node(self, cypher_graph):
        """MERGE creates a node when no match is found."""
        before_cnt = cypher_graph.cypher("MATCH (n:Person) RETURN count(*) AS cnt")[0]["cnt"]
        cypher_graph.cypher("MERGE (n:Person {name: 'Frank'})")
        assert cypher_graph.last_mutation_stats["nodes_created"] == 1

        after_cnt = cypher_graph.cypher("MATCH (n:Person) RETURN count(*) AS cnt")[0]["cnt"]
        assert after_cnt == before_cnt + 1

    def test_merge_matches_existing(self, cypher_graph):
        """MERGE does not create when a match is found."""
        before_cnt = cypher_graph.cypher("MATCH (n:Person) RETURN count(*) AS cnt")[0]["cnt"]
        cypher_graph.cypher("MERGE (n:Person {name: 'Alice'})")
        assert cypher_graph.last_mutation_stats["nodes_created"] == 0

        after_cnt = cypher_graph.cypher("MATCH (n:Person) RETURN count(*) AS cnt")[0]["cnt"]
        assert after_cnt == before_cnt

    def test_merge_on_create_set(self, cypher_graph):
        """MERGE ON CREATE SET runs when creating."""
        cypher_graph.cypher("MERGE (n:Person {name: 'Frank'}) ON CREATE SET n.age = 40")
        stats = cypher_graph.last_mutation_stats
        assert stats["nodes_created"] == 1
        assert stats["properties_set"] == 1

        rows = cypher_graph.cypher("MATCH (n:Person) WHERE n.name = 'Frank' RETURN n.age AS age")
        assert rows[0]["age"] == 40

    def test_merge_on_match_set(self, cypher_graph):
        """MERGE ON MATCH SET runs when matching existing."""
        cypher_graph.cypher("MERGE (n:Person {name: 'Alice'}) ON MATCH SET n.visits = 1")
        stats = cypher_graph.last_mutation_stats
        assert stats["nodes_created"] == 0
        assert stats["properties_set"] == 1

        rows = cypher_graph.cypher("MATCH (n:Person) WHERE n.name = 'Alice' RETURN n.visits AS visits")
        assert rows[0]["visits"] == 1

    def test_merge_relationship_exists(self, cypher_graph):
        """MERGE does not create duplicate edge when one already exists."""
        cypher_graph.cypher("""
            MATCH (a:Person {name: 'Alice'}), (b:Person {name: 'Bob'})
            MERGE (a)-[r:KNOWS]->(b)
        """)
        assert cypher_graph.last_mutation_stats["relationships_created"] == 0

    def test_merge_creates_relationship(self, cypher_graph):
        """MERGE creates edge when no matching edge exists."""
        cypher_graph.cypher("""
            MATCH (a:Person {name: 'Alice'}), (b:Person {name: 'Bob'})
            MERGE (a)-[r:FRIENDS]->(b)
        """)
        assert cypher_graph.last_mutation_stats["relationships_created"] == 1


# ============================================================================
# Mutation stats return format
# ============================================================================


class TestMutationStatsReturn:
    """Mutation queries return stats directly (not just via last_mutation_stats)."""

    def test_create_returns_stats(self):
        """CREATE returns ResultView with stats."""
        g = KnowledgeGraph()
        result = g.cypher("CREATE (:Person {name: 'Alice', age: 30})")
        assert result.stats is not None
        assert result.stats["nodes_created"] == 1
        assert len(result) == 0

    def test_set_returns_stats(self):
        """SET returns ResultView with stats."""
        g = KnowledgeGraph()
        g.cypher("CREATE (:Person {name: 'Alice', age: 30})")
        result = g.cypher("MATCH (p:Person {name: 'Alice'}) SET p.age = 31")
        assert result.stats is not None
        assert result.stats["properties_set"] >= 1

    def test_delete_returns_stats(self):
        """DETACH DELETE returns ResultView with stats."""
        g = KnowledgeGraph()
        g.cypher("CREATE (:Person {name: 'Alice'})")
        result = g.cypher("MATCH (p:Person {name: 'Alice'}) DETACH DELETE p")
        assert result.stats is not None
        assert result.stats["nodes_deleted"] == 1

    def test_mutation_with_return_has_rows_and_stats(self):
        """CREATE ... RETURN returns ResultView with both rows and stats."""
        g = KnowledgeGraph()
        result = g.cypher("CREATE (n:Person {name: 'Bob', age: 25}) RETURN n.name AS name")
        assert result.stats is not None
        assert result.stats["nodes_created"] == 1
        assert len(result) == 1
        assert result[0]["name"] == "Bob"

    def test_read_query_returns_result_view(self):
        """Read query returns ResultView with no stats."""
        g = KnowledgeGraph()
        g.cypher("CREATE (:Person {name: 'Alice', age: 30})")
        result = g.cypher("MATCH (p:Person) RETURN p.name AS name")
        assert result.stats is None
        assert len(result) == 1
        assert result[0]["name"] == "Alice"

    def test_last_mutation_stats_backwards_compat(self):
        """last_mutation_stats property still works."""
        g = KnowledgeGraph()
        g.cypher("CREATE (:Person {name: 'Alice'})")
        stats = g.last_mutation_stats
        assert stats is not None
        assert stats["nodes_created"] == 1


# ============================================================================
# Parameter in MATCH inline properties
# ============================================================================


class TestParamInMatchPatterns:
    """$param in MATCH (n:Type {prop: $param}) inline properties."""

    def test_string_param_in_match(self):
        """MATCH (n:Person {name: $name}) resolves string parameter."""
        g = KnowledgeGraph()
        g.cypher("CREATE (:Person {name: 'Alice', age: 30})")
        g.cypher("CREATE (:Person {name: 'Bob', age: 25})")
        result = g.cypher("MATCH (p:Person {name: $name}) RETURN p.age AS age", params={"name": "Alice"})
        assert len(result) == 1
        assert result[0] == {"age": 30}

    def test_integer_param_in_match(self):
        """MATCH (n:Person {age: $age}) resolves integer parameter."""
        g = KnowledgeGraph()
        g.cypher("CREATE (:Person {name: 'Alice', age: 30})")
        g.cypher("CREATE (:Person {name: 'Bob', age: 25})")
        result = g.cypher("MATCH (p:Person {age: $age}) RETURN p.name AS name", params={"age": 30})
        assert len(result) == 1
        assert result[0] == {"name": "Alice"}

    def test_param_in_where_predicate_pushdown(self):
        """WHERE p.name = $name is pushed into MATCH for index acceleration."""
        g = KnowledgeGraph()
        g.cypher("CREATE (:Person {name: 'Alice', age: 30})")
        g.cypher("CREATE (:Person {name: 'Bob', age: 25})")
        result = g.cypher("MATCH (p:Person) WHERE p.name = $name RETURN p.age AS age", params={"name": "Alice"})
        assert len(result) == 1
        assert result[0] == {"age": 30}

    def test_multiple_params_in_match(self):
        """Multiple $params in same MATCH pattern."""
        g = KnowledgeGraph()
        g.cypher("CREATE (:Person {name: 'Alice', age: 30, city: 'Oslo'})")
        g.cypher("CREATE (:Person {name: 'Bob', age: 30, city: 'Bergen'})")
        result = g.cypher(
            "MATCH (p:Person {age: $age}) WHERE p.city = $city RETURN p.name AS name",
            params={"age": 30, "city": "Oslo"},
        )
        assert len(result) == 1
        assert result[0] == {"name": "Alice"}


# ============================================================================
# WITH clause property access regression (v0.4.17)
# ============================================================================


class TestWithPropertyAccess:
    """Node properties must survive WITH clause — regression test for v0.4.17."""

    def test_with_aggregation_preserves_properties(self):
        """WITH p, count(x) AS c then RETURN p.prop should return correct values."""
        g = KnowledgeGraph()
        g.cypher("CREATE (:Person {name: 'Alice', age: 30, city: 'Oslo'})")
        g.cypher("CREATE (:Person {name: 'Bob', age: 25, city: 'Bergen'})")
        g.cypher("""
            MATCH (a:Person {name: 'Alice'}), (b:Person {name: 'Bob'})
            CREATE (a)-[:KNOWS]->(b)
        """)
        result = g.cypher("""
            MATCH (p:Person)-[:KNOWS]->(other:Person)
            WITH p, count(other) AS friends
            RETURN p.name AS name, p.age AS age, p.city AS city, friends
        """)
        assert len(result) == 1
        assert result[0]["name"] == "Alice"
        assert result[0]["age"] == 30
        assert result[0]["city"] == "Oslo"
        assert result[0]["friends"] == 1

    def test_with_passthrough_preserves_properties(self):
        """WITH p (no aggregation) then RETURN p.prop should work."""
        g = KnowledgeGraph()
        g.cypher("CREATE (:Person {name: 'Alice', age: 30, city: 'Oslo'})")
        result = g.cypher("""
            MATCH (p:Person)
            WITH p
            RETURN p.name AS name, p.age AS age, p.city AS city
        """)
        assert len(result) == 1
        assert result[0]["name"] == "Alice"
        assert result[0]["age"] == 30
        assert result[0]["city"] == "Oslo"

    def test_double_with_preserves_properties(self):
        """Node passing through two WITH clauses retains all properties."""
        g = KnowledgeGraph()
        g.cypher("CREATE (:Law {title: 'Criminal Code', law_id: 'LOV-2005', dept: 'Justice'})")
        g.cypher("CREATE (:CourtDecision {title: 'Case-1'})")
        g.cypher("CREATE (:Regulation {title: 'Reg-1'})")
        g.cypher("""
            MATCH (cd:CourtDecision {title: 'Case-1'}), (l:Law {title: 'Criminal Code'})
            CREATE (cd)-[:CITES_LAW]->(l)
        """)
        g.cypher("""
            MATCH (r:Regulation {title: 'Reg-1'}), (l:Law {title: 'Criminal Code'})
            CREATE (r)-[:LEGAL_BASIS]->(l)
        """)
        result = g.cypher("""
            MATCH (l:Law)
            OPTIONAL MATCH (cd:CourtDecision)-[:CITES_LAW]->(l)
            WITH l, count(cd) AS case_cites
            OPTIONAL MATCH (r:Regulation)-[:LEGAL_BASIS]->(l)
            WITH l, case_cites, count(r) AS reg_basis
            RETURN l.title AS law, l.law_id AS law_id, l.dept AS dept, case_cites, reg_basis
        """)
        assert len(result) == 1
        assert result[0]["law"] == "Criminal Code"
        assert result[0]["law_id"] == "LOV-2005"
        assert result[0]["dept"] == "Justice"
        assert result[0]["case_cites"] == 1
        assert result[0]["reg_basis"] == 1

    def test_with_where_on_node_property(self):
        """WHERE on node property after WITH still works."""
        g = KnowledgeGraph()
        g.cypher("CREATE (:Person {name: 'Alice', age: 30})")
        g.cypher("CREATE (:Person {name: 'Bob', age: 25})")
        g.cypher("CREATE (:Person {name: 'Charlie', age: 40})")
        g.cypher("""
            MATCH (a:Person {name: 'Alice'}), (b:Person {name: 'Bob'})
            CREATE (a)-[:KNOWS]->(b)
        """)
        g.cypher("""
            MATCH (a:Person {name: 'Alice'}), (c:Person {name: 'Charlie'})
            CREATE (a)-[:KNOWS]->(c)
        """)
        g.cypher("""
            MATCH (a:Person {name: 'Bob'}), (c:Person {name: 'Charlie'})
            CREATE (a)-[:KNOWS]->(c)
        """)
        result = g.cypher("""
            MATCH (p:Person)-[:KNOWS]->(other:Person)
            WITH p, count(other) AS friends
            WHERE friends >= 2
            RETURN p.name AS name, p.age AS age, friends
        """)
        assert len(result) == 1
        assert result[0]["name"] == "Alice"
        assert result[0]["age"] == 30
        assert result[0]["friends"] == 2


# ── range() function ─────────────────────────────────────────────────────


class TestRange:
    def test_range_basic(self, cypher_graph):
        result = cypher_graph.cypher("UNWIND range(1, 5) AS x RETURN x")
        assert [r["x"] for r in result] == [1, 2, 3, 4, 5]

    def test_range_with_step(self, cypher_graph):
        result = cypher_graph.cypher("UNWIND range(0, 10, 3) AS x RETURN x")
        assert [r["x"] for r in result] == [0, 3, 6, 9]

    def test_range_negative_step(self, cypher_graph):
        result = cypher_graph.cypher("UNWIND range(5, 1, -1) AS x RETURN x")
        assert [r["x"] for r in result] == [5, 4, 3, 2, 1]

    def test_range_single_element(self, cypher_graph):
        result = cypher_graph.cypher("UNWIND range(3, 3) AS x RETURN x")
        assert [r["x"] for r in result] == [3]

    def test_range_empty(self, cypher_graph):
        """range(5, 1) with default step=1 produces empty list."""
        result = cypher_graph.cypher("UNWIND range(5, 1) AS x RETURN x")
        assert len(result) == 0

    def test_range_in_return(self, cypher_graph):
        result = cypher_graph.cypher("MATCH (p:Person {name: 'Alice'}) RETURN range(1, 3) AS r")
        assert result[0]["r"] == [1, 2, 3]

    def test_range_step_zero_errors(self, cypher_graph):
        with pytest.raises(kglite.KgError, match="step must not be zero"):
            cypher_graph.cypher("UNWIND range(1, 5, 0) AS x RETURN x")


# ============================================================================
# Variable binding in MATCH pattern properties
# ============================================================================


class TestVariableBindingInPatterns:
    """WITH/UNWIND variables used in MATCH pattern properties: {prop: varName}."""

    def test_with_scalar_in_match_property(self):
        """WITH "Alice" AS name MATCH (n:Person {name: name}) RETURN n."""
        g = KnowledgeGraph()
        g.cypher("CREATE (:Person {name: 'Alice', age: 30})")
        g.cypher("CREATE (:Person {name: 'Bob', age: 25})")
        result = g.cypher('WITH "Alice" AS name MATCH (p:Person {name: name}) RETURN p.age AS age')
        assert len(result) == 1
        assert result[0] == {"age": 30}

    def test_unwind_variable_in_match_property(self):
        """UNWIND names AS name MATCH (n {name: name}) for each value."""
        g = KnowledgeGraph()
        g.cypher("CREATE (:Person {name: 'Alice', age: 30})")
        g.cypher("CREATE (:Person {name: 'Bob', age: 25})")
        result = g.cypher(
            'UNWIND ["Alice", "Bob"] AS name '
            "MATCH (p:Person {name: name}) RETURN p.name AS name, p.age AS age "
            "ORDER BY age"
        )
        assert len(result) == 2
        assert result[0] == {"name": "Bob", "age": 25}
        assert result[1] == {"name": "Alice", "age": 30}

    def test_integer_variable_in_match_property(self):
        """Integer variable binding in pattern properties."""
        g = KnowledgeGraph()
        g.cypher("CREATE (:Person {name: 'Alice', age: 30})")
        g.cypher("CREATE (:Person {name: 'Bob', age: 25})")
        result = g.cypher("WITH 30 AS target_age MATCH (p:Person {age: target_age}) RETURN p.name AS name")
        assert len(result) == 1
        assert result[0] == {"name": "Alice"}

    def test_variable_no_match_returns_empty(self):
        """Variable binding that matches nothing returns empty result."""
        g = KnowledgeGraph()
        g.cypher("CREATE (:Person {name: 'Alice'})")
        result = g.cypher('WITH "Nobody" AS name MATCH (p:Person {name: name}) RETURN p')
        assert len(result) == 0

    def test_variable_with_multiple_match_patterns(self):
        """Variable from first MATCH used in second MATCH pattern."""
        g = KnowledgeGraph()
        g.cypher("CREATE (:Person {name: 'Alice', city: 'Oslo'})")
        g.cypher("CREATE (:City {name: 'Oslo', country: 'Norway'})")
        result = g.cypher(
            "MATCH (p:Person {name: 'Alice'}) "
            "WITH p.city AS city_name "
            "MATCH (c:City {name: city_name}) "
            "RETURN c.country AS country"
        )
        assert len(result) == 1
        assert result[0] == {"country": "Norway"}


# ============================================================================
# Map literals in expressions
# ============================================================================


class TestMapLiterals:
    """Map literal expressions: {key: expr, key2: expr}."""

    def test_map_literal_in_return(self):
        """RETURN {name: n.name, age: n.age} AS person_map."""
        g = KnowledgeGraph()
        g.cypher("CREATE (:Person {name: 'Alice', age: 30})")
        result = g.cypher("MATCH (n:Person) RETURN {name: n.name, age: n.age} AS m")
        assert len(result) == 1
        m = result[0]["m"]
        # Result may be a dict (auto-parsed) or JSON string
        if isinstance(m, str):
            import json

            m = json.loads(m)
        assert m["name"] == "Alice"
        assert m["age"] == 30

    def test_map_literal_in_with(self):
        """WITH {x: 1, y: 2} AS point RETURN point."""
        g = KnowledgeGraph()
        result = g.cypher("WITH {x: 1, y: 2} AS point RETURN point")
        assert len(result) == 1
        m = result[0]["point"]
        if isinstance(m, str):
            import json

            m = json.loads(m)
        assert m["x"] == 1
        assert m["y"] == 2

    def test_map_literal_with_expressions(self):
        """Map literal values can be expressions, not just literals."""
        g = KnowledgeGraph()
        g.cypher("CREATE (:Person {name: 'Alice', age: 30})")
        result = g.cypher("MATCH (n:Person) RETURN {name: n.name, next_age: n.age + 1} AS m")
        assert len(result) == 1
        m = result[0]["m"]
        if isinstance(m, str):
            import json

            m = json.loads(m)
        assert m["name"] == "Alice"
        assert m["next_age"] == 31

    def test_empty_map_literal(self):
        """Empty map literal {}."""
        g = KnowledgeGraph()
        result = g.cypher("RETURN {} AS m")
        assert len(result) == 1
        m = result[0]["m"]
        if isinstance(m, str):
            import json

            m = json.loads(m)
        assert m == {}

    def test_map_literal_with_string_values(self):
        """Map literal with string values."""
        g = KnowledgeGraph()
        result = g.cypher('RETURN {status: "active", role: "admin"} AS m')
        assert len(result) == 1
        m = result[0]["m"]
        if isinstance(m, str):
            import json

            m = json.loads(m)
        assert m["status"] == "active"
        assert m["role"] == "admin"


class TestMultiMatchEmptyPropagation:
    """Regression: second MATCH must return 0 rows when first MATCH is empty."""

    def test_empty_first_match(self, cypher_graph):
        result = cypher_graph.cypher("MATCH (n:NonExistent) MATCH (m:Person) RETURN count(m) AS cnt")
        assert result[0]["cnt"] == 0

    def test_where_false_then_match(self, cypher_graph):
        result = cypher_graph.cypher("MATCH (n:Person) WHERE false MATCH (m:Person) RETURN count(m) AS cnt")
        assert result[0]["cnt"] == 0

    def test_empty_match_then_optional_match(self, cypher_graph):
        result = cypher_graph.cypher("MATCH (n:NonExistent) OPTIONAL MATCH (n)-[r]->(m) RETURN m")
        assert len(result) == 0

    def test_unwind_empty_then_match(self, cypher_graph):
        result = cypher_graph.cypher("UNWIND [] AS x MATCH (m:Person) RETURN count(m) AS cnt")
        assert result[0]["cnt"] == 0

    def test_normal_multi_match_still_works(self, cypher_graph):
        result = cypher_graph.cypher("MATCH (a:Person) MATCH (b:Person) RETURN count(*) AS cnt")
        assert result[0]["cnt"] == 25  # 5 persons x 5 persons

    def test_bound_variable_reuse_after_empty(self, cypher_graph):
        """WHERE equality filters to 0 → second MATCH must propagate empty."""
        result = cypher_graph.cypher("""
            MATCH (a:Person)-[:PURCHASED]->(p:Product)
            WHERE p.title = 'NonExistentProduct'
            MATCH (a)-[:PURCHASED]->(p2:Product)
            RETURN count(*) AS cnt
        """)
        assert result[0]["cnt"] == 0

    def test_non_pushable_where_then_second_match(self, cypher_graph):
        """STARTS WITH is not pushable — WHERE filters to 0, second MATCH must propagate empty."""
        result = cypher_graph.cypher("""
            MATCH (a:Person)-[:PURCHASED]->(p:Product)
            WHERE a.title STARTS WITH 'ZZZ'
            MATCH (a)-[:KNOWS]->(b:Person)
            RETURN count(*) AS cnt
        """)
        assert result[0]["cnt"] == 0

    def test_mixed_pushable_and_non_pushable_where(self, cypher_graph):
        """Pushable equality + non-pushable STARTS WITH → 0 rows, second MATCH empty."""
        result = cypher_graph.cypher("""
            MATCH (a:Person)-[:PURCHASED]->(p:Product)
            WHERE p.title = 'Laptop' AND a.title STARTS WITH 'ZZZ'
            MATCH (a)-[:KNOWS]->(b:Person)
            RETURN count(*) AS cnt
        """)
        assert result[0]["cnt"] == 0

    def test_where_to_zero_with_reused_variable(self, cypher_graph):
        """Second MATCH reuses variable from first — must not match globally on empty."""
        # Alice purchased Laptop+Phone (2 rows), WHERE filters to 0,
        # second MATCH (a)-[:KNOWS]->(b) would give 10 rows if run globally.
        result = cypher_graph.cypher("""
            MATCH (a:Person)-[:PURCHASED]->(p:Product)
            WHERE a.title = 'Alice' AND p.title = 'NonExistent'
            MATCH (a)-[:KNOWS]->(b:Person)
            RETURN count(*) AS cnt
        """)
        assert result[0]["cnt"] == 0


# ============================================================================
# Bug regression tests — guards against the issues named in each class
# docstring. All currently pass; classes remain named `TestBug*` for traceability.
# ============================================================================


class TestBugDatetimeFunction:
    """BUG: datetime() parses as date and crashes on the time portion."""

    def test_datetime_literal(self, cypher_graph):
        rows = cypher_graph.cypher("RETURN datetime('2024-03-15T10:30:00') AS dt")
        assert len(rows) == 1
        assert "2024-03-15" in str(rows[0]["dt"])

    def test_datetime_with_timezone(self, cypher_graph):
        rows = cypher_graph.cypher("RETURN datetime('2024-03-15T10:30:00Z') AS dt")
        assert len(rows) == 1


class TestBugDateInvalidInput:
    """BUG: date() crashes on invalid input instead of returning null."""

    def test_date_empty_string_returns_null(self, cypher_graph):
        rows = cypher_graph.cypher("RETURN date('') AS d")
        assert rows[0]["d"] is None

    def test_date_zero_month_returns_null(self, cypher_graph):
        rows = cypher_graph.cypher("RETURN date('2016-00-00') AS d")
        assert rows[0]["d"] is None

    def test_date_partial_month_returns_null(self, cypher_graph):
        """date('2025-03') should return null or '2025-03-01', not crash."""
        rows = cypher_graph.cypher("RETURN date('2016-13-01') AS d")
        assert rows[0]["d"] is None


class TestBugDatePropertyAccessor:
    """BUG: date('...').year syntax not supported on function results."""

    def test_date_literal_dot_year(self, cypher_graph):
        rows = cypher_graph.cypher("RETURN date('2024-06-15').year AS yr")
        assert rows[0]["yr"] == 2024

    def test_date_literal_dot_month(self, cypher_graph):
        rows = cypher_graph.cypher("RETURN date('2024-06-15').month AS mo")
        assert rows[0]["mo"] == 6

    def test_date_literal_dot_day(self, cypher_graph):
        rows = cypher_graph.cypher("RETURN date('2024-06-15').day AS dy")
        assert rows[0]["dy"] == 15


class TestBugRelationshipTypePipe:
    """BUG: Pipe | in relationship types not parsed."""

    def test_pipe_two_types(self, cypher_graph):
        rows = cypher_graph.cypher("""
            MATCH (a:Person)-[:KNOWS|PURCHASED]->(b)
            WHERE a.name = 'Alice'
            RETURN b.title
        """)
        # Alice KNOWS Bob+Charlie, PURCHASED Laptop+Phone → 4 results
        assert len(rows) == 4

    def test_pipe_three_types(self, cypher_graph):
        rows = cypher_graph.cypher("""
            MATCH (a)-[:KNOWS|PURCHASED|SOLD]->(b)
            RETURN a.title, b.title LIMIT 1
        """)
        assert len(rows) >= 1


class TestBugXorOperator:
    """BUG: XOR logical operator not implemented."""

    def test_xor_basic(self, cypher_graph):
        rows = cypher_graph.cypher("""
            MATCH (n:Person)
            WHERE n.city = 'Oslo' XOR n.age > 35
            RETURN n.name
        """)
        # Oslo: Alice(30), Charlie(35), Eve(40) — age > 35: Eve
        # XOR: Oslo-only (Alice, Charlie) + age>35-not-Oslo (none) = Alice, Charlie
        names = {r["n.name"] for r in rows}
        assert names == {"Alice", "Charlie"}


class TestBugModuloOperator:
    """BUG: Modulo % operator not implemented."""

    def test_modulo_basic(self, cypher_graph):
        rows = cypher_graph.cypher("RETURN 10 % 3 AS result")
        assert rows[0]["result"] == 1

    def test_modulo_with_property(self, cypher_graph):
        rows = cypher_graph.cypher("""
            MATCH (n:Person)
            WHERE n.age % 10 = 0
            RETURN n.name
        """)
        # age 30, 40 are divisible by 10 → Alice, Eve
        names = {r["n.name"] for r in rows}
        assert names == {"Alice", "Eve"}


class TestBugHeadLastFunctions:
    """BUG: head() and last() functions not implemented."""

    def test_head_basic(self, cypher_graph):
        rows = cypher_graph.cypher("RETURN head([1, 2, 3]) AS h")
        assert rows[0]["h"] == 1

    def test_last_basic(self, cypher_graph):
        rows = cypher_graph.cypher("RETURN last([1, 2, 3]) AS l")
        assert rows[0]["l"] == 3

    def test_head_empty_list(self, cypher_graph):
        rows = cypher_graph.cypher("RETURN head([]) AS h")
        assert rows[0]["h"] is None

    def test_last_empty_list(self, cypher_graph):
        rows = cypher_graph.cypher("RETURN last([]) AS l")
        assert rows[0]["l"] is None

    def test_head_with_collect(self, cypher_graph):
        rows = cypher_graph.cypher("""
            MATCH (n:Person)
            WITH collect(n.name) AS names
            RETURN head(names) AS first_name
        """)
        assert rows[0]["first_name"] is not None


class TestBugInWithVariable:
    """BUG: IN operator with variable reference fails (only literal lists work)."""

    def test_in_with_with_variable(self, cypher_graph):
        rows = cypher_graph.cypher("""
            WITH ['Oslo', 'Bergen'] AS cities
            MATCH (n:Person)
            WHERE n.city IN cities
            RETURN n.name
        """)
        assert len(rows) == 5  # all people are in Oslo or Bergen

    def test_in_with_collect_result(self, cypher_graph):
        rows = cypher_graph.cypher("""
            MATCH (a:Person)-[:KNOWS]->(b:Person)
            WHERE a.name = 'Alice'
            WITH collect(b.name) AS friends
            MATCH (n:Person)
            WHERE n.name IN friends
            RETURN n.name
        """)
        names = {r["n.name"] for r in rows}
        assert names == {"Bob", "Charlie"}

    def test_in_with_unwind_source(self, cypher_graph):
        rows = cypher_graph.cypher("""
            WITH ['Alice', 'Bob'] AS target_names
            MATCH (n:Person)
            WHERE n.name IN target_names
            RETURN n.name ORDER BY n.name
        """)
        assert [r["n.name"] for r in rows] == ["Alice", "Bob"]


class TestBugBooleanExpressionsInReturn:
    """BUG: Boolean/comparison expressions in RETURN clause fail."""

    def test_starts_with_in_return(self, cypher_graph):
        rows = cypher_graph.cypher("""
            MATCH (n:Person)
            WHERE n.name = 'Alice'
            RETURN n.name, n.name STARTS WITH 'A' AS starts_a
        """)
        assert rows[0]["starts_a"] is True

    def test_comparison_in_return(self, cypher_graph):
        rows = cypher_graph.cypher("""
            MATCH (n:Person)
            WHERE n.name = 'Alice'
            RETURN n.name, n.age > 25 AS over_25
        """)
        assert rows[0]["over_25"] is True

    def test_contains_in_return(self, cypher_graph):
        rows = cypher_graph.cypher("""
            MATCH (n:Person)
            WHERE n.name = 'Charlie'
            RETURN n.name CONTAINS 'arli' AS has_arli
        """)
        assert rows[0]["has_arli"] is True

    def test_regex_in_return(self, cypher_graph):
        rows = cypher_graph.cypher("""
            MATCH (n:Person)
            WHERE n.name = 'Alice'
            RETURN n.name =~ 'A.*' AS matches
        """)
        assert rows[0]["matches"] is True


class TestBugMapAllProperties:
    """BUG: Map all-properties {.*} projection not supported."""

    def test_map_all_properties(self, cypher_graph):
        rows = cypher_graph.cypher("""
            MATCH (n:Person)
            WHERE n.name = 'Alice'
            RETURN n {.*} AS props
        """)
        assert rows[0]["props"]["title"] == "Alice"
        assert rows[0]["props"]["age"] == 30


class TestBugStDevFunction:
    """BUG: stDev() aggregate function not recognized."""

    def test_stdev_on_property(self, cypher_graph):
        rows = cypher_graph.cypher("MATCH (n:Person) RETURN stDev(n.age) AS sd")
        assert rows[0]["sd"] is not None
        assert isinstance(rows[0]["sd"], float)
        assert rows[0]["sd"] > 0

    def test_stdev_on_unwind(self, cypher_graph):
        rows = cypher_graph.cypher("UNWIND [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] AS x RETURN stDev(x) AS sd")
        # sample stdev of 1..10 ≈ 3.0277
        assert abs(rows[0]["sd"] - 3.0277) < 0.01

    def test_stdev_single_value_returns_zero_or_null(self, cypher_graph):
        rows = cypher_graph.cypher("UNWIND [42] AS x RETURN stDev(x) AS sd")
        # sample stdev of single value: 0 or null (N-1 = 0)
        assert rows[0]["sd"] is None or rows[0]["sd"] == 0

    def test_stdev_with_grouping(self, cypher_graph):
        rows = cypher_graph.cypher("""
            MATCH (n:Person)
            RETURN n.city AS city, stDev(n.age) AS sd
            ORDER BY city
        """)
        assert len(rows) == 2  # Bergen, Oslo


class TestBugNullComparison:
    """BUG: null = null and null <> null fail with syntax error."""

    def test_null_equals_null(self, cypher_graph):
        rows = cypher_graph.cypher("RETURN null = null AS result")
        # Neo4j: null = null → null (not true, not false)
        assert rows[0]["result"] is None

    def test_null_not_equals_null(self, cypher_graph):
        rows = cypher_graph.cypher("RETURN null <> null AS result")
        assert rows[0]["result"] is None


class TestThreeValuedNullSemantics:
    """B1 + B2: WHERE predicates propagate NULL (Kleene three-valued logic).

    Bob and Diana have email=None in the fixture. Under correct openCypher
    semantics a comparison or string predicate involving NULL evaluates to
    NULL, and WHERE excludes the row.

    Before the fix, NotEquals(NULL, x) collapsed to true and `NOT (NULL
    CONTAINS x)` evaluated to true, keeping the missing-property rows.
    """

    # B1: comparison operators propagate NULL

    def test_b1_ne_with_null_excludes_missing(self, cypher_graph):
        rows = cypher_graph.cypher("MATCH (p:Person) WHERE p.email <> 'alice@test.com' RETURN p.title AS n ORDER BY n")
        # Bob, Diana excluded (NULL <> 'alice' is NULL); Alice excluded by inequality.
        assert [r["n"] for r in rows] == ["Charlie", "Eve"]

    def test_b1_eq_with_null_still_excludes(self, cypher_graph):
        rows = cypher_graph.cypher("MATCH (p:Person) WHERE p.email = 'alice@test.com' RETURN p.title AS n")
        assert [r["n"] for r in rows] == ["Alice"]

    def test_b1_lt_with_null_excludes_missing(self, cypher_graph):
        rows = cypher_graph.cypher("MATCH (p:Person) WHERE p.email < 'z' RETURN p.title AS n ORDER BY n")
        # Bob, Diana excluded (NULL < 'z' is NULL).
        assert [r["n"] for r in rows] == ["Alice", "Charlie", "Eve"]

    def test_b1_not_lt_with_null_excludes_missing(self, cypher_graph):
        rows = cypher_graph.cypher("MATCH (p:Person) WHERE NOT (p.email < 'z') RETURN p.title AS n ORDER BY n")
        # All three with email evaluate `< 'z'` as true → NOT true → false → drop.
        # Bob and Diana: NULL < 'z' is NULL → NOT NULL is NULL → drop.
        # Nobody matches.
        assert len(rows) == 0

    # B2: string predicates propagate NULL, NOT preserves it

    def test_b2_not_contains_with_null_excludes_missing(self, cypher_graph):
        rows = cypher_graph.cypher(
            "MATCH (p:Person) WHERE NOT (p.email CONTAINS 'alice') RETURN p.title AS n ORDER BY n"
        )
        # Before fix: kept Bob and Diana because NOT (NULL CONTAINS 'x') = NOT false = true.
        assert [r["n"] for r in rows] == ["Charlie", "Eve"]

    def test_b2_not_starts_with_with_null_excludes_missing(self, cypher_graph):
        rows = cypher_graph.cypher(
            "MATCH (p:Person) WHERE NOT (p.email STARTS WITH 'alice') RETURN p.title AS n ORDER BY n"
        )
        assert [r["n"] for r in rows] == ["Charlie", "Eve"]

    def test_b2_not_ends_with_with_null_excludes_missing(self, cypher_graph):
        rows = cypher_graph.cypher(
            "MATCH (p:Person) WHERE NOT (p.email ENDS WITH 'test.com') RETURN p.title AS n ORDER BY n"
        )
        # Everyone with email matches `ENDS WITH 'test.com'` → NOT true → false → drop.
        # Bob, Diana: NULL → NOT NULL → NULL → drop.
        assert len(rows) == 0

    def test_b2_contains_with_null_excludes_missing(self, cypher_graph):
        rows = cypher_graph.cypher("MATCH (p:Person) WHERE p.email CONTAINS 'alice' RETURN p.title AS n")
        # Bare positive CONTAINS — the existing collapse already gave the
        # right answer here, but lock it in alongside the fix.
        assert [r["n"] for r in rows] == ["Alice"]

    # Kleene composition: AND / OR with NULL operand

    def test_kleene_or_null_absorbs_when_other_is_true(self, cypher_graph):
        rows = cypher_graph.cypher(
            "MATCH (p:Person) WHERE p.email = 'never' OR p.title = 'Bob' RETURN p.title AS n ORDER BY n"
        )
        # For Bob: (NULL OR true) = true → kept. Others fail both sides.
        assert [r["n"] for r in rows] == ["Bob"]

    def test_kleene_and_null_absorbs_when_other_is_false(self, cypher_graph):
        rows = cypher_graph.cypher(
            "MATCH (p:Person) WHERE p.email <> 'never' AND p.title = 'Alice' RETURN p.title AS n"
        )
        # For Bob: (NULL AND false) = false → dropped. For Alice: (true AND true) = true.
        assert [r["n"] for r in rows] == ["Alice"]

    def test_kleene_and_null_when_other_is_true(self, cypher_graph):
        rows = cypher_graph.cypher(
            "MATCH (p:Person) WHERE p.email <> 'never' AND p.age > 0 RETURN p.title AS n ORDER BY n"
        )
        # For Bob/Diana: (NULL AND true) = NULL → dropped (row excluded).
        # For Alice/Charlie/Eve: (true AND true) = true → kept.
        assert [r["n"] for r in rows] == ["Alice", "Charlie", "Eve"]

    def test_kleene_or_null_when_other_is_false(self, cypher_graph):
        rows = cypher_graph.cypher("MATCH (p:Person) WHERE p.email = 'never' OR p.age < 0 RETURN p.title AS n")
        # Bob/Diana: (NULL OR false) = NULL → dropped.
        # Others: (false OR false) = false → dropped.
        assert len(rows) == 0


class TestNullInPredicate:
    """openCypher: `IN` with a NULL operand propagates NULL.

    Three cases:
      1. NULL LHS:                     `NULL IN [1, 2]`        → NULL
      2. NULL element, no match:       `'x' IN ['a', NULL]`    → NULL
      3. NULL inside expr-evaluated:   `(NULL + 1) IN [2]`     → NULL

    The deferred-fix path lives at where_clause.rs:543-552 (`Predicate::In`)
    and 592-603 (`Predicate::InExpression`). Both currently collapse to
    boolean (return `Ok(Some(false))` on no-match, even when NULL was
    involved). These tests pin the desired behaviour against the
    fixture-graph and are marked xfail with the source pointer until
    the fix lands — same playbook as the B1/B2 NULL fix.
    """

    def test_null_lhs_in_literal_list(self, cypher_graph):
        """NULL IN [1, 2, 3] → NULL → row excluded.

        Use Bob (email=None) so `p.email IN ['a', 'b']` exercises the
        NULL-LHS path. None of his emails match either literal so the
        pre-fix code returns false (row excluded), which happens to
        agree with the openCypher answer in *this* case. Differential
        check below tests the discriminating case.
        """
        # Pre-fix gives [Alice, Charlie] for the bare-positive query —
        # these aren't NULL so they're fine. Bob and Diana (email=None):
        # NULL IN [...] should be NULL → excluded. Pre-fix also excludes
        # (no match → false). Same answer in that case.
        # The discriminating case is the negation below: NULL IN [...]
        # under NOT keeps Bob/Diana on the pre-fix path but should drop
        # them under correct openCypher semantics.
        rows_neg = cypher_graph.cypher(
            "MATCH (p:Person) WHERE NOT (p.email IN ['alice@test.com', 'charlie@test.com']) "
            "RETURN p.title AS n ORDER BY n"
        )
        # Eve should match (her email exists, ne both); Bob/Diana have NULL email,
        # NULL IN [...] is NULL, NOT NULL is NULL, row excluded.
        # Pre-fix: NULL IN [...] is false, NOT false is true, Bob+Diana INCLUDED.
        assert [r["n"] for r in rows_neg] == ["Eve"], f"got {[r['n'] for r in rows_neg]}"

    def test_null_element_in_list_no_match(self, cypher_graph):
        """When the list contains NULL and the LHS doesn't match any
        non-NULL element, openCypher returns NULL (not false). Pre-fix
        returns false (which then incorrectly evaluates to true under NOT)."""
        rows = cypher_graph.cypher(
            "MATCH (p:Person) WHERE NOT (p.title IN ['Alice', null, 'Bob']) RETURN p.title AS n ORDER BY n"
        )
        # Charlie/Diana/Eve don't match the non-NULL entries; list contains NULL
        # → NULL → NOT NULL is NULL → rows excluded.
        # Pre-fix: NULL IN list returns false, NOT false is true, ALL excluded
        # rows returned. The desired behaviour is empty result.
        assert [r["n"] for r in rows] == [], f"got {[r['n'] for r in rows]}"

    def test_null_lhs_in_parameter_list(self, cypher_graph):
        """Same NULL-LHS rule but through `Predicate::InExpression` — the
        parser routes parameter-RHS `IN` to that arm instead of the
        literal-list `Predicate::In`."""
        rows = cypher_graph.cypher(
            "MATCH (p:Person) WHERE NOT (p.email IN $allowed) RETURN p.title AS n ORDER BY n",
            params={"allowed": ["alice@test.com", "charlie@test.com"]},
        )
        # Eve has an email that doesn't match either entry: NOT false = true → included.
        # Bob/Diana have NULL email: NULL IN [...] is NULL → NOT NULL is NULL → excluded.
        assert [r["n"] for r in rows] == ["Eve"], f"got {[r['n'] for r in rows]}"

    def test_null_in_parameter_list_no_match(self, cypher_graph):
        """RHS parameter containing NULL with no LHS match — same NULL
        propagation rule, exercises the saw_null branch of InExpression."""
        rows = cypher_graph.cypher(
            "MATCH (p:Person) WHERE NOT (p.title IN $names) RETURN p.title AS n ORDER BY n",
            params={"names": ["Alice", None, "Bob"]},
        )
        # Charlie/Diana/Eve: title not in [Alice, Bob]; list contains NULL → NULL → NOT NULL → excluded.
        # Alice/Bob: matched → NOT true → false → excluded.
        # Result: empty.
        assert [r["n"] for r in rows] == [], f"got {[r['n'] for r in rows]}"


class TestNumericBoundaries:
    """Pin the observable behaviour for numeric edge cases.

    Some of these are openCypher-spec-defensible, some are bugs. Each
    test documents which is which. Pinning the current behaviour means
    a future change is forced to be deliberate.
    """

    def test_int64_overflow_wraps_silently(self):
        """Int64::MAX + 1 silently wraps to Int64::MIN.

        This is the underlying Rust `wrapping_add` behaviour and is
        worth flagging — openCypher technically expects either an
        error or a Float64 promotion. KGLite chose silent wrap for
        consistency with arithmetic on `+` / `-`. Pinning so a future
        change to error/promote is deliberate.
        """
        g = KnowledgeGraph()
        rows = g.cypher("RETURN 9223372036854775807 + 1 AS r").to_list()
        assert rows[0]["r"] == -9223372036854775808  # Int64::MIN

    def test_int64_min_expressible_as_literal(self):
        """`-9223372036854775808` should parse as Int64::MIN."""
        g = KnowledgeGraph()
        rows = g.cypher("RETURN -9223372036854775808 AS r").to_list()
        assert rows[0]["r"] == -9223372036854775808

    def test_integer_division_truncates_toward_zero(self):
        """-5 / 2 = -2 (not -3); 5 / -2 = -2 (not -3).

        Matches Rust's i64 division semantics and most languages.
        Neo4j also rounds toward zero. Stable and defensible."""
        g = KnowledgeGraph()
        rows = g.cypher("RETURN -5 / 2 AS a, 5 / -2 AS b").to_list()
        assert rows[0]["a"] == -2
        assert rows[0]["b"] == -2

    def test_division_by_zero_returns_null(self):
        """Both int and float division by zero returns NULL.

        openCypher would prescribe NULL for int/0 and ±Inf / NaN for
        float/0, but KGLite has chosen the more conservative
        "always NULL" path. Pinning so a future Inf/NaN swap is
        explicit."""
        g = KnowledgeGraph()
        rows = g.cypher(
            "RETURN 5 / 0 AS int_zero, 1.0 / 0.0 AS pos_inf, -1.0 / 0.0 AS neg_inf, 0.0 / 0.0 AS nan"
        ).to_list()
        assert rows[0]["int_zero"] is None
        assert rows[0]["pos_inf"] is None
        assert rows[0]["neg_inf"] is None
        assert rows[0]["nan"] is None

    def test_modulo_sign_follows_dividend(self):
        """-5 % 3 = -2; 5 % -3 = 2. Sign of result follows the dividend.

        Rust / Java semantics; differs from Python's `(-5) % 3 == 1`."""
        g = KnowledgeGraph()
        rows = g.cypher("RETURN -5 % 3 AS a, 5 % -3 AS b").to_list()
        assert rows[0]["a"] == -2
        assert rows[0]["b"] == 2


class TestCollectionEdges:
    """Pin list/collection edge-case behaviour: indexing, slicing,
    comprehension over empty / NULL inputs."""

    def test_head_last_on_null_returns_null(self):
        """head(null) and last(null) propagate NULL (not raise)."""
        g = KnowledgeGraph()
        rows = g.cypher("RETURN head(null) AS h, last(null) AS l").to_list()
        assert rows[0] == {"h": None, "l": None}

    def test_negative_indexing(self):
        """`list[-1]` is the last element, `[-N]` the first if N == size."""
        g = KnowledgeGraph()
        rows = g.cypher("RETURN [1, 2, 3][-1] AS a, [1, 2, 3][-3] AS b").to_list()
        assert rows[0] == {"a": 3, "b": 1}

    def test_out_of_bounds_indexing_returns_null(self):
        """`list[N]` and `list[-N]` where |N| > size(list) returns NULL.
        openCypher's preferred semantics (no IndexError)."""
        g = KnowledgeGraph()
        rows = g.cypher("RETURN [1, 2, 3][10] AS pos, [1, 2, 3][-10] AS neg").to_list()
        assert rows[0] == {"pos": None, "neg": None}

    def test_out_of_bounds_slicing_returns_empty(self):
        """`list[10..20]` on a 3-element list returns [] (not error)."""
        g = KnowledgeGraph()
        rows = g.cypher("RETURN [1, 2, 3][10..20] AS r").to_list()
        assert rows[0]["r"] == []

    def test_list_comprehension_on_empty_input(self):
        """`[x IN [] WHERE … | …]` returns [] without raising."""
        g = KnowledgeGraph()
        rows = g.cypher("RETURN [x IN [] WHERE x IS NULL | x] AS r").to_list()
        assert rows[0]["r"] == []

    def test_slice_with_both_ends_omitted(self):
        """`[1,2,3][..]` should equal `[1, 2, 3]`."""
        g = KnowledgeGraph()
        rows = g.cypher("RETURN [1, 2, 3][..] AS r").to_list()
        assert rows[0]["r"] == [1, 2, 3]


class TestStringUnicodeEdges:
    """Pin string-operation behaviour on unicode / edge inputs.

    Probed during the 0.9.52+ test-suite fortification — all of these
    *worked* as openCypher would expect. Pinning so a future refactor
    that breaks unicode handling is caught immediately.
    """

    def test_substring_respects_char_boundaries_emoji(self):
        """substring() counts logical characters, not bytes. 'a😀b' is
        3 chars; substring(s, 0, 2) returns 'a😀' even though the emoji
        occupies 4 UTF-8 bytes."""
        g = KnowledgeGraph()
        rows = g.cypher("RETURN substring('a😀b', 0, 2) AS r, substring('a😀b', 1, 1) AS s").to_list()
        assert rows[0] == {"r": "a😀", "s": "😀"}

    def test_trim_handles_unicode_whitespace(self):
        """Non-breaking space (U+00A0) and ideographic space (U+3000)
        both count as trimmable whitespace."""
        g = KnowledgeGraph()
        rows = g.cypher("RETURN trim(' x') AS a, trim('　x') AS b").to_list()
        assert rows[0] == {"a": "x", "b": "x"}

    def test_regex_accented_char_range(self):
        """`=~ '^[À-ÿ]+$'` matches strings entirely composed of accented
        Latin chars in that Unicode range, rejects ASCII-mixed strings."""
        g = KnowledgeGraph()
        rows = g.cypher("RETURN 'é' =~ '^[À-ÿ]+$' AS pure, 'Café' =~ '^[À-ÿ]+$' AS mixed").to_list()
        # 'é' (U+00E9) is in [À-ÿ] (U+00C0..U+00FF), matches.
        # 'C' (U+0043) is in ASCII, so 'Café' fails the class.
        assert rows[0] == {"pure": True, "mixed": False}

    def test_toupper_tolower_on_accented_chars(self):
        """Accented Latin chars round-trip through case conversion."""
        g = KnowledgeGraph()
        rows = g.cypher("RETURN toUpper('café') AS u, toLower('CAFÉ') AS l").to_list()
        assert rows[0] == {"u": "CAFÉ", "l": "café"}


class TestAggregateNullHandling:
    """Pin aggregate-function behaviour over NULL-bearing data.

    openCypher rules (which KGLite already implements correctly per
    the 0.9.52 probing): aggregates skip NULL inputs. count(*) counts
    all rows; count(prop) counts non-NULL values. collect() drops
    NULLs. min/max return NULL on all-NULL columns.

    Pinning so a future change that propagates NULL through aggregates
    (a real Cypher dialect choice for some engines) is deliberate.
    """

    @staticmethod
    def _fixture():
        g = KnowledgeGraph()
        g.add_nodes(
            pd.DataFrame(
                [
                    {"id": 1, "name": "a", "val": 10},
                    {"id": 2, "name": "b", "val": None},
                    {"id": 3, "name": "c", "val": 30},
                    {"id": 4, "name": "d", "val": None},
                ]
            ),
            "T",
            "id",
            "name",
        )
        return g

    def test_count_star_vs_count_prop_with_nulls(self):
        """count(*) = 4 (all rows), count(n.val) = 2 (non-NULL values)."""
        g = self._fixture()
        rows = g.cypher("MATCH (n:T) RETURN count(*) AS star, count(n.val) AS prop").to_list()
        assert rows[0] == {"star": 4, "prop": 2}

    def test_min_max_on_all_null_column_returns_null(self):
        """min/max over a column that is NULL for every row → NULL (not error)."""
        g = self._fixture()
        rows = g.cypher("MATCH (n:T) RETURN min(n.nonexistent) AS mn, max(n.nonexistent) AS mx").to_list()
        assert rows[0] == {"mn": None, "mx": None}

    def test_collect_skips_nulls_by_default(self):
        """collect(prop) returns only the non-NULL values. openCypher
        does NOT preserve NULLs in collect by default — Neo4j and
        Memgraph agree on this."""
        g = self._fixture()
        rows = g.cypher("MATCH (n:T) RETURN collect(n.val) AS r").to_list()
        # Order-insensitive: just check the multiset.
        assert sorted(rows[0]["r"]) == [10, 30] or sorted(rows[0]["r"]) == [10.0, 30.0]


class TestPatternMatchingEdges:
    """Pin pattern-matching behaviour on shapes that don't appear in the
    shared fixtures: self-loops, parallel edges, zero-length variable-
    length paths. Each builds its own focused graph."""

    def test_self_loop_pattern(self):
        """`(a)-[:SELF]->(b)` correctly matches the loop with a == b."""
        g = KnowledgeGraph()
        g.cypher("CREATE (a:N {eid: 1, name: 'a'}), (b:N {eid: 2, name: 'b'})")
        g.cypher("MATCH (a:N {eid: 1}) CREATE (a)-[:SELF]->(a)")
        rows = g.cypher("MATCH (a:N)-[:SELF]->(b:N) RETURN a.eid AS src, b.eid AS tgt").to_list()
        assert rows == [{"src": 1, "tgt": 1}]

    def test_zero_length_var_path_includes_anchor(self):
        """`[:R*0..N]` matches the anchor node itself at length 0."""
        g = KnowledgeGraph()
        g.cypher("CREATE (a:N {eid: 1}), (b:N {eid: 2}), (c:N {eid: 3})")
        g.cypher("MATCH (a:N {eid:1}), (b:N {eid:2}) CREATE (a)-[:R]->(b)")
        g.cypher("MATCH (b:N {eid:2}), (c:N {eid:3}) CREATE (b)-[:R]->(c)")
        rows = g.cypher("MATCH (a:N {eid:1})-[:R*0..2]->(b:N) RETURN b.eid AS r ORDER BY r").to_list()
        # Distance 0: a itself; distance 1: b; distance 2: c.
        assert [r["r"] for r in rows] == [1, 2, 3]

    def test_zero_length_only_var_path_returns_anchor(self):
        """`[:R*0..0]` matches only the anchor (no hops)."""
        g = KnowledgeGraph()
        g.cypher("CREATE (a:N {eid: 1}), (b:N {eid: 2})")
        g.cypher("MATCH (a:N {eid:1}), (b:N {eid:2}) CREATE (a)-[:R]->(b)")
        rows = g.cypher("MATCH (a:N {eid:1})-[:R*0..0]->(b:N) RETURN b.eid AS r").to_list()
        assert rows == [{"r": 1}]

    def test_parallel_edges_count_correctly(self):
        """N parallel edges of the same type between the same pair must
        all surface in `MATCH ()-[r:T]->()` — `count(r)` returns N, not 1."""
        g = KnowledgeGraph()
        g.cypher("CREATE (a:N {eid: 1}), (b:N {eid: 2})")
        for _ in range(3):
            g.cypher("MATCH (a:N {eid:1}), (b:N {eid:2}) CREATE (a)-[:DUP]->(b)")
        rows = g.cypher("MATCH (a:N {eid:1})-[r:DUP]->(b:N {eid:2}) RETURN count(r) AS n").to_list()
        assert rows == [{"n": 3}]


class TestBugOrderByIntToFloat:
    """BUG: ORDER BY + LIMIT on aggregated columns converts int to float."""

    def test_count_stays_int_with_order_by(self, cypher_graph):
        rows = cypher_graph.cypher("""
            MATCH (a:Person)-[:KNOWS]->(b:Person)
            WITH a, count(b) AS friends
            RETURN a.title, friends
            ORDER BY friends DESC LIMIT 3
        """)
        # friends must be int, not float
        assert rows[0]["friends"] == 2
        assert isinstance(rows[0]["friends"], int)

    def test_count_star_through_with_stays_int(self, cypher_graph):
        rows = cypher_graph.cypher("""
            MATCH (n:Person)
            WITH n.city AS city, count(*) AS cnt
            RETURN city, cnt
            ORDER BY cnt DESC LIMIT 2
        """)
        assert isinstance(rows[0]["cnt"], int)

    def test_size_collect_stays_int_with_order_by(self, cypher_graph):
        rows = cypher_graph.cypher("""
            MATCH (a:Person)-[:KNOWS]->(b:Person)
            WITH a, collect(b.title) AS friends
            RETURN a.title, size(friends) AS num_friends
            ORDER BY num_friends DESC LIMIT 3
        """)
        assert isinstance(rows[0]["num_friends"], int)

    def test_sum_stays_int_with_order_by(self, cypher_graph):
        rows = cypher_graph.cypher("""
            MATCH (n:Person)
            WITH n.city AS city, sum(n.age) AS total_age
            RETURN city, total_age
            ORDER BY total_age DESC LIMIT 2
        """)
        # sum of ages should be int
        assert isinstance(rows[0]["total_age"], int)


class TestBugReturnStar:
    """BUG: RETURN * returns {'*': 1} instead of expanding bound variables."""

    def test_return_star_single_node(self, cypher_graph):
        rows = cypher_graph.cypher("""
            MATCH (n:Person) WHERE n.name = 'Alice' RETURN *
        """)
        assert len(rows) == 1
        row = rows[0]
        # Should have node data, not {'*': 1}
        assert "*" not in row
        assert "n" in row or "n.name" in row or "n.title" in row

    def test_return_star_with_relationship(self, cypher_graph):
        rows = cypher_graph.cypher("""
            MATCH (a:Person)-[r:KNOWS]->(b:Person)
            WHERE a.name = 'Alice'
            RETURN *
        """)
        assert len(rows) == 2
        row = rows[0]
        assert "*" not in row
        # Should contain info about a, r, b

    def test_return_star_after_with(self, cypher_graph):
        """RETURN * after WITH should return WITH-scoped variables."""
        rows = cypher_graph.cypher("""
            MATCH (n:Person)
            WITH n.name AS name, n.age AS age
            WHERE age > 30
            RETURN *
        """)
        # Charlie(35), Eve(40)
        assert len(rows) == 2
        assert "name" in rows[0]
        assert "age" in rows[0]


class TestBugMultiHopPath:
    """BUG: Path variable on explicit multi-hop only captures first hop."""

    def test_two_hop_path_length(self, cypher_graph):
        rows = cypher_graph.cypher("""
            MATCH p = (a:Person)-[:KNOWS]->(b:Person)-[:PURCHASED]->(pr:Product)
            WHERE a.name = 'Alice'
            RETURN length(p) AS hops
            LIMIT 1
        """)
        assert rows[0]["hops"] == 2  # two relationships

    def test_two_hop_path_nodes_count(self, cypher_graph):
        rows = cypher_graph.cypher("""
            MATCH p = (a:Person)-[:KNOWS]->(b:Person)-[:PURCHASED]->(pr:Product)
            WHERE a.name = 'Alice'
            RETURN [n IN nodes(p) | n.title] AS chain
            LIMIT 1
        """)
        assert len(rows[0]["chain"]) == 3  # a, b, pr

    def test_two_hop_path_relationships(self, cypher_graph):
        # Phase A.1 / C2 — relationships() returns Rel dicts; extract
        # `.type` for the legacy list-of-strings shape.
        rows = cypher_graph.cypher("""
            MATCH p = (a:Person)-[:KNOWS]->(b:Person)-[:PURCHASED]->(pr:Product)
            WHERE a.name = 'Alice'
            RETURN relationships(p) AS rels
            LIMIT 1
        """)
        rel_types = [r["type"] for r in rows[0]["rels"]]
        assert len(rel_types) == 2
        assert "KNOWS" in rel_types
        assert "PURCHASED" in rel_types

    def test_two_hop_path_intermediate_node(self, cypher_graph):
        """The intermediate node must appear in nodes(p)."""
        rows = cypher_graph.cypher("""
            MATCH p = (a:Person)-[:KNOWS]->(b:Person)-[:PURCHASED]->(pr:Product)
            WHERE a.name = 'Alice' AND b.name = 'Bob'
            RETURN [n IN nodes(p) | n.title] AS chain
        """)
        # Alice → Bob → Tablet
        assert len(rows) == 1
        chain = rows[0]["chain"]
        assert chain[0] == "Alice"
        assert chain[1] == "Bob"
        assert chain[2] == "Tablet"

    def test_three_hop_path(self, cypher_graph):
        rows = cypher_graph.cypher("""
            MATCH p = (a:Person)-[:KNOWS]->(b:Person)-[:KNOWS]->(c:Person)-[:PURCHASED]->(pr:Product)
            WHERE a.name = 'Alice'
            RETURN length(p) AS hops, [n IN nodes(p) | n.title] AS chain
            LIMIT 1
        """)
        assert rows[0]["hops"] == 3
        assert len(rows[0]["chain"]) == 4


class TestBugUnlabeledMatchTypeFilter:
    """BUG: MATCH (n) WHERE n.type = 'X' returns 0 even though nodes exist."""

    def test_unlabeled_type_equality(self, cypher_graph):
        rows = cypher_graph.cypher("""
            MATCH (n) WHERE n.type = 'Person' RETURN count(n) AS cnt
        """)
        assert rows[0]["cnt"] == 5

    def test_unlabeled_type_equality_product(self, cypher_graph):
        rows = cypher_graph.cypher("""
            MATCH (n) WHERE n.type = 'Product' RETURN count(n) AS cnt
        """)
        assert rows[0]["cnt"] == 3

    def test_unlabeled_type_equality_returns_data(self, cypher_graph):
        rows = cypher_graph.cypher("""
            MATCH (n) WHERE n.type = 'Person' RETURN n.title ORDER BY n.title LIMIT 2
        """)
        assert len(rows) == 2


class TestBugLabelsInconsistency:
    """BUG: labels(n) returns list in RETURN but string in GROUP BY."""

    def test_labels_always_returns_list(self, cypher_graph):
        """GROUP BY context should return list, same as plain RETURN."""
        rows = cypher_graph.cypher("""
            MATCH (n) RETURN labels(n) AS lbl, count(n) AS cnt
            ORDER BY cnt DESC
        """)
        # In plain RETURN, labels() → ['Person']. In GROUP BY, → 'Person'.
        # They should be consistent: always list.
        for row in rows:
            assert isinstance(row["lbl"], list), (
                f"labels() returned {type(row['lbl']).__name__} '{row['lbl']}', expected list"
            )

    def test_labels_filter_equality(self, cypher_graph):
        """Should be able to filter by labels(n) in WHERE.

        Phase A.1 / C2 — labels(n) returns native list ['Person'] now,
        so the WHERE comparison is against a list, not a string. The
        legacy `labels(n) = 'Person'` comparison was a string-vs-string
        comparison via the JSON-encoded list; that surface is gone.
        Use `labels(n)[0] = 'Person'` or membership instead.
        """
        rows = cypher_graph.cypher("""
            MATCH (n) WHERE labels(n)[0] = 'Person' RETURN count(n) AS cnt
        """)
        assert rows[0]["cnt"] == 5


def test_inline_map_value_accepts_node_property_expression():
    """An inline-map value may be a `var.prop` property-access expression,
    resolved at match time against a bound node OR a projected node value.
    Regression: kglite-docs 2026-05-29 #3 — `MATCH (b {id: first.id})`
    raised "Pattern parse error: Unexpected single '.'"."""
    g = KnowledgeGraph()
    g.add_nodes(
        pd.DataFrame([{"id": f"a{i}", "w": i} for i in range(3)]),
        "Assessment",
        "id",
        "id",
    )
    # Projected node value from collect()[0].
    rows = g.cypher(
        "MATCH (a:Assessment) WITH collect(a)[0] AS first "
        "MATCH (b:Assessment {id: first.id}) RETURN b.id AS id, b.w AS w"
    ).to_list()
    assert rows == [{"id": "a0", "w": 0}]
    # Correlated between two bound nodes.
    rows = g.cypher("MATCH (a:Assessment {id:'a1'}) MATCH (b:Assessment {id: a.id}) RETURN b.id AS id").to_list()
    assert rows == [{"id": "a1"}]


class TestMapSubscriptAccess:
    """BUG (0.10.14): `RETURN {x: 1}['x']` raised "Index must be an integer".

    Map subscript by string key is standard openCypher/Neo4j. The
    IndexAccess evaluator now dispatches on the index value's type:
    integer → list indexing (hot path, checked first); string → map /
    node / relationship key lookup; null → null. Missing keys resolve to
    null, never an error.
    """

    def test_map_literal_string_key(self, cypher_graph):
        assert cypher_graph.cypher("RETURN {x: 1}['x'] AS r")[0]["r"] == 1

    def test_properties_string_key(self, cypher_graph):
        rows = cypher_graph.cypher("MATCH (n:Person {name: 'Alice'}) RETURN properties(n)['city'] AS r")
        assert rows[0]["r"] == "Oslo"

    def test_dynamic_key_from_variable(self, cypher_graph):
        assert cypher_graph.cypher("WITH 'x' AS k RETURN {x: 1}[k] AS r")[0]["r"] == 1

    def test_missing_key_is_null(self, cypher_graph):
        assert cypher_graph.cypher("RETURN {x: 1}['nope'] AS r")[0]["r"] is None

    def test_null_key_is_null(self, cypher_graph):
        assert cypher_graph.cypher("RETURN {x: 1}[null] AS r")[0]["r"] is None

    def test_nested_map_subscript(self, cypher_graph):
        assert cypher_graph.cypher("RETURN {a: {b: 2}}['a']['b'] AS r")[0]["r"] == 2

    def test_node_dynamic_property_access(self, cypher_graph):
        """Neo4j supports dynamic property access on a node binding."""
        rows = cypher_graph.cypher("MATCH (n:Person {name: 'Alice'}) RETURN n['city'] AS r")
        assert rows[0]["r"] == "Oslo"

    def test_node_subscript_in_where(self, cypher_graph):
        rows = cypher_graph.cypher("MATCH (n:Person) WHERE n['city'] = 'Bergen' RETURN count(n) AS cnt")
        assert rows[0]["cnt"] == 2

    def test_list_indexing_still_works(self, cypher_graph):
        assert cypher_graph.cypher("RETURN [1, 2, 3][0] AS r")[0]["r"] == 1
        assert cypher_graph.cypher("RETURN [1, 2, 3][-1] AS r")[0]["r"] == 3
        assert cypher_graph.cypher("RETURN [1, 2, 3][9] AS r")[0]["r"] is None


class TestFusedCountRegressions:
    """Exact-value regressions for three fused/hinted count bugs (0.12.x).

    A. push_distinct_into_match: the executor's distinct-dedup branch
       skipped the residual (non-pushable) WHERE fused into the MATCH.
    B. try_count_simple_pattern slow path: missing connection-type
       post-filter on memory/mapped storage counted edges of OTHER
       connection types.
    C. fused OPTIONAL MATCH aggregate: count(*) returned 0 for unmatched
       upstream rows instead of 1 (the null-padded row counts).

    The matching differential-corpus entries assert optimized == naive;
    these tests pin the actual correct values.
    """

    def test_distinct_hint_respects_residual_where(self):
        # Pair sums: A->B = 30, A->C = 40, C->B = 50, D->B = 65.
        # Only D->B passes `a.age + b.age > 50`.
        g = KnowledgeGraph()
        g.add_nodes(
            pd.DataFrame({"pid": [1, 2, 3, 4], "name": ["A", "B", "C", "D"], "age": [10, 20, 30, 45]}),
            "Person",
            "pid",
            "name",
        )
        g.add_connections(pd.DataFrame({"f": [1, 1, 3, 4], "t": [2, 3, 2, 2]}), "KNOWS", "Person", "f", "Person", "t")
        q = "MATCH (a:Person)-[:KNOWS]->(b:Person) WHERE a.age + b.age > 50 RETURN DISTINCT b.name AS n"
        assert sorted(r["n"] for r in g.cypher(q).to_list()) == ["B"]
        # The pass disabled must agree (guards the executor-side fix).
        assert sorted(r["n"] for r in g.cypher(q, disabled_passes=["push_distinct_into_match"]).to_list()) == ["B"]

    def test_fused_optional_count_filters_connection_type(self):
        # a has BOTH an R and an S edge to the same peer (which passes the
        # property filter) — only the R edge may be counted.
        g = KnowledgeGraph()
        g.add_nodes(pd.DataFrame({"pid": [1], "name": ["a"]}), "P", "pid", "name")
        g.add_nodes(pd.DataFrame({"pid": [10], "name": ["b"], "y": [2]}), "Q", "pid", "name")
        g.add_connections(pd.DataFrame({"f": [1], "t": [10]}), "R", "P", "f", "Q", "t")
        g.add_connections(pd.DataFrame({"f": [1], "t": [10]}), "S", "P", "f", "Q", "t")
        q = "MATCH (a:P) OPTIONAL MATCH (a)-[:R]->(b {y: 2}) WITH a, count(b) AS c RETURN a.name AS n, c"
        assert g.cypher(q).to_list() == [{"n": "a", "c": 1}]
        assert g.cypher(q, disabled_passes=["fuse_optional_match_aggregate"]).to_list() == [{"n": "a", "c": 1}]

    def test_untyped_edge_count_includes_parallel_edges_and_self_loops(self):
        g = KnowledgeGraph()
        g.add_nodes(pd.DataFrame({"id": [1, 2, 3], "name": ["a", "b", "c"]}), "N", "id", "name")
        g.add_connections(
            pd.DataFrame({"source": [1, 1, 2], "target": [2, 2, 2]}),
            "R",
            "N",
            "source",
            "N",
            "target",
        )
        g.add_connections(
            pd.DataFrame({"source": [2], "target": [3]}),
            "S",
            "N",
            "source",
            "N",
            "target",
        )
        query = "MATCH ()-[r]->() RETURN count(r) AS n"
        assert g.cypher(query).to_list() == [{"n": 4}]
        assert g.cypher(query, disabled_passes=["fuse_count_short_circuits"]).to_list() == [{"n": 4}]
        assert KnowledgeGraph().cypher(query).to_list() == [{"n": 0}]

    def test_fused_optional_count_star_null_padded_row(self):
        # y has no outgoing R edge: OPTIONAL MATCH emits one null-padded
        # row, so count(*) = 1 while count(m) = 0.
        g = KnowledgeGraph()
        g.add_nodes(pd.DataFrame({"pid": [1, 2], "name": ["x", "y"]}), "P", "pid", "name")
        g.add_connections(pd.DataFrame({"f": [1], "t": [2]}), "R", "P", "f", "P", "t")
        q = (
            "MATCH (n:P) OPTIONAL MATCH (n)-[r:R]->(m) "
            "WITH n, count(*) AS c, count(m) AS cm, count(*) - count(m) AS diff "
            "RETURN n.name AS name, c, cm, diff ORDER BY name"
        )
        expected = [
            {"name": "x", "c": 1, "cm": 1, "diff": 0},
            {"name": "y", "c": 1, "cm": 0, "diff": 1},
        ]
        assert g.cypher(q).to_list() == expected
        assert g.cypher(q, disabled_passes=["fuse_optional_match_aggregate"]).to_list() == expected

    def test_fused_optional_multi_pattern_not_fused(self):
        # The fused operator computes ONE match_count summed across
        # patterns; per-variable counts over a multi-pattern OPTIONAL
        # MATCH can't be derived from it, so the fusion gate must bail.
        # Optimized and naive must agree (exact multi-pattern OPTIONAL
        # semantics are the materialized executor's contract).
        g = KnowledgeGraph()
        g.add_nodes(pd.DataFrame({"pid": [1, 2, 3, 4], "name": ["n1", "n2", "n3", "n4"]}), "P", "pid", "name")
        g.add_connections(pd.DataFrame({"f": [1, 1], "t": [2, 3]}), "X", "P", "f", "P", "t")
        g.add_connections(pd.DataFrame({"f": [2], "t": [4]}), "Y", "P", "f", "P", "t")
        q = (
            "MATCH (n:P {pid: 1}) OPTIONAL MATCH (n)-[:X]->(a), (n)-[:Y]->(b) "
            "WITH n, count(a) AS ca, count(b) AS cb RETURN n.name AS name, ca, cb"
        )
        # openCypher join-then-null-pad: the comma patterns succeed or fail
        # as ONE unit. Pattern (n)-[:Y]->(b) has no match from pid 1, so the
        # joined expansion is empty and BOTH variables null-pad — ca=0, cb=0
        # (not the quasi-independent ca=2 the old per-pattern union gave).
        expected = [{"name": "n1", "ca": 0, "cb": 0}]
        assert g.cypher(q).to_list() == expected
        assert g.cypher(q, disabled_passes=["fuse_optional_match_aggregate"]).to_list() == expected


class TestRelationshipIdentityContract:
    """openCypher: a relationship variable already bound on the row pins a
    later pattern that re-uses it to exactly that edge (0.12.x fix — the
    re-MATCH previously enumerated ALL edges and overwrote the binding)."""

    @staticmethod
    def _two_edge_graph():
        g = KnowledgeGraph()
        g.add_nodes(pd.DataFrame({"pid": [1, 2, 3, 4], "name": ["a", "b", "c", "d"]}), "P", "pid", "name")
        g.add_connections(pd.DataFrame({"f": [1, 3], "t": [2, 4], "w": [1, 2]}), "KNOWS", "P", "f", "P", "t")
        # Second connection type so type-mismatch shapes don't trip the
        # unknown-label schema warning.
        g.add_connections(pd.DataFrame({"f": [1], "t": [2]}), "LIKES", "P", "f", "P", "t")
        return g

    def test_with_rebind_pins_edge(self):
        g = self._two_edge_graph()
        rows = g.cypher(
            "MATCH ()-[r:KNOWS]->() WITH r, r.w AS orig_w MATCH (a)-[r]->(b) RETURN orig_w, r.w AS w ORDER BY orig_w"
        ).to_list()
        assert rows == [{"orig_w": 1, "w": 1}, {"orig_w": 2, "w": 2}]

    def test_unwind_rebind_pins_edge(self):
        g = self._two_edge_graph()
        rows = g.cypher(
            "MATCH ()-[r0:KNOWS]->() WITH collect(r0) AS rels "
            "UNWIND rels AS r MATCH (a)-[r]->(b) "
            "RETURN a.name AS an, b.name AS bn ORDER BY an"
        ).to_list()
        assert rows == [{"an": "a", "bn": "b"}, {"an": "c", "bn": "d"}]

    def test_rebind_binds_endpoints_from_stored_edge(self):
        g = self._two_edge_graph()
        rows = g.cypher(
            "MATCH ()-[r:KNOWS {w: 2}]->() WITH r MATCH (a)-[r]->(b) RETURN a.name AS an, b.name AS bn"
        ).to_list()
        assert rows == [{"an": "c", "bn": "d"}]

    def test_rebind_type_mismatch_yields_no_rows(self):
        g = self._two_edge_graph()
        rows = g.cypher("MATCH ()-[r:KNOWS {w: 2}]->() WITH r MATCH (a)-[r:LIKES]->(b) RETURN count(*) AS n").to_list()
        assert rows == [{"n": 0}]

    def test_rebind_direction_flip(self):
        # The bound edge must satisfy the new pattern's direction: with the
        # pattern reversed, `x` binds the edge target and `y` its source.
        g = self._two_edge_graph()
        rows = g.cypher(
            "MATCH ()-[r:KNOWS {w: 1}]->() WITH r MATCH (x)<-[r]-(y) RETURN x.name AS xn, y.name AS yn"
        ).to_list()
        assert rows == [{"xn": "b", "yn": "a"}]

    def test_optional_rebind_incompatible_null_pads(self):
        g = self._two_edge_graph()
        rows = g.cypher(
            "MATCH ()-[r:KNOWS {w: 2}]->() WITH r OPTIONAL MATCH (a)-[r:LIKES]->(b) RETURN a.name AS an, b.name AS bn"
        ).to_list()
        assert rows == [{"an": None, "bn": None}]

    def test_null_rel_var_matches_nothing(self):
        # An OPTIONAL MATCH miss leaves `r` null; a later MATCH re-using it
        # yields no rows (a null relationship can't match a pattern).
        g = self._two_edge_graph()
        rows = g.cypher(
            "MATCH (n:P {pid: 2}) OPTIONAL MATCH (n)-[r:KNOWS]->(m) WITH r MATCH (a)-[r]->(b) RETURN count(*) AS n"
        ).to_list()
        assert rows == [{"n": 0}]


class TestRelationshipTrailRule:
    """openCypher relationship uniqueness: within ONE MATCH clause, two
    different pattern edges (named or anonymous, across comma patterns)
    may not bind the same relationship. Separate MATCH clauses may."""

    @staticmethod
    def _one_edge_graph():
        g = KnowledgeGraph()
        g.add_nodes(pd.DataFrame({"pid": [1, 2], "name": ["a", "b"]}), "P", "pid", "name")
        g.add_connections(pd.DataFrame({"f": [1], "t": [2]}), "R", "P", "f", "P", "t")
        return g

    def test_named_comma_patterns_cannot_share_edge(self):
        g = self._one_edge_graph()
        q = "MATCH (a)-[r1:R]->(b), (c)-[r2:R]->(d) RETURN count(*) AS n"
        assert g.cypher(q).to_list() == [{"n": 0}]

    def test_anonymous_comma_patterns_cannot_share_edge(self):
        g = self._one_edge_graph()
        q = "MATCH (a)-[:R]->(b), (c)-[:R]->(d) RETURN count(*) AS n"
        assert g.cypher(q).to_list() == [{"n": 0}]

    def test_var_length_and_fixed_comma_patterns_cannot_share_edge(self):
        g = self._one_edge_graph()
        q = "MATCH (a)-[r1:R]->(b), (c)-[r2:R*1..1]->(d) RETURN count(*) AS n"
        assert g.cypher(q).to_list() == [{"n": 0}]

    def test_separate_match_clauses_may_share_edge(self):
        g = self._one_edge_graph()
        q = "MATCH (a)-[r1:R]->(b) MATCH (c)-[r2:R]->(d) RETURN count(*) AS n"
        assert g.cypher(q).to_list() == [{"n": 1}]

    def test_three_anonymous_patterns_injective_assignment(self):
        g = KnowledgeGraph()
        g.add_nodes(pd.DataFrame({"pid": [1, 2, 3], "name": ["a", "b", "c"]}), "P", "pid", "name")
        g.add_connections(pd.DataFrame({"f": [1, 2, 3], "t": [2, 3, 1]}), "R", "P", "f", "P", "t")
        # 3 edges into 3 anonymous slots: 3! injective assignments.
        q = "MATCH ()-[:R]->(), ()-[:R]->(), ()-[:R]->() RETURN count(*) AS n"
        assert g.cypher(q).to_list() == [{"n": 6}]

    def test_uniqueness_applies_on_joined_subsequent_match(self):
        # Non-first MATCH clause (upstream WITH) takes the join path —
        # the trail rule must hold there too.
        g = self._one_edge_graph()
        q = "MATCH (p:P {pid: 1}) WITH p MATCH (p)-[r1:R]->(b), (c)-[r2:R]->(d) RETURN count(*) AS n"
        assert g.cypher(q).to_list() == [{"n": 0}]

    def test_empty_comma_pattern_empties_clause(self):
        # Regression: an earlier pattern with no matches let the next
        # pattern re-enter the "first pattern" branch and fabricate rows.
        g = self._one_edge_graph()
        q = "MATCH (x:P {pid: 999}), (a)-[:R]->(b) RETURN count(*) AS n"
        assert g.cypher(q).to_list() == [{"n": 0}]
        q2 = "MATCH (a)-[:R]->(b), (x:P {pid: 999}) RETURN count(*) AS n"
        assert g.cypher(q2).to_list() == [{"n": 0}]


class TestOptionalMatchJoinSemantics:
    """Multi-pattern OPTIONAL MATCH is join-then-null-pad (openCypher): the
    comma patterns expand as one joined unit; only when the WHOLE join is
    empty does the single null-padded row appear. (0.12.x fix — patterns
    previously expanded independently, unioning per-pattern half-rows.)"""

    @staticmethod
    def _graph():
        g = KnowledgeGraph()
        g.add_nodes(
            pd.DataFrame({"pid": [1, 2, 3, 4, 5], "name": ["n1", "n2", "n3", "n4", "n5"]}),
            "P",
            "pid",
            "name",
        )
        g.add_connections(pd.DataFrame({"f": [1, 1], "t": [2, 3]}), "X", "P", "f", "P", "t")
        g.add_connections(pd.DataFrame({"f": [1, 1], "t": [4, 5]}), "Y", "P", "f", "P", "t")
        return g

    def test_one_empty_pattern_null_pads_all_vars(self):
        g = self._graph()
        # pid 2 has no X and no Y edges outgoing? n2 has none of either —
        # use pid 2 for the all-empty case and pid 1 with a Z-less pattern
        # for the half-empty case below.
        rows = g.cypher(
            "MATCH (n:P {pid: 2}) OPTIONAL MATCH (n)-[:X]->(a), (n)-[:Y]->(b) RETURN a.name AS an, b.name AS bn"
        ).to_list()
        assert rows == [{"an": None, "bn": None}]

    def test_half_empty_join_null_pads_all_vars(self):
        # X matches from pid 1 twice, Y from pid 1... both match; instead
        # anchor a pattern that cannot match: (n)-[:X]->(a), (a)-[:X]->(b).
        g = self._graph()
        rows = g.cypher(
            "MATCH (n:P {pid: 1}) OPTIONAL MATCH (n)-[:X]->(a), (a)-[:X]->(b) RETURN a.name AS an, b.name AS bn"
        ).to_list()
        # No X edge leaves n2/n3, so the join is empty — `a` must be null
        # too, not a half-row per X edge.
        assert rows == [{"an": None, "bn": None}]

    def test_both_match_cross_join(self):
        g = self._graph()
        rows = g.cypher(
            "MATCH (n:P {pid: 1}) OPTIONAL MATCH (n)-[:X]->(a), (n)-[:Y]->(b) "
            "RETURN a.name AS an, b.name AS bn ORDER BY an, bn"
        ).to_list()
        assert rows == [
            {"an": "n2", "bn": "n4"},
            {"an": "n2", "bn": "n5"},
            {"an": "n3", "bn": "n4"},
            {"an": "n3", "bn": "n5"},
        ]

    def test_counts_over_joined_expansion(self):
        g = self._graph()
        rows = g.cypher(
            "MATCH (n:P {pid: 1}) OPTIONAL MATCH (n)-[:X]->(a), (n)-[:Y]->(b) "
            "WITH n, count(a) AS ca, count(b) AS cb, count(*) AS c "
            "RETURN ca, cb, c"
        ).to_list()
        assert rows == [{"ca": 4, "cb": 4, "c": 4}]

    def test_count_star_on_failed_join_is_one(self):
        g = self._graph()
        rows = g.cypher(
            "MATCH (n:P {pid: 2}) OPTIONAL MATCH (n)-[:X]->(a), (n)-[:Y]->(b) "
            "WITH n, count(a) AS ca, count(*) AS c RETURN ca, c"
        ).to_list()
        assert rows == [{"ca": 0, "c": 1}]


class TestAggregateGroupKeyErrors:
    """Group-key evaluation errors must propagate through aggregation the
    same way they do without it (0.12.x fix — they were swallowed into
    NULL groups). Legitimate null groups (OPTIONAL miss, property access
    on null) still group as NULL."""

    @staticmethod
    def _graph():
        g = KnowledgeGraph()
        g.add_nodes(pd.DataFrame({"pid": [1, 2], "name": ["a", "b"]}), "P", "pid", "name")
        g.add_connections(pd.DataFrame({"f": [1], "t": [2]}), "R", "P", "f", "P", "t")
        return g

    def test_missing_parameter_group_key_errors(self):
        g = self._graph()
        q = "MATCH (n:P) RETURN $missing AS k, count(*) AS c"
        with pytest.raises(Exception, match="Missing parameter"):
            g.cypher(q).to_list()
        # Materialized aggregation path (fused node-scan disabled) agrees.
        with pytest.raises(Exception, match="Missing parameter"):
            g.cypher(q, disable_optimizer=True).to_list()

    def test_null_valued_group_keys_still_group(self):
        g = self._graph()
        rows = g.cypher("MATCH (n:P {pid: 2}) OPTIONAL MATCH (n)-[:R]->(m) RETURN m.name AS k, count(*) AS c").to_list()
        assert rows == [{"k": None, "c": 1}]
