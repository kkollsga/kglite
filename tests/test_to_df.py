"""Tests for to_df() and cypher(to_df=True) DataFrame export."""

import pandas as pd

import kglite


class TestToDF:
    """Tests for KnowledgeGraph.to_df()."""

    def test_basic_to_df(self, small_graph):
        df = small_graph.select("Person").to_df()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert "title" in df.columns
        assert "type" in df.columns
        assert "id" in df.columns
        assert "age" in df.columns
        assert "city" in df.columns
        assert set(df["title"]) == {"Alice", "Bob", "Charlie"}

    def test_filtered_to_df(self, small_graph):
        df = small_graph.select("Person").where({"city": "Oslo"}).to_df()
        assert len(df) == 2
        assert set(df["title"]) == {"Alice", "Charlie"}
        assert all(df["city"] == "Oslo")

    def test_include_type_false(self, small_graph):
        df = small_graph.select("Person").to_df(include_type=False)
        assert "type" not in df.columns
        assert "title" in df.columns
        assert "id" in df.columns

    def test_include_id_false(self, small_graph):
        df = small_graph.select("Person").to_df(include_id=False)
        assert "id" not in df.columns
        assert "title" in df.columns
        assert "type" in df.columns

    def test_include_both_false(self, small_graph):
        df = small_graph.select("Person").to_df(include_type=False, include_id=False)
        assert "type" not in df.columns
        assert "id" not in df.columns
        assert "title" in df.columns
        assert "age" in df.columns

    def test_empty_selection(self, small_graph):
        df = small_graph.select("Person").where({"city": "Nonexistent"}).to_df()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_empty_graph(self, empty_graph):
        df = empty_graph.to_df()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_traversal_to_df(self, small_graph):
        df = small_graph.select("Person").where({"title": "Alice"}).traverse("KNOWS").to_df()
        assert isinstance(df, pd.DataFrame)
        assert len(df) >= 1  # Alice knows Bob and Charlie

    def test_null_handling(self, social_graph):
        """Nodes with missing properties should have None in the DataFrame."""
        df = social_graph.select("Person").to_df()
        assert "email" in df.columns
        # Odd-numbered persons have None email
        null_count = df["email"].isna().sum()
        assert null_count > 0

    def test_multi_type_to_df(self, social_graph):
        """Nodes of different types should have None for missing properties."""
        # Select both Person and Company nodes
        g = social_graph
        persons = g.select("Person")
        companies = g.select("Company")
        combined = persons.union(companies)
        df = combined.to_df()
        assert len(df) == 25  # 20 persons + 5 companies
        # Company nodes shouldn't have 'age'; Person nodes shouldn't have 'industry'
        assert "age" in df.columns
        assert "industry" in df.columns
        # Check that companies have NaN for age
        company_rows = df[df["type"] == "Company"]
        assert company_rows["age"].isna().all()
        # Check that persons have NaN for industry
        person_rows = df[df["type"] == "Person"]
        assert person_rows["industry"].isna().all()

    def test_column_order(self, small_graph):
        """type, title, id should always come first."""
        df = small_graph.select("Person").to_df()
        cols = list(df.columns)
        assert cols[0] == "type"
        assert cols[1] == "title"
        assert cols[2] == "id"

    def test_values_correct(self, small_graph):
        """Verify actual data values are correct."""
        df = small_graph.select("Person").to_df()
        alice = df[df["title"] == "Alice"].iloc[0]
        assert alice["age"] == 28
        assert alice["city"] == "Oslo"
        assert alice["type"] == "Person"


class TestCypherToDF:
    """Tests for cypher(to_df=True)."""

    def test_cypher_to_df(self, small_graph):
        df = small_graph.cypher(
            "MATCH (n:Person) RETURN n.name AS name, n.age AS age ORDER BY n.age",
            to_df=True,
        )
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["name", "age"]
        assert len(df) == 3
        assert df.iloc[0]["name"] == "Alice"
        assert df.iloc[0]["age"] == 28

    def test_cypher_default_returns_result_view(self, small_graph):
        result = small_graph.cypher("MATCH (n:Person) RETURN n.name AS name, n.age AS age")
        assert isinstance(result, kglite.ResultView)
        assert len(result) == 3
        assert "name" in result[0]
        assert "age" in result[0]

    def test_to_dicts_is_alias_of_to_list(self, small_graph):
        result = small_graph.cypher("MATCH (n:Person) RETURN n.name AS name, n.age AS age ORDER BY n.age")
        rows = result.to_dicts()
        assert isinstance(rows, list)
        assert all(isinstance(r, dict) for r in rows)
        # Identical behaviour to to_list() — the polars/pandas-friendly name.
        assert rows == result.to_list()
        assert rows[0]["name"] == "Alice"

    def test_cypher_to_df_with_aggregation(self, small_graph):
        df = small_graph.cypher(
            "MATCH (n:Person) RETURN n.city AS city, count(*) AS cnt ORDER BY cnt DESC",
            to_df=True,
        )
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["city", "cnt"]
        oslo_row = df[df["city"] == "Oslo"].iloc[0]
        assert oslo_row["cnt"] == 2

    def test_cypher_to_df_empty_result(self, small_graph):
        df = small_graph.cypher(
            "MATCH (n:Person) WHERE n.age > 1000 RETURN n.name",
            to_df=True,
        )
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_cypher_to_df_with_join(self, small_graph):
        df = small_graph.cypher(
            """MATCH (a:Person)-[:KNOWS]->(b:Person)
               RETURN a.name AS person, b.name AS friend""",
            to_df=True,
        )
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3  # Alice->Bob, Bob->Charlie, Alice->Charlie
        assert set(df.columns) == {"person", "friend"}


class TestNumericColumnTransport:
    """A column whose every cell shares one unboxed numeric layout crosses the
    boundary as raw bytes rather than as boxed Python scalars. The frame must be
    indistinguishable from the boxed one — same dtype, same exact values, and
    still writable, which a read-only buffer view would refuse."""

    def test_homogeneous_columns_keep_their_boxed_dtypes(self, small_graph):
        df = small_graph.cypher(
            "MATCH (n:Person) RETURN n.age AS age, n.age * 1.5 AS scaled, n.age > 0 AS grown",
            to_df=True,
        )
        assert str(df.age.dtype) == "int64"
        assert str(df.scaled.dtype) == "float64"
        assert str(df.grown.dtype) == "bool"
        assert df.grown.tolist() == [True, True, True]

    def test_int64_extremes_survive_the_byte_transport(self):
        graph = kglite.KnowledgeGraph()
        extremes = [-(2**63), 2**63 - 1, 9007199254740993, 0]
        df = graph.cypher("UNWIND $v AS x RETURN x", params={"v": extremes}, to_df=True)
        assert str(df.x.dtype) == "int64"
        assert df.x.tolist() == extremes

    def test_transported_column_is_writable_and_owns_its_cells(self):
        graph = kglite.KnowledgeGraph()
        df = graph.cypher("UNWIND [1, 2, 3] AS x RETURN x", to_df=True)
        df.loc[0, "x"] = 99
        assert df.x.tolist() == [99, 2, 3]
        assert graph.cypher("UNWIND [1, 2, 3] AS x RETURN x", to_df=True).x.tolist() == [1, 2, 3]

    def test_one_null_or_one_other_type_keeps_the_boxed_policy(self):
        graph = kglite.KnowledgeGraph()
        nullable = graph.cypher("UNWIND [1, null, 3] AS x RETURN x", to_df=True)
        assert str(nullable.x.dtype) == "Int64"
        assert nullable.x.tolist()[0] == 1 and pd.isna(nullable.x.tolist()[1])
        mixed = graph.cypher("UNWIND [1, 1.5] AS x RETURN x", to_df=True)
        assert str(mixed.x.dtype) == "object"
        assert [type(v) for v in mixed.x.tolist()] == [int, float]
        boolean_mix = graph.cypher("UNWIND [true, 1] AS x RETURN x", to_df=True)
        assert str(boolean_mix.x.dtype) == "object"
        assert [type(v) for v in boolean_mix.x.tolist()] == [bool, int]
