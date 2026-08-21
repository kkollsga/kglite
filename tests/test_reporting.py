"""Tests for explain, operation reports, report history."""

import pandas as pd

from kglite import KnowledgeGraph


class TestExplain:
    def test_explain_select(self, social_graph):
        result = social_graph.select("Person")
        explanation = result.explain()
        assert isinstance(explanation, str)
        assert "Person" in explanation

    def test_explain_chained(self, social_graph):
        result = social_graph.select("Person").where({"city": "Oslo"})
        explanation = result.explain()
        assert "SELECT" in explanation.upper()
        assert "WHERE" in explanation.upper()

    def test_explain_shows_counts(self, social_graph):
        result = social_graph.select("Person").where({"city": "Oslo"})
        explanation = result.explain()
        # Should show node counts at each step
        assert any(c.isdigit() for c in explanation)

    def test_explain_traversal(self, petroleum_graph):
        result = petroleum_graph.select("Play").traverse("HAS_PROSPECT")
        explanation = result.explain()
        assert "TRAVERSE" in explanation.upper() or "traverse" in explanation.lower()

    def test_explain_empty(self, social_graph):
        result = social_graph.select("NonExistent")
        explanation = result.explain()
        assert isinstance(explanation, str)

    def test_explain_on_the_root_object_teaches_where_the_plan_lives(self, social_graph):
        """The plan lives on the object a fluent method returns, so calling
        explain() on the graph itself records nothing. The message has to say
        that, and point Cypher users at the prefixes — an external evaluation
        read the old bare 'No query operations recorded' as evidence that the
        method was vestigial."""
        message = social_graph.explain()
        assert message == (
            "No query operations recorded — explain() reports the fluent chain it is called on: "
            "graph.select(...).where(...).explain(). For Cypher, prefix the query with EXPLAIN or PROFILE."
        )

    def test_explain_docstring_points_at_the_chain_result_and_cypher_prefixes(self, social_graph):
        doc = type(social_graph).explain.__doc__
        assert "chain" in doc
        assert "EXPLAIN" in doc and "PROFILE" in doc
        assert "SELECT" in doc
        # The stale operator name the docstring advertised for several versions.
        assert "TYPE_FILTER" not in doc


class TestOperationReports:
    def test_add_nodes_report(self):
        graph = KnowledgeGraph()
        df = pd.DataFrame({"id": [1, 2], "name": ["A", "B"]})
        report = graph.add_nodes(df, "Node", "id", "name")
        assert "nodes_created" in report
        assert report["nodes_created"] == 2
        assert "processing_time_ms" in report

    def test_add_connections_report(self, small_graph):
        report = small_graph.last_report()
        assert report is not None

    def test_report_has_errors_field(self):
        graph = KnowledgeGraph()
        df = pd.DataFrame({"id": [1], "name": ["A"]})
        report = graph.add_nodes(df, "Node", "id", "name")
        assert "has_errors" in report

    def test_operation_index(self):
        graph = KnowledgeGraph()
        df = pd.DataFrame({"id": [1], "name": ["A"]})
        graph.add_nodes(df, "Node", "id", "name")
        idx = graph.operation_index()
        assert idx >= 1

    def test_report_history(self):
        graph = KnowledgeGraph()
        df1 = pd.DataFrame({"id": [1], "name": ["A"]})
        df2 = pd.DataFrame({"id": [10], "name": ["B"]})
        graph.add_nodes(df1, "TypeA", "id", "name")
        graph.add_nodes(df2, "TypeB", "id", "name")
        history = graph.report_history()
        assert len(history) >= 2
