"""Tests for node operations: add, retrieve, property mapping, counting."""

import pandas as pd
import pytest

import kglite
from kglite import KnowledgeGraph


class TestModuleAPI:
    def test_module_imports(self):
        assert hasattr(kglite, "KnowledgeGraph")
        assert hasattr(kglite, "load")

    def test_graph_creation(self):
        graph = KnowledgeGraph()
        assert graph is not None
        assert isinstance(graph.schema_text(), str)


class TestAddNodes:
    def test_add_nodes_basic(self):
        graph = KnowledgeGraph()
        df = pd.DataFrame({"id": [1, 2, 3], "name": ["A", "B", "C"], "value": [10, 20, 30]})
        report = graph.add_nodes(df, "TestNode", "id", "name")
        assert report["nodes_created"] == 3

    def test_add_nodes_empty_dataframe(self):
        graph = KnowledgeGraph()
        df = pd.DataFrame({"id": [], "name": []})
        report = graph.add_nodes(df, "EmptyType", "id", "name")
        assert report["nodes_created"] == 0

    def test_add_nodes_with_columns(self):
        graph = KnowledgeGraph()
        df = pd.DataFrame({"id": [1], "name": ["A"], "keep": ["yes"], "drop": ["no"]})
        graph.add_nodes(df, "Node", "id", "name", columns=["id", "name", "keep"])
        node = graph.select("Node").collect()[0]
        assert "keep" in node
        assert "drop" not in node

    def test_add_nodes_conflict_update(self):
        graph = KnowledgeGraph()
        df1 = pd.DataFrame({"id": [1], "name": ["A"], "v": [10]})
        df2 = pd.DataFrame({"id": [1], "name": ["A"], "v": [20]})
        graph.add_nodes(df1, "Node", "id", "name")
        graph.add_nodes(df2, "Node", "id", "name", conflict_handling="update")
        node = graph.select("Node").collect()[0]
        assert node["v"] == 20

    def test_add_nodes_conflict_skip(self):
        graph = KnowledgeGraph()
        df1 = pd.DataFrame({"id": [1], "name": ["A"], "v": [10]})
        df2 = pd.DataFrame({"id": [1], "name": ["A"], "v": [20]})
        graph.add_nodes(df1, "Node", "id", "name")
        graph.add_nodes(df2, "Node", "id", "name", conflict_handling="skip")
        node = graph.select("Node").collect()[0]
        assert node["v"] == 10

    def test_add_nodes_null_values(self):
        graph = KnowledgeGraph()
        df = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "name": ["A", "B", "C"],
                "optional": ["value", None, "other"],
            }
        )
        graph.add_nodes(df, "Node", "id", "name")
        nodes = graph.select("Node").collect()
        assert len(nodes) == 3


class TestPropertyMapping:
    def test_id_field_renamed(self):
        graph = KnowledgeGraph()
        df = pd.DataFrame({"user_id": [1], "name": ["Alice"]})
        graph.add_nodes(df, "User", "user_id", "name")
        node = graph.select("User").collect()[0]
        assert node["id"] == 1
        assert "user_id" not in node

    def test_title_field_renamed(self):
        graph = KnowledgeGraph()
        df = pd.DataFrame({"id": [1], "full_name": ["Alice"]})
        graph.add_nodes(df, "User", "id", "full_name")
        node = graph.select("User").collect()[0]
        assert node["title"] == "Alice"
        assert "full_name" not in node

    def test_other_fields_preserved(self):
        graph = KnowledgeGraph()
        df = pd.DataFrame({"id": [1], "name": ["A"], "age": [30], "city": ["Oslo"]})
        graph.add_nodes(df, "User", "id", "name")
        node = graph.select("User").collect()[0]
        assert node["age"] == 30
        assert node["city"] == "Oslo"


class TestRetrieveNodes:
    def test_get_nodes(self, small_graph):
        nodes = small_graph.select("Person").collect()
        assert len(nodes) == 3
        titles = {n["title"] for n in nodes}
        assert titles == {"Alice", "Bob", "Charlie"}

    def test_node_count(self, small_graph):
        count = small_graph.select("Person").len()
        assert count == 3

    def test_titles(self, small_graph):
        # titles returns flat list when no traversal (single parent)
        titles = small_graph.select("Person").titles()
        assert isinstance(titles, list)
        assert set(titles) == {"Alice", "Bob", "Charlie"}

    def test_ids(self, small_graph):
        ids = small_graph.select("Person").ids()
        assert set(ids) == {1, 2, 3}

    def test_indices(self, small_graph):
        indices = small_graph.select("Person").indices()
        assert len(indices) == 3

    def test_get_properties(self, small_graph):
        # get_properties returns flat list when no traversal (single parent)
        props = small_graph.select("Person").get_properties(["age", "city"])
        assert isinstance(props, list)
        assert len(props) == 3
        for row in props:
            assert len(row) == 2  # (age, city)

    def test_node(self, small_graph):
        node = small_graph.node("Person", 1)
        assert node is not None
        assert node["title"] == "Alice"
        assert node["age"] == 28

    def test_node_not_found(self, small_graph):
        node = small_graph.node("Person", 999)
        assert node is None

    def test_type_filter_nonexistent(self, small_graph):
        result = small_graph.select("NonExistent")
        assert result.len() == 0


class TestWideIntegerIds:
    """An integer id column is auto-detected as a key type. It used to be the
    compact 32-bit one unconditionally, so every id below 0 or from ``2**32``
    up parsed to nothing and its row was dropped with only a warning — a short
    load on exactly the id shapes (snowflake ids, hashes, negative sentinels) a
    caller cannot have invented."""

    WIDE = [-1, 0, 2**31, 2**32 - 1, 2**32, 2**63 - 1]

    def test_ids_outside_u32_load_by_default(self, recwarn):
        graph = KnowledgeGraph()
        df = pd.DataFrame({"id": self.WIDE, "name": list("abcdef")})
        report = graph.add_nodes(df, "T", "id", "name")
        assert report["nodes_created"] == len(self.WIDE)
        assert report["nodes_skipped"] == 0
        assert report["has_errors"] is False
        assert [str(w.message) for w in recwarn] == []
        loaded = graph.cypher("MATCH (n:T) RETURN n.id AS i ORDER BY i").to_df()
        assert sorted(loaded["i"].tolist()) == sorted(self.WIDE)

    def test_wide_ids_are_matchable_and_survive_a_round_trip(self, tmp_path):
        graph = KnowledgeGraph()
        df = pd.DataFrame({"id": self.WIDE, "name": list("abcdef")})
        graph.add_nodes(df, "T", "id", "name")

        # Point lookup on the widest id, through Cypher and through node().
        hit = graph.cypher("MATCH (n:T) WHERE n.id = 9223372036854775807 RETURN n.name AS n")
        assert [r["n"] for r in hit] == ["f"]
        assert graph.node("T", -1)["title"] == "a"

        path = tmp_path / "wide.kgl"
        graph.save(str(path))
        reloaded = kglite.load(str(path))
        back = reloaded.cypher("MATCH (n:T) RETURN n.id AS i ORDER BY i").to_df()
        assert sorted(back["i"].tolist()) == sorted(self.WIDE)

    def test_ids_that_all_fit_u32_keep_the_compact_key(self):
        # The narrow column must not be widened just because the wide one now
        # can be: `n.id` stays an int either way, so the observable difference
        # is that a compact column still resolves by the id index.
        graph = KnowledgeGraph()
        df = pd.DataFrame({"id": [1, 2, 4294967295], "name": ["a", "b", "c"]})
        graph.add_nodes(df, "T", "id", "name")
        assert graph.node("T", 4294967295)["title"] == "c"
        assert graph.schema()["node_types"]["T"]["properties"]["id"] == "UniqueId"

        # ...and the wide column really does take the other storage, so the
        # narrow assertion above is not vacuous.
        wide = KnowledgeGraph()
        wide.add_nodes(pd.DataFrame({"id": [-1, 2**40], "name": ["a", "b"]}), "T", "id", "name")
        assert wide.schema()["node_types"]["T"]["properties"]["id"] == "Int64"


class TestOnInvalidNodes:
    def _mixed(self):
        return pd.DataFrame({"id": [1, None, 3], "name": ["a", "b", "c"]})

    def test_default_warns_and_loads_the_rest(self, recwarn):
        graph = KnowledgeGraph()
        report = graph.add_nodes(self._mixed(), "T", "id", "name")
        assert (report["nodes_created"], report["nodes_skipped"]) == (2, 1)
        assert any("skipped" in str(w.message) for w in recwarn)

    def test_error_refuses_the_whole_call_naming_row_and_value(self):
        graph = KnowledgeGraph()
        with pytest.raises(kglite.ArgumentError) as excinfo:
            graph.add_nodes(self._mixed(), "T", "id", "name", on_invalid="error")
        message = str(excinfo.value)
        assert "1 of 3 rows" in message, message
        assert "row 1" in message, message
        assert "nan" in message.lower(), message
        # Refusal before mutation: nothing landed.
        assert graph.cypher("MATCH (n:T) RETURN count(n) AS c")[0]["c"] == 0

    def test_skip_is_silent_but_still_reports(self, recwarn):
        graph = KnowledgeGraph()
        report = graph.add_nodes(self._mixed(), "T", "id", "name", on_invalid="skip")
        assert (report["nodes_created"], report["nodes_skipped"]) == (2, 1)
        assert report["has_errors"] is True
        assert report["errors"]
        assert [str(w.message) for w in recwarn] == []

    def test_an_unknown_policy_is_rejected(self):
        graph = KnowledgeGraph()
        with pytest.raises(kglite.ArgumentError, match="on_invalid"):
            graph.add_nodes(self._mixed(), "T", "id", "name", on_invalid="raise")


class TestObjectColumnStringification:
    """An object column with anything other than text in it is stored as text.
    The coercion is the design (the columnar store has no heterogeneous
    variant); doing it in silence was the defect."""

    def test_mixed_object_column_warns_naming_column_and_row(self, recwarn):
        graph = KnowledgeGraph()
        df = pd.DataFrame({"id": [1, 2, 3], "v": [10, 20, "N/A"]})
        graph.add_nodes(df, "T", "id")
        messages = [str(w.message) for w in recwarn]
        assert any("'v'" in m and "row 0" in m and "column_types" in m for m in messages), messages
        # ...and the values really are text now, which is what the warning says.
        stored = graph.cypher("MATCH (n:T) WHERE n.id = 1 RETURN n.v AS v")[0]["v"]
        assert stored == "10"

    def test_a_clean_text_object_column_does_not_warn(self, recwarn):
        graph = KnowledgeGraph()
        df = pd.DataFrame({"id": [1, 2, 3], "s": ["a", None, "c"]})
        graph.add_nodes(df, "T", "id")
        assert [str(w.message) for w in recwarn] == []

    def test_error_policy_refuses_the_stringification(self):
        graph = KnowledgeGraph()
        df = pd.DataFrame({"id": [1, 2, 3], "v": [10, 20, "N/A"]})
        with pytest.raises(kglite.ArgumentError, match="object dtype"):
            graph.add_nodes(df, "T", "id", on_invalid="error")

    def test_an_explicit_column_type_silences_it(self, recwarn):
        graph = KnowledgeGraph()
        df = pd.DataFrame({"id": [1, 2, 3], "v": [10, 20, "N/A"]})
        graph.add_nodes(df, "T", "id", column_types={"v": "string"})
        assert [str(w.message) for w in recwarn] == []


class TestIdSkipMessage:
    def test_the_skip_reason_names_a_type_that_actually_fits(self, recwarn):
        # The old advice was `column_types={'id': 'string'}` for *any* unparsed
        # id, which turns integer keys into text rather than widening them.
        graph = KnowledgeGraph()
        df = pd.DataFrame({"id": [1, 2**40]})
        report = graph.add_nodes(df, "T", "id", column_types={"id": "uniqueid"})
        assert report["nodes_skipped"] == 1
        reason = " ".join(report["errors"])
        assert "'int64'" in reason, reason
        assert "'string'" in reason, reason

    def test_an_explicit_uniqueid_override_is_still_honoured(self, recwarn):
        # The widening is a change to *auto-detection* only. A caller who names
        # the compact key gets the compact key, and the row that does not fit
        # is still reported rather than silently widened behind their back.
        graph = KnowledgeGraph()
        graph.add_nodes(pd.DataFrame({"id": [1, 2**40]}), "T", "id", column_types={"id": "uniqueid"})
        assert graph.cypher("MATCH (n:T) RETURN n.id AS i").to_df()["i"].tolist() == [1]

    def test_error_names_the_value_that_does_not_fit(self):
        graph = KnowledgeGraph()
        with pytest.raises(kglite.ArgumentError) as excinfo:
            graph.add_nodes(
                pd.DataFrame({"id": [1, 2**40]}),
                "T",
                "id",
                column_types={"id": "uniqueid"},
                on_invalid="error",
            )
        assert "1099511627776" in str(excinfo.value), str(excinfo.value)
