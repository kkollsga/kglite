"""Tests for temporal queries: datetime columns, valid_at, valid_during."""

import warnings

import pandas as pd
import pytest

from kglite import KnowledgeGraph


class TestDateTimeHandling:
    def test_datetime_column_type(self):
        graph = KnowledgeGraph()
        df = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "name": ["A", "B", "C"],
                "created": ["2020-01-01", "2021-06-15", "2022-12-31"],
            }
        )
        graph.add_nodes(df, "Item", "id", "name", column_types={"created": "datetime"})
        nodes = graph.select("Item").collect()
        assert len(nodes) == 3

    def test_datetime_comparison_filter(self):
        graph = KnowledgeGraph()
        df = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "name": ["A", "B", "C"],
                "date": ["2020-01-01", "2021-06-15", "2022-12-31"],
            }
        )
        graph.add_nodes(df, "Item", "id", "name", column_types={"date": "datetime"})
        result = graph.select("Item").where({"date": {">=": "2021-01-01"}})
        assert result.len() == 2


class TestValidAt:
    def test_valid_at_basic(self, petroleum_graph):
        result = petroleum_graph.select("Estimate").valid_at("2020-06-15")
        assert result.len() > 0

    def test_valid_at_custom_fields(self, petroleum_graph):
        result = petroleum_graph.select("Prospect").valid_at(
            "2020-06-15",
            date_from_field="date_from",
            date_to_field="date_to",
        )
        assert result.len() > 0

    def test_valid_at_no_matches(self, petroleum_graph):
        result = petroleum_graph.select("Estimate").valid_at("2000-01-01")
        assert result.len() == 0


class TestValidDuring:
    def test_valid_during_basic(self, petroleum_graph):
        result = petroleum_graph.select("Estimate").valid_during("2020-01-01", "2020-06-30")
        assert result.len() > 0

    def test_valid_during_partial_overlap(self, petroleum_graph):
        result = petroleum_graph.select("Estimate").valid_during("2020-10-01", "2021-03-31")
        assert result.len() > 0

    def test_valid_during_no_overlap(self, petroleum_graph):
        result = petroleum_graph.select("Estimate").valid_during("2000-01-01", "2000-12-31")
        assert result.len() == 0


class TestDeclaredTemporalTextColumns:
    """A declared temporal column parses the text it is given, and says so when
    it cannot.

    `column_types={'v': 'datetime'}` means "force date-only" — including for a
    text column whose values carry a time. It used to accept only date-only
    spellings and store NULL for everything else, without a warning, without
    `has_errors`, and without `on_invalid='error'` firing: a whole column
    silently emptied. The blueprint CSV grammar has always taken the date part
    of a date+time string, so the two routes disagreed about one declared type.
    """

    @staticmethod
    def _load(values, declared, on_invalid="warn"):
        graph = KnowledgeGraph()
        df = pd.DataFrame({"id": list(range(len(values))), "v": values})
        report = graph.add_nodes(df, "T", "id", column_types={"v": declared}, on_invalid=on_invalid)
        rows = graph.cypher("MATCH (n:T) RETURN n.v AS v ORDER BY n.id").to_list()
        return report, [r["v"] for r in rows]

    @pytest.mark.parametrize(
        "values",
        [
            ["2024-03-15 08:30:00", "2024-07-01 23:59:59"],
            ["2024-03-15T08:30:00", "2024-07-01T23:59:59"],
            ["2024-03-15 08:30", "2024-07-01 23:59"],
            ["2024-03-15 08:30:00.500", "2024-07-01 23:59:59.250"],
        ],
    )
    def test_datetime_takes_the_date_part_of_text_carrying_a_time(self, values):
        _, stored = self._load(values, "datetime")
        assert stored == ["2024-03-15", "2024-07-01"]

    @pytest.mark.parametrize(
        "values",
        [
            ["2024-03-15 08:30", "2024-07-01 23:59"],
            ["2024-03-15 08:30:00.500", "2024-07-01 23:59:59.250"],
        ],
    )
    def test_timestamp_accepts_the_same_text_spellings(self, values):
        _, stored = self._load(values, "timestamp")
        assert [v.date().isoformat() for v in stored] == ["2024-03-15", "2024-07-01"]
        assert [v.hour for v in stored] == [8, 23]

    def test_date_only_text_still_loads_under_both_declarations(self):
        _, as_date = self._load(["2024-03-15"], "datetime")
        assert as_date == ["2024-03-15"]
        _, as_timestamp = self._load(["2024-03-15"], "timestamp")
        assert as_timestamp[0].isoformat() == "2024-03-15T00:00:00"

    def test_datetime64_column_still_forces_date_only(self):
        """The declared meaning is unchanged: `'datetime'` drops time-of-day."""
        _, stored = self._load(pd.to_datetime(["2024-03-15 08:30:00", "2024-07-01 23:59:59"]), "datetime")
        assert stored == ["2024-03-15", "2024-07-01"]

    @pytest.mark.parametrize("declared", ["datetime", "timestamp"])
    def test_an_unparseable_cell_warns_and_names_itself(self, declared):
        with pytest.warns(UserWarning, match="could not be parsed"):
            _, stored = self._load(["2024-03-15", "nonsense"], declared)
        assert stored[1] is None

    @pytest.mark.parametrize("declared", ["datetime", "timestamp"])
    def test_on_invalid_error_refuses_an_unparseable_cell(self, declared):
        with pytest.raises(Exception) as exc:
            self._load(["2024-03-15", "nonsense"], declared, on_invalid="error")
        message = str(exc.value)
        assert "'v'" in message
        assert "nonsense" in message

    def test_on_invalid_skip_stays_silent(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _, stored = self._load(["2024-03-15", "nonsense"], "datetime", on_invalid="skip")
        assert stored == ["2024-03-15", None]

    def test_a_fully_parseable_column_says_nothing(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _, stored = self._load(["2024-03-15 08:30:00"], "datetime")
        assert stored == ["2024-03-15"]
