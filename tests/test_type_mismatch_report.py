"""A type mismatch on a follow-up load is reported, and says the rows landed.

`add_nodes` lists a column whose values disagree with the property's recorded
type under `errors`, but writes the values anyway — a column may hold several
types, and a mismatch is a diagnostic, not a refusal (pinned since 0.9.9 by
`test_graph_mutations.py::test_warn_on_errors_without_skips`). The entry used
to read like a refusal while the row was stored; it now says the values were
written and what the recorded type became.
"""

from __future__ import annotations

import datetime as dt
import warnings

import pandas as pd

import kglite


def test_a_date_load_after_a_string_set_reports_a_written_mismatch() -> None:
    graph = kglite.KnowledgeGraph()
    graph.add_nodes(
        pd.DataFrame({"code": ["a"], "vt": [dt.date(2010, 1, 1)]}), "M", "code", column_types={"vt": "date"}
    )
    graph.cypher("MATCH (m:M {code: 'a'}) SET m.vt = 'soon'").to_list()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = graph.add_nodes(
            pd.DataFrame({"code": ["z"], "vt": [dt.date(2021, 1, 1)]}), "M", "code", column_types={"vt": "date"}
        )
    assert report["nodes_created"] == 1 and report["nodes_skipped"] == 0
    (message,) = report["errors"]
    assert "Type mismatch for property 'vt'" in message
    assert "The values were written" in message and "now 'DateTime'" in message
    assert graph.cypher("MATCH (m:M {code: 'z'}) RETURN m.vt AS vt").to_list() == [{"vt": dt.date(2021, 1, 1)}]


def test_a_matching_follow_up_load_reports_nothing() -> None:
    graph = kglite.KnowledgeGraph()
    graph.add_nodes(
        pd.DataFrame({"code": ["a"], "vt": [dt.date(2010, 1, 1)]}), "M", "code", column_types={"vt": "date"}
    )
    report = graph.add_nodes(
        pd.DataFrame({"code": ["b"], "vt": [dt.date(2011, 1, 1)]}), "M", "code", column_types={"vt": "date"}
    )
    assert not report.get("errors")
