"""TODO / FIXME / etc. annotations on File nodes.

The parser walks every comment looking for these markers; a fast
byte-pattern pre-check skips files that contain none, so files with
markers must still surface them in the JSON-encoded `annotations`
column on the File node.
"""

import textwrap

import pytest

pytest.importorskip("tree_sitter")

from kglite.code_tree import build  # noqa: E402


def _annotations_for(graph, suffix: str) -> list[dict]:
    # Phase A.1 / C4 removed the PreProcessedValue JSON-string inference
    # hack at the Python boundary. The code-tree extractor still stores
    # annotations as a JSON string property on the File node, so the
    # test parses it explicitly here. Future cleanup: emit the property
    # as a native `Value::List(Vec<Value::Map>)` from the extractor and
    # drop the json.loads (lands alongside the C7a slim-down or in a
    # focused code-tree follow-up).
    import json

    rows = graph.cypher(
        "MATCH (f:File) WHERE f.path ENDS WITH $suf AND f.annotations IS NOT NULL RETURN f.annotations AS a",
        params={"suf": suffix},
    ).to_list()
    if not rows:
        return []
    raw = rows[0]["a"]
    if isinstance(raw, str):
        return json.loads(raw)
    return raw


def test_todo_fixme_extracted(tmp_path):
    src = textwrap.dedent(
        """
        // TODO: handle the empty case
        // FIXME wire up the cache
        public class Sample
        {
            public int Compute() { return 1; }
        }
        """
    )
    (tmp_path / "Sample.cs").write_text(src)
    g = build(str(tmp_path))
    anns = _annotations_for(g, "Sample.cs")
    kinds = sorted(a["kind"] for a in anns)
    assert kinds == ["FIXME", "TODO"], anns


def test_no_annotations_when_keywords_absent(tmp_path):
    # File has the *word* "todo" only as part of an identifier — the
    # aho-corasick precheck is case-insensitive substring, but the regex
    # requires a word boundary, so the result must be no annotations.
    src = textwrap.dedent(
        """
        public class Plain
        {
            public int todoCount() { return 0; }
        }
        """
    )
    (tmp_path / "Plain.cs").write_text(src)
    g = build(str(tmp_path))
    anns = _annotations_for(g, "Plain.cs")
    assert anns == []


def test_python_todo_extracted(tmp_path):
    # Pre-check + walk applies across every language, not just C#.
    (tmp_path / "m.py").write_text(
        textwrap.dedent(
            """
            # TODO: split this module
            def handler():
                return 42
            """
        )
    )
    g = build(str(tmp_path))
    anns = _annotations_for(g, "m.py")
    assert any(a["kind"] == "TODO" for a in anns), anns
