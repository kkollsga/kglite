"""#6 — parse_json() Cypher function (operator report 2026-06-16).

Structured data (code-graph Function.parameters / Class.fields) is stored as
JSON strings because the columnar store holds scalars only. parse_json() parses
that string into a structured map/list/scalar so Cypher can predicate over it
with the existing any()/all()/comprehension/subscript machinery.
"""

from pathlib import Path

import kglite


def test_parse_json_object_field_access() -> None:
    g = kglite.KnowledgeGraph()
    rows = list(g.cypher('RETURN parse_json(\'{"name":"Alice","age":30}\') AS o'))
    obj = rows[0]["o"]
    assert obj["name"] == "Alice"
    assert obj["age"] == 30


def test_parse_json_list_and_index() -> None:
    g = kglite.KnowledgeGraph()
    rows = list(g.cypher('RETURN parse_json(\'[{"name":"x"},{"name":"y"}]\')[0]["name"] AS first'))
    assert rows[0]["first"] == "x"


def test_parse_json_invalid_returns_null() -> None:
    g = kglite.KnowledgeGraph()
    rows = list(g.cypher("RETURN parse_json('not json') AS r"))
    assert rows[0]["r"] is None

    g2 = kglite.KnowledgeGraph()
    # from_json alias behaves identically.
    rows2 = list(g2.cypher("RETURN from_json('{\"a\":1}')['a'] AS a"))
    assert rows2[0]["a"] == 1


def test_parse_json_predicate_over_parameters() -> None:
    """The operator's reproducer: find functions with a parameter of a given
    type, predicating over the JSON-string `parameters` property."""
    from kglite import code_tree

    root = Path(__file__).parent / "fixtures_parse_json_proj"
    if root.exists():
        import shutil

        shutil.rmtree(root)
    root.mkdir()
    try:
        (root / "m.py").write_text(
            "def takes_dataset(codes: Dataset, n: int):\n    return 1\n\ndef takes_nothing():\n    return 2\n"
        )
        g = code_tree.build(str(root))

        # Sanity: parameters is a JSON string with a type_annotation field.
        raw = list(g.cypher("MATCH (f:Function {name:'takes_dataset'}) RETURN f.parameters AS p"))[0]["p"]
        assert isinstance(raw, str) and "type_annotation" in raw, raw

        hits = {
            r["n"]
            for r in g.cypher(
                "MATCH (f:Function) "
                "WHERE any(p IN parse_json(f.parameters) WHERE p.type_annotation = 'Dataset') "
                "RETURN f.name AS n"
            )
        }
        assert hits == {"takes_dataset"}, hits
    finally:
        import shutil

        if root.exists():
            shutil.rmtree(root)
