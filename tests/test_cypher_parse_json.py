"""#6 — parse_json() Cypher function (operator report 2026-06-16).

Structured data (code-graph Function.parameters / Class.fields) is stored as
JSON strings because the columnar store holds scalars only. parse_json() parses
that string into a structured map/list/scalar so Cypher can predicate over it
with the existing any()/all()/comprehension/subscript machinery.
"""

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
    type, predicating over the JSON-string `parameters` property.

    Uses a hand-built code-schema graph — two Function nodes whose
    ``parameters`` property is a JSON string, exactly as the columnar store
    holds a code graph's structured parameter list (scalars only, so the
    list is serialized to JSON)."""
    g = kglite.KnowledgeGraph()
    # `parameters` mirrors what a code-graph build emits: a JSON string
    # holding a list of {name, type_annotation} maps.
    takes_dataset_params = '[{"name": "codes", "type_annotation": "Dataset"}, {"name": "n", "type_annotation": "int"}]'
    g.cypher(
        "CREATE (f:Function {id: 'm.takes_dataset', name: 'takes_dataset', parameters: $p})",
        params={"p": takes_dataset_params},
    )
    g.cypher("CREATE (f:Function {id: 'm.takes_nothing', name: 'takes_nothing', parameters: '[]'})")

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
