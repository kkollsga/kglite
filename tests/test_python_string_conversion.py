"""Exact strings and independent Python ownership across recursive conversion."""

import gc

import pytest

import kglite

TEXTS = ["", "plain text", "a\0b", "東京 café Ω 😀 e\u0301", "x" * 4096]


def assert_native(actual, expected):
    assert type(actual) is type(expected)
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_native(actual[key], expected[key])
    elif isinstance(expected, list):
        assert len(actual) == len(expected)
        for a, e in zip(actual, expected):
            assert_native(a, e)
    else:
        assert actual == expected


@pytest.mark.parametrize("text", TEXTS)
def test_string_scalar_and_nested_values_own_their_python_bytes(text):
    graph = kglite.KnowledgeGraph()
    value = {"text": text, "nested": [text, {"again": text}], "empty": ""}
    view = graph.cypher("RETURN $value AS v,$text AS s", params={"value": value, "text": text})
    expected = [{"v": value, "s": text}]
    first = view.to_list()
    assert_native(first, expected)
    first[0]["v"]["nested"][1]["again"] = "altered"
    second = view.to_list()
    assert_native(second, expected)
    del view, graph
    gc.collect()
    assert_native(second, expected)


def test_graph_entity_strings_survive_writes_destruction_and_container_edits():
    graph = kglite.KnowledgeGraph()
    text = "a\0b 東京 😀 e\u0301"
    graph.cypher(
        "CREATE(a:Doc{id:'a',title:$text,payload:$text,tags:[$text,'']}),"
        "(b:Doc{id:'b',title:'',payload:'',tags:[]}),"
        "(a)-[:NOTE{body:$text}]->(b)",
        params={"text": text},
    )
    view = graph.cypher(
        "MATCH p=(a:Doc)-[r:NOTE]->(b:Doc) RETURN a,r,p,{nested:[a,$text]} AS wrapped", params={"text": text}
    )
    a = {
        "id": 0,
        "labels": ["Doc"],
        "properties": {"id": "a", "title": text, "type": "Doc", "payload": text, "tags": [text, ""]},
    }
    b = {"id": 1, "labels": ["Doc"], "properties": {"id": "b", "title": "", "type": "Doc", "payload": "", "tags": []}}
    r = {"id": 0, "start": 0, "end": 1, "type": "NOTE", "properties": {"body": text}}
    expected = [{"a": a, "r": r, "p": {"nodes": [a, b], "relationships": [r]}, "wrapped": {"nested": [a, text]}}]
    first = view.to_list()
    assert_native(first, expected)
    first[0]["p"]["nodes"][0]["properties"]["tags"].append("altered")
    first[0]["wrapped"]["nested"][0]["properties"]["payload"] = "altered"
    assert_native(view.to_list(), expected)
    graph.cypher("MATCH(n:Doc) SET n.payload='updated',n.title='new'")
    graph.cypher("MATCH(n:Doc) DETACH DELETE n")
    del graph
    gc.collect()
    second = view.to_list()
    assert_native(second, expected)
    del view
    gc.collect()
    assert_native(second, expected)
