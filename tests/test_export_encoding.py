"""Format parsers must recover ordinary whitespace, properties and endpoints."""

import csv
import datetime
import json
import xml.etree.ElementTree as ET

import pandas as pd
import pytest

from kglite import KnowledgeGraph

TEXT = '雪 quote" slash\\ tab\t line\ncarriage\rend'
KEY = "ordinary\tkey"


def _graph():
    graph = KnowledgeGraph()
    graph.add_nodes(pd.DataFrame({"id": [1, 2], "title": [TEXT, "second"], KEY: [TEXT, "plain"]}), "Doc", "id", "title")
    graph.cypher(
        "MATCH(a:Doc),(b:Doc) WHERE a.id=1 AND b.id=2 "
        "CREATE(a)-[:LINK {source:'attribute source',target:'attribute target',type:'attribute type',note:$note}]->(b)",
        params={"note": TEXT},
    )
    return graph


@pytest.mark.parametrize("format", ["json", "d3"])
def test_json_controls_and_structural_edge_fields(format):
    data = json.loads(_graph().export_string(format=format))
    nodes = data["nodes"]
    first = next(node for node in nodes if node["id"] == 1)
    assert first["title"] == first[KEY] == TEXT
    assert len(data["links"]) == 1
    edge = data["links"][0]
    assert edge == {
        "source": next(i for i, node in enumerate(nodes) if node["id"] == 1),
        "target": next(i for i, node in enumerate(nodes) if node["id"] == 2),
        "type": "LINK",
        "note": TEXT,
    }


@pytest.mark.parametrize("format", ["graphml", "gexf"])
def test_xml_whitespace_roundtrip_in_text_and_attributes(format):
    root = ET.fromstring(_graph().export_string(format=format))
    if format == "graphml":
        ns = {"g": "http://graphml.graphdrawing.org/xmlns"}
        nodes = root.findall(".//g:node", ns)
        first = next(node for node in nodes if node.find("g:data[@key='node_id']", ns).text == "1")
        assert first.find("g:data[@key='node_title']", ns).text == TEXT
        assert first.find("g:data[@key='node_label']", ns).text == TEXT
        properties = json.loads(first.find("g:data[@key='node_properties']", ns).text)
        assert properties[KEY] == TEXT
        edge = root.find(".//g:edge", ns)
        assert json.loads(edge.find("g:data[@key='edge_properties']", ns).text)["note"] == TEXT
    else:
        ns = {"g": "http://www.gexf.net/1.2draft"}
        first = next(node for node in root.findall(".//g:node", ns) if node.attrib["label"] == TEXT)
        assert first.find("g:attvalues/g:attvalue[@for='1']", ns).attrib["value"] == TEXT


def test_csv_carriage_return_cells_roundtrip(tmp_path):
    graph = _graph()
    graph.export(str(tmp_path / "graph.csv"), format="csv")
    with (tmp_path / "graph_nodes.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2
    assert {row["title"] for row in rows} == {TEXT, "second"}
    graph.export_csv(str(tmp_path / "directory"))
    with (tmp_path / "directory/nodes/Doc.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    first = next(row for row in rows if row["id"] == "1")
    assert first["title"] == first[KEY] == TEXT


@pytest.mark.parametrize("format", ["json", "graphml"])
def test_timestamp_property_preserves_fraction(format):
    graph = KnowledgeGraph()
    value = datetime.datetime(2025, 1, 2, 3, 4, 5, 123456)
    graph.cypher("CREATE(:Event {id:1, ts:$ts})", params={"ts": value})
    output = graph.export_string(format=format)
    if format == "json":
        actual = json.loads(output)["nodes"][0]["ts"]
    else:
        root = ET.fromstring(output)
        ns = {"g": "http://graphml.graphdrawing.org/xmlns"}
        properties = root.find(".//g:node/g:data[@key='node_properties']", ns)
        actual = json.loads(properties.text)["ts"]
    assert actual == "2025-01-02T03:04:05.123456"


def test_finite_point_property_keeps_coordinate_values():
    graph = KnowledgeGraph()
    graph.cypher("CREATE(:Location {id:1, p:point(59.9,10.7)})")
    assert json.loads(graph.export_string("json"))["nodes"][0]["p"] == {"lat": 59.9, "lon": 10.7}


def test_plain_d3_edge_structural_fields_win_independently_of_control_escaping():
    graph = KnowledgeGraph()
    graph.cypher("CREATE(:Doc{id:1,title:'first'}),(:Doc{id:2,title:'second'})")
    graph.cypher(
        "MATCH(a:Doc),(b:Doc) WHERE a.id=1 AND b.id=2 "
        "CREATE(a)-[:LINK {source:'stored source',target:'stored target',type:'stored type',note:'ordinary'}]->(b)"
    )
    data = json.loads(graph.export_string(format="d3"))
    assert data["links"] == [
        {
            "source": next(i for i, node in enumerate(data["nodes"]) if node["id"] == 1),
            "target": next(i for i, node in enumerate(data["nodes"]) if node["id"] == 2),
            "type": "LINK",
            "note": "ordinary",
        }
    ]
