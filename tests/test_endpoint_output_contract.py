"""Internal endpoint references must not escape any public result consumer."""

import json
import subprocess
import sys

import pytest

import kglite


@pytest.fixture
def endpoint_graph():
    graph = kglite.KnowledgeGraph()
    graph.cypher("CREATE (a:Item {id:'a',title:'Alpha'}), (b:Item {id:'b',title:'Beta'}), (a)-[:LINK]->(b)")
    return graph


@pytest.mark.parametrize("handle", ["graph", "frozen", "session", "read_tx", "write_tx"])
@pytest.mark.parametrize("consumer", ["list", "scalar", "frame", "direct_frame"])
def test_endpoint_output_is_resolved_recursively(endpoint_graph, handle, consumer):
    graph = endpoint_graph
    subject = {
        "graph": lambda: graph,
        "frozen": graph.freeze,
        "session": graph.session,
        "read_tx": graph.begin_read,
        "write_tx": graph.begin,
    }[handle]()
    query = "MATCH ()-[r:LINK]->() RETURN {first:startNode(r),pair:[startNode(r),endNode(r)]} AS result"
    expected = {"first": "Alpha", "pair": ["Alpha", "Beta"]}
    try:
        if consumer == "direct_frame":
            assert subject.cypher(query, to_df=True).to_dict("records") == [{"result": expected}]
        else:
            result = subject.cypher(query)
            if consumer == "scalar":
                assert result.scalar() == expected
            elif consumer == "frame":
                assert result.to_df().to_dict("records") == [{"result": expected}]
            else:
                assert result.to_list() == [{"result": expected}]
                assert result.to_list() == [{"result": expected}]
    finally:
        if handle.endswith("tx"):
            subject.rollback()


def test_endpoint_output_uses_uncommitted_title_and_retains_earlier_result(endpoint_graph):
    graph = endpoint_graph
    query = "MATCH ()-[r:LINK]->() RETURN startNode(r) AS result"
    with graph.begin() as tx:
        earlier = tx.cypher(query)
        written = tx.cypher("MATCH(a:Item {id:'a'})-[r:LINK]->() SET a.title='Changed' RETURN startNode(r) AS result")
        assert earlier.to_list() == [{"result": "Alpha"}]
        assert written.to_list() == [{"result": "Changed"}]
        assert tx.cypher(query).scalar() == "Changed"
        assert graph.cypher(query).scalar() == "Alpha"
    assert graph.cypher(query).scalar() == "Changed"


def test_cli_endpoint_json_uses_the_same_resolved_output(endpoint_graph, tmp_path):
    path = tmp_path / "endpoints.kgl"
    endpoint_graph.save(str(path))
    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "kglite.cli",
            "query",
            str(path),
            "MATCH ()-[r:LINK]->() RETURN [startNode(r),endNode(r)] AS result",
            "--format",
            "json",
        ],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert process.returncode == 0, process.stderr
    assert json.loads(process.stdout) == [{"result": ["Alpha", "Beta"]}]


def test_ordinary_title_reference_chain_keeps_terminal_title():
    graph = kglite.KnowledgeGraph()
    graph.cypher(
        "CREATE (a:Item {id:1,title:'Alpha'}),(b:Item {id:2,title:'Beta'}),"
        "(c:Item {id:3,title:'Terminal'}),(a)-[:LINK]->(b),(b)-[:LINK]->(c)"
    )
    graph.cypher("MATCH (a:Item)-[r:LINK]->(:Item) SET a.title=endNode(r)")
    assert graph.cypher("MATCH (a:Item {id:1}) RETURN a.title AS title").scalar() == "Terminal"
