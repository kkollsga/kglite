"""`kglite.from_records` — inline JSON records loader (P4).

JSON-native sibling to `from_blueprint`: build a graph from inline node +
connection records, no CSV files. Column types are inferred (so JSON arrays
become native list properties), and missing edge endpoints are vivified.
"""

import pytest

import kglite


def _spec():
    return {
        "nodes": [
            {
                "type": "Person",
                "id_field": "id",
                "title_field": "name",
                "records": [
                    {"id": 1, "name": "Alice", "aliases": ["a", "b"], "age": 30},
                    {"id": 2, "name": "Bob", "aliases": ["c"], "age": 25},
                ],
            },
            {"type": "Org", "id_field": "id", "records": [{"id": 100, "name": "Acme"}]},
        ],
        "connections": [
            {
                "type": "WORKS_AT",
                "source_type": "Person",
                "source_id_field": "pid",
                "target_type": "Org",
                "target_id_field": "oid",
                "records": [{"pid": 1, "oid": 100, "since": 2020}],
            }
        ],
    }


def test_nodes_and_edges_from_dict():
    kg = kglite.from_records(_spec())
    people = kg.cypher("MATCH (n:Person) RETURN n.id AS id, n.name AS name, n.age AS age ORDER BY id").to_dicts()
    assert people == [
        {"id": 1, "name": "Alice", "age": 30},
        {"id": 2, "name": "Bob", "age": 25},
    ]
    edges = kg.cypher(
        "MATCH (p:Person)-[r:WORKS_AT]->(o:Org) RETURN p.name AS p, o.name AS o, r.since AS since"
    ).to_dicts()
    assert edges == [{"p": "Alice", "o": "Acme", "since": 2020}]


def test_list_property_inferred_native():
    kg = kglite.from_records(_spec())
    # JSON array → native list property, not a stringified blob.
    hit = kg.cypher("MATCH (n:Person) WHERE 'a' IN n.aliases RETURN n.id AS id").to_dicts()
    assert hit == [{"id": 1}]
    miss = kg.cypher("MATCH (n:Person) WHERE 'ab' IN n.aliases RETURN n.id AS id").to_dicts()
    assert miss == []


def test_string_spec_input():
    kg = kglite.from_records('{"nodes":[{"type":"T","id_field":"id","records":[{"id":1}]}]}')
    assert kg.cypher("MATCH (n:T) RETURN count(n) AS c").to_dicts() == [{"c": 1}]


def test_endpoint_vivification():
    # A connection whose endpoints have no node spec vivifies stubs.
    spec = {
        "connections": [
            {
                "type": "REF",
                "source_type": "Doc",
                "source_id_field": "s",
                "target_type": "Doc",
                "target_id_field": "t",
                "records": [{"s": 1, "t": 2}, {"s": 2, "t": 3}],
            }
        ]
    }
    kg = kglite.from_records(spec)
    assert kg.cypher("MATCH (n:Doc) RETURN count(n) AS c").to_dicts() == [{"c": 3}]
    assert kg.cypher("MATCH ()-[r:REF]->() RETURN count(r) AS c").to_dicts() == [{"c": 2}]


def _missing_endpoint_spec():
    return {
        "nodes": [{"type": "Doc", "id_field": "id", "records": [{"id": 1}, {"id": 2}]}],
        "connections": [
            {
                "type": "REF",
                "source_type": "Doc",
                "source_id_field": "s",
                "target_type": "Doc",
                "target_id_field": "t",
                "records": [
                    {"s": 1, "t": 2, "weight": 3},
                    {"s": 2, "t": 99, "weight": 4},
                    {"s": None, "t": 1, "weight": 5},
                ],
            }
        ],
    }


@pytest.mark.parametrize("storage", ["default", "mapped", "disk"])
def test_endpoint_drop_policy_across_storage_modes(storage, tmp_path):
    path = str(tmp_path / "graph") if storage == "disk" else None
    kg = kglite.from_records(
        _missing_endpoint_spec(),
        storage=storage,
        path=path,
        on_missing_endpoint="drop",
    )

    assert kg.cypher("MATCH (n:Doc) RETURN count(n) AS c").to_dicts() == [{"c": 2}]
    assert kg.cypher("MATCH ()-[r:REF]->() RETURN count(r) AS c").to_dicts() == [{"c": 1}]


def test_endpoint_error_policy_is_deterministic():
    with pytest.raises(
        ValueError,
        match=r"connections\[0\]\.records\[1\].*target endpoint Doc\(99\) does not exist",
    ):
        kglite.from_records(_missing_endpoint_spec(), on_missing_endpoint="error")


def test_invalid_endpoint_policy_raises():
    with pytest.raises(ValueError, match="unknown on_missing_endpoint mode"):
        kglite.from_records(_spec(), on_missing_endpoint="guess")


def test_malformed_json_raises():
    with pytest.raises(ValueError):
        kglite.from_records("{not valid json")


def test_missing_required_field_raises():
    with pytest.raises(ValueError):
        # node spec missing 'id_field'
        kglite.from_records({"nodes": [{"type": "X", "records": [{"id": 1}]}]})


def test_equivalent_to_add_nodes_add_connections():
    """from_records should match the equivalent imperative build."""
    kg_fr = kglite.from_records(_spec())

    import pandas as pd

    kg_imp = kglite.KnowledgeGraph()
    kg_imp.add_nodes(
        pd.DataFrame({"id": [1, 2], "name": ["Alice", "Bob"], "aliases": [["a", "b"], ["c"]], "age": [30, 25]}),
        node_type="Person",
        unique_id_field="id",
        node_title_field="name",
    )
    kg_imp.add_nodes(pd.DataFrame({"id": [100], "name": ["Acme"]}), node_type="Org", unique_id_field="id")
    kg_imp.add_connections(
        pd.DataFrame({"pid": [1], "oid": [100], "since": [2020]}),
        "WORKS_AT",
        "Person",
        "pid",
        "Org",
        "oid",
    )

    q = "MATCH (p:Person)-[:WORKS_AT]->(o:Org) RETURN count(*) AS c"
    assert kg_fr.cypher(q).to_dicts() == kg_imp.cypher(q).to_dicts()
    assert (
        kg_fr.cypher("MATCH (n:Person) RETURN count(n) AS c").to_dicts()
        == kg_imp.cypher("MATCH (n:Person) RETURN count(n) AS c").to_dicts()
    )
