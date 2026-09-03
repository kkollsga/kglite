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


def test_unknown_top_level_key_raises_naming_connections():
    with pytest.raises(
        ValueError,
        match=r"unknown key 'relationships'\. Accepted keys: .*'connections'",
    ):
        kglite.from_records(
            {
                "nodes": [{"type": "Doc", "id_field": "id", "records": [{"id": 1}]}],
                "relationships": [],
            }
        )


def test_unknown_node_spec_key_suggests_the_near_miss():
    with pytest.raises(
        ValueError,
        match=r"nodes\[0\]: unknown key 'id_feild'\. Did you mean 'id_field'\?",
    ):
        kglite.from_records({"nodes": [{"type": "Doc", "id_feild": "id", "records": [{"id": 1}]}]})


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


def test_node_spec_labels_are_stamped():
    """`labels` on a node spec is the from_records twin of the blueprint key:
    every node of the type carries them, so a union query can name one label
    instead of every type in it."""
    kg = kglite.from_records(
        {
            "nodes": [
                {
                    "type": "Person",
                    "id_field": "id",
                    "labels": ["Human", "Agent"],
                    "records": [{"id": 1}, {"id": 2}],
                }
            ]
        }
    )
    assert len(list(kg.cypher("MATCH (n:Human) RETURN n.id"))) == 2
    assert len(list(kg.cypher("MATCH (n:Agent) RETURN n.id"))) == 2


def test_labels_reach_vivified_endpoint_stubs():
    """A `vivify`d endpoint is a node of the declared type and must carry the
    type's labels; otherwise `MATCH (:Place)` silently misses exactly the
    nodes that arrived as an edge endpoint."""
    kg = kglite.from_records(
        {
            "nodes": [
                {"type": "Person", "id_field": "id", "records": [{"id": 1}]},
                {
                    "type": "City",
                    "id_field": "id",
                    "labels": ["Place"],
                    "records": [{"id": 10}],
                },
            ],
            "connections": [
                {
                    "type": "LIVES_IN",
                    "source_type": "Person",
                    "source_id_field": "src",
                    "target_type": "City",
                    "target_id_field": "tgt",
                    "records": [{"src": 1, "tgt": 10}, {"src": 1, "tgt": 99}],
                }
            ],
        }
    )
    assert len(list(kg.cypher("MATCH (n:Place) RETURN n.id"))) == 2


def test_labels_survive_an_empty_records_list():
    """A type whose nodes all arrive as vivified endpoints declares an empty
    `records` list; its labels must still be stamped. They were dropped before
    the stamping pass ever saw them, so `MATCH (n:Text)` found nothing."""
    kg = kglite.from_records(
        {
            "nodes": [
                {"type": "Doc", "id_field": "id", "labels": ["Text"], "records": []},
                {"type": "Src", "id_field": "id", "records": [{"id": 1}]},
            ],
            "connections": [
                {
                    "type": "CITES",
                    "source_type": "Src",
                    "source_id_field": "s",
                    "target_type": "Doc",
                    "target_id_field": "t",
                    "records": [{"s": 1, "t": 7}],
                }
            ],
        }
    )
    assert [r["id"] for r in kg.cypher("MATCH (n:Text) RETURN n.id AS id").to_dicts()] == [7]


def test_labels_must_be_an_array_of_strings():
    with pytest.raises(ValueError, match=r"nodes\[0\]: 'labels' must be an array of strings"):
        kglite.from_records({"nodes": [{"type": "Doc", "id_field": "id", "labels": "Text", "records": [{"id": 1}]}]})


def _union_spec(target_type, type_field=None, links=None):
    if links is None:
        links = [
            {"src": "M1", "tgt": "D1", "kind": "Disease"},
            {"src": "M1", "tgt": "P1", "kind": "Phenotype"},
            {"src": "M2", "tgt": "E1", "kind": "Exposure"},
        ]
    conn = {
        "type": "ASSOCIATED_WITH",
        "source_type": "Microbe",
        "source_id_field": "src",
        "target_type": target_type,
        "target_id_field": "tgt",
        "records": links,
    }
    if type_field is not None:
        conn["target_type_column"] = type_field
    return {
        "nodes": [
            {"type": "Microbe", "id_field": "id", "records": [{"id": "M1"}, {"id": "M2"}]},
            {"type": "Disease", "id_field": "id", "records": [{"id": "D1"}]},
            {"type": "Phenotype", "id_field": "id", "records": [{"id": "P1"}]},
            {"type": "Exposure", "id_field": "id", "records": [{"id": "E1"}]},
        ],
        "connections": [conn],
    }


def _landed(kg):
    return sorted(
        (r["src"], r["tgt"], r["t"])
        for r in kg.cypher(
            "MATCH (m:Microbe)-[:ASSOCIATED_WITH]->(x) RETURN m.id AS src, x.id AS tgt, head(labels(x)) AS t"
        ).to_list()
    )


UNION_EXPECTED = [
    ("M1", "D1", "Disease"),
    ("M1", "P1", "Phenotype"),
    ("M2", "E1", "Exposure"),
]


def test_target_type_list_routes_by_id_probe():
    kg = kglite.from_records(_union_spec(["Disease", "Phenotype", "Exposure"]))
    assert _landed(kg) == UNION_EXPECTED


def test_target_type_column_routes_each_record():
    kg = kglite.from_records(_union_spec(["Disease", "Phenotype", "Exposure"], type_field="kind"))
    assert _landed(kg) == UNION_EXPECTED


def test_a_record_naming_an_undeclared_target_type_raises():
    """from_records is the strict sibling: its key sets are closed and its
    values are too."""
    links = [{"src": "M1", "tgt": "X1", "kind": "Chemical"}]
    with pytest.raises(ValueError, match="'Chemical'"):
        kglite.from_records(_union_spec(["Disease", "Phenotype"], type_field="kind", links=links))


def test_a_record_missing_the_target_type_field_raises():
    links = [{"src": "M1", "tgt": "D1"}]
    with pytest.raises(ValueError, match="record has no string 'kind'"):
        kglite.from_records(_union_spec(["Disease", "Phenotype"], type_field="kind", links=links))


def test_a_string_target_type_is_unchanged():
    links = [{"src": "M1", "tgt": "D1", "kind": "Disease"}]
    kg = kglite.from_records(_union_spec("Disease", links=links))
    assert _landed(kg) == [("M1", "D1", "Disease")]


def test_an_empty_target_type_list_raises():
    with pytest.raises(ValueError, match="'target_type' names no node type"):
        kglite.from_records(_union_spec([]))


def test_union_target_drop_policy_still_drops_the_unresolvable():
    spec = _union_spec(
        ["Disease", "Phenotype", "Exposure"],
        type_field="kind",
        links=[
            {"src": "M1", "tgt": "D1", "kind": "Disease"},
            {"src": "M1", "tgt": "Z9", "kind": "Phenotype"},
        ],
    )
    kg = kglite.from_records(spec, on_missing_endpoint="drop")
    assert _landed(kg) == [("M1", "D1", "Disease")]


def _drop_probe_spec():
    return {
        "on_missing_endpoint": "drop",
        "nodes": [{"type": "Doc", "id_field": "id", "records": [{"id": 1}, {"id": 2}]}],
        "connections": [
            {
                "type": "LINKS",
                "source_type": "Doc",
                "source_id_field": "source",
                "target_type": "Doc",
                "target_id_field": "target",
                "records": [{"source": 1, "target": 2}, {"source": 1, "target": 99}],
            }
        ],
    }


def test_on_missing_endpoint_in_the_spec_is_honoured():
    """The guide lists `on_missing_endpoint` as a top-level spec key, and the
    shim used to overwrite it with the argument's default — so a spec asking
    for `drop` silently vivified instead."""
    kg = kglite.from_records(_drop_probe_spec())
    assert [r["i"] for r in kg.cypher("MATCH (n:Doc) RETURN n.id AS i ORDER BY i").to_list()] == [1, 2]


def test_the_argument_overrides_the_spec_key():
    kg = kglite.from_records(_drop_probe_spec(), on_missing_endpoint="vivify")
    assert [r["i"] for r in kg.cypher("MATCH (n:Doc) RETURN n.id AS i ORDER BY i").to_list()] == [1, 2, 99]


def test_neither_spelling_still_vivifies():
    spec = _drop_probe_spec()
    del spec["on_missing_endpoint"]
    kg = kglite.from_records(spec)
    assert [r["i"] for r in kg.cypher("MATCH (n:Doc) RETURN n.id AS i ORDER BY i").to_list()] == [1, 2, 99]
