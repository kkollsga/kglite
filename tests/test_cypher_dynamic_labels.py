"""Dynamic labels and relationship types supplied by query parameters.

`MATCH (n:$label)`, `MATCH (n:$(label))`, `-[:$type]->`, `CREATE (n:$label)`,
`SET n:$label`, `REMOVE n:$label`.

The point of the feature is not convenience: it is that a caller building a
query from untrusted input has **no escaping obligation left**. Before it, a
label or relationship type could only be spliced into the query text, so every
caller owned an injection surface. With it, identifiers are parameters exactly
like values are, and a parameter value is a NAME by construction — never
grammar. `test_a_parameter_value_is_a_name_never_syntax` is that claim.
"""

from __future__ import annotations

import pytest

import kglite


@pytest.fixture()
def graph() -> kglite.KnowledgeGraph:
    g = kglite.KnowledgeGraph()
    g.cypher("CREATE (a:Person {id: 1, name: 'Ada'}), (b:Person {id: 2, name: 'Bob'}), (c:Robot {id: 3, name: 'R2'})")
    g.cypher("MATCH (a:Person {id: 1}), (b:Person {id: 2}) CREATE (a)-[:KNOWS]->(b)")
    g.cypher("MATCH (a:Person {id: 1}), (c:Robot {id: 3}) CREATE (a)-[:OWNS]->(c)")
    return g


# ---------------------------------------------------------------- MATCH


@pytest.mark.parametrize("spelling", ["$label", "$(label)"])
def test_match_node_label_from_parameter(graph, spelling):
    rows = graph.cypher(
        f"MATCH (n:{spelling}) RETURN n.name AS name ORDER BY name",
        params={"label": "Person"},
    ).to_list()
    assert rows == [{"name": "Ada"}, {"name": "Bob"}]

    rows = graph.cypher(
        f"MATCH (n:{spelling}) RETURN n.name AS name ORDER BY name",
        params={"label": "Robot"},
    ).to_list()
    assert rows == [{"name": "R2"}]


def test_match_anonymous_node_label_from_parameter(graph):
    rows = graph.cypher("MATCH (:$label) RETURN count(*) AS n", params={"label": "Person"}).to_list()
    assert rows == [{"n": 2}]


@pytest.mark.parametrize("spelling", ["$type", "$(type)"])
def test_match_relationship_type_from_parameter(graph, spelling):
    rows = graph.cypher(
        f"MATCH (a)-[:{spelling}]->(b) RETURN a.name AS a, b.name AS b",
        params={"type": "KNOWS"},
    ).to_list()
    assert rows == [{"a": "Ada", "b": "Bob"}]

    rows = graph.cypher(
        f"MATCH (a)-[:{spelling}]->(b) RETURN a.name AS a, b.name AS b",
        params={"type": "OWNS"},
    ).to_list()
    assert rows == [{"a": "Ada", "b": "R2"}]


def test_parameter_in_a_relationship_type_alternation(graph):
    rows = graph.cypher(
        "MATCH (a)-[:KNOWS|$type]->(b) RETURN b.name AS b ORDER BY b",
        params={"type": "OWNS"},
    ).to_list()
    assert rows == [{"b": "Bob"}, {"b": "R2"}]


def test_parameter_as_a_secondary_label(graph):
    graph.cypher("MATCH (n:Person {id: 1}) SET n:Employee")
    rows = graph.cypher("MATCH (n:Person:$label) RETURN n.name AS name", params={"label": "Employee"}).to_list()
    assert rows == [{"name": "Ada"}]


def test_parameter_label_inside_an_exists_subquery(graph):
    rows = graph.cypher(
        "MATCH (a:Person) WHERE EXISTS { MATCH (a)-[:$type]->(:$label) } RETURN a.name AS name",
        params={"type": "OWNS", "label": "Robot"},
    ).to_list()
    assert rows == [{"name": "Ada"}]


# ---------------------------------------------------------------- write


def test_create_node_label_from_parameter(graph):
    graph.cypher("CREATE (n:$label {id: 9, name: 'Zed'})", params={"label": "Person"})
    rows = graph.cypher("MATCH (n:Person {id: 9}) RETURN n.name AS name").to_list()
    assert rows == [{"name": "Zed"}]


def test_create_relationship_type_from_parameter(graph):
    graph.cypher(
        "MATCH (a:Person {id: 2}), (c:Robot {id: 3}) CREATE (a)-[:$type]->(c)",
        params={"type": "OWNS"},
    )
    rows = graph.cypher("MATCH (a:Person)-[:OWNS]->(c:Robot) RETURN a.name AS name ORDER BY name").to_list()
    assert rows == [{"name": "Ada"}, {"name": "Bob"}]


def test_set_and_remove_label_from_parameter(graph):
    graph.cypher("MATCH (n:Person {id: 1}) SET n:$label", params={"label": "Employee"})
    assert graph.cypher("MATCH (n:Employee) RETURN n.name AS name").to_list() == [{"name": "Ada"}]

    graph.cypher("MATCH (n:Person {id: 1}) REMOVE n:$label", params={"label": "Employee"})
    assert graph.cypher("MATCH (n:Employee) RETURN n.name AS name").to_list() == []


def test_merge_label_from_parameter(graph):
    graph.cypher(
        "MERGE (n:$label {id: 1}) ON MATCH SET n.seen = true",
        params={"label": "Person"},
    )
    rows = graph.cypher("MATCH (n:Person {id: 1}) RETURN n.seen AS seen").to_list()
    assert rows == [{"seen": True}]


def test_where_label_check_from_parameter(graph):
    rows = graph.cypher(
        "MATCH (n) WHERE n:$label RETURN n.name AS name ORDER BY name",
        params={"label": "Robot"},
    ).to_list()
    assert rows == [{"name": "R2"}]


# ---------------------------------------------------------------- errors


def test_missing_parameter_is_a_clear_error(graph):
    with pytest.raises(Exception) as excinfo:
        graph.cypher("MATCH (n:$label) RETURN n")
    message = str(excinfo.value)
    assert "$label" in message or "label" in message
    assert "parameter" in message.lower()


def test_non_string_parameter_is_a_clear_error(graph):
    with pytest.raises(Exception) as excinfo:
        graph.cypher("MATCH (n:$label) RETURN n", params={"label": 7})
    message = str(excinfo.value).lower()
    assert "label" in message
    assert "string" in message


def test_empty_parameter_name_is_a_syntax_error(graph):
    with pytest.raises(Exception):
        graph.cypher("MATCH (n:$()) RETURN n", params={"label": "Person"})


# ---------------------------------------------------------------- semantics


def test_unknown_label_returns_empty_exactly_like_a_literal(graph):
    dynamic = graph.cypher("MATCH (n:$label) RETURN n.name AS name", params={"label": "Ghost"}).to_list()
    literal = graph.cypher("MATCH (n:Ghost) RETURN n.name AS name").to_list()
    assert dynamic == literal == []


def test_a_parameter_value_is_a_name_never_syntax(graph):
    """The strategic point: a parameter value can never become grammar.

    Every value below would change the query's *shape* if it were spliced into
    the query text. As a parameter it is a label name, so each one names a type
    that does not exist and the query returns no rows — never a syntax error,
    never a different query, never a leak of other labels' rows.
    """
    injections = [
        "Person) RETURN n UNION MATCH (m:Person",
        "`) RETURN n MATCH (m:Person",
        "Person`) {",
        "Person {id: 1}",
        "Person:Robot",
        "Person|Robot",
        "*",
        "'; DROP",
        "$label",
        "  ",
    ]
    for value in injections:
        rows = graph.cypher("MATCH (n:$label) RETURN n.name AS name", params={"label": value}).to_list()
        assert rows == [], f"parameter value {value!r} was not treated as a name"


def test_a_backticked_dollar_label_stays_a_literal_name(graph):
    """`` (n:`$label`) `` is a literal label named `$label`, not a parameter.

    The dynamic marker is carried out of band by the parser, so no spelling of
    a *literal* label can be mistaken for a parameter reference — which is what
    keeps the backtick-quoting rule (T9) meaning what it says.
    """
    graph.cypher("CREATE (n:`$label` {id: 42, name: 'literal'})")
    rows = graph.cypher("MATCH (n:`$label`) RETURN n.name AS name", params={"label": "Person"}).to_list()
    assert rows == [{"name": "literal"}]


def test_explain_of_a_dynamic_label_matches_the_literal_spelling(graph):
    dynamic = graph.cypher("EXPLAIN MATCH (n:$label) RETURN n.name AS name", params={"label": "Person"}).to_list()
    literal = graph.cypher("EXPLAIN MATCH (n:Person) RETURN n.name AS name").to_list()
    assert dynamic == literal


def test_the_same_query_text_reuses_no_stale_label(graph):
    """The parse/plan caches are keyed on query text — a resolved label must
    never be cached under it."""
    query = "MATCH (n:$label) RETURN count(*) AS n"
    assert graph.cypher(query, params={"label": "Person"}).to_list() == [{"n": 2}]
    assert graph.cypher(query, params={"label": "Robot"}).to_list() == [{"n": 1}]
    assert graph.cypher(query, params={"label": "Person"}).to_list() == [{"n": 2}]
