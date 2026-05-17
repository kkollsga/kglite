"""exists(prop) error-message guidance.

KGLite does not implement Neo4j's legacy `exists(n.prop)` property-existence
check. Operators writing this against the engine should see an error that
points them at `WHERE n.prop IS NOT NULL` — the modern, supported form.

This pins both the steered case (property access shape) and the generic
fallback (anything else inside the parens).
"""

from __future__ import annotations

import pandas as pd
import pytest

import kglite


@pytest.fixture
def graph() -> kglite.KnowledgeGraph:
    g = kglite.KnowledgeGraph()
    df = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Carol"],
            "email": ["a@x", None, "c@x"],
        }
    )
    g.add_nodes(df, "Person", "id", "name")
    return g


def test_exists_property_access_suggests_is_not_null(graph):
    """`exists(n.prop)` is Neo4j legacy syntax; the error must steer to IS NOT NULL."""
    with pytest.raises(Exception) as exc:
        graph.cypher("MATCH (n:Person) WHERE exists(n.email) RETURN n.name")
    msg = str(exc.value)
    assert "IS NOT NULL" in msg, f"error must mention IS NOT NULL alternative: {msg!r}"
    assert "Neo4j" in msg, f"error should label this as Neo4j legacy syntax: {msg!r}"


def test_exists_property_access_in_not(graph):
    """`NOT exists(n.prop)` — same steer applies."""
    with pytest.raises(Exception) as exc:
        graph.cypher("MATCH (n:Person) WHERE NOT exists(n.email) RETURN n.name")
    msg = str(exc.value)
    assert "IS NOT NULL" in msg


def test_exists_brace_form_still_works(graph):
    """The supported `EXISTS { ... }` pattern-existence form must NOT trigger the error."""
    rows = graph.cypher(
        "MATCH (n:Person) WHERE NOT EXISTS { (n)-[:KNOWS]->() } RETURN n.name AS n ORDER BY n"
    ).to_list()
    # All 3 persons have no KNOWS edges in this fixture.
    assert [r["n"] for r in rows] == ["Alice", "Bob", "Carol"]


def test_exists_paren_pattern_form_still_works(graph):
    """The supported `EXISTS((n)-[:R]->())` form must NOT trigger the error."""
    # Same fixture has no edges, so all rows survive the NOT EXISTS.
    rows = graph.cypher("MATCH (n:Person) WHERE NOT EXISTS((n)-[:KNOWS]->()) RETURN n.name AS n ORDER BY n").to_list()
    assert [r["n"] for r in rows] == ["Alice", "Bob", "Carol"]


def test_is_not_null_is_the_documented_alternative(graph):
    """The error suggests IS NOT NULL — confirm that alternative actually works."""
    rows = graph.cypher("MATCH (n:Person) WHERE n.email IS NOT NULL RETURN n.name AS n ORDER BY n").to_list()
    assert [r["n"] for r in rows] == ["Alice", "Carol"]
