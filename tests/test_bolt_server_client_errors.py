"""A statement that fails on what it was given is a client error on the wire.

`CypherExecution` — a malformed `valid_at` date, a property the type does not
have, a type with no declared validity interval — was published as
`Neo.DatabaseError.Statement.ExecutionFailed`, which tells a Neo4j driver and
its retry/routing logic that the *server* broke. It is the query's to fix, so
it now arrives as `Neo.ClientError.Statement.ArgumentError`, the class the
neo4j driver raises as `ClientError`.
"""

from __future__ import annotations

import pytest

neo4j = pytest.importorskip("neo4j")

pytestmark = [pytest.mark.bolt]

USER_ERRORS = [
    # A date that is not a date.
    ("MATCH (p:Person) RETURN valid_at(p, 'garbage', 'id', 'id') AS v", "the date argument 'garbage'"),
    # A bound property no Person has.
    ("MATCH (p:Person) RETURN valid_at(p, '2020', 'nope', 'id') AS v", "property 'nope' does not exist"),
    # A type with no declared validity interval.
    ("MATCH (p:Person) RETURN valid_at(p, '2020') AS v", "has no declared validity interval"),
]


@pytest.mark.parametrize(("query", "message"), USER_ERRORS, ids=["bad-date", "unknown-property", "undeclared"])
def test_a_user_input_error_is_a_client_error(bolt_server, query, message):
    with neo4j.GraphDatabase.driver(bolt_server, auth=("neo4j", "password")) as driver:
        with driver.session() as session:
            with pytest.raises(neo4j.exceptions.ClientError) as excinfo:
                session.run(query).consume()
            # The session stays usable after the failure.
            assert session.run("RETURN 1 AS one").single()["one"] == 1
    error = excinfo.value
    assert error.code == "Neo.ClientError.Statement.ArgumentError"
    assert not isinstance(error, neo4j.exceptions.DatabaseError)
    assert message in str(error)
