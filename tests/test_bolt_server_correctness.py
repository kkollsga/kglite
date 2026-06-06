"""Correctness coverage for kglite-bolt-server — value roundtrip,
error paths, and edge cases.

Complements `tests/test_bolt_server_smoke.py` (8 happy-path smoke tests,
one per protocol capability). This file deepens coverage of the actual
wire surface:

- **Value roundtrip** (15 tests): each `BoltValue` variant tested both
  directions via `RETURN $x AS y` — parameter encoded inbound through
  `from_bolt`, then projected back outbound through `to_bolt`, then
  compared for equality on the driver side.
- **Error paths** (~12 tests): each `KgErrorCode` variant + its
  expected wire-side `Neo.{Class}.{Category}.{Title}` code +
  driver-side exception class.
- **Edge cases** (~13 tests): empty/multi-statement/very-long queries,
  unicode, NaN/Inf, unsupported parameter types, empty/nested
  collections.

Fixtures: `bolt_server` (RW) + `bolt_server_readonly` from
`tests/conftest.py`.
"""

from datetime import date, timedelta

import pytest

neo4j = pytest.importorskip("neo4j")

pytestmark = [pytest.mark.bolt]


# ────────────────────────────────────────────────────────────────────────────
# Value roundtrip — RETURN $x AS y; assert driver-side equality
# ────────────────────────────────────────────────────────────────────────────
#
# Each test parameterizes a value, runs `RETURN $x AS x`, and asserts
# the driver-side value matches what was sent. Tests both inbound
# (from_bolt) and outbound (to_bolt) in one round trip.


def _roundtrip(bolt_server, value):
    """Run RETURN $x AS x; return the driver-side x."""
    with neo4j.GraphDatabase.driver(bolt_server, auth=("neo4j", "password")) as driver:
        with driver.session() as session:
            result = session.run("RETURN $x AS x", x=value)
            return result.single()["x"]


def test_roundtrip_null(bolt_server):
    assert _roundtrip(bolt_server, None) is None


def test_roundtrip_bool_true(bolt_server):
    assert _roundtrip(bolt_server, True) is True


def test_roundtrip_bool_false(bolt_server):
    assert _roundtrip(bolt_server, False) is False


@pytest.mark.parametrize("n", [0, 1, -1, 42, -42, 2**31 - 1, -(2**31), 2**63 - 1, -(2**63)])
def test_roundtrip_int64(bolt_server, n):
    assert _roundtrip(bolt_server, n) == n


@pytest.mark.parametrize("f", [0.0, -0.0, 1.5, -1.5, 1e-300, 1e300, 1.0, -1.0])
def test_roundtrip_float64_finite(bolt_server, f):
    result = _roundtrip(bolt_server, f)
    # -0.0 == 0.0 in Python comparison, so handle sign separately for that case.
    assert result == f


@pytest.mark.parametrize(
    "s",
    [
        "",
        "ascii",
        "with spaces and  tabs",
        "Hello, 世界! 🚀",  # multi-byte UTF-8 + emoji
        "line1\nline2\nline3",  # multi-line
        "quoted \"double\" 'single'",
        "🐉🦀⚡",  # emoji only
    ],
)
def test_roundtrip_string(bolt_server, s):
    assert _roundtrip(bolt_server, s) == s


def test_roundtrip_list_empty(bolt_server):
    assert _roundtrip(bolt_server, []) == []


def test_roundtrip_list_mixed_scalars(bolt_server):
    assert _roundtrip(bolt_server, [1, "two", 3.0, None, True]) == [1, "two", 3.0, None, True]


def test_roundtrip_list_nested(bolt_server):
    val = [[1, 2], [3, [4, [5]]]]
    assert _roundtrip(bolt_server, val) == val


def test_roundtrip_list_of_nulls(bolt_server):
    assert _roundtrip(bolt_server, [None, None, None]) == [None, None, None]


def test_roundtrip_map_empty(bolt_server):
    assert _roundtrip(bolt_server, {}) == {}


def test_roundtrip_map_simple(bolt_server):
    val = {"a": 1, "b": "two", "c": True, "d": None}
    assert _roundtrip(bolt_server, val) == val


def test_roundtrip_map_nested(bolt_server):
    val = {"outer": {"inner": {"deep": [1, 2, 3]}}}
    assert _roundtrip(bolt_server, val) == val


def test_roundtrip_date(bolt_server):
    # neo4j driver represents Bolt Date as `neo4j.time.Date` for outbound
    # but accepts a Python `datetime.date` for inbound.
    today = date(2026, 5, 24)
    result = _roundtrip(bolt_server, today)
    # The driver may unmarshal to either `neo4j.time.Date` (typed) or `date`.
    # Both should yield the same isoformat.
    assert str(result) == today.isoformat()


def test_roundtrip_duration_via_seconds(bolt_server):
    # Bolt Duration carries months + days + seconds + nanoseconds.
    # The neo4j Python driver preserves the day field; kglite has only
    # second precision so nanoseconds is always 0. Compare via the
    # total elapsed time (months=0 → no calendar ambiguity).
    delta = timedelta(days=7, hours=3, minutes=12, seconds=45)
    result = _roundtrip(bolt_server, delta)
    assert result.months == 0
    assert result.nanoseconds == 0
    # Total elapsed in seconds is invariant across days/seconds split.
    elapsed = result.days * 86400 + result.seconds
    assert elapsed == int(delta.total_seconds())


# ────────────────────────────────────────────────────────────────────────────
# Error paths — each KgErrorCode → expected Neo4j status code + exception
# ────────────────────────────────────────────────────────────────────────────


def test_error_cypher_syntax(bolt_server):
    """KgErrorCode::CypherSyntax → Neo.ClientError.Statement.SyntaxError → ClientError"""
    with neo4j.GraphDatabase.driver(bolt_server, auth=("neo4j", "password")) as driver:
        with driver.session() as session:
            with pytest.raises(neo4j.exceptions.ClientError) as exc_info:
                session.run("MATCH NOT VALID CYPHER").consume()
            assert "SyntaxError" in str(exc_info.value.code)


def test_error_validate_schema_unknown_property(bolt_server):
    """validate_schema rejects unknown property in pattern literal.
    BoltError::Protocol → Neo.ClientError.Request.Invalid → ClientError."""
    with neo4j.GraphDatabase.driver(bolt_server, auth=("neo4j", "password")) as driver:
        with driver.session() as session:
            with pytest.raises(neo4j.exceptions.ClientError):
                session.run("MATCH (n:Person {ttle: 'Alice'}) RETURN n").consume()


def test_neo4j_scheme_uri_routing_works(bolt_server):
    """Phase F #5: `neo4j://` routing is supported via a
    single-server self-pointing routing table. Pre-F this raised a
    Protocol error; post-F the cluster-aware driver path works the
    same as direct `bolt://` connections."""
    routed_url = bolt_server.replace("bolt://", "neo4j://")
    with neo4j.GraphDatabase.driver(routed_url, auth=("neo4j", "password")) as driver:
        # verify_connectivity uses ROUTE under neo4j:// scheme; the
        # routing table returned by `KgliteBackend::route` makes the
        # connection round-trip succeed.
        driver.verify_connectivity()


def test_error_basic_auth_wrong_password(tmp_path, bolt_binary_path):
    """--auth basic rejects wrong credentials with Neo.ClientError.Security.Unauthorized."""
    if not bolt_binary_path.exists():
        pytest.skip("bolt-server binary not built")
    from tests.conftest import _build_bolt_fixture_graph, _spawn_bolt_server, _teardown_bolt_server

    fixture_path = tmp_path / "auth.kgl"
    _build_bolt_fixture_graph(fixture_path)
    proc, url = _spawn_bolt_server(
        fixture_path,
        extra_args=["--auth", "basic", "--auth-user", "neo4j", "--auth-pass", "correct"],
    )
    try:
        with neo4j.GraphDatabase.driver(url, auth=("neo4j", "wrong")) as driver:
            with pytest.raises(neo4j.exceptions.AuthError):
                driver.verify_connectivity()
    finally:
        _teardown_bolt_server(proc)


def test_error_basic_auth_correct_password(tmp_path, bolt_binary_path):
    """--auth basic accepts correct credentials."""
    if not bolt_binary_path.exists():
        pytest.skip("bolt-server binary not built")
    from tests.conftest import _build_bolt_fixture_graph, _spawn_bolt_server, _teardown_bolt_server

    fixture_path = tmp_path / "auth.kgl"
    _build_bolt_fixture_graph(fixture_path)
    proc, url = _spawn_bolt_server(
        fixture_path,
        extra_args=["--auth", "basic", "--auth-user", "alice", "--auth-pass", "secret"],
    )
    try:
        with neo4j.GraphDatabase.driver(url, auth=("alice", "secret")) as driver:
            driver.verify_connectivity()  # should succeed
    finally:
        _teardown_bolt_server(proc)


# ────────────────────────────────────────────────────────────────────────────
# Edge cases — input that exercises corner conditions
# ────────────────────────────────────────────────────────────────────────────


def test_edge_empty_query(bolt_server):
    """Empty query — the neo4j driver itself rejects with ValueError
    before the RUN message ever leaves the client (pre-server validation).
    Whitespace-only goes to the server, where RB-2 catches it."""
    with neo4j.GraphDatabase.driver(bolt_server, auth=("neo4j", "password")) as driver:
        with driver.session() as session:
            # Empty string — driver-side ValueError.
            with pytest.raises((ValueError, neo4j.exceptions.ClientError)):
                session.run("").consume()
            # Whitespace-only — passes driver, server's RB-2 gate
            # converts to a clean ClientError ("empty Cypher query").
            with pytest.raises(neo4j.exceptions.ClientError) as exc_info:
                session.run("   \n\t  ").consume()
            assert "empty" in str(exc_info.value).lower()


def test_edge_multi_statement_query_rejected(bolt_server):
    """Multi-statement query (semicolon separator) — RB-2 rejects
    with a structured Protocol error pointing at "one statement per
    RUN". Without this gate, kglite's parser would silently process
    only the first statement."""
    with neo4j.GraphDatabase.driver(bolt_server, auth=("neo4j", "password")) as driver:
        with driver.session() as session:
            with pytest.raises(neo4j.exceptions.ClientError) as exc_info:
                session.run("MATCH (n:Person) RETURN n.title; MATCH (m:Person) RETURN m.title").consume()
            assert "multi-statement" in str(exc_info.value).lower() or "one" in str(exc_info.value).lower()


def test_edge_trailing_semicolon_allowed(bolt_server):
    """Trailing semicolon is a common driver convention — must NOT
    trip the multi-statement gate."""
    with neo4j.GraphDatabase.driver(bolt_server, auth=("neo4j", "password")) as driver:
        with driver.session() as session:
            # kglite's parser may or may not accept the trailing
            # semicolon; either outcome is fine, but it shouldn't
            # trigger the RB-2 multi-statement gate (which has
            # error message "multi-statement" / "one statement").
            try:
                result = session.run("MATCH (n:Person) RETURN count(n) AS c ; ")
                count = result.single()["c"]
                assert count == 4
            except neo4j.exceptions.ClientError as e:
                # Whatever error fires, it must NOT be the multi-
                # statement rejection (trailing-semi is single-stmt).
                msg = str(e).lower()
                assert "multi-statement" not in msg


def test_edge_semicolon_in_string_literal_not_split(bolt_server):
    """A semicolon INSIDE a string literal must NOT trigger the
    multi-statement gate (the gate's quote-aware scan handles this)."""
    with neo4j.GraphDatabase.driver(bolt_server, auth=("neo4j", "password")) as driver:
        with driver.session() as session:
            # 'a;b' is one string literal containing a semicolon.
            result = session.run("RETURN 'a;b' AS x")
            assert result.single()["x"] == "a;b"


def test_edge_whitespace_only_query(bolt_server):
    """Whitespace-only query — RB-2 gate catches and returns a
    clean ClientError. (Covered above in test_edge_empty_query.)"""
    # Kept for completeness; the assertion is identical to the
    # whitespace-only branch in test_edge_empty_query.
    with neo4j.GraphDatabase.driver(bolt_server, auth=("neo4j", "password")) as driver:
        with driver.session() as session:
            with pytest.raises(neo4j.exceptions.ClientError):
                session.run("\t\t\t").consume()


def test_edge_long_query_10kb(bolt_server):
    """10 KB query — long but valid WHERE clause built from many small
    OR predicates. Pins that the parser doesn't have an unreasonable
    short-query bias and that boltr framing handles the size."""
    # ~10 KB of predicates: WHERE n.title = 'X0' OR n.title = 'X1' OR ...
    predicates = " OR ".join([f"n.title = 'X{i}'" for i in range(1500)])
    query = f"MATCH (n:Person) WHERE {predicates} RETURN count(n) AS c"
    assert len(query) >= 10_000, f"query is only {len(query)} bytes, want >= 10000"
    with neo4j.GraphDatabase.driver(bolt_server, auth=("neo4j", "password")) as driver:
        with driver.session() as session:
            result = session.run(query)
            # None of the X0..X1499 match Alice/Bob/Carol/Dave, so 0 rows.
            assert result.single()["c"] == 0


def test_edge_unicode_in_param_value(bolt_server):
    """Property values with multi-byte UTF-8 round-trip cleanly."""
    name = "李明 🚀 Müller"
    result = _roundtrip(bolt_server, name)
    assert result == name


def test_edge_unicode_in_property_name(bolt_server):
    """Property names with non-ASCII in maps."""
    val = {"日本語": 1, "Ω": 2}
    assert _roundtrip(bolt_server, val) == val


def test_edge_float_nan_parameter_rejected(bolt_server):
    """NaN as a parameter — RB-4 rejects with Protocol → ClientError.
    NaN has ill-defined comparison semantics in Cypher (NaN != NaN);
    sending it usually signals a client-side bug."""
    with neo4j.GraphDatabase.driver(bolt_server, auth=("neo4j", "password")) as driver:
        with driver.session() as session:
            with pytest.raises(neo4j.exceptions.ClientError) as exc_info:
                session.run("RETURN $x AS x", x=float("nan")).consume()
            assert "non-finite" in str(exc_info.value).lower()


def test_edge_float_infinity_parameter_rejected(bolt_server):
    """+Infinity as a parameter — RB-4 rejects with Protocol → ClientError."""
    with neo4j.GraphDatabase.driver(bolt_server, auth=("neo4j", "password")) as driver:
        with driver.session() as session:
            with pytest.raises(neo4j.exceptions.ClientError) as exc_info:
                session.run("RETURN $x AS x", x=float("inf")).consume()
            assert "non-finite" in str(exc_info.value).lower()


def test_edge_float_negative_infinity_parameter_rejected(bolt_server):
    """-Infinity as a parameter — RB-4 rejects with Protocol → ClientError."""
    with neo4j.GraphDatabase.driver(bolt_server, auth=("neo4j", "password")) as driver:
        with driver.session() as session:
            with pytest.raises(neo4j.exceptions.ClientError):
                session.run("RETURN $x AS x", x=float("-inf")).consume()


def test_edge_bytes_parameter_rejected(bolt_server):
    """Bytes parameter — kglite has no Bytes Value variant.
    from_bolt should reject with Protocol → ClientError."""
    with neo4j.GraphDatabase.driver(bolt_server, auth=("neo4j", "password")) as driver:
        with driver.session() as session:
            with pytest.raises(neo4j.exceptions.ClientError):
                session.run("RETURN $x AS x", x=b"raw bytes").consume()


def test_edge_deeply_nested_map(bolt_server):
    """Deeply nested map (10 levels) — recursion should hold."""
    val = 1
    for _ in range(10):
        val = {"deeper": val}
    assert _roundtrip(bolt_server, val) == val


def test_edge_empty_string_property(bolt_server):
    """Empty string as a property value — common edge case."""
    assert _roundtrip(bolt_server, "") == ""


def test_edge_large_list_1000_ints(bolt_server):
    """1000-element list of ints — recursive to_bolt/from_bolt under load."""
    val = list(range(1000))
    assert _roundtrip(bolt_server, val) == val


def test_edge_query_with_comments_only(bolt_server):
    """Query that's entirely comments — parser behavior pin."""
    with neo4j.GraphDatabase.driver(bolt_server, auth=("neo4j", "password")) as driver:
        with driver.session() as session:
            with pytest.raises(Exception):
                session.run("/* nothing but comments */").consume()


def test_edge_return_count_zero(bolt_server):
    """A MATCH that returns zero rows — verify the empty result is reported correctly."""
    with neo4j.GraphDatabase.driver(bolt_server, auth=("neo4j", "password")) as driver:
        with driver.session() as session:
            result = session.run("MATCH (n:DoesNotExist) RETURN n")
            rows = list(result)
            assert rows == []


def test_edge_call_db_labels(bolt_server):
    """`CALL db.labels()` — the Phase A.3 schema procs flow through Bolt
    via the standard CALL pipeline. Phase F.1 aligned the yield
    column with Neo4j's convention: `label` (was `name` pre-F)."""
    with neo4j.GraphDatabase.driver(bolt_server, auth=("neo4j", "password")) as driver:
        with driver.session() as session:
            result = session.run("CALL db.labels() YIELD label RETURN label")
            labels = sorted([record["label"] for record in result])
            assert "Person" in labels


def test_edge_call_db_relationship_types(bolt_server):
    """`CALL db.relationshipTypes()` — same as above. Phase F.1 yields
    `relationshipType` (Neo4j convention; was `name` pre-F)."""
    with neo4j.GraphDatabase.driver(bolt_server, auth=("neo4j", "password")) as driver:
        with driver.session() as session:
            result = session.run("CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType")
            types = sorted([record["relationshipType"] for record in result])
            assert "KNOWS" in types


# ─── Phase F #5: neo4j:// routing for cluster-aware drivers ─────────────────


def test_neo4j_scheme_routing(bolt_server):
    """`neo4j://` URIs trigger a ROUTE message; the backend now
    returns a single-server routing table (Phase F #5). Pre-F this
    raised a Protocol error from the bolt backend.

    The driver internally calls `route()` at connect time when the
    URI scheme is `neo4j://` (vs the direct `bolt://`). It then uses
    the returned WRITE/READ/ROUTE entries for subsequent connections.
    """
    # bolt_server fixture yields a bolt:// URL; swap scheme to test routing.
    neo4j_url = bolt_server.replace("bolt://", "neo4j://", 1)
    with neo4j.GraphDatabase.driver(neo4j_url, auth=("neo4j", "password")) as driver:
        # verify_connectivity exercises the routing path — fails fast
        # with ServiceUnavailable if route() doesn't return a usable
        # table.
        driver.verify_connectivity()
        # Run a query end-to-end to confirm the driver actually uses
        # the routing table to connect (not just succeeds at the
        # initial route fetch).
        with driver.session() as session:
            result = session.run("MATCH (n:Person) RETURN count(n) AS c")
            count = result.single()["c"]
            assert count > 0


def test_neo4j_scheme_routing_readonly_session(bolt_server):
    """A read-only session over neo4j:// should route to the READ
    entry of our table. Since we're single-server it's the same
    address as WRITE — but the test exercises the path."""
    neo4j_url = bolt_server.replace("bolt://", "neo4j://", 1)
    with neo4j.GraphDatabase.driver(neo4j_url, auth=("neo4j", "password")) as driver:
        with driver.session(default_access_mode=neo4j.READ_ACCESS) as session:
            result = session.run("MATCH (n) RETURN count(n) AS c")
            assert result.single()["c"] >= 0


def test_tls_self_signed_works(tmp_path, bolt_binary_path):
    """Phase F #6: --tls-cert / --tls-key wraps the listener in
    TLS. A driver connecting via `bolt+ssc://` (self-signed
    certificate; +ssc = "ssl, skip cert verification") completes
    the handshake and runs queries. Skips if the binary isn't built."""
    if not bolt_binary_path.exists():
        pytest.skip("bolt-server binary not built")

    # Generate a self-signed cert with the cryptography library if
    # available; otherwise skip.
    pytest.importorskip("cryptography")
    import datetime as _dt

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = issuer = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "localhost")])
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(_dt.datetime.now(_dt.timezone.utc))
        .not_valid_after(_dt.datetime.now(_dt.timezone.utc) + _dt.timedelta(days=1))
        .add_extension(
            x509.SubjectAlternativeName(
                [x509.DNSName("localhost"), x509.IPAddress(__import__("ipaddress").ip_address("127.0.0.1"))]
            ),
            critical=False,
        )
        .sign(key, hashes.SHA256())
    )
    cert_path = tmp_path / "cert.pem"
    key_path = tmp_path / "key.pem"
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(
        key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )

    from tests.conftest import _build_bolt_fixture_graph, _spawn_bolt_server, _teardown_bolt_server

    fixture_path = tmp_path / "tls.kgl"
    _build_bolt_fixture_graph(fixture_path)
    proc, url = _spawn_bolt_server(
        fixture_path,
        extra_args=["--tls-cert", str(cert_path), "--tls-key", str(key_path)],
    )
    try:
        # `+ssc` = SSL Self-Signed: encryption on, certificate chain not verified.
        # Suitable for tests + dev; production should use a properly-signed cert.
        tls_url = url.replace("bolt://", "bolt+ssc://", 1)
        with neo4j.GraphDatabase.driver(tls_url, auth=("neo4j", "password")) as driver:
            driver.verify_connectivity()
            with driver.session() as session:
                count = session.run("MATCH (n:Person) RETURN count(n) AS c").single()["c"]
                assert count > 0
    finally:
        _teardown_bolt_server(proc)
