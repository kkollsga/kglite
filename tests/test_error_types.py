"""Phase A.2 / C1 — pin the typed exception class hierarchy.

These tests assert the *contract* — class names exported, inheritance
chains, and that `kglite.KgError` catches every typed engine error. They don't yet
exercise the executor's error paths (that's C2-C5's job; those tests
flip from xfail to passing as the migration sweeps through).

See `docs/python/error-handling.md` (added in C12) for the
full taxonomy and migration guidance.
"""

from __future__ import annotations

import pytest

import kglite

# ─── Module-level exports ────────────────────────────────────────────────────


class TestExports:
    """Every typed exception class is reachable via `kglite.X`."""

    @pytest.mark.parametrize(
        "name",
        [
            "KgError",
            "CypherError",
            "CypherSyntaxError",
            "CypherTimeoutError",
            "CypherExecutionError",
            "CypherTypeMismatchError",
            "SchemaError",
            "ValidationError",
            "ExprError",
            "NodeNotFoundError",
            "ConnectionNotFoundError",
            "PropertyNotFoundError",
            "FileError",
            "FileFormatError",
            "FileIoError",
            "ArgumentError",
            "MissingArgumentError",
            "InternerCollisionError",
            "InternalError",
        ],
    )
    def test_class_reachable(self, name: str):
        assert hasattr(kglite, name), f"kglite.{name} not exported — check error_py.rs::register and __init__.py"
        cls = getattr(kglite, name)
        assert isinstance(cls, type), f"kglite.{name} is not a class (got {type(cls).__name__})"


# ─── Inheritance hierarchy ───────────────────────────────────────────────────


class TestHierarchy:
    """The class hierarchy mirrors the Rust KgError enum taxonomy."""

    def test_kgerror_extends_exception(self):
        assert issubclass(kglite.KgError, Exception)

    def test_every_typed_exception_subclasses_kgerror(self):
        # If any of these fails, the create_exception! base argument
        # was wrong — every typed engine error should descend from KgError.
        typed = [
            kglite.CypherError,
            kglite.CypherSyntaxError,
            kglite.CypherTimeoutError,
            kglite.CypherExecutionError,
            kglite.CypherTypeMismatchError,
            kglite.SchemaError,
            kglite.ValidationError,
            kglite.ExprError,
            kglite.NodeNotFoundError,
            kglite.ConnectionNotFoundError,
            kglite.PropertyNotFoundError,
            kglite.FileError,
            kglite.FileFormatError,
            kglite.FileIoError,
            kglite.ArgumentError,
            kglite.MissingArgumentError,
            kglite.InternerCollisionError,
            kglite.InternalError,
        ]
        for cls in typed:
            assert issubclass(cls, kglite.KgError), f"{cls.__name__} doesn't subclass KgError — break in the hierarchy"

    def test_cypher_subhierarchy(self):
        # Cypher mid-tier: every Cypher-specific error extends CypherError.
        for cls in (
            kglite.CypherSyntaxError,
            kglite.CypherTimeoutError,
            kglite.CypherExecutionError,
            kglite.CypherTypeMismatchError,
        ):
            assert issubclass(cls, kglite.CypherError)
            assert issubclass(cls, kglite.KgError)

    def test_cypher_error_extends_kgerror(self):
        assert issubclass(kglite.CypherError, kglite.KgError)

    def test_specific_errors_are_not_each_others_subclasses(self):
        # CypherSyntaxError and SchemaError are siblings under KgError,
        # not in each other's chain.
        assert not issubclass(kglite.CypherSyntaxError, kglite.SchemaError)
        assert not issubclass(kglite.SchemaError, kglite.CypherSyntaxError)
        assert not issubclass(kglite.NodeNotFoundError, kglite.FileError)


# ─── Instantiation + isinstance ──────────────────────────────────────────────


class TestInstantiation:
    """User code can raise / catch the typed exceptions directly."""

    def test_raise_and_catch_specific(self):
        with pytest.raises(kglite.CypherSyntaxError, match="test message"):
            raise kglite.CypherSyntaxError("test message")

    def test_catch_via_cypher_base(self):
        with pytest.raises(kglite.CypherError):
            raise kglite.CypherTimeoutError("timeout")

    def test_catch_via_kgerror_engine_base(self):
        # Every typed engine exception is catchable via the shared base.
        with pytest.raises(kglite.KgError):
            raise kglite.SchemaError("missing prop")
        with pytest.raises(kglite.KgError):
            raise kglite.NodeNotFoundError("Person 'alice' not found")
        with pytest.raises(kglite.KgError):
            raise kglite.ArgumentError("bad arg")

    def test_catch_via_exception_fallback(self):
        # All typed errors are still Exception subclasses, so bare
        # `except Exception:` works as a last resort.
        with pytest.raises(Exception):
            raise kglite.CypherSyntaxError("x")

    def test_str_round_trip(self):
        e = kglite.CypherSyntaxError("expected RETURN")
        assert "expected RETURN" in str(e)


# ─── End-to-end: cypher() raises typed exceptions across all 3 modes ─────────


STORAGE_MODES = ("memory", "mapped", "disk")


@pytest.fixture(params=STORAGE_MODES, ids=STORAGE_MODES)
def small_graph_all_modes(request, tmp_path):
    """3-node fixture used to exercise typed exceptions from cypher()
    across all three storage backends. A.2 / C5 contract: the same
    exception type fires regardless of backend."""
    import pandas as pd

    if request.param == "memory":
        g = kglite.KnowledgeGraph()
    elif request.param == "mapped":
        g = kglite.KnowledgeGraph(storage="mapped")
    elif request.param == "disk":
        g = kglite.KnowledgeGraph(storage="disk", path=str(tmp_path / "g"))
    else:
        raise ValueError(request.param)

    persons = pd.DataFrame(
        [
            {"id": "alice", "name": "Alice", "age": 30},
            {"id": "bob", "name": "Bob", "age": 35},
            {"id": "carol", "name": "Carol", "age": 28},
        ]
    )
    g.add_nodes(persons, "Person", "id", "name")
    return g


class TestCypherExceptionsCrossMode:
    """Every Cypher error path produces the same typed exception across
    memory / mapped / disk. If a backend silently produces a different
    error type, the parity oracles wouldn't catch it but these would."""

    def test_syntax_error_raises_cypher_syntax_error(self, small_graph_all_modes):
        with pytest.raises(kglite.CypherSyntaxError):
            small_graph_all_modes.cypher("MATCH x RETURN y INVALID")

    def test_syntax_error_also_catchable_as_kgerror(self, small_graph_all_modes):
        with pytest.raises(kglite.KgError):
            small_graph_all_modes.cypher("MATCH x RETURN y INVALID")

    def test_syntax_error_message_carries_line_col(self, small_graph_all_modes):
        with pytest.raises(kglite.CypherSyntaxError) as exc_info:
            small_graph_all_modes.cypher("MATCH x RETURN y INVALID")
        msg = str(exc_info.value)
        assert "line" in msg.lower() and "col" in msg.lower(), f"expected line/col in message, got: {msg!r}"

    def test_syntax_error_exposes_line_col_attributes(self, small_graph_all_modes):
        with pytest.raises(kglite.CypherSyntaxError) as exc_info:
            small_graph_all_modes.cypher("MATCH (n)\nRETURN n INVALID")
        assert exc_info.value.line == 2
        assert isinstance(exc_info.value.col, int)
        assert exc_info.value.col >= 1

    def test_missing_parameter_raises_execution_error(self, small_graph_all_modes):
        with pytest.raises(kglite.CypherExecutionError):
            small_graph_all_modes.cypher("MATCH (n:Person) WHERE n.age > $nonexistent RETURN n.name")

    def test_missing_parameter_is_also_a_kgerror(self, small_graph_all_modes):
        with pytest.raises(kglite.KgError):
            small_graph_all_modes.cypher("MATCH (n:Person) WHERE n.age > $nonexistent RETURN n.name")

    def test_unknown_call_procedure_raises_execution_error(self, small_graph_all_modes):
        with pytest.raises(kglite.CypherExecutionError):
            small_graph_all_modes.cypher("CALL nonexistent_algo() YIELD node RETURN node")

    def test_invalid_regex_raises_execution_error(self, small_graph_all_modes):
        with pytest.raises(kglite.CypherExecutionError, match="Invalid regular expression"):
            small_graph_all_modes.cypher("MATCH (n:Person) WHERE n.name =~ '[invalid(' RETURN n")

    def test_division_by_zero_does_not_raise(self, small_graph_all_modes):
        # Cypher semantics: division by zero returns Null, not an error.
        # Pinning this here so a future "make it strict" change has to
        # update the test rather than slipping through.
        rows = list(small_graph_all_modes.cypher("RETURN 1 / 0 AS x"))
        assert rows[0]["x"] is None


class TestExceptionMessageHelpful:
    """Phase A.2 contract: error messages tell the user what went wrong
    AND how to fix it. These tests pin the diagnostic-quality bar."""

    def test_syntax_error_includes_caret(self):
        g = kglite.KnowledgeGraph()
        with pytest.raises(kglite.CypherSyntaxError) as exc_info:
            g.cypher("MATCH x RETURN y INVALID")
        msg = str(exc_info.value)
        # The caret excerpt should appear (format_parse_error_message
        # builds it with the offending line + caret).
        assert "^" in msg or "line" in msg.lower(), f"expected caret or line marker in message: {msg!r}"

    def test_missing_parameter_names_the_param(self):
        import pandas as pd

        g = kglite.KnowledgeGraph()
        g.add_nodes(pd.DataFrame([{"id": "x", "title": "X"}]), "T", "id", "title")
        with pytest.raises(kglite.CypherExecutionError, match=r"\$missing") as exc_info:
            g.cypher("MATCH (n:T) WHERE n.id = $missing RETURN n.id")
        # Confirm the message names the bad param.
        assert "missing" in str(exc_info.value).lower()
