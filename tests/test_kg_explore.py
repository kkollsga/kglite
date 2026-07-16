"""KnowledgeGraph.explore() — one-call codebase exploration.

Lexically ranks Function/Class/Interface nodes against a free-text
query, takes the top entries, 2-hop traverses CALLS/USES_TYPE/HAS_METHOD/
DEFINES/REFERENCES_FN, and returns a markdown report.

Fixtures are hand-built code-schema graphs — Function/Class nodes carrying
the properties explore() ranks and renders (``name``/title, ``file_path``,
``line_number``, ``end_line``, ``signature``, ``docstring``) plus CALLS
edges — so the tests exercise the ranker/traversal/renderer without paying
a parser to produce them.
"""

import pathlib

from kglite import KnowledgeGraph


def _build(entities: list[dict], calls: list[tuple[str, str]] | None = None) -> KnowledgeGraph:
    """Build a code-schema graph from entity specs.

    Each entity dict needs ``id`` and ``name``; ``kind`` (default
    ``Function``), ``file_path``, ``line_number``, ``end_line``,
    ``signature`` and ``docstring`` are optional. ``calls`` are
    ``(source_id, target_id)`` CALLS edges.
    """
    g = KnowledgeGraph()
    for ent in entities:
        props = {
            "id": ent["id"],
            "name": ent["name"],
            "file_path": ent.get("file_path"),
            "line_number": ent.get("line_number"),
            "end_line": ent.get("end_line"),
            "signature": ent.get("signature"),
            "docstring": ent.get("docstring"),
        }
        # Only emit properties that were supplied — a missing property is
        # a null read in Cypher/explore, exactly like a sparse code graph.
        keys = [k for k, v in props.items() if v is not None]
        assignments = ", ".join(f"{k}: ${k}" for k in keys)
        kind = ent.get("kind", "Function")
        g.cypher(
            f"CREATE (n:{kind} {{{assignments}}})",
            params={k: props[k] for k in keys},
        )
    for src, dst in calls or []:
        g.cypher(
            "MATCH (a {id: $src}), (b {id: $dst}) CREATE (a)-[:CALLS]->(b)",
            params={"src": src, "dst": dst},
        )
    return g


def test_explore_finds_entry_points_by_name():
    g = _build(
        [
            {
                "id": "auth.authenticate",
                "name": "authenticate",
                "file_path": "auth.py",
                "line_number": 1,
                "end_line": 3,
                "signature": "def authenticate(user, password)",
                "docstring": "Verify user credentials.",
            },
            {
                "id": "auth._check_password",
                "name": "_check_password",
                "file_path": "auth.py",
                "line_number": 5,
                "end_line": 6,
            },
            {
                "id": "auth.unrelated_helper",
                "name": "unrelated_helper",
                "file_path": "auth.py",
                "line_number": 8,
                "end_line": 9,
            },
        ],
        calls=[("auth.authenticate", "auth._check_password")],
    )
    md = g.explore("authenticate", max_entities=5, max_depth=1, include_source=False)
    assert "## Entry points" in md, md
    assert "authenticate" in md, md
    # The query-name match should rank above 'unrelated_helper'.
    auth_idx = md.find("authenticate")
    unrelated_idx = md.find("unrelated_helper")
    assert auth_idx >= 0
    if unrelated_idx >= 0:
        assert auth_idx < unrelated_idx, "authenticate should rank first"


def test_explore_traverses_to_neighbors():
    """Related functions reachable via CALLS show up under Related."""
    g = _build(
        [
            {
                "id": "auth.authenticate",
                "name": "authenticate",
                "file_path": "auth.py",
                "line_number": 1,
                "end_line": 2,
            },
            {
                "id": "auth._check_password",
                "name": "_check_password",
                "file_path": "auth.py",
                "line_number": 4,
                "end_line": 5,
            },
        ],
        calls=[("auth.authenticate", "auth._check_password")],
    )
    md = g.explore("authenticate", max_entities=3, max_depth=2, include_source=False)
    # Both the entry point and its CALLS neighbor should appear.
    assert "_check_password" in md, md


def test_explore_empty_graph_returns_no_match_message():
    """A graph with no matching entities yields a clear 'no match' message."""
    g = _build([{"id": "lib.add", "name": "add", "file_path": "lib.py", "line_number": 1, "end_line": 2}])
    md = g.explore("authenticate", max_entities=10)
    assert "No matching" in md or "0" in md, md


def test_explore_empty_query_handled():
    """Empty query is a benign no-op, not an error."""
    g = _build([{"id": "lib.f", "name": "f", "file_path": "lib.py", "line_number": 1, "end_line": 1}])
    md = g.explore("", max_entities=5)
    assert "empty query" in md.lower(), md


def test_explore_include_source_emits_code(tmp_path):
    """With include_source=True (default), source slices are emitted."""
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "auth.py").write_text(
        "def authenticate(user, password):\n    '''Verify credentials.'''\n    return password == 'secret'\n"
    )
    g = _build(
        [
            {
                "id": "auth.authenticate",
                "name": "authenticate",
                "file_path": "auth.py",
                "line_number": 1,
                "end_line": 3,
                "signature": "def authenticate(user, password)",
            }
        ]
    )
    md = g.explore("authenticate", max_entities=3, include_source=True, source_roots=[str(pkg)])
    assert "## Source" in md, md
    # The authenticate function body should appear in a fenced block.
    assert "def authenticate" in md, md


def test_explore_include_source_false_omits_source():
    g = _build(
        [
            {
                "id": "auth.authenticate",
                "name": "authenticate",
                "file_path": "auth.py",
                "line_number": 1,
                "end_line": 1,
            }
        ]
    )
    md = g.explore("authenticate", include_source=False, source_roots=[str(pathlib.Path.cwd())])
    assert "## Source" not in md, md


def test_explore_ranks_signature_match():
    """A name not matching the query but signature substring matching still surfaces."""
    g = _build(
        [
            {
                "id": "lib.lookup",
                "name": "lookup",
                "file_path": "lib.py",
                "line_number": 3,
                "end_line": 4,
                "signature": "def lookup(key: str) -> Optional[str]",
            },
            {
                "id": "lib.add",
                "name": "add",
                "file_path": "lib.py",
                "line_number": 6,
                "end_line": 7,
                "signature": "def add(a: int, b: int) -> int",
            },
        ]
    )
    md = g.explore("Optional", max_entities=3, include_source=False)
    # 'lookup' has 'Optional' in its return-type signature.
    assert "lookup" in md, md


def test_explore_on_non_code_graph_returns_no_match():
    """Calling explore() on a graph with no code-tree node types returns the
    'no match' message rather than erroring."""
    kg = KnowledgeGraph()
    md = kg.explore("anything")
    assert "No matching" in md or "0" in md, md
