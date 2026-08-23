"""Golden expected-value tests for Cypher scalar (non-aggregate) functions.

These lock the *exact current output* of each scalar-function category, so a
behaviour-preserving refactor of `evaluate_scalar_function` (splitting the
2k-line monolith into per-category submodules) cannot silently drop or
mis-route an arm.

Why not the differential corpus? `tests/test_cypher_differential.py` compares
optimised-vs-naive execution, and *both* paths call the same
`evaluate_scalar_function` — so a uniform regression from the split would pass
it silently (the harness docstring says it skips "bugs present in both paths").
Golden expected-value assertions are the correct net for shared evaluation
code. Temporal/path coverage is complemented by
`test_v0_9_03_date_functions.py` and `test_cypher_path_functions.py`.
"""

from __future__ import annotations

import pytest

import kglite

# (id, query, expected) — literal-driven (no fixture) unless marked graph/.
# Floats are compared with approx; expectations capture current behaviour.
SCALAR_CASES: list[tuple[str, str, object]] = [
    # ── string ──
    ("str_upper", "RETURN toUpper('ab') AS x", "AB"),
    ("str_lower", "RETURN toLower('AB') AS x", "ab"),
    ("str_tostring", "RETURN toString(42) AS x", "42"),
    # null in, null out — NOT the four-character string 'null'.
    ("str_tostring_null", "RETURN toString(null) AS x", None),
    ("str_tostring_null_coalesce", "RETURN coalesce(toString(null),'D') AS x", "D"),
    ("str_substring", "RETURN substring('hello',1,3) AS x", "ell"),
    ("str_replace", "RETURN replace('abcabc','a','X') AS x", "XbcXbc"),
    ("str_split", "RETURN split('a,b,c',',') AS x", ["a", "b", "c"]),
    ("str_split_two", "RETURN split('a,b',',') AS x", ["a", "b"]),
    # Empty delimiter → characters. Rust's str::split would add phantom
    # leading/trailing empties (['', 'a', '']); see test_split_empty_delimiter.
    ("str_split_empty_delim_one", "RETURN split('a','') AS x", ["a"]),
    ("str_split_empty_delim", "RETURN split('abc','') AS x", ["a", "b", "c"]),
    ("str_split_empty_subject", "RETURN split('',',') AS x", [""]),
    ("str_trim", "RETURN trim('  hi  ') AS x", "hi"),
    ("str_ltrim", "RETURN ltrim('  hi') AS x", "hi"),
    ("str_rtrim", "RETURN rtrim('hi  ') AS x", "hi"),
    ("str_left", "RETURN left('hello',3) AS x", "hel"),
    ("str_right", "RETURN right('hello',3) AS x", "llo"),
    ("str_reverse", "RETURN reverse('abc') AS x", "cba"),
    ("str_edit_dist", "RETURN text_edit_distance('kitten','sitting') AS x", 3),
    ("str_regex", "RETURN text_match_regex('abc123','[0-9]+') AS x", True),
    # ── numeric ──
    ("num_abs", "RETURN abs(-5) AS x", 5),
    ("num_ceil", "RETURN ceil(3.2) AS x", 4.0),
    ("num_floor", "RETURN floor(3.8) AS x", 3.0),
    ("num_round", "RETURN round(3.14159,2) AS x", 3.14),
    ("num_sqrt", "RETURN sqrt(16.0) AS x", 4.0),
    ("num_pow", "RETURN pow(2.0,10.0) AS x", 1024.0),
    ("num_sign", "RETURN sign(-3) AS x", -1),
    ("num_exp", "RETURN round(exp(1.0),5) AS x", 2.71828),
    ("num_ln", "RETURN round(ln(2.718281828),5) AS x", 1.0),
    ("num_log10", "RETURN log10(1000.0) AS x", 3.0),
    ("num_pi", "RETURN round(pi(),5) AS x", 3.14159),
    ("num_degrees", "RETURN round(degrees(pi()),1) AS x", 180.0),
    ("num_atan2", "RETURN round(atan2(1.0,1.0),5) AS x", 0.7854),
    # ── collection ──
    ("col_size", "RETURN size(['a','b','c']) AS x", 3),
    # size()/length() on a string count CHARACTERS, not UTF-8 bytes:
    # 'ø' is 2 bytes, each CJK char 3, the emoji 4.
    ("col_size_str_ascii", "RETURN size('hello') AS x", 5),
    ("col_size_str_nordic", "RETURN size('Tromsø') AS x", 6),
    ("col_size_str_cjk", "RETURN size('日本語') AS x", 3),
    ("col_size_str_emoji", "RETURN size('héllo🎉') AS x", 6),
    ("col_length_str_nordic", "RETURN length('Tromsø') AS x", 6),
    ("col_length_str_cjk", "RETURN length('日本語') AS x", 3),
    # The point of counting characters: size() composes with the
    # char-indexed substring()/left()/right().
    ("col_size_substring", "RETURN substring('Tromsø', size('Tromsø')-1) AS x", "ø"),
    ("col_size_left", "RETURN left('Tromsø', size('Tromsø')) AS x", "Tromsø"),
    ("col_size_right", "RETURN right('Tromsø', size('Tromsø')-5) AS x", "ø"),
    # Deliberate and parked: a string that looks like a JSON list still
    # reports its ELEMENT count, because the whole legacy collect-as-JSON
    # family (UNWIND, indexing, head/last/reverse, IN) coerces the same
    # shape. Pinned so it is not "fixed" piecemeal.
    ("col_size_bracketed_string", "RETURN size('[1,2,3]') AS x", 3),
    ("col_length_bracketed_string", "RETURN length('[1,2,3]') AS x", 3),
    ("col_head", "RETURN head([10,20]) AS x", 10),
    ("col_last", "RETURN last([10,20]) AS x", 20),
    ("col_range", "RETURN range(1,5) AS x", [1, 2, 3, 4, 5]),
    ("col_coalesce", "RETURN coalesce(null,7) AS x", 7),
    # NOTE: reverse([...]) on a list is exercised in test_reverse_list below
    # (it was a bug, since fixed).
    # ── temporal (see also test_v0_9_03_date_functions.py) ──
    ("tmp_date", "RETURN toString(date('2020-01-15')) AS x", "2020-01-15"),
    ("tmp_add_days", "RETURN toString(add_days(date('2020-01-01'),10)) AS x", "2020-01-11"),
    ("tmp_date_diff", "RETURN date_diff(date('2020-01-10'),date('2020-01-01')) AS x", 9),
    # ── spatial ──
    ("sp_point", "RETURN toString(point(1.0,2.0)) AS x", "point(1, 2)"),
    ("sp_distance", "RETURN round(distance(point(0.0,0.0),point(3.0,4.0)),5) AS x", 555098.58969),
    ("sp_latitude", "RETURN latitude(point(58.0,6.0)) AS x", 58.0),
    # ── utility (shape only for non-deterministic) ──
    ("u_uuid_len", "RETURN size(toString(randomUUID())) AS x", 36),
]


@pytest.mark.parametrize("name,query,expected", SCALAR_CASES, ids=[c[0] for c in SCALAR_CASES])
def test_scalar_function_value(name: str, query: str, expected: object) -> None:
    kg = kglite.KnowledgeGraph()
    got = kg.cypher(query).to_list()[0]["x"]
    if isinstance(expected, float):
        assert got == pytest.approx(expected), f"{name}: {got!r} != {expected!r}"
    else:
        assert got == expected, f"{name}: {got!r} != {expected!r}"


def _graph() -> kglite.KnowledgeGraph:
    kg = kglite.KnowledgeGraph()
    kg.cypher("CREATE (a:Person {name:'Alice', age:30})")
    kg.cypher("CREATE (b:Person {name:'Bob', age:25})")
    kg.cypher("MATCH (a:Person {name:'Alice'}),(b:Person {name:'Bob'}) CREATE (a)-[:KNOWS {since:2020}]->(b)")
    return kg


# ── graph functions (need a fixture) ──
GRAPH_CASES: list[tuple[str, str, object]] = [
    ("g_type", "MATCH (a:Person)-[r:KNOWS]->(b) RETURN type(r) AS x", "KNOWS"),
    ("g_labels", "MATCH (a:Person {name:'Alice'}) RETURN labels(a) AS x", ["Person"]),
    ("g_keys", "MATCH (a:Person {name:'Alice'}) RETURN keys(a) AS x", ["age", "id", "name", "title", "type"]),
    ("g_degree", "MATCH (a:Person {name:'Alice'}) RETURN degree(a) AS x", 1),
    ("g_outdegree", "MATCH (a:Person {name:'Alice'}) RETURN outDegree(a) AS x", 1),
    ("g_indegree", "MATCH (b:Person {name:'Bob'}) RETURN inDegree(b) AS x", 1),
]


@pytest.mark.parametrize("name,query,expected", GRAPH_CASES, ids=[c[0] for c in GRAPH_CASES])
def test_graph_function_value(name: str, query: str, expected: object) -> None:
    kg = _graph()
    got = kg.cypher(query).to_list()[0]["x"]
    assert got == expected, f"{name}: {got!r} != {expected!r}"


def test_inline_property_access_on_function_node() -> None:
    """Inline property access on a function-returned node/relationship must
    resolve (e.g. `startNode(r).name`), not return None. Regression for an
    expression-evaluator bug surfaced during the scalar-fn split — the bound
    form `WITH startNode(r) AS s RETURN s.name` always worked.
    """
    kg = _graph()
    assert kg.cypher("MATCH (a)-[r:KNOWS]->(b) RETURN startNode(r).name AS x").to_list()[0]["x"] == "Alice"
    assert kg.cypher("MATCH (a)-[r:KNOWS]->(b) RETURN endNode(r).name AS x").to_list()[0]["x"] == "Bob"


def test_reverse_list() -> None:
    """reverse() on a list must reverse elements, not stringify-and-char-reverse.

    Regression for a bug found while splitting evaluate_scalar_function:
    reverse() coerced its arg to a string first, so a list was JSON-stringified
    then char-reversed. reverse() on a string is covered by str_reverse above.
    """
    kg = kglite.KnowledgeGraph()
    assert kg.cypher("RETURN reverse([1,2,3]) AS x").to_list()[0]["x"] == [3, 2, 1]
    assert kg.cypher("RETURN reverse(['a','b']) AS x").to_list()[0]["x"] == ["b", "a"]
    # End-to-end: split() returns a native list, so reverse() of it reverses
    # elements (regression for the case the first reverse fix didn't reach).
    assert kg.cypher("RETURN reverse(split('a,b,c',',')) AS x").to_list()[0]["x"] == ["c", "b", "a"]
    # Consistency with head/last/size: a bracketed string is treated as a list.
    assert kg.cypher("RETURN reverse('[1, 2, 3]') AS x").to_list()[0]["x"] == [3, 2, 1]
    # A plain (non-bracketed) string still reverses characters.
    assert kg.cypher("RETURN reverse('abc') AS x").to_list()[0]["x"] == "cba"


def test_split_empty_delimiter_yields_characters() -> None:
    """split() with an empty delimiter splits into characters.

    The implementation used Rust's ``str::split``, which yields a phantom
    leading and trailing empty element for an empty pattern —
    ``split('a','')`` returned ``['', 'a', '']``. That is an artefact of the
    Rust API, not a Cypher answer. Neo4j's manual does not specify the
    empty-delimiter case, so kglite pins the per-character reading (the one
    JavaScript's ``String.split('')`` also gives) and documents it as a
    dialect note in CYPHER.md.
    """
    kg = kglite.KnowledgeGraph()
    assert kg.cypher("RETURN split('a','') AS x").to_list()[0]["x"] == ["a"]
    assert kg.cypher("RETURN split('abc','') AS x").to_list()[0]["x"] == ["a", "b", "c"]
    # Characters, not bytes — the same rule as size().
    assert kg.cypher("RETURN split('Tromsø','') AS x").to_list()[0]["x"] == list("Tromsø")
    # An empty original stays [''] whatever the delimiter.
    assert kg.cypher("RETURN split('','') AS x").to_list()[0]["x"] == [""]
    assert kg.cypher("RETURN split('',',') AS x").to_list()[0]["x"] == [""]
    # The result is a native list, so list ops compose.
    assert kg.cypher("RETURN size(split('abc','')) AS x").to_list()[0]["x"] == 3
    assert kg.cypher("RETURN head(split('abc','')) AS x").to_list()[0]["x"] == "a"


def test_tostring_null_is_null() -> None:
    """toString(null) is null, not the string 'null'.

    The old formatting made an absent value indistinguishable from a present
    one, and — worse — it survived coalesce(), which is the very call that
    exists to substitute a default for a missing value.
    """
    kg = kglite.KnowledgeGraph()
    kg.cypher("CREATE (n:Item {name:'a'})")
    assert kg.cypher("RETURN toString(null) AS x").to_list()[0]["x"] is None
    assert kg.cypher("MATCH (n:Item) RETURN toString(n.absent) AS x").to_list()[0]["x"] is None
    assert kg.cypher("RETURN toString(null) IS NULL AS x").to_list()[0]["x"] is True
    assert kg.cypher("RETURN coalesce(toString(null),'D') AS x").to_list()[0]["x"] == "D"
    # Non-null arguments still stringify.
    assert kg.cypher("RETURN toString(42) AS x").to_list()[0]["x"] == "42"
    assert kg.cypher("RETURN toString(true) AS x").to_list()[0]["x"] == "true"


def test_size_counts_characters_not_bytes() -> None:
    """size()/length() on a string count characters.

    Byte counting disagreed with substring()/left()/right(), which have
    always been char-indexed — so ``substring(s, size(s)-1)`` returned an
    empty string for any non-ASCII ``s`` instead of its last character.
    """
    kg = kglite.KnowledgeGraph()
    for text in ("Tromsø", "日本語", "héllo🎉", "abc", ""):
        assert kg.cypher("RETURN size($t) AS x", params={"t": text}).to_list()[0]["x"] == len(text)
        assert kg.cypher("RETURN length($t) AS x", params={"t": text}).to_list()[0]["x"] == len(text)
    # Composition with the char-indexed string functions.
    assert kg.cypher("RETURN substring('Tromsø', size('Tromsø')-1) AS x").to_list()[0]["x"] == "ø"
    assert kg.cypher("RETURN left('日本語', size('日本語')) AS x").to_list()[0]["x"] == "日本語"
