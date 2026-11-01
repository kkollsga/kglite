"""Absolute expected-value goldens for filtered scans and simple aggregates.

Pins the semantics of `MATCH (n:T) WHERE <pred> RETURN count/sum/min/max/avg`
before the scan kernels change: the raw-cell numeric compare in the column
filter and the aggregate tail of the fused scan. Every expectation is computed
in plain Python over the inserted data (Python's int/float comparison is exact,
like kglite's), never by asking kglite, and every test runs on all three
storage modes.

Edge values: integers around 2**53 against float literals (a cast to f64 would
call 2**53 + 1 equal to 2**53), signed zero, infinity, NaN (set through a
parameter: a DataFrame NaN loads as Null), Null cells, mixed Int64/Float64
columns, delete-then-create, writes after a mapped load, and valid-time.
"""

from __future__ import annotations

import math
import operator

import pandas as pd
import pytest

import kglite
from kglite import KnowledgeGraph

MODES = ("memory", "mapped", "disk")

TWO53 = 2**53
# Int64 column: exact-int edges around 2**53 and the i64 extremes, one Null.
INTS = [0, 5, -3, TWO53 - 1, TWO53, TWO53 + 1, TWO53 + 3, -TWO53 - 1, 2**62, 2**63 - 1, -(2**63) + 1, None]
# Float64 column: signed zero, inf, huge, one Null. Sums stay exact (binary fractions).
FLOATS = [0.5, 1.0, -0.0, 2.0, 1e300, -1.5, float(TWO53), 3.25, None, float("inf"), -float("inf"), 0.0]
N = len(INTS)
assert N == len(FLOATS)


def make_graph(mode: str, tmp_path) -> KnowledgeGraph:
    if mode == "memory":
        return KnowledgeGraph()
    if mode == "mapped":
        return KnowledgeGraph(storage="mapped")
    return KnowledgeGraph(storage="disk", path=str(tmp_path / "disk_graph"))


def load_t(graph: KnowledgeGraph) -> None:
    df = pd.DataFrame(
        {
            "id": range(N),
            "name": [f"t{i}" for i in range(N)],
            # nullable Int64: a plain int column with None would become float64
            "i": pd.array(INTS, dtype="Int64"),
            "f": FLOATS,
        }
    )
    graph.add_nodes(df, "T", "id", "name")


@pytest.fixture(scope="module", params=MODES)
def tg(request, tmp_path_factory):
    graph = make_graph(request.param, tmp_path_factory.mktemp(f"scan_{request.param}"))
    load_t(graph)
    return graph


def one(graph, query: str, **params) -> dict:
    rows = graph.cypher(query, params=params or None).to_list()
    assert len(rows) == 1, rows
    return rows[0]


# --- filter counts: every operator x int and float literals x both columns -------------------

OPS = {"<": operator.lt, "<=": operator.le, ">": operator.gt, ">=": operator.ge, "=": operator.eq, "<>": operator.ne}
# (cypher text, python value) — the same literal in both languages.
LITERALS = [
    ("0", 0),
    ("-3", -3),
    ("2", 2),
    ("9007199254740991", TWO53 - 1),
    ("9007199254740992", TWO53),
    ("9007199254740993", TWO53 + 1),
    ("9223372036854775807", 2**63 - 1),
    ("0.0", 0.0),
    ("-0.0", -0.0),
    ("0.5", 0.5),
    ("2.0", 2.0),
    ("9007199254740992.0", float(TWO53)),
    ("9007199254740994.0", float(TWO53 + 2)),
    ("9.223372036854775807e18", 9.223372036854775807e18),
    ("1e300", 1e300),
]


def expected_count(column, op, lit) -> int:
    # A Null cell never matches, `<>` included.
    return sum(1 for v in column if v is not None and OPS[op](v, lit))


@pytest.mark.parametrize("op", list(OPS))
@pytest.mark.parametrize("col,column", [("i", INTS), ("f", FLOATS)])
def test_filter_count_matches_exact_python_comparison(tg, col, column, op):
    """Int64 and Float64 columns against int and float literals, > 2**53 included."""
    for text, lit in LITERALS:
        got = one(tg, f"MATCH (n:T) WHERE n.{col} {op} {text} RETURN count(n) AS c")["c"]
        assert got == expected_count(column, op, lit), f"n.{col} {op} {text}"


def test_large_int_vs_float_literal_is_not_a_cast_to_f64(tg):
    # 2**53 + 1 as f64 is 2**53; the exact compare must still call it greater.
    assert one(tg, "MATCH (n:T) WHERE n.i > 9007199254740992.0 RETURN count(n) AS c")["c"] == sum(
        1 for v in INTS if v is not None and v > float(TWO53)
    )
    assert one(tg, "MATCH (n:T) WHERE n.i = 9007199254740992.0 RETURN count(n) AS c")["c"] == 1
    assert one(tg, "MATCH (n:T) WHERE n.i = 9007199254740993 RETURN count(n) AS c")["c"] == 1
    assert (
        one(tg, "MATCH (n:T) WHERE n.i < 9007199254740993 AND n.i >= 9007199254740992 RETURN count(n) AS c")["c"] == 1
    )


def test_signed_zero_and_float_column_vs_int_literal(tg):
    # -0.0 and 0.0 both equal the int literal 0.
    assert one(tg, "MATCH (n:T) WHERE n.f = 0 RETURN count(n) AS c")["c"] == 2
    assert one(tg, "MATCH (n:T) WHERE n.f > 1 RETURN count(n) AS c")["c"] == sum(
        1 for v in FLOATS if v is not None and v > 1
    )
    assert one(tg, "MATCH (n:T) WHERE n.f = 2 RETURN count(n) AS c")["c"] == 1
    assert one(tg, "MATCH (n:T) WHERE n.f < 9223372036854775807 RETURN count(n) AS c")["c"] == sum(
        1 for v in FLOATS if v is not None and v < 2**63 - 1
    )


def test_two_predicates_and_null_cells(tg):
    got = one(tg, "MATCH (n:T) WHERE n.i > 0 AND n.f < 2.0 RETURN count(n) AS c")["c"]
    assert got == sum(1 for a, b in zip(INTS, FLOATS) if a is not None and b is not None and a > 0 and b < 2.0)
    assert one(tg, "MATCH (n:T) WHERE n.i IS NULL RETURN count(n) AS c")["c"] == 1
    assert one(tg, "MATCH (n:T) WHERE n.f IS NOT NULL RETURN count(n) AS c")["c"] == N - 1


# --- aggregates, with and without WHERE -------------------------------------------------------


def int_sum(values):
    return sum(values)  # exact Python ints


def test_int_aggregates_are_exact_beyond_2_53(tg):
    present = [v for v in INTS if v is not None]
    row = one(
        tg,
        "MATCH (n:T) RETURN sum(n.i) AS s, min(n.i) AS mn, max(n.i) AS mx, count(n.i) AS c, count(DISTINCT n.i) AS d",
    )
    assert row["s"] == int_sum(present) and isinstance(row["s"], int)
    assert row["mn"] == min(present) and row["mx"] == max(present)
    assert row["c"] == len(present) and row["d"] == len(set(present))


def test_int_avg_is_sum_over_count(tg):
    present = [v for v in INTS if v is not None]
    got = one(tg, "MATCH (n:T) RETURN avg(n.i) AS a")["a"]
    assert got == pytest.approx(sum(present) / len(present), rel=1e-12)


def test_int_aggregates_with_where(tg):
    sel = [v for v in INTS if v is not None and TWO53 < v < 2**62]
    row = one(
        tg,
        "MATCH (n:T) WHERE n.i > 9007199254740992.0 AND n.i < 4611686018427387904 "
        "RETURN sum(n.i) AS s, min(n.i) AS mn, max(n.i) AS mx, count(DISTINCT n.i) AS d, count(n) AS c",
    )
    assert row == {"s": sum(sel), "mn": min(sel), "mx": max(sel), "d": len(set(sel)), "c": len(sel)}


def test_float_aggregates_with_inf_and_signed_zero(tg):
    # +inf and -inf are both stored, so the plain sum is NaN (inf + -inf).
    row = one(tg, "MATCH (n:T) RETURN sum(n.f) AS s, min(n.f) AS mn, max(n.f) AS mx, count(DISTINCT n.f) AS d")
    assert math.isnan(row["s"])
    assert row["mn"] == -float("inf") and row["mx"] == float("inf")
    # distinct under value equality: 0.0 and -0.0 are one value. Pinned: today's count.
    assert row["d"] == len({v for v in FLOATS if v is not None})


def test_float_aggregates_with_where(tg):
    sel = [v for v in FLOATS if v is not None and -10 < v < 100]
    row = one(
        tg,
        "MATCH (n:T) WHERE n.f > -10 AND n.f < 100 "
        "RETURN sum(n.f) AS s, min(n.f) AS mn, max(n.f) AS mx, avg(n.f) AS a, count(DISTINCT n.f) AS d",
    )
    assert row["s"] == sum(sel)
    assert row["mn"] == min(sel) and row["mx"] == max(sel)
    assert row["a"] == pytest.approx(sum(sel) / len(sel))
    assert row["d"] == len(set(sel))


def test_int_sum_overflow_raises(tg):
    # 2**62 + (2**63 - 1) leaves the i64 range: an error, not a wrapped or float result.
    with pytest.raises(kglite.CypherExecutionError, match="Integer overflow in sum"):
        tg.cypher("MATCH (n:T) WHERE n.i > 4611686018427387903 RETURN sum(n.i) AS s")


def test_aggregates_over_an_empty_selection(tg):
    row = one(
        tg,
        "MATCH (n:T) WHERE n.i > 9223372036854775807 "
        "RETURN count(n) AS c, sum(n.i) AS s, min(n.i) AS mn, max(n.i) AS mx, avg(n.i) AS a",
    )
    assert row["c"] == 0 and row["mn"] is None and row["mx"] is None and row["a"] is None
    assert row["s"] in (0, None)  # pinned below
    assert row["s"] == 0


# --- NaN cells ---------------------------------------------------------------------------------


@pytest.fixture(scope="module", params=MODES)
def nan_graph(request, tmp_path_factory):
    graph = make_graph(request.param, tmp_path_factory.mktemp(f"nan_{request.param}"))
    graph.cypher("CREATE (:Q {id: 1, f: 1.5, k: 1}), (:Q {id: 2, f: 2.5, k: 2}), (:Q {id: 3, f: 3.5, k: 3})")
    graph.cypher("CREATE (:Q {id: 4, f: 4.0, k: 4})")
    graph.cypher("MATCH (n:Q {id: 2}) SET n.f = $x", params={"x": float("nan")})
    return graph


def test_nan_cell_stored(nan_graph):
    assert math.isnan(one(nan_graph, "MATCH (n:Q {id: 2}) RETURN n.f AS f")["f"])


@pytest.mark.parametrize(
    "pred,expected",
    [
        ("n.f < 100", 3),  # NaN compares false
        ("n.f > -100", 3),
        ("n.f <= 100", 3),
        ("n.f >= -100", 3),
        ("n.f = 2.5", 0),  # the NaN row replaced 2.5
        ("n.f > 1 AND n.f < 4", 2),
        ("n.f = n.f", 3),  # NaN is not equal to itself
        # Pinned, surprising only by omission: `<>` is true for a NaN cell (IEEE != semantics).
        ("n.f <> 1.5", 3),
        ("n.f <> 1", 4),
    ],
)
def test_nan_in_filters(nan_graph, pred, expected):
    assert one(nan_graph, f"MATCH (n:Q) WHERE {pred} RETURN count(n) AS c")["c"] == expected


def test_nan_in_aggregates_is_pinned(nan_graph):
    # Pinned current behaviour, not a requirement: sum/avg poison to NaN, max ranks NaN
    # above every number (total order), min ignores it; count(DISTINCT) keeps NaN as a value.
    row = one(
        nan_graph,
        "MATCH (n:Q) RETURN sum(n.f) AS s, min(n.f) AS mn, max(n.f) AS mx, avg(n.f) AS a, "
        "count(DISTINCT n.f) AS d, count(n.f) AS c",
    )
    assert math.isnan(row["s"]) and math.isnan(row["mx"]) and math.isnan(row["a"])
    assert row["mn"] == 1.5
    assert row["d"] == 4 and row["c"] == 4


def test_nan_filtered_out_aggregate_is_clean(nan_graph):
    row = one(
        nan_graph, "MATCH (n:Q) WHERE n.f > 0 RETURN sum(n.f) AS s, min(n.f) AS mn, max(n.f) AS mx, avg(n.f) AS a"
    )
    assert row == {"s": 1.5 + 3.5 + 4.0 - 0.0, "mn": 1.5, "mx": 4.0, "a": (1.5 + 3.5 + 4.0) / 3}


# --- mixed Int64 / Float64 in one property -----------------------------------------------------


@pytest.fixture(scope="module", params=MODES)
def mixed_graph(request, tmp_path_factory):
    graph = make_graph(request.param, tmp_path_factory.mktemp(f"mixed_{request.param}"))
    graph.cypher("CREATE (:M {id: 1, x: 1}), (:M {id: 2, x: 2.5}), (:M {id: 3, x: 9007199254740993})")
    graph.cypher("CREATE (:M {id: 4, x: 4.0}), (:M {id: 5, x: -7}), (:M {id: 6})")
    return graph


MIXED = [1, 2.5, TWO53 + 1, 4.0, -7]


@pytest.mark.parametrize("op", list(OPS))
@pytest.mark.parametrize(
    "text,lit", [("2", 2), ("2.5", 2.5), ("9007199254740992.0", float(TWO53)), ("4", 4), ("-7.0", -7.0)]
)
def test_mixed_int_float_column_filters(mixed_graph, op, text, lit):
    got = one(mixed_graph, f"MATCH (n:M) WHERE n.x {op} {text} RETURN count(n) AS c")["c"]
    assert got == sum(1 for v in MIXED if OPS[op](v, lit))


def test_mixed_int_float_column_aggregates(mixed_graph):
    row = one(
        mixed_graph,
        "MATCH (n:M) RETURN min(n.x) AS mn, max(n.x) AS mx, count(DISTINCT n.x) AS d, count(n.x) AS c, sum(n.x) AS s",
    )
    assert row["mn"] == -7 and row["mx"] == TWO53 + 1
    assert row["d"] == 5 and row["c"] == 5
    # The sum mixes ints and floats; the f64 total is within one ulp of the exact one.
    assert row["s"] == pytest.approx(sum(MIXED), rel=1e-12)


def test_mixed_column_aggregates_with_where(mixed_graph):
    row = one(mixed_graph, "MATCH (n:M) WHERE n.x > 1 AND n.x < 5 RETURN sum(n.x) AS s, min(n.x) AS mn, max(n.x) AS mx")
    assert row == {"s": 2.5 + 4.0, "mn": 2.5, "mx": 4.0}


# --- the id column (the UniqueId surface) ------------------------------------------------------


def test_id_property_filters_and_aggregates(tg):
    # `n.id` of an integer-keyed type reads back as an integer in every mode; the
    # public API has no way to name the internal UniqueId variant otherwise.
    ids = list(range(N))
    assert one(tg, "MATCH (n:T) WHERE n.id > 4 RETURN count(n) AS c")["c"] == sum(1 for v in ids if v > 4)
    assert one(tg, "MATCH (n:T) WHERE n.id = 5 RETURN count(n) AS c")["c"] == 1
    assert one(tg, "MATCH (n:T) WHERE n.id >= 2.5 RETURN count(n) AS c")["c"] == sum(1 for v in ids if v >= 2.5)
    row = one(tg, "MATCH (n:T) WHERE n.id < 6 RETURN sum(n.id) AS s, min(n.id) AS mn, max(n.id) AS mx, avg(n.id) AS a")
    assert row == {"s": sum(range(6)), "mn": 0, "mx": 5, "a": sum(range(6)) / 6}


# --- delete then re-create with the same id ----------------------------------------------------


@pytest.mark.parametrize("mode", MODES)
def test_delete_then_create_same_id(mode, tmp_path):
    graph = make_graph(mode, tmp_path)
    graph.cypher("UNWIND range(1, 20) AS i CREATE (:D {id: i, x: i * 10, y: i / 2.0})")
    assert one(graph, "MATCH (n:D) WHERE n.x > 100 RETURN count(n) AS c, sum(n.x) AS s") == {
        "c": 10,
        "s": sum(range(110, 210, 10)),
    }
    graph.cypher("MATCH (n:D) WHERE n.id = 15 OR n.id = 3 DETACH DELETE n")
    graph.cypher("CREATE (:D {id: 15, x: 7, y: 99.5})")
    xs = {i: i * 10 for i in range(1, 21) if i != 3}
    xs[15] = 7
    ys = {i: i / 2.0 for i in range(1, 21) if i != 3}
    ys[15] = 99.5
    assert one(graph, "MATCH (n:D) RETURN count(n) AS c, sum(n.x) AS s, min(n.x) AS mn, max(n.x) AS mx") == {
        "c": 19,
        "s": sum(xs.values()),
        "mn": 7,
        "mx": 200,
    }
    got = one(graph, "MATCH (n:D) WHERE n.x > 100 RETURN count(n) AS c, sum(n.x) AS s")
    assert got == {"c": sum(1 for v in xs.values() if v > 100), "s": sum(v for v in xs.values() if v > 100)}
    assert one(graph, "MATCH (n:D) WHERE n.y > 50.0 RETURN count(n) AS c, max(n.y) AS m") == {"c": 1, "m": 99.5}
    assert one(graph, "MATCH (n:D) WHERE n.id = 15 RETURN n.x AS x")["x"] == 7
    assert one(graph, "MATCH (n:D) WHERE n.id = 3 RETURN count(n) AS c")["c"] == 0


# --- mapped graph: writes after load, and a store large enough to be file-backed ---------------

BIG_ROWS = 40_000  # a 40k-row i64 column is 320 KB: above the 256 KB v9 data-region threshold


@pytest.fixture(scope="module")
def big_saved(tmp_path_factory):
    graph = KnowledgeGraph()
    # scrambled so the delta encoding does not shrink the columns (see test_column_mapped.py)
    graph.cypher(
        f"UNWIND range(1, {BIG_ROWS}) AS i CREATE (:Big {{id: i, score: (i * 7919) % 40009 * 104729, ratio: i / 4.0}})"
    )
    path = str(tmp_path_factory.mktemp("big") / "g.kgl")
    graph.save(path)
    return path


def big_model():
    score = {i: (i * 7919) % 40009 * 104729 for i in range(1, BIG_ROWS + 1)}
    ratio = {i: i / 4.0 for i in range(1, BIG_ROWS + 1)}
    return score, ratio


@pytest.mark.parametrize("columns", ["mapped", "memory"])
def test_writes_after_a_load_are_visible_to_filters_and_aggregates(big_saved, columns):
    score, ratio = big_model()
    graph = kglite.load(big_saved, columns=columns)
    mapped_sections = [s for s in graph.verify_integrity()["sections"] if s["name"].startswith("col:") and s["mapped"]]
    # The file-backed path is only exercised if the large columns really are mapped.
    assert bool(mapped_sections) == (columns == "mapped")

    def check():
        sel = [v for v in score.values() if v > 2_000_000_000]
        got = one(
            graph, "MATCH (n:Big) WHERE n.score > 2000000000 RETURN count(n) AS c, sum(n.score) AS s, max(n.score) AS m"
        )
        assert got == {"c": len(sel), "s": sum(sel), "m": max(sel)}
        fsel = [v for v in ratio.values() if v < 100.0]
        got = one(graph, "MATCH (n:Big) WHERE n.ratio < 100.0 RETURN count(n) AS c, sum(n.ratio) AS s")
        assert got == {"c": len(fsel), "s": sum(fsel)}
        assert one(graph, "MATCH (n:Big) RETURN count(n) AS c, min(n.score) AS mn")["c"] == len(score)

    check()
    # Overlay writes over file-backed columns: lower a high score, raise a low one, change a float.
    graph.cypher("MATCH (n:Big {id: 1}) SET n.score = 3000000000")
    score[1] = 3_000_000_000
    graph.cypher("MATCH (n:Big) WHERE n.id = 2 SET n.ratio = 0.25, n.score = 1")
    ratio[2], score[2] = 0.25, 1
    check()
    # Creates past the loaded rows, a delete, then another create reusing an id.
    graph.cypher("CREATE (:Big {id: 900001, score: 4000000000, ratio: 1.5})")
    score[900001], ratio[900001] = 4_000_000_000, 1.5
    graph.cypher("MATCH (n:Big {id: 3}) DETACH DELETE n")
    del score[3], ratio[3]
    check()
    graph.cypher("CREATE (:Big {id: 3, score: 2500000000, ratio: 99.0})")
    score[3], ratio[3] = 2_500_000_000, 99.0
    check()
    assert one(graph, "MATCH (n:Big) WHERE n.ratio = 99.0 RETURN count(n) AS c")["c"] == sum(
        1 for v in ratio.values() if v == 99.0
    )


@pytest.mark.parametrize("mode", ["mapped"])
def test_mapped_small_store_writes_after_load(mode, tmp_path):
    base = KnowledgeGraph()
    load_t(base)
    path = str(tmp_path / "t.kgl")
    base.save(path)
    graph = kglite.load(path, storage="mapped")
    ints = list(INTS)
    graph.cypher("MATCH (n:T {id: 0}) SET n.i = 9007199254740993")
    ints[0] = TWO53 + 1
    graph.cypher("CREATE (:T {id: 100, name: 'extra', i: 9007199254740992, f: 7.5})")
    ints.append(TWO53)
    floats = [*FLOATS, 7.5]
    assert one(graph, "MATCH (n:T) WHERE n.i = 9007199254740992.0 RETURN count(n) AS c")["c"] == 1 + sum(
        1 for v in ints[:-1] if v == TWO53
    )
    assert one(graph, "MATCH (n:T) WHERE n.i > 9007199254740992 RETURN count(n) AS c")["c"] == sum(
        1 for v in ints if v is not None and v > TWO53
    )
    got = one(graph, "MATCH (n:T) WHERE n.f > 1.0 AND n.f < 10 RETURN count(n) AS c, sum(n.f) AS s")
    sel = [v for v in floats if v is not None and 1.0 < v < 10]
    assert got == {"c": len(sel), "s": sum(sel)}


# --- valid-time --------------------------------------------------------------------------------


@pytest.mark.parametrize("mode", MODES)
def test_valid_at_filtered_count_and_sum(mode, tmp_path):
    graph = make_graph(mode, tmp_path)
    rows = [
        # id, x, from, to  (half-open: valid on [from, to))
        (1, 10, "2020-01-01", "2021-01-01"),
        (2, 20, "2020-06-01", "2022-01-01"),
        (3, 30, "2021-01-01", "2023-01-01"),
        (4, 40, "2019-01-01", "2020-06-01"),
        (5, 50, "2020-01-01", "2030-01-01"),
    ]
    df = pd.DataFrame(
        {
            "id": [r[0] for r in rows],
            "name": [f"e{r[0]}" for r in rows],
            "x": [r[1] for r in rows],
            "vf": [r[2] for r in rows],
            "vt": [r[3] for r in rows],
        }
    )
    graph.add_nodes(df, "E", "id", "name")
    graph.set_temporal("E", "vf", "vt", convention="half_open")

    def expect(instant: str, lo: int):
        sel = [x for _, x, vf, vt in rows if vf <= instant < vt and x > lo]
        return {"c": len(sel), "s": sum(sel)}

    for instant in ("2020-03-01", "2020-06-01", "2021-01-01", "2022-06-01"):
        q = f"FOR VALID_TIME AS OF date('{instant}') MATCH (n:E) WHERE n.x > 15 RETURN count(n) AS c, sum(n.x) AS s"
        assert one(graph, q) == expect(instant, 15), instant
    q_all = "FOR VALID_TIME ALL MATCH (n:E) WHERE n.x > 15 RETURN count(n) AS c, sum(n.x) AS s"
    assert one(graph, q_all) == {"c": 4, "s": 140}
