"""Randomized write-sequence differential for the column kernel.

The kernel (`columnar_scan_aggregate`) walks raw column cells and the tombstone
vector, so its risk is state, not syntax: whatever sequence of writes produced
the store, the answer must equal the generic fused scan's. Hypothesis builds one
node type with Int64/Float64 properties (nulls, NaN, signed zero, 2**53 edges,
i64 extremes), applies a random sequence of CREATE / DELETE / SET / REMOVE /
delete-then-recreate / transaction commit and rollback / save-and-reload, and
after every step runs kernel-eligible queries with the pass on and off.

Equality is exact (type and bit pattern, NaN equal to NaN); an error must be
the same error on both sides. A final test asserts the kernel really answered
in a meaningful share of the examples, so the comparison cannot go vacuous.
"""

from __future__ import annotations

import math

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
import pytest

import kglite
from kglite import KnowledgeGraph

KERNEL_PASS = "columnar_scan_aggregate"

IDS = list(range(8))
INTS = st.one_of(
    st.none(),
    st.integers(-5, 12),
    st.sampled_from([2**53 - 1, 2**53, 2**53 + 1, -(2**53) - 1, 2**62, 2**63 - 1, -(2**63) + 1]),
)
FLOATS = st.one_of(
    st.none(),
    st.sampled_from([0.0, -0.0, 0.5, -1.5, 2.0, 1e300, float("inf"), -float("inf"), float("nan")]),
    st.floats(-20, 20, allow_nan=False),
)
ID = st.sampled_from(IDS)

SIMPLE_OPS = st.one_of(
    st.tuples(st.just("create"), ID, INTS, FLOATS),
    st.tuples(st.just("delete"), ID),
    st.tuples(st.just("set"), ID, INTS, FLOATS),
    st.tuples(st.just("remove"), ID, st.sampled_from(["a", "b"])),
    st.tuples(st.just("recreate"), ID, INTS, FLOATS),
    st.tuples(st.just("bulk_delete"), st.integers(-5, 12)),
)
OPS = st.one_of(
    SIMPLE_OPS,
    st.tuples(st.just("tx"), st.lists(SIMPLE_OPS, min_size=1, max_size=3), st.booleans()),
    st.tuples(st.just("reload"), st.sampled_from(["memory", "mapped"])),
)

QUERIES = [
    "MATCH (n:T) RETURN count(n) AS c",
    "MATCH (n:T) RETURN count(n.a) AS c, sum(n.a) AS s, min(n.a) AS lo, max(n.a) AS hi, avg(n.a) AS m",
    "MATCH (n:T) RETURN min(n.b) AS lo, max(n.b) AS hi, count(n.b) AS c",
    "MATCH (n:T) RETURN sum(n.b) AS s, avg(n.b) AS m",
    "MATCH (n:T) WHERE n.a > -3 RETURN count(n) AS c, sum(n.a) AS s",
    "MATCH (n:T) WHERE n.a >= 2 AND n.a < 9 RETURN count(n) AS c, min(n.a) AS lo, max(n.a) AS hi",
    "MATCH (n:T) WHERE n.b > -1.0 RETURN count(n) AS c, max(n.b) AS hi",
    "MATCH (n:T) WHERE n.b <= 0.5 AND n.a <> 3 RETURN count(n) AS c, sum(n.a) AS s",
    "MATCH (n:T) WHERE n.a > 9007199254740992.0 RETURN count(n) AS c, max(n.a) AS hi",
    "MATCH (n:T {a: 5}) RETURN count(n) AS c, max(n.b) AS hi",
]
# The queries whose PROFILE label is checked for the non-vacuity meter.
PROBES = [QUERIES[1], QUERIES[5], QUERIES[4]]

STATS = {"examples": 0, "with_rows": 0, "probes": 0, "kernel": 0}


def canon(value):
    if isinstance(value, float):
        return ("f", "nan" if math.isnan(value) else repr(value))
    if isinstance(value, dict):
        return tuple((k, canon(v)) for k, v in sorted(value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(canon(v) for v in value)
    return (type(value).__name__, value)


def run(graph, query, off: bool):
    kwargs = {"disabled_passes": [KERNEL_PASS]} if off else {}
    try:
        return canon(graph.cypher(query, **kwargs).to_list())
    except kglite.CypherExecutionError as e:
        return ("error", str(e))


def apply_simple(target, op):
    kind = op[0]
    if kind in ("create", "recreate"):
        if kind == "recreate":
            target.cypher("MATCH (n:T {k: $id}) DETACH DELETE n", params={"id": op[1]})
        props = {"id": op[1]}
        sets = ["k: $id"]
        for key, val in (("a", op[2]), ("b", op[3])):
            if val is not None:
                props[key] = val
                sets.append(f"{key}: ${key}")
        target.cypher(f"CREATE (:T {{{', '.join(sets)}}})", params=props)
    elif kind == "delete":
        target.cypher("MATCH (n:T {k: $id}) DETACH DELETE n", params={"id": op[1]})
    elif kind == "set":
        target.cypher("MATCH (n:T {k: $id}) SET n.a = $a, n.b = $b", params={"id": op[1], "a": op[2], "b": op[3]})
    elif kind == "remove":
        target.cypher(f"MATCH (n:T {{k: $id}}) REMOVE n.{op[2]}", params={"id": op[1]})
    elif kind == "bulk_delete":
        target.cypher("MATCH (n:T) WHERE n.a > $t DETACH DELETE n", params={"t": op[1]})


def new_graph(mode):
    return KnowledgeGraph(storage="mapped") if mode == "mapped" else KnowledgeGraph()


def check(graph, probe: bool):
    for q in QUERIES:
        on, off = run(graph, q, False), run(graph, q, True)
        assert on == off, f"{q}\n kernel on : {on}\n kernel off: {off}"
    if probe:
        STATS["probes"] += 1
        label = graph.cypher("PROFILE " + PROBES[1]).profile[0]["clause"]
        STATS["kernel"] += label.endswith("[column kernel]")


def drive(mode, seed_rows, ops, tmp_dir):
    graph = new_graph(mode)
    graph.cypher("CREATE (:T {k: -1, a: 1, b: 1.0})")  # types the columns before any sequence
    for i, a, b in seed_rows:
        apply_simple(graph, ("create", i, a, b))
    check(graph, probe=True)
    for step, op in enumerate(ops):
        if op[0] == "tx":
            tx = graph.begin()
            for inner in op[1]:
                apply_simple(tx, inner)
            tx.commit() if op[2] else tx.rollback()
        elif op[0] == "reload":
            path = str(tmp_dir / f"g{step}.kgl")
            graph.save(path)
            graph = kglite.load(path, storage=op[1])
        else:
            apply_simple(graph, op)
        check(graph, probe=step == len(ops) - 1)
    STATS["examples"] += 1
    STATS["with_rows"] += graph.cypher("MATCH (n:T) RETURN count(n) AS c").to_list()[0]["c"] > 0


@pytest.mark.parametrize("mode", ["memory", "mapped"])
def test_kernel_matches_generic_scan_through_random_write_sequences(mode, tmp_path_factory):
    before = dict(STATS)

    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(
        seed_rows=st.lists(st.tuples(ID, INTS, FLOATS), max_size=6),
        ops=st.lists(OPS, min_size=1, max_size=10),
    )
    def sequence(seed_rows, ops):
        drive(mode, seed_rows, ops, tmp_path_factory.mktemp("seq"))

    sequence()
    ran = STATS["examples"] - before["examples"]
    probes = STATS["probes"] - before["probes"]
    kernel = STATS["kernel"] - before["kernel"]
    assert ran >= 20, ran
    assert (STATS["with_rows"] - before["with_rows"]) * 2 >= ran, "most examples must end with rows"
    # Non-vacuity: the probe query is kernel-eligible on every example, so the
    # kernel must have answered nearly all of them (declines would be silent).
    assert kernel * 10 >= probes * 9, f"kernel answered {kernel}/{probes} probes"
