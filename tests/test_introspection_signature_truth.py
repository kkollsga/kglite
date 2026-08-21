"""describe()'s advertised signatures must be callable.

``describe(fluent=...)`` is the agent-facing API reference: an agent reads a
``sig="..."`` string or an ``<ex>`` example and calls it verbatim.  Every name
it advertises therefore has to exist, and every argument it names has to be a
real parameter of that method — otherwise the agent gets a ``TypeError`` (or,
worse, silently passes an argument in the wrong position).

This is a truth gate, not a style gate.  It compares the strings emitted by
``crates/kglite/src/graph/introspection/topics.rs`` against
``inspect.signature`` of the installed extension.  Anything it flags is a
documented signature that does not exist.
"""

from __future__ import annotations

import html
import inspect
import re

import pytest

import kglite

# Every fluent topic write_fluent_topics() accepts (topics.rs FLUENT_TOPIC_LIST).
FLUENT_TOPICS = [
    "selection",
    "traversal",
    "compare",
    "spatial",
    "temporal",
    "retrieval",
    "statistics",
    "algorithms",
    "vectors",
    "timeseries",
    "mutation",
    "loading",
    "export",
    "indexes",
    "set_ops",
    "subgraph",
    "schema",
    "transactions",
]

# Namespaces an advertised call can legitimately resolve against.
NAMESPACES = {
    "KnowledgeGraph": kglite.KnowledgeGraph,
    "ResultView": kglite.ResultView,
    "Session": kglite.Session,
    "Transaction": kglite.Transaction,
    "FrozenGraph": kglite.FrozenGraph,
    "Agg": kglite.Agg,
    "Spatial": kglite.Spatial,
    "kglite": kglite,
}

# Receiver prefixes used in the docs for readability ("tx.cypher(...)").
RECEIVER_PREFIXES = {"kglite", "graph", "g", "tx", "view", "result", "session"}

IDENT = re.compile(r"^[A-Za-z_]\w*$")
CALL_HEAD = re.compile(r"^([A-Za-z_][\w.]*)\((.*)\)$", re.S)
KWARG = re.compile(r"^([A-Za-z_]\w*)\s*=(?!=)")


@pytest.fixture(scope="module")
def describe_output() -> str:
    """Every describe() track that carries API signatures."""
    graph = kglite.KnowledgeGraph()
    chunks = [
        graph.describe(fluent=True),
        graph.describe(cypher=True),
    ]
    chunks += [graph.describe(fluent=[topic]) for topic in FLUENT_TOPICS]
    return "\n".join(chunks)


def _split_args(args: str) -> list[str]:
    """Split a signature's argument list on top-level commas."""
    depth = 0
    current = ""
    pieces = []
    for ch in args:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            pieces.append(current)
            current = ""
        else:
            current += ch
    if current.strip():
        pieces.append(current)
    return [p.strip() for p in pieces]


def _resolve(name: str) -> list[tuple[str, object]]:
    """All (namespace, attribute) pairs providing `name`."""
    if "." in name:
        head, _, tail = name.rpartition(".")
        if head in NAMESPACES:
            obj = getattr(NAMESPACES[head], tail, None)
            return [(head, obj)] if obj is not None else []
        if head in RECEIVER_PREFIXES:
            name = tail
        else:
            return []
    return [(ns, getattr(obj, name)) for ns, obj in NAMESPACES.items() if getattr(obj, name, None)]


def _parameters(obj) -> list[str] | None:
    """Real parameter names, or None when the object has no readable signature."""
    try:
        params = inspect.signature(obj).parameters
    except (TypeError, ValueError):
        return None
    return [name for name, p in params.items() if name != "self" and p.kind is not inspect.Parameter.VAR_KEYWORD]


def _max_positional(obj) -> int | None:
    """How many arguments the method accepts positionally (None = unbounded/unknown)."""
    try:
        params = inspect.signature(obj).parameters
    except (TypeError, ValueError):
        return None
    count = 0
    for name, p in params.items():
        if name == "self":
            continue
        if p.kind is inspect.Parameter.VAR_POSITIONAL:
            return None
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD):
            count += 1
    return count


def _is_subsequence(advertised: list[str], real: list[str]) -> bool:
    it = iter(real)
    return all(name in it for name in advertised)


def _check_call(name: str, args: str) -> str | None:
    """Return a failure description, or None when the advertised call is real."""
    targets = _resolve(name)
    if not targets:
        return f"{name}(...) — no such method on any kglite class"

    pieces = _split_args(args)
    positional = [p for p in pieces if IDENT.match(p)]
    keywords = [m.group(1) for m in (KWARG.match(p) for p in pieces) if m]

    problems = []
    for ns, obj in targets:
        real = _parameters(obj)
        if real is None:  # builtin without introspectable signature — cannot judge
            return None
        unknown_kw = [k for k in keywords if k not in real]
        unknown_pos = [p for p in positional if p not in real]
        if not unknown_kw and not unknown_pos and _is_subsequence(positional, real):
            return None
        detail = []
        if unknown_kw:
            detail.append(f"unknown keyword(s) {unknown_kw}")
        if unknown_pos:
            detail.append(f"unknown argument(s) {unknown_pos}")
        if not detail:
            detail.append("arguments advertised out of order")
        problems.append(f"{ns}.{name}{tuple(real)}: " + ", ".join(detail))
    return f"{name}({args}) — " + " | ".join(problems)


def test_advertised_signatures_are_callable(describe_output):
    """Every sig="..." in describe() names a real method with real arguments."""
    failures = []
    for raw in sorted(set(re.findall(r'sig="([^"]*)"', describe_output))):
        text = html.unescape(raw)
        for part in text.split(" / "):
            part = part.strip()
            if not part:
                continue
            head = CALL_HEAD.match(part)
            if not head:
                # A bare name in a "a(x) / b / c" listing — must still exist.
                if IDENT.match(part) and not _resolve(part):
                    failures.append(f"{part} — no such method on any kglite class")
                continue
            problem = _check_call(head.group(1), head.group(2))
            if problem:
                failures.append(problem)
    assert not failures, "describe() advertises signatures that do not exist:\n" + "\n".join(failures)


def test_advertised_examples_use_real_arguments(describe_output):
    """Keyword arguments used in <ex> examples are real parameters."""
    failures = []
    for example in re.findall(r"<ex[^>]*>(.*?)</ex>", describe_output, re.S):
        code = html.unescape(example).strip()
        for match in re.finditer(r"\.([A-Za-z_]\w*)\(", code):
            name = match.group(1)
            depth = 1
            index = match.end()
            buf = ""
            while index < len(code) and depth:
                ch = code[index]
                if ch in "([{":
                    depth += 1
                elif ch in ")]}":
                    depth -= 1
                    if depth == 0:
                        break
                buf += ch
                index += 1
            pieces = _split_args(buf)
            keywords = [m.group(1) for m in (KWARG.match(p) for p in pieces) if m]
            positional = len([p for p in pieces if p and not KWARG.match(p)])
            targets = _resolve(name)
            if not targets:
                # Not a kglite method (pandas, dict, …) — nothing to check against.
                continue
            problem = None
            for _ns, obj in targets:
                real = _parameters(obj)
                if real is None:
                    problem = None
                    break
                missing = [k for k in keywords if k not in real]
                slots = _max_positional(obj)
                too_many = slots is not None and positional > slots
                if not missing and not too_many:
                    problem = None
                    break
                detail = []
                if missing:
                    detail.append(f"has no keyword(s) {missing}")
                if too_many:
                    detail.append(f"takes {slots} positional argument(s), example passes {positional}")
                problem = f"{code} — {name}{tuple(real)} " + " and ".join(detail)
            if problem:
                failures.append(problem)
    assert not failures, "describe() examples pass arguments that do not exist:\n" + "\n".join(failures)


def test_shortest_path_family_is_undirected_by_default():
    """The advertised 'undirected by default' semantics are the real ones.

    describe() and the graph-algorithms guide both state that the fluent
    shortest-path family ignores edge direction unless ``direction=`` says
    otherwise.  If that default ever changes, the docs saying so become the
    fiction this module exists to prevent.
    """
    pd = pytest.importorskip("pandas")
    graph = kglite.KnowledgeGraph()
    graph.add_nodes(pd.DataFrame({"id": [1, 2, 3], "name": ["A", "B", "C"]}), "City", "id", "name")
    graph.add_connections(pd.DataFrame({"s": [1, 2], "t": [2, 3]}), "ROAD", "City", "s", "City", "t")

    assert graph.shortest_path_length("City", 1, "City", 3) == 2
    assert graph.shortest_path_length("City", 3, "City", 1) == 2
    assert graph.shortest_path("City", 3, "City", 1)["length"] == 2
    assert graph.shortest_path_ids("City", 3, "City", 1) == [3, 2, 1]
    assert graph.are_connected("City", 3, "City", 1)
    assert graph.shortest_path_lengths_batch("City", [(3, 1)]) == [2]


def test_every_family_member_takes_the_advertised_filters():
    """describe() advertises the same filters on all seven members — really.

    ``shortest_path_length`` / ``shortest_path_lengths_batch`` /
    ``are_connected`` gained ``connection_types`` / ``via_types`` /
    ``direction`` / ``timeout_ms`` in 0.16.6; before that the docs describing
    them as filter-less were the truth and this test pinned the gap.  It now
    pins the opposite: the arguments exist *and* change the answer, so a stub
    or describe() string that advertises them cannot be fiction again.
    """
    pd = pytest.importorskip("pandas")
    graph = kglite.KnowledgeGraph()
    graph.add_nodes(pd.DataFrame({"id": [1, 2, 3], "name": ["A", "B", "C"]}), "City", "id", "name")
    graph.add_nodes(pd.DataFrame({"id": [10], "name": ["Hub"]}), "Port", "id", "name")
    graph.add_connections(pd.DataFrame({"s": [1], "t": [2]}), "ROAD", "City", "s", "City", "t")
    # City 1 and City 3 meet only at the Port, and only by FERRY.
    graph.add_connections(pd.DataFrame({"s": [1, 3], "t": [10, 10]}), "FERRY", "City", "s", "Port", "t")

    for call in (
        lambda **kw: graph.shortest_path_length("City", 1, "City", 3, **kw),
        lambda **kw: graph.shortest_path_lengths_batch("City", [(1, 3)], **kw)[0],
        lambda **kw: 2 if graph.are_connected("City", 1, "City", 3, **kw) else None,
    ):
        assert call() == 2, "the Port route exists by default"
        assert call(connection_types=["ROAD"]) is None, "connection_types ignored"
        assert call(via_types=["City"]) is None, "via_types ignored"
        assert call(direction="incoming") is None, "direction ignored"
        assert call(timeout_ms=5000) == 2, "timeout_ms rejected"

    with pytest.raises(Exception, match="Invalid direction"):
        graph.shortest_path_length("City", 1, "City", 3, direction="sideways")
