#!/usr/bin/env python3
"""On-demand, independently authored Cypher differential check vs Neo4j.

The regular test suite remains Docker-free.  This command exports KGLite's own
fixtures to an explicitly selected Neo4j database and compares query results.
It is an empirical compatibility oracle, not an openCypher TCK runner: it does
not consume, translate, vendor, or depend on upstream conformance artifacts.

Run ``make neo4j-up``, then ``make neo4j-conformance``.  Exit status is zero
only when every non-skipped case agrees.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import datetime as dt
import math
from pathlib import Path
import sys
from typing import Any, Callable, Iterable

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tests"))

from conftest import (  # type: ignore  # noqa: E402
    build_file_imports_graph,
    build_multi_label_graph,
    build_small_graph,
    build_social_graph,
)
from test_cypher_differential import DIFFERENTIAL_QUERIES  # type: ignore  # noqa: E402

import kglite  # noqa: E402

FIXTURE_BUILDERS: dict[str, Callable[[], Any]] = {
    "small_graph": build_small_graph,
    "social_graph": build_social_graph,
    "file_imports_graph": build_file_imports_graph,
    "multi_label_graph": build_multi_label_graph,
}

INTENTIONAL_DIVERGENCES: dict[str, str] = {}

KGLITE_ONLY_MARKERS = (
    "kglite.",
    "refresh_stats",
    "text_score",
    "vector_score",
    "FORMAT CSV",
    "affected_tests",
)


@dataclass(frozen=True)
class QueryCase:
    """A comparison case with an explicit result and error contract."""

    name: str
    fixture: str
    query: str
    params: dict[str, Any] | None = None
    ordered: bool | None = None
    expected_error: str | None = None
    side_effect_query: str | None = None
    isolated: bool = False

    @classmethod
    def from_entry(cls, entry: QueryCase | tuple[str, str, str, dict[str, Any] | None]) -> QueryCase:
        if isinstance(entry, cls):
            return entry
        name, fixture, query, params = entry
        return cls(name, fixture, query, params)

    @property
    def order_sensitive(self) -> bool:
        if self.ordered is not None:
            return self.ordered
        return "ORDER BY" in self.query.upper()


@dataclass(frozen=True)
class QueryResult:
    columns: tuple[str, ...]
    rows: list[dict[str, Any]]


@dataclass(frozen=True)
class Failure:
    case: QueryCase
    reason: str
    kglite: Any = None
    neo4j: Any = None


@dataclass
class RunReport:
    counters: Counter[str] = field(default_factory=Counter)
    failures: list[Failure] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.failures and not self.counters["execution_error"]


def _is_kglite_only(query: str) -> bool:
    lowered = query.lower()
    return any(marker.lower() in lowered for marker in KGLITE_ONLY_MARKERS)


def _class_name(value: Any) -> str:
    return type(value).__name__.lower()


def _properties(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    try:
        return dict(value)
    except (TypeError, ValueError):
        return {}


def _canonical_map(value: Mapping[Any, Any]) -> tuple[Any, ...]:
    items = [(_canonical(k), _canonical(v)) for k, v in value.items()]
    return tuple(sorted(items, key=repr))


def _canonical(value: Any) -> Any:
    """Return a typed, immutable representation without importing Neo4j.

    Neo4j graph/time/spatial values are recognized by their public attributes,
    which keeps this module importable in the Docker-free test environment.
    KGLite's documented node/relationship/path dictionaries are normalized to
    the same shapes.  Engine-local element IDs are intentionally excluded.
    """

    if value is None:
        return ("null",)
    if isinstance(value, bool):
        return ("bool", value)
    if isinstance(value, int):
        return ("integer", value)
    if isinstance(value, float):
        if math.isnan(value):
            return ("float", "nan")
        if math.isinf(value):
            return ("float", "inf" if value > 0 else "-inf")
        return ("float", value.hex())
    if isinstance(value, str):
        return ("string", value)
    if isinstance(value, (bytes, bytearray, memoryview)):
        return ("bytes", bytes(value))

    if isinstance(value, dt.datetime):
        return ("datetime", value.isoformat())
    if isinstance(value, dt.date):
        return ("date", value.isoformat())
    if isinstance(value, dt.time):
        return ("time", value.isoformat())
    if isinstance(value, dt.timedelta):
        return ("duration", 0, value.days, value.seconds, value.microseconds * 1000)

    cls = _class_name(value)
    if cls in {"date", "time", "datetime", "localdate", "localtime", "localdatetime"}:
        iso = value.iso_format() if hasattr(value, "iso_format") else value.isoformat()
        return (cls.replace("local", "local_"), iso)
    if cls == "duration" and all(hasattr(value, attr) for attr in ("months", "days", "seconds")):
        return ("duration", value.months, value.days, value.seconds, getattr(value, "nanoseconds", 0))
    if cls == "point" or (hasattr(value, "srid") and hasattr(value, "x") and hasattr(value, "y")):
        coords = (value.x, value.y) + ((value.z,) if getattr(value, "z", None) is not None else ())
        return ("point", int(value.srid), tuple(_canonical(v) for v in coords))

    # Neo4j Node and Relationship implement Mapping, so their public graph
    # attributes must be checked before the generic map branch.
    if hasattr(value, "labels"):
        return ("node", tuple(sorted(str(label) for label in value.labels)), _canonical_map(_properties(value)))
    if hasattr(value, "start_node") and hasattr(value, "end_node"):
        relationship_type = getattr(value, "type", type(value).__name__)
        return ("relationship", str(relationship_type), _canonical_map(_properties(value)))

    if isinstance(value, Mapping):
        keys = set(value)
        if {"nodes", "relationships"} <= keys:
            return (
                "path",
                tuple(_canonical(v) for v in value["nodes"]),
                tuple(_canonical(v) for v in value["relationships"]),
            )
        if {"labels", "properties"} <= keys:
            return (
                "node",
                tuple(sorted(str(label) for label in value["labels"])),
                _canonical_map(value["properties"]),
            )
        if {"type", "properties"} <= keys and ({"start", "end"} <= keys or "id" in keys):
            return ("relationship", str(value["type"]), _canonical_map(value["properties"]))
        return ("map", _canonical_map(value))

    if cls == "path" and hasattr(value, "nodes") and hasattr(value, "relationships"):
        return (
            "path",
            tuple(_canonical(v) for v in value.nodes),
            tuple(_canonical(v) for v in value.relationships),
        )
    if cls == "relationship" or (hasattr(value, "start_node") and hasattr(value, "end_node")):
        relationship_type = getattr(value, "type", type(value).__name__)
        return ("relationship", str(relationship_type), _canonical_map(_properties(value)))
    if cls == "node" or hasattr(value, "labels"):
        return ("node", tuple(sorted(str(label) for label in value.labels)), _canonical_map(_properties(value)))

    if isinstance(value, set | frozenset):
        return ("set", tuple(sorted((_canonical(v) for v in value), key=repr)))
    if isinstance(value, Sequence):
        return ("list", tuple(_canonical(v) for v in value))

    raise TypeError(f"unsupported result value {type(value).__module__}.{type(value).__qualname__}")


def _normalise(result: QueryResult, order_sensitive: bool) -> tuple[tuple[str, ...], list[tuple[Any, ...]]]:
    rows = [tuple(_canonical(row[column]) for column in result.columns) for row in result.rows]
    if not order_sensitive:
        rows.sort(key=repr)
    return result.columns, rows


def _run_kglite(graph: Any, query: str, params: dict[str, Any] | None) -> QueryResult:
    result = graph.cypher(query, params=params)
    return QueryResult(tuple(result.columns), result.to_list())


def _run_neo4j(driver: Any, database: str, query: str, params: dict[str, Any] | None) -> QueryResult:
    with driver.session(database=database) as session:
        result = session.run(query, **(params or {}))
        columns = tuple(result.keys())
        return QueryResult(columns, [dict(record) for record in result])


def _error_category(error: Exception) -> str:
    text = " ".join(
        str(part) for part in (getattr(error, "code", ""), type(error).__name__, str(error)) if part
    ).lower()
    for category, markers in (
        ("syntax", ("syntax", "parse")),
        ("semantic", ("semantic", "undefined variable", "unknown variable")),
        ("constraint", ("constraint", "already exists")),
        ("type", ("typeerror", "type error", "typemismatch", "invalid type")),
        ("timeout", ("timeout", "timed out")),
        ("cancelled", ("cancel",)),
    ):
        if any(marker in text for marker in markers):
            return category
    return "runtime"


def _capture(call: Callable[[], QueryResult]) -> tuple[QueryResult | None, Exception | None]:
    try:
        return call(), None
    except Exception as error:  # comparison harness must record both engines
        return None, error


def run_conformance(
    entries: Iterable[QueryCase | tuple[str, str, str, dict[str, Any] | None]],
    *,
    builders: Mapping[str, Callable[[], Any]],
    driver: Any,
    database: str,
    export_graph: Callable[[Any], None],
    intentional_divergences: Mapping[str, str] | None = None,
    filter_text: str | None = None,
    verbose: bool = False,
) -> RunReport:
    """Compare cases, reactivating a fixture whenever the fixture changes."""

    report = RunReport()
    active_fixture: str | None = None
    graph: Any = None
    divergences = intentional_divergences or {}

    for entry in entries:
        case = QueryCase.from_entry(entry)
        if filter_text and filter_text not in case.name:
            continue
        if case.name in divergences:
            report.counters["skip_intentional_divergence"] += 1
            continue
        if _is_kglite_only(case.query):
            report.counters["skip_kglite_extension"] += 1
            continue
        builder = builders.get(case.fixture)
        if builder is None:
            report.counters["skip_missing_fixture"] += 1
            continue

        if case.fixture != active_fixture or case.isolated:
            graph = builder()
            export_graph(graph)
            active_fixture = case.fixture
            report.counters["fixture_activation"] += 1

        kg_result, kg_error = _capture(lambda: _run_kglite(graph, case.query, case.params))
        neo_result, neo_error = _capture(lambda: _run_neo4j(driver, database, case.query, case.params))

        if case.expected_error:
            kg_category = _error_category(kg_error) if kg_error else None
            neo_category = _error_category(neo_error) if neo_error else None
            if kg_category == neo_category == case.expected_error:
                report.counters["pass_expected_error"] += 1
            else:
                report.counters["fail_error_contract"] += 1
                report.failures.append(Failure(case, "expected error category differs", kg_category, neo_category))
        elif kg_error or neo_error:
            report.counters["execution_error"] += 1
            report.failures.append(
                Failure(
                    case,
                    "unexpected execution error",
                    _error_category(kg_error) if kg_error else None,
                    _error_category(neo_error) if neo_error else None,
                )
            )
        else:
            assert kg_result is not None and neo_result is not None
            kg_normal = _normalise(kg_result, case.order_sensitive)
            neo_normal = _normalise(neo_result, case.order_sensitive)
            if kg_normal == neo_normal:
                report.counters["pass_result"] += 1
            else:
                report.counters["fail_result"] += 1
                report.failures.append(Failure(case, "columns or rows differ", kg_normal, neo_normal))

        if case.side_effect_query:
            kg_side, kg_side_error = _capture(lambda: _run_kglite(graph, case.side_effect_query or "", None))
            neo_side, neo_side_error = _capture(
                lambda: _run_neo4j(driver, database, case.side_effect_query or "", None)
            )
            if kg_side_error or neo_side_error:
                report.counters["execution_error"] += 1
                report.failures.append(Failure(case, "side-effect probe failed", kg_side_error, neo_side_error))
            else:
                assert kg_side is not None and neo_side is not None
                if _normalise(kg_side, False) == _normalise(neo_side, False):
                    report.counters["pass_side_effect"] += 1
                else:
                    report.counters["fail_side_effect"] += 1
                    report.failures.append(Failure(case, "mutation side effects differ", kg_side, neo_side))

        if verbose:
            print(f"  checked {case.name}")

    return report


def _print_report(report: RunReport) -> None:
    checked = sum(count for name, count in report.counters.items() if name.startswith(("pass_", "fail_", "execution_")))
    detail = ", ".join(f"{name}={count}" for name, count in sorted(report.counters.items()))
    print(f"summary: {checked} checked — {detail}")
    for failure in report.failures:
        print(f"\nFAIL {failure.case.name}: {failure.reason}")
        print(f"  query: {failure.case.query}")
        print(f"  kglite: {failure.kglite!r}")
        print(f"  neo4j:  {failure.neo4j!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--uri", default="bolt://localhost:7687")
    parser.add_argument("--user", default="neo4j")
    parser.add_argument("--password", default="conformance")
    parser.add_argument("--database", default="neo4j")
    parser.add_argument("--filter")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    try:
        from neo4j import GraphDatabase
    except ImportError:
        print("Install the optional Neo4j driver with: pip install -e '.[neo4j]'", file=sys.stderr)
        return 2

    driver = GraphDatabase.driver(args.uri, auth=(args.user, args.password))
    try:
        report = run_conformance(
            DIFFERENTIAL_QUERIES,
            builders=FIXTURE_BUILDERS,
            driver=driver,
            database=args.database,
            export_graph=lambda graph: kglite.to_neo4j(
                graph,
                args.uri,
                auth=(args.user, args.password),
                database=args.database,
                clear=True,
                verbose=False,
            ),
            intentional_divergences=INTENTIONAL_DIVERGENCES,
            filter_text=args.filter,
            verbose=args.verbose,
        )
    finally:
        driver.close()
    _print_report(report)
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
