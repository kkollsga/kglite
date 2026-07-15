#!/usr/bin/env python3
"""Run the centralized production-source structural and complexity gates."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE = REPO_ROOT / "tests" / "api-baselines" / "source-quality.json"
ENUM_PATTERN = re.compile(r"GraphBackend::[A-Z]")
UNSAFE_PATTERN = re.compile(r"unsafe\s*\{")
DIRECT_BINCODE_PATTERN = re.compile(r"\bbincode\s*::|^\s*use\s+bincode\b", re.MULTILINE)
DIRECT_BINCODE_ALLOWLIST = {
    "crates/kglite/src/serde_codec/bincode_v1.rs",
    "crates/kglite/src/bincode_wire_contract_tests.rs",
}


def _production_rs_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for source_root in sorted((root / "crates").glob("*/src")):
        files.extend(source_root.rglob("*.rs"))
    return sorted(files)


CFG_TEST_LINE = re.compile(r"#\[cfg\(test\)\][ \t]*(?:\r?\n|$)")


def _skip_comment_or_literal(source: str, index: int) -> int | None:
    """Return the first byte after a Rust comment/literal, if one starts here."""
    if source.startswith("//", index):
        newline = source.find("\n", index + 2)
        return len(source) if newline < 0 else newline
    if source.startswith("/*", index):
        depth = 1
        cursor = index + 2
        while cursor < len(source) and depth:
            if source.startswith("/*", cursor):
                depth += 1
                cursor += 2
            elif source.startswith("*/", cursor):
                depth -= 1
                cursor += 2
            else:
                cursor += 1
        return cursor

    raw = re.match(r'(?:b|c)?r(#{0,255})"', source[index:])
    if raw:
        terminator = '"' + raw.group(1)
        end = source.find(terminator, index + raw.end())
        return len(source) if end < 0 else end + len(terminator)

    quote_index = index + 1 if source.startswith(('b"', 'c"'), index) else index
    if quote_index < len(source) and source[quote_index] == '"':
        cursor = quote_index + 1
        while cursor < len(source):
            if source[cursor] == "\\":
                cursor += 2
            elif source[cursor] == '"':
                return cursor + 1
            else:
                cursor += 1
        return len(source)

    # A character literal always has a closing quote. Lifetimes such as 'a do not.
    if source[index] == "'":
        cursor = index + 1
        if cursor < len(source) and source[cursor] == "\\":
            cursor += 2
        else:
            cursor += 1
        if cursor < len(source) and source[cursor] == "'":
            return cursor + 1
    return None


def _balanced_end(source: str, index: int, opening: str, closing: str) -> int:
    """Find the end of a balanced Rust delimiter group."""
    depth = 1
    cursor = index + 1
    while cursor < len(source) and depth:
        skipped = _skip_comment_or_literal(source, cursor)
        if skipped is not None:
            cursor = skipped
            continue
        if source[cursor] == opening:
            depth += 1
        elif source[cursor] == closing:
            depth -= 1
        cursor += 1
    return cursor


def _cfg_test_item_end(source: str, index: int) -> int:
    """Find the end of the item or statement controlled by ``cfg(test)``."""
    cursor = index
    while cursor < len(source):
        while cursor < len(source) and source[cursor].isspace():
            cursor += 1
        skipped = _skip_comment_or_literal(source, cursor) if cursor < len(source) else None
        if skipped is not None and source.startswith(("//", "/*"), cursor):
            cursor = skipped
            continue
        if not source.startswith("#[", cursor):
            break
        cursor = _balanced_end(source, cursor + 1, "[", "]")

    parens = 0
    brackets = 0
    while cursor < len(source):
        skipped = _skip_comment_or_literal(source, cursor)
        if skipped is not None:
            cursor = skipped
            continue
        char = source[cursor]
        if char == "(":
            parens += 1
        elif char == ")":
            parens = max(0, parens - 1)
        elif char == "[":
            brackets += 1
        elif char == "]":
            brackets = max(0, brackets - 1)
        elif char == "{" and parens == 0 and brackets == 0:
            return _balanced_end(source, cursor, "{", "}")
        elif char == ";" and parens == 0 and brackets == 0:
            return cursor + 1
        cursor += 1
    return len(source)


def _cfg_test_markers(source: str) -> list[tuple[int, int]]:
    """Locate line-level cfg(test) attributes outside comments and literals."""
    markers: list[tuple[int, int]] = []
    cursor = 0
    while cursor < len(source):
        skipped = _skip_comment_or_literal(source, cursor)
        if skipped is not None:
            cursor = skipped
            continue
        if source[cursor] == "#":
            line_start = source.rfind("\n", 0, cursor) + 1
            if not source[line_start:cursor].strip():
                marker = CFG_TEST_LINE.match(source, cursor)
                if marker:
                    markers.append((line_start, marker.end()))
                    cursor = marker.end()
                    continue
        cursor += 1
    return markers


def _strip_test_items(source: str) -> str:
    """Blank cfg(test)-controlled items while preserving later production code."""
    production = list(source)
    for start, marker_end in _cfg_test_markers(source):
        end = _cfg_test_item_end(source, marker_end)
        for index in range(start, end):
            if production[index] not in "\r\n":
                production[index] = " "
    return "\n".join(line for line in "".join(production).splitlines() if not line.lstrip().startswith("//"))


def _load_baseline(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if data.get("schema_version") != 1:
        raise ValueError(f"{path}: unsupported source-quality schema")
    return data


def _file_size_violations(root: Path, baseline: dict[str, Any]) -> list[str]:
    limit = baseline["limits"]["file_lines"]
    exceptions = baseline["file_exceptions"]
    violations: list[str] = []
    seen_exceptions: set[str] = set()
    for path in _production_rs_files(root):
        relative = path.relative_to(root).as_posix()
        lines = len(path.read_text().splitlines())
        exception = exceptions.get(relative)
        ceiling = exception["max_lines"] if exception else limit
        if exception:
            seen_exceptions.add(relative)
            if lines <= limit:
                violations.append(f"stale file exception {relative}: {lines} <= default {limit}")
        if lines > ceiling:
            violations.append(f"{relative}: {lines} lines exceeds ceiling {ceiling}")
    for relative in sorted(set(exceptions) - seen_exceptions):
        violations.append(f"stale file exception {relative}: file is missing")
    return violations


def _enum_match_violations(root: Path, baseline: dict[str, Any]) -> list[str]:
    graph_root = root / "crates" / "kglite" / "src" / "graph"
    whitelist = baseline["enum_match_whitelist"]
    hits: dict[str, int] = {}
    for path in sorted(graph_root.rglob("*.rs")):
        if path.name.endswith("_tests.rs"):
            continue
        relative = path.relative_to(graph_root).as_posix()
        count = len(ENUM_PATTERN.findall(_strip_test_items(path.read_text())))
        if count:
            hits[relative] = count
    violations = [
        f"{relative}: {count} GraphBackend variant match(es) are not whitelisted"
        for relative, count in hits.items()
        if relative not in whitelist
    ]
    for relative in sorted(set(whitelist) - set(hits)):
        violations.append(f"stale enum-match exception {relative}: no production match remains")
    return violations


def _unsafe_violations(root: Path, baseline: dict[str, Any]) -> list[str]:
    violations: list[str] = []
    for relative_root in baseline["safety_roots"]:
        for path in sorted((root / relative_root).rglob("*.rs")):
            lines = path.read_text().splitlines()
            for index, line in enumerate(lines):
                if line.lstrip().startswith("//") or not UNSAFE_PATTERN.search(line):
                    continue
                if not any("SAFETY" in item for item in lines[max(0, index - 5) : index]):
                    relative = path.relative_to(root).as_posix()
                    violations.append(f"{relative}:{index + 1}: unsafe block lacks a nearby SAFETY comment")
    return violations


def _module_violations(root: Path, baseline: dict[str, Any]) -> list[str]:
    violations: list[str] = []
    for relative, cap in baseline["mod_rs_caps"].items():
        path = root / relative
        if not path.is_file():
            violations.append(f"stale mod.rs cap {relative}: file is missing")
            continue
        lines = len(path.read_text().splitlines())
        if lines > cap:
            violations.append(f"{relative}: {lines} lines exceeds module cap {cap}")
    return violations


def _symbol_violations(root: Path) -> list[str]:
    storage_mod = (root / "crates" / "kglite" / "src" / "graph" / "storage" / "mod.rs").read_text()
    required = ["pub mod recording;", "pub use recording::RecordingGraph;"]
    return [f"storage/mod.rs lost required symbol: {symbol}" for symbol in required if symbol not in storage_mod]


def _codec_boundary_violations(root: Path) -> list[str]:
    """Keep the removable bincode dependency behind its one adapter."""
    violations: list[str] = []
    for path in _production_rs_files(root):
        relative = path.relative_to(root).as_posix()
        if relative in DIRECT_BINCODE_ALLOWLIST:
            continue
        production = _strip_test_items(path.read_text())
        if DIRECT_BINCODE_PATTERN.search(production):
            violations.append(f"{relative}: direct bincode use bypasses serde_codec")
    return violations


def _collect_function_metrics(root: Path) -> list[dict[str, Any]]:
    command = [
        "cargo",
        "run",
        "--quiet",
        "-p",
        "kglite",
        "--bin",
        "code_tree_stats",
        "--release",
        "--",
        str(root),
        "--function-metrics",
    ]
    result = subprocess.run(command, cwd=root, text=True, capture_output=True, check=False)
    if result.returncode:
        raise RuntimeError(f"function-metric command failed:\n{result.stdout}\n{result.stderr}")
    metrics = json.loads(result.stdout)
    return [
        metric
        for metric in metrics
        if metric["path"].startswith("crates/")
        and "/src/" in metric["path"]
        and metric["path"].endswith(".rs")
        and not metric["is_test"]
    ]


def _metric_identity(metric: dict[str, Any]) -> str:
    return f"{metric['path']}::{metric['qualified_name']}"


def _normalise_metric(metric: dict[str, Any]) -> dict[str, Any]:
    start = int(metric["start_line"])
    end = int(metric["end_line"] or start)
    return {
        "path": metric["path"],
        "qualified_name": metric["qualified_name"],
        "lines": max(1, end - start + 1),
        "branches": int(metric["branch_count"]),
        "nesting": int(metric["max_nesting"]),
    }


def _is_exception(metric: dict[str, Any], limits: dict[str, int]) -> bool:
    return (
        metric["lines"] > limits["function_lines"]
        or metric["branches"] > limits["function_branches"]
        or metric["nesting"] > limits["function_nesting"]
    )


def _function_violations(metrics: list[dict[str, Any]], baseline: dict[str, Any]) -> list[str]:
    limits = baseline["limits"]
    current = {_metric_identity(metric): _normalise_metric(metric) for metric in metrics}
    if len(current) != len(metrics):
        return ["function metric identities are not unique"]
    expected = {f"{metric['path']}::{metric['qualified_name']}": metric for metric in baseline["function_exceptions"]}
    violations: list[str] = []
    for identity, metric in current.items():
        exceptional = _is_exception(metric, limits)
        captured = expected.get(identity)
        if exceptional and captured is None:
            violations.append(f"new complex function {identity}: {metric}")
            continue
        if not exceptional and captured is not None:
            violations.append(f"stale function exception {identity}: now within defaults")
            continue
        if captured is None:
            continue
        for key in ("lines", "branches", "nesting"):
            if metric[key] > captured[key]:
                violations.append(f"function grew {identity}: {key} {captured[key]} -> {metric[key]}")
            elif metric[key] < captured[key]:
                violations.append(
                    f"function exception can tighten {identity}: {key} {captured[key]} -> {metric[key]}; "
                    "run --refresh-functions"
                )
    for identity in sorted(set(expected) - set(current)):
        violations.append(f"stale function exception {identity}: function is missing")
    return violations


def _function_cap_violations(metrics: list[dict[str, Any]], baseline: dict[str, Any]) -> list[str]:
    current = {_metric_identity(metric): _normalise_metric(metric) for metric in metrics}
    caps = {f"{metric['path']}::{metric['qualified_name']}": metric for metric in baseline["function_caps"]}
    if len(caps) != len(baseline["function_caps"]):
        return ["function cap identities are not unique"]

    violations: list[str] = []
    for identity, cap in caps.items():
        metric = current.get(identity)
        if metric is None:
            violations.append(f"stale function cap {identity}: function is missing")
            continue
        for key in ("lines", "branches", "nesting"):
            if metric[key] > cap[key]:
                violations.append(f"function exceeded cap {identity}: {key} {cap[key]} -> {metric[key]}")
    return violations


def _refresh_functions(path: Path, root: Path, baseline: dict[str, Any]) -> None:
    metrics = [_normalise_metric(metric) for metric in _collect_function_metrics(root)]
    exceptions = [metric for metric in metrics if _is_exception(metric, baseline["limits"])]
    exceptions.sort(key=lambda item: (item["path"], item["qualified_name"]))
    baseline["function_exceptions"] = exceptions
    path.write_text(json.dumps(baseline, indent=2, sort_keys=False) + "\n")
    print(f"refreshed {len(exceptions)} function ceilings in {path.relative_to(root)}")


def _check(root: Path, baseline: dict[str, Any]) -> list[str]:
    violations: list[str] = []
    violations.extend(_file_size_violations(root, baseline))
    violations.extend(_enum_match_violations(root, baseline))
    violations.extend(_unsafe_violations(root, baseline))
    violations.extend(_module_violations(root, baseline))
    violations.extend(_symbol_violations(root))
    violations.extend(_codec_boundary_violations(root))
    metrics = _collect_function_metrics(root)
    violations.extend(_function_violations(metrics, baseline))
    violations.extend(_function_cap_violations(metrics, baseline))
    return violations


def _self_test() -> None:
    baseline = {
        "limits": {
            "file_lines": 3,
            "function_lines": 10,
            "function_branches": 3,
            "function_nesting": 2,
        },
        "file_exceptions": {},
        "enum_match_whitelist": {},
        "safety_roots": ["crates/kglite/src/graph"],
        "mod_rs_caps": {},
        "function_exceptions": [],
        "function_caps": [],
    }
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        graph = root / "crates" / "kglite" / "src" / "graph"
        graph.mkdir(parents=True)
        source = graph / "sample.rs"
        source.write_text("one\ntwo\nthree\nfour\n")
        assert _file_size_violations(root, baseline)
        source.write_text("fn f() { GraphBackend::Memory(x); }\n")
        assert _enum_match_violations(root, baseline)
        source.write_text("fn f() { unsafe { call(); } }\n")
        assert _unsafe_violations(root, baseline)
        source.write_text("fn f() { bincode::serialize(&1).unwrap(); }\n")
        assert _codec_boundary_violations(root)

    mixed_source = """\
fn before() { GraphBackend::Memory(graph); }
const CFG_TEXT: &str = r#"
#[cfg(test)]
"#;
// #[cfg(test)] in a comment is not an attribute.
#[cfg(test)]
use crate::GraphBackend;
fn after_import() { GraphBackend::Mapped(graph); }
#[cfg(test)]
#[allow(dead_code)]
mod tests {
    const BRACES: &str = r#"{ not syntax }"#;
    /* nested { comment /* inner } */ still comment } */
    fn helper() { GraphBackend::Disk(graph); }
}
fn after_module() { GraphBackend::Recording(graph); }
fn with_test_statement() {
    #[cfg(test)]
    { GraphBackend::Disk(graph); }
    GraphBackend::Memory(graph);
}
"""
    stripped = _strip_test_items(mixed_source)
    assert len(ENUM_PATTERN.findall(stripped)) == 4
    assert "after_import" in stripped
    assert "after_module" in stripped
    assert "helper" not in stripped
    assert "CFG_TEXT" in stripped

    captured = {
        "path": "crates/demo/src/lib.rs",
        "qualified_name": "crate::large",
        "lines": 12,
        "branches": 1,
        "nesting": 1,
    }
    baseline["function_exceptions"] = [captured]
    raw = {
        "path": captured["path"],
        "qualified_name": captured["qualified_name"],
        "start_line": 1,
        "end_line": 13,
        "branch_count": 1,
        "max_nesting": 1,
        "is_test": False,
    }
    assert any("function grew" in item for item in _function_violations([raw], baseline))
    tightened = copy.deepcopy(raw)
    tightened["end_line"] = 11
    assert any("can tighten" in item for item in _function_violations([tightened], baseline))

    baseline["function_caps"] = [captured]
    assert not _function_cap_violations([tightened], baseline)
    assert any("exceeded cap" in item for item in _function_cap_violations([raw], baseline))
    assert any("function is missing" in item for item in _function_cap_violations([], baseline))
    print("source-quality self-test: OK")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--refresh-functions", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        _self_test()
        return 0

    root = args.root.resolve()
    baseline_path = args.baseline.resolve()
    baseline = _load_baseline(baseline_path)
    if args.refresh_functions:
        _refresh_functions(baseline_path, root, baseline)
        return 0

    violations = _check(root, baseline)
    if violations:
        print("source-quality gate failed:")
        for violation in violations:
            print(f"  - {violation}")
        return 1
    print("source-quality gate: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
