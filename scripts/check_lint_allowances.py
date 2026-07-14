#!/usr/bin/env python3
"""Inventory Rust lint allowances by stable item identity and prevent drift."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
import re
import tempfile
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE = REPO_ROOT / "tests" / "api-baselines" / "lint-allowances.json"
ALLOW_ATTRIBUTE = re.compile(r"(?P<inner>#!|#)\[\s*allow\s*\((?P<body>.*?)\)\s*\]", re.DOTALL)
COMMENT = re.compile(r"//.*?$|/\*.*?\*/", re.MULTILINE | re.DOTALL)
OTHER_ATTRIBUTE = re.compile(r"^\s*#\[.*?\]\s*", re.DOTALL)

CLASSIFICATION_REASONS = {
    "api-shape": "Changing the warned shape is deferred to the public API review.",
    "boundary": "The signature or import shape is imposed by a binding or protocol boundary.",
    "test": "The allowance exists only for a test, fixture, or numeric test literal.",
    "transitional": "Internal debt is inventoried for removal or localization in the next phase.",
    "dead-code": "Dead-code allowances are audited separately from style and API lints.",
}

API_SHAPE_LINTS = {
    "clippy::len_without_is_empty",
    "clippy::new_without_default",
    "clippy::result_large_err",
    "clippy::result_unit_err",
    "clippy::should_implement_trait",
}
BOUNDARY_LINTS = {
    "clippy::missing_safety_doc",
    "clippy::not_unsafe_ptr_arg_deref",
    "hidden_glob_reexports",
    "private_interfaces",
    "unused_imports",
}
TEST_LINTS = {"clippy::approx_constant", "unreachable_patterns"}


@dataclass(frozen=True)
class Allowance:
    path: str
    scope: str
    lint: str
    line: int
    has_reason: bool

    @property
    def identity(self) -> str:
        return f"{self.path}::{self.scope}::{self.lint}"


def _normalise(text: str) -> str:
    return " ".join(text.split())


def _scope_after(source: str, end: int, inner: bool) -> str:
    if inner:
        return "module"
    tail = source[end:]
    while True:
        stripped = COMMENT.sub("", tail, count=1).lstrip()
        attribute = OTHER_ATTRIBUTE.match(stripped)
        if attribute is None:
            tail = stripped
            break
        tail = stripped[attribute.end() :]

    item = re.search(
        r"(?s)\b(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?(?:unsafe\s+)?"
        r"(?:extern\s+\"[^\"]+\"\s+)?fn\s+([A-Za-z_][A-Za-z0-9_]*)\b.*?(?=\{|;)",
        tail[:2000],
    )
    if item:
        return f"fn:{item.group(1)}:{_normalise(item.group(0))[:300]}"

    named = re.search(
        r"(?s)\b(?:pub(?:\([^)]*\))?\s+)?(struct|enum|trait|type|const|static|mod)\s+"
        r"([A-Za-z_][A-Za-z0-9_]*)\b.*?(?=\{|;|=)",
        tail[:1200],
    )
    if named:
        return f"{named.group(1)}:{named.group(2)}"

    impl_item = re.match(r"(?s)\s*(impl\b.*?)(?=\{)", tail[:1200])
    if impl_item:
        return f"impl:{_normalise(impl_item.group(1))[:300]}"

    first_line = next((line.strip() for line in tail.splitlines() if line.strip()), "<end-of-file>")
    return f"statement:{_normalise(first_line)[:300]}"


def _has_reason(source: str, start: int, end: int) -> bool:
    line_start = source.rfind("\n", 0, start) + 1
    previous = source[:line_start].splitlines()[-3:]
    same_line_end = source.find("\n", end)
    if same_line_end < 0:
        same_line_end = len(source)
    nearby = "\n".join(previous) + source[end:same_line_end]
    comments = [line.strip() for line in nearby.splitlines() if line.strip().startswith("//")]
    return any(len(line.removeprefix("//").strip()) >= 12 for line in comments)


def _split_lints(body: str) -> list[str]:
    without_comments = COMMENT.sub("", body)
    return [lint.strip() for lint in without_comments.split(",") if lint.strip()]


def collect_allowances(root: Path) -> list[Allowance]:
    allowances: list[Allowance] = []
    for path in sorted((root / "crates").rglob("*.rs")):
        if "/target/" in path.as_posix():
            continue
        source = path.read_text()
        relative = path.relative_to(root).as_posix()
        for match in ALLOW_ATTRIBUTE.finditer(source):
            scope = _scope_after(source, match.end(), match.group("inner") == "#!")
            line = source.count("\n", 0, match.start()) + 1
            reason = _has_reason(source, match.start(), match.end())
            for lint in _split_lints(match.group("body")):
                allowances.append(Allowance(relative, scope, lint, line, reason))
    totals = Counter(allowance.identity for allowance in allowances)
    occurrences: dict[str, int] = defaultdict(int)
    unique: list[Allowance] = []
    for allowance in allowances:
        base = allowance.identity
        if totals[base] == 1:
            unique.append(allowance)
            continue
        occurrences[base] += 1
        unique.append(
            Allowance(
                allowance.path,
                f"{allowance.scope}@occurrence:{occurrences[base]}",
                allowance.lint,
                allowance.line,
                allowance.has_reason,
            )
        )
    return unique


def classify(allowance: Allowance) -> str:
    if allowance.lint == "dead_code":
        return "dead-code"
    if allowance.lint in TEST_LINTS or "tests" in Path(allowance.path).parts:
        return "test"
    if allowance.lint in API_SHAPE_LINTS:
        return "api-shape"
    if allowance.lint in BOUNDARY_LINTS:
        return "boundary"
    if allowance.lint == "clippy::too_many_arguments" and (
        allowance.path.startswith("crates/kglite-py/")
        or allowance.path.startswith("crates/kglite-c/")
        or allowance.path.startswith("crates/kglite-mcp-server/")
    ):
        return "boundary"
    return "transitional"


def _load(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if data.get("schema_version") != 1:
        raise ValueError(f"{path}: unsupported lint allowance schema")
    return data


def _entries(allowances: list[Allowance]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    ordinary: list[dict[str, str]] = []
    dead_code: list[dict[str, str]] = []
    for allowance in allowances:
        entry = {"identity": allowance.identity, "classification": classify(allowance)}
        (dead_code if allowance.lint == "dead_code" else ordinary).append(entry)
    ordinary.sort(key=lambda entry: entry["identity"])
    dead_code.sort(key=lambda entry: entry["identity"])
    return ordinary, dead_code


def _identities(entries: list[dict[str, str]]) -> set[str]:
    return {entry["identity"] for entry in entries}


def check(allowances: list[Allowance], baseline: dict[str, Any]) -> list[str]:
    ordinary, dead_code = _entries(allowances)
    violations: list[str] = []
    for key, current in (("allowances", ordinary), ("dead_code_allowances", dead_code)):
        current_ids = _identities(current)
        baseline_ids = _identities(baseline[key])
        for identity in sorted(current_ids - baseline_ids):
            violations.append(f"new unreviewed allowance: {identity}")
        for identity in sorted(baseline_ids - current_ids):
            violations.append(f"stale allowance baseline entry: {identity}")
        expected_classes = {entry["identity"]: entry["classification"] for entry in baseline[key]}
        for entry in current:
            expected = expected_classes.get(entry["identity"])
            if expected is not None and entry["classification"] != expected:
                violations.append(f"classification drift {entry['identity']}: {expected} -> {entry['classification']}")
    return violations


def write_baseline(path: Path, allowances: list[Allowance], previous: dict[str, Any] | None, bootstrap: bool) -> None:
    ordinary, dead_code = _entries(allowances)
    if previous is not None and not bootstrap:
        previous_ids = _identities(previous["allowances"] + previous["dead_code_allowances"])
        missing_reasons = [
            allowance.identity
            for allowance in allowances
            if allowance.identity not in previous_ids and not allowance.has_reason
        ]
        if missing_reasons:
            raise ValueError(
                "new allowances need a nearby explanatory comment before refresh:\n  "
                + "\n  ".join(sorted(missing_reasons))
            )
    data = {
        "schema_version": 1,
        "classification_reasons": CLASSIFICATION_REASONS,
        "allowances": ordinary,
        "dead_code_allowances": dead_code,
    }
    path.write_text(json.dumps(data, indent=2) + "\n")
    print(f"wrote {len(ordinary)} lint and {len(dead_code)} dead-code identities to {path}")


def self_test() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source_dir = root / "crates" / "demo" / "src"
        source_dir.mkdir(parents=True)
        source = source_dir / "lib.rs"
        source.write_text("#[allow(clippy::too_many_arguments)]\nfn demo(a: u8) {}\n")
        first = collect_allowances(root)
        source.write_text("\n\n#[allow(clippy::too_many_arguments)]\nfn demo(a: u8) {}\n")
        second = collect_allowances(root)
        assert first[0].identity == second[0].identity
        baseline = {
            "allowances": [],
            "dead_code_allowances": [],
        }
        assert check(second, baseline) == [f"new unreviewed allowance: {second[0].identity}"]
        assert not second[0].has_reason
        source.write_text(
            "// Boundary signature mirrors the wire protocol.\n"
            "#[allow(clippy::too_many_arguments)]\n"
            "fn demo(a: u8) {}\n"
        )
        assert collect_allowances(root)[0].has_reason
    print("lint-allowance self-test: OK")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--bootstrap", action="store_true")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0

    root = args.root.resolve()
    baseline_path = args.baseline.resolve()
    allowances = collect_allowances(root)
    if args.bootstrap or args.refresh:
        previous = _load(baseline_path) if baseline_path.exists() else None
        write_baseline(baseline_path, allowances, previous, args.bootstrap)
        return 0

    baseline = _load(baseline_path)
    violations = check(allowances, baseline)
    if violations:
        print("lint-allowance gate failed:")
        for violation in violations:
            print(f"  - {violation}")
        print("Remove the allowance, or add a reason and run --refresh.")
        return 1
    print(f"lint-allowance gate: OK ({len(allowances)} exact identities)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
