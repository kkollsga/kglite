"""Phase B.1v — `build(rev=…)` + `code_tree.diff()` validation across every
supported language.

Two properties are checked per language:

1. **Rev-build parity (the archive-path oracle).** ``build(path, rev=HEAD)``
   (git-archive → throwaway tempdir → build) and a plain working-tree
   ``build(path)`` of the *same* committed tree must diff to all-empty. This is
   the strongest single check that ``rev=`` builds honest content AND that the
   diff's build-root normalization (``_root_alias`` in ``_diff.py``) neutralizes
   the throwaway-tempdir basename under each language's qualified-name
   convention (Rust ``crate::`` / C++ ``::`` leads, dotted Python/TS/Dart/Swift
   leads, C#/Go/Java package leads).

2. **Bucket classification.** A 2-commit fixture plants one added, one removed,
   one body-grown ("changed"), one moved, and one unchanged entity (plus a
   changed constant where the language extracts constant values); ``diff(v1,
   HEAD)`` must sort each into the right bucket with the right type, and nothing
   spurious anywhere.

Fixtures are tree-sitter-idiomatic and each is sanity-built once (asserting the
"kept" entity is present) before diffing — a syntax error yields an empty
extraction that would otherwise masquerade as a passing "everything removed".

Move-detection is qualified-name-scoped and therefore language-dependent (see
``MOVE_DETECTED`` below); HTML has no compared code-entity type at all (see the
dedicated scoping test). These are asserted explicitly, not worked around.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess

import pytest

from kglite import code_tree


def _git(repo: Path, *args: str) -> str:
    out = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return out.stdout.strip()


def _names(items: list[dict]) -> set[str]:
    return {it["name"] for it in items}


# ── Language table ──────────────────────────────────────────────────────────
# Each spec carries two commit states as ``{rel_path: source}`` maps. Between
# them exactly these deltas are planted (names are the simple entity names):
#   removed  : ENTITY_OLD          (present in v1, gone in HEAD)
#   added    : ENTITY_NEW          (absent in v1, present in HEAD)
#   changed  : ENTITY_CHANGED body grows across real newlines -> loc_span delta
#   changed  : CONST_CHANGED value edit (only where the extractor stores a
#              constant value_preview -> ``const_change=True``)
#   moved    : ENTITY_MOVED file A -> file B, same simple name
#   unchanged: ENTITY_KEPT + a kept type node (sanity anchor)


@dataclass
class Spec:
    lang: str
    v1: dict[str, str]
    v2: dict[str, str]
    kept: str  # a simple name that must survive as a compared entity (sanity)
    added: set[str]
    removed: set[str]
    changed_loc: set[str]  # names expected to change via loc_span
    const_change: str | None  # constant name expected to change, or None
    move_detected: bool  # whether ENTITY_MOVED surfaces in the moved bucket
    moved_name: str = "fn_moved"


SPECS: list[Spec] = [
    # ── Python ──────────────────────────────────────────────────────────────
    Spec(
        lang="python",
        v1={
            "pkg/core.py": (
                "KEPT_C = 1\n"
                "CHG_C = 10\n"
                "def fn_old():\n    return 1\n"
                "def fn_kept():\n    return 2\n"
                "def fn_changed():\n    return 3\n"
                "class Kept:\n    pass\n"
            ),
            "pkg/util.py": "def fn_moved():\n    return 9\n",
        },
        v2={
            "pkg/core.py": (
                "KEPT_C = 1\n"
                "CHG_C = 20\n"
                "def fn_kept():\n    return 2\n"
                "def fn_changed():\n    x = 1\n    y = 2\n    return x + y\n"
                "def fn_new():\n    return 4\n"
                "class Kept:\n    pass\n"
            ),
            "pkg/moved.py": "def fn_moved():\n    return 9\n",
        },
        kept="fn_kept",
        added={"fn_new"},
        removed={"fn_old"},
        changed_loc={"fn_changed"},
        const_change="CHG_C",
        move_detected=True,
    ),
    # ── Rust ────────────────────────────────────────────────────────────────
    Spec(
        lang="rust",
        v1={
            "core.rs": (
                "const KEPT_C: i32 = 1;\n"
                "const CHG_C: i32 = 10;\n"
                "fn fn_old() -> i32 { 1 }\n"
                "fn fn_kept() -> i32 { 2 }\n"
                "fn fn_changed() -> i32 { 3 }\n"
                "struct Kept { x: i32 }\n"
            ),
            "util.rs": "fn fn_moved() -> i32 { 9 }\n",
        },
        v2={
            "core.rs": (
                "const KEPT_C: i32 = 1;\n"
                "const CHG_C: i32 = 20;\n"
                "fn fn_kept() -> i32 { 2 }\n"
                "fn fn_changed() -> i32 {\n    let x = 1;\n    let y = 2;\n    x + y\n}\n"
                "fn fn_new() -> i32 { 4 }\n"
                "struct Kept { x: i32 }\n"
            ),
            "moved.rs": "fn fn_moved() -> i32 { 9 }\n",
        },
        kept="fn_kept",
        added={"fn_new"},
        removed={"fn_old"},
        changed_loc={"fn_changed"},
        const_change="CHG_C",
        move_detected=True,
    ),
    # ── C++ (#define constants exercise the C.1 fix) ─────────────────────────
    Spec(
        lang="cpp",
        v1={
            "core.cpp": (
                "#define KEPT_C 1\n"
                "#define CHG_C 10\n"
                "int fn_old() { return 1; }\n"
                "int fn_kept() { return 2; }\n"
                "int fn_changed() { return 3; }\n"
                "struct Kept { int x; };\n"
            ),
            "util.cpp": "int fn_moved() { return 9; }\n",
        },
        v2={
            "core.cpp": (
                "#define KEPT_C 1\n"
                "#define CHG_C 20\n"
                "int fn_kept() { return 2; }\n"
                "int fn_changed() {\n    int x = 1;\n    int y = 2;\n    return x + y;\n}\n"
                "int fn_new() { return 4; }\n"
                "struct Kept { int x; };\n"
            ),
            "moved.cpp": "int fn_moved() { return 9; }\n",
        },
        kept="fn_kept",
        added={"fn_new"},
        removed={"fn_old"},
        changed_loc={"fn_changed"},
        const_change="CHG_C",
        move_detected=True,
    ),
    # ── Go (package-scoped qn: same-package move == unchanged) ───────────────
    Spec(
        lang="go",
        v1={
            "core.go": (
                "package main\n"
                "const KeptC = 1\n"
                "const ChgC = 10\n"
                "func FnOld() int { return 1 }\n"
                "func FnKept() int { return 2 }\n"
                "func FnChanged() int { return 3 }\n"
                "type Kept struct { X int }\n"
            ),
            "util.go": "package main\nfunc FnMoved() int { return 9 }\n",
        },
        v2={
            "core.go": (
                "package main\n"
                "const KeptC = 1\n"
                "const ChgC = 20\n"
                "func FnKept() int { return 2 }\n"
                "func FnChanged() int {\n\tx := 1\n\ty := 2\n\treturn x + y\n}\n"
                "func FnNew() int { return 4 }\n"
                "type Kept struct { X int }\n"
            ),
            "moved.go": "package main\nfunc FnMoved() int { return 9 }\n",
        },
        kept="FnKept",
        added={"FnNew"},
        removed={"FnOld"},
        changed_loc={"FnChanged"},
        const_change="ChgC",
        move_detected=False,  # main.FnMoved is package-scoped; qn survives the move
        moved_name="FnMoved",
    ),
    # ── Java (methods+class; namespace/class-scoped qn) ──────────────────────
    Spec(
        lang="java",
        v1={
            "Core.java": (
                "public class Core {\n"
                "  public static final int KEPT_C = 1;\n"
                "  public static final int CHG_C = 10;\n"
                "  public void fnOld() {}\n"
                "  public void fnKept() {}\n"
                "  public void fnChanged() { int x = 1; }\n"
                "}\n"
            ),
            "Util.java": "public class Util {\n  public void fnMoved() {}\n}\n",
        },
        v2={
            "Core.java": (
                "public class Core {\n"
                "  public static final int KEPT_C = 1;\n"
                "  public static final int CHG_C = 20;\n"
                "  public void fnKept() {}\n"
                "  public void fnChanged() {\n    int x = 1;\n    int y = 2;\n    int z = x + y;\n  }\n"
                "  public void fnNew() {}\n"
                "}\n"
            ),
            "Moved.java": "public class Util {\n  public void fnMoved() {}\n}\n",
        },
        kept="fnKept",
        added={"fnNew"},
        removed={"fnOld"},
        changed_loc={"fnChanged"},
        const_change="CHG_C",
        move_detected=False,  # Core.fnMoved-style qn is class-scoped
        moved_name="fnMoved",
    ),
    # ── C# (const value_preview populated -> const change detectable) ────────
    Spec(
        lang="csharp",
        v1={
            "Core.cs": (
                "namespace N {\n"
                "  public class Core {\n"
                "    public const int KeptC = 1;\n"
                "    public const int ChgC = 10;\n"
                "    public void FnOld() {}\n"
                "    public void FnKept() {}\n"
                "    public void FnChanged() { int x = 1; }\n"
                "  }\n"
                "}\n"
            ),
            "Util.cs": "namespace N {\n  public class Util { public void FnMoved() {} }\n}\n",
        },
        v2={
            "Core.cs": (
                "namespace N {\n"
                "  public class Core {\n"
                "    public const int KeptC = 1;\n"
                "    public const int ChgC = 20;\n"
                "    public void FnKept() {}\n"
                "    public void FnChanged() {\n      int x = 1;\n      int y = 2;\n      int z = x + y;\n    }\n"
                "    public void FnNew() {}\n"
                "  }\n"
                "}\n"
            ),
            "Moved.cs": "namespace N {\n  public class Util { public void FnMoved() {} }\n}\n",
        },
        kept="FnKept",
        added={"FnNew"},
        removed={"FnOld"},
        changed_loc={"FnChanged"},
        const_change="ChgC",  # value_preview populated for C# const
        move_detected=False,
        moved_name="FnMoved",
    ),
    # ── TypeScript (filename-first dotted qn: root heuristic absorbs a move) ─
    Spec(
        lang="typescript",
        v1={
            "core.ts": (
                "export const KEPT_C = 1;\n"
                "export const CHG_C = 10;\n"
                "export function fnOld() { return 1; }\n"
                "export function fnKept() { return 2; }\n"
                "export function fnChanged() { return 3; }\n"
                "export class Kept {}\n"
            ),
            "util.ts": "export function fnMoved() { return 9; }\n",
        },
        v2={
            "core.ts": (
                "export const KEPT_C = 1;\n"
                "export const CHG_C = 20;\n"
                "export function fnKept() { return 2; }\n"
                "export function fnChanged() {\n  const x = 1;\n  const y = 2;\n  return x + y;\n}\n"
                "export function fnNew() { return 4; }\n"
                "export class Kept {}\n"
            ),
            "moved.ts": "export function fnMoved() { return 9; }\n",
        },
        kept="fnKept",
        added={"fnNew"},
        removed={"fnOld"},
        changed_loc={"fnChanged"},
        const_change="CHG_C",
        move_detected=False,
        moved_name="fnMoved",
    ),
    # ── TSX (JSX component entities; the .tsx grammar path) ──────────────────
    Spec(
        lang="tsx",
        v1={
            "core.tsx": (
                'import React from "react";\n'
                "export const KEPT_C = 1;\n"
                "export const CHG_C = 10;\n"
                "export function OldBtn() { return <div/>; }\n"
                "export function KeptBtn() { return <div/>; }\n"
                "export function ChangedBtn() { return <div/>; }\n"
                "export class Panel extends React.Component { render() { return <span/>; } }\n"
            ),
            "util.tsx": 'import React from "react";\nexport function MovedBtn() { return <p/>; }\n',
        },
        v2={
            "core.tsx": (
                'import React from "react";\n'
                "export const KEPT_C = 1;\n"
                "export const CHG_C = 20;\n"
                "export function KeptBtn() { return <div/>; }\n"
                "export function ChangedBtn() {\n  const x = 1;\n  const y = 2;\n  return <div>{x + y}</div>;\n}\n"
                "export function NewBtn() { return <em/>; }\n"
                "export class Panel extends React.Component { render() { return <span/>; } }\n"
            ),
            "moved.tsx": 'import React from "react";\nexport function MovedBtn() { return <p/>; }\n',
        },
        kept="KeptBtn",
        added={"NewBtn"},
        removed={"OldBtn"},
        changed_loc={"ChangedBtn"},
        const_change="CHG_C",
        move_detected=False,
        moved_name="MovedBtn",
    ),
    # ── Dart (mixin + const; filename-first dotted qn) ───────────────────────
    Spec(
        lang="dart",
        v1={
            "core.dart": (
                "const keptC = 1;\n"
                "const chgC = 10;\n"
                "int fnOld() => 1;\n"
                "int fnKept() => 2;\n"
                "int fnChanged() => 3;\n"
                "class Kept {}\n"
            ),
            "util.dart": "int fnMoved() => 9;\n",
        },
        v2={
            "core.dart": (
                "const keptC = 1;\n"
                "const chgC = 20;\n"
                "int fnKept() => 2;\n"
                "int fnChanged() {\n  var x = 1;\n  var y = 2;\n  return x + y;\n}\n"
                "int fnNew() => 4;\n"
                "class Kept {}\n"
            ),
            "moved.dart": "int fnMoved() => 9;\n",
        },
        kept="fnKept",
        added={"fnNew"},
        removed={"fnOld"},
        changed_loc={"fnChanged"},
        const_change="chgC",
        move_detected=False,
        moved_name="fnMoved",
    ),
    # ── Swift (no Constant node; enum->Class; filename-first dotted qn) ──────
    Spec(
        lang="swift",
        v1={
            "core.swift": (
                "func fnOld() -> Int { return 1 }\n"
                "func fnKept() -> Int { return 2 }\n"
                "func fnChanged() -> Int { return 3 }\n"
                "struct Kept { var x: Int }\n"
            ),
            "util.swift": "func fnMoved() -> Int { return 9 }\n",
        },
        v2={
            "core.swift": (
                "func fnKept() -> Int { return 2 }\n"
                "func fnChanged() -> Int {\n    let x = 1\n    let y = 2\n    return x + y\n}\n"
                "func fnNew() -> Int { return 4 }\n"
                "struct Kept { var x: Int }\n"
            ),
            "moved.swift": "func fnMoved() -> Int { return 9 }\n",
        },
        kept="fnKept",
        added={"fnNew"},
        removed={"fnOld"},
        changed_loc={"fnChanged"},
        const_change=None,  # Swift `let` is not extracted as a Constant node
        move_detected=False,
        moved_name="fnMoved",
    ),
    # ── CSS (no functions; custom-property Constants are the only compared
    #        type; Selectors are intentionally excluded from the diff) ────────
    Spec(
        lang="css",
        v1={"a.css": ":root {\n  --kept: 1px;\n  --chg: 10px;\n  --old: 5px;\n}\n"},
        v2={"a.css": ":root {\n  --kept: 1px;\n  --chg: 20px;\n  --new: 7px;\n}\n"},
        kept="--kept",
        added={"--new"},
        removed={"--old"},
        changed_loc=set(),  # constants have no loc_span fingerprint
        const_change="--chg",
        move_detected=False,
        moved_name="",
    ),
    # ── PHP (namespaced == idiomatic PSR-4; namespace-scoped qn) ─────────────
    Spec(
        lang="php",
        v1={
            "Core.php": (
                "<?php\n"
                "namespace App;\n"
                "const KEPT_C = 1;\n"
                "const CHG_C = 10;\n"
                "function fnOld() { return 1; }\n"
                "function fnKept() { return 2; }\n"
                "function fnChanged() { return 3; }\n"
                "class Kept {}\n"
            ),
            "Util.php": "<?php\nnamespace App;\nfunction fnMoved() { return 9; }\n",
        },
        v2={
            "Core.php": (
                "<?php\n"
                "namespace App;\n"
                "const KEPT_C = 1;\n"
                "const CHG_C = 20;\n"
                "function fnKept() { return 2; }\n"
                "function fnChanged() {\n  $x = 1;\n  $y = 2;\n  return $x + $y;\n}\n"
                "function fnNew() { return 4; }\n"
                "class Kept {}\n"
            ),
            "Moved.php": "<?php\nnamespace App;\nfunction fnMoved() { return 9; }\n",
        },
        kept="fnKept",
        added={"fnNew"},
        removed={"fnOld"},
        changed_loc={"fnChanged"},
        const_change="CHG_C",
        move_detected=False,  # App\fnMoved is namespace-scoped
        moved_name="fnMoved",
    ),
]

SPEC_BY_LANG = {s.lang: s for s in SPECS}


def _write_commit(repo: Path, files: dict[str, str], msg: str, tag: str | None = None) -> None:
    for rel, content in files.items():
        p = repo / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", msg)
    if tag:
        _git(repo, "tag", tag)


def _make_repo(root: Path, spec: Spec) -> Path:
    repo = root / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@e.com")
    _git(repo, "config", "user.name", "T")
    _write_commit(repo, spec.v1, "commit1", tag="v1")
    # Delete files that disappear in v2, then write v2's files.
    for rel in set(spec.v1) - set(spec.v2):
        (repo / rel).unlink()
    _write_commit(repo, spec.v2, "commit2")
    return repo


@pytest.fixture(params=SPECS, ids=lambda s: s.lang)
def spec(request: pytest.FixtureRequest) -> Spec:
    return request.param


# ── 1. Rev-build parity oracle ──────────────────────────────────────────────


def test_rev_build_matches_working_tree(spec: Spec, tmp_path: Path) -> None:
    """``build(rev=HEAD)`` of the committed tree must be byte-identical, symbol
    for symbol, to a plain working-tree build of the same tree — the archive
    path plus the diff's build-root normalization, exercised together."""
    repo = _make_repo(tmp_path, spec)
    from_rev = code_tree.build(str(repo), rev="HEAD")
    from_wt = code_tree.build(str(repo))
    d = code_tree.diff(from_rev, from_wt)
    assert d["summary"]["added"] == 0, d["added"]
    assert d["summary"]["removed"] == 0, d["removed"]
    assert d["summary"]["changed"] == 0, d["changed"]
    assert d["summary"]["moved"] == 0, d["moved"]


# ── 2. Bucket classification ────────────────────────────────────────────────


@pytest.fixture()
def delta(spec: Spec, tmp_path: Path) -> dict:
    repo = _make_repo(tmp_path, spec)
    old = code_tree.build(str(repo), rev="v1")
    # Sanity: the kept anchor must actually parse in both revs, else an empty
    # extraction would masquerade as "everything removed".
    new = code_tree.build(str(repo))
    old_kept = old.cypher("MATCH (n) WHERE n.name = $k RETURN n.name AS name", params={"k": spec.kept}).to_dicts()
    new_kept = new.cypher("MATCH (n) WHERE n.name = $k RETURN n.name AS name", params={"k": spec.kept}).to_dicts()
    assert old_kept, f"{spec.lang}: kept anchor {spec.kept!r} missing in v1 build"
    assert new_kept, f"{spec.lang}: kept anchor {spec.kept!r} missing in HEAD build"
    return code_tree.diff(old, new)


def test_added_bucket(spec: Spec, delta: dict) -> None:
    assert _names(delta["added"]) == spec.added


def test_removed_bucket(spec: Spec, delta: dict) -> None:
    assert _names(delta["removed"]) == spec.removed


_CONTAINER_TYPES = {"Class", "Struct", "Enum", "Interface", "Mixin", "Trait", "Protocol"}


def test_changed_bucket(spec: Spec, delta: dict) -> None:
    expected = set(spec.changed_loc)
    if spec.const_change:
        expected.add(spec.const_change)
    changed_names = {it["name"] for it in delta["changed"]}
    # Every planted change must be present.
    assert expected <= changed_names, (expected, changed_names)
    # Any *extra* changed entity is only legitimate when it is a container type
    # (a class/struct/… whose body size shifted because its members churned) and
    # its sole reported delta is loc_span — never a stray Function/Constant.
    for it in delta["changed"]:
        if it["name"] in expected:
            continue
        assert it["type"] in _CONTAINER_TYPES, it
        assert set(it["changes"]) == {"loc_span"}, it
    # loc_span entities must report a loc_span delta.
    for name in spec.changed_loc:
        ch = next(it for it in delta["changed"] if it["name"] == name)
        assert "loc_span" in ch["changes"]
        assert ch["changes"]["loc_span"]["old"] != ch["changes"]["loc_span"]["new"]
    # constant change must report a value_preview delta.
    if spec.const_change:
        ch = next(it for it in delta["changed"] if it["name"] == spec.const_change)
        assert ch["type"] == "Constant"
        assert "value_preview" in ch["changes"]
        assert ch["changes"]["value_preview"]["old"] != ch["changes"]["value_preview"]["new"]


def test_moved_bucket(spec: Spec, delta: dict) -> None:
    if spec.move_detected:
        assert _names(delta["moved"]) == {spec.moved_name}
        mv = next(m for m in delta["moved"] if m["name"] == spec.moved_name)
        assert mv["old_file"] != mv["new_file"]
    else:
        # Qualified-name-scoped languages: the moved symbol keeps its identity
        # (package/namespace-scoped) or is absorbed by the root heuristic, so it
        # is NOT reported as moved — and crucially not lost as remove/add either.
        assert delta["moved"] == []
        if spec.moved_name:
            assert spec.moved_name not in _names(delta["added"])
            assert spec.moved_name not in _names(delta["removed"])


def test_kept_not_in_any_delta_bucket(spec: Spec, delta: dict) -> None:
    touched = (
        _names(delta["added"])
        | _names(delta["removed"])
        | {it["name"] for it in delta["changed"]}
        | {mv["name"] for mv in delta["moved"]}
    )
    assert spec.kept not in touched
    assert delta["summary"]["unchanged"] > 0


# ── 3. HTML scoping (no compared code-entity type) ──────────────────────────


def test_html_has_no_compared_entities_and_diff_raises(tmp_path: Path) -> None:
    """HTML emits only ``Element`` nodes — none of the compared code-entity
    types — so a code_tree ``diff`` of two HTML-only graphs raises the
    "not a code graph" error. This is a deliberate scoping choice (a symbol-
    level rev diff is not about markup structure); asserted here so a future
    change that starts comparing HTML entities is a conscious, test-visible
    decision.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@e.com")
    _git(repo, "config", "user.name", "T")
    _write_commit(
        repo,
        {
            "page.html": (
                "<html><body>\n"
                '<section id="intro"><h1>Title</h1></section>\n'
                '<div class="content"><p>Text</p></div>\n'
                "</body></html>\n"
            )
        },
        "c1",
    )
    g = code_tree.build(str(repo))
    node_types = set(g.node_types)
    assert "Element" in node_types
    assert not ({"Function", "Class", "Struct", "Enum", "Constant"} & node_types)
    with pytest.raises(ValueError, match="code_tree"):
        code_tree.diff(g, g)


# ── 4. Regression guard for the PHP no-namespace normalization defect ────────


def test_php_without_namespace_rev_parity(tmp_path: Path) -> None:
    """PHP source lacking a ``namespace`` declaration gets a synthetic
    namespace of ``<build-root-basename>\\<rel-path>`` (backslash-joined). The
    build-root basename differs between a ``rev=`` throwaway tempdir and a
    working-tree build; the diff's root normalization now strips ``\\``-joined
    build roots alongside ``.``-joined ones (``_leading``/``_strip_root`` treat
    both as build-root separators), so rev-vs-worktree parity holds and no
    class/method/constant is mis-reported as removed + added. Idiomatic PSR-4
    PHP (with a namespace) is unaffected — its stable namespace lead appears in
    both builds and is never stripped.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@e.com")
    _git(repo, "config", "user.name", "T")
    _write_commit(
        repo,
        {"m.php": "<?php\nclass Foo { public function bar() { return 1; } }\n"},
        "c1",
    )
    from_rev = code_tree.build(str(repo), rev="HEAD")
    from_wt = code_tree.build(str(repo))
    d = code_tree.diff(from_rev, from_wt)
    assert d["summary"]["added"] == 0, d["added"]
    assert d["summary"]["removed"] == 0, d["removed"]
