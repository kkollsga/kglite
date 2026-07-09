"""Phase B.1 — `code_tree.diff(a, b)`: structural diff of two code graphs.

Builds two revisions of a fixture git repo (reusing A.1's temp-git-repo
pattern) and asserts the add / remove / move / change classification, the
identity/fingerprint contract, and the graceful-failure behavior on
non-code graphs. Also unit-tests the reusable `match_entities` identity core
(which Phase B.2's multi-rev merge will reuse).
"""

from pathlib import Path
import subprocess

import pytest

import kglite
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


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    """A git repo with commit1 (tag `v1`) and commit2 (HEAD) exercising every
    diff bucket between the two.

    v1 → HEAD deltas:
      - `brand_new`  added         (new function in core.py)
      - `to_remove`  removed       (dropped from core.py)
      - `to_move`    moved         (util.py → moved.py, same simple name)
      - `THRESHOLD`  changed       (constant value 10 → 20)
      - `to_change`  changed       (body grows: loc span changes)
      - `keep_me`, `Stable`  unchanged
    """
    root = tmp_path / "repo"
    (root / "pkg").mkdir(parents=True)
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "test@example.com")
    _git(root, "config", "user.name", "Test")

    core = root / "pkg" / "core.py"
    util = root / "pkg" / "util.py"
    core.write_text(
        "THRESHOLD = 10\n"
        "\n"
        "def keep_me(x):\n"
        "    return x\n"
        "\n"
        "def to_remove(x):\n"
        "    return x * 2\n"
        "\n"
        "def to_change(x):\n"
        "    return x\n"
        "\n"
        "class Stable:\n"
        "    pass\n"
    )
    util.write_text("def to_move():\n    return 1\n")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "commit1")
    _git(root, "tag", "v1")

    # commit2 — apply all the deltas.
    core.write_text(
        "THRESHOLD = 20\n"  # changed value
        "\n"
        "def keep_me(x):\n"  # unchanged
        "    return x\n"
        "\n"
        "def to_change(x):\n"  # body grows -> loc_span changes
        "    y = x + 1\n"
        "    z = y + 1\n"
        "    return z\n"
        "\n"
        "def brand_new():\n"  # added
        "    return 99\n"
        "\n"
        "class Stable:\n"  # unchanged
        "    pass\n"
    )
    util.unlink()
    (root / "pkg" / "moved.py").write_text("def to_move():\n    return 1\n")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "commit2")
    return root


@pytest.fixture()
def delta(repo: Path) -> dict:
    old = code_tree.build(str(repo), rev="v1")
    new = code_tree.build(str(repo))  # working tree == commit2
    return code_tree.diff(old, new)


# ── The four buckets ────────────────────────────────────────────────────────


def test_added_exact(delta: dict) -> None:
    assert _names(delta["added"]) == {"brand_new"}
    item = next(it for it in delta["added"] if it["name"] == "brand_new")
    assert item["type"] == "Function"
    assert item["file"] == "pkg/core.py"
    assert item["line"] > 0


def test_removed_exact(delta: dict) -> None:
    assert _names(delta["removed"]) == {"to_remove"}
    item = next(it for it in delta["removed"] if it["name"] == "to_remove")
    assert item["type"] == "Function"
    assert item["file"] == "pkg/core.py"


def test_moved_carries_both_files(delta: dict) -> None:
    assert len(delta["moved"]) == 1
    mv = delta["moved"][0]
    assert mv["name"] == "to_move"
    assert mv["type"] == "Function"
    assert mv["old_file"] == "pkg/util.py"
    assert mv["new_file"] == "pkg/moved.py"
    # A move is NOT reported as an add or a remove.
    assert "to_move" not in _names(delta["added"])
    assert "to_move" not in _names(delta["removed"])


def test_changed_function_body(delta: dict) -> None:
    ch = next(it for it in delta["changed"] if it["name"] == "to_change")
    assert ch["type"] == "Function"
    # The body grew; the position-independent line span is the signal.
    assert "loc_span" in ch["changes"]
    assert ch["changes"]["loc_span"]["old"] != ch["changes"]["loc_span"]["new"]


def test_changed_constant_value(delta: dict) -> None:
    ch = next(it for it in delta["changed"] if it["name"] == "THRESHOLD")
    assert ch["type"] == "Constant"
    assert ch["changes"]["value_preview"]["old"] == "10"
    assert ch["changes"]["value_preview"]["new"] == "20"


def test_unchanged_in_no_bucket(delta: dict) -> None:
    touched = (
        _names(delta["added"])
        | _names(delta["removed"])
        | {it["name"] for it in delta["changed"]}
        | {mv["name"] for mv in delta["moved"]}
    )
    assert "keep_me" not in touched
    assert "Stable" not in touched


def test_summary_counts(delta: dict) -> None:
    s = delta["summary"]
    assert s["added"] == len(delta["added"]) == 1
    assert s["removed"] == len(delta["removed"]) == 1
    assert s["moved"] == len(delta["moved"]) == 1
    assert s["changed"] == len(delta["changed"]) == 2
    assert s["unchanged"] >= 2  # keep_me + Stable (+ module/other stable nodes)
    assert "Function" in s["types_compared"]
    assert "Constant" in s["types_compared"]


# ── Identity / self-diff ────────────────────────────────────────────────────


def test_self_diff_is_all_empty(repo: Path) -> None:
    g = code_tree.build(str(repo))
    d = code_tree.diff(g, g)
    assert d["added"] == []
    assert d["removed"] == []
    assert d["moved"] == []
    assert d["changed"] == []
    assert d["summary"]["unchanged"] > 0


def test_rev_vs_rev_of_same_tree_matches(repo: Path) -> None:
    """Two rev builds (both throwaway tempdirs, different basenames) must still
    match symbol-for-symbol — the root-prefix normalization's whole job."""
    a = code_tree.build(str(repo), rev="v1")
    b = code_tree.build(str(repo), rev="v1")
    d = code_tree.diff(a, b)
    assert d["summary"]["added"] == 0
    assert d["summary"]["removed"] == 0
    assert d["summary"]["changed"] == 0
    assert d["summary"]["moved"] == 0


def test_rename_is_remove_plus_add_not_moved(tmp_path: Path) -> None:
    """A rename in place (same file, new name) is honestly a remove + add, not
    a move — the documented contract."""
    root = tmp_path / "r"
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "t@e.com")
    _git(root, "config", "user.name", "T")
    f = root / "m.py"
    f.write_text("def old_name():\n    return 1\n")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "c1")
    _git(root, "tag", "v1")
    f.write_text("def new_name():\n    return 1\n")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "c2")

    d = code_tree.diff(code_tree.build(str(root), rev="v1"), code_tree.build(str(root)))
    assert "old_name" in _names(d["removed"])
    assert "new_name" in _names(d["added"])
    assert d["moved"] == []


# ── Graceful failure on non-code graphs ─────────────────────────────────────


def test_empty_graph_raises_clear_error() -> None:
    empty = kglite.KnowledgeGraph()
    good_repo_graph = kglite.KnowledgeGraph()
    with pytest.raises(ValueError, match="code_tree"):
        code_tree.diff(empty, good_repo_graph)


def test_non_code_graph_raises_clear_error(repo: Path) -> None:
    code = code_tree.build(str(repo))
    non_code = kglite.KnowledgeGraph()
    non_code.cypher("CREATE (:Person {name: 'ada'})")
    with pytest.raises(ValueError, match="code_tree"):
        code_tree.diff(code, non_code)
    # Order-independent: non-code graph on either side.
    with pytest.raises(ValueError, match="code_tree"):
        code_tree.diff(non_code, code)


# ── Reusable identity core (for Phase B.2) ──────────────────────────────────


def test_match_entities_partitions_by_qualified_name() -> None:
    a = [
        {"qualified_name": "m.a", "v": 1},
        {"qualified_name": "m.shared", "v": 2},
    ]
    b = [
        {"qualified_name": "m.shared", "v": 3},
        {"qualified_name": "m.b", "v": 4},
    ]
    matched, only_a, only_b = code_tree.match_entities(a, b)
    assert [(x["qualified_name"], y["qualified_name"]) for x, y in matched] == [("m.shared", "m.shared")]
    assert matched[0][0]["v"] == 2 and matched[0][1]["v"] == 3
    assert [r["qualified_name"] for r in only_a] == ["m.a"]
    assert [r["qualified_name"] for r in only_b] == ["m.b"]


def test_normalize_strips_backslash_joined_build_root() -> None:
    """Unnamespaced PHP gets a synthetic ``<build-root>\\<rel-path>\\<symbol>``
    qualified_name (backslash-joined). Two builds of the same tree differ only
    in that leading build-root basename, so root normalization must strip the
    ``\\``-joined lead just as it strips a ``.``-joined one — otherwise every
    entity mis-reports as removed+added. Exercised at the normalize/match seam,
    no PHP build required."""
    from kglite.code_tree._diff import _normalize_roots

    # Same tree, two throwaway build roots (rev tempdir vs worktree dir).
    a_by_type = {"Class": [{"qualified_name": "tmpAAAA\\m\\Foo", "name": "Foo"}]}
    b_by_type = {"Class": [{"qualified_name": "repo\\m\\Foo", "name": "Foo"}]}
    _normalize_roots(a_by_type, b_by_type)
    # Both leading roots stripped -> identical rel-path-relative qn.
    assert a_by_type["Class"][0]["qualified_name"] == "m\\Foo"
    assert b_by_type["Class"][0]["qualified_name"] == "m\\Foo"
    matched, only_a, only_b = code_tree.match_entities(a_by_type["Class"], b_by_type["Class"])
    assert len(matched) == 1
    assert only_a == [] and only_b == []


def test_normalize_leaves_stable_backslash_namespace_alone() -> None:
    """A real PHP ``namespace`` lead (``App\\…``) appears in *both* builds, so it
    is stable and must NOT be stripped as a build root — only the differing
    build-root basename is."""
    from kglite.code_tree._diff import _normalize_roots

    a_by_type = {"Class": [{"qualified_name": "App\\Foo", "name": "Foo"}]}
    b_by_type = {"Class": [{"qualified_name": "App\\Foo", "name": "Foo"}]}
    _normalize_roots(a_by_type, b_by_type)
    assert a_by_type["Class"][0]["qualified_name"] == "App\\Foo"
    assert b_by_type["Class"][0]["qualified_name"] == "App\\Foo"


def test_match_entities_skips_missing_keys() -> None:
    a = [{"qualified_name": None, "v": 1}, {"qualified_name": "m.x"}]
    b = [{"qualified_name": "m.x"}]
    matched, only_a, only_b = code_tree.match_entities(a, b)
    assert len(matched) == 1
    assert only_a == []
    assert only_b == []
