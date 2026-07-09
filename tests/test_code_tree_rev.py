"""Phase A.1 — `code_tree.build(rev=…)`: build a code graph from a git
revision without disturbing the working tree (git-archive → tempdir → build).

Core honesty property: a `rev` build reflects *committed* content at that
revision, never the working tree — uncommitted edits and untracked files
are invisible, and the working tree is left untouched.
"""

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


def _fn_names(g) -> set[str]:
    return {r["n"] for r in g.cypher("MATCH (f:Function) RETURN f.name AS n")}


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    """A git repo with two commits and a tag `v1` on the first.

    commit1 (tag v1): mod.py defines `fn_old`.
    commit2:          mod.py defines `fn_new` (fn_old removed).
    """
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "test@example.com")
    _git(root, "config", "user.name", "Test")

    mod = root / "mod.py"
    mod.write_text("def fn_old():\n    return 1\n")
    _git(root, "add", "mod.py")
    _git(root, "commit", "-q", "-m", "commit1")
    _git(root, "tag", "v1")

    mod.write_text("def fn_new():\n    return 2\n")
    _git(root, "add", "mod.py")
    _git(root, "commit", "-q", "-m", "commit2")
    return root


def test_working_tree_build_sees_head(repo: Path) -> None:
    g = code_tree.build(str(repo))
    assert "fn_new" in _fn_names(g)
    assert "fn_old" not in _fn_names(g)


def test_rev_by_tag_sees_old(repo: Path) -> None:
    g = code_tree.build(str(repo), rev="v1")
    names = _fn_names(g)
    assert "fn_old" in names
    assert "fn_new" not in names


def test_rev_by_short_sha_matches_tag(repo: Path) -> None:
    sha = _git(repo, "rev-parse", "--short", "v1")
    g = code_tree.build(str(repo), rev=sha)
    names = _fn_names(g)
    assert "fn_old" in names
    assert "fn_new" not in names


def test_rev_build_leaves_working_tree_untouched(repo: Path) -> None:
    before = sorted(p.name for p in repo.iterdir())
    status_before = _git(repo, "status", "--porcelain")
    code_tree.build(str(repo), rev="v1")
    after = sorted(p.name for p in repo.iterdir())
    # No new files materialized in the repo dir; git state unchanged.
    assert before == after
    assert _git(repo, "status", "--porcelain") == status_before
    # The working-tree file still holds commit2's content.
    assert "fn_new" in (repo / "mod.py").read_text()


def test_uncommitted_change_not_in_rev_build(repo: Path) -> None:
    """The honesty property: an uncommitted edit on disk is invisible to a
    rev build, which sees only committed content at that revision."""
    # Dirty the working tree: rewrite mod.py with a function that exists in
    # neither commit, WITHOUT committing it.
    (repo / "mod.py").write_text("def fn_dirty():\n    return 99\n")
    # HEAD is commit2 → committed content still defines fn_new, not fn_dirty.
    g = code_tree.build(str(repo), rev="HEAD")
    names = _fn_names(g)
    assert "fn_new" in names
    assert "fn_dirty" not in names
    # But a plain working-tree build DOES see the uncommitted edit.
    g_wt = code_tree.build(str(repo))
    assert "fn_dirty" in _fn_names(g_wt)


def test_rev_build_stamps_provenance(repo: Path) -> None:
    sha = _git(repo, "rev-parse", "v1")
    g = code_tree.build(str(repo), rev="v1")
    desc = g.describe()
    # The revision as given and its short SHA are recorded, queryable via
    # describe() (rendered from the graph's instructions).
    assert "v1" in desc
    assert sha[:12] in desc
    assert "revision" in desc.lower()


def test_bad_rev_raises_clear_error(repo: Path) -> None:
    with pytest.raises(Exception) as exc:
        code_tree.build(str(repo), rev="no-such-rev-xyz")
    msg = str(exc.value)
    assert "no-such-rev-xyz" in msg
    assert "revision" in msg.lower()


def test_non_git_dir_with_rev_raises_clear_error(tmp_path: Path) -> None:
    plain = tmp_path / "plain"
    plain.mkdir()
    (plain / "mod.py").write_text("def f():\n    return 1\n")
    with pytest.raises(Exception) as exc:
        code_tree.build(str(plain), rev="v1")
    assert "not a git repository" in str(exc.value).lower()
