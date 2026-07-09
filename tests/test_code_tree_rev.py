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


# ─── B.2c — multi-rev graphs via build(revs=[...]) ──────────────────────────


@pytest.fixture()
def repo3(tmp_path: Path) -> Path:
    """A git repo with three tagged commits touching one module.

    v1: mod.py defines `foo(a)` and `gone()`.
    v2: `gone` removed, `bar(x)` added, `foo` now calls bar.
    v3: `foo` signature widened to `(a, b)`.
    """
    root = tmp_path / "repo3"
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "test@example.com")
    _git(root, "config", "user.name", "Test")
    mod = root / "mod.py"

    def _commit(body: str, tag: str) -> None:
        mod.write_text(body)
        _git(root, "add", "mod.py")
        _git(root, "commit", "-q", "-m", tag)
        _git(root, "tag", tag)

    _commit("def foo(a):\n    return a + 1\n\n\ndef gone():\n    return 0\n", "v1")
    _commit("def foo(a):\n    return bar(a)\n\n\ndef bar(x):\n    return x + 1\n", "v2")
    _commit("def foo(a, b):\n    return bar(a) + b\n\n\ndef bar(x):\n    return x + 1\n", "v3")
    return root


def _revs_of(g, name: str) -> list[str]:
    """The `revs` list of the single Function `name`, via Cypher membership."""
    rows = g.cypher("MATCH (f:Function) WHERE f.name = $n RETURN f.revs AS revs", params={"n": name})
    assert len(rows) == 1, f"expected exactly one Function {name}, got {len(rows)}"
    return list(rows[0]["revs"])


def test_multirev_revs_queryable_from_python(repo3: Path) -> None:
    g = code_tree.build(str(repo3), revs=["v1", "v2", "v3"])
    # One node per entity across revs; revs list reflects presence.
    assert _revs_of(g, "foo") == ["v1", "v2", "v3"]
    assert _revs_of(g, "bar") == ["v2", "v3"]  # added in v2
    assert _revs_of(g, "gone") == ["v1"]  # removed after v1


def test_multirev_in_scoping_selects_one_rev(repo3: Path) -> None:
    g = code_tree.build(str(repo3), revs=["v1", "v2", "v3"])
    # `WHERE 'vX' IN n.revs` scopes to the functions present in that rev.
    v1 = {r["n"] for r in g.cypher("MATCH (f:Function) WHERE 'v1' IN f.revs RETURN f.name AS n")}
    v3 = {r["n"] for r in g.cypher("MATCH (f:Function) WHERE 'v3' IN f.revs RETURN f.name AS n")}
    assert v1 == {"foo", "gone"}
    assert v3 == {"foo", "bar"}


def test_multirev_rev_fp_detects_signature_change(repo3: Path) -> None:
    g = code_tree.build(str(repo3), revs=["v1", "v2", "v3"])
    rows = g.cypher("MATCH (f:Function) WHERE f.name = 'foo' RETURN f.rev_fp AS fp")
    fp = list(rows[0]["fp"])
    assert len(fp) == 3, "one fingerprint per rev"
    assert fp[0] == fp[1], "foo body-only edit v1->v2 doesn't change its fingerprint"
    assert fp[1] != fp[2], "foo signature widened v2->v3"


def test_multirev_newest_wins_property_columns(repo3: Path) -> None:
    g = code_tree.build(str(repo3), revs=["v1", "v2", "v3"])
    rows = g.cypher("MATCH (f:Function) WHERE f.name = 'foo' RETURN f.signature AS sig")
    # Plain (unscoped) property read reports the newest rev's value.
    assert "b" in rows[0]["sig"]


def test_multirev_provenance_lists_revs_in_describe(repo3: Path) -> None:
    g = code_tree.build(str(repo3), revs=["v1", "v2", "v3"])
    desc = g.describe()
    for tag in ("v1", "v2", "v3"):
        assert tag in desc
    # Teaches the scoping idiom + names the multi-rev nature.
    assert "IN n.revs" in desc or "IN\nn.revs" in desc


def test_rev_and_revs_mutually_exclusive(repo3: Path) -> None:
    with pytest.raises(Exception) as exc:
        code_tree.build(str(repo3), rev="v1", revs=["v1", "v2"])
    assert "mutually exclusive" in str(exc.value).lower()


def test_single_item_revs_list_works(repo3: Path) -> None:
    g = code_tree.build(str(repo3), revs=["v1"])
    # Same entity set as a plain v1 build, plus the (single-element) rev tag.
    names = {r["n"] for r in g.cypher("MATCH (f:Function) RETURN f.name AS n")}
    assert names == {"foo", "gone"}
    assert _revs_of(g, "foo") == ["v1"]


def test_duplicate_revs_labels_deduped(repo3: Path) -> None:
    # Duplicate labels collapse (order-preserving) before folding: a node's
    # `revs` list carries the label once, never ["v1", "v1"].
    g = code_tree.build(str(repo3), revs=["v1", "v1"])
    assert _revs_of(g, "foo") == ["v1"]


# ─── B.2d — CALL rev_diff procedure ─────────────────────────────────────────


def _rev_diff(g, frm: str, to: str, **extra) -> dict[str, set[str]]:
    """Run rev_diff and bucket qualified_names by bucket. CALL takes an inline
    map literal (not a $param), so build the map text."""
    pairs = [f"from: '{frm}'", f"to: '{to}'"] + [f"{k}: '{v}'" for k, v in extra.items()]
    rows = g.cypher(
        "CALL rev_diff({" + ", ".join(pairs) + "}) YIELD bucket, qualified_name RETURN bucket, qualified_name"
    )
    out: dict[str, set[str]] = {}
    for r in rows:
        out.setdefault(r["bucket"], set()).add(r["qualified_name"])
    return out


def test_rev_diff_added_and_removed_forward(repo3: Path) -> None:
    g = code_tree.build(str(repo3), revs=["v1", "v2", "v3"])
    d = _rev_diff(g, "v1", "v2")
    # bar added in v2; gone removed after v1.
    assert any(qn.endswith("bar") for qn in d.get("added", set())), d
    assert any(qn.endswith("gone") for qn in d.get("removed", set())), d


def test_rev_diff_direction_reverses_added_removed(repo3: Path) -> None:
    g = code_tree.build(str(repo3), revs=["v1", "v2", "v3"])
    d = _rev_diff(g, "v2", "v1")
    # Reversing from/to flips added<->removed.
    assert any(qn.endswith("gone") for qn in d.get("added", set())), d
    assert any(qn.endswith("bar") for qn in d.get("removed", set())), d


def test_rev_diff_changed_via_fingerprint(repo3: Path) -> None:
    g = code_tree.build(str(repo3), revs=["v1", "v2", "v3"])
    # foo's signature widened v2->v3 → changed; body-only v1->v2 is NOT changed.
    d23 = _rev_diff(g, "v2", "v3")
    assert any(qn.endswith("foo") for qn in d23.get("changed", set())), d23
    d12 = _rev_diff(g, "v1", "v2")
    assert not any(qn.endswith("foo") for qn in d12.get("changed", set())), d12


def test_rev_diff_full_columns(repo3: Path) -> None:
    g = code_tree.build(str(repo3), revs=["v1", "v2", "v3"])
    rows = g.cypher(
        "CALL rev_diff({from: 'v2', to: 'v3'}) "
        "YIELD bucket, type, qualified_name, name, file, line "
        "WHERE name = 'foo' RETURN bucket, type, name, file, line"
    )
    assert len(rows) == 1
    r = rows[0]
    assert r["bucket"] == "changed"
    assert r["type"] == "Function"
    assert r["name"] == "foo"
    assert r["file"] == "mod.py"
    assert isinstance(r["line"], int)


def test_rev_diff_node_type_scoping(repo3: Path) -> None:
    g = code_tree.build(str(repo3), revs=["v1", "v2", "v3"])
    # Scoping to Function only still finds the function-level changes.
    rows = g.cypher(
        "CALL rev_diff({from: 'v1', to: 'v2', node_type: 'Function'}) YIELD bucket, type RETURN DISTINCT type"
    )
    types = {r["type"] for r in rows}
    assert types == {"Function"}


def test_rev_diff_unknown_rev_errors(repo3: Path) -> None:
    g = code_tree.build(str(repo3), revs=["v1", "v2"])
    with pytest.raises(Exception) as exc:
        g.cypher("CALL rev_diff({from: 'v1', to: 'nope'}) YIELD bucket RETURN bucket")
    msg = str(exc.value)
    assert "nope" in msg
    assert "v1" in msg and "v2" in msg  # lists available revs


def test_rev_diff_not_multirev_errors(repo3: Path) -> None:
    g = code_tree.build(str(repo3))  # plain working-tree build, no revs props
    with pytest.raises(Exception) as exc:
        g.cypher("CALL rev_diff({from: 'v1', to: 'v2'}) YIELD bucket RETURN bucket")
    assert "not a multi-rev graph" in str(exc.value).lower()


def test_rev_diff_steered_in_describe(repo3: Path) -> None:
    g = code_tree.build(str(repo3), revs=["v1", "v2", "v3"])
    # The multi-rev provenance teaches CALL rev_diff for deltas.
    assert "rev_diff" in g.describe()
