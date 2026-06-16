"""Unit tests for kglite.claude_config.

All tests use tempfile paths (via ``path=``) — the helper's default-path
machinery is exercised by a single isolated test that monkeypatches
``Path.home`` and ``sys.platform``.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil

import pytest

from kglite import claude_config


def shutil_which_required(binary: str) -> str:
    """Like ``shutil.which`` but skip the test if the binary is unavailable."""
    p = shutil.which(binary)
    if p is None:
        pytest.skip(f"{binary} not on PATH; skipping resolution test")
    return p


def _write(p: Path, data: dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data))


def _read(p: Path) -> dict:
    return json.loads(p.read_text())


# ── default_path ──────────────────────────────────────────────────────


def test_default_path_macos(monkeypatch):
    monkeypatch.setattr("sys.platform", "darwin")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: Path("/home/u")))
    p = claude_config.default_path("claude_desktop")
    assert str(p) == "/home/u/Library/Application Support/Claude/claude_desktop_config.json"


def test_default_path_claude_code():
    p = claude_config.default_path("claude_code")
    assert p.name == ".claude.json"
    assert p.parent == Path.home()


def test_default_path_vscode():
    p = claude_config.default_path("vscode")
    assert p.parts[-2:] == (".vscode", "mcp.json")


def test_default_path_unknown_client():
    with pytest.raises(ValueError, match="Unknown client"):
        claude_config.default_path("nonexistent")


# ── list_mcps / get_mcp ───────────────────────────────────────────────


def test_list_mcps_missing_file(tmp_path):
    assert claude_config.list_mcps(path=tmp_path / "no.json") == {}


def test_list_mcps_empty_file(tmp_path):
    p = tmp_path / "cfg.json"
    p.write_text("")
    assert claude_config.list_mcps(path=p) == {}


def test_list_mcps_returns_copy(tmp_path):
    p = tmp_path / "cfg.json"
    _write(p, {"mcpServers": {"a": {"command": "x", "args": []}}})
    result = claude_config.list_mcps(path=p)
    result["a"]["command"] = "MUTATED"
    # On-disk untouched
    assert _read(p)["mcpServers"]["a"]["command"] == "x"


def test_get_mcp(tmp_path):
    p = tmp_path / "cfg.json"
    _write(p, {"mcpServers": {"a": {"command": "x", "args": []}}})
    assert claude_config.get_mcp("a", path=p) == {"command": "x", "args": []}
    assert claude_config.get_mcp("missing", path=p) is None


# ── add_mcp ───────────────────────────────────────────────────────────


def test_add_mcp_creates_file(tmp_path):
    p = tmp_path / "new.json"
    entry = claude_config.add_mcp("srv", "cmd", ["--a", "1"], path=p)
    assert entry == {"command": "cmd", "args": ["--a", "1"]}
    assert _read(p) == {"mcpServers": {"srv": {"command": "cmd", "args": ["--a", "1"]}}}


def test_add_mcp_with_env(tmp_path):
    p = tmp_path / "cfg.json"
    claude_config.add_mcp("srv", "cmd", env={"KEY": "val"}, path=p)
    assert _read(p)["mcpServers"]["srv"]["env"] == {"KEY": "val"}


def test_add_mcp_preserves_other_keys(tmp_path):
    """Safety property: must not clobber unrelated top-level keys."""
    p = tmp_path / "cfg.json"
    _write(
        p,
        {
            "mcpServers": {"existing": {"command": "x", "args": []}},
            "preferences": {"theme": "dark"},
            "unrelated": [1, 2, 3],
        },
    )
    claude_config.add_mcp("new", "y", path=p)
    cfg = _read(p)
    assert cfg["preferences"] == {"theme": "dark"}
    assert cfg["unrelated"] == [1, 2, 3]
    assert "existing" in cfg["mcpServers"]
    assert "new" in cfg["mcpServers"]


def test_add_mcp_raises_on_existing(tmp_path):
    p = tmp_path / "cfg.json"
    claude_config.add_mcp("srv", "cmd", path=p)
    with pytest.raises(FileExistsError, match="already exists"):
        claude_config.add_mcp("srv", "other", path=p)


def test_add_mcp_overwrite(tmp_path):
    p = tmp_path / "cfg.json"
    claude_config.add_mcp("srv", "cmd", ["--a"], path=p)
    claude_config.add_mcp("srv", "newcmd", ["--b"], overwrite=True, path=p)
    assert _read(p)["mcpServers"]["srv"] == {"command": "newcmd", "args": ["--b"]}


def test_add_mcp_dry_run(tmp_path):
    p = tmp_path / "cfg.json"
    entry = claude_config.add_mcp("srv", "cmd", path=p, dry_run=True)
    assert entry == {"command": "cmd", "args": []}
    assert not p.exists()


def test_add_mcp_vscode_writes_servers_key_with_type(tmp_path, monkeypatch):
    """vscode client writes the `servers` key (not `mcpServers`) and adds
    `type: stdio` to each entry."""
    p = tmp_path / ".vscode" / "mcp.json"
    monkeypatch.setattr(Path, "cwd", classmethod(lambda cls: tmp_path))
    claude_config.add_mcp("srv", "cmd", ["--a"], client="vscode")
    cfg = _read(p)
    assert "servers" in cfg
    assert "mcpServers" not in cfg
    assert cfg["servers"]["srv"] == {"type": "stdio", "command": "cmd", "args": ["--a"]}


def test_add_mcp_validates_name_and_command(tmp_path):
    p = tmp_path / "cfg.json"
    with pytest.raises(ValueError, match="name"):
        claude_config.add_mcp("", "cmd", path=p)
    with pytest.raises(ValueError, match="command"):
        claude_config.add_mcp("srv", "", path=p)


def test_add_mcp_rejects_both_client_and_path(tmp_path):
    with pytest.raises(ValueError, match="either client"):
        claude_config.add_mcp("srv", "cmd", path=tmp_path / "x.json", client="claude_desktop")


# ── command resolution (shutil.which) ────────────────────────────────


def test_add_mcp_resolves_bare_command_to_absolute_path(tmp_path):
    """A bare binary name gets stored as its absolute path so Claude Desktop's
    minimal-PATH subprocess env can find it."""
    p = tmp_path / "cfg.json"
    claude_config.add_mcp("srv", "python3", path=p)
    cmd = _read(p)["mcpServers"]["srv"]["command"]
    assert Path(cmd).is_absolute(), f"expected absolute path, got {cmd!r}"
    assert Path(cmd).name in {"python3", "python3.exe"}


def test_add_mcp_resolve_command_false_preserves_literal(tmp_path):
    """Opt-out for Docker shims, wrapper scripts, or commands with embedded args."""
    p = tmp_path / "cfg.json"
    claude_config.add_mcp("srv", "python3", resolve_command=False, path=p)
    assert _read(p)["mcpServers"]["srv"]["command"] == "python3"


def test_add_mcp_unresolvable_command_passes_through(tmp_path):
    """A binary that isn't on PATH stays as the literal string (no error)."""
    p = tmp_path / "cfg.json"
    claude_config.add_mcp("srv", "no-such-binary-xyzzy-12345", path=p)
    assert _read(p)["mcpServers"]["srv"]["command"] == "no-such-binary-xyzzy-12345"


def test_add_mcp_absolute_path_preserved(tmp_path):
    """An already-absolute path that exists should round-trip unchanged."""
    p = tmp_path / "cfg.json"
    abs_python = shutil_which_required("python3")
    claude_config.add_mcp("srv", abs_python, path=p)
    assert _read(p)["mcpServers"]["srv"]["command"] == abs_python


def test_edit_mcp_resolves_command_too(tmp_path):
    p = tmp_path / "cfg.json"
    claude_config.add_mcp("srv", "/literal/path", resolve_command=False, path=p)
    claude_config.edit_mcp("srv", command="python3", path=p)
    cmd = _read(p)["mcpServers"]["srv"]["command"]
    assert Path(cmd).is_absolute()
    assert Path(cmd).name in {"python3", "python3.exe"}


def test_edit_mcp_resolve_command_false(tmp_path):
    p = tmp_path / "cfg.json"
    claude_config.add_mcp("srv", "python3", path=p)
    claude_config.edit_mcp("srv", command="my-shim", resolve_command=False, path=p)
    assert _read(p)["mcpServers"]["srv"]["command"] == "my-shim"


# ── edit_mcp ──────────────────────────────────────────────────────────


def test_edit_mcp_patches_fields(tmp_path):
    p = tmp_path / "cfg.json"
    claude_config.add_mcp("srv", "cmd", ["--a"], env={"K": "v"}, path=p)
    claude_config.edit_mcp("srv", args=["--b", "--c"], path=p)
    entry = _read(p)["mcpServers"]["srv"]
    assert entry["command"] == "cmd"  # untouched
    assert entry["args"] == ["--b", "--c"]
    assert entry["env"] == {"K": "v"}  # untouched


def test_edit_mcp_clears_env_with_empty_dict(tmp_path):
    p = tmp_path / "cfg.json"
    claude_config.add_mcp("srv", "cmd", env={"K": "v"}, path=p)
    claude_config.edit_mcp("srv", env={}, path=p)
    assert "env" not in _read(p)["mcpServers"]["srv"]


def test_edit_mcp_raises_on_missing(tmp_path):
    p = tmp_path / "cfg.json"
    _write(p, {"mcpServers": {}})
    with pytest.raises(KeyError, match="not found"):
        claude_config.edit_mcp("missing", command="x", path=p)


def test_edit_mcp_dry_run(tmp_path):
    p = tmp_path / "cfg.json"
    claude_config.add_mcp("srv", "cmd", path=p)
    claude_config.edit_mcp("srv", command="newcmd", path=p, dry_run=True)
    assert _read(p)["mcpServers"]["srv"]["command"] == "cmd"


# ── delete_mcp ────────────────────────────────────────────────────────


def test_delete_mcp(tmp_path):
    p = tmp_path / "cfg.json"
    claude_config.add_mcp("srv", "cmd", path=p)
    assert claude_config.delete_mcp("srv", path=p) is True
    assert _read(p)["mcpServers"] == {}


def test_delete_mcp_raises_on_missing(tmp_path):
    p = tmp_path / "cfg.json"
    _write(p, {"mcpServers": {}})
    with pytest.raises(KeyError, match="not found"):
        claude_config.delete_mcp("missing", path=p)


def test_delete_mcp_missing_ok(tmp_path):
    p = tmp_path / "cfg.json"
    _write(p, {"mcpServers": {}})
    assert claude_config.delete_mcp("missing", missing_ok=True, path=p) is False


def test_delete_mcp_preserves_other_keys(tmp_path):
    p = tmp_path / "cfg.json"
    _write(
        p,
        {
            "mcpServers": {"a": {"command": "x", "args": []}},
            "preferences": {"theme": "dark"},
        },
    )
    claude_config.delete_mcp("a", path=p)
    cfg = _read(p)
    assert cfg["preferences"] == {"theme": "dark"}
    assert cfg["mcpServers"] == {}


# ── atomicity ─────────────────────────────────────────────────────────


def test_atomic_write_no_tmp_leak_on_success(tmp_path):
    p = tmp_path / "cfg.json"
    claude_config.add_mcp("srv", "cmd", path=p)
    leftover = [x for x in tmp_path.iterdir() if x.name != "cfg.json"]
    assert leftover == []
