"""Read/write helpers for MCP server entries in Claude client configs.

Supports three clients out of the box, plus arbitrary custom paths:

- ``claude_desktop`` — Claude Desktop app's ``claude_desktop_config.json``
  (platform-aware: macOS, Windows, Linux).
- ``claude_code`` — Claude Code CLI's ``~/.claude.json`` (same file is
  read by the Claude Code VS Code extension, since it shares the CLI).
- ``vscode`` — VS Code's native MCP support at ``./.vscode/mcp.json``
  (project-scope). Schema differs: uses ``servers`` (not
  ``mcpServers``) and each entry carries an explicit ``type: "stdio"``.

All mutations are atomic (write-to-tmp + ``os.replace``) and preserve
every other top-level key in the config — important because Claude
Desktop's config also stores ``preferences``, ``coworkScheduledTasks*``,
etc. that we must not clobber.

Example::

    from kglite.mcp_server import claude_config

    # Register kglite-mcp-server in Claude Desktop as a workspace
    claude_config.add_mcp(
        name="open-source",
        command="kglite-mcp-server",
        args=["--workspace", "/path/to/repos"],
        client="claude_desktop",
    )

    # List what's registered
    for name, spec in claude_config.list_mcps(client="claude_code").items():
        print(name, spec["command"])

    # Patch an existing entry
    claude_config.edit_mcp("open-source", args=["--workspace", "/new/path"])

    # Custom path (e.g. a per-project config)
    claude_config.list_mcps(path="/tmp/my-config.json")
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any


def _resolve_command(command: str, resolve: bool) -> str:
    """Resolve ``command`` to an absolute path via ``shutil.which`` when ``resolve``
    is True. Bare names (``"kglite-mcp-server"``) become absolute; already-absolute
    paths are returned unchanged when they exist; complex command strings (e.g.
    ``"uv tool run x"``) and unresolvable names pass through unchanged.

    Why default-on: Claude Desktop and Claude Code launch MCP subprocesses with
    a minimal PATH (no user-shell additions), so a bare binary name from a venv
    or conda env almost always fails to spawn. Resolving at config-write time
    sidesteps the silent-failure mode.
    """
    if not resolve:
        return command
    resolved = shutil.which(command)
    return resolved if resolved else command


@dataclass(frozen=True)
class _ClientSpec:
    name: str
    servers_key: str  # "mcpServers" (Claude) vs "servers" (VS Code)
    extra_entry_fields: dict[str, Any]  # e.g. {"type": "stdio"} for VS Code


_CLIENTS: dict[str, _ClientSpec] = {
    "claude_desktop": _ClientSpec("claude_desktop", "mcpServers", {}),
    "claude_code": _ClientSpec("claude_code", "mcpServers", {}),
    "vscode": _ClientSpec("vscode", "servers", {"type": "stdio"}),
}

SUPPORTED_CLIENTS: tuple[str, ...] = tuple(_CLIENTS)


def default_path(client: str = "claude_desktop") -> Path:
    """Return the default config path for ``client`` on the current platform.

    Custom paths can be passed to any mutation via ``path=...`` instead
    of ``client=...``.
    """
    if client == "claude_desktop":
        if sys.platform == "darwin":
            return Path.home() / "Library/Application Support/Claude/claude_desktop_config.json"
        if sys.platform == "win32":
            base = os.environ.get("APPDATA") or str(Path.home() / "AppData/Roaming")
            return Path(base) / "Claude" / "claude_desktop_config.json"
        return Path.home() / ".config/Claude/claude_desktop_config.json"
    if client == "claude_code":
        return Path.home() / ".claude.json"
    if client == "vscode":
        return Path.cwd() / ".vscode" / "mcp.json"
    raise ValueError(f"Unknown client {client!r}; choose from {sorted(SUPPORTED_CLIENTS)}")


def _resolve(client: str | None, path: str | Path | None) -> tuple[_ClientSpec, Path]:
    if path is not None and client is not None:
        raise ValueError("Pass either client= or path=, not both")
    if path is not None:
        # Custom path defaults to the Claude Desktop / Claude Code schema.
        return _CLIENTS["claude_desktop"], Path(path).expanduser()
    chosen = client or "claude_desktop"
    if chosen not in _CLIENTS:
        raise ValueError(f"Unknown client {chosen!r}; choose from {sorted(SUPPORTED_CLIENTS)}")
    return _CLIENTS[chosen], default_path(chosen)


def _load(p: Path) -> dict[str, Any]:
    if not p.exists():
        return {}
    text = p.read_text()
    if not text.strip():
        return {}
    return json.loads(text)


def _save_atomic(p: Path, data: dict[str, Any]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=p.name + ".", dir=str(p.parent))
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2)
            f.write("\n")
        os.replace(tmp_name, p)
    except Exception:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def list_mcps(
    *,
    client: str | None = None,
    path: str | Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Return the MCP server entries in the resolved config file.

    Returns an empty dict if the file does not exist or has no server
    entries. Pass ``client=`` to use a default path, or ``path=`` for a
    custom location.
    """
    spec, p = _resolve(client, path)
    cfg = _load(p)
    return dict(cfg.get(spec.servers_key, {}))


def get_mcp(
    name: str,
    *,
    client: str | None = None,
    path: str | Path | None = None,
) -> dict[str, Any] | None:
    """Return a single MCP server entry, or ``None`` if not present."""
    return list_mcps(client=client, path=path).get(name)


def add_mcp(
    name: str,
    command: str,
    args: list[str] | None = None,
    env: dict[str, str] | None = None,
    *,
    overwrite: bool = False,
    resolve_command: bool = True,
    client: str | None = None,
    path: str | Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Add a new MCP server entry. Returns the written (or would-be) entry.

    By default, ``command`` is resolved to its absolute path via
    ``shutil.which`` so the entry survives Claude Desktop's minimal-PATH
    subprocess environment. Pass ``resolve_command=False`` to store the
    literal string (useful for Docker shims, wrapper scripts, or commands
    with embedded args like ``"uv tool run x"``).

    Raises ``FileExistsError`` if ``name`` already exists, unless
    ``overwrite=True``. The config file is created if missing.
    """
    if not name or not isinstance(name, str):
        raise ValueError("name must be a non-empty string")
    if not command or not isinstance(command, str):
        raise ValueError("command must be a non-empty string")
    spec, p = _resolve(client, path)
    cfg = _load(p)
    servers = cfg.setdefault(spec.servers_key, {})
    if name in servers and not overwrite:
        raise FileExistsError(
            f"MCP server {name!r} already exists in {p}. Pass overwrite=True to replace it, or use edit_mcp()."
        )
    entry: dict[str, Any] = {
        **spec.extra_entry_fields,
        "command": _resolve_command(command, resolve_command),
        "args": list(args or []),
    }
    if env:
        entry["env"] = dict(env)
    servers[name] = entry
    if not dry_run:
        _save_atomic(p, cfg)
    return entry


def edit_mcp(
    name: str,
    *,
    command: str | None = None,
    args: list[str] | None = None,
    env: dict[str, str] | None = None,
    resolve_command: bool = True,
    client: str | None = None,
    path: str | Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Patch fields on an existing MCP server entry. Returns the updated entry.

    Any field set to ``None`` is left untouched. To clear ``env`` set it
    to ``{}``. When ``command`` is given it goes through ``shutil.which``
    resolution unless ``resolve_command=False`` (see ``add_mcp`` for
    rationale). Raises ``KeyError`` if ``name`` is not registered.
    """
    spec, p = _resolve(client, path)
    cfg = _load(p)
    servers = cfg.get(spec.servers_key, {})
    if name not in servers:
        raise KeyError(f"MCP server {name!r} not found in {p}")
    entry = dict(servers[name])
    if command is not None:
        entry["command"] = _resolve_command(command, resolve_command)
    if args is not None:
        entry["args"] = list(args)
    if env is not None:
        if env:
            entry["env"] = dict(env)
        else:
            entry.pop("env", None)
    cfg.setdefault(spec.servers_key, {})[name] = entry
    if not dry_run:
        _save_atomic(p, cfg)
    return entry


def delete_mcp(
    name: str,
    *,
    missing_ok: bool = False,
    client: str | None = None,
    path: str | Path | None = None,
    dry_run: bool = False,
) -> bool:
    """Remove an MCP server entry. Returns True if removed, False if absent.

    Raises ``KeyError`` if ``name`` is not registered, unless
    ``missing_ok=True``.
    """
    spec, p = _resolve(client, path)
    cfg = _load(p)
    servers = cfg.get(spec.servers_key, {})
    if name not in servers:
        if missing_ok:
            return False
        raise KeyError(f"MCP server {name!r} not found in {p}")
    del servers[name]
    if not dry_run:
        _save_atomic(p, cfg)
    return True
