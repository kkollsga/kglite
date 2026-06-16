"""Anti-drift invariant: there is exactly ONE MCP server — the Rust
`kglite-mcp-server` binary (`crates/kglite-mcp-server/`).

The Python MCP server was retired in 0.10.25 to stop the two implementations
drifting (duplicate skill dirs, tool descriptions, and `applies_when` logic).
These tests fail loudly if a second implementation — or the old Python one —
creeps back in.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent


def test_python_mcp_server_package_is_gone() -> None:
    assert importlib.util.find_spec("kglite.mcp_server") is None, (
        "kglite.mcp_server is importable again — the Python MCP server was "
        "retired in 0.10.25; the MCP server is the Rust kglite-mcp-server binary."
    )
    assert not (_REPO / "kglite" / "mcp_server").exists(), "kglite/mcp_server/ reappeared on disk"


def test_no_python_mcp_console_script() -> None:
    """`pip install kglite` must NOT ship a Python `kglite-mcp-server` entry
    point — the server is `cargo install kglite-mcp-server`."""
    pyproject = (_REPO / "pyproject.toml").read_text(encoding="utf-8")
    assert "kglite.mcp_server.server:main" not in pyproject, (
        "pyproject re-added the Python console script for kglite-mcp-server"
    )


def test_single_skills_directory() -> None:
    """Skill `.md` files live in exactly one place — the Rust crate. A second
    skills/ dir is the drift we removed."""
    rust_skills = _REPO / "crates" / "kglite-mcp-server" / "skills"
    assert rust_skills.is_dir(), "the canonical Rust skills/ dir is missing"
    assert (rust_skills / "explore.md").is_file(), "expected bundled skills under the Rust skills/ dir"

    py_skills = _REPO / "kglite" / "mcp_server" / "skills"
    assert not py_skills.exists(), "the old Python skills/ dir reappeared — single source of truth violated"


def test_no_second_mcp_server_crate() -> None:
    """Exactly one MCP-server crate under crates/."""
    crates = _REPO / "crates"
    server_crates = sorted(p.name for p in crates.iterdir() if p.is_dir() and "mcp-server" in p.name)
    assert server_crates == ["kglite-mcp-server"], f"unexpected MCP server crates: {server_crates}"
