"""Anti-drift invariant: there is exactly ONE MCP server *implementation* —
the Rust `kglite-mcp-server` library (`crates/kglite-mcp-server/`).

The Python MCP server was retired in 0.10.25 to stop two implementations
drifting (duplicate skill dirs, tool descriptions, `applies_when` logic). In
0.10.26 the server library is reachable two ways — both driving the *same*
Rust `run`, so no drift is possible:

  - `cargo install kglite-mcp-server` → the thin `src/main.rs` binary;
  - `pip install kglite` → the `kglite-mcp-server` console script, a thin
    `kglite/mcp_server.py` shim that forwards argv into `kglite._run_mcp_server`
    (the same library, statically linked into the wheel's extension).

These tests fail loudly if a *second implementation* — a parallel Python
server with its own handlers/skills, or a second server crate — creeps back
in. They deliberately ALLOW the thin shim, and guard that it stays thin.
"""

from __future__ import annotations

from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent


def test_no_python_mcp_server_package() -> None:
    """`kglite/mcp_server` must be a single thin shim *module*, never a
    *package* — a package dir is where the retired Python server (handlers,
    skills, request loop) lived. The shim is one file."""
    pkg = _REPO / "kglite" / "mcp_server"
    assert not pkg.exists(), (
        "kglite/mcp_server/ (a package dir) reappeared — the parallel Python "
        "server was retired in 0.10.25. The MCP server is the Rust library; "
        "the Python side is only the thin kglite/mcp_server.py shim."
    )
    assert (_REPO / "kglite" / "mcp_server.py").is_file(), (
        "the thin kglite/mcp_server.py console-script shim is missing"
    )


def test_mcp_shim_is_thin() -> None:
    """The shim forwards into the Rust `run` and reimplements nothing. Guard
    that it stays a forwarder: it must reference `_run_mcp_server` and must not
    pull in an MCP framework / build its own server."""
    shim = (_REPO / "kglite" / "mcp_server.py").read_text(encoding="utf-8")
    assert "_run_mcp_server" in shim, (
        "the shim no longer forwards into the Rust kglite._run_mcp_server entry "
        "point — it must not grow its own server implementation"
    )
    # A parallel Python server would import one of these frameworks. The shim
    # must not — argument parsing and the serve loop are entirely Rust-side.
    for forbidden in ("import mcp", "from mcp", "fastmcp", "rmcp"):
        assert forbidden not in shim, (
            f"kglite/mcp_server.py imports {forbidden!r} — that signals a Python "
            "server reimplementation creeping back into the shim"
        )


def test_console_script_points_at_shim() -> None:
    """The `kglite-mcp-server` console script must drive the bundled Rust
    server via the shim — not a reintroduced Python `server:main`."""
    pyproject = (_REPO / "pyproject.toml").read_text(encoding="utf-8")
    assert 'kglite-mcp-server = "kglite.mcp_server:main"' in pyproject, (
        "the kglite-mcp-server console script no longer points at the thin shim"
    )
    assert "kglite.mcp_server.server:main" not in pyproject, (
        "pyproject re-added a Python *server* console entry point (the retired package-style server), not the thin shim"
    )


def test_single_skills_directory() -> None:
    """Skill `.md` files live in exactly one place — the Rust crate. A second
    skills/ dir is the drift we removed."""
    rust_skills = _REPO / "crates" / "kglite-mcp-server" / "skills"
    assert rust_skills.is_dir(), "the canonical Rust skills/ dir is missing"
    assert (rust_skills / "explore.md").is_file(), "expected bundled skills under the Rust skills/ dir"

    py_skills = _REPO / "kglite" / "mcp_server" / "skills"
    assert not py_skills.exists(), "the old Python skills/ dir reappeared — single source of truth violated"


def test_single_server_implementation_crate() -> None:
    """Exactly one MCP-server crate under crates/, and the server body lives in
    its *library* (so the cargo bin and the wheel shim share one `run`)."""
    crates = _REPO / "crates"
    server_crates = sorted(p.name for p in crates.iterdir() if p.is_dir() and "mcp-server" in p.name)
    assert server_crates == ["kglite-mcp-server"], f"unexpected MCP server crates: {server_crates}"

    lib_rs = crates / "kglite-mcp-server" / "src" / "lib.rs"
    assert lib_rs.is_file(), "kglite-mcp-server must expose a library (src/lib.rs) so the bin + wheel share one run()"
    assert "pub fn run" in lib_rs.read_text(encoding="utf-8"), (
        "kglite-mcp-server's lib must expose `pub fn run` — the single entry point "
        "both the cargo bin and the wheel shim drive"
    )
