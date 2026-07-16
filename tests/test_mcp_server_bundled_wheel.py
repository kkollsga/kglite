"""The bundled MCP server reachable from `pip install kglite`.

As of 0.10.26 the Rust `kglite-mcp-server` lives in the wheel: its *library*
(`crates/kglite-mcp-server/src/lib.rs::run`) is statically linked into the
extension and exposed to Python as `kglite._run_mcp_server`, with the
`kglite-mcp-server` console script (a thin `kglite/mcp_server.py` shim)
forwarding argv into it. So `pip install kglite && kglite-mcp-server ...` runs
the identical server as `cargo install kglite-mcp-server`.

Unlike `test_mcp_server_smoke.py` (which drives the compiled cargo *binary*),
these tests drive the *wheel-hosted* server via `python -m kglite.mcp_server`
— so they run wherever the wheel is importable, no cargo build required. They
reuse the JSON-RPC stdio client from the smoke module.
"""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
from typing import Optional

import pandas as pd

import kglite
from tests.test_mcp_server_smoke import McpClient, _text_content


def _build_fixture_graph(path: Path) -> None:
    g = kglite.KnowledgeGraph()
    nodes = pd.DataFrame({"id": [1, 2, 3, 4], "title": ["Alice", "Bob", "Carol", "Dave"]})
    g.add_nodes(nodes, "Person", "id", "title")
    edges = pd.DataFrame({"src": [1, 2, 3], "dst": [2, 3, 4]})
    g.add_connections(edges, "KNOWS", "Person", "src", "Person", "dst")
    g.save(str(path))


def _spawn_wheel(args: list[str], cwd: Optional[Path] = None) -> McpClient:
    """Launch the bundled server through the shim module (the same code path
    the `kglite-mcp-server` console script runs) and complete the handshake."""
    proc = subprocess.Popen(
        [sys.executable, "-m", "kglite.mcp_server", *args],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=str(cwd) if cwd else None,
    )
    client = McpClient(proc)
    client.initialize()
    return client


def test_entry_point_is_bundled() -> None:
    """The wheel exposes the in-process Rust entry point and the thin shim."""
    assert hasattr(kglite, "_run_mcp_server"), "wheel is missing the bundled MCP server entry point"
    from kglite import mcp_server

    assert callable(mcp_server.main)


def test_bundled_server_boots_and_lists_tools(tmp_path: Path) -> None:
    manifest = tmp_path / "bare_mcp.yaml"
    manifest.write_text("name: Bundled Wheel Smoke\n")
    client = _spawn_wheel(["--mcp-config", str(manifest)])
    try:
        names = {t["name"] for t in client.list_tools()}
        assert "ping" in names
        assert "cypher_query" in names
        assert "graph_overview" in names
    finally:
        client.shutdown()


def test_bundled_server_runs_cypher_on_graph(tmp_path: Path) -> None:
    kgl = tmp_path / "fixture.kgl"
    _build_fixture_graph(kgl)
    client = _spawn_wheel(["--graph", str(kgl)])
    try:
        out = _text_content(client.call_tool("cypher_query", {"query": "MATCH (p:Person) RETURN count(p) AS n"}))
    finally:
        client.shutdown()
    assert "4" in out


def test_python_library_embedder_powers_text_score(tmp_path: Path) -> None:
    """A Python embedder library (`extensions.embedder.library: …`) lets the
    bundled server run `text_score()` via a Python embedder handed in by
    `_run_mcp_server`'s factory (which receives the config JSON). Uses a
    deterministic stub (no network, no real model): identical text → identical
    vector, so the exact-match node ranks top."""
    # A stub embedder shared by build-time (g.embed_texts) and serve-time (the
    # factory), so the stored node vectors and the query vector use one scheme.
    stub_mod = tmp_path / "stub_embed.py"
    stub_mod.write_text(
        "import hashlib\n"
        "class StubEmbedder:\n"
        "    dimension = 8\n"
        "    def embed(self, texts):\n"
        "        return [[float(b) for b in hashlib.sha256(t.encode()).digest()[:8]] for t in texts]\n"
    )
    sys.path.insert(0, str(tmp_path))
    try:
        import stub_embed  # type: ignore

        g = kglite.KnowledgeGraph()
        df = pd.DataFrame(
            {"id": [1, 2, 3], "title": ["A", "B", "C"], "summary": ["alpha alpha", "beta beta", "gamma gamma"]}
        )
        g.add_nodes(df, "Doc", "id", "title")
        g.set_embedder(stub_embed.StubEmbedder())
        g.embed_texts("Doc", "summary", show_progress=False)
        kgl = tmp_path / "docs.kgl"
        g.save(str(kgl))
    finally:
        sys.path.remove(str(tmp_path))

    manifest = tmp_path / "docs_mcp.yaml"
    # `library: stub` (anything but fastembed-rs) routes to the Python factory.
    manifest.write_text(
        "name: stub\ntrust:\n  allow_embedder: true\nextensions:\n  embedder:\n    library: stub\n    model: stub\n"
    )

    # Launch the bundled server with the same stub via the factory arg — the
    # real console-script path dispatches on library; here we inject the stub
    # (ignoring the config JSON) so the test is deterministic and offline.
    launcher = tmp_path / "launch.py"
    launcher.write_text(
        "import sys\n"
        f"sys.path.insert(0, {str(tmp_path)!r})\n"
        "import kglite, stub_embed\n"
        "kglite._run_mcp_server(sys.argv[1:], embedder_factory=lambda cfg: stub_embed.StubEmbedder())\n"
    )
    proc = subprocess.Popen(
        [sys.executable, str(launcher), "--graph", str(kgl), "--mcp-config", str(manifest)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    client = McpClient(proc)
    client.initialize()
    try:
        out = _text_content(
            client.call_tool(
                "cypher_query",
                {
                    "query": "MATCH (d:Doc) RETURN d.title AS t, "
                    "text_score(d, 'summary', 'beta beta') AS s ORDER BY s DESC"
                },
            )
        )
    finally:
        client.shutdown()
    # The query ran (no "requires the pip-hosted server" / embedder error) and
    # the exact-match node B (summary 'beta beta') is present and ranked first.
    assert "error" not in out.lower()[:60], out
    assert "3 row(s)" in out, out
    first_data_line = next(ln for ln in out.splitlines() if "B" in ln or "A" in ln or "C" in ln)
    assert "B" in first_data_line, f"exact-match node B should rank first:\n{out}"


def test_shim_exit_code_on_bad_args(tmp_path: Path) -> None:
    """clap parses argv Rust-side; a bad flag should make the shim exit
    non-zero (the server never reaches the serve loop)."""
    proc = subprocess.run(
        [sys.executable, "-m", "kglite.mcp_server", "--graph", str(tmp_path / "does_not_exist.kgl")],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
    )
    assert proc.returncode != 0


def test_selftest_on_wheel_install(tmp_path: Path) -> None:
    """`--selftest` must work on the *wheel* install, not just the cargo
    binary. Regression: the self-test re-spawns the server, and on the wheel
    `current_exe()` is the Python interpreter (the console script is a shim),
    so a naive re-spawn launched `python <server-flags>` and failed with
    "Unknown option". `kglite.mcp_server.main` now exports KGLITE_MCP_RESPAWN
    so the child is launched via the module entry. Drives the exact wheel
    code path (`python -m kglite.mcp_server`)."""
    kgl = tmp_path / "fixture.kgl"
    _build_fixture_graph(kgl)
    proc = subprocess.run(
        [sys.executable, "-m", "kglite.mcp_server", "--selftest", "--graph", str(kgl)],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=60,
    )
    out = proc.stdout.decode(errors="replace") + proc.stderr.decode(errors="replace")
    assert proc.returncode == 0, out
    assert "Selftest PASSED" in out
    assert "graph hydrates" in out


def test_selftest_on_wheel_install_bad_graph_fails(tmp_path: Path) -> None:
    """The wheel self-test still reports a genuine misconfiguration as a
    non-zero failure (not a false green)."""
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "kglite.mcp_server",
            "--selftest",
            "--graph",
            str(tmp_path / "missing.kgl"),
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=60,
    )
    out = proc.stdout.decode(errors="replace") + proc.stderr.decode(errors="replace")
    assert proc.returncode != 0
    assert "Selftest FAILED" in out


def _write_wide_local_workspace(tmp_path: Path, n_files: int = 400) -> tuple[Path, Path, Path]:
    """A broad `workspace.kind: local` root (many files across dirs) plus a
    tiny representative subdir. Mirrors the deployed code-review archetype's
    shape (a wide sandbox agents narrow with set_root_dir). Returns
    (manifest, root, small_subdir)."""
    root = tmp_path / "wide_root"
    for d in range(n_files // 20):
        pkg = root / f"pkg{d}"
        pkg.mkdir(parents=True, exist_ok=True)
        for f in range(20):
            (pkg / f"m{f}.py").write_text(
                f"class C{f}:\n    def meth(self, x): return x + {f}\ndef fn{f}(a): return a\n"
            )
    small = root / "sub_small"
    small.mkdir(parents=True, exist_ok=True)
    (small / "a.py").write_text("def g(x):\n    return x\nclass W:\n    def r(self): return 1\n")
    manifest = tmp_path / "wide_mcp.yaml"
    manifest.write_text(f"name: Wide Root Selftest\nworkspace: {{ kind: local, root: {root}, watch: true }}\n")
    return manifest, root, small


def test_selftest_wide_local_workspace_does_not_build_root(tmp_path: Path) -> None:
    """Regression (wide-root hang): `--selftest` against a local-workspace
    server with a *wide* root must NOT build a code graph over the whole root
    (that path no client uses; for a broad root it's unbounded → a silent
    hang). It stays registration-only and completes fast. Runs on the wheel
    install, the deployed shape."""
    manifest, _root, _small = _write_wide_local_workspace(tmp_path)
    # A 60s cap that the old build-the-whole-root behaviour would blow on a
    # genuinely wide tree; registration-only returns in a couple of seconds.
    proc = subprocess.run(
        [sys.executable, "-m", "kglite.mcp_server", "--selftest", "--mcp-config", str(manifest)],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=60,
    )
    out = proc.stdout.decode(errors="replace") + proc.stderr.decode(errors="replace")
    assert proc.returncode == 0, out
    assert "Selftest PASSED" in out
    # The contract: the wide root was NOT built; the operator is pointed at
    # --selftest-path. If someone reintroduces set_root_dir(root), this flips.
    assert "not built" in out
    assert "--selftest-path" in out


def test_selftest_path_builds_representative_subdir(tmp_path: Path) -> None:
    """`--selftest-path <subdir>` opts into a real build + hydration against a
    small representative directory (the way to verify a local-workspace build
    without touching the wide root)."""
    manifest, _root, small = _write_wide_local_workspace(tmp_path)
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "kglite.mcp_server",
            "--selftest",
            "--selftest-path",
            str(small),
            "--mcp-config",
            str(manifest),
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=60,
    )
    out = proc.stdout.decode(errors="replace") + proc.stderr.decode(errors="replace")
    assert proc.returncode == 0, out
    assert "Selftest PASSED" in out
    assert "graph hydrates" in out
    assert "count(n)" in out  # a real cypher round-trip ran
