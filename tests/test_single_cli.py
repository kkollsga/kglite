"""The standalone binary and Python wheel share one Rust CLI implementation."""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_python_cli_shim_is_thin() -> None:
    shim = (REPO / "kglite" / "cli.py").read_text(encoding="utf-8")
    assert "_run_cli" in shim
    for forbidden in ("argparse", "click", "typer", "def build", "def install"):
        assert forbidden not in shim


def test_cli_crate_has_one_library_implementation_and_thin_binary() -> None:
    crate = REPO / "crates" / "kglite-cli" / "src"
    library = (crate / "lib.rs").read_text(encoding="utf-8")
    binary = (crate / "main.rs").read_text(encoding="utf-8")
    assert "pub fn run" in library
    assert "kglite_cli::run(std::env::args_os())" in binary
    assert len(binary.splitlines()) <= 6


def test_main_wheel_embeds_shared_cli_without_python_package_dependency() -> None:
    cargo = (REPO / "crates" / "kglite-py" / "Cargo.toml").read_text(encoding="utf-8")
    pyproject = (REPO / "pyproject.toml").read_text(encoding="utf-8")
    assert 'kglite-cli = { path = "../kglite-cli" }' in cargo
    assert 'kglite = "kglite.cli:main"' in pyproject
    dependencies = pyproject.split("dependencies = [", 1)[1].split("]", 1)[0]
    assert "kglite-cli" not in dependencies
