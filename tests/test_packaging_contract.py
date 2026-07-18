"""Source contracts for Python extras and their CI consumers.

These tests deliberately inspect the project files without importing KGLite.
They catch metadata drift before a wheel is built; CI separately installs a
real wheel in a clean environment to prove the declared dependencies work.
"""

from __future__ import annotations

import ast
from pathlib import Path
import re
import subprocess
import sys
import zipfile

REPO_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = REPO_ROOT / "pyproject.toml"
WORKFLOWS = REPO_ROOT / ".github" / "workflows"


def _optional_dependency_blocks() -> dict[str, str]:
    text = PYPROJECT.read_text(encoding="utf-8")
    section = text.split("[project.optional-dependencies]", maxsplit=1)[1]
    section = section.split("\n[", maxsplit=1)[0]
    blocks: dict[str, str] = {}
    for match in re.finditer(r"(?m)^(\w+)\s*=\s*\[(.*?)\]", section, re.DOTALL):
        blocks[match.group(1)] = match.group(2)
    return blocks


def _requirement_names(block: str) -> set[str]:
    return {match.group(1).lower().replace("_", "-") for match in re.finditer(r"[\"']([A-Za-z0-9_-]+)", block)}


def test_networkx_extra_declares_every_runtime_dependency() -> None:
    requirements = _requirement_names(_optional_dependency_blocks()["networkx"])
    assert requirements == {"networkx", "pandas"}


def test_pandas_workflows_have_a_named_extra() -> None:
    requirements = _requirement_names(_optional_dependency_blocks()["pandas"])
    assert requirements == {"pandas"}


def test_ci_only_installs_declared_project_extras() -> None:
    declared = set(_optional_dependency_blocks())
    referenced: set[str] = set()
    for workflow in WORKFLOWS.glob("*.yml"):
        text = workflow.read_text(encoding="utf-8")
        for match in re.finditer(r"\.\[([^]]+)\]", text):
            referenced.update(part.strip() for part in match.group(1).split(","))

    assert referenced <= declared, f"workflow references undefined project extras: {sorted(referenced - declared)}"


def test_ci_exercises_networkx_bridge_dependencies() -> None:
    ci = (WORKFLOWS / "ci.yml").read_text(encoding="utf-8")
    assert ".[neo4j,networkx]" in ci


def test_classifiers_match_native_cpython_artifacts() -> None:
    text = PYPROJECT.read_text(encoding="utf-8")
    block = re.search(r"(?ms)^classifiers\s*=\s*(\[.*?\])", text)
    assert block is not None
    classifiers = set(ast.literal_eval(block.group(1)))
    assert "Programming Language :: Python :: Implementation :: CPython" in classifiers
    assert "Programming Language :: Python :: Implementation :: PyPy" not in classifiers
    assert "Operating System :: OS Independent" not in classifiers
    assert {
        "Operating System :: MacOS",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX :: Linux",
    } <= classifiers


def test_wheel_policy_and_support_page_match_workflow() -> None:
    workflow = (WORKFLOWS / "build_wheels.yml").read_text(encoding="utf-8")
    support = (REPO_ROOT / "docs" / "python" / "platform-support.md").read_text(encoding="utf-8")
    targets = set(re.findall(r"(?m)^\s+(?:-\s+)?target:\s+([\w-]+)\s*$", workflow))
    assert targets
    assert all(f"`{target}`" in support for target in targets)
    assert "continue-on-error: true" in workflow
    assert "uploads platform wheels" in support
    assert "source distribution" in support
    assert "dist/*.tar.gz" not in workflow


def test_cp310_abi3_policy_does_not_claim_pypy_compatibility() -> None:
    cargo = (REPO_ROOT / "crates" / "kglite-py" / "Cargo.toml").read_text(encoding="utf-8")
    support = (REPO_ROOT / "docs" / "python" / "platform-support.md").read_text(encoding="utf-8")
    assert 'abi3 = ["pyo3/abi3-py310"]' in cargo
    assert "PyPy is not a supported published-artifact target" in support
    assert "requires a `pp310`-compatible artifact" in support


def test_bundled_mcp_entry_point_stays_declared() -> None:
    pyproject = PYPROJECT.read_text(encoding="utf-8")
    assert 'kglite-mcp-server = "kglite.mcp_server:main"' in pyproject


def test_bundled_cli_entry_point_stays_declared() -> None:
    pyproject = PYPROJECT.read_text(encoding="utf-8")
    assert 'kglite = "kglite.cli:main"' in pyproject


def test_installed_wheel_smoke_executes_console_script() -> None:
    ci = (WORKFLOWS / "ci.yml").read_text(encoding="utf-8")
    assert "/tmp/kglite-networkx-smoke/bin/kglite --help" in ci
    assert "/tmp/kglite-networkx-smoke/bin/kglite-mcp-server --help" in ci


def test_ci_compiles_packaged_optional_features_outside_workspace() -> None:
    ci = (WORKFLOWS / "ci.yml").read_text(encoding="utf-8")
    script = (REPO_ROOT / "scripts" / "check_packaged_features.sh").read_text(encoding="utf-8")
    assert "bash scripts/check_packaged_features.sh" in ci
    assert "cargo package -p kglite" in script
    assert "--features parallel-bz2" in script


def _write_fake_wheel(
    path: Path,
    *,
    distribution: str = "kglite",
    include_extension: bool = True,
    license_expression: str = "MIT",
    license_bytes: bytes | None = None,
) -> None:
    dist_info = f"{distribution}-0.0.0.dist-info"
    with zipfile.ZipFile(path, "w") as wheel:
        wheel.writestr("kglite/cli.py", "")
        wheel.writestr("kglite/mcp_server.py", "")
        if include_extension:
            wheel.writestr("kglite/kglite.abi3.so", b"native")
        wheel.writestr(
            f"{dist_info}/entry_points.txt",
            "[console_scripts]\nkglite = kglite.cli:main\nkglite-mcp-server = kglite.mcp_server:main\n",
        )
        wheel.writestr(
            f"{dist_info}/METADATA",
            "\n".join(
                [
                    "Metadata-Version: 2.4",
                    f"Name: {distribution}",
                    "Version: 0.0.0",
                    f"License-Expression: {license_expression}",
                    "License-File: LICENSE",
                    "",
                ]
            ),
        )
        wheel.writestr(
            f"{dist_info}/licenses/LICENSE",
            license_bytes if license_bytes is not None else (REPO_ROOT / "LICENSE").read_bytes(),
        )


def test_wheel_inventory_checks_native_extension_and_mcp_entry_point(tmp_path: Path) -> None:
    checker = REPO_ROOT / "scripts" / "check_wheel_artifact.py"
    valid = tmp_path / "valid.whl"
    invalid = tmp_path / "invalid.whl"
    _write_fake_wheel(valid)
    _write_fake_wheel(invalid, include_extension=False)

    subprocess.run([sys.executable, checker, valid], check=True)
    result = subprocess.run(
        [sys.executable, checker, invalid],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert "missing the native kglite extension" in result.stderr


def test_release_workflow_inventories_all_wheels_and_smokes_native_targets() -> None:
    workflow = (WORKFLOWS / "build_wheels.yml").read_text(encoding="utf-8")
    assert workflow.count('python scripts/check_wheel_artifact.py "wheels/*.whl"') == 3
    assert "scripts/smoke_installed_wheel.py" in workflow
    assert "matrix.target == 'x86_64-pc-windows-msvc'" in workflow
    assert "matrix.target == 'aarch64-apple-darwin'" in workflow
    assert "matrix.target == 'x86_64-unknown-linux-gnu'" in workflow


def _run_license_checker(path: Path, *, expected_name: str = "kglite") -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            REPO_ROOT / "scripts" / "check_wheel_license.py",
            "--expected-name",
            expected_name,
            path,
        ],
        capture_output=True,
        text=True,
        check=False,
    )


def test_wheel_license_checker_accepts_exact_metadata_and_bytes(tmp_path: Path) -> None:
    wheel = tmp_path / "valid.whl"
    _write_fake_wheel(wheel)

    result = _run_license_checker(wheel)

    assert result.returncode == 0
    assert "MIT metadata+LICENSE=verified" in result.stdout


def test_wheel_license_checker_rejects_wrong_distribution(tmp_path: Path) -> None:
    wheel = tmp_path / "wrong-name.whl"
    _write_fake_wheel(wheel, distribution="another-project")

    result = _run_license_checker(wheel)

    assert result.returncode == 1
    assert "expected Name: kglite" in result.stderr


def test_wheel_license_checker_rejects_wrong_expression(tmp_path: Path) -> None:
    wheel = tmp_path / "wrong-expression.whl"
    _write_fake_wheel(wheel, license_expression="Apache-2.0")

    result = _run_license_checker(wheel)

    assert result.returncode == 1
    assert "expected License-Expression: MIT" in result.stderr


def test_wheel_license_checker_rejects_changed_license(tmp_path: Path) -> None:
    wheel = tmp_path / "changed-license.whl"
    _write_fake_wheel(wheel, license_bytes=b"not the project license")

    result = _run_license_checker(wheel)

    assert result.returncode == 1
    assert "embedded LICENSE differs" in result.stderr


def test_wheel_license_checker_rejects_duplicate_metadata(tmp_path: Path) -> None:
    wheel = tmp_path / "duplicate-metadata.whl"
    _write_fake_wheel(wheel)
    with zipfile.ZipFile(wheel, "a") as archive:
        archive.writestr("duplicate-0.0.0.dist-info/METADATA", "Name: kglite\n")

    result = _run_license_checker(wheel)

    assert result.returncode == 1
    assert "expected one .dist-info/METADATA, found 2" in result.stderr


def test_wheel_license_gate_covers_every_published_wheel_family() -> None:
    main_release = (WORKFLOWS / "build_wheels.yml").read_text(encoding="utf-8")
    cli_release = (WORKFLOWS / "build_cli_wheels.yml").read_text(encoding="utf-8")
    ci = (WORKFLOWS / "ci.yml").read_text(encoding="utf-8")
    documentation = (REPO_ROOT / "docs" / "explanation" / "dependency-licenses.md").read_text(encoding="utf-8")

    assert main_release.count('python scripts/check_wheel_artifact.py "wheels/*.whl"') == 3
    assert cli_release.count("python scripts/check_wheel_license.py --expected-name kglite-cli") == 3
    assert 'scripts/check_wheel_license.py --expected-name kglite "$RUNNER_TEMP/candidate-wheel/*.whl"' in ci
    assert "wheel-only" in documentation
    assert "does not build or validate source distributions or generate SBOMs" in " ".join(documentation.split())
