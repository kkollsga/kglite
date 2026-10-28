"""The publish-time wheel-set check can fail, and names what is missing."""

import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "check_wheel_set", Path(__file__).resolve().parent.parent / "scripts" / "check_wheel_set.py"
)
wheel_set = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(wheel_set)

COMPLETE = [
    "kglite-1.2.3-cp312-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
    "kglite-1.2.3-cp312-abi3-musllinux_1_2_x86_64.whl",
    "kglite-1.2.3-cp312-abi3-macosx_11_0_arm64.whl",
    "kglite-1.2.3-cp312-abi3-macosx_10_12_x86_64.whl",
    "kglite-1.2.3-cp312-abi3-win_amd64.whl",
    "kglite-1.2.3-cp315-abi3.abi3t-manylinux_2_28_x86_64.whl",
    "kglite-1.2.3-cp315-abi3.abi3t-macosx_11_0_arm64.whl",
    "kglite-1.2.3-cp315-abi3.abi3t-win_amd64.whl",
    "kglite-1.2.3.tar.gz",
]


def test_a_complete_set_passes():
    assert wheel_set.missing(COMPLETE) == []


def test_a_missing_abi3t_wheel_is_named():
    names = [n for n in COMPLETE if "abi3.abi3t-win_amd64" not in n]
    assert wheel_set.missing(names) == ["free-threaded (abi3t) Windows x86_64"]


def test_the_abi3_wheel_does_not_stand_in_for_the_abi3t_one():
    names = [n for n in COMPLETE if "abi3.abi3t" not in n]
    assert len(wheel_set.missing(names)) == 3


def test_a_missing_sdist_or_platform_wheel_fails():
    assert "sdist" in wheel_set.missing([n for n in COMPLETE if not n.endswith(".tar.gz")])
    assert wheel_set.missing([n for n in COMPLETE if "musllinux" not in n]) == ["abi3 Linux x86_64 musl"]
