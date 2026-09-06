"""A generated header must name real status symbols in its error contracts."""

from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def test_documented_c_status_names_resolve_to_the_published_enum():
    header = (ROOT / "crates/kglite-c/include/kglite.h").read_text(encoding="utf-8")
    declarations = set(re.findall(r"^\s*(KGLITE_STATUS_CODE_[A-Z0-9_]+)\s*=", header, re.M))
    assert {"KGLITE_STATUS_CODE_INVALID_ARGUMENT", "KGLITE_STATUS_CODE_NULL_POINTER"} <= declarations
    comments = "\n".join(re.findall(r"/\*.*?\*/", header, re.S))
    documented = set(re.findall(r"\bKGLITE_(?:ERR_|STATUS_CODE_)[A-Z0-9_]+\b", comments))
    assert documented, "the published header must describe its error statuses"
    assert documented <= declarations, f"undefined status names in C documentation: {sorted(documented - declarations)}"
