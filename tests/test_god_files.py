"""God-file size gate: reject new Rust source files larger than 2500 LoC.

Phase A.3 / 0.9.53 code-rot audit found one file (`cypher/planner/fusion.rs`)
already past 3000 LoC. The user explicitly accepted that as a deferred
refactor candidate — it's a registry of optimizer fusion passes that
collectively don't fit elsewhere. The allowlist records that decision and
its current ceiling so the file can't silently grow further.

The gate exists to catch the **next** file that drifts past 3000 LoC.
CLAUDE.md says ``Each pass through a file should leave it more
compartmentalised than you found it'' — this test makes the heuristic
mechanical.

If a legitimate new ceiling is needed, bump the entry here (or add a new
one) and explain why in the commit message. If a file just grew without a
clear reason, the right answer is to split it.
"""

from __future__ import annotations

from pathlib import Path

# Resolve the repo root relative to this test file (tests/ sits at the root).
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIRS = [REPO_ROOT / "crates" / "kglite-py" / "src", REPO_ROOT / "crates" / "kglite" / "src"]

# Hard cap ratcheted down by the library-hardening decomposition. Most files
# should remain well under the 1500-line soft target.
DEFAULT_LIMIT = 2500

# Per-file ceilings for files known to be over the default. Each entry pins
# the CURRENT line count so the file can't grow further without an explicit
# bump. Add a one-line justification with each entry.
# Empty: the former `fusion.rs` entry was RESOLVED in 0.10.10 by splitting
# the file into the `planner/fusion/` module directory (count / aggregate /
# topk / spatial submodules + shared helpers in mod.rs), so no single file
# exceeds the default limit. The correct remedy for a god-file is splitting,
# not raising a ceiling.
ALLOWLIST: dict[str, int] = {}


def _count_loc(path: Path) -> int:
    """Total line count (matches `wc -l` semantics — counts newlines)."""
    return sum(1 for _ in path.open("rb"))


def _rel_path(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT))


def test_no_new_god_files():
    """Fail if any .rs file under src/ exceeds 2500 LoC without an allowlist
    entry, or if an allowlisted file exceeds its pinned ceiling."""
    violations: list[str] = []
    rs_files = []
    for src_dir in SRC_DIRS:
        rs_files.extend(src_dir.rglob("*.rs"))
    for path in sorted(rs_files):
        loc = _count_loc(path)
        rel = _rel_path(path)
        ceiling = ALLOWLIST.get(rel, DEFAULT_LIMIT)
        if loc > ceiling:
            if rel in ALLOWLIST:
                violations.append(
                    f"{rel}: {loc} LoC > pinned ceiling {ceiling}. "
                    f"Either split the file or bump the entry in ALLOWLIST."
                )
            else:
                violations.append(
                    f"{rel}: {loc} LoC > default {DEFAULT_LIMIT}. "
                    f"Split the file, or add an ALLOWLIST entry with a justification."
                )
    if violations:
        msg = f"{len(violations)} god-file violation(s) — see tests/test_god_files.py:\n" + "\n".join(
            f"  - {v}" for v in violations
        )
        raise AssertionError(msg)


def test_allowlist_is_not_stale():
    """Allowlisted files must actually exceed the default — otherwise the
    entry is no longer needed and should be deleted (the file split worked)."""
    stale: list[str] = []
    for rel, ceiling in ALLOWLIST.items():
        path = REPO_ROOT / rel
        if not path.exists():
            stale.append(f"{rel}: no longer exists (delete ALLOWLIST entry)")
            continue
        loc = _count_loc(path)
        if loc <= DEFAULT_LIMIT:
            stale.append(
                f"{rel}: {loc} LoC is now ≤ default {DEFAULT_LIMIT}; delete the ALLOWLIST entry (refactor complete)."
            )
        if ceiling < loc:
            stale.append(
                f"{rel}: pinned ceiling {ceiling} < actual {loc}; "
                f"first test would catch this but the allowlist is "
                f"misconfigured — set ceiling ≥ loc."
            )
    if stale:
        msg = f"{len(stale)} stale ALLOWLIST entry(ies):\n" + "\n".join(f"  - {s}" for s in stale)
        raise AssertionError(msg)
