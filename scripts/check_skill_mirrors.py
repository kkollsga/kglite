#!/usr/bin/env python3
"""Assert the Claude and Codex instruction trees have not drifted apart.

Every repo here keeps agent instructions twice: `.claude/skills/` for Claude
and `.agents/skills/` for Codex, plus `CLAUDE.md` and `AGENTS.md`. Both are
hand-maintained, so they rot, and a stale copy does not merely lag — it
teaches a procedure the live copy warns against.

That is not hypothetical. On 2026-07-29 KGLite's `.agents/skills/release`
was 194 lines behind and still instructed the reader that the version bump is
"One line: `[workspace.package] version` … there is no per-manifest bump" —
the exact belief that broke the 0.15.0 release, sitting in the file a Codex
session would follow. codingest carried the same error in *both* trees.

The only legitimate difference between the two trees is that each names its
own conventions file, so this normalises `AGENTS.md` -> `CLAUDE.md` and the
`# AGENTS.md` title, then requires the rest to be identical.

The same applies to the root pair itself. `CLAUDE.md` and `AGENTS.md` carry
the whole convention set and nothing compared them until 2026-08-09, by which
point `AGENTS.md` was 26 lines stale: it lacked the "/release self-authorizes
its publish push" block and the "bump size is always patch" block, and still
taught the superseded "the bump size stays a human decision" doctrine — three
release-procedure statements a Codex session would have followed.
"""

from __future__ import annotations

from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parent.parent
CLAUDE = ROOT / ".claude" / "skills"
AGENTS = ROOT / ".agents" / "skills"
ROOT_CLAUDE = ROOT / "CLAUDE.md"
ROOT_AGENTS = ROOT / "AGENTS.md"

# Each root file names itself in its H1: "… — Claude Code Conventions" vs
# "… — Codex Conventions". That, and a cross-reference naming its own file,
# are the only differences the pair is allowed to have.
ROOT_TITLE = re.compile(r"^# (?P<repo>.+?) — (?:Claude Code|Codex) Conventions[ \t]*$", re.M)

SUMMARY: list[str] = []


def normalise(text: str) -> str:
    """Erase the one difference the two trees are allowed to have."""
    text = text.replace("AGENTS.md", "CLAUDE.md")
    return re.sub(r"^# CLAUDE\.md", "# CLAUDE.md", text, flags=re.M)


def normalise_root(text: str) -> str:
    """Erase the differences the two root conventions files are allowed to have."""
    text = text.replace("AGENTS.md", "CLAUDE.md")
    return ROOT_TITLE.sub(r"# \g<repo> — Conventions", text)


def first_difference(a: list[str], b: list[str]) -> tuple[int, str, str]:
    """1-based line number of the first divergence, with both sides' text."""
    for i, (x, y) in enumerate(zip(a, b), start=1):
        if x != y:
            return i, x, y
    n = min(len(a), len(b))
    return n + 1, a[n] if len(a) > n else "<end of file>", b[n] if len(b) > n else "<end of file>"


def _excerpt(line: str, limit: int = 96) -> str:
    line = line.rstrip()
    return line if len(line) <= limit else line[: limit - 1] + "…"


def check_root_pair() -> list[str]:
    """Compare the two root conventions files, rename-aware.

    `AGENTS.md` and `.agents/` are gitignored workstation-local adapters, so a
    fresh clone has neither and there is genuinely nothing to compare. They do
    install as one unit though: once *either* half is present the root pair
    must be present and in sync, otherwise deleting one file would turn the
    check green — the precise silence it exists to prevent.
    """
    if not (ROOT_AGENTS.is_file() or AGENTS.exists()):
        SUMMARY.append("no Codex adapter installed (no AGENTS.md, no .agents/) — nothing to compare")
        return []

    missing = [p.name for p in (ROOT_CLAUDE, ROOT_AGENTS) if not p.is_file()]
    if missing:
        return [
            f"root conventions: the Codex adapter is installed but {', '.join(missing)} is "
            "missing at the repo root — the pair must exist for the mirror check to mean anything"
        ]

    c = normalise_root(ROOT_CLAUDE.read_text(encoding="utf-8"))
    a = normalise_root(ROOT_AGENTS.read_text(encoding="utf-8"))
    if a == c:
        SUMMARY.append("root CLAUDE.md/AGENTS.md identical bar the rename")
        return []

    c_lines, a_lines = c.splitlines(), a.splitlines()
    lineno, claude_line, agents_line = first_difference(c_lines, a_lines)
    drift = sum(1 for x, y in zip(c_lines, a_lines) if x != y) + abs(len(c_lines) - len(a_lines))
    return [
        f"CLAUDE.md vs AGENTS.md: differ beyond the CLAUDE.md/AGENTS.md rename "
        f"(~{drift} line(s)); first at line {lineno}:",
        f"    CLAUDE.md:{lineno}: {_excerpt(claude_line)}",
        f"    AGENTS.md:{lineno}: {_excerpt(agents_line)}",
        "    -> CLAUDE.md is canonical; resync AGENTS.md before an agent follows the stale copy",
    ]


def check_skill_trees() -> list[str]:
    if not AGENTS.exists():
        SUMMARY.append("no .agents/skills tree — no skill trees to compare")
        return []
    if not CLAUDE.exists():
        return ["skill mirrors: .agents/skills exists but .claude/skills does not"]

    claude = {p.relative_to(CLAUDE) for p in CLAUDE.rglob("SKILL.md")}
    agents = {p.relative_to(AGENTS) for p in AGENTS.rglob("SKILL.md")}
    if not claude:
        return ["skill mirrors: found no SKILL.md under .claude/skills — the scan is broken"]

    problems: list[str] = []
    for missing in sorted(claude - agents):
        problems.append(f"{missing}: present in .claude/skills, absent from .agents/skills")
    for extra in sorted(agents - claude):
        problems.append(f"{extra}: present in .agents/skills, absent from .claude/skills")

    for rel in sorted(claude & agents):
        a = normalise((AGENTS / rel).read_text(encoding="utf-8"))
        c = normalise((CLAUDE / rel).read_text(encoding="utf-8"))
        if a != c:
            drift = sum(1 for x, y in zip(a.splitlines(), c.splitlines()) if x != y)
            problems.append(
                f"{rel}: trees differ beyond the CLAUDE.md/AGENTS.md rename "
                f"(~{drift} lines) — one side is stale; sync it before an agent follows it"
            )

    if not problems:
        SUMMARY.append(f"{len(claude)} skill(s) identical across .claude and .agents")
    return problems


def main() -> int:
    problems = check_root_pair() + check_skill_trees()
    if problems:
        print("instruction mirrors: DRIFTED", file=sys.stderr)
        for p in problems:
            print(f"  {p}", file=sys.stderr)
        return 1
    print("instruction mirrors: " + "; ".join(SUMMARY))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
