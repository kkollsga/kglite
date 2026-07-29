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
"""

from __future__ import annotations

from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parent.parent
CLAUDE = ROOT / ".claude" / "skills"
AGENTS = ROOT / ".agents" / "skills"


def normalise(text: str) -> str:
    """Erase the one difference the two trees are allowed to have."""
    text = text.replace("AGENTS.md", "CLAUDE.md")
    return re.sub(r"^# CLAUDE\.md", "# CLAUDE.md", text, flags=re.M)


def main() -> int:
    if not AGENTS.exists():
        print("skill mirrors: no .agents/skills tree — nothing to compare")
        return 0
    if not CLAUDE.exists():
        print("skill mirrors: .agents/skills exists but .claude/skills does not", file=sys.stderr)
        return 1

    claude = {p.relative_to(CLAUDE) for p in CLAUDE.rglob("SKILL.md")}
    agents = {p.relative_to(AGENTS) for p in AGENTS.rglob("SKILL.md")}
    if not claude:
        print("skill mirrors: found no SKILL.md under .claude/skills — the scan is broken", file=sys.stderr)
        return 1

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

    if problems:
        print("skill mirrors: DRIFTED", file=sys.stderr)
        for p in problems:
            print(f"  {p}", file=sys.stderr)
        return 1
    print(f"skill mirrors: {len(claude)} skill(s) identical across .claude and .agents")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
