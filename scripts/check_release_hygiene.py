#!/usr/bin/env python3
"""Structural gates on the paperwork a release edits by hand.

Two failure modes have shipped past review because nothing checked for
them:

1. **Merge-mangled CHANGELOG sections.** Integrating two branches that
   both appended to ``## [Unreleased]`` produces two ``### Changed``
   blocks (and two ``### Fixed`` blocks) under the same heading. Both
   0.15.0 integration merges did exactly this and both were hand-fixed
   after a human noticed. ``## [Unreleased]`` is promoted verbatim into
   the release section by the release skill, so a duplicate authored
   here ships as a duplicate.

2. **Surviving ``TODO:`` markers.** ``scripts/refresh_release_constants.py``
   deliberately writes ``TODO: describe what grew since the prior
   baseline`` into ``tests/test_phase5_parity.py`` for the maintainer to
   fill in. Nothing verified that it was filled in, so the marker could
   ride into a release commit unnoticed.

Scope note — only ``## [Unreleased]`` is linted for structure. The
released sections below it are a frozen historical record: they carry
215 bespoke ``###`` headings accumulated over ~200 releases
(``### Performance``, ``### Internal``, ``### Added — <topic>``, …), and
retro-linting them would mean rewriting already-published release notes
to satisfy a gate. Gating ``[Unreleased]`` gates every *future* release
section at the only moment the text is still cheap to fix, because
release sections are produced by promoting ``[Unreleased]`` verbatim.

Wired into ``make gate`` — pure file reading, no subprocess, no imports
of the project.
"""

from __future__ import annotations

from pathlib import Path
import re
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
CHANGELOG = REPO_ROOT / "CHANGELOG.md"

# https://keepachangelog.com/en/1.1.0/ — the six change groups. New
# `[Unreleased]` content uses these and nothing else; a bespoke heading
# is either a typo or a decision that belongs in the prose, not in the
# document structure.
KEEP_A_CHANGELOG_HEADINGS = ("Added", "Changed", "Deprecated", "Removed", "Fixed", "Security")

# Files `scripts/refresh_release_constants.py` rewrites in place and
# hands back to the maintainer to finish. Keep this list in step with
# that script's `PHASE4_TEST` / `PHASE5_TEST` constants; the generated
# JSON baselines are excluded because they are machine-written end to
# end and never carry a maintainer marker.
REFRESH_WRITTEN_FILES = (
    Path("tests/test_phase4_parity.py"),
    Path("tests/test_phase5_parity.py"),
)
TODO_MARKER_RE = re.compile(r"TODO:")

SECTION_RE = re.compile(r"^## ")
SUBSECTION_RE = re.compile(r"^###\s+(?P<heading>.*?)\s*$")


def check_changelog() -> list[str]:
    """Lint the structure of the ``## [Unreleased]`` section."""
    problems: list[str] = []
    lines = CHANGELOG.read_text(encoding="utf-8").splitlines()

    sections = [(i, line) for i, line in enumerate(lines) if SECTION_RE.match(line)]
    if not sections:
        return [f"{CHANGELOG.name}: no `## ` release sections found"]

    first_index, first_heading = sections[0]
    if first_heading.strip() != "## [Unreleased]":
        problems.append(
            f"{CHANGELOG.name}:{first_index + 1}: the first release section must be "
            f"`## [Unreleased]`, found {first_heading.strip()!r} — a release promotion "
            "removed it instead of emptying it"
        )
        return problems

    end = sections[1][0] if len(sections) > 1 else len(lines)
    seen: dict[str, int] = {}
    for offset in range(first_index + 1, end):
        match = SUBSECTION_RE.match(lines[offset])
        if match is None:
            continue
        heading = match.group("heading")
        lineno = offset + 1
        if heading in seen:
            problems.append(
                f"{CHANGELOG.name}:{lineno}: duplicate `### {heading}` under "
                f"[Unreleased] (first at line {seen[heading]}) — merge the two blocks"
            )
        else:
            seen[heading] = lineno
        if heading not in KEEP_A_CHANGELOG_HEADINGS:
            problems.append(
                f"{CHANGELOG.name}:{lineno}: `### {heading}` is not a Keep a Changelog "
                f"group; use one of {', '.join(KEEP_A_CHANGELOG_HEADINGS)}"
            )
    return problems


def check_refresh_todos() -> list[str]:
    """Fail on a `TODO:` marker left behind in a release-constants file."""
    problems: list[str] = []
    for relative in REFRESH_WRITTEN_FILES:
        path = REPO_ROOT / relative
        if not path.is_file():
            problems.append(f"{relative}: missing — scripts/refresh_release_constants.py writes it")
            continue
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if TODO_MARKER_RE.search(line):
                problems.append(f"{relative}:{lineno}: unresolved release-constants marker — {line.strip()!r}")
    return problems


def main() -> int:
    problems = check_changelog() + check_refresh_todos()
    if problems:
        print("release hygiene check failed:", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 1
    print("release hygiene: CHANGELOG [Unreleased] structure and release-constant markers clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
