#!/usr/bin/env python3
"""Ecosystem version-consistency checker and downstream release notifier.

WHY THIS EXISTS
---------------
A version requirement that *understates its real minimum*, or that is declared
*outside package metadata*, is structurally invisible to the repo that declares
it. Only a consumer — or a second declaration site — ever sees the disagreement.
Three real failures of that single class, all on 2026-07-27:

1. ``kglite-mcp-server`` declared ``mcp-methods = "0.4"`` while calling APIs
   added in 0.4.1. Our ``Cargo.lock`` held 0.4.1, so every local build and every
   CI run here resolved fine; a sibling with an older lock hit a compile failure
   against the published crate.
2. ``fastembed = "5"`` selects a feature that first existed in 5.9.0;
   ``mimalloc = "0.1"`` selects ``v2``, added in 0.1.49. Same shape.
3. codingest's ``.github/workflows/ci.yml`` pinned ``kglite==0.14.5`` while its
   wheel metadata required ``>=0.15.0,<0.16``. ``pip check`` fails on push, and
   *no local gate could see it* because the Makefile provisions its own venv.

So this script does two things:

* **check** — enumerate every place a dependency version is declared across the
  ecosystem (manifests, lockfiles, CI YAML, Dockerfiles, Makefiles, shell and
  Python scripts, and docs), then report disagreements.
* **notify** — write a release note into each *genuinely affected* downstream's
  ``inbox/unread/``. Deliberately NOT into every downstream's: see
  ``AFFECTED-ONLY`` below.

FINDING CATEGORIES
------------------
``contradiction``
    Two sites in the same repo name mutually exclusive requirements for the same
    dependency (failure 3). Latent build break. Exits non-zero.
``understated-floor``
    A manifest floor sits *below* the version the committed lockfile actually
    resolved (failure 1 and 2). Not provably wrong — plenty of floors float on
    purpose — but it is precisely the shape that is invisible locally, because
    the lock papers over it. Reported as a warning with the evidence.
``stale-downstream``
    A downstream's declared range on an ecosystem package *excludes* the current
    published upstream version. That repo cannot install the latest engine.
``exact-pin``
    A site pins an ecosystem package at an exact version that has been
    superseded. Not an error; it is the thing that silently rots.
``citation``
    Prose that *records* a version — "introduced in X", "the X → Y move",
    "verified against X on <date>" — rather than requiring one. Classified and
    listed, never reported as drift: telling a downstream to move a provenance
    line to the current version asks it to falsify its own record.
``site``
    Inventory: a declaration living outside package metadata. Always listed,
    even when consistent, because these are the sites that *can* drift with
    nothing watching them.

AFFECTED-ONLY (the notifier's core rule)
----------------------------------------
CLAUDE.md: *"Route to the party who can act. A note only belongs in another
project's inbox if it carries an actionable task for them."* and *"unread/ must
reflect only what still needs doing, so a stale 'you still have unread mail'
never hides a genuinely open item among resolved ones."*

A note that says "we released, nothing changes for you" is noise that trains a
maintainer to ignore the inbox — it actively degrades a mechanism that
currently works. So ``--notify`` writes a note only when the downstream is
genuinely affected:

* its declared range **excludes** the new version (blocked), or
* it pins the upstream at a now-superseded exact version somewhere, or
* a breaking change in this release touches a symbol the repo actually
  references (approximated by scanning its *code* — never comments or prose —
  for the symbols ``BREAKING_SYMBOLS_BY_VERSION`` records against **this**
  version, or the ones given with ``--breaking-symbol``).

Each note's "what changed" section is read from the announced release's own
``## [X.Y.Z]`` CHANGELOG entry. When that entry cannot be resolved the section
is omitted and a warning goes to stderr: a blurb that is silently a different
release's is worse than an absent one, and three shipped notes proved it.

When a downstream is unaffected the run prints ``SKIP <repo>: <reason>`` and
writes nothing. The decision *not* to notify is as much a feature as the note.

USAGE
-----
    python scripts/check_version_consistency.py                  # check
    python scripts/check_version_consistency.py --json           # machine output
    python scripts/check_version_consistency.py --notify --dry-run
    python scripts/check_version_consistency.py --notify         # writes inboxes

Exit codes: 0 clean, 1 contradictions found (or ``--fail-on stale`` tripped),
2 bad invocation. Missing sibling repos are skipped with a message, never a
crash — see ``--require-siblings`` to invert that for a release gate.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as _dt
import json
import os
from pathlib import Path
import re
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent


def _default_ecosystem_root(repo_root: Path) -> Path:
    """Where the sibling repos live.

    Normally ``../``. But this repo is routinely worked from a git worktree
    parked outside the ecosystem tree (``/Users/Shared/kglite-wt/<branch>``),
    where ``../`` holds only other worktrees and every sibling looks absent. A
    worktree's ``.git`` is a *file* pointing at the main checkout, so follow it
    and use the main checkout's parent instead.
    """
    env = os.environ.get("KGLITE_ECOSYSTEM_ROOT")
    if env:
        return Path(env)
    dotgit = repo_root / ".git"
    if dotgit.is_file():
        try:
            text = dotgit.read_text(encoding="utf-8").strip()
        except OSError:
            return repo_root.parent
        if text.startswith("gitdir:"):
            gitdir = Path(text.split(":", 1)[1].strip())
            # .../<main>/.git/worktrees/<name> -> <main>
            for parent in gitdir.parents:
                if parent.name == ".git":
                    return parent.parent.parent
    return repo_root.parent


ECOSYSTEM_ROOT = _default_ecosystem_root(REPO_ROOT)

# --------------------------------------------------------------------------
# Ecosystem configuration
# --------------------------------------------------------------------------

#: Sibling repos, in dependency order. ``upstream`` is the release source.
UPSTREAM_REPO = "KGLite"
DOWNSTREAM_REPOS = ("codingest", "kglite-datasets", "sonagram", "sonara", "mcp-methods")

#: Package names published by this ecosystem, grouped by the repo that owns
#: them. Any declaration of one of these anywhere is cross-repo coupling.
ECOSYSTEM_PACKAGES: dict[str, tuple[str, ...]] = {
    "KGLite": (
        "kglite",
        "kglite-c",
        "kglite-cli",
        "kglite-mcp-server",
        "kglite-bolt-server",
    ),
    "codingest": ("codingest", "codingest-py", "codingest-mcp"),
    "kglite-datasets": ("kglite-datasets",),
    "sonagram": ("sonagram",),
    "sonara": ("sonara", "sonara-python"),
    "mcp-methods": ("mcp-methods", "mcp-methods-macros"),
}

#: Third-party dependencies whose floors have burned us. These are watched for
#: `understated-floor` across every repo. Add to this list when a new one bites;
#: the value is the version at which the *feature we actually use* appeared, or
#: None to just compare against the lockfile.
WATCHED_THIRD_PARTY: dict[str, str | None] = {
    "fastembed": "5.9.0",  # the feature selected by `fastembed = "5"` is 5.9.0+
    "mimalloc": "0.1.49",  # the `v2` feature landed in 0.1.49
    "rmcp": None,
    "rmcp-macros": None,
    "pyo3": None,
}

#: Files we never walk into.
SKIP_DIRS = {
    ".git",
    "target",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    ".mypy_cache",
    ".ruff_cache",
    ".pytest_cache",
    ".uv-cache",
    ".hypothesis",
    "dist",
    "build",
    "_build",
    ".idea",
    ".vscode",
    "inbox",
    "dev-docs",
    ".agents",
    "site-packages",
    # Local working notes, not published surface. A stale version in someone's
    # design scratchpad is not an ecosystem finding and must never justify a
    # note in their inbox.
    "dev-documentation",
    "notes",
    "scratch",
}

MAX_FILE_BYTES = 2_000_000


# --------------------------------------------------------------------------
# Version + requirement algebra
# --------------------------------------------------------------------------

INF = (1 << 30, 0, 0)


def parse_version(text: str) -> tuple[int, int, int] | None:
    """Parse a dotted version into a 3-tuple, ignoring pre-release/build tails."""
    m = re.match(r"^\s*v?(\d+)(?:\.(\d+))?(?:\.(\d+))?", text.strip())
    if not m:
        return None
    return (int(m.group(1)), int(m.group(2) or 0), int(m.group(3) or 0))


def _next_caret(v: tuple[int, int, int], parts: int) -> tuple[int, int, int]:
    """Upper bound (exclusive) of a caret/compatible requirement.

    Cargo semantics, which PEP 440's ``~=`` matches closely enough for our use:
    the left-most non-zero component is the one held fixed.
    ``^1.2.3`` -> 2.0.0, ``^0.14.4`` -> 0.15.0, ``^0.0.3`` -> 0.0.4.
    A partial requirement widens: ``^0.14`` -> 0.15.0, ``^5`` -> 6.0.0.
    """
    major, minor, patch = v
    if major != 0:
        return (major + 1, 0, 0)
    if minor != 0 or parts >= 2:
        return (0, minor + 1, 0)
    return (0, 0, patch + 1)


@dataclasses.dataclass(frozen=True)
class Interval:
    """Half-open version interval ``[lo, hi)``."""

    lo: tuple[int, int, int]
    hi: tuple[int, int, int]

    def __bool__(self) -> bool:
        return self.lo < self.hi

    def contains(self, v: tuple[int, int, int]) -> bool:
        return self.lo <= v < self.hi

    def intersect(self, other: Interval) -> Interval:
        return Interval(max(self.lo, other.lo), min(self.hi, other.hi))


FULL = Interval((0, 0, 0), INF)

_CLAUSE_RE = re.compile(r"^\s*(\^|~=|~|>=|<=|==|=|>|<|!=)?\s*v?([0-9][0-9A-Za-z.\-+*]*)\s*$")


def parse_requirement(spec: str) -> Interval | None:
    """Turn a Cargo or PEP 440 requirement string into an interval.

    Returns ``None`` when the spec is a wildcard, a URL, a git ref, or anything
    else we deliberately decline to reason about — the caller treats that as
    "no constraint" rather than guessing.
    """
    spec = spec.strip()
    if not spec or spec in {"*", "any"}:
        return None
    if spec.startswith(("http", "git+", "file:", "./", "../")):
        return None

    result = FULL
    saw_clause = False
    for raw in re.split(r"[,;]", spec):
        raw = raw.strip()
        if not raw:
            continue
        # Drop PEP 440 environment markers and extras.
        raw = re.sub(r"\s*\[[^\]]*\]", "", raw)
        if raw.endswith(".*"):
            raw = raw[:-2]
            m = _CLAUSE_RE.match(raw)
            if not m:
                return None
            base = parse_version(m.group(2))
            if base is None:
                return None
            parts = len(m.group(2).split("."))
            clause = Interval(base, _next_caret(base, parts + 1))
            result = result.intersect(clause)
            saw_clause = True
            continue

        m = _CLAUSE_RE.match(raw)
        if not m:
            return None
        op = m.group(1) or "^"  # Cargo's bare `1.2` means `^1.2`
        text = m.group(2)
        base = parse_version(text)
        if base is None:
            return None
        parts = len([p for p in text.split(".") if p and p[0].isdigit()])

        if op in {"^", "~="}:
            clause = Interval(base, _next_caret(base, parts))
        elif op == "~":
            # Cargo tilde: `~1.2.3` -> <1.3.0, `~1.2` -> <1.3.0, `~1` -> <2.0.0
            if parts >= 2:
                clause = Interval(base, (base[0], base[1] + 1, 0))
            else:
                clause = Interval(base, (base[0] + 1, 0, 0))
        elif op in {"=", "=="}:
            if parts >= 3:
                clause = Interval(base, (base[0], base[1], base[2] + 1))
            else:
                clause = Interval(base, _next_caret(base, parts + 1))
        elif op == ">=":
            clause = Interval(base, INF)
        elif op == ">":
            clause = Interval((base[0], base[1], base[2] + 1), INF)
        elif op == "<":
            clause = Interval((0, 0, 0), base)
        elif op == "<=":
            clause = Interval((0, 0, 0), (base[0], base[1], base[2] + 1))
        elif op == "!=":
            continue  # recorded but not modelled; exclusions are rare here
        else:
            return None
        result = result.intersect(clause)
        saw_clause = True

    return result if saw_clause else None


def fmt_version(v: tuple[int, int, int]) -> str:
    return "∞" if v >= INF else "{}.{}.{}".format(*v)


# --------------------------------------------------------------------------
# Declaration records
# --------------------------------------------------------------------------


#: Sites whose disagreement is a real build break. Prose (`docs`) is excluded
#: deliberately: a migration guide or a README naming an old version is stale,
#: not contradictory, and conflating the two makes the gate cry wolf.
#: Sites that *execute* an install or are consumed by a resolver. Only these can
#: contradict each other.
#:
#: `script` and `source-string` are deliberately excluded. A version literal in
#: a Python script or a Rust string is usually advisory — a docstring, a help
#: message, an error hint — and treating it as binding produces false
#: contradictions that would block a release. (This script's own docstring cites
#: `kglite==0.14.5` as an example and promptly reported itself.) They are still
#: inventoried and still checked for staleness; they just cannot fail the gate.
BINDING_SITES = {
    "cargo-manifest",
    "python-metadata",
    "python-requirements",
    "python-constraints",
    "ci-yaml",
    "dockerfile",
    "makefile",
    "shell",
    "compose",
    "gradle-build",
}
DOC_SITES = {"docs", "docs-install"}
ADVISORY_SITES = {"script", "source-string"}
#: Prose that *records* a version rather than requiring one — see
#: ``is_version_citation``. Reported as an adjudicated class, never as drift.
CITATION_SITES = {"docs-citation"}


@dataclasses.dataclass
class Declaration:
    repo: str
    path: Path
    line: int
    package: str
    spec: str
    site: str  # cargo-manifest | cargo-lock | python-metadata | ...
    raw: str
    metadata: bool  # True when this is authoritative package metadata
    unit: Path = dataclasses.field(default=Path("."))  # resolution unit root

    @property
    def rel(self) -> str:
        try:
            return str(self.path.relative_to(ECOSYSTEM_ROOT))
        except ValueError:
            return str(self.path)

    @property
    def binding(self) -> bool:
        return self.site in BINDING_SITES

    @property
    def interval(self) -> Interval | None:
        return parse_requirement(self.spec)

    def as_dict(self) -> dict:
        return {
            "repo": self.repo,
            "path": str(self.path),
            "line": self.line,
            "package": self.package,
            "spec": self.spec,
            "site": self.site,
            "metadata": self.metadata,
        }


@dataclasses.dataclass
class Finding:
    kind: str
    severity: str  # error | warn | info
    repo: str
    package: str
    message: str
    declarations: list[Declaration] = dataclasses.field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "kind": self.kind,
            "severity": self.severity,
            "repo": self.repo,
            "package": self.package,
            "message": self.message,
            "declarations": [d.as_dict() for d in self.declarations],
        }


# --------------------------------------------------------------------------
# Scanners — one per declaration-site type
# --------------------------------------------------------------------------


def _tracked(name: str, tracked: set[str]) -> bool:
    return name.lower().replace("_", "-") in tracked


def _iter_files(root: Path):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".sonagram")]
        for name in filenames:
            path = Path(dirpath) / name
            try:
                if path.stat().st_size > MAX_FILE_BYTES and name != "Cargo.lock":
                    continue
            except OSError:
                continue
            yield path


def _read(path: Path) -> list[str] | None:
    try:
        return path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None


# ---- Cargo manifests ------------------------------------------------------

_CARGO_TABLE_RE = re.compile(r"^\s*\[([^\]]+)\]")
_CARGO_INLINE_RE = re.compile(r"^\s*(?P<name>[A-Za-z0-9_.-]+)\s*=\s*(?P<rhs>.+?)\s*$")
_CARGO_VERSION_IN_TABLE = re.compile(r'\bversion\s*=\s*"([^"]+)"')
_CARGO_PKG_RENAME = re.compile(r'\bpackage\s*=\s*"([^"]+)"')


def scan_cargo_manifest(repo: str, path: Path, tracked: set[str]) -> list[Declaration]:
    lines = _read(path)
    if lines is None:
        return []
    out: list[Declaration] = []
    table = ""
    for i, line in enumerate(lines, 1):
        m = _CARGO_TABLE_RE.match(line)
        if m:
            table = m.group(1).strip()
            continue
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        # `[dependencies.foo]` style
        dep_table = re.match(
            r"^(?:workspace\.)?(?:dependencies|dev-dependencies|build-dependencies|"
            r"target\..*\.dependencies)(?:\.([A-Za-z0-9_-]+))?$",
            table,
        )
        if dep_table and dep_table.group(1):
            vm = _CARGO_VERSION_IN_TABLE.search(line)
            if vm and _tracked(dep_table.group(1), tracked):
                out.append(
                    Declaration(
                        repo, path, i, dep_table.group(1), vm.group(1), "cargo-manifest", stripped, metadata=True
                    )
                )
            continue
        if not dep_table:
            continue

        im = _CARGO_INLINE_RE.match(line)
        if not im:
            continue
        name, rhs = im.group("name"), im.group("rhs")
        rename = _CARGO_PKG_RENAME.search(rhs)
        if rename:
            name = rename.group(1)
        if not _tracked(name, tracked):
            continue
        if rhs.startswith('"'):
            spec = rhs.strip('", ')
        else:
            vm = _CARGO_VERSION_IN_TABLE.search(rhs)
            if not vm:
                # path-only or git-only dependency: a real declaration site with
                # *no* version — worth surfacing on ecosystem packages.
                if "path" in rhs or "git" in rhs:
                    out.append(Declaration(repo, path, i, name, "", "cargo-manifest-pathonly", stripped, metadata=True))
                continue
            spec = vm.group(1)
        out.append(Declaration(repo, path, i, name, spec, "cargo-manifest", stripped, metadata=True))
    return out


def scan_cargo_lock(repo: str, path: Path, tracked: set[str]) -> list[Declaration]:
    lines = _read(path)
    if lines is None:
        return []
    out: list[Declaration] = []
    name = None
    name_line = 0
    for i, line in enumerate(lines, 1):
        s = line.strip()
        if s.startswith('name = "'):
            name = s[8:].rstrip('"')
            name_line = i
        elif s.startswith('version = "') and name and _tracked(name, tracked):
            out.append(
                Declaration(
                    repo, path, name_line, name, "=" + s[11:].rstrip('"'), "cargo-lock", f"{name} {s}", metadata=False
                )
            )
            name = None
    return out


# ---- Python metadata ------------------------------------------------------

_PY_REQ_RE = re.compile(r'^\s*["\'](?P<name>[A-Za-z0-9._-]+)\s*(?P<extras>\[[^\]]*\])?\s*(?P<spec>[^"\']*)["\']')


def scan_pyproject(repo: str, path: Path, tracked: set[str]) -> list[Declaration]:
    lines = _read(path)
    if lines is None:
        return []
    out: list[Declaration] = []
    for i, line in enumerate(lines, 1):
        s = line.strip()
        if s.startswith("#"):
            continue
        m = _PY_REQ_RE.match(line)
        if m and _tracked(m.group("name"), tracked):
            out.append(
                Declaration(
                    repo, path, i, m.group("name"), m.group("spec").strip(), "python-metadata", s, metadata=True
                )
            )
            continue
        # `dependencies = ["kglite>=0.15,<0.16"]` on one line
        for dm in re.finditer(r'["\']([A-Za-z0-9._-]+)(\[[^\]]*\])?\s*([<>=!~][^"\']*)["\']', line):
            if _tracked(dm.group(1), tracked):
                d = Declaration(repo, path, i, dm.group(1), dm.group(3).strip(), "python-metadata", s, metadata=True)
                if not any(x.line == i and x.package == d.package for x in out):
                    out.append(d)
    return out


_REQ_LINE_RE = re.compile(r"^\s*(?P<name>[A-Za-z0-9._-]+)\s*(?:\[[^\]]*\])?\s*(?P<spec>(?:[<>=!~][^\s;#]*)?)")


def scan_requirements(repo: str, path: Path, tracked: set[str]) -> list[Declaration]:
    lines = _read(path)
    if lines is None:
        return []
    site = "python-constraints" if "constraint" in path.name else "python-requirements"
    out: list[Declaration] = []
    for i, line in enumerate(lines, 1):
        s = line.strip()
        if not s or s.startswith(("#", "-", "http")):
            continue
        m = _REQ_LINE_RE.match(s)
        if m and _tracked(m.group("name"), tracked):
            out.append(Declaration(repo, path, i, m.group("name"), m.group("spec").strip(), site, s, metadata=False))
    return out


# ---- Free-text sites: CI YAML, Dockerfiles, Makefiles, scripts, docs ------

#: `pip install kglite==0.14.5`, `uv pip install 'kglite>=0.15'`,
#: `cargo install kglite-cli --version 0.15.0`, `kglite = "0.15"` in a doc, etc.
_INSTALL_RE = re.compile(
    r"""(?:pip\s+install|uv\s+pip\s+install|uv\s+add|pipx\s+install|poetry\s+add)
        [^\n]*?['"]?\b(?P<name>[A-Za-z0-9._-]+)(?:\[[^\]]*\])?
        (?P<spec>(?:[<>=!~]=?[0-9][0-9A-Za-z.*+-]*)(?:\s*,\s*[<>=!~]=?[0-9][0-9A-Za-z.*+-]*)*)""",
    re.VERBOSE,
)
_CARGO_INSTALL_RE = re.compile(
    r"cargo\s+install\s+(?:[^\n]*?\s)?(?P<name>[A-Za-z0-9._-]+)[^\n]*?--version\s+"
    r"['\"]?(?P<spec>[0-9][0-9A-Za-z.*+,<>=~^-]*)"
)
#: A bare `name==1.2.3` / `name>=1.2` token anywhere (matrix entries, env vars,
#: `KGLITE_VERSION: 0.15.0`, shell variables).
_BARE_PIN_RE = re.compile(
    r"(?<![A-Za-z0-9._-])(?P<name>[A-Za-z][A-Za-z0-9._-]{2,})"
    r"(?P<spec>(?:[<>=!~]=|==|>=|<=|[<>])\s*[0-9][0-9A-Za-z.*+-]*"
    r"(?:\s*,\s*(?:[<>=!~]=|==|>=|<=|[<>])\s*[0-9][0-9A-Za-z.*+-]*)*)"
)
#: `KGLITE_VERSION = "0.15.0"` / `kglite-version: 0.15.0`
_NAMED_VERSION_RE = re.compile(
    r"(?P<name>[A-Za-z][A-Za-z0-9._-]*)[_-]version\s*[:=]\s*['\"]?(?P<spec>[0-9][0-9A-Za-z.]*)",
    re.IGNORECASE,
)
#: Docs prose: "KGLite 0.14.3", "kglite v0.14.3", "requires kglite 0.15".
_DOC_VERSION_RE = re.compile(
    r"(?<![A-Za-z0-9._/-])(?P<name>[A-Za-z][A-Za-z0-9_-]{2,})[\s=]+v?(?P<spec>\d+\.\d+(?:\.\d+)?)"
    r"(?![0-9.]*[A-Za-z/])"
)


#: This file names example versions in its own docstring and tables; scanning
#: itself is pure self-reference.
SELF = Path(__file__).resolve()


def _site_for(path: Path) -> str | None:
    if path.resolve() == SELF:
        return None
    name = path.name
    parts = {p.lower() for p in path.parts}
    if ".github" in parts and path.suffix in {".yml", ".yaml"}:
        return "ci-yaml"
    if name.startswith("Dockerfile") or name.endswith(".dockerfile"):
        return "dockerfile"
    if name in {"Makefile", "makefile", "GNUmakefile"} or path.suffix == ".mk":
        return "makefile"
    if path.suffix in {".sh", ".bash", ".zsh"}:
        return "shell"
    if path.suffix == ".py" and "scripts" in parts:
        return "script"
    # An install instruction baked into a runtime error message is a real,
    # load-bearing declaration that no packaging tool will ever reconcile —
    # sonagram tells users `pip install kglite>=0.15` from Rust.
    if path.suffix in {".rs", ".py", ".pyi"}:
        return "source-string"
    if path.suffix in {".yml", ".yaml"} and "docker" in name.lower():
        return "compose"
    return None


def scan_freetext(repo: str, path: Path, tracked: set[str], site: str) -> list[Declaration]:
    lines = _read(path)
    if lines is None:
        return []
    out: list[Declaration] = []
    seen: set[tuple[int, str, str]] = set()

    def add(i: int, name: str, spec: str, raw: str) -> None:
        if not _tracked(name, tracked):
            return
        key = (i, name.lower(), spec)
        if key in seen:
            return
        seen.add(key)
        out.append(Declaration(repo, path, i, name, spec, site, raw.strip()[:200], metadata=False))

    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if not stripped or SUPPRESS_MARKER in stripped:
            continue
        for m in _INSTALL_RE.finditer(line):
            add(i, m.group("name"), m.group("spec"), stripped)
        # A Maven coordinate is an install instruction wherever it appears —
        # a workflow, a shell snippet, an error message — and it is the only
        # form the Java artifact's version is ever quoted in.
        for m in _MAVEN_COORD_RE.finditer(line):
            add(i, m.group("name"), "=" + m.group("spec"), stripped)
        if site == "source-string":
            # Source files are scanned *only* for embedded install instructions.
            # Anything broader turns every version-shaped literal in the tree
            # into a finding.
            continue
        for m in _CARGO_INSTALL_RE.finditer(line):
            add(i, m.group("name"), "=" + m.group("spec"), stripped)
        for m in _BARE_PIN_RE.finditer(line):
            add(i, m.group("name"), m.group("spec").replace(" ", ""), stripped)
        for m in _NAMED_VERSION_RE.finditer(line):
            add(i, m.group("name"), "=" + m.group("spec"), stripped)
    return out


#: Documents whose whole purpose is to name *old* versions. Flagging these
#: would be wrong, not merely noisy: a changelog entry, a migration guide, or a
#: recorded benchmark run is supposed to say 0.13.3 forever.
HISTORICAL_DOCS = re.compile(
    r"(CHANGELOG|BENCHMARKS|HISTORY|RELEASES)\.md$|migrat|/history/|\.kgl\.lock$",
    re.IGNORECASE,
)
#: Per-line opt-out for a deliberately pinned version in otherwise-live prose.
SUPPRESS_MARKER = "version-check: ignore"
#: A markdown table row that opens with a date is a ledger entry — sonagram's
#: GRAPH-GATE.md records "2026-07-17 | … sonara 0.2.3 sync" forever. Same class
#: as a changelog line, so it is history rather than a stale claim.
_LEDGER_ROW = re.compile(r"^\|\s*\d{4}-\d{2}-\d{2}\s*\|")


# ---- declaration vs citation ----------------------------------------------
#
# A version *declaration* — a pin, a documented floor, "requires >= X", an
# install line — states what you must use, so it goes stale the moment a
# release lands. A version *citation* — "introduced in X", "the X → Y move",
# "verified against X", a dated correction — states what happened, and is true
# forever. Telling a downstream to "move" one of those to the current version
# asks them to falsify their own record; codingest's PARITY.md provenance line
# was cited that way in three consecutive release notes.
#
# The classifier is deliberately conservative in one direction: anything it
# cannot recognise as a record stays a finding, because a missed stale pin
# costs more than a false positive that a human dismisses.

#: Past-tense / provenance cues. Presence of one inside the sentence carrying
#: the version means the sentence is talking about the past.
#: The past-tense forms are deliberate. An earlier draft accepted
#: ``ship(?:ped)? in`` and classified "these crates ship in lockstep" — a
#: present-tense statement about policy — as a historical record.
_CITATION_CUES = re.compile(
    r"\b(introduced|added\s+in|landed\s+in|shipped\s+in|released\s+in|"
    r"removed\s+in|deleted\s+in|renamed\s+in|fixed\s+in|broke\s+in|changed\s+in|"
    r"migrat(?:ed|ion|ing)|verified|was|were|used\s+to|since|as\s+of|onwards?|"
    r"previously|formerly|originally|earlier|historical\w*|history|no\s+longer|"
    r"already|first\s+appeared|described|correction|moved\s+from|"
    r"upgraded\s+from|bumped\s+from)\b",
    re.IGNORECASE,
)
#: Cues that make a sentence a live requirement even if it is phrased in prose.
#: These win over the citation cues, except against an explicit version range.
_DECLARATION_CUES = re.compile(
    r"\b(requires?|required|requirement|minimum|floor|pins?|pinned|depends?\s+on|"
    r"needs?|install(?:s|ing|ation)?|upgrade\s+to|bump\s+to|move\s+to|must\s+be|"
    r"at\s+least|or\s+newer|or\s+later|supported\s+version)\b",
    re.IGNORECASE,
)
#: ``0.15.8 → 0.15.11``, ``0.15.8 -> 0.15.11``, ``0.15.8 to 0.15.11``. Two
#: endpoints describe a completed transition; nothing can "move" a range to the
#: current version without rewriting which move actually happened. This wins
#: over the declaration cues, because "the engine floor moved 0.15.11 → 0.15.13"
#: is a record of a bump, not a bump instruction.
_VERSION_RANGE = re.compile(r"v?\d+\.\d+(?:\.\d+)?\s*(?:→|->|-->|=>|–|—|\bto\b)\s*v?\d+\.\d+")
#: An ISO date anywhere in the surrounding prose dates the statement.
_DATED_PROSE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
#: ``0.15.5+`` is a floor written compactly — a declaration, whatever else the
#: sentence says.
_FLOOR_SUFFIX = re.compile(r"^\s*\+")


def _prose_context(lines: list[str], index: int) -> tuple[str, int]:
    """The wrapped sentence around ``lines[index]``, and its start offset.

    Markdown wraps sentences across lines, so the verb that dates a statement
    ("…and it was\\nintroduced in kglite 0.15.5") routinely sits on a different
    line from the version. Neighbours are joined only while the sentence is
    unfinished — a preceding line that ends in terminal punctuation starts a new
    thought and is not evidence about this one.
    """
    before: list[str] = []
    i = index - 1
    while i >= 0 and len(before) < 2:
        prev = lines[i].strip()
        if not prev or prev.endswith((".", "!", "?")) or prev.startswith(("#", "|", "```")):
            break
        before.insert(0, prev)
        i -= 1
    after: list[str] = []
    tail = lines[index].strip()
    j = index + 1
    while j < len(lines) and len(after) < 2 and not tail.endswith((".", "!", "?")):
        nxt = lines[j].strip()
        if not nxt or nxt.startswith(("#", "|", "```")):
            break
        after.append(nxt)
        tail = nxt
        j += 1
    head = " ".join(before)
    line_start = len(head) + 1 if head else 0
    return " ".join([*before, lines[index].strip(), *after]), line_start


def _sentence_at(text: str, index: int) -> str:
    """The sentence of ``text`` containing offset ``index``."""
    starts = [m.end() for m in re.finditer(r"[.!?;:]\s+", text[:index])]
    start = starts[-1] if starts else 0
    m = re.search(r"[.!?;:]\s+", text[index:])
    end = index + m.start() + 1 if m else len(text)
    return text[start:end]


def is_version_citation(lines: list[str], index: int, match: re.Match[str]) -> bool:
    """True when this version mention records history rather than declaring a need."""
    if _FLOOR_SUFFIX.match(lines[index][match.end() :]):
        return False  # `0.15.5+` — a floor, not a record
    context, line_start = _prose_context(lines, index)
    lead_ws = len(lines[index]) - len(lines[index].lstrip())
    sentence = _sentence_at(context, line_start + max(0, match.start() - lead_ws))
    if _VERSION_RANGE.search(sentence):
        return True
    if _DECLARATION_CUES.search(sentence):
        return False
    return bool(_CITATION_CUES.search(sentence) or _DATED_PROSE.search(context))


# ---- Gradle / Maven coordinates -------------------------------------------

#: A Maven coordinate: ``io.github.kkollsga:kglite:0.15.9``. This is how the
#: Java artifact is *quoted* — in a Gradle `implementation(...)` line, a Maven
#: `<dependency>`, a README install block — and it is the one shape none of the
#: other scanners can see, because the separator is `:` rather than whitespace
#: or a comparison operator. A stale coordinate in a doc is the exact failure
#: this script exists to catch: the reader copies it and gets a version we no
#: longer ship.
_MAVEN_COORD_RE = re.compile(
    r"(?<![\w.-])(?P<group>[A-Za-z][A-Za-z0-9_.-]*)"
    r":(?P<name>[A-Za-z][A-Za-z0-9_.-]*)"
    r":(?P<spec>\d+\.\d+(?:\.\d+)?[0-9A-Za-z.+-]*)"
)

#: The Maven XML spelling of the same coordinate::
#:
#:     <artifactId>kglite</artifactId>
#:     <version>0.15.9</version>
#:
#: `_MAVEN_COORD_RE` cannot see it — there is no `:` anywhere — yet this is the
#: form a Java consumer is most likely to copy, because it is what a `pom.xml`
#: takes. A README carrying only the Gradle line would have exactly half its
#: install instructions watched, which is the shape where the *unwatched* half
#: is the one that goes stale unnoticed.
_MAVEN_XML_ARTIFACT_RE = re.compile(r"<artifactId>\s*(?P<name>[A-Za-z][\w.-]*)\s*</artifactId>")
_MAVEN_XML_VERSION_RE = re.compile(r"<version>\s*(?P<spec>\d+\.\d+(?:\.\d+)?[0-9A-Za-z.+-]*)\s*</version>")

#: A literal `version = "0.15.9"` in a Gradle build script.
#:
#: kglite-java deliberately has none: it reads `[workspace.package]` out of the
#: root Cargo.toml, so the Java artifact ships in lockstep with the crates and
#: the wheel by construction. A literal here would silently reintroduce the
#: second copy — and because the Java artifact is built from the same tag, the
#: drift would only surface as a wrong coordinate on Maven Central, after
#: publication, permanently.
_GRADLE_VERSION_RE = re.compile(r"""^\s*version\s*=\s*["'](?P<spec>\d+[^"']*)["']""")


def scan_gradle(repo: str, path: Path, tracked: set[str]) -> list[Declaration]:
    """Maven coordinates and own-version literals in a Gradle build script."""
    lines = _read(path)
    if lines is None:
        return []
    out: list[Declaration] = []
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("//") or SUPPRESS_MARKER in stripped:
            continue
        m = _GRADLE_VERSION_RE.match(line)
        if m:
            # Attributed to the Java artifact's own name, which is the Gradle
            # project name — `kglite`, same as the wheel and the engine crate.
            out.append(
                Declaration(
                    repo,
                    path,
                    i,
                    "kglite",
                    "=" + m.group("spec"),
                    "gradle-build",
                    stripped[:200],
                    metadata=True,
                )
            )
        for coord in _MAVEN_COORD_RE.finditer(line):
            name = coord.group("name")
            if _tracked(name, tracked):
                out.append(
                    Declaration(
                        repo,
                        path,
                        i,
                        name,
                        "=" + coord.group("spec"),
                        "gradle-build",
                        stripped[:200],
                        metadata=False,
                    )
                )
    return out


def scan_docs(repo: str, path: Path, tracked: set[str]) -> list[Declaration]:
    """Docs that state a concrete version in prose.

    Prose is the noisiest site, so this is deliberately conservative: it only
    matches ``<tracked-package> <x.y[.z]>`` with an optional ``v``, only for
    ecosystem packages (a doc saying ``pyo3 0.27`` is not our problem), and
    never inside a document that exists to record history.
    """
    if HISTORICAL_DOCS.search(str(path)):
        return []
    lines = _read(path)
    if lines is None:
        return []
    out: list[Declaration] = []
    seen: set[tuple[int, str]] = set()
    # A Maven `<dependency>` block splits the coordinate over three lines, so
    # which artifact a `<version>` belongs to is only knowable from a line
    # already passed. Carried forward rather than looked back for, because the
    # block is always written in that order and nothing else names an
    # artifactId.
    pending_artifact: str | None = None
    for i, line in enumerate(lines, 1):
        s = line.strip()
        if not s or s.startswith(("[", "|--", "---")) or SUPPRESS_MARKER in s:
            continue
        if _LEDGER_ROW.match(s):
            continue
        if m := _MAVEN_XML_ARTIFACT_RE.search(line):
            pending_artifact = m.group("name")
        elif m := _MAVEN_XML_VERSION_RE.search(line):
            name, pending_artifact = pending_artifact, None
            if name is not None and _tracked(name, tracked) and (i, name.lower()) not in seen:
                seen.add((i, name.lower()))
                out.append(
                    Declaration(repo, path, i, name, "=" + m.group("spec"), "docs-install", s[:200], metadata=False)
                )
        # An install line is the more specific reading of the same text, so it
        # wins; otherwise the two regexes double-report every `pip install X==Y`.
        for m in _INSTALL_RE.finditer(line):
            name = m.group("name")
            if _tracked(name, tracked) and (i, name.lower()) not in seen:
                seen.add((i, name.lower()))
                out.append(Declaration(repo, path, i, name, m.group("spec"), "docs-install", s[:200], metadata=False))
        for m in _MAVEN_COORD_RE.finditer(line):
            name = m.group("name")
            if _tracked(name, tracked) and (i, name.lower()) not in seen:
                seen.add((i, name.lower()))
                out.append(
                    Declaration(repo, path, i, name, "=" + m.group("spec"), "docs-install", s[:200], metadata=False)
                )
        for m in _DOC_VERSION_RE.finditer(line):
            name = m.group("name")
            if not _tracked(name, tracked) or (i, name.lower()) in seen:
                continue
            seen.add((i, name.lower()))
            # Prose is the only site where a version can be a *record* rather
            # than a requirement, so it is the only site that gets classified.
            # An install line is a declaration wherever it appears.
            site = "docs-citation" if is_version_citation(lines, i - 1, m) else "docs"
            out.append(Declaration(repo, path, i, name, "=" + m.group("spec"), site, s[:200], metadata=False))
    return out


# --------------------------------------------------------------------------
# Repo walk
# --------------------------------------------------------------------------


def assign_units(decls: list[Declaration], root: Path, lock_dirs: set[Path]) -> None:
    """Tag each declaration with the resolution unit that will resolve it.

    Two declarations only *contradict* if something resolves them together. A
    workspace and a standalone example with its own ``Cargo.lock`` are separate
    resolution units — the example pinning an older published version of its own
    parent crate is stale, not contradictory. The unit is the nearest ancestor
    directory carrying a lockfile, falling back to the repo root.
    """
    for d in decls:
        unit = root
        for parent in d.path.parents:
            if parent in lock_dirs:
                unit = parent
                break
            if parent == root:
                break
        d.unit = unit


def scan_repo(repo: str, root: Path, tracked: set[str], include_docs: bool) -> list[Declaration]:
    decls: list[Declaration] = []
    lock_dirs: set[Path] = set()
    for path in _iter_files(root):
        name = path.name
        if name in {"Cargo.lock", "uv.lock", "poetry.lock"}:
            lock_dirs.add(path.parent)
        if name == "Cargo.toml":
            decls += scan_cargo_manifest(repo, path, tracked)
        elif name == "Cargo.lock":
            decls += scan_cargo_lock(repo, path, tracked)
        elif name == "pyproject.toml":
            decls += scan_pyproject(repo, path, tracked)
        elif path.suffix in {".gradle", ".kts"}:
            decls += scan_gradle(repo, path, tracked)
        elif re.match(r"^(requirements|constraints).*\.txt$", name):
            decls += scan_requirements(repo, path, tracked)
        elif name == "uv.lock":
            decls += scan_cargo_lock(repo, path, tracked)  # same name/version shape
        elif path.suffix in {".md", ".rst"} and include_docs:
            decls += scan_docs(repo, path, tracked)
        else:
            site = _site_for(path)
            if site:
                decls += scan_freetext(repo, path, tracked, site)
    assign_units(decls, root, lock_dirs)
    return decls


def repo_own_version(root: Path) -> tuple[int, int, int] | None:
    """The repo's own declared version (workspace Cargo.toml, then pyproject)."""
    for rel in ("Cargo.toml", "pyproject.toml"):
        lines = _read(root / rel)
        if not lines:
            continue
        table = ""
        for line in lines:
            m = _CARGO_TABLE_RE.match(line)
            if m:
                table = m.group(1).strip()
                continue
            if table in {"workspace.package", "package", "project", "tool.poetry"}:
                vm = re.match(r'^\s*version\s*=\s*"([^"]+)"', line)
                if vm:
                    parsed = parse_version(vm.group(1))
                    if parsed:
                        return parsed
    return None


# --------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------


def _nearest_lock(d: Declaration, locks: list[Declaration]) -> Declaration | None:
    """The lock entry that actually governs ``d``, by resolution unit.

    Matching a manifest against *any* lockfile in the repo is wrong: mcp-methods
    carries a standalone ``examples/downstream_binary`` with its own lock pinned
    a major behind the workspace, and pairing that manifest with the workspace
    lock produces a nonsense verdict in both directions.
    """
    same_unit = [x for x in locks if x.unit == d.unit]
    if not same_unit:
        return None
    return max(same_unit, key=lambda x: x.interval.lo)  # type: ignore[union-attr]


def analyse(
    decls: list[Declaration],
    package_owner: dict[str, tuple[str, tuple[int, int, int]]],
) -> list[Finding]:
    """Turn raw declarations into findings.

    ``package_owner`` maps each ecosystem package to ``(repo, current_version)``
    so staleness is checked against the version its own repo currently declares
    — not only against KGLite. That generalisation is what catches mcp-methods'
    own stale example fixture as well as a downstream stuck below the engine.
    """
    findings: list[Finding] = []
    by_repo_pkg: dict[tuple[str, str], list[Declaration]] = {}
    for d in decls:
        by_repo_pkg.setdefault((d.repo, d.package.lower().replace("_", "-")), []).append(d)

    for (repo, pkg), group in sorted(by_repo_pkg.items()):
        constrained = [d for d in group if d.interval is not None]
        locks = [d for d in constrained if d.site == "cargo-lock"]
        binding = [d for d in constrained if d.binding]
        manifests = [d for d in binding if d.metadata]

        owner = package_owner.get(pkg)
        current = owner[1] if owner else None

        # A repo that tests both ends of its own declared range (kglite-datasets
        # runs a `kglite==0.13.0` CI matrix leg against a `>=0.13` floor) is
        # doing the right thing. An exact pin equal to the metadata floor is
        # that pattern, not rot.
        floor_pins: set[int] = set()
        metadata_floors = {d.interval.lo for d in manifests}  # type: ignore[union-attr]
        for d in binding:
            if _is_exact(d.spec) and d.interval.lo in metadata_floors:  # type: ignore[union-attr]
                floor_pins.add(id(d))

        # --- 1. contradictions, scoped to a resolution unit
        for unit in {d.unit for d in binding}:
            requirements = [d for d in binding if d.unit == unit]
            if len(requirements) < 2:
                continue
            acc = FULL
            for d in requirements:
                acc = acc.intersect(d.interval)  # type: ignore[arg-type]
            if acc:
                continue
            pair = _minimal_conflict(requirements) or requirements
            findings.append(
                Finding(
                    kind="contradiction",
                    severity="error",
                    repo=repo,
                    package=pkg,
                    message=(
                        f"{repo} declares mutually exclusive requirements for {pkg!r} "
                        f"within one resolution unit: no single version satisfies all "
                        f"{len(requirements)} binding site(s). This is a latent build break "
                        f"— whichever site the installer honours, another is violated."
                    ),
                    declarations=pair,
                )
            )

        # --- 2. understated floor vs. the lock that actually governs it
        known_floor = parse_version(WATCHED_THIRD_PARTY[pkg]) if WATCHED_THIRD_PARTY.get(pkg) else None
        for d in manifests:
            iv = d.interval
            assert iv is not None
            if known_floor and iv.lo < known_floor:
                findings.append(
                    Finding(
                        kind="understated-floor",
                        severity="warn",
                        repo=repo,
                        package=pkg,
                        message=(
                            f"{repo} declares {pkg} {d.spec!r} (floor {fmt_version(iv.lo)}) but "
                            f"the feature this ecosystem selects first exists in "
                            f"{fmt_version(known_floor)}. Raise the floor — the lockfile hides "
                            f"this from every build in this repo."
                        ),
                        declarations=[d],
                    )
                )
                continue  # the explicit floor is the better message; don't double-report
            lock = _nearest_lock(d, locks)
            if lock is None:
                continue
            lock_v = lock.interval.lo  # type: ignore[union-attr]
            if iv.lo < lock_v and iv.contains(lock_v):
                findings.append(
                    Finding(
                        kind="understated-floor",
                        severity="warn",
                        repo=repo,
                        package=pkg,
                        message=(
                            f"{repo} declares {pkg} {d.spec!r} (floor {fmt_version(iv.lo)}) but "
                            f"the governing lockfile resolved {fmt_version(lock_v)}. Every build "
                            f"here resolves fine; a consumer resolving fresh gets "
                            f"{fmt_version(iv.lo)} and may not compile. If the floor is real, "
                            f"say so; if it is not, this is invisible until someone else breaks."
                        ),
                        declarations=[d, lock],
                    )
                )

        # --- 3. staleness against the package's own current version
        if current is not None:
            for d in constrained:
                if d.site == "cargo-lock":
                    continue
                iv = d.interval
                assert iv is not None
                if d.site in CITATION_SITES:
                    # Historical prose. Surfaced so the classification is
                    # auditable — a silent exemption is how a scanner goes
                    # blind — but never as something to change.
                    if not iv.contains(current):
                        findings.append(
                            Finding(
                                kind="citation",
                                severity="info",
                                repo=repo,
                                package=pkg,
                                message=(
                                    f"{repo} prose records {pkg} {fmt_version(iv.lo)} as history "
                                    f"(a migration, an introduction, a verified move), not as a "
                                    f"requirement. Moving it to {fmt_version(current)} would "
                                    f"falsify the record."
                                ),
                                declarations=[d],
                            )
                        )
                    continue
                is_doc = d.site in DOC_SITES
                if not iv.contains(current):
                    if id(d) in floor_pins:
                        findings.append(
                            Finding(
                                kind="floor-test",
                                severity="info",
                                repo=repo,
                                package=pkg,
                                message=(
                                    f"{repo} pins {pkg} at {fmt_version(iv.lo)}, which equals its "
                                    f"own declared floor — a deliberate floor-test leg, not rot."
                                ),
                                declarations=[d],
                            )
                        )
                    elif is_doc:
                        # An upper-bound-only doc constraint (`pip install
                        # "kglite<0.14"`) is a deliberate pin-back instruction,
                        # not a stale statement of the current version.
                        if iv.lo == (0, 0, 0):
                            continue
                        if _acknowledged(d):
                            findings.append(
                                Finding(
                                    kind="acknowledged",
                                    severity="info",
                                    repo=repo,
                                    package=pkg,
                                    message=f"documented {fmt_version(iv.lo)} — {_acknowledged(d)}",
                                    declarations=[d],
                                )
                            )
                            continue
                        findings.append(
                            Finding(
                                kind="stale-docs",
                                severity="warn",
                                repo=repo,
                                package=pkg,
                                message=(
                                    f"{repo} documentation states {pkg} "
                                    f"{fmt_version(iv.lo)}; current is {fmt_version(current)}. "
                                    f"No gate watches prose, so this drifts silently."
                                ),
                                declarations=[d],
                            )
                        )
                    else:
                        findings.append(
                            Finding(
                                kind="stale-downstream",
                                severity="error",
                                repo=repo,
                                package=pkg,
                                message=(
                                    f"{repo} declares {pkg} {d.spec!r}, which EXCLUDES the current "
                                    f"{fmt_version(current)} (admits "
                                    f"{fmt_version(iv.lo)}..{fmt_version(iv.hi)}). This repo "
                                    f"cannot install the current {pkg}."
                                ),
                                declarations=[d],
                            )
                        )
                elif _is_exact(d.spec) and iv.lo < current and not is_doc:
                    findings.append(
                        Finding(
                            kind="exact-pin",
                            severity="warn",
                            repo=repo,
                            package=pkg,
                            message=(
                                f"{repo} pins {pkg} at exactly {fmt_version(iv.lo)}, superseded "
                                f"by {fmt_version(current)}."
                            ),
                            declarations=[d],
                        )
                    )

        # --- 4. inventory of declaration sites outside package metadata
        for d in group:
            if d.site in CITATION_SITES:
                continue  # reported as `citation`; not a drift surface
            if not d.metadata and d.site != "cargo-lock":
                findings.append(
                    Finding(
                        kind="site",
                        severity="info",
                        repo=repo,
                        package=pkg,
                        message=f"{d.site}: {pkg} {d.spec or '(no version)'}",
                        declarations=[d],
                    )
                )

    return findings


#: Documented versions that are *supposed* to name an older release, with the
#: reason. Prose cannot be classified mechanically — "use kglite 0.13.4 to
#: convert pre-0.14 artifacts" is correct forever, while "a thin KGLite 0.14.3
#: frontend" is rot, and no regex separates them. So the checker surfaces every
#: candidate and this table records the adjudicated ones, in the same spirit as
#: `scripts/check_lint_allowances.py`. Key: ``(path suffix, package)``.
#: A one-off can instead carry the inline `version-check: ignore` marker.
_BRIDGE = "0.13.4 is the documented pre-0.14 artifact-conversion bridge"
ACKNOWLEDGED: dict[tuple[str, str, str], str] = {
    ("KGLite", "README.md", "kglite"): _BRIDGE,
    ("KGLite", "docs/index.md", "kglite"): _BRIDGE,
    ("KGLite", "docs/python/value-projection.md", "kglite"): _BRIDGE,
    ("KGLite", "docs/python/guides/semantic-search.md", "kglite"): _BRIDGE,
    ("KGLite", "docs/python/guides/import-export.md", "kglite"): _BRIDGE,
    ("KGLite", "docs/rust/c-abi.md", "kglite"): _BRIDGE,
    ("KGLite", "docs/rust/api-reference.md", "kglite"): _BRIDGE,
    (
        "KGLite",
        "docs/rust/postcard-persistence-performance.md",
        "kglite",
    ): "a recorded benchmark A/B; the reference wheel is pinned by design",
    (
        "codingest",
        "docs/python-api.md",
        "kglite",
    ): "states which kglite release removed the Python builder — history, not a pin",
    (
        "codingest",
        "docs/mcp-parity.md",
        "kglite",
    ): "names the release that introduced WorkspaceGraphHooks/ServerExtensions — "
    "history, and it matches codingest's own 0.15.0 pin",
}


def _acknowledged(d: Declaration) -> str | None:
    """Reason this documented version is deliberate, or None.

    Keyed by repo as well as path: several repos have a ``README.md`` and a
    ``docs/index.md``, and an unscoped suffix match silently exonerated
    sonagram's genuinely stale strings.
    """
    posix = d.path.as_posix()
    pkg = d.package.lower().replace("_", "-")
    for (repo, suffix, want_pkg), reason in ACKNOWLEDGED.items():
        if repo == d.repo and want_pkg == pkg and posix.endswith(suffix):
            return reason
    return None


def _is_exact(spec: str) -> bool:
    return bool(re.match(r"^\s*(==|=)\s*\d+\.\d+\.\d+\s*$", spec))


def _minimal_conflict(decls: list[Declaration]) -> list[Declaration] | None:
    """Smallest pair of declarations that already contradict, for reporting."""
    for i, a in enumerate(decls):
        for b in decls[i + 1 :]:
            if not a.interval.intersect(b.interval):  # type: ignore[union-attr]
                return [a, b]
    return None


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

SEV_ORDER = {"error": 0, "warn": 1, "info": 2}
KIND_TITLE = {
    "contradiction": "CONTRADICTIONS — incompatible versions inside one resolution unit",
    "stale-downstream": "CROSS-REPO STALENESS — declared range excludes the current version",
    "understated-floor": "UNDERSTATED FLOORS — declared minimum below what actually builds",
    "exact-pin": "SUPERSEDED EXACT PINS",
    "stale-docs": "STALE DOCUMENTED VERSIONS — prose naming a superseded version",
    "floor-test": "DELIBERATE FLOOR-TEST PINS (not findings; shown for audit)",
    "citation": "HISTORICAL CITATIONS — prose recording a past version (classified, not flagged)",
    "acknowledged": "ACKNOWLEDGED HISTORICAL REFERENCES (adjudicated, no action)",
    "site": "DECLARATION SITES OUTSIDE PACKAGE METADATA (drift surface)",
}
KIND_ORDER = [
    "contradiction",
    "stale-downstream",
    "understated-floor",
    "exact-pin",
    "stale-docs",
    "floor-test",
    "citation",
    "acknowledged",
    "site",
]


def report(findings: list[Finding], upstream_version, show_sites: bool) -> None:
    print(f"ecosystem root: {ECOSYSTEM_ROOT}")
    print(f"upstream: {UPSTREAM_REPO} {fmt_version(upstream_version)}\n")
    for kind in KIND_ORDER:
        group = [f for f in findings if f.kind == kind]
        if not group:
            continue
        if kind == "site" and not show_sites:
            print(f"{KIND_TITLE[kind]}: {len(group)} (use --show-sites to list)\n")
            continue
        print(f"== {KIND_TITLE[kind]} ({len(group)}) ==")
        if kind == "site":
            for f in sorted(group, key=lambda f: (f.repo, f.declarations[0].site, f.package)):
                d = f.declarations[0]
                print(f"  [{d.site:<20}] {d.rel}:{d.line}  {d.package} {d.spec or '-'}")
        else:
            for f in sorted(group, key=lambda f: (f.repo, f.package)):
                print(f"  [{f.severity.upper()}] {f.repo} :: {f.package}")
                print(f"      {f.message}")
                for d in f.declarations:
                    print(f"      - {d.rel}:{d.line}  {d.raw}")
        print()


# --------------------------------------------------------------------------
# Part 2 — the downstream notifier
# --------------------------------------------------------------------------


@dataclasses.dataclass
class NotifyDecision:
    repo: str
    notify: bool
    reasons: list[str]
    skip_reason: str = ""
    blocked: bool = False
    findings: list[Finding] = dataclasses.field(default_factory=list)
    touched_symbols: list[tuple[str, str]] = dataclasses.field(default_factory=list)


#: Files a symbol reference can live in at all. Markdown, RST and text are
#: absent on purpose: prose *naming* an API is not a call site.
CODE_EXTS = {".rs", ".py", ".pyi", ".toml", ".java", ".kt", ".kts", ".c", ".h", ".cpp", ".hpp", ".go", ".js", ".ts"}

#: Comment syntax per extension: (line marker, has C-style block comments,
#: has Python triple-quoted strings).
_COMMENT_STYLE: dict[str, tuple[str, bool, bool]] = {
    ".rs": ("//", True, False),
    ".java": ("//", True, False),
    ".kt": ("//", True, False),
    ".kts": ("//", True, False),
    ".c": ("//", True, False),
    ".h": ("//", True, False),
    ".cpp": ("//", True, False),
    ".hpp": ("//", True, False),
    ".go": ("//", True, False),
    ".js": ("//", True, False),
    ".ts": ("//", True, False),
    ".toml": ("#", False, False),
    ".py": ("#", False, True),
    ".pyi": ("#", False, True),
}


def _cut_at_marker(text: str, marker: str) -> str:
    """Drop everything from an unquoted ``marker`` onwards."""
    out: list[str] = []
    quote: str | None = None
    i = 0
    while i < len(text):
        ch = text[i]
        if quote:
            if ch == "\\":
                out.append(text[i : i + 2])
                i += 2
                continue
            if ch == quote:
                quote = None
        elif ch in "\"'":
            quote = ch
        elif text.startswith(marker, i):
            break
        out.append(ch)
        i += 1
    return "".join(out)


def code_lines(suffix: str, lines: list[str]) -> list[tuple[int, str]]:
    """``(lineno, code)`` with comments and Python docstrings removed.

    A symbol named in a comment is a *note about* an API, not a use of it — and
    conflating the two produced the worst kind of evidence: codingest's release
    note cited a `Cargo.toml` comment describing a finished migration while
    missing the actual `NodeView` call site in `rev.rs`. The one hit offered was
    the one that wasn't a reference.
    """
    marker, block_comments, docstrings = _COMMENT_STYLE.get(suffix, ("#", False, False))
    out: list[tuple[int, str]] = []
    in_block = False
    in_doc: str | None = None
    for i, raw in enumerate(lines, 1):
        text = raw
        if in_block:
            end = text.find("*/")
            if end < 0:
                continue
            text = text[end + 2 :]
            in_block = False
        if in_doc:
            end = text.find(in_doc)
            if end < 0:
                continue
            text = text[end + 3 :]
            in_doc = None
        if block_comments:
            while True:
                start = text.find("/*")
                if start < 0:
                    break
                end = text.find("*/", start + 2)
                if end < 0:
                    text = text[:start]
                    in_block = True
                    break
                text = text[:start] + text[end + 2 :]
        if docstrings:
            for q in ('"""', "'''"):
                while True:
                    start = text.find(q)
                    if start < 0:
                        break
                    end = text.find(q, start + 3)
                    if end < 0:
                        text = text[:start]
                        in_doc = q
                        break
                    text = text[:start] + text[end + 3 :]
                if in_doc:
                    break
        text = _cut_at_marker(text, marker)
        if text.strip():
            out.append((i, text))
    return out


def find_symbol_uses(root: Path, symbols: list[str]) -> list[tuple[str, str]]:
    """Which of ``symbols`` this repo's *code* actually references.

    Approximates "a breaking change touches a surface it uses". Deliberately
    code-only: docs, changelogs and comments mentioning a symbol are not usage,
    and citing one as evidence sends a maintainer to a line that needs nothing.
    The hit is reported as ``path:line`` so the claim can be checked.
    """
    hits: list[tuple[str, str]] = []
    if not symbols:
        return hits
    found: dict[str, str] = {}
    for path in sorted(_iter_files(root)):
        if path.suffix not in CODE_EXTS or path.name in {"Cargo.lock"}:
            continue
        lines = _read(path)
        if lines is None:
            continue
        remaining = [s for s in symbols if s not in found]
        if not remaining:
            break
        try:
            rel = str(path.relative_to(root))
        except ValueError:
            rel = str(path)
        for lineno, code in code_lines(path.suffix, lines):
            for sym in remaining:
                if sym not in found and sym in code:
                    found[sym] = f"{rel}:{lineno}"
    for sym in symbols:
        if sym in found:
            hits.append((sym, found[sym]))
    return hits


def decide_notifications(
    findings: list[Finding],
    declarations: list[Declaration],
    repos: dict[str, Path],
    upstream_version: tuple[int, int, int],
    breaking_symbols: list[str],
) -> list[NotifyDecision]:
    decisions: list[NotifyDecision] = []
    for repo in DOWNSTREAM_REPOS:
        root = repos.get(repo)
        if root is None:
            decisions.append(NotifyDecision(repo, False, [], skip_reason="repo not present on disk"))
            continue

        # Only findings *about an upstream package* are this release's business.
        # mcp-methods' own stale example fixture is a real finding, but it is not
        # caused by a kglite release and must not ride a kglite release note.
        upstream_pkgs = set(ECOSYSTEM_PACKAGES[UPSTREAM_REPO])
        mine = [f for f in findings if f.repo == repo and f.package in upstream_pkgs]
        blocked = [f for f in mine if f.kind == "stale-downstream"]
        pins = [f for f in mine if f.kind == "exact-pin"]
        contradictions = [f for f in mine if f.kind == "contradiction"]
        stale_docs = [f for f in mine if f.kind == "stale-docs"]
        touched = find_symbol_uses(root, breaking_symbols)

        reasons: list[str] = []
        relevant: list[Finding] = []
        if blocked:
            reasons.append(
                f"declared range excludes {fmt_version(upstream_version)} "
                f"({len(blocked)} site{'s' if len(blocked) != 1 else ''}) — BLOCKED"
            )
            relevant += blocked
        if contradictions:
            reasons.append(f"{len(contradictions)} internal version contradiction(s)")
            relevant += contradictions
        if pins:
            reasons.append(
                f"pins upstream at a superseded exact version ({len(pins)} site{'s' if len(pins) != 1 else ''})"
            )
            relevant += pins
        if stale_docs:
            files = sorted({f.declarations[0].path.name for f in stale_docs})
            reasons.append(
                f"{len(stale_docs)} documented upstream version(s) now superseded, across {', '.join(files)}"
            )
            relevant += stale_docs
        if touched:
            reasons.append(
                "references "
                + ", ".join(f"`{s}`" for s, _ in touched)
                + " — touched by a breaking change in this release"
            )

        if reasons:
            decisions.append(
                NotifyDecision(repo, True, reasons, blocked=bool(blocked), findings=relevant, touched_symbols=touched)
            )
            continue

        # Why a repo is unaffected is worth stating precisely: "declares nothing
        # from us" and "declares us correctly" are different facts, and only the
        # second one means the next release needs re-checking.
        declares = any(d.package.lower().replace("_", "-") in upstream_pkgs and d.repo == repo for d in declarations)
        if not declares:
            skip = f"declares no {UPSTREAM_REPO} package anywhere — not a consumer"
        else:
            skip = (
                f"declares {fmt_version(upstream_version)} correctly everywhere: range "
                f"admits it, no superseded pin, no stale documented version, and no "
                f"reference to this release's breaking surface"
            )
        decisions.append(NotifyDecision(repo, False, [], skip_reason=skip))
    return decisions


# ---- "what changed" — read from the announced release's CHANGELOG entry -----

_CHANGELOG_RELEASE_RE = re.compile(r"^##\s*\[(?P<ver>[^\]]+)\]")
_CHANGELOG_SUBSECTION_RE = re.compile(r"^###\s+(?P<name>.+?)\s*$")
_CHANGELOG_BULLET_RE = re.compile(r"^-\s+(?P<text>.*)$")
_BOLD_LEAD_RE = re.compile(r"^\*\*(?P<lead>.+?)\*\*", re.DOTALL)

#: How many entries a note quotes before pointing at the file. A release with
#: twenty bullets does not become twenty asks.
MAX_HIGHLIGHTS = 8


def changelog_entry(path: Path, version: tuple[int, int, int]) -> list[str] | None:
    """The body lines of ``## [<version>]``, or None when there is no such entry.

    None is a meaningful answer and the caller must honour it: a note that
    cannot resolve its own release's entry carries no "what changed" section at
    all. The alternative — falling back to whatever text is nearest to hand —
    is the defect this function replaces.
    """
    lines = _read(path)
    if lines is None:
        return None
    body: list[str] | None = None
    for line in lines:
        m = _CHANGELOG_RELEASE_RE.match(line)
        if m:
            if body is not None:
                break  # next release heading ends the entry
            if parse_version(m.group("ver")) == version:
                body = []
            continue
        if body is not None:
            body.append(line)
    return body


def summarise_changelog_entry(body: list[str], version: str) -> list[str]:
    """Condense a changelog entry into one line per top-level bullet.

    Each entry in this project's changelog opens with a bold lead sentence that
    states the change; the paragraphs after it are evidence and measurement.
    Quoting the lead verbatim keeps the note honest — it is the changelog's own
    words about the release it names — while staying short enough to read in an
    inbox. A bullet without a bold lead falls back to its first sentence.
    """
    picked: list[tuple[str, str]] = []
    section = ""
    current: list[str] | None = None

    def flush() -> None:
        nonlocal current
        if current:
            text = " ".join(part.strip() for part in current).strip()
            lead = _lead_sentence(text)
            if lead:
                picked.append((section, lead))
        current = None

    for raw in body:
        sub = _CHANGELOG_SUBSECTION_RE.match(raw)
        if sub:
            flush()
            section = sub.group("name").strip()
            continue
        bullet = _CHANGELOG_BULLET_RE.match(raw)
        if bullet and not raw[:1].isspace():
            flush()
            current = [bullet.group("text")]
        elif current is not None:
            if not raw.strip():
                flush()
            else:
                current.append(raw)
    flush()

    out = [f"{sec}: {lead}" if sec else lead for sec, lead in picked[:MAX_HIGHLIGHTS]]
    if len(picked) > MAX_HIGHLIGHTS:
        out.append(f"…and {len(picked) - MAX_HIGHLIGHTS} further entr(ies) in `[{version}]` — read the CHANGELOG.")
    return out


def _lead_sentence(text: str) -> str:
    m = _BOLD_LEAD_RE.match(text)
    if m:
        return " ".join(m.group("lead").split())
    plain = " ".join(text.split())
    first = re.split(r"(?<=[.!?])\s", plain, maxsplit=1)[0]
    return first if len(first) <= 240 else first[:239].rstrip() + "…"


def release_highlights(upstream_root: Path, upstream_version: tuple[int, int, int]) -> list[str]:
    """What the announced release changed, condensed from its own entry.

    Empty when the entry cannot be resolved — and that emptiness is the whole
    point. Notes for 0.15.11, 0.15.12 and 0.15.13 all shipped 0.15.9's prose
    because this text came from a constant that no release step had to touch;
    an absent section is strictly better than a confidently wrong one.
    """
    ver = fmt_version(upstream_version)
    path = upstream_root / "CHANGELOG.md"
    body = changelog_entry(path, upstream_version)
    if body is None:
        print(
            f"warning: no `## [{ver}]` section in {path} — the release notes will carry "
            f'no "what changed" section rather than another release\'s.',
            file=sys.stderr,
        )
        return []
    highlights = summarise_changelog_entry(body, ver)
    if not highlights:
        print(
            f"warning: the `## [{ver}]` entry in {path} has no bullet entries to quote; "
            f'the release notes will carry no "what changed" section.',
            file=sys.stderr,
        )
    return highlights


def compose_note(
    decision: NotifyDecision,
    upstream_version: tuple[int, int, int],
    date: str,
    highlights: list[str],
) -> tuple[str, str]:
    """Return ``(filename, body)`` for a downstream release note.

    Follows the established inbox schema exactly (see ``.claude/skills/notify``
    and the existing notes in every sibling's ``inbox/read/``): H1 title, a
    From/To/Date/Type/Re metadata block, 1–3 paragraphs of context, then
    ``## Ask / action requested`` and ``## References``. The recipient appends
    their own ``## Status`` footer; we never write one.
    """
    ver = fmt_version(upstream_version)
    kinds = {f.kind for f in decision.findings}
    docs_only = kinds <= {"stale-docs"} and not decision.touched_symbols

    if decision.blocked:
        slug, note_type = "blocked-upgrade", "coordination"
        title = f"kglite {ver} published — your declared range excludes it"
        opening = (
            f"kglite {ver} is published. This note is going to {decision.repo} and not "
            f"to the rest of the ecosystem because {decision.repo} **cannot resolve it "
            f"as declared** — a version requirement here excludes {ver}, so the upgrade "
            f"fails before any code compiles."
        )
    elif docs_only:
        slug, note_type = "documented-version-drift", "heads-up"
        title = f"kglite {ver} published — your public docs still say an older version"
        opening = (
            f"kglite {ver} is published. Your package metadata already admits it, so "
            f"nothing is blocked. This note is going to {decision.repo} and not to the "
            f"rest of the ecosystem for one narrow reason: {decision.repo} states a "
            f"superseded kglite version in published prose, where no packaging tool, "
            f"lockfile, or CI job will ever notice the drift."
        )
    else:
        slug, note_type = "upgrade-required", "heads-up"
        title = f"kglite {ver} published — an upgrade step is waiting on you"
        opening = (
            f"kglite {ver} is published. This note is going to {decision.repo} and not "
            f"to the rest of the ecosystem because {decision.repo} carries at least one "
            f"declaration that this release supersedes or invalidates."
        )

    filename = f"{date}-from-kglite-{ver.replace('.', '-')}-{slug}.md"

    lines = [
        f"# {title}",
        "",
        "- **From:** kglite",
        f"- **To:** {decision.repo}",
        f"- **Date:** {date}",
        f"- **Type:** {note_type}",
        f"- **Re:** kglite {ver} (crates.io + PyPI)",
        "",
        opening,
        "",
        "**What is affected:**",
        "",
    ]
    lines += [f"- {reason}" for reason in decision.reasons]
    lines.append("")

    # Quoted source lines routinely contain backticks and pipes, so they go in a
    # fenced block rather than inline code — an inline span breaks on the first
    # backtick and silently mangles the evidence.
    sites: list[str] = []
    for f in decision.findings:
        for d in f.declarations:
            if d.site == "cargo-lock":
                continue
            entry = f"{d.rel}:{d.line}\n    {d.raw}"
            if entry not in sites:
                sites.append(entry)
    if sites:
        lines += ["**The exact sites:**", "", "```"]
        lines += sites
        lines += ["```", ""]

    if decision.touched_symbols:
        lines += ["**Breaking-change surface you actually reference:**", ""]
        lines += [f"- `{sym}` — in `{where}`" for sym, where in decision.touched_symbols]
        lines.append("")

    if highlights and not docs_only:
        lines += [
            f"**What changed in this release that can reach you** (lead lines of the `[{ver}]` CHANGELOG entry):",
            "",
        ]
        lines += [f"- {h}" for h in highlights]
        lines.append("")

    lines += ["## Ask / action requested", ""]
    if decision.blocked:
        lines.append(
            f"- Widen or bump every requirement above so it admits {ver}, refresh the "
            f"lockfile, and run your gate. Until that lands this repo is held on a "
            f"superseded engine."
        )
    elif docs_only:
        lines.append(
            f"- Update those strings to {ver} (or drop the version from the prose so it "
            f"stops needing maintenance). Nothing else here needs to change — your "
            f"manifests are already correct."
        )
    elif sites:
        lines.append(
            f"- Move the sites above to {ver} and refresh the lockfile. They sit outside "
            f"package metadata, so nothing in your own CI will notice they drifted."
        )
    # A note whose only reason is a referenced breaking symbol has no sites to
    # move; "move the sites above" pointed at nothing at all in that case.
    if decision.touched_symbols:
        lines.append(
            f"- Check those call sites against the `[{ver}]` CHANGELOG before upgrading; "
            f"they may need an edit, not just a version bump."
        )
    lines += [
        "",
        "## References",
        "",
        f"- kglite release: `{ver}` (crates.io + PyPI, tag `v{ver}`)",
        f"- CHANGELOG: `../KGLite/CHANGELOG.md` → `[{ver}]`",
        "- Detected by `../KGLite/scripts/check_version_consistency.py` "
        "(`make check-ecosystem-versions`); this note was generated by its "
        "`--notify` mode, which writes only to affected repos.",
        "",
    ]
    return filename, "\n".join(lines)


def run_notify(
    decisions: list[NotifyDecision],
    repos: dict[str, Path],
    upstream_version: tuple[int, int, int],
    date: str,
    highlights: list[str],
    dry_run: bool,
) -> int:
    ver = fmt_version(upstream_version)
    print(f"== NOTIFIER — kglite {ver} ==")
    print(f"{'DRY RUN — nothing will be written' if dry_run else 'writing to inbox/unread/'}\n")
    written = 0
    for d in sorted(decisions, key=lambda x: (not x.notify, x.repo)):
        if not d.notify:
            print(f"  SKIP   {d.repo:<16} — {d.skip_reason}")
            continue
        root = repos[d.repo]
        filename, body = compose_note(d, upstream_version, date, highlights)
        target = root / "inbox" / "unread" / filename
        print(f"  NOTIFY {d.repo:<16} -> {target}")
        for reason in d.reasons:
            print(f"           · {reason}")
        if dry_run:
            print()
            print("           ---8<--- note body ---8<---")
            for line in body.splitlines():
                print(f"           {line}")
            print("           ---8<--- end ---8<---\n")
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(body, encoding="utf-8")
            written += 1
    notified = sum(1 for d in decisions if d.notify)
    skipped = len(decisions) - notified
    print(f"\n{notified} downstream(s) affected, {skipped} deliberately not notified.")
    if not dry_run:
        print(f"{written} note(s) written. inbox/ is gitignored working state — never commit them.")
    return 0


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

#: Symbols whose *breaking* change could reach a downstream, keyed by **the
#: release that broke them**.
#:
#: The key is load-bearing, and it replaces a flat ``DEFAULT_BREAKING_SYMBOLS``
#: list that was the 0.15.9 set and stayed the 0.15.9 set: every release after
#: it asserted, in a downstream's inbox, that this release touched `NodeView`.
#: Three notes shipped that claim about releases containing no such change. A
#: symbol set that is not tied to the version it describes cannot go stale
#: *visibly* — it just keeps being right about one release and wrong about all
#: the others. Keyed, a release with no entry here simply does not fire the
#: breaking-surface criterion, which is the correct behaviour for the common
#: case (most releases break nothing).
#:
#: Add an entry when a release removes or changes a public symbol; never edit
#: an existing release's set to mean "the current release".
BREAKING_SYMBOLS_BY_VERSION: dict[str, list[str]] = {
    # 0.15.9 — D1/D2 Rust-API surgery (semver-major, shipped in a patch per
    # project policy; Python and C surfaces were additive-only that release).
    "0.15.9": [
        "NodeData::get_property",
        "NodeData::property_iter",
        "NodeData::properties_cloned",
        "DirGraph::column_stores",
        "NodeView",
        "GraphBackend::Forked",
        "GraphRead::node_view",
        "resolve_node_property",
    ],
}


def breaking_symbols_for(version: str) -> list[str]:
    """The symbols this exact release broke, or ``[]``.

    Deliberately an exact-version lookup with no fallback: inheriting the
    previous release's set is precisely the defect this table exists to remove.
    """
    return list(BREAKING_SYMBOLS_BY_VERSION.get(version, []))


def main(argv: list[str] | None = None) -> int:
    global ECOSYSTEM_ROOT
    default_ecosystem_root = ECOSYSTEM_ROOT.resolve()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, default=ECOSYSTEM_ROOT, help="parent directory holding the sibling repos")
    p.add_argument(
        "--upstream-version", default=None, help="override the upstream version (default: KGLite's workspace version)"
    )
    p.add_argument("--show-sites", action="store_true", help="list every non-metadata declaration site individually")
    p.add_argument("--no-docs", action="store_true", help="skip prose version strings in .md/.rst")
    p.add_argument(
        "--fail-on",
        choices=["contradiction", "stale", "none"],
        default="contradiction",
        help="which severity trips a non-zero exit (default: contradiction)",
    )
    p.add_argument(
        "--require-siblings",
        action="store_true",
        help="error when a sibling repo is absent (release gate); default is skip",
    )
    p.add_argument("--json", action="store_true", help="emit findings as JSON")
    p.add_argument("--notify", action="store_true", help="write release notes into affected downstreams' inbox/unread/")
    p.add_argument("--dry-run", action="store_true", help="with --notify: print what would be written, write nothing")
    p.add_argument(
        "--breaking-symbol",
        action="append",
        default=None,
        help="symbol broken by this release; repeatable (default: BREAKING_SYMBOLS_BY_VERSION[<version>], or none)",
    )
    p.add_argument("--date", default=None, help="date stamp for notes (default: today)")
    args = p.parse_args(argv)

    ECOSYSTEM_ROOT = args.root.resolve()
    using_default_root = ECOSYSTEM_ROOT == default_ecosystem_root

    # --- resolve repos, degrading cleanly when siblings are absent
    repos: dict[str, Path] = {}
    missing: list[str] = []
    for repo in (UPSTREAM_REPO, *DOWNSTREAM_REPOS):
        root = ECOSYSTEM_ROOT / repo
        if root.is_dir():
            repos[repo] = root
        else:
            missing.append(repo)

    if using_default_root or UPSTREAM_REPO not in repos:
        # The ecosystem root locates siblings, but the current checkout is the
        # upstream authority. In a release worktree the primary checkout can
        # still be one commit/version behind, so reading <root>/KGLite would
        # generate notes for the previous release.
        repos[UPSTREAM_REPO] = REPO_ROOT
        if UPSTREAM_REPO in missing:
            missing.remove(UPSTREAM_REPO)

    if missing:
        msg = f"note: {len(missing)} sibling repo(s) not present under {ECOSYSTEM_ROOT}: {', '.join(missing)}"
        if args.require_siblings:
            print(msg.replace("note:", "error:"), file=sys.stderr)
            print("      --require-siblings was set; this run cannot be trusted.", file=sys.stderr)
            return 1
        print(f"{msg}\n      skipping them; cross-repo checks cover the rest.\n")

    # --- upstream version
    if args.upstream_version:
        upstream_version = parse_version(args.upstream_version)
        if upstream_version is None:
            print(f"error: unparseable --upstream-version {args.upstream_version!r}", file=sys.stderr)
            return 2
    else:
        upstream_version = repo_own_version(repos[UPSTREAM_REPO])
        if upstream_version is None:
            print(f"error: could not read the upstream version from {repos[UPSTREAM_REPO]}/Cargo.toml", file=sys.stderr)
            return 2

    # --- tracked package set
    tracked = {p.lower() for pkgs in ECOSYSTEM_PACKAGES.values() for p in pkgs}
    tracked |= set(WATCHED_THIRD_PARTY)

    decls: list[Declaration] = []
    for repo, root in repos.items():
        decls += scan_repo(repo, root, tracked, include_docs=not args.no_docs)

    # Each ecosystem package is checked against the version its *own* repo
    # currently declares. The upstream override applies to KGLite's packages so
    # a release can be validated before the bump lands.
    package_owner: dict[str, tuple[str, tuple[int, int, int]]] = {}
    for owner_repo, pkgs in ECOSYSTEM_PACKAGES.items():
        if owner_repo == UPSTREAM_REPO:
            own = upstream_version
        elif owner_repo in repos:
            own = repo_own_version(repos[owner_repo])
        else:
            own = None
        if own is None:
            continue
        for pkg in pkgs:
            package_owner[pkg] = (owner_repo, own)

    findings = analyse(decls, package_owner)

    if args.notify:
        if args.breaking_symbol is not None:
            symbols = args.breaking_symbol
        else:
            symbols = breaking_symbols_for(fmt_version(upstream_version))
        date = args.date or _dt.date.today().isoformat()
        highlights = release_highlights(repos[UPSTREAM_REPO], upstream_version)
        decisions = decide_notifications(findings, decls, repos, upstream_version, symbols)
        return run_notify(decisions, repos, upstream_version, date, highlights, args.dry_run)

    if args.json:
        print(
            json.dumps(
                {
                    "ecosystem_root": str(ECOSYSTEM_ROOT),
                    "upstream": {"repo": UPSTREAM_REPO, "version": fmt_version(upstream_version)},
                    "missing_repos": missing,
                    "declaration_count": len(decls),
                    "findings": [f.as_dict() for f in findings],
                },
                indent=2,
            )
        )
    else:
        report(findings, upstream_version, args.show_sites)

    errors = [f for f in findings if f.kind == "contradiction"]
    stale = [f for f in findings if f.kind == "stale-downstream"]
    warns = [f for f in findings if f.severity == "warn"]
    sites = [f for f in findings if f.kind == "site"]

    if not args.json:
        print(
            f"summary: {len(decls)} declaration(s) scanned across {len(repos)} repo(s) — "
            f"{len(errors)} contradiction(s), {len(stale)} stale downstream(s), "
            f"{len(warns)} warning(s), {len(sites)} non-metadata site(s)."
        )

    if args.fail_on == "none":
        return 0
    if errors:
        return 1
    if args.fail_on == "stale" and stale:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
