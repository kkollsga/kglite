"""Skill loading + applies_when filtering for the Python entry point.

kglite ships per-tool and cross-tool methodology as bundled markdown files
inside the wheel (`kglite/mcp_server/skills/*.md`). At boot the Python entry
point discovers them (see `load_kglite_bundled_skills`), merges them with the
framework's `SkillRegistry.from_manifest` output (which carries framework
defaults + the manifest's `<basename>.skills/` project layer + operator-
declared paths), filters by `applies_when:` against the active graph's schema,
and surfaces each active skill two ways: injected into the tool descriptions it
attaches to (the live, agent-reachable channel — see `_apply_skill_hint` in
`server.py`) and as MCP `prompts/list` + `prompts/get` handlers (a fallback for
the rare clients that expose prompts to agents).

Why the load/merge/filter logic lives in Python rather than calling into Rust
(the 4.2 consolidation the operator proposal raises):

    kglite's Python entry point uses `mcp.server.lowlevel.Server`, not FastMCP.
    The mcp-methods crate (pinned 0.3.41; see crates/kglite-py/Cargo.toml)
    exposes only a FastMCP prompt registrar (`register_skills_as_prompts`) —
    there is NO lowlevel registrar. So the orchestration (layering, frontmatter
    parse, `applies_when` evaluation) is still owned upstream and reached via
    `kglite._mcp_internal.SkillRegistry.from_manifest`; this module only adds
    the kglite-bundled layer and re-evaluates `applies_when` against the LIVE
    graph at request time (the Rust registry can't, since the graph mutates
    post-boot in workspace mode). The `AppliesWhen`/`Skill` replica here can be
    deleted only once mcp-methods ships a lowlevel-server skill/prompt
    registrar (track for >=0.3.42); until then it is a deliberate, minimal
    lowlevel shim, not migration debt.

No standalone `mcp_methods` PyPI wheel is imported (dropped in 0.9.40);
everything goes through the vendored-via-Cargo Rust crate at one pinned
version. See docs/python/guides/mcp-skills.md for the operator-facing
skill-authoring guide.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import importlib.resources
import logging
from pathlib import Path
from typing import Any, Iterable

log = logging.getLogger("kglite.mcp_server.skills_loader")


@dataclass
class AppliesWhen:
    """Parsed `applies_when:` predicate block. None for a field means
    the predicate is absent (not declared)."""

    graph_has_node_type: list[str] | None = None
    graph_has_property: tuple[str, str] | None = None
    tool_registered: str | None = None
    extension_enabled: str | None = None

    def is_active(
        self,
        *,
        has_node_type: callable,
        has_property: callable,
        registered_tools: set[str],
        extensions: dict[str, Any],
    ) -> bool:
        """Evaluate every declared predicate against runtime state.
        AND semantics across populated clauses. An undeclared clause
        is treated as satisfied. A declared clause that the relevant
        checker reports false suppresses the skill."""
        if self.graph_has_node_type is not None and not any(has_node_type(t) for t in self.graph_has_node_type):
            return False
        if self.graph_has_property is not None:
            nt, prop = self.graph_has_property
            if not has_property(nt, prop):
                return False
        if self.tool_registered is not None and self.tool_registered not in registered_tools:
            return False
        if self.extension_enabled is not None and not extensions.get(self.extension_enabled):
            return False
        return True


@dataclass
class Skill:
    """A loaded skill — frontmatter metadata plus the markdown body."""

    name: str
    description: str
    body: str
    provenance: str  # "kglite-bundled" | "framework" | "project" | "domain_pack" | str
    auto_inject_hint: bool = True
    references_tools: list[str] = field(default_factory=list)
    applies_when: AppliesWhen | None = None


def _parse_frontmatter(text: str, path_for_error: str) -> tuple[dict[str, Any], str]:
    """Split YAML frontmatter from the markdown body. Returns
    `(frontmatter_dict, body_text)`. Frontmatter is the block between
    the first two `---` delimiters; missing-frontmatter case raises."""
    import yaml

    lines = text.splitlines(keepends=True)
    if not lines or lines[0].strip() != "---":
        raise ValueError(f"{path_for_error}: missing opening `---` frontmatter delimiter")

    end_idx = None
    for i, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            end_idx = i
            break
    if end_idx is None:
        raise ValueError(f"{path_for_error}: missing closing `---` frontmatter delimiter")

    frontmatter_text = "".join(lines[1:end_idx])
    body = "".join(lines[end_idx + 1 :]).lstrip()
    try:
        fm = yaml.safe_load(frontmatter_text) or {}
    except yaml.YAMLError as e:
        raise ValueError(f"{path_for_error}: frontmatter YAML parse error: {e}") from None
    if not isinstance(fm, dict):
        raise ValueError(f"{path_for_error}: frontmatter must be a mapping")
    return fm, body


def _parse_applies_when(raw: Any, path_for_error: str) -> AppliesWhen | None:
    """Parse the `applies_when:` block from frontmatter. Returns None
    when absent. Validates that each declared key is one of the
    framework's bounded predicate set; unknown keys raise."""
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError(f"{path_for_error}: applies_when must be a mapping")

    aw = AppliesWhen()
    for key, value in raw.items():
        if key == "graph_has_node_type":
            if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
                raise ValueError(f"{path_for_error}: applies_when.graph_has_node_type must be a list of strings")
            aw.graph_has_node_type = list(value)
        elif key == "graph_has_property":
            if not isinstance(value, dict) or "node_type" not in value or "prop_name" not in value:
                raise ValueError(
                    f"{path_for_error}: applies_when.graph_has_property must be a mapping with "
                    "`node_type` and `prop_name`"
                )
            aw.graph_has_property = (str(value["node_type"]), str(value["prop_name"]))
        elif key == "tool_registered":
            aw.tool_registered = str(value)
        elif key == "extension_enabled":
            aw.extension_enabled = str(value)
        else:
            raise ValueError(f"{path_for_error}: unknown applies_when key {key!r}")
    return aw


def _skill_from_text(text: str, path_for_error: str, provenance: str) -> Skill:
    fm, body = _parse_frontmatter(text, path_for_error)
    name = fm.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError(f"{path_for_error}: frontmatter.name must be a non-empty string")
    description = fm.get("description", "")
    if not isinstance(description, str):
        raise ValueError(f"{path_for_error}: frontmatter.description must be a string")
    auto_inject = fm.get("auto_inject_hint", True)
    if not isinstance(auto_inject, bool):
        raise ValueError(f"{path_for_error}: frontmatter.auto_inject_hint must be a bool")
    refs = fm.get("references_tools", [])
    if not isinstance(refs, list) or not all(isinstance(r, str) for r in refs):
        raise ValueError(f"{path_for_error}: frontmatter.references_tools must be a list of strings")
    applies_when = _parse_applies_when(fm.get("applies_when"), path_for_error)
    return Skill(
        name=name,
        description=description,
        body=body,
        provenance=provenance,
        auto_inject_hint=auto_inject,
        references_tools=list(refs),
        applies_when=applies_when,
    )


def load_kglite_bundled_skills() -> list[Skill]:
    """Discover and load every kglite-bundled skill from the wheel's
    `skills/` package-data directory — one `Skill` per `*.md` file whose
    frontmatter parses.

    0.10.25: switched from a hand-maintained `KGLITE_BUNDLED_SKILL_NAMES`
    allowlist to directory discovery. The allowlist silently orphaned any
    skill file added to `skills/` but not also added to the tuple (e.g.
    `explore.md` shipped inert for several releases). Discovery means
    dropping a `<name>.md` into `skills/` is all it takes to bundle it.
    A file that doesn't parse as a skill (bad/empty frontmatter — e.g. a
    stray README) is logged and skipped, so the directory tolerates
    non-skill markdown without the manual gate.

    To exclude a skill file from bundling without deleting it, prefix its
    name with an underscore (`_draft.md`) — discovery skips dotfiles and
    underscore-prefixed files."""
    skills: list[Skill] = []
    base = importlib.resources.files("kglite.mcp_server").joinpath("skills")
    paths = sorted(
        (p for p in base.iterdir() if p.name.endswith(".md") and not p.name.startswith((".", "_"))),
        key=lambda p: p.name,
    )
    for path in paths:
        name = path.name[: -len(".md")]
        try:
            text = path.read_text(encoding="utf-8")
        except (FileNotFoundError, OSError) as e:
            log.warning("kglite bundled skill %r unreadable: %s", name, e)
            continue
        try:
            skill = _skill_from_text(text, f"<bundled:{name}>", "kglite-bundled")
        except ValueError as e:
            log.warning("kglite bundled skill file %r skipped (not a skill): %s", name, e)
            continue
        skills.append(skill)
    return skills


def merge_skills(
    kglite_bundled: Iterable[Skill],
    framework_skills: Iterable[Skill],
) -> dict[str, Skill]:
    """Merge kglite's bundled skills with framework / project /
    operator skills. Framework-side entries win on name collision
    because mcp-methods' `from_manifest` returns the already-layered
    set — its priorities are project > domain_pack > framework_bundled.
    Anything coming from there is already the right choice; kglite's
    bundled is the lowest layer."""
    merged: dict[str, Skill] = {s.name: s for s in kglite_bundled}
    for s in framework_skills:
        merged[s.name] = s
    return merged


def _framework_skill_to_local(fw_skill: Any) -> Skill:
    """Convert a `kglite._mcp_internal.Skill` (pyo3) into our local dataclass.
    Handles the framework's `applies_when` dict shape on the way in."""
    aw_dict = fw_skill.applies_when
    applies_when: AppliesWhen | None = None
    if aw_dict:
        applies_when = AppliesWhen(
            graph_has_node_type=aw_dict.get("graph_has_node_type"),
            graph_has_property=(
                (aw_dict["graph_has_property"]["node_type"], aw_dict["graph_has_property"]["prop_name"])
                if "graph_has_property" in aw_dict
                else None
            ),
            tool_registered=aw_dict.get("tool_registered"),
            extension_enabled=aw_dict.get("extension_enabled"),
        )
    return Skill(
        name=fw_skill.name,
        description=fw_skill.description,
        body=fw_skill.body,
        provenance=fw_skill.provenance,
        auto_inject_hint=fw_skill.auto_inject_hint,
        references_tools=list(fw_skill.references_tools),
        applies_when=applies_when,
    )


def load_framework_skills(manifest_path: Path) -> list[Skill]:
    """Load framework defaults + project layer + operator-declared
    paths via `kglite._mcp_internal.SkillRegistry.from_manifest`. Returns
    an empty list when the manifest disables skills (`skills: false` or
    absent) — the framework registry yields no entries in that case.

    0.9.40+: dropped the dependency on the `mcp_methods` PyPI wheel.
    `kglite._mcp_internal` is built from `src/mcp_tools.rs` and delegates to
    `mcp_methods::server::SkillRegistry::from_manifest` (the Rust crate pinned
    in crates/kglite-py/Cargo.toml — currently 0.3.41; `from_manifest` landed
    in 0.3.38) — same layering orchestration upstream ships, no kglite-side
    replica of it."""
    try:
        from kglite import _mcp_internal as mcp_internal
    except ImportError:
        log.warning("kglite._mcp_internal not built; skipping framework skills")
        return []
    try:
        registry = mcp_internal.SkillRegistry.from_manifest(str(manifest_path))
    except Exception as e:  # noqa: BLE001
        log.warning("SkillRegistry.from_manifest failed: %s", e)
        return []
    return [_framework_skill_to_local(s) for s in registry.skills()]


def build_active_skill_set(
    manifest_path: Path | None,
    *,
    has_node_type: callable,
    has_property: callable,
    registered_tools: set[str],
    extensions: dict[str, Any],
    skills_opted_in: bool,
) -> dict[str, Skill]:
    """Top-level entry: load + merge + filter by applies_when. Returns
    a name → Skill dict of skills that should be exposed via
    prompts/list.

    `skills_opted_in` mirrors the framework's `SkillsSource` semantics:
    `False` (or `None` / unset on the manifest) → skills completely
    disabled, return empty. `True` → kglite-bundled + framework defaults
    + project / operator layers per the manifest's `skills:` declaration.

    Returns an empty dict when no manifest is present (bare mode —
    skills require a manifest) or when the manifest disables skills."""
    if manifest_path is None or not skills_opted_in:
        return {}
    kglite_bundled = load_kglite_bundled_skills()
    framework_skills = load_framework_skills(manifest_path)
    merged = merge_skills(kglite_bundled, framework_skills)
    active: dict[str, Skill] = {}
    for name, skill in merged.items():
        if skill.applies_when is None:
            active[name] = skill
            continue
        if skill.applies_when.is_active(
            has_node_type=has_node_type,
            has_property=has_property,
            registered_tools=registered_tools,
            extensions=extensions,
        ):
            active[name] = skill
        else:
            log.info("skill %r filtered out by applies_when (graph state mismatch)", name)
    return active
