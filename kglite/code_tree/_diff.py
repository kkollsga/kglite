"""Structural diff between two ``code_tree`` knowledge graphs.

``kglite.code_tree.diff(graph_a, graph_b)`` compares the code-entity nodes of
two graphs built by :func:`kglite.code_tree.build` — typically two revisions of
the same repository, e.g.::

    old = kglite.code_tree.build("/repo", rev="v1.0")
    new = kglite.code_tree.build("/repo")            # working tree
    delta = kglite.code_tree.diff(old, new)
    print(delta["summary"])   # {'added': 3, 'removed': 1, 'moved': 1, ...}

It is a pure-Python wrapper over the existing per-graph query surface (there is
no cross-graph Cypher — a query runs against a single graph, so a diff must pull
each graph's rows and join them in Python). One bulk ``cypher`` query per
compared node type per graph keeps it to ``2 × len(compared_types)`` queries
regardless of graph size — no per-node round trips.

See :func:`diff` for the exact contract, the identity/moved/changed heuristics,
and what "changed" can and cannot detect.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

# ── Which node types are compared ───────────────────────────────────────────
# The code-entity node types code_tree emits that carry a stable, cross-rev
# ``qualified_name`` identity. Structural context (File, Module, Project) and
# markup entities (Element, Selector, Doc) are intentionally EXCLUDED from
# add/remove/change reporting — a symbol-level rev-to-rev diff is not about
# them. (File moves are still reflected indirectly: a symbol whose file changed
# surfaces as "moved" below.)
_COMPARED_TYPES: tuple[str, ...] = (
    "Function",
    "Class",
    "Struct",
    "Mixin",
    "Enum",
    "Trait",
    "Protocol",
    "Interface",
    "Constant",
)

# ── Per-type change fingerprint ─────────────────────────────────────────────
# The already-materialized node properties whose divergence means "this entity
# changed", computed WITHOUT reparsing source. Each tuple lists the fields that
# form the fingerprint for that type. ``loc_span`` is synthetic
# (``end_line - line_number``): a *position-independent* body-size proxy — it
# does NOT move when unrelated code above the entity shifts, so it flags a body
# edit that grew/shrank the entity without a false positive on every downstream
# symbol. ``line_number`` itself is deliberately NOT a fingerprint field for
# exactly that reason.
#
# What "changed" DETECTS: signature edits (params/return/async — Function),
# visibility flips (pub↔private, all types), constant value edits
# (value_preview — Constant), enum-variant add/remove (variants — Enum),
# struct-field changes (fields — Struct), and body edits that change the line
# span (loc_span — all body-bearing types).
# What it CANNOT detect (documented honesty): a same-line-count body edit with
# an unchanged signature (e.g. ``return 1`` → ``return 2`` inside a function) —
# there is no property that captures body *text*, and we do not parse source.
# Such an edit shows as unchanged.
_FINGERPRINT: dict[str, tuple[str, ...]] = {
    "Function": ("signature", "visibility", "loc_span"),
    "Class": ("visibility", "loc_span"),
    "Struct": ("visibility", "fields", "loc_span"),
    "Mixin": ("visibility", "loc_span"),
    "Enum": ("visibility", "variants", "loc_span"),
    "Trait": ("visibility", "loc_span"),
    "Protocol": ("visibility", "loc_span"),
    "Interface": ("visibility", "loc_span"),
    "Constant": ("visibility", "value_preview"),
}


def _fetch(graph: Any, node_type: str) -> list[dict[str, Any]]:
    """Bulk-fetch every node of ``node_type`` as identity + fingerprint dicts.

    One Cypher query for the whole type. ``file``/``line`` are aliased to the
    stable diff vocabulary; ``loc_span`` is derived post-fetch.
    """
    src_fields = [f for f in _FINGERPRINT[node_type] if f != "loc_span"]
    returns = [
        "n.qualified_name AS qualified_name",
        "n.name AS name",
        "n.file_path AS file",
        "n.line_number AS line",
        "n.end_line AS end_line",
    ]
    returns += [f"n.{f} AS {f}" for f in src_fields]
    query = f"MATCH (n:{node_type}) RETURN " + ", ".join(returns)
    rows: list[dict[str, Any]] = graph.cypher(query).to_dicts()
    for r in rows:
        end_line, line = r.get("end_line"), r.get("line")
        r["loc_span"] = end_line - line if isinstance(end_line, int) and isinstance(line, int) else None
    return rows


# Build-root separators: code_tree prepends the build-directory basename to a
# module path joined with ``.`` (Python/TS/Dart/… dotted languages) OR ``\``
# (PHP without a ``namespace`` — the synthetic module path is
# ``<build-root>\<rel-path>``, backslash-joined per PHP convention). Both leads
# carry the throwaway-tempdir basename and must be neutralised. ``::`` is
# deliberately excluded: Rust (``crate::…``) and C++ (``::``) leads never embed
# the build-root basename and are already stable across builds.
_ROOT_SEPARATORS: tuple[str, ...] = (".", "\\")


def _leading(qualified_name: Any) -> str | None:
    """The first build-root segment of a qualified_name, or ``None``.

    A qualified_name participates only when it starts with a segment delimited
    by ``.`` or ``\\`` — the two separators code_tree uses to join the
    build-root basename onto a module path (dotted languages and unnamespaced
    PHP respectively). The leading segment is everything up to the *first* such
    delimiter. Rust (``crate::…``), external stubs (bare ``HashMap``), and
    rel-path languages produce ``::``-style or delimiter-free leads that are
    *already stable* across builds — they return ``None`` and are never treated
    as a strippable root.
    """
    if not isinstance(qualified_name, str):
        return None
    cuts = [i for i in (qualified_name.find(sep) for sep in _ROOT_SEPARATORS) if i != -1]
    return qualified_name[: min(cuts)] if cuts else None


def _root_alias(own: list[dict[str, Any]], other: list[dict[str, Any]]) -> str | None:
    """The build-root basename to strip from ``own``'s qualified_names.

    code_tree prefixes dotted qualified_names with the basename of the directory
    it built; a ``rev=`` build uses a throwaway tempdir, so the *same* symbol has
    a different prefix in each graph. We can't assume a single common root (a
    mixed-language repo has stable Rust ``crate::`` / rel-path leads alongside
    the unstable Python root), so we detect the root empirically: it is the
    dominant leading segment that appears in ``own`` but **not** in ``other``.
    In two builds of the same tree everything except the root basename is
    byte-identical, so the root is exactly the high-frequency leading segment
    that differs — every stable lead (``crate``, unchanged packages, external
    stubs) appears in both and is excluded. ``None`` when nothing differs
    (e.g. ``diff(g, g)``, or a repo with no root-prefixed language).
    """
    own_counts: dict[str, int] = defaultdict(int)
    for r in own:
        lead = _leading(r.get("qualified_name"))
        if lead is not None:
            own_counts[lead] += 1
    other_leads = {_leading(r.get("qualified_name")) for r in other if _leading(r.get("qualified_name")) is not None}
    only = [lead for lead in own_counts if lead not in other_leads]
    return max(only, key=lambda lead: own_counts[lead]) if only else None


def _strip_root(qualified_name: Any, root: str | None) -> Any:
    if root and isinstance(qualified_name, str):
        for sep in _ROOT_SEPARATORS:
            if qualified_name.startswith(root + sep):
                return qualified_name[len(root) + len(sep) :]
    return qualified_name


def _load(graph: Any, compared: list[str], present: set[str]) -> dict[str, list[dict[str, Any]]]:
    """Fetch every compared type from one graph (one Cypher query per type)."""
    return {t: (_fetch(graph, t) if t in present else []) for t in compared}


def _normalize_roots(
    a_by_type: dict[str, list[dict[str, Any]]],
    b_by_type: dict[str, list[dict[str, Any]]],
) -> None:
    """Strip each graph's build-root basename from its qualified_names in place.

    Makes identity (and the reported qualified_name) stable across builds — the
    whole reason ``diff(build(rev="v1"), build(...))`` matches a symbol instead
    of reporting every symbol as removed+added.
    """
    a_all = [r for recs in a_by_type.values() for r in recs]
    b_all = [r for recs in b_by_type.values() for r in recs]
    root_a = _root_alias(a_all, b_all)
    root_b = _root_alias(b_all, a_all)
    for r in a_all:
        r["qualified_name"] = _strip_root(r.get("qualified_name"), root_a)
    for r in b_all:
        r["qualified_name"] = _strip_root(r.get("qualified_name"), root_b)


def match_entities(
    a_records: list[dict[str, Any]],
    b_records: list[dict[str, Any]],
) -> tuple[
    list[tuple[dict[str, Any], dict[str, Any]]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Join two record lists on ``qualified_name`` — the reusable identity core.

    Returns ``(matched, only_a, only_b)`` where ``matched`` is a list of
    ``(a_record, b_record)`` pairs sharing a qualified_name, ``only_a`` are
    records whose qualified_name is absent from ``b``, and ``only_b`` the
    reverse.

    Factored out deliberately: Phase B.2's multi-rev merge reuses this exact
    identity match (N-way, via repeated pairwise application) — keep it free of
    any report-formatting or code_tree-type knowledge so it stays a pure
    set-join over ``qualified_name``-keyed records.
    """
    a_by = {r["qualified_name"]: r for r in a_records if r.get("qualified_name")}
    b_by = {r["qualified_name"]: r for r in b_records if r.get("qualified_name")}
    a_keys, b_keys = set(a_by), set(b_by)
    matched = [(a_by[k], b_by[k]) for k in a_keys & b_keys]
    only_a = [a_by[k] for k in a_keys - b_keys]
    only_b = [b_by[k] for k in b_keys - a_keys]
    return matched, only_a, only_b


def _changes(a_rec: dict[str, Any], b_rec: dict[str, Any], node_type: str) -> dict[str, dict[str, Any]]:
    """Fingerprint delta for a matched pair: ``{field: {"old", "new"}}``."""
    diffs: dict[str, dict[str, Any]] = {}
    for field in _FINGERPRINT[node_type]:
        av, bv = a_rec.get(field), b_rec.get(field)
        if av != bv:
            diffs[field] = {"old": av, "new": bv}
    return diffs


def _item(rec: dict[str, Any], node_type: str) -> dict[str, Any]:
    """A single add/remove report item from a record."""
    return {
        "type": node_type,
        "qualified_name": rec.get("qualified_name"),
        "name": rec.get("name"),
        "file": rec.get("file"),
        "line": rec.get("line"),
    }


def _detect_moves(
    only_a: list[dict[str, Any]],
    only_b: list[dict[str, Any]],
    node_type: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Classify unmatched records into moved / still-removed / still-added.

    ``moved`` = a record present in ``a`` but not ``b`` (by qualified_name)
    whose *simple name* reappears in ``b`` under a **different file**. This is
    the ONLY move signal: same simple name + different file. It deliberately
    does NOT chase renames — a symbol renamed in place (same file, new name)
    has a new qualified_name AND a new simple name, so it stays a
    remove + add pair, never a "move". Ambiguous same-name candidates are
    paired greedily 1:1; leftovers fall back to remove/add. This keeps the
    heuristic honest at the cost of missing genuine renames (documented).
    """
    b_by_name: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for br in only_b:
        b_by_name[br.get("name")].append(br)

    moved: list[dict[str, Any]] = []
    still_removed: list[dict[str, Any]] = []
    claimed_b: set[int] = set()

    for ar in only_a:
        candidates = b_by_name.get(ar.get("name"), [])
        for br in candidates:
            if id(br) in claimed_b:
                continue
            if br.get("file") != ar.get("file"):
                moved.append(
                    {
                        "type": node_type,
                        "name": ar.get("name"),
                        "old_qualified_name": ar.get("qualified_name"),
                        "new_qualified_name": br.get("qualified_name"),
                        "old_file": ar.get("file"),
                        "new_file": br.get("file"),
                        "old_line": ar.get("line"),
                        "new_line": br.get("line"),
                    }
                )
                claimed_b.add(id(br))
                break
        else:
            still_removed.append(ar)

    still_added = [br for br in only_b if id(br) not in claimed_b]
    return moved, still_removed, still_added


def _is_code_graph(node_types: set[str]) -> bool:
    return bool(set(_COMPARED_TYPES) & node_types)


def diff(graph_a: Any, graph_b: Any) -> dict[str, Any]:
    """Structural diff of two ``code_tree`` code graphs.

    Compares the code-entity nodes (``Function``, ``Class``, ``Struct``,
    ``Mixin``, ``Enum``, ``Trait``, ``Protocol``, ``Interface``, ``Constant``)
    of ``graph_a`` (the "before") and ``graph_b`` (the "after"), joined on each
    node's stable ``qualified_name``. Works on any two code_tree graphs, not
    just two revisions of one repo — it assumes nothing beyond the code_tree
    schema. Pairs naturally with ``build(rev=…)``::

        diff(build("/repo", rev="v1"), build("/repo", rev="v2"))

    Returns a dict::

        {
          "added":   [ {type, qualified_name, name, file, line}, … ],
          "removed": [ {type, qualified_name, name, file, line}, … ],
          "moved":   [ {type, name, old_qualified_name, new_qualified_name,
                        old_file, new_file, old_line, new_line}, … ],
          "changed": [ {type, qualified_name, name, file, line,
                        changes: {field: {old, new}}}, … ],
          "summary": {added, removed, moved, changed, unchanged,
                      types_compared: [str, …]},
        }

    ``file``/``line`` on ``added``/``removed``/``changed`` items come from the
    graph the entity exists in (``changed`` reports ``graph_b``'s location).

    Reported ``qualified_name``s are **root-relative**: code_tree prefixes every
    qualified_name with the basename of the directory it built, and a ``rev=``
    build uses a throwaway tempdir, so that prefix is stripped before matching
    and reporting. This is what lets ``diff(build(rev="v1"), build(...))`` match
    the same symbol across two builds instead of reporting everything as
    removed+added.

    **Identity & the "moved" heuristic (honest contract).** Identity is
    ``qualified_name``. A qualified_name present in one graph and absent from
    the other is a candidate add/remove. Among those, a symbol is reported as
    **moved** only when its *simple name* reappears under a **different file** —
    "moved" means exactly *same simple name, different file*. A genuine
    **rename** (same file, new name) changes both name and qualified_name, so
    it shows as a ``removed`` + ``added`` pair, never a move. This is
    intentional: ``qualified_name`` stability across revs is what makes the diff
    cheap and correct, and over-eager rename detection would mislead more than
    it helps.

    **What "changed" detects and does not.** For a symbol present under the same
    qualified_name in both graphs, ``changed`` fires when a cheap, already-stored
    fingerprint differs — a ``Function`` signature, any type's ``visibility``, a
    ``Constant``'s ``value_preview``, an ``Enum``'s ``variants``, a ``Struct``'s
    ``fields``, or the entity's line span (``end_line - line_number``, which
    catches body edits that changed the number of lines). It does **not** parse
    source, so a same-line-count body edit with an unchanged signature (e.g.
    ``return 1`` → ``return 2``) is invisible and the symbol is treated as
    unchanged. The reported ``changes`` map names each differing fingerprint
    field with its ``old``/``new`` value (``loc_span`` is the line span).

    Raises:
        ValueError: if either graph contains none of the compared code-entity
            node types — i.e. it is empty or was not built by ``code_tree``.
            (``diff(g, g)`` on a real code graph returns all-empty buckets.)
    """
    types_a = set(graph_a.node_types)
    types_b = set(graph_b.node_types)
    for label, node_types in (("graph_a", types_a), ("graph_b", types_b)):
        if not _is_code_graph(node_types):
            raise ValueError(
                f"{label} has none of the code_tree entity types "
                f"({', '.join(_COMPARED_TYPES)}); it looks empty or was not "
                f"built by kglite.code_tree.build(). Cannot diff a non-code graph."
            )

    compared = [t for t in _COMPARED_TYPES if t in types_a or t in types_b]
    a_by_type = _load(graph_a, compared, types_a)
    b_by_type = _load(graph_b, compared, types_b)
    _normalize_roots(a_by_type, b_by_type)

    added: list[dict[str, Any]] = []
    removed: list[dict[str, Any]] = []
    moved: list[dict[str, Any]] = []
    changed: list[dict[str, Any]] = []
    unchanged = 0

    for node_type in compared:
        matched, only_a, only_b = match_entities(a_by_type[node_type], b_by_type[node_type])
        for ar, br in matched:
            ch = _changes(ar, br, node_type)
            if ch:
                item = _item(br, node_type)
                item["changes"] = ch
                changed.append(item)
            else:
                unchanged += 1

        type_moved, still_removed, still_added = _detect_moves(only_a, only_b, node_type)
        moved.extend(type_moved)
        removed.extend(_item(r, node_type) for r in still_removed)
        added.extend(_item(r, node_type) for r in still_added)

    # Deterministic ordering for stable, greppable output.
    added.sort(key=lambda x: (x["type"], x["qualified_name"] or ""))
    removed.sort(key=lambda x: (x["type"], x["qualified_name"] or ""))
    changed.sort(key=lambda x: (x["type"], x["qualified_name"] or ""))
    moved.sort(key=lambda x: (x["type"], x["name"] or "", x["new_file"] or ""))

    return {
        "added": added,
        "removed": removed,
        "moved": moved,
        "changed": changed,
        "summary": {
            "added": len(added),
            "removed": len(removed),
            "moved": len(moved),
            "changed": len(changed),
            "unchanged": unchanged,
            "types_compared": compared,
        },
    }
