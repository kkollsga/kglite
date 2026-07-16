"""Rust function metrics for the source-quality gate.

The gate's function-size/branch/nesting ratchet used to read metrics off a
code graph built by the retired in-tree `code_tree` builder (via the
`code_tree_stats --function-metrics` bin). The builder now lives in the
external codingest project, and kglite's CI must not depend on a sibling
workspace — so this module ports the *exact metric semantics* the gate
consumes (the parser's `BRANCH_KINDS_RUST` + `compute_complexity` walk) to
Python tree-sitter, scoped to what the gate ever used: production `.rs`
functions.

Semantics mirrored from the retired `parsers/shared.rs`:
  - `branch_count` increments for every node whose kind is a branch kind.
  - `max_nesting` is the deepest stack of nested branch nodes.
  - Nested function items / closures are NOT descended into — they would
    inflate the outer function's metrics.

Identities changed with this port (module-path derivation differs from the
old builder), so the committed baseline was recaptured wholesale in the same
change; the ratchet continues from the new anchor.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

from tree_sitter import Language, Node, Parser
import tree_sitter_rust

BRANCH_KINDS = {
    "if_expression",
    "while_expression",
    "while_let_expression",
    "for_expression",
    "loop_expression",
    "match_arm",
    "try_expression",  # the `?` operator
}
NESTED_SCOPES = {"function_item", "closure_expression"}

_LANGUAGE = Language(tree_sitter_rust.language())


def _complexity(body: Node) -> tuple[int, int]:
    count = 0
    max_depth = 0

    def walk(node: Node, depth: int) -> None:
        nonlocal count, max_depth
        is_branch = node.type in BRANCH_KINDS
        next_depth = depth
        if is_branch:
            count += 1
            next_depth = depth + 1
            max_depth = max(max_depth, next_depth)
        for child in node.children:
            if child.type in NESTED_SCOPES:
                continue
            walk(child, next_depth)

    walk(body, 0)
    return count, max_depth


def _attr_text(node: Node, source: bytes) -> str:
    return source[node.start_byte : node.end_byte].decode("utf-8", "replace")


def _leading_attributes(node: Node, source: bytes) -> str:
    texts = []
    sib = node.prev_named_sibling
    while sib is not None and sib.type == "attribute_item":
        texts.append(_attr_text(sib, source))
        sib = sib.prev_named_sibling
    return "\n".join(texts)


def _is_test_fn(node: Node, source: bytes, in_test_scope: bool) -> bool:
    if in_test_scope:
        return True
    attrs = _leading_attributes(node, source)
    return "test" in attrs  # #[test], #[tokio::test], #[rstest], bench harnesses


def _name_of(node: Node, source: bytes) -> str:
    name = node.child_by_field_name("name")
    if name is not None:
        return _attr_text(name, source)
    ty = node.child_by_field_name("type")
    if ty is not None:
        return _attr_text(ty, source)
    return "?"


def collect_file(path: Path, rel: str) -> list[dict]:
    source = path.read_bytes()
    parser = Parser(_LANGUAGE)
    tree = parser.parse(source)
    out: list[dict] = []

    def visit(node: Node, scope: list[str], in_test_scope: bool) -> None:
        kind = node.type
        if kind == "mod_item":
            attrs = _leading_attributes(node, source)
            test_mod = "cfg(test" in attrs
            body = node.child_by_field_name("body")
            if body is not None:
                name = _name_of(node, source)
                for child in body.children:
                    visit(child, scope + [name], in_test_scope or test_mod)
            return
        if kind in ("impl_item", "trait_item"):
            body = node.child_by_field_name("body")
            if body is not None:
                ty = node.child_by_field_name("type")
                trait = node.child_by_field_name("trait")
                # Keep the trait AND full generic args: `impl Debug for X` and
                # `impl Display for X` (or `Foo<A>` vs `Foo<B>`) must not
                # collide — identities feed a per-function ratchet baseline.
                name = _attr_text(ty, source) if ty is not None else _name_of(node, source)
                if trait is not None:
                    name = f"{_attr_text(trait, source)} for {name}"
                name = " ".join(name.split())
                for child in body.children:
                    visit(child, scope + [name], in_test_scope)
            return
        if kind == "function_item":
            name = _name_of(node, source)
            qualified = "::".join(["crate", *scope, name])
            body = node.child_by_field_name("body")
            branches, nesting = _complexity(body) if body is not None else (0, 0)
            out.append(
                {
                    "path": rel,
                    "qualified_name": qualified,
                    "start_line": node.start_point[0] + 1,
                    "end_line": node.end_point[0] + 1,
                    "branch_count": branches,
                    "max_nesting": nesting,
                    "is_test": _is_test_fn(node, source, in_test_scope),
                }
            )
            # Nested function items become their own entries.
            if body is not None:
                for child in body.children:
                    visit(child, scope + [name], in_test_scope)
            return
        for child in node.children:
            visit(child, scope, in_test_scope)

    visit(tree.root_node, [], False)
    return out


def collect(root: Path) -> list[dict]:
    metrics: list[dict] = []
    for path in sorted((root / "crates").rglob("*.rs")):
        rel = path.relative_to(root).as_posix()
        if "/src/" not in rel or "/target/" in rel:
            continue
        metrics.extend(collect_file(path, rel))
    # `#[cfg]`-gated alternates (unix/windows, feature on/off) share a name;
    # keep the largest variant per identity so the ratchet cap covers every
    # compilation of the function.
    by_identity: dict[tuple[str, str], dict] = {}
    for m in metrics:
        key = (m["path"], m["qualified_name"])
        size = (m["end_line"] - m["start_line"], m["branch_count"], m["max_nesting"])
        prev = by_identity.get(key)
        if prev is None or size > (
            prev["end_line"] - prev["start_line"],
            prev["branch_count"],
            prev["max_nesting"],
        ):
            by_identity[key] = m
    metrics = list(by_identity.values())
    metrics.sort(key=lambda m: (m["path"], m["qualified_name"], m["start_line"]))
    return metrics


def main() -> int:
    root = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path.cwd()
    json.dump(collect(root), sys.stdout, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
