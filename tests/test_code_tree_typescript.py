"""TypeScript / TSX parsing.

`.tsx` files must be parsed with the JSX-aware grammar (`LANGUAGE_TSX`). With
the plain TypeScript grammar, every JSX component body desyncs into ERROR nodes
and the enclosing `export default function App()` loses its name (→ `unknown`).
"""

import textwrap

import pytest

pytest.importorskip("tree_sitter")

from kglite.code_tree import build  # noqa: E402


def _write(tmp_path, files: dict[str, str]) -> None:
    for rel, content in files.items():
        fp = tmp_path / rel
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(textwrap.dedent(content))


def _names(graph) -> set[str]:
    return {r["n"] for r in graph.cypher("MATCH (f:Function) RETURN f.name AS n").to_list()}


def test_tsx_export_default_function_named(tmp_path):
    """`export default function App() { return <div/> }` — the component is
    named `App`, not `unknown`, and the JSX body parses cleanly."""
    _write(
        tmp_path,
        {
            "src/App.tsx": """
            import { useState } from 'react'

            export default function App() {
                const [n, setN] = useState<number>(0)
                return <div className="app" onClick={() => setN(n + 1)}>{n}</div>
            }
            """,
            "src/Widget.tsx": """
            type Props = { title: string }

            export default function Widget({ title }: Props) {
                return <section><h1>{title}</h1></section>
            }
            """,
        },
    )
    names = _names(build(str(tmp_path)))
    assert "App" in names, names
    assert "Widget" in names, names
    assert "unknown" not in names, names


def test_ts_generics_not_misparsed_as_jsx(tmp_path):
    """A plain `.ts` file with type assertions / generics must keep using the
    TypeScript grammar (TSX would read `<T>` as JSX and mangle it)."""
    _write(
        tmp_path,
        {
            "src/util.ts": """
            export function identity<T>(x: T): T {
                return x
            }

            export function castIt(v: unknown): string {
                return (v as string)
            }
            """,
        },
    )
    names = _names(build(str(tmp_path)))
    assert "identity" in names, names
    assert "castIt" in names, names
    assert "unknown" not in names, names
