"""Calls inside *anonymous* functions (lambdas / closures / arrow
functions / func literals) must be attributed to the enclosing named
function — the anonymous function gets no graph node of its own, so its
call sites belong to whoever encloses it. This mirrors the Rust closure
handling (see test_code_tree_calls.py::TestRustClosureWalking) and closes
a class of extraction bug where every parser that listed its anonymous-
function node kind in the call-walk skip set silently dropped those calls.

One fixture per affected language; each asserts the enclosing function
resolves a CALLS edge to a unique-named in-project helper invoked only
from inside an anonymous function body.
"""

import textwrap

import pytest

pytest.importorskip("tree_sitter")

from kglite.code_tree import build  # noqa: E402


def _call_pairs(graph) -> set[tuple[str, str]]:
    rows = graph.cypher("MATCH (a:Function)-[:CALLS]->(b:Function) RETURN a.name AS caller, b.name AS callee").to_list()
    return {(r["caller"], r["callee"]) for r in rows}


def _build_one(tmp_path, rel: str, content: str):
    fp = tmp_path / rel
    fp.parent.mkdir(parents=True, exist_ok=True)
    fp.write_text(textwrap.dedent(content))
    return build(str(tmp_path))


def test_python_lambda(tmp_path):
    g = _build_one(
        tmp_path,
        "pkg/a.py",
        """
        def helper(x):
            return x + 1

        def caller(items):
            return sorted(items, key=lambda x: helper(x))
        """,
    )
    assert ("caller", "helper") in _call_pairs(g)


def test_typescript_arrow_callback(tmp_path):
    g = _build_one(
        tmp_path,
        "a.ts",
        """
        function helper(x: number): number { return x + 1; }

        function caller(items: number[]): number[] {
            return items.map(x => helper(x));
        }
        """,
    )
    assert ("caller", "helper") in _call_pairs(g)


def test_go_func_literal(tmp_path):
    g = _build_one(
        tmp_path,
        "main.go",
        """
        package main

        func helper(x int) int { return x + 1 }

        func caller() int {
            f := func() int { return helper(1) }
            return f()
        }
        """,
    )
    assert ("caller", "helper") in _call_pairs(g)


def test_java_lambda(tmp_path):
    g = _build_one(
        tmp_path,
        "A.java",
        """
        import java.util.List;
        class A {
            void helper(int x) {}
            void caller(List<Integer> xs) {
                xs.forEach(x -> helper(x));
            }
        }
        """,
    )
    assert ("caller", "helper") in _call_pairs(g)


def test_csharp_lambda(tmp_path):
    g = _build_one(
        tmp_path,
        "A.cs",
        """
        using System;
        class A {
            void Helper() {}
            void Caller() {
                Action a = () => Helper();
                a();
            }
        }
        """,
    )
    assert ("Caller", "Helper") in _call_pairs(g)


def test_cpp_lambda(tmp_path):
    g = _build_one(
        tmp_path,
        "a.cpp",
        """
        int helper(int x) { return x + 1; }

        int caller() {
            auto f = []() { return helper(1); };
            return f();
        }
        """,
    )
    assert ("caller", "helper") in _call_pairs(g)
