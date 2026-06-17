"""#4 — consistent provenance flags on code-graph nodes (operator report 2026-06-16).

`is_test` already existed on File / Module / Function. These tests cover the
additions:

  - `is_benchmark` (path-based: asv_bench/, benchmarks/, bench/) on File /
    Module / Function / Class.
  - `is_test` extended to Class (so test classes like xarray's PlotTestCase can
    be excluded from fan-out / centrality queries).
  - `is_generated` on File (derived from the skip_reason of machine-produced
    files).
"""

from pathlib import Path


def _build(src: Path):
    from kglite import code_tree

    return code_tree.build(str(src))


def _make_repo(root: Path) -> None:
    lib = root / "lib.py"
    lib.write_text("def real():\n    return 1\n\nclass Real:\n    pass\n")

    tests = root / "tests"
    tests.mkdir()
    (tests / "test_thing.py").write_text("def test_real():\n    return 1\n\nclass PlotTestCase:\n    pass\n")

    bench = root / "asv_bench" / "benchmarks"
    bench.mkdir(parents=True)
    (bench / "bench_thing.py").write_text("def time_real():\n    return 1\n\nclass BenchSuite:\n    pass\n")


def test_is_benchmark_on_function_and_class(tmp_path: Path) -> None:
    root = tmp_path / "proj"
    root.mkdir()
    _make_repo(root)
    g = _build(root)

    bench_fns = {r["n"] for r in g.cypher("MATCH (f:Function) WHERE f.is_benchmark = true RETURN f.name AS n")}
    assert "time_real" in bench_fns, bench_fns
    assert "real" not in bench_fns and "test_real" not in bench_fns

    bench_cls = {r["n"] for r in g.cypher("MATCH (c:Class) WHERE c.is_benchmark = true RETURN c.name AS n")}
    assert "BenchSuite" in bench_cls, bench_cls
    assert "Real" not in bench_cls


def test_is_test_on_class(tmp_path: Path) -> None:
    """Class nodes now carry is_test, so test classes can be filtered out."""
    root = tmp_path / "proj"
    root.mkdir()
    _make_repo(root)
    g = _build(root)

    test_cls = {r["n"] for r in g.cypher("MATCH (c:Class) WHERE c.is_test = true RETURN c.name AS n")}
    assert "PlotTestCase" in test_cls, test_cls

    # The library-only filter (the analysis use case) excludes test + benchmark.
    lib_cls = {
        r["n"]
        for r in g.cypher("MATCH (c:Class) WHERE c.is_test = false AND c.is_benchmark = false RETURN c.name AS n")
    }
    assert "Real" in lib_cls, lib_cls
    assert "PlotTestCase" not in lib_cls
    assert "BenchSuite" not in lib_cls


def test_is_benchmark_on_file(tmp_path: Path) -> None:
    root = tmp_path / "proj"
    root.mkdir()
    _make_repo(root)
    g = _build(root)

    bench_files = {r["p"] for r in g.cypher("MATCH (f:File) WHERE f.is_benchmark = true RETURN f.path AS p")}
    assert any("asv_bench" in p for p in bench_files), bench_files
    assert not any(p == "lib.py" for p in bench_files)


def test_is_generated_on_file(tmp_path: Path) -> None:
    """A minified/generated file is skipped but its File node reports
    is_generated = true; a hand-written file reports false."""
    root = tmp_path / "proj"
    root.mkdir()
    (root / "hand.py").write_text("def real():\n    return 1\n")
    # A minified JS file = one huge line (>=1024 bytes, no newline) → skipped
    # as "minified", which is_generated derives from. No trailing newline.
    long_line = "var x=" + ";".join(f"a{i}=1" for i in range(400)) + ";"
    assert len(long_line) >= 1024 and "\n" not in long_line
    (root / "bundle.min.js").write_text(long_line)
    g = _build(root)

    gen = {r["p"]: r["g"] for r in g.cypher("MATCH (f:File) RETURN f.path AS p, f.is_generated AS g")}
    # Hand-written file is explicitly not generated.
    assert gen.get("hand.py") is False, gen
    # The minified bundle, if present as a File node, is flagged generated.
    min_files = [p for p in gen if p.endswith(".min.js")]
    for p in min_files:
        assert gen[p] is True, (p, gen[p])


def test_is_external_on_function(tmp_path: Path) -> None:
    """A1a (operator report 2026-06-17): `is_external` must be emitted on
    Function (= false; every Function is in-repo), uniform with Class/File, so
    the advertised `WHERE n.is_external = false` library-only filter works on
    Function instead of silently matching nothing (null = false)."""
    root = tmp_path / "proj"
    root.mkdir()
    _make_repo(root)
    g = _build(root)

    # Every Function reports is_external explicitly false — never null.
    rows = g.cypher("MATCH (f:Function) RETURN f.name AS n, f.is_external AS ext")
    assert rows, "expected Function nodes"
    for r in rows:
        assert r["ext"] is False, f"{r['n']} has is_external={r['ext']!r} (expected False, not null)"

    # The documented library-only filter selects in-repo functions (not empty).
    lib_only = {
        r["n"]
        for r in g.cypher("MATCH (f:Function) WHERE f.is_external = false AND f.is_test = false RETURN f.name AS n")
    }
    assert "real" in lib_only, lib_only
