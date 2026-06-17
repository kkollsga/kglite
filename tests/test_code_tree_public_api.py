"""Public code_tree build API (Phase C, operator note #3).

Code-graph building must have a stable public entry point — `kglite.build_code_tree`
and `kglite.code_tree.build` — both reachable from a bare `import kglite`, so
consumers don't depend on the private `kglite._kglite_code_tree` module.
"""

import kglite


def _project(tmp_path):
    (tmp_path / "mod.py").write_text(
        "def greet(name):\n"
        "    return f'hi {name}'\n\n\n"
        "class Greeter:\n"
        "    def hello(self):\n"
        "        return greet('world')\n"
    )
    return str(tmp_path)


def test_top_level_build_code_tree(tmp_path):
    g = kglite.build_code_tree(_project(tmp_path))
    n = g.cypher("MATCH (f:Function) RETURN count(f) AS c").to_list()[0]["c"]
    assert n >= 1


def test_code_tree_build_reachable_from_bare_import(tmp_path):
    # No `from kglite import code_tree` first — must work off `import kglite`.
    g = kglite.code_tree.build(_project(tmp_path))
    assert g.cypher("MATCH (n) RETURN count(n) AS c").to_list()[0]["c"] >= 1


def test_public_api_advertised_in_all():
    for name in ("build_code_tree", "from_bytes", "FrozenGraph"):
        assert name in kglite.__all__, f"{name} should be advertised in kglite.__all__"


def test_both_entry_points_agree(tmp_path):
    p = _project(tmp_path)
    a = kglite.build_code_tree(p).cypher("MATCH (n) RETURN count(n) AS c").to_list()[0]["c"]
    b = kglite.code_tree.build(p).cypher("MATCH (n) RETURN count(n) AS c").to_list()[0]["c"]
    assert a == b
