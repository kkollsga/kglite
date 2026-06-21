"""C++ macro-decorated function parsing.

tree-sitter-cpp can parse macro-decorated definitions like
`SPDLOG_INLINE void foo()`, but without a heuristic to skip the macro tokens,
the parser may capture `SPDLOG_INLINE` as the return type and `unknown` as the
function name. The `looks_like_macro_decorator` helper recognizes all-caps
identifiers (length ≥ 2, optional underscores/digits) and skips them in
`get_return_type`, `get_name`, and parameter extraction.
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


def _function(graph, name_suffix: str) -> dict:
    rows = graph.cypher(
        "MATCH (f:Function) WHERE f.qualified_name ENDS WITH $suf RETURN f.name AS n, f.return_type AS rt LIMIT 1",
        params={"suf": name_suffix},
    ).to_list()
    return rows[0] if rows else {}


def test_macro_decorated_function_extracts_real_name_and_type(tmp_path):
    """SPDLOG_INLINE void foo() — name=foo, return_type=void (not SPDLOG_INLINE)."""
    _write(
        tmp_path,
        {
            "main.cpp": """
            #define SPDLOG_INLINE inline

            SPDLOG_INLINE void log_msg() {}
            """,
        },
    )
    g = build(str(tmp_path))
    fn = _function(g, "::log_msg")
    assert fn.get("n") == "log_msg", fn
    assert fn.get("rt") not in (
        "SPDLOG_INLINE",
        "FMT_API",
        None,
    ), f"return_type should be 'void', got {fn.get('rt')!r}"


def test_multiple_macros_stripped(tmp_path):
    """FMT_API FMT_INLINE int bar() — both decorators ignored."""
    _write(
        tmp_path,
        {
            "main.cpp": """
            #define FMT_API
            #define FMT_INLINE inline

            FMT_API FMT_INLINE int compute_value() { return 42; }
            """,
        },
    )
    g = build(str(tmp_path))
    fn = _function(g, "::compute_value")
    assert fn.get("n") == "compute_value", fn
    rt = fn.get("rt") or ""
    assert "FMT" not in rt, f"return type should not contain a macro name: {rt!r}"


def test_constexpr_qualifier_skipped_for_return_type(tmp_path):
    """`constexpr int foo()` — return type should be `int`, not `constexpr`."""
    _write(
        tmp_path,
        {
            "main.cpp": """
            constexpr int compute() { return 42; }
            const bool always_true() { return true; }
            inline static double pi() { return 3.14; }
            """,
        },
    )
    g = build(str(tmp_path))
    assert _function(g, "::compute").get("rt") == "int"
    assert _function(g, "::always_true").get("rt") == "bool"
    assert _function(g, "::pi").get("rt") == "double"


def test_template_type_return_captured(tmp_path):
    """Generic return types like `iteration_proxy<int>` should be captured."""
    _write(
        tmp_path,
        {
            "main.cpp": """
            template <typename T>
            class proxy {};

            class Container {
            public:
                proxy<int> items() const;
            };
            """,
        },
    )
    g = build(str(tmp_path))
    fn = _function(g, "::items")
    rt = fn.get("rt") or ""
    assert "proxy" in rt, f"expected template type, got {rt!r}"


def test_destructor_name_extracted(tmp_path):
    """`~Widget()` — name should be `~Widget`, not `unknown`."""
    _write(
        tmp_path,
        {
            "main.cpp": """
            class Widget {
            public:
                ~Widget() {}
            };
            """,
        },
    )
    g = build(str(tmp_path))
    fn = _function(g, "::~Widget")
    assert fn.get("n") == "~Widget", fn


def test_macro_decorated_constructor_extracts_parameters(tmp_path):
    """`MACRO(3) Constructor(int x)` — parameters captured despite parenthesized
    declarator wrapper that tree-sitter-cpp wraps around macro-prefixed methods."""
    _write(
        tmp_path,
        {
            "main.cpp": """
            #define JSON_HEDLEY_NON_NULL(...)

            class invalid_iterator {
            public:
                JSON_HEDLEY_NON_NULL(3)
                invalid_iterator(int id_, const char* what_arg) {}
            };
            """,
        },
    )
    g = build(str(tmp_path))
    rows = g.cypher(
        "MATCH (f:Function) WHERE f.qualified_name CONTAINS 'invalid_iterator' "
        "RETURN f.name AS n, f.param_count AS p ORDER BY f.line_number LIMIT 3"
    ).to_list()
    # The constructor should have name='invalid_iterator' and param_count=2.
    ctors = [r for r in rows if r["n"] == "invalid_iterator"]
    assert ctors, f"no constructor named 'invalid_iterator' found: {rows}"
    assert ctors[0]["p"] == 2, f"expected 2 params, got {ctors[0]['p']}"


def test_c_style_struct_param_type_extracted(tmp_path):
    """`struct llama_grammar * grammar` — type_annotation should capture
    the `struct llama_grammar` form, not be None. Previously the
    `struct_specifier` AST node fell through extract_parameters' kind
    matchers since `struct_specifier` doesn't contain "type" as substring.
    """
    _write(
        tmp_path,
        {
            "main.cpp": """
            struct llama_grammar { int dummy; };

            void llama_grammar_free(struct llama_grammar * grammar) {}
            """,
        },
    )
    g = build(str(tmp_path))
    rows = g.cypher(
        "MATCH (f:Function) WHERE f.qualified_name ENDS WITH '::llama_grammar_free' RETURN f.parameters AS p"
    ).to_list()
    import json

    raw = rows[0]["p"]
    params = raw if isinstance(raw, list) else (json.loads(raw) if raw else [])
    assert params, f"expected one parameter, got {params}"
    assert "llama_grammar" in (params[0]["type_annotation"] or ""), params[0]


def test_qualified_identifier_method_name_extracted(tmp_path):
    """`bool Foo::bar() const` — name should be `bar`, not `unknown`.
    Out-of-class definitions use `qualified_identifier` (Foo::bar) where
    the existing identifier-walk missed the trailing identifier.
    """
    _write(
        tmp_path,
        {
            "main.cpp": """
            class Foo { public: bool bar() const; };

            bool Foo::bar() const { return true; }
            """,
        },
    )
    g = build(str(tmp_path))
    # Expect the out-of-class definition's name extracted as `bar`.
    rows = g.cypher(
        "MATCH (f:Function) WHERE f.name = 'bar' AND f.qualified_name CONTAINS 'main' RETURN count(*) AS n"
    ).to_list()
    assert rows[0]["n"] >= 1


def test_reference_return_function_extracted(tmp_path):
    """`T & foo()` — `reference_declarator` wraps the function_declarator
    and used to leave name='unknown'.
    """
    _write(
        tmp_path,
        {
            "main.cpp": """
            struct stack { int dummy; };

            stack & get_stack() {
                static stack s; return s;
            }
            """,
        },
    )
    g = build(str(tmp_path))
    rows = g.cypher("MATCH (f:Function) WHERE f.qualified_name ENDS WITH '::get_stack' RETURN f.name AS n").to_list()
    assert rows and rows[0]["n"] == "get_stack", rows


def test_lowercase_function_name_not_treated_as_macro(tmp_path):
    """Don't false-positive on lowercase function names just because they
    contain uppercase letters in CamelCase or snake_case parts."""
    _write(
        tmp_path,
        {
            "main.cpp": """
            int io_count() { return 0; }
            int Reader_init() { return 0; }
            """,
        },
    )
    g = build(str(tmp_path))
    # Both real functions should be captured with their actual names.
    a = _function(g, "::io_count")
    b = _function(g, "::Reader_init")
    assert a.get("n") == "io_count", a
    assert b.get("n") == "Reader_init", b


def _type_names(graph) -> set[str]:
    """Names of every extracted class and struct (the C++ parser labels
    `class` as :Class and `struct` as :Struct)."""
    names = {r["t"] for r in graph.cypher("MATCH (c:Class) RETURN c.title AS t").to_list()}
    names |= {r["t"] for r in graph.cypher("MATCH (s:Struct) RETURN s.title AS t").to_list()}
    return names


def test_export_macro_in_class_keyword_slot_extracts_class(tmp_path):
    """`class KUZU_API Foo { … }` — the export-visibility macro between the
    `class` keyword and the type name used to make tree-sitter-cpp read the
    whole thing as a function (the real class dropped). The class, its base,
    and its methods must now all be extracted."""
    _write(
        tmp_path,
        {
            "lib.cpp": """
            #define KUZU_API
            struct Plain { int x; };
            class KUZU_API PhysicalOperator : public Base {
            public:
                KUZU_API void execute();
            };
            """,
        },
    )
    g = build(str(tmp_path))
    names = _type_names(g)
    assert "PhysicalOperator" in names, names
    assert "Plain" in names, names
    assert _function(g, "::execute").get("n") == "execute"
    assert not g.cypher("MATCH (f:Function) WHERE f.name='unknown' RETURN count(f) AS c").to_list()[0]["c"]


def test_cpp_headers_with_dot_h_extension_are_parsed_as_cpp(tmp_path):
    """Many C++ engines (kuzu, …) use `.h` for C++ headers. `.h` maps to C by
    extension, and the C grammar drops every `class`/`namespace`/`template`. When
    the repo has C++ source, `.h` files must be routed to the C++ parser."""
    _write(
        tmp_path,
        {
            # A definitively-C++ file makes this a C++ repo.
            "src/main.cpp": "int main() { return 0; }\n",
            # The C++ header uses `.h` and is wrapped in nested namespaces.
            "src/include/engine.h": """
            #define ENGINE_API
            namespace eng {
            namespace core {
            class ENGINE_API Engine : public Base {
            public:
                void run();
            };
            struct Config { int n; };
            enum class Mode { Fast, Slow };
            }  // namespace core
            }  // namespace eng
            """,
        },
    )
    g = build(str(tmp_path))
    names = _type_names(g)
    assert "Engine" in names, f"C++ class in a .h file was dropped: {names}"
    assert "Config" in names, names
    assert "Mode" in {r["t"] for r in g.cypher("MATCH (e:Enum) RETURN e.title AS t").to_list()}


def test_pure_c_dot_h_unaffected(tmp_path):
    """A repo with no C++ source keeps `.h` as C (no false C++ routing)."""
    _write(
        tmp_path,
        {
            "main.c": "int main(void) { return 0; }\n",
            "api.h": "struct Point { int x; int y; };\nint add(int a, int b);\n",
        },
    )
    g = build(str(tmp_path))
    # The C struct is still extracted (labeled :Struct), nothing misclassified.
    structs = {r["t"] for r in g.cypher("MATCH (s:Struct) RETURN s.title AS t").to_list()}
    assert "Point" in structs, structs
