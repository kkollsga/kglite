//! C and C++ language parsers (ported from parsers/cpp.py).

use serde_json::json;
use std::path::Path;
use tree_sitter::{Node, Parser, Tree};

use super::shared::{
    compute_complexity, count_lines, extract_comment_annotations, extract_procedure_annotations,
    get_type_parameters, is_generated_or_minified, looks_like_macro_decorator, node_text,
    BRANCH_KINDS_CPP, DEFAULT_COMMENT_TYPES,
};
use super::LanguageParser;
use crate::code_tree::models::{
    AttributeInfo, ClassInfo, ConstantInfo, EnumInfo, FileInfo, FunctionInfo, ParameterInfo,
    ParameterKind, ParseResult, TypeRelationship,
};

pub const C_NOISE_NAMES: &[&str] = &[
    "printf", "fprintf", "sprintf", "snprintf", "scanf", "sscanf", "puts", "fputs", "fgets",
    "getchar", "putchar", "malloc", "calloc", "realloc", "free", "exit", "abort", "atexit", "atoi",
    "atof", "memcpy", "memset", "memmove", "memcmp", "strlen", "strcmp", "strncmp", "strcpy",
    "strncpy", "strcat", "strstr", "strchr", "assert", "fopen", "fclose", "fread", "fwrite",
    "fseek", "ftell",
];

pub const CPP_NOISE_NAMES: &[&str] = &[
    // C noise
    "printf",
    "fprintf",
    "sprintf",
    "snprintf",
    "scanf",
    "sscanf",
    "puts",
    "fputs",
    "fgets",
    "getchar",
    "putchar",
    "malloc",
    "calloc",
    "realloc",
    "free",
    "exit",
    "abort",
    "atexit",
    "atoi",
    "atof",
    "memcpy",
    "memset",
    "memmove",
    "memcmp",
    "strlen",
    "strcmp",
    "strncmp",
    "strcpy",
    "strncpy",
    "strcat",
    "strstr",
    "strchr",
    "assert",
    "fopen",
    "fclose",
    "fread",
    "fwrite",
    "fseek",
    "ftell",
    // C++ extras
    "size",
    "empty",
    "begin",
    "end",
    "cbegin",
    "cend",
    "rbegin",
    "rend",
    "push_back",
    "pop_back",
    "emplace_back",
    "push_front",
    "pop_front",
    "emplace_front",
    "insert",
    "erase",
    "find",
    "count",
    "at",
    "front",
    "back",
    "data",
    "resize",
    "reserve",
    "clear",
    "swap",
    "shrink_to_fit",
    "move",
    "forward",
    "make_shared",
    "make_unique",
    "get",
    "reset",
    "cout",
    "cerr",
    "endl",
    "static_cast",
    "dynamic_cast",
    "reinterpret_cast",
    "const_cast",
];

const NESTED_SCOPES: &[&str] = &["function_definition", "lambda_expression"];

const TYPE_NODES: &[&str] = &[
    "primitive_type",
    "type_identifier",
    "sized_type_specifier",
    "struct_specifier",
    "enum_specifier",
    "union_specifier",
    // Generic / template return types: `iteration_proxy<int>`, `std::vector<T>`.
    "template_type",
    "qualified_identifier",
    // NOTE: `type_qualifier` (const, volatile, constexpr) intentionally excluded —
    // it's a qualifier, not a type. get_return_type skips past it to find the real type.
];

pub enum CppFlavor {
    C,
    Cpp,
}

pub struct CppParser {
    flavor: CppFlavor,
}

thread_local! {
    static C_PARSER: std::cell::RefCell<Parser> = {
        let mut p = Parser::new();
        p.set_language(&tree_sitter_c::LANGUAGE.into())
            .expect("loading tree-sitter-c grammar");
        std::cell::RefCell::new(p)
    };
    static CPP_PARSER: std::cell::RefCell<Parser> = {
        let mut p = Parser::new();
        p.set_language(&tree_sitter_cpp::LANGUAGE.into())
            .expect("loading tree-sitter-cpp grammar");
        std::cell::RefCell::new(p)
    };
}

impl CppParser {
    pub fn c() -> Self {
        CppParser {
            flavor: CppFlavor::C,
        }
    }
    pub fn cpp() -> Self {
        CppParser {
            flavor: CppFlavor::Cpp,
        }
    }

    fn parse_tree(&self, source: &[u8]) -> Option<Tree> {
        if self.is_cpp() {
            CPP_PARSER.with(|p| p.borrow_mut().parse(source, None))
        } else {
            C_PARSER.with(|p| p.borrow_mut().parse(source, None))
        }
    }

    fn is_cpp(&self) -> bool {
        matches!(self.flavor, CppFlavor::Cpp)
    }
    fn sep(&self) -> &'static str {
        if self.is_cpp() {
            "::"
        } else {
            "/"
        }
    }

    fn get_name<'a>(node: Node<'a>, source: &'a [u8], name_type: &str) -> Option<&'a str> {
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            // Destructor names: `~Widget` is a `destructor_name` node wrapping
            // the type identifier. Use its full text (including the `~`) so
            // graph queries can distinguish destructors from other members.
            if child.kind() == "destructor_name" {
                return Some(node_text(child, source));
            }
            // Out-of-class definitions like `bool Foo::bar() const` produce
            // a `qualified_identifier` (e.g. `Foo::bar`) where we want the
            // last segment as the method name. Drill down to the trailing
            // identifier child.
            if child.kind() == "qualified_identifier" {
                let mut qc = child.walk();
                let mut last_id: Option<&str> = None;
                for sub in child.children(&mut qc) {
                    if matches!(sub.kind(), "identifier" | "destructor_name") {
                        last_id = Some(node_text(sub, source));
                    }
                }
                if let Some(id) = last_id {
                    return Some(id);
                }
            }
            if child.kind() == name_type
                || child.kind() == "type_identifier"
                || child.kind() == "field_identifier"
                || child.kind() == "identifier"
            {
                let text = node_text(child, source);
                // Skip macro-shaped tokens (SPDLOG_INLINE, FMT_API, etc.) so
                // the parser doesn't pick them up as the function name.
                if looks_like_macro_decorator(text) {
                    continue;
                }
                return Some(text);
            }
        }
        None
    }

    fn get_doc_comment(node: Node, source: &[u8]) -> Option<String> {
        let sibling = node.prev_named_sibling()?;
        if sibling.kind() != "comment" {
            return None;
        }
        let text = node_text(sibling, source).trim();
        if let Some(rest) = text.strip_prefix("/**") {
            let rest = rest.strip_suffix("*/").unwrap_or(rest);
            let mut lines = Vec::new();
            for line in rest.split('\n') {
                let line = line.trim();
                let cleaned = if let Some(r) = line.strip_prefix("* ") {
                    r
                } else if let Some(r) = line.strip_prefix('*') {
                    r
                } else {
                    line
                };
                lines.push(cleaned);
            }
            let joined = lines.join("\n").trim().to_string();
            return if joined.is_empty() {
                None
            } else {
                Some(joined)
            };
        }
        if text.starts_with("///") {
            let mut doc_lines: Vec<String> = Vec::new();
            let mut s = Some(sibling);
            while let Some(cur) = s {
                if cur.kind() != "comment" {
                    break;
                }
                let text = node_text(cur, source).trim();
                let Some(rest) = text.strip_prefix("///") else {
                    break;
                };
                let content = rest.strip_prefix(' ').unwrap_or(rest);
                doc_lines.insert(0, content.to_string());
                s = cur.prev_named_sibling();
            }
            if doc_lines.is_empty() {
                None
            } else {
                Some(doc_lines.join("\n"))
            }
        } else {
            None
        }
    }

    fn get_signature(node: Node, source: &[u8]) -> String {
        let mut parts: Vec<&str> = Vec::new();
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            if matches!(
                child.kind(),
                "compound_statement" | "field_declaration_list"
            ) {
                break;
            }
            parts.push(node_text(child, source));
        }
        parts.join(" ")
    }

    fn get_return_type(node: Node, source: &[u8]) -> Option<String> {
        // C/C++ primitive-type keywords. Used to recover return types when
        // tree-sitter-cpp wraps the type keyword in an ERROR node — typically
        // happens when a macro decorator confuses the parser (e.g.
        // `SPDLOG_INLINE void foo()` parses as type_identifier=SPDLOG_INLINE
        // followed by ERROR(void) instead of primitive_type=void).
        const PRIMITIVE_KEYWORDS: &[&str] = &[
            "void", "bool", "char", "int", "short", "long", "float", "double", "signed",
            "unsigned", "size_t", "ssize_t", "auto",
        ];
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            let text = node_text(child, source);
            // Skip macro decorators like `SPDLOG_INLINE`, `FMT_API`. Without
            // this, tree-sitter-cpp may surface them as bare `identifier` or
            // `type_identifier` children, producing return_type="SPDLOG_INLINE"
            // instead of the real type.
            if looks_like_macro_decorator(text) {
                continue;
            }
            // Skip C++ qualifiers — `const`, `volatile`, `constexpr`, `inline`,
            // `static`, `noexcept` etc. They aren't types but appear before the
            // real return type. Without skipping, `constexpr int foo()` returns
            // "constexpr" as the type.
            if matches!(
                child.kind(),
                "type_qualifier" | "storage_class_specifier" | "virtual_specifier"
            ) {
                continue;
            }
            // Recover from tree-sitter ERROR wrappers around primitive types.
            if child.kind() == "ERROR" {
                let trimmed = text.trim();
                if PRIMITIVE_KEYWORDS.contains(&trimmed) {
                    return Some(trimmed.to_string());
                }
                continue;
            }
            if matches!(
                child.kind(),
                "function_declarator" | "identifier" | "pointer_declarator"
            ) {
                break;
            }
            if TYPE_NODES.contains(&child.kind()) {
                return Some(text.to_string());
            }
        }
        None
    }

    fn has_storage_class(node: Node, source: &[u8], specifier: &str) -> bool {
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            if child.kind() == "storage_class_specifier" && node_text(child, source) == specifier {
                return true;
            }
        }
        false
    }

    fn extract_calls(body: Node, source: &[u8]) -> Vec<(String, u32)> {
        let mut calls: Vec<(String, u32)> = Vec::new();
        fn walk(node: Node, source: &[u8], out: &mut Vec<(String, u32)>) {
            if node.kind() == "call_expression" {
                let line = node.start_position().row as u32 + 1;
                if let Some(func) = node.child(0) {
                    match func.kind() {
                        "identifier" => {
                            out.push((node_text(func, source).to_string(), line));
                        }
                        "field_expression" => {
                            let field = func.child_by_field_name("field");
                            let argument = func.child_by_field_name("argument");
                            match (field, argument) {
                                (Some(f), Some(a)) => {
                                    let field_name = node_text(f, source);
                                    let arg_text = node_text(a, source);
                                    let hint = arg_text
                                        .rsplit('.')
                                        .next()
                                        .and_then(|p| p.rsplit("->").next())
                                        .unwrap_or(arg_text);
                                    out.push((format!("{}.{}", hint, field_name), line));
                                }
                                (Some(f), None) => {
                                    out.push((node_text(f, source).to_string(), line));
                                }
                                _ => {}
                            }
                        }
                        "scoped_identifier" => {
                            let text = node_text(func, source);
                            let parts: Vec<&str> = text.split("::").collect();
                            if parts.len() >= 2 {
                                out.push((
                                    format!(
                                        "{}.{}",
                                        parts[parts.len() - 2],
                                        parts[parts.len() - 1]
                                    ),
                                    line,
                                ));
                            } else if let Some(last) = parts.last() {
                                out.push(((*last).to_string(), line));
                            }
                        }
                        _ => {}
                    }
                }
            }
            let mut cursor = node.walk();
            for child in node.children(&mut cursor) {
                if !NESTED_SCOPES.contains(&child.kind()) {
                    walk(child, source, out);
                }
            }
        }
        walk(body, source, &mut calls);
        calls
    }

    fn file_to_module_path(filepath: &Path, src_root: &Path) -> String {
        let rel = filepath.strip_prefix(src_root).unwrap_or(filepath);
        let mut parts: Vec<String> = rel
            .components()
            .filter_map(|c| c.as_os_str().to_str().map(str::to_string))
            .collect();
        if let Some(last) = parts.last_mut() {
            for ext in [".hpp", ".cpp", ".cxx", ".hxx", ".hh", ".cc", ".h", ".c"] {
                if let Some(stem) = last.strip_suffix(ext) {
                    *last = stem.to_string();
                    break;
                }
            }
        }
        parts.join("/")
    }

    fn extract_struct_fields(
        node: Node,
        source: &[u8],
        owner_qname: &str,
        rel_path: &str,
    ) -> Vec<AttributeInfo> {
        let mut out = Vec::new();
        let sep = if owner_qname.contains("::") {
            "::"
        } else {
            "."
        };
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            if child.kind() != "field_declaration_list" {
                continue;
            }
            let mut fc = child.walk();
            for field in child.children(&mut fc) {
                if field.kind() != "field_declaration" {
                    continue;
                }
                let mut type_ann: Option<String> = None;
                let mut names: Vec<String> = Vec::new();
                let mut inner = field.walk();
                for fc2 in field.children(&mut inner) {
                    if TYPE_NODES.contains(&fc2.kind()) {
                        type_ann = Some(node_text(fc2, source).to_string());
                    } else if fc2.kind() == "field_identifier" {
                        names.push(node_text(fc2, source).to_string());
                    } else if matches!(fc2.kind(), "pointer_declarator" | "array_declarator") {
                        let mut ic = fc2.walk();
                        for sub in fc2.children(&mut ic) {
                            if sub.kind() == "field_identifier" {
                                names.push(node_text(sub, source).to_string());
                                break;
                            }
                        }
                    }
                }
                for name in names {
                    out.push(AttributeInfo {
                        qualified_name: format!("{}{}{}", owner_qname, sep, name),
                        owner_qualified_name: owner_qname.to_string(),
                        type_annotation: type_ann.clone(),
                        visibility: "public".into(),
                        file_path: rel_path.to_string(),
                        line_number: field.start_position().row as u32 + 1,
                        default_value: None,
                        name,
                    });
                }
            }
        }
        out
    }

    /// Walk a node tree (typically a `parenthesized_declarator`) looking for
    /// a buried `function_declarator`. Used to unwrap macro-decorated function
    /// definitions like `JSON_HEDLEY_NON_NULL(3) Foo(int x)`, where tree-sitter-cpp
    /// puts the real declarator inside `parenthesized_declarator > ERROR >
    /// function_declarator`.
    fn find_buried_function_declarator(node: Node) -> Option<Node> {
        // Iterative DFS — avoid blowing the stack on weird ASTs.
        let mut stack = vec![node];
        while let Some(n) = stack.pop() {
            if n.kind() == "function_declarator" {
                return Some(n);
            }
            let mut cursor = n.walk();
            for c in n.children(&mut cursor) {
                stack.push(c);
            }
        }
        None
    }

    fn get_enum_variants(node: Node, source: &[u8]) -> Vec<String> {
        let mut out = Vec::new();
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            if child.kind() == "enumerator_list" {
                let mut ec = child.walk();
                for sub in child.children(&mut ec) {
                    if sub.kind() == "enumerator" {
                        if let Some(n) = Self::get_name(sub, source, "identifier") {
                            out.push(n.to_string());
                        }
                    }
                }
            }
        }
        out
    }

    fn parse_function(
        &self,
        node: Node,
        source: &[u8],
        module_path: &str,
        rel_path: &str,
        is_method: bool,
        owner: Option<&str>,
    ) -> FunctionInfo {
        let mut name = "unknown".to_string();
        let mut declarator: Option<Node> = None;
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            if matches!(
                child.kind(),
                "function_declarator" | "pointer_declarator" | "reference_declarator"
            ) {
                declarator = Some(child);
                break;
            }
            // Macro-decorated constructors like `JSON_HEDLEY_NON_NULL(3) Foo(...)`
            // produce a `parenthesized_declarator` containing the macro args (3)
            // followed by an ERROR wrapping the real `function_declarator`.
            // Walk through the wrapper to find the buried declarator.
            if child.kind() == "parenthesized_declarator" {
                if let Some(buried) = Self::find_buried_function_declarator(child) {
                    declarator = Some(buried);
                    break;
                }
            }
        }
        // Unwrap pointer_declarator and reference_declarator wrappers — both
        // appear when the return type is `T *` or `T &`. The real
        // function_declarator lives inside.
        while let Some(d) = declarator {
            if matches!(d.kind(), "pointer_declarator" | "reference_declarator") {
                let mut dc = d.walk();
                let mut found: Option<Node> = None;
                for c in d.children(&mut dc) {
                    if c.kind() == "function_declarator" {
                        found = Some(c);
                        break;
                    }
                }
                if let Some(f) = found {
                    declarator = Some(f);
                } else {
                    break;
                }
            } else {
                break;
            }
        }
        if let Some(d) = declarator {
            if let Some(fn_name) = Self::get_name(d, source, "identifier") {
                name = fn_name.to_string();
            }
        }
        let sep = self.sep();
        let prefix = match owner {
            Some(o) => format!("{}{}{}", module_path, sep, o),
            None => module_path.to_string(),
        };
        let qualified_name = format!("{}{}{}", prefix, sep, name);

        let mut body: Option<Node> = None;
        let mut cursor2 = node.walk();
        for child in node.children(&mut cursor2) {
            if child.kind() == "compound_statement" {
                body = Some(child);
                break;
            }
        }

        let is_static = Self::has_storage_class(node, source, "static");
        let visibility = if is_static { "private" } else { "public" };
        let calls = body
            .map(|b| Self::extract_calls(b, source))
            .unwrap_or_default();
        let parameters = Self::extract_parameters(declarator, source);
        let param_count = Some(
            parameters
                .iter()
                .filter(|p| p.kind != ParameterKind::Receiver)
                .count() as u32,
        );
        let (branch_count, max_nesting) = match body {
            Some(b) => {
                let (c, n) = compute_complexity(b, BRANCH_KINDS_CPP, NESTED_SCOPES);
                (Some(c), Some(n))
            }
            None => (None, None),
        };
        let is_recursive = Some(calls.iter().any(|(n, _)| n == &name));
        let docstring = Self::get_doc_comment(node, source);
        let procedure_names = extract_procedure_annotations(docstring.as_deref());

        FunctionInfo {
            visibility: visibility.into(),
            is_async: false,
            is_method,
            signature: Self::get_signature(node, source),
            file_path: rel_path.to_string(),
            line_number: node.start_position().row as u32 + 1,
            end_line: Some(node.end_position().row as u32 + 1),
            docstring,
            return_type: Self::get_return_type(node, source),
            decorators: Vec::new(),
            calls,
            references: Vec::new(),
            function_refs: Vec::new(),
            type_parameters: None,
            parameters,
            branch_count,
            param_count,
            max_nesting,
            is_recursive,
            procedure_names,
            metadata: Default::default(),
            qualified_name,
            name,
        }
    }

    /// Extract structured parameters from a C/C++ function definition.
    ///
    /// `declarator` is the resolved `function_declarator` (or None for malformed
    /// input). The parameters live in a `parameter_list` child of the declarator.
    /// tree-sitter-cpp wraps each parameter in a `parameter_declaration`.
    fn extract_parameters(declarator: Option<Node>, source: &[u8]) -> Vec<ParameterInfo> {
        let mut out = Vec::new();
        let Some(declarator) = declarator else {
            return out;
        };
        let mut cursor = declarator.walk();
        let Some(params_node) = declarator
            .children(&mut cursor)
            .find(|c| c.kind() == "parameter_list")
        else {
            return out;
        };
        let mut pcursor = params_node.walk();
        for child in params_node.children(&mut pcursor) {
            match child.kind() {
                "parameter_declaration" | "optional_parameter_declaration" => {
                    let mut name: Option<String> = None;
                    let mut type_ann: Option<String> = None;
                    let mut tcursor = child.walk();
                    for sub in child.children(&mut tcursor) {
                        let k = sub.kind();
                        if k == "identifier" && name.is_none() {
                            name = Some(node_text(sub, source).to_string());
                        } else if k == "pointer_declarator" || k == "reference_declarator" {
                            // Drill into declarator to find the bound name.
                            let mut dc = sub.walk();
                            for ds in sub.children(&mut dc) {
                                if ds.kind() == "identifier" {
                                    name = Some(node_text(ds, source).to_string());
                                    break;
                                }
                            }
                        } else if k.contains("type")
                            || k == "primitive_type"
                            || k == "qualified_identifier"
                            || k == "struct_specifier"
                            || k == "enum_specifier"
                            || k == "union_specifier"
                            || k == "class_specifier"
                            || k == "sized_type_specifier"
                        {
                            // C-style `struct llama_grammar *` parameters land
                            // here as struct_specifier (not in TYPE_NODES). Same
                            // for `enum X` / `union X` / `class X` parameter types.
                            type_ann = Some(node_text(sub, source).to_string());
                        }
                    }
                    let n = name.unwrap_or_default();
                    if n.is_empty() && type_ann.is_none() {
                        continue;
                    }
                    out.push(ParameterInfo {
                        name: if n.is_empty() { "_".into() } else { n },
                        type_annotation: type_ann,
                        default: None,
                        kind: ParameterKind::Positional,
                    });
                }
                "variadic_parameter" => {
                    out.push(ParameterInfo {
                        name: "...".into(),
                        type_annotation: None,
                        default: None,
                        kind: ParameterKind::Variadic,
                    });
                }
                _ => {}
            }
        }
        out
    }

    fn parse_struct(
        &self,
        node: Node,
        source: &[u8],
        module_path: &str,
        rel_path: &str,
        result: &mut ParseResult,
    ) {
        let Some(name) = Self::get_name(node, source, "type_identifier") else {
            return;
        };
        let name = name.to_string();
        let sep = self.sep();
        let qname = format!("{}{}{}", module_path, sep, name);
        result.classes.push(ClassInfo {
            qualified_name: qname.clone(),
            kind: "struct".into(),
            visibility: "public".into(),
            file_path: rel_path.to_string(),
            line_number: node.start_position().row as u32 + 1,
            end_line: Some(node.end_position().row as u32 + 1),
            docstring: Self::get_doc_comment(node, source),
            bases: Vec::new(),
            type_parameters: None,
            metadata: Default::default(),
            name: name.clone(),
        });
        result
            .attributes
            .extend(Self::extract_struct_fields(node, source, &qname, rel_path));
    }

    fn parse_enum(
        &self,
        node: Node,
        source: &[u8],
        module_path: &str,
        rel_path: &str,
        result: &mut ParseResult,
    ) {
        let Some(name) = Self::get_name(node, source, "type_identifier") else {
            return;
        };
        let name = name.to_string();
        let sep = self.sep();
        result.enums.push(EnumInfo {
            qualified_name: format!("{}{}{}", module_path, sep, name),
            visibility: "public".into(),
            file_path: rel_path.to_string(),
            line_number: node.start_position().row as u32 + 1,
            end_line: Some(node.end_position().row as u32 + 1),
            docstring: Self::get_doc_comment(node, source),
            variants: Self::get_enum_variants(node, source),
            variant_details: None,
            name,
        });
    }

    fn parse_typedef(
        &self,
        node: Node,
        source: &[u8],
        module_path: &str,
        rel_path: &str,
        result: &mut ParseResult,
    ) {
        let sep = self.sep();
        let mut typedef_name: Option<String> = None;
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            if child.kind() == "type_identifier" {
                typedef_name = Some(node_text(child, source).to_string());
            }
        }
        let mut cursor2 = node.walk();
        for child in node.children(&mut cursor2) {
            match child.kind() {
                "struct_specifier" => {
                    let struct_name =
                        Self::get_name(child, source, "type_identifier").map(str::to_string);
                    let name = struct_name.or_else(|| typedef_name.clone());
                    let Some(name) = name else {
                        continue;
                    };
                    let qname = format!("{}{}{}", module_path, sep, name);
                    result.classes.push(ClassInfo {
                        qualified_name: qname.clone(),
                        kind: "struct".into(),
                        visibility: "public".into(),
                        file_path: rel_path.to_string(),
                        line_number: node.start_position().row as u32 + 1,
                        end_line: Some(node.end_position().row as u32 + 1),
                        docstring: Self::get_doc_comment(node, source),
                        bases: Vec::new(),
                        type_parameters: None,
                        metadata: Default::default(),
                        name,
                    });
                    result
                        .attributes
                        .extend(Self::extract_struct_fields(child, source, &qname, rel_path));
                    return;
                }
                "enum_specifier" => {
                    let enum_name =
                        Self::get_name(child, source, "type_identifier").map(str::to_string);
                    let name = enum_name.or_else(|| typedef_name.clone());
                    let Some(name) = name else { continue };
                    result.enums.push(EnumInfo {
                        qualified_name: format!("{}{}{}", module_path, sep, name),
                        visibility: "public".into(),
                        file_path: rel_path.to_string(),
                        line_number: node.start_position().row as u32 + 1,
                        end_line: Some(node.end_position().row as u32 + 1),
                        docstring: Self::get_doc_comment(node, source),
                        variants: Self::get_enum_variants(child, source),
                        variant_details: None,
                        name,
                    });
                    return;
                }
                _ => {}
            }
        }
        let Some(name) = typedef_name else { return };
        result.constants.push(ConstantInfo {
            qualified_name: format!("{}{}{}", module_path, sep, name),
            kind: "type_alias".into(),
            type_annotation: None,
            value_preview: Some({
                let sig = Self::get_signature(node, source);
                let take = sig
                    .char_indices()
                    .nth(100)
                    .map(|(i, _)| i)
                    .unwrap_or(sig.len());
                sig[..take].to_string()
            }),
            visibility: "public".into(),
            file_path: rel_path.to_string(),
            line_number: node.start_position().row as u32 + 1,
            name,
        });
    }

    fn parse_preproc_include(node: Node, source: &[u8], file_info: &mut FileInfo) {
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            if matches!(child.kind(), "string_literal" | "system_lib_string") {
                let text = node_text(child, source);
                let trimmed = text.trim_matches(|c| c == '"' || c == '<' || c == '>');
                file_info.imports.push(trimmed.to_string());
            }
        }
    }

    fn parse_preproc_def(
        &self,
        node: Node,
        source: &[u8],
        module_path: &str,
        rel_path: &str,
        result: &mut ParseResult,
    ) {
        let Some(name) = Self::get_name(node, source, "identifier") else {
            return;
        };
        let name = name.to_string();
        let mut val_text: Option<String> = None;
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            if child.kind() == "preproc_arg" {
                let text = node_text(child, source).trim();
                let take = text
                    .char_indices()
                    .nth(100)
                    .map(|(i, _)| i)
                    .unwrap_or(text.len());
                val_text = Some(text[..take].to_string());
            }
        }
        let sep = self.sep();
        result.constants.push(ConstantInfo {
            qualified_name: format!("{}{}{}", module_path, sep, name),
            kind: "constant".into(),
            type_annotation: None,
            value_preview: val_text,
            visibility: "public".into(),
            file_path: rel_path.to_string(),
            line_number: node.start_position().row as u32 + 1,
            name,
        });
    }

    // ── C++-only helpers ───────────────────────────────────────────

    fn get_base_classes(node: Node, source: &[u8]) -> Vec<String> {
        let mut out = Vec::new();
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            if child.kind() != "base_class_clause" {
                continue;
            }
            let mut bc = child.walk();
            for sub in child.children(&mut bc) {
                match sub.kind() {
                    "type_identifier" | "qualified_identifier" => {
                        out.push(node_text(sub, source).to_string());
                    }
                    "template_type" => {
                        let mut tc = sub.walk();
                        for inner in sub.children(&mut tc) {
                            if inner.kind() == "type_identifier" {
                                out.push(node_text(inner, source).to_string());
                                break;
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
        out
    }

    fn get_access_specifier_text<'a>(node: Node<'a>, source: &'a [u8]) -> Option<&'a str> {
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            if matches!(child.kind(), "public" | "private" | "protected") {
                return Some(node_text(child, source));
            }
        }
        let text = node_text(node, source).trim_end_matches(':');
        if matches!(text, "public" | "private" | "protected") {
            Some(text)
        } else {
            None
        }
    }

    fn parse_class_specifier(
        &self,
        node: Node,
        source: &[u8],
        module_path: &str,
        rel_path: &str,
        result: &mut ParseResult,
        is_struct: bool,
    ) {
        let Some(name) = Self::get_name(node, source, "type_identifier") else {
            return;
        };
        let name = name.to_string();
        let qname = format!("{}::{}", module_path, name);
        let bases = Self::get_base_classes(node, source);
        let docstring = Self::get_doc_comment(node, source);
        let kind = if is_struct { "struct" } else { "class" };

        result.classes.push(ClassInfo {
            qualified_name: qname.clone(),
            kind: kind.into(),
            visibility: "public".into(),
            file_path: rel_path.to_string(),
            line_number: node.start_position().row as u32 + 1,
            end_line: Some(node.end_position().row as u32 + 1),
            docstring,
            bases: bases.clone(),
            type_parameters: None,
            metadata: Default::default(),
            name: name.clone(),
        });
        for base in &bases {
            result.type_relationships.push(TypeRelationship {
                source_type: name.clone(),
                target_type: Some(base.clone()),
                relationship: "extends".into(),
                methods: Vec::new(),
            });
        }
        let default_vis = if is_struct { "public" } else { "private" };
        self.parse_class_body(
            node,
            source,
            module_path,
            rel_path,
            &name,
            &qname,
            result,
            default_vis,
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn parse_class_body(
        &self,
        node: Node,
        source: &[u8],
        module_path: &str,
        rel_path: &str,
        class_name: &str,
        class_qname: &str,
        result: &mut ParseResult,
        default_vis: &str,
    ) {
        let mut method_rel = TypeRelationship {
            source_type: class_qname.to_string(),
            target_type: None,
            relationship: "inherent".into(),
            methods: Vec::new(),
        };
        let mut current_vis = default_vis.to_string();
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            if child.kind() != "field_declaration_list" {
                continue;
            }
            let mut bc = child.walk();
            for item in child.children(&mut bc) {
                match item.kind() {
                    "access_specifier" => {
                        if let Some(spec) = Self::get_access_specifier_text(item, source) {
                            current_vis = spec.to_string();
                        }
                    }
                    "function_definition" => {
                        let mut fn_info = self.parse_function(
                            item,
                            source,
                            module_path,
                            rel_path,
                            true,
                            Some(class_name),
                        );
                        fn_info.visibility = current_vis.clone();
                        method_rel.methods.push(fn_info.clone());
                        result.functions.push(fn_info);
                    }
                    "declaration" => {
                        let mut ic = item.walk();
                        let has_func_decl = item
                            .children(&mut ic)
                            .any(|c| c.kind() == "function_declarator");
                        if has_func_decl {
                            let mut fn_info = self.parse_function(
                                item,
                                source,
                                module_path,
                                rel_path,
                                true,
                                Some(class_name),
                            );
                            fn_info.visibility = current_vis.clone();
                            method_rel.methods.push(fn_info.clone());
                            result.functions.push(fn_info);
                        } else {
                            self.parse_cpp_field(
                                item,
                                source,
                                rel_path,
                                class_qname,
                                &current_vis,
                                result,
                            );
                        }
                    }
                    "field_declaration" => {
                        // Methods with template return types like `proxy<T> items()`
                        // come through as `field_declaration` containing a
                        // `function_declarator`. Route those to parse_function;
                        // genuine fields (no function_declarator) go to parse_cpp_field.
                        let mut ic = item.walk();
                        let has_func_decl = item
                            .children(&mut ic)
                            .any(|c| c.kind() == "function_declarator");
                        if has_func_decl {
                            let mut fn_info = self.parse_function(
                                item,
                                source,
                                module_path,
                                rel_path,
                                true,
                                Some(class_name),
                            );
                            fn_info.visibility = current_vis.clone();
                            method_rel.methods.push(fn_info.clone());
                            result.functions.push(fn_info);
                        } else {
                            self.parse_cpp_field(
                                item,
                                source,
                                rel_path,
                                class_qname,
                                &current_vis,
                                result,
                            );
                        }
                    }
                    "template_declaration" => {
                        let tparams = get_type_parameters(item, source, "template_parameter_list");
                        let mut tc = item.walk();
                        for sub in item.children(&mut tc) {
                            if sub.kind() == "function_definition" {
                                let mut fn_info = self.parse_function(
                                    sub,
                                    source,
                                    module_path,
                                    rel_path,
                                    true,
                                    Some(class_name),
                                );
                                fn_info.visibility = current_vis.clone();
                                fn_info.metadata.insert("is_template".into(), json!(true));
                                fn_info.type_parameters = tparams.clone();
                                method_rel.methods.push(fn_info.clone());
                                result.functions.push(fn_info);
                            }
                        }
                    }
                    "class_specifier" | "struct_specifier" => {
                        let is_struct = item.kind() == "struct_specifier";
                        let nested_module = format!("{}::{}", module_path, class_name);
                        self.parse_class_specifier(
                            item,
                            source,
                            &nested_module,
                            rel_path,
                            result,
                            is_struct,
                        );
                    }
                    _ => {}
                }
            }
        }
        if !method_rel.methods.is_empty() {
            result.type_relationships.push(method_rel);
        }
    }

    fn parse_cpp_field(
        &self,
        node: Node,
        source: &[u8],
        rel_path: &str,
        class_qname: &str,
        visibility: &str,
        result: &mut ParseResult,
    ) {
        let mut type_ann: Option<String> = None;
        let mut names: Vec<String> = Vec::new();
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            match child.kind() {
                "primitive_type"
                | "type_identifier"
                | "sized_type_specifier"
                | "template_type"
                | "qualified_identifier" => {
                    if type_ann.is_none() {
                        type_ann = Some(node_text(child, source).to_string());
                    }
                }
                "field_identifier" => names.push(node_text(child, source).to_string()),
                "init_declarator" | "pointer_declarator" | "reference_declarator" => {
                    let mut ic = child.walk();
                    for sub in child.children(&mut ic) {
                        if matches!(sub.kind(), "field_identifier" | "identifier") {
                            names.push(node_text(sub, source).to_string());
                            break;
                        }
                    }
                }
                _ => {}
            }
        }
        for name in names {
            result.attributes.push(AttributeInfo {
                qualified_name: format!("{}::{}", class_qname, name),
                owner_qualified_name: class_qname.to_string(),
                type_annotation: type_ann.clone(),
                visibility: visibility.to_string(),
                file_path: rel_path.to_string(),
                line_number: node.start_position().row as u32 + 1,
                default_value: None,
                name,
            });
        }
    }

    fn parse_namespace(
        &self,
        node: Node,
        source: &[u8],
        parent_path: &str,
        rel_path: &str,
        result: &mut ParseResult,
        file_info: &mut FileInfo,
    ) {
        let ns_name = Self::get_name(node, source, "identifier").map(str::to_string);
        let ns_path = match &ns_name {
            Some(n) => {
                file_info.submodule_declarations.push(n.clone());
                format!("{}::{}", parent_path, n)
            }
            None => parent_path.to_string(),
        };
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            if child.kind() == "declaration_list" {
                let mut dc = child.walk();
                for item in child.children(&mut dc) {
                    self.parse_cpp_top_level(item, source, &ns_path, rel_path, result, file_info);
                }
            }
        }
    }

    fn parse_cpp_top_level(
        &self,
        node: Node,
        source: &[u8],
        module_path: &str,
        rel_path: &str,
        result: &mut ParseResult,
        file_info: &mut FileInfo,
    ) {
        match node.kind() {
            "function_definition" => {
                result.functions.push(self.parse_function(
                    node,
                    source,
                    module_path,
                    rel_path,
                    false,
                    None,
                ));
            }
            "class_specifier" => {
                self.parse_class_specifier(node, source, module_path, rel_path, result, false);
            }
            "struct_specifier" => {
                self.parse_class_specifier(node, source, module_path, rel_path, result, true);
            }
            "declaration" => {
                let mut cursor = node.walk();
                let mut handled = false;
                for sub in node.children(&mut cursor) {
                    match sub.kind() {
                        "class_specifier" => {
                            self.parse_class_specifier(
                                sub,
                                source,
                                module_path,
                                rel_path,
                                result,
                                false,
                            );
                            handled = true;
                            break;
                        }
                        "struct_specifier" => {
                            self.parse_class_specifier(
                                sub,
                                source,
                                module_path,
                                rel_path,
                                result,
                                true,
                            );
                            handled = true;
                            break;
                        }
                        "enum_specifier" => {
                            self.parse_enum(sub, source, module_path, rel_path, result);
                            handled = true;
                            break;
                        }
                        _ => {}
                    }
                }
                if !handled {
                    let mut cursor2 = node.walk();
                    if node
                        .children(&mut cursor2)
                        .any(|c| c.kind() == "function_declarator")
                    {
                        result.functions.push(self.parse_function(
                            node,
                            source,
                            module_path,
                            rel_path,
                            false,
                            None,
                        ));
                    }
                }
            }
            "enum_specifier" => self.parse_enum(node, source, module_path, rel_path, result),
            "namespace_definition" => {
                self.parse_namespace(node, source, module_path, rel_path, result, file_info)
            }
            "template_declaration" => {
                let tparams = get_type_parameters(node, source, "template_parameter_list");
                let n_funcs = result.functions.len();
                let n_classes = result.classes.len();
                let mut cursor = node.walk();
                for sub in node.children(&mut cursor) {
                    if matches!(
                        sub.kind(),
                        "function_definition"
                            | "class_specifier"
                            | "struct_specifier"
                            | "declaration"
                    ) {
                        self.parse_cpp_top_level(
                            sub,
                            source,
                            module_path,
                            rel_path,
                            result,
                            file_info,
                        );
                    }
                }
                if let Some(tp) = tparams {
                    for fn_info in &mut result.functions[n_funcs..] {
                        fn_info.metadata.insert("is_template".into(), json!(true));
                        fn_info.type_parameters = Some(tp.clone());
                    }
                    for cls in &mut result.classes[n_classes..] {
                        cls.type_parameters = Some(tp.clone());
                    }
                }
            }
            "type_definition" => self.parse_typedef(node, source, module_path, rel_path, result),
            "preproc_include" => Self::parse_preproc_include(node, source, file_info),
            "preproc_def" => self.parse_preproc_def(node, source, module_path, rel_path, result),
            "linkage_specification" => {
                let n_funcs = result.functions.len();
                let mut cursor = node.walk();
                for sub in node.children(&mut cursor) {
                    match sub.kind() {
                        "declaration_list" => {
                            let mut dc = sub.walk();
                            for item in sub.children(&mut dc) {
                                self.parse_cpp_top_level(
                                    item,
                                    source,
                                    module_path,
                                    rel_path,
                                    result,
                                    file_info,
                                );
                            }
                        }
                        "function_definition" | "declaration" => {
                            self.parse_cpp_top_level(
                                sub,
                                source,
                                module_path,
                                rel_path,
                                result,
                                file_info,
                            );
                        }
                        _ => {}
                    }
                }
                for fn_info in &mut result.functions[n_funcs..] {
                    fn_info.metadata.insert("is_ffi".into(), json!(true));
                    fn_info
                        .metadata
                        .insert("ffi_kind".into(), json!("extern_c"));
                }
            }
            "using_declaration" => {
                let mut cursor = node.walk();
                for sub in node.children(&mut cursor) {
                    if matches!(sub.kind(), "scoped_identifier" | "qualified_identifier") {
                        file_info.imports.push(node_text(sub, source).to_string());
                        break;
                    }
                }
            }
            _ => {}
        }
    }

    fn parse_c_top_level(
        &self,
        node: Node,
        source: &[u8],
        module_path: &str,
        rel_path: &str,
        result: &mut ParseResult,
        file_info: &mut FileInfo,
    ) {
        match node.kind() {
            "function_definition" => {
                result.functions.push(self.parse_function(
                    node,
                    source,
                    module_path,
                    rel_path,
                    false,
                    None,
                ));
            }
            "declaration" => {
                let mut cursor = node.walk();
                let mut handled = false;
                for sub in node.children(&mut cursor) {
                    match sub.kind() {
                        "struct_specifier" => {
                            self.parse_struct(sub, source, module_path, rel_path, result);
                            handled = true;
                        }
                        "enum_specifier" => {
                            self.parse_enum(sub, source, module_path, rel_path, result);
                            handled = true;
                        }
                        _ => {}
                    }
                }
                let _ = handled;
            }
            "type_definition" => self.parse_typedef(node, source, module_path, rel_path, result),
            "preproc_include" => Self::parse_preproc_include(node, source, file_info),
            "preproc_def" => self.parse_preproc_def(node, source, module_path, rel_path, result),
            _ => {}
        }
    }
}

impl LanguageParser for CppParser {
    fn language_name(&self) -> &'static str {
        if self.is_cpp() {
            "cpp"
        } else {
            "c"
        }
    }
    fn file_extensions(&self) -> &'static [&'static str] {
        if self.is_cpp() {
            &["cpp", "cc", "cxx", "hpp", "hh", "hxx"]
        } else {
            &["c", "h"]
        }
    }
    fn noise_names(&self) -> &'static [&'static str] {
        if self.is_cpp() {
            CPP_NOISE_NAMES
        } else {
            C_NOISE_NAMES
        }
    }

    fn parse_file(&self, filepath: &Path, src_root: &Path) -> ParseResult {
        let Ok(source) = std::fs::read(filepath) else {
            return ParseResult::new();
        };

        let rel_path = filepath
            .strip_prefix(src_root)
            .unwrap_or(filepath)
            .to_string_lossy()
            .replace('\\', "/");
        let module_path = Self::file_to_module_path(filepath, src_root);
        let loc = count_lines(&source);
        let filename = filepath
            .file_name()
            .and_then(|o| o.to_str())
            .unwrap_or("")
            .to_string();
        let stem = filepath
            .file_stem()
            .and_then(|o| o.to_str())
            .unwrap_or("")
            .to_string();
        let is_test = stem.ends_with("_test")
            || stem.ends_with("_tests")
            || stem.starts_with("test_")
            || rel_path.contains("/test/")
            || rel_path.contains("/tests/")
            || rel_path.starts_with("test/")
            || rel_path.starts_with("tests/");

        let language = if self.is_cpp() { "cpp" } else { "c" };

        if let Some(reason) = is_generated_or_minified(&source) {
            let mut r = ParseResult::new();
            r.files.push(FileInfo {
                path: rel_path,
                filename,
                loc,
                module_path,
                language: language.to_string(),
                submodule_declarations: Vec::new(),
                imports: Vec::new(),
                exports: Vec::new(),
                annotations: None,
                is_test,
                skip_reason: Some(reason.to_string()),
            });
            return r;
        }

        let Some(tree) = self.parse_tree(&source) else {
            return ParseResult::new();
        };
        let root = tree.root_node();
        let mut file_info = FileInfo {
            path: rel_path.clone(),
            filename,
            loc,
            module_path: module_path.clone(),
            language: language.to_string(),
            submodule_declarations: Vec::new(),
            imports: Vec::new(),
            exports: Vec::new(),
            annotations: None,
            is_test,
            skip_reason: None,
        };
        let mut result = ParseResult::new();
        let mut cursor = root.walk();
        if self.is_cpp() {
            for child in root.children(&mut cursor) {
                self.parse_cpp_top_level(
                    child,
                    &source,
                    &module_path,
                    &rel_path,
                    &mut result,
                    &mut file_info,
                );
            }
        } else {
            for child in root.children(&mut cursor) {
                self.parse_c_top_level(
                    child,
                    &source,
                    &module_path,
                    &rel_path,
                    &mut result,
                    &mut file_info,
                );
            }
        }
        file_info.annotations = extract_comment_annotations(root, &source, DEFAULT_COMMENT_TYPES);
        result.files.push(file_info);
        result
    }
}
