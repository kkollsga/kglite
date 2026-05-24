//! PHP language parser.
//!
//! Coverage in 0.9.36:
//!   - `class` / `interface` / `trait` declarations → ClassInfo /
//!     InterfaceInfo (trait → ClassInfo kind="trait", matching the
//!     Rust-trait encoding).
//!   - Top-level `function` definitions and class `method` declarations
//!     → FunctionInfo.
//!   - `const` declarations (top-level + class-level) → ConstantInfo
//!     per const_element.
//!   - `namespace` declaration → FileInfo.module_path (backslash
//!     separator).
//!   - `use` declarations → FileInfo.imports.
//!   - PHP 8 attributes (`#[Route('/x')]`) → FunctionInfo.decorators
//!     so the existing 0.9.34 DECORATES pass picks them up.
//!   - Visibility modifiers (`public` / `protected` / `private`).
//!   - `static`, `final`, `abstract` modifiers as metadata.
//!
//! Not yet supported (follow-up scope):
//!   - `define('NAME', value)` constants — these are function calls,
//!     not declaration nodes, and need a separate post-pass that
//!     walks the call graph.
//!   - PHP fibers (`Fiber::start`) async detection. v1 marks every
//!     PHP function `is_async=false`.
//!   - Property declarations as AttributeInfo. The grammar exposes
//!     them but we don't currently model PHP class properties; the
//!     same applies to constructor property promotion.

use std::path::Path;
use tree_sitter::{Node, Parser, Tree};

use super::shared::{file_to_module_path, make_qualified, node_text};
use super::LanguageParser;
use crate::code_tree::models::{
    ClassInfo, ConstantInfo, FileInfo, FunctionInfo, InterfaceInfo, ParseResult, TypeRelationship,
};

/// PHP standard-library / language built-in names excluded from CALLS
/// resolution. The list is small on purpose — the call resolver's
/// 5-tier name lookup already disambiguates user-defined identifiers
/// well; we only need to swallow the truly ubiquitous names that would
/// otherwise generate edges to every same-name user function.
pub const PHP_NOISE_NAMES: &[&str] = &[
    "array",
    "count",
    "strlen",
    "isset",
    "empty",
    "unset",
    "print",
    "echo",
    "var_dump",
    "print_r",
    "gettype",
    "is_array",
    "is_string",
    "is_int",
    "is_bool",
    "is_null",
    "is_object",
    "is_callable",
    "trim",
    "explode",
    "implode",
    "str_replace",
    "preg_match",
    "json_encode",
    "json_decode",
    "in_array",
    "array_keys",
    "array_values",
    "array_map",
    "array_filter",
    "array_merge",
    "sprintf",
    "printf",
    "fopen",
    "fclose",
    "fread",
    "fwrite",
    "file_get_contents",
    "file_put_contents",
];

pub struct PhpParser;

thread_local! {
    static TS_PARSER: std::cell::RefCell<Parser> = {
        let mut p = Parser::new();
        p.set_language(&tree_sitter_php::LANGUAGE_PHP.into())
            .expect("loading tree-sitter-php grammar");
        std::cell::RefCell::new(p)
    };
}

impl PhpParser {
    pub fn new() -> Self {
        PhpParser
    }

    fn parse_tree(&self, source: &[u8]) -> Option<Tree> {
        TS_PARSER.with(|p| p.borrow_mut().parse(source, None))
    }

    /// Walk a program/namespace body and dispatch declarations.
    fn parse_block(
        block: Node,
        source: &[u8],
        module_path: &str,
        owner_prefix: &str,
        rel_path: &str,
        result: &mut ParseResult,
        file_info: &mut FileInfo,
    ) {
        let mut cursor = block.walk();
        for child in block.named_children(&mut cursor) {
            match child.kind() {
                "namespace_definition" => {
                    // Nested namespace block — recurse with the
                    // namespace-qualified module path.
                    let ns_name = child
                        .child_by_field_name("name")
                        .map(|n| node_text(n, source).to_string())
                        .unwrap_or_default();
                    let nested_module = if ns_name.is_empty() {
                        module_path.to_string()
                    } else if module_path.is_empty() {
                        ns_name
                    } else {
                        format!("{module_path}\\{ns_name}")
                    };
                    if let Some(body) = child.child_by_field_name("body") {
                        Self::parse_block(
                            body,
                            source,
                            &nested_module,
                            owner_prefix,
                            rel_path,
                            result,
                            file_info,
                        );
                    } else {
                        // `namespace Foo;` form (no body) — the rest of
                        // the file lives under this namespace. The
                        // top-level parse_file sets file_info.module_path
                        // when it sees this; nothing more to do here.
                    }
                }
                "namespace_use_declaration" => {
                    Self::extract_use_imports(child, source, file_info);
                }
                "class_declaration" => {
                    Self::parse_class(
                        child,
                        source,
                        module_path,
                        owner_prefix,
                        rel_path,
                        result,
                        "class",
                    );
                }
                "interface_declaration" => {
                    Self::parse_interface(
                        child,
                        source,
                        module_path,
                        owner_prefix,
                        rel_path,
                        result,
                    );
                }
                "trait_declaration" => {
                    Self::parse_class(
                        child,
                        source,
                        module_path,
                        owner_prefix,
                        rel_path,
                        result,
                        "trait",
                    );
                }
                "function_definition" => {
                    Self::parse_function(
                        child,
                        source,
                        module_path,
                        owner_prefix,
                        rel_path,
                        result,
                        owner_prefix.is_empty(),
                    );
                }
                "const_declaration" => {
                    Self::parse_const(child, source, module_path, owner_prefix, rel_path, result);
                }
                _ => {}
            }
        }
    }

    /// Extract `use Foo\Bar;` / `use Foo\{Bar, Baz};` declarations.
    fn extract_use_imports(node: Node, source: &[u8], file_info: &mut FileInfo) {
        let mut cursor = node.walk();
        for child in node.named_children(&mut cursor) {
            match child.kind() {
                "namespace_name" | "qualified_name" | "name" => {
                    let text = node_text(child, source).to_string();
                    if !text.is_empty() {
                        file_info.imports.push(text);
                    }
                }
                "namespace_use_clause" => {
                    // `Foo\Bar` or `Foo\Bar as Baz`. The first
                    // named child is the path.
                    let mut sub = child.walk();
                    for c in child.named_children(&mut sub) {
                        if matches!(c.kind(), "namespace_name" | "qualified_name" | "name") {
                            let text = node_text(c, source).to_string();
                            if !text.is_empty() {
                                file_info.imports.push(text);
                            }
                            break;
                        }
                    }
                }
                _ => {}
            }
        }
    }

    /// Extract `#[Attr(args)]` PHP-8 attributes attached to a
    /// declaration. Returns one decorator string per attribute, in
    /// source order. The string includes the parenthesised args when
    /// present so the DECORATES resolver and the (eventual) PHP route
    /// detector can both consume the same shape.
    fn extract_attributes(node: Node, source: &[u8]) -> Vec<String> {
        let mut out = Vec::new();
        let attrs = match node.child_by_field_name("attributes") {
            Some(n) => n,
            None => return out,
        };
        let mut cursor = attrs.walk();
        for group in attrs.named_children(&mut cursor) {
            if group.kind() != "attribute_group" {
                continue;
            }
            let mut sub = group.walk();
            for attr in group.named_children(&mut sub) {
                if attr.kind() != "attribute" {
                    continue;
                }
                // Attribute name is the first named child (a `name` /
                // `qualified_name` / `relative_name`). Args are in the
                // `parameters` field.
                let mut name_cursor = attr.walk();
                let mut head: Option<String> = None;
                for c in attr.named_children(&mut name_cursor) {
                    if matches!(c.kind(), "name" | "qualified_name" | "relative_name") {
                        head = Some(node_text(c, source).to_string());
                        break;
                    }
                }
                let Some(head) = head else { continue };
                let mut raw = head;
                if let Some(params) = attr.child_by_field_name("parameters") {
                    raw.push_str(node_text(params, source));
                }
                out.push(raw);
            }
        }
        out
    }

    /// Visibility from any of `visibility_modifier` / `abstract_modifier`
    /// / `static_modifier` / etc. children. PHP's grammar exposes these
    /// as direct children of the declaration, not in a `modifiers`
    /// wrapper.
    fn extract_visibility(node: Node, source: &[u8]) -> String {
        let mut cursor = node.walk();
        for child in node.named_children(&mut cursor) {
            if child.kind() == "visibility_modifier" {
                return node_text(child, source).to_string();
            }
        }
        "public".to_string()
    }

    fn has_modifier(node: Node, kind: &str) -> bool {
        let mut cursor = node.walk();
        let mut found = false;
        for child in node.named_children(&mut cursor) {
            if child.kind() == kind {
                found = true;
                break;
            }
        }
        found
    }

    fn extract_name<'a>(node: Node<'a>, source: &'a [u8]) -> Option<&'a str> {
        node.child_by_field_name("name")
            .map(|c| node_text(c, source))
    }

    fn extract_body(node: Node) -> Option<Node> {
        node.child_by_field_name("body")
    }

    fn parse_class(
        node: Node,
        source: &[u8],
        module_path: &str,
        owner_prefix: &str,
        rel_path: &str,
        result: &mut ParseResult,
        kind: &str,
    ) {
        let Some(name) = Self::extract_name(node, source) else {
            return;
        };
        let visibility = Self::extract_visibility(node, source);
        let line = node.start_position().row as u32 + 1;
        let end_line = node.end_position().row as u32 + 1;
        let qname = make_qualified(module_path, owner_prefix, name, '\\');

        result.classes.push(ClassInfo {
            qualified_name: qname.clone(),
            visibility: visibility.clone(),
            name: name.to_string(),
            kind: kind.to_string(),
            file_path: rel_path.to_string(),
            line_number: line,
            docstring: None,
            bases: Vec::new(),
            type_parameters: None,
            end_line: Some(end_line),
            metadata: Default::default(),
        });

        // Extract `extends` (`base_clause`) and `implements`
        // (`class_interface_clause`) and emit TypeRelationships.
        let mut cursor = node.walk();
        for child in node.named_children(&mut cursor) {
            match child.kind() {
                "base_clause" => {
                    let mut sub = child.walk();
                    for c in child.named_children(&mut sub) {
                        if matches!(c.kind(), "name" | "qualified_name" | "relative_name") {
                            let parent = node_text(c, source).to_string();
                            result.type_relationships.push(TypeRelationship {
                                source_type: qname.clone(),
                                target_type: Some(parent),
                                relationship: "extends".to_string(),
                                methods: Vec::new(),
                            });
                        }
                    }
                }
                "class_interface_clause" => {
                    let mut sub = child.walk();
                    for c in child.named_children(&mut sub) {
                        if matches!(c.kind(), "name" | "qualified_name" | "relative_name") {
                            let iface = node_text(c, source).to_string();
                            result.type_relationships.push(TypeRelationship {
                                source_type: qname.clone(),
                                target_type: Some(iface),
                                relationship: "implements".to_string(),
                                methods: Vec::new(),
                            });
                        }
                    }
                }
                _ => {}
            }
        }

        if let Some(body) = Self::extract_body(node) {
            let nested_prefix = qname.clone();
            let mut method_rel = TypeRelationship {
                source_type: qname.clone(),
                target_type: None,
                relationship: "inherent".to_string(),
                methods: Vec::new(),
            };
            let methods_start = result.functions.len();

            let mut body_cursor = body.walk();
            for child in body.named_children(&mut body_cursor) {
                match child.kind() {
                    "method_declaration" => {
                        Self::parse_function(
                            child,
                            source,
                            module_path,
                            &nested_prefix,
                            rel_path,
                            result,
                            true,
                        );
                    }
                    "const_declaration" => {
                        Self::parse_const(
                            child,
                            source,
                            module_path,
                            &nested_prefix,
                            rel_path,
                            result,
                        );
                    }
                    _ => {}
                }
            }

            // HAS_METHOD edges are computed by builder/type_edges.rs
            // from `inherent` TypeRelationships' `methods` Vec. Collect
            // the methods we just appended that belong directly to this
            // class (one separator past the nested_prefix).
            let direct_prefix = format!("{nested_prefix}\\");
            for fn_info in &result.functions[methods_start..] {
                if let Some(rest) = fn_info.qualified_name.strip_prefix(&direct_prefix) {
                    if !rest.contains('\\') {
                        method_rel.methods.push(fn_info.clone());
                    }
                }
            }
            if !method_rel.methods.is_empty() {
                result.type_relationships.push(method_rel);
            }
        }
    }

    fn parse_interface(
        node: Node,
        source: &[u8],
        module_path: &str,
        owner_prefix: &str,
        rel_path: &str,
        result: &mut ParseResult,
    ) {
        let Some(name) = Self::extract_name(node, source) else {
            return;
        };
        let visibility = Self::extract_visibility(node, source);
        let line = node.start_position().row as u32 + 1;
        let end_line = node.end_position().row as u32 + 1;
        let qname = make_qualified(module_path, owner_prefix, name, '\\');

        result.interfaces.push(InterfaceInfo {
            qualified_name: qname.clone(),
            visibility,
            name: name.to_string(),
            kind: "interface".to_string(),
            file_path: rel_path.to_string(),
            line_number: line,
            docstring: None,
            type_parameters: None,
            end_line: Some(end_line),
        });

        if let Some(body) = Self::extract_body(node) {
            let nested_prefix = qname;
            let mut cursor = body.walk();
            for child in body.named_children(&mut cursor) {
                if child.kind() == "method_declaration" {
                    Self::parse_function(
                        child,
                        source,
                        module_path,
                        &nested_prefix,
                        rel_path,
                        result,
                        true,
                    );
                }
            }
        }
    }

    fn parse_function(
        node: Node,
        source: &[u8],
        _module_path: &str,
        owner_prefix: &str,
        rel_path: &str,
        result: &mut ParseResult,
        is_method: bool,
    ) {
        let Some(name) = Self::extract_name(node, source) else {
            return;
        };
        let visibility = Self::extract_visibility(node, source);
        let is_static = Self::has_modifier(node, "static_modifier");
        let line = node.start_position().row as u32 + 1;
        let end_line = node.end_position().row as u32 + 1;
        let qname = if owner_prefix.is_empty() {
            name.to_string()
        } else {
            format!("{owner_prefix}\\{name}")
        };
        let return_type = node
            .child_by_field_name("return_type")
            .map(|c| node_text(c, source).to_string());
        let signature = Self::build_signature(node, source);
        let calls = Self::extract_calls(node, source);
        let decorators = Self::extract_attributes(node, source);
        let mut metadata: crate::code_tree::models::MetadataMap = Default::default();
        if is_static {
            metadata.insert("is_static".to_string(), serde_json::json!(true));
        }

        result.functions.push(FunctionInfo {
            qualified_name: qname,
            visibility,
            is_async: false,
            is_method,
            signature,
            file_path: rel_path.to_string(),
            line_number: line,
            name: name.to_string(),
            docstring: None,
            return_type,
            decorators,
            calls,
            references: Vec::new(),
            function_refs: Vec::new(),
            type_parameters: None,
            end_line: Some(end_line),
            parameters: Vec::new(),
            branch_count: None,
            param_count: None,
            max_nesting: None,
            is_recursive: None,
            procedure_names: Vec::new(),
            metadata,
        });
    }

    fn parse_const(
        node: Node,
        source: &[u8],
        module_path: &str,
        owner_prefix: &str,
        rel_path: &str,
        result: &mut ParseResult,
    ) {
        let visibility = Self::extract_visibility(node, source);
        let type_annotation = node
            .child_by_field_name("type")
            .map(|c| node_text(c, source).to_string());
        let line = node.start_position().row as u32 + 1;
        let mut cursor = node.walk();
        for child in node.named_children(&mut cursor) {
            if child.kind() != "const_element" {
                continue;
            }
            // const_element has no fields — walk children for the
            // `name` node (the identifier) and `expression` (the
            // value). value_preview is the raw const_element source
            // slice capped at 100 chars.
            let mut name_cursor = child.walk();
            let mut const_name: Option<String> = None;
            for c in child.named_children(&mut name_cursor) {
                if c.kind() == "name" {
                    const_name = Some(node_text(c, source).to_string());
                    break;
                }
            }
            let Some(const_name) = const_name else {
                continue;
            };
            let qname = if owner_prefix.is_empty() {
                if module_path.is_empty() {
                    const_name.clone()
                } else {
                    format!("{module_path}\\{const_name}")
                }
            } else {
                format!("{owner_prefix}::{const_name}")
            };
            let value_preview = {
                let raw = node_text(child, source);
                let take = raw
                    .char_indices()
                    .nth(100)
                    .map(|(i, _)| i)
                    .unwrap_or(raw.len());
                Some(raw[..take].to_string())
            };
            result.constants.push(ConstantInfo {
                qualified_name: qname,
                visibility: visibility.clone(),
                name: const_name,
                kind: "constant".to_string(),
                type_annotation: type_annotation.clone(),
                value_preview,
                file_path: rel_path.to_string(),
                line_number: line,
            });
        }
    }

    fn build_signature(node: Node, source: &[u8]) -> String {
        // Take everything before the function body (`compound_statement`
        // for methods, `compound_statement` or `;` for interface
        // methods).
        let mut parts: Vec<&str> = Vec::new();
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            if child.kind() == "compound_statement" {
                break;
            }
            parts.push(node_text(child, source));
        }
        parts.join(" ")
    }

    fn extract_calls(node: Node, source: &[u8]) -> Vec<(String, u32)> {
        let mut calls: Vec<(String, u32)> = Vec::new();
        fn walk(n: Node, source: &[u8], out: &mut Vec<(String, u32)>) {
            // PHP function/method call nodes.
            let kind = n.kind();
            if matches!(
                kind,
                "function_call_expression"
                    | "member_call_expression"
                    | "scoped_call_expression"
                    | "nullsafe_member_call_expression"
            ) {
                let line = n.start_position().row as u32 + 1;
                // For free-function calls, the first child is the
                // function name. For member/scoped calls the `name`
                // field gives the method.
                let callee = if let Some(name) = n.child_by_field_name("name") {
                    Some(node_text(name, source).to_string())
                } else if let Some(func) = n.child_by_field_name("function") {
                    Some(node_text(func, source).to_string())
                } else {
                    n.named_child(0).map(|c| node_text(c, source).to_string())
                };
                if let Some(callee) = callee {
                    let bare = callee.rsplit("\\").next().unwrap_or(&callee).trim();
                    let bare = bare.rsplit("::").next().unwrap_or(bare);
                    if !bare.is_empty() && !bare.contains(' ') && !bare.contains('(') {
                        out.push((bare.to_string(), line));
                    }
                }
            }
            let mut cursor = n.walk();
            for child in n.named_children(&mut cursor) {
                walk(child, source, out);
            }
        }
        walk(node, source, &mut calls);
        calls
    }
}

/// Find the first top-level `namespace_definition` that lacks a body
/// (`namespace Foo;` form). Its name becomes the file's
/// module_path. Block-form namespaces (`namespace Foo { ... }`) are
/// handled per-block in `parse_block`.
fn extract_file_namespace(program: Node, source: &[u8]) -> Option<String> {
    let mut cursor = program.walk();
    for child in program.named_children(&mut cursor) {
        if child.kind() == "namespace_definition" && child.child_by_field_name("body").is_none() {
            return child
                .child_by_field_name("name")
                .map(|n| node_text(n, source).to_string());
        }
    }
    None
}

impl LanguageParser for PhpParser {
    fn language_name(&self) -> &'static str {
        "php"
    }
    fn file_extensions(&self) -> &'static [&'static str] {
        &["php"]
    }
    fn noise_names(&self) -> &'static [&'static str] {
        PHP_NOISE_NAMES
    }

    fn parse_file(&self, filepath: &Path, src_root: &Path) -> ParseResult {
        let mut result = ParseResult::new();
        let Ok(source) = std::fs::read_to_string(filepath) else {
            return result;
        };
        let source_bytes = source.as_bytes();
        let rel_path = filepath
            .strip_prefix(src_root)
            .unwrap_or(filepath)
            .to_string_lossy()
            .to_string();
        let path_default_module = file_to_module_path(filepath, src_root, '\\');

        let Some(tree) = self.parse_tree(source_bytes) else {
            return result;
        };
        let root = tree.root_node();

        // Determine the file-level module_path. PHP files can declare
        // a top-level `namespace Foo;` that applies to everything
        // after; block-form `namespace Foo { ... }` is per-block (see
        // parse_block). If neither exists, fall back to the file path
        // derivation so unnamespaced PHP still gets unique qnames.
        let file_namespace = extract_file_namespace(root, source_bytes);
        let module_path = file_namespace.unwrap_or(path_default_module);

        let filename = filepath
            .file_name()
            .and_then(|o| o.to_str())
            .unwrap_or("")
            .to_string();
        let is_test =
            crate::code_tree::parsers::shared::is_test_path(&rel_path, &filename, &["Test.php"]);
        let mut file_info = FileInfo {
            path: rel_path.clone(),
            filename,
            loc: source.lines().count() as u32,
            module_path: module_path.clone(),
            language: "php".to_string(),
            submodule_declarations: Vec::new(),
            imports: Vec::new(),
            exports: Vec::new(),
            annotations: None,
            is_test,
            skip_reason: None,
        };

        Self::parse_block(
            root,
            source_bytes,
            &module_path,
            "",
            &rel_path,
            &mut result,
            &mut file_info,
        );

        result.files.push(file_info);
        result
    }
}
