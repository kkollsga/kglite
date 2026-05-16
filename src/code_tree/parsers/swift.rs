//! Swift language parser (minimal viable).
//!
//! Coverage in 0.9.34:
//!   - `class` / `struct` / `actor` / `enum` declarations → ClassInfo
//!     (kind tagged from `declaration_kind`).
//!   - `protocol` declarations → InterfaceInfo with kind="protocol".
//!   - Top-level and nested `func` declarations → FunctionInfo.
//!   - `import Foundation` → FileInfo.imports.
//!   - Visibility (`public` / `internal` / `fileprivate` / `private`).
//!
//! Not yet supported (follow-up scope):
//!   - `extension` blocks emit a Class entry, but don't yet synthesize
//!     IMPLEMENTS edges to the protocol(s) they conform to.
//!   - `init` / `deinit` / `subscript` / computed properties.
//!   - `@objc` / `@available` / other attributes as decorators.
//!   - `async` / `throws` flags on functions.
//!   - Enum cases as AttributeInfo.
//!
//! These all map cleanly onto the existing `FunctionInfo` / `ClassInfo`
//! shape; they didn't make this commit's budget but the parser
//! structure leaves slots for each.

use std::path::Path;
use tree_sitter::{Node, Parser, Tree};

use super::shared::node_text;
use super::LanguageParser;
use crate::code_tree::models::{
    ClassInfo, FileInfo, FunctionInfo, InterfaceInfo, ParseResult, TypeRelationship,
};

pub const SWIFT_NOISE_NAMES: &[&str] = &[
    "print",
    "String",
    "Int",
    "Float",
    "Double",
    "Bool",
    "Array",
    "Dictionary",
    "Optional",
    "Range",
    "Result",
    "Error",
    "fatalError",
    "preconditionFailure",
    "assert",
    "precondition",
    "abs",
    "min",
    "max",
];

pub struct SwiftParser;

thread_local! {
    static TS_PARSER: std::cell::RefCell<Parser> = {
        let mut p = Parser::new();
        p.set_language(&tree_sitter_swift::LANGUAGE.into())
            .expect("loading tree-sitter-swift grammar");
        std::cell::RefCell::new(p)
    };
}

impl SwiftParser {
    pub fn new() -> Self {
        SwiftParser
    }

    fn parse_tree(&self, source: &[u8]) -> Option<Tree> {
        TS_PARSER.with(|p| p.borrow_mut().parse(source, None))
    }

    /// Walk top-level children of `source_file` and dispatch to per-kind
    /// extractors. Owner scope (for nested methods/types) is passed as
    /// the dotted prefix to compose qualified names.
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
                "import_declaration" => {
                    if let Some(target) = Self::extract_import(child, source) {
                        file_info.imports.push(target);
                    }
                }
                "class_declaration" => {
                    Self::parse_class(child, source, module_path, owner_prefix, rel_path, result);
                }
                "protocol_declaration" => {
                    Self::parse_protocol(
                        child,
                        source,
                        module_path,
                        owner_prefix,
                        rel_path,
                        result,
                    );
                }
                "function_declaration" => {
                    Self::parse_function(
                        child,
                        source,
                        module_path,
                        owner_prefix,
                        rel_path,
                        result,
                        owner_prefix.is_empty(), // top-level → not a method
                    );
                }
                _ => {}
            }
        }
    }

    fn extract_import(node: Node, source: &[u8]) -> Option<String> {
        // `import Foundation` / `import struct Foo.Bar`
        // We want the module path (e.g. "Foundation" or "Foo.Bar"). The
        // last identifier-or-navigation child is the target.
        let mut cursor = node.walk();
        let mut last_target: Option<String> = None;
        for child in node.named_children(&mut cursor) {
            let kind = child.kind();
            if kind == "identifier" || kind == "navigation_expression" || kind == "user_type" {
                last_target = Some(node_text(child, source).to_string());
            }
        }
        last_target
    }

    fn extract_visibility(node: Node, source: &[u8]) -> String {
        // Look for a `modifiers` named child whose body contains a
        // `visibility_modifier` token.
        let mut cursor = node.walk();
        for child in node.named_children(&mut cursor) {
            if child.kind() == "modifiers" {
                let mut sub = child.walk();
                for m in child.named_children(&mut sub) {
                    if m.kind() == "visibility_modifier" {
                        return node_text(m, source).to_string();
                    }
                }
            }
        }
        "internal".to_string()
    }

    /// Resolve `class_declaration` → kind ("class" | "struct" | "enum" |
    /// "actor" | "extension"). The grammar exposes the keyword as the
    /// anonymous `declaration_kind` field; its `kind()` is the literal
    /// keyword token type ("struct", "class", etc.).
    fn extract_declaration_kind(node: Node, _source: &[u8]) -> String {
        node.child_by_field_name("declaration_kind")
            .map(|c| c.kind().to_string())
            .unwrap_or_else(|| "class".to_string())
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
    ) {
        let Some(name) = Self::extract_name(node, source) else {
            return;
        };
        let kind_kw = Self::extract_declaration_kind(node, source);
        let visibility = Self::extract_visibility(node, source);
        let line = node.start_position().row as u32 + 1;
        let end_line = node.end_position().row as u32 + 1;
        let qname = make_qualified(module_path, owner_prefix, name);

        // Map declaration_kind to KGLite's class.kind enum. `extension`
        // emits a Class entry too (it adds methods to an existing type)
        // — we tag it with kind="extension" so downstream code can
        // distinguish, and the methods inside still get qualified-named
        // under the extended type's owner_prefix.
        let kind = match kind_kw.as_str() {
            "struct" => "struct",
            "enum" => "enum",
            "actor" => "actor",
            "extension" => "extension",
            _ => "class",
        };

        // `enum` flows into ClassInfo (no special EnumInfo extraction
        // here — Swift enums are heavy on associated values and would
        // need their own pass).
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

        if let Some(body) = Self::extract_body(node) {
            let nested_prefix = qname.clone();
            // Synthesize a TypeRelationship so the builder emits
            // HAS_METHOD edges between the class and each of its
            // methods. Mirrors python.rs's `method_rel`.
            let mut method_rel = TypeRelationship {
                source_type: qname.clone(),
                target_type: None,
                relationship: "inherent".to_string(),
                methods: Vec::new(),
            };
            let methods_start = result.functions.len();

            let mut cursor = body.walk();
            for child in body.named_children(&mut cursor) {
                match child.kind() {
                    "function_declaration" => {
                        Self::parse_function(
                            child,
                            source,
                            module_path,
                            &nested_prefix,
                            rel_path,
                            result,
                            true, // body-of-class → is_method=true
                        );
                    }
                    "class_declaration" => {
                        Self::parse_class(
                            child,
                            source,
                            module_path,
                            &nested_prefix,
                            rel_path,
                            result,
                        );
                    }
                    "protocol_declaration" => {
                        Self::parse_protocol(
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

            // Anything appended to `result.functions` since `methods_start`
            // whose qname is directly under this class (one separator past
            // `nested_prefix`) is a method of *this* class — nested
            // class methods that were also appended are skipped because
            // they live under a deeper prefix.
            let direct_prefix = format!("{nested_prefix}.");
            for fn_info in &result.functions[methods_start..] {
                let rest = match fn_info.qualified_name.strip_prefix(&direct_prefix) {
                    Some(r) => r,
                    None => continue,
                };
                if !rest.contains('.') {
                    method_rel.methods.push(fn_info.clone());
                }
            }
            if !method_rel.methods.is_empty() {
                result.type_relationships.push(method_rel);
            }
        }

        // Extension → emit a TypeRelationship if the grammar carries
        // protocol conformance info. Stub for now — left for follow-up
        // once we know the field shape.
        if kind == "extension" {
            result.type_relationships.push(TypeRelationship {
                source_type: name.to_string(),
                target_type: None,
                relationship: "inherent".to_string(),
                methods: Vec::new(),
            });
        }
    }

    fn parse_protocol(
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
        let qname = make_qualified(module_path, owner_prefix, name);

        result.interfaces.push(InterfaceInfo {
            qualified_name: qname.clone(),
            visibility,
            name: name.to_string(),
            kind: "protocol".to_string(),
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
                match child.kind() {
                    // protocol methods come through as either
                    // `protocol_function_declaration` (no body) or
                    // `function_declaration` (default impl). Both are
                    // emitted as FunctionInfo with is_method=true.
                    "function_declaration" | "protocol_function_declaration" => {
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
                    _ => {}
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
        let line = node.start_position().row as u32 + 1;
        let end_line = node.end_position().row as u32 + 1;
        let qname = if owner_prefix.is_empty() {
            name.to_string()
        } else {
            format!("{owner_prefix}.{name}")
        };
        let return_type = node
            .child_by_field_name("return_type")
            .map(|c| node_text(c, source).to_string());
        let signature = Self::build_signature(node, source);
        let calls = Self::extract_calls(node, source);

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
            decorators: Vec::new(),
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
            metadata: Default::default(),
        });
    }

    fn build_signature(node: Node, source: &[u8]) -> String {
        // Crude: everything before the function body. Mirrors the
        // approach Go's parser takes.
        let mut parts: Vec<&str> = Vec::new();
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            if child.kind() == "function_body" {
                break;
            }
            parts.push(node_text(child, source));
        }
        parts.join(" ")
    }

    fn extract_calls(node: Node, source: &[u8]) -> Vec<(String, u32)> {
        let mut calls: Vec<(String, u32)> = Vec::new();
        fn walk(n: Node, source: &[u8], out: &mut Vec<(String, u32)>) {
            if n.kind() == "call_expression" {
                let line = n.start_position().row as u32 + 1;
                // Function position is the first named child.
                if let Some(first) = n.named_child(0) {
                    let text = node_text(first, source);
                    // For navigation_expression `a.b()`, take the
                    // terminal segment as the bare callee name.
                    let bare = text.rsplit('.').next().unwrap_or(text).trim();
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

fn make_qualified(module_path: &str, owner_prefix: &str, name: &str) -> String {
    match (module_path.is_empty(), owner_prefix.is_empty()) {
        (true, true) => name.to_string(),
        (true, false) => format!("{owner_prefix}.{name}"),
        (false, true) => format!("{module_path}.{name}"),
        (false, false) => format!("{owner_prefix}.{name}"),
    }
}

fn file_to_module_path(filepath: &Path, src_root: &Path) -> String {
    // Swift modules normally come from the build system (SPM/Xcode);
    // the file-tree shape doesn't encode them. We fall back to the
    // file basename (without `.swift`) so each file gets a unique
    // module name and the existing module-graph machinery still works.
    let stem = filepath.file_stem().and_then(|o| o.to_str()).unwrap_or("");
    let pkg = src_root.file_name().and_then(|o| o.to_str()).unwrap_or("");
    match (pkg.is_empty(), stem.is_empty()) {
        (true, _) => stem.to_string(),
        (false, true) => pkg.to_string(),
        (false, false) => format!("{pkg}.{stem}"),
    }
}

impl LanguageParser for SwiftParser {
    fn language_name(&self) -> &'static str {
        "swift"
    }
    fn file_extensions(&self) -> &'static [&'static str] {
        &["swift"]
    }
    fn noise_names(&self) -> &'static [&'static str] {
        SWIFT_NOISE_NAMES
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
        let module_path = file_to_module_path(filepath, src_root);

        let Some(tree) = self.parse_tree(source_bytes) else {
            return result;
        };

        let mut file_info = FileInfo {
            path: rel_path.clone(),
            filename: filepath
                .file_name()
                .and_then(|o| o.to_str())
                .unwrap_or("")
                .to_string(),
            loc: source.lines().count() as u32,
            module_path: module_path.clone(),
            language: "swift".to_string(),
            submodule_declarations: Vec::new(),
            imports: Vec::new(),
            exports: Vec::new(),
            annotations: None,
            is_test: rel_path.to_lowercase().contains("test"),
            skip_reason: None,
        };

        Self::parse_block(
            tree.root_node(),
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
