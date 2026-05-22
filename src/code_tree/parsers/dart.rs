//! Dart language parser.
//!
//! Backed by the `tree-sitter-dart` grammar (the `nielsenko/tree-sitter-dart`
//! packaging on crates.io). The grammar's root node is `source_file`, with
//! real `function_declaration` / `method_declaration` wrappers and a real
//! `call_expression` — so the parser walks declarations directly, no
//! signature/body sibling-pairing needed.
//!
//! Coverage so far:
//!   - `class` declarations → ClassInfo (kind="class").
//!   - `mixin` declarations → ClassInfo (kind="mixin") → `Mixin` graph nodes.
//!   - `extension` / `extension type` → ClassInfo (kind="extension" /
//!     "extension_type") → `Class` graph nodes tagged by kind.
//!   - `enum` declarations → EnumInfo (variants + enhanced-enum methods).
//!   - `extends` / `with` / `implements` → EXTENDS / IMPLEMENTS edges.
//!   - Methods + the `inherent` TypeRelationship for HAS_METHOD edges.
//!   - Top-level functions → FunctionInfo.
//!   - `import` / `export` directives → FileInfo.imports.
//!   - Visibility from the Dart naming convention (leading `_` = private).
//!
//! Follow-up phases: constructors and accessors, calls, part/part-of,
//! complexity metrics, and the Flutter pass.

use std::path::Path;
use tree_sitter::{Node, Parser, Tree};

use super::shared::node_text;
use super::LanguageParser;
use crate::code_tree::models::{
    ClassInfo, EnumInfo, FileInfo, FunctionInfo, ParseResult, TypeRelationship,
};

pub struct DartParser;

thread_local! {
    static TS_PARSER: std::cell::RefCell<Parser> = {
        let mut p = Parser::new();
        p.set_language(&tree_sitter_dart::LANGUAGE.into())
            .expect("loading tree-sitter-dart grammar");
        std::cell::RefCell::new(p)
    };
}

impl DartParser {
    pub fn new() -> Self {
        DartParser
    }

    fn parse_tree(&self, source: &[u8]) -> Option<Tree> {
        TS_PARSER.with(|p| p.borrow_mut().parse(source, None))
    }

    /// Walk the `source_file` root and dispatch each top-level declaration.
    fn parse_source_file(
        root: Node,
        source: &[u8],
        module_path: &str,
        rel_path: &str,
        result: &mut ParseResult,
        file_info: &mut FileInfo,
    ) {
        let mut cursor = root.walk();
        for child in root.named_children(&mut cursor) {
            match child.kind() {
                "import_or_export" => {
                    if let Some(target) = Self::extract_import(child, source) {
                        file_info.imports.push(target);
                    }
                }
                "class_declaration" => {
                    Self::parse_type_decl(child, source, module_path, rel_path, result, "class");
                }
                "mixin_declaration" => {
                    Self::parse_type_decl(child, source, module_path, rel_path, result, "mixin");
                }
                "extension_declaration" => {
                    Self::parse_type_decl(
                        child,
                        source,
                        module_path,
                        rel_path,
                        result,
                        "extension",
                    );
                }
                "extension_type_declaration" => {
                    Self::parse_type_decl(
                        child,
                        source,
                        module_path,
                        rel_path,
                        result,
                        "extension_type",
                    );
                }
                "enum_declaration" => {
                    Self::parse_enum(child, source, module_path, rel_path, result);
                }
                "function_declaration" => {
                    // Top-level function — owner_prefix empty → not a method.
                    Self::parse_function_declaration(child, source, "", rel_path, result, false);
                }
                _ => {}
            }
        }
    }

    /// `import_or_export` → the imported/exported library URI (quotes
    /// stripped). Both directions create a file-level dependency, so both
    /// land in `FileInfo.imports`.
    fn extract_import(node: Node, source: &[u8]) -> Option<String> {
        fn first_string<'a>(n: Node<'a>, source: &'a [u8]) -> Option<String> {
            if n.kind() == "string_literal" {
                return Some(strip_string_quotes(node_text(n, source)));
            }
            let mut cursor = n.walk();
            for child in n.named_children(&mut cursor) {
                if let Some(s) = first_string(child, source) {
                    return Some(s);
                }
            }
            None
        }
        first_string(node, source).filter(|s| !s.is_empty())
    }

    /// Parse a class / mixin / extension / extension-type declaration into a
    /// `ClassInfo`. `kind` tags the graph node type ("mixin" → `Mixin`,
    /// everything else → `Class`, distinguished by the `kind` property).
    fn parse_type_decl(
        node: Node,
        source: &[u8],
        module_path: &str,
        rel_path: &str,
        result: &mut ParseResult,
        kind: &str,
    ) {
        let name = decl_name(node, source).unwrap_or_else(|| {
            // Anonymous `extension on Foo { ... }` — synthesize a stable,
            // addressable name from the extended type.
            let ext = node
                .child_by_field_name("class")
                .map(|c| bare_type_name(node_text(c, source)))
                .unwrap_or_default();
            if ext.is_empty() {
                "extension".to_string()
            } else {
                format!("extension_on_{ext}")
            }
        });
        let line = node.start_position().row as u32 + 1;
        let end_line = node.end_position().row as u32 + 1;
        let qname = make_qualified(module_path, "", &name);

        let supertypes = collect_supertypes(node, source);
        result.classes.push(ClassInfo {
            name: name.clone(),
            qualified_name: qname.clone(),
            kind: kind.to_string(),
            visibility: visibility_from_name(&name).to_string(),
            file_path: rel_path.to_string(),
            line_number: line,
            docstring: None,
            bases: supertypes.iter().map(|(t, _)| t.clone()).collect(),
            type_parameters: None,
            end_line: Some(end_line),
            metadata: Default::default(),
        });
        for (target, relationship) in &supertypes {
            result.type_relationships.push(TypeRelationship {
                source_type: name.clone(),
                target_type: Some(target.clone()),
                relationship: relationship.to_string(),
                methods: Vec::new(),
            });
        }

        if let Some(body) = node.child_by_field_name("body") {
            let methods_start = result.functions.len();
            Self::walk_class_body(body, source, &qname, rel_path, result);
            append_inherent_rel(result, &qname, methods_start);
        }
    }

    /// Parse an `enum_declaration` into an `EnumInfo` (variants + any
    /// enhanced-enum methods).
    fn parse_enum(
        node: Node,
        source: &[u8],
        module_path: &str,
        rel_path: &str,
        result: &mut ParseResult,
    ) {
        let Some(name) = decl_name(node, source) else {
            return;
        };
        let line = node.start_position().row as u32 + 1;
        let end_line = node.end_position().row as u32 + 1;
        let qname = make_qualified(module_path, "", &name);

        let mut variants: Vec<String> = Vec::new();
        if let Some(body) = node.child_by_field_name("body") {
            let mut cursor = body.walk();
            for child in body.named_children(&mut cursor) {
                if child.kind() == "enum_constant" {
                    if let Some(v) = child.child_by_field_name("name") {
                        variants.push(node_text(v, source).to_string());
                    }
                }
            }
        }

        result.enums.push(EnumInfo {
            name: name.clone(),
            qualified_name: qname.clone(),
            visibility: visibility_from_name(&name).to_string(),
            file_path: rel_path.to_string(),
            line_number: line,
            docstring: None,
            variants,
            end_line: Some(end_line),
            variant_details: None,
        });

        for (target, relationship) in collect_supertypes(node, source) {
            result.type_relationships.push(TypeRelationship {
                source_type: name.clone(),
                target_type: Some(target),
                relationship: relationship.to_string(),
                methods: Vec::new(),
            });
        }

        // Enhanced enums may carry methods inside the enum body.
        if let Some(body) = node.child_by_field_name("body") {
            let methods_start = result.functions.len();
            Self::walk_class_body(body, source, &qname, rel_path, result);
            append_inherent_rel(result, &qname, methods_start);
        }
    }

    /// Walk a `class_body` / `extension_body` / `enum_body`, descending
    /// through `class_member` wrappers.
    fn walk_class_body(
        body: Node,
        source: &[u8],
        class_qname: &str,
        rel_path: &str,
        result: &mut ParseResult,
    ) {
        let mut cursor = body.walk();
        for member in body.named_children(&mut cursor) {
            match member.kind() {
                "class_member" => {
                    let mut inner = member.walk();
                    for item in member.named_children(&mut inner) {
                        Self::handle_member_item(item, source, class_qname, rel_path, result);
                    }
                }
                // Defensive: handle a directly-nested member if the
                // grammar ever inlines the `class_member` wrapper.
                "method_declaration" | "declaration" => {
                    Self::handle_member_item(member, source, class_qname, rel_path, result);
                }
                _ => {}
            }
        }
    }

    /// One item inside a `class_member` — a method (with body) or a bare
    /// `declaration` (abstract method / field). Constructors, accessors and
    /// fields are handled in later phases; here we extract plain methods.
    fn handle_member_item(
        item: Node,
        source: &[u8],
        class_qname: &str,
        rel_path: &str,
        result: &mut ParseResult,
    ) {
        match item.kind() {
            "method_declaration" => {
                let Some(msig) = item.child_by_field_name("signature") else {
                    return;
                };
                if let Some(fsig) = first_child_of_kind(msig, "function_signature") {
                    let body = item.child_by_field_name("body");
                    Self::parse_function(fsig, body, source, class_qname, rel_path, result, true);
                }
            }
            "declaration" => {
                // Abstract method: `declaration` directly carries a
                // `function_signature` with no body.
                if let Some(fsig) = first_child_of_kind(item, "function_signature") {
                    Self::parse_function(fsig, None, source, class_qname, rel_path, result, true);
                }
            }
            _ => {}
        }
    }

    fn parse_function_declaration(
        node: Node,
        source: &[u8],
        owner_prefix: &str,
        rel_path: &str,
        result: &mut ParseResult,
        is_method: bool,
    ) {
        let Some(sig) = node.child_by_field_name("signature") else {
            return;
        };
        let body = node.child_by_field_name("body");
        Self::parse_function(sig, body, source, owner_prefix, rel_path, result, is_method);
    }

    /// Emit a `FunctionInfo` from a `function_signature` node and its
    /// optional `function_body`.
    fn parse_function(
        sig: Node,
        body: Option<Node>,
        source: &[u8],
        owner_prefix: &str,
        rel_path: &str,
        result: &mut ParseResult,
        is_method: bool,
    ) {
        let Some(name) = sig
            .child_by_field_name("name")
            .map(|n| node_text(n, source))
        else {
            return;
        };
        let line = sig.start_position().row as u32 + 1;
        let end_line = body
            .map(|b| b.end_position().row as u32 + 1)
            .unwrap_or_else(|| sig.end_position().row as u32 + 1);
        let qname = if owner_prefix.is_empty() {
            name.to_string()
        } else {
            format!("{owner_prefix}.{name}")
        };
        let return_type = sig
            .child_by_field_name("return_type")
            .map(|n| node_text(n, source).trim().to_string())
            .filter(|s| !s.is_empty());

        result.functions.push(FunctionInfo {
            name: name.to_string(),
            qualified_name: qname,
            visibility: visibility_from_name(name).to_string(),
            is_async: false,
            is_method,
            signature: node_text(sig, source)
                .split_whitespace()
                .collect::<Vec<_>>()
                .join(" "),
            file_path: rel_path.to_string(),
            line_number: line,
            docstring: None,
            return_type,
            decorators: Vec::new(),
            calls: Vec::new(),
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
}

/// Collect a type declaration's supertypes as `(bare_name, relationship)`
/// pairs. `relationship` is `"extends"` for the base class and
/// `"implements"` for `with`-clause mixins and `implements`-clause
/// interfaces — `with` folds into the implements/HAS_METHOD graph since a
/// mixin contributes capability rather than a base class.
fn collect_supertypes(node: Node, source: &[u8]) -> Vec<(String, &'static str)> {
    fn add(out: &mut Vec<(String, &'static str)>, name: String, rel: &'static str) {
        if !name.is_empty() {
            out.push((name, rel));
        }
    }
    fn each_type<'a>(node: Node<'a>, source: &'a [u8], out: &mut Vec<String>) {
        let mut cursor = node.walk();
        for child in node.named_children(&mut cursor) {
            let t = bare_type_name(node_text(child, source));
            if !t.is_empty() {
                out.push(t);
            }
        }
    }

    let mut out: Vec<(String, &'static str)> = Vec::new();

    // `extends Base with M1, M2` — the `superclass` field carries the base
    // type and an optional nested `mixins` clause.
    if let Some(sc) = node.child_by_field_name("superclass") {
        let mut cursor = sc.walk();
        for child in sc.named_children(&mut cursor) {
            if child.kind() == "mixins" {
                let mut mixins = Vec::new();
                each_type(child, source, &mut mixins);
                for m in mixins {
                    add(&mut out, m, "implements");
                }
            } else {
                add(
                    &mut out,
                    bare_type_name(node_text(child, source)),
                    "extends",
                );
            }
        }
    }

    // `with M1, M2` on an enum — `mixins` is a direct child there.
    {
        let mut cursor = node.walk();
        for child in node.named_children(&mut cursor) {
            if child.kind() == "mixins" {
                let mut mixins = Vec::new();
                each_type(child, source, &mut mixins);
                for m in mixins {
                    add(&mut out, m, "implements");
                }
            }
        }
    }

    // `implements I1, I2` — the `interfaces` field.
    if let Some(intf) = node.child_by_field_name("interfaces") {
        let mut ifaces = Vec::new();
        each_type(intf, source, &mut ifaces);
        for i in ifaces {
            add(&mut out, i, "implements");
        }
    }

    out
}

/// Collect the methods appended to `result.functions` since `methods_start`
/// that belong directly to `owner_qname`, and push an `inherent`
/// TypeRelationship so the builder emits HAS_METHOD edges.
fn append_inherent_rel(result: &mut ParseResult, owner_qname: &str, methods_start: usize) {
    let direct_prefix = format!("{owner_qname}.");
    let mut method_rel = TypeRelationship {
        source_type: owner_qname.to_string(),
        target_type: None,
        relationship: "inherent".to_string(),
        methods: Vec::new(),
    };
    for fn_info in &result.functions[methods_start..] {
        if let Some(rest) = fn_info.qualified_name.strip_prefix(&direct_prefix) {
            if !rest.contains('.') {
                method_rel.methods.push(fn_info.clone());
            }
        }
    }
    if !method_rel.methods.is_empty() {
        result.type_relationships.push(method_rel);
    }
}

/// Declaration name. Handles the `extension_type_name` wrapper and returns
/// `None` for an anonymous extension (no `name` field).
fn decl_name(node: Node, source: &[u8]) -> Option<String> {
    let name_node = node.child_by_field_name("name")?;
    match name_node.kind() {
        "extension_type_name" => {
            first_child_of_kind(name_node, "identifier").map(|id| node_text(id, source).to_string())
        }
        _ => Some(node_text(name_node, source).to_string()),
    }
}

/// Dart visibility is by convention: a name whose first character is `_`
/// is library-private; everything else is public.
fn visibility_from_name(name: &str) -> &'static str {
    if name.starts_with('_') {
        "private"
    } else {
        "public"
    }
}

/// First named child of `node` whose kind matches `kind`.
fn first_child_of_kind<'a>(node: Node<'a>, kind: &str) -> Option<Node<'a>> {
    let mut cursor = node.walk();
    let found = node.named_children(&mut cursor).find(|c| c.kind() == kind);
    found
}

/// Bare type name — strips generic arguments (`Foo<T>` → `Foo`) and
/// nullability (`Foo?` → `Foo`); a prefixed type (`p.Foo`) is kept whole.
fn bare_type_name(text: &str) -> String {
    let t = text.trim();
    let end = t
        .find(|c: char| c == '<' || c == '?' || c.is_whitespace() || c == '(')
        .unwrap_or(t.len());
    t[..end].trim().to_string()
}

/// Strip one matching pair of surrounding `'` or `"` quotes.
fn strip_string_quotes(s: &str) -> String {
    let s = s.trim();
    for q in ['\'', '"'] {
        if let Some(inner) = s.strip_prefix(q).and_then(|r| r.strip_suffix(q)) {
            return inner.to_string();
        }
    }
    s.to_string()
}

fn make_qualified(module_path: &str, owner_prefix: &str, name: &str) -> String {
    match (module_path.is_empty(), owner_prefix.is_empty()) {
        (true, true) => name.to_string(),
        (true, false) => format!("{owner_prefix}.{name}"),
        (false, true) => format!("{module_path}.{name}"),
        (false, false) => format!("{owner_prefix}.{name}"),
    }
}

/// Dart has no build-system-independent module identity (libraries are
/// named by `pubspec.yaml` package + file path). Mirror the Swift parser:
/// derive a per-file module name from the source-root dir + file stem so
/// the module-graph machinery still has a unique handle per file.
fn file_to_module_path(filepath: &Path, src_root: &Path) -> String {
    let stem = filepath.file_stem().and_then(|o| o.to_str()).unwrap_or("");
    let pkg = src_root.file_name().and_then(|o| o.to_str()).unwrap_or("");
    match (pkg.is_empty(), stem.is_empty()) {
        (true, _) => stem.to_string(),
        (false, true) => pkg.to_string(),
        (false, false) => format!("{pkg}.{stem}"),
    }
}

impl LanguageParser for DartParser {
    fn language_name(&self) -> &'static str {
        "dart"
    }

    fn file_extensions(&self) -> &'static [&'static str] {
        &["dart"]
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

        let filename = filepath
            .file_name()
            .and_then(|o| o.to_str())
            .unwrap_or("")
            .to_string();
        let is_test =
            crate::code_tree::parsers::shared::is_test_path(&rel_path, &filename, &["_test.dart"]);

        let mut file_info = FileInfo {
            path: rel_path.clone(),
            filename,
            loc: source.lines().count() as u32,
            module_path: module_path.clone(),
            language: "dart".to_string(),
            submodule_declarations: Vec::new(),
            imports: Vec::new(),
            exports: Vec::new(),
            annotations: None,
            is_test,
            skip_reason: None,
        };

        Self::parse_source_file(
            tree.root_node(),
            source_bytes,
            &module_path,
            &rel_path,
            &mut result,
            &mut file_info,
        );

        result.files.push(file_info);
        result
    }
}
