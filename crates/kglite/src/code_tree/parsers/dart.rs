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
//!   - Methods, named & factory constructors, getters/setters, operators →
//!     FunctionInfo + the `inherent` TypeRelationship for HAS_METHOD edges.
//!     `is_async` / `is_static` / constructor & accessor flags in metadata.
//!   - Member fields → AttributeInfo; top-level `const`/`final` → ConstantInfo;
//!     `typedef` → ConstantInfo (kind="type_alias").
//!   - Top-level functions / getters / setters → FunctionInfo.
//!   - Call sites → `FunctionInfo.calls` (CALLS edges); structured
//!     parameters and cyclomatic branch metrics.
//!   - `import` / `export` directives → FileInfo.imports; `part of` files
//!     adopt their parent library's module path.
//!   - TODO/FIXME-style comment annotations; Dart 3 class modifiers.
//!   - Visibility from the Dart naming convention (leading `_` = private).
//!   - Flutter pass: `StatelessWidget` / `StatefulWidget` / `State`
//!     subclasses and their `build` methods are tagged for fast lookup.

use std::path::Path;
use tree_sitter::{Node, Parser, Tree};

use super::shared::{
    compute_complexity, extract_comment_annotations, file_to_module_path, make_qualified,
    node_text, BRANCH_KINDS_DART,
};
use super::LanguageParser;
use crate::code_tree::models::{
    AttributeInfo, ClassInfo, ConstantInfo, EnumInfo, FileInfo, FunctionInfo, MetadataMap,
    ParameterInfo, ParameterKind, ParseResult, TypeRelationship,
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

/// The constructor / accessor signature node kinds dispatched by
/// [`DartParser::dispatch_signature`].
const CONSTRUCTOR_KINDS: &[&str] = &[
    "constructor_signature",
    "factory_constructor_signature",
    "constant_constructor_signature",
    "redirecting_factory_constructor_signature",
];

/// Function/closure node kinds `compute_complexity` must not descend into —
/// their branches belong to the nested callable, not the parent.
const NESTED_SCOPES: &[&str] = &["function_expression", "local_function_declaration"];

/// Comment node kinds scanned for TODO/FIXME-style annotations.
const DART_COMMENT_TYPES: &[&str] = &["comment", "block_comment", "documentation_block_comment"];

/// Class-declaration modifier keywords surfaced into `ClassInfo.metadata`.
const CLASS_MODIFIERS: &[&str] = &["abstract", "base", "interface", "sealed", "final"];

/// Flutter widget base classes → the `flutter_widget` tag each implies.
const FLUTTER_WIDGET_BASES: &[(&str, &str)] = &[
    ("StatelessWidget", "stateless"),
    ("StatefulWidget", "stateful"),
    ("State", "state"),
];

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
        // The package root — the first `.`-segment of the module path —
        // anchors import-URI normalisation.
        let pkg_root = module_path.split('.').next().unwrap_or("");
        let mut cursor = root.walk();
        for child in root.named_children(&mut cursor) {
            match child.kind() {
                "import_or_export" => {
                    if let Some(target) = Self::extract_import(child, source, pkg_root) {
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
                "function_declaration" | "getter_declaration" | "setter_declaration" => {
                    // Top-level — owner_prefix empty → not a method.
                    Self::parse_outer_callable(child, source, "", rel_path, result);
                }
                "type_alias" => {
                    Self::parse_type_alias(child, source, module_path, rel_path, result);
                }
                "top_level_variable_declaration" => {
                    Self::parse_top_level_var(child, source, module_path, rel_path, result);
                }
                _ => {}
            }
        }
    }

    /// `import_or_export` → the imported/exported library, normalised to
    /// the synthetic module-path scheme so import edges resolve. Both
    /// directions create a file-level dependency, so both land in
    /// `FileInfo.imports`.
    fn extract_import(node: Node, source: &[u8], pkg_root: &str) -> Option<String> {
        first_string_literal(node, source)
            .filter(|s| !s.is_empty())
            .map(|uri| normalize_dart_import(&uri, pkg_root))
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
        let qname = make_qualified(module_path, "", &name, '.');

        let supertypes = collect_supertypes(node, source);

        // Dart 3 class modifiers (abstract / base / interface / sealed /
        // final) → metadata, with `is_abstract` lifted out for convenience.
        let mut metadata = MetadataMap::new();
        let mods: Vec<serde_json::Value> = CLASS_MODIFIERS
            .iter()
            .filter(|m| has_token_child(node, m))
            .map(|m| serde_json::Value::String((*m).to_string()))
            .collect();
        if mods.iter().any(|v| v.as_str() == Some("abstract")) {
            metadata.insert("is_abstract".into(), serde_json::Value::Bool(true));
        }
        if !mods.is_empty() {
            metadata.insert("modifiers".into(), serde_json::Value::Array(mods));
        }

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
            metadata,
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
        let qname = make_qualified(module_path, "", &name, '.');

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

    /// One item inside a `class_member`. `method_declaration` carries a
    /// `method_signature` + body; a bare `declaration` carries a bodyless
    /// signature (abstract method / redirecting constructor) or a field list.
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
                let body = item.child_by_field_name("body");
                let is_static = has_token_child(msig, "static");
                let mut cursor = msig.walk();
                for sig in msig.named_children(&mut cursor) {
                    Self::dispatch_signature(
                        sig,
                        body,
                        source,
                        class_qname,
                        rel_path,
                        result,
                        is_static,
                    );
                }
            }
            "declaration" => {
                let is_static = has_token_child(item, "static");
                let mut cursor = item.walk();
                for sig in item.named_children(&mut cursor) {
                    match sig.kind() {
                        "static_final_declaration_list"
                        | "initialized_identifier_list"
                        | "identifier_list" => {
                            Self::parse_fields(sig, source, class_qname, rel_path, result);
                        }
                        _ => Self::dispatch_signature(
                            sig,
                            None,
                            source,
                            class_qname,
                            rel_path,
                            result,
                            is_static,
                        ),
                    }
                }
            }
            _ => {}
        }
    }

    /// Dispatch a single signature node (function / getter / setter /
    /// operator / constructor) to a `FunctionInfo`.
    fn dispatch_signature(
        sig: Node,
        body: Option<Node>,
        source: &[u8],
        owner_prefix: &str,
        rel_path: &str,
        result: &mut ParseResult,
        is_static: bool,
    ) {
        let is_method = !owner_prefix.is_empty();
        let mut meta = MetadataMap::new();
        if is_static {
            meta.insert("is_static".into(), serde_json::Value::Bool(true));
        }
        match sig.kind() {
            "function_signature" => {
                let Some(name) = sig.child_by_field_name("name") else {
                    return;
                };
                let name = node_text(name, source).to_string();
                emit_function(
                    result,
                    rel_path,
                    source,
                    sig,
                    body,
                    name,
                    owner_prefix,
                    is_method,
                    meta,
                );
            }
            "getter_signature" | "setter_signature" => {
                let Some(raw) = sig.child_by_field_name("name") else {
                    return;
                };
                let raw = node_text(raw, source);
                let is_setter = sig.kind() == "setter_signature";
                meta.insert(
                    "accessor".into(),
                    serde_json::Value::String(if is_setter { "setter" } else { "getter" }.into()),
                );
                // A setter shares its bare name with the matching getter;
                // the `=` suffix (idiomatic Dart) keeps their qualified
                // names — and thus graph nodes — distinct.
                let name = if is_setter {
                    format!("{raw}=")
                } else {
                    raw.to_string()
                };
                emit_function(
                    result,
                    rel_path,
                    source,
                    sig,
                    body,
                    name,
                    owner_prefix,
                    is_method,
                    meta,
                );
            }
            "operator_signature" => {
                let op = sig
                    .child_by_field_name("operator")
                    .map(|o| node_text(o, source).trim().to_string())
                    .unwrap_or_default();
                meta.insert("is_operator".into(), serde_json::Value::Bool(true));
                emit_function(
                    result,
                    rel_path,
                    source,
                    sig,
                    body,
                    format!("operator{op}"),
                    owner_prefix,
                    true,
                    meta,
                );
            }
            k if CONSTRUCTOR_KINDS.contains(&k) => {
                Self::parse_constructor(sig, body, source, owner_prefix, rel_path, result);
            }
            _ => {}
        }
    }

    /// A constructor signature → `FunctionInfo`. The constructor `name`
    /// field is a sequence of identifiers (`Point` / `Point . origin`):
    /// unnamed constructors qualify as `Owner.Owner`, named ones as
    /// `Owner.Owner.named` — distinct, addressable, collision-free.
    fn parse_constructor(
        sig: Node,
        body: Option<Node>,
        source: &[u8],
        class_qname: &str,
        rel_path: &str,
        result: &mut ParseResult,
    ) {
        let ident_parts: Vec<String> = children_by_field(sig, "name")
            .into_iter()
            .filter(|n| n.kind() == "identifier")
            .map(|n| node_text(n, source).to_string())
            .collect();
        if ident_parts.is_empty() {
            return;
        }
        let ctor_name = ident_parts.join(".");
        let mut meta = MetadataMap::new();
        meta.insert("is_constructor".into(), serde_json::Value::Bool(true));
        if matches!(
            sig.kind(),
            "factory_constructor_signature" | "redirecting_factory_constructor_signature"
        ) {
            meta.insert("is_factory".into(), serde_json::Value::Bool(true));
        }
        if sig.kind() == "constant_constructor_signature" {
            meta.insert("is_const".into(), serde_json::Value::Bool(true));
        }
        if ctor_name.contains('.') {
            meta.insert("is_named".into(), serde_json::Value::Bool(true));
        }
        emit_function(
            result,
            rel_path,
            source,
            sig,
            body,
            ctor_name,
            class_qname,
            true,
            meta,
        );
    }

    /// Top-level `function_declaration` / `getter_declaration` /
    /// `setter_declaration` — a `signature` field plus a `body`.
    fn parse_outer_callable(
        node: Node,
        source: &[u8],
        owner_prefix: &str,
        rel_path: &str,
        result: &mut ParseResult,
    ) {
        let Some(sig) = node.child_by_field_name("signature") else {
            return;
        };
        let body = node.child_by_field_name("body");
        Self::dispatch_signature(sig, body, source, owner_prefix, rel_path, result, false);
    }

    /// A field list inside a `declaration` → one `AttributeInfo` per name.
    fn parse_fields(
        list: Node,
        source: &[u8],
        owner_qname: &str,
        rel_path: &str,
        result: &mut ParseResult,
    ) {
        let mut cursor = list.walk();
        for entry in list.named_children(&mut cursor) {
            let name_node = match entry.kind() {
                "identifier" => Some(entry),
                _ => entry.child_by_field_name("name"),
            };
            let Some(name_node) = name_node else {
                continue;
            };
            let name = node_text(name_node, source).to_string();
            let line = entry.start_position().row as u32 + 1;
            let default_value = entry
                .child_by_field_name("value")
                .map(|v| truncate_preview(node_text(v, source)));
            result.attributes.push(AttributeInfo {
                qualified_name: format!("{owner_qname}.{name}"),
                owner_qualified_name: owner_qname.to_string(),
                visibility: visibility_from_name(&name).to_string(),
                name,
                type_annotation: None,
                file_path: rel_path.to_string(),
                line_number: line,
                default_value,
            });
        }
    }

    /// A top-level `const` / `final` declaration → one `ConstantInfo` per
    /// declared name. Plain mutable `var`s are skipped — they are not
    /// constants.
    fn parse_top_level_var(
        node: Node,
        source: &[u8],
        module_path: &str,
        rel_path: &str,
        result: &mut ParseResult,
    ) {
        let modifier = node
            .child_by_field_name("modifier")
            .map(|m| node_text(m, source));
        if !matches!(modifier, Some("const") | Some("final")) {
            return;
        }
        let type_annotation = node
            .child_by_field_name("type")
            .map(|t| node_text(t, source).trim().to_string())
            .filter(|s| !s.is_empty());
        let line = node.start_position().row as u32 + 1;

        let mut cursor = node.walk();
        for list in node.named_children(&mut cursor) {
            if !matches!(
                list.kind(),
                "static_final_declaration_list" | "initialized_identifier_list"
            ) {
                continue;
            }
            let mut inner = list.walk();
            for entry in list.named_children(&mut inner) {
                let Some(name_node) = entry.child_by_field_name("name") else {
                    continue;
                };
                let name = node_text(name_node, source).to_string();
                let value_preview = entry
                    .child_by_field_name("value")
                    .map(|v| truncate_preview(node_text(v, source)));
                result.constants.push(ConstantInfo {
                    qualified_name: make_qualified(module_path, "", &name, '.'),
                    kind: "constant".to_string(),
                    type_annotation: type_annotation.clone(),
                    value_preview,
                    visibility: visibility_from_name(&name).to_string(),
                    file_path: rel_path.to_string(),
                    line_number: line,
                    name,
                });
            }
        }
    }

    /// `typedef Name = ...;` → a `ConstantInfo` with kind `type_alias`,
    /// matching how TypeScript type aliases are stored.
    fn parse_type_alias(
        node: Node,
        source: &[u8],
        module_path: &str,
        rel_path: &str,
        result: &mut ParseResult,
    ) {
        let Some(name_node) = first_child_of_kind(node, "type_identifier") else {
            return;
        };
        let name = node_text(name_node, source).to_string();
        let line = node.start_position().row as u32 + 1;
        // The aliased type — the first `type` / `function_type` child past
        // the alias name — becomes the value preview.
        let mut cursor = node.walk();
        let aliased = node
            .named_children(&mut cursor)
            .find(|c| matches!(c.kind(), "type" | "function_type") && c.id() != name_node.id())
            .map(|c| truncate_preview(node_text(c, source)));
        result.constants.push(ConstantInfo {
            qualified_name: make_qualified(module_path, "", &name, '.'),
            kind: "type_alias".to_string(),
            type_annotation: None,
            value_preview: aliased,
            visibility: visibility_from_name(&name).to_string(),
            file_path: rel_path.to_string(),
            line_number: line,
            name,
        });
    }
}

/// Build a `FunctionInfo` from a signature node and optional body, and push
/// it onto `result`. Shared by every callable shape — plain functions,
/// methods, accessors, operators, constructors.
#[allow(clippy::too_many_arguments)]
fn emit_function(
    result: &mut ParseResult,
    rel_path: &str,
    source: &[u8],
    sig: Node,
    body: Option<Node>,
    name: String,
    owner_prefix: &str,
    is_method: bool,
    metadata: MetadataMap,
) {
    let line = sig.start_position().row as u32 + 1;
    let end_line = body
        .map(|b| b.end_position().row as u32 + 1)
        .unwrap_or_else(|| sig.end_position().row as u32 + 1);
    let qname = if owner_prefix.is_empty() {
        name.clone()
    } else {
        format!("{owner_prefix}.{name}")
    };
    let return_type = sig
        .child_by_field_name("return_type")
        .map(|n| node_text(n, source).trim().to_string())
        .filter(|s| !s.is_empty());
    let visibility = visibility_from_name(&name).to_string();

    let parameters = extract_parameters(sig, source);
    let param_count = Some(parameters.len() as u32);
    let calls = body.map(|b| extract_calls(b, source)).unwrap_or_default();
    let (branch_count, max_nesting) = match body {
        Some(b) => {
            let (c, n) = compute_complexity(b, BRANCH_KINDS_DART, NESTED_SCOPES);
            (Some(c), Some(n))
        }
        None => (None, None),
    };
    // Heuristic self-recursion: a bare call to the function's own short name.
    let short = name.rsplit('.').next().unwrap_or(&name).to_string();
    let is_recursive = Some(calls.iter().any(|(callee, _)| *callee == short));

    result.functions.push(FunctionInfo {
        name,
        qualified_name: qname,
        visibility,
        is_async: body.map(is_body_async).unwrap_or(false),
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
        calls,
        references: Vec::new(),
        function_refs: Vec::new(),
        type_parameters: None,
        end_line: Some(end_line),
        parameters,
        branch_count,
        param_count,
        max_nesting,
        is_recursive,
        procedure_names: Vec::new(),
        metadata,
    });
}

/// Recursively collect call sites in a function body. Dart calls surface as
/// `call_expression` (bare or member-access callee) and `new` / `const`
/// constructor invocations.
fn extract_calls(node: Node, source: &[u8]) -> Vec<(String, u32)> {
    fn callee_name(func: Node, source: &[u8]) -> Option<String> {
        match func.kind() {
            "identifier" => Some(node_text(func, source).to_string()),
            "member_expression" => func
                .child_by_field_name("property")
                .map(|p| node_text(p, source).to_string()),
            _ => None,
        }
    }
    fn walk(n: Node, source: &[u8], out: &mut Vec<(String, u32)>) {
        let line = n.start_position().row as u32 + 1;
        match n.kind() {
            "call_expression" => {
                if let Some(name) = n
                    .child_by_field_name("function")
                    .and_then(|f| callee_name(f, source))
                {
                    if !name.is_empty() {
                        out.push((name, line));
                    }
                }
            }
            "new_expression" | "const_object_expression" => {
                if let Some(t) = n.child_by_field_name("type") {
                    let name = bare_type_name(node_text(t, source));
                    if !name.is_empty() {
                        out.push((name, line));
                    }
                }
            }
            _ => {}
        }
        let mut cursor = n.walk();
        for child in n.named_children(&mut cursor) {
            walk(child, source, out);
        }
    }
    let mut out = Vec::new();
    walk(node, source, &mut out);
    out
}

/// Structured parameter list for a signature. Best-effort: names are
/// reliable, declared types stay on the `signature` string (which the
/// USES_TYPE scanner already reads).
fn extract_parameters(sig: Node, source: &[u8]) -> Vec<ParameterInfo> {
    fn collect(node: Node, source: &[u8], out: &mut Vec<ParameterInfo>) {
        let mut cursor = node.walk();
        for child in node.named_children(&mut cursor) {
            match child.kind() {
                "formal_parameter" => {
                    let name = child
                        .child_by_field_name("name")
                        .map(|n| node_text(n, source).to_string())
                        .unwrap_or_default();
                    out.push(ParameterInfo {
                        name,
                        type_annotation: None,
                        default: None,
                        kind: ParameterKind::Positional,
                    });
                }
                // `[...]` / `{...}` optional & named parameter groups.
                "optional_formal_parameters" => collect(child, source, out),
                _ => {}
            }
        }
    }
    let mut out = Vec::new();
    for list in children_by_field(sig, "parameters") {
        if list.kind() == "formal_parameter_list" {
            collect(list, source, &mut out);
        }
    }
    out
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

/// Flutter pass: tag `StatelessWidget` / `StatefulWidget` / `State`
/// subclasses with `flutter_widget`, and their `build` methods with
/// `flutter_build`, so "show me the screens" queries are one hop away.
/// Pure metadata — no model-shape or builder change beyond two columns.
fn flutter_annotate(result: &mut ParseResult) {
    let mut widget_qnames: Vec<String> = Vec::new();
    for class in &mut result.classes {
        let tag = FLUTTER_WIDGET_BASES.iter().find_map(|&(base, tag)| {
            class
                .bases
                .iter()
                .any(|b| b.as_str() == base)
                .then_some(tag)
        });
        if let Some(tag) = tag {
            class.metadata.insert(
                "flutter_widget".into(),
                serde_json::Value::String(tag.into()),
            );
            widget_qnames.push(class.qualified_name.clone());
        }
    }
    if widget_qnames.is_empty() {
        return;
    }
    for func in &mut result.functions {
        if func.name != "build" {
            continue;
        }
        if let Some(owner) = func.qualified_name.strip_suffix(".build") {
            if widget_qnames.iter().any(|w| w == owner) {
                func.metadata
                    .insert("flutter_build".into(), serde_json::Value::Bool(true));
            }
        }
    }
}

/// Push an `inherent` TypeRelationship carrying every method appended to
/// `result.functions` since `methods_start` — the builder turns it into
/// HAS_METHOD edges. Dart has no nested type declarations, so everything a
/// `walk_class_body` appends is a direct member of `owner_qname`.
fn append_inherent_rel(result: &mut ParseResult, owner_qname: &str, methods_start: usize) {
    let methods: Vec<FunctionInfo> = result.functions[methods_start..].to_vec();
    if !methods.is_empty() {
        result.type_relationships.push(TypeRelationship {
            source_type: owner_qname.to_string(),
            target_type: None,
            relationship: "inherent".to_string(),
            methods,
        });
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

/// Dart visibility is by convention: a name whose first character is `_` is
/// library-private. The terminal `.`-segment is what carries the privacy —
/// a named constructor `Foo._internal` is private, `Foo.origin` is not.
fn visibility_from_name(name: &str) -> &'static str {
    let terminal = name.rsplit('.').next().unwrap_or(name);
    if terminal.starts_with('_') {
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

/// All children of `node` assigned the field name `field` — needed for
/// tree-sitter `multiple:true` fields, where `child_by_field_name` returns
/// only the first.
fn children_by_field<'a>(node: Node<'a>, field: &str) -> Vec<Node<'a>> {
    let mut cursor = node.walk();
    let mut out = Vec::new();
    if cursor.goto_first_child() {
        loop {
            if cursor.field_name() == Some(field) {
                out.push(cursor.node());
            }
            if !cursor.goto_next_sibling() {
                break;
            }
        }
    }
    out
}

/// Whether `node` has a direct child (named or anonymous) of kind `kind` —
/// used to spot keyword tokens like `static`.
fn has_token_child(node: Node, kind: &str) -> bool {
    let mut cursor = node.walk();
    let hit = node.children(&mut cursor).any(|c| c.kind() == kind);
    hit
}

/// Whether a `function_body` is declared `async` / `async*` / `sync*`.
fn is_body_async(body: Node) -> bool {
    let mut cursor = body.walk();
    let hit = body
        .children(&mut cursor)
        .any(|c| matches!(c.kind(), "async" | "async*" | "sync*"));
    hit
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

/// First ~100 chars of an expression, for `value_preview` / `default_value`.
fn truncate_preview(text: &str) -> String {
    let collapsed: String = text.split_whitespace().collect::<Vec<_>>().join(" ");
    collapsed.chars().take(100).collect()
}

/// Normalise a Dart import/export URI to the synthetic `<root>.<stem>`
/// module path so the builder's import resolver can match it against a
/// project file. `dart:` core libraries and `package:` URIs for *other*
/// packages are left verbatim — they are genuinely external and resolve
/// to nothing, which is correct. Relative imports and same-package
/// `package:` URIs resolve to a project module.
fn normalize_dart_import(uri: &str, pkg_root: &str) -> String {
    if pkg_root.is_empty() || uri.starts_with("dart:") {
        return uri.to_string();
    }
    let path_part = match uri.strip_prefix("package:") {
        Some(rest) => match rest.split_once('/') {
            // `package:<pkg>/...` resolves only within our own package.
            Some((pkg, p)) if pkg == pkg_root => p,
            _ => return uri.to_string(),
        },
        None => uri,
    };
    let stem = path_part
        .rsplit('/')
        .next()
        .unwrap_or(path_part)
        .strip_suffix(".dart")
        .unwrap_or(path_part);
    if stem.is_empty() {
        uri.to_string()
    } else {
        format!("{pkg_root}.{stem}")
    }
}

/// First `string_literal` descendant of `node`, surrounding quotes stripped.
fn first_string_literal(node: Node, source: &[u8]) -> Option<String> {
    if node.kind() == "string_literal" {
        return Some(strip_string_quotes(node_text(node, source)));
    }
    let mut cursor = node.walk();
    for child in node.named_children(&mut cursor) {
        if let Some(s) = first_string_literal(child, source) {
            return Some(s);
        }
    }
    None
}

/// If the file is a library `part`, resolve the parent library's module
/// path so `part` files collapse into one logical module. Only the URI
/// form (`part of 'parent.dart';`) maps onto the synthetic per-file module
/// scheme; the dotted-name form is left to the file's own module path.
fn resolve_part_of(root: Node, source: &[u8], src_root: &Path) -> Option<String> {
    let mut cursor = root.walk();
    for child in root.named_children(&mut cursor) {
        if child.kind() != "part_of_directive" {
            continue;
        }
        let uri = first_string_literal(child, source)?;
        let stem = uri
            .rsplit('/')
            .next()
            .unwrap_or(&uri)
            .strip_suffix(".dart")
            .unwrap_or(&uri);
        if stem.is_empty() {
            return None;
        }
        let pkg = src_root.file_name().and_then(|o| o.to_str()).unwrap_or("");
        return Some(if pkg.is_empty() {
            stem.to_string()
        } else {
            format!("{pkg}.{stem}")
        });
    }
    None
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

// `make_qualified` and `file_to_module_path` were inlined here pre-0.9.53;
// consolidated into `super::shared` with a `separator` parameter (Dart and
// most other languages use `.`; PHP uses `\`).

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

        let Some(tree) = self.parse_tree(source_bytes) else {
            return result;
        };
        let root = tree.root_node();

        // A `part of` file adopts its parent library's module path so the
        // split-across-files library collapses into one logical module.
        let module_path = resolve_part_of(root, source_bytes, src_root)
            .unwrap_or_else(|| file_to_module_path(filepath, src_root, '.'));

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
            annotations: extract_comment_annotations(root, source_bytes, DART_COMMENT_TYPES),
            is_test,
            skip_reason: None,
        };

        Self::parse_source_file(
            root,
            source_bytes,
            &module_path,
            &rel_path,
            &mut result,
            &mut file_info,
        );

        flutter_annotate(&mut result);

        result.files.push(file_info);
        result
    }
}
