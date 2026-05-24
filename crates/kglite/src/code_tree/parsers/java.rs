//! Java language parser (ported from parsers/java.py).

use serde_json::json;
use std::path::Path;
use tree_sitter::{Node, Parser, Tree};

use super::shared::{
    compute_complexity, count_lines, extract_comment_annotations, extract_procedure_annotations,
    get_type_parameters, is_generated_or_minified, node_text, BRANCH_KINDS_JAVA,
    DEFAULT_COMMENT_TYPES,
};
use super::LanguageParser;
use crate::code_tree::models::{
    AttributeInfo, ClassInfo, ConstantInfo, EnumInfo, FileInfo, FunctionInfo, InterfaceInfo,
    ParameterInfo, ParameterKind, ParseResult, TypeRelationship,
};

pub const JAVA_NOISE_NAMES: &[&str] = &[
    "toString",
    "equals",
    "hashCode",
    "compareTo",
    "getClass",
    "notify",
    "notifyAll",
    "wait",
    "clone",
    "finalize",
    "length",
    "size",
    "get",
    "set",
    "add",
    "remove",
    "contains",
    "isEmpty",
    "put",
    "keySet",
    "values",
    "entrySet",
    "iterator",
    "next",
    "hasNext",
    "clear",
    "stream",
    "map",
    "filter",
    "collect",
    "forEach",
    "reduce",
    "flatMap",
    "sorted",
    "distinct",
    "limit",
    "println",
    "printf",
    "print",
    "format",
    "read",
    "write",
    "close",
    "flush",
];

const NESTED_SCOPES: &[&str] = &[
    "method_declaration",
    "constructor_declaration",
    "lambda_expression",
];

const TYPE_NODES: &[&str] = &[
    "type_identifier",
    "integral_type",
    "boolean_type",
    "floating_point_type",
    "void_type",
    "generic_type",
    "array_type",
    "scoped_type_identifier",
];

pub struct JavaParser;

thread_local! {
    static JAVA_PARSER: std::cell::RefCell<Parser> = {
        let mut p = Parser::new();
        p.set_language(&tree_sitter_java::LANGUAGE.into())
            .expect("loading tree-sitter-java grammar");
        std::cell::RefCell::new(p)
    };
}

impl JavaParser {
    pub fn new() -> Self {
        JavaParser
    }
    fn parse_tree(&self, source: &[u8]) -> Option<Tree> {
        JAVA_PARSER.with(|p| p.borrow_mut().parse(source, None))
    }

    fn get_visibility(node: Node, source: &[u8]) -> &'static str {
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            if child.kind() == "modifiers" {
                let text = node_text(child, source);
                if text.contains("public") {
                    return "public";
                }
                if text.contains("protected") {
                    return "protected";
                }
                if text.contains("private") {
                    return "private";
                }
                return "package-private";
            }
        }
        "package-private"
    }

    fn has_modifier(node: Node, source: &[u8], modifier: &str) -> bool {
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            if child.kind() == "modifiers" {
                let mut mc = child.walk();
                for sub in child.children(&mut mc) {
                    if node_text(sub, source) == modifier {
                        return true;
                    }
                }
            }
        }
        false
    }

    fn get_annotations(node: Node, source: &[u8]) -> Vec<String> {
        let mut out = Vec::new();
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            if child.kind() == "modifiers" {
                let mut mc = child.walk();
                for sub in child.children(&mut mc) {
                    if matches!(sub.kind(), "annotation" | "marker_annotation") {
                        let text = node_text(sub, source);
                        let stripped = text.strip_prefix('@').unwrap_or(text);
                        out.push(stripped.to_string());
                    }
                }
            }
        }
        out
    }

    fn get_doc_comment(node: Node, source: &[u8]) -> Option<String> {
        let sibling = node.prev_named_sibling()?;
        if !matches!(sibling.kind(), "comment" | "block_comment") {
            return None;
        }
        let text = node_text(sibling, source).trim();
        let rest = text.strip_prefix("/**")?;
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
        if joined.is_empty() {
            None
        } else {
            Some(joined)
        }
    }

    fn get_name<'a>(node: Node<'a>, source: &'a [u8]) -> Option<&'a str> {
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            if child.kind() == "identifier" {
                return Some(node_text(child, source));
            }
        }
        None
    }

    fn get_signature(node: Node, source: &[u8]) -> String {
        let mut parts: Vec<&str> = Vec::new();
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            if matches!(child.kind(), "block" | "constructor_body") {
                break;
            }
            parts.push(node_text(child, source));
        }
        parts.join(" ")
    }

    fn get_return_type(node: Node, source: &[u8]) -> Option<String> {
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            if child.kind() == "identifier" {
                break;
            }
            if TYPE_NODES.contains(&child.kind()) {
                return Some(node_text(child, source).to_string());
            }
        }
        None
    }

    fn extract_calls(body: Node, source: &[u8]) -> Vec<(String, u32)> {
        let mut calls: Vec<(String, u32)> = Vec::new();
        fn walk(node: Node, source: &[u8], out: &mut Vec<(String, u32)>) {
            match node.kind() {
                "method_invocation" => {
                    let line = node.start_position().row as u32 + 1;
                    let name = node.child_by_field_name("name");
                    let obj = node.child_by_field_name("object");
                    match (name, obj) {
                        (Some(name), Some(obj)) => {
                            let method_name = node_text(name, source);
                            let obj_text = node_text(obj, source);
                            let hint = obj_text.rsplit('.').next().unwrap_or(obj_text);
                            if hint == "this" || hint == "super" {
                                out.push((method_name.to_string(), line));
                            } else {
                                out.push((format!("{}.{}", hint, method_name), line));
                            }
                        }
                        (Some(name), None) => {
                            out.push((node_text(name, source).to_string(), line));
                        }
                        _ => {}
                    }
                }
                "object_creation_expression" => {
                    let line = node.start_position().row as u32 + 1;
                    if let Some(t) = node.child_by_field_name("type") {
                        out.push((node_text(t, source).to_string(), line));
                    }
                }
                _ => {}
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

    fn get_package(root: Node, source: &[u8]) -> String {
        let mut cursor = root.walk();
        for child in root.children(&mut cursor) {
            if child.kind() == "package_declaration" {
                let mut sc = child.walk();
                for sub in child.children(&mut sc) {
                    if matches!(sub.kind(), "scoped_identifier" | "identifier") {
                        return node_text(sub, source).to_string();
                    }
                }
            }
        }
        String::new()
    }

    fn file_to_module_path(filepath: &Path, src_root: &Path, package: &str) -> String {
        if !package.is_empty() {
            return package.to_string();
        }
        let rel = filepath.strip_prefix(src_root).unwrap_or(filepath);
        let parent = rel.parent();
        let parts: Vec<String> = parent
            .map(|p| {
                p.components()
                    .filter_map(|c| c.as_os_str().to_str().map(str::to_string))
                    .collect()
            })
            .unwrap_or_default();
        if parts.is_empty() {
            filepath
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("")
                .to_string()
        } else {
            parts.join(".")
        }
    }

    fn get_superclass(node: Node, source: &[u8]) -> Option<String> {
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            if child.kind() == "superclass" {
                let mut sc = child.walk();
                for sub in child.children(&mut sc) {
                    match sub.kind() {
                        "type_identifier" => return Some(node_text(sub, source).to_string()),
                        "generic_type" => {
                            let mut ic = sub.walk();
                            for inner in sub.children(&mut ic) {
                                if inner.kind() == "type_identifier" {
                                    return Some(node_text(inner, source).to_string());
                                }
                            }
                        }
                        _ => {}
                    }
                }
            }
        }
        None
    }

    fn get_interfaces(node: Node, source: &[u8]) -> Vec<String> {
        let mut out = Vec::new();
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            if matches!(child.kind(), "super_interfaces" | "extends_interfaces") {
                let mut sc = child.walk();
                for sub in child.children(&mut sc) {
                    if sub.kind() == "type_list" {
                        let mut lc = sub.walk();
                        for t in sub.children(&mut lc) {
                            match t.kind() {
                                "type_identifier" => out.push(node_text(t, source).to_string()),
                                "generic_type" => {
                                    let mut ic = t.walk();
                                    for inner in t.children(&mut ic) {
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
                }
            }
        }
        out
    }

    fn get_enum_constants(node: Node, source: &[u8]) -> Vec<String> {
        let mut out = Vec::new();
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            if child.kind() == "enum_body" {
                let mut bc = child.walk();
                for sub in child.children(&mut bc) {
                    if sub.kind() == "enum_constant" {
                        if let Some(name) = Self::get_name(sub, source) {
                            out.push(name.to_string());
                        }
                    }
                }
            }
        }
        out
    }

    fn parse_method(
        node: Node,
        source: &[u8],
        module_path: &str,
        rel_path: &str,
        owner: Option<&str>,
    ) -> FunctionInfo {
        let name = Self::get_name(node, source)
            .unwrap_or("unknown")
            .to_string();
        let prefix = match owner {
            Some(o) => format!("{}.{}", module_path, o),
            None => module_path.to_string(),
        };
        let qualified_name = format!("{}.{}", prefix, name);
        let mut body: Option<Node> = None;
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            if matches!(child.kind(), "block" | "constructor_body") {
                body = Some(child);
                break;
            }
        }
        let mut metadata = crate::code_tree::models::MetadataMap::new();
        if Self::has_modifier(node, source, "static") {
            metadata.insert("is_static".into(), json!(true));
        }
        if Self::has_modifier(node, source, "abstract") {
            metadata.insert("is_abstract".into(), json!(true));
        }
        if Self::has_modifier(node, source, "native") {
            metadata.insert("is_ffi".into(), json!(true));
            metadata.insert("ffi_kind".into(), json!("jni"));
        }
        let calls = body
            .map(|b| Self::extract_calls(b, source))
            .unwrap_or_default();
        let parameters = Self::extract_parameters(node, source);
        let param_count = Some(
            parameters
                .iter()
                .filter(|p| p.kind != ParameterKind::Receiver)
                .count() as u32,
        );
        let (branch_count, max_nesting) = match body {
            Some(b) => {
                let (c, n) = compute_complexity(b, BRANCH_KINDS_JAVA, NESTED_SCOPES);
                (Some(c), Some(n))
            }
            None => (None, None),
        };
        let is_recursive = Some(calls.iter().any(|(n, _)| n == &name));
        let docstring = Self::get_doc_comment(node, source);
        let procedure_names = extract_procedure_annotations(docstring.as_deref());
        FunctionInfo {
            visibility: Self::get_visibility(node, source).to_string(),
            is_async: false,
            is_method: owner.is_some(),
            signature: Self::get_signature(node, source),
            file_path: rel_path.to_string(),
            line_number: node.start_position().row as u32 + 1,
            end_line: Some(node.end_position().row as u32 + 1),
            docstring,
            return_type: Self::get_return_type(node, source),
            decorators: Self::get_annotations(node, source),
            calls,
            references: Vec::new(),
            function_refs: Vec::new(),
            type_parameters: get_type_parameters(node, source, "type_parameters"),
            parameters,
            branch_count,
            param_count,
            max_nesting,
            is_recursive,
            procedure_names,
            metadata,
            qualified_name,
            name,
        }
    }

    /// Extract structured parameters from a Java method/constructor.
    /// Walks `formal_parameters` for `formal_parameter` and `spread_parameter`.
    fn extract_parameters(node: Node, source: &[u8]) -> Vec<ParameterInfo> {
        let mut out = Vec::new();
        let mut cursor = node.walk();
        let Some(params_node) = node
            .children(&mut cursor)
            .find(|c| c.kind() == "formal_parameters")
        else {
            return out;
        };
        let mut pcursor = params_node.walk();
        for child in params_node.children(&mut pcursor) {
            let (is_variadic, kind) = match child.kind() {
                "formal_parameter" => (false, "formal_parameter"),
                "spread_parameter" => (true, "spread_parameter"),
                _ => continue,
            };
            let _ = kind;
            let mut name: Option<String> = None;
            let mut type_ann: Option<String> = None;
            let mut tcursor = child.walk();
            for sub in child.children(&mut tcursor) {
                match sub.kind() {
                    "identifier" if name.is_none() => {
                        name = Some(node_text(sub, source).to_string())
                    }
                    k if k.contains("type")
                        || k == "generic_type"
                        || k == "array_type"
                        || k == "scoped_type_identifier" =>
                    {
                        type_ann = Some(node_text(sub, source).to_string());
                    }
                    _ => {}
                }
            }
            let Some(n) = name else { continue };
            out.push(ParameterInfo {
                name: n,
                type_annotation: type_ann,
                default: None,
                kind: if is_variadic {
                    ParameterKind::Variadic
                } else {
                    ParameterKind::Positional
                },
            });
        }
        out
    }

    #[allow(clippy::too_many_arguments)]
    fn parse_class(
        node: Node,
        source: &[u8],
        module_path: &str,
        rel_path: &str,
        result: &mut ParseResult,
        outer_name: Option<&str>,
    ) {
        let name = Self::get_name(node, source)
            .unwrap_or("unknown")
            .to_string();
        let qualified_name = match outer_name {
            Some(o) => format!("{}.{}.{}", module_path, o, name),
            None => format!("{}.{}", module_path, name),
        };
        let superclass = Self::get_superclass(node, source);
        let interfaces = Self::get_interfaces(node, source);
        let annotations = Self::get_annotations(node, source);
        let docstring = Self::get_doc_comment(node, source);
        let bases = superclass
            .as_ref()
            .map(|s| vec![s.clone()])
            .unwrap_or_default();
        let mut metadata = crate::code_tree::models::MetadataMap::new();
        if !annotations.is_empty() {
            metadata.insert("decorators".into(), json!(annotations));
        }
        if Self::has_modifier(node, source, "abstract") {
            metadata.insert("is_abstract".into(), json!(true));
        }

        result.classes.push(ClassInfo {
            qualified_name: qualified_name.clone(),
            kind: "class".into(),
            visibility: Self::get_visibility(node, source).to_string(),
            file_path: rel_path.to_string(),
            line_number: node.start_position().row as u32 + 1,
            end_line: Some(node.end_position().row as u32 + 1),
            docstring,
            bases,
            type_parameters: get_type_parameters(node, source, "type_parameters"),
            metadata,
            name: name.clone(),
        });

        if let Some(sc) = &superclass {
            result.type_relationships.push(TypeRelationship {
                source_type: name.clone(),
                target_type: Some(sc.clone()),
                relationship: "extends".into(),
                methods: Vec::new(),
            });
        }
        for iface in &interfaces {
            result.type_relationships.push(TypeRelationship {
                source_type: name.clone(),
                target_type: Some(iface.clone()),
                relationship: "implements".into(),
                methods: Vec::new(),
            });
        }
        Self::parse_class_body(
            node,
            source,
            module_path,
            rel_path,
            &name,
            &qualified_name,
            result,
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn parse_class_body(
        node: Node,
        source: &[u8],
        module_path: &str,
        rel_path: &str,
        class_name: &str,
        class_qname: &str,
        result: &mut ParseResult,
    ) {
        let mut method_rel = TypeRelationship {
            source_type: class_qname.to_string(),
            target_type: None,
            relationship: "inherent".into(),
            methods: Vec::new(),
        };
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            if child.kind() != "class_body" {
                continue;
            }
            let mut bc = child.walk();
            for item in child.children(&mut bc) {
                match item.kind() {
                    "method_declaration" | "constructor_declaration" => {
                        let fn_info = Self::parse_method(
                            item,
                            source,
                            module_path,
                            rel_path,
                            Some(class_name),
                        );
                        method_rel.methods.push(fn_info.clone());
                        result.functions.push(fn_info);
                    }
                    "field_declaration" => {
                        Self::parse_field(item, source, rel_path, class_qname, result);
                    }
                    "class_declaration" => {
                        Self::parse_class(
                            item,
                            source,
                            module_path,
                            rel_path,
                            result,
                            Some(class_name),
                        );
                    }
                    "interface_declaration" => {
                        Self::parse_interface(item, source, module_path, rel_path, result);
                    }
                    "enum_declaration" => {
                        Self::parse_enum(item, source, module_path, rel_path, result);
                    }
                    _ => {}
                }
            }
        }
        if !method_rel.methods.is_empty() {
            result.type_relationships.push(method_rel);
        }
    }

    fn parse_field(
        node: Node,
        source: &[u8],
        rel_path: &str,
        class_qname: &str,
        result: &mut ParseResult,
    ) {
        let is_static = Self::has_modifier(node, source, "static");
        let is_final = Self::has_modifier(node, source, "final");
        let visibility = Self::get_visibility(node, source).to_string();
        let mut type_ann: Option<String> = None;
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            if TYPE_NODES.contains(&child.kind()) {
                type_ann = Some(node_text(child, source).to_string());
                break;
            }
        }
        let mut cursor2 = node.walk();
        for child in node.children(&mut cursor2) {
            if child.kind() != "variable_declarator" {
                continue;
            }
            let Some(name) = Self::get_name(child, source) else {
                continue;
            };
            let name = name.to_string();
            let mut val_text: Option<String> = None;
            let mut sc = child.walk();
            for sub in child.children(&mut sc) {
                if !matches!(sub.kind(), "identifier" | "dimensions") && sub.is_named() {
                    let text = node_text(sub, source);
                    let take = text
                        .char_indices()
                        .nth(100)
                        .map(|(i, _)| i)
                        .unwrap_or(text.len());
                    val_text = Some(text[..take].to_string());
                    break;
                }
            }
            if is_static && is_final {
                result.constants.push(ConstantInfo {
                    qualified_name: format!("{}.{}", class_qname, name),
                    kind: "constant".into(),
                    type_annotation: type_ann.clone(),
                    value_preview: val_text,
                    visibility: visibility.clone(),
                    file_path: rel_path.to_string(),
                    line_number: node.start_position().row as u32 + 1,
                    name,
                });
            } else {
                result.attributes.push(AttributeInfo {
                    qualified_name: format!("{}.{}", class_qname, name),
                    owner_qualified_name: class_qname.to_string(),
                    type_annotation: type_ann.clone(),
                    visibility: visibility.clone(),
                    file_path: rel_path.to_string(),
                    line_number: node.start_position().row as u32 + 1,
                    default_value: val_text,
                    name,
                });
            }
        }
    }

    fn parse_interface(
        node: Node,
        source: &[u8],
        module_path: &str,
        rel_path: &str,
        result: &mut ParseResult,
    ) {
        let name = Self::get_name(node, source)
            .unwrap_or("unknown")
            .to_string();
        let qualified_name = format!("{}.{}", module_path, name);
        let extends = Self::get_interfaces(node, source);
        let docstring = Self::get_doc_comment(node, source);
        result.interfaces.push(InterfaceInfo {
            qualified_name: qualified_name.clone(),
            kind: "interface".into(),
            visibility: Self::get_visibility(node, source).to_string(),
            file_path: rel_path.to_string(),
            line_number: node.start_position().row as u32 + 1,
            end_line: Some(node.end_position().row as u32 + 1),
            docstring,
            type_parameters: get_type_parameters(node, source, "type_parameters"),
            name: name.clone(),
        });
        for base in &extends {
            result.type_relationships.push(TypeRelationship {
                source_type: name.clone(),
                target_type: Some(base.clone()),
                relationship: "extends".into(),
                methods: Vec::new(),
            });
        }
        let mut iface_rel = TypeRelationship {
            source_type: qualified_name,
            target_type: None,
            relationship: "inherent".into(),
            methods: Vec::new(),
        };
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            if child.kind() == "interface_body" {
                let mut bc = child.walk();
                for item in child.children(&mut bc) {
                    if item.kind() == "method_declaration" {
                        let fn_info =
                            Self::parse_method(item, source, module_path, rel_path, Some(&name));
                        iface_rel.methods.push(fn_info.clone());
                        result.functions.push(fn_info);
                    }
                }
            }
        }
        if !iface_rel.methods.is_empty() {
            result.type_relationships.push(iface_rel);
        }
    }

    fn parse_enum(
        node: Node,
        source: &[u8],
        module_path: &str,
        rel_path: &str,
        result: &mut ParseResult,
    ) {
        let name = Self::get_name(node, source)
            .unwrap_or("unknown")
            .to_string();
        result.enums.push(EnumInfo {
            qualified_name: format!("{}.{}", module_path, name),
            visibility: Self::get_visibility(node, source).to_string(),
            file_path: rel_path.to_string(),
            line_number: node.start_position().row as u32 + 1,
            end_line: Some(node.end_position().row as u32 + 1),
            docstring: Self::get_doc_comment(node, source),
            variants: Self::get_enum_constants(node, source),
            variant_details: None,
            name,
        });
    }
}

impl Default for JavaParser {
    fn default() -> Self {
        Self::new()
    }
}

impl LanguageParser for JavaParser {
    fn language_name(&self) -> &'static str {
        "java"
    }
    fn file_extensions(&self) -> &'static [&'static str] {
        &["java"]
    }
    fn noise_names(&self) -> &'static [&'static str] {
        JAVA_NOISE_NAMES
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
        let is_test = stem.ends_with("Test")
            || stem.ends_with("Tests")
            || rel_path.contains("/src/test/")
            || rel_path.starts_with("src/test/");

        if let Some(reason) = is_generated_or_minified(&source) {
            let module_path = stem.clone();
            let mut r = ParseResult::new();
            r.files.push(FileInfo {
                path: rel_path,
                filename,
                loc,
                module_path,
                language: "java".to_string(),
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
        let package = Self::get_package(root, &source);
        let module_path = Self::file_to_module_path(filepath, src_root, &package);
        let mut file_info = FileInfo {
            path: rel_path.clone(),
            filename,
            loc,
            module_path: module_path.clone(),
            language: "java".to_string(),
            submodule_declarations: Vec::new(),
            imports: Vec::new(),
            exports: Vec::new(),
            annotations: None,
            is_test,
            skip_reason: None,
        };
        let mut result = ParseResult::new();
        let mut cursor = root.walk();
        for child in root.children(&mut cursor) {
            match child.kind() {
                "class_declaration" => {
                    Self::parse_class(child, &source, &module_path, &rel_path, &mut result, None)
                }
                "interface_declaration" => {
                    Self::parse_interface(child, &source, &module_path, &rel_path, &mut result)
                }
                "enum_declaration" => {
                    Self::parse_enum(child, &source, &module_path, &rel_path, &mut result)
                }
                "import_declaration" => {
                    let mut ic = child.walk();
                    for sub in child.children(&mut ic) {
                        if sub.kind() == "scoped_identifier" {
                            file_info.imports.push(node_text(sub, &source).to_string());
                            break;
                        }
                    }
                }
                _ => {}
            }
        }
        file_info.annotations = extract_comment_annotations(root, &source, DEFAULT_COMMENT_TYPES);
        result.files.push(file_info);
        result
    }
}
