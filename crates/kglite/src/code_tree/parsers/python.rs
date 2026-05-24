//! Python language parser (ported from parsers/python.py).

use regex::Regex;
use serde_json::json;
use std::path::Path;
use std::sync::OnceLock;
use tree_sitter::{Node, Parser, Tree};

use super::shared::{
    compute_complexity, count_lines, extract_comment_annotations, extract_procedure_annotations,
    get_type_parameters, is_generated_or_minified, node_text, BRANCH_KINDS_PYTHON,
    DEFAULT_COMMENT_TYPES,
};
use super::LanguageParser;
use crate::code_tree::models::{
    AttributeInfo, ClassInfo, ConstantInfo, EnumInfo, FileInfo, FunctionInfo, InterfaceInfo,
    ParameterInfo, ParameterKind, ParseResult, TypeRelationship,
};

const ENUM_BASES: &[&str] = &["Enum", "IntEnum", "StrEnum", "Flag", "IntFlag", "auto"];
const PROTOCOL_BASES: &[&str] = &["Protocol"];

/// Identifier is ALL_CAPS (module-level constant).
fn constant_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"^[A-Z][A-Z_0-9]+$").expect("constant regex compiles"))
}

/// Framework-specific identifiers that break SCREAMING_SNAKE convention
/// but warrant treatment as top-level constants. Currently:
///   - `urlpatterns` — the canonical Django URL-routing list. Picked up
///     by the route-extractor pass.
fn is_framework_constant(name: &str) -> bool {
    matches!(name, "urlpatterns")
}

pub const PYTHON_NOISE_NAMES: &[&str] = &[
    "len",
    "str",
    "int",
    "float",
    "bool",
    "list",
    "dict",
    "set",
    "tuple",
    "print",
    "isinstance",
    "issubclass",
    "type",
    "range",
    "enumerate",
    "zip",
    "map",
    "filter",
    "sorted",
    "reversed",
    "any",
    "all",
    "min",
    "max",
    "sum",
    "abs",
    "round",
    "hash",
    "id",
    "repr",
    "super",
    "getattr",
    "setattr",
    "hasattr",
    "delattr",
    "callable",
    "iter",
    "next",
    "open",
    "format",
    "append",
    "extend",
    "update",
    "pop",
    "get",
    "keys",
    "values",
    "items",
    "join",
    "split",
    "strip",
    "replace",
    "startswith",
    "endswith",
];

const NESTED_SCOPES: &[&str] = &["function_definition", "lambda", "decorated_definition"];

pub struct PythonParser;

thread_local! {
    static PY_PARSER: std::cell::RefCell<Parser> = {
        let mut p = Parser::new();
        p.set_language(&tree_sitter_python::LANGUAGE.into())
            .expect("loading tree-sitter-python grammar");
        std::cell::RefCell::new(p)
    };
}

impl PythonParser {
    pub fn new() -> Self {
        PythonParser
    }

    fn parse_tree(&self, source: &[u8]) -> Option<Tree> {
        PY_PARSER.with(|p| p.borrow_mut().parse(source, None))
    }

    // ── Small helpers ───────────────────────────────────────────────

    fn get_visibility(name: &str) -> &'static str {
        if name.starts_with("__") && !name.ends_with("__") {
            "private"
        } else if name.starts_with('_') {
            "private"
        } else {
            "public"
        }
    }

    fn get_name<'a>(node: Node<'a>, source: &'a [u8]) -> &'a str {
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            if child.kind() == "identifier" {
                return node_text(child, source);
            }
        }
        "unknown"
    }

    fn get_block<'a>(node: Node<'a>) -> Option<Node<'a>> {
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            if child.kind() == "block" {
                return Some(child);
            }
        }
        None
    }

    fn get_docstring(node: Node, source: &[u8]) -> Option<String> {
        let block = Self::get_block(node)?;
        let mut cursor = block.walk();
        for child in block.children(&mut cursor) {
            match child.kind() {
                "expression_statement" => {
                    let mut sub_cursor = child.walk();
                    for sub in child.children(&mut sub_cursor) {
                        if sub.kind() == "string" {
                            let raw = node_text(sub, source);
                            for delim in ["\"\"\"", "'''", "\"", "'"] {
                                if raw.starts_with(delim) && raw.ends_with(delim) {
                                    let inner = &raw[delim.len()..raw.len() - delim.len()];
                                    return Some(inner.trim().to_string());
                                }
                            }
                            return Some(raw.to_string());
                        }
                    }
                    return None;
                }
                "comment" => continue,
                _ => return None,
            }
        }
        None
    }

    fn get_bases(node: Node, source: &[u8]) -> Vec<String> {
        let mut bases = Vec::new();
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            if child.kind() != "argument_list" {
                continue;
            }
            let mut arg_cursor = child.walk();
            for arg in child.children(&mut arg_cursor) {
                match arg.kind() {
                    "identifier" | "attribute" => {
                        bases.push(node_text(arg, source).to_string());
                    }
                    "subscript" => {
                        // e.g. Generic[T], Protocol[T] → take the base name
                        let mut sub_cursor = arg.walk();
                        for sub in arg.children(&mut sub_cursor) {
                            if matches!(sub.kind(), "identifier" | "attribute") {
                                bases.push(node_text(sub, source).to_string());
                                break;
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
        bases
    }

    fn get_decorators(decorated: Node, source: &[u8]) -> Vec<String> {
        let mut out = Vec::new();
        let mut cursor = decorated.walk();
        for child in decorated.children(&mut cursor) {
            if child.kind() == "decorator" {
                let text = node_text(child, source).trim();
                let stripped = text.strip_prefix('@').unwrap_or(text);
                out.push(stripped.to_string());
            }
        }
        out
    }

    fn get_decorated_inner<'a>(node: Node<'a>) -> Option<Node<'a>> {
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            if matches!(child.kind(), "function_definition" | "class_definition") {
                return Some(child);
            }
        }
        None
    }

    fn get_signature(node: Node, source: &[u8]) -> String {
        let mut parts: Vec<&str> = Vec::new();
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            match child.kind() {
                "block" => break,
                "comment" => continue,
                _ => parts.push(node_text(child, source)),
            }
        }
        parts
            .join(" ")
            .trim_end_matches(|c: char| c == ' ' || c == ':')
            .to_string()
    }

    fn get_return_type(node: Node, source: &[u8]) -> Option<String> {
        let mut saw_arrow = false;
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            if !child.is_named() && node_text(child, source) == "->" {
                saw_arrow = true;
            } else if saw_arrow {
                match child.kind() {
                    ":" | "block" => return None,
                    _ => return Some(node_text(child, source).to_string()),
                }
            }
        }
        None
    }

    fn is_async(node: Node, source: &[u8]) -> bool {
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            if !child.is_named() {
                let text = node_text(child, source);
                if text == "async" {
                    return true;
                }
                if text == "def" {
                    break;
                }
            }
        }
        false
    }

    /// Extract structured parameters from a Python function definition.
    /// Excludes implicit `self`/`cls`. Distinguishes `*args` (Variadic) and
    /// `**kwargs` (KwVariadic). `default` carries the raw default expression text.
    fn extract_parameters(node: Node, source: &[u8]) -> Vec<ParameterInfo> {
        let mut out = Vec::new();
        let mut cursor = node.walk();
        let Some(params_node) = node
            .children(&mut cursor)
            .find(|c| c.kind() == "parameters")
        else {
            return out;
        };
        let mut pcursor = params_node.walk();
        for child in params_node.children(&mut pcursor) {
            let kind = child.kind();
            let (name, type_ann, default, param_kind) = match kind {
                "identifier" => {
                    let n = node_text(child, source).to_string();
                    if matches!(n.as_str(), "self" | "cls") {
                        continue;
                    }
                    (n, None, None, ParameterKind::Positional)
                }
                "typed_parameter" => {
                    let mut name: Option<String> = None;
                    let mut type_ann: Option<String> = None;
                    let mut tcursor = child.walk();
                    for sub in child.children(&mut tcursor) {
                        match sub.kind() {
                            "identifier" if name.is_none() => {
                                name = Some(node_text(sub, source).to_string())
                            }
                            "type" => type_ann = Some(node_text(sub, source).to_string()),
                            _ => {}
                        }
                    }
                    let Some(n) = name else { continue };
                    if matches!(n.as_str(), "self" | "cls") {
                        continue;
                    }
                    (n, type_ann, None, ParameterKind::Positional)
                }
                "default_parameter" | "typed_default_parameter" => {
                    let mut name: Option<String> = None;
                    let mut type_ann: Option<String> = None;
                    let mut default: Option<String> = None;
                    let mut saw_eq = false;
                    let mut dcursor = child.walk();
                    for sub in child.children(&mut dcursor) {
                        if !sub.is_named() && node_text(sub, source) == "=" {
                            saw_eq = true;
                            continue;
                        }
                        match sub.kind() {
                            "identifier" if name.is_none() && !saw_eq => {
                                name = Some(node_text(sub, source).to_string())
                            }
                            "type" if !saw_eq => {
                                type_ann = Some(node_text(sub, source).to_string())
                            }
                            _ if saw_eq && sub.is_named() && default.is_none() => {
                                default = Some(node_text(sub, source).to_string());
                            }
                            _ => {}
                        }
                    }
                    let Some(n) = name else { continue };
                    if matches!(n.as_str(), "self" | "cls") {
                        continue;
                    }
                    (n, type_ann, default, ParameterKind::Positional)
                }
                "list_splat_pattern" => {
                    // *args
                    let n = node_text(child, source).trim_start_matches('*').to_string();
                    if n.is_empty() {
                        continue;
                    }
                    (n, None, None, ParameterKind::Variadic)
                }
                "dictionary_splat_pattern" => {
                    // **kwargs
                    let n = node_text(child, source).trim_start_matches('*').to_string();
                    if n.is_empty() {
                        continue;
                    }
                    (n, None, None, ParameterKind::KwVariadic)
                }
                _ => continue,
            };
            out.push(ParameterInfo {
                name,
                type_annotation: type_ann,
                default,
                kind: param_kind,
            });
        }
        out
    }

    fn extract_calls(body: Node, source: &[u8]) -> Vec<(String, u32)> {
        let mut calls: Vec<(String, u32)> = Vec::new();
        fn walk(node: Node, source: &[u8], out: &mut Vec<(String, u32)>) {
            if node.kind() == "call" {
                let line = node.start_position().row as u32 + 1;
                if let Some(func) = node.child(0) {
                    match func.kind() {
                        "identifier" => {
                            out.push((node_text(func, source).to_string(), line));
                        }
                        "attribute" => {
                            let mut parts: Vec<&str> = Vec::new();
                            let mut cursor = func.walk();
                            for child in func.children(&mut cursor) {
                                if child.kind() == "identifier" {
                                    parts.push(node_text(child, source));
                                }
                            }
                            if parts.len() >= 2 {
                                let receiver = parts[parts.len() - 2];
                                let method = parts[parts.len() - 1];
                                if receiver == "self" || receiver == "cls" {
                                    out.push((method.to_string(), line));
                                } else {
                                    out.push((format!("{}.{}", receiver, method), line));
                                }
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
            if let Some(stem) = last.strip_suffix(".pyi") {
                *last = stem.to_string();
            } else if let Some(stem) = last.strip_suffix(".py") {
                *last = stem.to_string();
            }
            if last == "__init__" {
                parts.pop();
            }
        }
        let pkg = src_root.file_name().and_then(|o| o.to_str()).unwrap_or("");
        if parts.is_empty() {
            pkg.to_string()
        } else if pkg.is_empty() {
            parts.join(".")
        } else {
            format!("{}.{}", pkg, parts.join("."))
        }
    }

    fn get_enum_variants(node: Node, source: &[u8]) -> Vec<String> {
        let mut out = Vec::new();
        let Some(block) = Self::get_block(node) else {
            return out;
        };
        let mut cursor = block.walk();
        for child in block.children(&mut cursor) {
            if child.kind() == "expression_statement" {
                let mut sub_cursor = child.walk();
                for sub in child.children(&mut sub_cursor) {
                    if sub.kind() == "assignment" {
                        let mut tgt_cursor = sub.walk();
                        for target in sub.children(&mut tgt_cursor) {
                            if target.kind() == "identifier" {
                                out.push(node_text(target, source).to_string());
                                break;
                            }
                        }
                    }
                }
            }
        }
        out
    }

    fn parse_import(node: Node, source: &[u8]) -> Option<String> {
        let mut cursor = node.walk();
        match node.kind() {
            "import_statement" => {
                for child in node.children(&mut cursor) {
                    if child.kind() == "dotted_name" {
                        return Some(node_text(child, source).to_string());
                    }
                }
                None
            }
            "import_from_statement" => {
                for child in node.children(&mut cursor) {
                    match child.kind() {
                        "dotted_name" => return Some(node_text(child, source).to_string()),
                        "relative_import" => return None,
                        _ => {}
                    }
                }
                None
            }
            _ => None,
        }
    }

    fn classify_decorators(decorators: &[String]) -> Vec<(&'static str, serde_json::Value)> {
        let mut flags: Vec<(&'static str, serde_json::Value)> = Vec::new();
        for dec in decorators {
            let base = dec.split('(').next().unwrap_or("");
            let base = base.rsplit('.').next().unwrap_or("");
            let flag = match base {
                "abstractmethod" => "is_abstract",
                "property" => "is_property",
                "staticmethod" => "is_static",
                "classmethod" => "is_classmethod",
                "overload" => "is_overload",
                _ => continue,
            };
            flags.push((flag, json!(true)));
        }
        flags
    }

    // ── Class body extraction ──────────────────────────────────────

    fn extract_class_attributes(
        class_node: Node,
        source: &[u8],
        owner_qname: &str,
        rel_path: &str,
        result: &mut ParseResult,
    ) {
        let Some(block) = Self::get_block(class_node) else {
            return;
        };

        // 1. Class-body assignments: x = value, x: type = value
        let mut cursor = block.walk();
        for child in block.children(&mut cursor) {
            if child.kind() != "expression_statement" {
                continue;
            }
            let mut sub_cursor = child.walk();
            for sub in child.children(&mut sub_cursor) {
                if sub.kind() != "assignment" {
                    continue;
                }
                let mut attr_name: Option<String> = None;
                let mut type_ann: Option<String> = None;
                let mut default_val: Option<String> = None;

                let mut scan = sub.walk();
                for sc in sub.children(&mut scan) {
                    if sc.kind() == "identifier" && attr_name.is_none() {
                        attr_name = Some(node_text(sc, source).to_string());
                    } else if sc.kind() == "type" {
                        type_ann = Some(node_text(sc, source).to_string());
                    }
                }

                let Some(attr_name) = attr_name else {
                    continue;
                };
                if constant_re().is_match(&attr_name) {
                    continue;
                }

                let mut saw_eq = false;
                let mut scan2 = sub.walk();
                for sc in sub.children(&mut scan2) {
                    if !sc.is_named() && node_text(sc, source) == "=" {
                        saw_eq = true;
                    } else if saw_eq && sc.is_named() {
                        let val = node_text(sc, source);
                        let take = val
                            .char_indices()
                            .nth(100)
                            .map(|(i, _)| i)
                            .unwrap_or(val.len());
                        default_val = Some(val[..take].to_string());
                        break;
                    }
                }

                result.attributes.push(AttributeInfo {
                    qualified_name: format!("{}.{}", owner_qname, attr_name),
                    owner_qualified_name: owner_qname.to_string(),
                    visibility: Self::get_visibility(&attr_name).to_string(),
                    name: attr_name,
                    type_annotation: type_ann,
                    file_path: rel_path.to_string(),
                    line_number: child.start_position().row as u32 + 1,
                    default_value: default_val,
                });
            }
        }

        // 2. self.x assignments in __init__
        let mut seen_names: std::collections::HashSet<String> = result
            .attributes
            .iter()
            .filter(|a| a.owner_qualified_name == owner_qname)
            .map(|a| a.name.clone())
            .collect();

        let mut cursor2 = block.walk();
        for child in block.children(&mut cursor2) {
            let fn_node = match child.kind() {
                "function_definition" => Some(child),
                "decorated_definition" => {
                    Self::get_decorated_inner(child).filter(|n| n.kind() == "function_definition")
                }
                _ => None,
            };
            if let Some(fn_node) = fn_node {
                if Self::get_name(fn_node, source) == "__init__" {
                    if let Some(init_block) = Self::get_block(fn_node) {
                        Self::walk_self_attrs(
                            init_block,
                            source,
                            owner_qname,
                            rel_path,
                            result,
                            &mut seen_names,
                        );
                    }
                    break;
                }
            }
        }
    }

    fn walk_self_attrs(
        node: Node,
        source: &[u8],
        owner_qname: &str,
        rel_path: &str,
        result: &mut ParseResult,
        seen_names: &mut std::collections::HashSet<String>,
    ) {
        if node.kind() == "assignment" {
            if let Some(left) = node.child(0) {
                if left.kind() == "attribute" {
                    let text = node_text(left, source);
                    if let Some(attr_name) = text.strip_prefix("self.") {
                        if !attr_name.contains('.') && !seen_names.contains(attr_name) {
                            let owned = attr_name.to_string();
                            seen_names.insert(owned.clone());
                            let mut default_val: Option<String> = None;
                            let mut saw_eq = false;
                            let mut cursor = node.walk();
                            for sc in node.children(&mut cursor) {
                                if !sc.is_named() && node_text(sc, source) == "=" {
                                    saw_eq = true;
                                } else if saw_eq && sc.is_named() {
                                    let val = node_text(sc, source);
                                    let take = val
                                        .char_indices()
                                        .nth(100)
                                        .map(|(i, _)| i)
                                        .unwrap_or(val.len());
                                    default_val = Some(val[..take].to_string());
                                    break;
                                }
                            }
                            result.attributes.push(AttributeInfo {
                                qualified_name: format!("{}.{}", owner_qname, owned),
                                owner_qualified_name: owner_qname.to_string(),
                                visibility: Self::get_visibility(&owned).to_string(),
                                name: owned,
                                type_annotation: None,
                                file_path: rel_path.to_string(),
                                line_number: node.start_position().row as u32 + 1,
                                default_value: default_val,
                            });
                        }
                    }
                }
            }
        }
        let mut cursor = node.walk();
        for c in node.children(&mut cursor) {
            Self::walk_self_attrs(c, source, owner_qname, rel_path, result, seen_names);
        }
    }

    // ── Top-level parsers ───────────────────────────────────────────

    fn parse_function(
        node: Node,
        source: &[u8],
        module_path: &str,
        rel_path: &str,
        is_method: bool,
        owner: Option<&str>,
    ) -> FunctionInfo {
        let name = Self::get_name(node, source).to_string();
        let prefix = match owner {
            Some(o) => format!("{}.{}", module_path, o),
            None => module_path.to_string(),
        };
        let qualified_name = format!("{}.{}", prefix, name);
        let block = Self::get_block(node);
        let calls = block
            .map(|b| Self::extract_calls(b, source))
            .unwrap_or_default();
        let parameters = Self::extract_parameters(node, source);
        let param_count = Some(
            parameters
                .iter()
                .filter(|p| p.kind != ParameterKind::Receiver)
                .count() as u32,
        );
        let (branch_count, max_nesting) = match block {
            Some(b) => {
                let (c, n) = compute_complexity(b, BRANCH_KINDS_PYTHON, NESTED_SCOPES);
                (Some(c), Some(n))
            }
            None => (None, None),
        };
        let is_recursive = Some(calls.iter().any(|(n, _)| n == &name));
        let docstring = Self::get_docstring(node, source);
        let procedure_names = extract_procedure_annotations(docstring.as_deref());

        FunctionInfo {
            visibility: Self::get_visibility(&name).to_string(),
            name,
            qualified_name,
            is_async: Self::is_async(node, source),
            is_method,
            signature: Self::get_signature(node, source),
            file_path: rel_path.to_string(),
            line_number: node.start_position().row as u32 + 1,
            end_line: Some(node.end_position().row as u32 + 1),
            docstring,
            return_type: Self::get_return_type(node, source),
            calls,
            references: Vec::new(),
            function_refs: Vec::new(),
            type_parameters: get_type_parameters(node, source, "type_parameter"),
            decorators: Vec::new(),
            parameters,
            branch_count,
            param_count,
            max_nesting,
            is_recursive,
            procedure_names,
            metadata: Default::default(),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn parse_class(
        node: Node,
        source: &[u8],
        module_path: &str,
        rel_path: &str,
        result: &mut ParseResult,
        decorators: Option<Vec<String>>,
    ) {
        let name = Self::get_name(node, source).to_string();
        let qualified_name = format!("{}.{}", module_path, name);
        let bases = Self::get_bases(node, source);
        let docstring = Self::get_docstring(node, source);

        let is_enum = bases.iter().any(|b| ENUM_BASES.contains(&b.as_str()));
        let is_protocol = bases.iter().any(|b| b == "Protocol");

        if is_enum {
            result.enums.push(EnumInfo {
                visibility: Self::get_visibility(&name).to_string(),
                name: name.clone(),
                qualified_name: qualified_name.clone(),
                file_path: rel_path.to_string(),
                line_number: node.start_position().row as u32 + 1,
                end_line: Some(node.end_position().row as u32 + 1),
                docstring,
                variants: Self::get_enum_variants(node, source),
                variant_details: None,
            });
            return;
        }

        if is_protocol {
            result.interfaces.push(InterfaceInfo {
                visibility: Self::get_visibility(&name).to_string(),
                name: name.clone(),
                qualified_name: qualified_name.clone(),
                kind: "protocol".to_string(),
                file_path: rel_path.to_string(),
                line_number: node.start_position().row as u32 + 1,
                end_line: Some(node.end_position().row as u32 + 1),
                docstring,
                type_parameters: get_type_parameters(node, source, "type_parameter"),
            });
        } else {
            let mut metadata = crate::code_tree::models::MetadataMap::new();
            let decs = decorators.clone().unwrap_or_default();
            metadata.insert("decorators".to_string(), json!(decs));
            result.classes.push(ClassInfo {
                visibility: Self::get_visibility(&name).to_string(),
                name: name.clone(),
                qualified_name: qualified_name.clone(),
                kind: "class".to_string(),
                file_path: rel_path.to_string(),
                line_number: node.start_position().row as u32 + 1,
                end_line: Some(node.end_position().row as u32 + 1),
                docstring,
                bases: bases.clone(),
                type_parameters: get_type_parameters(node, source, "type_parameter"),
                metadata,
            });
        }

        // Inheritance edges
        for base in bases.iter().filter(|b| b.as_str() != "Protocol") {
            result.type_relationships.push(TypeRelationship {
                source_type: name.clone(),
                target_type: Some(base.clone()),
                relationship: "extends".to_string(),
                methods: Vec::new(),
            });
        }

        // Methods in the class body
        let mut method_rel = TypeRelationship {
            source_type: qualified_name.clone(),
            target_type: None,
            relationship: "inherent".to_string(),
            methods: Vec::new(),
        };

        if let Some(block) = Self::get_block(node) {
            let mut cursor = block.walk();
            for child in block.children(&mut cursor) {
                let (fn_node, fn_decorators): (Option<Node>, Vec<String>) = match child.kind() {
                    "function_definition" => (Some(child), Vec::new()),
                    "decorated_definition" => {
                        if let Some(inner) = Self::get_decorated_inner(child) {
                            match inner.kind() {
                                "function_definition" => {
                                    (Some(inner), Self::get_decorators(child, source))
                                }
                                "class_definition" => {
                                    Self::parse_class(
                                        inner,
                                        source,
                                        &qualified_name,
                                        rel_path,
                                        result,
                                        Some(Self::get_decorators(child, source)),
                                    );
                                    (None, Vec::new())
                                }
                                _ => (None, Vec::new()),
                            }
                        } else {
                            (None, Vec::new())
                        }
                    }
                    "class_definition" => {
                        Self::parse_class(child, source, &qualified_name, rel_path, result, None);
                        (None, Vec::new())
                    }
                    _ => (None, Vec::new()),
                };

                if let Some(fn_node) = fn_node {
                    let mut fn_info = Self::parse_function(
                        fn_node,
                        source,
                        module_path,
                        rel_path,
                        true,
                        Some(&name),
                    );
                    fn_info.decorators = fn_decorators.clone();
                    for (flag, val) in Self::classify_decorators(&fn_decorators) {
                        fn_info.metadata.insert(flag.to_string(), val);
                    }
                    method_rel.methods.push(fn_info.clone());
                    result.functions.push(fn_info);
                }
            }
        }

        if !method_rel.methods.is_empty() {
            result.type_relationships.push(method_rel);
        }

        Self::extract_class_attributes(node, source, &qualified_name, rel_path, result);
    }
}

impl Default for PythonParser {
    fn default() -> Self {
        Self::new()
    }
}

impl LanguageParser for PythonParser {
    fn language_name(&self) -> &'static str {
        "python"
    }

    fn file_extensions(&self) -> &'static [&'static str] {
        &["py", "pyi"]
    }

    fn noise_names(&self) -> &'static [&'static str] {
        PYTHON_NOISE_NAMES
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

        let is_test = filename.starts_with("test_")
            || stem.ends_with("_test")
            || rel_path.contains("/tests/")
            || rel_path.starts_with("tests/");

        if let Some(reason) = is_generated_or_minified(&source) {
            let mut r = ParseResult::new();
            r.files.push(FileInfo {
                path: rel_path,
                filename,
                loc,
                module_path,
                language: "python".to_string(),
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
            language: "python".to_string(),
            submodule_declarations: Vec::new(),
            imports: Vec::new(),
            exports: Vec::new(),
            annotations: None,
            is_test,
            skip_reason: None,
        };

        let mut result = ParseResult::new();

        let mut root_cursor = root.walk();
        for child in root.children(&mut root_cursor) {
            match child.kind() {
                "function_definition" => {
                    result.functions.push(Self::parse_function(
                        child,
                        &source,
                        &module_path,
                        &rel_path,
                        false,
                        None,
                    ));
                }
                "decorated_definition" => {
                    if let Some(inner) = Self::get_decorated_inner(child) {
                        match inner.kind() {
                            "function_definition" => {
                                let decorators = Self::get_decorators(child, &source);
                                let mut fn_info = Self::parse_function(
                                    inner,
                                    &source,
                                    &module_path,
                                    &rel_path,
                                    false,
                                    None,
                                );
                                fn_info.decorators = decorators.clone();
                                for (flag, val) in Self::classify_decorators(&decorators) {
                                    fn_info.metadata.insert(flag.to_string(), val);
                                }
                                result.functions.push(fn_info);
                            }
                            "class_definition" => {
                                let decs = Self::get_decorators(child, &source);
                                Self::parse_class(
                                    inner,
                                    &source,
                                    &module_path,
                                    &rel_path,
                                    &mut result,
                                    Some(decs),
                                );
                            }
                            _ => {}
                        }
                    }
                }
                "class_definition" => {
                    Self::parse_class(child, &source, &module_path, &rel_path, &mut result, None);
                }
                "import_statement" | "import_from_statement" => {
                    if let Some(imp) = Self::parse_import(child, &source) {
                        file_info.imports.push(imp);
                    }
                }
                "expression_statement" => {
                    let mut sub_cursor = child.walk();
                    for sub in child.children(&mut sub_cursor) {
                        if sub.kind() != "assignment" {
                            continue;
                        }
                        let mut first_id: Option<String> = None;
                        let mut scan = sub.walk();
                        for sc in sub.children(&mut scan) {
                            if sc.kind() == "identifier" {
                                first_id = Some(node_text(sc, &source).to_string());
                                break;
                            }
                        }
                        let Some(first_id) = first_id else { continue };

                        if first_id == "__all__" {
                            let mut scan2 = sub.walk();
                            for sc in sub.children(&mut scan2) {
                                if sc.kind() == "list" {
                                    let mut list_cursor = sc.walk();
                                    for item in sc.children(&mut list_cursor) {
                                        if item.kind() == "string" {
                                            let text = node_text(item, &source)
                                                .trim_matches(|c| c == '"' || c == '\'');
                                            file_info.exports.push(text.to_string());
                                        }
                                    }
                                }
                            }
                        } else if constant_re().is_match(&first_id)
                            || is_framework_constant(&first_id)
                        {
                            let mut type_ann: Option<String> = None;
                            let mut default_val: Option<String> = None;
                            let mut scan3 = sub.walk();
                            for sc in sub.children(&mut scan3) {
                                if sc.kind() == "type" {
                                    type_ann = Some(node_text(sc, &source).to_string());
                                }
                            }
                            let mut saw_eq = false;
                            let mut scan4 = sub.walk();
                            for sc in sub.children(&mut scan4) {
                                if !sc.is_named() && node_text(sc, &source) == "=" {
                                    saw_eq = true;
                                } else if saw_eq && sc.is_named() {
                                    let val = node_text(sc, &source);
                                    let take = val
                                        .char_indices()
                                        .nth(100)
                                        .map(|(i, _)| i)
                                        .unwrap_or(val.len());
                                    default_val = Some(val[..take].to_string());
                                    break;
                                }
                            }
                            result.constants.push(ConstantInfo {
                                qualified_name: format!("{}.{}", module_path, first_id),
                                visibility: Self::get_visibility(&first_id).to_string(),
                                name: first_id,
                                kind: "constant".to_string(),
                                type_annotation: type_ann,
                                value_preview: default_val,
                                file_path: rel_path.to_string(),
                                line_number: child.start_position().row as u32 + 1,
                            });
                        }
                    }
                }
                "type_alias_statement" => {
                    let mut alias_name: Option<String> = None;
                    let mut scan = child.walk();
                    for sc in child.children(&mut scan) {
                        if sc.kind() == "identifier" {
                            alias_name = Some(node_text(sc, &source).to_string());
                            break;
                        }
                    }
                    if let Some(alias_name) = alias_name {
                        let mut val_text: Option<String> = None;
                        let mut saw_eq = false;
                        let mut scan2 = child.walk();
                        for sc in child.children(&mut scan2) {
                            if !sc.is_named() && node_text(sc, &source) == "=" {
                                saw_eq = true;
                            } else if saw_eq && sc.is_named() {
                                let val = node_text(sc, &source);
                                let take = val
                                    .char_indices()
                                    .nth(100)
                                    .map(|(i, _)| i)
                                    .unwrap_or(val.len());
                                val_text = Some(val[..take].to_string());
                                break;
                            }
                        }
                        result.constants.push(ConstantInfo {
                            qualified_name: format!("{}.{}", module_path, alias_name),
                            visibility: Self::get_visibility(&alias_name).to_string(),
                            name: alias_name,
                            kind: "type_alias".to_string(),
                            type_annotation: val_text,
                            value_preview: None,
                            file_path: rel_path.to_string(),
                            line_number: child.start_position().row as u32 + 1,
                        });
                    }
                }
                _ => {}
            }
        }

        // Submodule declarations from __init__ files.
        if matches!(file_info.filename.as_str(), "__init__.py" | "__init__.pyi") {
            if let Some(parent) = filepath.parent() {
                let mut entries: Vec<_> = std::fs::read_dir(parent)
                    .ok()
                    .into_iter()
                    .flatten()
                    .filter_map(Result::ok)
                    .collect();
                entries.sort_by_key(|e| e.file_name());
                for entry in entries {
                    let name = entry.file_name();
                    let name_str = name.to_string_lossy();
                    let path = entry.path();
                    if path.is_dir() {
                        if path.join("__init__.py").exists() {
                            file_info.submodule_declarations.push(name_str.to_string());
                        }
                    } else if path.is_file() {
                        let ext = path.extension().and_then(|o| o.to_str()).unwrap_or("");
                        if (ext == "py" || ext == "pyi")
                            && name_str != "__init__.py"
                            && name_str != "__init__.pyi"
                        {
                            let stem = path
                                .file_stem()
                                .and_then(|o| o.to_str())
                                .unwrap_or("")
                                .to_string();
                            file_info.submodule_declarations.push(stem);
                        }
                    }
                }
            }
        }

        file_info.annotations = extract_comment_annotations(root, &source, DEFAULT_COMMENT_TYPES);
        result.files.push(file_info);

        result
    }
}
