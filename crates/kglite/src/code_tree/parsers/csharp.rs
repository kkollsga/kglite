//! C# language parser (ported from parsers/csharp.py).

use serde_json::json;
use std::path::Path;
use tree_sitter::{Node, Parser, Tree};

use super::shared::{
    compute_complexity, count_lines, extract_comment_annotations, extract_procedure_annotations,
    get_type_parameters, is_generated_or_minified, node_text, BRANCH_KINDS_CSHARP,
    DEFAULT_COMMENT_TYPES,
};
use super::LanguageParser;
use crate::code_tree::models::{
    AttributeInfo, ClassInfo, ConstantInfo, EnumInfo, FileInfo, FunctionInfo, InterfaceInfo,
    ParameterInfo, ParameterKind, ParseResult, TypeRelationship,
};

pub const CSHARP_NOISE_NAMES: &[&str] = &[
    "ToString",
    "Equals",
    "GetHashCode",
    "CompareTo",
    "GetType",
    "ReferenceEquals",
    "MemberwiseClone",
    "Count",
    "Add",
    "Remove",
    "Contains",
    "Clear",
    "Insert",
    "ContainsKey",
    "TryGetValue",
    "Keys",
    "Values",
    "IndexOf",
    "CopyTo",
    "Any",
    "All",
    "Select",
    "Where",
    "FirstOrDefault",
    "First",
    "LastOrDefault",
    "Last",
    "Single",
    "SingleOrDefault",
    "ToList",
    "ToArray",
    "ToDictionary",
    "OrderBy",
    "OrderByDescending",
    "GroupBy",
    "Sum",
    "Max",
    "Min",
    "Average",
    "Aggregate",
    "Write",
    "WriteLine",
    "ReadLine",
    "Read",
    "Format",
    "Close",
    "Dispose",
    "Flush",
];

const NESTED_SCOPES: &[&str] = &[
    "method_declaration",
    "constructor_declaration",
    "lambda_expression",
    "local_function_statement",
];

const TYPE_NODES: &[&str] = &[
    "predefined_type",
    "type_identifier",
    "generic_name",
    "nullable_type",
    "array_type",
    "qualified_name",
    "void_keyword",
];

const MODIFIER_NONTYPES: &[&str] = &[
    "public",
    "private",
    "protected",
    "internal",
    "static",
    "virtual",
    "override",
    "abstract",
    "async",
    "sealed",
    "partial",
];

pub struct CSharpParser;

thread_local! {
    static CS_PARSER: std::cell::RefCell<Parser> = {
        let mut p = Parser::new();
        p.set_language(&tree_sitter_c_sharp::LANGUAGE.into())
            .expect("loading tree-sitter-c-sharp grammar");
        std::cell::RefCell::new(p)
    };
}

impl CSharpParser {
    pub fn new() -> Self {
        CSharpParser
    }
    fn parse_tree(&self, source: &[u8]) -> Option<Tree> {
        CS_PARSER.with(|p| p.borrow_mut().parse(source, None))
    }

    fn get_visibility<'a>(node: Node, source: &[u8], default: &'a str) -> String {
        let mut mods: std::collections::HashSet<String> = std::collections::HashSet::new();
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            if child.kind() == "modifier" {
                let text = node_text(child, source).to_string();
                if matches!(
                    text.as_str(),
                    "public" | "private" | "protected" | "internal"
                ) {
                    return text;
                }
                mods.insert(text);
            }
        }
        if mods.contains("protected") && mods.contains("internal") {
            return "protected internal".into();
        }
        if mods.contains("private") && mods.contains("protected") {
            return "private protected".into();
        }
        default.into()
    }

    fn has_modifier(node: Node, source: &[u8], modifier: &str) -> bool {
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            if child.kind() == "modifier" && node_text(child, source) == modifier {
                return true;
            }
        }
        false
    }

    fn get_attributes(node: Node, source: &[u8]) -> Vec<String> {
        let mut out: Vec<String> = Vec::new();
        let mut sibling = node.prev_named_sibling();
        while let Some(s) = sibling {
            match s.kind() {
                "attribute_list" => {
                    let mut cursor = s.walk();
                    for sub in s.children(&mut cursor) {
                        if sub.kind() == "attribute" {
                            if let Some(name) = Self::get_name(sub, source) {
                                out.insert(0, name.to_string());
                            }
                        }
                    }
                    sibling = s.prev_named_sibling();
                }
                "comment" => sibling = s.prev_named_sibling(),
                _ => break,
            }
        }
        out
    }

    fn get_doc_comment(node: Node, source: &[u8]) -> Option<String> {
        let mut doc_lines: Vec<String> = Vec::new();
        let mut sibling = node.prev_named_sibling();
        while let Some(s) = sibling {
            match s.kind() {
                "comment" => {
                    let text = node_text(s, source).trim();
                    if let Some(rest) = text.strip_prefix("///") {
                        let content = rest.strip_prefix(' ').unwrap_or(rest);
                        doc_lines.insert(0, content.to_string());
                        sibling = s.prev_named_sibling();
                        continue;
                    }
                    break;
                }
                "attribute_list" => sibling = s.prev_named_sibling(),
                _ => break,
            }
        }
        if doc_lines.is_empty() {
            None
        } else {
            Some(doc_lines.join("\n"))
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
            if matches!(child.kind(), "block" | "arrow_expression_clause") {
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
                let text = node_text(child, source);
                if !MODIFIER_NONTYPES.contains(&text) {
                    return Some(text.to_string());
                }
            }
        }
        None
    }

    fn extract_calls(body: Node, source: &[u8]) -> Vec<(String, u32)> {
        let mut calls: Vec<(String, u32)> = Vec::new();
        fn walk(node: Node, source: &[u8], out: &mut Vec<(String, u32)>) {
            match node.kind() {
                "invocation_expression" => {
                    let line = node.start_position().row as u32 + 1;
                    if let Some(func) = node.child(0) {
                        match func.kind() {
                            "identifier" => {
                                out.push((node_text(func, source).to_string(), line));
                            }
                            "member_access_expression" => {
                                let name = func.child_by_field_name("name");
                                let expr = func.child_by_field_name("expression");
                                match (name, expr) {
                                    (Some(n), Some(e)) => {
                                        let method_name = node_text(n, source);
                                        let expr_text = node_text(e, source);
                                        let hint =
                                            expr_text.rsplit('.').next().unwrap_or(expr_text);
                                        if hint == "this" || hint == "base" {
                                            out.push((method_name.to_string(), line));
                                        } else {
                                            out.push((format!("{}.{}", hint, method_name), line));
                                        }
                                    }
                                    (Some(n), None) => {
                                        out.push((node_text(n, source).to_string(), line));
                                    }
                                    _ => {}
                                }
                            }
                            _ => {}
                        }
                    }
                }
                "object_creation_expression" => {
                    let line = node.start_position().row as u32 + 1;
                    let mut cursor = node.walk();
                    for child in node.children(&mut cursor) {
                        if matches!(
                            child.kind(),
                            "identifier" | "type_identifier" | "generic_name"
                        ) {
                            out.push((node_text(child, source).to_string(), line));
                            break;
                        }
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

    fn get_namespace(root: Node, source: &[u8]) -> String {
        let mut cursor = root.walk();
        for child in root.children(&mut cursor) {
            if matches!(
                child.kind(),
                "namespace_declaration" | "file_scoped_namespace_declaration"
            ) {
                let mut sc = child.walk();
                for sub in child.children(&mut sc) {
                    if matches!(sub.kind(), "qualified_name" | "identifier") {
                        return node_text(sub, source).to_string();
                    }
                }
            }
        }
        String::new()
    }

    fn file_to_module_path(filepath: &Path, src_root: &Path, namespace: &str) -> String {
        if !namespace.is_empty() {
            return namespace.to_string();
        }
        let rel = filepath.strip_prefix(src_root).unwrap_or(filepath);
        let parts: Vec<String> = rel
            .parent()
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

    fn get_base_types(node: Node, source: &[u8]) -> Vec<String> {
        // C# grammar: `base_list = ":" type ("," type)*`, and the `<int>` in
        // `IEnumerable<int>` is parsed as part of the type's `generic_name`
        // node (so `<int>` is included in the node's text). Two earlier
        // bugs lived here:
        //   1. Filtering by a hardcoded list of node kinds dropped any
        //      base type whose grammar kind wasn't on the list — in
        //      practice every secondary base after the first was lost,
        //      because the dotnet/runtime test corpus uses kinds the
        //      filter never enumerated.
        //   2. Generic args were retained as part of the bare name,
        //      so `IEnumerable<int>` never matched the `IEnumerable`
        //      entry in the resolution index.
        let mut out = Vec::new();
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            if child.kind() != "base_list" {
                continue;
            }
            let mut bc = child.walk();
            for sub in child.children(&mut bc) {
                if !sub.is_named() {
                    continue;
                }
                // Primary-constructor argument list (`class Foo : Bar(x, y)`)
                // is a sibling of the type, not a type itself.
                if sub.kind() == "argument_list" {
                    continue;
                }
                let text = node_text(sub, source).trim();
                if text.is_empty() {
                    continue;
                }
                let base_name = match text.split_once('<') {
                    Some((head, _)) => head.trim().to_string(),
                    None => text.to_string(),
                };
                if !base_name.is_empty() {
                    out.push(base_name);
                }
            }
        }
        out
    }

    fn get_enum_members(node: Node, source: &[u8]) -> Vec<String> {
        let mut out = Vec::new();
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            if child.kind() == "enum_member_declaration_list" {
                let mut lc = child.walk();
                for sub in child.children(&mut lc) {
                    if sub.kind() == "enum_member_declaration" {
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
            if matches!(child.kind(), "block" | "arrow_expression_clause") {
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
        if Self::has_modifier(node, source, "virtual") {
            metadata.insert("is_virtual".into(), json!(true));
        }
        if Self::has_modifier(node, source, "override") {
            metadata.insert("is_override".into(), json!(true));
        }
        if Self::has_modifier(node, source, "extern") {
            metadata.insert("is_ffi".into(), json!(true));
            let attrs = Self::get_attributes(node, source);
            let kind = if attrs.iter().any(|a| a.contains("DllImport")) {
                "pinvoke"
            } else {
                "extern"
            };
            metadata.insert("ffi_kind".into(), json!(kind));
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
                let (c, n) = compute_complexity(b, BRANCH_KINDS_CSHARP, NESTED_SCOPES);
                (Some(c), Some(n))
            }
            None => (None, None),
        };
        let is_recursive = Some(calls.iter().any(|(n, _)| n == &name));
        let docstring = Self::get_doc_comment(node, source);
        let procedure_names = extract_procedure_annotations(docstring.as_deref());
        FunctionInfo {
            visibility: Self::get_visibility(node, source, "private"),
            is_async: Self::has_modifier(node, source, "async"),
            is_method: owner.is_some(),
            signature: Self::get_signature(node, source),
            file_path: rel_path.to_string(),
            line_number: node.start_position().row as u32 + 1,
            end_line: Some(node.end_position().row as u32 + 1),
            docstring,
            return_type: Self::get_return_type(node, source),
            decorators: Self::get_attributes(node, source),
            calls,
            references: Vec::new(),
            function_refs: Vec::new(),
            type_parameters: get_type_parameters(node, source, "type_parameter_list"),
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

    /// Extract structured parameters from a C# method/constructor.
    /// Walks `parameter_list` for `parameter` nodes. tree-sitter-c-sharp uses
    /// `params` modifier for variadic; we treat any parameter with a leading
    /// `params` modifier as `ParameterKind::Variadic`.
    fn extract_parameters(node: Node, source: &[u8]) -> Vec<ParameterInfo> {
        let mut out = Vec::new();
        let mut cursor = node.walk();
        let Some(params_node) = node
            .children(&mut cursor)
            .find(|c| c.kind() == "parameter_list")
        else {
            return out;
        };
        let mut pcursor = params_node.walk();
        for child in params_node.children(&mut pcursor) {
            if child.kind() != "parameter" {
                continue;
            }
            let mut name: Option<String> = None;
            let mut type_ann: Option<String> = None;
            let mut default: Option<String> = None;
            let mut is_params = false;
            let mut tcursor = child.walk();
            for sub in child.children(&mut tcursor) {
                let k = sub.kind();
                if k == "identifier" && name.is_none() {
                    name = Some(node_text(sub, source).to_string());
                } else if k == "parameter_modifier" {
                    if node_text(sub, source).contains("params") {
                        is_params = true;
                    }
                } else if k == "equals_value_clause" {
                    let txt = node_text(sub, source);
                    let cleaned = txt.trim_start_matches('=').trim().to_string();
                    if !cleaned.is_empty() {
                        default = Some(cleaned);
                    }
                } else if k.contains("type")
                    || k == "predefined_type"
                    || k == "qualified_name"
                    || k == "generic_name"
                {
                    type_ann = Some(node_text(sub, source).to_string());
                }
            }
            let Some(n) = name else { continue };
            out.push(ParameterInfo {
                name: n,
                type_annotation: type_ann,
                default,
                kind: if is_params {
                    ParameterKind::Variadic
                } else {
                    ParameterKind::Positional
                },
            });
        }
        out
    }

    #[allow(clippy::too_many_arguments)]
    fn parse_type_declaration(
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
        let base_types = Self::get_base_types(node, source);
        let attributes = Self::get_attributes(node, source);
        let docstring = Self::get_doc_comment(node, source);
        let mut metadata = crate::code_tree::models::MetadataMap::new();
        if !attributes.is_empty() {
            metadata.insert("decorators".into(), json!(attributes));
        }
        if Self::has_modifier(node, source, "abstract") {
            metadata.insert("is_abstract".into(), json!(true));
        }
        if Self::has_modifier(node, source, "sealed") {
            metadata.insert("is_sealed".into(), json!(true));
        }
        if Self::has_modifier(node, source, "partial") {
            metadata.insert("is_partial".into(), json!(true));
        }

        if node.kind() == "interface_declaration" {
            result.interfaces.push(InterfaceInfo {
                qualified_name: qualified_name.clone(),
                kind: "interface".into(),
                visibility: Self::get_visibility(node, source, "internal"),
                file_path: rel_path.to_string(),
                line_number: node.start_position().row as u32 + 1,
                end_line: Some(node.end_position().row as u32 + 1),
                docstring,
                type_parameters: get_type_parameters(node, source, "type_parameter_list"),
                name: name.clone(),
            });
            for base in &base_types {
                result.type_relationships.push(TypeRelationship {
                    source_type: name.clone(),
                    target_type: Some(base.clone()),
                    relationship: "extends".into(),
                    methods: Vec::new(),
                });
            }
        } else {
            let kind = if node.kind() == "struct_declaration" {
                "struct"
            } else {
                "class"
            };
            let first_base: Vec<String> = base_types.iter().take(1).cloned().collect();
            result.classes.push(ClassInfo {
                qualified_name: qualified_name.clone(),
                kind: kind.into(),
                visibility: Self::get_visibility(node, source, "internal"),
                file_path: rel_path.to_string(),
                line_number: node.start_position().row as u32 + 1,
                end_line: Some(node.end_position().row as u32 + 1),
                docstring,
                bases: first_base,
                type_parameters: get_type_parameters(node, source, "type_parameter_list"),
                metadata,
                name: name.clone(),
            });
            if !base_types.is_empty() {
                result.type_relationships.push(TypeRelationship {
                    source_type: name.clone(),
                    target_type: Some(base_types[0].clone()),
                    relationship: "extends".into(),
                    methods: Vec::new(),
                });
                for iface in base_types.iter().skip(1) {
                    result.type_relationships.push(TypeRelationship {
                        source_type: name.clone(),
                        target_type: Some(iface.clone()),
                        relationship: "implements".into(),
                        methods: Vec::new(),
                    });
                }
            }
        }
        Self::parse_type_body(
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
    fn parse_type_body(
        node: Node,
        source: &[u8],
        module_path: &str,
        rel_path: &str,
        type_name: &str,
        type_qname: &str,
        result: &mut ParseResult,
    ) {
        let mut method_rel = TypeRelationship {
            source_type: type_qname.to_string(),
            target_type: None,
            relationship: "inherent".into(),
            methods: Vec::new(),
        };
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            if child.kind() != "declaration_list" {
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
                            Some(type_name),
                        );
                        method_rel.methods.push(fn_info.clone());
                        result.functions.push(fn_info);
                    }
                    "field_declaration" => {
                        Self::parse_field(item, source, rel_path, type_qname, result)
                    }
                    "property_declaration" => {
                        Self::parse_property(item, source, rel_path, type_qname, result)
                    }
                    "class_declaration"
                    | "struct_declaration"
                    | "record_declaration"
                    | "interface_declaration" => {
                        Self::parse_type_declaration(
                            item,
                            source,
                            module_path,
                            rel_path,
                            result,
                            Some(type_name),
                        );
                    }
                    "enum_declaration" => {
                        Self::parse_enum(item, source, module_path, rel_path, result)
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
        type_qname: &str,
        result: &mut ParseResult,
    ) {
        let is_static = Self::has_modifier(node, source, "static");
        let is_const = Self::has_modifier(node, source, "const");
        let is_readonly = Self::has_modifier(node, source, "readonly");
        let visibility = Self::get_visibility(node, source, "private");
        let mut type_ann: Option<String> = None;
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            if child.kind() != "variable_declaration" {
                continue;
            }
            let mut vc = child.walk();
            for sub in child.children(&mut vc) {
                if TYPE_NODES.contains(&sub.kind()) {
                    type_ann = Some(node_text(sub, source).to_string());
                    break;
                }
            }
            let mut vc2 = child.walk();
            for sub in child.children(&mut vc2) {
                if sub.kind() != "variable_declarator" {
                    continue;
                }
                let Some(name) = Self::get_name(sub, source) else {
                    continue;
                };
                let name = name.to_string();
                let mut val_text: Option<String> = None;
                let mut ic = sub.walk();
                for inner in sub.children(&mut ic) {
                    if inner.kind() == "equals_value_clause" {
                        let mut cc = inner.walk();
                        for v in inner.children(&mut cc) {
                            if v.is_named() {
                                let text = node_text(v, source);
                                let take = text
                                    .char_indices()
                                    .nth(100)
                                    .map(|(i, _)| i)
                                    .unwrap_or(text.len());
                                val_text = Some(text[..take].to_string());
                                break;
                            }
                        }
                        break;
                    }
                }
                if is_const || (is_static && is_readonly) {
                    result.constants.push(ConstantInfo {
                        qualified_name: format!("{}.{}", type_qname, name),
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
                        qualified_name: format!("{}.{}", type_qname, name),
                        owner_qualified_name: type_qname.to_string(),
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
    }

    fn parse_property(
        node: Node,
        source: &[u8],
        rel_path: &str,
        type_qname: &str,
        result: &mut ParseResult,
    ) {
        let Some(name) = Self::get_name(node, source) else {
            return;
        };
        let name = name.to_string();
        let mut type_ann: Option<String> = None;
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            if TYPE_NODES.contains(&child.kind()) {
                type_ann = Some(node_text(child, source).to_string());
                break;
            }
        }
        result.attributes.push(AttributeInfo {
            qualified_name: format!("{}.{}", type_qname, name),
            owner_qualified_name: type_qname.to_string(),
            type_annotation: type_ann,
            visibility: Self::get_visibility(node, source, "private"),
            file_path: rel_path.to_string(),
            line_number: node.start_position().row as u32 + 1,
            default_value: None,
            name,
        });
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
            visibility: Self::get_visibility(node, source, "internal"),
            file_path: rel_path.to_string(),
            line_number: node.start_position().row as u32 + 1,
            end_line: Some(node.end_position().row as u32 + 1),
            docstring: Self::get_doc_comment(node, source),
            variants: Self::get_enum_members(node, source),
            variant_details: None,
            name,
        });
    }

    fn parse_top_level(
        node: Node,
        source: &[u8],
        module_path: &str,
        rel_path: &str,
        result: &mut ParseResult,
        file_info: &mut FileInfo,
    ) {
        match node.kind() {
            "class_declaration"
            | "struct_declaration"
            | "record_declaration"
            | "interface_declaration" => {
                Self::parse_type_declaration(node, source, module_path, rel_path, result, None);
            }
            "enum_declaration" => Self::parse_enum(node, source, module_path, rel_path, result),
            "namespace_declaration" | "file_scoped_namespace_declaration" => {
                let mut ns_name: Option<String> = None;
                let mut cursor = node.walk();
                for child in node.children(&mut cursor) {
                    match child.kind() {
                        "qualified_name" | "identifier" => {
                            ns_name = Some(node_text(child, source).to_string());
                        }
                        "declaration_list" => {
                            let ns_path =
                                ns_name.clone().unwrap_or_else(|| module_path.to_string());
                            let mut dc = child.walk();
                            for item in child.children(&mut dc) {
                                Self::parse_top_level(
                                    item, source, &ns_path, rel_path, result, file_info,
                                );
                            }
                        }
                        _ => {
                            if let Some(ns) = &ns_name {
                                if child.is_named() {
                                    Self::parse_top_level(
                                        child, source, ns, rel_path, result, file_info,
                                    );
                                }
                            }
                        }
                    }
                }
            }
            "using_directive" => {
                let mut cursor = node.walk();
                for child in node.children(&mut cursor) {
                    if matches!(child.kind(), "qualified_name" | "identifier") {
                        file_info.imports.push(node_text(child, source).to_string());
                        break;
                    }
                }
            }
            "global_statement" => {
                let mut cursor = node.walk();
                for child in node.children(&mut cursor) {
                    if child.is_named() {
                        Self::parse_top_level(
                            child,
                            source,
                            module_path,
                            rel_path,
                            result,
                            file_info,
                        );
                    }
                }
            }
            _ => {}
        }
    }
}

impl Default for CSharpParser {
    fn default() -> Self {
        Self::new()
    }
}

impl LanguageParser for CSharpParser {
    fn language_name(&self) -> &'static str {
        "csharp"
    }
    fn file_extensions(&self) -> &'static [&'static str] {
        &["cs"]
    }
    fn noise_names(&self) -> &'static [&'static str] {
        CSHARP_NOISE_NAMES
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
            || rel_path.contains("/Tests/")
            || rel_path.contains("/.Tests/")
            || rel_path.starts_with("Tests/")
            || rel_path.starts_with("tests/");

        if let Some(reason) = is_generated_or_minified(&source) {
            let module_path = stem.clone();
            let mut r = ParseResult::new();
            r.files.push(FileInfo {
                path: rel_path,
                filename,
                loc,
                module_path,
                language: "csharp".to_string(),
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
        let namespace = Self::get_namespace(root, &source);
        let module_path = Self::file_to_module_path(filepath, src_root, &namespace);
        let mut file_info = FileInfo {
            path: rel_path.clone(),
            filename,
            loc,
            module_path: module_path.clone(),
            language: "csharp".to_string(),
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
            Self::parse_top_level(
                child,
                &source,
                &module_path,
                &rel_path,
                &mut result,
                &mut file_info,
            );
        }
        file_info.annotations = extract_comment_annotations(root, &source, DEFAULT_COMMENT_TYPES);
        result.files.push(file_info);
        result
    }
}
