//! Go language parser (ported from parsers/go.py).

use std::path::Path;
use tree_sitter::{Node, Parser, Tree};

use super::shared::{
    compute_complexity, count_lines, extract_comment_annotations, extract_procedure_annotations,
    get_type_parameters, is_generated_or_minified, node_text, BRANCH_KINDS_GO,
    DEFAULT_COMMENT_TYPES,
};
use super::LanguageParser;
use crate::code_tree::models::{
    AttributeInfo, ClassInfo, ConstantInfo, FileInfo, FunctionInfo, InterfaceInfo, ParameterInfo,
    ParameterKind, ParseResult, TypeRelationship,
};

pub const GO_NOISE_NAMES: &[&str] = &[
    "len", "cap", "append", "copy", "delete", "make", "new", "close", "panic", "recover", "print",
    "println", "Error", "String", "Format", "Sprintf", "Fprintf", "Errorf", "Printf", "Println",
    "Print", "Fprintln", "Fprint", "Fatal", "Fatalf", "Fatalln",
];

const NESTED_SCOPES: &[&str] = &["function_declaration", "method_declaration", "func_literal"];

pub struct GoParser;

thread_local! {
    static GO_PARSER: std::cell::RefCell<Parser> = {
        let mut p = Parser::new();
        p.set_language(&tree_sitter_go::LANGUAGE.into())
            .expect("loading tree-sitter-go grammar");
        std::cell::RefCell::new(p)
    };
}

impl GoParser {
    pub fn new() -> Self {
        GoParser
    }

    fn parse_tree(&self, source: &[u8]) -> Option<Tree> {
        GO_PARSER.with(|p| p.borrow_mut().parse(source, None))
    }

    fn get_visibility(name: &str) -> &'static str {
        if name.chars().next().is_some_and(|c| c.is_ascii_uppercase()) {
            "exported"
        } else {
            "unexported"
        }
    }

    fn get_doc_comment(node: Node, source: &[u8]) -> Option<String> {
        let mut doc_lines: Vec<String> = Vec::new();
        let mut sibling = node.prev_named_sibling();
        while let Some(s) = sibling {
            if s.kind() != "comment" {
                break;
            }
            let text = node_text(s, source).trim();
            let Some(rest) = text.strip_prefix("//") else {
                break;
            };
            let content = rest.strip_prefix(' ').unwrap_or(rest);
            doc_lines.insert(0, content.to_string());
            sibling = s.prev_named_sibling();
        }
        if doc_lines.is_empty() {
            None
        } else {
            Some(doc_lines.join("\n"))
        }
    }

    fn get_name<'a>(node: Node<'a>, source: &'a [u8], name_type: &str) -> Option<&'a str> {
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            if child.kind() == name_type
                || child.kind() == "type_identifier"
                || child.kind() == "field_identifier"
            {
                return Some(node_text(child, source));
            }
        }
        None
    }

    fn get_signature(node: Node, source: &[u8]) -> String {
        let mut parts: Vec<&str> = Vec::new();
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            if child.kind() == "block" {
                break;
            }
            parts.push(node_text(child, source));
        }
        parts.join(" ")
    }

    fn get_return_type(node: Node, source: &[u8]) -> Option<String> {
        // Use the named `result` field. Avoids the method-receiver trap:
        // for `func (c *Call) Once() *Call`, naive child iteration sees the
        // receiver `parameter_list` first, then mistakes the method name
        // (`Once`) for the return type. The `result` field points directly
        // at the return-type node for both function_declaration and
        // method_declaration.
        node.child_by_field_name("result")
            .map(|c| node_text(c, source).to_string())
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
                        "selector_expression" => {
                            let field = func.child_by_field_name("field");
                            let operand = func.child_by_field_name("operand");
                            if let (Some(field), Some(operand)) = (field, operand) {
                                let field_name = node_text(field, source);
                                let op_text = node_text(operand, source);
                                let hint = op_text.rsplit('.').next().unwrap_or(op_text);
                                out.push((format!("{}.{}", hint, field_name), line));
                            } else if let Some(field) = field {
                                out.push((node_text(field, source).to_string(), line));
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

    fn get_package_name(root: Node, source: &[u8]) -> String {
        let mut cursor = root.walk();
        for child in root.children(&mut cursor) {
            if child.kind() == "package_clause" {
                let mut sc = child.walk();
                for sub in child.children(&mut sc) {
                    if sub.kind() == "package_identifier" {
                        return node_text(sub, source).to_string();
                    }
                }
            }
        }
        "main".to_string()
    }

    fn file_to_module_path(filepath: &Path, src_root: &Path, package: &str) -> String {
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
            package.to_string()
        } else {
            format!("{}/{}", package, parts.join("/"))
        }
    }

    fn extract_struct_fields(
        node: Node,
        source: &[u8],
        owner_qname: &str,
        rel_path: &str,
    ) -> Vec<AttributeInfo> {
        let mut out = Vec::new();
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
                let mut names: Vec<String> = Vec::new();
                let mut type_ann: Option<String> = None;
                let mut inner = field.walk();
                for sub in field.children(&mut inner) {
                    match sub.kind() {
                        "field_identifier" => names.push(node_text(sub, source).to_string()),
                        _ if sub.is_named()
                            && !matches!(sub.kind(), "tag" | "comment" | "field_identifier") =>
                        {
                            if type_ann.is_none() && names.is_empty() {
                                let text = node_text(sub, source);
                                let stripped = text.strip_prefix('*').unwrap_or(text);
                                names.push(stripped.to_string());
                                type_ann = Some(text.to_string());
                            } else if type_ann.is_none() {
                                type_ann = Some(node_text(sub, source).to_string());
                            }
                        }
                        _ => {}
                    }
                }
                for name in names {
                    out.push(AttributeInfo {
                        qualified_name: format!("{}.{}", owner_qname, name),
                        owner_qualified_name: owner_qname.to_string(),
                        type_annotation: type_ann.clone(),
                        visibility: Self::get_visibility(&name).to_string(),
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

    fn get_receiver_type<'a>(node: Node<'a>, source: &'a [u8]) -> Option<&'a str> {
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            if child.kind() == "parameter_list" {
                let mut pc = child.walk();
                for param in child.children(&mut pc) {
                    if param.kind() == "parameter_declaration" {
                        let mut sc = param.walk();
                        for sub in param.children(&mut sc) {
                            match sub.kind() {
                                "type_identifier" => return Some(node_text(sub, source)),
                                "pointer_type" => {
                                    let mut ic = sub.walk();
                                    for inner in sub.children(&mut ic) {
                                        if inner.kind() == "type_identifier" {
                                            return Some(node_text(inner, source));
                                        }
                                    }
                                }
                                _ => {}
                            }
                        }
                    }
                }
                break;
            }
        }
        None
    }

    fn parse_function(
        node: Node,
        source: &[u8],
        module_path: &str,
        rel_path: &str,
        is_method: bool,
        owner: Option<&str>,
    ) -> FunctionInfo {
        let name = Self::get_name(node, source, "identifier")
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
            if child.kind() == "block" {
                body = Some(child);
                break;
            }
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
                let (c, n) = compute_complexity(b, BRANCH_KINDS_GO, NESTED_SCOPES);
                (Some(c), Some(n))
            }
            None => (None, None),
        };
        let is_recursive = Some(calls.iter().any(|(n, _)| n == &name));
        let docstring = Self::get_doc_comment(node, source);
        let procedure_names = extract_procedure_annotations(docstring.as_deref());
        FunctionInfo {
            visibility: Self::get_visibility(&name).to_string(),
            is_async: false,
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
            type_parameters: get_type_parameters(node, source, "type_parameter_list"),
            decorators: Vec::new(),
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

    /// Extract structured parameters from a Go function/method declaration.
    /// Uses tree-sitter-go's named fields: `parameters` for the regular argument
    /// list (excluding the receiver) and `receiver` for the method receiver.
    /// `variadic_parameter_declaration` → `ParameterKind::Variadic`.
    /// Receiver entries → `ParameterKind::Receiver`, prepended so they appear
    /// first in display order (matching source-code reading order).
    fn extract_parameters(node: Node, source: &[u8]) -> Vec<ParameterInfo> {
        let mut out = Vec::new();
        // Receiver first (only present on method_declaration). Preserved at
        // the head of `parameters` so iteration matches source order.
        if let Some(recv) = node.child_by_field_name("receiver") {
            Self::extract_param_list(recv, source, true, &mut out);
        }
        if let Some(params) = node.child_by_field_name("parameters") {
            Self::extract_param_list(params, source, false, &mut out);
        }
        out
    }

    /// Walk a `parameter_list` node and append `ParameterInfo` entries.
    /// Shared between regular parameters and method receivers — the AST shape
    /// is identical, only the `ParameterKind` differs.
    fn extract_param_list(
        params_node: Node,
        source: &[u8],
        is_receiver: bool,
        out: &mut Vec<ParameterInfo>,
    ) {
        let mut pcursor = params_node.walk();
        for child in params_node.children(&mut pcursor) {
            let is_variadic = match child.kind() {
                "parameter_declaration" => false,
                "variadic_parameter_declaration" => true,
                _ => continue,
            };
            let kind = if is_receiver {
                ParameterKind::Receiver
            } else if is_variadic {
                ParameterKind::Variadic
            } else {
                ParameterKind::Positional
            };
            // Each declaration may bind multiple names: `a, b int`.
            let mut names: Vec<String> = Vec::new();
            let mut type_ann: Option<String> = None;
            let mut tcursor = child.walk();
            for sub in child.children(&mut tcursor) {
                match sub.kind() {
                    "identifier" => names.push(node_text(sub, source).to_string()),
                    k if k.contains("type") || k == "qualified_type" || k == "pointer_type" => {
                        type_ann = Some(node_text(sub, source).to_string());
                    }
                    _ => {}
                }
            }
            // Anonymous parameters: `func(int, string)` — type-only, no name.
            if names.is_empty() {
                if let Some(t) = type_ann.clone() {
                    out.push(ParameterInfo {
                        name: "_".into(),
                        type_annotation: Some(t),
                        default: None,
                        kind,
                    });
                }
                continue;
            }
            for n in names {
                out.push(ParameterInfo {
                    name: n,
                    type_annotation: type_ann.clone(),
                    default: None,
                    kind,
                });
            }
        }
    }
}

impl Default for GoParser {
    fn default() -> Self {
        Self::new()
    }
}

impl LanguageParser for GoParser {
    fn language_name(&self) -> &'static str {
        "go"
    }
    fn file_extensions(&self) -> &'static [&'static str] {
        &["go"]
    }
    fn noise_names(&self) -> &'static [&'static str] {
        GO_NOISE_NAMES
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
        let is_test = filename.ends_with("_test.go");

        if let Some(reason) = is_generated_or_minified(&source) {
            // package name requires parsing; fall back to filename stem.
            let module_path = filepath
                .file_stem()
                .and_then(|o| o.to_str())
                .unwrap_or("")
                .to_string();
            let mut r = ParseResult::new();
            r.files.push(FileInfo {
                path: rel_path,
                filename,
                loc,
                module_path,
                language: "go".to_string(),
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
        let package = Self::get_package_name(root, &source);
        let module_path = Self::file_to_module_path(filepath, src_root, &package);

        let mut file_info = FileInfo {
            path: rel_path.clone(),
            filename,
            loc,
            module_path: module_path.clone(),
            language: "go".to_string(),
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
                "function_declaration" => {
                    result.functions.push(Self::parse_function(
                        child,
                        &source,
                        &module_path,
                        &rel_path,
                        false,
                        None,
                    ));
                }
                "method_declaration" => {
                    let receiver = Self::get_receiver_type(child, &source).map(str::to_string);
                    let fn_info = Self::parse_function(
                        child,
                        &source,
                        &module_path,
                        &rel_path,
                        true,
                        receiver.as_deref(),
                    );
                    if let Some(recv) = receiver {
                        result.type_relationships.push(TypeRelationship {
                            source_type: recv,
                            target_type: None,
                            relationship: "inherent".into(),
                            methods: vec![fn_info.clone()],
                        });
                    }
                    result.functions.push(fn_info);
                }
                "type_declaration" => {
                    let mut tc = child.walk();
                    for spec in child.children(&mut tc) {
                        if spec.kind() != "type_spec" {
                            continue;
                        }
                        let Some(name) = Self::get_name(spec, &source, "type_identifier") else {
                            continue;
                        };
                        let name = name.to_string();
                        let qname = format!("{}.{}", module_path, name);
                        let docstring = Self::get_doc_comment(child, &source);
                        let mut type_node: Option<Node> = None;
                        let mut saw_name = false;
                        let mut sc = spec.walk();
                        for sub in spec.children(&mut sc) {
                            if sub.kind() == "type_identifier" && !saw_name {
                                saw_name = true;
                                continue;
                            }
                            if matches!(
                                sub.kind(),
                                "struct_type"
                                    | "interface_type"
                                    | "type_identifier"
                                    | "qualified_type"
                                    | "pointer_type"
                                    | "slice_type"
                                    | "map_type"
                                    | "channel_type"
                                    | "function_type"
                                    | "array_type"
                            ) {
                                type_node = Some(sub);
                                break;
                            }
                        }
                        let Some(type_node) = type_node else { continue };
                        match type_node.kind() {
                            "struct_type" => {
                                result.classes.push(ClassInfo {
                                    qualified_name: qname.clone(),
                                    kind: "struct".into(),
                                    visibility: Self::get_visibility(&name).to_string(),
                                    file_path: rel_path.clone(),
                                    line_number: child.start_position().row as u32 + 1,
                                    end_line: Some(child.end_position().row as u32 + 1),
                                    docstring,
                                    bases: Vec::new(),
                                    type_parameters: get_type_parameters(
                                        spec,
                                        &source,
                                        "type_parameter_list",
                                    ),
                                    metadata: Default::default(),
                                    name: name.clone(),
                                });
                                result.attributes.extend(Self::extract_struct_fields(
                                    type_node, &source, &qname, &rel_path,
                                ));
                            }
                            "interface_type" => {
                                result.interfaces.push(InterfaceInfo {
                                    qualified_name: qname.clone(),
                                    kind: "interface".into(),
                                    visibility: Self::get_visibility(&name).to_string(),
                                    file_path: rel_path.clone(),
                                    line_number: child.start_position().row as u32 + 1,
                                    end_line: Some(child.end_position().row as u32 + 1),
                                    docstring,
                                    type_parameters: get_type_parameters(
                                        spec,
                                        &source,
                                        "type_parameter_list",
                                    ),
                                    name: name.clone(),
                                });
                                let mut iface_rel = TypeRelationship {
                                    source_type: qname.clone(),
                                    target_type: None,
                                    relationship: "inherent".into(),
                                    methods: Vec::new(),
                                };
                                let mut mc = type_node.walk();
                                for ms in type_node.children(&mut mc) {
                                    if ms.kind() == "method_spec" {
                                        if let Some(fn_name) =
                                            Self::get_name(ms, &source, "field_identifier")
                                        {
                                            let fn_name = fn_name.to_string();
                                            let fn_info = FunctionInfo {
                                                visibility: Self::get_visibility(&fn_name)
                                                    .to_string(),
                                                is_method: true,
                                                signature: node_text(ms, &source).to_string(),
                                                file_path: rel_path.clone(),
                                                line_number: ms.start_position().row as u32 + 1,
                                                end_line: Some(ms.end_position().row as u32 + 1),
                                                qualified_name: format!("{}.{}", qname, fn_name),
                                                name: fn_name,
                                                ..Default::default()
                                            };
                                            iface_rel.methods.push(fn_info.clone());
                                            result.functions.push(fn_info);
                                        }
                                    }
                                }
                                if !iface_rel.methods.is_empty() {
                                    result.type_relationships.push(iface_rel);
                                }
                            }
                            _ => {
                                result.constants.push(ConstantInfo {
                                    qualified_name: qname,
                                    kind: "type_alias".into(),
                                    type_annotation: Some(
                                        node_text(type_node, &source).to_string(),
                                    ),
                                    value_preview: None,
                                    visibility: Self::get_visibility(&name).to_string(),
                                    file_path: rel_path.clone(),
                                    line_number: child.start_position().row as u32 + 1,
                                    name,
                                });
                            }
                        }
                    }
                }
                "const_declaration" | "var_declaration" => {
                    let kind = if child.kind() == "const_declaration" {
                        "constant"
                    } else {
                        "static"
                    };
                    let mut cc = child.walk();
                    for spec in child.children(&mut cc) {
                        if spec.kind() != "const_spec" && spec.kind() != "var_spec" {
                            continue;
                        }
                        let Some(name) = Self::get_name(spec, &source, "identifier") else {
                            continue;
                        };
                        let name = name.to_string();
                        let mut type_ann: Option<String> = None;
                        let mut val_text: Option<String> = None;
                        let mut sc = spec.walk();
                        for sub in spec.children(&mut sc) {
                            match sub.kind() {
                                "type_identifier" => {
                                    type_ann = Some(node_text(sub, &source).to_string())
                                }
                                "expression_list" => {
                                    let text = node_text(sub, &source);
                                    let take = text
                                        .char_indices()
                                        .nth(100)
                                        .map(|(i, _)| i)
                                        .unwrap_or(text.len());
                                    val_text = Some(text[..take].to_string());
                                }
                                _ => {}
                            }
                        }
                        result.constants.push(ConstantInfo {
                            qualified_name: format!("{}.{}", module_path, name),
                            kind: kind.into(),
                            type_annotation: type_ann,
                            value_preview: val_text,
                            visibility: Self::get_visibility(&name).to_string(),
                            file_path: rel_path.clone(),
                            line_number: spec.start_position().row as u32 + 1,
                            name,
                        });
                    }
                }
                "import_declaration" => {
                    let mut ic = child.walk();
                    for spec in child.children(&mut ic) {
                        match spec.kind() {
                            "import_spec" => {
                                let mut sc = spec.walk();
                                for sub in spec.children(&mut sc) {
                                    if sub.kind() == "interpreted_string_literal" {
                                        let text = node_text(sub, &source);
                                        file_info.imports.push(text.trim_matches('"').to_string());
                                    }
                                }
                            }
                            "import_spec_list" => {
                                let mut lc = spec.walk();
                                for item in spec.children(&mut lc) {
                                    if item.kind() == "import_spec" {
                                        let mut sc = item.walk();
                                        for sub in item.children(&mut sc) {
                                            if sub.kind() == "interpreted_string_literal" {
                                                let text = node_text(sub, &source);
                                                file_info
                                                    .imports
                                                    .push(text.trim_matches('"').to_string());
                                            }
                                        }
                                    }
                                }
                            }
                            _ => {}
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
