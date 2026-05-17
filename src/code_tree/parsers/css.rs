//! CSS language parser.
//!
//! Coverage in 0.9.36:
//!   - One `Selector` node per `rule_set` (top-level and nested inside
//!     `@media` blocks). Selector-list rules like `.foo, .bar, .baz`
//!     emit ONE node named `.foo, .bar, .baz`, not three — keeps real
//!     stylesheets bounded by source rather than by selector-list
//!     combinatorics.
//!   - One `ConstantInfo` per `--custom-property` declaration (CSS
//!     variables), with kind="css_custom_property" and the value text
//!     (≤100 chars) in `value_preview`.
//!   - `@import url(...)` / `@import "..."` → `FileInfo.imports`
//!     (the 0.9.34 File → File IMPORTS resolver picks up the rest).
//!
//! Not yet supported:
//!   - `@keyframes` / `@font-face` as structural nodes.
//!   - `var(--foo)` references emitting USES edges to the matching
//!     custom-property ConstantInfo.
//!   - HTML class-attribute → CSS class-selector cross-language joins.

use std::path::Path;
use tree_sitter::{Node, Parser, Tree};

use super::shared::node_text;
use super::LanguageParser;
use crate::code_tree::models::{ConstantInfo, FileInfo, ParseResult, SelectorInfo};

pub struct CssParser;

thread_local! {
    static TS_PARSER: std::cell::RefCell<Parser> = {
        let mut p = Parser::new();
        p.set_language(&tree_sitter_css::LANGUAGE.into())
            .expect("loading tree-sitter-css grammar");
        std::cell::RefCell::new(p)
    };
}

impl CssParser {
    pub fn new() -> Self {
        CssParser
    }

    fn parse_tree(&self, source: &[u8]) -> Option<Tree> {
        TS_PARSER.with(|p| p.borrow_mut().parse(source, None))
    }

    /// Walk the stylesheet (or a media block) and emit Selector +
    /// ConstantInfo rows. Top-level dispatch handles `@import` and
    /// nested `@media`; everything else either becomes a Selector or
    /// is parse noise.
    fn walk_stylesheet(
        node: Node,
        source: &[u8],
        rel_path: &str,
        result: &mut ParseResult,
        file_info: &mut FileInfo,
    ) {
        let mut cursor = node.walk();
        for child in node.named_children(&mut cursor) {
            match child.kind() {
                "rule_set" => {
                    Self::emit_rule_set(child, source, rel_path, result);
                }
                "import_statement" => {
                    if let Some(url) = Self::extract_import_url(child, source) {
                        file_info.imports.push(url);
                    }
                }
                "media_statement" => {
                    // Recurse into the @media block — nested rule_sets
                    // emit as normal Selectors. The @media wrapper
                    // itself is parse noise in v1 (no structural node).
                    if let Some(block) = Self::find_block(child) {
                        Self::walk_stylesheet(block, source, rel_path, result, file_info);
                    }
                }
                "at_rule" => {
                    // Generic @-rules (e.g. `@supports`, `@layer`).
                    // Recurse into any block child to surface nested
                    // rule_sets — same logic as @media.
                    if let Some(block) = Self::find_block(child) {
                        Self::walk_stylesheet(block, source, rel_path, result, file_info);
                    }
                }
                _ => {}
            }
        }
    }

    fn emit_rule_set(node: Node, source: &[u8], rel_path: &str, result: &mut ParseResult) {
        let line = node.start_position().row as u32 + 1;
        let end_line = node.end_position().row as u32 + 1;

        // The selectors are a `selectors` child whose raw text is the
        // canonical name (e.g. `.foo, .bar, .baz` or `#nav > li`).
        let selectors_node = Self::find_named_child(node, "selectors");
        let Some(selectors_node) = selectors_node else {
            return;
        };
        let raw = node_text(selectors_node, source).trim().to_string();
        if raw.is_empty() {
            return;
        }
        // Normalise whitespace inside the selector list so the same
        // logical rule produces the same name regardless of source
        // formatting.
        let canonical = raw.split_whitespace().collect::<Vec<_>>().join(" ");
        let qname = format!("{rel_path}:{line}:{}", slugify(&canonical));

        result.selectors.push(SelectorInfo {
            name: canonical,
            qualified_name: qname,
            kind: "rule".to_string(),
            file_path: rel_path.to_string(),
            line_number: line,
            end_line: Some(end_line),
        });

        // Inside the block, look for `--custom-property: value`
        // declarations and emit them as ConstantInfo.
        if let Some(block) = Self::find_block(node) {
            Self::extract_custom_properties(block, source, rel_path, result);
        }
    }

    /// Walk a rule_set's block and emit a ConstantInfo for every
    /// declaration whose property_name starts with `--` (CSS custom
    /// property / design token).
    fn extract_custom_properties(
        block: Node,
        source: &[u8],
        rel_path: &str,
        result: &mut ParseResult,
    ) {
        let mut cursor = block.walk();
        for child in block.named_children(&mut cursor) {
            if child.kind() != "declaration" {
                continue;
            }
            let prop_name = Self::find_named_child(child, "property_name")
                .map(|n| node_text(n, source).to_string());
            let Some(prop_name) = prop_name else { continue };
            if !prop_name.starts_with("--") {
                continue;
            }
            let line = child.start_position().row as u32 + 1;
            let raw = node_text(child, source);
            let take = raw
                .char_indices()
                .nth(100)
                .map(|(i, _)| i)
                .unwrap_or(raw.len());
            let value_preview = Some(raw[..take].trim().to_string());
            // qname: `{rel_path}:custom_property:{prop_name}`. CSS
            // custom properties are global to the document, but
            // scoping them by file is enough for uniqueness in
            // single-file fixtures.
            let qname = format!("{rel_path}:custom_property:{}", &prop_name);
            result.constants.push(ConstantInfo {
                qualified_name: qname,
                visibility: "public".to_string(),
                name: prop_name,
                kind: "css_custom_property".to_string(),
                type_annotation: None,
                value_preview,
                file_path: rel_path.to_string(),
                line_number: line,
            });
        }
    }

    /// Extract the URL string from an `@import` statement. Handles both
    /// `@import "./theme.css"` (string_value child) and
    /// `@import url("./theme.css")` (call_expression with string_value
    /// in arguments).
    fn extract_import_url(node: Node, source: &[u8]) -> Option<String> {
        let mut cursor = node.walk();
        for child in node.named_children(&mut cursor) {
            match child.kind() {
                "string_value" => {
                    return Some(strip_quotes(node_text(child, source)));
                }
                "call_expression" => {
                    // `url("...")` form — look for string_value in
                    // the call's children.
                    let mut sub = child.walk();
                    for c in child.named_children(&mut sub) {
                        if c.kind() == "arguments" {
                            let mut a = c.walk();
                            for arg in c.named_children(&mut a) {
                                if arg.kind() == "string_value" || arg.kind() == "plain_value" {
                                    return Some(strip_quotes(node_text(arg, source)));
                                }
                            }
                        }
                        if c.kind() == "string_value" {
                            return Some(strip_quotes(node_text(c, source)));
                        }
                    }
                }
                _ => {}
            }
        }
        None
    }

    fn find_named_child<'a>(node: Node<'a>, name: &str) -> Option<Node<'a>> {
        let mut cursor = node.walk();
        for child in node.named_children(&mut cursor) {
            if child.kind() == name {
                return Some(child);
            }
        }
        None
    }

    fn find_block(node: Node) -> Option<Node> {
        Self::find_named_child(node, "block")
    }
}

fn strip_quotes(s: &str) -> String {
    let trimmed = s.trim();
    let bytes = trimmed.as_bytes();
    if bytes.len() >= 2
        && (bytes[0] == b'"' || bytes[0] == b'\'')
        && bytes[bytes.len() - 1] == bytes[0]
    {
        trimmed[1..trimmed.len() - 1].to_string()
    } else {
        trimmed.to_string()
    }
}

fn slugify(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut prev_dash = false;
    for c in s.chars().take(80) {
        if c.is_ascii_alphanumeric() || c == '_' {
            out.push(c.to_ascii_lowercase());
            prev_dash = false;
        } else if !prev_dash && !out.is_empty() {
            out.push('-');
            prev_dash = true;
        }
    }
    out.trim_end_matches('-').to_string()
}

fn file_to_module_path(filepath: &Path, src_root: &Path) -> String {
    let stem = filepath.file_stem().and_then(|o| o.to_str()).unwrap_or("");
    let pkg = src_root.file_name().and_then(|o| o.to_str()).unwrap_or("");
    match (pkg.is_empty(), stem.is_empty()) {
        (true, _) => stem.to_string(),
        (false, true) => pkg.to_string(),
        (false, false) => format!("{pkg}.{stem}"),
    }
}

impl LanguageParser for CssParser {
    fn language_name(&self) -> &'static str {
        "css"
    }
    fn file_extensions(&self) -> &'static [&'static str] {
        &["css"]
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
        let is_test = crate::code_tree::parsers::shared::is_test_path(&rel_path, &filename, &[]);
        let mut file_info = FileInfo {
            path: rel_path.clone(),
            filename,
            loc: source.lines().count() as u32,
            module_path,
            language: "css".to_string(),
            submodule_declarations: Vec::new(),
            imports: Vec::new(),
            exports: Vec::new(),
            annotations: None,
            is_test,
            skip_reason: None,
        };

        Self::walk_stylesheet(
            tree.root_node(),
            source_bytes,
            &rel_path,
            &mut result,
            &mut file_info,
        );

        result.files.push(file_info);
        result
    }
}
