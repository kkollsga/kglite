//! HTML language parser (god-HTML-file ready).
//!
//! Coverage in 0.9.36:
//!   - `Element` nodes emitted only for elements with semantic
//!     interest: headings (`h1`-`h6`), elements with an `id`
//!     attribute, and `<form>` elements with `action`. Everything
//!     else stays parse noise to keep god-HTML graphs navigable.
//!   - `Element -[HAS_CHILD]-> Element` edges via `parent_qname`,
//!     accumulated for the outline view. Only counts ancestry where
//!     BOTH ends are emitted Elements. (Edge name avoids `CONTAINS`,
//!     which is a reserved Cypher keyword for substring matching.)
//!   - `<script src="...">` and `<link rel="stylesheet" href="...">`
//!     → `FileInfo.imports` (the existing 0.9.34 File → File IMPORTS
//!     resolver picks them up).
//!   - Inline `<script>...</script>` blocks are parsed by the existing
//!     `JstsParser::javascript()` — Function nodes from the body land
//!     in the result with qnames scoped to `{rel_path}:script_<n>`
//!     so they don't collide with same-named functions elsewhere.
//!
//! Not yet supported:
//!   - `<a href>` navigation graph (cross-page link discovery).
//!   - `<style>...</style>` block CSS parsing (analogous to script
//!     extraction but noisier on real god-HTML pages).
//!   - HTML5 microdata / aria-label structural hints.

use std::path::Path;
use tree_sitter::{Node, Parser, Tree};

use super::shared::node_text;
use super::typescript::JstsParser;
use super::LanguageParser;
use crate::code_tree::models::{ElementInfo, FileInfo, ParseResult};

pub struct HtmlParser;

thread_local! {
    static TS_PARSER: std::cell::RefCell<Parser> = {
        let mut p = Parser::new();
        p.set_language(&tree_sitter_html::LANGUAGE.into())
            .expect("loading tree-sitter-html grammar");
        std::cell::RefCell::new(p)
    };
}

const HEADING_TAGS: &[&str] = &["h1", "h2", "h3", "h4", "h5", "h6"];

impl HtmlParser {
    pub fn new() -> Self {
        HtmlParser
    }

    fn parse_tree(&self, source: &[u8]) -> Option<Tree> {
        TS_PARSER.with(|p| p.borrow_mut().parse(source, None))
    }

    /// Recursive walk: emit Element nodes for matching shapes, and
    /// invoke JS sub-parsing on embedded script bodies. Tracks the
    /// nearest emitted ancestor in `parent_qname` for HAS_CHILD edges.
    fn walk_element(
        node: Node,
        source: &[u8],
        rel_path: &str,
        result: &mut ParseResult,
        file_info: &mut FileInfo,
        parent_qname: Option<&str>,
        script_counter: &mut u32,
    ) {
        let kind = node.kind();
        let mut new_parent: Option<String> = parent_qname.map(str::to_string);

        match kind {
            "script_element" => {
                Self::handle_script(node, source, rel_path, result, file_info, script_counter);
                // script bodies don't contain HTML elements — return
                return;
            }
            "style_element" => {
                // CSS embedded inline. v1 scope: do nothing.
                return;
            }
            "element" => {
                // An element has start_tag + children. Extract tag name
                // + attributes to decide whether to emit an
                // ElementInfo.
                let start_tag = Self::find_named_child(node, "start_tag");
                let Some(start_tag) = start_tag else {
                    // Recurse into children regardless — could be a
                    // fragment without start_tag (rare).
                    Self::walk_children(
                        node,
                        source,
                        rel_path,
                        result,
                        file_info,
                        parent_qname,
                        script_counter,
                    );
                    return;
                };
                let tag = Self::extract_tag_name(start_tag, source).unwrap_or("");
                let attrs = Self::extract_attributes(start_tag, source);

                // Surface imports from `<link rel="stylesheet" href=...>`.
                if tag.eq_ignore_ascii_case("link") {
                    let rel = attrs
                        .iter()
                        .find(|(k, _)| k == "rel")
                        .map(|(_, v)| v.as_str());
                    let href = attrs
                        .iter()
                        .find(|(k, _)| k == "href")
                        .map(|(_, v)| v.as_str());
                    if let (Some(rel), Some(href)) = (rel, href) {
                        if rel.eq_ignore_ascii_case("stylesheet") && !href.is_empty() {
                            file_info.imports.push(href.to_string());
                        }
                    }
                    return;
                }

                let line = node.start_position().row as u32 + 1;
                let end_line = node.end_position().row as u32 + 1;

                // Decide whether this element warrants an Element node.
                let id = attrs
                    .iter()
                    .find(|(k, _)| k == "id")
                    .map(|(_, v)| v.clone());

                let emitted_qname = if HEADING_TAGS.contains(&tag.to_ascii_lowercase().as_str()) {
                    // Heading: name = text content (truncated).
                    let text = Self::extract_inner_text(node, source);
                    let name = truncate(&text, 100);
                    let anchor = id.clone().unwrap_or_else(|| name.clone());
                    let qname = format!("{rel_path}:{tag}:{}:{}", line, slugify(&anchor));
                    result.elements.push(ElementInfo {
                        name,
                        qualified_name: qname.clone(),
                        tag: tag.to_string(),
                        kind: "heading".to_string(),
                        id,
                        action: None,
                        method: None,
                        file_path: rel_path.to_string(),
                        line_number: line,
                        end_line: Some(end_line),
                        parent_qname: parent_qname.map(str::to_string),
                    });
                    Some(qname)
                } else if tag.eq_ignore_ascii_case("form") {
                    let action = attrs
                        .iter()
                        .find(|(k, _)| k == "action")
                        .map(|(_, v)| v.clone());
                    let method = attrs
                        .iter()
                        .find(|(k, _)| k == "method")
                        .map(|(_, v)| v.clone());
                    if action.is_some() {
                        let name = action.clone().unwrap();
                        let qname = format!("{rel_path}:form:{}:{}", line, slugify(&name));
                        result.elements.push(ElementInfo {
                            name,
                            qualified_name: qname.clone(),
                            tag: tag.to_string(),
                            kind: "form".to_string(),
                            id: id.clone(),
                            action,
                            method,
                            file_path: rel_path.to_string(),
                            line_number: line,
                            end_line: Some(end_line),
                            parent_qname: parent_qname.map(str::to_string),
                        });
                        Some(qname)
                    } else {
                        None
                    }
                } else if let Some(id_val) = id.clone() {
                    // Generic element with an `id` attribute.
                    let qname = format!("{rel_path}:{tag}:{}:{}", line, slugify(&id_val));
                    result.elements.push(ElementInfo {
                        name: id_val.clone(),
                        qualified_name: qname.clone(),
                        tag: tag.to_string(),
                        kind: "section".to_string(),
                        id,
                        action: None,
                        method: None,
                        file_path: rel_path.to_string(),
                        line_number: line,
                        end_line: Some(end_line),
                        parent_qname: parent_qname.map(str::to_string),
                    });
                    Some(qname)
                } else {
                    None
                };

                if emitted_qname.is_some() {
                    new_parent = emitted_qname;
                }

                Self::walk_children(
                    node,
                    source,
                    rel_path,
                    result,
                    file_info,
                    new_parent.as_deref(),
                    script_counter,
                );
            }
            _ => {
                Self::walk_children(
                    node,
                    source,
                    rel_path,
                    result,
                    file_info,
                    parent_qname,
                    script_counter,
                );
            }
        }
    }

    fn walk_children(
        node: Node,
        source: &[u8],
        rel_path: &str,
        result: &mut ParseResult,
        file_info: &mut FileInfo,
        parent_qname: Option<&str>,
        script_counter: &mut u32,
    ) {
        let mut cursor = node.walk();
        for child in node.named_children(&mut cursor) {
            Self::walk_element(
                child,
                source,
                rel_path,
                result,
                file_info,
                parent_qname,
                script_counter,
            );
        }
    }

    /// `<script>...</script>` handler. For external scripts (with
    /// `src`), populate `file_info.imports`. For inline scripts,
    /// invoke the JS sub-parser on the body and merge the results
    /// (functions, classes, calls) into the HTML parse result with
    /// qnames prefixed by `{rel_path}:script_<n>`.
    fn handle_script(
        node: Node,
        source: &[u8],
        rel_path: &str,
        result: &mut ParseResult,
        file_info: &mut FileInfo,
        script_counter: &mut u32,
    ) {
        let start_tag = Self::find_named_child(node, "start_tag");
        let Some(start_tag) = start_tag else {
            return;
        };
        let attrs = Self::extract_attributes(start_tag, source);

        // External: `<script src="...">`.
        if let Some((_, src)) = attrs.iter().find(|(k, _)| k == "src") {
            if !src.is_empty() {
                file_info.imports.push(src.clone());
            }
            return;
        }

        // Inline: parse the raw_text body via the JS sub-parser. We
        // wrap the JS parser's ParseResult so emitted functions get
        // qnames scoped to this script block, preventing collisions
        // when multiple blocks define same-named helpers.
        let raw = Self::find_named_child(node, "raw_text");
        let Some(raw) = raw else { return };
        let body = node_text(raw, source).to_string();
        if body.trim().is_empty() {
            return;
        }
        *script_counter += 1;
        let scope = format!("{rel_path}:script_{}", script_counter);

        // Write body to a temp file path the JS parser can consume.
        // The TS/JS parser builds qnames as `<file_stem>.<name>`
        // (using `/` between path segments, `.` between stem and
        // name). We want the strip pattern in `strip_js_tmp_prefix`
        // to be predictable, so use a fixed stem "block" — every
        // produced qname will start with `block.` regardless of
        // counter, and the strip is a one-shot match.
        let tmp_dir = std::env::temp_dir().join(format!(
            "kglite-html-script-{}-{}",
            std::process::id(),
            script_counter
        ));
        let _ = std::fs::create_dir_all(&tmp_dir);
        let tmp_path = tmp_dir.join("block.js");
        if std::fs::write(&tmp_path, &body).is_err() {
            return;
        }
        let sub = JstsParser::javascript();
        let sub_result = sub.parse_file(&tmp_path, &tmp_dir);
        let _ = std::fs::remove_file(&tmp_path);
        let _ = std::fs::remove_dir(&tmp_dir);

        // Rescope every extracted entity into the script-block
        // namespace. The JS sub-parser produced qnames of shape
        // `<tmp_pkg>.<tmp_stem>.<…>`; strip the first two segments
        // (throwaway tmp paths) and prepend the real scope so we end
        // up with `index.html:script_N.<original_name>`. File entries
        // from the sub-parser are dropped (the host HTML File is the
        // carrier). Line numbers shift to the script block's start
        // so source-location lookups still point at the right
        // position in the host HTML file.
        let script_start_line = node.start_position().row as u32;
        let rescope = |q: &str| -> String {
            let stripped = strip_js_tmp_prefix(q);
            format!("{}.{}", scope, stripped)
        };
        for mut fn_info in sub_result.functions {
            fn_info.qualified_name = rescope(&fn_info.qualified_name);
            fn_info.file_path = rel_path.to_string();
            fn_info.line_number = fn_info.line_number.saturating_add(script_start_line);
            fn_info.end_line = fn_info
                .end_line
                .map(|e| e.saturating_add(script_start_line));
            result.functions.push(fn_info);
        }
        for mut cls in sub_result.classes {
            cls.qualified_name = rescope(&cls.qualified_name);
            cls.file_path = rel_path.to_string();
            result.classes.push(cls);
        }
        for mut con in sub_result.constants {
            con.qualified_name = rescope(&con.qualified_name);
            con.file_path = rel_path.to_string();
            result.constants.push(con);
        }
        for mut rel in sub_result.type_relationships {
            rel.source_type = rescope(&rel.source_type);
            if let Some(t) = rel.target_type {
                rel.target_type = Some(rescope(&t));
            }
            for m in &mut rel.methods {
                m.qualified_name = rescope(&m.qualified_name);
                m.file_path = rel_path.to_string();
            }
            result.type_relationships.push(rel);
        }
    }

    // ── Tree-sitter helpers ─────────────────────────────────────────

    fn find_named_child<'a>(node: Node<'a>, name: &str) -> Option<Node<'a>> {
        let mut cursor = node.walk();
        for child in node.named_children(&mut cursor) {
            if child.kind() == name {
                return Some(child);
            }
        }
        None
    }

    fn extract_tag_name<'a>(start_tag: Node<'a>, source: &'a [u8]) -> Option<&'a str> {
        let mut cursor = start_tag.walk();
        for child in start_tag.named_children(&mut cursor) {
            if child.kind() == "tag_name" {
                return Some(node_text(child, source));
            }
        }
        None
    }

    fn extract_attributes(start_tag: Node, source: &[u8]) -> Vec<(String, String)> {
        let mut out = Vec::new();
        let mut cursor = start_tag.walk();
        for child in start_tag.named_children(&mut cursor) {
            if child.kind() != "attribute" {
                continue;
            }
            let mut name: Option<String> = None;
            let mut value: Option<String> = None;
            let mut sub = child.walk();
            for c in child.named_children(&mut sub) {
                match c.kind() {
                    "attribute_name" => {
                        name = Some(node_text(c, source).to_ascii_lowercase());
                    }
                    "attribute_value" => {
                        if value.is_none() {
                            value = Some(node_text(c, source).to_string());
                        }
                    }
                    "quoted_attribute_value" => {
                        // Walk for the inner attribute_value.
                        let mut sub2 = c.walk();
                        for cc in c.named_children(&mut sub2) {
                            if cc.kind() == "attribute_value" {
                                value = Some(node_text(cc, source).to_string());
                                break;
                            }
                        }
                    }
                    _ => {}
                }
            }
            if let Some(n) = name {
                out.push((n, value.unwrap_or_default()));
            }
        }
        out
    }

    /// Concatenate every text/raw-text descendant of an element to
    /// produce the heading's display name. Cheap walk; bails after
    /// 200 chars since headings rarely need more.
    fn extract_inner_text(node: Node, source: &[u8]) -> String {
        let mut out = String::new();
        fn walk(n: Node, source: &[u8], out: &mut String) {
            if out.len() > 200 {
                return;
            }
            let kind = n.kind();
            if kind == "text" || kind == "raw_text" {
                out.push_str(node_text(n, source));
                return;
            }
            let mut cursor = n.walk();
            for child in n.named_children(&mut cursor) {
                walk(child, source, out);
                if out.len() > 200 {
                    return;
                }
            }
        }
        walk(node, source, &mut out);
        out.split_whitespace().collect::<Vec<_>>().join(" ")
    }
}

/// Strip the JS-side `block.` qname prefix produced by the sub-parser
/// on our throwaway temp file (named `block.js` by `handle_script`).
/// Returns the inner name suitable for re-scoping under the host
/// HTML file's script block.
fn strip_js_tmp_prefix(qname: &str) -> String {
    qname
        .strip_prefix("block.")
        .map(str::to_string)
        .unwrap_or_else(|| qname.to_string())
}

fn slugify(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut prev_dash = false;
    for c in s.chars().take(60) {
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

fn truncate(s: &str, n: usize) -> String {
    if s.len() <= n {
        return s.to_string();
    }
    let mut end = n;
    while !s.is_char_boundary(end) && end > 0 {
        end -= 1;
    }
    format!("{}…", &s[..end])
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

impl LanguageParser for HtmlParser {
    fn language_name(&self) -> &'static str {
        "html"
    }
    fn file_extensions(&self) -> &'static [&'static str] {
        &["html", "htm"]
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
            language: "html".to_string(),
            submodule_declarations: Vec::new(),
            imports: Vec::new(),
            exports: Vec::new(),
            annotations: None,
            is_test,
            skip_reason: None,
        };

        let mut script_counter = 0u32;
        Self::walk_element(
            tree.root_node(),
            source_bytes,
            &rel_path,
            &mut result,
            &mut file_info,
            None,
            &mut script_counter,
        );

        result.files.push(file_info);
        result
    }
}
