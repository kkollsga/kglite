//! Dart language parser.
//!
//! Backed by the `tree-sitter-dart` grammar (the `nielsenko/tree-sitter-dart`
//! packaging on crates.io). The grammar's root node is `source_file`, with
//! real `function_declaration` / `method_declaration` wrappers and a real
//! `call_expression` — so the parser walks declarations directly, no
//! signature/body sibling-pairing needed.
//!
//! Phase 0 coverage: `.dart` files are recognised and recorded as File
//! nodes (path, loc, module, is_test). Entity extraction — classes,
//! functions, mixins, extensions, enums, constants — lands in later phases.

use std::path::Path;
use tree_sitter::{Parser, Tree};

use super::LanguageParser;
use crate::code_tree::models::{FileInfo, ParseResult};

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
        let rel_path = filepath
            .strip_prefix(src_root)
            .unwrap_or(filepath)
            .to_string_lossy()
            .to_string();
        let module_path = file_to_module_path(filepath, src_root);

        // Parse to validate the grammar loads and the source is
        // tree-sitter-parseable; the tree is walked for entities in
        // later phases.
        let Some(_tree) = self.parse_tree(source.as_bytes()) else {
            return result;
        };

        let filename = filepath
            .file_name()
            .and_then(|o| o.to_str())
            .unwrap_or("")
            .to_string();
        let is_test = crate::code_tree::parsers::shared::is_test_path(
            &rel_path,
            &filename,
            &["_test.dart"],
        );

        result.files.push(FileInfo {
            path: rel_path,
            filename,
            loc: source.lines().count() as u32,
            module_path,
            language: "dart".to_string(),
            submodule_declarations: Vec::new(),
            imports: Vec::new(),
            exports: Vec::new(),
            annotations: None,
            is_test,
            skip_reason: None,
        });
        result
    }
}
