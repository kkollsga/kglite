//! Language parsers: one file per language grammar.
//!
//! `LanguageParser` is the dispatch trait — `parse_file` handles a single
//! source file, `parse_directory` walks the tree in parallel via rayon.

pub mod cpp;
pub mod csharp;
pub mod go;
pub mod java;
pub mod python;
pub mod rust_lang;
pub mod shared;
pub mod swift;
pub mod typescript;

use crate::code_tree::models::ParseResult;
use rayon::prelude::*;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use walkdir::WalkDir;

/// Stack size for the parser worker pool.
///
/// Tree-sitter is recursive-descent. On macOS, std::thread (and therefore
/// rayon) workers default to ~2 MB stacks while the main thread gets 8 MB.
/// Pathological-but-valid inputs in the wild — e.g. dotnet/runtime's
/// `JIT/Regression/JitBlue/GitHub_10215.cs`, a regression test that is
/// literally a chain of thousands of `+` operators designed to stress
/// recursive tree walkers — overflow the worker stack and SIGBUS the
/// process. 16 MB clears every real-world case we've encountered without
/// meaningfully affecting RSS (stacks are demand-paged).
const PARSER_THREAD_STACK_SIZE: usize = 16 * 1024 * 1024;

/// Shared rayon pool used by every language parser's `parse_directory`.
/// Lazily initialised on first use, then reused across languages and across
/// repeat `build()` calls in the same process.
fn parser_pool() -> &'static rayon::ThreadPool {
    static POOL: OnceLock<rayon::ThreadPool> = OnceLock::new();
    POOL.get_or_init(|| {
        rayon::ThreadPoolBuilder::new()
            .stack_size(PARSER_THREAD_STACK_SIZE)
            .thread_name(|i| format!("kglite-parser-{i}"))
            .build()
            .expect("failed to build kglite parser thread pool")
    })
}

/// File extension → language identifier (mirrors registry.py::EXTENSION_MAP).
pub const EXTENSION_MAP: &[(&str, &str)] = &[
    ("rs", "rust"),
    ("py", "python"),
    ("pyi", "python"),
    ("ts", "typescript"),
    ("tsx", "typescript"),
    ("js", "javascript"),
    ("jsx", "javascript"),
    ("mjs", "javascript"),
    ("go", "go"),
    ("java", "java"),
    ("cs", "csharp"),
    ("c", "c"),
    ("h", "c"),
    ("cpp", "cpp"),
    ("cc", "cpp"),
    ("cxx", "cpp"),
    ("hpp", "cpp"),
    ("hh", "cpp"),
    ("hxx", "cpp"),
    ("swift", "swift"),
];

/// Look up a language identifier for a file path by extension.
pub fn language_for_path(path: &Path) -> Option<&'static str> {
    let ext = path.extension()?.to_str()?;
    EXTENSION_MAP
        .iter()
        .find(|(e, _)| *e == ext)
        .map(|(_, lang)| *lang)
}

/// Abstract language parser. One implementor per grammar.
pub trait LanguageParser: Sync {
    /// Canonical language identifier (e.g. `"python"`, `"rust"`).
    fn language_name(&self) -> &'static str;

    /// File extensions (without leading dot) this parser handles.
    fn file_extensions(&self) -> &'static [&'static str];

    /// Names to exclude from call-edge resolution (common stdlib etc.).
    /// Overridden by parsers that define language-specific noise sets.
    fn noise_names(&self) -> &'static [&'static str] {
        &[]
    }

    /// Parse a single source file at `filepath` (absolute) relative to `src_root`.
    fn parse_file(&self, filepath: &Path, src_root: &Path) -> ParseResult;

    /// Parse a pre-collected slice of file paths in parallel. The hot path
    /// for `build()`: the orchestrator walks the source tree once and
    /// dispatches each per-language slice through here, avoiding the N+1
    /// redundant walks that one-`parse_directory`-per-language would do.
    fn parse_files(&self, files: &[PathBuf], src_root: &Path) -> ParseResult {
        parser_pool().install(|| {
            files
                .par_iter()
                .map(|fp| self.parse_file(fp, src_root))
                .reduce(ParseResult::new, |mut acc, r| {
                    acc.merge(r);
                    acc
                })
        })
    }

    /// Parse every matching file under `src_root`. Convenience wrapper that
    /// walks the tree and delegates to `parse_files`. Build-time orchestrator
    /// uses a single shared walk instead — this entry point is kept for
    /// callers that want to parse a single language end-to-end.
    fn parse_directory(&self, src_root: &Path, verbose: bool) -> ParseResult {
        let files: Vec<PathBuf> = WalkDir::new(src_root)
            .into_iter()
            .filter_map(Result::ok)
            .filter(|e| e.file_type().is_file())
            .filter_map(|entry| {
                let path = entry.path();
                let ext = path.extension()?.to_str()?;
                if self.file_extensions().contains(&ext) {
                    Some(path.to_path_buf())
                } else {
                    None
                }
            })
            .collect();
        if verbose {
            eprintln!("  Found {} {} files", files.len(), self.language_name());
        }
        self.parse_files(&files, src_root)
    }
}

/// Detect every language present under `src_root`. Sorted for deterministic output.
pub fn detect_languages(src_root: &Path) -> Vec<&'static str> {
    let mut langs: std::collections::BTreeSet<&'static str> = std::collections::BTreeSet::new();
    for entry in WalkDir::new(src_root).into_iter().filter_map(Result::ok) {
        if entry.file_type().is_file() {
            if let Some(lang) = language_for_path(entry.path()) {
                langs.insert(lang);
            }
        }
    }
    langs.into_iter().collect()
}

/// Get a parser instance for the given language identifier.
///
/// Returns `None` if the language is unknown. Individual parsers are
/// slotted in as modules are ported; unimplemented languages return `None`.
pub fn get_parser(language: &str) -> Option<Box<dyn LanguageParser + Send + Sync>> {
    match language {
        "python" => Some(Box::new(python::PythonParser::new())),
        "rust" => Some(Box::new(rust_lang::RustParser::new())),
        "typescript" => Some(Box::new(typescript::JstsParser::typescript())),
        "javascript" => Some(Box::new(typescript::JstsParser::javascript())),
        "go" => Some(Box::new(go::GoParser::new())),
        "java" => Some(Box::new(java::JavaParser::new())),
        "csharp" => Some(Box::new(csharp::CSharpParser::new())),
        "c" => Some(Box::new(cpp::CppParser::c())),
        "cpp" => Some(Box::new(cpp::CppParser::cpp())),
        "swift" => Some(Box::new(swift::SwiftParser::new())),
        _ => None,
    }
}

/// Auto-detect languages under `src_root` and return parser instances for each.
pub fn get_parsers_for_directory(src_root: &Path) -> Vec<Box<dyn LanguageParser + Send + Sync>> {
    detect_languages(src_root)
        .into_iter()
        .filter_map(get_parser)
        .collect()
}
